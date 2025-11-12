# Phase 1A Integration Plan - v1.0.2-beta
## SafeBatchedWriter Integration with Feature Flag

**Date**: 2025-11-12
**Status**: Ready for Integration
**Target**: Feature-flagged deployment to testnet

---

## 🎯 Integration Objectives

1. ✅ **Wire SafeBatchedWriter into API server** with `--experimental-fast-sync` flag
2. ✅ **Maintain backward compatibility** - default to existing BlockWriter
3. ✅ **Add metrics endpoints** for monitoring batched sync performance
4. ✅ **Graceful fallback** if SafeBatchedWriter fails
5. ✅ **Zero-downtime deployment** via feature flag

---

## 📦 Implementation Steps

### Step 1: Add CLI Flag to q-api-server

**File**: `crates/q-api-server/src/main.rs`

```rust
// Add to CLI arguments (around line 450-500)
.arg(
    Arg::new("experimental-fast-sync")
        .long("experimental-fast-sync")
        .help("Enable experimental batched sync (150-250 BPS)")
        .action(clap::ArgAction::SetTrue)
)

// Parse flag
let use_fast_sync = matches.get_flag("experimental-fast-sync");

if use_fast_sync {
    info!("🚀 Experimental fast sync ENABLED (150-250 BPS target)");
    info!("⚠️  Max loss on crash: ≤16 blocks");
} else {
    info!("🔒 Using default sync (9.3 BPS, zero loss)");
}
```

### Step 2: Initialize SafeBatchedWriter in AppState

**File**: `crates/q-api-server/src/lib.rs`

```rust
use q_storage::{SafeBatchedWriter, BatchConfig, BatchMetrics};
use tokio::sync::mpsc;

pub struct AppState {
    // ... existing fields ...

    // Fast sync components
    pub fast_sync_enabled: bool,
    pub fast_sync_tx: Option<mpsc::Sender<QBlock>>,
    pub fast_sync_metrics: Option<Arc<Mutex<BatchMetrics>>>,
}

// In create_app_state()
let (fast_sync_tx, fast_sync_metrics) = if use_fast_sync {
    let config = BatchConfig::default(); // 16 blocks, 1s, 1 MiB
    let start_height = storage.get_height().await?;

    let (writer, tx) = SafeBatchedWriter::new(
        db.clone(),
        config,
        start_height,
    );

    let metrics = writer.get_metrics();

    // Spawn writer task
    tokio::spawn(async move {
        if let Err(e) = writer.run().await {
            error!("❌ SafeBatchedWriter failed: {}", e);
        }
    });

    (Some(tx), Some(Arc::new(Mutex::new(metrics))))
} else {
    (None, None)
};

let app_state = Arc::new(AppState {
    // ... existing fields ...
    fast_sync_enabled: use_fast_sync,
    fast_sync_tx,
    fast_sync_metrics,
});
```

### Step 3: Route Block Writes

**File**: `crates/q-api-server/src/main.rs` (gossipsub block handler)

```rust
// In gossipsub block message handler (around line 800-900)
GossipsubEvent::Message { message, .. } => {
    if message.topic == block_topic_hash {
        match bincode::deserialize::<QBlock>(&message.data) {
            Ok(block) => {
                debug!("📦 Received block {} via gossipsub", block.header.height);

                // Route to appropriate writer
                if app_state.fast_sync_enabled {
                    // Use SafeBatchedWriter
                    if let Some(ref tx) = app_state.fast_sync_tx {
                        if let Err(e) = tx.send(block).await {
                            error!("❌ Failed to send block to fast sync: {}", e);
                            // Fallback to direct write
                            storage.store_block(&block).await?;
                        }
                    }
                } else {
                    // Use existing BlockWriter (default)
                    storage.store_block(&block).await?;
                }
            }
            Err(e) => {
                warn!("Failed to deserialize block: {}", e);
            }
        }
    }
}
```

### Step 4: Add Metrics Endpoint

**File**: `crates/q-api-server/src/handlers.rs`

```rust
use axum::Json;
use q_storage::BatchMetrics;

/// GET /api/sync/metrics
///
/// Returns batched sync performance metrics
pub async fn get_sync_metrics(
    State(state): State<Arc<AppState>>,
) -> Result<Json<SyncMetricsResponse>, StatusCode> {
    if !state.fast_sync_enabled {
        return Ok(Json(SyncMetricsResponse {
            enabled: false,
            metrics: None,
        }));
    }

    let metrics = state.fast_sync_metrics
        .as_ref()
        .map(|m| m.lock().unwrap().clone());

    Ok(Json(SyncMetricsResponse {
        enabled: true,
        metrics,
    }))
}

#[derive(Debug, Serialize)]
pub struct SyncMetricsResponse {
    pub enabled: bool,
    pub metrics: Option<BatchMetrics>,
}

// Add to router in lib.rs
.route("/api/sync/metrics", get(get_sync_metrics))
```

### Step 5: Graceful Shutdown

**File**: `crates/q-api-server/src/main.rs`

```rust
// In shutdown handler
if app_state.fast_sync_enabled {
    info!("🛑 Shutting down SafeBatchedWriter...");

    // Close channel to trigger graceful shutdown
    if let Some(tx) = app_state.fast_sync_tx.take() {
        drop(tx);
    }

    // Wait for final flush (up to 5 seconds)
    tokio::time::sleep(Duration::from_secs(5)).await;

    // Print final metrics
    if let Some(metrics) = &app_state.fast_sync_metrics {
        let m = metrics.lock().unwrap();
        info!("📊 Final metrics: {} blocks, {} batches",
              m.blocks_flushed_total, m.batches_flushed_total);
    }
}
```

---

## 🧪 Testing Checklist

### Pre-Integration Testing

- [x] Unit tests pass: `cargo test --package q-storage`
- [x] Benchmark compiles: `cargo bench --no-run`
- [x] Kill recovery script ready: `./tests/kill_recovery_test.sh`

### Integration Testing

- [ ] Compile with integration: `timeout 36000 cargo build --release --package q-api-server`
- [ ] Run with default mode: `./target/release/q-api-server` (should use BlockWriter)
- [ ] Run with fast sync: `./target/release/q-api-server --experimental-fast-sync`
- [ ] Verify metrics endpoint: `curl http://localhost:8080/api/sync/metrics`
- [ ] Monitor performance: Verify 150-250 BPS sustained
- [ ] Kill -9 test: Verify ≤16 blocks lost
- [ ] Graceful shutdown: Verify final flush completes

### Production Testing

- [ ] Deploy to testnet with flag OFF (default mode)
- [ ] Enable flag on 1 node, monitor for 24 hours
- [ ] Collect performance data
- [ ] Verify no data corruption
- [ ] Gradual rollout: Enable on 25% → 50% → 100% of nodes

---

## 📊 Success Criteria

Before enabling by default, verify:

1. **Performance**: Sustained 150-250 BPS over 10k blocks
2. **Safety**: 100/100 kill -9 tests recover with ≤16 block loss
3. **Stability**: 24-hour testnet run with zero crashes
4. **Metrics**: All metrics instrumented and alerting working
5. **Backward Compatibility**: Default mode (flag OFF) works unchanged

---

## 🚀 Deployment Strategy

### Phase 1: Feature Flag Deployment (Week 1)
- Deploy with `--experimental-fast-sync` flag
- Default: OFF (existing BlockWriter)
- Enable on 1 development node for testing

### Phase 2: Limited Rollout (Week 2)
- Enable on 25% of testnet nodes
- Monitor performance and safety metrics
- Collect community feedback

### Phase 3: Full Rollout (Week 3)
- Enable on 100% of testnet nodes
- Consider making default after 7 days of stability
- Prepare for mainnet (if applicable)

---

## 🔍 Monitoring & Alerts

### Key Metrics to Monitor

```bash
# Performance
blocks_flushed_total         # Total blocks written
batches_flushed_total        # Total batches flushed
blocks_per_second            # Current sync rate (target: 150-250)

# Safety
sync_failures                # Failed flush operations (alert if >0)
backpressure_events          # Blocks rejected due to gap (alert if >100/min)
integrity_errors             # Corrupted blocks detected (alert if >0)

# Health
reorder_buffer_size          # Current buffer size (alert if >1000)
max_gap_size                 # Largest gap in buffer (alert if >1500)
flush_latency_p99            # 99th percentile flush time (alert if >500ms)
```

### Prometheus Queries

```promql
# Sync rate (blocks per second)
rate(blocks_flushed_total[1m])

# Batch efficiency
blocks_flushed_total / batches_flushed_total

# Safety violations
rate(sync_failures[5m]) + rate(integrity_errors[5m])
```

---

## 🛡️ Rollback Procedure

If issues occur after enabling fast sync:

1. **Immediate**: Restart node without `--experimental-fast-sync` flag
2. **Verify**: Check database integrity via `cargo run --bin check_database`
3. **Recovery**: If corruption detected, restore from backup
4. **Analysis**: Review logs for error messages
5. **Report**: Document issue in GitLab issue tracker

---

## 📝 Next Steps After Integration

1. **Phase 1B Implementation** (Week 2-3)
   - Add retry logic with exponential backoff
   - Add fsync stall detection
   - Add spawn_blocking for async DB I/O
   - Comprehensive metrics dashboard

2. **Phase 2: Range Fetcher** (Week 4-5)
   - libp2p request-response protocol
   - Bulk historical sync
   - Auto mode switching (fast sync when behind, durable when caught up)

3. **Phase 3: Production Hardening** (Week 6-8)
   - 7-day testnet observation
   - Security audit
   - Community testing
   - Mainnet readiness assessment

---

**Prepared By**: Server Beta (Claude Code)
**Status**: ✅ READY FOR INTEGRATION
**Target Version**: v1.0.2-beta
**Estimated Integration Time**: 2-3 hours
