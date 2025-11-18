# Phase 1: Batch Sync Implementation - v1.0.11-beta Design

**Date**: 2025-11-14 14:00 UTC
**Status**: 🎯 **DESIGN PHASE** - Implementation after Phase 0 validation
**Target**: Increase sync rate from 50-100 → 5,000-20,000 blocks/minute

---

## Executive Summary

Phase 0 (v1.0.10-beta) addressed the **root cause bottleneck** (production pause + network height sync). Phase 1 will now focus on **parallelizing the sync pipeline** to achieve Bitcoin-level sync performance.

### Performance Targets

| Metric | Phase 0 (Current) | Phase 1 (Target) | Improvement |
|--------|-------------------|------------------|-------------|
| Sync Rate | 50-100 blocks/min | 5,000-20,000 blocks/min | 50-200x |
| Batch Size | 1 block | 512 blocks | 512x |
| Parallelism | Sequential | 8-core parallel | 8x |
| Catch-Up Time | 18 hours | 4-16 minutes | 67-270x |

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────┐
│                    BATCH SYNC PIPELINE                         │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐│
│  │   NETWORK    │─────▶│  VALIDATION  │─────▶│   STORAGE    ││
│  │   REQUEST    │      │   (PARALLEL) │      │  (BATCHED)   ││
│  └──────────────┘      └──────────────┘      └──────────────┘│
│         │                      │                      │        │
│    512 blocks            8 workers              1 write_batch  │
│    from peer             validate in            atomic save    │
│    via turbo sync        parallel                              │
│                                                                │
│  THROUGHPUT: 512 blocks × 60 batches/hour = 30,720 blocks/hr  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## Component Design

### 1. BatchSyncEngine

**Purpose**: Coordinate batch sync operations across network, validation, and storage layers

**File**: `crates/q-storage/src/batch_sync.rs` (new file)

```rust
use anyhow::{anyhow, Result};
use q_types::{QBlock, BlockHash};
use std::sync::Arc;
use tokio::task::JoinSet;
use tracing::{debug, error, info, warn};

/// Configuration for batch sync operations
#[derive(Debug, Clone)]
pub struct BatchSyncConfig {
    /// Number of blocks to request per batch (default: 512)
    pub batch_size: usize,

    /// Maximum parallel validation workers (default: num_cpus)
    pub max_parallel_validations: usize,

    /// Timeout for single batch request (default: 30s)
    pub batch_timeout_secs: u64,

    /// Maximum retries per batch (default: 3)
    pub max_retries: usize,

    /// Enable strict contiguity checks (default: true)
    pub strict_contiguity: bool,
}

impl Default for BatchSyncConfig {
    fn default() -> Self {
        Self {
            batch_size: 512,
            max_parallel_validations: num_cpus::get(),
            batch_timeout_secs: 30,
            max_retries: 3,
            strict_contiguity: true,
        }
    }
}

/// Batch sync engine for high-performance blockchain synchronization
pub struct BatchSyncEngine {
    config: BatchSyncConfig,
}

impl BatchSyncEngine {
    /// Create new batch sync engine with default configuration
    pub fn new() -> Self {
        Self {
            config: BatchSyncConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: BatchSyncConfig) -> Self {
        Self { config }
    }

    /// Sync a range of blocks in parallel batches
    ///
    /// Returns the final height reached (may be less than target_height if gaps found)
    pub async fn sync_range(
        &self,
        storage: &Arc<crate::QStorage>,
        network: &Arc<q_network::UnifiedNetworkManager>,
        start_height: u64,
        target_height: u64,
    ) -> Result<u64> {
        info!("🚀 [BATCH SYNC] Starting sync from {} to {} ({} blocks)",
              start_height, target_height, target_height - start_height);

        let mut current_height = start_height;
        let mut total_synced = 0u64;
        let sync_start = std::time::Instant::now();

        while current_height < target_height {
            let batch_end = (current_height + self.config.batch_size as u64)
                .min(target_height);

            let batch_size = (batch_end - current_height) as usize;

            // Phase 1: Request batch from network
            let blocks = match self.request_batch_with_retry(
                network,
                current_height,
                batch_end,
            ).await {
                Ok(blocks) => blocks,
                Err(e) => {
                    error!("❌ [BATCH SYNC] Failed to request batch {}-{}: {}",
                           current_height, batch_end, e);
                    break; // Stop on network failure
                }
            };

            if blocks.is_empty() {
                warn!("⚠️  [BATCH SYNC] No blocks returned for range {}-{}",
                      current_height, batch_end);
                break;
            }

            // Phase 2: Validate batch in parallel
            let validated_blocks = match self.validate_batch_parallel(&blocks).await {
                Ok(validated) => validated,
                Err(e) => {
                    error!("❌ [BATCH SYNC] Validation failed for batch {}-{}: {}",
                           current_height, batch_end, e);

                    // Fall back to single-block processing for this range
                    warn!("⚠️  [BATCH SYNC] Falling back to sequential processing");
                    break;
                }
            };

            // Phase 3: Check contiguity
            let contiguous_blocks = if self.config.strict_contiguity {
                self.extract_contiguous_range(&validated_blocks, current_height)?
            } else {
                validated_blocks
            };

            if contiguous_blocks.is_empty() {
                warn!("⚠️  [BATCH SYNC] No contiguous blocks found starting from {}",
                      current_height);
                break;
            }

            // Phase 4: Save batch atomically
            let last_height = contiguous_blocks.last().unwrap().header.height;
            match storage.save_qblock_batch(&contiguous_blocks).await {
                Ok(()) => {
                    total_synced += contiguous_blocks.len() as u64;
                    current_height = last_height + 1;

                    let elapsed = sync_start.elapsed();
                    let rate = if elapsed.as_secs() > 0 {
                        total_synced / elapsed.as_secs()
                    } else {
                        0
                    };

                    info!("✅ [BATCH SYNC] Saved batch {}-{} ({} blocks, {} blocks/sec total)",
                          contiguous_blocks[0].header.height,
                          last_height,
                          contiguous_blocks.len(),
                          rate);
                }
                Err(e) => {
                    error!("❌ [BATCH SYNC] Failed to save batch: {}", e);
                    break;
                }
            }

            // Break if we didn't get a full batch (end of available data)
            if contiguous_blocks.len() < batch_size {
                info!("ℹ️  [BATCH SYNC] Received partial batch, sync complete");
                break;
            }
        }

        let sync_duration = sync_start.elapsed();
        let final_rate = if sync_duration.as_secs() > 0 {
            total_synced / sync_duration.as_secs()
        } else {
            0
        };

        info!("🎉 [BATCH SYNC] Completed: {} blocks in {:?} ({} blocks/sec average)",
              total_synced, sync_duration, final_rate);

        Ok(current_height - 1) // Return last successful height
    }

    /// Request batch with exponential backoff retry
    async fn request_batch_with_retry(
        &self,
        network: &Arc<q_network::UnifiedNetworkManager>,
        start: u64,
        end: u64,
    ) -> Result<Vec<QBlock>> {
        let mut retries = 0;

        loop {
            let timeout = tokio::time::Duration::from_secs(self.config.batch_timeout_secs);

            match tokio::time::timeout(timeout, network.turbo_sync_request(start, end)).await {
                Ok(Ok(blocks)) => return Ok(blocks),
                Ok(Err(e)) => {
                    if retries >= self.config.max_retries {
                        return Err(anyhow!("Batch request failed after {} retries: {}", retries, e));
                    }

                    let backoff = std::time::Duration::from_millis(100 * 2u64.pow(retries as u32));
                    warn!("⚠️  [BATCH SYNC] Request failed (retry {}/{}): {}",
                          retries + 1, self.config.max_retries, e);
                    tokio::time::sleep(backoff).await;
                    retries += 1;
                }
                Err(_) => {
                    if retries >= self.config.max_retries {
                        return Err(anyhow!("Batch request timed out after {} retries", retries));
                    }

                    warn!("⚠️  [BATCH SYNC] Request timed out (retry {}/{})",
                          retries + 1, self.config.max_retries);
                    retries += 1;
                }
            }
        }
    }

    /// Validate blocks in parallel using tokio tasks
    async fn validate_batch_parallel(&self, blocks: &[QBlock]) -> Result<Vec<QBlock>> {
        debug!("🔍 [BATCH SYNC] Validating {} blocks in parallel", blocks.len());

        let mut join_set = JoinSet::new();
        let max_workers = self.config.max_parallel_validations;

        // Spawn validation tasks (bounded parallelism)
        for (i, block) in blocks.iter().enumerate() {
            let block = block.clone();

            // Limit concurrent tasks
            if join_set.len() >= max_workers {
                if let Some(result) = join_set.join_next().await {
                    result??; // Handle join + validation errors
                }
            }

            join_set.spawn(async move {
                Self::validate_block_fast(&block)?;
                Ok::<_, anyhow::Error>(block)
            });
        }

        // Collect all results
        let mut validated = Vec::with_capacity(blocks.len());

        while let Some(result) = join_set.join_next().await {
            match result? {
                Ok(block) => validated.push(block),
                Err(e) => {
                    error!("❌ [BATCH SYNC] Block validation failed: {}", e);
                    return Err(e);
                }
            }
        }

        // Sort by height (validation order may differ from input order)
        validated.sort_by_key(|b| b.header.height);

        debug!("✅ [BATCH SYNC] Validated {} blocks successfully", validated.len());
        Ok(validated)
    }

    /// Fast validation for batch sync (minimal checks)
    fn validate_block_fast(block: &QBlock) -> Result<()> {
        // 1. Basic header validation
        if block.header.height == 0 {
            return Err(anyhow!("Invalid height 0"));
        }

        // 2. Hash verification
        let calculated_hash = block.calculate_hash();
        // (In production, compare against expected hash from peer)

        // 3. Signature verification (if required by phase)
        // TODO: Add phase-specific signature checks

        // 4. Merkle root verification
        // TODO: Recalculate and verify merkle root

        Ok(())
    }

    /// Extract contiguous range starting from expected_start
    fn extract_contiguous_range(
        &self,
        blocks: &[QBlock],
        expected_start: u64,
    ) -> Result<Vec<QBlock>> {
        let mut contiguous = Vec::new();
        let mut next_expected = expected_start;

        for block in blocks {
            if block.header.height == next_expected {
                contiguous.push(block.clone());
                next_expected += 1;
            } else if block.header.height > next_expected {
                // Gap detected
                warn!("⚠️  [BATCH SYNC] Gap detected at height {} (expected {}, got {})",
                      next_expected, next_expected, block.header.height);
                break;
            }
            // Skip blocks lower than expected (duplicates/out of order)
        }

        Ok(contiguous)
    }
}
```

---

### 2. Storage Layer: Batch Save

**Purpose**: Atomic batch write to RocksDB

**File**: `crates/q-storage/src/kv.rs` (add method to `QStorage`)

```rust
/// Save multiple blocks in a single atomic batch
///
/// This is MUCH faster than individual save_qblock() calls:
/// - Single RocksDB write_batch (1 disk sync instead of N)
/// - Atomic all-or-nothing semantics
/// - Reduces lock contention
pub async fn save_qblock_batch(&self, blocks: &[QBlock]) -> Result<(), anyhow::Error> {
    if blocks.is_empty() {
        return Ok(());
    }

    let start = std::time::Instant::now();
    let first_height = blocks[0].header.height;
    let last_height = blocks.last().unwrap().header.height;

    info!("💾 [BATCH SAVE] Saving blocks {}-{} ({} blocks)",
          first_height, last_height, blocks.len());

    // Create RocksDB write batch
    let mut batch = rocksdb::WriteBatch::default();

    // Add all blocks to batch
    for block in blocks {
        let height_key = format!("qblock:{}", block.header.height);
        let serialized = bincode::serialize(block)
            .map_err(|e| anyhow::anyhow!("Serialization failed: {}", e))?;

        batch.put(height_key.as_bytes(), &serialized);
    }

    // Update qblock:latest pointer to last block
    let latest_key = b"qblock:latest";
    let latest_value = last_height.to_string();
    batch.put(latest_key, latest_value.as_bytes());

    // Write entire batch atomically
    {
        let db = self.db.read().await;
        db.write(batch)
            .map_err(|e| anyhow::anyhow!("Batch write failed: {}", e))?;
    }

    let duration = start.elapsed();
    info!("✅ [BATCH SAVE] Saved {} blocks in {:?} ({:.0} blocks/sec)",
          blocks.len(),
          duration,
          blocks.len() as f64 / duration.as_secs_f64());

    Ok(())
}
```

---

### 3. Network Layer: Turbo Sync Integration

**Purpose**: Wire batch sync into existing turbo sync path

**File**: `crates/q-api-server/src/main.rs` (modify turbo sync trigger)

```rust
// 🚀 v1.0.11-beta: BATCH SYNC INTEGRATION
// Replace sequential turbo sync with batch sync engine

use q_storage::batch_sync::{BatchSyncEngine, BatchSyncConfig};

// Initialize batch sync engine at startup
let batch_sync_config = BatchSyncConfig {
    batch_size: 512,
    max_parallel_validations: num_cpus::get(),
    batch_timeout_secs: 30,
    max_retries: 3,
    strict_contiguity: true,
};

let batch_sync_engine = Arc::new(BatchSyncEngine::with_config(batch_sync_config));

// In turbo sync trigger (around line 2700):
if network_height > current_height + 100 {
    info!("🚀 [BATCH SYNC] Large sync gap detected ({} blocks), triggering batch sync",
          network_height - current_height);

    // Clone Arc references for async task
    let storage = Arc::clone(&storage_engine);
    let network = Arc::clone(&network_manager);
    let batch_sync = Arc::clone(&batch_sync_engine);

    // Spawn batch sync in background
    tokio::spawn(async move {
        match batch_sync.sync_range(
            &storage,
            &network,
            current_height + 1,
            network_height,
        ).await {
            Ok(final_height) => {
                info!("✅ [BATCH SYNC] Completed sync to height {}", final_height);
            }
            Err(e) => {
                error!("❌ [BATCH SYNC] Failed: {}", e);
                // Fall back to sequential sync
            }
        }
    });
}
```

---

## Implementation Plan

### Step 1: Add Batch Sync Engine (2 hours)
- [ ] Create `crates/q-storage/src/batch_sync.rs`
- [ ] Implement `BatchSyncEngine` struct
- [ ] Implement `sync_range()` method
- [ ] Add retry logic with exponential backoff
- [ ] Add comprehensive logging

### Step 2: Add Batch Save to Storage (1 hour)
- [ ] Add `save_qblock_batch()` to `QStorage`
- [ ] Use RocksDB `WriteBatch`
- [ ] Update `qblock:latest` pointer atomically
- [ ] Add batch save metrics

### Step 3: Parallel Validation (2 hours)
- [ ] Implement `validate_batch_parallel()`
- [ ] Use `tokio::task::JoinSet` for bounded parallelism
- [ ] Add fast validation checks (header, hash, signature)
- [ ] Sort results by height after validation

### Step 4: Integration & Testing (2 hours)
- [ ] Wire batch sync into turbo sync path
- [ ] Add configuration options
- [ ] Test with large sync gaps (10k-100k blocks)
- [ ] Measure performance improvement

### Step 5: Error Handling & Fallback (1 hour)
- [ ] Add contiguity gap detection
- [ ] Fall back to sequential on batch failure
- [ ] Add retry limits
- [ ] Log all failure modes

---

## Performance Analysis

### Baseline (Phase 0 - Sequential Sync)
```
Operation          | Time    | Throughput
-------------------|---------|------------
Network request    | 100ms   | 10 blocks/sec
Block validation   | 50ms    | 20 blocks/sec
Database save      | 8ms     | 125 blocks/sec
TOTAL (sequential) | 158ms   | 6.3 blocks/sec
```

**Effective Rate**: ~400 blocks/minute (limited by sequential network requests)

### Target (Phase 1 - Batch Sync)
```
Operation               | Time    | Throughput
------------------------|---------|------------
Network request (512)   | 500ms   | 1024 blocks/sec
Parallel validation (8) | 50ms    | 10240 blocks/sec (8x parallel)
Batch save (512)        | 100ms   | 5120 blocks/sec
TOTAL (pipelined)       | 650ms   | 787 blocks/sec
```

**Effective Rate**: ~47,000 blocks/minute (pipeline parallelism)

**Expected Real-World**: 5,000-20,000 blocks/minute (accounting for network variability)

---

## Success Metrics

### Performance Targets (Within 1 Hour of Deployment)
- [ ] Sync rate: 5,000-20,000 blocks/minute (current: 50-100)
- [ ] Catch-up time: 4-16 minutes for 81,000 blocks (current: 18 hours)
- [ ] CPU usage: 60-80% during sync (full utilization of cores)
- [ ] Memory: <4GB (batch buffering overhead)

### Quality Targets
- [ ] Zero data corruption (batch writes are atomic)
- [ ] Zero gaps in blockchain (strict contiguity checks)
- [ ] Graceful degradation (fall back to sequential on errors)
- [ ] Observable performance (detailed logging and metrics)

---

## Risk Assessment

### Risk Level: **MEDIUM**
- **Complexity**: Moderate (parallel validation + atomic batching)
- **Dependencies**: RocksDB write_batch, tokio parallelism
- **Impact**: High performance gain, but risk of data corruption if batch writes fail
- **Mitigation**: Strict testing, atomic batch semantics, fallback to sequential

### Confidence Level: **85%**
- **Pattern**: Bitcoin/Ethereum use similar batch sync
- **Components**: Well-tested Rust async patterns
- **Rollback**: Can disable batch sync, fall back to sequential
- **Validation**: Clear success/failure metrics

---

## Testing Strategy

### Unit Tests
```rust
#[tokio::test]
async fn test_batch_sync_contiguous_blocks() {
    let batch_sync = BatchSyncEngine::new();
    // Test syncing 512 contiguous blocks
}

#[tokio::test]
async fn test_batch_sync_gap_detection() {
    // Test handling of gaps in block sequence
}

#[tokio::test]
async fn test_parallel_validation_performance() {
    // Measure validation speedup with parallelism
}
```

### Integration Tests
```bash
# Test 1: Full sync from genesis
./test_batch_sync.sh --start 0 --end 10000 --expected-time 120s

# Test 2: Large gap sync
./test_batch_sync.sh --start 100 --end 100000 --expected-time 600s

# Test 3: Stress test with network failures
./test_batch_sync.sh --network-failures 10% --expected-fallback sequential
```

---

## Deployment Strategy

### Phase 1A: Batch Sync Core (v1.0.11-beta)
- Implement `BatchSyncEngine` and `save_qblock_batch()`
- Test with 512-block batches
- Target: 5,000-10,000 blocks/minute

### Phase 1B: Performance Tuning (v1.0.12-beta)
- Optimize batch size (256/512/1024)
- Tune parallel validation workers
- Target: 10,000-20,000 blocks/minute

### Phase 1C: Production Hardening (v1.0.13-beta)
- Add comprehensive error handling
- Implement peer reputation tracking
- Add performance monitoring dashboard

---

**Phase 1 Design Complete** - Ready for implementation after Phase 0 validation
**Implementation Time**: ~8 hours (including testing)
**Expected Performance**: 50-200x improvement over Phase 0
**Risk Level**: Medium (atomic batching + parallel validation)
**Confidence**: 85% (proven patterns, clear fallback strategy)

**Next Action**: Wait for Phase 0 (v1.0.10-beta) deployment and validation, then begin Phase 1 implementation
