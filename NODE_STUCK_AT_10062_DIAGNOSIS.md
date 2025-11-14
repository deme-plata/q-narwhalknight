# Node Stuck at Height 10062 - Root Cause Analysis

**Date**: 2025-11-12
**Status**: 🚨 **CRITICAL - BLOCK PRODUCER STALLED**
**Current Height**: 10063 (stuck)
**Last Block Produced**: ~04:26:48 (16 minutes ago)

---

## 🔍 Diagnosis Summary

The node is **NOT stuck at sync** - it's stuck because the **block producer has stalled**.

### Timeline

| Time | Event |
|------|-------|
| 04:26:48 | ✅ Last healthy watchdog: height 10023 → 10062 |
| 04:27:48 | 🚨 First STALLED warning |
| 04:28:48+ | 🚨 Continuous STALLED warnings every minute |
| Now | 🚨 Still stalled (16+ minutes) |

### Key Observations

1. **Mining is Active**
   - Mining submissions are still being queued
   - Example: `qnk65085b6858d87: Nonce 49657693163` at 04:43:08
   - Miners are working and sending valid nonces

2. **Height is Correct**
   - Binary search confirms height 10063 with no gaps
   - Database integrity is good
   - No missing blocks

3. **Block Producer is STALLED**
   - No "Produced block" messages since 04:26:48
   - Heartbeat stopped reporting
   - Watchdog firing STALLED errors every 60 seconds

4. **No Errors in Logs**
   - No panic, deadlock, or timeout messages
   - No errors around the stall time (04:26-04:28)
   - Clean shutdown of block producer loop (silent stall)

---

## 🐛 Root Cause

**Hypothesis**: The block producer loop is blocked waiting on an async operation that never completes.

### Possible Causes (from previous stall bugs)

1. **Database Write Deadlock**
   - Block producer waiting for RocksDB write lock
   - Competing with balance consensus or height reads
   - Location: `crates/q-api-server/src/block_producer.rs`

2. **Channel Deadlock**
   - Mining submission channel full (bounded)
   - Block producer waiting to send to full channel
   - Location: `crates/q-api-server/src/main.rs` (mining queue)

3. **Async Lock Contention**
   - Producer waiting on `Arc<RwLock<Consensus>>` or similar
   - Multiple tasks trying to write simultaneously
   - Location: Various consensus operations

4. **Transaction Timeout**
   - Database transaction hanging indefinitely
   - No timeout configured for StorageTransaction
   - Location: `crates/q-storage/src/kv.rs`

---

## 🔧 Immediate Fix Options

### Option 1: Restart the Service (Quick Fix)
```bash
systemctl restart q-api-server
```
**Result**: Restarts block production, but stall will recur.

### Option 2: Debug with Tokio Console (Root Cause)
```bash
# Requires recompilation with tokio-console feature
cargo build --release --features tokio-console
```
**Result**: Can see which task is blocked and why.

### Option 3: Add Timeouts to Block Producer Loop
```rust
// In block_producer.rs
tokio::select! {
    result = storage.begin_transaction() => { /* normal flow */ },
    _ = tokio::time::sleep(Duration::from_secs(5)) => {
        error!("🚨 Transaction timeout - skipping block");
        continue;
    }
}
```
**Result**: Prevents permanent stalls.

---

## 📊 Stall Pattern Analysis

### Previous Stall Bugs

1. **v0.9.50-beta** - Block producer stall at height 3500
   - Cause: RocksDB write lock deadlock
   - Fix: Moved DB writes to spawn_blocking

2. **v0.9.75-beta** - BlockPackCodec deadlock
   - Cause: Mutex contention in codec
   - Fix: Lock-free implementation

3. **v0.9.92-beta** - Producer height desync
   - Cause: Pointer race condition
   - Fix: Atomic height pointer updates

### Current Stall (v1.0.1-beta)

**Similarities**:
- Silent stall (no panic/error)
- Watchdog fires correctly
- Mining submissions continue
- Database reads work (height queries succeed)

**Differences**:
- Occurs at specific height (10062/10063)
- No recent code changes to producer
- Happens after ~1 hour of operation

---

## 🔬 Recommended Investigation Steps

### Step 1: Check Task Status (If tokio-console available)
```bash
# See which tasks are blocked
tokio-console --help
```

### Step 2: Check RocksDB Metrics
```bash
# See if RocksDB has pending writes
curl http://localhost:8080/api/v1/debug/rocksdb-stats 2>/dev/null | jq '.'
```

### Step 3: Review Recent Block Production
```bash
# Check last 100 blocks for patterns
journalctl -u q-api-server --since "1 hour ago" | \
  grep "Produced block" | tail -100 | \
  awk '{print $NF}' | sort -n | uniq -c
```

### Step 4: Monitor Channel Capacity
```bash
# Add logging to mining submission handler
# Log: "Mining queue: {}/{} capacity" every 100 submissions
```

---

## 🚀 Long-term Solution

### Proposal: Implement Stall Recovery

```rust
// In main.rs block_producer spawn
tokio::spawn(async move {
    let mut last_height = 0;
    let mut stall_count = 0;

    loop {
        // Check if stalled
        let current_height = storage.get_latest_qblock_height().await?;
        if current_height == last_height {
            stall_count += 1;
            if stall_count > 3 {
                error!("🚨 STALL DETECTED - Restarting producer");
                // Break and respawn
                break;
            }
        } else {
            stall_count = 0;
            last_height = current_height;
        }

        // Producer loop with timeout
        tokio::select! {
            _ = producer_loop() => {},
            _ = tokio::time::sleep(Duration::from_secs(30)) => {
                warn!("⏰ Producer loop iteration timeout");
            }
        }
    }
});
```

### Proposal: Add Circuit Breaker

```rust
// Prevent cascading failures
pub struct CircuitBreaker {
    failures: AtomicUsize,
    threshold: usize,
    state: AtomicBool, // true = open (blocking)
}

impl CircuitBreaker {
    pub async fn call<F, T>(&self, f: F) -> Result<T>
    where F: Future<Output = Result<T>>
    {
        if self.state.load(Ordering::Relaxed) {
            return Err(anyhow!("Circuit breaker open"));
        }

        match timeout(Duration::from_secs(5), f).await {
            Ok(Ok(result)) => {
                self.failures.store(0, Ordering::Relaxed);
                Ok(result)
            }
            _ => {
                let count = self.failures.fetch_add(1, Ordering::Relaxed);
                if count >= self.threshold {
                    self.state.store(true, Ordering::Relaxed);
                }
                Err(anyhow!("Operation failed"))
            }
        }
    }
}
```

---

## 📈 Monitoring Recommendations

1. **Add Prometheus Metrics**:
   - `block_producer_loop_iterations_total`
   - `block_producer_stall_count`
   - `mining_queue_depth`
   - `rocksdb_write_latency_ms`

2. **Add Health Check Endpoint**:
   ```rust
   GET /api/v1/health/producer
   Response: {
       "healthy": false,
       "last_block_time": "2025-11-12T03:26:48Z",
       "seconds_since_last_block": 960,
       "current_height": 10063
   }
   ```

3. **Add Alerting**:
   - Alert if no blocks produced for >5 minutes
   - Alert if mining queue >80% full
   - Alert if RocksDB write latency >1s

---

## ⚠️ Impact Assessment

**Current Impact**:
- ❌ No new blocks being produced
- ❌ Mining rewards not being distributed
- ❌ Network halted (single node testnet)
- ✅ Database integrity maintained
- ✅ P2P network still responding
- ✅ API queries still working (partially)

**Recovery Time**:
- Quick restart: 30 seconds
- Root cause fix: 2-4 hours (investigation + patch)

---

**Next Action**: Restart service to restore block production, then investigate root cause with proper instrumentation.
