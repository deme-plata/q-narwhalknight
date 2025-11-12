# Node Stuck Diagnosis - v0.9.14-beta

**Date**: 2025-11-06
**Issue**: Node repeatedly gets stuck and stops producing blocks
**Current Status**: ⚠️ **RECURRING PRODUCTION ISSUE**

---

## 🔍 Symptoms

- **User Report**: "Node gets stuck all the time" - requires frequent restarts
- **Initial Report**: Height stuck at 5908, no mining rewards
- **Actual Behavior**: Node produces blocks for ~1 hour, then block producer stops

---

## 📊 Evidence from Logs

### **Timeline of Recent Stuck Event:**

**Old Process (PID 118678):**
- Started: Nov 06 00:41:03 CET
- Last blocks produced: Nov 06 00:47:48 CET (height 7044)
- **Duration until stuck**: ~6 minutes 45 seconds
- **Symptoms**: Block producer stopped, mining submissions still queuing

**Current Process (PID 135029):**
- Started: Nov 06 02:28:45 CET
- Currently producing blocks at height 7200-7209 (as of 02:38:36)
- **Status**: Working normally (for now)

### **Key Log Observations:**

**Before Getting Stuck (00:47:48):**
```
INFO q_api_server::block_producer: ✅ BLOCK PRODUCED: Height 7044, Hash ffb1c302b2db6d64
INFO q_api_server::block_producer: 🎉 Producer #0-7 created block at height 7044
```

**After Getting Stuck (00:48:00+):**
- NO block production logs
- Mining submissions continue to queue: `⚡ Mining submission queued (non-blocking)`
- Peer height announcements continue
- Network appears functional
- **CRITICAL**: Block producer silently stops without error messages

### **Pattern Identified:**

1. ✅ Block producer starts normally with 8 parallel producers
2. ✅ Blocks produced successfully for 5-10 minutes
3. ❌ Block producer suddenly stops (NO error, NO warning, NO crash)
4. ⚠️ Mining submissions continue to queue but never processed
5. ⚠️ Node appears "stuck" - height doesn't increase
6. 🔄 Restart required to resume block production

---

## 🐛 Root Cause Analysis

### **Primary Suspect: Block Producer Task Stall**

**Hypothesis**: The block producer async task is **silently hanging** without crashing or logging errors.

**Possible Causes:**

#### **1. Deadlock or Lock Contention**
```rust
// crates/q-api-server/src/block_producer.rs uses RwLock for SharedBlockProducer
pending_solutions: Arc<SegQueue<MiningSolution>>  // Lock-free queue (good)
```
- Block producers use `Arc<RwLock<BlockProducer>>` wrapper
- If multiple producers hold read locks and one tries to write, potential deadlock
- Lock-free queue is good, but RwLock wrapper may cause contention

#### **2. Async Task Panic/Silent Failure**
```rust
// Spawned tasks may panic without propagating to main thread
tokio::spawn(async move {
    // Block producer loop here
    // If panics, task dies silently
});
```
- Tokio tasks that panic are silently dropped
- No error logging if task dies
- Main process continues running but block production stops

#### **3. Database Write Stall**
```
INFO q_storage: 💾 Saving QBlock at height 7044 with hash ffb1c302b2db6d64
```
- RocksDB writes may block if:
  * Compaction running
  * Disk I/O bottleneck
  * Write buffer full
- If storage write hangs, all producers may stall waiting for database

#### **4. Network Synchronization Conflict**
```
INFO q_api_server::block_producer: 🔄 [PRODUCER SYNC] Synchronizing all 8 producers
```
- Producer sync happens frequently
- If sync operation hangs (waiting for network peer data), all producers block
- Turbo sync may interfere with block production

#### **5. Time-Based Block Production Logic**
```rust
block_interval_secs: 15  // 15 second target block time
```
- If system time checks fail or clock skew occurs
- Producers may wait indefinitely for "next block time"
- Timer-based wakeup may fail

---

## 🔬 Diagnostic Commands

### **Check if Block Producer Tasks are Running:**
```bash
# Check for panicked tasks in logs
journalctl -u q-api-server --since "1 hour ago" | grep -iE "(panic|unwrap|expect)"

# Check for deadlock warnings
journalctl -u q-api-server --since "1 hour ago" | grep -iE "(deadlock|timeout|hung)"

# Check RocksDB performance
journalctl -u q-api-server --since "1 hour ago" | grep -iE "(compaction|flush|stall)"
```

### **Monitor Block Production in Real-Time:**
```bash
# Watch for block production
journalctl -u q-api-server -f | grep "BLOCK PRODUCED"

# Watch for producer sync
journalctl -u q-api-server -f | grep "PRODUCER SYNC"

# Watch for mining queue depth
journalctl -u q-api-server -f | grep "Mining submission queued"
```

### **Check Database Health:**
```bash
# Check RocksDB directory size
du -sh /opt/orobit/shared/q-narwhalknight/data-bep44-test/

# Check for RocksDB corruption
ls -lh /opt/orobit/shared/q-narwhalknight/data-bep44-test/hot/
ls -lh /opt/orobit/shared/q-narwhalknight/data-bep44-test/cold/
```

---

## 🛠️ Immediate Fixes Required

### **Fix #1: Add Block Producer Watchdog**

**Purpose**: Detect when block producer stops and automatically restart it

**Implementation**:
```rust
// Add to block_producer.rs
pub struct BlockProducerWatchdog {
    last_block_time: Arc<RwLock<Instant>>,
    watchdog_interval_secs: u64,
}

impl BlockProducerWatchdog {
    pub async fn monitor(&self, producers: Arc<RwLock<Vec<BlockProducer>>>) {
        loop {
            tokio::time::sleep(Duration::from_secs(self.watchdog_interval_secs)).await;

            let last_block = *self.last_block_time.read().await;
            let elapsed = last_block.elapsed();

            if elapsed > Duration::from_secs(60) {
                error!("🚨 WATCHDOG: Block producer stalled for {} seconds! Restarting...", elapsed.as_secs());
                // Restart block producers
                self.restart_producers(producers.clone()).await;
            }
        }
    }
}
```

### **Fix #2: Add Panic Handler for Block Producer Tasks**

**Purpose**: Catch and log panics in spawned tasks

**Implementation**:
```rust
// Wrap block producer spawn with panic handler
let producer_handle = tokio::spawn(async move {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // Block producer loop here
    }));

    if let Err(e) = result {
        error!("🚨 PANIC: Block producer task panicked: {:?}", e);
        // Restart producer
    }
});
```

### **Fix #3: Add Database Write Timeout**

**Purpose**: Prevent indefinite blocking on RocksDB writes

**Implementation**:
```rust
// Add timeout to block storage writes
use tokio::time::timeout;

let write_result = timeout(
    Duration::from_secs(5),
    storage.save_block(block)
).await;

match write_result {
    Ok(Ok(())) => info!("✅ Block saved successfully"),
    Ok(Err(e)) => error!("❌ Block save failed: {}", e),
    Err(_) => error!("⏱️ TIMEOUT: Block save took >5 seconds, aborting"),
}
```

### **Fix #4: Add Metrics Endpoint for Block Producer Health**

**Purpose**: Allow monitoring block producer status via API

**Implementation**:
```rust
// Add to handlers.rs
pub async fn block_producer_health(
    State(state): State<Arc<AppState>>
) -> Result<Json<ApiResponse<BlockProducerHealth>>, StatusCode> {
    let producers = state.block_producers.read().await;

    let health = BlockProducerHealth {
        active_producers: producers.len(),
        last_block_time: state.last_block_time.read().await.elapsed().as_secs(),
        pending_solutions: state.pending_solutions.len(),
        is_healthy: state.last_block_time.read().await.elapsed() < Duration::from_secs(30),
    };

    Ok(Json(ApiResponse::success(health)))
}
```

---

## 📋 Recommended Actions

### **Immediate (Within 1 hour):**
1. ✅ Add comprehensive logging before/after critical sections
2. ✅ Add watchdog timer to detect stalled producers
3. ✅ Add panic handlers to all spawned tasks
4. ✅ Add database write timeouts

### **Short-term (Within 24 hours):**
5. 🔄 Review RwLock usage and convert to lock-free patterns where possible
6. 🔄 Add Prometheus metrics for block producer health
7. 🔄 Implement automatic producer restart on stall
8. 🔄 Add stress testing to reproduce stall condition

### **Long-term (Within 1 week):**
9. 🔄 Refactor block producer to use actor model (no shared state)
10. 🔄 Add circuit breaker pattern for database writes
11. 🔄 Implement producer task supervision tree
12. 🔄 Add distributed tracing for debugging async task flow

---

## 🎯 Success Criteria

**FIXED** when:
- ✅ Node runs continuously for 24+ hours without getting stuck
- ✅ Block producer stall is detected within 30 seconds
- ✅ Automatic recovery happens without manual restart
- ✅ Monitoring API shows producer health status
- ✅ All panics/errors are logged with full context

---

## 📊 Related Issues

- **Distributed AI Deserialization Error**: `Found an Option discriminant that wasn't 0 or 1` on topic `qnk/ai/node-capability/v1`
- **Performance Benchmark API**: User requested easy-to-use endpoint to visualize N× speedup

---

**Status**: ⚠️ **CRITICAL - REQUIRES IMMEDIATE FIX**
**Priority**: **P0 - Production Blocker**
**Next Step**: Implement watchdog + panic handler + enhanced logging
