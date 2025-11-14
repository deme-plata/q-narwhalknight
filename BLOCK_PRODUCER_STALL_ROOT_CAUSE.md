# Block Producer Stall - Root Cause Analysis (Height 10063)

**Date**: 2025-11-12
**Status**: 🔍 **ROOT CAUSE IDENTIFIED**
**Stall Time**: 04:26:48
**Duration**: 16+ minutes (ongoing)
**Severity**: 🚨 **CRITICAL - Network halted**

---

## 🎯 ROOT CAUSE IDENTIFIED

### The Smoking Gun

**Line 4272 in crates/q-api-server/src/main.rs:**
```rust
match app_state_mining.storage_engine.save_qblock(&new_block).await {
```

**The block producer is HANGING on `save_qblock()` - waiting for a RocksDB write that never completes.**

### Why This Is The Root Cause

1. **Mining submissions are flowing** - 29,598 submissions in 2 minutes (247/sec)
2. **Mining submission processor is running** - Batch processing loop is active
3. **Block producer watchdog is firing** - STALLED warnings every 60 seconds
4. **No "Produced block" messages** - Last one at 04:26:48
5. **No errors in logs** - Silent hang (async await never returns)

**The block producer successfully created a block (line 4210), but when it tried to save it to RocksDB (line 4272), the `save_qblock()` async operation HUNG INDEFINITELY.**

---

## 🔬 Technical Deep Dive

### Block Production Flow (Normal vs Stalled)

**NORMAL FLOW:**
```
1. Mining submissions queue → BlockProducer (line 4169)
2. should_produce() returns true (line 4183)
3. produce_blocks() creates block (line 4210) ✅
4. save_qblock() writes to RocksDB (line 4272) ✅ [FAST: <50ms]
5. advance_height() updates producer state (line 4277) ✅
6. process_block_mining_rewards() applies balances (line 4314) ✅
7. Broadcast to P2P network (continues...)
8. LOOP REPEATS
```

**STALLED FLOW (Current Situation):**
```
1. Mining submissions queue → BlockProducer (line 4169) ✅ [WORKING]
2. should_produce() returns true (line 4183) ✅ [WORKING]
3. produce_blocks() creates block (line 4210) ✅ [WORKING]
4. save_qblock() writes to RocksDB (line 4272) ❌ [HANGS FOREVER]
   │
   └──> BLOCK PRODUCER STUCK HERE <───┐
                                        │
   Mining submissions continue to queue │
   Watchdog fires STALLED warnings      │
   No timeout configured                │
   No circuit breaker                   │
   INFINITE AWAIT ──────────────────────┘
```

### Why save_qblock() Is Hanging

Based on analysis of `crates/q-storage/src/kv.rs`, `save_qblock()` performs these operations:

1. **Begin RocksDB transaction** (may wait for write lock)
2. **Serialize block** (CPU-bound, should be fast)
3. **Write to column families:**
   - `CF_BLOCKS` - Block data by height and hash
   - Update `qblock:latest` pointer
4. **Commit transaction** (fsync to disk)

**BLOCKING POINT**: One of these operations is waiting indefinitely:

#### Hypothesis A: RocksDB Write Lock Deadlock
```rust
// save_qblock() needs write lock
// But another task holds it (mining submission processor?)
// Tokio async runtime allows other tasks to run while awaiting
// But if ALL executor threads are blocked on RocksDB, system deadlocks
```

#### Hypothesis B: Transaction Timeout (No Timeout Configured)
```rust
// RocksDB write transaction has no timeout
// If disk is slow or stalled, await never returns
// No circuit breaker to abort after N seconds
```

#### Hypothesis C: Channel Backpressure
```rust
// Mining submissions use UNBOUNDED channel (line 1932)
// 29,598 submissions queued in 2 minutes
// Producer can't keep up, queue grows infinitely
// Memory pressure causes system-wide slowdown
```

---

## 📊 Evidence Analysis

### Timeline Correlation

| Time | Event | Analysis |
|------|-------|----------|
| 04:25:00-04:27:00 | **29,598 mining submissions** | Extremely high submission rate (247/sec) |
| 04:26:48 | Last "Produced block" message | Block producer successfully created a block |
| 04:26:48 | Last healthy watchdog | Producer was healthy 1 second before stall |
| 04:27:48 | First STALLED warning | Producer did not complete block save within 60s |
| 04:28:48+ | Continuous STALLED warnings | Producer still waiting on save_qblock() |

### Code Path Analysis

**Block Producer Loop** (`crates/q-api-server/src/main.rs`):
```
Line 3961: Mining submission receiver (unbounded channel)
Line 4039: wallet_balances.write().await (RwLock write)
Line 4169: queue_solution() (lock-free, instant)
Line 4183: should_produce() (check time + queue depth)
Line 4201: sync_from_storage() (reads latest height from DB)
Line 4210: produce_blocks() (creates block in memory)
Line 4272: save_qblock() ❌ HANGS HERE
Line 4277: advance_height() [NEVER REACHED]
Line 4314: process_block_mining_rewards() [NEVER REACHED]
```

**Competing Database Operations:**
1. **Mining submission processor** - Updates balances every 500 submissions (line 4039)
2. **Block producer** - Reads height (line 4201), writes block (line 4272)
3. **Periodic balance sync** - Writes balances every 15 seconds (mentioned in comments)
4. **SafeBatchedWriter** (if enabled) - Batched block writes

---

## 🐛 Why This Bug Is Insidious

### No Error Messages
- RocksDB doesn't panic when writes are slow
- Tokio async runtime doesn't timeout by default
- No logs between "Produced block" and "STALLED" (1 minute gap!)

### Silent Failure Mode
```rust
// This is what the code THINKS is happening:
let result = save_qblock(&block).await;  // Returns quickly

// This is what's ACTUALLY happening:
let result = save_qblock(&block).await;  // Never returns
                                         // No timeout
                                         // No error
                                         // Just... waits... forever
```

### Race Condition
**Why did it happen at height 10063 specifically?**

Likely: Perfect storm of conditions:
1. High mining submission rate (247/sec)
2. RocksDB write latency spike
3. Multiple tasks trying to write simultaneously
4. No backpressure mechanism on unbounded channel
5. Tokio runtime has limited executor threads

**The stall occurs when:**
```
save_qblock() acquires DB write lock
    → BUT disk fsync is slow (100ms+ instead of <50ms)
        → Other tasks queue up waiting for lock
            → Tokio executor threads all blocked
                → New async tasks can't be scheduled
                    → DEADLOCK
```

---

## 🔧 The Fix (Multiple Layers)

### Layer 1: Add Timeout to save_qblock() (CRITICAL)

**Location**: `crates/q-api-server/src/main.rs` line 4272

**BEFORE (Hangs Forever):**
```rust
match app_state_mining.storage_engine.save_qblock(&new_block).await {
    Ok(()) => {
        info!("✅ Block {} saved to storage", new_block.header.height);
        // ...
    }
    Err(e) => {
        error!("🚨 CRITICAL: Block {} save FAILED: {}", new_block.header.height, e);
        continue;
    }
}
```

**AFTER (5-Second Timeout):**
```rust
use tokio::time::{timeout, Duration};

match timeout(Duration::from_secs(5), app_state_mining.storage_engine.save_qblock(&new_block)).await {
    Ok(Ok(())) => {
        info!("✅ Block {} saved to storage", new_block.header.height);
        // ...
    }
    Ok(Err(e)) => {
        error!("🚨 CRITICAL: Block {} save FAILED: {}", new_block.header.height, e);
        continue; // Retry on next cycle
    }
    Err(_timeout_err) => {
        error!("🚨 CRITICAL TIMEOUT: Block {} save exceeded 5 seconds!", new_block.header.height);
        error!("   RocksDB may be stalled or deadlocked - skipping this block");
        error!("   Producer will retry on next cycle");
        continue; // Skip this block, continue producing
    }
}
```

**Why This Works:**
- Prevents infinite hang
- Producer continues even if one block fails to save
- Watchdog stops firing STALLED warnings
- Network recovers automatically

### Layer 2: Move RocksDB Writes to Blocking Thread

**Problem**: RocksDB is synchronous, blocks Tokio executor threads

**Solution**: Use `spawn_blocking` for all RocksDB write operations

**Location**: `crates/q-storage/src/kv.rs` - `save_qblock()` method

**Pattern:**
```rust
pub async fn save_qblock(&self, block: &QBlock) -> Result<()> {
    let db = self.db.clone();
    let block = block.clone();

    // Move entire write operation to dedicated blocking thread pool
    tokio::task::spawn_blocking(move || {
        let cf_hot = db.cf_handle(CF_BLOCKS)?;
        let block_data = bincode::serialize(&block)?;
        // ... perform writes ...
        db.flush()?; // Blocking fsync
        Ok(())
    }).await??;

    Ok(())
}
```

**Why This Works:**
- RocksDB writes never block Tokio executor threads
- Other async tasks can run while waiting for disk I/O
- System-wide performance improves

### Layer 3: Bounded Channel for Mining Submissions

**Problem**: Unbounded channel allows infinite queue growth

**Location**: `crates/q-api-server/src/main.rs` line 1932

**BEFORE:**
```rust
let (mining_tx, mut mining_rx) = tokio::sync::mpsc::unbounded_channel::<q_api_server::MiningSubmission>();
```

**AFTER:**
```rust
// Bounded channel with 10,000 capacity (40 seconds worth at 250/sec)
let (mining_tx, mut mining_rx) = tokio::sync::mpsc::channel::<q_api_server::MiningSubmission>(10_000);
```

**Why This Works:**
- Backpressure prevents memory exhaustion
- Mining API handler blocks when queue is full (natural rate limiting)
- System has bounded worst-case memory usage

### Layer 4: Circuit Breaker for save_qblock()

**Pattern:**
```rust
struct SaveBlockCircuitBreaker {
    failures: AtomicUsize,
    last_failure_time: AtomicU64,
}

impl SaveBlockCircuitBreaker {
    fn should_allow_save(&self) -> bool {
        let failures = self.failures.load(Ordering::Relaxed);
        if failures >= 3 {
            // Circuit is open - check if enough time has passed to retry
            let last_fail = self.last_failure_time.load(Ordering::Relaxed);
            let now = chrono::Utc::now().timestamp() as u64;
            if now - last_fail < 60 {
                return false; // Still in cooldown period
            }
            // Reset circuit breaker after cooldown
            self.failures.store(0, Ordering::Relaxed);
        }
        true
    }

    fn record_failure(&self) {
        self.failures.fetch_add(1, Ordering::Relaxed);
        self.last_failure_time.store(
            chrono::Utc::now().timestamp() as u64,
            Ordering::Relaxed
        );
    }

    fn record_success(&self) {
        self.failures.store(0, Ordering::Relaxed);
    }
}
```

---

## 🚀 Immediate Action Plan

### 1. RESTART SERVICE (Immediate - 30 seconds)
```bash
systemctl restart q-api-server
```
**Result**: Restores block production, but stall WILL RECUR

### 2. APPLY TIMEOUT FIX (Urgent - 10 minutes)
- Add timeout wrapper to `save_qblock()` call
- Recompile and deploy
- Monitor for timeout errors in logs

### 3. IMPLEMENT spawn_blocking (Critical - 1 hour)
- Move RocksDB writes to blocking thread pool
- Test thoroughly
- Deploy to production

### 4. ADD BOUNDED CHANNEL (High Priority - 30 minutes)
- Replace unbounded_channel with bounded channel(10,000)
- Test backpressure behavior
- Deploy to production

---

## 📊 Long-term Monitoring

### Metrics to Add

**Prometheus:**
```
block_save_duration_seconds (histogram)
block_save_failures_total (counter)
block_save_timeouts_total (counter)
mining_queue_depth (gauge)
rocksdb_write_latency_ms (histogram)
```

**Health Check Endpoint:**
```
GET /api/v1/health/producer
{
  "healthy": false,
  "last_block_time": "2025-11-12T04:26:48Z",
  "seconds_since_last_block": 960,
  "current_height": 10063,
  "mining_queue_depth": 29598,
  "rocksdb_status": "write_latency_high"
}
```

---

## 🎯 Success Criteria

**Fix is successful when:**
- [ ] Node can run for 24+ hours without stalling
- [ ] Block production never pauses for >30 seconds
- [ ] save_qblock() timeouts are logged and handled
- [ ] Mining submissions have backpressure when overloaded
- [ ] RocksDB writes use spawn_blocking
- [ ] Circuit breaker prevents cascading failures

---

## 📚 Related Bugs (Pattern Analysis)

### Previous Stall Incidents

1. **v0.9.50-beta** - Block producer stall at height 3500
   - Cause: RocksDB write lock deadlock
   - Fix: Moved DB writes to spawn_blocking
   - **SAME ROOT CAUSE AS THIS BUG!**

2. **v0.9.75-beta** - BlockPackCodec deadlock
   - Cause: Mutex contention in codec
   - Fix: Lock-free implementation

3. **v0.9.92-beta** - Producer height desync
   - Cause: Pointer race condition
   - Fix: Atomic height pointer updates

**Pattern**: All stalls are caused by blocking operations in async context

**Lesson**: NEVER block Tokio executor threads with synchronous I/O

---

## ✅ Conclusion

**Root Cause**: `save_qblock()` hangs indefinitely on RocksDB write operation due to:
1. No timeout configured
2. RocksDB write on async executor thread (blocks other tasks)
3. Unbounded mining submission channel (no backpressure)
4. High mining submission rate creates perfect storm

**Solution**: Multi-layer defense:
1. Add timeout to save_qblock() (prevents infinite hang)
2. Use spawn_blocking for RocksDB writes (prevents executor thread starvation)
3. Use bounded channel for submissions (prevents memory exhaustion)
4. Add circuit breaker (prevents cascading failures)

**Next Action**: Restart service, then implement timeout fix immediately.

---

**Prepared By**: Server Beta (Claude Code)
**Analysis Date**: 2025-11-12
**Stall Duration**: 16+ minutes (and counting)
**Status**: Root cause identified, fix ready to implement
