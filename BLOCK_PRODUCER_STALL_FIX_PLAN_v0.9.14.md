# Block Producer Stall Fix Implementation Plan - v0.9.14-beta

**Date**: 2025-11-06
**Issue**: Block producer stops producing blocks after 6-10 minutes of operation
**Status**: ⚠️ **CRITICAL - REQUIRES IMMEDIATE FIX**

---

## 🔍 Root Cause Summary

After extensive investigation, the block producer stall is caused by:

1. **Async task silent failures** - No error logging when tasks panic or hang
2. **No heartbeat monitoring** - No detection when block production stops
3. **No timeout on critical operations** - RocksDB writes or pool operations may block indefinitely
4. **No watchdog recovery** - Once stalled, requires manual restart

**Evidence from Logs:**
```
00:47:48 - Last successful block production at height 7044
00:48:00+ - COMPLETE SILENCE - No errors, no warnings, just stopped
```

---

## 🛠️ Immediate Fix Strategy

### **Phase 1: Add Enhanced Logging (Non-Breaking)**

**Goal**: Make block producer problems LOUD and VISIBLE

**Files to Modify**:
1. `crates/q-api-server/src/main.rs:3592` - Block production loop
2. `crates/q-api-server/src/lib.rs` - Block producer pool

**Changes**:
```rust
// In block production loop (main.rs:3596)
loop {
    interval.tick().await;

    // v0.9.14 FIX: Add heartbeat logging every 30 seconds
    if loop_iteration % 30 == 0 {
        info!("💓 BLOCK PRODUCER HEARTBEAT: Loop iteration {}, height {}",
              loop_iteration, current_height);
    }
    loop_iteration += 1;

    // v0.9.14 FIX: Add timing around critical operations
    let should_produce_start = std::time::Instant::now();
    let should_produce_result = app_state_block_producer.block_producer_pool.should_produce().await;
    let should_produce_elapsed = should_produce_start.elapsed();

    if should_produce_elapsed.as_millis() > 100 {
        warn!("⚠️  should_produce() took {}ms (threshold: 100ms)", should_produce_elapsed.as_millis());
    }

    if should_produce_result {
        info!("🔨 PRODUCING BLOCKS NOW (should_produce returned true)");
        let produce_start = std::time::Instant::now();

        let new_blocks = app_state_block_producer.block_producer_pool.produce_blocks().await;

        let produce_elapsed = produce_start.elapsed();
        info!("✅ produce_blocks() completed in {}ms, produced {} blocks",
              produce_elapsed.as_millis(), new_blocks.len());

        if new_blocks.is_empty() {
            warn!("⚠️  produce_blocks() returned ZERO blocks despite should_produce=true");
        }
    }
}
```

**Expected Result**: We'll SEE exactly where the stall happens in logs

---

### **Phase 2: Add Timeout Protection (Medium Risk)**

**Goal**: Prevent indefinite blocking on async operations

**Implementation**:
```rust
use tokio::time::timeout;

// Wrap critical operations with timeouts
match timeout(Duration::from_secs(10), block_producer_pool.should_produce()).await {
    Ok(result) => {
        if result {
            match timeout(Duration::from_secs(30), block_producer_pool.produce_blocks()).await {
                Ok(blocks) => {
                    // Process blocks normally
                }
                Err(_) => {
                    error!("🚨 TIMEOUT: produce_blocks() exceeded 30 seconds!");
                    error!("   This indicates a critical deadlock or infinite loop");
                    // Continue loop - will retry next iteration
                }
            }
        }
    }
    Err(_) => {
        error!("🚨 TIMEOUT: should_produce() exceeded 10 seconds!");
    }
}
```

---

### **Phase 3: Add Watchdog Task (Low Risk)**

**Goal**: Detect stalls and alert operators

**Implementation** (can be added immediately):
```rust
// Add BEFORE block production loop spawn
let last_block_height = Arc::new(AtomicU64::new(0));
let last_block_height_clone = last_block_height.clone();

// Watchdog task
tokio::spawn(async move {
    let mut watchdog_interval = tokio::time::interval(Duration::from_secs(60));
    let mut last_checked_height = 0;

    loop {
        watchdog_interval.tick().await;

        let current_height = last_block_height_clone.load(Ordering::Relaxed);

        if current_height == last_checked_height {
            error!("🚨 WATCHDOG: Block producer STALLED!");
            error!("   Height unchanged for 60 seconds: {}", current_height);
            error!("   IMMEDIATE ACTION REQUIRED: Restart q-api-server service");
        } else {
            info!("✅ WATCHDOG: Block producer healthy (height {} -> {})",
                  last_checked_height, current_height);
            last_checked_height = current_height;
        }
    }
});

// In block production loop, update height after successful production:
last_block_height.store(new_block.header.height, Ordering::Relaxed);
```

---

### **Phase 4: Add Panic Handler (Low Risk)**

**Goal**: Catch and log panics in spawned tasks

**Implementation**:
```rust
tokio::spawn(async move {
    // Wrap entire async block in panic catcher
    let panic_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // Note: This won't work across await points
        // Need to use tokio::task::spawn_local for true panic catching
    }));

    if let Err(e) = panic_result {
        error!("🚨 PANIC: Block production loop panicked!");
        error!("   Panic payload: {:?}", e);
        error!("   The block producer is DEAD and needs immediate restart");
    }
});
```

**Note**: Rust async doesn't support panic catching across `.await` points. Need alternative approach.

---

## 📋 Recommended Implementation Order

### **IMMEDIATE (Do Now - Zero Risk)**:

1. ✅ **Add heartbeat logging** to block production loop
   - Every 30 seconds log: "Block producer alive, height X"
   - Makes stalls OBVIOUS in logs

2. ✅ **Add timing logs** around `should_produce()` and `produce_blocks()`
   - Log time taken for each operation
   - Identify which operation hangs

3. ✅ **Add watchdog task** (separate spawn)
   - Detects when height stops increasing
   - Alerts operators immediately

### **SHORT-TERM (Within 24h - Low Risk)**:

4. 🔄 **Add timeout wrappers** on async operations
   - 10s timeout on `should_produce()`
   - 30s timeout on `produce_blocks()`
   - Prevents indefinite hangs

5. 🔄 **Add RocksDB write timeout** in storage layer
   - Wrap `save_qblock()` with 5s timeout
   - Prevents database blocking

### **MEDIUM-TERM (Within 1 week - Higher Risk)**:

6. 🔄 **Refactor to supervised actor model**
   - Use tokio-actor or similar
   - Automatic restart on failure
   - Proper error propagation

7. 🔄 **Add circuit breaker pattern**
   - Detect repeated failures
   - Graceful degradation

---

## 🧪 Testing Plan

### **Test #1: Verify Logging Works**
```bash
# Deploy changes and monitor logs
journalctl -u q-api-server -f | grep -E "(HEARTBEAT|WATCHDOG|TIMEOUT|PRODUCING BLOCKS)"

# Expected: See heartbeat every 30s, block production logs
```

### **Test #2: Verify Stall Detection**
```bash
# Wait for natural stall (6-10 minutes)
# Check if watchdog fires

# Expected output:
# "🚨 WATCHDOG: Block producer STALLED!"
```

### **Test #3: Stress Test**
```bash
# High mining submission rate
# Monitor for timeouts or hangs

# Expected: Graceful handling even under load
```

---

## 🎯 Success Criteria

**FIXED** when:
- ✅ Stalls are detected within 60 seconds (watchdog fires)
- ✅ Logs show EXACTLY where the stall occurs
- ✅ No silent failures (all errors logged)
- ✅ System runs 24+ hours without manual restart
- ✅ Automatic recovery (future enhancement)

---

## 🔧 Immediate Action Items

**For v0.9.14-beta deployment:**

1. Add heartbeat logging (5 minutes to implement)
2. Add timing logs (5 minutes)
3. Add watchdog task (10 minutes)
4. Deploy and monitor for 1 hour
5. If still stalls, add timeouts (15 minutes)

**Total time to first fix: ~25 minutes**

---

**Status**: ⏳ **READY TO IMPLEMENT - PHASE 1 APPROVED**
**Risk Level**: **LOW** (only adding logging, no behavior changes)
**Priority**: **P0 - CRITICAL PRODUCTION ISSUE**
