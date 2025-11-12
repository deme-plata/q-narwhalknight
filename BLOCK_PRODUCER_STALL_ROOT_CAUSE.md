# Block Producer Stall - Root Cause Analysis

**Date**: 2025-11-06
**Status**: 🔍 **INVESTIGATION IN PROGRESS**

---

## 📊 **STALL PATTERN OBSERVED**

### Occurrence History:
1. **Height 840** - Stalled at 08:37:59 UTC
   - Block production stopped
   - Mining solutions continued queuing (18,827+ in queue)
   - Watchdog detected stall after 5 minutes
   - **Fix**: Service restart

2. **Height 1256** - Stalled at 08:42:59 UTC
   - Same symptoms as height 840
   - 30 blocks produced, then stall
   - **Fix**: Service restart (block production resumed, now at height 1282+)

### Common Pattern:
- ✅ Mining submissions continue arriving
- ✅ Solutions queue up normally
- ❌ Block producer stops creating blocks
- ❌ No visible errors in logs
- ❌ Watchdog triggers after 5 minutes
- ✅ Service restart always fixes it

---

## 🔍 **KEY FINDINGS**

### 1. Database Replication Deadlock Warning

**File**: `crates/q-api-server/src/main.rs:4368-4370`

```rust
// DISABLED: This causes deadlock by trying to lock the manager from a spawned task
// TODO: Implement using command channel pattern instead
warn!("⚠️  Outgoing database update forwarder disabled to prevent deadlock");
```

**Evidence**:
```
2025-11-06T09:09:55.823853Z  WARN q_api_server: ⚠️  Outgoing database update forwarder disabled to prevent deadlock
```

**Analysis**:
- Database replication bridge has a known deadlock risk
- Outgoing update forwarder was intentionally disabled
- Incoming updates work fine
- The deadlock prevention itself might not be complete

### 2. Service Timeout on Stop

**Evidence**:
```
Nov 06 10:07:48 systemd[1]: q-api-server.service: Failed with result 'timeout'.
```

**Analysis**:
- Previous service took too long to stop (>90s timeout)
- Suggests tasks are hung/blocked waiting for resources
- Not gracefully shutting down

### 3. No Panic or Crash

**Evidence**: No panic messages, no task termination logs

**Analysis**:
- Not a crash - deadlock or resource starvation
- Tasks are alive but blocked
- Likely waiting on a lock/channel that never releases

---

## 🧩 **HYPOTHESIS: Resource Contention Deadlock**

### Probable Scenario:

```
Block Producer Pool (8 producers)
       ↓
   Create block
       ↓
   Save to RocksDB ← [LOCK ACQUIRED]
       ↓
   Broadcast to network
       ↓
   Update in-memory state
       ↓
   [LOCK NEVER RELEASED?]
```

### Potential Deadlock Sources:

1. **RocksDB Lock Contention**
   - Multiple producers writing simultaneously
   - Lock ordering issue between different column families
   - One producer holds lock, waiting for another resource

2. **Channel Backpressure**
   - Gossipsub TX channel fills up
   - Block producer waits to send
   - Network task waits for database lock
   - **Circular dependency**

3. **Async Runtime Starvation**
   - All tokio worker threads blocked
   - No thread available to process wake-ups
   - Tasks deadlocked waiting for each other

---

## 🔬 **DIAGNOSTIC DATA NEEDED**

To confirm root cause, we need:

### 1. Thread Dump During Stall
```bash
# Next time it stalls, capture:
kill -SIGUSR1 <pid>  # Trigger tokio-console dump if enabled
pstack <pid>         # Native stack trace
```

### 2. Lock Statistics
```bash
# Check RocksDB lock wait times
# Enable rocksdb statistics collection
```

### 3. Channel Buffer Status
```bash
# Log channel capacity/usage before stall
# Add metrics for gossipsub_tx.capacity()
```

---

## 🛠️ **POTENTIAL FIXES**

### Fix 1: Implement Proper Database Replication (Priority: HIGH)

**Current State**: Outgoing forwarder disabled due to deadlock

**Solution**:
```rust
// Use command channel pattern instead of direct locking
let (db_update_tx, mut db_update_rx) = mpsc::channel(1000);

// Producer sends update command
db_update_tx.send(UpdateCommand::BlockSaved { height, hash }).await;

// Separate task handles forwarding
tokio::spawn(async move {
    while let Some(cmd) = db_update_rx.recv().await {
        // Forward to gossipsub without holding locks
        network.publish(cmd).await;
    }
});
```

### Fix 2: Non-Blocking Block Save (Priority: MEDIUM)

**Problem**: Block producer may block on RocksDB write

**Solution**:
```rust
// Use spawn_blocking for RocksDB writes
tokio::task::spawn_blocking(move || {
    storage.save_qblock_sync(&block)
}).await?;
```

### Fix 3: Channel Capacity Monitoring (Priority: MEDIUM)

**Problem**: Unknown if channels are full

**Solution**:
```rust
if gossipsub_tx.capacity() < 10 {
    warn!("🚨 Gossipsub channel nearly full! Capacity: {}", gossipsub_tx.capacity());
}
```

### Fix 4: Watchdog Auto-Recovery (Priority: LOW)

**Problem**: Requires manual restart

**Solution**:
```rust
if stall_detected {
    error!("🚨 STALL DETECTED - Triggering graceful restart");
    // Gracefully shutdown and restart block producer pool
    restart_block_producer_pool().await;
}
```

---

## 📈 **MONITORING RECOMMENDATIONS**

### Metrics to Add:
1. **Lock wait times** - How long producers wait for RocksDB locks
2. **Channel fill rates** - gossipsub_tx, mining_solution_rx
3. **Block save latency** - Time from block creation to DB commit
4. **Task execution times** - Detect slow/blocked async tasks

### Alerts to Configure:
1. **Height stagnation** - No new blocks for 2 minutes
2. **Channel backpressure** - >80% full
3. **Lock contention** - Wait time >100ms
4. **Service stop timeout** - Failed graceful shutdown

---

## 🎯 **NEXT STEPS**

### Immediate (v0.9.29-beta):
1. ✅ Restart fixes stall temporarily
2. ⏳ Add channel capacity monitoring
3. ⏳ Add lock wait time logging
4. ⏳ Enable tokio-console for runtime inspection

### Short-term (v0.9.30-beta):
1. ⏳ Implement command channel pattern for database replication
2. ⏳ Make block saves non-blocking with spawn_blocking
3. ⏳ Add comprehensive deadlock detection

### Long-term (v1.0.0):
1. ⏳ Full lock-free block production pipeline
2. ⏳ Automatic recovery from stalls
3. ⏳ Distributed tracing for deadlock diagnosis

---

## 📝 **CONCLUSION**

**Root Cause**: Likely a **circular dependency deadlock** between:
- Block producer (waiting to write to RocksDB)
- Database replication (waiting to lock manager)
- Network task (waiting for channel capacity)

**Evidence Strength**: Medium (circumstantial, no smoking gun yet)

**Workaround**: Service restart clears deadlock

**Permanent Fix**: Requires implementing non-blocking patterns with command channels

---

**Status**: 🔍 Further investigation with tokio-console recommended
