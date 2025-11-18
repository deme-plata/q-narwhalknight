# Mining Handler Stops After Initial Block Production
## v1.0.3.1-beta Bug Analysis

**Date**: November 16, 2025 01:23 CET
**Version**: v1.0.3.1-beta
**Status**: CRITICAL - Block production stops after initial burst
**Priority**: P0 - Network halted

---

## Executive Summary

The timer reset fix (v1.0.3.1-beta) was successfully deployed and PARTIALLY worked:
- ✅ Blocks 1757-1760 produced immediately after restart (timer fix working!)
- ❌ Block production stopped completely at 01:18:35 (height 1760)
- ❌ Mining handler loop appears to have stopped processing

This indicates **TWO SEPARATE BUGS**:
1. Timer reset bug - **FIXED** ✅
2. Mining handler loop stall - **NOT FIXED** ❌

---

## Timeline of Events

### 01:08:26 - Service Restart with New Binary
- New binary deployed with timer reset fix
- Service started successfully (PID 431223)
- Initial sync completed

### 01:18:31-01:18:35 - Block Production Burst
- Block 1756 processing started
- Blocks 1757, 1758, 1759, 1760 saved successfully
- Each block saved ~8 times (parallel producers working)
- "🔨 PRODUCING BLOCKS NOW" messages every ~1 second

### 01:18:35 - Production Stops
- Last block saved: 1760 at 01:18:35.561995Z
- NO further "Block saved" messages
- NO further "should_produce" checks
- NO further height queries

### 01:18:36 onwards - Mining Handler Silent
- Mining submissions continue to arrive
- NO processing of mining queue
- NO block production attempts
- Height stuck at 1786 (database shows 1786, but last logged save was 1760)

---

## Evidence

### Binary Verification
```
/opt/orobit/shared/q-narwhalknight/target/release/q-api-server: Nov 16 01:02 (NEW)
/proc/431223/exe -> /opt/orobit/shared/q-narwhalknight/target/release/q-api-server
```
✅ Running binary IS the new one with timer fix

### Timer Fix Verification
```bash
$ strings /opt/orobit/shared/q-narwhalknight/target/release/q-api-server | grep "Producer synced"
Producer synced to height  - timer reset for immediate production
```
✅ Timer fix code is present in the binary

### Last Block Production Activity
```
Nov 16 01:18:34 q-api-server[431223]: ✅ Block 1759 saved successfully
Nov 16 01:18:34 q-api-server[431223]: 🔨 PRODUCING BLOCKS NOW (should_produce returned true)
Nov 16 01:18:34 q-api-server[431223]: ✅ Block 1760 saved successfully
Nov 16 01:18:35 q-api-server[431223]: ✅ Block 1760 saved successfully (last at 01:18:35.561995Z)
```

### Current Activity (01:20:00 onwards)
```
Nov 16 01:21:35 q-api-server[431223]: Mining submission queued (non-blocking): Miner: qnkf9c1446ab6c2f
Nov 16 01:21:35 q-api-server[431223]: Mining submission queued (non-blocking): Miner: qnkf9c1446ab6c2f
...
```
- Mining submissions arriving normally
- NO height checks
- NO should_produce() calls
- NO block production attempts

### Error Analysis
- NO panics or crashes detected
- NO fatal errors in logs
- Only "InsufficientPeers" warnings (expected - solo mining)
- Mining handler loop appears to have stopped executing

---

## Root Cause Hypothesis

The mining handler loop in `crates/q-api-server/src/main.rs` (lines ~4200-4500) likely has a **blocking operation** or **deadlock condition** that causes it to stop processing after producing a few blocks.

### Possible Causes

#### 1. Async Runtime Deadlock
The mining handler is an async task that processes submissions in batches. If it encounters a blocking operation that never completes, the entire loop stalls.

**Code Location**: `crates/q-api-server/src/main.rs` lines 4215-4460

```rust
while let Some(submission) = mining_rx.recv().await {
    batch_buffer.push(submission);

    if batch_buffer.len() >= 500 || last_batch_process.elapsed().as_millis() >= 20 {
        // PHASE 1-5: Process batch
        // If any phase blocks indefinitely, loop stops
    }
}
```

#### 2. Channel Backpressure
If the block producer pool's command channels fill up, sends might block indefinitely.

**Code Location**: `crates/q-api-server/src/lockfree_producer.rs` line 40
```rust
const CHANNEL_CAPACITY: usize = 10_000;
```

If 10,000 commands queue up without being processed, new sends will block.

#### 3. Producer Task Death
If all 8 producer tasks die, `should_produce()` will hang waiting for responses that never come.

**Code Location**: `crates/q-api-server/src/lockfree_producer.rs` lines 296-350

```rust
pub async fn should_produce(&self) -> Result<bool, PoolError> {
    // Query all 8 producers via channels
    // If producers are dead, this might timeout or hang
}
```

#### 4. Database Contention
RocksDB write lock contention could cause block saves to hang.

**Code Location**: Block save operations in parallel producers

---

## Diagnostic Steps Needed

### 1. Check Producer Task Liveness
```bash
journalctl -u q-api-server --since="01:18:35" | grep -E "(Producer.*died|task.*dead|channel closed)"
```

### 2. Check for Timeout Errors
```bash
journalctl -u q-api-server --since="01:18:35" | grep -iE "(timeout|timed out|deadline)"
```

### 3. Check Database Operations
```bash
journalctl -u q-api-server --since="01:18:35" | grep -iE "(rocksdb|database|save block)"
```

### 4. Add Debug Logging to Mining Handler
Add logs at each phase of the mining handler loop to identify where it's stuck:

```rust
debug!("🔍 Mining handler: Received submission");
debug!("🔍 Mining handler: Processing batch of {} submissions", batch_buffer.len());
debug!("🔍 Mining handler: Calling should_produce()");
debug!("🔍 Mining handler: should_produce returned: {}", should_produce);
debug!("🔍 Mining handler: Producing blocks...");
```

---

## Recommended Fixes

### Short-term (Emergency)
1. Add comprehensive debug logging to mining handler loop
2. Add timeout protection around should_produce() calls
3. Add circuit breaker to restart mining handler if it stalls

### Medium-term
1. Implement mining handler health monitoring
2. Add periodic "heartbeat" logs from mining handler
3. Implement automatic recovery if handler stalls

### Long-term
1. Redesign mining handler to avoid blocking operations
2. Implement proper backpressure handling
3. Add comprehensive integration tests for mining handler edge cases

---

## Next Steps

**IMMEDIATE** (Next 10 minutes):
1. Add debug logging to identify where mining handler is stuck
2. Rebuild and deploy with logging
3. Trigger block production and capture logs

**IF THAT DOESN'T REVEAL THE ISSUE**:
1. Use Task tool to explore mining handler implementation thoroughly
2. Add instrumentation to track async task lifecycle
3. Implement health monitoring for all critical async tasks

---

## Lessons Learned

1. **Multiple Bugs Can Have Similar Symptoms**: The timer bug and the mining handler stall both cause "height stuck", but have different root causes.

2. **Initial Success Doesn't Mean Complete Fix**: Blocks producing initially, then stopping indicates a separate issue from startup.

3. **Need Better Observability**: Without metrics/monitoring, diagnosing async task failures is extremely difficult.

4. **Silent Failures Are Deadly**: The mining handler stopped silently with no error messages - crash-fast would be better.

---

## References

- `crates/q-api-server/src/main.rs` - Mining handler loop (lines 4215-4460)
- `crates/q-api-server/src/lockfree_producer.rs` - Producer pool implementation
- `crates/q-api-server/src/block_producer.rs` - Individual producer with timer fix

---

**Status**: Investigating mining handler stall - need more diagnostic logging
