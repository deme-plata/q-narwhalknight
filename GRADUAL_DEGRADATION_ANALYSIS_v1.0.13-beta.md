# Gradual Degradation Analysis - v1.0.13-beta

**Date**: November 16, 2025
**Document Version**: 1.0
**Based On**: External AI consultation (aireply12.md)

---

## Executive Summary

The gradual degradation pattern (4 blocks → 442 blocks → 2,484 blocks) suggests a **time-based saturation failure** rather than a pure synchronization race. External AI analysis identified **producer command queue saturation** as the likely root cause.

### Mathematical Evidence

```
Observed pattern:
- v1.0.2-beta: 4 blocks (~4 seconds)
- v1.0.8-beta: 442 blocks (~442 seconds ≈ 7.4 minutes)
- v1.0.9-beta: 2,484 blocks (~2,484 seconds ≈ 41 minutes)

Channel configuration:
- Bounded channel capacity: 10,000 commands
- Number of producers: 8
- Commands per block: ~2-3 (should_produce, sync_from_storage, etc.)
- Block rate: 1 per second

Expected saturation:
10,000 capacity ÷ (8 producers × 2.5 commands/sec) ≈ 500 seconds ≈ 8.3 minutes
```

**Actual failure at 41 minutes suggests** command processing backlog, where new commands accumulate faster than they're consumed.

---

## Root Cause: Producer Command Queue Saturation

### The Issue

Each producer has a bounded channel with capacity 10,000:

```rust
// crates/q-api-server/src/lockfree_producer.rs (approx line 200)
let (command_tx, mut command_rx) = mpsc::channel::<ProducerCommand>(10_000);
```

**Commands sent per block cycle**:
1. `should_produce()` - Called every 1 second by time-based loop
2. `sync_from_storage()` - Called after each block production
3. `produce_block()` - Called when mining solution found
4. Other administrative commands (get_height, health checks, etc.)

**Under high load**:
- Mining submissions queue up `produce_block()` commands
- Time-based loop sends `should_produce()` every second
- Sync operations send `sync_from_storage()` after each block
- **Result**: Channel fills up → `try_send()` fails → `should_produce()` returns `false`

---

## Why v1.0.13-beta Crash-Fast Fixes Help

### Fix #1: Explicit Error Types

**Before** (silent failure):
```rust
if let Err(e) = self.command_tx.try_send(...) {
    error!("Failed to send");  // Logged but hidden
    return false;  // Treated as "don't produce"
}
```

**After v1.0.13-beta** (loud crash):
```rust
if let Err(e) = self.command_tx.try_send(...) {
    error!("🚨 FATAL: Channel send failed: {:?}", e);
    return Err(ShouldProduceError::CommandSendFailed(...));
}

// At pool level:
match pool.should_produce().await {
    Ok(result) => result,
    Err(e) => {
        error!("🚨 FATAL: Producer pool unhealthy: {}", e);
        std::process::exit(1);  // Crash and let systemd restart
    }
}
```

**Impact**: Instead of silent 41-minute degradation → **immediate crash** → systemd restart → fresh state

---

## Additional Fixes Needed (Based on External AI)

### Fix #3: Channel Capacity Monitoring

**Add to producer health check**:

```rust
// crates/q-api-server/src/lockfree_producer.rs
impl LockFreeBlockProducer {
    pub fn get_channel_stats(&self) -> ChannelStats {
        ChannelStats {
            capacity: 10_000,
            len: self.command_tx.capacity() - self.command_tx.max_capacity(),  // Approximate
            is_full: self.command_tx.is_full(),  // If API available
            is_closed: self.command_tx.is_closed(),
        }
    }
}

impl LockFreeProducerPool {
    pub async fn monitor_channel_health(&self) {
        for (id, producer) in self.producers.iter().enumerate() {
            let stats = producer.get_channel_stats();

            if stats.len > 8000 {  // 80% capacity
                warn!("⚠️  Producer #{} channel at {}/10000 capacity!", id, stats.len);
            }

            if stats.is_full {
                error!("🚨 Producer #{} channel FULL - will drop commands!", id);
                // Consider: increase capacity or crash-fast
            }
        }
    }
}
```

**Add to watchdog** (main.rs):
```rust
// Every 10 seconds
if let Err(e) = app_state.block_producer_pool.monitor_channel_health().await {
    error!("Channel health check failed: {}", e);
}
```

### Fix #4: Oneshot Channel Pattern Fix

**Current problem**:
```rust
// BAD: Await receiver even if send failed
let (reply_tx, reply_rx) = oneshot::channel();
self.command_tx.try_send(ProducerCommand::ShouldProduce(reply_tx));  // Ignores error
match reply_rx.await {  // Will hang if send failed!
    ...
}
```

**Fixed pattern** (already in v1.0.13-beta):
```rust
// GOOD: Only await if send succeeded
let (reply_tx, reply_rx) = oneshot::channel();

if let Err(e) = self.command_tx.try_send(ProducerCommand::ShouldProduce(reply_tx)) {
    return Err(ShouldProduceError::CommandSendFailed(e));  // Don't await!
}

// Only reach here if send succeeded
match timeout(TIMEOUT, reply_rx).await {
    ...
}
```

### Fix #5: Increase Channel Capacity (If Needed)

If monitoring shows consistent saturation:

```rust
// Increase from 10,000 to 100,000
let (command_tx, mut command_rx) = mpsc::channel::<ProducerCommand>(100_000);
```

**Trade-offs**:
- ✅ More headroom for bursts
- ✅ Better throughput under high mining load
- ❌ More memory usage (8 producers × 100k capacity × command size)
- ❌ Delayed crash detection (failures take longer to surface)

**Recommendation**: Start with monitoring, only increase if data shows consistent >80% usage.

---

## Testing Strategy

### Test 1: Reproduce Queue Saturation

```bash
#!/bin/bash
# reproduce_queue_saturation.sh

# Start API server with debug logging
RUST_LOG=debug cargo run --release --bin q-api-server &
API_PID=$!

# Start 10 miners to create high command load
for i in {1..10}; do
    Q_DB_PATH=./data-miner-$i timeout 36000 ./target/release/q-miner --threads 8 &
    MINER_PIDS+=($!)
done

# Monitor for channel saturation warnings
journalctl -u q-api-server -f | grep -E "(channel.*capacity|channel.*FULL|🚨)"

# After test
kill $API_PID ${MINER_PIDS[@]}
```

**Expected**:
- v1.0.13-beta: Crash with "Channel send failed" after ~500 seconds
- With capacity monitoring: Warnings at 80% capacity, then crash

### Test 2: Validate Crash-Fast Behavior

```bash
#!/bin/bash
# validate_crash_fast.sh

systemctl stop q-api-server
rm -rf data-mine12
systemctl start q-api-server

# Monitor for crashes
journalctl -u q-api-server -f | grep -E "(FATAL|exit|systemd.*restart)" &

# Run for 1 hour
sleep 3600

# Check restart count
RESTARTS=$(journalctl -u q-api-server --since "1 hour ago" | grep "Started" | wc -l)
echo "Service restarted $RESTARTS times"
```

**Expected**:
- WITHOUT crash-fast: 0 restarts, silent deadlock at ~41 minutes
- WITH v1.0.13-beta: 1-2 restarts, each recovers within seconds

---

## Deployment Plan

### Phase 1: Deploy v1.0.13-beta (Current Build)

**Changes**:
- ✅ Explicit error types (`Result<bool, Error>` instead of `bool`)
- ✅ Crash-fast on producer failures
- ✅ Height invariant enforcement

**Expected Impact**:
- Silent 41-minute deadlock → **Immediate crash on channel failure**
- Systemd auto-restart → Network recovers within seconds
- Crash logs show exact failure mode

### Phase 2: Add Channel Monitoring (v1.0.14-beta)

**Additional Changes**:
- Channel capacity monitoring in watchdog
- Warning logs at 80% capacity
- Metrics export for capacity tracking

**Expected Impact**:
- Visibility into queue saturation patterns
- Early warning before crashes
- Data to inform capacity increases

### Phase 3: Optimize Command Processing (v1.0.15-beta)

**Potential Optimizations**:
- Prioritize `should_produce()` commands over mining commands
- Batch sync operations to reduce command count
- Implement command queue backpressure
- Consider unbounded channel with memory limits

---

## Monitoring Commands

### Check Channel Health
```bash
# Look for capacity warnings
journalctl -u q-api-server | grep -E "channel.*capacity|channel.*FULL"

# Count crash-fast restarts
systemctl status q-api-server | grep "Active:"
journalctl -u q-api-server --since "1 hour ago" | grep "FATAL.*Producer pool unhealthy"
```

### Measure Time to Failure
```bash
# Track blocks until crash
journalctl -u q-api-server -f | grep -E "Block #|FATAL" |
    awk '/Block #/{block=$2} /FATAL/{print "Failed after block " block; exit}'
```

### Verify Crash-Fast Working
```bash
# Should see immediate crashes, not gradual degradation
journalctl -u q-api-server --since "10 minutes ago" |
    grep -E "(FATAL|exit.*1|systemd.*restart)"
```

---

## Success Metrics

### v1.0.13-beta Goals

**Primary Objective**: Replace silent deadlock with loud crash-and-restart

| Metric | Before (v1.0.9-beta) | Target (v1.0.13-beta) |
|--------|----------------------|------------------------|
| Time to detect failure | 41+ minutes | <1 second |
| Network downtime | Indefinite (manual fix) | <5 seconds (auto-restart) |
| Diagnostic visibility | Silent logs | Loud FATAL errors |
| Recovery method | Manual DB reset | Automatic restart |

**Secondary Objective**: Gather data for long-term fix

- Channel capacity usage patterns
- Command queue saturation frequency
- Optimal channel size determination

### Long-Term Goals (v1.0.15+)

- **Zero crashes** from channel saturation
- **Proactive capacity management** based on load
- **Graceful degradation** instead of hard crashes
- **Sub-second recovery** from producer failures

---

## External AI Recommendations Summary

1. ✅ **Make errors explicit** - v1.0.13-beta implements `Result<bool, Error>`
2. ✅ **Crash-fast philosophy** - v1.0.13-beta exits on producer failures
3. 🔄 **Channel monitoring** - Pending v1.0.14-beta
4. ✅ **Oneshot pattern fix** - v1.0.13-beta only awaits after successful send
5. 🔄 **Capacity tuning** - Data collection in progress

**Quote from External AI**:
> "The producer-pool API conflates infrastructure failures (task dead, channel full, timeout) with 'no, don't produce', so pool-level liveness fails silently."

**Our Response**: v1.0.13-beta separates these failure modes and makes them explicit through the type system and crash-fast behavior.

---

**Status**: ✅ v1.0.13-beta implementing first phase of fixes
**Next**: Deploy and monitor for crash-fast behavior
**Long-term**: Add channel monitoring and capacity optimization

**Document Author**: Claude Code (Server Beta)
**Based On**: External AI consultation (aireply12.md)
**Last Updated**: 2025-11-16 08:45 UTC
