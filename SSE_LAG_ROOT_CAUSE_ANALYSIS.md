# SSE Lag Root Cause Analysis - CRITICAL BUG 🚨

**Date**: 2025-11-03 18:42 CET
**Status**: ROOT CAUSE IDENTIFIED
**Severity**: CRITICAL - System unusable due to event spam

---

## 🔍 Symptoms

### Frontend Logs (`quillon.xyz-1762191563260.log`)
```
📨 App.tsx: SSE event received - type: sse-lag {"lagged_events": 32814}
```

**Analysis**:
- Frontend SSE connection lagged by **32,814 events**
- This means 32,814 events were queued faster than the frontend could process them
- SSE buffer (100,000 events from streaming.rs:261) filled up and started dropping events
- Frontend never received `balance-updated` or `new-block` events (0 occurrences in logs)
- DAGKnight never received new blocks (0 `🎨 DAGKnight: Received new-block` logs)

### Backend Logs (`journalctl`)
```
Nov 03 18:41:39 ... 📡 [SSE] Broadcasting BalanceUpdated: wallet=qnk24e1dcabef93f, old=25957.67229, new=25957.67328, reason=mining_reward, subscribers=38
Nov 03 18:41:39 ... 📡 [SSE] Broadcasting BalanceUpdated: wallet=qnk24e1dcabef93f, old=25957.67328, new=25957.67427, reason=mining_reward, subscribers=38
Nov 03 18:41:39 ... 📡 [SSE] Broadcasting BalanceUpdated: wallet=qnk24e1dcabef93f, old=25957.67427, new=25957.67526, reason=mining_reward, subscribers=38
... (50+ events in ONE SECOND, ALL with same timestamp 17:41:39.147xxx)
```

**Analysis**:
- Backend broadcasting **individual BalanceUpdated events for EVERY mining solution**
- All 50+ events occur in **0.0004 seconds** (same millisecond!)
- Each mining solution = +0.00099 QNK reward = separate SSE event
- 38 SSE subscribers (users/tabs) receiving ALL events

---

## 🚨 ROOT CAUSE

### Issue: Individual Mining Reward Events

**Location**: `crates/q-api-server/src/main.rs` (block production loop)

Every mining solution triggers:
1. Balance update in database
2. `StreamEvent::BalanceUpdated` broadcast via SSE
3. Event sent to ALL 38 subscribers

**Scale of Problem**:
- Node height 6000+
- Mining active with multiple miners
- Each block can contain 100+ mining solutions
- 100 solutions × 1 event each = 100 SSE events per block
- Block every 10 seconds = 10 events/second MINIMUM
- With 38 subscribers = 380 events/second total throughput
- **Actual**: Much worse during block production bursts

### Issue: No Event Batching

**Current Behavior**:
```rust
// main.rs: For EACH mining solution in a block:
for solution in &block.mining_solutions {
    // Update balance
    storage.update_balance(wallet, +reward).await?;

    // Broadcast individual event
    event_broadcaster.broadcast(StreamEvent::BalanceUpdated {
        wallet_address,
        old_balance,
        new_balance,
        change_reason: "mining_reward",
        timestamp,
    });
}
```

**Problem**: 100 solutions = 100 separate database writes + 100 separate SSE events!

### Issue: Event Flooding = SSE Lag

**SSE Channel Capacity** (streaming.rs:261):
```rust
let (tx, _rx) = broadcast::channel(100000); // 100k event buffer
```

**What Happens**:
1. Block with 100 solutions produced
2. 100 `BalanceUpdated` events broadcast instantly
3. Events queued for 38 subscribers = 3,800 queued events
4. Frontend can't process fast enough (JavaScript is single-threaded)
5. Buffer fills up
6. Tokio sends `sse-lag` event with lagged count
7. Frontend processes lag event but original events are LOST

---

## 📊 Impact Assessment

### User Experience
- ❌ Balance never updates in real-time
- ❌ DAGKnight visualization frozen (no blocks received)
- ❌ Mining stats never update
- ❌ Transaction confirmations never appear
- ❌ Faucet dispense events never received

### System Performance
- 🔥 CPU: Massive serialization overhead (serde_json for every event)
- 🔥 Memory: 100k event buffer constantly near capacity
- 🔥 Network: Excessive SSE traffic (megabytes per second)
- 🔥 Database: Individual writes instead of batched transactions

### Network Impact
- 38 subscribers × 10 events/sec = 380 events/sec minimum
- Each event ~200 bytes JSON = 76 KB/sec minimum
- During block production burst: 10x higher = 760 KB/sec
- With multiple tabs open: exponentially worse

---

## ✅ SOLUTION: Event Batching & Aggregation

### Solution 1: Batch Balance Updates Per Block (CRITICAL)

**Change**: Only send ONE `BalanceUpdated` event per wallet per block

```rust
// BEFORE (current - BAD):
for solution in &block.mining_solutions {
    update_balance(wallet, +reward);
    broadcast(BalanceUpdated { old, new, reason: "mining_reward" });
}

// AFTER (batched - GOOD):
let mut balance_deltas: HashMap<WalletAddress, f64> = HashMap::new();
for solution in &block.mining_solutions {
    *balance_deltas.entry(solution.wallet).or_insert(0.0) += reward;
}

// Apply all balance updates atomically
for (wallet, delta) in balance_deltas {
    let old_balance = get_balance(wallet);
    let new_balance = old_balance + delta;
    update_balance(wallet, new_balance);

    // ONE event per wallet per block
    broadcast(BalanceUpdated {
        wallet_address: wallet,
        old_balance,
        new_balance,
        change_reason: "mining_rewards_batch", // Changed to indicate batch
        solution_count: count_for_wallet, // Add count of solutions
        timestamp,
    });
}
```

**Benefits**:
- 100 solutions for same wallet = 1 event (99% reduction!)
- Atomic database transaction (all-or-nothing)
- Correct final balance (no race conditions)
- Dramatically reduces SSE traffic

### Solution 2: Aggregate NewBlock Events

**Change**: Send block events AFTER all balance updates complete

```rust
// Process all balance updates FIRST
process_all_balance_updates_batched(&block).await?;

// THEN broadcast NewBlock event ONCE
broadcast(StreamEvent::NewBlock {
    height: block.height,
    hash: block_hash,
    solutions_count: block.mining_solutions.len(),
    unique_miners: balance_deltas.len(), // How many unique wallets got rewards
    total_rewards: total_reward_amount,
    producer_id,
    timestamp,
});
```

### Solution 3: Debounce Balance Updates (Optional)

For additional safety, add client-side debouncing:

```typescript
// App.tsx: Debounce balance updates
const debouncedBalanceUpdate = useMemo(
  () => debounce((newBalance: number) => {
    setNodeData(prev => ({ ...prev, balance: newBalance }));
  }, 500), // Wait 500ms after last update
  []
);
```

### Solution 4: Event Rate Limiting

Add rate limit to SSE broadcaster:

```rust
// streaming.rs: Only broadcast balance updates every N milliseconds per wallet
let mut last_broadcast = HashMap::<WalletAddress, Instant>::new();
const MIN_BROADCAST_INTERVAL: Duration = Duration::from_millis(1000); // 1 second

if let Some(&last) = last_broadcast.get(&wallet_address) {
    if last.elapsed() < MIN_BROADCAST_INTERVAL {
        // Skip this event, still too soon
        return Ok(());
    }
}
last_broadcast.insert(wallet_address, Instant::now());
```

---

## 🎯 Implementation Priority

### Phase 1: CRITICAL (Immediate)
1. ✅ **Batch balance updates per block** - Reduces events by 99%
2. ✅ **Remove individual mining reward SSE events** - Stop the spam
3. ✅ **Send ONE aggregated balance update per wallet per block**

### Phase 2: Important (Next)
4. **Add event rate limiting** - Prevent future event floods
5. **Add balance_delta field** - Show how much changed in batch
6. **Add solution_count field** - Show number of mining solutions in batch

### Phase 3: Nice-to-Have
7. Client-side debouncing for extra safety
8. SSE health monitoring dashboard
9. Event compression for large payloads

---

## 📝 Files to Modify

### 1. `crates/q-api-server/src/main.rs`
**Lines**: Block production loop (~3037-3100)

**Changes**:
- Accumulate balance deltas in HashMap
- Apply all balance updates atomically
- Broadcast ONE event per wallet per block
- Add solution_count to event

### 2. `crates/q-api-server/src/streaming.rs`
**Lines**: StreamEvent enum definition

**Changes**:
- Add optional `solution_count: Option<usize>` to BalanceUpdated
- Add optional `balance_delta: Option<f64>` field
- Update serialization

### 3. `crates/q-storage/src/balance_consensus.rs`
**Changes**:
- Add `batch_update_balances()` function
- Accept HashMap of wallet → delta
- Apply all updates in single transaction
- Return map of wallet → (old, new) balances

---

## 🧪 Testing Plan

### Test 1: Verify Event Reduction
```bash
# Before fix: Count balance events during 1 block
journalctl -u q-api-server.service --since "10 seconds ago" | grep "BalanceUpdated" | wc -l
# Expected: 100+ events

# After fix: Same test
# Expected: 1-10 events (one per unique miner)
```

### Test 2: Verify No SSE Lag
```bash
# Watch frontend console logs for sse-lag
# Expected: NO sse-lag events after fix
```

### Test 3: Verify Balance Updates Work
```bash
# Start miner, mine block, check balance in UI
# Expected: Balance updates correctly with batched amount
```

---

## 📊 Expected Improvements

### Event Reduction
- **Before**: 100 events per block (100 solutions)
- **After**: 1-10 events per block (unique miners)
- **Reduction**: 90-99%

### SSE Throughput
- **Before**: 380 events/second × 38 subscribers = 14,440 events/sec
- **After**: 10 events/second × 38 subscribers = 380 events/sec
- **Reduction**: 97%

### Network Bandwidth
- **Before**: ~760 KB/sec during block production
- **After**: ~20 KB/sec
- **Reduction**: 97%

### User Experience
- **Before**: SSE lag, no updates, frozen UI
- **After**: Real-time balance updates, smooth DAGKnight visualization

---

## 🚀 Next Steps

1. **Immediate**: Implement balance batching in main.rs
2. **Test**: Verify event reduction in journalctl
3. **Deploy**: Build and deploy v0.8.10-beta with fix
4. **Monitor**: Watch for sse-lag events (should be zero)
5. **Validate**: Test balance updates and DAGKnight visualization

---

**Priority**: 🚨 CRITICAL
**Impact**: High - Fixes completely broken SSE system
**Complexity**: Medium - Requires careful balance update refactoring
**Risk**: Low - Batching is safer than individual updates
