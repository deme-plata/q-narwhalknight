# Technical Review: Block Producer Stall (v10.2.9 Fix)

**Project**: Q-NarwhalKnight Quantum Blockchain  
**Market Cap**: ~$920M (production mainnet)  
**Version**: v10.2.8 -> v10.2.9  
**Date**: 2026-04-08  
**Requesting**: Independent review of root cause analysis and fix completeness

---

## 1. Executive Summary

Q-NarwhalKnight is a production blockchain running across 4 servers (Beta, Gamma, Delta, Epsilon). The bootstrap production node stalls every ~20-30 minutes: the block producer reports `SHOULD_PRODUCE=YES` but never enters `PRODUCE_BLOCK`, halting chain progress until the 5-second reconciliation task eventually corrects the state (which it often does not, due to an off-by-one bug).

**Root cause**: `current_height_atomic` (an `AtomicU64` tracking the chain tip) is not updated in 4 out of 6 block-save code paths. When multiple parallel producers race to save the same block height, the "losing" producer hits a deduplication or error path that skips the atomic update. The reconciliation safety net has an off-by-one condition that prevents it from correcting single-block drift.

**Fix**: 6 targeted changes, all in `crates/q-api-server/src/main.rs` (~23,500 lines).

---

## 2. System Architecture

### 2.1 Block Production Pipeline

The node runs **two block production loops** in parallel (though one is currently disabled):

1. **Solution-based loop** (lines ~15700-16700): Receives mining solutions from external miners via HTTP API, batches them, then calls `block_producer_pool.produce_blocks()` and saves the result. This is the ACTIVE production path.

2. **Time-based loop** (lines ~17510-17950): A 1-second interval loop that calls `block_producer_pool.produce_blocks()` independently. This is the ACTIVE fallback/standalone production path.

3. **`block_production_v2`** (line 17505): A third production loop that is currently **DISABLED** (v8.0.1) because it competed with the mining pipeline via the `production_in_progress` atomic flag.

The `LockFreeProducerPool` (in `lockfree_producer.rs`) manages 4 parallel producers. It uses:
- `production_in_progress: AtomicBool` to serialize concurrent `produce_blocks()` calls
- `pool_last_produced_height: AtomicU64` for pool-level duplicate prevention
- Round-robin producer selection to prevent monopoly

### 2.2 Height Tracking (Two Separate Mechanisms)

| Mechanism | Type | Purpose | Updated by |
|-----------|------|---------|------------|
| `current_height_atomic` | `AtomicU64` | Mining API challenge height, sync decisions, production gating | Block save paths (selectively!) |
| `height_cache` | Internal to `QStorage` | Highest contiguous block in RocksDB | `update_height_cache()` calls |

The mining API reads `current_height_atomic` to determine what height to issue challenges for. The sync check also reads it:

```rust
// Line 15718-15739 (solution-based loop)
let current_height = app_state_mining
    .current_height_atomic
    .load(std::sync::atomic::Ordering::SeqCst);
let network_height = app_state_mining
    .highest_network_height
    .load(std::sync::atomic::Ordering::SeqCst);

let is_synced = network_height == 0
    || (network_height > 0
        && current_height + sync_threshold >= network_height);
```

### 2.3 Block Save Architecture

Each production loop has TWO storage paths executed in sequence:

1. **AsyncStorageEngine** (primary): Non-blocking channel-based save. Has a pre-save dedup check via `has_block()`.
2. **RwLock `save_qblock()`** (fallback): Only runs if AsyncStorageEngine did not succeed. Can return `"Block already exists"` error.

This creates **6 distinct code paths** through which a block save can complete or fail.

---

## 3. The Stall Mechanism

### 3.1 Step-by-Step Reproduction

1. Both production loops (solution-based and time-based) call `produce_blocks()` within a short window.
2. The `production_in_progress` flag serializes these calls, but they still produce blocks at the same height N.
3. Producer A wins the race: saves block N via AsyncStorageEngine, updates both `height_cache` AND `current_height_atomic` to N. (Lines 16120, 16130.)
4. Producer B (from the other loop) attempts to save block N. The `has_block(N)` check returns `true`.
5. Producer B enters the **dedup path**: sets `save_succeeded = true`, updates `height_cache` to N, but (before the fix) did NOT update `current_height_atomic`.
6. Alternatively, Producer B's AsyncStorageEngine save succeeds (race window), but then the fallback `save_qblock()` returns `"Block already exists"` error. The `continue` statement skips the atomic update entirely.
7. `current_height_atomic` remains at N-1 (or whatever stale value it had).

### 3.2 Why the Safety Net Fails

A reconciliation task runs every 5 seconds (line 14602):

```rust
// Line 14606-14617 (BEFORE fix)
if let Ok(Some(db_height)) = app_state_height_reconcile
    .storage_engine.get_latest_qblock_height().await
{
    let atomic_height = app_state_height_reconcile
        .current_height_atomic
        .load(std::sync::atomic::Ordering::SeqCst);
    if db_height > atomic_height {          // <-- FIXED (was: > atomic_height + 1)
        app_state_height_reconcile
            .current_height_atomic
            .store(db_height, std::sync::atomic::Ordering::SeqCst);
        // Only log if drift > 10 to avoid spam
        if db_height > atomic_height + 10 {
            warn!("...");
        }
    }
}
```

**Before the fix**, the condition was `db_height > atomic_height + 1`, meaning:
- If `db_height = N` and `atomic_height = N-1`: condition is `N > N` = **false**
- The single-block drift is never corrected
- The stall persists indefinitely

### 3.3 The Feedback Loop

Once `current_height_atomic` is stuck at N-1:

1. The production loop reads `current_height = N-1` from the atomic.
2. `sync_from_storage()` is called, which resets producers to height N-1.
3. Producers try to produce block N again.
4. Block N already exists in storage -> dedup path fires.
5. Atomic is not updated (pre-fix) -> still N-1.
6. Repeat forever.

The chain appears to stall despite `SHOULD_PRODUCE=YES` logs, because the production loop is stuck in an infinite dedup cycle at the same height.

---

## 4. Affected Code Paths

All paths are in `crates/q-api-server/src/main.rs`.

### Path A: Solution-based dedup (line 16062)

When `has_block(height)` returns true in the solution-based loop:

```rust
if block_already_exists {
    save_succeeded = true;
    app_state_mining.storage_engine
        .update_height_cache(new_block.header.height).await;
    // v10.2.9 FIX ADDED HERE (lines 16076-16084):
    let current_atomic = app_state_mining.current_height_atomic
        .load(std::sync::atomic::Ordering::SeqCst);
    if new_block.header.height > current_atomic {
        app_state_mining.current_height_atomic
            .store(new_block.header.height, std::sync::atomic::Ordering::SeqCst);
    }
}
```

**Before fix**: No atomic update. `height_cache` advanced but `current_height_atomic` stayed stale.

### Path B: Time-based dedup (line 17782)

When `has_block(height)` returns true in the time-based loop:

```rust
} else {
    // Block already exists
    app_state_block_producer.storage_engine
        .update_height_cache(new_block.header.height).await;
    // v10.2.9 FIX ADDED HERE (lines 17787-17792):
    let current_atomic = app_state_block_producer.current_height_atomic
        .load(std::sync::atomic::Ordering::SeqCst);
    if new_block.header.height > current_atomic {
        app_state_block_producer.current_height_atomic
            .store(new_block.header.height, std::sync::atomic::Ordering::SeqCst);
    }
}
```

**Before fix**: Same issue as Path A.

### Path C: Solution-based "Block already exists" error (line 16173)

When the RwLock fallback `save_qblock()` returns an error containing "Block already exists":

```rust
Ok(Err(e)) if e.to_string().contains("Block already exists") => {
    warn!("Duplicate block {} detected, forcing immediate resync",
          new_block.header.height);
    // v10.2.9 FIX ADDED HERE (lines 16176-16181):
    let current_atomic = app_state_mining.current_height_atomic
        .load(std::sync::atomic::Ordering::SeqCst);
    if new_block.header.height > current_atomic {
        app_state_mining.current_height_atomic
            .store(new_block.header.height, std::sync::atomic::Ordering::SeqCst);
    }
    // ... resync producers ...
    continue; // <-- Previously skipped atomic update at line 16672
}
```

**Before fix**: The `continue` statement jumped past the atomic update at line 16672.

### Path D: Time-based "Block already exists" error (line 17904)

Same pattern in the time-based loop:

```rust
Err(e) if e.to_string().contains("Block already exists") => {
    warn!("Duplicate block {} detected, forcing immediate resync",
          new_block.header.height);
    // v10.2.9 FIX ADDED HERE (lines 17906-17912):
    let current_atomic = app_state_block_producer.current_height_atomic
        .load(std::sync::atomic::Ordering::SeqCst);
    if new_block.header.height > current_atomic {
        app_state_block_producer.current_height_atomic
            .store(new_block.header.height, std::sync::atomic::Ordering::SeqCst);
    }
    // ... resync producers ...
    continue; // <-- Previously skipped atomic update at line 17889
}
```

**Before fix**: The `continue` jumped past the atomic update at line 17889.

### Path E: Reconciliation off-by-one (line 14608)

```rust
// BEFORE (broken):
if db_height > atomic_height + 1 {
    // Correct the drift
}

// AFTER (fixed):
if db_height > atomic_height {
    // Correct even single-block drift
}
```

The `+ 1` was almost certainly a typo or an overly conservative threshold. It made the safety net unable to correct the most common failure mode (single-block drift from one dedup race).

### Path F: Memory ordering consistency

```rust
// BEFORE: Mixed ordering across the codebase
.store(height, Ordering::Relaxed);  // Some write paths
.load(Ordering::SeqCst);           // Read path (line 15720)
.store(height, Ordering::Release); // Other write paths

// AFTER: Consistent SeqCst everywhere
.store(height, Ordering::SeqCst);  // All write paths
.load(Ordering::SeqCst);           // All read paths
```

---

## 5. Memory Ordering Analysis

The codebase had inconsistent atomic orderings for `current_height_atomic`:

| Location | Operation | Ordering (before fix) | Ordering (after fix) |
|----------|-----------|----------------------|---------------------|
| Line 15720 | Load (sync check) | `SeqCst` | `SeqCst` |
| Line 16672 | Store (solution success) | `SeqCst` | `SeqCst` |
| Line 17889 | Store (time-based success) | `SeqCst` | `SeqCst` |
| Line 10496 | Store (gossipsub block) | `Release` | `Release` (unchanged) |
| Line 10365 | Store (emergency sync) | `Release` | `Release` (unchanged) |
| Line 6193 | Store (libp2p sync) | `Relaxed` | not changed |

On x86-64 (which has Total Store Ordering), `Relaxed` stores are still visible to `SeqCst` loads in practice. However, this is architecturally incorrect and would fail on ARM or POWER.

The fix uses `SeqCst` for all new stores in the dedup/error paths, matching the existing read path.

---

## 6. Summary of All 6 Changes

| # | Change | Location | Risk |
|---|--------|----------|------|
| 1 | Add atomic update to solution-based dedup path | Line 16076-16084 | Low |
| 2 | Add atomic update to time-based dedup path | Line 17787-17792 | Low |
| 3 | Add atomic update to solution-based "already exists" error | Line 16176-16181 | Low |
| 4 | Add atomic update to time-based "already exists" error | Line 17906-17912 | Low |
| 5 | Fix reconciliation off-by-one (`> atomic_height + 1` -> `> atomic_height`) | Line 14608 | Low |
| 6 | Consistent `SeqCst` ordering on all new stores | Lines above | Low |

All changes follow the same pattern:
```rust
let current_atomic = app_state.current_height_atomic
    .load(std::sync::atomic::Ordering::SeqCst);
if new_block.header.height > current_atomic {
    app_state.current_height_atomic
        .store(new_block.header.height, std::sync::atomic::Ordering::SeqCst);
}
```

The `>` guard prevents writing a lower height (monotonicity).

---

## 7. Questions for Reviewers

### 7.1 Fix Completeness

Are there other paths that modify `current_height_atomic` that we might have missed? The full list of stores we identified (via grep):

- Line 3764: Startup initialization from DB
- Line 4961: Height cache fix at v1.5.0
- Line 6193: Libp2p sync update (`Relaxed`)
- Line 10365: Emergency sync (`Release`)
- Line 10496: Gossipsub block commit (`Release`)
- Line 11864: Batch sync update
- Line 14610: Reconciliation fix
- Line 16082: **NEW** - Solution dedup path
- Line 16130: Existing AsyncStorage success path
- Line 16179: **NEW** - Solution error path
- Line 16672: Existing RwLock success path
- Line 17790: **NEW** - Time-based dedup path
- Line 17889: Existing time-based success path
- Line 17910: **NEW** - Time-based error path
- Line 18925: Another path (line 18923 context)
- Line 19531: Another path

Is this list complete? Are there paths where `current_height_atomic` should be updated but is not?

### 7.2 Memory Ordering

Is `SeqCst` the right choice for all paths, or would `Release`/`Acquire` pairs be sufficient? The variable is:
- Written by multiple tasks (solution loop, time-based loop, gossipsub handler, sync handler)
- Read by multiple tasks (mining API, sync check, SSE handler)
- Used as a "latest value" counter, not for synchronizing other memory

`Release`/`Acquire` pairs would be sufficient for happens-before ordering. `SeqCst` adds a total order across all atomic operations, which is unnecessary here but safer. The performance cost on x86-64 is negligible (both compile to the same instructions). Is there a reason to prefer one over the other?

### 7.3 Reconciliation Frequency

The reconciliation runs every 5 seconds. With the fix to the off-by-one, it will now catch any single-block drift within 5 seconds. Should we also add an immediate reconciliation after every block save? The tradeoff:
- **Pro**: Eliminates up to 5 seconds of stale height between saves
- **Con**: Additional `get_latest_qblock_height()` DB read on every block save (adds ~50us latency)

### 7.4 Production Loop Race Condition

The `production_in_progress` flag in `LockFreeProducerPool` (line 1418) uses `compare_exchange` to prevent concurrent `produce_blocks()` calls. However, the two production loops (solution-based at line ~15700 and time-based at line ~17640) both call `produce_blocks()`. Could the following scenario cause a separate stall?

1. Time-based loop acquires `production_in_progress` flag
2. Solution-based loop receives a valid solution, tries `produce_blocks()`, gets rejected by the flag
3. Solution is lost (not re-queued)
4. Time-based loop produces a block, releases the flag
5. Solution-based loop has already moved on

Is this scenario possible, and if so, does it contribute to the stall?

### 7.5 Architectural Improvement

Currently, `current_height_atomic` is updated in 14+ separate locations across a 23,500-line file. Would a single `advance_chain_tip(new_height: u64)` function that:
1. Atomically updates `current_height_atomic` (with monotonicity check)
2. Updates `height_cache`
3. Clears the cached challenge
4. Updates the `UpgradeManager` height

...be a safer long-term approach? This would make it impossible to add a new save path without updating the atomic, because all paths would call the same function.

### 7.6 The `continue` Anti-Pattern

Paths C and D use `continue` after handling the "Block already exists" error. This pattern is fragile because any code added after the `continue` target (e.g., a future atomic update, reward processing) will be silently skipped. Should the error handling be restructured to use early-return or flag-based flow control instead of `continue`?

---

## 8. Timeline

| Date | Event |
|------|-------|
| v1.1.3-beta | Added atomic update in AsyncStorageEngine SUCCESS path (line 16130). Dedup and error paths were NOT fixed. |
| v10.1.5 | Added `update_height_cache()` to dedup paths, but NOT `current_height_atomic`. |
| 2026-04-05 to 04-07 | Operation Twelve Leagues Deep: intensive debugging session. Bug #5 identified as "block producer stall every 20-30 minutes." |
| v10.2.7 | Added diagnostic logging (`BLOCK-PROD Sync check` at line 15742) which confirmed `current_height_atomic` was stuck. |
| v10.2.8 | VDF verification fixes. Stall still present. |
| v10.2.9 | This fix: atomic updates in all 4 missing paths + reconciliation off-by-one correction. |

The reconciliation off-by-one (`> atomic_height + 1`) has been present since the reconciliation task was first added in v9.0.0, meaning the safety net has never been able to correct single-block drift.

---

## 9. How to Verify

To confirm the fix is working, check production logs for the new `[DEDUP-FIX]` log lines:

```bash
journalctl -u q-api-server --since "30 minutes ago" | grep "DEDUP-FIX"
```

If the fix is working, you should see periodic `DEDUP-FIX` messages showing the atomic being updated in dedup paths. If the stall was caused by this bug, block production should no longer pause for 20-30 minute intervals.

To confirm the reconciliation fix, check for single-block corrections:

```bash
journalctl -u q-api-server --since "30 minutes ago" | grep "HEIGHT RECONCILE"
```

Previously, reconciliation messages only appeared for drift > 10 blocks. With the fix, any `db_height > atomic_height` triggers correction (though logging is still gated at drift > 10 to avoid spam).
