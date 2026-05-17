# Technical Review: Genesis Sync Balance Integrity (v10.5.4)

**Date**: 2026-05-06  
**Branch**: `feature/safe-batched-sync-v1.0.2`  
**Commit range**: ceb1896c → (this patch)  
**Author**: Server Alpha  
**Reviewer target**: DeepSeek R1 (external review)

---

## 1. Executive Summary

Fresh nodes starting with `Q_SKIP_CHECKPOINT=1` (forced full-genesis sync) correctly downloaded all 16.7 million blocks from Epsilon in under 52 seconds via warp sync, but displayed **zero wallet balances** afterward. The root cause was a single mis-scoped conditional: the flag that was intended to disable the height probe in `turbo_sync` also disabled the HTTP balance state sync, leaving `wallet_balances` empty in RAM.

Additionally, three supporting diagnostic gaps were identified:
- No warning when the Transaction commit path advances `qblock:latest` non-sequentially (a legitimate but opaque behavior)
- No logging of wallet count before/after `merge_http_snapshot` (making balance-sync regressions invisible)
- No mechanism to suppress the catch-up loop independently from warp sync (needed for controlled genesis replay)

This document covers all four bugs and the five code changes that address them.

---

## 2. Root Cause Analysis

### Bug 1 (CRITICAL): `Q_SKIP_CHECKPOINT=1` skips balance state sync

**File**: `crates/q-api-server/src/main.rs`  
**Lines before fix**: ~24135–24145

**Before** (broken):
```rust
let skip_state_snapshot = std::env::var("Q_SKIP_CHECKPOINT")
    .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
    .unwrap_or(false);
if skip_state_snapshot {
    info!("🚫 [STATE SYNC] Q_SKIP_CHECKPOINT=1 — skipping HTTP state snapshot...");
} else {
    q_api_server::state_sync_api::spawn_state_sync_task(app_state_for_sync, our_port);
    info!("🔄 [STATE SYNC] Background state sync task spawned");
}
```

The variable name `skip_state_snapshot` was ambiguous. The intent of `Q_SKIP_CHECKPOINT=1` is to tell `turbo_sync.rs` not to perform an HTTP height probe (which would jump the sync pointer to Epsilon's current height of ~16.7M, then only download the top K blocks). However, the implementation also skips `spawn_state_sync_task`, which fetches wallet balances from peers via `do_http_state_sync`.

**Result**: Node had correct `contiguous_height = 16,679,203` but `wallet_balances.len() = 0` in RAM. Every `/api/v1/balance` query returned 0.

**After** (fixed):
```rust
let skip_checkpoint = std::env::var("Q_SKIP_CHECKPOINT")
    .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
    .unwrap_or(false);
if skip_checkpoint {
    warn!("⚠️  [GENESIS SYNC] Q_SKIP_CHECKPOINT=1 — height probe skipped in turbo_sync.");
    warn!("    Balance state sync will still run...");
}
let skip_balance_sync = std::env::var("Q_SKIP_BALANCE_SYNC")
    .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
    .unwrap_or(false);
if !skip_balance_sync {
    q_api_server::state_sync_api::spawn_state_sync_task(app_state_for_sync, our_port);
}
```

The fix decouples the two behaviors:
- `Q_SKIP_CHECKPOINT=1` → only the height probe in `turbo_sync.rs` is skipped
- `Q_SKIP_BALANCE_SYNC=1` → newly introduced, explicitly skips balance HTTP sync (escape hatch for edge cases)
- Default behavior (neither set): both height probe and balance sync run normally

---

### Bug 2 (MEDIUM): Transaction commit path advances `qblock:latest` non-sequentially without diagnostic logging

**File**: `crates/q-storage/src/transaction.rs`  
**Lines before fix**: ~376–401

The `v1.0.64-beta` CRITICAL FIX wrote `qblock:latest = max(batch_heights)` inside `Transaction::commit()`. This is correct for catch-up sync behavior — getting to the chain tip fast — but it creates a pointer that is potentially **ahead of the actual contiguous chain**.

For example:
- Warp sync has sequentially committed blocks 0–15,755,104
- Catch-up Transaction downloads blocks 15,755,105–16,674,116 (may be sparse/non-contiguous)
- `Transaction::commit()` sets `qblock:latest = 16,674,116`
- `get_highest_contiguous_block()` now returns `16,674,116`
- But blocks between 15.7M and 16.7M may have gaps

The `write_batch_qblocks` path in `lib.rs` does sequential continuity verification correctly (the contiguity loop at ~line 1341–1382), but `Transaction::commit()` bypasses this. There was no logging to distinguish these two code paths.

**After** (diagnostic logging added):
```rust
// v10.5.4: DEBUG — show when Transaction advances pointer non-sequentially.
if max_height > current_pointer {
    let gap = max_height - current_pointer;
    if gap > 1000 {
        warn!("🔍 [TX POINTER DEBUG] Non-sequential advance: {} → {} (gap: {} blocks). \
               This is the catch-up sync jumping pointer ahead of sequential warp sync. \
               height_cache will reflect {} but blocks between {} and {} may have gaps.",
              current_pointer, max_height, gap, max_height, current_pointer, max_height);
    } else {
        debug!("🔍 [TX POINTER DEBUG] Sequential advance: {} → {} (+{} blocks)",
               current_pointer, max_height, gap);
    }
}
```

Threshold of 1000 blocks chosen to avoid noise from normal small-batch commits while catching the 693K-block non-sequential jump clearly in logs.

---

### Bug 3 (HIGH): No logging to distinguish height_cache semantics

**File**: `crates/q-storage/src/lib.rs`  
**Lines before fix**: ~1356–1358 (inside `write_batch_qblocks`)

The existing log `"🔍 [TURBO CONTIGUITY] Batch did NOT advance pointer"` was accurate but had no operational guidance. When a batch of blocks doesn't extend the sequential chain (e.g., they are beyond a gap), the log gave no indication that this creates a gap or how it will be resolved.

**After** (added warning with operational context):
```rust
} else if !heights.is_empty() {
    info!("🔍 [TURBO CONTIGUITY] Batch did NOT advance pointer ...");
    warn!("⚠️  [GENESIS INTEGRITY] Batch {} blocks ({}-{}) did NOT extend contiguous chain \
           (current contiguous: {}, first_height: {}, expected_next: {}). \
           This creates a gap — sequential warp sync will fill it later.",
          heights.len(), heights.first().unwrap(), heights.last().unwrap(),
          contiguous_height, heights.first().unwrap(), contiguous_height + 1);
}
```

---

### Bug 4 (MEDIUM): Catch-up loop interferes with sequential genesis sync

**File**: `crates/q-api-server/src/main.rs`  
**Lines**: ~19947–19975 (sync loop, after gap calculation)

When running with `Q_SKIP_CHECKPOINT=1`, the warp sync Phase 2 (32 parallel streams) correctly downloads blocks from genesis. However, within the same startup window (~52 seconds), P2P gossipsub broadcasts from other nodes set `current_height_atomic` to the network tip (e.g., 16,674,116).

The catch-up sync loop (`gap > NEAR_TIP_THRESHOLD`) then sees:
- `current_height` = warp sync progress (e.g., 15,755,104)
- `network_height` = 16,674,116 (from gossip)
- `gap` = 693K blocks → triggers catch-up sync

The catch-up sync uses `Transaction::commit()` which advances `qblock:latest` non-sequentially. The sequential warp sync must then fill the 693K gap afterward. While the end result is correct (all blocks eventually present), the process is harder to reason about and creates a window where `contiguous_height` reflects the Transaction pointer, not the true sequential chain end.

**After** (new `Q_GENESIS_SYNC_ONLY=1` env var):
```rust
let genesis_sync_only = std::env::var("Q_GENESIS_SYNC_ONLY")
    .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
    .unwrap_or(false);

if genesis_sync_only {
    let warp_progress = current_height;
    debug!("🌱 [GENESIS SYNC ONLY] Warp sync progress: height={}/{} ({:.1}%). Catch-up suppressed.",
           warp_progress, network_height,
           if network_height > 0 { warp_progress as f64 / network_height as f64 * 100.0 } else { 0.0 });
    adaptive_interval_ms = 5000;
    continue;
}
```

When set, the sync loop iteration short-circuits before the `if (current_height == 0 && network_height > 0) || (gap > NEAR_TIP_THRESHOLD)` condition. Warp sync Phase 2 continues uninterrupted. The loop still wakes on `sync_trigger` (gossipsub) but does not initiate catch-up downloads.

---

## 3. The Balance Data Gap: Why Blocks ≠ Balances

This bug exposed a fundamental architectural assumption: **block data and balance data are maintained independently**.

### Block storage (RocksDB, sequential)
- Warp sync Phase 2 writes blocks to `CF_BLOCKS` column family via `write_batch_qblocks`
- Sequential contiguity is verified before advancing `qblock:latest`
- Block data persists across restarts

### Balance storage (in-memory HashMap + RocksDB, non-sequential)
- `wallet_balances: Arc<RwLock<HashMap<[u8;32], u128>>>` in `AppState`
- Populated either by:
  - **HTTP state sync** (`do_http_state_sync` in `state_sync_api.rs`): fetches a snapshot from Epsilon
  - **P2P gossipsub balance updates**: real-time propagation of balance changes
  - **Chain scanning / migration**: disabled by default (`CHAIN_SCAN_ENABLED=false`)
- **Not reconstructed from block data on startup** — chain scanning would take hours

### The gap
On a fresh node with `Q_SKIP_CHECKPOINT=1` (before this fix):
1. All 16.7M blocks are downloaded correctly (block store complete)
2. `wallet_balances` HashMap starts empty on every startup
3. `spawn_state_sync_task` was skipped → HTTP sync never runs
4. P2P gossipsub balance updates only propagate **new** changes → historical balances not recovered
5. Result: `wallet_balances.len() == 0` despite correct block height

### Fix correctness
After this fix, `spawn_state_sync_task` always runs (unless `Q_SKIP_BALANCE_SYNC=1`). The HTTP sync fetches a full `FullStateSnapshot` from Epsilon (which has been running since genesis) and merges all wallet balances into RAM. This is the same mechanism used in normal node startup — `Q_SKIP_CHECKPOINT=1` no longer interferes with it.

---

## 4. All Fix Changes: Before/After

### Change 1: `crates/q-api-server/src/main.rs` (~line 24135)

Decouple balance sync from height-probe skip. See section 2, Bug 1 above.

### Change 2: `crates/q-storage/src/transaction.rs` (~line 376)

Add diagnostic warning when Transaction advances pointer by >1000 blocks non-sequentially. See section 2, Bug 2 above.

### Change 3: `crates/q-storage/src/lib.rs` (~line 1356)

Add `[GENESIS INTEGRITY]` warning when `write_batch_qblocks` batch does not extend the sequential chain. See section 2, Bug 3 above.

### Change 4: `crates/q-api-server/src/state_sync_api.rs` (~line 1102)

Add `[BALANCE SYNC]` log lines showing wallet count before and after `merge_http_snapshot`:

**Before**:
```rust
info!("🔄 [STATE SYNC HTTP] Using best snapshot from {} ({} wallets)", peer_url, snapshot.wallet_balances.len());
let result = merge_http_snapshot(app_state, &snapshot).await;
```

**After**:
```rust
info!("🔄 [STATE SYNC HTTP] Using best snapshot from {} ({} wallets)", peer_url, snapshot.wallet_balances.len());

let before_count = {
    let balances = app_state.wallet_balances.read().await;
    balances.len()
};
info!("📊 [BALANCE SYNC] Before: {} wallets in RAM. Applying {} wallets from {}.",
      before_count, snapshot.wallet_balances.len(), peer_url);

let result = merge_http_snapshot(app_state, &snapshot).await;

let after_count = {
    let balances = app_state.wallet_balances.read().await;
    balances.len()
};
info!("📊 [BALANCE SYNC] After: {} wallets in RAM (+{} new wallets applied).",
      after_count, after_count.saturating_sub(before_count));
```

### Change 5: `crates/q-api-server/src/main.rs` (~line 19947)

Add `Q_GENESIS_SYNC_ONLY=1` gate before the catch-up loop condition. See section 2, Bug 4 above.

---

## 5. Test Procedure

### 5.1 Verify balance sync now runs with Q_SKIP_CHECKPOINT=1

```bash
# Start fresh node with Q_SKIP_CHECKPOINT=1
Q_SKIP_CHECKPOINT=1 Q_DB_PATH=/tmp/test-genesis ./q-api-server --port 8090

# Within 2 minutes, check logs for balance sync:
journalctl -u q-api-server --since "2 minutes ago" | grep "BALANCE SYNC"
# Expected:
# 📊 [BALANCE SYNC] Before: 0 wallets in RAM. Applying 847 wallets from http://...
# 📊 [BALANCE SYNC] After: 847 wallets in RAM (+847 new wallets applied).

# Should NOT see:
# 🚫 [STATE SYNC] Q_SKIP_CHECKPOINT=1 — skipping HTTP state snapshot...
```

### 5.2 Verify balance sync is skippable with Q_SKIP_BALANCE_SYNC=1

```bash
Q_SKIP_CHECKPOINT=1 Q_SKIP_BALANCE_SYNC=1 ./q-api-server --port 8090
# Should see:
# ⚠️  [GENESIS SYNC] Q_SKIP_CHECKPOINT=1 — height probe skipped in turbo_sync.
# 🚫 [STATE SYNC] Q_SKIP_BALANCE_SYNC=1 — skipping HTTP balance snapshot entirely.
# (and no 📊 [BALANCE SYNC] lines)
```

### 5.3 Verify TX POINTER DEBUG fires during catch-up

```bash
# On a node syncing from scratch without Q_GENESIS_SYNC_ONLY:
journalctl -u q-api-server -f | grep "TX POINTER DEBUG"
# Should see non-sequential advance warning when catch-up Transaction commits:
# 🔍 [TX POINTER DEBUG] Non-sequential advance: 15755104 → 16674116 (gap: 919012 blocks).
```

### 5.4 Verify Q_GENESIS_SYNC_ONLY suppresses catch-up

```bash
Q_SKIP_CHECKPOINT=1 Q_GENESIS_SYNC_ONLY=1 ./q-api-server --port 8090
journalctl -u q-api-server -f | grep "GENESIS SYNC ONLY"
# Should see periodic:
# 🌱 [GENESIS SYNC ONLY] Warp sync progress: height=5234192/16674116 (31.4%). Catch-up suppressed.
# Should NOT see:
# ✅ [SYNC ACTIVATION] Reason: Behind network
```

### 5.5 Regression: Normal nodes unaffected

```bash
# Start without any special flags (normal operation)
./q-api-server --port 8080
# Should see:
# 🔄 [STATE SYNC] Background balance state sync task spawned (queries Epsilon first)
# NOT any Q_SKIP warnings
```

---

## 6. Architecture Recommendations

### 6.1 Introduce a proper contiguous-height tracker

The current `height_cache` has mixed semantics:
- Set by `write_batch_qblocks` (sequential, contiguity-verified)
- Also set by `Transaction::commit()` via `qblock:latest` pointer (non-sequential)
- Also loaded from `qblock:contiguous_verified` on startup (persisted sequential tip)

**Recommendation**: Separate the two into distinct fields:
- `contiguous_height: AtomicU64` — only advanced by `write_batch_qblocks`, always means "all blocks 0..N exist"
- `pointer_height: AtomicU64` — advanced by both paths, means "highest known block height, may have gaps below"

Expose both via the `/api/v1/status` endpoint so operators can see: `"contiguous_height": 15755104, "pointer_height": 16674116` and understand that a gap fill is in progress.

### 6.2 Decouple balance recovery modes

Currently there are three ways to populate `wallet_balances`:
1. HTTP state sync from peer (fast, ~2 minutes, but requires a synced peer)
2. P2P gossipsub balance updates (real-time, but only gets new changes)
3. Chain scanning / migration (complete, but disabled — would take hours on 16.7M blocks)

**Recommendation**: Make mode selection explicit via env vars:

| Mode | Env var | When to use |
|------|---------|-------------|
| HTTP sync (default) | (none) | Normal operation, fastest to get balances |
| P2P gossip only | `Q_SKIP_BALANCE_SYNC=1` | When no trusted HTTP peer available |
| Chain scan | `Q_CHAIN_SCAN_BALANCES=1` | Full self-auditing node (accept multi-hour startup) |

### 6.3 Warp sync progress reporting

The 52-second full sync is impressive but currently invisible to operators. The logs show individual batch commits but no overall progress bar tied to "blocks 0 → 16.7M downloaded". The existing `sync_start_time` / `sync_start_height` mechanism could feed a progress line like:

```
🌊 [WARP SYNC] 8,234,192 / 16,674,116 blocks (49.4%) — 158,000 blk/s — ETA 53s
```

This would make genesis syncs observable without having to grep multiple log lines.

### 6.4 Balance verification after HTTP sync

After `do_http_state_sync` applies balances, there is no consistency check against the local block data. On a fully-synced node, the block-derived balances should match the HTTP-fetched balances within the emission variance. A spot-check that samples 100 random wallets and verifies their balance matches what chain replay would produce would catch future balance-sync divergence early.

---

## 7. Files Changed

| File | Change type | Description |
|------|------------|-------------|
| `crates/q-api-server/src/main.rs` | Bug fix | Decouple Q_SKIP_CHECKPOINT from balance sync |
| `crates/q-api-server/src/main.rs` | Feature | Q_GENESIS_SYNC_ONLY=1 gate for catch-up loop |
| `crates/q-storage/src/transaction.rs` | Diagnostic | Non-sequential pointer advance warning |
| `crates/q-storage/src/lib.rs` | Diagnostic | Genesis integrity gap warning in write_batch_qblocks |
| `crates/q-api-server/src/state_sync_api.rs` | Diagnostic | Before/after wallet count logging in do_http_state_sync |

---

## 8. Risk Assessment

| Change | Risk | Rationale |
|--------|------|-----------|
| Bug 1 fix (always run balance sync) | LOW | Balance sync was always correct for normal nodes; fix only removes an incorrect skip |
| Q_SKIP_BALANCE_SYNC=1 new env var | LOW | Additive; default false means no behavior change for existing nodes |
| Q_GENESIS_SYNC_ONLY=1 new env var | LOW | Additive; default false means no behavior change; only activated explicitly |
| Transaction non-sequential warning | LOW | Logging only, no behavioral change |
| Genesis integrity gap warning | LOW | Logging only, no behavioral change |
| Balance before/after count logging | LOW | Two additional `RwLock::read()` acquisitions in `do_http_state_sync`; cost is negligible (runs once at startup) |

All changes are backward compatible. No database schema changes. No consensus changes.

---

*Document version: v1.0 — 2026-05-06*  
*Prepared for DeepSeek R1 external review*
