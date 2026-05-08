# Technical Review: Sync Integrity Issues — Block History Loss & Balance Divergence

**Date**: 2026-05-08  
**Version**: v10.7.0  
**Severity**: HIGH — affects every fresh node deployment; balance divergence poses BAL-001 activation risk  
**Discovered via**: Docker sync test (`q-sync-test-v10.7.0`) compared against live Epsilon archive node  

---

## Executive Summary

A sync test against a fresh Docker container revealed two independent but related integrity
failures in the turbo-sync subsystem:

| Issue | Impact | Severity |
|-------|--------|----------|
| **SYNC-001**: Fresh nodes silently discard all historical block data | Block explorer returns 404 for any block below ~16.75M | HIGH |
| **SYNC-002**: Fresh nodes accumulate incorrect balance state during sync | 62 missing wallets, 27,235 QUG supply discrepancy vs archive | HIGH |

Both issues trace to deliberate speed optimisations in `turbo_sync.rs` that were made without
adequately documenting their correctness trade-offs. They compound each other: SYNC-001 means
fresh nodes cannot serve historical block data, and SYNC-002 means the balance state they do
hold is wrong. Together they mean **a fresh-synced node cannot be used as a reliable source
of truth for either block data or wallet balances**.

BAL-001 (balance root enforcement at block 18,600,000, ~1.05M blocks away) will surface
SYNC-002 as a consensus failure if not fixed before activation.

---

## Issue 1 — SYNC-001: Warp Sync Silently Discards Historical Blocks

### Observed Symptoms

- Fresh node reports `current_height = 17,544,086`
- `/api/v1/blocks/1000`, `/api/v1/blocks/100000`, `/api/v1/blocks/5000000` all return `404 Not Found`
- DB size: **19 GB** (180 SST files) vs live archive node's **262 GB** (88,885 SST files)
- Only the most recent ~1.75M blocks (heights ~15.8M–17.5M) exist in the database

### Root Cause: `probe_network_gap()` Overrides Sync Start Height

**File**: `crates/q-storage/src/turbo_sync.rs` ~lines 6250–6280

On every fresh boot (when `local_height < 100`), turbo sync calls `probe_network_gap()`.
This function performs a binary search across peers to find the lowest block height any peer
can serve. Because peers only retain the most recent ~1.75M blocks (their own warp-sync
window), `probe_network_gap()` returns approximately `~15,800,000` as the "gap floor".

The code then sets:

```rust
effective_start_height = gap_floor;  // ~15,800,000
```

Blocks 1 through `gap_floor - 1` are never requested. They are not available on any peer
and are simply skipped. The entire pre-checkpoint block history is lost on every fresh node
deployment.

**File**: `crates/q-api-server/src/main.rs` ~lines 6642–6656

Simultaneously, `current_height_atomic` is updated to the MAX block height in any batch
received — not the contiguous stored height:

```rust
let max_height = blocks.iter().map(|b| b.header.height).max().unwrap_or(0);
if max_height > current_atomic {
    current_height_atomic_clone.store(max_height, Ordering::Release);
    info!("📈 [LIBP2P SYNC] Updated current_height_atomic: {} → {}", ...);
}
```

This creates a misleading picture: the node announces height 17.544M but has no data below
~15.8M. Explorer queries for any height in that gap return 404.

### Why `Q_SKIP_CHECKPOINT=1` Does Not Help

`Q_SKIP_CHECKPOINT=1` correctly bypasses `probe_network_gap()` — the call is gated by
`!skip_checkpoint && local_height < 100 && effective_start_height == 0`, so when
`skip_checkpoint = true` the probe is skipped and `effective_start_height` remains `0`,
which normalises to height `1` at line 6398. The flag works as designed.

The real reason a fresh node still ends up without historical blocks is that **no peer on
the network has them**. Every other node also warp-synced and only holds the most recent
~1.75M blocks. Epsilon's 262 GB archive is the only node retaining the full chain. With
`Q_SKIP_CHECKPOINT=1`, the node correctly tries to sync from height 1, receives nothing from
peers for heights 1–15,799,999 (none available), and eventually catches up from whatever
the lowest available peer height is (~15.8M). The height counter jumps to 15.8M when the
first available peer blocks arrive, not because of a checkpoint skip.

**The fix is not to patch `Q_SKIP_CHECKPOINT` — it is correct. The fix is to make Epsilon
the authoritative source for historical block data**, either via the archive proxy fallback
or by ensuring a full-archive node is always reachable on the network.

### Block Retrieval Has No Archive Fallback

**File**: `crates/q-storage/src/lib.rs` ~line 2379 (`get_qblock_any_format`)

All block lookups search only `hot_db` in `CF_BLOCKS`. Three key formats are tried
sequentially (`qblock:height:{N}`, `qblock:dag:{N}:{proposer}`, legacy binary key). If none
match, `Ok(None)` is returned — the API handler converts this to `StatusCode::NOT_FOUND`.
There is no fallback to a remote archive node.

The same applies to P2P block serving: `create_block_pack()` calls `get_qblocks_range()`,
which returns an empty `Vec` for missing ranges, causing the block-pack request to fail with
"No blocks found in range X-Y". A fresh-synced node cannot help peers sync historical data.

### Impact

- **Block explorer**: Non-functional for all heights below the warp-sync floor (~15.8M)
- **P2P**: Fresh nodes cannot contribute block history to new peers; they are sync consumers only
- **Trust**: A node claiming height 17.5M but unable to answer questions about height 1M is misleading
- **BAL-001 adjacency**: If a fresh node is elected as a validator, it cannot serve the block
  history needed for balance root verification by auditing tools

### Fix Design

#### Short-Term Fix (1–2 days): Archive Proxy Fallback

Add `Q_ARCHIVE_NODE_URL` environment variable. When `get_qblock_any_format()` returns `None`
locally, transparently proxy the request to the archive node via HTTP:

```rust
// crates/q-api-server/src/handlers.rs — get_block_by_height()
match state.storage_engine.get_qblock_any_format(height).await {
    Ok(Some(block)) => return Ok(Json(ApiResponse::success(block))),
    Ok(None) => {
        // Proxy to archive node if configured
        if let Some(archive_url) = &state.archive_node_url {
            let url = format!("{}/api/v1/blocks/{}", archive_url, height);
            if let Ok(block) = fetch_from_archive(&url).await {
                return Ok(Json(ApiResponse::success(block)));
            }
        }
        return Err(StatusCode::NOT_FOUND);
    }
    Err(_) => return Err(StatusCode::INTERNAL_SERVER_ERROR),
}
```

This makes block explorer queries work transparently on all nodes without requiring 262 GB
of local storage. Configure all non-archive nodes with
`Q_ARCHIVE_NODE_URL=http://89.149.241.126:8080` in their `.env`.

**Risk**: Adds external HTTP dependency to block reads. Mitigate with a short timeout (3s)
and treat archive failure as a cache miss, not an error.

**Note on `Q_SKIP_CHECKPOINT`**: the flag is correctly implemented and does not need changing.
The probe is properly gated on `!skip_checkpoint`. The reason nodes still lack historical
blocks with the flag set is that no peer has them — peers only retain the most recent ~1.75M
blocks. `Q_SKIP_CHECKPOINT=1` makes the node try from height 1, but gets nothing back from
peers below ~15.8M. The archive proxy is the correct fix.

#### Medium-Term Fix (1–2 weeks): Node Type Declaration + Honest Height Reporting

1. **Report contiguous height, not MAX received** in `current_height_atomic`:
   Read `qblock:latest` (the contiguous stored height) after each batch commit and use that
   for the atomic. A node that claims height 17.5M but cannot answer queries below 15.8M is
   dishonest and misleads both users and peers.

2. **Add `node_type` to P2P peer-height announcements**: `"light"` (warp-synced, no history)
   vs `"archive"` (full chain). Light nodes should not be asked for historical block ranges.
   Do not proxy archive responses into `create_block_pack()` without explicit opt-in — it
   adds latency, failure coupling, and creates an implicit archive relay without the storage.

---

## Issue 2 — SYNC-002: Balance State Diverges During Turbo Sync

### Observed Symptoms

At essentially the same block height (49-block difference):

| Metric | Fresh sync node | Live archive node | Delta |
|--------|----------------|-------------------|-------|
| Wallet count | 1,279 | 1,341 | −62 wallets |
| Total supply | 607,302 QUG | 580,067 QUG | +27,235 QUG |
| Balance root | `8acdf4cb...` | `9647d692...` | mismatch |

The fresh node has **fewer wallets** but **more QUG** — impossible under a correct replay.

### Root Cause: Three Co-Contributing Bugs

#### Bug A — Transfer Transactions Silently Skipped During Fast Sync

**File**: `crates/q-storage/src/turbo_sync.rs` ~lines 4367–4401

```rust
let extreme_skip_balances_threshold: u64 = 5_000;  // hardcoded default
let blocks_behind = network_height - current_height;
let skip_balances = blocks_behind > extreme_skip_balances_threshold;

if skip_balances {
    engine.process_block_coinbase_only_tx(&tx, block).await  // ← rewards only
} else {
    engine.process_block_mining_rewards_tx(&tx, block).await  // ← full processing
}
```

Any fresh node that is more than 5,000 blocks behind (which is every fresh node ever) enters
`coinbase_only` mode for the **entire** sync. This means transfer transactions — wallet-to-wallet
sends — are never applied to the balance DB during turbo sync.

**This is the smoking gun for the 62 missing wallets.** Each of those 62 addresses received
QUG only via transfers from other wallets, never via a mining coinbase reward. Since transfers
were skipped, those wallets were never created in the fresh node's balance DB.

**This is also the root cause of the 27K QUG surplus.** Mining rewards (coinbase) were
applied: wallets received +QUG. But the corresponding transfer debits (−QUG from sender
wallets) were not applied. Net result: the system gained QUG that was never balanced by
deductions. This is effectively synthetic inflation on the fresh node.

#### Bug B — `total_minted_supply` Never Persisted to RocksDB

**File**: `crates/q-storage/src/turbo_sync.rs` ~lines 4403–4465

Turbo sync commits wallet balance rows to RocksDB after each batch. However, it never
persists `total_minted_supply` to a durable key. The supply counter lives only in the
in-memory `Arc<RwLock<u128>>`. On node restart, it resets to 0 and is recomputed from
the in-memory wallet map — which is already wrong due to Bug A.

#### Bug C — Supply Not Recomputed After Sync Completes

**File**: `crates/q-api-server/src/main.rs` ~lines 20898–20950

When the sync-complete transition fires (`now_synced && !was_synced`), the code correctly
reloads `wallet_balances` from RocksDB:

```rust
let persisted = app_state.storage_engine.load_wallet_balances().await?;
*app_state.wallet_balances.write().await = persisted;
// ← total_minted_supply is NOT recomputed here
```

`total_minted_supply` keeps its stale turbo-sync value. Even if the wallet balance map were
correct (which it isn't due to Bug A), the supply figure would not reflect it.

### Impact

- **Wallet balances are wrong on every fresh-synced node** — 62+ wallets missing, supply
  inflated by tens of thousands of QUG
- **The `balance_root` on a fresh node will never match the archive node** — nodes will
  diverge in consensus
- **BAL-001 activation risk (HIGH)**: At block 18,600,000 (~1.05M blocks away), producers
  must embed the balance root in every block and validators check it. If fresh nodes have a
  different balance DB from the archive node, their produced blocks will be rejected by
  validators running the correct state. This will partition the network.
- **The integrity API `wallet_count` and `total_supply_display` fields are unreliable** on
  any node that was not running continuously from genesis

### Fix Design

#### Fix A (Critical): Remove the Balance-Skip Threshold

**File**: `crates/q-storage/src/turbo_sync.rs` ~line 4367

Delete the `extreme_skip_balances_threshold` entirely. Always call the full transaction
processor:

```rust
// BEFORE (broken):
let skip_balances = blocks_behind > extreme_skip_balances_threshold;
if skip_balances {
    engine.process_block_coinbase_only_tx(&tx, block).await
} else {
    engine.process_block_mining_rewards_tx(&tx, block).await
}

// AFTER (correct):
engine.process_block_mining_rewards_tx(&tx, block).await
```

The sync speed impact is measurable but acceptable — the warp-sync tested at ~47K
blocks/sec even with full processing for the recent portion. Historical blocks with no
transfers will process equally fast either way.

#### Fix B (Critical): Persist `total_minted_supply` After Each Batch

**File**: `crates/q-storage/src/lib.rs` — add two functions:

```rust
pub async fn save_total_minted_supply(&self, supply: u128) -> Result<()> {
    let value = supply.to_be_bytes();
    self.hot_db.put(CF_MANIFEST, b"meta:total_minted_supply", &value)?;
    Ok(())
}

pub async fn load_total_minted_supply(&self) -> Result<u128> {
    match self.hot_db.get(CF_MANIFEST, b"meta:total_minted_supply")? {
        Some(bytes) if bytes.len() == 16 => {
            Ok(u128::from_be_bytes(bytes.try_into().unwrap()))
        }
        _ => Ok(0),
    }
}
```

Call `save_total_minted_supply()` in turbo sync after each batch commit, passing the current
running sum of all wallet balances.

#### Fix C: Recompute Supply After Sync Completes

**File**: `crates/q-api-server/src/main.rs` ~line 20930

```rust
if now_synced && !was_synced {
    // Existing wallet balance reload
    let persisted = storage.load_wallet_balances().await?;
    let total: u128 = persisted.values().sum();
    *app_state.wallet_balances.write().await = persisted;
    
    // Fix C: recompute supply from the reloaded balances
    *app_state.total_minted_supply.write().await = total;
    info!("🔄 [SYNC COMPLETE] Recomputed total_minted_supply: {} base units", total);
}
```

Additionally, on startup, load the persisted supply from RocksDB (Fix B) as the initial value
rather than starting from 0.

---

## BAL-001 Activation Risk Assessment

BAL-001 activates at block **18,600,000** (~1,055,000 blocks from now at ~1 block/sec = ~12.2 days).

| Scenario | Risk if SYNC-002 unfixed |
|----------|--------------------------|
| Fresh node produces blocks | Its blocks embed the wrong `balance_root`; all other validators reject them |
| Fresh node validates blocks | It incorrectly rejects valid blocks from archive nodes (different balance root) |
| Network split | Any node with a diverged balance state will fork off from the canonical chain |

**The 62-wallet discrepancy on a single test node represents a worst-case delta from the
archive state. On mainnet, this delta will grow** — every day that mining continues without
the fix, more transfer-only wallets accumulate on archive nodes that fresh nodes will never
have. By block 18.6M, the gap may be hundreds of wallets and hundreds of thousands of QUG.

**Recommendation**: Fix SYNC-002 before block 18,400,000 (~2.3 days of buffer before
BAL-001 activation). Deploy and run a fresh sync test to verify balance root convergence
before the activation height.

---

## Recommended Fix Priority and Deployment Order

| Priority | Fix | File(s) | Effort | Deploy by |
|----------|-----|---------|--------|-----------|
| 🔴 P0 | SYNC-002 Bug A: remove transfer skip | `turbo_sync.rs` ~4367 | 30 min | Immediately |
| 🔴 P0 | SYNC-002 Bug C: recompute supply post-sync | `main.rs` ~20930 | 30 min | Immediately |
| 🟠 P1 | SYNC-002 Bug B: persist supply to RocksDB | `lib.rs`, `turbo_sync.rs` | 2 hours | Before BAL-001 |
| 🟠 P1 | SYNC-001 Short-term: archive proxy fallback | `handlers.rs`, `main.rs` | 4 hours | This week |
| 🟡 P2 | SYNC-001 Medium-term: honest height reporting | `main.rs`, P2P layer | 1 day | Next sprint |
| 🟡 P2 | Add `node_type` to P2P announcements | P2P layer | 2 days | Next sprint |

---

## Test Plan

### After Implementing P0 Fixes

1. Stop the existing `q-sync-test-v10.7.0` container
2. Delete its DB: `rm -rf /home/orobit/docker-sync-test-v10.7.0/`
3. Build a new binary with the fixes (bump version to v10.7.1)
4. Spin a fresh container with the same parameters
5. Let it sync to tip (expected: ~6–10 min to warp-sync floor, then gradual catch-up)
6. Run comparison:

```bash
# Balance root must match (allow ±10 block height difference)
curl -sf http://localhost:8086/api/v1/integrity/balance-root
curl -sf http://localhost:8080/api/v1/integrity/balance-root

# Wallet count must be within ±5 of live node
# total_supply must be within 0.1% of live node

# Supply health must be true on both (already fixed in v10.7.0)
curl -sf http://localhost:8086/api/v1/integrity/full | python3 -c '
import sys,json; d=json.load(sys.stdin)["data"]
print("supply_healthy:", d["supply_healthy"])
print("wallet_count:", d["wallet_count"])
print("all_healthy:", d["all_healthy"])
'
```

### After Implementing P1 Fix (Archive Proxy)

```bash
# On a fresh node with Q_ARCHIVE_NODE_URL=http://89.149.241.126:8080:
curl http://localhost:8086/api/v1/blocks/1000       # must return block data
curl http://localhost:8086/api/v1/blocks/100000     # must return block data
curl http://localhost:8086/api/v1/blocks/5000000    # must return block data
curl http://localhost:8086/api/v1/blocks/16000000   # must return block data
```

All four must return valid block JSON (not 404).

### Regression Tests to Add

```
crates/q-storage/tests/sync_balance_determinism_tests.rs
  - test_fresh_sync_wallet_count_matches_archive()
  - test_fresh_sync_supply_matches_archive()
  - test_transfer_wallets_present_after_sync()
  - test_total_supply_persisted_across_restart()

  # Minimal deterministic transfer correctness test (most important):
  # Block 1: coinbase A +100
  # Block 2: transfer A → B 40
  # Expected: A=60, B=40, supply=100, wallet_count=2
  # Must produce identical result from both normal processing and turbo sync.
  - test_turbo_sync_applies_transfer_debits_and_credits()
  - test_coinbase_only_processor_not_used_during_fresh_sync()
  - test_balance_root_stable_across_restart()
  - test_current_height_reports_contiguous_height_not_max_seen()
  - test_missing_historical_block_returns_archive_proxy_source_when_configured()
  - test_light_node_does_not_advertise_archive_capability()
```

### Hard Invariant: No Balance Skipping

Now that the balance-skip variables are removed, add a compile-time guard against
re-introduction. Place in `turbo_sync.rs` as a module-level comment or assertion:

```rust
// INVARIANT (v10.7.1+): Balance skipping during sync is FORBIDDEN.
// Wallet state is consensus-critical (BAL-001, block 18,600,000).
// Any optimization that skips transfer debits/credits creates a different
// state machine from archive replay. Do not re-introduce Q_EXTREME_SKIP_BALANCES.
// If this needs revisiting, balance roots must be verified identical post-sync first.
```

---

## Appendix: Key File Reference

| File | Line Range | Topic |
|------|-----------|-------|
| `crates/q-storage/src/turbo_sync.rs` | 6250–6280 | Checkpoint probe overrides `effective_start_height` |
| `crates/q-storage/src/turbo_sync.rs` | 4367–4401 | Balance skip threshold (Bug A) |
| `crates/q-storage/src/turbo_sync.rs` | 4403–4465 | Missing `save_total_minted_supply` (Bug B) |
| `crates/q-api-server/src/main.rs` | 6642–6656 | `current_height_atomic` set to MAX, not contiguous |
| `crates/q-api-server/src/main.rs` | 20898–20950 | Post-sync reload missing supply recompute (Bug C) |
| `crates/q-storage/src/lib.rs` | 2379–2510 | `get_qblock_any_format` — no archive fallback |
| `crates/q-storage/src/lib.rs` | 2011–2150 | `get_qblocks_range` — returns empty Vec silently |
| `crates/q-api-server/src/handlers.rs` | 735–758 | Block handler — 404 on miss, no proxy |
| `crates/q-api-server/src/integrity_api.rs` | 186, 419 | `supply_healthy` check (already fixed in v10.7.0) |
