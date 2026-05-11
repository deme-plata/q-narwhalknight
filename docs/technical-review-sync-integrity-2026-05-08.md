# Technical Review: Sync Integrity — Block History Loss, Balance Divergence, and BAL-001 Enforcement Risk

**Date**: 2026-05-08  
**Version**: v10.7.1  
**Severity**: CRITICAL — balance divergence on checkpoint nodes poses imminent network fork risk at BAL-001 enforcement  
**Discovered via**: Docker sync test against a fresh container, compared to live Epsilon archive node  
**Author**: Server Alpha (automated technical review)

---

## Executive Summary

A sync test against a fresh Docker container exposed two independent integrity failures in the turbo-sync subsystem, plus a replay-gating defect that prevents existing nodes from self-correcting:

- **SYNC-001 (Block History Loss)**: Fresh nodes that bootstrap from the warp-sync checkpoint acquire only blocks above ~16,750,000. Blocks 1–16,749,999 are never downloaded. The node announces height ~17.5M but cannot answer any explorer or P2P request for pre-checkpoint data. *Short-term fix: archive proxy via `Q_ARCHIVE_NODE_URL`; no code exists yet.*
- **SYNC-002 (Balance Divergence)**: Three co-contributing bugs caused fresh nodes to accumulate incorrect wallet state during sync — skipped transfer transactions, lost supply counter, and missing post-sync recomputation. *All three were fixed in v10.7.1.*
- **Replay Gating Defect**: The SYNC-006 replay mechanism that was supposed to correct pre-v10.7.1 nodes (Beta, Gamma, Delta) is being incorrectly skipped due to a genesis-detection false positive. *Fix in progress: admin reset endpoint + corrected detection logic.*
- **BAL-001 Enforcement Timeline**: Balance root enforcement activates at block **20,000,000** (~245,000 blocks from now, approximately 2.8 days at 1 bps). Checkpoint nodes with diverged balances will compute wrong `state_root` values and their blocks will be rejected by Epsilon. The network will fork unless balance state is corrected before that height.

All issues must be resolved before block 20,000,000. The replay path exists; the blocker is ensuring it actually runs.

---

## Issue 1: Block History Loss After Warp-Sync (SYNC-001)

### Root Cause

**File**: `crates/q-storage/src/turbo_sync.rs` (checkpoint probe + `effective_start_height` override)

On every fresh boot where `local_height < 100`, `turbo_sync.rs` calls `probe_network_gap()`. This function binary-searches peers to find the lowest block height any peer can serve. Because all production nodes bootstrapped from the same checkpoint snapshot, the lowest available height across the peer set is approximately **16,750,000** — the checkpoint height. `probe_network_gap()` returns this as the gap floor, and the code then sets:

```rust
effective_start_height = gap_floor;  // ~16,750,000
```

Blocks 1 through 16,749,999 are never requested. No peer holds them (Epsilon's 219 GB archive is the only full-history node), and the sync simply skips the entire pre-checkpoint chain.

**File**: `crates/q-api-server/src/main.rs` (current_height_atomic)

Compounding the deception, `current_height_atomic` is updated to the MAX block height seen in any received batch, not the lowest contiguous stored height:

```rust
let max_height = blocks.iter().map(|b| b.header.height).max().unwrap_or(0);
if max_height > current_atomic {
    current_height_atomic_clone.store(max_height, Ordering::Release);
}
```

The result: a node that downloaded only blocks 16,750,000–17,500,000 announces itself at height 17,500,000. Any explorer request for a block below 16.75M returns `404 Not Found`. Any P2P request for historical block ranges yields an empty response. The node is a consumer, not a contributor, of block data.

### Impact

- Block explorer is non-functional for all heights below the checkpoint (~16.75M). This covers approximately 98.5% of chain history by block count.
- Fresh nodes cannot serve historical block ranges to new peers via `create_block_pack()`. They propagate the problem: every new node that syncs from another warp-synced node also lacks history.
- The misleading height announcement erodes trust in height metrics across the network.
- If a warp-synced node is used for balance auditing against the full chain, results will be incomplete.

### Fix Design

**Short-term (1–2 days): Archive proxy via `Q_ARCHIVE_NODE_URL`**

No code for this exists yet. When `get_qblock_any_format()` in `crates/q-storage/src/lib.rs` returns `None` locally, the API handler in `crates/q-api-server/src/handlers.rs` should transparently proxy the request to the archive node:

```rust
// handlers.rs — get_block_by_height
match state.storage.get_qblock_any_format(height).await {
    Ok(Some(block)) => return Ok(Json(block)),
    Ok(None) => {
        if let Some(archive_url) = &state.config.archive_node_url {
            // proxy to Epsilon, 3s timeout
            if let Ok(block) = fetch_from_archive(archive_url, height).await {
                return Ok(Json(block));
            }
        }
        return Err(StatusCode::NOT_FOUND);
    }
    Err(e) => return Err(StatusCode::INTERNAL_SERVER_ERROR),
}
```

Configure non-archive nodes: `Q_ARCHIVE_NODE_URL=http://89.149.241.126:8080` in `.env`.

**Medium-term (1–2 weeks):**

1. Report contiguous stored height in `current_height_atomic`, not the maximum received. Read from `qblock:latest` after each batch commit.
2. Add a `node_type` field (`"light"` vs `"archive"`) to P2P peer-height announcements so peers know not to request historical ranges from warp-synced nodes.

### Status

SHORT-TERM FIX NEEDED. The `Q_ARCHIVE_NODE_URL` proxy has not been implemented. The height-reporting issue is also unaddressed. No deployment blocker for BAL-001, but should be tracked as P1 work.

---

## Issue 2: Balance Divergence on Fresh-Synced Nodes (SYNC-002)

Three co-contributing bugs caused fresh nodes to accumulate incorrect wallet state during turbo sync. All three are fixed in v10.7.1, but nodes that synced before v10.7.1 (Beta, Gamma, Delta) still hold stale data and require a replay run to converge.

### Bug A — Transfer Transactions Silently Skipped During Fast Sync (FIXED in v10.7.1)

**File**: `crates/q-storage/src/turbo_sync.rs` ~line 4389 (documented in comment)

When the node was more than 5,000 blocks behind the network, turbo sync called only `process_block_coinbase_only_tx()` — applying mining reward credits but skipping all wallet-to-wallet transfer transactions. Since every fresh node is always more than 5,000 blocks behind at sync start, this condition applied for the entire sync on every fresh deployment.

The consequences were severe: any wallet that received QUG exclusively via transfers (never via a coinbase reward) was never created in the local balance DB. Mining rewards were credited to source wallets but the corresponding debits for transfers out of those wallets were never applied, creating net synthetic inflation.

**Fix**: v10.7.1 removes the skip threshold entirely. All transactions are always processed regardless of how far behind the node is. The comment at `turbo_sync.rs:4389` documents this change and prohibits re-introduction of any balance-skip optimisation.

### Bug B — `total_minted_supply` Not Persisted to RocksDB (FIXED in v10.7.1)

**File**: `crates/q-storage/src/turbo_sync.rs` (batch commit path)

The running total of minted supply was maintained only as an in-memory counter during turbo sync. It was never written to RocksDB. On node restart, the counter reset to zero and was recomputed from the in-memory wallet map — which was already incorrect due to Bug A. The supply figure was therefore doubly wrong: corrupted at accumulation time and lost at restart.

**Fix**: v10.7.1 calls `save_total_supply()` after every batch commit in turbo sync, persisting the supply counter to a durable RocksDB key.

### Bug C — Supply Not Recomputed After Sync Completes (FIXED)

**File**: `crates/q-api-server/src/main.rs` (sync-complete handler)

The original `now_synced && !was_synced` transition trigger was unreliable — it depended on a state edge that could be missed. Even when it fired, it reloaded wallet balances from RocksDB but did not recompute `total_minted_supply` from the reloaded map.

**Fix**: The edge-triggered handler was replaced with a SYNC-006 polling task in `main.rs`. This task runs on 30-second cycles, waits until the chain height exceeds the checkpoint height, and then calls `replay_post_checkpoint_balances()` to bring the balance state to a consistent and complete post-checkpoint view.

### Remaining Issue: Replay Gating Incorrectly Skips Pre-v10.7.1 Nodes

Nodes that synced before v10.7.1 (Beta, Gamma, Delta) still hold stale balance state from when Bugs A and B were active. The SYNC-006 replay task (`replay_post_checkpoint_balances`) exists to repair them, but it is being incorrectly bypassed:

**Problem 1 — `meta:balance_replay_v10.7.8` flag**: RocksDB stores a flag after the first replay run. If that run executed against corrupted data (which it did, before v10.7.1), the flag prevents a corrective second run. The node considers itself already replayed.

**Problem 2 — Genesis detection false positive**: The replay is gated by a check that calls `get_qblock_any_format(1_000_000)`. The intent is to detect genesis nodes (which ran from block 1 and don't need replay). However, checkpoint nodes that received blocks below the checkpoint height via P2P gossip also pass this check — they have block 1,000,000 in their DB even though they bootstrapped from the checkpoint snapshot. The check incorrectly classifies them as genesis nodes and skips replay.

**Fix in progress**:

1. Admin endpoint `POST /api/v1/admin/reset-balance-replay` — clears the `meta:balance_replay_v10.7.8` flag from RocksDB, allowing SYNC-006 to run a fresh replay on next cycle.

2. Replace `get_qblock_any_format(1_000_000)` with `is_checkpoint_applied()` flag — a boolean persisted in RocksDB when the checkpoint snapshot is loaded. This correctly identifies checkpoint nodes regardless of what blocks they later received via gossip. Genesis nodes (Epsilon) never set this flag.

Until these fixes are deployed and the replay runs successfully on Beta, Gamma, and Delta, those nodes hold balance state that diverges from Epsilon's authoritative DB.

### Status

Bug A, B, C — FIXED in v10.7.1. Replay gating defect — FIX IN PROGRESS. Blocking BAL-001 enforcement.

---

## BAL-001 Enforcement Risk

Balance root v1 (`BAL-001`) entered shadow mode at block **17,742,000**. As of the discovery of these issues, the current network height is approximately **17,756,000** — the network is 14,000 blocks into shadow mode.

**Shadow mode behaviour**: When a gossiped block carries a `state_root` that does not match the local computation, the mismatch is logged but the block is accepted. Divergence is visible in logs but does not affect consensus.

**Enforcement behaviour** (activates at block **20,000,000**, approximately 244,000 blocks and ~2.8 days from now at 1 bps): Blocks with a missing or incorrect `state_root` are outright rejected. A node with a diverged balance DB will compute a different `state_root` than Epsilon. Every block that node produces will be rejected by Epsilon-adjacent validators.

### What Breaks If Unfixed

If checkpoint nodes (Beta, Gamma, Delta) still have diverged balance state at block 20,000,000:

- Their produced blocks embed the wrong `state_root` and are rejected by Epsilon.
- They reject Epsilon's blocks as having the wrong `state_root` from their perspective.
- The network splits: Epsilon (genesis, authoritative) continues on the canonical chain; checkpoint nodes fork off.
- Block production on the fork is sustained until enough validators reject its blocks, at which point it stalls.
- Users connected to Beta or Gamma see their balance state frozen or rolled back when their client reconnects to the canonical chain.

This is a live network fork risk with a hard deadline 2.8 days out.

### Mitigation

1. **Deploy the admin reset endpoint** to Beta, Gamma, and Delta.
2. **Call `POST /api/v1/admin/reset-balance-replay`** on each node to clear the stale replay flag.
3. **Deploy the corrected genesis detection** (use `is_checkpoint_applied()` instead of block lookup).
4. **Verify SYNC-006 replay runs to completion** on each node — check logs for `replay_post_checkpoint_balances` completion message.
5. **Compare balance roots** across all nodes via the integrity API before block 20,000,000.

---

## Fix Priority and Deployment Order

The following order minimises risk and respects the BAL-001 deadline:

1. **(IMMEDIATE — P0)** Deploy v10.7.1 binary to Beta, Gamma, Delta. The three core balance bugs (A, B, C) are fixed in this version. Without this, any fresh-synced replacement node would continue diverging.

2. **(IMMEDIATE — P0)** Deploy the corrected genesis detection (`is_checkpoint_applied()` replacing block-1M lookup). This ensures SYNC-006 does not skip replay on checkpoint nodes.

3. **(IMMEDIATE — P0)** Deploy the admin reset endpoint (`POST /api/v1/admin/reset-balance-replay`). Call it on Beta, Gamma, and Delta to clear the stale replay flag and allow SYNC-006 to rerun.

4. **(WITHIN 24 HOURS — P0)** Confirm SYNC-006 replay completed on all three nodes. Compare balance roots against Epsilon. All four nodes must agree on `state_root` by block 20,000,000.

5. **(THIS WEEK — P1)** Implement the `Q_ARCHIVE_NODE_URL` archive proxy fallback for block lookups. Needed for explorer functionality but not a BAL-001 blocker.

6. **(NEXT SPRINT — P2)** Honest height reporting (contiguous stored height, not MAX received). Node type annotations in P2P peer announcements.

---

## Test Plan

### Verifying Balance Convergence on Existing Nodes (Steps 1–4)

After deploying the corrected genesis detection and triggering a fresh replay on each checkpoint node:

```bash
# On each checkpoint node (Beta port 8080, Gamma port 8808, Delta port 8080):
curl -sf http://<host>:<port>/api/v1/integrity/balance-root | python3 -m json.tool

# Compare against Epsilon (authoritative):
curl -sf http://89.149.241.126:8080/api/v1/integrity/balance-root | python3 -m json.tool
```

All nodes must return the same `balance_root` hash (allowing for a ±10 block height difference due to live block production during the comparison window).

Additionally verify:

```bash
# Wallet count within ±5 across all nodes
# Total supply within 0.01% across all nodes
# supply_healthy = true on all nodes
curl -sf http://<host>:<port>/api/v1/integrity/full | python3 -c '
import sys, json
d = json.load(sys.stdin)["data"]
print("wallet_count:", d["wallet_count"])
print("total_supply_display:", d.get("total_supply_display"))
print("supply_healthy:", d["supply_healthy"])
print("all_healthy:", d["all_healthy"])
'
```

### Verifying Fix on a Fresh Sync (Regression Test)

After all fixes are deployed:

1. Spin a fresh Docker container with a v10.7.1+ binary and an empty database.
2. Let it sync to chain tip (expected ~6 hours from genesis sync, or warp-sync to checkpoint in ~30 minutes).
3. Compare balance root and wallet count against Epsilon.
4. The balance root must match Epsilon's within a 10-block window. Wallet count must be within ±5.
5. Verify no `coinbase_only` log lines appear during sync (confirming the skip threshold is removed).

### Verifying Archive Proxy (After P1 Fix)

On a fresh warp-synced node with `Q_ARCHIVE_NODE_URL=http://89.149.241.126:8080`:

```bash
curl http://localhost:8080/api/v1/blocks/1000        # must return block JSON (proxied from Epsilon)
curl http://localhost:8080/api/v1/blocks/5000000     # must return block JSON
curl http://localhost:8080/api/v1/blocks/16000000    # must return block JSON
curl http://localhost:8080/api/v1/blocks/16750000    # must return block JSON (at checkpoint boundary)
curl http://localhost:8080/api/v1/blocks/17000000    # must return block JSON (locally stored)
```

All five requests must return valid block data. The first four proxy through Epsilon; the last is served locally.

### Regression Test Suite to Add

```
crates/q-storage/tests/sync_balance_integrity_v2_tests.rs

  test_fresh_sync_applies_all_transfer_transactions()
  test_fresh_sync_wallet_count_matches_genesis_node()
  test_fresh_sync_supply_matches_genesis_node()
  test_total_minted_supply_survives_restart()
  test_balance_root_converges_between_checkpoint_and_genesis_node()
  test_replay_gating_uses_is_checkpoint_applied_not_block_lookup()
  test_replay_flag_reset_allows_second_replay()
  test_coinbase_only_processor_never_used_during_fresh_sync()
```

---

## Appendix: Key File Reference

| File | Topic |
|------|-------|
| `crates/q-storage/src/turbo_sync.rs` ~line 4389 | Bug A fix — skip threshold removal; comment documents invariant |
| `crates/q-storage/src/turbo_sync.rs` (batch commit) | Bug B fix — `save_total_supply()` call after each commit |
| `crates/q-storage/src/turbo_sync.rs` ~line 6250 | `probe_network_gap()` — sets `effective_start_height` to checkpoint floor |
| `crates/q-api-server/src/main.rs` ~line 6642 | `current_height_atomic` set to MAX received (honest reporting not yet fixed) |
| `crates/q-api-server/src/main.rs` (SYNC-006 task) | Polling replay task replacing edge-triggered sync-complete handler |
| `crates/q-storage/src/lib.rs` `get_qblock_any_format()` | Block lookup — no archive fallback exists yet |
| `crates/q-api-server/src/handlers.rs` (block endpoint) | Returns 404 on miss — archive proxy not yet implemented |
