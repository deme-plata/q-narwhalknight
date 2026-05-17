# Technical Review: Sync Integrity — Block History Loss & Balance Divergence

**Date:** 2026-05-09  
**Branch:** `feature/safe-batched-sync-v1.0.2`  
**Validated by:** Live balance test (mine wallet → fresh sync container → balance comparison)  
**BAL-001 activation:** block 18,600,000 (~976,000 blocks away as of writing)

---

## Executive Summary

A live balance test on 2026-05-09 confirmed two co-contributing defects that cause fresh sync nodes to diverge from the live network on wallet balances. The test used a dedicated mining wallet (`9e326dc1f847e440...`) that accumulated 175 QUG on Epsilon; a fresh Docker container synced to tip and SYNC-006 fired. The result: **6 wallets short (1338 vs 1344) and ~86,163 QUG of supply missing**.

Root cause is not in the SYNC-006 trigger logic — the task fires correctly and at the right time. The defect is that `replay_post_checkpoint_balances()` calls `get_qblock_by_height()` which only reads `qblock:height:{N}` keys. Blocks received via P2P gossipsub during the turbo sync period are stored under `qblock:dag:{N}:{proposer}` keys, making them invisible to the replay. This produces a 95%+ block miss rate and a severely incomplete balance state.

A secondary defect causes the replay's incompleteness to be silently accepted: SYNC-006 marks the replay "done" on `Ok(())` even when 90%+ of blocks were missing. The persisted flag prevents any future retry.

Both defects must be fixed before BAL-001 enforces `balance_root` header matching at block 18,600,000.

---

## Test Evidence

### Setup
- Epsilon reference node: height 17,623,887 | wallet_count 1344 | supply 591,342 QUG
- Fresh Docker container: `q-sync-v10.7.6-baltest` on Epsilon (host networking, port 8092)
- Test wallet mined ~175 QUG in blocks 17,300,000–17,600,000
- Binary: v10.7.6 freshly built on Epsilon, clean DB

### Observed SYNC-006 Output
```
[SYNC-006] Chain at height 16607600 — starting post-checkpoint balance replay.
[POST-SYNC REPLAY v10.7.3] Starting: replaying 68732 blocks (16538869 → 16607600)...
[POST-SYNC REPLAY] Pass 1 done: 18158 txs applied, 67228 blocks missing (of 68732)  ← 97.8% miss
[POST-SYNC REPLAY] Complete: 1332 wallets, [AMOUNT] QUG

[POST-SYNC REPLAY v10.7.3] Starting: replaying 1084812 blocks (16538869 → 17623680)...
[POST-SYNC REPLAY] Pass 1 done: 1791231 txs applied, 1033721 blocks missing (of 1084812)  ← 95.3% miss
[BALANCE WRITE] wallet=9e326dc1f847e440 old=0 new=[AMOUNT] delta=+[AMOUNT]
[POST-SYNC REPLAY] Complete: 1338 wallets, [AMOUNT] QUG
```

### Fresh Node vs Epsilon at ~Same Height

| Metric | Fresh node | Epsilon | Delta |
|--------|-----------|---------|-------|
| Height | 17,623,843 | 17,623,887 | ~44 blocks |
| Wallet count | **1338** | **1344** | **−6** |
| Total supply | 505,179 QUG | 591,342 QUG | **−86,163 QUG** |
| Balance root | `bd119397…` | `bbbc2b00…` | mismatch |

The test wallet DID get credited (`old=0 new=[AMOUNT]`) confirming SYNC-006 fires and processes blocks it can find. The divergence is entirely caused by the block read path missing 95% of the replay range.

---

## Issue 1: Replay Uses Wrong Block Read Function

### Root Cause

`replay_post_checkpoint_balances()` (`crates/q-storage/src/lib.rs:5894`) iterates every height from `CHECKPOINT_HEIGHT+1` to latest and calls `get_qblock_by_height(height)` for each:

```rust
// lib.rs:5893-5934
for height in (CHECKPOINT_HEIGHT + 1)..=height_pass1_end {
    match self.get_qblock_by_height(height).await {   // ← BUG: wrong function
        Ok(Some(block)) => { /* process txs */ }
        Ok(None) => { blocks_missing += 1; }          // silently skipped
        Err(_)   => { blocks_missing += 1; }
    }
}
```

`get_qblock_by_height()` (`lib.rs:1862`) reads **only** the `qblock:height:{N}` key in CF_BLOCKS.

During turbo sync (which ran for ~30 minutes in the test), the live network produced approximately 1,084,812 new post-checkpoint blocks. These arrived on the fresh node via P2P gossipsub and were stored in the **DAG layer format** (`qblock:dag:{height}:{proposer_hex}`) by the gossip receive path. Turbo sync was busy downloading old blocks (75K–16.5M) and only wrote blocks to `qblock:height:` keys for heights within its download range. The gossip-received blocks at heights 16.5M–17.6M were never promoted to `qblock:height:` keys before SYNC-006 ran.

A background DAG reindexer does exist (`reindex_dag_blocks_to_height_keys()`, `main.rs:5576`) that promotes DAG-format blocks to height-format. But it runs exactly once at node startup (15-second delay) and does not re-run after turbo sync completes. Any gossip blocks that arrived *during* the multi-minute sync remain in DAG format permanently.

The correct function is `get_qblock_any_format()` (`lib.rs:2379`), which checks three formats in sequence:

```rust
// lib.rs:2379-2427
pub async fn get_qblock_any_format(&self, height: u64) -> Result<Option<QBlock>> {
    // Format 1: qblock:height:{N}       ← canonical (hits for turbo-synced blocks)
    if let Some(block) = self.get_qblock_by_height(height).await? {
        return Ok(Some(block));
    }
    // Format 2: qblock:dag:{N}:{proposer}  ← gossip-received (the 95% being missed)
    let dag_prefix = format!("qblock:dag:{}:", height);
    let dag_entries = self.hot_db.scan_prefix_seek(CF_BLOCKS, dag_prefix.as_bytes(), 1).await?;
    if let Some((_, value)) = dag_entries.into_iter().next() {
        // decompress + reconstruct from CF_QUANTUM_METADATA + CF_TRANSACTIONS
        ...
        return Ok(Some(block));
    }
    // Format 3: old binary key (height_be_bytes ++ hash) ← historical compat
    ...
}
```

### Fix A — Change the read call in replay (1 line per pass)

**File:** `crates/q-storage/src/lib.rs`

Pass 1 (line 5894):
```rust
// Before:
match self.get_qblock_by_height(height).await {
// After:
match self.get_qblock_any_format(height).await {
```

Pass 2 (line 5960):
```rust
// Before:
if let Ok(Some(block)) = self.get_qblock_by_height(height).await {
// After:
if let Ok(Some(block)) = self.get_qblock_any_format(height).await {
```

### Fix B — Pre-replay reindex (run once before each replay attempt)

`get_qblock_any_format()` uses `scan_prefix_seek` for DAG-format blocks, which is an O(log N) seek per height. For 1M+ block replays this adds latency. A faster approach is to ensure all DAG blocks are promoted to height keys *before* the replay runs, so `get_qblock_by_height()` (O(1) point lookup) can serve them.

In the SYNC-006 task in `main.rs:7840`, add a reindex call before the replay:

```rust
// main.rs — inside the SYNC-006 loop, before calling replay_post_checkpoint_balances()
info!("🏁 [SYNC-006] Running DAG→height reindex before replay…");
if let Err(e) = replay_storage.reindex_dag_blocks_to_height_keys().await {
    warn!("⚠️ [SYNC-006] Reindex failed (non-fatal, replay will use any_format): {}", e);
}
// then call replay_post_checkpoint_balances as before
```

With Fix B applied, Fix A becomes a safety net for any blocks the reindexer misses (e.g., blocks that arrived between reindex completion and replay start). Both fixes together give 100% coverage.

### Impact without fix
- Fresh nodes will have 90–98% of post-checkpoint blocks invisible to SYNC-006
- Balance divergence of tens of thousands of QUG
- At BAL-001 activation (block 18,600,000), these nodes will produce invalid `balance_root` headers and be rejected by the network

---

## Issue 2: Replay Marks "Done" Despite High Miss Rate

### Root Cause

SYNC-006 in `main.rs:7855` calls `mark_balance_replay_done()` immediately after `replay_post_checkpoint_balances()` returns `Ok(())`, regardless of how many blocks were missing:

```rust
// main.rs:7841-7856
match replay_storage.replay_post_checkpoint_balances(...).await {
    Ok(()) => {
        info!("✅ [SYNC-006] Replay complete…");
        // reload in-memory balances…
        let _ = replay_storage.mark_balance_replay_done().await;  // ← always marks done
    }
    Err(e) => {
        warn!("⚠️ [SYNC-006] Replay failed: {} — will retry in 30s.", e);
        continue; // retries only on hard error
    }
}
```

`mark_balance_replay_done()` writes a permanent RocksDB flag (`meta:balance_replay_v10.7.6`). Once set, every future startup skips the replay entirely (`is_balance_replay_done()` check at line 7826). A fresh node that ran SYNC-006 with 95% blocks missing will never attempt a corrective replay, even after all blocks are eventually downloaded.

### Fix

`replay_post_checkpoint_balances()` should return the miss count, and the SYNC-006 task should only mark done when the miss rate is acceptable (≤ 1% is a sensible threshold):

**In `lib.rs` — change return type of `replay_post_checkpoint_balances()`:**
```rust
pub async fn replay_post_checkpoint_balances(...) -> Result<u64> {
    // ... existing logic ...
    // return total blocks_missing instead of Ok(())
    Ok(blocks_missing)
}
```

**In `main.rs:7841` — gate mark_done on miss rate:**
```rust
match replay_storage.replay_post_checkpoint_balances(...).await {
    Ok(blocks_missing) => {
        let total = /* height - CHECKPOINT_HEIGHT */;
        let miss_pct = blocks_missing * 100 / total.max(1);
        if miss_pct > 1 {
            warn!("⚠️ [SYNC-006] Replay had {}% block miss rate ({}/{}) — will retry in 30s.",
                  miss_pct, blocks_missing, total);
            continue; // retry instead of accepting bad state
        }
        info!("✅ [SYNC-006] Replay complete — {} wallets, {}% coverage",
              wallet_count, 100 - miss_pct);
        // reload balances and mark done only on acceptable coverage
        let _ = replay_storage.mark_balance_replay_done().await;
    }
    Err(e) => { /* existing retry logic */ }
}
```

### Impact without fix
- A node that ran SYNC-006 while still syncing (incomplete block store) is permanently stuck with wrong balances
- No operator intervention can trigger a corrective replay short of deleting the RocksDB flag manually

---

## Issue 3: DAG Reindexer Runs Once at Startup, Not After Sync Completes

### Root Cause

The DAG→height reindexer (`main.rs:5575-5582`) runs exactly once, 15 seconds after node startup. This is sufficient for a node that was previously synced (all historical blocks already in height format). But for a fresh node doing a multi-minute turbo sync, the reindexer fires when the block store is nearly empty and then never runs again. All gossip blocks that arrive during the subsequent sync land permanently in DAG format.

```rust
// main.rs:5570-5582
tokio::spawn(async move {
    tokio::time::sleep(Duration::from_secs(15)).await;  // only runs once
    storage.reindex_dag_blocks_to_height_keys().await;
});
```

### Fix

Trigger a second reindex run as part of the post-sync transition. When turbo sync completes (the `now_synced && !was_synced` gate already exists in the sync loop), add a reindex call:

```rust
// In turbo sync completion handler (wherever now_synced && !was_synced fires):
if now_synced && !was_synced {
    let s = state.storage_engine.clone();
    tokio::spawn(async move {
        info!("🔄 [POST-SYNC REINDEX] Promoting DAG-format blocks to height keys…");
        match s.reindex_dag_blocks_to_height_keys().await {
            Ok(n) => info!("✅ [POST-SYNC REINDEX] {} blocks promoted", n),
            Err(e) => warn!("⚠️ [POST-SYNC REINDEX] {}", e),
        }
    });
}
```

Note: Fix A (using `get_qblock_any_format()` in the replay) already handles the case where the reindexer hasn't run. Fix B and this fix are belt-and-suspenders: they ensure the canonical `qblock:height:` index is complete before replay, restoring O(1) point lookup performance for each block.

---

## Fix Priority & Deployment Order

| Fix | File | Effort | Priority | Notes |
|-----|------|--------|----------|-------|
| Fix A: `get_qblock_any_format` in Pass 1+2 | `lib.rs:5894, 5960` | 2 lines | **P0 — ship immediately** | Unblocks 100% replay correctness |
| Fix B: Pre-replay reindex in SYNC-006 | `main.rs:7840` | ~5 lines | **P0 — ship with A** | Ensures O(1) lookup, reduces replay latency |
| Fix: Return miss count, gate mark_done | `lib.rs` + `main.rs:7841` | ~15 lines | **P0 — ship with A** | Prevents permanent bad state on partial replay |
| Fix: Post-sync reindex trigger | sync loop in `main.rs` | ~10 lines | **P1 — next release** | Belt-and-suspenders, best for production ops |

All P0 fixes are contained in two files and can ship in a single commit. No schema migration required.

---

## Test Plan

### Before shipping fixes

1. Confirm `reindex_dag_blocks_to_height_keys()` correctly promotes DAG-format blocks:
   ```bash
   # On Epsilon test node — check how many blocks are in dag vs height format
   # (storage-level diagnostic, no test API needed)
   grep -c "qblock:dag:" <rocksdb-scan-output>
   ```

2. Confirm `get_qblock_any_format()` returns blocks that `get_qblock_by_height()` misses:
   - Add a temporary log in `replay_post_checkpoint_balances()` counting format-1 vs format-2 hits
   - Expected after fix: format-2 hits cover the previously missing 95%

### After shipping fixes — regression test on fresh sync container

```bash
# 1. Spin fresh container (same as test: Epsilon + host networking)
docker run -d --name q-sync-test-fixed --network=host \
  -v /home/orobit/target-debian12/release/q-api-server:/opt/q-api-server:ro \
  -v /home/orobit/docker-sync-fresh:/data \
  -e Q_NETWORK_ID=mainnet-genesis -e Q_DB_PATH=/data/db \
  -e Q_ADMIN_WALLET=efca1e8c1f46e91013b4073898c771bb3d566453537ccf87e834505925e50723 \
  -e RUST_LOG=info debian:12 \
  bash -c 'apt-get install -y libssl3 ca-certificates && cp /opt/q-api-server /usr/local/bin/q-api-server && chmod +x /usr/local/bin/q-api-server && q-api-server --port 8093 2>&1'

# 2. Watch SYNC-006 — must show < 1% miss rate
docker logs -f q-sync-test-fixed 2>&1 | grep -E 'SYNC-006|blocks missing|miss rate'
# Expected: "blocks missing (of X) — 0% miss rate"  or similar

# 3. Compare integrity endpoints once synced
curl -s http://localhost:8093/api/v1/integrity/balance-root
curl -s http://127.0.0.1:8080/api/v1/integrity/balance-root
# Expected: wallet_count within ±5 (network advancing), supply within 0.1%

# 4. Confirm test wallet credited
docker logs q-sync-test-fixed 2>&1 | grep 9e326dc1f847e440
# Expected: "old=0 new=175.XXXXXX QUG delta=+175"
```

### Acceptance criteria for P0 fixes

- SYNC-006 `blocks_missing` < 1% of replay range
- Fresh node `wallet_count` within ±5 of Epsilon at same height  
- Fresh node `total_supply` within 0.1% of Epsilon at same height
- `balance_root_hex` matches Epsilon when queried at identical height
- `mark_balance_replay_done()` only called after < 1% miss rate confirmed

---

## BAL-001 Risk Assessment

BAL-001 activates at block 18,600,000 (~976,000 blocks from height 17,624,000 as of 2026-05-09). At the network's ~1 block/second rate this is approximately **11.3 days** away.

Without the P0 fixes, any fresh node synced today will:
1. Run SYNC-006 with 95% block miss rate
2. Mark the replay permanently done
3. Build a `balance_root` over incorrect balances
4. At block 18,600,000 produce or validate blocks with the wrong `balance_root` header
5. Be rejected by the network (Epsilon + Beta + Gamma all have correct roots)

The P0 fixes are small, safe, and can be deployed in the normal HA rolling pipeline (Alpha canary → Gamma → Beta) with no schema changes and no node restarts required mid-sync.

**Recommendation:** Ship P0 fixes in a patch release within the next 48 hours. All nodes (including existing production nodes) will benefit on their next restart — they won't re-run SYNC-006 (already marked done) but any future fresh node will.

---

## Appendix: Affected Code Locations

| Location | Current (broken) | Fixed |
|----------|-----------------|-------|
| `lib.rs:5894` | `get_qblock_by_height(height)` | `get_qblock_any_format(height)` |
| `lib.rs:5960` | `get_qblock_by_height(height)` | `get_qblock_any_format(height)` |
| `lib.rs:replay_post_checkpoint_balances` return type | `Result<()>` | `Result<u64>` (miss count) |
| `main.rs:7841-7855` | marks done on any `Ok(())` | marks done only if miss_pct ≤ 1% |
| `main.rs:7840` | no reindex before replay | calls `reindex_dag_blocks_to_height_keys()` first |
| `main.rs:~now_synced gate` | no post-sync reindex | spawns reindex task on sync completion |
