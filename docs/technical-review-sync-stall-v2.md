# Technical Review v2: Epsilon Sync Stall — Root Cause & Safe Fix

**Date:** 2026-04-06  
**Severity:** Critical (mainnet, $920M cap)  
**Prepared for:** DeepSeek peer review  
**Status:** Corruption fixed, sync stalled due to 3 compounding bugs  

## 1. Current State

| Server | Height | Network Tip | Gap | Status |
|--------|--------|-------------|-----|--------|
| Epsilon | 13,475,447 | 13,558,759 | 83K | Stuck — sync stalled |
| Beta | 13,344,443 | 13,558,759 | 214K | Stuck — same bugs |
| Delta | 13,558,759 | 13,558,759 | 0 | Producing blocks |
| Gamma | Initializing | — | — | Just restarted |

The corruption from kill -9 is fixed (710 orphaned blocks deleted, pointers reset).
The sync stall is caused by 3 separate code bugs that compound into a deadlock.

## 2. The Three Bugs

### Bug 1: HEIGHT CLAMP Poisons Sync Target (PRIMARY)

**File:** `crates/q-api-server/src/main.rs:14528-14538`

```rust
let hard_cap = local_height + 10_000;
if network_height > hard_cap && local_height > 100_000 {
    // Clamps highest_network_height atomic to local + 10K
    app_state.highest_network_height.store(hard_cap, SeqCst);
}
```

**Problem:** This clamp runs every 15 seconds and overwrites the `highest_network_height` 
atomic. The sync loop at line 19394 then reads this CLAMPED value:

```rust
let sync_target = network_height.min(current_height + 10000);
// network_height is already clamped to current + 10K
// so sync_target = current + 10K = only 10K blocks per batch
```

After syncing the first 10K blocks (13,475,448 to 13,485,447), the node compares:
- new_height (13,485,447) vs network_height (13,485,447 — clamped)
- Thinks it's caught up -> enters cooldown
- On next iteration, clamp recalculates but sync attempts timeout

**This is by design for anti-poisoning** but it's preventing recovery from large gaps.
The node needs ~8 successful iterations of +10K each to catch up, with cooldown between each.

### Bug 2: Peer Returns 0 Blocks Despite Claiming High Height

**Evidence from logs:**
```
Received 0 blocks: 13480038..=13484627 (RTT: 45ms)
Received 0 blocks: 13484628..=13489217 (RTT: 54ms)
Received 0 blocks: 13502988..=13507577 (RTT: 55ms)
```

Peer `12D3KooWAaxRjk6m` (NOT our server — unknown user node) claims height 13,558,823
but returns 0 blocks for every request. Fast RTT (45ms) proves it's responding — just empty.

**File:** `crates/q-storage/src/turbo_sync.rs:5030-5103`

The turbo sync treats 0-block responses as **successful chunks** (`apply_blocks_vec` 
with empty vec returns `Ok(())`). The sync thinks it completed even though no blocks 
were stored. This is a bug: 0 blocks from a peer claiming a higher height should be a 
FAILURE, not success.

### Bug 3: Serialization Mismatch (0xc8 First Byte)

**Evidence:**
```
Failed to parse response: not valid bincode, CBOR or JSON (first byte: 0xc8, len: 580873)
```

Some peers (likely running older versions) send block-pack responses in a format that
v10.2.8 cannot deserialize. The 0xc8 byte suggests MessagePack or a compression frame.
This causes valid block data (580KB+) to be discarded.

**File:** `crates/q-network/src/unified_network_manager.rs` (block-pack codec)

## 3. Why These Bugs Create a Deadlock

```
1. Peer announces height 13.5M
2. HEIGHT CLAMP caps sync target to 13,485,447 (current + 10K)
3. Turbo sync requests chunks from only qualified peer (12D3KooWAaxRjk6m)
4. Peer returns 0 blocks -> turbo sync thinks it succeeded -> no progress
5. Other responses fail with 0xc8 parse error -> discarded
6. Delta (our server, has all blocks) may not be registered as sync peer
7. Sync loop enters cooldown, repeats -> zero progress
```

## 4. Proposed Safe Fixes

### Fix A: Treat 0-block responses as failures (LOW RISK, HIGH IMPACT)

**File:** `crates/q-storage/src/turbo_sync.rs`

In the chunk handler, after receiving blocks from a peer:

```rust
// CURRENT (buggy):
let blocks = response.blocks;  // Could be empty vec
self.apply_blocks_vec(blocks).await?;  // "Success" with 0 blocks

// PROPOSED:
let blocks = response.blocks;
if blocks.is_empty() && requested_range_size > 0 {
    warn!("Peer {} returned 0 blocks for range {}-{} — treating as failure",
          peer_id, start, end);
    return Err(anyhow::anyhow!("Peer returned 0 blocks"));
}
self.apply_blocks_vec(blocks).await?;
```

**Risk:** Very low. This only changes error handling for an already-broken case.
A peer returning 0 blocks when blocks were requested is ALWAYS wrong.

**Impact:** Turbo sync will retry with a different peer instead of treating empty 
responses as success. If Delta is available, it will be tried next.

### Fix B: Don't clamp sync target for the sync loop (MEDIUM RISK)

**File:** `crates/q-api-server/src/main.rs:14528`

The HEIGHT CLAMP should only affect display/mining, not the sync target.
Store the unclamped network height in a separate atomic for the sync loop:

```rust
// Store unclamped height for sync decisions
app_state.unclamped_network_height.store(network_height, SeqCst);

// Only clamp the display/mining height
if network_height > hard_cap && local_height > 100_000 {
    app_state.highest_network_height.store(hard_cap, SeqCst);
}
```

Then in the sync loop (line 19394):
```rust
let sync_target = unclamped_network_height.min(current_height + 50_000);
// Allow up to 50K blocks per batch during catch-up
```

**Risk:** Medium. Increasing the batch size could stress the node during sync.
The 10K clamp exists to prevent memory exhaustion from huge sync batches.
Mitigation: keep the per-chunk size small (5K blocks), just allow more chunks per batch.

### Fix C: Ensure Delta is registered as sync peer (IMMEDIATE, NO CODE CHANGE)

Check if Delta (`12D3KooWLJJRvqo6m`) is registered in Epsilon's turbo sync peer list.
If not, restart Epsilon to force re-discovery. Delta was serving blocks successfully 
earlier (2 chunks at 38ms) so the protocol works between them.

## 5. Recommended Execution Order

**Phase 1 (immediate, no code change):**
- Verify Delta is connected to Epsilon and registered as sync peer
- If not, investigate why and restart if needed

**Phase 2 (code fix, low risk):**
- Fix A: Treat 0-block responses as failures in turbo_sync.rs
- This is a 3-line change with zero mainnet risk

**Phase 3 (code fix, after Phase 2 verified):**
- Fix B: Separate clamped vs unclamped network height
- Allow larger sync batches during catch-up mode
- Test on Epsilon first before deploying to Beta

## 6. What NOT to Do

1. Do NOT wipe the database — 13.4M+ valid blocks would take 6+ hours to re-sync
2. Do NOT disable HEIGHT CLAMP entirely — it protects against malicious height poisoning
3. Do NOT skip the 0-block fix — the fake "success" is the immediate cause of stall
4. Do NOT increase chunk size beyond 10K blocks — memory exhaustion risk

## 7. Verification Plan

After applying fixes:
1. Monitor: `journalctl -u q-api-server -f | grep -E 'current_height|Downloaded|applied'`
2. Height should advance by ~10K every 30-60 seconds
3. Full catch-up (83K blocks on Epsilon) should take ~15 minutes
4. Explorer should show advancing height within 5 minutes of fix
5. Block production resumes when height reaches network tip

## 8. Root Cause of the Incident (Summary)

```
kill -9 on Epsilon
  -> RocksDB partial writes (200 corrupt blocks)
  -> v10.2.8 cleanup deleted corrupt blocks (correct)
  -> Left 200-block gap with orphaned blocks above
  -> Auto-repair fought pointer reset (cascade loop)
  -> Manual tool deleted orphaned blocks (correct)
  -> Pointer reset to 13,475,447 from checkpoint
  -> Turbo sync starts, downloads 10K blocks
  -> HEIGHT CLAMP caps at +10K, sync thinks it's caught up
  -> Next batch gets 0-block responses from broken peer
  -> Treats 0 blocks as "success" -> no progress
  -> Deadlock: sync never advances past first 10K batch
```

---

**Prepared by:** Claude Code (Server Alpha)  
**Review requested from:** DeepSeek  
**Classification:** Mainnet Critical  
**Recommendation:** Apply Fix A immediately (3-line change), then Fix B
