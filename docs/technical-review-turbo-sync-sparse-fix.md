# Technical Review: Turbo Sync Sparse DAG Fix (v10.2.9-fix)

**Date**: 2026-04-10
**Severity**: Critical (sync completely broken for new nodes)
**Files Changed**:
- `crates/q-storage/src/lib.rs` (~line 2100, `get_qblocks_range`)
- `crates/q-storage/src/turbo_sync.rs` (~line 5050, chunk response handler)

---

## 1. Root Cause: Two Safety Fixes That Contradict Each Other

The turbo sync deadlock was caused by two of our own fixes interacting destructively:

**v10.2.8** (turbo_sync.rs): "If a peer returns 0 blocks for a requested range, treat it as FAILURE and return Err."

Rationale: A peer claiming height 12M but returning nothing is lying or broken. Without this, empty responses counted as "success" and the sync cursor advanced past blocks it never received — causing permanent gaps.

**v10.2.9** (lib.rs): "If the first 10 blocks in a range are missing from our DB, abort early and return the (empty) result."

Rationale: When peers request blocks from corrupt height ranges (e.g., 6M-12M with thousands of corrupt entries from old format data), scanning the full range causes an I/O storm. The early-abort prevents the serving node from doing 200+ RocksDB reads when the first 10 all fail.

**The deadlock**: The v10.2.9 early-abort treated missing blocks (None from `multi_get`) the same as corrupt blocks (data present but deserialization fails). In a sparse DAG, the first 10 heights often have no blocks at all — they are simply empty heights. So the serving node aborts after 10 missing entries and returns 0 blocks. The syncing node sees 0 blocks and, per v10.2.8, treats it as a peer failure. Every chunk request fails. Sync is completely dead.

## 2. Why It Worked Two Days Ago

Before v10.2.9's early-abort patch, `get_qblocks_range` would scan the entire requested range (e.g., 200 heights) regardless of how many were missing at the start. In a sparse DAG, blocks might exist at heights 15, 47, 89, 133, etc. — the function would skip the missing ones and return whatever it found. The v10.2.8 "0 blocks = failure" check rarely triggered because the full scan almost always found *some* blocks in a 200-height window.

After v10.2.9, the scan aborts after the first 10 consecutive missing entries. In sparse regions, this means it never reaches the heights that do have blocks.

## 3. The Two Fixes

### Fix 1: Early-abort only on CORRUPT blocks, not missing blocks (lib.rs)

**Change**: The `else` branch (block not found in DB, `None` from `multi_get`) no longer increments `consecutive_failures`. Instead, it resets the counter to 0.

**Before**: Both missing blocks and corrupt blocks incremented the abort counter.
**After**: Only corrupt blocks (data present but fails deserialization) increment it. Missing blocks reset it.

**Safety**: This preserves the original v10.2.9 protection against I/O storms from corrupt data ranges. If 10 consecutive blocks have data in the DB but cannot be deserialized, the abort still triggers. But a run of empty heights (normal DAG sparsity) will not trigger the abort, allowing the scan to continue and find blocks at later heights.

**Edge case**: A range with alternating corrupt and missing blocks (corrupt, missing, corrupt, missing...) will never trigger the abort because the missing blocks reset the counter. This is acceptable — the corrupt blocks are individually deleted (line 2154), and the scan still terminates at the end of the requested range.

### Fix 2: Accept 0-block responses as valid sparse ranges (turbo_sync.rs)

**Change**: When a peer returns 0 blocks, instead of returning `Err(...)`, log it as a sparse range event and return `Ok(())`. The cursor advances past this range.

**Before**: 0 blocks = `Err` = peer scored as failed = chunk retried (infinitely, since every retry hits the same sparse range).
**After**: 0 blocks = `Ok` = cursor advances = sync proceeds to the next range.

**Safety**: The function returns `Ok(())` early, skipping `apply_blocks_vec` (nothing to apply). The `active_parallel_streams` counter is decremented. Apollo gravity-assist tracking still records the (empty) chunk. No blocks are marked as synced that were not actually received — the cursor simply moves forward.

## 4. Questions for AI Reviewers

### Is accepting 0-block responses safe for a DAG chain?

In a DAG (as opposed to a linear chain), not every height necessarily contains a block. The DAG may have blocks at heights [1, 3, 7, 15, 16, 17, 45, ...] with gaps. Accepting empty responses for gap regions is semantically correct — there is genuinely nothing to sync at those heights.

However, this creates a risk: the sync cursor can advance past heights where blocks DO exist on the peer but were not returned (e.g., due to a bug in the serving node's query, network corruption, or a malicious peer). The syncing node will never know it missed those blocks unless a separate integrity check catches the gap later.

### Could a malicious peer exploit this to stall sync?

A malicious peer could return empty responses for every range, causing the sync cursor to advance to the tip without actually syncing any blocks. The syncing node would believe it is "caught up" with 0 blocks.

**Mitigation**: After sync completes, the node should verify it has a reasonable number of blocks relative to the announced height. If a peer claims height 12M but the node only received 1M blocks total, that is suspicious. This check does not currently exist and should be added.

### Should there be a maximum gap size before treating as suspicious?

Yes. A reasonable heuristic: if more than N consecutive chunk requests (e.g., 50 chunks x 200 heights = 10,000 heights) return 0 blocks, the peer should be flagged and the sync should try a different peer. DAG sparsity should not produce 10,000 consecutive empty heights in normal operation.

### How do Kaspa and other DAG chains handle sparse height ranges during sync?

Kaspa uses a "headers-first" approach where the DAG structure (block headers with parent references) is synced separately from block bodies. This means the syncing node knows exactly which heights have blocks before requesting bodies, avoiding the "blind range request" pattern that causes this issue.

An alternative approach for Q-NarwhalKnight would be to add a lightweight "height bitmap" exchange during sync negotiation, where the peer sends a compressed bitmap of which heights contain blocks. This would eliminate blind requests entirely.

---

## 5. Summary

| Aspect | Before Fix | After Fix |
|--------|-----------|-----------|
| Missing blocks in range scan | Increment abort counter | Reset abort counter (not a failure) |
| Corrupt blocks in range scan | Increment abort counter | Increment abort counter (unchanged) |
| 0-block peer response | Hard error, retry forever | Ok, advance cursor |
| Sparse DAG sync | Deadlocked | Proceeds through gaps |
| I/O storm protection (corrupt) | Active | Active (unchanged) |

Both fixes are minimal and targeted. The early-abort I/O protection for corrupt data ranges is preserved. The only behavioral change is that missing/sparse data is no longer treated as a failure condition.
