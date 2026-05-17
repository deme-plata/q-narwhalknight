# Technical Review: Epsilon Block Corruption & Sync Stall

**Date:** 2026-04-06  
**Severity:** Critical (mainnet, $920M cap)  
**Affected Server:** Epsilon (89.149.241.126) — 10Gbit supernode  
**Root Cause:** `kill -9` during active RocksDB writes  
**Status:** Corruption cleaned, sync stalled due to architectural conflict  

## 1. Incident Timeline

| Time | Event |
|------|-------|
| ~2026-04-05 | `kill -9` sent to Epsilon during active block writes |
| 2026-04-05 17:11 | v10.2.7 deployed with debug logging — no improvement |
| 2026-04-06 13:36 | Investigation begins — 100% mining rejection on all 8 shards |
| 2026-04-06 15:50 | v10.2.8 deployed with `cleanup_corrupt_blocks_near_tip()` |
| 2026-04-06 15:50:15 | **180 corrupt blocks deleted** (heights 13,489,244-13,489,423) |
| 2026-04-06 16:08 | First deploy: cleanup in `recover()` — ran at height 0, no-op |
| 2026-04-06 17:49 | Second deploy: cleanup after `scan_highest_contiguous_block_internal()` — works |
| 2026-04-06 17:50 | Corruption cleaned, but 200-block gap remains |
| 2026-04-06 18:20+ | Turbo sync downloads blocks ABOVE gap but height never advances |
| 2026-04-06 19:30 | Third restart — auto-repair keeps resetting pointer back to 13,489,443 |

## 2. Database State (Verified with fix-corrupt-tip --dry-run)

```
Height Range         Status    Details
-------------------------------------------------------------------
13,489,140-13,489,238  VALID    99 blocks, ~40-51KB each (healthy)
13,489,239-13,489,242  VALID    4 blocks, 876-2476 bytes (small but valid)
13,489,243-13,489,442  MISSING  200 blocks -- deleted by our fix + original corruption
13,489,443             CORRUPT  1948 bytes -- partial write from kill -9
13,489,444             CORRUPT  2268 bytes -- partial write from kill -9
13,489,445-13,489,450  VALID    6 blocks, ~51KB each (healthy)
13,489,450+            VALID    Turbo sync downloaded blocks above this

Current Pointers:
  qblock:latest     = 13,489,443 (POINTS TO CORRUPT BLOCK!)
  qblock:safe_floor = 13,489,443
  qblock:tip_height = 13,489,242
```

## 3. Root Cause: kill -9 During RocksDB Writes

`kill -9` (SIGKILL) does not allow RocksDB to flush its WAL (Write-Ahead Log) or 
memtables cleanly. Blocks being written at the moment of the kill end up as:

- **Truncated data** ("io error: unexpected end of file")
- **Garbage bytes** ("tag for enum is not valid, found 120")
- **Partial bincode** (valid header but truncated body)

The kill corrupted ~200 blocks around the write frontier (13,489,243-13,489,444).

## 4. Why the Fix Partially Worked

### What v10.2.8 `cleanup_corrupt_blocks_near_tip()` Did Correctly:
1. Scanned 200 blocks below recovered height
2. Detected 180 blocks that fail deserialization
3. Deleted them from RocksDB
4. Reset `qblock:latest`, `qblock:safe_floor`, `qblock:tip_height` to 13,489,243

### Why Height Doesn't Advance (The Gap Problem):

```
Valid blocks:  ############ ... ################## ... ################
Heights:       13,489,238  13,489,242 | GAP (200 missing) | 13,489,443+
                                      |-- turbo sync only goes FORWARD
                                          from current_height (13,489,443)
                                          never looks back to fill this gap
```

- `current_height` = 13,489,443 (set by startup auto-repair)
- Turbo sync requests blocks from 13,489,444+ (above current_height)
- The gap at 13,489,243-13,489,442 is BELOW current_height
- Contiguous height can't advance past 13,489,242 (gap starts there)
- Block production requires contiguous chain -- stays stuck

### The Auto-Repair Cascade:

Every time we reset `qblock:latest` to below the gap (e.g., 13,489,242):

1. `scan_highest_contiguous_block_internal()` returns height from cache/pointer
2. `verify_database_integrity()` reads pointer (13,489,242)
3. Finds "actual" height = 0 (cache was force-set to 0)
4. "Corrects" pointer to 0
5. `repair_pointer_to_contiguous()` scans backwards from 0
6. Finds blocks at 13,489,443 -> sets pointer BACK to 13,489,443
7. `current_height_atomic` initialized to 13,489,443
8. **Back to square one**

This is an architectural conflict between:
- **Our fix** (wants pointer below gap so turbo sync fills it)
- **Auto-repair** (always pushes pointer to highest existing block)

## 5. Proposed Safe Fix

### Option A: Delete blocks above the gap, let turbo sync rebuild (RECOMMENDED)

**Rationale:** Instead of fighting the auto-repair system, work WITH it. Delete the 
isolated blocks above the gap (13,489,443-13,489,450 + any turbo-synced blocks above).
The highest contiguous block becomes 13,489,242. Auto-repair finds 13,489,242, sets 
pointer there. Turbo sync fills forward from 13,489,243.

**Steps:**
1. Stop Epsilon (`systemctl stop q-api-server`)
2. Run `fix-corrupt-tip` with `--force-delete` for heights 13,489,443-13,489,450 + `--reset-pointer 13489242`
3. Also delete any turbo-synced blocks above 13,489,450 (they came from the previous broken session)
4. Start Epsilon -- auto-repair finds 13,489,242 as highest block -- no conflict
5. Turbo sync fills from 13,489,243 forward

**Risk:** LOW -- only deletes blocks that are either:
- Corrupt (13,489,443-13,489,444 -- confirmed by tool scan)
- Isolated above a 200-block gap (useless without the blocks below them)
- Downloaded during a broken session (will be re-downloaded cleanly)

**Recovery time:** ~1-2 hours to sync ~68K blocks from Delta

### Option B: Code fix to disable auto-repair for this restart cycle

**Rationale:** Add an environment variable `Q_SKIP_AUTO_REPAIR=1` that disables the 
`repair_pointer_to_contiguous()` function for one restart cycle. Set pointer to 
13,489,242, start with flag, turbo sync fills gap, remove flag.

**Risk:** MEDIUM -- touching auto-repair logic on mainnet is risky

### Option C: Manual HTTP state sync

**Rationale:** Use HTTP full-state sync from Delta to get the missing blocks.

**Risk:** LOW but slow -- HTTP sync is designed for full state, not targeted gap-fill

## 6. Recommendation

**Option A** is safest and simplest:

```bash
# 1. Ensure Epsilon is stopped
systemctl stop q-api-server

# 2. Delete corrupt + isolated blocks above gap, reset pointer
./fix-corrupt-tip --db-path /home/orobit/data-mainnet-genesis/hot \
  --force-delete 13489443,13489444,13489445,13489446,13489447,13489448,13489449,13489450 \
  --reset-pointer 13489242 \
  --apply

# 3. Start Epsilon
systemctl start q-api-server

# 4. Monitor: height should advance past 13,489,242 within minutes
journalctl -u q-api-server -f | grep 'current_height'
```

The 6 valid blocks at 13,489,445-13,489,450 are sacrificed (they'll be re-downloaded),
but this eliminates the gap and lets the auto-repair system work correctly.

## 7. The Deeper Problem: Turbo-Synced Blocks Above the Gap

The turbo sync sessions during previous restarts downloaded thousands of blocks above 
13,489,450 (ranges like 13,494,240+, 13,508,628+, etc.). These blocks are valid but 
isolated -- they exist above the gap and would cause the same auto-repair problem.

**These must also be deleted.** The fix-corrupt-tip tool has a 100-block scan limit,
so for the turbo-synced blocks we need a broader deletion strategy. Options:
1. Extend the tool's range limit
2. Use a separate script to delete all blocks above 13,489,450
3. Accept that auto-repair will find them and add code to handle the gap

**Recommendation:** Delete ALL blocks above 13,489,242 using the `--force-delete` flag 
with the full range, or add a new tool mode that deletes a range.

## 8. Prevention

1. **Never use `kill -9` on the node** -- use `systemctl stop` or `kill -TERM`
2. If `kill -9` is necessary, run `fix-corrupt-tip --dry-run` before restarting
3. The v10.2.8 `cleanup_corrupt_blocks_near_tip()` prevents future corruption from persisting
4. Consider adding RocksDB WAL sync on every Nth block write for crash resilience

## 9. Serialization Mismatch Issue (Secondary)

During investigation, we discovered block-pack responses between v10.2.3 (Delta) and 
v10.2.8 (Beta/Epsilon) fail with "not valid bincode, CBOR or JSON (first byte: 0xc8)".
Deploying v10.2.8 to Delta resolved this. All production nodes should run the same version.

## 10. Architectural Recommendations

The auto-repair cascade reveals a design tension:
- **Auto-repair** assumes the highest existing block is the correct tip
- **Gap-fill** assumes blocks below current_height will be filled in background
- **Neither handles the case** where corruption creates a gap below the tip with valid blocks above

Suggested improvements:
1. Auto-repair should check for contiguity, not just block existence
2. Turbo sync gap-fill should work BOTH forward and backward from current_height  
3. `cleanup_corrupt_blocks_near_tip()` should also delete isolated valid blocks above the gap
4. Add a startup flag to override auto-repair for recovery scenarios

---

**Prepared by:** Claude Code (Server Alpha)  
**Review requested from:** DeepSeek  
**Classification:** Mainnet Critical -- Handle with care
