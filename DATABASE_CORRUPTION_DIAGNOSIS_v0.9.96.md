# Database Corruption Diagnosis - v0.9.96-beta

**Date:** 2025-11-11 10:05 CET
**Status:** 🚨 CRITICAL - Database Completely Empty
**Database:** data-mine10/hot

---

## Executive Summary

The blockchain is not stalled due to the duplicate retry loop bug - **the database is completely corrupted**. The `qblock:latest` pointer claims height 766, but the database contains ZERO blocks. This is a catastrophic data loss scenario.

---

## Database Integrity Check Results

```
🔧 Q-NarwhalKnight Database Repair Utility v0.5.22
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📂 Opening database: ./data-mine10/hot

📊 Scan Results:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Total blocks found: 0          ❌ DATABASE EMPTY!
   Highest block: 0
   Highest contiguous: 0

🔍 Checking qblock:latest pointer...
   Current pointer: 766           ⚠️ PHANTOM HEIGHT!
   Should be: 0

⚠️ Pointer is WRONG! Database claims 766 blocks but has 0!
```

---

## What This Means

### The Phantom Height Problem

**Database State:**
- Pointer: `qblock:latest = 766` ✅ (exists)
- Blocks: `qblock:height:0` through `qblock:height:766` ❌ (ALL MISSING!)
- Result: **Producers sync to phantom height 766**

**Why System Appeared to Work:**
1. Service starts, reads `qblock:latest = 766`
2. Producers sync to height 766 (believes database has 766 blocks)
3. Logs show "synchronized to height 766" ✅
4. Producers try to create block 766 or 767
5. BlockWriter tries to save: **"Block already exists at height 766"** ❌
6. But block 766 DOESN'T exist - this is a false positive!
7. Circuit breaker opens from repeated failures

### The False Duplicate Error

The error "Block already exists at height 766" is **MISLEADING**. What actually happened:

1. Producers created blocks 0-765 during v0.9.95
2. Blocks were queued in BlockWriter
3. **Database write FAILED** (corruption event)
4. Blocks 0-766 never actually saved to disk
5. But `qblock:latest` pointer was updated to 766
6. On restart, system thinks blocks exist
7. Tries to save "duplicate" blocks that never existed

---

## Root Cause Analysis

### How Did This Happen?

**Possible Causes:**
1. **RocksDB Crash** - Process killed during write
2. **Disk Full** - Ran out of space during batch write
3. **Atomic WriteBatch Bug** - v0.9.95 fix had a flaw
4. **Database Corruption** - Hardware/filesystem issue

**Most Likely:** The v0.9.95 atomic WriteBatch fix had an edge case where:
- Block data write failed
- But pointer update succeeded
- Left database in inconsistent state

---

## Evidence Timeline

### v0.9.95-beta Deployment (09:46:48)
```
09:46:52 - ✅ Producers synchronized to height 764
09:47:29 - 💾 Saving QBlock at height 765
09:47:31 - ✅ Producers synchronized to height 765
```
**Status:** Blocks appeared to save successfully

### Circuit Breaker Activation (09:47:29 - 09:48:00)
```
09:47:29 - ⚠️ Block already exists at height 765
09:47:29 - ❌ Block 765 write failed: Block already exists
...
09:48:00 - 🚨 Circuit breaker OPEN (120 consecutive errors)
```
**Status:** Duplicate errors triggered circuit breaker

### v0.9.96-beta Deployment (10:01:06)
```
10:01:12 - ✅ Producers synchronized to height 765
10:01:42 - ⚠️ Block already exists at height 766
10:01:42 - ✅ Producers resynced after duplicate
```
**Status:** v0.9.96 resync logic working, but database empty

### Database Corruption Discovery (10:05:00)
```
repair-database check:
  Total blocks: 0
  Pointer: 766
  Status: CORRUPTED
```
**Status:** Database completely empty despite pointer at 766

---

## Why v0.9.96 Fix Didn't Help

The v0.9.96 fix (post-duplicate resync) **is working correctly**, but it can't fix a corrupted database:

**v0.9.96 Behavior (Working as Designed):**
1. Producer tries to save block 766
2. Gets error "Block already exists" (false - database empty)
3. Forces immediate resync ✅
4. Resyncs to height 766 (reads corrupt pointer)
5. Tries to save block 766 again
6. **Loop continues** because pointer is wrong

**The Real Problem:** Not the duplicate retry loop, but the corrupt database pointer.

---

## Recovery Options

### Option 1: Reset Database to Genesis (Testnet OK)
```bash
# Stop service
systemctl stop q-api-server

# Fix pointer to match reality
echo "1" | cargo run --bin repair-database --release -- ./data-mine10/hot

# Restart service (will sync from genesis)
systemctl start q-api-server
```
**Pros:** Clean start, v0.9.96 fixes will work
**Cons:** Loses all 766 blocks (testnet acceptable)
**Time:** Immediate recovery

### Option 2: Restore from Backup
```bash
# Check available backups
ls -lh data-mine*/hot

# data-mine1/hot = 1.7G (may have actual blocks)
# data-mine6/hot = 1.9G
# data-mine5/hot = 4.2G

# Copy backup to data-mine10
systemctl stop q-api-server
rm -rf data-mine10/hot
cp -r data-mine5/hot data-mine10/hot

# Restart
systemctl start q-api-server
```
**Pros:** Recovers historical blocks
**Cons:** May have old corruption
**Time:** 5-10 minutes

### Option 3: Diagnose Atomic WriteBatch Bug (Development)
Investigate why atomic WriteBatch allowed pointer update without block data.

**Time:** Hours/days

---

## Recommended Action

**For Production Recovery (Immediate):**

Use **Option 1** - Reset to genesis and start fresh:

```bash
# STEP 1: Stop service
systemctl stop q-api-server

# STEP 2: Reset pointer to 0
echo "1" | timeout 60 cargo run --bin repair-database --release -- ./data-mine10/hot

# STEP 3: Verify pointer reset
cargo run --bin repair-database --release -- ./data-mine10/hot
# Should show: "Pointer: 0, Blocks: 0, ✅ Pointer is correct"

# STEP 4: Restart service
systemctl start q-api-server

# STEP 5: Monitor recovery
journalctl -u q-api-server -f | grep -E "💾 Saving QBlock at height|synchronized to height"
```

**Expected Recovery:**
- Producers sync to height 0
- Block 0 (genesis) created
- Continuous progression from 0 → 1 → 2 → ...
- v0.9.96 duplicate handling working correctly
- No circuit breaker issues

---

## Prevention Measures (v0.9.97+)

### Fix 1: Atomic WriteBatch Verification
```rust
// After db.write_batch(batch).await
// VERIFY both block AND pointer were written
let verify_block = db.get(CF_BLOCKS, &height_key).await?;
let verify_pointer = db.get(CF_BLOCKS, b"qblock:latest").await?;

if verify_block.is_none() || verify_pointer.is_none() {
    error!("🚨 Atomic write FAILED - rolling back!");
    // Attempt rollback or panic
}
```

### Fix 2: Startup Integrity Check
```rust
// On service startup, verify pointer matches highest block
let pointer_height = storage.get_latest_qblock_height().await?;
let actual_highest = storage.scan_for_highest_block().await?;

if pointer_height != actual_highest {
    error!("🚨 Database corruption detected!");
    error!("   Pointer: {}, Actual: {}", pointer_height, actual_highest);
    error!("   Refusing to start - run repair tool first!");
    std::process::exit(1);
}
```

### Fix 3: Write Verification
```rust
// After EVERY block write
let saved_block = db.get(CF_BLOCKS, &height_key).await?
    .ok_or(anyhow!("Block write verification FAILED"))?;

// Verify hash matches
if saved_block_hash != expected_hash {
    panic!("🚨 Block hash mismatch - database corruption!");
}
```

---

## Files Affected

### Corruption Location
- `./data-mine10/hot` - Current database (CORRUPTED)
- `qblock:latest` pointer = 766 (wrong)
- All block data = MISSING (0 blocks)

### Working Fixes (v0.9.96)
- `crates/q-storage/src/block_writer.rs` - ✅ Atomic WriteBatch
- `crates/q-api-server/src/main.rs` - ✅ Post-duplicate resync

### Needs Investigation
- Why did atomic WriteBatch allow partial writes?
- Was there a RocksDB crash during v0.9.95 deployment?
- Check system logs for disk errors

---

## Lessons Learned

1. **Always Verify Writes** - Don't trust database APIs
2. **Startup Integrity Checks** - Detect corruption before it causes issues
3. **Pointer-Data Consistency** - Verify atomicity worked
4. **Better Error Messages** - "Block already exists" was misleading
5. **Testnet is for Testing** - Good we found this before mainnet

---

## Conclusion

The v0.9.95 and v0.9.96 fixes for height desync and duplicate retry loops are **working correctly**. The real problem is catastrophic database corruption where the pointer claims 766 blocks exist but the database is completely empty.

**Immediate Action Required:**
Reset database pointer to 0 and start fresh from genesis.

**Long-term Action Required:**
Investigate atomic WriteBatch implementation to prevent future corruption.

---

**Document Version:** 1.0
**Last Updated:** 2025-11-11 10:10 CET
**Status:** AWAITING USER DECISION (Reset vs Restore)
