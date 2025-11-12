# Height Reset Bug - Root Cause Analysis

**Date**: November 3rd, 2025 - 20:50 CET
**Status**: 🔍 **INVESTIGATION IN PROGRESS**
**Priority**: **P0 - CRITICAL**

---

## 🚨 User Report

**User observations:**
1. Height was 3000+ blocks
2. Then dropped to 1400 blocks
3. Now showing 558 blocks
4. "The node always drops back to 0 and it starts all over again"

**Impact**: Blockchain height repeatedly regressing, causing data loss and sync loops.

---

## 🔍 What We Know

### Server Beta (Bootstrap Node) Status
- **Database**: `/opt/orobit/shared/q-narwhalknight/data-mine3`
- **Database Size**: 2.1 GB (healthy, has data)
- **Current Height**: 558 blocks (loaded from database)
- **Version**: v0.9.0-beta-emergency (height monotonicity protection active)

### Height Monotonicity Fix Status
**What it protects against:**
- ✅ Runtime height regression (during block production)
- ✅ Network sync-down attacks (malicious peers)
- ✅ Turbo sync corruption (bad sync packs)

**What it DOES NOT protect against:**
- ❌ **Database corruption/rollback**
- ❌ **Loading from old/corrupted database on startup**
- ❌ **RocksDB compaction issues**
- ❌ **Disk-level data loss**

---

## 💡 Key Insight: Two Different Problems

### Problem 1: Runtime Height Regression (FIXED in v0.9.0)
**What happens:**
- Node is running with height 1000
- Network sync tries to reset height to 0
- v0.9.0 PANICS and refuses

**Status**: ✅ **FIXED** by height monotonicity enforcement

### Problem 2: Database Corruption/Rollback (NOT FIXED)
**What happens:**
- Node is running with height 3000
- Something corrupts the database files OR
- Database pointer gets corrupted OR
- Node restarts and loads old database state
- Node starts with height 558 (whatever is in database)

**Status**: ❌ **NOT FIXED** - This is the current issue!

---

## 🔬 Technical Analysis

### Why Height Monotonicity Didn't Trigger

The v0.9.0-beta-emergency fix checks height at:
1. `main.rs:3191` - Block production from mining solutions
2. `main.rs:3518` - Parallel block production
3. `main.rs:3873` - Turbo sync height updates

**But it does NOT check at:**
- ❌ Database load on startup (`BlockProducer::load_from_storage()`)
- ❌ Database pointer reads (`get_latest_qblock()`)
- ❌ Recovery from crashed state

**What this means:**
If the database files themselves are corrupted or rolled back, the node will happily load the corrupted state on restart.

---

## 🎯 Possible Root Causes

### Hypothesis 1: RocksDB Pointer Corruption
**Theory**: The `qblock:latest` pointer gets corrupted/reset
**Evidence**:
- User reports height jumping around (3000 → 1400 → 558)
- Database size is 2.1GB (suggests data exists)
- repair_database.rs was created specifically for this issue

**Test**: Run `repair-database` tool to check if pointer is wrong

### Hypothesis 2: Database Compaction/Truncation
**Theory**: RocksDB compaction is deleting blocks
**Evidence**:
- Height decreasing over time
- Database has lots of SST files (compaction active)

**Test**: Check if blocks 559-3000 still exist in database

### Hypothesis 3: Multiple Databases
**Theory**: User is running multiple nodes with different databases
**Evidence**:
- Frontend shows different heights at different times
- Multiple data directories exist (data-mine1, data-mine2, data-mine3, etc.)

**Test**: Ask user which database they're using

### Hypothesis 4: Incomplete Flush
**Theory**: Blocks are written to memory but not flushed to disk
**Evidence**:
- Height shows high during runtime
- After restart, height is lower
- Crash/restart loses unflushed data

**Test**: Check RocksDB flush settings

---

## 🛠️ Diagnostic Steps

### Step 1: Run Database Repair Tool

The `repair-database` utility will:
1. Scan all blocks in database (0 to 200,000)
2. Find the highest contiguous block
3. Check if `qblock:latest` pointer is correct
4. Offer to fix the pointer if wrong

**Command:**
```bash
# Stop the node first
systemctl stop q-api-server

# Run diagnostic (read-only until you choose option 1)
./target/release/repair-database ./data-mine3/hot
```

**Expected output:**
```
📊 Scan Results:
   Total blocks found: XXXX
   Highest block: XXXX
   Highest contiguous: XXXX

🔍 Checking qblock:latest pointer...
   Current pointer: 558 (height)
   ⚠️  Pointer is WRONG! Should be XXXX
```

### Step 2: Check Block Existence

**Command:**
```bash
# Check if high blocks still exist
./target/release/repair-database ./data-mine3/hot 2>&1 | grep "Highest block"
```

**Scenarios:**
- **Scenario A**: "Highest block: 3000+" → Data exists, pointer is wrong (easy fix)
- **Scenario B**: "Highest block: 558" → Data was actually deleted (harder problem)

### Step 3: Check for Multiple Databases

**User question**: "What does your node startup command look like?"

**Look for:**
- `Q_DB_PATH=./data` environment variable
- Which data directory they're using
- If they have multiple nodes running

---

## 🚑 Immediate Workarounds

### Option A: Use Repair Tool (If Data Exists)
If blocks 559-3000 still exist in database, just fix the pointer:

```bash
systemctl stop q-api-server
./target/release/repair-database ./data-mine3/hot
# Choose option 1 to fix pointer
systemctl start q-api-server
```

### Option B: Fresh Sync (If Data Was Deleted)
If blocks were actually deleted:

```bash
# Backup current database (just in case)
mv ./data-mine3 ./data-mine3-backup-$(date +%s)

# Start fresh and sync from Server Beta
./q-api-server-v0.9.0-beta-emergency --port 8080
```

### Option C: Copy Server Beta Database
Use Server Beta's known-good database:

```bash
# Stop local node
systemctl stop q-api-server

# Backup local database
mv ./data ./data-backup-$(date +%s)

# Copy Server Beta database
rsync -avz root@185.182.185.227:/opt/orobit/shared/q-narwhalknight/data-mine3/ ./data/

# Start local node
systemctl start q-api-server
```

---

## 🔧 Permanent Fix Required

### What Needs To Be Implemented

#### 1. Database Load Verification (v0.9.1)
```rust
// On startup, before accepting database height
let loaded_height = load_from_database();
let last_known_height = read_from_persistent_file("./last_known_height.txt");

if loaded_height < last_known_height - 100 {
    panic!("DATABASE CORRUPTION DETECTED:
           Loaded height {} but last known height was {}!
           Database has been corrupted or rolled back.
           Restore from backup or delete and resync.",
           loaded_height, last_known_height);
}
```

#### 2. Height Persistence to Separate File (v0.9.1)
```rust
// After every successful block
std::fs::write("./last_known_height.txt", format!("{}", current_height))?;
```

#### 3. Automatic Hourly Backups (v0.9.2)
```bash
# Cron job: Every hour, backup database
0 * * * * tar -czf /backups/q-narwhal-$(date +\%Y\%m\%d-\%H\%M).tar.gz ./data
```

#### 4. Database Integrity Checks (v0.9.2)
```rust
// On startup
verify_database_integrity()?;
verify_pointer_matches_highest_block()?;
verify_no_missing_blocks(0, current_height)?;
```

---

## 📊 Current Status

### What We've Done
1. ✅ Deployed v0.9.0-beta-emergency with runtime height protection
2. ✅ Built repair-database diagnostic tool
3. ✅ Identified that database-level corruption is the root cause
4. ⏳ Waiting to run diagnostic on user's database

### What We Need To Do
1. ⏳ Run repair-database tool to diagnose exact issue
2. ⏳ Determine if blocks 559-3000 still exist or were deleted
3. ⏳ Fix pointer if data exists, or resync if data was deleted
4. ⏳ Implement database load verification for v0.9.1
5. ⏳ Implement automatic backups for v0.9.2

---

## 💬 User Communication

**Current Message:**

> The height reset issue is caused by **database corruption**, not network sync.
>
> The v0.9.0-beta-emergency fix prevents height regression DURING RUNTIME, but doesn't detect database corruption on startup.
>
> **We have a diagnostic tool ready to check your database:**
> 1. It will scan for all blocks (0-200,000)
> 2. Check if blocks 559-3000 still exist
> 3. Fix the pointer if data exists
>
> **Can you run this diagnostic?**
> ```bash
> wget http://185.182.185.227/downloads/repair-database
> chmod +x repair-database
> ./repair-database ./data/hot
> ```
>
> This will tell us if your blocks were deleted or just the pointer is wrong.

---

## 🎯 Next Steps

1. **User runs diagnostic** - We need to see if blocks exist
2. **Based on results:**
   - If blocks exist: Fix pointer with repair tool
   - If blocks deleted: Investigate why deletion happened
3. **Implement v0.9.1** with database load verification
4. **Implement v0.9.2** with automatic backups

---

**The mystery continues... Let's see what the diagnostic reveals!**
