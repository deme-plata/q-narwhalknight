# Catastrophic Data Corruption Technical Review - v1.0.0-beta

**Date:** 2025-11-11
**Severity:** CRITICAL - Total Data Loss
**Impact:** 100% of blockchain data (900+ blocks) permanently lost
**System:** Q-NarwhalKnight Blockchain (Server Beta - 185.182.185.227)
**Version:** v1.0.0-beta with Sequential Height Advancement Fix

---

## Executive Summary

A catastrophic data corruption event occurred on Server Beta, resulting in the complete loss of all blockchain blocks (heights 1-900+) despite the database structure remaining intact. Both the active database (`data-mine10`) and its backup (`data-mine10.corrupted-missing-block2-20251111-193131`) contain ZERO blocks, with only metadata pointers remaining that incorrectly reference non-existent block heights.

This represents the most severe data loss incident in the project's history, requiring deep technical analysis to understand the failure mechanism and prevent recurrence.

---

## Timeline of Events

### Background Context (Pre-Incident)
- **v1.0.0-beta Deployment:** Successfully deployed sequential height advancement fix
- **Purpose:** Prevent race conditions in lock-free block producer
- **Expected Behavior:** Block production should proceed sequentially: 1 → 2 → 3 → 4...
- **Network State:** Bootstrap node running, serving blocks to network

### Incident Timeline

**2025-11-11 12:05 - Initial Database Creation**
```
Created: data-mine10.corrupted-missing-block2-20251111-193131
Purpose: Emergency backup during "missing block 2" incident
```

**2025-11-11 19:07 - Last Known Good State**
```
Backup database last modified: 19:07 CET
State: Contains SST files but inspection reveals 0 blocks
qblock:latest pointer: 1 (incorrect, should be 0)
```

**2025-11-11 19:29:02 - Emergency Reset Executed**
```
Emergency bootstrap reset script ran
Action: Moved data-mine10 to backup location
Expected: Service restart with fresh genesis
Actual: Service restart FAILED or never executed
```

**2025-11-11 19:34:56 - Service Started (Wrong Process)**
```
systemctl shows service started at 19:34:56
Problem: Old process still running WITHOUT database
Result: All API endpoints return HTTP 404
Node reports height 749-900 but no blocks exist
```

**2025-11-11 ~19:30-19:34 - Data Loss Window**
```
Duration: ~4-5 minutes
Event: All blocks deleted/lost from database
Mechanism: UNKNOWN - this is the critical mystery
Evidence: Both databases now contain 0 blocks
```

---

## Technical Analysis

### Database State Investigation

#### Database Structure (Intact)
Both databases successfully open with all 20 column families:
```
✅ default
✅ blocks              ← Contains 0 blocks (CRITICAL FAILURE)
✅ dag_vertices
✅ bullshark_cert
✅ manifest
✅ transactions
✅ balances
✅ block_hash_to_height
✅ ai_chats
✅ ai_credits
✅ ai_transactions
✅ ai_treasury
✅ ai_attachments
✅ payment_proposals
✅ payment_votes
✅ payment_locks
✅ banned_peers
✅ sync_certificates
✅ peer_trust
✅ processed_updates
```

**Key Observation:** The RocksDB database structure is PERFECT. Column families exist, CURRENT manifest points to valid MANIFEST file, OPTIONS are intact. Yet NO BLOCKS exist in the `blocks` column family.

#### Backup Database (`data-mine10.corrupted-missing-block2-20251111-193131/hot`)

**Repair Tool Output:**
```
📊 Scan Results:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Total blocks found: 0
   Highest block: 0
   ✅ No gaps detected - chain is contiguous!
   Highest contiguous: 0

🔍 Checking qblock:latest pointer...
   Current pointer: 1 (height)
   ⚠️  Pointer is WRONG! Should be 0
```

**File System Evidence:**
```bash
$ ls -lh ./data-mine10.corrupted-missing-block2-20251111-193131/hot/
total 256120
-rw-r--r-- 1 root root     1560 Nov 11 13:26 000047.sst
-rw-r--r-- 1 root root     3145 Nov 11 18:32 000269.sst
-rw-r--r-- 1 root root 15477438 Nov 11 19:07 000270.log  ← 15MB WAL
-rw-r--r-- 1 root root     2926 Nov 11 18:33 000271.sst
-rw-r--r-- 1 root root 67211845 Nov 11 18:33 000273.sst  ← 67MB SST
-rw-r--r-- 1 root root 67210535 Nov 11 18:33 000274.sst  ← 67MB SST
-rw-r--r-- 1 root root 67200363 Nov 11 18:33 000275.sst  ← 67MB SST
-rw-r--r-- 1 root root 35803381 Nov 11 18:33 000276.sst  ← 35MB SST
-rw-r--r-- 1 root root   220029 Nov 11 19:34 000277.log  ← 220KB WAL (after restart)
-rw-r--r-- 1 root root  5162455 Nov 11 19:07 000278.sst  ← 5MB SST
-rw-r--r-- 1 root root    14638 Nov 11 19:07 MANIFEST-000206
```

**Critical Mystery:** The database has **256MB of SST files** but repair tool finds **0 blocks**. Where did the blocks go?

#### Current Database (`data-mine10/hot`)

**Repair Tool Output:**
```
📊 Scan Results:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Total blocks found: 0
   Highest block: 0
   ✅ No gaps detected - chain is contiguous!
   Highest contiguous: 0

🔍 Checking qblock:latest pointer...
   Current pointer: 900 (height)
   ⚠️  Pointer is WRONG! Should be 0
```

**Key Evidence:**
- qblock:latest points to height 900
- Node was reporting heights 749-900 before shutdown
- But scan of blocks column family finds 0 blocks from height 0-200,000
- Database was created fresh at 19:34 (new genesis attempt)

---

## Root Cause Hypotheses

### Hypothesis 1: Write-Ahead Log (WAL) Flush Failure

**Theory:** Blocks were written to WAL but never flushed to SST files before crash.

**Evidence FOR:**
- Large WAL file (15MB at 19:07) suggests uncommitted writes
- v1.0.0-beta lacks Phase 10 durability guarantees (`sync=true`)
- Emergency script killed service, potentially during write

**Evidence AGAINST:**
- Multiple SST files exist (237MB total) - data WAS flushed at some point
- SST file timestamps show writes up to 19:07, covering hours of operation
- RocksDB should recover from WAL on restart

**Likelihood:** 30% - Partial explanation but doesn't account for SST files being empty

---

### Hypothesis 2: Column Family Key Prefix Mismatch

**Theory:** Blocks were written with wrong key prefix, so repair tool can't find them.

**Evidence FOR:**
- Repair tool searches for `qblock:height:{N}` keys
- If blocks stored with different prefix (e.g., `block:{N}` or `qblock:{N}`), scan would miss them
- Database opened successfully, meaning data structure is valid

**Evidence AGAINST:**
- The key format `qblock:height:{N}` has been consistent since v0.9.0
- All API endpoints returned 404, suggesting blocks genuinely missing
- Other column families (balances, transactions) also appear empty

**Likelihood:** 15% - Would be caught in testing, but worth investigating

---

### Hypothesis 3: RocksDB Compaction Deleted Everything

**Theory:** RocksDB compaction process mistakenly deleted all blocks as "garbage".

**Evidence FOR:**
- Large SST files (67MB each) suggest compaction was active
- Timestamps show SST files created at 18:33, suggesting compaction happened
- If tombstones (deletion markers) were written for all blocks, compaction would delete data

**Evidence AGAINST:**
- No code path exists that would delete all blocks
- Compaction is automatic and shouldn't corrupt data
- RocksDB is battle-tested; this would be a catastrophic RocksDB bug

**Likelihood:** 5% - RocksDB is extremely reliable, unlikely to be root cause

---

### Hypothesis 4: Sequential Height Fix Prevented Block Writes

**Theory:** The sequential height advancement fix blocked block creation, and node only THOUGHT it had blocks.

**Evidence FOR:**
- v1.0.0-beta introduced strict sequential processing
- Lock-free producers wait for previous height before creating next block
- Node reported height 749-900 but blocks don't exist
- Watchdog detected "Block producer STALLED! Height unchanged for 60 seconds"

**Evidence AGAINST:**
- The sequential fix is in the APPLICATION layer, not storage layer
- If blocks weren't created, current_height wouldn't advance
- qblock:latest pointer shows height 900, implying blocks WERE created at some point
- The fix should PREVENT creation, not DELETE existing blocks

**Likelihood:** 40% - Most likely hypothesis. Sequential fix may have caused blocks to be "skipped" in storage while height pointer advanced.

---

### Hypothesis 5: Database Pointer Corruption + Stale Read

**Theory:** Blocks exist in SST files but MANIFEST/pointer corruption causes them to be invisible.

**Evidence FOR:**
- qblock:latest pointers are wrong (1 and 900 when they should be 0)
- 256MB of SST files exist - something must be in there
- MANIFEST file may point to wrong SST files or wrong key ranges
- RocksDB opened successfully, so MANIFEST isn't completely corrupt

**Evidence AGAINST:**
- Repair tool scans blocks column family directly, bypassing high-level pointers
- If blocks existed at any key, scan would find them
- RocksDB's repair mechanisms should detect and fix manifest corruption

**Likelihood:** 25% - Plausible, but repair tool should have found blocks if they exist

---

### Hypothesis 6: Emergency Reset Script Bug

**Theory:** Emergency reset script moved database before service stopped, causing data loss during move.

**Evidence FOR:**
- Emergency script ran at 19:29:02
- Service timestamp shows 19:34:56 (6 minutes later)
- Moving database while service is writing = guaranteed corruption
- Script may have killed service, moved DB, but service restarted incorrectly

**Evidence AGAINST:**
- Script explicitly stops service BEFORE moving database
- Backup database also has 0 blocks, suggesting corruption BEFORE move
- If corruption during move, we'd expect partial data or filesystem errors

**Likelihood:** 35% - Timing discrepancy is suspicious and worth investigating

---

## Critical Technical Questions for Expert Consultation

### Question 1: RocksDB Key Storage and Compaction
**For: RocksDB experts**

Given:
- Column family `blocks` has 256MB of SST files
- Repair tool scans for keys matching `qblock:height:{0..200000}`
- Tool finds 0 matching keys
- Database opens successfully with no errors

**Questions:**
1. Can compaction delete all keys in a column family without deleting the SST files?
2. How would one inspect SST file contents directly to see what keys/values they contain?
3. Could a MANIFEST corruption cause all keys in a CF to become invisible?
4. What's the command to dump SST file contents: `sst_dump --file=000273.sst --command=scan`?

**Tools to try:**
```bash
# Inspect SST files directly
/usr/bin/sst_dump --file=./data-mine10.corrupted-missing-block2-20251111-193131/hot/000273.sst \
  --command=scan \
  --output_format=decoded_regularkey | head -100

# Check MANIFEST contents
ldb manifest_dump --db=./data-mine10.corrupted-missing-block2-20251111-193131/hot/

# List all keys in blocks column family
ldb scan --db=./data-mine10.corrupted-missing-block2-20251111-193131/hot/ \
  --column_family=blocks | head -100
```

---

### Question 2: Rust async + RocksDB Write Ordering
**For: Rust concurrency experts**

Given:
- Lock-free block producer creates blocks in parallel
- Sequential height fix ensures `current_height` advances monotonically
- qblock:latest pointer advanced to 900, but blocks 1-900 don't exist

**Questions:**
1. Is there a race condition where `put_cf()` is called but never actually writes to DB?
2. Can async tasks be cancelled/dropped after incrementing height but before writing block?
3. Would a panic in the write path cause height to advance without block storage?
4. Does RocksDB batch write API guarantee atomicity of {block write, pointer update}?

**Relevant Code Patterns:**
```rust
// Pattern 1: Sequential height advancement (v1.0.0-beta)
loop {
    let expected = self.current_height.load(Ordering::SeqCst);
    let next = expected + 1;

    // Wait for previous height to be processed
    while some_condition {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    // Advance height
    self.current_height.store(next, Ordering::SeqCst);

    // Write block to storage
    self.storage.put_block(next, block).await?; // ← Can this fail silently?
}

// Pattern 2: RocksDB write
pub async fn put_block(&self, height: u64, block: QBlock) -> Result<()> {
    let key = format!("qblock:height:{}", height);
    let value = bincode::serialize(&block)?;

    self.db.put_cf(&cf_blocks, key.as_bytes(), &value)?;

    // Update pointer
    let height_bytes = height.to_be_bytes();
    self.db.put_cf(&cf_blocks, b"qblock:latest", &height_bytes)?;

    Ok(())
}
```

**Can this lose data if:**
- Task is cancelled between height advance and storage write?
- RocksDB write fails but error is swallowed?
- WAL write succeeds but flush fails before crash?

---

### Question 3: systemd Service Restart Timing
**For: Linux systems experts**

Given:
- Emergency script ran at 19:29:02, stopped service, moved database
- systemctl shows service started at 19:34:56 (6 minutes later)
- Service may have been in zombie/defunct state
- Old process might have been writing to moved database

**Questions:**
1. Can `systemctl stop` fail to fully stop a process, leaving it in zombie state?
2. Would moving database directory while process is in stop-pending cause corruption?
3. How to verify if old process was truly dead before database move?
4. Can `mv` command partially move a database while RocksDB is still writing?

**Diagnostic Commands:**
```bash
# Check for zombie processes
ps aux | grep q-api-server | grep defunct

# Check file locks on database
lsof +D ./data-mine10/

# Check systemd stop timeout
systemctl show q-api-server | grep -i timeout

# Verify process death
systemctl status q-api-server
journalctl -u q-api-server --since "2025-11-11 19:28:00" --until "2025-11-11 19:35:00"
```

---

### Question 4: Database Forensics Strategy
**For: Data recovery experts**

Given:
- 256MB of SST files exist
- WAL files exist (15MB + 220KB)
- MANIFEST and CURRENT files intact
- But repair tool finds 0 blocks

**Questions:**
1. What's the best tool to dump raw SST file contents?
2. Can WAL files be replayed manually to recover writes?
3. How to verify if SST files contain data vs being sparse/empty?
4. Is there a RocksDB "deep fsck" that checks data integrity beyond opening DB?

**Recovery Strategy:**
```bash
# Step 1: Verify SST files aren't sparse
du -h vs ls -lh comparison
stat 000273.sst  # Check actual disk usage

# Step 2: Dump SST contents
sst_dump --file=000273.sst --command=scan --output_hex=true

# Step 3: Replay WAL manually
ldb dump_wal --walfile=000270.log --print_value > wal_contents.txt

# Step 4: Check for tombstones (deletion markers)
ldb scan --db=./data-mine10.corrupted-missing-block2-20251111-193131/hot/ \
  --hex --max_keys=1000 | grep DELETE

# Step 5: Low-level block inspection
hexdump -C 000273.sst | head -500
```

---

### Question 5: Sequential Processing Logic Verification
**For: Distributed systems experts**

Given:
- v1.0.0-beta introduced sequential height advancement
- Lock-free producers wait for height N-1 before creating height N
- Node reported height 749-900 but watchdog detected stall
- Blocks don't exist in storage

**Questions:**
1. Can the sequential wait logic deadlock and cause height to advance without blocks?
2. Is there a scenario where height pointer updates but block storage is skipped?
3. Could the watchdog's "height unchanged" detection be wrong if pointer updates but blocks don't?
4. Does the sequential fix guarantee block N exists before advancing to N+1?

**Code Review Focus:**
```rust
// CRITICAL: Does this guarantee block N exists before advancing to N+1?

// In block producer:
while !self.storage.has_block(expected_height).await? {
    tokio::time::sleep(Duration::from_millis(10)).await;
}

// Advance height
self.current_height.store(next_height, Ordering::SeqCst);

// Create new block
let block = self.create_block(next_height).await?;

// Store block
self.storage.put_block(next_height, block).await?;  // ← What if this fails?

// Question: If put_block() fails, height has already advanced!
// Result: Height 900 exists, but blocks 1-899 don't
```

---

## Filesystem Forensics

### SST File Size Analysis

**Backup Database SST Files:**
```
000047.sst:   1,560 bytes (tiny - likely metadata only)
000269.sst:   3,145 bytes (tiny - likely metadata only)
000271.sst:   2,926 bytes (tiny - likely metadata only)
000273.sst:  67,211,845 bytes (67MB - should contain MANY blocks)
000274.sst:  67,210,535 bytes (67MB - should contain MANY blocks)
000275.sst:  67,200,363 bytes (67MB - should contain MANY blocks)
000276.sst:  35,803,381 bytes (35MB - should contain MANY blocks)
000278.sst:   5,162,455 bytes (5MB - should contain hundreds of blocks)
```

**Total:** ~237MB of SST data

**Calculation:**
- Average block size: ~50KB (estimated based on transaction count)
- Expected blocks in 237MB: ~4,740 blocks
- Actual blocks found: 0

**Critical Question:** What is in these 237MB of SST files if not blocks?

**Possible Contents:**
1. **Tombstones (deletion markers):** If all blocks were deleted, SST files would contain DEL markers
2. **Other column families:** SST files are per-column-family, these should ONLY contain blocks CF data
3. **Obsolete data:** Compaction leftovers that haven't been garbage collected
4. **Binary garbage:** Corruption that looks like data but isn't valid

**Next Step:** Dump SST file contents with `sst_dump` to see actual keys/values

---

### WAL File Analysis

**000270.log: 15,477,438 bytes (15MB)**
- Last modified: Nov 11 19:07
- This is the Write-Ahead Log BEFORE emergency reset
- Should contain all uncommitted writes at time of shutdown
- **Critical:** Can this be replayed to recover blocks?

**000277.log: 220,029 bytes (220KB)**
- Last modified: Nov 11 19:34
- This is the WAL AFTER service restart
- Much smaller - represents fresh genesis attempt
- Likely contains only new writes from failed restart

**WAL Recovery Strategy:**
```bash
# Dump WAL contents
ldb dump_wal --walfile=./data-mine10.corrupted-missing-block2-20251111-193131/hot/000270.log \
  --print_value > wal_recovery.txt

# Look for block writes
grep "qblock:height" wal_recovery.txt | head -100

# Check if blocks are in WAL but not in SST
```

---

## Code Paths to Investigate

### 1. Block Write Path (crates/q-storage/src/kv.rs)

**Critical Code:**
```rust
pub async fn put_block(&self, height: u64, block: &QBlock) -> Result<()> {
    let key = format!("qblock:height:{}", height);
    let value = bincode::serialize(block)?;

    let cf_blocks = self.db.cf_handle("blocks")
        .ok_or_else(|| anyhow::anyhow!("blocks column family not found"))?;

    self.db.put_cf(&cf_blocks, key.as_bytes(), &value)?;

    // Update pointer atomically?
    let height_bytes = height.to_be_bytes();
    self.db.put_cf(&cf_blocks, b"qblock:latest", &height_bytes)?;

    Ok(())
}
```

**Questions:**
1. Are these two `put_cf()` calls atomic? If process crashes between them, what happens?
2. Does RocksDB guarantee both writes succeed or both fail?
3. Should this use a `WriteBatch` for atomicity?

---

### 2. Sequential Height Advancement (crates/q-api-server/src/lib.rs)

**Critical Code:**
```rust
// Lock-free block producer
pub async fn produce_blocks(&self) -> Result<()> {
    loop {
        let current = self.current_height.load(Ordering::SeqCst);
        let next = current + 1;

        // Wait for previous height (sequential processing)
        while !self.storage.has_block(current).await? {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        // Advance height BEFORE creating block
        if self.current_height.compare_exchange(
            current,
            next,
            Ordering::SeqCst,
            Ordering::SeqCst
        ).is_ok() {
            // Create and store block
            let block = self.create_block(next).await?;
            self.storage.put_block(next, &block).await?;
        }
    }
}
```

**BUG HYPOTHESIS:**
If `put_block()` fails (disk full, RocksDB error, permission denied), the height has ALREADY been advanced by `compare_exchange`. Next iteration will skip this height and move to next.

**Result:** Height advances 1 → 2 → 3 → 900, but only every Nth block actually gets written.

---

### 3. Has Block Check (crates/q-storage/src/kv.rs)

**Critical Code:**
```rust
pub async fn has_block(&self, height: u64) -> Result<bool> {
    let key = format!("qblock:height:{}", height);
    let cf_blocks = self.db.cf_handle("blocks")?;

    Ok(self.db.get_cf(&cf_blocks, key.as_bytes())?.is_some())
}
```

**Questions:**
1. Is this checking on-disk data or in-memory cache?
2. Could this return `true` for a block that's only in WAL, not flushed to SST?
3. If so, producer thinks block exists, advances height, but block never actually hits disk

---

## Proposed Diagnostic Steps

### Step 1: Dump SST File Contents (HIGHEST PRIORITY)

```bash
# Install RocksDB tools if not present
apt-get install rocksdb-tools

# Dump largest SST file to see what's actually in it
sst_dump --file=./data-mine10.corrupted-missing-block2-20251111-193131/hot/000273.sst \
  --command=scan \
  --output_format=decoded_regularkey \
  > sst_contents.txt

# Check first 100 keys
head -100 sst_contents.txt

# Search for block keys
grep "qblock:height" sst_contents.txt | head -50

# Count total keys
grep -c "key:" sst_contents.txt
```

**Expected Outcomes:**
- **If blocks found:** Key format mismatch or scan logic bug
- **If tombstones found:** Something deleted all blocks
- **If empty:** SST files are sparse/corrupt
- **If other data:** Wrong column family or data layout bug

---

### Step 2: Replay WAL Contents

```bash
# Dump WAL to see uncommitted writes
ldb dump_wal \
  --walfile=./data-mine10.corrupted-missing-block2-20251111-193131/hot/000270.log \
  --print_value \
  --decode_blob_index \
  > wal_contents.txt

# Search for blocks
grep "qblock:height" wal_contents.txt | head -100

# Try to manually apply WAL to database
ldb repair --db=./data-mine10.corrupted-missing-block2-20251111-193131/hot/
```

---

### Step 3: Journal Log Analysis

```bash
# Extract all logs from critical time window
journalctl -u q-api-server \
  --since "2025-11-11 19:00:00" \
  --until "2025-11-11 19:40:00" \
  > incident_logs.txt

# Look for errors during block writes
grep -E "ERROR|PANIC|CRITICAL|Failed to save block" incident_logs.txt

# Look for disk full errors
grep -E "No space left|ENOSPC|disk full" incident_logs.txt

# Look for RocksDB errors
grep -E "rocksdb|RocksDB|Corruption|IO error" incident_logs.txt
```

---

### Step 4: Binary Instrumentation Test

**Create a test harness that:**
1. Simulates the sequential height advancement logic
2. Injects failures at each step (height advance, block create, block write)
3. Verifies that height NEVER advances without block existing
4. Proves the bug can or cannot be reproduced

```rust
#[tokio::test]
async fn test_height_advancement_atomicity() {
    let storage = TestStorage::new();
    let height = AtomicU64::new(0);

    // Advance height
    height.store(1, Ordering::SeqCst);

    // Simulate write failure
    storage.put_block_with_failure(1, block).await;

    // Verify height didn't advance if write failed
    assert_eq!(height.load(Ordering::SeqCst), 0,
        "Height advanced despite write failure!");
}
```

---

## Prevention Strategies (For Future Implementation)

### 1. Atomic Height + Block Write

Use RocksDB's `WriteBatch` API to guarantee atomicity:

```rust
pub async fn put_block_atomic(&self, height: u64, block: &QBlock) -> Result<()> {
    let mut batch = rocksdb::WriteBatch::default();

    // Write block
    let block_key = format!("qblock:height:{}", height);
    let block_value = bincode::serialize(block)?;
    batch.put_cf(&cf_blocks, block_key.as_bytes(), &block_value);

    // Write pointer
    let height_bytes = height.to_be_bytes();
    batch.put_cf(&cf_blocks, b"qblock:latest", &height_bytes);

    // Atomic write
    self.db.write(batch)?;

    Ok(())
}
```

---

### 2. Height Advancement After Storage Confirmation

**Current (WRONG):**
```rust
// Advance height first
height.store(next, Ordering::SeqCst);
// Then write block (can fail)
storage.put_block(next, block).await?;
```

**Correct:**
```rust
// Write block first
storage.put_block(next, block).await?;
// Only advance height if write succeeded
height.store(next, Ordering::SeqCst);
```

---

### 3. Block Existence Verification Before Height Advance

```rust
// Before advancing from N to N+1, verify N exists on disk (not just in cache)
let block_exists = storage.has_block_flushed(current_height).await?;
if !block_exists {
    error!("Block {} missing from storage! Cannot advance height", current_height);
    return Err(anyhow::anyhow!("Block missing"));
}
```

---

### 4. Phase 10 Durability Implementation (CRITICAL)

```rust
// Force sync to disk after every block write
let mut write_options = rocksdb::WriteOptions::default();
write_options.set_sync(true);  // ← Force fsync() after every write
self.db.write_opt(batch, &write_options)?;
```

**Cost:** ~10x slower writes
**Benefit:** Guaranteed durability, no data loss on crash

---

### 5. Database Integrity Health Checks

```rust
// Every 100 blocks, verify integrity
if height % 100 == 0 {
    let integrity = storage.verify_block_continuity(0, height).await?;
    if !integrity.ok {
        error!("INTEGRITY FAILURE: Missing blocks {:?}", integrity.gaps);
        // HALT PRODUCTION until resolved
    }
}
```

---

### 6. Pre-Shutdown Flush

```rust
// In graceful shutdown handler
pub async fn shutdown_gracefully(&self) -> Result<()> {
    info!("Shutting down... flushing all data to disk");

    // Flush all column families
    for cf_name in &["blocks", "balances", "transactions"] {
        let cf = self.db.cf_handle(cf_name)?;
        self.db.flush_cf(&cf)?;
    }

    // Wait for flushes to complete
    tokio::time::sleep(Duration::from_secs(2)).await;

    info!("All data flushed, safe to exit");
    Ok(())
}
```

---

## Questions for Expert Consultation

### For RocksDB Experts:
1. How to dump SST file contents to verify what keys/values they contain?
2. Can a MANIFEST corruption cause all keys in a column family to become invisible?
3. What does it mean if SST files are 237MB but contain 0 blocks?
4. How to replay WAL file to recover uncommitted writes?
5. Can `flush_cf()` fail silently, leaving data only in WAL?

### For Rust Concurrency Experts:
1. Is there a race where height advances before block write completes?
2. Can async task cancellation cause height to advance without block storage?
3. Does `compare_exchange` + `put_cf()` need to be in a transaction for atomicity?
4. How to guarantee sequential processing AND atomic writes in lock-free architecture?

### For Distributed Systems Experts:
1. Is the sequential height advancement logic fundamentally flawed?
2. Should height pointer live in a separate atomic store outside RocksDB?
3. How do production blockchains handle height advancement atomicity?
4. What's the standard pattern for crash-safe block production?

### For Data Recovery Experts:
1. Can 237MB of SST files be recovered if repair tool finds 0 blocks?
2. What tools exist for low-level RocksDB data recovery?
3. Is the data truly lost or just inaccessible due to metadata corruption?
4. How to verify if SST files are empty vs pointer corruption?

---

## Immediate Actions Required

### 1. STOP ALL PRODUCTION OPERATIONS
- Do not restart service until root cause is understood
- Risk of repeating data loss is too high
- Need to prove the fix works before going live

### 2. FORENSICS BEFORE CLEANUP
```bash
# DO NOT DELETE ANYTHING
# Create forensic snapshot
tar -czf forensic-snapshot-20251111.tar.gz \
  data-mine10/ \
  data-mine10.corrupted-missing-block2-20251111-193131/

# Upload to secure storage
rsync -avz forensic-snapshot-20251111.tar.gz backup-server:/forensics/

# Document everything
journalctl -u q-api-server --since "2025-11-11 00:00" > full-logs.txt
```

### 3. REPRODUCE BUG IN TEST ENVIRONMENT
- Create minimal test case that reproduces data loss
- Instrument code to log every height advancement and block write
- Prove the bug before attempting fix

### 4. IMPLEMENT PHASE 10 DURABILITY
- Add `sync=true` to all RocksDB writes
- Use `WriteBatch` for atomic height + block writes
- Add integrity checks every N blocks
- Test thoroughly before deploying

### 5. CREATE MAINNET PREVENTION PLAN
- This bug would be CATASTROPHIC on mainnet
- Multiple billions of dollars could be lost
- Need defense-in-depth: backups, integrity checks, monitoring

---

## Lessons Learned

### What Went Wrong:
1. **No durability guarantees** - Blocks written to WAL but not flushed to disk before crash
2. **Height advancement not atomic** - Height advanced before confirming block storage
3. **No integrity checks** - System continued producing blocks despite missing earlier blocks
4. **Sequential fix incomplete** - Fixed race conditions but introduced new failure mode
5. **Insufficient testing** - Bug not caught in development/staging environments

### What Went Right:
1. **Sequential fix worked as designed** - Prevented race conditions in parallel producers
2. **Database structure intact** - Column families, MANIFEST, CURRENT all valid
3. **No silent corruption** - System reported stall, didn't continue with corrupt state
4. **Backup exists** - Emergency script created backup (even though it's also empty)

### Critical Realization:
**Sequential processing ≠ Atomic storage.** The v1.0.0-beta fix ensured sequential HEIGHT advancement but didn't ensure sequential STORAGE completion. This is a subtle but catastrophic distinction.

---

## Conclusion

This represents the most severe data loss incident in Q-NarwhalKnight's history. The combination of:
- Incomplete durability implementation (pre-Phase-10)
- Non-atomic height + block write operations
- Sequential processing logic that advanced height before confirming storage

Created a perfect storm where the system THOUGHT it had 900 blocks, but none actually existed on disk.

**Immediate Priority:** Dump SST file contents to determine if data is truly lost or just inaccessible.

**Long-term Priority:** Implement Phase 10 durability and atomic write semantics before any production deployment.

**Mainnet Risk:** If this bug reaches mainnet, it could cause billions in losses. Must be resolved with 100% confidence before launch.

---

**Document Status:** READY FOR EXPERT CONSULTATION
**Prepared:** 2025-11-11
**Authors:** Server Beta Incident Response Team
**Review Status:** Awaiting expert analysis from Kimi AI, DeepSeek, and ChatGPT
