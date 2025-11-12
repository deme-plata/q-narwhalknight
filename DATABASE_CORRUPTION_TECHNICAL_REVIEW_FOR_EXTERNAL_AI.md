# DATABASE CORRUPTION TECHNICAL REVIEW - External AI Analysis Request

**Date**: 2025-11-11 06:00 CET
**System**: Q-NarwhalKnight Blockchain (Quantum-resistant DAG-BFT consensus)
**Issue**: Recurring database corruption (11th occurrence)
**Version**: v0.9.92-beta (lock-free producer implementation)
**Request**: Please review this technical analysis and propose better solutions

---

## 1. SYSTEM ARCHITECTURE

### Database Stack:
- **Storage Engine**: RocksDB (embedded key-value store)
- **Configuration**:
  - Hot storage: `./data-mine9/hot/` (active blockchain data)
  - Cold storage: `./data-mine9/cold/` (archival, Narwhal mempool payloads)
  - Write-ahead logging (WAL) enabled
  - Column families: 19 total, including "blocks", "balances", "transactions"

### Block Production Architecture:
```
┌─────────────────────────────────────────────────────┐
│          Lock-Free Producer Pool (v0.9.92)         │
│                                                     │
│  ┌──────────┐  ┌──────────┐       ┌──────────┐   │
│  │Producer#0│  │Producer#1│  ...  │Producer#7│   │
│  │(channel) │  │(channel) │       │(channel) │   │
│  └────┬─────┘  └────┬─────┘       └────┬─────┘   │
│       │             │                    │         │
│       └─────────────┴────────────────────┘         │
│                     │                               │
│              BlockProducer                         │
│              └─produce_block()                     │
│                └─returns QBlock (in-memory)        │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│              main.rs (2 loops)                      │
│  ┌──────────────────────────────────────────────┐  │
│  │ Loop 1: Mining handler (mining submissions) │  │
│  │   - produce_blocks() → Vec<(id, QBlock)>    │  │
│  │   - storage.save_qblock(&block)              │  │
│  └──────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────┐  │
│  │ Loop 2: Time-based producer (every 3s)      │  │
│  │   - produce_blocks() → Vec<(id, QBlock)>    │  │
│  │   - storage.save_qblock(&block)              │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│        QStorage::save_qblock() - lib.rs:442         │
│                                                     │
│  1. Serialize block (bincode)                      │
│  2. Batch write:                                   │
│     - qblock:height:{height} → block_data          │
│     - qblock:hash:{hash} → block_data              │
│     - qblock:latest → height (pointer)             │
│  3. hot_db.write_batch(batch) ← CRITICAL           │
│  4. Log: "✅ Saved QBlock {height}"                │
└─────────────────────────────────────────────────────┘
```

---

## 2. THE RECURRING PROBLEM (11 Occurrences)

### Symptom Pattern:
1. ✅ Logs show: "✅ BLOCK PRODUCED: Height {N}"
2. ✅ Logs show: "💾 Saving QBlock at height {N}"
3. ✅ Logs show: "✅ Saved QBlock {N} in {X}ms"
4. ✅ SST files created (215+ MB of new data)
5. ❌ Database query returns: 0 blocks found
6. ❌ Pointer orphaned at height {N} but no blocks exist

### Latest Occurrence Evidence:

**Logs from Nov 11 05:31 CET:**
```
05:31:27 INFO q_api_server::block_producer: ✅ BLOCK PRODUCED: Height 7292, Hash 3ac59a2737ec20a6, Solutions 55
05:31:27 INFO q_storage: 💾 Saving QBlock at height 7292 with hash 3ac59a2737ec20a6
05:31:27 INFO q_storage: ✅ Saved QBlock 7292 in 24ms (55 mining solutions)
(Repeated for blocks 7290-7293 across 8 producers)
```

**SST Files Created:**
```bash
-rw-r--r-- 1 root root  65M Nov 11 05:31 data-mine9/hot/147419.sst
-rw-r--r-- 1 root root 121M Nov 11 05:31 data-mine9/hot/147420.sst
-rw-r--r-- 1 root root  29M Nov 11 05:31 data-mine9/hot/147421.sst
-rw-r--r-- 1 root root 1.4K Nov 11 05:31 data-mine9/hot/147423.sst
-rw-r--r-- 1 root root 1.1K Nov 11 05:31 data-mine9/hot/147426.sst
Total: 215+ MB of new data written to disk
```

**Database Query Result:**
```bash
$ ./target/release/repair-database ./data-mine9/hot

📂 Opening database: ./data-mine9/hot
📋 Found 19 column families

🔍 Scanning for blocks (this may take time)...
   Scanning height 0...
   Scanning height 10000...

Total blocks found: 0
Highest block: 0
Current pointer: 7293 (ORPHANED)
⚠️ Pointer is WRONG! Should be 0
```

---

## 3. SUSPECTED ROOT CAUSES

### Hypothesis A: RocksDB Compaction Deletion
**Theory**: Force-kill interrupted write → corrupt state → compaction deletes blocks as "garbage"

**Evidence**:
- Service was force-killed (kill -9 PID 2395067) at ~05:11 CET
- Blocks 7290-7293 were created AFTER restart at 05:31 CET
- SST files exist but repair tool finds 0 blocks
- RocksDB compaction may have deleted blocks thinking they were orphaned

**Problems with this theory**:
1. Fresh blocks created 20 minutes AFTER restart shouldn't be affected by pre-restart corruption
2. Compaction typically merges/consolidates, not deletes all data
3. WAL replay should restore uncommitted writes
4. Multiple SST files created suggests successful writes

### Hypothesis B: Column Family Mismatch
**Theory**: Blocks written to wrong column family

**Evidence**:
- `save_qblock()` writes to `CF_BLOCKS = "blocks"` (lib.rs:78)
- `repair-database` reads from `CF_BLOCKS = "blocks"` (repair_database.rs:11)
- Same constant used, should be consistent

**Problems with this theory**:
1. Constants match exactly (`"blocks"`)
2. Would cause consistent failure, not intermittent
3. Doesn't explain why it works sometimes

### Hypothesis C: Serialization Format Incompatibility
**Theory**: Write uses bincode, read uses different format

**Evidence**:
```rust
// lib.rs:453 - Writing:
let block_data = bincode::serialize(block)
    .context("Failed to serialize QBlock")?;

// Key format: "qblock:height:{height}"
batch.push((CF_BLOCKS, height_key.into_bytes(), block_data.clone()));
```

**Problems with this theory**:
1. Same serialization library (bincode) used throughout
2. Would cause deserialization errors, not missing data
3. Repair tool doesn't attempt deserialization, just checks key existence

### Hypothesis D: Async Flush Failure
**Theory**: `write_batch()` returns success but data never flushed to disk

**Evidence**:
```rust
// lib.rs:472-473
self.hot_db.write_batch(batch).await
    .context("Failed to write QBlock batch to database")?;
```

**Critical Questions**:
1. Does `write_batch()` wait for disk sync or just WAL write?
2. Is there a race between write_batch() and process termination?
3. Are SST files created but not yet indexed by RocksDB manifest?

**Supporting Evidence**:
- Logs show "✅ Saved QBlock" (write_batch succeeded)
- SST files exist (data written to disk)
- Repair tool finds nothing (not indexed in manifest?)

### Hypothesis E: Database Path Mismatch
**Theory**: Writing to one database, reading from another

**Evidence**:
- Service uses `Q_DB_PATH=./data-mine9` (systemd environment)
- Repair tool uses `./data-mine9/hot` (explicit argument)
- Both should point to same RocksDB instance

**Problems with this theory**:
1. SST files in correct directory
2. 19 column families found (correct database)
3. Just blocks missing, other data intact

---

## 4. ROCKSDB WRITE_BATCH IMPLEMENTATION

### Current Code (lib.rs:442-490):
```rust
pub async fn save_qblock(&self, block: &q_types::block::QBlock) -> Result<()> {
    let start_time = SystemTime::now();
    let block_hash = block.calculate_hash();

    info!("💾 Saving QBlock at height {} with hash {}",
          block.header.height, hex::encode(&block_hash[..8]));

    // Serialize block
    let block_data = bincode::serialize(block)
        .context("Failed to serialize QBlock")?;

    // Prepare batch writes for atomic storage
    let mut batch = Vec::new();

    // Store by height: qblock:height:{height}
    let height_key = format!("qblock:height:{}", block.header.height);
    batch.push((CF_BLOCKS, height_key.into_bytes(), block_data.clone()));

    // Store by hash: qblock:hash:{hash_hex}
    let hash_key = format!("qblock:hash:{}", hex::encode(block_hash));
    batch.push((CF_BLOCKS, hash_key.into_bytes(), block_data.clone()));

    // Store latest height pointer: qblock:latest
    let latest_height_bytes = block.header.height.to_be_bytes().to_vec();
    batch.push((CF_BLOCKS, b"qblock:latest".to_vec(), latest_height_bytes));

    // Commit atomically
    self.hot_db.write_batch(batch).await
        .context("Failed to write QBlock batch to database")?;

    let latency = start_time.elapsed().unwrap_or(Duration::from_millis(0));

    info!("✅ Saved QBlock {} in {}ms ({} mining solutions)",
          block.header.height, latency.as_millis(), block.mining_solutions.len());

    Ok(())
}
```

### KV Layer (kv.rs - write_batch implementation):
```rust
pub async fn write_batch(&self, batch: Vec<(String, Vec<u8>, Vec<u8>)>) -> Result<()> {
    let db = self.db.clone();

    tokio::task::spawn_blocking(move || {
        let mut batch = rocksdb::WriteBatch::default();

        for (cf_name, key, value) in batch_iter {
            let cf = db.cf_handle(&cf_name)
                .ok_or_else(|| anyhow::anyhow!("Column family {} not found", cf_name))?;
            batch.put_cf(cf, key, value);
        }

        db.write(batch)?; // ← CRITICAL: Does this sync to disk?
        Ok(())
    })
    .await?
}
```

### RocksDB Write Options (UNKNOWN):
- **Question 1**: Are write options configured for durability?
- **Question 2**: Is WAL sync enabled (`sync: true`)?
- **Question 3**: Is manual flush required after write_batch?
- **Question 4**: Does `db.write()` wait for disk fsync or just WAL write?

---

## 5. CRITICAL TIMING ANALYSIS

### Event Timeline (Nov 11, 2025):

```
04:50 CET - Service running v0.9.92-beta (previous binary)
          - Blocks 6929-6937 being produced
          - Memory: 14.9 GB (high, suggesting issue)

05:04 CET - Compilation of new v0.9.92-beta completed
          - Binary size: 121 MB
          - Includes lock-free producer fixes

05:11 CET - Service force-killed (kill -9 PID 2382289)
          - SIGKILL sent (no graceful shutdown)
          - Database in middle of writes?
          - WAL may have uncommitted data

05:12 CET - New service started (PID 2395067)
          - Lock-free producer pool initialized
          - 8 producers spawned successfully
          - Memory dropped to 261 MB (good sign)

05:31 CET - Blocks 7290-7293 created by all 8 producers
          - 24 total blocks created (8 producers × 3 rounds)
          - All logged as "✅ Saved QBlock"
          - 215 MB of SST files created
          - BUT: repair tool finds 0 blocks

05:50 CET - Database corruption discovered
          - Running repair tool showed 0 blocks
          - Pointer orphaned at 7293
          - All blocks missing from "blocks" column family
```

### Critical Question:
**How can blocks created 20 minutes AFTER a clean restart be affected by pre-restart corruption?**

---

## 6. PARALLEL ISSUES

### Issue 1: Parallel Block Production
- 8 producers creating blocks simultaneously
- All 8 producers create blocks at same height (e.g., 7292)
- Each with different hash (different producer_id field)
- Batch writes happening in parallel

**Potential Race Condition**:
```
Producer #0: save_qblock(height=7292, hash=3ac59a27) → write_batch([...])
Producer #1: save_qblock(height=7292, hash=28d45789) → write_batch([...])
Producer #2: save_qblock(height=7292, hash=6bde3e38) → write_batch([...])
...all happening simultaneously...

Question: Can parallel write_batch calls corrupt the database?
Question: Does RocksDB handle concurrent writes to same key gracefully?
Question: Is the "qblock:latest" pointer being overwritten 8 times simultaneously?
```

### Issue 2: Error Handling
```rust
// main.rs:4570-4572
if let Err(e) = app_state_block_producer.storage_engine.save_qblock(&new_block).await {
    error!("❌ Failed to save block {}: {}", new_block.header.height, e);
}
// ERROR IS LOGGED BUT EXECUTION CONTINUES!
```

**Problem**: If `save_qblock()` returns error, it's logged but:
- Block is still broadcast to network
- Node status updated with new height
- Balance updates processed
- But block NOT in database!

**No error logged in this case**, suggesting:
- Either `save_qblock()` succeeded (returned Ok)
- Or error logging failed completely

---

## 7. ALTERNATIVE STORAGE IMPLEMENTATION

### There are TWO `save_qblock` implementations:

#### Implementation A: lib.rs:442 (Currently Used)
```rust
pub async fn save_qblock(&self, block: &QBlock) -> Result<()> {
    // Serialize with bincode
    let block_data = bincode::serialize(block)?;

    // Batch write to CF_BLOCKS
    // Keys: qblock:height:{h}, qblock:hash:{h}, qblock:latest
    self.hot_db.write_batch(batch).await?;
}
```

#### Implementation B: transaction.rs:161 (Transactional)
```rust
pub async fn save_qblock(&self, block: &QBlock) -> Result<()> {
    // Serialize with postcard (different format!)
    let block_bytes = postcard::to_allocvec(block)?;

    // Direct put operations (not batch)
    self.put("blocks", &height_key, &block_bytes).await?;
    self.put("block_hash_to_height", &block_hash, &height_key).await?;

    // Conditional pointer update (prevents gaps!)
    let current_pointer = self.get_current_height_from_pointer().await?;
    if block.header.height == 0 || block.header.height == current_pointer + 1 {
        self.put("blocks", b"qblock:latest", &height_key).await?;
    }
}
```

**Critical Difference**:
- Implementation A: Always updates pointer (can skip blocks)
- Implementation B: Only updates pointer if contiguous (prevents gaps)
- Implementation A: Uses bincode serialization
- Implementation B: Uses postcard serialization
- Implementation A: Batch write
- Implementation B: Individual puts

**Question**: Is the WRONG implementation being called? Could this explain inconsistency?

---

## 8. ROCKSDB CONFIGURATION

### Database Initialization (kv.rs):
```rust
let mut opts = Options::default();
opts.create_if_missing(true);
opts.create_missing_column_families(true);

// QUESTION: Where are durability settings?
// - WAL sync mode?
// - Flush policy?
// - Compaction settings?
// - Write buffer size?

let db = DB::open_cf(&opts, path, &column_families)?;
```

### Missing Configuration (Suspected):
1. **Write Options**:
   - `WriteOptions::sync(true)` - Force fsync on write?
   - `WriteOptions::disable_wal(false)` - Ensure WAL enabled?

2. **Flush Policy**:
   - Manual flush after critical writes?
   - `db.flush_cf()` after block writes?

3. **Durability Settings**:
   - `Options::paranoid_checks(true)` - Detect corruption early?
   - `Options::max_open_files()` - Prevent file handle exhaustion?

---

## 9. REPAIR TOOL ANALYSIS

### Repair Tool Logic (repair_database.rs):
```rust
const CF_BLOCKS: &str = "blocks";

// Scan for blocks
for height in 0..100_000 {
    let key = format!("qblock:height:{}", height);
    if let Ok(Some(_)) = db.get_cf(&cf_blocks, key.as_bytes()) {
        total_blocks += 1;
        highest_found = height;
    }
}
```

**Findings**:
- Scans up to height 100,000
- Looks for keys: `"qblock:height:0"`, `"qblock:height:1"`, etc.
- Uses same CF_BLOCKS constant as writer
- Just checks key existence (doesn't deserialize)
- Logs show: scanned height 0, 10000, 20000, ... all empty

**Critical Observation**:
- If serialization was the issue → keys would exist but values corrupt
- If column family was wrong → CF not found error
- If path was wrong → different database opened
- Current result: Keys simply don't exist → data was deleted or never written

---

## 10. MEMORY ANALYSIS

### Before Force-Kill (05:11 CET):
```
Memory: 14.9 GB (VERY HIGH - normal is ~500 MB)
Blocks: 6929-6937 being created
Issue: Memory exhaustion risk
```

### After Restart (05:12 CET):
```
Memory: 261 MB (normal)
Lock-free producers: 8 spawned successfully
Block production: resumed normally
```

**Question**: Was the 14.9 GB memory caused by:
1. Unbounded channel growth (pre-v0.9.92)?
2. Memory-mapped SST files?
3. Uncommitted RocksDB buffers?
4. Memory leak in block producer?

**Hypothesis**: High memory → forced OS-level eviction → database corruption?

---

## 11. PREVIOUS FIXES THAT FAILED

### Fix Attempt 1: Pointer-based repair (v0.9.29)
```rust
// Only update pointer if block extends contiguous chain
if block.header.height == current_pointer + 1 {
    self.put("blocks", b"qblock:latest", &height_key).await?;
}
```
**Result**: Still had corruption

### Fix Attempt 2: Atomic batch writes (v0.9.0)
```rust
// Use batch writes for atomicity
self.hot_db.write_batch(batch).await?;
```
**Result**: Still had corruption

### Fix Attempt 3: Graceful shutdown timeout (v0.9.76)
```systemd
TimeoutStopSec=300
KillMode=mixed
```
**Result**: Still had corruption (when force-killed)

### Fix Attempt 4: Lock-free producers (v0.9.92)
```rust
// Eliminate deadlocks with channel-based architecture
let (command_tx, command_rx) = mpsc::channel(CHANNEL_CAPACITY);
```
**Result**: Deadlock fixed, but corruption persists

---

## 12. QUESTIONS FOR EXTERNAL AI ANALYSIS

### Critical Questions:

1. **RocksDB Write Guarantees**:
   - Does `db.write(batch)` guarantee durability or just WAL write?
   - Do we need explicit `db.flush_cf()` after `write()`?
   - Can parallel `write_batch()` calls corrupt the database?

2. **Force-Kill Impact**:
   - Why would blocks created 20 minutes AFTER restart be missing?
   - Can pre-restart corruption affect post-restart writes?
   - Does RocksDB manifest corruption cause "invisible" writes?

3. **Parallel Block Production**:
   - 8 producers writing different blocks at same height - safe?
   - Is overwriting `qblock:latest` pointer 8 times/second problematic?
   - Should we serialize block writes (defeating parallelism purpose)?

4. **Compaction Theory**:
   - Can RocksDB compaction delete recently written data?
   - Would compaction leave SST files but remove keys from manifest?
   - How to prevent compaction from deleting valid data?

5. **Alternative Explanations**:
   - Could this be a kernel-level issue (filesystem, memory)?
   - Could ContaboVPS have disk issues (silent data loss)?
   - Could high memory (14.9GB) have caused OS-level corruption?

---

## 13. REPRODUCTION STEPS

### Minimal Reproduction (Theoretical):

```bash
# 1. Start node with high memory pressure
Q_DB_PATH=./test-db cargo run --release --bin q-api-server

# 2. Trigger parallel block production
# (Submit thousands of mining solutions)
for i in {1..10000}; do
    curl -X POST http://localhost:8080/api/v1/mining/submit -d '{...}'
done

# 3. Force kill while blocks are being written
kill -9 $(pgrep q-api-server)

# 4. Restart and check database
cargo run --bin repair-database ./test-db/hot

# Expected: Blocks missing but SST files exist
```

### Observed Inconsistency:
- Sometimes corruption happens immediately after force-kill
- Sometimes corruption happens hours later during normal operation
- No clear pattern to when/why it occurs

---

## 14. PROPOSED SOLUTIONS (Need Validation)

### Solution A: Force Disk Sync
```rust
pub async fn save_qblock(&self, block: &QBlock) -> Result<()> {
    // Write batch
    self.hot_db.write_batch(batch).await?;

    // FORCE FLUSH TO DISK (new)
    self.hot_db.flush_cf(CF_BLOCKS).await?;

    Ok(())
}
```
**Pros**: Guarantees durability
**Cons**: 10x slower writes (24ms → 240ms per block)

### Solution B: Write-Ahead Log (Application-Level)
```rust
// Before writing to RocksDB:
self.write_to_wal(block)?; // Append-only file
self.hot_db.write_batch(batch).await?;
self.mark_wal_committed(block.header.height)?;

// On startup:
self.replay_uncommitted_wal_entries()?;
```
**Pros**: Recoverable on crash
**Cons**: Complex implementation, double storage

### Solution C: Serialize Block Writes
```rust
// Single writer task (defeats parallelism)
let (block_tx, mut block_rx) = mpsc::channel(1000);

tokio::spawn(async move {
    while let Some(block) = block_rx.recv().await {
        storage.save_qblock(&block).await?;
    }
});
```
**Pros**: No parallel write conflicts
**Cons**: Bottleneck (100 blocks/sec max)

### Solution D: Use Transactional Implementation
```rust
// Switch from lib.rs:save_qblock to transaction.rs:save_qblock
// Uses proper transactions with MVCC
let tx = storage.begin_transaction().await?;
tx.save_qblock(&block).await?;
tx.commit().await?;
```
**Pros**: ACID guarantees
**Cons**: Slower, more complex

### Solution E: Database Replication
```rust
// Write to multiple databases simultaneously
self.primary_db.save_qblock(&block).await?;
self.replica_db.save_qblock(&block).await?;

// On corruption: restore from replica
```
**Pros**: Backup on corruption
**Cons**: 2x storage, 2x write time

---

## 15. REQUEST TO EXTERNAL AI

Please analyze this technical review and provide:

1. **Root Cause Identification**:
   - What is the ACTUAL root cause of recurring corruption?
   - Why do blocks created post-restart go missing?
   - Is this a RocksDB bug, configuration issue, or application bug?

2. **Specific Code Issues**:
   - Review the `save_qblock()` implementation
   - Identify race conditions in parallel block production
   - Point out any RocksDB anti-patterns

3. **Recommended Solution**:
   - What is the BEST fix (not just a workaround)?
   - How to prevent this from ever happening again?
   - Should we switch storage engines entirely (e.g., PostgreSQL)?

4. **Testing Strategy**:
   - How to reliably reproduce this issue?
   - What tests would catch this before production?
   - How to verify the fix actually works?

5. **Production Mitigation**:
   - What should we do RIGHT NOW to prevent data loss?
   - Should we add hourly backups?
   - Should we disable parallel block production?

---

## 16. ADDITIONAL CONTEXT

### System Information:
- **OS**: Debian Linux 6.1.0-37-amd64
- **VPS**: Contabo Cloud VPS (potential disk I/O issues?)
- **RocksDB Version**: (check Cargo.lock)
- **Rust Version**: 1.70+
- **Service Management**: systemd with TimeoutStopSec=300

### Recent Changes:
- v0.9.90-beta: Phase 9 network transition
- v0.9.92-beta: Lock-free producer implementation (eliminated deadlocks)
- Parallel block production: 8 producers writing simultaneously
- High memory usage before crash: 14.9 GB (normally 500 MB)

### Previous Occurrences:
1. v0.5.23-beta: First occurrence (sync-down data loss)
2. v0.6.6-beta: Crash recovery bug
3. v0.7.3-beta: RocksDB persistence fix
4. v0.8.2-beta: Balances column family fix
5. v0.9.0-beta: Emergency deployment (height reset)
6. v0.9.22-beta: Catastrophic data loss
7. v0.9.29-beta: Pointer race condition fix
8. v0.9.56-beta: BlockPack deserialization corruption
9. v0.9.59-beta: Genesis gap fix
10. v0.9.76-beta: Recurring data loss
11. v0.9.92-beta: **THIS OCCURRENCE** (blocks missing post-restart)

**Pattern**: Every "fix" has failed. The issue keeps recurring with different symptoms.

---

## 17. FILES FOR REFERENCE

### Key Source Files:
1. `crates/q-storage/src/lib.rs:442-490` - save_qblock() implementation
2. `crates/q-storage/src/kv.rs` - RocksDB write_batch() wrapper
3. `crates/q-storage/src/transaction.rs:161-192` - Alternative save_qblock()
4. `crates/q-api-server/src/main.rs:4522-4672` - Block production loop
5. `crates/q-storage/src/bin/repair_database.rs` - Database repair tool
6. `/etc/systemd/system/q-api-server.service` - Service configuration

### Log Excerpts:
Available in journalctl from Nov 11 04:50-05:50 CET

### Database Structure:
- 19 column families
- 1.1 GB total size
- 280 MB in SST files created at 05:31
- 0 blocks found in "blocks" CF

---

## 18. URGENCY AND IMPACT

- **Severity**: CRITICAL - Data loss on production blockchain
- **Frequency**: 11 occurrences over ~6 months
- **User Impact**: Miners lose rewards, chain requires resyncing
- **Business Impact**: Cannot launch mainnet with this issue
- **Current Status**: Development blocked until resolved

**This is not acceptable for a production blockchain. We need a definitive solution.**

---

**Please provide your analysis and recommendations. Thank you.**

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>
