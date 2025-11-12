# RocksDB Block Persistence Bug - Technical Review for AI Consultation

**Date**: November 2, 2025
**Version**: Q-NarwhalKnight v0.7.2-beta (pre-fix)
**Severity**: CRITICAL - 100% data loss on restart
**Status**: Fix implemented in v0.7.3-beta (under review)

---

## Executive Summary

A catastrophic bug was discovered where **ALL blockchain blocks are lost on service restart**, despite logs claiming successful writes. This affects 100% of nodes and prevents P2P sync from functioning. The root cause is RocksDB memtable configuration preventing flushes to persistent SST files.

---

## 1. Problem Description

### Observed Symptoms

**What Users See:**
- Node syncs to height 6,220
- Service restarts
- Node resets to height 0-12
- P2P sync fails with "No blocks found in range 13-5012"
- Infinite loop: sync → restart → data loss

**What Logs Show (Deceptive):**
```
2025-11-02T07:49:05 INFO q_storage: 💾 Saving QBlock at height 4471
2025-11-02T07:49:05 INFO q_storage: ✅ Saved QBlock 4471 in 57ms (100 mining solutions)
2025-11-02T07:44:05 INFO q_api_server: 📦 Retrieved block 6645 from RocksDB
```

**What's Actually Happening:**
```bash
$ ./repair_database ./data-mine1/hot
📊 Scan Results:
   Total blocks found: 0
   Highest block: 0
   Current pointer: 4702 (height)
   ⚠️  Pointer is WRONG! Should be 0
```

**RocksDB Internal State:**
```
# From data-mine1/hot/LOG
Flush(GB): cumulative 0.000, interval 0.000
AddFile(Total Files): cumulative 0, interval 0
AddFile(Keys): cumulative 0, interval 0
```

**Conclusion**: Blocks are written to in-memory memtables but **NEVER flushed to SST files**. On restart, all data is lost.

---

## 2. Root Cause Analysis

### 2.1 RocksDB Architecture Background

RocksDB uses a multi-tier write architecture:

```
Write Path:
┌─────────────────────────────────────────────────────────┐
│ 1. Application calls write_batch()                      │
│    ↓                                                     │
│ 2. Data goes to WAL (Write-Ahead Log) on disk          │
│    ↓                                                     │
│ 3. Data goes to MemTable (in-memory sorted buffer)     │
│    ↓                                                     │
│ 4. When MemTable fills → flush to SST file (on disk)   │
│    ↓                                                     │
│ 5. WAL can be deleted after successful SST flush       │
└─────────────────────────────────────────────────────────┘
```

**Key Point**: Data is NOT durable until it's in an SST file! WAL is only for crash recovery during active sessions.

### 2.2 The Bug: Memtable Never Triggers Flush

**File**: `crates/q-storage/src/kv.rs:247-252` (before fix)

```rust
fn create_blocks_cf() -> ColumnFamilyDescriptor {
    let mut opts = Options::default();
    opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
    opts.set_block_based_table_factory(&rocksdb::BlockBasedOptions::default());
    // NO FLUSH SETTINGS!
    ColumnFamilyDescriptor::new(CF_BLOCKS, opts)
}
```

**Default RocksDB Options** (inherited from DB-level config):
- `write_buffer_size`: 64 MB (normal mode) or 256 MB (turbo sync mode)
- `min_write_buffer_number_to_merge`: 0 (default = wait for multiple buffers)
- `max_write_buffer_number`: 4-8 buffers

**Block Characteristics**:
- Average block size: ~10 KB
- Blocks written: ~6,000 over several hours
- Total data written: 6,000 × 10 KB = **60 MB**

**The Problem**:
```
64 MB write buffer > 60 MB of blocks written
→ MemTable NEVER FILLS UP
→ Auto-flush NEVER TRIGGERS
→ Blocks stay in memory indefinitely
```

### 2.3 Why `flush_cf()` Didn't Help

**File**: `crates/q-storage/src/kv.rs:465-469` (before fix)

```rust
for cf_name in cf_names_to_flush {
    let cf_handle = self.get_cf(cf_name)?;
    self.db.flush_cf(&cf_handle)
        .context("RocksDB flush failed")?;
}
```

**The code DOES call `flush_cf()`!** So why didn't it work?

**Theory 1: Empty Memtable Optimization**
RocksDB may skip flushing if memtable is "small enough" (< some internal threshold). With only 60MB of data spread across hours, each individual flush might be skipped.

**Theory 2: Silent Failure**
The `?` operator propagates errors, but if the error is swallowed somewhere upstream, we'd never see it in logs.

**Theory 3: WAL Recovery Confusion**
RocksDB has complex WAL replay logic. If WAL is configured with `wal_size_limit_mb=0` (unlimited), it might preserve WAL indefinitely but never actually commit memtables to SST files.

**Evidence Supporting Theory 1**:
- RocksDB LOG shows **ZERO flushes** ever executed
- No error messages in application logs
- SST files exist (1.6GB) but contain ONLY metadata (manifest, payments CFs)

### 2.4 Related Configuration Issues

**File**: `crates/q-storage/src/kv.rs:87-133`

```rust
// DB-level options
let mut opts = Options::default();
opts.set_write_buffer_size(64 * 1024 * 1024); // 64MB (normal mode)
opts.set_max_write_buffer_number(4);

// WAL settings
opts.set_wal_ttl_seconds(0); // Never delete WAL by time
opts.set_wal_size_limit_mb(0); // Unlimited WAL size
opts.set_manual_wal_flush(false); // Auto-flush WAL
```

**Potential Conflict**:
- WAL is preserved indefinitely (`wal_size_limit_mb=0`)
- Memtable has 64MB buffer before flush
- RocksDB might be relying on WAL for durability instead of SST files

**This is WRONG!** WAL is only for crash recovery during an active session. After service stop/restart:
1. RocksDB replays WAL
2. If WAL is corrupted or deleted → data loss
3. Only SST files are truly durable across restarts

---

## 3. Proposed Fix (v0.7.3-beta)

### 3.1 Solution Overview

**Strategy**: Force aggressive memtable flushes for the `blocks` column family.

**Key Changes**:
1. Reduce `write_buffer_size` from 64MB → **4MB**
2. Set `min_write_buffer_number_to_merge` → **1** (flush immediately)
3. Set `max_write_buffer_number` → **2** (small buffer count)
4. Add explicit flush error logging

### 3.2 Implementation

**File**: `crates/q-storage/src/kv.rs:247-260` (v0.7.3-beta)

```rust
fn create_blocks_cf() -> ColumnFamilyDescriptor {
    let mut opts = Options::default();
    opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
    opts.set_block_based_table_factory(&rocksdb::BlockBasedOptions::default());

    // 🚨 CRITICAL FIX v0.7.3-beta: Force immediate flushes
    opts.set_write_buffer_size(4 * 1024 * 1024); // 4MB - flush after ~400 blocks
    opts.set_min_write_buffer_number_to_merge(1); // Flush immediately
    opts.set_max_write_buffer_number(2); // Small buffer count
    opts.set_disable_auto_compactions(false); // Enable auto compactions
    opts.set_level_zero_file_num_compaction_trigger(2); // Compact aggressively

    ColumnFamilyDescriptor::new(CF_BLOCKS, opts)
}
```

**Math Check**:
- 4MB buffer ÷ 10KB per block = **400 blocks per flush**
- 6,000 blocks ÷ 400 = **15 flushes total**
- Flush every ~24 minutes (assuming 1 block/6 seconds)

**File**: `crates/q-storage/src/kv.rs:473-483` (v0.7.3-beta)

```rust
for cf_name in cf_names_to_flush {
    let cf_handle = self.get_cf(cf_name)?;
    if let Err(e) = self.db.flush_cf(&cf_handle) {
        // 🚨 LOUD error logging
        warn!("❌ CRITICAL: flush_cf() failed for CF '{}': {}", cf_name, e);
        warn!("   This means data may be lost on restart!");
        return Err(e).context(format!("RocksDB flush failed for CF '{}'", cf_name));
    } else {
        debug!("✅ Flushed CF '{}' to SST successfully", cf_name);
    }
}
```

### 3.3 Expected Behavior After Fix

**Flush Frequency**:
```
Block 1-400   → Flush #1 → blocks 1-400 in SST
Block 401-800 → Flush #2 → blocks 401-800 in SST
...
Block 5801-6000 → Flush #15 → blocks 5801-6000 in SST
```

**RocksDB LOG (expected)**:
```
Flush(GB): cumulative 0.060, interval 0.004
AddFile(Total Files): cumulative 15, interval 1
AddFile(Keys): cumulative 6000, interval 400
```

**repair_database (expected)**:
```
📊 Scan Results:
   Total blocks found: 6000
   Highest block: 6000
   ✅ No gaps detected - chain is contiguous!
```

---

## 4. Questions for AI Review

### 4.1 RocksDB Configuration

**Q1**: Is our understanding of memtable flush triggers correct?
- Does RocksDB skip `flush_cf()` if memtable is below a certain size?
- What is that threshold?

**Q2**: Are there hidden flush prevention settings we're missing?
- `disable_auto_compactions` (we explicitly set to `false`)
- `allow_concurrent_memtable_write` (default?)
- `enable_write_thread_adaptive_yield` (default?)

**Q3**: Is 4MB write buffer appropriate for blockchain workloads?
- Trade-off: More frequent flushes = more SST files = slower reads
- Alternative: 8MB? 16MB?

### 4.2 WAL Configuration

**Q4**: Is our WAL configuration safe?
```rust
opts.set_wal_ttl_seconds(0); // Never delete WAL by time
opts.set_wal_size_limit_mb(0); // Unlimited WAL size
opts.set_manual_wal_flush(false); // Auto-flush WAL
```

**Q5**: Should we use `manual_wal_flush=true` and flush explicitly?
- Benefit: More control over durability
- Downside: More complexity, potential bugs

**Q6**: Does unlimited WAL size (`wal_size_limit_mb=0`) prevent memtable flushes?
- Could RocksDB be accumulating data in WAL indefinitely?
- Should we set a limit (e.g., 1GB)?

### 4.3 Write Options

**Q7**: Are our write options correct?
```rust
let mut write_opts = rocksdb::WriteOptions::default();
write_opts.set_sync(true); // Force fsync()
write_opts.disable_wal(false); // Keep WAL enabled
```

**Q8**: Does `set_sync(true)` force memtable flush OR just WAL fsync?
- Our assumption: It only fsyncs WAL to disk
- Is this correct?

### 4.4 Column Family Interactions

**Q9**: Could other CFs be interfering?
- We have 13 column families total
- `blocks` CF has minimal data (60MB over hours)
- `manifest` CF has frequent updates (compactions happening)
- Could manifest activity be preventing blocks CF from flushing?

**Q10**: Should we use a separate DB instance for blocks?
- Pro: Isolated flush behavior, no CF interference
- Con: More complexity, multiple DB handles

### 4.5 Alternative Approaches

**Q11**: Should we use manual flush after every write?
```rust
pub async fn save_qblock(&self, block: &QBlock) -> Result<()> {
    // ... write block ...
    self.hot_db.flush_cf("blocks").await?; // EXPLICIT FLUSH
    Ok(())
}
```
- Pro: Guaranteed persistence
- Con: Performance hit (fsync after every block)

**Q12**: Should we use `CompactRange` periodically?
```rust
// Force compaction every 1000 blocks
if block_height % 1000 == 0 {
    self.db.compact_range_cf(&cf_blocks, None, None);
}
```

**Q13**: Should we use `SyncWAL` + delayed flush strategy?
- Keep data in WAL with frequent fsyncs
- Flush to SST less frequently (every 10k blocks)
- Pro: Good performance + durability
- Con: Large WAL files

---

## 5. Testing Plan

### 5.1 Verification Tests

**Test 1: Immediate Flush Check**
```bash
# Start node with v0.7.3-beta
systemctl start q-api-server

# Mine 500 blocks
# Wait for "✅ Flushed CF 'blocks' to SST successfully" log

# Kill service (hard kill)
systemctl stop q-api-server

# Check persistence
./repair_database ./data-mine1/hot
# Expected: 400-500 blocks found
```

**Test 2: Long-Running Persistence**
```bash
# Run for 24 hours, mine 10k blocks
# Restart service multiple times (simulated crashes)
# Verify: repair_database finds all 10k blocks
```

**Test 3: P2P Sync Test**
```bash
# Node A: Fresh start with v0.7.3-beta (has 5000 blocks)
# Node B: Fresh start with v0.7.3-beta (height 0)
# Wait for Node B to sync from Node A
# Restart both nodes
# Verify: Both have 5000 blocks persisted
```

### 5.2 Performance Tests

**Test 4: Write Throughput**
```bash
# Measure: Blocks/second before and after fix
# Expected: Slight decrease due to more frequent flushes
# Acceptable: 10-20% slower if durability is guaranteed
```

**Test 5: Read Latency**
```bash
# Measure: API endpoint response time for block retrieval
# Expected: Slight increase due to more SST files
# Acceptable: <100ms for block reads (vs <50ms before)
```

---

## 6. Risk Assessment

### 6.1 Potential Issues with Fix

**Risk 1: Too Many SST Files**
- 15 flushes → 15 SST files per 6000 blocks
- Compaction will merge them, but there's a lag
- **Mitigation**: Set `level_zero_file_num_compaction_trigger=2` (aggressive)

**Risk 2: Write Amplification**
- More frequent flushes = more compaction work
- **Mitigation**: Acceptable trade-off for data durability

**Risk 3: Slower Performance**
- Flush overhead every 400 blocks
- **Mitigation**: Still faster than single-block writes

**Risk 4: Disk Space**
- More SST files = more disk usage (before compaction)
- **Mitigation**: Modern servers have plenty of disk space

### 6.2 Rollback Plan

If v0.7.3-beta causes performance issues:

**Option A**: Increase buffer to 8MB (flush every 800 blocks)
**Option B**: Use manual flush every N blocks instead of automatic
**Option C**: Revert to v0.7.2 and implement WAL-based recovery

---

## 7. Open Questions for Experts

1. **Why did `flush_cf()` not trigger ANY flushes in v0.7.2?**
   - Is there a RocksDB option that completely disables flushes?
   - Could it be a bug in RocksDB 0.22.0?

2. **Is 4MB buffer safe for production?**
   - Are there hidden risks with small buffers?
   - Could it cause write stalls?

3. **Should we add a `FlushOptions` parameter?**
   ```rust
   let mut flush_opts = FlushOptions::default();
   flush_opts.set_wait(true); // Block until flush completes
   flush_opts.set_allow_write_stall(false);
   self.db.flush_cf_opt(&cf_handle, &flush_opts)?;
   ```

4. **Is there a RocksDB debugging mode to trace flush decisions?**
   - Something like `ROCKSDB_LOG_LEVEL=debug` to see why flushes were skipped?

5. **Should we implement a "flush watchdog"?**
   ```rust
   // Background task that forces flush if no flush in 5 minutes
   tokio::spawn(async move {
       loop {
           tokio::time::sleep(Duration::from_secs(300)).await;
           if last_flush > 5_minutes_ago {
               warn!("⚠️  No flush in 5 minutes, forcing flush!");
               db.flush_cf(&blocks_cf)?;
           }
       }
   });
   ```

---

## 8. Conclusion

This is a **catastrophic bug** that prevents the blockchain from functioning. The fix is straightforward (reduce buffer size + explicit settings), but we want expert validation before deploying to production.

**Key Concerns**:
1. Are we missing any RocksDB configuration subtleties?
2. Will 4MB buffers cause performance issues?
3. Is there a better solution (e.g., manual flush strategy)?

**Request to AI Reviewers**:
Please review our root cause analysis and proposed fix. Point out any flaws in our understanding, suggest improvements, and validate the testing plan.

---

## Appendix A: Related Code Files

- `crates/q-storage/src/kv.rs` - RocksDB wrapper
- `crates/q-storage/src/lib.rs` - Storage engine
- `crates/q-storage/src/bin/repair_database.rs` - Diagnostic tool

## Appendix B: RocksDB Version

```toml
[dependencies]
rocksdb = { version = "0.22.0", default-features = false, features = ["lz4", "snappy"] }
```

## Appendix C: System Information

- **OS**: Linux 6.1.0-37-amd64
- **RAM**: 96GB
- **Disk**: NVMe SSD
- **RocksDB Config**: Hot DB (blocks, vertices, transactions), Cold DB (large payloads)

---

**End of Technical Review**

**Authors**: Claude (Sonnet 4.5), Q-NarwhalKnight Development Team
**Review Date**: November 2, 2025
**Next Review**: After AI consultation (DeepSeek, Grok feedback)
