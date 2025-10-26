# Balance Persistence - Root Cause Analysis & Fix History

## Date: 2025-10-25
## Issue: Critical data loss on service restart (3-40% balance loss)
## Status: v0.0.18-beta - Testing WAL-only approach

---

## Problem Evolution

The user reported losing 1-2 thousand coins (approximately 3% of total balance) after restarting the node binary via the Debian service file. After multiple fix attempts, the data loss actually INCREASED:

### Test Results Summary:

| Version | Fix Applied | Balance Before | Balance After | Loss Amount | Loss % |
|---------|-------------|----------------|---------------|-------------|--------|
| v0.0.15-beta | Added WAL fsync | 77,752 QUG | 61,036 QUG | 16,716 QUG | 21.5% |
| v0.0.16-beta | Added generic flush() | 33,381 QUG | 23,937 QUG | 9,444 QUG | 28.3% |
| v0.0.17-beta | Added CF-specific flush_cf() | 23,915 QUG | 14,430 QUG | 9,485 QUG | 39.7% |

**CRITICAL OBSERVATION**: The data loss got WORSE with each "fix", not better!

---

## Root Cause Discovery

### The Fatal Flaw in v0.0.15-17

The fixes added immediate `flush_cf()` calls after every RocksDB write:

```rust
// BUGGY APPROACH (v0.0.15-17)
async fn put_sync(&self, cf: &str, key: &[u8], value: &[u8]) -> Result<()> {
    // Step 1: Write to WAL with fsync
    self.db.put_cf_opt(&cf_handle, key, value, &write_opts)?;

    // Step 2: IMMEDIATELY flush memtable to SST  ❌ THIS WAS THE PROBLEM!
    self.db.flush_cf(&cf_handle)?;

    Ok(())
}
```

### Why This Made Things Worse

1. **WAL Premature Deletion**: When RocksDB flushes a memtable to SST files, it archives/deletes corresponding WAL entries
2. **Cross-CF WAL Sharing**: RocksDB uses a shared WAL for all column families
3. **Race Condition**: If process is killed between flush and WAL cleanup, data can be lost:
   ```
   Time 0: Write data to memtable + WAL (fsynced) ✅
   Time 1: Flush memtable to SST file ✅
   Time 2: RocksDB marks WAL entries as "safe to delete" ❌
   Time 3: Process killed (systemctl restart)
   Time 4: On restart: WAL already deleted, SST file may be incomplete → DATA LOST
   ```

### Evidence

- WAL files were only 110 bytes (too small for actual transaction data)
- `strings` search showed data existed in SST files
- But on restart, OLD data was being loaded
- This pattern matches WAL being deleted before SST flush completed

---

## v0.0.18-beta: The Correct Approach

### Key Insight

**WAL with fsync is SUFFICIENT for durability**. Immediate flushing defeats the purpose of having a WAL!

### The Fix

**File**: `crates/q-storage/src/kv.rs`

#### 1. Removed Immediate Flushes

```rust
// CORRECT APPROACH (v0.0.18)
async fn put_sync(&self, cf: &str, key: &[u8], value: &[u8]) -> Result<()> {
    let mut write_opts = rocksdb::WriteOptions::default();
    write_opts.set_sync(true); // Force fsync to disk
    write_opts.disable_wal(false); // Keep WAL enabled

    self.db.put_cf_opt(&cf_handle, key, value, &write_opts)?;

    // REMOVED flush_cf() - WAL with fsync is sufficient!
    // RocksDB will flush memtable to SST naturally, and WAL will be
    // preserved until flush completes safely.

    Ok(())
}
```

#### 2. Added WAL Preservation Options

**File**: `crates/q-storage/src/kv.rs:110-114`

```rust
// CRITICAL: WAL preservation settings for data durability
opts.set_wal_size_limit_mb(0); // Never delete WAL by size (let RocksDB manage)
opts.set_wal_ttl_seconds(0); // Never delete WAL by time
opts.set_manual_wal_flush(false); // Auto-flush WAL with set_sync(true)
```

### How It Works

1. **Write Phase**:
   - Data written to memtable (in-memory)
   - Data written to WAL (on-disk, fsynced)
   - `set_sync(true)` ensures WAL is fsynced before returning

2. **Background Flush** (RocksDB managed):
   - RocksDB flushes memtable to SST when:
     - Memtable reaches size limit (64MB)
     - Write buffer count limit reached (4 buffers)
     - Manual flush triggered by user
   - WAL preserved until flush COMPLETES

3. **Crash Recovery**:
   - On restart, RocksDB replays WAL
   - Memtable reconstructed from WAL
   - Old SST + replayed WAL = complete state ✅

### Why This Works

- ✅ WAL is fsynced to disk (survives crash)
- ✅ WAL is preserved until SST flush completes
- ✅ No race condition between flush and WAL deletion
- ✅ RocksDB WAL replay guarantees all committed data is recovered

---

## Technical Background: RocksDB Durability Model

### Write Path

```
User Write → Memtable (RAM) ┬→ WAL (disk, optional fsync)
                              └→ SST Files (disk, via background flush)
```

### Durability Levels

1. **No fsync** (default):
   - Fast writes (1-5ms)
   - OS buffers may hold data
   - Hard kill → DATA LOST

2. **WAL with fsync** (set_sync=true):
   - Slower writes (10-50ms)
   - Data on physical storage
   - Hard kill → DATA SAFE (via WAL replay)

3. **WAL + Immediate Flush** (our broken approach):
   - Slowest writes (50-200ms)
   - Creates race condition
   - Hard kill → RISK OF DATA LOSS

### The Solution

**Use #2 (WAL with fsync)** and let RocksDB manage flushes naturally.

---

## Deployment Plan (v0.0.18-beta)

### Build Commands

```bash
timeout 36000 cargo build --release --package q-storage
timeout 36000 cargo build --release --package q-api-server
```

### Testing Procedure

1. Record current wallet balance
2. Let mining run for 5-10 minutes (accumulate rewards)
3. Record new balance
4. Restart service: `systemctl restart q-api-server.service`
5. Check balance after restart
6. **Expected**: ZERO data loss (balance preserved)

### Success Criteria

- Balance after restart ≥ Balance before restart
- No "old data" being loaded
- WAL files grow to reasonable size (>1KB)
- WAL replay succeeds on restart

---

## If v0.0.18 Fails

If this approach still shows data loss, next investigations:

1. **systemd Service File**: Check `TimeoutStopSec` - may be killing process too fast
2. **Multiple Databases**: Verify only one RocksDB instance per path
3. **File System**: Check if underlying FS properly supports fsync (ext4, xfs should work)
4. **RocksDB Bug**: Investigate if specific RocksDB version has WAL replay bug
5. **Alternative**: Switch to manual checkpoint strategy with periodic full flushes

---

## Files Modified

### v0.0.18-beta Changes

1. **`crates/q-storage/src/kv.rs:292-309`** - Removed flush_cf() from put_sync()
2. **`crates/q-storage/src/kv.rs:332-354`** - Removed flush_cf() from write_batch()
3. **`crates/q-storage/src/kv.rs:110-114`** - Added WAL preservation options

---

## References

- [RocksDB WAL Documentation](https://github.com/facebook/rocksdb/wiki/Write-Ahead-Log)
- [RocksDB Durability Guide](https://github.com/facebook/rocksdb/wiki/Durability)
- [write_options.set_sync()](https://docs.rs/rocksdb/latest/rocksdb/struct.WriteOptions.html#method.set_sync)

---

**Version**: v0.0.18-beta
**Priority**: CRITICAL - Testing in progress
**Risk**: MEDIUM - Reverting to standard RocksDB pattern, but untested with our workload
