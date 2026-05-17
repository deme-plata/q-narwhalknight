# Database Corruption Root Cause Analysis

**Date**: 2025-12-07
**Version**: v1.1.9-beta
**Severity**: CRITICAL - Data Loss Risk

---

## Executive Summary

The database corruption (`qblock:latest` pointer pointing to non-existent blocks) is caused by **multiple concurrent write paths with inconsistent atomicity guarantees**. While individual write paths have been hardened, the system has THREE separate code paths that can update the height pointer, creating race conditions and data inconsistency.

---

## Root Causes Identified

### Root Cause #1: Multiple Write Paths (Design Flaw)

The system has **THREE separate ways** to write blocks to the database:

```
Path 1: BlockWriter (Single-threaded)
  ├── save_qblock() → block_writer.write_block()
  ├── Uses atomic WriteBatch for block + pointer
  └── CORRECT: Serialized, atomic

Path 2: save_qblocks_batch() (Batch sync)
  ├── Called by P2P sync handlers
  ├── Uses atomic WriteBatch for batch
  ├── Has orphan rejection (v1.0.79)
  └── MOSTLY CORRECT: But scans can race with other writes

Path 3: save_qblocks_batch_turbo() (Turbo sync)
  ├── Called by TurboSync parallel downloader
  ├── Skips orphan rejection (blocks arrive out of order)
  ├── Has single-writer lock (v1.1.0)
  └── RISKY: Turbo scan can extend pointer past actual contiguous
```

**Problem**: These paths don't share a lock. If Path 1 and Path 2 run concurrently, they can both read the same height pointer, both decide to update it, and one overwrites the other's block data without knowing.

### Root Cause #2: Turbo Scan Race Condition (v1.0.88-1.1.0)

In `save_qblocks_batch_internal()` lines 897-978, the turbo scan logic:

```rust
// Phase 1: Scan forward to find contiguous chain extent
let current_contiguous = self.height_cache.cached();
let mut scan_height = current_contiguous;

while scanned < MAX_SCAN {
    let next_height = scan_height + 1;
    match self.hot_db.get(CF_BLOCKS, height_key.as_bytes()).await {
        Ok(Some(_)) => scan_height = next_height,  // Found block
        _ => break,  // Gap - stop
    }
}

// Update pointer to scan_height
self.hot_db.put(CF_BLOCKS, b"qblock:latest", &latest_height_bytes).await?;
```

**Race Window**: Between reading `height_cache.cached()` and updating `qblock:latest`, another process could:
1. Delete blocks in that range (via chain reorganization)
2. Write blocks with different content (via P2P gossip)
3. Crash, leaving WAL unflushed

### Root Cause #3: WAL Durability Gap in Turbo Mode

```rust
// In save_qblocks_batch_internal() lines 871-884:
let use_turbo = std::env::var("Q_TURBO_SYNC")
    .map(|v| v == "1" || v.to_lowercase() == "true")
    .unwrap_or(true);  // Default: ON

if use_turbo {
    // TURBO MODE: WAL enabled, no fsync per write
    // Caller must call sync_wal() periodically
    self.hot_db.write_batch_turbo(batch).await?;
} else {
    // SAFE MODE: fsync per write
    self.hot_db.write_batch(batch).await?;
}
```

**Problem**: In turbo mode, blocks are written to WAL but not immediately synced to disk. If the system crashes:
1. Height pointer may be updated (in WAL)
2. But block data may be lost (not flushed)
3. On restart: pointer points to non-existent block

### Root Cause #4: BlockWriter Gap Detection is Incomplete

In `block_writer.rs` lines 199-210:

```rust
let should_update_pointer = if height == 0 {
    true // Always set pointer for genesis
} else if height == current_height + 1 {
    true // Normal chain extension
} else if height <= current_height {
    false // Old block or duplicate
} else {
    // Gap detected - log but don't update pointer
    warn!("Gap detected: current={}, incoming={}", current_height, height);
    false
};
```

**Problem**: This only prevents pointer advancement for blocks that would create gaps. It does NOT:
1. Verify the existing blocks at heights 0→current_height still exist
2. Prevent pointer regression if a chain reorg deletes blocks
3. Handle the case where pointer already points to missing block

### Root Cause #5: Height Cache Desync

The `height_cache` is updated in multiple places:
- After `save_qblock()` succeeds (lib.rs:637)
- After `save_qblocks_batch_turbo()` (turbo_sync.rs:1413)
- After turbo sync completes (turbo_sync.rs:2123-2124)
- During startup scan (lib.rs:1232-1234)

**Problem**: If a write path updates the cache but the database write fails or is rolled back, the cache will be ahead of the actual database state.

---

## Corruption Scenarios

### Scenario A: Parallel Sync + Block Production

```
Time    BlockProducer               TurboSync
0ms     save_qblock(height=1005)    save_qblocks_batch_turbo([1010,1011,1012])
5ms     read pointer=1004           read pointer=1004
10ms    write block 1005            write blocks 1010-1012
15ms    update pointer→1005         turbo scan: 1004→1005→GAP at 1006
20ms                                doesn't update pointer (gap detected)
25ms    reply: success              reply: success
30ms    cache update: 1005          cache update: 1012  ← DESYNC!
```

Now height_cache=1012 but pointer=1005, and blocks 1006-1009 are missing.

### Scenario B: Crash During Turbo Sync

```
Time    TurboSync                   Database State
0ms     save_qblocks_batch_turbo([1000-1100])
50ms    write_batch_turbo succeeds  WAL has data (not flushed)
60ms    update pointer→1100         pointer in WAL
70ms    update cache→1100           cache=1100
80ms    ** SYSTEM CRASH **
--- restart ---
0ms     WAL recovery                Some blocks may be lost
10ms    pointer reads 1100          But blocks 1050-1080 missing
20ms    verify_database_integrity   Detects corruption!
```

### Scenario C: P2P Block Gossip Race

```
Time    P2P Handler 1               P2P Handler 2
0ms     receive block 1006          receive block 1007
5ms     call save_qblock(1006)      call save_qblock(1007)
10ms    BlockWriter queue pos 1     BlockWriter queue pos 2
15ms    process: height=1006        (waiting in queue)
20ms    check pointer: 1005         (waiting)
25ms    height==pointer+1: ✓        (waiting)
30ms    write block 1006            (waiting)
35ms    update pointer→1006         (waiting)
40ms    reply success               process: height=1007
45ms                                check pointer: 1006
50ms                                height==pointer+1: ✓
```

In this scenario BlockWriter serializes correctly, but the issue is if blocks arrive out of order (1007 before 1006), the gap detection prevents 1007 from extending the pointer.

---

## Fixes Required

### Fix 1: Global Write Serialization (CRITICAL)

All block writes must go through a single lock:

```rust
// In QStorage:
write_lock: Arc<tokio::sync::Mutex<()>>,

pub async fn save_qblock(&self, block: &QBlock) -> Result<()> {
    let _guard = self.write_lock.lock().await;
    self.block_writer.write_block(block.clone()).await
}

pub async fn save_qblocks_batch_turbo(&self, blocks: &[QBlock]) -> Result<()> {
    let _guard = self.write_lock.lock().await;  // Same lock!
    self.save_qblocks_batch_internal(blocks, true).await
}
```

### Fix 2: Synchronous WAL Flush Before Pointer Update

```rust
// Before updating pointer:
self.hot_db.flush().await?;  // Ensure all blocks are on disk
// Then update pointer
self.hot_db.put(CF_BLOCKS, b"qblock:latest", &height_bytes).await?;
self.hot_db.flush().await?;  // Ensure pointer is on disk
```

### Fix 3: Pointer Verification on Every Read

```rust
pub async fn get_latest_qblock_height(&self) -> Result<Option<u64>> {
    let pointer = self.read_pointer().await?;
    if let Some(height) = pointer {
        // VERIFY block exists!
        let key = format!("qblock:height:{}", height);
        if self.hot_db.get(CF_BLOCKS, key.as_bytes()).await?.is_none() {
            error!("CORRUPTION: Pointer {} points to missing block!", height);
            // Auto-repair: scan backwards
            return self.repair_pointer_to_contiguous().await;
        }
    }
    Ok(pointer)
}
```

### Fix 4: Startup Integrity Check (Already Implemented but Needs Hardening)

The existing `verify_database_integrity()` should:
1. Run BEFORE any sync starts
2. Block all P2P until verification passes
3. Auto-repair if possible, or refuse to start

### Fix 5: Height Cache Must Be Database-Backed

```rust
// Don't trust the cache blindly:
pub async fn get_local_height(&self) -> u64 {
    let cached = self.height_cache.cached();

    // Periodically verify cache matches database (every 100 heights)
    if cached % 100 == 0 {
        let db_height = self.get_latest_qblock_height().await
            .unwrap_or(Some(0)).unwrap_or(0);
        if db_height != cached {
            warn!("Cache desync: cache={}, db={}", cached, db_height);
            self.height_cache.update(db_height).await;
            return db_height;
        }
    }

    cached
}
```

---

## Missing Blocks Analysis (Heights 1004-1016+)

The missing blocks at heights 1004-1016 are likely caused by:

1. **Turbo sync downloaded blocks 1017+** before blocks 1004-1016 arrived
2. **Pointer was updated to 1017+** (turbo scan found contiguous from there)
3. **Blocks 1004-1016 never arrived** (network issue) OR arrived but were rejected as "gaps"
4. **On restart, pointer pointed to 1017** but blocks 1004-1016 don't exist

To fix the current state:
```bash
# Option 1: Run gap cleanup utility
cargo run --bin cleanup-gaps -- --db-path ./data-mine1/hot

# Option 2: Re-sync from genesis (most reliable)
rm -rf ./data-mine1
# Restart node - will sync from network
```

---

## Immediate Recommendations

1. **Disable Q_TURBO_SYNC** until fixes are deployed:
   ```bash
   export Q_TURBO_SYNC=false
   ```

2. **Enable startup integrity check** (verify_database_integrity) is called

3. **Implement global write lock** across all three write paths

4. **Add periodic cache verification** against database

5. **Consider checkpoint files** that record last verified height to disk

---

## Conclusion

The corruption is a systemic issue caused by:
1. Multiple concurrent write paths without global coordination
2. Optimistic pointer updates that assume blocks exist
3. WAL durability gaps in turbo mode
4. Cache desync between height_cache and actual database

The fixes require coordinated changes across `lib.rs`, `block_writer.rs`, `turbo_sync.rs`, and startup initialization. This is a **high-priority fix** for mainnet readiness.
