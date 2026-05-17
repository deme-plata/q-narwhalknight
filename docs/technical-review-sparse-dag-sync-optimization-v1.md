# Technical Review: Sparse DAG Sync Optimization
## From 124 blocks/sec to 1000+ blocks/sec on Sparse Block Ranges
### Date: 2026-04-19 | Status: OPTIMIZATION NEEDED | Risk: ZERO

---

## 1. The Problem

The DAG block fix is working — new nodes now start syncing from height 75K instead of 14M. But sync speed through the DAG range is **124 blocks/sec** vs the expected **1000+ blocks/sec**.

Root cause: turbo sync requests blocks in fixed 1000-height chunks. For sparse DAG blocks (1 block per ~500 heights), each chunk returns 0-2 blocks. The node spends 99% of its time requesting empty ranges.

```
CURRENT BEHAVIOR (slow):
Request:  blocks 100000-101000  → 2 blocks found, 998 empty
Request:  blocks 101000-102000  → 0 blocks found, 1000 empty
Request:  blocks 102000-103000  → 0 blocks found, 1000 empty
Request:  blocks 103000-104000  → 1 block found,  999 empty
... 15,000 empty requests to sync 545K blocks across 15M heights ...

DESIRED BEHAVIOR (fast):
Request:  "give me next 200 blocks after height 100000"
Response: 200 blocks at heights 100441, 100442, 100443, ..., 115234
          (jumps over gaps automatically)
```

---

## 2. Current Architecture

### Syncing Node (Requesting Blocks)

```rust
// turbo_sync.rs — current chunk strategy
let chunks: Vec<(u64, u64)> = (start..end)
    .step_by(chunk_size)           // Fixed 1000-height steps
    .map(|h| (h, h + chunk_size))
    .collect();

for (from, to) in chunks {
    // P2P request: "give me blocks from height {from} to {to}"
    let blocks = peer.request_blocks(from, to).await;
    // Returns 0-2 blocks for sparse DAG range
    // Wastes a full network round-trip for each empty chunk
}
```

### Serving Node (Providing Blocks)

```rust
// get_qblocks_range() — current serving strategy
pub async fn get_qblocks_range(&self, start_height: u64, limit: usize) -> Vec<QBlock> {
    // Step 1: Try qblock:height:{N} for each height in range
    let keys: Vec<_> = (start..=end)
        .map(|h| format!("qblock:height:{}", h))
        .collect();
    let results = self.hot_db.multi_get(CF_BLOCKS, &keys).await;
    
    // Step 2: For missing heights, try qblock:dag:{N}: prefix
    for height in missing_heights {
        let entries = self.hot_db.scan_prefix_seek(
            CF_BLOCKS, 
            format!("qblock:dag:{}:", height).as_bytes(), 
            1
        ).await;
        // Per-height seek — O(log N) each, but 1000 of them = slow
    }
}
```

### Why It's Slow

| Operation | Count per chunk | Time per op | Total per chunk |
|-----------|----------------|-------------|-----------------|
| multi_get (1000 height keys) | 1 | ~5ms | 5ms |
| scan_prefix_seek (per missing height) | ~998 | ~0.5ms | 500ms |
| Network round-trip | 1 | ~30ms | 30ms |
| **Total per chunk** | | | **~535ms** |
| **Blocks found per chunk** | | | **~2** |
| **Effective speed** | | | **~4 blocks/sec** |

The actual 124 b/s is higher because some chunks near DAG-dense heights return more blocks. But for the majority of the range, it's single-digit blocks per second.

---

## 3. The Fix: Seek-Forward Serving

Instead of the syncing node requesting fixed height ranges, the serving node should **seek forward** from the requested start height and return the next N blocks regardless of height gaps.

### Option A: Change the Serving Path (RECOMMENDED)

Add a new mode to `get_qblocks_range` that scans forward through all key formats:

```rust
/// Serve the next `limit` blocks starting at or after `start_height`.
/// Seeks forward through all key formats (height, DAG, binary),
/// skipping gaps automatically. For sparse DAG ranges, this is
/// O(1 seek + limit iterations) instead of O(range × seek).
pub async fn get_qblocks_forward(
    &self, 
    start_height: u64, 
    limit: usize
) -> Result<Vec<QBlock>> {
    let mut blocks = Vec::with_capacity(limit);
    
    // Strategy: seek to "qblock:dag:{start_height}:" and iterate forward
    // collecting the next `limit` deserializable blocks.
    // This is ONE seek + forward iteration — same as ldb scan.
    
    let seek_key = format!("qblock:dag:{}:", start_height);
    let cf_handle = self.hot_db.get_cf(CF_BLOCKS)?;
    
    let iter = self.hot_db.db.iterator_cf(
        &cf_handle,
        rocksdb::IteratorMode::From(seek_key.as_bytes(), rocksdb::Direction::Forward),
    );
    
    let mut seen_heights: HashSet<u64> = HashSet::new();
    
    for item in iter {
        let (key, value) = item?;
        let key_str = String::from_utf8_lossy(&key);
        
        // Only process qblock:dag: keys
        if !key_str.starts_with("qblock:dag:") {
            break; // Passed the DAG key range
        }
        
        // Parse height from "qblock:dag:{height}:{proposer}"
        let parts: Vec<&str> = key_str.split(':').collect();
        if parts.len() < 3 { continue; }
        let height: u64 = match parts[2].parse() {
            Ok(h) => h,
            Err(_) => continue,
        };
        
        // Skip if below our start (string sort can land before numeric start)
        if height < start_height { continue; }
        
        // Deduplicate — only one block per height
        if seen_heights.contains(&height) { continue; }
        
        // Deserialize (with old format fallback)
        match deserialize_qblock_with_fallback(&value) {
            Ok(block) => {
                seen_heights.insert(height);
                blocks.push(block);
                if blocks.len() >= limit {
                    break;
                }
            }
            Err(_) => continue, // Skip unreadable blocks
        }
    }
    
    // Also check qblock:height: keys in the same range
    // (for heights where turbo-sync blocks exist alongside DAG blocks)
    if !blocks.is_empty() {
        let max_height = blocks.last().unwrap().header.height;
        let height_blocks = self.get_qblocks_range(start_height, 
            (max_height - start_height + 1) as usize).await?;
        
        for block in height_blocks {
            if !seen_heights.contains(&block.header.height) {
                seen_heights.insert(block.header.height);
                blocks.push(block);
            }
        }
        
        blocks.sort_by_key(|b| b.header.height);
        blocks.truncate(limit);
    }
    
    Ok(blocks)
}
```

### Performance Comparison

| Strategy | Seek ops | Iterations | Network RTT | Blocks per RTT | Effective speed |
|----------|----------|-----------|-------------|----------------|-----------------|
| Current (per-height) | 1000/chunk | 0 | 1 | 0-2 | ~4 b/s |
| **Forward-seek** | **1/request** | **200** | **1** | **200** | **1000+ b/s** |

The forward-seek does ONE RocksDB seek to position the iterator at the first DAG key at or after the requested height, then walks forward collecting blocks. No empty probes, no wasted round-trips.

### Option B: Change the Syncing Strategy

Instead of fixed-height chunks, the syncing node sends "give me next N blocks after height X":

```rust
// turbo_sync.rs — sparse-aware sync
// Instead of: request_blocks(100000, 101000)  // "blocks 100K to 101K"
// Do:         request_blocks_forward(100000, 200)  // "next 200 blocks after 100K"

// The serving node responds with blocks at heights:
// 100441, 100442, 100443, ..., 115234  (wherever they exist)
// The syncing node sets its cursor to the last received height + 1
```

This requires a new P2P message type or a flag on the existing block-pack request.

### Option C: Combine Both (Best Performance)

1. Serving node implements `get_qblocks_forward()` 
2. Sync/blocks API endpoint gets a `mode=forward` parameter
3. Checkpoint sync probe uses `mode=forward` automatically for sparse ranges
4. Turbo sync detects sparse ranges (>50% empty responses) and switches to forward mode

---

## 4. Implementation Plan

### Phase 1: Forward-Seek on Serving Node (Today)

Add `get_qblocks_forward()` to StorageEngine. Wire it to the sync/blocks API:

```
GET /api/v1/sync/blocks?from_height=100000&limit=200&mode=forward
```

When `mode=forward`, the endpoint calls `get_qblocks_forward()` instead of `get_qblocks_range()`. Backward compatible — existing requests without `mode` use the current behavior.

**Files to modify:**

| File | Change |
|------|--------|
| `crates/q-storage/src/lib.rs` | Add `get_qblocks_forward()` |
| `crates/q-api-server/src/handlers.rs` | Add `mode` param to sync_blocks handler |

### Phase 2: P2P Block-Pack Forward Mode (This Week)

Add forward mode to the P2P block-pack request/response:

```rust
// In the block-pack request message:
struct BlockPackRequest {
    from_height: u64,
    limit: u32,
    mode: SyncMode,  // NEW: Range (default) or Forward
}

enum SyncMode {
    Range,    // Current: blocks from_height to from_height+limit
    Forward,  // New: next `limit` blocks at or after from_height
}
```

Backward compatible — old nodes ignore the `mode` field and serve Range by default.

### Phase 3: Auto-Detection (Next Release)

Turbo sync automatically switches to Forward mode when it detects sparse ranges:

```rust
// In turbo_sync chunk processing:
if consecutive_sparse_responses > 5 {
    // Switch from Range to Forward mode
    info!("Detected sparse block range — switching to forward-seek sync");
    sync_mode = SyncMode::Forward;
}
```

---

## 5. Expected Performance After Fix

| Height Range | Block Density | Current Speed | After Fix |
|-------------|--------------|---------------|-----------|
| 75K - 400K | ~1 per 1 height (dense DAG) | 400+ b/s | 1000+ b/s |
| 400K - 1M | ~1 per 100 heights (sparse) | ~10 b/s | 1000+ b/s |
| 1M - 10M | ~1 per 500 heights (very sparse) | ~2 b/s | 500+ b/s |
| 10M - 15M | gap (deleted qblock:height:) | 0 b/s | 0 b/s (no data) |
| 15M - 15.8M | continuous qblock:height: | 1000+ b/s | 1000+ b/s (unchanged) |

The forward-seek eliminates empty probes entirely. Each network round-trip returns a full batch of real blocks regardless of height gaps.

---

## 6. String Sort Caveat

RocksDB sorts keys lexicographically. `"qblock:dag:2000000"` sorts AFTER `"qblock:dag:19999999"` because `'2' > '1'`. This means a forward iterator starting at `"qblock:dag:100000:"` will traverse:

```
qblock:dag:100000:...   ← start here
qblock:dag:100441:...   ← found
qblock:dag:100442:...   ← found
...
qblock:dag:1015441:...  ← found (1M, sorts after 100K because "10" < "2")
qblock:dag:1015487:...  ← found
...
qblock:dag:19999999:... ← all "1*" heights exhausted
qblock:dag:2000000:...  ← NOW we reach 2M heights
qblock:dag:2000941:...  ← found
...
qblock:dag:9027291:...  ← last DAG block
```

This means the iterator walks ALL DAG keys in string-sort order, NOT numeric order. For the forward-seek optimization, this is fine — we collect `limit` blocks regardless of order, then sort by height before returning. The syncing node receives blocks in numeric order.

For the syncing node's cursor, use the HIGHEST height in each response:

```rust
// After receiving a forward-seek batch:
let max_height = blocks.iter().map(|b| b.header.height).max().unwrap();
next_request_start = max_height + 1;
```

This ensures we don't request blocks we already have, even though the iterator order doesn't match numeric order.

---

## 7. Questions for DeepSeek

### Q1: Should the forward-seek scan ALL key prefixes or just DAG?

Current block keys have three formats:
- `qblock:height:{N}` — continuous blocks from turbo sync
- `qblock:dag:{N}:{proposer}` — sparse blocks from gossipsub
- Binary `height.to_be_bytes() + hash` — old finalize_block format

The forward-seek could scan all three formats in a single pass using `iterator_cf(IteratorMode::Start)` and filtering keys. But this would also iterate over `qblock:hash:` reverse-index keys, `qblock:latest` pointer, etc.

Better approach: scan `qblock:dag:` prefix first (sparse range), then `qblock:height:` prefix (dense range), merge results. Two iterators but cleaner separation.

### Q2: Should we change the P2P protocol or just the HTTP API?

Option A: Add `mode=forward` to HTTP only. The checkpoint probe and HTTP sync benefit immediately. P2P turbo sync continues using range mode (unchanged protocol).

Option B: Add forward mode to P2P block-pack too. More complex but benefits all sync paths.

For a $1.1B mainnet, changing the P2P protocol requires all nodes to upgrade. HTTP-only is safer and immediately useful for the checkpoint sync probe.

### Q3: How to handle the string-sort ordering in the syncing node?

If the serving node returns blocks in string-sort order (100441, 100442, 1015441, 2000941...), the syncing node needs to understand that these heights aren't sequential. It should:
a) Sort by height before processing
b) Set cursor to max(received_heights) + 1
c) Request again until no more blocks are returned

Is there a risk of infinite loops if the cursor keeps jumping past blocks that sort earlier in string order?

### Q4: Memory bounds for the forward-seek iterator?

The iterator walks RocksDB's LSM tree lazily — it doesn't load all keys into memory. Each `next()` call reads one key-value pair. With `limit=200`, we read at most 200 blocks (~500KB) before stopping.

Is there a case where the iterator could use excessive memory? (e.g., large compaction happening during iteration, or bloom filter memory allocation for the seek)

---

## 8. Safety Statement

This optimization is **read-only and backward compatible**:

- **ZERO database writes** — forward-seek is a read operation
- **ZERO P2P protocol changes** (if HTTP-only approach)
- **Backward compatible** — `mode=forward` is a new optional parameter; existing requests work unchanged
- **No consensus impact** — block serving order doesn't affect validation
- **Bounded memory** — `limit` parameter caps blocks per response
- **Existing behavior preserved** — `get_qblocks_range()` unchanged; forward-seek is a NEW function

---

## 9. Summary

| Aspect | Current | After Optimization |
|--------|---------|-------------------|
| Sync speed (sparse range) | 2-124 b/s | 500-1000+ b/s |
| Seeks per 200 blocks | ~1000 | 1 |
| Empty network round-trips | ~99% | ~0% |
| Full sync ETA | 34+ hours | ~6-8 hours |
| Code change risk | N/A | Zero (read-only, additive) |

The blocks are accessible. The checkpoint probe works. The last piece is making sync FAST through sparse ranges. One seek instead of a thousand.

---

*Generated 2026-04-19 — Quillon Foundation*
*Based on live sync test: Epsilon Docker container syncing from height 75K*
*Current rate: 124 b/s. Target: 1000+ b/s.*
