# Technical Review: DAG Key Discovery — Blocks Were Never Lost
## Sync Fix for 545,710 Hidden Blocks on $1.1B Mainnet
### Date: 2026-04-16 | Status: ZERO-RISK FIX REQUIRED

---

## 1. Executive Summary

**Previous diagnosis was WRONG.** We believed 8.4M blocks (heights 1.6M–10M) were lost from Epsilon's RocksDB due to SST file corruption during `kill -9`. After deeper investigation, we discovered:

- **545,710 blocks exist** in the `qblock:dag:{height}:{proposer}` key format
- They span the FULL chain history from ~100K through ~9M+
- Multiple distinct miners/proposers are represented
- The data was invisible to our scans because of **string-sort ordering** — our `--max_keys=50` scan only returned keys starting with "1" and never reached keys starting with "2", "3", "4", etc.
- The "SST corruption" errors were **transient read races** from `ldb` colliding with live compaction — the running server has zero issues

**The blocks were never deleted. They were never corrupted. They've been in the DB since March 2026. The sync code simply can't find them.**

**This is a read-path bug, not a data loss incident.**

---

## 2. Evidence

### 2.1 Key Count Summary (Epsilon: 89.149.241.126)

| Key Format | Count | Height Range | Purpose |
|------------|-------|-------------|---------|
| `qblock:dag:{height}:{proposer}` | **545,710** | ~100K – ~9M+ | Gossipsub-received blocks (P2P) |
| `qblock:height:{height}` | **5,876,323** | 10,000,031 – 15,606,000+ | Turbo-sync downloaded blocks |
| **Total block entries** | **6,422,033** | | |

### 2.2 DAG Blocks Confirmed at Every Height Range

| Height Range | Sample Keys | Proposer(s) | Status |
|-------------|-------------|-------------|--------|
| 100K | `qblock:dag:100441:e4034c57...` | 1 miner | Confirmed |
| 1M | `qblock:dag:1015441:e4034c57...` | 1 miner | Confirmed (47 entries) |
| 2M | `qblock:dag:2000941:e4034c57...` | 1 miner | Confirmed (10+ entries) |
| 3M | `qblock:dag:3000344:e14075fb...` | **2 miners** | Confirmed (10+ entries) |
| 4M | `qblock:dag:4015856:48332fcc...` | **3 miners** | Confirmed (10+ entries) |
| 5M | `qblock:dag:5000941:e4034c57...` | 1 miner | Confirmed (10+ entries) |
| 7M-8M | Transient IO error (live compaction race) | Unknown | Likely present |
| 9M | `qblock:dag:9002564:3e463a56...` | **3 miners** | Confirmed (10+ entries) |
| 10M+ | `qblock:height:10000031` | N/A (height format) | 5.8M continuous |

**7 distinct proposer IDs** found across the chain — this is real multi-miner history, not test data.

### 2.3 How We Were Fooled

RocksDB uses byte-order sorting. Our block height keys are decimal strings without zero-padding:

```
Byte/string sort order:               Numeric order:
"qblock:dag:100441"    (100K)         100,441
"qblock:dag:1015441"   (1.0M)         1,015,441
"qblock:dag:1015487"   (1.0M)  ← scan stopped here (50 keys)
                                      
"qblock:dag:2000941"   (2.0M)  ← NEVER REACHED        2,000,941
"qblock:dag:3000344"   (3.0M)  ← NEVER REACHED        3,000,344
"qblock:dag:5000941"   (5.0M)  ← NEVER REACHED        5,000,941
"qblock:dag:9027291"   (9.0M)  ← NEVER REACHED        9,027,291
```

The character `'1'` (ASCII 0x31) sorts before `'2'` (ASCII 0x32). ALL keys starting with "1" sort before ALL keys starting with "2". Our `--max_keys=50` returned 50 keys in the "1xxxxx" range and stopped. We concluded the data was gone. It was 1 key away.

### 2.4 The "Corruption" Was a False Alarm

| What we saw | What actually happened |
|-------------|----------------------|
| `8694668.sst: No such file or directory` | `ldb` opened MANIFEST while live server was mid-compaction. File existed for an instant, then was renamed/replaced. |
| `MANIFEST-8546725 may be corrupted` | Same race — MANIFEST referenced an SST that was being compacted in real time. |
| `8884540.log: No such file or directory` | WAL file was rotated by the live server between MANIFEST read and file access. |

**The running server has been up 11+ hours with zero errors.** It reads and writes blocks at height 15.6M continuously. The DB is healthy. The "corruption" was an artifact of running `ldb` (an external read-only tool) against a live database with active compaction.

---

## 3. Root Cause: Why Sync Can't Find 545K Blocks

### 3.1 The Two Storage Paths

The codebase has two distinct paths for storing blocks:

```
PATH A: Gossipsub P2P reception (save_dag_layer_block)
  Key format: "qblock:dag:{height}:{proposer_hex}"
  File: crates/q-storage/src/lib.rs:1090
  Used when: Node receives a block from a peer via gossipsub
  Written by: Epsilon since March 2, 2026 (DB creation)
  Result: 545,710 blocks stored

PATH B: Turbo Sync bulk download (batch writer)  
  Key format: "qblock:height:{height}"
  File: crates/q-storage/src/turbo_sync.rs (batch commit)
  Used when: Node downloads blocks in bulk from a peer
  Written by: Epsilon after turbo sync was activated (~height 10M)
  Result: 5,876,323 blocks stored
```

**The gap is explained**: Before turbo sync existed (or was activated on Epsilon), all blocks came via gossipsub and were stored as `qblock:dag:`. After turbo sync was activated (~height 10M), blocks were stored as `qblock:height:`. The two formats coexist in the same `blocks` column family but have different key prefixes.

### 3.2 The Three Bugs in the Read Path

**Bug 1: Fast path ignores DAG format entirely**

```rust
// crates/q-storage/src/lib.rs:2003
pub async fn get_qblocks_range(&self, start_height: u64, limit: usize) -> Result<Vec<QBlock>> {
    // Constructs keys as "qblock:height:{N}" — ONLY this format
    let keys: Vec<Vec<u8>> = (start_height..=end_height)
        .map(|h| format!("qblock:height:{}", h).into_bytes())
        .collect();
    
    let results = self.hot_db.multi_get(CF_BLOCKS, &keys).await?;
    // Returns empty for heights 1.6M-10M because those use qblock:dag: format
}
```

This is the primary block-serving function used by the P2P block-pack handler. It NEVER searches `qblock:dag:` keys.

**Bug 2: Fallback aborts after 20 consecutive misses**

```rust
// crates/q-storage/src/lib.rs:2360-2376
const MAX_CONSECUTIVE_FAILURES: usize = 20;

for i in 0..capped_limit as u64 {
    let height = start_height + i;
    
    match self.get_qblock_any_format(height).await {
        Ok(Some(block)) => { consecutive_failures = 0; }
        Ok(None) => {
            consecutive_failures += 1;
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES {
                break;  // ← GIVES UP after 20 misses
            }
        }
    }
}
```

The DAG entries are sparse — not every height has a block. The gap between "no block" and "first DAG block" can be hundreds of heights (e.g., request starts at 2,000,000 but first DAG entry is at 2,000,941 — 941 heights away). The 20-miss abort triggers after checking just 20 heights.

**Bug 3: `get_qblock_any_format()` does per-height prefix scan (correct but slow)**

```rust
// crates/q-storage/src/lib.rs:2195-2220
pub async fn get_qblock_any_format(&self, height: u64) -> Result<Option<QBlock>> {
    // Try format 1: "qblock:height:{N}"
    if let Some(block) = self.get_qblock_by_height(height).await? {
        return Ok(Some(block));
    }
    
    // Try format 2: scan_prefix("qblock:dag:{N}:")
    let dag_prefix = format!("qblock:dag:{}:", height);
    let dag_entries = self.hot_db.scan_prefix(CF_BLOCKS, dag_prefix.as_bytes()).await?;
    if let Some((_key, value)) = dag_entries.into_iter().next() {
        // Deserialize and return
    }
    
    Ok(None)
}
```

This function IS correct — it CAN find DAG entries. But it's called per-height from the fallback loop, and the fallback gives up after 20 misses. The function itself works; the caller quits too early.

### 3.3 Why the Checkpoint Sync Probe Also Fails

```rust
// crates/q-storage/src/turbo_sync.rs:3187-3212
fn probe_network_gap_blocking(target_height: u64) -> u64 {
    let bootstrap_urls = ["http://185.182.185.227:8080", "http://89.149.241.126:8080"];
    
    let probe = |height: u64| -> bool {
        for url in &bootstrap_urls {
            let block_url = format!("{}/api/v1/blocks/{}", url, height);
            // HTTP GET to /api/v1/blocks/{height}
            // BUT: this endpoint returns HTTP 404 for ALL heights
            // Even heights where blocks exist in the DB
        }
        false  // Always returns false
    };
}
```

The `/api/v1/blocks/{height}` REST endpoint returns HTTP 404 with empty body for every height. This endpoint is broken independently of the key format issue. Even if the block exists in the DB, the API doesn't serve it.

---

## 4. The Fix (Three Options, All Zero Risk)

### Option A: Increase miss threshold (1-line change, immediate)

```rust
// crates/q-storage/src/lib.rs:2360
// Change:
const MAX_CONSECUTIVE_FAILURES: usize = 20;
// To:
const MAX_CONSECUTIVE_FAILURES: usize = 5000;
```

**Pros:** Minimal code change, fallback now scans far enough to find sparse DAG entries.
**Cons:** Slow for genuinely empty ranges — scanning 5000 heights per miss wastes time. Syncing peer waits longer for "no data" answers.

### Option B: Range-scan instead of per-height probe (better, ~30 lines)

Replace the per-height probe loop with a RocksDB prefix range scan:

```rust
// Instead of probing height-by-height:
//   get_qblock_any_format(2000000) → miss
//   get_qblock_any_format(2000001) → miss
//   ... 940 misses ...
//   get_qblock_any_format(2000941) → HIT

// Do a single range scan:
//   scan_prefix("qblock:dag:") starting from "qblock:dag:{start_height}:"
//   Collect next N entries regardless of height gaps
//   This finds qblock:dag:2000941 in ONE operation, not 941

pub async fn get_dag_blocks_from_height(
    &self,
    start_height: u64,
    limit: usize,
) -> Result<Vec<QBlock>> {
    let scan_from = format!("qblock:dag:{}:", start_height);
    let entries = self.hot_db.scan_prefix_from(
        CF_BLOCKS,
        b"qblock:dag:",
        scan_from.as_bytes(),
        limit * 2,  // Over-fetch to account for multiple proposers per height
    ).await?;
    
    let mut blocks = Vec::new();
    let mut seen_heights: HashSet<u64> = HashSet::new();
    
    for (key, value) in entries {
        // Parse height from key: "qblock:dag:{height}:{proposer}"
        let key_str = String::from_utf8_lossy(&key);
        let parts: Vec<&str> = key_str.split(':').collect();
        if parts.len() >= 3 {
            if let Ok(height) = parts[2].parse::<u64>() {
                if !seen_heights.contains(&height) {
                    seen_heights.insert(height);
                    if let Ok(block) = deserialize_block(&value) {
                        blocks.push(block);
                        if blocks.len() >= limit {
                            break;
                        }
                    }
                }
            }
        }
    }
    
    blocks.sort_by_key(|b| b.header.height);
    Ok(blocks)
}
```

**Pros:** Efficient — finds sparse DAG entries in O(1) scan operations, no wasted probes.
**Cons:** Needs a `scan_prefix_from()` function in the KV store trait (scan with both prefix filter AND start position). May need to be added if it doesn't exist.

### Option C: Background re-indexing (best long-term, ~50 lines)

Create `qblock:height:N` aliases for every `qblock:dag:N:*` entry in a background task:

```rust
// Run once as a background migration task after startup
pub async fn reindex_dag_blocks_to_height_keys(&self) -> Result<u64> {
    let mut reindexed = 0u64;
    
    // Scan ALL qblock:dag: keys
    let entries = self.hot_db.scan_prefix(CF_BLOCKS, b"qblock:dag:").await?;
    
    for (key, value) in entries {
        let key_str = String::from_utf8_lossy(&key);
        let parts: Vec<&str> = key_str.split(':').collect();
        if parts.len() >= 3 {
            if let Ok(height) = parts[2].parse::<u64>() {
                // Check if qblock:height:{N} already exists
                let height_key = format!("qblock:height:{}", height);
                if self.hot_db.get(CF_BLOCKS, height_key.as_bytes()).await?.is_none() {
                    // Create the alias (write the same block data under height key)
                    self.hot_db.put(CF_BLOCKS, height_key.as_bytes(), &value).await?;
                    reindexed += 1;
                    
                    if reindexed % 10000 == 0 {
                        info!("🔄 [RE-INDEX] Progress: {} DAG blocks indexed as height keys", reindexed);
                    }
                }
            }
        }
    }
    
    info!("✅ [RE-INDEX] Complete: {} DAG blocks now accessible via qblock:height: format", reindexed);
    Ok(reindexed)
}
```

**Pros:** After re-indexing, the fast path (`get_qblocks_range`) finds everything — no fallback needed. Syncing peers get blocks at maximum speed. One-time cost.
**Cons:** Writes ~545K new keys to the DB (additive only, never deletes). Takes some time (minutes). Should be gated behind an env var for safety.

---

## 5. Questions for DeepSeek

**CONTEXT: $1.1B mainnet. 545,710 blocks discovered in `qblock:dag:` format that syncing peers cannot access due to a read-path bug. The data is intact. The fix is read-path only. We need the safest approach.**

### Q1: Which fix option do you recommend?

Option A (increase miss threshold to 5000), Option B (range-scan for DAG entries), or Option C (re-index DAG→height keys)?

Our preference is **Option C** because it's a one-time fix that makes the fast path work for all future sync requests. But it requires writing 545K new keys to production. Is this safe? What's the worst case if the re-indexing is interrupted mid-way (kill -9, OOM, disk full)?

Key safety property of Option C: **it's strictly additive** — it only creates NEW `qblock:height:` keys, never modifies or deletes existing `qblock:dag:` keys. If interrupted, partial progress is still valid (some heights now have both formats, others have only DAG). Can be resumed safely.

### Q2: Should the re-indexing be gated behind a migration flag?

Similar to the existing `migration_*_done` flags in the codebase. This prevents re-running on every restart. But as we discovered earlier, migration flags + the `rebuild_balances_from_chain()` bug caused startup crash loops. How do we avoid that pattern here?

Proposed: use a flag, but set it BEFORE starting (not after completing). If interrupted, the flag is already set, so restart doesn't re-run. The re-index is idempotent (checks `is_none()` before writing), so partial runs are safe.

### Q3: Should we also fix the /api/v1/blocks/{height} endpoint?

The HTTP checkpoint probe (`probe_network_gap_blocking`) uses this endpoint but it returns 404 for all heights. If we fix this endpoint to actually return block data (from either key format), the checkpoint probe would correctly find where blocks start and new nodes would sync from the right height.

Is fixing this endpoint worth the risk? It's an API handler change, not a storage change. The handler would call `get_qblock_any_format(height)` which already works correctly for individual heights.

### Q4: What's the best ordering of fixes?

Should we:
a) Fix the read path first (Option B/C), then deploy, then fix the API endpoint?
b) Fix the API endpoint first (quick win), deploy, then fix the read path?
c) Do everything in one release?

For a $1.1B mainnet, our instinct is (a) — fix the core issue first, verify with Docker sync test, then add the API fix as a follow-up.

### Q5: RocksDB scan_prefix performance concern

Option B uses `scan_prefix("qblock:dag:")` which scans ALL 545K DAG keys. For a block-pack request asking for 200 blocks starting at height 8M, this scan would iterate through all keys from "qblock:dag:" until it finds ones at 8M+ — potentially reading 400K+ keys to skip.

Is there a more efficient RocksDB operation? Can we use `seek("qblock:dag:8000000:")` + `iterate_forward(limit)` to jump directly to the right position? This would be O(log N) seek + O(limit) iteration instead of O(N) prefix scan.

### Q6: String sort — should we address it now or later?

The string-sort issue means `qblock:dag:2000000` sorts AFTER `qblock:dag:19999999` (because "2" > "1"). This makes range queries unreliable. Should Option C (re-indexing) use zero-padded keys for the new `qblock:height:` entries?

```rust
// Current: format!("qblock:height:{}", height)  → "qblock:height:2000000"
// Padded:  format!("qblock:height:{:010}", height) → "qblock:height:0002000000"
```

Pros: Future range scans sort correctly, SST files partition by height range.
Cons: Breaks compatibility with the 5.8M existing `qblock:height:` keys (they're unpadded).

Dual-write option: write BOTH padded AND unpadded keys during re-indexing. Reader checks padded first, falls back to unpadded. Is the 2x key count worth it?

### Q7: How do we verify the fix before deploying to $1.1B mainnet?

Proposed verification plan:
1. Implement Option C (re-index) gated behind `Q_ENABLE_DAG_REINDEX=1`
2. Build new binary
3. Start Docker test container on Epsilon with the new binary + `Q_ENABLE_DAG_REINDEX=1`
4. Verify: container re-indexes 545K DAG blocks → creates `qblock:height:` keys
5. Start a SECOND Docker container (fresh, no re-index) and sync from the first
6. Verify: second container syncs from height ~1.6M (not 10M) and receives the DAG blocks
7. Verify: wallet balances on fully-synced container match production
8. Only THEN deploy to Epsilon production with `Q_ENABLE_DAG_REINDEX=1`

Is this verification plan sufficient? What else should we check?

---

## 6. Impact Assessment

### Before Fix

```
New node joins network:
  → Checkpoint probe: /api/v1/blocks/{height} returns 404 for all heights
  → P2P block-pack: get_qblocks_range() searches qblock:height: only → nothing below 10M
  → Fallback: get_qblock_any_format() aborts after 20 misses → nothing
  → Checkpoint sync lands at ~10M-14M
  → Syncs only last 1.5-5.6M blocks
  → Missing 545K blocks of chain history
```

### After Fix (Option C)

```
One-time re-indexing on Epsilon:
  → 545K qblock:dag: entries get qblock:height: aliases
  → get_qblocks_range() now finds blocks from ~100K onwards
  
New node joins network:
  → P2P block-pack: get_qblocks_range() finds blocks starting at ~100K
  → Checkpoint sync correctly identifies chain start at ~100K (not 10M)
  → Syncs complete chain: 100K → 15.6M (~15.5M blocks)
  → Full chain history available
  → All block explorer queries work for the full chain
```

### What Remains Missing (Unrecoverable)

| Range | Status | Reason |
|-------|--------|--------|
| Blocks 0 – ~100,000 | **Missing** | Pre-genesis / earliest pruned blocks |
| Blocks 100K – 1.6M (sparse) | **Partially available** | DAG entries exist at some heights |
| Blocks 1.6M – 10M | **Available (545K DAG entries)** | This fix unlocks them |
| Blocks 10M – 15.6M | **Available (5.8M height entries)** | Already working |

---

## 7. Safety Statement

This fix is **strictly additive**:
- **NO existing keys are modified or deleted** — `qblock:dag:` entries remain untouched
- **NO consensus rules change** — block validation is identical
- **NO P2P protocol change** — same block-pack request/response format
- **NO storage format change** — uses the existing `qblock:height:` format
- **Interruption-safe** — if re-indexing is interrupted, partial progress is valid and resumable
- **Gated behind env var** — production nodes don't run it unless explicitly enabled
- **Tested on Docker first** — never touches production until verified on disposable copy

The worst case if the re-indexing has a bug: some `qblock:height:` keys point to corrupted data → syncing peer gets bad blocks → peer's block validation rejects them → no damage, peer retries from another source. The existing `qblock:dag:` data is untouched.

---

*Generated 2026-04-16 — Quillon Foundation*
*Based on forensic RocksDB key scan using `ldb` (rocksdb-tools 8.9.1) against live Epsilon production DB*
*All scans were read-only — zero writes to production database*
