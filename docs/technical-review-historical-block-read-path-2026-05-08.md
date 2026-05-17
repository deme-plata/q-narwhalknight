# Technical Review: Historical Block Read Path — Making the Code Understand Old Block Formats
**Date:** 2026-05-08  
**Version:** v10.7.1  
**Author:** Server Beta  
**Reviewer:** DeepSeek R1  
**Status:** Revised v2 — implementation-ready after Era 1 key audit  

---

## 1. Executive Summary

Epsilon holds a 219 GB RocksDB database containing every block from genesis. Despite this,
`/api/v1/sync/blocks?from_height=1` returns zero blocks. The P2P block-pack serving has
the same limitation. Fresh nodes attempting a genesis sync receive no blocks below
approximately height 13,478,000 and must rely entirely on the balance checkpoint instead.

The root cause is not missing data — it is **historical format blindness**. The current
read path only understands modern block indexes and one transitional DAG key format. Older
blocks were persisted under earlier storage layouts that are still present on disk but no
longer queried or decoded by the current code. The data is present; the code does not know
how to ask for it.

The fix is to extend the read path through a sequence of legacy fallbacks, each read-only
and additive. This is far safer than a bulk re-index of 13.5 million historical records on
a live production node.

---

## 2. Block Key Architecture — Three Eras

Epsilon's chain spans three distinct storage eras, each producing a different key scheme in
RocksDB:

| Height range | Key format | Column family | Count |
|---|---|---|---|
| 0 — 13,477,999 | Unknown — **not in CF_BLOCKS** | Unknown CF (hypothesis: CF_DAG_VERTICES) | ~13.5M blocks |
| 13,478,000 — ~14,063,000 | `qblock:dag:{height}:{proposer_hex}` | CF_BLOCKS | ~545,710 blocks |
| ~14,063,000 — present | `qblock:height:{height}` | CF_BLOCKS | All recent blocks |

### Era 1 — DAG gossip era (0 – 13,477,999)

Blocks in this range were received via gossipsub during the early network before turbo sync
existed. There are no entries under `qblock:height:{h}` or `qblock:dag:{h}:{proposer}` in
`CF_BLOCKS` for these heights. The leading hypothesis is that they were written to
`CF_DAG_VERTICES` or a compact archive structure. **This hypothesis has not yet been
confirmed by key-space audit.** The Era 1 read-path implementation must not begin until the
exact write path and key scheme are confirmed (see Section 5).

### Era 2 — Transitional DAG key era (13,478,000 – ~14,063,000)

As `save_dag_layer_block()` was introduced, incoming gossipsub blocks began landing under
`qblock:dag:{height}:{proposer_hex}` in `CF_BLOCKS` (`lib.rs:1116`). These ~545,710 blocks
are present on disk and the DAG fallback added in v10.3.6 already queries this key format.
Some entries in this range fail deserialization due to intermediate binary layout drift.

### Era 3 — Modern canonical key era (~14,063,000 – present)

`save_qblock_batch()` writes the canonical pointer `qblock:height:{height}` to `CF_BLOCKS`.
This is the primary format. All blocks produced by the current node are here.

---

## 3. The Current Read Path and Its Gaps

### 3.1 Primary path: `qblock:height:{h}` multi_get

**File**: `crates/q-storage/src/lib.rs`, `get_qblocks_range()` — lines 2075–2180

```rust
let keys: Vec<Vec<u8>> = (start_height..=end_height)
    .map(|h| format!("qblock:height:{}", h).into_bytes())
    .collect();
let results = self.hot_db.multi_get(CF_BLOCKS, &keys).await?;
```

Covers Era 3 only. For heights below ~14,063,000 every key returns `None`. The function
does not abort on `None` — `consecutive_failures` resets to 0 on each miss (line 2178),
not on deserialization failure. It falls through to the DAG fallback.

### 3.2 DAG fallback: `qblock:dag:{h}:{proposer}` prefix scan

**File**: `lib.rs` — lines 2183–2278 (added v10.3.6)

```rust
let dag_prefix = format!("qblock:dag:{}:", height);
let dag_entries = self.hot_db.scan_prefix_seek(CF_BLOCKS, dag_prefix.as_bytes(), 1).await?;
```

Covers Era 2 heights where the key exists. Read-only, no data changes. The existing
implementation requests only the **first** matching entry. If multiple proposers exist for a
height, only one is tried. The first match is taken as canonical.

**Why some Era 2 heights still return empty:**  
Deserialization of stored bytes fails for some entries — the `QBlock` struct changed layout
during the Era 2 period. `deserialize_qblock_with_fallback` does not cover all intermediate
layouts. Failures are silently skipped at line 2253–2258.

**Why Era 1 heights return empty:**  
No `qblock:dag:{h}:{proposer}` keys exist in `CF_BLOCKS` for heights 0–13,477,999. The
prefix scan finds nothing. There is no further fallback.

### 3.3 `consecutive_failures` — important boundary behaviour

`consecutive_failures` increments only when a key exists (`Some(value)`) but deserialization
fails. It resets to 0 on success AND on `None` (missing). This means:

- **Era 1 heights (all None):** No early-abort fires — but with 13.5M missing heights
  requested, the primary loop runs to completion without short-circuiting, then all 13.5M
  heights flow into the DAG fallback's per-height prefix scan. At one seek per height this
  is catastrophically expensive (see Section 4.4).
- **Corrupt Era 2 entries (Some, deser fails):** Early-abort fires after 10 consecutive
  failures if no blocks have been collected yet.
- **Valid Era 2 entries:** DAG fallback recovers them; no abort.

---

## 4. The Fix: Extend the Read Path Without Touching Stored Data

The correct approach is to teach the code to read existing formats rather than rewrite
what is stored. Three sequential resolvers replace the current mixed primary+fallback code.

### 4.1 Restructured resolver shape

```rust
async fn get_qblocks_range(&self, start_height: u64, limit: usize) -> Result<Vec<QBlock>> {
    let mut blocks = Vec::new();
    let mut missing: Vec<u64> = (start_height..=end_height).collect();

    // Resolver 1: modern canonical keys (Era 3)
    let primary = self.read_canonical_height_keys(&missing).await?;
    merge_hits(&mut blocks, &mut missing, primary, &mut metrics.primary);

    // Resolver 2: transitional DAG keys (Era 2)
    let dag = self.read_dag_block_keys(&missing).await?;
    merge_hits(&mut blocks, &mut missing, dag, &mut metrics.dag);

    // Resolver 3: early vertex storage (Era 1) — implement only after key audit
    let vertices = self.read_legacy_dag_vertices(&missing).await?;
    merge_hits(&mut blocks, &mut missing, vertices, &mut metrics.vertex);

    blocks.sort_by_key(|b| b.header.height);
    metrics.total_missing = missing.len();
    Ok(blocks)
}
```

Each resolver is independently testable. Each can be enabled, disabled, or profiled
separately without touching the others.

### 4.2 Fix Era 2: deserialization layout variants

Some Era 2 blocks were written when `QBlock` had different field ordering, missing optional
fields, or a pre-`QuantumMetadata` layout. Before adding new variants to
`q_types/src/legacy.rs`, inspect the actual raw bytes from a known failing height:

```bash
# Dump raw value bytes at a failing Era 2 height
ssh root@89.149.241.126 "ldb --db=/home/orobit/data-mainnet-genesis \
    --column_family=blocks get 'qblock:dag:13500000:' 2>/dev/null | head -1"
```

The byte inspection must confirm:
- Is it a raw bincode `QBlock`? (check magic bytes / length prefix)
- Is it wrapped in LZ4 compression? (check the `is_precompressed()` header)
- What bincode configuration was in use? (little-endian vs big-endian, fixed vs varint)

Bincode configuration mismatches across versions are a common silent failure: matching the
exact settings from the writing era is as important as matching the struct layout. Only after
confirming the exact format should a new variant be added:

```rust
// In q_types/src/legacy.rs — deserialize_qblock_with_fallback():
// Add missing era variant after existing attempts:
if let Ok(block) = bincode::deserialize::<QBlockV8>(&data) {  // exact version TBD
    return Ok(block.into());
}
```

Add a regression test using **real serialized bytes** copied from a failing Era 2 block.
Synthetic struct tests will not catch layout drift; the actual bytes will.

### 4.3 Fix Era 2: try multiple DAG candidates per height

The current implementation requests `scan_prefix_seek(..., 1)` — only the first proposer
entry. If the first entry is corrupt but a second is valid, the height is permanently missed.
Safer pattern:

```rust
const MAX_DAG_CANDIDATES: usize = 16;
let entries = self.hot_db.scan_prefix_seek(CF_BLOCKS, dag_prefix.as_bytes(),
                                            MAX_DAG_CANDIDATES).await?;

for (_key, value) in entries {
    let deser = try_deserialize_block(&value);
    match deser {
        Ok(block) if block.header.height == height => {
            blocks.push(block);
            dag_hits += 1;
            break;  // first valid entry for this height wins
        }
        Ok(block) => {
            warn!(requested = height, decoded = block.header.height,
                  "DAG fallback decoded wrong height — skipping");
        }
        Err(e) => {
            dag_deser_errors += 1;
        }
    }
}
```

**Always validate height after deserialization.** A decoded block whose `header.height`
does not match the requested height is silently discarded, never served to peers.

### 4.4 Fix Era 1: bounded batch iterator, not per-height seek

> ⚠️ Do not implement this until the Era 1 key scheme is confirmed (Section 5).

For a request like `from_height=1&limit=1000`, iterating 1,000 individual prefix seeks over
`CF_DAG_VERTICES` is acceptable. But `from_height=1&limit=13000000` would trigger 13 million
seeks — catastrophically expensive. The Era 1 resolver must use a **batch iterator strategy**:

```rust
async fn read_legacy_dag_vertices(&self, missing: &[u64]) -> Result<Vec<QBlock>> {
    if missing.is_empty() { return Ok(vec![]); }

    let min_h = *missing.iter().min().unwrap();
    let max_h = *missing.iter().max().unwrap();

    // Single forward iterator scan over the full missing range
    let prefix_from = format!("vertex:{}:", min_h);  // key scheme TBD
    let prefix_to   = format!("vertex:{}:", max_h + 1);

    let missing_set: HashSet<u64> = missing.iter().copied().collect();
    let mut found = Vec::new();

    let iter = self.hot_db.range_iter(CF_DAG_VERTICES, &prefix_from, &prefix_to).await?;
    for (key, value) in iter {
        if let Some(height) = parse_height_from_vertex_key(&key) {
            if missing_set.contains(&height) {
                if let Ok(block) = reconstruct_qblock_from_vertex(&value, height) {
                    found.push(block);
                }
            }
        }
    }
    Ok(found)
}
```

One iterator scan over a contiguous key range replaces N individual seeks. This is O(range)
regardless of how many heights are missing within that range.

### 4.5 Define what makes a recovered block acceptable

Historical DAG vertices were accepted by the network at the time of production. They should
be valid canonical blocks. However, the reconstructed `QBlock` must satisfy all of these
before being served to syncing peers:

| Property | Check |
|---|---|
| Height matches requested height | `block.header.height == requested_height` |
| Previous hash present and non-zero | `block.header.prev_block_hash != [0u8; 32]` |
| Transactions parseable | `block.transactions` non-empty or `block.header.tx_root == EMPTY_HASH` |
| Proposer / signature fields present | `block.header.producer_public_key.len() > 0` |
| Block hash recomputes correctly | Optional for serving; required before declaring archive |

Blocks that fail these checks should be logged and skipped, not served.

---

## 5. Era 1 Key Scheme Audit — Three Options

Before writing any Era 1 fallback code, confirm the actual key format using one of:

**Option A — `ldb` key range scan**
```bash
ssh root@89.149.241.126 "ldb --db=/home/orobit/data-mainnet-genesis \
    --column_family=dag_vertices scan --from='v' --to='w' --max_keys=20 2>/dev/null"
```

**Option B — Write call audit**
```bash
grep -n "CF_DAG_VERTICES\|dag_vertices" \
    crates/q-storage/src/*.rs | grep -E '"put|\.write|save"'
```
The write-side code shows the exact key format used at write time.

**Option C — SST file dump (most reliable)**

Using RocksDB's `SstFileReader` or `ldb dump` on an early SST file bypasses any column
family configuration uncertainty and gives raw key strings from the actual 219 GB database:
```bash
ssh root@89.149.241.126 "ldb --db=/home/orobit/data-mainnet-genesis dump \
    --column_family=dag_vertices --count_only 2>/dev/null"
# Then: list early SST files and dump a sample
```

**Do not proceed to Era 1 implementation until the output of one of these options confirms
the key prefix format.**

---

## 6. Observability — Add Metrics Before Adding Fallbacks

Silent skipping of historical block misses is the original bug. The fix must not introduce
new silent skipping. Add structured counters to all three resolvers:

```rust
struct BlockReadMetrics {
    primary_hits:         u64,
    primary_misses:       u64,
    dag_hits:             u64,
    dag_deser_errors:     u64,
    dag_wrong_height:     u64,
    vertex_hits:          u64,   // Era 1 — add when resolver is implemented
    vertex_deser_errors:  u64,
    total_missing_after_all_fallbacks: u64,
}
```

Log a summary at INFO level for every range request that yields any misses after all
resolvers are exhausted. Log the first successfully recovered Era 1 block at INFO level
to confirm the fix is live without requiring log archaeology.

---

## 7. Why Not Re-index

"Re-indexing" would mean iterating old formats and writing new `qblock:height:{h}` keys
into `CF_BLOCKS`. This is rejected for three reasons:

1. **Live-node risk**: Writing millions of new keys to Epsilon's production RocksDB while
   it serves the live network risks I/O interference, compaction pressure, and column family
   lock contention. A bug in a 13.5M-row write migration can corrupt valid data
   irreversibly.

2. **Not necessary**: `scan_prefix_seek` already proves the read-time fallback pattern
   works. The v10.3.6 DAG fallback recovered 545,710 Era 2 blocks with zero writes.

3. **Reversible vs. irreversible**: A read-path code change can be reverted in one commit.
   A 13.5M-row write migration cannot be easily undone if it produces incorrect results.

Re-indexing remains available as an optional **offline** optimization after the read-path
fix is validated and Epsilon is no longer serving live traffic on that DB.

---

## 8. Test Requirements

### Minimum functional test matrix

| Test | Input | Expected output |
|---|---|---|
| Genesis blocks | `from_height=1&limit=10` | Returns genesis-era blocks |
| Era 1 → Era 2 boundary | `from_height=13477990&limit=20` | Returns blocks across boundary |
| Era 2 DAG-key blocks | `from_height=13500000&limit=10` | Returns valid blocks |
| Era 2 → Era 3 boundary | `from_height=14063000&limit=10` | Returns blocks across boundary |
| Known failing Era 2 height | Single height known to fail deser | Now deserializes |
| Corrupt / missing height | Single corrupt entry in range | Skipped, rest of range still returned |
| Large request | `from_height=1&limit=1000` | Bounded latency, no runaway scans |

### Regression test — real bytes required

Add one test in `crates/q-storage/tests/` that deserializes **actual bytes copied from a
failing Era 2 height** in Epsilon's DB. Synthetic struct tests will not catch layout drift.
The real bytes will. This test should pass before the deserialization fix is merged.

### Performance test

Profile `get_qblocks_range(1, 1000)` with the Era 1 fallback enabled. Confirm it completes
in under 500ms on NVMe. Per-height prefix seeks must not be used for large ranges.

---

## 9. Impact Table — After Fix

| Sync scenario | Before fix | After fix |
|---|---|---|
| Heights 0–13.5M via P2P | 0 blocks returned | Blocks served from Era 1 resolver |
| Heights 13.5M–15.5M via P2P | Partial (deser failures) | All 545K DAG-era blocks served |
| Heights 15.5M+ via P2P | Already works | Unchanged |
| `/api/v1/sync/blocks?from_height=1` | Empty response | Full block stream from genesis |
| Fresh node archive declaration | Cannot — incomplete history | Can once all eras readable |
| `Q_ARCHIVE_NODE_URL` proxy (SYNC-001) | Proxies from ~15.5M only | Full genesis history via proxy |

---

## 10. Relationship to Other Open Issues

| Issue | Relationship |
|---|---|
| SYNC-001 (block history loss on fresh nodes) | This fix enables Epsilon to serve historical blocks; SYNC-001's archive proxy then makes those blocks reachable by any node pointing to Epsilon |
| SYNC-002 (transfer-skip balance bug, fixed v10.7.1) | Orthogonal — SYNC-002 was a balance-processing bug; this is a block-retrieval bug |
| BAL-001 (block 18,600,000 enforcement) | Indirectly related — nodes auditing historical balance roots need historical block access |
| Genesis sync double-counting (observed 2026-05-08) | Separate: P2P snapshot + block replay double-applies transactions; fixed by checkpoint |

---

## 11. Recommended Implementation Order

| Priority | Task | Effort | Gate |
|---|---|---|---|
| P0 | Audit Era 1 column family and key layout (Section 5, Option C) | 1 hour | Blocks Era 1 implementation |
| P0 | Add `BlockReadMetrics` counters to all resolvers | 1 hour | No gate |
| P1 | Inspect raw bytes from failing Era 2 height, confirm bincode config | 1 hour | No gate |
| P1 | Add missing `QBlock` layout variant to `deserialize_qblock_with_fallback` | 2–4 hours | Byte inspection done |
| P1 | Add real-bytes regression test for failing Era 2 height | 1 hour | Same |
| P1 | Fix `scan_prefix_seek(..., 1)` → multi-candidate loop with height validation | 2 hours | No gate |
| P2 | Add Era 1 batch iterator resolver | 4–8 hours | Era 1 audit done |
| P2 | Performance test: `from_height=1&limit=1000` under 500ms | 30 min | Era 1 resolver done |
| P3 | Declare Epsilon `node_type=archive` in P2P announcements | 1 hour | All eras readable |
| P4 | Optional offline re-index on a cold DB copy | 1–2 days | Not blocking |
