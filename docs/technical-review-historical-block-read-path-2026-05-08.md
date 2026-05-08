# Technical Review: Historical Block Read Path — Making the Code Understand Old Block Formats
**Date:** 2026-05-08  
**Version:** v10.7.1  
**Author:** Server Beta  
**Status:** Open — fix not yet implemented  

---

## 1. Executive Summary

Epsilon holds a 219 GB RocksDB database containing every block from genesis. Despite this,
`/api/v1/sync/blocks?from_height=1` returns zero blocks. The P2P block-pack serving has
the same limitation. Fresh nodes attempting a genesis sync receive no blocks below
approximately height 13,478,000 and must rely entirely on the balance checkpoint instead.

The root cause is not missing data — it is a **narrow read path**. `get_qblocks_range()`
queries only two key formats in one column family (`CF_BLOCKS`). Blocks written during the
early chain existed under different key schemes or different column families that the current
read path does not attempt. The data is present in the database; the code does not know how
to ask for it.

This document describes the key architecture, what the read path currently tries, where it
falls short, and the fix: teach the code to read the additional formats rather than
re-writing stored data.

---

## 2. Block Key Architecture — Three Eras

Epsilon's chain spans three distinct storage eras, each producing a different key scheme in
RocksDB:

| Height range | Key format | Column family | Count |
|---|---|---|---|
| 0 — 13,477,999 | Unknown / not in CF_BLOCKS | Unknown CF | ~13.5M blocks |
| 13,478,000 — ~14,063,000 | `qblock:dag:{height}:{proposer_hex}` | CF_BLOCKS | ~545,710 blocks |
| ~14,063,000 — present | `qblock:height:{height}` | CF_BLOCKS | All recent blocks |

### Era 1 — DAG gossip era (0 – 13,477,999)

Blocks in this range were received via gossipsub during the early network before turbo sync
existed. They were written directly as raw DAG vertex payloads. There are no entries under
`qblock:height:{h}` or `qblock:dag:{h}:{proposer}` in `CF_BLOCKS` for these heights. The
data is believed to be in `CF_DAG_VERTICES` or a compact archive structure, but the exact
key scheme has not yet been audited (see Section 5).

### Era 2 — Transitional DAG key era (13,478,000 – ~14,063,000)

As the codebase matured and `save_dag_layer_block()` was introduced, incoming gossipsub
blocks began landing under `qblock:dag:{height}:{proposer_hex}` in `CF_BLOCKS`
(see `lib.rs:1116`). These ~545,710 blocks are fully present and retrievable; the DAG
fallback added in v10.3.6 already handles them.

### Era 3 — Modern canonical key era (~14,063,000 – present)

`save_qblock_batch()` writes the canonical pointer `qblock:height:{height}` to `CF_BLOCKS`.
This is the only format the primary read path queries. All blocks produced by the current
node are in this format.

---

## 3. The Current Read Path and Its Gaps

### 3.1 Primary path: `qblock:height:{h}` multi_get

**File**: `crates/q-storage/src/lib.rs`  
**Function**: `get_qblocks_range()` — lines 2075–2180

```rust
let keys: Vec<Vec<u8>> = (start_height..=end_height)
    .map(|h| format!("qblock:height:{}", h).into_bytes())
    .collect();
let results = self.hot_db.multi_get(CF_BLOCKS, &keys).await?;
```

This covers Era 3 only. For any height below ~14,063,000, every key in the batch returns
`None`. The function does not abort early on `None` (missing = not corrupt — `consecutive_failures`
is reset to 0 on each miss, line 2178). It falls through to the DAG fallback.

### 3.2 DAG fallback: `qblock:dag:{h}:{proposer}` prefix scan

**File**: `crates/q-storage/src/lib.rs` — lines 2183–2278 (added v10.3.6)

```rust
// For each height that multi_get returned None, check if a DAG entry exists.
let dag_prefix = format!("qblock:dag:{}:", height);
let dag_entries = self.hot_db.scan_prefix_seek(CF_BLOCKS, dag_prefix.as_bytes(), 1).await?;
```

This covers Era 2. When `qblock:height:{h}` misses, the code scans for
`qblock:dag:{h}:*`, deserializes the first result with `deserialize_qblock_with_fallback`,
and appends the block to the output. A re-sort is applied at the end.

This fallback is read-only — no writes, no key changes. It works as intended for Era 2
heights that still deserialize cleanly.

**Why it still returns empty for some Era 2 heights**: Some blocks stored in this era used
intermediate binary layouts that `deserialize_qblock_with_fallback` cannot parse. When
deserialization fails, the entry is silently skipped (`dag_deser_errors` counter at line
2254–2258). If every block in the requested range fails deserialization, the response
contains zero blocks even though the data is present on disk.

**Why it returns empty for all Era 1 heights**: There are no `qblock:dag:{h}:{proposer}`
keys in `CF_BLOCKS` for heights 0–13,477,999. The prefix scan finds nothing. The fallback
is exhausted.

### 3.3 Early-abort guard

**File**: `crates/q-storage/src/lib.rs` — lines 2116–2124

```rust
if consecutive_failures >= MAX_CONSECUTIVE_FAILURES && blocks.is_empty() {
    break; // abort after 10 consecutive deserialization failures with 0 blocks returned
}
```

`consecutive_failures` only increments when a key exists but deserialization fails. It
resets to 0 on both success and on `None` (missing). This means:

- Heights with all-`None` keys (Era 1): no early abort, but empty result
- Heights with corrupt Era 2 entries: early abort after 10 failures, empty result
- Heights with valid Era 2 entries: DAG fallback recovers them, no abort

---

## 4. The Fix: Extend the Read Path Without Touching Stored Data

The user's framing is exactly correct: **teach the code to read existing formats, not
rewrite what is stored**.

### 4.1 Fix deserialization for Era 2 blocks that currently fail

The `deserialize_qblock_with_fallback` function in `q_types::legacy` tries a sequence of
known binary layouts. Some Era 2 blocks were written during a period when the `QBlock`
struct had different field ordering, optional fields missing, or a pre-`QuantumMetadata`
layout.

**Action**: Audit the failing layouts by inspecting actual bytes from CF_BLOCKS at heights
known to fail (e.g., height 13,500,000). Add the missing layout variant to
`q_types/src/legacy.rs`. This is additive — no existing code path changes.

```rust
// In q_types/src/legacy.rs — deserialize_qblock_with_fallback():
// Add after existing fallback attempts:
if let Ok(block) = bincode::deserialize::<QBlockV8>(&data) {    // add missing era
    return Ok(block.into());
}
```

### 4.2 Add Era 1 read path for CF_DAG_VERTICES

For heights 0–13,477,999 the blocks were written to `CF_DAG_VERTICES` as raw vertex
payloads during the gossipsub era. The read path currently never queries this column family
from `get_qblocks_range()`.

**Action**: After the DAG fallback, add a third fallback that queries `CF_DAG_VERTICES`
for heights still not found. The vertex key scheme needs to be confirmed by auditing the
early DB entries (see Section 5), but the read-path extension follows the same pattern:

```rust
// After the qblock:dag: fallback in get_qblocks_range():
// ═══════════════════════════════════════════════════════
// Era 1 fallback: CF_DAG_VERTICES (heights 0–13,477,999)
// ═══════════════════════════════════════════════════════
let vertex_needed: Vec<u64> = still_missing_heights; // heights not found by primary or DAG fallback
for &height in &vertex_needed {
    let vertex_prefix = format!("vertex:{}:", height); // key scheme TBD — see Section 5
    if let Ok(entries) = self.hot_db.scan_prefix_seek(CF_DAG_VERTICES, vertex_prefix.as_bytes(), 1).await {
        if let Some((_key, value)) = entries.into_iter().next() {
            if let Ok(block) = reconstruct_qblock_from_vertex(&value, height) {
                blocks.push(block);
            }
        }
    }
}
```

This is read-only. The stored data is not touched; only the query path changes.

---

## 5. Investigation Required: Era 1 Key Scheme

Before implementing the Era 1 fallback, the actual key format in `CF_DAG_VERTICES` must be
confirmed. Two approaches:

**Option A — Key range scan**:
```bash
# On Epsilon, use RocksDB ldb tool to list keys in CF_DAG_VERTICES around height 1,000,000
ssh root@89.149.241.126 "ldb --db=/home/orobit/data-mainnet-genesis \
    --column_family=dag_vertices scan --from='vertex:1000000' --to='vertex:1000001' --max_keys=5"
```

**Option B — Storage code audit**:  
Search for `CF_DAG_VERTICES` write calls in the codebase:
```bash
grep -n "CF_DAG_VERTICES\|dag_vertices" crates/q-storage/src/*.rs | grep "put\|write\|save"
```

The write code will show exactly what key format was used, allowing the read path to be
written without guessing.

---

## 6. Why This is the Right Approach vs. Re-indexing

"Re-indexing" would mean iterating the old formats and writing new `qblock:height:{h}` keys
into CF_BLOCKS. This has two problems:

1. **Mainnet risk**: Writing millions of new keys to Epsilon's production DB while it is
   serving the live network risks I/O interference and RocksDB compaction pressure during
   a write storm. A bug in the migration writes could corrupt valid data.

2. **Not necessary**: The existing `scan_prefix_seek` infrastructure is already capable of
   read-time lookups. The DAG fallback (v10.3.6) proved this approach: no writes were needed
   to recover 545,710 Era 2 blocks; the read path was simply extended.

Extending the read path is:
- **Zero-risk to stored data** — read-only operations only
- **Incremental** — Era 2 deserialization fix and Era 1 read path can be added and tested
  independently
- **Reversible** — adding a code fallback can be reverted; a 13.5M-row write migration
  cannot be easily undone
- **Already proven** — the v10.3.6 DAG fallback is exactly this pattern, working in production

---

## 7. Impact on Fresh Node Sync

Once both fallbacks are fully functional:

| Sync scenario | Before fix | After fix |
|---|---|---|
| Heights 0–13.5M via P2P | 0 blocks returned | Blocks served from CF_DAG_VERTICES |
| Heights 13.5M–15.5M via P2P | Some blocks missing (deser failures) | All 545,710 DAG-era blocks served |
| Heights 15.5M+ via P2P | Already works | Unchanged |
| `/api/v1/sync/blocks?from_height=1` | Empty response | Full block stream from genesis |
| Fresh node declares itself "archive" | Cannot — incomplete history | Can, once all eras readable |
| `Q_ARCHIVE_NODE_URL` proxy (SYNC-001 fix) | Proxies from 15.5M only | Full genesis history via proxy |

---

## 8. Relationship to Other Open Issues

| Issue | Relationship |
|---|---|
| SYNC-001 (block history loss on fresh nodes) | This fix enables Epsilon to serve historical blocks; SYNC-001's archive proxy then makes those blocks reachable by any node pointing to Epsilon |
| SYNC-002 (transfer-skip balance bug, fixed v10.7.1) | Orthogonal — SYNC-002 was a balance-processing bug; this is a block-retrieval bug |
| BAL-001 (block 18,600,000 enforcement) | Indirectly related — nodes that need to audit historical balance roots will need historical block access |
| Genesis sync double-counting (observed 2026-05-08) | Separate: occurs because P2P snapshot + block replay double-applies transactions; fixed by checkpoint, not by this issue |

---

## 9. Recommended Next Steps

| Priority | Task | Effort |
|---|---|---|
| P1 | Audit CF_DAG_VERTICES key scheme at genesis heights (Section 5) | 1 hour |
| P1 | Fix Era 2 deserialization failures — add missing `QBlock` layout variant | 2–4 hours |
| P2 | Add Era 1 fallback read path in `get_qblocks_range()` | 4–8 hours |
| P2 | Test: `curl http://epsilon:8080/api/v1/sync/blocks?from_height=1&limit=10` must return blocks | 30 min |
| P3 | Declare Epsilon as `node_type=archive` in P2P announcements once all eras are readable | 1 hour |
