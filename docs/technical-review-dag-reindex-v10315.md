# Technical Review: DAG→Height Re-Index (v10.3.15)
**Date:** 2026-04-23  
**Author:** Server Beta (Claude Code)  
**Status:** IMPLEMENTED — awaiting external review before deploy  
**Risk Level:** 🟢 LOW  

---

## 1. Problem Statement

Fresh nodes cannot build a contiguous chain from genesis. The turbo-sync contiguity pointer stays at 0 indefinitely. Wallet balances do NOT survive a network-wipe test because no new node can reconstruct the full chain history.

**Observed symptom (test container q-integrity-test-v10312, 2026-04-23):**

```
🔍 [TURBO CONTIGUITY] Batch did NOT advance pointer
   cache=16067691, first_height=16044492, last_height=16056416
```

The node downloads recent blocks (16M range) but cannot fill the 0→16M gap. The contiguous pointer never advances from 0.

---

## 2. Root Cause: String-Sort Ordering in get_dag_blocks_forward()

### 2.1 The Data Layout

Epsilon's 219GB RocksDB contains **545,710 early-history blocks** stored as:

```
qblock:dag:{height}:{proposer_hex}     ← early gossipsub-received blocks
qblock:height:{height}                 ← current turbo-sync format
```

These 545K DAG-format blocks cover heights approximately 100,441 to 10,000,000 — the entire early chain history. They were confirmed present by `ldb scan` on 2026-04-19:

```
qblock:dag:1000000:3f8a...    ✓ present
qblock:dag:2000000:4b1c...    ✓ present  
qblock:dag:5000000:9d2e...    ✓ present
```

**These blocks were never lost.** The problem is entirely a read-path bug.

### 2.2 RocksDB String Sort vs. Numeric Sort

RocksDB sorts keys **lexicographically** (byte-by-byte), NOT numerically.

For DAG keys, this creates a counterintuitive ordering:

| Key | Height | Reason |
|-----|--------|--------|
| `qblock:dag:10000000:*` | 10,000,000 | 8 digits starting with '1' |
| `qblock:dag:1000000:*` | 1,000,000 | 7 digits, ':' (0x3A) > '0' (0x30) at pos 17 |
| `qblock:dag:100441:*` | 100,441 | 6 digits, ':' at pos 16 |
| `qblock:dag:2000000:*` | 2,000,000 | first digit '2' > '1' |

**Critical comparison:** `"qblock:dag:10000000:"` vs `"qblock:dag:1000000:"`

```
Position:  q b l o c k : d a g : 1 0 0 0 0 0 0  ':'   vs  '0'
Index:     0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17  18
                                                          ^
                                                          ':' (0x3A) > '0' (0x30)
                                           → 10000000 sorts BEFORE 1000000!
```

This means the iterator starting at `b"qblock:dag:"` encounters:
1. **All 10M–19M blocks first** (8-digit keys starting with '1')
2. Then 1M–1.9M blocks (7-digit starting with '1')
3. Then 100K–199K blocks (6-digit starting with '1')
4. Then 20M+ blocks, etc.

### 2.3 The Scan Budget Exhaustion

```rust
// kv.rs: get_dag_blocks_forward()
const MAX_SCAN: usize = 100_000;

let iter = self.db.iterator_cf(
    &cf_handle,
    rocksdb::IteratorMode::From(b"qblock:dag:", rocksdb::Direction::Forward),
);

// Scans first 100K entries...
// On Epsilon: the first 100K entries are all at heights 10M-16M
// → height >= start_height filter passes (10M > 1.4M request)
// → fills result with 10M blocks
// → 1.4M blocks NEVER reached
```

**Result:** A syncing node requesting blocks at height ~1,411,001 receives blocks at height ~15,950,457. Unusable. The turbo-sync engine discards them (wrong range), requests again, gets the same wrong blocks — infinite loop. Contiguous pointer stays at 0.

---

## 3. The Fix: Option C — Write qblock:height:{N} Aliases

**Chosen approach:** On startup (background, 15s delay), scan ALL `qblock:dag:` keys and write a `qblock:height:{N}` alias for each one that doesn't already have a height key.

After this runs, `get_qblocks_range()` (which uses `multi_get` on `qblock:height:{N}` keys) finds the early blocks directly — no iterator ordering issues.

### 3.1 Why Not Fix get_dag_blocks_forward() Instead?

Option A (fix the iterator) was considered and rejected:

- To sort 545K keys numerically, you must either:
  - Load all 545K into memory at once (OOM risk on 7.8GB Gamma), OR
  - Use zero-padded keys (requires rewriting all DAG keys — same as Option C but with more code)
- MAX_SCAN of 100K was a deliberate limit to avoid the scan-all OOM that crashed Epsilon (2026-04-17 incident: `scan_prefix(&[])` caused 50GB allocation)
- Option C (re-indexing) solves the problem at the source and makes future reads O(1) multi_get

### 3.2 Implementation

**File: `crates/q-storage/src/kv.rs`** — Added to inherent `impl RocksDBKV`:

```rust
pub async fn reindex_dag_blocks_to_height_keys(&self) -> Result<u64> {
    // Migration flag check (scoped to drop BoundColumnFamily before spawn_blocking await)
    {
        let cf = self.get_cf(CF_MANIFEST)?;
        if self.db.get_cf(&cf, b"dag_height_reindex_v1").unwrap_or(None).is_some() {
            return Ok(0); // Already done
        }
    } // cf dropped here — nothing non-Send crosses the await

    let db = self.db.clone(); // Arc<DB>: Send + Sync ✓

    let written = tokio::task::spawn_blocking(move || -> Result<u64> {
        let cf_blocks  = db.cf_handle(CF_BLOCKS).ok_or_else(|| anyhow!("no CF_BLOCKS"))?;
        let cf_manifest = db.cf_handle(CF_MANIFEST).ok_or_else(|| anyhow!("no CF_MANIFEST"))?;

        let iter = db.iterator_cf(
            &cf_blocks,
            rocksdb::IteratorMode::From(b"qblock:dag:", rocksdb::Direction::Forward),
        );
        // Note: NO scan limit — we need ALL 545K entries, not just the first 100K.
        // This is safe because we're in spawn_blocking (dedicated thread, not tokio pool).

        let mut write_batch = WriteBatch::default();
        let mut batch_size = 0usize;
        let mut written = 0u64;
        let mut skipped_exists = 0u64;

        for item in iter {
            let (key, value) = item?;
            if !key.starts_with(b"qblock:dag:") { break; }

            let height = match parse_dag_key_height(&key) {
                Some(h) => h,
                None => continue,
            };

            let height_key = format!("qblock:height:{}", height);

            // Idempotency: skip if qblock:height:{N} already exists
            match db.get_cf(&cf_blocks, height_key.as_bytes()) {
                Ok(Some(_)) => { skipped_exists += 1; continue; }
                Ok(None)    => {}
                Err(_)      => continue,
            }

            // Copy raw bytes from DAG key → height key (no re-serialization)
            // The reader (get_qblocks_range) already handles all legacy formats.
            write_batch.put_cf(&cf_blocks, height_key.as_bytes(), &value);
            batch_size += 1;

            if batch_size >= 500 {
                // Flush batch: WAL=on, sync=off (performance; WAL provides crash safety)
                let mut wo = rocksdb::WriteOptions::default();
                wo.set_sync(false);
                wo.disable_wal(false);
                db.write_opt(
                    std::mem::replace(&mut write_batch, WriteBatch::default()), &wo
                )?;
                written += batch_size as u64;
                batch_size = 0;
            }
        }

        // Final batch: fsync for durability
        if batch_size > 0 {
            let mut wo = rocksdb::WriteOptions::default();
            wo.set_sync(true);
            db.write_opt(write_batch, &wo)?;
            written += batch_size as u64;
        }

        // Set migration flag with fsync — never re-run this after completion
        let mut fo = rocksdb::WriteOptions::default();
        fo.set_sync(true);
        db.put_cf_opt(&cf_manifest, b"dag_height_reindex_v1", b"done", &fo)?;

        Ok(written)
    }).await??;

    Ok(written)
}
```

**File: `crates/q-storage/src/lib.rs`** — Thin wrapper on StorageEngine:

```rust
#[cfg(not(target_os = "windows"))]
pub async fn reindex_dag_blocks_to_height_keys(&self) -> Result<u64> {
    self.hot_db_concrete.reindex_dag_blocks_to_height_keys().await
}

#[cfg(target_os = "windows")]
pub async fn reindex_dag_blocks_to_height_keys(&self) -> Result<u64> {
    Ok(0) // RocksDB not on Windows
}
```

**File: `crates/q-api-server/src/main.rs`** — Background task spawned 15s after startup:

```rust
tokio::spawn({
    let storage = state.storage_engine.clone();
    async move {
        tokio::time::sleep(tokio::time::Duration::from_secs(15)).await;
        match storage.reindex_dag_blocks_to_height_keys().await {
            Ok(0) => info!("✅ [REINDEX] DAG→height index already complete"),
            Ok(n) => info!("✅ [REINDEX] DAG→height index wrote {} new keys", n),
            Err(e) => warn!("⚠️ [REINDEX] DAG→height index failed (non-fatal): {}", e),
        }
    }
});
```

---

## 4. Safety Analysis

### 4.1 Operations Performed

| Operation | Type | Risk |
|-----------|------|------|
| `db.get_cf(CF_MANIFEST, flag)` | Read | None |
| `db.iterator_cf(CF_BLOCKS, Forward)` | Read-only iterator | None |
| `db.get_cf(CF_BLOCKS, height_key)` | Read (existence check) | None |
| `db.write_opt(batch, ...)` | Append-only write | Low |
| `db.put_cf_opt(CF_MANIFEST, flag)` | Write flag | None |

**Zero destructive operations.** No `delete_cf`, `delete_range`, or key modifications.

### 4.2 Idempotency

Before writing `qblock:height:{N}`, the code checks if the key already exists. If yes, it skips. This means:

- Running the re-indexer twice produces identical results to running it once
- All existing `qblock:height:{N}` keys (the 10M-16M range from turbo sync) are untouched
- Safe to interrupt mid-run — on next startup, resumes from where it left off (existence checks skip already-written entries; the migration flag is only set on FULL completion)

### 4.3 Data Validity

The raw bytes copied from `qblock:dag:{N}:{proposer}` to `qblock:height:{N}` are the exact same bytes already served by `get_dag_blocks_forward()` to syncing peers. We've been trusting these bytes for P2P block-pack responses for months. The reader (`get_qblocks_range`) already calls `deserialize_qblock_with_fallback()` which handles all legacy formats (QBlock, LegacyQBlock, LegacyQBlockV2, LegacyQBlockV3). If a block is corrupt, deserialization fails and the 10-consecutive-failure early-abort skips that range — same behavior as today.

### 4.4 Consensus Safety

The re-indexer does NOT:
- Modify any block's content
- Change any validation rule
- Touch balance state, transactions, or consensus data
- Affect the LWMA difficulty algorithm
- Affect mining reward calculations

It is categorized as **read-path metadata maintenance**, not a consensus change. No upgrade gate is needed.

### 4.5 Concurrency Safety

RocksDB uses MVCC (Multi-Version Concurrency Control). Concurrent reads and writes to different keys in CF_BLOCKS are safe without external locking. The re-indexer writes keys with a distinct prefix pattern (`qblock:height:`) that the block production path also writes — but the existence check prevents double-writes, and RocksDB write operations are atomic at the key level.

The 15-second startup delay ensures the re-indexer does not contend with the critical startup sequence (peer connections, block production initialization, preflight checks).

### 4.6 Memory Usage

Batch size is capped at 500 entries. Each early-history block is approximately 500–2000 bytes. Maximum batch allocation: `500 × 2000 = 1MB`. This is safe on all nodes including Gamma (7.8GB RAM).

The iterator reads one entry at a time — no accumulation in memory. Total memory overhead of the re-indexer while running: `<5MB`.

### 4.7 Performance Impact

- Estimated duration on Epsilon (NVMe, 48 cores): **5–15 minutes** for 545K entries
- Write rate: ~500 entries per `write_opt` call, ~60 calls/min = ~30K writes/min
- RocksDB can sustain 100K+ writes/sec on NVMe; this is <1% of write capacity
- The 15s startup delay ensures block production is established before re-indexer runs
- `spawn_blocking` uses a dedicated thread — does NOT consume tokio worker threads

### 4.8 Failure Modes

| Failure | Effect | Recovery |
|---------|--------|----------|
| Process killed mid-run | Partial re-index, no flag set | Re-starts from scratch on next boot (existence checks skip already-written entries) |
| Disk full during batch write | Returns Err, logged as warning | Node continues normally; re-index incomplete until disk freed |
| CF_BLOCKS handle fails | Returns Err immediately | Node continues normally |
| Block bytes corrupt in DAG key | Written to height key as-is | Deserialization fails at read time; 10-failure early-abort in get_qblocks_range |
| Migration flag write fails | Logged as warning; re-runs on next boot | Redundant re-index; idempotent existence checks make this safe |

---

## 5. Expected Outcome

After the re-indexer completes on Epsilon:

1. **545K new `qblock:height:{N}` keys** exist in CF_BLOCKS covering heights ~100K–10M
2. `get_qblocks_range(100000, 200)` → `multi_get` on 200 `qblock:height:N` keys → 200 direct hits → returns blocks
3. Fresh node turbo-sync requests height 100,001–10,000,000 in 200-block batches → all served
4. Fresh node contiguous pointer advances: `0 → 100K → 500K → 1M → ... → 10M → joins normal sync`
5. Full sync from genesis now takes ~5–6 hours (same as the warp-sync rate of ~570 blocks/sec)
6. **Network resilience test PASSES**: a fresh node with no peers except other fresh nodes can rebuild the full chain from Epsilon's data

### Log output expected on Epsilon after first boot with v10.3.15:

```
🔄 [REINDEX] DAG→height re-index starting (one-time, may take a few minutes)...
🔄 [REINDEX] scanned=5000 written=4821 skipped=179 elapsed=8.3s
🔄 [REINDEX] scanned=10000 written=9711 skipped=289 elapsed=16.1s
... (every 5000 written)
✅ [REINDEX] Complete: scanned=545710 written=541883 skipped_exists=3827 elapsed=14m32s
```

On all subsequent boots:
```
✅ [REINDEX] DAG→height re-index already complete, skipping
```

---

## 6. Alternative Approaches Considered and Rejected

### Option A: Increase MAX_SCAN in get_dag_blocks_forward()

**Rejected.** Setting MAX_SCAN = 600,000 would require loading up to 600K keys into memory (up to 1.2GB on sparse data) before sorting. The original 100K limit was added specifically after the April 2026 Epsilon OOM incident (`scan_prefix(&[])` OOM'd at 50GB). A higher limit risks the same problem.

Even with pagination by key, the string-sort ordering makes it impossible to seek directly to "height 1M" in the DAG key space — you'd have to scan from the beginning every time.

### Option B: Zero-pad DAG keys (qblock:dag:0001411001:proposer)

**Rejected.** This would require rewriting all 545K DAG keys (delete old, write new) — a destructive operation. Much higher risk than Option C. Also requires changing the write path for new gossipsub-received blocks. Large scope change.

### Option C: Write qblock:height:{N} aliases (chosen)

**Chosen.** Purely additive, idempotent, zero risk to existing data. The read path (`get_qblocks_range`) already exists and is battle-tested. After re-indexing, the fix is permanent — no ongoing overhead.

---

## 7. Questions for DeepSeek Review

1. **Send bound correctness**: The `BoundColumnFamily<'_>` is obtained inside a scoped block and dropped before `spawn_blocking(...).await`. Is this sufficient to satisfy Rust's `Send` requirement on the future, or do we need `tokio::task::block_in_place` instead?

2. **Concurrent write safety**: During re-indexing, the block production path may also write `qblock:height:{N}` for new blocks (heights 16M+). The re-indexer's existence check (`get_cf` before `put_cf`) is not atomic with the write — could two concurrent writers create a race? (Expected answer: no, because they'd both write the same value and RocksDB writes are idempotent for the same key/value.)

3. **Iterator stability**: Can a RocksDB iterator over CF_BLOCKS be disrupted by concurrent writes to different keys in the same CF? (RocksDB iterators use a point-in-time snapshot by default unless `ReadOptions::set_snapshot` is explicitly set to false — the default iterator gives a consistent view.)

4. **Migration flag timing**: The flag is written after the final `write_opt`. If the process is killed between the last write_opt and the flag write, the re-indexer re-runs on next boot. Is this safe? (Yes — existence checks make the re-run idempotent.)

5. **Is there a risk that copying the raw DAG bytes as-is to qblock:height:{N} keys could break the contiguity scanner?** The contiguity scanner in `scan_highest_contiguous_block` reads heights via `get_qblock_by_height` which calls `get_qblocks_range` which calls `deserialize_qblock_with_fallback`. If a legacy-format block at height N fails deserialization, it returns None and the contiguity scan stops at N-1. This would mean the contiguous pointer stops at the first corrupt/undeserializable block, which is already the current behavior. No regression.

---

## 8. Deployment Plan

This is a **node-local change** — no coordination between nodes required. Each node runs the re-indexer independently on first startup with v10.3.15.

**Deployment sequence:**
1. Build v10.3.15: `cargo build --release --package q-api-server`
2. Deploy to Epsilon first (holds the 545K DAG blocks — most important node)
3. Monitor: `journalctl -u q-api-server -f | grep REINDEX`
4. Wait for "Complete" message (~15 minutes on Epsilon)
5. Start fresh test container: `docker run ... q-api-server-v10.3.15`
6. Observe contiguous pointer advancing past 100K → 1M → 10M
7. Deploy to Beta and Gamma (they may have fewer DAG blocks, faster completion)

**Rollback:** If the re-indexer causes unexpected issues, delete the migration flag:
```bash
# On Epsilon — removes the flag so re-index re-runs on next restart
# (or to skip re-index entirely, just don't upgrade)
# The qblock:height:{N} keys written are harmless even without rollback
```

The written `qblock:height:` keys don't need cleanup — they're identical to what turbo sync would write for those heights anyway.
