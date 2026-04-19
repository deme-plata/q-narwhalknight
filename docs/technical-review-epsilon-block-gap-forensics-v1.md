# Technical Review: Epsilon Block Gap Forensics
## Root Cause Analysis — Missing Blocks 1.6M–10M
### Date: 2026-04-16 | Investigator: Server Beta (Claude Code)

---

## 1. Executive Summary

Epsilon (89.149.241.126) — the 10Gbit supernode and primary sync source — is missing blocks between heights 1,645,947 and ~10,000,031. New nodes syncing from Epsilon only receive blocks from ~10M onwards, missing ~8.4M blocks of chain history (roughly Feb 28 – March 28, 2026).

**The blocks did exist at some point.** Epsilon's oldest SST files date to March 2, 2026 (day 8 of mainnet, ~block 2.3M). But the block data is no longer accessible.

**Financial impact: ZERO.** All mining rewards from the missing period are accounted for in wallet balances via P2P state sync. Those blocks contained only coinbase transactions (no user transfers, no DEX swaps).

**Network impact: HIGH.** New nodes get an incomplete chain. Block explorer shows gaps. Chain integrity verification cannot fully validate the history.

---

## 2. Evidence Collected

### 2.1 RocksDB Key Scan Results

**Epsilon (89.149.241.126) — DB: 181 GB, 67,989 SST files:**

| Key Format | Earliest Found | Latest Found | Notes |
|------------|---------------|-------------|-------|
| `qblock:height:{N}` | **10,000,031** | 15,604,748 (tip) | ~5.6M blocks continuous |
| `qblock:dag:{N}:{hash}` | **100,441** | ~1,015,487 | Only ~50 sparse entries at 100K + 1M |
| `qblock:hash:{hash}` | (not scanned) | (not scanned) | Reverse index, not primary storage |

**Beta (185.182.185.227) — DB: 104 GB:**

| Key Format | Earliest Found | Latest Found |
|------------|---------------|-------------|
| `qblock:height:{N}` | **10,005,233** | 15,606,880 (tip) |
| `qblock:dag:{N}:{hash}` | **13,496,194** | (recent) |

**Both bootstrap nodes have nearly identical `qblock:height:` start points (~10M).**

### 2.2 SST File Age Distribution (Epsilon)

| Period | SST Count | Approx Chain Height | Notes |
|--------|-----------|-------------------|-------|
| March 1-15, 2026 | **867** | blocks 1.6M – 6M | Earliest: March 2 (000086.sst) |
| March 15-31 | 12,532 | blocks 6M – 10M | Heavy compaction period |
| April 1-15 | 45,270 | blocks 10M – 15.4M | Active chain growth |
| April 16 | 10,215 | blocks 15.4M – 15.6M | Today |

**867 SST files survive from the March 1-15 period.** However, examining the 5 oldest SSTs (from March 2):

| SST File | Column Family | Entries | Size |
|----------|--------------|---------|------|
| 000086.sst | `cf_qno_stats` | 1 | 1.3 KB |
| 013389.sst | `transactions` | 61,743 | 135 MB |
| 017749.sst | `transactions` | 39,299 | 110 MB |
| 028921.sst | `cf_emails_by_folder` | 1 | 1.5 KB |
| 028923.sst | `cf_emails_by_wallet` | 1 | 1.5 KB |

**None of the surviving March SSTs are in the `blocks` column family.** The `transactions` SSTs from March 2 prove data WAS being written, but the `blocks` CF SSTs from that era have been compacted away.

### 2.3 DB Corruption Evidence

```
Corruption: IO error: No such file or directory: 
  /home/orobit/data-mainnet-genesis/hot/8694668.sst
  MANIFEST-8546725 may be corrupted
  /home/orobit/data-mainnet-genesis/hot/8695152.log: No such file or directory
```

- `8694668.sst` — missing (recent high-numbered SST, compaction output)
- `8695152.log` — missing WAL file
- MANIFEST references these files but they don't exist on disk

### 2.4 Block Storage Architecture

The codebase has TWO write paths for blocks:

```
Path 1: save_dag_layer_block() → "qblock:dag:{height}:{proposer_hex}"
  File: crates/q-storage/src/lib.rs:1090
  Used by: gossipsub block receiver (P2P incoming blocks)
  
Path 2: save_qblock() → "qblock:height:{height}"
  File: crates/q-storage/src/lib.rs (turbo_sync batch writer)
  Used by: turbo sync bulk download
```

The read path for serving blocks to peers:
```
get_qblocks_range() → multi_get on "qblock:height:{N}" keys ONLY
  → Fast path, returns empty if no qblock:height: keys exist

get_qblocks_range_any_format() → falls back to:
  1. get_qblocks_range() [fast path]
  2. get_qblock_any_format(height) [slow path, per-block]:
     a. get_qblock_by_height() → "qblock:height:{N}"
     b. scan_prefix("qblock:dag:{N}:") → finds DAG entries
     c. get by binary key (height.to_be_bytes())
  3. Early abort after 20 consecutive misses
```

### 2.5 The Checkpoint Sync HTTP Probe (Broken)

```rust
// turbo_sync.rs:3187 — probe_network_gap_blocking()
let bootstrap_urls = ["http://185.182.185.227:8080", "http://89.149.241.126:8080"];

// Checks /api/v1/blocks/{height} for each URL
// BUT: this endpoint returns HTTP 404 with empty body for ALL heights
// Result: probe ALWAYS falls through to "No blocks found on peers"
```

The `/api/v1/blocks/{height}` endpoint is broken — returns 404 for every height, even blocks that exist.

---

## 3. Timeline Reconstruction

| Date | Block Height | Event | Evidence |
|------|-------------|-------|----------|
| Feb 22 | 0 | Mainnet genesis | Genesis timestamp 1771761600 |
| Feb 28 | ~1,645,947 | Pruning bug deletes blocks 0–1.6M | technical-review-gap-detection doc |
| Mar 2 | ~2.3M | **Epsilon's oldest SST files** (transactions CF) | SST 013389.sst, 017749.sst |
| Mar 2 | ~2.3M | Epsilon syncing blocks via gossipsub → `qblock:dag:` keys | `qblock:dag:100441-100443` survives |
| Mar 5 | ~3.2M | Epsilon binary updated to v911 | Memory note: ExecStart updated |
| ~Mar 10-15 | ~5-6M | **UNKNOWN EVENT** — blocks CF SSTs from this era are gone | 867 surviving SSTs are transactions/metadata, not blocks |
| ~Mar 28 | ~10M | `qblock:height:` keys begin at 10,000,031 | RocksDB scan confirms |
| Apr 8 | ~13.6M | First surviving checkpoint | checkpoint_13647476.json.tmp |
| Apr 12 | ~14.3M | Pruning bug fixed (v10.3.0) | technical-review-gap-detection doc |
| Apr 16 | 15.6M | Current tip | Live data |

**The critical gap**: Between March 2 (when transactions SSTs prove Epsilon was active) and March 28 (when `qblock:height:` keys begin), the `blocks` CF data was lost. The `transactions` CF data from that era survived in old SSTs, but the corresponding `blocks` CF SSTs did not.

---

## 4. Hypotheses

### Hypothesis A: Compaction Ate The Blocks (Most Likely)

RocksDB level compaction merges SSTs from lower levels into higher levels. During compaction:
1. Old SSTs (containing blocks 1.6M-10M in `blocks` CF) are read
2. New merged SST is written (e.g., `8694668.sst`)
3. Old SSTs are deleted (normal compaction behavior)
4. If the new SST is lost (crash, `kill -9`, disk error), the data is gone

**Supporting evidence:**
- `8694668.sst` is MISSING (confirmed) — high-numbered = recent compaction output
- Multiple documented `kill -9` incidents on Epsilon (memory notes: `kill9_corrupt_block_fix.md`)
- The `blocks` CF undergoes heavy compaction (15M+ blocks = billions of key-value pairs)
- `transactions` CF SSTs survived because they're separate and compacted on a different schedule

**Counter-evidence:**
- Compaction is normally atomic (old SSTs deleted only after new SST is fsync'd)
- BUT `kill -9` during compaction can leave the DB in an inconsistent state where the MANIFEST references a file that was never fully written

### Hypothesis B: DB Reset / Wipe Around March 15-28

If the Epsilon DB was manually wiped and restarted (e.g., during a version upgrade or bug fix), blocks before the restart point would be lost.

**Supporting evidence:**
- Multiple version upgrades documented in March (v7.x → v8.x → v9.x)
- Memory notes mention "cargo clean" requirements after constant changes
- Service file was updated March 5 (binary swap)

**Counter-evidence:**
- If the DB was wiped, the oldest SSTs would date from the wipe, not from March 2
- 867 SSTs from March 1-15 survive, suggesting the DB was NOT wiped

### Hypothesis C: Blocks Were Never in `qblock:height:` Format Before 10M

Early chain versions may have stored blocks ONLY as `qblock:dag:` entries via `save_dag_layer_block()`. The `qblock:height:` format was introduced later (possibly around March 28 when the keys begin at 10M).

**Supporting evidence:**
- The `save_dag_layer_block()` path is the P2P gossipsub receiver path
- Early Epsilon would have received blocks via gossipsub (not turbo sync from a peer)
- `qblock:dag:100441-100443` at height 100K are gossipsub-received blocks
- `qblock:height:` keys may have only been created by turbo sync (which wasn't active in early chain)
- Only 50 `qblock:dag:` entries survive (the rest were compacted away per Hypothesis A)

**Counter-evidence:**
- If blocks existed as `qblock:dag:`, the `get_qblock_any_format()` fallback should find them when peers ask
- But `get_qblock_any_format()` does `scan_prefix("qblock:dag:{height}:")` which IS correct
- So either the `qblock:dag:` data was also compacted/lost, or the scan_prefix fails

### Hypothesis D: String-Sorted Key Problem in Compaction

RocksDB sorts keys lexicographically (byte order). Numeric height keys stored as decimal strings sort incorrectly:

```
String sort:  "100441" < "1015441" < "10000031" < "2000000" < "5000000"
Numeric sort: 100441 < 1015441 < 2000000 < 5000000 < 10000031
```

This means `qblock:height:10000031` sorts BEFORE `qblock:height:2000000` in RocksDB. During compaction, keys from vastly different numeric heights can end up in the same SST file. If that file is lost, it creates seemingly random gaps across the numeric height range.

**Supporting evidence:**
- Confirmed: `qblock:height:10000031` is the first key when scanning from `qblock:height:` prefix
- This would explain why blocks at numeric height 2M are missing even though "lower" (string-order) keys exist
- The same SST file could have contained keys for heights 10M, 2M, 3M, 5M all mixed together

**This is the most likely contributing factor to the confusing gap pattern.**

---

## 5. The 181 GB Mystery

Epsilon's DB is 181 GB but only has ~5.6M blocks in `qblock:height:` format. Where's the rest?

**Breakdown estimate:**
- `blocks` CF with ~5.6M blocks × ~2-5 KB average = 11-28 GB
- `transactions` CF (61K+ entries in oldest SST alone, 15.6M blocks total) = 50-80 GB
- `quantum_metadata` CF = 10-20 GB
- Other CFs (balances, tokens, DEX, AI, emails, etc.) = 5-10 GB
- RocksDB overhead (bloom filters, indexes, compression metadata) = 10-20 GB
- **Compaction space amplification**: RocksDB can use 1.1-2x the logical data size due to level structure
- **Tombstones and deleted data**: `delete_by_prefix` operations leave tombstone markers until compaction

Plausible: 181 GB ≈ 80 GB logical data × 2.2x space amplification.

**The "missing" blocks don't necessarily account for significant space.** Blocks from 1.6M-10M at ~2 KB each = ~16 GB, which is well within the compaction amplification margin.

---

## 6. Questions for DeepSeek

**CRITICAL CONTEXT: This is a LIVE $1.1B mainnet. Any recommendation must be ZERO RISK to the current chain state. We would rather accept the gap permanently than risk making it worse. Every proposed action must answer: "What happens if this goes wrong? Can we undo it? What's the worst case?"**

### Q1: Can RocksDB lose data during compaction interrupted by kill -9?

We have documented evidence of `kill -9` on Epsilon (see `kill9_corrupt_block_fix.md`). The MANIFEST references `8694668.sst` which does not exist on disk. Can this happen if:
a) Compaction was creating `8694668.sst` (writing output SST)
b) `kill -9` hit during the write (before fsync)
c) On restart, MANIFEST was already updated to reference the new SST, but the file is incomplete/missing

If yes: is there a way to detect all keys that were in the lost SST file by analyzing the surviving MANIFEST and SST files?

### Q2: Can `ldb repair` recover data from this specific corruption pattern? Is it safe?

`ldb repair --db=<path>` rebuilds the MANIFEST by scanning all SST files. If the blocks from 1.6M-10M were in old SST files that have since been compacted into the now-missing `8694668.sst`, would repair recover them? Or would it only recover the current state (which already excludes those blocks)?

Specifically: after a compaction merges SSTs A+B into SST C, are SSTs A and B immediately deleted? If so, repair can only find what's in SST C (which is missing). If there's a delay, old SSTs might still exist.

**SAFETY QUESTION: Can `ldb repair` ever make things WORSE? Can it accidentally delete existing valid data? Can it corrupt the MANIFEST further? We will NOT run repair on production until you confirm it is strictly additive (can only recover data, never lose data). And even then, only on a copy first.**

### Q3: Is the string-sorted key problem causing hidden data loss?

Our block height keys use decimal string encoding:
```
qblock:height:10000031  (sorts before qblock:height:2000000 in byte order)
```

This means during compaction, keys for numerically distant heights are interleaved in the same SST files. If a single SST file is lost, keys from multiple numeric height ranges become invisible simultaneously.

**Should we migrate to zero-padded keys?** e.g., `qblock:height:00010000031` (11-digit zero-padded) so that RocksDB sort order matches numeric order? This would:
- Prevent cross-range key mixing in SST files
- Make range scans efficient (no need for multi-format fallback)
- Allow RocksDB bloom filters to work correctly on height ranges
- Make `ldb scan --from --to` give correct numeric ordering

What's the safest way to migrate 5.6M+ existing keys to zero-padded format without downtime?

### Q4: Why does `get_qblock_any_format()` fail to find `qblock:dag:` entries?

The function does `scan_prefix("qblock:dag:{height}:")` for each height in the range. On Epsilon, `qblock:dag:100441:e4034c57046136a4` exists (confirmed by ldb). But when a syncing peer requests blocks at height 100,441, they get 0 blocks.

Possible explanations:
a) The block-pack handler uses `get_qblocks_range()` (fast path) which skips DAG format entirely
b) When fast path returns 0, the fallback `get_qblocks_range_any_format()` is called but aborts after 20 consecutive misses (before reaching height 100,441 in the requested range)
c) The `scan_prefix` on the `blocks` CF is broken due to the MANIFEST corruption

Can you trace the exact code path when a peer requests blocks starting at height 100,000? Does the `get_qblocks_range_any_format()` fallback actually fire, and if so, does it reach the `scan_prefix("qblock:dag:100441:")` call?

### Q5: What is the safest path forward for a $1.1B mainnet?

Given that the gap is COSMETIC (all balances are correct, no user funds affected), rank these options by risk:

**Option A: Re-index existing data.** If `qblock:dag:` entries exist for heights 1.6M-10M (even sparsely), create `qblock:height:` aliases for them. Then `get_qblocks_range()` fast path would find them. Risk: write operations to production DB. What if the re-indexing corrupts existing data?

**Option B: Replay from transactions CF.** The `transactions` CF has entries from March 2 (oldest surviving SST). Can we reconstruct block structures from transaction data + quantum_metadata? Risk: reconstructed blocks might not match originals (different serialization, missing fields).

**Option C: Accept the gap and move forward.** Declare blocks 0-10M as historical (balances accounted for, no user transactions). New nodes start from 10M via checkpoint. Add a genesis state snapshot at height 10M. Risk: ZERO — no DB modifications.

**Option D: Copy DB, repair copy, verify, then decide.** rsync the 181 GB DB to a separate directory. Run `ldb repair` on the copy. If blocks are recovered: verify integrity. If the procedure is proven safe, THEN apply to production during a maintenance window. Risk: LOW (only operates on copy).

**For a $1.1B mainnet, we strongly lean toward Option C (accept) or Option D (copy-first). Please confirm our instinct or argue for a different approach. What would you do if this were Bitcoin's mainnet?**

### Q6: Can we prevent this from happening again?

Proposed mitigations:
1. **Never use `kill -9` on the node process** (use SIGTERM with 30s timeout)
2. **Enable RocksDB paranoid file checks** (`paranoid_file_checks = true` in column family options)
3. **Enable checksum verification on compaction** (`verify_checksums_in_compaction = true`)
4. **Periodic MANIFEST backup** (copy MANIFEST to a safe location every hour)
5. **Zero-padded height keys** (prevent cross-range mixing in SST files)
6. **Background block archival** (periodically dump blocks to a separate archive for disaster recovery)

Which of these are worth the performance trade-off? Are there other mitigations we're missing?

### Q7: Is this a known RocksDB failure mode?

The specific pattern is:
- Large DB (181 GB, 67K SST files)
- Heavy write workload (15M+ blocks, each 2-5 KB)
- Frequent compaction (level-based, default settings)
- `kill -9` during compaction
- Result: MANIFEST references non-existent SST file

Is this documented in RocksDB issues? Are there known configuration changes that prevent it?

---

## 7. MAINNET SAFETY CONSTRAINTS ($1.1B Market Cap)

### Non-Negotiable Rules

This is a **live mainnet with $1.1B market capitalization**, 279 wallets, 40+ active miners, and real user funds. Any recovery action must satisfy ALL of the following:

1. **ZERO risk of making it worse.** If there is any chance an action could corrupt additional data, delete existing blocks, or affect wallet balances — we do NOT do it.

2. **No production downtime for speculative fixes.** Epsilon serves 10Gbit sync traffic to every new node joining the network. Stopping it for a repair that MIGHT recover old blocks is not worth the risk to current users who need sync NOW.

3. **Read-only investigation first.** Every diagnostic step must be provably read-only. No writes to the production DB. No MANIFEST modifications. No compaction triggers. Copy-on-read only.

4. **The gap is cosmetic, not financial.** All 279 wallet balances are correct (verified via P2P state sync). The missing blocks contained ONLY coinbase transactions. No user has lost funds. The urgency is "nice to have history" not "emergency recovery."

5. **The chain continues to grow correctly.** Blocks 10M → 15.6M (and counting) are intact. New blocks validate correctly. Mining works. DEX works. The network is healthy. Don't break what works to fix what's missing.

6. **Any fix must be tested on a disposable copy first.** Before touching Epsilon's production DB, the exact same procedure must succeed on a byte-for-byte copy of the DB on a separate disk/machine.

### What we WILL do (Zero Risk):

| Action | Risk | When |
|--------|------|------|
| Document the gap in genesis config | ZERO | Now |
| Copy MANIFEST hourly to backup dir | ZERO (read-only copy) | Now |
| Add `kill -9` prevention to all service files | ZERO | Now |
| Scan Epsilon DB read-only with `ldb` / `sst_dump` | ZERO (read-only tools) | Now |
| Check Beta DB for blocks Epsilon is missing | ZERO | Now |
| Check if any other node on the network has the blocks | ZERO (P2P query) | Now |

### What we WILL NOT do (Until Proven Safe on Copy):

| Action | Risk | Why Not Yet |
|--------|------|-------------|
| `ldb repair` on production | **HIGH** — repair rewrites MANIFEST, could make corruption worse | Must test on DB copy first |
| Zero-padded key migration on production | **HIGH** — rewrites millions of keys, one bug = total data loss | Must be thoroughly reviewed + tested |
| Stop Epsilon for maintenance | **MEDIUM** — degrades sync for all new nodes | Only if we have a proven recovery procedure |
| Delete/recreate SST files | **CRITICAL** — irreversible data destruction | NEVER on production |
| Modify compaction settings on live DB | **MEDIUM** — could trigger unexpected compaction cascade | Must test on copy first |

### Recovery Strategy (If We Pursue It):

```
Phase 1: INVESTIGATE (read-only, zero risk)
  ├── Full ldb scan of ALL key prefixes in blocks CF
  ├── sst_dump every SST file → catalog which CF/keys each contains
  ├── Check Beta, Gamma, Delta (when back) for blocks Epsilon is missing
  ├── Check if any community miner node has full history
  └── Present findings to DeepSeek for review

Phase 2: COPY + TEST (zero risk to production)
  ├── rsync Epsilon DB to a separate directory (/home/orobit/db-recovery-copy/)
  ├── Run ldb repair on the COPY
  ├── Scan recovered DB for blocks 1.6M-10M
  ├── If blocks recovered: verify data integrity (deserialize, check hashes)
  └── If no recovery: accept the gap, document, move on

Phase 3: APPLY (only if Phase 2 succeeds AND passes DeepSeek review)
  ├── Schedule maintenance window (low-traffic hours, announce 24h ahead)
  ├── Stop Epsilon
  ├── Backup current MANIFEST to safe location
  ├── Apply the proven repair procedure
  ├── Verify: scan blocks CF, confirm recovery, check no regression
  └── Restart Epsilon
```

### The "Accept the Gap" Option

If recovery proves impossible or too risky, the correct response is:

1. **Document the gap** in the genesis configuration: "Block history available from height 10,000,031. Earlier blocks (coinbase-only, Feb 22 – Mar 28) were lost to a storage incident. All mining rewards from that period are preserved in wallet balances."

2. **Update the checkpoint sync** to start from 10M by default (instead of probing and sometimes landing at 14M).

3. **Add a genesis snapshot** at height 10M that includes the complete UTXO set / balance state, so new nodes don't need blocks before 10M at all.

This is the **safest option** and arguably the right one for a $1.1B mainnet. Bitcoin itself had a similar early history where block data from 2009 was lost by many nodes — the network didn't stop. What matters is the current state is correct, and it is.

---

## 8. Data Summary

```
Epsilon DB: /home/orobit/data-mainnet-genesis/hot
  Size: 181 GB
  SST files: 67,989
  MANIFEST: MANIFEST-8546725 (29 MB, corruption detected)
  Missing files: 8694668.sst, 8695152.log
  
  blocks CF key ranges:
    qblock:height:  10,000,031 → 15,604,748 (5.6M blocks, continuous)
    qblock:dag:     100,441 → ~1,015,487 (50 entries, sparse)
    
  transactions CF: entries from March 2 survive (61K+ in oldest SST)
  
  Confirmed gap: heights 1,645,948 → 10,000,030 (8.35M blocks missing)
  Financial impact: ZERO (balances correct via P2P state sync)
  Chain integrity: DEGRADED (cannot verify historical blocks in gap)
```

---

---

## 9. Final Note to DeepSeek

We are the engineers responsible for a $1.1B blockchain with real user funds. We discovered that ~8.4M blocks of early history are missing from our primary sync node due to what appears to be RocksDB SST file loss during a `kill -9` compaction interruption.

**We are NOT panicking.** The chain is healthy, balances are correct, and the network continues to produce blocks normally. The missing blocks contained only coinbase transactions from the first month of operation.

**We ARE being cautious.** We refuse to touch the production database with any write operation until we have:
1. A complete understanding of the root cause
2. A proven recovery procedure tested on a disposable copy
3. Independent expert verification that the procedure cannot make things worse

**We are comfortable with accepting the gap permanently** if the recovery risk exceeds the benefit. An incomplete block explorer is far less costly than a corrupted database on a $1.1B mainnet.

Please review our analysis, confirm or correct our hypotheses, and advise on the safest path forward. We value safety over completeness.

---

*Generated 2026-04-16 — Quillon Foundation*
*Epsilon forensics performed via read-only `ldb scan` and `sst_dump` (no writes to production DB)*
