# TR-2026-004: DAGKNIGHT SIMD + io_uring Consensus Optimization

**Author**: Server Beta (Claude Code)
**Date**: 2026-03-13
**Status**: PROPOSED — Awaiting Peer Review
**Branch**: `feature/safe-batched-sync-v1.0.2`
**Risk Level**: LOW-MEDIUM (all feature-gated, no consensus changes)

---

## 1. Abstract

This technical review proposes four optimizations to the DAGKNIGHT consensus implementation in Q-NarwhalKnight. These target the **node implementation layer** — how fast nodes execute the consensus algorithm — without modifying the consensus mathematics or block validation rules.

The optimizations address three gaps where existing infrastructure (`q-crypto-simd`, `q-kernel-io`, `q-flux`) is built but not wired into the consensus hot path:

| Gap | Current State | Proposed Fix | Expected Gain |
|-----|---------------|-------------|---------------|
| Sequential sig verify | `tx.verify_signature()` in a loop | Batch via `ParallelEd25519Verifier` | 5-10x |
| HashSet DAG operations | `HashSet<[u8;32]>` for anticone sets | Bitfield + SIMD vectorization | 50-100x for set ops |
| Blocking WAL fsync | `db.flush_wal(true)` blocks 2ms | io_uring async fsync | 40% latency reduction |
| Sequential pipeline | receive → validate → store | 3-stage overlapping pipeline | ~50% throughput |

All changes are backward-compatible and feature-gated.

---

## 2. Current Architecture Analysis

### 2.1 DAG Ordering Engine

**File**: `crates/q-dag-knight/src/ordering_rules.rs`

The `OrderingEngine` maintains the causal graph as:
```rust
causal_graph: RwLock<HashMap<VertexId, HashSet<VertexId>>>
```
Where `VertexId = [u8; 32]`.

**Performance characteristics**:
- `HashSet::contains([u8; 32])`: ~40ns (SipHash + equality check + cache miss)
- `HashSet::intersection()`: O(min(|A|, |B|)) with per-element hash lookups
- For N=10,000 active vertices, anticone computation: ~10,000 × 40ns = ~400μs

**Memory characteristics**:
- Each `HashSet<[u8; 32]>` entry: 32 bytes (key) + 8 bytes (hash) + ~24 bytes (bucket overhead) = ~64 bytes
- For 10,000 vertices with average 5,000 causal relationships: ~320MB
- Cache unfriendly: hash table entries scattered across heap

### 2.2 Block Signature Verification

**File**: `crates/q-api-server/src/main.rs:5295-5355`

Current implementation:
```rust
for block in &blocks {
    for tx in &block.transactions {
        if !tx.is_coinbase() {
            tx.verify_signature()?;  // Ed25519 verify, ~50μs each
        }
    }
}
```

**Existing but unused**: `crates/q-crypto-simd/src/parallel_ed25519.rs` — `ParallelEd25519Verifier` with rayon-based batch verification. Benchmarked at 25x speedup for batches >64 signatures.

### 2.3 Storage Write Path

**File**: `crates/q-storage/src/safe_batched_writer.rs`

```
receive blocks → batch into WriteBatch → db.write(batch) → db.flush_wal(true) → repeat
                                                              ↑
                                                         BLOCKS ~2ms
```

**Existing but stubbed**: `crates/q-kernel-io/src/uring.rs` — `IoUringEngine` with scalar fallback (no real io_uring ops). The **real** io_uring implementation is in `crates/q-flux/src/io_uring_loop.rs` (production-proven, proper SQE/CQE handling).

### 2.4 Async Pipeline

**File**: `crates/q-storage/src/async_pipeline.rs`

Reddio-style pipeline with hot cache, prefetch workers, and `pipeline_depth: 3`. Currently handles balance reads but NOT block validation or DAG ordering.

---

## 3. Proposed Optimizations

### 3.1 Phase 1: Batch Signature Verification

**Scope**: ~50 lines in `main.rs`
**Risk**: NONE — mathematically equivalent
**Feature gate**: None needed (always-on, strictly faster)

**Change**: Replace the sequential `tx.verify_signature()` loop with:
```rust
let all_sigs: Vec<(Message, Signature, PublicKey)> = blocks.iter()
    .flat_map(|b| b.transactions.iter())
    .filter(|tx| !tx.is_coinbase())
    .map(|tx| (tx.signing_message(), tx.signature(), tx.sender_pubkey()))
    .collect();

let results = ParallelEd25519Verifier::verify_batch_parallel(&all_sigs);
```

**Correctness argument**: Ed25519 batch verification is a well-studied optimization (see [dalek-cryptography batch verification](https://docs.rs/ed25519-dalek/latest/ed25519_dalek/#batch-signature-verification)). The existing `ParallelEd25519Verifier` uses rayon `par_iter` to distribute individual verifications across cores — it does NOT use probabilistic batch verification, so it is deterministically equivalent to sequential verification.

**Failure handling**: On batch failure, re-verify individually to identify the specific invalid signature and reject only the containing block.

**Performance model**:
- Current: N signatures × 50μs/sig = N × 50μs (sequential)
- Proposed: N signatures / num_cores × 50μs/sig (parallel)
- On Epsilon (48 cores): 1000 sigs in ~1ms instead of ~50ms

### 3.2 Phase 2: Bitfield DAG Representation

**Scope**: New module `crates/q-dag-knight/src/simd_sets.rs` + feature gate in `ordering_rules.rs`
**Risk**: LOW — feature-gated, node-local computation
**Feature gate**: `#[cfg(feature = "simd-dag")]`, default OFF

**Data structure**:
```
VertexId [u8; 32]  ──map──>  compact index u32  ──index into──>  bit position in u64 array

Past set of vertex i:     [u64, u64, u64, ...] where bit j = 1 means j is in past(i)
Future set of vertex i:   [u64, u64, u64, ...] where bit j = 1 means j is in future(i)
Anticone of vertex i:     NOT(past[i]) AND NOT(future[i]) AND NOT(self_mask[i])
```

**SIMD acceleration**:
```
                     HashSet           AVX2 (256-bit)     AVX-512 (512-bit)
Vertices per op:     1                 4 (4×u64)          8 (8×u64)
Anticone (10K):      10,000 ops        2,500 ops          1,250 ops
Time (10K):          ~400μs            ~10μs              ~5μs
Speedup:             1x                40x                80x
```

**Correctness argument**: The bitfield representation is a bijective mapping from set theory to binary arithmetic. Set intersection = bitwise AND, set union = bitwise OR, set complement = bitwise NOT. These operations are associative, commutative, and distribute identically to their set-theoretic counterparts. The topological sort output is compared via proptest against the existing `HashSet`-based implementation.

**Memory model**:
- Bitfield for 10,000 vertices: 10,000 × ceil(10,000/64) × 8 bytes = ~12.5MB
- vs HashSet: ~320MB for equivalent coverage
- **25x memory reduction** with better cache locality

**Index lifecycle**:
- New vertices get the next available index via `VertexIndexMap::assign(vertex_id) -> u32`
- Old vertices freed during `cleanup_cache()` (already called periodically)
- Index recycling: freed indices added to a `VecDeque<u32>` free list
- Bitfield capacity grows in powers of 2 (amortized O(1) growth)

### 3.3 Phase 3: io_uring WAL Sync

**Scope**: Modify `safe_batched_writer.rs` flush path
**Risk**: MEDIUM — new I/O path, but RocksDB WAL remains source of truth
**Feature gate**: `#[cfg(all(target_os = "linux", feature = "uring-storage"))]`, default OFF

**Architecture**:
```
                    BEFORE                           AFTER
                    ──────                           ─────
Batch N:  [prepare]──[write]──[fsync]──             [prepare]──[write]──[submit_fsync]──
                                       │                                                │
Batch N+1:                             └──[prepare]──[write]──[fsync]──     [prepare]──[write]──[wait_if_needed]──
                                                                        ↑
                                                               fsync overlapped with
                                                               next batch preparation
```

**Why NOT replace RocksDB entirely**:
RocksDB manages its own LSM-tree compaction, column family isolation, write-ahead log format, and snapshot semantics. Replacing it with raw io_uring file I/O would require reimplementing all of these — thousands of engineer-years of battle-tested code. Instead, io_uring supplements RocksDB at the kernel level:

1. RocksDB writes to its WAL file (standard `write()` syscall, buffered)
2. We call `io_uring_prep_fsync()` on the WAL file descriptor
3. While fsync drains to disk, we prepare the next write batch
4. We check completion before the next `db.write()` to ensure durability

**Kernel requirements**: io_uring requires Linux 5.1+. All production servers run 6.1.0-37-amd64.

**Correctness argument**: RocksDB's WAL durability guarantee is maintained because:
- `db.write(batch)` returns after the write is in the WAL (page cache)
- Our io_uring fsync ensures the WAL reaches persistent storage
- The only change is that fsync is non-blocking, allowing overlap with CPU work
- If io_uring fails, we fall back to standard `db.flush_wal(true)` (blocking)

**Performance model**:
- Current fsync latency: ~2ms on NVMe (Epsilon), ~5ms on SSD (Gamma)
- io_uring fsync: same disk latency, but overlapped with next batch preparation (~1.5ms effective)
- Net improvement: 20-40% reduction in per-batch time

### 3.4 Phase 4: Pipeline Parallelism

**Scope**: Restructure block sync loop in `main.rs`
**Risk**: LOW — feature-gated, bounded channels prevent memory issues
**Feature gate**: `#[cfg(feature = "pipeline-sync")]`

**Design**:
```rust
// Three stages connected by bounded channels
let (rx_to_val, val_rx) = mpsc::channel::<Vec<QBlock>>(8);
let (val_to_store, store_rx) = mpsc::channel::<Vec<QBlock>>(8);

// Stage 1: Receive (existing block-pack handler)
// Stage 2: Validate (batch sig verify from Phase 1)
// Stage 3: Store (existing SafeBatchedWriter)
```

**Ordering guarantee**: Blocks arrive in height order from turbo sync. Each pipeline stage processes batches in FIFO order. `SafeBatchedWriter` already has height-ordered insertion. Sequence numbers on batches provide an additional ordering check.

---

## 4. Compatibility Analysis

### 4.1 Consensus Compatibility

**No consensus rules are changed.** All four optimizations affect only the speed at which existing rules are evaluated:

| Optimization | Consensus-visible behavior | Changed? |
|-------------|---------------------------|----------|
| Batch sig verify | Accept/reject same signatures | NO |
| Bitfield DAG | Same topological ordering output | NO |
| io_uring WAL | Same data written to disk | NO |
| Pipeline | Same blocks stored in same order | NO |

### 4.2 Wire Protocol Compatibility

No changes to:
- Gossipsub message format
- Block-pack request/response protocol
- Peer height announcements
- Network ID or version handshake

### 4.3 Storage Format Compatibility

No changes to:
- RocksDB column family structure
- Block serialization format (bincode)
- Key encoding scheme
- Compression format (LZ4)

### 4.4 Backward Compatibility

A node with these optimizations enabled can sync from and serve blocks to nodes without them. The optimizations are purely internal to each node.

---

## 5. Testing Requirements

### 5.1 Unit Tests

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| `bitfield_matches_hashset` | proptest: 10K random DAGs | Identical ordering output |
| `batch_verify_matches_individual` | 1000 random signatures | Same accept/reject for each |
| `uring_fsync_durability` | Write + kill -9 + recover | All committed batches present |
| `pipeline_ordering` | 10K blocks through pipeline | Heights strictly monotonic |

### 5.2 Benchmarks

```
benches/dag_simd_benchmarks.rs:
  - anticone_1k, anticone_10k, anticone_100k (hashset vs bitfield)
  - topo_sort_1k, topo_sort_10k (hashset vs bitfield)
  - batch_verify_64, batch_verify_256, batch_verify_1024
  - wal_sync_standard, wal_sync_uring
```

### 5.3 Integration Tests

1. Fresh node sync from Epsilon with all optimizations enabled
2. Compare final block hash at height N with unoptimized node
3. Must match exactly — any divergence is a correctness failure

### 5.4 Soak Testing

| Server | Phase | Duration | Success Criteria |
|--------|-------|----------|------------------|
| Gamma (8GB) | All phases | 48 hours | No OOM, no panics, height progresses |
| Beta (48GB) | All phases | 24 hours | Same height as unoptimized Epsilon |
| Epsilon (64GB) | All phases | 24 hours | BPS improvement measurable |

---

## 6. Performance Projections

### 6.1 Block Sync Speed (genesis sync to current tip)

| Optimization | Current BPS | Expected BPS | Improvement |
|-------------|-------------|-------------|-------------|
| Baseline | ~320 | — | — |
| + Batch sigs | ~320 → ~500 | 500 | 1.6x |
| + Bitfield DAG | ~500 → ~600 | 600 | 1.9x |
| + io_uring WAL | ~600 → ~750 | 750 | 2.3x |
| + Pipeline | ~750 → ~1000 | 1000 | 3.1x |

*Note: BPS is currently bottlenecked by the 200-block server-side cap (being raised to 1000 on Epsilon in v9.8.2). With that change + these optimizations, ~1000+ BPS is realistic.*

### 6.2 At-Tip Block Processing Latency

| Operation | Current | Optimized | Notes |
|-----------|---------|-----------|-------|
| Sig verification (10 tx) | ~500μs | ~100μs | Limited by core count vs batch size |
| DAG anticone (1K vertices) | ~40μs | ~0.5μs | 80x bitfield speedup |
| DAG topo sort (1K vertices) | ~100μs | ~10μs | Cache-friendly traversal |
| WAL sync | ~2ms | ~1.2ms | io_uring overlap |
| **Total block processing** | **~2.6ms** | **~1.3ms** | **2x improvement** |

---

## 7. Open Questions for Review

1. **Phase 2 index recycling**: Should freed vertex indices be recycled immediately, or should the bitfield grow monotonically and compact during `cleanup_cache()`? Monotonic growth is simpler but wastes memory; recycling is more complex but memory-efficient.

2. **Phase 3 RocksDB WAL FD access**: RocksDB doesn't expose the WAL file descriptor directly. We need to either (a) open the WAL file independently for fsync-only access, or (b) use RocksDB's `FlushWAL()` with `sync=false` followed by our own io_uring fsync on the WAL directory. Option (b) is safer but may not guarantee ordering.

3. **Phase 4 pipeline depth**: Should the pipeline be 3 stages (receive/validate/store) or 4 stages (receive/deserialize/validate/store)? Deserialization is ~0.5ms per batch and could overlap with I/O, but a 4th stage adds complexity.

4. **AVX-512 on production servers**: Do Epsilon/Beta/Gamma CPUs support AVX-512? If only AVX2, the bitfield speedup is 40x instead of 80x — still very significant.

---

## 8. References

- DAGKNIGHT paper: "DAGKNIGHT: A Parameterless Generalization of Nakamoto Consensus" (Sompolinsky, 2022)
- Reddio async pipeline: arXiv:2503.04595 (implemented in `crates/q-storage/src/async_pipeline.rs`)
- ed25519-dalek batch verification: https://docs.rs/ed25519-dalek/latest/ed25519_dalek/
- io_uring design: https://kernel.dk/io_uring.pdf
- q-flux io_uring implementation: `crates/q-flux/src/io_uring_loop.rs` (production-proven)

---

## 9. Reviewer Checklist

- [ ] Confirm no consensus rule changes
- [ ] Confirm feature gates are default-OFF (except Phase 1)
- [ ] Confirm fallback paths exist for all hardware-specific code
- [ ] Review bitfield correctness argument
- [ ] Review io_uring durability guarantee
- [ ] Review pipeline ordering guarantee
- [ ] Verify CPU SIMD capabilities on production servers
- [ ] Approve implementation order
- [ ] Approve soak test plan
