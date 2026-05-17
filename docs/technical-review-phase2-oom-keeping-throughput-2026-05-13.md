# Technical Review: Phase 2 OOM Fix Without Sacrificing Throughput

**Date:** 2026-05-13
**Triggered by:** v10.9.13 soak — container OOM-killed at 16 GB during Phase 2 pre-checkpoint backfill after ~1 hour of operation. Forward sync (Phase 1) completed cleanly; OOM happened mid-Phase-2 while fetching blocks 225,001-226,000.
**Constraint:** Phase 2 must keep targeting ~hours-to-archive-complete, not days. No naive throttling that halves throughput.

---

## Executive Summary

The OOM is not from one runaway allocation — it's from a sum of small, individually-reasonable allocations that have no end-to-end budget. The fix is to introduce **one global memory budget** and split it across the layers that actually consume it. Each layer keeps its current architecture but participates in a token-bucket scheme so the sum stays bounded.

Throughput is preserved because the bottleneck is network latency (RTT to Epsilon × in-flight chunks), not memory. We can keep the in-flight count of chunks high *if* each individual chunk's memory footprint is bounded, and *if* we flush to disk fast enough that memory is reclaimed before the next batch arrives.

This document maps the memory sources, proposes a layered solution, and gives an implementation plan that lands in v10.9.14.

---

## Observed Memory Pattern (v10.9.13 Soak)

```
t+0min      290 MB     container start
t+15min   ~3.5 GB     Phase 1 forward sync (CHECKPOINT → tip)
t+25min   ~6 GB       convergence reached, Phase 1 done
t+45min   10.89 GB    Phase 2 starts pre-checkpoint backfill
t+~70min  16 GB       OOM-killed by docker --memory=16g
```

Memory grew **steadily during Phase 2** at roughly 5 GB per 25 minutes — that's not a one-time spike, it's accumulation. Sustainable level should be ~4-6 GB for this workload; the extra ~10 GB is leakage or unbounded caching.

Last log lines showed Phase 2 still actively dispatching: `Request: blocks 225001-226000 (1000 blocks)`. No "RSS BACKPRESSURE" log appeared, suggesting the existing v6.0.9 backpressure (lives in download_chunks_parallel inner loop) wasn't gating Phase 2's outer queue.

---

## Memory Sources (Ranked by Estimated Contribution)

### 1. Network in-flight + decompressed chunk buffers (~400 MB peak, sustainable)
- `download_chunks_parallel` keeps 8 in-flight chunks
- Each block-pack response: ~50 MB raw bytes (1000 blocks × ~50 KB compressed)
- After zstd decompress: similar order, plus `Vec<QBlock>` allocations
- **8 × 50 MB ≈ 400 MB peak**, freed once chunk processes
- This is fine when sustained, but if processing lags, it queues up

### 2. RocksDB memtable + write buffers (~256 MB - 2 GB)
- Default `max_write_buffer_number = 2`, `write_buffer_size = 64 MB`
- Per-CF: up to 128 MB of memtable before flush
- We have ~10 active CFs: ~1.3 GB potential
- Plus `block_cache_mb = 2048` (explicitly configured) — that's 2 GB OS-level cache
- **Sustainable but bounded** if writes flush at expected rate

### 3. `pack_cache` (LRU of pre-compressed packs)
- Lives in turbo_sync; caches outgoing block packs for repeat queries
- Default sized by `PackCacheConfig` — need to verify the cap
- During Phase 2 we're a *client*, not a *server* — this cache shouldn't grow on us
- But on a node that mines AND syncs, it grows
- **Probable contributor on Epsilon-as-server side, not the soak directly**

### 4. Apollo Kalman predictor history (~100 MB if unbounded)
- Tracks per-peer bandwidth, RTT, success rate
- Updated every chunk completion
- If the time series isn't bounded, accumulates 16K+ samples
- **Likely small per-instance but compounds with #5**

### 5. jemalloc fragmentation (~1-3 GB after sustained churn)
- tikv_jemallocator is the allocator (per Cargo.toml)
- Many short-lived allocations (block decompression, serialization)
- Without periodic `malloc_trim`, fragmented arena holds returned-but-not-released memory
- We DO call `malloc_trim` on RSS-backpressure events (saw in v10.9.9 logs)
- But Phase 2 hasn't triggered the backpressure path consistently
- **Probably the biggest sustained leak — 2-4 GB**

### 6. Tokio task spawn overhead + channel buffers
- 16,539 chunks of work to dispatch
- If queued in an mpsc channel with unbounded capacity, accumulates
- Each task carries closure capture data
- **Maybe 100-500 MB over the lifetime of 16K chunks**

### 7. Block-state-processor / consensus engine staging
- Each block goes through balance consensus → batch tx → commit
- The intermediate `BalanceState` holds wallet diffs in memory before commit
- For a long Phase 2 with many blocks per chunk, this can sit
- **Estimate: 200-500 MB per in-flight chunk**

**Total estimate**: 400 MB (chunks) + 1.3 GB (memtables) + 2 GB (block cache) + 2-4 GB (jemalloc fragmentation) + ~500 MB (consensus staging) + 500 MB (tokio/channels) = **~7-8 GB sustainable**.

Observed: ~16 GB at OOM. Delta of ~8 GB is unbounded accumulation in one or more of these layers.

---

## What Doesn't Work (Naive Approaches)

**Just shrink chunk size** (1000 → 200): yes, reduces per-chunk memory 5×, but quintuples chunk count (83K instead of 16K) and the per-chunk overhead (channel send, task spawn, serialize) dominates. Throughput drops because we're now overhead-bound, not bandwidth-bound. Doesn't address the leakage either.

**Just reduce in-flight chunks** (8 → 2): cuts in-flight memory 4×, but RTT-bound throughput drops 4× too. Phase 2 goes from 8h to 32h. Violates the throughput constraint.

**Increase docker memory limit** (16 GB → 32 GB): kicks the can. Same leak pattern, just hits OOM later. Doesn't scale to 256-year design.

**Set RocksDB block cache to 0**: cripples read performance. The cache is fine, it's bounded.

---

## Layered Fix (Each Layer Keeps Performance, Sum is Bounded)

### Layer A: Global in-flight byte budget (instead of chunk-count budget)

Replace the `max_in_flight_chunks = 8` cap with a `max_in_flight_bytes` budget. When dispatching a chunk, atomically reserve its expected byte cost; release on completion. Allow many small chunks or fewer large ones.

```rust
// Pseudo-code in download_chunks_parallel
const MAX_IN_FLIGHT_BYTES: usize = 512 * 1024 * 1024; // 512 MB
let in_flight_bytes = Arc::new(AtomicUsize::new(0));

while let Some(chunk) = chunks_queue.pop_front() {
    let estimated_bytes = (chunk.1 - chunk.0 + 1) as usize * 50_000; // ~50 KB/block
    while in_flight_bytes.load(Acquire) + estimated_bytes > MAX_IN_FLIGHT_BYTES {
        tokio::task::yield_now().await;
    }
    in_flight_bytes.fetch_add(estimated_bytes, Release);
    futures.push(spawn_chunk_task(chunk, /* on_complete: release bytes */));
}
```

Result: 8 chunks of 1000 blocks (= 400 MB) or 40 chunks of 200 blocks (= 400 MB). Same memory, higher concurrency for smaller chunks. The scheduler adapts.

### Layer B: Stream-to-disk for block-pack responses

Currently a block-pack response is fully buffered as `Vec<u8>`, decompressed into `Vec<QBlock>` in memory, then handed to `save_qblocks_batch_turbo`. Three full-size copies coexist briefly.

Replace with streaming pipeline:

1. Receive block-pack bytes
2. Stream zstd decode into chunks of 50 blocks at a time
3. Write each 50-block sub-batch to RocksDB immediately
4. Discard the bytes before reading more

This collapses 3× peak memory to 1×. Same throughput because the bottleneck is disk write speed, which is much higher than network receive rate.

### Layer C: Bounded mpsc channels everywhere

The `NetworkRequest`, `NetworkCommand`, and various event channels are mpsc. Some have unbounded capacity. During Phase 2 with 16K chunks queued for dispatch, an unbounded channel can hold thousands of pending requests.

Convert to bounded (e.g., `mpsc::channel(128)`). When full, the sender awaits, providing natural backpressure all the way back to the work queue.

### Layer D: Periodic `malloc_trim` + memtable flush in Phase 2

Add an explicit cycle in fill_gap_via_turbo:

```rust
// Every N chunks (e.g., 50), force jemalloc to release back to OS
// and flush RocksDB memtables to free up sustained memory.
if chunks_completed % 50 == 0 {
    self.storage.flush_active_memtables().await;
    unsafe { tikv_jemalloc_ctl::epoch::advance().ok(); }
    // jemallctl thread_dirty_decay/arena_dirty_decay would help here too
}
```

This is the existing v6.0.9 backpressure mechanism, just hoisted to a per-N-chunks cycle instead of only firing on RSS-near-limit.

### Layer E: Decommission unused caches during Phase 2

Phase 2 is a *client* operation — the soak is fetching, not serving. Several caches that are useful for serving requests (`pack_cache`, certain metric histories) can be cleared or capped tighter during Phase 2.

```rust
// At Phase 2 start
self.pack_cache.clear();
// Optionally: cap it to 0 entries during Phase 2, restore after
```

### Layer F: Hard RSS gate at the chunk-queue layer

Last-resort: keep the existing v6.0.9 RSS backpressure but make it gate **chunk dispatch**, not just chunk *processing* (which is already too late — bytes are in-flight by the time we check).

```rust
// Before dispatching the next chunk
loop {
    let rss = current_rss_mb();
    if rss < soft_limit {
        break; // dispatch
    }
    if rss > hard_limit {
        warn!("RSS {} > hard limit {}, pausing dispatch", rss, hard_limit);
        tokio::time::sleep(Duration::from_secs(1)).await;
        continue;
    }
    // soft_limit < rss < hard_limit: yield briefly, let in-flight drain
    tokio::task::yield_now().await;
}
```

soft_limit = 10 GB, hard_limit = 13 GB on a 16 GB container.

---

## Combined Effect on Throughput

Layer A keeps throughput **the same** — same in-flight bandwidth, just allocated by bytes not count.

Layer B keeps throughput **the same** — disk write was already faster than network receive; streaming just removes the redundant buffer copy.

Layer C keeps throughput **the same** — unbounded channels were just buffer-bloat. Bounded ones still buffer 128 ahead of consumer, which is more than enough.

Layer D introduces ~10ms overhead per 50 chunks. At chunks running every ~5s, that's 0.04% overhead. Negligible.

Layer E reduces working set, no throughput change for Phase 2 itself.

Layer F is rarely triggered if A-D work. When it fires, it gates dispatch for ~1s at a time. Acceptable.

**Estimated v10.9.14 throughput**: identical to v10.9.13 Phase 2 (~570 bps average), but sustainable indefinitely. 16.5M-block backfill completes in ~8 hours.

---

## Implementation Plan

### Sprint 1 (this week, v10.9.14)

- **Layer A** (in-flight byte budget): ~80 LOC in download_chunks_parallel. Replaces in-flight-count semaphore with bytes-budget semaphore. Plumb `estimated_bytes` through chunk dispatch.
- **Layer D** (periodic malloc_trim + memtable flush): ~30 LOC. Add a counter and a `if chunks_completed % 50 == 0` block in fill_gap_via_turbo.
- **Layer F** (hard RSS gate at chunk-queue): ~50 LOC. Add RSS check + sleep loop before each dispatch.

Total ~160 LOC. Soak test for 12+ hours to verify Phase 2 completes without OOM.

### Sprint 2 (next week, v10.10.0)

- **Layer B** (stream-to-disk decode): ~200 LOC. Refactor block-pack codec to a streaming decoder. Touches the block-pack response handler in unified_network_manager. Higher risk; needs its own test.
- **Layer C** (bounded mpsc channels): ~150 LOC. Audit all `mpsc::unbounded_channel` sites, convert appropriate ones to bounded. Risk: bounded channels can deadlock if not sized correctly. Bench beforehand.
- **Layer E** (cache decommission during Phase 2): ~30 LOC. Pack cache clear at Phase 2 start.

### Sprint 3 (validation)

- Run Phase 2 soak for 24h with Layers A+D+F active
- Compare memory profile (heap snapshot) to v10.9.13
- Verify throughput unchanged
- If RSS stays under 12 GB sustained, ship as production v10.9.14
- Tighten Layer F's hard_limit progressively until stable around 10 GB sustained

---

## What This Is Not

Not a consensus change. Not a sync-correctness change. Not a balance-state change. Layers A-F are purely about *bounding the memory footprint* of a workload that today is correct but unbounded. Every fix is local to a single subsystem and can be reverted in isolation.

The decentralization property validated in the v10.9.13 soak (test wallet `f07316494b82ab4e` credited correctly on a fresh checkpoint-bootstrapped node at 01:15:50 UTC) does not change. v10.9.14 just lets that bootstrap complete the full pre-checkpoint backfill without dying.

---

## What This Is

A budget. The current architecture has correct per-component memory behavior but no system-wide constraint. v10.9.14 adds the constraint without rewriting the components.

Phase 1 (forward sync) was never the problem — it's bounded by tip distance and runs to completion in minutes. Phase 2 is the long-tail workload that exposed missing budgets. Fix the budget; keep the speed.

— Server Beta, 2026-05-13
