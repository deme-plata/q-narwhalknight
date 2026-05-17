# Technical Review: Epsilon Memory Stall — Jemalloc Fragmentation (2026-04-14)

**Date:** 2026-04-14  
**Severity:** HIGH (node unresponsive, mining stalled 140+ minutes)  
**Node:** Epsilon (89.149.241.126) — 10Gbit supernode  
**Pattern:** Recurring every 2-6 hours after restart  

---

## What Happened

Epsilon froze ~2.5 hours after restart. RSS hit 52.2GB against 50GB cgroup high limit. 5.8GB in swap, 0B available. Mining stalled 140+ minutes.

## Memory Breakdown

| Component | Size | Expected? |
|-----------|------|-----------|
| RSS total | 52,250 MB | NO |
| je_alloc (jemalloc) | **45,015 MB** | **NO — THE PROBLEM** |
| je_resident | 49,551 MB | NO |
| RocksDB cache | 4,094 MB | YES (config=4096) |
| RocksDB memtable | 4 MB | YES |
| App caches | 0 MB | YES |

**41GB gap:** je_alloc (45GB) minus RocksDB cache (4GB) = 41GB of jemalloc arena fragmentation. Memory allocated by temporary buffers (block serialization, P2P messages, sync responses), freed by Rust, but NOT returned to OS by jemalloc.

## Root Cause

Jemalloc's default dirty page decay (10s) is too slow for Epsilon's throughput:
- 400+ mining challenges/sec
- Turbo sync: 200 blocks/request (50-150MB responses)
- P2P gossipsub: 91 msgs/sec on miner-stats (2MB/sec)
- Each request allocates → processes → frees → jemalloc holds pages
- Re-allocation outpaces decay → RSS grows monotonically → OOM

## Fix: Periodic Arena Purge

```rust
// In health watchdog (every 300s):
#[cfg(not(target_os = "windows"))]
unsafe {
    tikv_jemalloc_ctl::epoch::advance().ok();
    for arena in 0..tikv_jemalloc_ctl::arenas::narenas::read().unwrap_or(64) {
        let key = format!("arena.{}.purge", arena);
        tikv_jemalloc_ctl::raw::write(key.as_bytes(), 0u64).ok();
    }
    tikv_jemalloc_ctl::raw::write(b"arenas.dirty_decay_ms\0", 1000i64).ok();
    tikv_jemalloc_ctl::raw::write(b"arenas.muzzy_decay_ms\0", 1000i64).ok();
}
```

Forces jemalloc to `madvise(MADV_DONTNEED)` on unused pages. Used by TiKV, ScyllaDB. Zero risk — only returns already-freed pages.

**Expected:** RSS stable at 8-15GB instead of growing to 52GB.

## Previous Fixes (Applied, Not Sufficient Alone)

- ROCKSDB_BLOCK_CACHE_MB=4096 — RocksDB is only 4GB of 52GB
- MemoryHigh=50G — throttling causes stall, not fix
- Block-pack semaphore=4 — limits peak but not fragmentation
- Health watchdog cache cleanup — app caches already 0

## Long-term: Reduce Allocation Churn

1. Buffer pools for block serialization (reuse Vec instead of new)
2. Streaming responses for block packs (don't serialize 200 blocks into one Vec)
3. Zero-copy P2P where possible
4. Arena-per-task jemalloc configuration
