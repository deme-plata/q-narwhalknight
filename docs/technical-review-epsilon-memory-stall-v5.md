# Technical Review v5: Epsilon Memory-Induced P2P Death — Recurring Stall Pattern

**Date:** 2026-04-12  
**Severity:** Critical (mainnet, $1B+ market cap)  
**Node:** Epsilon supernode (89.149.241.126) — 10Gbit, 48 cores, 64GB RAM  
**Pattern:** Recurring — observed 4 times in 24 hours (every ~6-8 hours)  
**Status:** Phase 1 fix applied (ROCKSDB_BLOCK_CACHE_MB=16384→4096); monitoring  

---

## 1. Incident Pattern

```
Time=0h:   Restart → 62MB RSS, peers connected, synced to tip
Time=2-4h: RSS climbs to 5-15GB (RocksDB block cache filling + page cache)
Time=4-6h: RSS reaches 25-30GB → hits systemd MemoryHigh=30G
Time=6-8h: P2P connections drop to 0/0 → height freezes → "Synced 100%" at stale height
           Block producer: SHOULD_PRODUCE → YES, but produce_block() times out (>120s)
           Manual restart required — cycle repeats identically
```

---

## 2. Root Cause

### 2.1 Primary: RocksDB Block Cache 16GB on 30GB cgroup

**File:** `crates/q-storage/src/kv.rs` lines 128-162, 226-230

Auto-tune computes 3841MB for xxlarge tier, but env var `ROCKSDB_BLOCK_CACHE_MB=16384` overrides to **16GB** — 53% of the 30G budget consumed by cache alone.

### 2.2 Secondary: No Direct I/O on xxlarge

**File:** `crates/q-storage/src/kv.rs` lines 323-340

Direct I/O disabled for xlarge/xxlarge. Kernel page cache (counted against cgroup) adds 5-10GB.

### 2.3 Memory Budget (Before vs After Fix)

| Component | Before | After Fix |
|-----------|--------|-----------|
| RocksDB block cache | **16 GB** | **4 GB** |
| Kernel page cache | 5-10 GB | 5-10 GB |
| RocksDB memtables | ~0.5 GB | ~0.5 GB |
| jemalloc + tokio + P2P | 2-4 GB | 2-4 GB |
| Block-pack buffers | 0.2-0.6 GB | 0.2-0.6 GB |
| **Total** | **25-34 GB** | **13-22 GB** |

### 2.4 Why P2P Dies at MemoryHigh

cgroup v2 `memory.high` triggers **synchronous direct reclaim** on every allocation:
1. Tokio worker threads stall for ms-to-seconds
2. libp2p swarm event loop misses heartbeats
3. Yamux keepalive pings go unanswered → remote peers close connections
4. `peer_count` → 0, reconnect attempts also stall under pressure
5. Block production enters but can't gossip → MASTER TIMEOUT after 120s

### 2.5 Why "Synced 100%" Is Misleading

With 0 peers, no height announcements arrive. `current_height == stale_network_height` → reports 100% at a frozen height.

---

## 3. Phase 1 Fix (Applied 2026-04-12 — Config Only, Zero Code Risk)

```bash
# /etc/systemd/system/q-api-server.service on Epsilon
# Changed: ROCKSDB_BLOCK_CACHE_MB=16384 → 4096
Environment="ROCKSDB_BLOCK_CACHE_MB=4096"
```

**Why safe:** Block cache is a read-only LRU. Reducing it → cache misses read from disk. No data loss, no consensus change. Same value used on Beta. Instantly reversible.

---

## 4. Phase 2 Fixes (Code, MEDIUM Risk — Next Release)

### 4.1 Cgroup-Aware Cache Sizing
**File:** `crates/q-storage/src/kv.rs`  
Read `/sys/fs/cgroup/memory.high`, use `min(cgroup_limit, physical_ram) - 10GB` as effective RAM for auto-tune.

### 4.2 Memory Pressure Watchdog
**File:** `crates/q-api-server/src/main.rs`  
Every 10s: `pressure = rss / cgroup_limit`. At >80%: jemalloc purge + reduce block-pack semaphore. At >90%: halve RocksDB cache + refuse block-pack. At <70%: restore.

### 4.3 Graceful P2P Reconnection
**File:** `crates/q-network/src/unified_network_manager.rs`  
`force_reconnect` flag set when pressure drops. Re-dial bootstrap with 30s timeout. CRITICAL log after 5 min with 0 peers.

---

## 5. Safety Matrix

| Fix | Consensus? | Validation? | Balances? | Writes? | Reversible? |
|-----|---|---|---|---|---|
| Block cache env var | No | No | No | No | Yes |
| Cgroup-aware sizing | No | No | No | No | Yes |
| Memory watchdog | No | No | No | No | Yes |
| P2P reconnection | No | No | No | No | Yes |

---

## 6. Timeline

```
✅ Phase 1 (2026-04-12): ROCKSDB_BLOCK_CACHE_MB=4096 — monitoring 24h
⏳ Phase 2 Week 1: Cgroup-aware cache sizing (4.1)
⏳ Phase 2 Week 2: Memory pressure watchdog (4.2)
⏳ Phase 2 Week 3: Graceful P2P reconnection (4.3)
```

## 7. Key Files

- `crates/q-storage/src/kv.rs:128-340` — block cache auto-tune, direct I/O, env var
- `crates/q-network/src/unified_network_manager.rs:2412-2570` — P2P health, reconnect, block-pack semaphore
- `crates/q-api-server/src/main.rs:20870-21210` — memory cleanup, MASTER TIMEOUT
- `crates/q-storage/src/memory_limiter.rs:38-128` — MemoryPressure, RSS tracking
