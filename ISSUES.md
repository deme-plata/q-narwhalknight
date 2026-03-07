# Q-NarwhalKnight — Open Issues / Tasks

Issues are assigned to Claude Code agents. Pick an unassigned issue, create a feature branch, and push your work.

---

## Issue #1: `q-queue` — High-Performance Universal Queue System

**Priority**: High
**Status**: Open
**Assignee**: Unassigned
**Branch**: `feature/q-queue`
**Crate**: `crates/q-queue/`

### Summary

Build `q-queue` — a universal queue fabric that handles both low-latency IPC (inter-process communication) on a single machine and distributed durable messaging across clusters. Sub-microsecond latencies for local queues, millions of messages/sec for networked queues.

### Context

Existing systems have trade-offs:
- **Chronicle Queue**: sub-microsecond latencies but JVM-based, single-machine only
- **Apache Kafka**: durability + distribution but ~275 MB/s per node
- **Redpanda**: ~666 MB/s per node, shared-nothing architecture
- **VAST Message Broker**: 1.1 GB/s per node using disaggregated shared-everything

Our Rust-based design pushes beyond these using **io_uring**, **lock-free algorithms**, **kernel bypass**, and **SIMD-accelerated serialization**.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Queue Client Library                     │
│  (Producer/Consumer API with local fallback)                 │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼───────┐     ┌───────▼───────┐     ┌───────▼───────┐
│   Local IPC   │     │   Persistent  │     │  Distributed  │
│    Queue      │     │    Queue      │     │    Queue      │
│ (shared memory)│     │ (memory-mapped)│     │ (RDMA/TCP)    │
└───────────────┘     └───────────────┘     └───────────────┘
                                                    │
                                      ┌─────────────┴─────────────┐
                                      │     Storage Nodes         │
                                      │  (io_uring, memory-mapped)│
                                      └───────────────────────────┘
```

### Implementation Plan

#### Phase 1: Local IPC Queue (MVP)

**Files to create:**
- `crates/q-queue/Cargo.toml`
- `crates/q-queue/src/lib.rs` — module declarations
- `crates/q-queue/src/slot.rs` — lock-free slot with per-element versioning
- `crates/q-queue/src/ring.rs` — SPSC and SPMC ring buffer implementations
- `crates/q-queue/src/notify.rs` — io_uring `msg_ring` cross-thread notifications
- `crates/q-queue/src/bench.rs` — latency + throughput benchmarks

**Design — Lock-Free Queue with Per-Element Versioning:**

Each slot in the ring buffer has its own atomic version:

```rust
struct Slot<T> {
    version: AtomicU64,  // odd = ready, even = writing
    data: UnsafeCell<MaybeUninit<T>>,
}

pub struct Queue<T> {
    slots: Box<[Slot<T>]>,
    producer_pos: AtomicUsize,
    consumer_pos: AtomicUsize,
}
```

Write algorithm:
```rust
fn enqueue(&self, value: T) -> Result<(), T> {
    loop {
        let pos = self.producer_pos.load(Ordering::Relaxed);
        let slot = &self.slots[pos % self.capacity];
        let version = slot.version.load(Ordering::Acquire);

        if version & 1 == 0 && version / 2 == pos / self.capacity {
            let new_version = version + 1;
            if slot.version.compare_exchange_weak(
                version, new_version, Ordering::AcqRel, Ordering::Acquire
            ).is_ok() {
                unsafe { (*slot.data.get()).write(value); }
                slot.version.store(version + 2, Ordering::Release);
                self.producer_pos.fetch_add(1, Ordering::Release);
                return Ok(());
            }
        }
    }
}
```

Read algorithm:
```rust
fn dequeue(&self) -> Option<T> {
    let pos = self.consumer_pos.load(Ordering::Relaxed);
    let slot = &self.slots[pos % self.capacity];
    let version = slot.version.load(Ordering::Acquire);

    if version & 1 == 1 && version / 2 == pos / self.capacity {
        let value = unsafe { (*slot.data.get()).assume_init_read() };
        slot.version.store(version + 1, Ordering::Release);
        self.consumer_pos.fetch_add(1, Ordering::Release);
        Some(value)
    } else {
        None
    }
}
```

**Target**: <500 ns latency for same-machine transfers.

#### Phase 2: Persistent Queue (Durable Local)

**Files to add:**
- `crates/q-queue/src/persistent.rs` — memory-mapped file segments
- `crates/q-queue/src/segment.rs` — pre-allocated segment management
- `crates/q-queue/src/compaction.rs` — background compaction

**Design:**
- Memory-mapped files pre-allocated in fixed-size segments
- Append-only writes with checksums; reads via random access
- Zero-copy snapshots — consumers map read-only views of segments
- Background compaction to reclaim space from expired messages

**Target**: >10M msg/sec on NVMe.

#### Phase 3: Distributed Queue (Cluster Mode)

**Files to add:**
- `crates/q-queue/src/distributed.rs` — cluster coordination
- `crates/q-queue/src/transport.rs` — RDMA + TCP with io_uring
- `crates/q-queue/src/partition.rs` — consistent hashing
- `crates/q-queue/src/replication.rs` — erasure-coded replication

**Design:**
- RDMA for ultra-low latency when available; fallback to TCP with io_uring
- Disaggregated storage — compute nodes handle protocol, storage nodes hold data
- Consistent hashing for partition assignment
- Optional Kafka protocol compatibility

**Target**: >2 GB/s per node throughput.

#### Phase 4: io_uring Deep Integration

**Files to add:**
- `crates/q-queue/src/uring.rs` — io_uring event loop
- `crates/q-queue/src/multishot.rs` — multishot recv for network

**Design:**
```rust
// Multishot receive for TCP connection
let sqe = io_uring::opcode::RecvMsgMultishot::new(fd, msg_hdr, flags)
    .build()
    .user_data(connection_id);

// Cross-thread signaling via msg_ring (zero syscalls)
io_uring_prep_msg_ring(sqe, target_ring_fd, 0, wakeup_data, 0);
```

#### Phase 5: SIMD Serialization

**Files to add:**
- `crates/q-queue/src/simd_serde.rs` — SIMD-accelerated serialization

```rust
#[cfg(target_arch = "x86_64")]
unsafe fn serialize_int_simd(buf: &mut [u8], values: &[i64]) {
    // Process 4 i64s at once using AVX2
}
```

### Performance Targets

| Scenario | Target | Comparison |
|----------|--------|------------|
| Same-machine IPC (2 threads) | <500 ns latency | Chronicle Queue: ~780 ns |
| Same-machine persistent | >10M msg/sec | Chronicle Queue: ~1.6M/sec/thread |
| Distributed (single node) | >2 GB/s throughput | VAST: 1.1 GB/s, Redpanda: 666 MB/s |
| Distributed (3-node cluster) | >5 GB/s aggregate | Kafka: ~825 MB/s (3×275) |

### Workspace Integration

Add to root `Cargo.toml`:
```toml
members = [
    # ... existing ...
    "crates/q-queue",
]
```

Dependencies (mostly already in workspace):
- `io-uring` (already via q-kernel-io)
- `memmap2`
- `crossbeam` (already in workspace)
- Standard: `tokio`, `anyhow`, `tracing`, `serde`

### How to Start

```bash
git clone git://185.182.185.227/q-narwhalknight
cd q-narwhalknight
git checkout -b feature/q-queue
mkdir -p crates/q-queue/src
# Start with Phase 1: slot.rs + ring.rs
```

---

## Issue #2: `q-flux` Phase 2 — io_uring + SIMD HTTP Parsing

**Priority**: Medium
**Status**: Open (blocked by Phase 1 production testing)
**Assignee**: Unassigned
**Branch**: `feature/q-flux-phase2`
**Crate**: `crates/q-flux/`

### Summary

Replace tokio I/O in q-flux with raw io_uring event loops and add SIMD HTTP header parsing for maximum throughput.

### Files to Create/Modify

- `crates/q-flux/src/io_uring_loop.rs` — real io_uring event loop per worker
- `crates/q-flux/src/simd_parse.rs` — AVX2/SSE4.2 HTTP header scanning
- `crates/q-flux/src/buffer_pool.rs` — pre-allocated buffers registered with io_uring
- `crates/q-flux/src/splice.rs` — zero-copy splice() for WebSocket/large body forwarding

### Reuse from Existing Crates

- `q-kernel-io::ZeroCopyNetworking` — socket configuration, NUMA awareness
- `q-kernel-io::KernelMemoryManager` — NUMA-local buffer allocation
- `q-crypto-simd` patterns — CPU feature detection, runtime dispatch

---

## Issue #3: `q-flux` Phase 3 — HTTP/2, HTTP/3 (QUIC), kTLS

**Priority**: Low
**Status**: Open
**Assignee**: Unassigned
**Branch**: `feature/q-flux-phase3`

### Summary

Add protocol expansion to q-flux:
- HTTP/2 via `h2` crate (multiplexing for browser connections)
- QUIC via `quinn` crate (UDP-based, zero-RTT resume for miners)
- kTLS offload for bulk encryption (kernel handles AES-GCM)
- ALPN-based protocol detection on TLS accept

---

## Issue #4: `q-flux` Phase 4 — libp2p Compatibility Booster

**Priority**: Low
**Status**: Open
**Assignee**: Unassigned
**Branch**: `feature/q-flux-phase4`

### Summary

Proxy-layer awareness of libp2p traffic:
- Detect libp2p WebSocket upgrade requests (multistream-select)
- Per-peer bandwidth tier enforcement at proxy layer
- Circuit breaking for slow/abusive peers
- Gossipsub message dedup at proxy layer (reduce backend load)
- Metrics: per-peer connection count, bandwidth, latency
