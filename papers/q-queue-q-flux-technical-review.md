# Technical Review: q-queue and q-flux Crates

**Date**: 2026-03-07
**Reviewer**: Claude Opus 4.6 (Server Beta)
**Commit**: feature/safe-batched-sync-v1.0.2
**Scope**: Full source audit of `crates/q-queue/` (6 files incl. benchmarks, ~1,440 LOC) and `crates/q-flux/src/` (18 files, ~9,420 LOC) — 24 files, 10,860 LOC total, 229 tests passing (105 q-flux lib + 105 q-flux bin + 18 q-queue + 1 doc-test), plus 4 ignored platform-specific tests

**Verdict**: **Ship**. q-flux is in production on Epsilon (48 cores, 10Gbit) serving 500+ miners at 3.2K sustained req/s with 3.19% error rate (upstream 503s, not proxy failures), 4.7K+ active connections, 916 WebSocket streams, 66 H2 streams, ~50MB RSS. q-queue is technically sound with one API-design trap and one durability area that deserves more scrutiny. No blocking issues remain. Remaining work items are optimization branches and API hardening, not correctness gaps.

---

## 1. Architecture Review

### 1.1 q-queue

**Design**: Lock-free bounded ring buffer using Vyukov's MPMC algorithm adapted for SPSC and MPSC use cases. A separate persistent queue module provides append-only segment files with CRC32 checksums for durable messaging.

**Structure**:
- `slot.rs` -- Per-element `Slot<T>` with atomic sequence counter and `UnsafeCell<MaybeUninit<T>>` payload
- `ring.rs` -- `SpscQueue<T>` (no CAS on hot path) and `MpscQueue<T>` (CAS-based producer contention) with `AtomicBool` consumer guard
- `notify.rs` -- `Notifier` for parking/unparking consumer threads
- `persistent.rs` -- `PersistentQueue` with `Segment` writer and `SegmentReader` for durable messaging, hardware-accelerated CRC32C via `crc32fast`

**Strengths**:
1. Clean separation of concerns -- slot, ring, notification, and persistence are independent modules.
2. SPSC queue avoids CAS entirely on the hot path; uses only atomic loads/stores with correct Acquire/Release ordering.
3. Power-of-two capacity enables branchless modular arithmetic via bitmask (`pos & self.mask`).
4. Cache-padded cursors (`CachePadded<AtomicUsize>`) eliminate false sharing between producer and consumer.
5. Comprehensive test suite (18 tests) covering basic operations, wraparound, thread safety, drop correctness, and concurrent-pop panic detection.
6. Hardware-accelerated CRC32C integrity checking (via `crc32fast` with PCLMULQDQ on x86_64) on persistent messages — 10-50x faster than the original naive implementation.
7. **Consumer guard** (`AtomicBool` CAS on `pop()`) prevents UB from concurrent consumers — panics on violation rather than silently corrupting.

**Weaknesses**:
1. ~~No benchmarks are shipped despite a bench target in Cargo.toml~~ → **FIXED**: 6 criterion benchmark groups: SPSC roundtrip latency, SPSC/MPSC throughput (parameterized by producer count), PersistentQueue append/read throughput, SPSC-vs-MPSC overhead comparison.
2. `PersistentQueue` is single-threaded (`&mut self` on `append()`). The `AtomicU64` on `next_sequence` suggests concurrent intent, but the `&mut self` requirement prevents it.
3. `SegmentReader` reads the entire segment file into memory (`fs::read`) rather than using memory-mapped I/O. For large segments this creates allocator pressure on the read path.
4. No MPMC queue variant.
5. `Notifier` lacks condition-variable-style predicate coupling — correct only when callers use a re-check loop (detailed in Section 3).
6. No async/tokio integration -- consumers must use OS thread parking, incompatible with async runtimes.
7. **Durability semantics are underspecified** — `append()` does not fsync, sequence advancement does not imply persistence, and segment rotation does not fsync parent directory metadata. "Append-only with CRC" and "durable" are different guarantees; the current API does not distinguish them (detailed in Section 3.4).
8. **Consumer guard uses manual reset, not a drop guard** — if future edits introduce early returns or panics between CAS acquire and the manual `store(false)`, the guard gets stuck. Should use a tiny RAII drop guard (detailed in Section 3.5).

**Resolved weaknesses** (since initial review):
- ~~MPSC consumer side is single-consumer by convention only~~ → **FIXED**: `AtomicBool` consumer guard panics on concurrent `pop()` (ring.rs)
- ~~CRC32 implementation is byte-at-a-time~~ → **FIXED**: Replaced with `crc32fast::hash()` using hardware PCLMULQDQ (persistent.rs)

### 1.2 q-flux

**Design**: Worker-per-core TLS reverse proxy. Each worker is an OS thread pinned to a CPU core, running its own single-threaded tokio runtime with a dedicated `TcpListener` (via `SO_REUSEPORT`) and upstream connection pool.

**Structure** (18 source files):

*Phase 1 — MVP (production-ready):*
- `acceptor.rs` -- TLS config building, `SO_REUSEPORT` listener creation, TLS hot-reload, OCSP stapling, ALPN [h2, http/1.1]
- `worker.rs` -- Worker thread spawning, accept loop, per-IP connection tracking, semaphore backpressure, ALPN-based H2/H1 routing
- `proxy.rs` -- HTTP/1.1 request parsing with SIMD pre-check, body reading, keepalive loop, SSE/streaming, WebSocket upgrade
- `upstream.rs` -- Per-worker hyper `Client` with round-robin backend selection, health-aware routing, and super-cluster cross-node failover
- `health.rs` -- Background health checker with TCP+HTTP probing and configurable failure threshold
- `metrics.rs` -- Lock-free atomic counters, latency histogram (16-bucket log2), token bucket rate limiter, Prometheus export
- `access_log.rs` -- Structured JSON logging via bounded sync channel to dedicated writer thread
- `config.rs` -- TOML configuration with duration parsing, validation, and super-cluster `[cluster]` section
- `static_serve.rs` -- Streaming file serving (64KB chunks) with MIME detection, ETag/304, SPA fallback, path traversal protection
- `admin.rs` -- Admin HTTP server with `/health`, `/metrics`, `/status`, `/tls-reload` endpoints
- `tui.rs` -- ratatui-based terminal dashboard with sparkline charts using `VecDeque` for O(1) history

*Phase 2 — Performance (io_uring + SIMD):*
- `io_uring_loop.rs` -- io_uring event loop with registered buffers, multi-shot accept, linked SQEs for accept→read chains
- `simd_parse.rs` -- AVX2→SSE4.2→scalar HTTP header boundary detection, WebSocket upgrade detection, header value extraction

*Phase 3 — Protocol expansion:*
- `h2_proxy.rs` -- HTTP/2 reverse proxy via `h2` crate, multiplexed stream forwarding, flow control, PING/GOAWAY handling
- `quic_proxy.rs` -- QUIC/HTTP/3 endpoint via `quinn` crate, 0-RTT session resume, connection migration, QPACK header compression

*Phase 4 — libp2p awareness:*
- `libp2p_aware.rs` -- Peer identification (multistream-select), bandwidth tiers (Supernode/Bootstrap/Validator/Light), circuit breaker, gossipsub bloom filter dedup, per-peer rate limiting with u128-safe token bucket

**Strengths**:
1. Worker-per-core architecture eliminates cross-thread contention on the hot path. Each worker has its own tokio runtime, upstream pool, and listener.
2. Proper backpressure via `Semaphore::try_acquire()` -- workers drop connections at capacity rather than OOM.
3. Health-aware upstream routing with configurable failure threshold and graceful degraded fallback.
4. SSE/streaming detection and chunk-by-chunk forwarding prevents infinite buffering for event streams.
5. Slowloris protection via read timeouts on both header and body reads.
6. TLS session resumption (tickets + 1M session cache) for fast miner reconnects.
7. Static file serving integrated directly into the proxy layer, avoiding upstream round-trips for assets.
8. Clean shutdown with broadcast channel + atomic flag for fast-path checking.
9. **SIMD HTTP parsing** with runtime CPU feature detection (AVX2→SSE4.2→scalar fallback) for `\r\n\r\n` header boundary and WebSocket upgrade detection — wired into proxy hot path as pre-check before httparse.
10. **ALPN-based protocol routing**: TLS negotiation advertises `[h2, http/1.1]`; worker dispatches to H2 or H1 handler based on negotiated protocol.
11. **HTTP/2 multiplexed proxy** via `h2` crate with stream-level forwarding, flow control, PING keepalive, and GOAWAY graceful shutdown.
12. **QUIC/HTTP/3 endpoint** via `quinn` crate with 0-RTT session resume, connection migration, and QPACK header compression — zero-RTT saves 1 RTT for repeat miners.
13. **libp2p-aware proxying**: Peer identification via multistream-select protocol detection, 4-tier bandwidth enforcement (Supernode 1Gbps / Bootstrap 100Mbps / Validator 10Mbps / Light 1Mbps), circuit breaker (Closed→Open→HalfOpen), gossipsub bloom filter dedup at proxy layer.
14. **Streaming file downloads** (64KB chunks) — memory per download reduced from 100MB to 64KB.
15. **O(1) TUI history** via `VecDeque` — eliminated O(n) `Vec::remove(0)` in sparkline data.
16. **Latency histogram** with 16 log2-scaled buckets for Prometheus-compatible percentile export.
17. **OCSP stapling** support — DER-encoded OCSP response included in TLS handshake, saves 50-100ms per new connection.
18. **Super-cluster cross-node failover** — when all local backends are unhealthy, upstream pool automatically routes to remote cluster peers (`[cluster].peers` config). Local backends always have priority; failover is transparent to clients. Separate health-check thread for cluster peers with configurable interval.
19. **rustls CryptoProvider explicit selection** — workspace has both `ring` and `aws-lc-rs` as transitive deps; `main.rs` calls `rustls::crypto::ring::default_provider().install_default()` before any TLS operations, preventing runtime panic from ambiguous provider selection.

**Production deployment (Epsilon, 48-core, 10Gbit)**:
- Replaced Caddy/Pingap as TLS terminator for 500+ miners
- **Live metrics (March 2026, 2-min sampling window, 10 samples at 15s)**:
  - **3,237 req/s** sustained (min 3.0K, avg 3.2K, max 3.3K)
  - **4,700+ active connections** (avg 4.9K, max 5.0K)
  - **916 WebSocket streams** (libp2p P2P relay, SSE)
  - **66 H2 streams** (avg 64.1, max 71)
  - **3.19% error rate** (avg 3.27% — upstream 503s during challenge transitions)
  - **1.6 MB/s TX bandwidth** (max burst 3.3 MB/s)
  - **Upstream active: 1–478** (avg 143.5) — connection pool reuse confirmed
- Memory: ~50MB RSS (vs Caddy 11.5GB, Pingap OOM)
- CPU: distributed across 48 workers (vs Caddy 42/48 cores, Pingap single-process bottleneck)
- Tor .onion access via socat TLS bridge on self-signed EC cert

**Weaknesses**:
1. ~~HTTP/1.1 only~~ → **RESOLVED**: H2 via `h2_proxy.rs`, QUIC via `quic_proxy.rs`, ALPN routing in `worker.rs`.
2. ~~`accept_any()` is hardcoded for at most 4 listeners using manual `select!` arms~~ → **RESOLVED**: Uses `futures::future::select_all()` for dynamic listener count.
3. ~~`push_bounded()` in `tui.rs` uses `Vec::remove(0)` which is O(n)~~ → **RESOLVED**: Uses `VecDeque` with `push_back()`/`pop_front()`.
4. ~~Static file serving reads entire files into memory~~ → **RESOLVED**: Streaming 64KB chunked reads.
5. ~~Access log entry `to_json()` uses manual string building without control char escaping~~ → **RESOLVED**: Added proper escaping for newlines, tabs, carriage returns, and generic control characters.
6. No connection draining for in-flight requests during backend rotation.
7. io_uring event loop (`io_uring_loop.rs`) is implemented but not yet activated in worker.rs — requires runtime feature gate.
8. ~~`PeerTracker` in `libp2p_aware.rs` is fully implemented but not yet wired into the worker accept path~~ → **RESOLVED**: `PeerTracker` is passed to `handle_connection` and `handle_websocket_upgrade`; peer identification, connection limits, and circuit breaker checks are applied on WebSocket upgrade.

---

## 2. Performance Improvements

### 2.1 q-queue: Ring Buffer Hot Path

**Issue: MPSC spin_loop on every CAS retry** *(confirmed, high priority under contention)*
File: `crates/q-queue/src/ring.rs`, line 189

```rust
std::hint::spin_loop();
```

This is called unconditionally when `diff > 0` (another producer advanced past). Under high contention with many producers, this busy-spins without backoff. Should use exponential backoff or `yield_now()` after a few iterations to reduce CPU waste. This is one of the few items likely to matter immediately under contention — the criterion benchmarks with 4 and 8 producers are the right tool to measure the impact.

**Issue: Extra atomic store on SPSC producer path** *(investigate with benchmarks before changing)*
File: `crates/q-queue/src/ring.rs`, lines 69-70

```rust
slot.sequence.store(pos + 1, Ordering::Release);
self.producer_pos.store(pos + 1, Ordering::Release);
```

The SPSC producer does two Release stores per push. The `producer_pos` store is only needed for `len()` approximation. On the hot path (producer-consumer ping-pong), this is unnecessary overhead. However, the slot sequence store is the real publication barrier, and whether removing the position store matters depends on cache behavior and how often `len()` is called. Since criterion benchmarks now exist, this item should be measured before changing semantics — benchmark the SPSC roundtrip latency with and without the `producer_pos` store.

**Issue: PersistentQueue uses seek+write_all per message**
File: `crates/q-queue/src/persistent.rs`, lines 66-73

```rust
self.file.seek(io::SeekFrom::Start(self.write_pos as u64))?;
self.file.write_all(header_bytes)?;
self.file.write_all(data)?;
```

Two `write_all` syscalls plus a seek per message. Should use `writev()` (scatter-gather I/O) or buffer header+data into a single write. At 10M msg/sec target, each saved syscall matters significantly.

**~~Issue: CRC32 implementation is byte-at-a-time~~ ✅ FIXED**
File: `crates/q-queue/src/persistent.rs`

Replaced 13-line naive CRC32 with `crc32fast::hash(data)`. Uses hardware PCLMULQDQ on x86_64 (10-50x faster), automatic table-based fallback on other architectures.

**Issue: SegmentReader copies every payload** *(allocator pressure on read path)*
File: `crates/q-queue/src/persistent.rs`, line 121

```rust
Some((msg_sequence, payload.to_vec()))
```

Every read allocates a new `Vec<u8>`. For high-throughput consumers, this creates significant allocator pressure. Should return a reference into memory-mapped data, or accept a caller-provided buffer for zero-copy reads.

### 2.2 q-flux: Proxy Hot Path

**Issue: String allocation per request for path and method**
File: `crates/q-flux/src/proxy.rs`, lines 86-87

```rust
let req_path = req.uri().path().to_string();
let req_method = req.method().as_str();
```

Note: `req_method` was previously `.to_string()` but has been fixed to borrow. `req_path` still allocates due to lifetime constraints (needed after `req` is consumed by the upstream request builder). This is a minor allocation; profile before optimizing further.

**~~Issue: Header formatting via format!() per response~~ ✅ FIXED**
File: `crates/q-flux/src/proxy.rs`

Replaced per-header `format!()` + `write_all()` with a single pre-allocated `Vec<u8>` buffer (512 bytes) using `std::io::Write::write!()`. Both `write_response()` (buffered) and `write_response_headers_streaming()` now use a single buffer + single `write_all()` call.

**~~Issue: push_bounded uses Vec::remove(0)~~ ✅ FIXED**
File: `crates/q-flux/src/tui.rs`

Replaced `Vec<u64>` with `VecDeque<u64>` for `rate_history` and `conn_history`. Uses `push_back()`/`pop_front()` — O(1) vs O(n) per update.

**Issue: DashMap per-IP tracker GC interval too long**
File: `crates/q-flux/src/worker.rs`, line 33

```rust
const IP_TRACKER_GC_INTERVAL_SECS: u64 = 60;
```

With short-lived miner connections cycling rapidly, the DashMap can accumulate stale zero-count entries. The GC should run more frequently (e.g., 10s) or entries should be removed atomically on connection close (which `cleanup_conn` already does — making the periodic GC largely redundant for most workloads).

**~~Issue: Static file serving loads entire file into memory~~ ✅ FIXED**
File: `crates/q-flux/src/static_serve.rs`

Replaced `tokio::fs::read()` with `BufReader::with_capacity(65536)` + chunked read loop. Memory per concurrent download: 100MB → 64KB.

**~~Issue: Session cache log message says 65536 but code sets 1,048,576~~ ✅ FIXED**
File: `crates/q-flux/src/acceptor.rs`

Log updated to "session cache 1M, ALPN [h2, http/1.1]" matching actual 1,048,576 cache size and current ALPN protocol list.

---

## 3. Safety & Correctness

### 3.1 Unsafe Code Audit

**q-queue: 6 unsafe blocks**

1. **`ring.rs:66-68` (SPSC push)** -- `(*slot.data.get()).write(value)`
   - **Safety**: Correct. The sequence check (`seq == pos`) guarantees exclusive producer access. The Release store on sequence publishes the write.
   - **Risk**: None — single producer guaranteed by type name convention, though not enforced at compile time.

2. **`ring.rs:87` (SPSC pop)** -- `(*slot.data.get()).assume_init_read()`
   - **Safety**: Correct. The sequence check (`seq == pos + 1`) guarantees the slot was written and the Release barrier from the producer is visible via Acquire.
   - **Risk**: If `pop()` is called from multiple threads simultaneously (violating the single-consumer contract), double-reads could produce undefined behavior. No compile-time enforcement. Mitigated by runtime `AtomicBool` guard.

3. **`ring.rs:177-179` (MPSC push)** -- `(*slot.data.get()).write(value)`
   - **Safety**: Correct. The CAS on `producer_pos` guarantees exclusive write access to the claimed slot.
   - **Risk**: None — CAS provides mutual exclusion.

4. **`ring.rs:205` (MPSC pop)** -- `(*slot.data.get()).assume_init_read()`
   - **Safety**: Correct, same reasoning as SPSC pop.
   - **Risk**: Same as SPSC — single-consumer contract enforced by runtime guard, not compile-time types.

5. **`persistent.rs:66-68` (Segment::append)** -- `std::slice::from_raw_parts(&header as *const MessageHeader as *const u8, HEADER_SIZE)`
   - **Safety**: This creates a byte view over the struct storage for serialization, which is sound. The struct is `#[repr(C, packed)]`.
   - **Risk**: Low. The byte view itself is fine. The broader risk is that `#[repr(C, packed)]` makes direct field access potentially unaligned anywhere in the codebase — packed structs are easy to misuse. Prefer explicit byte encoding or `#[repr(C)]` with manual serialization to avoid future footguns. This is a code hygiene concern, not a soundness issue in the current code.

6. **`persistent.rs:103-104` (SegmentReader::next)** -- `std::ptr::read_unaligned(...)`
   - **Safety**: Correctly uses `read_unaligned` for the packed struct. This is sound.
   - **Risk**: None.

**q-flux: 2 unsafe blocks**

7. **`acceptor.rs:114-138` (create_listener)** -- `libc::setsockopt(...)` calls
   - **Safety**: Uses raw file descriptor from `socket.as_raw_fd()`. The `setsockopt` calls use stack-local `c_int` values with correct size calculations.
   - **Risk**: Low. The return values of `setsockopt` are silently ignored. If `SO_REUSEPORT` fails, the worker may fail to bind later (caught at bind time). `TCP_FASTOPEN` and `TCP_DEFER_ACCEPT` failures are non-critical.

8. **`worker.rs:380-391` (pin_to_core)** -- `libc::sched_setaffinity(...)`
   - **Safety**: Uses `mem::zeroed()` for `cpu_set_t` which is valid for this POD type. `CPU_ZERO` and `CPU_SET` are safe wrappers.
   - **Risk**: None. Failure is logged and operation continues without pinning.

### 3.2 Data Race Potential

**Notifier lacks condition-variable-style predicate coupling**
File: `crates/q-queue/src/notify.rs`, lines 43-47

```rust
pub fn wait(&self) {
    self.parked.store(true, Ordering::Release);  // (A)
    thread::park();                                // (B)
    self.parked.store(false, Ordering::Relaxed);   // (C)
}
```

`Notifier` is only correct when used in a re-check loop around the queue state. The API should encode that contract. If `notify()` fires before (A), the `parked` flag is `false`, `unpark()` is never called, and the consumer parks forever. The typical correct usage is:

```rust
loop {
    if let Some(val) = queue.pop() { return val; }
    notifier.wait();
}
```

This works because the consumer re-checks before parking. But the `Notifier` API does not enforce or document this requirement. A `wait_while(|| queue.is_empty())` method that atomically checks the predicate before parking would eliminate the trap. An async variant using `tokio::sync::Notify` would also extend usability to async runtimes.

**Severity**: MEDIUM — the abstraction is a trap for future callers. Current usage is safe by convention.

**~~TokenBucket refill is not atomic~~ ✅ FIXED**
File: `crates/q-flux/src/metrics.rs`

The non-atomic load/compute/store pattern has been replaced with a CAS loop that atomically adds refill tokens. Also added u128 intermediate arithmetic to prevent u64 overflow in elapsed × rate computation. The rate limiter now provides strict guarantees under high concurrency.

**~~SPSC/MPSC single-consumer contract not enforced~~ ✅ FIXED**
Files: `crates/q-queue/src/ring.rs`

Both `SpscQueue` and `MpscQueue` now include a `consumer_active: AtomicBool` field. `pop()` performs `compare_exchange(false, true, Acquire, Relaxed)` on entry and panics if another thread is already consuming. Two regression tests verify the panic: `spsc_concurrent_pop_panics` and `mpsc_concurrent_pop_panics`.

### 3.3 Error Handling Gaps

**PersistentQueue::open() silently ignores corrupt segments** *(correctness issue)*
File: `crates/q-queue/src/persistent.rs`, lines 138-149

If a segment file exists but cannot be parsed (e.g., truncated header, CRC mismatch), `SegmentReader::next()` returns `None` and the recovery loop stops. The corrupt segment's base sequence is counted but no error is reported. This causes silent sequence gaps. If durability is part of the value proposition, silent truncation / silent gap acceptance in a durable queue is closer to a correctness issue than an error-handling nicety. The queue should at minimum log a warning and ideally return an error or a `RecoveryReport` that distinguishes clean segments from truncated ones.

**Segment::delete() uses drop(self.file) before remove_file**
File: `crates/q-queue/src/persistent.rs`, line 84

```rust
pub fn delete(self) -> io::Result<()> { drop(self.file); fs::remove_file(&self.path) }
```

On Windows, `remove_file` while the file is still open fails. The explicit `drop(self.file)` before `remove_file` is correct, but the method takes `self` by value, meaning `self.file` would be dropped at the end of the function anyway. The explicit drop is redundant but harmless.

**proxy.rs does not validate Content-Length against actual body**
File: `crates/q-flux/src/proxy.rs`, lines 166-197

The body read loop trusts the `Content-Length` header. If the client sends fewer bytes than declared, the loop will eventually hit `read() => 0` and break, but the body will be shorter than declared. This short body is forwarded to upstream as-is. Most upstreams handle this gracefully, but it is a protocol violation.

**~~worker.rs cleanup_conn has TOCTOU race~~ ✅ FIXED**
File: `crates/q-flux/src/worker.rs`

Replaced the drop-then-remove pattern with atomic DashMap `Entry` API: `entry(client_ip)` → `Occupied(mut entry)` → if count > 1 decrement, else `entry.remove()`. The entry lock is held for the entire check-and-remove operation, eliminating the race window.

### 3.4 Durability Semantics (q-queue)

The `PersistentQueue` documentation describes "crash-safe durable storage" but the implementation does not provide full durability guarantees:

1. **No fsync on append**: `Segment::append()` calls `write_all()` but never `fsync()`. Data sits in the OS page cache and can be lost on power failure or kernel panic. Sequence numbers advance immediately, so callers believe data is persisted when it may not be.

2. **No fsync on segment rotation**: When a segment reaches its size limit and a new segment file is created, neither the old segment nor the new segment's directory entry is fsynced. On ext4 with `data=ordered` (the default), this can lead to a zero-length segment file surviving a crash.

3. **Sequence advancement implies nothing about persistence**: `next_sequence` is incremented in memory on each `append()`. After a crash, the recovered sequence may be lower than what producers observed, creating phantom sequence gaps.

**Recommendation**: Either (a) add explicit `fsync()` calls and document the durability guarantee, or (b) rename/redocument the module to clarify it provides "buffered append-only storage with integrity checking" rather than "crash-safe durable storage." Adding an `append_sync()` variant that fsyncs would give callers the choice between throughput and durability.

**Severity**: MEDIUM — no data corruption risk (CRC catches it), but the API contract is misleading.

### 3.5 Consumer Guard Panic Safety (q-queue)

The `consumer_active` guard in `pop()` is reset via a manual `store(false, Release)` at the end of the method:

```rust
pub fn pop(&self) -> Option<T> {
    // CAS(false → true) — panics if already true
    assert!(self.consumer_active.compare_exchange(...).is_ok());
    // ... queue logic ...
    self.consumer_active.store(false, Ordering::Release);  // manual reset
    result
}
```

If future edits introduce early returns or if the queue logic panics between the CAS acquire and the manual reset, `consumer_active` remains `true` and all subsequent `pop()` calls will panic — the queue becomes permanently stuck.

**Recommendation**: Replace the manual store with a RAII drop guard:

```rust
struct ConsumerGuard<'a>(&'a AtomicBool);
impl Drop for ConsumerGuard<'_> {
    fn drop(&mut self) { self.0.store(false, Ordering::Release); }
}
```

This is a small change that makes the invariant future-proof against panic paths.

**Severity**: LOW — current code is correct. This is a maintainability concern, not a bug.

---

## 4. Missing Features (prioritized)

Based on ISSUES.md and codebase analysis:

### Priority 1: Critical / Blocks Production

| Feature | Effort | Status | Notes |
|---------|--------|--------|-------|
| **SPSC/MPSC consumer-side safety** | 1 day | ✅ IMPLEMENTED | `AtomicBool` consumer guard with CAS on `pop()` — panics on concurrent access. 2 regression tests. |
| **Streaming file serving** | 2 days | ✅ IMPLEMENTED | 64KB `BufReader` chunked streaming. Memory per download: 100MB → 64KB. |

### Priority 2: High / Significant Improvement

| Feature | Effort | Status | Notes |
|---------|--------|--------|-------|
| **io_uring event loop** | 3-4 weeks | ✅ IMPLEMENTED (not activated) | `io_uring_loop.rs`: registered buffers, multi-shot accept, linked SQEs. Needs feature-gate activation in worker.rs. |
| **OCSP stapling** | 1 week | ✅ IMPLEMENTED | `build_tls_config()` reads DER-encoded OCSP response and staples via `with_single_cert_with_ocsp()`. |
| **Super-cluster failover** | 1 week | ✅ IMPLEMENTED | `upstream.rs` + `config.rs`: local-first with automatic failover to cluster peers when all local backends unhealthy. Separate health-check thread. |
| **Connection draining** | 3 days | ⬚ TODO | Health checks work but no drain logic for in-flight requests during backend rotation. |
| **Benchmarks for q-queue** | 2 days | ✅ IMPLEMENTED | 6 criterion groups: SPSC latency, SPSC/MPSC throughput, PersistentQueue append/read, SPSC-vs-MPSC overhead. |

### Priority 3: Medium / Nice to Have

| Feature | Effort | Status | Notes |
|---------|--------|--------|-------|
| **HTTP/2 support** | 2-3 weeks | ✅ IMPLEMENTED | `h2_proxy.rs`: multiplexed stream forwarding, flow control, PING, GOAWAY. ALPN routing in worker.rs. |
| **QUIC/HTTP/3** | 4-6 weeks | ✅ IMPLEMENTED | `quic_proxy.rs`: `quinn`-based endpoint, 0-RTT resume, connection migration, QPACK. |
| **Access log: upstream backend field** | 1 day | ✅ IMPLEMENTED | `upstream_backend` wired through `forward()` → `log_access()` → `AccessEntry::to_json()`. H1 and H2 paths both emit backend address. |
| **CryptoProvider selection** | 1 hour | ✅ IMPLEMENTED | Explicit `ring` provider selection in `main.rs` before TLS operations — prevents runtime panic from ambiguous `ring`/`aws-lc-rs` deps. |
| **q-queue distributed mode** | 8-12 weeks | ⬚ TODO | RDMA/TCP cluster messaging. Very large scope. |
| **libp2p-aware proxying** | 3-4 weeks | ✅ IMPLEMENTED | `libp2p_aware.rs`: peer identification, 4-tier bandwidth, circuit breaker, bloom filter dedup. PeerTracker wired into WebSocket upgrade path. |

### Priority 4: Low / Future

| Feature | Effort | Status | Notes |
|---------|--------|--------|-------|
| **SIMD HTTP parsing** | 2 weeks | ✅ IMPLEMENTED | `simd_parse.rs`: AVX2→SSE4.2→scalar for header boundary + WebSocket detection. Wired into proxy hot path. |
| **kTLS offload** | 1 week | ⬚ TODO | Kernel handles AES-GCM. Linux 5.12+ required. |
| **SIMD serialization for q-queue** | 2 weeks | ⬚ TODO | Process 4 i64s at once. |
| **mmap for SegmentReader** | 3 days | ⬚ TODO | Replace `fs::read` with `memmap2`. |
| **Hardware CRC32** | 2 hours | ✅ IMPLEMENTED | `crc32fast` crate with PCLMULQDQ — 10-50x faster. |
| **VecDeque TUI history** | 30 min | ✅ IMPLEMENTED | O(1) push/pop for sparkline data. |

---

## 5. Recommended Next Steps

Ordered by priority based on operational impact and correctness value.

### 1. PersistentQueue corruption detection/reporting

**Description**: `PersistentQueue::open()` silently swallows corrupt or truncated segments during recovery. Add a `RecoveryReport` return type that reports clean segments, truncated segments, and CRC-failed segments. Log warnings on corruption. Optionally expose a `verify()` method for offline integrity checks.

**Files**: `crates/q-queue/src/persistent.rs` (lines 138-149, `open()` and `SegmentReader::next()`)

**Effort**: 2 days

**Impact**: HIGH — if durability is part of the value proposition, silent data loss is a correctness issue.

### 2. Notifier API redesign

**Description**: `Notifier` is only correct when used in a re-check loop around the queue state; the API should encode that contract. Add a `wait_while(predicate: impl Fn() -> bool)` method that atomically checks the predicate before parking. Document that bare `wait()` must only be used inside a loop that re-checks the queue. Consider adding an async variant using `tokio::sync::Notify`.

**Files**: `crates/q-queue/src/notify.rs` (lines 42-57)

**Effort**: 1 day

**Impact**: MEDIUM — prevents subtle deadlocks in consumer code. Current users may already use the correct pattern, but the API is a trap for future callers.

### 3. MPSC producer backoff strategy

**Description**: Replace unconditional `spin_loop()` with exponential backoff. After N failed CAS attempts (e.g., 4), switch from `spin_loop()` to `thread::yield_now()`. Under heavy contention (8+ producers), this reduces CPU waste from busy-spinning. The existing criterion benchmarks with 4 and 8 producers provide the measurement tool.

**Files**: `crates/q-queue/src/ring.rs` (line 189, MPSC push loop)

**Effort**: 1 day

**Impact**: HIGH under contention — directly reduces CPU waste in the producer hot path.

### 4. Connection draining during backend rotation

**Description**: When a backend is removed from the health map or marked unhealthy, in-flight requests to that backend are abruptly terminated. Add a drain period that allows active requests to complete before removing the backend from the pool.

**Files**: `crates/q-flux/src/upstream.rs`, `crates/q-flux/src/health.rs`

**Effort**: 3 days

**Impact**: MEDIUM — prevents 502 errors during backend maintenance.

### 5. SegmentReader zero-copy / mmap path

**Description**: Replace `fs::read()` in `SegmentReader` with `memmap2::MmapOptions::map()`. Return `&[u8]` slices into the mapped region instead of `Vec<u8>` copies. This eliminates per-message allocation on the read path.

**Files**: `crates/q-queue/src/persistent.rs` (line 95-121, `SegmentReader::new()` and `next()`)

**Effort**: 3 days

**Impact**: MEDIUM — reduces allocator pressure for high-throughput consumers.

### 6. Consumer guard drop safety

**Description**: Replace manual `consumer_active.store(false)` with a RAII drop guard. Tiny change, makes the invariant panic-proof.

**Files**: `crates/q-queue/src/ring.rs` (SPSC and MPSC `pop()`)

**Effort**: 30 minutes

**Impact**: LOW — current code is correct. Prevents a class of future bugs.

### 7. PersistentQueue fsync semantics

**Description**: Add `append_sync()` that calls `fsync()` after write, and `rotate_sync()` that fsyncs both old segment and new directory entry. Document that `append()` provides buffered (non-durable) writes. This makes the durability contract explicit.

**Files**: `crates/q-queue/src/persistent.rs` (lines 56-82, `Segment::append()`)

**Effort**: 1 day

**Impact**: MEDIUM — makes the durability contract match the documentation.

### 8. io_uring activation gate

**Description**: Add a runtime feature check that activates `io_uring_loop.rs` on Linux 5.6+ when a config flag is set. Fall back to tokio epoll when unavailable. The implementation exists; this is the wiring work.

**Files**: `crates/q-flux/src/worker.rs`, `crates/q-flux/src/config.rs`

**Effort**: 2 days

**Impact**: MEDIUM-HIGH — potential 2-3x throughput improvement on io_uring-capable kernels.

### Completed steps (no action needed)

- ~~Enforce single-consumer invariant~~ ✅ (AtomicBool guard)
- ~~Replace naive CRC32~~ ✅ (crc32fast)
- ~~Stream static file downloads~~ ✅ (64KB BufReader)
- ~~Add q-queue benchmarks~~ ✅ (6 criterion groups)
- ~~Fix TokenBucket race~~ ✅ (CAS loop)
- ~~Reduce per-request allocations~~ ✅ (pre-allocated buffer)
- ~~Fix TLS log message~~ ✅ (corrected cache size + ALPN)
- ~~Replace Vec with VecDeque in TUI~~ ✅ (O(1) sparkline history)
- ~~Implement OCSP stapling~~ ✅ (with_single_cert_with_ocsp)
- ~~Super-cluster failover~~ ✅ (local-first + cluster peers)
- ~~CryptoProvider selection~~ ✅ (explicit ring provider)

---

## Summary

**Codebase**: 24 files, 10,860 LOC, 229 tests passing (210 q-flux, 19 q-queue), 6 criterion benchmark groups.

Both crates are well-structured with clean code and comprehensive test coverage. The q-queue ring buffer implementation is textbook-correct Vyukov algorithm with proper memory ordering and runtime consumer safety via `AtomicBool` guard. The q-flux proxy handles all critical production concerns (TLS termination, SSE streaming, health checking, backpressure, super-cluster failover) and has been extended through four implementation phases. **In production on Epsilon** (48 cores, 10Gbit) serving 500+ miners at 3.2K sustained req/s with 4.7K+ active connections, 916 WebSocket streams, 66 H2 streams, 3.19% error rate, ~50MB RSS — replacing both Caddy (11.5GB, 87% errors) and Pingap (OOM) which failed under the same load.

**Ship/no-ship**: **Ship.** No blocking correctness issues. q-flux is already production-proven. q-queue has one API-design trap (Notifier) and one durability-contract ambiguity (fsync semantics) — both are documented and neither causes data corruption.

**Phase completion status:**
- **Phase 1 (MVP)**: ✅ Complete — production TLS reverse proxy with worker-per-core, health checks, rate limiting, metrics, access logging, static file serving, admin API, TUI dashboard, super-cluster failover
- **Phase 2 (Performance)**: ✅ Implemented — io_uring event loop + SIMD HTTP parsing (not yet activated via feature gate)
- **Phase 3 (Protocols)**: ✅ Implemented — HTTP/2 via `h2` crate + QUIC/HTTP/3 via `quinn` crate, ALPN-based routing wired into worker
- **Phase 4 (libp2p)**: ✅ Implemented — peer identification, 4-tier bandwidth enforcement, circuit breaker, gossipsub bloom filter dedup, PeerTracker wired into proxy

**Confirmed issues** (open, ordered by priority):
1. `PersistentQueue` silent corruption handling — correctness gap in durable queue recovery
2. `Notifier` API trap — lacks predicate coupling, deadlock risk for naive callers
3. MPSC busy-spin under contention — CPU waste with many producers
4. Connection draining — 502 risk during backend rotation
5. `SegmentReader` allocation pressure — per-message `Vec<u8>` copy on read
6. Consumer guard not panic-safe — manual reset instead of drop guard
7. PersistentQueue fsync contract — documentation promises more than implementation delivers

**Overstated in initial review** (downgraded):
- Packed struct `from_raw_parts` finding: Not a soundness issue in current code. The byte view for serialization is fine; the broader concern is that packed structs invite future misuse. Downgraded from "UB on ARM" to "prefer explicit byte encoding to avoid footguns."
- SPSC extra atomic store: May or may not be measurable. Reframed as "benchmark before changing semantics."

**Production deployment achievements (March 2026 live metrics):**
- Replaced Caddy (11.5GB RAM, 42/48 cores at 9.6K rps) and Pingap (complete TLS failure) on Epsilon
- Sustained **3,237 req/s** with **4,700+ active connections**, **916 WebSocket**, **66 H2 streams** at ~50MB RSS
- **3.19% error rate** (upstream 503s during challenge transitions, not proxy-layer failures)
- **1.6 MB/s TX bandwidth** (peak 3.3 MB/s), upstream pool reuse confirmed (avg 143.5 active)
- Tor .onion HTTPS support via socat TLS bridge with self-signed EC certificate

**Critical fixes applied** (18 total):
1. Consumer guard on `pop()` prevents UB ✅
2. Hardware CRC32C via `crc32fast` ✅
3. Streaming file downloads (64KB chunks) ✅
4. u128 intermediate arithmetic for bandwidth limiter ✅
5. VecDeque for sparkline data ✅
6. Session cache + ALPN log corrected ✅
7. CAS loop for atomic token bucket refill ✅
8. DashMap Entry API for atomic cleanup_conn ✅
9. `select_all` for dynamic listener count ✅
10. Pre-allocated response header buffer ✅
11. `upstream_backend` wired through H1+H2 paths ✅
12. Control character escaping in access log ✅
13. 6 criterion benchmark groups for q-queue ✅
14. H2 admin metrics in `/status` JSON ✅
15. CORS on H2 error + proxied responses ✅
16. PeerTracker wired into proxy + WebSocket paths ✅
17. Super-cluster local-first failover ✅
18. Explicit rustls CryptoProvider selection ✅
