# Technical Review: q-queue and q-flux Crates

**Date**: 2026-03-07
**Reviewer**: Claude Opus 4.6 (Server Beta)
**Commit**: feature/safe-batched-sync-v1.0.2
**Scope**: Full source audit of `crates/q-queue/src/` (5 files, ~480 LOC) and `crates/q-flux/src/` (12 files, ~1960 LOC)

---

## 1. Architecture Review

### 1.1 q-queue

**Design**: Lock-free bounded ring buffer using Vyukov's MPMC algorithm adapted for SPSC and MPSC use cases. A separate persistent queue module provides crash-safe durable storage via append-only segment files with CRC32 checksums.

**Structure**:
- `slot.rs` -- Per-element `Slot<T>` with atomic sequence counter and `UnsafeCell<MaybeUninit<T>>` payload
- `ring.rs` -- `SpscQueue<T>` (no CAS on hot path) and `MpscQueue<T>` (CAS-based producer contention)
- `notify.rs` -- `Notifier` for parking/unparking consumer threads
- `persistent.rs` -- `PersistentQueue` with `Segment` writer and `SegmentReader` for durable messaging

**Strengths**:
1. Clean separation of concerns -- slot, ring, notification, and persistence are independent modules.
2. SPSC queue avoids CAS entirely on the hot path; uses only atomic loads/stores with correct Acquire/Release ordering.
3. Power-of-two capacity enables branchless modular arithmetic via bitmask (`pos & self.mask`).
4. Cache-padded cursors (`CachePadded<AtomicUsize>`) eliminate false sharing between producer and consumer.
5. Comprehensive test suite (11 tests) covering basic operations, wraparound, thread safety, and drop correctness.
6. CRC32 integrity checking on persistent messages catches corruption.

**Weaknesses**:
1. No benchmarks are shipped despite a bench target in Cargo.toml -- the `bench.rs` file referenced in ISSUES.md Phase 1 was never created.
2. `PersistentQueue` is single-threaded (`&mut self` on `append()`). The `AtomicU64` on `next_sequence` suggests concurrent intent, but the `&mut self` requirement prevents it.
3. `SegmentReader` reads the entire segment file into memory (`fs::read`) rather than using memory-mapped I/O. For large segments this is wasteful.
4. No MPMC queue variant. The MPSC consumer side is single-consumer by convention only -- there is no compile-time enforcement.
5. `Notifier::wait()` has a race condition (detailed in Section 3).
6. No async/tokio integration -- consumers must use OS thread parking, incompatible with async runtimes.

### 1.2 q-flux

**Design**: Worker-per-core TLS reverse proxy. Each worker is an OS thread pinned to a CPU core, running its own single-threaded tokio runtime with a dedicated `TcpListener` (via `SO_REUSEPORT`) and upstream connection pool.

**Structure**:
- `acceptor.rs` -- TLS config building, `SO_REUSEPORT` listener creation, TLS hot-reload
- `worker.rs` -- Worker thread spawning, accept loop, per-IP connection tracking, semaphore backpressure
- `proxy.rs` -- HTTP/1.1 request parsing, body reading, keepalive loop, SSE/streaming, WebSocket upgrade
- `upstream.rs` -- Per-worker hyper `Client` with round-robin backend selection and health-aware routing
- `health.rs` -- Background health checker with TCP+HTTP probing and configurable failure threshold
- `metrics.rs` -- Lock-free atomic counters, latency histogram, token bucket rate limiter, Prometheus export
- `access_log.rs` -- Structured JSON logging via bounded sync channel to dedicated writer thread
- `config.rs` -- TOML configuration with duration parsing and validation
- `static_serve.rs` -- Static file serving with MIME detection, ETag/304, SPA fallback, path traversal protection
- `admin.rs` -- Admin HTTP server with `/health`, `/metrics`, `/status` endpoints
- `tui.rs` -- ratatui-based terminal dashboard with sparkline charts

**Strengths**:
1. Worker-per-core architecture eliminates cross-thread contention on the hot path. Each worker has its own tokio runtime, upstream pool, and listener.
2. Proper backpressure via `Semaphore::try_acquire()` -- workers drop connections at capacity rather than OOM.
3. Health-aware upstream routing with configurable failure threshold and graceful degraded fallback.
4. SSE/streaming detection and chunk-by-chunk forwarding prevents infinite buffering for event streams.
5. Slowloris protection via read timeouts on both header and body reads.
6. TLS session resumption (tickets + 1M session cache) for fast miner reconnects.
7. Static file serving integrated directly into the proxy layer, avoiding upstream round-trips for assets.
8. Clean shutdown with broadcast channel + atomic flag for fast-path checking.

**Weaknesses**:
1. HTTP/1.1 only -- no HTTP/2 multiplexing, no QUIC/HTTP/3 (tracked in ISSUES.md #3).
2. `accept_any()` is hardcoded for at most 4 listeners using manual `select!` arms. Does not generalize.
3. `push_bounded()` in `tui.rs` uses `Vec::remove(0)` which is O(n). Should use `VecDeque`.
4. Static file serving reads entire files into memory (`tokio::fs::read`). Binary downloads (50-100MB) will spike memory.
5. Access log entry `to_json()` uses manual string building instead of a proper JSON serializer. This risks malformed JSON if unexpected characters appear in paths or user agents (e.g., control characters, newlines).
6. No connection draining for in-flight requests during backend rotation -- health checks mark backends unhealthy but requests already dispatched may fail.

---

## 2. Performance Improvements

### 2.1 q-queue: Ring Buffer Hot Path

**Issue: Extra atomic store on SPSC producer path**
File: `/opt/orobit/shared/q-narwhalknight/crates/q-queue/src/ring.rs`, lines 69-70

```rust
slot.sequence.store(pos + 1, Ordering::Release);
self.producer_pos.store(pos + 1, Ordering::Release);
```

The SPSC producer does two Release stores per push. The `producer_pos` store is only needed for `len()` approximation. On the hot path (producer-consumer ping-pong), this is unnecessary overhead. Consider making `len()` iterate slots instead, or using a cached position that is updated lazily.

**Issue: MPSC spin_loop on every CAS retry**
File: `/opt/orobit/shared/q-narwhalknight/crates/q-queue/src/ring.rs`, line 189

```rust
std::hint::spin_loop();
```

This is called unconditionally when `diff > 0` (another producer advanced past). Under high contention with many producers, this busy-spins without backoff. Should use exponential backoff or `yield_now()` after a few iterations to reduce CPU waste.

**Issue: PersistentQueue uses seek+write_all per message**
File: `/opt/orobit/shared/q-narwhalknight/crates/q-queue/src/persistent.rs`, lines 66-73

```rust
self.file.seek(io::SeekFrom::Start(self.write_pos as u64))?;
self.file.write_all(header_bytes)?;
self.file.write_all(data)?;
```

Two `write_all` syscalls plus a seek per message. Should use `writev()` (scatter-gather I/O) or buffer header+data into a single write. At 10M msg/sec target, each saved syscall matters significantly.

**Issue: CRC32 implementation is byte-at-a-time**
File: `/opt/orobit/shared/q-narwhalknight/crates/q-queue/src/persistent.rs`, lines 17-30

```rust
fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            // ...
        }
    }
    !crc
}
```

This is a naive bit-by-bit CRC32 implementation. On modern x86_64, the `crc32` crate uses hardware CRC32C instructions (`_mm_crc32_u64`) and is 10-50x faster. For the 10M msg/sec target, this is a bottleneck.

**Issue: SegmentReader copies every payload**
File: `/opt/orobit/shared/q-narwhalknight/crates/q-queue/src/persistent.rs`, line 121

```rust
Some((msg_sequence, payload.to_vec()))
```

Every read allocates a new `Vec<u8>`. For high-throughput consumers, this creates significant allocator pressure. Should return a reference into the memory-mapped data, or use a pre-allocated buffer.

### 2.2 q-flux: Proxy Hot Path

**Issue: String allocation per request for path and method**
File: `/opt/orobit/shared/q-narwhalknight/crates/q-flux/src/proxy.rs`, lines 46-47

```rust
let req_path = req.uri().path().to_string();
let req_method = req.method().as_str().to_string();
```

Two heap allocations per request that could be avoided by borrowing.

**Issue: Header formatting via format!() per response**
File: `/opt/orobit/shared/q-narwhalknight/crates/q-flux/src/proxy.rs`, lines 312-330

```rust
let status_line = format!("HTTP/1.1 {} {}\r\n", ...);
// ...
let header_line = format!("{}: {}\r\n", key, value.to_str().unwrap_or(""));
```

Multiple `format!()` calls per response header. At 10K+ RPS, this is significant allocation overhead. Should pre-allocate a response buffer and write directly via `write!()` into a reusable `Vec<u8>`.

**Issue: push_bounded uses Vec::remove(0)**
File: `/opt/orobit/shared/q-narwhalknight/crates/q-flux/src/tui.rs`, lines 481-486

```rust
fn push_bounded(buf: &mut Vec<u64>, val: u64, max: usize) {
    buf.push(val);
    if buf.len() > max {
        buf.remove(0);
    }
}
```

`Vec::remove(0)` shifts all elements left -- O(n) per call. With `SPARKLINE_LEN = 120`, this is 120 copies every 500ms. Replace with `VecDeque` for O(1) push/pop or use a circular buffer index.

**Issue: DashMap per-IP tracker GC interval too long**
File: `/opt/orobit/shared/q-narwhalknight/crates/q-flux/src/worker.rs`, line 33

```rust
const IP_TRACKER_GC_INTERVAL_SECS: u64 = 60;
```

With short-lived miner connections cycling rapidly, the DashMap can accumulate hundreds of thousands of stale zero-count entries in 60 seconds. The GC should run more frequently (e.g., 10s) or entries should be removed atomically on connection close (which `cleanup_conn` already does -- making the periodic GC largely redundant).

**Issue: Static file serving loads entire file into memory**
File: `/opt/orobit/shared/q-narwhalknight/crates/q-flux/src/static_serve.rs`, line 194

```rust
let body = match tokio::fs::read(&resp.path).await {
```

Binary downloads (q-api-server, q-miner) are 50-100MB. Reading the full file into memory creates a 100MB allocation per concurrent download. Should use `tokio::io::copy()` to stream the file in chunks.

**Issue: Session cache log message says 65536 but code sets 1,048,576**
File: `/opt/orobit/shared/q-narwhalknight/crates/q-flux/src/acceptor.rs`, lines 83 vs 89

```rust
config.session_storage = rustls::server::ServerSessionMemoryCache::new(1_048_576);
// ...
tracing::info!("TLS config: session tickets enabled, session cache 65536, ALPN [http/1.1]");
```

The log message reports 65,536 sessions but the actual cache is 1,048,576. This is misleading for operations debugging.

---

## 3. Safety & Correctness

### 3.1 Unsafe Code Audit

**q-queue: 5 unsafe blocks**

1. **`ring.rs:66-68` (SPSC push)** -- `(*slot.data.get()).write(value)`
   - **Safety**: Correct. The sequence check (`seq == pos`) guarantees exclusive producer access. The Release store on sequence publishes the write.
   - **Risk**: None -- single producer guaranteed by type name convention, though not enforced at compile time.

2. **`ring.rs:87` (SPSC pop)** -- `(*slot.data.get()).assume_init_read()`
   - **Safety**: Correct. The sequence check (`seq == pos + 1`) guarantees the slot was written and the Release barrier from the producer is visible via Acquire.
   - **Risk**: If `pop()` is called from multiple threads simultaneously (violating the single-consumer contract), double-reads could produce undefined behavior. No compile-time enforcement.

3. **`ring.rs:177-179` (MPSC push)** -- `(*slot.data.get()).write(value)`
   - **Safety**: Correct. The CAS on `producer_pos` guarantees exclusive write access to the claimed slot.
   - **Risk**: None -- CAS provides mutual exclusion.

4. **`ring.rs:205` (MPSC pop)** -- `(*slot.data.get()).assume_init_read()`
   - **Safety**: Correct, same reasoning as SPSC pop.
   - **Risk**: Same as SPSC -- single-consumer contract not enforced at type level.

5. **`persistent.rs:66-68` (Segment::append)** -- `std::slice::from_raw_parts(&header as *const MessageHeader as *const u8, HEADER_SIZE)`
   - **Safety**: `MessageHeader` is `#[repr(C, packed)]` so the cast is valid. However, reading packed structs through pointers can cause unaligned reads on some architectures.
   - **Risk**: On x86_64 this works fine. On ARM or other alignment-sensitive architectures, this would cause UB.

6. **`persistent.rs:103-104` (SegmentReader::next)** -- `std::ptr::read_unaligned(...)`
   - **Safety**: Correctly uses `read_unaligned` for the packed struct. This is sound.
   - **Risk**: None.

**q-flux: 1 unsafe block**

7. **`acceptor.rs:114-138` (create_listener)** -- `libc::setsockopt(...)` calls
   - **Safety**: Uses raw file descriptor from `socket.as_raw_fd()`. The `setsockopt` calls use stack-local `c_int` values with correct size calculations.
   - **Risk**: Low. The return values of `setsockopt` are silently ignored. If `SO_REUSEPORT` fails, the worker may fail to bind later (caught at bind time). `TCP_FASTOPEN` and `TCP_DEFER_ACCEPT` failures are non-critical.

8. **`worker.rs:380-391` (pin_to_core)** -- `libc::sched_setaffinity(...)`
   - **Safety**: Uses `mem::zeroed()` for `cpu_set_t` which is valid for this POD type. `CPU_ZERO` and `CPU_SET` are safe wrappers.
   - **Risk**: None. Failure is logged and operation continues without pinning.

### 3.2 Data Race Potential

**CRITICAL: Notifier::wait() has a lost-wakeup race**
File: `/opt/orobit/shared/q-narwhalknight/crates/q-queue/src/notify.rs`, lines 43-47

```rust
pub fn wait(&self) {
    self.parked.store(true, Ordering::Release);  // (A)
    thread::park();                                // (B)
    self.parked.store(false, Ordering::Relaxed);   // (C)
}
```

Between (A) and (B), the producer can call `notify()`, which calls `unpark()`. But `thread::park()` can return spuriously. The real problem is: if `notify()` fires between (A) and (B), `unpark()` consumes the unpark token and `park()` at (B) returns immediately -- this case works. But if `notify()` fires BEFORE (A), the `parked` flag is `false`, so `unpark()` is never called, and the consumer will park forever.

This is a fundamental design issue: the notifier must be used within a check-park loop:
```rust
loop {
    if let Some(val) = queue.pop() { return val; }
    notifier.wait();
}
```
This is safe because the consumer checks the queue before parking. But the `Notifier` API does not enforce this pattern.

**TokenBucket refill is not atomic**
File: `/opt/orobit/shared/q-narwhalknight/crates/q-flux/src/metrics.rs`, lines 105-111

```rust
if self.last_refill_us.compare_exchange_weak(
    last, now_us, Ordering::AcqRel, Ordering::Relaxed
).is_ok() {
    let current = self.tokens.load(Ordering::Relaxed);
    let refilled = (current + new_tokens).min(self.capacity_milli);
    self.tokens.store(refilled, Ordering::Release);  // Not atomic with CAS above
}
```

Between the CAS on `last_refill_us` and the `tokens.store`, another thread can consume tokens. The store overwrites those consumed tokens, effectively "restoring" them. Under high concurrency, this allows slightly exceeding the configured rate. The impact is minor (over-granting by a few tokens per refill window) but violates strict rate limiting guarantees.

**SPSC/MPSC single-consumer contract not enforced**
Files: `/opt/orobit/shared/q-narwhalknight/crates/q-queue/src/ring.rs`

Both `SpscQueue` and `MpscQueue` implement `Sync`, allowing `pop()` to be called from any thread. The "single consumer" invariant is documented but not enforced. Two threads calling `pop()` simultaneously could both read `seq == pos + 1`, both call `assume_init_read()`, and produce UB (double-free or use-after-move).

Fix: Either use a `&mut self` receiver for `pop()` (preventing aliased calls) or add an `AtomicBool` guard.

### 3.3 Error Handling Gaps

**PersistentQueue::open() silently ignores corrupt segments**
File: `/opt/orobit/shared/q-narwhalknight/crates/q-queue/src/persistent.rs`, lines 138-149

If a segment file exists but cannot be parsed (e.g., truncated header), `SegmentReader::next()` returns `None` and the loop stops. The corrupt segment's base sequence is counted but no error is reported. This could cause sequence gaps.

**Segment::delete() uses drop(self.file) before remove_file**
File: `/opt/orobit/shared/q-narwhalknight/crates/q-queue/src/persistent.rs`, line 84

```rust
pub fn delete(self) -> io::Result<()> { drop(self.file); fs::remove_file(&self.path) }
```

On Windows, `remove_file` while the file is still open fails. The explicit `drop(self.file)` before `remove_file` is correct, but the method takes `self` by value, meaning `self.file` would be dropped at the end of the function anyway. The explicit drop is redundant but harmless.

**proxy.rs does not validate Content-Length against actual body**
File: `/opt/orobit/shared/q-narwhalknight/crates/q-flux/src/proxy.rs`, lines 94-134

The body read loop trusts the `Content-Length` header. If the client sends fewer bytes than declared, the loop will eventually hit `read() => 0` and break, but the body will be shorter than declared. This short body is forwarded to upstream as-is. Most upstreams handle this gracefully, but it is a protocol violation.

**worker.rs cleanup_conn has double-entry potential**
File: `/opt/orobit/shared/q-narwhalknight/crates/q-flux/src/worker.rs`, lines 316-323

```rust
if *count == 0 {
    drop(count);
    ip_tracker.remove(&client_ip);
}
```

The `drop(count)` releases the DashMap entry lock. Between `drop(count)` and `ip_tracker.remove(&client_ip)`, another thread can call `entry(client_ip).or_insert(0)` and increment the count to 1. Then `remove()` deletes it, losing the count. This is a low-probability race but could cause the per-IP counter to drift negative (wrapping to u64::MAX) over time.

---

## 4. Missing Features (prioritized)

Based on ISSUES.md and codebase analysis:

### Priority 1: Critical / Blocks Production

| Feature | Effort | ISSUES.md Ref | Notes |
|---------|--------|---------------|-------|
| **SPSC/MPSC consumer-side safety** | 1 day | N/A | Compile-time single-consumer enforcement to prevent UB. Either newtype wrapper that owns `pop()` access, or `AtomicBool` runtime guard. |
| **Streaming file serving** | 2 days | N/A | `static_serve.rs` reads entire files into memory. 100MB binary downloads at 10 concurrent users = 1GB. Use `tokio::io::copy()` with chunked transfer encoding. |

### Priority 2: High / Significant Improvement

| Feature | Effort | ISSUES.md Ref | Notes |
|---------|--------|---------------|-------|
| **io_uring event loop** | 3-4 weeks | Issue #2 | Replace tokio I/O with raw io_uring for zero-syscall networking. Blocked by Phase 1 production testing. |
| **OCSP stapling** | 1 week | Issue #14 | Saves 50-100ms per new TLS connection. Requires `x509-parser` + periodic OCSP fetcher. |
| **Connection draining** | 3 days | Issue #8 (partial) | Health checks work but no drain logic for in-flight requests during backend rotation. |
| **Benchmarks for q-queue** | 2 days | Issue #1 Phase 1 | Cargo.toml declares a bench target but no benchmark file exists. Cannot validate <500ns claim. |

### Priority 3: Medium / Nice to Have

| Feature | Effort | ISSUES.md Ref | Notes |
|---------|--------|---------------|-------|
| **HTTP/2 support** | 2-3 weeks | Issue #3 | Browser multiplexing. Use `h2` crate. Requires ALPN negotiation (already have ALPN infrastructure). |
| **QUIC/HTTP/3** | 4-6 weeks | Issue #3 | Zero-RTT resume for miners. Use `quinn` crate. Most impactful for mobile/high-latency miners. |
| **Access log: upstream backend field** | 1 day | Issue #13 (partial) | `AccessEntry` is defined but not wired into the proxy path. Missing `upstream_backend` field. |
| **q-queue distributed mode** | 8-12 weeks | Issue #1 Phase 3 | RDMA/TCP cluster messaging. Very large scope; should be a separate project milestone. |
| **libp2p-aware proxying** | 3-4 weeks | Issue #4 | Per-peer bandwidth tiers, gossipsub dedup at proxy layer. |

### Priority 4: Low / Future

| Feature | Effort | ISSUES.md Ref | Notes |
|---------|--------|---------------|-------|
| **SIMD HTTP parsing** | 2 weeks | Issue #2 | AVX2/SSE4.2 header scanning. Marginal gain given httparse is already reasonably fast. |
| **kTLS offload** | 1 week | Issue #3 | Kernel handles AES-GCM. Linux 5.12+ required. Only useful at very high bandwidth. |
| **SIMD serialization for q-queue** | 2 weeks | Issue #1 Phase 5 | Process 4 i64s at once. Niche use case unless queue serialization is proven bottleneck. |
| **mmap for SegmentReader** | 3 days | N/A | Replace `fs::read` with `memmap2`. Zero-copy reads, lower memory usage for large segments. |

---

## 5. Recommended Next Steps

Ordered by priority. Each item includes description, affected files, estimated effort, and expected impact.

### 1. Enforce single-consumer invariant on SPSC/MPSC pop()

**Description**: The `pop()` method on both `SpscQueue` and `MpscQueue` can be called from any thread due to `Sync` impl. Two threads calling `pop()` simultaneously causes undefined behavior (double-read of `MaybeUninit`). Add a runtime guard (`AtomicBool`) or refactor to return a `Consumer<T>` handle that is `!Sync`.

**Files**: `/opt/orobit/shared/q-narwhalknight/crates/q-queue/src/ring.rs` (lines 76-91, 196-209)

**Effort**: 1 day

**Impact**: HIGH -- eliminates the only soundness hole in the queue crate. Without this, any multi-threaded misuse causes memory corruption.

### 2. Replace naive CRC32 with hardware-accelerated version

**Description**: The byte-at-a-time CRC32 in `persistent.rs` is 10-50x slower than hardware CRC32C on x86_64. Replace with the `crc32fast` crate which auto-detects SSE4.2 and falls back to a table-based implementation.

**Files**: `/opt/orobit/shared/q-narwhalknight/crates/q-queue/src/persistent.rs` (lines 17-30), `crates/q-queue/Cargo.toml`

**Effort**: 2 hours

**Impact**: HIGH -- directly gates the 10M msg/sec persistent queue target. Every message write and read computes CRC32.

### 3. Stream static file downloads instead of buffering

**Description**: `static_serve.rs` calls `tokio::fs::read()` which loads the entire file into memory. Binary downloads are 50-100MB. Use `tokio::fs::File::open()` + `tokio::io::copy()` with a chunked transfer response to stream the file.

**Files**: `/opt/orobit/shared/q-narwhalknight/crates/q-flux/src/static_serve.rs` (lines 193-226)

**Effort**: 1 day

**Impact**: HIGH -- prevents OOM or excessive memory use during concurrent binary downloads. 10 concurrent 100MB downloads = 1GB saved.

### 4. Fix Notifier lost-wakeup documentation and API

**Description**: Document that `Notifier::wait()` must be used in a check-park loop. Add a `wait_while(predicate)` method that atomically checks the predicate before parking, eliminating the race window. Consider adding an async variant using `tokio::sync::Notify` for tokio-based consumers.

**Files**: `/opt/orobit/shared/q-narwhalknight/crates/q-queue/src/notify.rs` (lines 42-57)

**Effort**: 1 day

**Impact**: MEDIUM -- prevents subtle deadlocks in consumer code. Current users may already use the correct pattern, but the API is a trap.

### 5. Add q-queue benchmarks

**Description**: Create `benches/queue_bench.rs` with criterion benchmarks for: SPSC push/pop round-trip latency, MPSC throughput with 1/4/8 producers, PersistentQueue append throughput, SegmentReader sequential read throughput. This validates the <500ns and 10M msg/sec targets.

**Files**: `/opt/orobit/shared/q-narwhalknight/crates/q-queue/benches/queue_bench.rs` (new file)

**Effort**: 2 days

**Impact**: MEDIUM -- without benchmarks, performance claims are unvalidated. Criterion provides statistical rigor and regression detection.

### 6. Fix TokenBucket non-atomic refill race

**Description**: The refill logic in `TokenBucket::try_acquire()` has a window where consumed tokens can be restored. Replace the load/compute/store pattern with a CAS loop that atomically adds tokens, or accept the slight over-granting and document it.

**Files**: `/opt/orobit/shared/q-narwhalknight/crates/q-flux/src/metrics.rs` (lines 96-127)

**Effort**: 4 hours

**Impact**: LOW-MEDIUM -- only matters under very high concurrency. For rate limiting miners at 100 RPS/IP, the practical impact is negligible, but the code should be correct by construction.

### 7. Reduce per-request allocations in proxy path

**Description**: Replace `req.uri().path().to_string()` and `req.method().as_str().to_string()` with borrows. Replace per-header `format!()` in `write_response()` with a pre-allocated buffer using `write!()`. This eliminates 3-5 heap allocations per request.

**Files**: `/opt/orobit/shared/q-narwhalknight/crates/q-flux/src/proxy.rs` (lines 46-47, 312-330)

**Effort**: 1 day

**Impact**: MEDIUM -- measurable at 10K+ RPS. Each allocation is ~50ns including deallocation, so 5 allocations * 10K RPS = 500K allocations/sec saved.

### 8. Fix TLS session cache log message

**Description**: The log message at `acceptor.rs:89` says "session cache 65536" but the actual cache size is 1,048,576. Fix the log to match reality.

**Files**: `/opt/orobit/shared/q-narwhalknight/crates/q-flux/src/acceptor.rs` (line 89)

**Effort**: 5 minutes

**Impact**: LOW -- but incorrect operational logs cause confusion during debugging.

### 9. Replace Vec with VecDeque in TUI sparkline history

**Description**: `push_bounded()` uses `Vec::remove(0)` which is O(n). Replace `rate_history` and `conn_history` with `VecDeque<u64>` and use `push_back()`/`pop_front()`.

**Files**: `/opt/orobit/shared/q-narwhalknight/crates/q-flux/src/tui.rs` (lines 22-23, 128-129, 481-486)

**Effort**: 30 minutes

**Impact**: LOW -- only affects TUI rendering at 2 Hz. But it is a code quality issue that shows up in code reviews.

### 10. Implement OCSP stapling

**Description**: Add a background task that fetches OCSP responses from the CA every 6 hours, caches the DER-encoded response, and includes it in TLS handshakes via `CertifiedKey::with_ocsp()`. This eliminates the 50-100ms OCSP lookup penalty for new connections.

**Files**: `/opt/orobit/shared/q-narwhalknight/crates/q-flux/src/acceptor.rs` (new `OcspStapler` struct), `crates/q-flux/Cargo.toml` (add `x509-parser`), config.rs (add OCSP config fields)

**Effort**: 1 week

**Impact**: MEDIUM -- saves 50-100ms per new (non-resumed) TLS connection. With 24K+ new connections/sec at peak, this is significant aggregate latency savings.

---

## Summary

Both crates are well-structured with clean code and good test coverage for their current scope. The q-queue ring buffer implementation is textbook-correct Vyukov algorithm with proper memory ordering. The q-flux proxy handles the critical production concerns (TLS termination, SSE streaming, health checking, backpressure) competently.

The most urgent items are:
1. **Soundness**: Enforce single-consumer invariant on `pop()` to prevent UB (q-queue)
2. **Performance**: Replace naive CRC32 and add benchmarks (q-queue)
3. **Memory safety**: Stream static file downloads instead of full buffering (q-flux)

The largest missing feature areas are io_uring integration (Issue #2) and HTTP/2+QUIC support (Issue #3), both of which are multi-week efforts tracked in ISSUES.md.
