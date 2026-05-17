# Q-NarwhalKnight — Open Issues / Tasks

Issues are assigned to Claude Code agents. Pick an unassigned issue, create a feature branch, and push your work.

---

## Issue #1: `q-queue` — High-Performance Universal Queue System

**Priority**: High
**Status**: Phase 1+2 DONE (SPSC + MPSC ring buffers, persistent WAL with CRC32, 19 tests passing, 1,065 LOC)
**Assignee**: Server Beta
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
**Status**: SKIPPED (user deferred — Phase 1 pipeline sufficient for current scale)
**Assignee**: Server Beta
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

**Priority**: Medium
**Status**: PARTIAL — HTTP/2 DONE (hyper-based, 878 LOC, 24 tests); QUIC scaffolded behind `quic` feature flag; kTLS deferred
**Assignee**: Server Beta
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
**Status**: DONE — PeerTracker wired into WebSocket upgrade path with libp2p handshake detection, peer scoring, circuit breakers, bandwidth limits (1,186 LOC, 52 tests)
**Assignee**: Server Beta
**Branch**: `feature/q-flux-phase4`

### Summary

Proxy-layer awareness of libp2p traffic:
- Detect libp2p WebSocket upgrade requests (multistream-select)
- Per-peer bandwidth tier enforcement at proxy layer
- Circuit breaking for slow/abusive peers
- Gossipsub message dedup at proxy layer (reduce backend load)
- Metrics: per-peer connection count, bandwidth, latency

---

## Issue #5: `q-flux` — SSE/Streaming Response Support

**Priority**: Critical (blocks mining deployment)
**Status**: DONE
**Assignee**: Server Beta
**Branch**: `feature/q-flux-sse`
**Crate**: `crates/q-flux/`

### Summary

The current q-flux `proxy.rs` collects the entire upstream response body into memory before sending to the client. This breaks **Server-Sent Events (SSE)** which are infinite streams — the proxy will hang forever waiting for the body to finish.

Mining clients use SSE on `/api/v1/sse` for real-time block updates, balance changes, and mining stats. This MUST work for production.

### What to Fix

In `crates/q-flux/src/proxy.rs`, the `write_response()` function calls `body.collect().await` which waits for the complete body. For SSE responses (`content-type: text/event-stream`), we need to stream chunks as they arrive.

### Implementation

1. Detect SSE responses by checking `content-type: text/event-stream` header
2. For SSE: write headers immediately, then loop reading chunks from upstream and writing to client
3. For normal responses: keep the current collect-then-write approach (simpler, allows Content-Length)
4. Also handle `Transfer-Encoding: chunked` responses by streaming chunks

```rust
async fn write_response<S>(stream: &mut S, resp: hyper::Response<Incoming>, metrics: &Metrics) -> Result<()> {
    let (parts, body) = resp.into_parts();
    let is_sse = parts.headers.get("content-type")
        .and_then(|v| v.to_str().ok())
        .map(|v| v.contains("text/event-stream"))
        .unwrap_or(false);

    // Write status + headers
    write_response_headers(stream, &parts).await?;

    if is_sse {
        // Stream mode: forward chunks as they arrive
        use http_body_util::BodyExt;
        let mut body = body;
        while let Some(frame) = body.frame().await {
            match frame {
                Ok(frame) => {
                    if let Some(data) = frame.data_ref() {
                        stream.write_all(data).await?;
                        stream.flush().await?; // Critical for SSE — flush each event!
                        metrics.bytes_tx(data.len() as u64);
                    }
                }
                Err(e) => break,
            }
        }
    } else {
        // Buffered mode: collect full body
        let body_bytes = body.collect().await?.to_bytes();
        // ... write content-length + body ...
    }
}
```

### Test

```bash
curl -N https://localhost:443/api/v1/sse
# Should see streaming events, not hang
```

---

## Issue #6: `q-flux` — Graceful Shutdown + Signal Handling

**Priority**: High
**Status**: DONE
**Assignee**: Server Beta
**Branch**: `feature/q-flux-shutdown`
**Crate**: `crates/q-flux/`

### Summary

q-flux has no graceful shutdown. When killed (SIGTERM from systemd), all in-flight connections are dropped immediately. Need:

1. Catch SIGTERM/SIGINT on main thread
2. Signal all workers to stop accepting new connections
3. Wait up to N seconds for in-flight requests to complete
4. Force-close remaining connections after timeout
5. Log final metrics snapshot before exit

### Implementation

In `main.rs`:
```rust
// Create a shutdown signal shared across workers
let (shutdown_tx, _) = tokio::sync::broadcast::channel::<()>(1);

// ... spawn workers with shutdown_rx ...

// Wait for signal on main thread
let rt = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
rt.block_on(async {
    tokio::signal::ctrl_c().await.ok();
    tracing::info!("Shutdown signal received, draining connections...");
    shutdown_tx.send(()).ok();
    tokio::time::sleep(Duration::from_secs(30)).await; // drain timeout
});
```

In `worker.rs`, each worker checks `shutdown_rx` in the accept loop and stops accepting when signaled, but finishes in-flight requests.

---

## Issue #7: `q-flux` — Admin API + Prometheus Metrics Endpoint

**Priority**: Medium
**Status**: DONE
**Assignee**: Server Beta
**Branch**: `feature/q-flux-admin`
**Crate**: `crates/q-flux/`

### Summary

Add an internal admin listener (e.g. `127.0.0.1:9090`) that exposes:

1. **`GET /metrics`** — Prometheus-formatted metrics for Grafana dashboards
2. **`GET /health`** — Health check (200 OK if upstream reachable, 503 if not)
3. **`GET /status`** — JSON status: active connections, RPS, upstream health, worker info, uptime
4. **`POST /reload`** — Hot-reload TLS certificates (for Let's Encrypt renewal)

### Metrics to Export (Prometheus format)

```
# HELP q_flux_connections_active Current active connections
# TYPE q_flux_connections_active gauge
q_flux_connections_active 1234

# HELP q_flux_requests_total Total requests processed
# TYPE q_flux_requests_total counter
q_flux_requests_total{status="2xx"} 5678901
q_flux_requests_total{status="4xx"} 123
q_flux_requests_total{status="5xx"} 7

# HELP q_flux_tls_handshakes_total Total TLS handshakes
# TYPE q_flux_tls_handshakes_total counter
q_flux_tls_handshakes_total{result="ok"} 456789
q_flux_tls_handshakes_total{result="fail"} 23

# HELP q_flux_upstream_latency_seconds Upstream response latency
# TYPE q_flux_upstream_latency_seconds histogram
q_flux_upstream_latency_seconds_bucket{le="0.001"} 400000
q_flux_upstream_latency_seconds_bucket{le="0.005"} 450000
q_flux_upstream_latency_seconds_bucket{le="0.01"} 460000
```

### Implementation

- Spawn a separate tokio task on the metrics reporter thread
- Use `hyper` to serve the admin endpoints on a different port
- Read from the existing `Metrics` struct (already has atomic counters)
- Add histogram tracking for latency (new field in Metrics)

---

## Issue #8: `q-flux` — Connection Draining + Upstream Health Checks

**Priority**: High
**Status**: DONE
**Assignee**: Server Beta
**Branch**: `feature/q-flux-health`
**Crate**: `crates/q-flux/`

### Summary

Two related features needed for production reliability:

#### A. Active Upstream Health Checks

Currently q-flux only detects upstream failures on actual requests. Add background health checks:

```rust
// Every 5 seconds, probe each backend
async fn health_check_loop(backends: &[String], status: Arc<DashMap<String, bool>>) {
    loop {
        for backend in backends {
            let healthy = tokio::net::TcpStream::connect(backend)
                .await.is_ok();
            status.insert(backend.clone(), healthy);
        }
        tokio::time::sleep(Duration::from_secs(5)).await;
    }
}
```

- Skip unhealthy backends in round-robin
- Log when backends go up/down
- Configurable health check interval and path (e.g. `GET /api/v1/status`)

#### B. Connection Draining for Backend Rotation

When deploying new backend versions (via `ha-deploy.sh`), the backend restarts. q-flux needs to:

1. Detect backend going down (health check fails)
2. Stop sending new requests to that backend
3. Wait for in-flight requests to complete (or timeout)
4. Resume sending when health check passes again

### Config Addition

```toml
[upstream]
health_check_interval = "5s"
health_check_path = "/api/v1/status"
health_check_timeout = "3s"
drain_timeout = "30s"
```

---

## Issue #9: `q-flux` — TLS Session Resumption + OCSP Stapling

**Priority**: Medium
**Status**: DONE (Session resumption + tickets implemented; OCSP stapling deferred)
**Branch**: `feature/q-flux-tls-perf`
**Crate**: `crates/q-flux/`

### Summary

Optimize TLS performance for miners who reconnect frequently:

#### A. TLS Session Resumption (Tickets)

Miners disconnect and reconnect every few minutes. Without session resumption, each reconnect does a full TLS handshake (~2ms). With tickets, the resumed handshake is ~0.5ms.

```rust
// In acceptor.rs, enable session tickets:
let mut config = ServerConfig::builder()
    .with_no_client_auth()
    .with_single_cert(certs, key)?;

// Ticketer rotates keys automatically
config.ticketer = rustls::crypto::ring::Ticketer::new()?;
// Or for cross-worker session sharing:
config.session_storage = rustls::server::ServerSessionMemoryCache::new(65536);
```

#### B. OCSP Stapling

Staple the OCSP response to avoid clients doing separate OCSP lookups:

```rust
// Load OCSP response (refresh periodically via background task)
let ocsp = std::fs::read("/etc/letsencrypt/live/quillon.xyz/ocsp.der").ok();
config.cert_resolver = Arc::new(OcspCertResolver { cert_chain, key, ocsp });
```

#### C. TLS 1.3 Only (Optional)

Consider restricting to TLS 1.3 only (faster handshake, simpler, more secure). All modern mining clients support it.

```rust
config.versions = &[&rustls::version::TLS13];
```

### Performance Impact

| Metric | Without | With Session Resumption |
|--------|---------|------------------------|
| Full TLS handshake | ~2ms | ~2ms (first time) |
| Resumed handshake | ~2ms | ~0.5ms (75% faster) |
| Handshakes/sec (48 cores) | ~24K | ~96K |

---

## Issue #10: `q-flux` — TLS Certificate Hot-Reload

**Priority**: Medium
**Status**: DONE (SharedTlsConfig with RwLock swap, wired into workers + POST /tls-reload admin endpoint)
**Assignee**: Server Beta
**Branch**: `feature/q-flux-tls-reload`
**Crate**: `crates/q-flux/`

### Summary

Add `POST /reload` to the admin API that hot-reloads TLS certificates without restarting q-flux. Essential for Let's Encrypt auto-renewal (certificates rotate every 90 days).

### Implementation

1. Wrap `Arc<ServerConfig>` in an `ArcSwap` so workers can atomically see the new config
2. On `POST /reload`, re-read cert+key files, build new `ServerConfig`, swap it in
3. Workers pick up the new config on the next TLS handshake (zero connection disruption)
4. Return JSON response with old/new certificate serial numbers and expiry dates

### Dependencies

- Add `arc-swap` crate to workspace

```rust
// In acceptor.rs
use arc_swap::ArcSwap;

pub type SharedTlsConfig = Arc<ArcSwap<ServerConfig>>;

pub fn reload_tls(shared: &SharedTlsConfig, tls: &TlsConfig) -> Result<()> {
    let new_config = build_tls_config(tls)?;
    shared.store(new_config);
    Ok(())
}
```

---

## Issue #11: `q-flux` — Request Latency Histogram

**Priority**: Medium
**Status**: DONE (LatencyHistogram + Prometheus export, wired into proxy.rs at all response paths)
**Assignee**: Server Beta
**Branch**: `feature/q-flux-latency`
**Crate**: `crates/q-flux/`

### Summary

Add latency tracking to the metrics system. Currently we track counts but not timing. Need:

1. Per-request latency measurement (time from header parse to last response byte)
2. Histogram with configurable buckets (1ms, 5ms, 10ms, 25ms, 50ms, 100ms, 250ms, 500ms, 1s, 5s)
3. Expose as Prometheus histogram on `GET /metrics`

### Implementation

Add an atomic histogram to `MetricsInner`:

```rust
struct LatencyHistogram {
    buckets: [(f64, AtomicU64); 10], // (upper_bound_ms, count)
    sum: AtomicU64,                   // total microseconds
    count: AtomicU64,                 // total observations
}
```

In `proxy.rs`, wrap `handle_connection` request processing with:
```rust
let start = Instant::now();
// ... process request ...
metrics.record_latency(start.elapsed());
```

---

## Issue #12: `q-flux` — Token Bucket Rate Limiter

**Priority**: Medium
**Status**: DONE (TokenBucket + per-IP RateLimiter, wired into worker accept loop + config)
**Assignee**: Server Beta
**Branch**: `feature/q-flux-ratelimit`
**Crate**: `crates/q-flux/`

### Summary

The current per-IP rate limiting only counts concurrent connections. Add a token bucket rate limiter for requests-per-second control:

1. Per-IP token bucket: configurable rate (e.g. 100 req/s) and burst (e.g. 200)
2. Global rate limit: total RPS across all clients
3. Return `429 Too Many Requests` with `Retry-After` header
4. DashMap-based storage with periodic cleanup of stale entries

### Config

```toml
[limits]
rps_per_ip = 100        # sustained rate per IP
burst_per_ip = 200      # burst capacity per IP
global_rps = 50000      # global limit across all IPs
```

### Implementation

```rust
struct TokenBucket {
    tokens: AtomicU64,      // available tokens × 1000 (fixed-point)
    last_refill: AtomicU64, // timestamp in microseconds
    rate: u64,              // tokens per second × 1000
    capacity: u64,          // max tokens × 1000
}
```

---

## Issue #13: `q-flux` — Structured Access Logging

**Priority**: Low
**Status**: DONE (AccessLogger + JSON serialization, wired into worker→proxy pipeline)
**Assignee**: Server Beta
**Branch**: `feature/q-flux-access-log`
**Crate**: `crates/q-flux/`

### Summary

Add structured access logging in JSON format for production observability. Each request produces one log line with:

- Timestamp (ISO 8601)
- Client IP + port
- Method, path, HTTP version
- Response status code
- Upstream backend used
- Request/response size (bytes)
- Total latency (ms)
- TLS version + cipher
- User-Agent header

### Implementation

- Use a dedicated log writer thread with a bounded channel (no I/O in hot path)
- Write to configurable file (from `config.logging.access_log`)
- Rotate logs via external tool (logrotate) — write to stdout/file, not manage rotation

```json
{"ts":"2026-03-07T12:00:00Z","ip":"1.2.3.4","method":"POST","path":"/api/v1/mining/submit","status":200,"upstream":"127.0.0.1:8080","rx":256,"tx":128,"latency_ms":4.2,"tls":"TLS1.3","ua":"q-miner/9.2.4"}
```

---

## Issue #14: `q-flux` — OCSP Stapling

**Priority**: Low
**Status**: DONE — `with_single_cert_with_ocsp()` in acceptor.rs, `ocsp_staple` config field, DER file validation at startup
**Assignee**: Server Beta
**Branch**: `feature/q-flux-ocsp`
**Crate**: `crates/q-flux/`

### Summary

Implement OCSP stapling to avoid clients making separate OCSP lookups during TLS handshake. This saves ~50-100ms per new connection.

### Implementation

1. Custom `rustls::server::ResolvesServerCert` that includes the OCSP response
2. Background task that fetches OCSP response from the CA every 6 hours
3. Parse OCSP responder URL from the certificate's Authority Information Access extension
4. Cache the DER-encoded OCSP response in memory
5. Include in the TLS handshake via `CertifiedKey::new(certs, key).with_ocsp(ocsp_der)`

### Dependencies

- `x509-parser` for extracting OCSP responder URL from cert
- `reqwest` (or raw HTTP) for fetching OCSP response from CA

---

## Issue #15: `q-flux` — HTTP/2 Upstream Multiplexing

**Priority**: Medium
**Status**: DONE — hyper-based HTTP/2 server via `service_fn`, ALPN dispatch, H2Metrics, static file routing, Prometheus export (878 LOC, 24 tests)
**Assignee**: Server Beta
**Branch**: `feature/q-flux-h2`
**Crate**: `crates/q-flux/`

### Summary

Wire the existing `h2_proxy.rs` scaffold into production. HTTP/2 multiplexing allows many requests over a single TCP connection, eliminating head-of-line blocking for browser-based wallet users.

### Implementation

1. ALPN negotiation: detect `h2` in TLS handshake, route to `h2_proxy::handle_h2_connection()`
2. Stream-level proxying: map each H2 stream → HTTP/1.1 upstream request
3. Server push: preload `/assets/index.js` on wallet page load
4. Flow control: per-stream + connection-level window management
5. GOAWAY handling: graceful shutdown of H2 connections during rolling deploy

### Acceptance Criteria

- `curl --http2 https://quillon.xyz/api/v1/status` returns valid response
- Wallet loads in Chrome DevTools showing H2 protocol
- Benchmark: ≥2× RPS improvement for browser clients vs HTTP/1.1

---

## Issue #16: `q-flux` — QUIC Transport (HTTP/3)

**Priority**: Low
**Status**: Open — scaffold in `quic_proxy.rs`
**Assignee**: Server Beta
**Branch**: `feature/q-flux-quic`
**Crate**: `crates/q-flux/`

### Summary

Add QUIC/HTTP/3 support via the `quinn` crate. QUIC eliminates TCP head-of-line blocking and enables 0-RTT connection resumption — ideal for miners that reconnect frequently.

### Implementation

1. Bind UDP socket on port 443 alongside TCP (QUIC runs over UDP)
2. QUIC listener in each worker using `quinn::Endpoint`
3. 0-RTT session resumption for miners (skip full handshake on reconnect)
4. Alt-Svc header: advertise QUIC availability to HTTP/1.1 clients
5. Stream multiplexing: map QUIC streams to upstream HTTP/1.1 requests

### Dependencies

- `quinn` crate (QUIC implementation)
- Feature-gated: `#[cfg(feature = "quic")]` — optional, not required for base build

---

## Issue #17: `q-flux` — Prometheus Grafana Dashboard Template

**Priority**: Low
**Status**: DONE — `crates/q-flux/grafana/q-flux-dashboard.json` (691 lines, 9 panel sections)
**Assignee**: Server Beta
**Branch**: `feature/q-flux-grafana`

### Summary

Create a Grafana dashboard JSON template that visualizes all q-flux Prometheus metrics from `GET /metrics`. Ship as `crates/q-flux/grafana/q-flux-dashboard.json`.

### Panels

1. **Connections**: active connections (gauge), connections/sec (rate of total)
2. **Requests**: RPS by status code (2xx/4xx/5xx stacked), total requests counter
3. **Latency**: p50/p95/p99 from histogram, heatmap of request durations
4. **TLS**: handshake success/fail rate, session resumption ratio
5. **Upstream**: active upstream connections, connect failures, timeouts
6. **Rate Limiting**: rate-limited requests/sec, active IPs in rate limiter
7. **Bandwidth**: bytes received/sent per second
8. **WebSocket**: active WebSocket connections, upgrade rate

---

## Issue #18: `q-flux` — Connection Draining During TLS Reload

**Priority**: Medium
**Status**: DONE — SharedTlsConfig with drain_notify (tokio::sync::Notify), reload_count (AtomicU64), `drain_timeout_secs` config field, worker drain watcher task
**Assignee**: Server Beta
**Branch**: `feature/q-flux-tls-drain`
**Crate**: `crates/q-flux/`

### Summary

Current `POST /tls-reload` swaps certs instantly. Existing connections continue using the old cert until they close. Add an option to gracefully drain connections using the old cert within a configurable timeout.

### Implementation

1. `POST /tls-reload?drain=30s` — reload certs and drain old connections within 30s
2. Track which `Arc<ServerConfig>` generation each connection is using
3. After reload, stop accepting new connections on old config
4. Wait up to drain timeout for existing connections to complete
5. Force-close remaining old-cert connections after timeout
6. Return JSON: `{"reloaded": true, "drained": 142, "forced_closed": 3}`

---

## Issue #19: `q-queue` — Phase 3 Distributed Queue

**Priority**: Low
**Status**: Open
**Assignee**: Unassigned
**Branch**: `feature/q-queue-distributed`
**Crate**: `crates/q-queue/`

### Summary

Add distributed queue mode to q-queue for cross-node message passing. Uses TCP (with optional RDMA) for transport, consistent hashing for partitioning, and erasure coding for replication.

### Files to Create

- `crates/q-queue/src/distributed.rs` — cluster coordination + partition assignment
- `crates/q-queue/src/transport.rs` — TCP with io_uring, optional RDMA
- `crates/q-queue/src/partition.rs` — consistent hashing ring
- `crates/q-queue/src/replication.rs` — erasure-coded replication (Reed-Solomon)

### Target

- >2 GB/s per node throughput
- <1ms cross-node latency on 10GbE

---

## Issue #20: `q-flux` — Per-Peer Bandwidth Enforcement

**Priority**: Medium
**Status**: DONE — PeerTracker with DashMap per-peer state, TokenBucket per-tier bandwidth limits, BandwidthLimiter enforcing Bootstrap/Validator/Miner/Light tiers, wired into WebSocket upgrade path
**Assignee**: Server Beta
**Branch**: `feature/q-flux-peer-bw`
**Crate**: `crates/q-flux/`

### Summary

Wire `libp2p_aware.rs` PeerTracker into the proxy layer. Enforce per-peer bandwidth tiers at the reverse proxy level so abusive peers can't starve legitimate miners.

### Implementation

1. Detect libp2p multistream-select handshake in first bytes of WebSocket upgrade
2. Extract peer ID from Noise handshake or X-Peer-ID header
3. Classify peer into tier (Bootstrap/Validator/Miner/Light/Unknown)
4. Enforce bandwidth limits per tier using token bucket at proxy layer
5. Circuit breaker: auto-block peers that exceed limits 3× in 5 minutes
6. Metrics: per-peer connection count, bandwidth usage, circuit breaker state

### Tiers

| Tier | Bandwidth | Max Connections |
|------|-----------|-----------------|
| Bootstrap | Unlimited | 1000 |
| Validator | 100 MB/s | 100 |
| Miner | 10 MB/s | 10 |
| Light | 1 MB/s | 5 |
| Unknown | 512 KB/s | 2 |
