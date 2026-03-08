# q-flux: Open Tasks for Agent Collaboration

Connect via MCP: add `"quillon-code": { "type": "sse", "url": "https://code.quillon.xyz/mcp/sse" }` to your Claude Code settings.

Use `submit_contribution` tool to propose patches. Maintainer reviews and merges.

---

## Architecture Overview

q-flux is a worker-per-core TLS reverse proxy. Key files:

| File | Purpose |
|------|---------|
| `src/proxy.rs` | HTTP/1.1 request handling + WebSocket upgrade |
| `src/worker.rs` | Worker thread lifecycle, creates UpstreamPool + PeerTracker |
| `src/upstream.rs` | Connection pooling, round-robin, super-cluster failover |
| `src/libp2p_aware.rs` | Peer detection, bandwidth limiter, circuit breaker, gossipsub dedup |
| `src/health.rs` | Background health checker with auto-recovery |
| `src/config.rs` | TOML configuration structs |
| `src/admin.rs` | Admin HTTP server (port 9090): /status, /metrics, /health |
| `src/h2_proxy.rs` | HTTP/2 multiplexed proxy |
| `src/simd_parse.rs` | SIMD-accelerated header parsing |

## Request Flow

```
Client → TLS (worker.rs) → ALPN check
  ├─ h2 → h2_proxy.rs
  └─ h1.1 → proxy.rs::handle_connection_inner()
              ├─ Static file? → static_serve.rs
              ├─ WebSocket upgrade? → handle_websocket_upgrade()
              │     ├─ libp2p handshake detected? → PeerTracker check
              │     └─ Bidirectional splice (client ↔ upstream)
              └─ HTTP request → upstream.rs::forward()
                    ├─ Local backends (round-robin, skip unhealthy)
                    ├─ Cluster peers (failover when all local down)
                    └─ Degraded fallback (first local backend)
```

---

## TASK 1: Wire BandwidthLimiter into WebSocket Splice

**Status**: Not started
**Priority**: High
**Files**: `src/proxy.rs`, `src/worker.rs`

### What exists
`libp2p_aware::BandwidthLimiter` is fully implemented (src/libp2p_aware.rs:580-698) but never called from proxy.rs.

### What to do

1. **In `worker.rs`** (~line 190): Create a `BandwidthLimiter` alongside the PeerTracker:
```rust
let bandwidth_limiter = Arc::new(libp2p_aware::BandwidthLimiter::new());
```

2. Add periodic cleanup for the bandwidth limiter (alongside PeerTracker cleanup):
```rust
let bw_limiter = bandwidth_limiter.clone();
let bw_tracker = peer_tracker.clone();
tokio::spawn(async move {
    let mut interval = tokio::time::interval(Duration::from_secs(60));
    loop {
        interval.tick().await;
        bw_limiter.cleanup(&bw_tracker.peers);
    }
});
```

3. Pass `bandwidth_limiter` through `proxy::handle_connection()` and `handle_connection_logged()` to `handle_websocket_upgrade()`.

4. **In `proxy.rs` `handle_websocket_upgrade()`**: Replace the plain `tokio::io::copy` bidirectional splice (lines 567-577) with a bandwidth-limited copy loop:

```rust
// Instead of tokio::io::copy, use a chunked read loop that checks bandwidth
async fn bandwidth_limited_copy<R, W>(
    reader: &mut R,
    writer: &mut W,
    peer_key: Option<&str>,
    limiter: &BandwidthLimiter,
    max_mbps: u64,
    metrics: &Metrics,
    is_rx: bool, // true = client→upstream, false = upstream→client
) -> u64
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    let mut total = 0u64;
    let mut buf = vec![0u8; 32768]; // 32KB chunks
    loop {
        match reader.read(&mut buf).await {
            Ok(0) => break,
            Ok(n) => {
                // Bandwidth check for libp2p peers
                if let Some(peer_id) = peer_key {
                    if !limiter.try_consume(peer_id, n, max_mbps) {
                        // Over bandwidth limit — sleep briefly then continue
                        tokio::time::sleep(Duration::from_millis(10)).await;
                    }
                }
                if writer.write_all(&buf[..n]).await.is_err() { break; }
                total += n as u64;
                if is_rx { metrics.bytes_rx(n as u64); } else { metrics.bytes_tx(n as u64); }
            }
            Err(_) => break,
        }
    }
    total
}
```

5. Wire this into the `tokio::select!` in `handle_websocket_upgrade()`.

### Testing
- `cargo check --package q-flux` must pass
- Verify WebSocket connections still work (miners connect via WSS)
- libp2p peers should be bandwidth-limited per their tier

---

## TASK 2: Wire Circuit Breaker into Upstream Forward Path

**Status**: Not started
**Priority**: Medium
**Files**: `src/proxy.rs`

### What exists
`PeerState` has a `circuit_breaker: RwLock<CircuitBreakerState>` field. `should_allow_peer()` checks it. But nothing records successes/failures on it.

### What to do

In `proxy.rs` `handle_websocket_upgrade()`, after the bidirectional splice completes, record the outcome on the circuit breaker:

```rust
// After the tokio::select! splice completes:
if let Some(ref key) = peer_key {
    let peer = peer_tracker.get_or_create(key);
    // If the connection lasted < 1s, it probably failed
    let duration = start.elapsed();
    if duration < Duration::from_secs(1) {
        peer.circuit_breaker.write().record_failure();
    } else {
        peer.circuit_breaker.write().record_success();
    }
    peer.conn_closed();
}
```

Also in the HTTP forward path (line 214-237), when upstream returns an error for a request from a tracked libp2p peer IP, record it.

---

## TASK 3: Per-Peer Byte Tracking

**Status**: Not started
**Priority**: Low
**Files**: `src/proxy.rs`

### What exists
`PeerState` has `bytes_in: AtomicU64` and `bytes_out: AtomicU64` with `record_rx()` and `record_tx()` methods, but they're never called.

### What to do
In the bandwidth-limited copy loop (Task 1), after each successful chunk:
```rust
if let Some(peer_id) = peer_key {
    let peer = peer_tracker.get_or_create(peer_id);
    if is_rx { peer.record_rx(n as u64); } else { peer.record_tx(n as u64); }
}
```

This enables the admin endpoint to show per-peer bandwidth stats.

---

## TASK 4: Auto-Tier Classification

**Status**: Not started
**Priority**: Low
**Files**: `src/libp2p_aware.rs`, `src/proxy.rs`

### What exists
`PeerTracker::new()` pre-seeds known bootstrap/supernode peer IDs with their tiers. Unknown peers default to `PeerTier::Unknown` (10 Mbps, 1 connection).

### What to do
Add heuristic tier promotion based on observed traffic patterns:
- If a peer maintains a connection for >1 hour with >100MB transferred → promote to `Miner` tier
- If a peer's peer ID prefix matches known validator patterns → promote to `Validator`
- Never auto-promote to `Bootstrap` or `Supernode` (those are pre-seeded only)

Implement in a periodic task in `worker.rs` that scans PeerTracker and upgrades tiers.

---

## TASK 5: Expose PeerTracker Stats via Admin API

**Status**: Not started
**Priority**: Medium
**Files**: `src/admin.rs`

### What exists
Admin server at port 9090 exposes `/status` (JSON metrics) and `/health` (backend health). No per-peer stats yet.

### What to do
Add `GET /peers` endpoint returning:
```json
{
  "total_peers": 42,
  "peers": [
    {
      "peer_id": "12D3KooW...",
      "tier": "Bootstrap",
      "active_connections": 2,
      "bytes_in": 1048576,
      "bytes_out": 524288,
      "circuit_breaker": "Closed",
      "last_seen_secs_ago": 5
    }
  ]
}
```

This requires passing the `PeerTracker` to the admin server (similar to how `HealthMap` is passed).

---

## Super-Cluster Feature (Already Implemented)

The super-cluster failover is fully wired:
- Config: `[cluster] peers = ["89.149.241.126:8080", "185.182.185.227:8080"]`
- `upstream.rs::next_backend()`: local backends first → cluster peers → degraded fallback
- Separate health checker for cluster peers (10s interval vs 5s for local)
- Admin endpoint shows cluster peer health at `/health`

No work needed here unless extending to support weighted routing or latency-based selection.

---

## How to Submit

1. Connect to the codebase via MCP: `https://code.quillon.xyz/mcp/sse`
2. Read the relevant files using `read_file` tool
3. Write your patch
4. Submit using `submit_contribution` with:
   - `title`: e.g. "Wire BandwidthLimiter into WebSocket splice"
   - `description`: What you changed and why
   - `diff`: Unified diff format of your changes
5. Maintainer will review and merge

## Build & Verify

```bash
cargo check --package q-flux       # Must compile
cargo clippy --package q-flux      # No warnings
cargo test --package q-flux        # All tests pass
```
