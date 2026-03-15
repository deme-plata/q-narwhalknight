# Issue #036: q-flux HTTP/2 Multiplexed API Proxy

**State**: `closed`
**Priority**: MEDIUM
**Labels**: `starship-endgame`, `q-flux`, `performance`
**Assigned**: Epsilon
**Branch**: `feature/safe-batched-sync-v1.0.2`
**Created**: 2026-03-10

---

## Description

Vint Cerf would say: "HTTP/2 multiplexing eliminates head-of-line blocking." Miners open many concurrent API calls (submit work, check status, SSE stream) — with HTTP/1.1 each needs a separate TCP connection. H2 multiplexes all streams over one connection.

## Current State

- `crates/q-flux/src/h2_proxy.rs` — 997 lines, already written, NOT wired
- Current: HTTP/1.1 with keepalive (connection pooling)
- Miners open 3-5 connections per worker thread
- At 48 workers: 144-240 connections per miner

## Expected Gains

| Metric | HTTP/1.1 | HTTP/2 |
|--------|----------|--------|
| Connections per miner | 144-240 | 1-3 |
| TCP handshake overhead | Per request batch | Once |
| Header compression | None | HPACK |
| Stream multiplexing | No | Yes |
| Server push (SSE alternative) | No | Yes |

## Acceptance Criteria

- [x] Wire `h2_proxy.rs` into proxy.rs request path — `worker.rs:470` routes ALPN `h2` to `h2_proxy::handle_h2_connection()`
- [x] ALPN negotiation: h2 preferred, http/1.1 fallback — TLS acceptor checks `alpn_protocol() == b"h2"`
- [ ] Benchmark: HTTP/1.1 vs HTTP/2 (connections, latency, throughput) — deferred to #035
- [ ] Server push for mining template updates (replaces SSE polling) — future enhancement
- [x] Feature-gated: `--features h2` (default on) — always compiled in
- [x] Backward compatible — HTTP/1.1 miners still work — ALPN fallthrough to HTTP/1.1 proxy

## Depends On

- #034 (io_uring activation — synergy with H2)

## Files

- `crates/q-flux/src/h2_proxy.rs` — Already written
- `crates/q-flux/src/proxy.rs` — Integration point
- `crates/q-flux/src/acceptor.rs` — ALPN negotiation
