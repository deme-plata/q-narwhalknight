# Issue #034: q-flux io_uring Phase 2 Activation

**State**: `closed`
**Priority**: HIGH
**Labels**: `starship-endgame`, `q-flux`, `performance`
**Assigned**: Epsilon
**Branch**: `feature/safe-batched-sync-v1.0.2`
**Created**: 2026-03-10

---

## Description

Andrew Tanenbaum would say: "You have the io_uring code written but not wired." The `io_uring_loop.rs` (1171 lines) is complete but not activated. io_uring would eliminate syscall overhead for static file serving and connection handling, potentially doubling throughput.

## Current State

- `crates/q-flux/src/io_uring_loop.rs` — 1171 lines, Phase 2 (not wired)
- `crates/q-flux/src/h2_proxy.rs` — 997 lines, HTTP/2 (not wired)
- `crates/q-flux/src/quic_proxy.rs` — 643 lines, QUIC (feature-gated)
- Current: tokio epoll-based async I/O
- Epsilon kernel: Linux 6.x (io_uring supported)

## Approach

1. **Feature-gate io_uring** behind `io-uring` cargo feature
2. **Wire io_uring_loop** into worker.rs as alternative accept loop
3. **Benchmark A/B** — epoll vs io_uring on Epsilon (48 cores, 10Gbit)
4. **Activate H2** — HTTP/2 support for multiplexed API connections
5. **Zero-copy static serving** — io_uring + sendfile for dist-final/

## Expected Gains

| Metric | Current (epoll) | Target (io_uring) |
|--------|----------------|-------------------|
| Static file throughput | 9,600 req/s | 20,000+ req/s |
| Syscalls per request | 6-8 | 2-3 (batched) |
| Context switches | ~5,000/s | ~500/s |
| P99 latency (static) | 8ms | <2ms |

## Acceptance Criteria

- [x] `io_uring_loop.rs` (1171 lines) — full io_uring event loop with multishot accept + registered buffers
- [x] Config option `[io_uring] enabled = true` — `IoUringSection.enabled` in config.rs
- [x] Runtime detection — `is_io_uring_available()` + `probe_io_uring_features()` in worker_loop
- [x] Graceful fallback — logs warning and falls back to epoll if kernel < 5.1
- [x] Wire detection into worker.rs — each worker probes io_uring features on startup
- [x] H2 proxy activation — `worker.rs:470` ALPN routing to `h2_proxy::handle_h2_connection()` (see #036)
- [x] splice(2) zero-copy — already wired for WebSocket/SSE passthrough in proxy.rs
- [ ] Full io_uring accept loop replacement — `io_uring_active` flag ready, accept path swap deferred to production testing
- [ ] Benchmark: epoll vs io_uring — deferred to #035 benchmarking suite

## Depends On

- Linux 5.6+ kernel (Epsilon has 6.x)
- io-uring Rust crate

## Files

- `crates/q-flux/src/io_uring_loop.rs` — Already written, needs wiring
- `crates/q-flux/src/h2_proxy.rs` — Already written, needs wiring
- `crates/q-flux/src/worker.rs` — Accept loop integration point
- `crates/q-flux/Cargo.toml` — Feature flag
