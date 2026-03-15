# PR-013: q-flux Phase 2 — io_uring + HTTP/2 Activation

**State**: `open`
**Branch**: `feature/safe-batched-sync-v1.0.2`
**Created**: 2026-03-10
**Closes**: #034, #036

---

## Summary

Activates the already-written but not-wired Phase 2 features in q-flux:

1. **#034 io_uring** — Eliminate syscall overhead, target 20K+ req/s on Epsilon
2. **#036 HTTP/2** — Multiplexed connections, reduce per-miner connections from 240 to 3

Both codebases already exist (`io_uring_loop.rs`: 1171 lines, `h2_proxy.rs`: 997 lines) — this PR wires them into the production path behind feature flags.

### What was wired

**HTTP/2 (complete)**:
- `worker.rs:510-520`: ALPN protocol check routes `h2` negotiated connections to `h2_proxy::handle_h2_connection()`
- Transparent to HTTP/1.1 clients — falls through to standard proxy path

**io_uring accept (complete)**:
- `worker.rs`: When `config.io_uring.enabled = true` and kernel supports io_uring, each worker spawns a blocking thread running `io_uring_accept_thread()`
- Multishot accept: single SQE continuously produces CQEs for each accepted connection (kernel >= 5.19)
- Fallback: single-shot accept with automatic resubmission on older kernels
- Graceful degradation: if io_uring unavailable at runtime, falls back to tokio epoll
- If io_uring accept channel closes, worker automatically falls back to epoll
- Adaptive sleep: 50μs spin when active, 1ms when idle (no busy-wait)

### Files changed

| File | Change |
|------|--------|
| `crates/q-flux/src/worker.rs` | +130 lines: `io_uring_accept_thread()` function, dual-path accept loop |
| `crates/q-flux/src/lib.rs` | Remove `#[allow(dead_code)]` from io_uring_loop module |

## Test Plan

- [ ] A/B benchmark: epoll vs io_uring (wrk2 on Epsilon)
- [ ] A/B benchmark: HTTP/1.1 vs HTTP/2 (connection count, latency)
- [x] Graceful fallback on kernels without io_uring (runtime check + log warning)
- [x] HTTP/1.1 backward compatibility verified (ALPN routing — only h2 goes to h2_proxy)
- [ ] 24-hour soak test on Epsilon
