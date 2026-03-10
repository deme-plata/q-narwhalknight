# Issue #007: OS-Level Auto-Tuning

**State**: `closed`
**Priority**: HIGH
**Labels**: `starship-endgame`, `os`, `performance`
**Assigned**: Beta
**Branch**: `project/starship-endgame-revolution`
**Created**: 2026-03-08
**Updated**: 2026-03-10

---

## Description

Auto-detect OS and apply maximum performance settings on startup. Works on both Linux and Windows.

## Linux Tuning

- [x] `sysctl -w net.core.somaxconn=65535`
- [x] `sysctl -w vm.nr_hugepages=1024`
- [x] `sysctl -w kernel.sched_min_granularity_ns=100000`
- [x] CPU frequency governor -> `performance`
- [x] Disable transparent huge pages compaction
- [x] Set IRQ affinity away from mining cores
- [x] RPS/XPS network queue steering to non-mining cores

## Windows Tuning

- [x] Set process priority to HIGH
- [x] Set thread affinity masks
- [x] Disable power throttling
- [x] Large pages privilege
- [x] Disable Nagle algorithm on sockets

## Progress

- **2026-03-08**: `crates/q-compute/src/os_tuner.rs` created with full Linux + Windows tuning. `apply_all()` calls all tuning functions with graceful fallback if not root.
- **2026-03-10**: Wired into q-api-server startup via compute_api.rs
- **2026-03-10**: IRQ affinity steering (`steer_irq_affinity`) and RPS/XPS queue tuning (`tune_network_queues`) implemented. All Linux tuning complete. All tests passing (23/23). Issue CLOSED.

## Files

- `crates/q-compute/src/os_tuner.rs`
