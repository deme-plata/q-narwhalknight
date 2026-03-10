# Issue #012: Async GPU Monitoring — Unblock Tokio Runtime

**State**: `closed`
**Priority**: HIGH
**Labels**: `starship-endgame`, `performance`, `bug`
**Assigned**: Beta
**Branch**: `feature/safe-batched-sync-v1.0.2`
**Created**: 2026-03-10
**Closed**: 2026-03-10

---

## Description

`ResourceMonitor::sample_gpu()` calls `nvidia-smi` (or `rocm-smi`) synchronously via `std::process::Command`, blocking the tokio worker thread for 50-500ms on every 100ms sample tick. This stalls mining submissions, SSE streaming, and P2P message handling.

## Root Cause

`crates/q-compute/src/resource_monitor.rs` lines ~83-140: `try_nvidia_smi()` and `try_rocm_smi()` use blocking `Command::output()` inside an async task.

## Fix

1. Switch to `tokio::process::Command` for async execution
2. Cache GPU results for 2s (GPU utilization doesn't change at 100ms granularity)
3. Detect GPU backend once at startup (not on every sample)
4. If GPU query takes >200ms, fall back to cached value and log warning

## Acceptance Criteria

- [x] GPU monitoring uses `tokio::process::Command`
- [x] Results cached for 2s with async refresh
- [x] Backend detection runs once at startup
- [x] Fallback to cached/zero on timeout
- [x] No blocking of tokio runtime during GPU queries

## Implementation

### Resource Monitor (`resource_monitor.rs`)
- `GpuBackend` enum: `NvidiaSmi`, `RocmSmi`, `Sysinfo`, `None`, `Unknown`
- `GpuBackend::detect()` — async, probes once at startup
- `GpuCache` struct with 2-second TTL
- Separate tokio task for GPU polling (every 2s, isolated from main 100ms loop)
- `GPU_QUERY_TIMEOUT = 200ms` — returns cached value if exceeded
- Main sample loop reads GPU data from cache (non-blocking `RwLock::read()`)
- `try_nvidia_smi_async()` / `try_rocm_smi_async()` — fully async
- `try_sysinfo_gpu()` — sync but lightweight (thermal sensors only)

### Tests
- `test_gpu_monitoring_does_not_block_runtime` — spawns monitor + concurrent task, verifies no starvation
- `test_gpu_backend_detect_completes` — ensures detection returns within 5s timeout
- `test_gpu_stats_default`, `test_gpu_cache_starts_expired`

## Files

- `crates/q-compute/src/resource_monitor.rs`
