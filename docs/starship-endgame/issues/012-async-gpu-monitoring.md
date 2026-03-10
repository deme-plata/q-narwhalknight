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
- `GpuBackend::detect()` -- async, probes once at startup
- `GpuCache` struct with 2-second TTL
- Separate tokio task for GPU polling (every 2s, isolated from main 100ms loop)
- `GPU_QUERY_TIMEOUT = 2s` -- returns cached value if exceeded (bumped from 200ms to 2s)
- Main sample loop reads GPU data from cache (non-blocking `RwLock::read()`)
- `try_nvidia_smi_async()` / `try_rocm_smi_async()` -- fully async via `tokio::process::Command`
- `try_sysinfo_gpu()` -- sync but lightweight (thermal sensors only)
- nvidia-smi now queries: `utilization.gpu,memory.used,memory.total,temperature.gpu,name`
- rocm-smi now queries with `--showtemp` flag for temperature data
- Graceful degradation: if no GPU detected, task exits immediately (log once, never retry)
- Timeout warning logged once per timeout sequence (not on every 2s tick)

### ResourceSnapshot extensions (`lib.rs`)
- `gpu_temperature: f32` -- GPU temperature in degrees Celsius (0.0 if unavailable)
- `gpu_name: String` -- GPU device name (empty string if unavailable)

### Parsing functions (pure, testable)
- `parse_nvidia_smi_output()` -- parses CSV from nvidia-smi
- `parse_rocm_smi_output()` -- parses multi-line rocm-smi output
- `parse_proc_diskstats()` -- parses /proc/diskstats for disk I/O
- `parse_proc_net_dev()` -- parses /proc/net/dev for network bytes

### Tests (30+ unit tests)
- nvidia-smi parsing: typical, idle, full load, multi-GPU, no name, empty, garbage, partial values, blank lines, datacenter GPU
- rocm-smi parsing: typical, zero utilization, partial output, empty, no match, temperature variants
- /proc/diskstats parsing: typical sda, nvme, loop/dm/ram skip, empty, short lines, vda virtual disk
- /proc/net/dev parsing: typical, multiple interfaces, empty
- Async integration: runtime non-blocking, backend detect completion
- Constants: GPU_QUERY_TIMEOUT = 2s, GPU_CACHE_TTL = 2s

## Files

- `crates/q-compute/src/resource_monitor.rs` -- main implementation
- `crates/q-compute/src/lib.rs` -- ResourceSnapshot struct (added gpu_temperature, gpu_name)
- `crates/q-compute/src/metrics.rs` -- updated test helper
- `crates/q-compute/src/tunnel.rs` -- updated test helper
