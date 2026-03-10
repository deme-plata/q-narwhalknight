# Issue #012: Async GPU Monitoring — Unblock Tokio Runtime

**State**: `open`
**Priority**: HIGH
**Labels**: `starship-endgame`, `performance`, `bug`
**Assigned**: Beta
**Branch**: `feature/safe-batched-sync-v1.0.2`
**Created**: 2026-03-10

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

- [ ] GPU monitoring uses `tokio::process::Command`
- [ ] Results cached for 2s with async refresh
- [ ] Backend detection runs once at startup
- [ ] Fallback to cached/zero on timeout
- [ ] No blocking of tokio runtime during GPU queries

## Files

- `crates/q-compute/src/resource_monitor.rs`
