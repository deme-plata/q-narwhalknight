# Issue #001: Compute Orchestrator Core

**State**: `closed`
**Priority**: CRITICAL
**Labels**: `starship-endgame`, `core`, `compute`
**Assigned**: Beta
**Branch**: `project/starship-endgame-revolution`
**Created**: 2026-03-08
**Updated**: 2026-03-10

---

## Description

Create `crates/q-compute/` with the adaptive resource governor that monitors CPU/GPU/RAM/NET every 100ms and assigns work across 8 priority layers. Wire it into q-api-server so it runs at node startup.

## Acceptance Criteria

- [x] Resource sampling at 100ms resolution (`ResourceMonitor`)
- [x] Priority preemption (mining always wins) (`Orchestrator`)
- [x] Core pinning with work-stealing (`Orchestrator::schedule_cores`)
- [x] CLI flag `--compute-mode=full|mining-only|eco`
- [x] Prometheus metrics export (feature-gated `metrics` module with 9 gauges + per-layer labels)
- [x] Disk I/O monitoring via `/proc/diskstats`
- [x] GPU monitoring via multi-backend detection (nvidia-smi, rocm-smi, sysinfo fallback with cached backend)

## Progress

- **2026-03-08**: `crates/q-compute/` created with orchestrator, resource_monitor, os_tuner, trainer, tunnel_manager
- **2026-03-10**: Disk I/O monitoring added to resource_monitor. `compute_api.rs` added to q-api-server with 5 endpoints (status, mode, resources, trainer, toggle). Compilation verified clean.
- **2026-03-10**: Prometheus metrics module (`metrics.rs`) added behind `metrics` feature flag. GPU monitoring upgraded to multi-backend detection (nvidia-smi → rocm-smi → sysinfo). All 7/7 acceptance criteria met. 81 tests passing. Issue CLOSED.

## Blocks

- #002 (P2P tunnels need orchestrator for capacity announcements)
- #004 (Trainer depends on orchestrator layers)

## Files

- `crates/q-compute/src/lib.rs` — Orchestrator, ComputeMode, LayerConfig
- `crates/q-compute/src/resource_monitor.rs` — CPU/GPU/RAM/NET/Disk sampling
- `crates/q-compute/src/os_tuner.rs` — Linux/Windows performance tuning
- `crates/q-compute/src/trainer.rs` — Game trainer performance cheats
- `crates/q-compute/src/tunnel_manager.rs` — P2P compute tunnel mgmt
- `crates/q-api-server/src/compute_api.rs` — REST API endpoints
