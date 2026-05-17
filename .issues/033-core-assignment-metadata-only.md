# #033: Core assignments are metadata-only, never enforced

**Priority**: HIGH
**File(s)**: `crates/q-compute/src/orchestrator.rs`
**Risk**: Core isolation is entirely illusory; OS scheduler ignores assignments

## Problem

`LayerAssignment.cores: Vec<usize>` stores which CPU cores are "assigned" to each compute layer, but this assignment is never enforced at the OS level. The vec is populated by the scheduler (lines 220, 226, 255) and reported in the dashboard, but no `sched_setaffinity()`, `core_affinity::set_for_current()`, or cgroup CPU pinning is ever applied based on these assignments.

The only actual core pinning in the codebase is in `trainer.rs` F1 (`apply_infinite_cores`), which pins the main thread to core 0 — unrelated to the per-layer assignment system.

As a result:
- The dashboard shows "Layer X: cores [4,5,6,7]" but those cores run whatever the OS schedules
- Mining threads and inference threads compete freely on all cores
- The 75% mining / 25% spare core split is cosmetic

## Fix

1. When assigning cores to a layer, call `core_affinity::set_for_current()` or `sched_setaffinity()` on the actual worker threads of that layer.
2. For tokio-based layers, use `tokio::runtime::Builder::new_multi_thread().worker_threads(N)` with a dedicated runtime per layer, and pin those runtime threads to the assigned cores.
3. On Linux, consider using cgroups v2 `cpuset` controller for hard isolation between layers.
4. If true core pinning is not desired, rename the field to something like `cores_budgeted` and document that it is advisory only.

## Testing

- cargo check --package q-compute
- cargo test --package q-compute
