# Issue #013: Orchestrator Core Enforcement — Actually Pin Cores

**State**: `closed`
**Priority**: HIGH
**Labels**: `starship-endgame`, `compute`, `performance`
**Assigned**: Beta
**Branch**: `feature/safe-batched-sync-v1.0.2`
**Created**: 2026-03-10
**Closed**: 2026-03-10

---

## Description

The orchestrator assigns core budgets to layers (e.g. "Mining gets 6 cores, AI gets 2") but never enforces these limits. The OS scheduler can freely move threads between cores, defeating the purpose of the orchestrator.

## Current Behavior

`orchestrator.rs` sets `assignment.core_budget = N` but this is metadata only — no actual `sched_setaffinity`, `core_affinity::set_for_current()`, or cgroup enforcement.

## Fix

1. Use `core_affinity` crate (already a dependency) to pin layer worker threads to specific core ranges
2. Mining layer: pin to cores 0..N (highest priority, best cache locality)
3. AI Inference layer: pin to cores N..N+M
4. Other layers: share remaining cores with soft affinity
5. For cgroup-capable systems: create `/sys/fs/cgroup/cpu/qnk-layer-{N}/` with `cpu.max` quotas
6. Fallback: if no cgroup or affinity available, advisory-only (current behavior)

## Acceptance Criteria

- [x] Mining threads pinned to dedicated cores via `core_affinity`
- [x] Inference pool respects core range from orchestrator
- [x] Layer isolation prevents cross-layer cache thrashing
- [x] Graceful fallback when affinity fails (warn, don't crash)
- [x] Test: verify `sched_getaffinity` reflects assignments

## Implementation (v9.7.0)

### Core Enforcer (`crates/q-compute/src/core_enforcer.rs`)
- `CoreEnforcer` struct with `enforce_layer_affinity()` and `release_affinity()`
- Linux: raw `libc::sched_setaffinity` with `cpu_set_t` — pins ALL cores in range
- Non-Linux: graceful degradation to `AffinityResult::UnsupportedPlatform`
- `get_thread_affinity()` — reads back current affinity via `sched_getaffinity`

### Orchestrator Integration (`orchestrator.rs`)
- `core_enforcer` field on `Orchestrator` for tracked per-layer enforcement
- `enforce_affinity_for_layer()` — static helper for ad-hoc thread pinning
- `release_affinity_for_layer()` — releases pinning, restores all-core scheduling
- `CoreRange::to_cpuset_str()` — produces cgroup cpuset format
- `CoreRange::to_core_vec()` — converts range to Vec<usize>

### Inference Pool Pinning (`inference_pool.rs`)
- Round-robin `next_core_index` counter for per-task core assignment
- Each spawned inference task calls `core_affinity::set_for_current()` before processing
- Graceful fallback if affinity syscall fails

### Tests (13 new tests)
- `test_core_enforcer_creation`, `test_enforce_empty_core_set`
- `test_enforce_out_of_range_cores`, `test_release_without_prior_enforce`
- `test_enforce_and_release_core0`, `test_enforce_multiple_cores`
- `test_different_layers_independent`
- `test_linux_cpu_set_operations` (Linux-only)
- `test_linux_sched_getaffinity_reflects_set` (Linux-only, verifies affinity readback)
- `test_non_linux_graceful_degradation` (non-Linux only)
- `test_core_range_cpuset_str`, `test_core_range_to_vec`
- `test_core_enforcer_accessible`, `test_release_affinity_graceful`

## Depends On

- #001 (Orchestrator assigns core budgets)

## Files

- `crates/q-compute/src/core_enforcer.rs` (NEW)
- `crates/q-compute/src/orchestrator.rs`
- `crates/q-compute/src/inference_pool.rs`
- `crates/q-compute/src/lib.rs`
