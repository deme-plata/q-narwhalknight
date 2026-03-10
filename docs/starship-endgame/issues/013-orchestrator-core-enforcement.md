# Issue #013: Orchestrator Core Enforcement — Actually Pin Cores

**State**: `open`
**Priority**: HIGH
**Labels**: `starship-endgame`, `compute`, `performance`
**Assigned**: Beta
**Branch**: `feature/safe-batched-sync-v1.0.2`
**Created**: 2026-03-10

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

- [ ] Mining threads pinned to dedicated cores via `core_affinity`
- [ ] Inference pool respects core range from orchestrator
- [ ] Layer isolation prevents cross-layer cache thrashing
- [ ] Graceful fallback when affinity fails (warn, don't crash)
- [ ] Test: verify `sched_getaffinity` reflects assignments

## Depends On

- #001 (Orchestrator assigns core budgets)

## Files

- `crates/q-compute/src/orchestrator.rs`
- `crates/q-compute/src/inference_pool.rs`
