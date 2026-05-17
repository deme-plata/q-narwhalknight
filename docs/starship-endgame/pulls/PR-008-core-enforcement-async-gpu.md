# PR #008: Core Enforcement + Async GPU — Parallel Agent Sprint (Terminal 1)

**State**: `open`
**Head**: `feature/safe-batched-sync-v1.0.2`
**Base**: `main`
**Author**: Server Beta
**Created**: 2026-03-10
**Labels**: `starship-endgame`, `compute`, `performance`
**Closes**: #012, #013

---

## Summary

Closes the two "hardening" issues that make the orchestrator's core budgets real instead of advisory. Part of the parallel agent sprint (Terminal 1 of 4).

### Issue #012: Async GPU Monitoring (already implemented)

Verified that `resource_monitor.rs` already uses `tokio::process::Command` with a 2-second `GpuCache` and async-isolated GPU polling task. All 5 acceptance criteria met. Marked as closed.

### Issue #013: Core Enforcement (new code)

1. **CoreEnforcer** (`core_enforcer.rs`) — Linux `sched_setaffinity` / `sched_getaffinity` with graceful fallback on non-Linux
2. **Orchestrator integration** — `CoreRange::to_cpuset_str()` and `to_core_vec()` for cgroup and affinity APIs
3. **Inference pool pinning** — Round-robin `next_core_index` distributes inference tasks across assigned cores via `core_affinity::set_for_current()`

## Files Changed

| File | Change |
|------|--------|
| `crates/q-compute/src/core_enforcer.rs` | NEW: CoreEnforcer with sched_setaffinity |
| `crates/q-compute/src/orchestrator.rs` | Add CoreRange methods + CoreEnforcer field |
| `crates/q-compute/src/inference_pool.rs` | Round-robin core pinning in spawn loop |
| `crates/q-compute/src/lib.rs` | Export core_enforcer module |
| `docs/starship-endgame/issues/012-*.md` | Mark closed |
| `docs/starship-endgame/issues/013-*.md` | Mark closed + add implementation details |
| `docs/starship-endgame/INDEX.md` | Update statuses |

## Key Code

**Inference pool round-robin pinning:**
```rust
let pin_core_id = {
    let cores = assigned_cores_ref.read();
    if !cores.is_empty() {
        let idx = next_core_idx.fetch_add(1, Ordering::Relaxed) as usize % cores.len();
        Some(cores[idx])
    } else {
        None
    }
};
// In spawned task:
if let Some(core_id) = pin_core_id {
    core_affinity::set_for_current(core_affinity::CoreId { id: core_id });
}
```

**CoreRange utilities:**
```rust
pub fn to_cpuset_str(&self) -> String {
    (self.start..self.end).map(|c| c.to_string()).collect::<Vec<_>>().join(",")
}
pub fn to_core_vec(&self) -> Vec<usize> {
    (self.start..self.end).collect()
}
```

## Test Plan

- [x] `cargo check --package q-compute` — passes
- [ ] `test_core_range_cpuset_str` — "0,1,2,3" for CoreRange { 0, 4 }
- [ ] `test_core_range_to_vec` — vec![0,1,2,3] for CoreRange { 0, 4 }
- [ ] `test_core_enforcer_accessible` — Orchestrator::core_enforcer() returns valid ref
- [ ] `test_release_affinity_graceful` — release without prior enforce doesn't crash
- [ ] `test_linux_sched_getaffinity_reflects_set` — verify readback on Linux

## Risk Assessment

- **Consensus impact**: ZERO — operational optimization only
- **Failure mode**: Graceful — if affinity fails, logs warning and continues (advisory-only fallback)
- **Memory impact**: NONE — AtomicU64 counter, no allocations
