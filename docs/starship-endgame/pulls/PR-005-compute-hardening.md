# PR #005: Compute Orchestrator Hardening — Async GPU, Core Enforcement, Revenue

**State**: `draft`
**Head**: `feature/safe-batched-sync-v1.0.2`
**Base**: `main`
**Author**: Server Beta
**Created**: 2026-03-10
**Labels**: `starship-endgame`, `compute`, `performance`
**Closes**: #012, #013, #014

---

## Summary

Hardens the compute orchestrator from a working prototype to production-quality infrastructure. Three fixes that make the orchestrator actually enforce its decisions and stop blocking the async runtime.

### What's included

1. **Async GPU Monitoring (#012)** — Replace blocking `nvidia-smi` calls with `tokio::process::Command`, cache results for 2s, detect backend once at startup
2. **Core Enforcement (#013)** — Use `core_affinity` crate to actually pin mining/inference threads to their assigned core ranges instead of advisory-only budgets
3. **Inference Revenue Wiring (#014)** — Wire `InferenceWorkerPool` revenue callback to orchestrator layer stats, sync `max_concurrent` with core budget, add per-token pricing config

### Why this matters

Without these fixes:
- GPU monitoring blocks tokio for 50-500ms per sample (breaks SSE + mining)
- Core assignments are decorative — OS freely moves threads across all cores
- AI inference revenue shows $0 even when serving thousands of tokens

## Files Changed (planned)

| File | Change |
|------|--------|
| `crates/q-compute/src/resource_monitor.rs` | Async GPU monitoring + 2s cache |
| `crates/q-compute/src/orchestrator.rs` | Core pinning via `core_affinity` |
| `crates/q-compute/src/inference_pool.rs` | Revenue callback + dynamic max_concurrent |

## Test Plan

- [ ] `cargo test --package q-compute` — all tests pass
- [ ] Verify GPU monitoring doesn't block (add `#[tokio::test]` with timing assertion)
- [ ] Verify core affinity applied (check `/proc/self/status` Cpus_allowed)
- [ ] Verify inference revenue appears in `GET /api/v1/compute/status`
- [ ] Benchmark: measure tokio task latency before/after async GPU fix

## Risk Assessment

- **Consensus impact**: ZERO — purely operational optimization
- **Memory impact**: NONE — no new allocations
- **Failure mode**: Graceful — all three features fail silently with warnings
