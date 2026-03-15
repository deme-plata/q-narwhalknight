# PR #015: Compute Economics — Billing, Reputation & Multi-GPU

**State**: `open`
**Head**: `feature/safe-batched-sync-v1.0.2`
**Base**: `main`
**Author**: Server Beta
**Created**: 2026-03-10
**Labels**: `starship-endgame`, `compute`, `economics`
**Closes**: #021 (Billing & Metering), #022 (Node Reputation), #023 (Multi-GPU Scheduling)

---

## Summary

Three modules for compute economics — billing, reputation scoring, and multi-GPU scheduling.

### What's included

- **metering.rs** (670 lines) — Compute Billing & Metering
  - `RateCard` — per-resource pricing (CPU, GPU, RAM, bandwidth)
  - `MeteringHandle` — per-task resource tracking
  - `MeteringSink` — aggregated billing with revenue summary
  - 14 unit tests

- **compute_reputation.rs** (804 lines) — Node Reputation Scoring
  - 6-dimensional scoring: uptime, completion, latency, throughput, honesty, stake
  - Exponential decay (0.998/epoch)
  - Automatic blacklisting for dishonest nodes
  - 12 unit tests

- **gpu_scheduler.rs** (763 lines) — Multi-GPU Scheduling
  - Device detection (NVIDIA, AMD, Intel)
  - VRAM-aware layer assignment
  - Load balancing across GPUs
  - 15 unit tests

### Files changed

| File | Lines | Change |
|------|-------|--------|
| `crates/q-compute/src/metering.rs` | 670 | NEW |
| `crates/q-compute/src/compute_reputation.rs` | 804 | NEW |
| `crates/q-compute/src/gpu_scheduler.rs` | 763 | NEW |
| `crates/q-compute/src/lib.rs` | +6 | Module declarations |

### Test plan

- [x] `cargo test --package q-compute` — all 41 tests pass
- [x] `cargo check --package q-compute` — compiles clean
