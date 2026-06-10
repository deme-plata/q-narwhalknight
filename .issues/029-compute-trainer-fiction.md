# #029: Trainer boost percentages are fiction

**Priority**: MEDIUM
**File(s)**: `crates/q-compute/src/trainer.rs`
**Risk**: Misleading dashboard data, false performance claims

## Problem

`estimated_boost_pct()` sums hardcoded percentages based solely on whether each AtomicBool flag is set, regardless of whether the corresponding optimization was actually applied or had any effect. For example:

- F3 (speed_hack) claims +400% for "SIMD+GPU" but the apply function only sets environment variables and detects hardware features. There is no GPU compute kernel anywhere in the codebase. The +400% is fabricated.
- F1 (infinite_cores) claims +150% for core pinning but only pins the current thread to core 0 — it does not pin mining threads at all.
- F9 (teleport) claims +40% for zero-copy but only sets environment variables that nothing reads.

The resulting boost percentage is displayed on the dashboard and reported in ComputeStatus, giving operators a false sense of performance gains.

## Fix

1. Replace hardcoded percentages with measured deltas. Before each optimization is applied, record a baseline metric (e.g., hashes/sec from mining). After applying, measure again and compute the real delta.
2. If measurement is not feasible at apply time, remove the numeric claim and replace with a boolean "applied" / "skipped" / "failed" status per cheat.
3. For F3 specifically, remove the +400% GPU claim entirely since no GPU compute code exists. Report only SIMD level detected.
4. Add a `CheatStatus` enum: `Applied`, `Skipped(reason)`, `Failed(error)` — return this from each `apply_*` method instead of logging and ignoring failures.

## Testing

- cargo check --package q-compute
- cargo test --package q-compute
