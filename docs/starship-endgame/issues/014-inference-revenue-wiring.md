# Issue #014: Inference Pool Revenue Callback — Wire to Orchestrator

**State**: `open`
**Priority**: MEDIUM
**Labels**: `starship-endgame`, `ai-inference`, `revenue`
**Assigned**: Beta
**Branch**: `feature/safe-batched-sync-v1.0.2`
**Created**: 2026-03-10

---

## Description

The `InferenceWorkerPool` has a `set_orchestrator_callback()` to report task completions and revenue back to the orchestrator's layer stats. The callback mechanism exists in `orchestrator.rs:81` but the revenue path is incomplete — `revenue_earned_micro_qug` in `AIInferenceStats` is always 0 because the inference engine doesn't call the callback with actual token counts.

## Current Behavior

- `InferenceWorkerPool::new()` creates pool with `max_concurrent = 2` regardless of assigned cores
- `set_orchestrator_callback` is called during construction but...
- The `spawn()` background loop processes tasks but revenue is self-tracked in the pool, not forwarded to the orchestrator's layer aggregation

## Fix

1. Wire revenue callback: when inference task completes, invoke `callback(ComputeLayer::AiInference, tokens * price_per_token)`
2. Sync `max_concurrent` with assigned core budget (currently hardcoded to 2)
3. When orchestrator removes all AI cores, `max_concurrent` should go to 0 (currently `.max(1)` prevents this)
4. Add pricing config: `Q_AI_PRICE_MICRO_QUG_PER_TOKEN` env var (default: 1)

## Acceptance Criteria

- [ ] Revenue flows from inference pool → orchestrator layer stats
- [ ] `max_concurrent` tracks assigned core budget
- [ ] Zero cores = zero concurrent inference tasks
- [ ] Dashboard shows accurate AI inference revenue
- [ ] Configurable per-token pricing

## Files

- `crates/q-compute/src/inference_pool.rs`
- `crates/q-compute/src/orchestrator.rs`
