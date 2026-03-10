# PR #009: Inference Revenue Wiring — Parallel Agent Sprint (Terminal 2)

**State**: `open`
**Head**: `feature/safe-batched-sync-v1.0.2`
**Base**: `main`
**Author**: Server Beta
**Created**: 2026-03-10
**Labels**: `starship-endgame`, `compute`, `economics`
**Closes**: #014

---

## Summary

Wires AI inference task completion to revenue tracking so the compute dashboard shows actual earnings instead of $0. Part of the parallel agent sprint (Terminal 2 of 4).

### What was broken

- `orchestrator.layer_stats[AI_INFERENCE].revenue_micro_qug` existed but nothing wrote to it
- `InferenceWorkerPool` had no concept of pricing — tokens generated but never priced
- `ai_api.rs` served completions but didn't record revenue

### What's fixed

1. **ModelTier pricing** — `Small` (100), `Medium` (250), `Large` (500), `XL` (1000) micro-QUG per 1K tokens
2. **Revenue callback** — `InferenceWorkerPool` accepts `Arc<dyn Fn(u64) + Send + Sync>` callback
3. **Job completion tracking** — `record_job_completion(tokens, model_tier)` calculates revenue and calls callback
4. **Revenue summary** — `get_revenue_summary()` returns per-wallet breakdown
5. **API wiring** — `ai_api.rs` calls `record_job_completion()` after each chat completion

## Files Changed

| File | Change |
|------|--------|
| `crates/q-compute/src/inference_pool.rs` | ModelTier enum, RevenueCallback, job tracking |
| `crates/q-compute/src/orchestrator.rs` | Pass revenue callback when creating pool |
| `crates/q-api-server/src/ai_api.rs` | Wire completion → record_job_completion() |
| `docs/starship-endgame/issues/014-*.md` | Update progress |

## Revenue Calculation

```
tokens_generated = 1500
model_tier = ModelTier::Medium  // 250 micro-QUG per 1K tokens
revenue = (1500 * 250) / 1000 = 375 micro-QUG

// Callback updates orchestrator:
orchestrator.layer_stats[ComputeLayer::AiInference].revenue_micro_qug += 375
```

## Test Plan

- [ ] `cargo check --package q-compute` — passes
- [ ] `cargo check --package q-api-server` — passes
- [ ] Revenue appears in `GET /api/v1/compute/status` after inference request
- [ ] ModelTier pricing: Small=100, Medium=250, Large=500, XL=1000 per 1K tokens
- [ ] Revenue callback fires on job completion
- [ ] get_revenue_summary() returns correct per-wallet breakdown

## Risk Assessment

- **Consensus impact**: ZERO — revenue is display-only metadata, not on-chain
- **Memory impact**: MINIMAL — per-wallet HashMap, bounded by unique wallets
- **Failure mode**: Revenue stays at 0 if callback not set (current behavior)
