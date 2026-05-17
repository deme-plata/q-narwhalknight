# Issue #021: Compute Billing & Metering — Track Every Cycle

**State**: `in_progress`
**Priority**: HIGH
**Labels**: `starship-endgame`, `economics`, `billing`
**Assigned**: Beta
**Branch**: `feature/safe-batched-sync-v1.0.2`
**Created**: 2026-03-10

---

## Description

The PaaS billing system (`paas_billing_v2.rs`) handles balance reservations and settlement, but it's not wired into the compute orchestrator. Nodes performing useful work (AI inference, ZK proofs, render jobs) have no standardized metering — revenue tracking is ad-hoc per layer. We need a unified metering pipeline that:

1. Measures compute consumption per task (CPU-seconds, GPU-seconds, memory-GB-seconds)
2. Prices work using a configurable rate card
3. Settles payments on-chain via micro-transactions or batched state channels
4. Provides real-time billing dashboard for node operators

## Current State

- ✅ `MeteringSink` trait implemented with `start_task()`, `record_sample()`, `finalize_task()`
- ✅ Rate card configuration for CPU/GPU/memory pricing per layer
- ✅ Per-job cost tracking and usage summaries
- ✅ Rate limiting by credit balance
- ✅ Configurable pricing tiers
- **Implementation**: `crates/q-compute/src/metering.rs` (659 lines)

## Architecture

```
Task Submitted → Orchestrator assigns to Layer
  → Layer Worker starts task
  → MeteringSink records: { task_id, layer, cpu_ms, gpu_ms, mem_mb_s }
  → Task completes → MeteringSink finalizes
  → BillingManagerV2.settle_bill(task_id, metered_cost)
  → On-chain micro-tx or state channel update
  → Orchestrator.layer_stats[layer].revenue_micro_qug += cost
```

## Acceptance Criteria

- [ ] `MeteringSink` trait with `start_task()`, `record_sample()`, `finalize_task()`
- [ ] Rate card config: price per CPU-second, GPU-second, memory-GB-second per layer
- [ ] All 8 compute layers report metered usage to `PaaSBillingManagerV2`
- [ ] `GET /api/v1/compute/billing` returns per-layer revenue breakdown
- [ ] Settlement batches micro-payments (every 60s or 100 tasks, whichever first)
- [ ] Node operator dashboard shows earnings per hour/day/week

## Depends On

- #001 (Orchestrator layer assignment)
- #014 (Inference revenue wiring — template for other layers)

## Files

- `crates/q-compute/src/metering.rs` — NEW: MeteringSink + rate card
- `crates/q-api-server/src/paas_billing_v2.rs` — Wire metering into settlement
- `crates/q-api-server/src/compute_api.rs` — Add billing endpoint
- `crates/q-compute/src/orchestrator.rs` — Inject MeteringSink into layer workers
