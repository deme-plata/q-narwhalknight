# #039: Orchestrator lacks feedback loop on actual utilization

**Priority**: HIGH
**File(s)**: `crates/q-compute/src/orchestrator.rs`
**Risk**: Cores assigned to idle layers waste compute capacity

## Problem

The orchestrator assigns cores to layers based on the compute mode and overall CPU idle percentage, but never checks whether those layers are actually using their assigned cores. Once cores are assigned, they remain assigned until CPU utilization exceeds 90% (the reclaim threshold on line 275).

Scenario: In Full mode with 8 spare cores, 3 layers are active (AI Inference, ZK Proofs, Bridge Verify). Each gets ~2-3 cores. But if no inference requests are queued, no ZK proofs are needed, and no bridge verifications are pending, those 8 cores sit assigned but idle. The orchestrator reports "CPU 40% idle" but 8 cores are reserved for layers doing nothing.

The `tasks_pending` counters exist in `LayerAssignment` but are never checked during the scheduling decision. The scheduler only looks at global `idle_cpu` percentage and does not correlate assigned cores with actual layer utilization.

## Fix

1. In the 1-second scheduler loop, check each layer's `tasks_pending` count. If a layer has 0 pending tasks for N consecutive cycles, deallocate its cores and return them to the spare pool.
2. Implement a demand-based allocation: layers request cores when they have work, and release cores when their queue drains.
3. Track per-layer CPU utilization by comparing the layer's assigned cores' individual CPU usage (from `cpu_per_core`) against a threshold. If assigned cores are <5% utilized for 10 seconds, reclaim them.
4. Add hysteresis to prevent rapid allocation/deallocation oscillation — require sustained demand (e.g., 3 consecutive cycles with pending tasks) before assigning cores.

## Testing

- cargo check --package q-compute
- cargo test --package q-compute
