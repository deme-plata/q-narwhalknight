# #032: Orchestrator has no priority weighting between layers

**Priority**: MEDIUM
**File(s)**: `crates/q-compute/src/orchestrator.rs`
**Risk**: Low-priority layers starve high-priority layers of cores

## Problem

The orchestrator divides spare cores equally across all active layers (line 250):

```rust
let cores_per_layer = spare_count / layers_to_fill.len().max(1);
```

In Nuke mode, 7 non-mining layers each get `spare_count / 7` cores. This means AI Inference (Layer 1, priority 1) gets the same number of cores as Idle Crypto (Layer 7, priority 7), despite the priority ordering defined in `ComputeLayer`.

The 8-layer priority system (Mining > AI Inference > ZK Proofs > Bridge > IPFS > VDF > Render > Idle Crypto) is documented but never enforced in core assignment. The `ComputeLayer` enum even derives `PartialOrd`/`Ord` to enable priority comparison, but the scheduler ignores it.

## Fix

1. Assign cores proportionally by inverse priority rank. For example, with 24 spare cores across 7 layers: AI Inference gets 7 cores, ZK Proofs gets 5, Bridge gets 4, etc., using a weighted distribution formula.
2. Alternative: assign cores greedily in priority order. Fill Layer 1 up to its demand, then Layer 2 with remaining, and so on. Layers with no pending tasks get zero cores.
3. Integrate with `tasks_pending` counters — layers with zero pending tasks should not receive cores regardless of priority.

## Testing

- cargo check --package q-compute
- cargo test --package q-compute
