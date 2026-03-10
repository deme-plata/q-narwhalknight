# #034: Inference pool revenue callback never wired

**Priority**: MEDIUM
**File(s)**: `crates/q-compute/src/inference_pool.rs`, `crates/q-compute/src/orchestrator.rs`
**Risk**: Inference revenue not tracked in orchestrator totals

## Problem

`InferenceWorkerPool` has a `set_orchestrator_callback()` method (line 135) that accepts a closure for reporting completed tasks and revenue back to the orchestrator. However, this method is never called anywhere in the codebase.

The orchestrator creates the inference pool (line 69):
```rust
let inference_pool = Arc::new(InferenceWorkerPool::new());
```

But never calls `set_orchestrator_callback()` on it. Since the pool is wrapped in `Arc`, and `set_orchestrator_callback` takes `&mut self`, it cannot be called after the `Arc` is created anyway.

As a result, when inference tasks complete, the `if let Some(ref record_fn) = record` check on line 268 always evaluates to `None`, and the orchestrator's per-layer `tasks_completed` and `revenue_micro_qug` counters for AiInference are never incremented. The pool tracks its own internal stats, but these are disconnected from the orchestrator's aggregate `total_revenue_micro_qug`.

## Fix

1. Wire the callback before wrapping in `Arc`. In `Orchestrator::new()`, create the pool, call `set_orchestrator_callback()` with a closure that calls `self.record_task(ComputeLayer::AiInference, revenue)`, then wrap in `Arc`.
2. Alternatively, remove the callback pattern entirely and have the pool hold an `Arc<AtomicU64>` shared with the orchestrator's AiInference `LayerAssignment` directly.
3. Fix the `&mut self` requirement — either use interior mutability (`RwLock<Option<...>>`) or set the callback at construction time via a builder pattern.

## Testing

- cargo check --package q-compute
- cargo test --package q-compute
