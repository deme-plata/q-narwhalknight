# #035: Inference pool max_concurrent hardcoded to 2

**Priority**: MEDIUM
**File(s)**: `crates/q-compute/src/inference_pool.rs`
**Risk**: Resource waste or contention depending on actual core count

## Problem

`InferenceWorkerPool::new()` initializes `max_concurrent` to 2 (line 121):

```rust
max_concurrent: Arc::new(AtomicU64::new(2)),
```

The pool starts accepting tasks when `update_cores()` gives it cores and sets `accepting = true`. However, before `update_cores()` is ever called, the pool already has `max_concurrent = 2`. If `spawn()` is called and `accepting` is set to true through any path before `update_cores()`, the pool will run 2 concurrent inference tasks with zero assigned cores.

Additionally, `update_cores()` computes `max_conc = (num_cores / 2).max(1)`. When called with 0 cores, it sets `max_concurrent = 1` (not 0), because of `.max(1)`. Combined with the `accepting` flag being the only gate, this means a pool with 0 assigned cores and `accepting = true` could still run 1 task.

The initial value of 2 is also arbitrary — on a 2-core machine where mining takes 75% (1-2 cores), running 2 inference tasks concurrently will cause severe contention.

## Fix

1. Initialize `max_concurrent` to 0 instead of 2. Only set it to a positive value when `update_cores()` is called with actual cores.
2. In `update_cores()`, when `num_cores == 0`, set `max_concurrent = 0` instead of `.max(1)`.
3. Add a check in the spawn loop: if `max_concurrent == 0`, skip task dequeue regardless of `accepting` flag.
4. Consider making the cores-to-tasks ratio configurable rather than hardcoding `num_cores / 2`.

## Testing

- cargo check --package q-compute
- cargo test --package q-compute
