# #031: Ring buffer uses O(n) drain

**Priority**: LOW
**File(s)**: `crates/q-compute/src/resource_monitor.rs`
**Risk**: Minor CPU waste in hot sampling loop

## Problem

The history ring buffer (line 151-156) uses `Vec::push()` followed by `Vec::drain(0..100)` when the vec exceeds 600 entries. `Vec::drain(0..N)` from the front is O(n) because it must shift all remaining elements left in memory. With 600 samples, this means copying ~500 `ResourceSnapshot` structs (each containing a `Vec<f32>` for per-core CPU data) every 10 seconds.

```rust
h.push(snapshot);
if h.len() > 600 {
    h.drain(0..100); // O(n) shift of 500+ elements
}
```

While not catastrophic, this creates a periodic CPU spike inside a 100ms hot loop that is specifically designed to measure CPU utilization — the measurement itself perturbs the result.

## Fix

Replace `Vec<ResourceSnapshot>` with `VecDeque<ResourceSnapshot>` which provides O(1) `push_back()` and O(1) `pop_front()`. Drain one element at a time when exceeding capacity:

```rust
history.push_back(snapshot);
while history.len() > 600 {
    history.pop_front();
}
```

Alternatively, use a fixed-size circular buffer (`circular_buffer` or manual index tracking) to eliminate all allocation overhead.

## Testing

- cargo check --package q-compute
- cargo test --package q-compute
