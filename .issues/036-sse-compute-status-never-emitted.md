# #036: SSE ComputeStatus event defined but never emitted

**Priority**: MEDIUM
**File(s)**: `crates/q-api-server/src/streaming.rs`, `crates/q-compute/src/orchestrator.rs`
**Risk**: Dashboard shows stale or missing compute data

## Problem

`StreamEvent::ComputeStatus` is defined in the SSE streaming enum (streaming.rs line 481) and has an event name mapping to `"compute-status"` (line 1262), but no code in the entire codebase ever constructs or sends this event.

The frontend dashboard expects periodic compute status updates via SSE to show real-time CPU/GPU/RAM utilization, layer assignments, trainer status, and inference pool activity. Without emission, the dashboard either shows nothing or relies on polling the REST API, which defeats the purpose of the SSE architecture.

The orchestrator has a `status()` method that returns a complete `ComputeStatus` struct, but nothing calls it periodically and wraps it into a `StreamEvent::ComputeStatus` for broadcast.

## Fix

1. In the orchestrator's `spawn()` loop (which already ticks every 1 second), construct a `StreamEvent::ComputeStatus` from `self.status()` and send it to the SSE broadcast channel.
2. Pass the SSE sender (`tokio::sync::broadcast::Sender<StreamEvent>`) into the orchestrator at construction time.
3. Rate-limit emission to every 2-5 seconds to avoid flooding SSE clients with high-frequency resource snapshots.
4. Alternatively, emit on significant state changes only (mode change, layer activation/deactivation, trainer toggle).

## Testing

- cargo check --package q-compute
- cargo test --package q-compute
- cargo check --package q-api-server
