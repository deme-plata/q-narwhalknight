# Issue #022: Upstream Round-Robin and Failover Test Suite

**Priority**: Medium
**Status**: Open
**Labels**: testing, reliability

## Problem

`upstream.rs` has 0 unit tests despite implementing critical logic:
- Round-robin backend selection
- Health-based backend skipping
- Super-cluster failover (local → cluster peers)
- Retry on different backend via `forward_excluding()`
- Backpressure via semaphore (capacity limits)
- Inline health recovery on successful responses

## Proposed Tests

1. **Round-robin**: 3 backends, verify even distribution over N requests
2. **Skip unhealthy**: Mark backend unhealthy, verify it's skipped
3. **Cluster failover**: All local backends unhealthy → routes to cluster peer
4. **Fallback**: All backends unhealthy (local + cluster) → falls back to first local
5. **Exclude**: `next_backend_excluding("B")` never returns "B"
6. **Semaphore**: At capacity → immediate reject with "at capacity" error
7. **Inline recovery**: Successful request marks previously-unhealthy backend as healthy

## Notes

Testing `forward()` and `forward_excluding()` requires mocking the hyper client or running a local test server. Consider using `wiremock` or `hyper::server` for integration tests.
