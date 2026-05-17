# Issue #002: Wire Circuit Breaker into Upstream Forward Path

**Status**: Done
**Priority**: Medium
**Component**: q-flux
**Assignee**: Server Beta
**Labels**: enhancement, reliability

## Description

`CircuitBreakerState` exists on every `PeerState` and `should_allow_peer()` checks it, but nothing ever calls `record_failure()` or `record_success()`. The circuit breaker is always in Closed state.

## Files to Change
- `crates/q-flux/src/proxy.rs` — record outcomes after WebSocket splice and HTTP forward

## Acceptance Criteria
- [ ] Short-lived WebSocket connections (<1s) record failure
- [ ] Successful connections record success
- [ ] Circuit breaker opens after threshold failures (default 5)
- [ ] Half-open probe allows one connection through after timeout
