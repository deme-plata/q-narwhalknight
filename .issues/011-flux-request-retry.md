# Issue #011: Automatic Request Retry on Upstream Failure

**Status**: Done
**Priority**: High
**Component**: q-flux
**Assignee**: Server Beta
**Labels**: enhancement, reliability

## Description

When `upstream.forward()` fails (connect error or timeout), the proxy immediately returns 502. For idempotent requests (GET, HEAD, OPTIONS), it should retry on a different backend before giving up.

## Approach
- Only retry idempotent methods (GET, HEAD, OPTIONS)
- Max 1 retry (try 2 backends total)
- Skip the backend that just failed
- Log the retry attempt

## Files to Change
- `crates/q-flux/src/proxy.rs` — retry logic in handle_connection_inner after upstream error
- `crates/q-flux/src/upstream.rs` — add `forward_excluding()` that skips a specific backend
