# Issue #012: Add X-Request-ID Header for Request Tracing

**Status**: Done
**Priority**: Low
**Component**: q-flux
**Assignee**: Server Beta
**Labels**: enhancement, observability

## Description

Add a unique `X-Request-ID` header to every proxied request for end-to-end tracing. If the client sends one, preserve it. If not, generate a short UUID.

## Approach
- Check incoming `X-Request-ID` header
- If missing, generate one (e.g., first 16 chars of UUID)
- Forward to upstream
- Include in response headers
- Include in access log

## Files to Change
- `crates/q-flux/src/proxy.rs` — generate/forward X-Request-ID
- `crates/q-flux/src/access_log.rs` — log the request ID
