# Issue #024: Request Body Streaming for Large Uploads

**Status**: Planned
**Priority**: Medium
**Component**: q-flux
**Labels**: performance, proxy

## Description

Currently, `handle_connection_inner` reads the full request body into a `Vec<u8>` before forwarding to upstream. For large uploads (e.g., smart contract deployments, bulk transaction submissions), this buffers the entire payload in memory.

Implement streaming body forwarding: pipe the client body to upstream as chunks arrive, reducing peak memory usage and time-to-first-byte on the upstream side.

## Approach

1. For POST/PUT with Content-Length > 1MB, use chunked streaming instead of buffered read
2. Pipe client reads directly to upstream writes in 64KB chunks
3. Maintain body_limit enforcement via a running byte counter
4. Keep current buffered path for small bodies (< 1MB) — lower overhead for typical API calls
5. Retry logic cannot apply to streamed bodies (body is consumed); document this tradeoff

## Benefits

- Reduced peak memory: 64KB per streaming request vs full body size
- Lower TTFB for upstream on large requests
- Prevents OOM on many concurrent large uploads

## Files to Change

- `crates/q-flux/src/proxy.rs` — streaming body forwarding in `handle_connection_inner`
- `crates/q-flux/src/config.rs` — `streaming_threshold` config (default 1MB)
