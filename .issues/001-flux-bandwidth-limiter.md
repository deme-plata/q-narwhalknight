# Issue #001: Wire BandwidthLimiter into WebSocket Splice

**Status**: Open
**Priority**: High
**Component**: q-flux
**Assignee**: Unassigned
**Labels**: enhancement, libp2p, performance

## Description

`BandwidthLimiter` is fully implemented in `libp2p_aware.rs:580-698` but never called from `proxy.rs`. WebSocket connections (including libp2p peers) currently have unlimited bandwidth through the proxy.

## Files to Change
- `crates/q-flux/src/worker.rs` — create BandwidthLimiter, pass to proxy
- `crates/q-flux/src/proxy.rs` — replace `tokio::io::copy` with bandwidth-limited copy

## Acceptance Criteria
- [ ] BandwidthLimiter created per worker in worker.rs
- [ ] Passed through handle_connection → handle_websocket_upgrade
- [ ] WebSocket splice uses chunked read with bandwidth check
- [ ] `cargo check --package q-flux` passes
- [ ] Non-libp2p WebSocket connections unaffected
