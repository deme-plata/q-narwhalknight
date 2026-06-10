# Issue #010: Graceful Connection Draining on Config Reload

**Status**: Open
**Priority**: Medium
**Component**: q-flux
**Assignee**: Unassigned
**Labels**: enhancement, reliability

## Description

When TLS config is reloaded (cert rotation), new connections get the new config via Arc swap. But there's no active notification to long-lived WebSocket connections that they should reconnect. Add optional drain signaling:

1. After TLS reload, set a "draining" flag on old connections
2. For HTTP/1.1: add `Connection: close` header on next response
3. For WebSocket: send a close frame with reason "TLS rotated, please reconnect"
4. For libp2p: rely on the peer's reconnection logic (libp2p handles this natively)

## Files to Change
- `crates/q-flux/src/proxy.rs` — check drain flag in keepalive loop and WebSocket splice
- `crates/q-flux/src/worker.rs` — propagate drain notification from TLS watcher
