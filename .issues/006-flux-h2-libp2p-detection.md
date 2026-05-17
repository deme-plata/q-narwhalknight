# Issue #006: libp2p Detection for HTTP/2 Connections

**Status**: Open
**Priority**: Medium
**Component**: q-flux
**Assignee**: Unassigned
**Labels**: enhancement, libp2p, h2

## Description

libp2p handshake detection only works in the HTTP/1.1 path (`proxy.rs`). HTTP/2 connections via `h2_proxy.rs` bypass PeerTracker entirely. Some libp2p implementations negotiate h2 via ALPN.

## Files to Change
- `crates/q-flux/src/h2_proxy.rs` — add PeerTracker parameter, detect libp2p on h2 streams
- `crates/q-flux/src/worker.rs` — pass PeerTracker to h2_proxy

## Notes
- h2 libp2p detection may need to inspect stream data frames rather than initial handshake
- Lower priority since most libp2p nodes currently use h1.1 WebSocket transport
