# Issue #003: Per-Peer Byte Tracking

**Status**: Open
**Priority**: Low
**Component**: q-flux
**Assignee**: Unassigned
**Labels**: enhancement, observability

## Description

`PeerState` has `bytes_in` and `bytes_out` AtomicU64 fields with `record_rx()` and `record_tx()` methods, but they are never called. Per-peer bandwidth stats are always zero.

## Files to Change
- `crates/q-flux/src/proxy.rs` — call record_rx/record_tx in WebSocket copy loop

## Depends On
- Issue #001 (bandwidth-limited copy loop provides the right place to hook this)
