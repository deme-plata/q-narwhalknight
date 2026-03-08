# Issue #007: Wire GossipsubDedup into WebSocket Data Path

**Status**: Open
**Priority**: Low
**Component**: q-flux
**Assignee**: Unassigned
**Labels**: enhancement, libp2p, performance

## Description

`GossipsubDedup` bloom filter is implemented (`libp2p_aware.rs:716+`) but not wired into the data path. When multiple peers relay the same gossipsub message through the proxy, the proxy could detect and drop duplicates before they reach the backend, reducing load.

## Complexity Note

This is complex because it requires:
1. Parsing WebSocket frames in the bidirectional splice
2. Identifying gossipsub protocol messages within the multistream-select framing
3. Hashing message content for bloom filter lookup

Consider deferring until bandwidth limiter (Issue #001) is done, since both modify the same copy loop.

## Files to Change
- `crates/q-flux/src/proxy.rs` — frame-aware WebSocket copy with dedup check
- `crates/q-flux/src/worker.rs` — create shared GossipsubDedup instance
