# Issue #005: Expose PeerTracker Stats via Admin /peers Endpoint

**Status**: Done
**Priority**: Medium
**Component**: q-flux
**Assignee**: Server Beta
**Labels**: enhancement, observability, admin

## Description

Admin server (port 9090) has /status and /health but no per-peer visibility. Add `GET /peers` returning peer list with tier, connections, bytes, circuit breaker state.

## Expected Response
```json
{
  "total_peers": 42,
  "peers": [
    {
      "peer_id": "12D3KooW...",
      "tier": "Bootstrap",
      "active_connections": 2,
      "bytes_in": 1048576,
      "bytes_out": 524288,
      "circuit_breaker": "Closed",
      "last_seen_secs_ago": 5
    }
  ]
}
```

## Files to Change
- `crates/q-flux/src/admin.rs` — add /peers handler
- `crates/q-flux/src/worker.rs` — pass PeerTracker to admin server
