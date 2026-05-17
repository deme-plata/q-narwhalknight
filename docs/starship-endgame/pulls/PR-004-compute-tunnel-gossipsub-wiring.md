# PR #004: P2P Compute Tunnel — Gossipsub Wiring

**State**: `open`
**Head**: `feature/safe-batched-sync-v1.0.2`
**Base**: `main`
**Author**: Server Beta
**Created**: 2026-03-10
**Labels**: `starship-endgame`, `p2p`, `networking`
**Closes**: #002 (partial)

---

## Summary

Wires the existing `q-compute` tunnel module into the live gossipsub network, enabling P2P compute peer discovery. Nodes now announce their compute capacity (CPU cores, GPU TFLOPS, RAM, bandwidth) every 30 seconds and track discovered peers in a score-based registry.

### What's included

- **`compute_tunnel_topic()`** added to `NetworkId` — `/qnk/{network}/compute-tunnel`
- **Topic subscription** at node startup (Phase 2 initialization)
- **Periodic announcements** — 30s interval task publishes `ComputePeerInfo` via orchestrator
- **Gossipsub handler** — incoming announcements parsed → `process_peer_announcement()` → `PeerRegistry.upsert()`
- **TunnelManager integration** — wired into Orchestrator (shared PeerRegistry, live tunnel info in status)
- **Score-based peer selection** — `get_best_peer_for_task()` ranks peers by GPU/CPU/RAM for task-specific routing

### What this enables

- Nodes discover each other's compute capacity via gossipsub
- Dashboard shows cluster peers with their resources
- Task routing foundation: select best peer for GPU/AI/ZK tasks
- Tunnel manager ready for stream establishment (next PR)

## Commits

```
(pending commit) feat(v9.5.0): Wire compute tunnel gossipsub — topic, announcements, handler
```

## Files Changed

| File | Change |
|------|--------|
| `crates/q-types/src/lib.rs` | MODIFIED — add `compute_tunnel_topic()` to NetworkId |
| `crates/q-compute/src/orchestrator.rs` | MODIFIED — TunnelManager integration, shared PeerRegistry |
| `crates/q-api-server/src/main.rs` | MODIFIED — topic subscription + handler + announcement task |

## Test Plan

- [x] `cargo test --package q-compute` — 75 tests pass
- [x] `cargo test --package q-compute --features metrics` — 81 tests pass
- [x] `cargo check --package q-api-server` — compiles clean
- [ ] Manual: deploy to Beta+Epsilon, verify peer announcements in logs
- [ ] Manual: `curl localhost:8080/api/v1/compute/status` shows `cluster_peers` array

## Risk Assessment

- **Consensus impact**: ZERO — gossipsub messages only, no block validation changes
- **Memory impact**: NEGLIGIBLE — PeerRegistry bounded by 60s TTL eviction
- **Network impact**: LOW — one ~500 byte JSON message per node per 30 seconds
- **Failure mode**: Graceful — if orchestrator missing, announcement task is a no-op
