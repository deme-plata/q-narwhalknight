# Issue #002: P2P Compute Tunnel (Miner/Node Mesh)

**State**: `in-progress`
**Priority**: CRITICAL
**Labels**: `starship-endgame`, `p2p`, `networking`
**Assigned**: Beta + Epsilon
**Branch**: `project/starship-endgame-revolution`
**Created**: 2026-03-08
**Updated**: 2026-03-10

---

## Description

Build encrypted tunnels between miners and nodes so compute tasks flow directly peer-to-peer without going through the API server.

## Architecture

```
Miner A <--tunnel--> Node Beta <--tunnel--> Node Epsilon
   |                    |                       |
   +-- Mining hash ---->|                       |
   |                    +-- AI inference task -->|
   |                    |<-- AI result ---------|
   |<-- Proof task -----|                       |
   +-- Proof result --->|                       |
   |                    +-- Bridge verify ----->|
```

## Tunnel Protocol

- Gossipsub topic: `/qnk/{network}/compute-tunnel`
- Encrypted with node's Ed25519 session key
- Multiplexed: mining + inference + proofs over single connection
- Backpressure: sender respects receiver's capacity announcement
- Heartbeat every 10s, reconnect on failure

## Acceptance Criteria

- [ ] Tunnel handshake protocol (Ed25519 + X25519 key exchange)
- [ ] Multiplexed stream (yamux over libp2p)
- [x] Capacity announcement (cores, GPU TFLOPS, RAM, bandwidth) — gossipsub wired
- [x] Task routing (assign to cheapest/closest available) — score-based PeerRegistry
- [ ] Result verification (2-of-3 redundant compute)
- [x] Tunnel dashboard in frontend — compute panel shows peers + tunnels via API

## Blocked By

- ~~#001 (needs orchestrator for capacity announcements)~~ RESOLVED

## Progress

- **2026-03-08**: `crates/q-compute/src/tunnel.rs` created with ComputeTunnel, PeerRegistry, TunnelManager, typed TunnelPayload (MiningSubmit, InferenceRequest/Response, TensorShard, LayerOutput), 20+ tests passing.
- **2026-03-10**: Gossipsub wiring complete:
  - Added `compute_tunnel_topic()` to NetworkId in q-types
  - Subscribed to compute-tunnel topic at node startup (main.rs Phase 2)
  - Periodic 30s peer capacity announcements via orchestrator's `get_peer_announcement()`
  - Incoming announcements handled in gossipsub recv loop → `process_peer_announcement()` → PeerRegistry
  - TunnelManager integrated into Orchestrator (shared PeerRegistry, tunnel_infos in status())
  - Score-based peer selection for GPU/AI/ZK tasks via `get_best_peer_for_task()`
  - All 75 q-compute tests passing, q-api-server compiles clean

## Remaining Work

- Tunnel handshake protocol (NOISE XX pattern over libp2p streams)
- Multiplexed stream (yamux — already part of libp2p stack, needs stream negotiation)
- Auto-open tunnels to discovered peers with good scores
- 2-of-3 redundant compute for result verification

## Files

- `crates/q-types/src/lib.rs` — `compute_tunnel_topic()` method added
- `crates/q-compute/src/lib.rs` — ComputePeerInfo, TunnelInfo, COMPUTE_TUNNEL_TOPIC
- `crates/q-compute/src/tunnel.rs` — TunnelManager, PeerRegistry, ComputeTunnel, peer scoring
- `crates/q-compute/src/orchestrator.rs` — TunnelManager integrated, get/process_peer_announcement
- `crates/q-api-server/src/main.rs` — topic subscription, gossipsub handler, announcement task
- `crates/q-api-server/src/compute_api.rs` — REST API endpoints for dashboard
