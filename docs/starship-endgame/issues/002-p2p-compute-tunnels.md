# Issue #002: P2P Compute Tunnel (Miner/Node Mesh)

**State**: `open`
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
- [ ] Capacity announcement (cores, GPU TFLOPS, RAM, bandwidth)
- [ ] Task routing (assign to cheapest/closest available)
- [ ] Result verification (2-of-3 redundant compute)
- [ ] Tunnel dashboard in frontend

## Blocked By

- #001 (needs orchestrator for capacity announcements)

## Files (planned)

- `crates/q-network/src/unified_network_manager.rs` — gossipsub topic subscription
- `crates/q-compute/src/tunnel_manager.rs` — tunnel lifecycle (exists, needs wiring)
