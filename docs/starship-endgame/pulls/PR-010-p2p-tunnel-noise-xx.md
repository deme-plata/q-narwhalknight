# PR #010: P2P Tunnel NOISE XX Streams — Parallel Agent Sprint (Terminal 3)

**State**: `open`
**Head**: `feature/safe-batched-sync-v1.0.2`
**Base**: `main`
**Author**: Server Beta
**Created**: 2026-03-10
**Labels**: `starship-endgame`, `p2p`, `security`
**Closes**: #002 (partial — 5/6 criteria)

---

## Summary

Adds encrypted multiplexed streams to the compute tunnel module. Previously, tunnels only used gossipsub topic messaging — no actual encrypted point-to-point channels existed between compute peers. Part of the parallel agent sprint (Terminal 3 of 4).

### What's new

1. **NOISE XX handshake** — `snow` crate with `Noise_XX_25519_ChaChaPoly_SHA256` pattern for mutual authentication
2. **yamux multiplexing** — Multiple logical streams over a single TCP connection
3. **TunnelStream** struct — Wraps yamux stream with NOISE encryption
4. **`establish_tunnel()`** — Initiator-side: TCP connect → NOISE handshake → yamux mux
5. **`accept_tunnel()`** — Responder-side: TCP accept → NOISE handshake → yamux server
6. **`open_stream()`** — Open additional multiplexed streams on existing tunnel
7. **`auto_connect_top_peers(n)`** — Connect to N best-scored peers from PeerRegistry

### Why NOISE XX (not IK or NK)?

- **XX**: Both sides authenticate with static keys — required for compute task routing
- **IK**: Initiator sends identity in first message — vulnerable to replay
- **NK**: No initiator authentication — can't verify who's requesting compute

## Files Changed

| File | Change |
|------|--------|
| `crates/q-compute/Cargo.toml` | Add snow = "0.9", yamux = "0.13" |
| `crates/q-compute/src/tunnel.rs` | TunnelStream, handshake, multiplexing |
| `crates/q-compute/src/lib.rs` | Export TunnelStream, StreamData payload variant |

## Connection Flow

```
Initiator                          Responder
    │                                  │
    ├── TCP connect ──────────────────►│
    │                                  │
    ├── NOISE XX e ──────────────────►│  (ephemeral key)
    │◄── NOISE XX e, ee, s, es ───────┤  (ephemeral + static)
    ├── NOISE XX s, se ──────────────►│  (complete handshake)
    │                                  │
    ├── yamux client ─────────────────►│  (multiplexing layer)
    │◄── yamux server ────────────────┤
    │                                  │
    ├── open_stream(0) → task data ──►│  (logical stream per task)
    ├── open_stream(1) → tensor shard►│
    │◄── stream(0) result ────────────┤
    │◄── stream(1) result ────────────┤
```

## Remaining Work (#002 criterion 6/6)

- [ ] NAT traversal / hole-punching for peers behind firewalls
- This PR covers criteria 1-5; NAT traversal is a separate follow-up

## Test Plan

- [ ] `cargo check --package q-compute` — passes with snow + yamux
- [ ] NOISE XX handshake completes between two in-process endpoints
- [ ] yamux multiplexing: open 10 streams, send data on each, verify receipt
- [ ] TunnelManager maintains peer_id → TunnelStream map
- [ ] auto_connect_top_peers(3) connects to 3 highest-scored peers
- [ ] Tunnel survives peer disconnect and reconnects automatically

## Risk Assessment

- **Consensus impact**: ZERO — compute tunnels are separate from block propagation
- **Security**: NOISE XX provides mutual authentication + forward secrecy
- **Failure mode**: If handshake fails, falls back to gossipsub-only (current behavior)
- **Dependencies**: snow 0.9 and yamux 0.13 are well-maintained, widely used
