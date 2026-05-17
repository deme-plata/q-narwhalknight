# PR #018: Command Center — Water Robot Swarm Network Visualization

**State**: `open`
**Head**: `feature/command-center-tui`
**Base**: `feature/safe-batched-sync-v1.0.2`
**Author**: Server Beta + AI Swarm
**Created**: 2026-03-13
**Labels**: `starship-endgame`, `tui`, `animation`, `water-robots`, `collaborative`
**Closes**: #037, #038, #039, #040

## Summary

Transforms the q-miner Network tab (Tab #3) into a **Command Center** with:

1. **Radar Display** — Sonar-style rotating sweep showing connected peers as blips positioned by latency
2. **Swarm Status** — Real-time mesh health metrics + peer roster with reliability bars
3. **Water Robot Ocean** — Animated ASCII ocean where nodes swim as sea creatures, exchanging data as visible particles
4. **Mesh Topology** — Toggle-able [T] network graph showing connections and data flow

## Architecture

### New Files
| File | Purpose | Lines (est.) |
|------|---------|-------------|
| `crates/q-miner/src/ui/tui_views/command_center.rs` | Main Command Center layout + dispatch | ~200 |
| `crates/q-miner/src/ui/tui_views/radar.rs` | Radar sweep animation + peer blips | ~300 |
| `crates/q-miner/src/ui/tui_views/swarm_ocean.rs` | Water robot creatures + ocean waves + particles | ~400 |
| `crates/q-miner/src/ui/tui_views/mesh_topology.rs` | ASCII graph topology view | ~250 |

### Modified Files
| File | Change |
|------|--------|
| `crates/q-miner/src/ui/tui_views/mod.rs` | Add command_center module, wire Tab #3 |
| `crates/q-miner/src/ui/tui_app.rs` | Add Command Center state fields, key bindings |
| `crates/q-miner/src/shared_state.rs` | Optional: add per-peer data if needed |

### Data Flow
```
SharedMinerState (atomics)
    ├─ p2p_peer_count, p2p_connected
    ├─ p2p_challenges_received, p2p_solutions_broadcast
    ├─ bytes_downloaded, bytes_uploaded
    └─ last_challenge_latency_us
        ↓
CommandCenterState (per-tick snapshot)
    ├─ RadarState { sweep_angle, peer_blips[] }
    ├─ SwarmState { creatures[], particles[], wave_phase }
    └─ TopologyState { nodes[], edges[], flow_particles[] }
        ↓
draw_command_center(frame, area, state)
    ├─ draw_radar(frame, radar_area, state)
    ├─ draw_swarm_status(frame, status_area, state)
    ├─ draw_ocean(frame, ocean_area, state)
    └─ draw_topology(frame, topo_area, state) [toggle]
```

## Key Bindings
| Key | Action |
|-----|--------|
| `R` | Cycle radar zoom (1x → 2x → 4x) |
| `S` | Focus swarm animation (full-screen ocean) |
| `P` | Toggle peer details panel |
| `T` | Toggle topology view |
| `Tab` | Switch to next TUI tab (existing) |

## Collaboration Guide for Other AIs

This PR is designed for **multi-agent parallel development**:

- **Agent A**: Radar display (`radar.rs`) — pure rendering, no external deps
- **Agent B**: Swarm ocean (`swarm_ocean.rs`) — animation engine, creature system
- **Agent C**: Topology view (`mesh_topology.rs`) — graph layout algorithm
- **Agent D**: Integration (`command_center.rs` + `mod.rs` + `tui_app.rs`) — wiring

Each module is self-contained with a `draw_*()` function that takes `(frame, area, state)`.
No cross-module dependencies except through `CommandCenterState`.

## Water Robot Operational Potential

The water robot metaphor maps naturally to P2P blockchain nodes:

| Concept | Blockchain | Water Robot |
|---------|-----------|-------------|
| Node | Validator/miner | Autonomous underwater vehicle |
| Gossipsub | Block propagation | Acoustic communication |
| Peer discovery | DHT/Kademlia | Sonar ping/echo |
| Mesh network | P2P overlay | Swarm formation |
| Block producer | Mining | Data collection mission |
| Consensus | BFT agreement | Coordinated swarm behavior |
| Latency | Network hops | Signal propagation delay |

This Command Center visualization makes these parallels tangible and visually stunning.

## Test Plan

- [ ] `cargo check --package q-miner` compiles
- [ ] Radar renders correctly in 80x24 terminal
- [ ] Ocean animation runs smoothly (no flicker, <16ms render)
- [ ] Peer count updates reflected in real-time
- [ ] Key bindings work: R, S, P, T
- [ ] Graceful degradation at small terminal sizes
- [ ] No regression in other TUI tabs

## Risk Assessment

- **LOW**: Pure TUI rendering changes, no consensus/storage/P2P impact
- All changes confined to `crates/q-miner/src/ui/` directory
- Existing Network tab functionality preserved (just enhanced visually)
- No new dependencies required (uses existing ratatui)
