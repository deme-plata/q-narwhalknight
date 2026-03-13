# Issue #037: Command Center — TUI Network Tab Overhaul

**State**: `open`
**Priority**: HIGH
**Labels**: `starship-endgame`, `tui`, `network`, `visualization`, `water-robots`
**Assigned**: All Agents (collaborative)
**Branch**: `feature/command-center-tui`
**Created**: 2026-03-13
**Updated**: 2026-03-13

## Description

Replace the current Network tab (Tab #3) in the q-miner TUI with a **Command Center** — a dynamic, animated visualization of the P2P swarm showing nodes as "water robots" swimming through the network mesh.

The Command Center should feel like a real-time military/space operations dashboard — think submarine sonar room meets NASA mission control.

## Visual Concept

```
┌─ COMMAND CENTER ──────────────────────────────────────────────────────┐
│ ┌─ RADAR ─────────────┐  ┌─ SWARM STATUS ────────────────────────┐  │
│ │         · ·         │  │ Active Nodes: 4/5    Mesh Health: 98% │  │
│ │      ·  ◉  ·        │  │ Gossip Rate: 142/s   Block Prop: 23ms│  │
│ │    ·   YOU   ·      │  │ Challenges: ↓847     Solutions: ↑12   │  │
│ │      · Ω  ·         │  │ Bandwidth: ↓4.2MB/s  ↑1.1MB/s        │  │
│ │    ·    ·    ·      │  ├─ PEER ROSTER ─────────────────────────┤  │
│ │  Epsilon  Delta     │  │ ε Epsilon  10Gbit  23ms  ██████████ 99%│  │
│ │      · Gamma ·      │  │ γ Gamma    1Gbit   45ms  ████████░░ 82%│  │
│ │         ·           │  │ δ Delta    1Gbit   67ms  ███████░░░ 71%│  │
│ └─────────────────────┘  │ β Beta     100Mbit 12ms  █████████░ 94%│  │
│ ┌─ SWARM ANIMATION ──────┴───────────────────────────────────────┐  │
│ │  ≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋  │  │
│ │  ≋≋ 🐋 ←──block──→ 🐬 ≋≋≋≋≋ 🦈 ←──sync──→ 🐙 ≋≋≋≋≋≋≋≋≋  │  │
│ │  ≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋ 🐠←gossip→🐟 ≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋≋  │  │
│ │  ≋≋≋≋≋ waves carry data between water robots ≋≋≋≋≋≋≋≋≋≋≋≋≋≋  │  │
│ └────────────────────────────────────────────────────────────────┘  │
│ [R] Radar zoom  [S] Swarm focus  [P] Peer details  [T] Topology   │
└────────────────────────────────────────────────────────────────────────┘
```

## Sub-Features

### 1. Radar Display (Top-Left Quadrant)
- Circular radar sweep animation (rotating line)
- "You" node at center
- Connected peers as blips at distances proportional to latency
- Blip size = connection strength/reliability
- Sweep reveals peers as it passes over them (fade-in effect)
- Color: Green (healthy), Yellow (degraded), Red (disconnected)

### 2. Swarm Status Panel (Top-Right)
- Real-time counters for mesh health metrics
- Peer roster with per-peer: latency, bandwidth tier, reliability bar
- Color-coded health indicators

### 3. Swarm Animation (Bottom Half)
- ASCII ocean with wave patterns (≋ characters animated)
- Nodes as sea creatures (whale=supernode, dolphin=full node, fish=light node)
- Data flow visualized as particles traveling between creatures
- Dynamic: creatures move, waves flow, data packets animate

### 4. Topology View (Toggle with [T])
- ASCII art mesh graph showing connections between all peers
- Line thickness = connection strength
- Animated data flow along connection lines

## Acceptance Criteria

- [ ] Radar display with sweep animation and peer blips
- [ ] Swarm status panel with real-time metrics from SharedMinerState
- [ ] Peer roster with latency, bandwidth tier, reliability bars
- [ ] Animated ocean/swarm visualization with water robot creatures
- [ ] Data flow particles between nodes (block propagation, gossip)
- [ ] Keyboard shortcuts: [R] radar zoom, [S] swarm focus, [P] peer details, [T] topology
- [ ] Smooth 250ms tick animation (matches existing TUI tick rate)
- [ ] No performance regression (renders under 16ms per frame)
- [ ] Works in 80x24 minimum terminal size (graceful degradation)

## Technical Notes

- Extend `crates/q-miner/src/ui/tui_views/network_view.rs` or create `command_center.rs`
- Use existing `SharedMinerState` atomics: `p2p_peer_count`, `p2p_connected`, etc.
- Follow animation pattern from `q_animation.rs` (phase states, tick-based, seed hashing)
- Use ratatui buffer manipulation for ASCII art (same as starship_animation.rs)
- Data source: existing `/api/v1/network-topology` and `/api/v1/active-peers` endpoints

## Blocks
- None (standalone feature)

## Related
- Issue #037 sub-issues: #038 (Radar), #039 (Swarm Animation), #040 (Topology)
