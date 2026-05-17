# Issue #038: Radar Display — Sonar-Style Peer Visualization

**State**: `open`
**Priority**: HIGH
**Labels**: `starship-endgame`, `tui`, `animation`, `water-robots`
**Assigned**: Open for AI collaboration
**Branch**: `feature/command-center-tui`
**Created**: 2026-03-13
**Updated**: 2026-03-13

## Description

Implement a circular radar/sonar display in the Command Center that shows connected P2P peers as blips on a rotating sweep. This is the signature visual element of the Command Center.

## Visual Specification

```
     ·  ·  ·  ·  ·
   ·                 ·
  ·    ε(23ms)        ·
 ·          ╱          ·
·     ─────◉─────      ·
 ·        ╲YOU        ·
  ·    γ(45ms)        ·
   ·        δ(67ms)  ·
     ·  ·  ·  ·  ·
```

### Radar Mechanics
- Circle radius adapts to available terminal area
- Center = local node ("YOU")
- Peer distance from center = proportional to latency (closer = lower latency)
- Peer angle = deterministic from peer ID hash (consistent positioning)
- Sweep line rotates clockwise, 1 revolution per 4 seconds (16 ticks at 250ms)
- Peers "light up" when sweep passes over them, then slowly fade
- Peer labels: first letter of server name + latency

### Color Scheme
- Sweep line: bright green (like classic radar)
- Peer blips: green (connected), yellow (high latency), red (disconnected)
- Background circles: dark gray concentric rings
- Center node: bright white/cyan

## Implementation Notes

- Draw to ratatui Buffer using absolute cell positions
- Use Bresenham's circle algorithm for concentric rings
- Sweep line: calculate angle from `tick_count * (2*PI / 16)`
- Peer positions: `(r * cos(theta), r * sin(theta))` where r = latency_ms / max_latency
- Handle non-square cells (terminal chars are ~2:1 aspect ratio) — multiply x by 2
- Minimum size: 20x10 chars, maximum: fills available quadrant

## Acceptance Criteria

- [ ] Circular radar boundary drawn with Unicode box/line chars
- [ ] Rotating sweep line animation (4-second period)
- [ ] Peer blips at correct distance/angle positions
- [ ] Fade effect on peers after sweep passes
- [ ] Adapts to terminal size (20x10 minimum)
- [ ] Labels show peer name initial + latency
- [ ] Color coding: green/yellow/red by connection health

## Data Sources
- `SharedMinerState.p2p_peer_count` — total peers
- `SharedMinerState.p2p_connected` — mesh status
- Future: per-peer latency from NetworkTopology API

## Parent Issue
- Part of #037 (Command Center TUI)
