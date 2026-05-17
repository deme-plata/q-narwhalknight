# Issue #040: Mesh Topology — Interactive Network Graph

**State**: `open`
**Priority**: MEDIUM
**Labels**: `starship-endgame`, `tui`, `network`, `water-robots`
**Assigned**: Open for AI collaboration
**Branch**: `feature/command-center-tui`
**Created**: 2026-03-13
**Updated**: 2026-03-13

## Description

Add a toggleable [T] topology view to the Command Center that shows the full P2P mesh as an ASCII graph with animated data flow along edges.

## Visual Specification

```
┌─ MESH TOPOLOGY ──────────────────────────────────────┐
│                                                       │
│           [Epsilon]                                   │
│          /  10Gbit  \                                 │
│         /    23ms    \                                │
│    [Beta]─────────[Delta]                             │
│     100M\   12ms   /1Gbit                             │
│      12ms \       / 67ms                              │
│           [Gamma]                                     │
│            1Gbit                                      │
│             45ms                                      │
│                                                       │
│   ◉ = you    ─── = strong    --- = weak               │
│   Data flow: ·····→ (animated dots along edges)       │
└───────────────────────────────────────────────────────┘
```

### Features
- Force-directed layout positioning (or fixed known positions)
- Edge labels: bandwidth tier + latency
- Edge style: solid (strong), dashed (weak), dotted (degraded)
- Animated particles flowing along edges to show data direction
- Node highlight: currently selected peer shows details

## Acceptance Criteria

- [ ] ASCII graph of mesh topology with labeled edges
- [ ] Animated data flow dots along connection lines
- [ ] Edge styling based on connection quality
- [ ] Toggle with [T] key from main Command Center view
- [ ] Node selection with arrow keys for details panel

## Parent Issue
- Part of #037 (Command Center TUI)
