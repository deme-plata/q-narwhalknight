# Issue #008: Tunnel Mesh Visualization

**State**: `in-progress`
**Priority**: LOW
**Labels**: `starship-endgame`, `frontend`, `visualization`
**Assigned**: Beta
**Branch**: `project/starship-endgame-revolution`
**Created**: 2026-03-08
**Updated**: 2026-03-10

---

## Description

Frontend visualization showing compute tunnels between all nodes. Real-time data flow, capacity heatmap, task routing.

```
     Beta ---------- Epsilon
    / |  \          / |
   /  |   \        /  |
 Gamma  Delta  Alpha  |
   \    |    /        |
    \   |   /        /
     Windows Node ---
```

## Acceptance Criteria

- [ ] D3.js force-directed graph
- [x] Compute status panel in admin dashboard (DeployControlPanel)
- [x] Resource utilization bars (CPU, GPU, RAM, NET)
- [x] Layer allocation breakdown
- [x] Trainer status with active cheats
- [ ] Real-time bandwidth per tunnel
- [ ] Click node -> see compute breakdown
- [ ] Animate task flow between nodes
- [x] SSE compute events (every 5s from backend)

## Progress

- **2026-03-10**: Compute tab added to DeployControlPanel with resource bars, layer breakdown, trainer status, and tunnel connections list. SSE compute_status event added.

## Files

- `gui/quantum-wallet/src/components/DeployControlPanel.tsx` — Compute tab
- `crates/q-api-server/src/compute_api.rs` — API endpoints
- `crates/q-api-server/src/lib.rs` — SSE events
