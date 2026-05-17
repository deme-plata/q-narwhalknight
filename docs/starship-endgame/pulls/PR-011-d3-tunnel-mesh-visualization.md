# PR #011: D3.js Tunnel Mesh Visualization — Parallel Agent Sprint (Terminal 4)

**State**: `open`
**Head**: `feature/safe-batched-sync-v1.0.2`
**Base**: `main`
**Author**: Server Gamma
**Created**: 2026-03-10
**Labels**: `starship-endgame`, `frontend`, `visualization`
**Closes**: #008 (D3 graph portion)

---

## Summary

Creates an interactive force-directed graph showing the P2P compute tunnel mesh between nodes. Part of the parallel agent sprint (Terminal 4 of 4, running on Gamma).

### What's included

- **TunnelMeshGraph.tsx** — D3.js force-directed graph React component
- Nodes sized by `available_cores`, colored by compute mode
- Edges represent active tunnels, width = bandwidth, opacity = latency
- "This node" highlighted at center
- Tooltips with peer_id, cores, GPU, mode, latency
- Auto-refresh every 10 seconds
- Responsive sizing (fills parent container)

### Color scheme

| Compute Mode | Color |
|-------------|-------|
| Eco | Blue (#3B82F6) |
| Balanced | Green (#22C55E) |
| Full | Orange (#F97316) |
| Nuke | Red (#EF4444) |
| This Node | Gold (#EAB308) |

## Files Changed

| File | Change |
|------|--------|
| `gui/quantum-wallet/src/components/TunnelMeshGraph.tsx` | NEW: D3 force graph |
| `gui/quantum-wallet/package.json` | Add d3, @types/d3 |

## Visual Layout

```
                     ┌─────────┐
                     │  Peer C │ (4 cores, Eco)
                     │  ○ blue │
                     └────┬────┘
                          │ thin line (low bandwidth)
    ┌─────────┐     ┌────┴────┐     ┌─────────┐
    │  Peer A │─────│  THIS   │─────│  Peer B │
    │  ● red  │thick│  ★ gold │thick│  ◉ green│
    │  Nuke   │line │  Center │line │  Full   │
    └─────────┘     └────┬────┘     └─────────┘
                         │ medium line
                    ┌────┴────┐
                    │  Peer D │ (8 cores, Nuke)
                    │  ● red  │
                    └─────────┘
```

## Data Source

```typescript
// Fetches from existing compute status endpoint
const response = await fetch('/api/v1/compute/status');
const data = response.json();
// data.peers: [{ peer_id, cores, gpu, mode, latency_ms, bandwidth_kbps }]
// If endpoint not available, shows placeholder with TODO comment
```

## Test Plan

- [ ] `npm run build` — compiles without errors
- [ ] Component renders with mock data when API unavailable
- [ ] Force graph stabilizes within 2 seconds
- [ ] Tooltip shows peer info on hover
- [ ] Auto-refresh updates graph without resetting positions
- [ ] Responsive: renders correctly at 320px, 768px, 1440px widths
- [ ] No memory leak from D3 simulation (cleanup on unmount)

## Risk Assessment

- **Consensus impact**: ZERO — frontend-only, no Rust code changed
- **Bundle size**: d3 adds ~250KB gzipped (acceptable for admin panel)
- **Fallback**: If no peers connected, shows empty graph with "No tunnels active" message
