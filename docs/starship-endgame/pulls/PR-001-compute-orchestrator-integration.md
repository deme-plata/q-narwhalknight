# PR #001: Starship Endgame — Compute Orchestrator Integration

**State**: `open`
**Head**: `feature/safe-batched-sync-v1.0.2`
**Base**: `main`
**Author**: Server Beta
**Created**: 2026-03-10
**Labels**: `starship-endgame`, `compute`, `performance`
**Closes**: #001 (partial), #004 (partial), #007 (partial), #008 (partial)

---

## Summary

Integrates the `q-compute` crate into the q-api-server, wiring the Compute Orchestrator, Resource Monitor, OS Tuner, and Trainer into the running node. This is the foundation that all other Starship Endgame features build on.

### What's included

- **q-compute crate** (`crates/q-compute/`) — new crate with:
  - `Orchestrator` — 8-layer adaptive scheduler (mining, inference, ZK, bridge, IPFS, VDF, render, idle)
  - `ResourceMonitor` — 100ms CPU/GPU/RAM/NET/Disk sampling via `/proc` + sysinfo
  - `OsTuner` — Linux + Windows auto-performance tuning (hugepages, sched_fifo, governor, etc.)
  - `Trainer` — 10/12 "cheat engine" performance tricks implemented (F1-F10)
  - `TunnelManager` — P2P compute tunnel lifecycle (framework, needs wiring to gossipsub)

- **compute_api.rs** — 5 REST API endpoints:
  - `GET /api/v1/compute/status` — full compute status
  - `POST /api/v1/compute/mode` — change compute mode at runtime
  - `GET /api/v1/compute/resources` — current ResourceSnapshot
  - `GET /api/v1/compute/trainer` — trainer status + active cheats
  - `POST /api/v1/compute/trainer/toggle` — toggle individual cheats

- **Frontend compute panel** in DeployControlPanel:
  - Resource utilization bars (CPU, GPU, RAM, NET)
  - Layer allocation breakdown
  - Trainer status with active cheats list
  - Tunnel connections display

- **SSE compute events** — `compute_status` event every 5s

## Commits (compute-specific)

```
52b5a8fa feat(v9.5.0): Starship Endgame compute orchestrator + login positioning
cfd2c894 fix(q-compute): Starship audit fixes #029-#039 — honest metrics, async GPU, weighted scheduler
0b8237cd feat: Starship Endgame Revolution — 100% compute utilization project
```

## Files Changed (compute-specific)

| File | Change |
|------|--------|
| `crates/q-compute/Cargo.toml` | NEW — crate manifest |
| `crates/q-compute/src/lib.rs` | NEW — Orchestrator + ComputeMode + LayerConfig |
| `crates/q-compute/src/resource_monitor.rs` | NEW — ResourceMonitor + ResourceSnapshot |
| `crates/q-compute/src/os_tuner.rs` | NEW — OsTuner (Linux + Windows) |
| `crates/q-compute/src/trainer.rs` | NEW — Trainer with F1-F10 cheats |
| `crates/q-compute/src/tunnel_manager.rs` | NEW — TunnelManager framework |
| `crates/q-api-server/src/compute_api.rs` | NEW — REST API endpoints |
| `crates/q-api-server/src/lib.rs` | MODIFIED — add compute routes |
| `crates/q-api-server/Cargo.toml` | MODIFIED — add q-compute dependency |
| `gui/quantum-wallet/src/components/DeployControlPanel.tsx` | MODIFIED — Compute tab |

## Test Plan

- [x] `cargo check --package q-compute` — compiles clean
- [x] `cargo check --package q-api-server` — compiles with compute integration
- [ ] `cargo test --package q-compute` — unit tests pass
- [ ] Manual: `curl localhost:8080/api/v1/compute/status` returns JSON
- [ ] Manual: Frontend shows compute dashboard in admin panel
- [ ] Verify: OS tuning applies gracefully without root (warn, don't crash)
- [ ] Verify: Resource monitor doesn't increase baseline memory > 5MB

## Risk Assessment

- **Consensus impact**: ZERO — purely operational optimization, no block validation changes
- **Memory impact**: LOW — ResourceMonitor samples are bounded (1000 history entries)
- **Failure mode**: Graceful — all OS tuning fails silently without root privileges
- **Rollback**: Safe — removing q-compute dependency returns to previous behavior

## Review Checklist

- [ ] Code reviewed by AI peer
- [ ] No security vulnerabilities (no user input → OS commands)
- [ ] No consensus changes
- [ ] Frontend build succeeds
- [ ] Backend build succeeds
