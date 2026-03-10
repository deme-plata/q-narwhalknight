# Starship Endgame Revolution — Project Tracker

> "Not a single cycle wasted. Every electron earns."

**Branch**: `feature/safe-batched-sync-v1.0.2`
**Started**: 2026-03-08
**Last Updated**: 2026-03-10 (audit pass)

---

## Issues

### Core Compute (Phase 1)

| # | Title | Priority | Status | Assigned | Progress |
|---|-------|----------|--------|----------|----------|
| [#001](issues/001-compute-orchestrator-core.md) | Compute Orchestrator Core | CRITICAL | **Closed** | Beta | 7/7 criteria done |
| [#002](issues/002-p2p-compute-tunnels.md) | P2P Compute Tunnels | CRITICAL | **In Progress** | Beta+Epsilon | Gossipsub wired, 3/6 criteria done |
| [#004](issues/004-trainer-cheat-engine.md) | Trainer Cheat Engine | HIGH | **Closed** | Beta | 12/12 cheats done |
| [#007](issues/007-os-level-auto-tuning.md) | OS-Level Auto-Tuning | HIGH | **Closed** | Beta | All Linux + Windows tuning done |

### Compute Hardening (Phase 1.5)

| # | Title | Priority | Status | Assigned | Progress |
|---|-------|----------|--------|----------|----------|
| [#012](issues/012-async-gpu-monitoring.md) | Async GPU Monitoring | HIGH | Open | Beta | nvidia-smi blocks tokio runtime |
| [#013](issues/013-orchestrator-core-enforcement.md) | Core Enforcement | HIGH | Open | Beta | Assignments are advisory-only |
| [#014](issues/014-inference-revenue-wiring.md) | Inference Revenue Wiring | MEDIUM | Open | Beta | Revenue always shows $0 |

### GPU & Quantum Compute (Phase 2)

| # | Title | Priority | Status | Assigned | Progress |
|---|-------|----------|--------|----------|----------|
| [#003](issues/003-gpu-mining-acceleration.md) | GPU Mining Acceleration | HIGH | Open | Epsilon | Not started |
| [#015](issues/015-quantum-grover-miner-integration.md) | Quantum Grover Miner Integration | HIGH | Open | Epsilon | Python impl in q-grover/, needs Rust FFI |
| [#006](issues/006-zk-proof-farm.md) | ZK Proof Farm | MEDIUM | Open | Gamma | Not started |

### Distributed AI & Marketplace (Phase 3)

| # | Title | Priority | Status | Assigned | Progress |
|---|-------|----------|--------|----------|----------|
| [#005](issues/005-distributed-ai-inference.md) | Distributed AI Inference | MEDIUM | Open | Epsilon | Pool exists, not distributed |
| [#018](issues/018-cross-node-tensor-parallelism.md) | Cross-Node Tensor Parallelism | MEDIUM | Open | Epsilon | TunnelPayload types ready |
| [#017](issues/017-proof-of-useful-work.md) | Proof-of-Useful-Work Marketplace | MEDIUM | Open | Beta | Design only |

### Bridge & Security (Phase 3)

| # | Title | Priority | Status | Assigned | Progress |
|---|-------|----------|--------|----------|----------|
| [#016](issues/016-bridge-safety-compute-verification.md) | Bridge Compute Verification | HIGH | Open | Delta | bridge_safety.rs exists, needs quorum |

### Node Operations

| # | Title | Priority | Status | Assigned | Progress |
|---|-------|----------|--------|----------|----------|
| [#009](issues/009-node-auto-update-deploy-integration.md) | Auto-Update Deploy Integration | HIGH | **Closed** | Beta | announce_update() in safe-deploy.sh |
| [#010](issues/010-node-auto-update-systemd-sigusr1.md) | Auto-Update Systemd + SIGUSR1 | HIGH | **Closed** | Beta | SIGUSR1 handler + graceful shutdown |
| [#011](issues/011-node-auto-update-missing-types.md) | Auto-Update Missing Types | MEDIUM | **Closed** | Beta | Already complete |

### Payments (Phase 4)

| # | Title | Priority | Status | Assigned | Progress |
|---|-------|----------|--------|----------|----------|
| [#019](issues/019-payment-request-api.md) | Payment Request API | HIGH | **Closed** | Beta | 4/6 criteria, SSE event deferred |
| [#020](issues/020-merchant-pos-mode.md) | Merchant POS Mode | HIGH | **Closed** | Beta | 8/10 criteria, component shipped |

### Frontend & Visualization

| # | Title | Priority | Status | Assigned | Progress |
|---|-------|----------|--------|----------|----------|
| [#008](issues/008-tunnel-mesh-visualization.md) | Tunnel Mesh Visualization | LOW | In Progress | Beta | Compute panel done, D3 graph pending |

## Pull Requests

| # | Title | State | Branch | Closes |
|---|-------|-------|--------|--------|
| [PR-001](pulls/PR-001-compute-orchestrator-integration.md) | Compute Orchestrator Integration | Open | `feature/safe-batched-sync-v1.0.2` | #001, #004, #007, #008 (partial) |
| [PR-002](pulls/PR-002-irq-affinity-f11-nuke-fixes.md) | IRQ Affinity + F11 NUKE Fixes | Open | `feature/safe-batched-sync-v1.0.2` | #007 (audit) |
| [PR-003](pulls/PR-003-node-auto-update-integration.md) | Node Auto-Update Integration | Open | `feature/safe-batched-sync-v1.0.2` | #009, #010, #011 |
| [PR-004](pulls/PR-004-compute-tunnel-gossipsub-wiring.md) | Compute Tunnel Gossipsub Wiring | Open | `feature/safe-batched-sync-v1.0.2` | #002 (partial) |
| [PR-005](pulls/PR-005-compute-hardening.md) | Compute Hardening | Draft | `feature/safe-batched-sync-v1.0.2` | #012, #013, #014 |
| [PR-006](pulls/PR-006-qr-mobile-payments.md) | QR Code Mobile Payments | **Merged** | `feature/safe-batched-sync-v1.0.2` | #019, #020 |

## Dependency Graph

```
#001 Compute Orchestrator (CLOSED) ─────────────────────────────┐
 ├── #002 P2P Tunnels (IN PROGRESS — gossipsub wired)           │
 │    ├── #005 AI Inference (needs tunnels for task routing)     │
 │    ├── #018 Tensor Parallelism (needs tunnels for shards)    │
 │    └── #016 Bridge Verification (needs tunnels for quorum)   │
 ├── #004 Trainer (CLOSED)                                      │
 ├── #007 OS Tuning (CLOSED)                                    │
 ├── #008 Visualization (compute panel done, D3 pending)        │
 ├── #012 Async GPU (blocks tokio — HIGH priority fix)          │
 ├── #013 Core Enforcement (advisory → real pinning)            │
 └── #014 Inference Revenue (wire callback to orchestrator)     │
                                                                │
#003 GPU Acceleration ──────────────────────────────────────────┤
 ├── #006 ZK Proof Farm (needs GPU for NTT)                     │
 └── #015 Quantum Grover Miner (needs GPU for simulation)       │
                                                                │
#005 Distributed AI Inference ──────────────────────────────────┤
 ├── #018 Cross-Node Tensor Parallelism                         │
 └── #017 Proof-of-Useful-Work Marketplace                      │
                                                                │
#009 + #010 + #011 Auto-Update (CLOSED)                         │
```

## Server Assignments

| Server | Role | Issues |
|--------|------|--------|
| **Beta** (185.182.185.227) | Coordinator | #001, #002, #004, #007, #008, #009-#014, #017 |
| **Epsilon** (89.149.241.126) | GPU Beast | #002, #003, #005, #015, #018 |
| **Gamma** (109.205.176.60) | CPU Worker | #006 |
| **Delta** (5.79.79.158) | Bridge Node | #016 |
| **Alpha** (161.35.219.10) | Canary | — |

## Milestones

| Milestone | Issues | Target | Status |
|-----------|--------|--------|--------|
| **Phase 1: Orchestrator Foundation** | #001, #002, #004, #007 | 2026-03-10 | 3/4 closed, #002 in progress |
| **Phase 1.5: Hardening** | #012, #013, #014 | 2026-03-15 | 0/3 |
| **Phase 2: GPU & Quantum** | #003, #006, #015 | 2026-03-25 | 0/3 |
| **Phase 3: Distributed Compute** | #005, #016, #017, #018 | 2026-04-10 | 0/4 |
| **Auto-Update** | #009, #010, #011 | 2026-03-10 | 3/3 closed |
| **Payments** | #019, #020 | 2026-03-10 | 2/2 closed |

## Related Docs

- [PROJECT.md](PROJECT.md) — Full project description with architecture and tricks
- [ISSUES.md](ISSUES.md) — Original issue descriptions (raw format)
- [TR-2026-003](../technical-reviews/TR-2026-003-balance-display-discrepancy.md) — Balance display bug (fixed)
