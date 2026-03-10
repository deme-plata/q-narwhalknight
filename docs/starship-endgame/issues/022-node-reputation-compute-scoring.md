# Issue #022: Node Reputation for Compute Task Assignment

**State**: `open`
**Priority**: HIGH
**Labels**: `starship-endgame`, `p2p`, `reputation`
**Assigned**: Beta
**Branch**: `feature/safe-batched-sync-v1.0.2`
**Created**: 2026-03-10

---

## Description

`PeerReputationManager` (563+ lines in `peer_reputation.rs`) tracks peer scores for block propagation but has no awareness of compute performance. When the orchestrator dispatches work to remote peers via tunnels, it picks randomly from `PeerRegistry` — a node with 99% task failure rate gets the same priority as one with 100% success.

We need compute-specific reputation scoring so the tunnel manager preferentially routes tasks to reliable, fast peers.

## Scoring Dimensions

| Dimension | Weight | Source |
|-----------|--------|--------|
| Task success rate | 30% | `ResultVerifier` outcomes |
| Average latency | 20% | Tunnel RTT measurements |
| Uptime (last 24h) | 15% | Gossipsub heartbeat tracking |
| Capacity honesty | 15% | Announced vs actual cores/GPU |
| Result quality | 10% | AI inference BLEU/perplexity checks |
| Payment reliability | 10% | Settlement success rate |

## Current State

- `PeerReputationManager` exists with `update_score()`, `decay_scores()`, `ban()`
- `PeerRegistry` in `tunnel.rs` tracks peer capacity announcements
- `ResultVerifier` in `tunnel.rs` verifies task results but doesn't feed back to reputation
- `QuantumReputationSystem` in `q-oracle` is quantum-aware but not compute-aware

## Acceptance Criteria

- [ ] `ComputeReputation` struct with 6 scoring dimensions
- [ ] `PeerRegistry::get_best_peers(n, task_type)` returns peers sorted by compute reputation
- [ ] `ResultVerifier` outcome feeds back into `ComputeReputation` (success/failure/timeout)
- [ ] Automatic demotion: 3 consecutive failures → peer excluded for 10 minutes
- [ ] Automatic promotion: 100 consecutive successes → peer gets priority routing
- [ ] `GET /api/v1/compute/peers` includes reputation scores
- [ ] Reputation persists across restarts (RocksDB-backed)

## Depends On

- #002 (P2P tunnels for task routing)
- #001 (Orchestrator for task dispatch)

## Files

- `crates/q-compute/src/compute_reputation.rs` — NEW: Compute-specific scoring
- `crates/q-compute/src/tunnel.rs` — Wire ResultVerifier → ComputeReputation
- `crates/q-storage/src/peer_reputation.rs` — Add compute dimensions to storage
- `crates/q-api-server/src/compute_api.rs` — Expose reputation in peer list
