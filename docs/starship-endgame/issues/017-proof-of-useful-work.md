# Issue #017: Proof-of-Useful-Work — Replace Idle Crypto with Revenue

**State**: `in_progress`
**Priority**: MEDIUM
**Labels**: `starship-endgame`, `consensus`, `economics`
**Assigned**: Beta
**Branch**: `feature/safe-batched-sync-v1.0.2`
**Created**: 2026-03-10

---

## Description

Layer 7 (Idle Crypto) wastes energy on synthetic work when no real tasks are available. Replace it with a Proof-of-Useful-Work system where idle compute contributes to tasks that have real economic value — and nodes earn fractional QUG for completing them.

## Useful Work Categories

| Work Type | Revenue Source | Difficulty |
|-----------|---------------|------------|
| AI inference requests | User-paid per token | LOW |
| ZK proof generation | dApp-paid per proof | MEDIUM |
| IPFS pinning | Storage-paid per GB/month | LOW |
| Render jobs | Client-paid per frame | HIGH |
| VDF computation | Protocol-paid per epoch | LOW |
| Model fine-tuning | Researcher-paid per hour | HIGH |

## Architecture

```
Idle cores detected (Layer 7)
  → Query proof-of-work marketplace (local + P2P)
  → Pick highest-revenue task that fits available resources
  → Execute task with deadline guarantee
  → Submit result + proof of completion
  → Receive micro-payment (on-chain or state channel)
```

## Acceptance Criteria

- [ ] Work marketplace API: `POST /api/v1/compute/marketplace/submit` (submit job)
- [ ] Work marketplace API: `GET /api/v1/compute/marketplace/available` (list jobs)
- [ ] Idle layer picks from marketplace instead of synthetic work
- [ ] Revenue tracked per task type in orchestrator stats
- [ ] P2P job announcement on dedicated gossipsub topic
- [ ] Configurable: node operator chooses which work types to accept

## Depends On

- #001 (Orchestrator Layer 7 assignment)
- #002 (P2P tunnels for task distribution)
- #005 (AI inference as a work type)
- #006 (ZK proofs as a work type)

## Progress

**Current**: marketplace.rs (926 lines) — MarketplaceManager with 6 work types (AI inference, ZK proofs, IPFS pinning, render jobs, VDF computation, model fine-tuning). Bid/ask protocol with revenue tracking. Integration with Layer 7 idle dispatch pending.

## Files

- `crates/q-compute/src/marketplace.rs` — MarketplaceManager, task registry, revenue tracking
- `crates/q-api-server/src/marketplace_api.rs` — REST endpoints for marketplace
- `crates/q-compute/src/orchestrator.rs` — Wire marketplace into Layer 7
