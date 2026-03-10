# Issue #016: Bridge Safety — Compute-Verified Cross-Chain Proofs

**State**: `in_progress`
**Priority**: HIGH
**Labels**: `starship-endgame`, `bridge`, `security`
**Assigned**: Delta
**Branch**: `feature/safe-batched-sync-v1.0.2`
**Created**: 2026-03-10

---

## Description

The `bridge_safety.rs` module verifies cross-chain deposits by querying external RPC endpoints (BTC, ETH, ZEC, IRON). Currently each node independently queries the external chain — there's no P2P attestation or redundant verification. The compute orchestrator's Bridge Verify layer (Layer 3) should coordinate multi-node verification for trustless bridge safety.

## Architecture

```
Bridge Deposit Detected
  → Layer 3 (BridgeVerify) assigns verification task
  → 3 nodes independently query external chain RPC
  → Each node signs attestation with Ed25519
  → Attestations published on /qnk/{network}/bridge-attestations topic
  → 2-of-3 quorum required before credit is issued
```

## Acceptance Criteria

- [ ] Orchestrator assigns bridge verification to Layer 3 workers
- [ ] Verification task distributed to N peers via compute tunnels
- [ ] Each verifier independently queries external chain and signs result
- [ ] Attestation gossipsub protocol (already have `bridge_attestations_topic()`)
- [ ] 2-of-3 quorum before crediting deposit
- [ ] Timeout handling: if quorum not reached in 5 minutes, escalate to manual review
- [ ] Dashboard: show verification status per pending bridge swap

## Depends On

- #001 (Orchestrator Layer 3 assignment)
- #002 (P2P tunnels for task routing)
- Bridge safety module (`bridge_safety.rs`)

## Progress

**Current**: bridge_verification.rs (1089 lines) — AttestationCollector with 2-of-3 quorum voting, timeout escalation to Layer 4 review. Gossipsub publisher for bridge-attestations topic. Integration pending with orchestrator Layer 3 dispatch.

## Files

- `crates/q-compute/src/bridge_verification.rs` — AttestationCollector, quorum voting, escalation logic
- `crates/q-api-server/src/bridge_safety.rs` — Existing safety controller
- `crates/q-api-server/src/bitcoin_bridge_api.rs` — Existing bridge API
- `crates/q-compute/src/orchestrator.rs` — Layer 3 task routing
