# Issue #006: ZK Proof Farm

**State**: `in_progress`
**Priority**: MEDIUM
**Labels**: `starship-endgame`, `zk`, `proofs`
**Assigned**: Gamma
**Branch**: `feature/safe-batched-sync-v1.0.2`
**Created**: 2026-03-08
**Updated**: 2026-03-10

## Progress

- `zk_proof_farm.rs`: 1732 lines implemented
  - STARK/SNARK/Bulletproof proof generation
  - GPU-accelerated NTT (Cooley-Tukey algorithm)
  - Recursive proof batching for cost amortization
  - 26 comprehensive unit tests

---

## Description

Background ZK proof generation using idle compute. Other users/apps can request proofs and pay QUG.

## Acceptance Criteria

- [ ] zk-STARK proof generation as background task
- [ ] GPU-accelerated NTT (Number Theoretic Transform)
- [ ] Proof marketplace API
- [ ] Recursive proof batching (amortize cost)
- [ ] Verification: any node can verify in O(log n)

## Dependencies

- #001 (orchestrator manages proofs as Layer 2)
- #003 (GPU acceleration for NTT)
