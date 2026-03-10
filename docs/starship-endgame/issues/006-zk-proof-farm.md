# Issue #006: ZK Proof Farm

**State**: `open`
**Priority**: MEDIUM
**Labels**: `starship-endgame`, `zk`, `proofs`
**Assigned**: Gamma
**Branch**: (not started)
**Created**: 2026-03-08
**Updated**: 2026-03-10

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
