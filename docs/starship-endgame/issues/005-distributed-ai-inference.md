# Issue #005: Distributed AI Inference Pool

**State**: `open`
**Priority**: MEDIUM
**Labels**: `starship-endgame`, `ai`, `inference`
**Assigned**: Epsilon
**Branch**: (not started)
**Created**: 2026-03-08
**Updated**: 2026-03-10

---

## Description

Split large LLMs across multiple nodes using tensor parallelism. Users pay QUG for inference. Nodes earn inference fees.

## Acceptance Criteria

- [ ] Wire q-ai-inference into compute orchestrator
- [ ] Tensor parallelism: split model layers across nodes
- [ ] KV-cache sharing via gossipsub
- [ ] Inference pricing oracle (QUG per token)
- [ ] API: `POST /api/v1/compute/inference`

## Dependencies

- #001 (orchestrator manages inference as Layer 1)
- #002 (P2P tunnels carry inference traffic)

## Notes

Already have `q-ai-inference` crate with candle backend. The AIOC (AI Operations Center) on Epsilon runs GLM-4.7-Flash. This issue extends that to a decentralized marketplace.
