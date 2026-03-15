# Issue #032: Peer-Reviewed Academic Whitepaper

**State**: `open`
**Priority**: HIGH
**Labels**: `starship-endgame`, `academic`, `documentation`
**Assigned**: Beta
**Branch**: `feature/safe-batched-sync-v1.0.2`
**Created**: 2026-03-10

---

## Description

Silvio Micali would say: "Show me the paper." Our system has novel contributions that deserve peer review:

1. **Zero-message DAG-Knight BFT with quantum VDF anchor election** — extends DISC 2021
2. **Adaptive rate-independent emission** — R(lambda) invariant across any block rate
3. **Post-quantum cryptographic agility** — height-gated phase transitions on live network
4. **Homological fork detection** — Betti number topology for consensus consistency
5. **q-flux: libp2p-aware reverse proxy** — blockchain-specific load balancer

## Current State

- `papers/` directory has LaTeX files but none submitted to conferences
- `papers/quantum-aesthetics.pdf` — early theoretical paper
- `papers/qug-emission-economics.tex` — emission model (internal)
- No arxiv preprints or conference submissions

## Target Venues

| Paper | Venue | Deadline |
|-------|-------|----------|
| DAG-Knight + Quantum VDF | DISC 2026 / PODC 2026 | May-June 2026 |
| Adaptive Emission Model | Financial Cryptography 2027 | Sept 2026 |
| PQ Crypto Agility | USENIX Security 2027 | Feb 2027 |
| q-flux Architecture | NSDI 2027 / OSDI 2027 | May 2027 |

## Paper Structure (Main Paper)

```
Title: "Q-NarwhalKnight: Zero-Message DAG-BFT with
        Post-Quantum Cryptographic Agility"

1. Introduction
   - Problem: BFT consensus with quantum threat model
   - Contribution summary

2. System Model
   - Network assumptions (partial synchrony)
   - Byzantine fault model (f < n/3)
   - Quantum threat phases (Q0-Q4)

3. DAG-Knight Consensus
   - Zero-message ordering from DAG structure
   - Quantum VDF anchor election
   - Homological fork detection (Betti numbers)
   - Theorem: Safety under f Byzantine faults
   - Theorem: Liveness under partial synchrony

4. Cryptographic Agility
   - Height-gated upgrade mechanism
   - Ed25519 -> Dilithium5 transition
   - Hybrid mode: classical + PQ simultaneously
   - SQIsign certificates (95.6% size reduction)

5. Adaptive Emission
   - Rate-independent reward: R(lambda) = Target / (lambda * T_year)
   - Budget-based error correction
   - Geometric series convergence proof (21M)

6. Implementation & Evaluation
   - Rust implementation (15K+ LoC consensus, 15K+ LoC proxy)
   - 5-server production deployment
   - Throughput: 48K+ TPS
   - Finality: <2.9s
   - Tor latency: <300ms

7. Related Work
   - Bitcoin (Nakamoto 2008), Ethereum, Solana
   - DAG-Knight (DISC 2021), Narwhal (EuroSys 2022)
   - NIST PQ standards (Dilithium, Kyber)

8. Conclusion
```

## Acceptance Criteria

- [ ] `papers/qnk-dagknight-pq-consensus.tex` — Main paper (12-15 pages, double-column)
- [ ] `papers/qnk-adaptive-emission.tex` — Emission model paper (8-10 pages)
- [ ] `papers/qnk-flux-proxy.tex` — q-flux systems paper (12 pages)
- [ ] Formal proofs of safety and liveness theorems
- [ ] Performance evaluation with reproducible benchmarks
- [ ] arxiv preprint submitted
- [ ] At least 1 conference submission

## Depends On

- #029 (TLA+ spec provides formal basis for proofs)

## Files

- `papers/qnk-dagknight-pq-consensus.tex` — Main paper
- `papers/qnk-adaptive-emission.tex` — Emission paper
- `papers/qnk-flux-proxy.tex` — Systems paper
- `papers/figures/` — Evaluation plots
