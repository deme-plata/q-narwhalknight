# PR #017: Quantum Mining, ZK Proofs & P2P Marketplace

**State**: `open`
**Head**: `feature/safe-batched-sync-v1.0.2`
**Base**: `main`
**Author**: Server Beta
**Created**: 2026-03-10
**Labels**: `starship-endgame`, `compute`, `quantum`, `zk`
**Closes**: #006 (ZK Proof Farm), #015 (Quantum Grover Miner), #027 (Marketplace P2P)

---

## Summary

Final three Starship Endgame compute modules: quantum-inspired mining, zero-knowledge proof generation, and P2P marketplace protocol.

### What's included

- **zk_proof_farm.rs** (1732 lines) — ZK Proof Farm
  - STARK/SNARK/Bulletproof proof generation
  - GPU-accelerated NTT (Cooley-Tukey radix-2 DIT over Goldilocks field p=2^64-2^32+1)
  - Recursive proof batching for O(log N) verification
  - Priority queue with configurable concurrency
  - 26 unit tests

- **grover_backend.rs** (1303 lines) — Quantum Grover Mining
  - QuantumCircuitSimulator with oracle construction
  - Amplitude amplification for O(sqrt(N)) nonce search
  - Auto-detection: quantum simulation vs classical fallback
  - MevProtection: quantum RNG nonce ordering prevents front-running
  - 22 unit tests

- **marketplace_p2p.rs** (1228 lines) — Compute Marketplace P2P
  - MarketplaceRouter: gossipsub encode/decode, rate limiting (10/s/peer)
  - OrderBook: aggregated bids, capacities, filters
  - WinnerSelection: score = (1/price) * reputation * (1/time)
  - SettlementManager: full job lifecycle tracking
  - 19 unit tests

### Total: 4,263 lines, 67 tests

### Files changed

| File | Lines | Change |
|------|-------|--------|
| `crates/q-compute/src/zk_proof_farm.rs` | 1732 | NEW |
| `crates/q-compute/src/grover_backend.rs` | 1303 | NEW |
| `crates/q-compute/src/marketplace_p2p.rs` | 1228 | NEW |
| `crates/q-compute/src/lib.rs` | +3 | Module declarations |

### Test plan

- [x] `cargo test --package q-compute` — all 67 new tests pass
- [x] `cargo check --package q-compute` — compiles clean
- [ ] Integration: wire grover_backend into q-miner for quantum-enhanced mining
