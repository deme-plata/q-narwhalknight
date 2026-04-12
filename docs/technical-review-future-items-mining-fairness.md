# Q-NarwhalKnight Future Improvements: Technical Review

**Date:** 2026-04-12  
**Prepared for:** DeepSeek Peer Review  
**Network Value:** $1B mainnet — all proposals require rigorous risk analysis

---

## 1. CPU Mining Fairness — "Fair Share Rewards"

### What Exists Today

Mining rewards are distributed at three levels in `crates/q-api-server/src/block_producer.rs`:

**Level 1 — Per-Solution Equal Split (current production default):**
```rust
let miner_reward_per_solution = (total_reward - dev_fee) / solutions.len();
```
If 10 solutions are in a block, each miner gets exactly 1/10 regardless of difficulty. A CPU miner submitting one low-difficulty solution gets the same reward as a GPU miner submitting one high-difficulty solution.

**Level 2 — Local PPLNS Pool (v9.1.2):**
When enabled via Stratum, rewards are proportional to difficulty contributed within a PPLNS window. But most miners connect via REST API, bypassing PPLNS entirely.

**Level 3 — Distributed PPLNS via CRDT (v10.0.0):**
Cross-node PPLNS using conflict-free replicated data types. Exists in code but requires Stratum connections.

**The Fairness Problem:**
The `MiningSolution` struct carries `hash_rate_hs` and `difficulty_target` fields but these are **not used** for reward weighting — only for SSE display. REST API miners (the majority) always get equal-per-solution rewards.

### Proposed Fix (4 Phases)

| Phase | Change | Risk | Effort |
|-------|--------|------|--------|
| **A** | Weight rewards by actual solution difficulty | LOW | 3-5 days |
| **B** | VardiffController for REST miners | MEDIUM | 5-8 days |
| **C** | Mandatory PPLNS for all mining paths | MEDIUM | 8-12 days |
| **D** | Per-algorithm difficulty lanes (CPU vs GPU) | HIGH | 20-30 days |

**Phase A detail:** Modify `create_coinbase_transactions` to weight by difficulty achieved:
```
weight_i = 1.0 / difficulty_of(solution_i.hash)
reward_i = miner_total * (weight_i / total_weight)
```
CPU miners benefit because their solutions, while fewer, are weighted by actual difficulty.

**Testing (BEFORE implementation):**
- Property test: sum of rewards = total allocation (no rounding leaks)
- Property test: X% of difficulty = X% of reward (within 1%)
- Simulation: 100 CPU miners (1 KH/s) vs 1 GPU miner (100 MH/s)

---

## 2. Block Consensus Signature Counters

### What Exists Today

`SecurityMetrics` in `crates/q-network/src/security_metrics.rs` only tracks AI worker verification — **not** block consensus signatures. The `verify_spectral_signature_extended` function (dispatches Ed25519/SQIsign/Dilithium5/hybrid) has **no instrumentation**.

### Proposed

Add per-algorithm `AtomicU64` counters:
- `ed25519_verifications_total/failed`
- `sqisign_verifications_total/failed`
- `hybrid_verifications_total/failed`

Increment after each verification in `verify_spectral_signature_extended`. `Ordering::Relaxed` — zero consensus impact, ~1ns per increment.

**Risk: ZERO** | **Effort: 2-3 days** | **Priority: LOW**

---

## 3. Quantum Threat Timeline (Frontend-only)

Horizontal timeline (2019-2040) showing:
- IBM quantum milestones (53→1,121→4,158→100K+ qubits)
- Algorithm break thresholds: Ed25519 at ~2,330 logical qubits, AES-256 at ~6,681
- Current position with pulsing indicator
- All data from published sources (Nature, IBM Roadmap, Gidney & Ekera 2021)
- Physical vs logical qubit distinction explained

**Risk: ZERO** (frontend only) | **Effort: 3-5 days**

---

## 4. Attack Cost Calculator (Frontend-only)

Interactive component — select an algorithm, see:
- Classical security bits (NIST)
- Quantum security bits (Grover/Shor reduction)
- BKZ block size for lattice attacks
- Estimated USD at current cloud prices
- Comparison ratio vs Bitcoin's secp256k1

**Risk: ZERO** (frontend only) | **Effort: 4-6 days**

---

## 5. Testing Infrastructure Improvements

### What Exists

- Unit tests in each crate (`#[cfg(test)]`)
- 28+ integration test files in `tests/`
- Docker compose files (50-node, 5-node, minimal)

### What's Missing

- **No mining reward fairness tests**
- **No signature counter accuracy tests**
- **No PPLNS proportionality tests**
- **No cross-node coinbase determinism tests**

### Proposed (Priority Order)

1. **Mining fairness property tests** — proptest crate, verify proportional rewards (3-4 days)
2. **Signature counter tests** — concurrent verification, verify counter accuracy (1 day)
3. **Docker integration harness** — 3-node mining reward comparison (4-5 days)
4. **Emission property tests** — total supply cap, halving correctness (2-3 days)

**Risk: ZERO** (tests don't change production code) | **Priority: HIGH** (write tests BEFORE implementing mining changes)

---

## 6. SQIsign FFI Linking (Future, CRITICAL Risk)

### Current State

Three-layer architecture:
1. `q-sqisign-sys` — raw C FFI bindings to NIST Round 2 reference implementation
2. `q-sqisign` — safe Rust wrapper with Zeroize
3. `q-crypto-advanced/src/sqisign.rs` — high-level API with `#[cfg(feature = "sqisign-ffi")]`

**CRITICAL FINDING:** Without the FFI feature flag, the scaffold verify function **always returns `Ok(true)`** (line 625). If any code path accidentally uses scaffold verification for consensus, invalid blocks would be accepted. Must audit all call sites.

### Requirements for Mainnet Activation

1. Compile C reference implementation for target platforms
2. Fuzz all FFI entry points (minimum 1M iterations)
3. Verify against NIST KAT vectors
4. Performance validation: SQIsign verify is ~50ms; 250 solutions × 50ms = 12.5s per block — dangerously close to block time
5. Key migration plan with height-gated activation
6. Minimum 30 days testnet soak

**Risk: CRITICAL** | **Effort: 30-60 days** | **Priority: LOW** (Ed25519 not quantum-threatened until ~2030s)

---

## 7. Advanced-Crypto Feature Flag

### Current State

The `advanced-crypto` feature flag is now a **no-op** — `q-crypto-advanced` is always included. The flag only gates FROST threshold committee operations:
- `FrostBlockSigner` — t-of-n validator committees
- `DKGCoordinator` — distributed key generation
- Genus-2 VDF is already active (via q-dag-knight features)
- SQIsign FFI has its own separate feature flag

### Recommendation

Document current state, don't change behavior. FROST requires 10+ independent validators and 90+ days testnet testing before mainnet activation.

**Risk: LOW** (documentation) | **Effort: 2 days**

---

## Priority Ordering

| # | Item | Risk | Effort | Dependencies |
|---|------|------|--------|-------------|
| 1 | Testing Infrastructure | ZERO | 10-12 days | None |
| 2 | Mining Fairness Phase A | LOW | 3-5 days | Tests first (item 1) |
| 3 | Signature Counters | ZERO | 2-3 days | None |
| 4 | Quantum Threat Timeline | ZERO | 3-5 days | None (parallel) |
| 5 | Attack Cost Calculator | ZERO | 4-6 days | None (parallel) |
| 6 | Mining Fairness Phase B-C | MEDIUM | 13-20 days | Phase A validated |
| 7 | Advanced-Crypto Docs | LOW | 2 days | None |
| 8 | SQIsign FFI | CRITICAL | 30-60 days | Items 1, 7 complete |
| 9 | Mining Fairness Phase D | HIGH | 20-30 days | Phase C + economic analysis |

Items 3, 4, 5 can run **in parallel**. Item 1 (testing) should come BEFORE item 2 (mining changes).
