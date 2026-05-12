# Technical Review: IVC / Recursive ZK-SNARK Prerequisites & Roadmap
**Date**: 2026-05-12  
**Scope**: Whitepaper audit, prerequisite gap analysis, revised priority table  
**Status**: Honest assessment — not a PR description  
**Cross-reference**: `papers/RECURSIVE_SNARK_WEAK_SUBJECTIVITY_ELIMINATION.md` (v1.0.0-draft)

---

## Executive Summary

The recursive IVC whitepaper describes the correct long-term bootstrap mechanism. The design is
sound in concept: each epoch produces a LatticeGuard proof πₙ that recursively includes
verification of πₙ₋₁, so a new node verifies the entire chain history in ~10ms by checking a
single proof rather than replaying blocks. This is the right 256-year architecture.

The gap between the whitepaper and deployment is larger than the original sync architecture
review acknowledged. Three hard prerequisites must be in place before any IVC code can ship
to mainnet:

1. **State root in block headers** — the circuit proves state transitions; without committed
   state roots in every block header, you cannot construct the StateTransitionCircuit. This
   was P4 in the prior review. It must be understood as a gating dependency, not just a
   nice-to-have.

2. **Recursive circuit implementation** — the LatticeGuard verifier circuit (C1 in the epoch
   circuit) is currently `todo!("Implement Dilithium verification circuit")`. The BFT
   signature sub-circuit requires ~100K constraints per Dilithium signature; no arithmetic
   gadgets for NTT-domain lattice arithmetic or SHAKE256 exist yet. This is 3–6 months of
   circuit engineering.

3. **Distributed proving infrastructure** — epoch proof generation at ~850K constraints takes
   8–12s on a single GPU. For the first full bootstrap of 17.8M blocks organized into ~18
   epochs (1M blocks/epoch), 18 sequential proofs are needed. With distributed provers and
   GPU pooling this is feasible, but the P2P protocol for proof task distribution (new
   gossipsub topics, proof submission, incentives) doesn't exist yet.

**Revised verdict**: IVC is not a Q3 2026 deliverable. The realistic earliest deployment is
Q1–Q2 2027 **if** state roots ship in Q3 2026 and circuit work begins immediately. Rolling
checkpoints (current P2) remain the right bridge mechanism for the 12–18 months before IVC
is ready.

---

## Whitepaper Audit: What's Real vs. Placeholder

### What's real and working today

| Component | File | Status |
|-----------|------|--------|
| LatticeGuard SNARK prover/verifier | `crates/q-lattice-guard/` | Functional |
| ZK-STARK system with GPU path | `crates/q-zk-stark/` | Functional |
| libp2p gossipsub + Kademlia DHT | `crates/q-network/` | Functional |
| Block storage and DAG structure | `crates/q-storage/` | Functional |
| Dilithium5 signatures | `crates/q-types/` | Functional |

These are genuine. The whitepaper is not vaporware on the infrastructure side.

### What's designed but not implemented

| Component | Status | Gap |
|-----------|--------|-----|
| `LatticeGuardVerifierCircuit` | Struct defined, `synthesize()` written | Gadgets for RLWE commitment verification are stubs |
| `BFTSignatureCircuit` | Struct defined | `add_dilithium_verification()` is `todo!()` |
| `StateTransitionCircuit` | Struct defined | `add_blake3_circuit()` / `add_vdf_constraints()` missing |
| `EpochTransitionCircuit` | Top-level circuit defined | Sub-circuit embedding not implemented |
| `ProverNode` | Outlined | No gossipsub integration for proof tasks |
| `LightClient` | Outlined | No proof request/response protocol |

The whitepaper itself is honest about this — the status is "Research & Design." The circuit
code shown is an architectural sketch, not runnable Rust.

### Critical constraint estimate

The whitepaper estimates ~850K constraints per epoch. That estimate is **optimistic**:

- **Dilithium signature verification**: ~100K constraints per signature for Dilithium3.
  Dilithium5 (our current key size) is closer to 150K. With 5 validators minimum: 750K
  constraints for BFT alone, before the recursive verifier (100K) and state transition
  (200K). Realistic total: **1–1.2M constraints per epoch** under current assumptions.

- **SHAKE256 in-circuit**: The transcript (Fiat-Shamir) uses SHAKE256. This is expensive in
  arithmetic circuits (~50K constraints for a single SHAKE256 invocation). The whitepaper
  proposes switching to Poseidon (an algebraically-friendly hash), which would reduce this
  to ~3–5K constraints. This is a required substitution, not optional.

- **Signature aggregation**: If Dilithium signatures can be aggregated before the circuit
  sees them (aggregate one signature over all 5 validator messages), BFT circuit cost drops
  from ~750K to ~150K. Dilithium does not natively support aggregation — this is an open
  research question (Section 10.1 of the whitepaper). Without it, the BFT sub-circuit
  dominates cost.

**Revised constraint estimate without aggregation**: ~1.2M constraints  
**Revised constraint estimate with aggregation**: ~450K constraints  
**Proving time (RTX 4090) without aggregation**: ~15–20s per epoch  
**Proving time (RTX 4090) with aggregation**: ~6–8s per epoch

---

## The Three Gates Before IVC Can Ship

### Gate 1: State Root in Block Headers

**Why this is a hard gate**: The `StateTransitionCircuit` must prove that applying epoch
blocks transitions the balance state from `prev_state_root` to `new_state_root`. For this
proof to be verifiable, `current_state_root` must be a public input committed into each
block — specifically, into the block header so all nodes verify it during normal block
validation.

Without state roots in headers:
- The circuit has no authoritative target to prove toward
- A prover could claim `new_state_root = X` without any on-chain commitment to verify against
- Nodes receiving the epoch proof cannot validate the claimed state root against chain data

This is why the prior review classified it P4/Q3 2026 but with the note "critical for ZK
circuit design." The dependency is tighter than P4 implied: **IVC circuit work cannot
produce a deployable spec until state roots are in headers**. Circuit engineering can
proceed, but the proof-over-actual-chain cannot be tested end-to-end without it.

**Activation complexity**: As analyzed in the prior review, this is a consensus rule change
requiring a height-gated activation with 90% stake signaling and a 6-month announcement
window. Old nodes reject blocks with the new field; new nodes reject blocks without it.
Plan 2–4 weeks of implementation, ~6 months of deployment runway.

**Implication for IVC timeline**: If state roots ship by Q4 2026 (implementation in Q3,
activation announced, 6-month window), the earliest IVC can prove over real committed state
roots is Q2 2027. This is the realistic timeline.

### Gate 2: Recursive Circuit Gadgets

The epoch transition circuit needs four arithmetic gadgets that do not yet exist:

**Gadget 1: In-circuit RLWE commitment verification**  
File: `LatticeGuardVerifierCircuit::add_commitment_verification_constraints()`  
Requires: NTT (Number Theoretic Transform) gadget over the RLWE modulus. This is the
inner loop of lattice crypto — encoding polynomial multiplication mod q as R1CS constraints.
~20–30K constraints per commitment. 3–4 weeks to implement and test.

**Gadget 2: Poseidon hash**  
The whitepaper already identifies this: replace BLAKE3/SHAKE256 with Poseidon for the
Fiat-Shamir transcript. Poseidon is designed for arithmetic circuits — ~3K constraints
vs. ~50K for a bitwise hash. The Poseidon parameters must match those used in the LatticeGuard
proof; otherwise the transcript reconstructed inside the verifier circuit won't match the
transcript used when the proof was generated. This requires a coordinated change to the
LatticeGuard prover. 2–3 weeks.

**Gadget 3: In-circuit Dilithium verification**  
The `todo!()` in `BFTSignatureCircuit::add_dilithium_verification()`. This is the hardest
gadget: Dilithium verification requires:
- Polynomial arithmetic (NTT multiply, reduce)
- Range checks (||z||_∞ < γ₁ - β)
- SHAKE256 hash (→ Poseidon replacement)
- Matrix-vector multiply over RLWE dimension

Estimated ~150K constraints for Dilithium5. With the signature aggregation research
question unresolved, this must be implemented for individual-signature mode first. 6–8 weeks.

**Gadget 4: BLAKE3 for block header hashing**  
`StateTransitionCircuit::add_blake3_circuit()` — BLAKE3 verification inside a circuit.
~40–60K constraints per invocation. Alternative: switch to Poseidon for block headers (a
consensus change), saving circuit cost but requiring a hard fork. Recommend NOT doing this —
the constraint for in-circuit hash verification is manageable, and changing the block hash
function is a high-risk consensus change with low payoff. 2–3 weeks.

**Total gadget implementation estimate**: 14–18 weeks (parallel teams possible)

### Gate 3: First-Epoch Bootstrap Problem

Before epoch proofs can serve a practical purpose, someone must generate the **genesis epoch
proof** — a proof covering blocks 0 through ~1M. This proof takes the genesis state root as
`prev_state_root` and proves every transaction was correctly applied through h=1,000,000.

With the current ~850K constraint estimate and a single RTX 4090 (10s/epoch), the 18 epochs
needed to cover 17.8M existing blocks would require **180 seconds total** — about 3 minutes.

This seems fine. The catch: the first epoch proof requires running the
`EpochTransitionCircuit` over 1M blocks from genesis. Each block must be fetched, parsed,
and fed into the state transition witness. The *witness generation* time (preparing inputs
for the circuit) is likely to dominate the proving time for historical epochs, potentially
by 10–100×. For a 1M-block epoch, even at 1,300 blocks/sec, building the witness takes
~13 minutes before proving begins.

**Practical implication**: First-ever proof of the 17.8M block history requires careful
witness generation engineering, not just circuit proving. Plan for 4–8 hours of total
compute time on the first bootstrap, distributed across provers.

---

## Revised Priority Table

The IVC path clarifies the dependency order significantly. Here is the corrected table
merging both reviews:

| Priority | Fix / Feature | Deadline | Blocks IVC? | Risk |
|----------|--------------|----------|-------------|------|
| **P0** | Deploy v10.9.5 to Gamma, verify wallets=1348 | Before h=20M | No | Low |
| **P0** | Beta replay reaches wallets=1348 | Before h=20M | No | Low |
| **P1** | Remove coinbase-only turbo sync threshold | This sprint | No | Medium |
| **P1** | SYNC-006 persistent polling loop | This sprint | No | Low |
| **P1** | `save_wallet_balances_replay()` bypass max-wins | This sprint | No | Low |
| **P1** | v10.9.6 new-wallet-only import after replay | This sprint | No | Low |
| **P2A** | **IVC circuit gadgets** (Poseidon, NTT, Dilithium-in-circuit) | Q3 2026 | IVC gate 2 | High |
| **P2B** | **Rolling checkpoint generation protocol** | Q3 2026 | Bridge until IVC | High |
| **P3** | Archive node proxy (Q_ARCHIVE_NODE_URL) | Q4 2026 | No | Low |
| **P3** | Fix `current_height_atomic` → contiguous stored height | Q4 2026 | No | Medium |
| **P4** | **State root in block headers** (6-month activation window) | Q3 announce / Q1 2027 activate | **IVC gate 1** | High |
| **P5** | IVC epoch proofs end-to-end | Q1 2027 earliest | Depends on P4 | Very High |
| **P5** | P2P proof generation network (ProverNode, incentives) | Q2 2027 | Depends on P4 | High |
| **P5** | Light client bootstrap (10ms verify) | Q2 2027 | Depends on P5 | High |
| **P6** | Persistent DAG block reindex (eliminate per-boot cost) | Q4 2026 | No | Medium |

**P2A and P2B run in parallel.** Rolling checkpoints address the immediate sync problem
for the next 12–18 months. IVC gadget work begins now so the circuit is ready when state
roots activate. These are not alternatives — both must happen.

---

## What "Design Starts Now" Means in Practice

The whitepaper says circuit design should start in Q3. "Design" here means:

**Q2 2026 (immediately):**
1. Decide Poseidon parameterization. The Poseidon parameters (t, α, full/partial rounds)
   must be fixed now because they affect the LatticeGuard prover's transcript computation.
   Changing them later requires reproving all historical epochs. Parameters from the
   Poseidon paper (t=3, α=5) are a reasonable starting point.

2. Benchmark the NTT gadget. The RLWE commitment verification constraint count determines
   whether the per-epoch circuit fits in memory for GPU proving. Write a standalone NTT
   arithmetic circuit and measure actual R1CS size before committing to the architecture.

3. Prototype Dilithium verification circuit for one key/signature pair. The `todo!()` must
   become a working (if slow) implementation before the full circuit can be designed.
   This prototype reveals the actual constraint count and flags any hardness assumptions
   that don't hold in arithmetic circuit form.

**Q3 2026 (parallel with state root implementation):**
1. Implement all four gadgets (NTT, Poseidon, Dilithium, BLAKE3-in-circuit).
2. Write and test `EpochTransitionCircuit::synthesize()` against mock epoch data.
3. Generate first test epoch proof (for a 1K-block mock epoch) and measure actual proving time.
4. Begin state root implementation (separate from circuit work but both in Q3).

**Q4 2026:**
1. Announce state root activation height (targeting Q1 2027 activation).
2. Complete `ProverNode` P2P integration with mock provers on testnet.
3. Benchmark witness generation for 1M-block epochs on production hardware.
4. Design incentive mechanism for proof generation (reward parameters, slashing rules).

**Q1 2027:**
1. State root activation on mainnet.
2. Generate genesis epoch proof (covering h=0 to tip as of activation).
3. Begin epoch-by-epoch proof production.
4. Light client available for new nodes (download state + πₙ, verify in ~10ms).

---

## The Signature Aggregation Research Question

The whitepaper (Section 10.1) asks: "Can we aggregate Dilithium signatures to reduce BFT
circuit size?" This is the most consequential open research question for deployment timeline.

Without aggregation: ~750K constraints for BFT alone (5 validators × 150K each). Total
circuit ~1.2M constraints. Proving time ~15–20s/epoch on RTX 4090.

With aggregation: ~150K constraints for BFT. Total circuit ~450K. Proving time ~6–8s/epoch.

The difference is not trivial — the 2.5× speedup matters for keeping epoch proving practical
on commodity hardware. Current state of Dilithium aggregation research:

**Dilithium does not have a known aggregation scheme as of 2025.** Lattice-based signature
aggregation is an active research area (see EAGLE, HAWK, and related schemes), but none
produce aggregated Dilithium5-compatible signatures. The whitepaper cannot assume aggregation
will be available.

**Recommendation**: Design the circuit without aggregation first. Use the 5-signature
(~1.2M constraint) version as the baseline. If aggregation research produces a scheme
compatible with Dilithium5 before the circuit ships, integrate it. If not, the baseline
still works — it just requires slightly better hardware for provers.

---

## Performance Reality Check: 17.8M Block History

For the first-ever IVC deployment, we need to prove the existing 17.8M block history.

**Epoch structure**: With 1M blocks/epoch, this is 18 epochs. But epoch 0 (genesis) needs
special handling — it proves from genesis state (no previous proof) using a base case
circuit rather than recursive composition. This is typically handled by a "base case" proof
that proves the validity of the genesis block only (no recursion needed).

**Proving time estimate** (1.2M constraints, single RTX 4090 at 15s/epoch):  
18 epochs × 15s = 270s ≈ 4.5 minutes of GPU time.

But witness generation (not proving time) is the bottleneck:
- For epoch 1 (h=0 to 1,000,000): loading 1M blocks from Epsilon's DB, hashing, building
  state transition witness. At 1,300 blocks/sec processing rate: ~13 minutes just for data.
- Total witness generation for all 18 epochs: ~234 minutes ≈ 4 hours.
- With parallelized witness generation across 18 cores: ~13 minutes.

**Realistic first-epoch bootstrap time with engineering effort**: 15–30 minutes on a
dedicated proving cluster. Without parallelization: 4–5 hours on a single machine.

This is manageable. It's a one-time cost at activation (Q1 2027). After that, each new
1M-block epoch needs ~15s of GPU time to generate its proof — fully pipelined with block
production.

---

## What Rolling Checkpoints Must Deliver Before IVC

Rolling checkpoints (P2B) serve as the bridge. For this bridge to be adequate until Q1 2027,
they need to be deployed and working by Q3 2026 at the latest — otherwise new nodes joining
in late 2026 face multi-hour sync times.

**Minimum viable rolling checkpoint protocol:**
1. Every 1M blocks, N validator nodes generate and sign a checkpoint snapshot  
2. Checkpoint includes: block hash at height H, balance root at H, wallet map  
3. Requires 2/3 stake co-signature (same quorum as BFT finality)  
4. Checkpoint gossiped on new topic `/qnk/mainnet-genesis/checkpoint`  
5. New nodes accept most recent checkpoint with >50% of visible stake endorsement  
6. Post-checkpoint replay time: 1M blocks / 1,300 blocks/sec = ~13 minutes (acceptable)  

Key difference from IVC: checkpoints still require trusting a validator quorum. They
eliminate the linear replay growth problem but don't eliminate weak subjectivity. IVC
eliminates both.

**If rolling checkpoints are not implemented before IVC ships**, the bridge period
(Q3 2026 – Q1 2027) will see growing sync times for new nodes as the chain extends past
the hardcoded checkpoint. At 1M blocks/month, by Q1 2027 the hardcoded checkpoint will
be 7M blocks stale — ~90 minutes of replay. This is unacceptable UX but survivable for
a small network.

Rolling checkpoints are a should-have, not a hard gate, for IVC. But skipping them means
accepting degraded new-node UX for ~9 months.

---

## Open Questions That Need Decisions Before Circuit Work Begins

1. **Epoch size**: 1M blocks is a reasonable starting point but should be validated. Too
   small = frequent proofs, gossip overhead. Too large = long witness generation. 500K–2M
   blocks is the viable range; 1M blocks is the center of that range.

2. **Poseidon vs. keeping BLAKE3**: In-circuit BLAKE3 costs ~50K constraints per invocation.
   In-circuit Poseidon costs ~3K. If we switch block header hashing to Poseidon, we save
   ~47K constraints per block hash in the StateTransitionCircuit. But this is a consensus
   change. Decision: use in-circuit BLAKE3 (no consensus change), accept the cost.

3. **Base case for genesis**: The recursive chain starts with epoch 0. Epoch 0's proof has
   no `previous_proof` to recursively verify (there is no epoch -1). The standard approach
   is a "base circuit" that proves the genesis block is valid and establishes the genesis
   state root, without recursive verification. This circuit runs once and is never rerun.
   Decision: implement a `GenesisEpochCircuit` that takes the genesis block and produces
   `π₀` with `previous_state_root = [0u8; 32]` (zero root = no prior history).

4. **Proof storage**: Where are epoch proofs stored long-term? Options:
   a. DHT (lossy — no guarantee all proofs are retained)  
   b. Dedicated archive nodes with proof indexing  
   c. Embedded in the block history (one proof per epoch, gossiped alongside blocks)
   
   Recommendation: (c) — proofs become part of the canonical chain history, stored in the
   same RocksDB with key `meta:epoch_proof:{epoch}`. This makes them available to all full
   nodes without requiring a separate DHT lookup.

5. **Incentive mechanism**: The whitepaper proposes a reward for the first prover to submit
   a valid epoch proof. This creates a race condition: multiple GPU provers compete. The
   winner collects the reward; others' work is wasted. An alternative: deterministically
   assign epoch proofs to validators by VRF (Verifiable Random Function), reducing wasted
   compute. Decision needed before ProverNode implementation.

6. **Recursive proof composition paradigm** *(critical — must decide before circuit
   engineering scales up)*: BLS12-381 is not pairing-friendly for verifying its own proofs
   natively. Efficient recursion over BLS12-381 R1CS requires one of:

   **Option A — Cycle of curves (e.g. BLS12-381 + BN254)**  
   - Verify a BLS12-381 Groth16 proof *inside* a BN254 R1CS circuit, then vice versa.  
   - Well-understood and battle-tested (used by Zcash Sapling, Filecoin).  
   - Adds a second curve's field arithmetic to the circuit; ~50–100K extra constraints per
     recursion step for the in-circuit Groth16 verifier.  
   - Requires maintaining two separate proving pipelines.

   **Option B — Folding scheme (Nova / SuperNova / ProtoStar)**  
   - No expensive recursive SNARK verifier inside the circuit; instead "fold" two
     relaxed R1CS instances together. The folding proof is O(1) and cheap.  
   - Much smaller per-step overhead (hundreds of constraints vs. 100K).  
   - Less battle-tested; Nova's security model is newer (Kothapalli et al. 2022).  
   - The final "compression" step still requires a SNARK, but only once at the end.

   **Recommended path**: Start with Option B (Nova-style folding) for prototyping —
   it avoids the cycle-of-curves complexity and is faster to prototype with arkworks'
   `nova-snark` crate. Revisit cycle-of-curves if proving-time benchmarks are
   unsatisfactory. **This decision must be made before spending more than 2 weeks on
   circuit gadgets**, as it determines whether `EpochTransitionCircuit` embeds a
   Groth16 verifier or a folding accumulator.

7. **Bootstrap trust minimisation**: The first epoch proof requires running a trusted
   prover over the entire 17.8M-block history (or at minimum over each epoch boundary
   state root). If the initial proving is done by the core team, this is a **trusted
   setup moment** — users must trust that the genesis-state witness was generated
   honestly. Options to mitigate:

   - **Multi-party witness generation**: Multiple independent nodes each generate
     witness data for different epochs; compare Merkle roots before proving. Any
     discrepancy is a signal of manipulation.
   - **Delay-then-verify**: Publish the epoch proofs publicly; allow a 30-day challenge
     window before the network accepts them as canonical. Any node that can produce a
     contradicting valid state root invalidates the proof.
   - **On-chain transparency log**: Record proof commitments on-chain before bootstrap
     completes, so the provenance is auditable.
   
   Decision needed before designing the bootstrap UX. Without this, IVC's trust
   model (P3: "new nodes don't need to trust a checkpoint") is weakened at genesis.

---

## Verdict: What This Changes About the Roadmap

The IVC whitepaper is architecturally correct. The code audit confirms the infrastructure
exists. The gaps are:

1. **State roots (P4 → critical dependency)**: Must be understood as the enabling
   prerequisite for IVC, not an independent enhancement. Announce Q3 2026, activate Q1 2027.

2. **Circuit gadgets (P2A)**: Begin immediately. Poseidon parameterization this week,
   NTT benchmark within 30 days, Dilithium prototype within 60 days.

3. **Rolling checkpoints (P2B)**: Parallel workstream. Provides UX coverage until IVC.
   Without them, new nodes face multi-hour sync times in late 2026.

4. **IVC deployment (Q1 2027)**: Feasible if P2A and P4 are on track. First deployment
   proves existing 17.8M blocks in ~15–30 minutes of distributed GPU time. After that,
   each new epoch proves in ~15s — fully automated.

The 256-year architecture question from the prior review has a concrete answer now:
the path is rolling checkpoints (18 months) → state roots (Q1 2027) → IVC deployment
(Q2 2027) → light clients with 10ms verification (Q2 2027). This is achievable with
parallel engineering tracks. The single most important thing to do this week is fix the
immediate BAL-001 risk (coinbase-only turbo sync, SYNC-006, wallet count divergence)
and simultaneously start the Poseidon parameterization decision so circuit work can begin.

---

---

## Appendix: q-ivc Crate — Gadget Status (2026-05-12)

### Compile verification (3 independent Debian 12 checks via `rust:bookworm` on Epsilon)

| Commit | Check | Result |
|--------|-------|--------|
| `5c9ec19` | Poseidon real permutation | ✅ 0 errors, 7 warnings, 36.52s |
| `bb6ebff` | BLAKE3 G function + compress | ✅ 0 errors, 7 warnings, 33.10s |
| `1044cf8` | NTT Horner + all-coeff norm | ✅ 0 errors, 5 warnings, 35.89s |

### Gadget status by file

**`src/gadgets/poseidon.rs`** — `PoseidonGadget` (**real R1CS**)  
Implements actual Poseidon permutation: AddRoundConstants → x^5 S-box → MDS.  
- t=3, α=5, 8 full + 57 partial rounds (128-bit on BLS12-381)  
- MDS: Cauchy construction `1/(xᵢ + yⱼ)` computed via field inversion  
- Round constants: SHA3-256 with domain separator `POSEIDON_BLS12381_RC_T3_V1`  
- **~243 R1CS constraints** per permutation (72 from full rounds, 171 from partial)  
- Tests: hash2 constraint count assertion (≥240), determinism, collision resistance  
- Note: Round constants must match LatticeGuard prover transcript exactly.

**`src/gadgets/blake3.rs`** — `Blake3Gadget` (**G function + compression + verify_hash**)  
Implements the full BLAKE3 verification chain with correct `ark-r1cs-std 0.4` UInt32 API:  
- `UInt32::addmany(&[...])` for modular addition (not `wrapping_add` — doesn't exist)  
- `a.xor(&b)` for XOR (returns `Result<UInt32<F>>`)  
- `a.rotr(n)` for rotation (free wire permutation, 0 constraints)  
- Compression function: 7 rounds × 8 G calls (correct BLAKE3 column+diagonal structure)  
- **`verify_hash`: real FpVar↔UInt32 bridge now implemented** (was placeholder):  
  `fpvar_to_uint32`: `to_bits_le()` → range-enforce bits 32..field_size = 0 → `from_bits_le` (~476c/word)  
  `uint32_to_fpvar`: `UInt32::to_bits_le()` → `Boolean::le_bits_to_fp_var` (~32c/word)  
  Full chain: bridge_in + compress + bridge_out + enforce_equal ≈ **44K constraints/block**
- **~640 constraints per G call, ~35,840 for full 7-round compress**  
- Tests: `test_verify_hash_satisfied` (oracle from native_compress), `test_verify_hash_wrong_rejected`

**`src/gadgets/ntt.rs`** — `NttVerifierGadget<F>` (**full NTT butterfly now implemented**)  
- `verify_polynomial_eval`: real Horner's method — (n-1) mul constraints  
- `verify_infinity_norm`: checks ALL n coefficients via `is_cmp` (one-sided)  
- **`verify_ntt_product`: real pointwise a[i]·b[i] == c[i] equality check** (was always-true placeholder)  
- **`ntt` / `intt`: Cooley-Tukey iterative DIT butterfly** (caller provides bit-reversal-indexed roots)  
  Each butterfly: 1 R1CS mul + 2 free additions. Cost: (n/2)×log₂(n) multiplications.  
  For n=256: 1024 constraints per NTT; forward+inverse = ~2K constraints.  
- **`poly_mul`**: NTT → pointwise_mul → INTT — full polynomial multiplication (~3.6K for n=256)  
- Tests: NTT+INTT round-trip (n=2), `poly_mul` identity, wrong-claim rejection

**`src/gadgets/dilithium.rs`** — `DilithiumVerifierGadget` (**scaffold**)  
4-step structure wired to NTT norm + Poseidon challenge hash. BFT threshold counter.  
Actual Az matrix-vector product still `w_prime = sig_z[..8].clone()` (requires NTT gadget).

**`src/circuits/epoch_transition.rs`** — `EpochTransitionCircuit<F>` (**scaffold**)  
Composes all four gadgets. Constraint wiring is correct; constraint bodies are placeholders.

### What "real" means vs. what's still placeholder

| Gadget | Real | Placeholder remaining |
|--------|------|-----------------------|
| Poseidon permutation | ✅ S-box, MDS, round constants | — |
| BLAKE3 G function | ✅ addmany, xor, rotr | — |
| BLAKE3 compress | ✅ 7-round structure | — |
| BLAKE3 verify_hash | ✅ FpVar↔UInt32 bridge + compress + equality (~44K) | — |
| NTT Horner eval | ✅ (n-1) constraints | — |
| NTT Cooley-Tukey butterfly | ✅ ntt/intt/poly_mul (~3.6K for n=256) | Negacyclic ring (X^n+1); root table generation |
| NTT norm check | ✅ all coefficients, is_cmp | Two-sided (negative field representations) |
| NTT product verify | ✅ pointwise a[i]·b[i]==c[i] | — |
| Dilithium Az product | ❌ | Requires negacyclic NTT + matrix-vector wiring |
| Recursive verifier | ❌ | Depends on recursion paradigm decision (OQ-6) |

### Poseidon parameter note

The Cauchy MDS + SHA3 domain-derived constants in the current implementation are
cryptographically sound for development. For production, the parameters should be
re-derived using a seeded MPC ceremony or the `ark-crypto-primitives::crh::poseidon`
parameter generation pipeline with a published seed (ChaCha20 from a verifiable beacon).
The current domain separator `POSEIDON_BLS12381_RC_T3_V1` pins the version; changing
it invalidates all prior proofs.

---

*Review reflects codebase state on branch `feature/safe-batched-sync-v1.0.2`, 2026-05-12.*  
*Last updated: 2026-05-12 — NTT butterfly, poly_mul, and BLAKE3 verify_hash promoted from placeholder to real.*  
*Whitepaper reference: `papers/RECURSIVE_SNARK_WEAK_SUBJECTIVITY_ELIMINATION.md` v1.0.0-draft*  
*Prior review: `docs/technical-review-sync-architecture-2026-05-12.md`*  
*q-ivc crate: commit `55f20a26`, verified compile on Debian 12 2026-05-12*
