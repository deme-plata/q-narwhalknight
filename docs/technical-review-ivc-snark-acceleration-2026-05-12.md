# Technical Review: IVC Recursive ZK-SNARK Acceleration Request

**Date:** 2026-05-12
**Branch:** `feature/safe-batched-sync-v1.0.2`
**Audience:** DeepSeek (asked for concrete code contributions)
**Purpose:** Accelerate the Q-NarwhalKnight `q-ivc` recursive SNARK from ~6-12 months of engineering down to a 6-8 week deliverable schedule by parallelizing the gadget work.

---

## 1. Background and Goal

Q-NarwhalKnight is a post-quantum DAG-BFT chain with a 256-year design horizon. Today new nodes bootstrap from a binary-embedded balance checkpoint at h=16,538,868. That checkpoint will eventually be too stale to be useful (in 1 year a 6-hour replay; in 5 years 33 hours). The long-term replacement is an **Incrementally Verifiable Computation** (IVC) recursive SNARK that produces, every epoch (~1M blocks), a proof π_n which:

1. Verifies the prior epoch proof π_{n-1} inside its own circuit
2. Verifies a 2f+1 BFT signature quorum over the epoch's blocks
3. Verifies the BLAKE3 hash-chain integrity of all block headers in the epoch
4. Verifies the state transition `prev_state_root → next_state_root` produced by applying the epoch's blocks

A new node downloads `(current_state, π_n)` and verifies in ~10ms. No checkpoint trust, no peer trust, no chain replay. This is the endgame for decentralization.

The complete design is in `papers/RECURSIVE_SNARK_WEAK_SUBJECTIVITY_ELIMINATION.md`. Most of the cryptographic engineering — recursive composition, the cycle of curves, distributed proving infrastructure — is well-understood prior art. The bottleneck for us is the **circuit gadget implementation**.

---

## 2. Current State of `crates/q-ivc/`

After this session's work, the crate has the following structure:

```
crates/q-ivc/src/
├── lib.rs                         # crate root, status doc
├── gadgets/
│   ├── poseidon.rs                # ✅ real permutation, ported to arkworks FpVar
│   ├── blake3.rs                  # ✅ UInt32 API + G function + compression scaffold
│   ├── ntt.rs                     # ✅ Cooley-Tukey butterfly + poly_mul + negacyclic + signed norm
│   └── dilithium.rs               # ⚠️ Az−c·t via NTT poly_mul; HighBits and full witness types are TODO
└── circuits/
    └── epoch_transition.rs        # ⚠️ four sub-circuits wired; state transition is placeholder
```

What works in R1CS today:

| Gadget | Status | Constraint cost (n=256) |
|--------|--------|-------------------------|
| Poseidon permutation | Real, single permutation works | ~243 / hash |
| BLAKE3 G function | Real, ChaCha-style mixing | ~7 / G call |
| BLAKE3 compression (8 rounds) | Real, witness verification with FpVar↔UInt32 bridge | ~3 800 / block |
| NTT forward (DIT butterfly) | Real | ~1 024 |
| NTT inverse + scale | Real | ~1 280 |
| `poly_mul` (cyclic) | Real | ~3 600 |
| `poly_mul_negacyclic` (Dilithium's X^n+1 ring) | Real, with pre-twist/post-untwist | ~3 600 |
| `verify_signed_norm` (Dilithium z-vector) | Real, two-sided range check | ~401 / coeff |
| Dilithium5 `Az − c·t` | Real over NTT, but **scalar-input scaffold** in `verify()` | ~231K / signature |
| Recursive verifier of π_{n-1} | **Placeholder** — commitment-based, not real inner verifier | — |
| State transition (apply blocks) | **Placeholder** — only enforces `prev_root != next_root` | — |

Tests still failing as of commit `128b7f33`:
- `test_ntt_intt_roundtrip_n2` — root convention bug in the test (roots should be 1, not ω=-1; bug is in test data not code)
- `test_poly_mul_n2`, `test_poly_mul_negacyclic_n2`, `test_verify_signed_norm_in_range` — same root convention issue cascading down

Native (non-circuit) reference implementations exist in `crates/q-lattice-guard/src/ntt.rs` and are correct.

---

## 3. The Real Work Ahead

For the circuit to produce **valid proofs over real data**, we need four things in order of difficulty:

### A. Honest recursive verifier (HARDEST)

Today the recursive sub-circuit is a Poseidon commitment to the prior epoch's public inputs. That is **not** recursive verification — it proves nothing about the prior proof's validity. To do this properly:

- Switch the proving system to a **cycle of curves** (Pasta / Vesta from Halo2, or BLS12-377 over BW6-761 in arkworks)
- Implement an inner verifier circuit that runs the Groth16/PLONK verifier inside R1CS
- Wire `prev_epoch_proof` as a witness whose validity is constrained by the inner verifier

This is the canonical "recursion via folding" problem. **Nova**, **HyperNova**, and **Halo2** all solve it. The arkworks ecosystem has `ark-r1cs-std/ark-relations` infrastructure for this but no out-of-the-box recursive composition crate — that's the part we'd most like DeepSeek's help with.

### B. State transition gadget

For each block in the epoch:
1. Decode the block's transactions (witness)
2. For each transaction: debit sender, credit recipient, validate signatures
3. Recompute the BLAKE3-domain-separated balance hash of the resulting wallet table
4. Constrain `next_state_root == BLAKE3(domain_sep || sorted_wallets || balances)`

In a circuit, "the wallet table" is a Merkle/Verkle/sparse Merkle tree of wallet→balance. Each transaction is a Merkle proof of the sender, an update, a Merkle proof of the recipient, an update, and a recomputed root.

Constraint budget per transaction: ~50K (Merkle path × 2 + sig + balance arithmetic). Per block: ~10× txs × 50K = 500K. Per million-block epoch: 500B — clearly we batch and use **non-Merkle** schemes (e.g., per-block delta commitments) or skip per-tx and verify only the aggregate state delta.

The state-transition gadget design itself is a major spec decision.

### C. Full Dilithium5 verifier with structured witness

The current `DilithiumVerifierGadget::verify` takes flat `&[FpVar<F>]` slices for `sig_z` etc. and uses Poseidon to hash transcript inputs. The real Dilithium5 verify needs:

- `PublicKeyVar` and `SignatureVar` structured types (k×L polynomial matrix, vectors of polynomials)
- `compute_az_minus_ct_negacyclic` is implemented — needs to be called from `verify()`
- `HighBits` gadget (extract the high-order γ₂ bits of each coefficient) — the missing piece
- Hint vector check (`UseHint` from the Dilithium spec)
- Challenge derivation (Poseidon transcript matches signing-side coordinated change)
- Norm check via `verify_signed_infinity_norm` — already implemented

When this is done, one `DilithiumVerifierGadget::verify` call adds ~231K + ~30K (HighBits) + ~3K (Poseidon) = **~265K constraints per signature**. With 5 validators in BFT mode: ~1.3M constraints just for BFT verification.

### D. Aggregation to reduce signature cost

5× Dilithium verifies at 265K each is feasible but expensive. Production deployment likely wants signature aggregation (BLS-style for Dilithium, or a STARK-friendly aggregate sig scheme). This is research territory — there is no off-the-shelf in arkworks.

---

## 4. What We Want From DeepSeek (Concrete Asks)

Listed in priority order. Each item is intended to be a self-contained code drop we can integrate into `crates/q-ivc/src/` with light review.

### Ask 1 — Inner-verifier sub-circuit (P0, ~3 weeks human-equivalent)

Need: a working `RecursiveVerifierGadget<F: PrimeField>` that takes a Groth16 (or PLONK) `Proof` and `VerifyingKey` as witness, and constrains the proof verification equation inside R1CS.

Acceptable approaches in order of preference:
1. **Halo2 / Pasta cycle** ported to arkworks-style API (we use ark-r1cs-std heavily; we'd port the test framework if needed)
2. **BLS12-377 / BW6-761 cycle** using `ark-groth16` over BLS12-377 as the inner proof, `ark-groth16` over BW6-761 as the outer
3. **Folding scheme** (Nova-style or HyperNova): produce a single accumulator that gets folded each epoch, verified once

If you can give us option 3 (folding), it's the lowest constraint count per epoch. We're open to whichever is most mature.

Specific deliverable: a single `gadgets/recursive_verifier.rs` file with:

```rust
pub struct RecursiveVerifierGadget;

impl RecursiveVerifierGadget {
    /// Constrain that `proof` is a valid Groth16 proof of `vk` for public inputs `public`.
    /// Returns Boolean<F> = true iff the verification equation holds.
    pub fn verify_groth16<F: PrimeField>(
        cs: ConstraintSystemRef<F>,
        vk: &VerifyingKeyVar<F>,
        proof: &ProofVar<F>,
        public_inputs: &[FpVar<F>],
    ) -> Result<Boolean<F>, SynthesisError>;
}
```

…or the folding-scheme analogue.

Constraint budget target: ≤ 500K constraints per verification (compared to ~1.5M for naive in-circuit Groth16 verifier).

### Ask 2 — Sparse Merkle state-transition gadget (P0, ~2 weeks)

Need: a `StateTransitionGadget` that verifies one transaction's effect on a sparse Merkle tree of (wallet_addr_32 → balance_u128).

```rust
pub struct StateTransitionGadget;

impl StateTransitionGadget {
    /// Verify that applying transaction `tx` to a state with root `pre_root`
    /// yields a new state with root `post_root`. Witnesses include Merkle paths
    /// for sender and recipient.
    pub fn apply_transaction<F: PrimeField>(
        cs: ConstraintSystemRef<F>,
        pre_root: &[FpVar<F>; 8],     // 32 bytes packed as 8 × u32
        post_root: &[FpVar<F>; 8],
        sender_addr: &[UInt8<F>; 32],
        sender_balance_pre: &FpVar<F>,
        sender_path: &[FpVar<F>],     // 32-depth Merkle path
        recipient_addr: &[UInt8<F>; 32],
        recipient_balance_pre: &FpVar<F>,
        recipient_path: &[FpVar<F>],
        amount: &FpVar<F>,
    ) -> Result<(), SynthesisError>;
}
```

Hash: BLAKE3 (we have the gadget already in `gadgets/blake3.rs`).

Constraint budget target: ≤ 30K per transaction (BLAKE3 node-hash is ~3.8K, depth 32, so 32 × 3.8K = 122K per Merkle proof × 2 proofs = 244K — we need a non-BLAKE3 hash for in-circuit Merkle, probably Poseidon at ~250/node × 32 = 8K per proof × 2 = 16K. Plus balance arithmetic and signature verify ≈ 30K). Help us pick the right hash.

### Ask 3 — HighBits / UseHint Dilithium decomposition gadget (P1, ~1 week)

The hardest remaining piece of `DilithiumVerifierGadget`. Spec is well-defined (FIPS 204 ML-DSA, formerly CRYSTALS-Dilithium):

- `HighBits(r, α) = (r₁, r₀)` where r = r₁·α + r₀ and |r₀| ≤ α/2
- `UseHint(h, r, α)` reconstructs the high bits from a 1-bit hint

For Dilithium5: α = 2γ₂ = 95232. Coefficients live in F_q for q = 8380417. Inside our circuit we use F_r of BLS12-381 (much larger), so the math itself is fine — the challenge is the bit-decomposition cost.

Specific deliverable: `gadgets/dilithium.rs` additions:

```rust
pub fn high_bits<F: PrimeField>(
    cs: ConstraintSystemRef<F>,
    coefficient: &FpVar<F>,
    alpha: u64,
) -> Result<(FpVar<F>, FpVar<F>), SynthesisError>;  // (r₁, r₀)

pub fn use_hint<F: PrimeField>(
    cs: ConstraintSystemRef<F>,
    hint_bit: &Boolean<F>,
    coefficient: &FpVar<F>,
    alpha: u64,
) -> Result<FpVar<F>, SynthesisError>;  // reconstructed high bits
```

Constraint budget target: ≤ 50 constraints per coefficient. Dilithium5 has 256 coefficients × 8 polynomials in w' = 2048 calls = ≤ 100K constraints.

### Ask 4 — Single recursive proof end-to-end (P1, integration task, ~1 week)

Once asks 1-3 land, an end-to-end demo:
- Take a witness from `crates/q-storage` (real two-block state transition)
- Build the EpochTransitionCircuit with real inputs
- Generate a Groth16 proof
- Verify the proof out-of-circuit
- Print: constraint count, proving time, verification time, proof size

Target numbers we'd consider successful for a 100-block "mini-epoch" proof:
- Constraints: < 5M
- Proving time on a 48-core CPU: < 30 minutes
- Verification time: < 50ms
- Proof size: < 5KB (Groth16) or < 200KB (PLONK with folding)

### Ask 5 — GPU proving prototype (P2, optional but valuable)

The 1M-block-per-epoch design means a single proof generation can take hours on CPU. A GPU prover using `ark-msm` or `ec-gpu` reduces this 10-50×. Optional ask; we can pursue this in parallel.

---

## 5. Code Conventions and Integration

To make merging your contributions painless:

- **Crate**: `crates/q-ivc/src/` only (no changes to other crates without flagging)
- **Field**: Default to `ark_bls12_381::Fr`. Generic over `F: PrimeField` where reasonable
- **R1CS lib**: arkworks (`ark-r1cs-std`, `ark-relations`). Compatible types: `FpVar<F>`, `UInt8<F>`, `UInt32<F>`, `Boolean<F>`
- **Tests**: every gadget must have at least: (a) a smoke test that generates constraints without panic, (b) a positive test that satisfies the circuit, (c) a negative test that's rejected. Use `ConstraintSystem::<Fr>::new_ref()`
- **Constraint counts**: every test should `println!` the `cs.num_constraints()` so we can track budget
- **Naming**: snake_case functions, PascalCase types. Document constraint cost on every public function
- **Native reference**: if your code has a native (out-of-circuit) analog, put it in `crates/q-lattice-guard/` so we can test correctness by comparing native vs in-circuit output on the same inputs

We do not use GitHub. The repo is at `code.quillon.xyz` (local git). To contribute:
1. Send us a patch (`git format-patch`) or a tarball of changed files
2. We'll review and apply on the `feature/safe-batched-sync-v1.0.2` branch
3. Tests must pass with `cargo test --package q-ivc`

---

## 6. Timeline Hope

Without help: 6-12 months of single-engineer circuit work.

With DeepSeek delivering asks 1-3 over 4-6 weeks:
- Week 1-2: ask 3 (HighBits) — unblocks full Dilithium gadget
- Week 2-4: ask 2 (state transition) — unblocks the state-changing sub-circuit
- Week 3-6: ask 1 (recursive verifier) — the big one, parallelizable
- Week 6-7: ask 4 (end-to-end demo) — integration
- Week 7-8: hardening, real-data validation against `q-storage` witnesses

That puts a working (not optimized) IVC proof end-to-end at **early July 2026**. Optimization and GPU proving extend to Q4 2026. Production deployment in Q1 2027.

---

## 7. Why This Matters

When this lands, Q-NarwhalKnight becomes the first post-quantum L1 with cryptographic genesis-to-now verification. New nodes never replay. The "Epsilon is authoritative" social trust assumption is replaced with a 5KB proof anyone can verify in 50ms. That is what "decentralized" actually means for a 256-year chain.

We're not asking for a research paper. We're asking for arkworks code that compiles, passes `cargo test`, and has constraint counts within the stated budgets. The math is mostly solved; the engineering is what kills us. If you can help carry that, you make this real years ahead of schedule.

---

*Q-NarwhalKnight is at branch `feature/safe-batched-sync-v1.0.2`. Current chain height ~17.8M blocks. Mainnet live since genesis. Source at `code.quillon.xyz` (mirror available on request).*
