# DeepSeek Handoff — Nova Phase 2, Job N2 (δ-circuit as StepCircuit)

**Date:** 2026-05-14
**Project:** Quillon Graph — live mainnet, ~$2 B USD market cap
**Track:** Nova Phase 2 — recursive folding wrapper
**Companion docs:**
- `docs/deepseek-job-board-nova-phase2-2026-05-14.md` (the master job board — N1 through N8)
- `papers/quillon-recursive-lattice-snark-whitepaper-v2-2026-05-13.pdf` (architecture)
- `docs/deepseek-submission-n1-draft-2026-05-14.md` (N1 draft — needs API verification before merge)

**Purpose.** Detailed handoff for **Job N2** — implementing the δ-circuit as a Nova `StepCircuit`. This is the load-bearing piece of Phase 2: it's the single R1CS predicate that proves "applying one block to the previous state produces the next state, under every consensus rule."

---

# 🚨 BEFORE YOU START READING — three rules

Past submissions have failed at the same three points. Read this first:

## Rule 1: VERIFY EVERY EXTERNAL API BEFORE WRITING CODE

The nova-snark crate's API surface changes between versions. **Multiple past submissions invented API signatures that "looked right" and submitted code that didn't compile.** Before writing a single line of N2 code:

1. `cargo doc --package nova-snark --open` — open the actual API documentation
2. Read the `StepCircuit` trait definition. Note exactly what associated types and required methods it has. Write down the signature.
3. Read the `RecursiveSNARK` struct's public methods. Note the exact argument order.
4. Read `bellpepper-core::ConstraintSystem` and `bellpepper-core::num::AllocatedNum` — the constraint allocation API.
5. Confirm what curve types are exposed (`bn256_grumpkin::bn256::Point`? `pasta::pallas::Point`? both?).

**Action item:** before submitting any code, paste back to coordination:
- The exact `StepCircuit` trait signature you targeted
- The exact `PublicParams::setup` signature
- The exact `AllocatedNum` arithmetic API (e.g., `a.add(cs.namespace(...), &b)` vs `AllocatedNum::add(cs, &a, &b)`)

If you skip this step and submit code with invented API signatures, the submission will be rejected.

## Rule 2: NO PLACEHOLDERS

The δ-circuit composes many gadgets. Some of those gadgets (notably `MerklePathGadget::enforce_membership` and `Blake3Gadget::hash_message` from companion-board Jobs A and B) **don't exist yet**. If you need one of those, **don't fake it**. Examples of fakes that will be rejected:

```rust
// BAD — invented placeholder gadget
let mut merkle = MerklePathPlaceholder::new();
merkle.set_root(prev_root); // no-op
// "We'll wire the real Merkle path later"

// BAD — silently dropping constraint
// TODO: enforce signature here once Dilithium gadget is integrated
let signature_valid = Boolean::constant(true);
```

If a primitive doesn't exist:
- Stop and surface it in coordination
- Either ask for the gadget to be added as a sub-job, or scope your N2 PR to "just the gadgets that DO exist" with a clear comment that other constraints are pending

## Rule 3: TEST FAILURE PATHS

Every constraint you add must have a corresponding test that confirms it FAILS for wrong input. A "the happy path passes" test alone is not enough — soundness regressions are silent. The acceptance list in this doc requires four tests including three negative ones; do not skip them.

---

# WHAT N2 BUILDS

You are implementing the δ-circuit defined in §3.2 of the whitepaper as a Nova `StepCircuit`. The step semantics:

**Public inputs (`z`):**
- `state_root_prev: [u8; 32]` packed into BN256-field elements (your choice: 1 element splitting hi/lo 128 bits, or 4 elements of 64 bits each — document your choice)
- `block_height: u64` (1 field element)
- `header_hash: [u8; 32]` (same packing as state root)

**Public output (`z_next`):**
- `state_root_next: [u8; 32]` (same packing)
- `block_height + 1: u64`
- `next_header_hash: [u8; 32]` — the header hash of the NEXT block if available, else zero

**Private witnesses:**
- Full block body (txs, coinbase, timestamp, signatures)
- For every wallet touched: Merkle path in `state_root_prev`, intermediate root, Merkle path in intermediate root
- For every tx: Dilithium5 public key + signature `(z, h, c̃, c_poly)` + message hash
- NTT anchor election witness

**Constraints enforced** (from whitepaper §3.2):

1. **Header hash:** `header_hash = BLAKE3(header_bytes)`
2. **For every transaction:**
   - `Verify_Dilithium5(pk, msg_tx, sig)` via `DilithiumVerifierGadget::verify_structured` (exists today)
   - Membership of `(from, b_from)` in `state_root_prev` (Merkle gadget — **may not exist yet**)
   - Membership of `(to, b_to)` in `state_root_prev`
   - Balance sufficiency: `b_from >= a + f` via `enforce_norm_bound`
   - Updated `(from, b_from - a - f)` membership in intermediate root
   - Updated `(to, b_to + a)` membership in root that follows the from-side update
3. **Coinbase emission:** `e <= R(height)` (piecewise lookup)
4. **NTT anchor election:** validate producer's anchor claim via `NttVerifierGadget`
5. **Final state-root equality:** after all txs + coinbase, resulting root must equal public `state_root_next`

---

# WHAT EXISTS TODAY (read before writing)

## In-circuit gadgets — production, `crates/q-ivc/src/gadgets/`

| Gadget | File | API entry point | Status |
|--------|------|-----------------|--------|
| BLAKE3 single-block | `blake3.rs` | `Blake3Gadget::verify_hash(cs, msg, hash)` | ✅ shipped, ~600 LOC |
| BLAKE3 multi-block | `blake3.rs` | `Blake3Gadget::hash_message(...)` | ❌ **NOT IMPLEMENTED** — companion-board Job A |
| Poseidon transcript | `poseidon.rs` | `PoseidonGadget::hash_many(cs, &[FpVar<F>])` | ✅ shipped |
| NTT butterfly + signed-norm | `ntt.rs` | `NttVerifierGadget::*` | ✅ shipped |
| Dilithium5 structured | `dilithium.rs` | `DilithiumVerifierGadget::verify_structured(cs, msg, pk, sig, roots)` | ✅ **fresh today** — hint-weight, q-range, μ-prefix gates added v10.9.20 |
| Merkle-path | `merkle.rs` | `MerklePathGadget::enforce_membership(...)` | ❌ **NOT IMPLEMENTED** — companion-board Job B |

**Critical:** N2 needs both BLAKE3 multi-block (for block headers >1 KB) and Merkle-path (for SMT membership). **Neither exists.** Options:

1. **Block N2 on A and B**: don't start until those are done.
2. **Stub the missing gadgets transparently**: write `// TODO: pending Job A` on the missing pieces and ONLY land the parts that compose existing gadgets. Submit as "N2 partial — pending A and B."
3. **Implement A/B inline**: scope creep, probably wrong.

**Recommendation:** option 2. Write the δ-circuit STRUCTURE with all five constraint groups present, but use clear `unimplemented!()` or commented-out blocks for the BLAKE3-multi-block and Merkle-path call sites. That way when A and B land, wiring is one find-and-replace per call site. Surface this scope reality in coordination before starting.

## State commitment — production, `crates/q-storage/src/balance_smt.rs`

- `BalanceSmt` — sparse Merkle tree, depth 256, BLAKE3 with explicit domain separators
- 743 LOC, twelve test cases
- Use `BalanceSmt::prove(&self, addr: &[u8; 32]) -> Result<SmtProof>` to get a real Merkle proof for test fixtures. **Do not mock proofs.**

## Block types — production, `crates/q-types/`

- `Block` struct — read its fields in `crates/q-types/src/block.rs`
- `Transaction` struct — `crates/q-types/src/transaction.rs`. Has `signing_payload()` returning the 32 bytes that get signed.
- `SignaturePhase` — `Phase0Ed25519`, `Phase1Dilithium5`, `HybridEd25519Dilithium5`, etc.

For N2: assume the chain has migrated to Phase 1 (Dilithium5 only) by the time Phase 2 activates. Don't try to support Ed25519 in-circuit — Ed25519 verification in R1CS is millions of constraints and out of scope.

## Upgrade gate — `crates/q-consensus-guard/src/upgrade_gate.rs`

- `Upgrade::HybridSignaturesV1` shipped today (dormant on mainnet)
- For Phase 2 you'll need to add `Upgrade::NovaPhase2` (same pattern). NOT YET your job — N2 doesn't activate the gate, just builds the StepCircuit.

---

# FILES YOU OWN FOR N2

- `crates/q-ivc/src/circuits/delta.rs` (new — the δ-circuit composition)
- `crates/q-ivc/src/circuits/mod.rs` (add `pub mod delta;`)
- `crates/q-ivc/src/lib.rs` (already has `pub mod circuits;`)
- `crates/q-ivc/Cargo.toml` (no new deps — reuse gadgets + nova-snark from N1)

## Files you must NOT modify

- Any file under `crates/q-ivc/src/gadgets/` — gadgets are read-only for N2
- `crates/q-types/`, `crates/q-storage/`, `crates/q-network/`, `crates/q-api-server/` — out of scope

If you need a primitive that doesn't exist in a gadget, **raise it in coordination**. Don't patch sideways into gadgets/.

---

# PUBLIC API

```rust
use ark_bls12_381::Fr;
use ark_r1cs_std::fields::fp::FpVar;
use nova_snark::traits::circuit::StepCircuit;

pub struct DeltaCircuit {
    // Witness fields — populated by the prover driver (N3)
    pub block: Block,
    pub state_root_prev: [u8; 32],
    pub state_root_next: [u8; 32],
    pub block_height: u64,
    pub header_hash: [u8; 32],
    pub next_header_hash: [u8; 32],
    pub merkle_proofs: Vec<SmtProof>,           // one per touched wallet
    pub intermediate_roots: Vec<[u8; 32]>,      // intermediate roots between txs
    pub ntt_anchor_witness: AnchorWitness,
}

impl StepCircuit<Fr> for DeltaCircuit {
    fn arity(&self) -> usize {
        // 4 elements for state_root_prev + 1 for height + 4 for header_hash = 9
        // (Adjust to your packing scheme — document the choice)
        9
    }

    fn synthesize<CS: ConstraintSystem<Fr>>(
        &self,
        cs: &mut CS,
        z: &[AllocatedNum<Fr>],
    ) -> Result<Vec<AllocatedNum<Fr>>, SynthesisError> {
        // ... five constraint groups
    }

    fn output(&self, z: &[Fr]) -> Vec<Fr> {
        // Mirror of synthesize, in native field arithmetic
    }
}
```

**NOTE:** the actual `StepCircuit` trait signature depends on the nova-snark version. **Verify it before writing.** The above is illustrative — your real implementation may have different associated types and method signatures.

---

# ACCEPTANCE CRITERIA (PR description checklist)

- [ ] `crates/q-ivc/src/circuits/delta.rs` defines `DeltaCircuit` struct and `impl StepCircuit<Fr> for DeltaCircuit`
- [ ] `synthesize` body enforces all five constraint groups using existing gadgets — **no inline cryptography**, no copy-paste of gadget bodies
- [ ] Missing gadgets (BLAKE3 multi-block, Merkle-path) are clearly marked `unimplemented!("pending Job A/B")` if you go the partial route — surface this in coordination
- [ ] Test `test_delta_one_transaction_satisfiable`: build a one-transaction block, generate witnesses, call `cs.is_satisfied()`, assert true
- [ ] Test `test_delta_invalid_signature_rejected`: flip one bit in tx signature, assert unsatisfied
- [ ] Test `test_delta_double_spend_rejected`: two txs from same sender exceeding balance, assert unsatisfied
- [ ] Test `test_delta_wrong_state_root_rejected`: modify public `state_root_next`, assert unsatisfied
- [ ] Constraint count reported in test output for the satisfiable case. Expected order of magnitude for one-tx: 50K-500K constraints
- [ ] PR description includes:
  - Exact nova-snark crate version pinned
  - Verified `StepCircuit` API signature
  - Which constraint groups are real vs. `unimplemented!()`
  - Constraint counts per test case

---

# COMMON FAILURE MODES (avoid)

## "I'll write the StepCircuit even though the API may be wrong"

Don't. Verify first, write second.

## "I'll mock the missing gadgets"

Don't. Use `unimplemented!()` with a job-pointer message, or block your PR on Jobs A/B.

## "The tests passing is good enough"

Tests must include FAILURE paths. A `cs.is_satisfied() == true` test alone proves only that you didn't add a contradictory constraint, not that you added the right constraints.

## "I'll skip the negative tests"

The acceptance criteria require three negative tests (bad signature, double-spend, wrong state root). Skipping them blocks merge.

## "The constraint count is too high, I'll optimize later"

Constraint count for a real one-tx δ-circuit is going to be high (Dilithium5 alone is ~13K constraints, BLAKE3 of a 1KB header is ~80K, Merkle paths are ~5K each). Don't try to "optimize" by removing gadget calls. The math has to be done; the only question is how to compose it.

## "Field packing is just bytes — I'll figure it out"

Decide your packing scheme up front and stick to it. `[u8; 32]` → field elements has multiple valid encodings (4 × u64 little-endian, 2 × u128, etc.). Different choices cascade through `arity()`, `synthesize`, and `output`. Pick one, document it at the top of the file, use it everywhere.

---

# VERIFICATION

```
cd /opt/orobit/shared/q-narwhalknight
timeout 600 cargo check --package q-ivc 2>&1 | tail -15
timeout 600 cargo test --package q-ivc --lib circuits::delta 2>&1 | tail -30
```

Both must succeed. All four tests must pass (or be clearly marked `#[ignore = "pending Job X"]` with a coordination ticket — but no silent skips).

---

# COORDINATION

- **Channel:** `#dev-coordination` on Discord
- **Status updates:** post when you start, when you hit a blocker, when you're ready for review
- **Block on missing primitives early.** If Job A (multi-block BLAKE3) or Job B (Merkle path) isn't done and you can't proceed, say so DAY ONE — don't wait until day five
- **Push partial work.** Branches don't have to be PR-ready to be visible
- **Post a one-page summary at the end** to `docs/nova-phase2-postmortems/n2.md`

---

# IF YOU'RE READING THIS COLD

If you're a fresh agent (human or LLM) picking up N2 with no context:

1. Read this entire doc first
2. Read `docs/deepseek-job-board-nova-phase2-2026-05-14.md` for the full Phase 2 picture (jobs N1 through N8)
3. Read the whitepaper §3.2 (δ-circuit definition)
4. Read `crates/q-ivc/src/gadgets/dilithium.rs::verify_structured` (the most complex gadget you'll compose)
5. Confirm N1's nova-snark dep is actually in place and verified (check `docs/deepseek-submission-n1-draft-2026-05-14.md` for the draft status)
6. Then start. Verify API, surface missing gadgets, write code, test failure paths.

Good luck. Don't ship placeholders.
