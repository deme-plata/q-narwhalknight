# DeepSeek Handoff: Blueprint 1B (Merkle Gadget) + Blueprint 2 (δ-Circuit)

**Date:** 2026-05-13
**Project:** Q-NarwhalKnight (Quillon Graph), codename NarwhalKnight
**Production status:** **LIVE MAINNET · ~$2 BILLION USD MARKET CAP · GENESIS NODES SERVING USERS**

---

# 🚨 READ THIS BEFORE WRITING A SINGLE LINE OF CODE 🚨

You are extending a **live, production blockchain** with a **$2 B market cap**. Every code change has financial blast-radius for real users with real money. The rules below are non-negotiable.

## Constraint 1 — NO DATABASE CHANGES

The user has explicitly stated: **no DB changes, only new keys**.

The previous PR (Blueprint 1A, already merged into `crates/q-storage/src/balance_smt.rs`) added a **new column family** `cf_balance_smt`. That is the **only** schema addition allowed in this whole feature track. It runs in **shadow mode**: data is written into it but never read by consensus, validation, or balance lookup.

Concretely this means:

- **Do not modify any existing column family.** The full list of touched-by-other-code CFs is in `crates/q-storage/src/lib.rs` lines 419–470. Do not even *read* from `CF_BALANCES`, `CF_WALLET_BALANCES`, `CF_BLOCKS`, `CF_STATE_ROOTS`, `CF_TRANSACTIONS`, or anything else listed there.
- **Do not modify `crates/q-storage/src/balance_smt.rs`.** It shipped clean. Do not extend it. If you find a bug, file a comment in the doc, do not patch the file.
- **Do not modify `save_wallet_balance`, `save_wallet_balances`, `save_wallet_balances_batch`**, or any function that writes to a balance CF. The integration of the SMT into `save_wallet_balances` is a **separate PR by a Beta engineer**, not your work.
- **Do not modify `balance_root_v1`** computation in any way. It is the consensus-critical root that the live $2 B chain depends on. `balance_root_v2` (the SMT) is shadow-only until activation height ~2 months from now.
- **Do not introduce a new RocksDB schema migration.** No `delete_range`, no compaction-filter, no MANIFEST mutation, no `set_options`.

## Constraint 2 — NO API / CONSENSUS / NETWORK CHANGES IN THIS PR

The work in this handoff is **pure arithmetic-circuit synthesis** in `crates/q-ivc/`. Your code:

- ✗ does NOT run during block production
- ✗ does NOT run during block sync
- ✗ does NOT run inside `validate_block` or `accept_block`
- ✗ does NOT bind to any gossipsub topic
- ✗ does NOT add any HTTP endpoint
- ✓ DOES compile into a library that future provers/verifiers will link against
- ✓ DOES have unit tests that synthesize circuits in-memory only
- ✓ DOES consume the deterministic `SmtProof` output of `BalanceSmt::prove()` as test fixtures

Files you may **create**:
- `crates/q-ivc/src/gadgets/merkle.rs` ← Blueprint 1B
- `crates/q-ivc/src/circuits/delta_block.rs` ← Blueprint 2

Files you may **edit** (one-line module declarations only):
- `crates/q-ivc/src/gadgets/mod.rs` — add `pub mod merkle;`
- `crates/q-ivc/src/circuits/mod.rs` — add `pub mod delta_block;`
- `crates/q-ivc/Cargo.toml` — ONLY if a new dev-dependency is strictly required (e.g., `q-storage` as dev-dep for fixture generation). Production deps unchanged.

Files you must **NOT TOUCH** under any circumstances:
- Anything under `crates/q-storage/`, `crates/q-api-server/`, `crates/q-network/`, `crates/q-types/`, `crates/q-consensus-guard/`
- The top-level `Cargo.toml` (no version bumps)
- `crates/q-ivc/src/gadgets/blake3.rs`, `poseidon.rs`, `ntt.rs`, `dilithium.rs` — already shipped, do not modify
- `crates/q-ivc/src/circuits/epoch_transition.rs` — existing skeleton; do not replace

## Constraint 3 — CARGO HYGIENE

- Run `timeout 600 cargo check --package q-ivc` after every batch of edits. The repo's q-ivc crate must always compile cleanly.
- Run `timeout 1800 cargo test --package q-ivc` before declaring done. All tests including yours must pass.
- Do not add `unwrap()` or `panic!()` in non-test code. Use `Result` / `SynthesisError` everywhere.
- No `unsafe`. No FFI. No external process calls.

## Constraint 4 — TEST DISCIPLINE

For every gadget function and every circuit, your PR must include:

1. **Happy-path test** — generate a valid witness, synthesize, assert circuit is satisfied.
2. **At least 3 adversarial tests** — each one introduces a specific corruption (wrong sibling, wrong leaf, wrong root, off-by-one bit). Each must assert the circuit becomes UNsatisfied.
3. **Cross-check test** — compute the expected hash/root using the production code paths (`BalanceSmt::prove()`, `blake3::hash()`, etc.) and assert the circuit produces the same value.

If a single test is missing for a function you wrote, the PR is rejected.

## Constraint 5 — SOAK BEFORE INTEGRATION

This is the part that's easy to forget. Even after your PR lands clean:

- Your code is **not** activated in production
- Your code's only consumer is `cargo test`
- The δ-circuit's first production use is months away (Phase 2 in the V2 plan)
- We will run the test suite on Beta and Epsilon for ≥ 1 week before considering this code "trusted" for the prover wiring

This is not a slight on your work — it is how a $2 B chain operates. No code goes near consensus paths without weeks of dwell-time.

---

# Reference Material You Should Already Have Read

Before starting, ensure you have read:

1. `docs/technical-plan-instant-bootstrap-recursive-snark-v2-2026-05-13.md` — the why
2. `docs/blueprints-ivc-snark-2026-05-13.md` — the architecture
3. `crates/q-storage/src/balance_smt.rs` — the SMT (shipped, 600 LOC)
4. `crates/q-ivc/src/gadgets/blake3.rs` — the BLAKE3 gadget you will compose against
5. `crates/q-ivc/src/gadgets/dilithium.rs` — the Dilithium5 verifier gadget you will compose against

If any of the existing gadget interfaces don't match what's documented below, **adapt your code to match the gadget, not the other way around**.

---

# Blueprint 1B — Merkle-Path Gadget

## File

`crates/q-ivc/src/gadgets/merkle.rs` (new, ~600 LOC)

## Constraint System

`F: PrimeField` parametric — concrete instantiation will be BN254 `Fr` for Nova (Phase 2), lattice modulus for Phase 4. Your gadget must work with either.

## Required Types from Existing Gadgets

Inspect `crates/q-ivc/src/gadgets/blake3.rs` for the exact signature. The pattern you should rely on is:

```rust
// Hash an arbitrary-length byte message to a 32-byte digest (8 × UInt32).
// You will need to add this if it doesn't exist (it is small — call the existing
// compress() in a chain). Discuss with Beta before writing it; existing primitives
// may suffice via verify_hash + alloc_hash.
pub fn hash_message<F: PrimeField>(
    cs: ConstraintSystemRef<F>,
    msg: &[UInt8<F>],
) -> Result<Vec<UInt32<F>>, SynthesisError>;
```

If `hash_message` doesn't exist as a standalone helper, you have two options:

**Option A (preferred):** add it as a *private* function within `gadgets/blake3.rs`, then call it from `merkle.rs`. This requires one edit to `blake3.rs` — flag it explicitly in your PR description so reviewers know to scrutinize.

**Option B:** inline the BLAKE3 compress loop inside `merkle.rs`. Repeats logic; rejected unless Option A is blocked by a constraint we discover.

## Public API (must match exactly)

```rust
use ark_ff::PrimeField;
use ark_r1cs_std::{
    boolean::Boolean,
    eq::EqGadget,
    fields::fp::FpVar,
    select::CondSelectGadget,
    uint8::UInt8,
    uint32::UInt32,
    ToBitsGadget,
};
use ark_relations::r1cs::{ConstraintSystemRef, SynthesisError};

pub struct MerklePathGadget;

impl MerklePathGadget {
    /// Compute leaf hash inside the circuit.
    /// MUST produce the same bytes as `crates/q-storage/src/balance_smt.rs::leaf_hash_raw`.
    /// Domain: BLAKE3("smt_leaf_v2" || addr[32] || balance.to_le_bytes()[16]).
    pub fn leaf_hash<F: PrimeField>(
        cs: ConstraintSystemRef<F>,
        addr_bits: &[Boolean<F>; 256],
        balance: &FpVar<F>,
    ) -> Result<Vec<UInt32<F>>, SynthesisError>;

    /// Compute root from leaf, addr, siblings, empty bitmap, and empty constants.
    /// `addr_bits[i]` is the MSB-first bit at position i. `addr_bits[0]` is the
    /// most significant bit of the address byte 0; this matches
    /// `balance_smt.rs::addr_bit(addr, i)`.
    ///
    /// `siblings[d]` is the sibling at depth d+1 reached when descending past depth d.
    /// `empty_bitmap[d]` is `true` iff the sibling at depth d+1 is the empty subtree
    /// hash at depth d+1 — in which case use `empty_subtree_hashes[d+1]` instead.
    pub fn compute_root<F: PrimeField>(
        cs: ConstraintSystemRef<F>,
        leaf_hash: &[UInt32<F>],
        addr_bits: &[Boolean<F>; 256],
        siblings: &[Vec<UInt32<F>>; 256],
        empty_bitmap: &[Boolean<F>; 256],
        empty_subtree_hashes: &[Vec<UInt32<F>>; 257],
    ) -> Result<Vec<UInt32<F>>, SynthesisError>;

    /// Enforce `compute_root(...) == expected_root` as a circuit constraint.
    pub fn enforce_membership<F: PrimeField>(
        cs: ConstraintSystemRef<F>,
        addr_bits: &[Boolean<F>; 256],
        balance: &FpVar<F>,
        siblings: &[Vec<UInt32<F>>; 256],
        empty_bitmap: &[Boolean<F>; 256],
        empty_subtree_hashes: &[Vec<UInt32<F>>; 257],
        expected_root: &[UInt32<F>],
    ) -> Result<(), SynthesisError>;
}
```

## Constraint Generation Order (must be exactly this)

For each depth `d` from 0 (root-side) to 255 (just above the leaf):

You iterate the **leaf-to-root** direction in the circuit. The fold loop in `BalanceSmt::fold_to_root` (the production code) goes leaf-to-root with index variable `d_from_leaf` running 0..256, where:

```rust
let depth = SMT_DEPTH - 1 - d_from_leaf;   // 255, 254, ..., 0
let bit = addr_bit(&addr, depth);
```

Your circuit **must produce the exact same hash bytes** as the production code. So your loop is:

```rust
let mut current = leaf_hash;
for d_from_leaf in 0..256 {
    let depth = 255 - d_from_leaf;

    // Choose sibling: empty constant or supplied sibling
    let sibling = CondSelectGadget::conditionally_select(
        &empty_bitmap[depth],
        &empty_subtree_hashes[depth + 1],
        &siblings[depth],
    )?;

    // Bit decides ordering: if bit==0 (path goes left), current is left.
    // If bit==1 (path goes right), current is right.
    // The production code: `if addr_bit(addr, depth) { (sibling, current) } else { (current, sibling) }`
    let bit = &addr_bits[depth];

    // The CondSelectGadget over a tuple is the standard arkworks pattern:
    let left = bit.select(&sibling, &current)?;     // bit==1 -> left=sibling
    let right = bit.select(&current, &sibling)?;    // bit==1 -> right=current

    // Compose message: "smt_node_v2" || left || right, then BLAKE3.
    let mut msg = Vec::new();
    for &b in b"smt_node_v2".iter() {
        msg.push(UInt8::constant(b));
    }
    msg.extend(uint32_vec_to_be_bytes(&left));
    msg.extend(uint32_vec_to_be_bytes(&right));

    current = Blake3Gadget::hash_message(cs.clone(), &msg)?;
}
// `current` is now the root.
```

**Byte-ordering inside UInt32:** the existing `Blake3Gadget` operates on 32-bit words. When you serialize a node's hash (32 bytes) into the next BLAKE3 input, you must use the same byte order BLAKE3 uses internally — which is **little-endian for the state, but the message bytes are flat**. Confirm by checking what `Blake3Gadget::alloc_hash` and `Blake3Gadget::compress` expect for input byte ordering. If unclear, write a tiny test outside the circuit using `blake3::hash(b"smt_node_v2" || left || right)` where `left` and `right` are obtained from `node_hash_raw` (production code) and confirm your gadget produces the same output.

## Fixture Generation for Tests

Add `q-storage` as a `[dev-dependencies]` entry in `crates/q-ivc/Cargo.toml` so your tests can call `BalanceSmt::prove()` to generate real proofs. This is the *only* file in q-storage that's allowed to be referenced; the gadget itself does not depend on q-storage.

```toml
[dev-dependencies]
q-storage = { path = "../q-storage" }
tempfile = "3.8"
```

Test pattern:

```rust
#[test]
fn merkle_gadget_accepts_valid_proof() {
    // 1. Set up production SMT
    let (db, _tmp) = open_test_db();
    let smt = BalanceSmt::open(db).unwrap();
    let addr = [0x42u8; 32];
    let balance = 12345u128;
    let root = smt.update_batch(&[(addr, balance)]).unwrap();
    let proof = smt.prove(&addr, balance).unwrap();

    // 2. Synthesize circuit with proof as witness, root as public input
    let cs = ConstraintSystem::<Fr>::new_ref();
    let addr_bits = bytes_to_msb_bool_array(&cs, &proof.addr);
    let balance_var = FpVar::new_witness(cs.clone(), || Ok(Fr::from(balance)))?;
    let sibling_vars = proof.siblings.map(|s| bytes_to_uint32_vec(&cs, &s));
    let empty_bm_bools = bitmap_to_bool_array(&cs, &proof.empty_bitmap);
    let empty_const_vars = compute_empty_subtree_hashes_as_circuit_constants(&cs);
    let root_var = bytes_to_uint32_vec(&cs, &root);

    MerklePathGadget::enforce_membership(
        cs.clone(),
        &addr_bits,
        &balance_var,
        &sibling_vars,
        &empty_bm_bools,
        &empty_const_vars,
        &root_var,
    ).unwrap();

    assert!(cs.is_satisfied().unwrap());
}
```

## Required Tests (minimum)

| # | Name | What it asserts |
|---|------|-----------------|
| 1 | `merkle_gadget_accepts_valid_proof` | Happy path with single-leaf tree |
| 2 | `merkle_gadget_accepts_dense_tree_proof` | Happy path with 100 inserts |
| 3 | `merkle_gadget_rejects_wrong_root` | Same proof, flipped root byte → UNSAT |
| 4 | `merkle_gadget_rejects_tampered_sibling` | Real sibling byte-flip → UNSAT |
| 5 | `merkle_gadget_rejects_wrong_balance` | balance off-by-one → UNSAT |
| 6 | `merkle_gadget_rejects_wrong_addr_bit` | One addr bit flipped → UNSAT |
| 7 | `merkle_gadget_rejects_empty_bitmap_lie` | `empty_bitmap[d]=true` but sibling != empty → UNSAT |
| 8 | `merkle_gadget_circuit_size_reasonable` | Print constraint count; assert < 1M (sanity) |

## Constraint Cost Target

The blueprint estimates ~590,000 constraints per single path. If your implementation produces > 1.5 M constraints, something is wrong — either your BLAKE3 compose loop has a bug, or you're re-allocating witnesses inside the loop. Compare against the existing `blake3.rs::verify_hash` cost per invocation.

---

# Blueprint 2 — δ-Circuit Composition

## File

`crates/q-ivc/src/circuits/delta_block.rs` (new, ~1500 LOC)

## What This Circuit Proves

```
PUBLIC INPUT:  (state_root_prev, state_root_next, block_header_hash, block_height)
WITNESS:       block bytes + Merkle paths + signatures + anchor witness
CLAIM:         "The state transition from state_root_prev to state_root_next,
                effected by the block whose hash is block_header_hash and
                whose height is block_height, is valid according to QNK rules."
```

## Production Block Header (for header hash gadget input)

From `crates/q-types/src/block.rs::BlockHeader` (read it directly for the full list):

- `height: u64`
- `phase: u8`
- `network_id: String`
- `prev_block_hash: BlockHash` (32 bytes)
- `solutions_root: BlockHash`
- `tx_root: BlockHash`
- `state_root: BlockHash` ← THIS IS `state_root_next` in your circuit
- `timestamp: u64`
- `dag_round: u64`
- `vdf_proof: VDFProof` (variable-length bytes)
- `anchor_validator: Option<String>`
- `proposer: NodeId`
- `producer_id: u8`
- + a few more fields (read the file)

For Blueprint 2 you do **NOT** need to enforce the full header structure inside the circuit. You only need:
1. Witness the header bytes (whatever serialization the production code uses, almost certainly `postcard` or `bincode`)
2. `Blake3Gadget::verify_hash(header_bytes_var, public_block_header_hash_var)`
3. Extract just the fields you care about from the witnessed bytes (`state_root`, `height`) — assert they match the corresponding public inputs

This avoids re-implementing the header layout in R1CS. The hash check pins the entire header content; field extraction is a single-byte-range constraint per field.

## Public API

```rust
pub struct DeltaBlockCircuit<F: PrimeField> {
    // Public inputs (Some(value) when proving, None when keygen)
    pub state_root_prev: Option<[u8; 32]>,
    pub state_root_next: Option<[u8; 32]>,
    pub block_header_hash: Option<[u8; 32]>,
    pub block_height: Option<u64>,

    // Private witness
    pub block_header_bytes: Option<Vec<u8>>,
    pub transactions: Option<Vec<TxWitness>>,
    pub coinbase: Option<CoinbaseWitness>,

    // Precomputed at compile time. Passed as &Vec<[u8;32]> for cheap clone.
    pub empty_subtree_hashes: Vec<[u8; 32]>,  // length 257
}

pub struct TxWitness {
    pub from_addr: [u8; 32],
    pub to_addr: [u8; 32],
    pub amount: u128,
    pub fee: u128,
    pub nonce: u64,
    pub from_pk: Vec<u8>,          // Dilithium5 pubkey
    pub sig: Vec<u8>,              // Dilithium5 signature
    pub signed_msg: Vec<u8>,       // canonical signed bytes
    // SMT paths against state_root_prev
    pub from_path_prev: SmtPathBytes,
    pub to_path_prev: SmtPathBytes,
    // Updated balances (from_bal_new = from_bal_old - amt - fee, to_bal_new = to_bal_old + amt)
    pub from_balance_prev: u128,
    pub to_balance_prev: u128,
}

pub struct CoinbaseWitness {
    pub miner_addr: [u8; 32],
    pub emission: u128,
    pub scheduled_emission: u128,  // looked up from height-indexed table at compile time
    pub miner_balance_prev: u128,
    pub miner_path_prev: SmtPathBytes,
}

/// Native (non-circuit) struct mirroring what BalanceSmt::prove returns.
pub struct SmtPathBytes {
    pub siblings: [[u8; 32]; 256],
    pub empty_bitmap: [u8; 32],
}

impl<F: PrimeField> ConstraintSynthesizer<F> for DeltaBlockCircuit<F> {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        // ... see below
    }
}
```

## Constraint Generation (exactly this order, exactly these steps)

```rust
fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
    // ─── 1. Public inputs ───────────────────────────────────────────
    let state_root_prev_var = alloc_hash_public(cs.clone(), self.state_root_prev)?;
    let state_root_next_var = alloc_hash_public(cs.clone(), self.state_root_next)?;
    let block_header_hash_var = alloc_hash_public(cs.clone(), self.block_header_hash)?;
    let block_height_var = FpVar::new_input(cs.clone(), || {
        self.block_height.map(Fr::from).ok_or(SynthesisError::AssignmentMissing)
    })?;

    // Empty subtree hashes — circuit constants (no witness cost)
    let empty_const: Vec<Vec<UInt32<F>>> = self.empty_subtree_hashes.iter()
        .map(|h| bytes_to_uint32_constants(&cs, h)).collect();

    // ─── 2. Block header bytes → hash matches public ────────────────
    let header_bytes_var = alloc_witness_bytes(cs.clone(), &self.block_header_bytes)?;
    Blake3Gadget::verify_hash(&header_bytes_var, &block_header_hash_var)?;

    // Extract state_root and height from header bytes (single byte-range constraint)
    // and assert they match the public inputs we already allocated.
    let header_state_root = extract_bytes(&header_bytes_var, OFFSET_STATE_ROOT, 32)?;
    let header_height = extract_u64(&header_bytes_var, OFFSET_HEIGHT)?;
    enforce_equal_hash(&header_state_root, &state_root_next_var)?;
    header_height.enforce_equal(&block_height_var)?;
    // ↑ This pins block_header_hash to (state_root_next, block_height) at hash-of-bytes level.

    // ─── 3. Loop over transactions ──────────────────────────────────
    let txs = self.transactions.unwrap_or_default();
    let mut current_root = state_root_prev_var.clone();

    for tx in &txs {
        // 3a. Witnesses
        let from_addr_bits = bytes_to_msb_bool_array(&cs, &tx.from_addr)?;
        let to_addr_bits = bytes_to_msb_bool_array(&cs, &tx.to_addr)?;
        let amount_var = FpVar::new_witness(cs.clone(), || Ok(Fr::from(tx.amount)))?;
        let fee_var = FpVar::new_witness(cs.clone(), || Ok(Fr::from(tx.fee)))?;
        let from_bal_old_var = FpVar::new_witness(cs.clone(), || Ok(Fr::from(tx.from_balance_prev)))?;
        let to_bal_old_var = FpVar::new_witness(cs.clone(), || Ok(Fr::from(tx.to_balance_prev)))?;

        // 3b. Range checks (use enforce_norm_bound from dilithium.rs)
        enforce_norm_bound(cs.clone(), &amount_var, 128)?;
        enforce_norm_bound(cs.clone(), &fee_var, 64)?;
        enforce_norm_bound(cs.clone(), &from_bal_old_var, 128)?;
        enforce_norm_bound(cs.clone(), &to_bal_old_var, 128)?;

        // 3c. Signature verifies (Dilithium5)
        let pk_var = alloc_witness_bytes(cs.clone(), &tx.from_pk)?;
        let sig_var = alloc_witness_bytes(cs.clone(), &tx.sig)?;
        let msg_var = alloc_witness_bytes(cs.clone(), &tx.signed_msg)?;
        DilithiumVerifierGadget::verify(&pk_var, &msg_var, &sig_var)?;

        // 3d. Bind signed_msg to (from_addr, to_addr, amount, fee, nonce) canonically.
        // The signed_msg bytes must contain these fields; assert equality of the bytes.
        // (Production canonical encoding is in q-types::transaction::canonical_signing_bytes —
        // mirror it in the circuit via byte-extract.)
        assert_canonical_signed_msg(&msg_var, &tx.from_addr, &tx.to_addr,
                                     &amount_var, &fee_var, tx.nonce)?;

        // 3e. Merkle: prev-state contains (from_addr, from_bal_old_var)
        let from_siblings_var = alloc_sibling_array(&cs, &tx.from_path_prev.siblings)?;
        let from_empty_bm_var = alloc_empty_bitmap(&cs, &tx.from_path_prev.empty_bitmap)?;
        MerklePathGadget::enforce_membership(
            cs.clone(),
            &from_addr_bits,
            &from_bal_old_var,
            &from_siblings_var,
            &from_empty_bm_var,
            &empty_const,
            &current_root,
        )?;

        // 3f. Merkle: prev-state contains (to_addr, to_bal_old_var)
        let to_siblings_var = alloc_sibling_array(&cs, &tx.to_path_prev.siblings)?;
        let to_empty_bm_var = alloc_empty_bitmap(&cs, &tx.to_path_prev.empty_bitmap)?;
        MerklePathGadget::enforce_membership(
            cs.clone(),
            &to_addr_bits,
            &to_bal_old_var,
            &to_siblings_var,
            &to_empty_bm_var,
            &empty_const,
            &current_root,
        )?;

        // 3g. Sufficient balance: from_bal_new = from_bal_old - amount - fee, ≥ 0
        let from_bal_new_var = &from_bal_old_var - &amount_var - &fee_var;
        enforce_norm_bound(cs.clone(), &from_bal_new_var, 128)?;

        // 3h. Credit: to_bal_new = to_bal_old + amount
        let to_bal_new_var = &to_bal_old_var + &amount_var;
        enforce_norm_bound(cs.clone(), &to_bal_new_var, 128)?;

        // 3i. The from-side update produces an intermediate root.
        // We compute it inside the circuit by re-using the SAME sibling array
        // (siblings don't change between prev and next when only this leaf updates).
        let intermediate_root = MerklePathGadget::compute_root_from_leaf(
            cs.clone(),
            &from_addr_bits,
            &from_bal_new_var,        // updated balance for from
            &from_siblings_var,
            &from_empty_bm_var,
            &empty_const,
        )?;

        // 3j. The to-side update against intermediate_root yields the next current_root.
        // BUT — siblings of `to` may have changed because the `from`-update modified
        // nodes on the path. We use a SEPARATE witness `to_path_intermediate` here.
        // This is a subtle point: the production SMT's pending map handles this; the
        // circuit must witness BOTH path versions explicitly.
        //
        // For Phase 1 of the implementation, assume from_addr and to_addr have
        // path prefixes that diverge early (depth < 32), making the to-side siblings
        // unchanged. Add an assertion that the first 32 bits of from_addr ≠ to_addr.
        // (This excludes self-transfers and prefix-collision edge cases — they will
        // be added in Phase 1.5; flag them clearly as TODO.)

        let to_addr_first_32_bits = &to_addr_bits[..32];
        let from_addr_first_32_bits = &from_addr_bits[..32];
        enforce_not_equal_bits(to_addr_first_32_bits, from_addr_first_32_bits)?;

        let next_root_after_to = MerklePathGadget::compute_root_from_leaf(
            cs.clone(),
            &to_addr_bits,
            &to_bal_new_var,
            &to_siblings_var,
            &to_empty_bm_var,
            &empty_const,
        )?;

        // 3k. Combine: intermediate_root (only from-side) + to-side update => next root.
        // For Phase 1 (32-bit prefix divergence): both updates fold into the root
        // independently; we take the to-side result modulo from-side update.
        //
        // SIMPLIFIED: for Phase 1 we use the next-state SMT witness directly. The δ-circuit
        // must witness the FINAL state after BOTH txs apply, not compute it incrementally
        // from prev. We accept a witness `tx.from_path_next_siblings` and
        // `tx.to_path_next_siblings` from BalanceSmt::prove() against state_root_next, and
        // enforce both leaves are present in state_root_next with the updated balances.
        //
        // SEE BLUEPRINT 1B above — we use TWO Merkle membership proofs per leaf-update:
        // one against state_root_prev (with old balance), one against state_root_next
        // (with new balance). The δ-circuit reduces to a series of these checks.
        current_root = next_root_after_to;
    }

    // ─── 4. Coinbase ────────────────────────────────────────────────
    let coinbase = self.coinbase.expect("coinbase required");
    let cb_addr_bits = bytes_to_msb_bool_array(&cs, &coinbase.miner_addr)?;
    let cb_emission_var = FpVar::new_witness(cs.clone(), || Ok(Fr::from(coinbase.emission)))?;
    let cb_scheduled_var = FpVar::new_witness(cs.clone(), || Ok(Fr::from(coinbase.scheduled_emission)))?;
    // emission <= scheduled
    let diff = &cb_scheduled_var - &cb_emission_var;
    enforce_norm_bound(cs.clone(), &diff, 128)?;
    // Apply coinbase to miner balance (similar to tx steps)
    // ...

    // ─── 5. Final equality: computed root == state_root_next ────────
    for (a, b) in current_root.iter().zip(state_root_next_var.iter()) {
        a.enforce_equal(b)?;
    }

    Ok(())
}
```

## Simplifications for Initial Implementation

To keep the first PR shippable, **explicitly defer these** with `// PHASE 1.5: TODO` markers:

1. **Self-transfers** (`from_addr == to_addr`): asserted away via the prefix-divergence check.
2. **Prefix-collision pairs** (`from_addr` and `to_addr` share first 32 bits): asserted away.
3. **VDF proof verification** inside the circuit: the anchor election witness check is a separate gadget (`NttVerifierGadget`); for now hash the VDF proof bytes into the block header and verify the header hash matches — that's enough to commit to the VDF without re-verifying it in-circuit.
4. **Smart contract / DEX / token operations**: out of scope for the δ-circuit's first version. Only QUG (native) transfers.

Each deferral must be a clearly-labeled `// PHASE 1.5: TODO` comment with one-sentence justification.

## Required Tests (minimum)

| # | Name | What it asserts |
|---|------|-----------------|
| 1 | `delta_circuit_empty_block_satisfies` | Block with only coinbase, no txs → SAT |
| 2 | `delta_circuit_single_transfer_satisfies` | One A→B transfer, valid → SAT |
| 3 | `delta_circuit_5_transfers_satisfies` | 5 random transfers, all valid → SAT |
| 4 | `delta_circuit_rejects_wrong_state_root_next` | Flip 1 byte of state_root_next → UNSAT |
| 5 | `delta_circuit_rejects_insufficient_balance` | amt > balance → UNSAT |
| 6 | `delta_circuit_rejects_bad_signature` | Flip signature bit → UNSAT |
| 7 | `delta_circuit_rejects_unmatched_header_hash` | Flip header bytes → UNSAT |
| 8 | `delta_circuit_rejects_emission_exceeding_schedule` | coinbase > scheduled → UNSAT |
| 9 | `delta_circuit_constraint_count_sanity` | Print constraint count for 1, 5, 10 txs; assert < 1B (sanity) |

## Constraint Cost Target

The blueprint estimates ~440 M constraints per block at 100 transactions. For your Phase 1 implementation with 5–10 transactions, expect 25 M to 50 M constraints. Above 100 M for 5 transactions is a red flag.

---

# Acceptance Criteria (PR Review Checklist)

The reviewer will check, in order:

- [ ] **No file outside `crates/q-ivc/` is modified** (except possibly `Cargo.toml` dev-deps)
- [ ] **No existing column family is referenced** in any code path
- [ ] **`balance_root_v1` and `save_wallet_balances` are not touched**
- [ ] **`crates/q-storage/src/balance_smt.rs` is not modified**
- [ ] **`cargo check --package q-ivc` passes clean**
- [ ] **`cargo test --package q-ivc` passes all tests**
- [ ] **Each public function has ≥ 1 happy-path + ≥ 3 adversarial tests**
- [ ] **Each public function has 1 cross-check test against production code path**
- [ ] **No `unwrap()` or `panic!()` in non-test code**
- [ ] **No `unsafe`**
- [ ] **Constraint counts within blueprint targets**
- [ ] **All `PHASE 1.5: TODO` deferrals clearly commented with 1-sentence rationale**

A single failed checkbox blocks merge.

---

# How to Submit Your Work

When ready:

1. Push your branch to `code.quillon.xyz` (the project's local git server, not GitHub)
2. Server Beta will pull, review, and run the test suite on Beta + Epsilon
3. **Do not deploy to any production server.** The work stays inside the q-ivc crate, which is not used by the running `q-api-server` binary. Soak happens via test suite + future prover wiring, not by replacing live binaries.

If you have any uncertainty about scope — *any* — ask before coding. The cost of a clarifying question is zero. The cost of a misplaced edit on a $2 B chain is enormous.

— Server Beta, 2026-05-13
