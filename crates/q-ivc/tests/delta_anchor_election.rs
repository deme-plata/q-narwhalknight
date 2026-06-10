//! δ-circuit anchor-election adversarial tests.
//!
//! Covers Phase A's anchor binding shipped in
//! `crates/q-ivc/src/host/anchor_witness.rs` against the full δ-circuit.
//! Two regimes:
//!
//! 1. **Disabled-binding sentinel (`public_commitment = F::zero()`).**
//!    Polynomial allocation + infinity-norm bound fires but the producer-id
//!    binding is off. Backward-compat for pre-Phase-A test fixtures.
//!
//! 2. **Active binding (`public_commitment` honestly computed).** In-circuit
//!    Poseidon over `(coeffs[..K] || claimed_producer_id || round)`
//!    enforced equal to the public commitment. Substituting any of the
//!    three values without re-running Poseidon over a matching preimage
//!    off-chain fails the equality check.

use ark_bls12_381::Fr;
use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystem};

use q_ivc::circuits::{
    AnchorWitness, CoinbaseWitness, DeltaBlockCircuit, DeltaBlockInputs,
};
use q_ivc::gadgets::merkle::{SMT_DEPTH, precompute_empty_subtree_hashes};
use q_ivc::host::compute_anchor_commitment_native;

fn genesis_root() -> [u8; 32] {
    precompute_empty_subtree_hashes()[0]
}

fn zero_header_hash() -> [u8; 32] {
    *blake3::hash(&[0u8; 64]).as_bytes()
}

fn no_op_coinbase() -> CoinbaseWitness<Fr> {
    CoinbaseWitness {
        producer_addr: [0u8; 32],
        amount: 0,
        producer_balance_prev: 0,
        producer_siblings: [[0u8; 32]; SMT_DEPTH],
        producer_empty_bitmap: [0xFFu8; 32],
        _marker: core::marker::PhantomData,
    }
}

fn anchor_with(
    bytes: Vec<u8>,
    claimed_producer_id: u32,
    public_commitment: Fr,
) -> AnchorWitness<Fr> {
    AnchorWitness {
        claimed_producer_id,
        ntt_witness: bytes,
        public_commitment,
        _marker: core::marker::PhantomData,
    }
}

fn no_op_block_with_anchor(anchor: AnchorWitness<Fr>) -> DeltaBlockInputs<Fr> {
    let g = genesis_root();
    DeltaBlockInputs {
        state_root_prev: g,
        state_root_next: g,
        block_header_hash: zero_header_hash(),
        block_height: 1,
        block_header_bytes: vec![0u8; 64],
        transactions: Vec::new(),
        coinbase: no_op_coinbase(),
        anchor,
    }
}

// ── Disabled-binding regime ─────────────────────────────────────────────

#[test]
fn delta_accepts_anchor_with_disabled_binding_and_non_empty_witness() {
    let inputs = no_op_block_with_anchor(anchor_with(
        vec![0x42u8; 16],
        7,
        Fr::from(0u32), // binding off
    ));
    let cs = ConstraintSystem::<Fr>::new_ref();
    DeltaBlockCircuit { inputs }
        .generate_constraints(cs.clone())
        .unwrap();
    assert!(
        cs.is_satisfied().unwrap(),
        "polynomial allocation + norm bound must remain satisfiable on a valid no-op block"
    );
}

#[test]
fn delta_anchor_witness_only_adds_constraints() {
    let cs_empty = ConstraintSystem::<Fr>::new_ref();
    DeltaBlockCircuit {
        inputs: no_op_block_with_anchor(anchor_with(Vec::new(), 0, Fr::from(0u32))),
    }
    .generate_constraints(cs_empty.clone())
    .unwrap();
    let baseline = cs_empty.num_constraints();

    let cs_with = ConstraintSystem::<Fr>::new_ref();
    DeltaBlockCircuit {
        inputs: no_op_block_with_anchor(anchor_with(vec![0x42u8; 16], 7, Fr::from(0u32))),
    }
    .generate_constraints(cs_with.clone())
    .unwrap();
    let with_witness = cs_with.num_constraints();

    assert!(
        with_witness >= baseline,
        "non-empty anchor witness must not skip constraints (with_witness={with_witness}, baseline={baseline})"
    );
    assert!(
        cs_empty.is_satisfied().unwrap() && cs_with.is_satisfied().unwrap(),
        "both empty and non-empty anchor paths must be satisfiable on a valid no-op block"
    );
}

// ── Active-binding regime (Phase A producer-id binding) ─────────────────

#[test]
fn delta_accepts_anchor_with_honest_commitment() {
    let bytes = vec![0xAAu8; 32];
    let producer_id = 7u32;
    let block_height = 1u64;
    let commit = compute_anchor_commitment_native::<Fr>(&bytes, producer_id, block_height);
    assert_ne!(
        commit,
        Fr::from(0u32),
        "honest commit must not collide with the disabled-binding sentinel"
    );

    let inputs = no_op_block_with_anchor(anchor_with(bytes, producer_id, commit));
    let cs = ConstraintSystem::<Fr>::new_ref();
    DeltaBlockCircuit { inputs }
        .generate_constraints(cs.clone())
        .unwrap();
    assert!(
        cs.is_satisfied().unwrap(),
        "δ-circuit must ACCEPT an anchor whose honest commitment matches the in-circuit Poseidon"
    );
}

#[test]
fn delta_rejects_wrong_claimed_producer_id() {
    // Honest commit for producer_id=7; circuit built with producer_id=99
    // — the in-circuit Poseidon recomputes to a different value, equality
    // with the committed value fails, the constraint system must reject.
    let bytes = vec![0xAAu8; 32];
    let honest_producer = 7u32;
    let dishonest_producer = 99u32;
    let block_height = 1u64;
    let commit =
        compute_anchor_commitment_native::<Fr>(&bytes, honest_producer, block_height);

    let inputs = no_op_block_with_anchor(anchor_with(bytes, dishonest_producer, commit));
    let cs = ConstraintSystem::<Fr>::new_ref();
    DeltaBlockCircuit { inputs }
        .generate_constraints(cs.clone())
        .unwrap();
    assert!(
        !cs.is_satisfied().unwrap(),
        "δ-circuit must REJECT a claimed_producer_id that does not match the binding"
    );
}

#[test]
fn delta_rejects_tampered_anchor_witness() {
    // Honest commit for one witness; circuit built with a different
    // witness bytes — in-circuit Poseidon over the tampered coeffs
    // doesn't match the committed value.
    let honest_bytes = vec![0xAAu8; 32];
    let producer_id = 7u32;
    let block_height = 1u64;
    let commit =
        compute_anchor_commitment_native::<Fr>(&honest_bytes, producer_id, block_height);

    let mut tampered = honest_bytes.clone();
    tampered[0] ^= 0xFF;

    let inputs = no_op_block_with_anchor(anchor_with(tampered, producer_id, commit));
    let cs = ConstraintSystem::<Fr>::new_ref();
    DeltaBlockCircuit { inputs }
        .generate_constraints(cs.clone())
        .unwrap();
    assert!(
        !cs.is_satisfied().unwrap(),
        "δ-circuit must REJECT tampered anchor witness bytes against an honest commitment"
    );
}

#[test]
fn delta_rejects_wrong_round_in_commitment() {
    // Honest commit was for block_height=99; circuit built with
    // block_height=1 — in-circuit Poseidon over the actual round
    // doesn't match the committed value.
    let bytes = vec![0xAAu8; 32];
    let producer_id = 7u32;
    let honest_round = 99u64;
    let commit = compute_anchor_commitment_native::<Fr>(&bytes, producer_id, honest_round);

    let mut inputs = no_op_block_with_anchor(anchor_with(bytes, producer_id, commit));
    inputs.block_height = 1; // ← disagrees with the commit's round
    let cs = ConstraintSystem::<Fr>::new_ref();
    DeltaBlockCircuit { inputs }
        .generate_constraints(cs.clone())
        .unwrap();
    assert!(
        !cs.is_satisfied().unwrap(),
        "δ-circuit must REJECT a block_height that does not match the commitment's round"
    );
}
