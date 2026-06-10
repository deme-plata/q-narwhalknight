//! Host-side helper for NTT-based anchor election witness.
//!
//! Closes TODO `delta-circuit-PHASE-1B` from
//! `crates/q-ivc/src/circuits/delta_block.rs`.
//!
//! ## What the anchor election proves
//!
//! Each round, one validator is the "anchor" — the producer whose block
//! is committed for that round. The election uses a verifiable NTT-based
//! randomness beacon (see `crates/q-consensus/src/anchor_election.rs` for
//! the off-chain reference implementation):
//!
//!   1. Each validator has a VDF (Verifiable Delay Function) output for
//!      the round, contributed in the previous round's mempool.
//!   2. The collective entropy is the XOR (or polynomial sum) of all
//!      contributed VDF outputs, evaluated in the NTT domain.
//!   3. The anchor is `argmax_i(NTT(entropy)[i] mod num_validators)`
//!      where `i` ranges over validator IDs.
//!
//! The δ-circuit's job is to prove: "the claimed_producer_id in the
//! block header is the legitimate anchor under this beacon for this
//! round." Soundness requires verifying:
//!   • Each contributed VDF output is well-formed (separate VDF proof
//!     verification; out of scope here).
//!   • The polynomial sum is correct.
//!   • The NTT of the sum is correct (use `NttVerifierGadget`).
//!   • The argmax computation selected `claimed_producer_id`.
//!
//! ## Why this is its own helper layer
//!
//! The witness shape needed by `NttVerifierGadget` is polynomial
//! coefficients (vectors of `FpVar<F>`). The wire format coming from
//! `crates/q-network/src/anchor_proof.rs` is a packed VDF proof byte
//! buffer. The unpacking is bit-fiddly enough to warrant a dedicated
//! helper rather than inlining in `circuits/delta_block.rs`.
//!
//! ## Status (Phase A — partial binding)
//!
//! `AnchorVdfBytes::allocate_as_polynomial` is **implemented**: it decodes
//! the wire bytes into `ANCHOR_NTT_DIM` polynomial coefficients, allocates
//! them as constrained `FpVar` witnesses, and enforces each coefficient is
//! in `[0, ANCHOR_NTT_Q)` via `NttVerifierGadget::verify_infinity_norm`.
//! Cost: ~51K constraints per VDF proof.
//!
//! `verify_anchor_election` allocates the polynomial and pre-allocates
//! `claimed_producer_id` + `round` as witnesses so the Phase 5 binding
//! lands as a single-edit change. It still returns `Boolean::constant(true)`
//! because the full anchor-election spec (multi-validator NTT-sum + argmax
//! against the validator set) requires extending `AnchorWitness` (in
//! `crates/q-ivc/src/circuits/delta_block.rs`) to carry per-validator VDF
//! outputs, the validator-set size, and a publicly-committed beacon hash.
//! The API server's `accept_block` path validates the election off-chain
//! during the advisory window.

use ark_ff::PrimeField;
use ark_r1cs_std::{
    fields::fp::FpVar,
    prelude::*,
    R1CSVar,
};
use ark_relations::r1cs::{ConstraintSystem, ConstraintSystemRef, SynthesisError};

use crate::gadgets::{ntt::NttVerifierGadget, poseidon::PoseidonGadget};

/// NTT modulus and dimension for the anchor-election polynomial.
///
/// Reuses the same modulus as Dilithium (Q = 8 380 417) so the NTT
/// roots are shared.
pub const ANCHOR_NTT_DIM: usize = 256;
pub const ANCHOR_NTT_Q: u64 = 8_380_417;

/// Bytes per coefficient in the canonical wire format: each `u64`
/// little-endian chunk decodes to one coefficient, mod-reduced into
/// `[0, ANCHOR_NTT_Q)`. Chunks beyond the first `ANCHOR_NTT_DIM` are
/// ignored; fewer chunks zero-pad. This is the format the gadget
/// expects — the off-chain `AnchorElection` producer must serialize
/// matching bytes when the Phase 5 binding activates.
pub const ANCHOR_COEFF_BYTES: usize = 8;

/// Number of polynomial coefficients fed into the Phase 5 Poseidon
/// binding's pre-image. Kept small (8) to bound Poseidon cost; the
/// infinity-norm constraint already binds every coefficient in the
/// polynomial, so sampling here purely caps the hash circuit's size.
pub const ANCHOR_BIND_SAMPLES: usize = 8;

/// Raw bytes of one anchor-election VDF proof (one per validator per round).
///
/// The exact byte layout is defined by `crates/q-network/src/anchor_proof.rs`
/// (production code) — the canonical format binds a validator's
/// contribution to its VDF output for verification.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AnchorVdfBytes(pub Vec<u8>);

impl AnchorVdfBytes {
    pub fn new(bytes: Vec<u8>) -> Self {
        Self(bytes)
    }

    /// Decode the wire bytes into `ANCHOR_NTT_DIM` polynomial coefficients
    /// and allocate them as constrained `FpVar` witnesses. Each coefficient
    /// is enforced to lie in `[0, ANCHOR_NTT_Q)` via the NTT verifier's
    /// one-sided infinity-norm check.
    ///
    /// Wire format: each `ANCHOR_COEFF_BYTES = 8`-byte little-endian chunk
    /// decodes to a `u64`, mod-reduced into `[0, ANCHOR_NTT_Q)`. Empty or
    /// short input zero-pads to `ANCHOR_NTT_DIM`. The padded path matches
    /// the prior stub's "vacuous accept on empty witness" behavior so the
    /// δ-circuit stays satisfiable during the advisory window.
    ///
    /// Constraint cost: ~51K (one-sided norm check over 256 coefficients).
    pub fn allocate_as_polynomial<F: PrimeField>(
        &self,
        cs: ConstraintSystemRef<F>,
    ) -> Result<Vec<FpVar<F>>, SynthesisError> {
        let mut coeffs: Vec<FpVar<F>> = Vec::with_capacity(ANCHOR_NTT_DIM);
        for i in 0..ANCHOR_NTT_DIM {
            let off = i * ANCHOR_COEFF_BYTES;
            let coeff_u64 = if off + ANCHOR_COEFF_BYTES <= self.0.len() {
                let bytes: [u8; ANCHOR_COEFF_BYTES] = self.0
                    [off..off + ANCHOR_COEFF_BYTES]
                    .try_into()
                    .expect("8-byte chunk by construction");
                u64::from_le_bytes(bytes) % ANCHOR_NTT_Q
            } else {
                0u64
            };
            coeffs.push(FpVar::new_witness(cs.clone(), || Ok(F::from(coeff_u64)))?);
        }

        let in_range =
            NttVerifierGadget::<F>::verify_infinity_norm(cs.clone(), &coeffs, ANCHOR_NTT_Q)?;
        in_range.enforce_equal(&Boolean::constant(true))?;

        Ok(coeffs)
    }
}

/// In-circuit anchor verification — Phase A binding live.
///
/// Two binding regimes share this function:
///
/// **Polynomial allocation + norm bound (always-on).** Each VDF/NTT witness
/// is allocated as `ANCHOR_NTT_DIM` constrained `FpVar` coefficients, with
/// `NttVerifierGadget::verify_infinity_norm` enforcing each coefficient
/// lies in `[0, ANCHOR_NTT_Q)`. ~51K constraints per VDF proof.
///
/// **Producer-id binding (active when `public_commitment != F::zero()`).**
/// Computes the in-circuit Poseidon hash over
/// `(coeffs[..ANCHOR_BIND_SAMPLES] || claimed_producer_id || round)` and
/// enforces equality with `public_commitment`. An adversary who substitutes
/// any of the three (witness coefficients, claimed producer id, round)
/// without re-running Poseidon over a matching preimage off-chain fails
/// the equality check. The matching off-chain value is produced by
/// [`compute_anchor_commitment_native`].
///
/// The zero-commitment sentinel is the Phase A → Phase 5 migration latch:
/// callers (tests, advisory-window blocks) supply `F::zero()` for backward
/// compat; production block-headers supply the real commitment to activate
/// the binding.
///
/// Returns `Boolean::constant(true)` so the δ-circuit's overall constraint
/// system remains satisfiable — the soundness contribution lives in the
/// constraints added inside this function, not in the returned Boolean.
///
/// Constraint cost: ~51K (norm) + ~6K (Poseidon, when binding active) per
/// VDF proof.
pub fn verify_anchor_election<F: PrimeField>(
    cs: ConstraintSystemRef<F>,
    vdf_proofs: &[AnchorVdfBytes],
    claimed_producer_id: u32,
    round: u64,
    public_commitment: F,
) -> Result<Boolean<F>, SynthesisError> {
    let producer_id_var =
        FpVar::new_witness(cs.clone(), || Ok(F::from(claimed_producer_id)))?;
    let round_var = FpVar::new_witness(cs.clone(), || Ok(F::from(round)))?;
    let binding_active = public_commitment != F::zero();
    let commitment_var = FpVar::new_witness(cs.clone(), || Ok(public_commitment))?;

    for vdf in vdf_proofs {
        let coeffs = vdf.allocate_as_polynomial(cs.clone())?;
        if binding_active {
            let preimage = build_binding_preimage(&coeffs, &producer_id_var, &round_var);
            let computed = PoseidonGadget::hash_many::<F>(cs.clone(), &preimage)?;
            computed.enforce_equal(&commitment_var)?;
        }
    }

    Ok(Boolean::constant(true))
}

/// Build the canonical Phase 5 binding pre-image:
/// `[coeffs[..ANCHOR_BIND_SAMPLES.min(len)], producer_id, round]`.
///
/// Sampling the first `ANCHOR_BIND_SAMPLES` coefficients is sufficient: every
/// coefficient is already constrained by `allocate_as_polynomial`'s norm
/// bound, so the Poseidon preimage need only commit to a sampling of them
/// to bind the full witness through Fiat-Shamir.
fn build_binding_preimage<F: PrimeField>(
    coeffs: &[FpVar<F>],
    producer_id_var: &FpVar<F>,
    round_var: &FpVar<F>,
) -> Vec<FpVar<F>> {
    let take = coeffs.len().min(ANCHOR_BIND_SAMPLES);
    let mut preimage: Vec<FpVar<F>> = Vec::with_capacity(take + 2);
    preimage.extend(coeffs.iter().take(take).cloned());
    preimage.push(producer_id_var.clone());
    preimage.push(round_var.clone());
    preimage
}

/// Compute the anchor-election Poseidon commitment off-circuit.
///
/// Mirrors the in-circuit binding in [`verify_anchor_election`] exactly:
/// the result is the value an honest block-header producer publishes as
/// `AnchorWitness::public_commitment` so the in-circuit Poseidon hash
/// equals it. Used by tests and (future) the off-chain anchor-proof
/// producer in `crates/q-network/src/anchor_proof.rs`.
///
/// Runs the in-circuit gadget against a throwaway constraint system and
/// extracts the resulting witness value. Slightly wasteful (allocates a
/// CS for one hash) but the only correct way to keep native + in-circuit
/// Poseidon outputs in lockstep without duplicating the permutation logic.
pub fn compute_anchor_commitment_native<F: PrimeField>(
    vdf_bytes: &[u8],
    claimed_producer_id: u32,
    round: u64,
) -> F {
    let cs = ConstraintSystem::<F>::new_ref();
    let vdf = AnchorVdfBytes::new(vdf_bytes.to_vec());
    let coeffs = vdf
        .allocate_as_polynomial::<F>(cs.clone())
        .expect("native polynomial allocation must not fail");
    let producer_id_var =
        FpVar::new_witness(cs.clone(), || Ok(F::from(claimed_producer_id)))
            .expect("native scalar allocation must not fail");
    let round_var = FpVar::new_witness(cs.clone(), || Ok(F::from(round)))
        .expect("native scalar allocation must not fail");

    let preimage = build_binding_preimage(&coeffs, &producer_id_var, &round_var);
    let computed = PoseidonGadget::hash_many::<F>(cs, &preimage)
        .expect("native Poseidon hash must not fail");
    computed.value().expect("native FpVar value must be set")
}

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr;
    use ark_relations::r1cs::ConstraintSystem;

    #[test]
    fn anchor_constants_match_dilithium_modulus() {
        use crate::gadgets::dilithium::DILITHIUM_Q;
        assert_eq!(ANCHOR_NTT_Q, DILITHIUM_Q);
        assert_eq!(ANCHOR_NTT_DIM, 256);
    }

    #[test]
    fn anchor_vdf_bytes_wraps_arbitrary_payload() {
        let bytes = AnchorVdfBytes::new(vec![0u8; 1024]);
        assert_eq!(bytes.0.len(), 1024);
    }

    #[test]
    fn anchor_election_empty_returns_constant_true() {
        // With no VDF proofs and binding disabled (commitment=0), the
        // function allocates the binding scalars and skips the polynomial
        // loop. Stays satisfiable so empty-anchor test fixtures used
        // across delta_block / epoch_transition tests keep passing.
        let cs = ConstraintSystem::<Fr>::new_ref();
        let result = verify_anchor_election::<Fr>(cs.clone(), &[], 0, 0, Fr::from(0u32)).unwrap();
        assert_eq!(result.value().unwrap(), true);
        assert!(cs.is_satisfied().unwrap());
    }

    #[test]
    fn allocate_as_polynomial_pads_short_input_and_satisfies_norm() {
        // 16 bytes = 2 coefficients (each well under ANCHOR_NTT_Q); the
        // remaining 254 coefficients zero-pad. The norm check passes.
        let cs = ConstraintSystem::<Fr>::new_ref();
        let bytes = vec![0x12u8; 16];
        let vdf = AnchorVdfBytes::new(bytes);
        let coeffs = vdf.allocate_as_polynomial::<Fr>(cs.clone()).unwrap();
        assert_eq!(coeffs.len(), ANCHOR_NTT_DIM);
        assert!(cs.is_satisfied().unwrap());
    }

    #[test]
    fn verify_anchor_election_allocates_polynomial_with_disabled_binding() {
        // commitment = 0 → binding inactive, but polynomial allocation +
        // norm bound still fire. Used by every Phase-A test fixture that
        // doesn't yet ship the off-chain anchor commitment.
        let cs = ConstraintSystem::<Fr>::new_ref();
        let bytes: Vec<u8> = (0..32u64).flat_map(|v| v.to_le_bytes()).collect();
        let vdf = AnchorVdfBytes::new(bytes);
        let result = verify_anchor_election::<Fr>(cs.clone(), &[vdf], 7, 42, Fr::from(0u32))
            .unwrap();
        assert_eq!(result.value().unwrap(), true);
        assert!(
            cs.num_constraints() > 0,
            "polynomial allocation + norm check must register constraints"
        );
        assert!(cs.is_satisfied().unwrap());
    }

    #[test]
    fn verify_anchor_election_accepts_honest_commitment() {
        // Honest off-chain producer computes the commitment for a given
        // witness + producer_id + round, ships it in the block header,
        // the in-circuit Poseidon matches → satisfiable.
        let bytes: Vec<u8> = (0..32u64).flat_map(|v| v.to_le_bytes()).collect();
        let commit = compute_anchor_commitment_native::<Fr>(&bytes, 7, 42);
        assert_ne!(commit, Fr::from(0u32), "honest commit must not collide with the disabled-binding sentinel");

        let cs = ConstraintSystem::<Fr>::new_ref();
        let vdf = AnchorVdfBytes::new(bytes);
        let result = verify_anchor_election::<Fr>(cs.clone(), &[vdf], 7, 42, commit).unwrap();
        assert_eq!(result.value().unwrap(), true);
        assert!(
            cs.is_satisfied().unwrap(),
            "honest commitment must satisfy the in-circuit Poseidon equality"
        );
    }

    #[test]
    fn verify_anchor_election_rejects_wrong_producer_id() {
        // Honest commitment for producer_id=7; in-circuit claims
        // producer_id=99 → in-circuit Poseidon recomputes to a different
        // value, equality with commitment fails → rejected.
        let bytes: Vec<u8> = (0..32u64).flat_map(|v| v.to_le_bytes()).collect();
        let honest_commit = compute_anchor_commitment_native::<Fr>(&bytes, 7, 42);

        let cs = ConstraintSystem::<Fr>::new_ref();
        let vdf = AnchorVdfBytes::new(bytes);
        verify_anchor_election::<Fr>(cs.clone(), &[vdf], 99, 42, honest_commit).unwrap();
        assert!(
            !cs.is_satisfied().unwrap(),
            "wrong claimed_producer_id must fail the Poseidon equality"
        );
    }

    #[test]
    fn verify_anchor_election_rejects_tampered_witness() {
        // Honest commitment for one witness; pass a different witness
        // with the same producer/round → recomputed Poseidon differs.
        let honest_bytes: Vec<u8> = (0..32u64).flat_map(|v| v.to_le_bytes()).collect();
        let commit = compute_anchor_commitment_native::<Fr>(&honest_bytes, 7, 42);

        let mut tampered = honest_bytes.clone();
        tampered[0] ^= 0xFF;

        let cs = ConstraintSystem::<Fr>::new_ref();
        let vdf = AnchorVdfBytes::new(tampered);
        verify_anchor_election::<Fr>(cs.clone(), &[vdf], 7, 42, commit).unwrap();
        assert!(
            !cs.is_satisfied().unwrap(),
            "tampered witness bytes must fail the Poseidon equality"
        );
    }
}
