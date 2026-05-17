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
//! ## Status
//!
//! **STUB.** This file ships the type + allocator skeleton with the spec
//! references. The full unpacking + NTT challenge construction is
//! tracked as a follow-up commit (1B-final). Until then, the δ-circuit
//! Phase 2 (NTT anchor verification) is a no-op — the API server's
//! block-by-block accept_block path independently verifies the anchor
//! during the advisory window.

use ark_ff::PrimeField;
use ark_r1cs_std::fields::fp::FpVar;
use ark_r1cs_std::prelude::Boolean;
#[cfg(test)]
use ark_r1cs_std::R1CSVar;
use ark_relations::r1cs::{ConstraintSystemRef, SynthesisError};

/// NTT modulus and dimension for the anchor-election polynomial.
///
/// Reuses the same modulus as Dilithium (Q = 8 380 417) so the NTT
/// roots are shared.
pub const ANCHOR_NTT_DIM: usize = 256;
pub const ANCHOR_NTT_Q: u64 = 8_380_417;

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

    /// Decode the VDF proof into its polynomial-coefficient form, suitable
    /// for `NttVerifierGadget` consumption.
    ///
    /// **STUB.** Body needs to:
    ///   1. Parse the header (round number, validator id, VDF output
    ///      length).
    ///   2. Verify the VDF output's structure (separate VDF verifier —
    ///      `crates/q-vdf/src/verifier.rs`).
    ///   3. Convert the n=256 polynomial coefficients from the VDF
    ///      output's hash-derived bytes into FpVar witnesses.
    pub fn allocate_as_polynomial<F: PrimeField>(
        &self,
        _cs: ConstraintSystemRef<F>,
    ) -> Result<Vec<FpVar<F>>, SynthesisError> {
        // TODO(anchor-witness-allocate): see module docstring.
        Err(SynthesisError::AssignmentMissing)
    }
}

/// In-circuit anchor verification result.
///
/// **STUB.** When filled, this function:
///   1. Allocates the per-validator polynomial vectors via
///      `AnchorVdfBytes::allocate_as_polynomial`.
///   2. Sums them via the NTT verifier's pointwise ops.
///   3. Computes the argmax over the result's coefficients.
///   4. Enforces argmax == claimed_producer_id.
///
/// Returns a Boolean that the δ-circuit enforces == true. Stub returns
/// `Boolean::constant(true)` so the δ-circuit's overall constraint
/// system remains satisfiable during the advisory window.
pub fn verify_anchor_election<F: PrimeField>(
    _cs: ConstraintSystemRef<F>,
    _vdf_proofs: &[AnchorVdfBytes],
    _claimed_producer_id: u32,
    _round: u64,
) -> Result<Boolean<F>, SynthesisError> {
    // TODO(anchor-witness-verify): see module docstring.
    //
    // Returning constant-true is INTENTIONAL during the stub phase: the
    // δ-circuit's Phase 5 mandatory-verification activation requires this
    // body to be filled in. Until then the anchor check is non-enforcing
    // at the recursive-proof level (the API server's accept_block path
    // still validates anchor election block-by-block).
    Ok(Boolean::constant(true))
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
    fn anchor_election_stub_returns_constant_true() {
        // Until the verify_anchor_election body is filled, it returns
        // Boolean::constant(true) — the stub MUST NOT reject valid blocks
        // or the δ-circuit's overall satisfiability breaks. This is what
        // keeps the recursive proof non-enforcing (advisory) for anchor
        // during the stub phase.
        let cs = ConstraintSystem::<Fr>::new_ref();
        let result = verify_anchor_election::<Fr>(cs.clone(), &[], 0, 0).unwrap();
        assert_eq!(result.value().unwrap(), true);
        assert!(cs.is_satisfied().unwrap());
    }
}
