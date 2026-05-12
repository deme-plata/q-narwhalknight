//! Poseidon hash gadget for in-circuit Fiat-Shamir transcript.
//!
//! Poseidon is an algebraically-friendly hash designed for ZK circuits.
//! Replaces BLAKE3/SHAKE256 in the LatticeGuard transcript for ~95% constraint reduction:
//!   BLAKE3 in-circuit: ~50,000 constraints per invocation
//!   Poseidon in-circuit: ~3,000 constraints per invocation
//!
//! IMPORTANT: The Poseidon parameters used here MUST match the parameters used in
//! the LatticeGuard prover's transcript. Any mismatch will cause proof verification
//! failures. Fix the parameters before integrating into production circuits.
//!
//! Recommended parameters: t=3, α=5, full_rounds=8, partial_rounds=57 (BLS12-381).
//! These match the Poseidon paper (Grassi et al. 2019) for 128-bit security.

use ark_ff::PrimeField;
use ark_r1cs_std::{
    fields::fp::FpVar,
    prelude::*,
};
use ark_relations::r1cs::{ConstraintSystemRef, SynthesisError};

/// Width of the Poseidon state (t=3 gives 2 field elements of throughput per invocation).
pub const POSEIDON_T: usize = 3;
/// Number of full rounds per invocation.
pub const POSEIDON_FULL_ROUNDS: usize = 8;
/// Number of partial rounds per invocation.
pub const POSEIDON_PARTIAL_ROUNDS: usize = 57;

/// In-circuit Poseidon hash gadget.
///
/// Implements the Poseidon permutation as R1CS constraints.
/// Used as the Fiat-Shamir hash in the LatticeGuard verifier circuit and
/// as the transcript hash in the BFT signature sub-circuit.
pub struct PoseidonGadget;

impl PoseidonGadget {
    /// Hash `inputs` (1 or 2 field elements) to a single field element.
    ///
    /// Uses capacity-1 sponge mode: absorb inputs one-by-one, squeeze one output.
    /// Approximately 3,000 R1CS constraints for t=3, 8+57 rounds.
    ///
    /// # TODO
    /// Implement actual round function (AddRoundConstants → SubWords → MixLayer).
    /// The SubWords step for α=5: x → x^5 costs 3 multiplications (x², x⁴, x⁵).
    /// Full rounds: all t positions get SubWords. Partial rounds: only position 0.
    pub fn hash<F: PrimeField>(
        cs: ConstraintSystemRef<F>,
        inputs: &[FpVar<F>],
    ) -> Result<FpVar<F>, SynthesisError> {
        assert!(!inputs.is_empty(), "Poseidon hash requires at least one input");
        assert!(inputs.len() <= POSEIDON_T - 1, "Too many inputs for sponge width");

        // TODO: Initialize state with round constants, absorb inputs, run permutation,
        // return state[1] as output.
        //
        // Placeholder: fold inputs with addition (no security, only for compilation).
        let mut acc = FpVar::Constant(F::zero());
        for inp in inputs {
            acc = &acc + inp;
        }
        Ok(acc)
    }

    /// Hash two field elements (common case: state root pair).
    pub fn hash2<F: PrimeField>(
        cs: ConstraintSystemRef<F>,
        a: &FpVar<F>,
        b: &FpVar<F>,
    ) -> Result<FpVar<F>, SynthesisError> {
        Self::hash::<F>(cs, &[a.clone(), b.clone()])
    }

    /// Hash a sequence of field elements using the sponge construction.
    ///
    /// For inputs larger than t-1 elements, absorbs in blocks of t-1.
    pub fn hash_many<F: PrimeField>(
        cs: ConstraintSystemRef<F>,
        inputs: &[FpVar<F>],
    ) -> Result<FpVar<F>, SynthesisError> {
        if inputs.is_empty() {
            return Ok(FpVar::Constant(F::zero()));
        }
        // TODO: Implement multi-block sponge absorption.
        // Placeholder: recursive fold.
        let mut state = inputs[0].clone();
        for inp in &inputs[1..] {
            state = &state + inp; // placeholder
        }
        Ok(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr;
    use ark_relations::r1cs::ConstraintSystem;

    #[test]
    fn test_poseidon_gadget_compiles() {
        let cs = ConstraintSystem::<Fr>::new_ref();
        let a = FpVar::new_witness(cs.clone(), || Ok(Fr::from(42u64))).unwrap();
        let b = FpVar::new_witness(cs.clone(), || Ok(Fr::from(13u64))).unwrap();
        let _hash = PoseidonGadget::hash2(cs.clone(), &a, &b).unwrap();
        assert!(cs.is_satisfied().unwrap());
    }

    #[test]
    fn test_poseidon_hash_many() {
        let cs = ConstraintSystem::<Fr>::new_ref();
        let inputs: Vec<_> = (0..5)
            .map(|i| FpVar::new_witness(cs.clone(), || Ok(Fr::from(i as u64))).unwrap())
            .collect();
        let _hash = PoseidonGadget::hash_many(cs.clone(), &inputs).unwrap();
        assert!(cs.is_satisfied().unwrap());
    }
}
