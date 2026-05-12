//! In-circuit NTT (Number Theoretic Transform) gadget for RLWE commitment verification.
//!
//! The native NTT lives in `q-lattice-guard/src/ntt.rs` and runs outside the circuit.
//! This module encodes the NTT as R1CS constraints so that a prover can demonstrate
//! they performed the transform correctly without revealing the polynomial coefficients.
//!
//! Architecture:
//!   Native NTT (q-lattice-guard) → produces witness values
//!   In-circuit NTT (this file)   → constrains those witness values
//!
//! Status: SCAFFOLD — constraint structure outlined, actual ring arithmetic TODO.
//! The `verify_polynomial_eval` function shows the correct constraint interface.

use ark_ff::PrimeField;
use ark_r1cs_std::{
    fields::fp::FpVar,
    prelude::*,
};
use ark_relations::r1cs::{ConstraintSystemRef, SynthesisError};
use std::marker::PhantomData;

/// In-circuit RLWE commitment verifier.
///
/// Verifies that a commitment (a, b) is a valid RLWE encryption:
///   b = a · s + e + m  (mod q, in R_q = Z_q[X]/(X^n + 1))
///
/// Constraint count estimate: ~30K–50K per commitment (pending NTT gadget implementation).
pub struct NttVerifierGadget<F: PrimeField> {
    _phantom: PhantomData<F>,
}

impl<F: PrimeField> NttVerifierGadget<F> {
    /// Verify a polynomial evaluation at a challenge point.
    ///
    /// Constrains: poly(challenge) == claimed_eval
    /// Uses Horner's method: O(n) multiplications for degree-n polynomial.
    ///
    /// # Arguments
    /// * `coeffs` – polynomial coefficients (allocated as witnesses)
    /// * `challenge` – evaluation point (public input)
    /// * `claimed_eval` – claimed result (public input)
    ///
    /// # TODO
    /// Replace the placeholder sum constraint with actual Horner evaluation.
    /// The constraint count for degree-1024 poly is approximately 1024 × 3 = 3K gates.
    pub fn verify_polynomial_eval(
        cs: ConstraintSystemRef<F>,
        coeffs: &[FpVar<F>],
        challenge: &FpVar<F>,
        claimed_eval: &FpVar<F>,
    ) -> Result<Boolean<F>, SynthesisError> {
        // Horner's method: result = c[n-1]
        //   for i in (0..n-1).rev(): result = result * challenge + c[i]
        //
        // TODO: Replace this placeholder with the actual Horner loop.
        // Current placeholder: sum(coeffs) == claimed_eval (NOT cryptographically correct,
        // only for compilation/architecture demonstration).

        let mut acc = FpVar::Constant(F::zero());
        for c in coeffs {
            acc = &acc + c; // placeholder: should be Horner step
        }
        let _ = challenge; // referenced to show the interface

        // Returns a Boolean indicating whether the evaluation is consistent.
        acc.is_eq(claimed_eval)
    }

    /// Verify that a vector of coefficients satisfies the range bound ||v||_∞ < bound.
    ///
    /// Required for Dilithium norm checks (||z||_∞ < γ₁ - β).
    /// Each coefficient must be decomposed into bits and checked against the bound.
    ///
    /// # TODO
    /// Implement bit decomposition gadget per coefficient.
    /// Estimate: ~20 constraints per coefficient × 256 coefficients = 5K constraints.
    pub fn verify_infinity_norm(
        cs: ConstraintSystemRef<F>,
        coeffs: &[FpVar<F>],
        bound: u64,
    ) -> Result<Boolean<F>, SynthesisError> {
        let bound_var = FpVar::Constant(F::from(bound));
        // TODO: For each coefficient, decompose to bits and range-check.
        // Placeholder: check that the first coefficient is less than the bound
        // (this is NOT correct — all coefficients must be checked).
        if coeffs.is_empty() {
            return Ok(Boolean::constant(true));
        }
        coeffs[0].is_cmp(&bound_var, std::cmp::Ordering::Less, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr;
    use ark_relations::r1cs::ConstraintSystem;

    #[test]
    fn test_ntt_gadget_compiles() {
        let cs = ConstraintSystem::<Fr>::new_ref();
        let coeffs: Vec<FpVar<Fr>> = (0..4)
            .map(|i| FpVar::new_witness(cs.clone(), || Ok(Fr::from(i as u64))).unwrap())
            .collect();
        let challenge = FpVar::new_input(cs.clone(), || Ok(Fr::from(2u64))).unwrap();
        // placeholder: sum = 0+1+2+3 = 6
        let claimed = FpVar::new_input(cs.clone(), || Ok(Fr::from(6u64))).unwrap();

        let result = NttVerifierGadget::verify_polynomial_eval(
            cs.clone(), &coeffs, &challenge, &claimed
        ).unwrap();
        result.enforce_equal(&Boolean::constant(true)).unwrap();
        assert!(cs.is_satisfied().unwrap(), "NTT gadget placeholder should be satisfied");
    }
}
