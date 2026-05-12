//! In-circuit NTT (Number Theoretic Transform) gadget for RLWE commitment verification.
//!
//! The native NTT lives in `q-lattice-guard/src/ntt.rs` and runs outside the circuit.
//! This module constrains those witness values in R1CS so a prover can demonstrate
//! correct transform execution without revealing polynomial coefficients.
//!
//! Architecture:
//!   Native NTT (q-lattice-guard) → produces witness values (preimage knowledge)
//!   In-circuit NTT (this file)   → constrains those witness values (proof of correctness)
//!
//! ## Implemented
//!   - `verify_polynomial_eval`: Horner's method — correct O(n) evaluation
//!   - `verify_infinity_norm`: one-sided range check on all coefficients
//!
//! ## Scaffold (TODO)
//!   - Full Cooley-Tukey butterfly NTT: ~100K constraints for n=256
//!   - Matrix-vector product A·z for Dilithium verification
//!   - Two-sided infinity norm (handles negative field representations)
//!
//! ## Constraint budgets
//!   Polynomial eval (Horner, n=256):  256 × 1 mul + 256 add = ~256 constraints
//!   Infinity norm (all coeffs, n=256): 256 × is_cmp ≈ 256 × 200 = ~51K constraints
//!   Full NTT butterfly (n=256):       ~100K constraints (unimplemented)

use ark_ff::PrimeField;
use ark_r1cs_std::{
    boolean::Boolean,
    fields::fp::FpVar,
    prelude::*,
};
use ark_relations::r1cs::{ConstraintSystemRef, SynthesisError};
use std::marker::PhantomData;

/// In-circuit RLWE commitment and polynomial verifier.
pub struct NttVerifierGadget<F: PrimeField> {
    _phantom: PhantomData<F>,
}

impl<F: PrimeField> NttVerifierGadget<F> {
    /// Verify a polynomial evaluation at a challenge point using Horner's method.
    ///
    /// Constrains: poly(challenge) == claimed_eval
    ///
    /// Horner's method for p(x) = c[0] + c[1]x + ... + c[n-1]x^(n-1):
    ///   result ← c[n-1]
    ///   for i from n-2 down to 0: result ← result × challenge + c[i]
    ///
    /// Constraint cost: (n-1) × 1 mul + (n-1) × 1 add ≈ n-1 total mul constraints.
    /// For n=256: ~255 multiplication constraints.
    pub fn verify_polynomial_eval(
        _cs: ConstraintSystemRef<F>,
        coeffs: &[FpVar<F>],
        challenge: &FpVar<F>,
        claimed_eval: &FpVar<F>,
    ) -> Result<Boolean<F>, SynthesisError> {
        if coeffs.is_empty() {
            return claimed_eval.is_eq(&FpVar::Constant(F::zero()));
        }

        // Horner's method: start from the highest-degree coefficient
        let mut acc = coeffs[coeffs.len() - 1].clone();
        for i in (0..coeffs.len() - 1).rev() {
            // acc = acc × challenge + coeffs[i]
            acc = acc * challenge + coeffs[i].clone();
        }

        acc.is_eq(claimed_eval)
    }

    /// Verify ||v||_∞ < bound: all coefficients satisfy |coeff| < bound.
    ///
    /// Checks all n coefficients (not just the first — previous placeholder was wrong).
    ///
    /// NOTE: This is a one-sided check (coeff < bound). For Dilithium norm verification,
    /// coefficients are in [-bound, bound] and negative values are stored as F::p - |v|
    /// in the field. A complete check requires ALSO verifying coeff > F::p - bound
    /// for the negative range. Full two-sided check via bit decomposition is TODO.
    ///
    /// For the IVC scaffold, this is sufficient to demonstrate the circuit interface
    /// and constraint count (~200 constraints per is_cmp call × n coefficients).
    pub fn verify_infinity_norm(
        _cs: ConstraintSystemRef<F>,
        coeffs: &[FpVar<F>],
        bound: u64,
    ) -> Result<Boolean<F>, SynthesisError> {
        if coeffs.is_empty() {
            return Ok(Boolean::constant(true));
        }

        let bound_var = FpVar::Constant(F::from(bound));
        let mut all_ok = Boolean::constant(true);

        for coeff in coeffs {
            // coeff < bound (strict)
            let in_range =
                coeff.is_cmp(&bound_var, std::cmp::Ordering::Less, false)?;
            all_ok = all_ok.and(&in_range)?;
        }

        Ok(all_ok)
    }

    /// Verify that two polynomials satisfy: a·b == c (mod X^n + 1) under NTT.
    ///
    /// This is the core operation needed for Dilithium's A·z computation.
    ///
    /// TODO: Implement in-circuit NTT using Cooley-Tukey butterfly constraints.
    /// Each butterfly: 1 multiplication + 2 additions = ~3 constraints.
    /// For n=256, log2(256)=8 stages: 256/2 × 8 = 1024 butterflies ≈ 3K constraints.
    /// Total NTT cost: ~6K (forward NTT) + 256 pointwise muls + ~6K (inverse) = ~13K.
    pub fn verify_ntt_product(
        _cs: ConstraintSystemRef<F>,
        _a_ntt: &[FpVar<F>],
        _b_ntt: &[FpVar<F>],
        _c_ntt: &[FpVar<F>],
    ) -> Result<Boolean<F>, SynthesisError> {
        // TODO: Implement pointwise equality check in NTT domain.
        // For now: return constant true (placeholder).
        Ok(Boolean::constant(true))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr;
    use ark_relations::r1cs::ConstraintSystem;

    #[test]
    fn test_horner_evaluation_correct() {
        let cs = ConstraintSystem::<Fr>::new_ref();

        // p(x) = 1 + 2x + 3x^2
        // p(2) = 1 + 4 + 12 = 17
        let coeffs: Vec<FpVar<Fr>> = vec![1u64, 2, 3]
            .into_iter()
            .map(|v| FpVar::new_witness(cs.clone(), || Ok(Fr::from(v))).unwrap())
            .collect();
        let challenge = FpVar::new_input(cs.clone(), || Ok(Fr::from(2u64))).unwrap();
        let claimed = FpVar::new_input(cs.clone(), || Ok(Fr::from(17u64))).unwrap();

        let result = NttVerifierGadget::verify_polynomial_eval(
            cs.clone(), &coeffs, &challenge, &claimed,
        )
        .unwrap();
        result.enforce_equal(&Boolean::constant(true)).unwrap();
        assert!(cs.is_satisfied().unwrap(), "Horner evaluation failed for p(2)=17");
        println!(
            "Horner eval (n=3) constraints: {} (expected ~2 mul)",
            cs.num_constraints()
        );
    }

    #[test]
    fn test_horner_evaluation_wrong_claim_rejected() {
        let cs = ConstraintSystem::<Fr>::new_ref();

        // p(x) = 1 + 2x, p(3) = 7, but claim is 8 → should be unsatisfied
        let coeffs: Vec<FpVar<Fr>> = vec![1u64, 2]
            .into_iter()
            .map(|v| FpVar::new_witness(cs.clone(), || Ok(Fr::from(v))).unwrap())
            .collect();
        let challenge = FpVar::new_input(cs.clone(), || Ok(Fr::from(3u64))).unwrap();
        let wrong_claim = FpVar::new_input(cs.clone(), || Ok(Fr::from(8u64))).unwrap();

        let result = NttVerifierGadget::verify_polynomial_eval(
            cs.clone(), &coeffs, &challenge, &wrong_claim,
        )
        .unwrap();
        result.enforce_equal(&Boolean::constant(true)).unwrap();
        // The circuit should be unsatisfiable since 1 + 2×3 = 7 ≠ 8
        assert!(!cs.is_satisfied().unwrap(), "Wrong claim should be rejected");
    }

    #[test]
    fn test_infinity_norm_all_coefficients_checked() {
        let cs = ConstraintSystem::<Fr>::new_ref();

        // 8 coefficients all in range [0, 100)
        let coeffs: Vec<FpVar<Fr>> = (0u64..8)
            .map(|i| FpVar::new_witness(cs.clone(), || Ok(Fr::from(i * 10))).unwrap())
            .collect();

        let result =
            NttVerifierGadget::verify_infinity_norm(cs.clone(), &coeffs, 100).unwrap();
        result.enforce_equal(&Boolean::constant(true)).unwrap();
        assert!(cs.is_satisfied().unwrap(), "All coefficients in range should pass");
        println!(
            "Infinity norm (n=8) constraints: {} (expected ~8 × is_cmp)",
            cs.num_constraints()
        );
    }

    #[test]
    fn test_ntt_gadget_compiles() {
        let cs = ConstraintSystem::<Fr>::new_ref();
        let coeffs: Vec<FpVar<Fr>> = (0..4)
            .map(|i| FpVar::new_witness(cs.clone(), || Ok(Fr::from(i as u64))).unwrap())
            .collect();
        let challenge = FpVar::new_input(cs.clone(), || Ok(Fr::from(2u64))).unwrap();
        // p(2) = 0 + 1×2 + 2×4 + 3×8 = 34
        let claimed = FpVar::new_input(cs.clone(), || Ok(Fr::from(34u64))).unwrap();

        let result = NttVerifierGadget::verify_polynomial_eval(
            cs.clone(), &coeffs, &challenge, &claimed,
        )
        .unwrap();
        result.enforce_equal(&Boolean::constant(true)).unwrap();
        assert!(cs.is_satisfied().unwrap(), "NTT Horner gadget should be satisfied");
    }
}
