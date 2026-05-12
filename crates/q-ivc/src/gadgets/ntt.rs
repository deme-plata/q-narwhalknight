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
//!   - `verify_ntt_product`: pointwise a[i]·b[i] == c[i] equality in NTT domain
//!   - `ntt` / `intt`: Cooley-Tukey iterative DIT butterfly (caller provides roots)
//!   - `pointwise_mul`: element-wise multiplication in NTT domain (n constraints)
//!   - `poly_mul`: full polynomial multiplication: NTT → pointwise → INTT
//!
//! ## Scaffold (TODO)
//!   - Matrix-vector product A·z for Dilithium ring verification
//!   - Two-sided infinity norm (negative coefficients stored as p - |v| in field)
//!   - Root-of-unity table generation for BLS12-381 Fr (caller currently provides)
//!   - Negacyclic NTT for F[X]/(X^n + 1) (current: cyclic, F[X]/(X^n − 1))
//!
//! ## Constraint budgets
//!   Polynomial eval (Horner, n=256):    ~255 mul constraints
//!   Infinity norm (all coeffs, n=256):  256 × is_cmp ≈ 256 × 200 = ~51K constraints
//!   NTT butterfly (n=256, 8 stages):    (n/2)×log₂n = 1024 butterflies
//!     Each butterfly: 1 field mul + 2 add = 1 R1CS constraint
//!     Total per NTT: ~1K constraints; forward+inverse: ~2K
//!   poly_mul (n=256):                   2 NTTs + n muls + INTT ≈ 3K constraints
//!   verify_ntt_product (n=256):         n muls (is_eq) ≈ 256 constraints
//!
//! ## Root convention for `ntt` / `intt`
//!   `roots` must be length n. For stage m and index i within that stage,
//!   the twiddle factor is `roots[m + i]`. This is the standard bit-reversal-indexed
//!   precomputed table: `roots[k] = ω^(bit_rev(k))` for k in 1..n where ω is a
//!   primitive n-th root of unity in F. `roots[0]` is unused.
//!
//!   For negacyclic NTT (RLWE / Dilithium's X^n + 1 ring), use a primitive 2n-th
//!   root of unity ψ and pre-multiply coefficients by ψ^i before calling ntt.
//!   Callers should pre-compute using `q-lattice-guard::ntt::compute_roots(n)`.

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
    // ─── Polynomial evaluation ────────────────────────────────────────────────

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

    // ─── Norm verification ────────────────────────────────────────────────────

    /// Verify ||v||_∞ < bound: all coefficients satisfy coeff < bound.
    ///
    /// Checks all n coefficients (not just the first).
    ///
    /// NOTE: One-sided check only (coeff < bound). For Dilithium norm verification,
    /// coefficients in [-bound, bound] have negative values stored as F::p - |v|
    /// in the field. A complete check requires ALSO verifying coeff > F::p - bound
    /// for the negative range. Full two-sided check via bit decomposition is TODO.
    ///
    /// Constraint cost: n × is_cmp ≈ n × 200 constraints. For n=256: ~51K.
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

    // ─── NTT-domain product verification ─────────────────────────────────────

    /// Verify a·b == c in NTT domain: checks a_ntt[i] · b_ntt[i] == c_ntt[i] for all i.
    ///
    /// This is the core correctness check for Dilithium's A·z computation when the
    /// caller has already applied the NTT to all three operands and provides c as a
    /// claimed product witness.
    ///
    /// Constraint cost: n multiplications + n is_eq checks ≈ 2n constraints.
    /// For n=256: ~512 constraints.
    pub fn verify_ntt_product(
        _cs: ConstraintSystemRef<F>,
        a_ntt: &[FpVar<F>],
        b_ntt: &[FpVar<F>],
        c_ntt: &[FpVar<F>],
    ) -> Result<Boolean<F>, SynthesisError> {
        if a_ntt.len() != b_ntt.len() || b_ntt.len() != c_ntt.len() {
            return Ok(Boolean::constant(false));
        }
        if a_ntt.is_empty() {
            return Ok(Boolean::constant(true));
        }

        let mut all_ok = Boolean::constant(true);
        for ((a, b), c) in a_ntt.iter().zip(b_ntt.iter()).zip(c_ntt.iter()) {
            let product = a.clone() * b.clone();
            let eq = product.is_eq(c)?;
            all_ok = all_ok.and(&eq)?;
        }
        Ok(all_ok)
    }

    // ─── Cooley-Tukey NTT butterfly ───────────────────────────────────────────

    /// In-circuit Cooley-Tukey iterative DIT NTT.
    ///
    /// Constrains the butterfly network for a length-n polynomial.
    /// Each butterfly adds one R1CS multiplication constraint:
    ///   u = a[j],  v = a[j+t] · roots[m+i]
    ///   a[j]   = u + v   (addition, free)
    ///   a[j+t] = u - v   (subtraction, free)
    ///
    /// Total: (n/2) × log₂(n) multiplications.
    /// For n=256: 1024 butterfly multiplications ≈ 1024 R1CS constraints.
    ///
    /// `roots`: length-n array; `roots[k] = ω^(bit_rev(k))` for k in 1..n.
    /// `roots[0]` is unused. n must be a power of 2 and ≥ 2.
    pub fn ntt(
        _cs: &ConstraintSystemRef<F>,
        a: &[FpVar<F>],
        roots: &[F],
    ) -> Result<Vec<FpVar<F>>, SynthesisError> {
        let n = a.len();
        assert!(
            n >= 2 && n.is_power_of_two(),
            "ntt: n={} must be a power of 2 and ≥ 2",
            n
        );
        assert_eq!(roots.len(), n, "ntt: roots.len() must equal n={}", n);

        let mut a = a.to_vec();
        let mut t = n;
        let mut m = 1usize;

        while m < n {
            t /= 2;
            for i in 0..m {
                let s = FpVar::Constant(roots[m + i]);
                let j1 = 2 * i * t;
                for j in j1..(j1 + t) {
                    let u = a[j].clone();
                    let v = a[j + t].clone() * s.clone();
                    a[j] = u.clone() + v.clone();
                    a[j + t] = u - v;
                }
            }
            m *= 2;
        }

        Ok(a)
    }

    /// In-circuit inverse NTT: same butterfly network with inverse roots, scaled by n⁻¹.
    ///
    /// `inv_roots[k] = ω^{-bit_rev(k)}` for k in 1..n.
    /// `n_inv = n⁻¹` in the field F (caller must compute this natively).
    ///
    /// Constraint cost: same as `ntt` plus n multiplications for the n⁻¹ scaling.
    pub fn intt(
        cs: &ConstraintSystemRef<F>,
        a: &[FpVar<F>],
        inv_roots: &[F],
        n_inv: F,
    ) -> Result<Vec<FpVar<F>>, SynthesisError> {
        let mut result = Self::ntt(cs, a, inv_roots)?;
        let n_inv_const = FpVar::Constant(n_inv);
        for x in result.iter_mut() {
            *x = x.clone() * n_inv_const.clone();
        }
        Ok(result)
    }

    /// Pointwise multiplication of two NTT-domain polynomials: c[i] = a[i] · b[i].
    ///
    /// Constraint cost: n multiplications.
    pub fn pointwise_mul(
        a: &[FpVar<F>],
        b: &[FpVar<F>],
    ) -> Result<Vec<FpVar<F>>, SynthesisError> {
        assert_eq!(a.len(), b.len(), "pointwise_mul: length mismatch");
        Ok(a.iter()
            .zip(b.iter())
            .map(|(ai, bi)| ai.clone() * bi.clone())
            .collect())
    }

    /// Full polynomial multiplication via NTT: c = a · b in F[X]/(X^n − 1).
    ///
    /// Steps: forward NTT on a and b → pointwise mul → inverse NTT.
    ///
    /// Note: This computes the product in the *cyclic* ring F[X]/(X^n − 1).
    /// For RLWE / Dilithium's negacyclic ring F[X]/(X^n + 1), the caller should
    /// pre-multiply coefficients by a primitive 2n-th root of unity ψ before
    /// calling ntt, and divide by ψ after intt.
    ///
    /// Constraint cost (n=256):
    ///   2 × NTT:           2 × 1024 = 2048 multiplications
    ///   pointwise_mul:            256 multiplications
    ///   INTT:             1024 + 256 = 1280 multiplications
    ///   Total:                   ~3.6K R1CS constraints
    pub fn poly_mul(
        cs: &ConstraintSystemRef<F>,
        a: &[FpVar<F>],
        b: &[FpVar<F>],
        fwd_roots: &[F],
        inv_roots: &[F],
        n_inv: F,
    ) -> Result<Vec<FpVar<F>>, SynthesisError> {
        let a_ntt = Self::ntt(cs, a, fwd_roots)?;
        let b_ntt = Self::ntt(cs, b, fwd_roots)?;
        let c_ntt = Self::pointwise_mul(&a_ntt, &b_ntt)?;
        Self::intt(cs, &c_ntt, inv_roots, n_inv)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr;
    use ark_ff::Field;
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

    #[test]
    fn test_verify_ntt_product_correct() {
        let cs = ConstraintSystem::<Fr>::new_ref();

        // a = [2, 3, 5], b = [4, 6, 10], c = a*b pointwise = [8, 18, 50]
        let a: Vec<FpVar<Fr>> = [2u64, 3, 5]
            .iter()
            .map(|&v| FpVar::new_witness(cs.clone(), || Ok(Fr::from(v))).unwrap())
            .collect();
        let b: Vec<FpVar<Fr>> = [4u64, 6, 10]
            .iter()
            .map(|&v| FpVar::new_witness(cs.clone(), || Ok(Fr::from(v))).unwrap())
            .collect();
        let c: Vec<FpVar<Fr>> = [8u64, 18, 50]
            .iter()
            .map(|&v| FpVar::new_witness(cs.clone(), || Ok(Fr::from(v))).unwrap())
            .collect();

        let ok = NttVerifierGadget::verify_ntt_product(cs.clone(), &a, &b, &c).unwrap();
        ok.enforce_equal(&Boolean::constant(true)).unwrap();
        assert!(cs.is_satisfied().unwrap(), "pointwise product should be verified");
    }

    #[test]
    fn test_verify_ntt_product_wrong_rejected() {
        let cs = ConstraintSystem::<Fr>::new_ref();

        // a[0]*b[0] = 2*4 = 8, but c[0] claims 9 → should fail
        let a: Vec<FpVar<Fr>> = [2u64, 1]
            .iter()
            .map(|&v| FpVar::new_witness(cs.clone(), || Ok(Fr::from(v))).unwrap())
            .collect();
        let b: Vec<FpVar<Fr>> = [4u64, 1]
            .iter()
            .map(|&v| FpVar::new_witness(cs.clone(), || Ok(Fr::from(v))).unwrap())
            .collect();
        let c_wrong: Vec<FpVar<Fr>> = [9u64, 1]  // 9 ≠ 8
            .iter()
            .map(|&v| FpVar::new_witness(cs.clone(), || Ok(Fr::from(v))).unwrap())
            .collect();

        let ok =
            NttVerifierGadget::verify_ntt_product(cs.clone(), &a, &b, &c_wrong).unwrap();
        ok.enforce_equal(&Boolean::constant(true)).unwrap();
        assert!(!cs.is_satisfied().unwrap(), "Wrong pointwise product should be rejected");
    }

    /// Test ntt + intt round-trip: INTT(NTT(a)) == a.
    ///
    /// Uses n=2 with the trivial root ω = 1 (so ntt acts as [a+b, a-b] and
    /// intt with inv_root=1 and n_inv=2⁻¹ inverts it).
    /// This tests the butterfly constraint structure, not field-specific roots.
    #[test]
    fn test_ntt_intt_roundtrip_n2() {
        let cs = ConstraintSystem::<Fr>::new_ref();

        let a0 = Fr::from(7u64);
        let a1 = Fr::from(13u64);

        let a: Vec<FpVar<Fr>> = [a0, a1]
            .iter()
            .map(|&v| FpVar::new_witness(cs.clone(), || Ok(v)).unwrap())
            .collect();

        // For n=2, roots[1] = ω where ω is a primitive 2nd root of unity.
        // The 2nd root of unity is -1 in any field (since ω² = 1, ω ≠ 1).
        // roots[0] unused; roots[1] = -1 = p-1.
        let neg_one = Fr::from(0u64) - Fr::one();
        let fwd_roots = vec![Fr::one(), neg_one];
        // Inverse roots: same as forward for n=2 (since (-1)^{-1} = -1)
        let inv_roots = vec![Fr::one(), neg_one];
        // n_inv = 2^{-1} in Fr
        let n_inv = Fr::from(2u64).inverse().unwrap();

        let a_ntt = NttVerifierGadget::ntt(&cs, &a, &fwd_roots).unwrap();
        let a_back = NttVerifierGadget::intt(&cs, &a_ntt, &inv_roots, n_inv).unwrap();

        assert!(cs.is_satisfied().unwrap(), "NTT round-trip circuit unsatisfied");

        // Verify values are correct
        assert_eq!(a_back[0].value().unwrap(), a0, "INTT(NTT(a))[0] should equal a[0]");
        assert_eq!(a_back[1].value().unwrap(), a1, "INTT(NTT(a))[1] should equal a[1]");

        println!(
            "NTT+INTT round-trip (n=2) constraints: {}",
            cs.num_constraints()
        );
    }

    /// Test poly_mul: (1 + x)(1 + x) = 1 + 2x + x² in F[X]/(X^2 - 1).
    ///
    /// Over the cyclic ring X^2-1: x²=1, so 1 + 2x + x² = 2 + 2x.
    #[test]
    fn test_poly_mul_n2() {
        let cs = ConstraintSystem::<Fr>::new_ref();

        // a = b = [1, 1]  (polynomial 1 + x)
        let a: Vec<FpVar<Fr>> = [Fr::one(), Fr::one()]
            .iter()
            .map(|&v| FpVar::new_witness(cs.clone(), || Ok(v)).unwrap())
            .collect();
        let b: Vec<FpVar<Fr>> = [Fr::one(), Fr::one()]
            .iter()
            .map(|&v| FpVar::new_witness(cs.clone(), || Ok(v)).unwrap())
            .collect();

        let neg_one = Fr::from(0u64) - Fr::one();
        let fwd_roots = vec![Fr::one(), neg_one];
        let inv_roots = vec![Fr::one(), neg_one];
        let n_inv = Fr::from(2u64).inverse().unwrap();

        let c = NttVerifierGadget::poly_mul(
            &cs, &a, &b, &fwd_roots, &inv_roots, n_inv,
        )
        .unwrap();

        assert!(cs.is_satisfied().unwrap(), "poly_mul circuit unsatisfied");

        // In F[X]/(X^2 - 1): (1+x)^2 = 1 + 2x + x² = (1 + 1) + 2x = 2 + 2x
        assert_eq!(
            c[0].value().unwrap(),
            Fr::from(2u64),
            "poly_mul[0] should be 2"
        );
        assert_eq!(
            c[1].value().unwrap(),
            Fr::from(2u64),
            "poly_mul[1] should be 2"
        );

        println!("poly_mul (n=2) constraints: {}", cs.num_constraints());
    }
}
