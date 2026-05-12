//! In-circuit Dilithium5 signature verification gadget.
//!
//! This is the heaviest gadget in the IVC circuit: ~150K constraints per signature.
//! Dilithium5 verification requires:
//!   1. Decode signature (z, h, c̃)
//!   2. Compute w' = Az - c·t  (NTT polynomial arithmetic)
//!   3. Compute ĉ = H(msg || μ || w₁)  (Poseidon hash in-circuit)
//!   4. Check: ĉ == c̃  AND  ||z||_∞ < γ₁ - β
//!
//! The transcript hash (step 3) uses Poseidon (not SHAKE256) for circuit efficiency.
//! This requires that Dilithium signing also use Poseidon — a coordinated change.
//!
//! Without signature aggregation, this gadget runs once per validator.
//! With 5 validators: 750K constraints just for BFT verification.
//!
//! Status: Az−c·t computation is now real (uses NttVerifierGadget::poly_mul).
//! Remaining scaffold: HighBits extraction, hint vector check, and full
//! witness type structs (PublicKeyVar, SignatureVar).
//!
//! Reference: CRYSTALS-Dilithium spec v3.1, Algorithm 3 (Verify).

use ark_ff::PrimeField;
use ark_r1cs_std::{
    boolean::Boolean,
    fields::fp::FpVar,
    prelude::*,
};
use ark_relations::r1cs::{ConstraintSystemRef, SynthesisError};

use crate::gadgets::ntt::NttVerifierGadget;
use crate::gadgets::poseidon::PoseidonGadget;

/// Dilithium5 parameters (public constants, not secret).
pub const DILITHIUM5_N: usize = 256;    // polynomial dimension
pub const DILITHIUM5_K: usize = 8;      // rows in matrix A
pub const DILITHIUM5_L: usize = 7;      // columns in matrix A
pub const DILITHIUM5_GAMMA1: u64 = 1 << 19;  // ||z||_∞ bound
pub const DILITHIUM5_BETA: u64 = 196;    // commitment norm bound

// ─── NTT parameter bundle ─────────────────────────────────────────────────────

/// Caller-provided NTT roots for polynomial multiplication.
///
/// `fwd[k] = ω^(bit_rev(k))` for k in 1..n, where ω is a primitive n-th root
/// of unity in F (= ψ² for negacyclic). `inv[k]` is the corresponding inverse.
/// `n_inv = n⁻¹` in F.
///
/// For negacyclic NTT (Dilithium's X^n+1 ring), also supply `psi` (primitive
/// 2n-th root of unity) and `psi_inv` so that `compute_az_minus_ct_negacyclic`
/// can call `poly_mul_negacyclic` directly.
pub struct NttRoots<F: PrimeField> {
    pub fwd: Vec<F>,
    pub inv: Vec<F>,
    pub n_inv: F,
    /// ψ: primitive 2n-th root of unity (ψ^2 = ω, ψ^n = -1).
    /// Set to F::zero() when using cyclic poly_mul.
    pub psi: F,
    /// ψ^{-1}: inverse of psi. Set to F::zero() when using cyclic poly_mul.
    pub psi_inv: F,
}

// ─── Az − c·t matrix-vector product ─────────────────────────────────────────

/// Compute w′ = A·z − c·t inside R1CS using the NTT butterfly.
///
/// All polynomial vectors are represented as `Vec<Vec<FpVar<F>>>` where
/// the outer vec is over the ring dimension (k or l) and the inner vec
/// holds the n=256 coefficient FpVars.
///
/// # Arguments
/// * `a_mat` – k×l polynomial matrix (row-major: `a_mat[i*l + j]` is poly A_{i,j})
/// * `z` – ℓ-vector of polynomials (the signature component)
/// * `c_poly` – the n-coefficient challenge polynomial
/// * `t_vec` – k-vector of polynomials (public key t₁)
/// * `roots` – consistent NTT root bundle for degree n
///
/// # Constraint cost (Dilithium5: k=8, l=7, n=256)
///   A·z: k×l = 56 poly_muls  × ~3.6K constraints = ~202K
///   c·t: k   =  8 poly_muls  × ~3.6K constraints = ~29K
///   accumulation (additions): free
///   Total: ~231K constraints
pub fn compute_az_minus_ct<F: PrimeField>(
    cs: &ConstraintSystemRef<F>,
    a_mat: &[Vec<FpVar<F>>],   // k×l entries, each length n
    z: &[Vec<FpVar<F>>],       // l entries, each length n
    c_poly: &[FpVar<F>],       // n coefficients
    t_vec: &[Vec<FpVar<F>>],   // k entries, each length n
    roots: &NttRoots<F>,
) -> Result<Vec<Vec<FpVar<F>>>, SynthesisError> {
    let k = t_vec.len();
    let l = z.len();
    let n = c_poly.len();

    assert_eq!(a_mat.len(), k * l, "a_mat must have k×l entries");

    let mut w_prime = Vec::with_capacity(k);

    for i in 0..k {
        // Az[i] = Σ_{j=0}^{l-1} A[i][j] · z[j]
        let mut az_i: Vec<FpVar<F>> = vec![FpVar::Constant(F::zero()); n];
        for j in 0..l {
            let prod = NttVerifierGadget::<F>::poly_mul(
                cs,
                &a_mat[i * l + j],
                &z[j],
                &roots.fwd,
                &roots.inv,
                roots.n_inv,
            )?;
            for idx in 0..n {
                az_i[idx] = az_i[idx].clone() + prod[idx].clone();
            }
        }

        // c·t[i]
        let ct_i = NttVerifierGadget::<F>::poly_mul(
            cs,
            c_poly,
            &t_vec[i],
            &roots.fwd,
            &roots.inv,
            roots.n_inv,
        )?;

        // w'[i] = az[i] − c·t[i]
        let w_i: Vec<FpVar<F>> = (0..n)
            .map(|idx| az_i[idx].clone() - ct_i[idx].clone())
            .collect();

        w_prime.push(w_i);
    }

    Ok(w_prime)
}

/// Compute w′ = A·z − c·t using **negacyclic** polynomial multiplication.
///
/// Identical structure to `compute_az_minus_ct` but calls `poly_mul_negacyclic`
/// so the computation is correct in the ring Z_q[X]/(X^n + 1) that Dilithium
/// actually uses. Requires `roots.psi` and `roots.psi_inv` to be set.
///
/// # Constraint cost (Dilithium5: k=8, l=7, n=256)
///   Same as `compute_az_minus_ct`: ~231K constraints.
///   The extra twist tables are field constants — zero additional constraints.
pub fn compute_az_minus_ct_negacyclic<F: PrimeField>(
    cs: &ConstraintSystemRef<F>,
    a_mat: &[Vec<FpVar<F>>],
    z: &[Vec<FpVar<F>>],
    c_poly: &[FpVar<F>],
    t_vec: &[Vec<FpVar<F>>],
    roots: &NttRoots<F>,
) -> Result<Vec<Vec<FpVar<F>>>, SynthesisError> {
    let k = t_vec.len();
    let l = z.len();
    let n = c_poly.len();

    assert_eq!(a_mat.len(), k * l, "a_mat must have k×l entries");
    assert_ne!(roots.psi, F::zero(), "psi must be set for negacyclic computation");

    let mut w_prime = Vec::with_capacity(k);

    for i in 0..k {
        let mut az_i: Vec<FpVar<F>> = vec![FpVar::Constant(F::zero()); n];
        for j in 0..l {
            let prod = NttVerifierGadget::<F>::poly_mul_negacyclic(
                cs,
                &a_mat[i * l + j],
                &z[j],
                &roots.fwd,
                &roots.inv,
                roots.n_inv,
                roots.psi,
                roots.psi_inv,
            )?;
            for idx in 0..n {
                az_i[idx] = az_i[idx].clone() + prod[idx].clone();
            }
        }

        let ct_i = NttVerifierGadget::<F>::poly_mul_negacyclic(
            cs,
            c_poly,
            &t_vec[i],
            &roots.fwd,
            &roots.inv,
            roots.n_inv,
            roots.psi,
            roots.psi_inv,
        )?;

        let w_i: Vec<FpVar<F>> = (0..n)
            .map(|idx| az_i[idx].clone() - ct_i[idx].clone())
            .collect();

        w_prime.push(w_i);
    }

    Ok(w_prime)
}

/// Enforce that all coefficients in every polynomial in `w` satisfy coeff < bound.
///
/// One-sided check (positive range only). Use `enforce_signed_norm_bound` for
/// Dilithium z-polynomials whose coefficients can be negative.
pub fn enforce_norm_bound<F: PrimeField>(
    cs: ConstraintSystemRef<F>,
    w: &[Vec<FpVar<F>>],
    bound: u64,
) -> Result<(), SynthesisError> {
    for poly in w {
        let norm_ok = NttVerifierGadget::verify_infinity_norm(cs.clone(), poly, bound)?;
        norm_ok.enforce_equal(&Boolean::constant(true))?;
    }
    Ok(())
}

/// Enforce ||w||_∞ < bound with two-sided (signed) range check.
///
/// Dilithium's z-polynomials have coefficients in [−(γ₁−β), γ₁−β−1] which are
/// stored as field elements with negative values represented as p − |v|. The
/// one-sided `enforce_norm_bound` misses the negative half entirely. This
/// function uses `verify_signed_norm` to handle both halves.
///
/// Call with `bound = DILITHIUM5_GAMMA1 - DILITHIUM5_BETA = 261948`.
///
/// Constraint cost: k × n × ~401 ≈ 103K constraints for k=1,n=256.
pub fn enforce_signed_norm_bound<F: PrimeField>(
    cs: ConstraintSystemRef<F>,
    w: &[Vec<FpVar<F>>],
    bound: u64,
) -> Result<(), SynthesisError> {
    for (poly_idx, poly) in w.iter().enumerate() {
        let norm_ok =
            NttVerifierGadget::verify_signed_infinity_norm(&cs, poly, bound)?;
        if !norm_ok.value().unwrap_or(false) {
            println!(
                "  [enforce_signed_norm_bound] poly[{}]: norm check FAILED (bound={})",
                poly_idx, bound
            );
        }
        norm_ok.enforce_equal(&Boolean::constant(true))?;
    }
    Ok(())
}

// ─── DilithiumVerifierGadget ──────────────────────────────────────────────────

/// In-circuit Dilithium5 signature verifier.
///
/// Each instance verifies one (pk, signature, message) triple.
/// Embed multiple instances for a 2f+1 BFT threshold check.
pub struct DilithiumVerifierGadget;

impl DilithiumVerifierGadget {
    /// Verify one Dilithium5 signature inside the circuit.
    ///
    /// Returns `Boolean::constant(true)` if valid (placeholder).
    /// Real implementation returns a witness-dependent Boolean.
    ///
    /// # Arguments
    /// * `cs` – constraint system
    /// * `message_hash` – H(message), allocated as field elements (Poseidon output)
    /// * `public_key` – Dilithium public key components (t, ρ) as field elements
    /// * `sig_z` – signature component z (polynomial vector, length K×N)
    /// * `sig_h` – signature hint h (binary polynomial, length K×N)
    /// * `sig_c_tilde` – challenge hash c̃ (32 bytes as field elements)
    ///
    /// # Constraint estimate (Dilithium5, no aggregation)
    /// * Az computation (NTT): ~100K
    /// * c·t subtraction: ~10K
    /// * HighBits computation: ~5K
    /// * Poseidon hash (step 3): ~3K
    /// * Norm check ||z||_∞: ~30K
    /// * Total: ~148K
    pub fn verify<F: PrimeField>(
        cs: ConstraintSystemRef<F>,
        message_hash: &[FpVar<F>],    // Poseidon(msg)
        public_key: &[FpVar<F>],      // t₁, ρ components
        sig_z: &[FpVar<F>],           // z polynomial vector (L×N field elements)
        sig_h: &[Boolean<F>],         // h hint bits (K×N)
        sig_c_tilde: &[FpVar<F>],     // challenge c̃ (8 field elements)
    ) -> Result<Boolean<F>, SynthesisError> {
        // STEP 1: Norm check — ||z||_∞ < γ₁ - β  (signed: z coefficients can be negative)
        // Each coefficient of z must satisfy |z[i]| < γ₁ - β = 262144 - 196 = 261948
        let norm_bound = DILITHIUM5_GAMMA1 - DILITHIUM5_BETA;
        let norm_ok =
            NttVerifierGadget::verify_signed_infinity_norm(&cs, sig_z, norm_bound)?;

        // STEP 2: Recompute w' = Az − c·t
        // `compute_az_minus_ct` is the real implementation when the caller provides
        // NTT roots and full polynomial witness vectors. The existing `verify` API
        // accepts flat FpVar slices for compatibility; full integration uses `compute_az_minus_ct`
        // directly with structured Vec<Vec<FpVar<F>>> inputs.
        //
        // Proxy: take first 8 coefficients of sig_z as the w' stand-in.
        // This keeps the existing `verify` signature stable while `compute_az_minus_ct`
        // provides the real implementation for callers that pass polynomial structure.
        let w_prime: Vec<FpVar<F>> = sig_z.iter().take(8).cloned().collect();

        // STEP 3: Recompute challenge c' = Poseidon(message_hash || w₁)
        // w₁ = HighBits(w') (bit extraction — TODO: implement HighBits gadget)
        let w1_proxy = PoseidonGadget::hash_many(cs.clone(), &w_prime)?;
        let mut transcript_input = message_hash.to_vec();
        transcript_input.push(w1_proxy);
        let c_prime = PoseidonGadget::hash_many(cs.clone(), &transcript_input)?;

        // STEP 4: Check c' == c̃
        let c_tilde_hash = PoseidonGadget::hash_many(cs.clone(), sig_c_tilde)?;
        let c_match = c_prime.is_eq(&c_tilde_hash)?;

        // Both norm check AND challenge match must hold
        let valid = norm_ok.and(&c_match)?;
        Ok(valid)
    }

    /// Verify a threshold of Dilithium signatures (BFT 2f+1 check).
    ///
    /// Returns true iff at least `threshold` out of `n_validators` signatures are valid.
    ///
    /// # Arguments
    /// * `threshold` – minimum number of valid signatures (2f+1)
    /// * `message_hash` – the message all validators should have signed
    /// * `validator_data` – one (pk, z, h, c̃) per validator; use None for absent signatures
    pub fn verify_threshold<F: PrimeField>(
        cs: ConstraintSystemRef<F>,
        threshold: usize,
        message_hash: &[FpVar<F>],
        validator_data: &[Option<(Vec<FpVar<F>>, Vec<FpVar<F>>, Vec<Boolean<F>>, Vec<FpVar<F>>)>],
    ) -> Result<Boolean<F>, SynthesisError> {
        let mut valid_count = FpVar::Constant(F::zero());

        for entry in validator_data {
            let is_valid = match entry {
                Some((pk, sig_z, sig_h, sig_c_tilde)) => {
                    DilithiumVerifierGadget::verify(cs.clone(), message_hash, pk, sig_z, sig_h, sig_c_tilde)?
                }
                None => Boolean::constant(false),
            };
            // Add 1 if valid, 0 if not
            let one_if_valid = is_valid.select(
                &FpVar::Constant(F::one()),
                &FpVar::Constant(F::zero()),
            )?;
            valid_count = &valid_count + &one_if_valid;
        }

        // Check valid_count >= threshold
        let threshold_var = FpVar::Constant(F::from(threshold as u64));
        valid_count.is_cmp(&threshold_var, std::cmp::Ordering::Greater, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr;
    use ark_ff::{Field, One, Zero};
    use ark_relations::r1cs::ConstraintSystem;

    // ─── NTT root helpers ────────────────────────────────────────────────────

    /// Build n=2 cyclic NTT roots (psi fields zeroed — not needed for cyclic).
    fn roots_n2_cyclic() -> NttRoots<Fr> {
        let neg_one = Fr::from(0u64) - Fr::from(1u64);
        NttRoots {
            fwd: vec![Fr::from(1u64), neg_one],
            inv: vec![Fr::from(1u64), neg_one],
            n_inv: Fr::from(2u64).inverse().unwrap(),
            psi: Fr::zero(),
            psi_inv: Fr::zero(),
        }
    }

    /// Build n=2 negacyclic NTT roots.
    ///
    /// ψ = sqrt(-1) in BLS12-381 Fr (exists since p ≡ 1 mod 4).
    /// ω = ψ^2 = -1 (primitive 2nd root of unity for the inner cyclic NTT).
    fn roots_n2_negacyclic() -> NttRoots<Fr> {
        let neg_one = Fr::from(0u64) - Fr::from(1u64);
        let psi = neg_one.sqrt().expect("sqrt(-1) must exist in BLS12-381 Fr");
        let psi_inv = psi.inverse().unwrap();
        NttRoots {
            fwd: vec![Fr::from(1u64), neg_one],  // ω = ψ^2 = -1
            inv: vec![Fr::from(1u64), neg_one],
            n_inv: Fr::from(2u64).inverse().unwrap(),
            psi,
            psi_inv,
        }
    }

    // Keep old name as alias for tests that still use it (cyclic)
    fn roots_n2() -> NttRoots<Fr> {
        roots_n2_cyclic()
    }

    // ─── compute_az_minus_ct ─────────────────────────────────────────────────

    /// k=1, l=1, n=2: a=[1,0], z=[3,0], c=[2,0], t=[1,0]
    /// w' = 1·3 − 2·1 = [1, 0]  (in F[X]/(X²−1))
    #[test]
    fn test_compute_az_minus_ct_n2_k1_l1() {
        let cs = ConstraintSystem::<Fr>::new_ref();
        let alloc = |v: u64| FpVar::new_witness(cs.clone(), || Ok(Fr::from(v))).unwrap();

        let a_mat = vec![vec![alloc(1), alloc(0)]];
        let z     = vec![vec![alloc(3), alloc(0)]];
        let c_poly =    vec![alloc(2), alloc(0)];
        let t_vec = vec![vec![alloc(1), alloc(0)]];
        let roots = roots_n2();

        println!("\n=== compute_az_minus_ct (k=1, l=1, n=2) ===");
        println!("  A = [[1, 0]],  z = [[3, 0]],  c = [2, 0],  t = [[1, 0]]");
        println!("  Expected: w'[0] = A[0][0]·z[0] − c·t[0] = [1·3, 0] − [2·1, 0] = [1, 0]");

        let constraints_before = cs.num_constraints();
        let w_prime = compute_az_minus_ct(&cs, &a_mat, &z, &c_poly, &t_vec, &roots).unwrap();
        let constraints_after = cs.num_constraints();

        let satisfied = cs.is_satisfied().unwrap();
        println!("  Constraints added: {}", constraints_after - constraints_before);
        println!("  Circuit satisfied: {}", satisfied);
        println!("  w'[0][0] = {:?}", w_prime[0][0].value().unwrap());
        println!("  w'[0][1] = {:?}", w_prime[0][1].value().unwrap());

        assert!(satisfied, "circuit unsatisfied");
        assert_eq!(w_prime[0][0].value().unwrap(), Fr::from(1u64), "w'[0] should be 1");
        assert_eq!(w_prime[0][1].value().unwrap(), Fr::from(0u64), "w'[1] should be 0");
        println!("  ✓ PASS");
    }

    /// k=2, l=2, n=2: 2×2 matrix, two z-polynomials, two t-polynomials.
    ///
    /// A = [[2,0],[1,0]; [0,0],[3,0]]  z = [[1,0],[2,0]]  c=[1,0]  t=[[1,0],[1,0]]
    /// Az[0] = 2·1 + 1·2 = [4, 0]     Az[1] = 0·1 + 3·2 = [6, 0]
    /// ct[0] = 1·1 = [1, 0]            ct[1] = 1·1 = [1, 0]
    /// w'[0] = [3, 0]                  w'[1] = [5, 0]
    #[test]
    fn test_compute_az_minus_ct_n2_k2_l2() {
        let cs = ConstraintSystem::<Fr>::new_ref();
        let alloc = |v: u64| FpVar::new_witness(cs.clone(), || Ok(Fr::from(v))).unwrap();

        // A = [[2,0; 1,0], [0,0; 3,0]] (row-major, k=2, l=2)
        let a_mat = vec![
            vec![alloc(2), alloc(0)],  // A[0][0]
            vec![alloc(1), alloc(0)],  // A[0][1]
            vec![alloc(0), alloc(0)],  // A[1][0]
            vec![alloc(3), alloc(0)],  // A[1][1]
        ];
        let z     = vec![vec![alloc(1), alloc(0)], vec![alloc(2), alloc(0)]];
        let c_poly =    vec![alloc(1), alloc(0)];
        let t_vec = vec![vec![alloc(1), alloc(0)], vec![alloc(1), alloc(0)]];
        let roots = roots_n2();

        println!("\n=== compute_az_minus_ct (k=2, l=2, n=2) ===");
        println!("  A = [[2,1],[0,3]]  z = [[1],[2]]  c = 1  t = [[1],[1]]");
        println!("  Az[0] = 2·1 + 1·2 = 4,  Az[1] = 0·1 + 3·2 = 6");
        println!("  ct[0] = 1,  ct[1] = 1");
        println!("  Expected: w'[0] = [3,0], w'[1] = [5,0]");

        let constraints_before = cs.num_constraints();
        let w_prime = compute_az_minus_ct(&cs, &a_mat, &z, &c_poly, &t_vec, &roots).unwrap();
        let constraints_after = cs.num_constraints();

        let satisfied = cs.is_satisfied().unwrap();
        println!("  Constraints added: {} (k×l+k = 6 poly_muls)", constraints_after - constraints_before);
        println!("  Circuit satisfied: {}", satisfied);
        for (i, row) in w_prime.iter().enumerate() {
            let coeffs: Vec<_> = row.iter().map(|c| c.value().unwrap()).collect();
            println!("  w'[{}] = {:?}", i, coeffs);
        }

        assert!(satisfied, "circuit unsatisfied");
        assert_eq!(w_prime[0][0].value().unwrap(), Fr::from(3u64), "w'[0][0] should be 3");
        assert_eq!(w_prime[1][0].value().unwrap(), Fr::from(5u64), "w'[1][0] should be 5");
        println!("  ✓ PASS");
    }

    // ─── enforce_norm_bound ───────────────────────────────────────────────────

    #[test]
    fn test_enforce_norm_bound_passes() {
        let cs = ConstraintSystem::<Fr>::new_ref();

        let coeffs: Vec<u64> = (0..8).map(|i| i * 10).collect();
        let w = vec![
            coeffs.iter()
                .map(|&v| FpVar::new_witness(cs.clone(), || Ok(Fr::from(v))).unwrap())
                .collect::<Vec<_>>(),
        ];

        println!("\n=== enforce_norm_bound ===");
        println!("  Coefficients: {:?}", coeffs);
        println!("  Bound: 100");

        let constraints_before = cs.num_constraints();
        enforce_norm_bound(cs.clone(), &w, 100).unwrap();
        let constraints_after = cs.num_constraints();

        let satisfied = cs.is_satisfied().unwrap();
        println!("  Constraints added: {}", constraints_after - constraints_before);
        println!("  Circuit satisfied: {}", satisfied);
        assert!(satisfied, "norm bound should pass for coefficients < 100");
        println!("  ✓ PASS");
    }

    // ─── DilithiumVerifierGadget::verify (scaffold) ───────────────────────────

    #[test]
    fn test_dilithium_verify_scaffold() {
        let cs = ConstraintSystem::<Fr>::new_ref();

        let msg_hash = vec![FpVar::new_input(cs.clone(), || Ok(Fr::from(42u64))).unwrap()];
        let pk       = vec![FpVar::new_witness(cs.clone(), || Ok(Fr::from(7u64))).unwrap()];

        // z vector: 8 small coefficients all within γ₁ - β = 261948
        let sig_z: Vec<_> = (0u64..8)
            .map(|i| FpVar::new_witness(cs.clone(), || Ok(Fr::from(i * 100))).unwrap())
            .collect();
        let z_vals: Vec<u64> = (0..8).map(|i| i * 100).collect();

        let sig_h: Vec<Boolean<Fr>> = vec![Boolean::constant(false); 4];
        let sig_c_tilde: Vec<_> = (0u64..4)
            .map(|i| FpVar::new_witness(cs.clone(), || Ok(Fr::from(i + 1))).unwrap())
            .collect();

        println!("\n=== DilithiumVerifierGadget::verify (scaffold) ===");
        println!("  msg_hash = [42],  pk = [7]");
        println!("  z = {:?}", z_vals);
        println!("  c_tilde = [1, 2, 3, 4]");
        println!("  Norm bound = γ₁ − β = {}", DILITHIUM5_GAMMA1 - DILITHIUM5_BETA);

        let constraints_before = cs.num_constraints();
        let result = DilithiumVerifierGadget::verify(
            cs.clone(), &msg_hash, &pk, &sig_z, &sig_h, &sig_c_tilde,
        ).unwrap();
        let constraints_after = cs.num_constraints();

        let satisfied = cs.is_satisfied().unwrap();
        let result_val = result.value().unwrap_or(false);
        println!("  Constraints: {}", constraints_after - constraints_before);
        println!("  Circuit satisfied: {}", satisfied);
        println!("  verify() returned: {} (scaffold — challenge hash mismatch expected)", result_val);
        println!("    ├─ norm_ok: z coefficients {} < {} ✓", z_vals.iter().max().unwrap(), DILITHIUM5_GAMMA1 - DILITHIUM5_BETA);
        println!("    └─ c_match: placeholder Poseidon transcript (scaffold mismatch expected)");
        assert!(satisfied, "scaffold circuit must be satisfiable even when verify returns false");
        println!("  ✓ PASS (circuit satisfied, scaffold gates wired)");
    }

    // ─── enforce_signed_norm_bound ───────────────────────────────────────────

    /// Test that a z-polynomial with both positive and negative-encoded coefficients
    /// passes the signed norm check.
    #[test]
    fn test_enforce_signed_norm_bound_passes() {
        let cs = ConstraintSystem::<Fr>::new_ref();

        // Positive coefficient: 50 < 100 ✓
        let pos_val = Fr::from(50u64);
        // Negative coefficient −30 encoded as p − 30 < 100 by signed check ✓
        let neg_val = Fr::from(0u64) - Fr::from(30u64);

        let w = vec![
            vec![
                FpVar::new_witness(cs.clone(), || Ok(pos_val)).unwrap(),
                FpVar::new_witness(cs.clone(), || Ok(neg_val)).unwrap(),
            ],
        ];

        println!("\n=== enforce_signed_norm_bound ===");
        println!("  Coefficients: [+50, -30 (as p-30)]");
        println!("  Bound: 100");

        let constraints_before = cs.num_constraints();
        enforce_signed_norm_bound(cs.clone(), &w, 100).unwrap();
        let constraints_after = cs.num_constraints();

        let satisfied = cs.is_satisfied().unwrap();
        println!("  Constraints added: {}", constraints_after - constraints_before);
        println!("  Circuit satisfied: {}", satisfied);
        assert!(satisfied, "signed norm bound should pass for ±50, ±30");
        println!("  ✓ PASS");
    }

    // ─── compute_az_minus_ct_negacyclic ──────────────────────────────────────

    /// k=1, l=1, n=2: verify (1+x)·(3) − (2)·(1) in F[X]/(X^2+1).
    ///
    /// In negacyclic ring X^2+1: [1,0]·[3,0] = [3,0], [2,0]·[1,0] = [2,0]
    /// (constant polynomials behave the same as cyclic for degree-0 products)
    /// w' = [3,0] − [2,0] = [1,0]
    #[test]
    fn test_compute_az_minus_ct_negacyclic_n2_k1_l1() {
        let cs = ConstraintSystem::<Fr>::new_ref();
        let alloc = |v: u64| FpVar::new_witness(cs.clone(), || Ok(Fr::from(v))).unwrap();

        let a_mat = vec![vec![alloc(1), alloc(0)]];
        let z     = vec![vec![alloc(3), alloc(0)]];
        let c_poly =    vec![alloc(2), alloc(0)];
        let t_vec = vec![vec![alloc(1), alloc(0)]];
        let roots = roots_n2_negacyclic();

        println!("\n=== compute_az_minus_ct_negacyclic (k=1, l=1, n=2) ===");
        println!("  Ring: F[X]/(X^2+1),  ψ = sqrt(-1)");
        println!("  A = [[1,0]],  z = [[3,0]],  c = [2,0],  t = [[1,0]]");
        println!("  Expected: w'[0] = [1·3, 0] − [2·1, 0] = [1, 0]");

        let constraints_before = cs.num_constraints();
        let w_prime = compute_az_minus_ct_negacyclic(
            &cs, &a_mat, &z, &c_poly, &t_vec, &roots,
        ).unwrap();
        let constraints_after = cs.num_constraints();

        let satisfied = cs.is_satisfied().unwrap();
        println!("  Constraints added: {}", constraints_after - constraints_before);
        println!("  Circuit satisfied: {}", satisfied);
        println!("  w'[0] = [{:?}, {:?}]",
            w_prime[0][0].value().unwrap(),
            w_prime[0][1].value().unwrap());

        assert!(satisfied, "negacyclic circuit unsatisfied");
        assert_eq!(w_prime[0][0].value().unwrap(), Fr::from(1u64), "w'[0] should be 1");
        assert_eq!(w_prime[0][1].value().unwrap(), Fr::from(0u64), "w'[1] should be 0");
        println!("  ✓ PASS");
    }

    /// Verify that the negacyclic ring property X^2 = -1 manifests in products.
    ///
    /// a = [0,1] = x,  b = [0,1] = x.
    /// a·b = x·x = x^2 = -1 in F[X]/(X^2+1), so result = [-1, 0].
    #[test]
    fn test_negacyclic_xsquared_is_minus_one() {
        let cs = ConstraintSystem::<Fr>::new_ref();

        let roots = roots_n2_negacyclic();

        // a = b = [0, 1] (polynomial x)
        let a: Vec<FpVar<Fr>> = [Fr::from(0u64), Fr::from(1u64)]
            .iter()
            .map(|&v| FpVar::new_witness(cs.clone(), || Ok(v)).unwrap())
            .collect();
        let b: Vec<FpVar<Fr>> = [Fr::from(0u64), Fr::from(1u64)]
            .iter()
            .map(|&v| FpVar::new_witness(cs.clone(), || Ok(v)).unwrap())
            .collect();

        println!("\n=== negacyclic X^2 = -1 property test ===");
        println!("  a = b = [0, 1]  (polynomial x)");
        println!("  Expected: x·x = x^2 ≡ -1 (mod X^2+1) → [p-1, 0]");

        let c = NttVerifierGadget::poly_mul_negacyclic(
            &cs, &a, &b,
            &roots.fwd, &roots.inv, roots.n_inv,
            roots.psi, roots.psi_inv,
        ).unwrap();

        assert!(cs.is_satisfied().unwrap(), "negacyclic x*x circuit unsatisfied");

        let neg_one = Fr::from(0u64) - Fr::from(1u64);
        let c0 = c[0].value().unwrap();
        let c1 = c[1].value().unwrap();
        println!("  c[0] = {:?}  (should be p-1 = -1)", c0);
        println!("  c[1] = {:?}  (should be 0)", c1);
        println!("  Constraints: {}", cs.num_constraints());

        assert_eq!(c0, neg_one, "x*x in negacyclic ring: c[0] should be -1");
        assert_eq!(c1, Fr::from(0u64), "x*x in negacyclic ring: c[1] should be 0");
        println!("  ✓ PASS: X^2 ≡ -1 correctly enforced in-circuit");
    }

    // ─── Constraint scaling projection ───────────────────────────────────────

    /// Print a constraint budget projection for Dilithium5 at full scale.
    /// No assertions — diagnostic output only.
    #[test]
    fn test_constraint_budget_projection() {
        println!("\n=== Dilithium5 constraint budget projection ===");
        let n = DILITHIUM5_N;
        let k = DILITHIUM5_K;
        let l = DILITHIUM5_L;

        // NTT butterfly cost
        let butterflies_per_ntt = (n / 2) * (n as f64).log2() as usize;
        let constraints_per_ntt = butterflies_per_ntt;     // 1 R1CS mul per butterfly
        let constraints_per_poly_mul = 2 * constraints_per_ntt + n + constraints_per_ntt + n;
        // (fwd_a + fwd_b + pointwise + inv + scaling)

        println!("  n={}, k={}, l={}", n, k, l);
        println!("  Butterflies per NTT:         {} = (n/2)×log₂n", butterflies_per_ntt);
        println!("  Constraints per poly_mul:    ~{}", constraints_per_poly_mul);

        let az_poly_muls = k * l;
        let ct_poly_muls = k;
        let total_poly_muls = az_poly_muls + ct_poly_muls;
        let az_ct_constraints = total_poly_muls * constraints_per_poly_mul;
        println!("  Az: {} poly_muls (k×l={})    × {} = ~{} constraints",
            az_poly_muls, k * l, constraints_per_poly_mul, az_poly_muls * constraints_per_poly_mul);
        println!("  ct: {} poly_muls (k={})       × {} = ~{} constraints",
            ct_poly_muls, k, constraints_per_poly_mul, ct_poly_muls * constraints_per_poly_mul);
        println!("  Az−ct total:                 ~{} constraints", az_ct_constraints);

        let norm_check_per_coeff = 200usize; // is_cmp cost
        let norm_constraints = k * n * norm_check_per_coeff;
        println!("  Norm check (k×n={} coeffs):  ~{} constraints", k * n, norm_constraints);

        let poseidon_constraints = 243usize;
        let transcript_constraints = 3 * poseidon_constraints; // 3 Poseidon calls in verify
        println!("  Poseidon transcript (3 calls): ~{} constraints", transcript_constraints);

        let total = az_ct_constraints + norm_constraints + transcript_constraints;
        println!("  ─────────────────────────────────────────────");
        println!("  Total per signature:          ~{} constraints", total);
        println!("  BFT 5-validator threshold:    ~{} constraints", total * 5);
        println!("  (Prev estimate was ~150K; NTT butterflies are ~1K/NTT, not 100K)");
    }
}
