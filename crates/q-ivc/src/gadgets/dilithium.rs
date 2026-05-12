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
//! Status: SCAFFOLD — structure defined, arithmetic constraints are TODO.
//! The matrix-vector multiplication (Az) is the dominant cost and requires
//! the NTT gadget to be implemented first.
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
        // STEP 1: Norm check — ||z||_∞ < γ₁ - β
        // Each coefficient of z must satisfy |z[i]| < γ₁ - β = 262144 - 196 = 261948
        let norm_bound = DILITHIUM5_GAMMA1 - DILITHIUM5_BETA;
        let norm_ok = NttVerifierGadget::verify_infinity_norm(cs.clone(), sig_z, norm_bound)?;

        // STEP 2: Recompute w' = NTT_A(z) - NTT_c(t₁)  (polynomial arithmetic)
        // TODO: Implement NTT matrix-vector multiply for A·z and c·t.
        // This is the core NTT gadget call.
        //
        // Placeholder: derive a proxy w' from the inputs for wiring demonstration.
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
    use ark_relations::r1cs::ConstraintSystem;

    #[test]
    fn test_dilithium_gadget_compiles() {
        let cs = ConstraintSystem::<Fr>::new_ref();

        let msg_hash = vec![
            FpVar::new_input(cs.clone(), || Ok(Fr::from(1u64))).unwrap(),
        ];
        let pk = vec![FpVar::new_witness(cs.clone(), || Ok(Fr::from(2u64))).unwrap()];
        // z vector (shortened for test)
        let sig_z: Vec<_> = (0..8)
            .map(|i| FpVar::new_witness(cs.clone(), || Ok(Fr::from(i as u64))).unwrap())
            .collect();
        let sig_h: Vec<Boolean<Fr>> = vec![Boolean::constant(false); 4];
        let sig_c_tilde: Vec<_> = (0..4)
            .map(|i| FpVar::new_witness(cs.clone(), || Ok(Fr::from(i as u64))).unwrap())
            .collect();

        let result = DilithiumVerifierGadget::verify(
            cs.clone(), &msg_hash, &pk, &sig_z, &sig_h, &sig_c_tilde,
        ).unwrap();

        // Placeholder always returns satisfiable (norm check passes trivially)
        println!("Dilithium gadget scaffold constraint count: {}", cs.num_constraints());
        assert!(cs.is_satisfied().unwrap(), "Dilithium gadget placeholder should be satisfiable");
    }
}
