//! STARK verifier for Shamir consistency
//!
//! Verifies proofs WITHOUT any trusted setup.
//! All randomness is derived from public transcript (Fiat-Shamir).

use winter_math::fields::f64::BaseElement;
use winter_verifier::{verify, AcceptableOptions};
use winter_crypto::{hashers::Blake3_256, DefaultRandomCoin, MerkleTree};
use winter_air::{ProofOptions, FieldExtension, proof::Proof};

use super::air::{ShamirConsistencyAir, ShamirPublicInputs};
use crate::error::{TemporalError, TemporalResult};

/// Type alias for the hasher
type WinterHasher = Blake3_256<BaseElement>;

/// Verify a STARK proof of Shamir consistency
///
/// # Arguments
/// * `proof_bytes` - Serialized STARK proof
/// * `key_commitment` - Commitment to the OTP key
/// * `share_commitments` - Commitments to each share
/// * `threshold` - k parameter
/// * `total_trustees` - n parameter
///
/// # Returns
/// Ok(()) if proof is valid, error otherwise
///
/// # Security Note
/// This verification requires NO TRUSTED SETUP.
/// All verifier randomness is derived from the public transcript.
pub fn verify_proof(
    proof_bytes: Vec<u8>,
    key_commitment: &[u8; 32],
    share_commitments: &[[u8; 32]],
    threshold: usize,
    total_trustees: usize,
) -> TemporalResult<()> {
    // Deserialize proof
    let proof = Proof::from_bytes(&proof_bytes)
        .map_err(|e| TemporalError::ProofVerificationFailed(format!("Invalid proof format: {:?}", e)))?;

    // Reconstruct public inputs
    let num_chunks = 1; // Would need to be passed or inferred
    let public_inputs = ShamirPublicInputs::from_bytes(
        key_commitment,
        share_commitments,
        threshold,
        total_trustees,
        num_chunks,
    );

    // Define acceptable proof options
    // This allows some flexibility in accepted proofs
    let acceptable = AcceptableOptions::OptionSet(vec![
        // 128-bit security
        ProofOptions::new(28, 8, 16, FieldExtension::Quadratic, 8, 127),
        // 100-bit security (faster)
        ProofOptions::new(20, 8, 12, FieldExtension::Quadratic, 8, 127),
        // Higher security
        ProofOptions::new(40, 16, 20, FieldExtension::Quadratic, 4, 63),
    ]);

    // Verify the proof
    // This is the core STARK verification - NO TRUSTED SETUP
    verify::<ShamirConsistencyAir, WinterHasher, DefaultRandomCoin<WinterHasher>, MerkleTree<WinterHasher>>(
        proof,
        public_inputs,
        &acceptable,
    )
    .map_err(|e| TemporalError::ProofVerificationFailed(format!("{:?}", e)))?;

    Ok(())
}

/// Quick verification for envelopes
pub fn verify_envelope_proof(
    envelope: &crate::envelope::TemporalEnvelope,
) -> TemporalResult<()> {
    verify_proof(
        envelope.stark_proof.clone(),
        &envelope.key_commitment,
        &envelope.share_commitments,
        envelope.metadata.threshold,
        envelope.metadata.total_trustees,
    )
}

/// Batch verify multiple proofs (more efficient than individual verification)
pub fn batch_verify_proofs(
    proofs: &[(Vec<u8>, [u8; 32], Vec<[u8; 32]>, usize, usize)],
) -> TemporalResult<Vec<bool>> {
    // For now, verify individually
    // Future optimization: use batch verification techniques
    proofs
        .iter()
        .map(|(proof_bytes, key_commitment, share_commitments, threshold, total_trustees)| {
            verify_proof(
                proof_bytes.clone(),
                key_commitment,
                share_commitments,
                *threshold,
                *total_trustees,
            )
            .map(|_| true)
            .or_else(|_| Ok(false))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stark::prover::generate_test_proof;

    #[test]
    fn test_verify_valid_proof() {
        let secret = b"test secret for verification";
        let (proof_bytes, key_commitment, share_commitments) =
            generate_test_proof(secret, 2, 3).unwrap();

        let result = verify_proof(
            proof_bytes,
            &key_commitment,
            &share_commitments,
            2,
            3,
        );

        // Should verify successfully
        assert!(result.is_ok());
    }

    #[test]
    fn test_verify_invalid_commitment() {
        let secret = b"test secret";
        let (proof_bytes, _key_commitment, share_commitments) =
            generate_test_proof(secret, 2, 3).unwrap();

        // Use wrong commitment
        let wrong_commitment = [0u8; 32];

        let result = verify_proof(
            proof_bytes,
            &wrong_commitment,
            &share_commitments,
            2,
            3,
        );

        // Should fail verification
        assert!(result.is_err());
    }
}
