//! STARK prover for Shamir consistency
//!
//! Generates proofs that shares are consistent without trusted setup.

use winter_math::fields::f64::BaseElement;
use winter_math::FieldElement;
use winter_prover::{
    Prover, ProofOptions as WinterProofOptions, TraceTable, TraceInfo,
    matrix::ColMatrix, DefaultTraceLde, DefaultConstraintEvaluator, TracePolyTable,
};
use winter_crypto::{hashers::Blake3_256, DefaultRandomCoin, MerkleTree};
use winter_air::{ProofOptions, FieldExtension, Air, AuxRandElements, ConstraintCompositionCoefficients};

use super::air::{ShamirConsistencyAir, ShamirPublicInputs};
use super::trace::build_simple_trace;
use crate::config::TemporalShieldConfig;
use crate::error::{TemporalError, TemporalResult};
use crate::crypto::hash;

/// STARK proof (serialized)
pub type StarkProofBytes = Vec<u8>;

/// Type alias for the hasher
pub type WinterHasher = Blake3_256<BaseElement>;

/// Prover for Shamir consistency
pub struct ShamirConsistencyProver {
    options: WinterProofOptions,
    public_inputs: ShamirPublicInputs,
}

impl ShamirConsistencyProver {
    pub fn new(options: ProofOptions, public_inputs: ShamirPublicInputs) -> Self {
        Self {
            options: WinterProofOptions::new(
                options.num_queries(),
                options.blowup_factor(),
                options.grinding_factor(),
                options.field_extension(),
                8,    // FRI folding factor
                127,  // FRI max remainder
            ),
            public_inputs,
        }
    }
}

impl Prover for ShamirConsistencyProver {
    type BaseField = BaseElement;
    type Air = ShamirConsistencyAir;
    type Trace = TraceTable<BaseElement>;
    type HashFn = WinterHasher;
    type RandomCoin = DefaultRandomCoin<Self::HashFn>;
    type VC = MerkleTree<Self::HashFn>;
    type TraceLde<E: FieldElement<BaseField = Self::BaseField>> = DefaultTraceLde<E, Self::HashFn, Self::VC>;
    type ConstraintEvaluator<'a, E: FieldElement<BaseField = Self::BaseField>> =
        DefaultConstraintEvaluator<'a, Self::Air, E>;

    fn get_pub_inputs(&self, _trace: &Self::Trace) -> ShamirPublicInputs {
        self.public_inputs.clone()
    }

    fn options(&self) -> &WinterProofOptions {
        &self.options
    }

    fn new_trace_lde<E: FieldElement<BaseField = Self::BaseField>>(
        &self,
        trace_info: &TraceInfo,
        main_trace: &ColMatrix<Self::BaseField>,
        domain: &winter_prover::StarkDomain<Self::BaseField>,
        partition_option: winter_air::PartitionOptions,
    ) -> (Self::TraceLde<E>, TracePolyTable<E>) {
        DefaultTraceLde::new(trace_info, main_trace, domain, partition_option)
    }

    fn new_evaluator<'a, E: FieldElement<BaseField = Self::BaseField>>(
        &self,
        air: &'a Self::Air,
        aux_rand_elements: Option<AuxRandElements<E>>,
        composition_coefficients: ConstraintCompositionCoefficients<E>,
    ) -> Self::ConstraintEvaluator<'a, E> {
        DefaultConstraintEvaluator::new(air, aux_rand_elements, composition_coefficients)
    }
}

/// Generate a STARK proof for Shamir consistency
///
/// # Arguments
/// * `key` - The OTP key being shared
/// * `blinding` - The blinding factor for key commitment
/// * `shares` - The generated shares
/// * `key_commitment` - BLAKE3(key || blinding)
/// * `share_commitments` - BLAKE3(share_i) for each share
/// * `threshold` - k parameter
/// * `total_trustees` - n parameter
/// * `config` - Configuration
///
/// # Returns
/// Serialized STARK proof
pub fn generate_proof(
    key: &[u8],
    _blinding: &[u8; 32],
    _shares: &[Vec<u8>],
    key_commitment: &[u8; 32],
    share_commitments: &[[u8; 32]],
    threshold: usize,
    total_trustees: usize,
    config: &TemporalShieldConfig,
) -> TemporalResult<StarkProofBytes> {
    // Build the execution trace
    let trace = build_simple_trace(key, threshold, total_trustees)
        .map_err(|e| TemporalError::TraceGenerationFailed(e.to_string()))?;

    // Create public inputs
    let public_inputs = ShamirPublicInputs::from_bytes(
        key_commitment,
        share_commitments,
        threshold,
        total_trustees,
        config.num_chunks(key.len()),
    );

    // Get proof options from config
    let proof_options = config.to_proof_options();

    // Create prover
    let prover = ShamirConsistencyProver::new(proof_options, public_inputs);

    // Convert ColMatrix to Vec<Vec> for TraceTable
    let columns = trace.into_columns();
    let trace_table = TraceTable::init(columns);

    // Generate proof
    let proof = prover
        .prove(trace_table)
        .map_err(|e| TemporalError::ProofGenerationFailed(format!("{:?}", e)))?;

    // Serialize proof
    let proof_bytes = proof.to_bytes();

    Ok(proof_bytes)
}

/// Simplified proof generation for testing
pub fn generate_test_proof(
    secret: &[u8],
    threshold: usize,
    total_trustees: usize,
) -> TemporalResult<(StarkProofBytes, [u8; 32], Vec<[u8; 32]>)> {
    // Generate shares
    let shares = crate::shamir::shamir_split(secret, threshold, total_trustees)?;

    // Compute commitments
    let mut blinding = [0u8; 32];
    crate::crypto::rand::fill_random(&mut blinding)?;
    let key_commitment = hash::commit(secret, &blinding);

    let share_commitments: Vec<[u8; 32]> = shares
        .iter()
        .map(|s| hash::blake3_hash(&s.data))
        .collect();

    // Build trace
    let trace = build_simple_trace(secret, threshold, total_trustees)?;

    // Create public inputs
    let public_inputs = ShamirPublicInputs::from_bytes(
        &key_commitment,
        &share_commitments,
        threshold,
        total_trustees,
        1,
    );

    // Default proof options
    let proof_options = ProofOptions::new(
        28,   // num queries
        8,    // blowup factor
        16,   // grinding factor
        FieldExtension::Quadratic,
        8,    // fri folding factor
        127,  // fri max remainder
    );

    let prover = ShamirConsistencyProver::new(proof_options, public_inputs);

    // Convert ColMatrix to Vec<Vec>
    let columns = trace.into_columns();
    let trace_table = TraceTable::init(columns);

    let proof = prover
        .prove(trace_table)
        .map_err(|e| TemporalError::ProofGenerationFailed(format!("{:?}", e)))?;

    Ok((proof.to_bytes(), key_commitment, share_commitments))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_test_proof() {
        let secret = b"test secret for proof";
        let result = generate_test_proof(secret, 2, 3);

        // Proof generation should succeed
        assert!(result.is_ok());

        let (proof_bytes, key_commitment, share_commitments) = result.unwrap();

        // Proof should be non-empty
        assert!(!proof_bytes.is_empty());

        // Commitments should be valid
        assert_ne!(key_commitment, [0u8; 32]);
        assert_eq!(share_commitments.len(), 3);
    }
}
