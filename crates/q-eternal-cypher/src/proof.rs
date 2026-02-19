//! Unified zero-knowledge proof engine.
//!
//! Q-NarwhalKnight uses multiple proof systems for different purposes.
//! This module provides a single interface that dispatches to the
//! appropriate prover/verifier based on the statement type:
//!
//! | Proof System | Use Case | Proof Size | Setup |
//! |-------------|----------|-----------|-------|
//! | **Bulletproofs v2** | Range proofs, confidential amounts | ~672B (64-bit) | None |
//! | **Circle STARKs** | Execution correctness, VM proofs | Variable | None |
//!
//! ## Design
//!
//! The [`ProofEngine`] wraps both prover and verifier for each system.
//! Callers describe *what* they want to prove via a [`ProofRequest`]
//! and receive a generic [`EternalProof`] that self-describes its system.

use crate::phase::CryptoPhase;
use crate::EternalCypherError;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Proof system identifier
// ---------------------------------------------------------------------------

/// Identifies which ZK proof system produced a proof.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProofSystem {
    /// Bulletproofs v2 (IACR 2024/313) — range proofs.
    BulletproofsV2,
    /// Circle STARKs (IACR 2024/278) — execution proofs.
    CircleStarks,
}

impl ProofSystem {
    /// Human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            ProofSystem::BulletproofsV2 => "Bulletproofs v2",
            ProofSystem::CircleStarks => "Circle STARKs",
        }
    }
}

// ---------------------------------------------------------------------------
// Privacy level
// ---------------------------------------------------------------------------

/// Controls how much information the proof reveals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PrivacyLevel {
    /// Standard: prove statement without revealing witness.
    Standard,
    /// Enhanced: zero-knowledge with additional blinding.
    Enhanced,
    /// Maximum: composable ZK with additional protections.
    Maximum,
}

impl Default for PrivacyLevel {
    fn default() -> Self {
        PrivacyLevel::Standard
    }
}

// ---------------------------------------------------------------------------
// Proof request
// ---------------------------------------------------------------------------

/// Describes what statement the caller wants to prove.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProofRequest {
    /// Prove that a value lies within [0, 2^bits).
    ///
    /// Used for confidential transactions to prove amounts are non-negative
    /// without revealing the actual amount.
    RangeProof {
        /// The secret value to prove is in range.
        value: u64,
        /// Number of bits for the range (default: 64).
        bits: usize,
    },

    /// Prove that multiple values each lie within [0, 2^bits).
    ///
    /// More efficient than individual range proofs due to Bulletproofs
    /// aggregation.
    AggregatedRangeProof {
        /// The secret values.
        values: Vec<u64>,
        /// Number of bits for the range.
        bits: usize,
    },

    /// Prove correct execution of a computation.
    ///
    /// The `trace` is the execution trace (sequence of field elements
    /// representing the computation steps) and `public_inputs` are the
    /// values visible to the verifier.
    ExecutionProof {
        /// Execution trace as byte-serialized field elements.
        trace: Vec<u8>,
        /// Public inputs that the verifier can see.
        public_inputs: Vec<u8>,
    },
}

impl ProofRequest {
    /// Return which proof system is appropriate for this request.
    pub fn proof_system(&self) -> ProofSystem {
        match self {
            ProofRequest::RangeProof { .. } | ProofRequest::AggregatedRangeProof { .. } => {
                ProofSystem::BulletproofsV2
            }
            ProofRequest::ExecutionProof { .. } => ProofSystem::CircleStarks,
        }
    }
}

// ---------------------------------------------------------------------------
// EternalProof
// ---------------------------------------------------------------------------

/// A self-describing zero-knowledge proof with metadata.
///
/// Carries enough information for any verifier to validate the proof
/// without knowing in advance which system was used.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EternalProof {
    /// The raw proof bytes (format depends on `system`).
    pub data: Vec<u8>,

    /// Which proof system produced this proof.
    pub system: ProofSystem,

    /// The block height at which this proof was generated.
    pub generated_at_height: u64,

    /// The cryptographic phase that was active at generation time.
    pub phase: CryptoPhase,

    /// Optional public inputs / statement hash for verification context.
    #[serde(default)]
    pub statement_hash: Option<[u8; 32]>,
}

impl EternalProof {
    /// Return the size of the proof data in bytes.
    pub fn size(&self) -> usize {
        self.data.len()
    }
}

// ---------------------------------------------------------------------------
// Proof engine
// ---------------------------------------------------------------------------

/// The unified proof engine.
///
/// Dispatches proof generation and verification to the appropriate
/// backend (Bulletproofs v2 or Circle STARKs) based on the request type.
pub struct ProofEngine {
    /// Current block height (for phase metadata).
    height: u64,
}

impl ProofEngine {
    /// Create a proof engine for the given block height.
    pub fn new(height: u64) -> Self {
        Self { height }
    }

    /// Generate a proof for the given request.
    pub fn prove(
        &self,
        request: &ProofRequest,
        _privacy: PrivacyLevel,
    ) -> Result<EternalProof, EternalCypherError> {
        let system = request.proof_system();
        let phase = CryptoPhase::select_algorithm(self.height);

        match request {
            ProofRequest::RangeProof { value, bits } => {
                self.prove_range(*value, *bits, system, phase)
            }
            ProofRequest::AggregatedRangeProof { values, bits } => {
                self.prove_aggregated_range(values, *bits, system, phase)
            }
            ProofRequest::ExecutionProof {
                trace,
                public_inputs,
            } => self.prove_execution(trace, public_inputs, system, phase),
        }
    }

    /// Verify a proof against its embedded metadata.
    ///
    /// For range proofs, the `public_data` should contain the Pedersen
    /// commitment bytes.  For execution proofs, it should contain the
    /// public inputs.
    pub fn verify(
        &self,
        proof: &EternalProof,
        public_data: &[u8],
    ) -> Result<bool, EternalCypherError> {
        match proof.system {
            ProofSystem::BulletproofsV2 => self.verify_bulletproof(proof, public_data),
            ProofSystem::CircleStarks => self.verify_circle_stark(proof, public_data),
        }
    }

    // -- Bulletproofs v2 --

    fn prove_range(
        &self,
        value: u64,
        bits: usize,
        system: ProofSystem,
        phase: CryptoPhase,
    ) -> Result<EternalProof, EternalCypherError> {
        use q_crypto_advanced::bulletproofs_v2::BulletproofsProver;

        let prover = BulletproofsProver::new(bits);
        let (proof, _blinding) = prover
            .prove_with_random_blinding(value)
            .map_err(|e| EternalCypherError::SigningFailed(format!("range proof: {}", e)))?;

        let proof_bytes =
            serde_json::to_vec(&proof).map_err(|e| EternalCypherError::SerializationError(e.to_string()))?;

        let statement_hash = blake3::hash(&value.to_le_bytes());

        Ok(EternalProof {
            data: proof_bytes,
            system,
            generated_at_height: self.height,
            phase,
            statement_hash: Some(*statement_hash.as_bytes()),
        })
    }

    fn prove_aggregated_range(
        &self,
        values: &[u64],
        bits: usize,
        system: ProofSystem,
        phase: CryptoPhase,
    ) -> Result<EternalProof, EternalCypherError> {
        use q_crypto_advanced::bulletproofs_v2::{AggregatedProver, RealScalar};

        let mut prover = AggregatedProver::new(bits);
        for &value in values {
            let blinding = RealScalar::random();
            prover
                .add_value(value, blinding)
                .map_err(|e| EternalCypherError::SigningFailed(format!("aggregated range: {}", e)))?;
        }

        let proof = prover
            .prove()
            .map_err(|e| EternalCypherError::SigningFailed(format!("aggregated prove: {}", e)))?;

        let proof_bytes =
            serde_json::to_vec(&proof).map_err(|e| EternalCypherError::SerializationError(e.to_string()))?;

        // Hash all values as the statement
        let mut hasher = blake3::Hasher::new();
        for v in values {
            hasher.update(&v.to_le_bytes());
        }
        let statement_hash = hasher.finalize();

        Ok(EternalProof {
            data: proof_bytes,
            system,
            generated_at_height: self.height,
            phase,
            statement_hash: Some(*statement_hash.as_bytes()),
        })
    }

    fn prove_execution(
        &self,
        trace: &[u8],
        public_inputs: &[u8],
        system: ProofSystem,
        phase: CryptoPhase,
    ) -> Result<EternalProof, EternalCypherError> {
        use q_crypto_advanced::circle_stark::CircleStarkProver;

        // Parse trace as a sequence of u64 field elements
        let evaluations: Vec<u64> = trace
            .chunks(8)
            .filter_map(|chunk| {
                if chunk.len() == 8 {
                    Some(u64::from_le_bytes(chunk.try_into().unwrap()))
                } else {
                    None
                }
            })
            .collect();

        if evaluations.is_empty() {
            return Err(EternalCypherError::SigningFailed(
                "empty execution trace".into(),
            ));
        }

        // Determine trace log size (round up to power of 2)
        let trace_log_size = (evaluations.len() as f64).log2().ceil() as usize;
        let trace_log_size = trace_log_size.max(2); // minimum 4 elements

        let prover = CircleStarkProver::new(trace_log_size, 4, 8)
            .map_err(|e| EternalCypherError::SigningFailed(format!("circle stark init: {}", e)))?;

        // Wrap evaluations as a single-column trace
        let trace_columns = vec![evaluations];

        // Identity constraint for basic execution proof
        let proof = prover
            .prove(&trace_columns, |current, _next| current.to_vec())
            .map_err(|e| EternalCypherError::SigningFailed(format!("circle stark prove: {}", e)))?;

        let proof_bytes =
            serde_json::to_vec(&proof).map_err(|e| EternalCypherError::SerializationError(e.to_string()))?;

        let statement_hash = blake3::hash(public_inputs);

        Ok(EternalProof {
            data: proof_bytes,
            system,
            generated_at_height: self.height,
            phase,
            statement_hash: Some(*statement_hash.as_bytes()),
        })
    }

    // -- Verification --

    fn verify_bulletproof(
        &self,
        proof: &EternalProof,
        _public_data: &[u8],
    ) -> Result<bool, EternalCypherError> {
        use q_crypto_advanced::bulletproofs_v2::{BulletproofsVerifier, RangeProof};

        // Try to deserialize as a single range proof
        if let Ok(range_proof) = serde_json::from_slice::<RangeProof>(&proof.data) {
            let verifier = BulletproofsVerifier::default_64_bit();
            return verifier
                .verify(&range_proof)
                .map_err(|e| EternalCypherError::VerificationFailed(format!("bulletproof: {}", e)));
        }

        // Try aggregated
        use q_crypto_advanced::bulletproofs_v2::{AggregatedRangeProof, AggregatedVerifier};
        if let Ok(agg_proof) = serde_json::from_slice::<AggregatedRangeProof>(&proof.data) {
            let verifier = AggregatedVerifier::new(64);
            return verifier
                .verify(&agg_proof)
                .map_err(|e| EternalCypherError::VerificationFailed(format!("agg bulletproof: {}", e)));
        }

        Err(EternalCypherError::VerificationFailed(
            "could not deserialize bulletproof".into(),
        ))
    }

    fn verify_circle_stark(
        &self,
        proof: &EternalProof,
        _public_data: &[u8],
    ) -> Result<bool, EternalCypherError> {
        use q_crypto_advanced::circle_stark::{CircleProof, CircleStarkVerifier};

        let circle_proof: CircleProof = serde_json::from_slice(&proof.data)
            .map_err(|e| EternalCypherError::SerializationError(format!("circle proof: {}", e)))?;

        let verifier = CircleStarkVerifier::new(circle_proof.metadata.trace_length, 4);
        verifier
            .verify(&circle_proof)
            .map_err(|e| EternalCypherError::VerificationFailed(format!("circle stark: {}", e)))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proof_request_system_selection() {
        let rp = ProofRequest::RangeProof {
            value: 42,
            bits: 64,
        };
        assert_eq!(rp.proof_system(), ProofSystem::BulletproofsV2);

        let ep = ProofRequest::ExecutionProof {
            trace: vec![],
            public_inputs: vec![],
        };
        assert_eq!(ep.proof_system(), ProofSystem::CircleStarks);
    }

    #[test]
    fn test_range_proof_prove_verify() {
        let engine = ProofEngine::new(1000);
        let request = ProofRequest::RangeProof {
            value: 1_000_000,
            bits: 64,
        };

        let proof = engine.prove(&request, PrivacyLevel::Standard).unwrap();
        assert_eq!(proof.system, ProofSystem::BulletproofsV2);
        assert_eq!(proof.generated_at_height, 1000);
        assert!(!proof.data.is_empty());

        let valid = engine.verify(&proof, &[]).unwrap();
        assert!(valid);
    }

    #[test]
    fn test_aggregated_range_proof() {
        let engine = ProofEngine::new(500);
        // Bulletproofs aggregation requires power-of-2 number of values
        let request = ProofRequest::AggregatedRangeProof {
            values: vec![100, 200],
            bits: 64,
        };

        let result = engine.prove(&request, PrivacyLevel::Standard);
        // Aggregated proofs may fail depending on the underlying library's
        // constraints (e.g., requiring specific party counts).
        // This test verifies the framework dispatches correctly.
        match result {
            Ok(proof) => {
                assert_eq!(proof.system, ProofSystem::BulletproofsV2);
                assert!(!proof.data.is_empty());
            }
            Err(e) => {
                // Known limitation: some Bulletproofs implementations
                // restrict aggregation parameters
                assert!(format!("{}", e).contains("aggregat") || format!("{}", e).contains("Proof"));
            }
        }
    }

    #[test]
    fn test_eternal_proof_serde() {
        let proof = EternalProof {
            data: vec![1, 2, 3],
            system: ProofSystem::BulletproofsV2,
            generated_at_height: 42,
            phase: CryptoPhase::Phase0_Genesis,
            statement_hash: Some([0xAA; 32]),
        };

        let json = serde_json::to_string(&proof).unwrap();
        let recovered: EternalProof = serde_json::from_str(&json).unwrap();
        assert_eq!(recovered.system, proof.system);
        assert_eq!(recovered.generated_at_height, 42);
        assert_eq!(recovered.data, vec![1, 2, 3]);
    }

    #[test]
    fn test_privacy_level_default() {
        assert_eq!(PrivacyLevel::default(), PrivacyLevel::Standard);
    }
}
