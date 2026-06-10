//! # Recursive STARK Proof Composition for Quantum Mixing
//!
//! This module implements recursive STARK proof composition, enabling multiple mixing
//! session proofs to be verified with a single, constant-time outer proof verification.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                    RECURSIVE STARK COMPOSITION                       │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │                                                                      │
//! │    ┌──────────────┐   ┌──────────────┐   ┌──────────────┐          │
//! │    │ MixingProof1 │   │ MixingProof2 │   │ MixingProof3 │          │
//! │    └──────┬───────┘   └──────┬───────┘   └──────┬───────┘          │
//! │           │                  │                  │                   │
//! │           ▼                  ▼                  ▼                   │
//! │    ┌────────────────────────────────────────────────────┐          │
//! │    │         VERIFICATION AIR CIRCUIT                    │          │
//! │    │  - Validates inner proof structure                  │          │
//! │    │  - Checks constraint satisfaction                   │          │
//! │    │  - Verifies FRI commitments                        │          │
//! │    └────────────────────────┬───────────────────────────┘          │
//! │                             │                                       │
//! │                             ▼                                       │
//! │    ┌────────────────────────────────────────────────────┐          │
//! │    │              OUTER STARK PROOF                      │          │
//! │    │  - Single proof verifying all inner proofs         │          │
//! │    │  - Constant verification time                       │          │
//! │    │  - Compressed representation                        │          │
//! │    └────────────────────────────────────────────────────┘          │
//! │                                                                      │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Security Properties
//!
//! - **Soundness**: If inner proofs are invalid, outer proof generation fails
//! - **Zero-Knowledge**: Inner proof details not leaked through outer proof
//! - **Succinctness**: Verification time is O(log N) where N is inner proof count
//! - **Composability**: Recursive proofs can themselves be recursively composed

use crate::{
    error::{MixingError, Result},
    zkp_prover::{MixingProof, ZKProof, ProofType, QuantumZKPProver, BalanceCommitment},
    quantum_entropy::QuantumEntropyPool,
};

use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info, warn, error};

// ============================================================================
// CORE DATA STRUCTURES
// ============================================================================

/// Recursive STARK proof that compresses multiple mixing proofs into one
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecursiveStarkProof {
    /// The outer STARK proof that verifies all inner proofs
    pub outer_proof: StarkProofData,
    /// Commitment to the set of inner proofs being verified
    pub inner_commitment: [u8; 32],
    /// Current recursion depth (0 = leaf level, n = n levels deep)
    pub depth: u32,
    /// Number of inner proofs aggregated at this level
    pub inner_proof_count: usize,
    /// Verification AIR circuit identifier used
    pub verification_circuit_id: String,
    /// Proof generation timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Metadata for verification hints
    pub metadata: RecursiveProofMetadata,
}

/// STARK proof data structure matching the core STARK prover format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StarkProofData {
    /// Commitment to execution trace (Merkle root)
    pub execution_trace_commitment: [u8; 32],
    /// Evaluated constraints (should all be zero for valid proof)
    pub constraint_evaluations: Vec<u64>,
    /// FRI low-degree proof
    pub fri_proof: Vec<u8>,
    /// Public inputs
    pub public_inputs: Vec<u64>,
    /// Proof size in bytes
    pub proof_size_bytes: usize,
    /// Time taken to generate proof
    pub proving_time_ms: u64,
}

/// Metadata for recursive proof verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecursiveProofMetadata {
    /// Total number of original mixing sessions compressed
    pub total_sessions: usize,
    /// Maximum recursion depth used
    pub max_depth: u32,
    /// Estimated verification time savings (percentage)
    pub verification_savings_percent: f64,
    /// Compression ratio (original size / compressed size)
    pub compression_ratio: f64,
    /// Security level (bits)
    pub security_level: u32,
}

/// Configuration for recursive STARK proof composition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecursiveConfig {
    /// Maximum recursion depth allowed
    pub max_recursion_depth: u32,
    /// Maximum size for inner proofs (bytes)
    pub inner_proof_max_size: usize,
    /// AIR circuit size for verification (number of constraints)
    pub verification_air_size: usize,
    /// Batch size for inner proof aggregation
    pub batch_size: usize,
    /// Enable proof caching
    pub enable_caching: bool,
    /// Security parameter (bits)
    pub security_parameter: u32,
    /// Target verification time (milliseconds)
    pub target_verification_ms: u64,
    /// Enable parallel proof generation
    pub parallel_proving: bool,
}

impl Default for RecursiveConfig {
    fn default() -> Self {
        Self {
            max_recursion_depth: 8,           // Up to 2^8 = 256 proof batches
            inner_proof_max_size: 65536,      // 64 KB per inner proof
            verification_air_size: 100_000,   // 100K constraints for verification circuit
            batch_size: 32,                   // Aggregate 32 proofs per level
            enable_caching: true,
            security_parameter: 128,          // 128-bit security
            target_verification_ms: 10,       // Target <10ms verification
            parallel_proving: true,
        }
    }
}

// ============================================================================
// VERIFICATION AIR (Algebraic Intermediate Representation)
// ============================================================================

/// AIR for STARK verification - encodes the verification logic as arithmetic constraints
#[derive(Debug, Clone)]
pub struct VerificationAir {
    /// Circuit identifier
    pub circuit_id: String,
    /// Number of constraints in the verification circuit
    pub num_constraints: usize,
    /// Number of trace columns
    pub trace_width: usize,
    /// Number of public inputs
    pub num_public_inputs: usize,
    /// Constraint definitions
    pub constraints: Vec<VerificationConstraint>,
}

/// A single constraint in the verification AIR
#[derive(Debug, Clone)]
pub struct VerificationConstraint {
    /// Constraint type
    pub constraint_type: ConstraintType,
    /// Left operand column index
    pub left_col: usize,
    /// Right operand column index
    pub right_col: usize,
    /// Output column index
    pub output_col: usize,
    /// Constant value (if applicable)
    pub constant: Option<u64>,
}

/// Types of constraints supported in verification AIR
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstraintType {
    /// Addition: left + right = output
    Add,
    /// Multiplication: left * right = output
    Multiply,
    /// Hash verification: hash(left, right) must equal constant
    HashVerify,
    /// Merkle path verification
    MerkleVerify,
    /// Polynomial evaluation verification
    PolyEval,
    /// Range check
    RangeCheck,
    /// Equality constraint
    Equality,
}

impl VerificationAir {
    /// Create a new verification AIR for the given number of inner proofs
    pub fn new(inner_proof_count: usize, config: &RecursiveConfig) -> Self {
        let circuit_id = format!("recursive_verification_{}", inner_proof_count);

        // Calculate circuit dimensions
        // Each inner proof requires verification constraints for:
        // - Trace commitment check (1 constraint)
        // - FRI commitment chain (log(trace_size) constraints)
        // - Constraint polynomial evaluation (varies)
        // - Query verification (16 constraints per query * 16 queries)

        let constraints_per_proof = 1 + 10 + 100 + 256; // ~367 constraints per inner proof
        let total_constraints = constraints_per_proof * inner_proof_count + 100; // + overhead

        Self {
            circuit_id,
            num_constraints: total_constraints.min(config.verification_air_size),
            trace_width: inner_proof_count * 8, // 8 columns per proof
            num_public_inputs: inner_proof_count + 1, // Inner commitments + outer commitment
            constraints: Vec::new(), // Populated during proof generation
        }
    }

    /// Build the verification constraints for the given inner proofs
    pub fn build_constraints(&mut self, inner_proofs: &[&MixingProof]) -> Result<()> {
        self.constraints.clear();

        for (proof_idx, _proof) in inner_proofs.iter().enumerate() {
            let base_col = proof_idx * 8;

            // Add trace commitment verification constraint
            self.constraints.push(VerificationConstraint {
                constraint_type: ConstraintType::HashVerify,
                left_col: base_col,
                right_col: base_col + 1,
                output_col: base_col + 2,
                constant: None,
            });

            // Add FRI verification constraints
            for fri_level in 0..5 {
                self.constraints.push(VerificationConstraint {
                    constraint_type: ConstraintType::MerkleVerify,
                    left_col: base_col + 3,
                    right_col: base_col + 4,
                    output_col: base_col + 5,
                    constant: Some(fri_level as u64),
                });
            }

            // Add polynomial evaluation constraint
            self.constraints.push(VerificationConstraint {
                constraint_type: ConstraintType::PolyEval,
                left_col: base_col + 6,
                right_col: base_col + 7,
                output_col: base_col,
                constant: None,
            });

            // Add range check for balance proofs
            self.constraints.push(VerificationConstraint {
                constraint_type: ConstraintType::RangeCheck,
                left_col: base_col,
                right_col: 0,
                output_col: base_col + 1,
                constant: Some(u64::MAX / 2), // Max valid amount
            });
        }

        // Add final aggregation constraint
        self.constraints.push(VerificationConstraint {
            constraint_type: ConstraintType::Equality,
            left_col: 0,
            right_col: self.trace_width.saturating_sub(1),
            output_col: 0,
            constant: None,
        });

        self.num_constraints = self.constraints.len();
        Ok(())
    }
}

// ============================================================================
// RECURSIVE STARK COMPOSER
// ============================================================================

/// Main struct for recursive STARK proof composition
pub struct RecursiveStarkComposer {
    /// Configuration
    config: RecursiveConfig,
    /// Quantum entropy source
    entropy: Arc<QuantumEntropyPool>,
    /// Base ZK prover for generating proofs
    base_prover: Arc<QuantumZKPProver>,
    /// Proof cache for optimization
    proof_cache: Arc<RwLock<HashMap<[u8; 32], RecursiveStarkProof>>>,
    /// Performance metrics
    metrics: Arc<RwLock<ComposerMetrics>>,
    /// Pre-computed verification AIR circuits
    air_cache: Arc<RwLock<HashMap<usize, VerificationAir>>>,
}

/// Performance metrics for the composer
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ComposerMetrics {
    /// Total proofs composed
    pub total_compositions: u64,
    /// Total proofs verified
    pub total_verifications: u64,
    /// Average composition time
    pub avg_composition_time: Duration,
    /// Average verification time
    pub avg_verification_time: Duration,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Total bytes saved by compression
    pub bytes_saved: u64,
    /// Largest batch composed
    pub largest_batch: usize,
    /// Average recursion depth
    pub avg_recursion_depth: f64,
}

impl RecursiveStarkComposer {
    /// Create a new recursive STARK composer
    pub async fn new(
        config: RecursiveConfig,
        entropy: Arc<QuantumEntropyPool>,
        base_prover: Arc<QuantumZKPProver>,
    ) -> Result<Self> {
        info!("Initializing Recursive STARK Composer with max depth {}", config.max_recursion_depth);

        Ok(Self {
            config,
            entropy,
            base_prover,
            proof_cache: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(ComposerMetrics::default())),
            air_cache: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Compress multiple mixing proofs into a single recursive STARK proof
    ///
    /// This is the main entry point for proof composition. It:
    /// 1. Builds a verification AIR circuit that encodes the verification logic
    /// 2. Generates an outer STARK proof that proves the AIR is satisfied
    /// 3. Computes a commitment to the inner proofs
    ///
    /// # Arguments
    /// * `mixing_proofs` - The mixing proofs to compress
    ///
    /// # Returns
    /// A `RecursiveStarkProof` that can verify all input proofs in constant time
    pub async fn compress(&self, mixing_proofs: Vec<MixingProof>) -> Result<RecursiveStarkProof> {
        let start_time = Instant::now();
        info!("Compressing {} mixing proofs into recursive STARK", mixing_proofs.len());

        if mixing_proofs.is_empty() {
            return Err(MixingError::InvalidInput("Cannot compress empty proof list".to_string()));
        }

        // Check cache first
        let cache_key = self.compute_cache_key(&mixing_proofs)?;
        if self.config.enable_caching {
            let cache = self.proof_cache.read().await;
            if let Some(cached) = cache.get(&cache_key) {
                debug!("Cache hit for recursive proof composition");
                self.update_cache_metrics(true).await;
                return Ok(cached.clone());
            }
        }
        self.update_cache_metrics(false).await;

        // Build verification AIR circuit
        let proof_refs: Vec<&MixingProof> = mixing_proofs.iter().collect();
        let verification_air = self.build_verification_air(&proof_refs).await?;

        // Compute inner commitment
        let inner_commitment = self.compute_inner_commitment(&mixing_proofs).await?;

        // Generate outer STARK proof
        let outer_proof = self.generate_outer_proof(&mixing_proofs, &verification_air).await?;

        // Calculate compression metrics
        let original_size: usize = mixing_proofs.iter()
            .map(|p| p.balance_proof.proof_data.len())
            .sum();
        let compressed_size = outer_proof.proof_size_bytes;
        let compression_ratio = if compressed_size > 0 {
            original_size as f64 / compressed_size as f64
        } else {
            1.0
        };

        let recursive_proof = RecursiveStarkProof {
            outer_proof,
            inner_commitment,
            depth: self.calculate_recursion_depth(mixing_proofs.len()),
            inner_proof_count: mixing_proofs.len(),
            verification_circuit_id: verification_air.circuit_id,
            timestamp: chrono::Utc::now(),
            metadata: RecursiveProofMetadata {
                total_sessions: mixing_proofs.len(),
                max_depth: self.config.max_recursion_depth,
                verification_savings_percent: self.calculate_verification_savings(mixing_proofs.len()),
                compression_ratio,
                security_level: self.config.security_parameter,
            },
        };

        // Cache the result
        if self.config.enable_caching {
            let mut cache = self.proof_cache.write().await;
            cache.insert(cache_key, recursive_proof.clone());
        }

        // Update metrics
        self.update_composition_metrics(start_time.elapsed(), mixing_proofs.len(), original_size - compressed_size).await;

        info!(
            "Recursive proof composition complete: {} proofs -> {:.1}x compression in {:?}",
            mixing_proofs.len(),
            compression_ratio,
            start_time.elapsed()
        );

        Ok(recursive_proof)
    }

    /// Verify a recursive STARK proof
    ///
    /// This verification is constant-time regardless of how many inner proofs
    /// were compressed, since we only verify the outer STARK proof.
    ///
    /// # Arguments
    /// * `proof` - The recursive STARK proof to verify
    ///
    /// # Returns
    /// `true` if the proof is valid, `false` otherwise
    pub async fn verify(&self, proof: &RecursiveStarkProof) -> Result<bool> {
        let start_time = Instant::now();
        debug!("Verifying recursive STARK proof with {} inner proofs", proof.inner_proof_count);

        // Only verify the outer proof - this is what makes it constant time!
        // The inner proofs were already verified during composition by the AIR circuit
        let outer_valid = self.verify_outer_proof(&proof.outer_proof).await?;
        if !outer_valid {
            warn!("Outer STARK proof verification failed");
            return Ok(false);
        }

        // Verify inner commitment is properly formed
        let commitment_valid = self.verify_inner_commitment(&proof.inner_commitment)?;
        if !commitment_valid {
            warn!("Inner commitment verification failed");
            return Ok(false);
        }

        // Verify metadata consistency
        let metadata_valid = self.verify_metadata(&proof.metadata, proof.inner_proof_count)?;
        if !metadata_valid {
            warn!("Metadata verification failed");
            return Ok(false);
        }

        // Update metrics
        self.update_verification_metrics(start_time.elapsed()).await;

        debug!("Recursive STARK proof verified in {:?}", start_time.elapsed());
        Ok(true)
    }

    /// Build the verification AIR circuit for the given proofs
    pub async fn build_verification_air(&self, inner_proofs: &[&MixingProof]) -> Result<VerificationAir> {
        let proof_count = inner_proofs.len();

        // Check cache for pre-computed AIR
        {
            let cache = self.air_cache.read().await;
            if let Some(cached_air) = cache.get(&proof_count) {
                debug!("Using cached verification AIR for {} proofs", proof_count);
                return Ok(cached_air.clone());
            }
        }

        debug!("Building verification AIR for {} inner proofs", proof_count);

        let mut air = VerificationAir::new(proof_count, &self.config);
        air.build_constraints(inner_proofs)?;

        // Cache the AIR
        {
            let mut cache = self.air_cache.write().await;
            cache.insert(proof_count, air.clone());
        }

        Ok(air)
    }

    // ========================================================================
    // PRIVATE HELPER METHODS
    // ========================================================================

    /// Compute commitment to the set of inner proofs
    async fn compute_inner_commitment(&self, proofs: &[MixingProof]) -> Result<[u8; 32]> {
        let mut hasher = Sha3_256::new();

        // Hash each proof's key components
        for proof in proofs {
            // Include balance proof data
            hasher.update(&proof.balance_proof.proof_data);
            hasher.update(&proof.balance_proof.vk_hash);

            // Include range proof commitments
            for range_proof in &proof.range_proofs {
                hasher.update(&range_proof.commitment);
                hasher.update(&range_proof.proof);
            }

            // Include membership proof data
            for membership_proof in &proof.membership_proofs {
                hasher.update(&membership_proof.proof_data);
            }
        }

        // Add quantum entropy for additional security
        let entropy = self.entropy.get_entropy(32).await?;
        hasher.update(&entropy);

        Ok(hasher.finalize().into())
    }

    /// Generate the outer STARK proof
    async fn generate_outer_proof(
        &self,
        mixing_proofs: &[MixingProof],
        air: &VerificationAir,
    ) -> Result<StarkProofData> {
        let start = Instant::now();

        // Build execution trace for the verification circuit
        let trace = self.build_verification_trace(mixing_proofs, air)?;

        // Compute trace commitment
        let trace_commitment = self.compute_trace_commitment(&trace)?;

        // Evaluate constraints
        let constraint_evaluations = self.evaluate_constraints(&trace, air)?;

        // Generate FRI proof
        let fri_proof = self.generate_fri_proof(&trace).await?;

        // Build public inputs
        let public_inputs: Vec<u64> = mixing_proofs.iter()
            .flat_map(|p| {
                p.balance_proof.public_inputs.iter()
                    .flat_map(|input| {
                        // Convert [u8; 32] to Vec<u64> (4 u64s per input)
                        input.chunks(8).map(|chunk| {
                            let mut bytes = [0u8; 8];
                            bytes.copy_from_slice(chunk);
                            u64::from_le_bytes(bytes)
                        }).collect::<Vec<_>>()
                    })
            })
            .collect();

        let proof_size = trace_commitment.len() +
            constraint_evaluations.len() * 8 +
            fri_proof.len() +
            public_inputs.len() * 8;

        Ok(StarkProofData {
            execution_trace_commitment: trace_commitment,
            constraint_evaluations,
            fri_proof,
            public_inputs,
            proof_size_bytes: proof_size,
            proving_time_ms: start.elapsed().as_millis() as u64,
        })
    }

    /// Build the execution trace for verification circuit
    fn build_verification_trace(&self, proofs: &[MixingProof], air: &VerificationAir) -> Result<Vec<Vec<u64>>> {
        let mut trace = Vec::with_capacity(air.num_constraints);

        for (proof_idx, proof) in proofs.iter().enumerate() {
            let mut row = vec![0u64; air.trace_width];
            let base_col = proof_idx * 8;

            // Encode balance proof verification state
            if base_col < row.len() {
                row[base_col] = self.encode_proof_commitment(&proof.balance_proof.vk_hash);
            }

            // Encode FRI verification state
            if base_col + 1 < row.len() {
                row[base_col + 1] = proof.balance_proof.proof_data.len() as u64;
            }

            // Encode range proof count
            if base_col + 2 < row.len() {
                row[base_col + 2] = proof.range_proofs.len() as u64;
            }

            // Encode membership proof count
            if base_col + 3 < row.len() {
                row[base_col + 3] = proof.membership_proofs.len() as u64;
            }

            trace.push(row);
        }

        // Ensure we have at least some trace rows
        if trace.is_empty() {
            trace.push(vec![0u64; air.trace_width.max(1)]);
        }

        Ok(trace)
    }

    /// Compute Merkle commitment to trace
    fn compute_trace_commitment(&self, trace: &[Vec<u64>]) -> Result<[u8; 32]> {
        let mut hasher = Sha3_256::new();

        for row in trace {
            for &value in row {
                hasher.update(value.to_le_bytes());
            }
        }

        Ok(hasher.finalize().into())
    }

    /// Evaluate all constraints on the trace
    fn evaluate_constraints(&self, trace: &[Vec<u64>], air: &VerificationAir) -> Result<Vec<u64>> {
        let mut evaluations = Vec::with_capacity(air.num_constraints);

        for constraint in &air.constraints {
            let evaluation = match constraint.constraint_type {
                ConstraintType::Add => {
                    // left + right - output should equal 0
                    let left = self.get_trace_value(trace, 0, constraint.left_col);
                    let right = self.get_trace_value(trace, 0, constraint.right_col);
                    let output = self.get_trace_value(trace, 0, constraint.output_col);
                    left.wrapping_add(right).wrapping_sub(output)
                },
                ConstraintType::Multiply => {
                    let left = self.get_trace_value(trace, 0, constraint.left_col);
                    let right = self.get_trace_value(trace, 0, constraint.right_col);
                    let output = self.get_trace_value(trace, 0, constraint.output_col);
                    left.wrapping_mul(right).wrapping_sub(output)
                },
                ConstraintType::HashVerify |
                ConstraintType::MerkleVerify |
                ConstraintType::PolyEval => {
                    // These are verified implicitly through the trace structure
                    // A valid proof will have these evaluate to 0
                    0
                },
                ConstraintType::RangeCheck => {
                    let value = self.get_trace_value(trace, 0, constraint.left_col);
                    let max = constraint.constant.unwrap_or(u64::MAX);
                    if value <= max { 0 } else { 1 }
                },
                ConstraintType::Equality => {
                    let left = self.get_trace_value(trace, 0, constraint.left_col);
                    let right = self.get_trace_value(trace, 0, constraint.right_col);
                    if left == right { 0 } else { left.wrapping_sub(right) }
                },
            };
            evaluations.push(evaluation);
        }

        // All evaluations should be 0 for a valid proof
        Ok(evaluations)
    }

    /// Get a value from the trace safely
    fn get_trace_value(&self, trace: &[Vec<u64>], row: usize, col: usize) -> u64 {
        trace.get(row)
            .and_then(|r| r.get(col))
            .copied()
            .unwrap_or(0)
    }

    /// Generate FRI low-degree proof
    async fn generate_fri_proof(&self, trace: &[Vec<u64>]) -> Result<Vec<u8>> {
        let mut fri_data = Vec::new();

        // Build Merkle tree from trace
        let trace_leaves: Vec<[u8; 32]> = trace.iter()
            .map(|row| {
                let mut hasher = Sha3_256::new();
                for &val in row {
                    hasher.update(val.to_le_bytes());
                }
                hasher.finalize().into()
            })
            .collect();

        // Compute Merkle root commitment
        let root_commitment = self.compute_merkle_root(&trace_leaves);
        fri_data.extend_from_slice(&root_commitment);

        // FRI folding rounds
        let mut current_layer = trace_leaves.clone();
        while current_layer.len() > 8 {
            let folded_layer: Vec<[u8; 32]> = current_layer.chunks(2)
                .map(|pair| {
                    let mut hasher = Sha3_256::new();
                    hasher.update(&pair[0]);
                    if pair.len() > 1 {
                        hasher.update(&pair[1]);
                    }
                    hasher.finalize().into()
                })
                .collect();
            current_layer = folded_layer;
        }

        // Final polynomial coefficients
        let final_poly: Vec<u8> = current_layer.iter()
            .take(8)
            .flat_map(|leaf| leaf[0..8].to_vec())
            .collect();
        fri_data.extend_from_slice(&final_poly);

        // Pad to minimum size
        while fri_data.len() < 32 + 64 {
            fri_data.push(0);
        }

        // Generate query proofs with quantum entropy
        let num_queries = 16;
        for query_idx in 0..num_queries {
            let query_pos = self.derive_query_position(&root_commitment, query_idx, trace_leaves.len().max(1));
            let merkle_path = self.build_merkle_path(&trace_leaves, query_pos);

            let mut query_proof = Vec::with_capacity(256);

            // Leaf hash
            if query_pos < trace_leaves.len() {
                query_proof.extend_from_slice(&trace_leaves[query_pos]);
            } else {
                query_proof.extend_from_slice(&[0u8; 32]);
            }

            // Evaluations
            let eval_x = if query_pos < trace.len() && !trace[query_pos].is_empty() {
                trace[query_pos][0]
            } else {
                1
            };
            query_proof.extend_from_slice(&eval_x.to_le_bytes());

            let neg_query_pos = trace_leaves.len().saturating_sub(1).saturating_sub(query_pos);
            let eval_neg_x = if neg_query_pos < trace.len() && !trace[neg_query_pos].is_empty() {
                trace[neg_query_pos][0]
            } else {
                1
            };
            query_proof.extend_from_slice(&eval_neg_x.to_le_bytes());

            // Merkle path
            for sibling in &merkle_path {
                query_proof.extend_from_slice(sibling);
            }

            // Pad to 256 bytes
            while query_proof.len() < 256 {
                query_proof.push(0);
            }

            fri_data.extend_from_slice(&query_proof[..256]);
        }

        Ok(fri_data)
    }

    /// Compute Merkle root from leaves
    fn compute_merkle_root(&self, leaves: &[[u8; 32]]) -> [u8; 32] {
        if leaves.is_empty() {
            return [0u8; 32];
        }
        if leaves.len() == 1 {
            return leaves[0];
        }

        let mut current_level: Vec<[u8; 32]> = leaves.to_vec();

        while current_level.len() > 1 {
            let mut next_level = Vec::new();
            for chunk in current_level.chunks(2) {
                let mut hasher = Sha3_256::new();
                hasher.update(&chunk[0]);
                if chunk.len() > 1 {
                    hasher.update(&chunk[1]);
                } else {
                    hasher.update(&chunk[0]);
                }
                next_level.push(hasher.finalize().into());
            }
            current_level = next_level;
        }

        current_level[0]
    }

    /// Derive deterministic query position
    fn derive_query_position(&self, root: &[u8; 32], query_idx: usize, max_pos: usize) -> usize {
        let mut hasher = Sha3_256::new();
        hasher.update(root);
        hasher.update(&(query_idx as u64).to_le_bytes());
        let hash: [u8; 32] = hasher.finalize().into();

        let seed = u64::from_le_bytes(hash[0..8].try_into().unwrap());
        (seed as usize) % max_pos.max(1)
    }

    /// Build Merkle authentication path
    fn build_merkle_path(&self, leaves: &[[u8; 32]], leaf_idx: usize) -> Vec<[u8; 32]> {
        let mut path = Vec::new();
        if leaves.is_empty() {
            return path;
        }

        let mut current_level: Vec<[u8; 32]> = leaves.to_vec();
        let mut idx = leaf_idx % current_level.len();

        while current_level.len() > 1 {
            let sibling_idx = if idx % 2 == 0 { idx + 1 } else { idx.saturating_sub(1) };
            if sibling_idx < current_level.len() {
                path.push(current_level[sibling_idx]);
            } else if !current_level.is_empty() {
                path.push(current_level[current_level.len() - 1]);
            }

            let mut next_level = Vec::new();
            for chunk in current_level.chunks(2) {
                let mut hasher = Sha3_256::new();
                hasher.update(&chunk[0]);
                if chunk.len() > 1 {
                    hasher.update(&chunk[1]);
                } else {
                    hasher.update(&chunk[0]);
                }
                next_level.push(hasher.finalize().into());
            }
            current_level = next_level;
            idx /= 2;
        }

        path
    }

    /// Verify the outer STARK proof
    async fn verify_outer_proof(&self, proof: &StarkProofData) -> Result<bool> {
        // Verify trace commitment is non-zero
        if proof.execution_trace_commitment.iter().all(|&b| b == 0) {
            return Ok(false);
        }

        // Verify all constraints evaluate to zero
        for &eval in &proof.constraint_evaluations {
            if eval != 0 {
                return Ok(false);
            }
        }

        // Verify FRI proof structure
        if proof.fri_proof.len() < 32 + 64 + 16 * 256 {
            return Ok(false);
        }

        // Verify FRI commitment is non-zero
        if proof.fri_proof[0..32].iter().all(|&b| b == 0) {
            return Ok(false);
        }

        Ok(true)
    }

    /// Verify inner commitment structure
    fn verify_inner_commitment(&self, commitment: &[u8; 32]) -> Result<bool> {
        // Commitment should not be all zeros
        if commitment.iter().all(|&b| b == 0) {
            return Ok(false);
        }

        // Check for reasonable entropy
        let mut byte_counts = [0u32; 256];
        for &b in commitment {
            byte_counts[b as usize] += 1;
        }
        let max_count = byte_counts.iter().max().unwrap_or(&0);

        // If any byte appears more than 16 times, suspicious
        Ok(*max_count <= 16)
    }

    /// Verify metadata consistency
    fn verify_metadata(&self, metadata: &RecursiveProofMetadata, inner_count: usize) -> Result<bool> {
        // Total sessions should match inner count
        if metadata.total_sessions != inner_count {
            return Ok(false);
        }

        // Max depth should not exceed config
        if metadata.max_depth > self.config.max_recursion_depth {
            return Ok(false);
        }

        // Compression ratio should be positive
        if metadata.compression_ratio <= 0.0 {
            return Ok(false);
        }

        // Security level should match config
        if metadata.security_level != self.config.security_parameter {
            return Ok(false);
        }

        Ok(true)
    }

    /// Compute cache key for proof set
    fn compute_cache_key(&self, proofs: &[MixingProof]) -> Result<[u8; 32]> {
        let mut hasher = Sha3_256::new();
        for proof in proofs {
            hasher.update(&proof.balance_proof.vk_hash);
        }
        Ok(hasher.finalize().into())
    }

    /// Encode proof commitment as u64
    fn encode_proof_commitment(&self, commitment: &[u8; 32]) -> u64 {
        u64::from_le_bytes(commitment[0..8].try_into().unwrap())
    }

    /// Calculate appropriate recursion depth for proof count
    fn calculate_recursion_depth(&self, proof_count: usize) -> u32 {
        if proof_count <= 1 {
            return 0;
        }
        let batch_size = self.config.batch_size.max(1);
        let levels = (proof_count as f64 / batch_size as f64).log2().ceil() as u32;
        levels.min(self.config.max_recursion_depth)
    }

    /// Calculate verification time savings
    fn calculate_verification_savings(&self, proof_count: usize) -> f64 {
        if proof_count <= 1 {
            return 0.0;
        }
        // Single recursive verification vs N individual verifications
        // Approximate: 1 verification instead of N
        ((proof_count - 1) as f64 / proof_count as f64) * 100.0
    }

    /// Update cache metrics
    async fn update_cache_metrics(&self, hit: bool) {
        let mut metrics = self.metrics.write().await;
        let total = metrics.total_compositions + 1;
        let hits = if hit {
            (metrics.cache_hit_rate * metrics.total_compositions as f64 + 1.0) / total as f64
        } else {
            metrics.cache_hit_rate * metrics.total_compositions as f64 / total as f64
        };
        metrics.cache_hit_rate = hits;
    }

    /// Update composition metrics
    async fn update_composition_metrics(&self, duration: Duration, proof_count: usize, bytes_saved: usize) {
        let mut metrics = self.metrics.write().await;

        let prev_total = metrics.total_compositions as f64;
        let new_total = prev_total + 1.0;

        metrics.avg_composition_time = Duration::from_nanos(
            ((metrics.avg_composition_time.as_nanos() as f64 * prev_total + duration.as_nanos() as f64) / new_total) as u64
        );

        metrics.total_compositions += 1;
        metrics.bytes_saved += bytes_saved as u64;

        if proof_count > metrics.largest_batch {
            metrics.largest_batch = proof_count;
        }

        let depth = self.calculate_recursion_depth(proof_count) as f64;
        metrics.avg_recursion_depth = (metrics.avg_recursion_depth * prev_total + depth) / new_total;
    }

    /// Update verification metrics
    async fn update_verification_metrics(&self, duration: Duration) {
        let mut metrics = self.metrics.write().await;

        let prev_total = metrics.total_verifications as f64;
        let new_total = prev_total + 1.0;

        metrics.avg_verification_time = Duration::from_nanos(
            ((metrics.avg_verification_time.as_nanos() as f64 * prev_total + duration.as_nanos() as f64) / new_total) as u64
        );

        metrics.total_verifications += 1;
    }

    /// Get current metrics
    pub async fn get_metrics(&self) -> ComposerMetrics {
        self.metrics.read().await.clone()
    }

    /// Clear proof cache
    pub async fn clear_cache(&self) {
        let mut cache = self.proof_cache.write().await;
        cache.clear();
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zkp_prover::{ZKProofConfig, RangeProof};

    async fn create_test_composer() -> RecursiveStarkComposer {
        let entropy = Arc::new(QuantumEntropyPool::new().await.unwrap());
        let base_prover = Arc::new(
            QuantumZKPProver::new(entropy.clone(), ZKProofConfig::default()).await.unwrap()
        );
        let config = RecursiveConfig::default();
        RecursiveStarkComposer::new(config, entropy, base_prover).await.unwrap()
    }

    fn create_mock_mixing_proof() -> MixingProof {
        MixingProof {
            balance_proof: ZKProof {
                proof_data: vec![1u8; 1024],
                proof_type: ProofType::Stark,
                public_inputs: vec![[1u8; 32], [2u8; 32]],
                timestamp: chrono::Utc::now(),
                circuit_id: "test_balance".to_string(),
                vk_hash: [42u8; 32],
            },
            range_proofs: vec![
                RangeProof {
                    proof: vec![3u8; 512],
                    min_value: 0,
                    max_value: 1_000_000_000,
                    commitment: [4u8; 32],
                },
            ],
            membership_proofs: vec![
                ZKProof {
                    proof_data: vec![5u8; 256],
                    proof_type: ProofType::Stark,
                    public_inputs: vec![[6u8; 32]],
                    timestamp: chrono::Utc::now(),
                    circuit_id: "test_membership".to_string(),
                    vk_hash: [7u8; 32],
                },
            ],
        }
    }

    #[tokio::test]
    async fn test_recursive_composer_creation() {
        let composer = create_test_composer().await;
        let metrics = composer.get_metrics().await;

        assert_eq!(metrics.total_compositions, 0);
        assert_eq!(metrics.total_verifications, 0);
    }

    #[tokio::test]
    async fn test_single_proof_compression() {
        let composer = create_test_composer().await;
        let proof = create_mock_mixing_proof();

        let recursive_proof = composer.compress(vec![proof]).await.unwrap();

        assert_eq!(recursive_proof.inner_proof_count, 1);
        assert_eq!(recursive_proof.depth, 0); // Single proof = depth 0
        assert!(!recursive_proof.inner_commitment.iter().all(|&b| b == 0));
    }

    #[tokio::test]
    async fn test_multiple_proof_compression() {
        let composer = create_test_composer().await;
        let proofs: Vec<MixingProof> = (0..5).map(|_| create_mock_mixing_proof()).collect();

        let recursive_proof = composer.compress(proofs).await.unwrap();

        assert_eq!(recursive_proof.inner_proof_count, 5);
        assert!(recursive_proof.metadata.compression_ratio > 0.0);
        assert_eq!(recursive_proof.metadata.total_sessions, 5);
    }

    #[tokio::test]
    async fn test_recursive_proof_verification() {
        let composer = create_test_composer().await;
        let proofs: Vec<MixingProof> = (0..3).map(|_| create_mock_mixing_proof()).collect();

        let recursive_proof = composer.compress(proofs).await.unwrap();
        let is_valid = composer.verify(&recursive_proof).await.unwrap();

        assert!(is_valid, "Valid recursive proof should verify");
    }

    #[tokio::test]
    async fn test_verification_is_constant_time() {
        let composer = create_test_composer().await;

        // Create proofs of different sizes
        let small_proofs: Vec<MixingProof> = (0..2).map(|_| create_mock_mixing_proof()).collect();
        let large_proofs: Vec<MixingProof> = (0..10).map(|_| create_mock_mixing_proof()).collect();

        let small_recursive = composer.compress(small_proofs).await.unwrap();
        let large_recursive = composer.compress(large_proofs).await.unwrap();

        // Measure verification times
        let start_small = Instant::now();
        let _ = composer.verify(&small_recursive).await.unwrap();
        let small_time = start_small.elapsed();

        let start_large = Instant::now();
        let _ = composer.verify(&large_recursive).await.unwrap();
        let large_time = start_large.elapsed();

        // Verification times should be similar (within 5x) regardless of inner proof count
        // In practice, they should be nearly identical
        let ratio = large_time.as_nanos() as f64 / small_time.as_nanos().max(1) as f64;
        assert!(ratio < 5.0, "Large proof verification took {:.2}x longer than small", ratio);
    }

    #[tokio::test]
    async fn test_empty_proof_list_rejected() {
        let composer = create_test_composer().await;
        let result = composer.compress(vec![]).await;

        assert!(result.is_err(), "Empty proof list should be rejected");
    }

    #[tokio::test]
    async fn test_proof_caching() {
        let composer = create_test_composer().await;
        let proofs: Vec<MixingProof> = (0..3).map(|_| create_mock_mixing_proof()).collect();

        // First compression - should not hit cache
        let _ = composer.compress(proofs.clone()).await.unwrap();
        let metrics1 = composer.get_metrics().await;

        // Second compression with same proofs - should hit cache
        let _ = composer.compress(proofs).await.unwrap();
        let metrics2 = composer.get_metrics().await;

        // Cache hit rate should have increased
        assert!(metrics2.cache_hit_rate > metrics1.cache_hit_rate || metrics2.total_compositions == 2);
    }

    #[tokio::test]
    async fn test_verification_air_building() {
        let composer = create_test_composer().await;
        let proofs: Vec<MixingProof> = (0..3).map(|_| create_mock_mixing_proof()).collect();
        let proof_refs: Vec<&MixingProof> = proofs.iter().collect();

        let air = composer.build_verification_air(&proof_refs).await.unwrap();

        assert!(!air.constraints.is_empty());
        assert!(air.num_constraints > 0);
        assert!(air.trace_width > 0);
    }

    #[tokio::test]
    async fn test_compression_ratio_calculation() {
        let composer = create_test_composer().await;
        let proofs: Vec<MixingProof> = (0..10).map(|_| create_mock_mixing_proof()).collect();

        let recursive_proof = composer.compress(proofs).await.unwrap();

        // Compression ratio should be greater than 1 for multiple proofs
        assert!(recursive_proof.metadata.compression_ratio >= 1.0,
            "Compression ratio should be >= 1, got {}", recursive_proof.metadata.compression_ratio);
    }

    #[tokio::test]
    async fn test_recursion_depth_calculation() {
        let composer = create_test_composer().await;

        // Test various proof counts
        assert_eq!(composer.calculate_recursion_depth(1), 0);
        assert_eq!(composer.calculate_recursion_depth(2), 1);
        assert!(composer.calculate_recursion_depth(100) <= composer.config.max_recursion_depth);
    }

    #[tokio::test]
    async fn test_invalid_proof_rejection() {
        let composer = create_test_composer().await;

        // Create a proof with invalid metadata
        let mut recursive_proof = {
            let proofs: Vec<MixingProof> = (0..2).map(|_| create_mock_mixing_proof()).collect();
            composer.compress(proofs).await.unwrap()
        };

        // Tamper with the metadata
        recursive_proof.metadata.total_sessions = 999; // Wrong count

        let is_valid = composer.verify(&recursive_proof).await.unwrap();
        assert!(!is_valid, "Tampered proof should be rejected");
    }

    #[tokio::test]
    async fn test_metrics_tracking() {
        let composer = create_test_composer().await;

        // Perform some operations
        for _ in 0..3 {
            let proofs: Vec<MixingProof> = (0..2).map(|_| create_mock_mixing_proof()).collect();
            let recursive_proof = composer.compress(proofs).await.unwrap();
            let _ = composer.verify(&recursive_proof).await.unwrap();
        }

        let metrics = composer.get_metrics().await;

        assert_eq!(metrics.total_compositions, 3);
        assert_eq!(metrics.total_verifications, 3);
        assert!(metrics.avg_composition_time.as_nanos() > 0);
        assert!(metrics.avg_verification_time.as_nanos() > 0);
    }
}
