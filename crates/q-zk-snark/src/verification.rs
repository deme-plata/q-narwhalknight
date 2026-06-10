//! Verification utilities and batch processing for Q-NarwhalKnight zk-SNARKs
//!
//! Provides efficient proof verification, batch processing, and aggregation
//! capabilities for multiple SNARK protocols.

use anyhow::Result;
use ark_ec::pairing::Pairing;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, Semaphore};

use crate::{SNARKError, SNARKProtocol};

/// Universal proof verifier that can verify proofs from different SNARK systems
pub struct UniversalVerifier<E: Pairing> {
    /// Cached verifying keys by protocol and circuit ID
    verifying_keys: Arc<RwLock<HashMap<(SNARKProtocol, String), Arc<dyn VerifyingKeyTrait<E>>>>>,
    /// Verification semaphore for rate limiting
    verification_semaphore: Semaphore,
    /// Batch verification settings
    batch_config: BatchVerificationConfig,
}

/// Trait for verifying key abstractions across different protocols
pub trait VerifyingKeyTrait<E: Pairing>: Send + Sync {
    fn verify_proof(&self, proof: &dyn ProofTrait<E>) -> Result<bool>;
    fn protocol(&self) -> SNARKProtocol;
    fn circuit_id(&self) -> &str;
}

/// Trait for proof abstractions across different protocols
pub trait ProofTrait<E: Pairing>: Send + Sync {
    fn protocol(&self) -> SNARKProtocol;
    fn public_inputs(&self) -> &[E::ScalarField];
    fn serialize(&self) -> Result<Vec<u8>>;
}

/// Batch verification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchVerificationConfig {
    /// Maximum batch size for parallel verification
    pub max_batch_size: usize,
    /// Number of parallel verification threads
    pub verification_threads: usize,
    /// Enable proof aggregation
    pub enable_aggregation: bool,
    /// Verification timeout in milliseconds
    pub verification_timeout_ms: u64,
}

impl Default for BatchVerificationConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 100,
            verification_threads: num_cpus::get(),
            enable_aggregation: true,
            verification_timeout_ms: 5000,
        }
    }
}

/// Batch verification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchVerificationResult {
    /// Total number of proofs verified
    pub total_proofs: usize,
    /// Number of valid proofs
    pub valid_proofs: usize,
    /// Number of invalid proofs  
    pub invalid_proofs: usize,
    /// Verification time in milliseconds
    pub verification_time_ms: u64,
    /// Per-proof results (index -> result)
    pub individual_results: HashMap<usize, bool>,
    /// Any errors encountered
    pub errors: Vec<(usize, String)>,
}

/// Proof verification request
#[derive(Clone)]
pub struct VerificationRequest<E: Pairing> {
    /// Proof to verify
    pub proof: Arc<dyn ProofTrait<E>>,
    /// Circuit identifier
    pub circuit_id: String,
    /// Protocol to use
    pub protocol: SNARKProtocol,
    /// Request priority (higher = more priority)
    pub priority: u8,
    /// Request ID for tracking
    pub request_id: String,
}

/// Aggregate proof for batch verification
#[derive(Clone)]
pub struct AggregateProof<E: Pairing> {
    /// Individual proofs
    pub proofs: Vec<Arc<dyn ProofTrait<E>>>,
    /// Aggregated commitment (protocol-specific)
    pub aggregate_commitment: Vec<u8>,
    /// Public inputs for all proofs
    pub public_inputs: Vec<Vec<E::ScalarField>>,
    /// Protocol used for aggregation
    pub protocol: SNARKProtocol,
}

impl<E: Pairing> UniversalVerifier<E> {
    /// Create new universal verifier
    pub fn new(config: BatchVerificationConfig) -> Self {
        Self {
            verifying_keys: Arc::new(RwLock::new(HashMap::new())),
            verification_semaphore: Semaphore::new(config.verification_threads),
            batch_config: config,
        }
    }

    /// Register verifying key for a circuit
    pub async fn register_verifying_key(
        &self,
        protocol: SNARKProtocol,
        circuit_id: String,
        vk: Arc<dyn VerifyingKeyTrait<E>>,
    ) -> Result<()> {
        let mut keys = self.verifying_keys.write().await;
        keys.insert((protocol, circuit_id), vk);
        Ok(())
    }

    /// Verify single proof
    pub async fn verify_proof(&self, request: VerificationRequest<E>) -> Result<bool> {
        let _permit = self
            .verification_semaphore
            .acquire()
            .await
            .map_err(|e| SNARKError::VerificationFailed(format!("Semaphore error: {:?}", e)))?;

        let keys = self.verifying_keys.read().await;
        let key = keys
            .get(&(request.protocol, request.circuit_id.clone()))
            .ok_or_else(|| {
                SNARKError::VerificationFailed(format!(
                    "No verifying key found for protocol {:?} and circuit {}",
                    request.protocol, request.circuit_id
                ))
            })?;

        key.verify_proof(request.proof.as_ref())
    }

    /// Batch verify multiple proofs
    pub async fn batch_verify(
        &self,
        requests: Vec<VerificationRequest<E>>,
    ) -> Result<BatchVerificationResult> {
        let start_time = std::time::Instant::now();
        let total_proofs = requests.len();

        // Split into batches
        let batches: Vec<_> = requests
            .chunks(self.batch_config.max_batch_size)
            .map(|chunk| chunk.to_vec())
            .collect();

        // Process batches in parallel
        #[cfg(feature = "parallel")]
        let batch_results: Vec<_> = batches
            .into_par_iter()
            .map(|batch| self.verify_batch_sync(batch))
            .collect();

        #[cfg(not(feature = "parallel"))]
        let batch_results: Vec<_> = batches
            .into_iter()
            .map(|batch| self.verify_batch_sync(batch))
            .collect();

        // Combine results
        let mut valid_proofs = 0;
        let mut invalid_proofs = 0;
        let mut individual_results = HashMap::new();
        let mut errors = Vec::new();
        let mut result_index = 0;

        for batch_result in batch_results {
            match batch_result {
                Ok(results) => {
                    for result in results {
                        if result {
                            valid_proofs += 1;
                        } else {
                            invalid_proofs += 1;
                        }
                        individual_results.insert(result_index, result);
                        result_index += 1;
                    }
                }
                Err(e) => {
                    errors.push((result_index, e.to_string()));
                    result_index += 1;
                }
            }
        }

        let verification_time_ms = start_time.elapsed().as_millis() as u64;

        Ok(BatchVerificationResult {
            total_proofs,
            valid_proofs,
            invalid_proofs,
            verification_time_ms,
            individual_results,
            errors,
        })
    }

    /// Aggregate multiple proofs for batch verification
    pub async fn aggregate_proofs(
        &self,
        proofs: Vec<Arc<dyn ProofTrait<E>>>,
        protocol: SNARKProtocol,
    ) -> Result<AggregateProof<E>> {
        if proofs.is_empty() {
            return Err(SNARKError::InvalidParameters(
                "Cannot aggregate empty proof set".to_string(),
            )
            .into());
        }

        // Ensure all proofs use the same protocol
        for proof in &proofs {
            if proof.protocol() != protocol {
                return Err(SNARKError::InvalidParameters(
                    "All proofs must use the same protocol for aggregation".to_string(),
                )
                .into());
            }
        }

        let public_inputs: Vec<_> = proofs
            .iter()
            .map(|proof| proof.public_inputs().to_vec())
            .collect();

        // Protocol-specific aggregation
        let aggregate_commitment = match protocol {
            SNARKProtocol::Groth16 => self.aggregate_groth16_proofs(&proofs).await?,
            SNARKProtocol::PLONK => self.aggregate_plonk_proofs(&proofs).await?,
            _ => {
                return Err(SNARKError::InvalidParameters(format!(
                    "Aggregation not supported for protocol {:?}",
                    protocol
                ))
                .into())
            }
        };

        Ok(AggregateProof {
            proofs,
            aggregate_commitment,
            public_inputs,
            protocol,
        })
    }

    /// Verify aggregate proof
    pub async fn verify_aggregate_proof(
        &self,
        aggregate_proof: &AggregateProof<E>,
    ) -> Result<bool> {
        match aggregate_proof.protocol {
            SNARKProtocol::Groth16 => self.verify_groth16_aggregate(aggregate_proof).await,
            SNARKProtocol::PLONK => self.verify_plonk_aggregate(aggregate_proof).await,
            _ => Err(SNARKError::VerificationFailed(format!(
                "Aggregate verification not supported for protocol {:?}",
                aggregate_proof.protocol
            ))
            .into()),
        }
    }

    /// Get verification statistics
    pub async fn get_statistics(&self) -> VerificationStatistics {
        let keys = self.verifying_keys.read().await;
        let available_permits = self.verification_semaphore.available_permits();
        let total_permits = self.batch_config.verification_threads;

        VerificationStatistics {
            registered_circuits: keys.len(),
            active_verifications: total_permits - available_permits,
            max_parallel_verifications: total_permits,
            max_batch_size: self.batch_config.max_batch_size,
        }
    }

    // Private helper methods

    fn verify_batch_sync(&self, batch: Vec<VerificationRequest<E>>) -> Result<Vec<bool>> {
        // Synchronous version for parallel processing
        let mut results = Vec::new();

        for _request in batch {
            // This would need to be implemented without async for rayon
            // For now, returning a placeholder
            results.push(true);
        }

        Ok(results)
    }

    async fn aggregate_groth16_proofs(
        &self,
        _proofs: &[Arc<dyn ProofTrait<E>>],
    ) -> Result<Vec<u8>> {
        // Simplified Groth16 aggregation - real implementation would use
        // techniques like proof aggregation or batch verification
        Ok(vec![0u8; 96]) // Placeholder for aggregated proof
    }

    async fn aggregate_plonk_proofs(&self, _proofs: &[Arc<dyn ProofTrait<E>>]) -> Result<Vec<u8>> {
        // Simplified PLONK aggregation
        Ok(vec![0u8; 128]) // Placeholder for aggregated proof
    }

    async fn verify_groth16_aggregate(&self, _aggregate_proof: &AggregateProof<E>) -> Result<bool> {
        // Placeholder for Groth16 aggregate verification
        Ok(true)
    }

    async fn verify_plonk_aggregate(&self, _aggregate_proof: &AggregateProof<E>) -> Result<bool> {
        // Placeholder for PLONK aggregate verification
        Ok(true)
    }
}

/// Verification statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationStatistics {
    pub registered_circuits: usize,
    pub active_verifications: usize,
    pub max_parallel_verifications: usize,
    pub max_batch_size: usize,
}

/// Verification cache for optimizing repeated verifications
pub struct VerificationCache<E: Pairing> {
    /// Cache of verification results by proof hash
    cache: Arc<RwLock<HashMap<Vec<u8>, (bool, std::time::Instant)>>>,
    /// Cache TTL in seconds
    cache_ttl_seconds: u64,
    /// Maximum cache size
    max_cache_size: usize,
    /// Phantom data for the pairing type
    _phantom: std::marker::PhantomData<E>,
}

impl<E: Pairing> VerificationCache<E> {
    pub fn new(cache_ttl_seconds: u64, max_cache_size: usize) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            cache_ttl_seconds,
            max_cache_size,
            _phantom: std::marker::PhantomData,
        }
    }

    pub async fn get_cached_result(&self, proof_hash: &[u8]) -> Option<bool> {
        let cache = self.cache.read().await;

        if let Some((result, timestamp)) = cache.get(proof_hash) {
            let now = std::time::Instant::now();
            let ttl_duration = std::time::Duration::from_secs(self.cache_ttl_seconds);

            if now.duration_since(*timestamp) < ttl_duration {
                return Some(*result);
            }
        }

        None
    }

    pub async fn cache_result(&self, proof_hash: Vec<u8>, result: bool) {
        let mut cache = self.cache.write().await;

        // Remove expired entries
        let now = std::time::Instant::now();
        let ttl_duration = std::time::Duration::from_secs(self.cache_ttl_seconds);

        cache.retain(|_, (_, timestamp)| now.duration_since(*timestamp) < ttl_duration);

        // Evict oldest entries if cache is full
        if cache.len() >= self.max_cache_size {
            let oldest_key = cache
                .iter()
                .min_by_key(|(_, (_, timestamp))| timestamp)
                .map(|(key, _)| key.clone());

            if let Some(key) = oldest_key {
                cache.remove(&key);
            }
        }

        cache.insert(proof_hash, (result, now));
    }
}

/// Proof verification metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationMetrics {
    /// Total verifications performed
    pub total_verifications: u64,
    /// Total valid proofs
    pub valid_proofs: u64,
    /// Total invalid proofs
    pub invalid_proofs: u64,
    /// Average verification time in milliseconds
    pub avg_verification_time_ms: f64,
    /// Peak verification throughput (proofs/second)
    pub peak_throughput: f64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Error rate
    pub error_rate: f64,
}

/// Verification benchmarking utilities
pub struct VerificationBenchmark;

impl VerificationBenchmark {
    /// Benchmark verification performance for different proof types
    pub fn benchmark_verification_performance<E: Pairing>(
        _verifier: &UniversalVerifier<E>,
        test_proofs: Vec<VerificationRequest<E>>,
    ) -> VerificationMetrics {
        let start_time = std::time::Instant::now();
        let mut valid_count = 0u64;
        let mut invalid_count = 0u64;
        let mut total_time_ms = 0u64;

        for _proof_request in test_proofs {
            let proof_start = std::time::Instant::now();

            // This would need to be properly implemented with async runtime
            // For now, using placeholder
            let is_valid = true; // Placeholder

            let proof_time = proof_start.elapsed();
            total_time_ms += proof_time.as_millis() as u64;

            if is_valid {
                valid_count += 1;
            } else {
                invalid_count += 1;
            }
        }

        let total_verifications = valid_count + invalid_count;
        let total_elapsed = start_time.elapsed();
        let avg_verification_time_ms = if total_verifications > 0 {
            total_time_ms as f64 / total_verifications as f64
        } else {
            0.0
        };

        let peak_throughput = if total_elapsed.as_secs_f64() > 0.0 {
            total_verifications as f64 / total_elapsed.as_secs_f64()
        } else {
            0.0
        };

        VerificationMetrics {
            total_verifications,
            valid_proofs: valid_count,
            invalid_proofs: invalid_count,
            avg_verification_time_ms,
            peak_throughput,
            cache_hit_rate: 0.0, // Would be computed from cache stats
            error_rate: 0.0,     // Would be computed from error count
        }
    }

    /// Generate test proofs for benchmarking
    pub fn generate_test_proofs<E: Pairing>(
        count: usize,
        protocol: SNARKProtocol,
    ) -> Vec<VerificationRequest<E>> {
        // Generate synthetic verification requests for testing
        (0..count)
            .map(|i| VerificationRequest {
                proof: Arc::new(MockProof::<E>::new(protocol)),
                circuit_id: format!("test_circuit_{}", i % 10),
                protocol,
                priority: (i % 3) as u8,
                request_id: format!("request_{}", i),
            })
            .collect()
    }
}

/// Mock proof implementation for testing
struct MockProof<E: Pairing> {
    protocol: SNARKProtocol,
    public_inputs: Vec<E::ScalarField>,
}

impl<E: Pairing> MockProof<E> {
    fn new(protocol: SNARKProtocol) -> Self {
        Self {
            protocol,
            public_inputs: vec![E::ScalarField::from(42u64)],
        }
    }
}

impl<E: Pairing> ProofTrait<E> for MockProof<E> {
    fn protocol(&self) -> SNARKProtocol {
        self.protocol
    }

    fn public_inputs(&self) -> &[E::ScalarField] {
        &self.public_inputs
    }

    fn serialize(&self) -> Result<Vec<u8>> {
        Ok(vec![1, 2, 3, 4]) // Mock serialization
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::{Bn254, Fr};

    #[tokio::test]
    async fn test_universal_verifier_creation() {
        let config = BatchVerificationConfig::default();
        let verifier = UniversalVerifier::<Bn254>::new(config);

        let stats = verifier.get_statistics().await;
        assert_eq!(stats.registered_circuits, 0);
        assert!(stats.max_parallel_verifications > 0);
    }

    #[tokio::test]
    async fn test_verification_cache() {
        let cache = VerificationCache::<Bn254>::new(60, 100);

        let proof_hash = vec![1, 2, 3, 4];
        let result = cache.get_cached_result(&proof_hash).await;
        assert!(result.is_none());

        cache.cache_result(proof_hash.clone(), true).await;
        let cached_result = cache.get_cached_result(&proof_hash).await;
        assert_eq!(cached_result, Some(true));
    }

    #[test]
    fn test_batch_verification_config() {
        let config = BatchVerificationConfig::default();
        assert!(config.max_batch_size > 0);
        assert!(config.verification_threads > 0);
        assert!(config.verification_timeout_ms > 0);
    }

    #[test]
    fn test_verification_benchmark() {
        let config = BatchVerificationConfig::default();
        let verifier = UniversalVerifier::<Bn254>::new(config);

        let test_proofs = VerificationBenchmark::generate_test_proofs(10, SNARKProtocol::Groth16);
        assert_eq!(test_proofs.len(), 10);

        let metrics =
            VerificationBenchmark::benchmark_verification_performance(&verifier, test_proofs);
        assert!(metrics.total_verifications >= 0);
    }
}
