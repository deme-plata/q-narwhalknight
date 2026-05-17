//! Genus-2 VDF Integration for Quantum-Resistant Anchor Election
//!
//! This module integrates the Genus-2 hyperelliptic curve VDF from IACR 2025/1050
//! for quantum-resistant time-locking in DAG-Knight consensus.
//!
//! ## Why Genus-2 VDF?
//!
//! Traditional VDFs (RSA, class group) are vulnerable to Shor's algorithm.
//! Genus-2 hyperelliptic curves provide:
//! - Resistance to quantum attacks (Shor's algorithm ineffective)
//! - Efficient verification via Wesolowski's protocol
//! - Smaller parameters than lattice-based alternatives
//!
//! ## Integration with DAG-Knight
//!
//! The Genus-2 VDF is used for:
//! 1. **Anchor Election**: Time-locked randomness for fair leader selection
//! 2. **Commit Delay**: Verifiable delay for commit decisions
//! 3. **Quantum Beacon**: High-quality entropy injection into consensus
//!
//! ## Usage
//! ```ignore
//! use q_dag_knight::genus2_vdf_integration::{Genus2VDFEngine, Genus2VDFConfig};
//!
//! // Create engine with quantum-safe parameters
//! let config = Genus2VDFConfig::quantum_safe();
//! let engine = Genus2VDFEngine::new(config)?;
//!
//! // Compute verifiable delay
//! let challenge = [0u8; 32];
//! let result = engine.compute_delay(&challenge, 1000).await?;
//!
//! // Verify (cannot be parallelized, proves time passed)
//! assert!(engine.verify(&result)?);
//! ```

#[cfg(feature = "advanced-crypto")]
use q_crypto_advanced::genus2_vdf::{
    Genus2Params, Genus2Vdf, Genus2Level,
    VdfOutput, VdfBatchVerifier,
};

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// Configuration for Genus-2 VDF engine
#[derive(Debug, Clone)]
pub struct Genus2VDFConfig {
    /// Security level (affects curve parameters)
    pub security_level: Genus2SecurityLevel,
    /// Minimum iterations for acceptable delay
    pub min_iterations: u64,
    /// Maximum iterations (safety limit)
    pub max_iterations: u64,
    /// Target computation time in milliseconds
    pub target_time_ms: u64,
    /// Enable parallel checkpoint verification
    pub parallel_verification: bool,
}

/// Security levels for Genus-2 VDF
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Genus2SecurityLevel {
    /// 128-bit classical / quantum security
    Standard,
    /// 192-bit security for high-value operations
    High,
    /// 256-bit security for critical consensus operations
    Paranoid,
}

impl Default for Genus2VDFConfig {
    fn default() -> Self {
        Self {
            security_level: Genus2SecurityLevel::Standard,
            min_iterations: 1000,
            max_iterations: 100_000_000,
            target_time_ms: 1000, // 1 second target
            parallel_verification: true,
        }
    }
}

impl Genus2VDFConfig {
    /// Create quantum-safe configuration (recommended)
    /// v1.5.0-beta: Aligned min_iterations with anchor_election.rs Phase::Phase1 (5000)
    pub fn quantum_safe() -> Self {
        Self {
            security_level: Genus2SecurityLevel::Standard,
            min_iterations: 5_000,  // Matches Phase1 anchor election
            max_iterations: 10_000_000,
            target_time_ms: 2000, // 2 seconds for security
            parallel_verification: true,
        }
    }

    /// Create high-security configuration
    /// v1.5.0-beta: Aligned min_iterations with anchor_election.rs Phase::Phase2 (10000)
    pub fn high_security() -> Self {
        Self {
            security_level: Genus2SecurityLevel::High,
            min_iterations: 10_000,  // Matches Phase2 anchor election
            max_iterations: 100_000_000,
            target_time_ms: 5000, // 5 seconds
            parallel_verification: true,
        }
    }
}

/// VDF computation result with quantum-resistant proof
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Genus2VDFResult {
    /// Input challenge
    pub challenge: [u8; 32],
    /// Output after delay
    pub output: Vec<u8>,
    /// Proof of sequential computation
    pub proof: Vec<u8>,
    /// Number of iterations performed
    pub iterations: u64,
    /// Actual computation time
    pub computation_time_ms: u64,
    /// Security level used
    pub security_level: Genus2SecurityLevel,
    /// Checkpoints for parallel verification
    pub checkpoints: Vec<[u8; 32]>,
}

impl Genus2VDFResult {
    /// Get the output as a 32-byte hash (for consensus use)
    pub fn output_hash(&self) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(&self.output);
        hasher.finalize().into()
    }

    /// Get entropy quality estimate (0.0 - 1.0)
    pub fn entropy_quality(&self) -> f64 {
        // Higher iterations = better entropy mixing
        let iteration_factor = (self.iterations as f64 / 10_000.0).min(1.0);
        // More checkpoints = better verification
        let checkpoint_factor = (self.checkpoints.len() as f64 / 16.0).min(1.0);
        // Security level bonus
        let security_factor = match self.security_level {
            Genus2SecurityLevel::Standard => 0.8,
            Genus2SecurityLevel::High => 0.9,
            Genus2SecurityLevel::Paranoid => 1.0,
        };

        (iteration_factor * 0.4 + checkpoint_factor * 0.2 + security_factor * 0.4).min(1.0)
    }
}

/// Genus-2 VDF engine for quantum-resistant time-locking
#[cfg(feature = "advanced-crypto")]
pub struct Genus2VDFEngine {
    config: Genus2VDFConfig,
    params: Genus2Params,
    vdf: Genus2Vdf,
}

#[cfg(feature = "advanced-crypto")]
impl Genus2VDFEngine {
    /// Create new Genus-2 VDF engine
    pub fn new(config: Genus2VDFConfig) -> Result<Self> {
        let level = match config.security_level {
            Genus2SecurityLevel::Standard => Genus2Level::Standard,
            Genus2SecurityLevel::High => Genus2Level::High,
            Genus2SecurityLevel::Paranoid => Genus2Level::Paranoid,
        };

        let params = Genus2Params::new(level)?;
        let vdf = Genus2Vdf::new(params.clone());

        info!(
            "Genus-2 VDF engine initialized with {:?} security",
            config.security_level
        );

        Ok(Self { config, params, vdf })
    }

    /// Compute verifiable delay function
    pub async fn compute_delay(
        &self,
        challenge: &[u8; 32],
        iterations: u64,
    ) -> Result<Genus2VDFResult> {
        // Validate iterations
        if iterations < self.config.min_iterations {
            return Err(anyhow!(
                "Iterations {} below minimum {}",
                iterations,
                self.config.min_iterations
            ));
        }
        if iterations > self.config.max_iterations {
            return Err(anyhow!(
                "Iterations {} exceeds maximum {}",
                iterations,
                self.config.max_iterations
            ));
        }

        let start = Instant::now();
        debug!(
            "Starting Genus-2 VDF computation: {} iterations",
            iterations
        );

        // Compute VDF (sequential, cannot be parallelized)
        let output = self.vdf.evaluate(challenge, iterations)?;

        let computation_time = start.elapsed();
        let computation_time_ms = computation_time.as_millis() as u64;

        info!(
            "Genus-2 VDF computed {} iterations in {}ms",
            iterations, computation_time_ms
        );

        // Generate checkpoints for parallel verification
        let checkpoints = self.generate_checkpoints(challenge, iterations)?;

        Ok(Genus2VDFResult {
            challenge: *challenge,
            output: output.to_bytes(),
            proof: output.proof_bytes(),
            iterations,
            computation_time_ms,
            security_level: self.config.security_level,
            checkpoints,
        })
    }

    /// Compute VDF with automatic iteration calibration
    pub async fn compute_timed_delay(
        &self,
        challenge: &[u8; 32],
        target_time_ms: u64,
    ) -> Result<Genus2VDFResult> {
        // Calibrate iterations based on target time
        let iterations = self.calibrate_iterations(target_time_ms)?;
        self.compute_delay(challenge, iterations).await
    }

    /// Verify VDF result
    pub fn verify(&self, result: &Genus2VDFResult) -> Result<bool> {
        let output = VdfOutput::from_bytes(&result.output, &self.params)?;

        if self.config.parallel_verification && !result.checkpoints.is_empty() {
            // Use parallel checkpoint verification
            self.verify_with_checkpoints(result)
        } else {
            // Standard verification
            self.vdf
                .verify(&result.challenge, &output)
                .map_err(|e| anyhow!("Verification failed: {}", e))
        }
    }

    /// Verify using checkpoints (faster, parallelizable)
    ///
    /// # Security
    /// This function now includes proper bounds checking to prevent
    /// array index out of bounds panics from malicious inputs.
    fn verify_with_checkpoints(&self, result: &Genus2VDFResult) -> Result<bool> {
        // SECURITY: Validate checkpoint array is not empty before division
        if result.checkpoints.is_empty() {
            // Fall back to standard verification if no checkpoints
            return self.vdf
                .verify(&result.challenge, &VdfOutput::from_bytes(&result.output, &self.params)?)
                .map_err(|e| anyhow!("Verification failed: {}", e));
        }

        // SECURITY: Validate iterations to prevent division by zero
        let checkpoint_count = result.checkpoints.len() as u64;
        if checkpoint_count == 0 || result.iterations == 0 {
            return Err(anyhow!("Invalid checkpoint or iteration count"));
        }

        // Verify each checkpoint segment
        let segment_size = result.iterations.checked_div(checkpoint_count + 1)
            .ok_or_else(|| anyhow!("Division overflow in segment calculation"))?;

        for (i, checkpoint) in result.checkpoints.iter().enumerate() {
            // SECURITY: Explicit bounds check (though enumerate guarantees i < len)
            let segment_start = if i == 0 {
                result.challenge
            } else {
                // SECURITY: i-1 is safe here because i > 0
                // But we add explicit bounds check for defense in depth
                match result.checkpoints.get(i.saturating_sub(1)) {
                    Some(prev) => *prev,
                    None => {
                        warn!("Checkpoint index {} out of bounds", i.saturating_sub(1));
                        return Ok(false);
                    }
                }
            };

            // Verify segment
            let output = VdfOutput::from_bytes(&checkpoint[..], &self.params)?;
            if !self.vdf.verify(&segment_start, &output)? {
                warn!("Checkpoint {} verification failed", i);
                return Ok(false);
            }
        }

        // Verify final segment
        // SECURITY: Use get().copied() pattern instead of unwrap_or
        let final_start = result.checkpoints.last()
            .copied()
            .unwrap_or(result.challenge);
        let final_output = VdfOutput::from_bytes(&result.output, &self.params)?;
        Ok(self.vdf.verify(&final_start, &final_output)?)
    }

    /// Generate checkpoints for parallel verification
    fn generate_checkpoints(
        &self,
        challenge: &[u8; 32],
        iterations: u64,
    ) -> Result<Vec<[u8; 32]>> {
        let num_checkpoints: usize = 8; // 8 checkpoints for ~8x parallel verification speedup
        let segment_size = iterations / (num_checkpoints as u64 + 1);

        let mut checkpoints = Vec::with_capacity(num_checkpoints);
        let mut current = *challenge;

        for i in 0..num_checkpoints {
            let _target_iteration = ((i + 1) as u64) * segment_size;
            let output = self.vdf.evaluate(&current, segment_size)?;

            let mut checkpoint = [0u8; 32];
            let output_bytes = output.to_bytes();
            let copy_len = output_bytes.len().min(32);
            checkpoint[..copy_len].copy_from_slice(&output_bytes[..copy_len]);

            checkpoints.push(checkpoint);
            current = checkpoint;
        }

        Ok(checkpoints)
    }

    /// Calibrate iterations for target time
    fn calibrate_iterations(&self, target_time_ms: u64) -> Result<u64> {
        // Use a small sample to estimate iterations/ms
        let sample_iterations = 1000u64;
        let start = Instant::now();

        let challenge = [0u8; 32];
        let _ = self.vdf.evaluate(&challenge, sample_iterations)?;

        let sample_time_ms = start.elapsed().as_millis() as u64;
        if sample_time_ms == 0 {
            return Ok(self.config.min_iterations);
        }

        let iterations_per_ms = sample_iterations / sample_time_ms.max(1);
        let estimated_iterations = iterations_per_ms * target_time_ms;

        // Clamp to valid range
        Ok(estimated_iterations
            .max(self.config.min_iterations)
            .min(self.config.max_iterations))
    }

    /// Get current configuration
    pub fn config(&self) -> &Genus2VDFConfig {
        &self.config
    }
}

/// Batch verifier for multiple VDF results
#[cfg(feature = "advanced-crypto")]
pub struct Genus2BatchVerifier {
    verifier: VdfBatchVerifier,
    params: Genus2Params,
    results: Vec<Genus2VDFResult>,
}

#[cfg(feature = "advanced-crypto")]
impl Genus2BatchVerifier {
    /// Create new batch verifier
    pub fn new(security_level: Genus2SecurityLevel) -> Result<Self> {
        let level = match security_level {
            Genus2SecurityLevel::Standard => Genus2Level::Standard,
            Genus2SecurityLevel::High => Genus2Level::High,
            Genus2SecurityLevel::Paranoid => Genus2Level::Paranoid,
        };

        let params = Genus2Params::new(level)?;
        let verifier = VdfBatchVerifier::new(params.clone());

        Ok(Self {
            verifier,
            params,
            results: Vec::new(),
        })
    }

    /// Add result to batch
    pub fn add(&mut self, result: Genus2VDFResult) {
        self.results.push(result);
    }

    /// Verify all results in batch
    ///
    /// Returns true if ALL results verified successfully, false otherwise.
    pub fn verify_all(&mut self) -> Result<bool> {
        // Add all pending results to the verifier
        for result in &self.results {
            let output = VdfOutput::from_bytes(&result.output, &self.params)?;
            self.verifier.add(result.challenge.to_vec(), output);
        }

        // Verify and check all results
        let verification_results = self.verifier.verify_all()
            .map_err(|e| anyhow!("Batch verification failed: {}", e))?;

        // Return true only if ALL verifications passed
        Ok(verification_results.iter().all(|&v| v))
    }

    /// Get number of results in batch
    pub fn count(&self) -> usize {
        self.results.len()
    }

    /// Clear all pending verifications
    pub fn clear(&mut self) {
        self.results.clear();
        self.verifier.clear();
    }
}

/// Convert Genus-2 VDF result to DAG-Knight compatible format
#[cfg(feature = "advanced-crypto")]
impl Genus2VDFResult {
    /// Convert to QuantumVDFProof format for backward compatibility
    pub fn to_quantum_vdf_proof(&self) -> super::quantum_vdf::QuantumVDFProof {
        let mut proof_bytes = [0u8; 64];
        let copy_len = self.proof.len().min(64);
        proof_bytes[..copy_len].copy_from_slice(&self.proof[..copy_len]);

        super::quantum_vdf::QuantumVDFProof {
            challenge: self.challenge,
            proof: proof_bytes,
            quantum_seed: Some(self.output_hash()),
            computation_time: Duration::from_millis(self.computation_time_ms),
            difficulty: self.iterations,
            entropy_estimate: self.entropy_quality(),
            parallel_witnesses: self.checkpoints.clone(),
        }
    }
}

// Fallback for when advanced-crypto is disabled
#[cfg(not(feature = "advanced-crypto"))]
pub struct Genus2VDFEngine;

#[cfg(not(feature = "advanced-crypto"))]
impl Genus2VDFEngine {
    pub fn new(_config: Genus2VDFConfig) -> Result<Self> {
        Err(anyhow!(
            "Genus-2 VDF requires the 'advanced-crypto' feature. Enable it in Cargo.toml."
        ))
    }
}

#[cfg(all(test, feature = "advanced-crypto"))]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_genus2_vdf_creation() {
        let config = Genus2VDFConfig::default();
        let engine = Genus2VDFEngine::new(config);
        assert!(engine.is_ok());
    }

    #[tokio::test]
    async fn test_genus2_vdf_computation() {
        let config = Genus2VDFConfig {
            min_iterations: 100,
            max_iterations: 10_000,
            ..Default::default()
        };
        let engine = Genus2VDFEngine::new(config).unwrap();

        let challenge = [42u8; 32];
        let result = engine.compute_delay(&challenge, 1000).await;

        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.challenge, challenge);
        assert_eq!(result.iterations, 1000);
        assert!(result.computation_time_ms > 0);
    }

    #[tokio::test]
    async fn test_genus2_vdf_verification() {
        let config = Genus2VDFConfig {
            min_iterations: 100,
            max_iterations: 10_000,
            parallel_verification: false,
            ..Default::default()
        };
        let engine = Genus2VDFEngine::new(config).unwrap();

        let challenge = [1u8; 32];
        let result = engine.compute_delay(&challenge, 500).await.unwrap();

        let is_valid = engine.verify(&result);
        assert!(is_valid.is_ok());
        assert!(is_valid.unwrap());
    }

    #[tokio::test]
    async fn test_genus2_entropy_quality() {
        let config = Genus2VDFConfig {
            min_iterations: 100,
            ..Default::default()
        };
        let engine = Genus2VDFEngine::new(config).unwrap();

        let challenge = [99u8; 32];
        let result = engine.compute_delay(&challenge, 10_000).await.unwrap();

        let quality = result.entropy_quality();
        assert!(quality > 0.5, "Entropy quality should be > 0.5");
        assert!(quality <= 1.0, "Entropy quality should be <= 1.0");
    }

    #[tokio::test]
    async fn test_quantum_vdf_proof_conversion() {
        let config = Genus2VDFConfig {
            min_iterations: 100,
            ..Default::default()
        };
        let engine = Genus2VDFEngine::new(config).unwrap();

        let challenge = [7u8; 32];
        let result = engine.compute_delay(&challenge, 1000).await.unwrap();

        let proof = result.to_quantum_vdf_proof();
        assert_eq!(proof.challenge, challenge);
        assert_eq!(proof.difficulty, 1000);
        assert!(proof.quantum_seed.is_some());
    }
}
