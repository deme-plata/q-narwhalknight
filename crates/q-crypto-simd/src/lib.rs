// Q-NarwhalKnight SIMD Crypto Optimization Library
// Phase 3: Vectorized cryptographic operations for high-performance consensus

//! # Q-Crypto-SIMD: SIMD-Optimized Cryptographic Operations
//!
//! This crate provides vectorized implementations of cryptographic primitives
//! used in the Q-NarwhalKnight consensus system. It leverages SIMD (Single 
//! Instruction, Multiple Data) instructions to achieve significant performance
//! improvements in signature verification, hashing, and post-quantum operations.
//!
//! ## Features
//!
//! - **Batch Signature Verification**: Process multiple Ed25519/Dilithium signatures simultaneously
//! - **Vectorized Hashing**: Parallel computation of SHA-256, SHA3, and Blake3 hashes
//! - **SIMD-Optimized Cache Operations**: Memory-aligned data structures for cache efficiency
//! - **Runtime CPU Detection**: Automatic selection of optimal SIMD instruction set
//! - **Post-Quantum Vectorization**: Batch operations for Kyber and Dilithium
//!
//! ## Performance Targets
//!
//! - **4-8x speedup** in cryptographic operations through vectorization
//! - **Memory-aligned operations** for optimal cache performance
//! - **Zero-copy batch processing** for high-throughput scenarios
//! - **Scalable to 500,000+ TPS** through SIMD acceleration

use anyhow::Result;
use std::sync::Arc;
use tracing::{info, debug, warn};

/// SIMD operation result with performance metrics
#[derive(Debug, Clone)]
pub struct SimdResult {
    /// Number of operations completed
    pub operations_completed: u64,
    /// Performance gain over scalar implementation (multiplier)
    pub performance_gain: f64,
    /// Energy efficiency improvement (multiplier)
    pub energy_efficiency: f64,
    /// Number of valid signatures (for verification operations)
    pub valid_signatures: u32,
}

impl SimdResult {
    /// Create new result with basic metrics
    pub fn new(operations: u64, gain: f64) -> Self {
        Self {
            operations_completed: operations,
            performance_gain: gain,
            energy_efficiency: gain * 0.7, // Estimate efficiency as 70% of performance gain
            valid_signatures: 0,
        }
    }
    
    /// Create result for signature verification operations
    pub fn with_signatures(operations: u64, gain: f64, valid_sigs: u32) -> Self {
        Self {
            operations_completed: operations,
            performance_gain: gain,
            energy_efficiency: gain * 0.7,
            valid_signatures: valid_sigs,
        }
    }
}

pub mod cpu_detection;
pub mod batch_verification;
pub mod vectorized_hashing;
pub mod cache_aligned;
pub mod avx512;
pub mod benchmarks;
pub mod parallel_ed25519;
pub mod simd_merkle;  // Phase 3.1: SIMD-optimized Merkle tree computation

// Re-export key types
pub use batch_verification::{BatchSignatureVerifier, BatchVerificationResult};
pub use vectorized_hashing::{SimdHasher, HashBatch};
pub use cache_aligned::{CacheAlignedBuffer, SimdCache};
pub use cpu_detection::{CpuFeatures, detect_cpu_features};
pub use simd_merkle::SimdMerkleTree;  // Phase 3.1 export

/// SIMD crypto engine configuration
#[derive(Debug, Clone)]
pub struct SimdCryptoConfig {
    /// Maximum batch size for signature verification
    pub max_signature_batch: usize,
    /// Maximum batch size for hash computation
    pub max_hash_batch: usize,
    /// Use AVX-512 if available
    pub enable_avx512: bool,
    /// Use AVX2 if available
    pub enable_avx2: bool,
    /// Memory alignment for cache efficiency
    pub cache_alignment: usize,
}

impl Default for SimdCryptoConfig {
    fn default() -> Self {
        // v5.1.0: Auto-scale batch sizes based on core count for high-core systems
        let num_cores = num_cpus::get();
        Self {
            max_signature_batch: (num_cores * 8).clamp(256, 4096),  // 256 on 32-core, 2048 on 256-core
            max_hash_batch: (num_cores * 4).clamp(128, 2048),       // 128 on 32-core, 1024 on 256-core
            enable_avx512: true,
            enable_avx2: true,
            cache_alignment: 64,
        }
    }
}

/// Main SIMD crypto engine for Q-NarwhalKnight
#[derive(Debug)]
pub struct SimdCryptoEngine {
    config: SimdCryptoConfig,
    cpu_features: CpuFeatures,
    batch_verifier: Arc<BatchSignatureVerifier>,
    simd_hasher: Arc<SimdHasher>,
    cache_manager: Arc<SimdCache>,
}

impl SimdCryptoEngine {
    /// Create new SIMD crypto engine with automatic CPU detection
    pub async fn new(config: SimdCryptoConfig) -> Result<Self> {
        let cpu_features = detect_cpu_features();
        
        info!("Initializing SIMD crypto engine");
        info!("CPU Features: AVX2={}, AVX-512={}, NEON={}", 
              cpu_features.has_avx2, cpu_features.has_avx512, cpu_features.has_neon);
        
        // Select optimal implementations based on CPU capabilities
        let batch_verifier = Arc::new(
            BatchSignatureVerifier::new(&cpu_features, config.max_signature_batch).await?
        );
        
        let simd_hasher = Arc::new(
            SimdHasher::new(&cpu_features, config.max_hash_batch).await?
        );
        
        let cache_manager = Arc::new(
            SimdCache::new(config.cache_alignment).await?
        );
        
        Ok(Self {
            config,
            cpu_features,
            batch_verifier,
            simd_hasher,
            cache_manager,
        })
    }
    
    /// Get reference to batch signature verifier
    pub fn batch_verifier(&self) -> &BatchSignatureVerifier {
        &self.batch_verifier
    }
    
    /// Get reference to SIMD hasher
    pub fn simd_hasher(&self) -> &SimdHasher {
        &self.simd_hasher
    }
    
    /// Get reference to cache manager
    pub fn cache_manager(&self) -> &SimdCache {
        &self.cache_manager
    }
    
    /// Get CPU features information
    pub fn cpu_features(&self) -> &CpuFeatures {
        &self.cpu_features
    }
    
    /// Get engine configuration
    pub fn config(&self) -> &SimdCryptoConfig {
        &self.config
    }
    
    /// Verify multiple signatures in a single batch operation
    /// This is the primary interface for consensus signature validation
    pub async fn batch_verify_signatures(
        &self,
        signatures: &[q_types::Signature],
        messages: &[&[u8]],
        public_keys: &[q_types::PublicKey],
    ) -> Result<BatchVerificationResult> {
        debug!("Batch verifying {} signatures using SIMD", signatures.len());
        
        if signatures.len() != messages.len() || signatures.len() != public_keys.len() {
            return Err(anyhow::anyhow!("Mismatched batch sizes"));
        }
        
        self.batch_verifier.verify_batch(signatures, messages, public_keys).await
    }
    
    /// Compute multiple hashes in parallel using SIMD
    /// Used for merkle tree computation and block hashing
    pub async fn batch_compute_hashes(
        &self,
        data: &[&[u8]],
        algorithm: HashAlgorithm,
    ) -> Result<Vec<Vec<u8>>> {
        debug!("Batch computing {} hashes using SIMD", data.len());
        
        self.simd_hasher.compute_batch(data, algorithm).await
    }
    
    /// Optimize memory layout for SIMD operations
    pub async fn align_for_simd(&self, data: &[u8]) -> Result<CacheAlignedBuffer> {
        self.cache_manager.align_buffer(data).await
    }
    
    /// Generate performance report for SIMD operations
    pub async fn performance_report(&self) -> Result<SimdPerformanceReport> {
        Ok(SimdPerformanceReport {
            cpu_features: self.cpu_features.clone(),
            signature_throughput: self.batch_verifier.throughput_estimate(),
            hash_throughput: self.simd_hasher.throughput_estimate(),
            memory_efficiency: self.cache_manager.efficiency_report().await?,
        })
    }
}

/// Hash algorithms supported by SIMD hasher
#[derive(Debug, Clone, Copy)]
pub enum HashAlgorithm {
    Sha256,
    Sha3_256,
    Blake3,
}

/// Performance report for SIMD crypto operations
#[derive(Debug, Clone)]
pub struct SimdPerformanceReport {
    pub cpu_features: CpuFeatures,
    pub signature_throughput: f64,  // signatures/second
    pub hash_throughput: f64,       // hashes/second
    pub memory_efficiency: f64,     // cache hit ratio
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_simd_engine_creation() {
        let config = SimdCryptoConfig::default();
        let engine = SimdCryptoEngine::new(config).await.unwrap();
        
        assert!(engine.cpu_features().has_avx2 || engine.cpu_features().has_neon);
    }
    
    #[tokio::test]
    async fn test_performance_report() {
        let config = SimdCryptoConfig::default();
        let engine = SimdCryptoEngine::new(config).await.unwrap();
        
        let report = engine.performance_report().await.unwrap();
        assert!(report.signature_throughput > 0.0);
        assert!(report.hash_throughput > 0.0);
    }
}