// Vectorized Hash Computation using SIMD
// Parallel hashing for merkle trees and block validation

use anyhow::Result;
use std::sync::Arc;
use tracing::{debug, warn};
use crate::{cpu_detection::{CpuFeatures, optimal_batch_size}, HashAlgorithm};

/// Batch of data to be hashed
#[derive(Debug, Clone)]
pub struct HashBatch {
    pub data: Vec<Vec<u8>>,
    pub algorithm: HashAlgorithm,
}

/// Result of batch hash computation
#[derive(Debug, Clone)]
pub struct HashBatchResult {
    pub hashes: Vec<Vec<u8>>,
    pub duration_us: u64,
    pub throughput_mb_per_sec: f64,
}

/// SIMD-optimized hash computation engine
#[derive(Debug)]
pub struct SimdHasher {
    cpu_features: CpuFeatures,
    max_batch_size: usize,
    optimal_batch_size: usize,
}

impl SimdHasher {
    /// Create new SIMD hasher
    pub async fn new(cpu_features: &CpuFeatures, max_batch_size: usize) -> Result<Self> {
        let optimal_batch_size = optimal_batch_size(cpu_features, 32); // Average hash input size
        
        debug!("Creating SIMD hasher:");
        debug!("  Max batch size: {}", max_batch_size);
        debug!("  Optimal batch size: {}", optimal_batch_size);
        debug!("  CPU features: SHA-NI={}, AES-NI={}", cpu_features.has_sha_ni, cpu_features.has_aes_ni);
        
        Ok(Self {
            cpu_features: cpu_features.clone(),
            max_batch_size,
            optimal_batch_size,
        })
    }
    
    /// Compute hashes for a batch of data using SIMD optimizations
    pub async fn compute_batch(
        &self,
        data: &[&[u8]],
        algorithm: HashAlgorithm,
    ) -> Result<Vec<Vec<u8>>> {
        let start_time = std::time::Instant::now();
        
        let total_items = data.len();
        let mut results = Vec::with_capacity(total_items);
        
        // Process data in optimal-sized batches
        let batch_size = std::cmp::min(self.optimal_batch_size, total_items);
        
        for chunk_start in (0..total_items).step_by(batch_size) {
            let chunk_end = std::cmp::min(chunk_start + batch_size, total_items);
            
            let chunk_results = self.compute_chunk(
                &data[chunk_start..chunk_end],
                algorithm,
            ).await?;
            
            results.extend(chunk_results);
        }
        
        let duration = start_time.elapsed();
        let duration_us = duration.as_micros() as u64;
        
        let total_bytes: usize = data.iter().map(|d| d.len()).sum();
        let throughput_mb_per_sec = if duration_us > 0 {
            (total_bytes as f64) * 1_000_000.0 / (duration_us as f64) / (1024.0 * 1024.0)
        } else {
            f64::INFINITY
        };
        
        debug!("Batch hashing completed: {} items, {:.2} MB/s", 
               total_items, throughput_mb_per_sec);
        
        Ok(results)
    }
    
    /// Compute hashes for a chunk of data using best available implementation
    async fn compute_chunk(
        &self,
        data: &[&[u8]],
        algorithm: HashAlgorithm,
    ) -> Result<Vec<Vec<u8>>> {
        match algorithm {
            HashAlgorithm::Sha256 => self.compute_sha256_batch(data).await,
            HashAlgorithm::Sha3_256 => self.compute_sha3_batch(data).await,
            HashAlgorithm::Blake3 => self.compute_blake3_batch(data).await,
        }
    }
    
    /// Compute SHA-256 hashes using hardware acceleration if available
    async fn compute_sha256_batch(&self, data: &[&[u8]]) -> Result<Vec<Vec<u8>>> {
        if self.cpu_features.has_sha_ni {
            self.compute_sha256_with_sha_ni(data).await
        } else if self.cpu_features.has_avx2 {
            self.compute_sha256_with_avx2(data).await
        } else {
            self.compute_sha256_scalar(data).await
        }
    }
    
    /// SHA-256 with SHA-NI (SHA New Instructions) acceleration
    #[cfg(target_arch = "x86_64")]
    async fn compute_sha256_with_sha_ni(&self, data: &[&[u8]]) -> Result<Vec<Vec<u8>>> {
        debug!("Using SHA-NI for SHA-256 batch computation of {} items", data.len());
        
        // Use ring's hardware-accelerated implementation when available
        let mut results = Vec::with_capacity(data.len());
        
        for input in data {
            let digest = ring::digest::digest(&ring::digest::SHA256, input);
            results.push(digest.as_ref().to_vec());
        }
        
        Ok(results)
    }
    
    /// SHA-256 with AVX2 vectorization
    #[cfg(target_arch = "x86_64")]
    async fn compute_sha256_with_avx2(&self, data: &[&[u8]]) -> Result<Vec<Vec<u8>>> {
        debug!("Using AVX2 for SHA-256 batch computation of {} items", data.len());
        
        // For now, use the same implementation as SHA-NI
        // Full AVX2 vectorization would require implementing parallel SHA-256 rounds
        self.compute_sha256_with_sha_ni(data).await
    }
    
    /// Scalar SHA-256 implementation
    async fn compute_sha256_scalar(&self, data: &[&[u8]]) -> Result<Vec<Vec<u8>>> {
        debug!("Using scalar SHA-256 for {} items", data.len());
        
        let mut results = Vec::with_capacity(data.len());
        
        for input in data {
            use sha2::{Sha256, Digest};
            let mut hasher = Sha256::new();
            hasher.update(input);
            let hash = hasher.finalize().to_vec();
            results.push(hash);
        }
        
        Ok(results)
    }
    
    /// Compute SHA3-256 hashes in batch
    async fn compute_sha3_batch(&self, data: &[&[u8]]) -> Result<Vec<Vec<u8>>> {
        debug!("Computing SHA3-256 for {} items", data.len());
        
        let mut results = Vec::with_capacity(data.len());
        
        for input in data {
            use sha3::{Sha3_256, Digest};
            let mut hasher = Sha3_256::new();
            hasher.update(input);
            let hash = hasher.finalize().to_vec();
            results.push(hash);
        }
        
        Ok(results)
    }
    
    /// Compute Blake3 hashes in batch (Blake3 has built-in parallelization)
    async fn compute_blake3_batch(&self, data: &[&[u8]]) -> Result<Vec<Vec<u8>>> {
        debug!("Computing Blake3 for {} items", data.len());
        
        let mut results = Vec::with_capacity(data.len());
        
        for input in data {
            let hash = blake3::hash(input);
            results.push(hash.as_bytes().to_vec());
        }
        
        Ok(results)
    }
    
    /// Compute a merkle tree root using SIMD-optimized hashing
    pub async fn compute_merkle_root(&self, leaves: &[&[u8]]) -> Result<Vec<u8>> {
        if leaves.is_empty() {
            return Ok(vec![0u8; 32]); // Empty tree root
        }
        
        if leaves.len() == 1 {
            return Ok(blake3::hash(leaves[0]).as_bytes().to_vec());
        }
        
        // Compute leaf hashes in parallel
        let mut level = self.compute_batch(leaves, HashAlgorithm::Blake3).await?;
        
        // Build tree bottom-up
        while level.len() > 1 {
            let mut next_level = Vec::new();
            
            // Process pairs of hashes
            for chunk in level.chunks(2) {
                if chunk.len() == 2 {
                    // Combine two hashes
                    let mut combined = Vec::with_capacity(64);
                    combined.extend_from_slice(&chunk[0]);
                    combined.extend_from_slice(&chunk[1]);
                    
                    let parent_hash = blake3::hash(&combined).as_bytes().to_vec();
                    next_level.push(parent_hash);
                } else {
                    // Odd number of nodes, promote the last one
                    next_level.push(chunk[0].clone());
                }
            }
            
            level = next_level;
        }
        
        Ok(level.into_iter().next().unwrap())
    }
    
    /// Get estimated throughput for this hasher
    pub fn throughput_estimate(&self) -> f64 {
        // Rough estimates in MB/s based on CPU features
        if self.cpu_features.has_sha_ni {
            2000.0  // 2 GB/s with SHA-NI
        } else if self.cpu_features.has_avx2 {
            1000.0  // 1 GB/s with AVX2
        } else if self.cpu_features.has_neon {
            500.0   // 500 MB/s with NEON
        } else {
            250.0   // 250 MB/s scalar
        }
    }
}

/// Stub implementations for non-x86_64 platforms
#[cfg(not(target_arch = "x86_64"))]
impl SimdHasher {
    async fn compute_sha256_with_sha_ni(&self, data: &[&[u8]]) -> Result<Vec<Vec<u8>>> {
        self.compute_sha256_scalar(data).await
    }
    
    async fn compute_sha256_with_avx2(&self, data: &[&[u8]]) -> Result<Vec<Vec<u8>>> {
        self.compute_sha256_scalar(data).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu_detection::detect_cpu_features;
    
    #[tokio::test]
    async fn test_simd_hasher_creation() {
        let cpu_features = detect_cpu_features();
        let hasher = SimdHasher::new(&cpu_features, 32).await.unwrap();
        
        assert!(hasher.throughput_estimate() > 0.0);
    }
    
    #[tokio::test]
    async fn test_batch_hashing() {
        let cpu_features = detect_cpu_features();
        let hasher = SimdHasher::new(&cpu_features, 32).await.unwrap();
        
        let data = vec![b"hello".as_slice(), b"world".as_slice(), b"test".as_slice()];
        let results = hasher.compute_batch(&data, HashAlgorithm::Blake3).await.unwrap();
        
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].len(), 32); // Blake3 output size
    }
    
    #[tokio::test]
    async fn test_merkle_root_computation() {
        let cpu_features = detect_cpu_features();
        let hasher = SimdHasher::new(&cpu_features, 32).await.unwrap();
        
        let leaves = vec![b"leaf1".as_slice(), b"leaf2".as_slice(), b"leaf3".as_slice()];
        let root = hasher.compute_merkle_root(&leaves).await.unwrap();
        
        assert_eq!(root.len(), 32); // Blake3 hash size
    }
    
    #[tokio::test]
    async fn test_empty_merkle_root() {
        let cpu_features = detect_cpu_features();
        let hasher = SimdHasher::new(&cpu_features, 32).await.unwrap();
        
        let root = hasher.compute_merkle_root(&[]).await.unwrap();
        assert_eq!(root.len(), 32);
        assert_eq!(root, vec![0u8; 32]); // Empty tree root
    }
}