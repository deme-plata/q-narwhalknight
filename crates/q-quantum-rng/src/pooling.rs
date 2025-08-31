/// Entropy pooling and management for quantum random number generation

use std::collections::VecDeque;
use sha3::{Digest, Sha3_256};
use tracing::{debug, warn};

/// Entropy pool for buffering and mixing quantum randomness
pub struct EntropyPool {
    /// Main entropy buffer
    buffer: VecDeque<u8>,
    
    /// Maximum pool size
    max_size: usize,
    
    /// Quality scores for entropy chunks
    quality_scores: VecDeque<f64>,
    
    /// Mixing state for von Neumann whitening
    mixing_state: [u8; 32],
    
    /// Pool statistics
    stats: PoolStats,
}

/// Pool statistics
#[derive(Debug, Clone)]
pub struct PoolStats {
    pub total_entropy_added: u64,
    pub total_entropy_extracted: u64,
    pub average_quality: f64,
    pub current_fill_level: f64,
    pub mixing_operations: u64,
}

/// Pool status information
#[derive(Debug, Clone)]
pub struct PoolStatus {
    pub available_bytes: usize,
    pub capacity: usize,
    pub fill_percentage: f64,
    pub average_quality: f64,
    pub is_healthy: bool,
}

/// Pooling strategy for entropy management
#[derive(Debug, Clone)]
pub enum PoolingStrategy {
    /// First In, First Out
    FIFO,
    
    /// Quality-based selection (highest quality first)
    QualityBased,
    
    /// Mixed strategy with randomization
    Mixed,
}

impl EntropyPool {
    /// Create new entropy pool
    pub fn new(max_size: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(max_size),
            max_size,
            quality_scores: VecDeque::new(),
            mixing_state: [0u8; 32],
            stats: PoolStats {
                total_entropy_added: 0,
                total_entropy_extracted: 0,
                average_quality: 0.0,
                current_fill_level: 0.0,
                mixing_operations: 0,
            },
        }
    }

    /// Add entropy to the pool
    pub fn add_entropy(&mut self, entropy: Vec<u8>, quality: f64) {
        debug!("Adding {} bytes of entropy (quality: {:.3})", entropy.len(), quality);

        // If pool is full, remove oldest entropy
        while self.buffer.len() + entropy.len() > self.max_size {
            self.remove_oldest_chunk();
        }

        // Add new entropy
        let chunk_size = entropy.len();
        for byte in entropy {
            self.buffer.push_back(byte);
        }
        self.quality_scores.push_back(quality);

        // Update statistics
        self.stats.total_entropy_added += chunk_size as u64;
        self.update_average_quality();
        self.stats.current_fill_level = self.buffer.len() as f64 / self.max_size as f64;

        // Mix entropy periodically
        if self.buffer.len() >= 256 {
            self.mix_entropy();
        }
    }

    /// Extract bytes from entropy pool
    pub fn extract_bytes(&mut self, count: usize) -> Option<Vec<u8>> {
        if self.buffer.len() < count {
            return None;
        }

        let mut extracted = Vec::with_capacity(count);
        for _ in 0..count {
            if let Some(byte) = self.buffer.pop_front() {
                extracted.push(byte);
            }
        }

        // Update quality scores (remove corresponding entries)
        if !self.quality_scores.is_empty() {
            // Approximate quality tracking - in real implementation would be more precise
            self.quality_scores.pop_front();
        }

        self.stats.total_entropy_extracted += count as u64;
        self.stats.current_fill_level = self.buffer.len() as f64 / self.max_size as f64;

        debug!("Extracted {} bytes from entropy pool", count);
        Some(extracted)
    }

    /// Get pool status
    pub fn get_status(&self) -> PoolStatus {
        PoolStatus {
            available_bytes: self.buffer.len(),
            capacity: self.max_size,
            fill_percentage: self.stats.current_fill_level * 100.0,
            average_quality: self.stats.average_quality,
            is_healthy: self.is_healthy(),
        }
    }

    /// Check if pool is healthy
    fn is_healthy(&self) -> bool {
        self.stats.average_quality > 0.8 && 
        self.stats.current_fill_level > 0.1
    }

    /// Remove oldest entropy chunk
    fn remove_oldest_chunk(&mut self) {
        // Remove approximately one chunk (256 bytes)
        let chunk_size = 256.min(self.buffer.len());
        for _ in 0..chunk_size {
            self.buffer.pop_front();
        }
        
        if !self.quality_scores.is_empty() {
            self.quality_scores.pop_front();
        }

        warn!("Removed oldest entropy chunk due to pool overflow");
    }

    /// Update average quality score
    fn update_average_quality(&mut self) {
        if self.quality_scores.is_empty() {
            self.stats.average_quality = 0.0;
            return;
        }

        let sum: f64 = self.quality_scores.iter().sum();
        self.stats.average_quality = sum / self.quality_scores.len() as f64;
    }

    /// Mix entropy using cryptographic techniques
    fn mix_entropy(&mut self) {
        if self.buffer.len() < 64 {
            return;
        }

        debug!("Mixing entropy pool");

        // Extract 32 bytes for mixing
        let mut mix_material = Vec::new();
        for _ in 0..32 {
            if let Some(byte) = self.buffer.pop_front() {
                mix_material.push(byte);
            }
        }

        // Hash with current mixing state
        let mut hasher = Sha3_256::new();
        hasher.update(&self.mixing_state);
        hasher.update(&mix_material);
        let hash = hasher.finalize();

        // Update mixing state
        self.mixing_state.copy_from_slice(&hash);

        // Add mixed entropy back to pool
        for &byte in hash.iter() {
            self.buffer.push_back(byte);
        }

        self.stats.mixing_operations += 1;
    }

    /// Perform von Neumann debiasing
    pub fn von_neumann_debias(&self, input: &[u8]) -> Vec<u8> {
        let mut output = Vec::new();
        let mut i = 0;

        while i + 1 < input.len() {
            let bit1 = input[i] & 1;
            let bit2 = input[i + 1] & 1;

            match (bit1, bit2) {
                (0, 1) => output.push(0),
                (1, 0) => output.push(1),
                _ => {}, // Discard 00 and 11 pairs
            }

            i += 2;
        }

        output
    }

    /// Apply quality-based filtering
    pub fn quality_filter(&mut self, min_quality: f64) {
        let mut filtered_buffer = VecDeque::new();
        let mut filtered_scores = VecDeque::new();
        let mut removed_count = 0;

        // This is a simplified approach - real implementation would track chunks precisely
        while let (Some(score), Some(byte)) = (self.quality_scores.pop_front(), self.buffer.pop_front()) {
            if score >= min_quality {
                filtered_buffer.push_back(byte);
                filtered_scores.push_back(score);
            } else {
                removed_count += 1;
            }
        }

        self.buffer = filtered_buffer;
        self.quality_scores = filtered_scores;

        if removed_count > 0 {
            warn!("Filtered out {} low-quality entropy bytes", removed_count);
            self.update_average_quality();
        }
    }

    /// Get pool statistics
    pub fn get_stats(&self) -> PoolStats {
        self.stats.clone()
    }

    /// Clear the entropy pool
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.quality_scores.clear();
        self.mixing_state = [0u8; 32];
        
        // Reset stats but preserve totals
        self.stats.current_fill_level = 0.0;
        self.stats.average_quality = 0.0;
        self.stats.mixing_operations = 0;
    }

    /// Estimate entropy content using Shannon entropy
    pub fn estimate_entropy(&self) -> f64 {
        if self.buffer.is_empty() {
            return 0.0;
        }

        let mut counts = [0u32; 256];
        for &byte in &self.buffer {
            counts[byte as usize] += 1;
        }

        let length = self.buffer.len() as f64;
        let mut entropy = 0.0;

        for &count in &counts {
            if count > 0 {
                let p = count as f64 / length;
                entropy -= p * p.log2();
            }
        }

        entropy
    }

    /// Compact the pool by removing low-quality entropy
    pub fn compact(&mut self, target_fill: f64) {
        if self.stats.current_fill_level <= target_fill {
            return;
        }

        let target_size = (self.max_size as f64 * target_fill) as usize;
        let bytes_to_remove = self.buffer.len().saturating_sub(target_size);

        // Remove bytes with lowest quality first
        // Simplified implementation - would be more sophisticated in practice
        for _ in 0..bytes_to_remove {
            self.buffer.pop_front();
            if !self.quality_scores.is_empty() {
                self.quality_scores.pop_front();
            }
        }

        self.stats.current_fill_level = self.buffer.len() as f64 / self.max_size as f64;
        self.update_average_quality();
        
        debug!("Compacted entropy pool to {:.1}% capacity", self.stats.current_fill_level * 100.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entropy_pool_creation() {
        let pool = EntropyPool::new(1024);
        let status = pool.get_status();
        
        assert_eq!(status.available_bytes, 0);
        assert_eq!(status.capacity, 1024);
        assert_eq!(status.fill_percentage, 0.0);
    }

    #[test]
    fn test_add_extract_entropy() {
        let mut pool = EntropyPool::new(1024);
        
        // Add entropy
        let entropy = vec![1, 2, 3, 4, 5];
        pool.add_entropy(entropy, 0.9);
        
        let status = pool.get_status();
        assert_eq!(status.available_bytes, 5);
        
        // Extract entropy
        let extracted = pool.extract_bytes(3).unwrap();
        assert_eq!(extracted, vec![1, 2, 3]);
        
        let status = pool.get_status();
        assert_eq!(status.available_bytes, 2);
    }

    #[test]
    fn test_pool_overflow() {
        let mut pool = EntropyPool::new(10); // Small pool
        
        // Add more entropy than capacity
        let entropy1 = vec![1, 2, 3, 4, 5];
        let entropy2 = vec![6, 7, 8, 9, 10];
        let entropy3 = vec![11, 12, 13, 14, 15]; // This should cause overflow
        
        pool.add_entropy(entropy1, 0.9);
        pool.add_entropy(entropy2, 0.8);
        pool.add_entropy(entropy3, 0.7);
        
        let status = pool.get_status();
        assert_eq!(status.capacity, 10);
        assert!(status.available_bytes <= 10);
    }

    #[test]
    fn test_von_neumann_debiasing() {
        let pool = EntropyPool::new(1024);
        
        // Test with biased input
        let input = vec![0b00110101, 0b10101010]; // Mix of bits
        let output = pool.von_neumann_debias(&input);
        
        // Should have some output (exact amount depends on bit patterns)
        assert!(!output.is_empty());
    }

    #[test]
    fn test_quality_filtering() {
        let mut pool = EntropyPool::new(1024);
        
        // Add entropy with varying quality
        pool.add_entropy(vec![1, 2, 3], 0.9);  // High quality
        pool.add_entropy(vec![4, 5, 6], 0.5);  // Low quality
        pool.add_entropy(vec![7, 8, 9], 0.95); // High quality
        
        // Filter out low quality
        pool.quality_filter(0.8);
        
        let status = pool.get_status();
        assert!(status.average_quality >= 0.8);
    }

    #[test]
    fn test_entropy_estimation() {
        let mut pool = EntropyPool::new(1024);
        
        // Add uniform entropy
        let uniform_entropy: Vec<u8> = (0..256).map(|i| i as u8).collect();
        pool.add_entropy(uniform_entropy, 1.0);
        
        let entropy = pool.estimate_entropy();
        assert!(entropy > 7.0); // Should be close to 8.0 for uniform distribution
    }
}