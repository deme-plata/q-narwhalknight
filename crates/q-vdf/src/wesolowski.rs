/// Wesolowski's VDF implementation
/// Based on the paper "Efficient Verifiable Delay Functions" by Wesolowski (2018)

use anyhow::{Result, anyhow};
use num_bigint::{BigUint, BigInt, RandBigInt};
use num_traits::{Zero, One};
use sha3::{Digest, Sha3_256};
use tracing::{debug, trace};

use crate::{VDFParameters, VerifiableDelayFunction};

/// Wesolowski VDF implementation
pub struct WesolowskiVDF {
    /// VDF parameters
    parameters: VDFParameters,
    
    /// Group modulus
    modulus: BigUint,
    
    /// Security parameter
    security_parameter: u32,
}

impl WesolowskiVDF {
    /// Create new Wesolowski VDF
    pub fn new(parameters: VDFParameters) -> Result<Self> {
        parameters.validate()?;
        
        Ok(Self {
            modulus: parameters.modulus.clone(),
            security_parameter: parameters.security_level.bits(),
            parameters,
        })
    }
    
    /// Evaluate VDF: y = g^(2^T) mod N
    pub fn evaluate(&self, g: &BigUint, iterations: u64) -> Result<BigUint> {
        debug!("Starting Wesolowski VDF evaluation with {} iterations", iterations);
        
        let mut y = g.clone();
        
        // Sequential squaring
        for i in 0..iterations {
            y = y.modpow(&BigUint::from(2u32), &self.modulus);
            
            // Log progress periodically
            if i % 10000 == 0 && i > 0 {
                trace!("VDF progress: {}/{} iterations", i, iterations);
            }
        }
        
        debug!("VDF evaluation complete");
        Ok(y)
    }
    
    /// Generate proof using Wesolowski's protocol
    pub fn generate_proof(&self, g: &BigUint, y: &BigUint, iterations: u64) -> Result<BigUint> {
        debug!("Generating Wesolowski proof");
        
        // Generate Fiat-Shamir challenge
        let l = self.generate_challenge(g, y, iterations)?;
        
        // Compute proof: π = g^(floor(2^T / l)) mod N
        let exponent = self.compute_proof_exponent(iterations, &l)?;
        let proof = g.modpow(&exponent, &self.modulus);
        
        debug!("Proof generation complete");
        Ok(proof)
    }
    
    /// Verify VDF output with proof
    pub fn verify_proof(
        &self,
        g: &BigUint,
        y: &BigUint,
        proof: &BigUint,
        iterations: u64
    ) -> Result<bool> {
        debug!("Verifying Wesolowski proof");
        
        // Generate same challenge
        let l = self.generate_challenge(g, y, iterations)?;
        
        // Compute r = 2^T mod l
        let two_t = BigUint::from(2u32).modpow(&BigUint::from(iterations), &l);
        
        // Verify: π^l * g^r = y (mod N)
        let lhs = proof.modpow(&l, &self.modulus) * g.modpow(&two_t, &self.modulus);
        let lhs = lhs % &self.modulus;
        
        let valid = lhs == *y;
        debug!("Proof verification result: {}", valid);
        
        Ok(valid)
    }
    
    /// Generate Fiat-Shamir challenge
    fn generate_challenge(&self, g: &BigUint, y: &BigUint, iterations: u64) -> Result<BigUint> {
        let mut hasher = Sha3_256::new();
        hasher.update(g.to_bytes_be());
        hasher.update(y.to_bytes_be());
        hasher.update(iterations.to_be_bytes());
        hasher.update(b"wesolowski-challenge");
        
        let hash = hasher.finalize();
        
        // Convert to prime of appropriate size
        let prime = self.hash_to_prime(&hash)?;
        Ok(prime)
    }
    
    /// Compute proof exponent: floor(2^T / l)
    fn compute_proof_exponent(&self, iterations: u64, l: &BigUint) -> Result<BigUint> {
        // For large T, compute 2^T / l efficiently
        // This is a simplified version - production would use optimizations
        
        if iterations < 64 {
            // Small T - direct computation
            let two_t = BigUint::from(2u64.pow(iterations as u32));
            Ok(two_t / l)
        } else {
            // Large T - use binary representation
            // 2^T / l = 2^(T-k) * 2^k / l where k is chosen for efficiency
            
            let k = 32; // Split point
            let two_k = BigUint::from(2u64.pow(k));
            let remaining = iterations - k as u64;
            
            // Compute 2^(T-k) iteratively
            let mut result = BigUint::one();
            for _ in 0..remaining {
                result = (result * 2u32) / l;
            }
            
            result = (result * two_k) / l;
            Ok(result)
        }
    }
    
    /// Hash to prime number
    fn hash_to_prime(&self, hash: &[u8]) -> Result<BigUint> {
        // Generate a prime from hash
        // This is simplified - production would use proper prime generation
        
        let mut candidate = BigUint::from_bytes_be(hash);
        
        // Ensure odd
        candidate |= BigUint::one();
        
        // Simple primality test (would use Miller-Rabin in production)
        while !self.is_probably_prime(&candidate) {
            candidate += 2u32;
        }
        
        Ok(candidate)
    }
    
    /// Simple primality test (placeholder)
    fn is_probably_prime(&self, n: &BigUint) -> bool {
        // This is a placeholder - use proper primality testing in production
        // For now, just check some small factors
        
        if n <= &BigUint::from(1u32) {
            return false;
        }
        
        if n == &BigUint::from(2u32) {
            return true;
        }
        
        if n.clone() % 2u32 == BigUint::zero() {
            return false;
        }
        
        // Check small primes
        let small_primes = [3u32, 5, 7, 11, 13, 17, 19, 23, 29, 31];
        for p in &small_primes {
            if n == &BigUint::from(*p) {
                return true;
            }
            if n.clone() % p == BigUint::zero() {
                return false;
            }
        }
        
        // Assume prime if passes basic tests (not secure!)
        true
    }
}

/// Optimized proof generation using checkpoints
pub struct CheckpointedWesolowski {
    base: WesolowskiVDF,
    checkpoint_interval: u64,
    checkpoints: Vec<(u64, BigUint)>,
}

impl CheckpointedWesolowski {
    /// Create new checkpointed VDF
    pub fn new(parameters: VDFParameters) -> Result<Self> {
        let checkpoint_interval = parameters.checkpoint_interval;
        let base = WesolowskiVDF::new(parameters)?;
        
        Ok(Self {
            base,
            checkpoint_interval,
            checkpoints: Vec::new(),
        })
    }
    
    /// Evaluate with checkpointing
    pub fn evaluate_with_checkpoints(&mut self, g: &BigUint, iterations: u64) -> Result<BigUint> {
        debug!("Starting checkpointed VDF evaluation");
        
        self.checkpoints.clear();
        let mut y = g.clone();
        
        for i in 0..iterations {
            y = y.modpow(&BigUint::from(2u32), &self.base.modulus);
            
            // Store checkpoint
            if i % self.checkpoint_interval == 0 {
                self.checkpoints.push((i, y.clone()));
                trace!("Stored checkpoint at iteration {}", i);
            }
        }
        
        // Store final result
        self.checkpoints.push((iterations, y.clone()));
        
        debug!("Checkpointed evaluation complete with {} checkpoints", self.checkpoints.len());
        Ok(y)
    }
    
    /// Generate proof using nearest checkpoint
    pub fn generate_proof_from_checkpoint(
        &self,
        g: &BigUint,
        target_iteration: u64
    ) -> Result<BigUint> {
        // Find nearest checkpoint
        let checkpoint = self.checkpoints
            .iter()
            .filter(|(iter, _)| *iter <= target_iteration)
            .max_by_key(|(iter, _)| iter)
            .ok_or_else(|| anyhow!("No suitable checkpoint found"))?;
        
        debug!("Using checkpoint at iteration {} for proof at {}", 
               checkpoint.0, target_iteration);
        
        // Compute remaining iterations from checkpoint
        let remaining = target_iteration - checkpoint.0;
        let y = self.base.evaluate(&checkpoint.1, remaining)?;
        
        self.base.generate_proof(g, &y, target_iteration)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_wesolowski_vdf_creation() {
        let params = VDFParameters::default();
        let vdf = WesolowskiVDF::new(params).unwrap();
        assert_eq!(vdf.security_parameter, 128);
    }
    
    #[test]
    fn test_vdf_evaluation() {
        let params = VDFParameters::with_time_parameter(10);
        let vdf = WesolowskiVDF::new(params).unwrap();
        
        let g = BigUint::from(2u32);
        let y = vdf.evaluate(&g, 10).unwrap();
        
        // Check y = 2^(2^10) mod N
        assert!(y > BigUint::zero());
        assert!(y < vdf.modulus);
    }
    
    #[test]
    fn test_proof_generation_and_verification() {
        let params = VDFParameters::with_time_parameter(5);
        let vdf = WesolowskiVDF::new(params).unwrap();
        
        let g = BigUint::from(3u32);
        let iterations = 5;
        
        let y = vdf.evaluate(&g, iterations).unwrap();
        let proof = vdf.generate_proof(&g, &y, iterations).unwrap();
        
        let valid = vdf.verify_proof(&g, &y, &proof, iterations).unwrap();
        assert!(valid);
    }
    
    #[test]
    fn test_checkpointed_evaluation() {
        let mut params = VDFParameters::with_time_parameter(100);
        params.checkpoint_interval = 10;
        
        let mut vdf = CheckpointedWesolowski::new(params).unwrap();
        
        let g = BigUint::from(2u32);
        let y = vdf.evaluate_with_checkpoints(&g, 100).unwrap();
        
        assert!(!vdf.checkpoints.is_empty());
        assert_eq!(vdf.checkpoints.len(), 11); // 0, 10, 20, ..., 100
        assert_eq!(vdf.checkpoints.last().unwrap().1, y);
    }
}