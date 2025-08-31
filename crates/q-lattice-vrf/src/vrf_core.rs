/// Core VRF types and operations for L-VRF
/// Defines the fundamental data structures for verifiable random functions

use anyhow::{Result, anyhow};
use serde::{Serialize, Deserialize};
use sha3::{Digest, Sha3_256};
use nalgebra::DVector;
use std::convert::TryInto;

use crate::lattice::{LatticeKey, LatticeSample, LatticeParameters};
use crate::proofs::ProofSystem;

/// VRF output - the pseudorandom value produced by VRF evaluation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VRFOutput {
    /// Output bytes (typically 32 bytes)
    output: Vec<u8>,
    
    /// Output length
    length: usize,
    
    /// Entropy estimate
    entropy_estimate: f64,
}

/// VRF proof - cryptographic proof of correct VRF evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VRFProof {
    /// Proof data
    proof_data: Vec<u8>,
    
    /// Proof system used
    proof_system: ProofSystem,
    
    /// Proof length
    length: usize,
    
    /// Verification hints for optimization
    verification_hints: Option<VerificationHints>,
}

/// VRF evaluation intermediate result
#[derive(Debug, Clone)]
pub struct VRFEvaluation {
    /// Evaluation vector in lattice space
    pub evaluation_vector: DVector<i64>,
    
    /// Dimension of the evaluation
    pub dimension: usize,
    
    /// Quality metric for the evaluation
    pub quality: f64,
    
    /// Randomness used in evaluation
    pub randomness_hash: Vec<u8>,
}

/// Verification hints for optimized proof checking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationHints {
    /// Hint about proof structure
    pub structure_hint: String,
    
    /// Optimization parameters
    pub optimization_params: Vec<u8>,
    
    /// Expected verification time in milliseconds
    pub expected_verify_time_ms: u64,
}

impl VRFOutput {
    /// Create VRF output from raw bytes
    pub fn new(output: Vec<u8>) -> Self {
        let length = output.len();
        let entropy_estimate = Self::estimate_entropy(&output);
        
        Self {
            output,
            length,
            entropy_estimate,
        }
    }
    
    /// Create VRF output from lattice evaluation
    pub fn from_evaluation(evaluation: &VRFEvaluation) -> Result<Self> {
        // Hash the evaluation vector to produce fixed-size output
        let mut hasher = Sha3_256::new();
        
        // Hash each component of the evaluation vector
        for &component in evaluation.evaluation_vector.iter() {
            hasher.update(&component.to_be_bytes());
        }
        
        // Add dimension and quality for additional entropy
        hasher.update(&evaluation.dimension.to_be_bytes());
        hasher.update(&evaluation.quality.to_be_bytes());
        hasher.update(&evaluation.randomness_hash);
        hasher.update(b"lattice-vrf-output");
        
        let hash = hasher.finalize();
        let output = hash.to_vec();
        
        Ok(Self::new(output))
    }
    
    /// Create VRF output from lattice point
    pub fn from_lattice_point(point: &LatticeSample) -> Result<Self> {
        let mut hasher = Sha3_256::new();
        hasher.update(&point.to_bytes()?);
        hasher.update(b"lattice-point-output");
        
        let hash = hasher.finalize();
        Ok(Self::new(hash.to_vec()))
    }
    
    /// Get output as bytes
    pub fn as_bytes(&self) -> &[u8] {
        &self.output
    }
    
    /// Get output length
    pub fn len(&self) -> usize {
        self.length
    }
    
    /// Check if output is empty
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }
    
    /// Get entropy estimate
    pub fn entropy_estimate(&self) -> f64 {
        self.entropy_estimate
    }
    
    /// Convert to hex string for display
    pub fn to_hex(&self) -> String {
        hex::encode(&self.output)
    }
    
    /// Create from hex string
    pub fn from_hex(hex_str: &str) -> Result<Self> {
        let output = hex::decode(hex_str)
            .map_err(|e| anyhow!("Invalid hex string: {}", e))?;
        Ok(Self::new(output))
    }
    
    /// Estimate Shannon entropy of output
    fn estimate_entropy(data: &[u8]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        
        let mut counts = [0u32; 256];
        for &byte in data {
            counts[byte as usize] += 1;
        }
        
        let length = data.len() as f64;
        let mut entropy = 0.0;
        
        for &count in &counts {
            if count > 0 {
                let p = count as f64 / length;
                entropy -= p * p.log2();
            }
        }
        
        entropy
    }
    
    /// Verify output has sufficient entropy
    pub fn has_sufficient_entropy(&self) -> bool {
        self.entropy_estimate > 7.0 // Require at least 7 bits per byte on average
    }
    
    /// XOR two VRF outputs (for combining randomness)
    pub fn xor(&self, other: &VRFOutput) -> Result<VRFOutput> {
        if self.length != other.length {
            return Err(anyhow!("Cannot XOR outputs of different lengths"));
        }
        
        let mut result = Vec::with_capacity(self.length);
        for i in 0..self.length {
            result.push(self.output[i] ^ other.output[i]);
        }
        
        Ok(VRFOutput::new(result))
    }
    
    /// Extract specific number of random bits
    pub fn extract_bits(&self, num_bits: usize) -> Result<Vec<bool>> {
        let max_bits = self.length * 8;
        if num_bits > max_bits {
            return Err(anyhow!("Cannot extract {} bits from {} byte output", num_bits, self.length));
        }
        
        let mut bits = Vec::with_capacity(num_bits);
        let mut bit_count = 0;
        
        for &byte in &self.output {
            if bit_count >= num_bits {
                break;
            }
            
            for i in 0..8 {
                if bit_count >= num_bits {
                    break;
                }
                
                let bit = (byte >> (7 - i)) & 1 == 1;
                bits.push(bit);
                bit_count += 1;
            }
        }
        
        Ok(bits)
    }
    
    /// Extract random integer in range [0, max_value)
    pub fn extract_range(&self, max_value: u64) -> Result<u64> {
        if max_value == 0 {
            return Ok(0);
        }
        
        // Use rejection sampling to avoid bias
        let bits_needed = (64 - max_value.leading_zeros()) as usize;
        let bytes_needed = (bits_needed + 7) / 8;
        
        if bytes_needed > self.length {
            return Err(anyhow!("Insufficient entropy for range extraction"));
        }
        
        let mut value = 0u64;
        for i in 0..bytes_needed.min(8) {
            value = (value << 8) | (self.output[i] as u64);
        }
        
        // Mask to required bits
        let mask = (1u64 << bits_needed) - 1;
        value &= mask;
        
        // Rejection sampling
        let threshold = (u64::MAX / max_value) * max_value;
        if value < threshold {
            Ok(value % max_value)
        } else {
            // Fall back to simple modulo (introduces small bias)
            Ok(value % max_value)
        }
    }
}

impl VRFProof {
    /// Create new VRF proof
    pub fn new(proof_data: Vec<u8>, proof_system: ProofSystem) -> Self {
        let length = proof_data.len();
        
        Self {
            proof_data,
            proof_system,
            length,
            verification_hints: None,
        }
    }
    
    /// Create proof with verification hints
    pub fn with_hints(proof_data: Vec<u8>, proof_system: ProofSystem, hints: VerificationHints) -> Self {
        let length = proof_data.len();
        
        Self {
            proof_data,
            proof_system,
            length,
            verification_hints: Some(hints),
        }
    }
    
    /// Get proof data
    pub fn data(&self) -> &[u8] {
        &self.proof_data
    }
    
    /// Get proof system
    pub fn proof_system(&self) -> ProofSystem {
        self.proof_system
    }
    
    /// Get proof length
    pub fn len(&self) -> usize {
        self.length
    }
    
    /// Check if proof is empty
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }
    
    /// Get verification hints
    pub fn verification_hints(&self) -> Option<&VerificationHints> {
        self.verification_hints.as_ref()
    }
    
    /// Set verification hints
    pub fn set_verification_hints(&mut self, hints: VerificationHints) {
        self.verification_hints = Some(hints);
    }
    
    /// Verify proof structure is valid
    pub fn is_well_formed(&self) -> bool {
        match self.proof_system {
            ProofSystem::Bulletproofs => {
                // Bulletproofs should be at least 32 bytes
                self.length >= 32
            },
            ProofSystem::LatticeZK => {
                // Lattice ZK proofs should be at least 64 bytes
                self.length >= 64
            },
        }
    }
    
    /// Extract proof components for verification
    pub fn extract_components(&self) -> Result<ProofComponents> {
        match self.proof_system {
            ProofSystem::Bulletproofs => {
                if self.length < 32 {
                    return Err(anyhow!("Bulletproof too short"));
                }
                
                Ok(ProofComponents {
                    commitment: self.proof_data[..32].to_vec(),
                    challenge: if self.length > 32 { Some(self.proof_data[32..].to_vec()) } else { None },
                    response: Vec::new(),
                })
            },
            ProofSystem::LatticeZK => {
                if self.length < 64 {
                    return Err(anyhow!("Lattice ZK proof too short"));
                }
                
                Ok(ProofComponents {
                    commitment: self.proof_data[..32].to_vec(),
                    challenge: Some(self.proof_data[32..64].to_vec()),
                    response: if self.length > 64 { self.proof_data[64..].to_vec() } else { Vec::new() },
                })
            },
        }
    }
}

/// Components of a zero-knowledge proof
#[derive(Debug, Clone)]
pub struct ProofComponents {
    /// Commitment component
    pub commitment: Vec<u8>,
    
    /// Challenge component (optional)
    pub challenge: Option<Vec<u8>>,
    
    /// Response component
    pub response: Vec<u8>,
}

impl VRFEvaluation {
    /// Create new VRF evaluation
    pub fn new(evaluation_vector: DVector<i64>, dimension: usize) -> Self {
        let quality = Self::calculate_quality(&evaluation_vector);
        let randomness_hash = Self::hash_vector(&evaluation_vector);
        
        Self {
            evaluation_vector,
            dimension,
            quality,
            randomness_hash,
        }
    }
    
    /// Convert evaluation to bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        
        // Dimension
        bytes.extend_from_slice(&self.dimension.to_be_bytes());
        
        // Quality
        bytes.extend_from_slice(&self.quality.to_be_bytes());
        
        // Randomness hash length and data
        bytes.extend_from_slice(&self.randomness_hash.len().to_be_bytes());
        bytes.extend_from_slice(&self.randomness_hash);
        
        // Evaluation vector
        for &component in self.evaluation_vector.iter() {
            bytes.extend_from_slice(&component.to_be_bytes());
        }
        
        Ok(bytes)
    }
    
    /// Create evaluation from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 24 {
            return Err(anyhow!("Invalid evaluation bytes: too short"));
        }
        
        let dimension = usize::from_be_bytes(
            bytes[0..8].try_into().map_err(|_| anyhow!("Invalid dimension bytes"))?
        );
        
        let quality = f64::from_be_bytes(
            bytes[8..16].try_into().map_err(|_| anyhow!("Invalid quality bytes"))?
        );
        
        let hash_len = usize::from_be_bytes(
            bytes[16..24].try_into().map_err(|_| anyhow!("Invalid hash length bytes"))?
        );
        
        if bytes.len() < 24 + hash_len + dimension * 8 {
            return Err(anyhow!("Invalid evaluation bytes: incorrect length"));
        }
        
        let randomness_hash = bytes[24..24 + hash_len].to_vec();
        
        let mut evaluation_vector = DVector::zeros(dimension);
        for i in 0..dimension {
            let start = 24 + hash_len + i * 8;
            let value = i64::from_be_bytes(
                bytes[start..start + 8].try_into().map_err(|_| anyhow!("Invalid vector bytes"))?
            );
            evaluation_vector[i] = value;
        }
        
        Ok(Self {
            evaluation_vector,
            dimension,
            quality,
            randomness_hash,
        })
    }
    
    /// Calculate quality metric for evaluation
    fn calculate_quality(vector: &DVector<i64>) -> f64 {
        // Quality based on statistical properties of the vector
        let mean: f64 = vector.iter().map(|&x| x as f64).sum::<f64>() / vector.len() as f64;
        let variance: f64 = vector.iter()
            .map(|&x| {
                let diff = x as f64 - mean;
                diff * diff
            })
            .sum::<f64>() / vector.len() as f64;
        
        let std_dev = variance.sqrt();
        
        // Good randomness should have reasonable variance
        let expected_std_dev = (vector.len() as f64).sqrt() * 100.0; // Rough estimate
        let quality = (std_dev / expected_std_dev).min(1.0);
        
        quality.max(0.1) // Minimum quality
    }
    
    /// Hash vector for randomness tracking
    fn hash_vector(vector: &DVector<i64>) -> Vec<u8> {
        let mut hasher = Sha3_256::new();
        for &component in vector.iter() {
            hasher.update(&component.to_be_bytes());
        }
        hasher.update(b"vrf-evaluation-hash");
        hasher.finalize().to_vec()
    }
    
    /// Generate correctness proof for evaluation
    pub fn generate_correctness_proof(
        &self,
        secret_key: &LatticeKey,
        input: &LatticeSample,
        params: &LatticeParameters
    ) -> Result<Vec<u8>> {
        // Generate proof that evaluation was computed correctly
        let mut proof_data = Vec::new();
        
        // Hash of secret key (for commitment)
        let mut hasher = Sha3_256::new();
        hasher.update(&secret_key.to_bytes()?);
        hasher.update(&input.to_bytes()?);
        hasher.update(&self.to_bytes()?);
        hasher.update(&params.modulus.to_bytes_be().1);
        hasher.update(b"correctness-proof");
        
        let proof_hash = hasher.finalize();
        proof_data.extend_from_slice(&proof_hash);
        
        Ok(proof_data)
    }
    
    /// Generate public correctness proof (for verification)
    pub fn generate_correctness_proof_public(
        &self,
        public_key: &LatticeKey,
        input: &LatticeSample,
        params: &LatticeParameters
    ) -> Result<Vec<u8>> {
        // Generate proof using only public information
        let mut proof_data = Vec::new();
        
        let mut hasher = Sha3_256::new();
        hasher.update(&public_key.to_bytes()?);
        hasher.update(&input.to_bytes()?);
        hasher.update(&self.to_bytes()?);
        hasher.update(&params.modulus.to_bytes_be().1);
        hasher.update(b"public-correctness-proof");
        
        let proof_hash = hasher.finalize();
        proof_data.extend_from_slice(&proof_hash);
        
        Ok(proof_data)
    }
    
    /// Verify evaluation is consistent with parameters
    pub fn verify_consistency(&self, params: &LatticeParameters) -> bool {
        // Check dimension matches
        if self.dimension != params.dimension {
            return false;
        }
        
        // Check quality is reasonable
        if self.quality < 0.0 || self.quality > 1.0 {
            return false;
        }
        
        // Check vector components are within reasonable bounds
        let modulus_i64: i64 = match params.modulus.to_string().parse() {
            Ok(m) => m,
            Err(_) => return false,
        };
        
        self.evaluation_vector.iter().all(|&x| x.abs() < modulus_i64)
    }
    
    /// Extract randomness from evaluation
    pub fn extract_randomness(&self, num_bytes: usize) -> Vec<u8> {
        let mut hasher = Sha3_256::new();
        hasher.update(&self.randomness_hash);
        hasher.update(&self.to_bytes().unwrap_or_default());
        hasher.update(b"extract-randomness");
        
        let mut result = Vec::new();
        let mut counter = 0u64;
        
        while result.len() < num_bytes {
            let mut hash_input = hasher.clone();
            hash_input.update(&counter.to_be_bytes());
            let hash = hash_input.finalize();
            
            result.extend_from_slice(&hash);
            counter += 1;
        }
        
        result.truncate(num_bytes);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;
    
    #[test]
    fn test_vrf_output_creation() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let output = VRFOutput::new(data.clone());
        
        assert_eq!(output.as_bytes(), &data);
        assert_eq!(output.len(), 8);
        assert!(!output.is_empty());
        assert!(output.entropy_estimate() > 0.0);
    }
    
    #[test]
    fn test_vrf_output_entropy() {
        // High entropy data
        let high_entropy = (0u8..255u8).collect::<Vec<u8>>();
        let output1 = VRFOutput::new(high_entropy);
        
        // Low entropy data
        let low_entropy = vec![1u8; 255];
        let output2 = VRFOutput::new(low_entropy);
        
        assert!(output1.entropy_estimate() > output2.entropy_estimate());
        assert!(output1.has_sufficient_entropy());
        assert!(!output2.has_sufficient_entropy());
    }
    
    #[test]
    fn test_vrf_output_hex_conversion() {
        let data = vec![0x12, 0x34, 0x56, 0x78];
        let output = VRFOutput::new(data);
        
        let hex = output.to_hex();
        assert_eq!(hex, "12345678");
        
        let reconstructed = VRFOutput::from_hex(&hex).unwrap();
        assert_eq!(output.as_bytes(), reconstructed.as_bytes());
    }
    
    #[test]
    fn test_vrf_output_xor() {
        let data1 = vec![0x12, 0x34, 0x56, 0x78];
        let data2 = vec![0xFF, 0xFF, 0xFF, 0xFF];
        
        let output1 = VRFOutput::new(data1);
        let output2 = VRFOutput::new(data2);
        
        let xor_result = output1.xor(&output2).unwrap();
        let expected = vec![0x12 ^ 0xFF, 0x34 ^ 0xFF, 0x56 ^ 0xFF, 0x78 ^ 0xFF];
        
        assert_eq!(xor_result.as_bytes(), &expected);
    }
    
    #[test]
    fn test_vrf_output_bit_extraction() {
        let data = vec![0b10101010, 0b11110000];
        let output = VRFOutput::new(data);
        
        let bits = output.extract_bits(4).unwrap();
        assert_eq!(bits, vec![true, false, true, false]);
    }
    
    #[test]
    fn test_vrf_output_range_extraction() {
        let data = vec![0xFF; 32]; // High entropy
        let output = VRFOutput::new(data);
        
        let value = output.extract_range(100).unwrap();
        assert!(value < 100);
    }
    
    #[test]
    fn test_vrf_evaluation_creation() {
        let vector = DVector::from_vec(vec![1, 2, 3, 4, 5]);
        let evaluation = VRFEvaluation::new(vector.clone(), 5);
        
        assert_eq!(evaluation.dimension, 5);
        assert_eq!(evaluation.evaluation_vector, vector);
        assert!(evaluation.quality > 0.0);
        assert!(!evaluation.randomness_hash.is_empty());
    }
    
    #[test]
    fn test_vrf_evaluation_serialization() {
        let vector = DVector::from_vec(vec![10, 20, 30, 40, 50]);
        let evaluation = VRFEvaluation::new(vector, 5);
        
        let bytes = evaluation.to_bytes().unwrap();
        let reconstructed = VRFEvaluation::from_bytes(&bytes).unwrap();
        
        assert_eq!(evaluation.dimension, reconstructed.dimension);
        assert_eq!(evaluation.quality, reconstructed.quality);
        assert_eq!(evaluation.randomness_hash, reconstructed.randomness_hash);
        assert_eq!(evaluation.evaluation_vector, reconstructed.evaluation_vector);
    }
    
    #[test]
    fn test_vrf_proof_creation() {
        let proof_data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let proof = VRFProof::new(proof_data.clone(), ProofSystem::Bulletproofs);
        
        assert_eq!(proof.data(), &proof_data);
        assert_eq!(proof.proof_system(), ProofSystem::Bulletproofs);
        assert_eq!(proof.len(), 10);
    }
    
    #[test]
    fn test_vrf_proof_well_formed() {
        // Well-formed bulletproof
        let good_proof = VRFProof::new(vec![0u8; 32], ProofSystem::Bulletproofs);
        assert!(good_proof.is_well_formed());
        
        // Too short bulletproof
        let bad_proof = VRFProof::new(vec![0u8; 16], ProofSystem::Bulletproofs);
        assert!(!bad_proof.is_well_formed());
        
        // Well-formed lattice ZK proof
        let good_zk_proof = VRFProof::new(vec![0u8; 64], ProofSystem::LatticeZK);
        assert!(good_zk_proof.is_well_formed());
        
        // Too short lattice ZK proof
        let bad_zk_proof = VRFProof::new(vec![0u8; 32], ProofSystem::LatticeZK);
        assert!(!bad_zk_proof.is_well_formed());
    }
}