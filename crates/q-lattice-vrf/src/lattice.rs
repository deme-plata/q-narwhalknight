/// Lattice cryptographic operations for L-VRF
/// Implements lattice-based mathematical operations for quantum-resistant VRF

use anyhow::{Result, anyhow};
use serde::{Serialize, Deserialize};
use sha3::{Digest, Sha3_256};
use nalgebra::{DMatrix, DVector};
use num_bigint::BigInt;
use num_traits::{Zero, One};
use rand::RngCore;
use std::convert::TryInto;

use crate::parameters::{SecurityLevel, LatticeConfig};
use crate::vrf_core::VRFEvaluation;

/// Lattice parameters for VRF operations
#[derive(Debug, Clone)]
pub struct LatticeParameters {
    /// Lattice dimension
    pub dimension: usize,
    
    /// Modulus for operations
    pub modulus: BigInt,
    
    /// Gaussian parameter for sampling
    pub gaussian_parameter: f64,
    
    /// Security level
    pub security_level: SecurityLevel,
    
    /// Basis matrix A
    pub basis_matrix: DMatrix<i64>,
    
    /// Additional parameters for optimization
    pub config: LatticeConfig,
}

/// Lattice key (secret or public)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatticeKey {
    /// Key material as lattice vector
    key_vector: DVector<i64>,
    
    /// Key type identifier
    key_type: KeyType,
    
    /// Associated lattice dimension
    dimension: usize,
}

/// Key type enumeration
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum KeyType {
    Secret,
    Public,
}

/// Lattice sample representation
#[derive(Debug, Clone)]
pub struct LatticeSample {
    /// Sample vector in lattice space
    pub vector: DVector<i64>,
    
    /// Sample dimension
    pub dimension: usize,
    
    /// Quality metric for the sample
    pub quality: f64,
}

impl LatticeParameters {
    /// Create new lattice parameters for given security level
    pub fn new(security_level: SecurityLevel) -> Result<Self> {
        let config = LatticeConfig::for_security_level(security_level);
        
        let dimension = config.dimension;
        let modulus = config.modulus.clone();
        let gaussian_parameter = config.gaussian_parameter;
        
        // Generate random basis matrix
        let basis_matrix = Self::generate_basis_matrix(dimension, &modulus)?;
        
        Ok(Self {
            dimension,
            modulus,
            gaussian_parameter,
            security_level,
            basis_matrix,
            config,
        })
    }
    
    /// Generate random basis matrix for lattice
    fn generate_basis_matrix(dimension: usize, modulus: &BigInt) -> Result<DMatrix<i64>> {
        let mut matrix = DMatrix::zeros(dimension, dimension);
        let modulus_i64: i64 = modulus.to_string().parse()
            .map_err(|_| anyhow!("Modulus too large for i64"))?;
        
        // Fill with random values mod modulus
        let mut rng = rand::rngs::OsRng;
        for i in 0..dimension {
            for j in 0..dimension {
                let mut bytes = [0u8; 8];
                rng.fill_bytes(&mut bytes);
                let value = i64::from_be_bytes(bytes);
                matrix[(i, j)] = value.rem_euclid(modulus_i64);
            }
        }
        
        // Ensure matrix is invertible by making diagonal dominant
        for i in 0..dimension {
            matrix[(i, i)] = matrix[(i, i)].abs() + modulus_i64 / 4;
        }
        
        Ok(matrix)
    }
    
    /// Sample from discrete Gaussian distribution
    pub fn sample_gaussian(&self) -> Result<DVector<i64>> {
        let mut sample = DVector::zeros(self.dimension);
        let mut rng = rand::rngs::OsRng;
        
        for i in 0..self.dimension {
            // Box-Muller transform for Gaussian sampling
            let u1: f64 = (rng.next_u64() as f64) / (u64::MAX as f64);
            let u2: f64 = (rng.next_u64() as f64) / (u64::MAX as f64);
            
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let gaussian_sample = (z * self.gaussian_parameter).round() as i64;
            
            sample[i] = gaussian_sample;
        }
        
        Ok(sample)
    }
    
    /// Sample uniformly from lattice cosets
    pub fn sample_uniform(&self) -> Result<DVector<i64>> {
        let mut sample = DVector::zeros(self.dimension);
        let mut rng = rand::rngs::OsRng;
        let modulus_i64: i64 = self.modulus.to_string().parse()
            .map_err(|_| anyhow!("Modulus too large for i64"))?;
        
        for i in 0..self.dimension {
            let mut bytes = [0u8; 8];
            rng.fill_bytes(&mut bytes);
            let value = i64::from_be_bytes(bytes);
            sample[i] = value.rem_euclid(modulus_i64);
        }
        
        Ok(sample)
    }
}

impl LatticeKey {
    /// Generate new secret key
    pub async fn generate_secret(params: &LatticeParameters) -> Result<Self> {
        // Sample secret key from discrete Gaussian
        let key_vector = params.sample_gaussian()?;
        
        Ok(Self {
            key_vector,
            key_type: KeyType::Secret,
            dimension: params.dimension,
        })
    }
    
    /// Compute public key from secret key
    pub fn compute_public_key(&self, params: &LatticeParameters) -> Result<Self> {
        if self.key_type != KeyType::Secret {
            return Err(anyhow!("Can only compute public key from secret key"));
        }
        
        // Public key = A * secret_key (mod q)
        let public_vector = &params.basis_matrix * &self.key_vector;
        let modulus_i64: i64 = params.modulus.to_string().parse()
            .map_err(|_| anyhow!("Modulus too large for i64"))?;
        
        // Reduce mod q
        let mut reduced_vector = DVector::zeros(params.dimension);
        for i in 0..params.dimension {
            reduced_vector[i] = public_vector[i].rem_euclid(modulus_i64);
        }
        
        Ok(Self {
            key_vector: reduced_vector,
            key_type: KeyType::Public,
            dimension: params.dimension,
        })
    }
    
    /// Evaluate VRF using secret key
    pub fn evaluate_vrf(
        &self,
        input: &LatticeSample,
        randomness: &[u8],
        params: &LatticeParameters
    ) -> Result<VRFEvaluation> {
        if self.key_type != KeyType::Secret {
            return Err(anyhow!("VRF evaluation requires secret key"));
        }
        
        // Hash randomness with input for deterministic randomness
        let mut hasher = Sha3_256::new();
        hasher.update(randomness);
        hasher.update(&input.to_bytes()?);
        let hashed_randomness = hasher.finalize();
        
        // Convert hash to lattice vector
        let mut random_vector = DVector::zeros(params.dimension);
        for i in 0..params.dimension {
            let start = (i * 4) % 32;
            let bytes: [u8; 4] = hashed_randomness[start..start + 4].try_into()
                .map_err(|_| anyhow!("Failed to extract randomness bytes"))?;
            let value = i32::from_be_bytes(bytes) as i64;
            random_vector[i] = value;
        }
        
        // VRF evaluation: combine secret key, input, and randomness
        let evaluation_vector = &self.key_vector + &input.vector + &random_vector;
        
        // Apply lattice trapdoor operation
        let trapdoor_result = self.apply_trapdoor_function(&evaluation_vector, params)?;
        
        VRFEvaluation::new(trapdoor_result, input.dimension)
    }
    
    /// Public evaluation for verification
    pub fn public_evaluate(&self, input: &LatticeSample, params: &LatticeParameters) -> Result<VRFEvaluation> {
        if self.key_type != KeyType::Public {
            return Err(anyhow!("Public evaluation requires public key"));
        }
        
        // Public evaluation using public key and basis matrix
        let public_eval = &params.basis_matrix * &self.key_vector + &input.vector;
        let modulus_i64: i64 = params.modulus.to_string().parse()
            .map_err(|_| anyhow!("Modulus too large for i64"))?;
        
        // Reduce mod q
        let mut reduced_eval = DVector::zeros(params.dimension);
        for i in 0..params.dimension {
            reduced_eval[i] = public_eval[i].rem_euclid(modulus_i64);
        }
        
        VRFEvaluation::new(reduced_eval, input.dimension)
    }
    
    /// Apply lattice trapdoor function
    fn apply_trapdoor_function(&self, input: &DVector<i64>, params: &LatticeParameters) -> Result<DVector<i64>> {
        // Simplified trapdoor operation
        // Real implementation would use advanced lattice trapdoors (e.g., GPV, MP12)
        
        let mut result = DVector::zeros(params.dimension);
        let modulus_i64: i64 = params.modulus.to_string().parse()
            .map_err(|_| anyhow!("Modulus too large for i64"))?;
        
        for i in 0..params.dimension {
            // Apply modular arithmetic with secret key
            result[i] = (input[i] * self.key_vector[i]).rem_euclid(modulus_i64);
        }
        
        Ok(result)
    }
    
    /// Compute commitment to key
    pub fn compute_commitment(&self, params: &LatticeParameters) -> Result<Vec<u8>> {
        let mut hasher = Sha3_256::new();
        hasher.update(&self.to_bytes()?);
        hasher.update(&params.modulus.to_bytes_be().1);
        hasher.update(b"lattice-key-commitment");
        
        Ok(hasher.finalize().to_vec())
    }
    
    /// Convert key to bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        
        // Key type
        bytes.push(match self.key_type {
            KeyType::Secret => 0,
            KeyType::Public => 1,
        });
        
        // Dimension
        bytes.extend_from_slice(&self.dimension.to_be_bytes());
        
        // Key vector
        for i in 0..self.dimension {
            bytes.extend_from_slice(&self.key_vector[i].to_be_bytes());
        }
        
        Ok(bytes)
    }
    
    /// Create key from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 9 {
            return Err(anyhow!("Invalid key bytes: too short"));
        }
        
        let key_type = match bytes[0] {
            0 => KeyType::Secret,
            1 => KeyType::Public,
            _ => return Err(anyhow!("Invalid key type")),
        };
        
        let dimension = usize::from_be_bytes(
            bytes[1..9].try_into().map_err(|_| anyhow!("Invalid dimension bytes"))?
        );
        
        if bytes.len() != 9 + dimension * 8 {
            return Err(anyhow!("Invalid key bytes: incorrect length"));
        }
        
        let mut key_vector = DVector::zeros(dimension);
        for i in 0..dimension {
            let start = 9 + i * 8;
            let value = i64::from_be_bytes(
                bytes[start..start + 8].try_into().map_err(|_| anyhow!("Invalid vector bytes"))?
            );
            key_vector[i] = value;
        }
        
        Ok(Self {
            key_vector,
            key_type,
            dimension,
        })
    }
}

impl LatticeSample {
    /// Create lattice sample from hash
    pub fn from_hash(hash: &[u8], params: &LatticeParameters) -> Result<Self> {
        let mut vector = DVector::zeros(params.dimension);
        let modulus_i64: i64 = params.modulus.to_string().parse()
            .map_err(|_| anyhow!("Modulus too large for i64"))?;
        
        // Use hash to generate lattice vector
        for i in 0..params.dimension {
            let start = (i * 4) % hash.len();
            let end = ((start + 4).min(hash.len())).max(start + 1);
            
            let mut bytes = [0u8; 4];
            let slice_len = end - start;
            bytes[..slice_len].copy_from_slice(&hash[start..end]);
            
            let value = i32::from_be_bytes(bytes) as i64;
            vector[i] = value.rem_euclid(modulus_i64);
        }
        
        // Calculate quality metric based on vector properties
        let quality = Self::calculate_quality(&vector);
        
        Ok(Self {
            vector,
            dimension: params.dimension,
            quality,
        })
    }
    
    /// Calculate quality metric for lattice sample
    fn calculate_quality(vector: &DVector<i64>) -> f64 {
        let norm_sq: f64 = vector.iter().map(|&x| (x as f64).powi(2)).sum();
        let norm = norm_sq.sqrt();
        
        // Quality based on inverse of norm (smaller norms are better)
        let dimension = vector.len() as f64;
        let expected_norm = (dimension / 2.0).sqrt();
        
        (expected_norm / (norm + 1.0)).min(1.0)
    }
    
    /// Convert sample to bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        
        // Dimension
        bytes.extend_from_slice(&self.dimension.to_be_bytes());
        
        // Quality
        bytes.extend_from_slice(&self.quality.to_be_bytes());
        
        // Vector components
        for i in 0..self.dimension {
            bytes.extend_from_slice(&self.vector[i].to_be_bytes());
        }
        
        Ok(bytes)
    }
    
    /// Create sample from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 16 {
            return Err(anyhow!("Invalid sample bytes: too short"));
        }
        
        let dimension = usize::from_be_bytes(
            bytes[0..8].try_into().map_err(|_| anyhow!("Invalid dimension bytes"))?
        );
        
        let quality = f64::from_be_bytes(
            bytes[8..16].try_into().map_err(|_| anyhow!("Invalid quality bytes"))?
        );
        
        if bytes.len() != 16 + dimension * 8 {
            return Err(anyhow!("Invalid sample bytes: incorrect length"));
        }
        
        let mut vector = DVector::zeros(dimension);
        for i in 0..dimension {
            let start = 16 + i * 8;
            let value = i64::from_be_bytes(
                bytes[start..start + 8].try_into().map_err(|_| anyhow!("Invalid vector bytes"))?
            );
            vector[i] = value;
        }
        
        Ok(Self {
            vector,
            dimension,
            quality,
        })
    }
    
    /// Check if sample is within lattice bounds
    pub fn is_valid(&self, params: &LatticeParameters) -> bool {
        if self.dimension != params.dimension {
            return false;
        }
        
        let modulus_i64: i64 = match params.modulus.to_string().parse() {
            Ok(m) => m,
            Err(_) => return false,
        };
        
        // Check all components are within modulus
        self.vector.iter().all(|&x| x >= 0 && x < modulus_i64)
    }
    
    /// Compute inner product with another sample
    pub fn inner_product(&self, other: &LatticeSample) -> Result<i64> {
        if self.dimension != other.dimension {
            return Err(anyhow!("Dimension mismatch for inner product"));
        }
        
        let product = self.vector.dot(&other.vector);
        Ok(product)
    }
    
    /// Add two lattice samples
    pub fn add(&self, other: &LatticeSample, modulus: &BigInt) -> Result<LatticeSample> {
        if self.dimension != other.dimension {
            return Err(anyhow!("Dimension mismatch for addition"));
        }
        
        let modulus_i64: i64 = modulus.to_string().parse()
            .map_err(|_| anyhow!("Modulus too large for i64"))?;
        
        let mut result_vector = DVector::zeros(self.dimension);
        for i in 0..self.dimension {
            result_vector[i] = (self.vector[i] + other.vector[i]).rem_euclid(modulus_i64);
        }
        
        let quality = (self.quality + other.quality) / 2.0;
        
        Ok(LatticeSample {
            vector: result_vector,
            dimension: self.dimension,
            quality,
        })
    }
    
    /// Scale sample by scalar
    pub fn scale(&self, scalar: i64, modulus: &BigInt) -> Result<LatticeSample> {
        let modulus_i64: i64 = modulus.to_string().parse()
            .map_err(|_| anyhow!("Modulus too large for i64"))?;
        
        let mut result_vector = DVector::zeros(self.dimension);
        for i in 0..self.dimension {
            result_vector[i] = (self.vector[i] * scalar).rem_euclid(modulus_i64);
        }
        
        Ok(LatticeSample {
            vector: result_vector,
            dimension: self.dimension,
            quality: self.quality, // Quality preserved under scaling
        })
    }
}

/// Lattice operations utilities
pub struct LatticeOperations;

impl LatticeOperations {
    /// Compute lattice basis reduction (simplified LLL)
    pub fn reduce_basis(basis: &DMatrix<i64>) -> Result<DMatrix<i64>> {
        // Simplified LLL reduction - real implementation would use full LLL
        let mut reduced = basis.clone();
        let dimension = basis.nrows();
        
        // Gram-Schmidt orthogonalization
        for i in 1..dimension {
            for j in 0..i {
                let numerator = reduced.row(i).dot(&reduced.row(j));
                let denominator = reduced.row(j).dot(&reduced.row(j));
                
                if denominator != 0 {
                    let mu = numerator / denominator;
                    for k in 0..basis.ncols() {
                        reduced[(i, k)] -= mu * reduced[(j, k)];
                    }
                }
            }
        }
        
        Ok(reduced)
    }
    
    /// Compute shortest vector approximation
    pub fn shortest_vector_approx(lattice: &DMatrix<i64>) -> Result<DVector<i64>> {
        // Return first basis vector as approximation
        // Real implementation would use BKZ or other SVP algorithms
        Ok(lattice.row(0).transpose())
    }
    
    /// Check if point is in lattice
    pub fn is_in_lattice(point: &DVector<i64>, basis: &DMatrix<i64>, modulus: i64) -> Result<bool> {
        // Solve Ax = point (mod modulus) for integer solution x
        // Simplified check - real implementation would use proper lattice algorithms
        
        let dimension = basis.nrows();
        if point.len() != dimension {
            return Ok(false);
        }
        
        // Check if point is close to lattice using basis vectors
        for i in 0..dimension {
            let projection = basis.row(i).dot(point);
            if projection.rem_euclid(modulus) != 0 {
                return Ok(false);
            }
        }
        
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_lattice_parameters_creation() {
        let params = LatticeParameters::new(SecurityLevel::Standard).unwrap();
        assert!(params.dimension > 0);
        assert!(params.modulus > BigInt::zero());
        assert!(params.gaussian_parameter > 0.0);
    }
    
    #[test]
    fn test_key_generation_and_derivation() {
        let params = LatticeParameters::new(SecurityLevel::Standard).unwrap();
        
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let secret_key = LatticeKey::generate_secret(&params).await.unwrap();
            let public_key = secret_key.compute_public_key(&params).unwrap();
            
            assert_eq!(secret_key.dimension, params.dimension);
            assert_eq!(public_key.dimension, params.dimension);
            assert_ne!(secret_key.to_bytes().unwrap(), public_key.to_bytes().unwrap());
        });
    }
    
    #[test]
    fn test_lattice_sample_creation() {
        let params = LatticeParameters::new(SecurityLevel::Standard).unwrap();
        let hash = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                   17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32];
        
        let sample = LatticeSample::from_hash(&hash, &params).unwrap();
        assert_eq!(sample.dimension, params.dimension);
        assert!(sample.quality >= 0.0 && sample.quality <= 1.0);
        assert!(sample.is_valid(&params));
    }
    
    #[test]
    fn test_lattice_sample_operations() {
        let params = LatticeParameters::new(SecurityLevel::Standard).unwrap();
        let hash1 = [1; 32];
        let hash2 = [2; 32];
        
        let sample1 = LatticeSample::from_hash(&hash1, &params).unwrap();
        let sample2 = LatticeSample::from_hash(&hash2, &params).unwrap();
        
        // Test addition
        let sum = sample1.add(&sample2, &params.modulus).unwrap();
        assert_eq!(sum.dimension, params.dimension);
        
        // Test scaling
        let scaled = sample1.scale(3, &params.modulus).unwrap();
        assert_eq!(scaled.dimension, params.dimension);
        
        // Test inner product
        let product = sample1.inner_product(&sample2).unwrap();
        assert!(product != 0); // Very likely for random samples
    }
    
    #[test]
    fn test_key_serialization() {
        let params = LatticeParameters::new(SecurityLevel::Standard).unwrap();
        
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let original_key = LatticeKey::generate_secret(&params).await.unwrap();
            let bytes = original_key.to_bytes().unwrap();
            let reconstructed_key = LatticeKey::from_bytes(&bytes).unwrap();
            
            assert_eq!(original_key.to_bytes().unwrap(), reconstructed_key.to_bytes().unwrap());
        });
    }
    
    #[test]
    fn test_sample_serialization() {
        let params = LatticeParameters::new(SecurityLevel::Standard).unwrap();
        let hash = [42; 32];
        
        let original_sample = LatticeSample::from_hash(&hash, &params).unwrap();
        let bytes = original_sample.to_bytes().unwrap();
        let reconstructed_sample = LatticeSample::from_bytes(&bytes).unwrap();
        
        assert_eq!(original_sample.dimension, reconstructed_sample.dimension);
        assert_eq!(original_sample.quality, reconstructed_sample.quality);
        
        for i in 0..original_sample.dimension {
            assert_eq!(original_sample.vector[i], reconstructed_sample.vector[i]);
        }
    }
}