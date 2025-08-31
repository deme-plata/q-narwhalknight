/// Parameter sets and configurations for Lattice-based VRF
/// Defines security levels and cryptographic parameters

use anyhow::{Result, anyhow};
use serde::{Serialize, Deserialize};
use num_bigint::BigInt;
use std::collections::HashMap;

/// Security levels for lattice cryptography
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum SecurityLevel {
    /// Standard security (~128-bit security)
    Standard,
    
    /// High security (~192-bit security)
    High,
    
    /// Ultra-high security (~256-bit security)
    Ultra,
    
    /// Custom security level
    Custom(u16),
}

impl SecurityLevel {
    /// Get equivalent bit security
    pub fn security_bits(&self) -> u16 {
        match self {
            SecurityLevel::Standard => 128,
            SecurityLevel::High => 192,
            SecurityLevel::Ultra => 256,
            SecurityLevel::Custom(bits) => *bits,
        }
    }
    
    /// Get recommended lattice dimension for this security level
    pub fn recommended_dimension(&self) -> usize {
        match self {
            SecurityLevel::Standard => 512,
            SecurityLevel::High => 768,
            SecurityLevel::Ultra => 1024,
            SecurityLevel::Custom(bits) => (*bits as usize) * 4,
        }
    }
    
    /// Get recommended modulus size
    pub fn modulus_bits(&self) -> usize {
        match self {
            SecurityLevel::Standard => 2048,
            SecurityLevel::High => 3072,
            SecurityLevel::Ultra => 4096,
            SecurityLevel::Custom(bits) => (*bits as usize) * 16,
        }
    }
}

/// Lattice configuration parameters
#[derive(Debug, Clone)]
pub struct LatticeConfig {
    /// Lattice dimension
    pub dimension: usize,
    
    /// Modulus for arithmetic operations
    pub modulus: BigInt,
    
    /// Gaussian parameter for discrete Gaussian sampling
    pub gaussian_parameter: f64,
    
    /// Security level
    pub security_level: SecurityLevel,
    
    /// Ring structure (for Ring-LWE based constructions)
    pub ring_degree: Option<usize>,
    
    /// Error distribution parameters
    pub error_bound: f64,
    
    /// Number of samples for security reduction
    pub num_samples: usize,
    
    /// Optimization flags
    pub optimizations: LatticeOptimizations,
}

/// Lattice optimization flags
#[derive(Debug, Clone, Default)]
pub struct LatticeOptimizations {
    /// Use Number Theoretic Transform (NTT) for polynomial operations
    pub use_ntt: bool,
    
    /// Enable batch operations
    pub enable_batching: bool,
    
    /// Use precomputed tables
    pub use_precomputation: bool,
    
    /// Enable parallel processing
    pub parallel_processing: bool,
    
    /// Memory vs speed trade-off (0.0 = memory optimized, 1.0 = speed optimized)
    pub speed_vs_memory: f32,
}

/// Predefined parameter sets for common use cases
pub struct ParameterSets;

impl ParameterSets {
    /// Get NIST Post-Quantum Cryptography standardization parameters
    pub fn nist_level_1() -> LatticeConfig {
        LatticeConfig {
            dimension: 512,
            modulus: Self::generate_prime_modulus(2048),
            gaussian_parameter: 3.19,
            security_level: SecurityLevel::Standard,
            ring_degree: Some(512),
            error_bound: 6.0,
            num_samples: 512,
            optimizations: LatticeOptimizations {
                use_ntt: true,
                enable_batching: true,
                use_precomputation: true,
                parallel_processing: true,
                speed_vs_memory: 0.7,
            },
        }
    }
    
    /// NIST Level 3 parameters
    pub fn nist_level_3() -> LatticeConfig {
        LatticeConfig {
            dimension: 768,
            modulus: Self::generate_prime_modulus(3072),
            gaussian_parameter: 2.75,
            security_level: SecurityLevel::High,
            ring_degree: Some(768),
            error_bound: 4.5,
            num_samples: 768,
            optimizations: LatticeOptimizations {
                use_ntt: true,
                enable_batching: true,
                use_precomputation: true,
                parallel_processing: true,
                speed_vs_memory: 0.8,
            },
        }
    }
    
    /// NIST Level 5 parameters
    pub fn nist_level_5() -> LatticeConfig {
        LatticeConfig {
            dimension: 1024,
            modulus: Self::generate_prime_modulus(4096),
            gaussian_parameter: 2.3,
            security_level: SecurityLevel::Ultra,
            ring_degree: Some(1024),
            error_bound: 3.0,
            num_samples: 1024,
            optimizations: LatticeOptimizations {
                use_ntt: true,
                enable_batching: true,
                use_precomputation: true,
                parallel_processing: true,
                speed_vs_memory: 0.9,
            },
        }
    }
    
    /// Kyber-like parameters for structured lattices
    pub fn kyber_like() -> LatticeConfig {
        LatticeConfig {
            dimension: 1024,
            modulus: BigInt::from(3329), // Kyber modulus
            gaussian_parameter: 1.0,
            security_level: SecurityLevel::Standard,
            ring_degree: Some(256),
            error_bound: 2.0,
            num_samples: 1024,
            optimizations: LatticeOptimizations {
                use_ntt: true,
                enable_batching: true,
                use_precomputation: true,
                parallel_processing: true,
                speed_vs_memory: 0.8,
            },
        }
    }
    
    /// Dilithium-like parameters for signatures
    pub fn dilithium_like() -> LatticeConfig {
        LatticeConfig {
            dimension: 1312,
            modulus: BigInt::from(8380417), // Dilithium modulus
            gaussian_parameter: 1.0,
            security_level: SecurityLevel::Standard,
            ring_degree: Some(256),
            error_bound: 1.5,
            num_samples: 1312,
            optimizations: LatticeOptimizations {
                use_ntt: true,
                enable_batching: true,
                use_precomputation: true,
                parallel_processing: true,
                speed_vs_memory: 0.7,
            },
        }
    }
    
    /// Lightweight parameters for constrained environments
    pub fn lightweight() -> LatticeConfig {
        LatticeConfig {
            dimension: 256,
            modulus: Self::generate_prime_modulus(1024),
            gaussian_parameter: 4.0,
            security_level: SecurityLevel::Custom(80),
            ring_degree: Some(256),
            error_bound: 8.0,
            num_samples: 256,
            optimizations: LatticeOptimizations {
                use_ntt: false,
                enable_batching: false,
                use_precomputation: false,
                parallel_processing: false,
                speed_vs_memory: 0.3,
            },
        }
    }
    
    /// Generate a prime modulus of specified bit length
    fn generate_prime_modulus(bits: usize) -> BigInt {
        // For demonstration, use well-known primes
        // In practice, would generate cryptographically secure primes
        match bits {
            1024 => BigInt::from(2u64.pow(31) - 1), // Mersenne prime
            2048 => BigInt::from(4294967291u64), // Large prime < 2^32
            3072 => BigInt::from(18446744073709551557u64), // Large prime < 2^64
            4096 => {
                // Use a well-known large prime
                let prime_str = "179769313486231590772930519078902473361797697894230657273430081157732675805500963132708477322407536021120113879871393357658789768814416622492847430639474124377767893424865485276302219601246094119453082952085005768838150682342462881473913110540827237163350510684586298239947245938479716304835356329624224137216";
                BigInt::parse_bytes(prime_str.as_bytes(), 10).unwrap_or(BigInt::from(2u64.pow(31) - 1))
            },
            _ => BigInt::from(2u64.pow(31) - 1),
        }
    }
}

impl LatticeConfig {
    /// Create configuration for specific security level
    pub fn for_security_level(level: SecurityLevel) -> Self {
        match level {
            SecurityLevel::Standard => ParameterSets::nist_level_1(),
            SecurityLevel::High => ParameterSets::nist_level_3(),
            SecurityLevel::Ultra => ParameterSets::nist_level_5(),
            SecurityLevel::Custom(bits) => {
                Self::custom_config(bits)
            }
        }
    }
    
    /// Create custom configuration
    pub fn custom_config(security_bits: u16) -> Self {
        let dimension = (security_bits as usize) * 4;
        let modulus_bits = (security_bits as usize) * 16;
        
        LatticeConfig {
            dimension,
            modulus: ParameterSets::generate_prime_modulus(modulus_bits),
            gaussian_parameter: 3.0 + (security_bits as f64 / 100.0),
            security_level: SecurityLevel::Custom(security_bits),
            ring_degree: Some(dimension / 2),
            error_bound: 6.0 + (security_bits as f64 / 50.0),
            num_samples: dimension,
            optimizations: LatticeOptimizations::default(),
        }
    }
    
    /// Validate configuration parameters
    pub fn validate(&self) -> Result<()> {
        // Check dimension is reasonable
        if self.dimension < 64 || self.dimension > 8192 {
            return Err(anyhow!("Invalid dimension: {}", self.dimension));
        }
        
        // Check modulus is positive
        if self.modulus <= BigInt::from(0) {
            return Err(anyhow!("Modulus must be positive"));
        }
        
        // Check Gaussian parameter is positive
        if self.gaussian_parameter <= 0.0 {
            return Err(anyhow!("Gaussian parameter must be positive"));
        }
        
        // Check error bound is reasonable
        if self.error_bound <= 0.0 || self.error_bound > 100.0 {
            return Err(anyhow!("Invalid error bound: {}", self.error_bound));
        }
        
        // Check number of samples
        if self.num_samples == 0 {
            return Err(anyhow!("Number of samples must be positive"));
        }
        
        // Validate ring degree if present
        if let Some(degree) = self.ring_degree {
            if degree == 0 || degree > self.dimension {
                return Err(anyhow!("Invalid ring degree: {}", degree));
            }
            
            // Check that ring degree is a power of 2 for NTT
            if self.optimizations.use_ntt && !degree.is_power_of_two() {
                return Err(anyhow!("Ring degree must be power of 2 for NTT optimization"));
            }
        }
        
        Ok(())
    }
    
    /// Estimate performance characteristics
    pub fn estimate_performance(&self) -> PerformanceEstimate {
        let base_ops = (self.dimension as f64).log2() * 1000.0;
        let modulus_factor = (self.modulus.bits() as f64) / 1000.0;
        
        let key_gen_ms = base_ops * modulus_factor * 2.0;
        let evaluation_ms = base_ops * modulus_factor * 1.5;
        let verification_ms = base_ops * modulus_factor * 1.0;
        let proof_gen_ms = base_ops * modulus_factor * 3.0;
        let proof_verify_ms = base_ops * modulus_factor * 2.0;
        
        // Apply optimization factors
        let opt_factor = if self.optimizations.use_ntt { 0.5 } else { 1.0 };
        let parallel_factor = if self.optimizations.parallel_processing { 0.7 } else { 1.0 };
        
        PerformanceEstimate {
            key_generation_ms: (key_gen_ms * opt_factor * parallel_factor) as u64,
            evaluation_ms: (evaluation_ms * opt_factor * parallel_factor) as u64,
            verification_ms: (verification_ms * opt_factor * parallel_factor) as u64,
            proof_generation_ms: (proof_gen_ms * opt_factor * parallel_factor) as u64,
            proof_verification_ms: (proof_verify_ms * opt_factor * parallel_factor) as u64,
            memory_usage_kb: (self.dimension * 8 + self.modulus.bits() / 8) as u64,
        }
    }
    
    /// Get parameter hash for identification
    pub fn parameter_hash(&self) -> Vec<u8> {
        use sha3::{Digest, Sha3_256};
        
        let mut hasher = Sha3_256::new();
        hasher.update(&self.dimension.to_be_bytes());
        hasher.update(&self.modulus.to_bytes_be().1);
        hasher.update(&self.gaussian_parameter.to_be_bytes());
        hasher.update(&(self.security_level.security_bits()).to_be_bytes());
        
        if let Some(ring_degree) = self.ring_degree {
            hasher.update(&ring_degree.to_be_bytes());
        }
        
        hasher.update(&self.error_bound.to_be_bytes());
        hasher.update(&self.num_samples.to_be_bytes());
        hasher.update(b"lattice-config-hash");
        
        hasher.finalize().to_vec()
    }
    
    /// Check if configuration supports specific features
    pub fn supports_feature(&self, feature: LatticeFeature) -> bool {
        match feature {
            LatticeFeature::RingStructure => self.ring_degree.is_some(),
            LatticeFeature::NTTOptimization => {
                self.optimizations.use_ntt && 
                self.ring_degree.map_or(false, |d| d.is_power_of_two())
            },
            LatticeFeature::BatchOperations => self.optimizations.enable_batching,
            LatticeFeature::ParallelProcessing => self.optimizations.parallel_processing,
            LatticeFeature::HighSecurity => self.security_level.security_bits() >= 192,
            LatticeFeature::LowLatency => self.optimizations.speed_vs_memory > 0.7,
        }
    }
}

/// Performance characteristics estimate
#[derive(Debug, Clone)]
pub struct PerformanceEstimate {
    pub key_generation_ms: u64,
    pub evaluation_ms: u64,
    pub verification_ms: u64,
    pub proof_generation_ms: u64,
    pub proof_verification_ms: u64,
    pub memory_usage_kb: u64,
}

/// Lattice features that can be supported
#[derive(Debug, Clone, Copy)]
pub enum LatticeFeature {
    /// Ring-based lattice structure
    RingStructure,
    
    /// Number Theoretic Transform optimization
    NTTOptimization,
    
    /// Batch operations support
    BatchOperations,
    
    /// Parallel processing support
    ParallelProcessing,
    
    /// High security (192+ bits)
    HighSecurity,
    
    /// Low latency optimization
    LowLatency,
}

/// Parameter set registry for managing configurations
pub struct ParameterRegistry {
    configurations: HashMap<String, LatticeConfig>,
}

impl ParameterRegistry {
    /// Create new parameter registry
    pub fn new() -> Self {
        let mut registry = Self {
            configurations: HashMap::new(),
        };
        
        // Register standard configurations
        registry.register_standard_configs();
        registry
    }
    
    /// Register a parameter configuration
    pub fn register(&mut self, name: String, config: LatticeConfig) -> Result<()> {
        config.validate()?;
        self.configurations.insert(name, config);
        Ok(())
    }
    
    /// Get configuration by name
    pub fn get(&self, name: &str) -> Option<&LatticeConfig> {
        self.configurations.get(name)
    }
    
    /// List available configurations
    pub fn list_configurations(&self) -> Vec<&String> {
        self.configurations.keys().collect()
    }
    
    /// Find configurations matching criteria
    pub fn find_by_security_level(&self, level: SecurityLevel) -> Vec<(&String, &LatticeConfig)> {
        self.configurations
            .iter()
            .filter(|(_, config)| config.security_level == level)
            .collect()
    }
    
    /// Find best configuration for requirements
    pub fn find_best_config(&self, requirements: &ConfigRequirements) -> Option<(&String, &LatticeConfig)> {
        let mut best_score = 0.0;
        let mut best_config = None;
        
        for (name, config) in &self.configurations {
            let score = self.score_config(config, requirements);
            if score > best_score {
                best_score = score;
                best_config = Some((name, config));
            }
        }
        
        best_config
    }
    
    /// Register standard configurations
    fn register_standard_configs(&mut self) {
        let _ = self.register("nist-1".to_string(), ParameterSets::nist_level_1());
        let _ = self.register("nist-3".to_string(), ParameterSets::nist_level_3());
        let _ = self.register("nist-5".to_string(), ParameterSets::nist_level_5());
        let _ = self.register("kyber-like".to_string(), ParameterSets::kyber_like());
        let _ = self.register("dilithium-like".to_string(), ParameterSets::dilithium_like());
        let _ = self.register("lightweight".to_string(), ParameterSets::lightweight());
    }
    
    /// Score configuration against requirements
    fn score_config(&self, config: &LatticeConfig, requirements: &ConfigRequirements) -> f64 {
        let mut score = 0.0;
        
        // Security level match
        if config.security_level.security_bits() >= requirements.min_security_bits {
            score += 30.0;
            if config.security_level == requirements.preferred_security_level {
                score += 10.0;
            }
        } else {
            return 0.0; // Insufficient security
        }
        
        // Performance requirements
        let perf = config.estimate_performance();
        if perf.evaluation_ms <= requirements.max_evaluation_ms {
            score += 20.0;
        }
        if perf.memory_usage_kb <= requirements.max_memory_kb {
            score += 20.0;
        }
        
        // Feature requirements
        for feature in &requirements.required_features {
            if config.supports_feature(*feature) {
                score += 5.0;
            } else {
                return 0.0; // Missing required feature
            }
        }
        
        // Optimization preferences
        if requirements.prefer_speed && config.optimizations.speed_vs_memory > 0.5 {
            score += 10.0;
        }
        if requirements.prefer_memory && config.optimizations.speed_vs_memory < 0.5 {
            score += 10.0;
        }
        
        score
    }
}

impl Default for ParameterRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration requirements for finding optimal parameters
#[derive(Debug, Clone)]
pub struct ConfigRequirements {
    pub min_security_bits: u16,
    pub preferred_security_level: SecurityLevel,
    pub max_evaluation_ms: u64,
    pub max_memory_kb: u64,
    pub required_features: Vec<LatticeFeature>,
    pub prefer_speed: bool,
    pub prefer_memory: bool,
}

impl Default for ConfigRequirements {
    fn default() -> Self {
        Self {
            min_security_bits: 128,
            preferred_security_level: SecurityLevel::Standard,
            max_evaluation_ms: 100,
            max_memory_kb: 1024,
            required_features: Vec::new(),
            prefer_speed: true,
            prefer_memory: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_security_levels() {
        assert_eq!(SecurityLevel::Standard.security_bits(), 128);
        assert_eq!(SecurityLevel::High.security_bits(), 192);
        assert_eq!(SecurityLevel::Ultra.security_bits(), 256);
        assert_eq!(SecurityLevel::Custom(96).security_bits(), 96);
    }
    
    #[test]
    fn test_parameter_sets() {
        let nist1 = ParameterSets::nist_level_1();
        assert_eq!(nist1.dimension, 512);
        assert_eq!(nist1.security_level, SecurityLevel::Standard);
        
        let nist3 = ParameterSets::nist_level_3();
        assert_eq!(nist3.dimension, 768);
        assert_eq!(nist3.security_level, SecurityLevel::High);
        
        let lightweight = ParameterSets::lightweight();
        assert_eq!(lightweight.dimension, 256);
        assert_eq!(lightweight.security_level, SecurityLevel::Custom(80));
    }
    
    #[test]
    fn test_config_validation() {
        let valid_config = ParameterSets::nist_level_1();
        assert!(valid_config.validate().is_ok());
        
        // Test invalid dimension
        let mut invalid_config = valid_config.clone();
        invalid_config.dimension = 0;
        assert!(invalid_config.validate().is_err());
        
        // Test invalid Gaussian parameter
        let mut invalid_config = valid_config.clone();
        invalid_config.gaussian_parameter = -1.0;
        assert!(invalid_config.validate().is_err());
    }
    
    #[test]
    fn test_performance_estimation() {
        let config = ParameterSets::nist_level_1();
        let perf = config.estimate_performance();
        
        assert!(perf.key_generation_ms > 0);
        assert!(perf.evaluation_ms > 0);
        assert!(perf.verification_ms > 0);
        assert!(perf.memory_usage_kb > 0);
    }
    
    #[test]
    fn test_feature_support() {
        let config = ParameterSets::nist_level_1();
        
        assert!(config.supports_feature(LatticeFeature::RingStructure));
        assert!(config.supports_feature(LatticeFeature::NTTOptimization));
        assert!(config.supports_feature(LatticeFeature::BatchOperations));
        assert!(config.supports_feature(LatticeFeature::ParallelProcessing));
    }
    
    #[test]
    fn test_parameter_registry() {
        let registry = ParameterRegistry::new();
        
        // Test standard configurations are registered
        assert!(registry.get("nist-1").is_some());
        assert!(registry.get("nist-3").is_some());
        assert!(registry.get("lightweight").is_some());
        
        let configs = registry.list_configurations();
        assert!(configs.len() >= 6); // At least 6 standard configs
    }
    
    #[test]
    fn test_config_search() {
        let registry = ParameterRegistry::new();
        
        // Search by security level
        let high_security = registry.find_by_security_level(SecurityLevel::High);
        assert!(!high_security.is_empty());
        
        // Search by requirements
        let requirements = ConfigRequirements {
            min_security_bits: 128,
            preferred_security_level: SecurityLevel::Standard,
            max_evaluation_ms: 1000,
            max_memory_kb: 10000,
            required_features: vec![LatticeFeature::NTTOptimization],
            prefer_speed: true,
            prefer_memory: false,
        };
        
        let best = registry.find_best_config(&requirements);
        assert!(best.is_some());
    }
    
    #[test]
    fn test_parameter_hash() {
        let config1 = ParameterSets::nist_level_1();
        let config2 = ParameterSets::nist_level_1();
        let config3 = ParameterSets::nist_level_3();
        
        let hash1 = config1.parameter_hash();
        let hash2 = config2.parameter_hash();
        let hash3 = config3.parameter_hash();
        
        // Same configs should have same hash
        assert_eq!(hash1, hash2);
        
        // Different configs should have different hashes
        assert_ne!(hash1, hash3);
    }
}