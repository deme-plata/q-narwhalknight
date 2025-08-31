/// VDF parameter management and configuration

use anyhow::{Result, anyhow};
use serde::{Serialize, Deserialize};
use num_bigint::BigUint;
use std::time::Duration;

use crate::VDFType;

/// VDF security levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SecurityLevel {
    /// Low security (testing only)
    Low,
    
    /// Standard security (128-bit)
    Standard,
    
    /// High security (192-bit)
    High,
    
    /// Ultra security (256-bit)
    Ultra,
    
    /// Custom security level
    Custom(u32),
}

impl SecurityLevel {
    /// Get security parameter in bits
    pub fn bits(&self) -> u32 {
        match self {
            SecurityLevel::Low => 80,
            SecurityLevel::Standard => 128,
            SecurityLevel::High => 192,
            SecurityLevel::Ultra => 256,
            SecurityLevel::Custom(bits) => *bits,
        }
    }
    
    /// Get recommended time parameter (iterations)
    pub fn time_parameter(&self) -> u64 {
        match self {
            SecurityLevel::Low => 1000,
            SecurityLevel::Standard => 10000,
            SecurityLevel::High => 100000,
            SecurityLevel::Ultra => 1000000,
            SecurityLevel::Custom(bits) => (*bits as u64) * 100,
        }
    }
    
    /// Get recommended modulus size in bits
    pub fn modulus_bits(&self) -> usize {
        match self {
            SecurityLevel::Low => 1024,
            SecurityLevel::Standard => 2048,
            SecurityLevel::High => 3072,
            SecurityLevel::Ultra => 4096,
            SecurityLevel::Custom(bits) => (*bits as usize) * 16,
        }
    }
}

/// VDF parameters configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VDFParameters {
    /// Security level
    pub security_level: SecurityLevel,
    
    /// Time parameter (number of sequential steps)
    pub time_parameter: u64,
    
    /// RSA modulus for groups of unknown order
    pub modulus: BigUint,
    
    /// VDF type to use
    pub vdf_type: VDFType,
    
    /// Proof generation interval (for long computations)
    pub checkpoint_interval: u64,
    
    /// Maximum proof size in bytes
    pub max_proof_size: usize,
    
    /// Enable parallel proof generation
    pub parallel_proofs: bool,
    
    /// Target evaluation time
    pub target_eval_time: Duration,
    
    /// Quantum enhancement enabled
    pub quantum_enhanced: bool,
}

impl Default for VDFParameters {
    fn default() -> Self {
        let security_level = SecurityLevel::Standard;
        let modulus = generate_default_modulus(security_level.modulus_bits());
        
        Self {
            security_level,
            time_parameter: security_level.time_parameter(),
            modulus,
            vdf_type: VDFType::Wesolowski,
            checkpoint_interval: 1000,
            max_proof_size: 4096,
            parallel_proofs: true,
            target_eval_time: Duration::from_millis(100),
            quantum_enhanced: false,
        }
    }
}

impl VDFParameters {
    /// Create parameters with specific time parameter
    pub fn with_time_parameter(time_parameter: u64) -> Self {
        Self {
            time_parameter,
            ..Default::default()
        }
    }
    
    /// Create parameters for specific security level
    pub fn for_security_level(level: SecurityLevel) -> Self {
        let modulus = generate_default_modulus(level.modulus_bits());
        
        Self {
            security_level: level,
            time_parameter: level.time_parameter(),
            modulus,
            ..Default::default()
        }
    }
    
    /// Create quantum-enhanced parameters
    pub fn quantum_enhanced(level: SecurityLevel) -> Self {
        let mut params = Self::for_security_level(level);
        params.quantum_enhanced = true;
        params.vdf_type = VDFType::QuantumHybrid;
        params
    }
    
    /// Validate parameters
    pub fn validate(&self) -> Result<()> {
        // Check time parameter is reasonable
        if self.time_parameter == 0 {
            return Err(anyhow!("Time parameter must be positive"));
        }
        
        if self.time_parameter > 10_000_000_000 {
            return Err(anyhow!("Time parameter too large"));
        }
        
        // Check modulus is valid
        if self.modulus.bits() < 512 {
            return Err(anyhow!("Modulus too small for security"));
        }
        
        // Check checkpoint interval
        if self.checkpoint_interval > self.time_parameter {
            return Err(anyhow!("Checkpoint interval exceeds time parameter"));
        }
        
        // Check proof size
        if self.max_proof_size < 32 {
            return Err(anyhow!("Maximum proof size too small"));
        }
        
        Ok(())
    }
    
    /// Estimate computation time
    pub fn estimate_computation_time(&self) -> Duration {
        // Rough estimate: 1 microsecond per iteration
        // This would be calibrated in practice
        let micros = self.time_parameter;
        Duration::from_micros(micros)
    }
    
    /// Estimate verification time
    pub fn estimate_verification_time(&self) -> Duration {
        // Verification is typically log(T) where T is time parameter
        let log_t = (self.time_parameter as f64).log2();
        let micros = (log_t * 100.0) as u64;
        Duration::from_micros(micros)
    }
    
    /// Get checkpoint count
    pub fn checkpoint_count(&self) -> u64 {
        (self.time_parameter + self.checkpoint_interval - 1) / self.checkpoint_interval
    }
    
    /// Adjust time parameter for target duration
    pub fn adjust_for_target_time(&mut self, actual_time: Duration) {
        if actual_time < self.target_eval_time {
            // Increase time parameter
            let ratio = self.target_eval_time.as_millis() as f64 / actual_time.as_millis() as f64;
            self.time_parameter = (self.time_parameter as f64 * ratio) as u64;
        } else if actual_time > self.target_eval_time * 2 {
            // Decrease time parameter
            let ratio = actual_time.as_millis() as f64 / self.target_eval_time.as_millis() as f64;
            self.time_parameter = (self.time_parameter as f64 / ratio) as u64;
        }
        
        // Keep within reasonable bounds
        self.time_parameter = self.time_parameter.max(100).min(10_000_000);
    }
}

/// Generate default RSA modulus for given bit size
fn generate_default_modulus(bits: usize) -> BigUint {
    // In production, generate proper RSA modulus
    // For now, use pre-computed values
    match bits {
        1024 => {
            BigUint::parse_bytes(
                b"135066410865995223349603216278805969938881475605667027524485143851526510604859533833940287150571909441798207282164471551373680419703964191743046496589274256239341020864383202110372958725762358509643110564073501508187510676594629205563685529475213500852879416377328533906109750544334999811150056977236890927563",
                10
            ).unwrap()
        },
        2048 => {
            BigUint::parse_bytes(
                b"25195908475657893494027183240048398571429282126204032027777137836043662020707595556264018525880784406918290641249515082189298559149176184502808489120072844992687392807287776735971418347270261896375014971824691165077613379859095700097330459748808428401797429100642458691817195118746121515172654632282216869987549182422433637259085141865462043576798423387184774447920739934236584823824281198163815010674810451660377306056201619676256133844143603833904414952634432190114657544454178424020924616515723350778707749817125772467962926386356373289912154831438167899885040445364023527381951378636564391212010397122822120720357",
                10
            ).unwrap()
        },
        _ => {
            // Default to 2048-bit modulus
            BigUint::parse_bytes(
                b"25195908475657893494027183240048398571429282126204032027777137836043662020707595556264018525880784406918290641249515082189298559149176184502808489120072844992687392807287776735971418347270261896375014971824691165077613379859095700097330459748808428401797429100642458691817195118746121515172654632282216869987549182422433637259085141865462043576798423387184774447920739934236584823824281198163815010674810451660377306056201619676256133844143603833904414952634432190114657544454178424020924616515723350778707749817125772467962926386356373289912154831438167899885040445364023527381951378636564391212010397122822120720357",
                10
            ).unwrap()
        }
    }
}

/// VDF benchmark parameters
#[derive(Debug, Clone)]
pub struct BenchmarkParameters {
    /// Time parameters to test
    pub time_parameters: Vec<u64>,
    
    /// Number of iterations per test
    pub iterations: usize,
    
    /// Warmup iterations
    pub warmup: usize,
    
    /// Enable detailed metrics
    pub detailed_metrics: bool,
}

impl Default for BenchmarkParameters {
    fn default() -> Self {
        Self {
            time_parameters: vec![100, 1000, 10000, 100000],
            iterations: 10,
            warmup: 2,
            detailed_metrics: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_security_levels() {
        assert_eq!(SecurityLevel::Standard.bits(), 128);
        assert_eq!(SecurityLevel::High.bits(), 192);
        assert_eq!(SecurityLevel::Ultra.bits(), 256);
        assert_eq!(SecurityLevel::Custom(512).bits(), 512);
    }
    
    #[test]
    fn test_default_parameters() {
        let params = VDFParameters::default();
        assert_eq!(params.security_level, SecurityLevel::Standard);
        assert!(params.validate().is_ok());
    }
    
    #[test]
    fn test_parameter_validation() {
        let mut params = VDFParameters::default();
        
        // Valid parameters
        assert!(params.validate().is_ok());
        
        // Invalid time parameter
        params.time_parameter = 0;
        assert!(params.validate().is_err());
        
        // Restore and test checkpoint interval
        params.time_parameter = 1000;
        params.checkpoint_interval = 2000;
        assert!(params.validate().is_err());
    }
    
    #[test]
    fn test_time_estimation() {
        let params = VDFParameters::with_time_parameter(1000);
        let estimated = params.estimate_computation_time();
        assert!(estimated.as_micros() > 0);
        
        let verify_time = params.estimate_verification_time();
        assert!(verify_time < estimated);
    }
    
    #[test]
    fn test_checkpoint_calculation() {
        let params = VDFParameters {
            time_parameter: 10000,
            checkpoint_interval: 1000,
            ..Default::default()
        };
        
        assert_eq!(params.checkpoint_count(), 10);
    }
}