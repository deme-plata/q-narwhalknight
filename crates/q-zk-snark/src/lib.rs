//! Q-NarwhalKnight Zero-Knowledge SNARK Toolkit
//!
//! This crate provides a comprehensive suite of zk-SNARK protocols optimized
//! for the Q-NarwhalKnight blockchain, including Groth16, PLONK, Marlin, and Sonic.

use anyhow::Result;

pub mod circuits;
pub mod groth16;
pub mod plonk;
pub mod verification;
pub mod wallet_privacy;

// Re-exports for convenience
pub use circuits::*;
pub use groth16::*;
pub use plonk::*;
pub use wallet_privacy::*;

/// Core SNARK trait that all proof systems must implement
pub trait SNARK<F: ark_ff::Field> {
    type Circuit;
    type ProvingKey;
    type VerifyingKey;
    type Proof;
    type PublicInput;

    /// Generate proving and verifying keys for a circuit
    fn setup(circuit: &Self::Circuit) -> Result<(Self::ProvingKey, Self::VerifyingKey)>;

    /// Generate a proof for the given circuit and witness
    fn prove(
        proving_key: &Self::ProvingKey,
        circuit: &Self::Circuit,
        public_input: &Self::PublicInput,
    ) -> Result<Self::Proof>;

    /// Verify a proof
    fn verify(
        verifying_key: &Self::VerifyingKey,
        public_input: &Self::PublicInput,
        proof: &Self::Proof,
    ) -> Result<bool>;
}

/// Supported SNARK protocols
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum SNARKProtocol {
    /// Groth16 - most efficient verification
    Groth16,
    /// PLONK - universal setup
    PLONK,
    /// Marlin - transparent setup
    Marlin,
    /// Sonic - updatable setup  
    Sonic,
}

/// SNARK configuration parameters
#[derive(Debug, Clone)]
pub struct SNARKConfig {
    /// Protocol to use
    pub protocol: SNARKProtocol,
    /// Security parameter in bits
    pub security_bits: usize,
    /// Enable parallel proving
    pub parallel_proving: bool,
    /// Maximum circuit size
    pub max_constraints: usize,
    /// Batch verification support
    pub batch_verification: bool,
}

impl Default for SNARKConfig {
    fn default() -> Self {
        Self {
            protocol: SNARKProtocol::Groth16,
            security_bits: 128,
            parallel_proving: true,
            max_constraints: 1_000_000,
            batch_verification: true,
        }
    }
}

/// Universal SNARK interface that dispatches to specific protocols
pub struct UniversalSNARK {
    config: SNARKConfig,
}

impl UniversalSNARK {
    /// Create a new universal SNARK with the given configuration
    pub fn new(config: SNARKConfig) -> Self {
        Self { config }
    }

    /// Get recommended protocol for given constraints
    pub fn recommend_protocol(constraints: usize) -> SNARKProtocol {
        match constraints {
            0..=10_000 => SNARKProtocol::Groth16,         // Small circuits
            10_001..=100_000 => SNARKProtocol::PLONK,     // Medium circuits
            100_001..=1_000_000 => SNARKProtocol::Marlin, // Large circuits
            _ => SNARKProtocol::Sonic,                    // Very large circuits
        }
    }
}

/// Error types for the SNARK toolkit
#[derive(Debug, thiserror::Error)]
pub enum SNARKError {
    #[error("Circuit compilation failed: {0}")]
    CircuitCompilation(String),

    #[error("Proving failed: {0}")]
    ProvingFailed(String),

    #[error("Verification failed: {0}")]
    VerificationFailed(String),

    #[error("Setup failed: {0}")]
    SetupFailed(String),

    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Cryptographic error: {0}")]
    Cryptographic(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_protocol_recommendation() {
        assert_eq!(
            UniversalSNARK::recommend_protocol(5_000),
            SNARKProtocol::Groth16
        );
        assert_eq!(
            UniversalSNARK::recommend_protocol(50_000),
            SNARKProtocol::PLONK
        );
        assert_eq!(
            UniversalSNARK::recommend_protocol(500_000),
            SNARKProtocol::Marlin
        );
        assert_eq!(
            UniversalSNARK::recommend_protocol(2_000_000),
            SNARKProtocol::Sonic
        );
    }

    #[test]
    fn test_config_defaults() {
        let config = SNARKConfig::default();
        assert_eq!(config.protocol, SNARKProtocol::Groth16);
        assert_eq!(config.security_bits, 128);
        assert!(config.parallel_proving);
        assert!(config.batch_verification);
    }
}
