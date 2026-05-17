//! Configuration types for TemporalShield-STARK
//!
//! Defines all configuration parameters for the protocol.

use serde::{Deserialize, Serialize};
use crate::error::{TemporalError, TemporalResult};

/// 256-bit prime for Shamir secret sharing
/// This is the secp256k1 field prime: 2^256 - 2^32 - 977
pub const FIELD_PRIME_SECP256K1: [u8; 32] = [
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xfe, 0xff, 0xff, 0xfc, 0x2f,
];

/// Main configuration for TemporalShield
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TemporalShieldConfig {
    /// Minimum trustees required for reconstruction (k)
    /// Must satisfy: 1 <= k <= n
    pub threshold: usize,

    /// Total number of trustees (n)
    /// Recommended: n >= 3 for reasonable security
    pub total_trustees: usize,

    /// Field prime for Shamir arithmetic (256-bit)
    pub field_prime: [u8; 32],

    /// STARK security level in bits
    /// Common values: 80, 100, 128
    pub security_level: usize,

    /// Blowup factor for STARK (trace expansion)
    /// Higher = more security, larger proofs
    /// Typical: 4, 8, 16
    pub blowup_factor: usize,

    /// Field extension for STARK
    /// Quadratic provides good security/performance tradeoff
    pub field_extension: FieldExtension,

    /// Number of FRI queries
    /// More queries = higher security, larger proofs
    pub fri_num_queries: usize,

    /// FRI folding factor
    /// Typical: 4, 8, 16
    pub fri_folding_factor: usize,

    /// Maximum FRI remainder polynomial degree
    pub fri_max_remainder_degree: usize,

    /// Grinding factor for proof-of-work in STARK
    /// Higher = more prover work, harder to forge
    pub grinding_factor: u32,

    /// Maximum message size in bytes
    /// Prevents DoS via huge messages
    pub max_message_size: usize,

    /// Chunk size for splitting key into field elements
    pub chunk_size: usize,
}

/// Field extension options for STARK
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum FieldExtension {
    /// No extension (base field only)
    None,
    /// Quadratic extension (degree 2)
    Quadratic,
    /// Cubic extension (degree 3)
    Cubic,
}

impl Default for TemporalShieldConfig {
    fn default() -> Self {
        Self::secure_128()
    }
}

impl TemporalShieldConfig {
    /// Create configuration for 128-bit security level
    pub fn secure_128() -> Self {
        Self {
            threshold: 3,
            total_trustees: 5,
            field_prime: FIELD_PRIME_SECP256K1,
            security_level: 128,
            blowup_factor: 8,
            field_extension: FieldExtension::Quadratic,
            fri_num_queries: 28,
            fri_folding_factor: 8,
            fri_max_remainder_degree: 127,
            grinding_factor: 16,
            max_message_size: 1024 * 1024 * 10, // 10 MB
            chunk_size: 31, // 31 bytes to fit in 256-bit field with room for reduction
        }
    }

    /// Create configuration for 100-bit security level (faster proofs)
    pub fn secure_100() -> Self {
        Self {
            threshold: 3,
            total_trustees: 5,
            field_prime: FIELD_PRIME_SECP256K1,
            security_level: 100,
            blowup_factor: 8,
            field_extension: FieldExtension::Quadratic,
            fri_num_queries: 20,
            fri_folding_factor: 8,
            fri_max_remainder_degree: 127,
            grinding_factor: 12,
            max_message_size: 1024 * 1024 * 10,
            chunk_size: 31,
        }
    }

    /// Create configuration for maximum security (256-bit target)
    pub fn maximum_security() -> Self {
        Self {
            threshold: 5,
            total_trustees: 9,
            field_prime: FIELD_PRIME_SECP256K1,
            security_level: 128, // STARK security bounded by field size
            blowup_factor: 16,
            field_extension: FieldExtension::Cubic,
            fri_num_queries: 50,
            fri_folding_factor: 4,
            fri_max_remainder_degree: 63,
            grinding_factor: 20,
            max_message_size: 1024 * 1024 * 10,
            chunk_size: 31,
        }
    }

    /// Create a custom configuration with validation
    pub fn custom(
        threshold: usize,
        total_trustees: usize,
        security_level: usize,
    ) -> TemporalResult<Self> {
        let mut config = Self::secure_128();
        config.threshold = threshold;
        config.total_trustees = total_trustees;
        config.security_level = security_level;
        config.validate()?;
        Ok(config)
    }

    /// Validate the configuration
    pub fn validate(&self) -> TemporalResult<()> {
        // Threshold validation
        if self.threshold == 0 || self.threshold > self.total_trustees {
            return Err(TemporalError::InvalidThreshold {
                threshold: self.threshold,
                total_trustees: self.total_trustees,
            });
        }

        // Minimum trustees for meaningful security
        if self.total_trustees < 2 {
            return Err(TemporalError::InvalidThreshold {
                threshold: self.threshold,
                total_trustees: self.total_trustees,
            });
        }

        // Security level bounds
        if self.security_level < 80 || self.security_level > 256 {
            return Err(TemporalError::ConfigMismatch);
        }

        // Blowup factor must be power of 2
        if !self.blowup_factor.is_power_of_two() || self.blowup_factor < 2 {
            return Err(TemporalError::ConfigMismatch);
        }

        Ok(())
    }

    /// Compute a hash of the configuration for verification
    pub fn hash(&self) -> [u8; 32] {
        use blake3::Hasher;
        let mut hasher = Hasher::new();
        hasher.update(&(self.threshold as u64).to_le_bytes());
        hasher.update(&(self.total_trustees as u64).to_le_bytes());
        hasher.update(&self.field_prime);
        hasher.update(&(self.security_level as u64).to_le_bytes());
        hasher.update(&(self.blowup_factor as u64).to_le_bytes());
        hasher.update(&[self.field_extension as u8]);
        hasher.update(&(self.fri_num_queries as u64).to_le_bytes());
        *hasher.finalize().as_bytes()
    }

    /// Get the number of chunks needed for a message of given size
    pub fn num_chunks(&self, message_len: usize) -> usize {
        (message_len + self.chunk_size - 1) / self.chunk_size
    }

    /// Convert to Winterfell ProofOptions
    pub fn to_proof_options(&self) -> winter_air::ProofOptions {
        use winter_air::{FieldExtension as WinterFE, ProofOptions};

        let field_ext = match self.field_extension {
            FieldExtension::None => WinterFE::None,
            FieldExtension::Quadratic => WinterFE::Quadratic,
            FieldExtension::Cubic => WinterFE::Cubic,
        };

        ProofOptions::new(
            self.fri_num_queries,
            self.blowup_factor,
            self.grinding_factor,
            field_ext,
            self.fri_folding_factor,
            self.fri_max_remainder_degree,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_valid() {
        let config = TemporalShieldConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_invalid_threshold() {
        let config = TemporalShieldConfig {
            threshold: 6,
            total_trustees: 5,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_hash_deterministic() {
        let config1 = TemporalShieldConfig::default();
        let config2 = TemporalShieldConfig::default();
        assert_eq!(config1.hash(), config2.hash());
    }
}
