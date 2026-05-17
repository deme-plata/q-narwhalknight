//! Zero-Knowledge Enhanced P2P Networking for Q-NarwhalKnight
//!
//! This crate implements anonymous, verifiable P2P connections using zero-knowledge proofs.
//! It provides privacy-preserving validator identity verification, network membership proofs,
//! and connection quality attestations without revealing sensitive network topology or metrics.

pub mod anonymous_identity;
pub mod connection_quality;
pub mod network_membership;
pub mod zk_p2p_manager;

// Re-export main types
pub use anonymous_identity::{OnionOwnershipProof, ValidatorEligibilityProof};
pub use connection_quality::{ConnectionQualityProof, ConsensusParticipationProof};
pub use network_membership::{MerkleTree, NetworkMembershipProof, verify_membership_proof};
pub use zk_p2p_manager::{VerifiedP2pConnection, ZkP2pConfig, ZkP2pManager};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::time::SystemTime;

/// ZK-enhanced P2P connection verification status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ZkVerificationStatus {
    /// Not verified
    Unverified,
    /// Partially verified (some proofs valid)
    PartiallyVerified,
    /// Fully verified (all proofs valid)
    FullyVerified,
    /// Verification failed
    Failed(String),
}

/// Verified peer information with ZK proofs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifiedPeer {
    /// Onion service address
    pub onion_address: String,
    /// Network membership proof
    pub membership_proof: NetworkMembershipProof,
    /// Validator eligibility proof
    pub eligibility_proof: ValidatorEligibilityProof,
    /// Discovery timestamp
    pub discovery_timestamp: SystemTime,
    /// Current verification status
    pub verification_status: ZkVerificationStatus,
}

/// ZK-enhanced P2P connection errors
#[derive(Debug, thiserror::Error)]
pub enum ZkP2pError {
    #[error("Identity proof verification failed: {0}")]
    IdentityVerification(String),

    #[error("Network membership proof failed: {0}")]
    MembershipVerification(String),

    #[error("Connection quality proof failed: {0}")]
    QualityVerification(String),

    #[error("ZK proof generation failed: {0}")]
    ProofGeneration(String),

    #[error("DNS steganography failed: {0}")]
    DnsSteganography(String),

    #[error("Tor connection failed: {0}")]
    TorConnection(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Configuration error: {0}")]
    Configuration(String),
}

/// Helper functions for field arithmetic conversions
pub fn field_from_bytes(bytes: &[u8]) -> ark_bn254::Fr {
    use ark_ff::PrimeField;

    // Take first 32 bytes and convert to field element
    let mut repr = [0u8; 32];
    let len = std::cmp::min(bytes.len(), 32);
    repr[..len].copy_from_slice(&bytes[..len]);

    ark_bn254::Fr::from_le_bytes_mod_order(&repr)
}

pub fn field_from_string(s: &str) -> ark_bn254::Fr {
    field_from_bytes(s.as_bytes())
}

pub fn field_from_u64(val: u64) -> ark_bn254::Fr {
    ark_bn254::Fr::from(val)
}

/// Generate cryptographic randomness for ZK proofs
pub fn generate_secret_key() -> [u8; 32] {
    use rand::RngCore;
    let mut key = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut key);
    key
}

/// Derive public key from secret key (simplified)
pub fn derive_public_key(secret_key: &[u8; 32]) -> [u8; 32] {
    blake3::hash(secret_key).into()
}

/// Generate onion address from secret key
pub fn generate_onion_address(secret_key: &[u8; 32]) -> String {
    let hash = blake3::hash(secret_key);
    format!("{}.qnk.onion", hex::encode(&hash.as_bytes()[..16]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_conversions() {
        let bytes = b"test_data_for_field_conversion";
        let field_val = field_from_bytes(bytes);

        // Should not be zero for non-zero input
        assert_ne!(field_val, ark_bn254::Fr::from(0u64));
    }

    #[test]
    fn test_key_generation() {
        let key1 = generate_secret_key();
        let key2 = generate_secret_key();

        // Keys should be different
        assert_ne!(key1, key2);

        // Public keys should be deterministic
        let pub1a = derive_public_key(&key1);
        let pub1b = derive_public_key(&key1);
        assert_eq!(pub1a, pub1b);
    }

    #[test]
    fn test_onion_address_generation() {
        let secret_key = generate_secret_key();
        let address = generate_onion_address(&secret_key);

        assert!(address.ends_with(".qnk.onion"));
        assert_eq!(
            address.len(),
            "0123456789abcdef0123456789abcdef.qnk.onion".len()
        );
    }
}
