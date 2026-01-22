//! TemporalEnvelope - The sealed container for protected data
//!
//! Contains all information needed to verify and reconstruct the protected message.

use serde::{Deserialize, Serialize};
use crate::crypto::EncryptedShare;

/// The sealed envelope stored on-chain or transmitted
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalEnvelope {
    /// C = M ⊕ K (OTP-encrypted message)
    /// Information-theoretically secure ciphertext
    pub ciphertext: Vec<u8>,

    /// Commitment to OTP key: BLAKE3(K || blinding)
    /// Allows verification without revealing K
    pub key_commitment: [u8; 32],

    /// Commitments to each share: {BLAKE3(s_i)}
    /// Enables share verification during reconstruction
    pub share_commitments: Vec<[u8; 32]>,

    /// ML-KEM encrypted shares: {(trustee_id, encrypted_share, kem_ciphertext, nonce)}
    /// Each share protected with post-quantum key encapsulation
    pub encrypted_shares: Vec<EncryptedShare>,

    /// zk-STARK proof of Shamir consistency (serialized)
    /// NO TRUSTED SETUP - all randomness is public
    pub stark_proof: Vec<u8>,

    /// Envelope metadata
    pub metadata: EnvelopeMetadata,
}

/// Metadata about the envelope
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvelopeMetadata {
    /// Unix timestamp when envelope was created
    pub timestamp: u64,

    /// Protocol version (2 = STARK-enhanced)
    pub version: u16,

    /// Threshold parameter (k)
    pub threshold: usize,

    /// Total trustees (n)
    pub total_trustees: usize,

    /// Hash of the configuration used
    pub config_hash: [u8; 32],

    /// Original message size (for validation)
    pub message_size: usize,

    /// Number of field element chunks
    pub num_chunks: usize,
}

impl TemporalEnvelope {
    /// Create a new envelope with the given components
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ciphertext: Vec<u8>,
        key_commitment: [u8; 32],
        share_commitments: Vec<[u8; 32]>,
        encrypted_shares: Vec<EncryptedShare>,
        stark_proof: Vec<u8>,
        threshold: usize,
        total_trustees: usize,
        config_hash: [u8; 32],
        num_chunks: usize,
    ) -> Self {
        let message_size = ciphertext.len();
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Self {
            ciphertext,
            key_commitment,
            share_commitments,
            encrypted_shares,
            stark_proof,
            metadata: EnvelopeMetadata {
                timestamp,
                version: 2, // STARK-enhanced version
                threshold,
                total_trustees,
                config_hash,
                message_size,
                num_chunks,
            },
        }
    }

    /// Get the size of the envelope in bytes (approximate)
    pub fn size(&self) -> usize {
        self.ciphertext.len()
            + 32 // key_commitment
            + self.share_commitments.len() * 32
            + self.encrypted_shares.iter().map(|s| s.size()).sum::<usize>()
            + self.stark_proof.len()
            + std::mem::size_of::<EnvelopeMetadata>()
    }

    /// Verify basic structural integrity (not cryptographic verification)
    pub fn check_structure(&self) -> bool {
        // Check share count matches
        if self.share_commitments.len() != self.metadata.total_trustees {
            return false;
        }
        if self.encrypted_shares.len() != self.metadata.total_trustees {
            return false;
        }

        // Check threshold validity
        if self.metadata.threshold == 0 || self.metadata.threshold > self.metadata.total_trustees {
            return false;
        }

        // Check ciphertext matches declared size
        if self.ciphertext.len() != self.metadata.message_size {
            return false;
        }

        // Check version
        if self.metadata.version != 2 {
            return false;
        }

        true
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>, crate::error::TemporalError> {
        bincode::serialize(self).map_err(|e| crate::error::TemporalError::SerializationFailed(e.to_string()))
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, crate::error::TemporalError> {
        bincode::deserialize(bytes).map_err(|e| crate::error::TemporalError::DeserializationFailed(e.to_string()))
    }

    /// Get a hex-encoded identifier for this envelope
    pub fn id(&self) -> String {
        use blake3::Hasher;
        let mut hasher = Hasher::new();
        hasher.update(&self.key_commitment);
        hasher.update(&self.metadata.timestamp.to_le_bytes());
        let hash = hasher.finalize();
        hex::encode(&hash.as_bytes()[..16])
    }
}

/// Public inputs extracted from envelope for STARK verification
#[derive(Debug, Clone)]
pub struct EnvelopePublicInputs {
    /// Key commitment (field elements)
    pub key_commitment: [u8; 32],
    /// Share commitments
    pub share_commitments: Vec<[u8; 32]>,
    /// Threshold k
    pub threshold: usize,
    /// Total trustees n
    pub total_trustees: usize,
    /// Number of chunks
    pub num_chunks: usize,
}

impl From<&TemporalEnvelope> for EnvelopePublicInputs {
    fn from(envelope: &TemporalEnvelope) -> Self {
        Self {
            key_commitment: envelope.key_commitment,
            share_commitments: envelope.share_commitments.clone(),
            threshold: envelope.metadata.threshold,
            total_trustees: envelope.metadata.total_trustees,
            num_chunks: envelope.metadata.num_chunks,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_envelope_structure_check() {
        let envelope = TemporalEnvelope {
            ciphertext: vec![0u8; 100],
            key_commitment: [0u8; 32],
            share_commitments: vec![[0u8; 32]; 5],
            encrypted_shares: vec![],
            stark_proof: vec![],
            metadata: EnvelopeMetadata {
                timestamp: 0,
                version: 2,
                threshold: 3,
                total_trustees: 5,
                config_hash: [0u8; 32],
                message_size: 100,
                num_chunks: 4,
            },
        };
        // Will fail because encrypted_shares is empty but total_trustees is 5
        assert!(!envelope.check_structure());
    }

    #[test]
    fn test_envelope_serialization() {
        let envelope = TemporalEnvelope {
            ciphertext: vec![1, 2, 3, 4],
            key_commitment: [42u8; 32],
            share_commitments: vec![[1u8; 32], [2u8; 32]],
            encrypted_shares: vec![],
            stark_proof: vec![5, 6, 7],
            metadata: EnvelopeMetadata {
                timestamp: 1234567890,
                version: 2,
                threshold: 1,
                total_trustees: 2,
                config_hash: [0u8; 32],
                message_size: 4,
                num_chunks: 1,
            },
        };

        let bytes = envelope.to_bytes().unwrap();
        let restored = TemporalEnvelope::from_bytes(&bytes).unwrap();

        assert_eq!(envelope.ciphertext, restored.ciphertext);
        assert_eq!(envelope.key_commitment, restored.key_commitment);
        assert_eq!(envelope.metadata.timestamp, restored.metadata.timestamp);
    }
}
