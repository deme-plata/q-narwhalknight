//! Temporal Memo Protector - TemporalShield protection for private transaction memos
//!
//! Wraps private transaction memos in TemporalEnvelope for HNDL attack resistance.
//! Uses (3,5) threshold sharing by default for private memos.
//!
//! ## Version Bytes for Backward Compatibility
//! - 0x00: Plaintext memo
//! - 0x01: AES-encrypted memo (legacy)
//! - 0x02: TemporalShield-protected memo (post-quantum)

use std::sync::Arc;

use q_temporal_shield::{
    TemporalShield, TemporalShieldConfig, TemporalEnvelope, TrusteePublicKey,
};
use tracing::{info, warn, error};

/// Version bytes for memo format detection
pub const VERSION_PLAINTEXT: u8 = 0x00;
pub const VERSION_AES_ENCRYPTED: u8 = 0x01;
pub const VERSION_TEMPORAL_SHIELD: u8 = 0x02;

/// Decoded memo format for backward compatibility
#[derive(Debug, Clone)]
pub enum MemoFormat {
    /// Plaintext memo (version 0x00)
    Plaintext(Vec<u8>),
    /// AES-encrypted memo (version 0x01, legacy)
    AesEncrypted(Vec<u8>),
    /// TemporalShield-protected memo (version 0x02, post-quantum)
    TemporalShield(TemporalEnvelope),
    /// Unknown/legacy format
    Legacy(Vec<u8>),
}

/// Temporal Memo Protector - protects transaction memos with TemporalShield
pub struct TemporalMemoProtector {
    /// TemporalShield instance
    shield: TemporalShield,
    /// Trustees for memo protection (3-of-5 threshold)
    trustees: Vec<TrusteePublicKey>,
    /// Threshold (k)
    threshold: usize,
    /// Total trustees (n)
    total_trustees: usize,
}

impl TemporalMemoProtector {
    /// Create a new memo protector with the given trustees
    ///
    /// # Arguments
    /// * `trustees` - Vector of trustee public keys (must be 5 for default config)
    /// * `threshold` - Minimum shares needed to reconstruct (default: 3)
    pub fn new(trustees: Vec<TrusteePublicKey>, threshold: usize) -> Result<Self, String> {
        let total_trustees = trustees.len();

        if total_trustees < threshold {
            return Err(format!(
                "Not enough trustees: have {}, need at least {}",
                total_trustees, threshold
            ));
        }

        let config = TemporalShieldConfig::custom(threshold, total_trustees, 128)
            .map_err(|e| format!("Invalid config: {:?}", e))?;
        let shield = TemporalShield::new(config);

        Ok(Self {
            shield,
            trustees,
            threshold,
            total_trustees,
        })
    }

    /// Create with default (3,5) threshold
    pub fn new_default(trustees: Vec<TrusteePublicKey>) -> Result<Self, String> {
        if trustees.len() != 5 {
            return Err(format!(
                "Default config requires exactly 5 trustees, got {}",
                trustees.len()
            ));
        }
        Self::new(trustees, 3)
    }

    /// Protect a memo with TemporalShield
    ///
    /// Returns a TemporalEnvelope that can be stored/transmitted safely.
    pub fn protect_memo(&self, memo: &[u8]) -> Result<TemporalEnvelope, String> {
        if memo.is_empty() {
            return Err("Cannot protect empty memo".to_string());
        }

        self.shield
            .protect(memo, &self.trustees)
            .map_err(|e| format!("Protection failed: {:?}", e))
    }

    /// Encode a protected envelope for transaction storage
    ///
    /// Returns bytes with version prefix for backward compatibility.
    pub fn encode_for_transaction(envelope: &TemporalEnvelope) -> Result<Vec<u8>, String> {
        let envelope_bytes = envelope.to_bytes()
            .map_err(|e| format!("Serialization failed: {:?}", e))?;

        let mut encoded = Vec::with_capacity(1 + envelope_bytes.len());
        encoded.push(VERSION_TEMPORAL_SHIELD);
        encoded.extend(envelope_bytes);

        Ok(encoded)
    }

    /// Decode a memo with automatic format detection
    pub fn decode_memo(data: &[u8]) -> Result<MemoFormat, String> {
        if data.is_empty() {
            return Err("Empty memo data".to_string());
        }

        match data.first() {
            Some(&VERSION_PLAINTEXT) => {
                Ok(MemoFormat::Plaintext(data[1..].to_vec()))
            }
            Some(&VERSION_AES_ENCRYPTED) => {
                Ok(MemoFormat::AesEncrypted(data[1..].to_vec()))
            }
            Some(&VERSION_TEMPORAL_SHIELD) => {
                let envelope = TemporalEnvelope::from_bytes(&data[1..])
                    .map_err(|e| format!("Failed to parse envelope: {:?}", e))?;
                Ok(MemoFormat::TemporalShield(envelope))
            }
            _ => {
                // Legacy format without version byte
                Ok(MemoFormat::Legacy(data.to_vec()))
            }
        }
    }

    /// Verify a TemporalShield-protected memo
    pub fn verify_memo(&self, envelope: &TemporalEnvelope) -> Result<(), String> {
        self.shield
            .verify(envelope)
            .map_err(|e| format!("Verification failed: {:?}", e))
    }

    /// Reconstruct memo from decrypted shares
    ///
    /// # Arguments
    /// * `envelope` - The protected envelope
    /// * `decrypted_shares` - At least `threshold` decrypted shares: (index, share_data)
    pub fn reconstruct_memo(
        &self,
        envelope: &TemporalEnvelope,
        decrypted_shares: &[(usize, Vec<u8>)],
    ) -> Result<Vec<u8>, String> {
        self.shield
            .reconstruct(envelope, decrypted_shares)
            .map_err(|e| format!("Reconstruction failed: {:?}", e))
    }

    /// Get threshold (k)
    pub fn threshold(&self) -> usize {
        self.threshold
    }

    /// Get total trustees (n)
    pub fn total_trustees(&self) -> usize {
        self.total_trustees
    }
}

/// Helper to create a plaintext-encoded memo
pub fn encode_plaintext_memo(memo: &[u8]) -> Vec<u8> {
    let mut encoded = Vec::with_capacity(1 + memo.len());
    encoded.push(VERSION_PLAINTEXT);
    encoded.extend(memo);
    encoded
}

/// Helper to create an AES-encrypted memo (legacy compatibility)
pub fn encode_aes_memo(encrypted_memo: &[u8]) -> Vec<u8> {
    let mut encoded = Vec::with_capacity(1 + encrypted_memo.len());
    encoded.push(VERSION_AES_ENCRYPTED);
    encoded.extend(encrypted_memo);
    encoded
}

/// Check if a memo is TemporalShield-protected
pub fn is_temporal_protected(data: &[u8]) -> bool {
    data.first() == Some(&VERSION_TEMPORAL_SHIELD)
}

/// Check if a memo needs migration to TemporalShield
pub fn needs_migration(data: &[u8]) -> bool {
    match data.first() {
        Some(&VERSION_PLAINTEXT) | Some(&VERSION_AES_ENCRYPTED) => true,
        Some(&VERSION_TEMPORAL_SHIELD) => false,
        _ => true, // Legacy format needs migration
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use q_temporal_shield::TrusteePublicKey;

    fn generate_test_trustees(n: usize) -> Vec<TrusteePublicKey> {
        (0..n)
            .map(|i| {
                let keypair = TrusteePublicKey::generate(Some(format!("Test-{}", i))).unwrap();
                keypair.public_key
            })
            .collect()
    }

    #[test]
    fn test_memo_protection_roundtrip() {
        let trustees = generate_test_trustees(5);
        let protector = TemporalMemoProtector::new_default(trustees.clone()).unwrap();

        let memo = b"Secret memo for private transaction";
        let envelope = protector.protect_memo(memo).unwrap();

        // Verify the envelope
        assert!(protector.verify_memo(&envelope).is_ok());

        // Encode for transaction
        let encoded = TemporalMemoProtector::encode_for_transaction(&envelope).unwrap();
        assert_eq!(encoded[0], VERSION_TEMPORAL_SHIELD);

        // Decode
        let decoded = TemporalMemoProtector::decode_memo(&encoded).unwrap();
        match decoded {
            MemoFormat::TemporalShield(env) => {
                assert!(protector.verify_memo(&env).is_ok());
            }
            _ => panic!("Expected TemporalShield format"),
        }
    }

    #[test]
    fn test_version_detection() {
        // Plaintext
        let plaintext = encode_plaintext_memo(b"hello");
        assert!(needs_migration(&plaintext));
        assert!(!is_temporal_protected(&plaintext));

        // AES
        let aes = encode_aes_memo(b"encrypted");
        assert!(needs_migration(&aes));
        assert!(!is_temporal_protected(&aes));

        // Temporal
        let trustees = generate_test_trustees(5);
        let protector = TemporalMemoProtector::new_default(trustees).unwrap();
        let envelope = protector.protect_memo(b"memo").unwrap();
        let temporal = TemporalMemoProtector::encode_for_transaction(&envelope).unwrap();
        assert!(!needs_migration(&temporal));
        assert!(is_temporal_protected(&temporal));
    }

    #[test]
    fn test_backward_compatibility() {
        // Test decoding legacy formats
        let plaintext = encode_plaintext_memo(b"hello world");
        match TemporalMemoProtector::decode_memo(&plaintext).unwrap() {
            MemoFormat::Plaintext(data) => assert_eq!(data, b"hello world"),
            _ => panic!("Expected Plaintext"),
        }

        let aes = encode_aes_memo(b"encrypted_data");
        match TemporalMemoProtector::decode_memo(&aes).unwrap() {
            MemoFormat::AesEncrypted(data) => assert_eq!(data, b"encrypted_data"),
            _ => panic!("Expected AesEncrypted"),
        }

        // Legacy format (no version byte, first byte not 0x00-0x02)
        let legacy = vec![0xFF, 0x01, 0x02];
        match TemporalMemoProtector::decode_memo(&legacy).unwrap() {
            MemoFormat::Legacy(data) => assert_eq!(data, vec![0xFF, 0x01, 0x02]),
            _ => panic!("Expected Legacy"),
        }
    }

    #[test]
    fn test_wrong_trustee_count() {
        let trustees = generate_test_trustees(3); // Wrong count for default
        let result = TemporalMemoProtector::new_default(trustees);
        assert!(result.is_err());
    }

    #[test]
    fn test_custom_threshold() {
        let trustees = generate_test_trustees(7);
        let protector = TemporalMemoProtector::new(trustees, 4).unwrap();

        assert_eq!(protector.threshold(), 4);
        assert_eq!(protector.total_trustees(), 7);
    }
}
