//! # TemporalShield-STARK
//!
//! A cryptographic protocol combining information-theoretic encryption (one-time pad)
//! with post-quantum secure threshold secret sharing and zk-STARK proofs.
//!
//! ## Key Features
//!
//! - **Information-theoretic encryption**: OTP provides perfect secrecy for the payload
//! - **Post-quantum secure**: ML-KEM-1024 for key encapsulation
//! - **No trusted setup**: zk-STARKs eliminate ceremony and toxic waste
//! - **Threshold security**: Shamir (k,n) secret sharing
//! - **Verifiable**: Anyone can verify consistency without secrets
//!
//! ## Security Properties
//!
//! 1. **Payload**: C = M ⊕ K has information-theoretic secrecy
//! 2. **Key shares**: Protected by ML-KEM, requires breaking lattice problem
//! 3. **Threshold**: Any k-1 shares reveal zero bits about K
//! 4. **Proofs**: STARK consistency proofs are post-quantum secure
//!
//! ## Example Usage
//!
//! ```rust,ignore
//! use q_temporal_shield::{TemporalShield, TemporalShieldConfig};
//! use q_temporal_shield::trustee::TrusteePublicKey;
//!
//! // Generate trustees
//! let trustees: Vec<_> = (0..5)
//!     .map(|_| TrusteePublicKey::generate(None).unwrap())
//!     .collect();
//! let public_keys: Vec<_> = trustees.iter().map(|t| t.public_key.clone()).collect();
//!
//! // Create TemporalShield with (3,5) threshold
//! let config = TemporalShieldConfig::custom(3, 5, 128).unwrap();
//! let shield = TemporalShield::new(config);
//!
//! // Protect a message
//! let message = b"Secret message with indefinite value";
//! let envelope = shield.protect(message, &public_keys).unwrap();
//!
//! // Verify (anyone can do this - NO TRUSTED SETUP)
//! assert!(shield.verify(&envelope).is_ok());
//!
//! // Reconstruct with k=3 trustees
//! let shares: Vec<_> = trustees[0..3].iter()
//!     .map(|t| {
//!         let encrypted = &envelope.encrypted_shares[t.public_key.id];
//!         t.private_key.decrypt_share(encrypted).unwrap()
//!     })
//!     .collect();
//! let recovered = shield.reconstruct(&envelope, &shares).unwrap();
//! assert_eq!(recovered, message);
//! ```

#![warn(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]

pub mod config;
pub mod envelope;
pub mod error;
pub mod crypto;
pub mod shamir;
pub mod stark;
pub mod trustee;

pub use config::TemporalShieldConfig;
pub use envelope::TemporalEnvelope;
pub use error::{TemporalError, TemporalResult};
pub use trustee::{TrusteePublicKey, TrusteePrivateKey, TrusteeKeyPair};

use crypto::{hash, kem, rand, EncryptedShare};
use shamir::{ShamirShare, shamir_split, shamir_reconstruct};

/// Main TemporalShield-STARK implementation
///
/// Combines OTP encryption, Shamir secret sharing, ML-KEM key encapsulation,
/// and zk-STARK proofs for defense-in-depth against HNDL attacks.
pub struct TemporalShield {
    config: TemporalShieldConfig,
}

impl TemporalShield {
    /// Create a new TemporalShield instance with the given configuration
    pub fn new(config: TemporalShieldConfig) -> Self {
        Self { config }
    }

    /// Create with default 128-bit security configuration
    pub fn default_secure() -> Self {
        Self::new(TemporalShieldConfig::default())
    }

    /// Protect a message with TemporalShield-STARK
    ///
    /// # Arguments
    /// * `message` - The plaintext to protect
    /// * `trustees` - Public keys of trustees who will hold shares
    ///
    /// # Returns
    /// A `TemporalEnvelope` containing the encrypted message and all necessary
    /// data for verification and reconstruction.
    ///
    /// # Security
    /// - The message is XOR'd with a random OTP key (information-theoretic secrecy)
    /// - The OTP key is split using Shamir (k,n) threshold sharing
    /// - Each share is encrypted with the trustee's ML-KEM public key
    /// - A zk-STARK proves consistency (NO TRUSTED SETUP)
    pub fn protect(
        &self,
        message: &[u8],
        trustees: &[TrusteePublicKey],
    ) -> TemporalResult<TemporalEnvelope> {
        // Validate inputs
        self.config.validate()?;

        if trustees.len() != self.config.total_trustees {
            return Err(TemporalError::TrusteeCountMismatch {
                expected: self.config.total_trustees,
                actual: trustees.len(),
            });
        }

        if message.is_empty() {
            return Err(TemporalError::KeyLengthMismatch {
                key_len: 0,
                ciphertext_len: 0,
            });
        }

        if message.len() > self.config.max_message_size {
            return Err(TemporalError::KeyLengthMismatch {
                key_len: message.len(),
                ciphertext_len: self.config.max_message_size,
            });
        }

        // Step 1: Generate OTP key and encrypt message
        let otp_key = rand::random_bytes(message.len())?;
        let ciphertext = xor_bytes(message, &otp_key);

        // Step 2: Commit to OTP key
        let blinding = rand::random_32()?;
        let key_commitment = hash::commit(&otp_key, &blinding);

        // Step 3: Split OTP key into shares
        let shares = shamir_split(
            &otp_key,
            self.config.threshold,
            self.config.total_trustees,
        )?;

        // Step 4: Commit to each share
        let share_commitments: Vec<[u8; 32]> = shares
            .iter()
            .map(|s| hash::blake3_hash(&s.data))
            .collect();

        // Step 5: Generate STARK proof of consistency
        let stark_proof = stark::generate_proof(
            &otp_key,
            &blinding,
            &shares.iter().map(|s| s.data.clone()).collect::<Vec<_>>(),
            &key_commitment,
            &share_commitments,
            self.config.threshold,
            self.config.total_trustees,
            &self.config,
        )?;

        // Step 6: Encrypt each share for its trustee
        let encrypted_shares = self.encrypt_shares_for_trustees(&shares, trustees)?;

        // Step 7: Create envelope
        let num_chunks = self.config.num_chunks(message.len());
        let envelope = TemporalEnvelope::new(
            ciphertext,
            key_commitment,
            share_commitments,
            encrypted_shares,
            stark_proof,
            self.config.threshold,
            self.config.total_trustees,
            self.config.hash(),
            num_chunks,
        );

        // Step 8: Verify our own envelope before returning
        if !envelope.check_structure() {
            return Err(TemporalError::SerializationFailed(
                "Envelope structure check failed".to_string(),
            ));
        }

        Ok(envelope)
    }

    /// Verify a TemporalEnvelope
    ///
    /// # Security
    /// This verification requires NO TRUSTED SETUP.
    /// All randomness is derived from the public transcript (Fiat-Shamir).
    pub fn verify(&self, envelope: &TemporalEnvelope) -> TemporalResult<()> {
        // Check basic structure
        if !envelope.check_structure() {
            return Err(TemporalError::InvalidProofFormat);
        }

        // Verify metadata matches config
        if envelope.metadata.threshold != self.config.threshold
            || envelope.metadata.total_trustees != self.config.total_trustees
        {
            return Err(TemporalError::MetadataMismatch);
        }

        // Verify STARK proof
        stark::verify_proof(
            envelope.stark_proof.clone(),
            &envelope.key_commitment,
            &envelope.share_commitments,
            envelope.metadata.threshold,
            envelope.metadata.total_trustees,
        )?;

        Ok(())
    }

    /// Reconstruct the original message from decrypted shares
    ///
    /// # Arguments
    /// * `envelope` - The TemporalEnvelope
    /// * `decrypted_shares` - Decrypted shares from at least k trustees
    ///
    /// # Returns
    /// The original message
    pub fn reconstruct(
        &self,
        envelope: &TemporalEnvelope,
        decrypted_shares: &[(usize, Vec<u8>)],
    ) -> TemporalResult<Vec<u8>> {
        // Verify we have enough shares
        if decrypted_shares.len() < envelope.metadata.threshold {
            return Err(TemporalError::InsufficientShares {
                have: decrypted_shares.len(),
                need: envelope.metadata.threshold,
            });
        }

        // Verify each share against its commitment
        for (index, share_data) in decrypted_shares {
            let commitment = envelope.share_commitments.get(*index)
                .ok_or(TemporalError::InvalidShareIndex(*index))?;

            let computed = hash::blake3_hash(share_data);
            if &computed != commitment {
                return Err(TemporalError::ShareCommitmentMismatch { index: *index });
            }
        }

        // Convert to ShamirShare format
        let shares: Vec<ShamirShare> = decrypted_shares
            .iter()
            .map(|(idx, data)| ShamirShare {
                index: (*idx + 1) as u64,
                data: data.clone(),
            })
            .collect();

        // Reconstruct OTP key
        let otp_key = shamir_reconstruct(
            &shares,
            envelope.metadata.threshold,
            envelope.metadata.message_size,
        )?;

        // Verify key length matches ciphertext
        if otp_key.len() != envelope.ciphertext.len() {
            return Err(TemporalError::KeyLengthMismatch {
                key_len: otp_key.len(),
                ciphertext_len: envelope.ciphertext.len(),
            });
        }

        // Decrypt: M = C ⊕ K
        let message = xor_bytes(&envelope.ciphertext, &otp_key);

        Ok(message)
    }

    /// Get the configuration
    pub fn config(&self) -> &TemporalShieldConfig {
        &self.config
    }

    // Helper: Encrypt shares for trustees
    fn encrypt_shares_for_trustees(
        &self,
        shares: &[ShamirShare],
        trustees: &[TrusteePublicKey],
    ) -> TemporalResult<Vec<EncryptedShare>> {
        shares
            .iter()
            .zip(trustees.iter())
            .enumerate()
            .map(|(idx, (share, trustee))| {
                kem::encrypt_share_for_trustee(
                    &share.data,
                    &trustee.kem_public_key,
                    trustee.id,
                    idx as u32,
                )
            })
            .collect()
    }
}

/// XOR two byte slices
fn xor_bytes(a: &[u8], b: &[u8]) -> Vec<u8> {
    a.iter().zip(b.iter()).map(|(x, y)| x ^ y).collect()
}

/// Quick helper to create a TemporalShield with default config
pub fn temporal_shield() -> TemporalShield {
    TemporalShield::default_secure()
}

#[cfg(test)]
mod tests {
    use super::*;
    use trustee::TrusteeKeyPair;

    fn generate_test_trustees(n: usize) -> Vec<TrusteeKeyPair> {
        (0..n)
            .map(|i| TrusteePublicKey::generate(Some(format!("Test-{}", i))).unwrap())
            .collect()
    }

    #[test]
    fn test_protect_and_reconstruct() {
        let config = TemporalShieldConfig::custom(2, 3, 100).unwrap();
        let shield = TemporalShield::new(config);

        let trustees = generate_test_trustees(3);
        let public_keys: Vec<_> = trustees.iter().map(|t| t.public_key.clone()).collect();

        let message = b"Test message for TemporalShield!";
        let envelope = shield.protect(message, &public_keys).unwrap();

        // Verify envelope structure
        assert!(envelope.check_structure());
        assert_eq!(envelope.ciphertext.len(), message.len());
        assert_eq!(envelope.encrypted_shares.len(), 3);

        // Decrypt shares from 2 trustees
        let decrypted_shares: Vec<_> = trustees[0..2]
            .iter()
            .enumerate()
            .map(|(idx, t)| {
                let encrypted = &envelope.encrypted_shares[idx];
                let share = t.private_key.decrypt_share(encrypted).unwrap();
                (idx, share)
            })
            .collect();

        // Reconstruct
        let recovered = shield.reconstruct(&envelope, &decrypted_shares).unwrap();
        assert_eq!(recovered, message);
    }

    #[test]
    fn test_insufficient_shares() {
        let config = TemporalShieldConfig::custom(3, 5, 100).unwrap();
        let shield = TemporalShield::new(config);

        let trustees = generate_test_trustees(5);
        let public_keys: Vec<_> = trustees.iter().map(|t| t.public_key.clone()).collect();

        let message = b"Secret message";
        let envelope = shield.protect(message, &public_keys).unwrap();

        // Only decrypt 2 shares (need 3)
        let decrypted_shares: Vec<_> = trustees[0..2]
            .iter()
            .enumerate()
            .map(|(idx, t)| {
                let encrypted = &envelope.encrypted_shares[idx];
                let share = t.private_key.decrypt_share(encrypted).unwrap();
                (idx, share)
            })
            .collect();

        // Should fail
        let result = shield.reconstruct(&envelope, &decrypted_shares);
        assert!(result.is_err());
    }

    #[test]
    fn test_trustee_count_mismatch() {
        let config = TemporalShieldConfig::custom(2, 5, 100).unwrap();
        let shield = TemporalShield::new(config);

        let trustees = generate_test_trustees(3); // Wrong count
        let public_keys: Vec<_> = trustees.iter().map(|t| t.public_key.clone()).collect();

        let result = shield.protect(b"test", &public_keys);
        assert!(result.is_err());
    }

    #[test]
    fn test_envelope_serialization() {
        let config = TemporalShieldConfig::default();
        let shield = TemporalShield::new(config.clone());

        let trustees = generate_test_trustees(5);
        let public_keys: Vec<_> = trustees.iter().map(|t| t.public_key.clone()).collect();

        let message = b"Test serialization";
        let envelope = shield.protect(message, &public_keys).unwrap();

        // Serialize and deserialize
        let bytes = envelope.to_bytes().unwrap();
        let restored = TemporalEnvelope::from_bytes(&bytes).unwrap();

        assert_eq!(envelope.ciphertext, restored.ciphertext);
        assert_eq!(envelope.key_commitment, restored.key_commitment);
        assert_eq!(envelope.metadata.threshold, restored.metadata.threshold);
    }

    #[test]
    fn test_large_message() {
        let config = TemporalShieldConfig::default();
        let shield = TemporalShield::new(config);

        let trustees = generate_test_trustees(5);
        let public_keys: Vec<_> = trustees.iter().map(|t| t.public_key.clone()).collect();

        // Large message (10KB)
        let message: Vec<u8> = (0..10240).map(|i| (i % 256) as u8).collect();

        let envelope = shield.protect(&message, &public_keys).unwrap();

        // Decrypt with 3 trustees
        let decrypted_shares: Vec<_> = trustees[0..3]
            .iter()
            .enumerate()
            .map(|(idx, t)| {
                let encrypted = &envelope.encrypted_shares[idx];
                let share = t.private_key.decrypt_share(encrypted).unwrap();
                (idx, share)
            })
            .collect();

        let recovered = shield.reconstruct(&envelope, &decrypted_shares).unwrap();
        assert_eq!(recovered, message);
    }
}
