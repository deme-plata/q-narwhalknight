//! Cryptographic primitives for TemporalShield
//!
//! Wraps post-quantum and symmetric cryptography.

pub mod hash;
pub mod kem;
pub mod aead;
pub mod rand;

use serde::{Deserialize, Serialize};

/// An encrypted share for a trustee
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedShare {
    /// Trustee identifier (hash of public key)
    pub trustee_id: [u8; 32],

    /// Share index (1-indexed)
    pub index: u32,

    /// XChaCha20-Poly1305 encrypted share data
    pub encrypted_data: Vec<u8>,

    /// ML-KEM ciphertext (for key encapsulation)
    pub kem_ciphertext: Vec<u8>,

    /// Nonce for AEAD
    pub nonce: [u8; 24],
}

impl EncryptedShare {
    /// Get the size in bytes
    pub fn size(&self) -> usize {
        32 + 4 + self.encrypted_data.len() + self.kem_ciphertext.len() + 24
    }
}
