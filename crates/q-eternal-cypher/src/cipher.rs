//! Unified authenticated encryption engine.
//!
//! This module provides a single entry point for all symmetric encryption in
//! Q-NarwhalKnight.  The primary cipher is **AEGIS-256** (IACR 2024/268),
//! which runs 2-5x faster than AES-GCM on hardware with AES-NI.
//!
//! ## Cipher Selection by Phase
//!
//! | Phase | Default Cipher | Rationale |
//! |-------|----------------|-----------|
//! | Phase 0 (Genesis) | AEGIS-256 | Best available AEAD |
//! | Phase 1 (Hybrid) | AEGIS-256 | Same |
//! | Phase 2+ (PQ) | AEGIS-256 | Symmetric ciphers are already quantum-safe |
//!
//! Symmetric ciphers are not affected by quantum computers (Grover's algorithm
//! only halves effective security, and 256-bit keys provide 128-bit PQ security),
//! so AEGIS-256 is used in all phases.
//!
//! ## Wire Format
//!
//! Encrypted payloads use the following self-describing envelope:
//!
//! ```text
//! [ 1 byte: cipher_id ] [ 32 bytes: nonce ] [ N bytes: ciphertext + tag ]
//! ```
//!
//! This allows any node to decrypt without knowing which cipher was used.

use crate::EternalCypherError;
use serde::{Deserialize, Serialize};

// Re-export the underlying AEGIS types for callers that need low-level access.
pub use q_crypto_advanced::aegis::{Aegis256, AegisKey, AegisNonce};

// ---------------------------------------------------------------------------
// Cipher identifiers
// ---------------------------------------------------------------------------

/// Identifies the AEAD cipher used for an encrypted payload.
///
/// Stored as a single byte at the start of the wire-format envelope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum CipherId {
    /// AEGIS-256 (primary, hardware-accelerated via AES-NI).
    Aegis256 = 0x01,
}

impl CipherId {
    /// Parse a cipher ID from a byte.
    pub fn from_byte(b: u8) -> Result<Self, EternalCypherError> {
        match b {
            0x01 => Ok(CipherId::Aegis256),
            _ => Err(EternalCypherError::AlgorithmNotAvailable {
                height: 0,
                reason: format!("unknown cipher id: 0x{:02x}", b),
            }),
        }
    }

    /// Return the human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            CipherId::Aegis256 => "AEGIS-256",
        }
    }
}

// ---------------------------------------------------------------------------
// Sealed envelope
// ---------------------------------------------------------------------------

/// A self-describing encrypted envelope that includes all metadata needed
/// for decryption (cipher id, nonce) alongside the ciphertext.
///
/// This is the type produced by [`CipherEngine::seal`] and consumed by
/// [`CipherEngine::open`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SealedEnvelope {
    /// Which cipher produced this envelope.
    pub cipher: CipherId,
    /// The nonce used (32 bytes for AEGIS-256).
    pub nonce: Vec<u8>,
    /// Ciphertext including the authentication tag appended.
    pub ciphertext: Vec<u8>,
}

impl SealedEnvelope {
    /// Serialize to the compact wire format: `[cipher_id | nonce | ciphertext]`.
    pub fn to_wire(&self) -> Vec<u8> {
        let mut wire = Vec::with_capacity(1 + self.nonce.len() + self.ciphertext.len());
        wire.push(self.cipher as u8);
        wire.extend_from_slice(&self.nonce);
        wire.extend_from_slice(&self.ciphertext);
        wire
    }

    /// Parse from the compact wire format.
    pub fn from_wire(data: &[u8]) -> Result<Self, EternalCypherError> {
        if data.len() < 1 + 32 {
            return Err(EternalCypherError::SerializationError(
                "envelope too short".into(),
            ));
        }
        let cipher = CipherId::from_byte(data[0])?;
        let nonce_len = match cipher {
            CipherId::Aegis256 => 32,
        };
        if data.len() < 1 + nonce_len {
            return Err(EternalCypherError::SerializationError(
                "envelope missing nonce".into(),
            ));
        }
        let nonce = data[1..1 + nonce_len].to_vec();
        let ciphertext = data[1 + nonce_len..].to_vec();
        Ok(Self {
            cipher,
            nonce,
            ciphertext,
        })
    }
}

// ---------------------------------------------------------------------------
// Cipher engine
// ---------------------------------------------------------------------------

/// The unified cipher engine.
///
/// Wraps a 256-bit symmetric key and provides `seal` / `open` methods that
/// produce self-describing [`SealedEnvelope`] values.
///
/// # Key Management
///
/// The key should be derived from a [`CrystalSeed`](crate::CrystalSeed)
/// using the domain `"qnk-storage-v1"` (for on-disk encryption) or an
/// appropriate per-session domain for network traffic.
pub struct CipherEngine {
    key: AegisKey,
    cipher: CipherId,
}

impl CipherEngine {
    /// Create a new cipher engine with the given key.
    ///
    /// Defaults to AEGIS-256.
    pub fn new(key: AegisKey) -> Self {
        Self {
            key,
            cipher: CipherId::Aegis256,
        }
    }

    /// Create a cipher engine from raw 32-byte key material.
    pub fn from_raw_key(key_bytes: &[u8; 32]) -> Self {
        Self::new(AegisKey::new(*key_bytes))
    }

    /// Derive a cipher engine from a password and salt using Argon2id + HKDF.
    pub fn from_password(password: &[u8], salt: &[u8]) -> Result<Self, EternalCypherError> {
        let key = AegisKey::derive_from_password(password, salt)
            .map_err(|e| EternalCypherError::KeyError(e.to_string()))?;
        Ok(Self::new(key))
    }

    /// Derive a cipher engine from a [`CrystalSeed`](crate::CrystalSeed).
    pub fn from_seed(seed: &crate::CrystalSeed, domain: &str) -> Self {
        let raw = seed.derive_raw(domain);
        Self::from_raw_key(&raw)
    }

    /// Return the cipher identifier.
    pub fn cipher_id(&self) -> CipherId {
        self.cipher
    }

    /// Encrypt `plaintext` with `associated_data` and return a sealed envelope.
    ///
    /// A fresh random nonce is generated for each call.
    pub fn seal(
        &self,
        plaintext: &[u8],
        associated_data: &[u8],
    ) -> Result<SealedEnvelope, EternalCypherError> {
        match self.cipher {
            CipherId::Aegis256 => {
                let nonce = AegisNonce::generate();
                let ciphertext = Aegis256::encrypt(&self.key, &nonce, plaintext, associated_data)
                    .map_err(|e| EternalCypherError::SigningFailed(e.to_string()))?;
                Ok(SealedEnvelope {
                    cipher: CipherId::Aegis256,
                    nonce: nonce.as_bytes().to_vec(),
                    ciphertext,
                })
            }
        }
    }

    /// Decrypt a [`SealedEnvelope`] and return the plaintext.
    pub fn open(
        &self,
        envelope: &SealedEnvelope,
        associated_data: &[u8],
    ) -> Result<Vec<u8>, EternalCypherError> {
        match envelope.cipher {
            CipherId::Aegis256 => {
                let nonce = AegisNonce::from_slice(&envelope.nonce)
                    .map_err(|e| EternalCypherError::KeyError(e.to_string()))?;
                Aegis256::decrypt(&self.key, &nonce, &envelope.ciphertext, associated_data)
                    .map_err(|e| EternalCypherError::VerificationFailed(e.to_string()))
            }
        }
    }

    /// Convenience: encrypt and return the compact wire-format bytes.
    pub fn encrypt_to_wire(
        &self,
        plaintext: &[u8],
        associated_data: &[u8],
    ) -> Result<Vec<u8>, EternalCypherError> {
        self.seal(plaintext, associated_data).map(|e| e.to_wire())
    }

    /// Convenience: decrypt from compact wire-format bytes.
    pub fn decrypt_from_wire(
        &self,
        wire: &[u8],
        associated_data: &[u8],
    ) -> Result<Vec<u8>, EternalCypherError> {
        let envelope = SealedEnvelope::from_wire(wire)?;
        self.open(&envelope, associated_data)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seal_open_roundtrip() {
        let key = AegisKey::generate();
        let engine = CipherEngine::new(key);

        let plaintext = b"The Eternal Cypher bears witness to every transaction.";
        let aad = b"block-42";

        let envelope = engine.seal(plaintext, aad).unwrap();
        let recovered = engine.open(&envelope, aad).unwrap();
        assert_eq!(recovered, plaintext);
    }

    #[test]
    fn test_wire_format_roundtrip() {
        let engine = CipherEngine::from_raw_key(&[0xAA; 32]);
        let plaintext = b"compact wire format test";
        let aad = b"";

        let wire = engine.encrypt_to_wire(plaintext, aad).unwrap();
        assert_eq!(wire[0], CipherId::Aegis256 as u8);

        let recovered = engine.decrypt_from_wire(&wire, aad).unwrap();
        assert_eq!(recovered, plaintext);
    }

    #[test]
    fn test_sealed_envelope_from_wire_too_short() {
        let result = SealedEnvelope::from_wire(&[0x01; 10]);
        assert!(result.is_err());
    }

    #[test]
    fn test_wrong_key_fails() {
        let engine1 = CipherEngine::from_raw_key(&[0x11; 32]);
        let engine2 = CipherEngine::from_raw_key(&[0x22; 32]);

        let envelope = engine1.seal(b"secret", b"").unwrap();
        let result = engine2.open(&envelope, b"");
        assert!(result.is_err());
    }

    #[test]
    fn test_wrong_aad_fails() {
        let engine = CipherEngine::from_raw_key(&[0x33; 32]);
        let envelope = engine.seal(b"data", b"correct-aad").unwrap();
        let result = engine.open(&envelope, b"wrong-aad");
        assert!(result.is_err());
    }

    #[test]
    fn test_cipher_id_roundtrip() {
        assert_eq!(CipherId::from_byte(0x01).unwrap(), CipherId::Aegis256);
        assert!(CipherId::from_byte(0xFF).is_err());
    }

    #[test]
    fn test_from_seed() {
        let seed = crate::CrystalSeed::generate();
        let engine1 = CipherEngine::from_seed(&seed, "test/encryption");
        let engine2 = CipherEngine::from_seed(&seed, "test/encryption");

        // Same seed + domain should produce same key
        let envelope = engine1.seal(b"hello", b"").unwrap();
        let recovered = engine2.open(&envelope, b"").unwrap();
        assert_eq!(recovered, b"hello");
    }

    #[test]
    fn test_different_domains_incompatible() {
        let seed = crate::CrystalSeed::generate();
        let engine_a = CipherEngine::from_seed(&seed, "domain/a");
        let engine_b = CipherEngine::from_seed(&seed, "domain/b");

        let envelope = engine_a.seal(b"data", b"").unwrap();
        assert!(engine_b.open(&envelope, b"").is_err());
    }

    #[test]
    fn test_empty_plaintext() {
        let engine = CipherEngine::from_raw_key(&[0x44; 32]);
        let envelope = engine.seal(b"", b"").unwrap();
        let recovered = engine.open(&envelope, b"").unwrap();
        assert!(recovered.is_empty());
    }

    #[test]
    fn test_large_data_1mb() {
        let engine = CipherEngine::from_raw_key(&[0x55; 32]);
        let plaintext: Vec<u8> = (0..1_000_000).map(|i| (i % 256) as u8).collect();

        let envelope = engine.seal(&plaintext, b"bulk").unwrap();
        let recovered = engine.open(&envelope, b"bulk").unwrap();
        assert_eq!(recovered, plaintext);
    }
}
