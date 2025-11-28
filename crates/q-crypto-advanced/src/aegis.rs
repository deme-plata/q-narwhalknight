//! AEGIS-256 Authenticated Encryption (IACR 2024/268)
//!
//! AEGIS is a family of fast authenticated encryption algorithms designed to
//! take advantage of AES-NI instructions. AEGIS-256 provides:
//!
//! - **2-5x faster than AES-GCM** on modern CPUs with AES-NI
//! - **256-bit key** and **256-bit nonce** (no nonce reuse concerns)
//! - **128-bit authentication tag** (configurable up to 256-bit)
//! - **Misuse resistance**: More forgiving of implementation errors

use crate::errors::CryptoError;
use aegis::aegis256::Aegis256 as AegisImpl;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use std::ops::Deref;
use zeroize::{Zeroize, ZeroizeOnDrop};

/// Tag size in bytes (128-bit tag)
pub const TAG_BYTES: usize = 16;

/// AEGIS-256 key (32 bytes)
#[derive(Clone, Zeroize, ZeroizeOnDrop)]
pub struct AegisKey([u8; 32]);

impl AegisKey {
    /// Create a new key from bytes
    pub fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Generate a random key
    pub fn generate() -> Self {
        let mut key = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut key);
        Self(key)
    }

    /// Create from a slice (must be exactly 32 bytes)
    pub fn from_slice(slice: &[u8]) -> Result<Self, CryptoError> {
        if slice.len() != 32 {
            return Err(CryptoError::InvalidKeyLength(slice.len()));
        }
        let mut key = [0u8; 32];
        key.copy_from_slice(slice);
        Ok(Self(key))
    }

    /// Get the key bytes
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Derive a key from a password using Argon2id
    pub fn derive_from_password(password: &[u8], salt: &[u8]) -> Result<Self, CryptoError> {
        use argon2::{Argon2, Algorithm, Version, Params};
        use hkdf::Hkdf;
        use sha2::Sha256;

        // Argon2id parameters (OWASP recommended)
        let params = Params::new(65536, 4, 1, Some(32))
            .map_err(|e| CryptoError::InternalError(e.to_string()))?;

        let argon2 = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);

        // Derive intermediate key
        let mut intermediate = [0u8; 32];
        argon2
            .hash_password_into(password, salt, &mut intermediate)
            .map_err(|e| CryptoError::InternalError(e.to_string()))?;

        // Use HKDF for final key derivation
        let hkdf = Hkdf::<Sha256>::new(Some(salt), &intermediate);
        let mut key = [0u8; 32];
        hkdf.expand(b"aegis-256-key", &mut key)
            .map_err(|_| CryptoError::InternalError("HKDF expansion failed".into()))?;

        // Zeroize intermediate
        intermediate.zeroize();

        Ok(Self(key))
    }
}

impl Deref for AegisKey {
    type Target = [u8; 32];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Serialize for AegisKey {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&hex::encode(&self.0))
    }
}

impl<'de> Deserialize<'de> for AegisKey {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let hex_str = String::deserialize(deserializer)?;
        let bytes = hex::decode(&hex_str).map_err(serde::de::Error::custom)?;
        Self::from_slice(&bytes).map_err(serde::de::Error::custom)
    }
}

/// AEGIS-256 nonce (32 bytes)
#[derive(Clone, Serialize, Deserialize)]
pub struct AegisNonce([u8; 32]);

impl AegisNonce {
    /// Create a new nonce from bytes
    pub fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Generate a random nonce
    pub fn generate() -> Self {
        let mut nonce = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut nonce);
        Self(nonce)
    }

    /// Create from a slice (must be exactly 32 bytes)
    pub fn from_slice(slice: &[u8]) -> Result<Self, CryptoError> {
        if slice.len() != 32 {
            return Err(CryptoError::InvalidNonceLength(slice.len()));
        }
        let mut nonce = [0u8; 32];
        nonce.copy_from_slice(slice);
        Ok(Self(nonce))
    }

    /// Get the nonce bytes
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl Deref for AegisNonce {
    type Target = [u8; 32];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// AEGIS-256 Authenticated Encryption
pub struct Aegis256;

impl Aegis256 {
    /// Authentication tag size (16 bytes = 128 bits)
    pub const TAG_SIZE: usize = TAG_BYTES;

    /// Encrypt data with associated data
    pub fn encrypt(
        key: &AegisKey,
        nonce: &AegisNonce,
        plaintext: &[u8],
        associated_data: &[u8],
    ) -> Result<Vec<u8>, CryptoError> {
        let aegis: AegisImpl<TAG_BYTES> = AegisImpl::new(key.as_bytes(), nonce.as_bytes());

        // Encrypt and get ciphertext + tag
        let (ciphertext, tag) = aegis.encrypt(plaintext, associated_data);

        // Combine ciphertext and tag
        let mut result = ciphertext;
        result.extend_from_slice(&tag);

        Ok(result)
    }

    /// Decrypt data with associated data
    pub fn decrypt(
        key: &AegisKey,
        nonce: &AegisNonce,
        ciphertext: &[u8],
        associated_data: &[u8],
    ) -> Result<Vec<u8>, CryptoError> {
        if ciphertext.len() < Self::TAG_SIZE {
            return Err(CryptoError::CiphertextTooShort);
        }

        let aegis: AegisImpl<TAG_BYTES> = AegisImpl::new(key.as_bytes(), nonce.as_bytes());

        // Split ciphertext and tag
        let (ct, tag_bytes) = ciphertext.split_at(ciphertext.len() - Self::TAG_SIZE);

        // Convert tag bytes to array
        let mut tag = [0u8; TAG_BYTES];
        tag.copy_from_slice(tag_bytes);

        // Decrypt and verify
        aegis
            .decrypt(ct, &tag, associated_data)
            .map_err(|_| CryptoError::DecryptionFailed)
    }

    /// Encrypt in-place (for zero-copy operations)
    pub fn encrypt_in_place(
        key: &AegisKey,
        nonce: &AegisNonce,
        buffer: &mut Vec<u8>,
        associated_data: &[u8],
    ) -> Result<(), CryptoError> {
        let aegis: AegisImpl<TAG_BYTES> = AegisImpl::new(key.as_bytes(), nonce.as_bytes());

        // Create a copy to get ciphertext
        let (ciphertext, tag) = aegis.encrypt(buffer, associated_data);

        // Replace buffer contents
        buffer.clear();
        buffer.extend_from_slice(&ciphertext);
        buffer.extend_from_slice(&tag);

        Ok(())
    }

    /// Decrypt in-place (for zero-copy operations)
    pub fn decrypt_in_place(
        key: &AegisKey,
        nonce: &AegisNonce,
        buffer: &mut Vec<u8>,
        associated_data: &[u8],
    ) -> Result<(), CryptoError> {
        if buffer.len() < Self::TAG_SIZE {
            return Err(CryptoError::CiphertextTooShort);
        }

        let aegis: AegisImpl<TAG_BYTES> = AegisImpl::new(key.as_bytes(), nonce.as_bytes());

        // Split buffer into ciphertext and tag
        let ct_len = buffer.len() - Self::TAG_SIZE;

        // Extract tag
        let mut tag = [0u8; TAG_BYTES];
        tag.copy_from_slice(&buffer[ct_len..]);

        // Decrypt
        let plaintext = aegis
            .decrypt(&buffer[..ct_len], &tag, associated_data)
            .map_err(|_| CryptoError::DecryptionFailed)?;

        // Replace buffer contents
        buffer.clear();
        buffer.extend_from_slice(&plaintext);

        Ok(())
    }
}

/// Streaming encryption context for large data
pub struct AegisStreamEncryptor {
    key: AegisKey,
    nonce_counter: u64,
    base_nonce: [u8; 24],
}

impl AegisStreamEncryptor {
    /// Create a new streaming encryptor
    pub fn new(key: AegisKey) -> Self {
        let mut base_nonce = [0u8; 24];
        rand::thread_rng().fill_bytes(&mut base_nonce);

        Self {
            key,
            nonce_counter: 0,
            base_nonce,
        }
    }

    /// Get the base nonce (needed for decryption)
    pub fn base_nonce(&self) -> &[u8; 24] {
        &self.base_nonce
    }

    /// Encrypt a chunk of data
    pub fn encrypt_chunk(
        &mut self,
        chunk: &[u8],
        associated_data: &[u8],
    ) -> Result<Vec<u8>, CryptoError> {
        // Construct nonce with counter
        let mut nonce_bytes = [0u8; 32];
        nonce_bytes[..24].copy_from_slice(&self.base_nonce);
        nonce_bytes[24..].copy_from_slice(&self.nonce_counter.to_le_bytes());

        let nonce = AegisNonce::new(nonce_bytes);
        self.nonce_counter += 1;

        Aegis256::encrypt(&self.key, &nonce, chunk, associated_data)
    }
}

/// Streaming decryption context
pub struct AegisStreamDecryptor {
    key: AegisKey,
    nonce_counter: u64,
    base_nonce: [u8; 24],
}

impl AegisStreamDecryptor {
    /// Create a new streaming decryptor
    pub fn new(key: AegisKey, base_nonce: [u8; 24]) -> Self {
        Self {
            key,
            nonce_counter: 0,
            base_nonce,
        }
    }

    /// Decrypt a chunk of data
    pub fn decrypt_chunk(
        &mut self,
        ciphertext: &[u8],
        associated_data: &[u8],
    ) -> Result<Vec<u8>, CryptoError> {
        // Reconstruct nonce with counter
        let mut nonce_bytes = [0u8; 32];
        nonce_bytes[..24].copy_from_slice(&self.base_nonce);
        nonce_bytes[24..].copy_from_slice(&self.nonce_counter.to_le_bytes());

        let nonce = AegisNonce::new(nonce_bytes);
        self.nonce_counter += 1;

        Aegis256::decrypt(&self.key, &nonce, ciphertext, associated_data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aegis_basic() {
        let key = AegisKey::generate();
        let nonce = AegisNonce::generate();

        let plaintext = b"Hello, AEGIS-256!";
        let aad = b"metadata";

        let ciphertext = Aegis256::encrypt(&key, &nonce, plaintext, aad).unwrap();
        let decrypted = Aegis256::decrypt(&key, &nonce, &ciphertext, aad).unwrap();

        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_aegis_empty_plaintext() {
        let key = AegisKey::generate();
        let nonce = AegisNonce::generate();

        let plaintext = b"";
        let aad = b"";

        let ciphertext = Aegis256::encrypt(&key, &nonce, plaintext, aad).unwrap();
        let decrypted = Aegis256::decrypt(&key, &nonce, &ciphertext, aad).unwrap();

        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_aegis_wrong_key() {
        let key1 = AegisKey::generate();
        let key2 = AegisKey::generate();
        let nonce = AegisNonce::generate();

        let plaintext = b"Secret data";
        let aad = b"";

        let ciphertext = Aegis256::encrypt(&key1, &nonce, plaintext, aad).unwrap();
        let result = Aegis256::decrypt(&key2, &nonce, &ciphertext, aad);

        assert!(result.is_err());
    }

    #[test]
    fn test_aegis_tampered_ciphertext() {
        let key = AegisKey::generate();
        let nonce = AegisNonce::generate();

        let plaintext = b"Secret data";
        let aad = b"";

        let mut ciphertext = Aegis256::encrypt(&key, &nonce, plaintext, aad).unwrap();

        // Tamper with ciphertext
        if !ciphertext.is_empty() {
            ciphertext[0] ^= 0xFF;
        }

        let result = Aegis256::decrypt(&key, &nonce, &ciphertext, aad);
        assert!(result.is_err());
    }

    #[test]
    fn test_aegis_wrong_aad() {
        let key = AegisKey::generate();
        let nonce = AegisNonce::generate();

        let plaintext = b"Secret data";

        let ciphertext = Aegis256::encrypt(&key, &nonce, plaintext, b"correct").unwrap();
        let result = Aegis256::decrypt(&key, &nonce, &ciphertext, b"wrong");

        assert!(result.is_err());
    }

    #[test]
    fn test_aegis_in_place() {
        let key = AegisKey::generate();
        let nonce = AegisNonce::generate();

        let plaintext = b"Data to encrypt in place".to_vec();
        let aad = b"header";

        let mut buffer = plaintext.clone();
        Aegis256::encrypt_in_place(&key, &nonce, &mut buffer, aad).unwrap();

        assert_ne!(buffer[..plaintext.len()], plaintext[..]);

        Aegis256::decrypt_in_place(&key, &nonce, &mut buffer, aad).unwrap();
        assert_eq!(buffer, plaintext);
    }

    #[test]
    fn test_aegis_streaming() {
        let key = AegisKey::generate();
        let mut encryptor = AegisStreamEncryptor::new(key.clone());

        let chunks = vec![
            b"First chunk of data".to_vec(),
            b"Second chunk".to_vec(),
            b"Third and final chunk!".to_vec(),
        ];

        let aad = b"stream-aad";

        // Encrypt all chunks
        let encrypted: Vec<Vec<u8>> = chunks
            .iter()
            .map(|chunk| encryptor.encrypt_chunk(chunk, aad).unwrap())
            .collect();

        // Decrypt all chunks
        let base_nonce = *encryptor.base_nonce();
        let mut decryptor = AegisStreamDecryptor::new(key, base_nonce);

        for (i, ct) in encrypted.iter().enumerate() {
            let decrypted = decryptor.decrypt_chunk(ct, aad).unwrap();
            assert_eq!(decrypted, chunks[i]);
        }
    }

    #[test]
    fn test_aegis_large_data() {
        let key = AegisKey::generate();
        let nonce = AegisNonce::generate();

        // 1 MB of data
        let plaintext: Vec<u8> = (0..1_000_000).map(|i| (i % 256) as u8).collect();
        let aad = b"large-data-test";

        let ciphertext = Aegis256::encrypt(&key, &nonce, &plaintext, aad).unwrap();
        let decrypted = Aegis256::decrypt(&key, &nonce, &ciphertext, aad).unwrap();

        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_key_derivation() {
        let password = b"my_secure_password";
        let salt = b"random_salt_here_16!";

        let key1 = AegisKey::derive_from_password(password, salt).unwrap();
        let key2 = AegisKey::derive_from_password(password, salt).unwrap();

        // Same password + salt should produce same key
        assert_eq!(key1.as_bytes(), key2.as_bytes());

        // Different salt should produce different key
        let key3 = AegisKey::derive_from_password(password, b"different_salt__!").unwrap();
        assert_ne!(key1.as_bytes(), key3.as_bytes());
    }

    #[test]
    fn test_key_serialization() {
        let key = AegisKey::generate();

        let json = serde_json::to_string(&key).unwrap();
        let key2: AegisKey = serde_json::from_str(&json).unwrap();

        assert_eq!(key.as_bytes(), key2.as_bytes());
    }
}
