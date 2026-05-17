//! Authenticated Encryption with Associated Data (AEAD)
//!
//! Uses XChaCha20-Poly1305 for symmetric encryption.

use chacha20poly1305::{
    XChaCha20Poly1305,
    aead::{Aead, KeyInit, Payload},
};
use crate::error::{TemporalError, TemporalResult};

/// Encrypt data using XChaCha20-Poly1305
pub fn encrypt(key: &[u8; 32], nonce: &[u8; 24], plaintext: &[u8]) -> TemporalResult<Vec<u8>> {
    let cipher = XChaCha20Poly1305::new(key.into());

    cipher
        .encrypt(nonce.into(), plaintext)
        .map_err(|_| TemporalError::EncryptionFailed("XChaCha20-Poly1305 encryption failed".to_string()))
}

/// Decrypt data using XChaCha20-Poly1305
pub fn decrypt(key: &[u8; 32], nonce: &[u8; 24], ciphertext: &[u8]) -> TemporalResult<Vec<u8>> {
    let cipher = XChaCha20Poly1305::new(key.into());

    cipher
        .decrypt(nonce.into(), ciphertext)
        .map_err(|_| TemporalError::DecryptionFailed("XChaCha20-Poly1305 decryption failed (authentication failed)".to_string()))
}

/// Encrypt with associated data (AAD)
pub fn encrypt_with_aad(
    key: &[u8; 32],
    nonce: &[u8; 24],
    plaintext: &[u8],
    aad: &[u8],
) -> TemporalResult<Vec<u8>> {
    let cipher = XChaCha20Poly1305::new(key.into());

    let payload = Payload {
        msg: plaintext,
        aad,
    };

    cipher
        .encrypt(nonce.into(), payload)
        .map_err(|_| TemporalError::EncryptionFailed("XChaCha20-Poly1305 encryption with AAD failed".to_string()))
}

/// Decrypt with associated data (AAD)
pub fn decrypt_with_aad(
    key: &[u8; 32],
    nonce: &[u8; 24],
    ciphertext: &[u8],
    aad: &[u8],
) -> TemporalResult<Vec<u8>> {
    let cipher = XChaCha20Poly1305::new(key.into());

    let payload = Payload {
        msg: ciphertext,
        aad,
    };

    cipher
        .decrypt(nonce.into(), payload)
        .map_err(|_| TemporalError::DecryptionFailed("XChaCha20-Poly1305 decryption with AAD failed".to_string()))
}

/// Get the authentication tag size
pub const TAG_SIZE: usize = 16;

/// Get the nonce size
pub const NONCE_SIZE: usize = 24;

/// Get the key size
pub const KEY_SIZE: usize = 32;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encrypt_decrypt() {
        let key = [42u8; 32];
        let nonce = [1u8; 24];
        let plaintext = b"Hello, World!";

        let ciphertext = encrypt(&key, &nonce, plaintext).unwrap();
        assert_ne!(ciphertext.as_slice(), plaintext);

        let decrypted = decrypt(&key, &nonce, &ciphertext).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_decrypt_wrong_key() {
        let key = [42u8; 32];
        let wrong_key = [43u8; 32];
        let nonce = [1u8; 24];
        let plaintext = b"Hello, World!";

        let ciphertext = encrypt(&key, &nonce, plaintext).unwrap();
        let result = decrypt(&wrong_key, &nonce, &ciphertext);

        assert!(result.is_err());
    }

    #[test]
    fn test_decrypt_wrong_nonce() {
        let key = [42u8; 32];
        let nonce = [1u8; 24];
        let wrong_nonce = [2u8; 24];
        let plaintext = b"Hello, World!";

        let ciphertext = encrypt(&key, &nonce, plaintext).unwrap();
        let result = decrypt(&key, &wrong_nonce, &ciphertext);

        assert!(result.is_err());
    }

    #[test]
    fn test_encrypt_decrypt_with_aad() {
        let key = [42u8; 32];
        let nonce = [1u8; 24];
        let plaintext = b"Secret message";
        let aad = b"public metadata";

        let ciphertext = encrypt_with_aad(&key, &nonce, plaintext, aad).unwrap();
        let decrypted = decrypt_with_aad(&key, &nonce, &ciphertext, aad).unwrap();

        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_decrypt_wrong_aad() {
        let key = [42u8; 32];
        let nonce = [1u8; 24];
        let plaintext = b"Secret message";
        let aad = b"public metadata";
        let wrong_aad = b"wrong metadata";

        let ciphertext = encrypt_with_aad(&key, &nonce, plaintext, aad).unwrap();
        let result = decrypt_with_aad(&key, &nonce, &ciphertext, wrong_aad);

        assert!(result.is_err());
    }

    #[test]
    fn test_ciphertext_size() {
        let key = [42u8; 32];
        let nonce = [1u8; 24];
        let plaintext = b"Hello";

        let ciphertext = encrypt(&key, &nonce, plaintext).unwrap();
        // Ciphertext = plaintext + tag (16 bytes)
        assert_eq!(ciphertext.len(), plaintext.len() + TAG_SIZE);
    }
}
