/// Kyber1024 Post-Quantum Key Encapsulation Module
/// Phase 6: NIST Level 5 post-quantum key exchange for wallet encryption
///
/// This module provides Kyber1024 KEM for quantum-resistant encryption of wallet secrets

use anyhow::{anyhow, Result};
use pqcrypto_kyber::kyber1024;
use pqcrypto_traits::kem::{Ciphertext as KemCiphertext, PublicKey as KemPublicKey, SecretKey as KemSecretKey, SharedSecret as KemSharedSecret};

/// Kyber1024 keypair for quantum-resistant key exchange
pub struct Kyber1024KeyPair {
    pub public_key: kyber1024::PublicKey,
    pub secret_key: kyber1024::SecretKey,
}

/// Kyber1024 encapsulated ciphertext
pub struct Kyber1024Ciphertext {
    pub ciphertext: kyber1024::Ciphertext,
}

impl Kyber1024KeyPair {
    /// Generate a new Kyber1024 keypair using quantum random number generation
    pub fn generate() -> Self {
        let (public_key, secret_key) = kyber1024::keypair();
        Self {
            public_key,
            secret_key,
        }
    }

    /// Encapsulate a shared secret using recipient's public key
    /// Returns (shared_secret, ciphertext)
    pub fn encapsulate(public_key: &kyber1024::PublicKey) -> (Vec<u8>, Vec<u8>) {
        let (shared_secret, ciphertext) = kyber1024::encapsulate(public_key);
        (
            shared_secret.as_bytes().to_vec(),
            ciphertext.as_bytes().to_vec(),
        )
    }

    /// Decapsulate shared secret from ciphertext using secret key
    pub fn decapsulate(&self, ciphertext: &[u8]) -> Result<Vec<u8>> {
        let ct = kyber1024::Ciphertext::from_bytes(ciphertext)
            .map_err(|_| anyhow!("Invalid Kyber1024 ciphertext"))?;

        let shared_secret = kyber1024::decapsulate(&ct, &self.secret_key);
        Ok(shared_secret.as_bytes().to_vec())
    }

    /// Get public key bytes
    pub fn public_key_bytes(&self) -> Vec<u8> {
        self.public_key.as_bytes().to_vec()
    }

    /// Get secret key bytes
    pub fn secret_key_bytes(&self) -> Vec<u8> {
        self.secret_key.as_bytes().to_vec()
    }

    /// Reconstruct keypair from bytes
    pub fn from_bytes(public_key: &[u8], secret_key: &[u8]) -> Result<Self> {
        let pk = kyber1024::PublicKey::from_bytes(public_key)
            .map_err(|_| anyhow!("Invalid Kyber1024 public key"))?;
        let sk = kyber1024::SecretKey::from_bytes(secret_key)
            .map_err(|_| anyhow!("Invalid Kyber1024 secret key"))?;

        Ok(Self {
            public_key: pk,
            secret_key: sk,
        })
    }
}

/// Hybrid encryption: Kyber1024 for key exchange + AES-256-GCM for data
pub struct KyberHybridEncryption;

impl KyberHybridEncryption {
    /// Encrypt data using recipient's Kyber public key
    /// Returns (ciphertext, kyber_ciphertext)
    pub fn encrypt(plaintext: &[u8], recipient_public_key: &[u8]) -> Result<(Vec<u8>, Vec<u8>)> {
        use aes_gcm::{
            aead::{Aead, KeyInit},
            Aes256Gcm, Nonce,
        };

        // Reconstruct recipient's public key
        let pk = kyber1024::PublicKey::from_bytes(recipient_public_key)
            .map_err(|_| anyhow!("Invalid recipient public key"))?;

        // Encapsulate shared secret
        let (shared_secret, kyber_ciphertext) = Kyber1024KeyPair::encapsulate(&pk);

        // Use first 32 bytes of shared secret as AES-256 key
        let mut aes_key = [0u8; 32];
        aes_key.copy_from_slice(&shared_secret[..32]);

        // Generate random nonce for AES-GCM
        use rand::RngCore;
        let mut nonce = [0u8; 12];
        rand::rngs::OsRng.fill_bytes(&mut nonce);

        // Encrypt plaintext with AES-256-GCM
        let cipher = Aes256Gcm::new(&aes_key.into());
        let nonce_obj = Nonce::from_slice(&nonce);

        let mut ciphertext = cipher
            .encrypt(nonce_obj, plaintext)
            .map_err(|e| anyhow!("AES encryption failed: {}", e))?;

        // Prepend nonce to ciphertext
        let mut result = nonce.to_vec();
        result.append(&mut ciphertext);

        Ok((result, kyber_ciphertext))
    }

    /// Decrypt data using Kyber secret key
    pub fn decrypt(
        ciphertext_with_nonce: &[u8],
        kyber_ciphertext: &[u8],
        secret_key: &[u8],
    ) -> Result<Vec<u8>> {
        use aes_gcm::{
            aead::{Aead, KeyInit},
            Aes256Gcm, Nonce,
        };

        // Reconstruct secret key
        let sk = kyber1024::SecretKey::from_bytes(secret_key)
            .map_err(|_| anyhow!("Invalid secret key"))?;

        // Decapsulate shared secret
        let ct = kyber1024::Ciphertext::from_bytes(kyber_ciphertext)
            .map_err(|_| anyhow!("Invalid Kyber ciphertext"))?;

        let shared_secret = kyber1024::decapsulate(&ct, &sk);

        // Use first 32 bytes as AES-256 key
        let mut aes_key = [0u8; 32];
        aes_key.copy_from_slice(&shared_secret.as_bytes()[..32]);

        // Extract nonce and ciphertext
        if ciphertext_with_nonce.len() < 12 {
            return Err(anyhow!("Ciphertext too short"));
        }

        let nonce = &ciphertext_with_nonce[..12];
        let ciphertext = &ciphertext_with_nonce[12..];

        // Decrypt with AES-256-GCM
        let cipher = Aes256Gcm::new(&aes_key.into());
        let nonce_obj = Nonce::from_slice(nonce);

        let plaintext = cipher
            .decrypt(nonce_obj, ciphertext)
            .map_err(|e| anyhow!("AES decryption failed: {}", e))?;

        Ok(plaintext)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kyber1024_keypair_generation() {
        let keypair = Kyber1024KeyPair::generate();
        assert_eq!(
            keypair.public_key.as_bytes().len(),
            kyber1024::public_key_bytes()
        );
        assert_eq!(
            keypair.secret_key.as_bytes().len(),
            kyber1024::secret_key_bytes()
        );
    }

    #[test]
    fn test_kyber1024_encapsulation_decapsulation() {
        let keypair = Kyber1024KeyPair::generate();

        // Encapsulate
        let (shared_secret1, ciphertext) =
            Kyber1024KeyPair::encapsulate(&keypair.public_key);

        // Decapsulate
        let shared_secret2 = keypair
            .decapsulate(&ciphertext)
            .expect("Decapsulation failed");

        assert_eq!(
            shared_secret1, shared_secret2,
            "Shared secrets should match"
        );
        assert_eq!(shared_secret1.len(), 32, "Shared secret should be 32 bytes");
    }

    #[test]
    fn test_kyber_hybrid_encryption() {
        let keypair = Kyber1024KeyPair::generate();
        let message = b"Quantum-resistant encrypted wallet data";

        // Encrypt
        let (ciphertext, kyber_ct) = KyberHybridEncryption::encrypt(
            message,
            keypair.public_key.as_bytes(),
        )
        .expect("Encryption failed");

        // Decrypt
        let plaintext = KyberHybridEncryption::decrypt(
            &ciphertext,
            &kyber_ct,
            keypair.secret_key.as_bytes(),
        )
        .expect("Decryption failed");

        assert_eq!(plaintext, message, "Decrypted message should match original");
    }

    #[test]
    fn test_kyber_keypair_serialization() {
        let keypair = Kyber1024KeyPair::generate();

        let pk_bytes = keypair.public_key_bytes();
        let sk_bytes = keypair.secret_key_bytes();

        // Reconstruct keypair
        let restored = Kyber1024KeyPair::from_bytes(&pk_bytes, &sk_bytes)
            .expect("Keypair restoration failed");

        // Test that restored keypair works
        let (ss1, ct) = Kyber1024KeyPair::encapsulate(&restored.public_key);
        let ss2 = restored.decapsulate(&ct).expect("Decapsulation failed");

        assert_eq!(ss1, ss2, "Restored keypair should work correctly");
    }

    #[test]
    fn test_kyber_wrong_key_decryption_fails() {
        let keypair1 = Kyber1024KeyPair::generate();
        let keypair2 = Kyber1024KeyPair::generate();
        let message = b"Secret message";

        // Encrypt with keypair1's public key
        let (ciphertext, kyber_ct) = KyberHybridEncryption::encrypt(
            message,
            keypair1.public_key.as_bytes(),
        )
        .expect("Encryption failed");

        // Try to decrypt with keypair2's secret key (should fail or produce garbage)
        let result = KyberHybridEncryption::decrypt(
            &ciphertext,
            &kyber_ct,
            keypair2.secret_key.as_bytes(),
        );

        // Either decryption fails or produces wrong data
        match result {
            Err(_) => { /* Expected - decryption failed */ }
            Ok(plaintext) => {
                assert_ne!(plaintext, message, "Wrong key should not decrypt correctly");
            }
        }
    }

    #[test]
    fn test_kyber_ciphertext_size() {
        let keypair = Kyber1024KeyPair::generate();
        let message = b"Test";

        let (ciphertext, kyber_ct) = KyberHybridEncryption::encrypt(
            message,
            keypair.public_key.as_bytes(),
        )
        .expect("Encryption failed");

        // Kyber1024 ciphertext is 1568 bytes
        assert_eq!(kyber_ct.len(), kyber1024::ciphertext_bytes());

        // AES ciphertext = 12 (nonce) + message_len + 16 (GCM tag)
        assert!(ciphertext.len() >= 12 + message.len() + 16);
    }
}
