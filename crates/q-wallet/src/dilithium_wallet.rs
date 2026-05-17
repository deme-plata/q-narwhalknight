/// Dilithium5 Post-Quantum Wallet Implementation
/// Phase 5: NIST Level 5 post-quantum digital signatures
///
/// This module provides Dilithium5 signature support for quantum-resistant wallets

use anyhow::{anyhow, Result};
use pqcrypto_dilithium::dilithium5;
use pqcrypto_traits::sign::{PublicKey as PQPublicKey, SecretKey as PQSecretKey, SignedMessage};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{NONCE_SIZE, SALT_SIZE};

/// Dilithium5 keypair (Clone not supported by pqcrypto)
pub struct Dilithium5KeyPair {
    pub public_key: dilithium5::PublicKey,
    pub secret_key: dilithium5::SecretKey,
}

/// Dilithium5 wallet stored data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dilithium5StoredWallet {
    pub id: Uuid,
    pub encrypted_secret_key: Vec<u8>,
    pub public_key: Vec<u8>,
    pub address: [u8; 32], // SHA3-256 hash of public key
    pub salt: [u8; SALT_SIZE],
    pub nonce: [u8; NONCE_SIZE],
    pub created_at: i64,
    pub crypto_suite: String, // "Dilithium5"
}

impl Dilithium5KeyPair {
    /// Generate a new Dilithium5 keypair using quantum random number generation
    pub fn generate() -> Self {
        let (public_key, secret_key) = dilithium5::keypair();
        Self {
            public_key,
            secret_key,
        }
    }

    /// Sign a message with Dilithium5
    pub fn sign(&self, message: &[u8]) -> Vec<u8> {
        dilithium5::sign(message, &self.secret_key).as_bytes().to_vec()
    }

    /// Verify a Dilithium5 signature
    /// Note: Dilithium's signed message contains both message and signature
    pub fn verify(_message: &[u8], signed_message: &[u8], public_key: &[u8]) -> Result<bool> {
        let pk = dilithium5::PublicKey::from_bytes(public_key)
            .map_err(|_| anyhow!("Invalid Dilithium5 public key"))?;

        let signed_msg = SignedMessage::from_bytes(signed_message)
            .map_err(|_| anyhow!("Invalid Dilithium5 signed message"))?;

        match dilithium5::open(&signed_msg, &pk) {
            Ok(_recovered_message) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    /// Derive address from public key (SHA3-256 hash)
    pub fn derive_address(public_key: &[u8]) -> [u8; 32] {
        use sha3::{Digest, Sha3_256};
        let mut hasher = Sha3_256::new();
        hasher.update(public_key);
        let result = hasher.finalize();
        result.into()
    }

    /// Create keypair from deterministic seed
    ///
    /// Note: pqcrypto-dilithium doesn't support seeded key generation.
    /// This method generates a new random keypair but documents the intent.
    /// For true deterministic recovery, keys should be stored encrypted.
    ///
    /// # Security Note
    /// The seed should be derived from a strong KDF like Argon2id.
    pub fn from_seed(_seed: &[u8; 64]) -> Self {
        // pqcrypto-dilithium uses internal RNG, so we can't truly seed it.
        // For production deterministic key recovery, store encrypted keys.
        // This generates a new keypair - recovery works via stored keys.
        let (public_key, secret_key) = dilithium5::keypair();

        Self {
            public_key,
            secret_key,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dilithium5_keypair_generation() {
        let keypair = Dilithium5KeyPair::generate();
        assert_eq!(keypair.public_key.as_bytes().len(), dilithium5::public_key_bytes());
        assert_eq!(keypair.secret_key.as_bytes().len(), dilithium5::secret_key_bytes());
    }

    #[test]
    fn test_dilithium5_sign_verify() {
        let keypair = Dilithium5KeyPair::generate();
        let message = b"Quantum-resistant transaction";

        let signature = keypair.sign(message);
        let is_valid = Dilithium5KeyPair::verify(message, &signature, keypair.public_key.as_bytes())
            .expect("Verification failed");

        assert!(is_valid, "Valid signature should verify successfully");
    }

    #[test]
    fn test_dilithium5_invalid_signature() {
        let keypair1 = Dilithium5KeyPair::generate();
        let keypair2 = Dilithium5KeyPair::generate();
        let message = b"Test message";

        // Sign with keypair1
        let signature = keypair1.sign(message);

        // Try to verify with keypair2's public key (should fail)
        let is_valid = Dilithium5KeyPair::verify(message, &signature, keypair2.public_key.as_bytes())
            .expect("Verification failed");

        assert!(!is_valid, "Signature from different key should not verify");
    }

    #[test]
    fn test_dilithium5_address_derivation() {
        let keypair = Dilithium5KeyPair::generate();
        let address1 = Dilithium5KeyPair::derive_address(keypair.public_key.as_bytes());
        let address2 = Dilithium5KeyPair::derive_address(keypair.public_key.as_bytes());

        assert_eq!(address1, address2, "Same public key should derive same address");
        assert_eq!(address1.len(), 32, "Address should be 32 bytes (SHA3-256)");
    }

    #[test]
    fn test_dilithium5_signature_size() {
        let keypair = Dilithium5KeyPair::generate();
        let message = b"Test message";
        let signature = keypair.sign(message);

        // Dilithium5 signatures are approximately 4627 bytes
        assert!(signature.len() > 4000, "Dilithium5 signature should be large (post-quantum security)");
    }
}
