/// Crypto-Agile Hybrid Wallet Implementation
/// Phase 7: Supports both classical (Ed25519) and post-quantum (Dilithium5) signatures
///
/// This module provides a hybrid wallet that can operate in three modes:
/// - Q0 (Classical): Ed25519 only
/// - Q1 (Hybrid): Ed25519 + Dilithium5 dual signatures
/// - Q2 (Post-Quantum): Dilithium5 only

use anyhow::{anyhow, Result};
use ed25519_dalek::{Signature as Ed25519Signature, Signer, SigningKey as Ed25519SigningKey, VerifyingKey as Ed25519VerifyingKey};
use pqcrypto_traits::sign::PublicKey as PQPublicKey;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use uuid::Uuid;

use crate::{NONCE_SIZE, SALT_SIZE};
use crate::dilithium_wallet::Dilithium5KeyPair;

/// Cryptographic phase indicating the level of quantum resistance
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CryptoPhase {
    /// Q0: Classical cryptography only (Ed25519)
    Q0,
    /// Q1: Hybrid mode with both classical and post-quantum (Ed25519 + Dilithium5)
    Q1,
    /// Q2: Post-quantum only (Dilithium5)
    Q2,
}

/// Hybrid wallet supporting multiple cryptographic phases
pub struct HybridWallet {
    pub id: Uuid,
    pub phase: CryptoPhase,
    pub ed25519_key: Option<Ed25519SigningKey>,
    pub dilithium5_key: Option<Dilithium5KeyPair>,
}

/// Stored hybrid wallet data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridStoredWallet {
    pub id: Uuid,
    pub phase: CryptoPhase,

    // Ed25519 keys (for Q0 and Q1)
    pub ed25519_public_key: Option<Vec<u8>>,
    pub encrypted_ed25519_secret_key: Option<Vec<u8>>,

    // Dilithium5 keys (for Q1 and Q2)
    pub dilithium5_public_key: Option<Vec<u8>>,
    pub encrypted_dilithium5_secret_key: Option<Vec<u8>>,

    // Primary address (derived from highest security key available)
    pub address: [u8; 32],

    // Encryption metadata
    pub salt: [u8; SALT_SIZE],
    pub nonce: [u8; NONCE_SIZE],
    pub created_at: i64,
}

/// Hybrid signature containing both Ed25519 and Dilithium5 signatures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridSignature {
    pub phase: CryptoPhase,
    pub ed25519_signature: Option<Vec<u8>>,
    pub dilithium5_signature: Option<Vec<u8>>,
}

impl HybridWallet {
    /// Create a new hybrid wallet for the specified cryptographic phase
    pub fn generate(phase: CryptoPhase) -> Self {
        let id = Uuid::new_v4();

        let (ed25519_key, dilithium5_key) = match phase {
            CryptoPhase::Q0 => {
                // Classical only: Ed25519
                let mut secret_bytes = [0u8; 32];
                rand::rngs::OsRng.fill_bytes(&mut secret_bytes);
                let ed25519_key = Ed25519SigningKey::from_bytes(&secret_bytes);
                (Some(ed25519_key), None)
            }
            CryptoPhase::Q1 => {
                // Hybrid: Both Ed25519 and Dilithium5
                let mut secret_bytes = [0u8; 32];
                rand::rngs::OsRng.fill_bytes(&mut secret_bytes);
                let ed25519_key = Ed25519SigningKey::from_bytes(&secret_bytes);
                let dilithium5_key = Dilithium5KeyPair::generate();
                (Some(ed25519_key), Some(dilithium5_key))
            }
            CryptoPhase::Q2 => {
                // Post-quantum only: Dilithium5
                let dilithium5_key = Dilithium5KeyPair::generate();
                (None, Some(dilithium5_key))
            }
        };

        Self {
            id,
            phase,
            ed25519_key,
            dilithium5_key,
        }
    }

    /// Sign a message with the appropriate signature(s) for the wallet's phase
    pub fn sign(&self, message: &[u8]) -> Result<HybridSignature> {
        match self.phase {
            CryptoPhase::Q0 => {
                let ed25519_key = self.ed25519_key.as_ref()
                    .ok_or_else(|| anyhow!("Ed25519 key not available for Q0 wallet"))?;

                let signature = ed25519_key.sign(message);

                Ok(HybridSignature {
                    phase: CryptoPhase::Q0,
                    ed25519_signature: Some(signature.to_bytes().to_vec()),
                    dilithium5_signature: None,
                })
            }
            CryptoPhase::Q1 => {
                // Hybrid mode: Create both Ed25519 and Dilithium5 signatures
                let ed25519_key = self.ed25519_key.as_ref()
                    .ok_or_else(|| anyhow!("Ed25519 key not available for Q1 wallet"))?;
                let dilithium5_key = self.dilithium5_key.as_ref()
                    .ok_or_else(|| anyhow!("Dilithium5 key not available for Q1 wallet"))?;

                let ed25519_sig = ed25519_key.sign(message);
                let dilithium5_sig = dilithium5_key.sign(message);

                Ok(HybridSignature {
                    phase: CryptoPhase::Q1,
                    ed25519_signature: Some(ed25519_sig.to_bytes().to_vec()),
                    dilithium5_signature: Some(dilithium5_sig),
                })
            }
            CryptoPhase::Q2 => {
                let dilithium5_key = self.dilithium5_key.as_ref()
                    .ok_or_else(|| anyhow!("Dilithium5 key not available for Q2 wallet"))?;

                let signature = dilithium5_key.sign(message);

                Ok(HybridSignature {
                    phase: CryptoPhase::Q2,
                    ed25519_signature: None,
                    dilithium5_signature: Some(signature),
                })
            }
        }
    }

    /// Verify a hybrid signature
    pub fn verify(
        message: &[u8],
        signature: &HybridSignature,
        ed25519_public_key: Option<&[u8]>,
        dilithium5_public_key: Option<&[u8]>,
    ) -> Result<bool> {
        match signature.phase {
            CryptoPhase::Q0 => {
                let ed25519_sig = signature.ed25519_signature.as_ref()
                    .ok_or_else(|| anyhow!("Ed25519 signature missing for Q0"))?;
                let ed25519_pk = ed25519_public_key
                    .ok_or_else(|| anyhow!("Ed25519 public key missing for Q0"))?;

                Self::verify_ed25519(message, ed25519_sig, ed25519_pk)
            }
            CryptoPhase::Q1 => {
                // Hybrid mode: Both signatures must be valid
                let ed25519_sig = signature.ed25519_signature.as_ref()
                    .ok_or_else(|| anyhow!("Ed25519 signature missing for Q1"))?;
                let dilithium5_sig = signature.dilithium5_signature.as_ref()
                    .ok_or_else(|| anyhow!("Dilithium5 signature missing for Q1"))?;

                let ed25519_pk = ed25519_public_key
                    .ok_or_else(|| anyhow!("Ed25519 public key missing for Q1"))?;
                let dilithium5_pk = dilithium5_public_key
                    .ok_or_else(|| anyhow!("Dilithium5 public key missing for Q1"))?;

                let ed25519_valid = Self::verify_ed25519(message, ed25519_sig, ed25519_pk)?;
                let dilithium5_valid = Dilithium5KeyPair::verify(message, dilithium5_sig, dilithium5_pk)?;

                Ok(ed25519_valid && dilithium5_valid)
            }
            CryptoPhase::Q2 => {
                let dilithium5_sig = signature.dilithium5_signature.as_ref()
                    .ok_or_else(|| anyhow!("Dilithium5 signature missing for Q2"))?;
                let dilithium5_pk = dilithium5_public_key
                    .ok_or_else(|| anyhow!("Dilithium5 public key missing for Q2"))?;

                Dilithium5KeyPair::verify(message, dilithium5_sig, dilithium5_pk)
            }
        }
    }

    /// Verify Ed25519 signature
    fn verify_ed25519(message: &[u8], signature_bytes: &[u8], public_key_bytes: &[u8]) -> Result<bool> {
        use ed25519_dalek::Verifier;

        let pk = Ed25519VerifyingKey::from_bytes(
            public_key_bytes.try_into()
                .map_err(|_| anyhow!("Invalid Ed25519 public key length"))?
        ).map_err(|e| anyhow!("Invalid Ed25519 public key: {}", e))?;

        let sig = Ed25519Signature::from_bytes(
            signature_bytes.try_into()
                .map_err(|_| anyhow!("Invalid Ed25519 signature length"))?
        );

        match pk.verify(message, &sig) {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    /// Derive address from public keys (uses highest security key available)
    pub fn derive_address(&self) -> [u8; 32] {
        match self.phase {
            CryptoPhase::Q0 => {
                // Use Ed25519 public key for Q0
                let pk = self.ed25519_key.as_ref()
                    .expect("Ed25519 key required for Q0")
                    .verifying_key()
                    .to_bytes();
                Self::hash_to_address(&pk)
            }
            CryptoPhase::Q1 => {
                // Use Dilithium5 public key for Q1 (higher security)
                let pk = self.dilithium5_key.as_ref()
                    .expect("Dilithium5 key required for Q1")
                    .public_key.as_bytes();
                Self::hash_to_address(pk)
            }
            CryptoPhase::Q2 => {
                // Use Dilithium5 public key for Q2
                let pk = self.dilithium5_key.as_ref()
                    .expect("Dilithium5 key required for Q2")
                    .public_key.as_bytes();
                Self::hash_to_address(pk)
            }
        }
    }

    /// Hash public key to address using SHA3-256
    fn hash_to_address(public_key: &[u8]) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(public_key);
        hasher.finalize().into()
    }

    /// Get Ed25519 public key bytes if available
    pub fn ed25519_public_key_bytes(&self) -> Option<Vec<u8>> {
        self.ed25519_key.as_ref().map(|k| k.verifying_key().to_bytes().to_vec())
    }

    /// Get Dilithium5 public key bytes if available
    pub fn dilithium5_public_key_bytes(&self) -> Option<Vec<u8>> {
        self.dilithium5_key.as_ref().map(|k| k.public_key.as_bytes().to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q0_wallet_generation() {
        let wallet = HybridWallet::generate(CryptoPhase::Q0);
        assert_eq!(wallet.phase, CryptoPhase::Q0);
        assert!(wallet.ed25519_key.is_some());
        assert!(wallet.dilithium5_key.is_none());
    }

    #[test]
    fn test_q1_wallet_generation() {
        let wallet = HybridWallet::generate(CryptoPhase::Q1);
        assert_eq!(wallet.phase, CryptoPhase::Q1);
        assert!(wallet.ed25519_key.is_some());
        assert!(wallet.dilithium5_key.is_some());
    }

    #[test]
    fn test_q2_wallet_generation() {
        let wallet = HybridWallet::generate(CryptoPhase::Q2);
        assert_eq!(wallet.phase, CryptoPhase::Q2);
        assert!(wallet.ed25519_key.is_none());
        assert!(wallet.dilithium5_key.is_some());
    }

    #[test]
    fn test_q0_sign_verify() {
        let wallet = HybridWallet::generate(CryptoPhase::Q0);
        let message = b"Test message for Q0";

        let signature = wallet.sign(message).expect("Signing failed");
        assert_eq!(signature.phase, CryptoPhase::Q0);
        assert!(signature.ed25519_signature.is_some());
        assert!(signature.dilithium5_signature.is_none());

        let ed25519_pk = wallet.ed25519_public_key_bytes();
        let is_valid = HybridWallet::verify(
            message,
            &signature,
            ed25519_pk.as_deref(),
            None,
        ).expect("Verification failed");

        assert!(is_valid, "Q0 signature should be valid");
    }

    #[test]
    fn test_q1_sign_verify() {
        let wallet = HybridWallet::generate(CryptoPhase::Q1);
        let message = b"Test message for Q1";

        let signature = wallet.sign(message).expect("Signing failed");
        assert_eq!(signature.phase, CryptoPhase::Q1);
        assert!(signature.ed25519_signature.is_some());
        assert!(signature.dilithium5_signature.is_some());

        let ed25519_pk = wallet.ed25519_public_key_bytes();
        let dilithium5_pk = wallet.dilithium5_public_key_bytes();

        let is_valid = HybridWallet::verify(
            message,
            &signature,
            ed25519_pk.as_deref(),
            dilithium5_pk.as_deref(),
        ).expect("Verification failed");

        assert!(is_valid, "Q1 dual signature should be valid");
    }

    #[test]
    fn test_q2_sign_verify() {
        let wallet = HybridWallet::generate(CryptoPhase::Q2);
        let message = b"Test message for Q2";

        let signature = wallet.sign(message).expect("Signing failed");
        assert_eq!(signature.phase, CryptoPhase::Q2);
        assert!(signature.ed25519_signature.is_none());
        assert!(signature.dilithium5_signature.is_some());

        let dilithium5_pk = wallet.dilithium5_public_key_bytes();
        let is_valid = HybridWallet::verify(
            message,
            &signature,
            None,
            dilithium5_pk.as_deref(),
        ).expect("Verification failed");

        assert!(is_valid, "Q2 signature should be valid");
    }

    #[test]
    fn test_q1_requires_both_valid_signatures() {
        let wallet = HybridWallet::generate(CryptoPhase::Q1);
        let wrong_wallet = HybridWallet::generate(CryptoPhase::Q1);
        let message = b"Test message";

        let signature = wallet.sign(message).expect("Signing failed");

        // Use wrong Ed25519 key
        let wrong_ed25519_pk = wrong_wallet.ed25519_public_key_bytes();
        let correct_dilithium5_pk = wallet.dilithium5_public_key_bytes();

        let is_valid = HybridWallet::verify(
            message,
            &signature,
            wrong_ed25519_pk.as_deref(),
            correct_dilithium5_pk.as_deref(),
        ).expect("Verification failed");

        assert!(!is_valid, "Q1 signature with wrong Ed25519 key should fail");
    }

    #[test]
    fn test_address_derivation() {
        let q0_wallet = HybridWallet::generate(CryptoPhase::Q0);
        let q1_wallet = HybridWallet::generate(CryptoPhase::Q1);
        let q2_wallet = HybridWallet::generate(CryptoPhase::Q2);

        let q0_addr = q0_wallet.derive_address();
        let q1_addr = q1_wallet.derive_address();
        let q2_addr = q2_wallet.derive_address();

        assert_eq!(q0_addr.len(), 32);
        assert_eq!(q1_addr.len(), 32);
        assert_eq!(q2_addr.len(), 32);

        // Addresses should be different (statistically certain)
        assert_ne!(q0_addr, q1_addr);
        assert_ne!(q1_addr, q2_addr);
        assert_ne!(q0_addr, q2_addr);
    }

    #[test]
    fn test_q1_hybrid_signature_size() {
        let wallet = HybridWallet::generate(CryptoPhase::Q1);
        let message = b"Test";

        let signature = wallet.sign(message).expect("Signing failed");

        // Ed25519 signature is 64 bytes
        assert_eq!(signature.ed25519_signature.as_ref().unwrap().len(), 64);

        // Dilithium5 signed message is large (~4627 bytes)
        assert!(signature.dilithium5_signature.as_ref().unwrap().len() > 4000);
    }
}
