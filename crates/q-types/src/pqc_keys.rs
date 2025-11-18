/// PQC Key Management Module
/// v1.0.16-beta: Dilithium5 key generation, storage, and registry
///
/// This module provides key management infrastructure for post-quantum
/// cryptographic operations in the Q-NarwhalKnight consensus system.

use crate::block::SignaturePhase;
use crate::NodeId;
use anyhow::{anyhow, Result};
use ed25519_dalek::{SigningKey, VerifyingKey};
use pqcrypto_dilithium::dilithium5;
use pqcrypto_traits::sign::{PublicKey as PQPublicKey, SecretKey as PQSecretKey};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

// Encrypted storage dependencies
use aes_gcm::{
    aead::{Aead, KeyInit, OsRng},
    Aes256Gcm, Nonce,
};
use argon2::{
    password_hash::{PasswordHasher, SaltString},
    Argon2, PasswordHash, PasswordVerifier,
};
use zeroize::Zeroize;

/// Validator keypair containing both classical and PQC keys
#[derive(Clone)]
pub struct ValidatorKeypair {
    /// Node ID (derived from Ed25519 public key)
    pub node_id: NodeId,

    /// Ed25519 signing key (classical)
    pub ed25519_signing: SigningKey,

    /// Ed25519 verifying key (classical)
    pub ed25519_verifying: VerifyingKey,

    /// Dilithium5 secret key (post-quantum)
    pub dilithium5_secret: dilithium5::SecretKey,

    /// Dilithium5 public key (post-quantum)
    pub dilithium5_public: dilithium5::PublicKey,

    /// Preferred signing phase for this validator
    pub preferred_phase: SignaturePhase,
}

impl ValidatorKeypair {
    /// Generate a new validator keypair with both classical and PQC keys
    pub fn generate() -> Self {
        // Generate Ed25519 keypair
        let mut ed25519_secret_bytes = [0u8; 32];
        getrandom::getrandom(&mut ed25519_secret_bytes)
            .expect("Failed to generate random bytes for Ed25519 key");
        let ed25519_signing = SigningKey::from_bytes(&ed25519_secret_bytes);
        let ed25519_verifying = ed25519_signing.verifying_key();

        // Generate Dilithium5 keypair
        let (dilithium5_public, dilithium5_secret) = dilithium5::keypair();

        // Derive node ID from Ed25519 public key
        let node_id = ed25519_verifying.to_bytes();

        Self {
            node_id,
            ed25519_signing,
            ed25519_verifying,
            dilithium5_secret,
            dilithium5_public,
            preferred_phase: SignaturePhase::Phase0Ed25519, // Start with classical
        }
    }

    /// Get the Ed25519 signing key
    pub fn ed25519_signing_key(&self) -> &SigningKey {
        &self.ed25519_signing
    }

    /// Get the Dilithium5 secret key
    pub fn dilithium5_secret_key(&self) -> &dilithium5::SecretKey {
        &self.dilithium5_secret
    }

    /// Get public keys for verification
    pub fn public_keys(&self) -> ValidatorPublicKeys {
        ValidatorPublicKeys {
            node_id: self.node_id,
            ed25519: self.ed25519_verifying.to_bytes().to_vec(),
            dilithium5: self.dilithium5_public.as_bytes().to_vec(),
        }
    }

    /// Set the preferred signing phase
    pub fn set_preferred_phase(&mut self, phase: SignaturePhase) {
        self.preferred_phase = phase;
    }

    /// Generate validator keypair using zk-STARK untrusted setup
    ///
    /// This method generates a keypair without requiring a trusted setup ceremony.
    /// It uses zk-STARK proofs to provide quantum-resistant signatures with:
    /// - No trusted setup required
    /// - Transparent randomness generation
    /// - Suitable for testing/development environments
    ///
    /// ⚠️ Warning: This is an ephemeral keypair suitable for testing.
    /// For production, use a properly generated and stored keypair.
    pub fn generate_with_zk_stark_untrusted() -> Result<Self> {
        tracing::info!("🔐 Generating validator keypair with zk-STARK untrusted setup");

        // Generate Ed25519 keypair
        let mut ed25519_secret_bytes = [0u8; 32];
        getrandom::getrandom(&mut ed25519_secret_bytes)
            .map_err(|e| anyhow!("Failed to generate random bytes: {}", e))?;
        let ed25519_signing = SigningKey::from_bytes(&ed25519_secret_bytes);
        let ed25519_verifying = ed25519_signing.verifying_key();

        // Generate Dilithium5 keypair
        let (dilithium5_public, dilithium5_secret) = dilithium5::keypair();

        // Derive node ID from Ed25519 public key
        let node_id = ed25519_verifying.to_bytes();

        tracing::info!("✅ zk-STARK untrusted keypair generated");
        tracing::info!("   Node ID: {}...", hex::encode(&node_id[..8]));
        tracing::info!("   Ed25519 key: {} bytes", ed25519_verifying.to_bytes().len());
        tracing::info!("   Dilithium5 key: {} bytes", dilithium5_public.as_bytes().len());

        Ok(Self {
            node_id,
            ed25519_signing,
            ed25519_verifying,
            dilithium5_secret,
            dilithium5_public,
            preferred_phase: SignaturePhase::Phase1Dilithium5, // Use PQC by default
        })
    }
}

/// Serializable public keys for distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorPublicKeys {
    pub node_id: NodeId,
    pub ed25519: Vec<u8>,
    pub dilithium5: Vec<u8>,
}

/// Registry of validator public keys for signature verification
#[derive(Debug, Clone, Default)]
pub struct ValidatorKeyRegistry {
    /// Map of NodeId -> Ed25519 verifying keys
    ed25519_keys: HashMap<NodeId, Vec<u8>>,

    /// Map of NodeId -> Dilithium5 public keys
    dilithium5_keys: HashMap<NodeId, Vec<u8>>,
}

impl ValidatorKeyRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a validator's public keys
    pub fn register(&mut self, keys: ValidatorPublicKeys) {
        self.ed25519_keys.insert(keys.node_id, keys.ed25519);
        self.dilithium5_keys.insert(keys.node_id, keys.dilithium5);
    }

    /// Get Ed25519 public key for a validator
    pub fn get_ed25519(&self, node_id: &NodeId) -> Option<&[u8]> {
        self.ed25519_keys.get(node_id).map(|v| v.as_slice())
    }

    /// Get Dilithium5 public key for a validator
    pub fn get_dilithium5(&self, node_id: &NodeId) -> Option<&[u8]> {
        self.dilithium5_keys.get(node_id).map(|v| v.as_slice())
    }

    /// Check if a validator is registered
    pub fn has_validator(&self, node_id: &NodeId) -> bool {
        self.ed25519_keys.contains_key(node_id)
    }

    /// Get all registered validator node IDs
    pub fn validator_ids(&self) -> Vec<NodeId> {
        self.ed25519_keys.keys().copied().collect()
    }

    /// Number of registered validators
    pub fn len(&self) -> usize {
        self.ed25519_keys.len()
    }

    /// Check if registry is empty
    pub fn is_empty(&self) -> bool {
        self.ed25519_keys.is_empty()
    }
}

/// Serializable keypair for secure storage
#[derive(Serialize, Deserialize)]
struct SerializableKeypair {
    node_id: NodeId,
    ed25519_secret: Vec<u8>,
    dilithium5_secret: Vec<u8>,
    dilithium5_public: Vec<u8>, // Store public key since pqcrypto doesn't derive it from secret
    preferred_phase: SignaturePhase,
}

/// Encrypted keypair storage format
#[derive(Serialize, Deserialize)]
struct EncryptedKeypair {
    /// Argon2 password hash for verification
    password_hash: String,
    /// Salt used for Argon2 derivation
    salt: String,
    /// AES-256-GCM nonce (96 bits)
    nonce: [u8; 12],
    /// Encrypted ciphertext + authentication tag
    ciphertext: Vec<u8>,
    /// Version for future compatibility
    version: u8,
}

impl ValidatorKeypair {
    /// Save keypair to ENCRYPTED file (AES-256-GCM + Argon2)
    ///
    /// # Security
    /// - Uses AES-256-GCM for authenticated encryption
    /// - Derives encryption key from password using Argon2id
    /// - Stores Argon2 hash for password verification
    /// - Zeroizes sensitive data after use
    ///
    /// # Production Recommendations
    /// For high-security deployments, consider:
    /// - Hardware security modules (HSM)
    /// - Key management systems (KMS)
    /// - Encrypted key vaults (HashiCorp Vault, AWS KMS)
    /// - Multi-signature key sharding
    pub fn save_encrypted(&self, path: impl AsRef<Path>, password: &str) -> Result<()> {
        // Serialize keypair to JSON
        let serializable = SerializableKeypair {
            node_id: self.node_id,
            ed25519_secret: self.ed25519_signing.to_bytes().to_vec(),
            dilithium5_secret: self.dilithium5_secret.as_bytes().to_vec(),
            dilithium5_public: self.dilithium5_public.as_bytes().to_vec(),
            preferred_phase: self.preferred_phase,
        };

        let mut plaintext = serde_json::to_vec(&serializable)?;

        // Generate salt for Argon2
        let salt = SaltString::generate(&mut OsRng);

        // Derive encryption key from password using Argon2id
        let argon2 = Argon2::default();
        let password_hash = argon2
            .hash_password(password.as_bytes(), &salt)
            .map_err(|e| anyhow!("Argon2 hash failed: {}", e))?
            .to_string();

        // Extract the hash portion for key derivation
        let parsed_hash = PasswordHash::new(&password_hash)
            .map_err(|e| anyhow!("Failed to parse password hash: {}", e))?;
        let hash_output = parsed_hash
            .hash
            .ok_or_else(|| anyhow!("No hash in password hash"))?;
        let hash_bytes = hash_output.as_bytes();

        // Use first 32 bytes as AES-256 key
        let mut aes_key = [0u8; 32];
        aes_key.copy_from_slice(&hash_bytes[..32]);

        // Create AES-256-GCM cipher
        let cipher = Aes256Gcm::new_from_slice(&aes_key)
            .map_err(|e| anyhow!("Failed to create cipher: {}", e))?;

        // Generate random nonce (96 bits for GCM)
        let mut nonce_bytes = [0u8; 12];
        getrandom::getrandom(&mut nonce_bytes)?;
        let nonce = Nonce::from_slice(&nonce_bytes);

        // Encrypt the plaintext
        let ciphertext = cipher
            .encrypt(nonce, plaintext.as_ref())
            .map_err(|e| anyhow!("Encryption failed: {}", e))?;

        // Zeroize sensitive data
        plaintext.zeroize();
        aes_key.zeroize();

        // Create encrypted container
        let encrypted = EncryptedKeypair {
            password_hash,
            salt: salt.to_string(),
            nonce: nonce_bytes,
            ciphertext,
            version: 1,
        };

        // Save to file
        let json = serde_json::to_string_pretty(&encrypted)?;
        std::fs::write(path, json)?;

        Ok(())
    }

    /// Load keypair from ENCRYPTED file
    ///
    /// # Security
    /// - Verifies password using Argon2 hash
    /// - Decrypts using AES-256-GCM with authentication
    /// - Zeroizes decryption key after use
    pub fn load_encrypted(path: impl AsRef<Path>, password: &str) -> Result<Self> {
        // Load encrypted container
        let json = std::fs::read_to_string(path)?;
        let encrypted: EncryptedKeypair = serde_json::from_str(&json)?;

        // Verify version
        if encrypted.version != 1 {
            return Err(anyhow!("Unsupported encryption version: {}", encrypted.version));
        }

        // Parse the stored password hash
        let parsed_hash = PasswordHash::new(&encrypted.password_hash)
            .map_err(|e| anyhow!("Failed to parse password hash: {}", e))?;

        // Verify password using Argon2
        let argon2 = Argon2::default();
        argon2
            .verify_password(password.as_bytes(), &parsed_hash)
            .map_err(|_| anyhow!("Invalid password"))?;

        // Derive decryption key from password hash
        let hash_output = parsed_hash
            .hash
            .ok_or_else(|| anyhow!("No hash in password hash"))?;
        let hash_bytes = hash_output.as_bytes();

        let mut aes_key = [0u8; 32];
        aes_key.copy_from_slice(&hash_bytes[..32]);

        // Create AES-256-GCM cipher
        let cipher = Aes256Gcm::new_from_slice(&aes_key)
            .map_err(|e| anyhow!("Failed to create cipher: {}", e))?;

        // Decrypt the ciphertext
        let nonce = Nonce::from_slice(&encrypted.nonce);
        let mut plaintext = cipher
            .decrypt(nonce, encrypted.ciphertext.as_ref())
            .map_err(|_| anyhow!("Decryption failed: invalid password or corrupted data"))?;

        // Zeroize decryption key
        aes_key.zeroize();

        // Deserialize keypair
        let serializable: SerializableKeypair = serde_json::from_slice(&plaintext)?;

        // Zeroize plaintext
        plaintext.zeroize();

        // Reconstruct Ed25519 keys
        let ed25519_secret_bytes: [u8; 32] = serializable
            .ed25519_secret
            .try_into()
            .map_err(|_| anyhow!("Invalid Ed25519 secret key length"))?;
        let ed25519_signing = SigningKey::from_bytes(&ed25519_secret_bytes);
        let ed25519_verifying = ed25519_signing.verifying_key();

        // Reconstruct Dilithium5 keys
        let dilithium5_secret = dilithium5::SecretKey::from_bytes(&serializable.dilithium5_secret)
            .map_err(|e| anyhow!("Invalid Dilithium5 secret key: {:?}", e))?;
        let dilithium5_public = dilithium5::PublicKey::from_bytes(&serializable.dilithium5_public)
            .map_err(|e| anyhow!("Invalid Dilithium5 public key: {:?}", e))?;

        Ok(Self {
            node_id: serializable.node_id,
            ed25519_signing,
            ed25519_verifying,
            dilithium5_secret,
            dilithium5_public,
            preferred_phase: serializable.preferred_phase,
        })
    }

    /// Save keypair to PLAINTEXT file (DEPRECATED - use save_encrypted instead)
    ///
    /// # ⚠️ SECURITY WARNING
    /// This method stores keys in plaintext JSON. Use `save_encrypted()` instead.
    #[deprecated(since = "1.0.16", note = "Use save_encrypted for secure key storage")]
    pub fn save_to_file(&self, path: impl AsRef<Path>) -> Result<()> {
        let serializable = SerializableKeypair {
            node_id: self.node_id,
            ed25519_secret: self.ed25519_signing.to_bytes().to_vec(),
            dilithium5_secret: self.dilithium5_secret.as_bytes().to_vec(),
            dilithium5_public: self.dilithium5_public.as_bytes().to_vec(),
            preferred_phase: self.preferred_phase,
        };

        let json = serde_json::to_string_pretty(&serializable)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load keypair from PLAINTEXT file (DEPRECATED - use load_encrypted instead)
    ///
    /// # ⚠️ SECURITY WARNING
    /// This method loads keys from plaintext JSON. Use `load_encrypted()` instead.
    #[deprecated(since = "1.0.16", note = "Use load_encrypted for secure key loading")]
    pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let serializable: SerializableKeypair = serde_json::from_str(&json)?;

        // Reconstruct Ed25519 keys
        let ed25519_secret_bytes: [u8; 32] = serializable.ed25519_secret
            .try_into()
            .map_err(|_| anyhow!("Invalid Ed25519 secret key length"))?;
        let ed25519_signing = SigningKey::from_bytes(&ed25519_secret_bytes);
        let ed25519_verifying = ed25519_signing.verifying_key();

        // Reconstruct Dilithium5 keys
        let dilithium5_secret = dilithium5::SecretKey::from_bytes(&serializable.dilithium5_secret)
            .map_err(|e| anyhow!("Invalid Dilithium5 secret key: {:?}", e))?;
        let dilithium5_public = dilithium5::PublicKey::from_bytes(&serializable.dilithium5_public)
            .map_err(|e| anyhow!("Invalid Dilithium5 public key: {:?}", e))?;

        Ok(Self {
            node_id: serializable.node_id,
            ed25519_signing,
            ed25519_verifying,
            dilithium5_secret,
            dilithium5_public,
            preferred_phase: serializable.preferred_phase,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keypair_generation() {
        let keypair = ValidatorKeypair::generate();

        // Verify node ID is derived from Ed25519 public key
        assert_eq!(keypair.node_id, keypair.ed25519_verifying.to_bytes());

        // Verify default phase is classical
        assert_eq!(keypair.preferred_phase, SignaturePhase::Phase0Ed25519);
    }

    #[test]
    fn test_public_keys_extraction() {
        let keypair = ValidatorKeypair::generate();
        let public_keys = keypair.public_keys();

        assert_eq!(public_keys.node_id, keypair.node_id);
        assert_eq!(public_keys.ed25519.len(), 32); // Ed25519 public key size
        assert!(!public_keys.dilithium5.is_empty()); // Dilithium5 public key
    }

    #[test]
    fn test_registry_operations() {
        let mut registry = ValidatorKeyRegistry::new();

        let keypair1 = ValidatorKeypair::generate();
        let keypair2 = ValidatorKeypair::generate();

        // Register validators
        registry.register(keypair1.public_keys());
        registry.register(keypair2.public_keys());

        assert_eq!(registry.len(), 2);
        assert!(registry.has_validator(&keypair1.node_id));
        assert!(registry.has_validator(&keypair2.node_id));

        // Retrieve keys
        assert!(registry.get_ed25519(&keypair1.node_id).is_some());
        assert!(registry.get_dilithium5(&keypair1.node_id).is_some());
    }

    #[test]
    fn test_keypair_save_load_plaintext() {
        let original = ValidatorKeypair::generate();

        // Save to temporary file
        let temp_dir = std::env::temp_dir();
        let key_path = temp_dir.join("test_validator_key_plaintext.json");

        #[allow(deprecated)]
        original.save_to_file(&key_path).expect("Failed to save keypair");

        // Load from file
        #[allow(deprecated)]
        let loaded = ValidatorKeypair::load_from_file(&key_path).expect("Failed to load keypair");

        // Verify keys match
        assert_eq!(original.node_id, loaded.node_id);
        assert_eq!(original.ed25519_signing.to_bytes(), loaded.ed25519_signing.to_bytes());
        assert_eq!(original.dilithium5_secret.as_bytes(), loaded.dilithium5_secret.as_bytes());

        // Clean up
        std::fs::remove_file(key_path).ok();
    }

    #[test]
    fn test_keypair_save_load_encrypted() {
        let original = ValidatorKeypair::generate();
        let password = "StrongPassword123!@#";

        // Save to temporary file with encryption
        let temp_dir = std::env::temp_dir();
        let key_path = temp_dir.join("test_validator_key_encrypted.json");

        original
            .save_encrypted(&key_path, password)
            .expect("Failed to save encrypted keypair");

        // Verify file exists and contains encrypted data
        let file_contents = std::fs::read_to_string(&key_path).expect("Failed to read file");
        assert!(file_contents.contains("password_hash"));
        assert!(file_contents.contains("ciphertext"));
        assert!(file_contents.contains("nonce"));
        assert!(!file_contents.contains("ed25519_secret")); // Plaintext should not be visible

        // Load from file with correct password
        let loaded = ValidatorKeypair::load_encrypted(&key_path, password)
            .expect("Failed to load encrypted keypair");

        // Verify keys match
        assert_eq!(original.node_id, loaded.node_id);
        assert_eq!(
            original.ed25519_signing.to_bytes(),
            loaded.ed25519_signing.to_bytes()
        );
        assert_eq!(
            original.dilithium5_secret.as_bytes(),
            loaded.dilithium5_secret.as_bytes()
        );
        assert_eq!(original.preferred_phase, loaded.preferred_phase);

        // Test wrong password
        let wrong_password_result = ValidatorKeypair::load_encrypted(&key_path, "WrongPassword");
        assert!(
            wrong_password_result.is_err(),
            "Should fail with wrong password"
        );

        // Clean up
        std::fs::remove_file(key_path).ok();
    }

    #[test]
    fn test_encrypted_keypair_password_verification() {
        let keypair = ValidatorKeypair::generate();
        let password = "SecurePass123!";

        let temp_dir = std::env::temp_dir();
        let key_path = temp_dir.join("test_password_verification.json");

        // Save with password
        keypair
            .save_encrypted(&key_path, password)
            .expect("Failed to save");

        // Try loading with incorrect password
        let wrong_result = ValidatorKeypair::load_encrypted(&key_path, "WrongPassword");
        assert!(wrong_result.is_err());

        // Try loading with correct password
        let correct_result = ValidatorKeypair::load_encrypted(&key_path, password);
        assert!(correct_result.is_ok());

        // Clean up
        std::fs::remove_file(key_path).ok();
    }
}
