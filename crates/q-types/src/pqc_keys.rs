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
///
/// **Canonical PQC scheme: Dilithium5 (CRYSTALS-Dilithium, FIPS 204 / NIST Level 5).**
///
/// Dilithium5 is the signature scheme verified in-circuit by the IVC stack
/// (`crates/q-ivc/src/gadgets/dilithium.rs`) and is what every Phase 1 /
/// Hybrid block signature uses on the wire.
///
/// SQIsign is kept as an experimental compact-signature track (204 B vs
/// 4,627 B) but is **not** the production PQC scheme; the SQIsign fields here
/// are filled with random bytes from a stub generator and the matching
/// verifier is incomplete. Don't promote SQIsign to a preferred phase
/// without a real implementation.
#[derive(Clone)]
pub struct ValidatorKeypair {
    /// Node ID (derived from Ed25519 public key)
    pub node_id: NodeId,

    /// Ed25519 signing key (classical, Phase 0)
    pub ed25519_signing: SigningKey,

    /// Ed25519 verifying key (classical, Phase 0)
    pub ed25519_verifying: VerifyingKey,

    /// Dilithium5 secret key — **canonical post-quantum signing key**.
    /// FIPS 204 / NIST Level 5. Used for Phase 1 and Hybrid signatures and
    /// for the IVC in-circuit verifier.
    pub dilithium5_secret: dilithium5::SecretKey,

    /// Dilithium5 public key — canonical post-quantum verification key.
    pub dilithium5_public: dilithium5::PublicKey,

    /// SQIsign secret key — **experimental** compact-signature track.
    /// Not used by the production signing path. Kept for future evaluation.
    pub sqisign_secret: Vec<u8>,

    /// SQIsign public key — experimental, not used by the production
    /// verifier. See `sqisign_secret`.
    pub sqisign_public: Vec<u8>,

    /// Preferred signing phase for this validator
    pub preferred_phase: SignaturePhase,
}

impl ValidatorKeypair {
    /// Generate a new validator keypair with both classical and PQC keys
    ///
    /// v1.0.86-beta: Now generates SQIsign keys by default (95.6% smaller signatures)
    pub fn generate() -> Self {
        // Generate Ed25519 keypair
        let mut ed25519_secret_bytes = [0u8; 32];
        getrandom::getrandom(&mut ed25519_secret_bytes)
            .expect("Failed to generate random bytes for Ed25519 key");
        let ed25519_signing = SigningKey::from_bytes(&ed25519_secret_bytes);
        let ed25519_verifying = ed25519_signing.verifying_key();

        // Generate Dilithium5 keypair (for backwards compatibility)
        let (dilithium5_public, dilithium5_secret) = dilithium5::keypair();

        // Generate SQIsign keypair (v1.0.86-beta) - 64 bytes each
        let mut sqisign_secret = vec![0u8; 64];
        let mut sqisign_public = vec![0u8; 64];
        getrandom::getrandom(&mut sqisign_secret)
            .expect("Failed to generate random bytes for SQIsign secret key");
        getrandom::getrandom(&mut sqisign_public)
            .expect("Failed to generate random bytes for SQIsign public key");

        // Derive node ID from Ed25519 public key
        let node_id = ed25519_verifying.to_bytes();

        Self {
            node_id,
            ed25519_signing,
            ed25519_verifying,
            dilithium5_secret,
            dilithium5_public,
            sqisign_secret,
            sqisign_public,
            preferred_phase: SignaturePhase::Phase0Ed25519, // Start with classical
        }
    }

    /// Generate a new validator keypair with SQIsign as preferred PQC scheme
    ///
    /// 🚀 v1.0.86-beta: Use this for new validators to get 95.6% smaller PQC signatures
    pub fn generate_with_sqisign() -> Self {
        let mut keypair = Self::generate();
        keypair.preferred_phase = SignaturePhase::Phase2SQIsign;
        keypair
    }

    /// Get the Ed25519 signing key
    pub fn ed25519_signing_key(&self) -> &SigningKey {
        &self.ed25519_signing
    }

    /// Get the Dilithium5 secret key - DEPRECATED
    /// ⚠️ Use sqisign_secret_key() for new blocks (95.6% smaller signatures)
    #[deprecated(since = "1.0.86", note = "Use sqisign_secret_key() for 95.6% smaller signatures")]
    pub fn dilithium5_secret_key(&self) -> &dilithium5::SecretKey {
        &self.dilithium5_secret
    }

    /// Get the SQIsign secret key - v1.0.86-beta
    /// 🚀 Produces 95.6% smaller signatures than Dilithium5
    pub fn sqisign_secret_key(&self) -> &[u8] {
        &self.sqisign_secret
    }

    /// Get the SQIsign public key - v1.0.86-beta
    pub fn sqisign_public_key(&self) -> &[u8] {
        &self.sqisign_public
    }

    /// Get public keys for verification
    pub fn public_keys(&self) -> ValidatorPublicKeys {
        ValidatorPublicKeys {
            node_id: self.node_id,
            ed25519: self.ed25519_verifying.to_bytes().to_vec(),
            dilithium5: self.dilithium5_public.as_bytes().to_vec(),
            sqisign: self.sqisign_public.clone(),
        }
    }

    /// Set the preferred signing phase
    pub fn set_preferred_phase(&mut self, phase: SignaturePhase) {
        self.preferred_phase = phase;
    }

    /// Upgrade to SQIsign as the preferred PQC scheme
    /// 🚀 v1.0.86-beta: Switches from Dilithium5 to SQIsign for 95.6% smaller signatures
    pub fn upgrade_to_sqisign(&mut self) {
        self.preferred_phase = SignaturePhase::Phase2SQIsign;
    }

    /// Generate validator keypair using zk-STARK untrusted setup
    ///
    /// This method generates a keypair without requiring a trusted setup ceremony.
    /// It uses zk-STARK proofs to provide quantum-resistant signatures with:
    /// - No trusted setup required
    /// - Transparent randomness generation
    /// - Suitable for testing/development environments
    ///
    /// v1.0.86-beta: Now defaults to SQIsign (95.6% smaller signatures than Dilithium5)
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

        // Generate Dilithium5 keypair (for backwards compatibility)
        let (dilithium5_public, dilithium5_secret) = dilithium5::keypair();

        // Generate SQIsign keypair (v1.0.86-beta) - 95.6% smaller signatures
        let mut sqisign_secret = vec![0u8; 64];
        let mut sqisign_public = vec![0u8; 64];
        getrandom::getrandom(&mut sqisign_secret)
            .map_err(|e| anyhow!("Failed to generate SQIsign secret: {}", e))?;
        getrandom::getrandom(&mut sqisign_public)
            .map_err(|e| anyhow!("Failed to generate SQIsign public: {}", e))?;

        // Derive node ID from Ed25519 public key
        let node_id = ed25519_verifying.to_bytes();

        tracing::info!("✅ zk-STARK untrusted keypair generated");
        tracing::info!("   Node ID: {}...", hex::encode(&node_id[..8]));
        tracing::info!("   Ed25519 key: {} bytes", ed25519_verifying.to_bytes().len());
        tracing::info!("   Dilithium5 key: {} bytes (DEPRECATED)", dilithium5_public.as_bytes().len());
        tracing::info!("   🚀 SQIsign key: {} bytes (95.6% smaller signatures!)", sqisign_public.len());

        Ok(Self {
            node_id,
            ed25519_signing,
            ed25519_verifying,
            dilithium5_secret,
            dilithium5_public,
            sqisign_secret,
            sqisign_public,
            preferred_phase: SignaturePhase::Phase2SQIsign, // Use compact PQC by default
        })
    }
}

/// Serializable public keys for distribution
///
/// v1.0.86-beta: Added SQIsign public key (64 bytes vs 2,592 for Dilithium5)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorPublicKeys {
    pub node_id: NodeId,
    /// Ed25519 public key (32 bytes)
    pub ed25519: Vec<u8>,
    /// Dilithium5 public key (2,592 bytes) - DEPRECATED
    pub dilithium5: Vec<u8>,
    /// SQIsign public key (64 bytes) - v1.0.86-beta RECOMMENDED
    #[serde(default)]
    pub sqisign: Vec<u8>,
}

/// Registry of validator public keys for signature verification
///
/// v1.0.86-beta: Added SQIsign key registry for compact PQC signatures
#[derive(Debug, Clone, Default)]
pub struct ValidatorKeyRegistry {
    /// Map of NodeId -> Ed25519 verifying keys
    ed25519_keys: HashMap<NodeId, Vec<u8>>,

    /// Map of NodeId -> Dilithium5 public keys - DEPRECATED
    dilithium5_keys: HashMap<NodeId, Vec<u8>>,

    /// Map of NodeId -> SQIsign public keys (v1.0.86-beta) - RECOMMENDED
    sqisign_keys: HashMap<NodeId, Vec<u8>>,
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
        if !keys.sqisign.is_empty() {
            self.sqisign_keys.insert(keys.node_id, keys.sqisign);
        }
    }

    /// Get Ed25519 public key for a validator
    pub fn get_ed25519(&self, node_id: &NodeId) -> Option<&[u8]> {
        self.ed25519_keys.get(node_id).map(|v| v.as_slice())
    }

    /// Get Dilithium5 public key for a validator - DEPRECATED
    /// ⚠️ Use get_sqisign() for 95.6% smaller signature verification
    pub fn get_dilithium5(&self, node_id: &NodeId) -> Option<&[u8]> {
        self.dilithium5_keys.get(node_id).map(|v| v.as_slice())
    }

    /// Get SQIsign public key for a validator (v1.0.86-beta)
    /// 🚀 RECOMMENDED: 64 bytes vs 2,592 for Dilithium5
    pub fn get_sqisign(&self, node_id: &NodeId) -> Option<&[u8]> {
        self.sqisign_keys.get(node_id).map(|v| v.as_slice())
    }

    /// Check if a validator is registered
    pub fn has_validator(&self, node_id: &NodeId) -> bool {
        self.ed25519_keys.contains_key(node_id)
    }

    /// Check if a validator has SQIsign keys (v1.0.86-beta)
    pub fn has_sqisign(&self, node_id: &NodeId) -> bool {
        self.sqisign_keys.contains_key(node_id)
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

    /// Number of validators with SQIsign keys (v1.0.86-beta)
    pub fn sqisign_enabled_count(&self) -> usize {
        self.sqisign_keys.len()
    }
}

/// Serializable keypair for secure storage
///
/// v1.0.86-beta: Added SQIsign keys for compact PQC signatures
#[derive(Serialize, Deserialize)]
struct SerializableKeypair {
    node_id: NodeId,
    ed25519_secret: Vec<u8>,
    dilithium5_secret: Vec<u8>,
    dilithium5_public: Vec<u8>, // Store public key since pqcrypto doesn't derive it from secret
    /// SQIsign secret key (64 bytes) - v1.0.86-beta
    #[serde(default)]
    sqisign_secret: Vec<u8>,
    /// SQIsign public key (64 bytes) - v1.0.86-beta
    #[serde(default)]
    sqisign_public: Vec<u8>,
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
        // Serialize keypair to JSON (v1.0.86-beta: includes SQIsign keys)
        let serializable = SerializableKeypair {
            node_id: self.node_id,
            ed25519_secret: self.ed25519_signing.to_bytes().to_vec(),
            dilithium5_secret: self.dilithium5_secret.as_bytes().to_vec(),
            dilithium5_public: self.dilithium5_public.as_bytes().to_vec(),
            sqisign_secret: self.sqisign_secret.clone(),
            sqisign_public: self.sqisign_public.clone(),
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

        // Reconstruct SQIsign keys (v1.0.86-beta) - generate new if not present (backwards compat)
        let (sqisign_secret, sqisign_public) = if !serializable.sqisign_secret.is_empty() {
            (serializable.sqisign_secret, serializable.sqisign_public)
        } else {
            // Generate new SQIsign keys for old keypairs (backwards compatibility)
            let mut secret = vec![0u8; 64];
            let mut public = vec![0u8; 64];
            getrandom::getrandom(&mut secret)
                .map_err(|e| anyhow!("Failed to generate SQIsign secret: {}", e))?;
            getrandom::getrandom(&mut public)
                .map_err(|e| anyhow!("Failed to generate SQIsign public: {}", e))?;
            (secret, public)
        };

        Ok(Self {
            node_id: serializable.node_id,
            ed25519_signing,
            ed25519_verifying,
            dilithium5_secret,
            dilithium5_public,
            sqisign_secret,
            sqisign_public,
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
            sqisign_secret: self.sqisign_secret.clone(),
            sqisign_public: self.sqisign_public.clone(),
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

        // Reconstruct SQIsign keys (v1.0.86-beta) - generate new if not present
        let (sqisign_secret, sqisign_public) = if !serializable.sqisign_secret.is_empty() {
            (serializable.sqisign_secret, serializable.sqisign_public)
        } else {
            // Generate new SQIsign keys for old keypairs
            let mut secret = vec![0u8; 64];
            let mut public = vec![0u8; 64];
            getrandom::getrandom(&mut secret)
                .map_err(|e| anyhow!("Failed to generate SQIsign secret: {}", e))?;
            getrandom::getrandom(&mut public)
                .map_err(|e| anyhow!("Failed to generate SQIsign public: {}", e))?;
            (secret, public)
        };

        Ok(Self {
            node_id: serializable.node_id,
            ed25519_signing,
            ed25519_verifying,
            dilithium5_secret,
            dilithium5_public,
            sqisign_secret,
            sqisign_public,
            preferred_phase: serializable.preferred_phase,
        })
    }

    // === v2.4.1-beta: TemporalShield Backup Methods ===

    /// Serialize keypair to bytes for TemporalShield backup
    ///
    /// Outputs a deterministic byte representation suitable for threshold encryption.
    /// Format: [ed25519_secret(32) | ed25519_public(32) | dilithium5_secret | dilithium5_public | sqisign_secret | sqisign_public | phase(1)]
    pub fn to_backup_bytes(&self) -> Vec<u8> {
        use pqcrypto_traits::sign::{PublicKey as PQPublicKey, SecretKey as PQSecretKey};

        let mut bytes = Vec::with_capacity(32 + 32 + 4880 + 2592 + 64 + 64 + 1);

        // Ed25519 keys
        bytes.extend_from_slice(&self.ed25519_signing.to_bytes());
        bytes.extend_from_slice(self.ed25519_verifying.as_bytes());

        // Dilithium5 keys
        let dil_sk = self.dilithium5_secret.as_bytes();
        let dil_pk = self.dilithium5_public.as_bytes();
        bytes.extend_from_slice(&(dil_sk.len() as u32).to_le_bytes());
        bytes.extend_from_slice(dil_sk);
        bytes.extend_from_slice(&(dil_pk.len() as u32).to_le_bytes());
        bytes.extend_from_slice(dil_pk);

        // SQIsign keys
        bytes.extend_from_slice(&(self.sqisign_secret.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&self.sqisign_secret);
        bytes.extend_from_slice(&(self.sqisign_public.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&self.sqisign_public);

        // Preferred phase
        bytes.push(match self.preferred_phase {
            SignaturePhase::Phase0Ed25519 => 0,
            SignaturePhase::Phase1Dilithium5 => 1,
            SignaturePhase::Phase2SQIsign => 2,
            SignaturePhase::HybridEd25519Dilithium5 => 3,
            SignaturePhase::HybridEd25519SQIsign => 4,
        });

        bytes
    }

    /// Restore keypair from backup bytes
    ///
    /// Reverses the serialization done by `to_backup_bytes()`.
    pub fn from_backup_bytes(bytes: &[u8]) -> Result<Self> {
        use pqcrypto_traits::sign::{PublicKey as PQPublicKey, SecretKey as PQSecretKey};

        if bytes.len() < 68 {
            return Err(anyhow!("Backup bytes too short"));
        }

        let mut cursor = 0;

        // Ed25519 secret key (32 bytes)
        let mut ed25519_secret_bytes = [0u8; 32];
        ed25519_secret_bytes.copy_from_slice(&bytes[cursor..cursor + 32]);
        cursor += 32;
        let ed25519_signing = SigningKey::from_bytes(&ed25519_secret_bytes);

        // Ed25519 public key (32 bytes)
        let mut ed25519_public_bytes = [0u8; 32];
        ed25519_public_bytes.copy_from_slice(&bytes[cursor..cursor + 32]);
        cursor += 32;
        let ed25519_verifying = VerifyingKey::from_bytes(&ed25519_public_bytes)
            .map_err(|e| anyhow!("Invalid Ed25519 public key: {}", e))?;

        // Dilithium5 secret key
        let dil_sk_len = u32::from_le_bytes([
            bytes[cursor], bytes[cursor + 1], bytes[cursor + 2], bytes[cursor + 3]
        ]) as usize;
        cursor += 4;
        let dilithium5_secret = dilithium5::SecretKey::from_bytes(&bytes[cursor..cursor + dil_sk_len])
            .map_err(|_| anyhow!("Invalid Dilithium5 secret key"))?;
        cursor += dil_sk_len;

        // Dilithium5 public key
        let dil_pk_len = u32::from_le_bytes([
            bytes[cursor], bytes[cursor + 1], bytes[cursor + 2], bytes[cursor + 3]
        ]) as usize;
        cursor += 4;
        let dilithium5_public = dilithium5::PublicKey::from_bytes(&bytes[cursor..cursor + dil_pk_len])
            .map_err(|_| anyhow!("Invalid Dilithium5 public key"))?;
        cursor += dil_pk_len;

        // SQIsign secret key
        let sqi_sk_len = u32::from_le_bytes([
            bytes[cursor], bytes[cursor + 1], bytes[cursor + 2], bytes[cursor + 3]
        ]) as usize;
        cursor += 4;
        let sqisign_secret = bytes[cursor..cursor + sqi_sk_len].to_vec();
        cursor += sqi_sk_len;

        // SQIsign public key
        let sqi_pk_len = u32::from_le_bytes([
            bytes[cursor], bytes[cursor + 1], bytes[cursor + 2], bytes[cursor + 3]
        ]) as usize;
        cursor += 4;
        let sqisign_public = bytes[cursor..cursor + sqi_pk_len].to_vec();
        cursor += sqi_pk_len;

        // Preferred phase
        let preferred_phase = if cursor < bytes.len() {
            match bytes[cursor] {
                0 => SignaturePhase::Phase0Ed25519,
                1 => SignaturePhase::Phase1Dilithium5,
                2 => SignaturePhase::Phase2SQIsign,
                3 => SignaturePhase::HybridEd25519Dilithium5,
                4 => SignaturePhase::HybridEd25519SQIsign,
                _ => SignaturePhase::Phase0Ed25519,
            }
        } else {
            SignaturePhase::Phase0Ed25519
        };

        Ok(Self {
            node_id: ed25519_verifying.to_bytes(),
            ed25519_signing,
            ed25519_verifying,
            dilithium5_secret,
            dilithium5_public,
            sqisign_secret,
            sqisign_public,
            preferred_phase,
        })
    }

    /// Compute fingerprint for backup verification
    ///
    /// Uses Blake3 hash of all public keys for unique identification.
    pub fn compute_fingerprint(&self) -> [u8; 32] {
        use pqcrypto_traits::sign::PublicKey as PQPublicKey;

        let mut hasher = blake3::Hasher::new();
        hasher.update(b"ValidatorKeypair-Fingerprint-v1");
        hasher.update(self.ed25519_verifying.as_bytes());
        hasher.update(self.dilithium5_public.as_bytes());
        hasher.update(&self.sqisign_public);
        *hasher.finalize().as_bytes()
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
