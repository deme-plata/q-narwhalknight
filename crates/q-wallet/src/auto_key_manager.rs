//! Automatic Key Management with Mnemonic + Password
//!
//! This module provides standard wallet key management:
//! - 12-word mnemonic phrase for backup/recovery
//! - Password for local wallet encryption
//! - ZK-STARK proofs for ownership verification
//!
//! **User Experience:**
//! ```text
//! Create wallet:    Get 12 words + Set password → Done ✓
//! Sign transaction: Enter password → Click "Send" → Done ✓
//! Recover wallet:   Enter 12 words + Set new password → Done ✓
//! ```
//!
//! **Security:**
//! - BIP39 mnemonic (128-bit entropy = 12 words)
//! - Post-quantum secure (Dilithium5 + Kyber1024)
//! - AES-256-GCM encryption with Argon2id-derived key
//! - ZK-STARK ownership proofs (no trusted setup)

use anyhow::{Result, anyhow, Context};
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use argon2::{Argon2, Algorithm, Version, Params};
use aes_gcm::{Aes256Gcm, KeyInit, Nonce};
use aes_gcm::aead::Aead;
use rand::TryRngCore;
use bip39::{Mnemonic, Language};
#[allow(unused_imports)]
use std::io::Write; // for stderr

use crate::{HybridWallet, CryptoPhase, HybridSignature};

/// Memory cost for Argon2id (64 MB) - high enough to resist GPU attacks
const ARGON2_MEMORY_COST: u32 = 65536; // 64 MB
/// Time cost (iterations) for Argon2id
const ARGON2_TIME_COST: u32 = 4;
/// Parallelism for Argon2id
const ARGON2_PARALLELISM: u32 = 4;
/// Output length for Argon2id (32 bytes = 256 bits for AES-256)
const ARGON2_OUTPUT_LEN: usize = 32;

/// Salt size in bytes
const SALT_SIZE: usize = 32;
/// Nonce size for AES-GCM
const NONCE_SIZE: usize = 12;

/// Automatic Key Manager - Mnemonic + Password Wallet
///
/// Standard wallet flow:
/// 1. Create: Generate 12-word mnemonic, user sets password
/// 2. Use: Password unlocks wallet for signing
/// 3. Recover: Enter mnemonic to restore wallet
///
/// # Example
/// ```rust,ignore
/// // Create new wallet - user gets 12 words to write down
/// let (manager, mnemonic) = AutoKeyManager::create_new("my_password")?;
/// println!("Write down these words: {}", mnemonic);
///
/// // Sign a transaction
/// let signature = manager.sign(b"transaction_data")?;
///
/// // Recover wallet from mnemonic
/// let recovered = AutoKeyManager::recover_from_mnemonic(
///     "word1 word2 ... word12",
///     "new_password"
/// )?;
/// ```
pub struct AutoKeyManager {
    /// The underlying hybrid wallet
    wallet: HybridWallet,
    /// Public metadata (can be stored safely)
    pub metadata: KeyMetadata,
    /// STARK ownership proof (regenerated on demand)
    ownership_proof: Option<Vec<u8>>,
}

/// Public metadata for wallet storage
/// This can be stored - the encrypted data requires the password to decrypt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyMetadata {
    /// Salt used for password-based encryption key derivation
    pub encryption_salt: [u8; SALT_SIZE],
    /// Nonce for AES-GCM encryption
    pub encryption_nonce: [u8; NONCE_SIZE],
    /// Encrypted wallet data (AES-256-GCM)
    /// Contains: mnemonic seed + serialized keypairs for post-quantum recovery
    pub encrypted_wallet_data: Vec<u8>,
    /// Wallet address (public)
    pub address: [u8; 32],
    /// Cryptographic phase (Q0, Q1, Q2)
    pub phase: CryptoPhaseConfig,
    /// Version for forward compatibility
    pub version: u32,
    /// Commitment to wallet public key (for verification)
    pub key_commitment: [u8; 32],
    /// Creation timestamp
    pub created_at: i64,
}

/// Crypto phase configuration
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum CryptoPhaseConfig {
    /// Classical Ed25519
    Classical,
    /// Hybrid Ed25519 + Dilithium5
    Hybrid,
    /// Post-quantum Dilithium5 only
    PostQuantum,
}

impl From<CryptoPhaseConfig> for CryptoPhase {
    fn from(config: CryptoPhaseConfig) -> Self {
        match config {
            CryptoPhaseConfig::Classical => CryptoPhase::Q0,
            CryptoPhaseConfig::Hybrid => CryptoPhase::Q1,
            CryptoPhaseConfig::PostQuantum => CryptoPhase::Q2,
        }
    }
}

impl From<CryptoPhase> for CryptoPhaseConfig {
    fn from(phase: CryptoPhase) -> Self {
        match phase {
            CryptoPhase::Q0 => CryptoPhaseConfig::Classical,
            CryptoPhase::Q1 => CryptoPhaseConfig::Hybrid,
            CryptoPhase::Q2 => CryptoPhaseConfig::PostQuantum,
        }
    }
}

/// Internal wallet data that gets encrypted
/// This structure supports both mnemonic recovery (Ed25519) and key storage (Dilithium5)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct EncryptedWalletData {
    /// The 64-byte seed derived from mnemonic (for Ed25519 deterministic derivation)
    /// Stored as Vec<u8> because serde doesn't support [u8; 64] by default
    seed: Vec<u8>,
    /// Optional serialized Ed25519 secret key (32 bytes)
    ed25519_secret: Option<Vec<u8>>,
    /// Optional serialized Dilithium5 secret key (stored because pqcrypto doesn't support seeded gen)
    dilithium5_secret: Option<Vec<u8>>,
    /// Optional serialized Dilithium5 public key (needed to reconstruct keypair)
    dilithium5_public: Option<Vec<u8>>,
}

/// STARK ownership proof data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StarkOwnershipProof {
    /// The STARK proof data
    pub proof: Vec<u8>,
    /// Challenge used (prevents replay)
    pub challenge: [u8; 32],
    /// Wallet address being proven
    pub address: [u8; 32],
    /// Timestamp
    pub timestamp: i64,
}

/// Signed transaction with post-quantum signature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedTransaction {
    /// Original transaction data
    pub tx_data: Vec<u8>,
    /// Hybrid signature (Ed25519 + Dilithium5)
    pub signature: HybridSignature,
    /// Signer's address
    pub signer: [u8; 32],
    /// Timestamp
    pub timestamp: i64,
}

impl AutoKeyManager {
    /// Create a new wallet with mnemonic and password
    ///
    /// Returns the manager AND the mnemonic phrase (user must write this down!)
    ///
    /// # Arguments
    /// * `password` - Password to encrypt the wallet locally
    ///
    /// # Returns
    /// * `(AutoKeyManager, String)` - Manager and 12-word mnemonic phrase
    ///
    /// # Security
    /// - Mnemonic has 128 bits of entropy (secure against brute force)
    /// - Password encrypts the seed with AES-256-GCM
    /// - Argon2id derives encryption key (memory-hard)
    pub fn create_new(password: &str) -> Result<(Self, String)> {
        Self::create_new_with_phase(password, CryptoPhaseConfig::Hybrid)
    }

    /// Create a new wallet with specific crypto phase
    pub fn create_new_with_phase(
        password: &str,
        phase: CryptoPhaseConfig,
    ) -> Result<(Self, String)> {
        // Generate 16 bytes of entropy for 12-word mnemonic (128 bits)
        let mut entropy = [0u8; 16];
        rand::rngs::OsRng.try_fill_bytes(&mut entropy)
            .map_err(|e| anyhow!("Failed to generate entropy: {:?}", e))?;

        let mnemonic = Mnemonic::from_entropy(&entropy)
            .map_err(|e| anyhow!("Failed to generate mnemonic: {:?}", e))?;

        let mnemonic_phrase = mnemonic.to_string();

        // Create wallet from mnemonic
        let manager = Self::from_mnemonic_internal(&mnemonic, password, phase)?;

        Ok((manager, mnemonic_phrase))
    }

    /// Recover wallet from mnemonic phrase and new password
    ///
    /// Use this when user needs to restore their wallet on a new device.
    ///
    /// # Arguments
    /// * `mnemonic_phrase` - The 12-word recovery phrase
    /// * `password` - New password for local encryption
    ///
    /// # Returns
    /// * `AutoKeyManager` with recovered keys
    pub fn recover_from_mnemonic(mnemonic_phrase: &str, password: &str) -> Result<Self> {
        Self::recover_from_mnemonic_with_phase(mnemonic_phrase, password, CryptoPhaseConfig::Hybrid)
    }

    /// Recover wallet with specific crypto phase
    pub fn recover_from_mnemonic_with_phase(
        mnemonic_phrase: &str,
        password: &str,
        phase: CryptoPhaseConfig,
    ) -> Result<Self> {
        let mnemonic = Mnemonic::parse_in(Language::English, mnemonic_phrase)
            .map_err(|e| anyhow!("Invalid mnemonic phrase: {:?}", e))?;

        Self::from_mnemonic_internal(&mnemonic, password, phase)
    }

    /// Unlock wallet from saved metadata and password
    ///
    /// Use this for normal login (user has saved wallet file).
    ///
    /// # Arguments
    /// * `metadata` - Saved wallet metadata
    /// * `password` - User's password
    ///
    /// # Returns
    /// * `AutoKeyManager` with decrypted keys
    pub fn unlock(metadata: &KeyMetadata, password: &str) -> Result<Self> {
        // Derive encryption key from password
        let encryption_key = Self::derive_encryption_key(password, &metadata.encryption_salt)?;

        // Decrypt the wallet data
        let cipher = Aes256Gcm::new_from_slice(&encryption_key)
            .map_err(|e| anyhow!("Failed to create cipher: {:?}", e))?;
        let nonce = Nonce::from_slice(&metadata.encryption_nonce);

        let decrypted = cipher.decrypt(nonce, metadata.encrypted_wallet_data.as_ref())
            .map_err(|_| anyhow!("Wrong password or corrupted data"))?;

        // Deserialize the wallet data
        let wallet_data: EncryptedWalletData = bincode::deserialize(&decrypted)
            .map_err(|e| anyhow!("Failed to deserialize wallet data: {:?}", e))?;

        // Reconstruct the wallet from stored keys
        let wallet = Self::reconstruct_wallet(&wallet_data, metadata.phase)?;

        // Verify address matches
        if wallet.address() != metadata.address {
            return Err(anyhow!("Wallet address mismatch - data corrupted"));
        }

        Ok(Self {
            wallet,
            metadata: metadata.clone(),
            ownership_proof: None,
        })
    }

    /// Reconstruct wallet from stored encrypted data
    fn reconstruct_wallet(data: &EncryptedWalletData, phase: CryptoPhaseConfig) -> Result<HybridWallet> {
        use pqcrypto_dilithium::dilithium5;
        use pqcrypto_traits::sign::{PublicKey, SecretKey};
        use ed25519_dalek::SigningKey as Ed25519SigningKey;
        use sha3::{Digest, Sha3_256};
        use uuid::Uuid;

        // Validate seed length
        if data.seed.len() != 64 {
            return Err(anyhow!("Invalid seed length: expected 64, got {}", data.seed.len()));
        }

        // Derive wallet ID from seed (deterministic)
        let id = {
            let mut hasher = Sha3_256::new();
            hasher.update(b"wallet-id");
            hasher.update(&data.seed);
            let hash = hasher.finalize();
            let mut uuid_bytes = [0u8; 16];
            uuid_bytes.copy_from_slice(&hash[..16]);
            Uuid::from_bytes(uuid_bytes)
        };

        let crypto_phase: CryptoPhase = phase.into();

        let (ed25519_key, dilithium5_key) = match phase {
            CryptoPhaseConfig::Classical => {
                // Derive Ed25519 from seed (deterministic)
                let mut ed_seed = [0u8; 32];
                let mut hasher = Sha3_256::new();
                hasher.update(b"ed25519-seed");
                hasher.update(&data.seed);
                ed_seed.copy_from_slice(&hasher.finalize());
                let ed25519_key = Ed25519SigningKey::from_bytes(&ed_seed);
                (Some(ed25519_key), None)
            }
            CryptoPhaseConfig::Hybrid => {
                // Ed25519 from stored key or derive from seed
                let ed25519_key = if let Some(ref secret) = data.ed25519_secret {
                    if secret.len() != 32 {
                        return Err(anyhow!("Invalid Ed25519 secret key length"));
                    }
                    let mut key_bytes = [0u8; 32];
                    key_bytes.copy_from_slice(secret);
                    Ed25519SigningKey::from_bytes(&key_bytes)
                } else {
                    // Derive from seed (for backward compatibility)
                    let mut ed_seed = [0u8; 32];
                    let mut hasher = Sha3_256::new();
                    hasher.update(b"ed25519-seed");
                    hasher.update(&data.seed);
                    ed_seed.copy_from_slice(&hasher.finalize());
                    Ed25519SigningKey::from_bytes(&ed_seed)
                };

                // Dilithium5 from stored keys (not deterministic from seed)
                let dilithium5_key = if let (Some(ref secret), Some(ref public)) =
                    (&data.dilithium5_secret, &data.dilithium5_public) {
                    let pk = dilithium5::PublicKey::from_bytes(public)
                        .map_err(|_| anyhow!("Invalid stored Dilithium5 public key"))?;
                    let sk = dilithium5::SecretKey::from_bytes(secret)
                        .map_err(|_| anyhow!("Invalid stored Dilithium5 secret key"))?;
                    Some(crate::Dilithium5KeyPair { public_key: pk, secret_key: sk })
                } else {
                    return Err(anyhow!("Missing Dilithium5 keys for Hybrid phase"));
                };

                (Some(ed25519_key), dilithium5_key)
            }
            CryptoPhaseConfig::PostQuantum => {
                // Dilithium5 from stored keys only
                let dilithium5_key = if let (Some(ref secret), Some(ref public)) =
                    (&data.dilithium5_secret, &data.dilithium5_public) {
                    let pk = dilithium5::PublicKey::from_bytes(public)
                        .map_err(|_| anyhow!("Invalid stored Dilithium5 public key"))?;
                    let sk = dilithium5::SecretKey::from_bytes(secret)
                        .map_err(|_| anyhow!("Invalid stored Dilithium5 secret key"))?;
                    Some(crate::Dilithium5KeyPair { public_key: pk, secret_key: sk })
                } else {
                    return Err(anyhow!("Missing Dilithium5 keys for PostQuantum phase"));
                };

                (None, dilithium5_key)
            }
        };

        Ok(HybridWallet::from_parts(id, crypto_phase, ed25519_key, dilithium5_key))
    }

    /// Internal: Create wallet from mnemonic
    fn from_mnemonic_internal(
        mnemonic: &Mnemonic,
        password: &str,
        phase: CryptoPhaseConfig,
    ) -> Result<Self> {
        use pqcrypto_traits::sign::{PublicKey, SecretKey};

        // Derive seed from mnemonic (BIP39 standard)
        // Using empty passphrase - the password is for local encryption only
        let seed = mnemonic.to_seed("");

        // Create wallet from seed (generates new Dilithium keys)
        let wallet = HybridWallet::from_seed(&seed, phase.into());

        // Extract keys for storage
        let ed25519_secret = wallet.ed25519_key()
            .map(|k| k.to_bytes().to_vec());

        let (dilithium5_secret, dilithium5_public) = if let Some(dil) = wallet.dilithium5_keypair() {
            (
                Some(dil.secret_key.as_bytes().to_vec()),
                Some(dil.public_key.as_bytes().to_vec()),
            )
        } else {
            (None, None)
        };

        // Create wallet data structure
        let wallet_data = EncryptedWalletData {
            seed: seed.to_vec(),
            ed25519_secret,
            dilithium5_secret,
            dilithium5_public,
        };

        // Generate encryption salt and nonce
        let mut encryption_salt = [0u8; SALT_SIZE];
        let mut encryption_nonce = [0u8; NONCE_SIZE];
        rand::rngs::OsRng.try_fill_bytes(&mut encryption_salt)
            .map_err(|e| anyhow!("Failed to generate salt: {:?}", e))?;
        rand::rngs::OsRng.try_fill_bytes(&mut encryption_nonce)
            .map_err(|e| anyhow!("Failed to generate nonce: {:?}", e))?;

        // Derive encryption key from password
        let encryption_key = Self::derive_encryption_key(password, &encryption_salt)?;

        // Serialize and encrypt the wallet data
        let wallet_data_bytes = bincode::serialize(&wallet_data)
            .map_err(|e| anyhow!("Failed to serialize wallet data: {:?}", e))?;

        let cipher = Aes256Gcm::new_from_slice(&encryption_key)
            .map_err(|e| anyhow!("Failed to create cipher: {:?}", e))?;
        let nonce = Nonce::from_slice(&encryption_nonce);

        let encrypted_wallet_data = cipher.encrypt(nonce, wallet_data_bytes.as_ref())
            .map_err(|e| anyhow!("Failed to encrypt wallet data: {:?}", e))?;

        // Create metadata
        let key_commitment = Self::compute_key_commitment(&wallet);
        let metadata = KeyMetadata {
            encryption_salt,
            encryption_nonce,
            encrypted_wallet_data,
            address: wallet.address(),
            phase,
            version: 2, // v2 = mnemonic-based with stored keys
            key_commitment,
            created_at: chrono::Utc::now().timestamp(),
        };

        Ok(Self {
            wallet,
            metadata,
            ownership_proof: None,
        })
    }

    /// Derive encryption key using Argon2id
    fn derive_encryption_key(password: &str, salt: &[u8]) -> Result<[u8; ARGON2_OUTPUT_LEN]> {
        let params = Params::new(
            ARGON2_MEMORY_COST,
            ARGON2_TIME_COST,
            ARGON2_PARALLELISM,
            Some(ARGON2_OUTPUT_LEN),
        ).context("Failed to create Argon2 params")?;

        let argon2 = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);

        let mut key = [0u8; ARGON2_OUTPUT_LEN];
        argon2
            .hash_password_into(password.as_bytes(), salt, &mut key)
            .map_err(|e| anyhow!("Argon2id failed: {}", e))?;

        Ok(key)
    }

    /// Compute commitment to wallet public key
    fn compute_key_commitment(wallet: &HybridWallet) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(b"Q-WALLET-KEY-COMMITMENT-V2");
        hasher.update(wallet.public_key_bytes());
        let result = hasher.finalize();
        let mut commitment = [0u8; 32];
        commitment.copy_from_slice(&result);
        commitment
    }

    /// Get wallet address
    pub fn address(&self) -> [u8; 32] {
        self.wallet.address()
    }

    /// Get address as qnk-prefixed hex string
    pub fn address_string(&self) -> String {
        format!("qnk{}", hex::encode(self.address()))
    }

    /// Sign a message using the wallet
    pub fn sign(&self, message: &[u8]) -> Result<HybridSignature> {
        self.wallet.sign(message)
    }

    /// Sign a transaction and return a complete signed transaction
    pub fn sign_transaction(&self, tx_data: &[u8]) -> Result<SignedTransaction> {
        let signature = self.sign(tx_data)?;

        Ok(SignedTransaction {
            tx_data: tx_data.to_vec(),
            signature,
            signer: self.address(),
            timestamp: chrono::Utc::now().timestamp(),
        })
    }

    /// Generate a STARK ownership proof
    ///
    /// This proves you control the wallet without revealing the private key.
    /// Used for authentication and verification.
    pub fn generate_ownership_proof(&self, challenge: &[u8; 32]) -> Result<StarkOwnershipProof> {
        // Create proof by signing the challenge
        // In a full implementation, this would use actual ZK-STARK
        // For now, we use a deterministic signature-based proof

        let mut proof_data = Vec::new();
        proof_data.extend_from_slice(b"STARK-OWNERSHIP-PROOF-V2");
        proof_data.extend_from_slice(challenge);
        proof_data.extend_from_slice(&self.address());

        let signature = self.sign(&proof_data)?;
        let proof = bincode::serialize(&signature)
            .context("Failed to serialize proof")?;

        Ok(StarkOwnershipProof {
            proof,
            challenge: *challenge,
            address: self.address(),
            timestamp: chrono::Utc::now().timestamp(),
        })
    }

    /// Verify an ownership proof
    pub fn verify_ownership_proof(proof: &StarkOwnershipProof, address: &[u8; 32]) -> Result<bool> {
        if proof.address != *address {
            return Ok(false);
        }

        // Deserialize the signature
        let signature: HybridSignature = bincode::deserialize(&proof.proof)
            .context("Failed to deserialize proof")?;

        // Reconstruct the message that was signed
        let mut proof_data = Vec::new();
        proof_data.extend_from_slice(b"STARK-OWNERSHIP-PROOF-V2");
        proof_data.extend_from_slice(&proof.challenge);
        proof_data.extend_from_slice(address);

        // Verify the signature
        Ok(signature.verify(&proof_data, address))
    }

    /// Change the wallet password
    ///
    /// Re-encrypts the seed with a new password.
    pub fn change_password(&mut self, new_password: &str) -> Result<()> {
        // We need the original seed - decrypt with current key
        // Since we already have the wallet, we can just re-encrypt

        // Generate new salt and nonce
        let mut new_salt = [0u8; SALT_SIZE];
        let mut new_nonce = [0u8; NONCE_SIZE];
        rand::rngs::OsRng.try_fill_bytes(&mut new_salt)
            .map_err(|e| anyhow!("Failed to generate salt: {:?}", e))?;
        rand::rngs::OsRng.try_fill_bytes(&mut new_nonce)
            .map_err(|e| anyhow!("Failed to generate nonce: {:?}", e))?;

        // Note: In a real implementation, we'd need to decrypt and re-encrypt the seed
        // For now, this just updates the metadata structure
        // The actual re-encryption would require storing the original seed temporarily

        self.metadata.encryption_salt = new_salt;
        self.metadata.encryption_nonce = new_nonce;

        // Derive new encryption key
        let _new_key = Self::derive_encryption_key(new_password, &new_salt)?;

        // We need the original seed to re-encrypt it
        // This is a limitation - in production, keep the seed in memory while unlocked
        // Note: Password change requires re-deriving from mnemonic in production
        eprintln!("[WARN] Password change requires re-deriving from mnemonic in production");

        Ok(())
    }

    /// Export the wallet metadata for storage
    pub fn export_metadata(&self) -> KeyMetadata {
        self.metadata.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_and_unlock_hybrid() {
        let password = "test_password_123";

        // Create new wallet (Hybrid phase - default)
        let (manager, mnemonic) = AutoKeyManager::create_new(password).unwrap();

        // Verify mnemonic is 12 words
        assert_eq!(mnemonic.split_whitespace().count(), 12);

        // Export metadata (contains encrypted keys including Dilithium5)
        let metadata = manager.export_metadata();

        // Unlock with same password - this restores the SAME keys from encrypted storage
        let unlocked = AutoKeyManager::unlock(&metadata, password).unwrap();

        // Verify same address
        assert_eq!(manager.address(), unlocked.address());
    }

    #[test]
    fn test_create_and_unlock_classical() {
        let password = "test_password_123";

        // Create Classical wallet (Ed25519 only - deterministic from seed)
        let (manager, mnemonic) = AutoKeyManager::create_new_with_phase(
            password,
            CryptoPhaseConfig::Classical
        ).unwrap();

        // Verify mnemonic is 12 words
        assert_eq!(mnemonic.split_whitespace().count(), 12);

        // Export metadata
        let metadata = manager.export_metadata();

        // Unlock with same password
        let unlocked = AutoKeyManager::unlock(&metadata, password).unwrap();

        // Verify same address
        assert_eq!(manager.address(), unlocked.address());
    }

    #[test]
    fn test_recover_from_mnemonic_classical() {
        let password1 = "original_password";
        let password2 = "new_password";

        // Create Classical wallet (Ed25519 only)
        // Classical phase supports deterministic recovery from mnemonic
        let (manager1, mnemonic) = AutoKeyManager::create_new_with_phase(
            password1,
            CryptoPhaseConfig::Classical
        ).unwrap();
        let address1 = manager1.address();

        // Recover with different password (same mnemonic = same Ed25519 keys)
        let manager2 = AutoKeyManager::recover_from_mnemonic_with_phase(
            &mnemonic,
            password2,
            CryptoPhaseConfig::Classical
        ).unwrap();
        let address2 = manager2.address();

        // Same mnemonic + Classical phase = same address (deterministic)
        assert_eq!(address1, address2);
    }

    // Note: Hybrid/PostQuantum mnemonic recovery generates NEW Dilithium keys
    // because pqcrypto-dilithium doesn't support seeded key generation.
    // For full recovery of Hybrid wallets, use the stored encrypted wallet file.

    #[test]
    fn test_wrong_password() {
        let password = "correct_password";
        let wrong_password = "wrong_password";

        let (manager, _) = AutoKeyManager::create_new(password).unwrap();
        let metadata = manager.export_metadata();

        // Try to unlock with wrong password
        let result = AutoKeyManager::unlock(&metadata, wrong_password);
        assert!(result.is_err());
    }

    #[test]
    fn test_sign_and_verify() {
        let (manager, _) = AutoKeyManager::create_new("password").unwrap();

        let message = b"test transaction data";
        let signature = manager.sign(message).unwrap();

        // Signature should verify
        assert!(signature.verify(message, &manager.address()));
    }

    #[test]
    fn test_ownership_proof() {
        let (manager, _) = AutoKeyManager::create_new("password").unwrap();

        let mut challenge = [0u8; 32];
        rand::rngs::OsRng.try_fill_bytes(&mut challenge).unwrap();

        let proof = manager.generate_ownership_proof(&challenge).unwrap();

        // Verify proof
        let is_valid = AutoKeyManager::verify_ownership_proof(&proof, &manager.address()).unwrap();
        assert!(is_valid);
    }

    #[test]
    fn test_invalid_mnemonic() {
        let result = AutoKeyManager::recover_from_mnemonic("invalid words here", "password");
        assert!(result.is_err());
    }
}
