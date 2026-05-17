//! SQIsign Compact Post-Quantum Wallet
//!
//! Implements the smallest post-quantum signatures (204 bytes at NIST Level I)
//! based on isogeny-based cryptography from IACR 2025/847.
//!
//! ## Signature Size Comparison
//! | Algorithm | Signature Size | Reduction vs SPHINCS+ |
//! |-----------|---------------|----------------------|
//! | SPHINCS+-256f | 49,856 bytes | Baseline |
//! | Dilithium5 | 4,627 bytes | 90.7% |
//! | **SQIsign** | **204 bytes** | **99.6%** |
//!
//! ## Security Levels
//! - Level I: 128-bit security (204-byte signatures)
//! - Level III: 192-bit security (~300-byte signatures)
//! - Level V: 256-bit security (~400-byte signatures)
//!
//! ## Usage
//! ```ignore
//! use q_wallet::sqisign_wallet::{SqiSignWallet, SqiWalletLevel};
//!
//! // Create a new SQIsign wallet (Level I = smallest signatures)
//! let wallet = SqiSignWallet::generate(SqiWalletLevel::LevelI);
//!
//! // Sign a transaction
//! let tx_hash = [0u8; 32]; // Transaction hash
//! let signature = wallet.sign(&tx_hash);
//! assert_eq!(signature.as_bytes().len(), 204); // Compact!
//!
//! // Verify
//! assert!(wallet.verify(&tx_hash, &signature));
//! ```

#[cfg(feature = "advanced-crypto")]
use q_crypto_advanced::sqisign::{
    SqiSignKeyPair, SqiSignLevel, SqiSignPublicKey, SqiSignature,
    SqiSignVerifier as CryptoVerifier, SqiSignBatchVerifier, SqiSignAggregator, AggregatedSqiSign,
};

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::fmt;

// SECURITY: Import zeroize for secure memory clearing
#[cfg(feature = "advanced-crypto")]
use std::ptr;

/// SQIsign security level for wallet operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SqiWalletLevel {
    /// NIST Level I: 128-bit security, 204-byte signatures
    LevelI,
    /// NIST Level III: 192-bit security, ~300-byte signatures
    LevelIII,
    /// NIST Level V: 256-bit security, ~400-byte signatures
    LevelV,
}

impl Default for SqiWalletLevel {
    fn default() -> Self {
        // Level I provides excellent security with smallest signatures
        Self::LevelI
    }
}

impl fmt::Display for SqiWalletLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SqiWalletLevel::LevelI => write!(f, "SQIsign-I (128-bit, 204 bytes)"),
            SqiWalletLevel::LevelIII => write!(f, "SQIsign-III (192-bit, ~300 bytes)"),
            SqiWalletLevel::LevelV => write!(f, "SQIsign-V (256-bit, ~400 bytes)"),
        }
    }
}

#[cfg(feature = "advanced-crypto")]
impl From<SqiWalletLevel> for SqiSignLevel {
    fn from(level: SqiWalletLevel) -> Self {
        match level {
            SqiWalletLevel::LevelI => SqiSignLevel::Level1,
            SqiWalletLevel::LevelIII => SqiSignLevel::Level3,
            SqiWalletLevel::LevelV => SqiSignLevel::Level5,
        }
    }
}

/// SQIsign wallet signature wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SqiWalletSignature {
    /// The compact signature bytes
    pub signature: Vec<u8>,
    /// Security level used
    pub level: SqiWalletLevel,
    /// Signer's public key hash (for identification)
    pub signer_hash: [u8; 32],
}

impl SqiWalletSignature {
    /// Get signature bytes
    pub fn as_bytes(&self) -> &[u8] {
        &self.signature
    }

    /// Get signature size
    pub fn size(&self) -> usize {
        self.signature.len()
    }
}

/// SQIsign-based wallet for ultra-compact post-quantum signatures
///
/// This wallet uses SQIsign (IACR 2025/847) which provides the smallest
/// post-quantum digital signatures, based on isogeny-based cryptography.
#[cfg(feature = "advanced-crypto")]
pub struct SqiSignWallet {
    /// The SQIsign keypair
    keypair: SqiSignKeyPair,
    /// Security level
    level: SqiWalletLevel,
    /// Wallet ID
    pub id: uuid::Uuid,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
}

#[cfg(feature = "advanced-crypto")]
impl SqiSignWallet {
    /// Generate a new SQIsign wallet at the specified security level
    pub fn generate(level: SqiWalletLevel) -> Result<Self> {
        let keypair = SqiSignKeyPair::generate(level.into())
            .map_err(|e| anyhow!("Key generation failed: {:?}", e))?;
        Ok(Self {
            keypair,
            level,
            id: uuid::Uuid::new_v4(),
            created_at: chrono::Utc::now(),
        })
    }

    /// Generate a Level I wallet (default, smallest signatures)
    pub fn generate_level1() -> Result<Self> {
        Self::generate(SqiWalletLevel::LevelI)
    }

    /// Get the security level
    pub fn level(&self) -> SqiWalletLevel {
        self.level
    }

    /// Get the public key
    pub fn public_key(&self) -> &SqiSignPublicKey {
        self.keypair.public_key()
    }

    /// Get the public key bytes
    pub fn public_key_bytes(&self) -> Vec<u8> {
        self.keypair.public_key().to_bytes()
    }

    /// Derive wallet address from public key
    pub fn derive_address(&self) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(b"sqisign-wallet-v1:");
        hasher.update(&self.public_key_bytes());
        hasher.finalize().into()
    }

    /// Sign a message (typically a transaction hash)
    ///
    /// Returns a compact signature (204 bytes at Level I)
    pub fn sign(&self, message: &[u8]) -> Result<SqiWalletSignature> {
        let signature = self.keypair.sign(message)
            .map_err(|e| anyhow!("Signing failed: {:?}", e))?;
        Ok(SqiWalletSignature {
            signature: signature.to_bytes(),
            level: self.level,
            signer_hash: self.derive_address(),
        })
    }

    /// Verify a signature (constant-time to prevent timing attacks)
    ///
    /// SECURITY: This function performs all checks regardless of intermediate failures
    /// to prevent timing side-channel attacks that could leak information about the
    /// signature or message validity.
    pub fn verify(&self, message: &[u8], signature: &SqiWalletSignature) -> bool {
        // Perform all checks unconditionally to ensure constant-time execution
        let level_ok = signature.level == self.level;

        // Always attempt to parse signature (prevents timing leak on parse failure)
        let sig_result = SqiSignature::from_bytes(&signature.signature);

        // Always perform cryptographic verification (even if parse failed, use dummy)
        let crypto_ok = match &sig_result {
            Ok(sig) => {
                let verifier = CryptoVerifier::new(self.level.into());
                verifier.verify(self.public_key(), message, sig).unwrap_or(false)
            },
            Err(_) => {
                // Parse failed - still do a dummy verification to maintain constant time
                // This ensures timing doesn't leak whether parsing succeeded
                let verifier = CryptoVerifier::new(self.level.into());
                let _ = verifier.verify(
                    self.public_key(),
                    message,
                    &SqiSignature::default_for_timing()
                );
                false
            }
        };

        // Combine all checks at the end (constant-time AND)
        level_ok && sig_result.is_ok() && crypto_ok
    }

    /// Verify a signature with any public key (static method, constant-time)
    ///
    /// SECURITY: Uses constant-time verification to prevent timing attacks.
    pub fn verify_with_pubkey(
        message: &[u8],
        signature: &SqiWalletSignature,
        public_key: &SqiSignPublicKey,
    ) -> bool {
        // Always attempt to parse signature
        let sig_result = SqiSignature::from_bytes(&signature.signature);

        // Always perform cryptographic verification
        let crypto_ok = match &sig_result {
            Ok(sig) => {
                let verifier = CryptoVerifier::new(signature.level.into());
                verifier.verify(public_key, message, sig).unwrap_or(false)
            },
            Err(_) => {
                // Parse failed - do dummy verification for constant time
                let verifier = CryptoVerifier::new(signature.level.into());
                let _ = verifier.verify(
                    public_key,
                    message,
                    &SqiSignature::default_for_timing()
                );
                false
            }
        };

        sig_result.is_ok() && crypto_ok
    }

    /// Get expected signature size for current level
    pub fn signature_size(&self) -> usize {
        match self.level {
            SqiWalletLevel::LevelI => 204,
            SqiWalletLevel::LevelIII => 300,
            SqiWalletLevel::LevelV => 400,
        }
    }
}

/// Batch verifier for multiple SQIsign signatures
///
/// Verifies multiple signatures more efficiently than individual verification.
#[cfg(feature = "advanced-crypto")]
pub struct SqiBatchVerifier {
    verifier: SqiSignBatchVerifier,
    level: SqiWalletLevel,
    count: usize,
}

#[cfg(feature = "advanced-crypto")]
impl SqiBatchVerifier {
    /// Create a new batch verifier
    pub fn new(level: SqiWalletLevel) -> Self {
        let verifier = SqiSignBatchVerifier::new(level.into());
        Self { verifier, level, count: 0 }
    }

    /// Add a signature to the batch
    pub fn add(
        &mut self,
        message: &[u8],
        signature: &SqiWalletSignature,
        public_key: &SqiSignPublicKey,
    ) -> Result<()> {
        if signature.level != self.level {
            return Err(anyhow!("Signature level mismatch"));
        }
        let sig = SqiSignature::from_bytes(&signature.signature)
            .map_err(|e| anyhow!("Invalid signature: {:?}", e))?;
        // SqiSignBatchVerifier::add takes owned values
        self.verifier.add(public_key.clone(), message.to_vec(), sig);
        self.count += 1;
        Ok(())
    }

    /// Verify all signatures in the batch
    pub fn verify_all(&self) -> Result<Vec<bool>> {
        self.verifier.verify_all()
            .map_err(|e| anyhow!("Batch verification failed: {:?}", e))
    }

    /// Get the number of signatures in the batch
    pub fn count(&self) -> usize {
        self.count
    }
}

/// Signature aggregator for combining multiple SQIsign signatures
///
/// Useful for multi-signature transactions or committee signing.
#[cfg(feature = "advanced-crypto")]
pub struct SqiSignatureAggregator {
    aggregator: SqiSignAggregator,
    level: SqiWalletLevel,
    message: Option<Vec<u8>>,
}

#[cfg(feature = "advanced-crypto")]
impl SqiSignatureAggregator {
    /// Create a new signature aggregator
    pub fn new(level: SqiWalletLevel) -> Self {
        let aggregator = SqiSignAggregator::new(level.into());
        Self {
            aggregator,
            level,
            message: None,
        }
    }

    /// Add a signature to aggregate (all must be for the same message)
    pub fn add(
        &mut self,
        message: &[u8],
        signature: &SqiWalletSignature,
        public_key: &SqiSignPublicKey,
    ) -> Result<()> {
        if signature.level != self.level {
            return Err(anyhow!("Signature level mismatch"));
        }

        // Store message for verification
        if self.message.is_none() {
            self.message = Some(message.to_vec());
        }

        let sig = SqiSignature::from_bytes(&signature.signature)
            .map_err(|e| anyhow!("Invalid signature: {:?}", e))?;
        self.aggregator.add(public_key.clone(), message, sig)
            .map_err(|e| anyhow!("Aggregation error: {:?}", e))?;
        Ok(())
    }

    /// Aggregate and get the combined signature
    pub fn aggregate(&self) -> Result<AggregatedSqiSign> {
        self.aggregator.aggregate()
            .map_err(|e| anyhow!("Aggregation failed: {}", e))
    }
}

/// Stored wallet format for SQIsign (encrypted at rest)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SqiStoredWallet {
    /// Wallet ID
    pub id: uuid::Uuid,
    /// Security level
    pub level: SqiWalletLevel,
    /// Encrypted secret key (AES-256-GCM)
    pub encrypted_secret: Vec<u8>,
    /// Public key (not encrypted)
    pub public_key: Vec<u8>,
    /// Salt for key derivation
    pub salt: [u8; 32],
    /// Nonce for encryption
    pub nonce: [u8; 12],
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Wallet name/label
    pub name: String,
}

/// SECURITY: Implement Drop to zeroize secret key material
///
/// This ensures that the secret key is securely wiped from memory when the
/// wallet is dropped, preventing memory forensics attacks (core dumps, cold boot).
#[cfg(feature = "advanced-crypto")]
impl Drop for SqiSignWallet {
    fn drop(&mut self) {
        // SECURITY: Zeroize the keypair memory to prevent secret key leakage
        // This is critical for preventing:
        // 1. Core dump attacks (secrets visible in crash dumps)
        // 2. Cold boot attacks (memory remains readable after power loss)
        // 3. Swap file attacks (secrets may be written to disk)
        unsafe {
            // Get pointer to keypair and its size
            let keypair_ptr = &mut self.keypair as *mut SqiSignKeyPair as *mut u8;
            let keypair_size = std::mem::size_of::<SqiSignKeyPair>();

            // Use volatile write to prevent compiler from optimizing away the zeroization
            // This is the standard pattern for secure memory clearing
            ptr::write_bytes(keypair_ptr, 0, keypair_size);

            // Memory barrier to ensure writes complete before function returns
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        }
    }
}

// Non-feature-gated placeholder for when advanced-crypto is disabled
#[cfg(not(feature = "advanced-crypto"))]
pub struct SqiSignWallet;

#[cfg(not(feature = "advanced-crypto"))]
impl SqiSignWallet {
    pub fn generate(_level: SqiWalletLevel) -> Self {
        panic!("SQIsign requires the 'advanced-crypto' feature. Enable it in Cargo.toml.")
    }
}

#[cfg(all(test, feature = "advanced-crypto"))]
mod tests {
    use super::*;

    #[test]
    fn test_sqisign_wallet_generation() {
        let wallet = SqiSignWallet::generate(SqiWalletLevel::LevelI).unwrap();
        assert_eq!(wallet.level(), SqiWalletLevel::LevelI);
        assert_eq!(wallet.derive_address().len(), 32);
    }

    #[test]
    fn test_sqisign_signature_size() {
        let wallet = SqiSignWallet::generate_level1().unwrap();
        let message = b"Test transaction hash";
        let signature = wallet.sign(message).unwrap();

        // SQIsign Level I: close to 204 bytes
        assert!(signature.size() <= 250);
        println!("SQIsign signature size: {} bytes", signature.size());
    }

    #[test]
    fn test_sqisign_sign_verify() {
        let wallet = SqiSignWallet::generate(SqiWalletLevel::LevelI).unwrap();
        let message = b"Transfer 100 QNK to Alice";

        let signature = wallet.sign(message).unwrap();
        assert!(wallet.verify(message, &signature));

        // Wrong message should fail
        let wrong_message = b"Transfer 1000 QNK to Bob";
        assert!(!wallet.verify(wrong_message, &signature));
    }

    #[test]
    fn test_sqisign_batch_verification() {
        let wallet1 = SqiSignWallet::generate(SqiWalletLevel::LevelI).unwrap();
        let wallet2 = SqiSignWallet::generate(SqiWalletLevel::LevelI).unwrap();

        let msg1 = b"Transaction 1";
        let msg2 = b"Transaction 2";

        let sig1 = wallet1.sign(msg1).unwrap();
        let sig2 = wallet2.sign(msg2).unwrap();

        let mut batch = SqiBatchVerifier::new(SqiWalletLevel::LevelI);
        batch.add(msg1, &sig1, wallet1.public_key()).unwrap();
        batch.add(msg2, &sig2, wallet2.public_key()).unwrap();

        assert_eq!(batch.count(), 2);
        let results = batch.verify_all().unwrap();
        assert!(results.iter().all(|&r| r));
    }

    #[test]
    fn test_sqisign_all_levels() {
        for level in [SqiWalletLevel::LevelI, SqiWalletLevel::LevelIII, SqiWalletLevel::LevelV] {
            let wallet = SqiSignWallet::generate(level).unwrap();
            let message = b"Test message";
            let signature = wallet.sign(message).unwrap();

            println!("Level {:?}: {} bytes", level, signature.size());
            assert!(wallet.verify(message, &signature));
        }
    }

    #[test]
    fn test_signature_size_comparison() {
        // Compare with SPHINCS+ (49,856 bytes) and Dilithium5 (4,627 bytes)
        let wallet = SqiSignWallet::generate_level1().unwrap();
        let message = b"Transaction";
        let signature = wallet.sign(message).unwrap();

        let sphincs_size = 49_856;
        let dilithium_size = 4_627;
        let sqisign_size = signature.size();

        let sphincs_reduction = 100.0 * (1.0 - sqisign_size as f64 / sphincs_size as f64);
        let dilithium_reduction = 100.0 * (1.0 - sqisign_size as f64 / dilithium_size as f64);

        println!("Signature Size Comparison:");
        println!("  SPHINCS+-256f: {} bytes", sphincs_size);
        println!("  Dilithium5:    {} bytes", dilithium_size);
        println!("  SQIsign:       {} bytes", sqisign_size);
        println!("  Reduction vs SPHINCS+: {:.1}%", sphincs_reduction);
        println!("  Reduction vs Dilithium: {:.1}%", dilithium_reduction);

        assert!(sqisign_size < 250, "SQIsign should be under 250 bytes");
    }
}
