//! Q-Wallet: Post-Quantum Crypto-Agile Wallet Module
//!
//! This module provides quantum-resistant wallet implementations supporting multiple
//! cryptographic phases:
//! - Q0 (Classical): Ed25519
//! - Q1 (Hybrid): Ed25519 + Dilithium5
//! - Q2 (Post-Quantum): Dilithium5 only
//!
//! Key features:
//! - Multi-phase cryptographic support
//! - Post-quantum key encapsulation with Kyber1024
//! - SPHINCS+ alternative signatures
//! - Secure key storage with AES-256-GCM encryption

use anyhow::Result;

// Encryption constants
pub const SALT_SIZE: usize = 32;
pub const NONCE_SIZE: usize = 12;

// Export all wallet implementations
pub mod dilithium_wallet;
pub mod kyber_wallet;
pub mod sphincs_wallet;
pub mod hybrid_wallet;

// ✨ v2.3.5-beta: Automatic key management with ZK-STARK proofs
// Users only need a password - everything else is automatic
pub mod auto_key_manager;

// ✨ v1.0.58-beta: SQIsign compact signatures (IACR 2025/847)
// Smallest post-quantum signatures: 204 bytes vs 49KB SPHINCS+
#[cfg(feature = "advanced-crypto")]
pub mod sqisign_wallet;

// Re-export key types for convenience
pub use dilithium_wallet::{Dilithium5KeyPair, Dilithium5StoredWallet};
pub use kyber_wallet::{Kyber1024KeyPair, Kyber1024Ciphertext, KyberHybridEncryption};
// v2.4.9-beta: True hybrid encryption (X25519 + Kyber1024) - defense-in-depth
pub use kyber_wallet::{TrueHybridKeypair, TrueHybridEncryption};
pub use hybrid_wallet::{HybridWallet, HybridStoredWallet, HybridSignature, CryptoPhase};

// ✨ v2.3.5-beta: Automatic key management - "just works" wallet creation
pub use auto_key_manager::{AutoKeyManager, KeyMetadata, StarkOwnershipProof, SignedTransaction};

// ✨ v1.0.58-beta: SQIsign compact signatures (99.6% smaller than SPHINCS+)
#[cfg(feature = "advanced-crypto")]
pub use sqisign_wallet::{
    SqiSignWallet, SqiWalletLevel, SqiWalletSignature, SqiStoredWallet,
    SqiBatchVerifier, SqiSignatureAggregator,
};

/// Main wallet interface for the Q-NarwhalKnight system
pub struct QWallet {
    pub hybrid: HybridWallet,
}

impl QWallet {
    /// Create a new wallet for the specified cryptographic phase
    pub fn new(phase: CryptoPhase) -> Self {
        Self {
            hybrid: HybridWallet::generate(phase),
        }
    }

    /// Create a Q0 (classical) wallet with Ed25519
    pub fn new_q0() -> Self {
        Self::new(CryptoPhase::Q0)
    }

    /// Create a Q1 (hybrid) wallet with Ed25519 + Dilithium5
    pub fn new_q1() -> Self {
        Self::new(CryptoPhase::Q1)
    }

    /// Create a Q2 (post-quantum) wallet with Dilithium5
    pub fn new_q2() -> Self {
        Self::new(CryptoPhase::Q2)
    }

    /// Sign a message using the wallet's cryptographic phase
    pub fn sign(&self, message: &[u8]) -> Result<HybridSignature> {
        self.hybrid.sign(message)
    }

    /// Get the wallet's address
    pub fn address(&self) -> [u8; 32] {
        self.hybrid.derive_address()
    }

    /// Get the wallet ID
    pub fn id(&self) -> uuid::Uuid {
        self.hybrid.id
    }

    /// Get the current cryptographic phase
    pub fn phase(&self) -> CryptoPhase {
        self.hybrid.phase
    }
}

// API-compatible exports for existing code
pub struct WalletManager {
    default_phase: CryptoPhase,
}

impl WalletManager {
    /// Create a new wallet manager with default phase Q1 (hybrid)
    pub fn new() -> Self {
        Self {
            default_phase: CryptoPhase::Q1,
        }
    }

    /// Create a new wallet manager with specified default phase
    pub fn with_phase(phase: CryptoPhase) -> Self {
        Self {
            default_phase: phase,
        }
    }

    /// Create a new wallet (returns wallet ID)
    pub async fn create_wallet(&self, _name: &str, _password: &str) -> Result<String> {
        let wallet = QWallet::new(self.default_phase);
        Ok(wallet.id().to_string())
    }

    /// Get wallet balance (placeholder - needs blockchain integration)
    pub async fn get_balance(&self, _wallet_id: &str) -> Result<u64> {
        // TODO: Query blockchain for actual balance
        Ok(0)
    }

    /// Sign a transaction with the wallet
    pub async fn sign_transaction(
        &self,
        _wallet_id: &str,
        _transaction: serde_json::Value,
        _password: Option<&str>,
    ) -> Result<serde_json::Value> {
        // TODO: Unlock wallet with password and sign transaction
        Ok(serde_json::json!({"signed": true}))
    }

    /// Get wallet information
    pub async fn get_wallet(&self, wallet_id: &str) -> Result<Option<serde_json::Value>> {
        // TODO: Load wallet from secure storage
        Ok(Some(serde_json::json!({
            "id": wallet_id,
            "phase": "Q1",
            "balance": 0
        })))
    }

    /// List all wallets
    pub async fn list_wallets(&self) -> Result<Vec<serde_json::Value>> {
        // TODO: Load all wallets from secure storage
        Ok(vec![])
    }

    /// Create a transaction (helper method)
    pub async fn create_transaction(
        &self,
        _request: serde_json::Value,
    ) -> Result<serde_json::Value> {
        Ok(serde_json::json!({
            "tx_id": uuid::Uuid::new_v4().to_string(),
            "status": "created"
        }))
    }
}

impl Default for WalletManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory-based wallet store (for testing/development)
pub struct MemoryWalletStore {
    // TODO: Implement actual storage
}

impl MemoryWalletStore {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for MemoryWalletStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qwallet_q0_creation() {
        let wallet = QWallet::new_q0();
        assert_eq!(wallet.phase(), CryptoPhase::Q0);
        assert_eq!(wallet.address().len(), 32);
    }

    #[test]
    fn test_qwallet_q1_creation() {
        let wallet = QWallet::new_q1();
        assert_eq!(wallet.phase(), CryptoPhase::Q1);
    }

    #[test]
    fn test_qwallet_q2_creation() {
        let wallet = QWallet::new_q2();
        assert_eq!(wallet.phase(), CryptoPhase::Q2);
    }

    #[test]
    fn test_qwallet_sign_q0() {
        let wallet = QWallet::new_q0();
        let message = b"Test transaction";

        let signature = wallet.sign(message).expect("Signing failed");
        assert_eq!(signature.phase, CryptoPhase::Q0);
        assert!(signature.ed25519_signature.is_some());
        assert!(signature.dilithium5_signature.is_none());
    }

    #[test]
    fn test_qwallet_sign_q1() {
        let wallet = QWallet::new_q1();
        let message = b"Test transaction";

        let signature = wallet.sign(message).expect("Signing failed");
        assert_eq!(signature.phase, CryptoPhase::Q1);
        assert!(signature.ed25519_signature.is_some());
        assert!(signature.dilithium5_signature.is_some());
    }

    #[test]
    fn test_qwallet_sign_q2() {
        let wallet = QWallet::new_q2();
        let message = b"Test transaction";

        let signature = wallet.sign(message).expect("Signing failed");
        assert_eq!(signature.phase, CryptoPhase::Q2);
        assert!(signature.ed25519_signature.is_none());
        assert!(signature.dilithium5_signature.is_some());
    }

    #[tokio::test]
    async fn test_wallet_manager() {
        let manager = WalletManager::new();
        let wallet_id = manager.create_wallet("test", "password").await.unwrap();
        assert!(!wallet_id.is_empty());

        let balance = manager.get_balance(&wallet_id).await.unwrap();
        assert_eq!(balance, 0);
    }
}
