use q_types::*;
use anyhow::Result;
use async_trait::async_trait;
use ed25519_dalek::{Keypair, PublicKey, SecretKey, Signature, Signer};
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::collections::HashMap;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Wallet management for Q-NarwhalKnight
/// Phase 0: Ed25519 keys with secure storage
/// Phase 1+: Will be extended for post-quantum cryptography

#[async_trait]
pub trait WalletStore: Send + Sync {
    async fn save_wallet(&self, wallet: &StoredWallet) -> Result<()>;
    async fn load_wallet(&self, id: &Uuid) -> Result<Option<StoredWallet>>;
    async fn list_wallets(&self) -> Result<Vec<WalletInfo>>;
    async fn delete_wallet(&self, id: &Uuid) -> Result<()>;
}

/// Stored wallet with encrypted private key
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredWallet {
    pub info: WalletInfo,
    pub encrypted_private_key: Vec<u8>,
    pub salt: [u8; 32],
    pub nonce: [u8; 24],
}

/// In-memory wallet store for development/testing
pub struct MemoryWalletStore {
    wallets: RwLock<HashMap<Uuid, StoredWallet>>,
}

impl MemoryWalletStore {
    pub fn new() -> Self {
        Self {
            wallets: RwLock::new(HashMap::new()),
        }
    }
}

#[async_trait]
impl WalletStore for MemoryWalletStore {
    async fn save_wallet(&self, wallet: &StoredWallet) -> Result<()> {
        let mut wallets = self.wallets.write().await;
        wallets.insert(wallet.info.id, wallet.clone());
        Ok(())
    }

    async fn load_wallet(&self, id: &Uuid) -> Result<Option<StoredWallet>> {
        let wallets = self.wallets.read().await;
        Ok(wallets.get(id).cloned())
    }

    async fn list_wallets(&self) -> Result<Vec<WalletInfo>> {
        let wallets = self.wallets.read().await;
        Ok(wallets.values().map(|w| w.info.clone()).collect())
    }

    async fn delete_wallet(&self, id: &Uuid) -> Result<()> {
        let mut wallets = self.wallets.write().await;
        wallets.remove(id);
        Ok(())
    }
}

/// Wallet manager handles key generation, storage, and signing
pub struct WalletManager<S: WalletStore> {
    store: S,
}

impl<S: WalletStore> WalletManager<S> {
    pub fn new(store: S) -> Self {
        Self { store }
    }

    /// Create a new wallet with Ed25519 keypair
    pub async fn create_wallet(&self, password: Option<&str>) -> Result<WalletInfo> {
        let mut csprng = OsRng {};
        let keypair = Keypair::generate(&mut csprng);
        let wallet_id = Uuid::new_v4();
        
        // Derive address from public key hash
        let address = Self::public_key_to_address(&keypair.public);
        
        // Encrypt private key if password provided
        let (encrypted_private_key, salt, nonce) = if let Some(pwd) = password {
            self.encrypt_private_key(&keypair.secret, pwd)?
        } else {
            (keypair.secret.to_bytes().to_vec(), [0u8; 32], [0u8; 24])
        };

        let wallet_info = WalletInfo {
            id: wallet_id,
            address,
            public_key: keypair.public.to_bytes().to_vec(),
            balance: 0, // Will be updated by querying the ledger
            nonce: 0,
            created_at: chrono::Utc::now(),
        };

        let stored_wallet = StoredWallet {
            info: wallet_info.clone(),
            encrypted_private_key,
            salt,
            nonce,
        };

        self.store.save_wallet(&stored_wallet).await?;
        Ok(wallet_info)
    }

    /// Load wallet information
    pub async fn get_wallet(&self, id: &Uuid) -> Result<Option<WalletInfo>> {
        if let Some(wallet) = self.store.load_wallet(id).await? {
            Ok(Some(wallet.info))
        } else {
            Ok(None)
        }
    }

    /// List all wallets
    pub async fn list_wallets(&self) -> Result<Vec<WalletInfo>> {
        self.store.list_wallets().await
    }

    /// Sign a transaction
    pub async fn sign_transaction(
        &self,
        wallet_id: &Uuid,
        mut transaction: Transaction,
        password: Option<&str>,
    ) -> Result<Transaction> {
        let stored_wallet = self.store
            .load_wallet(wallet_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Wallet not found"))?;

        // Decrypt private key
        let private_key = self.decrypt_private_key(
            &stored_wallet.encrypted_private_key,
            &stored_wallet.salt,
            &stored_wallet.nonce,
            password,
        )?;

        let keypair = Keypair {
            secret: private_key,
            public: PublicKey::from_bytes(&stored_wallet.info.public_key)?,
        };

        // Create transaction hash for signing
        transaction.signature = vec![]; // Clear signature field
        let tx_hash = transaction.hash();
        
        // Sign the transaction hash
        let signature = keypair.sign(&tx_hash);
        transaction.signature = signature.to_bytes().to_vec();

        Ok(transaction)
    }

    /// Create a transaction ready for signing
    pub async fn create_transaction(
        &self,
        wallet_id: &Uuid,
        to: Address,
        amount: Amount,
        fee: Amount,
    ) -> Result<Transaction> {
        let wallet = self.get_wallet(wallet_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Wallet not found"))?;

        let transaction = Transaction {
            id: [0u8; 32], // Will be set after signing
            from: wallet.address,
            to,
            amount,
            fee,
            nonce: wallet.nonce + 1, // TODO: Get actual nonce from ledger
            signature: vec![],
            timestamp: chrono::Utc::now(),
        };

        Ok(transaction)
    }

    /// Convert public key to address (Phase 0: simple hash)
    fn public_key_to_address(public_key: &PublicKey) -> Address {
        let mut hasher = Sha3_256::new();
        hasher.update(public_key.as_bytes());
        hasher.finalize().into()
    }

    /// Encrypt private key with password (simple implementation for Phase 0)
    fn encrypt_private_key(&self, private_key: &SecretKey, password: &str) -> Result<(Vec<u8>, [u8; 32], [u8; 24])> {
        use pbkdf2::pbkdf2_hmac;
        use sha3::Sha3_256;
        use rand::RngCore;
        
        let mut salt = [0u8; 32];
        let mut nonce = [0u8; 24];
        OsRng.fill_bytes(&mut salt);
        OsRng.fill_bytes(&mut nonce);
        
        let mut key = [0u8; 32];
        pbkdf2_hmac::<Sha3_256>(password.as_bytes(), &salt, 100_000, &mut key);
        
        // Simple XOR encryption for Phase 0 (will use proper AEAD in production)
        let private_bytes = private_key.to_bytes();
        let mut encrypted = private_bytes.to_vec();
        for (i, byte) in encrypted.iter_mut().enumerate() {
            *byte ^= key[i % 32];
        }
        
        Ok((encrypted, salt, nonce))
    }

    /// Decrypt private key with password
    fn decrypt_private_key(
        &self,
        encrypted_key: &[u8],
        salt: &[u8; 32],
        _nonce: &[u8; 24],
        password: Option<&str>,
    ) -> Result<SecretKey> {
        if encrypted_key.len() != 32 {
            return Err(anyhow::anyhow!("Invalid encrypted key length"));
        }

        let decrypted = if let Some(pwd) = password {
            use pbkdf2::pbkdf2_hmac;
            use sha3::Sha3_256;
            
            let mut key = [0u8; 32];
            pbkdf2_hmac::<Sha3_256>(pwd.as_bytes(), salt, 100_000, &mut key);
            
            let mut decrypted = encrypted_key.to_vec();
            for (i, byte) in decrypted.iter_mut().enumerate() {
                *byte ^= key[i % 32];
            }
            decrypted
        } else {
            // No password encryption
            encrypted_key.to_vec()
        };

        let private_key = SecretKey::from_bytes(&decrypted)?;
        Ok(private_key)
    }

    /// Verify a transaction signature
    pub fn verify_transaction(&self, transaction: &Transaction) -> Result<bool> {
        if transaction.signature.is_empty() {
            return Ok(false);
        }

        // Reconstruct the signed message
        let mut tx_for_verification = transaction.clone();
        tx_for_verification.signature = vec![];
        let tx_hash = tx_for_verification.hash();

        // Extract public key from address (this is a simplification)
        // In production, we'd need to look up the public key from the ledger
        let signature = Signature::from_bytes(&transaction.signature)?;
        
        // For Phase 0, we'll need to get the public key from somewhere
        // This is a placeholder - in practice, the public key would be stored
        // with the account or derived from the address
        todo!("Public key lookup from address needed")
    }

    /// Update wallet balance (called by ledger sync)
    pub async fn update_balance(&self, wallet_id: &Uuid, new_balance: Amount, new_nonce: u64) -> Result<()> {
        if let Some(mut stored_wallet) = self.store.load_wallet(wallet_id).await? {
            stored_wallet.info.balance = new_balance;
            stored_wallet.info.nonce = new_nonce;
            self.store.save_wallet(&stored_wallet).await?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_wallet_creation() {
        let store = MemoryWalletStore::new();
        let manager = WalletManager::new(store);

        let wallet = manager.create_wallet(Some("password123")).await.unwrap();
        assert_eq!(wallet.balance, 0);
        assert_eq!(wallet.nonce, 0);
        assert_eq!(wallet.public_key.len(), 32);
    }

    #[tokio::test]
    async fn test_transaction_signing() {
        let store = MemoryWalletStore::new();
        let manager = WalletManager::new(store);

        let wallet = manager.create_wallet(Some("password123")).await.unwrap();
        let to_address = [1u8; 32]; // Dummy address
        
        let mut transaction = manager
            .create_transaction(&wallet.id, to_address, 1000, 10)
            .await
            .unwrap();

        transaction = manager
            .sign_transaction(&wallet.id, transaction, Some("password123"))
            .await
            .unwrap();

        assert!(!transaction.signature.is_empty());
        assert_eq!(transaction.signature.len(), 64); // Ed25519 signature size
    }
}