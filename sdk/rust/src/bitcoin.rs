//! Bitcoin wallet implementation with UTXO management

use crate::error::{PaaSError, Result};
use bitcoin::{PrivateKey, PublicKey, Address, Network};
use bitcoin::secp256k1::{Secp256k1, SecretKey};
use std::str::FromStr;

/// Bitcoin wallet with private key management
pub struct BitcoinWallet {
    private_key: PrivateKey,
    public_key: PublicKey,
    address: Address,
    network: Network,
}

impl BitcoinWallet {
    /// Create wallet from WIF (Wallet Import Format) private key
    pub fn from_wif(wif: &str) -> Result<Self> {
        let private_key = PrivateKey::from_wif(wif)
            .map_err(|e| PaaSError::WalletError(format!("Invalid WIF key: {}", e)))?;

        let secp = Secp256k1::new();
        let public_key = private_key.public_key(&secp);
        let network = private_key.network;

        let address = Address::p2wpkh(&public_key, network)
            .map_err(|e| PaaSError::WalletError(format!("Failed to generate address: {}", e)))?;

        Ok(Self {
            private_key,
            public_key,
            address,
            network,
        })
    }

    /// Create wallet from raw private key bytes
    pub fn from_bytes(key_bytes: &[u8], network: Network) -> Result<Self> {
        let secret_key = SecretKey::from_slice(key_bytes)
            .map_err(|e| PaaSError::WalletError(format!("Invalid private key: {}", e)))?;

        let private_key = PrivateKey::new(secret_key, network);
        let secp = Secp256k1::new();
        let public_key = private_key.public_key(&secp);

        let address = Address::p2wpkh(&public_key, network)
            .map_err(|e| PaaSError::WalletError(format!("Failed to generate address: {}", e)))?;

        Ok(Self {
            private_key,
            public_key,
            address,
            network,
        })
    }

    /// Get the wallet address
    pub fn address(&self) -> &Address {
        &self.address
    }

    /// Get the public key
    pub fn public_key(&self) -> &PublicKey {
        &self.public_key
    }

    /// Get the network
    pub fn network(&self) -> Network {
        self.network
    }

    /// Export private key in WIF format
    pub fn to_wif(&self) -> String {
        self.private_key.to_wif()
    }

    /// Sign a message (for authentication, not transactions)
    pub fn sign_message(&self, message: &[u8]) -> Result<Vec<u8>> {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(message);
        let hash = hasher.finalize();

        let secp = Secp256k1::new();
        let message_hash = bitcoin::secp256k1::Message::from_digest_slice(&hash)
            .map_err(|e| PaaSError::WalletError(format!("Invalid message hash: {}", e)))?;

        let signature = secp.sign_ecdsa(&message_hash, &self.private_key.inner);
        Ok(signature.serialize_compact().to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wallet_creation() {
        // Test wallet creation from WIF
        let wif = "L1aW4aubDFB7yfras2S1mN3bqg9nwySY8nkoLmJebSLD5BWv3ENZ"; // Example mainnet WIF
        let wallet = BitcoinWallet::from_wif(wif).unwrap();
        assert_eq!(wallet.network(), Network::Bitcoin);
    }
}
