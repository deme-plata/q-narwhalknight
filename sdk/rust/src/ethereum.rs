//! Ethereum wallet implementation

use crate::error::{PaaSError, Result};
use ethers::prelude::*;
use ethers::signers::{LocalWallet, Signer};
use std::str::FromStr;

/// Ethereum wallet with private key management
pub struct EthereumWallet {
    wallet: LocalWallet,
}

impl EthereumWallet {
    /// Create wallet from private key hex string
    pub fn from_private_key(private_key_hex: &str) -> Result<Self> {
        let wallet = LocalWallet::from_str(private_key_hex)
            .map_err(|e| PaaSError::WalletError(format!("Invalid private key: {}", e)))?;

        Ok(Self { wallet })
    }

    /// Create wallet from mnemonic phrase
    pub fn from_mnemonic(mnemonic: &str, index: u32) -> Result<Self> {
        let wallet = MnemonicBuilder::<coins_bip39::English>::default()
            .phrase(mnemonic)
            .index(index)
            .map_err(|e| PaaSError::WalletError(format!("Invalid mnemonic builder: {:?}", e)))?
            .build()
            .map_err(|e| PaaSError::WalletError(format!("Invalid mnemonic: {}", e)))?;

        Ok(Self { wallet })
    }

    /// Get the wallet address
    pub fn address(&self) -> Address {
        self.wallet.address()
    }

    /// Get the chain ID
    pub fn chain_id(&self) -> u64 {
        self.wallet.chain_id()
    }

    /// Sign a message
    pub async fn sign_message(&self, message: &[u8]) -> Result<Signature> {
        self.wallet.sign_message(message)
            .await
            .map_err(|e| PaaSError::EthereumError(format!("Failed to sign message: {}", e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wallet_creation() {
        // Test wallet creation from private key
        let pk = "4c0883a69102937d6231471b5dbb6204fe512961708279f8d5d2b4f0c6e7e5e6";
        let wallet = EthereumWallet::from_private_key(pk).unwrap();
        assert_ne!(wallet.address(), Address::zero());
    }
}
