//! Q-NarwhalKnight Privacy-as-a-Service SDK
//!
//! Production-ready Rust SDK for integrating Q-NarwhalKnight's
//! Privacy-as-a-Service into blockchain applications.
//!
//! # Features
//!
//! - ✅ Bitcoin transaction mixing with UTXO management
//! - ✅ Ethereum transaction privacy with MEV protection
//! - ✅ Automatic retry logic with exponential backoff
//! - ✅ Client-side signing (keys never leave your machine)
//! - ✅ Type-safe API with comprehensive error handling
//!
//! # Example - Bitcoin
//!
//! ```no_run
//! use q_paas_sdk::{PaaSClient, BitcoinWallet, PrivacyLevel};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Initialize client
//!     let client = PaaSClient::new(
//!         "your_api_key".to_string(),
//!         "https://quillon.xyz".to_string()
//!     );
//!
//!     // Create wallet
//!     let wallet = BitcoinWallet::from_wif("YOUR_WIF_KEY")?;
//!
//!     // Mix transaction
//!     let result = client.mix_bitcoin_transaction(
//!         "signed_tx_hex",
//!         PrivacyLevel::Maximum,
//!         true // use Tor
//!     ).await?;
//!
//!     println!("Transaction ID: {}", result.mix_id);
//!     Ok(())
//! }
//! ```

pub mod client;
pub mod types;
pub mod error;
pub mod bitcoin;
pub mod ethereum;
pub mod retry;

pub use client::PaaSClient;
pub use types::{PrivacyLevel, MixResult, ApiKeyInfo, BillingInfo};
pub use error::{PaaSError, Result};

#[cfg(feature = "bitcoin-support")]
pub use bitcoin::BitcoinWallet;

#[cfg(feature = "ethereum-support")]
pub use ethereum::EthereumWallet;

/// SDK version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
