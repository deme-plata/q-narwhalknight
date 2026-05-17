//! Development Fee System
//!
//! Transparent 1% development fee to sustain Q-NarwhalKnight project development.
//!
//! **Economic Model:**
//! - 1% of all mining rewards go to the founder wallet for ongoing development
//! - 99% goes to the miner who found the solution
//! - Fully transparent and documented in project README
//!
//! **Security Model:**
//! - Miners must be authenticated via AEGIS-QL post-quantum signatures
//! - Only miners with valid AEGIS-QL credentials can submit solutions
//! - This prevents unauthorized forks and ensures network integrity
//!
//! **Rationale:**
//! - Sustainable funding for continuous development and improvements
//! - Post-quantum security research and implementation
//! - Network infrastructure and bootstrap node maintenance
//! - Academic paper publications and peer review
//! - Community support and documentation

use serde::{Deserialize, Serialize};
use q_aegis_ql::{AegisError, AegisQL, PublicKey, SecretKey, Signature};

/// Development fee percentage (1% = 0.01)
pub const DEV_FEE_PERCENT: f64 = 0.02; // v8.6.0: raised from 1% to 2%

/// Founder wallet address (master development wallet)
/// This wallet receives 1% of all mining rewards to fund:
/// - Core protocol development
/// - Post-quantum cryptography research
/// - Network infrastructure maintenance
/// - Academic publications and peer review
/// - Community support and documentation
pub const FOUNDER_WALLET: &str = "qnk8f7a6b5c4d3e2f1a0b9c8d7e6f5a4b3c2d1e0f1a2b3c4d5e6f7a8b9c0d1e2f3a";

/// Founder wallet as bytes (derived from string above)
pub fn founder_wallet_bytes() -> [u8; 32] {
    // Decode the qnk-prefixed address
    let hex_part = FOUNDER_WALLET.strip_prefix("qnk").unwrap();
    let mut bytes = [0u8; 32];
    hex::decode_to_slice(hex_part, &mut bytes).expect("Invalid founder wallet address");
    bytes
}

/// Development fee configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DevFeeConfig {
    /// Fee percentage (0.01 = 1%)
    pub fee_percent: f64,
    /// Founder wallet address
    pub founder_wallet: [u8; 32],
    /// Whether to enforce AEGIS-QL authentication
    pub require_aegis_auth: bool,
}

impl Default for DevFeeConfig {
    fn default() -> Self {
        Self {
            fee_percent: DEV_FEE_PERCENT,
            founder_wallet: founder_wallet_bytes(),
            require_aegis_auth: true,
        }
    }
}

/// Calculate development fee split
///
/// # Returns
/// (miner_amount, dev_fee_amount)
pub fn calculate_dev_fee_split(total_reward: u64) -> (u64, u64) {
    let dev_fee = (total_reward as f64 * DEV_FEE_PERCENT) as u64;
    let miner_reward = total_reward - dev_fee;
    (miner_reward, dev_fee)
}

/// Miner authentication credentials
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinerCredentials {
    /// Miner wallet address
    pub wallet_address: String,
    /// AEGIS-QL public key for signature verification
    pub aegis_public_key: PublicKey,
}

/// Miner authentication system
pub struct MinerAuth {
    aegis: AegisQL,
}

impl MinerAuth {
    pub fn new() -> Self {
        Self {
            aegis: AegisQL::new(),
        }
    }

    /// Verify a mining solution submission with AEGIS-QL authentication
    ///
    /// This prevents unauthorized miners/forks from submitting solutions
    pub fn verify_miner_auth(
        &self,
        credentials: &MinerCredentials,
        solution_data: &[u8],
        signature: &Signature,
    ) -> Result<bool, AegisError> {
        self.aegis.verify(solution_data, signature, &credentials.aegis_public_key)
    }

    /// Generate miner credentials (for testing/registration)
    pub fn generate_miner_credentials(&mut self, wallet_address: String) -> Result<(MinerCredentials, SecretKey), AegisError> {
        let (public_key, secret_key) = self.aegis.generate_keypair()?;

        let credentials = MinerCredentials {
            wallet_address,
            aegis_public_key: public_key,
        };

        Ok((credentials, secret_key))
    }
}

impl Default for MinerAuth {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dev_fee_calculation() {
        // Test 1% fee on 100 QNK reward
        let total = 100_000_000_000; // 100 QNK in smallest units
        let (miner, dev) = calculate_dev_fee_split(total);

        assert_eq!(dev, 1_000_000_000); // 1 QNK
        assert_eq!(miner, 99_000_000_000); // 99 QNK
        assert_eq!(miner + dev, total);
    }

    #[test]
    fn test_founder_wallet() {
        let wallet = founder_wallet_bytes();
        assert_eq!(wallet.len(), 32);
    }

    #[test]
    fn test_miner_auth() {
        let mut auth = MinerAuth::new();
        let wallet = "qnk1234567890abcdef".to_string();

        let result = auth.generate_miner_credentials(wallet.clone());
        assert!(result.is_ok());

        let (credentials, secret_key) = result.unwrap();
        assert_eq!(credentials.wallet_address, wallet);
    }

    #[test]
    fn test_miner_signature_verification() {
        let mut aegis = AegisQL::new();
        let (public_key, secret_key) = aegis.generate_keypair().unwrap();

        let solution_data = b"mining_solution_nonce_12345";
        let signature = aegis.sign(solution_data, &secret_key).unwrap();

        let auth = MinerAuth::new();
        let credentials = MinerCredentials {
            wallet_address: "test_wallet".to_string(),
            aegis_public_key: public_key,
        };

        let is_valid = auth.verify_miner_auth(&credentials, solution_data, &signature).unwrap();
        assert!(is_valid);
    }
}
