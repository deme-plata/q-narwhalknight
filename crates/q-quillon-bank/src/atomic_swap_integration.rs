/// Quillon Bank Atomic Swap Integration
///
/// This module integrates Bitcoin atomic swaps into the Quillon Bank system.
/// Users can swap BTC for QNKUSD (and vice versa) using trustless HTLCs.
///
/// Flow:
/// 1. User initiates swap: wants to exchange 0.5 BTC → 10,000 QNKUSD
/// 2. Quillon Bank creates atomic swap proposal with hash lock
/// 3. User sends BTC to HTLC address on Bitcoin
/// 4. Bank detects BTC lock and locks QNKUSD in HTLC on Q-Network
/// 5. User reveals secret to claim QNKUSD (and mint it if needed)
/// 6. Bank uses revealed secret to claim BTC
/// 7. Swap complete: User has QNKUSD, Bank has BTC collateral

use anyhow::{anyhow, Result};
use bigdecimal::ToPrimitive;
use q_bitcoin_bridge::{AtomicSwapManager, AtomicSwapProposal, SwapState};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info};

use super::{Address, AssetType, QuillonBankSystem};
use crate::oracle_integration::BankingOracleIntegration;

/// Atomic swap direction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SwapDirection {
    /// User sends BTC, receives QNKUSD
    BtcToQnkusd,
    /// User sends QNKUSD, receives BTC
    QnkusdToBtc,
}

/// Atomic swap request from user
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwapRequest {
    pub direction: SwapDirection,
    pub btc_amount: u64,            // Satoshis
    pub qnkusd_amount: u128,        // QNKUSD base units
    pub user_address: String,        // Q-NarwhalKnight address
    pub user_btc_pubkey: Vec<u8>,   // User's Bitcoin public key for HTLC
}

/// Atomic swap response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwapResponse {
    pub swap_id: String,
    pub htlc_address: String,       // Bitcoin address to send funds to
    pub hash_lock: String,          // Hash lock (hex encoded)
    pub timelock_btc: u32,          // Bitcoin block height for timeout
    pub timelock_qnk: String,       // Q-Network timestamp for timeout
    pub status: String,
}

/// Quillon Bank with atomic swap capabilities
pub struct QuillonBankSwapSystem {
    pub bank: Arc<QuillonBankSystem>,
    pub swap_manager: Arc<RwLock<AtomicSwapManager>>,
    pub oracle: Arc<BankingOracleIntegration>,
    pub bank_btc_pubkey: Vec<u8>, // Bank's Bitcoin public key
}

impl QuillonBankSwapSystem {
    /// Create new Quillon Bank swap system
    pub async fn new(
        bank: QuillonBankSystem,
        bitcoin_rpc_url: String,
        bitcoin_rpc_user: String,
        bitcoin_rpc_password: String,
    ) -> Result<Self> {
        let swap_manager = AtomicSwapManager::new(
            bitcoin_rpc_url,
            bitcoin_rpc_user,
            bitcoin_rpc_password,
            bitcoin::Network::Bitcoin, // Use mainnet in production
        )?;

        // Get bank's BTC public key (in production, load from secure storage)
        let bank_btc_pubkey = vec![3u8; 33]; // Placeholder compressed pubkey

        let oracle: Arc<BankingOracleIntegration> = Arc::new(BankingOracleIntegration::new().await?);
        oracle.initialize().await?;

        Ok(Self {
            bank: Arc::new(bank),
            swap_manager: Arc::new(RwLock::new(swap_manager)),
            oracle,
            bank_btc_pubkey,
        })
    }

    /// Create atomic swap proposal
    pub async fn create_swap(&self, request: SwapRequest) -> Result<SwapResponse> {
        info!(
            "💱 Creating atomic swap: {} sats BTC → {} QNKUSD",
            request.btc_amount,
            request.qnkusd_amount as f64 / 1_000_000_000_000.0
        );

        // Verify swap rate using real-time oracle prices
        self.verify_swap_rate(&request).await?;

        // Generate secret and hash lock
        let (secret, hash_lock) = AtomicSwapManager::generate_secret();

        info!("🔒 Generated hash lock: {}", hex::encode(&hash_lock));

        // Create swap proposal
        let swap_manager = self.swap_manager.write().await;
        let proposal = swap_manager
            .create_swap_proposal(
                request.user_address.clone(),
                request.btc_amount,
                request.qnkusd_amount,
                hash_lock,
                request.user_btc_pubkey.clone(),
                self.bank_btc_pubkey.clone(),
            )
            .await?;

        // Get HTLC address for user to send BTC to
        let htlc_address = swap_manager.get_htlc_address(&proposal)?;

        info!(
            "📍 HTLC address generated: {}",
            htlc_address.to_string()
        );

        // Store secret securely (in production, use HSM or encrypted storage)
        // For now, just log it
        info!("🔑 Secret (KEEP SECURE): {}", hex::encode(&secret));

        let response = SwapResponse {
            swap_id: proposal.swap_id.clone(),
            htlc_address: htlc_address.to_string(),
            hash_lock: hex::encode(&hash_lock),
            timelock_btc: proposal.timelock_btc,
            timelock_qnk: proposal.timelock_qnk.to_rfc3339(),
            status: "proposed".to_string(),
        };

        Ok(response)
    }

    /// Verify swap rate is fair using oracle prices
    async fn verify_swap_rate(&self, request: &SwapRequest) -> Result<()> {
        // Get real-time BTC price
        let btc_price = self.oracle.get_price(&AssetType::BTC).await
            .map_err(|e| anyhow!("Failed to get BTC price: {}", e))?;

        // Calculate expected QNKUSD amount (assuming 1:1 USD peg)
        let btc_value_usd = (request.btc_amount as f64 / 100_000_000.0) * btc_price.to_f64().unwrap_or(0.0);
        let expected_qnkusd = (btc_value_usd * 1_000_000_000_000.0) as u128;

        // Allow 2% slippage
        let min_acceptable = (expected_qnkusd as f64 * 0.98) as u128;
        let max_acceptable = (expected_qnkusd as f64 * 1.02) as u128;

        if request.qnkusd_amount < min_acceptable || request.qnkusd_amount > max_acceptable {
            return Err(anyhow!(
                "Swap rate out of acceptable range: requested {} QNKUSD, expected {} QNKUSD (±2%)",
                request.qnkusd_amount as f64 / 1_000_000_000_000.0,
                expected_qnkusd as f64 / 1_000_000_000_000.0
            ));
        }

        info!(
            "✅ Swap rate verified: {} BTC @ ${} = ${} → {} QNKUSD",
            request.btc_amount as f64 / 100_000_000.0,
            btc_price,
            btc_value_usd,
            request.qnkusd_amount as f64 / 1_000_000_000_000.0
        );

        Ok(())
    }

    /// Monitor Bitcoin for HTLC funding
    pub async fn monitor_btc_lock(&self, swap_id: &str) -> Result<()> {
        info!("👀 Monitoring Bitcoin for HTLC lock: {}", swap_id);

        let swap_manager = self.swap_manager.write().await;
        let (btc_txid, btc_vout) = swap_manager.monitor_btc_lock(swap_id).await?;

        info!("✅ BTC locked detected: {} (vout: {})", btc_txid, btc_vout);

        // Lock QNKUSD in response
        let qnk_tx_hash = swap_manager.lock_qnkusd(swap_id, btc_txid, btc_vout).await?;

        info!("✅ QNKUSD locked in HTLC: {}", qnk_tx_hash);

        Ok(())
    }

    /// Process user claim of QNKUSD (reveals secret, triggers minting)
    pub async fn process_user_claim(&self, swap_id: &str, secret: Vec<u8>) -> Result<()> {
        info!("💎 Processing user claim for swap: {}", swap_id);

        // Verify secret and update swap state
        let swap_manager = self.swap_manager.write().await;
        swap_manager.process_qnkusd_claim(swap_id, secret.clone()).await?;

        // Get swap details
        let proposal = swap_manager.get_swap_status(swap_id).await?;

        // Mint QNKUSD for user (collateralized by the BTC in HTLC)
        let user_address = Address::new(); // Parse from proposal.user_address
        self.bank
            .mint_qnkusd(
                &user_address,
                proposal.btc_amount as u128,
                AssetType::BTC,
                proposal.qnkusd_amount,
            )
            .await
            .map_err(|e| anyhow!("Failed to mint QNKUSD: {}", e))?;

        info!(
            "✅ Minted {} QNKUSD for user",
            proposal.qnkusd_amount as f64 / 1_000_000_000_000.0
        );

        // Now claim BTC using revealed secret
        let btc_claim_txid = swap_manager.claim_btc(swap_id).await?;

        info!("✅ Bank claimed BTC: {}", btc_claim_txid);
        info!("🎉 Atomic swap completed successfully!");

        Ok(())
    }

    /// Get swap status
    pub async fn get_swap_status(&self, swap_id: &str) -> Result<AtomicSwapProposal> {
        let swap_manager = self.swap_manager.read().await;
        swap_manager.get_swap_status(swap_id).await
    }

    /// List all active swaps
    pub async fn list_active_swaps(&self) -> Result<Vec<AtomicSwapProposal>> {
        let swap_manager = self.swap_manager.read().await;
        Ok(swap_manager.list_active_swaps().await)
    }

    /// Handle refund after timeout
    pub async fn handle_refund(&self, swap_id: &str) -> Result<()> {
        info!("⏰ Processing refund for swap: {}", swap_id);

        let swap_manager = self.swap_manager.write().await;
        let proposal = swap_manager.get_swap_status(swap_id).await?;

        match proposal.state {
            SwapState::BtcLocked { .. } => {
                // User never claimed QNKUSD, refund their BTC
                let refund_txid = swap_manager.refund_btc(swap_id).await?;
                info!("✅ BTC refunded to user: {}", refund_txid);
            }
            SwapState::QnkusdLocked { .. } => {
                // Bank never claimed BTC, refund QNKUSD to bank
                let refund_tx_hash = swap_manager.refund_qnkusd(swap_id).await?;
                info!("✅ QNKUSD refunded to bank: {}", refund_tx_hash);
            }
            _ => {
                return Err(anyhow!("Swap not in refundable state"));
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_swap_rate_verification() {
        // Test that swap rate verification works correctly
        // This would need a mock oracle in production tests
    }

    #[test]
    fn test_swap_request_serialization() {
        let request = SwapRequest {
            direction: SwapDirection::BtcToQnkusd,
            btc_amount: 50_000_000, // 0.5 BTC
            qnkusd_amount: 10_000_000_000_000_000, // 10,000 QNKUSD
            user_address: "qnk1abc123".to_string(),
            user_btc_pubkey: vec![2u8; 33],
        };

        let json = serde_json::to_string(&request).unwrap();
        let deserialized: SwapRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(request.btc_amount, deserialized.btc_amount);
        assert_eq!(request.qnkusd_amount, deserialized.qnkusd_amount);
    }
}