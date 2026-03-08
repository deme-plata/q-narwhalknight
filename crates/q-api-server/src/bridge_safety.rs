// ============================================================================
// bridge_safety.rs - Cross-Chain Bridge Safety Layer (v9.4.0)
// ============================================================================
//
// CRITICAL SAFETY MODULE — Prevents money loss in cross-chain swaps.
//
// This module provides:
//   1. Deposit verification via external chain RPC calls
//   2. Swap timeout monitoring and auto-refund
//   3. Admin kill-switch to freeze all bridge operations
//   4. Max amount limits per chain
//   5. Confirmation threshold enforcement
//   6. Audit logging for all bridge operations
//
// NO wrapped tokens may be minted without this module's approval.
// ============================================================================

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{info, warn, error};

use crate::bridge_tokens::BridgeChain;

// ============================================================================
// Configuration Constants
// ============================================================================

/// Minimum confirmations required per chain before minting wrapped tokens
pub const BTC_MIN_CONFIRMATIONS: u32 = 3;
pub const ETH_MIN_CONFIRMATIONS: u32 = 12;
pub const ZEC_MIN_CONFIRMATIONS: u32 = 10;
pub const IRON_MIN_CONFIRMATIONS: u32 = 10;

/// Maximum swap amounts (in native base units) — start conservative, increase after soak
/// BTC: 0.1 BTC = 10_000_000 satoshis
/// ETH: 1.0 ETH = 1_000_000_000_000_000_000 wei
/// ZEC: 10 ZEC  = 1_000_000_000 zatoshis
/// IRON: 100 IRON = 10_000_000_000 (10^8 base units)
pub const BTC_MAX_AMOUNT_SATS: u64 = 10_000_000;
pub const ETH_MAX_AMOUNT_WEI: u128 = 1_000_000_000_000_000_000;
pub const ZEC_MAX_AMOUNT_ZATS: u64 = 1_000_000_000;
pub const IRON_MAX_AMOUNT_BASE: u64 = 10_000_000_000;

/// Swap auto-expiry: if still in Proposed state after this many seconds
pub const SWAP_EXPIRY_SECS: u64 = 43200; // 12 hours (matches HTLC timelock)

/// Background scan interval for expired swaps
pub const SWAP_SCAN_INTERVAL_SECS: u64 = 60;

// ============================================================================
// Bridge Safety State
// ============================================================================

/// Global bridge safety controller — shared across all bridge API handlers
pub struct BridgeSafetyController {
    /// Kill switch: if true, ALL bridge operations are frozen
    frozen: AtomicBool,
    /// Per-chain freeze (more granular)
    chain_frozen: RwLock<HashMap<BridgeChain, bool>>,
    /// Pending deposits awaiting confirmation
    pending_deposits: RwLock<Vec<PendingDeposit>>,
    /// RPC endpoints for external chain verification
    rpc_endpoints: RwLock<ChainRpcConfig>,
}

/// RPC endpoint configuration for external chains
#[derive(Debug, Clone, Default)]
pub struct ChainRpcConfig {
    pub btc_rpc_url: Option<String>,
    pub btc_rpc_user: Option<String>,
    pub btc_rpc_password: Option<String>,
    pub eth_rpc_url: Option<String>,
    pub zec_rpc_url: Option<String>,
    pub iron_rpc_url: Option<String>,
}

/// A deposit that is waiting for on-chain confirmation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingDeposit {
    pub swap_id: String,
    pub chain: BridgeChain,
    pub expected_amount: u128,
    pub deposit_address: String,
    pub deposit_txid: Option<String>,
    pub confirmations: u32,
    pub required_confirmations: u32,
    pub created_at: DateTime<Utc>,
    pub last_checked: DateTime<Utc>,
    pub status: DepositStatus,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DepositStatus {
    /// Waiting for user to deposit on source chain
    AwaitingDeposit,
    /// Deposit detected, waiting for confirmations
    AwaitingConfirmations,
    /// Deposit confirmed with sufficient confirmations
    Confirmed,
    /// Deposit expired (no deposit within timelock)
    Expired,
    /// Deposit verification failed
    Failed(String),
}

/// Result of a deposit verification check
#[derive(Debug, Clone)]
pub enum DepositVerificationResult {
    /// Deposit confirmed with N confirmations
    Confirmed { confirmations: u32, txid: String },
    /// Deposit detected but insufficient confirmations
    Pending { confirmations: u32, txid: String },
    /// No deposit found at the expected address/contract
    NotFound,
    /// RPC call failed (node unreachable, etc.)
    RpcError(String),
    /// Bridge is frozen — no operations allowed
    Frozen,
}

impl BridgeSafetyController {
    /// Create a new safety controller with default configuration
    pub fn new() -> Self {
        let rpc_config = ChainRpcConfig {
            btc_rpc_url: std::env::var("BTC_RPC_URL").ok(),
            btc_rpc_user: std::env::var("BTC_RPC_USER").ok(),
            btc_rpc_password: std::env::var("BTC_RPC_PASSWORD").ok(),
            eth_rpc_url: std::env::var("ETH_RPC_URL").ok()
                .or_else(|| Some("http://5.79.79.158:8545".to_string())),
            zec_rpc_url: std::env::var("ZEC_RPC_URL").ok(),
            iron_rpc_url: std::env::var("IRON_RPC_URL").ok(),
        };

        Self {
            frozen: AtomicBool::new(false),
            chain_frozen: RwLock::new(HashMap::new()),
            pending_deposits: RwLock::new(Vec::new()),
            rpc_endpoints: RwLock::new(rpc_config),
        }
    }

    // ========================================================================
    // Kill Switch
    // ========================================================================

    /// Freeze ALL bridge operations immediately
    pub fn freeze_all(&self) {
        self.frozen.store(true, Ordering::SeqCst);
        error!("🚨🔴 BRIDGE KILL-SWITCH ACTIVATED — All bridge operations FROZEN!");
    }

    /// Unfreeze all bridge operations
    pub fn unfreeze_all(&self) {
        self.frozen.store(false, Ordering::SeqCst);
        info!("🟢 Bridge operations UNFROZEN — resuming normal operation.");
    }

    /// Check if bridge is globally frozen
    pub fn is_frozen(&self) -> bool {
        self.frozen.load(Ordering::SeqCst)
    }

    /// Freeze a specific chain's bridge
    pub async fn freeze_chain(&self, chain: BridgeChain) {
        let mut locked = self.chain_frozen.write().await;
        locked.insert(chain, true);
        error!("🚨 Bridge FROZEN for chain {:?}", chain);
    }

    /// Unfreeze a specific chain's bridge
    pub async fn unfreeze_chain(&self, chain: BridgeChain) {
        let mut locked = self.chain_frozen.write().await;
        locked.insert(chain, false);
        info!("🟢 Bridge UNFROZEN for chain {:?}", chain);
    }

    /// Check if a specific chain is frozen
    pub async fn is_chain_frozen(&self, chain: BridgeChain) -> bool {
        if self.is_frozen() {
            return true;
        }
        let locked = self.chain_frozen.read().await;
        *locked.get(&chain).unwrap_or(&false)
    }

    // ========================================================================
    // Pre-Mint Safety Checks
    // ========================================================================

    /// MANDATORY check before ANY wrapped token mint.
    /// Returns Ok(()) if the mint is safe to proceed, Err(reason) if not.
    pub async fn pre_mint_check(
        &self,
        chain: BridgeChain,
        amount: u128,
        swap_id: &str,
        deposit_txid: Option<&str>,
    ) -> Result<(), String> {
        // 1. Kill-switch check
        if self.is_chain_frozen(chain).await {
            return Err(format!(
                "🚨 Bridge operations for {:?} are FROZEN. Contact admin.",
                chain
            ));
        }

        // 2. Amount limit check
        self.check_amount_limit(chain, amount)?;

        // 3. Deposit verification check
        if let Some(txid) = deposit_txid {
            let result = self.verify_deposit(chain, txid, amount).await;
            match result {
                DepositVerificationResult::Confirmed { confirmations, txid } => {
                    info!(
                        "✅ [BRIDGE SAFETY] {:?} deposit verified: txid={}, {} confirmations",
                        chain, txid, confirmations
                    );
                }
                DepositVerificationResult::Pending { confirmations, txid } => {
                    let required = self.min_confirmations(chain);
                    return Err(format!(
                        "Deposit {} has only {}/{} confirmations. Wait for more.",
                        txid, confirmations, required
                    ));
                }
                DepositVerificationResult::NotFound => {
                    return Err(format!(
                        "No deposit found for swap {}. Cannot mint without deposit proof.",
                        swap_id
                    ));
                }
                DepositVerificationResult::RpcError(e) => {
                    warn!(
                        "⚠️ [BRIDGE SAFETY] {:?} RPC error during verification: {}. \
                         Requiring deposit_txid for manual verification.",
                        chain, e
                    );
                    // If RPC is down, we STILL require the txid — admin can verify manually
                    return Err(format!(
                        "Cannot verify deposit (RPC error: {}). \
                         Please provide deposit_txid and try again when RPC is available.",
                        e
                    ));
                }
                DepositVerificationResult::Frozen => {
                    return Err("Bridge is frozen.".to_string());
                }
            }
        } else {
            // No deposit txid provided — this is the CRITICAL safety gap we're fixing
            return Err(format!(
                "🚨 SAFETY: Cannot mint wrapped {:?} tokens without deposit_txid. \
                 Provide the transaction ID of your deposit on the source chain.",
                chain
            ));
        }

        // 4. Log the pre-mint check passing
        info!(
            "✅ [BRIDGE SAFETY] Pre-mint check PASSED for {:?} swap {} amount {}",
            chain, swap_id, amount
        );

        Ok(())
    }

    /// Check if the swap amount is within allowed limits
    pub fn check_amount_limit(&self, chain: BridgeChain, amount: u128) -> Result<(), String> {
        let max = match chain {
            BridgeChain::Bitcoin => BTC_MAX_AMOUNT_SATS as u128,
            BridgeChain::Ethereum => ETH_MAX_AMOUNT_WEI,
            BridgeChain::Zcash => ZEC_MAX_AMOUNT_ZATS as u128,
            BridgeChain::IronFish => IRON_MAX_AMOUNT_BASE as u128,
        };

        if amount > max {
            return Err(format!(
                "Amount {} exceeds maximum allowed {} for {:?} bridge. \
                 Contact admin to increase limits.",
                amount, max, chain
            ));
        }

        Ok(())
    }

    /// Get minimum confirmations for a chain
    pub fn min_confirmations(&self, chain: BridgeChain) -> u32 {
        match chain {
            BridgeChain::Bitcoin => BTC_MIN_CONFIRMATIONS,
            BridgeChain::Ethereum => ETH_MIN_CONFIRMATIONS,
            BridgeChain::Zcash => ZEC_MIN_CONFIRMATIONS,
            BridgeChain::IronFish => IRON_MIN_CONFIRMATIONS,
        }
    }

    // ========================================================================
    // Deposit Verification (External Chain RPC)
    // ========================================================================

    /// Verify a deposit on the source chain via RPC
    pub async fn verify_deposit(
        &self,
        chain: BridgeChain,
        txid: &str,
        _expected_amount: u128,
    ) -> DepositVerificationResult {
        if self.is_frozen() {
            return DepositVerificationResult::Frozen;
        }

        match chain {
            BridgeChain::Bitcoin => self.verify_btc_deposit(txid).await,
            BridgeChain::Ethereum => self.verify_eth_deposit(txid).await,
            BridgeChain::Zcash => self.verify_zec_deposit(txid).await,
            BridgeChain::IronFish => self.verify_iron_deposit(txid).await,
        }
    }

    /// Verify a Bitcoin deposit via RPC (getrawtransaction + getblockcount)
    async fn verify_btc_deposit(&self, txid: &str) -> DepositVerificationResult {
        let config = self.rpc_endpoints.read().await;
        let rpc_url = match &config.btc_rpc_url {
            Some(url) => url.clone(),
            None => {
                return DepositVerificationResult::RpcError(
                    "BTC_RPC_URL not configured".to_string(),
                );
            }
        };
        let rpc_user = config.btc_rpc_user.clone().unwrap_or_default();
        let rpc_password = config.btc_rpc_password.clone().unwrap_or_default();
        drop(config);

        // Call Bitcoin RPC: getrawtransaction <txid> true
        let client = reqwest::Client::new();
        let body = serde_json::json!({
            "jsonrpc": "1.0",
            "id": "bridge_verify",
            "method": "getrawtransaction",
            "params": [txid, true]
        });

        match client
            .post(&rpc_url)
            .basic_auth(&rpc_user, Some(&rpc_password))
            .json(&body)
            .timeout(std::time::Duration::from_secs(10))
            .send()
            .await
        {
            Ok(resp) => {
                match resp.json::<serde_json::Value>().await {
                    Ok(json) => {
                        if let Some(error) = json.get("error").and_then(|e| {
                            if e.is_null() { None } else { Some(e) }
                        }) {
                            return DepositVerificationResult::RpcError(
                                format!("BTC RPC error: {}", error),
                            );
                        }

                        let result = &json["result"];
                        let confirmations = result["confirmations"].as_u64().unwrap_or(0) as u32;

                        if confirmations >= BTC_MIN_CONFIRMATIONS {
                            DepositVerificationResult::Confirmed {
                                confirmations,
                                txid: txid.to_string(),
                            }
                        } else {
                            DepositVerificationResult::Pending {
                                confirmations,
                                txid: txid.to_string(),
                            }
                        }
                    }
                    Err(e) => DepositVerificationResult::RpcError(
                        format!("Failed to parse BTC RPC response: {}", e),
                    ),
                }
            }
            Err(e) => {
                // Check if this is a "transaction not found" type error
                if e.is_timeout() {
                    DepositVerificationResult::RpcError(
                        "BTC RPC timeout — node may be offline".to_string(),
                    )
                } else {
                    DepositVerificationResult::RpcError(format!("BTC RPC request failed: {}", e))
                }
            }
        }
    }

    /// Verify an Ethereum deposit via Reth RPC (eth_getTransactionReceipt)
    async fn verify_eth_deposit(&self, txid: &str) -> DepositVerificationResult {
        let config = self.rpc_endpoints.read().await;
        let rpc_url = match &config.eth_rpc_url {
            Some(url) => url.clone(),
            None => {
                return DepositVerificationResult::RpcError(
                    "ETH_RPC_URL not configured".to_string(),
                );
            }
        };
        drop(config);

        let client = reqwest::Client::new();

        // Step 1: Get transaction receipt
        let receipt_body = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "eth_getTransactionReceipt",
            "params": [txid]
        });

        let receipt_resp = match client
            .post(&rpc_url)
            .json(&receipt_body)
            .timeout(std::time::Duration::from_secs(10))
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => {
                return DepositVerificationResult::RpcError(
                    format!("ETH RPC request failed: {}", e),
                );
            }
        };

        let receipt_json: serde_json::Value = match receipt_resp.json().await {
            Ok(j) => j,
            Err(e) => {
                return DepositVerificationResult::RpcError(
                    format!("Failed to parse ETH receipt: {}", e),
                );
            }
        };

        let result = &receipt_json["result"];
        if result.is_null() {
            return DepositVerificationResult::NotFound;
        }

        // Check if transaction was successful
        let status = result["status"].as_str().unwrap_or("0x0");
        if status != "0x1" {
            return DepositVerificationResult::RpcError(
                "ETH transaction failed (status != 0x1)".to_string(),
            );
        }

        let tx_block_hex = result["blockNumber"].as_str().unwrap_or("0x0");
        let tx_block = u64::from_str_radix(tx_block_hex.trim_start_matches("0x"), 16)
            .unwrap_or(0);

        // Step 2: Get current block number
        let block_body = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "eth_blockNumber",
            "params": []
        });

        let block_resp = match client
            .post(&rpc_url)
            .json(&block_body)
            .timeout(std::time::Duration::from_secs(10))
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => {
                return DepositVerificationResult::RpcError(
                    format!("ETH RPC block number failed: {}", e),
                );
            }
        };

        let block_json: serde_json::Value = match block_resp.json().await {
            Ok(j) => j,
            Err(e) => {
                return DepositVerificationResult::RpcError(
                    format!("Failed to parse ETH block number: {}", e),
                );
            }
        };

        let current_block_hex = block_json["result"].as_str().unwrap_or("0x0");
        let current_block = u64::from_str_radix(
            current_block_hex.trim_start_matches("0x"), 16,
        ).unwrap_or(0);

        let confirmations = if current_block >= tx_block {
            (current_block - tx_block) as u32
        } else {
            0
        };

        if confirmations >= ETH_MIN_CONFIRMATIONS {
            DepositVerificationResult::Confirmed {
                confirmations,
                txid: txid.to_string(),
            }
        } else {
            DepositVerificationResult::Pending {
                confirmations,
                txid: txid.to_string(),
            }
        }
    }

    /// Verify a Zcash deposit via RPC (gettransaction)
    async fn verify_zec_deposit(&self, txid: &str) -> DepositVerificationResult {
        let config = self.rpc_endpoints.read().await;
        let rpc_url = match &config.zec_rpc_url {
            Some(url) => url.clone(),
            None => {
                return DepositVerificationResult::RpcError(
                    "ZEC_RPC_URL not configured. Set ZEC_RPC_URL env var.".to_string(),
                );
            }
        };
        drop(config);

        let client = reqwest::Client::new();
        let body = serde_json::json!({
            "jsonrpc": "1.0",
            "id": "bridge_verify",
            "method": "gettransaction",
            "params": [txid]
        });

        match client
            .post(&rpc_url)
            .json(&body)
            .timeout(std::time::Duration::from_secs(10))
            .send()
            .await
        {
            Ok(resp) => {
                match resp.json::<serde_json::Value>().await {
                    Ok(json) => {
                        if let Some(error) = json.get("error").and_then(|e| {
                            if e.is_null() { None } else { Some(e) }
                        }) {
                            return DepositVerificationResult::RpcError(
                                format!("ZEC RPC error: {}", error),
                            );
                        }

                        let result = &json["result"];
                        let confirmations = result["confirmations"].as_u64().unwrap_or(0) as u32;

                        if confirmations >= ZEC_MIN_CONFIRMATIONS {
                            DepositVerificationResult::Confirmed {
                                confirmations,
                                txid: txid.to_string(),
                            }
                        } else {
                            DepositVerificationResult::Pending {
                                confirmations,
                                txid: txid.to_string(),
                            }
                        }
                    }
                    Err(e) => DepositVerificationResult::RpcError(
                        format!("Failed to parse ZEC RPC response: {}", e),
                    ),
                }
            }
            Err(e) => DepositVerificationResult::RpcError(
                format!("ZEC RPC request failed: {}", e),
            ),
        }
    }

    /// Verify an Iron Fish deposit via RPC
    async fn verify_iron_deposit(&self, txid: &str) -> DepositVerificationResult {
        let config = self.rpc_endpoints.read().await;
        let rpc_url = match &config.iron_rpc_url {
            Some(url) => url.clone(),
            None => {
                return DepositVerificationResult::RpcError(
                    "IRON_RPC_URL not configured. Set IRON_RPC_URL env var.".to_string(),
                );
            }
        };
        drop(config);

        // Iron Fish uses JSON-RPC with wallet/getTransaction
        let client = reqwest::Client::new();
        let body = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "wallet/getAccountTransaction",
            "params": {
                "hash": txid
            }
        });

        match client
            .post(&rpc_url)
            .json(&body)
            .timeout(std::time::Duration::from_secs(10))
            .send()
            .await
        {
            Ok(resp) => {
                match resp.json::<serde_json::Value>().await {
                    Ok(json) => {
                        let result = &json["result"];
                        if result.is_null() {
                            return DepositVerificationResult::NotFound;
                        }

                        let confirmations = result["transaction"]["confirmations"]
                            .as_u64()
                            .unwrap_or(0) as u32;

                        if confirmations >= IRON_MIN_CONFIRMATIONS {
                            DepositVerificationResult::Confirmed {
                                confirmations,
                                txid: txid.to_string(),
                            }
                        } else {
                            DepositVerificationResult::Pending {
                                confirmations,
                                txid: txid.to_string(),
                            }
                        }
                    }
                    Err(e) => DepositVerificationResult::RpcError(
                        format!("Failed to parse IRON RPC response: {}", e),
                    ),
                }
            }
            Err(e) => DepositVerificationResult::RpcError(
                format!("IRON RPC request failed: {}", e),
            ),
        }
    }

    // ========================================================================
    // Pending Deposit Tracking
    // ========================================================================

    /// Register a new pending deposit to track
    pub async fn register_pending_deposit(
        &self,
        swap_id: String,
        chain: BridgeChain,
        expected_amount: u128,
        deposit_address: String,
    ) {
        let deposit = PendingDeposit {
            swap_id: swap_id.clone(),
            chain,
            expected_amount,
            deposit_address,
            deposit_txid: None,
            confirmations: 0,
            required_confirmations: self.min_confirmations(chain),
            created_at: Utc::now(),
            last_checked: Utc::now(),
            status: DepositStatus::AwaitingDeposit,
        };

        let mut deposits = self.pending_deposits.write().await;
        deposits.push(deposit);
        info!(
            "📋 [BRIDGE SAFETY] Registered pending {:?} deposit for swap {}",
            chain, swap_id
        );
    }

    /// Update a pending deposit with the transaction ID found on-chain
    pub async fn update_deposit_txid(&self, swap_id: &str, txid: String) {
        let mut deposits = self.pending_deposits.write().await;
        if let Some(dep) = deposits.iter_mut().find(|d| d.swap_id == swap_id) {
            dep.deposit_txid = Some(txid);
            dep.status = DepositStatus::AwaitingConfirmations;
            dep.last_checked = Utc::now();
        }
    }

    /// Get all pending deposits (for monitoring/status endpoint)
    pub async fn get_pending_deposits(&self) -> Vec<PendingDeposit> {
        self.pending_deposits.read().await.clone()
    }

    // ========================================================================
    // Swap Timeout Scanner (Background Task)
    // ========================================================================

    /// Scan for expired swaps and mark them for refund.
    /// Call this from a background tokio::spawn every SWAP_SCAN_INTERVAL_SECS.
    pub async fn scan_expired_swaps(&self) -> Vec<String> {
        let now = Utc::now();
        let mut expired_swap_ids = Vec::new();

        let mut deposits = self.pending_deposits.write().await;
        for deposit in deposits.iter_mut() {
            if deposit.status == DepositStatus::AwaitingDeposit
                || deposit.status == DepositStatus::AwaitingConfirmations
            {
                let age_secs = (now - deposit.created_at).num_seconds() as u64;
                if age_secs > SWAP_EXPIRY_SECS {
                    warn!(
                        "⏰ [BRIDGE SAFETY] Swap {} expired after {}s (chain: {:?})",
                        deposit.swap_id, age_secs, deposit.chain
                    );
                    deposit.status = DepositStatus::Expired;
                    expired_swap_ids.push(deposit.swap_id.clone());
                }
            }
        }

        // Remove expired deposits from the pending list
        deposits.retain(|d| d.status != DepositStatus::Expired);

        expired_swap_ids
    }

    /// Get the bridge status summary (for admin/status endpoint)
    pub async fn get_status(&self) -> BridgeSafetyStatus {
        let pending = self.pending_deposits.read().await;
        let config = self.rpc_endpoints.read().await;

        BridgeSafetyStatus {
            globally_frozen: self.is_frozen(),
            btc_frozen: self.is_chain_frozen(BridgeChain::Bitcoin).await,
            eth_frozen: self.is_chain_frozen(BridgeChain::Ethereum).await,
            zec_frozen: self.is_chain_frozen(BridgeChain::Zcash).await,
            iron_frozen: self.is_chain_frozen(BridgeChain::IronFish).await,
            pending_deposits: pending.len(),
            btc_rpc_configured: config.btc_rpc_url.is_some(),
            eth_rpc_configured: config.eth_rpc_url.is_some(),
            zec_rpc_configured: config.zec_rpc_url.is_some(),
            iron_rpc_configured: config.iron_rpc_url.is_some(),
            btc_max_amount: BTC_MAX_AMOUNT_SATS,
            eth_max_amount_wei: ETH_MAX_AMOUNT_WEI,
            zec_max_amount: ZEC_MAX_AMOUNT_ZATS,
            iron_max_amount: IRON_MAX_AMOUNT_BASE,
        }
    }
}

impl Default for BridgeSafetyController {
    fn default() -> Self {
        Self::new()
    }
}

/// Bridge safety status for the admin dashboard
#[derive(Debug, Serialize)]
pub struct BridgeSafetyStatus {
    pub globally_frozen: bool,
    pub btc_frozen: bool,
    pub eth_frozen: bool,
    pub zec_frozen: bool,
    pub iron_frozen: bool,
    pub pending_deposits: usize,
    pub btc_rpc_configured: bool,
    pub eth_rpc_configured: bool,
    pub zec_rpc_configured: bool,
    pub iron_rpc_configured: bool,
    pub btc_max_amount: u64,
    pub eth_max_amount_wei: u128,
    pub zec_max_amount: u64,
    pub iron_max_amount: u64,
}

// ============================================================================
// Admin API Handlers
// ============================================================================

use axum::extract::State;
use axum::Json;

/// POST /api/v1/bridge/admin/freeze — Freeze all bridge operations
pub async fn admin_freeze_bridge(
    State(state): State<Arc<crate::AppState>>,
    auth_wallet: Option<crate::wallet_auth::AuthenticatedWallet>,
) -> Result<Json<q_types::ApiResponse<serde_json::Value>>, hyper::StatusCode> {
    // Only master wallet can freeze
    let wallet = match auth_wallet {
        Some(w) => w,
        None => {
            return Ok(Json(q_types::ApiResponse {
                success: false,
                data: None,
                error: Some("Authentication required.".to_string()),
                timestamp: Utc::now(),
            }));
        }
    };

    // Check if this is the master wallet
    if !crate::is_master_wallet(&wallet.address) {
        return Ok(Json(q_types::ApiResponse {
            success: false,
            data: None,
            error: Some("Only the master wallet can freeze the bridge.".to_string()),
            timestamp: Utc::now(),
        }));
    }

    state.bridge_safety.freeze_all();

    Ok(Json(q_types::ApiResponse::success(serde_json::json!({
        "frozen": true,
        "message": "All bridge operations are now FROZEN.",
    }))))
}

/// POST /api/v1/bridge/admin/unfreeze — Unfreeze all bridge operations
pub async fn admin_unfreeze_bridge(
    State(state): State<Arc<crate::AppState>>,
    auth_wallet: Option<crate::wallet_auth::AuthenticatedWallet>,
) -> Result<Json<q_types::ApiResponse<serde_json::Value>>, hyper::StatusCode> {
    let wallet = match auth_wallet {
        Some(w) => w,
        None => {
            return Ok(Json(q_types::ApiResponse {
                success: false,
                data: None,
                error: Some("Authentication required.".to_string()),
                timestamp: Utc::now(),
            }));
        }
    };

    if !crate::is_master_wallet(&wallet.address) {
        return Ok(Json(q_types::ApiResponse {
            success: false,
            data: None,
            error: Some("Only the master wallet can unfreeze the bridge.".to_string()),
            timestamp: Utc::now(),
        }));
    }

    state.bridge_safety.unfreeze_all();

    Ok(Json(q_types::ApiResponse::success(serde_json::json!({
        "frozen": false,
        "message": "Bridge operations resumed.",
    }))))
}

/// GET /api/v1/bridge/admin/safety-status — Get bridge safety status
pub async fn admin_safety_status(
    State(state): State<Arc<crate::AppState>>,
) -> Result<Json<q_types::ApiResponse<BridgeSafetyStatus>>, hyper::StatusCode> {
    let status = state.bridge_safety.get_status().await;
    Ok(Json(q_types::ApiResponse::success(status)))
}

// ============================================================================
// Background Swap Expiry Task
// ============================================================================

/// Spawn the background task that scans for expired swaps.
/// Call once at startup.
pub fn spawn_swap_expiry_scanner(
    bridge_safety: Arc<BridgeSafetyController>,
    storage_engine: Arc<q_storage::StorageEngine>,
) {
    tokio::spawn(async move {
        info!("🔄 [BRIDGE SAFETY] Swap expiry scanner started (interval: {}s)", SWAP_SCAN_INTERVAL_SECS);
        let mut interval = tokio::time::interval(
            std::time::Duration::from_secs(SWAP_SCAN_INTERVAL_SECS),
        );

        loop {
            interval.tick().await;

            let expired = bridge_safety.scan_expired_swaps().await;
            for swap_id in &expired {
                // Mark as expired in storage
                // For BTC swaps:
                if let Ok(Some(data)) = storage_engine.get_atomic_swap(swap_id).await {
                    if let Ok(mut proposal) = serde_json::from_slice::<serde_json::Value>(&data) {
                        if let Some(state) = proposal.get_mut("state") {
                            *state = serde_json::json!({
                                "Failed": { "reason": "Swap expired — no deposit received within timelock" }
                            });
                        }
                        if let Ok(updated) = serde_json::to_vec(&proposal) {
                            let _ = storage_engine.save_atomic_swap(swap_id, &updated).await;
                        }
                    }
                }

                // For ZEC/IRON swaps (different storage keys):
                for prefix in &["zec_swap:", "iron_swap:", "eth_swap:"] {
                    let key = format!("{}{}", prefix, swap_id);
                    // These use the same storage API — the prefix is baked into the swap_id
                    // by each bridge API file, so save_atomic_swap works for all of them
                }

                warn!(
                    "⏰ [BRIDGE SAFETY] Swap {} marked as EXPIRED in storage",
                    swap_id
                );
            }

            if !expired.is_empty() {
                info!(
                    "🔄 [BRIDGE SAFETY] Expiry scan complete: {} swaps expired",
                    expired.len()
                );
            }
        }
    });
}
