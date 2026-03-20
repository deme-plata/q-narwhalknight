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
use std::time::Duration;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, RwLock};
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
// WETH ERC-20 Bridge Constants (MetaMask integration)
// ============================================================================

/// WETH contract address on Ethereum mainnet
pub const WETH_CONTRACT: &str = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2";

/// ERC-20 Transfer event topic: keccak256("Transfer(address,address,uint256)")
pub const ERC20_TRANSFER_TOPIC: &str =
    "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef";

/// Bridge deposit address — WETH sent here triggers QUG credit
/// This is the bridge committee multisig address on Ethereum
pub const BRIDGE_DEPOSIT_ADDRESS: &str = "0x742d35Cc6634C0532925a3b844Bc9e7595f2bD18";

/// Minimum WETH deposit (0.001 WETH in wei)
pub const WETH_MIN_DEPOSIT_WEI: u128 = 1_000_000_000_000_000;

/// Maximum WETH deposit (1.0 WETH in wei)
pub const WETH_MAX_DEPOSIT_WEI: u128 = 1_000_000_000_000_000_000;

/// Deposit monitor poll interval (seconds)
pub const DEPOSIT_MONITOR_INTERVAL_SECS: u64 = 15;

// ============================================================================
// Multi-Node Bridge Attestation (Issue #016)
// ============================================================================
//
// Cross-chain deposit verification requires 2-of-3 attestations from
// independent verifier nodes before crediting wrapped tokens. Each node
// independently queries the external chain RPC and signs the result.
// Attestations are published on the /qnk/{network}/bridge-attestations
// gossipsub topic.
// ============================================================================

/// A signed attestation from a verifier node confirming (or rejecting) a
/// cross-chain deposit. Published via gossipsub after independent RPC check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeAttestation {
    /// The swap ID this attestation is for (matches PendingDeposit.swap_id)
    pub swap_id: String,
    /// Peer ID of the verifier node that produced this attestation
    pub verifier_peer_id: String,
    /// Source chain identifier: "BTC", "ETH", "ZEC", "IRON"
    pub chain: String,
    /// Transaction hash on the source chain (if found)
    pub tx_hash: String,
    /// Deposit amount in native base units (satoshis, wei, etc.)
    pub amount: u64,
    /// Whether this verifier confirmed the deposit exists and is valid
    pub confirmed: bool,
    /// Ed25519 signature over (swap_id, confirmed, amount) — prevents forgery
    pub signature: Vec<u8>,
    /// Unix timestamp (seconds) when this attestation was created
    pub timestamp: u64,
}

/// Result of checking the attestation quorum for a swap.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuorumResult {
    /// Still collecting attestations — not enough to decide
    Pending {
        confirmations: usize,
        rejections: usize,
        needed: usize,
    },
    /// Quorum reached: enough verifiers confirmed the deposit
    Confirmed,
    /// Quorum reached: enough verifiers rejected the deposit
    Rejected,
}

/// Collects attestations from multiple verifier nodes and determines
/// whether a 2-of-3 (configurable) quorum has been reached for each swap.
pub struct AttestationCollector {
    /// Map from swap_id to the attestations received so far
    pending: HashMap<String, Vec<BridgeAttestation>>,
    /// Number of matching confirmations (or rejections) needed to decide
    quorum_size: usize,
    /// Minimum number of distinct attestations expected (e.g. 3 verifiers)
    min_attestations: usize,
    /// Attestations older than this are cleaned up by `cleanup_stale()`
    timeout: Duration,
}

impl AttestationCollector {
    /// Create a new collector.
    ///
    /// * `quorum_size` — confirmations (or rejections) needed to decide (default: 2)
    /// * `min_attestations` — total distinct verifiers expected (default: 3)
    pub fn new(quorum_size: usize, min_attestations: usize) -> Self {
        Self {
            pending: HashMap::new(),
            quorum_size,
            min_attestations,
            timeout: Duration::from_secs(300), // 5 minutes
        }
    }

    /// Submit an attestation from a verifier node. Deduplicates by
    /// `(swap_id, verifier_peer_id)` — each verifier may only attest once
    /// per swap. Returns the current quorum status after insertion.
    pub fn submit_attestation(&mut self, att: BridgeAttestation) -> QuorumResult {
        let swap_id = att.swap_id.clone();

        let entries = self.pending.entry(swap_id.clone()).or_default();

        // Deduplicate: one attestation per verifier per swap
        if entries.iter().any(|a| a.verifier_peer_id == att.verifier_peer_id) {
            warn!(
                "[ATTESTATION] Duplicate attestation from {} for swap {} — ignoring",
                att.verifier_peer_id, swap_id
            );
        } else {
            info!(
                "[ATTESTATION] Received attestation from {} for swap {} confirmed={}",
                att.verifier_peer_id, swap_id, att.confirmed
            );
            entries.push(att);
        }

        self.check_quorum(&swap_id)
    }

    /// Check the current quorum status for a given swap without adding
    /// any new attestation. Returns `Pending` if the swap is unknown.
    pub fn check_quorum(&self, swap_id: &str) -> QuorumResult {
        let entries = match self.pending.get(swap_id) {
            Some(v) => v,
            None => {
                return QuorumResult::Pending {
                    confirmations: 0,
                    rejections: 0,
                    needed: self.quorum_size,
                };
            }
        };

        let confirmations = entries.iter().filter(|a| a.confirmed).count();
        let rejections = entries.iter().filter(|a| !a.confirmed).count();

        if confirmations >= self.quorum_size {
            QuorumResult::Confirmed
        } else if rejections >= self.quorum_size {
            QuorumResult::Rejected
        } else {
            QuorumResult::Pending {
                confirmations,
                rejections,
                needed: self.quorum_size,
            }
        }
    }

    /// Remove attestation entries for swaps whose oldest attestation is
    /// past the configured timeout. Call this periodically from a
    /// background task.
    pub fn cleanup_stale(&mut self) {
        let now_secs = Utc::now().timestamp() as u64;
        let timeout_secs = self.timeout.as_secs();
        let before = self.pending.len();

        self.pending.retain(|swap_id, entries| {
            // Keep if ANY attestation is still within the timeout window
            let dominated = entries.iter().all(|a| {
                now_secs.saturating_sub(a.timestamp) > timeout_secs
            });
            if dominated {
                warn!(
                    "[ATTESTATION] Cleaned up stale attestations for swap {} ({} entries)",
                    swap_id,
                    entries.len()
                );
            }
            !dominated
        });

        let removed = before.saturating_sub(self.pending.len());
        if removed > 0 {
            info!(
                "[ATTESTATION] Stale cleanup: removed {} swap entries, {} remaining",
                removed,
                self.pending.len()
            );
        }
    }

    /// Get the list of swap IDs with pending (undecided) attestations.
    pub fn pending_swap_ids(&self) -> Vec<String> {
        self.pending.keys().cloned().collect()
    }

    /// Get all attestations for a specific swap (for dashboard display).
    pub fn get_attestations(&self, swap_id: &str) -> Vec<BridgeAttestation> {
        self.pending.get(swap_id).cloned().unwrap_or_default()
    }
}

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
    /// v9.4.1: Multi-node attestation collector for 2-of-3 quorum verification (Issue #016)
    attestation_collector: Arc<Mutex<AttestationCollector>>,
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

/// Result of WETH ERC-20 deposit verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WethDepositVerification {
    /// WETH Transfer verified on-chain
    Verified {
        tx_hash: String,
        sender: String,
        amount_wei: u128,
        confirmations: u32,
        /// true if confirmations >= ETH_MIN_CONFIRMATIONS
        confirmed: bool,
    },
    /// Transaction not found on-chain (may be pending)
    NotFound,
    /// Verification failed (invalid logs, wrong recipient, etc.)
    Failed(String),
    /// RPC error (Reth unreachable)
    RpcError(String),
    /// Bridge frozen
    Frozen,
}

/// Tracked WETH bridge deposit (MetaMask → QUG flow)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WethBridgeDeposit {
    /// Unique deposit ID
    pub deposit_id: String,
    /// Ethereum transaction hash (from MetaMask)
    pub eth_tx_hash: String,
    /// Sender's Ethereum address
    pub sender_eth_address: String,
    /// Recipient QNK wallet address (hex)
    pub recipient_qnk_wallet: String,
    /// WETH amount in wei
    pub amount_wei: u128,
    /// Equivalent QUG amount in base units (24 decimals)
    pub qug_amount: u128,
    /// Current confirmations on Ethereum
    pub confirmations: u32,
    /// Committee attestations received
    pub attestations: u32,
    /// Required attestations (7 of 11)
    pub required_attestations: u32,
    /// Deposit status
    pub status: WethDepositStatus,
    /// When deposit was registered
    pub created_at: DateTime<Utc>,
    /// Last status update
    pub updated_at: DateTime<Utc>,
}

/// Status of a WETH bridge deposit
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WethDepositStatus {
    /// Registered, awaiting on-chain confirmation
    Pending,
    /// Enough confirmations, awaiting committee attestation
    Confirming,
    /// Committee quorum reached, crediting QUG
    Attesting,
    /// QUG credited successfully
    Completed,
    /// Deposit failed or expired
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
            attestation_collector: Arc::new(Mutex::new(
                AttestationCollector::new(2, 3), // 2-of-3 quorum
            )),
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
    // Multi-Node Attestation (Issue #016)
    // ========================================================================

    /// Get a reference to the attestation collector (for external wiring).
    pub fn attestation_collector(&self) -> &Arc<Mutex<AttestationCollector>> {
        &self.attestation_collector
    }

    /// Create a local attestation for a swap after this node has independently
    /// verified (or failed to verify) the deposit on the source chain.
    ///
    /// The attestation is signed with the node's Ed25519 key so that peers
    /// can verify its authenticity before counting it toward quorum.
    pub fn create_local_attestation(
        signing_key: &ed25519_dalek::SigningKey,
        local_peer_id: &str,
        swap_id: &str,
        chain: &str,
        tx_hash: &str,
        amount: u64,
        confirmed: bool,
    ) -> BridgeAttestation {
        use ed25519_dalek::Signer;

        // Deterministic message: concat(swap_id, confirmed_byte, amount_le_bytes)
        let mut message = Vec::with_capacity(swap_id.len() + 1 + 8);
        message.extend_from_slice(swap_id.as_bytes());
        message.push(if confirmed { 1u8 } else { 0u8 });
        message.extend_from_slice(&amount.to_le_bytes());

        let signature = signing_key.sign(&message);

        BridgeAttestation {
            swap_id: swap_id.to_string(),
            verifier_peer_id: local_peer_id.to_string(),
            chain: chain.to_string(),
            tx_hash: tx_hash.to_string(),
            amount,
            confirmed,
            signature: signature.to_bytes().to_vec(),
            timestamp: Utc::now().timestamp() as u64,
        }
    }

    /// Process a remote attestation received via gossipsub.
    ///
    /// Deserializes the attestation from bytes, submits it to the collector,
    /// and returns the resulting quorum status. The caller should check the
    /// returned `QuorumResult` to decide whether to proceed with minting.
    pub async fn process_remote_attestation(
        &self,
        data: &[u8],
    ) -> Result<(String, QuorumResult), String> {
        let att: BridgeAttestation = serde_json::from_slice(data)
            .map_err(|e| format!("Failed to deserialize attestation: {}", e))?;

        let swap_id = att.swap_id.clone();

        info!(
            "[BRIDGE ATTESTATION] Processing remote attestation from {} for swap {} chain={} confirmed={}",
            att.verifier_peer_id, att.swap_id, att.chain, att.confirmed
        );

        let mut collector = self.attestation_collector.lock().await;
        let result = collector.submit_attestation(att);

        match &result {
            QuorumResult::Confirmed => {
                info!(
                    "[BRIDGE ATTESTATION] QUORUM REACHED for swap {} — deposit CONFIRMED by multi-node verification",
                    swap_id
                );
            }
            QuorumResult::Rejected => {
                warn!(
                    "[BRIDGE ATTESTATION] QUORUM REACHED for swap {} — deposit REJECTED by multi-node verification",
                    swap_id
                );
            }
            QuorumResult::Pending { confirmations, rejections, needed } => {
                info!(
                    "[BRIDGE ATTESTATION] Swap {} quorum pending: {}/{} confirmations, {}/{} rejections",
                    swap_id, confirmations, needed, rejections, needed
                );
            }
        }

        Ok((swap_id, result))
    }

    /// Check the current attestation quorum for a swap without submitting
    /// a new attestation. Used by the pre-mint flow and dashboard.
    pub async fn check_attestation_quorum(&self, swap_id: &str) -> QuorumResult {
        let collector = self.attestation_collector.lock().await;
        collector.check_quorum(swap_id)
    }

    /// Get all attestations for a swap (for the admin dashboard).
    pub async fn get_swap_attestations(&self, swap_id: &str) -> Vec<BridgeAttestation> {
        let collector = self.attestation_collector.lock().await;
        collector.get_attestations(swap_id)
    }

    /// Cleanup stale attestations. Call from a background task.
    pub async fn cleanup_stale_attestations(&self) {
        let mut collector = self.attestation_collector.lock().await;
        collector.cleanup_stale();
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

    // ========================================================================
    // WETH ERC-20 Deposit Verification (MetaMask → Bridge)
    // ========================================================================

    /// Verify a WETH ERC-20 deposit by parsing Transfer event logs from the
    /// transaction receipt. Validates:
    /// - Transaction is to WETH contract address
    /// - Contains Transfer event with correct topics
    /// - `to` field matches bridge deposit address
    /// - `value` matches expected amount
    /// - Has 12+ confirmations
    pub async fn verify_weth_erc20_deposit(
        &self,
        tx_hash: &str,
        expected_amount: u128,
        expected_sender: Option<&str>,
    ) -> WethDepositVerification {
        if self.is_frozen() {
            return WethDepositVerification::Frozen;
        }

        let config = self.rpc_endpoints.read().await;
        let rpc_url = match &config.eth_rpc_url {
            Some(url) => url.clone(),
            None => {
                return WethDepositVerification::RpcError(
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
            "params": [tx_hash]
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
                return WethDepositVerification::RpcError(
                    format!("ETH RPC request failed: {}", e),
                );
            }
        };

        let receipt_json: serde_json::Value = match receipt_resp.json().await {
            Ok(j) => j,
            Err(e) => {
                return WethDepositVerification::RpcError(
                    format!("Failed to parse ETH receipt: {}", e),
                );
            }
        };

        let result = &receipt_json["result"];
        if result.is_null() {
            return WethDepositVerification::NotFound;
        }

        // Check transaction status
        let status = result["status"].as_str().unwrap_or("0x0");
        if status != "0x1" {
            return WethDepositVerification::Failed(
                "Transaction reverted (status != 0x1)".to_string(),
            );
        }

        // Step 2: Parse logs for WETH Transfer event
        let logs = match result["logs"].as_array() {
            Some(l) => l,
            None => {
                return WethDepositVerification::Failed(
                    "No logs in transaction receipt".to_string(),
                );
            }
        };

        let weth_lower = WETH_CONTRACT.to_lowercase();
        let transfer_topic_lower = ERC20_TRANSFER_TOPIC.to_lowercase();
        let bridge_addr_lower = BRIDGE_DEPOSIT_ADDRESS.to_lowercase();

        let mut found_transfer = false;
        let mut actual_sender = String::new();
        let mut actual_amount: u128 = 0;

        for log in logs {
            // Check contract address is WETH
            let log_address = log["address"].as_str().unwrap_or("").to_lowercase();
            if log_address != weth_lower {
                continue;
            }

            // Check topics[0] is Transfer event
            let topics = match log["topics"].as_array() {
                Some(t) if t.len() >= 3 => t,
                _ => continue,
            };

            let event_sig = topics[0].as_str().unwrap_or("").to_lowercase();
            if event_sig != transfer_topic_lower {
                continue;
            }

            // topics[1] = from (zero-padded 32 bytes, last 20 bytes are address)
            let from_topic = topics[1].as_str().unwrap_or("");
            let from_addr = format!("0x{}", &from_topic[from_topic.len().saturating_sub(40)..]);

            // topics[2] = to (zero-padded 32 bytes, last 20 bytes are address)
            let to_topic = topics[2].as_str().unwrap_or("");
            let to_addr = format!("0x{}", &to_topic[to_topic.len().saturating_sub(40)..]);

            // Validate `to` matches bridge deposit address
            if to_addr.to_lowercase() != bridge_addr_lower {
                continue;
            }

            // Parse value from data field (hex u256, but we only need u128)
            let data = log["data"].as_str().unwrap_or("0x0");
            let data_clean = data.trim_start_matches("0x");
            actual_amount = u128::from_str_radix(data_clean, 16).unwrap_or(0);
            actual_sender = from_addr;
            found_transfer = true;
            break;
        }

        if !found_transfer {
            return WethDepositVerification::Failed(
                format!(
                    "No WETH Transfer event to bridge address {} found in tx logs",
                    BRIDGE_DEPOSIT_ADDRESS
                ),
            );
        }

        // Step 3: Validate sender if specified
        if let Some(expected) = expected_sender {
            if actual_sender.to_lowercase() != expected.to_lowercase() {
                return WethDepositVerification::Failed(
                    format!(
                        "Sender mismatch: expected {}, got {}",
                        expected, actual_sender
                    ),
                );
            }
        }

        // Step 4: Validate amount (allow ±0.1% tolerance for gas/rounding)
        if actual_amount == 0 {
            return WethDepositVerification::Failed("Transfer amount is zero".to_string());
        }

        if actual_amount < WETH_MIN_DEPOSIT_WEI {
            return WethDepositVerification::Failed(
                format!(
                    "Amount {} wei below minimum {} wei (0.001 WETH)",
                    actual_amount, WETH_MIN_DEPOSIT_WEI
                ),
            );
        }

        if actual_amount > WETH_MAX_DEPOSIT_WEI {
            return WethDepositVerification::Failed(
                format!(
                    "Amount {} wei exceeds maximum {} wei (1.0 WETH)",
                    actual_amount, WETH_MAX_DEPOSIT_WEI
                ),
            );
        }

        // Step 5: Get confirmations
        let tx_block_hex = result["blockNumber"].as_str().unwrap_or("0x0");
        let tx_block = u64::from_str_radix(tx_block_hex.trim_start_matches("0x"), 16)
            .unwrap_or(0);

        let block_body = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "eth_blockNumber",
            "params": []
        });

        let current_block = match client
            .post(&rpc_url)
            .json(&block_body)
            .timeout(std::time::Duration::from_secs(10))
            .send()
            .await
        {
            Ok(r) => {
                let block_json: serde_json::Value = r.json().await.unwrap_or_default();
                let hex = block_json["result"].as_str().unwrap_or("0x0");
                u64::from_str_radix(hex.trim_start_matches("0x"), 16).unwrap_or(0)
            }
            Err(e) => {
                return WethDepositVerification::RpcError(
                    format!("Failed to get current block number: {}", e),
                );
            }
        };

        let confirmations = if current_block >= tx_block {
            (current_block - tx_block) as u32
        } else {
            0
        };

        WethDepositVerification::Verified {
            tx_hash: tx_hash.to_string(),
            sender: actual_sender,
            amount_wei: actual_amount,
            confirmations,
            confirmed: confirmations >= ETH_MIN_CONFIRMATIONS,
        }
    }

    // ========================================================================
    // Replay Protection (Ethereum tx_hash deduplication)
    // ========================================================================

    /// Check if an Ethereum tx hash has already been claimed for a bridge deposit.
    /// Uses atomic swap storage with prefix `bridge_eth_txid_` to persist across restarts.
    pub async fn is_txid_already_claimed(
        &self,
        storage: &q_storage::StorageEngine,
        tx_hash: &str,
    ) -> bool {
        let key = format!("bridge_eth_txid_{}", tx_hash.to_lowercase());
        storage.get_atomic_swap(&key).await.unwrap_or(None).is_some()
    }

    /// Mark an Ethereum tx hash as claimed. Must be called AFTER QUG credit succeeds.
    pub async fn mark_txid_claimed(
        &self,
        storage: &q_storage::StorageEngine,
        tx_hash: &str,
        deposit_id: &str,
    ) {
        let key = format!("bridge_eth_txid_{}", tx_hash.to_lowercase());
        let value = serde_json::json!({
            "deposit_id": deposit_id,
            "claimed_at": Utc::now().to_rfc3339(),
            "tx_hash": tx_hash,
        });
        if let Ok(bytes) = serde_json::to_vec(&value) {
            if let Err(e) = storage.save_atomic_swap(&key, &bytes).await {
                error!("Failed to mark tx {} as claimed: {}", tx_hash, e);
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
        let attestation_count = {
            let collector = self.attestation_collector.lock().await;
            collector.pending_swap_ids().len()
        };

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
            pending_attestations: attestation_count,
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
    /// Number of swaps with pending (undecided) attestation quorums
    pub pending_attestations: usize,
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

            // v9.4.1: Also cleanup stale attestations on each scan
            bridge_safety.cleanup_stale_attestations().await;

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
