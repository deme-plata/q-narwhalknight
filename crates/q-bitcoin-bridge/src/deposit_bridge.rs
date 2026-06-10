/// Bitcoin Deposit Bridge for QUG — Phase 1
///
/// Simple one-way bridge: users deposit BTC on-chain, bridge mints wBTC on QUG DEX.
/// Uses Bitcoin Knots wallet RPC for address generation and deposit tracking.
///
/// Security model (reviewed by Claude Code + Gemma4 2026-04-10):
/// - 6 confirmations before minting
/// - 0.1 BTC max per deposit
/// - 1 BTC max total bridge TVL (kill switch)
/// - Single-threaded mint processor (no race conditions)
/// - Atomic dedup: txid:vout key written with mint in single WriteBatch
/// - Reorg monitoring: background job re-checks last 10 blocks every 60s
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, Mutex};
use tracing::{debug, error, info, warn};

use crate::real_bitcoin_client::{BitcoinConfig, RealBitcoinClient};

// ============================================================================
// Constants
// ============================================================================

/// Minimum confirmations before minting wBTC (Gemma4 recommended 6+)
pub const BTC_MIN_CONFIRMATIONS: u32 = 6;

/// Maximum deposit amount in satoshis (0.1 BTC)
pub const BTC_MAX_DEPOSIT_SATS: u64 = 10_000_000;

/// Maximum total bridge TVL in satoshis (1 BTC) — kill switch triggers above this
pub const BTC_MAX_BRIDGE_TVL_SATS: u64 = 100_000_000;

/// Minimum deposit amount in satoshis (0.0001 BTC = 10,000 sats, covers dust + fees)
pub const BTC_MIN_DEPOSIT_SATS: u64 = 10_000;

/// Polling interval for deposit detection
pub const DEPOSIT_POLL_INTERVAL_SECS: u64 = 15;

/// Reorg check interval (re-verify recent mints)
pub const REORG_CHECK_INTERVAL_SECS: u64 = 60;

/// Maximum deposit addresses per wallet per hour
pub const MAX_ADDRS_PER_WALLET_PER_HOUR: u32 = 3;

/// Deposit address expiry (48 hours — after this, address is no longer monitored)
pub const DEPOSIT_ADDR_EXPIRY_SECS: u64 = 48 * 3600;

/// Bitcoin Knots wallet name
pub const WALLET_NAME: &str = "qug-bridge";

// ============================================================================
// Types
// ============================================================================

/// Deposit status state machine
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DepositStatus {
    /// Address generated, waiting for user to send BTC
    Awaiting,
    /// Transaction detected in mempool or with < MIN_CONFIRMATIONS
    Detected { txid: String, vout: u32, confirmations: u32 },
    /// Enough confirmations, minting wBTC
    Confirming { txid: String, vout: u32, confirmations: u32 },
    /// wBTC minted successfully
    Minted { txid: String, vout: u32, mint_op_id: String },
    /// Deposit expired (no BTC received within expiry window)
    Expired,
    /// Error during processing
    Failed { reason: String },
}

/// A single deposit address record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepositAddress {
    /// Unique deposit ID
    pub deposit_id: String,
    /// Bitcoin address (bech32, bc1q...)
    pub btc_address: String,
    /// QUG wallet that will receive wBTC
    pub qug_wallet: [u8; 32],
    /// Bitcoin Knots wallet label
    pub label: String,
    /// Current status
    pub status: DepositStatus,
    /// Amount received in satoshis (0 until detected)
    pub amount_sats: u64,
    /// Created timestamp (unix)
    pub created_at: u64,
    /// Last updated timestamp (unix)
    pub updated_at: u64,
}

/// Wallet transaction info from Bitcoin Knots listtransactions RPC
#[derive(Debug, Clone, Deserialize)]
pub struct WalletTransaction {
    pub address: Option<String>,
    pub category: String,  // "receive", "send", "generate", "immature"
    pub amount: f64,
    pub label: Option<String>,
    pub vout: Option<u32>,
    pub confirmations: Option<i64>,
    pub blockhash: Option<String>,
    pub blockheight: Option<u64>,
    pub blockindex: Option<u32>,
    pub blocktime: Option<u64>,
    pub txid: Option<String>,
    pub time: Option<u64>,
    pub timereceived: Option<u64>,
}

/// Deposit event emitted to SSE/frontend
#[derive(Debug, Clone, Serialize)]
pub struct DepositEvent {
    pub deposit_id: String,
    pub btc_address: String,
    pub status: String,
    pub amount_sats: u64,
    pub confirmations: u32,
    pub qug_wallet: String,
}

// ============================================================================
// Wallet RPC Client (extends RealBitcoinClient for wallet-specific calls)
// ============================================================================

/// Bitcoin Knots wallet RPC client — wraps RealBitcoinClient with wallet-specific methods
pub struct BridgeWalletClient {
    config: BitcoinConfig,
    client: reqwest::Client,
    request_id: std::sync::atomic::AtomicU64,
}

impl BridgeWalletClient {
    /// Create a new wallet RPC client pointing at the qug-bridge wallet
    pub async fn new(config: BitcoinConfig) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(config.request_timeout)
            .connect_timeout(config.connection_timeout)
            .user_agent("Q-NarwhalKnight-Bridge/1.0")
            .build()?;

        let wallet_client = Self {
            config,
            client,
            request_id: std::sync::atomic::AtomicU64::new(1),
        };

        // Verify wallet is loaded
        let info: Value = wallet_client.wallet_rpc("getwalletinfo", json!([])).await?;
        let wallet_name = info.get("walletname").and_then(|v| v.as_str()).unwrap_or("unknown");
        if wallet_name != WALLET_NAME {
            return Err(anyhow!("Expected wallet '{}', got '{}'", WALLET_NAME, wallet_name));
        }
        let balance = info.get("balance").and_then(|v| v.as_f64()).unwrap_or(0.0);
        info!("₿ Bridge wallet '{}' loaded, balance: {} BTC", wallet_name, balance);

        Ok(wallet_client)
    }

    /// Make an RPC call to the Bitcoin Knots wallet endpoint
    async fn wallet_rpc<T: for<'de> Deserialize<'de>>(
        &self,
        method: &str,
        params: Value,
    ) -> Result<T> {
        let id = self.request_id.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        // Wallet-specific endpoint: /wallet/<wallet_name>
        let url = format!("{}/wallet/{}", self.config.rpc_url, WALLET_NAME);

        let payload = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });

        let mut attempts = 0;
        let mut last_error = None;

        while attempts < self.config.max_retries {
            attempts += 1;

            match self.client
                .post(&url)
                .basic_auth(&self.config.rpc_user, Some(&self.config.rpc_password))
                .json(&payload)
                .send()
                .await
            {
                Ok(resp) => {
                    if resp.status().is_success() {
                        let body: Value = resp.json().await?;
                        if let Some(error) = body.get("error").filter(|e| !e.is_null()) {
                            let code = error.get("code").and_then(|c| c.as_i64()).unwrap_or(-1);
                            let msg = error.get("message").and_then(|m| m.as_str()).unwrap_or("unknown");
                            return Err(anyhow!("RPC error {}: {}", code, msg));
                        }
                        if let Some(result) = body.get("result") {
                            return Ok(serde_json::from_value(result.clone())?);
                        }
                        return Err(anyhow!("RPC returned null result for {}", method));
                    } else {
                        last_error = Some(anyhow!("HTTP {}", resp.status()));
                    }
                }
                Err(e) => {
                    last_error = Some(anyhow!("Request failed: {}", e));
                }
            }

            if attempts < self.config.max_retries {
                warn!("₿ Wallet RPC {} failed (attempt {}), retrying...", method, attempts);
                tokio::time::sleep(Duration::from_millis(1000 * attempts as u64)).await;
            }
        }

        Err(last_error.unwrap_or_else(|| anyhow!("All wallet RPC attempts failed for {}", method)))
    }

    // ========================================================================
    // Wallet-specific RPC methods
    // ========================================================================

    /// Generate a new deposit address with label and address type
    pub async fn get_new_deposit_address(
        &self,
        label: &str,
        address_type: &str,
    ) -> Result<String> {
        self.wallet_rpc("getnewaddress", json!([label, address_type])).await
    }

    /// Get amount received at a specific address (with min confirmations filter)
    pub async fn get_received_by_address(
        &self,
        address: &str,
        min_conf: u32,
    ) -> Result<f64> {
        self.wallet_rpc("getreceivedbyaddress", json!([address, min_conf])).await
    }

    /// List transactions filtered by label
    pub async fn list_transactions(
        &self,
        label: &str,
        count: u32,
    ) -> Result<Vec<WalletTransaction>> {
        self.wallet_rpc("listtransactions", json!([label, count, 0, true])).await
    }

    /// Get wallet transaction by txid (wallet-aware, returns confirmations etc.)
    pub async fn get_wallet_transaction(&self, txid: &str) -> Result<Value> {
        self.wallet_rpc("gettransaction", json!([txid, true, true])).await
    }

    /// Get total wallet balance
    pub async fn get_balance(&self) -> Result<f64> {
        self.wallet_rpc("getbalance", json!(["*", 0])).await
    }

    /// Get wallet info (balance, tx count, keypool, etc.)
    pub async fn get_wallet_info(&self) -> Result<Value> {
        self.wallet_rpc("getwalletinfo", json!([])).await
    }

    /// List all labels in wallet
    pub async fn list_labels(&self) -> Result<Vec<String>> {
        self.wallet_rpc("listlabels", json!([])).await
    }

    /// Get addresses by label
    pub async fn get_addresses_by_label(&self, label: &str) -> Result<Value> {
        self.wallet_rpc("getaddressesbylabel", json!([label])).await
    }

    /// Send BTC to an external address — returns txid
    pub async fn send_to_address(&self, btc_address: &str, amount_btc: f64) -> Result<String> {
        self.wallet_rpc("sendtoaddress", json!([btc_address, amount_btc])).await
    }

    /// Send BTC to an external address with a target confirmation horizon.
    /// `conf_target` is in blocks; Knots picks fee via fee estimation.
    /// Empty `comment`/`comment_to` are required positional placeholders.
    pub async fn send_to_address_with_target(
        &self,
        btc_address: &str,
        amount_btc: f64,
        conf_target: u32,
    ) -> Result<String> {
        // Positional args: address, amount, comment, comment_to, subtractfeefromamount,
        //   replaceable, conf_target, estimate_mode
        self.wallet_rpc(
            "sendtoaddress",
            json!([btc_address, amount_btc, "", "", false, true, conf_target, "economical"]),
        )
        .await
    }

    /// Get spendable wallet balance in satoshis (min 1 confirmation)
    pub async fn get_balance_sats(&self) -> Result<u64> {
        let btc: f64 = self.wallet_rpc("getbalance", json!(["*", 1])).await?;
        Ok((btc * 100_000_000.0).round() as u64)
    }
}

// ============================================================================
// Deposit Bridge Manager
// ============================================================================

/// Configuration for the deposit bridge
#[derive(Debug, Clone)]
pub struct DepositBridgeConfig {
    pub btc_rpc_url: String,
    pub btc_rpc_user: String,
    pub btc_rpc_pass: String,
    pub min_confirmations: u32,
    pub max_deposit_sats: u64,
    pub max_bridge_tvl_sats: u64,
    pub poll_interval: Duration,
    pub enabled: bool,
}

impl Default for DepositBridgeConfig {
    fn default() -> Self {
        Self {
            // SECURITY: No hardcoded credentials — require env vars
            btc_rpc_url: String::new(),
            btc_rpc_user: String::new(),
            btc_rpc_pass: String::new(),
            min_confirmations: BTC_MIN_CONFIRMATIONS,
            max_deposit_sats: BTC_MAX_DEPOSIT_SATS,
            max_bridge_tvl_sats: BTC_MAX_BRIDGE_TVL_SATS,
            poll_interval: Duration::from_secs(DEPOSIT_POLL_INTERVAL_SECS),
            enabled: true,
        }
    }
}

impl DepositBridgeConfig {
    /// Create config from environment variables.
    /// Returns None if required BTC_RPC_URL is not set (bridge disabled).
    pub fn from_env() -> Option<Self> {
        let url = std::env::var("BTC_RPC_URL").ok()?;
        if url.is_empty() {
            return None;
        }

        let user = std::env::var("BTC_RPC_USER").unwrap_or_default();
        let pass = std::env::var("BTC_RPC_PASS").unwrap_or_default();

        if user.is_empty() || pass.is_empty() {
            warn!("₿ BTC_RPC_URL is set but BTC_RPC_USER/BTC_RPC_PASS missing — bridge disabled");
            return None;
        }

        let mut min_confirmations = BTC_MIN_CONFIRMATIONS;
        if let Ok(confs) = std::env::var("BTC_MIN_CONFIRMATIONS") {
            if let Ok(n) = confs.parse::<u32>() {
                min_confirmations = n;
            }
        }
        // SECURITY: Never allow fewer than 3 confirmations
        min_confirmations = min_confirmations.max(3);

        let enabled = std::env::var("BTC_BRIDGE_ENABLED")
            .map(|v| v != "0" && v.to_lowercase() != "false")
            .unwrap_or(true);

        Some(Self {
            btc_rpc_url: url,
            btc_rpc_user: user,
            btc_rpc_pass: pass,
            min_confirmations,
            max_deposit_sats: BTC_MAX_DEPOSIT_SATS,
            max_bridge_tvl_sats: BTC_MAX_BRIDGE_TVL_SATS,
            poll_interval: Duration::from_secs(DEPOSIT_POLL_INTERVAL_SECS),
            enabled,
        })
    }
}

/// The main deposit bridge manager
///
/// Handles address generation, deposit detection, confirmation tracking,
/// and wBTC minting. Processes mints sequentially through a channel to
/// prevent race conditions (per Gemma4 review).
pub struct DepositBridge {
    /// Wallet RPC client
    wallet: Arc<BridgeWalletClient>,
    /// Bridge configuration
    config: DepositBridgeConfig,
    /// Pending deposits indexed by deposit_id
    pending_deposits: Arc<Mutex<HashMap<String, DepositAddress>>>,
    /// BTC address → deposit_id lookup
    addr_to_deposit: Arc<Mutex<HashMap<String, String>>>,
    /// SECURITY: Minted txid:vout dedup set — prevents double-minting
    /// Key: "txid:vout", Value: deposit_id
    minted_txids: Arc<Mutex<HashMap<String, String>>>,
    /// SECURITY: Rate limit tracker — wallet_hex → Vec<timestamp>
    addr_gen_timestamps: Arc<Mutex<HashMap<String, Vec<u64>>>>,
    /// Mint event sender (for SSE notifications)
    event_tx: mpsc::UnboundedSender<DepositEvent>,
    /// Kill switch — set to true to halt all minting
    killed: Arc<std::sync::atomic::AtomicBool>,
    /// Total minted satoshis (for TVL check)
    total_minted_sats: Arc<std::sync::atomic::AtomicU64>,
}

impl DepositBridge {
    /// Create a new deposit bridge
    pub async fn new(
        config: DepositBridgeConfig,
        event_tx: mpsc::UnboundedSender<DepositEvent>,
    ) -> Result<Self> {
        let btc_config = BitcoinConfig {
            rpc_url: config.btc_rpc_url.clone(),
            rpc_user: config.btc_rpc_user.clone(),
            rpc_password: config.btc_rpc_pass.clone(),
            network: bitcoin::network::constants::Network::Bitcoin,
            tor_proxy: None,
            connection_timeout: Duration::from_secs(10),
            request_timeout: Duration::from_secs(30),
            max_retries: 3,
        };

        let wallet = Arc::new(BridgeWalletClient::new(btc_config).await?);

        Ok(Self {
            wallet,
            config,
            pending_deposits: Arc::new(Mutex::new(HashMap::new())),
            addr_to_deposit: Arc::new(Mutex::new(HashMap::new())),
            minted_txids: Arc::new(Mutex::new(HashMap::new())),
            addr_gen_timestamps: Arc::new(Mutex::new(HashMap::new())),
            event_tx,
            killed: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            total_minted_sats: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        })
    }

    /// Check if bridge is operational
    pub fn is_alive(&self) -> bool {
        self.config.enabled && !self.killed.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Emergency kill switch
    pub fn kill(&self) {
        warn!("🛑 BTC BRIDGE KILL SWITCH ACTIVATED");
        self.killed.store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// Generate a new deposit address for a QUG wallet
    pub async fn create_deposit_address(
        &self,
        qug_wallet: [u8; 32],
    ) -> Result<DepositAddress> {
        if !self.is_alive() {
            return Err(anyhow!("Bridge is disabled or killed"));
        }

        // SECURITY: Rate limiting — max 3 addresses per wallet per hour
        let wallet_hex = hex::encode(&qug_wallet[..8]);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        {
            let mut timestamps = self.addr_gen_timestamps.lock().await;
            let entries = timestamps.entry(wallet_hex.clone()).or_default();
            // Remove entries older than 1 hour
            entries.retain(|&t| now - t < 3600);
            if entries.len() >= MAX_ADDRS_PER_WALLET_PER_HOUR as usize {
                return Err(anyhow!(
                    "Rate limit: max {} addresses per wallet per hour",
                    MAX_ADDRS_PER_WALLET_PER_HOUR
                ));
            }
            entries.push(now);
        }

        let deposit_id = uuid::Uuid::new_v4().to_string();
        let label = format!("deposit:{}", wallet_hex);

        // Generate bech32 address via Bitcoin Knots wallet
        let btc_address = self.wallet
            .get_new_deposit_address(&label, "bech32")
            .await?;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let deposit = DepositAddress {
            deposit_id: deposit_id.clone(),
            btc_address: btc_address.clone(),
            qug_wallet,
            label,
            status: DepositStatus::Awaiting,
            amount_sats: 0,
            created_at: now,
            updated_at: now,
        };

        // Track in memory
        {
            let mut pending = self.pending_deposits.lock().await;
            pending.insert(deposit_id.clone(), deposit.clone());
        }
        {
            let mut addr_map = self.addr_to_deposit.lock().await;
            addr_map.insert(btc_address.clone(), deposit_id.clone());
        }

        info!(
            "₿ New deposit address {} for QUG wallet {}",
            btc_address,
            hex::encode(&qug_wallet[..8])
        );

        // Emit event
        let _ = self.event_tx.send(DepositEvent {
            deposit_id,
            btc_address: btc_address.clone(),
            status: "awaiting".to_string(),
            amount_sats: 0,
            confirmations: 0,
            qug_wallet: hex::encode(qug_wallet),
        });

        Ok(deposit)
    }

    /// Get deposit status by ID
    pub async fn get_deposit(&self, deposit_id: &str) -> Option<DepositAddress> {
        let pending = self.pending_deposits.lock().await;
        pending.get(deposit_id).cloned()
    }

    /// Get deposit status by ID, only if owned by the given wallet (IDOR protection)
    pub async fn get_deposit_for_wallet(
        &self,
        deposit_id: &str,
        qug_wallet: &[u8; 32],
    ) -> Option<DepositAddress> {
        let pending = self.pending_deposits.lock().await;
        pending.get(deposit_id)
            .filter(|d| d.qug_wallet == *qug_wallet)
            .cloned()
    }

    /// Check if a txid:vout has already been minted (dedup check)
    pub async fn is_txid_minted(&self, txid: &str, vout: u32) -> bool {
        let dedup_key = format!("{}:{}", txid, vout);
        let minted = self.minted_txids.lock().await;
        minted.contains_key(&dedup_key)
    }

    /// List all deposits for a QUG wallet
    pub async fn list_deposits_for_wallet(&self, qug_wallet: &[u8; 32]) -> Vec<DepositAddress> {
        let pending = self.pending_deposits.lock().await;
        pending.values()
            .filter(|d| d.qug_wallet == *qug_wallet)
            .cloned()
            .collect()
    }

    /// Check whether the bridge wallet has enough spendable BTC to back `required_sats` of new wBTC.
    /// Returns Ok(true) if sufficient, Ok(false) if not, Err if the RPC call fails.
    pub async fn check_reserve_available(&self, required_sats: u64) -> Result<bool> {
        if !self.is_alive() {
            return Ok(false);
        }
        let balance_sats = self.wallet.get_balance_sats().await?;
        let already_issued = self.total_minted_sats.load(std::sync::atomic::Ordering::Relaxed);
        let available = balance_sats.saturating_sub(already_issued);
        Ok(available >= required_sats)
    }

    /// Send BTC to a user's on-chain address as part of a wBTC withdrawal.
    ///
    /// SECURITY: This MUST only be called AFTER the caller has already deducted
    /// the wBTC from the user's balance in RocksDB. The caller is responsible for
    /// that atomic deduction. Returns the on-chain txid.
    ///
    /// `conf_target` selects Knots fee estimation horizon (blocks). Pass `None`
    /// for the wallet default.
    pub async fn send_withdrawal(
        &self,
        btc_address: &str,
        amount_sats: u64,
        conf_target: Option<u32>,
    ) -> Result<String> {
        if !self.is_alive() {
            return Err(anyhow!("Bridge is disabled"));
        }
        if amount_sats < BTC_MIN_DEPOSIT_SATS {
            return Err(anyhow!("Amount {} sats below minimum {} sats", amount_sats, BTC_MIN_DEPOSIT_SATS));
        }
        if amount_sats > BTC_MAX_DEPOSIT_SATS {
            return Err(anyhow!("Amount {} sats exceeds maximum {} sats per withdrawal", amount_sats, BTC_MAX_DEPOSIT_SATS));
        }

        // Verify we have reserves
        let balance_sats = self.wallet.get_balance_sats().await?;
        if balance_sats < amount_sats {
            return Err(anyhow!("Insufficient BTC reserves: have {} sats, need {} sats", balance_sats, amount_sats));
        }

        let amount_btc = amount_sats as f64 / 100_000_000.0;
        let txid = match conf_target {
            Some(t) => self.wallet.send_to_address_with_target(btc_address, amount_btc, t).await?,
            None => self.wallet.send_to_address(btc_address, amount_btc).await?,
        };

        // Reduce total_minted_sats to reflect that BTC left the bridge
        let prev = self.total_minted_sats.fetch_update(
            std::sync::atomic::Ordering::SeqCst,
            std::sync::atomic::Ordering::SeqCst,
            |v| Some(v.saturating_sub(amount_sats)),
        );
        let _ = prev;

        info!("₿ Withdrawal sent: {} sats → {} (txid: {})", amount_sats, btc_address, txid);
        Ok(txid)
    }

    /// Poll for deposit updates — called every DEPOSIT_POLL_INTERVAL_SECS
    ///
    /// This is the core deposit detection loop. It queries Bitcoin Knots
    /// for transactions matching our deposit labels and updates statuses.
    pub async fn poll_deposits(&self) -> Result<Vec<DepositAddress>> {
        if !self.is_alive() {
            return Ok(vec![]);
        }

        let mut updated = Vec::new();
        let deposit_ids: Vec<(String, DepositAddress)> = {
            let pending = self.pending_deposits.lock().await;
            pending.iter()
                .filter(|(_, d)| matches!(d.status,
                    DepositStatus::Awaiting |
                    DepositStatus::Detected { .. } |
                    DepositStatus::Confirming { .. }
                ))
                .map(|(id, d)| (id.clone(), d.clone()))
                .collect()
        };

        for (deposit_id, deposit) in deposit_ids {
            match self.check_deposit_status(&deposit).await {
                Ok(Some(new_status)) => {
                    let mut pending = self.pending_deposits.lock().await;
                    if let Some(d) = pending.get_mut(&deposit_id) {
                        let now = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs();

                        // Update amount if detected
                        if let DepositStatus::Detected { ref txid, vout, confirmations } = new_status {
                            if d.amount_sats == 0 {
                                // Fetch actual amount from transaction
                                if let Ok(amount) = self.get_deposit_amount(txid, vout).await {
                                    d.amount_sats = amount;
                                }
                            }
                        }
                        if let DepositStatus::Confirming { ref txid, vout, confirmations } = new_status {
                            if d.amount_sats == 0 {
                                if let Ok(amount) = self.get_deposit_amount(txid, vout).await {
                                    d.amount_sats = amount;
                                }
                            }
                        }

                        d.status = new_status.clone();
                        d.updated_at = now;
                        updated.push(d.clone());

                        // Emit status event
                        let (status_str, confs) = match &new_status {
                            DepositStatus::Detected { confirmations, .. } =>
                                ("detected".to_string(), *confirmations),
                            DepositStatus::Confirming { confirmations, .. } =>
                                ("confirming".to_string(), *confirmations),
                            DepositStatus::Minted { .. } =>
                                ("minted".to_string(), self.config.min_confirmations),
                            _ => ("unknown".to_string(), 0),
                        };

                        let _ = self.event_tx.send(DepositEvent {
                            deposit_id: deposit_id.clone(),
                            btc_address: d.btc_address.clone(),
                            status: status_str,
                            amount_sats: d.amount_sats,
                            confirmations: confs,
                            qug_wallet: hex::encode(d.qug_wallet),
                        });
                    }
                }
                Ok(None) => {
                    // No change
                }
                Err(e) => {
                    warn!("₿ Error checking deposit {}: {}", deposit_id, e);
                }
            }
        }

        Ok(updated)
    }

    /// Check the status of a single deposit against Bitcoin Knots
    async fn check_deposit_status(
        &self,
        deposit: &DepositAddress,
    ) -> Result<Option<DepositStatus>> {
        // Check expiry
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        if matches!(deposit.status, DepositStatus::Awaiting)
            && now - deposit.created_at > DEPOSIT_ADDR_EXPIRY_SECS
        {
            return Ok(Some(DepositStatus::Expired));
        }

        // Query transactions for this deposit's label
        let txs = self.wallet
            .list_transactions(&deposit.label, 10)
            .await?;

        // Find receive transactions for our address
        for tx in &txs {
            if tx.category != "receive" {
                continue;
            }
            let tx_addr = match &tx.address {
                Some(a) => a,
                None => continue,
            };
            if tx_addr != &deposit.btc_address {
                continue;
            }
            let txid = match &tx.txid {
                Some(t) => t.clone(),
                None => continue,
            };
            let vout = tx.vout.unwrap_or(0);
            let confirmations = tx.confirmations.unwrap_or(0).max(0) as u32;

            // Amount check — use round() to avoid IEEE 754 precision loss
            let amount_sats = (tx.amount.abs() * 100_000_000.0).round() as u64;
            if amount_sats < BTC_MIN_DEPOSIT_SATS {
                debug!("₿ Deposit too small: {} sats (min {})", amount_sats, BTC_MIN_DEPOSIT_SATS);
                continue;
            }
            if amount_sats > self.config.max_deposit_sats {
                warn!("₿ Deposit exceeds max: {} sats (max {})", amount_sats, self.config.max_deposit_sats);
                return Ok(Some(DepositStatus::Failed {
                    reason: format!("Amount {} sats exceeds maximum {} sats", amount_sats, self.config.max_deposit_sats),
                }));
            }

            // TVL check
            let current_tvl = self.total_minted_sats.load(std::sync::atomic::Ordering::Relaxed);
            if current_tvl + amount_sats > self.config.max_bridge_tvl_sats {
                warn!("₿ Bridge TVL would exceed max: {} + {} > {}",
                    current_tvl, amount_sats, self.config.max_bridge_tvl_sats);
                self.kill();
                return Ok(Some(DepositStatus::Failed {
                    reason: "Bridge TVL limit reached".to_string(),
                }));
            }

            // SECURITY: Check txid:vout dedup set — prevent double-minting
            let dedup_key = format!("{}:{}", txid, vout);
            {
                let minted = self.minted_txids.lock().await;
                if minted.contains_key(&dedup_key) {
                    debug!("₿ Skipping already-minted UTXO {}", dedup_key);
                    continue;
                }
            }

            // State transition based on confirmations
            if confirmations >= self.config.min_confirmations {
                // Ready to mint! Return Confirming — the actual mint happens in the mint processor
                return Ok(Some(DepositStatus::Confirming { txid, vout, confirmations }));
            } else {
                return Ok(Some(DepositStatus::Detected { txid, vout, confirmations }));
            }
        }

        Ok(None) // No transaction found yet
    }

    /// Get the deposit amount in satoshis from a transaction output
    async fn get_deposit_amount(&self, txid: &str, vout: u32) -> Result<u64> {
        let tx: Value = self.wallet.get_wallet_transaction(txid).await?;
        let amount = tx.get("amount")
            .and_then(|a| a.as_f64())
            .unwrap_or(0.0);
        Ok((amount.abs() * 100_000_000.0).round() as u64)
    }

    /// Get all deposits that are ready to mint (have enough confirmations)
    pub async fn get_mintable_deposits(&self) -> Vec<DepositAddress> {
        let pending = self.pending_deposits.lock().await;
        pending.values()
            .filter(|d| matches!(d.status, DepositStatus::Confirming { .. }))
            .cloned()
            .collect()
    }

    /// Mark a deposit as minted (called by the mint processor after successful mint)
    ///
    /// SECURITY: Records txid:vout in dedup set to prevent double-minting.
    /// This must be called atomically with the actual wBTC mint operation.
    pub async fn mark_minted(
        &self,
        deposit_id: &str,
        txid: String,
        vout: u32,
        mint_op_id: String,
    ) -> Result<()> {
        // SECURITY: Write dedup key FIRST (before updating status)
        let dedup_key = format!("{}:{}", txid, vout);
        {
            let mut minted = self.minted_txids.lock().await;
            if minted.contains_key(&dedup_key) {
                return Err(anyhow!(
                    "DOUBLE-MINT BLOCKED: UTXO {} already minted as deposit {}",
                    dedup_key,
                    minted.get(&dedup_key).unwrap()
                ));
            }
            minted.insert(dedup_key.clone(), deposit_id.to_string());
        }

        let mut pending = self.pending_deposits.lock().await;
        if let Some(deposit) = pending.get_mut(deposit_id) {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();

            // Update TVL counter
            self.total_minted_sats.fetch_add(
                deposit.amount_sats,
                std::sync::atomic::Ordering::Relaxed,
            );

            deposit.status = DepositStatus::Minted {
                txid: txid.clone(),
                vout,
                mint_op_id: mint_op_id.clone(),
            };
            deposit.updated_at = now;

            info!(
                "₿ ✅ Deposit {} minted: {} sats → QUG wallet {}",
                deposit_id,
                deposit.amount_sats,
                hex::encode(&deposit.qug_wallet[..8])
            );

            // Emit minted event
            let _ = self.event_tx.send(DepositEvent {
                deposit_id: deposit_id.to_string(),
                btc_address: deposit.btc_address.clone(),
                status: "minted".to_string(),
                amount_sats: deposit.amount_sats,
                confirmations: self.config.min_confirmations,
                qug_wallet: hex::encode(deposit.qug_wallet),
            });
        }
        Ok(())
    }

    /// Mark a deposit as failed
    pub async fn mark_failed(&self, deposit_id: &str, reason: String) {
        let mut pending = self.pending_deposits.lock().await;
        if let Some(deposit) = pending.get_mut(deposit_id) {
            deposit.status = DepositStatus::Failed { reason: reason.clone() };
            error!("₿ ❌ Deposit {} failed: {}", deposit_id, reason);
        }
    }

    /// Get bridge status summary
    pub async fn get_status(&self) -> BridgeStatus {
        let pending = self.pending_deposits.lock().await;
        let wallet_balance = self.wallet.get_balance().await.unwrap_or(0.0);

        let mut awaiting = 0u32;
        let mut detected = 0u32;
        let mut confirming = 0u32;
        let mut minted = 0u32;
        let mut failed = 0u32;

        for d in pending.values() {
            match d.status {
                DepositStatus::Awaiting => awaiting += 1,
                DepositStatus::Detected { .. } => detected += 1,
                DepositStatus::Confirming { .. } => confirming += 1,
                DepositStatus::Minted { .. } => minted += 1,
                DepositStatus::Failed { .. } | DepositStatus::Expired => failed += 1,
            }
        }

        BridgeStatus {
            alive: self.is_alive(),
            wallet_balance_btc: wallet_balance,
            total_minted_sats: self.total_minted_sats.load(std::sync::atomic::Ordering::Relaxed),
            deposits_awaiting: awaiting,
            deposits_detected: detected,
            deposits_confirming: confirming,
            deposits_minted: minted,
            deposits_failed: failed,
            max_deposit_sats: self.config.max_deposit_sats,
            max_tvl_sats: self.config.max_bridge_tvl_sats,
            min_confirmations: self.config.min_confirmations,
        }
    }

    /// Load pending deposits from storage (call on startup)
    pub async fn load_pending_deposits(&self, deposits: Vec<DepositAddress>) {
        let mut pending = self.pending_deposits.lock().await;
        let mut addr_map = self.addr_to_deposit.lock().await;
        let mut minted = self.minted_txids.lock().await;
        let mut tvl = 0u64;

        for d in deposits {
            addr_map.insert(d.btc_address.clone(), d.deposit_id.clone());

            // Rebuild minted dedup set and TVL from previously minted deposits
            if let DepositStatus::Minted { ref txid, vout, .. } = d.status {
                let dedup_key = format!("{}:{}", txid, vout);
                minted.insert(dedup_key, d.deposit_id.clone());
                tvl += d.amount_sats;
            }

            pending.insert(d.deposit_id.clone(), d);
        }

        self.total_minted_sats.store(tvl, std::sync::atomic::Ordering::Relaxed);
        info!("₿ Loaded {} deposits from storage ({} minted, TVL: {} sats)",
            pending.len(), minted.len(), tvl);
    }

    /// Get all deposits for persistence (call periodically to save to RocksDB)
    pub async fn get_all_deposits(&self) -> Vec<DepositAddress> {
        let pending = self.pending_deposits.lock().await;
        pending.values().cloned().collect()
    }
}

/// Bridge status summary for API/dashboard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeStatus {
    pub alive: bool,
    pub wallet_balance_btc: f64,
    pub total_minted_sats: u64,
    pub deposits_awaiting: u32,
    pub deposits_detected: u32,
    pub deposits_confirming: u32,
    pub deposits_minted: u32,
    pub deposits_failed: u32,
    pub max_deposit_sats: u64,
    pub max_tvl_sats: u64,
    pub min_confirmations: u32,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deposit_status_transitions() {
        let status = DepositStatus::Awaiting;
        assert_eq!(status, DepositStatus::Awaiting);

        let detected = DepositStatus::Detected {
            txid: "abc123".to_string(),
            vout: 0,
            confirmations: 1,
        };
        assert!(matches!(detected, DepositStatus::Detected { confirmations: 1, .. }));

        let confirming = DepositStatus::Confirming {
            txid: "abc123".to_string(),
            vout: 0,
            confirmations: 6,
        };
        assert!(matches!(confirming, DepositStatus::Confirming { confirmations: 6, .. }));
    }

    #[test]
    fn test_deposit_bridge_config_defaults() {
        let config = DepositBridgeConfig::default();
        assert_eq!(config.min_confirmations, 6);
        assert_eq!(config.max_deposit_sats, 10_000_000);
        assert_eq!(config.max_bridge_tvl_sats, 100_000_000);
        assert!(config.enabled);
    }

    #[test]
    fn test_satoshi_conversion() {
        let btc = 0.1_f64;
        let sats = (btc * 100_000_000.0) as u64;
        assert_eq!(sats, 10_000_000);

        let btc2 = 0.0001_f64;
        let sats2 = (btc2 * 100_000_000.0) as u64;
        assert_eq!(sats2, 10_000);
    }

    #[test]
    fn test_deposit_address_serialization() {
        let deposit = DepositAddress {
            deposit_id: "test-id".to_string(),
            btc_address: "bc1qtest".to_string(),
            qug_wallet: [0u8; 32],
            label: "deposit:00000000".to_string(),
            status: DepositStatus::Awaiting,
            amount_sats: 0,
            created_at: 1000,
            updated_at: 1000,
        };

        let json = serde_json::to_string(&deposit).unwrap();
        let decoded: DepositAddress = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.deposit_id, "test-id");
        assert_eq!(decoded.btc_address, "bc1qtest");
    }
}
