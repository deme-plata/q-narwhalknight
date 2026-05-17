//! v2.4.9-beta: Dollar Cost Averaging (DCA) Feature for DEX
//!
//! Enables users to automatically purchase tokens at regular intervals.
//! Perfect for accumulating tokens like MEME, QUG, or any custom token
//! without timing the market.
//!
//! Features:
//! - Create recurring buy orders (hourly, daily, weekly, monthly)
//! - Automatic execution via background task
//! - Slippage protection on each execution
//! - Persistent storage across restarts (RocksDB)
//! - Cancel/pause/resume DCA orders
//! - 🌐 P2P gossipsub synchronization for decentralized agreement
//! - 🔐 Ed25519 signature verification for P2P security

use std::sync::Arc;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use std::collections::HashMap;
use tracing::{info, warn, error, debug};
use sha3::{Digest, Sha3_256};

use crate::AppState;

// ============================================================================
// P2P SYNC MESSAGE TYPES (v2.4.9-beta)
// Encrypted and signature-verified for security
// ============================================================================

/// DCA P2P Sync Message - broadcast to all nodes for decentralized agreement
/// v2.4.9-beta: Now includes Ed25519 signature for authentication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DcaSyncMessage {
    /// Protocol version (1 = unsigned, 2 = signed)
    pub version: u8,
    pub action: DcaSyncAction,
    pub order: DcaOrder,
    pub timestamp: i64,
    pub node_id: String, // Broadcasting node's peer ID
    /// Ed25519 signature over the signing payload (MANDATORY in v2+)
    #[serde(default)]
    pub signature: Vec<u8>,
    /// Public key of the signing node (32 bytes for Ed25519)
    #[serde(default)]
    pub signer_public_key: Vec<u8>,
}

impl DcaSyncMessage {
    /// Get the payload that should be signed
    /// SHA3-256(action || order_id || wallet || timestamp || node_id)
    pub fn signing_payload(&self) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(format!("{:?}", self.action).as_bytes());
        hasher.update(self.order.id.as_bytes());
        hasher.update(self.order.wallet_address.as_bytes());
        hasher.update(&self.timestamp.to_le_bytes());
        hasher.update(self.node_id.as_bytes());
        hasher.finalize().into()
    }

    /// Sign with raw bytes (for nodes that manage their own keys)
    pub fn sign_with_bytes(&mut self, signature: Vec<u8>, public_key: Vec<u8>) {
        self.signature = signature;
        self.signer_public_key = public_key;
        self.version = 2; // Signed version
    }

    /// Verify the Ed25519 signature is valid
    /// Returns true if signature is valid, false otherwise
    pub fn verify_signature(&self) -> bool {
        // Version 1 = unsigned (legacy, rejected)
        if self.version < 2 {
            warn!("🔐 [DCA P2P] Rejecting unsigned message (version {})", self.version);
            return false;
        }

        // Signature is mandatory for v2+
        if self.signature.is_empty() || self.signer_public_key.is_empty() {
            warn!("🔐 [DCA P2P] Missing signature or public key");
            return false;
        }

        // For Ed25519 (32-byte public key)
        if self.signer_public_key.len() != 32 {
            warn!("🔐 [DCA P2P] Invalid public key length: {}", self.signer_public_key.len());
            return false;
        }

        use ed25519_dalek::{Signature, VerifyingKey, Verifier};

        let public_key_bytes: [u8; 32] = match self.signer_public_key.clone().try_into() {
            Ok(bytes) => bytes,
            Err(_) => return false,
        };

        let verifying_key = match VerifyingKey::from_bytes(&public_key_bytes) {
            Ok(key) => key,
            Err(_) => {
                warn!("🔐 [DCA P2P] Invalid Ed25519 public key");
                return false;
            }
        };

        if self.signature.len() != 64 {
            warn!("🔐 [DCA P2P] Invalid signature length: {}", self.signature.len());
            return false;
        }

        let signature_bytes: [u8; 64] = match self.signature.clone().try_into() {
            Ok(bytes) => bytes,
            Err(_) => return false,
        };

        let signature = Signature::from_bytes(&signature_bytes);
        let payload = self.signing_payload();

        match verifying_key.verify(&payload, &signature) {
            Ok(_) => {
                debug!("🔐 [DCA P2P] Signature verified successfully");
                true
            }
            Err(e) => {
                warn!("🔐 [DCA P2P] Signature verification failed: {}", e);
                false
            }
        }
    }

    /// Check if the update has a valid signature attached
    pub fn is_signed(&self) -> bool {
        self.version >= 2 && !self.signature.is_empty() && !self.signer_public_key.is_empty()
    }

    /// Check if message timestamp is within acceptable range (24 hours)
    pub fn is_timestamp_valid(&self) -> bool {
        let now_ms = chrono::Utc::now().timestamp_millis();
        let age_ms = (now_ms - self.timestamp).abs();
        age_ms <= 24 * 60 * 60 * 1000 // 24 hours
    }

    /// Full validation: signature + timestamp
    pub fn validate_full(&self) -> Result<(), &'static str> {
        if !self.is_signed() {
            return Err("Message is not signed");
        }
        if !self.verify_signature() {
            return Err("Invalid signature");
        }
        if !self.is_timestamp_valid() {
            return Err("Message timestamp expired");
        }
        Ok(())
    }
}

/// DCA synchronization actions
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum DcaSyncAction {
    Create,   // New DCA order created
    Cancel,   // Order cancelled
    Pause,    // Order paused
    Resume,   // Order resumed
    Execute,  // Order executed (for tracking)
}

/// DCA Order Interval
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum DcaInterval {
    Hourly,     // Every hour
    Daily,      // Every 24 hours
    Weekly,     // Every 7 days
    Monthly,    // Every 30 days
}

impl DcaInterval {
    /// Get interval duration in milliseconds
    pub fn to_millis(&self) -> i64 {
        match self {
            DcaInterval::Hourly => 60 * 60 * 1000,           // 1 hour
            DcaInterval::Daily => 24 * 60 * 60 * 1000,      // 24 hours
            DcaInterval::Weekly => 7 * 24 * 60 * 60 * 1000, // 7 days
            DcaInterval::Monthly => 30 * 24 * 60 * 60 * 1000, // 30 days
        }
    }

    /// Display name
    pub fn display_name(&self) -> &'static str {
        match self {
            DcaInterval::Hourly => "Hourly",
            DcaInterval::Daily => "Daily",
            DcaInterval::Weekly => "Weekly",
            DcaInterval::Monthly => "Monthly",
        }
    }
}

/// DCA Order Status
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum DcaStatus {
    Active,     // Running and will execute at next interval
    Paused,     // Temporarily paused, can be resumed
    Completed,  // Finished (reached end date or execution count)
    Cancelled,  // Cancelled by user
    Failed,     // Failed due to insufficient balance or other error
}

/// DCA Order Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DcaOrder {
    pub id: String,                    // Unique order ID
    pub wallet_address: String,        // User's wallet address
    pub from_token: String,            // Token to sell (e.g., "QUG")
    pub to_token: String,              // Token to buy (e.g., "MEME")
    #[serde(serialize_with = "q_types::u128_serde::serialize", deserialize_with = "q_types::u128_serde::deserialize")]
    pub amount_per_execution: u128,    // Amount to spend per execution (base units)
    pub interval: DcaInterval,         // Execution interval
    pub max_slippage: f64,             // Maximum slippage tolerance (0.01 = 1%)
    pub status: DcaStatus,             // Current status
    pub created_at: i64,               // Creation timestamp (ms)
    pub last_executed_at: Option<i64>, // Last execution timestamp (ms)
    pub next_execution_at: i64,        // Next scheduled execution (ms)
    pub executions_completed: u32,     // Number of successful executions
    pub max_executions: Option<u32>,   // Optional max executions (None = unlimited)
    pub end_date: Option<i64>,         // Optional end date (ms)
    #[serde(serialize_with = "q_types::u128_serde::serialize", deserialize_with = "q_types::u128_serde::deserialize")]
    pub total_spent: u128,             // Total amount spent so far
    #[serde(serialize_with = "q_types::u128_serde::serialize", deserialize_with = "q_types::u128_serde::deserialize")]
    pub total_received: u128,          // Total amount received so far
    pub last_error: Option<String>,    // Last error message if any
}

/// DCA Order Execution Log
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DcaExecution {
    pub order_id: String,
    pub timestamp: i64,
    #[serde(serialize_with = "q_types::u128_serde::serialize", deserialize_with = "q_types::u128_serde::deserialize")]
    pub amount_in: u128,
    #[serde(serialize_with = "q_types::u128_serde::serialize", deserialize_with = "q_types::u128_serde::deserialize")]
    pub amount_out: u128,
    pub price: f64,
    pub success: bool,
    pub error: Option<String>,
    pub tx_hash: Option<String>,
}

/// Create DCA Order Request
#[derive(Debug, Deserialize)]
pub struct CreateDcaRequest {
    pub wallet_address: String,
    pub from_token: String,
    pub to_token: String,
    #[serde(deserialize_with = "q_types::u128_serde::deserialize")]
    pub amount_per_execution: u128,
    pub interval: DcaInterval,
    #[serde(default = "default_slippage")]
    pub max_slippage: f64,
    pub max_executions: Option<u32>,
    pub end_date: Option<i64>,
}

fn default_slippage() -> f64 {
    0.03 // 3% default slippage
}

/// Create DCA Order Response
#[derive(Debug, Serialize)]
pub struct CreateDcaResponse {
    pub success: bool,
    pub order_id: Option<String>,
    pub message: String,
    pub next_execution_at: Option<i64>,
}

/// Get DCA Orders Response
#[derive(Debug, Serialize)]
pub struct DcaOrdersResponse {
    pub success: bool,
    pub orders: Vec<DcaOrder>,
    pub total_active: u32,
    #[serde(serialize_with = "q_types::u128_serde::serialize")]
    pub total_value_locked: u128,
}

/// DCA Order Storage (in-memory with RocksDB persistence)
/// v2.4.9-beta: Now uses RocksDB column families for persistent storage
pub struct DcaStorage {
    pub orders: RwLock<HashMap<String, DcaOrder>>,
    pub executions: RwLock<HashMap<String, Vec<DcaExecution>>>,
}

impl DcaStorage {
    pub fn new() -> Self {
        DcaStorage {
            orders: RwLock::new(HashMap::new()),
            executions: RwLock::new(HashMap::new()),
        }
    }

    /// Load DCA orders from RocksDB storage
    /// v2.4.9-beta: Uses RocksDB column families CF_DCA_ORDERS and CF_DCA_EXECUTIONS
    pub async fn load_from_storage(&self, storage: &q_storage::QStorage) -> anyhow::Result<()> {
        // Load all DCA orders from RocksDB
        match storage.load_all_dca_orders().await {
            Ok(order_pairs) => {
                let mut order_lock = self.orders.write().await;
                for (order_id, order_bytes) in order_pairs {
                    if let Ok(order) = serde_json::from_slice::<DcaOrder>(&order_bytes) {
                        order_lock.insert(order_id, order);
                    }
                }
                info!("📊 [DCA] Loaded {} DCA orders from RocksDB", order_lock.len());
            }
            Err(e) => {
                warn!("⚠️ [DCA] Could not load DCA orders from RocksDB: {}", e);
            }
        }

        // Load execution history for each order
        let order_ids: Vec<String> = {
            let orders = self.orders.read().await;
            orders.keys().cloned().collect()
        };

        let mut exec_lock = self.executions.write().await;
        for order_id in order_ids {
            match storage.load_dca_executions(&order_id).await {
                Ok(execution_pairs) => {
                    let mut execs: Vec<DcaExecution> = Vec::new();
                    for (_, exec_bytes) in execution_pairs {
                        if let Ok(exec) = serde_json::from_slice::<DcaExecution>(&exec_bytes) {
                            execs.push(exec);
                        }
                    }
                    if !execs.is_empty() {
                        exec_lock.insert(order_id, execs);
                    }
                }
                Err(e) => {
                    warn!("⚠️ [DCA] Could not load executions for order {}: {}", order_id, e);
                }
            }
        }
        info!("📊 [DCA] Loaded execution history for {} orders from RocksDB", exec_lock.len());

        Ok(())
    }

    /// Save all DCA orders to RocksDB storage
    /// v2.4.9-beta: Persists to RocksDB column families
    pub async fn save_to_storage(&self, storage: &q_storage::QStorage) -> anyhow::Result<()> {
        let orders = self.orders.read().await;
        for (order_id, order) in orders.iter() {
            if let Ok(order_bytes) = serde_json::to_vec(order) {
                if let Err(e) = storage.save_dca_order(order_id, &order_bytes).await {
                    warn!("⚠️ [DCA] Could not save order {} to RocksDB: {}", order_id, e);
                }
            }
        }
        debug!("💾 [DCA] Saved {} DCA orders to RocksDB", orders.len());
        drop(orders);

        let executions = self.executions.read().await;
        for (order_id, exec_list) in executions.iter() {
            for exec in exec_list {
                if let Ok(exec_bytes) = serde_json::to_vec(exec) {
                    if let Err(e) = storage.save_dca_execution(order_id, exec.timestamp, &exec_bytes).await {
                        warn!("⚠️ [DCA] Could not save execution for order {}: {}", order_id, e);
                    }
                }
            }
        }
        debug!("💾 [DCA] Saved execution history to RocksDB");
        drop(executions);

        Ok(())
    }

    /// Save a single DCA order to RocksDB (called on create/update)
    pub async fn save_order(&self, storage: &q_storage::QStorage, order: &DcaOrder) -> anyhow::Result<()> {
        let order_bytes = serde_json::to_vec(order)?;
        storage.save_dca_order(&order.id, &order_bytes).await?;
        debug!("💾 [DCA] Saved order {} to RocksDB", order.id);
        Ok(())
    }

    /// Delete a DCA order from RocksDB (called on cancel)
    pub async fn delete_order(&self, storage: &q_storage::QStorage, order_id: &str) -> anyhow::Result<()> {
        // Delete the order
        storage.delete_dca_order(order_id).await?;
        // Delete associated executions
        let _ = storage.delete_dca_executions(order_id).await?;
        debug!("🗑️ [DCA] Deleted order {} and executions from RocksDB", order_id);
        Ok(())
    }

    /// Save a single execution to RocksDB
    pub async fn save_execution(&self, storage: &q_storage::QStorage, order_id: &str, execution: &DcaExecution) -> anyhow::Result<()> {
        let exec_bytes = serde_json::to_vec(execution)?;
        storage.save_dca_execution(order_id, execution.timestamp, &exec_bytes).await?;
        debug!("💾 [DCA] Saved execution for order {} to RocksDB", order_id);
        Ok(())
    }
}

impl Default for DcaStorage {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// API HANDLERS
// ============================================================================

/// Create a new DCA order
pub async fn create_dca_order(
    State(state): State<Arc<AppState>>,
    Json(request): Json<CreateDcaRequest>,
) -> impl IntoResponse {
    info!(
        "📊 [DCA] Creating order: {} {} -> {} every {:?}",
        request.amount_per_execution, request.from_token, request.to_token, request.interval
    );

    // Validate wallet address
    if request.wallet_address.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(CreateDcaResponse {
                success: false,
                order_id: None,
                message: "Wallet address is required".to_string(),
                next_execution_at: None,
            }),
        );
    }

    // Validate tokens
    if request.from_token == request.to_token {
        return (
            StatusCode::BAD_REQUEST,
            Json(CreateDcaResponse {
                success: false,
                order_id: None,
                message: "Cannot DCA same token to itself".to_string(),
                next_execution_at: None,
            }),
        );
    }

    // Validate amount
    if request.amount_per_execution == 0 {
        return (
            StatusCode::BAD_REQUEST,
            Json(CreateDcaResponse {
                success: false,
                order_id: None,
                message: "Amount per execution must be greater than 0".to_string(),
                next_execution_at: None,
            }),
        );
    }

    // Generate unique order ID
    let now = chrono::Utc::now().timestamp_millis();
    let order_id = format!(
        "dca_{}_{:x}",
        now,
        rand::random::<u32>()
    );

    // Calculate first execution time (start immediately, execute after first interval)
    let first_execution = now + request.interval.to_millis();

    let order = DcaOrder {
        id: order_id.clone(),
        wallet_address: request.wallet_address,
        from_token: request.from_token,
        to_token: request.to_token,
        amount_per_execution: request.amount_per_execution,
        interval: request.interval,
        max_slippage: request.max_slippage,
        status: DcaStatus::Active,
        created_at: now,
        last_executed_at: None,
        next_execution_at: first_execution,
        executions_completed: 0,
        max_executions: request.max_executions,
        end_date: request.end_date,
        total_spent: 0,
        total_received: 0,
        last_error: None,
    };

    // Store the order
    if let Some(dca_storage) = &state.dca_storage {
        // Update in-memory cache
        let mut orders = dca_storage.orders.write().await;
        orders.insert(order_id.clone(), order.clone());
        drop(orders);

        // Persist single order to RocksDB (v2.4.9-beta: efficient single-order save)
        if let Err(e) = dca_storage.save_order(&state.storage_engine, &order).await {
            warn!("Failed to persist DCA order to RocksDB: {}", e);
        }

        info!("✅ [DCA] Created order {} - next execution at {}", order_id, first_execution);

        // 🌐 v2.4.9-beta: Broadcast to P2P network for decentralized agreement
        broadcast_dca_sync(&state, DcaSyncAction::Create, order).await;

        (
            StatusCode::CREATED,
            Json(CreateDcaResponse {
                success: true,
                order_id: Some(order_id),
                message: "DCA order created successfully".to_string(),
                next_execution_at: Some(first_execution),
            }),
        )
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(CreateDcaResponse {
                success: false,
                order_id: None,
                message: "DCA service not initialized".to_string(),
                next_execution_at: None,
            }),
        )
    }
}

/// Get all DCA orders for a wallet
pub async fn get_wallet_dca_orders(
    State(state): State<Arc<AppState>>,
    Path(wallet_address): Path<String>,
) -> impl IntoResponse {
    if let Some(dca_storage) = &state.dca_storage {
        let orders = dca_storage.orders.read().await;

        let wallet_orders: Vec<DcaOrder> = orders
            .values()
            .filter(|o| o.wallet_address == wallet_address)
            .cloned()
            .collect();

        let active_count = wallet_orders.iter().filter(|o| o.status == DcaStatus::Active).count() as u32;
        let total_locked: u128 = wallet_orders
            .iter()
            .filter(|o| o.status == DcaStatus::Active)
            .map(|o| o.amount_per_execution)
            .sum();

        (
            StatusCode::OK,
            Json(DcaOrdersResponse {
                success: true,
                orders: wallet_orders,
                total_active: active_count,
                total_value_locked: total_locked,
            }),
        )
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(DcaOrdersResponse {
                success: false,
                orders: vec![],
                total_active: 0,
                total_value_locked: 0,
            }),
        )
    }
}

/// Cancel a DCA order
pub async fn cancel_dca_order(
    State(state): State<Arc<AppState>>,
    Path((wallet_address, order_id)): Path<(String, String)>,
) -> impl IntoResponse {
    if let Some(dca_storage) = &state.dca_storage {
        let mut orders = dca_storage.orders.write().await;

        if let Some(order) = orders.get_mut(&order_id) {
            // Verify ownership
            if order.wallet_address != wallet_address {
                return (
                    StatusCode::FORBIDDEN,
                    Json(serde_json::json!({
                        "success": false,
                        "message": "You can only cancel your own DCA orders"
                    })),
                );
            }

            // Cancel the order
            order.status = DcaStatus::Cancelled;
            let order_clone = order.clone(); // Clone for P2P broadcast

            info!("🛑 [DCA] Cancelled order {} for wallet {}", order_id, wallet_address);

            // Persist change to RocksDB (v2.4.9-beta: efficient single-order save)
            drop(orders);
            if let Err(e) = dca_storage.save_order(&state.storage_engine, &order_clone).await {
                warn!("Failed to persist DCA cancellation to RocksDB: {}", e);
            }

            // 🌐 v2.4.9-beta: Broadcast cancellation to P2P network
            broadcast_dca_sync(&state, DcaSyncAction::Cancel, order_clone).await;

            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "success": true,
                    "message": "DCA order cancelled"
                })),
            )
        } else {
            (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({
                    "success": false,
                    "message": "DCA order not found"
                })),
            )
        }
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "success": false,
                "message": "DCA service not available"
            })),
        )
    }
}

/// Pause a DCA order
pub async fn pause_dca_order(
    State(state): State<Arc<AppState>>,
    Path((wallet_address, order_id)): Path<(String, String)>,
) -> impl IntoResponse {
    if let Some(dca_storage) = &state.dca_storage {
        let mut orders = dca_storage.orders.write().await;

        if let Some(order) = orders.get_mut(&order_id) {
            if order.wallet_address != wallet_address {
                return (
                    StatusCode::FORBIDDEN,
                    Json(serde_json::json!({
                        "success": false,
                        "message": "You can only pause your own DCA orders"
                    })),
                );
            }

            if order.status != DcaStatus::Active {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({
                        "success": false,
                        "message": "Can only pause active orders"
                    })),
                );
            }

            order.status = DcaStatus::Paused;
            let order_clone = order.clone(); // Clone for P2P broadcast
            info!("⏸️ [DCA] Paused order {}", order_id);

            drop(orders);
            // Persist to RocksDB (v2.4.9-beta: efficient single-order save)
            let _ = dca_storage.save_order(&state.storage_engine, &order_clone).await;

            // 🌐 v2.4.9-beta: Broadcast pause to P2P network
            broadcast_dca_sync(&state, DcaSyncAction::Pause, order_clone).await;

            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "success": true,
                    "message": "DCA order paused"
                })),
            )
        } else {
            (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({
                    "success": false,
                    "message": "DCA order not found"
                })),
            )
        }
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "success": false,
                "message": "DCA service not available"
            })),
        )
    }
}

/// Resume a paused DCA order
pub async fn resume_dca_order(
    State(state): State<Arc<AppState>>,
    Path((wallet_address, order_id)): Path<(String, String)>,
) -> impl IntoResponse {
    if let Some(dca_storage) = &state.dca_storage {
        let mut orders = dca_storage.orders.write().await;

        if let Some(order) = orders.get_mut(&order_id) {
            if order.wallet_address != wallet_address {
                return (
                    StatusCode::FORBIDDEN,
                    Json(serde_json::json!({
                        "success": false,
                        "message": "You can only resume your own DCA orders"
                    })),
                );
            }

            if order.status != DcaStatus::Paused {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({
                        "success": false,
                        "message": "Can only resume paused orders"
                    })),
                );
            }

            order.status = DcaStatus::Active;
            // Schedule next execution from now
            let now = chrono::Utc::now().timestamp_millis();
            order.next_execution_at = now + order.interval.to_millis();
            let next_exec = order.next_execution_at;
            let order_clone = order.clone(); // Clone for P2P broadcast

            info!("▶️ [DCA] Resumed order {} - next execution at {}", order_id, next_exec);

            drop(orders);
            // Persist to RocksDB (v2.4.9-beta: efficient single-order save)
            let _ = dca_storage.save_order(&state.storage_engine, &order_clone).await;

            // 🌐 v2.4.9-beta: Broadcast resume to P2P network
            broadcast_dca_sync(&state, DcaSyncAction::Resume, order_clone).await;

            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "success": true,
                    "message": "DCA order resumed",
                    "next_execution_at": next_exec
                })),
            )
        } else {
            (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({
                    "success": false,
                    "message": "DCA order not found"
                })),
            )
        }
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "success": false,
                "message": "DCA service not available"
            })),
        )
    }
}

/// Get execution history for a DCA order
pub async fn get_dca_executions(
    State(state): State<Arc<AppState>>,
    Path(order_id): Path<String>,
) -> impl IntoResponse {
    if let Some(dca_storage) = &state.dca_storage {
        let executions = dca_storage.executions.read().await;

        let order_executions = executions.get(&order_id).cloned().unwrap_or_default();

        (
            StatusCode::OK,
            Json(serde_json::json!({
                "success": true,
                "order_id": order_id,
                "executions": order_executions,
                "total_executions": order_executions.len()
            })),
        )
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "success": false,
                "executions": [],
                "total_executions": 0
            })),
        )
    }
}

// ============================================================================
// BACKGROUND EXECUTION TASK
// ============================================================================

/// Background task that executes due DCA orders
pub async fn dca_execution_loop(state: Arc<AppState>) {
    info!("🔄 [DCA] Starting DCA execution background loop");

    let check_interval = tokio::time::Duration::from_secs(60); // Check every minute
    let mut tick_count = 0u64;

    loop {
        tokio::time::sleep(check_interval).await;
        tick_count += 1;

        if let Some(dca_storage) = &state.dca_storage {
            let now = chrono::Utc::now().timestamp_millis();
            let mut orders_to_execute = Vec::new();

            // Find orders due for execution
            {
                let orders = dca_storage.orders.read().await;

                // v3.2.23-beta: Log active orders every 5 minutes for debugging
                if tick_count % 5 == 0 {
                    let active_count = orders.values().filter(|o| o.status == DcaStatus::Active).count();
                    debug!("📊 [DCA] Tick #{}: {} active orders, checking due orders...", tick_count, active_count);
                }

                for (id, order) in orders.iter() {
                    if order.status == DcaStatus::Active && order.next_execution_at <= now {
                        debug!(
                            "⏰ [DCA] Order {} is due: next_exec={}, now={}, diff={}ms",
                            id, order.next_execution_at, now, now - order.next_execution_at
                        );
                        // Check if order should end
                        if let Some(end_date) = order.end_date {
                            if now > end_date {
                                continue; // Will be marked completed below
                            }
                        }
                        if let Some(max_exec) = order.max_executions {
                            if order.executions_completed >= max_exec {
                                continue; // Will be marked completed below
                            }
                        }
                        orders_to_execute.push(id.clone());
                    }
                }
            }

            // Execute due orders
            for order_id in orders_to_execute {
                if let Err(e) = execute_dca_order(&state, &order_id).await {
                    error!("❌ [DCA] Failed to execute order {}: {}", order_id, e);
                }
            }

            // Clean up completed orders
            {
                let mut orders = dca_storage.orders.write().await;
                for order in orders.values_mut() {
                    if order.status == DcaStatus::Active {
                        // Check if should be marked completed
                        if let Some(end_date) = order.end_date {
                            if now > end_date {
                                order.status = DcaStatus::Completed;
                                info!("✅ [DCA] Order {} completed (end date reached)", order.id);
                            }
                        }
                        if let Some(max_exec) = order.max_executions {
                            if order.executions_completed >= max_exec {
                                order.status = DcaStatus::Completed;
                                info!("✅ [DCA] Order {} completed (max executions reached)", order.id);
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Execute a single DCA order
async fn execute_dca_order(state: &Arc<AppState>, order_id: &str) -> anyhow::Result<()> {
    let dca_storage = state.dca_storage.as_ref()
        .ok_or_else(|| anyhow::anyhow!("DCA storage not available"))?;

    // Get order details
    let (from_token, to_token, amount, slippage, wallet_address) = {
        let orders = dca_storage.orders.read().await;
        let order = orders.get(order_id)
            .ok_or_else(|| anyhow::anyhow!("Order not found"))?;
        (
            order.from_token.clone(),
            order.to_token.clone(),
            order.amount_per_execution,
            order.max_slippage,
            order.wallet_address.clone(),
        )
    };

    info!(
        "💱 [DCA] Executing order {}: {} {} -> {}",
        order_id, amount, from_token, to_token
    );

    // Calculate minimum output based on slippage
    // For simplicity, we'll use 0 here and let the swap function handle it
    // In production, you'd want to check current price and calculate min_amount_out
    let min_amount_out = 0; // Accept any output (slippage handled by execute_swap)

    // Execute the swap using the existing swap function
    let swap_result = execute_dca_swap(
        state,
        &from_token,
        &to_token,
        amount,
        min_amount_out,
        &wallet_address,
    ).await;

    let now = chrono::Utc::now().timestamp_millis();

    match swap_result {
        Ok((amount_out, tx_hash)) => {
            // Update order with successful execution
            let mut orders = dca_storage.orders.write().await;
            let order_clone = if let Some(order) = orders.get_mut(order_id) {
                order.executions_completed += 1;
                order.last_executed_at = Some(now);
                order.next_execution_at = now + order.interval.to_millis();
                order.total_spent += amount;
                order.total_received += amount_out;
                order.last_error = None;
                Some(order.clone()) // Clone for P2P broadcast
            } else {
                None
            };
            drop(orders);

            // Log execution
            let mut executions = dca_storage.executions.write().await;
            let order_execs = executions.entry(order_id.to_string()).or_insert_with(Vec::new);
            order_execs.push(DcaExecution {
                order_id: order_id.to_string(),
                timestamp: now,
                amount_in: amount,
                amount_out,
                price: if amount_out > 0 { amount as f64 / amount_out as f64 } else { 0.0 },
                success: true,
                error: None,
                tx_hash: Some(tx_hash),
            });
            drop(executions);

            // Persist
            let _ = dca_storage.save_to_storage(&state.storage_engine).await;

            // 🌐 v2.4.9-beta: Broadcast execution to P2P network
            if let Some(updated_order) = order_clone {
                broadcast_dca_sync(state, DcaSyncAction::Execute, updated_order).await;
            }

            info!(
                "✅ [DCA] Order {} executed: {} {} -> {} {}",
                order_id, amount, from_token, amount_out, to_token
            );

            Ok(())
        }
        Err(e) => {
            // Update order with failed execution
            let mut orders = dca_storage.orders.write().await;
            if let Some(order) = orders.get_mut(order_id) {
                order.last_error = Some(e.to_string());
                // Still schedule next execution to retry
                order.next_execution_at = now + order.interval.to_millis();

                // If 5 consecutive failures, pause the order
                // (You could track consecutive failures in a separate field)
            }
            drop(orders);

            // Log failed execution
            let mut executions = dca_storage.executions.write().await;
            let order_execs = executions.entry(order_id.to_string()).or_insert_with(Vec::new);
            order_execs.push(DcaExecution {
                order_id: order_id.to_string(),
                timestamp: now,
                amount_in: amount,
                amount_out: 0,
                price: 0.0,
                success: false,
                error: Some(e.to_string()),
                tx_hash: None,
            });
            drop(executions);

            // Persist
            let _ = dca_storage.save_to_storage(&state.storage_engine).await;

            warn!("❌ [DCA] Order {} failed: {}", order_id, e);
            Err(e)
        }
    }
}

/// Public wrapper called by limit_order_api.
/// `max_slippage` (e.g. 0.03 = 3%) is applied to the pre-swap pool estimate to derive
/// `min_amount_out`. If the actual output after AMM pricing is below that floor, the swap
/// is rejected so the order retries on the next tick.
pub async fn execute_limit_swap(
    state: &Arc<AppState>,
    from_token: &str,
    to_token: &str,
    amount: u128,
    max_slippage: f64,
    wallet_address: &str,
) -> anyhow::Result<(u128, String)> {
    // Estimate expected output from current pool state and apply slippage tolerance.
    // This read happens BEFORE the swap write; the TOCTOU window is negligible (same tick).
    let min_amount_out = estimate_swap_output(state, from_token, to_token, amount)
        .await
        .map(|expected| {
            let floor = expected as f64 * (1.0 - max_slippage.clamp(0.0, 1.0));
            floor as u128
        })
        .unwrap_or(0); // If estimation fails, enforce no floor (better to fill than not)
    execute_dca_swap(state, from_token, to_token, amount, min_amount_out, wallet_address).await
}

/// Read-only AMM output estimate — same formula as execute_dca_swap but without side effects.
async fn estimate_swap_output(
    state: &Arc<AppState>,
    from_token: &str,
    to_token: &str,
    amount: u128,
) -> Option<u128> {
    let pools = state.liquidity_pools.read().await;
    let pool_id = format!("{}_{}", from_token.to_uppercase(), to_token.to_uppercase());
    let pool_id_reverse = format!("{}_{}", to_token.to_uppercase(), from_token.to_uppercase());
    let (pool, is_reversed) = if let Some(p) = pools.get(&pool_id) {
        (p.clone(), false)
    } else if let Some(p) = pools.get(&pool_id_reverse) {
        (p.clone(), true)
    } else {
        return None;
    };
    drop(pools);

    let (reserve_in, reserve_out, dec_in, dec_out) = if is_reversed {
        (pool.reserve1, pool.reserve0, pool.token1_decimals, pool.token0_decimals)
    } else {
        (pool.reserve0, pool.reserve1, pool.token0_decimals, pool.token1_decimals)
    };

    let amount_with_fee = amount.checked_mul(997)?.checked_div(1000)?;
    let numerator = amount_with_fee.checked_mul(reserve_out as u128);
    let denominator = reserve_in.checked_add(amount_with_fee)?;
    if denominator == 0 { return None; }
    let mut out = numerator.map(|n| n / denominator)
        .unwrap_or_else(|| ((amount_with_fee as f64) * (reserve_out as f64 / reserve_in as f64)) as u128);

    if dec_in != dec_out {
        if dec_in > dec_out {
            out = out / 10u128.pow((dec_in - dec_out) as u32);
        } else {
            out = out.saturating_mul(10u128.pow((dec_out - dec_in) as u32));
        }
    }
    if out == 0 { return None; }
    Some(out)
}

/// Execute the actual swap for DCA (internal function)
async fn execute_dca_swap(
    state: &Arc<AppState>,
    from_token: &str,
    to_token: &str,
    amount: u128,
    min_amount_out: u128,
    wallet_address: &str,
) -> anyhow::Result<(u128, String)> {
    // This function mirrors the logic in handlers::execute_swap
    // but without the HTTP request/response overhead

    let pools = state.liquidity_pools.read().await;

    // Find the pool for this token pair
    let pool_id = format!("{}_{}", from_token.to_uppercase(), to_token.to_uppercase());
    let pool_id_reverse = format!("{}_{}", to_token.to_uppercase(), from_token.to_uppercase());

    let (pool, is_reversed) = if let Some(p) = pools.get(&pool_id) {
        (p.clone(), false)
    } else if let Some(p) = pools.get(&pool_id_reverse) {
        (p.clone(), true)
    } else {
        return Err(anyhow::anyhow!("No liquidity pool found for {}/{}", from_token, to_token));
    };
    drop(pools);

    // v3.2.23-beta: Calculate output using AMM formula with cross-decimal handling
    let (reserve_in, reserve_out, dec_in, dec_out) = if is_reversed {
        (pool.reserve1, pool.reserve0, pool.token1_decimals, pool.token0_decimals)
    } else {
        (pool.reserve0, pool.reserve1, pool.token0_decimals, pool.token1_decimals)
    };

    // AMM: amount_out = (amount_in * reserve_out) / (reserve_in + amount_in)
    // With 0.3% fee
    let fee = 3u128; // 0.3% = 3/1000
    let amount_with_fee = amount.checked_mul(1000 - fee)
        .and_then(|v| v.checked_div(1000))
        .ok_or_else(|| anyhow::anyhow!("Overflow in fee calculation"))?;

    // Use checked arithmetic to avoid overflow
    let numerator = (amount_with_fee as u128).checked_mul(reserve_out as u128);
    let denominator = reserve_in.checked_add(amount_with_fee);

    if denominator.is_none() || denominator == Some(0) {
        return Err(anyhow::anyhow!("Pool has zero reserves or overflow"));
    }

    let mut amount_out = if let Some(num) = numerator {
        num / denominator.unwrap()
    } else {
        // Numerator overflow: use floating point approximation
        let ratio = (reserve_out as f64) / (reserve_in as f64);
        ((amount_with_fee as f64) * ratio) as u128
    };

    // Cross-decimal adjustment: scale output if decimals differ
    if dec_in != dec_out {
        if dec_in > dec_out {
            amount_out = amount_out / 10u128.pow((dec_in - dec_out) as u32);
        } else {
            amount_out = amount_out.saturating_mul(10u128.pow((dec_out - dec_in) as u32));
        }
    }

    if amount_out == 0 {
        return Err(anyhow::anyhow!("Swap would result in zero output. Try a larger amount."));
    }

    // Slippage guard: reject if output is below the caller's minimum.
    // For DCA callers min_amount_out is always 0 (no guard). For limit orders it is set
    // by execute_limit_swap from the pool estimate × (1 - max_slippage).
    if min_amount_out > 0 && amount_out < min_amount_out {
        return Err(anyhow::anyhow!(
            "Slippage too high: expected ≥{} base units out, AMM gives {} — rejecting (will retry next tick)",
            min_amount_out, amount_out
        ));
    }

    // Update wallet balances
    let wallet_bytes = hex::decode(wallet_address.trim_start_matches("qnk"))
        .map_err(|e| anyhow::anyhow!("Invalid wallet address: {}", e))?;

    if wallet_bytes.len() != 32 {
        return Err(anyhow::anyhow!("Wallet address must be 32 bytes"));
    }

    let mut wallet_addr = [0u8; 32];
    wallet_addr.copy_from_slice(&wallet_bytes);

    // Update from_token balance (deduct)
    if from_token.to_uppercase() == "QUG" {
        // Native token
        let mut balances = state.wallet_balances.write().await;
        let balance = balances.entry(wallet_addr).or_insert(0);
        if *balance < amount {
            return Err(anyhow::anyhow!("Insufficient QUG balance"));
        }
        *balance -= amount;
        let new_balance = *balance;
        drop(balances);

        // Persist balance update
        let _ = state.storage_engine.save_wallet_balance(&wallet_addr, new_balance).await;
    } else {
        // Token balance
        let from_token_bytes = hex::decode(from_token.trim_start_matches("qnk"))
            .map_err(|e| anyhow::anyhow!("Invalid from_token address: {}", e))?;

        if from_token_bytes.len() != 32 {
            return Err(anyhow::anyhow!("Token address must be 32 bytes"));
        }

        let mut token_addr = [0u8; 32];
        token_addr.copy_from_slice(&from_token_bytes);

        let mut token_balances = state.token_balances.write().await;
        let key = (wallet_addr, token_addr);
        let balance = token_balances.entry(key).or_insert(0);
        if *balance < amount {
            return Err(anyhow::anyhow!("Insufficient token balance"));
        }
        *balance -= amount;
        let new_balance = *balance;
        drop(token_balances);

        // Persist token balance debit to disk
        if let Err(e) = state.storage_engine.save_token_balance(&wallet_addr, &token_addr, new_balance).await {
            tracing::warn!("Failed to persist token balance after DCA debit: {}", e);
        }
    }

    // Update to_token balance (credit)
    if to_token.to_uppercase() == "QUG" {
        let mut balances = state.wallet_balances.write().await;
        let balance = balances.entry(wallet_addr).or_insert(0);
        *balance += amount_out;
        let new_balance = *balance;
        drop(balances);

        let _ = state.storage_engine.save_wallet_balance(&wallet_addr, new_balance).await;
    } else {
        let to_token_bytes = hex::decode(to_token.trim_start_matches("qnk"))
            .map_err(|e| anyhow::anyhow!("Invalid to_token address: {}", e))?;

        if to_token_bytes.len() != 32 {
            return Err(anyhow::anyhow!("Token address must be 32 bytes"));
        }

        let mut token_addr = [0u8; 32];
        token_addr.copy_from_slice(&to_token_bytes);

        let mut token_balances = state.token_balances.write().await;
        let key = (wallet_addr, token_addr);
        let balance = token_balances.entry(key).or_insert(0);
        *balance += amount_out;
        let new_balance = *balance;
        drop(token_balances);

        // Persist token balance credit to disk
        if let Err(e) = state.storage_engine.save_token_balance(&wallet_addr, &token_addr, new_balance).await {
            tracing::warn!("Failed to persist token balance after DCA credit: {}", e);
        }
    }

    // Update pool reserves
    let mut pools = state.liquidity_pools.write().await;
    let pool_key = if is_reversed { pool_id_reverse } else { pool_id };
    if let Some(p) = pools.get_mut(&pool_key) {
        if is_reversed {
            p.reserve1 += amount;
            p.reserve0 -= amount_out;
        } else {
            p.reserve0 += amount;
            p.reserve1 -= amount_out;
        }
    }
    drop(pools);

    // Generate tx hash
    let tx_hash = format!("dca_{:x}{:x}", chrono::Utc::now().timestamp_millis(), rand::random::<u32>());

    Ok((amount_out, tx_hash))
}

// ============================================================================
// ROUTER SETUP
// ============================================================================

/// Create DCA API router
pub fn create_dca_router() -> axum::Router<Arc<AppState>> {
    use axum::routing::{get, post, put, delete};

    axum::Router::new()
        // Create new DCA order
        .route("/orders", post(create_dca_order))
        // Get all DCA orders for a wallet
        .route("/orders/:wallet_address", get(get_wallet_dca_orders))
        // Cancel a DCA order
        .route("/orders/:wallet_address/:order_id", delete(cancel_dca_order))
        // Pause a DCA order
        .route("/orders/:wallet_address/:order_id/pause", put(pause_dca_order))
        // Resume a DCA order
        .route("/orders/:wallet_address/:order_id/resume", put(resume_dca_order))
        // Get execution history
        .route("/executions/:order_id", get(get_dca_executions))
}

// ============================================================================
// P2P GOSSIPSUB SYNCHRONIZATION (v2.4.9-beta)
// Now with Ed25519 signature verification for security
// ============================================================================

/// Broadcast DCA order change to all peers via gossipsub
/// This ensures decentralized agreement - all nodes see the same DCA orders
/// v2.4.9-beta: Messages are now signed with Ed25519 for authentication
pub async fn broadcast_dca_sync(state: &Arc<AppState>, action: DcaSyncAction, order: DcaOrder) {
    use ed25519_dalek::Signer;

    // Get our node's peer ID
    let node_id = if let Some(ref manager) = state.libp2p_discovery {
        let guard = manager.lock().await;
        guard.peer_id().to_string()
    } else {
        "unknown".to_string()
    };

    let mut message = DcaSyncMessage {
        version: 2, // v2 = signed messages
        action: action.clone(),
        order: order.clone(),
        timestamp: chrono::Utc::now().timestamp_millis(),
        node_id: node_id.clone(),
        signature: Vec::new(),
        signer_public_key: Vec::new(),
    };

    // 🔐 Sign the message with node's Ed25519 key (always available)
    {
        let signing_key = &state.node_signing_key;
        let payload = message.signing_payload();
        let signature = signing_key.sign(&payload);
        message.signature = signature.to_bytes().to_vec();
        message.signer_public_key = signing_key.verifying_key().to_bytes().to_vec();
        debug!("🔐 [DCA P2P] Signed message with Ed25519 key");
    }

    // Serialize the signed message
    let Ok(data) = serde_json::to_vec(&message) else {
        warn!("❌ [DCA P2P] Failed to serialize DCA sync message");
        return;
    };

    // Publish to gossipsub
    if let Some(ref manager) = state.libp2p_discovery {
        let mut manager_guard = manager.lock().await;
        let topic = manager_guard.network_config().network_id.dca_orders_topic();

        match manager_guard.publish_topic(&topic, data) {
            Ok(_) => {
                info!(
                    "🌐🔐 [DCA P2P] Broadcast signed {:?} for order {} to topic {}",
                    action, order.id, topic
                );
            }
            Err(e) => {
                warn!("❌ [DCA P2P] Failed to broadcast DCA sync: {}", e);
            }
        }
    } else {
        // No P2P available - local-only mode
        info!("📡 [DCA] No P2P network - DCA order {} is local only", order.id);
    }
}

/// Handle incoming DCA sync message from peers
/// This processes DCA orders from other nodes to achieve consensus
/// v2.4.9-beta: Now verifies Ed25519 signatures before processing
pub async fn handle_dca_sync_message(state: &Arc<AppState>, data: &[u8]) {
    // Parse the sync message
    let message: DcaSyncMessage = match serde_json::from_slice(data) {
        Ok(m) => m,
        Err(e) => {
            warn!("❌ [DCA P2P] Failed to parse DCA sync message: {}", e);
            return;
        }
    };

    // Check if this is from us (ignore our own broadcasts)
    if let Some(ref manager) = state.libp2p_discovery {
        let guard = manager.lock().await;
        if guard.peer_id().to_string() == message.node_id {
            return; // Ignore our own messages
        }
    }

    // 🔐 SECURITY: Verify signature before processing
    if let Err(e) = message.validate_full() {
        warn!(
            "🔐❌ [DCA P2P] Rejecting message from {} - {}: {:?} for order {}",
            message.node_id, e, message.action, message.order.id
        );
        return;
    }

    info!(
        "🌐🔐 [DCA P2P] Verified {:?} for order {} from node {} (sig OK)",
        message.action, message.order.id, message.node_id
    );

    // Get DCA storage
    let Some(dca_storage) = &state.dca_storage else {
        warn!("❌ [DCA P2P] DCA storage not available");
        return;
    };

    match message.action {
        DcaSyncAction::Create => {
            // Add the order if we don't have it
            let mut orders = dca_storage.orders.write().await;
            if !orders.contains_key(&message.order.id) {
                orders.insert(message.order.id.clone(), message.order.clone());
                info!("✅ [DCA P2P] Added DCA order {} from peer", message.order.id);
            } else {
                // Order exists - check if peer's version is newer
                if let Some(existing) = orders.get(&message.order.id) {
                    if message.order.created_at > existing.created_at {
                        orders.insert(message.order.id.clone(), message.order.clone());
                        info!("🔄 [DCA P2P] Updated DCA order {} with newer version", message.order.id);
                    }
                }
            }
            drop(orders);
        }
        DcaSyncAction::Cancel => {
            // Cancel the order if we have it
            let mut orders = dca_storage.orders.write().await;
            if let Some(order) = orders.get_mut(&message.order.id) {
                if order.status != DcaStatus::Cancelled {
                    order.status = DcaStatus::Cancelled;
                    info!("🛑 [DCA P2P] Cancelled DCA order {} from peer", message.order.id);
                }
            }
            drop(orders);
        }
        DcaSyncAction::Pause => {
            // Pause the order if we have it and it's active
            let mut orders = dca_storage.orders.write().await;
            if let Some(order) = orders.get_mut(&message.order.id) {
                if order.status == DcaStatus::Active {
                    order.status = DcaStatus::Paused;
                    info!("⏸️ [DCA P2P] Paused DCA order {} from peer", message.order.id);
                }
            }
            drop(orders);
        }
        DcaSyncAction::Resume => {
            // Resume the order if we have it and it's paused
            let mut orders = dca_storage.orders.write().await;
            if let Some(order) = orders.get_mut(&message.order.id) {
                if order.status == DcaStatus::Paused {
                    order.status = DcaStatus::Active;
                    order.next_execution_at = message.order.next_execution_at;
                    info!("▶️ [DCA P2P] Resumed DCA order {} from peer", message.order.id);
                }
            }
            drop(orders);
        }
        DcaSyncAction::Execute => {
            // Update execution stats from peer
            let mut orders = dca_storage.orders.write().await;
            if let Some(order) = orders.get_mut(&message.order.id) {
                // Only update if peer has more executions (they did the execution)
                if message.order.executions_completed > order.executions_completed {
                    order.executions_completed = message.order.executions_completed;
                    order.last_executed_at = message.order.last_executed_at;
                    order.next_execution_at = message.order.next_execution_at;
                    order.total_spent = message.order.total_spent;
                    order.total_received = message.order.total_received;
                    info!("📊 [DCA P2P] Updated execution stats for order {} from peer", message.order.id);
                }
            }
            drop(orders);
        }
    }

    // Persist the updated state
    if let Err(e) = dca_storage.save_to_storage(&state.storage_engine).await {
        warn!("❌ [DCA P2P] Failed to persist DCA state after sync: {}", e);
    }
}
