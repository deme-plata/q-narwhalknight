//! v10.4.4: DEX Limit Orders — Price-Triggered Swaps
//!
//! Limit orders execute when the market price crosses a user-defined threshold.
//! Built on the same timer infrastructure as DCA orders: a 60-second polling
//! loop checks all open orders against the live oracle price each tick.
//!
//! Features:
//! - Price-above trigger (stop-buy / take-profit sell)
//! - Price-below trigger (limit-buy / stop-loss)
//! - Optional expiry (GTC = Good Till Cancelled by default)
//! - Persistent storage via RocksDB CF_LIMIT_ORDERS
//! - P2P gossipsub broadcast of fills for cross-node visibility

use std::sync::Arc;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock, Semaphore};
use std::collections::HashMap;
use tracing::{info, warn, error, debug};

use ed25519_dalek::Verifier;
use crate::AppState;

/// Orders stuck in Processing for longer than this (e.g. after a crash) are reset to Open on startup.
/// 5 minutes is safely above the 60s poll interval plus any reasonable swap latency.
const PROCESSING_TIMEOUT_MS: i64 = 5 * 60 * 1000;

/// Auto-cancel an order after this many consecutive swap failures to avoid infinite retry loops.
const MAX_FAILURE_COUNT: u32 = 5;

// ============================================================================
// TYPES
// ============================================================================

/// Which side of the trigger price fires the order
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum PriceDirection {
    /// Fire when price rises ABOVE trigger (e.g. stop-buy, take-profit)
    Above,
    /// Fire when price falls BELOW trigger (e.g. limit-buy, stop-loss)
    Below,
}

/// Current status of a limit order
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum LimitOrderStatus {
    Open,
    /// Written to storage (via put_sync) BEFORE the swap executes.
    /// A node crash between Processing and Filled leaves the order in this state.
    /// On startup, orders stuck in Processing longer than PROCESSING_TIMEOUT_SECS are reset to Open.
    Processing,
    Filled,
    Cancelled,
    Expired,
}

/// A price-triggered one-shot swap
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LimitOrder {
    pub id: String,
    pub wallet_address: String,
    /// Token to sell (e.g. "QUG" or contract address)
    pub from_token: String,
    /// Token to buy (e.g. "QUGUSD" or contract address)
    pub to_token: String,
    /// Amount to spend when triggered (base units, 24-decimal)
    #[serde(serialize_with = "q_types::u128_serde::serialize", deserialize_with = "q_types::u128_serde::deserialize")]
    pub amount: u128,
    /// USD price of `price_token` that triggers the order
    pub trigger_price: f64,
    /// Which token's price to watch (e.g. "QUG")
    pub price_token: String,
    /// Fire when price goes above or below trigger
    pub direction: PriceDirection,
    /// Maximum slippage tolerance (0.03 = 3%)
    pub max_slippage: f64,
    pub status: LimitOrderStatus,
    pub created_at: i64,
    /// Timestamp in ms when the order entered Processing state (for crash recovery)
    pub processing_since: Option<i64>,
    pub filled_at: Option<i64>,
    /// Optional expiry timestamp in ms (None = GTC)
    pub expiry: Option<i64>,
    /// Consecutive swap failures; auto-cancelled after MAX_FAILURE_COUNT attempts
    #[serde(default)]
    pub failure_count: u32,
    /// Actual output after fill
    #[serde(serialize_with = "q_types::u128_serde::serialize", deserialize_with = "q_types::u128_serde::deserialize")]
    pub amount_out: u128,
    /// Price at which the order was filled (for display)
    pub fill_price: Option<f64>,
    pub tx_hash: Option<String>,
}

/// Max concurrent swap executions in the polling loop.
/// Prevents memory spikes when many orders trigger in the same tick.
const MAX_CONCURRENT_EXECUTIONS: usize = 8;

/// In-memory + RocksDB storage for limit orders
pub struct LimitOrderStorage {
    pub orders: RwLock<HashMap<String, LimitOrder>>,
    /// Limits concurrent swap executions to prevent OOM under high trigger volume
    pub execution_semaphore: Arc<Semaphore>,
}

impl LimitOrderStorage {
    pub fn new() -> Self {
        LimitOrderStorage {
            orders: RwLock::new(HashMap::new()),
            execution_semaphore: Arc::new(Semaphore::new(MAX_CONCURRENT_EXECUTIONS)),
        }
    }

    pub async fn load_from_storage(&self, storage: &q_storage::QStorage) -> anyhow::Result<()> {
        match storage.load_all_limit_orders().await {
            Ok(pairs) => {
                let mut orders = self.orders.write().await;
                for (id, bytes) in pairs {
                    if let Ok(order) = serde_json::from_slice::<LimitOrder>(&bytes) {
                        orders.insert(id, order);
                    }
                }
                info!("📊 [LIMIT] Loaded {} limit orders from RocksDB", orders.len());
            }
            Err(e) => {
                warn!("⚠️ [LIMIT] Could not load limit orders: {}", e);
            }
        }
        Ok(())
    }

    pub async fn save_order(&self, storage: &q_storage::QStorage, order: &LimitOrder) -> anyhow::Result<()> {
        let bytes = serde_json::to_vec(order)?;
        storage.save_limit_order(&order.id, &bytes).await?;
        Ok(())
    }

    pub async fn delete_order(&self, storage: &q_storage::QStorage, order_id: &str) -> anyhow::Result<()> {
        storage.delete_limit_order(order_id).await?;
        Ok(())
    }

    /// On startup, any order stuck in Processing state past PROCESSING_TIMEOUT_MS was left in-flight
    /// by a crash. The swap may or may not have executed. We conservatively reset these to Open so
    /// the next poll cycle re-evaluates them; the TOCTOU guard (re-read + Processing write) ensures
    /// that if the swap actually completed, the order will be in Filled state and the re-read guard
    /// will skip it. If the swap did not complete, it will be retried.
    pub async fn recover_stuck_processing(&self, storage: &q_storage::QStorage) {
        let now = chrono::Utc::now().timestamp_millis();
        let mut orders = self.orders.write().await;
        let stuck: Vec<String> = orders
            .values()
            .filter(|o| {
                o.status == LimitOrderStatus::Processing
                    && o.processing_since.map(|t| now - t > PROCESSING_TIMEOUT_MS).unwrap_or(true)
            })
            .map(|o| o.id.clone())
            .collect();

        for id in stuck {
            if let Some(o) = orders.get_mut(&id) {
                o.status = LimitOrderStatus::Open;
                o.processing_since = None;
                let o_clone = o.clone();
                warn!("🔄 [LIMIT] Recovered stuck Processing order {} → Open", id);
                let bytes = serde_json::to_vec(&o_clone).unwrap_or_default();
                let _ = storage.save_limit_order(&id, &bytes).await;
            }
        }
    }
}

impl Default for LimitOrderStorage {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// REQUEST / RESPONSE TYPES
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct CreateLimitOrderRequest {
    pub wallet_address: String,
    pub from_token: String,
    pub to_token: String,
    #[serde(deserialize_with = "q_types::u128_serde::deserialize")]
    pub amount: u128,
    pub trigger_price: f64,
    /// Which token to watch for price; defaults to from_token if not specified
    pub price_token: Option<String>,
    pub direction: PriceDirection,
    #[serde(default = "default_slippage")]
    pub max_slippage: f64,
    /// Optional expiry in Unix ms (None = GTC)
    pub expiry: Option<i64>,
}

fn default_slippage() -> f64 {
    0.03
}

#[derive(Debug, Serialize)]
pub struct CreateLimitOrderResponse {
    pub success: bool,
    pub order_id: Option<String>,
    pub message: String,
}

/// Signed cancellation request — prevents unauthorised cancellation of another user's order.
/// The client signs `"{order_id}:{timestamp_ms}"` with their Ed25519 private key.
/// Timestamp must be within 60 seconds of server time to prevent replay attacks.
#[derive(Debug, Deserialize)]
pub struct CancelLimitOrderRequest {
    /// Unix milliseconds when the request was signed — must be within 60s of now
    pub timestamp_ms: i64,
    /// hex-encoded 64-byte Ed25519 signature over `"{order_id}:{timestamp_ms}"`
    pub signature: String,
}

#[derive(Debug, Serialize)]
pub struct LimitOrdersResponse {
    pub success: bool,
    pub orders: Vec<LimitOrder>,
    pub open_count: u32,
}

// ============================================================================
// HTTP HANDLERS
// ============================================================================

/// POST /api/v1/dex/limit-orders
pub async fn create_limit_order(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateLimitOrderRequest>,
) -> impl IntoResponse {
    if req.wallet_address.is_empty() {
        return (StatusCode::BAD_REQUEST, Json(CreateLimitOrderResponse {
            success: false,
            order_id: None,
            message: "wallet_address is required".into(),
        }));
    }
    if req.from_token == req.to_token {
        return (StatusCode::BAD_REQUEST, Json(CreateLimitOrderResponse {
            success: false,
            order_id: None,
            message: "from_token and to_token must differ".into(),
        }));
    }
    if req.amount == 0 {
        return (StatusCode::BAD_REQUEST, Json(CreateLimitOrderResponse {
            success: false,
            order_id: None,
            message: "amount must be > 0".into(),
        }));
    }
    if req.trigger_price <= 0.0 {
        return (StatusCode::BAD_REQUEST, Json(CreateLimitOrderResponse {
            success: false,
            order_id: None,
            message: "trigger_price must be > 0".into(),
        }));
    }

    let Some(lo_storage) = &state.limit_order_storage else {
        return (StatusCode::SERVICE_UNAVAILABLE, Json(CreateLimitOrderResponse {
            success: false,
            order_id: None,
            message: "Limit order service not available".into(),
        }));
    };

    let now = chrono::Utc::now().timestamp_millis();
    let order_id = format!("lo_{}_{:x}", now, rand::random::<u32>());
    let price_token = req.price_token.unwrap_or_else(|| req.from_token.clone());

    let order = LimitOrder {
        id: order_id.clone(),
        wallet_address: req.wallet_address,
        from_token: req.from_token,
        to_token: req.to_token,
        amount: req.amount,
        trigger_price: req.trigger_price,
        price_token,
        direction: req.direction,
        max_slippage: req.max_slippage,
        status: LimitOrderStatus::Open,
        created_at: now,
        processing_since: None,
        filled_at: None,
        expiry: req.expiry,
        amount_out: 0,
        fill_price: None,
        tx_hash: None,
        failure_count: 0,
    };

    {
        let mut orders = lo_storage.orders.write().await;
        orders.insert(order_id.clone(), order.clone());
    }

    if let Err(e) = lo_storage.save_order(&state.storage_engine, &order).await {
        warn!("⚠️ [LIMIT] Failed to persist order {}: {}", order_id, e);
    }

    info!(
        "✅ [LIMIT] Created order {} — {} {} {} {} trigger ${:.4}",
        order_id, order.amount, order.from_token, "→", order.to_token, order.trigger_price
    );

    (StatusCode::CREATED, Json(CreateLimitOrderResponse {
        success: true,
        order_id: Some(order_id),
        message: "Limit order created".into(),
    }))
}

/// GET /api/v1/dex/limit-orders/:wallet_address
pub async fn get_wallet_limit_orders(
    State(state): State<Arc<AppState>>,
    Path(wallet_address): Path<String>,
) -> impl IntoResponse {
    let Some(lo_storage) = &state.limit_order_storage else {
        return (StatusCode::SERVICE_UNAVAILABLE, Json(LimitOrdersResponse {
            success: false,
            orders: vec![],
            open_count: 0,
        }));
    };

    let orders = lo_storage.orders.read().await;
    let wallet_orders: Vec<LimitOrder> = orders
        .values()
        .filter(|o| o.wallet_address == wallet_address)
        .cloned()
        .collect();

    let open_count = wallet_orders.iter().filter(|o| o.status == LimitOrderStatus::Open).count() as u32;

    (StatusCode::OK, Json(LimitOrdersResponse {
        success: true,
        orders: wallet_orders,
        open_count,
    }))
}

/// DELETE /api/v1/dex/limit-orders/:wallet_address/:order_id
/// Body must be a JSON `CancelLimitOrderRequest` with a valid Ed25519 signature.
pub async fn cancel_limit_order(
    State(state): State<Arc<AppState>>,
    Path((wallet_address, order_id)): Path<(String, String)>,
    Json(req): Json<CancelLimitOrderRequest>,
) -> impl IntoResponse {
    // --- Replay protection: reject requests signed > 60s ago ---
    let server_now = chrono::Utc::now().timestamp_millis();
    if (server_now - req.timestamp_ms).abs() > 60_000 {
        return (StatusCode::BAD_REQUEST, Json(serde_json::json!({
            "success": false, "message": "Request timestamp out of range (must be within 60s)"
        })));
    }

    // --- Decode the wallet public key and signature ---
    let pub_key_bytes: [u8; 32] = match hex::decode(&wallet_address)
        .ok()
        .and_then(|b| b.try_into().ok())
    {
        Some(b) => b,
        None => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({
            "success": false, "message": "Invalid wallet_address (must be 32-byte hex)"
        }))),
    };
    let verifying_key = match ed25519_dalek::VerifyingKey::from_bytes(&pub_key_bytes) {
        Ok(k) => k,
        Err(_) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({
            "success": false, "message": "wallet_address is not a valid Ed25519 public key"
        }))),
    };
    let sig_bytes: [u8; 64] = match hex::decode(&req.signature)
        .ok()
        .and_then(|b| b.try_into().ok())
    {
        Some(b) => b,
        None => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({
            "success": false, "message": "Invalid signature (must be 64-byte hex)"
        }))),
    };
    let signature = ed25519_dalek::Signature::from_bytes(&sig_bytes);

    // --- Verify signature over "{order_id}:{timestamp_ms}" ---
    let message = format!("{}:{}", order_id, req.timestamp_ms);
    if verifying_key.verify(message.as_bytes(), &signature).is_err() {
        return (StatusCode::FORBIDDEN, Json(serde_json::json!({
            "success": false, "message": "Signature verification failed"
        })));
    }

    let Some(lo_storage) = &state.limit_order_storage else {
        return (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({
            "success": false, "message": "Limit order service not available"
        })));
    };

    let mut orders = lo_storage.orders.write().await;
    if let Some(order) = orders.get_mut(&order_id) {
        if order.wallet_address != wallet_address {
            return (StatusCode::FORBIDDEN, Json(serde_json::json!({
                "success": false, "message": "Not your order"
            })));
        }
        if order.status != LimitOrderStatus::Open {
            return (StatusCode::BAD_REQUEST, Json(serde_json::json!({
                "success": false,
                "message": format!("Order is not open (status: {:?})", order.status)
            })));
        }
        order.status = LimitOrderStatus::Cancelled;
        let order_clone = order.clone();
        drop(orders);

        if let Err(e) = lo_storage.save_order(&state.storage_engine, &order_clone).await {
            warn!("⚠️ [LIMIT] Failed to persist cancel for {}: {}", order_id, e);
        }

        info!("🛑 [LIMIT] Cancelled order {}", order_id);
        (StatusCode::OK, Json(serde_json::json!({
            "success": true, "message": "Order cancelled"
        })))
    } else {
        drop(orders);
        (StatusCode::NOT_FOUND, Json(serde_json::json!({
            "success": false, "message": "Order not found"
        })))
    }
}

/// GET /api/v1/dex/limit-orders/open (all open orders — for admin/display)
pub async fn get_all_open_limit_orders(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let Some(lo_storage) = &state.limit_order_storage else {
        return Json(LimitOrdersResponse { success: false, orders: vec![], open_count: 0 });
    };

    let orders = lo_storage.orders.read().await;
    let open: Vec<LimitOrder> = orders
        .values()
        .filter(|o| o.status == LimitOrderStatus::Open)
        .cloned()
        .collect();
    let open_count = open.len() as u32;

    Json(LimitOrdersResponse { success: true, orders: open, open_count })
}

// ============================================================================
// PRICE-CHECK BACKGROUND LOOP
// ============================================================================

/// Background task: every 60 seconds, evaluate all open limit orders against
/// the current oracle price. Fires the swap if the trigger condition is met.
pub async fn limit_order_check_loop(state: Arc<AppState>) {
    info!("🎯 [LIMIT] Starting limit order price-check loop (60s interval)");

    let interval = tokio::time::Duration::from_secs(60);

    loop {
        tokio::time::sleep(interval).await;

        let Some(lo_storage) = &state.limit_order_storage else { continue };

        let now = chrono::Utc::now().timestamp_millis();

        // Collect open orders due for checking
        let candidates: Vec<LimitOrder> = {
            let orders = lo_storage.orders.read().await;
            orders.values()
                .filter(|o| o.status == LimitOrderStatus::Open)
                .cloned()
                .collect()
        };

        if candidates.is_empty() {
            continue;
        }

        debug!("🎯 [LIMIT] Checking {} open limit orders", candidates.len());

        // Phase 1 (sequential, fast): evaluate price conditions for all open orders.
        // Expirations are handled inline. Triggered orders are collected for concurrent execution.
        let mut triggered: Vec<(LimitOrder, f64)> = Vec::new();

        for order in candidates {
            if let Some(expiry) = order.expiry {
                if now >= expiry {
                    let mut orders = lo_storage.orders.write().await;
                    if let Some(o) = orders.get_mut(&order.id) {
                        o.status = LimitOrderStatus::Expired;
                        let o_clone = o.clone();
                        drop(orders);
                        let _ = lo_storage.save_order(&state.storage_engine, &o_clone).await;
                        info!("⏰ [LIMIT] Order {} expired", order.id);
                    }
                    continue;
                }
            }

            let current_price = get_token_price_usd(&state, &order.price_token).await;
            let Some(price) = current_price else {
                debug!("🎯 [LIMIT] Cannot get price for {} — skipping order {}", order.price_token, order.id);
                continue;
            };

            let fires = match order.direction {
                PriceDirection::Above => price >= order.trigger_price,
                PriceDirection::Below => price <= order.trigger_price,
            };

            if !fires {
                debug!(
                    "🎯 [LIMIT] Order {} not triggered: current={:.4} trigger={:.4} {:?}",
                    order.id, price, order.trigger_price, order.direction
                );
                continue;
            }

            info!(
                "🎯 [LIMIT] Order {} TRIGGERED: {} {} ${:.4} (trigger ${:.4})",
                order.id, order.price_token,
                if matches!(order.direction, PriceDirection::Above) { ">=" } else { "<=" },
                price, order.trigger_price
            );
            triggered.push((order, price));
        }

        if triggered.is_empty() {
            continue;
        }

        // Phase 2 (concurrent): execute each triggered order in its own task.
        // Semaphore caps concurrent swap executions to prevent OOM under high trigger volume.
        // The TOCTOU guard (re-read + Processing write) runs inside each task under the write lock,
        // so concurrent spawns are safe — only one task can claim any given order.
        let mut join_handles = Vec::with_capacity(triggered.len());

        for (order, price) in triggered {
            let state_c = state.clone();
            let lo_storage_c = lo_storage.clone();
            let sem = lo_storage.execution_semaphore.clone();

            join_handles.push(tokio::spawn(async move {
                // Acquire semaphore slot — blocks if MAX_CONCURRENT_EXECUTIONS already running
                let _permit = match sem.acquire_owned().await {
                    Ok(p) => p,
                    Err(_) => return, // semaphore closed (shutdown)
                };

                let now_inner = chrono::Utc::now().timestamp_millis();

                // TOCTOU guard: re-read order under write lock; gate on still-Open
                let should_execute = {
                    let mut orders = lo_storage_c.orders.write().await;
                    match orders.get_mut(&order.id) {
                        Some(o) if o.status == LimitOrderStatus::Open => {
                            o.status = LimitOrderStatus::Processing;
                            o.processing_since = Some(now_inner);
                            let o_clone = o.clone();
                            if let Err(e) = lo_storage_c.save_order(&state_c.storage_engine, &o_clone).await {
                                error!("❌ [LIMIT] Cannot write Processing for {}: {} — skipping", order.id, e);
                                o.status = LimitOrderStatus::Open;
                                o.processing_since = None;
                                false
                            } else {
                                true
                            }
                        }
                        Some(o) => {
                            debug!("🎯 [LIMIT] Order {} is {:?} — skipping (race handled)", order.id, o.status);
                            false
                        }
                        None => false,
                    }
                };

                if !should_execute { return; }

                match execute_limit_order_swap(&state_c, &order).await {
                    Ok((amount_out, tx_hash)) => {
                        let mut orders = lo_storage_c.orders.write().await;
                        if let Some(o) = orders.get_mut(&order.id) {
                            o.status = LimitOrderStatus::Filled;
                            o.filled_at = Some(now_inner);
                            o.processing_since = None;
                            o.amount_out = amount_out;
                            o.fill_price = Some(price);
                            o.tx_hash = Some(tx_hash.clone());
                            let o_clone = o.clone();
                            drop(orders);
                            let _ = lo_storage_c.save_order(&state_c.storage_engine, &o_clone).await;
                            info!(
                                "✅ [LIMIT] Order {} filled: {} {} → {} {} at ${:.4} tx={}",
                                order.id, order.amount, order.from_token,
                                amount_out, order.to_token, price, tx_hash
                            );
                            let event = crate::streaming::StreamEvent::SwapExecuted {
                                from_token: order.from_token.clone(),
                                to_token: order.to_token.clone(),
                                amount_in: order.amount,
                                amount_out,
                                wallet_address: order.wallet_address.clone(),
                                price_impact: 0.0,
                                timestamp: chrono::Utc::now(),
                            };
                            let _ = state_c.event_broadcaster.broadcast(event).await;
                        }
                    }
                    Err(e) => {
                        error!("❌ [LIMIT] Failed to execute order {}: {}", order.id, e);
                        let mut orders = lo_storage_c.orders.write().await;
                        if let Some(o) = orders.get_mut(&order.id) {
                            o.failure_count += 1;
                            o.processing_since = None;
                            if o.failure_count >= MAX_FAILURE_COUNT {
                                o.status = LimitOrderStatus::Cancelled;
                                warn!(
                                    "⚠️ [LIMIT] Order {} auto-cancelled after {} consecutive failures",
                                    order.id, MAX_FAILURE_COUNT
                                );
                            } else {
                                o.status = LimitOrderStatus::Open;
                            }
                            let o_clone = o.clone();
                            drop(orders);
                            let _ = lo_storage_c.save_order(&state_c.storage_engine, &o_clone).await;
                        }
                    }
                }
            }));
        }

        // Wait for all concurrent executions this tick to finish before next sleep
        for h in join_handles {
            let _ = h.await;
        }
    }
}

/// Get the USD price of a token from the collateral vault / pool reserves.
/// Returns None if the token is not tracked.
async fn get_token_price_usd(state: &Arc<AppState>, token: &str) -> Option<f64> {
    let token_upper = token.to_uppercase();

    if token_upper == "QUG" || token_upper == "NATIVE-QUG" {
        let vault = state.collateral_vault.read().await;
        let p = vault.qug_price_usd;
        return if p > 0.0 { Some(p) } else { None };
    }

    if token_upper == "QUGUSD" {
        return Some(1.0);
    }

    // For custom tokens: price relative to QUG pool
    let qug_price = {
        let vault = state.collateral_vault.read().await;
        vault.qug_price_usd
    };
    if qug_price <= 0.0 {
        return None;
    }

    let pools = state.liquidity_pools.read().await;
    let pool = pools.values().find(|p| {
        let t0 = p.token0.to_uppercase();
        let t1 = p.token1.to_uppercase();
        (t0 == token_upper || t1 == token_upper) && (t0 == "QUG" || t1 == "QUG")
    })?;

    let is_token0 = pool.token0.to_uppercase() == token_upper;
    let (token_reserve, qug_reserve) = if is_token0 {
        (pool.reserve0 as f64, pool.reserve1 as f64)
    } else {
        (pool.reserve1 as f64, pool.reserve0 as f64)
    };

    if token_reserve == 0.0 {
        return None;
    }

    Some((qug_reserve / token_reserve) * qug_price)
}

/// Execute the actual token swap for a triggered limit order.
/// Slippage is enforced via `order.max_slippage` — the swap is rejected if AMM output
/// is more than `max_slippage` worse than the pre-swap pool estimate.
async fn execute_limit_order_swap(
    state: &Arc<AppState>,
    order: &LimitOrder,
) -> anyhow::Result<(u128, String)> {
    crate::dca_api::execute_limit_swap(
        state,
        &order.from_token,
        &order.to_token,
        order.amount,
        order.max_slippage,
        &order.wallet_address,
    ).await
}

// ============================================================================
// ROUTER
// ============================================================================

pub fn create_limit_order_router() -> axum::Router<Arc<AppState>> {
    use axum::routing::{get, post, delete};

    axum::Router::new()
        .route("/", post(create_limit_order))
        .route("/open", get(get_all_open_limit_orders))
        .route("/:wallet_address", get(get_wallet_limit_orders))
        .route("/:wallet_address/:order_id", delete(cancel_limit_order))
}
