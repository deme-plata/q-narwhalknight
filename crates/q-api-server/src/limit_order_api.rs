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
use tokio::sync::RwLock;
use std::collections::HashMap;
use tracing::{info, warn, error, debug};

use crate::AppState;

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
    pub filled_at: Option<i64>,
    /// Optional expiry timestamp in ms (None = GTC)
    pub expiry: Option<i64>,
    /// Actual output after fill
    #[serde(serialize_with = "q_types::u128_serde::serialize", deserialize_with = "q_types::u128_serde::deserialize")]
    pub amount_out: u128,
    /// Price at which the order was filled (for display)
    pub fill_price: Option<f64>,
    pub tx_hash: Option<String>,
}

/// In-memory + RocksDB storage for limit orders
pub struct LimitOrderStorage {
    pub orders: RwLock<HashMap<String, LimitOrder>>,
}

impl LimitOrderStorage {
    pub fn new() -> Self {
        LimitOrderStorage {
            orders: RwLock::new(HashMap::new()),
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
        filled_at: None,
        expiry: req.expiry,
        amount_out: 0,
        fill_price: None,
        tx_hash: None,
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
pub async fn cancel_limit_order(
    State(state): State<Arc<AppState>>,
    Path((wallet_address, order_id)): Path<(String, String)>,
) -> impl IntoResponse {
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

        for order in candidates {
            // Expire if past expiry
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

            // Get current price for the watched token
            let current_price = get_token_price_usd(&state, &order.price_token).await;
            let Some(price) = current_price else {
                debug!("🎯 [LIMIT] Cannot get price for {} — skipping order {}", order.price_token, order.id);
                continue;
            };

            let triggered = match order.direction {
                PriceDirection::Above => price >= order.trigger_price,
                PriceDirection::Below => price <= order.trigger_price,
            };

            if !triggered {
                debug!(
                    "🎯 [LIMIT] Order {} not triggered: current={:.4} trigger={:.4} {:?}",
                    order.id, price, order.trigger_price, order.direction
                );
                continue;
            }

            info!(
                "🎯 [LIMIT] Order {} TRIGGERED: {} {} {} ${:.4} (trigger ${:.4})",
                order.id, order.price_token, if matches!(order.direction, PriceDirection::Above) { ">=" } else { "<=" },
                "price", price, order.trigger_price
            );

            // Execute the swap
            match execute_limit_order_swap(&state, &order).await {
                Ok((amount_out, tx_hash)) => {
                    let mut orders = lo_storage.orders.write().await;
                    if let Some(o) = orders.get_mut(&order.id) {
                        o.status = LimitOrderStatus::Filled;
                        o.filled_at = Some(now);
                        o.amount_out = amount_out;
                        o.fill_price = Some(price);
                        o.tx_hash = Some(tx_hash.clone());
                        let o_clone = o.clone();
                        drop(orders);

                        let _ = lo_storage.save_order(&state.storage_engine, &o_clone).await;

                        info!(
                            "✅ [LIMIT] Order {} filled: {} {} → {} {} at ${:.4} tx={}",
                            order.id, order.amount, order.from_token,
                            amount_out, order.to_token, price, tx_hash
                        );

                        // Broadcast fill via SSE (reuse SwapExecuted so frontend picks it up)
                        let event = crate::streaming::StreamEvent::SwapExecuted {
                            from_token: order.from_token.clone(),
                            to_token: order.to_token.clone(),
                            amount_in: order.amount,
                            amount_out,
                            wallet_address: order.wallet_address.clone(),
                            price_impact: 0.0,
                            timestamp: chrono::Utc::now(),
                        };
                        let _ = state.event_broadcaster.broadcast(event).await;
                    } else {
                        drop(orders);
                    }
                }
                Err(e) => {
                    error!("❌ [LIMIT] Failed to execute order {}: {}", order.id, e);
                    // Leave as Open — will retry next tick unless it expires
                }
            }
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
/// Reuses the same balance-update logic as DCA swap execution.
async fn execute_limit_order_swap(
    state: &Arc<AppState>,
    order: &LimitOrder,
) -> anyhow::Result<(u128, String)> {
    // Delegate to the DCA swap function (same pool + balance logic)
    crate::dca_api::execute_limit_swap(
        state,
        &order.from_token,
        &order.to_token,
        order.amount,
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
