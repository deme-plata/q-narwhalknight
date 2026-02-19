// ============================================================================
// v6.5.0: Exchange Listing RWA — Exchange Listing Package Marketplace
// ============================================================================
//
// Three tiers of exchange listing packages:
//   GOLD   ($100,000) — Top-tier exchanges, 90-day premium marketing, 20+ influencers
//   SILVER ($25,000)  — Mid-tier exchanges, 30-day marketing, 5 influencers
//   BRONZE ($5,000)   — Small exchange listing, basic social media marketing
//
// Payment methods:
//   1. QUG (native coin) — Deducted from wallet_balances
//   2. QUGUSD (stablecoin) — Deducted from token_balances
//   3. Stripe USD (fiat) — Creates Stripe PaymentIntent, credited on confirmation
//
// Persistence: RocksDB CF_MANIFEST with "listing_order_" prefix + in-memory Vec

use axum::{
    extract::{Json, Path, Query, State},
    http::StatusCode,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{error, info, warn};

use crate::contracts_api::ApiResponse;
use crate::wallet_auth::AuthenticatedWallet;
use crate::AppState;

// ============================================================================
// Constants
// ============================================================================

/// XLIST RWA token address — "XLIST" in ASCII + zeros
pub const XLIST_TOKEN_ADDRESS: [u8; 32] = [
    0x58, 0x4C, 0x49, 0x53, 0x54, 0x00, 0x00, 0x00, // "XLIST" in ASCII
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

// ============================================================================
// Types
// ============================================================================

/// Listing tier enum
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ListingTier {
    Gold,
    Silver,
    Bronze,
}

impl ListingTier {
    /// Price in USD (display units)
    pub fn price_usd(&self) -> f64 {
        match self {
            ListingTier::Gold => 100_000.0,
            ListingTier::Silver => 25_000.0,
            ListingTier::Bronze => 5_000.0,
        }
    }

    /// Price in QUGUSD base units (24 decimals)
    pub fn price_qugusd_base(&self) -> u128 {
        match self {
            ListingTier::Gold => 100_000 * 10u128.pow(24),
            ListingTier::Silver => 25_000 * 10u128.pow(24),
            ListingTier::Bronze => 5_000 * 10u128.pow(24),
        }
    }

    pub fn display_name(&self) -> &str {
        match self {
            ListingTier::Gold => "Gold Exchange Listing Package",
            ListingTier::Silver => "Silver Exchange Listing Package",
            ListingTier::Bronze => "Bronze Exchange Listing Package",
        }
    }

    pub fn exchange_tier(&self) -> &str {
        match self {
            ListingTier::Gold => "Top-Tier (Binance, Coinbase, Kraken, OKX)",
            ListingTier::Silver => "Mid-Tier (Gate.io, KuCoin, MEXC, Bybit)",
            ListingTier::Bronze => "Small Exchange (LBank, BitMart, Poloniex)",
        }
    }

    pub fn marketing_duration_days(&self) -> u32 {
        match self {
            ListingTier::Gold => 90,
            ListingTier::Silver => 30,
            ListingTier::Bronze => 7,
        }
    }

    pub fn influencer_count(&self) -> u32 {
        match self {
            ListingTier::Gold => 20,
            ListingTier::Silver => 5,
            ListingTier::Bronze => 0,
        }
    }

    pub fn features(&self) -> Vec<&str> {
        match self {
            ListingTier::Gold => vec![
                "Top-tier exchange listing application & support",
                "90-day premium marketing campaign",
                "20+ crypto influencer partnerships",
                "Professional market maker coordination",
                "CoinGecko & CoinMarketCap priority listing",
                "Dedicated listing manager",
                "PR & media coverage (CoinDesk, CoinTelegraph)",
                "Community management setup (Discord, Telegram)",
                "AMA sessions with exchanges",
                "Custom trading pair setup (USDT, BTC, ETH)",
            ],
            ListingTier::Silver => vec![
                "Mid-tier exchange listing application",
                "30-day targeted marketing campaign",
                "5 crypto influencer partnerships",
                "CoinGecko & CoinMarketCap listing",
                "Social media marketing (Twitter, Reddit)",
                "Community channel setup",
                "Trading pair: USDT",
            ],
            ListingTier::Bronze => vec![
                "Small exchange listing application",
                "7-day social media campaign",
                "CoinGecko listing submission",
                "Basic Twitter marketing",
                "Trading pair: USDT",
            ],
        }
    }
}

/// Payment method for listing purchase
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PaymentMethod {
    /// Pay with QUG (native coin)
    Qug,
    /// Pay with QUGUSD (stablecoin)
    Qugusd,
    /// Pay with Stripe (fiat USD)
    StripeUsd,
}

/// A listing order
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListingOrder {
    pub order_id: String,
    pub buyer_wallet: String,
    pub tier: ListingTier,
    pub payment_method: PaymentMethod,
    /// Amount paid (display units for QUG/QUGUSD, cents for Stripe)
    pub amount_paid: String,
    /// Token symbol being listed
    pub token_symbol: String,
    /// Token contract address (hex)
    pub token_address: String,
    /// Project name
    pub project_name: String,
    /// Project website
    pub project_website: Option<String>,
    /// Contact email
    pub contact_email: String,
    /// Additional notes/description
    pub notes: Option<String>,
    /// Stripe payment intent ID (only for stripe payments)
    pub stripe_payment_intent_id: Option<String>,
    /// Order status
    pub status: String, // "pending_payment", "paid", "processing", "exchange_contacted", "listed", "rejected"
    /// Target exchange name (filled during processing)
    pub target_exchange: Option<String>,
    /// Marketing campaign start date
    pub marketing_start: Option<u64>,
    /// Marketing campaign end date
    pub marketing_end: Option<u64>,
    pub created_at: u64,
    pub updated_at: u64,
}

// ============================================================================
// Request/Response Types
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct PurchaseListingRequest {
    pub tier: ListingTier,
    pub payment_method: PaymentMethod,
    pub token_symbol: String,
    pub token_address: String,
    pub project_name: String,
    pub project_website: Option<String>,
    pub contact_email: String,
    pub notes: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct FulfillListingRequest {
    pub order_id: String,
    pub status: String,
    pub target_exchange: Option<String>,
    pub marketing_start: Option<u64>,
    pub marketing_end: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub struct ListingQueryParams {
    pub status: Option<String>,
}

// ============================================================================
// Persistence Helpers
// ============================================================================

const LISTING_ORDER_PREFIX: &str = "listing_order_";

async fn persist_listing_order(state: &AppState, order: &ListingOrder) {
    let key = format!("{}{}", LISTING_ORDER_PREFIX, order.order_id);
    match serde_json::to_vec(order) {
        Ok(data) => {
            let kv = state.storage_engine.get_kv();
            if let Err(e) = kv
                .put_sync(q_storage::CF_MANIFEST, key.as_bytes(), &data)
                .await
            {
                warn!("Failed to persist listing order {}: {}", order.order_id, e);
            }
        }
        Err(e) => warn!("Failed to serialize listing order: {}", e),
    }
}

pub async fn load_listing_orders_from_db(state: &AppState) -> Vec<ListingOrder> {
    let mut orders = Vec::new();
    let prefix = LISTING_ORDER_PREFIX.as_bytes();
    let kv = state.storage_engine.get_kv();

    match kv.scan_prefix(q_storage::CF_MANIFEST, prefix).await {
        Ok(entries) => {
            for (_key, value) in entries {
                match serde_json::from_slice::<ListingOrder>(&value) {
                    Ok(order) => orders.push(order),
                    Err(e) => warn!("Failed to deserialize listing order: {}", e),
                }
            }
        }
        Err(e) => warn!("Failed to load listing orders from DB: {}", e),
    }

    orders.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    orders
}

// ============================================================================
// API Handlers
// ============================================================================

/// GET /api/v1/contracts/listing/packages — Get available listing packages
pub async fn listing_get_packages(
    State(_state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let packages: Vec<serde_json::Value> = [ListingTier::Gold, ListingTier::Silver, ListingTier::Bronze]
        .iter()
        .map(|tier| {
            serde_json::json!({
                "tier": tier,
                "name": tier.display_name(),
                "price_usd": tier.price_usd(),
                "exchange_tier": tier.exchange_tier(),
                "marketing_duration_days": tier.marketing_duration_days(),
                "influencer_count": tier.influencer_count(),
                "features": tier.features(),
                "payment_methods": ["qug", "qugusd", "stripe_usd"],
            })
        })
        .collect();

    Ok(Json(ApiResponse::success(serde_json::json!({
        "packages": packages,
        "total_packages": 3,
    }))))
}

/// POST /api/v1/contracts/listing/purchase — Purchase a listing package
pub async fn listing_purchase(
    auth: AuthenticatedWallet,
    State(state): State<Arc<AppState>>,
    Json(request): Json<PurchaseListingRequest>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let wallet = auth.address;
    let wallet_hex = hex::encode(wallet);
    let tier = request.tier;
    let price_usd = tier.price_usd();

    info!(
        "📋 [LISTING] Purchase request: tier={:?}, payment={:?}, token={}, wallet=qnk{}",
        tier, request.payment_method, request.token_symbol, &wallet_hex[..16]
    );

    // Generate order ID
    let order_id = format!(
        "XL-{}-{}-{}",
        match tier {
            ListingTier::Gold => "G",
            ListingTier::Silver => "S",
            ListingTier::Bronze => "B",
        },
        chrono::Utc::now().timestamp(),
        &wallet_hex[..8]
    );

    let mut amount_paid_display;
    let mut stripe_intent_id: Option<String> = None;
    let initial_status;

    match &request.payment_method {
        PaymentMethod::Qug => {
            // Pay with QUG — need QUG price from collateral vault
            let qug_price_usd = {
                let vault = state.collateral_vault.read().await;
                vault.qug_price_usd
            };

            if qug_price_usd <= 0.0 {
                return Ok(Json(ApiResponse::error(
                    "QUG price unavailable. Try again later.".to_string(),
                )));
            }

            // Calculate QUG amount needed
            let qug_amount_display = price_usd / qug_price_usd;
            let qug_amount_base = (qug_amount_display * 1e24) as u128;

            // Check QUG balance
            {
                let balances = state.wallet_balances.read().await;
                let current = balances.get(&wallet).copied().unwrap_or(0);
                if current < qug_amount_base {
                    let current_display = current as f64 / 1e24;
                    return Ok(Json(ApiResponse::error(format!(
                        "Insufficient QUG balance. Need {:.2} QUG (${:.0}), have {:.2} QUG",
                        qug_amount_display, price_usd, current_display
                    ))));
                }
            }

            // Deduct QUG
            {
                let mut balances = state.wallet_balances.write().await;
                if let Some(balance) = balances.get_mut(&wallet) {
                    *balance = balance.saturating_sub(qug_amount_base);
                    info!(
                        "💰 [LISTING] Deducted {:.2} QUG from wallet qnk{} for {:?} package",
                        qug_amount_display, &wallet_hex[..16], tier
                    );
                }
            }

            // Persist balance
            {
                let balances = state.wallet_balances.read().await;
                let new_balance = balances.get(&wallet).copied().unwrap_or(0);
                drop(balances);
                let _ = state
                    .storage_engine
                    .save_wallet_balance(&wallet, new_balance)
                    .await;
            }

            amount_paid_display = format!("{:.2} QUG", qug_amount_display);
            initial_status = "paid".to_string();
        }

        PaymentMethod::Qugusd => {
            // Pay with QUGUSD stablecoin
            let qugusd_addr = q_types::QUGUSD_TOKEN_ADDRESS;
            let balance_key = (wallet, qugusd_addr);
            let qugusd_amount_base = tier.price_qugusd_base();

            // Check QUGUSD balance
            {
                let token_balances = state.token_balances.read().await;
                let current = token_balances.get(&balance_key).copied().unwrap_or(0);
                if current < qugusd_amount_base {
                    let current_display = current as f64 / 1e24;
                    return Ok(Json(ApiResponse::error(format!(
                        "Insufficient QUGUSD balance. Need ${:.0}, have ${:.2}",
                        price_usd, current_display
                    ))));
                }
            }

            // Deduct QUGUSD
            {
                let mut token_balances = state.token_balances.write().await;
                if let Some(balance) = token_balances.get_mut(&balance_key) {
                    *balance = balance.saturating_sub(qugusd_amount_base);
                    info!(
                        "💵 [LISTING] Deducted ${:.0} QUGUSD from wallet qnk{} for {:?} package",
                        price_usd, &wallet_hex[..16], tier
                    );
                }
            }

            // Persist token balance
            {
                let token_balances = state.token_balances.read().await;
                let new_balance = token_balances.get(&balance_key).copied().unwrap_or(0);
                drop(token_balances);
                let _ = state
                    .storage_engine
                    .save_token_balance(&wallet, &qugusd_addr, new_balance)
                    .await;
            }

            amount_paid_display = format!("${:.0} QUGUSD", price_usd);
            initial_status = "paid".to_string();
        }

        PaymentMethod::StripeUsd => {
            // Use pre-initialized Stripe client from AppState
            let stripe_client = match &state.stripe_client {
                Some(c) => c,
                None => {
                    error!("Stripe not configured - set STRIPE_SECRET_KEY env var");
                    return Ok(Json(ApiResponse::error(
                        "Stripe payments not available. Please use QUG or QUGUSD.".to_string(),
                    )));
                }
            };

            let amount_cents = (price_usd * 100.0) as i64;

            let mut create_intent =
                stripe::CreatePaymentIntent::new(amount_cents, stripe::Currency::USD);
            create_intent.metadata = Some(std::collections::HashMap::from([
                ("wallet_address".to_string(), format!("qnk{}", wallet_hex)),
                ("order_id".to_string(), order_id.clone()),
                ("tier".to_string(), format!("{:?}", tier)),
                ("purpose".to_string(), "exchange_listing_package".to_string()),
            ]));

            match stripe::PaymentIntent::create(stripe_client, create_intent).await {
                Ok(intent) => {
                    stripe_intent_id = Some(intent.id.to_string());
                    amount_paid_display = format!("${:.0} USD (Stripe)", price_usd);
                    info!(
                        "💳 [LISTING] Stripe PaymentIntent created: {} for ${:.0}",
                        intent.id, price_usd
                    );
                }
                Err(e) => {
                    error!("Stripe PaymentIntent creation failed: {}", e);
                    return Ok(Json(ApiResponse::error(format!(
                        "Payment processing error: {}",
                        e
                    ))));
                }
            }

            initial_status = "pending_payment".to_string();
        }
    }

    // Create the order
    let now = chrono::Utc::now().timestamp() as u64;
    let order = ListingOrder {
        order_id: order_id.clone(),
        buyer_wallet: format!("qnk{}", wallet_hex),
        tier,
        payment_method: request.payment_method.clone(),
        amount_paid: amount_paid_display.clone(),
        token_symbol: request.token_symbol.clone(),
        token_address: request.token_address.clone(),
        project_name: request.project_name.clone(),
        project_website: request.project_website.clone(),
        contact_email: request.contact_email.clone(),
        notes: request.notes.clone(),
        stripe_payment_intent_id: stripe_intent_id.clone(),
        status: initial_status.clone(),
        target_exchange: None,
        marketing_start: None,
        marketing_end: None,
        created_at: now,
        updated_at: now,
    };

    // Store in memory
    {
        let mut orders = state.listing_orders.write().await;
        orders.push(order.clone());
    }

    // Persist to RocksDB
    persist_listing_order(&state, &order).await;

    info!(
        "📋 [LISTING] Order {} created: {:?} package, {}, token={}, wallet=qnk{}",
        order_id, tier, amount_paid_display, request.token_symbol, &wallet_hex[..16]
    );

    let mut response = serde_json::json!({
        "order_id": order_id,
        "tier": tier,
        "status": initial_status,
        "amount_paid": amount_paid_display,
        "token_symbol": request.token_symbol,
        "project_name": request.project_name,
        "message": format!(
            "Exchange listing package ({:?}) {} for {}.",
            tier,
            if initial_status == "paid" { "purchased successfully" } else { "created — awaiting payment" },
            request.token_symbol
        ),
    });

    // Include Stripe client secret for frontend payment flow
    if let Some(intent_id) = &stripe_intent_id {
        response["stripe_payment_intent_id"] = serde_json::json!(intent_id);
    }

    Ok(Json(ApiResponse::success(response)))
}

/// GET /api/v1/contracts/listing/orders — Get listing orders
pub async fn listing_get_orders(
    auth: AuthenticatedWallet,
    State(state): State<Arc<AppState>>,
    Query(params): Query<ListingQueryParams>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let wallet = auth.address;
    let wallet_hex = hex::encode(wallet);
    let is_admin = wallet == q_types::BANK_MASTER_ACCOUNT;

    let orders = state.listing_orders.read().await;

    let filtered: Vec<&ListingOrder> = orders
        .iter()
        .filter(|o| {
            // Admin sees all, users see their own
            let wallet_match = is_admin || o.buyer_wallet == format!("qnk{}", wallet_hex);
            // Optional status filter
            let status_match = params
                .status
                .as_ref()
                .map_or(true, |s| o.status == *s);
            wallet_match && status_match
        })
        .collect();

    Ok(Json(ApiResponse::success(serde_json::json!({
        "orders": filtered,
        "total": filtered.len(),
        "is_admin": is_admin,
    }))))
}

/// POST /api/v1/contracts/listing/fulfill — Admin updates order status
pub async fn listing_fulfill(
    auth: AuthenticatedWallet,
    State(state): State<Arc<AppState>>,
    Json(request): Json<FulfillListingRequest>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let wallet = auth.address;
    if wallet != q_types::BANK_MASTER_ACCOUNT {
        return Ok(Json(ApiResponse::error(
            "Admin access required".to_string(),
        )));
    }

    let now = chrono::Utc::now().timestamp() as u64;
    let mut updated = false;

    {
        let mut orders = state.listing_orders.write().await;
        if let Some(order) = orders.iter_mut().find(|o| o.order_id == request.order_id) {
            order.status = request.status.clone();
            order.updated_at = now;

            if let Some(exchange) = &request.target_exchange {
                order.target_exchange = Some(exchange.clone());
            }
            if let Some(start) = request.marketing_start {
                order.marketing_start = Some(start);
            }
            if let Some(end) = request.marketing_end {
                order.marketing_end = Some(end);
            }

            // Persist updated order
            let order_clone = order.clone();
            drop(orders);
            persist_listing_order(&state, &order_clone).await;
            updated = true;

            info!(
                "📋 [LISTING] Order {} updated: status={}, exchange={:?}",
                request.order_id, request.status, request.target_exchange
            );
        }
    }

    if updated {
        Ok(Json(ApiResponse::success(serde_json::json!({
            "order_id": request.order_id,
            "status": request.status,
            "message": "Order updated successfully",
        }))))
    } else {
        Ok(Json(ApiResponse::error(format!(
            "Order {} not found",
            request.order_id
        ))))
    }
}

/// GET /api/v1/contracts/listing/stats — Get listing statistics
pub async fn listing_get_stats(
    auth: AuthenticatedWallet,
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let wallet = auth.address;
    if wallet != q_types::BANK_MASTER_ACCOUNT {
        return Ok(Json(ApiResponse::error(
            "Admin access required".to_string(),
        )));
    }

    let orders = state.listing_orders.read().await;

    let total = orders.len();
    let gold_count = orders.iter().filter(|o| o.tier == ListingTier::Gold).count();
    let silver_count = orders
        .iter()
        .filter(|o| o.tier == ListingTier::Silver)
        .count();
    let bronze_count = orders
        .iter()
        .filter(|o| o.tier == ListingTier::Bronze)
        .count();

    let paid_count = orders
        .iter()
        .filter(|o| o.status != "pending_payment" && o.status != "rejected")
        .count();
    let listed_count = orders.iter().filter(|o| o.status == "listed").count();
    let pending_count = orders
        .iter()
        .filter(|o| o.status == "pending_payment" || o.status == "paid" || o.status == "processing")
        .count();

    // Revenue estimation
    let total_revenue_usd: f64 = orders
        .iter()
        .filter(|o| o.status != "pending_payment" && o.status != "rejected")
        .map(|o| o.tier.price_usd())
        .sum();

    // Payment method breakdown
    let qug_payments = orders
        .iter()
        .filter(|o| matches!(o.payment_method, PaymentMethod::Qug))
        .count();
    let qugusd_payments = orders
        .iter()
        .filter(|o| matches!(o.payment_method, PaymentMethod::Qugusd))
        .count();
    let stripe_payments = orders
        .iter()
        .filter(|o| matches!(o.payment_method, PaymentMethod::StripeUsd))
        .count();

    Ok(Json(ApiResponse::success(serde_json::json!({
        "total_orders": total,
        "by_tier": {
            "gold": gold_count,
            "silver": silver_count,
            "bronze": bronze_count,
        },
        "by_status": {
            "paid": paid_count,
            "listed": listed_count,
            "pending": pending_count,
        },
        "by_payment_method": {
            "qug": qug_payments,
            "qugusd": qugusd_payments,
            "stripe_usd": stripe_payments,
        },
        "total_revenue_usd": total_revenue_usd,
    }))))
}

/// POST /api/v1/contracts/listing/confirm-stripe — Confirm Stripe payment
pub async fn listing_confirm_stripe(
    auth: AuthenticatedWallet,
    State(state): State<Arc<AppState>>,
    Json(request): Json<StripeConfirmRequest>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let wallet = auth.address;
    let wallet_hex = hex::encode(wallet);

    // Find the order with this Stripe intent
    let now = chrono::Utc::now().timestamp() as u64;
    let mut confirmed = false;

    {
        let mut orders = state.listing_orders.write().await;
        if let Some(order) = orders.iter_mut().find(|o| {
            o.stripe_payment_intent_id.as_deref() == Some(&request.payment_intent_id)
                && o.buyer_wallet == format!("qnk{}", wallet_hex)
                && o.status == "pending_payment"
        }) {
            order.status = "paid".to_string();
            order.updated_at = now;
            confirmed = true;

            let order_clone = order.clone();
            let order_id = order.order_id.clone();
            drop(orders);
            persist_listing_order(&state, &order_clone).await;

            info!(
                "💳 [LISTING] Stripe payment confirmed for order {}: {}",
                order_id, request.payment_intent_id
            );
        }
    }

    if confirmed {
        Ok(Json(ApiResponse::success(serde_json::json!({
            "confirmed": true,
            "message": "Payment confirmed. Your listing package is now being processed.",
        }))))
    } else {
        Ok(Json(ApiResponse::error(
            "Order not found or already confirmed".to_string(),
        )))
    }
}

#[derive(Debug, Deserialize)]
pub struct StripeConfirmRequest {
    pub payment_intent_id: String,
}
