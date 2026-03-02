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
            ListingTier::Gold => 200_000.0,   // v8.6.0: doubled from 100K
            ListingTier::Silver => 50_000.0,  // v8.6.0: doubled from 25K
            ListingTier::Bronze => 12_000.0,  // v8.6.0: raised from 5K
        }
    }

    /// Price in QUGUSD base units (24 decimals)
    pub fn price_qugusd_base(&self) -> u128 {
        match self {
            ListingTier::Gold => 200_000 * 10u128.pow(24),   // v8.6.0: doubled from 100K
            ListingTier::Silver => 50_000 * 10u128.pow(24),  // v8.6.0: doubled from 25K
            ListingTier::Bronze => 12_000 * 10u128.pow(24),  // v8.6.0: raised from 5K
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
            ListingTier::Gold => 60, // v8.6.0: reduced from 90 days
            ListingTier::Silver => 30,
            ListingTier::Bronze => 7,
        }
    }

    pub fn influencer_count(&self) -> u32 {
        match self {
            ListingTier::Gold => 12, // v8.6.0: reduced from 20
            ListingTier::Silver => 5,
            ListingTier::Bronze => 0,
        }
    }

    pub fn features(&self) -> Vec<&str> {
        match self {
            ListingTier::Gold => vec![
                "Top-tier exchange listing application & support",
                "60-day premium marketing campaign",
                "12+ crypto influencer partnerships",
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

// ============================================================================
// v8.2.8: XLIST CROWDFUNDING CAMPAIGNS
// ============================================================================
//
// Community crowdfunding for exchange listings.
// Anyone can contribute QUG, QUGUSD, or Stripe USD toward a shared goal.
//
// THE GAME:
//   Phase 1 "Funding"  — progress bar fills, leaderboard ranks contributors
//   Phase 2 "Endgame"  — target hit! Celebration + listing purchase triggered
//   Phase 3 "Perks"    — contributors unlock tier-based rewards
//
// CONTRIBUTOR TIERS (based on % of total raised):
//   Diamond (≥5%)   — 3× airdrop multiplier, governance vote, name on listing announcement
//   Gold    (≥1%)   — 2× airdrop multiplier, governance vote
//   Silver  (≥0.1%) — 1.5× airdrop multiplier
//   Bronze  (any)   — 1× airdrop multiplier, supporter badge
//
// EARLY BIRD: First 50 contributors get +0.5× bonus multiplier regardless of amount

const CAMPAIGN_PREFIX: &str = "xlist_campaign_";
const CONTRIBUTION_PREFIX: &str = "xlist_contrib_";

/// A crowdfunding campaign targeting a specific exchange listing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListingCampaign {
    pub campaign_id: String,
    pub exchange_name: String,
    pub exchange_logo: Option<String>,
    pub target_usd: f64,
    pub raised_usd: f64,
    pub contributor_count: u32,
    pub early_bird_slots: u32,
    pub early_bird_claimed: u32,
    pub status: CampaignStatus,
    pub tier: ListingTier,
    pub description: String,
    pub perks: Vec<PerkTier>,
    pub created_at: u64,
    pub updated_at: u64,
    pub funded_at: Option<u64>,
    pub listed_at: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CampaignStatus {
    /// Accepting contributions
    Funding,
    /// Target reached, processing listing
    Funded,
    /// Exchange listing confirmed
    Listed,
    /// Campaign cancelled, refunds available
    Cancelled,
}

/// A single contribution to a campaign
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Contribution {
    pub contribution_id: String,
    pub campaign_id: String,
    pub wallet: String,
    pub amount_usd: f64,
    pub payment_method: PaymentMethod,
    pub raw_amount: String,
    pub stripe_payment_intent_id: Option<String>,
    pub is_early_bird: bool,
    pub created_at: u64,
}

/// Perk tier definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerkTier {
    pub name: String,
    pub min_percent: f64,
    pub airdrop_multiplier: f64,
    pub governance_vote: bool,
    pub badge: String,
    pub perks: Vec<String>,
}

/// Contributor summary with computed perks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContributorSummary {
    pub wallet: String,
    pub total_usd: f64,
    pub contribution_count: u32,
    pub percent_of_total: f64,
    pub tier_name: String,
    pub airdrop_multiplier: f64,
    pub is_early_bird: bool,
    pub rank: u32,
    pub badges: Vec<String>,
}

fn default_perks() -> Vec<PerkTier> {
    vec![
        PerkTier {
            name: "Diamond".into(),
            min_percent: 5.0,
            airdrop_multiplier: 3.0,
            governance_vote: true,
            badge: "💎".into(),
            perks: vec![
                "3× airdrop multiplier on listing".into(),
                "Governance vote on next exchange".into(),
                "Name in listing announcement".into(),
                "VIP Discord role".into(),
                "Priority access to future campaigns".into(),
            ],
        },
        PerkTier {
            name: "Gold".into(),
            min_percent: 1.0,
            airdrop_multiplier: 2.0,
            governance_vote: true,
            badge: "🥇".into(),
            perks: vec![
                "2× airdrop multiplier on listing".into(),
                "Governance vote on next exchange".into(),
                "Gold Discord role".into(),
            ],
        },
        PerkTier {
            name: "Silver".into(),
            min_percent: 0.1,
            airdrop_multiplier: 1.5,
            governance_vote: false,
            badge: "🥈".into(),
            perks: vec![
                "1.5× airdrop multiplier on listing".into(),
                "Silver Discord role".into(),
            ],
        },
        PerkTier {
            name: "Bronze".into(),
            min_percent: 0.0,
            airdrop_multiplier: 1.0,
            governance_vote: false,
            badge: "🥉".into(),
            perks: vec![
                "1× airdrop multiplier".into(),
                "Supporter badge".into(),
            ],
        },
    ]
}

fn compute_contributor_tier(percent: f64, perks: &[PerkTier]) -> (&PerkTier, f64) {
    for perk in perks {
        if percent >= perk.min_percent {
            return (perk, perk.airdrop_multiplier);
        }
    }
    (&perks[perks.len() - 1], 1.0)
}

// Persistence helpers
async fn persist_campaign(state: &AppState, campaign: &ListingCampaign) {
    let key = format!("{}{}", CAMPAIGN_PREFIX, campaign.campaign_id);
    match serde_json::to_vec(campaign) {
        Ok(data) => {
            let kv = state.storage_engine.get_kv();
            if let Err(e) = kv.put_sync(q_storage::CF_MANIFEST, key.as_bytes(), &data).await {
                warn!("Failed to persist campaign {}: {}", campaign.campaign_id, e);
            }
        }
        Err(e) => warn!("Failed to serialize campaign: {}", e),
    }
}

async fn persist_contribution(state: &AppState, contrib: &Contribution) {
    let key = format!("{}{}_{}", CONTRIBUTION_PREFIX, contrib.campaign_id, contrib.contribution_id);
    match serde_json::to_vec(contrib) {
        Ok(data) => {
            let kv = state.storage_engine.get_kv();
            if let Err(e) = kv.put_sync(q_storage::CF_MANIFEST, key.as_bytes(), &data).await {
                warn!("Failed to persist contribution {}: {}", contrib.contribution_id, e);
            }
        }
        Err(e) => warn!("Failed to serialize contribution: {}", e),
    }
}

pub async fn load_campaigns_from_db(state: &AppState) -> Vec<ListingCampaign> {
    let mut campaigns = Vec::new();
    let prefix = CAMPAIGN_PREFIX.as_bytes();
    let kv = state.storage_engine.get_kv();
    match kv.scan_prefix(q_storage::CF_MANIFEST, prefix).await {
        Ok(entries) => {
            for (_key, value) in entries {
                if let Ok(c) = serde_json::from_slice::<ListingCampaign>(&value) {
                    campaigns.push(c);
                }
            }
        }
        Err(e) => warn!("Failed to load campaigns: {}", e),
    }
    campaigns
}

pub async fn load_contributions_for_campaign(state: &AppState, campaign_id: &str) -> Vec<Contribution> {
    let mut contribs = Vec::new();
    let prefix = format!("{}{}_", CONTRIBUTION_PREFIX, campaign_id);
    let kv = state.storage_engine.get_kv();
    match kv.scan_prefix(q_storage::CF_MANIFEST, prefix.as_bytes()).await {
        Ok(entries) => {
            for (_key, value) in entries {
                if let Ok(c) = serde_json::from_slice::<Contribution>(&value) {
                    contribs.push(c);
                }
            }
        }
        Err(e) => warn!("Failed to load contributions for {}: {}", campaign_id, e),
    }
    contribs
}

// ============================================================================
// v8.4.0: Seed default exchange listing campaigns
// ============================================================================

/// Seed initial exchange listing campaigns if none exist in the database.
/// Based on real quotes from listing agents (Kim Peterson, Feb 2026).
pub async fn seed_default_campaigns(state: &AppState) {
    let existing = state.listing_campaigns.read().await;
    if !existing.is_empty() {
        info!("📋 [XLIST] {} campaigns already exist, skipping seed", existing.len());
        return;
    }
    drop(existing);

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let campaigns = vec![
        ListingCampaign {
            campaign_id: "XLIST-MEXC-2026".to_string(),
            exchange_name: "MEXC".to_string(),
            exchange_logo: Some("https://altcoinsbox.com/wp-content/uploads/2023/01/mexc-logo.webp".to_string()),
            target_usd: 60_000.0,
            raised_usd: 0.0,
            contributor_count: 0,
            early_bird_slots: 100,
            early_bird_claimed: 0,
            status: CampaignStatus::Funding,
            tier: ListingTier::Gold,
            description: "Get QUG listed on MEXC — a top-15 exchange with 10M+ users. \
                MEXC is the best launchpad for new L1 projects with deep liquidity and global reach. \
                Quote: $60,000 USDT via verified listing agent. \
                Every contributor earns airdrop multipliers and governance votes on the next exchange target.".to_string(),
            perks: default_perks(),
            created_at: now,
            updated_at: now,
            funded_at: None,
            listed_at: None,
        },
        ListingCampaign {
            campaign_id: "XLIST-BITMART-2026".to_string(),
            exchange_name: "BitMart".to_string(),
            exchange_logo: Some("https://altcoinsbox.com/wp-content/uploads/2023/01/bitmart-logo.webp".to_string()),
            target_usd: 30_000.0,
            raised_usd: 0.0,
            contributor_count: 0,
            early_bird_slots: 75,
            early_bird_claimed: 0,
            status: CampaignStatus::Funding,
            tier: ListingTier::Silver,
            description: "Get QUG listed on BitMart — a popular tier-2 exchange in the top 30 on CMC. \
                BitMart has strong retail volume and a proven track record for new coin launches. \
                Quote: $30,000 USDT. Building exchange presence before targeting tier-1.".to_string(),
            perks: default_perks(),
            created_at: now,
            updated_at: now,
            funded_at: None,
            listed_at: None,
        },
        ListingCampaign {
            campaign_id: "XLIST-LBANK-2026".to_string(),
            exchange_name: "LBank".to_string(),
            exchange_logo: Some("https://altcoinsbox.com/wp-content/uploads/2023/01/lbank-logo.webp".to_string()),
            target_usd: 20_000.0,
            raised_usd: 0.0,
            contributor_count: 0,
            early_bird_slots: 50,
            early_bird_claimed: 0,
            status: CampaignStatus::Funding,
            tier: ListingTier::Bronze,
            description: "Get QUG listed on LBank — an accessible tier-2 exchange with fast onboarding. \
                LBank is a great first CEX listing to establish market presence and get tracked on CoinGecko/CMC. \
                Quote: $20,000 USDT. Lowest barrier to entry for our first exchange listing.".to_string(),
            perks: default_perks(),
            created_at: now,
            updated_at: now,
            funded_at: None,
            listed_at: None,
        },
    ];

    let count = campaigns.len();
    for campaign in &campaigns {
        persist_campaign(state, campaign).await;
    }

    let mut lock = state.listing_campaigns.write().await;
    *lock = campaigns;

    info!("🚀 [XLIST] Seeded {} exchange listing campaigns (MEXC $60K, BitMart $30K, LBank $20K)", count);
}

// ============================================================================
// API Handlers — Crowdfunding
// ============================================================================

/// GET /api/v1/contracts/listing/campaigns — List all campaigns
pub async fn campaign_list(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let campaigns = {
        let lock = state.listing_campaigns.read().await;
        lock.clone()
    };

    Ok(Json(ApiResponse::success(serde_json::json!({
        "campaigns": campaigns,
        "active_count": campaigns.iter().filter(|c| c.status == CampaignStatus::Funding).count(),
        "total_raised_usd": campaigns.iter().map(|c| c.raised_usd).sum::<f64>(),
    }))))
}

/// GET /api/v1/contracts/listing/campaigns/:id — Campaign details + leaderboard
pub async fn campaign_details(
    Path(campaign_id): Path<String>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let campaigns = state.listing_campaigns.read().await;
    let campaign = campaigns.iter().find(|c| c.campaign_id == campaign_id);

    let Some(campaign) = campaign else {
        return Ok(Json(ApiResponse::error("Campaign not found".into())));
    };

    let contributions = load_contributions_for_campaign(&state, &campaign_id).await;

    // Aggregate per wallet
    let mut wallet_totals: std::collections::HashMap<String, (f64, u32, bool)> = std::collections::HashMap::new();
    for c in &contributions {
        let entry = wallet_totals.entry(c.wallet.clone()).or_insert((0.0, 0, false));
        entry.0 += c.amount_usd;
        entry.1 += 1;
        if c.is_early_bird {
            entry.2 = true;
        }
    }

    let mut leaderboard: Vec<ContributorSummary> = wallet_totals
        .into_iter()
        .map(|(wallet, (total_usd, count, early_bird))| {
            let percent = if campaign.raised_usd > 0.0 {
                (total_usd / campaign.raised_usd) * 100.0
            } else {
                0.0
            };
            let (tier, mut multiplier) = compute_contributor_tier(percent, &campaign.perks);
            if early_bird {
                multiplier += 0.5; // Early bird bonus
            }
            ContributorSummary {
                wallet: wallet.clone(),
                total_usd,
                contribution_count: count,
                percent_of_total: percent,
                tier_name: tier.name.clone(),
                airdrop_multiplier: multiplier,
                is_early_bird: early_bird,
                rank: 0,
                badges: {
                    let mut b = vec![tier.badge.clone()];
                    if early_bird { b.push("🐦".into()); }
                    b
                },
            }
        })
        .collect();

    leaderboard.sort_by(|a, b| b.total_usd.partial_cmp(&a.total_usd).unwrap_or(std::cmp::Ordering::Equal));
    for (i, entry) in leaderboard.iter_mut().enumerate() {
        entry.rank = (i + 1) as u32;
    }

    // Build per-contribution list with payment method for transparency
    let contribution_details: Vec<serde_json::Value> = contributions.iter().map(|c| {
        serde_json::json!({
            "wallet": &c.wallet[..16],
            "amount_usd": c.amount_usd,
            "payment_method": c.payment_method,
            "raw_amount": c.raw_amount,
            "is_early_bird": c.is_early_bird,
            "created_at": c.created_at,
        })
    }).collect();

    Ok(Json(ApiResponse::success(serde_json::json!({
        "campaign": campaign,
        "leaderboard": leaderboard,
        "contributions": contribution_details,
        "contribution_count": contributions.len(),
        "progress_percent": (campaign.raised_usd / campaign.target_usd * 100.0).min(100.0),
        "remaining_usd": (campaign.target_usd - campaign.raised_usd).max(0.0),
        "is_funded": campaign.status == CampaignStatus::Funded || campaign.status == CampaignStatus::Listed,
    }))))
}

/// POST /api/v1/contracts/listing/campaigns/contribute — Contribute to a campaign
#[derive(Debug, Deserialize)]
pub struct ContributeRequest {
    pub campaign_id: String,
    pub payment_method: PaymentMethod,
    pub amount_usd: f64,
}

pub async fn campaign_contribute(
    auth: AuthenticatedWallet,
    State(state): State<Arc<AppState>>,
    Json(req): Json<ContributeRequest>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let wallet = auth.address;
    let wallet_hex = hex::encode(&wallet);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    if req.amount_usd < 1.0 {
        return Ok(Json(ApiResponse::error("Minimum contribution is $1".into())));
    }
    if req.amount_usd > 100_000.0 {
        return Ok(Json(ApiResponse::error("Maximum single contribution is $100,000".into())));
    }

    // Find campaign
    let mut campaigns = state.listing_campaigns.write().await;
    let campaign = campaigns.iter_mut().find(|c| c.campaign_id == req.campaign_id);
    let Some(campaign) = campaign else {
        return Ok(Json(ApiResponse::error("Campaign not found".into())));
    };

    if campaign.status != CampaignStatus::Funding {
        return Ok(Json(ApiResponse::error(
            format!("Campaign is {:?}, not accepting contributions", campaign.status),
        )));
    }

    // Cap contribution to remaining amount
    let remaining = campaign.target_usd - campaign.raised_usd;
    let actual_amount = req.amount_usd.min(remaining);

    // Process payment
    let (stripe_intent_id, payment_status) = match req.payment_method {
        PaymentMethod::Qug => {
            // Deduct QUG at current price
            let qug_price = {
                let vault = state.collateral_vault.read().await;
                vault.qug_price_usd
            };
            if qug_price <= 0.0 {
                return Ok(Json(ApiResponse::error("QUG price unavailable".into())));
            }
            let qug_needed_f64 = actual_amount / qug_price;
            let qug_needed = (qug_needed_f64 * 1e24) as u128;
            let mut balances = state.wallet_balances.write().await;
            let balance = balances.get(&wallet).copied().unwrap_or(0u128);
            if balance < qug_needed {
                let have_display = balance as f64 / 1e24;
                return Ok(Json(ApiResponse::error(
                    format!("Insufficient QUG. Need {:.4} QUG (${:.2}), have {:.4} QUG", qug_needed_f64, actual_amount, have_display),
                )));
            }
            *balances.entry(wallet).or_insert(0u128) -= qug_needed;
            info!("💰 [XLIST CROWDFUND] {} contributed {:.4} QUG (${:.2}) to campaign {}",
                &wallet_hex[..16], qug_needed_f64, actual_amount, campaign.campaign_id);
            (None, "paid")
        }
        PaymentMethod::Qugusd => {
            let qugusd_addr = q_types::QUGUSD_TOKEN_ADDRESS;
            let balance_key = (wallet, qugusd_addr);
            let amount_base = (actual_amount * 1e24) as u128;
            let mut token_bals = state.token_balances.write().await;
            let balance = token_bals.get(&balance_key).copied().unwrap_or(0);
            if balance < amount_base {
                let have_display = balance as f64 / 1e24;
                return Ok(Json(ApiResponse::error(
                    format!("Insufficient QUGUSD. Need ${:.2}, have ${:.2}", actual_amount, have_display),
                )));
            }
            *token_bals.entry(balance_key).or_insert(0) -= amount_base;
            info!("💰 [XLIST CROWDFUND] {} contributed ${:.2} QUGUSD to campaign {}",
                &wallet_hex[..16], actual_amount, campaign.campaign_id);
            (None, "paid")
        }
        PaymentMethod::StripeUsd => {
            // Create Stripe PaymentIntent
            if let Some(stripe_client) = &state.stripe_client {
                let amount_cents = (actual_amount * 100.0) as i64;
                let mut create_intent =
                    stripe::CreatePaymentIntent::new(amount_cents, stripe::Currency::USD);
                create_intent.metadata = Some(std::collections::HashMap::from([
                    ("type".into(), "xlist_crowdfund".into()),
                    ("campaign_id".into(), campaign.campaign_id.clone()),
                    ("wallet".into(), wallet_hex.clone()),
                ]));
                match stripe::PaymentIntent::create(stripe_client, create_intent).await {
                    Ok(intent) => {
                        let intent_id = intent.id.to_string();
                        let _client_secret = intent.client_secret.clone();
                        (Some(intent_id), "pending_payment")
                    }
                    Err(e) => {
                        error!("Stripe PaymentIntent creation failed: {}", e);
                        return Ok(Json(ApiResponse::error("Payment processing error".into())));
                    }
                }
            } else {
                return Ok(Json(ApiResponse::error("Stripe payments not configured".into())));
            }
        }
    };

    let is_early_bird = campaign.early_bird_claimed < campaign.early_bird_slots;
    if is_early_bird {
        campaign.early_bird_claimed += 1;
    }

    let contribution_id = format!("XC-{}-{}", now, &wallet_hex[..16.min(wallet_hex.len())]);
    let contribution = Contribution {
        contribution_id: contribution_id.clone(),
        campaign_id: campaign.campaign_id.clone(),
        wallet: wallet_hex.clone(),
        amount_usd: actual_amount,
        payment_method: req.payment_method.clone(),
        raw_amount: format!("{:.2}", actual_amount),
        stripe_payment_intent_id: stripe_intent_id.clone(),
        is_early_bird,
        created_at: now,
    };

    // Update campaign totals (only if payment is confirmed, not pending stripe)
    if payment_status == "paid" {
        campaign.raised_usd += actual_amount;
        campaign.contributor_count += 1;
        campaign.updated_at = now;

        // Check if funded
        if campaign.raised_usd >= campaign.target_usd {
            campaign.status = CampaignStatus::Funded;
            campaign.funded_at = Some(now);
            info!("🎉 [XLIST CROWDFUND] Campaign {} FUNDED! ${:.2} raised for {}",
                campaign.campaign_id, campaign.raised_usd, campaign.exchange_name);
        }
    }

    let campaign_snapshot = campaign.clone();
    persist_campaign(&state, &campaign_snapshot).await;
    persist_contribution(&state, &contribution).await;

    let progress = (campaign_snapshot.raised_usd / campaign_snapshot.target_usd * 100.0).min(100.0);

    Ok(Json(ApiResponse::success(serde_json::json!({
        "contribution_id": contribution_id,
        "amount_usd": actual_amount,
        "payment_status": payment_status,
        "stripe_client_secret": stripe_intent_id,
        "is_early_bird": is_early_bird,
        "campaign_progress": progress,
        "campaign_raised": campaign_snapshot.raised_usd,
        "campaign_target": campaign_snapshot.target_usd,
        "is_funded": campaign_snapshot.status == CampaignStatus::Funded,
    }))))
}

/// POST /api/v1/contracts/listing/campaigns/create — Admin creates a campaign
#[derive(Debug, Deserialize)]
pub struct CreateCampaignRequest {
    pub exchange_name: String,
    pub exchange_logo: Option<String>,
    pub target_usd: f64,
    pub tier: ListingTier,
    pub description: String,
    pub early_bird_slots: Option<u32>,
}

pub async fn campaign_create(
    auth: AuthenticatedWallet,
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateCampaignRequest>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    // Admin only
    if auth.address != q_types::BANK_MASTER_ACCOUNT {
        return Ok(Json(ApiResponse::error("Admin only".into())));
    }

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let campaign_id = format!("XLIST-{}-{}", req.exchange_name.to_uppercase().replace(' ', ""), now);

    let campaign = ListingCampaign {
        campaign_id: campaign_id.clone(),
        exchange_name: req.exchange_name,
        exchange_logo: req.exchange_logo,
        target_usd: req.target_usd,
        raised_usd: 0.0,
        contributor_count: 0,
        early_bird_slots: req.early_bird_slots.unwrap_or(50),
        early_bird_claimed: 0,
        status: CampaignStatus::Funding,
        tier: req.tier,
        description: req.description,
        perks: default_perks(),
        created_at: now,
        updated_at: now,
        funded_at: None,
        listed_at: None,
    };

    persist_campaign(&state, &campaign).await;
    state.listing_campaigns.write().await.push(campaign.clone());

    info!("🚀 [XLIST CROWDFUND] New campaign created: {} — ${:.0} target for {}",
        campaign_id, campaign.target_usd, campaign.exchange_name);

    Ok(Json(ApiResponse::success(serde_json::json!({
        "campaign": campaign,
    }))))
}

/// GET /api/v1/contracts/listing/campaigns/:id/my-perks — Get caller's perks
pub async fn campaign_my_perks(
    Path(campaign_id): Path<String>,
    auth: AuthenticatedWallet,
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let wallet_hex = hex::encode(auth.address);
    let campaigns = state.listing_campaigns.read().await;
    let campaign = campaigns.iter().find(|c| c.campaign_id == campaign_id);

    let Some(campaign) = campaign else {
        return Ok(Json(ApiResponse::error("Campaign not found".into())));
    };

    let contributions = load_contributions_for_campaign(&state, &campaign_id).await;
    let my_contribs: Vec<&Contribution> = contributions.iter().filter(|c| c.wallet == wallet_hex).collect();

    if my_contribs.is_empty() {
        return Ok(Json(ApiResponse::success(serde_json::json!({
            "contributed": false,
            "message": "You haven't contributed to this campaign yet",
        }))));
    }

    let total_usd: f64 = my_contribs.iter().map(|c| c.amount_usd).sum();
    let percent = if campaign.raised_usd > 0.0 { (total_usd / campaign.raised_usd) * 100.0 } else { 0.0 };
    let is_early_bird = my_contribs.iter().any(|c| c.is_early_bird);
    let (tier, mut multiplier) = compute_contributor_tier(percent, &campaign.perks);
    if is_early_bird { multiplier += 0.5; }

    Ok(Json(ApiResponse::success(serde_json::json!({
        "contributed": true,
        "total_usd": total_usd,
        "contribution_count": my_contribs.len(),
        "percent_of_total": percent,
        "tier": tier.name,
        "badge": tier.badge,
        "airdrop_multiplier": multiplier,
        "is_early_bird": is_early_bird,
        "governance_vote": tier.governance_vote,
        "perks": tier.perks,
        "campaign_status": campaign.status,
        "is_funded": campaign.status == CampaignStatus::Funded || campaign.status == CampaignStatus::Listed,
    }))))
}
