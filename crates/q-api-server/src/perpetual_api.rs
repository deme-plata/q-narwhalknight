//! Perpetual Futures API - Decentralized Derivatives Trading
//!
//! v2.5.0-beta: Perpetual contracts with up to 10x leverage
//! v2.7.8-beta: Fixed to use QUGUSD balance instead of QUG for collateral
//!
//! Features:
//! - Long/Short positions with configurable leverage (1-10x)
//! - Mark price from oracle for fair liquidations
//! - Funding rate mechanism to anchor price to spot
//! - Automatic liquidation engine
//! - Insurance fund for bad debt coverage

use anyhow::Result;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::Json,
    routing::{delete, get, post, put},
    Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

use crate::handlers::parse_wallet_address;
use crate::AppState;
use q_storage::BalanceStorage;

// ============================================================================
// v2.7.8-beta: QUGUSD Balance Helper Functions
// ============================================================================

/// QUGUSD address constant (used for token_balances lookup)
const QUGUSD_ADDR: [u8; 32] = [0x51, 0x55, 0x47, 0x55, 0x53, 0x44, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

/// Get total QUGUSD balance for a wallet (minted + swapped)
/// Perpetuals use QUGUSD as collateral, NOT QUG
/// Get QUGUSD balance for a wallet (minted + swapped)
/// v2.7.9-beta: Returns u128 for larger token supplies
pub async fn get_qugusd_balance(state: &Arc<AppState>, wallet_addr: &[u8; 32]) -> u128 {
    // Source 1: Minted QUGUSD from collateral vault (u64, cast to u128)
    let minted: u128 = {
        let vault = state.collateral_vault.read().await;
        vault.minted_qugusd.get(wallet_addr).copied().unwrap_or(0) as u128
    };

    // Source 2: QUGUSD from swaps/transfers (in token_balances, now u128)
    let swapped: u128 = {
        let token_balances = state.token_balances.read().await;
        token_balances.get(&(*wallet_addr, QUGUSD_ADDR)).copied().unwrap_or(0)
    };

    minted + swapped
}

/// Deduct QUGUSD from wallet (minted first, then swapped)
/// Returns true if successful, false if insufficient balance
/// v3.0.4: Migrated to u128 for 24-decimal precision
pub async fn deduct_qugusd(state: &Arc<AppState>, wallet_addr: &[u8; 32], amount: u128) -> bool {
    let total = get_qugusd_balance(state, wallet_addr).await;
    if total < amount {
        return false;
    }

    let mut remaining: u128 = amount;

    // Deduct from minted first (now u128)
    {
        let mut vault = state.collateral_vault.write().await;
        if let Some(minted) = vault.minted_qugusd.get_mut(wallet_addr) {
            let deduct = remaining.min(*minted);
            *minted = minted.saturating_sub(deduct);
            remaining = remaining.saturating_sub(deduct);
        }
    }

    // Then from swapped (token_balances is u128)
    if remaining > 0 {
        let mut token_balances = state.token_balances.write().await;
        if let Some(swapped) = token_balances.get_mut(&(*wallet_addr, QUGUSD_ADDR)) {
            *swapped = swapped.saturating_sub(remaining);
        }
    }

    true
}

/// Add QUGUSD to wallet (goes to minted pool for simplicity)
/// v3.0.4: Migrated to u128 for 24-decimal precision
pub async fn add_qugusd(state: &Arc<AppState>, wallet_addr: &[u8; 32], amount: u128) {
    let mut vault = state.collateral_vault.write().await;
    *vault.minted_qugusd.entry(*wallet_addr).or_insert(0) += amount;
}

// ============================================================================
// Constants
// ============================================================================

/// Maximum allowed leverage
pub const MAX_LEVERAGE: u8 = 10;

/// Minimum leverage
pub const MIN_LEVERAGE: u8 = 1;

/// Initial margin requirement (1/leverage)
pub const INITIAL_MARGIN_RATE: f64 = 0.08; // v8.6.0: 8% allows 12.5x leverage (was 10% / 10x)

/// Maintenance margin - liquidation threshold
pub const MAINTENANCE_MARGIN_RATE: f64 = 0.065; // v8.6.0: 6.5% safer liquidation threshold (was 5%)

/// Taker fee rate
pub const TAKER_FEE_RATE: f64 = 0.002; // 0.20% — v8.6.0: Increased for higher protocol revenue

/// Maker fee rate
pub const MAKER_FEE_RATE: f64 = 0.001; // 0.10% — v8.6.0: Increased for higher protocol revenue

/// Funding interval in seconds (8 hours)
pub const FUNDING_INTERVAL_SECS: u64 = 8 * 60 * 60;

/// Maximum funding rate per interval
pub const MAX_FUNDING_RATE: f64 = 0.003; // v8.6.0: 0.3% per 8h (was 0.1%), 3x for stronger spot anchoring

/// Liquidator reward percentage
pub const LIQUIDATOR_REWARD_RATE: f64 = 0.08; // 8% of remaining margin — v8.6.0: Increased for higher protocol revenue

/// Minimum position size (in base units with 8 decimals)
pub const MIN_POSITION_SIZE: u64 = 1_000_000; // 0.01 tokens

/// Minimum collateral (in quote units with 8 decimals)
pub const MIN_COLLATERAL: u64 = 10_000_000; // 0.1 QUGUSD

// ============================================================================
// Phase 5: Insurance Fund & Risk Controls Constants
// ============================================================================

/// Percentage of trading fees that go to insurance fund
pub const INSURANCE_FUND_FEE_RATE: f64 = 0.35; // 35% of fees go to insurance — v8.6.0: Increased for higher protocol revenue

/// Maximum position size per user (in quote value with 8 decimals)
pub const MAX_POSITION_SIZE_USD: u64 = 100_000_00_000_000; // $100,000 max position

/// Maximum total open interest (long + short) per market
pub const MAX_OPEN_INTEREST_USD: u64 = 10_000_000_00_000_000; // $10M total OI

/// Maximum number of open positions per user
pub const MAX_POSITIONS_PER_USER: usize = 10;

/// ADL threshold - when insurance fund drops below this % of open interest
pub const ADL_THRESHOLD_RATE: f64 = 0.025; // v8.6.0: 2.5% of OI (was 1%), earlier ADL trigger for safety

/// ADL priority score decay factor
pub const ADL_DECAY_FACTOR: f64 = 0.95;

// ============================================================================
// Data Types
// ============================================================================

/// Position side - Long profits when price goes up, Short profits when price goes down
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PositionSide {
    Long,
    Short,
}

impl PositionSide {
    pub fn opposite(&self) -> Self {
        match self {
            PositionSide::Long => PositionSide::Short,
            PositionSide::Short => PositionSide::Long,
        }
    }
}

/// Position status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PositionStatus {
    Open,
    Closed,
    Liquidated,
}

/// Order side for the order book
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OrderSide {
    Buy,
    Sell,
}

/// Order type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OrderType {
    Market,
    Limit,
    StopLoss,
    TakeProfit,
}

/// Order status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OrderStatus {
    Open,
    Filled,
    PartiallyFilled,
    Cancelled,
}

/// Perpetual position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerpPosition {
    pub id: String,
    pub wallet_address: String,
    pub market: String,
    pub side: PositionSide,
    pub size: u64,                    // Position size in base units (8 decimals)
    pub entry_price: u64,             // Average entry price (8 decimals)
    pub leverage: u8,                 // 1-10x
    pub collateral: u64,              // Margin deposited (8 decimals)
    pub liquidation_price: u64,       // Price at which position gets liquidated
    pub unrealized_pnl: i64,          // Current unrealized P&L
    pub realized_pnl: i64,            // Realized P&L from partial closes
    pub funding_payments: i64,        // Accumulated funding payments
    pub status: PositionStatus,
    pub opened_at: i64,
    pub closed_at: Option<i64>,
    pub close_price: Option<u64>,
}

/// Perpetual order
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerpOrder {
    pub id: String,
    pub wallet_address: String,
    pub market: String,
    pub side: OrderSide,
    pub order_type: OrderType,
    pub size: u64,                    // Order size in base units
    pub price: Option<u64>,           // Limit price (None for market orders)
    pub filled_size: u64,             // Amount filled
    pub leverage: u8,
    pub reduce_only: bool,            // Only reduce existing position
    pub status: OrderStatus,
    pub created_at: i64,
    pub updated_at: i64,
}

/// Perpetual market configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerpMarket {
    pub symbol: String,               // e.g., "QUG-PERP"
    pub base_token: String,           // e.g., "QUG"
    pub quote_token: String,          // e.g., "QUGUSD"
    pub max_leverage: u8,
    pub maintenance_margin: f64,
    pub initial_margin: f64,
    pub taker_fee: f64,
    pub maker_fee: f64,
    pub funding_interval: u64,
    pub max_funding_rate: f64,
    pub mark_price: u64,              // Current mark price from oracle
    pub index_price: u64,             // Spot price from oracle
    pub last_funding_time: i64,
    pub funding_rate: f64,            // Current funding rate
    pub open_interest_long: u64,      // Total long positions
    pub open_interest_short: u64,     // Total short positions
    pub volume_24h: u64,
    pub insurance_fund: u64,
    pub enabled: bool,
}

/// Trade record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerpTrade {
    pub id: String,
    pub market: String,
    pub wallet_address: String,
    pub position_id: String,
    pub side: PositionSide,
    pub size: u64,
    pub price: u64,
    pub fee: u64,
    pub pnl: i64,
    pub is_liquidation: bool,
    pub timestamp: i64,
}

/// Funding payment record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundingPayment {
    pub id: String,
    pub market: String,
    pub wallet_address: String,
    pub position_id: String,
    pub funding_rate: f64,
    pub payment: i64,                 // Positive = received, Negative = paid
    pub position_size: u64,
    pub timestamp: i64,
}

/// Liquidation record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Liquidation {
    pub id: String,
    pub market: String,
    pub wallet_address: String,
    pub position_id: String,
    pub side: PositionSide,
    pub size: u64,
    pub entry_price: u64,
    pub liquidation_price: u64,
    pub collateral_lost: u64,
    pub insurance_fund_contribution: i64,
    pub liquidator: Option<String>,
    pub liquidator_reward: u64,
    pub timestamp: i64,
}

// ============================================================================
// Phase 2: Order Book Data Structures
// ============================================================================

/// Time-in-force options for orders
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimeInForce {
    /// Good-til-cancelled (default) - remains on book until filled or cancelled
    GTC,
    /// Immediate-or-cancel - fill what you can, cancel the rest
    IOC,
    /// Fill-or-kill - fill entirely or cancel entirely
    FOK,
    /// Post-only - only add to book, cancel if would match immediately
    PostOnly,
}

impl Default for TimeInForce {
    fn default() -> Self {
        TimeInForce::GTC
    }
}

/// A single price level in the order book
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceLevel {
    pub price: u64,           // Price in quote units (8 decimals)
    pub size: u64,            // Total size at this price
    pub order_count: u32,     // Number of orders at this level
}

/// Order book for a perpetual market
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    pub market: String,
    pub bids: Vec<PriceLevel>,    // Sorted descending (best bid first)
    pub asks: Vec<PriceLevel>,    // Sorted ascending (best ask first)
    pub last_update: i64,
    pub sequence: u64,            // For detecting missed updates
}

impl OrderBook {
    pub fn new(market: &str) -> Self {
        Self {
            market: market.to_string(),
            bids: Vec::new(),
            asks: Vec::new(),
            last_update: chrono::Utc::now().timestamp_millis(),
            sequence: 0,
        }
    }

    /// Get best bid price
    pub fn best_bid(&self) -> Option<u64> {
        self.bids.first().map(|l| l.price)
    }

    /// Get best ask price
    pub fn best_ask(&self) -> Option<u64> {
        self.asks.first().map(|l| l.price)
    }

    /// Get spread
    pub fn spread(&self) -> Option<u64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(ask.saturating_sub(bid)),
            _ => None,
        }
    }

    /// Get mid price
    pub fn mid_price(&self) -> Option<u64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid + ask) / 2),
            _ => None,
        }
    }
}

/// Order fill result from matching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderFill {
    pub order_id: String,
    pub fill_price: u64,
    pub fill_size: u64,
    pub is_maker: bool,
    pub fee: u64,
    pub timestamp: i64,
}

/// Result of order placement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaceOrderResult {
    pub order_id: String,
    pub status: OrderStatus,
    pub filled_size: u64,
    pub remaining_size: u64,
    pub average_fill_price: Option<u64>,
    pub fills: Vec<OrderFill>,
    pub message: String,
}

/// Limit order with full details for order book
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LimitOrder {
    pub id: String,
    pub wallet_address: String,
    pub market: String,
    pub side: OrderSide,
    pub price: u64,
    pub original_size: u64,
    pub remaining_size: u64,
    pub filled_size: u64,
    pub leverage: u8,
    pub time_in_force: TimeInForce,
    pub reduce_only: bool,
    pub post_only: bool,
    pub status: OrderStatus,
    pub created_at: i64,
    pub updated_at: i64,
}

/// Request to place a limit order (Phase 2)
#[derive(Debug, Deserialize)]
pub struct PlaceLimitOrderRequest {
    pub wallet_address: String,
    pub market: String,
    pub side: OrderSide,
    pub price: u64,
    pub size: u64,
    pub leverage: u8,
    #[serde(default)]
    pub time_in_force: TimeInForce,
    #[serde(default)]
    pub reduce_only: bool,
    #[serde(default)]
    pub post_only: bool,
}

// ============================================================================
// Phase 2: Order Matching Engine
// ============================================================================

/// Order matching engine for limit orders
pub struct OrderMatchingEngine;

impl OrderMatchingEngine {
    /// Match an incoming order against the order book
    /// Returns (fills, remaining_size)
    pub fn match_order(
        order_book: &mut OrderBook,
        orders: &mut HashMap<String, LimitOrder>,
        incoming_side: OrderSide,
        incoming_price: u64,
        incoming_size: u64,
        time_in_force: TimeInForce,
        post_only: bool,
    ) -> (Vec<OrderFill>, u64) {
        let mut fills = Vec::new();
        let mut remaining = incoming_size;
        let now = chrono::Utc::now().timestamp_millis();

        // Post-only orders cannot cross the spread
        if post_only {
            let would_cross = match incoming_side {
                OrderSide::Buy => order_book.best_ask().map(|a| incoming_price >= a).unwrap_or(false),
                OrderSide::Sell => order_book.best_bid().map(|b| incoming_price <= b).unwrap_or(false),
            };
            if would_cross {
                // Post-only order would cross - return no fills
                return (fills, 0); // Signal to cancel by returning 0 remaining
            }
        }

        // Get the opposite side of the book to match against
        let opposite_levels = match incoming_side {
            OrderSide::Buy => &mut order_book.asks,  // Buyer matches against asks
            OrderSide::Sell => &mut order_book.bids, // Seller matches against bids
        };

        // Sort opposite levels properly (asks ascending, bids descending)
        match incoming_side {
            OrderSide::Buy => opposite_levels.sort_by(|a, b| a.price.cmp(&b.price)),
            OrderSide::Sell => opposite_levels.sort_by(|a, b| b.price.cmp(&a.price)),
        }

        // Match against price levels
        let mut levels_to_remove = Vec::new();

        for (idx, level) in opposite_levels.iter_mut().enumerate() {
            if remaining == 0 {
                break;
            }

            // Check if price is acceptable
            let price_ok = match incoming_side {
                OrderSide::Buy => incoming_price >= level.price,  // Buyer pays at most their limit
                OrderSide::Sell => incoming_price <= level.price, // Seller receives at least their limit
            };

            if !price_ok {
                break; // No more matchable levels
            }

            // Fill against this level
            let fill_size = remaining.min(level.size);
            let fill = OrderFill {
                order_id: format!("fill_{}", now),
                fill_price: level.price,
                fill_size,
                is_maker: false, // Incoming order is taker
                fee: calculate_fee(fill_size, level.price, true),
                timestamp: now,
            };
            fills.push(fill);

            remaining -= fill_size;
            level.size -= fill_size;
            level.order_count = level.order_count.saturating_sub(1);

            if level.size == 0 {
                levels_to_remove.push(idx);
            }
        }

        // Remove empty levels (in reverse order to preserve indices)
        for idx in levels_to_remove.into_iter().rev() {
            opposite_levels.remove(idx);
        }

        // Update order book metadata
        order_book.last_update = now;
        order_book.sequence += 1;

        // Handle time-in-force
        match time_in_force {
            TimeInForce::IOC => {
                // Immediate-or-cancel: Cancel remaining (don't add to book)
                remaining = 0;
            }
            TimeInForce::FOK => {
                // Fill-or-kill: If not fully filled, cancel entire order
                if remaining > 0 && remaining < incoming_size {
                    // Partial fill - rollback (in real impl, but we'll just cancel)
                    remaining = 0;
                }
            }
            TimeInForce::GTC | TimeInForce::PostOnly => {
                // Good-til-cancelled: Remaining goes to book (handled by caller)
            }
        }

        (fills, remaining)
    }

    /// Add a limit order to the order book
    pub fn add_to_book(
        order_book: &mut OrderBook,
        side: OrderSide,
        price: u64,
        size: u64,
    ) {
        let levels = match side {
            OrderSide::Buy => &mut order_book.bids,
            OrderSide::Sell => &mut order_book.asks,
        };

        // Find existing level or insert new one
        if let Some(level) = levels.iter_mut().find(|l| l.price == price) {
            level.size += size;
            level.order_count += 1;
        } else {
            levels.push(PriceLevel {
                price,
                size,
                order_count: 1,
            });

            // Re-sort: bids descending, asks ascending
            match side {
                OrderSide::Buy => levels.sort_by(|a, b| b.price.cmp(&a.price)),
                OrderSide::Sell => levels.sort_by(|a, b| a.price.cmp(&b.price)),
            }
        }

        order_book.last_update = chrono::Utc::now().timestamp_millis();
        order_book.sequence += 1;
    }

    /// Remove an order from the order book
    pub fn remove_from_book(
        order_book: &mut OrderBook,
        side: OrderSide,
        price: u64,
        size: u64,
    ) {
        let levels = match side {
            OrderSide::Buy => &mut order_book.bids,
            OrderSide::Sell => &mut order_book.asks,
        };

        if let Some(level) = levels.iter_mut().find(|l| l.price == price) {
            level.size = level.size.saturating_sub(size);
            level.order_count = level.order_count.saturating_sub(1);
        }

        // Remove empty levels
        levels.retain(|l| l.size > 0);

        order_book.last_update = chrono::Utc::now().timestamp_millis();
        order_book.sequence += 1;
    }
}

// ============================================================================
// Storage
// ============================================================================

/// Perpetual futures storage - in-memory with RocksDB persistence
#[derive(Debug)]
pub struct PerpStorage {
    pub positions: RwLock<HashMap<String, PerpPosition>>,
    pub orders: RwLock<HashMap<String, PerpOrder>>,
    pub markets: RwLock<HashMap<String, PerpMarket>>,
    pub trades: RwLock<Vec<PerpTrade>>,
    pub funding_payments: RwLock<Vec<FundingPayment>>,
    pub liquidations: RwLock<Vec<Liquidation>>,
    /// Order books indexed by market symbol
    pub order_books: RwLock<HashMap<String, OrderBook>>,
    /// Limit orders indexed by order ID (Phase 2)
    pub limit_orders: RwLock<HashMap<String, LimitOrder>>,
}

impl PerpStorage {
    pub fn new() -> Self {
        let mut markets = HashMap::new();

        // Initialize default markets
        markets.insert("QUG-PERP".to_string(), PerpMarket {
            symbol: "QUG-PERP".to_string(),
            base_token: "QUG".to_string(),
            quote_token: "QUGUSD".to_string(),
            max_leverage: MAX_LEVERAGE,
            maintenance_margin: MAINTENANCE_MARGIN_RATE,
            initial_margin: INITIAL_MARGIN_RATE,
            taker_fee: TAKER_FEE_RATE,
            maker_fee: MAKER_FEE_RATE,
            funding_interval: FUNDING_INTERVAL_SECS,
            max_funding_rate: MAX_FUNDING_RATE,
            mark_price: 100_000_000, // $1.00 initial
            index_price: 100_000_000,
            last_funding_time: chrono::Utc::now().timestamp(),
            funding_rate: 0.0,
            open_interest_long: 0,
            open_interest_short: 0,
            volume_24h: 0,
            insurance_fund: 0,
            enabled: true,
        });

        // Initialize order books for each market
        let mut order_books = HashMap::new();
        order_books.insert("QUG-PERP".to_string(), OrderBook::new("QUG-PERP"));

        Self {
            positions: RwLock::new(HashMap::new()),
            orders: RwLock::new(HashMap::new()),
            markets: RwLock::new(markets),
            trades: RwLock::new(Vec::new()),
            funding_payments: RwLock::new(Vec::new()),
            liquidations: RwLock::new(Vec::new()),
            order_books: RwLock::new(order_books),
            limit_orders: RwLock::new(HashMap::new()),
        }
    }

    /// Load perpetual data from RocksDB
    pub async fn load_from_storage(&self, storage: &q_storage::QStorage) -> Result<()> {
        // Load positions
        match storage.load_all_perp_positions().await {
            Ok(position_pairs) => {
                let mut positions = self.positions.write().await;
                for (pos_id, pos_bytes) in position_pairs {
                    if let Ok(pos) = serde_json::from_slice::<PerpPosition>(&pos_bytes) {
                        positions.insert(pos_id, pos);
                    }
                }
                info!("📈 [PERP] Loaded {} positions from RocksDB", positions.len());
            }
            Err(e) => {
                warn!("⚠️ [PERP] Could not load positions: {}", e);
            }
        }

        // Load orders
        match storage.load_all_perp_orders().await {
            Ok(order_pairs) => {
                let mut orders = self.orders.write().await;
                for (order_id, order_bytes) in order_pairs {
                    if let Ok(order) = serde_json::from_slice::<PerpOrder>(&order_bytes) {
                        if order.status == OrderStatus::Open || order.status == OrderStatus::PartiallyFilled {
                            orders.insert(order_id, order);
                        }
                    }
                }
                info!("📈 [PERP] Loaded {} open orders from RocksDB", orders.len());
            }
            Err(e) => {
                warn!("⚠️ [PERP] Could not load orders: {}", e);
            }
        }

        Ok(())
    }

    /// Save a position to RocksDB
    pub async fn save_position(&self, storage: &q_storage::QStorage, position: &PerpPosition) -> Result<()> {
        let pos_bytes = serde_json::to_vec(position)?;
        storage.save_perp_position(&position.id, &pos_bytes).await?;
        debug!("💾 [PERP] Saved position {} to RocksDB", position.id);
        Ok(())
    }

    /// Save an order to RocksDB
    pub async fn save_order(&self, storage: &q_storage::QStorage, order: &PerpOrder) -> Result<()> {
        let order_bytes = serde_json::to_vec(order)?;
        storage.save_perp_order(&order.id, &order_bytes).await?;
        debug!("💾 [PERP] Saved order {} to RocksDB", order.id);
        Ok(())
    }

    /// Save a trade to RocksDB
    pub async fn save_trade(&self, storage: &q_storage::QStorage, trade: &PerpTrade) -> Result<()> {
        let trade_bytes = serde_json::to_vec(trade)?;
        storage.save_perp_trade(&trade.id, trade.timestamp, &trade_bytes).await?;
        debug!("💾 [PERP] Saved trade {} to RocksDB", trade.id);
        Ok(())
    }
}

impl Default for PerpStorage {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Core Logic
// ============================================================================

/// Calculate liquidation price for a position
pub fn calculate_liquidation_price(
    entry_price: u64,
    leverage: u8,
    side: PositionSide,
) -> u64 {
    let margin_ratio = 1.0 / leverage as f64;
    let liq_threshold = margin_ratio - MAINTENANCE_MARGIN_RATE;

    match side {
        PositionSide::Long => {
            // Long liquidated when price drops
            // liq_price = entry * (1 - (initial_margin - maintenance_margin))
            ((entry_price as f64) * (1.0 - liq_threshold)) as u64
        }
        PositionSide::Short => {
            // Short liquidated when price rises
            ((entry_price as f64) * (1.0 + liq_threshold)) as u64
        }
    }
}

/// Calculate unrealized PnL for a position
pub fn calculate_unrealized_pnl(
    position: &PerpPosition,
    mark_price: u64,
) -> i64 {
    let size = position.size as f64 / 1e24;
    let entry = position.entry_price as f64 / 1e24;
    let mark = mark_price as f64 / 1e24;

    let pnl = match position.side {
        PositionSide::Long => (mark - entry) * size,
        PositionSide::Short => (entry - mark) * size,
    };

    (pnl * 1e24) as i64
}

/// Calculate margin ratio (equity / position value)
pub fn calculate_margin_ratio(
    position: &PerpPosition,
    mark_price: u64,
) -> f64 {
    let unrealized_pnl = calculate_unrealized_pnl(position, mark_price);
    let equity = position.collateral as i64 + unrealized_pnl + position.funding_payments;

    if equity <= 0 {
        return 0.0;
    }

    let position_value = (position.size as f64 / 1e24) * (mark_price as f64 / 1e24);

    (equity as f64 / 1e24) / position_value
}

/// Check if a position should be liquidated
pub fn should_liquidate(position: &PerpPosition, mark_price: u64) -> bool {
    let margin_ratio = calculate_margin_ratio(position, mark_price);
    margin_ratio < MAINTENANCE_MARGIN_RATE
}

/// Calculate required collateral for a position
pub fn calculate_required_collateral(
    size: u64,
    price: u64,
    leverage: u8,
) -> u64 {
    let position_value = (size as f64 / 1e24) * (price as f64 / 1e24);
    let required = position_value / leverage as f64;
    (required * 1e24) as u64
}

/// Calculate trading fee
pub fn calculate_fee(size: u64, price: u64, is_taker: bool) -> u64 {
    let notional = (size as f64 / 1e24) * (price as f64 / 1e24);
    let fee_rate = if is_taker { TAKER_FEE_RATE } else { MAKER_FEE_RATE };
    (notional * fee_rate * 1e24) as u64
}

/// Calculate funding payment for a position
pub fn calculate_funding_payment(
    position: &PerpPosition,
    funding_rate: f64,
    mark_price: u64,
) -> i64 {
    let position_value = (position.size as f64 / 1e24) * (mark_price as f64 / 1e24);
    let payment = position_value * funding_rate;

    // Longs pay when funding is positive, receive when negative
    // Shorts receive when funding is positive, pay when negative
    let signed_payment = match position.side {
        PositionSide::Long => -payment,
        PositionSide::Short => payment,
    };

    (signed_payment * 1e24) as i64
}

/// Calculate position value in USD (with 8 decimals)
pub fn calculate_position_value_usd(size: u64, price: u64) -> u64 {
    let size_f = size as f64 / 1e24;
    let price_f = price as f64 / 1e24;
    (size_f * price_f * 1e24) as u64
}

/// Phase 5: Calculate ADL priority score for a position
/// Higher score = more profitable = liquidated first during ADL
pub fn calculate_adl_priority(position: &PerpPosition, mark_price: u64) -> f64 {
    let pnl = calculate_unrealized_pnl(position, mark_price);
    let pnl_ratio = pnl as f64 / position.collateral as f64;

    // Score based on: leverage * profit_ratio
    // More leveraged + more profitable = higher priority for ADL
    (position.leverage as f64) * pnl_ratio.max(0.0)
}

// ============================================================================
// API Request/Response Types
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct OpenPositionRequest {
    pub wallet_address: String,
    pub market: String,
    pub side: PositionSide,
    pub size: u64,              // Size in base units
    pub leverage: u8,           // 1-10
    pub collateral: u64,        // Margin to deposit
}

#[derive(Debug, Serialize)]
pub struct OpenPositionResponse {
    pub success: bool,
    pub position_id: Option<String>,
    pub entry_price: Option<u64>,
    pub liquidation_price: Option<u64>,
    pub fee: Option<u64>,
    pub message: String,
}

#[derive(Debug, Deserialize)]
pub struct ClosePositionRequest {
    pub wallet_address: String,
    pub size: Option<u64>,      // None = close entire position
}

#[derive(Debug, Serialize)]
pub struct ClosePositionResponse {
    pub success: bool,
    pub exit_price: Option<u64>,
    pub realized_pnl: Option<i64>,
    pub fee: Option<u64>,
    pub returned_collateral: Option<u64>,
    pub message: String,
}

#[derive(Debug, Deserialize)]
pub struct AddMarginRequest {
    pub wallet_address: String,
    pub amount: u64,
}

#[derive(Debug, Serialize)]
pub struct AddMarginResponse {
    pub success: bool,
    pub new_collateral: Option<u64>,
    pub new_liquidation_price: Option<u64>,
    pub message: String,
}

// v2.7.8-beta: Position editing - Remove Margin
#[derive(Debug, Deserialize)]
pub struct RemoveMarginRequest {
    pub wallet_address: String,
    pub amount: u64,
}

#[derive(Debug, Serialize)]
pub struct RemoveMarginResponse {
    pub success: bool,
    pub new_collateral: Option<u64>,
    pub new_liquidation_price: Option<u64>,
    pub returned_amount: Option<u64>,
    pub message: String,
}

// v2.7.8-beta: Position editing - Adjust Leverage
#[derive(Debug, Deserialize)]
pub struct AdjustLeverageRequest {
    pub wallet_address: String,
    pub new_leverage: u8,
}

#[derive(Debug, Serialize)]
pub struct AdjustLeverageResponse {
    pub success: bool,
    pub old_leverage: Option<u8>,
    pub new_leverage: Option<u8>,
    pub new_liquidation_price: Option<u64>,
    pub collateral_change: Option<i64>, // positive = need more, negative = can withdraw
    pub message: String,
}

#[derive(Debug, Deserialize)]
pub struct PlaceOrderRequest {
    pub wallet_address: String,
    pub market: String,
    pub side: OrderSide,
    pub order_type: OrderType,
    pub size: u64,
    pub price: Option<u64>,     // Required for limit orders
    pub leverage: u8,
    pub reduce_only: bool,
}

#[derive(Debug, Serialize)]
pub struct PlaceOrderResponse {
    pub success: bool,
    pub order_id: Option<String>,
    pub message: String,
}

#[derive(Debug, Serialize)]
pub struct MarketInfoResponse {
    pub market: PerpMarket,
    pub orderbook: OrderbookSnapshot,
}

#[derive(Debug, Serialize)]
pub struct OrderbookSnapshot {
    pub bids: Vec<(u64, u64)>,  // (price, size)
    pub asks: Vec<(u64, u64)>,
    pub timestamp: i64,
}

#[derive(Debug, Serialize)]
pub struct PositionsResponse {
    pub positions: Vec<PerpPosition>,
}

#[derive(Debug, Serialize)]
pub struct OrdersResponse {
    pub orders: Vec<PerpOrder>,
}

#[derive(Debug, Serialize)]
pub struct TradesResponse {
    pub trades: Vec<PerpTrade>,
}

#[derive(Debug, Serialize)]
pub struct FundingRateResponse {
    pub market: String,
    pub funding_rate: f64,
    pub next_funding_time: i64,
    pub mark_price: u64,
    pub index_price: u64,
}

#[derive(Debug, Serialize)]
pub struct AccountResponse {
    pub wallet_address: String,
    pub margin_balance: u64,
    pub available_balance: u64,
    pub unrealized_pnl: i64,
    pub positions_count: usize,
    pub open_orders_count: usize,
}

// ============================================================================
// API Handlers
// ============================================================================

/// List all perpetual markets
async fn list_markets(
    State(state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    let markets = if let Some(ref perp_storage) = state.perp_storage {
        let markets = perp_storage.markets.read().await;
        markets.values().cloned().collect::<Vec<_>>()
    } else {
        vec![]
    };

    Json(serde_json::json!({
        "success": true,
        "markets": markets
    }))
}

/// Get market info including orderbook
async fn get_market_info(
    State(state): State<Arc<AppState>>,
    Path(symbol): Path<String>,
) -> Json<serde_json::Value> {
    let perp_storage = match &state.perp_storage {
        Some(s) => s,
        None => return Json(serde_json::json!({
            "success": false,
            "message": "Perpetual trading not enabled"
        })),
    };

    let markets = perp_storage.markets.read().await;

    match markets.get(&symbol) {
        Some(market) => {
            // Build orderbook from open orders
            let orders = perp_storage.orders.read().await;
            let mut bids: Vec<(u64, u64)> = Vec::new();
            let mut asks: Vec<(u64, u64)> = Vec::new();

            for order in orders.values() {
                if order.market == symbol && order.status == OrderStatus::Open {
                    if let Some(price) = order.price {
                        let remaining = order.size - order.filled_size;
                        match order.side {
                            OrderSide::Buy => bids.push((price, remaining)),
                            OrderSide::Sell => asks.push((price, remaining)),
                        }
                    }
                }
            }

            // Sort bids descending, asks ascending
            bids.sort_by(|a, b| b.0.cmp(&a.0));
            asks.sort_by(|a, b| a.0.cmp(&b.0));

            Json(serde_json::json!({
                "success": true,
                "market": market,
                "orderbook": {
                    "bids": bids.into_iter().take(20).collect::<Vec<_>>(),
                    "asks": asks.into_iter().take(20).collect::<Vec<_>>(),
                    "timestamp": chrono::Utc::now().timestamp_millis()
                }
            }))
        }
        None => Json(serde_json::json!({
            "success": false,
            "message": "Market not found"
        })),
    }
}

/// Get funding rate info
async fn get_funding_rate(
    State(state): State<Arc<AppState>>,
    Path(symbol): Path<String>,
) -> Json<serde_json::Value> {
    let perp_storage = match &state.perp_storage {
        Some(s) => s,
        None => return Json(serde_json::json!({
            "success": false,
            "message": "Perpetual trading not enabled"
        })),
    };

    let markets = perp_storage.markets.read().await;

    match markets.get(&symbol) {
        Some(market) => {
            let next_funding = market.last_funding_time + market.funding_interval as i64;

            Json(serde_json::json!({
                "success": true,
                "market": symbol,
                "funding_rate": market.funding_rate,
                "next_funding_time": next_funding,
                "mark_price": market.mark_price,
                "index_price": market.index_price
            }))
        }
        None => Json(serde_json::json!({
            "success": false,
            "message": "Market not found"
        })),
    }
}

/// Get user positions
async fn get_positions(
    State(state): State<Arc<AppState>>,
    Path(wallet_address): Path<String>,
) -> Json<serde_json::Value> {
    let perp_storage = match &state.perp_storage {
        Some(s) => s,
        None => return Json(serde_json::json!({
            "success": false,
            "message": "Perpetual trading not enabled"
        })),
    };

    let positions = perp_storage.positions.read().await;
    let markets = perp_storage.markets.read().await;

    let user_positions: Vec<_> = positions.values()
        .filter(|p| p.wallet_address == wallet_address && p.status == PositionStatus::Open)
        .map(|p| {
            let mark_price = markets.get(&p.market)
                .map(|m| m.mark_price)
                .unwrap_or(p.entry_price);

            let mut pos = p.clone();
            pos.unrealized_pnl = calculate_unrealized_pnl(&pos, mark_price);
            pos
        })
        .collect();

    Json(serde_json::json!({
        "success": true,
        "positions": user_positions
    }))
}

/// Open a new position
async fn open_position(
    State(state): State<Arc<AppState>>,
    Json(req): Json<OpenPositionRequest>,
) -> Json<OpenPositionResponse> {
    info!("📈 [PERP] Opening position: {:?}", req);

    let perp_storage = match &state.perp_storage {
        Some(s) => s,
        None => return Json(OpenPositionResponse {
            success: false,
            position_id: None,
            entry_price: None,
            liquidation_price: None,
            fee: None,
            message: "Perpetual trading not enabled".to_string(),
        }),
    };

    // Validate leverage
    if req.leverage < MIN_LEVERAGE || req.leverage > MAX_LEVERAGE {
        return Json(OpenPositionResponse {
            success: false,
            position_id: None,
            entry_price: None,
            liquidation_price: None,
            fee: None,
            message: format!("Leverage must be between {} and {}", MIN_LEVERAGE, MAX_LEVERAGE),
        });
    }

    // Validate size
    if req.size < MIN_POSITION_SIZE {
        return Json(OpenPositionResponse {
            success: false,
            position_id: None,
            entry_price: None,
            liquidation_price: None,
            fee: None,
            message: format!("Minimum position size is {}", MIN_POSITION_SIZE as f64 / 1e24),
        });
    }

    // Phase 5: Check max positions per user
    {
        let positions = perp_storage.positions.read().await;
        let user_position_count = positions.values()
            .filter(|p| p.wallet_address == req.wallet_address && p.status == PositionStatus::Open)
            .count();

        if user_position_count >= MAX_POSITIONS_PER_USER {
            return Json(OpenPositionResponse {
                success: false,
                position_id: None,
                entry_price: None,
                liquidation_price: None,
                fee: None,
                message: format!("Maximum {} open positions per user", MAX_POSITIONS_PER_USER),
            });
        }
    }

    // Get market
    let markets = perp_storage.markets.read().await;
    let market = match markets.get(&req.market) {
        Some(m) if m.enabled => m.clone(),
        Some(_) => return Json(OpenPositionResponse {
            success: false,
            position_id: None,
            entry_price: None,
            liquidation_price: None,
            fee: None,
            message: "Market is disabled".to_string(),
        }),
        None => return Json(OpenPositionResponse {
            success: false,
            position_id: None,
            entry_price: None,
            liquidation_price: None,
            fee: None,
            message: "Market not found".to_string(),
        }),
    };
    drop(markets);

    // Use mark price as entry price (for market orders)
    let entry_price = market.mark_price;

    // Phase 5: Check position size limit
    let position_value = calculate_position_value_usd(req.size, entry_price);
    if position_value > MAX_POSITION_SIZE_USD {
        return Json(OpenPositionResponse {
            success: false,
            position_id: None,
            entry_price: None,
            liquidation_price: None,
            fee: None,
            message: format!(
                "Position size ${:.2} exceeds maximum ${:.2}",
                position_value as f64 / 1e24,
                MAX_POSITION_SIZE_USD as f64 / 1e24
            ),
        });
    }

    // Phase 5: Check open interest limit
    let total_oi = market.open_interest_long + market.open_interest_short;
    let total_oi_usd = calculate_position_value_usd(total_oi + req.size, entry_price);
    if total_oi_usd > MAX_OPEN_INTEREST_USD {
        return Json(OpenPositionResponse {
            success: false,
            position_id: None,
            entry_price: None,
            liquidation_price: None,
            fee: None,
            message: format!(
                "Market open interest limit reached (${:.2}M max)",
                MAX_OPEN_INTEREST_USD as f64 / 1e24 / 1_000_000.0
            ),
        });
    }

    // Calculate required collateral
    let required_collateral = calculate_required_collateral(req.size, entry_price, req.leverage);
    if req.collateral < required_collateral {
        return Json(OpenPositionResponse {
            success: false,
            position_id: None,
            entry_price: None,
            liquidation_price: None,
            fee: None,
            message: format!(
                "Insufficient collateral. Required: {:.4} QUGUSD",
                required_collateral as f64 / 1e24
            ),
        });
    }

    // v2.7.8-beta: Check QUGUSD balance (not QUG!) for perpetual collateral
    // Parse wallet address to bytes (supports both 20-byte ETH-style and 32-byte addresses)
    let wallet_bytes: [u8; 32] = match parse_wallet_address(&req.wallet_address) {
        Ok(bytes) => bytes,
        Err(e) => {
            error!("🚨 [PERP] Invalid wallet address: {} - {}", &req.wallet_address, e);
            return Json(OpenPositionResponse {
                success: false,
                position_id: None,
                entry_price: None,
                liquidation_price: None,
                fee: None,
                message: format!("Invalid wallet address: {}", e),
            });
        }
    };

    // Calculate fee first so we know total required
    let fee = calculate_fee(req.size, entry_price, true);
    let total_required = req.collateral + fee;

    let qugusd_balance = get_qugusd_balance(&state, &wallet_bytes).await;
    info!("💰 [PERP] Wallet {} QUGUSD balance: {:.4} (required: {:.4})",
        &req.wallet_address[..16.min(req.wallet_address.len())],
        qugusd_balance as f64 / 1e24,
        total_required as f64 / 1e24
    );

    if qugusd_balance < total_required as u128 {
        return Json(OpenPositionResponse {
            success: false,
            position_id: None,
            entry_price: None,
            liquidation_price: None,
            fee: None,
            message: format!(
                "Insufficient QUGUSD balance. Required: {:.4} (collateral: {:.4} + fee: {:.4}), Available: {:.4}",
                total_required as f64 / 1e24,
                req.collateral as f64 / 1e24,
                fee as f64 / 1e24,
                qugusd_balance as f64 / 1e24
            ),
        });
    }

    // Calculate liquidation price
    let liquidation_price = calculate_liquidation_price(entry_price, req.leverage, req.side);

    // Generate position ID
    let position_id = format!("perp_{}_{}", chrono::Utc::now().timestamp_millis(), &req.wallet_address[..8]);

    // Create position
    let position = PerpPosition {
        id: position_id.clone(),
        wallet_address: req.wallet_address.clone(),
        market: req.market.clone(),
        side: req.side,
        size: req.size,
        entry_price,
        leverage: req.leverage,
        collateral: req.collateral,
        liquidation_price,
        unrealized_pnl: 0,
        realized_pnl: 0,
        funding_payments: 0,
        status: PositionStatus::Open,
        opened_at: chrono::Utc::now().timestamp(),
        closed_at: None,
        close_price: None,
    };

    // v2.7.8-beta: Deduct collateral + fee from QUGUSD balance (not QUG!)
    // v3.0.4: Cast to u128 for deduct_qugusd
    if !deduct_qugusd(&state, &wallet_bytes, total_required as u128).await {
        return Json(OpenPositionResponse {
            success: false,
            position_id: None,
            entry_price: None,
            liquidation_price: None,
            fee: None,
            message: "Failed to deduct QUGUSD collateral".to_string(),
        });
    }
    info!("💸 [PERP] Deducted {:.4} QUGUSD from wallet {} for position {}",
        total_required as f64 / 1e24,
        &req.wallet_address[..16.min(req.wallet_address.len())],
        position_id
    );

    // Store position
    {
        let mut positions = perp_storage.positions.write().await;
        positions.insert(position_id.clone(), position.clone());
    }

    // Update open interest and insurance fund (Phase 5)
    {
        let mut markets = perp_storage.markets.write().await;
        if let Some(m) = markets.get_mut(&req.market) {
            match req.side {
                PositionSide::Long => m.open_interest_long += req.size,
                PositionSide::Short => m.open_interest_short += req.size,
            }
            // Phase 5: Add portion of fee to insurance fund
            let insurance_contribution = (fee as f64 * INSURANCE_FUND_FEE_RATE) as u64;
            m.insurance_fund += insurance_contribution;
            info!("🛡️ [PERP] Insurance fund contribution: {} (fund now: {})",
                insurance_contribution as f64 / 1e24,
                m.insurance_fund as f64 / 1e24);
        }
    }

    // Save to RocksDB
    if let Err(e) = perp_storage.save_position(&state.storage_engine, &position).await {
        warn!("⚠️ [PERP] Failed to persist position: {}", e);
    }

    // Record trade
    let trade = PerpTrade {
        id: format!("trade_{}_{}", chrono::Utc::now().timestamp_millis(), &req.wallet_address[..8]),
        market: req.market.clone(),
        wallet_address: req.wallet_address.clone(),
        position_id: position_id.clone(),
        side: req.side,
        size: req.size,
        price: entry_price,
        fee,
        pnl: 0,
        is_liquidation: false,
        timestamp: chrono::Utc::now().timestamp(),
    };

    {
        let mut trades = perp_storage.trades.write().await;
        trades.push(trade.clone());
    }

    if let Err(e) = perp_storage.save_trade(&state.storage_engine, &trade).await {
        warn!("⚠️ [PERP] Failed to persist trade: {}", e);
    }

    info!("✅ [PERP] Position opened: {} {} {} @ {} (liq: {})",
        position_id,
        format!("{:?}", req.side),
        req.size as f64 / 1e24,
        entry_price as f64 / 1e24,
        liquidation_price as f64 / 1e24
    );

    Json(OpenPositionResponse {
        success: true,
        position_id: Some(position_id),
        entry_price: Some(entry_price),
        liquidation_price: Some(liquidation_price),
        fee: Some(fee),
        message: "Position opened successfully".to_string(),
    })
}

/// Close a position
async fn close_position(
    State(state): State<Arc<AppState>>,
    Path(position_id): Path<String>,
    Json(req): Json<ClosePositionRequest>,
) -> Json<ClosePositionResponse> {
    info!("📉 [PERP] Closing position: {}", position_id);

    let perp_storage = match &state.perp_storage {
        Some(s) => s,
        None => return Json(ClosePositionResponse {
            success: false,
            exit_price: None,
            realized_pnl: None,
            fee: None,
            returned_collateral: None,
            message: "Perpetual trading not enabled".to_string(),
        }),
    };

    // Get position
    let mut positions = perp_storage.positions.write().await;
    let position = match positions.get_mut(&position_id) {
        Some(p) if p.wallet_address == req.wallet_address && p.status == PositionStatus::Open => p,
        Some(_) => return Json(ClosePositionResponse {
            success: false,
            exit_price: None,
            realized_pnl: None,
            fee: None,
            returned_collateral: None,
            message: "Position not found or not owned by you".to_string(),
        }),
        None => return Json(ClosePositionResponse {
            success: false,
            exit_price: None,
            realized_pnl: None,
            fee: None,
            returned_collateral: None,
            message: "Position not found".to_string(),
        }),
    };

    // Get current mark price
    let markets = perp_storage.markets.read().await;
    let exit_price = markets.get(&position.market)
        .map(|m| m.mark_price)
        .unwrap_or(position.entry_price);
    drop(markets);

    // Determine close size
    let close_size = req.size.unwrap_or(position.size).min(position.size);
    let is_full_close = close_size == position.size;

    // Calculate PnL
    let pnl_per_unit = match position.side {
        PositionSide::Long => exit_price as i64 - position.entry_price as i64,
        PositionSide::Short => position.entry_price as i64 - exit_price as i64,
    };
    let realized_pnl = (pnl_per_unit as f64 * (close_size as f64 / 1e24)) as i64;

    // Calculate fee
    let fee = calculate_fee(close_size, exit_price, true);

    // Calculate collateral to return
    let collateral_ratio = close_size as f64 / position.size as f64;
    let collateral_to_return = (position.collateral as f64 * collateral_ratio) as u64;
    let funding_portion = (position.funding_payments as f64 * collateral_ratio) as i64;

    // Total to return: collateral + PnL + funding - fee
    let total_return = (collateral_to_return as i64 + realized_pnl + funding_portion - fee as i64).max(0) as u64;

    // Update or close position
    if is_full_close {
        position.status = PositionStatus::Closed;
        position.closed_at = Some(chrono::Utc::now().timestamp());
        position.close_price = Some(exit_price);
        position.realized_pnl = realized_pnl;
    } else {
        // Partial close
        position.size -= close_size;
        position.collateral -= collateral_to_return;
        position.realized_pnl += realized_pnl;
        position.funding_payments -= funding_portion;
        // Recalculate liquidation price for remaining position
        position.liquidation_price = calculate_liquidation_price(
            position.entry_price,
            position.leverage,
            position.side
        );
    }

    let updated_position = position.clone();
    drop(positions);

    // Update open interest and insurance fund (Phase 5)
    {
        let mut markets = perp_storage.markets.write().await;
        if let Some(m) = markets.get_mut(&updated_position.market) {
            match updated_position.side {
                PositionSide::Long => m.open_interest_long = m.open_interest_long.saturating_sub(close_size),
                PositionSide::Short => m.open_interest_short = m.open_interest_short.saturating_sub(close_size),
            }
            // Phase 5: Add portion of close fee to insurance fund
            let insurance_contribution = (fee as f64 * INSURANCE_FUND_FEE_RATE) as u64;
            m.insurance_fund += insurance_contribution;
            info!("🛡️ [PERP] Insurance fund contribution: {} (fund now: {})",
                insurance_contribution as f64 / 1e24,
                m.insurance_fund as f64 / 1e24);
        }
    }

    // v2.7.8-beta: Return QUGUSD to user (not QUG!)
    // v3.0.4: Cast to u128 for add_qugusd
    let wallet_bytes = parse_wallet_address(&req.wallet_address).unwrap_or([0u8; 32]);
    add_qugusd(&state, &wallet_bytes, total_return as u128).await;
    info!("💰 [PERP] Returned {:.4} QUGUSD to wallet {}",
        total_return as f64 / 1e24,
        &req.wallet_address[..16.min(req.wallet_address.len())]
    );

    // Save to RocksDB
    if let Err(e) = perp_storage.save_position(&state.storage_engine, &updated_position).await {
        warn!("⚠️ [PERP] Failed to persist position update: {}", e);
    }

    // Record trade
    let trade = PerpTrade {
        id: format!("trade_{}_{}", chrono::Utc::now().timestamp_millis(), &req.wallet_address[..8]),
        market: updated_position.market.clone(),
        wallet_address: req.wallet_address.clone(),
        position_id: position_id.clone(),
        side: updated_position.side.opposite(),
        size: close_size,
        price: exit_price,
        fee,
        pnl: realized_pnl,
        is_liquidation: false,
        timestamp: chrono::Utc::now().timestamp(),
    };

    {
        let mut trades = perp_storage.trades.write().await;
        trades.push(trade.clone());
    }

    if let Err(e) = perp_storage.save_trade(&state.storage_engine, &trade).await {
        warn!("⚠️ [PERP] Failed to persist trade: {}", e);
    }

    info!("✅ [PERP] Position closed: {} @ {} PnL: {} (returned: {})",
        position_id,
        exit_price as f64 / 1e24,
        realized_pnl as f64 / 1e24,
        total_return as f64 / 1e24
    );

    Json(ClosePositionResponse {
        success: true,
        exit_price: Some(exit_price),
        realized_pnl: Some(realized_pnl),
        fee: Some(fee),
        returned_collateral: Some(total_return),
        message: if is_full_close { "Position closed".to_string() } else { "Position partially closed".to_string() },
    })
}

/// Add margin to a position
async fn add_margin(
    State(state): State<Arc<AppState>>,
    Path(position_id): Path<String>,
    Json(req): Json<AddMarginRequest>,
) -> Json<AddMarginResponse> {
    let perp_storage = match &state.perp_storage {
        Some(s) => s,
        None => return Json(AddMarginResponse {
            success: false,
            new_collateral: None,
            new_liquidation_price: None,
            message: "Perpetual trading not enabled".to_string(),
        }),
    };

    // v2.7.8-beta: Check QUGUSD balance for margin
    let wallet_bytes: [u8; 32] = match parse_wallet_address(&req.wallet_address) {
        Ok(bytes) => bytes,
        Err(e) => {
            return Json(AddMarginResponse {
                success: false,
                new_collateral: None,
                new_liquidation_price: None,
                message: format!("Invalid wallet address: {}", e),
            });
        }
    };

    let qugusd_balance = get_qugusd_balance(&state, &wallet_bytes).await;
    if qugusd_balance < req.amount as u128 {
        return Json(AddMarginResponse {
            success: false,
            new_collateral: None,
            new_liquidation_price: None,
            message: format!(
                "Insufficient QUGUSD balance. Required: {:.4}, Available: {:.4}",
                req.amount as f64 / 1e24,
                qugusd_balance as f64 / 1e24
            ),
        });
    }

    // Get and update position
    let mut positions = perp_storage.positions.write().await;
    let position = match positions.get_mut(&position_id) {
        Some(p) if p.wallet_address == req.wallet_address && p.status == PositionStatus::Open => p,
        _ => return Json(AddMarginResponse {
            success: false,
            new_collateral: None,
            new_liquidation_price: None,
            message: "Position not found".to_string(),
        }),
    };

    // Add margin
    position.collateral += req.amount;

    // Recalculate effective leverage and liquidation price
    let markets = perp_storage.markets.read().await;
    let mark_price = markets.get(&position.market).map(|m| m.mark_price).unwrap_or(position.entry_price);
    drop(markets);

    let position_value = (position.size as f64 / 1e24) * (mark_price as f64 / 1e24);
    let effective_leverage = (position_value / (position.collateral as f64 / 1e24)).ceil() as u8;
    position.leverage = effective_leverage.min(MAX_LEVERAGE);
    position.liquidation_price = calculate_liquidation_price(position.entry_price, position.leverage, position.side);

    let updated_position = position.clone();
    drop(positions);

    // v2.7.8-beta: Deduct QUGUSD from user balance
    // v3.0.4: Cast to u128 for deduct_qugusd
    if !deduct_qugusd(&state, &wallet_bytes, req.amount as u128).await {
        return Json(AddMarginResponse {
            success: false,
            new_collateral: None,
            new_liquidation_price: None,
            message: "Failed to deduct QUGUSD".to_string(),
        });
    }

    // Save to RocksDB
    if let Err(e) = perp_storage.save_position(&state.storage_engine, &updated_position).await {
        warn!("⚠️ [PERP] Failed to persist position: {}", e);
    }

    Json(AddMarginResponse {
        success: true,
        new_collateral: Some(updated_position.collateral),
        new_liquidation_price: Some(updated_position.liquidation_price),
        message: "Margin added successfully".to_string(),
    })
}

/// v2.7.8-beta: Remove margin from a position (withdraw excess collateral)
async fn remove_margin(
    State(state): State<Arc<AppState>>,
    Path(position_id): Path<String>,
    Json(req): Json<RemoveMarginRequest>,
) -> Json<RemoveMarginResponse> {
    let perp_storage = match &state.perp_storage {
        Some(s) => s,
        None => return Json(RemoveMarginResponse {
            success: false,
            new_collateral: None,
            new_liquidation_price: None,
            returned_amount: None,
            message: "Perpetual trading not enabled".to_string(),
        }),
    };

    // Get position
    let mut positions = perp_storage.positions.write().await;
    let position = match positions.get_mut(&position_id) {
        Some(p) if p.wallet_address == req.wallet_address && p.status == PositionStatus::Open => p,
        _ => return Json(RemoveMarginResponse {
            success: false,
            new_collateral: None,
            new_liquidation_price: None,
            returned_amount: None,
            message: "Position not found or not owned by you".to_string(),
        }),
    };

    // Get mark price
    let markets = perp_storage.markets.read().await;
    let mark_price = markets.get(&position.market).map(|m| m.mark_price).unwrap_or(position.entry_price);
    drop(markets);

    // Calculate minimum required collateral (must maintain margin ratio)
    let position_value = (position.size as f64 / 1e24) * (mark_price as f64 / 1e24);
    let min_collateral = (position_value / position.leverage as f64 * 1e24) as u64;

    // Add safety buffer (120% of minimum)
    let safe_min_collateral = (min_collateral as f64 * 1.2) as u64;

    if position.collateral <= safe_min_collateral {
        return Json(RemoveMarginResponse {
            success: false,
            new_collateral: None,
            new_liquidation_price: None,
            returned_amount: None,
            message: format!(
                "Cannot remove margin. Current: {:.4}, Minimum required: {:.4}",
                position.collateral as f64 / 1e24,
                safe_min_collateral as f64 / 1e24
            ),
        });
    }

    let max_removable = position.collateral - safe_min_collateral;
    let amount_to_remove = req.amount.min(max_removable);

    if amount_to_remove == 0 {
        return Json(RemoveMarginResponse {
            success: false,
            new_collateral: None,
            new_liquidation_price: None,
            returned_amount: None,
            message: "No margin available to remove".to_string(),
        });
    }

    // Remove margin
    position.collateral -= amount_to_remove;

    // Recalculate liquidation price
    position.liquidation_price = calculate_liquidation_price(position.entry_price, position.leverage, position.side);

    let updated_position = position.clone();
    drop(positions);

    // Return QUGUSD to user
    // v3.0.4: Cast to u128 for add_qugusd
    let wallet_bytes = parse_wallet_address(&req.wallet_address).unwrap_or([0u8; 32]);
    add_qugusd(&state, &wallet_bytes, amount_to_remove as u128).await;

    // Save to RocksDB
    if let Err(e) = perp_storage.save_position(&state.storage_engine, &updated_position).await {
        warn!("⚠️ [PERP] Failed to persist position: {}", e);
    }

    info!("💸 [PERP] Removed {:.4} QUGUSD margin from position {}",
        amount_to_remove as f64 / 1e24, position_id);

    Json(RemoveMarginResponse {
        success: true,
        new_collateral: Some(updated_position.collateral),
        new_liquidation_price: Some(updated_position.liquidation_price),
        returned_amount: Some(amount_to_remove),
        message: format!("Removed {:.4} QUGUSD margin", amount_to_remove as f64 / 1e24),
    })
}

/// v2.7.8-beta: Adjust leverage on a position
async fn adjust_leverage(
    State(state): State<Arc<AppState>>,
    Path(position_id): Path<String>,
    Json(req): Json<AdjustLeverageRequest>,
) -> Json<AdjustLeverageResponse> {
    let perp_storage = match &state.perp_storage {
        Some(s) => s,
        None => return Json(AdjustLeverageResponse {
            success: false,
            old_leverage: None,
            new_leverage: None,
            new_liquidation_price: None,
            collateral_change: None,
            message: "Perpetual trading not enabled".to_string(),
        }),
    };

    // Validate leverage
    if req.new_leverage < MIN_LEVERAGE || req.new_leverage > MAX_LEVERAGE {
        return Json(AdjustLeverageResponse {
            success: false,
            old_leverage: None,
            new_leverage: None,
            new_liquidation_price: None,
            collateral_change: None,
            message: format!("Leverage must be between {} and {}", MIN_LEVERAGE, MAX_LEVERAGE),
        });
    }

    // Get position
    let mut positions = perp_storage.positions.write().await;
    let position = match positions.get_mut(&position_id) {
        Some(p) if p.wallet_address == req.wallet_address && p.status == PositionStatus::Open => p,
        _ => return Json(AdjustLeverageResponse {
            success: false,
            old_leverage: None,
            new_leverage: None,
            new_liquidation_price: None,
            collateral_change: None,
            message: "Position not found or not owned by you".to_string(),
        }),
    };

    let old_leverage = position.leverage;

    if old_leverage == req.new_leverage {
        return Json(AdjustLeverageResponse {
            success: false,
            old_leverage: Some(old_leverage),
            new_leverage: Some(req.new_leverage),
            new_liquidation_price: None,
            collateral_change: None,
            message: "Leverage is already at the requested level".to_string(),
        });
    }

    // Get mark price
    let markets = perp_storage.markets.read().await;
    let mark_price = markets.get(&position.market).map(|m| m.mark_price).unwrap_or(position.entry_price);
    drop(markets);

    // Calculate required collateral for new leverage
    let position_value = (position.size as f64 / 1e24) * (mark_price as f64 / 1e24);
    let new_required_collateral = (position_value / req.new_leverage as f64 * 1e24) as u64;
    let collateral_change = new_required_collateral as i64 - position.collateral as i64;

    // Parse wallet address
    let wallet_bytes = parse_wallet_address(&req.wallet_address).unwrap_or([0u8; 32]);

    // If increasing leverage (lower collateral requirement), return excess
    // If decreasing leverage (higher collateral requirement), need more funds
    if collateral_change > 0 {
        // Need more collateral - check QUGUSD balance
        let qugusd_balance = get_qugusd_balance(&state, &wallet_bytes).await;
        if qugusd_balance < collateral_change as u128 {
            return Json(AdjustLeverageResponse {
                success: false,
                old_leverage: Some(old_leverage),
                new_leverage: Some(req.new_leverage),
                new_liquidation_price: None,
                collateral_change: Some(collateral_change),
                message: format!(
                    "Insufficient QUGUSD. Need {:.4} more, have {:.4}",
                    collateral_change as f64 / 1e24,
                    qugusd_balance as f64 / 1e24
                ),
            });
        }
        // Deduct additional collateral (v3.0.4: migrated to u128)
        deduct_qugusd(&state, &wallet_bytes, collateral_change as u128).await;
        position.collateral = new_required_collateral;
    } else if collateral_change < 0 {
        // Can return excess collateral (v3.0.4: migrated to u128)
        let excess = (-collateral_change) as u128;
        add_qugusd(&state, &wallet_bytes, excess).await;
        position.collateral = new_required_collateral;
    }

    // Update leverage and liquidation price
    position.leverage = req.new_leverage;
    position.liquidation_price = calculate_liquidation_price(position.entry_price, req.new_leverage, position.side);

    let updated_position = position.clone();
    drop(positions);

    // Save to RocksDB
    if let Err(e) = perp_storage.save_position(&state.storage_engine, &updated_position).await {
        warn!("⚠️ [PERP] Failed to persist position: {}", e);
    }

    info!("⚙️ [PERP] Adjusted leverage on position {}: {}x -> {}x (collateral change: {})",
        position_id, old_leverage, req.new_leverage, collateral_change);

    Json(AdjustLeverageResponse {
        success: true,
        old_leverage: Some(old_leverage),
        new_leverage: Some(req.new_leverage),
        new_liquidation_price: Some(updated_position.liquidation_price),
        collateral_change: Some(collateral_change),
        message: format!(
            "Leverage adjusted from {}x to {}x",
            old_leverage, req.new_leverage
        ),
    })
}

/// Get user's open orders
async fn get_orders(
    State(state): State<Arc<AppState>>,
    Path(wallet_address): Path<String>,
) -> Json<serde_json::Value> {
    let perp_storage = match &state.perp_storage {
        Some(s) => s,
        None => return Json(serde_json::json!({
            "success": false,
            "message": "Perpetual trading not enabled"
        })),
    };

    let orders = perp_storage.orders.read().await;
    let user_orders: Vec<_> = orders.values()
        .filter(|o| o.wallet_address == wallet_address &&
                   (o.status == OrderStatus::Open || o.status == OrderStatus::PartiallyFilled))
        .cloned()
        .collect();

    Json(serde_json::json!({
        "success": true,
        "orders": user_orders
    }))
}

/// Place a limit order
async fn place_order(
    State(state): State<Arc<AppState>>,
    Json(req): Json<PlaceOrderRequest>,
) -> Json<PlaceOrderResponse> {
    let perp_storage = match &state.perp_storage {
        Some(s) => s,
        None => return Json(PlaceOrderResponse {
            success: false,
            order_id: None,
            message: "Perpetual trading not enabled".to_string(),
        }),
    };

    // Validate
    if req.order_type == OrderType::Limit && req.price.is_none() {
        return Json(PlaceOrderResponse {
            success: false,
            order_id: None,
            message: "Limit orders require a price".to_string(),
        });
    }

    if req.leverage < MIN_LEVERAGE || req.leverage > MAX_LEVERAGE {
        return Json(PlaceOrderResponse {
            success: false,
            order_id: None,
            message: format!("Leverage must be between {} and {}", MIN_LEVERAGE, MAX_LEVERAGE),
        });
    }

    // Check market exists
    let markets = perp_storage.markets.read().await;
    if !markets.contains_key(&req.market) {
        return Json(PlaceOrderResponse {
            success: false,
            order_id: None,
            message: "Market not found".to_string(),
        });
    }
    drop(markets);

    let order_id = format!("order_{}_{}", chrono::Utc::now().timestamp_millis(), &req.wallet_address[..8]);

    let order = PerpOrder {
        id: order_id.clone(),
        wallet_address: req.wallet_address.clone(),
        market: req.market.clone(),
        side: req.side,
        order_type: req.order_type,
        size: req.size,
        price: req.price,
        filled_size: 0,
        leverage: req.leverage,
        reduce_only: req.reduce_only,
        status: OrderStatus::Open,
        created_at: chrono::Utc::now().timestamp(),
        updated_at: chrono::Utc::now().timestamp(),
    };

    // Store order
    {
        let mut orders = perp_storage.orders.write().await;
        orders.insert(order_id.clone(), order.clone());
    }

    // Save to RocksDB
    if let Err(e) = perp_storage.save_order(&state.storage_engine, &order).await {
        warn!("⚠️ [PERP] Failed to persist order: {}", e);
    }

    info!("📝 [PERP] Order placed: {} {:?} {} @ {:?}",
        order_id, req.side, req.size as f64 / 1e24, req.price.map(|p| p as f64 / 1e24));

    Json(PlaceOrderResponse {
        success: true,
        order_id: Some(order_id),
        message: "Order placed successfully".to_string(),
    })
}

/// Cancel an order
async fn cancel_order(
    State(state): State<Arc<AppState>>,
    Path((wallet_address, order_id)): Path<(String, String)>,
) -> Json<serde_json::Value> {
    let perp_storage = match &state.perp_storage {
        Some(s) => s,
        None => return Json(serde_json::json!({
            "success": false,
            "message": "Perpetual trading not enabled"
        })),
    };

    let mut orders = perp_storage.orders.write().await;

    match orders.get_mut(&order_id) {
        Some(order) if order.wallet_address == wallet_address => {
            order.status = OrderStatus::Cancelled;
            order.updated_at = chrono::Utc::now().timestamp();

            let updated_order = order.clone();
            drop(orders);

            if let Err(e) = perp_storage.save_order(&state.storage_engine, &updated_order).await {
                warn!("⚠️ [PERP] Failed to persist order cancellation: {}", e);
            }

            Json(serde_json::json!({
                "success": true,
                "message": "Order cancelled"
            }))
        }
        _ => Json(serde_json::json!({
            "success": false,
            "message": "Order not found"
        })),
    }
}

/// Get trade history
async fn get_trades(
    State(state): State<Arc<AppState>>,
    Path(wallet_address): Path<String>,
) -> Json<serde_json::Value> {
    let perp_storage = match &state.perp_storage {
        Some(s) => s,
        None => return Json(serde_json::json!({
            "success": false,
            "message": "Perpetual trading not enabled"
        })),
    };

    let trades = perp_storage.trades.read().await;
    let user_trades: Vec<_> = trades.iter()
        .filter(|t| t.wallet_address == wallet_address)
        .cloned()
        .collect();

    Json(serde_json::json!({
        "success": true,
        "trades": user_trades
    }))
}

/// Get account summary
async fn get_account(
    State(state): State<Arc<AppState>>,
    Path(wallet_address): Path<String>,
) -> Json<serde_json::Value> {
    let perp_storage = match &state.perp_storage {
        Some(s) => s,
        None => return Json(serde_json::json!({
            "success": false,
            "message": "Perpetual trading not enabled"
        })),
    };

    // v2.7.8-beta: Get QUGUSD balance (perpetual collateral), not QUG
    let wallet_bytes = parse_wallet_address(&wallet_address).unwrap_or([0u8; 32]);
    let balance = get_qugusd_balance(&state, &wallet_bytes).await;

    let positions = perp_storage.positions.read().await;
    let markets = perp_storage.markets.read().await;
    let orders = perp_storage.orders.read().await;

    let user_positions: Vec<_> = positions.values()
        .filter(|p| p.wallet_address == wallet_address && p.status == PositionStatus::Open)
        .collect();

    let total_collateral: u64 = user_positions.iter().map(|p| p.collateral).sum();
    let total_unrealized_pnl: i64 = user_positions.iter()
        .map(|p| {
            let mark_price = markets.get(&p.market).map(|m| m.mark_price).unwrap_or(p.entry_price);
            calculate_unrealized_pnl(p, mark_price)
        })
        .sum();

    let open_orders_count = orders.values()
        .filter(|o| o.wallet_address == wallet_address && o.status == OrderStatus::Open)
        .count();

    Json(serde_json::json!({
        "success": true,
        "wallet_address": wallet_address,
        "margin_balance": balance,
        "available_balance": balance.saturating_sub(total_collateral as u128),
        "total_collateral": total_collateral,
        "unrealized_pnl": total_unrealized_pnl,
        "positions_count": user_positions.len(),
        "open_orders_count": open_orders_count
    }))
}

// ============================================================================
// Phase 2: Limit Order & Order Book Handlers
// ============================================================================

/// Response for limit order placement
#[derive(Debug, Serialize)]
pub struct PlaceLimitOrderResponse {
    pub success: bool,
    pub order_id: Option<String>,
    pub status: String,
    pub filled_size: u64,
    pub remaining_size: u64,
    pub average_fill_price: Option<u64>,
    pub fills: Vec<OrderFill>,
    pub message: String,
}

/// Place a limit order with order book matching (Phase 2)
async fn place_limit_order(
    State(state): State<Arc<AppState>>,
    Json(req): Json<PlaceLimitOrderRequest>,
) -> Json<PlaceLimitOrderResponse> {
    info!("📊 [PERP] Placing limit order: {:?} {} {} @ {}",
        req.side, req.size as f64 / 1e24, req.market, req.price as f64 / 1e24);

    let perp_storage = match &state.perp_storage {
        Some(s) => s,
        None => return Json(PlaceLimitOrderResponse {
            success: false,
            order_id: None,
            status: "rejected".to_string(),
            filled_size: 0,
            remaining_size: 0,
            average_fill_price: None,
            fills: vec![],
            message: "Perpetual trading not enabled".to_string(),
        }),
    };

    // Validate leverage
    if req.leverage < MIN_LEVERAGE || req.leverage > MAX_LEVERAGE {
        return Json(PlaceLimitOrderResponse {
            success: false,
            order_id: None,
            status: "rejected".to_string(),
            filled_size: 0,
            remaining_size: 0,
            average_fill_price: None,
            fills: vec![],
            message: format!("Leverage must be between {} and {}", MIN_LEVERAGE, MAX_LEVERAGE),
        });
    }

    // Validate size
    if req.size < MIN_POSITION_SIZE {
        return Json(PlaceLimitOrderResponse {
            success: false,
            order_id: None,
            status: "rejected".to_string(),
            filled_size: 0,
            remaining_size: 0,
            average_fill_price: None,
            fills: vec![],
            message: format!("Minimum order size is {}", MIN_POSITION_SIZE as f64 / 1e24),
        });
    }

    // Check market exists
    {
        let markets = perp_storage.markets.read().await;
        if !markets.contains_key(&req.market) {
            return Json(PlaceLimitOrderResponse {
                success: false,
                order_id: None,
                status: "rejected".to_string(),
                filled_size: 0,
                remaining_size: 0,
                average_fill_price: None,
                fills: vec![],
                message: "Market not found".to_string(),
            });
        }
    }

    // Calculate required collateral
    let required_collateral = calculate_required_collateral(req.size, req.price, req.leverage);

    // v2.7.8-beta: Check QUGUSD balance for limit orders
    let wallet_bytes: [u8; 32] = match parse_wallet_address(&req.wallet_address) {
        Ok(bytes) => bytes,
        Err(e) => {
            return Json(PlaceLimitOrderResponse {
                success: false,
                order_id: None,
                status: "rejected".to_string(),
                filled_size: 0,
                remaining_size: 0,
                average_fill_price: None,
                fills: vec![],
                message: format!("Invalid wallet address: {}", e),
            });
        }
    };

    let qugusd_balance = get_qugusd_balance(&state, &wallet_bytes).await;
    if qugusd_balance < required_collateral as u128 {
        return Json(PlaceLimitOrderResponse {
            success: false,
            order_id: None,
            status: "rejected".to_string(),
            filled_size: 0,
            remaining_size: 0,
            average_fill_price: None,
            fills: vec![],
            message: format!(
                "Insufficient QUGUSD balance. Required: {:.4}, Available: {:.4}",
                required_collateral as f64 / 1e24,
                qugusd_balance as f64 / 1e24
            ),
        });
    }

    let now = chrono::Utc::now();
    let order_id = format!("limit_{}_{}", now.timestamp_millis(), &req.wallet_address[..8.min(req.wallet_address.len())]);

    // v2.7.8-beta: Lock QUGUSD collateral
    // v3.0.4: Cast to u128 for deduct_qugusd
    if !deduct_qugusd(&state, &wallet_bytes, required_collateral as u128).await {
        return Json(PlaceLimitOrderResponse {
            success: false,
            order_id: None,
            status: "rejected".to_string(),
            filled_size: 0,
            remaining_size: 0,
            average_fill_price: None,
            fills: vec![],
            message: "Failed to lock QUGUSD collateral".to_string(),
        });
    }

    // Get order book and attempt matching
    let (fills, remaining_size) = {
        let mut order_books = perp_storage.order_books.write().await;
        let mut limit_orders = perp_storage.limit_orders.write().await;

        let order_book = order_books.get_mut(&req.market).unwrap();

        OrderMatchingEngine::match_order(
            order_book,
            &mut limit_orders,
            req.side,
            req.price,
            req.size,
            req.time_in_force,
            req.post_only,
        )
    };

    // Calculate fill stats
    let filled_size = req.size - remaining_size;
    let average_fill_price = if !fills.is_empty() {
        let total_value: u64 = fills.iter().map(|f| f.fill_price * f.fill_size / 100_000_000).sum();
        let total_size: u64 = fills.iter().map(|f| f.fill_size).sum();
        if total_size > 0 {
            Some(total_value * 100_000_000 / total_size)
        } else {
            None
        }
    } else {
        None
    };

    // Determine order status
    let (status, status_str) = if remaining_size == 0 && filled_size == req.size {
        (OrderStatus::Filled, "filled")
    } else if filled_size > 0 && remaining_size > 0 {
        (OrderStatus::PartiallyFilled, "partially_filled")
    } else if req.post_only && fills.is_empty() && remaining_size == 0 {
        // Post-only order was cancelled because it would cross
        (OrderStatus::Cancelled, "cancelled_post_only")
    } else if remaining_size > 0 {
        (OrderStatus::Open, "open")
    } else {
        (OrderStatus::Cancelled, "cancelled")
    };

    // If order has remaining size and is GTC/PostOnly, add to order book
    if remaining_size > 0 && (req.time_in_force == TimeInForce::GTC || req.time_in_force == TimeInForce::PostOnly) {
        // Create limit order record
        let limit_order = LimitOrder {
            id: order_id.clone(),
            wallet_address: req.wallet_address.clone(),
            market: req.market.clone(),
            side: req.side,
            price: req.price,
            original_size: req.size,
            remaining_size,
            filled_size,
            leverage: req.leverage,
            time_in_force: req.time_in_force,
            reduce_only: req.reduce_only,
            post_only: req.post_only,
            status,
            created_at: now.timestamp(),
            updated_at: now.timestamp(),
        };

        // Add to order book
        {
            let mut order_books = perp_storage.order_books.write().await;
            if let Some(book) = order_books.get_mut(&req.market) {
                OrderMatchingEngine::add_to_book(book, req.side, req.price, remaining_size);
            }
        }

        // Store the limit order
        {
            let mut limit_orders = perp_storage.limit_orders.write().await;
            limit_orders.insert(order_id.clone(), limit_order);
        }

        info!("📊 [PERP] Limit order added to book: {} {} @ {} (remaining: {})",
            order_id, format!("{:?}", req.side), req.price as f64 / 1e24, remaining_size as f64 / 1e24);
    } else {
        // v2.7.8-beta: Return unused QUGUSD collateral for filled/cancelled orders
        // v3.0.4: Cast to u128 for add_qugusd
        let collateral_used = (filled_size as f64 / req.size as f64 * required_collateral as f64) as u64;
        let collateral_to_return = required_collateral.saturating_sub(collateral_used);
        if collateral_to_return > 0 {
            add_qugusd(&state, &wallet_bytes, collateral_to_return as u128).await;
        }
    }

    // If we had fills, create positions for the filled portion
    if filled_size > 0 {
        // Get mark price
        let entry_price = average_fill_price.unwrap_or(req.price);
        let liquidation_price = calculate_liquidation_price(entry_price, req.leverage,
            match req.side {
                OrderSide::Buy => PositionSide::Long,
                OrderSide::Sell => PositionSide::Short,
            }
        );

        let position = PerpPosition {
            id: format!("perp_{}_{}", now.timestamp_millis(), &req.wallet_address[..8.min(req.wallet_address.len())]),
            wallet_address: req.wallet_address.clone(),
            market: req.market.clone(),
            side: match req.side {
                OrderSide::Buy => PositionSide::Long,
                OrderSide::Sell => PositionSide::Short,
            },
            size: filled_size,
            entry_price,
            leverage: req.leverage,
            collateral: (filled_size as f64 / req.size as f64 * required_collateral as f64) as u64,
            liquidation_price,
            unrealized_pnl: 0,
            realized_pnl: 0,
            funding_payments: 0,
            status: PositionStatus::Open,
            opened_at: now.timestamp(),
            closed_at: None,
            close_price: None,
        };

        {
            let mut positions = perp_storage.positions.write().await;
            positions.insert(position.id.clone(), position);
        }

        info!("✅ [PERP] Position created from limit order fill: {} size {} @ {}",
            order_id, filled_size as f64 / 1e24, entry_price as f64 / 1e24);
    }

    info!("📊 [PERP] Limit order result: {} status={} filled={} remaining={}",
        order_id, status_str, filled_size as f64 / 1e24, remaining_size as f64 / 1e24);

    Json(PlaceLimitOrderResponse {
        success: true,
        order_id: Some(order_id),
        status: status_str.to_string(),
        filled_size,
        remaining_size,
        average_fill_price,
        fills,
        message: format!("Order {} - filled {} of {}", status_str, filled_size as f64 / 1e24, req.size as f64 / 1e24),
    })
}

/// Get order book depth for a market
async fn get_orderbook_depth(
    State(state): State<Arc<AppState>>,
    Path(symbol): Path<String>,
) -> Json<serde_json::Value> {
    let perp_storage = match &state.perp_storage {
        Some(s) => s,
        None => return Json(serde_json::json!({
            "success": false,
            "message": "Perpetual trading not enabled"
        })),
    };

    let order_books = perp_storage.order_books.read().await;

    match order_books.get(&symbol) {
        Some(book) => {
            Json(serde_json::json!({
                "success": true,
                "market": symbol,
                "bids": book.bids.iter().map(|l| {
                    serde_json::json!({
                        "price": l.price,
                        "size": l.size,
                        "order_count": l.order_count
                    })
                }).take(20).collect::<Vec<_>>(),
                "asks": book.asks.iter().map(|l| {
                    serde_json::json!({
                        "price": l.price,
                        "size": l.size,
                        "order_count": l.order_count
                    })
                }).take(20).collect::<Vec<_>>(),
                "best_bid": book.best_bid(),
                "best_ask": book.best_ask(),
                "spread": book.spread(),
                "mid_price": book.mid_price(),
                "last_update": book.last_update,
                "sequence": book.sequence
            }))
        }
        None => Json(serde_json::json!({
            "success": false,
            "message": "Market not found"
        })),
    }
}

/// Get user's limit orders
async fn get_limit_orders(
    State(state): State<Arc<AppState>>,
    Path(wallet_address): Path<String>,
) -> Json<serde_json::Value> {
    let perp_storage = match &state.perp_storage {
        Some(s) => s,
        None => return Json(serde_json::json!({
            "success": false,
            "message": "Perpetual trading not enabled"
        })),
    };

    let limit_orders = perp_storage.limit_orders.read().await;
    let user_orders: Vec<_> = limit_orders.values()
        .filter(|o| o.wallet_address == wallet_address && o.status == OrderStatus::Open)
        .cloned()
        .collect();

    Json(serde_json::json!({
        "success": true,
        "orders": user_orders
    }))
}

/// Cancel a limit order
async fn cancel_limit_order(
    State(state): State<Arc<AppState>>,
    Path((wallet_address, order_id)): Path<(String, String)>,
) -> Json<serde_json::Value> {
    let perp_storage = match &state.perp_storage {
        Some(s) => s,
        None => return Json(serde_json::json!({
            "success": false,
            "message": "Perpetual trading not enabled"
        })),
    };

    let mut limit_orders = perp_storage.limit_orders.write().await;

    match limit_orders.get_mut(&order_id) {
        Some(order) if order.wallet_address == wallet_address && order.status == OrderStatus::Open => {
            let remaining_size = order.remaining_size;
            let price = order.price;
            let side = order.side;
            let market = order.market.clone();

            order.status = OrderStatus::Cancelled;
            order.updated_at = chrono::Utc::now().timestamp();

            drop(limit_orders);

            // Remove from order book
            {
                let mut order_books = perp_storage.order_books.write().await;
                if let Some(book) = order_books.get_mut(&market) {
                    OrderMatchingEngine::remove_from_book(book, side, price, remaining_size);
                }
            }

            // v2.7.8-beta: Return locked QUGUSD collateral
            // v3.0.4: Cast to u128 for add_qugusd
            let collateral_to_return = remaining_size / 10; // Simplified calculation
            if collateral_to_return > 0 {
                let wallet_bytes = parse_wallet_address(&wallet_address).unwrap_or([0u8; 32]);
                add_qugusd(&state, &wallet_bytes, collateral_to_return as u128).await;
            }

            info!("🚫 [PERP] Limit order cancelled: {} returned {} QUGUSD collateral",
                order_id, collateral_to_return as f64 / 1e24);

            Json(serde_json::json!({
                "success": true,
                "message": "Order cancelled",
                "returned_collateral": collateral_to_return
            }))
        }
        Some(_) => Json(serde_json::json!({
            "success": false,
            "message": "Order not owned by you or already cancelled/filled"
        })),
        None => Json(serde_json::json!({
            "success": false,
            "message": "Order not found"
        })),
    }
}

// ============================================================================
// Router
// ============================================================================

/// Create the perpetual futures API router
pub fn create_perp_router() -> Router<Arc<AppState>> {
    Router::new()
        // Markets
        .route("/markets", get(list_markets))
        .route("/markets/:symbol", get(get_market_info))
        .route("/markets/:symbol/funding", get(get_funding_rate))

        // Positions
        .route("/positions/:wallet_address", get(get_positions))
        .route("/positions/open", post(open_position))
        .route("/positions/:position_id/close", post(close_position))
        .route("/positions/:position_id/margin", post(add_margin))
        // v2.7.8-beta: Position editing endpoints
        .route("/positions/:position_id/margin/remove", post(remove_margin))
        .route("/positions/:position_id/leverage", post(adjust_leverage))

        // Orders (market orders)
        .route("/orders/:wallet_address", get(get_orders))
        .route("/orders", post(place_order))
        .route("/orders/:wallet_address/:order_id", delete(cancel_order))

        // Phase 2: Order Book & Limit Orders
        .route("/orderbook/:symbol", get(get_orderbook_depth))
        .route("/limit-orders", post(place_limit_order))
        .route("/limit-orders/:wallet_address", get(get_limit_orders))
        .route("/limit-orders/:wallet_address/:order_id", delete(cancel_limit_order))

        // Account & History
        .route("/account/:wallet_address", get(get_account))
        .route("/trades/:wallet_address", get(get_trades))
}

// ============================================================================
// Background Tasks
// ============================================================================

/// Liquidation check loop - runs every second
pub async fn liquidation_loop(app_state: Arc<AppState>) {
    info!("🔥 [PERP] Starting liquidation engine...");

    let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(1));

    loop {
        interval.tick().await;

        let perp_storage = match &app_state.perp_storage {
            Some(s) => s,
            None => continue,
        };

        let markets = perp_storage.markets.read().await;
        let mut positions = perp_storage.positions.write().await;

        let mut liquidations_to_process = Vec::new();

        for (pos_id, position) in positions.iter() {
            if position.status != PositionStatus::Open {
                continue;
            }

            let mark_price = markets.get(&position.market)
                .map(|m| m.mark_price)
                .unwrap_or(position.entry_price);

            if should_liquidate(position, mark_price) {
                liquidations_to_process.push((pos_id.clone(), mark_price));
            }
        }

        drop(markets);

        // Process liquidations
        for (pos_id, mark_price) in liquidations_to_process {
            if let Some(position) = positions.get_mut(&pos_id) {
                warn!("🔥 [PERP] LIQUIDATING position {} at price {}",
                    pos_id, mark_price as f64 / 1e24);

                // Calculate loss
                let pnl = calculate_unrealized_pnl(position, mark_price);
                let remaining = (position.collateral as i64 + pnl + position.funding_payments).max(0) as u64;

                // Mark as liquidated
                position.status = PositionStatus::Liquidated;
                position.closed_at = Some(chrono::Utc::now().timestamp());
                position.close_price = Some(mark_price);
                position.realized_pnl = pnl;

                // v2.7.8-beta: Return any remaining QUGUSD collateral (after liquidator reward)
                // v3.0.4: Cast to u128 for add_qugusd
                let liquidator_reward = (remaining as f64 * LIQUIDATOR_REWARD_RATE) as u64;
                let user_return = remaining.saturating_sub(liquidator_reward);

                if user_return > 0 {
                    let wallet_bytes = parse_wallet_address(&position.wallet_address).unwrap_or([0u8; 32]);
                    add_qugusd(&app_state, &wallet_bytes, user_return as u128).await;
                }

                // Record liquidation
                let liq_record = Liquidation {
                    id: format!("liq_{}_{}", chrono::Utc::now().timestamp_millis(), &pos_id[..8]),
                    market: position.market.clone(),
                    wallet_address: position.wallet_address.clone(),
                    position_id: pos_id.clone(),
                    side: position.side,
                    size: position.size,
                    entry_price: position.entry_price,
                    liquidation_price: mark_price,
                    collateral_lost: position.collateral - user_return,
                    insurance_fund_contribution: if pnl < -(position.collateral as i64) {
                        pnl + position.collateral as i64
                    } else { 0 },
                    liquidator: None,
                    liquidator_reward,
                    timestamp: chrono::Utc::now().timestamp(),
                };

                {
                    let mut liquidations = perp_storage.liquidations.write().await;
                    liquidations.push(liq_record);
                }

                info!("🔥 [PERP] Liquidation complete: {} lost {} collateral, returned {}",
                    pos_id, position.collateral - user_return, user_return);
            }
        }
    }
}

/// Funding rate calculation loop - runs every 8 hours
pub async fn funding_loop(app_state: Arc<AppState>) {
    info!("💰 [PERP] Starting funding rate engine...");

    // Check every minute if funding is due
    let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(60));

    loop {
        interval.tick().await;

        let perp_storage = match &app_state.perp_storage {
            Some(s) => s,
            None => continue,
        };

        let now = chrono::Utc::now().timestamp();

        // Collect markets that need funding update
        let markets_to_update: Vec<(String, u64, f64)> = {
            let mut markets = perp_storage.markets.write().await;
            let mut updates = Vec::new();

            for (symbol, market) in markets.iter_mut() {
                let next_funding = market.last_funding_time + market.funding_interval as i64;

                if now >= next_funding {
                    // Calculate new funding rate based on mark vs index price
                    let premium = (market.mark_price as f64 - market.index_price as f64) / market.index_price as f64;
                    let new_rate = premium.clamp(-market.max_funding_rate, market.max_funding_rate);

                    market.funding_rate = new_rate;
                    market.last_funding_time = now;

                    info!("💰 [PERP] {} funding rate: {:.4}% (mark: {}, index: {})",
                        symbol,
                        new_rate * 100.0,
                        market.mark_price as f64 / 1e24,
                        market.index_price as f64 / 1e24
                    );

                    updates.push((symbol.clone(), market.mark_price, new_rate));
                }
            }
            updates
        };

        // Apply funding to positions (markets lock is now released)
        for (market_symbol, mark_price, funding_rate) in markets_to_update {
            let mut positions = perp_storage.positions.write().await;

            for position in positions.values_mut() {
                if position.market == market_symbol && position.status == PositionStatus::Open {
                    let payment = calculate_funding_payment(position, funding_rate, mark_price);
                    position.funding_payments += payment;

                    // Record funding payment
                    let funding_record = FundingPayment {
                        id: format!("fund_{}_{}", now, &position.id[..8]),
                        market: market_symbol.clone(),
                        wallet_address: position.wallet_address.clone(),
                        position_id: position.id.clone(),
                        funding_rate,
                        payment,
                        position_size: position.size,
                        timestamp: now,
                    };

                    let mut payments = perp_storage.funding_payments.write().await;
                    payments.push(funding_record);
                }
            }
        }
    }
}

/// Mark price update loop - fetches from DEX/oracle
pub async fn mark_price_loop(app_state: Arc<AppState>) {
    info!("📊 [PERP] Starting mark price oracle...");

    let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(5));

    loop {
        interval.tick().await;

        let perp_storage = match &app_state.perp_storage {
            Some(s) => s,
            None => continue,
        };

        // Get spot price from DEX liquidity pools
        let spot_price = {
            let pools = app_state.liquidity_pools.read().await;

            // Look for QUG-QUGUSD pool to get spot price
            let mut found_price = None;
            let qug_zero_hex = hex::encode([0u8; 32]);
            let qugusd_hex = hex::encode(q_types::QUGUSD_TOKEN_ADDRESS);
            for pool in pools.values() {
                // v4.0.4: Match both symbol format AND hex address format (P2P pools use hex)
                let t0 = pool.token0.to_uppercase();
                let t1 = pool.token1.to_uppercase();
                let t0_is_qug = t0 == "QUG" || t0 == "NATIVE-QUG" || pool.token0 == qug_zero_hex;
                let t1_is_qug = t1 == "QUG" || t1 == "NATIVE-QUG" || pool.token1 == qug_zero_hex;
                let t0_is_qugusd = t0 == "QUGUSD" || pool.token0 == qugusd_hex;
                let t1_is_qugusd = t1 == "QUGUSD" || pool.token1 == qugusd_hex;
                if (t0_is_qug && t1_is_qugusd) || (t0_is_qugusd && t1_is_qug) {
                    let price = if t0_is_qug {
                        // price = reserve1 / reserve0
                        ((pool.reserve1 as f64 / pool.reserve0 as f64) * 1e24) as u64
                    } else {
                        ((pool.reserve0 as f64 / pool.reserve1 as f64) * 1e24) as u64
                    };
                    found_price = Some(price);
                    break;
                }
            }
            found_price
        }; // pools lock released here

        // Update mark price if we found a spot price
        if let Some(spot_price) = spot_price {
            let mut markets = perp_storage.markets.write().await;
            if let Some(market) = markets.get_mut("QUG-PERP") {
                market.index_price = spot_price;
                // Mark price follows index with some smoothing
                market.mark_price = (market.mark_price * 9 + spot_price) / 10;
            }
        }
    }
}

/// Phase 5: Auto-Deleveraging (ADL) loop - runs when insurance fund is critically low
/// ADL closes the most profitable positions to cover bad debt from liquidations
pub async fn adl_loop(app_state: Arc<AppState>) {
    info!("⚡ [PERP] Starting Auto-Deleveraging (ADL) engine...");

    // Check every 10 seconds for ADL conditions
    let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(10));

    loop {
        interval.tick().await;

        let perp_storage = match &app_state.perp_storage {
            Some(s) => s,
            None => continue,
        };

        // Check each market for ADL conditions
        let markets = perp_storage.markets.read().await;

        for (symbol, market) in markets.iter() {
            // Calculate total open interest value
            let total_oi = market.open_interest_long + market.open_interest_short;
            let total_oi_value = calculate_position_value_usd(total_oi, market.mark_price);

            // Calculate insurance fund ratio (fund / OI)
            let insurance_ratio = if total_oi_value > 0 {
                market.insurance_fund as f64 / total_oi_value as f64
            } else {
                1.0 // No OI, fund is sufficient
            };

            // Trigger ADL if insurance fund ratio drops below threshold
            if insurance_ratio < ADL_THRESHOLD_RATE && total_oi > 0 {
                warn!("⚡ [PERP] ADL TRIGGERED for {} - Insurance ratio: {:.4}% (threshold: {:.4}%)",
                    symbol, insurance_ratio * 100.0, ADL_THRESHOLD_RATE * 100.0);

                // Determine which side needs deleveraging (opposite of net OI imbalance)
                // If longs > shorts, deleverage longs (they've been winning)
                let _deleverage_longs = market.open_interest_long > market.open_interest_short;
                let mark_price = market.mark_price;
                let market_symbol = symbol.clone();
                drop(markets);

                // Get positions and sort by ADL priority
                let mut positions = perp_storage.positions.write().await;

                // Collect positions with their ADL scores
                let mut scored_positions: Vec<(String, f64)> = positions
                    .iter()
                    .filter(|(_, p)| p.market == market_symbol && p.status == PositionStatus::Open)
                    .map(|(id, p)| {
                        let score = calculate_adl_priority(p, mark_price);
                        (id.clone(), score)
                    })
                    .collect();

                // Sort by score descending (highest priority first)
                scored_positions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                // ADL the top position (if any with positive score)
                if let Some((pos_id, score)) = scored_positions.first() {
                    if *score > 0.0 {
                        if let Some(position) = positions.get_mut(pos_id) {
                            let adl_size = position.size / 2; // Close 50% of position

                            if adl_size > 0 {
                                // Calculate partial close PnL
                                let pnl = calculate_unrealized_pnl(position, mark_price);
                                let partial_pnl = pnl * adl_size as i64 / position.size as i64;

                                // Reduce position
                                let old_size = position.size;
                                position.size -= adl_size;
                                position.collateral = (position.collateral as f64 * position.size as f64 / old_size as f64) as u64;
                                position.realized_pnl += partial_pnl;

                                // v2.7.8-beta: Return partial QUGUSD collateral + PnL to user
                                // v3.0.4: Cast to u128 for add_qugusd
                                let return_amount = (position.collateral as i64 + partial_pnl).max(0) as u64;
                                if return_amount > 0 {
                                    let wallet_bytes = parse_wallet_address(&position.wallet_address).unwrap_or([0u8; 32]);
                                    add_qugusd(&app_state, &wallet_bytes, return_amount as u128).await;
                                }

                                warn!("⚡ [PERP] ADL executed: position {} reduced by {:.4} QUGUSD (score: {:.4})",
                                    pos_id, adl_size as f64 / 1e24, score);
                            }
                        }
                    }
                }

                // Re-acquire markets lock to update OI
                let mut markets = perp_storage.markets.write().await;
                if let Some(m) = markets.get_mut(&market_symbol) {
                    // Recalculate OI from positions
                    let positions = perp_storage.positions.read().await;
                    let (long_oi, short_oi) = positions
                        .values()
                        .filter(|p| p.market == market_symbol && p.status == PositionStatus::Open)
                        .fold((0u64, 0u64), |(l, s), p| {
                            match p.side {
                                PositionSide::Long => (l + p.size, s),
                                PositionSide::Short => (l, s + p.size),
                            }
                        });
                    m.open_interest_long = long_oi;
                    m.open_interest_short = short_oi;
                }

                break; // Only process one market per tick
            }
        }
    }
}
