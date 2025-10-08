/// Trading bot types and data structures
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Trading pair (e.g., QNK/USDT, TOKEN/QNK)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TradingPair {
    pub base: String,    // e.g., "QNK", "TOKEN1"
    pub quote: String,   // e.g., "QNK", "USDT"
}

impl TradingPair {
    pub fn new(base: impl Into<String>, quote: impl Into<String>) -> Self {
        Self {
            base: base.into(),
            quote: quote.into(),
        }
    }

    pub fn symbol(&self) -> String {
        format!("{}/{}", self.base, self.quote)
    }
}

/// Order side (buy or sell)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderSide {
    Buy,
    Sell,
}

impl std::fmt::Display for OrderSide {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            OrderSide::Buy => write!(f, "BUY"),
            OrderSide::Sell => write!(f, "SELL"),
        }
    }
}

/// Order type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderType {
    Market,           // Execute immediately at market price
    Limit,            // Execute at specific price or better
    StopLoss,         // Sell when price drops below threshold
    TakeProfit,       // Sell when price rises above threshold
}

/// Order status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderStatus {
    Pending,
    PartiallyFilled,
    Filled,
    Cancelled,
    Failed,
}

/// Trading order
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub id: String,
    pub pair: TradingPair,
    pub side: OrderSide,
    pub order_type: OrderType,
    pub amount: Decimal,
    pub price: Option<Decimal>,  // None for market orders
    pub filled_amount: Decimal,
    pub status: OrderStatus,
    pub wallet_id: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Market ticker data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ticker {
    pub pair: TradingPair,
    pub last_price: Decimal,
    pub bid: Decimal,
    pub ask: Decimal,
    pub volume_24h: Decimal,
    pub high_24h: Decimal,
    pub low_24h: Decimal,
    pub change_24h: Decimal,
    pub timestamp: DateTime<Utc>,
}

/// Order book entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookEntry {
    pub price: Decimal,
    pub amount: Decimal,
}

/// Order book
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    pub pair: TradingPair,
    pub bids: Vec<OrderBookEntry>,  // Buy orders (highest first)
    pub asks: Vec<OrderBookEntry>,  // Sell orders (lowest first)
    pub timestamp: DateTime<Utc>,
}

/// Wallet balance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalletBalance {
    pub wallet_id: String,
    pub qnk_balance: Decimal,
    pub custom_tokens: HashMap<String, Decimal>,
}

/// Trade execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeResult {
    pub order_id: String,
    pub pair: TradingPair,
    pub side: OrderSide,
    pub amount: Decimal,
    pub price: Decimal,
    pub total_value: Decimal,
    pub fee: Decimal,
    pub timestamp: DateTime<Utc>,
    pub success: bool,
    pub error: Option<String>,
}

/// Trading signal from strategy
#[derive(Debug, Clone)]
pub enum TradingSignal {
    Buy {
        pair: TradingPair,
        amount: Decimal,
        price: Option<Decimal>,
        reason: String,
    },
    Sell {
        pair: TradingPair,
        amount: Decimal,
        price: Option<Decimal>,
        reason: String,
    },
    Hold,
}

/// Portfolio position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub token: String,
    pub amount: Decimal,
    pub average_entry_price: Decimal,
    pub current_price: Decimal,
    pub unrealized_pnl: Decimal,
    pub unrealized_pnl_percentage: Decimal,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub total_trades: usize,
    pub successful_trades: usize,
    pub failed_trades: usize,
    pub win_rate: f64,
    pub total_volume: Decimal,
    pub total_profit_loss: Decimal,
    pub roi_percentage: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: Decimal,
    pub avg_trade_duration_minutes: f64,
}
