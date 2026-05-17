//! DEX Activity Bot Strategy
//!
//! This strategy generates organic trading activity on the DEX using a portion
//! of the development fee. It creates a natural flow of buy/sell orders to:
//! - Provide liquidity depth
//! - Generate price discovery
//! - Create trading volume
//! - Make the DEX appear active and healthy
//!
//! ## Funding Model
//! - Uses 10% of the 1% dev fee (0.1% of total rewards)
//! - 99% goes to miner
//! - 0.9% goes to development
//! - 0.1% goes to DEX activity (this bot)
//!
//! ## Trading Patterns
//! - Random small buys/sells throughout the day
//! - Occasional larger trades for volume
//! - Mean-reverting trades to prevent price manipulation
//! - Primary focus on QBANK token trading

use super::*;
use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc, Duration};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use std::collections::HashMap;
use tracing::{debug, info, warn};

/// DEX Activity Bot Configuration
#[derive(Debug, Clone)]
pub struct DexActivityConfig {
    /// Minimum time between trades (seconds)
    pub min_trade_interval_secs: u64,
    /// Maximum time between trades (seconds)
    pub max_trade_interval_secs: u64,
    /// Minimum trade size as percentage of wallet balance
    pub min_trade_size_percent: Decimal,
    /// Maximum trade size as percentage of wallet balance
    pub max_trade_size_percent: Decimal,
    /// Target daily trade count
    pub target_daily_trades: u64,
    /// Maximum price deviation from market (prevents manipulation)
    pub max_price_deviation_percent: Decimal,
    /// Whether to enable mean-reverting behavior
    pub mean_reversion_enabled: bool,
    /// Primary token to trade (QBANK)
    pub primary_token: String,
    /// Trading pairs to cycle through
    pub trading_pairs: Vec<String>,
    /// Enable random bursts of activity
    pub burst_mode_enabled: bool,
    /// Probability of burst mode (0.0-1.0)
    pub burst_probability: f64,
    /// Number of trades in a burst
    pub burst_trade_count: u64,
}

impl Default for DexActivityConfig {
    fn default() -> Self {
        Self {
            min_trade_interval_secs: 30,    // v8.6.0: lowered from 60s for more responsive trading
            max_trade_interval_secs: 1800,   // v8.6.0: lowered from 3600s (30 min max gap)
            min_trade_size_percent: Decimal::new(1, 2),  // 0.01 = 1%
            max_trade_size_percent: Decimal::new(8, 2),  // v8.6.0: raised from 0.05 to 0.08 = 8%
            target_daily_trades: 200, // v8.6.0: raised from 100 for more DEX activity
            max_price_deviation_percent: Decimal::new(2, 2), // 2%
            mean_reversion_enabled: true,
            primary_token: "QBANK".to_string(),
            trading_pairs: vec![
                "QBANK/ORB".to_string(),
                "QBANK/USDT".to_string(),
            ],
            burst_mode_enabled: true,
            burst_probability: 0.05, // 5% chance of burst
            burst_trade_count: 5,
        }
    }
}

/// DEX Activity Bot State
pub struct DexActivityState {
    /// Last trade timestamp
    pub last_trade_time: DateTime<Utc>,
    /// Trades executed today
    pub trades_today: u64,
    /// Total volume traded today
    pub volume_today: Decimal,
    /// Recent price history for mean reversion
    pub price_history: Vec<(DateTime<Utc>, Decimal)>,
    /// Running position (net buy/sell)
    pub net_position: Decimal,
    /// Total trades executed
    pub total_trades: u64,
    /// Current pair index
    pub current_pair_index: usize,
    /// Burst mode active
    pub burst_mode_active: bool,
    /// Remaining burst trades
    pub burst_trades_remaining: u64,
    /// Random seed for reproducibility
    pub rng: StdRng,
}

impl Default for DexActivityState {
    fn default() -> Self {
        Self {
            last_trade_time: Utc::now() - Duration::hours(1),
            trades_today: 0,
            volume_today: Decimal::ZERO,
            price_history: Vec::new(),
            net_position: Decimal::ZERO,
            total_trades: 0,
            current_pair_index: 0,
            burst_mode_active: false,
            burst_trades_remaining: 0,
            rng: StdRng::from_entropy(),
        }
    }
}

/// DEX Activity Bot Strategy
///
/// Generates organic trading activity using development fee allocation.
/// Primary focus on QBANK token to create market depth and price discovery.
pub struct DexActivityStrategy {
    name: String,
    config: DexActivityConfig,
    state: DexActivityState,
    /// Wallet ID that holds the DEX activity funds (from dev fee)
    activity_wallet_id: String,
    /// Maximum daily spend (to prevent runaway)
    max_daily_spend: Decimal,
    /// Current daily spend
    daily_spend: Decimal,
    /// Last reset date
    last_reset: DateTime<Utc>,
}

impl DexActivityStrategy {
    pub fn new(activity_wallet_id: String, config: DexActivityConfig) -> Self {
        Self {
            name: "DEX Activity Bot".to_string(),
            config,
            state: DexActivityState::default(),
            activity_wallet_id,
            max_daily_spend: Decimal::new(1000, 0), // Max 1000 QNK per day
            daily_spend: Decimal::ZERO,
            last_reset: Utc::now(),
        }
    }

    /// Create with default configuration for QBANK trading
    pub fn new_for_qbank(activity_wallet_id: String) -> Self {
        let config = DexActivityConfig {
            primary_token: "QBANK".to_string(),
            trading_pairs: vec![
                "QBANK/ORB".to_string(),
                "QBANK/USDT".to_string(),
                "ORB/USDT".to_string(),
            ],
            target_daily_trades: 150, // Active trading
            burst_mode_enabled: true,
            burst_probability: 0.1, // 10% chance for "explosion of trades"
            burst_trade_count: 10,
            ..Default::default()
        };

        Self::new(activity_wallet_id, config)
    }

    /// Calculate time until next trade
    fn calculate_next_trade_time(&mut self) -> Duration {
        let range = self.config.max_trade_interval_secs - self.config.min_trade_interval_secs;
        let random_offset = self.state.rng.gen_range(0..range);
        let interval = self.config.min_trade_interval_secs + random_offset;
        Duration::seconds(interval as i64)
    }

    /// Calculate trade size based on balance and configuration
    fn calculate_trade_size(&mut self, balance: &WalletBalance) -> Decimal {
        let available = balance.qnk_balance;
        if available <= Decimal::ZERO {
            return Decimal::ZERO;
        }

        // Random size between min and max percentage
        let min_pct = self.config.min_trade_size_percent.to_f64().unwrap_or(0.01);
        let max_pct = self.config.max_trade_size_percent.to_f64().unwrap_or(0.05);
        let random_pct = self.state.rng.gen_range(min_pct..max_pct);

        let trade_size = available * Decimal::from_f64(random_pct).unwrap_or(Decimal::new(1, 2));

        // Cap at remaining daily budget
        let remaining = self.max_daily_spend - self.daily_spend;
        trade_size.min(remaining).max(Decimal::ZERO)
    }

    /// Determine if we should buy or sell (with mean reversion)
    fn determine_trade_direction(&mut self, ticker: &Ticker) -> OrderSide {
        // Mean reversion: if we've been buying, tend to sell, and vice versa
        if self.config.mean_reversion_enabled {
            // Strong position on one side? Revert
            let threshold = Decimal::new(100, 0); // 100 tokens net position threshold
            if self.state.net_position > threshold {
                // We're long, more likely to sell
                if self.state.rng.gen_bool(0.7) {
                    return OrderSide::Sell;
                }
            } else if self.state.net_position < -threshold {
                // We're short, more likely to buy
                if self.state.rng.gen_bool(0.7) {
                    return OrderSide::Buy;
                }
            }
        }

        // Price-based decision with some randomness
        if self.state.price_history.len() >= 5 {
            let avg_price: Decimal = self.state.price_history
                .iter()
                .rev()
                .take(5)
                .map(|(_, p)| *p)
                .sum::<Decimal>() / Decimal::new(5, 0);

            let deviation = (ticker.last_price - avg_price) / avg_price;
            let deviation_threshold = self.config.max_price_deviation_percent;

            // Buy when price is below average, sell when above
            if deviation < -deviation_threshold && self.state.rng.gen_bool(0.6) {
                return OrderSide::Buy;
            } else if deviation > deviation_threshold && self.state.rng.gen_bool(0.6) {
                return OrderSide::Sell;
            }
        }

        // Random choice with slight buy bias (for long-term appreciation)
        if self.state.rng.gen_bool(0.55) {
            OrderSide::Buy
        } else {
            OrderSide::Sell
        }
    }

    /// Check if burst mode should activate
    fn check_burst_mode(&mut self) -> bool {
        if !self.config.burst_mode_enabled {
            return false;
        }

        if self.state.burst_mode_active {
            return self.state.burst_trades_remaining > 0;
        }

        // Random chance to enter burst mode
        if self.state.rng.gen_bool(self.config.burst_probability) {
            info!("🚀 DEX Activity Bot entering BURST MODE!");
            self.state.burst_mode_active = true;
            self.state.burst_trades_remaining = self.config.burst_trade_count;
            return true;
        }

        false
    }

    /// Get the next trading pair to trade
    fn get_next_pair(&mut self) -> TradingPair {
        if self.config.trading_pairs.is_empty() {
            return TradingPair::new(&self.config.primary_token, "ORB");
        }

        let pair_str = &self.config.trading_pairs[self.state.current_pair_index];
        self.state.current_pair_index =
            (self.state.current_pair_index + 1) % self.config.trading_pairs.len();

        // Parse pair string (e.g., "QBANK/ORB")
        let parts: Vec<&str> = pair_str.split('/').collect();
        if parts.len() == 2 {
            TradingPair::new(parts[0], parts[1])
        } else {
            TradingPair::new(&self.config.primary_token, "ORB")
        }
    }

    /// Reset daily counters if needed
    fn check_daily_reset(&mut self) {
        let now = Utc::now();
        let last_date = self.last_reset.date_naive();
        let current_date = now.date_naive();

        if current_date > last_date {
            info!("🔄 DEX Activity Bot: Daily reset");
            self.state.trades_today = 0;
            self.state.volume_today = Decimal::ZERO;
            self.daily_spend = Decimal::ZERO;
            self.last_reset = now;
        }
    }

    /// Generate a trade reason for logging
    fn generate_trade_reason(&self, side: OrderSide) -> String {
        let reasons = if side == OrderSide::Buy {
            vec![
                "Organic liquidity provision",
                "Market depth enhancement",
                "Price support activity",
                "Volume generation",
                "Spread tightening",
            ]
        } else {
            vec![
                "Position rebalancing",
                "Profit taking activity",
                "Liquidity provision",
                "Market making",
                "Volume generation",
            ]
        };

        let burst_prefix = if self.state.burst_mode_active { "🚀 BURST: " } else { "" };
        format!("{}{}", burst_prefix, reasons[self.state.total_trades as usize % reasons.len()])
    }
}

#[async_trait]
impl Strategy for DexActivityStrategy {
    fn name(&self) -> &str {
        &self.name
    }

    async fn analyze(&mut self, ticker: &Ticker, balance: &WalletBalance) -> Result<TradingSignal> {
        // Check for daily reset
        self.check_daily_reset();

        // Check if we've exceeded daily trade target
        if self.state.trades_today >= self.config.target_daily_trades && !self.state.burst_mode_active {
            debug!("DEX Activity Bot: Daily trade target reached ({})", self.state.trades_today);
            return Ok(TradingSignal::Hold);
        }

        // Check daily spend limit
        if self.daily_spend >= self.max_daily_spend {
            debug!("DEX Activity Bot: Daily spend limit reached");
            return Ok(TradingSignal::Hold);
        }

        let now = Utc::now();

        // Check burst mode or normal interval
        let should_trade = if self.check_burst_mode() {
            // In burst mode, trade more frequently
            let burst_interval = Duration::seconds(5); // 5 seconds between burst trades
            now > self.state.last_trade_time + burst_interval
        } else {
            // Normal mode: check interval
            let next_trade_time = self.state.last_trade_time + self.calculate_next_trade_time();
            now >= next_trade_time
        };

        if !should_trade {
            return Ok(TradingSignal::Hold);
        }

        // Update price history
        self.state.price_history.push((now, ticker.last_price));
        if self.state.price_history.len() > 100 {
            self.state.price_history.remove(0);
        }

        // Calculate trade parameters
        let trade_size = self.calculate_trade_size(balance);
        if trade_size <= Decimal::ZERO {
            return Ok(TradingSignal::Hold);
        }

        let side = self.determine_trade_direction(ticker);
        let pair = self.get_next_pair();
        let reason = self.generate_trade_reason(side);

        // Update state
        self.state.last_trade_time = now;
        self.state.total_trades += 1;

        if self.state.burst_mode_active {
            self.state.burst_trades_remaining -= 1;
            if self.state.burst_trades_remaining == 0 {
                info!("💥 DEX Activity Bot: Burst mode complete!");
                self.state.burst_mode_active = false;
            }
        }

        // Generate signal
        info!(
            "🤖 DEX Activity Bot: {} {} {} @ {} ({})",
            if side == OrderSide::Buy { "BUY" } else { "SELL" },
            trade_size,
            pair.symbol(),
            ticker.last_price,
            reason
        );

        match side {
            OrderSide::Buy => Ok(TradingSignal::Buy {
                pair,
                amount: trade_size,
                price: None, // Market order
                reason,
            }),
            OrderSide::Sell => Ok(TradingSignal::Sell {
                pair,
                amount: trade_size,
                price: None, // Market order
                reason,
            }),
        }
    }

    async fn on_order_filled(&mut self, order: &Order) -> Result<()> {
        // Update tracking
        self.state.trades_today += 1;
        self.state.volume_today += order.filled_amount;
        self.daily_spend += order.filled_amount * order.price.unwrap_or(Decimal::ONE);

        // Update net position for mean reversion
        match order.side {
            OrderSide::Buy => {
                self.state.net_position += order.filled_amount;
            }
            OrderSide::Sell => {
                self.state.net_position -= order.filled_amount;
            }
        }

        info!(
            "✅ DEX Activity Bot trade filled: {} {} @ {} | Daily: {} trades, {} volume",
            order.side,
            order.filled_amount,
            order.price.unwrap_or(Decimal::ZERO),
            self.state.trades_today,
            self.state.volume_today
        );

        Ok(())
    }
}

/// DEX Activity Wallet Manager
///
/// Manages the wallet that receives the 0.1% DEX activity allocation
/// from the development fee (10% of the 1% dev fee).
pub struct DexActivityWallet {
    /// Wallet address
    pub address: String,
    /// Current balance
    pub balance: Decimal,
    /// Total received from dev fee
    pub total_received: Decimal,
    /// Total spent on trading
    pub total_spent: Decimal,
    /// Last top-up timestamp
    pub last_topup: DateTime<Utc>,
}

impl DexActivityWallet {
    pub fn new(address: String) -> Self {
        Self {
            address,
            balance: Decimal::ZERO,
            total_received: Decimal::ZERO,
            total_spent: Decimal::ZERO,
            last_topup: Utc::now(),
        }
    }

    /// Add funds from dev fee allocation
    pub fn receive_from_dev_fee(&mut self, amount: Decimal) {
        self.balance += amount;
        self.total_received += amount;
        self.last_topup = Utc::now();
        info!(
            "💰 DEX Activity Wallet received {} from dev fee (Total: {})",
            amount, self.total_received
        );
    }

    /// Spend on trading activity
    pub fn spend(&mut self, amount: Decimal) -> bool {
        if self.balance >= amount {
            self.balance -= amount;
            self.total_spent += amount;
            true
        } else {
            false
        }
    }

    /// Get available balance for trading
    pub fn available(&self) -> Decimal {
        self.balance
    }
}

/// Calculate DEX activity allocation from mining reward
///
/// Returns (miner_amount, dev_amount, dex_activity_amount)
pub fn calculate_dex_activity_split(total_reward: u64) -> (u64, u64, u64) {
    // Total dev fee is 1%
    let total_dev_fee = (total_reward as f64 * 0.01) as u64;

    // DEX activity gets 10% of dev fee (0.1% of total)
    let dex_activity = (total_dev_fee as f64 * 0.10) as u64;

    // Remaining dev fee (0.9% of total)
    let dev_fee = total_dev_fee - dex_activity;

    // Miner gets 99% of total
    let miner_reward = total_reward - total_dev_fee;

    (miner_reward, dev_fee, dex_activity)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dex_activity_split() {
        let total = 100_000_000_000u64; // 100 QNK in smallest units
        let (miner, dev, dex) = calculate_dex_activity_split(total);

        // Miner gets 99%
        assert_eq!(miner, 99_000_000_000);
        // Dev gets 0.9%
        assert_eq!(dev, 900_000_000);
        // DEX activity gets 0.1%
        assert_eq!(dex, 100_000_000);
        // Total should sum correctly
        assert_eq!(miner + dev + dex, total);
    }

    #[test]
    fn test_strategy_creation() {
        let strategy = DexActivityStrategy::new_for_qbank("test_wallet".to_string());
        assert_eq!(strategy.name(), "DEX Activity Bot");
        assert_eq!(strategy.config.primary_token, "QBANK");
    }

    #[test]
    fn test_wallet_operations() {
        let mut wallet = DexActivityWallet::new("test_address".to_string());

        wallet.receive_from_dev_fee(Decimal::new(100, 0));
        assert_eq!(wallet.available(), Decimal::new(100, 0));

        assert!(wallet.spend(Decimal::new(50, 0)));
        assert_eq!(wallet.available(), Decimal::new(50, 0));

        assert!(!wallet.spend(Decimal::new(100, 0))); // Not enough
        assert_eq!(wallet.available(), Decimal::new(50, 0));
    }

    #[test]
    fn test_config_defaults() {
        let config = DexActivityConfig::default();
        assert!(config.burst_mode_enabled);
        assert_eq!(config.target_daily_trades, 100);
        assert!(config.mean_reversion_enabled);
    }
}
