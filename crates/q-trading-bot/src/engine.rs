/// Main trading engine
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use sled::Db;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::time::{sleep, Duration};
use tracing::{error, info, warn};

use crate::api_client::ApiClient;
use crate::config::BotConfig;
use crate::strategies::{Strategy, grid::GridStrategy};
use crate::types::*;
use crate::wallet_manager::WalletManager;

pub struct TradingEngine {
    config: BotConfig,
    api_client: ApiClient,
    wallet_manager: WalletManager,
    strategies: Vec<Box<dyn Strategy>>,
    history: TradeHistory,
    dry_run: bool,
    daily_pnl: Decimal,
    daily_pnl_reset: DateTime<Utc>,
}

impl TradingEngine {
    pub async fn new(config: BotConfig, api_endpoint: String, dry_run: bool) -> Result<TradingEngine> {
        config.validate()?;

        let api_client = ApiClient::new(api_endpoint);
        let wallet_manager = WalletManager::new(config.wallets.clone(), api_client.clone());
        let history = TradeHistory::open("./trade-history.db")?;

        // Initialize strategies
        let mut strategies: Vec<Box<dyn Strategy>> = Vec::new();
        for strategy_config in &config.strategies {
            if !strategy_config.enabled {
                continue;
            }

            match strategy_config.strategy_type.as_str() {
                "grid" => {
                    let grid_levels = strategy_config.parameters.get("grid_levels")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(10) as usize;
                    let price_range = strategy_config.parameters.get("price_range_percent")
                        .and_then(|v| v.as_f64())
                        .map(|f| Decimal::try_from(f).unwrap())
                        .unwrap_or(Decimal::new(5, 0));

                    for pair_str in &strategy_config.pairs {
                        let parts: Vec<&str> = pair_str.split('/').collect();
                        if parts.len() == 2 {
                            let pair = TradingPair::new(parts[0], parts[1]);
                            let strategy = GridStrategy::new(
                                strategy_config.name.clone(),
                                pair,
                                grid_levels,
                                price_range,
                            );
                            strategies.push(Box::new(strategy));
                        }
                    }
                }
                _ => {
                    warn!("Unknown strategy type: {}", strategy_config.strategy_type);
                }
            }
        }

        info!("Initialized {} strategies", strategies.len());

        Ok(Self {
            config,
            api_client,
            wallet_manager,
            strategies,
            history,
            dry_run,
            daily_pnl: Decimal::ZERO,
            daily_pnl_reset: Utc::now(),
        })
    }

    pub async fn run(&mut self) -> Result<()> {
        info!("🤖 Trading engine starting");

        let update_interval = Duration::from_secs(self.config.general.update_interval_seconds);

        loop {
            if let Err(e) = self.update_cycle().await {
                error!("Error in update cycle: {}", e);
            }

            sleep(update_interval).await;
        }
    }

    async fn update_cycle(&mut self) -> Result<()> {
        // Reset daily P&L if new day
        let now = Utc::now();
        if now.date_naive() > self.daily_pnl_reset.date_naive() {
            info!("New trading day - resetting daily P&L");
            self.daily_pnl = Decimal::ZERO;
            self.daily_pnl_reset = now;
        }

        // Check if daily loss limit exceeded
        if self.config.risk_management.stop_on_daily_loss {
            let max_loss = self.config.risk_management.max_daily_loss;
            if self.daily_pnl < -max_loss {
                warn!("❌ Daily loss limit exceeded ({:.2}%), stopping trading", max_loss * Decimal::new(100, 0));
                return Ok(());
            }
        }

        // Get balances
        let balances = self.wallet_manager.get_all_balances().await?;

        // Update each strategy (avoid double &mut self via index-based iteration)
        for i in 0..self.strategies.len() {
            let name = self.strategies[i].name().to_string();
            if let Err(e) = Self::process_strategy_static(self.strategies[i].as_mut(), &balances).await {
                error!("Error processing strategy {}: {}", name, e);
            }
        }

        Ok(())
    }

    async fn process_strategy_static(
        strategy: &mut dyn Strategy,
        balances: &HashMap<String, WalletBalance>,
    ) -> Result<()> {
        // Get ticker data for this strategy's pairs
        // Analyze and execute trades
        // This is a simplified implementation - full version would:
        // 1. Fetch ticker data
        // 2. Call strategy.analyze()
        // 3. Execute resulting signals
        // 4. Track P&L
        // 5. Apply risk management

        Ok(())
    }
}

/// Trade history database
pub struct TradeHistory {
    db: Db,
}

impl TradeHistory {
    pub fn open(path: &str) -> Result<Self> {
        let db = sled::open(path)?;
        Ok(Self { db })
    }

    pub fn record_trade(&self, trade: &TradeResult) -> Result<()> {
        let key = format!("{}_{}", trade.timestamp.timestamp(), trade.order_id);
        let value = serde_json::to_vec(trade)?;
        self.db.insert(key.as_bytes(), value)?;
        Ok(())
    }

    pub fn get_recent_trades(&self, limit: usize) -> Result<Vec<TradeResult>> {
        let mut trades = Vec::new();
        for item in self.db.iter().rev().take(limit) {
            let (_, value) = item?;
            if let Ok(trade) = serde_json::from_slice(&value) {
                trades.push(trade);
            }
        }
        Ok(trades)
    }

    pub fn get_statistics(&self) -> Result<TradeStatistics> {
        let mut stats = TradeStatistics::default();
        for item in self.db.iter() {
            let (_, value) = item?;
            if let Ok(trade) = serde_json::from_slice::<TradeResult>(&value) {
                stats.total_trades += 1;
                if trade.success {
                    stats.successful_trades += 1;
                } else {
                    stats.failed_trades += 1;
                }
                stats.total_volume += trade.total_value;
                // P&L calculation would go here
            }
        }
        Ok(stats)
    }
}

#[derive(Debug, Default)]
pub struct TradeStatistics {
    pub total_trades: usize,
    pub successful_trades: usize,
    pub failed_trades: usize,
    pub total_volume: Decimal,
    pub total_profit_loss: Decimal,
}
