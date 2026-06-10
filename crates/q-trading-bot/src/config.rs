/// Bot configuration management
use anyhow::{Context, Result};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Main bot configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BotConfig {
    /// General settings
    pub general: GeneralConfig,

    /// Wallet configurations
    pub wallets: Vec<WalletConfig>,

    /// Trading strategies
    pub strategies: Vec<StrategyConfig>,

    /// Risk management settings
    pub risk_management: RiskManagementConfig,

    /// Trading pairs to monitor
    pub trading_pairs: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralConfig {
    /// Update interval in seconds
    pub update_interval_seconds: u64,

    /// Maximum concurrent orders
    pub max_concurrent_orders: usize,

    /// Enable auto-rebalancing
    pub auto_rebalance: bool,

    /// Logging level
    pub log_level: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalletConfig {
    /// Wallet ID
    pub id: String,

    /// Wallet name/label
    pub name: String,

    /// Private key (encrypted in production)
    pub private_key: Option<String>,

    /// Enable trading for this wallet
    pub enabled: bool,

    /// Maximum trading amount per trade
    pub max_trade_amount: Decimal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyConfig {
    /// Strategy name
    pub name: String,

    /// Strategy type
    #[serde(rename = "type")]
    pub strategy_type: String,

    /// Enable this strategy
    pub enabled: bool,

    /// Wallet IDs to use for this strategy
    pub wallets: Vec<String>,

    /// Trading pairs for this strategy
    pub pairs: Vec<String>,

    /// Strategy-specific parameters
    pub parameters: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskManagementConfig {
    /// Maximum loss per trade (as decimal, e.g., 0.02 = 2%)
    pub max_loss_per_trade: Decimal,

    /// Maximum daily loss (as decimal)
    pub max_daily_loss: Decimal,

    /// Stop trading if daily loss exceeded
    pub stop_on_daily_loss: bool,

    /// Maximum portfolio allocation per token (as decimal)
    pub max_allocation_per_token: Decimal,

    /// Minimum balance to keep in native token
    pub min_native_balance: Decimal,

    /// Enable stop-loss orders
    pub enable_stop_loss: bool,

    /// Default stop-loss percentage
    pub default_stop_loss_percentage: Decimal,

    /// Enable take-profit orders
    pub enable_take_profit: bool,

    /// Default take-profit percentage
    pub default_take_profit_percentage: Decimal,
}

impl Default for BotConfig {
    fn default() -> Self {
        Self {
            general: GeneralConfig {
                update_interval_seconds: 60,
                max_concurrent_orders: 10,
                auto_rebalance: false,
                log_level: "info".to_string(),
            },
            wallets: vec![
                WalletConfig {
                    id: "wallet_1".to_string(),
                    name: "Trading Wallet 1".to_string(),
                    private_key: None,
                    enabled: true,
                    max_trade_amount: Decimal::new(1000, 0),
                }
            ],
            strategies: vec![
                StrategyConfig {
                    name: "Grid Trading".to_string(),
                    strategy_type: "grid".to_string(),
                    enabled: true,
                    wallets: vec!["wallet_1".to_string()],
                    pairs: vec!["QNK/USDT".to_string()],
                    parameters: {
                        let mut params = HashMap::new();
                        params.insert("grid_levels".to_string(), serde_json::json!(10));
                        params.insert("price_range_percent".to_string(), serde_json::json!(5.0));
                        params
                    },
                },
                StrategyConfig {
                    name: "Market Making".to_string(),
                    strategy_type: "market_maker".to_string(),
                    enabled: false,
                    wallets: vec!["wallet_1".to_string()],
                    pairs: vec!["TOKEN1/QNK".to_string()],
                    parameters: {
                        let mut params = HashMap::new();
                        params.insert("spread_percent".to_string(), serde_json::json!(0.5));
                        params.insert("order_size".to_string(), serde_json::json!(100));
                        params
                    },
                },
            ],
            risk_management: RiskManagementConfig {
                max_loss_per_trade: Decimal::new(2, 2),  // 2%
                max_daily_loss: Decimal::new(10, 2),      // 10%
                stop_on_daily_loss: true,
                max_allocation_per_token: Decimal::new(25, 2),  // 25%
                min_native_balance: Decimal::new(10, 0),  // 10 QNK
                enable_stop_loss: true,
                default_stop_loss_percentage: Decimal::new(5, 2),  // 5%
                enable_take_profit: true,
                default_take_profit_percentage: Decimal::new(10, 2),  // 10%
            },
            trading_pairs: vec![
                "QNK/USDT".to_string(),
                "TOKEN1/QNK".to_string(),
                "TOKEN2/QNK".to_string(),
            ],
        }
    }
}

impl BotConfig {
    /// Load configuration from TOML file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref())
            .context("Failed to read config file")?;

        let config: BotConfig = toml::from_str(&content)
            .context("Failed to parse TOML config")?;

        Ok(config)
    }

    /// Save configuration to TOML file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = toml::to_string_pretty(self)
            .context("Failed to serialize config")?;

        std::fs::write(path.as_ref(), content)
            .context("Failed to write config file")?;

        Ok(())
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        // Check at least one wallet is enabled
        if !self.wallets.iter().any(|w| w.enabled) {
            anyhow::bail!("No enabled wallets found in configuration");
        }

        // Check at least one strategy is enabled
        if !self.strategies.iter().any(|s| s.enabled) {
            anyhow::bail!("No enabled strategies found in configuration");
        }

        // Validate risk management parameters
        if self.risk_management.max_loss_per_trade > Decimal::new(50, 2) {
            anyhow::bail!("max_loss_per_trade cannot exceed 50%");
        }

        if self.risk_management.max_daily_loss > Decimal::ONE {
            anyhow::bail!("max_daily_loss cannot exceed 100%");
        }

        Ok(())
    }
}
