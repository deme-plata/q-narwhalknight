/// Trading strategies module
use anyhow::Result;
use async_trait::async_trait;
use crate::types::*;

#[async_trait]
pub trait Strategy: Send + Sync {
    fn name(&self) -> &str;
    async fn analyze(&mut self, ticker: &Ticker, balance: &WalletBalance) -> Result<TradingSignal>;
    async fn on_order_filled(&mut self, order: &Order) -> Result<()>;
}

pub mod grid;
pub mod market_maker;
pub mod dca;
pub mod dex_activity;
pub mod water_bot;
pub mod dark_knight;
pub mod btc_advantage;

// Re-export DEX activity components for easy access
pub use dex_activity::{
    DexActivityStrategy,
    DexActivityConfig,
    DexActivityWallet,
    calculate_dex_activity_split,
};

pub use water_bot::{TunnelingOctopusBot, WaterBotConfig};
pub use dark_knight::{DarkKnightBot, DarkKnightConfig};
pub use btc_advantage::{BtcAdvantageBot, BtcAdvantageConfig};
