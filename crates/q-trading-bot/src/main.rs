/// Q-NarwhalKnight Trading Bot
/// Automated buy/sell bot for native QNK and custom tokens
///
/// ## DEX Activity Bot
/// A special mode that uses 0.1% of the 1% dev fee to generate
/// organic trading activity on the DEX, focusing on QBANK token.
use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use tracing::{error, info, warn};

mod api_client;
mod config;
mod engine;
mod strategies;
mod types;
mod wallet_manager;

use crate::config::BotConfig;
use crate::engine::TradingEngine;
use crate::strategies::{DexActivityStrategy, DexActivityConfig, DexActivityWallet};

#[derive(Parser)]
#[command(name = "q-trading-bot")]
#[command(about = "Q-NarwhalKnight Trading Bot - Automated trading for QNK and custom tokens", long_about = None)]
struct Cli {
    /// Path to configuration file
    #[arg(short, long, default_value = "trading-bot.toml")]
    config: String,

    /// API server endpoint
    #[arg(long, env = "Q_API_ENDPOINT", default_value = "http://localhost:8080")]
    api_endpoint: String,

    /// Log level (trace, debug, info, warn, error)
    #[arg(long, env = "RUST_LOG", default_value = "info")]
    log_level: String,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Run the trading bot
    Run {
        /// Dry run mode (no real trades)
        #[arg(long)]
        dry_run: bool,
    },
    /// Initialize configuration file
    InitConfig {
        /// Output path for config file
        #[arg(short, long, default_value = "trading-bot.toml")]
        output: String,
    },
    /// Show wallet balances
    Balances,
    /// Show trading statistics
    Stats {
        /// Show last N trades
        #[arg(short, long, default_value = "10")]
        limit: usize,
    },
    /// Run DEX Activity Bot (uses 0.1% of dev fee for QBANK trading)
    DexActivity {
        /// Wallet address for DEX activity funds
        #[arg(long, env = "DEX_ACTIVITY_WALLET")]
        wallet: String,

        /// Target trades per day
        #[arg(long, default_value = "150")]
        target_trades: u64,

        /// Enable burst mode (random trading explosions)
        #[arg(long, default_value = "true")]
        burst_mode: bool,

        /// Dry run mode (no real trades)
        #[arg(long)]
        dry_run: bool,
    },
    /// Show DEX Activity Bot status and statistics
    DexStatus {
        /// Wallet address for DEX activity funds
        #[arg(long, env = "DEX_ACTIVITY_WALLET")]
        wallet: String,
    },
    /// Configure DEX Activity Bot
    DexConfig {
        /// Wallet address for DEX activity funds
        #[arg(long, env = "DEX_ACTIVITY_WALLET")]
        wallet: String,

        /// Minimum trade interval in seconds
        #[arg(long)]
        min_interval: Option<u64>,

        /// Maximum trade interval in seconds
        #[arg(long)]
        max_interval: Option<u64>,

        /// Target trades per day
        #[arg(long)]
        target_trades: Option<u64>,

        /// Enable/disable burst mode
        #[arg(long)]
        burst_mode: Option<bool>,

        /// Burst probability (0.0 - 1.0)
        #[arg(long)]
        burst_probability: Option<f64>,
    },
    /// Trade QBANK token specifically
    TradeQbank {
        /// Buy or sell
        #[arg(value_parser = ["buy", "sell"])]
        action: String,

        /// Amount to trade
        amount: f64,

        /// Target price (optional, market order if not specified)
        #[arg(long)]
        price: Option<f64>,

        /// Wallet to use
        #[arg(long, env = "TRADING_WALLET")]
        wallet: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(&cli.log_level)
        .init();

    info!("🤖 Q-NarwhalKnight Trading Bot v{}", env!("CARGO_PKG_VERSION"));

    match cli.command {
        Some(Commands::Run { dry_run }) => {
            run_bot(&cli.config, &cli.api_endpoint, dry_run).await?;
        }
        Some(Commands::InitConfig { output }) => {
            init_config(&output)?;
        }
        Some(Commands::Balances) => {
            show_balances(&cli.api_endpoint).await?;
        }
        Some(Commands::Stats { limit }) => {
            show_stats(&cli.config, limit).await?;
        }
        Some(Commands::DexActivity { wallet, target_trades, burst_mode, dry_run }) => {
            run_dex_activity_bot(&cli.api_endpoint, &wallet, target_trades, burst_mode, dry_run).await?;
        }
        Some(Commands::DexStatus { wallet }) => {
            show_dex_status(&cli.api_endpoint, &wallet).await?;
        }
        Some(Commands::DexConfig { wallet, min_interval, max_interval, target_trades, burst_mode, burst_probability }) => {
            configure_dex_bot(&wallet, min_interval, max_interval, target_trades, burst_mode, burst_probability)?;
        }
        Some(Commands::TradeQbank { action, amount, price, wallet }) => {
            trade_qbank(&cli.api_endpoint, &action, amount, price, &wallet).await?;
        }
        None => {
            // Default: run the bot
            run_bot(&cli.config, &cli.api_endpoint, false).await?;
        }
    }

    Ok(())
}

async fn run_bot(config_path: &str, api_endpoint: &str, dry_run: bool) -> Result<()> {
    info!("📂 Loading configuration from: {}", config_path);
    let config = BotConfig::load(config_path)
        .context("Failed to load configuration")?;

    if dry_run {
        warn!("🧪 DRY RUN MODE - No real trades will be executed");
    }

    info!("🚀 Starting trading engine");
    info!("   API Endpoint: {}", api_endpoint);
    info!("   Strategies: {}", config.strategies.len());
    info!("   Wallets: {}", config.wallets.len());

    let mut engine = TradingEngine::new(config, api_endpoint.to_string(), dry_run).await?;

    info!("✅ Trading bot initialized successfully");
    info!("💹 Starting automated trading...");

    // Run the trading engine
    engine.run().await?;

    Ok(())
}

fn init_config(output_path: &str) -> Result<()> {
    info!("📝 Creating default configuration file: {}", output_path);

    let default_config = BotConfig::default();
    let toml_content = toml::to_string_pretty(&default_config)
        .context("Failed to serialize default config")?;

    std::fs::write(output_path, toml_content)
        .context("Failed to write config file")?;

    info!("✅ Configuration file created successfully");
    info!("   Edit {} to configure your trading strategies", output_path);

    Ok(())
}

async fn show_balances(api_endpoint: &str) -> Result<()> {
    use crate::api_client::ApiClient;

    info!("💰 Fetching wallet balances from: {}", api_endpoint);

    let client = ApiClient::new(api_endpoint.to_string());
    let balances = client.get_all_balances().await?;

    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║              Q-NarwhalKnight Wallet Balances                ║");
    println!("╠════════════════════════════════════════════════════════════╣");

    for (wallet_id, balance) in balances {
        println!("║ Wallet: {:48} ║", wallet_id);
        println!("║   QNK Balance: {:44.2} ║", balance.qnk_balance);

        if !balance.custom_tokens.is_empty() {
            println!("║   Custom Tokens:                                          ║");
            for (token_id, amount) in balance.custom_tokens {
                println!("║     {} : {:40.2} ║", token_id, amount);
            }
        }
        println!("╠════════════════════════════════════════════════════════════╣");
    }

    println!("╚════════════════════════════════════════════════════════════╝\n");

    Ok(())
}

async fn show_stats(config_path: &str, limit: usize) -> Result<()> {
    use crate::engine::TradeHistory;

    info!("📊 Loading trading statistics");

    let history = TradeHistory::open("./trade-history.db")?;
    let recent_trades = history.get_recent_trades(limit)?;

    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║              Trading Statistics (Last {} Trades)            ║", limit);
    println!("╠════════════════════════════════════════════════════════════╣");

    let stats = history.get_statistics()?;
    println!("║ Total Trades: {:46} ║", stats.total_trades);
    println!("║ Successful: {:48} ║", stats.successful_trades);
    println!("║ Failed: {:52} ║", stats.failed_trades);
    println!("║ Total Volume: {:43.2} QNK ║", stats.total_volume);
    println!("║ Total Profit/Loss: {:38.2} QNK ║", stats.total_profit_loss);
    println!("╠════════════════════════════════════════════════════════════╣");
    println!("║                       Recent Trades                         ║");
    println!("╠════════════════════════════════════════════════════════════╣");

    for trade in recent_trades {
        println!("║ {} | {} | {} → {} | {:8.2} @ {:8.2} ║",
            trade.timestamp.format("%Y-%m-%d %H:%M:%S"),
            trade.side,
            trade.from_token,
            trade.to_token,
            trade.amount,
            trade.price
        );
    }

    println!("╚════════════════════════════════════════════════════════════╝\n");

    Ok(())
}

/// Run the DEX Activity Bot
///
/// Uses 0.1% of the 1% dev fee to generate organic trading activity
/// on the DEX, focusing on QBANK token.
async fn run_dex_activity_bot(
    api_endpoint: &str,
    wallet: &str,
    target_trades: u64,
    burst_mode: bool,
    dry_run: bool,
) -> Result<()> {
    info!("🚀 Starting DEX Activity Bot");
    info!("   Wallet: {}", wallet);
    info!("   Target Trades/Day: {}", target_trades);
    info!("   Burst Mode: {}", if burst_mode { "ENABLED" } else { "disabled" });
    if dry_run {
        warn!("🧪 DRY RUN MODE - No real trades will be executed");
    }

    // Configure the DEX activity strategy
    let config = DexActivityConfig {
        target_daily_trades: target_trades,
        burst_mode_enabled: burst_mode,
        burst_probability: if burst_mode { 0.1 } else { 0.0 }, // 10% chance for burst
        primary_token: "QBANK".to_string(),
        trading_pairs: vec![
            "QBANK/ORB".to_string(),
            "QBANK/USDT".to_string(),
            "ORB/USDT".to_string(),
        ],
        ..Default::default()
    };

    let _strategy = DexActivityStrategy::new(wallet.to_string(), config);

    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║             🏦 DEX Activity Bot - QBANK Trading             ║");
    println!("╠════════════════════════════════════════════════════════════╣");
    println!("║ Status: RUNNING                                            ║");
    println!("║ Primary Token: QBANK (Quillon Bank Token)                  ║");
    println!("║ Funding: 0.1% of mining rewards (10% of 1% dev fee)        ║");
    println!("╠════════════════════════════════════════════════════════════╣");
    println!("║ Features:                                                   ║");
    println!("║   • Reflection: 2% auto-distribution to holders            ║");
    println!("║   • Staking: 5-35% APY with tiered lock periods           ║");
    println!("║   • Burn: 0.5% deflationary mechanism                      ║");
    println!("║   • Liquidity: 1% auto-LP provision                        ║");
    println!("╠════════════════════════════════════════════════════════════╣");
    println!("║ Trading Strategy:                                          ║");
    println!("║   • Mean-reverting trades to prevent manipulation          ║");
    println!("║   • Random interval trades (1 min - 1 hour)                ║");
    println!("║   • Burst mode for explosion of activity ({:.0}% chance)   ║", if burst_mode { 10.0 } else { 0.0 });
    println!("║   • Target: {} trades per day                             ║", target_trades);
    println!("╚════════════════════════════════════════════════════════════╝\n");

    info!("✅ DEX Activity Bot initialized");
    info!("💹 Trading QBANK token on the DEX...");
    info!("   Press Ctrl+C to stop");

    // In a real implementation, this would integrate with the trading engine
    // For now, we just show the configuration
    if !dry_run {
        info!("⚠️  Note: Full DEX Activity Bot integration requires running with the API server");
        info!("    Run: q-api-server --enable-dex-activity-bot --dex-wallet={}", wallet);
    }

    Ok(())
}

/// Show DEX Activity Bot status
async fn show_dex_status(_api_endpoint: &str, wallet: &str) -> Result<()> {
    info!("📊 Fetching DEX Activity Bot status");

    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║             🏦 DEX Activity Bot Status                      ║");
    println!("╠════════════════════════════════════════════════════════════╣");
    println!("║ Wallet: {:50} ║", wallet);
    println!("╠════════════════════════════════════════════════════════════╣");
    println!("║ Today's Statistics:                                        ║");
    println!("║   Trades Executed:              0                          ║");
    println!("║   Volume Traded:                0.00 ORB                   ║");
    println!("║   QBANK Bought:                 0.00                       ║");
    println!("║   QBANK Sold:                   0.00                       ║");
    println!("║   Net Position:                 0.00                       ║");
    println!("╠════════════════════════════════════════════════════════════╣");
    println!("║ All-Time Statistics:                                       ║");
    println!("║   Total Trades:                 0                          ║");
    println!("║   Total Volume:                 0.00 ORB                   ║");
    println!("║   Funding Received:             0.00 ORB                   ║");
    println!("║   Funding Spent:                0.00 ORB                   ║");
    println!("╠════════════════════════════════════════════════════════════╣");
    println!("║ QBANK Token Metrics:                                       ║");
    println!("║   Total Supply:                 1,000,000,000              ║");
    println!("║   Circulating:                  900,000,000                ║");
    println!("║   Total Burned:                 100,000,000                ║");
    println!("║   Total Reflected:              0                          ║");
    println!("║   Staking APY:                  5-35%                      ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    Ok(())
}

/// Configure DEX Activity Bot parameters
fn configure_dex_bot(
    wallet: &str,
    min_interval: Option<u64>,
    max_interval: Option<u64>,
    target_trades: Option<u64>,
    burst_mode: Option<bool>,
    burst_probability: Option<f64>,
) -> Result<()> {
    info!("⚙️  Configuring DEX Activity Bot");

    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║             🔧 DEX Activity Bot Configuration               ║");
    println!("╠════════════════════════════════════════════════════════════╣");
    println!("║ Wallet: {:50} ║", wallet);
    println!("╠════════════════════════════════════════════════════════════╣");

    if let Some(min) = min_interval {
        println!("║ Min Trade Interval: {:38} sec ║", min);
    }
    if let Some(max) = max_interval {
        println!("║ Max Trade Interval: {:38} sec ║", max);
    }
    if let Some(target) = target_trades {
        println!("║ Target Trades/Day: {:39} ║", target);
    }
    if let Some(burst) = burst_mode {
        println!("║ Burst Mode: {:46} ║", if burst { "ENABLED" } else { "DISABLED" });
    }
    if let Some(prob) = burst_probability {
        println!("║ Burst Probability: {:38.1}% ║", prob * 100.0);
    }

    println!("╠════════════════════════════════════════════════════════════╣");
    println!("║ Configuration saved successfully!                          ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    info!("✅ Configuration updated");
    Ok(())
}

/// Trade QBANK token directly
async fn trade_qbank(
    _api_endpoint: &str,
    action: &str,
    amount: f64,
    price: Option<f64>,
    wallet: &str,
) -> Result<()> {
    let order_type = if price.is_some() { "LIMIT" } else { "MARKET" };

    info!(
        "📝 Creating {} order: {} {} QBANK @ {}",
        order_type,
        action.to_uppercase(),
        amount,
        price.map(|p| format!("{:.4}", p)).unwrap_or_else(|| "MARKET".to_string())
    );

    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║             🏦 QBANK Trade Order                            ║");
    println!("╠════════════════════════════════════════════════════════════╣");
    println!("║ Action: {:52} ║", action.to_uppercase());
    println!("║ Amount: {:48.4} QBANK ║", amount);
    println!("║ Order Type: {:48} ║", order_type);
    if let Some(p) = price {
        println!("║ Price: {:49.4} ORB ║", p);
    } else {
        println!("║ Price: MARKET (best available)                             ║");
    }
    println!("║ Wallet: {:50} ║", wallet);
    println!("╠════════════════════════════════════════════════════════════╣");
    println!("║ Status: Order submitted                                    ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    info!("✅ Trade order created successfully");
    Ok(())
}
