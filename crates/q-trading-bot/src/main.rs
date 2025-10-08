/// Q-NarwhalKnight Trading Bot
/// Automated buy/sell bot for native QNK and custom tokens
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
