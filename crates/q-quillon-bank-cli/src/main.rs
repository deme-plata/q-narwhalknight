/// Quillon Bank CLI - Board Control Interface
///
/// Centralized command-line interface for managing Quillon Bank operations
/// including QNKUSD stablecoin, lending, treasury, and quantum features.

use anyhow::Result;
use clap::{Command, Arg, ArgMatches, ArgAction};
use colored::*;
use tracing::{info, error};

mod auth;
mod client;
mod commands;
mod config;
mod display;
mod modules;

use auth::AuthManager;
use client::QuilonBankClient;
use config::CliConfig;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"))
        )
        .init();

    // Build CLI
    let app = build_cli();
    let matches = app.get_matches();

    // Load configuration
    let config = CliConfig::load()?;

    // Execute command
    if let Err(e) = execute_command(&matches, config).await {
        error!("{}", format!("❌ Error: {}", e).red());
        std::process::exit(1);
    }

    Ok(())
}

fn build_cli() -> Command {
    Command::new("quillon-bank")
        .version("0.1.0")
        .author("Quillon Bank Board")
        .about("🏦 Quillon Bank - Quantum-Enhanced Banking CLI")
        .subcommand_required(true)
        .arg_required_else_help(true)
        .subcommand(
            Command::new("init")
                .about("Initialize CLI configuration")
                .arg(Arg::new("board-member")
                    .long("board-member")
                    .action(ArgAction::SetTrue)
                    .help("Initialize as board member"))
                .arg(Arg::new("generate-keys")
                    .long("generate-keys")
                    .action(ArgAction::SetTrue)
                    .help("Generate authentication keys"))
        )
        .subcommand(
            Command::new("auth")
                .about("Authentication management")
                .subcommand_required(true)
                .subcommand(
                    Command::new("login")
                        .about("Login as board member")
                        .arg(Arg::new("role")
                            .long("role")
                            .value_name("ROLE")
                            .default_value("board")
                            .help("Role to authenticate as"))
                        .arg(Arg::new("key-file")
                            .long("key-file")
                            .value_name("PATH")
                            .help("Path to authentication key"))
                        .arg(Arg::new("mfa-token")
                            .long("mfa-token")
                            .value_name("TOKEN")
                            .help("MFA token (Google Authenticator)"))
                )
                .subcommand(Command::new("status").about("Check authentication status"))
                .subcommand(Command::new("logout").about("Logout current session"))
        )
        .subcommand(
            Command::new("status")
                .about("View bank status")
                .arg(Arg::new("full")
                    .long("full")
                    .action(ArgAction::SetTrue)
                    .help("Show full detailed status"))
        )
        .subcommand(
            Command::new("stablecoin")
                .about("QNKUSD stablecoin management")
                .subcommand_required(true)
                .subcommand(
                    Command::new("mint")
                        .about("Mint new QNKUSD")
                        .arg(Arg::new("amount")
                            .long("amount")
                            .value_name("AMOUNT")
                            .required(true)
                            .help("Amount to mint"))
                        .arg(Arg::new("collateral-type")
                            .long("collateral-type")
                            .value_name("TYPE")
                            .required(true)
                            .help("Collateral type (BTC, ETH, USDC)"))
                        .arg(Arg::new("collateral-amount")
                            .long("collateral-amount")
                            .value_name("AMOUNT")
                            .required(true)
                            .help("Collateral amount"))
                        .arg(Arg::new("reason")
                            .long("reason")
                            .value_name("REASON")
                            .help("Reason for minting"))
                )
                .subcommand(
                    Command::new("burn")
                        .about("Burn QNKUSD")
                        .arg(Arg::new("amount")
                            .long("amount")
                            .value_name("AMOUNT")
                            .required(true)
                            .help("Amount to burn"))
                        .arg(Arg::new("recipient")
                            .long("recipient")
                            .value_name("ADDRESS")
                            .required(true)
                            .help("Recipient address for collateral return"))
                        .arg(Arg::new("collateral-type")
                            .long("collateral-type")
                            .value_name("TYPE")
                            .required(true)
                            .help("Collateral type to return"))
                )
                .subcommand(
                    Command::new("collateral")
                        .about("Collateral management")
                        .subcommand_required(true)
                        .subcommand(Command::new("status").about("View collateral status"))
                        .subcommand(
                            Command::new("add")
                                .about("Add collateral")
                                .arg(Arg::new("type")
                                    .long("type")
                                    .value_name("TYPE")
                                    .required(true)
                                    .help("Collateral type"))
                                .arg(Arg::new("amount")
                                    .long("amount")
                                    .value_name("AMOUNT")
                                    .required(true)
                                    .help("Amount to add"))
                                .arg(Arg::new("reason")
                                    .long("reason")
                                    .value_name("REASON")
                                    .help("Reason for adding collateral"))
                        )
                        .subcommand(
                            Command::new("rebalance")
                                .about("Rebalance collateral mix")
                                .arg(Arg::new("target-btc")
                                    .long("target-btc")
                                    .value_name("PERCENT")
                                    .help("Target BTC percentage"))
                                .arg(Arg::new("target-eth")
                                    .long("target-eth")
                                    .value_name("PERCENT")
                                    .help("Target ETH percentage"))
                                .arg(Arg::new("target-usdc")
                                    .long("target-usdc")
                                    .value_name("PERCENT")
                                    .help("Target USDC percentage"))
                        )
                )
                .subcommand(
                    Command::new("peg")
                        .about("Peg stability management")
                        .subcommand_required(true)
                        .subcommand(Command::new("status").about("Check peg status"))
                        .subcommand(
                            Command::new("adjust")
                                .about("Adjust stability parameters")
                                .arg(Arg::new("collateral-ratio")
                                    .long("collateral-ratio")
                                    .value_name("PERCENT")
                                    .help("Target collateral ratio"))
                                .arg(Arg::new("mint-fee")
                                    .long("mint-fee")
                                    .value_name("PERCENT")
                                    .help("Minting fee"))
                                .arg(Arg::new("burn-fee")
                                    .long("burn-fee")
                                    .value_name("PERCENT")
                                    .help("Burning fee"))
                        )
                )
        )
        .subcommand(
            Command::new("lending")
                .about("Lending operations")
                .subcommand_required(true)
                .subcommand(
                    Command::new("applications")
                        .about("View loan applications")
                        .arg(Arg::new("status")
                            .long("status")
                            .value_name("STATUS")
                            .help("Filter by status"))
                        .arg(Arg::new("min-amount")
                            .long("min-amount")
                            .value_name("AMOUNT")
                            .help("Minimum loan amount"))
                )
                .subcommand(
                    Command::new("approve")
                        .about("Approve loan")
                        .arg(Arg::new("loan-id")
                            .required(true)
                            .help("Loan ID to approve"))
                        .arg(Arg::new("amount")
                            .long("amount")
                            .value_name("AMOUNT")
                            .required(true)
                            .help("Loan amount"))
                        .arg(Arg::new("interest-rate")
                            .long("interest-rate")
                            .value_name("RATE")
                            .required(true)
                            .help("Interest rate (%)"))
                        .arg(Arg::new("term")
                            .long("term")
                            .value_name("TERM")
                            .required(true)
                            .help("Loan term (e.g., 12-months)"))
                )
                .subcommand(
                    Command::new("at-risk")
                        .about("View loans at risk")
                        .arg(Arg::new("collateral-ratio-below")
                            .long("collateral-ratio-below")
                            .value_name("PERCENT")
                            .help("Show loans below collateral ratio"))
                )
                .subcommand(
                    Command::new("liquidate")
                        .about("Liquidate loan")
                        .arg(Arg::new("loan-id")
                            .required(true)
                            .help("Loan ID to liquidate"))
                        .arg(Arg::new("reason")
                            .long("reason")
                            .value_name("REASON")
                            .help("Reason for liquidation"))
                        .arg(Arg::new("notify-customer")
                            .long("notify-customer")
                            .action(ArgAction::SetTrue)
                            .help("Notify customer"))
                )
        )
        .subcommand(
            Command::new("accounts")
                .about("Account management")
                .subcommand_required(true)
                .subcommand(
                    Command::new("list")
                        .about("List accounts")
                        .arg(Arg::new("min-balance")
                            .long("min-balance")
                            .value_name("AMOUNT")
                            .help("Minimum balance"))
                        .arg(Arg::new("sort-by")
                            .long("sort-by")
                            .value_name("FIELD")
                            .help("Sort by field"))
                        .arg(Arg::new("limit")
                            .long("limit")
                            .value_name("N")
                            .help("Limit results"))
                )
                .subcommand(Command::new("pending-approvals").about("View pending account approvals"))
                .subcommand(
                    Command::new("approve")
                        .about("Approve account")
                        .arg(Arg::new("account-id")
                            .required(true)
                            .help("Account ID"))
                        .arg(Arg::new("credit-limit")
                            .long("credit-limit")
                            .value_name("AMOUNT")
                            .help("Initial credit limit"))
                        .arg(Arg::new("privacy-tier")
                            .long("privacy-tier")
                            .value_name("TIER")
                            .help("Privacy tier (standard, enhanced, quantum)"))
                )
        )
        .subcommand(
            Command::new("treasury")
                .about("Treasury management")
                .subcommand_required(true)
                .subcommand(
                    Command::new("reserves")
                        .about("Reserve management")
                        .subcommand_required(true)
                        .subcommand(Command::new("status").about("View reserve status"))
                        .subcommand(
                            Command::new("allocate")
                                .about("Allocate reserves")
                                .arg(Arg::new("to")
                                    .long("to")
                                    .value_name("TARGET")
                                    .required(true)
                                    .help("Allocation target"))
                                .arg(Arg::new("amount")
                                    .long("amount")
                                    .value_name("AMOUNT")
                                    .required(true)
                                    .help("Amount to allocate"))
                                .arg(Arg::new("reason")
                                    .long("reason")
                                    .value_name("REASON")
                                    .help("Reason for allocation"))
                        )
                )
                .subcommand(
                    Command::new("profits")
                        .about("Profit management")
                        .subcommand_required(true)
                        .subcommand(
                            Command::new("calculate")
                                .about("Calculate profits")
                                .arg(Arg::new("period")
                                    .long("period")
                                    .value_name("PERIOD")
                                    .required(true)
                                    .help("Period (e.g., 2025-09)"))
                        )
                        .subcommand(
                            Command::new("distribute")
                                .about("Distribute profits")
                                .arg(Arg::new("board-dividend")
                                    .long("board-dividend")
                                    .value_name("AMOUNT")
                                    .required(true)
                                    .help("Board dividend amount"))
                                .arg(Arg::new("reserves")
                                    .long("reserves")
                                    .value_name("AMOUNT")
                                    .required(true)
                                    .help("Amount to reserves"))
                                .arg(Arg::new("customer-rewards")
                                    .long("customer-rewards")
                                    .value_name("AMOUNT")
                                    .required(true)
                                    .help("Customer rewards amount"))
                        )
                )
        )
        .subcommand(
            Command::new("risk")
                .about("Risk management")
                .subcommand_required(true)
                .subcommand(
                    Command::new("assessment")
                        .about("Risk assessment")
                        .arg(Arg::new("type")
                            .value_name("TYPE")
                            .default_value("daily")
                            .help("Assessment type (daily, market, credit)"))
                )
                .subcommand(
                    Command::new("liquidations")
                        .about("Liquidation management")
                        .subcommand_required(true)
                        .subcommand(Command::new("queue").about("View liquidation queue"))
                        .subcommand(
                            Command::new("execute")
                                .about("Execute liquidations")
                                .arg(Arg::new("threshold")
                                    .long("threshold")
                                    .value_name("PERCENT")
                                    .help("Collateral threshold"))
                        )
                )
        )
        .subcommand(
            Command::new("claude-mode")
                .about("🤖 Claude Code natural language interface")
                .arg(Arg::new("autonomous")
                    .long("autonomous")
                    .action(ArgAction::SetTrue)
                    .help("Enable autonomous operations"))
                .arg(Arg::new("allow-liquidations")
                    .long("allow-liquidations")
                    .action(ArgAction::SetTrue)
                    .help("Allow automatic liquidations"))
                .arg(Arg::new("max-mint")
                    .long("max-mint")
                    .value_name("AMOUNT")
                    .help("Maximum mint per day"))
        )
        .subcommand(
            Command::new("ask")
                .about("Ask a natural language question")
                .arg(Arg::new("question")
                    .required(true)
                    .help("Question to ask"))
        )
        .subcommand(
            Command::new("analytics")
                .about("Analytics and reporting")
                .subcommand_required(true)
                .subcommand(Command::new("daily-summary").about("Generate daily summary"))
                .subcommand(
                    Command::new("customers")
                        .about("Customer analytics")
                        .arg(Arg::new("segment")
                            .long("segment")
                            .value_name("SEGMENT")
                            .help("Customer segment"))
                )
        )
        .subcommand(
            Command::new("paas")
                .about("🔒 Privacy-as-a-Service management")
                .subcommand_required(true)
                .subcommand(Command::new("stats").about("View PaaS statistics"))
                .subcommand(
                    Command::new("audit")
                        .about("Query audit records")
                        .arg(Arg::new("wallet")
                            .long("wallet")
                            .value_name("ADDRESS")
                            .help("Filter by wallet address"))
                        .arg(Arg::new("service")
                            .long("service")
                            .value_name("SERVICE")
                            .help("Filter by service type"))
                        .arg(Arg::new("limit")
                            .long("limit")
                            .value_name("N")
                            .default_value("50")
                            .help("Maximum records to return"))
                )
                .subcommand(
                    Command::new("reservations")
                        .about("View active reservations")
                        .arg(Arg::new("wallet")
                            .long("wallet")
                            .value_name("ADDRESS")
                            .help("Filter by wallet address"))
                )
                .subcommand(Command::new("billing-stats").about("View billing statistics"))
                .subcommand(Command::new("idempotency-stats").about("View idempotency cache stats"))
                .subcommand(Command::new("pricing").about("View current pricing"))
                .subcommand(
                    Command::new("api-keys")
                        .about("API key management")
                        .subcommand_required(true)
                        .subcommand(
                            Command::new("list")
                                .about("List API keys")
                                .arg(Arg::new("wallet")
                                    .long("wallet")
                                    .value_name("ADDRESS")
                                    .help("Filter by wallet address"))
                        )
                        .subcommand(
                            Command::new("generate")
                                .about("Generate new API key")
                                .arg(Arg::new("wallet")
                                    .long("wallet")
                                    .value_name("ADDRESS")
                                    .required(true)
                                    .help("Wallet address"))
                                .arg(Arg::new("tier")
                                    .long("tier")
                                    .value_name("TIER")
                                    .required(true)
                                    .help("API tier (free, developer, production, enterprise)"))
                                .arg(Arg::new("expires-days")
                                    .long("expires-days")
                                    .value_name("DAYS")
                                    .default_value("90")
                                    .help("Expiration in days"))
                        )
                        .subcommand(
                            Command::new("rotate")
                                .about("Rotate API key")
                                .arg(Arg::new("key-id")
                                    .required(true)
                                    .help("API key ID to rotate"))
                        )
                        .subcommand(
                            Command::new("revoke")
                                .about("Revoke API key")
                                .arg(Arg::new("key-id")
                                    .required(true)
                                    .help("API key ID to revoke"))
                                .arg(Arg::new("reason")
                                    .long("reason")
                                    .value_name("REASON")
                                    .help("Reason for revocation"))
                        )
                )
        )
}

async fn execute_command(matches: &ArgMatches, config: CliConfig) -> Result<()> {
    match matches.subcommand() {
        Some(("init", sub_matches)) => {
            commands::init::execute(sub_matches, &config).await
        }
        Some(("auth", sub_matches)) => {
            commands::auth::execute(sub_matches, &config).await
        }
        Some(("status", sub_matches)) => {
            commands::status::execute(sub_matches, &config).await
        }
        Some(("stablecoin", sub_matches)) => {
            commands::stablecoin::execute(sub_matches, &config).await
        }
        Some(("lending", sub_matches)) => {
            commands::lending::execute(sub_matches, &config).await
        }
        Some(("accounts", sub_matches)) => {
            commands::accounts::execute(sub_matches, &config).await
        }
        Some(("treasury", sub_matches)) => {
            commands::treasury::execute(sub_matches, &config).await
        }
        Some(("risk", sub_matches)) => {
            commands::risk::execute(sub_matches, &config).await
        }
        Some(("claude-mode", sub_matches)) => {
            commands::claude::execute(sub_matches, &config).await
        }
        Some(("ask", sub_matches)) => {
            commands::ask::execute(sub_matches, &config).await
        }
        Some(("analytics", sub_matches)) => {
            commands::analytics::execute(sub_matches, &config).await
        }
        Some(("paas", sub_matches)) => {
            commands::paas::execute(sub_matches, &config).await
        }
        _ => {
            println!("{}", "❌ Unknown command".red());
            Ok(())
        }
    }
}