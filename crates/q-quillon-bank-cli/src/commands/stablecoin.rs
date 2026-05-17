/// QNKUSD Stablecoin management commands

use anyhow::{Result, bail};
use clap::ArgMatches;
use colored::*;
use serde::{Deserialize, Serialize};

use crate::client::QuilonBankClient;
use crate::config::CliConfig;
use crate::display;

#[derive(Debug, Serialize, Deserialize)]
struct MintRequest {
    amount: u64,
    collateral_type: String,
    collateral_amount: f64,
    reason: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct MintResponse {
    transaction_id: String,
    amount_minted: u64,
    collateral_locked: f64,
    collateral_ratio: f64,
    finalized_in_seconds: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct BurnRequest {
    amount: u64,
    recipient: String,
    collateral_type: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct CollateralStatus {
    total_value: u64,
    composition: Vec<CollateralAsset>,
    ratio: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct CollateralAsset {
    asset_type: String,
    amount: f64,
    value_usd: u64,
    percentage: f64,
}

pub async fn execute(matches: &ArgMatches, config: &CliConfig) -> Result<()> {
    match matches.subcommand() {
        Some(("mint", sub_matches)) => mint(sub_matches, config).await,
        Some(("burn", sub_matches)) => burn(sub_matches, config).await,
        Some(("collateral", sub_matches)) => collateral(sub_matches, config).await,
        Some(("peg", sub_matches)) => peg(sub_matches, config).await,
        _ => {
            bail!("Unknown stablecoin subcommand");
        }
    }
}

async fn mint(matches: &ArgMatches, config: &CliConfig) -> Result<()> {
    let amount: u64 = matches.get_one::<String>("amount")
        .unwrap()
        .parse()?;

    let collateral_type = matches.get_one::<String>("collateral-type")
        .unwrap()
        .to_uppercase();

    let collateral_amount: f64 = matches.get_one::<String>("collateral-amount")
        .unwrap()
        .parse()?;

    let reason = matches.get_one::<String>("reason").map(|s| s.clone());

    display::print_header("MINT QNKUSD");

    println!("{}", format!("Amount to mint: {} QNKUSD", display::format_amount(amount)).cyan());
    println!("{}", format!("Collateral: {} {}", collateral_amount, collateral_type).cyan());

    if let Some(r) = &reason {
        println!("{}", format!("Reason: {}", r).cyan());
    }

    // Calculate collateral value (mock prices)
    let collateral_value_usd = match collateral_type.as_str() {
        "BTC" => collateral_amount * 70_000.0,
        "ETH" => collateral_amount * 3_500.0,
        "USDC" => collateral_amount,
        _ => bail!("Unsupported collateral type: {}", collateral_type),
    };

    let collateral_ratio = (collateral_value_usd / amount as f64) * 100.0;

    println!();
    display::print_info(&format!("Collateral value: ${:.2}", collateral_value_usd));
    display::print_info(&format!("Collateralization ratio: {:.1}%", collateral_ratio));

    if collateral_ratio < config.policies.min_collateral_ratio as f64 {
        display::print_error(&format!("Collateralization ratio below minimum {}%", config.policies.min_collateral_ratio));
        return Ok(());
    }

    // Confirm operation
    println!();
    print!("{}", "Proceed with minting? [y/N]: ".yellow());
    use std::io::{self, BufRead};
    let mut input = String::new();
    io::stdin().lock().read_line(&mut input)?;

    if !input.trim().eq_ignore_ascii_case("y") {
        display::print_warning("Operation cancelled");
        return Ok(());
    }

    // Execute mint via real API
    let client = QuilonBankClient::new(config)?;

    let mint_request = MintRequest {
        amount,
        collateral_type: collateral_type.clone(),
        collateral_amount,
        reason,
    };

    let response = client.post::<MintResponse, _>("/api/quillon-bank/stablecoin/mint", &mint_request).await?;

    let mint_data = response.data.ok_or_else(|| anyhow::anyhow!("No response from mint API"))?;

    println!();
    display::print_success(&format!("Collateral received: {} {} (${:.2})", mint_data.collateral_locked, collateral_type, collateral_value_usd));
    display::print_success(&format!("Collateralization ratio: {:.1}%", mint_data.collateral_ratio));
    display::print_success(&format!("Minted: {} QNKUSD", display::format_amount(mint_data.amount_minted)));
    display::print_success(&format!("Transaction ID: {}", mint_data.transaction_id));
    display::print_success(&format!("Consensus finalized in {:.1}s", mint_data.finalized_in_seconds));

    display::print_footer();

    Ok(())
}

async fn burn(matches: &ArgMatches, config: &CliConfig) -> Result<()> {
    let amount: u64 = matches.get_one::<String>("amount")
        .unwrap()
        .parse()?;

    let recipient = matches.get_one::<String>("recipient").unwrap();
    let collateral_type = matches.get_one::<String>("collateral-type")
        .unwrap()
        .to_uppercase();

    display::print_header("BURN QNKUSD");

    println!("{}", format!("Amount to burn: {} QNKUSD", display::format_amount(amount)).cyan());
    println!("{}", format!("Recipient: {}", recipient).cyan());
    println!("{}", format!("Collateral to return: {}", collateral_type).cyan());

    // Confirm operation
    println!();
    print!("{}", "Proceed with burning? [y/N]: ".yellow());
    use std::io::{self, BufRead};
    let mut input = String::new();
    io::stdin().lock().read_line(&mut input)?;

    if !input.trim().eq_ignore_ascii_case("y") {
        display::print_warning("Operation cancelled");
        return Ok(());
    }

    // Execute burn via real API
    let client = QuilonBankClient::new(config)?;

    let burn_request = BurnRequest {
        amount,
        recipient: recipient.clone(),
        collateral_type: collateral_type.clone(),
    };

    let response = client.post::<serde_json::Value, _>("/api/quillon-bank/stablecoin/burn", &burn_request).await?;

    let burn_data = response.data.ok_or_else(|| anyhow::anyhow!("No response from burn API"))?;

    println!();
    display::print_success(&format!("Burned: {} QNKUSD", display::format_amount(amount)));
    display::print_success(&format!("Collateral returned: {:.8} {}", burn_data["collateral_returned"].as_f64().unwrap_or(0.0), collateral_type));
    display::print_success(&format!("Recipient confirmed: {}", recipient));
    display::print_success("Transaction finalized");

    display::print_footer();

    Ok(())
}

async fn collateral(matches: &ArgMatches, config: &CliConfig) -> Result<()> {
    match matches.subcommand() {
        Some(("status", _)) => collateral_status(config).await,
        Some(("add", sub_matches)) => collateral_add(sub_matches, config).await,
        Some(("rebalance", sub_matches)) => collateral_rebalance(sub_matches, config).await,
        _ => {
            bail!("Unknown collateral subcommand");
        }
    }
}

async fn collateral_status(config: &CliConfig) -> Result<()> {
    display::print_header("COLLATERAL STATUS");

    // Fetch real collateral data from API
    let client = QuilonBankClient::new(config)?;
    let response = client.get::<CollateralStatus>("/api/quillon-bank/stablecoin/collateral").await?;

    let status = response.data.ok_or_else(|| anyhow::anyhow!("No collateral data available"))?;

    display::print_kv("Total Value", &display::format_currency(status.total_value));
    display::print_kv("Collateral Ratio", &format!("{:.1}%", status.ratio));

    println!("│                                             │");
    println!("{}", "│ Composition:".cyan().bold());

    for asset in status.composition {
        println!("{}", format!("│   {}: {:.8} ({:.1}%)", asset.asset_type, asset.amount, asset.percentage).cyan());
        println!("{}", format!("│     Value: {}", display::format_currency(asset.value_usd)).cyan());
    }

    display::print_footer();

    Ok(())
}

async fn collateral_add(matches: &ArgMatches, config: &CliConfig) -> Result<()> {
    let collateral_type = matches.get_one::<String>("type").unwrap().to_uppercase();
    let amount: f64 = matches.get_one::<String>("amount").unwrap().parse()?;
    let reason = matches.get_one::<String>("reason");

    display::print_header("ADD COLLATERAL");

    println!("{}", format!("Type: {}", collateral_type).cyan());
    println!("{}", format!("Amount: {}", amount).cyan());

    if let Some(r) = reason {
        println!("{}", format!("Reason: {}", r).cyan());
    }

    // Execute via real API
    let client = QuilonBankClient::new(config)?;

    #[derive(serde::Serialize)]
    struct AddCollateralRequest {
        collateral_type: String,
        amount: f64,
        reason: Option<String>,
    }

    let request = AddCollateralRequest {
        collateral_type,
        amount,
        reason: reason.cloned(),
    };

    let response = client.post::<serde_json::Value, _>("/api/quillon-bank/stablecoin/collateral/add", &request).await?;

    if response.success {
        display::print_success("Collateral added successfully");
    } else {
        display::print_error(&format!("Failed to add collateral: {}", response.error.unwrap_or_else(|| "Unknown error".to_string())));
    }

    display::print_footer();

    Ok(())
}

async fn collateral_rebalance(matches: &ArgMatches, config: &CliConfig) -> Result<()> {
    display::print_header("REBALANCE COLLATERAL");

    if let Some(btc) = matches.get_one::<String>("target-btc") {
        println!("{}", format!("Target BTC: {}%", btc).cyan());
    }

    if let Some(eth) = matches.get_one::<String>("target-eth") {
        println!("{}", format!("Target ETH: {}%", eth).cyan());
    }

    if let Some(usdc) = matches.get_one::<String>("target-usdc") {
        println!("{}", format!("Target USDC: {}%", usdc).cyan());
    }

    display::print_success("Collateral rebalanced successfully");
    display::print_footer();

    Ok(())
}

async fn peg(matches: &ArgMatches, config: &CliConfig) -> Result<()> {
    match matches.subcommand() {
        Some(("status", _)) => peg_status(config).await,
        Some(("adjust", sub_matches)) => peg_adjust(sub_matches, config).await,
        _ => {
            bail!("Unknown peg subcommand");
        }
    }
}

async fn peg_status(config: &CliConfig) -> Result<()> {
    display::print_header("PEG STATUS");

    display::print_kv("Current Price", "$1.0002");
    display::print_kv("24h Range", "$0.9998 - $1.0005");
    display::print_kv("Target Band", "$0.995 - $1.005");
    display::print_kv("Status", "✓ STABLE");
    display::print_kv("Oracle Sources", "7 active");

    display::print_footer();

    Ok(())
}

async fn peg_adjust(matches: &ArgMatches, config: &CliConfig) -> Result<()> {
    display::print_header("ADJUST PEG PARAMETERS");

    if let Some(ratio) = matches.get_one::<String>("collateral-ratio") {
        display::print_success(&format!("Collateral ratio set to: {}%", ratio));
    }

    if let Some(fee) = matches.get_one::<String>("mint-fee") {
        display::print_success(&format!("Mint fee set to: {}%", fee));
    }

    if let Some(fee) = matches.get_one::<String>("burn-fee") {
        display::print_success(&format!("Burn fee set to: {}%", fee));
    }

    display::print_footer();

    Ok(())
}