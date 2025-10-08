/// Bank status dashboard

use anyhow::Result;
use clap::ArgMatches;
use colored::*;
use colored::Colorize;
use serde::{Deserialize, Serialize};

use crate::client::QuilonBankClient;
use crate::config::CliConfig;
use crate::display;

#[derive(Debug, Deserialize, Serialize)]
struct BankStatus {
    qnkusd: StablecoinStatus,
    banking: BankingStatus,
    risk: RiskStatus,
    quantum: QuantumStatus,
}

#[derive(Debug, Deserialize, Serialize)]
struct StablecoinStatus {
    total_supply: u64,
    collateral_value: u64,
    collateralization_ratio: f64,
    peg_price: f64,
}

#[derive(Debug, Deserialize, Serialize)]
struct BankingStatus {
    active_accounts: u64,
    total_deposits: u64,
    active_loans: u64,
    average_credit_score: f64,
}

#[derive(Debug, Deserialize, Serialize)]
struct RiskStatus {
    loans_at_risk_count: u64,
    loans_at_risk_value: u64,
    liquidation_queue: u64,
    reserve_ratio: f64,
}

#[derive(Debug, Deserialize, Serialize)]
struct QuantumStatus {
    quantum_vaults: u64,
    post_quantum_transactions_24h: u64,
    quantum_privacy_adoption: f64,
}

pub async fn execute(matches: &ArgMatches, config: &CliConfig) -> Result<()> {
    let full = matches.get_flag("full");

    display::print_header(&format!("QUILLON BANK DAILY STATUS - {}", chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")));

    let client = QuilonBankClient::new(config)?;

    // Fetch real status from Quillon Bank API
    let status = match fetch_real_status(&client).await {
        Ok(s) => s,
        Err(e) => {
            display::print_error(&format!("Failed to fetch bank status: {}", e));
            display::print_warning("Using fallback data - check API connection");
            return Err(e);
        }
    };

    // Display QNKUSD Stablecoin status
    println!("{}", "│ QNKUSD Stablecoin".cyan().bold());
    display::print_kv("  Total Supply", &format!("{} QNKUSD", display::format_amount(status.qnkusd.total_supply)));
    display::print_kv("  Collateral Value", &display::format_currency(status.qnkusd.collateral_value));
    display::print_kv("  Collateralization", &format!("{:.1}%", status.qnkusd.collateralization_ratio));

    let peg_status = if status.qnkusd.peg_price >= 0.995 && status.qnkusd.peg_price <= 1.005 {
        format!("${:.4} (✓ Within range)", status.qnkusd.peg_price).green()
    } else {
        format!("${:.4} (⚠ Out of range)", status.qnkusd.peg_price).yellow()
    };
    println!("{}", format!("│   Peg Status: {:<15} │", peg_status));

    println!("│                                             │");

    // Display Banking Operations
    println!("{}", "│ Banking Operations".cyan().bold());
    display::print_kv("  Active Accounts", &display::format_amount(status.banking.active_accounts));
    display::print_kv("  Total Deposits", &display::format_currency(status.banking.total_deposits));
    display::print_kv("  Active Loans", &display::format_currency(status.banking.active_loans));
    display::print_kv("  Avg Credit Score", &format!("{:.1}", status.banking.average_credit_score));

    println!("│                                             │");

    // Display Risk Metrics
    println!("{}", "│ Risk Metrics".cyan().bold());
    display::print_kv("  Loans at Risk", &format!("{} ({})", status.risk.loans_at_risk_count, display::format_currency(status.risk.loans_at_risk_value)));
    display::print_kv("  Liquidation Queue", &format!("{} accounts", status.risk.liquidation_queue));
    display::print_kv("  Reserve Ratio", &format!("{:.1}%", status.risk.reserve_ratio));

    if full {
        println!("│                                             │");

        // Display Quantum Features
        println!("{}", "│ Quantum Features".cyan().bold());
        display::print_kv("  Quantum Vaults", &format!("{} active", display::format_amount(status.quantum.quantum_vaults)));
        display::print_kv("  PQ Transactions", &format!("{} (24h)", display::format_amount(status.quantum.post_quantum_transactions_24h)));
        display::print_kv("  Privacy Adoption", &format!("{:.1}%", status.quantum.quantum_privacy_adoption * 100.0));
    }

    display::print_footer();

    // Alerts
    if status.qnkusd.collateralization_ratio < 105.0 {
        display::print_warning("Collateralization ratio below minimum 105%!");
    }

    if status.risk.liquidation_queue > 0 {
        display::print_warning(&format!("{} loans pending liquidation", status.risk.liquidation_queue));
    }

    Ok(())
}

async fn fetch_real_status(client: &QuilonBankClient) -> Result<BankStatus> {
    // Fetch QNKUSD stablecoin status
    let stablecoin_response = client.get::<serde_json::Value>("/api/quillon-bank/stablecoin/status").await?;

    // Fetch banking metrics
    let banking_response = client.get::<serde_json::Value>("/api/quillon-bank/metrics").await?;

    // Fetch risk metrics
    let risk_response = client.get::<serde_json::Value>("/api/quillon-bank/risk/status").await?;

    // Fetch quantum features status
    let quantum_response = client.get::<serde_json::Value>("/api/quillon-bank/quantum/status").await?;

    // Parse responses
    let stablecoin_data = stablecoin_response.data.ok_or_else(|| anyhow::anyhow!("No stablecoin data"))?;
    let banking_data = banking_response.data.ok_or_else(|| anyhow::anyhow!("No banking data"))?;
    let risk_data = risk_response.data.ok_or_else(|| anyhow::anyhow!("No risk data"))?;
    let quantum_data = quantum_response.data.ok_or_else(|| anyhow::anyhow!("No quantum data"))?;

    Ok(BankStatus {
        qnkusd: StablecoinStatus {
            total_supply: stablecoin_data["total_supply"].as_u64().unwrap_or(0),
            collateral_value: stablecoin_data["collateral_value"].as_u64().unwrap_or(0),
            collateralization_ratio: stablecoin_data["collateralization_ratio"].as_f64().unwrap_or(0.0),
            peg_price: stablecoin_data["peg_price"].as_f64().unwrap_or(1.0),
        },
        banking: BankingStatus {
            active_accounts: banking_data["active_accounts"].as_u64().unwrap_or(0),
            total_deposits: banking_data["total_deposits"].as_u64().unwrap_or(0),
            active_loans: banking_data["active_loans"].as_u64().unwrap_or(0),
            average_credit_score: banking_data["average_credit_score"].as_f64().unwrap_or(0.0),
        },
        risk: RiskStatus {
            loans_at_risk_count: risk_data["loans_at_risk_count"].as_u64().unwrap_or(0),
            loans_at_risk_value: risk_data["loans_at_risk_value"].as_u64().unwrap_or(0),
            liquidation_queue: risk_data["liquidation_queue"].as_u64().unwrap_or(0),
            reserve_ratio: risk_data["reserve_ratio"].as_f64().unwrap_or(0.0),
        },
        quantum: QuantumStatus {
            quantum_vaults: quantum_data["quantum_vaults"].as_u64().unwrap_or(0),
            post_quantum_transactions_24h: quantum_data["post_quantum_transactions_24h"].as_u64().unwrap_or(0),
            quantum_privacy_adoption: quantum_data["quantum_privacy_adoption"].as_f64().unwrap_or(0.0),
        },
    })
}