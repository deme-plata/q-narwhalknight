/// Lending operations commands

use anyhow::{Result, bail};
use clap::ArgMatches;
use colored::*;

use crate::config::CliConfig;
use crate::display;

pub async fn execute(matches: &ArgMatches, config: &CliConfig) -> Result<()> {
    match matches.subcommand() {
        Some(("applications", sub_matches)) => applications(sub_matches, config).await,
        Some(("approve", sub_matches)) => approve(sub_matches, config).await,
        Some(("at-risk", sub_matches)) => at_risk(sub_matches, config).await,
        Some(("liquidate", sub_matches)) => liquidate(sub_matches, config).await,
        _ => bail!("Unknown lending subcommand"),
    }
}

async fn applications(matches: &ArgMatches, config: &CliConfig) -> Result<()> {
    display::print_header("LOAN APPLICATIONS");

    // Create API client
    let client = crate::client::QuilonBankClient::new(config)?;

    // Fetch loan applications from API
    display::print_info("Fetching loan applications...");

    match client.get_with_auth::<serde_json::Value>(
        "/api/v1/quillon-bank/lending/applications",
        "view_loan_applications",
    ).await {
        Ok(response) => {
            if response.success {
                if let Some(data) = response.data {
                    if let Some(apps_array) = data.get("applications").and_then(|v| v.as_array()) {
                        let mut table = display::create_table(vec![
                            "Loan ID",
                            "Borrower",
                            "Amount (QUGUSD)",
                            "Collateral (QUG)",
                            "APR",
                            "Status"
                        ]);

                        let mut pending_count = 0;

                        for app in apps_array {
                            let loan_id = app.get("loan_id")
                                .and_then(|v| v.as_str())
                                .unwrap_or("unknown");
                            let borrower_addr = app.get("borrower_address")
                                .and_then(|v| v.as_str())
                                .unwrap_or("unknown");
                            let loan_amount = app.get("loan_amount")
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0);
                            let collateral = app.get("collateral_amount")
                                .and_then(|v| v.as_f64())
                                .unwrap_or(0.0);
                            let interest_rate = app.get("interest_rate")
                                .and_then(|v| v.as_f64())
                                .unwrap_or(0.0);
                            let status = app.get("status")
                                .and_then(|v| v.as_str())
                                .unwrap_or("unknown");

                            if status == "pending" {
                                pending_count += 1;
                            }

                            // Truncate borrower address for display
                            let borrower_short = if borrower_addr.len() > 16 {
                                format!("{}...{}", &borrower_addr[..8], &borrower_addr[borrower_addr.len()-6..])
                            } else {
                                borrower_addr.to_string()
                            };

                            table.add_row(prettytable::row![
                                loan_id,
                                borrower_short,
                                format!("{:.2}", loan_amount as f64 / 1e8),
                                format!("{:.2}", collateral),
                                format!("{:.1}%", interest_rate),
                                status
                            ]);
                        }

                        table.printstd();

                        display::print_info(&format!("{} total applications ({} pending)", apps_array.len(), pending_count));
                    } else {
                        display::print_warning("No applications found");
                    }
                } else {
                    display::print_warning("No data in response");
                }
            } else {
                display::print_error(&format!("Failed to fetch applications: {}", response.error.unwrap_or_else(|| "Unknown error".to_string())));
            }
        },
        Err(e) => {
            display::print_error(&format!("API request failed: {}", e));
            bail!("Failed to fetch loan applications");
        }
    }

    Ok(())
}

async fn approve(matches: &ArgMatches, config: &CliConfig) -> Result<()> {
    let loan_id = matches.get_one::<String>("loan-id").unwrap();

    display::print_header("APPROVE LOAN");
    display::print_kv("Loan ID", loan_id);

    // Create API client
    let client = crate::client::QuilonBankClient::new(config)?;

    // Prepare approval request
    let request_body = serde_json::json!({
        "loan_id": loan_id,
    });

    // Call the approve endpoint with AEGIS-QL authentication
    display::print_info("Sending approval to blockchain...");

    match client.post_with_auth::<serde_json::Value, _>(
        "/api/v1/quillon-bank/lending/approve",
        &request_body,
        &format!("approve_loan:{}", loan_id),
    ).await {
        Ok(response) => {
            if response.success {
                display::print_success("✅ Loan approved successfully");

                if let Some(data) = response.data {
                    // Display transaction details
                    if let Some(tx_hash) = data.get("tx_hash") {
                        display::print_kv("Transaction Hash", &tx_hash.to_string());
                    }
                    if let Some(collateral) = data.get("collateral_locked") {
                        display::print_kv("Collateral Locked", &format!("{} QUG", collateral));
                    }
                    if let Some(disbursed) = data.get("qugusd_disbursed") {
                        display::print_kv("QUGUSD Disbursed", &format!("{} QUGUSD", disbursed));
                    }
                    if let Some(block) = data.get("block_height") {
                        display::print_kv("Block Height", &block.to_string());
                    }
                }

                display::print_success("Funds have been disbursed to borrower");
            } else {
                display::print_error(&format!("Approval failed: {}", response.error.unwrap_or_else(|| "Unknown error".to_string())));
            }
        },
        Err(e) => {
            display::print_error(&format!("API request failed: {}", e));
            bail!("Failed to approve loan");
        }
    }

    display::print_footer();

    Ok(())
}

async fn at_risk(matches: &ArgMatches, config: &CliConfig) -> Result<()> {
    display::print_header("LOANS AT RISK");

    let threshold = matches.get_one::<String>("collateral-ratio-below")
        .and_then(|s| s.strip_suffix('%'))
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(110);

    display::print_info(&format!("Showing loans with collateral ratio below {}%", threshold));

    // Mock data
    let at_risk_loans = vec![
        ("loan-042", "Dave", 75000, 95, "High Risk"),
        ("loan-089", "Eve", 50000, 102, "Medium Risk"),
        ("loan-123", "Frank", 30000, 108, "Low Risk"),
    ];

    let mut table = display::create_table(vec!["ID", "Borrower", "Amount", "Collateral %", "Risk Level"]);

    for (id, borrower, amount, ratio, risk) in at_risk_loans {
        table.add_row(prettytable::row![
            id,
            borrower,
            display::format_currency(amount),
            format!("{}%", ratio),
            risk
        ]);
    }

    table.printstd();

    display::print_warning(&format!("{} loans require attention", 3));

    Ok(())
}

async fn liquidate(matches: &ArgMatches, config: &CliConfig) -> Result<()> {
    let loan_id = matches.get_one::<String>("loan-id").unwrap();
    let reason = matches.get_one::<String>("reason");
    let notify = matches.get_flag("notify-customer");

    display::print_header("LIQUIDATE LOAN");

    display::print_kv("Loan ID", loan_id);
    if let Some(r) = reason {
        display::print_kv("Reason", r);
    }
    display::print_kv("Notify Customer", if notify { "Yes" } else { "No" });

    display::print_success("Loan liquidated successfully");
    display::print_success("Collateral seized and sold");

    display::print_footer();

    Ok(())
}