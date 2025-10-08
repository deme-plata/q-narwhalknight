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

    // Mock data
    let applications = vec![
        ("loan-001", "Alice", 50000, 750, "pending"),
        ("loan-002", "Bob", 25000, 680, "pending"),
        ("loan-003", "Carol", 100000, 820, "pending"),
    ];

    let mut table = display::create_table(vec!["ID", "Borrower", "Amount", "Credit Score", "Status"]);

    for (id, borrower, amount, score, status) in applications {
        table.add_row(prettytable::row![
            id,
            borrower,
            display::format_currency(amount),
            score,
            status
        ]);
    }

    table.printstd();

    display::print_info(&format!("{} pending applications", 3));

    Ok(())
}

async fn approve(matches: &ArgMatches, config: &CliConfig) -> Result<()> {
    let loan_id = matches.get_one::<String>("loan-id").unwrap();
    let amount: u64 = matches.get_one::<String>("amount").unwrap().parse()?;
    let interest_rate: f64 = matches.get_one::<String>("interest-rate").unwrap().parse()?;
    let term = matches.get_one::<String>("term").unwrap();

    display::print_header("APPROVE LOAN");

    display::print_kv("Loan ID", loan_id);
    display::print_kv("Amount", &display::format_currency(amount));
    display::print_kv("Interest Rate", &format!("{:.1}%", interest_rate));
    display::print_kv("Term", term);

    display::print_success("Loan approved successfully");
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