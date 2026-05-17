/// Risk management commands

use anyhow::{Result, bail};
use clap::ArgMatches;
use colored::*;

use crate::config::CliConfig;
use crate::display;

pub async fn execute(matches: &ArgMatches, config: &CliConfig) -> Result<()> {
    match matches.subcommand() {
        Some(("assessment", sub_matches)) => assessment(sub_matches, config).await,
        Some(("liquidations", sub_matches)) => liquidations(sub_matches, config).await,
        _ => bail!("Unknown risk subcommand"),
    }
}

async fn assessment(matches: &ArgMatches, config: &CliConfig) -> Result<()> {
    let assessment_type = matches.get_one::<String>("type")
        .map(|s| s.as_str())
        .unwrap_or("daily");

    display::print_header(&format!("RISK ASSESSMENT - {}", assessment_type.to_uppercase()));

    println!("{}", "│ Credit Risk: LOW ✓".green());
    display::print_kv("  Loans at risk", "0.3% of portfolio");

    println!("│                                             │");

    println!("{}", "│ Market Risk: MEDIUM ⚠".yellow());
    display::print_kv("  BTC volatility", "15% (24h)");
    display::print_kv("  Recommend", "Review collateral");

    println!("│                                             │");

    println!("{}", "│ Liquidity Risk: LOW ✓".green());
    display::print_kv("  Liquid assets", "25% of deposits");

    println!("│                                             │");

    println!("{}", "│ Operational Risk: LOW ✓".green());
    display::print_kv("  System uptime", "99.98%");
    display::print_kv("  Failed transactions", "0.02%");

    display::print_footer();

    Ok(())
}

async fn liquidations(matches: &ArgMatches, config: &CliConfig) -> Result<()> {
    match matches.subcommand() {
        Some(("queue", _)) => liquidations_queue(config).await,
        Some(("execute", sub_matches)) => liquidations_execute(sub_matches, config).await,
        _ => bail!("Unknown liquidations subcommand"),
    }
}

async fn liquidations_queue(config: &CliConfig) -> Result<()> {
    display::print_header("LIQUIDATION QUEUE");

    let queue = vec![
        ("loan-042", "Dave", 75000, 95, "Immediate"),
        ("loan-089", "Eve", 50000, 102, "24h grace"),
        ("loan-123", "Frank", 30000, 108, "Warning sent"),
    ];

    let mut table = display::create_table(vec!["ID", "Borrower", "Amount", "Collateral %", "Status"]);

    for (id, borrower, amount, ratio, status) in queue {
        table.add_row(prettytable::row![
            id,
            borrower,
            display::format_currency(amount),
            format!("{}%", ratio),
            status
        ]);
    }

    table.printstd();

    display::print_warning(&format!("{} loans pending liquidation", 3));

    Ok(())
}

async fn liquidations_execute(matches: &ArgMatches, config: &CliConfig) -> Result<()> {
    let threshold = matches.get_one::<String>("threshold");

    display::print_header("EXECUTE LIQUIDATIONS");

    if let Some(t) = threshold {
        display::print_info(&format!("Liquidating loans with collateral below {}", t));
    }

    display::print_success("Liquidated 1 loan");
    display::print_info("2 loans given 24h grace period");
    display::print_info("Notifications sent to affected customers");

    display::print_footer();

    Ok(())
}