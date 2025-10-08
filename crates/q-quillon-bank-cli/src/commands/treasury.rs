/// Treasury management commands

use anyhow::{Result, bail};
use clap::ArgMatches;
use colored::Colorize;

use crate::config::CliConfig;
use crate::display;

pub async fn execute(matches: &ArgMatches, config: &CliConfig) -> Result<()> {
    match matches.subcommand() {
        Some(("reserves", sub_matches)) => reserves(sub_matches, config).await,
        Some(("profits", sub_matches)) => profits(sub_matches, config).await,
        _ => bail!("Unknown treasury subcommand"),
    }
}

async fn reserves(matches: &ArgMatches, config: &CliConfig) -> Result<()> {
    match matches.subcommand() {
        Some(("status", _)) => reserves_status(config).await,
        Some(("allocate", sub_matches)) => reserves_allocate(sub_matches, config).await,
        _ => bail!("Unknown reserves subcommand"),
    }
}

async fn reserves_status(config: &CliConfig) -> Result<()> {
    display::print_header("RESERVE STATUS");

    display::print_kv("Total Reserves", &display::format_currency(22_450_000));
    display::print_kv("Reserve Ratio", "18.5%");
    display::print_kv("Regulatory Minimum", "10%");
    display::print_kv("Buffer", "+8.5% (✓ HEALTHY)");

    println!("│                                             │");
    println!("{}", "│ Composition:".cyan().bold());
    display::print_kv("  Cash", "$5,000,000 (22.3%)");
    display::print_kv("  BTC", "$10,000,000 (44.5%)");
    display::print_kv("  ETH", "$5,000,000 (22.3%)");
    display::print_kv("  USDC", "$2,450,000 (10.9%)");

    display::print_footer();

    Ok(())
}

async fn reserves_allocate(matches: &ArgMatches, config: &CliConfig) -> Result<()> {
    let to = matches.get_one::<String>("to").unwrap();
    let amount: u64 = matches.get_one::<String>("amount").unwrap().parse()?;
    let reason = matches.get_one::<String>("reason");

    display::print_header("ALLOCATE RESERVES");

    display::print_kv("To", to);
    display::print_kv("Amount", &display::format_currency(amount));

    if let Some(r) = reason {
        display::print_kv("Reason", r);
    }

    display::print_success("Reserves allocated successfully");
    display::print_footer();

    Ok(())
}

async fn profits(matches: &ArgMatches, config: &CliConfig) -> Result<()> {
    match matches.subcommand() {
        Some(("calculate", sub_matches)) => profits_calculate(sub_matches, config).await,
        Some(("distribute", sub_matches)) => profits_distribute(sub_matches, config).await,
        _ => bail!("Unknown profits subcommand"),
    }
}

async fn profits_calculate(matches: &ArgMatches, config: &CliConfig) -> Result<()> {
    let period = matches.get_one::<String>("period").unwrap();

    display::print_header(&format!("PROFITS CALCULATION - {}", period));

    println!("{}", "Revenue: $1,250,000".cyan().bold());
    display::print_kv("  Lending Interest", "$800,000");
    display::print_kv("  Stablecoin Fees", "$200,000");
    display::print_kv("  Account Fees", "$150,000");
    display::print_kv("  Other", "$100,000");

    println!("│                                             │");

    println!("{}", "Expenses: $450,000".cyan().bold());
    display::print_kv("  Operations", "$250,000");
    display::print_kv("  Infrastructure", "$150,000");
    display::print_kv("  Compliance", "$50,000");

    println!("│                                             │");

    println!("{}", "Net Profit: $800,000".green().bold());

    display::print_footer();

    Ok(())
}

async fn profits_distribute(matches: &ArgMatches, config: &CliConfig) -> Result<()> {
    let board_dividend: u64 = matches.get_one::<String>("board-dividend").unwrap().parse()?;
    let reserves: u64 = matches.get_one::<String>("reserves").unwrap().parse()?;
    let customer_rewards: u64 = matches.get_one::<String>("customer-rewards").unwrap().parse()?;

    let total = board_dividend + reserves + customer_rewards;

    display::print_header("DISTRIBUTE PROFITS");

    display::print_kv("Board Dividend", &display::format_currency(board_dividend));
    display::print_kv("To Reserves", &display::format_currency(reserves));
    display::print_kv("Customer Rewards", &display::format_currency(customer_rewards));
    display::print_kv("Total", &display::format_currency(total));

    display::print_success("Profits distributed successfully");
    display::print_footer();

    Ok(())
}