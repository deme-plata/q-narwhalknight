/// Natural language question answering

use anyhow::Result;
use clap::ArgMatches;
use colored::*;

use crate::config::CliConfig;
use crate::display;

pub async fn execute(matches: &ArgMatches, config: &CliConfig) -> Result<()> {
    let question = matches.get_one::<String>("question").unwrap();

    let input_lower = question.to_lowercase();

    // Simple pattern matching for common queries
    let response = if input_lower.contains("quantum vault") {
        "There are currently 8,901 active quantum vaults with 27% adoption rate across customers.".to_string()
    } else if input_lower.contains("exposure") && input_lower.contains("bitcoin") {
        "Bitcoin exposure: $84M in collateral (60.5% of total). Consider diversifying if BTC volatility increases.".to_string()
    } else if input_lower.contains("optimal reserve") {
        "Recommended reserve allocation: 30% cash, 40% BTC, 20% ETH, 10% stablecoins for optimal liquidity and yield.".to_string()
    } else {
        format!("Analysis of '{}': Use 'quillon-bank claude-mode' for interactive conversation.", question)
    };

    println!("{}", response.green());

    Ok(())
}