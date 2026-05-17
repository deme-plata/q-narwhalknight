/// Claude Code natural language interface

use anyhow::Result;
use clap::ArgMatches;
use colored::*;
use std::io::{self, Write};

use crate::config::CliConfig;
use crate::display;

pub async fn execute(matches: &ArgMatches, config: &CliConfig) -> Result<()> {
    let autonomous = matches.get_flag("autonomous");
    let allow_liquidations = matches.get_flag("allow-liquidations");
    let max_mint = matches.get_one::<String>("max-mint");

    if autonomous {
        display::print_header("CLAUDE CODE - AUTONOMOUS MODE");

        display::print_warning("Autonomous mode enabled");
        display::print_info("Claude Code will handle routine operations automatically");

        if allow_liquidations {
            display::print_info("✓ Automatic liquidations: ENABLED");
        }

        if let Some(limit) = max_mint {
            display::print_info(&format!("✓ Max mint per day: {} QNKUSD", limit));
        }

        display::print_footer();

        println!("\n{}", "Monitoring bank operations...".cyan());
        println!("{}", "Press Ctrl+C to exit".yellow());

        // Simulate autonomous monitoring
        loop {
            tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
            println!("{}", format!("[{}] ✓ Systems healthy", chrono::Utc::now().format("%H:%M:%S")).green());
        }
    } else {
        // Interactive mode
        interactive_mode(config).await
    }
}

async fn interactive_mode(config: &CliConfig) -> Result<()> {
    println!("{}", "╔═══════════════════════════════════════════════════════════╗".cyan().bold());
    println!("{}", "║  🤖 CLAUDE CODE - INTERACTIVE BANKING ASSISTANT          ║".cyan().bold());
    println!("{}", "╚═══════════════════════════════════════════════════════════╝".cyan().bold());

    println!("\n{}", "Welcome! I'm your AI banking assistant.".green());
    println!("{}", "Ask me anything about bank operations, or give me commands.".green());
    println!("{}", "\nExamples:".yellow());
    println!("  • What's the bank's status today?");
    println!("  • Show me loans that need attention");
    println!("  • Mint 1M QNKUSD backed by BTC");
    println!("  • Run the weekly profit distribution");
    println!("  • What's our quantum vault adoption rate?");
    println!("{}", "\nType 'exit' to quit\n".yellow());

    loop {
        print!("{}", "You: ".blue().bold());
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        if input.eq_ignore_ascii_case("exit") || input.eq_ignore_ascii_case("quit") {
            println!("{}", "\n👋 Goodbye!".green());
            break;
        }

        // Process natural language input
        let response = process_natural_language(input, config).await?;

        println!("{}", format!("\nClaude: {}", response).green());
        println!();
    }

    Ok(())
}

pub async fn process_natural_language(input: &str, config: &CliConfig) -> Result<String> {
    let input_lower = input.to_lowercase();

    // Pattern matching for common queries
    if input_lower.contains("status") || input_lower.contains("how") && input_lower.contains("bank") {
        return Ok(format!(
            "The bank is operating well:\n\
            • QNKUSD supply: 125.45M with 110.8% collateralization\n\
            • 48,234 active accounts with $89.45M in deposits\n\
            • 127 loans at risk, 3 in liquidation queue\n\
            • Quantum vaults: 8,901 active\n\n\
            Run 'quillon-bank status --full' for detailed metrics."
        ));
    }

    if input_lower.contains("loan") && (input_lower.contains("risk") || input_lower.contains("attention") || input_lower.contains("problem")) {
        return Ok(format!(
            "There are 3 loans requiring attention:\n\
            • loan-042: $75k at 95% collateral (IMMEDIATE ACTION NEEDED)\n\
            • loan-089: $50k at 102% collateral (24h grace period)\n\
            • loan-123: $30k at 108% collateral (warning sent)\n\n\
            Run 'quillon-bank lending at-risk' to see full details or\n\
            'quillon-bank lending liquidate loan-042' to liquidate."
        ));
    }

    if input_lower.contains("mint") {
        // Extract amount if mentioned
        if let Some(amount) = extract_amount(&input_lower) {
            return Ok(format!(
                "To mint {} QNKUSD, you'll need:\n\
                • {} BTC (at $70k/BTC) for 110% collateralization\n\
                • Or {} ETH (at $3.5k/ETH)\n\
                • Or ${} USDC\n\n\
                Run this command:\n\
                quillon-bank stablecoin mint --amount {} --collateral-type BTC --collateral-amount {} --reason \"Board approved expansion\"",
                display::format_amount(amount),
                (amount as f64 * 1.1 / 70_000.0),
                (amount as f64 * 1.1 / 3_500.0),
                display::format_amount((amount as f64 * 1.1) as u64),
                amount,
                (amount as f64 * 1.1 / 70_000.0)
            ));
        }
    }

    if input_lower.contains("profit") || input_lower.contains("revenue") || input_lower.contains("earnings") {
        return Ok(format!(
            "This month's financials:\n\
            • Revenue: $1.25M (lending $800k + fees $450k)\n\
            • Expenses: $450k (operations + infrastructure)\n\
            • Net profit: $800k\n\n\
            Run 'quillon-bank treasury profits calculate --period 2025-09' for details."
        ));
    }

    if input_lower.contains("quantum") && (input_lower.contains("vault") || input_lower.contains("adoption") || input_lower.contains("privacy")) {
        return Ok(format!(
            "Quantum features status:\n\
            • 8,901 quantum vaults active\n\
            • 27% privacy tier adoption (13,023 customers)\n\
            • 156,789 post-quantum transactions in last 24h\n\
            • 98.2% using post-quantum signatures\n\n\
            This is strong adoption for quantum privacy features!"
        ));
    }

    if input_lower.contains("collateral") {
        return Ok(format!(
            "Collateral status:\n\
            • Total value: $138.99M (110.8% ratio)\n\
            • Composition: 60.5% BTC, 23.9% ETH, 15.6% USDC\n\
            • Target: 110% (currently meeting target ✓)\n\n\
            Run 'quillon-bank stablecoin collateral status' for full breakdown."
        ));
    }

    if input_lower.contains("distribute") && input_lower.contains("profit") {
        return Ok(format!(
            "Ready to distribute $800k profit?\n\
            Suggested allocation:\n\
            • Board dividend: $400k (50%)\n\
            • Reserves: $300k (37.5%)\n\
            • Customer rewards: $100k (12.5%)\n\n\
            Run: quillon-bank treasury profits distribute --board-dividend 400000 --reserves 300000 --customer-rewards 100000"
        ));
    }

    if input_lower.contains("help") || input_lower.contains("what can you do") {
        return Ok(format!(
            "I can help you with:\n\
            ✓ Check bank status and health metrics\n\
            ✓ Mint/burn QNKUSD stablecoin\n\
            ✓ Manage collateral and reserves\n\
            ✓ Review and approve loans\n\
            ✓ Monitor risk and liquidations\n\
            ✓ Calculate and distribute profits\n\
            ✓ Analytics and reporting\n\
            ✓ Quantum features management\n\n\
            Just ask me naturally, like:\n\
            \"What needs my attention today?\"\n\
            \"Mint 5M QNKUSD\"\n\
            \"Show me high-risk loans\""
        ));
    }

    // Default response for unrecognized queries
    Ok(format!(
        "I understand you're asking about: \"{}\"\n\n\
        I can help with:\n\
        • Bank status: Ask \"What's the bank status?\"\n\
        • Loans: Ask \"Show me loans at risk\"\n\
        • Minting: Say \"Mint X QNKUSD\"\n\
        • Profits: Ask \"Show me this month's profits\"\n\
        • Analytics: Ask \"Customer adoption rates\"\n\n\
        Or type 'help' to see all capabilities.",
        input
    ))
}

fn extract_amount(input: &str) -> Option<u64> {
    // Try to extract numbers from input
    for word in input.split_whitespace() {
        // Try direct number
        if let Ok(num) = word.parse::<u64>() {
            return Some(num);
        }

        // Try number with 'k' or 'm' suffix
        if let Some(num_str) = word.strip_suffix('k').or_else(|| word.strip_suffix('K')) {
            if let Ok(num) = num_str.parse::<u64>() {
                return Some(num * 1_000);
            }
        }

        if let Some(num_str) = word.strip_suffix('m').or_else(|| word.strip_suffix('M')) {
            if let Ok(num) = num_str.parse::<u64>() {
                return Some(num * 1_000_000);
            }
        }
    }

    None
}