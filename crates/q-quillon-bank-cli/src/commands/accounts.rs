/// Account management commands

use anyhow::{Result, bail};
use clap::ArgMatches;

use crate::config::CliConfig;
use crate::display;

pub async fn execute(matches: &ArgMatches, config: &CliConfig) -> Result<()> {
    match matches.subcommand() {
        Some(("list", sub_matches)) => list(sub_matches, config).await,
        Some(("pending-approvals", _)) => pending_approvals(config).await,
        Some(("approve", sub_matches)) => approve(sub_matches, config).await,
        _ => bail!("Unknown accounts subcommand"),
    }
}

async fn list(matches: &ArgMatches, config: &CliConfig) -> Result<()> {
    display::print_header("ACCOUNTS LIST");

    // Mock high-value accounts
    let accounts = vec![
        ("acc-001", "Alice Corp", 1_500_000, "quantum"),
        ("acc-002", "Bob Ventures", 950_000, "enhanced"),
        ("acc-003", "Carol Industries", 750_000, "quantum"),
        ("acc-004", "Dave Holdings", 500_000, "standard"),
    ];

    let mut table = display::create_table(vec!["ID", "Name", "Balance", "Privacy Tier"]);

    for (id, name, balance, tier) in accounts {
        table.add_row(prettytable::row![
            id,
            name,
            display::format_currency(balance),
            tier
        ]);
    }

    table.printstd();

    display::print_info(&format!("Showing {} high-value accounts", 4));

    Ok(())
}

async fn pending_approvals(config: &CliConfig) -> Result<()> {
    display::print_header("PENDING ACCOUNT APPROVALS");

    let pending = vec![
        ("app-001", "New Tech LLC", "Standard", "Verified"),
        ("app-002", "Crypto Fund", "Enhanced", "Pending KYC"),
        ("app-003", "Privacy Corp", "Quantum", "Verified"),
    ];

    let mut table = display::create_table(vec!["ID", "Applicant", "Requested Tier", "Status"]);

    for (id, applicant, tier, status) in pending {
        table.add_row(prettytable::row![id, applicant, tier, status]);
    }

    table.printstd();

    display::print_info(&format!("{} pending approvals", 3));

    Ok(())
}

async fn approve(matches: &ArgMatches, config: &CliConfig) -> Result<()> {
    let account_id = matches.get_one::<String>("account-id").unwrap();
    let credit_limit = matches.get_one::<String>("credit-limit");
    let privacy_tier = matches.get_one::<String>("privacy-tier");

    display::print_header("APPROVE ACCOUNT");

    display::print_kv("Account ID", account_id);

    if let Some(limit) = credit_limit {
        display::print_kv("Credit Limit", &display::format_currency(limit.parse().unwrap_or(0)));
    }

    if let Some(tier) = privacy_tier {
        display::print_kv("Privacy Tier", tier);
    }

    display::print_success("Account approved successfully");
    display::print_footer();

    Ok(())
}