/// Initialize CLI configuration

use anyhow::Result;
use clap::ArgMatches;
use colored::*;

use crate::auth::AuthManager;
use crate::config::CliConfig;
use crate::display;

pub async fn execute(matches: &ArgMatches, config: &CliConfig) -> Result<()> {
    display::print_header("QUILLON BANK CLI INITIALIZATION");

    let is_board_member = matches.get_flag("board-member");
    let generate_keys = matches.get_flag("generate-keys");

    if is_board_member {
        println!("{}", "Initializing as board member...".cyan());
    }

    // Create configuration directory
    let config_dir = CliConfig::config_path()?.parent().unwrap().to_path_buf();
    std::fs::create_dir_all(&config_dir)?;

    display::print_success(&format!("Created config directory: {}", config_dir.display()));

    // Generate keys if requested
    if generate_keys {
        let auth_manager = AuthManager::new(config.clone());
        auth_manager.generate_keys()?;
    }

    // Save default configuration
    let default_config = CliConfig::default();
    default_config.save()?;

    display::print_success(&format!("Created configuration file: {}", CliConfig::config_path()?.display()));

    display::print_footer();

    println!("\n{}", "Next steps:".yellow().bold());
    println!("  1. Edit configuration: {}", CliConfig::config_path()?.display());
    println!("  2. Login: quillon-bank auth login");
    println!("  3. Check status: quillon-bank status");

    Ok(())
}