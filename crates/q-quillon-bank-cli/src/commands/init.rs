/// Initialize CLI configuration with AEGIS-QL post-quantum security

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
    let is_founder = is_board_member; // For backward compatibility

    if is_founder {
        println!("{}", "🏦 Initializing as FOUNDER (CEO/Board Chair)...".cyan().bold());
        println!("{}", "   Post-quantum security with AEGIS-QL + ZK-STARK".cyan());
    }

    // Create configuration directory
    let config_dir = CliConfig::config_path()?.parent().unwrap().to_path_buf();
    std::fs::create_dir_all(&config_dir)?;

    display::print_success(&format!("Created config directory: {}", config_dir.display()));

    // Generate keys if requested
    if generate_keys {
        let auth_manager = AuthManager::new(config.clone());

        if is_founder {
            // Generate AEGIS-QL keys with ZK-STARK trustless setup
            println!("\n{}", "🔐 Generating AEGIS-QL keys with ZK-STARK proof...".yellow().bold());
            let (_wallet, _pubkey, _seckey, _proof) = auth_manager.generate_aegis_keys()?;

            println!("\n{}", "✅ AEGIS-QL keypair generation complete!".green().bold());
            println!("{}", "   These keys provide post-quantum security for bank administration.".green());
            println!("{}", "   The ZK-STARK proof enables trustless key verification.".green());
        } else {
            // Generate classical Ed25519 keys (legacy)
            auth_manager.generate_keys()?;
        }
    }

    // Save default configuration
    let default_config = CliConfig::default();
    default_config.save()?;

    display::print_success(&format!("Created configuration file: {}", CliConfig::config_path()?.display()));

    display::print_footer();

    println!("\n{}", "Next steps:".yellow().bold());

    if is_founder && generate_keys {
        println!("  1. Verify your wallet address matches the founder wallet");
        println!("  2. Login with AEGIS-QL: quillon-bank auth login --role founder");
        println!("  3. Check bank status: quillon-bank status --full");
        println!("  4. Mint QUGUSD: quillon-bank stablecoin mint --amount 100000 --collateral-type QUG --collateral-amount 50");
        println!("\n{}", "🔒 Your private key is stored securely with 0600 permissions".yellow());
        println!("{}", "🔒 Keep your key safe - only you can sign bank operations!".yellow());
    } else {
        println!("  1. Edit configuration: {}", CliConfig::config_path()?.display());
        println!("  2. Login: quillon-bank auth login");
        println!("  3. Check status: quillon-bank status");
    }

    Ok(())
}