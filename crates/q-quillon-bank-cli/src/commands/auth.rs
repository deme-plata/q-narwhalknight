/// Authentication commands

use anyhow::{Result, bail};
use clap::ArgMatches;
use colored::*;

use crate::auth::{AuthManager, AuthSession};
use crate::client::QuilonBankClient;
use crate::config::CliConfig;
use crate::display;

pub async fn execute(matches: &ArgMatches, config: &CliConfig) -> Result<()> {
    match matches.subcommand() {
        Some(("login", sub_matches)) => login(sub_matches, config).await,
        Some(("status", _)) => status(config).await,
        Some(("logout", _)) => logout(config).await,
        _ => {
            bail!("Unknown auth subcommand");
        }
    }
}

async fn login(matches: &ArgMatches, config: &CliConfig) -> Result<()> {
    display::print_header("BOARD MEMBER LOGIN");

    let auth_manager = AuthManager::new(config.clone());

    // Load keypair
    let keypair = auth_manager.load_keypair()?;
    display::print_success("Loaded authentication key");

    // Check MFA if enabled
    if config.board.mfa_enabled {
        let mfa_token = matches.get_one::<String>("mfa-token");

        if let Some(token) = mfa_token {
            if !auth_manager.verify_mfa(token)? {
                bail!("Invalid MFA token");
            }
            display::print_success("MFA verified");
        } else {
            bail!("MFA token required (use --mfa-token)");
        }
    }

    // Request authentication challenge from server
    let client = QuilonBankClient::new(config)?;

    // For now, create a mock session
    // TODO: Implement full authentication flow with server
    let verifying_key = keypair.verifying_key();
    let key_bytes = verifying_key.to_bytes();
    let token = format!("board-token-{:02x}{:02x}{:02x}{:02x}",
        key_bytes[0], key_bytes[1], key_bytes[2], key_bytes[3]);

    let session = AuthSession {
        member_id: config.board.member_id.clone(),
        role: "board".to_string(),
        token,
        expires_at: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs() + 86400, // 24 hours
    };

    auth_manager.save_session(&session)?;

    display::print_success(&format!("Authenticated as: {}", session.member_id));
    display::print_info(&format!("Session expires in 24 hours"));

    display::print_footer();

    Ok(())
}

async fn status(config: &CliConfig) -> Result<()> {
    display::print_header("AUTHENTICATION STATUS");

    let auth_manager = AuthManager::new(config.clone());

    if let Some(session) = auth_manager.load_session()? {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs();

        let remaining = session.expires_at.saturating_sub(now);
        let remaining_hours = remaining / 3600;

        display::print_kv("Member ID", &session.member_id);
        display::print_kv("Role", &session.role);
        display::print_kv("Status", "Authenticated ✓");
        display::print_kv("Expires in", &format!("{} hours", remaining_hours));
    } else {
        display::print_warning("Not authenticated");
        println!("\n  Run: quillon-bank auth login");
    }

    display::print_footer();

    Ok(())
}

async fn logout(config: &CliConfig) -> Result<()> {
    let auth_manager = AuthManager::new(config.clone());
    auth_manager.clear_session()?;

    display::print_success("Logged out successfully");

    Ok(())
}