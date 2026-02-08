//! Quillon Bank CLI - Bank Administration Tool
//!
//! v3.9.1-beta: Command-line interface for bank administrators to:
//! - View and respond to user messages
//! - Manage loan applications
//! - Issue and approve death certificates
//! - Execute inheritance transfers
//!
//! Usage:
//!   quillon-bank-cli messages list                    # List all user messages
//!   quillon-bank-cli messages respond <wallet> <msg>  # Respond to a user
//!   quillon-bank-cli messages unread                  # Show unread messages
//!   quillon-bank-cli loans list                       # List all loan applications
//!   quillon-bank-cli loans approve <loan_id>          # Approve a loan
//!   quillon-bank-cli identity list                    # List all identities
//!   quillon-bank-cli identity approve <wallet>        # Approve identity verification
//!   quillon-bank-cli death-cert list                  # List death certificates
//!   quillon-bank-cli death-cert approve <cert_id>     # Approve death certificate
//!   quillon-bank-cli inheritance execute <cert_id>    # Execute inheritance transfer

use clap::{Parser, Subcommand};
use colored::Colorize;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::io::{self, Write};

const DEFAULT_API_URL: &str = "http://localhost:8080";

#[derive(Parser)]
#[command(name = "quillon-bank-cli")]
#[command(about = "Quillon Bank Administration CLI", long_about = None)]
#[command(version = "3.9.1-beta")]
struct Cli {
    /// API server URL
    #[arg(short, long, default_value = DEFAULT_API_URL)]
    api_url: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Manage user messages
    Messages {
        #[command(subcommand)]
        action: MessageAction,
    },
    /// Manage loan applications
    Loans {
        #[command(subcommand)]
        action: LoanAction,
    },
    /// Manage user identities
    Identity {
        #[command(subcommand)]
        action: IdentityAction,
    },
    /// Manage death certificates
    DeathCert {
        #[command(subcommand)]
        action: DeathCertAction,
    },
    /// Execute inheritance transfers
    Inheritance {
        #[command(subcommand)]
        action: InheritanceAction,
    },
}

#[derive(Subcommand)]
enum MessageAction {
    /// List all messages (optionally filter by wallet)
    List {
        /// Filter by wallet address
        #[arg(short, long)]
        wallet: Option<String>,
    },
    /// Show unread messages from users
    Unread,
    /// Respond to a user message
    Respond {
        /// User's wallet address
        wallet: String,
        /// Response message (use quotes for multi-word)
        message: String,
        /// Optional subject line
        #[arg(short, long)]
        subject: Option<String>,
    },
}

#[derive(Subcommand)]
enum LoanAction {
    /// List all loan applications
    List,
    /// Show pending loans only
    Pending,
    /// Approve a loan application
    Approve { loan_id: String },
    /// Reject a loan application
    Reject {
        loan_id: String,
        /// Rejection reason
        #[arg(short, long)]
        reason: Option<String>,
    },
}

#[derive(Subcommand)]
enum IdentityAction {
    /// List all registered identities
    List,
    /// Show pending identity verifications
    Pending,
    /// Approve identity verification
    Approve { wallet: String },
    /// View identity details
    View { wallet: String },
}

#[derive(Subcommand)]
enum DeathCertAction {
    /// List all death certificates
    List,
    /// Show pending certificates
    Pending,
    /// Approve a death certificate
    Approve { cert_id: String },
    /// View certificate details
    View { cert_id: String },
}

#[derive(Subcommand)]
enum InheritanceAction {
    /// View inheritance info for a wallet
    View { wallet: String },
    /// Execute approved inheritance transfer
    Execute { cert_id: String },
}

// ============================================================================
// API Response Types
// ============================================================================

#[derive(Debug, Deserialize)]
struct ApiResponse<T> {
    success: bool,
    data: Option<T>,
    error: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
struct BankMessage {
    id: String,
    from: String,
    wallet_address: String,
    content: String,
    subject: Option<String>,
    loan_id: Option<String>,
    timestamp: i64,
    read: bool,
}

#[derive(Debug, Deserialize)]
struct LoanApplication {
    id: String,
    borrower_address: String,
    amount: f64,
    collateral: f64,
    interest_rate: f64,
    status: String,
    created_at: i64,
}

#[derive(Debug, Deserialize)]
struct UserIdentity {
    wallet_address: String,
    display_name: Option<String>,
    created_at: i64,
    verified: bool,
    kyc_level: u8,
    is_deceased: bool,
    beneficiary_address: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DeathCertificate {
    id: String,
    deceased_wallet: String,
    beneficiary_wallet: String,
    issued_at: i64,
    approved: bool,
    approved_by: Option<String>,
    executed: bool,
    reason: String,
}

#[derive(Debug, Deserialize)]
struct InheritanceInfo {
    deceased_wallet: String,
    beneficiary_wallet: String,
    total_balance: u128,
    transfer_ready: bool,
}

// ============================================================================
// Main
// ============================================================================

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let client = Client::new();
    let api_url = cli.api_url;

    println!("{}", "╔════════════════════════════════════════════════════════════╗".cyan());
    println!("{}", "║             🏦 Quillon Bank Administration CLI             ║".cyan());
    println!("{}", "║                     v3.9.1-beta                             ║".cyan());
    println!("{}", "╚════════════════════════════════════════════════════════════╝".cyan());
    println!();

    match cli.command {
        Commands::Messages { action } => handle_messages(&client, &api_url, action).await?,
        Commands::Loans { action } => handle_loans(&client, &api_url, action).await?,
        Commands::Identity { action } => handle_identity(&client, &api_url, action).await?,
        Commands::DeathCert { action } => handle_death_cert(&client, &api_url, action).await?,
        Commands::Inheritance { action } => handle_inheritance(&client, &api_url, action).await?,
    }

    Ok(())
}

// ============================================================================
// Message Handlers
// ============================================================================

async fn handle_messages(
    client: &Client,
    api_url: &str,
    action: MessageAction,
) -> Result<(), Box<dyn std::error::Error>> {
    match action {
        MessageAction::List { wallet } => {
            println!("{}", "📬 Bank Messages".green().bold());
            println!("{}", "─".repeat(60));

            let url = if let Some(ref w) = wallet {
                format!("{}/api/v1/quillon-bank/messages/{}", api_url, w)
            } else {
                format!("{}/api/v1/quillon-bank/messages/admin/list", api_url)
            };

            let response: Vec<BankMessage> = client.get(&url).send().await?.json().await?;

            if response.is_empty() {
                println!("{}", "No messages found.".yellow());
            } else {
                for msg in response {
                    let from_label = if msg.from == "user" {
                        "👤 User".blue()
                    } else {
                        "🏦 Bank".green()
                    };
                    let read_status = if msg.read { "✓".green() } else { "●".red() };
                    let time = chrono::DateTime::from_timestamp_millis(msg.timestamp)
                        .map(|t| t.format("%Y-%m-%d %H:%M").to_string())
                        .unwrap_or_else(|| "Unknown".to_string());

                    println!();
                    println!("{} {} [{}] {}", read_status, from_label, time, msg.id.dimmed());
                    println!("  Wallet: {}", msg.wallet_address.yellow());
                    if let Some(subject) = &msg.subject {
                        println!("  Subject: {}", subject.cyan());
                    }
                    println!("  Message: {}", msg.content);
                }
            }
        }

        MessageAction::Unread => {
            println!("{}", "📭 Unread Messages from Users".green().bold());
            println!("{}", "─".repeat(60));

            let url = format!("{}/api/v1/quillon-bank/messages/admin/list", api_url);
            let response: Vec<BankMessage> = client.get(&url).send().await?.json().await?;

            let unread: Vec<_> = response
                .into_iter()
                .filter(|m| !m.read && m.from == "user")
                .collect();

            if unread.is_empty() {
                println!("{}", "No unread messages! 🎉".green());
            } else {
                println!("{} unread message(s):", unread.len().to_string().red().bold());
                for msg in unread {
                    let time = chrono::DateTime::from_timestamp_millis(msg.timestamp)
                        .map(|t| t.format("%Y-%m-%d %H:%M").to_string())
                        .unwrap_or_else(|| "Unknown".to_string());

                    println!();
                    println!("{} [{}]", "● NEW".red().bold(), time);
                    println!("  From: {}", msg.wallet_address.yellow());
                    if let Some(subject) = &msg.subject {
                        println!("  Subject: {}", subject.cyan());
                    }
                    println!("  Message: {}", msg.content);
                }
            }
        }

        MessageAction::Respond { wallet, message, subject } => {
            println!("{}", "📤 Sending Bank Response".green().bold());
            println!("{}", "─".repeat(60));
            println!("To: {}", wallet.yellow());
            println!("Message: {}", message);

            let url = format!("{}/api/v1/quillon-bank/messages/admin/respond", api_url);
            let body = serde_json::json!({
                "wallet_address": wallet,
                "content": message,
                "subject": subject,
            });

            let response: ApiResponse<BankMessage> = client
                .post(&url)
                .json(&body)
                .send()
                .await?
                .json()
                .await?;

            if response.success {
                println!();
                println!("{}", "✅ Message sent successfully!".green().bold());
                if let Some(msg) = response.data {
                    println!("Message ID: {}", msg.id);
                }
            } else {
                println!("{}", format!("❌ Failed: {}", response.error.unwrap_or_default()).red());
            }
        }
    }

    Ok(())
}

// ============================================================================
// Loan Handlers
// ============================================================================

async fn handle_loans(
    client: &Client,
    api_url: &str,
    action: LoanAction,
) -> Result<(), Box<dyn std::error::Error>> {
    match action {
        LoanAction::List | LoanAction::Pending => {
            let show_pending_only = matches!(action, LoanAction::Pending);
            println!(
                "{}",
                if show_pending_only {
                    "📋 Pending Loan Applications".green().bold()
                } else {
                    "📋 All Loan Applications".green().bold()
                }
            );
            println!("{}", "─".repeat(60));

            let url = format!("{}/api/v1/quillon-bank/lending/applications", api_url);
            let response: Vec<LoanApplication> = client.get(&url).send().await?.json().await?;

            let loans: Vec<_> = if show_pending_only {
                response.into_iter().filter(|l| l.status == "pending").collect()
            } else {
                response
            };

            if loans.is_empty() {
                println!("{}", "No loan applications found.".yellow());
            } else {
                for loan in loans {
                    let status_color = match loan.status.as_str() {
                        "approved" | "active" => loan.status.green(),
                        "pending" => loan.status.yellow(),
                        "rejected" | "liquidated" => loan.status.red(),
                        _ => loan.status.normal(),
                    };

                    println!();
                    println!("Loan ID: {} [{}]", loan.id.cyan(), status_color);
                    println!("  Borrower: {}", loan.borrower_address.yellow());
                    println!("  Amount: {} QUG", loan.amount);
                    println!("  Collateral: {} QUG", loan.collateral);
                    println!("  Interest: {}%", loan.interest_rate);
                }
            }
        }

        LoanAction::Approve { loan_id } => {
            println!("{}", "✅ Approving Loan".green().bold());
            println!("Loan ID: {}", loan_id.cyan());

            let url = format!("{}/api/v1/quillon-bank/lending/approve", api_url);
            let body = serde_json::json!({ "loan_id": loan_id });

            let response: ApiResponse<serde_json::Value> = client
                .post(&url)
                .json(&body)
                .send()
                .await?
                .json()
                .await?;

            if response.success {
                println!("{}", "✅ Loan approved successfully!".green().bold());
            } else {
                println!("{}", format!("❌ Failed: {}", response.error.unwrap_or_default()).red());
            }
        }

        LoanAction::Reject { loan_id, reason } => {
            println!("{}", "❌ Rejecting Loan".red().bold());
            println!("Loan ID: {}", loan_id.cyan());
            if let Some(ref r) = reason {
                println!("Reason: {}", r);
            }

            // Note: Rejection endpoint would need to be added to API
            println!("{}", "⚠️  Loan rejection endpoint not yet implemented.".yellow());
        }
    }

    Ok(())
}

// ============================================================================
// Identity Handlers
// ============================================================================

async fn handle_identity(
    client: &Client,
    api_url: &str,
    action: IdentityAction,
) -> Result<(), Box<dyn std::error::Error>> {
    match action {
        IdentityAction::List | IdentityAction::Pending => {
            let show_pending_only = matches!(action, IdentityAction::Pending);
            println!(
                "{}",
                if show_pending_only {
                    "🪪 Pending Identity Verifications".green().bold()
                } else {
                    "🪪 All Registered Identities".green().bold()
                }
            );
            println!("{}", "─".repeat(60));

            // Note: Would need a list endpoint
            println!("{}", "⚠️  Identity list endpoint not yet implemented.".yellow());
            println!("Use: quillon-bank-cli identity view <wallet> to view specific identity");
        }

        IdentityAction::Approve { wallet } => {
            println!("{}", "✅ Approving Identity".green().bold());
            println!("Wallet: {}", wallet.yellow());

            let url = format!("{}/api/v1/quillon-bank/identity/admin/approve", api_url);
            let response: ApiResponse<bool> = client
                .post(&url)
                .json(&wallet)
                .send()
                .await?
                .json()
                .await?;

            if response.success && response.data == Some(true) {
                println!("{}", "✅ Identity verified successfully!".green().bold());
            } else {
                println!("{}", format!("❌ Failed: {}", response.error.unwrap_or("Identity not found".to_string())).red());
            }
        }

        IdentityAction::View { wallet } => {
            println!("{}", "🪪 Identity Details".green().bold());
            println!("{}", "─".repeat(60));

            let url = format!("{}/api/v1/quillon-bank/identity/{}", api_url, wallet);
            let response: ApiResponse<Option<UserIdentity>> = client.get(&url).send().await?.json().await?;

            if let Some(Some(identity)) = response.data {
                println!("Wallet: {}", identity.wallet_address.yellow());
                if let Some(name) = &identity.display_name {
                    println!("Display Name: {}", name);
                }
                println!(
                    "Verified: {}",
                    if identity.verified { "✅ Yes".green() } else { "❌ No".red() }
                );
                println!("KYC Level: {}", identity.kyc_level);
                println!(
                    "Deceased: {}",
                    if identity.is_deceased { "💀 Yes".red() } else { "No".normal() }
                );
                if let Some(beneficiary) = &identity.beneficiary_address {
                    println!("Beneficiary: {}", beneficiary);
                }
            } else {
                println!("{}", "Identity not found.".yellow());
            }
        }
    }

    Ok(())
}

// ============================================================================
// Death Certificate Handlers
// ============================================================================

async fn handle_death_cert(
    client: &Client,
    api_url: &str,
    action: DeathCertAction,
) -> Result<(), Box<dyn std::error::Error>> {
    match action {
        DeathCertAction::List | DeathCertAction::Pending => {
            let show_pending_only = matches!(action, DeathCertAction::Pending);
            println!(
                "{}",
                if show_pending_only {
                    "💀 Pending Death Certificates".green().bold()
                } else {
                    "💀 All Death Certificates".green().bold()
                }
            );
            println!("{}", "─".repeat(60));

            // Note: Would need a list endpoint
            println!("{}", "⚠️  Death certificate list endpoint not yet implemented.".yellow());
        }

        DeathCertAction::Approve { cert_id } => {
            println!("{}", "✅ Approving Death Certificate".green().bold());
            println!("Certificate ID: {}", cert_id.cyan());

            let url = format!("{}/api/v1/quillon-bank/identity/admin/death-certificate/approve", api_url);
            let response: ApiResponse<bool> = client
                .post(&url)
                .json(&cert_id)
                .send()
                .await?
                .json()
                .await?;

            if response.success && response.data == Some(true) {
                println!("{}", "✅ Death certificate approved!".green().bold());
                println!("The deceased's identity has been marked accordingly.");
                println!("Run 'inheritance execute {}' to transfer assets.", cert_id);
            } else {
                println!("{}", format!("❌ Failed: {}", response.error.unwrap_or("Certificate not found".to_string())).red());
            }
        }

        DeathCertAction::View { cert_id } => {
            println!("{}", "💀 Death Certificate Details".green().bold());
            println!("{}", "─".repeat(60));
            println!("Certificate ID: {}", cert_id.cyan());
            println!("{}", "⚠️  Certificate view endpoint not yet implemented.".yellow());
        }
    }

    Ok(())
}

// ============================================================================
// Inheritance Handlers
// ============================================================================

async fn handle_inheritance(
    client: &Client,
    api_url: &str,
    action: InheritanceAction,
) -> Result<(), Box<dyn std::error::Error>> {
    match action {
        InheritanceAction::View { wallet } => {
            println!("{}", "💰 Inheritance Information".green().bold());
            println!("{}", "─".repeat(60));

            let url = format!("{}/api/v1/quillon-bank/identity/inheritance/{}", api_url, wallet);
            let response: ApiResponse<Option<InheritanceInfo>> = client.get(&url).send().await?.json().await?;

            if let Some(Some(info)) = response.data {
                println!("Deceased Wallet: {}", info.deceased_wallet.red());
                println!("Beneficiary Wallet: {}", info.beneficiary_wallet.green());
                println!("Total Balance: {} QUG", info.total_balance as f64 / 1e24);
                println!(
                    "Transfer Ready: {}",
                    if info.transfer_ready { "✅ Yes".green() } else { "❌ No (pending approval)".yellow() }
                );
            } else {
                println!("{}", "No inheritance information found for this wallet.".yellow());
            }
        }

        InheritanceAction::Execute { cert_id } => {
            println!("{}", "💰 Executing Inheritance Transfer".green().bold());
            println!("{}", "─".repeat(60));
            println!("Certificate ID: {}", cert_id.cyan());

            // Confirmation prompt
            print!("Are you sure you want to execute this inheritance transfer? (yes/no): ");
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;

            if input.trim().to_lowercase() != "yes" {
                println!("{}", "Transfer cancelled.".yellow());
                return Ok(());
            }

            let url = format!("{}/api/v1/quillon-bank/identity/admin/transfer", api_url);
            let response: ApiResponse<String> = client
                .post(&url)
                .json(&cert_id)
                .send()
                .await?
                .json()
                .await?;

            if response.success {
                println!();
                println!("{}", "✅ Inheritance transfer executed successfully!".green().bold());
                if let Some(msg) = response.data {
                    println!("{}", msg);
                }
            } else {
                println!("{}", format!("❌ Failed: {}", response.error.unwrap_or_default()).red());
            }
        }
    }

    Ok(())
}
