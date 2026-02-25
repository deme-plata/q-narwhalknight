//! Quillon Bank CLI - Bank Administration Tool
//!
//! v8.0.0: Command-line interface for bank administrators to:
//! - View and respond to user messages
//! - Manage loan applications (approve/reject)
//! - Manage user identities
//! - Issue and approve death certificates
//! - Execute inheritance transfers
//!
//! Usage:
//!   quillon-bank-cli messages list                    # List all user messages
//!   quillon-bank-cli messages respond <wallet> <msg>  # Respond to a user
//!   quillon-bank-cli messages unread                  # Show unread messages
//!   quillon-bank-cli loans list                       # List all loan applications
//!   quillon-bank-cli loans approve <loan_id>          # Approve a loan
//!   quillon-bank-cli loans reject <loan_id>           # Reject a loan
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
#[command(version = "8.0.0")]
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
    /// Broadcast emails to all email-registered users
    Email {
        #[command(subcommand)]
        action: EmailAction,
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

#[derive(Subcommand)]
enum EmailAction {
    /// Broadcast an email to all email-registered users
    Broadcast {
        /// Email subject line
        #[arg(short, long)]
        subject: String,
        /// Email body text
        #[arg(short, long)]
        body: String,
    },
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

/// Bank message - the `from` field is an enum serialized as "user" or "bank"
#[derive(Debug, Deserialize, Serialize)]
struct BankMessage {
    id: String,
    from: serde_json::Value, // Server sends enum {"user"} or {"bank"}, handle flexibly
    wallet_address: String,
    content: String,
    subject: Option<String>,
    loan_id: Option<String>,
    timestamp: i64,
    read: bool,
}

impl BankMessage {
    fn from_str(&self) -> &str {
        match &self.from {
            serde_json::Value::String(s) => s.as_str(),
            _ => "unknown",
        }
    }
}

/// Loan application matching server's response format
#[derive(Debug, Deserialize)]
struct LoanApplication {
    loan_id: String,
    borrower_address: String,
    #[serde(deserialize_with = "deserialize_loan_amount")]
    loan_amount: f64,
    collateral_amount: f64,
    #[serde(default)]
    collateral_type: String,
    #[serde(default)]
    term_months: u32,
    interest_rate: f64,
    #[serde(default)]
    monthly_payment: f64,
    status: String,
    created_at: i64,
}

/// Flexible u128/string/number deserializer for loan amounts
fn deserialize_loan_amount<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let v = serde_json::Value::deserialize(deserializer)?;
    match v {
        serde_json::Value::Number(n) => Ok(n.as_f64().unwrap_or(0.0)),
        serde_json::Value::String(s) => s.parse::<f64>().map_err(serde::de::Error::custom),
        _ => Ok(0.0),
    }
}

/// Wrapper for loan applications response: {"applications": [...]}
#[derive(Debug, Deserialize)]
struct LoanApplicationsData {
    applications: Vec<LoanApplication>,
}

#[derive(Debug, Deserialize)]
struct UserIdentity {
    wallet_address: String,
    display_name: Option<String>,
    #[serde(default)]
    email_hash: Option<String>,
    created_at: i64,
    verified: bool,
    kyc_level: u8,
    is_deceased: bool,
    #[serde(default)]
    death_certificate_id: Option<String>,
    beneficiary_address: Option<String>,
    #[serde(default)]
    last_active: i64,
}

#[derive(Debug, Deserialize)]
struct DeathCertificate {
    id: String,
    deceased_wallet: String,
    beneficiary_wallet: String,
    issued_at: i64,
    approved: bool,
    approved_by: Option<String>,
    #[serde(default)]
    approved_at: Option<i64>,
    executed: bool,
    #[serde(default)]
    executed_at: Option<i64>,
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
// Helper: Build admin client with X-Admin-Local header
// ============================================================================

/// Send a GET request with localhost admin bypass header
async fn admin_get(client: &Client, url: &str) -> Result<reqwest::Response, reqwest::Error> {
    client
        .get(url)
        .header("X-Admin-Local", "true")
        .send()
        .await
}

/// Send a POST request with localhost admin bypass header
async fn admin_post(
    client: &Client,
    url: &str,
    body: &serde_json::Value,
) -> Result<reqwest::Response, reqwest::Error> {
    client
        .post(url)
        .header("X-Admin-Local", "true")
        .json(body)
        .send()
        .await
}

/// Send a POST request with localhost admin bypass header (string body for Json<String>)
async fn admin_post_string(
    client: &Client,
    url: &str,
    body: &str,
) -> Result<reqwest::Response, reqwest::Error> {
    client
        .post(url)
        .header("X-Admin-Local", "true")
        .header("Content-Type", "application/json")
        .body(format!("\"{}\"", body))
        .send()
        .await
}

// ============================================================================
// Main
// ============================================================================

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let client = Client::new();
    let api_url = cli.api_url;

    println!(
        "{}",
        "╔════════════════════════════════════════════════════════════╗".cyan()
    );
    println!(
        "{}",
        "║             🏦 Quillon Bank Administration CLI             ║".cyan()
    );
    println!(
        "{}",
        "║                        v8.0.0                              ║".cyan()
    );
    println!(
        "{}",
        "╚════════════════════════════════════════════════════════════╝".cyan()
    );
    println!();

    match cli.command {
        Commands::Messages { action } => handle_messages(&client, &api_url, action).await?,
        Commands::Loans { action } => handle_loans(&client, &api_url, action).await?,
        Commands::Identity { action } => handle_identity(&client, &api_url, action).await?,
        Commands::DeathCert { action } => handle_death_cert(&client, &api_url, action).await?,
        Commands::Inheritance { action } => handle_inheritance(&client, &api_url, action).await?,
        Commands::Email { action } => handle_email(&client, &api_url, action).await?,
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

            let response = admin_get(client, &url).await?;
            let messages: Vec<BankMessage> = response.json().await?;

            if messages.is_empty() {
                println!("{}", "No messages found.".yellow());
            } else {
                for msg in messages {
                    let from_str = msg.from_str();
                    let from_label = if from_str == "user" {
                        "👤 User".blue()
                    } else {
                        "🏦 Bank".green()
                    };
                    let read_status = if msg.read {
                        "✓".green()
                    } else {
                        "●".red()
                    };
                    let time = chrono::DateTime::from_timestamp_millis(msg.timestamp)
                        .map(|t| t.format("%Y-%m-%d %H:%M").to_string())
                        .unwrap_or_else(|| "Unknown".to_string());

                    println!();
                    println!(
                        "{} {} [{}] {}",
                        read_status,
                        from_label,
                        time,
                        msg.id.dimmed()
                    );
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
            let response = admin_get(client, &url).await?;
            let messages: Vec<BankMessage> = response.json().await?;

            let unread: Vec<_> = messages
                .into_iter()
                .filter(|m| !m.read && m.from_str() == "user")
                .collect();

            if unread.is_empty() {
                println!("{}", "No unread messages!".green());
            } else {
                println!(
                    "{} unread message(s):",
                    unread.len().to_string().red().bold()
                );
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

        MessageAction::Respond {
            wallet,
            message,
            subject,
        } => {
            println!("{}", "📤 Sending Bank Response".green().bold());
            println!("{}", "─".repeat(60));
            println!("To: {}", wallet.yellow());
            println!("Message: {}", message);

            let url = format!(
                "{}/api/v1/quillon-bank/messages/admin/respond",
                api_url
            );
            let body = serde_json::json!({
                "message_id": "",
                "wallet_address": wallet,
                "content": message,
                "subject": subject,
            });

            let response: ApiResponse<BankMessage> =
                admin_post(client, &url, &body).await?.json().await?;

            if response.success {
                println!();
                println!("{}", "✅ Message sent successfully!".green().bold());
                if let Some(msg) = response.data {
                    println!("Message ID: {}", msg.id);
                }
            } else {
                println!(
                    "{}",
                    format!("❌ Failed: {}", response.error.unwrap_or_default()).red()
                );
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
            // Server returns ApiResponse<{"applications": [...]}>
            let api_resp: ApiResponse<LoanApplicationsData> =
                client.get(&url).send().await?.json().await?;

            let all_loans = api_resp
                .data
                .map(|d| d.applications)
                .unwrap_or_default();

            let loans: Vec<_> = if show_pending_only {
                all_loans
                    .into_iter()
                    .filter(|l| l.status == "pending")
                    .collect()
            } else {
                all_loans
            };

            if loans.is_empty() {
                println!("{}", "No loan applications found.".yellow());
            } else {
                for loan in loans {
                    let status_color = match loan.status.as_str() {
                        "approved" | "active" | "paid" => loan.status.green(),
                        "pending" => loan.status.yellow(),
                        "rejected" | "liquidated" => loan.status.red(),
                        _ => loan.status.normal(),
                    };

                    let amount_display = loan.loan_amount / 1e24;
                    let time = chrono::DateTime::from_timestamp(loan.created_at, 0)
                        .map(|t| t.format("%Y-%m-%d %H:%M").to_string())
                        .unwrap_or_else(|| "Unknown".to_string());

                    println!();
                    println!(
                        "Loan ID: {} [{}]",
                        loan.loan_id.cyan(),
                        status_color
                    );
                    println!("  Borrower: {}", loan.borrower_address.yellow());
                    println!("  Amount: {:.4} QUGUSD", amount_display);
                    println!(
                        "  Collateral: {:.4} {} QUG",
                        loan.collateral_amount, loan.collateral_type
                    );
                    println!("  Interest: {:.2}%", loan.interest_rate);
                    println!("  Term: {} months", loan.term_months);
                    println!("  Monthly Payment: {:.4} QUGUSD", loan.monthly_payment);
                    println!("  Applied: {}", time);
                }
            }
        }

        LoanAction::Approve { loan_id } => {
            println!("{}", "✅ Approving Loan".green().bold());
            println!("Loan ID: {}", loan_id.cyan());

            let url = format!("{}/api/v1/quillon-bank/lending/approve", api_url);
            let body = serde_json::json!({ "loan_id": loan_id });

            let response: ApiResponse<serde_json::Value> =
                admin_post(client, &url, &body).await?.json().await?;

            if response.success {
                println!("{}", "✅ Loan approved successfully!".green().bold());
                if let Some(data) = &response.data {
                    if let Some(disbursed) = data.get("qugusd_disbursed") {
                        println!("  QUGUSD Disbursed: {}", disbursed);
                    }
                }
            } else {
                println!(
                    "{}",
                    format!("❌ Failed: {}", response.error.unwrap_or_default()).red()
                );
            }
        }

        LoanAction::Reject { loan_id, reason } => {
            println!("{}", "❌ Rejecting Loan".red().bold());
            println!("Loan ID: {}", loan_id.cyan());
            if let Some(ref r) = reason {
                println!("Reason: {}", r);
            }

            let url = format!("{}/api/v1/quillon-bank/lending/reject", api_url);
            let body = serde_json::json!({
                "loan_id": loan_id,
                "reason": reason.unwrap_or_else(|| "No reason provided".to_string()),
            });

            let response: ApiResponse<serde_json::Value> =
                admin_post(client, &url, &body).await?.json().await?;

            if response.success {
                println!("{}", "✅ Loan rejected successfully.".green().bold());
            } else {
                println!(
                    "{}",
                    format!("❌ Failed: {}", response.error.unwrap_or_default()).red()
                );
            }
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

            let url = format!(
                "{}/api/v1/quillon-bank/identity/admin/list",
                api_url
            );
            let api_resp: ApiResponse<Vec<UserIdentity>> =
                admin_get(client, &url).await?.json().await?;

            let all_identities = api_resp.data.unwrap_or_default();
            let identities: Vec<_> = if show_pending_only {
                all_identities
                    .into_iter()
                    .filter(|i| !i.verified)
                    .collect()
            } else {
                all_identities
            };

            if identities.is_empty() {
                println!("{}", "No identities found.".yellow());
            } else {
                for identity in identities {
                    let verified_status = if identity.verified {
                        "✅ Verified".green()
                    } else {
                        "❌ Pending".red()
                    };
                    let time =
                        chrono::DateTime::from_timestamp_millis(identity.created_at)
                            .map(|t| t.format("%Y-%m-%d %H:%M").to_string())
                            .unwrap_or_else(|| "Unknown".to_string());

                    println!();
                    println!(
                        "Wallet: {} [{}]",
                        identity.wallet_address.yellow(),
                        verified_status
                    );
                    if let Some(name) = &identity.display_name {
                        println!("  Name: {}", name);
                    }
                    println!("  KYC Level: {}", identity.kyc_level);
                    if identity.is_deceased {
                        println!("  Status: {}", "💀 Deceased".red());
                    }
                    if let Some(beneficiary) = &identity.beneficiary_address {
                        println!("  Beneficiary: {}", beneficiary);
                    }
                    println!("  Registered: {}", time);
                }
            }
        }

        IdentityAction::Approve { wallet } => {
            println!("{}", "✅ Approving Identity".green().bold());
            println!("Wallet: {}", wallet.yellow());

            let url = format!(
                "{}/api/v1/quillon-bank/identity/admin/approve",
                api_url
            );
            let response: ApiResponse<bool> =
                admin_post_string(client, &url, &wallet).await?.json().await?;

            if response.success && response.data == Some(true) {
                println!("{}", "✅ Identity verified successfully!".green().bold());
            } else {
                println!(
                    "{}",
                    format!(
                        "❌ Failed: {}",
                        response
                            .error
                            .unwrap_or("Identity not found".to_string())
                    )
                    .red()
                );
            }
        }

        IdentityAction::View { wallet } => {
            println!("{}", "🪪 Identity Details".green().bold());
            println!("{}", "─".repeat(60));

            let url = format!(
                "{}/api/v1/quillon-bank/identity/{}",
                api_url, wallet
            );
            let response: ApiResponse<Option<UserIdentity>> =
                client.get(&url).send().await?.json().await?;

            if let Some(Some(identity)) = response.data {
                println!("Wallet: {}", identity.wallet_address.yellow());
                if let Some(name) = &identity.display_name {
                    println!("Display Name: {}", name);
                }
                println!(
                    "Verified: {}",
                    if identity.verified {
                        "✅ Yes".green()
                    } else {
                        "❌ No".red()
                    }
                );
                println!("KYC Level: {}", identity.kyc_level);
                println!(
                    "Deceased: {}",
                    if identity.is_deceased {
                        "💀 Yes".red()
                    } else {
                        "No".normal()
                    }
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

            let url = format!(
                "{}/api/v1/quillon-bank/identity/admin/death-certificate/list",
                api_url
            );
            let api_resp: ApiResponse<Vec<DeathCertificate>> =
                admin_get(client, &url).await?.json().await?;

            let all_certs = api_resp.data.unwrap_or_default();
            let certs: Vec<_> = if show_pending_only {
                all_certs
                    .into_iter()
                    .filter(|c| !c.approved)
                    .collect()
            } else {
                all_certs
            };

            if certs.is_empty() {
                println!("{}", "No death certificates found.".yellow());
            } else {
                for cert in certs {
                    let status = if cert.executed {
                        "Executed".green()
                    } else if cert.approved {
                        "Approved".yellow()
                    } else {
                        "Pending".red()
                    };
                    let time =
                        chrono::DateTime::from_timestamp_millis(cert.issued_at)
                            .map(|t| t.format("%Y-%m-%d %H:%M").to_string())
                            .unwrap_or_else(|| "Unknown".to_string());

                    println!();
                    println!("Cert ID: {} [{}]", cert.id.cyan(), status);
                    println!("  Deceased: {}", cert.deceased_wallet.red());
                    println!("  Beneficiary: {}", cert.beneficiary_wallet.green());
                    println!("  Reason: {}", cert.reason);
                    println!("  Issued: {}", time);
                    if let Some(by) = &cert.approved_by {
                        println!("  Approved by: {}", by);
                    }
                }
            }
        }

        DeathCertAction::Approve { cert_id } => {
            println!("{}", "✅ Approving Death Certificate".green().bold());
            println!("Certificate ID: {}", cert_id.cyan());

            let url = format!(
                "{}/api/v1/quillon-bank/identity/admin/death-certificate/approve",
                api_url
            );
            let response: ApiResponse<bool> =
                admin_post_string(client, &url, &cert_id).await?.json().await?;

            if response.success && response.data == Some(true) {
                println!("{}", "✅ Death certificate approved!".green().bold());
                println!("The deceased's identity has been marked accordingly.");
                println!(
                    "Run 'inheritance execute {}' to transfer assets.",
                    cert_id
                );
            } else {
                println!(
                    "{}",
                    format!(
                        "❌ Failed: {}",
                        response
                            .error
                            .unwrap_or("Certificate not found".to_string())
                    )
                    .red()
                );
            }
        }

        DeathCertAction::View { cert_id } => {
            println!("{}", "💀 Death Certificate Details".green().bold());
            println!("{}", "─".repeat(60));

            let url = format!(
                "{}/api/v1/quillon-bank/identity/death-certificate/{}",
                api_url, cert_id
            );
            let response: ApiResponse<Option<DeathCertificate>> =
                client.get(&url).send().await?.json().await?;

            if let Some(Some(cert)) = response.data {
                let status = if cert.executed {
                    "Executed".green()
                } else if cert.approved {
                    "Approved (pending execution)".yellow()
                } else {
                    "Pending approval".red()
                };

                println!("Certificate ID: {}", cert.id.cyan());
                println!("Status: {}", status);
                println!("Deceased Wallet: {}", cert.deceased_wallet.red());
                println!("Beneficiary Wallet: {}", cert.beneficiary_wallet.green());
                println!("Reason: {}", cert.reason);
                let time =
                    chrono::DateTime::from_timestamp_millis(cert.issued_at)
                        .map(|t| t.format("%Y-%m-%d %H:%M UTC").to_string())
                        .unwrap_or_else(|| "Unknown".to_string());
                println!("Issued: {}", time);
                if let Some(by) = &cert.approved_by {
                    println!("Approved by: {}", by);
                }
            } else {
                println!("{}", "Certificate not found.".yellow());
            }
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

            let url = format!(
                "{}/api/v1/quillon-bank/identity/inheritance/{}",
                api_url, wallet
            );
            let response: ApiResponse<Option<InheritanceInfo>> =
                client.get(&url).send().await?.json().await?;

            if let Some(Some(info)) = response.data {
                println!("Deceased Wallet: {}", info.deceased_wallet.red());
                println!("Beneficiary Wallet: {}", info.beneficiary_wallet.green());
                println!(
                    "Total Balance: {:.4} QUG",
                    info.total_balance as f64 / 1e24
                );
                println!(
                    "Transfer Ready: {}",
                    if info.transfer_ready {
                        "✅ Yes".green()
                    } else {
                        "❌ No (pending approval)".yellow()
                    }
                );
            } else {
                println!(
                    "{}",
                    "No inheritance information found for this wallet.".yellow()
                );
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

            let url = format!(
                "{}/api/v1/quillon-bank/identity/admin/transfer",
                api_url
            );
            let response: ApiResponse<String> =
                admin_post_string(client, &url, &cert_id)
                    .await?
                    .json()
                    .await?;

            if response.success {
                println!();
                println!(
                    "{}",
                    "✅ Inheritance transfer executed successfully!"
                        .green()
                        .bold()
                );
                if let Some(msg) = response.data {
                    println!("{}", msg);
                }
            } else {
                println!(
                    "{}",
                    format!("❌ Failed: {}", response.error.unwrap_or_default()).red()
                );
            }
        }
    }

    Ok(())
}

// ============================================================================
// Email Handlers
// ============================================================================

async fn handle_email(
    client: &Client,
    api_url: &str,
    action: EmailAction,
) -> Result<(), Box<dyn std::error::Error>> {
    match action {
        EmailAction::Broadcast { subject, body } => {
            println!("{}", "📧 Broadcasting Bank Email".green().bold());
            println!("{}", "─".repeat(60));
            println!("  Subject: {}", subject.cyan());
            let preview = if body.len() > 80 {
                format!("{}...", &body[..80])
            } else {
                body.clone()
            };
            println!("  Body:    {}", preview);
            println!();

            print!("Send broadcast to ALL email users? [y/N] ");
            io::stdout().flush()?;
            let mut confirm = String::new();
            io::stdin().read_line(&mut confirm)?;
            if confirm.trim().to_lowercase() != "y" {
                println!("{}", "Cancelled.".yellow());
                return Ok(());
            }

            let payload = serde_json::json!({
                "subject": subject,
                "body": body,
            });

            let response: ApiResponse<String> =
                admin_post(client, &format!("{}/api/v1/quillon-bank/email/broadcast", api_url), &payload)
                    .await?
                    .json()
                    .await?;

            if response.success {
                println!();
                println!(
                    "{}",
                    "✅ Broadcast sent successfully!".green().bold()
                );
                if let Some(msg) = response.data {
                    println!("  {}", msg);
                }
            } else {
                println!(
                    "{}",
                    format!("❌ Failed: {}", response.error.unwrap_or_default()).red()
                );
            }
        }
    }

    Ok(())
}
