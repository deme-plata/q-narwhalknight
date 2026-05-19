/// Quillon Mail: Decentralized Email with Crypto Transfers
/// v7.3.2: Wallet-to-wallet email via P2P gossipsub + external email via SMTP
///
/// Features:
/// - Wallet-to-wallet encrypted email (P2P gossipsub)
/// - External email (SMTP send/receive via quillon.xyz)
/// - Crypto transfers attached to emails (QUG/QUGUSD/custom tokens)
/// - End-to-end encryption using Ed25519/ChaCha20
/// - SSE notifications for new emails

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    routing::{delete, get, post, put},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::streaming::StreamEvent;
use crate::wallet_auth::AuthenticatedWallet;
use crate::AppState;
use q_types::*;

// ============================================================================
// Router
// ============================================================================

pub fn email_router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/send", post(send_email))
        .route("/inbox", get(get_inbox))
        .route("/sent", get(get_sent))
        .route("/message/:id", get(get_email))
        .route("/message/:id", delete(delete_email))
        .route("/message/:id/read", put(mark_read))
        .route("/unread-count", get(get_unread_count))
        .route("/mark-all-read", post(mark_all_read))
        .route("/search", get(search_emails))
        .route("/contacts", get(get_contacts))
        .route("/folder/:folder", get(get_folder))
        // v7.3.3: Email alias settings + welcome email
        .route("/settings", get(get_email_settings))
        .route("/settings", put(update_email_settings))
        .route("/welcome", post(send_welcome_email))
}

// ============================================================================
// Request/Response Types
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct SendEmailRequest {
    pub to: String,
    pub subject: String,
    pub body: String,
    #[serde(default)]
    pub body_html: Option<String>,
    #[serde(default)]
    pub crypto_amount: Option<String>,
    #[serde(default)]
    pub crypto_token: Option<String>,
    #[serde(default)]
    pub reply_to: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct SendEmailResponse {
    pub email_id: String,
    pub delivery_method: String,
    pub crypto_transfer: Option<CryptoTransferInfo>,
}

#[derive(Debug, Serialize)]
pub struct CryptoTransferInfo {
    pub token: String,
    pub amount: String,
    pub tx_hash: String,
}

#[derive(Debug, Serialize)]
pub struct UnreadCountResponse {
    pub count: u64,
}

#[derive(Debug, Deserialize)]
pub struct PaginationParams {
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default)]
    pub offset: usize,
    #[serde(default)]
    pub unread_only: Option<bool>,
}

fn default_limit() -> usize {
    50
}

#[derive(Debug, Deserialize)]
pub struct SearchParams {
    pub q: String,
    #[serde(default = "default_limit")]
    pub limit: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailSettings {
    pub alias: Option<String>,            // Custom alias e.g. "demetri" → demetri@quillon.xyz
    pub display_name: Option<String>,     // Display name shown in emails
    pub signature: Option<String>,        // Email signature text
    pub auto_reply: Option<String>,       // Auto-reply message (None = disabled)
    pub notifications_enabled: bool,      // SSE notifications for new emails
    #[serde(default)]
    pub welcome_sent: bool,               // Whether welcome email has been sent
}

impl Default for EmailSettings {
    fn default() -> Self {
        Self {
            alias: None,
            display_name: None,
            signature: None,
            auto_reply: None,
            notifications_enabled: true,
            welcome_sent: false,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct UpdateEmailSettingsRequest {
    pub alias: Option<String>,
    pub display_name: Option<String>,
    pub signature: Option<String>,
    pub auto_reply: Option<String>,
    pub notifications_enabled: Option<bool>,
}

// ============================================================================
// Handlers
// ============================================================================

/// POST /api/v1/email/send — Send an email (wallet-to-wallet or external)
async fn send_email(
    auth_wallet: Option<AuthenticatedWallet>,
    State(state): State<Arc<AppState>>,
    Json(req): Json<SendEmailRequest>,
) -> Result<Json<ApiResponse<SendEmailResponse>>, StatusCode> {
    // Require authentication
    let auth = match auth_wallet {
        Some(w) => w,
        None => {
            return Ok(Json(ApiResponse::error(
                "Authentication required. Provide X-Wallet-Auth header.".to_string(),
            )))
        }
    };

    let sender_wallet = auth.address;
    let email_id = Uuid::new_v4().to_string();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Resolve recipient
    let (to_wallet, to_email, delivery_method) = resolve_recipient(&state, &req.to).await;

    // Handle crypto transfer if requested
    let mut crypto_transfer = None;
    let mut crypto_info = None;

    if let (Some(ref amount_str), Some(ref token)) = (&req.crypto_amount, &req.crypto_token) {
        match process_crypto_transfer(
            &state,
            &sender_wallet,
            &to_wallet,
            amount_str,
            token,
            &email_id,
            &req.subject,
        )
        .await
        {
            Ok((transfer, info)) => {
                crypto_transfer = Some(transfer);
                crypto_info = Some(info);
            }
            Err(e) => {
                return Ok(Json(ApiResponse::error(format!("Crypto transfer failed: {}", e))));
            }
        }
    }

    // Create email message
    let email = EmailMessage {
        id: email_id.clone(),
        from_wallet: sender_wallet,
        from_email: Some(format!("{}@quillon.xyz", &hex::encode(sender_wallet)[..8])),
        to_wallet,
        to_email: to_email.clone(),
        subject: req.subject.clone(),
        body: req.body.clone(),
        body_html: req.body_html.clone(),
        encrypted: to_wallet.is_some(), // E2E for wallet-to-wallet
        signature: Vec::new(), // TODO: Sign with sender's key
        timestamp: now,
        read: false,
        folder: "inbox".to_string(),
        thread_id: Some(req.reply_to.clone().unwrap_or_else(|| email_id.clone())),
        in_reply_to: req.reply_to.clone(),
        crypto_transfer,
        delivery_method: delivery_method.clone(),
    };

    // Save to recipient's inbox
    if let Err(e) = state.storage_engine.save_email(&email).await {
        error!("Failed to save email to inbox: {}", e);
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    }

    // Save to sender's sent folder
    let mut sent_copy = email.clone();
    sent_copy.folder = "sent".to_string();
    sent_copy.read = true;
    if let Err(e) = state.storage_engine.save_email(&sent_copy).await {
        error!("Failed to save sent copy: {}", e);
    }

    // Update contacts
    if let Some(ref to_w) = to_wallet {
        let contact = EmailContact {
            wallet_address: *to_w,
            display_name: None,
            email_address: to_email.clone(),
            last_contacted: now,
            message_count: 1,
        };
        let _ = state.storage_engine.save_email_contact(&sender_wallet, &contact).await;
    }

    // Deliver via P2P or queue for SMTP
    let method_str = match delivery_method {
        DeliveryMethod::P2PGossipsub => {
            if let Err(e) = publish_email_p2p(&state, &email).await {
                warn!("P2P email publish failed (stored locally): {}", e);
            }
            "p2p"
        }
        DeliveryMethod::SmtpOutbound => {
            // Queue for SMTP MTA delivery
            let outbound = OutboundEmail {
                id: Uuid::new_v4().to_string(),
                from_wallet: sender_wallet,
                from_email: format!("{}@quillon.xyz", &hex::encode(sender_wallet)[..8]),
                to_email: to_email.clone().unwrap_or_default(),
                subject: req.subject.clone(),
                body: req.body.clone(),
                body_html: req.body_html.clone(),
                timestamp: now,
                status: OutboundStatus::Pending,
                retry_count: 0,
                last_error: None,
                next_retry_at: None,
                email_id: Some(email_id.clone()),
            };
            if let Err(e) = state.storage_engine.save_outbound_email(&outbound).await {
                error!("Failed to queue outbound email: {}", e);
            }
            "smtp"
        }
        _ => "unknown",
    };

    // Emit SSE events
    let preview = if req.body.len() > 100 {
        format!("{}...", &req.body[..100])
    } else {
        req.body.clone()
    };

    let _ = state
        .event_broadcaster
        .broadcast(StreamEvent::EmailSent {
            email_id: email_id.clone(),
            to_address: req.to.clone(),
            subject: req.subject.clone(),
            delivery_method: method_str.to_string(),
            timestamp: chrono::Utc::now(),
        })
        .await;

    // If recipient is local, emit EmailReceived too
    if to_wallet.is_some() {
        let has_crypto = crypto_info.is_some();
        let _ = state
            .event_broadcaster
            .broadcast(StreamEvent::EmailReceived {
                email_id: email_id.clone(),
                from_address: format!("qnk{}", hex::encode(sender_wallet)),
                subject: req.subject.clone(),
                preview,
                has_crypto,
                crypto_amount: crypto_info.as_ref().map(|c| {
                    c.amount.parse::<f64>().unwrap_or(0.0)
                }),
                crypto_token: crypto_info.as_ref().map(|c| c.token.clone()),
                timestamp: chrono::Utc::now(),
            })
            .await;
    }

    info!(
        "📧 Email {} sent from {} to {} via {}",
        email_id,
        &hex::encode(sender_wallet)[..8],
        req.to,
        method_str
    );

    Ok(Json(ApiResponse::success(SendEmailResponse {
        email_id,
        delivery_method: method_str.to_string(),
        crypto_transfer: crypto_info,
    })))
}

/// GET /api/v1/email/inbox — Get inbox emails
async fn get_inbox(
    auth_wallet: Option<AuthenticatedWallet>,
    State(state): State<Arc<AppState>>,
    Query(params): Query<PaginationParams>,
) -> Result<Json<ApiResponse<Vec<EmailMessage>>>, StatusCode> {
    let auth = match auth_wallet {
        Some(w) => w,
        None => return Ok(Json(ApiResponse::error("Authentication required".to_string()))),
    };

    match state
        .storage_engine
        .get_inbox(&auth.address, "inbox", params.limit, params.offset)
        .await
    {
        Ok(mut emails) => {
            if params.unread_only.unwrap_or(false) {
                emails.retain(|e| !e.read);
            }
            Ok(Json(ApiResponse::success(emails)))
        }
        Err(e) => {
            error!("Failed to get inbox: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// GET /api/v1/email/sent — Get sent emails
async fn get_sent(
    auth_wallet: Option<AuthenticatedWallet>,
    State(state): State<Arc<AppState>>,
    Query(params): Query<PaginationParams>,
) -> Result<Json<ApiResponse<Vec<EmailMessage>>>, StatusCode> {
    let auth = match auth_wallet {
        Some(w) => w,
        None => return Ok(Json(ApiResponse::error("Authentication required".to_string()))),
    };

    match state
        .storage_engine
        .get_inbox(&auth.address, "sent", params.limit, params.offset)
        .await
    {
        Ok(emails) => Ok(Json(ApiResponse::success(emails))),
        Err(e) => {
            error!("Failed to get sent: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// GET /api/v1/email/message/:id — Get single email
async fn get_email(
    auth_wallet: Option<AuthenticatedWallet>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<ApiResponse<Option<EmailMessage>>>, StatusCode> {
    let _auth = match auth_wallet {
        Some(w) => w,
        None => return Ok(Json(ApiResponse::error("Authentication required".to_string()))),
    };

    match state.storage_engine.get_email(&id).await {
        Ok(email) => Ok(Json(ApiResponse::success(email))),
        Err(e) => {
            error!("Failed to get email {}: {}", id, e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// DELETE /api/v1/email/message/:id — Delete email (move to trash)
async fn delete_email(
    auth_wallet: Option<AuthenticatedWallet>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<ApiResponse<bool>>, StatusCode> {
    let _auth = match auth_wallet {
        Some(w) => w,
        None => return Ok(Json(ApiResponse::error("Authentication required".to_string()))),
    };

    match state.storage_engine.delete_email(&id).await {
        Ok(()) => Ok(Json(ApiResponse::success(true))),
        Err(e) => {
            error!("Failed to delete email {}: {}", id, e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// PUT /api/v1/email/message/:id/read — Mark email as read
async fn mark_read(
    auth_wallet: Option<AuthenticatedWallet>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<ApiResponse<bool>>, StatusCode> {
    let _auth = match auth_wallet {
        Some(w) => w,
        None => return Ok(Json(ApiResponse::error("Authentication required".to_string()))),
    };

    match state.storage_engine.mark_email_read(&id).await {
        Ok(()) => Ok(Json(ApiResponse::success(true))),
        Err(e) => {
            error!("Failed to mark email {} as read: {}", id, e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// GET /api/v1/email/unread-count — Get unread count
async fn get_unread_count(
    auth_wallet: Option<AuthenticatedWallet>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<UnreadCountResponse>>, StatusCode> {
    let auth = match auth_wallet {
        Some(w) => w,
        None => return Ok(Json(ApiResponse::error("Authentication required".to_string()))),
    };

    match state.storage_engine.get_unread_count(&auth.address).await {
        Ok(count) => Ok(Json(ApiResponse::success(UnreadCountResponse { count }))),
        Err(e) => {
            error!("Failed to get unread count: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// POST /api/v1/email/mark-all-read — Mark all inbox emails as read
async fn mark_all_read(
    auth_wallet: Option<AuthenticatedWallet>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<UnreadCountResponse>>, StatusCode> {
    let auth = match auth_wallet {
        Some(w) => w,
        None => return Ok(Json(ApiResponse::error("Authentication required".to_string()))),
    };

    match state.storage_engine.mark_all_inbox_read(&auth.address).await {
        Ok(marked) => {
            info!("📖 Marked {} emails as read for {}", marked, &hex::encode(auth.address)[..8]);
            Ok(Json(ApiResponse::success(UnreadCountResponse { count: 0 })))
        }
        Err(e) => {
            error!("Failed to mark all as read: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// GET /api/v1/email/search — Search emails
async fn search_emails(
    auth_wallet: Option<AuthenticatedWallet>,
    State(state): State<Arc<AppState>>,
    Query(params): Query<SearchParams>,
) -> Result<Json<ApiResponse<Vec<EmailMessage>>>, StatusCode> {
    let auth = match auth_wallet {
        Some(w) => w,
        None => return Ok(Json(ApiResponse::error("Authentication required".to_string()))),
    };

    match state
        .storage_engine
        .search_emails(&auth.address, &params.q, params.limit)
        .await
    {
        Ok(emails) => Ok(Json(ApiResponse::success(emails))),
        Err(e) => {
            error!("Failed to search emails: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// GET /api/v1/email/contacts — Get email contacts
async fn get_contacts(
    auth_wallet: Option<AuthenticatedWallet>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<Vec<EmailContact>>>, StatusCode> {
    let auth = match auth_wallet {
        Some(w) => w,
        None => return Ok(Json(ApiResponse::error("Authentication required".to_string()))),
    };

    match state.storage_engine.get_email_contacts(&auth.address).await {
        Ok(contacts) => Ok(Json(ApiResponse::success(contacts))),
        Err(e) => {
            error!("Failed to get contacts: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// GET /api/v1/email/folder/:folder — Get emails by folder
async fn get_folder(
    auth_wallet: Option<AuthenticatedWallet>,
    State(state): State<Arc<AppState>>,
    Path(folder): Path<String>,
    Query(params): Query<PaginationParams>,
) -> Result<Json<ApiResponse<Vec<EmailMessage>>>, StatusCode> {
    let auth = match auth_wallet {
        Some(w) => w,
        None => return Ok(Json(ApiResponse::error("Authentication required".to_string()))),
    };

    let valid_folders = ["inbox", "sent", "drafts", "trash", "quillon-bank"];
    if !valid_folders.contains(&folder.as_str()) {
        return Ok(Json(ApiResponse::error(format!("Invalid folder: {}", folder))));
    }

    match state
        .storage_engine
        .get_inbox(&auth.address, &folder, params.limit, params.offset)
        .await
    {
        Ok(emails) => Ok(Json(ApiResponse::success(emails))),
        Err(e) => {
            error!("Failed to get folder {}: {}", folder, e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

// ============================================================================
// Email Settings & Welcome Email
// ============================================================================

/// GET /api/v1/email/settings — Get email settings (alias, display name, etc.)
async fn get_email_settings(
    auth_wallet: Option<AuthenticatedWallet>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<EmailSettings>>, StatusCode> {
    let auth = match auth_wallet {
        Some(w) => w,
        None => return Ok(Json(ApiResponse::error("Authentication required".to_string()))),
    };

    let wallet_hex = hex::encode(auth.address);
    let key = format!("email_settings:{}", wallet_hex);

    match state.storage_engine.get_email_settings(&wallet_hex).await {
        Ok(Some(settings)) => {
            let email_settings: EmailSettings = serde_json::from_value(settings).unwrap_or(EmailSettings {
                alias: Some(wallet_hex[..8].to_string()),
                ..Default::default()
            });
            Ok(Json(ApiResponse::success(email_settings)))
        }
        Ok(None) => {
            Ok(Json(ApiResponse::success(EmailSettings {
                alias: Some(wallet_hex[..8].to_string()),
                ..Default::default()
            })))
        }
        Err(e) => {
            error!("Failed to get email settings: {}", e);
            Ok(Json(ApiResponse::success(EmailSettings {
                alias: Some(wallet_hex[..8].to_string()),
                ..Default::default()
            })))
        }
    }
}

/// PUT /api/v1/email/settings — Update email settings
async fn update_email_settings(
    auth_wallet: Option<AuthenticatedWallet>,
    State(state): State<Arc<AppState>>,
    Json(req): Json<UpdateEmailSettingsRequest>,
) -> Result<Json<ApiResponse<EmailSettings>>, StatusCode> {
    let auth = match auth_wallet {
        Some(w) => w,
        None => return Ok(Json(ApiResponse::error("Authentication required".to_string()))),
    };

    let wallet_hex = hex::encode(auth.address);

    // Validate alias (alphanumeric, 3-20 chars, lowercase)
    if let Some(ref alias) = req.alias {
        let clean = alias.trim().to_lowercase();
        if clean.len() < 3 || clean.len() > 20 {
            return Ok(Json(ApiResponse::error("Alias must be 3-20 characters".to_string())));
        }
        if !clean.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.') {
            return Ok(Json(ApiResponse::error("Alias can only contain letters, numbers, underscores, and dots".to_string())));
        }
        let reserved = ["admin", "support", "help", "info", "noreply", "system", "root", "postmaster"];
        if reserved.contains(&clean.as_str()) {
            return Ok(Json(ApiResponse::error("This alias is reserved".to_string())));
        }

        // Check if alias is already taken by another wallet
        if let Ok(Some(existing)) = state.storage_engine.get_email_alias_wallet(&clean).await {
            if !existing.is_empty() && existing != wallet_hex {
                return Ok(Json(ApiResponse::error("This alias is already taken".to_string())));
            }
        }

        // Register the alias → wallet mapping
        let _ = state.storage_engine.save_email_alias(&clean, &wallet_hex).await;
    }

    // Load existing settings or create new
    let mut settings = match state.storage_engine.get_email_settings(&wallet_hex).await {
        Ok(Some(val)) => serde_json::from_value::<EmailSettings>(val).unwrap_or_default(),
        _ => EmailSettings::default(),
    };

    // Apply updates
    if let Some(alias) = req.alias {
        // Remove old alias mapping if changed
        if let Some(ref old_alias) = settings.alias {
            if old_alias != &alias.trim().to_lowercase() {
                let _ = state.storage_engine.delete_email_alias(old_alias).await;
            }
        }
        settings.alias = Some(alias.trim().to_lowercase());
    }
    if let Some(display_name) = req.display_name {
        settings.display_name = if display_name.trim().is_empty() { None } else { Some(display_name.trim().to_string()) };
    }
    if let Some(signature) = req.signature {
        settings.signature = if signature.trim().is_empty() { None } else { Some(signature) };
    }
    if req.auto_reply.is_some() {
        settings.auto_reply = req.auto_reply;
    }
    if let Some(notif) = req.notifications_enabled {
        settings.notifications_enabled = notif;
    }

    // Save
    let data = serde_json::to_vec(&settings).unwrap_or_default();
    if let Err(e) = state.storage_engine.save_email_settings(&wallet_hex, &data).await {
        error!("Failed to save email settings: {}", e);
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    }

    info!("📧 Email settings updated for {} (alias: {:?})", &wallet_hex[..8], settings.alias);
    Ok(Json(ApiResponse::success(settings)))
}

/// POST /api/v1/email/welcome — Send a welcome email to the user
async fn send_welcome_email(
    auth_wallet: Option<AuthenticatedWallet>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<bool>>, StatusCode> {
    let auth = match auth_wallet {
        Some(w) => w,
        None => return Ok(Json(ApiResponse::error("Authentication required".to_string()))),
    };

    let wallet_hex = hex::encode(auth.address);

    // Check if welcome already sent
    if let Ok(Some(val)) = state.storage_engine.get_email_settings(&wallet_hex).await {
        if let Ok(settings) = serde_json::from_value::<EmailSettings>(val) {
            if settings.welcome_sent {
                return Ok(Json(ApiResponse::success(true))); // Already sent
            }
        }
    }

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let email_id = Uuid::new_v4().to_string();
    let system_wallet = [0u8; 32]; // System wallet (zero address)
    let alias = &wallet_hex[..8];

    let welcome_body = format!(
        r#"Welcome to Quillon Mail — the world's first decentralized blockchain email system.

Your email address: {}@quillon.xyz

What makes Quillon Mail special:

  Decentralized — Your emails are stored on-chain and synced via P2P gossipsub
  End-to-End Encrypted — Wallet-to-wallet emails are encrypted with your keys
  Crypto Transfers — Attach QUG, QUGUSD, or any token directly to emails
  Zero Censorship — No central server can block or read your messages
  AI Assistant — Use the built-in AI to help draft and improve your emails

Quick Start:
  1. Set up your email alias in Settings (gear icon) to get a custom @quillon.xyz address
  2. Send your first email to a wallet address or any external email
  3. Attach crypto to transfer tokens alongside your message

You're now part of a new era of communication. Every email is a statement of digital sovereignty.

Welcome aboard,
The Quillon Team"#,
        alias
    );

    let email = EmailMessage {
        id: email_id.clone(),
        from_wallet: system_wallet,
        from_email: Some("system@quillon.xyz".to_string()),
        to_wallet: Some(auth.address),
        to_email: Some(format!("{}@quillon.xyz", alias)),
        subject: "Welcome to Quillon Mail".to_string(),
        body: welcome_body,
        body_html: None,
        encrypted: false,
        signature: Vec::new(),
        timestamp: now,
        read: true, // v8.2.9: Welcome email is pre-read — no persistent notification badge
        folder: "inbox".to_string(),
        thread_id: Some(email_id.clone()),
        in_reply_to: None,
        crypto_transfer: None,
        delivery_method: DeliveryMethod::P2PGossipsub,
    };

    if let Err(e) = state.storage_engine.save_email(&email).await {
        error!("Failed to save welcome email: {}", e);
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    }

    // Mark welcome as sent in settings
    let mut settings = match state.storage_engine.get_email_settings(&wallet_hex).await {
        Ok(Some(val)) => serde_json::from_value::<EmailSettings>(val).unwrap_or_default(),
        _ => EmailSettings {
            alias: Some(alias.to_string()),
            ..Default::default()
        },
    };
    settings.welcome_sent = true;
    let data = serde_json::to_vec(&settings).unwrap_or_default();
    let _ = state.storage_engine.save_email_settings(&wallet_hex, &data).await;

    // v8.2.9: No SSE broadcast for welcome email — it's pre-read and shouldn't
    // trigger the unread notification badge or email-received event listener

    info!("📧 Welcome email sent to {}", &wallet_hex[..8]);
    Ok(Json(ApiResponse::success(true)))
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Resolve a recipient address to wallet and/or email
async fn resolve_recipient(
    state: &Arc<AppState>,
    to: &str,
) -> (Option<[u8; 32]>, Option<String>, DeliveryMethod) {
    // Check if it's a hex wallet address (64 chars) or qnk-prefixed
    let wallet_hex = if to.starts_with("qnk") && to.len() == 67 {
        Some(&to[3..])
    } else if to.len() == 64 && to.chars().all(|c| c.is_ascii_hexdigit()) {
        Some(to)
    } else {
        None
    };

    if let Some(hex) = wallet_hex {
        if let Ok(bytes) = hex::decode(hex) {
            if bytes.len() == 32 {
                let mut addr = [0u8; 32];
                addr.copy_from_slice(&bytes);
                return (Some(addr), None, DeliveryMethod::P2PGossipsub);
            }
        }
    }

    // Check if it's user@quillon.xyz — resolve to wallet via alias or hex prefix lookup
    if to.ends_with("@quillon.xyz") {
        let username = to.strip_suffix("@quillon.xyz").unwrap_or("");

        // First: try resolving as a custom alias (e.g. "demetri@quillon.xyz")
        if !username.is_empty() {
            if let Ok(Some(wallet_hex)) = state.storage_engine.get_email_alias_wallet(username).await {
                if let Ok(bytes) = hex::decode(&wallet_hex) {
                    if bytes.len() == 32 {
                        let mut addr = [0u8; 32];
                        addr.copy_from_slice(&bytes);
                        info!("📧 Resolved alias '{}' to wallet {}", username, &wallet_hex[..8]);
                        return (Some(addr), Some(to.to_string()), DeliveryMethod::P2PGossipsub);
                    }
                }
            }
        }

        // Second: try resolving as a wallet hex prefix (e.g. "a1b2c3d4@quillon.xyz")
        if username.len() >= 8 && username.chars().all(|c| c.is_ascii_hexdigit()) {
            let balances = state.wallet_balances.read().await;
            for (addr, _) in balances.iter() {
                let addr_hex = hex::encode(addr);
                if addr_hex.starts_with(username) {
                    return (Some(*addr), Some(to.to_string()), DeliveryMethod::P2PGossipsub);
                }
            }
        }

        // Unknown quillon.xyz user — deliver via P2P broadcast
        warn!("📧 Could not resolve @quillon.xyz recipient: {}", to);
        return (None, Some(to.to_string()), DeliveryMethod::P2PGossipsub);
    }

    // External email address — deliver via SMTP
    if to.contains('@') {
        return (None, Some(to.to_string()), DeliveryMethod::SmtpOutbound);
    }

    // Fallback: try resolving as a bare alias name (e.g. user typed "demetri" without @quillon.xyz)
    if !to.is_empty() && to.len() <= 20 && to.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.') {
        if let Ok(Some(wallet_hex)) = state.storage_engine.get_email_alias_wallet(to).await {
            if let Ok(bytes) = hex::decode(&wallet_hex) {
                if bytes.len() == 32 {
                    let mut addr = [0u8; 32];
                    addr.copy_from_slice(&bytes);
                    info!("📧 Resolved bare alias '{}' to wallet {}", to, &wallet_hex[..8]);
                    return (Some(addr), Some(format!("{}@quillon.xyz", to)), DeliveryMethod::P2PGossipsub);
                }
            }
        }
    }

    // Truly unknown recipient
    (None, Some(to.to_string()), DeliveryMethod::P2PGossipsub)
}

/// Process a crypto transfer attached to an email
async fn process_crypto_transfer(
    state: &Arc<AppState>,
    sender: &[u8; 32],
    recipient: &Option<[u8; 32]>,
    amount_str: &str,
    token: &str,
    email_id: &str,
    email_subject: &str,
) -> Result<(CryptoTransfer, CryptoTransferInfo), String> {
    let recipient = recipient.ok_or("Crypto transfers require a wallet recipient")?;

    let amount_display: f64 = amount_str.parse().map_err(|_| "Invalid amount")?;
    if amount_display <= 0.0 {
        return Err("Amount must be positive".to_string());
    }

    let is_qug = token == "QUG";
    let is_qugusd = token == "QUGUSD";

    if is_qug {
        // QUG transfer via wallet_balances
        let amount_raw = (amount_display * 1e24) as u128;

        let mut balances = state.wallet_balances.write().await;
        let sender_balance = balances.get(sender).copied().unwrap_or(0);

        if sender_balance < amount_raw {
            return Err(format!(
                "Insufficient QUG balance: have {:.6}, need {:.6}",
                sender_balance as f64 / 1e24,
                amount_display
            ));
        }

        // Debit sender, credit recipient
        *balances.entry(*sender).or_insert(0) -= amount_raw;
        *balances.entry(recipient).or_insert(0) += amount_raw;
        let new_sender_balance = balances.get(sender).copied().unwrap_or(0);
        let new_recipient_balance = balances.get(&recipient).copied().unwrap_or(0);
        drop(balances);

        // Persist to storage
        let _ = state.storage_engine.save_wallet_balance(sender, new_sender_balance).await;
        let _ = state.storage_engine.save_wallet_balance(&recipient, new_recipient_balance).await;

        // Create tx hash from email_id
        let tx_hash = blake3::hash(format!("email_crypto:{}:{}", email_id, token).as_bytes());
        let mut hash = [0u8; 32];
        hash.copy_from_slice(tx_hash.as_bytes());

        let memo = if email_subject.trim().is_empty() {
            "Email transfer".to_string()
        } else {
            format!("Email: {}", email_subject.trim())
        };
        let tx_data = format!("email_crypto:{}", memo);
        let email_tx = q_types::Transaction {
            id: hash,
            from: *sender,
            to: recipient,
            amount: amount_raw,
            fee: 0,
            nonce: 0,
            signature: Vec::new(),
            timestamp: chrono::Utc::now(),
            data: tx_data.into_bytes(),
            token_type: q_types::TokenType::Qug,
            fee_token_type: q_types::TokenType::Qug,
            tx_type: q_types::TransactionType::Transfer,
            pqc_signature: None,
            signature_phase: q_types::TxSignaturePhase::Phase0Ed25519,
            pqc_public_key: None,
        };

        if let Err(e) = state.storage_engine.save_transaction(&email_tx).await {
            warn!("Failed to persist email-crypto tx to storage index: {}", e);
        }

        // Emit balance update SSE events
        let _ = state
            .event_broadcaster
            .broadcast(StreamEvent::BalanceUpdated {
                wallet_address: format!("qnk{}", hex::encode(sender)),
                old_balance: (sender_balance as f64) / 1e24,
                new_balance: ((sender_balance - amount_raw) as f64) / 1e24,
                change_reason: format!("email_crypto_send:{}", email_id),
                timestamp: chrono::Utc::now(),
                block_hash: None,
                block_height: None,
                confirmation_status: "confirmed".to_string(),
                from_address: None,
                tx_hash: None,
                memo: None,
            })
            .await;

        Ok((
            CryptoTransfer {
                token_type: "QUG".to_string(),
                amount: amount_raw,
                tx_hash: hash,
                confirmed: true,
            },
            CryptoTransferInfo {
                token: "QUG".to_string(),
                amount: amount_str.to_string(),
                tx_hash: hex::encode(hash),
            },
        ))
    } else if is_qugusd {
        // QUGUSD transfer via token_balances
        let qugusd_addr: [u8; 32] = {
            let mut addr = [0u8; 32];
            addr[0..6].copy_from_slice(b"QUGUSD");
            addr
        };

        let amount_raw = (amount_display * 1e24) as u128;
        let mut token_balances = state.token_balances.write().await;

        let sender_key = (*sender, qugusd_addr);
        let sender_balance = token_balances.get(&sender_key).copied().unwrap_or(0);

        if sender_balance < amount_raw {
            return Err("Insufficient QUGUSD balance".to_string());
        }

        *token_balances.entry(sender_key).or_insert(0) -= amount_raw;
        *token_balances.entry((recipient, qugusd_addr)).or_insert(0) += amount_raw;
        drop(token_balances);

        let tx_hash = blake3::hash(format!("email_crypto:{}:{}", email_id, token).as_bytes());
        let mut hash = [0u8; 32];
        hash.copy_from_slice(tx_hash.as_bytes());

        let memo = if email_subject.trim().is_empty() {
            "Email transfer".to_string()
        } else {
            format!("Email: {}", email_subject.trim())
        };
        let tx_data = format!("email_crypto:{}", memo);
        let email_tx = q_types::Transaction {
            id: hash,
            from: *sender,
            to: recipient,
            amount: amount_raw,
            fee: 0,
            nonce: 0,
            signature: Vec::new(),
            timestamp: chrono::Utc::now(),
            data: tx_data.into_bytes(),
            token_type: q_types::TokenType::Qugusd,
            fee_token_type: q_types::TokenType::Qug,
            tx_type: q_types::TransactionType::Transfer,
            pqc_signature: None,
            signature_phase: q_types::TxSignaturePhase::Phase0Ed25519,
            pqc_public_key: None,
        };

        if let Err(e) = state.storage_engine.save_transaction(&email_tx).await {
            warn!("Failed to persist email-crypto tx to storage index: {}", e);
        }

        Ok((
            CryptoTransfer {
                token_type: "QUGUSD".to_string(),
                amount: amount_raw,
                tx_hash: hash,
                confirmed: true,
            },
            CryptoTransferInfo {
                token: "QUGUSD".to_string(),
                amount: amount_str.to_string(),
                tx_hash: hex::encode(hash),
            },
        ))
    } else {
        // Custom token transfer
        let token_addr = if token.len() == 64 {
            let bytes = hex::decode(token).map_err(|_| "Invalid token address")?;
            let mut addr = [0u8; 32];
            if bytes.len() == 32 {
                addr.copy_from_slice(&bytes);
                addr
            } else {
                return Err("Invalid token address length".to_string());
            }
        } else {
            return Err(format!("Unknown token: {}. Use QUG, QUGUSD, or a 64-char hex token address.", token));
        };

        // Use 24-decimal for custom tokens (amount_display * 1e24)
        let amount_raw = (amount_display * 1e24) as u128;
        let mut token_balances = state.token_balances.write().await;

        let sender_key = (*sender, token_addr);
        let sender_balance = token_balances.get(&sender_key).copied().unwrap_or(0);

        if sender_balance < amount_raw {
            return Err("Insufficient token balance".to_string());
        }

        *token_balances.entry(sender_key).or_insert(0) -= amount_raw;
        *token_balances.entry((recipient, token_addr)).or_insert(0) += amount_raw;
        drop(token_balances);

        let tx_hash = blake3::hash(format!("email_crypto:{}:{}", email_id, token).as_bytes());
        let mut hash = [0u8; 32];
        hash.copy_from_slice(tx_hash.as_bytes());

        Ok((
            CryptoTransfer {
                token_type: token.to_string(),
                amount: amount_raw,
                tx_hash: hash,
                confirmed: true,
            },
            CryptoTransferInfo {
                token: token.to_string(),
                amount: amount_str.to_string(),
                tx_hash: hex::encode(hash),
            },
        ))
    }
}

/// Publish email via P2P gossipsub for wallet-to-wallet delivery
async fn publish_email_p2p(state: &Arc<AppState>, email: &EmailMessage) -> Result<(), String> {
    let network_id = std::env::var("Q_NETWORK_ID").unwrap_or_else(|_| "mainnet-genesis".to_string());
    let topic = format!("/qnk/{}/email", network_id);

    let email_bytes = serde_json::to_vec(email).map_err(|e| format!("Serialize error: {}", e))?;

    if let Some(ref tx) = state.libp2p_command_tx {
        tx.send(q_network::NetworkCommand::PublishTokenAnnouncement {
            topic,
            announcement_bytes: email_bytes,
        })
        .map_err(|e| format!("P2P publish error: {}", e))?;

        debug!("📧 Published email {} via P2P gossipsub", email.id);
        Ok(())
    } else {
        Err("P2P network not available".to_string())
    }
}

/// Handle incoming P2P email message (called from main.rs gossipsub handler)
pub async fn handle_p2p_email(state: &Arc<AppState>, data: &[u8]) {
    match serde_json::from_slice::<EmailMessage>(data) {
        Ok(email) => {
            // Check if we are the recipient
            let is_local = if let Some(ref to_wallet) = email.to_wallet {
                let balances = state.wallet_balances.read().await;
                balances.contains_key(to_wallet)
            } else {
                false
            };

            if !is_local {
                debug!("📧 P2P email {} is not for us, ignoring", email.id);
                return;
            }

            // v8.2.11: Skip if email already exists — prevents overwriting
            // read=true back to read=false on gossipsub retransmissions
            if let Ok(Some(_)) = state.storage_engine.get_email(&email.id).await {
                debug!("📧 P2P email {} already exists, skipping duplicate", email.id);
                return;
            }

            // Store in our inbox
            if let Err(e) = state.storage_engine.save_email(&email).await {
                error!("Failed to save P2P email: {}", e);
                return;
            }

            // Emit SSE notification
            let preview = if email.body.len() > 100 {
                format!("{}...", &email.body[..100])
            } else {
                email.body.clone()
            };

            let has_crypto = email.crypto_transfer.is_some();
            let _ = state
                .event_broadcaster
                .broadcast(StreamEvent::EmailReceived {
                    email_id: email.id.clone(),
                    from_address: format!("qnk{}", hex::encode(email.from_wallet)),
                    subject: email.subject.clone(),
                    preview,
                    has_crypto,
                    crypto_amount: email.crypto_transfer.as_ref().map(|c| c.amount as f64 / 1e24),
                    crypto_token: email.crypto_transfer.as_ref().map(|c| c.token_type.clone()),
                    timestamp: chrono::Utc::now(),
                })
                .await;

            info!(
                "📬 Received P2P email {} from {} (crypto: {})",
                email.id,
                &hex::encode(email.from_wallet)[..8],
                has_crypto
            );
        }
        Err(e) => {
            warn!("Failed to deserialize P2P email: {}", e);
        }
    }
}
