// v7.3.0: Admin Settings API — Node operator admin panel
// Separate from deploy_admin_api (FOUNDER_WALLET). This uses the configurable
// --admin-wallet / Q_ADMIN_WALLET which defaults to FOUNDER_WALLET but can be
// set to any wallet by the node operator.

use axum::{
    extract::{Json, State},
    http::{HeaderMap, StatusCode},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{info, warn};

use chrono::Utc;
use crate::AppState;

// ============================================================================
// Helpers
// ============================================================================

/// Extract wallet address from request headers (same pattern as deploy_admin_api)
fn extract_wallet(headers: &HeaderMap) -> Option<String> {
    if let Some(auth) = headers.get("x-wallet-auth") {
        if let Ok(auth_str) = auth.to_str() {
            let wallet = auth_str.split(':').next().unwrap_or("");
            let clean = wallet.replace("qnk", "").replace("qug", "");
            if clean.len() == 64 {
                return Some(clean);
            }
        }
    }
    if let Some(auth) = headers.get("authorization") {
        if let Ok(auth_str) = auth.to_str() {
            let token = auth_str.strip_prefix("Bearer ").unwrap_or(auth_str);
            let clean = token.replace("qnk", "").replace("qug", "");
            if clean.len() == 64 {
                return Some(clean);
            }
        }
    }
    None
}

async fn is_node_admin(headers: &HeaderMap, state: &AppState) -> bool {
    // 1) Classic check: X-Wallet-Auth or raw hex Bearer matches admin_wallet or FOUNDER_WALLET
    if let Some(wallet) = extract_wallet(headers) {
        if wallet == state.admin_wallet {
            return true;
        }
        // v9.0.3: Also accept FOUNDER_WALLET (master wallet always has admin access)
        if wallet == crate::aegis_auth_middleware::FOUNDER_WALLET {
            return true;
        }
    }

    // 2) OAuth2 check: any valid, non-expired Bearer token → treat as admin
    //    (node operators who log in via OAuth2 should see admin panel)
    if let Some(auth) = headers.get("authorization") {
        if let Ok(auth_str) = auth.to_str() {
            if let Some(token) = auth_str.strip_prefix("Bearer ") {
                if !token.is_empty() {
                    // Check as OAuth2 access token
                    if let Some(access_token) = state.oauth2_storage.get_access_token(token).await {
                        if access_token.expires_at > Utc::now() {
                            return true;
                        }
                    }
                    // v9.0.3: Also check if the Bearer value is a wallet address
                    // (frontend sends wallet as Bearer when no OAuth2 token exists)
                    let clean = token.replace("qnk", "").replace("qug", "");
                    if clean.len() == 64 {
                        if clean == state.admin_wallet || clean == crate::aegis_auth_middleware::FOUNDER_WALLET {
                            return true;
                        }
                    }
                }
            }
        }
    }

    false
}

// ============================================================================
// Response types
// ============================================================================

#[derive(Serialize)]
pub struct IsAdminResponse {
    pub is_admin: bool,
}

#[derive(Serialize)]
pub struct AdminSettingsResponse {
    pub admin_wallet: String,
    pub version: String,
    pub uptime_secs: u64,
    pub height: u64,
    pub network_height: u64,
    pub peers: usize,
    pub network_id: String,
    pub oauth2_clients: usize,
    pub oauth2_active_tokens: usize,
    pub oauth2_consents: usize,
}

#[derive(Serialize)]
pub struct ConsentEntry {
    pub client_id: String,
    pub scopes: Vec<String>,
    pub granted_at: String,
}

#[derive(Deserialize)]
pub struct RevokeConsentRequest {
    pub client_id: String,
}

#[derive(Serialize)]
pub struct NodeInfoResponse {
    pub version: String,
    pub uptime_secs: u64,
    pub height: u64,
    pub network_height: u64,
    pub peers: usize,
    pub network_id: String,
    pub mining_healthy: bool,
}

// ============================================================================
// Handlers
// ============================================================================

/// GET /api/v1/admin/is-admin
/// Always responds (no 403). Returns { is_admin: true/false }.
pub async fn is_admin(
    headers: HeaderMap,
    State(state): State<Arc<AppState>>,
) -> Json<IsAdminResponse> {
    Json(IsAdminResponse {
        is_admin: is_node_admin(&headers, &state).await,
    })
}

/// GET /api/v1/admin/settings
/// Returns admin wallet, node stats, and OAuth2 summary. Admin-only.
pub async fn admin_settings(
    headers: HeaderMap,
    State(state): State<Arc<AppState>>,
) -> Result<Json<AdminSettingsResponse>, StatusCode> {
    if !is_node_admin(&headers, &state).await {
        return Err(StatusCode::FORBIDDEN);
    }

    let uptime = state.start_time.elapsed().as_secs();
    let height = state.current_height_atomic.load(std::sync::atomic::Ordering::Relaxed);
    let network_height = state.highest_network_height.load(std::sync::atomic::Ordering::Relaxed);
    let peers = state
        .libp2p_peer_count
        .as_ref()
        .map(|c| c.load(std::sync::atomic::Ordering::Relaxed))
        .unwrap_or(0);

    let oauth2 = &state.oauth2_storage;
    let client_count = oauth2.client_count().await;
    let active_tokens = oauth2.active_token_count().await;
    let admin_wallet_clean = extract_wallet(&headers).unwrap_or_default();
    let consent_count = oauth2.get_consents_for_wallet(&admin_wallet_clean).await.len();

    Ok(Json(AdminSettingsResponse {
        admin_wallet: format!("{}...{}", &state.admin_wallet[..8], &state.admin_wallet[56..]),
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_secs: uptime,
        height,
        network_height,
        peers,
        network_id: std::env::var("Q_NETWORK_ID").unwrap_or_else(|_| "mainnet-genesis".to_string()),
        oauth2_clients: client_count,
        oauth2_active_tokens: active_tokens,
        oauth2_consents: consent_count,
    }))
}

/// GET /api/v1/admin/oauth2/consents
/// Lists OAuth2 consents granted by the admin wallet. Admin-only.
pub async fn oauth2_consents(
    headers: HeaderMap,
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<ConsentEntry>>, StatusCode> {
    if !is_node_admin(&headers, &state).await {
        return Err(StatusCode::FORBIDDEN);
    }

    let wallet = extract_wallet(&headers).unwrap_or_default();
    let consents = state.oauth2_storage.get_consents_for_wallet(&wallet).await;

    let entries: Vec<ConsentEntry> = consents
        .into_iter()
        .map(|c| ConsentEntry {
            client_id: c.client_id,
            scopes: c.scopes,
            granted_at: c.granted_at.to_rfc3339(),
        })
        .collect();

    Ok(Json(entries))
}

/// POST /api/v1/admin/oauth2/revoke-consent
/// Revokes a consent and associated tokens. Admin-only.
pub async fn revoke_consent(
    headers: HeaderMap,
    State(state): State<Arc<AppState>>,
    Json(body): Json<RevokeConsentRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    if !is_node_admin(&headers, &state).await {
        return Err(StatusCode::FORBIDDEN);
    }

    let wallet = extract_wallet(&headers).unwrap_or_default();
    let revoked = state.oauth2_storage.revoke_consent(&wallet, &body.client_id).await;

    if revoked {
        Ok(Json(serde_json::json!({ "revoked": true, "client_id": body.client_id })))
    } else {
        warn!("Admin tried to revoke non-existent consent for client {}", body.client_id);
        Ok(Json(serde_json::json!({ "revoked": false, "client_id": body.client_id })))
    }
}

// ============================================================================
// User-level OAuth2 endpoints (any authenticated wallet, NOT admin-only)
// ============================================================================

/// GET /api/v1/oauth2/my-consents
/// Returns OAuth2 consents for the requesting wallet. Any authenticated user.
pub async fn my_oauth2_consents(
    headers: HeaderMap,
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<ConsentEntry>>, StatusCode> {
    let wallet = extract_wallet(&headers).ok_or(StatusCode::UNAUTHORIZED)?;

    let consents = state.oauth2_storage.get_consents_for_wallet(&wallet).await;

    let entries: Vec<ConsentEntry> = consents
        .into_iter()
        .map(|c| ConsentEntry {
            client_id: c.client_id,
            scopes: c.scopes,
            granted_at: c.granted_at.to_rfc3339(),
        })
        .collect();

    Ok(Json(entries))
}

/// POST /api/v1/oauth2/my-consents/revoke
/// Revokes a consent and associated tokens for the requesting wallet.
pub async fn my_revoke_consent(
    headers: HeaderMap,
    State(state): State<Arc<AppState>>,
    Json(body): Json<RevokeConsentRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let wallet = extract_wallet(&headers).ok_or(StatusCode::UNAUTHORIZED)?;

    let revoked = state.oauth2_storage.revoke_consent(&wallet, &body.client_id).await;

    Ok(Json(serde_json::json!({ "revoked": revoked, "client_id": body.client_id })))
}

/// GET /api/v1/admin/node/info
/// Returns basic node info. Admin-only.
pub async fn node_info(
    headers: HeaderMap,
    State(state): State<Arc<AppState>>,
) -> Result<Json<NodeInfoResponse>, StatusCode> {
    if !is_node_admin(&headers, &state).await {
        return Err(StatusCode::FORBIDDEN);
    }

    let uptime = state.start_time.elapsed().as_secs();
    let height = state.current_height_atomic.load(std::sync::atomic::Ordering::Relaxed);
    let network_height = state.highest_network_height.load(std::sync::atomic::Ordering::Relaxed);
    let peers = state
        .libp2p_peer_count
        .as_ref()
        .map(|c| c.load(std::sync::atomic::Ordering::Relaxed))
        .unwrap_or(0);
    let mining_healthy = state.mining_is_healthy.load(std::sync::atomic::Ordering::Relaxed);

    Ok(Json(NodeInfoResponse {
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_secs: uptime,
        height,
        network_height,
        peers,
        network_id: std::env::var("Q_NETWORK_ID").unwrap_or_else(|_| "mainnet-genesis".to_string()),
        mining_healthy,
    }))
}

// ============================================================================
// v7.3.1: Node Operator Fee Settings (master-wallet-only)
// ============================================================================

/// Check if the requesting wallet is the MASTER wallet (founder), not just admin
fn is_master_wallet(headers: &HeaderMap, _state: &AppState) -> bool {
    match extract_wallet(headers) {
        Some(wallet) => wallet == crate::aegis_auth_middleware::FOUNDER_WALLET,
        None => false,
    }
}

#[derive(Serialize)]
pub struct NodeOperatorFeeResponse {
    pub node_operator_fee_promille: u64,
    pub node_operator_fee_percent: String,
    pub dex_protocol_fee_bps: u64,
    pub dex_protocol_fee_percent: String,
    pub admin_wallet: String,
    pub admin_wallet_balance_qug: f64,
    pub founder_wallet_balance_qug: f64,
}

#[derive(Deserialize)]
pub struct UpdateOperatorFeeRequest {
    /// Promille of fees routed to node operator (0-500, i.e. 0%-50%)
    #[serde(default)]
    pub node_operator_fee_promille: Option<u64>,
    /// DEX protocol fee in basis points (0-10, i.e. 0%-0.1%)
    #[serde(default)]
    pub dex_protocol_fee_bps: Option<u64>,
}

/// GET /api/v1/admin/operator-fees
/// Returns node operator fee settings. v9.0.3: Any admin can read (master wallet required to write).
pub async fn get_operator_fees(
    headers: HeaderMap,
    State(state): State<Arc<AppState>>,
) -> Result<Json<NodeOperatorFeeResponse>, StatusCode> {
    // v9.0.3: Allow any admin to READ fee settings (not just master wallet)
    if !is_master_wallet(&headers, &state) && !is_node_admin(&headers, &state).await {
        return Err(StatusCode::FORBIDDEN);
    }

    let promille = state.node_operator_fee_promille.load(std::sync::atomic::Ordering::Relaxed);
    let dex_bps = state.dex_protocol_fee_bps.load(std::sync::atomic::Ordering::Relaxed);

    // Get admin wallet balance
    let admin_balance = {
        let balances = state.wallet_balances.read().await;
        if let Ok(bytes) = hex::decode(&state.admin_wallet) {
            if bytes.len() == 32 {
                let mut addr = [0u8; 32];
                addr.copy_from_slice(&bytes);
                balances.get(&addr).copied().unwrap_or(0) as f64 / 1e24
            } else { 0.0 }
        } else { 0.0 }
    };

    // Get founder wallet balance
    let founder_balance = {
        let balances = state.wallet_balances.read().await;
        if let Ok(bytes) = hex::decode(crate::aegis_auth_middleware::FOUNDER_WALLET) {
            if bytes.len() == 32 {
                let mut addr = [0u8; 32];
                addr.copy_from_slice(&bytes);
                balances.get(&addr).copied().unwrap_or(0) as f64 / 1e24
            } else { 0.0 }
        } else { 0.0 }
    };

    Ok(Json(NodeOperatorFeeResponse {
        node_operator_fee_promille: promille,
        node_operator_fee_percent: format!("{:.1}%", promille as f64 / 10.0),
        dex_protocol_fee_bps: dex_bps,
        dex_protocol_fee_percent: format!("{:.2}%", dex_bps as f64 / 100.0),
        admin_wallet: format!("{}...{}", &state.admin_wallet[..8], &state.admin_wallet[56..]),
        admin_wallet_balance_qug: admin_balance,
        founder_wallet_balance_qug: founder_balance,
    }))
}

/// POST /api/v1/admin/operator-fees
/// Update node operator fee settings. Master-wallet-only.
pub async fn update_operator_fees(
    headers: HeaderMap,
    State(state): State<Arc<AppState>>,
    Json(req): Json<UpdateOperatorFeeRequest>,
) -> Result<Json<NodeOperatorFeeResponse>, StatusCode> {
    if !is_master_wallet(&headers, &state) {
        return Err(StatusCode::FORBIDDEN);
    }

    if let Some(promille) = req.node_operator_fee_promille {
        if promille > 500 {
            return Err(StatusCode::BAD_REQUEST); // Max 50%
        }
        let old = state.node_operator_fee_promille.swap(promille, std::sync::atomic::Ordering::SeqCst);
        info!(
            "💰 [ADMIN] Node operator fee updated: {} promille ({:.1}%) → {} promille ({:.1}%)",
            old, old as f64 / 10.0, promille, promille as f64 / 10.0
        );
    }

    if let Some(bps) = req.dex_protocol_fee_bps {
        if bps > 10 {
            return Err(StatusCode::BAD_REQUEST); // Max 0.1%
        }
        let old = state.dex_protocol_fee_bps.swap(bps, std::sync::atomic::Ordering::SeqCst);
        info!(
            "💰 [ADMIN] DEX protocol fee updated: {} bps ({:.2}%) → {} bps ({:.2}%)",
            old, old as f64 / 100.0, bps, bps as f64 / 100.0
        );
    }

    // Return updated state
    get_operator_fees(headers, State(state)).await
}

// ============================================================================
// v8.1.1: Operator Fee Earnings API
// ============================================================================

#[derive(Serialize)]
pub struct OperatorFeeEarnings {
    pub admin_wallet: String,
    pub fee_share_promille: u64,
    pub fee_share_percent: String,
    pub session_earnings_qug: f64,
    pub total_earnings_qug: f64,
    pub fee_tx_count: u64,
    pub node_uptime_secs: u64,
}

/// GET /api/v1/admin/fee-earnings
/// Returns fee earnings for the node operator. Admin-wallet gated (not master-only).
pub async fn get_fee_earnings(
    headers: HeaderMap,
    State(state): State<Arc<AppState>>,
) -> Result<Json<OperatorFeeEarnings>, StatusCode> {
    if !is_node_admin(&headers, &state).await {
        return Err(StatusCode::FORBIDDEN);
    }

    let promille = state.node_operator_fee_promille.load(std::sync::atomic::Ordering::Relaxed);
    let session_raw = state.operator_fees_earned_session.load(std::sync::atomic::Ordering::Relaxed);
    let total_raw = state.operator_fees_earned_total.load(std::sync::atomic::Ordering::Relaxed);
    let tx_count = state.operator_fee_tx_count.load(std::sync::atomic::Ordering::Relaxed);

    // Convert from micro-QUG (1e-6) stored in AtomicU64 to QUG
    let session_qug = session_raw as f64 / 1_000_000.0;
    let total_qug = total_raw as f64 / 1_000_000.0;

    let uptime = state.start_time.elapsed().as_secs();

    Ok(Json(OperatorFeeEarnings {
        admin_wallet: format!("{}...{}", &state.admin_wallet[..8], &state.admin_wallet[56..]),
        fee_share_promille: promille,
        fee_share_percent: format!("{:.1}%", promille as f64 / 10.0),
        session_earnings_qug: session_qug,
        total_earnings_qug: total_qug,
        fee_tx_count: tx_count,
        node_uptime_secs: uptime,
    }))
}

/// Record a fee earning event (called from contracts_api when operator gets credited)
pub fn record_operator_fee(state: &AppState, amount_qug_micro: u64) {
    state.operator_fees_earned_session.fetch_add(amount_qug_micro, std::sync::atomic::Ordering::Relaxed);
    state.operator_fees_earned_total.fetch_add(amount_qug_micro, std::sync::atomic::Ordering::Relaxed);
    state.operator_fee_tx_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

// ============================================================================
// v7.3.1: Node Update Check API
// ============================================================================

#[derive(Serialize)]
pub struct NodeUpdateInfo {
    pub current_version: String,
    pub latest_version: Option<String>,
    pub update_available: bool,
    pub download_url: Option<String>,
}

/// GET /api/v1/admin/node/update-check
/// Check if a newer node binary is available. Admin-only.
pub async fn check_node_update(
    headers: HeaderMap,
    State(state): State<Arc<AppState>>,
) -> Result<Json<NodeUpdateInfo>, StatusCode> {
    if !is_node_admin(&headers, &state).await {
        return Err(StatusCode::FORBIDDEN);
    }

    let current = env!("CARGO_PKG_VERSION").to_string();

    // Check latest version from bootstrap node
    let latest = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
    {
        Ok(client) => {
            match client.get("https://quillon.xyz/api/v1/status").send().await {
                Ok(resp) => {
                    if let Ok(body) = resp.json::<serde_json::Value>().await {
                        body.get("data")
                            .and_then(|d| d.get("version"))
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string())
                    } else { None }
                }
                Err(_) => None,
            }
        }
        Err(_) => None,
    };

    let update_available = latest.as_ref().map(|l| l != &current).unwrap_or(false);
    let download_url = if update_available {
        latest.as_ref().map(|v| format!("https://quillon.xyz/downloads/q-api-server-v{}", v))
    } else { None };

    Ok(Json(NodeUpdateInfo {
        current_version: current,
        latest_version: latest,
        update_available,
        download_url,
    }))
}

// ============================================================================
// v9.1.4: Mining Mode Switch — Dynamic solo/pool mode switching for all miners
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct MiningModeSwitchRequest {
    pub target_mode: String,
    #[serde(default)]
    pub pool_url: Option<String>,
    #[serde(default)]
    pub reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct MiningModeSwitchResponse {
    pub success: bool,
    pub previous_mode: String,
    pub new_mode: String,
    pub pool_url: Option<String>,
    pub sse_subscribers: usize,
}

#[derive(Debug, Serialize)]
pub struct MiningModeStatusResponse {
    pub forced_mode: String,
    pub pool_url: Option<String>,
}

/// POST /api/v1/admin/mining/mode-switch
/// Admin-only: force all connected miners to switch mining mode at runtime.
pub async fn mining_mode_switch(
    headers: HeaderMap,
    State(state): State<Arc<AppState>>,
    Json(req): Json<MiningModeSwitchRequest>,
) -> Result<Json<MiningModeSwitchResponse>, StatusCode> {
    if !is_node_admin(&headers, &state).await {
        return Err(StatusCode::FORBIDDEN);
    }

    let target = req.target_mode.to_lowercase();
    let new_val = match target.as_str() {
        "solo" => 1u8,
        "pool" => 2u8,
        "none" | "clear" => 0u8,
        _ => {
            warn!("⛏️ [MODE-SWITCH] Invalid target_mode: {}", target);
            return Err(StatusCode::BAD_REQUEST);
        }
    };

    // Pool mode requires a pool_url
    if new_val == 2 && req.pool_url.is_none() {
        warn!("⛏️ [MODE-SWITCH] pool mode requires pool_url");
        return Err(StatusCode::BAD_REQUEST);
    }

    let old_val = state.forced_mining_mode.swap(new_val, std::sync::atomic::Ordering::SeqCst);
    let previous_mode = match old_val {
        1 => "solo".to_string(),
        2 => "pool".to_string(),
        _ => "none".to_string(),
    };
    let new_mode = match new_val {
        1 => "solo".to_string(),
        2 => "pool".to_string(),
        _ => "none".to_string(),
    };

    // Update pool URL
    {
        let mut url = state.forced_pool_url.write().await;
        *url = req.pool_url.clone();
    }

    // Broadcast SSE event to all connected miners
    let sse_subscribers = state.event_broadcaster.subscriber_count();
    let event = crate::streaming::StreamEvent::MiningModeSwitch {
        target_mode: new_mode.clone(),
        pool_url: req.pool_url.clone(),
        reason: req.reason.clone(),
        timestamp: Utc::now(),
    };
    state.event_broadcaster.broadcast(event);

    info!(
        "⛏️ [MODE-SWITCH] Admin switched mining mode: {} → {} (pool_url: {:?}, reason: {:?}, sse_subscribers: {})",
        previous_mode, new_mode, req.pool_url, req.reason, sse_subscribers
    );

    Ok(Json(MiningModeSwitchResponse {
        success: true,
        previous_mode,
        new_mode,
        pool_url: req.pool_url,
        sse_subscribers,
    }))
}

/// GET /api/v1/admin/mining/mode-status
/// Returns the current forced mining mode and pool URL.
pub async fn mining_mode_status(
    headers: HeaderMap,
    State(state): State<Arc<AppState>>,
) -> Result<Json<MiningModeStatusResponse>, StatusCode> {
    if !is_node_admin(&headers, &state).await {
        return Err(StatusCode::FORBIDDEN);
    }

    let mode_val = state.forced_mining_mode.load(std::sync::atomic::Ordering::SeqCst);
    let forced_mode = match mode_val {
        1 => "solo".to_string(),
        2 => "pool".to_string(),
        _ => "none".to_string(),
    };
    let pool_url = state.forced_pool_url.read().await.clone();

    Ok(Json(MiningModeStatusResponse {
        forced_mode,
        pool_url,
    }))
}
