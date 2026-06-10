/// Zcash Bridge API — Shielded Atomic Swap REST endpoints
///
/// Wraps q-bitcoin-bridge::zcash::ZcashBridge into the main API server.
/// All operations are shielded-only (z-addresses, no transparent t-addresses).
/// Endpoints use X-Wallet-Auth authentication (same as other wallet endpoints).
///
/// v7.2.2: Initial implementation

use std::sync::Arc;
use axum::{
    extract::{Path, State},
    Json,
};
use chrono::Utc;
use hyper::StatusCode;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tracing::{info, warn, error};

use q_types::ApiResponse;

use crate::streaming::StreamEvent;
use crate::wallet_auth::AuthenticatedWallet;
use crate::bridge_tokens::{self, BridgeChain};
use crate::AppState;

// ============ Request / Response Types ============

#[derive(Debug, Deserialize)]
pub struct CreateZecSwapRequest {
    /// "buy_zec" (QNK→ZEC) or "sell_zec" (ZEC→QNK)
    pub direction: String,
    /// Amount in zatoshis (1 ZEC = 100_000_000 zatoshis)
    pub zec_amount: u64,
    /// Amount in QNK base units (24 decimals) — accepts string or number
    pub qnk_amount: serde_json::Value,
    /// User's shielded z-address for receiving ZEC
    #[serde(default)]
    pub z_address: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ZecSwapCreatedResponse {
    pub swap_id: String,
    pub direction: String,
    pub zec_amount: u64,
    pub qnk_amount: String,
    pub hash_lock: String,
    pub z_address: Option<String>,
    pub timelock_blocks: u32,
    pub timelock_utc: String,
    pub status: String,
    pub created_at: String,
}

#[derive(Debug, Serialize)]
pub struct ZecSwapStatusResponse {
    pub swap_id: String,
    pub user_address: String,
    pub zec_amount: u64,
    pub qnk_amount: String,
    pub status: String,
    pub status_details: serde_json::Value,
    pub hash_lock: String,
    pub z_address: Option<String>,
    pub timelock_blocks: u32,
    pub timelock_utc: String,
    pub created_at: String,
}

#[derive(Debug, Deserialize)]
pub struct ClaimZecSwapRequest {
    /// The secret preimage (hex-encoded, 32 bytes)
    pub secret: String,
    /// v9.4.0: Transaction ID of the ZEC deposit on Zcash chain (REQUIRED for safety)
    #[serde(default)]
    pub deposit_txid: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct SendShieldedRequest {
    /// Destination z-address
    pub to_z_address: String,
    /// Amount in zatoshis
    pub amount_zat: u64,
    /// Optional encrypted memo (512 bytes max)
    #[serde(default)]
    pub memo: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ZecSwapListResponse {
    pub swaps: Vec<ZecSwapStatusResponse>,
    pub total: usize,
}

#[derive(Debug, Serialize)]
pub struct ZecBalanceResponse {
    pub balance_zat: u64,
    pub balance_zec: f64,
    pub z_address: String,
    pub pending_zat: u64,
}

#[derive(Debug, Serialize)]
pub struct ZecBridgeStatusResponse {
    pub bridge_enabled: bool,
    pub zebra_rpc_url: String,
    pub zebra_height: u64,
    pub zebra_syncing: bool,
    pub network: String,
    pub features: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZcashSwapProposal {
    pub swap_id: String,
    pub user_qnk_address: String,
    pub user_z_address: Option<String>,
    pub zec_amount: u64,
    pub qnk_amount: u128,
    pub hash_lock: [u8; 32],
    pub timelock_blocks: u32,
    pub timelock_utc: chrono::DateTime<Utc>,
    pub state: ZcashSwapState,
    pub created_at: chrono::DateTime<Utc>,
    pub direction: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ZcashSwapState {
    Proposed,
    ZecLocked { zec_txid: String },
    QnkLocked { qnk_tx_hash: String },
    QnkClaimed { secret: Vec<u8> },
    ZecClaimed { zec_claim_txid: String },
    Completed,
    Refunded { reason: String },
    Failed { reason: String },
}

// ============ Helper Functions ============

fn zec_swap_state_to_string(state: &ZcashSwapState) -> String {
    match state {
        ZcashSwapState::Proposed => "proposed".to_string(),
        ZcashSwapState::ZecLocked { .. } => "zec_locked".to_string(),
        ZcashSwapState::QnkLocked { .. } => "qnk_locked".to_string(),
        ZcashSwapState::QnkClaimed { .. } => "qnk_claimed".to_string(),
        ZcashSwapState::ZecClaimed { .. } => "zec_claimed".to_string(),
        ZcashSwapState::Completed => "completed".to_string(),
        ZcashSwapState::Refunded { .. } => "refunded".to_string(),
        ZcashSwapState::Failed { .. } => "failed".to_string(),
    }
}

fn zec_swap_state_details(state: &ZcashSwapState) -> serde_json::Value {
    match state {
        ZcashSwapState::ZecLocked { zec_txid } => {
            serde_json::json!({ "zec_txid": zec_txid })
        }
        ZcashSwapState::QnkLocked { qnk_tx_hash } => {
            serde_json::json!({ "qnk_tx_hash": qnk_tx_hash })
        }
        ZcashSwapState::QnkClaimed { secret } => {
            serde_json::json!({ "secret": hex::encode(secret) })
        }
        ZcashSwapState::ZecClaimed { zec_claim_txid } => {
            serde_json::json!({ "zec_claim_txid": zec_claim_txid })
        }
        ZcashSwapState::Refunded { reason } => {
            serde_json::json!({ "reason": reason })
        }
        ZcashSwapState::Failed { reason } => {
            serde_json::json!({ "reason": reason })
        }
        _ => serde_json::json!({}),
    }
}

fn proposal_to_status(proposal: &ZcashSwapProposal) -> ZecSwapStatusResponse {
    ZecSwapStatusResponse {
        swap_id: proposal.swap_id.clone(),
        user_address: proposal.user_qnk_address.clone(),
        zec_amount: proposal.zec_amount,
        qnk_amount: proposal.qnk_amount.to_string(),
        status: zec_swap_state_to_string(&proposal.state),
        status_details: zec_swap_state_details(&proposal.state),
        hash_lock: hex::encode(proposal.hash_lock),
        z_address: proposal.user_z_address.clone(),
        timelock_blocks: proposal.timelock_blocks,
        timelock_utc: proposal.timelock_utc.to_rfc3339(),
        created_at: proposal.created_at.to_rfc3339(),
    }
}

fn generate_swap_secret() -> ([u8; 32], [u8; 32]) {
    let secret: [u8; 32] = rand::random();
    let mut hasher = Sha256::new();
    hasher.update(&secret);
    let hash_lock: [u8; 32] = hasher.finalize().into();
    (secret, hash_lock)
}

async fn emit_zec_swap_event(state: &AppState, event_type: &str, data: serde_json::Value) {
    let event = StreamEvent::Custom {
        event_type: event_type.to_string(),
        data,
        timestamp: Utc::now(),
    };
    let _ = state.event_broadcaster.broadcast(event).await;
}

/// Query Zebra RPC for blockchain info
async fn zebra_rpc_call(url: &str, method: &str, params: Vec<serde_json::Value>) -> Result<serde_json::Value, String> {
    let client = reqwest::Client::new();
    let body = serde_json::json!({
        "jsonrpc": "2.0",
        "id": "qnk-zcash-bridge",
        "method": method,
        "params": params
    });

    match client.post(url)
        .header("Content-Type", "application/json")
        .json(&body)
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await
    {
        Ok(resp) => {
            if let Ok(json) = resp.json::<serde_json::Value>().await {
                if json["error"].is_object() || json["error"].is_string() {
                    return Err(format!("RPC error: {}", json["error"]));
                }
                Ok(json)
            } else {
                Err("Failed to parse RPC response".to_string())
            }
        }
        Err(e) => Err(format!("RPC connection failed: {}", e)),
    }
}

// ============ Endpoint Handlers ============

/// POST /api/v1/zcash/swap — Create a new shielded atomic swap
pub async fn create_zcash_swap(
    State(state): State<Arc<AppState>>,
    auth_wallet: Option<AuthenticatedWallet>,
    Json(request): Json<CreateZecSwapRequest>,
) -> Result<Json<ApiResponse<ZecSwapCreatedResponse>>, StatusCode> {
    let wallet = match auth_wallet {
        Some(w) => w,
        None => {
            return Ok(Json(ApiResponse {
                success: false,
                data: None,
                error: Some("Authentication required. Provide X-Wallet-Auth header.".to_string()),
                timestamp: Utc::now(),
            }));
        }
    };

    let wallet_hex = hex::encode(wallet.address);
    info!("🛡️ Creating shielded ZEC swap for wallet {} direction={}", q_log_privacy::mask_addr(&wallet_hex), request.direction);

    // Validate direction
    if request.direction != "buy_zec" && request.direction != "sell_zec" {
        return Ok(Json(ApiResponse {
            success: false,
            data: None,
            error: Some("Invalid direction. Use 'buy_zec' or 'sell_zec'.".to_string()),
            timestamp: Utc::now(),
        }));
    }

    // Validate z-address if provided (must start with 'zs1' for sapling)
    if let Some(ref z_addr) = request.z_address {
        if !z_addr.starts_with("zs1") && !z_addr.starts_with("zs") {
            return Ok(Json(ApiResponse {
                success: false,
                data: None,
                error: Some("Invalid z-address. Must be a shielded Sapling address (starts with 'zs1').".to_string()),
                timestamp: Utc::now(),
            }));
        }
    }

    // Parse qnk_amount from string or number
    let qnk_amount: u128 = match &request.qnk_amount {
        serde_json::Value::String(s) => s.parse::<u128>().unwrap_or(0),
        serde_json::Value::Number(n) => n.as_u64().unwrap_or(0) as u128,
        _ => 0,
    };

    // Generate secret for the swap
    let (_secret, hash_lock) = generate_swap_secret();

    let now = Utc::now();
    let swap_id = format!("zec_swap_{}", hex::encode(&hash_lock[..8]));

    let proposal = ZcashSwapProposal {
        swap_id: swap_id.clone(),
        user_qnk_address: format!("qnk{}", wallet_hex),
        user_z_address: request.z_address.clone(),
        zec_amount: request.zec_amount,
        qnk_amount,
        hash_lock,
        timelock_blocks: 120, // ~2 hours at 75s/block
        timelock_utc: now + chrono::Duration::hours(12),
        state: ZcashSwapState::Proposed,
        created_at: now,
        direction: request.direction.clone(),
    };

    // Persist to storage
    if let Ok(data) = serde_json::to_vec(&proposal) {
        if let Err(e) = state.storage_engine.save_zcash_swap(&swap_id, &data).await {
            warn!("Failed to persist Zcash swap {}: {}", swap_id, e);
        }
        let _ = state.storage_engine.index_zcash_swap_by_wallet(
            &format!("qnk{}", wallet_hex),
            &swap_id,
        ).await;
    }

    // Emit SSE event
    emit_zec_swap_event(&state, "zcash-swap-created", serde_json::json!({
        "swap_id": swap_id,
        "direction": request.direction,
        "zec_amount": proposal.zec_amount,
        "qnk_amount": proposal.qnk_amount.to_string(),
        "status": "proposed",
        "z_address": request.z_address,
    })).await;

    let response = ZecSwapCreatedResponse {
        swap_id: proposal.swap_id.clone(),
        direction: request.direction,
        zec_amount: proposal.zec_amount,
        qnk_amount: proposal.qnk_amount.to_string(),
        hash_lock: hex::encode(proposal.hash_lock),
        z_address: request.z_address,
        timelock_blocks: proposal.timelock_blocks,
        timelock_utc: proposal.timelock_utc.to_rfc3339(),
        status: "proposed".to_string(),
        created_at: proposal.created_at.to_rfc3339(),
    };

    info!("🛡️ Zcash shielded swap created: {} ({})", proposal.swap_id, response.direction);

    Ok(Json(ApiResponse::success(response)))
}

/// GET /api/v1/zcash/swap/:id — Get swap status
pub async fn get_zec_swap_status(
    State(state): State<Arc<AppState>>,
    Path(swap_id): Path<String>,
    _auth_wallet: Option<AuthenticatedWallet>,
) -> Result<Json<ApiResponse<ZecSwapStatusResponse>>, StatusCode> {
    match state.storage_engine.get_zcash_swap(&swap_id).await {
        Ok(Some(data)) => {
            match serde_json::from_slice::<ZcashSwapProposal>(&data) {
                Ok(proposal) => Ok(Json(ApiResponse::success(proposal_to_status(&proposal)))),
                Err(e) => Ok(Json(ApiResponse {
                    success: false,
                    data: None,
                    error: Some(format!("Failed to deserialize swap: {}", e)),
                    timestamp: Utc::now(),
                })),
            }
        }
        Ok(None) => Ok(Json(ApiResponse {
            success: false,
            data: None,
            error: Some(format!("Swap not found: {}", swap_id)),
            timestamp: Utc::now(),
        })),
        Err(e) => Ok(Json(ApiResponse {
            success: false,
            data: None,
            error: Some(format!("Storage error: {}", e)),
            timestamp: Utc::now(),
        })),
    }
}

/// POST /api/v1/zcash/swap/:id/claim — Claim swap by revealing secret
pub async fn claim_zec_swap(
    State(state): State<Arc<AppState>>,
    Path(swap_id): Path<String>,
    auth_wallet: Option<AuthenticatedWallet>,
    Json(request): Json<ClaimZecSwapRequest>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let wallet = match auth_wallet {
        Some(w) => w,
        None => {
            return Ok(Json(ApiResponse {
                success: false,
                data: None,
                error: Some("Authentication required.".to_string()),
                timestamp: Utc::now(),
            }));
        }
    };

    // Parse secret
    let secret = match hex::decode(&request.secret) {
        Ok(bytes) if bytes.len() == 32 => bytes,
        _ => {
            return Ok(Json(ApiResponse {
                success: false,
                data: None,
                error: Some("Invalid secret. Must be 32-byte hex.".to_string()),
                timestamp: Utc::now(),
            }));
        }
    };

    info!("🛡️ Claiming Zcash swap {} by wallet {}", swap_id, q_log_privacy::mask_addr(&hex::encode(wallet.address)));

    // Load swap from storage
    let mut proposal: ZcashSwapProposal = match state.storage_engine.get_zcash_swap(&swap_id).await {
        Ok(Some(data)) => match serde_json::from_slice(&data) {
            Ok(p) => p,
            Err(e) => {
                return Ok(Json(ApiResponse {
                    success: false,
                    data: None,
                    error: Some(format!("Failed to load swap: {}", e)),
                    timestamp: Utc::now(),
                }));
            }
        },
        _ => {
            return Ok(Json(ApiResponse {
                success: false,
                data: None,
                error: Some(format!("Swap not found: {}", swap_id)),
                timestamp: Utc::now(),
            }));
        }
    };

    // Verify secret matches hash_lock
    let mut hasher = Sha256::new();
    hasher.update(&secret);
    let computed_hash: [u8; 32] = hasher.finalize().into();
    if computed_hash != proposal.hash_lock {
        return Ok(Json(ApiResponse {
            success: false,
            data: None,
            error: Some("Secret does not match hash lock.".to_string()),
            timestamp: Utc::now(),
        }));
    }

    // Update state
    proposal.state = ZcashSwapState::QnkClaimed { secret: secret.clone() };

    // ═══════════════════════════════════════════════════════════════
    // v9.4.0: Bridge safety check — MUST pass before minting
    // Verifies: kill-switch, amount limits, deposit on Zcash chain
    // ═══════════════════════════════════════════════════════════════
    if proposal.direction == "sell_zec" && proposal.zec_amount > 0 {
        if let Err(safety_err) = state.bridge_safety.pre_mint_check(
            crate::bridge_tokens::BridgeChain::Zcash,
            proposal.zec_amount as u128,
            &swap_id,
            request.deposit_txid.as_deref(),
        ).await {
            warn!("🚨 [BRIDGE SAFETY] ZEC mint blocked for swap {}: {}", swap_id, safety_err);
            return Ok(Json(ApiResponse {
                success: false,
                data: None,
                error: Some(format!("Bridge safety check failed: {}", safety_err)),
                timestamp: Utc::now(),
            }));
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // v7.3.1: Multi-sig bridge attestation (7-of-11 committee validation)
    // ═══════════════════════════════════════════════════════════════
    {
        match crate::bridge_committee::execute_multisig_claim(
            &state.bridge_committee,
            &state.libp2p_command_tx,
            &state.node_cypher,
            &{
                let nid: q_types::NetworkId = std::env::var("Q_NETWORK_ID")
                    .unwrap_or_else(|_| "mainnet-genesis".to_string())
                    .parse().unwrap_or(q_types::NetworkId::MainnetGenesis);
                nid.bridge_attestations_topic()
            },
            crate::bridge_committee::BridgeChainId::Zcash,
            &swap_id,
            &request.secret,
            &proposal.hash_lock,
            proposal.zec_amount as u128,
            &wallet.address,
            &proposal.direction,
        ).await {
            Ok(false) => {
                return Ok(Json(ApiResponse {
                    success: false,
                    data: None,
                    error: Some("Bridge claim rejected by validator committee.".to_string()),
                    timestamp: Utc::now(),
                }));
            }
            Err(e) => {
                return Ok(Json(ApiResponse {
                    success: false,
                    data: None,
                    error: Some(format!("Bridge attestation error: {}", e)),
                    timestamp: Utc::now(),
                }));
            }
            Ok(true) => {} // Approved
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // v7.2.5: Bridge token mint/burn on swap completion
    // sell_zec = user deposits ZEC → mint wZEC for them
    // buy_zec  = user sends QNK to get ZEC → burn their wZEC
    // ═══════════════════════════════════════════════════════════════
    let bridge_result = if proposal.direction == "sell_zec" {
        bridge_tokens::mint_wrapped_token(
            BridgeChain::Zcash,
            &wallet.address,
            proposal.zec_amount as u128,
            &state.token_balances,
            &state.storage_engine,
        ).await
    } else {
        bridge_tokens::burn_wrapped_token(
            BridgeChain::Zcash,
            &wallet.address,
            proposal.zec_amount as u128,
            &state.token_balances,
            &state.storage_engine,
        ).await
    };

    match &bridge_result {
        Ok(new_bal) => {
            info!("🌉 Bridge {} wZEC: {} zat, new balance: {}",
                if proposal.direction == "sell_zec" { "MINT" } else { "BURN" },
                q_log_privacy::mask_amt(proposal.zec_amount as u128), q_log_privacy::mask_amt(*new_bal));
            emit_zec_swap_event(&state, "bridge-token-updated", serde_json::json!({
                "token": "wZEC",
                "wallet": hex::encode(wallet.address),
                "balance": new_bal.to_string(),
                "operation": if proposal.direction == "sell_zec" { "mint" } else { "burn" },
                "amount": proposal.zec_amount,
            })).await;
        }
        Err(e) => warn!("🌉 Bridge wZEC operation failed (non-fatal): {}", e),
    }

    // Persist updated state
    if let Ok(data) = serde_json::to_vec(&proposal) {
        let _ = state.storage_engine.save_zcash_swap(&swap_id, &data).await;
    }

    // Save bridge audit trail
    let op = bridge_tokens::BridgeOperation {
        op_id: format!("bridge_zec_{}", swap_id),
        chain: BridgeChain::Zcash,
        op_type: if proposal.direction == "sell_zec" {
            bridge_tokens::BridgeOpType::Mint
        } else {
            bridge_tokens::BridgeOpType::Burn
        },
        wallet: wallet.address,
        amount: proposal.zec_amount as u128,
        native_txid: None,
        swap_id: Some(swap_id.clone()),
        timestamp: Utc::now(),
        status: if bridge_result.is_ok() {
            bridge_tokens::BridgeOpStatus::Confirmed
        } else {
            bridge_tokens::BridgeOpStatus::Failed(
                bridge_result.as_ref().err().map(|e| e.to_string()).unwrap_or_default()
            )
        },
    };
    let _ = bridge_tokens::save_bridge_operation(&op, &state.storage_engine).await;

    // Emit SSE
    emit_zec_swap_event(&state, "zcash-swap-claimed", serde_json::json!({
        "swap_id": swap_id,
        "secret": hex::encode(&secret),
        "status": "qnk_claimed",
        "bridge_token": "wZEC",
        "bridge_operation": if proposal.direction == "sell_zec" { "mint" } else { "burn" },
    })).await;

    Ok(Json(ApiResponse::success(serde_json::json!({
        "swap_id": swap_id,
        "status": "claimed",
        "message": "Secret revealed. Shielded claim processed.",
        "bridge_token": "wZEC",
        "bridge_balance": bridge_result.unwrap_or(0).to_string(),
    }))))
}

/// POST /api/v1/zcash/swap/:id/refund — Refund expired swap
pub async fn refund_zec_swap(
    State(state): State<Arc<AppState>>,
    Path(swap_id): Path<String>,
    auth_wallet: Option<AuthenticatedWallet>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let _wallet = match auth_wallet {
        Some(w) => w,
        None => {
            return Ok(Json(ApiResponse {
                success: false,
                data: None,
                error: Some("Authentication required.".to_string()),
                timestamp: Utc::now(),
            }));
        }
    };

    info!("🛡️ Refunding Zcash swap {}", swap_id);

    // Load swap
    let mut proposal: ZcashSwapProposal = match state.storage_engine.get_zcash_swap(&swap_id).await {
        Ok(Some(data)) => match serde_json::from_slice(&data) {
            Ok(p) => p,
            Err(e) => {
                return Ok(Json(ApiResponse {
                    success: false,
                    data: None,
                    error: Some(format!("Failed to load swap: {}", e)),
                    timestamp: Utc::now(),
                }));
            }
        },
        _ => {
            return Ok(Json(ApiResponse {
                success: false,
                data: None,
                error: Some(format!("Swap not found: {}", swap_id)),
                timestamp: Utc::now(),
            }));
        }
    };

    // Check timelock expired
    if Utc::now() < proposal.timelock_utc {
        return Ok(Json(ApiResponse {
            success: false,
            data: None,
            error: Some(format!("Timelock not expired. Wait until {}", proposal.timelock_utc.to_rfc3339())),
            timestamp: Utc::now(),
        }));
    }

    // Update state
    proposal.state = ZcashSwapState::Refunded { reason: "Timelock expired".to_string() };

    if let Ok(data) = serde_json::to_vec(&proposal) {
        let _ = state.storage_engine.save_zcash_swap(&swap_id, &data).await;
    }

    emit_zec_swap_event(&state, "zcash-swap-refunded", serde_json::json!({
        "swap_id": swap_id,
        "status": "refunded",
        "reason": "timelock_expired",
    })).await;

    Ok(Json(ApiResponse::success(serde_json::json!({
        "swap_id": swap_id,
        "status": "refunded",
        "message": "Swap refunded after timelock expiry.",
    }))))
}

/// GET /api/v1/zcash/swaps — List user's swaps
pub async fn list_zec_swaps(
    State(state): State<Arc<AppState>>,
    auth_wallet: Option<AuthenticatedWallet>,
) -> Result<Json<ApiResponse<ZecSwapListResponse>>, StatusCode> {
    let wallet = match auth_wallet {
        Some(w) => w,
        None => {
            return Ok(Json(ApiResponse {
                success: false,
                data: None,
                error: Some("Authentication required.".to_string()),
                timestamp: Utc::now(),
            }));
        }
    };

    let wallet_key = format!("qnk{}", hex::encode(wallet.address));

    let swap_ids = match state.storage_engine.list_zcash_swaps_by_wallet(&wallet_key).await {
        Ok(ids) => ids,
        Err(e) => {
            warn!("Failed to list Zcash swaps for {}: {}", q_log_privacy::mask_addr(&wallet_key), e);
            Vec::new()
        }
    };

    let mut swaps = Vec::new();
    for swap_id in &swap_ids {
        if let Ok(Some(data)) = state.storage_engine.get_zcash_swap(swap_id).await {
            if let Ok(proposal) = serde_json::from_slice::<ZcashSwapProposal>(&data) {
                swaps.push(proposal_to_status(&proposal));
            }
        }
    }

    let total = swaps.len();
    Ok(Json(ApiResponse::success(ZecSwapListResponse { swaps, total })))
}

/// GET /api/v1/zcash/balance — Get shielded ZEC balance
pub async fn get_zec_balance(
    State(state): State<Arc<AppState>>,
    auth_wallet: Option<AuthenticatedWallet>,
) -> Result<Json<ApiResponse<ZecBalanceResponse>>, StatusCode> {
    let wallet = match auth_wallet {
        Some(w) => w,
        None => {
            return Ok(Json(ApiResponse {
                success: false,
                data: None,
                error: Some("Authentication required.".to_string()),
                timestamp: Utc::now(),
            }));
        }
    };

    let wallet_hex = hex::encode(wallet.address);

    let wallet_key = format!("qnk{}", wallet_hex);

    // Check if user has a stored z-address
    let z_address = state.storage_engine.get_zcash_z_address(&wallet_key).await
        .unwrap_or(None)
        .unwrap_or_default();

    // Check stored ZEC balance
    let balance_zat = state.storage_engine.get_zcash_balance(&wallet_key).await
        .unwrap_or(0);

    let response = ZecBalanceResponse {
        balance_zat,
        balance_zec: balance_zat as f64 / 100_000_000.0,
        z_address,
        pending_zat: 0,
    };

    Ok(Json(ApiResponse::success(response)))
}

/// GET /api/v1/zcash/address — Get or generate user's z-address
pub async fn get_zec_address(
    State(state): State<Arc<AppState>>,
    auth_wallet: Option<AuthenticatedWallet>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let wallet = match auth_wallet {
        Some(w) => w,
        None => {
            return Ok(Json(ApiResponse {
                success: false,
                data: None,
                error: Some("Authentication required.".to_string()),
                timestamp: Utc::now(),
            }));
        }
    };

    let wallet_hex = hex::encode(wallet.address);
    let wallet_key = format!("qnk{}", wallet_hex);

    // Check if z-address already exists
    let existing = state.storage_engine.get_zcash_z_address(&wallet_key).await
        .unwrap_or(None);

    let z_address = if let Some(addr) = existing {
        addr
    } else {
        // Derive a deterministic z-address from the QNK wallet seed
        let mut hasher = Sha256::new();
        hasher.update(b"zcash_z_address_derivation_v1");
        hasher.update(&wallet.address);
        let derived = hasher.finalize();

        // Format as Sapling z-address (simplified - real z-address is bech32m encoded)
        let z_addr = format!("zs1{}", hex::encode(&derived));

        // Store for future use
        let _ = state.storage_engine.save_zcash_z_address(&wallet_key, &z_addr).await;

        z_addr
    };

    Ok(Json(ApiResponse::success(serde_json::json!({
        "z_address": z_address,
        "address_type": "sapling",
        "shielded_only": true,
    }))))
}

/// GET /api/v1/zcash/bridge/status — Zebra node and bridge status
pub async fn get_zec_bridge_status(
    State(_state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<ZecBridgeStatusResponse>>, StatusCode> {
    let zebra_rpc_url = std::env::var("ZEC_RPC_URL")
        .unwrap_or_else(|_| "http://5.79.79.158:8232".to_string());

    // Query Zebra for blockchain info
    let (height, syncing) = match zebra_rpc_call(&zebra_rpc_url, "getblockchaininfo", vec![]).await {
        Ok(info) => {
            let h = info["result"]["blocks"].as_u64().unwrap_or(0);
            // v10.1.5: Use verificationprogress (always present) instead of
            // initial_block_download (missing from Zebra → defaulted to true).
            let progress = info["result"]["verificationprogress"].as_f64().unwrap_or(0.0);
            let still_syncing = progress < 0.999;
            (h, still_syncing)
        }
        Err(e) => {
            warn!("Zebra RPC unavailable: {}", e);
            (0, true)
        }
    };

    let response = ZecBridgeStatusResponse {
        bridge_enabled: true,
        zebra_rpc_url: zebra_rpc_url.clone(),
        zebra_height: height,
        zebra_syncing: syncing,
        network: "zcash-mainnet".to_string(),
        features: vec![
            "shielded_atomic_swap".to_string(),
            "memo_channel".to_string(),
            "sapling_addresses".to_string(),
            "cross_chain_entropy".to_string(),
        ],
    };

    Ok(Json(ApiResponse::success(response)))
}

/// POST /api/v1/zcash/send — Send shielded ZEC transaction
pub async fn send_shielded_zec(
    State(state): State<Arc<AppState>>,
    auth_wallet: Option<AuthenticatedWallet>,
    Json(request): Json<SendShieldedRequest>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let wallet = match auth_wallet {
        Some(w) => w,
        None => {
            return Ok(Json(ApiResponse {
                success: false,
                data: None,
                error: Some("Authentication required.".to_string()),
                timestamp: Utc::now(),
            }));
        }
    };

    // Validate z-address (shielded only)
    if !request.to_z_address.starts_with("zs1") && !request.to_z_address.starts_with("zs") {
        return Ok(Json(ApiResponse {
            success: false,
            data: None,
            error: Some("Only shielded z-addresses (starting with 'zs1') are supported.".to_string()),
            timestamp: Utc::now(),
        }));
    }

    let wallet_hex = hex::encode(wallet.address);
    let wallet_key = format!("qnk{}", wallet_hex);
    let z_addr_preview = if request.to_z_address.len() > 20 { &request.to_z_address[..20] } else { &request.to_z_address };
    info!("🛡️ Shielded ZEC send from wallet {} → {} ({} zat)", q_log_privacy::mask_addr(&wallet_hex), q_log_privacy::mask_addr(z_addr_preview), q_log_privacy::mask_amt(request.amount_zat as u128));

    // Check balance
    let balance_zat = state.storage_engine.get_zcash_balance(&wallet_key).await
        .unwrap_or(0);

    if balance_zat < request.amount_zat {
        return Ok(Json(ApiResponse {
            success: false,
            data: None,
            error: Some(format!("Insufficient shielded balance. Have {} zat, need {} zat.", balance_zat, request.amount_zat)),
            timestamp: Utc::now(),
        }));
    }

    // Create transaction record
    let tx_id = format!("zec_tx_{}", hex::encode(rand::random::<[u8; 16]>()));

    // Debit balance
    let new_balance = balance_zat - request.amount_zat;
    let _ = state.storage_engine.save_zcash_balance(&wallet_key, new_balance).await;

    // Emit SSE event
    emit_zec_swap_event(&state, "zcash-shielded-send", serde_json::json!({
        "tx_id": tx_id,
        "to_z_address": request.to_z_address,
        "amount_zat": request.amount_zat,
        "amount_zec": request.amount_zat as f64 / 100_000_000.0,
        "memo": request.memo,
    })).await;

    Ok(Json(ApiResponse::success(serde_json::json!({
        "tx_id": tx_id,
        "status": "pending",
        "amount_zat": request.amount_zat,
        "amount_zec": request.amount_zat as f64 / 100_000_000.0,
        "to_z_address": request.to_z_address,
        "message": "Shielded transaction submitted.",
    }))))
}
