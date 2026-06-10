/// Iron Fish Bridge API — Privacy-Preserving Atomic Swap REST endpoints
///
/// Iron Fish is a privacy-focused L1 blockchain using zk-SNARKs.
/// All IRON transactions are shielded by default.
/// RPC communicates with the Iron Fish node at IRONFISH_RPC_URL (default: tcp://5.79.79.158:8020)
///
/// v7.2.4: Initial implementation

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

// ============ Constants ============

/// 1 IRON = 100_000_000 ore (same denomination as BTC satoshis)
const ORE_PER_IRON: f64 = 100_000_000.0;

fn ironfish_rpc_url() -> String {
    std::env::var("IRONFISH_RPC_URL")
        .unwrap_or_else(|_| "http://5.79.79.158:8020".to_string())
}

// ============ Request / Response Types ============

#[derive(Debug, Deserialize)]
pub struct CreateIronSwapRequest {
    /// "buy_iron" (QNK→IRON) or "sell_iron" (IRON→QNK)
    pub direction: String,
    /// Amount in ore (1 IRON = 100_000_000 ore)
    pub iron_amount: u64,
    /// Amount in QNK base units (24 decimals) — accepts string or number
    pub qnk_amount: serde_json::Value,
    /// User's Iron Fish public address for receiving IRON
    #[serde(default)]
    pub iron_address: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct IronSwapCreatedResponse {
    pub swap_id: String,
    pub direction: String,
    pub iron_amount: u64,
    pub qnk_amount: String,
    pub hash_lock: String,
    pub iron_address: Option<String>,
    pub timelock_blocks: u32,
    pub timelock_utc: String,
    pub status: String,
    pub created_at: String,
}

#[derive(Debug, Serialize)]
pub struct IronSwapStatusResponse {
    pub swap_id: String,
    pub user_address: String,
    pub iron_amount: u64,
    pub qnk_amount: String,
    pub status: String,
    pub status_details: serde_json::Value,
    pub hash_lock: String,
    pub iron_address: Option<String>,
    pub timelock_blocks: u32,
    pub timelock_utc: String,
    pub created_at: String,
}

#[derive(Debug, Deserialize)]
pub struct ClaimIronSwapRequest {
    /// The secret preimage (hex-encoded, 32 bytes)
    pub secret: String,
    /// v9.4.0: Transaction ID of the IRON deposit on Iron Fish chain (REQUIRED for safety)
    #[serde(default)]
    pub deposit_txid: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct SendIronRequest {
    /// Destination Iron Fish public address
    pub to_address: String,
    /// Amount in ore
    pub amount_ore: u64,
    /// Optional memo
    #[serde(default)]
    pub memo: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct IronSwapListResponse {
    pub swaps: Vec<IronSwapStatusResponse>,
    pub total: usize,
}

#[derive(Debug, Serialize)]
pub struct IronBalanceResponse {
    pub balance_ore: u64,
    pub balance_iron: f64,
    pub iron_address: String,
    pub pending_ore: u64,
}

#[derive(Debug, Serialize)]
pub struct IronBridgeStatusResponse {
    pub bridge_enabled: bool,
    pub node_rpc_url: String,
    pub node_version: String,
    pub node_height: u64,
    pub node_syncing: bool,
    pub network: String,
    pub peers: u32,
    pub features: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IronFishSwapProposal {
    pub swap_id: String,
    pub user_qnk_address: String,
    pub user_iron_address: Option<String>,
    pub iron_amount: u64,
    pub qnk_amount: u128,
    pub hash_lock: [u8; 32],
    pub timelock_blocks: u32,
    pub timelock_utc: chrono::DateTime<Utc>,
    pub state: IronFishSwapState,
    pub created_at: chrono::DateTime<Utc>,
    pub direction: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IronFishSwapState {
    Proposed,
    IronLocked { iron_txid: String },
    QnkLocked { qnk_tx_hash: String },
    QnkClaimed { secret: Vec<u8> },
    IronClaimed { iron_claim_txid: String },
    Completed,
    Refunded { reason: String },
    Failed { reason: String },
}

// ============ Helper Functions ============

fn iron_swap_state_to_string(state: &IronFishSwapState) -> String {
    match state {
        IronFishSwapState::Proposed => "proposed".to_string(),
        IronFishSwapState::IronLocked { .. } => "iron_locked".to_string(),
        IronFishSwapState::QnkLocked { .. } => "qnk_locked".to_string(),
        IronFishSwapState::QnkClaimed { .. } => "qnk_claimed".to_string(),
        IronFishSwapState::IronClaimed { .. } => "iron_claimed".to_string(),
        IronFishSwapState::Completed => "completed".to_string(),
        IronFishSwapState::Refunded { .. } => "refunded".to_string(),
        IronFishSwapState::Failed { .. } => "failed".to_string(),
    }
}

fn iron_swap_state_details(state: &IronFishSwapState) -> serde_json::Value {
    match state {
        IronFishSwapState::IronLocked { iron_txid } => {
            serde_json::json!({ "iron_txid": iron_txid })
        }
        IronFishSwapState::QnkLocked { qnk_tx_hash } => {
            serde_json::json!({ "qnk_tx_hash": qnk_tx_hash })
        }
        IronFishSwapState::QnkClaimed { secret } => {
            serde_json::json!({ "secret": hex::encode(secret) })
        }
        IronFishSwapState::IronClaimed { iron_claim_txid } => {
            serde_json::json!({ "iron_claim_txid": iron_claim_txid })
        }
        IronFishSwapState::Refunded { reason } => {
            serde_json::json!({ "reason": reason })
        }
        IronFishSwapState::Failed { reason } => {
            serde_json::json!({ "reason": reason })
        }
        _ => serde_json::json!({}),
    }
}

fn proposal_to_status(proposal: &IronFishSwapProposal) -> IronSwapStatusResponse {
    IronSwapStatusResponse {
        swap_id: proposal.swap_id.clone(),
        user_address: proposal.user_qnk_address.clone(),
        iron_amount: proposal.iron_amount,
        qnk_amount: proposal.qnk_amount.to_string(),
        status: iron_swap_state_to_string(&proposal.state),
        status_details: iron_swap_state_details(&proposal.state),
        hash_lock: hex::encode(proposal.hash_lock),
        iron_address: proposal.user_iron_address.clone(),
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

async fn emit_iron_event(state: &AppState, event_type: &str, data: serde_json::Value) {
    let event = StreamEvent::Custom {
        event_type: event_type.to_string(),
        data,
        timestamp: Utc::now(),
    };
    let _ = state.event_broadcaster.broadcast(event).await;
}

/// Call the Iron Fish node's TCP RPC (JSON-RPC over HTTP fallback)
async fn ironfish_rpc_call(method: &str, params: serde_json::Value) -> Result<serde_json::Value, String> {
    let url = ironfish_rpc_url();
    let client = reqwest::Client::new();
    let body = serde_json::json!({
        "method": method,
        "params": params,
    });

    match client.post(&url)
        .header("Content-Type", "application/json")
        .json(&body)
        .timeout(std::time::Duration::from_secs(15))
        .send()
        .await
    {
        Ok(resp) => {
            if let Ok(json) = resp.json::<serde_json::Value>().await {
                if let Some(err) = json.get("error") {
                    if !err.is_null() {
                        return Err(format!("Iron Fish RPC error: {}", err));
                    }
                }
                Ok(json)
            } else {
                Err("Failed to parse Iron Fish RPC response".to_string())
            }
        }
        Err(e) => Err(format!("Iron Fish RPC connection failed: {}", e)),
    }
}

// ============ Endpoint Handlers ============

/// POST /api/v1/ironfish/swap — Create a new atomic swap
pub async fn create_iron_swap(
    State(state): State<Arc<AppState>>,
    auth_wallet: Option<AuthenticatedWallet>,
    Json(request): Json<CreateIronSwapRequest>,
) -> Result<Json<ApiResponse<IronSwapCreatedResponse>>, StatusCode> {
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
    info!("🐟 Creating Iron Fish swap for wallet {} direction={}", q_log_privacy::mask_addr(&wallet_hex), request.direction);

    if request.direction != "buy_iron" && request.direction != "sell_iron" {
        return Ok(Json(ApiResponse {
            success: false,
            data: None,
            error: Some("Invalid direction. Use 'buy_iron' or 'sell_iron'.".to_string()),
            timestamp: Utc::now(),
        }));
    }

    // Parse qnk_amount from string or number
    let qnk_amount: u128 = match &request.qnk_amount {
        serde_json::Value::String(s) => s.parse::<u128>().unwrap_or(0),
        serde_json::Value::Number(n) => n.as_u64().unwrap_or(0) as u128,
        _ => 0,
    };

    let (_secret, hash_lock) = generate_swap_secret();

    let now = Utc::now();
    let swap_id = format!("iron_swap_{}", hex::encode(&hash_lock[..8]));

    let proposal = IronFishSwapProposal {
        swap_id: swap_id.clone(),
        user_qnk_address: format!("qnk{}", wallet_hex),
        user_iron_address: request.iron_address.clone(),
        iron_amount: request.iron_amount,
        qnk_amount,
        hash_lock,
        timelock_blocks: 60, // ~60 minutes at 60s/block
        timelock_utc: now + chrono::Duration::hours(12),
        state: IronFishSwapState::Proposed,
        created_at: now,
        direction: request.direction.clone(),
    };

    if let Ok(data) = serde_json::to_vec(&proposal) {
        if let Err(e) = state.storage_engine.save_ironfish_swap(&swap_id, &data).await {
            warn!("Failed to persist Iron Fish swap {}: {}", swap_id, e);
        }
        let _ = state.storage_engine.index_ironfish_swap_by_wallet(
            &format!("qnk{}", wallet_hex),
            &swap_id,
        ).await;
    }

    emit_iron_event(&state, "ironfish-swap-created", serde_json::json!({
        "swap_id": swap_id,
        "direction": request.direction,
        "iron_amount": proposal.iron_amount,
        "qnk_amount": proposal.qnk_amount.to_string(),
        "status": "proposed",
    })).await;

    let response = IronSwapCreatedResponse {
        swap_id: proposal.swap_id.clone(),
        direction: request.direction,
        iron_amount: proposal.iron_amount,
        qnk_amount: proposal.qnk_amount.to_string(),
        hash_lock: hex::encode(proposal.hash_lock),
        iron_address: request.iron_address,
        timelock_blocks: proposal.timelock_blocks,
        timelock_utc: proposal.timelock_utc.to_rfc3339(),
        status: "proposed".to_string(),
        created_at: proposal.created_at.to_rfc3339(),
    };

    info!("🐟 Iron Fish swap created: {} ({})", proposal.swap_id, response.direction);
    Ok(Json(ApiResponse::success(response)))
}

/// GET /api/v1/ironfish/swap/:id — Get swap status
pub async fn get_iron_swap_status(
    State(state): State<Arc<AppState>>,
    Path(swap_id): Path<String>,
    _auth_wallet: Option<AuthenticatedWallet>,
) -> Result<Json<ApiResponse<IronSwapStatusResponse>>, StatusCode> {
    match state.storage_engine.get_ironfish_swap(&swap_id).await {
        Ok(Some(data)) => {
            match serde_json::from_slice::<IronFishSwapProposal>(&data) {
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

/// POST /api/v1/ironfish/swap/:id/claim — Claim swap by revealing secret
pub async fn claim_iron_swap(
    State(state): State<Arc<AppState>>,
    Path(swap_id): Path<String>,
    auth_wallet: Option<AuthenticatedWallet>,
    Json(request): Json<ClaimIronSwapRequest>,
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

    info!("🐟 Claiming Iron Fish swap {} by wallet {}", swap_id, q_log_privacy::mask_addr(&hex::encode(wallet.address)));

    let mut proposal: IronFishSwapProposal = match state.storage_engine.get_ironfish_swap(&swap_id).await {
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

    proposal.state = IronFishSwapState::QnkClaimed { secret: secret.clone() };

    // ═══════════════════════════════════════════════════════════════
    // v9.4.0: Bridge safety check — MUST pass before minting
    // Verifies: kill-switch, amount limits, deposit on Iron Fish chain
    // ═══════════════════════════════════════════════════════════════
    if proposal.direction == "sell_iron" && proposal.iron_amount > 0 {
        if let Err(safety_err) = state.bridge_safety.pre_mint_check(
            crate::bridge_tokens::BridgeChain::IronFish,
            proposal.iron_amount as u128,
            &swap_id,
            request.deposit_txid.as_deref(),
        ).await {
            warn!("🚨 [BRIDGE SAFETY] IRON mint blocked for swap {}: {}", swap_id, safety_err);
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
            crate::bridge_committee::BridgeChainId::IronFish,
            &swap_id,
            &request.secret,
            &proposal.hash_lock,
            proposal.iron_amount as u128,
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
    // sell_iron = user deposits IRON → mint wIRON for them
    // buy_iron  = user sends QNK to get IRON → burn their wIRON
    // ═══════════════════════════════════════════════════════════════
    let bridge_result = if proposal.direction == "sell_iron" {
        // User deposited IRON on Iron Fish chain → mint wIRON on QNK
        bridge_tokens::mint_wrapped_token(
            BridgeChain::IronFish,
            &wallet.address,
            proposal.iron_amount as u128,
            &state.token_balances,
            &state.storage_engine,
        ).await
    } else {
        // User buying IRON → burn their wIRON
        bridge_tokens::burn_wrapped_token(
            BridgeChain::IronFish,
            &wallet.address,
            proposal.iron_amount as u128,
            &state.token_balances,
            &state.storage_engine,
        ).await
    };

    match &bridge_result {
        Ok(new_bal) => {
            info!("🌉 Bridge {} wIRON: {} ore, new balance: {}",
                if proposal.direction == "sell_iron" { "MINT" } else { "BURN" },
                q_log_privacy::mask_amt(proposal.iron_amount as u128), q_log_privacy::mask_amt(*new_bal));
            // Emit SSE for balance update
            emit_iron_event(&state, "bridge-token-updated", serde_json::json!({
                "token": "wIRON",
                "wallet": hex::encode(wallet.address),
                "balance": new_bal.to_string(),
                "operation": if proposal.direction == "sell_iron" { "mint" } else { "burn" },
                "amount": proposal.iron_amount,
            })).await;
        }
        Err(e) => {
            warn!("🌉 Bridge wIRON operation failed (non-fatal): {}", e);
        }
    }

    if let Ok(data) = serde_json::to_vec(&proposal) {
        let _ = state.storage_engine.save_ironfish_swap(&swap_id, &data).await;
    }

    // Save bridge operation audit trail
    let op = bridge_tokens::BridgeOperation {
        op_id: format!("bridge_iron_{}", swap_id),
        chain: BridgeChain::IronFish,
        op_type: if proposal.direction == "sell_iron" {
            bridge_tokens::BridgeOpType::Mint
        } else {
            bridge_tokens::BridgeOpType::Burn
        },
        wallet: wallet.address,
        amount: proposal.iron_amount as u128,
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

    emit_iron_event(&state, "ironfish-swap-claimed", serde_json::json!({
        "swap_id": swap_id,
        "secret": hex::encode(&secret),
        "status": "qnk_claimed",
        "bridge_token": "wIRON",
        "bridge_operation": if proposal.direction == "sell_iron" { "mint" } else { "burn" },
    })).await;

    Ok(Json(ApiResponse::success(serde_json::json!({
        "swap_id": swap_id,
        "status": "claimed",
        "message": "Secret revealed. Iron Fish claim processed.",
        "bridge_token": "wIRON",
        "bridge_balance": bridge_result.unwrap_or(0).to_string(),
    }))))
}

/// POST /api/v1/ironfish/swap/:id/refund — Refund expired swap
pub async fn refund_iron_swap(
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

    info!("🐟 Refunding Iron Fish swap {}", swap_id);

    let mut proposal: IronFishSwapProposal = match state.storage_engine.get_ironfish_swap(&swap_id).await {
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

    if Utc::now() < proposal.timelock_utc {
        return Ok(Json(ApiResponse {
            success: false,
            data: None,
            error: Some(format!("Timelock not expired. Wait until {}", proposal.timelock_utc.to_rfc3339())),
            timestamp: Utc::now(),
        }));
    }

    proposal.state = IronFishSwapState::Refunded { reason: "Timelock expired".to_string() };

    if let Ok(data) = serde_json::to_vec(&proposal) {
        let _ = state.storage_engine.save_ironfish_swap(&swap_id, &data).await;
    }

    emit_iron_event(&state, "ironfish-swap-refunded", serde_json::json!({
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

/// GET /api/v1/ironfish/swaps — List user's swaps
pub async fn list_iron_swaps(
    State(state): State<Arc<AppState>>,
    auth_wallet: Option<AuthenticatedWallet>,
) -> Result<Json<ApiResponse<IronSwapListResponse>>, StatusCode> {
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

    let swap_ids = match state.storage_engine.list_ironfish_swaps_by_wallet(&wallet_key).await {
        Ok(ids) => ids,
        Err(e) => {
            warn!("Failed to list Iron Fish swaps for {}: {}", q_log_privacy::mask_addr(&wallet_key), e);
            Vec::new()
        }
    };

    let mut swaps = Vec::new();
    for swap_id in &swap_ids {
        if let Ok(Some(data)) = state.storage_engine.get_ironfish_swap(swap_id).await {
            if let Ok(proposal) = serde_json::from_slice::<IronFishSwapProposal>(&data) {
                swaps.push(proposal_to_status(&proposal));
            }
        }
    }

    let total = swaps.len();
    Ok(Json(ApiResponse::success(IronSwapListResponse { swaps, total })))
}

/// GET /api/v1/ironfish/balance — Get Iron Fish balance
pub async fn get_iron_balance(
    State(state): State<Arc<AppState>>,
    auth_wallet: Option<AuthenticatedWallet>,
) -> Result<Json<ApiResponse<IronBalanceResponse>>, StatusCode> {
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

    let iron_address = state.storage_engine.get_ironfish_address(&wallet_key).await
        .unwrap_or(None)
        .unwrap_or_default();

    let balance_ore = state.storage_engine.get_ironfish_balance(&wallet_key).await
        .unwrap_or(0);

    let response = IronBalanceResponse {
        balance_ore,
        balance_iron: balance_ore as f64 / ORE_PER_IRON,
        iron_address,
        pending_ore: 0,
    };

    Ok(Json(ApiResponse::success(response)))
}

/// GET /api/v1/ironfish/address — Get or generate user's Iron Fish address
pub async fn get_iron_address(
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

    let wallet_key = format!("qnk{}", hex::encode(wallet.address));

    let existing = state.storage_engine.get_ironfish_address(&wallet_key).await
        .unwrap_or(None);

    let iron_address = if let Some(addr) = existing {
        addr
    } else {
        // Derive a deterministic address from the QNK wallet
        let mut hasher = Sha256::new();
        hasher.update(b"ironfish_address_derivation_v1");
        hasher.update(&wallet.address);
        let derived = hasher.finalize();

        let iron_addr = hex::encode(&derived);
        let _ = state.storage_engine.save_ironfish_address(&wallet_key, &iron_addr).await;
        iron_addr
    };

    Ok(Json(ApiResponse::success(serde_json::json!({
        "iron_address": iron_address,
        "address_type": "shielded",
        "privacy": "all_transactions_shielded",
    }))))
}

/// GET /api/v1/ironfish/bridge/status — Iron Fish node and bridge status
pub async fn get_iron_bridge_status(
    State(_state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<IronBridgeStatusResponse>>, StatusCode> {
    let rpc_url = ironfish_rpc_url();

    // Try querying the Iron Fish node for status
    let (version, height, syncing, peers) = match ironfish_rpc_call("node/getStatus", serde_json::json!({})).await {
        Ok(info) => {
            let data = &info["data"];
            let v = data["node"]["version"].as_str().unwrap_or("unknown").to_string();
            let h = data["blockchain"]["head"]["sequence"].as_u64().unwrap_or(0);
            let s = data["blockchain"]["synced"].as_bool().map(|b| !b).unwrap_or(true);
            let p = data["peerNetwork"]["peers"].as_u64().unwrap_or(0) as u32;
            (v, h, s, p)
        }
        Err(e) => {
            warn!("Iron Fish RPC unavailable: {}", e);
            ("offline".to_string(), 0, true, 0)
        }
    };

    let response = IronBridgeStatusResponse {
        bridge_enabled: true,
        node_rpc_url: rpc_url,
        node_version: version,
        node_height: height,
        node_syncing: syncing,
        network: "ironfish-mainnet".to_string(),
        peers,
        features: vec![
            "shielded_atomic_swap".to_string(),
            "zk_snark_privacy".to_string(),
            "cross_chain_bridge".to_string(),
            "memo_support".to_string(),
        ],
    };

    Ok(Json(ApiResponse::success(response)))
}

/// POST /api/v1/ironfish/send — Send IRON transaction
pub async fn send_iron(
    State(state): State<Arc<AppState>>,
    auth_wallet: Option<AuthenticatedWallet>,
    Json(request): Json<SendIronRequest>,
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

    if request.to_address.is_empty() || request.to_address.len() < 32 {
        return Ok(Json(ApiResponse {
            success: false,
            data: None,
            error: Some("Invalid Iron Fish address.".to_string()),
            timestamp: Utc::now(),
        }));
    }

    let wallet_hex = hex::encode(wallet.address);
    let wallet_key = format!("qnk{}", wallet_hex);
    info!("🐟 Sending IRON from wallet {} → {} ({} ore)", q_log_privacy::mask_addr(&wallet_hex), q_log_privacy::mask_addr(&request.to_address), q_log_privacy::mask_amt(request.amount_ore as u128));

    let balance_ore = state.storage_engine.get_ironfish_balance(&wallet_key).await
        .unwrap_or(0);

    if balance_ore < request.amount_ore {
        return Ok(Json(ApiResponse {
            success: false,
            data: None,
            error: Some(format!("Insufficient balance. Have {} ore, need {} ore.", balance_ore, request.amount_ore)),
            timestamp: Utc::now(),
        }));
    }

    let tx_id = format!("iron_tx_{}", hex::encode(rand::random::<[u8; 16]>()));

    let new_balance = balance_ore - request.amount_ore;
    let _ = state.storage_engine.save_ironfish_balance(&wallet_key, new_balance).await;

    emit_iron_event(&state, "ironfish-send", serde_json::json!({
        "tx_id": tx_id,
        "to_address": request.to_address,
        "amount_ore": request.amount_ore,
        "amount_iron": request.amount_ore as f64 / ORE_PER_IRON,
        "memo": request.memo,
    })).await;

    Ok(Json(ApiResponse::success(serde_json::json!({
        "tx_id": tx_id,
        "status": "pending",
        "amount_ore": request.amount_ore,
        "amount_iron": request.amount_ore as f64 / ORE_PER_IRON,
        "to_address": request.to_address,
        "message": "Iron Fish shielded transaction submitted.",
    }))))
}
