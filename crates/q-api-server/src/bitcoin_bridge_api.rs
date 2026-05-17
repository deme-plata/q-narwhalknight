/// Bitcoin Bridge API — Atomic Swap REST endpoints
///
/// Wraps q-bitcoin-bridge::AtomicSwapManager into the main API server.
/// Endpoints use X-Wallet-Auth authentication (same as other wallet endpoints).
///
/// v7.2.0: Initial implementation

use std::sync::Arc;
use axum::{
    extract::{Path, State},
    Json,
};
use chrono::Utc;
use hyper::StatusCode;
use serde::{Deserialize, Serialize};
use tracing::{info, warn, error};

use q_bitcoin_bridge::atomic_swap::{AtomicSwapManager, AtomicSwapProposal, SwapState};
use q_types::ApiResponse;

use crate::streaming::StreamEvent;
use crate::wallet_auth::AuthenticatedWallet;
use crate::bridge_tokens::{self, BridgeChain};
use crate::AppState;

// ============ Request / Response Types ============

#[derive(Debug, Deserialize)]
pub struct CreateSwapRequest {
    /// "buy_btc" (QNK→BTC) or "sell_btc" (BTC→QNK)
    pub direction: String,
    /// Amount in satoshis (for BTC side)
    pub btc_amount: u64,
    /// Amount in QNK base units (24 decimals) — accepts string or number
    pub qnk_amount: serde_json::Value,
    /// User's Bitcoin public key (hex-encoded, 33 bytes compressed)
    pub user_btc_pubkey: String,
    /// Destination BTC address (for buy_btc direction)
    #[serde(default)]
    pub btc_destination: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct SwapCreatedResponse {
    pub swap_id: String,
    pub direction: String,
    pub btc_amount: u64,
    pub qnk_amount: String,
    pub hash_lock: String,
    pub htlc_address: Option<String>,
    pub timelock_btc: u32,
    pub timelock_qnk: String,
    pub status: String,
    pub created_at: String,
}

#[derive(Debug, Serialize)]
pub struct SwapStatusResponse {
    pub swap_id: String,
    pub user_address: String,
    pub btc_amount: u64,
    pub qnk_amount: String,
    pub status: String,
    pub status_details: serde_json::Value,
    pub hash_lock: String,
    pub timelock_btc: u32,
    pub timelock_qnk: String,
    pub created_at: String,
}

#[derive(Debug, Deserialize)]
pub struct ClaimSwapRequest {
    /// The secret preimage (hex-encoded, 32 bytes)
    pub secret: String,
    /// v9.4.0: Transaction ID of the BTC deposit on Bitcoin chain (REQUIRED for safety)
    #[serde(default)]
    pub deposit_txid: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct SwapListResponse {
    pub swaps: Vec<SwapStatusResponse>,
    pub total: usize,
}

#[derive(Debug, Serialize)]
pub struct BtcBalanceResponse {
    pub balance_sats: u64,
    pub balance_btc: f64,
    pub watched_addresses: Vec<String>,
}

// ============ Helper Functions ============

fn swap_state_to_string(state: &SwapState) -> String {
    match state {
        SwapState::Proposed => "proposed".to_string(),
        SwapState::BtcLocked { .. } => "btc_locked".to_string(),
        SwapState::QnkusdLocked { .. } => "qnk_locked".to_string(),
        SwapState::QnkusdClaimed { .. } => "qnk_claimed".to_string(),
        SwapState::BtcClaimed { .. } => "btc_claimed".to_string(),
        SwapState::Completed => "completed".to_string(),
        SwapState::Refunded => "refunded".to_string(),
        SwapState::Failed { .. } => "failed".to_string(),
    }
}

fn swap_state_details(state: &SwapState) -> serde_json::Value {
    match state {
        SwapState::BtcLocked { btc_txid, btc_vout } => {
            serde_json::json!({ "btc_txid": btc_txid, "btc_vout": btc_vout })
        }
        SwapState::QnkusdLocked { qnk_tx_hash } => {
            serde_json::json!({ "qnk_tx_hash": qnk_tx_hash })
        }
        SwapState::QnkusdClaimed { secret } => {
            serde_json::json!({ "secret": hex::encode(secret) })
        }
        SwapState::BtcClaimed { btc_claim_txid } => {
            serde_json::json!({ "btc_claim_txid": btc_claim_txid })
        }
        SwapState::Failed { reason } => {
            serde_json::json!({ "reason": reason })
        }
        _ => serde_json::json!({}),
    }
}

fn proposal_to_status(proposal: &AtomicSwapProposal) -> SwapStatusResponse {
    SwapStatusResponse {
        swap_id: proposal.swap_id.clone(),
        user_address: proposal.user_address.clone(),
        btc_amount: proposal.btc_amount,
        qnk_amount: proposal.qnkusd_amount.to_string(),
        status: swap_state_to_string(&proposal.state),
        status_details: swap_state_details(&proposal.state),
        hash_lock: hex::encode(proposal.hash_lock),
        timelock_btc: proposal.timelock_btc,
        timelock_qnk: proposal.timelock_qnk.to_rfc3339(),
        created_at: proposal.created_at.to_rfc3339(),
    }
}

async fn emit_swap_event(state: &AppState, event_type: &str, data: serde_json::Value) {
    let event = StreamEvent::Custom {
        event_type: event_type.to_string(),
        data,
        timestamp: Utc::now(),
    };
    let _ = state.event_broadcaster.broadcast(event).await;
}

// ============ Endpoint Handlers ============

/// POST /api/v1/bitcoin/swap — Create a new atomic swap
pub async fn create_atomic_swap(
    State(state): State<Arc<AppState>>,
    auth_wallet: Option<AuthenticatedWallet>,
    Json(request): Json<CreateSwapRequest>,
) -> Result<Json<ApiResponse<SwapCreatedResponse>>, StatusCode> {
    // Require authentication
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
    info!("⚛️ Creating atomic swap for wallet {} direction={}", q_log_privacy::mask_addr(&wallet_hex), request.direction);

    // Validate direction
    if request.direction != "buy_btc" && request.direction != "sell_btc" {
        return Ok(Json(ApiResponse {
            success: false,
            data: None,
            error: Some("Invalid direction. Use 'buy_btc' or 'sell_btc'.".to_string()),
            timestamp: Utc::now(),
        }));
    }

    // Parse user BTC pubkey
    let user_btc_pubkey = match hex::decode(&request.user_btc_pubkey) {
        Ok(bytes) if bytes.len() == 33 => bytes,
        _ => {
            return Ok(Json(ApiResponse {
                success: false,
                data: None,
                error: Some("Invalid user_btc_pubkey. Must be 33-byte compressed pubkey in hex.".to_string()),
                timestamp: Utc::now(),
            }));
        }
    };

    let swap_manager = match &state.atomic_swap_manager {
        Some(mgr) => mgr.clone(),
        None => {
            return Ok(Json(ApiResponse {
                success: false,
                data: None,
                error: Some("Bitcoin bridge not configured on this node.".to_string()),
                timestamp: Utc::now(),
            }));
        }
    };

    // Parse qnk_amount from string or number
    let qnk_amount: u128 = match &request.qnk_amount {
        serde_json::Value::String(s) => s.parse::<u128>().unwrap_or(0),
        serde_json::Value::Number(n) => n.as_u64().unwrap_or(0) as u128,
        _ => 0,
    };

    // Generate secret for the swap (bank side generates, user gets hash)
    let (_secret, hash_lock) = AtomicSwapManager::generate_secret();

    // v9.4.0: Derive bank BTC pubkey from node's Ed25519 signing key
    // Uses SHA256(node_verifying_key) as x-coordinate with 0x02 prefix for compressed format
    let bank_btc_pubkey = {
        use sha2::{Sha256, Digest as Sha2Digest};
        let verifying_key = state.node_signing_key.verifying_key();
        let hash = Sha256::digest(verifying_key.as_bytes());
        let mut pubkey = vec![0x02u8]; // compressed pubkey prefix (even y-coordinate)
        pubkey.extend_from_slice(&hash);
        pubkey
    };

    // Create swap proposal
    let proposal = match swap_manager.create_swap_proposal(
        format!("qnk{}", wallet_hex),
        request.btc_amount,
        qnk_amount,
        hash_lock,
        user_btc_pubkey,
        bank_btc_pubkey,
    ).await {
        Ok(p) => p,
        Err(e) => {
            error!("Failed to create swap proposal: {}", e);
            return Ok(Json(ApiResponse {
                success: false,
                data: None,
                error: Some(format!("Failed to create swap: {}", e)),
                timestamp: Utc::now(),
            }));
        }
    };

    // Get HTLC address
    let htlc_address = swap_manager.get_htlc_address(&proposal)
        .ok()
        .map(|addr| addr.to_string());

    // Persist to storage
    if let Ok(data) = serde_json::to_vec(&proposal) {
        if let Err(e) = state.storage_engine.save_atomic_swap(&proposal.swap_id, &data).await {
            warn!("Failed to persist atomic swap {}: {}", proposal.swap_id, e);
        }
        // Index by wallet
        let _ = state.storage_engine.index_atomic_swap_by_wallet(
            &format!("qnk{}", wallet_hex),
            &proposal.swap_id,
        ).await;
        // v7.2.5: Save direction for bridge mint/burn lookup
        let dir_key = format!("btc_swap_dir:{}", proposal.swap_id);
        let _ = state.storage_engine.get_kv().put(
            q_storage::CF_MANIFEST,
            dir_key.as_bytes(),
            request.direction.as_bytes(),
        ).await;
    }

    // Emit SSE event
    emit_swap_event(&state, "atomic-swap-created", serde_json::json!({
        "swap_id": proposal.swap_id,
        "direction": request.direction,
        "btc_amount": proposal.btc_amount,
        "qnk_amount": proposal.qnkusd_amount.to_string(),
        "status": "proposed",
        "htlc_address": htlc_address,
    })).await;

    let response = SwapCreatedResponse {
        swap_id: proposal.swap_id.clone(),
        direction: request.direction,
        btc_amount: proposal.btc_amount,
        qnk_amount: proposal.qnkusd_amount.to_string(),
        hash_lock: hex::encode(proposal.hash_lock),
        htlc_address,
        timelock_btc: proposal.timelock_btc,
        timelock_qnk: proposal.timelock_qnk.to_rfc3339(),
        status: "proposed".to_string(),
        created_at: proposal.created_at.to_rfc3339(),
    };

    info!("⚛️ Atomic swap created: {} ({})", proposal.swap_id, response.direction);

    Ok(Json(ApiResponse::success(response)))
}

/// GET /api/v1/bitcoin/swap/:id — Get swap status
pub async fn get_swap_status(
    State(state): State<Arc<AppState>>,
    Path(swap_id): Path<String>,
    _auth_wallet: Option<AuthenticatedWallet>,
) -> Result<Json<ApiResponse<SwapStatusResponse>>, StatusCode> {
    let swap_manager = match &state.atomic_swap_manager {
        Some(mgr) => mgr.clone(),
        None => {
            return Ok(Json(ApiResponse {
                success: false,
                data: None,
                error: Some("Bitcoin bridge not configured.".to_string()),
                timestamp: Utc::now(),
            }));
        }
    };

    // Try in-memory first (active swaps)
    match swap_manager.get_swap_status(&swap_id).await {
        Ok(proposal) => {
            return Ok(Json(ApiResponse::success(proposal_to_status(&proposal))));
        }
        Err(_) => {}
    }

    // Fall back to storage
    match state.storage_engine.get_atomic_swap(&swap_id).await {
        Ok(Some(data)) => {
            match serde_json::from_slice::<AtomicSwapProposal>(&data) {
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

/// POST /api/v1/bitcoin/swap/:id/claim — Claim swap by revealing secret
pub async fn claim_swap(
    State(state): State<Arc<AppState>>,
    Path(swap_id): Path<String>,
    auth_wallet: Option<AuthenticatedWallet>,
    Json(request): Json<ClaimSwapRequest>,
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

    let swap_manager = match &state.atomic_swap_manager {
        Some(mgr) => mgr.clone(),
        None => {
            return Ok(Json(ApiResponse {
                success: false,
                data: None,
                error: Some("Bitcoin bridge not configured.".to_string()),
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

    info!("⚛️ Claiming swap {} by wallet {}", swap_id, q_log_privacy::mask_addr(&hex::encode(wallet.address)));

    // Process the claim
    match swap_manager.process_qnkusd_claim(&swap_id, secret.clone()).await {
        Ok(()) => {
            // Get proposal details for bridge operation
            let btc_amount = if let Ok(proposal) = swap_manager.get_swap_status(&swap_id).await {
                if let Ok(data) = serde_json::to_vec(&proposal) {
                    let _ = state.storage_engine.save_atomic_swap(&swap_id, &data).await;
                }
                proposal.btc_amount
            } else {
                0u64
            };

            // v7.2.5: Load direction from storage (saved during create)
            let direction = {
                let dir_key = format!("btc_swap_dir:{}", swap_id);
                match state.storage_engine.get_kv().get(q_storage::CF_MANIFEST, dir_key.as_bytes()).await {
                    Ok(Some(bytes)) => String::from_utf8(bytes).unwrap_or_default(),
                    _ => String::new(),
                }
            };

            // ═══════════════════════════════════════════════════════════════
            // v9.4.0: Bridge safety check — MUST pass before minting
            // Verifies: kill-switch, amount limits, deposit on Bitcoin chain
            // ═══════════════════════════════════════════════════════════════
            if btc_amount > 0 && direction == "sell_btc" {
                if let Err(safety_err) = state.bridge_safety.pre_mint_check(
                    crate::bridge_tokens::BridgeChain::Bitcoin,
                    btc_amount as u128,
                    &swap_id,
                    request.deposit_txid.as_deref(),
                ).await {
                    warn!("🚨 [BRIDGE SAFETY] BTC mint blocked for swap {}: {}", swap_id, safety_err);
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
            // Falls back to single-node if committee too small
            // ═══════════════════════════════════════════════════════════════
            if btc_amount > 0 {
                use sha2::{Sha256, Digest as Sha2Digest};
                let hash_lock: [u8; 32] = Sha256::digest(&secret).into();
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
                    crate::bridge_committee::BridgeChainId::Bitcoin,
                    &swap_id,
                    &request.secret,
                    &hash_lock,
                    btc_amount as u128,
                    &wallet.address,
                    &direction,
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
                    Ok(true) => {} // Approved — proceed with mint/burn
                }
            }

            // ═══════════════════════════════════════════════════════════════
            // v7.2.5: Bridge token mint/burn on swap completion
            // sell_btc = user deposits BTC → mint wBTC for them
            // buy_btc  = user sends QNK to get BTC → burn their wBTC
            // ═══════════════════════════════════════════════════════════════
            let bridge_result = if direction == "sell_btc" && btc_amount > 0 {
                bridge_tokens::mint_wrapped_token(
                    BridgeChain::Bitcoin,
                    &wallet.address,
                    btc_amount as u128,
                    &state.token_balances,
                    &state.storage_engine,
                ).await
            } else if direction == "buy_btc" && btc_amount > 0 {
                bridge_tokens::burn_wrapped_token(
                    BridgeChain::Bitcoin,
                    &wallet.address,
                    btc_amount as u128,
                    &state.token_balances,
                    &state.storage_engine,
                ).await
            } else {
                Ok(0) // No bridge operation if direction/amount unknown
            };

            match &bridge_result {
                Ok(new_bal) if btc_amount > 0 => {
                    info!("🌉 Bridge {} wBTC: {} sat, new balance: {}",
                        if direction == "sell_btc" { "MINT" } else { "BURN" },
                        q_log_privacy::mask_amt(btc_amount as u128), q_log_privacy::mask_amt(*new_bal));
                    emit_swap_event(&state, "bridge-token-updated", serde_json::json!({
                        "token": "wBTC",
                        "wallet": hex::encode(wallet.address),
                        "balance": new_bal.to_string(),
                        "operation": if direction == "sell_btc" { "mint" } else { "burn" },
                        "amount": btc_amount,
                    })).await;
                }
                Err(e) => warn!("🌉 Bridge wBTC operation failed (non-fatal): {}", e),
                _ => {}
            }

            // Save bridge audit trail
            if btc_amount > 0 {
                let op = bridge_tokens::BridgeOperation {
                    op_id: format!("bridge_btc_{}", swap_id),
                    chain: BridgeChain::Bitcoin,
                    op_type: if direction == "sell_btc" {
                        bridge_tokens::BridgeOpType::Mint
                    } else {
                        bridge_tokens::BridgeOpType::Burn
                    },
                    wallet: wallet.address,
                    amount: btc_amount as u128,
                    native_txid: request.deposit_txid.clone(),
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
            }

            // Emit SSE
            emit_swap_event(&state, "atomic-swap-claimed", serde_json::json!({
                "swap_id": swap_id,
                "secret": hex::encode(&secret),
                "status": "qnk_claimed",
                "bridge_token": "wBTC",
                "bridge_operation": if direction == "sell_btc" { "mint" } else { "burn" },
            })).await;

            Ok(Json(ApiResponse::success(serde_json::json!({
                "swap_id": swap_id,
                "status": "claimed",
                "message": "Secret revealed. QNK claim processed.",
                "bridge_token": "wBTC",
                "bridge_balance": bridge_result.unwrap_or(0).to_string(),
            }))))
        }
        Err(e) => {
            error!("Failed to claim swap {}: {}", swap_id, e);
            Ok(Json(ApiResponse {
                success: false,
                data: None,
                error: Some(format!("Claim failed: {}", e)),
                timestamp: Utc::now(),
            }))
        }
    }
}

/// POST /api/v1/bitcoin/swap/:id/refund — Refund expired swap
pub async fn refund_swap(
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

    let swap_manager = match &state.atomic_swap_manager {
        Some(mgr) => mgr.clone(),
        None => {
            return Ok(Json(ApiResponse {
                success: false,
                data: None,
                error: Some("Bitcoin bridge not configured.".to_string()),
                timestamp: Utc::now(),
            }));
        }
    };

    info!("⚛️ Refunding swap {}", swap_id);

    // Try BTC refund first, then QNK refund
    let btc_refund = swap_manager.refund_btc(&swap_id).await;
    let qnk_refund = swap_manager.refund_qnkusd(&swap_id).await;

    match (btc_refund, qnk_refund) {
        (Ok(btc_txid), _) => {
            // Update storage
            if let Ok(proposal) = swap_manager.get_swap_status(&swap_id).await {
                if let Ok(data) = serde_json::to_vec(&proposal) {
                    let _ = state.storage_engine.save_atomic_swap(&swap_id, &data).await;
                }
            }

            emit_swap_event(&state, "atomic-swap-refunded", serde_json::json!({
                "swap_id": swap_id,
                "refund_txid": btc_txid,
                "type": "btc",
            })).await;

            Ok(Json(ApiResponse::success(serde_json::json!({
                "swap_id": swap_id,
                "status": "refunded",
                "btc_refund_txid": btc_txid,
            }))))
        }
        (_, Ok(qnk_hash)) => {
            emit_swap_event(&state, "atomic-swap-refunded", serde_json::json!({
                "swap_id": swap_id,
                "refund_hash": qnk_hash,
                "type": "qnk",
            })).await;

            Ok(Json(ApiResponse::success(serde_json::json!({
                "swap_id": swap_id,
                "status": "refunded",
                "qnk_refund_hash": qnk_hash,
            }))))
        }
        (Err(e1), Err(e2)) => {
            Ok(Json(ApiResponse {
                success: false,
                data: None,
                error: Some(format!("Refund failed — BTC: {}, QNK: {}", e1, e2)),
                timestamp: Utc::now(),
            }))
        }
    }
}

/// GET /api/v1/bitcoin/swaps — List user's swaps
pub async fn list_swaps(
    State(state): State<Arc<AppState>>,
    auth_wallet: Option<AuthenticatedWallet>,
) -> Result<Json<ApiResponse<SwapListResponse>>, StatusCode> {
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

    // Get swap IDs from index
    let swap_ids = match state.storage_engine.list_atomic_swaps_by_wallet(&wallet_key).await {
        Ok(ids) => ids,
        Err(e) => {
            warn!("Failed to list swaps for {}: {}", q_log_privacy::mask_addr(&wallet_key), e);
            Vec::new()
        }
    };

    let mut swaps = Vec::new();
    for swap_id in &swap_ids {
        if let Ok(Some(data)) = state.storage_engine.get_atomic_swap(swap_id).await {
            if let Ok(proposal) = serde_json::from_slice::<AtomicSwapProposal>(&data) {
                swaps.push(proposal_to_status(&proposal));
            }
        }
    }

    let total = swaps.len();
    Ok(Json(ApiResponse::success(SwapListResponse { swaps, total })))
}

/// GET /api/v1/bitcoin/balance — Get BTC balance info
pub async fn get_btc_balance(
    State(_state): State<Arc<AppState>>,
    auth_wallet: Option<AuthenticatedWallet>,
) -> Result<Json<ApiResponse<BtcBalanceResponse>>, StatusCode> {
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

    // For now, return placeholder — actual BTC balance tracking requires watching
    // specific addresses associated with this wallet's HTLC contracts
    let response = BtcBalanceResponse {
        balance_sats: 0,
        balance_btc: 0.0,
        watched_addresses: Vec::new(),
    };

    Ok(Json(ApiResponse::success(response)))
}

/// GET /api/v1/bitcoin/status — Bitcoin bridge status
pub async fn get_bridge_status(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let connected = state.atomic_swap_manager.is_some() || state.deposit_bridge.is_some();

    let btc_rpc_url = std::env::var("BTC_RPC_URL")
        .unwrap_or_else(|_| "http://5.79.79.158:8332".to_string());

    Ok(Json(ApiResponse::success(serde_json::json!({
        "bridge_enabled": connected,
        "btc_rpc_url": btc_rpc_url,
        "btc_network": "bitcoin",
        "swap_protocol": "HTLC",
        "features": ["atomic_swap", "htlc", "p2wsh"],
    }))))
}
