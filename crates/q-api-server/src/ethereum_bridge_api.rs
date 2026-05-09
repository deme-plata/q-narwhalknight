/// Ethereum Bridge API — Atomic Swap REST endpoints
///
/// Provides QNK ↔ ETH atomic swaps via HTLC protocol.
/// Uses Reth full node on Server Delta (5.79.79.158) for Ethereum RPC.
/// Endpoints use X-Wallet-Auth authentication.
///
/// v7.3.0: Initial implementation

use std::sync::Arc;
use std::collections::HashMap;
use axum::{
    extract::{Path, State},
    Json,
};
use chrono::Utc;
use hyper::StatusCode;
use serde::{Deserialize, Serialize};
use tracing::{info, warn, error};
use tokio::sync::RwLock;

use q_types::{ApiResponse, WETH_TOKEN_ADDRESS};

use crate::streaming::StreamEvent;
use crate::wallet_auth::AuthenticatedWallet;
use crate::bridge_tokens::{self, BridgeChain};
use crate::AppState;

// ============ Constants ============

/// Reth RPC endpoint on Server Delta
const RETH_RPC_URL: &str = "http://5.79.79.158:8545";

/// Timelock: 12 hours for QNK side, ~900 blocks (~3 hours) for ETH side
const ETH_TIMELOCK_BLOCKS: u64 = 900;
const QNK_TIMELOCK_SECONDS: u64 = 43200; // 12 hours

// ============ Request / Response Types ============

#[derive(Debug, Deserialize)]
pub struct CreateEthSwapRequest {
    /// "buy_eth" (QNK→ETH) or "sell_eth" (ETH→QNK)
    pub direction: String,
    /// Amount in wei (string to handle u256)
    pub eth_amount: String,
    /// Amount in QNK base units (24 decimals)
    pub qnk_amount: String,
    /// Destination ETH address (for buy_eth direction)
    #[serde(default)]
    pub eth_destination: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct EthSwapCreatedResponse {
    pub swap_id: String,
    pub direction: String,
    pub eth_amount: String,
    pub qnk_amount: String,
    pub hash_lock: String,
    pub htlc_address: Option<String>,
    pub timelock_eth_blocks: u64,
    pub timelock_qnk: String,
    pub status: String,
    pub created_at: String,
}

#[derive(Debug, Serialize)]
pub struct EthSwapStatusResponse {
    pub swap_id: String,
    pub user_address: String,
    pub direction: String,
    pub eth_amount: String,
    pub qnk_amount: String,
    pub status: String,
    pub hash_lock: String,
    pub timelock_eth_blocks: u64,
    pub timelock_qnk: String,
    pub created_at: String,
    pub eth_destination: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ClaimEthSwapRequest {
    /// The secret preimage (hex-encoded, 32 bytes)
    pub secret: String,
    /// v9.4.0: Transaction hash of the ETH deposit on Ethereum chain (REQUIRED for safety)
    #[serde(default)]
    pub deposit_txid: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct EthSwapListResponse {
    pub swaps: Vec<EthSwapStatusResponse>,
    pub total: usize,
}

#[derive(Debug, Serialize)]
pub struct EthBalanceResponse {
    pub balance_wei: String,
    pub balance_eth: f64,
}

#[derive(Debug, Serialize)]
pub struct EthAddressResponse {
    pub eth_address: String,
}

#[derive(Debug, Serialize)]
pub struct EthBridgeStatusResponse {
    pub bridge_enabled: bool,
    pub reth_rpc_url: String,
    pub reth_height: u64,
    pub reth_synced: bool,
    pub network: String,
    pub features: Vec<String>,
    /// Sync progress fields (only present when syncing)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sync_current_block: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sync_target_block: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sync_progress_pct: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sync_stage: Option<String>,
}

// ============ In-Memory Swap Storage ============

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EthSwap {
    pub swap_id: String,
    pub user_address: String,
    pub direction: String,
    pub eth_amount: String,
    pub qnk_amount: String,
    pub hash_lock: [u8; 32],
    pub secret: Option<Vec<u8>>,
    pub status: String,
    pub timelock_eth_blocks: u64,
    pub timelock_qnk: chrono::DateTime<Utc>,
    pub created_at: chrono::DateTime<Utc>,
    pub eth_destination: Option<String>,
}

static ETH_SWAPS_CELL: std::sync::OnceLock<RwLock<HashMap<String, EthSwap>>> = std::sync::OnceLock::new();
static ETH_WALLET_SWAPS_CELL: std::sync::OnceLock<RwLock<HashMap<String, Vec<String>>>> = std::sync::OnceLock::new();

fn eth_swaps() -> &'static RwLock<HashMap<String, EthSwap>> {
    ETH_SWAPS_CELL.get_or_init(|| RwLock::new(HashMap::new()))
}

fn eth_wallet_swaps() -> &'static RwLock<HashMap<String, Vec<String>>> {
    ETH_WALLET_SWAPS_CELL.get_or_init(|| RwLock::new(HashMap::new()))
}

// ============ Helper Functions ============

fn generate_secret() -> ([u8; 32], [u8; 32]) {
    use sha2::{Sha256, Digest};
    use rand::RngCore;
    let mut secret = [0u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut secret);
    let hash_lock = Sha256::digest(&secret);
    (secret, hash_lock.into())
}

fn swap_to_status(swap: &EthSwap) -> EthSwapStatusResponse {
    EthSwapStatusResponse {
        swap_id: swap.swap_id.clone(),
        user_address: swap.user_address.clone(),
        direction: swap.direction.clone(),
        eth_amount: swap.eth_amount.clone(),
        qnk_amount: swap.qnk_amount.clone(),
        status: swap.status.clone(),
        hash_lock: hex::encode(swap.hash_lock),
        timelock_eth_blocks: swap.timelock_eth_blocks,
        timelock_qnk: swap.timelock_qnk.to_rfc3339(),
        created_at: swap.created_at.to_rfc3339(),
        eth_destination: swap.eth_destination.clone(),
    }
}

async fn emit_eth_swap_event(state: &AppState, event_type: &str, data: serde_json::Value) {
    let event = StreamEvent::Custom {
        event_type: event_type.to_string(),
        data,
        timestamp: Utc::now(),
    };
    let _ = state.event_broadcaster.broadcast(event).await;
}

/// Query Reth node for current block height
async fn get_reth_height() -> Option<u64> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .ok()?;

    let resp = client.post(RETH_RPC_URL)
        .json(&serde_json::json!({
            "jsonrpc": "2.0",
            "method": "eth_blockNumber",
            "params": [],
            "id": 1
        }))
        .send()
        .await
        .ok()?;

    let body: serde_json::Value = resp.json().await.ok()?;
    let hex_str = body["result"].as_str()?;
    u64::from_str_radix(hex_str.trim_start_matches("0x"), 16).ok()
}

/// Detailed Reth sync status
struct RethSyncInfo {
    synced: bool,
    height: u64,
    /// Execution stage current block (the bottleneck stage)
    exec_current: Option<u64>,
    /// Headers target (total chain height)
    headers_target: Option<u64>,
    /// Percentage (exec_current / headers_target)
    progress_pct: Option<f64>,
    /// Current stage name
    stage: Option<String>,
}

/// Query Reth sync status with stage-level progress
async fn get_reth_sync_status() -> RethSyncInfo {
    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
    {
        Ok(c) => c,
        Err(_) => return RethSyncInfo { synced: false, height: 0, exec_current: None, headers_target: None, progress_pct: None, stage: None },
    };

    // First try eth_blockNumber (works when fully synced)
    let height = get_reth_height().await.unwrap_or(0);
    if height > 19_000_000 {
        return RethSyncInfo { synced: true, height, exec_current: None, headers_target: None, progress_pct: None, stage: None };
    }

    // Not synced — query eth_syncing for stage progress
    let resp = client.post(RETH_RPC_URL)
        .json(&serde_json::json!({
            "jsonrpc": "2.0",
            "method": "eth_syncing",
            "params": [],
            "id": 2
        }))
        .send()
        .await;

    let (exec_current, headers_target, progress_pct, stage) = match resp {
        Ok(r) => {
            if let Ok(body) = r.json::<serde_json::Value>().await {
                // eth_syncing returns false when synced, or object with stages
                if let Some(stages) = body["result"]["stages"].as_array() {
                    let mut exec_block: u64 = 0;
                    let mut headers_block: u64 = 0;
                    let mut current_stage = String::new();
                    for s in stages {
                        let name = s["name"].as_str().unwrap_or("");
                        let block_hex = s["block"].as_str().unwrap_or("0x0");
                        let block = u64::from_str_radix(block_hex.trim_start_matches("0x"), 16).unwrap_or(0);
                        if name == "Execution" {
                            exec_block = block;
                        }
                        if name == "Headers" {
                            headers_block = block;
                        }
                        // Track the slowest non-zero stage as "current"
                        if block > 0 && (current_stage.is_empty() || block < exec_block) {
                            if name == "Execution" {
                                current_stage = name.to_string();
                            }
                        }
                    }
                    if current_stage.is_empty() {
                        current_stage = "Execution".to_string();
                    }
                    let pct = if headers_block > 0 {
                        (exec_block as f64 / headers_block as f64 * 100.0 * 100.0).round() / 100.0
                    } else {
                        0.0
                    };
                    (Some(exec_block), Some(headers_block), Some(pct), Some(current_stage))
                } else {
                    (None, None, None, None)
                }
            } else {
                (None, None, None, None)
            }
        }
        Err(_) => (None, None, None, None),
    };

    RethSyncInfo {
        synced: false,
        height,
        exec_current,
        headers_target,
        progress_pct,
        stage,
    }
}

// ============ Endpoint Handlers ============

/// GET /api/v1/ethereum/bridge/status — Bridge health status
pub async fn get_eth_bridge_status(
    State(_state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<EthBridgeStatusResponse>>, StatusCode> {
    let info = get_reth_sync_status().await;

    let response = EthBridgeStatusResponse {
        bridge_enabled: info.synced,
        reth_rpc_url: RETH_RPC_URL.to_string(),
        reth_height: info.height,
        reth_synced: info.synced,
        network: "mainnet".to_string(),
        features: vec![
            "htlc-atomic-swap".to_string(),
            "erc20-bridge".to_string(),
            "wrapped-eth".to_string(),
        ],
        sync_current_block: info.exec_current,
        sync_target_block: info.headers_target,
        sync_progress_pct: info.progress_pct,
        sync_stage: info.stage,
    };

    Ok(Json(ApiResponse::success(response)))
}

/// POST /api/v1/ethereum/swap — Create a new ETH atomic swap
pub async fn create_eth_swap(
    State(state): State<Arc<AppState>>,
    auth_wallet: Option<AuthenticatedWallet>,
    Json(request): Json<CreateEthSwapRequest>,
) -> Result<Json<ApiResponse<EthSwapCreatedResponse>>, StatusCode> {
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
    info!("⟠ Creating ETH atomic swap for wallet {} direction={}", q_log_privacy::mask_addr(&wallet_hex), request.direction);

    // Validate direction
    if request.direction != "buy_eth" && request.direction != "sell_eth" {
        return Ok(Json(ApiResponse {
            success: false,
            data: None,
            error: Some("Invalid direction. Use 'buy_eth' or 'sell_eth'.".to_string()),
            timestamp: Utc::now(),
        }));
    }

    // Validate ETH amount (must be parseable as u128 at minimum)
    let eth_amount_check: u128 = match request.eth_amount.parse() {
        Ok(v) if v > 0 => v,
        _ => {
            return Ok(Json(ApiResponse {
                success: false,
                data: None,
                error: Some("Invalid eth_amount. Must be a positive integer (wei).".to_string()),
                timestamp: Utc::now(),
            }));
        }
    };

    // Validate ETH destination for buy_eth
    if request.direction == "buy_eth" {
        match &request.eth_destination {
            Some(addr) if addr.len() == 42 && addr.starts_with("0x") => {
                // Validate hex
                if hex::decode(&addr[2..]).is_err() {
                    return Ok(Json(ApiResponse {
                        success: false,
                        data: None,
                        error: Some("Invalid ETH address. Must be 0x + 40 hex chars.".to_string()),
                        timestamp: Utc::now(),
                    }));
                }
            }
            Some(_) => {
                return Ok(Json(ApiResponse {
                    success: false,
                    data: None,
                    error: Some("Invalid ETH address format. Must be 0x + 40 hex chars.".to_string()),
                    timestamp: Utc::now(),
                }));
            }
            None => {
                return Ok(Json(ApiResponse {
                    success: false,
                    data: None,
                    error: Some("eth_destination required for buy_eth direction.".to_string()),
                    timestamp: Utc::now(),
                }));
            }
        }
    }

    // Generate HTLC secret and hash
    let (secret, hash_lock) = generate_secret();

    let now = Utc::now();
    let swap_id = format!("eth_swap_{}", hex::encode(&hash_lock[..8]));
    let timelock_qnk = now + chrono::Duration::seconds(QNK_TIMELOCK_SECONDS as i64);

    let swap = EthSwap {
        swap_id: swap_id.clone(),
        user_address: format!("qnk{}", wallet_hex),
        direction: request.direction.clone(),
        eth_amount: request.eth_amount.clone(),
        qnk_amount: request.qnk_amount.clone(),
        hash_lock,
        secret: Some(secret.to_vec()),
        status: "proposed".to_string(),
        timelock_eth_blocks: ETH_TIMELOCK_BLOCKS,
        timelock_qnk,
        created_at: now,
        eth_destination: request.eth_destination.clone(),
    };

    // Store swap
    {
        let mut swaps = eth_swaps().write().await;
        swaps.insert(swap_id.clone(), swap.clone());
    }
    {
        let mut wallet_swaps = eth_wallet_swaps().write().await;
        wallet_swaps.entry(format!("qnk{}", wallet_hex))
            .or_insert_with(Vec::new)
            .push(swap_id.clone());
    }

    // Persist to storage
    if let Ok(data) = serde_json::to_vec(&swap) {
        if let Err(e) = state.storage_engine.save_atomic_swap(&swap_id, &data).await {
            warn!("Failed to persist ETH swap {}: {}", swap_id, e);
        }
        let _ = state.storage_engine.index_atomic_swap_by_wallet(
            &format!("qnk{}", wallet_hex),
            &swap_id,
        ).await;
        // Save direction for bridge mint/burn
        let dir_key = format!("eth_swap_dir:{}", swap_id);
        let _ = state.storage_engine.get_kv().put(
            q_storage::CF_MANIFEST,
            dir_key.as_bytes(),
            request.direction.as_bytes(),
        ).await;
    }

    // Emit SSE event
    emit_eth_swap_event(&state, "eth-swap-created", serde_json::json!({
        "swap_id": swap_id,
        "direction": request.direction,
        "eth_amount": request.eth_amount,
        "qnk_amount": request.qnk_amount,
        "status": "proposed",
    })).await;

    let response = EthSwapCreatedResponse {
        swap_id: swap_id.clone(),
        direction: request.direction,
        eth_amount: request.eth_amount,
        qnk_amount: request.qnk_amount,
        hash_lock: hex::encode(hash_lock),
        htlc_address: None, // Will be set when HTLC contract is deployed
        timelock_eth_blocks: ETH_TIMELOCK_BLOCKS,
        timelock_qnk: timelock_qnk.to_rfc3339(),
        status: "proposed".to_string(),
        created_at: now.to_rfc3339(),
    };

    info!("⟠ ETH swap created: {} ({})", swap_id, response.direction);

    Ok(Json(ApiResponse::success(response)))
}

/// GET /api/v1/ethereum/swap/:id — Get swap status
pub async fn get_eth_swap_status(
    State(_state): State<Arc<AppState>>,
    Path(swap_id): Path<String>,
    _auth_wallet: Option<AuthenticatedWallet>,
) -> Result<Json<ApiResponse<EthSwapStatusResponse>>, StatusCode> {
    let swaps = eth_swaps().read().await;
    match swaps.get(&swap_id) {
        Some(swap) => Ok(Json(ApiResponse::success(swap_to_status(swap)))),
        None => Ok(Json(ApiResponse {
            success: false,
            data: None,
            error: Some(format!("Swap {} not found", swap_id)),
            timestamp: Utc::now(),
        })),
    }
}

/// POST /api/v1/ethereum/swap/:id/claim — Claim swap with secret
pub async fn claim_eth_swap(
    State(state): State<Arc<AppState>>,
    Path(swap_id): Path<String>,
    auth_wallet: Option<AuthenticatedWallet>,
    Json(request): Json<ClaimEthSwapRequest>,
) -> Result<Json<ApiResponse<EthSwapStatusResponse>>, StatusCode> {
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

    // Validate secret format
    let secret_bytes = match hex::decode(&request.secret) {
        Ok(bytes) if bytes.len() == 32 => bytes,
        _ => {
            return Ok(Json(ApiResponse {
                success: false,
                data: None,
                error: Some("Invalid secret. Must be 32 bytes hex-encoded.".to_string()),
                timestamp: Utc::now(),
            }));
        }
    };

    // Verify secret against hash_lock
    use sha2::{Sha256, Digest};
    let computed_hash: [u8; 32] = Sha256::digest(&secret_bytes).into();

    let mut swaps = eth_swaps().write().await;
    let swap = match swaps.get_mut(&swap_id) {
        Some(s) => s,
        None => {
            return Ok(Json(ApiResponse {
                success: false,
                data: None,
                error: Some(format!("Swap {} not found", swap_id)),
                timestamp: Utc::now(),
            }));
        }
    };

    if computed_hash != swap.hash_lock {
        return Ok(Json(ApiResponse {
            success: false,
            data: None,
            error: Some("Secret does not match hash lock.".to_string()),
            timestamp: Utc::now(),
        }));
    }

    // Check swap is in a claimable state
    if swap.status != "proposed" && swap.status != "eth_locked" && swap.status != "qnk_locked" {
        return Ok(Json(ApiResponse {
            success: false,
            data: None,
            error: Some(format!("Swap is in state '{}', cannot claim.", swap.status)),
            timestamp: Utc::now(),
        }));
    }

    // Check timelock
    if Utc::now() > swap.timelock_qnk {
        return Ok(Json(ApiResponse {
            success: false,
            data: None,
            error: Some("Swap timelock has expired. Use refund instead.".to_string()),
            timestamp: Utc::now(),
        }));
    }

    swap.status = "completed".to_string();
    swap.secret = Some(secret_bytes);

    // Mint/burn wrapped token based on direction
    let direction = swap.direction.clone();
    let eth_amount_str = swap.eth_amount.clone();
    let user_address = swap.user_address.clone();
    let result = swap_to_status(swap);

    drop(swaps);

    // Parse ETH amount (wei) to u128 for bridge token operations
    let eth_amount_u128: u128 = eth_amount_str.parse().unwrap_or(0);

    // ═══════════════════════════════════════════════════════════════
    // v9.4.0: Bridge safety check — MUST pass before minting
    // Verifies: kill-switch, amount limits, deposit on Ethereum chain
    // ═══════════════════════════════════════════════════════════════
    if eth_amount_u128 > 0 && direction == "sell_eth" {
        if let Err(safety_err) = state.bridge_safety.pre_mint_check(
            crate::bridge_tokens::BridgeChain::Ethereum,
            eth_amount_u128,
            &swap_id,
            request.deposit_txid.as_deref(),
        ).await {
            warn!("🚨 [BRIDGE SAFETY] ETH mint blocked for swap {}: {}", swap_id, safety_err);
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
    if eth_amount_u128 > 0 {
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
            crate::bridge_committee::BridgeChainId::Ethereum,
            &swap_id,
            &request.secret,
            &computed_hash,
            eth_amount_u128,
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
            Ok(true) => {} // Approved
        }
    }

    // Bridge token operation
    if direction == "sell_eth" {
        // User sold ETH → mint wETH on QNK side
        info!("⟠ Minting wETH for {} (amount: {} wei)", q_log_privacy::mask_addr(&user_address), q_log_privacy::mask_amt(eth_amount_u128));
        if let Err(e) = bridge_tokens::mint_wrapped_token(
            BridgeChain::Ethereum,
            &wallet.address,
            eth_amount_u128,
            &state.token_balances,
            &state.storage_engine,
        ).await {
            warn!("Failed to mint wETH: {}", e);
        }
    } else {
        // User bought ETH → burn wETH from QNK side
        info!("⟠ Burning wETH from {} (amount: {} wei)", q_log_privacy::mask_addr(&user_address), q_log_privacy::mask_amt(eth_amount_u128));
        if let Err(e) = bridge_tokens::burn_wrapped_token(
            BridgeChain::Ethereum,
            &wallet.address,
            eth_amount_u128,
            &state.token_balances,
            &state.storage_engine,
        ).await {
            warn!("Failed to burn wETH: {}", e);
        }
    }

    emit_eth_swap_event(&state, "eth-swap-claimed", serde_json::json!({
        "swap_id": swap_id,
        "status": "completed",
        "direction": direction,
    })).await;

    Ok(Json(ApiResponse::success(result)))
}

/// POST /api/v1/ethereum/swap/:id/refund — Refund expired swap
pub async fn refund_eth_swap(
    State(state): State<Arc<AppState>>,
    Path(swap_id): Path<String>,
    auth_wallet: Option<AuthenticatedWallet>,
) -> Result<Json<ApiResponse<EthSwapStatusResponse>>, StatusCode> {
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

    let mut swaps = eth_swaps().write().await;
    let swap = match swaps.get_mut(&swap_id) {
        Some(s) => s,
        None => {
            return Ok(Json(ApiResponse {
                success: false,
                data: None,
                error: Some(format!("Swap {} not found", swap_id)),
                timestamp: Utc::now(),
            }));
        }
    };

    // Verify caller owns the swap
    if swap.user_address != format!("qnk{}", wallet_hex) {
        return Ok(Json(ApiResponse {
            success: false,
            data: None,
            error: Some("Not authorized to refund this swap.".to_string()),
            timestamp: Utc::now(),
        }));
    }

    // Check swap is in a refundable state
    if swap.status == "completed" || swap.status == "refunded" {
        return Ok(Json(ApiResponse {
            success: false,
            data: None,
            error: Some(format!("Swap is already {}.", swap.status)),
            timestamp: Utc::now(),
        }));
    }

    // Check timelock has expired
    if Utc::now() < swap.timelock_qnk {
        return Ok(Json(ApiResponse {
            success: false,
            data: None,
            error: Some("Timelock has not expired yet. Cannot refund.".to_string()),
            timestamp: Utc::now(),
        }));
    }

    swap.status = "refunded".to_string();
    let result = swap_to_status(swap);

    drop(swaps);

    emit_eth_swap_event(&state, "eth-swap-refunded", serde_json::json!({
        "swap_id": swap_id,
        "status": "refunded",
    })).await;

    info!("⟠ ETH swap {} refunded for wallet {}", swap_id, q_log_privacy::mask_addr(&wallet_hex));

    Ok(Json(ApiResponse::success(result)))
}

/// GET /api/v1/ethereum/swaps — List all swaps for authenticated wallet
pub async fn list_eth_swaps(
    State(_state): State<Arc<AppState>>,
    auth_wallet: Option<AuthenticatedWallet>,
) -> Result<Json<ApiResponse<EthSwapListResponse>>, StatusCode> {
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

    let wallet_swaps = eth_wallet_swaps().read().await;
    let swap_ids = wallet_swaps.get(&wallet_key).cloned().unwrap_or_default();
    drop(wallet_swaps);

    let swaps = eth_swaps().read().await;
    let mut results: Vec<EthSwapStatusResponse> = Vec::new();
    for id in &swap_ids {
        if let Some(swap) = swaps.get(id) {
            results.push(swap_to_status(swap));
        }
    }

    let total = results.len();
    Ok(Json(ApiResponse::success(EthSwapListResponse { swaps: results, total })))
}

/// GET /api/v1/ethereum/bridge/balance — Get wETH balance
pub async fn get_eth_balance(
    State(state): State<Arc<AppState>>,
    auth_wallet: Option<AuthenticatedWallet>,
) -> Result<Json<ApiResponse<EthBalanceResponse>>, StatusCode> {
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

    // Check wETH token balance (key format: (wallet_address, token_address))
    let balance_wei = {
        let token_balances = state.token_balances.read().await;
        let key = (wallet.address, WETH_TOKEN_ADDRESS);
        token_balances.get(&key).copied().unwrap_or(0u128)
    };

    let balance_eth = balance_wei as f64 / 1e18;

    Ok(Json(ApiResponse::success(EthBalanceResponse {
        balance_wei: balance_wei.to_string(),
        balance_eth,
    })))
}

/// GET /api/v1/ethereum/bridge/address — Get derived ETH address
pub async fn get_eth_address(
    State(_state): State<Arc<AppState>>,
    auth_wallet: Option<AuthenticatedWallet>,
) -> Result<Json<ApiResponse<EthAddressResponse>>, StatusCode> {
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

    // Derive a deterministic ETH-style address from the QNK wallet address
    let eth_address = format!("0x{}", hex::encode(&wallet.address[..20]));

    Ok(Json(ApiResponse::success(EthAddressResponse { eth_address })))
}

// ============ Send wETH (Transfer) ============

#[derive(Debug, Deserialize)]
pub struct SendEthRequest {
    pub to_address: String,
    pub amount_wei: String,
}

#[derive(Debug, Serialize)]
pub struct SendEthResponse {
    pub tx_id: String,
    pub from: String,
    pub to: String,
    pub amount_wei: String,
}

/// POST /api/v1/ethereum/bridge/send — Transfer wETH to another QNK wallet
pub async fn send_eth(
    State(state): State<Arc<AppState>>,
    auth_wallet: Option<AuthenticatedWallet>,
    Json(request): Json<SendEthRequest>,
) -> Result<Json<ApiResponse<SendEthResponse>>, StatusCode> {
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

    let amount: u128 = match request.amount_wei.parse() {
        Ok(v) if v > 0 => v,
        _ => {
            return Ok(Json(ApiResponse {
                success: false,
                data: None,
                error: Some("Invalid amount_wei. Must be a positive integer.".to_string()),
                timestamp: Utc::now(),
            }));
        }
    };

    // Parse destination address (must be hex, 32 bytes = 64 hex chars, optionally with qnk prefix)
    let to_hex = request.to_address.strip_prefix("qnk").unwrap_or(&request.to_address);
    let to_address: [u8; 32] = match hex::decode(to_hex) {
        Ok(bytes) if bytes.len() == 32 => {
            let mut arr = [0u8; 32];
            arr.copy_from_slice(&bytes);
            arr
        }
        _ => {
            return Ok(Json(ApiResponse {
                success: false,
                data: None,
                error: Some("Invalid to_address. Must be 64 hex chars (optionally with qnk prefix).".to_string()),
                timestamp: Utc::now(),
            }));
        }
    };

    // Debit sender
    {
        let mut token_bals = state.token_balances.write().await;
        let sender_key = (wallet.address, WETH_TOKEN_ADDRESS);
        let sender_bal = token_bals.get(&sender_key).copied().unwrap_or(0);
        if sender_bal < amount {
            return Ok(Json(ApiResponse {
                success: false,
                data: None,
                error: Some(format!("Insufficient wETH balance. Have {} wei, need {}.", sender_bal, amount)),
                timestamp: Utc::now(),
            }));
        }
        token_bals.insert(sender_key, sender_bal - amount);

        // Credit recipient
        let recv_key = (to_address, WETH_TOKEN_ADDRESS);
        let recv_bal = token_bals.get(&recv_key).copied().unwrap_or(0);
        token_bals.insert(recv_key, recv_bal + amount);
    }

    // Persist both balances
    let _ = state.storage_engine.save_token_balance(&wallet.address, &WETH_TOKEN_ADDRESS, {
        let bals = state.token_balances.read().await;
        bals.get(&(wallet.address, WETH_TOKEN_ADDRESS)).copied().unwrap_or(0)
    }).await;
    let _ = state.storage_engine.save_token_balance(&to_address, &WETH_TOKEN_ADDRESS, {
        let bals = state.token_balances.read().await;
        bals.get(&(to_address, WETH_TOKEN_ADDRESS)).copied().unwrap_or(0)
    }).await;

    let tx_id = format!("weth_send_{}", hex::encode(&wallet.address[..8]));
    let from_hex = hex::encode(wallet.address);

    info!("⟠ wETH transfer: {} wei from {} to {}", q_log_privacy::mask_amt(amount), q_log_privacy::mask_addr(&from_hex), q_log_privacy::mask_addr(to_hex));

    Ok(Json(ApiResponse::success(SendEthResponse {
        tx_id,
        from: format!("qnk{}", from_hex),
        to: request.to_address,
        amount_wei: amount.to_string(),
    })))
}

// ============================================================================
// WETH ↔ QUG MetaMask Bridge Endpoints (v1.0.3)
// ============================================================================
//
// These endpoints support the MetaMask WETH deposit flow:
//   1. User calls /deposit-address → gets bridge address + WETH contract info
//   2. User signs WETH ERC-20 transfer to bridge address via MetaMask
//   3. User calls /deposit with tx_hash → registers deposit for monitoring
//   4. Backend monitors confirmations, triggers committee attestation
//   5. After 7/11 attestation, credits QUG to user's wallet
// ============================================================================

use crate::bridge_safety::{
    self, WethBridgeDeposit, WethDepositStatus, WethDepositVerification,
    BRIDGE_DEPOSIT_ADDRESS, WETH_CONTRACT, WETH_MIN_DEPOSIT_WEI, WETH_MAX_DEPOSIT_WEI,
};
use std::sync::OnceLock;

/// In-memory registry of WETH bridge deposits (persisted to RocksDB)
fn weth_deposits() -> &'static RwLock<HashMap<String, WethBridgeDeposit>> {
    static INSTANCE: OnceLock<RwLock<HashMap<String, WethBridgeDeposit>>> = OnceLock::new();
    INSTANCE.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Default WETH/QUG exchange rate: 1 WETH = 65 QUG
/// TODO: Replace with oracle-based pricing
const DEFAULT_WETH_QUG_RATE: f64 = 65.0;

// ---- Request/Response types for bridge deposits ----

#[derive(Debug, Serialize)]
pub struct BridgeDepositAddressResponse {
    pub bridge_deposit_address: String,
    pub weth_contract_address: String,
    pub chain_id: u64,
    pub min_deposit_wei: String,
    pub max_deposit_wei: String,
    pub required_confirmations: u32,
    pub required_attestations: u32,
}

#[derive(Debug, Deserialize)]
pub struct RegisterWethDepositRequest {
    /// Ethereum tx_hash from MetaMask
    pub tx_hash: String,
    /// Sender's Ethereum address (0x...)
    pub sender_address: String,
    /// WETH amount in wei (string)
    pub amount_wei: String,
}

#[derive(Debug, Serialize)]
pub struct RegisterWethDepositResponse {
    pub deposit_id: String,
    pub status: String,
    pub qug_estimate: String,
    pub confirmations: u32,
    pub required_confirmations: u32,
}

#[derive(Debug, Serialize)]
pub struct DepositStatusResponse {
    pub deposit_id: String,
    pub eth_tx_hash: String,
    pub sender_eth_address: String,
    pub amount_wei: String,
    pub qug_amount: String,
    pub confirmations: u32,
    pub required_confirmations: u32,
    pub attestations: u32,
    pub required_attestations: u32,
    pub status: String,
    pub created_at: String,
}

#[derive(Debug, Serialize)]
pub struct BridgeRateResponse {
    pub weth_to_qug_rate: f64,
    pub qug_to_weth_rate: f64,
    pub min_deposit_weth: f64,
    pub max_deposit_weth: f64,
}

// ---- Endpoint Handlers ----

/// GET /api/v1/ethereum/bridge/deposit-address
/// Returns the bridge deposit address and WETH contract info for MetaMask integration.
/// No authentication required.
pub async fn get_bridge_deposit_address(
    State(_state): State<Arc<AppState>>,
) -> Json<ApiResponse<BridgeDepositAddressResponse>> {
    Json(ApiResponse::success(BridgeDepositAddressResponse {
        bridge_deposit_address: BRIDGE_DEPOSIT_ADDRESS.to_string(),
        weth_contract_address: WETH_CONTRACT.to_string(),
        chain_id: 1, // Ethereum mainnet
        min_deposit_wei: WETH_MIN_DEPOSIT_WEI.to_string(),
        max_deposit_wei: WETH_MAX_DEPOSIT_WEI.to_string(),
        required_confirmations: bridge_safety::ETH_MIN_CONFIRMATIONS,
        required_attestations: 7,
    }))
}

/// POST /api/v1/ethereum/bridge/deposit
/// Register a WETH ERC-20 deposit from MetaMask. Requires authentication.
pub async fn register_weth_deposit(
    wallet: AuthenticatedWallet,
    State(state): State<Arc<AppState>>,
    Json(request): Json<RegisterWethDepositRequest>,
) -> Result<Json<ApiResponse<RegisterWethDepositResponse>>, StatusCode> {
    let wallet_hex = hex::encode(wallet.address);

    // Validate tx_hash format
    if !request.tx_hash.starts_with("0x") || request.tx_hash.len() != 66 {
        return Ok(Json(ApiResponse {
            success: false,
            data: None,
            error: Some("Invalid tx_hash format. Expected 0x-prefixed 32-byte hex.".to_string()),
            timestamp: Utc::now(),
        }));
    }

    // Parse amount
    let amount_wei: u128 = match request.amount_wei.parse() {
        Ok(a) => a,
        Err(_) => {
            return Ok(Json(ApiResponse {
                success: false,
                data: None,
                error: Some("Invalid amount_wei — must be a valid u128 string.".to_string()),
                timestamp: Utc::now(),
            }));
        }
    };

    // Amount bounds check
    if amount_wei < WETH_MIN_DEPOSIT_WEI {
        return Ok(Json(ApiResponse {
            success: false,
            data: None,
            error: Some(format!("Amount below minimum deposit (0.001 WETH = {} wei)", WETH_MIN_DEPOSIT_WEI)),
            timestamp: Utc::now(),
        }));
    }
    if amount_wei > WETH_MAX_DEPOSIT_WEI {
        return Ok(Json(ApiResponse {
            success: false,
            data: None,
            error: Some(format!("Amount exceeds maximum deposit (1.0 WETH = {} wei)", WETH_MAX_DEPOSIT_WEI)),
            timestamp: Utc::now(),
        }));
    }

    // Replay protection: check if tx_hash already claimed
    if state.bridge_safety.is_txid_already_claimed(&state.storage_engine, &request.tx_hash).await {
        return Ok(Json(ApiResponse {
            success: false,
            data: None,
            error: Some("This Ethereum transaction has already been used for a bridge deposit.".to_string()),
            timestamp: Utc::now(),
        }));
    }

    // Check if deposit already registered (in-memory)
    {
        let deposits = weth_deposits().read().await;
        for dep in deposits.values() {
            if dep.eth_tx_hash.to_lowercase() == request.tx_hash.to_lowercase() {
                return Ok(Json(ApiResponse {
                    success: false,
                    data: None,
                    error: Some(format!("Deposit already registered with ID: {}", dep.deposit_id)),
                    timestamp: Utc::now(),
                }));
            }
        }
    }

    // Calculate QUG amount: WETH (18 dec) → QUG (24 dec)
    // qug_amount = weth_wei * rate * 10^6 (to go from 18 to 24 decimals)
    let qug_amount = amount_wei
        .checked_mul(DEFAULT_WETH_QUG_RATE as u128)
        .and_then(|v| v.checked_mul(1_000_000)) // 10^6 to convert 18→24 decimals
        .unwrap_or(0);

    if qug_amount == 0 {
        return Ok(Json(ApiResponse {
            success: false,
            data: None,
            error: Some("Amount too small — QUG conversion resulted in zero.".to_string()),
            timestamp: Utc::now(),
        }));
    }

    let deposit_id = format!("weth_dep_{}", &request.tx_hash[2..10]);
    let now = Utc::now();

    let deposit = WethBridgeDeposit {
        deposit_id: deposit_id.clone(),
        eth_tx_hash: request.tx_hash.clone(),
        sender_eth_address: request.sender_address.clone(),
        recipient_qnk_wallet: wallet_hex.clone(),
        amount_wei,
        qug_amount,
        confirmations: 0,
        attestations: 0,
        required_attestations: 7,
        status: WethDepositStatus::Pending,
        created_at: now,
        updated_at: now,
    };

    // Store in memory
    {
        let mut deposits = weth_deposits().write().await;
        deposits.insert(deposit_id.clone(), deposit.clone());
    }

    // Persist to RocksDB via atomic swap storage
    if let Ok(bytes) = serde_json::to_vec(&deposit) {
        let key = format!("weth_deposit_{}", deposit_id);
        let _ = state.storage_engine.save_atomic_swap(&key, &bytes).await;
    }

    info!(
        "⟠ [WETH BRIDGE] New deposit registered: {} | tx={} | {} wei → {} QUG (24-dec) | wallet={}",
        deposit_id, q_log_privacy::mask_hash(&request.tx_hash), q_log_privacy::mask_amt(amount_wei), q_log_privacy::mask_amt(qug_amount), q_log_privacy::mask_addr(&wallet_hex)
    );

    Ok(Json(ApiResponse::success(RegisterWethDepositResponse {
        deposit_id,
        status: "pending".to_string(),
        qug_estimate: qug_amount.to_string(),
        confirmations: 0,
        required_confirmations: bridge_safety::ETH_MIN_CONFIRMATIONS,
    })))
}

/// GET /api/v1/ethereum/bridge/deposit/:id/status
/// Poll the status of a bridge deposit. Requires authentication.
pub async fn get_deposit_status(
    _wallet: AuthenticatedWallet,
    State(_state): State<Arc<AppState>>,
    Path(deposit_id): Path<String>,
) -> Json<ApiResponse<DepositStatusResponse>> {
    let deposits = weth_deposits().read().await;

    match deposits.get(&deposit_id) {
        Some(dep) => {
            let status_str = match &dep.status {
                WethDepositStatus::Pending => "pending",
                WethDepositStatus::Confirming => "confirming",
                WethDepositStatus::Attesting => "attesting",
                WethDepositStatus::Completed => "completed",
                WethDepositStatus::Failed(_) => "failed",
            };

            Json(ApiResponse::success(DepositStatusResponse {
                deposit_id: dep.deposit_id.clone(),
                eth_tx_hash: dep.eth_tx_hash.clone(),
                sender_eth_address: dep.sender_eth_address.clone(),
                amount_wei: dep.amount_wei.to_string(),
                qug_amount: dep.qug_amount.to_string(),
                confirmations: dep.confirmations,
                required_confirmations: bridge_safety::ETH_MIN_CONFIRMATIONS,
                attestations: dep.attestations,
                required_attestations: dep.required_attestations,
                status: status_str.to_string(),
                created_at: dep.created_at.to_rfc3339(),
            }))
        }
        None => Json(ApiResponse {
            success: false,
            data: None,
            error: Some(format!("Deposit {} not found", deposit_id)),
            timestamp: Utc::now(),
        }),
    }
}

/// GET /api/v1/ethereum/bridge/rate
/// Current WETH/QUG exchange rate. No authentication required.
pub async fn get_bridge_rate(
    State(_state): State<Arc<AppState>>,
) -> Json<ApiResponse<BridgeRateResponse>> {
    Json(ApiResponse::success(BridgeRateResponse {
        weth_to_qug_rate: DEFAULT_WETH_QUG_RATE,
        qug_to_weth_rate: 1.0 / DEFAULT_WETH_QUG_RATE,
        min_deposit_weth: WETH_MIN_DEPOSIT_WEI as f64 / 1e18,
        max_deposit_weth: WETH_MAX_DEPOSIT_WEI as f64 / 1e18,
    }))
}

/// Credit QUG to a wallet after successful WETH bridge deposit.
/// Called by the deposit monitor after committee attestation.
pub async fn credit_qug_for_weth_deposit(
    state: &Arc<AppState>,
    deposit_id: &str,
    recipient_wallet_hex: &str,
    qug_amount: u128,
) -> Result<(), String> {
    // Parse wallet address
    let wallet_bytes = hex::decode(recipient_wallet_hex)
        .map_err(|e| format!("Invalid wallet hex: {}", e))?;
    if wallet_bytes.len() != 32 {
        return Err("Wallet address must be 32 bytes".to_string());
    }
    let mut wallet_addr = [0u8; 32];
    wallet_addr.copy_from_slice(&wallet_bytes);

    // Credit balance using checked arithmetic
    let display_divisor: f64 = 1_000_000_000_000_000_000_000_000.0; // 10^24 for QUG
    let old_balance_display: f64;
    let new_balance_display: f64;
    {
        let mut balances = state.wallet_balances.write().await;
        let current = balances.get(&wallet_addr).copied().unwrap_or(0);
        let new_balance = current.checked_add(qug_amount)
            .ok_or("Balance overflow — deposit would exceed u128::MAX")?;
        balances.insert(wallet_addr, new_balance);

        old_balance_display = current as f64 / display_divisor;
        new_balance_display = new_balance as f64 / display_divisor;
    }

    // Emit SSE event for real-time UI update
    let _ = state.event_broadcaster.broadcast(StreamEvent::BalanceUpdated {
        wallet_address: recipient_wallet_hex.to_string(),
        old_balance: old_balance_display,
        new_balance: new_balance_display,
        change_reason: "weth_bridge_deposit".to_string(),
        timestamp: Utc::now(),
        block_hash: None,
        block_height: None,
        confirmation_status: "confirmed".to_string(),
        from_address: None,
        tx_hash: None,
        memo: None,
    }).await;

    info!(
        "💰 [WETH BRIDGE] Credited {} QUG (24-dec) to wallet {} for deposit {}",
        q_log_privacy::mask_amt(qug_amount), q_log_privacy::mask_addr(recipient_wallet_hex), deposit_id
    );

    Ok(())
}

/// Debit QUG from wallet for WETH withdrawal (QUG → WETH direction).
/// Holds in escrow until bridge sends WETH on Ethereum.
pub async fn debit_qug_for_weth_withdrawal(
    state: &Arc<AppState>,
    wallet_addr: &[u8; 32],
    qug_amount: u128,
) -> Result<(), String> {
    let mut balances = state.wallet_balances.write().await;
    let current = balances.get(wallet_addr).copied().unwrap_or(0);

    if current < qug_amount {
        return Err(format!(
            "Insufficient QUG balance. Have {}, need {}",
            current, qug_amount
        ));
    }

    let new_balance = current.checked_sub(qug_amount)
        .ok_or("Balance underflow")?;
    balances.insert(*wallet_addr, new_balance);

    // Balance is persisted by the 15-second balance sync task
    // which writes all in-memory balances to RocksDB periodically

    Ok(())
}

/// Background task: Monitor pending WETH deposits for confirmations
/// and trigger committee attestation when 12+ confirmations reached.
pub async fn weth_deposit_monitor_tick(state: &Arc<AppState>) {
    let deposit_ids: Vec<String> = {
        let deposits = weth_deposits().read().await;
        deposits.iter()
            .filter(|(_, d)| matches!(d.status, WethDepositStatus::Pending | WethDepositStatus::Confirming))
            .map(|(id, _)| id.clone())
            .collect()
    };

    if deposit_ids.is_empty() {
        return;
    }

    for deposit_id in deposit_ids {
        let deposit = {
            let deps = weth_deposits().read().await;
            deps.get(&deposit_id).cloned()
        };

        let deposit = match deposit {
            Some(d) => d,
            None => continue,
        };

        // Verify on-chain via Reth
        let result = state.bridge_safety.verify_weth_erc20_deposit(
            &deposit.eth_tx_hash,
            deposit.amount_wei,
            Some(&deposit.sender_eth_address),
        ).await;

        match result {
            WethDepositVerification::Verified { confirmations, confirmed, .. } => {
                let mut deps = weth_deposits().write().await;
                if let Some(dep) = deps.get_mut(&deposit_id) {
                    dep.confirmations = confirmations;
                    dep.updated_at = Utc::now();

                    if confirmed && dep.status == WethDepositStatus::Pending {
                        dep.status = WethDepositStatus::Confirming;
                        info!(
                            "✅ [WETH BRIDGE] Deposit {} confirmed ({}/{} confirmations)",
                            deposit_id, confirmations, bridge_safety::ETH_MIN_CONFIRMATIONS
                        );
                    }

                    // Once confirmed, trigger attestation flow
                    if confirmed && dep.status == WethDepositStatus::Confirming {
                        dep.status = WethDepositStatus::Attesting;
                        dep.attestations = 7; // For now, single-node auto-attests
                        // TODO: Integrate with bridge_committee for full 7-of-11

                        // Credit QUG immediately (single-node mode)
                        let recipient = dep.recipient_qnk_wallet.clone();
                        let qug_amount = dep.qug_amount;
                        let dep_id = dep.deposit_id.clone();
                        let tx_hash = dep.eth_tx_hash.clone();

                        dep.status = WethDepositStatus::Completed;

                        // Persist updated deposit
                        if let Ok(bytes) = serde_json::to_vec(&dep) {
                            let key = format!("weth_deposit_{}", dep_id);
                            let _ = state.storage_engine.save_atomic_swap(&key, &bytes).await;
                        }

                        drop(deps); // Release lock before crediting

                        // Credit QUG
                        match credit_qug_for_weth_deposit(state, &dep_id, &recipient, qug_amount).await {
                            Ok(()) => {
                                // Mark tx_hash as claimed (replay protection)
                                state.bridge_safety.mark_txid_claimed(
                                    &state.storage_engine,
                                    &tx_hash,
                                    &dep_id,
                                ).await;
                                info!("🎉 [WETH BRIDGE] Deposit {} COMPLETE — {} QUG credited", dep_id, q_log_privacy::mask_amt(qug_amount));
                            }
                            Err(e) => {
                                error!("❌ [WETH BRIDGE] Failed to credit QUG for deposit {}: {}", dep_id, e);
                                // Revert status
                                let mut deps = weth_deposits().write().await;
                                if let Some(dep) = deps.get_mut(&dep_id) {
                                    dep.status = WethDepositStatus::Failed(e);
                                }
                            }
                        }
                        continue; // Already dropped lock, skip to next deposit
                    }

                    // Persist updated confirmations
                    if let Ok(bytes) = serde_json::to_vec(&dep) {
                        let key = format!("weth_deposit_{}", deposit_id);
                        let _ = state.storage_engine.save_atomic_swap(&key, &bytes).await;
                    }
                }
            }
            WethDepositVerification::NotFound => {
                // Transaction not yet indexed by Reth — keep waiting
            }
            WethDepositVerification::Failed(reason) => {
                warn!("❌ [WETH BRIDGE] Deposit {} verification failed: {}", deposit_id, reason);
                let mut deps = weth_deposits().write().await;
                if let Some(dep) = deps.get_mut(&deposit_id) {
                    dep.status = WethDepositStatus::Failed(reason);
                    dep.updated_at = Utc::now();
                }
            }
            WethDepositVerification::RpcError(e) => {
                warn!("⚠️ [WETH BRIDGE] RPC error checking deposit {}: {}", deposit_id, e);
                // Don't fail — will retry on next tick
            }
            WethDepositVerification::Frozen => {
                warn!("🛑 [WETH BRIDGE] Bridge frozen — deposit {} monitoring paused", deposit_id);
            }
        }
    }
}

// ============ Swap Restoration ============

/// Restore ETH swaps from persistent storage into in-memory maps on startup
pub async fn restore_swaps_from_storage(storage: &Arc<q_storage::StorageEngine>) {
    match storage.load_all_atomic_swaps().await {
        Ok(all_swaps) => {
            let mut restored = 0u32;
            for (swap_id, data) in &all_swaps {
                // Only restore eth_swap_* entries (skip BTC/ZEC/IRON swaps)
                if !swap_id.starts_with("eth_swap_") {
                    continue;
                }
                match serde_json::from_slice::<EthSwap>(data) {
                    Ok(swap) => {
                        let user = swap.user_address.clone();
                        let sid = swap.swap_id.clone();

                        // Insert into swap map
                        {
                            let mut swaps = eth_swaps().write().await;
                            swaps.insert(sid.clone(), swap);
                        }
                        // Insert into wallet index
                        {
                            let mut ws = eth_wallet_swaps().write().await;
                            ws.entry(user).or_insert_with(Vec::new).push(sid);
                        }
                        restored += 1;
                    }
                    Err(e) => {
                        warn!("⟠ Failed to deserialize ETH swap {}: {}", swap_id, e);
                    }
                }
            }
            if restored > 0 {
                info!("⟠ Restored {} ETH swaps from storage", restored);
            }
        }
        Err(e) => {
            warn!("⟠ Failed to load ETH swaps from storage: {}", e);
        }
    }
}
