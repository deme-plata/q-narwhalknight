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

/// Query Reth sync status
async fn get_reth_sync_status() -> (bool, u64) {
    match get_reth_height().await {
        Some(height) => {
            // Consider synced if height > 19M (approximate current Ethereum height)
            (height > 19_000_000, height)
        }
        None => (false, 0),
    }
}

// ============ Endpoint Handlers ============

/// GET /api/v1/ethereum/bridge/status — Bridge health status
pub async fn get_eth_bridge_status(
    State(_state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<EthBridgeStatusResponse>>, StatusCode> {
    let (synced, height) = get_reth_sync_status().await;

    let response = EthBridgeStatusResponse {
        bridge_enabled: height > 0,
        reth_rpc_url: RETH_RPC_URL.to_string(),
        reth_height: height,
        reth_synced: synced,
        network: "mainnet".to_string(),
        features: vec![
            "htlc-atomic-swap".to_string(),
            "erc20-bridge".to_string(),
            "wrapped-eth".to_string(),
        ],
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
    info!("⟠ Creating ETH atomic swap for wallet {} direction={}", wallet_hex, request.direction);

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
        info!("⟠ Minting wETH for {} (amount: {} wei)", user_address, eth_amount_str);
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
        info!("⟠ Burning wETH from {} (amount: {} wei)", user_address, eth_amount_str);
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

    info!("⟠ ETH swap {} refunded for wallet {}", swap_id, wallet_hex);

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

    info!("⟠ wETH transfer: {} wei from {} to {}", amount, from_hex, to_hex);

    Ok(Json(ApiResponse::success(SendEthResponse {
        tx_id,
        from: format!("qnk{}", from_hex),
        to: request.to_address,
        amount_wei: amount.to_string(),
    })))
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
