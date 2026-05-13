/// Bitcoin Deposit Bridge API — Phase 1 (v10.9.3)
///
/// Endpoints for BTC↔wBTC via Bitcoin Knots on Delta.
///
/// Endpoints:
///   POST /api/v1/bitcoin/deposit/address   — generate deposit address (auth required)
///   GET  /api/v1/bitcoin/deposit/:id       — get deposit status
///   GET  /api/v1/bitcoin/deposits          — list deposits for authenticated wallet
///   GET  /api/v1/bitcoin/deposit/bridge-status — bridge health/TVL summary
///   POST /api/v1/bitcoin/withdraw          — redeem wBTC for on-chain BTC (auth required)

use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{info, warn};

use crate::handlers::ApiResponse;
use crate::wallet_auth::AuthenticatedWallet;
use crate::AppState;
use q_types::WBTC_TOKEN_ADDRESS;

use q_bitcoin_bridge::bitcoin::{address::NetworkUnchecked, Address, Network as BtcNetwork};
use std::str::FromStr;

// ============================================================================
// Request / Response types
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct CreateDepositRequest {
    /// Optional: amount hint in satoshis (informational, not enforced on-chain)
    pub amount_hint_sats: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct DepositAddressResponse {
    pub deposit_id: String,
    pub btc_address: String,
    pub min_deposit_sats: u64,
    pub max_deposit_sats: u64,
    pub min_confirmations: u32,
    pub expires_in_secs: u64,
    pub qr_uri: String,
}

#[derive(Debug, Serialize)]
pub struct DepositStatusResponse {
    pub deposit_id: String,
    pub btc_address: String,
    pub status: String,
    pub amount_sats: u64,
    pub amount_btc: f64,
    pub confirmations: u32,
    pub min_confirmations: u32,
    pub txid: Option<String>,
    pub created_at: u64,
    pub updated_at: u64,
}

#[derive(Debug, Serialize)]
pub struct BridgeStatusResponse {
    pub alive: bool,
    pub wallet_balance_btc: f64,
    pub total_minted_sats: u64,
    pub total_minted_btc: f64,
    pub deposits_awaiting: u32,
    pub deposits_detected: u32,
    pub deposits_confirming: u32,
    pub deposits_minted: u32,
    pub deposits_failed: u32,
    pub max_deposit_sats: u64,
    pub max_deposit_btc: f64,
    pub max_tvl_sats: u64,
    pub max_tvl_btc: f64,
    pub min_confirmations: u32,
}

// ============================================================================
// Handlers
// ============================================================================

/// POST /api/v1/bitcoin/deposit/address
///
/// Generate a new BTC deposit address for the authenticated wallet.
/// The user sends BTC to this address; after 6 confirmations, wBTC is minted.
pub async fn create_deposit_address(
    State(state): State<Arc<AppState>>,
    auth_wallet: Option<AuthenticatedWallet>,
    Json(request): Json<CreateDepositRequest>,
) -> Result<Json<ApiResponse<DepositAddressResponse>>, StatusCode> {
    let wallet = match auth_wallet {
        Some(w) => w,
        None => {
            return Ok(Json(ApiResponse::error(
                "Authentication required. Log in to your wallet first.".to_string(),
            )));
        }
    };

    let bridge = match &state.deposit_bridge {
        Some(b) => b,
        None => {
            return Ok(Json(ApiResponse::error(
                "Bitcoin deposit bridge is not enabled on this node.".to_string(),
            )));
        }
    };

    // Validate amount hint if provided
    if let Some(hint) = request.amount_hint_sats {
        if hint > q_bitcoin_bridge::deposit_bridge::BTC_MAX_DEPOSIT_SATS {
            return Ok(Json(ApiResponse::error(format!(
                "Amount exceeds maximum deposit of {} sats (0.1 BTC)",
                q_bitcoin_bridge::deposit_bridge::BTC_MAX_DEPOSIT_SATS
            ))));
        }
        if hint > 0 && hint < q_bitcoin_bridge::deposit_bridge::BTC_MIN_DEPOSIT_SATS {
            return Ok(Json(ApiResponse::error(format!(
                "Amount below minimum deposit of {} sats",
                q_bitcoin_bridge::deposit_bridge::BTC_MIN_DEPOSIT_SATS
            ))));
        }
    }

    match bridge.create_deposit_address(wallet.address).await {
        Ok(deposit) => {
            let amount_for_qr = request.amount_hint_sats.unwrap_or(0);
            let qr_uri = if amount_for_qr > 0 {
                format!(
                    "bitcoin:{}?amount={:.8}",
                    deposit.btc_address,
                    amount_for_qr as f64 / 100_000_000.0
                )
            } else {
                format!("bitcoin:{}", deposit.btc_address)
            };

            info!(
                "₿ Deposit address created: {} for wallet {}",
                deposit.btc_address,
                hex::encode(&wallet.address[..8])
            );

            Ok(Json(ApiResponse::success(DepositAddressResponse {
                deposit_id: deposit.deposit_id,
                btc_address: deposit.btc_address,
                min_deposit_sats: q_bitcoin_bridge::deposit_bridge::BTC_MIN_DEPOSIT_SATS,
                max_deposit_sats: q_bitcoin_bridge::deposit_bridge::BTC_MAX_DEPOSIT_SATS,
                min_confirmations: q_bitcoin_bridge::deposit_bridge::BTC_MIN_CONFIRMATIONS,
                expires_in_secs: q_bitcoin_bridge::deposit_bridge::DEPOSIT_ADDR_EXPIRY_SECS,
                qr_uri,
            })))
        }
        Err(e) => {
            // SECURITY: Log full error server-side, return sanitized message to client
            warn!("₿ Failed to create deposit address: {}", e);
            let client_msg = if e.to_string().contains("Rate limit") {
                e.to_string() // Rate limit errors are safe to show
            } else if e.to_string().contains("disabled") || e.to_string().contains("killed") {
                "Bridge is temporarily unavailable. Please try again later.".to_string()
            } else {
                "Failed to create deposit address. Please try again later.".to_string()
            };
            Ok(Json(ApiResponse::error(client_msg)))
        }
    }
}

/// GET /api/v1/bitcoin/deposit/:id
///
/// Get the status of a specific deposit. Requires auth — only the deposit owner can query.
pub async fn get_deposit_status(
    State(state): State<Arc<AppState>>,
    auth_wallet: Option<AuthenticatedWallet>,
    Path(deposit_id): Path<String>,
) -> Result<Json<ApiResponse<DepositStatusResponse>>, StatusCode> {
    let wallet = match auth_wallet {
        Some(w) => w,
        None => {
            return Ok(Json(ApiResponse::error(
                "Authentication required.".to_string(),
            )));
        }
    };

    let bridge = match &state.deposit_bridge {
        Some(b) => b,
        None => {
            return Ok(Json(ApiResponse::error(
                "Bitcoin deposit bridge is not enabled.".to_string(),
            )));
        }
    };

    match bridge.get_deposit_for_wallet(&deposit_id, &wallet.address).await {
        Some(deposit) => {
            let (status_str, confs, txid) = match &deposit.status {
                q_bitcoin_bridge::deposit_bridge::DepositStatus::Awaiting => {
                    ("awaiting".to_string(), 0u32, None)
                }
                q_bitcoin_bridge::deposit_bridge::DepositStatus::Detected {
                    txid,
                    confirmations,
                    ..
                } => ("detected".to_string(), *confirmations, Some(txid.clone())),
                q_bitcoin_bridge::deposit_bridge::DepositStatus::Confirming {
                    txid,
                    confirmations,
                    ..
                } => ("confirming".to_string(), *confirmations, Some(txid.clone())),
                q_bitcoin_bridge::deposit_bridge::DepositStatus::Minted { txid, .. } => (
                    "minted".to_string(),
                    q_bitcoin_bridge::deposit_bridge::BTC_MIN_CONFIRMATIONS,
                    Some(txid.clone()),
                ),
                q_bitcoin_bridge::deposit_bridge::DepositStatus::Expired => {
                    ("expired".to_string(), 0, None)
                }
                q_bitcoin_bridge::deposit_bridge::DepositStatus::Failed { reason } => {
                    (format!("failed: {}", reason), 0, None)
                }
            };

            Ok(Json(ApiResponse::success(DepositStatusResponse {
                deposit_id: deposit.deposit_id,
                btc_address: deposit.btc_address,
                status: status_str,
                amount_sats: deposit.amount_sats,
                amount_btc: deposit.amount_sats as f64 / 100_000_000.0,
                confirmations: confs,
                min_confirmations: q_bitcoin_bridge::deposit_bridge::BTC_MIN_CONFIRMATIONS,
                txid,
                created_at: deposit.created_at,
                updated_at: deposit.updated_at,
            })))
        }
        None => Ok(Json(ApiResponse::error(format!(
            "Deposit {} not found",
            deposit_id
        )))),
    }
}

/// GET /api/v1/bitcoin/deposits
///
/// List all deposits for the authenticated wallet.
pub async fn list_deposits(
    State(state): State<Arc<AppState>>,
    auth_wallet: Option<AuthenticatedWallet>,
) -> Result<Json<ApiResponse<Vec<DepositStatusResponse>>>, StatusCode> {
    let wallet = match auth_wallet {
        Some(w) => w,
        None => {
            return Ok(Json(ApiResponse::error(
                "Authentication required.".to_string(),
            )));
        }
    };

    let bridge = match &state.deposit_bridge {
        Some(b) => b,
        None => {
            return Ok(Json(ApiResponse::error(
                "Bitcoin deposit bridge is not enabled.".to_string(),
            )));
        }
    };

    let deposits = bridge.list_deposits_for_wallet(&wallet.address).await;
    let responses: Vec<DepositStatusResponse> = deposits
        .into_iter()
        .map(|deposit| {
            let (status_str, confs, txid) = match &deposit.status {
                q_bitcoin_bridge::deposit_bridge::DepositStatus::Awaiting => {
                    ("awaiting".to_string(), 0u32, None)
                }
                q_bitcoin_bridge::deposit_bridge::DepositStatus::Detected {
                    txid,
                    confirmations,
                    ..
                } => ("detected".to_string(), *confirmations, Some(txid.clone())),
                q_bitcoin_bridge::deposit_bridge::DepositStatus::Confirming {
                    txid,
                    confirmations,
                    ..
                } => ("confirming".to_string(), *confirmations, Some(txid.clone())),
                q_bitcoin_bridge::deposit_bridge::DepositStatus::Minted { txid, .. } => (
                    "minted".to_string(),
                    q_bitcoin_bridge::deposit_bridge::BTC_MIN_CONFIRMATIONS,
                    Some(txid.clone()),
                ),
                q_bitcoin_bridge::deposit_bridge::DepositStatus::Expired => {
                    ("expired".to_string(), 0, None)
                }
                q_bitcoin_bridge::deposit_bridge::DepositStatus::Failed { reason } => {
                    (format!("failed: {}", reason), 0, None)
                }
            };

            DepositStatusResponse {
                deposit_id: deposit.deposit_id,
                btc_address: deposit.btc_address,
                status: status_str,
                amount_sats: deposit.amount_sats,
                amount_btc: deposit.amount_sats as f64 / 100_000_000.0,
                confirmations: confs,
                min_confirmations: q_bitcoin_bridge::deposit_bridge::BTC_MIN_CONFIRMATIONS,
                txid,
                created_at: deposit.created_at,
                updated_at: deposit.updated_at,
            }
        })
        .collect();

    Ok(Json(ApiResponse::success(responses)))
}

// ============================================================================
// Withdrawal (wBTC → BTC)
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct WithdrawRequest {
    /// Destination on-chain BTC address (bech32 bc1q… or legacy)
    pub btc_address: String,
    /// Amount to withdraw in satoshis
    pub amount_sats: u64,
    /// Optional fee priority: "economy" (~60min), "normal" (~30min), "fast" (~10min).
    /// Maps to Knots `conf_target` of 30, 6, and 2 blocks respectively.
    /// Defaults to "normal" when omitted.
    #[serde(default)]
    pub fee_priority: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct WithdrawResponse {
    pub txid: String,
    pub amount_sats: u64,
    pub amount_btc: f64,
    pub btc_address: String,
}

/// POST /api/v1/bitcoin/withdraw
///
/// Redeem wBTC for real on-chain BTC.
/// Deducts wBTC from the caller's balance, then broadcasts a Bitcoin transaction.
///
/// SECURITY order: wBTC is deducted BEFORE the Bitcoin transaction is broadcast.
/// If the broadcast fails, the wBTC is refunded.
pub async fn withdraw_wbtc(
    State(state): State<Arc<AppState>>,
    auth_wallet: Option<AuthenticatedWallet>,
    Json(request): Json<WithdrawRequest>,
) -> Result<Json<ApiResponse<WithdrawResponse>>, StatusCode> {
    let wallet = match auth_wallet {
        Some(w) => w,
        None => return Ok(Json(ApiResponse::error("Authentication required.".to_string()))),
    };

    let bridge = match &state.deposit_bridge {
        Some(b) => b.clone(),
        None => return Ok(Json(ApiResponse::error(
            "Bitcoin bridge is not enabled on this node.".to_string(),
        ))),
    };

    // Validate amount
    if request.amount_sats < q_bitcoin_bridge::deposit_bridge::BTC_MIN_DEPOSIT_SATS {
        return Ok(Json(ApiResponse::error(format!(
            "Minimum withdrawal is {} sats",
            q_bitcoin_bridge::deposit_bridge::BTC_MIN_DEPOSIT_SATS
        ))));
    }
    if request.amount_sats > q_bitcoin_bridge::deposit_bridge::BTC_MAX_DEPOSIT_SATS {
        return Ok(Json(ApiResponse::error(format!(
            "Maximum withdrawal is {} sats (0.1 BTC)",
            q_bitcoin_bridge::deposit_bridge::BTC_MAX_DEPOSIT_SATS
        ))));
    }

    // Validate BTC address against mainnet using rust-bitcoin parser.
    let addr_str = request.btc_address.trim();
    let parsed = match Address::<NetworkUnchecked>::from_str(addr_str) {
        Ok(a) => match a.require_network(BtcNetwork::Bitcoin) {
            Ok(a) => a,
            Err(e) => {
                return Ok(Json(ApiResponse::error(format!(
                    "Address is not on Bitcoin mainnet: {}",
                    e
                ))));
            }
        },
        Err(e) => {
            return Ok(Json(ApiResponse::error(format!(
                "Invalid BTC address: {}",
                e
            ))));
        }
    };
    let canonical = parsed.to_string();
    let addr = canonical.as_str();

    // Map fee priority → Knots conf_target (blocks).
    let conf_target: Option<u32> = match request
        .fee_priority
        .as_deref()
        .map(|s| s.to_ascii_lowercase())
        .as_deref()
    {
        Some("economy") => Some(30),
        Some("normal") | None | Some("") => Some(6),
        Some("fast") => Some(2),
        Some(other) => {
            return Ok(Json(ApiResponse::error(format!(
                "Unknown fee_priority '{}'. Use 'economy', 'normal', or 'fast'.",
                other
            ))));
        }
    };

    // wBTC is stored in 8-decimal satoshi units
    let required_wbtc = request.amount_sats; // 1 wBTC sat = 1 BTC sat

    // --- Step 1: Check and deduct wBTC balance ---
    let token_key = (wallet.address, WBTC_TOKEN_ADDRESS);
    let old_balance = {
        let balances = state.token_balances.read().await;
        balances.get(&token_key).copied().unwrap_or(0)
    };
    if old_balance < required_wbtc as u128 {
        return Ok(Json(ApiResponse::error(format!(
            "Insufficient wBTC balance: have {} sats, need {} sats",
            old_balance, required_wbtc
        ))));
    }

    let new_balance = old_balance - required_wbtc as u128;
    {
        let mut balances = state.token_balances.write().await;
        balances.insert(token_key, new_balance);
    }
    if let Err(e) = state.storage_engine
        .save_token_balance(&wallet.address, &WBTC_TOKEN_ADDRESS, new_balance)
        .await
    {
        warn!("₿ Failed to persist wBTC deduction before withdrawal: {}", e);
        // Restore balance and abort — storage error means we can't guarantee atomicity
        let mut balances = state.token_balances.write().await;
        balances.insert(token_key, old_balance);
        return Ok(Json(ApiResponse::error(
            "Storage error — withdrawal aborted. No funds moved.".to_string(),
        )));
    }

    // --- Step 2: Broadcast the Bitcoin transaction ---
    match bridge.send_withdrawal(addr, request.amount_sats, conf_target).await {
        Ok(txid) => {
            info!(
                "₿ wBTC withdrawal: {} sats → {} (txid: {}) for wallet {}",
                request.amount_sats, addr, txid,
                hex::encode(&wallet.address[..8])
            );
            Ok(Json(ApiResponse::success(WithdrawResponse {
                txid,
                amount_sats: request.amount_sats,
                amount_btc: request.amount_sats as f64 / 100_000_000.0,
                btc_address: addr.to_string(),
            })))
        }
        Err(e) => {
            // Broadcast failed — refund the wBTC we just deducted
            warn!("₿ BTC broadcast failed after wBTC deduction — refunding: {}", e);
            let mut balances = state.token_balances.write().await;
            balances.insert(token_key, old_balance);
            drop(balances);
            let _ = state.storage_engine
                .save_token_balance(&wallet.address, &WBTC_TOKEN_ADDRESS, old_balance)
                .await;
            Ok(Json(ApiResponse::error(format!(
                "Bitcoin transaction failed — your wBTC was not deducted: {}",
                e
            ))))
        }
    }
}

/// GET /api/v1/bitcoin/deposit/bridge-status
///
/// Get bridge health, TVL, and deposit statistics.
/// Wallet balance is redacted for unauthenticated callers.
pub async fn get_deposit_bridge_status(
    State(state): State<Arc<AppState>>,
    auth_wallet: Option<AuthenticatedWallet>,
) -> Result<Json<ApiResponse<BridgeStatusResponse>>, StatusCode> {
    let bridge = match &state.deposit_bridge {
        Some(b) => b,
        None => {
            return Ok(Json(ApiResponse::error(
                "Bitcoin deposit bridge is not enabled.".to_string(),
            )));
        }
    };

    let status = bridge.get_status().await;

    // SECURITY: Only show wallet balance to authenticated users
    let wallet_balance = if auth_wallet.is_some() {
        status.wallet_balance_btc
    } else {
        -1.0 // Sentinel: redacted
    };

    Ok(Json(ApiResponse::success(BridgeStatusResponse {
        alive: status.alive,
        wallet_balance_btc: wallet_balance,
        total_minted_sats: status.total_minted_sats,
        total_minted_btc: status.total_minted_sats as f64 / 100_000_000.0,
        deposits_awaiting: status.deposits_awaiting,
        deposits_detected: status.deposits_detected,
        deposits_confirming: status.deposits_confirming,
        deposits_minted: status.deposits_minted,
        deposits_failed: status.deposits_failed,
        max_deposit_sats: status.max_deposit_sats,
        max_deposit_btc: status.max_deposit_sats as f64 / 100_000_000.0,
        max_tvl_sats: status.max_tvl_sats,
        max_tvl_btc: status.max_tvl_sats as f64 / 100_000_000.0,
        min_confirmations: status.min_confirmations,
    })))
}
