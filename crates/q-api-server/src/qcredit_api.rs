/// QCREDIT Yield Vault API (v8.5.5)
///
/// Lock QUG 1:1 → mint QCREDIT, earn tiered yield (5-25% APY).
/// Digital credit layer: L1 capital (QUG) → L2 credit (QCREDIT) → L3 products.
use axum::{
    extract::State,
    http::StatusCode,
    Json,
};
use q_types::{ApiResponse, QCREDIT_TOKEN_ADDRESS};
use q_vm::contracts::qcredit_vault::CreditTier;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{info, warn};

use crate::AppState;
use q_types::Address;

/// Parse a wallet hex string (with or without "qnk" prefix) into Address
fn parse_wallet(wallet: &str) -> Result<Address, String> {
    let hex_part = if wallet.starts_with("qnk") { &wallet[3..] } else { wallet };
    let bytes = hex::decode(hex_part).map_err(|_| "Invalid wallet address hex".to_string())?;
    if bytes.len() != 32 {
        return Err("Wallet address must be 32 bytes".to_string());
    }
    let mut addr = [0u8; 32];
    addr.copy_from_slice(&bytes);
    Ok(addr)
}

// ============ REQUEST/RESPONSE TYPES ============

#[derive(Debug, Deserialize)]
pub struct LockRequest {
    pub wallet: String,
    pub amount: String,    // Human-readable QUG amount (e.g., "100.5")
    pub tier: String,      // "bronze", "silver", "gold", "platinum"
}

#[derive(Debug, Deserialize)]
pub struct UnlockRequest {
    pub wallet: String,
    pub position_index: usize,
}

#[derive(Debug, Deserialize)]
pub struct ClaimRequest {
    pub wallet: String,
    pub position_index: usize,
}

#[derive(Debug, Serialize)]
pub struct QCreditStatusResponse {
    pub total_locked: String,
    pub total_locked_raw: String,
    pub total_qcredit_supply: String,
    pub protocol_reserve: String,
    pub total_yield_paid: String,
    pub position_count: usize,
    pub tiers: Vec<TierResponse>,
}

#[derive(Debug, Serialize)]
pub struct TierResponse {
    pub name: String,
    pub lock_days: u64,
    pub apy_percent: f64,
}

#[derive(Debug, Serialize)]
pub struct PositionResponse {
    pub positions: Vec<PositionDetail>,
    pub total_locked: String,
    pub total_pending_yield: String,
}

#[derive(Debug, Serialize)]
pub struct PositionDetail {
    pub index: usize,
    pub amount_locked: String,
    pub qcredit_minted: String,
    pub tier: String,
    pub apy_percent: f64,
    pub lock_timestamp: u64,
    pub unlock_timestamp: u64,
    pub is_unlockable: bool,
    pub claimed_yield: String,
    pub pending_yield: String,
    pub lock_days_remaining: i64,
}

const QUG_DIVISOR: f64 = 1e24;

fn format_qug(raw: u128) -> String {
    format!("{:.8}", raw as f64 / QUG_DIVISOR)
}

// ============ HANDLERS ============

/// GET /api/v1/qcredit/status — vault overview
pub async fn get_status(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<QCreditStatusResponse>>, StatusCode> {
    let vault = state.qcredit_vault.read().await;
    let status = vault.status();

    Ok(Json(ApiResponse::success(QCreditStatusResponse {
        total_locked: format_qug(status.total_locked),
        total_locked_raw: status.total_locked.to_string(),
        total_qcredit_supply: format_qug(status.total_qcredit_supply),
        protocol_reserve: format_qug(status.protocol_reserve),
        total_yield_paid: format_qug(status.total_yield_paid),
        position_count: status.position_count,
        tiers: status.tiers.iter().map(|t| TierResponse {
            name: t.tier.display_name().to_string(),
            lock_days: t.lock_days,
            apy_percent: t.apy_percent,
        }).collect(),
    })))
}

/// GET /api/v1/qcredit/tiers — tier info
pub async fn get_tiers(
) -> Result<Json<ApiResponse<Vec<TierResponse>>>, StatusCode> {
    let tiers = q_vm::contracts::QCreditVault::get_tiers();
    Ok(Json(ApiResponse::success(tiers.iter().map(|t| TierResponse {
        name: t.tier.display_name().to_string(),
        lock_days: t.lock_days,
        apy_percent: t.apy_percent,
    }).collect())))
}

/// GET /api/v1/qcredit/position?wallet=<hex> — user positions + pending yield
pub async fn get_position(
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<PositionResponse>>, StatusCode> {
    let wallet_str = match params.get("wallet") {
        Some(w) => w.clone(),
        None => return Ok(Json(ApiResponse::error("Missing 'wallet' query parameter".into()))),
    };
    let address = match parse_wallet(&wallet_str) {
        Ok(a) => a,
        Err(e) => return Ok(Json(ApiResponse::error(e))),
    };
    let wallet_hex = hex::encode(address);
    let now = chrono::Utc::now().timestamp() as u64;
    let vault = state.qcredit_vault.read().await;
    let positions_with_yield = vault.get_positions_with_yield(&wallet_hex, now);

    let mut total_locked: u128 = 0;
    let mut total_pending: u128 = 0;
    let details: Vec<PositionDetail> = positions_with_yield
        .iter()
        .enumerate()
        .map(|(i, (pos, pending))| {
            total_locked += pos.amount_locked;
            total_pending += pending;
            let remaining_secs = pos.unlock_timestamp as i64 - now as i64;
            PositionDetail {
                index: i,
                amount_locked: format_qug(pos.amount_locked),
                qcredit_minted: format_qug(pos.qcredit_minted),
                tier: pos.tier.display_name().to_string(),
                apy_percent: pos.tier.apy_bps() as f64 / 100.0,
                lock_timestamp: pos.lock_timestamp,
                unlock_timestamp: pos.unlock_timestamp,
                is_unlockable: pos.is_unlockable(now),
                claimed_yield: format_qug(pos.claimed_yield),
                pending_yield: format_qug(*pending),
                lock_days_remaining: if remaining_secs > 0 { remaining_secs / 86400 } else { 0 },
            }
        })
        .collect();

    Ok(Json(ApiResponse::success(PositionResponse {
        positions: details,
        total_locked: format_qug(total_locked),
        total_pending_yield: format_qug(total_pending),
    })))
}

/// POST /api/v1/qcredit/lock — lock QUG, mint QCREDIT
pub async fn lock_qug(
    State(state): State<Arc<AppState>>,
    Json(req): Json<LockRequest>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let address = match parse_wallet(&req.wallet) {
        Ok(a) => a,
        Err(e) => return Ok(Json(ApiResponse::error(e))),
    };
    let wallet_hex = hex::encode(address);
    let now = chrono::Utc::now().timestamp() as u64;

    // Parse amount (human-readable to 24-decimal base units)
    let amount_f64: f64 = req.amount.parse().map_err(|_| StatusCode::BAD_REQUEST)?;
    if amount_f64 <= 0.0 {
        return Ok(Json(ApiResponse::error("Amount must be > 0".into())));
    }
    let amount_raw = (amount_f64 * QUG_DIVISOR) as u128;

    // Parse tier
    let tier = match CreditTier::from_str_name(&req.tier) {
        Some(t) => t,
        None => return Ok(Json(ApiResponse::error("Invalid tier. Use: bronze, silver, gold, platinum".into()))),
    };

    // Check QUG balance (wallet_balances stores Amount = u128)
    {
        let balances = state.wallet_balances.read().await;
        let bal = balances.get(&address).copied().unwrap_or(0);
        if bal < amount_raw {
            return Ok(Json(ApiResponse::error(
                format!("Insufficient QUG balance: have {}, need {}", format_qug(bal), format_qug(amount_raw))
            )));
        }
    }

    // Deduct QUG from wallet
    {
        let mut balances = state.wallet_balances.write().await;
        let entry = balances.entry(address).or_insert(0);
        *entry = entry.saturating_sub(amount_raw);
    }

    // Lock in vault
    let position = {
        let mut vault = state.qcredit_vault.write().await;
        match vault.lock(&wallet_hex, amount_raw, tier, now) {
            Ok(pos) => pos,
            Err(e) => {
                // Refund QUG on failure
                let mut balances = state.wallet_balances.write().await;
                let entry = balances.entry(address).or_insert(0);
                *entry = entry.saturating_add(amount_raw);
                return Ok(Json(ApiResponse::error(e)));
            }
        }
    };

    // Mint QCREDIT to wallet token balance
    {
        let mut token_balances = state.token_balances.write().await;
        let key = (address, QCREDIT_TOKEN_ADDRESS);
        let entry = token_balances.entry(key).or_insert(0);
        *entry = entry.saturating_add(amount_raw);
    }

    // Persist vault + balances
    if let Err(e) = persist_vault(&state).await {
        warn!("Failed to persist QCREDIT vault: {}", e);
    }
    let _ = state.storage_engine.save_token_balance(&address, &QCREDIT_TOKEN_ADDRESS, {
        let tb = state.token_balances.read().await;
        tb.get(&(address, QCREDIT_TOKEN_ADDRESS)).copied().unwrap_or(0)
    }).await;
    // v9.5.1: Persist QUG balance after deduction
    let _ = state.storage_engine.save_wallet_balance(&address, {
        let wb = state.wallet_balances.read().await;
        wb.get(&address).copied().unwrap_or(0)
    }).await;

    info!(
        "💳 [QCREDIT] {} locked {} QUG -> {} QCREDIT ({} tier, unlock at {})",
        &wallet_hex[..8], format_qug(amount_raw), format_qug(amount_raw),
        tier.display_name(), position.unlock_timestamp
    );

    Ok(Json(ApiResponse::success(serde_json::json!({
        "locked": format_qug(amount_raw),
        "qcredit_minted": format_qug(amount_raw),
        "tier": tier.display_name(),
        "unlock_timestamp": position.unlock_timestamp,
    }))))
}

/// POST /api/v1/qcredit/unlock — burn QCREDIT, return QUG + yield
pub async fn unlock_position(
    State(state): State<Arc<AppState>>,
    Json(req): Json<UnlockRequest>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let address = match parse_wallet(&req.wallet) {
        Ok(a) => a,
        Err(e) => return Ok(Json(ApiResponse::error(e))),
    };
    let wallet_hex = hex::encode(address);
    let now = chrono::Utc::now().timestamp() as u64;

    let (qug_returned, yield_claimed) = {
        let mut vault = state.qcredit_vault.write().await;
        match vault.unlock(&wallet_hex, req.position_index, now) {
            Ok(result) => result,
            Err(e) => return Ok(Json(ApiResponse::error(e))),
        }
    };

    // Burn QCREDIT from token balance
    {
        let mut token_balances = state.token_balances.write().await;
        let key = (address, QCREDIT_TOKEN_ADDRESS);
        let entry = token_balances.entry(key).or_insert(0);
        *entry = entry.saturating_sub(qug_returned);
    }

    // Credit QUG back + yield
    {
        let mut balances = state.wallet_balances.write().await;
        let entry = balances.entry(address).or_insert(0);
        *entry = entry.saturating_add(qug_returned.saturating_add(yield_claimed));
    }

    // Persist vault + balances
    if let Err(e) = persist_vault(&state).await {
        warn!("Failed to persist QCREDIT vault: {}", e);
    }
    // v9.5.1: Persist QCREDIT token balance after burn
    let _ = state.storage_engine.save_token_balance(&address, &QCREDIT_TOKEN_ADDRESS, {
        let tb = state.token_balances.read().await;
        tb.get(&(address, QCREDIT_TOKEN_ADDRESS)).copied().unwrap_or(0)
    }).await;
    // v9.5.1: Persist QUG balance after credit
    let _ = state.storage_engine.save_wallet_balance(&address, {
        let wb = state.wallet_balances.read().await;
        wb.get(&address).copied().unwrap_or(0)
    }).await;

    info!(
        "💳 [QCREDIT] {} unlocked: {} QUG returned + {} yield",
        &wallet_hex[..8], format_qug(qug_returned), format_qug(yield_claimed)
    );

    Ok(Json(ApiResponse::success(serde_json::json!({
        "qug_returned": format_qug(qug_returned),
        "yield_claimed": format_qug(yield_claimed),
        "total_received": format_qug(qug_returned + yield_claimed),
    }))))
}

/// POST /api/v1/qcredit/claim — claim accrued yield without unlocking
pub async fn claim_yield(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ClaimRequest>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let address = match parse_wallet(&req.wallet) {
        Ok(a) => a,
        Err(e) => return Ok(Json(ApiResponse::error(e))),
    };
    let wallet_hex = hex::encode(address);
    let now = chrono::Utc::now().timestamp() as u64;

    let yield_amount = {
        let mut vault = state.qcredit_vault.write().await;
        match vault.claim_yield(&wallet_hex, req.position_index, now) {
            Ok(amount) => amount,
            Err(e) => return Ok(Json(ApiResponse::error(e))),
        }
    };

    // Credit yield as QUG
    {
        let mut balances = state.wallet_balances.write().await;
        let entry = balances.entry(address).or_insert(0);
        *entry = entry.saturating_add(yield_amount);
    }

    // Persist vault + QUG balance
    if let Err(e) = persist_vault(&state).await {
        warn!("Failed to persist QCREDIT vault: {}", e);
    }
    // v9.5.1: Persist QUG balance after yield credit
    let _ = state.storage_engine.save_wallet_balance(&address, {
        let wb = state.wallet_balances.read().await;
        wb.get(&address).copied().unwrap_or(0)
    }).await;

    info!(
        "💳 [QCREDIT] {} claimed {} yield",
        &wallet_hex[..8], format_qug(yield_amount)
    );

    Ok(Json(ApiResponse::success(serde_json::json!({
        "yield_claimed": format_qug(yield_amount),
    }))))
}

/// Persist vault state to RocksDB
async fn persist_vault(state: &Arc<AppState>) -> Result<(), String> {
    let vault = state.qcredit_vault.read().await;
    let bytes = serde_json::to_vec(&*vault).map_err(|e| e.to_string())?;
    state.storage_engine.save_qcredit_vault(&bytes).await.map_err(|e| e.to_string())
}
