/// QUSD Issuer-Controlled Stablecoin API (v8.5.9)
///
/// USD-pegged stablecoin with transparent founder mint authority (like USDT/USDC).
/// All mints/burns are recorded in an append-only audit log for full transparency.
/// No hidden operations — every supply change is publicly verifiable.
use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use q_types::{ApiResponse, QUSD_TOKEN_ADDRESS, QUSD_DECIMALS};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{info, warn, error};

use crate::AppState;
use crate::aegis_auth_middleware::FOUNDER_WALLET;
use q_api_server::wallet_auth::AuthenticatedWallet;

// ============ REQUEST/RESPONSE TYPES ============

#[derive(Debug, Deserialize)]
pub struct QusdMintRequest {
    pub target_wallet: String,
    pub amount: String,         // Human-readable (e.g., "1000.50")
    #[serde(default)]
    pub memo: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct QusdBurnRequest {
    pub amount: String,         // Human-readable (e.g., "500.00")
    #[serde(default)]
    pub memo: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct QusdTransferRequest {
    pub to: String,             // Recipient wallet hex
    pub amount: String,         // Human-readable (e.g., "100.00")
}

#[derive(Debug, Serialize)]
pub struct QusdMintResponse {
    pub tx_hash: String,
    pub target_wallet: String,
    pub amount: String,
    pub amount_raw: String,
    pub new_total_supply: String,
    pub memo: String,
    pub timestamp: i64,
}

#[derive(Debug, Serialize)]
pub struct QusdBurnResponse {
    pub tx_hash: String,
    pub amount: String,
    pub amount_raw: String,
    pub new_total_supply: String,
    pub memo: String,
    pub timestamp: i64,
}

#[derive(Debug, Serialize)]
pub struct QusdTransferResponse {
    pub tx_hash: String,
    pub from: String,
    pub to: String,
    pub amount: String,
}

#[derive(Debug, Serialize)]
pub struct QusdBalanceResponse {
    pub address: String,
    pub balance: String,
    pub balance_raw: String,
}

#[derive(Debug, Serialize)]
pub struct QusdSupplyResponse {
    pub total_supply: String,
    pub total_supply_raw: String,
    pub total_minted: String,
    pub total_burned: String,
    pub mint_count: u64,
    pub burn_count: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct QusdAuditEntry {
    pub action: String,       // "mint" or "burn"
    pub amount: String,       // Human-readable
    pub amount_raw: String,   // Base units
    pub wallet: String,       // Target wallet for mints, founder for burns
    pub memo: String,
    pub timestamp: i64,
    pub tx_hash: String,
}

// ============ HELPER FUNCTIONS ============

const QUSD_DIVISOR: f64 = 1e24;

/// Parse a human-readable QUSD amount to base units (24 decimals)
fn parse_qusd_amount(amount_str: &str) -> Result<u128, String> {
    let amount_str = amount_str.trim();
    let parts: Vec<&str> = amount_str.split('.').collect();
    if parts.len() > 2 {
        return Err("Invalid amount format".to_string());
    }
    let whole: u128 = parts[0].parse().map_err(|_| "Invalid whole number")?;
    let frac: u128 = if parts.len() == 2 {
        let frac_str = parts[1];
        if frac_str.len() > QUSD_DECIMALS as usize {
            return Err(format!("Too many decimal places (max {})", QUSD_DECIMALS));
        }
        let padded = format!("{:0<width$}", frac_str, width = QUSD_DECIMALS as usize);
        padded.parse().map_err(|_| "Invalid fractional part")?
    } else {
        0
    };
    let base: u128 = 10u128.pow(QUSD_DECIMALS as u32);
    Ok(whole.checked_mul(base).ok_or("Amount overflow")? + frac)
}

/// Format base units to human-readable QUSD amount
fn format_qusd_amount(raw: u128) -> String {
    let base: u128 = 10u128.pow(QUSD_DECIMALS as u32);
    let whole = raw / base;
    let frac = raw % base;
    // Show 8 decimal places for readability
    let frac_str = format!("{:024}", frac);
    let trimmed = &frac_str[..8];
    format!("{}.{}", whole, trimmed)
}

/// Generate a tx_hash from timestamp and action
fn generate_tx_hash(action: &str, timestamp: i64, amount: u128) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    action.hash(&mut hasher);
    timestamp.hash(&mut hasher);
    amount.hash(&mut hasher);
    format!("qusd_{:016x}", hasher.finish())
}

/// Check if the authenticated wallet is the founder
fn is_founder(auth: &AuthenticatedWallet) -> bool {
    hex::encode(auth.address) == FOUNDER_WALLET
}

// ============ ENDPOINTS ============

/// POST /api/v1/qusd/mint — Mint QUSD to a target wallet (founder only)
pub async fn mint_qusd(
    auth: AuthenticatedWallet,
    State(state): State<Arc<AppState>>,
    Json(req): Json<QusdMintRequest>,
) -> Result<Json<ApiResponse<QusdMintResponse>>, StatusCode> {
    // Auth: Only founder can mint
    if !is_founder(&auth) {
        warn!("🚫 [QUSD] Non-founder attempted to mint: {}", hex::encode(auth.address));
        return Ok(Json(ApiResponse::error("Forbidden: only founder wallet can mint QUSD".to_string())));
    }

    // Parse amount
    let amount_raw = match parse_qusd_amount(&req.amount) {
        Ok(a) if a > 0 => a,
        Ok(_) => return Ok(Json(ApiResponse::error("Amount must be greater than 0".to_string()))),
        Err(e) => return Ok(Json(ApiResponse::error(format!("Invalid amount: {}", e)))),
    };

    // Parse target wallet
    let target_bytes: [u8; 32] = match hex::decode(&req.target_wallet) {
        Ok(b) if b.len() == 32 => {
            let mut arr = [0u8; 32];
            arr.copy_from_slice(&b);
            arr
        }
        _ => return Ok(Json(ApiResponse::error("Invalid target wallet address (must be 64 hex chars)".to_string()))),
    };

    let timestamp = chrono::Utc::now().timestamp();
    let memo = req.memo.unwrap_or_default();
    let tx_hash = generate_tx_hash("mint", timestamp, amount_raw);

    // Credit QUSD to target wallet using existing token balance system
    let current_balance = state.storage_engine.get_token_balance(&target_bytes, &QUSD_TOKEN_ADDRESS).await.unwrap_or(0);
    let new_balance = current_balance.saturating_add(amount_raw);
    if let Err(e) = state.storage_engine.save_token_balance(&target_bytes, &QUSD_TOKEN_ADDRESS, new_balance).await {
        error!("💵 [QUSD] Failed to save token balance: {}", e);
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    }

    // Update in-memory token balances
    {
        let mut token_balances = state.token_balances.write().await;
        let key = (target_bytes, QUSD_TOKEN_ADDRESS);
        let entry = token_balances.entry(key).or_insert(0);
        *entry = new_balance;
    }

    // Record audit entry
    let audit_entry = serde_json::json!({
        "action": "mint",
        "amount": format_qusd_amount(amount_raw),
        "amount_raw": amount_raw.to_string(),
        "wallet": req.target_wallet,
        "memo": memo,
        "timestamp": timestamp,
        "tx_hash": tx_hash,
    });

    if let Err(e) = state.storage_engine.save_qusd_audit_entry(&audit_entry).await {
        error!("💵 [QUSD] Failed to save audit entry: {}", e);
        // Balance already credited — log but don't fail
    }

    // Update in-memory state
    {
        let mut audit_log = state.qusd_audit_log.write().await;
        audit_log.push(audit_entry);
    }
    {
        let mut supply = state.qusd_total_supply.write().await;
        *supply = supply.saturating_add(amount_raw);
    }

    let new_supply = *state.qusd_total_supply.read().await;

    info!(
        "💵 [QUSD] MINTED {} QUSD to {} (memo: {}, tx: {}, new supply: {})",
        format_qusd_amount(amount_raw), &req.target_wallet[..16], memo, tx_hash,
        format_qusd_amount(new_supply)
    );

    Ok(Json(ApiResponse::success(QusdMintResponse {
        tx_hash,
        target_wallet: req.target_wallet,
        amount: format_qusd_amount(amount_raw),
        amount_raw: amount_raw.to_string(),
        new_total_supply: format_qusd_amount(new_supply),
        memo,
        timestamp,
    })))
}

/// POST /api/v1/qusd/burn — Burn QUSD from founder's balance (founder only)
pub async fn burn_qusd(
    auth: AuthenticatedWallet,
    State(state): State<Arc<AppState>>,
    Json(req): Json<QusdBurnRequest>,
) -> Result<Json<ApiResponse<QusdBurnResponse>>, StatusCode> {
    // Auth: Only founder can burn
    if !is_founder(&auth) {
        warn!("🚫 [QUSD] Non-founder attempted to burn: {}", hex::encode(auth.address));
        return Ok(Json(ApiResponse::error("Forbidden: only founder wallet can burn QUSD".to_string())));
    }

    // Parse amount
    let amount_raw = match parse_qusd_amount(&req.amount) {
        Ok(a) if a > 0 => a,
        Ok(_) => return Ok(Json(ApiResponse::error("Amount must be greater than 0".to_string()))),
        Err(e) => return Ok(Json(ApiResponse::error(format!("Invalid amount: {}", e)))),
    };

    // Check founder has enough QUSD to burn
    let founder_bytes: [u8; 32] = {
        let mut arr = [0u8; 32];
        if let Ok(b) = hex::decode(FOUNDER_WALLET) {
            if b.len() == 32 { arr.copy_from_slice(&b); }
        }
        arr
    };

    let current_balance = state.storage_engine.get_token_balance(&founder_bytes, &QUSD_TOKEN_ADDRESS).await.unwrap_or(0);
    if current_balance < amount_raw {
        return Ok(Json(ApiResponse::error(format!(
            "Insufficient QUSD balance: have {}, want to burn {}",
            format_qusd_amount(current_balance), format_qusd_amount(amount_raw)
        ))));
    }

    let new_balance = current_balance - amount_raw;
    let timestamp = chrono::Utc::now().timestamp();
    let memo = req.memo.unwrap_or_default();
    let tx_hash = generate_tx_hash("burn", timestamp, amount_raw);

    // Debit QUSD from founder
    if let Err(e) = state.storage_engine.save_token_balance(&founder_bytes, &QUSD_TOKEN_ADDRESS, new_balance).await {
        error!("💵 [QUSD] Failed to save burn balance: {}", e);
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    }

    // Update in-memory token balances
    {
        let mut token_balances = state.token_balances.write().await;
        let key = (founder_bytes, QUSD_TOKEN_ADDRESS);
        token_balances.insert(key, new_balance);
    }

    // Record audit entry
    let audit_entry = serde_json::json!({
        "action": "burn",
        "amount": format_qusd_amount(amount_raw),
        "amount_raw": amount_raw.to_string(),
        "wallet": FOUNDER_WALLET,
        "memo": memo,
        "timestamp": timestamp,
        "tx_hash": tx_hash,
    });

    if let Err(e) = state.storage_engine.save_qusd_audit_entry(&audit_entry).await {
        error!("💵 [QUSD] Failed to save burn audit entry: {}", e);
    }

    // Update in-memory state
    {
        let mut audit_log = state.qusd_audit_log.write().await;
        audit_log.push(audit_entry);
    }
    {
        let mut supply = state.qusd_total_supply.write().await;
        *supply = supply.saturating_sub(amount_raw);
    }

    let new_supply = *state.qusd_total_supply.read().await;

    info!(
        "🔥 [QUSD] BURNED {} QUSD (memo: {}, tx: {}, new supply: {})",
        format_qusd_amount(amount_raw), memo, tx_hash, format_qusd_amount(new_supply)
    );

    Ok(Json(ApiResponse::success(QusdBurnResponse {
        tx_hash,
        amount: format_qusd_amount(amount_raw),
        amount_raw: amount_raw.to_string(),
        new_total_supply: format_qusd_amount(new_supply),
        memo,
        timestamp,
    })))
}

/// POST /api/v1/qusd/transfer — Transfer QUSD between wallets (any holder)
pub async fn transfer_qusd(
    auth: AuthenticatedWallet,
    State(state): State<Arc<AppState>>,
    Json(req): Json<QusdTransferRequest>,
) -> Result<Json<ApiResponse<QusdTransferResponse>>, StatusCode> {
    let from_bytes = auth.address;
    let from_hex = hex::encode(from_bytes);

    // Parse amount
    let amount_raw = match parse_qusd_amount(&req.amount) {
        Ok(a) if a > 0 => a,
        Ok(_) => return Ok(Json(ApiResponse::error("Amount must be greater than 0".to_string()))),
        Err(e) => return Ok(Json(ApiResponse::error(format!("Invalid amount: {}", e)))),
    };

    // Parse recipient
    let to_bytes: [u8; 32] = match hex::decode(&req.to) {
        Ok(b) if b.len() == 32 => {
            let mut arr = [0u8; 32];
            arr.copy_from_slice(&b);
            arr
        }
        _ => return Ok(Json(ApiResponse::error("Invalid recipient address (must be 64 hex chars)".to_string()))),
    };

    if from_bytes == to_bytes {
        return Ok(Json(ApiResponse::error("Cannot transfer to self".to_string())));
    }

    // Check sender balance
    let sender_balance = state.storage_engine.get_token_balance(&from_bytes, &QUSD_TOKEN_ADDRESS).await.unwrap_or(0);
    if sender_balance < amount_raw {
        return Ok(Json(ApiResponse::error(format!(
            "Insufficient QUSD balance: have {}, want to send {}",
            format_qusd_amount(sender_balance), format_qusd_amount(amount_raw)
        ))));
    }

    // Debit sender, credit recipient
    let new_sender = sender_balance - amount_raw;
    let recipient_balance = state.storage_engine.get_token_balance(&to_bytes, &QUSD_TOKEN_ADDRESS).await.unwrap_or(0);
    let new_recipient = recipient_balance.saturating_add(amount_raw);

    if let Err(e) = state.storage_engine.save_token_balance(&from_bytes, &QUSD_TOKEN_ADDRESS, new_sender).await {
        error!("💵 [QUSD] Failed to save sender balance: {}", e);
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    }
    if let Err(e) = state.storage_engine.save_token_balance(&to_bytes, &QUSD_TOKEN_ADDRESS, new_recipient).await {
        error!("💵 [QUSD] Failed to save recipient balance: {}", e);
        // Attempt to rollback sender
        let _ = state.storage_engine.save_token_balance(&from_bytes, &QUSD_TOKEN_ADDRESS, sender_balance).await;
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    }

    // Update in-memory token balances
    {
        let mut token_balances = state.token_balances.write().await;
        token_balances.insert((from_bytes, QUSD_TOKEN_ADDRESS), new_sender);
        token_balances.insert((to_bytes, QUSD_TOKEN_ADDRESS), new_recipient);
    }

    let timestamp = chrono::Utc::now().timestamp();
    let tx_hash = generate_tx_hash("transfer", timestamp, amount_raw);

    info!(
        "💵 [QUSD] TRANSFER {} QUSD from {}.. to {}.. (tx: {})",
        format_qusd_amount(amount_raw), &from_hex[..16], &req.to[..req.to.len().min(16)], tx_hash
    );

    Ok(Json(ApiResponse::success(QusdTransferResponse {
        tx_hash,
        from: from_hex,
        to: req.to,
        amount: format_qusd_amount(amount_raw),
    })))
}

/// GET /api/v1/qusd/balance/:address — Check QUSD balance (public)
pub async fn get_qusd_balance(
    Path(address): Path<String>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<QusdBalanceResponse>>, StatusCode> {
    let addr_bytes: [u8; 32] = match hex::decode(&address) {
        Ok(b) if b.len() == 32 => {
            let mut arr = [0u8; 32];
            arr.copy_from_slice(&b);
            arr
        }
        _ => return Ok(Json(ApiResponse::error("Invalid address (must be 64 hex chars)".to_string()))),
    };

    let balance = state.storage_engine.get_token_balance(&addr_bytes, &QUSD_TOKEN_ADDRESS).await.unwrap_or(0);

    Ok(Json(ApiResponse::success(QusdBalanceResponse {
        address,
        balance: format_qusd_amount(balance),
        balance_raw: balance.to_string(),
    })))
}

/// GET /api/v1/qusd/supply — Total supply info (public)
pub async fn get_qusd_supply(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<QusdSupplyResponse>>, StatusCode> {
    let audit_log = state.qusd_audit_log.read().await;

    let mut total_minted: u128 = 0;
    let mut total_burned: u128 = 0;
    let mut mint_count: u64 = 0;
    let mut burn_count: u64 = 0;

    for entry in audit_log.iter() {
        let action = entry.get("action").and_then(|v| v.as_str()).unwrap_or("");
        let amount_str = entry.get("amount_raw").and_then(|v| v.as_str()).unwrap_or("0");
        let amount: u128 = amount_str.parse().unwrap_or(0);
        match action {
            "mint" => { total_minted = total_minted.saturating_add(amount); mint_count += 1; }
            "burn" => { total_burned = total_burned.saturating_add(amount); burn_count += 1; }
            _ => {}
        }
    }

    let total_supply = total_minted.saturating_sub(total_burned);

    Ok(Json(ApiResponse::success(QusdSupplyResponse {
        total_supply: format_qusd_amount(total_supply),
        total_supply_raw: total_supply.to_string(),
        total_minted: format_qusd_amount(total_minted),
        total_burned: format_qusd_amount(total_burned),
        mint_count,
        burn_count,
    })))
}

/// GET /api/v1/qusd/audit — Full transparent audit log (public)
pub async fn get_qusd_audit(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<Vec<QusdAuditEntry>>>, StatusCode> {
    let audit_log = state.qusd_audit_log.read().await;

    let entries: Vec<QusdAuditEntry> = audit_log.iter().map(|entry| {
        QusdAuditEntry {
            action: entry.get("action").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            amount: entry.get("amount").and_then(|v| v.as_str()).unwrap_or("0").to_string(),
            amount_raw: entry.get("amount_raw").and_then(|v| v.as_str()).unwrap_or("0").to_string(),
            wallet: entry.get("wallet").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            memo: entry.get("memo").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            timestamp: entry.get("timestamp").and_then(|v| v.as_i64()).unwrap_or(0),
            tx_hash: entry.get("tx_hash").and_then(|v| v.as_str()).unwrap_or("").to_string(),
        }
    }).collect();

    Ok(Json(ApiResponse::success(entries)))
}
