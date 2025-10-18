/// Stablecoin API - QUG/QUGUSD Dual-Token Endpoints
///
/// This module provides API endpoints for:
/// - Multi-token balance queries
/// - QUGUSD minting (lock QUG as collateral)
/// - QUG redemption (burn QUGUSD to unlock)
/// - Position health monitoring
/// - Liquidation interface
/// - Fee statistics

use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use q_types::{ApiResponse, TokenType, TokenInfo, QUG_TOKEN_ADDRESS, QUGUSD_TOKEN_ADDRESS};
use q_vm::contracts::{CollateralVault, MintResult, RedeemResult, PositionHealth, VaultStats};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, info, warn};

use crate::AppState;
use q_api_server::wallet_auth::AuthenticatedWallet;

/// Multi-token balance response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiTokenBalanceResponse {
    pub address: String,
    pub tokens: TokenBalances,
    pub total_usd_value: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenBalances {
    #[serde(rename = "QUG")]
    pub qug: TokenBalance,
    #[serde(rename = "QUGUSD")]
    pub qugusd: TokenBalance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenBalance {
    pub balance: String,              // Human-readable (e.g., "1234.56789012")
    pub balance_base_units: u64,      // Raw base units
    pub usd_value: f64,                // USD value
}

/// Mint QUGUSD request
#[derive(Debug, Deserialize)]
pub struct MintQUGUSDRequest {
    pub qug_amount: String,           // Human-readable QUG amount (e.g., "1000.0")
    #[serde(default)]
    pub slippage_tolerance: f64,      // Default 0.01 (1%)
}

/// Mint QUGUSD response
#[derive(Debug, Serialize)]
pub struct MintQUGUSDResponse {
    pub qug_locked: String,
    pub qugusd_minted: String,
    pub collateral_ratio: f64,
    pub liquidation_price: f64,
}

/// Redeem QUG request
#[derive(Debug, Deserialize)]
pub struct RedeemQUGRequest {
    pub qugusd_amount: String,         // Human-readable QUGUSD amount
}

/// Redeem QUG response
#[derive(Debug, Serialize)]
pub struct RedeemQUGResponse {
    pub qugusd_burned: String,
    pub qug_unlocked: String,
    pub remaining_collateral_ratio: f64,
}

/// Position health response
#[derive(Debug, Serialize)]
pub struct PositionHealthResponse {
    pub address: String,
    pub qug_locked: String,
    pub qugusd_minted: String,
    pub collateral_ratio: f64,
    pub health_status: String,          // "healthy", "warning", "danger", "liquidatable"
    pub liquidation_price: f64,
    pub qug_price_current: f64,
}

/// Liquidation request
#[derive(Debug, Deserialize)]
pub struct LiquidateRequest {
    pub liquidated_address: String,
}

/// Liquidation response
#[derive(Debug, Serialize)]
pub struct LiquidateResponse {
    pub liquidator: String,
    pub liquidated_user: String,
    pub qug_seized: String,
    pub qugusd_burned: String,
    pub liquidator_bonus: String,
}

/// Fee statistics response
#[derive(Debug, Serialize)]
pub struct FeeStatsResponse {
    pub last_24h: FeeStats24h,
    pub all_time: FeeStatsAllTime,
}

#[derive(Debug, Serialize)]
pub struct FeeStats24h {
    pub total_fees_qugusd: String,
    pub bank_share: String,
    pub qug_buyback_amount: String,
    pub miner_distribution: String,
    pub qug_burned: String,
}

#[derive(Debug, Serialize)]
pub struct FeeStatsAllTime {
    pub total_fees_collected: String,
    pub total_qug_burned: String,
}

/// GET /api/v1/wallet/tokens - Get multi-token balances (AUTHENTICATED)
///
/// This endpoint requires wallet authentication via X-Wallet-Auth header.
/// The wallet address is extracted from the signed request for privacy.
pub async fn get_multi_token_balance(
    auth: AuthenticatedWallet,
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<MultiTokenBalanceResponse>>, StatusCode> {
    let addr_bytes = auth.address;
    let address_hex = hex::encode(addr_bytes);

    debug!("📊 [AUTHENTICATED] Getting multi-token balance for wallet");
    info!("✅ Wallet authenticated successfully via {:?}", auth.scheme);

    // Get QUG balance from wallet_balances (native balance)
    let qug_balance = {
        let wallet_balances = state.wallet_balances.read().await;
        wallet_balances.get(&addr_bytes).copied().unwrap_or(0)
    };

    // Get QUGUSD balance from token_balances map
    let qugusd_balance = {
        let token_balances = state.token_balances.read().await;
        token_balances.get(&(addr_bytes, QUGUSD_TOKEN_ADDRESS)).copied().unwrap_or(0)
    };

    // Get current QUG price from vault
    let qug_price_usd = state.collateral_vault.read().await.qug_price_usd;

    // Calculate USD values
    let qug_usd_value = (qug_balance as f64 / 1e8) * qug_price_usd;
    let qugusd_usd_value = qugusd_balance as f64 / 1e8; // QUGUSD is pegged to $1

    let response = MultiTokenBalanceResponse {
        address: address_hex,
        tokens: TokenBalances {
            qug: TokenBalance {
                balance: format!("{:.8}", qug_balance as f64 / 1e8),
                balance_base_units: qug_balance,
                usd_value: qug_usd_value,
            },
            qugusd: TokenBalance {
                balance: format!("{:.8}", qugusd_balance as f64 / 1e8),
                balance_base_units: qugusd_balance,
                usd_value: qugusd_usd_value,
            },
        },
        total_usd_value: qug_usd_value + qugusd_usd_value,
    };

    info!(
        "✅ Retrieved balances: QUG={:.4}, QUGUSD={:.4}",
        qug_balance as f64 / 1e8,
        qugusd_balance as f64 / 1e8
    );

    Ok(Json(ApiResponse::success(response)))
}

/// POST /api/v1/stablecoin/mint - Mint QUGUSD by locking QUG (AUTHENTICATED)
pub async fn mint_qugusd(
    auth: AuthenticatedWallet,
    State(state): State<Arc<AppState>>,
    Json(request): Json<MintQUGUSDRequest>,
) -> Result<Json<ApiResponse<MintQUGUSDResponse>>, StatusCode> {
    let user_address = auth.address;
    info!("🏦 [AUTHENTICATED] Minting QUGUSD with {} QUG for wallet {}",
        request.qug_amount, hex::encode(&user_address[..8]));

    // Parse QUG amount
    let qug_amount_f64: f64 = request.qug_amount.parse()
        .map_err(|_| StatusCode::BAD_REQUEST)?;
    let qug_amount_base_units = (qug_amount_f64 * 1e8) as u64;

    // Mint QUGUSD
    let mut vault_write = state.collateral_vault.write().await;
    let mint_result = match vault_write.mint_qugusd(user_address, qug_amount_base_units) {
        Ok(result) => result,
        Err(e) => {
            warn!("❌ Mint failed: {}", e);
            return Ok(Json(ApiResponse::error(format!("Mint failed: {}", e))));
        }
    };

    let response = MintQUGUSDResponse {
        qug_locked: format!("{:.8}", mint_result.qug_locked as f64 / 1e8),
        qugusd_minted: format!("{:.8}", mint_result.qugusd_minted as f64 / 1e8),
        collateral_ratio: mint_result.collateral_ratio,
        liquidation_price: mint_result.liquidation_price,
    };

    info!(
        "✅ Minted {:.4} QUGUSD (locked {:.4} QUG)",
        mint_result.qugusd_minted as f64 / 1e8,
        mint_result.qug_locked as f64 / 1e8
    );

    Ok(Json(ApiResponse::success(response)))
}

/// POST /api/v1/stablecoin/redeem - Redeem QUG by burning QUGUSD (AUTHENTICATED)
pub async fn redeem_qug(
    auth: AuthenticatedWallet,
    State(state): State<Arc<AppState>>,
    Json(request): Json<RedeemQUGRequest>,
) -> Result<Json<ApiResponse<RedeemQUGResponse>>, StatusCode> {
    let user_address = auth.address;
    info!("🔓 [AUTHENTICATED] Redeeming {} QUGUSD for QUG for wallet {}",
        request.qugusd_amount, hex::encode(&user_address[..8]));

    // Parse QUGUSD amount
    let qugusd_amount_f64: f64 = request.qugusd_amount.parse()
        .map_err(|_| StatusCode::BAD_REQUEST)?;
    let qugusd_amount_base_units = (qugusd_amount_f64 * 1e8) as u64;

    // Redeem QUG
    let mut vault_write = state.collateral_vault.write().await;
    let redeem_result = match vault_write.redeem_qug(user_address, qugusd_amount_base_units) {
        Ok(result) => result,
        Err(e) => {
            warn!("❌ Redeem failed: {}", e);
            return Ok(Json(ApiResponse::error(format!("Redeem failed: {}", e))));
        }
    };

    let response = RedeemQUGResponse {
        qugusd_burned: format!("{:.8}", redeem_result.qugusd_burned as f64 / 1e8),
        qug_unlocked: format!("{:.8}", redeem_result.qug_unlocked as f64 / 1e8),
        remaining_collateral_ratio: redeem_result.remaining_collateral_ratio,
    };

    info!(
        "✅ Redeemed {:.4} QUG (burned {:.4} QUGUSD)",
        redeem_result.qug_unlocked as f64 / 1e8,
        redeem_result.qugusd_burned as f64 / 1e8
    );

    Ok(Json(ApiResponse::success(response)))
}

/// GET /api/v1/stablecoin/position/{address} - Get position health
pub async fn get_position_health(
    State(state): State<Arc<AppState>>,
    Path(address): Path<String>,
) -> Result<Json<ApiResponse<PositionHealthResponse>>, StatusCode> {
    debug!("🔍 Getting position health for: {}", address);

    // Parse address
    let addr_bytes = match hex::decode(&address) {
        Ok(bytes) if bytes.len() == 32 => {
            let mut arr = [0u8; 32];
            arr.copy_from_slice(&bytes);
            arr
        }
        _ => {
            return Ok(Json(ApiResponse::error(
                "Invalid address format".to_string()
            )));
        }
    };

    let vault_read = state.collateral_vault.read().await;

    // Get position data
    let qug_locked = vault_read.locked_qug.get(&addr_bytes).copied().unwrap_or(0);
    let qugusd_minted = vault_read.minted_qugusd.get(&addr_bytes).copied().unwrap_or(0);

    let collateral_ratio = vault_read.get_collateral_ratio(&addr_bytes)
        .unwrap_or(0.0);

    let health_status = vault_read.get_position_health(&addr_bytes)
        .unwrap_or(PositionHealth::Healthy);

    // Calculate liquidation price
    let liquidation_price = if qugusd_minted > 0 {
        let qugusd_value = qugusd_minted as f64 / 1e8;
        let qug_amount = qug_locked as f64 / 1e8;
        (qugusd_value * 1.10) / qug_amount // 110% ratio
    } else {
        0.0
    };

    let health_str = match health_status {
        PositionHealth::Healthy => "healthy",
        PositionHealth::Warning => "warning",
        PositionHealth::Danger => "danger",
        PositionHealth::Liquidatable => "liquidatable",
    };

    let response = PositionHealthResponse {
        address: address.clone(),
        qug_locked: format!("{:.8}", qug_locked as f64 / 1e8),
        qugusd_minted: format!("{:.8}", qugusd_minted as f64 / 1e8),
        collateral_ratio,
        health_status: health_str.to_string(),
        liquidation_price,
        qug_price_current: vault_read.qug_price_usd,
    };

    Ok(Json(ApiResponse::success(response)))
}

/// GET /api/v1/stablecoin/vault/stats - Get vault statistics
pub async fn get_vault_stats(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<VaultStats>>, StatusCode> {
    debug!("📊 Getting vault statistics");

    let vault_read = state.collateral_vault.read().await;
    let stats = vault_read.get_vault_stats();

    info!(
        "✅ Vault stats: {:.2} QUG locked, {:.2} QUGUSD minted, ratio={:.2}%",
        stats.total_qug_locked as f64 / 1e8,
        stats.total_qugusd_minted as f64 / 1e8,
        stats.global_collateral_ratio * 100.0
    );

    Ok(Json(ApiResponse::success(stats)))
}

/// GET /api/v1/stats/fees - Get fee distribution statistics
pub async fn get_fee_stats(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<FeeStatsResponse>>, StatusCode> {
    debug!("📊 Getting fee statistics");

    // TODO: Implement actual fee tracking
    // For now, return placeholder data

    let response = FeeStatsResponse {
        last_24h: FeeStats24h {
            total_fees_qugusd: "1234.56".to_string(),
            bank_share: "493.82".to_string(),
            qug_buyback_amount: "370.37".to_string(),
            miner_distribution: "370.37".to_string(),
            qug_burned: "37.04".to_string(),
        },
        all_time: FeeStatsAllTime {
            total_fees_collected: "1234567.89".to_string(),
            total_qug_burned: "12345.67".to_string(),
        },
    };

    Ok(Json(ApiResponse::success(response)))
}

/// GET /api/v1/stablecoin/liquidatable - Get liquidatable positions
pub async fn get_liquidatable_positions(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<Vec<String>>>, StatusCode> {
    debug!("🔍 Getting liquidatable positions");

    let vault_read = state.collateral_vault.read().await;
    let liquidatable = vault_read.get_liquidatable_positions();

    // Convert to hex strings
    let liquidatable_addrs: Vec<String> = liquidatable
        .iter()
        .map(|addr| hex::encode(addr))
        .collect();

    info!("⚡ Found {} liquidatable positions", liquidatable_addrs.len());

    Ok(Json(ApiResponse::success(liquidatable_addrs)))
}

/// POST /api/v1/stablecoin/liquidate - Liquidate undercollateralized position
pub async fn liquidate_position(
    State(state): State<Arc<AppState>>,
    Json(request): Json<LiquidateRequest>,
) -> Result<Json<ApiResponse<LiquidateResponse>>, StatusCode> {
    info!("⚡ Liquidating position: {}", request.liquidated_address);

    // Parse addresses
    let liquidated_bytes = match hex::decode(&request.liquidated_address) {
        Ok(bytes) if bytes.len() == 32 => {
            let mut arr = [0u8; 32];
            arr.copy_from_slice(&bytes);
            arr
        }
        _ => {
            return Ok(Json(ApiResponse::error(
                "Invalid liquidated address format".to_string()
            )));
        }
    };

    // TODO: Get liquidator address from authentication
    let liquidator_address = [1u8; 32]; // Placeholder

    // Perform liquidation
    let mut vault_write = state.collateral_vault.write().await;
    let liq_result = match vault_write.liquidate(liquidator_address, liquidated_bytes) {
        Ok(result) => result,
        Err(e) => {
            warn!("❌ Liquidation failed: {}", e);
            return Ok(Json(ApiResponse::error(format!("Liquidation failed: {}", e))));
        }
    };

    let response = LiquidateResponse {
        liquidator: hex::encode(liq_result.liquidator),
        liquidated_user: hex::encode(liq_result.liquidated_user),
        qug_seized: format!("{:.8}", liq_result.qug_seized as f64 / 1e8),
        qugusd_burned: format!("{:.8}", liq_result.qugusd_burned as f64 / 1e8),
        liquidator_bonus: format!("{:.8}", liq_result.liquidator_bonus as f64 / 1e8),
    };

    info!(
        "✅ Liquidated position: seized {:.4} QUG",
        liq_result.qug_seized as f64 / 1e8
    );

    Ok(Json(ApiResponse::success(response)))
}
