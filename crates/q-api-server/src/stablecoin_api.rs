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
use q_types::{ApiResponse, TokenInfo, TokenType, QUGUSD_TOKEN_ADDRESS, QUG_TOKEN_ADDRESS};
use q_vm::contracts::{CollateralVault, MintResult, PositionHealth, RedeemResult, VaultStats};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info, warn};

use crate::AppState;
use q_api_server::wallet_auth::AuthenticatedWallet;

/// Multi-token balance response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiTokenBalanceResponse {
    pub address: String,
    pub tokens: HashMap<String, TokenBalance>,  // Changed to HashMap for dynamic tokens
    pub total_usd_value: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenBalance {
    pub balance: String,          // Human-readable (e.g., "1234.56789012")
    #[serde(serialize_with = "q_types::u128_serde::serialize", deserialize_with = "q_types::u128_serde::deserialize")]
    pub balance_base_units: u128, // Raw base units
    pub usd_value: f64,           // USD value
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,            // Token name (for custom tokens)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contract_address: Option<String>,  // Contract address (for custom tokens)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decimals: Option<u8>,            // Decimals (for custom tokens)
}

/// Mint QUGUSD request
#[derive(Debug, Deserialize)]
pub struct MintQUGUSDRequest {
    pub qug_amount: String, // Human-readable QUG amount (e.g., "1000.0")
    #[serde(default)]
    pub slippage_tolerance: f64, // Default 0.01 (1%)
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
    pub qugusd_amount: String, // Human-readable QUGUSD amount
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
    pub health_status: String, // "healthy", "warning", "danger", "liquidatable"
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

    let mut tokens = HashMap::new();
    let mut total_usd_value = 0.0;

    // Get QUG balance from wallet_balances (native balance)
    let qug_balance = {
        let wallet_balances = state.wallet_balances.read().await;
        wallet_balances.get(&addr_bytes).copied().unwrap_or(0)
    };

    // Get QUGUSD balance from BOTH sources:
    // 1. CDP position (minted stablecoins via collateral)
    // 2. token_balances (received via swaps/transfers)
    let qugusd_balance = {
        // Source 1: Minted QUGUSD from collateral vault
        let minted_qugusd = {
            let vault = state.collateral_vault.read().await;
            vault.minted_qugusd.get(&addr_bytes).copied().unwrap_or(0)
        };

        // Source 2: QUGUSD from swaps/transfers stored in token_balances
        let swapped_qugusd = {
            let token_balances = state.token_balances.read().await;
            let balance_key = (addr_bytes, QUGUSD_TOKEN_ADDRESS);
            token_balances.get(&balance_key).copied().unwrap_or(0)
        };

        // Combine both sources
        let total = minted_qugusd as u128 + swapped_qugusd;
        if total > 0 {
            // v3.0.5: Use 1e24 for logging consistency
            info!(
                "💰 QUGUSD balance for {}: minted={}, swapped={}, total={}",
                &address_hex[..16],
                minted_qugusd as f64 / 1e24,
                swapped_qugusd as f64 / 1e24,
                total as f64 / 1e24
            );
        }
        total
    };

    // Get current QUG price from vault
    let qug_price_usd = state.collateral_vault.read().await.qug_price_usd;

    // v3.0.5-beta FIX: Use 1e24 divisor (not 1e8!) to match new decimal precision
    // QUG uses 24 decimal places: 1 QUG = 10^24 base units
    const QUG_DIVISOR: f64 = 1e24;

    // Calculate USD values for native tokens
    let qug_usd_value = (qug_balance as f64 / QUG_DIVISOR) * qug_price_usd;
    let qugusd_usd_value = qugusd_balance as f64 / QUG_DIVISOR; // QUGUSD is pegged to $1

    // Add QUG token
    tokens.insert(
        "QUG".to_string(),
        TokenBalance {
            balance: format!("{:.8}", qug_balance as f64 / QUG_DIVISOR),
            balance_base_units: qug_balance as u128,
            usd_value: qug_usd_value,
            name: Some("Quillon".to_string()),
            contract_address: Some(hex::encode(QUG_TOKEN_ADDRESS)),
            decimals: Some(24), // v3.0.5: Updated to 24 decimals
        },
    );
    total_usd_value += qug_usd_value;

    // Add QUGUSD token
    tokens.insert(
        "QUGUSD".to_string(),
        TokenBalance {
            balance: format!("{:.8}", qugusd_balance as f64 / QUG_DIVISOR),
            balance_base_units: qugusd_balance,
            usd_value: qugusd_usd_value,
            name: Some("Quillon USD".to_string()),
            contract_address: Some(hex::encode(QUGUSD_TOKEN_ADDRESS)),
            decimals: Some(24), // v3.0.5: Updated to 24 decimals
        },
    );
    total_usd_value += qugusd_usd_value;

    // ============================================
    // 🔧 v2.9.21-beta: CRITICAL FIX - Read token balances from RocksDB, not in-memory HashMap
    // ROOT CAUSE: In-memory HashMap can be stale after swap confirmation
    // FIX: Always read from RocksDB (source of truth) for API responses
    // ============================================
    let deployed_contracts = state.orobit_ecosystem.deployed_contracts.read().await;

    // Load token balances directly from RocksDB (guaranteed to be up-to-date)
    let rocksdb_balances = match state.storage_engine.load_token_balances().await {
        Ok(balances) => balances,
        Err(e) => {
            warn!("⚠️ [v2.9.21] Failed to load token balances from RocksDB: {}, falling back to in-memory", e);
            // Fallback to in-memory if RocksDB fails
            state.token_balances.read().await.clone()
        }
    };

    // v3.6.14: Count matches for debugging
    let mut matched_tokens = 0u32;
    let mut skipped_tokens = 0u32;

    for ((wallet_addr, token_addr), balance) in rocksdb_balances.iter() {
        // Only include tokens for this wallet
        if wallet_addr != &addr_bytes {
            skipped_tokens += 1;
            continue;
        }
        matched_tokens += 1;

        // Skip native tokens (already added above)
        if token_addr == &QUG_TOKEN_ADDRESS || token_addr == &QUGUSD_TOKEN_ADDRESS {
            continue;
        }

        // Look up contract metadata
        // Convert [u8; 32] to ContractAddress for lookup
        let contract_addr = q_vm::contracts::orobit_smart_contracts::ContractAddress(*token_addr);
        if let Some(contract_info) = deployed_contracts.get(&contract_addr) {
            let symbol = contract_info.metadata.symbol.clone().unwrap_or_else(|| "UNKNOWN".to_string());
            // Get decimals from deployment_params if available
            let decimals = contract_info.deployment_params
                .get("decimals")
                .and_then(|v| v.as_u64())
                .unwrap_or(8) as u8;

            // v3.9.5-beta: CORRUPTION DETECTION - Decimal-aware balance validation
            // Previously used a fixed 1e31 threshold which was WRONG for 24-decimal tokens,
            // causing legitimate balances to be permanently destroyed.
            // Fix: Max sane balance depends on token decimals:
            //   8 decimals:  max ~1e26 base units (10^18 tokens)
            //   24 decimals: max ~1e42 base units (10^18 tokens)
            // Use 10^(decimals + 18) as the ceiling (1 quintillion tokens max)
            let max_sane_balance: u128 = if decimals <= 8 {
                10u128.pow(26) // 1e26 for 8-decimal tokens
            } else if decimals <= 24 {
                // 10^(decimals + 18) but cap to avoid overflow
                10u128.saturating_pow((decimals as u32) + 18)
            } else {
                u128::MAX // Very high decimals - don't validate
            };
            // v4.3.0: Auto-repair balances inflated by swap bug.
            // Before v4.3.0, swap handler stored final_amount_out in 24-decimal format,
            // but minted balances use 2*decimals format. For 8-decimal tokens, swap amounts
            // are 10^8x too large. Detect and correct: if display value > 100 billion tokens,
            // the balance is almost certainly from the swap bug and should be divided by
            // 10^(24 - 2*decimals) to convert from 24-dec to 2*decimals format.
            let target_exp = 2u32 * decimals as u32;
            let final_balance = if decimals < 12 && target_exp < 24 {
                let correction_shift = 24 - target_exp;
                let correction_factor = 10u128.pow(correction_shift);
                let display_at_correct_format = *balance as f64 / 10f64.powi(target_exp as i32);
                // If display value > 100 billion, balance is in 24-decimal format (swap bug)
                if display_at_correct_format > 1e11 {
                    let corrected = *balance / correction_factor;
                    let corrected_display = corrected as f64 / 10f64.powi(target_exp as i32);
                    warn!(
                        "🔧 [v4.3.0] Auto-repairing swap-inflated balance for {} {}: {} → {} (display: {:.2} → {:.2}, shift={})",
                        &address_hex[..8], symbol, balance, corrected, display_at_correct_format, corrected_display, correction_shift
                    );
                    // Persist the corrected balance to both in-memory and RocksDB
                    let wa = *wallet_addr;
                    let ta = *token_addr;
                    {
                        let mut token_balances = state.token_balances.write().await;
                        token_balances.insert((wa, ta), corrected);
                    }
                    let _ = state.storage_engine.save_token_balance(&wa, &ta, corrected).await;
                    corrected
                } else {
                    *balance
                }
            } else if *balance > max_sane_balance {
                warn!(
                    "🚨 [v3.9.5] Suspicious balance for {} {}: {} base_units (exceeds max for {} decimals)",
                    &address_hex[..8], symbol, balance, decimals
                );
                *balance
            } else {
                *balance
            };

            // v4.1.0: Token balances are stored in 10^(2*decimals) format due to double-conversion
            // (frontend sends base_units = display*10^decimals, backend multiplies by 10^decimals again).
            // For 8-decimal tokens: stored as display * 10^16, so divide by 10^(2*decimals).
            // For 24-decimal tokens (QUG/QUGUSD): divide by 10^24 (handled separately above).
            let balance_divisor = if decimals < 24 {
                10f64.powi(target_exp as i32)
            } else {
                1e24
            };
            let balance_display = final_balance as f64 / balance_divisor;

            tokens.insert(
                symbol.clone(),
                TokenBalance {
                    balance: format!("{:.8}", balance_display),
                    balance_base_units: final_balance,
                    usd_value: 0.0, // Custom tokens don't have USD pricing yet
                    name: Some(contract_info.metadata.name.clone()),
                    // v2.4.2: Include qnk prefix so oracle price lookup matches pool addresses
                    contract_address: Some(format!("qnk{}", hex::encode(token_addr))),
                    decimals: Some(decimals), // v4.1.0: Use actual token decimals
                },
            );

            // v3.6.16: Log with corruption status
            if final_balance != *balance {
                info!(
                    "🪙 [v3.6.16] Custom token for {} (CORRECTED): {} = {:.8} (was {} base_units, now 0)",
                    &address_hex[..8], symbol, balance_display, balance
                );
            } else {
                info!(
                    "🪙 [v3.6.16] Custom token for {}: {} = {:.8} (balance_base_units={})",
                    &address_hex[..8], symbol, balance_display, balance
                );
            }
        } else if token_addr[0] == 0x1F {
            // v4.3.0: Index fund tokens (QNK10, DEFI5) use deterministic blake3 addresses
            // marked with first byte 0x1F. They won't match deployed_contracts.
            // Identify which fund by checking known addresses.
            let (fund_symbol, fund_name) = {
                let mut known_fund: Option<(&str, &str)> = None;
                for (sym, name) in &[("QNK10", "QNK Top 10 Index"), ("DEFI5", "DeFi Top 5 Index")] {
                    let mut hasher = blake3::Hasher::new();
                    hasher.update(b"QNK-INDEX-FUND:");
                    hasher.update(sym.as_bytes());
                    let hash = hasher.finalize();
                    let mut expected_addr = [0u8; 32];
                    expected_addr.copy_from_slice(hash.as_bytes());
                    expected_addr[0] = 0x1F;
                    if &expected_addr == token_addr {
                        known_fund = Some((sym, name));
                        break;
                    }
                }
                known_fund.unwrap_or(("INDEX", "Index Fund"))
            };

            // Index funds use 24 decimals (same as QUG/QUGUSD)
            let balance_display = *balance as f64 / 1e24;

            // v4.5.0: Calculate actual USD value based on NAV
            // QNK10 = 3x QUG price per share, DEFI5 = 2x QUG price per share
            let nav_multiplier = if fund_symbol == "QNK10" { 3.0 } else { 2.0 };
            let qug_price = {
                let vault = state.collateral_vault.read().await;
                vault.qug_price_usd
            };
            let nav_per_share = qug_price * nav_multiplier;
            let usd_value = balance_display * nav_per_share;

            info!(
                "🏦 [v4.5.0] Index fund for {}: {} = {:.4} shares @ ${:.2}/share = ${:.2} (QUG@${:.2}, {}x)",
                &address_hex[..8], fund_symbol, balance_display, nav_per_share, usd_value, qug_price, nav_multiplier
            );

            tokens.insert(
                fund_symbol.to_string(),
                TokenBalance {
                    balance: format!("{:.8}", balance_display),
                    balance_base_units: *balance,
                    usd_value,
                    name: Some(fund_name.to_string()),
                    contract_address: Some(format!("qnk{}", hex::encode(token_addr))),
                    decimals: Some(24),
                },
            );
        }
    }

    // v3.6.14: Log match stats
    if matched_tokens > 0 || skipped_tokens > 0 {
        info!(
            "📊 [v3.6.14] Wallet {}: {} tokens matched, {} from other wallets",
            &address_hex[..8],
            matched_tokens,
            skipped_tokens
        );
    }

    let response = MultiTokenBalanceResponse {
        address: address_hex.clone(),
        tokens,
        total_usd_value,
    };

    // v3.6.14: Count custom tokens (total minus QUG and QUGUSD)
    let custom_count = response.tokens.len().saturating_sub(2);

    // v3.6.14: INFO level for debugging - shows token count per wallet
    info!(
        "📊 [v3.6.14] Retrieved {} tokens for wallet {} ({} custom, {} total in RocksDB)",
        response.tokens.len(),
        &address_hex[..8],
        custom_count,
        rocksdb_balances.len()
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
    info!(
        "🏦 [AUTHENTICATED] Minting QUGUSD with {} QUG for wallet {}",
        request.qug_amount,
        hex::encode(&user_address[..8])
    );

    // Parse QUG amount
    // v3.0.4: Migrated to u128 with 24-decimal precision
    let qug_amount_f64: f64 = request
        .qug_amount
        .parse()
        .map_err(|_| StatusCode::BAD_REQUEST)?;
    let qug_amount_base_units: u128 = (qug_amount_f64 * 1e24) as u128;

    // Mint QUGUSD
    let mut vault_write = state.collateral_vault.write().await;
    let mint_result = match vault_write.mint_qugusd(user_address, qug_amount_base_units) {
        Ok(result) => result,
        Err(e) => {
            warn!("❌ Mint failed: {}", e);
            return Ok(Json(ApiResponse::error(format!("Mint failed: {}", e))));
        }
    };

    // v3.0.4: Use 24-decimal precision (1e24)
    let response = MintQUGUSDResponse {
        qug_locked: format!("{:.8}", mint_result.qug_locked as f64 / 1e24),
        qugusd_minted: format!("{:.8}", mint_result.qugusd_minted as f64 / 1e24),
        collateral_ratio: mint_result.collateral_ratio,
        liquidation_price: mint_result.liquidation_price,
    };

    info!(
        "✅ Minted {:.4} QUGUSD (locked {:.4} QUG)",
        mint_result.qugusd_minted as f64 / 1e24,
        mint_result.qug_locked as f64 / 1e24
    );

    // v3.6.11-beta: CRITICAL FIX - Persist CollateralVault to storage after minting
    // Without this, minted QUGUSD balances are lost on restart!
    let vault_clone = vault_write.clone();
    drop(vault_write); // Release the write lock before async persist

    if let Ok(vault_bytes) = bincode::serialize(&vault_clone) {
        if let Err(e) = state.storage_engine.save_collateral_vault_data(&vault_bytes).await {
            warn!("⚠️ Failed to persist CollateralVault after mint: {}", e);
        } else {
            info!("💾 CollateralVault persisted after mint (minted_qugusd={})", vault_clone.total_qugusd_minted);
        }
    }

    Ok(Json(ApiResponse::success(response)))
}

/// POST /api/v1/stablecoin/redeem - Redeem QUG by burning QUGUSD (AUTHENTICATED)
pub async fn redeem_qug(
    auth: AuthenticatedWallet,
    State(state): State<Arc<AppState>>,
    Json(request): Json<RedeemQUGRequest>,
) -> Result<Json<ApiResponse<RedeemQUGResponse>>, StatusCode> {
    let user_address = auth.address;
    info!(
        "🔓 [AUTHENTICATED] Redeeming {} QUGUSD for QUG for wallet {}",
        request.qugusd_amount,
        hex::encode(&user_address[..8])
    );

    // Parse QUGUSD amount
    // v3.0.4: Migrated to u128 with 24-decimal precision
    let qugusd_amount_f64: f64 = request
        .qugusd_amount
        .parse()
        .map_err(|_| StatusCode::BAD_REQUEST)?;
    let qugusd_amount_base_units: u128 = (qugusd_amount_f64 * 1e24) as u128;

    // Redeem QUG
    let mut vault_write = state.collateral_vault.write().await;
    let redeem_result = match vault_write.redeem_qug(user_address, qugusd_amount_base_units) {
        Ok(result) => result,
        Err(e) => {
            warn!("❌ Redeem failed: {}", e);
            return Ok(Json(ApiResponse::error(format!("Redeem failed: {}", e))));
        }
    };

    // v3.0.4: Use 24-decimal precision (1e24)
    let response = RedeemQUGResponse {
        qugusd_burned: format!("{:.8}", redeem_result.qugusd_burned as f64 / 1e24),
        qug_unlocked: format!("{:.8}", redeem_result.qug_unlocked as f64 / 1e24),
        remaining_collateral_ratio: redeem_result.remaining_collateral_ratio,
    };

    info!(
        "✅ Redeemed {:.4} QUG (burned {:.4} QUGUSD)",
        redeem_result.qug_unlocked as f64 / 1e24,
        redeem_result.qugusd_burned as f64 / 1e24
    );

    // v3.6.11-beta: CRITICAL FIX - Persist CollateralVault to storage after redeem
    let vault_clone = vault_write.clone();
    drop(vault_write); // Release the write lock before async persist

    if let Ok(vault_bytes) = bincode::serialize(&vault_clone) {
        if let Err(e) = state.storage_engine.save_collateral_vault_data(&vault_bytes).await {
            warn!("⚠️ Failed to persist CollateralVault after redeem: {}", e);
        } else {
            info!("💾 CollateralVault persisted after redeem (minted_qugusd={})", vault_clone.total_qugusd_minted);
        }
    }

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
                "Invalid address format".to_string(),
            )));
        }
    };

    let vault_read = state.collateral_vault.read().await;

    // Get position data
    let qug_locked = vault_read.locked_qug.get(&addr_bytes).copied().unwrap_or(0);
    let qugusd_minted = vault_read
        .minted_qugusd
        .get(&addr_bytes)
        .copied()
        .unwrap_or(0);

    let collateral_ratio = vault_read.get_collateral_ratio(&addr_bytes).unwrap_or(0.0);

    let health_status = vault_read
        .get_position_health(&addr_bytes)
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
    let liquidatable_addrs: Vec<String> =
        liquidatable.iter().map(|addr| hex::encode(addr)).collect();

    info!(
        "⚡ Found {} liquidatable positions",
        liquidatable_addrs.len()
    );

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
                "Invalid liquidated address format".to_string(),
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
            return Ok(Json(ApiResponse::error(format!(
                "Liquidation failed: {}",
                e
            ))));
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

    // v3.6.11-beta: CRITICAL FIX - Persist CollateralVault to storage after liquidation
    let vault_clone = vault_write.clone();
    drop(vault_write); // Release the write lock before async persist

    if let Ok(vault_bytes) = bincode::serialize(&vault_clone) {
        if let Err(e) = state.storage_engine.save_collateral_vault_data(&vault_bytes).await {
            warn!("⚠️ Failed to persist CollateralVault after liquidation: {}", e);
        } else {
            info!("💾 CollateralVault persisted after liquidation");
        }
    }

    Ok(Json(ApiResponse::success(response)))
}
