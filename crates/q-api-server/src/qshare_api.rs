//! QSHARE-1 REST handlers — `/api/v1/qshare/*`.
//!
//! Per `docs/standards/qshare-treasury-protocol-spec.md` §5. Four endpoints:
//!
//!   GET  /api/v1/qshare/state                — read NAV, treasury composition, lifetime stats
//!   GET  /api/v1/qshare/premium              — read premium ratio + eligibility window
//!   POST /api/v1/qshare/try_mint_signed      — auth-gated mint trigger
//!   POST /api/v1/qshare/try_buyback_signed   — auth-gated buyback trigger
//!
//! The mint/buyback handlers call into the in-memory `QShareContract` held in
//! `AppState`. AMM pool depth + TWAP are read from `state.liquidity_pools` when
//! a QSHARE/QUG pool exists; if no pool is seeded, mint/buyback return the
//! "POOL_NOT_SEEDED" error per spec §2.4.
//!
//! Status v10.10.7:
//!   ✅ GET /state    — live; reads QShareContract under read lock
//!   ✅ GET /premium  — live; falls back to "not observable" when no pool
//!   ✅ POST /try_mint_signed — live; calls QShareContract::try_autonomous_mint
//!   ✅ POST /try_buyback_signed — live; calls QShareContract::try_buyback
//!
//! Persistence to CF_CONTRACTS is v10.10.8 follow-up.

use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::AppState;
use crate::wallet_auth::AuthenticatedWallet;
use q_vm::contracts::qshare_token::{
    DexPoolSnapshot, MintError, BuybackError, QShareContract,
};

// ============ READ RESPONSES ============

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QShareStateResponse {
    pub nav_per_qshare_raw: String,
    pub total_treasury_qug_raw: String,
    pub treasury_pending_qug_raw: String,
    pub circulating_qshare_raw: String,
    pub last_mint_height: u64,
    pub last_buyback_height: u64,
    pub mint_cooldown_blocks: u64,
    pub buyback_cooldown_blocks: u64,
    pub next_mint_eligible_at_height: u64,
    pub next_buyback_eligible_at_height: u64,
    pub lifetime_mints: u64,
    pub lifetime_buybacks: u64,
    pub lifetime_qug_accumulated_raw: String,
    pub lifetime_qshare_burned_raw: String,
    pub computed_at_timestamp: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QSharePremiumResponse {
    pub nav_per_qshare_raw: String,
    pub market_price_twap_qug_per_qshare_raw: Option<String>,
    pub premium_ratio_bps: Option<u64>,
    pub zone: String, // "mint" | "buyback" | "neutral" | "no_pool"
    pub pool_qug_reserves_raw: Option<String>,
    pub pool_qshare_reserves_raw: Option<String>,
    pub pool_sufficiently_deep: bool,
    pub next_mint_eligible_at_height: u64,
    pub next_buyback_eligible_at_height: u64,
}

// ============ ERROR ENVELOPE ============

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QShareApiError {
    pub code: String,
    pub message: String,
}

fn error_response(status: StatusCode, code: &str, message: impl Into<String>) -> axum::response::Response {
    let body = serde_json::json!({
        "error": QShareApiError { code: code.to_string(), message: message.into() }
    });
    (status, axum::Json(body)).into_response()
}

// ============ DEX POOL SNAPSHOT BUILDER ============

/// Read the QSHARE/QUG liquidity pool from AppState and build a DexPoolSnapshot.
/// Returns None if no pool exists.
async fn read_qshare_pool(state: &AppState) -> Option<DexPoolSnapshot> {
    let pools = state.liquidity_pools.read().await;
    // Pool keys aren't stable across versions; look for any pool whose tokens
    // include QSHARE_TOKEN_ADDRESS paired with QUG (zero address or QUG sentinel).
    let qshare_addr = q_types::QSHARE_TOKEN_ADDRESS;
    let qug_addr = q_types::QUG_TOKEN_ADDRESS;
    for (_pool_id, pool) in pools.iter() {
        let (token0, token1) = (pool.token0, pool.token1);
        if (token0 == qshare_addr && token1 == qug_addr) || (token0 == qug_addr && token1 == qshare_addr) {
            // Reserves are ordered (token0, token1). Map to (qug_reserves, qshare_reserves).
            let (qug_r, qshare_r) = if token0 == qug_addr {
                (pool.reserve0, pool.reserve1)
            } else {
                (pool.reserve1, pool.reserve0)
            };
            // TWAP — for v10.10.7 we use spot ratio as a stand-in. Real
            // TWAP needs the price-history oracle; v10.10.8 follow-up.
            let twap = if qshare_r > 0 {
                // qug_per_qshare = qug_reserves × 10^24 / qshare_reserves
                qug_r
                    .checked_mul(10u128.pow(24))
                    .and_then(|x| x.checked_div(qshare_r))
                    .unwrap_or(0)
            } else {
                0
            };
            return Some(DexPoolSnapshot {
                qug_reserves: qug_r,
                qshare_reserves: qshare_r,
                twap_qug_per_qshare: twap,
            });
        }
    }
    None
}

// ============ HANDLERS ============

/// GET /api/v1/qshare/state — public read of treasury composition + NAV.
pub async fn get_state(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let now_ts = chrono::Utc::now().timestamp();
    let contract = state.qshare_contract.read().await;
    let total = contract.nav_total_qug_equivalent(now_ts as u64);
    let nav = contract.nav_per_qshare(now_ts as u64);
    let resp = QShareStateResponse {
        nav_per_qshare_raw: nav.to_string(),
        total_treasury_qug_raw: total.to_string(),
        treasury_pending_qug_raw: contract.treasury_pending_qug.to_string(),
        circulating_qshare_raw: contract.circulating_qshare.to_string(),
        last_mint_height: contract.last_mint_height,
        last_buyback_height: contract.last_buyback_height,
        mint_cooldown_blocks: contract.mint_cooldown_blocks,
        buyback_cooldown_blocks: contract.buyback_cooldown_blocks,
        next_mint_eligible_at_height: contract.next_mint_eligible_at_height(),
        next_buyback_eligible_at_height: contract.next_buyback_eligible_at_height(),
        lifetime_mints: contract.lifetime_mints,
        lifetime_buybacks: contract.lifetime_buybacks,
        lifetime_qug_accumulated_raw: contract.lifetime_qug_accumulated.to_string(),
        lifetime_qshare_burned_raw: contract.lifetime_qshare_burned.to_string(),
        computed_at_timestamp: now_ts,
    };
    (StatusCode::OK, axum::Json(resp)).into_response()
}

/// GET /api/v1/qshare/premium — public read of premium ratio + eligibility.
pub async fn get_premium(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let now_ts = chrono::Utc::now().timestamp();
    let contract = state.qshare_contract.read().await;
    let nav = contract.nav_per_qshare(now_ts as u64);

    let pool = read_qshare_pool(&state).await;
    let (premium_bps, zone, pool_deep, pool_qug, pool_qshare, market_price) = match &pool {
        None => (None, "no_pool".to_string(), false, None, None, None),
        Some(p) => {
            let bps = contract.premium_ratio_bps(p, now_ts as u64);
            let zone = match bps {
                Some(b) if b >= 1500 => "mint",
                Some(b) if b <= 950 => "buyback",
                Some(_) => "neutral",
                None => "no_pool",
            }.to_string();
            (
                bps,
                zone,
                p.is_sufficiently_deep(),
                Some(p.qug_reserves.to_string()),
                Some(p.qshare_reserves.to_string()),
                Some(p.twap_qug_per_qshare.to_string()),
            )
        }
    };

    let resp = QSharePremiumResponse {
        nav_per_qshare_raw: nav.to_string(),
        market_price_twap_qug_per_qshare_raw: market_price,
        premium_ratio_bps: premium_bps,
        zone,
        pool_qug_reserves_raw: pool_qug,
        pool_qshare_reserves_raw: pool_qshare,
        pool_sufficiently_deep: pool_deep,
        next_mint_eligible_at_height: contract.next_mint_eligible_at_height(),
        next_buyback_eligible_at_height: contract.next_buyback_eligible_at_height(),
    };
    (StatusCode::OK, axum::Json(resp)).into_response()
}

/// POST /api/v1/qshare/try_mint_signed — auth-gated permissionless mint trigger.
pub async fn try_mint_signed(
    State(state): State<Arc<AppState>>,
    auth: AuthenticatedWallet,
) -> impl IntoResponse {
    let now_ts = chrono::Utc::now().timestamp() as u64;
    // Block height for cooldown gating — read from node status.
    let current_height = state.node_status.read().await.current_height;

    let pool = match read_qshare_pool(&state).await {
        Some(p) => p,
        None => {
            return error_response(
                StatusCode::PRECONDITION_FAILED,
                "POOL_NOT_SEEDED",
                "QSHARE/QUG AMM pool does not exist yet — seed liquidity before mint can trigger",
            );
        }
    };

    // Trigger fee = MINT_TRIGGER_FEE_QUG (0.01 QUG). For v10.10.7 we treat
    // the call itself as the fee proof; v10.10.8 will enforce actual fee
    // transfer + bounty payout via the standard Transaction path.
    let trigger_fee_paid = 10u128.pow(22); // 0.01 QUG

    let mut contract = state.qshare_contract.write().await;
    match contract.try_autonomous_mint(auth.address, trigger_fee_paid, &pool, current_height, now_ts) {
        Ok(result) => {
            let body = serde_json::json!({
                "minted_qshare_raw": result.event.minted_qshare.to_string(),
                "qug_accumulated_raw": result.event.qug_accumulated.to_string(),
                "new_nav_per_qshare_raw": result.event.new_nav_per_qshare.to_string(),
                "premium_ratio_bps_at_trigger": result.event.premium_ratio_bps,
                "bounty_paid_qug_raw": result.bounty_paid_to_caller.to_string(),
                "new_circulating_qshare_raw": result.new_circulating_qshare.to_string(),
                "block_height": result.event.block_height,
            });
            (StatusCode::OK, axum::Json(body)).into_response()
        }
        Err(e) => mint_error_response(&e),
    }
}

/// POST /api/v1/qshare/try_buyback_signed — auth-gated permissionless buyback.
pub async fn try_buyback_signed(
    State(state): State<Arc<AppState>>,
    auth: AuthenticatedWallet,
) -> impl IntoResponse {
    let now_ts = chrono::Utc::now().timestamp() as u64;
    let current_height = state.node_status.read().await.current_height;

    let pool = match read_qshare_pool(&state).await {
        Some(p) => p,
        None => {
            return error_response(
                StatusCode::PRECONDITION_FAILED,
                "POOL_NOT_SEEDED",
                "QSHARE/QUG AMM pool does not exist yet — seed liquidity before buyback can trigger",
            );
        }
    };

    let trigger_fee_paid = 10u128.pow(22);

    let mut contract = state.qshare_contract.write().await;
    match contract.try_buyback(auth.address, trigger_fee_paid, &pool, current_height, now_ts) {
        Ok(result) => {
            let body = serde_json::json!({
                "burned_qshare_raw": result.event.burned_qshare.to_string(),
                "qug_spent_raw": result.event.qug_spent.to_string(),
                "new_nav_per_qshare_raw": result.event.new_nav_per_qshare.to_string(),
                "discount_ratio_bps_at_trigger": result.event.discount_ratio_bps,
                "bounty_paid_qug_raw": result.bounty_paid_to_caller.to_string(),
                "new_circulating_qshare_raw": result.new_circulating_qshare.to_string(),
                "block_height": result.event.block_height,
            });
            (StatusCode::OK, axum::Json(body)).into_response()
        }
        Err(e) => buyback_error_response(&e),
    }
}

// ============ ERROR MAPPING ============

fn mint_error_response(err: &MintError) -> axum::response::Response {
    use MintError::*;
    let (status, code, msg) = match err {
        PremiumBelowThreshold { current_bps, required_bps } => (
            StatusCode::PRECONDITION_FAILED,
            "PREMIUM_BELOW_THRESHOLD",
            format!("premium {} bps < required {} bps (×1000)", current_bps, required_bps),
        ),
        CooldownActive { current_height, eligible_at } => (
            StatusCode::TOO_EARLY,
            "COOLDOWN_ACTIVE",
            format!("at height {}, eligible at {}", current_height, eligible_at),
        ),
        PoolTooShallow { current_qug_reserves, required } => (
            StatusCode::PRECONDITION_FAILED,
            "POOL_TOO_SHALLOW",
            format!("pool QUG reserves {} < required {}", current_qug_reserves, required),
        ),
        TriggerFeeInsufficient { paid, required } => (
            StatusCode::PAYMENT_REQUIRED,
            "TRIGGER_FEE_INSUFFICIENT",
            format!("paid {} raw < required {} raw", paid, required),
        ),
        NavOracleStale => (
            StatusCode::SERVICE_UNAVAILABLE,
            "NAV_ORACLE_STALE",
            "NAV oracle is stale or returned None — cannot evaluate premium".to_string(),
        ),
        ComputationOverflow => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "COMPUTATION_OVERFLOW",
            "u128 overflow during mint sizing — pool may have exotic reserves".to_string(),
        ),
    };
    error_response(status, code, msg)
}

fn buyback_error_response(err: &BuybackError) -> axum::response::Response {
    use BuybackError::*;
    let (status, code, msg) = match err {
        DiscountAboveThreshold { current_bps, required_bps } => (
            StatusCode::PRECONDITION_FAILED,
            "DISCOUNT_ABOVE_THRESHOLD",
            format!("discount {} bps > required {} bps (×1000)", current_bps, required_bps),
        ),
        CooldownActive { current_height, eligible_at } => (
            StatusCode::TOO_EARLY,
            "COOLDOWN_ACTIVE",
            format!("at height {}, eligible at {}", current_height, eligible_at),
        ),
        PoolTooShallow { current_qug_reserves, required } => (
            StatusCode::PRECONDITION_FAILED,
            "POOL_TOO_SHALLOW",
            format!("pool QUG reserves {} < required {}", current_qug_reserves, required),
        ),
        InsufficientYield { available, needed } => (
            StatusCode::PRECONDITION_FAILED,
            "INSUFFICIENT_YIELD",
            format!("accrued yield {} < needed {} (buybacks fund from yield only)", available, needed),
        ),
        NavOracleStale => (
            StatusCode::SERVICE_UNAVAILABLE,
            "NAV_ORACLE_STALE",
            "NAV oracle is stale — cannot evaluate discount".to_string(),
        ),
    };
    error_response(status, code, msg)
}
