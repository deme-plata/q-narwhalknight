//! Mining Pool API endpoints
//!
//! v2.2.1-beta: Stratum mining pool integration with PPLNS rewards
//! Always enabled - no feature flag required
//!
//! Provides HTTP API for:
//! - Pool statistics and status
//! - Worker management
//! - Payout history
//! - Share submission (via Stratum protocol)

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    routing::get,
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, warn};

use crate::AppState;

/// Pool statistics response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolStatsResponse {
    /// Pool name
    pub name: String,
    /// Pool version
    pub version: String,
    /// Total hashrate (H/s)
    pub hashrate: f64,
    /// Number of active workers
    pub workers: usize,
    /// Total blocks found
    pub blocks_found: u64,
    /// Current round
    pub current_round: u64,
    /// Current difficulty
    pub difficulty: f64,
    /// Pool fee (basis points, 100 = 1%)
    pub fee_bps: u64,
    /// Minimum payout threshold (atomic units)
    pub min_payout: u64,
    /// Total shares this round
    pub shares_this_round: u64,
    /// Pool uptime in seconds
    pub uptime_seconds: u64,
    /// Stratum port
    pub stratum_port: u16,
}

/// Worker statistics response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerStatsResponse {
    /// Worker ID
    pub worker_id: String,
    /// Wallet address
    pub wallet_address: String,
    /// Current hashrate (H/s)
    pub hashrate: f64,
    /// Current difficulty
    pub difficulty: f64,
    /// Total shares submitted
    pub shares_submitted: u64,
    /// Stale shares
    pub shares_stale: u64,
    /// Invalid shares
    pub shares_invalid: u64,
    /// Blocks found
    pub blocks_found: u64,
    /// Last share time (unix timestamp)
    pub last_share_time: i64,
    /// Connected since (unix timestamp)
    pub connected_since: i64,
    /// Is currently connected
    pub is_connected: bool,
}

/// Payout history entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PayoutEntry {
    /// Payout ID
    pub id: u64,
    /// Amount (atomic units)
    pub amount: u64,
    /// Transaction hash (if completed)
    pub tx_hash: Option<String>,
    /// Status
    pub status: String,
    /// Timestamp
    pub timestamp: i64,
}

/// Pending balance response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingBalanceResponse {
    /// Wallet address
    pub wallet_address: String,
    /// Pending balance (atomic units)
    pub pending_balance: u64,
    /// Estimated payout time (if above threshold)
    pub estimated_payout: Option<String>,
}

/// Query params for worker list
#[derive(Debug, Clone, Deserialize)]
pub struct WorkerListQuery {
    /// Filter by wallet address
    pub wallet: Option<String>,
    /// Limit results
    pub limit: Option<usize>,
    /// Offset for pagination
    pub offset: Option<usize>,
}

/// Query params for payout history
#[derive(Debug, Clone, Deserialize)]
pub struct PayoutHistoryQuery {
    /// Filter by wallet address
    pub wallet: Option<String>,
    /// Limit results
    pub limit: Option<usize>,
}

/// Get pool statistics
pub async fn get_pool_stats(
    State(state): State<Arc<AppState>>,
) -> Result<Json<PoolStatsResponse>, StatusCode> {
    let pool = state.mining_pool.as_ref().ok_or_else(|| {
        warn!("Mining pool not initialized");
        StatusCode::SERVICE_UNAVAILABLE
    })?;

    let stats = pool.stats();
    let config = pool.config();

    Ok(Json(PoolStatsResponse {
        name: config.name.clone(),
        version: "v2.2.1-beta".to_string(),
        hashrate: stats.hashrate,
        workers: stats.workers,
        blocks_found: stats.blocks_found,
        current_round: pool.current_round_id(),
        difficulty: stats.network_difficulty,
        fee_bps: config.fees.effective_fee_bps(),
        min_payout: config.payout.min_payout,
        shares_this_round: stats.total_shares,
        uptime_seconds: stats.uptime_seconds,
        stratum_port: config.stratum.port,
    }))
}

/// Get worker statistics
pub async fn get_workers(
    State(state): State<Arc<AppState>>,
    Query(query): Query<WorkerListQuery>,
) -> Result<Json<Vec<WorkerStatsResponse>>, StatusCode> {
    let pool = state.mining_pool.as_ref().ok_or_else(|| {
        warn!("Mining pool not initialized");
        StatusCode::SERVICE_UNAVAILABLE
    })?;

    let workers = pool.worker_manager().get_all_workers();
    let limit = query.limit.unwrap_or(100);
    let offset = query.offset.unwrap_or(0);

    let mut responses: Vec<WorkerStatsResponse> = workers
        .into_iter()
        .filter(|w| {
            if let Some(ref wallet) = query.wallet {
                w.wallet_address == *wallet
            } else {
                true
            }
        })
        .skip(offset)
        .take(limit)
        .map(|w| {
            WorkerStatsResponse {
                worker_id: w.id.to_string(),
                wallet_address: w.wallet_address.clone(),
                hashrate: w.hashrate,
                difficulty: w.difficulty,
                shares_submitted: w.stats.total_shares,
                shares_stale: w.stats.stale_shares,
                shares_invalid: w.stats.rejected_shares,
                blocks_found: w.stats.blocks_found,
                last_share_time: w.last_activity.timestamp(),
                connected_since: w.connected_at.timestamp(),
                is_connected: matches!(w.state, q_mining_pool::worker::WorkerState::Active),
            }
        })
        .collect();

    // Sort by hashrate descending
    responses.sort_by(|a, b| b.hashrate.partial_cmp(&a.hashrate).unwrap_or(std::cmp::Ordering::Equal));

    Ok(Json(responses))
}

/// Get specific worker by ID
pub async fn get_worker(
    State(state): State<Arc<AppState>>,
    Path(worker_id): Path<String>,
) -> Result<Json<WorkerStatsResponse>, StatusCode> {
    let pool = state.mining_pool.as_ref().ok_or_else(|| {
        warn!("Mining pool not initialized");
        StatusCode::SERVICE_UNAVAILABLE
    })?;

    // Find worker in the list
    let workers = pool.worker_manager().get_all_workers();
    let worker = workers.into_iter()
        .find(|w| w.id.to_string() == worker_id)
        .ok_or_else(|| {
            debug!("Worker not found: {}", worker_id);
            StatusCode::NOT_FOUND
        })?;

    Ok(Json(WorkerStatsResponse {
        worker_id: worker.id.to_string(),
        wallet_address: worker.wallet_address.clone(),
        hashrate: worker.hashrate,
        difficulty: worker.difficulty,
        shares_submitted: worker.stats.total_shares,
        shares_stale: worker.stats.stale_shares,
        shares_invalid: worker.stats.rejected_shares,
        blocks_found: worker.stats.blocks_found,
        last_share_time: worker.last_activity.timestamp(),
        connected_since: worker.connected_at.timestamp(),
        is_connected: matches!(worker.state, q_mining_pool::worker::WorkerState::Active),
    }))
}

/// Get pending balance for a wallet
pub async fn get_pending_balance(
    State(state): State<Arc<AppState>>,
    Path(wallet): Path<String>,
) -> Result<Json<PendingBalanceResponse>, StatusCode> {
    let pool = state.mining_pool.as_ref().ok_or_else(|| {
        warn!("Mining pool not initialized");
        StatusCode::SERVICE_UNAVAILABLE
    })?;

    let pending = pool.get_pending_balance(&wallet);
    let min_payout = pool.config().payout.min_payout;

    let estimated_payout = if pending >= min_payout {
        Some("Next payout cycle".to_string())
    } else if pending > 0 {
        let remaining = min_payout - pending;
        Some(format!("{} more required", remaining))
    } else {
        None
    };

    Ok(Json(PendingBalanceResponse {
        wallet_address: wallet,
        pending_balance: pending,
        estimated_payout,
    }))
}

/// Get payout history
pub async fn get_payout_history(
    State(state): State<Arc<AppState>>,
    Query(query): Query<PayoutHistoryQuery>,
) -> Result<Json<Vec<PayoutEntry>>, StatusCode> {
    let pool = state.mining_pool.as_ref().ok_or_else(|| {
        warn!("Mining pool not initialized");
        StatusCode::SERVICE_UNAVAILABLE
    })?;

    let limit = query.limit.unwrap_or(50);

    let payouts = if let Some(wallet) = query.wallet {
        pool.get_wallet_payouts(&wallet)
    } else {
        // Use payout_processor for recent batches
        pool.payout_processor().recent_batches(limit)
            .into_iter()
            .flat_map(|batch| batch.payouts)
            .collect()
    };

    let entries: Vec<PayoutEntry> = payouts
        .into_iter()
        .take(limit)
        .map(|p| PayoutEntry {
            id: p.id,
            amount: p.amount,
            tx_hash: p.tx_hash,
            status: format!("{:?}", p.status),
            timestamp: p.created_at.timestamp(),
        })
        .collect();

    Ok(Json(entries))
}

/// Get payout statistics
pub async fn get_payout_stats(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let pool = state.mining_pool.as_ref().ok_or_else(|| {
        warn!("Mining pool not initialized");
        StatusCode::SERVICE_UNAVAILABLE
    })?;

    let stats = pool.payout_stats();

    Ok(Json(serde_json::json!({
        "total_paid": stats.total_paid,
        "pending_total": stats.pending_total,
        "pending_wallets": stats.pending_wallets,
        "min_payout": stats.min_payout,
        "payouts_completed": stats.payouts_completed,
        "batches_completed": stats.batches_completed,
    })))
}

/// Create the pool API router
pub fn create_pool_router() -> Router<Arc<AppState>> {
    Router::new()
        // Pool overview
        .route("/stats", get(get_pool_stats))
        // Workers
        .route("/workers", get(get_workers))
        .route("/workers/:worker_id", get(get_worker))
        // Balances and payouts
        .route("/balance/:wallet", get(get_pending_balance))
        .route("/payouts", get(get_payout_history))
        .route("/payouts/stats", get(get_payout_stats))
}
