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
/// v9.1.6: Merges Stratum pool stats with HTTP API mining stats.
/// All miners use HTTP API (/api/v1/mining/submit), not Stratum (port 3333),
/// so pool dashboard showed 0 workers/hashrate. Now reflects actual PPLNS miners.
pub async fn get_pool_stats(
    State(state): State<Arc<AppState>>,
) -> Result<Json<PoolStatsResponse>, StatusCode> {
    let pool = state.mining_pool.as_ref().ok_or_else(|| {
        warn!("Mining pool not initialized");
        StatusCode::SERVICE_UNAVAILABLE
    })?;

    let stratum_stats = pool.stats();
    let config = pool.config();

    // v9.1.6: Read HTTP API mining stats (the actual active miners)
    let (http_miners, http_hashrate, http_blocks) =
        if let Some(ref mining_stats_arc) = state.mining_statistics {
            let stats = mining_stats_arc.read().await;
            let hashrate_sum: f64 = stats.active_miners.values().map(|m| m.last_hashrate).sum();
            (stats.active_miners.len(), hashrate_sum, stats.total_solutions_accepted)
        } else {
            (0, 0.0, 0)
        };

    // Merge: use whichever source has more data (HTTP API dominates in practice)
    let total_workers = stratum_stats.workers + http_miners;
    let total_hashrate = stratum_stats.hashrate + http_hashrate;
    let total_blocks = stratum_stats.blocks_found.max(http_blocks);

    // Current block height for round tracking
    let current_height = state.current_height_atomic.load(std::sync::atomic::Ordering::Relaxed);

    Ok(Json(PoolStatsResponse {
        name: config.name.clone(),
        version: "v2.2.1-beta".to_string(),
        hashrate: total_hashrate,
        workers: total_workers,
        blocks_found: total_blocks,
        current_round: if pool.current_round_id() > 0 { pool.current_round_id() } else { current_height },
        difficulty: stratum_stats.network_difficulty,
        fee_bps: config.fees.effective_fee_bps(),
        min_payout: config.payout.min_payout,
        shares_this_round: stratum_stats.total_shares,
        uptime_seconds: stratum_stats.uptime_seconds,
        stratum_port: config.stratum.port,
    }))
}

/// Get worker statistics
/// v9.1.6: Includes HTTP API miners (from MiningStatistics) alongside Stratum workers.
pub async fn get_workers(
    State(state): State<Arc<AppState>>,
    Query(query): Query<WorkerListQuery>,
) -> Result<Json<Vec<WorkerStatsResponse>>, StatusCode> {
    let pool = state.mining_pool.as_ref().ok_or_else(|| {
        warn!("Mining pool not initialized");
        StatusCode::SERVICE_UNAVAILABLE
    })?;

    let limit = query.limit.unwrap_or(100);
    let offset = query.offset.unwrap_or(0);

    // Stratum workers
    let stratum_workers = pool.worker_manager().get_all_workers();
    let mut responses: Vec<WorkerStatsResponse> = stratum_workers
        .into_iter()
        .filter(|w| {
            if let Some(ref wallet) = query.wallet {
                w.wallet_address == *wallet
            } else {
                true
            }
        })
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

    // v9.1.6: Add HTTP API miners from MiningStatistics
    if let Some(ref mining_stats_arc) = state.mining_statistics {
        let stats = mining_stats_arc.read().await;
        for (key, miner) in &stats.active_miners {
            // key format is "wallet_address:worker_id"
            let parts: Vec<&str> = key.splitn(2, ':').collect();
            let wallet_addr = parts.first().copied().unwrap_or(key.as_str());
            let worker_id = parts.get(1).copied().unwrap_or("http");

            // Filter by wallet if requested
            if let Some(ref wallet_filter) = query.wallet {
                if wallet_addr != wallet_filter.as_str() {
                    continue;
                }
            }

            let now_ts = chrono::Utc::now().timestamp();
            responses.push(WorkerStatsResponse {
                worker_id: format!("http-{}", worker_id),
                wallet_address: wallet_addr.to_string(),
                hashrate: miner.last_hashrate, // Already H/s (v3.5.6+)
                difficulty: 0.0,
                shares_submitted: miner.total_solutions,
                shares_stale: 0,
                shares_invalid: 0,
                blocks_found: 0,
                last_share_time: now_ts, // Active miners are recent
                connected_since: now_ts - 300, // Approximate
                is_connected: true,
            });
        }
    }

    // Sort by hashrate descending, then paginate
    responses.sort_by(|a, b| b.hashrate.partial_cmp(&a.hashrate).unwrap_or(std::cmp::Ordering::Equal));
    let responses: Vec<_> = responses.into_iter().skip(offset).take(limit).collect();

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

/// Round history entry for API response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundResponse {
    /// Round number
    pub round_id: u64,
    /// Block height found
    pub block_height: u64,
    /// Block hash (hex)
    pub block_hash: String,
    /// Total block reward (atomic units)
    pub block_reward: u64,
    /// Pool fee (atomic units)
    pub pool_fee: u64,
    /// Dev fee (atomic units)
    pub dev_fee: u64,
    /// Miner rewards (atomic units)
    pub miner_rewards: u64,
    /// Number of payouts
    pub payout_count: usize,
    /// Worker who found the block
    pub found_by: String,
    /// Timestamp
    pub timestamp: i64,
    /// Total shares in round
    pub total_shares: u64,
    /// Total difficulty in round
    pub total_difficulty: f64,
}

/// Get round history
pub async fn get_rounds(
    State(state): State<Arc<AppState>>,
    Query(query): Query<PayoutHistoryQuery>,
) -> Result<Json<Vec<RoundResponse>>, StatusCode> {
    let pool = state.mining_pool.as_ref().ok_or_else(|| {
        warn!("Mining pool not initialized");
        StatusCode::SERVICE_UNAVAILABLE
    })?;

    let limit = query.limit.unwrap_or(50);
    let rounds = pool.get_round_history();

    let responses: Vec<RoundResponse> = rounds
        .into_iter()
        .rev() // Most recent first
        .take(limit)
        .map(|r| RoundResponse {
            round_id: r.round_id,
            block_height: r.block_height,
            block_hash: hex::encode(&r.block_hash),
            block_reward: r.block_reward,
            pool_fee: r.pool_fee,
            dev_fee: r.dev_fee,
            miner_rewards: r.miner_rewards,
            payout_count: r.payouts.len(),
            found_by: r.found_by.to_string(),
            timestamp: r.timestamp.timestamp(),
            total_shares: r.total_shares,
            total_difficulty: r.total_difficulty,
        })
        .collect();

    Ok(Json(responses))
}

/// Get specific round by ID
pub async fn get_round(
    State(state): State<Arc<AppState>>,
    Path(round_id): Path<u64>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let pool = state.mining_pool.as_ref().ok_or_else(|| {
        warn!("Mining pool not initialized");
        StatusCode::SERVICE_UNAVAILABLE
    })?;

    let rounds = pool.get_round_history();
    let round = rounds.into_iter()
        .find(|r| r.round_id == round_id)
        .ok_or(StatusCode::NOT_FOUND)?;

    let payouts: Vec<serde_json::Value> = round.payouts.iter().map(|p| {
        serde_json::json!({
            "wallet_address": p.wallet_address,
            "amount": p.amount,
            "proportion": p.proportion,
            "difficulty_contribution": p.difficulty_contribution,
        })
    }).collect();

    Ok(Json(serde_json::json!({
        "round_id": round.round_id,
        "block_height": round.block_height,
        "block_hash": hex::encode(&round.block_hash),
        "block_reward": round.block_reward,
        "pool_fee": round.pool_fee,
        "dev_fee": round.dev_fee,
        "miner_rewards": round.miner_rewards,
        "found_by": round.found_by.to_string(),
        "timestamp": round.timestamp.timestamp(),
        "total_shares": round.total_shares,
        "total_difficulty": round.total_difficulty,
        "payouts": payouts,
    })))
}

/// Pool node info response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolNodeResponse {
    /// Peer ID (hex)
    pub peer_id: String,
    /// Stratum port
    pub stratum_port: u16,
    /// Current hashrate (H/s)
    pub hashrate: f64,
    /// Worker count
    pub worker_count: u32,
    /// Region
    pub region: String,
    /// Version
    pub version: String,
    /// Last seen (unix timestamp)
    pub last_seen: u64,
    /// Is accepting connections
    pub accepting_connections: bool,
}

/// Get discovered pool nodes
pub async fn get_pool_nodes(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<PoolNodeResponse>>, StatusCode> {
    let coordinator = state.distributed_pool_coordinator.as_ref().ok_or_else(|| {
        warn!("Distributed pool coordinator not initialized");
        StatusCode::SERVICE_UNAVAILABLE
    })?;

    let coord = coordinator.read().await;
    let nodes = coord.known_nodes().await;

    let responses: Vec<PoolNodeResponse> = nodes
        .into_iter()
        .map(|n| PoolNodeResponse {
            peer_id: hex::encode(&n.peer_id),
            stratum_port: n.stratum_port,
            hashrate: n.hashrate,
            worker_count: n.worker_count,
            region: n.region.clone(),
            version: n.version.clone(),
            last_seen: n.last_seen,
            accepting_connections: n.accepting_connections,
        })
        .collect();

    Ok(Json(responses))
}

/// Hashrate history entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HashrateEntry {
    /// Hashrate (H/s)
    pub hashrate: f64,
    /// Worker count
    pub workers: usize,
    /// Timestamp (unix seconds)
    pub timestamp: i64,
}

/// Get hashrate history
pub async fn get_hashrate_history(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<HashrateEntry>>, StatusCode> {
    let history = state.pool_hashrate_history.read().await;
    Ok(Json(history.clone()))
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
        // Round history (v5.7.0)
        .route("/rounds", get(get_rounds))
        .route("/rounds/:round_id", get(get_round))
        // Pool nodes (v5.7.0)
        .route("/nodes", get(get_pool_nodes))
        // Hashrate history (v5.7.0)
        .route("/hashrate/history", get(get_hashrate_history))
}
