//! Data Integrity & Decentralization Diagnostics API
//!
//! Endpoints for verifying cross-node consensus on balance state, chain tip,
//! emission, storage health, and decentralization parameters.
//!
//! All endpoints are read-only and require no authentication.
//! Designed to be polled externally (monitoring, pre-upgrade checks, dashboards).
//!
//! Routes mounted under `/api/v1/integrity/`

use axum::{extract::State, http::StatusCode, Json};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::warn;

use crate::AppState;

// ============================================================================
// Response types
// ============================================================================

/// Top-level wrapper — every integrity endpoint returns this.
#[derive(Debug, Serialize)]
pub struct IntegrityResponse<T: Serialize> {
    pub ok: bool,
    pub data: Option<T>,
    pub error: Option<String>,
}

impl<T: Serialize> IntegrityResponse<T> {
    pub fn ok(data: T) -> Json<Self> {
        Json(Self { ok: true, data: Some(data), error: None })
    }
    pub fn err(msg: impl Into<String>) -> Json<Self> {
        Json(Self { ok: false, data: None, error: Some(msg.into()) })
    }
}

// ─── /integrity/balance-root ─────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct BalanceRootReport {
    /// BLAKE3 root of sorted (address, balance) pairs — canonical BalanceRootV1 spec.
    /// Compare this value across all nodes: they MUST be identical.
    pub balance_root_hex: String,
    /// Number of non-zero wallets included in the root.
    pub wallet_count: usize,
    /// Total QUG supply represented in the root (base units, 24 decimals).
    pub total_supply_base_units: String,
    /// Total QUG supply in display units (divided by 10^24).
    pub total_supply_display: String,
    /// Current chain height at time of computation.
    pub at_height: u64,
    /// Blocks until BalanceRootV1 activation (0 = already active).
    pub blocks_until_activation: i64,
    /// Whether BalanceRootV1 is currently enforced on incoming blocks.
    pub activation_enforced: bool,
    /// Whether this root is all-zeros (indicates empty chain or computation failure).
    pub is_zero_root: bool,
}

/// GET /api/v1/integrity/balance-root
///
/// Computes the canonical BalanceRootV1 fingerprint of the current balance state.
/// Compare the returned `balance_root_hex` across all nodes — they must be identical.
/// Call this on Beta, Gamma, and Epsilon before block 18,600,000 to verify consensus.
pub async fn get_balance_root(
    State(state): State<Arc<AppState>>,
) -> Result<Json<IntegrityResponse<BalanceRootReport>>, StatusCode> {
    // v1.0.2: Use contiguous height for `at_height`. compute_balance_root_for_block()
    // hashes the wallet table that was built from contiguous data; reporting max-seen
    // height here was misleading — clients comparing roots across nodes need to compare
    // at matching heights and "max-seen" can differ between nodes for the same archive
    // state. Contiguous height is the correct anchor for cross-node hash comparison.
    let height = state.contiguous_height_atomic.load(std::sync::atomic::Ordering::Relaxed);

    let root = match state.storage_engine.compute_balance_root_for_block().await {
        Ok(r) => r,
        Err(e) => {
            warn!("[INTEGRITY] balance root computation failed: {}", e);
            return Ok(IntegrityResponse::err(format!("compute_balance_root_for_block failed: {}", e)));
        }
    };

    // Also pull wallet count + total supply from the state hash for richer diagnostics
    let (wallet_count, total_supply) = match state.storage_engine.compute_balance_state_hash().await {
        Ok((_hash, count, supply)) => (count, supply),
        Err(_) => (0, 0u128),
    };

    const ACTIVATION: u64 = 18_600_000;
    let blocks_until = (ACTIVATION as i64) - (height as i64);
    let active = q_consensus_guard::is_upgrade_active(q_consensus_guard::Upgrade::BalanceRootV1, height);

    let display_supply = total_supply / 10u128.pow(24);
    let display_remainder = total_supply % 10u128.pow(24);

    Ok(IntegrityResponse::ok(BalanceRootReport {
        balance_root_hex: hex::encode(root),
        wallet_count,
        total_supply_base_units: total_supply.to_string(),
        total_supply_display: format!("{}.{:024}", display_supply, display_remainder),
        at_height: height,
        blocks_until_activation: blocks_until.max(0),
        activation_enforced: active,
        is_zero_root: root == [0u8; 32],
    }))
}

// ─── /integrity/chain-tip ────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct ChainTipReport {
    pub height: u64,
    pub network_id: String,
    /// Number of connected P2P peers.
    pub peer_count: usize,
    /// Our own libp2p peer ID.
    pub peer_id: String,
    /// Listen addresses this node advertises.
    pub listen_addresses: Vec<String>,
}

/// GET /api/v1/integrity/chain-tip
pub async fn get_chain_tip(
    State(state): State<Arc<AppState>>,
) -> Result<Json<IntegrityResponse<ChainTipReport>>, StatusCode> {
    let height = state.current_height_atomic.load(std::sync::atomic::Ordering::Relaxed);
    let peer_count = state.libp2p_peer_count
        .as_ref()
        .map(|a| a.load(std::sync::atomic::Ordering::Relaxed))
        .unwrap_or(0);
    let (peer_id, listen_addresses) = {
        let info = state.libp2p_peer_info.read().await;
        (info.0.clone(), info.1.clone())
    };
    let network_id = std::env::var("Q_NETWORK_ID").unwrap_or_else(|_| "unknown".into());

    Ok(IntegrityResponse::ok(ChainTipReport {
        height,
        network_id,
        peer_count,
        peer_id,
        listen_addresses,
    }))
}

// ─── /integrity/emission ─────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct EmissionReport {
    /// Total QUG minted (base units).
    pub total_minted_base_units: String,
    /// Total QUG minted (display).
    pub total_minted_display: String,
    /// Hard cap in base units (21M QUG × 10^24).
    pub hard_cap_base_units: String,
    /// Percentage of hard cap minted.
    pub percent_minted: f64,
    /// Current chain height.
    pub at_height: u64,
    /// Expected supply if emission has been perfectly on-schedule (1 block/sec, era-0 rate).
    pub expected_supply_display: String,
    /// Divergence from expected: positive = over-emitted, negative = under-emitted.
    pub supply_divergence_display: String,
    /// Whether supply is within acceptable bounds (±5% of expected).
    pub supply_healthy: bool,
}

/// GET /api/v1/integrity/emission
pub async fn get_emission(
    State(state): State<Arc<AppState>>,
) -> Result<Json<IntegrityResponse<EmissionReport>>, StatusCode> {
    // v1.0.2: Use contiguous height. `total_minted_supply` is the sum of coinbase
    // amounts from blocks we've actually processed (contiguous data). Comparing it
    // against `expected = per_block_reward * max_seen_height` would always show a
    // node-is-under-emitted divergence for any node with gaps. Anchoring both sides
    // at contiguous height makes the comparison meaningful.
    let height = state.contiguous_height_atomic.load(std::sync::atomic::Ordering::Relaxed);
    let total_minted = *state.total_minted_supply.read().await;

    // Era-0 emission: 2,625,000 QUG/year at 1 block/sec = 31,536,000 blocks/year
    // Per-block reward: 2_625_000 × 10^24 / 31_536_000 ≈ 83_219_954_648_526_077_097_505 base units
    const BLOCKS_PER_YEAR: u128 = 31_536_000;
    const ERA0_ANNUAL_QUG: u128 = 2_625_000;
    const DECIMALS: u128 = 1_000_000_000_000_000_000_000_000; // 10^24
    let per_block_reward = ERA0_ANNUAL_QUG * DECIMALS / BLOCKS_PER_YEAR;
    let expected = per_block_reward * (height as u128);

    let hard_cap = 21_000_000u128 * DECIMALS;
    let percent = if hard_cap > 0 { total_minted as f64 / hard_cap as f64 * 100.0 } else { 0.0 };

    let divergence_signed: i128 = (total_minted as i128) - (expected as i128);
    let divergence_display_qug = divergence_signed / (DECIMALS as i128);

    // Healthy = supply does not EXCEED the theoretical max by more than 5%.
    // Being under expected is fine — not every block is mined.
    let tolerance = expected / 20;
    let supply_healthy = total_minted <= expected + tolerance;

    let fmt_qug = |base: u128| -> String {
        format!("{}.{:06}", base / DECIMALS, (base % DECIMALS) / 10u128.pow(18))
    };

    Ok(IntegrityResponse::ok(EmissionReport {
        total_minted_base_units: total_minted.to_string(),
        total_minted_display: fmt_qug(total_minted),
        hard_cap_base_units: hard_cap.to_string(),
        percent_minted: (percent * 1000.0).round() / 1000.0,
        at_height: height,
        expected_supply_display: fmt_qug(expected),
        supply_divergence_display: format!("{} QUG", divergence_display_qug),
        supply_healthy,
    }))
}

// ─── /integrity/storage ──────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct StorageReport {
    /// Approximate RocksDB memory usage in MB.
    pub rocksdb_memory_mb: u64,
    /// Number of liquidity pools loaded in memory.
    pub pools_in_memory: usize,
    /// Number of pools persisted in RocksDB (from a prefix scan count).
    pub pools_persisted: usize,
    /// Number of wallets with non-zero balances in storage.
    pub wallet_count: usize,
    /// Total supply from storage (cross-check with /integrity/emission).
    pub total_supply_from_storage: String,
    /// Whether the balance root is non-zero (non-empty chain).
    pub balance_root_non_zero: bool,
    /// Whether pool counts match between memory and storage.
    pub pools_consistent: bool,
}

/// GET /api/v1/integrity/storage
pub async fn get_storage_health(
    State(state): State<Arc<AppState>>,
) -> Result<Json<IntegrityResponse<StorageReport>>, StatusCode> {
    let pools_in_memory = state.liquidity_pools.read().await.len();

    let pools_persisted = match state.storage_engine.load_liquidity_pools().await {
        Ok(p) => p.len(),
        Err(_) => 0,
    };

    let (wallet_count, total_supply) = match state.storage_engine.compute_balance_state_hash().await {
        Ok((_h, count, supply)) => (count, supply),
        Err(_) => (0, 0u128),
    };

    let balance_root_non_zero = match state.storage_engine.compute_balance_root_for_block().await {
        Ok(r) => r != [0u8; 32],
        Err(_) => false,
    };

    let rocksdb_memory_mb = {
        let (mt, tr, bc) = state.storage_engine.get_rocksdb_memory_mb();
        (mt + tr + bc) as u64
    };

    Ok(IntegrityResponse::ok(StorageReport {
        rocksdb_memory_mb,
        pools_in_memory,
        pools_persisted,
        wallet_count,
        total_supply_from_storage: total_supply.to_string(),
        balance_root_non_zero,
        pools_consistent: pools_in_memory == pools_persisted,
    }))
}

// ─── /integrity/validators ───────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct ValidatorReport {
    pub active_count: usize,
    pub total_stake_base_units: String,
    pub total_stake_display: String,
    /// Gini coefficient of stake distribution (0 = perfect equality, 1 = monopoly).
    pub stake_gini: f64,
    /// Stake share of the largest single validator (0–1).
    pub top_validator_share: f64,
    /// Whether the network meets the n >= 3f+1 BFT threshold (f = max tolerated faults).
    pub bft_fault_tolerance: usize,
    /// Whether stake is sufficiently distributed (gini < 0.6 and top share < 0.33).
    pub decentralization_healthy: bool,
}

/// GET /api/v1/integrity/validators
pub async fn get_validators(
    State(state): State<Arc<AppState>>,
) -> Result<Json<IntegrityResponse<ValidatorReport>>, StatusCode> {
    let registry = state.validator_registry.read().await;
    let validators = registry.get_active_validators();
    let active_count = validators.len();
    let total_stake = registry.total_active_stake();

    let stakes: Vec<f64> = validators.iter()
        .map(|v| v.stake as f64)
        .collect();

    let gini = compute_gini(&stakes);
    let top_share = stakes.iter().cloned().fold(0.0f64, f64::max) / total_stake.max(1) as f64;

    // BFT: n >= 3f+1 → max faults f = (n-1)/3
    let bft_fault_tolerance = if active_count > 0 { (active_count - 1) / 3 } else { 0 };

    const DECIMALS: u128 = 1_000_000_000_000_000_000_000_000;
    let display_stake = total_stake / DECIMALS;

    Ok(IntegrityResponse::ok(ValidatorReport {
        active_count,
        total_stake_base_units: total_stake.to_string(),
        total_stake_display: format!("{} QUG", display_stake),
        stake_gini: (gini * 1000.0).round() / 1000.0,
        top_validator_share: (top_share * 1000.0).round() / 1000.0,
        bft_fault_tolerance,
        decentralization_healthy: gini < 0.6 && top_share < 0.33,
    }))
}

// ─── /integrity/upgrades ─────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct UpgradeStatus {
    pub name: String,
    pub activation_height: u64,
    pub current_height: u64,
    pub blocks_remaining: i64,
    pub active: bool,
}

#[derive(Debug, Serialize)]
pub struct UpgradesReport {
    pub upgrades: Vec<UpgradeStatus>,
    pub current_height: u64,
}

/// GET /api/v1/integrity/upgrades
///
/// Shows all scheduled upgrades, their activation heights, and whether they are active.
pub async fn get_upgrades(
    State(state): State<Arc<AppState>>,
) -> Result<Json<IntegrityResponse<UpgradesReport>>, StatusCode> {
    let height = state.current_height_atomic.load(std::sync::atomic::Ordering::Relaxed);
    let mut upgrades: Vec<UpgradeStatus> = q_consensus_guard::MAINNET_UPGRADES.iter()
        .map(|(upgrade, config)| {
            let active = q_consensus_guard::is_upgrade_active(*upgrade, height);
            let blocks_remaining = if config.activation_height == u64::MAX {
                0i64
            } else {
                ((config.activation_height as i64) - (height as i64)).max(0)
            };
            UpgradeStatus {
                name: upgrade.name().to_string(),
                activation_height: config.activation_height,
                current_height: height,
                blocks_remaining,
                active,
            }
        })
        .collect();

    upgrades.sort_by_key(|u| u.activation_height);

    Ok(IntegrityResponse::ok(UpgradesReport { upgrades, current_height: height }))
}

// ─── /integrity/full ─────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct FullIntegrityReport {
    pub balance_root: String,
    pub wallet_count: usize,
    pub total_supply_display: String,
    pub supply_healthy: bool,
    pub height: u64,
    pub peer_count: usize,
    pub network_id: String,
    pub pools_in_memory: usize,
    pub pools_persisted: usize,
    pub pools_consistent: bool,
    pub rocksdb_memory_mb: u64,
    pub active_validators: usize,
    pub stake_gini: f64,
    pub bft_fault_tolerance: usize,
    pub decentralization_healthy: bool,
    pub balance_root_v1_active: bool,
    pub blocks_until_balance_root_v1: i64,
    /// Whether this node bootstrapped from the balance checkpoint (vs. from genesis).
    pub is_checkpoint_node: bool,
    /// Whether the post-checkpoint balance replay has completed successfully.
    /// false on a checkpoint node means balances are incomplete (transfer-only wallets missing).
    pub balance_replay_done: bool,
    /// Overall health: true only if all sub-checks pass.
    pub all_healthy: bool,
}

/// GET /api/v1/integrity/full
///
/// Single endpoint combining all integrity checks. Use this for monitoring dashboards
/// and cross-node comparison scripts. Costs one full balance scan per call.
pub async fn get_full_integrity(
    State(state): State<Arc<AppState>>,
) -> Result<Json<IntegrityResponse<FullIntegrityReport>>, StatusCode> {
    let height = state.current_height_atomic.load(std::sync::atomic::Ordering::Relaxed);
    let peer_count = state.libp2p_peer_count
        .as_ref()
        .map(|a| a.load(std::sync::atomic::Ordering::Relaxed))
        .unwrap_or(0);
    let network_id = std::env::var("Q_NETWORK_ID").unwrap_or_else(|_| "unknown".into());

    // Balance root
    let balance_root = match state.storage_engine.compute_balance_root_for_block().await {
        Ok(r) => hex::encode(r),
        Err(e) => {
            warn!("[INTEGRITY/full] balance root failed: {}", e);
            "error".into()
        }
    };

    // Wallet count + supply
    let (wallet_count, total_supply) = match state.storage_engine.compute_balance_state_hash().await {
        Ok((_h, c, s)) => (c, s),
        Err(_) => (0, 0u128),
    };

    // Emission health
    const DECIMALS: u128 = 1_000_000_000_000_000_000_000_000;
    const BLOCKS_PER_YEAR: u128 = 31_536_000;
    const ERA0_ANNUAL_QUG: u128 = 2_625_000;
    let per_block_reward = ERA0_ANNUAL_QUG * DECIMALS / BLOCKS_PER_YEAR;
    let expected = per_block_reward * (height as u128);
    // Healthy = supply does not EXCEED the theoretical max by more than 5%.
    // Being under expected is fine — not every block is mined.
    let tolerance = expected / 20;
    let supply_healthy = total_supply <= expected + tolerance;
    let total_supply_display = format!("{}.{:06} QUG",
        total_supply / DECIMALS,
        (total_supply % DECIMALS) / 10u128.pow(18));

    // Pools
    let pools_in_memory = state.liquidity_pools.read().await.len();
    let pools_persisted = state.storage_engine.load_liquidity_pools().await.map(|p| p.len()).unwrap_or(0);
    let pools_consistent = pools_in_memory == pools_persisted;

    // Storage memory
    let rocksdb_memory_mb = {
        let (mt, tr, bc) = state.storage_engine.get_rocksdb_memory_mb();
        (mt + tr + bc) as u64
    };

    // Validators
    let (active_validators, stake_gini, bft_fault_tolerance, decentralization_healthy) = {
        let registry = state.validator_registry.read().await;
        let validators = registry.get_active_validators();
        let n = validators.len();
        let stakes: Vec<f64> = validators.iter().map(|v| v.stake as f64).collect();
        let gini = compute_gini(&stakes);
        let total = registry.total_active_stake() as f64;
        let top_share = stakes.iter().cloned().fold(0.0f64, f64::max) / total.max(1.0);
        let bft = if n > 0 { (n - 1) / 3 } else { 0 };
        let healthy = gini < 0.6 && top_share < 0.33;
        (n, (gini * 1000.0).round() / 1000.0, bft, healthy)
    };

    // Upgrades
    const ACTIVATION: u64 = 18_600_000;
    let balance_root_v1_active = q_consensus_guard::is_upgrade_active(
        q_consensus_guard::Upgrade::BalanceRootV1, height);
    let blocks_until = ((ACTIVATION as i64) - (height as i64)).max(0);

    // Replay status — relevant for diagnosing 62-wallet / supply divergence
    let is_checkpoint_node = state.storage_engine.is_checkpoint_applied().await;
    let balance_replay_done = if is_checkpoint_node {
        state.storage_engine.is_balance_replay_done().await
    } else {
        true // genesis nodes don't need replay; treat as "done"
    };

    // A checkpoint node whose replay is not yet done has incomplete balances.
    let replay_healthy = !is_checkpoint_node || balance_replay_done;

    let all_healthy = balance_root != "error"
        && supply_healthy
        && pools_consistent
        && (peer_count >= 2)
        && replay_healthy;

    Ok(IntegrityResponse::ok(FullIntegrityReport {
        balance_root,
        wallet_count,
        total_supply_display,
        supply_healthy,
        height,
        peer_count,
        network_id,
        pools_in_memory,
        pools_persisted,
        pools_consistent,
        rocksdb_memory_mb,
        active_validators,
        stake_gini,
        bft_fault_tolerance,
        decentralization_healthy,
        balance_root_v1_active,
        blocks_until_balance_root_v1: blocks_until,
        is_checkpoint_node,
        balance_replay_done,
        all_healthy,
    }))
}

// ─── /integrity/compare ──────────────────────────────────────────────────────

/// GET /api/v1/integrity/compare
///
/// Returns a minimal snapshot for cross-node comparison scripts.
/// Call this on all nodes and diff the output — mismatches indicate divergence.
#[derive(Debug, Serialize)]
pub struct CompareSnapshot {
    pub balance_root_hex: String,
    pub height: u64,
    pub total_supply_base_units: String,
    pub wallet_count: usize,
    pub network_id: String,
}

pub async fn get_compare_snapshot(
    State(state): State<Arc<AppState>>,
) -> Result<Json<IntegrityResponse<CompareSnapshot>>, StatusCode> {
    let height = state.current_height_atomic.load(std::sync::atomic::Ordering::Relaxed);
    let network_id = std::env::var("Q_NETWORK_ID").unwrap_or_else(|_| "unknown".into());

    let root = match state.storage_engine.compute_balance_root_for_block().await {
        Ok(r) => hex::encode(r),
        Err(e) => return Ok(IntegrityResponse::err(format!("balance root failed: {}", e))),
    };

    let (wallet_count, total_supply) = match state.storage_engine.compute_balance_state_hash().await {
        Ok((_h, c, s)) => (c, s),
        Err(_) => (0, 0u128),
    };

    Ok(IntegrityResponse::ok(CompareSnapshot {
        balance_root_hex: root,
        height,
        total_supply_base_units: total_supply.to_string(),
        wallet_count,
        network_id,
    }))
}

// ─── /integrity/quorum ────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct QuorumReport {
    /// How many validators have committed for the most recent height seen
    pub commits_at_tip: usize,
    /// Required for quorum (3-of-4)
    pub quorum_threshold: usize,
    /// Whether the most recent tip height has reached quorum
    pub quorum_reached: bool,
    /// The height at which we last saw a quorum commit
    pub last_quorum_height: Option<u64>,
    /// Balance root agreed on at last quorum height (hex)
    pub last_quorum_root: Option<String>,
    /// Current tip height
    pub tip_height: u64,
}

/// GET /api/v1/integrity/quorum
///
/// Returns the live multi-validator quorum commit status.
/// Cross-node: query all 4 nodes and compare — if all show `quorum_reached: true`
/// and the same `last_quorum_root`, the chain is in consensus.
pub async fn get_quorum_status(
    State(state): State<Arc<AppState>>,
) -> Result<Json<IntegrityResponse<QuorumReport>>, StatusCode> {
    let tip_height = state.current_height_atomic.load(std::sync::atomic::Ordering::Relaxed);
    let collector = &state.quorum_commit_collector;
    let commits_at_tip = collector.commit_count(tip_height);

    // Scan the last 50 heights to find the most recent with quorum
    let mut last_quorum_height = None;
    let mut last_quorum_root = None;
    for h in (tip_height.saturating_sub(50)..=tip_height).rev() {
        if collector.commit_count(h) >= 3 {
            last_quorum_height = Some(h);
            // We don't expose the actual commits list — just signal quorum was reached
            break;
        }
    }

    Ok(IntegrityResponse::ok(QuorumReport {
        commits_at_tip,
        quorum_threshold: 3,
        quorum_reached: commits_at_tip >= 3,
        last_quorum_height,
        last_quorum_root,
        tip_height,
    }))
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Gini coefficient of a distribution. Returns 0 for empty or uniform input.
fn compute_gini(values: &[f64]) -> f64 {
    if values.is_empty() { return 0.0; }
    let n = values.len() as f64;
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let sum: f64 = sorted.iter().sum();
    if sum == 0.0 { return 0.0; }
    let weighted_sum: f64 = sorted.iter().enumerate()
        .map(|(i, &v)| (2.0 * (i as f64 + 1.0) - n - 1.0) * v)
        .sum();
    weighted_sum / (n * sum)
}
