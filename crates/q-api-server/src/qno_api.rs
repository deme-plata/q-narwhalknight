/// Quantum Neural Oracle (QNO) API - Prediction Staking Endpoints
///
/// This module provides API endpoints for:
/// - Prediction domain staking (stake QUG on AI prediction outcomes)
/// - Staking position management (stake, unstake, claim)
/// - Reward claiming with accrual
/// - Domain statistics
/// - Overall staking stats
/// - P2P gossip integration for decentralized validation
///
/// v1.4.2-beta: Added RocksDB persistence, unstake, reward accrual, P2P sync

use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use q_types::ApiResponse;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use q_api_server::wallet_auth::AuthenticatedWallet;
use q_api_server::AppState;
use q_api_server::oracle_integration::OracleService;

// Re-export types from q-storage for consistency
pub use q_storage::qno_storage::{PredictionDomain, QnoOperation, QnoStorage, StakingPosition, StakingStats};

// ============================================================================
// Request/Response Types
// ============================================================================

/// Request to stake in a prediction domain
#[derive(Debug, Deserialize)]
pub struct StakeRequest {
    pub domain: String,
    pub amount: String,          // Human-readable amount (e.g., "1000.5")
    pub confidence: f64,         // 0.1 to 1.0
    pub lock_days: u32,          // Lock period in days
    pub prediction_value: f64,   // User's predicted value (e.g., gas price, block time)
}

/// Response after staking
#[derive(Debug, Serialize)]
pub struct StakeResponse {
    pub stake_id: String,
    pub domain: String,
    pub amount: String,
    pub confidence: f64,
    pub lock_days: u32,
    pub expected_apy: f64,
    pub unlock_date: String,     // ISO 8601 date
    pub prediction_value: f64,   // Captured prediction for resolution
}

/// Request to unstake (early withdrawal)
#[derive(Debug, Deserialize)]
pub struct UnstakeRequest {
    pub stake_id: String,
}

/// Response after unstaking
#[derive(Debug, Serialize)]
pub struct UnstakeResponse {
    pub success: bool,
    pub principal_returned: String,
    pub penalty_amount: String,
    pub penalty_percentage: f64,
    pub message: String,
}

/// Request to claim rewards
#[derive(Debug, Deserialize)]
pub struct ClaimRequest {
    pub stake_id: String,
}

/// Response after claiming
#[derive(Debug, Serialize)]
pub struct ClaimResponse {
    pub success: bool,
    pub principal_amount: String,
    pub reward_amount: String,
    pub total_returned: String,
    pub prediction_accuracy: f64,
    pub transaction_hash: String,
}

/// Leaderboard entry
#[derive(Debug, Serialize)]
pub struct LeaderboardEntry {
    pub rank: u32,
    pub address: String,  // Truncated for privacy
    pub total_staked: String,
    pub total_rewards: String,
    pub accuracy: f64,
}

// ============================================================================
// Utility Functions
// ============================================================================

const QUG_DECIMALS: u32 = 8; // Must match q_types::QUG_DECIMALS

fn parse_amount(amount_str: &str) -> Result<u128, String> {
    let amount: f64 = amount_str
        .parse()
        .map_err(|_| "Invalid amount format".to_string())?;

    if amount <= 0.0 {
        return Err("Amount must be positive".to_string());
    }

    let base_units = (amount * 10f64.powi(QUG_DECIMALS as i32)) as u128;
    Ok(base_units)
}

fn format_amount(base_units: u128) -> String {
    let amount = base_units as f64 / 10f64.powi(QUG_DECIMALS as i32);
    format!("{:.8}", amount)
}

fn get_lock_multiplier(lock_days: u32) -> f64 {
    match lock_days {
        0..=6 => 1.0,
        7..=13 => 1.0,
        14..=29 => 1.25,
        30..=89 => 1.5,
        90..=179 => 2.0,
        _ => 2.5,
    }
}

/// Calculate early withdrawal penalty percentage
/// Higher penalty for earlier withdrawal
fn calculate_unstake_penalty(lock_days: u32, days_remaining: u64) -> f64 {
    if days_remaining == 0 {
        return 0.0; // No penalty if lock period ended
    }

    let progress = 1.0 - (days_remaining as f64 / lock_days as f64);

    // Base penalty: 10% at start, decreases linearly
    // Additional penalty for longer lock periods
    let base_penalty = 0.10 * (1.0 - progress);

    // Longer lock periods have slightly higher early penalty
    let lock_factor = match lock_days {
        0..=6 => 1.0,
        7..=29 => 1.2,
        30..=89 => 1.5,
        90..=179 => 1.8,
        _ => 2.0,
    };

    (base_penalty * lock_factor).min(0.50) // Cap at 50%
}

// ============================================================================
// API Handlers
// ============================================================================

/// GET /api/v1/qno/domains - Get all prediction domains
pub async fn get_prediction_domains(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<Vec<PredictionDomain>>>, StatusCode> {
    info!("🔮 [QNO] Fetching prediction domains");

    let qno = state.qno_storage.read().await;
    if let Some(storage) = qno.as_ref() {
        let domains = storage.get_domains().await;
        let active_domains: Vec<_> = domains.into_iter().filter(|d| d.active).collect();
        Ok(Json(ApiResponse::success(active_domains)))
    } else {
        // Fallback: return default domains if storage not initialized
        warn!("⚠️ [QNO] Storage not initialized, returning empty domains");
        Ok(Json(ApiResponse::success(vec![])))
    }
}

/// GET /api/v1/qno/stats - Get overall staking statistics
pub async fn get_staking_stats(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<StakingStats>>, StatusCode> {
    info!("📊 [QNO] Fetching staking statistics");

    let qno = state.qno_storage.read().await;
    if let Some(storage) = qno.as_ref() {
        let stats = storage.get_stats().await;
        Ok(Json(ApiResponse::success(stats)))
    } else {
        Ok(Json(ApiResponse::success(StakingStats::default())))
    }
}

/// GET /api/v1/qno/stakes - Get staking positions for address (AUTHENTICATED)
pub async fn get_staking_positions(
    auth: AuthenticatedWallet,
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<Vec<StakingPosition>>>, StatusCode> {
    let address_hex = hex::encode(auth.address);
    info!("🔍 [QNO] Fetching staking positions for authenticated wallet");

    let qno = state.qno_storage.read().await;
    if let Some(storage) = qno.as_ref() {
        let mut positions = storage.get_positions(&address_hex).await;

        // Update status for any positions that have unlocked
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        for position in &mut positions {
            if position.status == "active" && now >= position.unlocks_at {
                position.status = "unlocked".to_string();
            }
        }

        Ok(Json(ApiResponse::success(positions)))
    } else {
        Ok(Json(ApiResponse::success(vec![])))
    }
}

/// POST /api/v1/qno/stake - Create a new staking position (AUTHENTICATED)
pub async fn stake_prediction(
    auth: AuthenticatedWallet,
    State(state): State<Arc<AppState>>,
    Json(request): Json<StakeRequest>,
) -> Result<Json<ApiResponse<StakeResponse>>, StatusCode> {
    let address_hex = hex::encode(auth.address);
    info!("🎯 [QNO] Processing stake request for authenticated wallet");

    // Validate confidence level
    if request.confidence < 0.1 || request.confidence > 1.0 {
        warn!("❌ [QNO] Invalid confidence level: {}", request.confidence);
        return Ok(Json(ApiResponse::error("Confidence must be between 0.1 and 1.0".to_string())));
    }

    // Parse and validate amount
    let amount = match parse_amount(&request.amount) {
        Ok(a) => a,
        Err(e) => {
            warn!("❌ [QNO] Invalid amount: {}", e);
            return Ok(Json(ApiResponse::error(e)));
        }
    };

    // Get QNO storage
    let qno = state.qno_storage.read().await;
    let storage = match qno.as_ref() {
        Some(s) => s,
        None => {
            error!("❌ [QNO] Storage not initialized");
            return Ok(Json(ApiResponse::error("QNO system not initialized".to_string())));
        }
    };

    // Validate domain exists
    let domain = match storage.get_domain(&request.domain).await {
        Some(d) if d.active => d,
        _ => {
            warn!("❌ [QNO] Invalid domain: {}", request.domain);
            return Ok(Json(ApiResponse::error("Invalid prediction domain".to_string())));
        }
    };

    // Check min/max stake
    if amount < domain.min_stake {
        return Ok(Json(ApiResponse::error(format!(
            "Minimum stake for {} is {} QUG",
            domain.name,
            format_amount(domain.min_stake)
        ))));
    }
    if amount > domain.max_stake {
        return Ok(Json(ApiResponse::error(format!(
            "Maximum stake for {} is {} QUG",
            domain.name,
            format_amount(domain.max_stake)
        ))));
    }

    // Calculate timing (do this before the lock to minimize lock hold time)
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let unlock_timestamp = now + (request.lock_days as u64 * 24 * 60 * 60);
    let lock_multiplier = get_lock_multiplier(request.lock_days);

    // Calculate expected APY with confidence and lock multiplier
    let expected_apy = domain.apy * request.confidence * lock_multiplier;

    // Generate stake ID
    let stake_id = Uuid::new_v4().to_string();

    // ATOMIC: Check balance AND deduct in a single lock scope to prevent TOCTOU race
    // This prevents double-spending from concurrent staking requests
    let balance_deducted = {
        let mut wallet_balances = state.wallet_balances.write().await;
        let balance = wallet_balances.get(&auth.address).copied().unwrap_or(0);

        if balance < amount {
            warn!("❌ [QNO] Insufficient balance for stake: {} < {}", balance, amount);
            false
        } else {
            // Deduct balance atomically within the same lock
            if let Some(bal) = wallet_balances.get_mut(&auth.address) {
                *bal = bal.saturating_sub(amount);
            }
            true
        }
    };

    if !balance_deducted {
        return Ok(Json(ApiResponse::error("Insufficient QUG balance".to_string())));
    }

    // Create position (after successful balance deduction)
    let position = StakingPosition {
        id: stake_id.clone(),
        wallet_address: address_hex.clone(),
        domain: request.domain.clone(),
        domain_name: domain.name.clone(),
        amount,
        confidence: request.confidence,
        lock_days: request.lock_days,
        lock_multiplier,
        staked_at: now,
        unlocks_at: unlock_timestamp,
        reward: 0,
        accrued_reward: 0,
        status: "active".to_string(),
        prediction_accuracy: 0.0,
    };

    // Persist balance deduction
    {
        let balances = state.wallet_balances.read().await;
        if let Some(&new_balance) = balances.get(&auth.address) {
            drop(balances);
            if let Err(e) = state.save_wallet_balance(&auth.address, new_balance).await {
                warn!("⚠️ [QNO] Failed to persist balance deduction: {}", e);
            }
        }
    }

    // Check if this is a new validator for the domain
    let existing_positions = storage.get_positions(&address_hex).await;
    let is_new_validator = !existing_positions.iter().any(|p| p.domain == request.domain);

    // Store position to disk
    if let Err(e) = storage.add_position(position).await {
        error!("❌ [QNO] Failed to store position: {}", e);
        // Refund balance on failure
        let mut wallet_balances = state.wallet_balances.write().await;
        if let Some(bal) = wallet_balances.get_mut(&auth.address) {
            *bal = bal.saturating_add(amount);
        }
        return Ok(Json(ApiResponse::error("Failed to create stake".to_string())));
    }

    // Update domain stats
    let validator_delta = if is_new_validator { 1 } else { 0 };
    if let Err(e) = storage.update_domain_stats(&request.domain, amount as i64, validator_delta).await {
        warn!("⚠️ [QNO] Failed to update domain stats: {}", e);
    }

    // Create prediction record for resolution system (v1.4.3-beta)
    if let Err(e) = storage.create_prediction_record(
        &stake_id,
        &request.domain,
        &address_hex,
        request.prediction_value,
        request.confidence,
    ).await {
        warn!("⚠️ [QNO] Failed to create prediction record: {}", e);
        // Non-fatal: stake is still valid, resolution may not work for this stake
    } else {
        info!("📊 [QNO] Prediction record created: {} predicts {:.4} for {}",
              stake_id, request.prediction_value, request.domain);
    }

    // Broadcast via P2P gossip (if available)
    // v2.4.9-beta: Sign QNO operations with node signing key
    if let Some(ref cmd_tx) = state.libp2p_command_tx {
        use ed25519_dalek::Signer;

        // Build the stake operation payload
        let stake_payload = serde_json::to_vec(&QnoOperation::Stake {
            position: StakingPosition {
                id: stake_id.clone(),
                wallet_address: address_hex.clone(),
                domain: request.domain.clone(),
                domain_name: domain.name.clone(),
                amount,
                confidence: request.confidence,
                lock_days: request.lock_days,
                lock_multiplier,
                staked_at: now,
                unlocks_at: unlock_timestamp,
                reward: 0,
                accrued_reward: 0,
                status: "active".to_string(),
                prediction_accuracy: 0.0,
            },
            signature: vec![], // Outer message has cryptographic signature
            timestamp: now,
        }).unwrap_or_default();

        // v2.4.9-beta: Sign stake payload with node signing key
        let signature = state.node_signing_key.sign(&stake_payload);
        let public_key = state.node_signing_key.verifying_key().to_bytes().to_vec();

        debug!("🔐 Signed QNO stake {} with Ed25519 ({} bytes)",
               stake_id, signature.to_bytes().len());

        let qno_message = q_network::distributed_qno::QnoGossipMessage::new_stake(
            hex::encode(state.node_id),
            state.libp2p_peer_info.read().await.0.clone(),
            &stake_payload,
            Some(signature.to_bytes().to_vec()),
            Some(public_key),
        );

        let _ = cmd_tx.send(q_network::NetworkCommand::PublishQnoOperation {
            topic: q_network::distributed_qno::TOPIC_QNO_STAKES.to_string(),
            message: qno_message,
        });
        debug!("📤 [QNO P2P] Broadcast stake operation for {} in {}", stake_id, request.domain);
    }

    // Format unlock date
    let unlock_date = chrono::DateTime::from_timestamp(unlock_timestamp as i64, 0)
        .map(|dt| dt.format("%Y-%m-%dT%H:%M:%SZ").to_string())
        .unwrap_or_else(|| "Unknown".to_string());

    info!(
        "✅ [QNO] Stake created: {} QUG in {} (APY: {:.1}%)",
        format_amount(amount),
        domain.name,
        expected_apy
    );

    Ok(Json(ApiResponse::success(StakeResponse {
        stake_id,
        domain: request.domain,
        amount: format_amount(amount),
        confidence: request.confidence,
        lock_days: request.lock_days,
        expected_apy,
        unlock_date,
        prediction_value: request.prediction_value,
    })))
}

/// POST /api/v1/qno/unstake - Early withdrawal with penalty (AUTHENTICATED)
pub async fn unstake_prediction(
    auth: AuthenticatedWallet,
    State(state): State<Arc<AppState>>,
    Json(request): Json<UnstakeRequest>,
) -> Result<Json<ApiResponse<UnstakeResponse>>, StatusCode> {
    let address_hex = hex::encode(auth.address);
    info!("🔓 [QNO] Processing unstake request for stake: {}", request.stake_id);

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    // Get QNO storage
    let qno = state.qno_storage.read().await;
    let storage = match qno.as_ref() {
        Some(s) => s,
        None => {
            return Ok(Json(ApiResponse::error("QNO system not initialized".to_string())));
        }
    };

    // Find the position
    let positions = storage.get_positions(&address_hex).await;
    let position = match positions.iter().find(|p| p.id == request.stake_id) {
        Some(p) => p.clone(),
        None => {
            warn!("❌ [QNO] Position not found: {}", request.stake_id);
            return Ok(Json(ApiResponse::error("Staking position not found".to_string())));
        }
    };

    // Check if already claimed
    if position.status == "claimed" {
        return Ok(Json(ApiResponse::error("Position already claimed".to_string())));
    }

    // Calculate penalty
    let days_remaining = if now >= position.unlocks_at {
        0
    } else {
        (position.unlocks_at - now) / (24 * 60 * 60)
    };

    let penalty_percentage = calculate_unstake_penalty(position.lock_days, days_remaining);
    // Use basis points (1/10000) to avoid f64 precision loss with large amounts
    let penalty_bps = (penalty_percentage * 10_000.0) as u128;
    let penalty_amount = position.amount.saturating_mul(penalty_bps) / 10_000;
    let principal_returned = position.amount.saturating_sub(penalty_amount);

    // Remove position from storage
    if let Err(e) = storage.remove_position(&address_hex, &request.stake_id).await {
        error!("❌ [QNO] Failed to remove position: {}", e);
        return Ok(Json(ApiResponse::error("Failed to process unstake".to_string())));
    }

    // Update domain stats
    if let Err(e) = storage.update_domain_stats(&position.domain, -(position.amount as i64), -1).await {
        warn!("⚠️ [QNO] Failed to update domain stats: {}", e);
    }

    // Return principal minus penalty to wallet
    {
        let mut wallet_balances = state.wallet_balances.write().await;
        let balance = wallet_balances.entry(auth.address).or_insert(0);
        *balance += principal_returned;
    }

    // Persist balance
    {
        let balances = state.wallet_balances.read().await;
        if let Some(&new_balance) = balances.get(&auth.address) {
            drop(balances);
            if let Err(e) = state.save_wallet_balance(&auth.address, new_balance).await {
                warn!("⚠️ [QNO] Failed to persist balance: {}", e);
            }
        }
    }

    // Broadcast unstake via P2P gossip
    if let Some(ref cmd_tx) = state.libp2p_command_tx {
        let qno_message = q_network::distributed_qno::QnoGossipMessage::new_unstake(
            hex::encode(state.node_id),
            state.libp2p_peer_info.read().await.0.clone(),
            &serde_json::to_vec(&QnoOperation::Unstake {
                wallet_address: address_hex.clone(),
                stake_id: request.stake_id.clone(),
                penalty_amount,
                signature: vec![],
                timestamp: now,
            }).unwrap_or_default(),
            None,
            None,
        );

        let _ = cmd_tx.send(q_network::NetworkCommand::PublishQnoOperation {
            topic: q_network::distributed_qno::TOPIC_QNO_STAKES.to_string(),
            message: qno_message,
        });
        debug!("📤 [QNO P2P] Broadcast unstake operation for {}", request.stake_id);
    }

    let message = if penalty_percentage > 0.0 {
        format!(
            "Early unstake successful. {:.1}% penalty applied ({} days remaining).",
            penalty_percentage * 100.0,
            days_remaining
        )
    } else {
        "Unstake successful. Lock period completed, no penalty applied.".to_string()
    };

    info!(
        "✅ [QNO] Unstake successful: {} QUG returned (penalty: {} QUG, {:.1}%)",
        format_amount(principal_returned),
        format_amount(penalty_amount),
        penalty_percentage * 100.0
    );

    Ok(Json(ApiResponse::success(UnstakeResponse {
        success: true,
        principal_returned: format_amount(principal_returned),
        penalty_amount: format_amount(penalty_amount),
        penalty_percentage: penalty_percentage * 100.0,
        message,
    })))
}

/// POST /api/v1/qno/claim - Claim rewards from an unlocked position (AUTHENTICATED)
pub async fn claim_reward(
    auth: AuthenticatedWallet,
    State(state): State<Arc<AppState>>,
    Json(request): Json<ClaimRequest>,
) -> Result<Json<ApiResponse<ClaimResponse>>, StatusCode> {
    let address_hex = hex::encode(auth.address);
    info!("💰 [QNO] Processing claim request for stake: {}", request.stake_id);

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    // Get QNO storage
    let qno = state.qno_storage.read().await;
    let storage = match qno.as_ref() {
        Some(s) => s,
        None => {
            return Ok(Json(ApiResponse::error("QNO system not initialized".to_string())));
        }
    };

    // Find the position
    let positions = storage.get_positions(&address_hex).await;
    let position = match positions.iter().find(|p| p.id == request.stake_id) {
        Some(p) => p.clone(),
        None => {
            warn!("❌ [QNO] Position not found: {}", request.stake_id);
            return Ok(Json(ApiResponse::error("Staking position not found".to_string())));
        }
    };

    // Check if already claimed
    if position.status == "claimed" {
        return Ok(Json(ApiResponse::error("Reward already claimed".to_string())));
    }

    // Check if unlocked
    if now < position.unlocks_at {
        let remaining_days = (position.unlocks_at - now) / (24 * 60 * 60);
        return Ok(Json(ApiResponse::error(format!(
            "Position still locked for {} days. Use unstake for early withdrawal.",
            remaining_days
        ))));
    }

    // Get domain APY for final reward calculation
    let domain_apy = storage.get_domain(&position.domain).await
        .map(|d| d.apy)
        .unwrap_or(15.0);

    // Calculate final reward based on actual time staked
    // Use accrued_reward if available, otherwise calculate
    let time_staked_days = (position.unlocks_at - position.staked_at) / (24 * 60 * 60);
    let time_fraction = time_staked_days as f64 / 365.0;

    // Simulated prediction accuracy (75-95%)
    // In production, this would come from QNO consensus based on actual predictions
    let prediction_accuracy = 0.75 + (now % 20) as f64 / 100.0;

    let reward_rate = (domain_apy / 100.0)
        * time_fraction
        * position.confidence
        * prediction_accuracy
        * position.lock_multiplier;

    // Use basis points (1/10000000 for high precision) to avoid f64 precision loss
    let reward_bps = (reward_rate * 10_000_000.0) as u128;
    let final_reward = position.amount.saturating_mul(reward_bps) / 10_000_000;
    let total_return = position.amount + final_reward;

    // Update position to claimed
    let updated = storage.update_position(&address_hex, &request.stake_id, |p| {
        p.status = "claimed".to_string();
        p.reward = final_reward;
        p.prediction_accuracy = prediction_accuracy;
    }).await;

    if let Err(e) = updated {
        error!("❌ [QNO] Failed to update position: {}", e);
        return Ok(Json(ApiResponse::error("Failed to process claim".to_string())));
    }

    // Update domain stats
    if let Err(e) = storage.update_domain_stats(&position.domain, -(position.amount as i64), 0).await {
        warn!("⚠️ [QNO] Failed to update domain stats: {}", e);
    }

    // Add to total rewards paid
    if let Err(e) = storage.add_rewards_paid(final_reward).await {
        warn!("⚠️ [QNO] Failed to update rewards paid: {}", e);
    }

    // Return principal + reward to wallet
    {
        let mut wallet_balances = state.wallet_balances.write().await;
        let balance = wallet_balances.entry(auth.address).or_insert(0);
        *balance += total_return;
    }

    // Persist balance
    {
        let balances = state.wallet_balances.read().await;
        if let Some(&new_balance) = balances.get(&auth.address) {
            drop(balances);
            if let Err(e) = state.save_wallet_balance(&auth.address, new_balance).await {
                warn!("⚠️ [QNO] Failed to persist balance: {}", e);
            }
        }
    }

    // Broadcast claim via P2P gossip
    if let Some(ref cmd_tx) = state.libp2p_command_tx {
        let qno_message = q_network::distributed_qno::QnoGossipMessage::new_claim(
            hex::encode(state.node_id),
            state.libp2p_peer_info.read().await.0.clone(),
            &serde_json::to_vec(&QnoOperation::Claim {
                wallet_address: address_hex.clone(),
                stake_id: request.stake_id.clone(),
                reward_amount: final_reward,
                principal_returned: position.amount,
                signature: vec![],
                timestamp: now,
            }).unwrap_or_default(),
            None,
            None,
        );

        let _ = cmd_tx.send(q_network::NetworkCommand::PublishQnoOperation {
            topic: q_network::distributed_qno::TOPIC_QNO_CLAIMS.to_string(),
            message: qno_message,
        });
        debug!("📤 [QNO P2P] Broadcast claim operation for {}", request.stake_id);
    }

    // Generate transaction hash
    let tx_hash = format!("qno_{}", Uuid::new_v4().to_string().replace("-", ""));

    info!(
        "✅ [QNO] Claim successful: {} QUG principal + {} QUG reward (accuracy: {:.1}%)",
        format_amount(position.amount),
        format_amount(final_reward),
        prediction_accuracy * 100.0
    );

    Ok(Json(ApiResponse::success(ClaimResponse {
        success: true,
        principal_amount: format_amount(position.amount),
        reward_amount: format_amount(final_reward),
        total_returned: format_amount(total_return),
        prediction_accuracy: prediction_accuracy * 100.0,
        transaction_hash: tx_hash,
    })))
}

/// GET /api/v1/qno/domain/:domain_id - Get specific domain details
pub async fn get_domain_details(
    Path(domain_id): Path<String>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<PredictionDomain>>, StatusCode> {
    info!("🔍 [QNO] Fetching domain details: {}", domain_id);

    let qno = state.qno_storage.read().await;
    if let Some(storage) = qno.as_ref() {
        match storage.get_domain(&domain_id).await {
            Some(domain) => Ok(Json(ApiResponse::success(domain))),
            None => Ok(Json(ApiResponse::error("Domain not found".to_string()))),
        }
    } else {
        Ok(Json(ApiResponse::error("QNO system not initialized".to_string())))
    }
}

/// GET /api/v1/qno/leaderboard - Get top stakers leaderboard
pub async fn get_leaderboard(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<Vec<LeaderboardEntry>>>, StatusCode> {
    info!("🏆 [QNO] Fetching leaderboard");

    let qno = state.qno_storage.read().await;
    if let Some(storage) = qno.as_ref() {
        let positions = storage.get_all_positions().await;

        // Aggregate by address
        let mut aggregated: HashMap<String, (u128, u128, f64, u32)> = HashMap::new();

        for (addr, user_positions) in positions.iter() {
            let total_staked: u128 = user_positions.iter().map(|p| p.amount).sum();
            let total_rewards: u128 = user_positions.iter().map(|p| p.reward + p.accrued_reward).sum();
            let accuracy_sum: f64 = user_positions.iter()
                .filter(|p| p.prediction_accuracy > 0.0)
                .map(|p| p.prediction_accuracy)
                .sum();
            let accuracy_count = user_positions.iter()
                .filter(|p| p.prediction_accuracy > 0.0)
                .count() as u32;

            aggregated.insert(addr.clone(), (total_staked, total_rewards, accuracy_sum, accuracy_count));
        }

        // Sort by total staked and take top 10
        let mut sorted: Vec<_> = aggregated.into_iter().collect();
        sorted.sort_by(|a, b| b.1.0.cmp(&a.1.0));

        let leaderboard: Vec<LeaderboardEntry> = sorted
            .into_iter()
            .take(10)
            .enumerate()
            .map(|(i, (addr, (staked, rewards, acc_sum, count)))| {
                let avg_accuracy = if count > 0 { (acc_sum / count as f64) * 100.0 } else { 0.0 };
                let truncated_addr = if addr.len() > 10 {
                    format!("{}...{}", &addr[..6], &addr[addr.len()-4..])
                } else {
                    addr.clone()
                };
                LeaderboardEntry {
                    rank: (i + 1) as u32,
                    address: truncated_addr,
                    total_staked: format_amount(staked),
                    total_rewards: format_amount(rewards),
                    accuracy: avg_accuracy,
                }
            })
            .collect();

        Ok(Json(ApiResponse::success(leaderboard)))
    } else {
        Ok(Json(ApiResponse::success(vec![])))
    }
}

// ============================================================================
// Background Tasks
// ============================================================================

/// Start background reward accrual task
/// Should be called from main server startup
pub fn start_reward_accrual_task(qno_storage: Arc<RwLock<Option<Arc<QnoStorage>>>>) {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(60)); // Every minute

        loop {
            interval.tick().await;

            let storage = qno_storage.read().await;
            if let Some(s) = storage.as_ref() {
                match s.accrue_rewards().await {
                    Ok(count) => {
                        if count > 0 {
                            debug!("📈 [QNO] Accrued rewards for {} positions", count);
                        }
                    }
                    Err(e) => {
                        warn!("⚠️ [QNO] Failed to accrue rewards: {}", e);
                    }
                }
            }
        }
    });

    info!("🚀 [QNO] Started background reward accrual task");
}

// ============================================================================
// P2P Integration (Placeholder for gossipsub)
// ============================================================================

/// Gossipsub topic for QNO operations
pub const QNO_GOSSIP_TOPIC: &str = "/qnk/qno/operations/1.0.0";

/// Validate and apply a QNO operation received from P2P gossip
pub async fn handle_p2p_qno_operation(
    qno_storage: &Arc<QnoStorage>,
    operation: QnoOperation,
) -> bool {
    match qno_storage.validate_and_apply_operation(&operation).await {
        Ok(applied) => {
            if applied {
                debug!("✅ [QNO P2P] Applied operation from peer");
            }
            applied
        }
        Err(e) => {
            warn!("❌ [QNO P2P] Failed to apply operation: {}", e);
            false
        }
    }
}

// ============================================================================
// Prediction Resolution API Types
// ============================================================================

// Re-export resolution types from qno_storage
pub use q_storage::qno_storage::{
    PredictionOutcome, OutcomeType, PredictionRecord, SlashingRecord, SlashReason, ResolutionConfig,
};

/// Request to submit an oracle outcome
#[derive(Debug, Deserialize)]
pub struct OracleOutcomeRequest {
    pub domain: String,
    pub outcome_type: String,       // "GasFee", "BlockTime", etc.
    pub actual_value: f64,
    pub confidence_threshold: f64,  // Default 0.90
    pub oracle_signature: String,   // Hex-encoded signature
}

/// Response after processing oracle outcome
#[derive(Debug, Serialize)]
pub struct OracleOutcomeResponse {
    pub outcome_id: String,
    pub domain: String,
    pub predictions_resolved: u32,
    pub average_accuracy: f64,
    pub slashings_applied: u32,
}

/// Response for slashing history
#[derive(Debug, Serialize)]
pub struct SlashingHistoryResponse {
    pub total_slashings: u32,
    pub total_slashed_amount: String,
    pub records: Vec<SlashingRecordDisplay>,
}

/// Display-friendly slashing record
#[derive(Debug, Serialize)]
pub struct SlashingRecordDisplay {
    pub id: String,
    pub stake_id: String,
    pub domain: String,
    pub reason: String,
    pub slash_amount: String,
    pub slash_percentage: f64,
    pub timestamp: String,
}

/// Prediction record status response
#[derive(Debug, Serialize)]
pub struct PredictionStatusResponse {
    pub stake_id: String,
    pub domain: String,
    pub prediction_value: f64,
    pub resolved: bool,
    pub accuracy_score: Option<f64>,
    pub failure_count: u32,
}

// ============================================================================
// Prediction Resolution API Endpoints
// ============================================================================

/// POST /api/v1/qno/oracle/outcome - Submit oracle outcome (ORACLE ONLY)
/// This endpoint should be protected by oracle authentication in production
pub async fn submit_oracle_outcome(
    State(state): State<Arc<AppState>>,
    Json(request): Json<OracleOutcomeRequest>,
) -> Result<Json<ApiResponse<OracleOutcomeResponse>>, StatusCode> {
    info!("🔮 [QNO Oracle] Processing outcome for domain: {}", request.domain);

    // Get QNO storage
    let qno = state.qno_storage.read().await;
    let storage = match qno.as_ref() {
        Some(s) => s,
        None => {
            return Ok(Json(ApiResponse::error("QNO system not initialized".to_string())));
        }
    };

    // Validate domain exists
    let domain = match storage.get_domain(&request.domain).await {
        Some(d) => d,
        None => {
            warn!("❌ [QNO Oracle] Invalid domain: {}", request.domain);
            return Ok(Json(ApiResponse::error("Invalid prediction domain".to_string())));
        }
    };

    // Parse outcome type
    let outcome_type = match request.outcome_type.to_lowercase().as_str() {
        "gasfee" | "gas_fee" | "gas-fee" => OutcomeType::GasFee,
        "blocktime" | "block_time" | "block-time" => OutcomeType::BlockTime,
        "networkload" | "network_load" | "network-load" => OutcomeType::NetworkLoad,
        "validatoruptime" | "validator_uptime" | "validator-uptime" => OutcomeType::ValidatorUptime,
        "crosschain" | "cross_chain" | "cross-chain" => OutcomeType::CrossChain,
        "defitvl" | "defi_tvl" | "defi-tvl" => OutcomeType::DefiTvl,
        other => OutcomeType::Custom(other.to_string()),
    };

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    // Create outcome
    let outcome_id = uuid::Uuid::new_v4().to_string();
    let outcome = PredictionOutcome {
        id: outcome_id.clone(),
        domain: request.domain.clone(),
        outcome_type,
        predicted_value: 0.0, // Not applicable for oracle submission
        actual_value: request.actual_value,
        timestamp: now,
        confidence_threshold: request.confidence_threshold,
        oracle_signature: hex::decode(&request.oracle_signature).unwrap_or_default(),
    };

    // Store outcome
    if let Err(e) = storage.store_prediction_outcome(outcome.clone()).await {
        error!("❌ [QNO Oracle] Failed to store outcome: {}", e);
        return Ok(Json(ApiResponse::error("Failed to store oracle outcome".to_string())));
    }

    // Resolve all pending predictions for this domain
    let results: Vec<(String, f64, Option<SlashingRecord>)> = match storage.resolve_domain_predictions(&outcome).await {
        Ok(r) => r,
        Err(e) => {
            warn!("⚠️ [QNO Oracle] Failed to resolve predictions: {}", e);
            vec![]
        }
    };

    let predictions_resolved = results.len() as u32;
    let slashings_applied = results.iter().filter(|(_, _, s)| s.is_some()).count() as u32;
    let average_accuracy = if results.is_empty() {
        0.0
    } else {
        results.iter().map(|(_, acc, _)| *acc).sum::<f64>() / results.len() as f64
    };

    info!(
        "✅ [QNO Oracle] Outcome {} resolved {} predictions (avg accuracy: {:.1}%, slashings: {})",
        outcome_id, predictions_resolved, average_accuracy * 100.0, slashings_applied
    );

    // Broadcast oracle outcome and resolution results via P2P (v1.4.3)
    if let Some(ref cmd_tx) = state.libp2p_command_tx {
        // Broadcast oracle outcome
        let outcome_payload = serde_json::json!({
            "outcome_id": outcome_id,
            "domain": request.domain,
            "actual_value": request.actual_value,
            "timestamp": now,
        });
        let oracle_message = q_network::distributed_qno::QnoGossipMessage::new_oracle_outcome(
            hex::encode(state.node_id),
            state.libp2p_peer_info.read().await.0.clone(),
            &serde_json::to_vec(&outcome_payload).unwrap_or_default(),
            None,
            None,
        );
        let _ = cmd_tx.send(q_network::NetworkCommand::PublishQnoOperation {
            topic: q_network::distributed_qno::TOPIC_QNO_ORACLE.to_string(),
            message: oracle_message,
        });

        // Broadcast resolution results
        let resolution_payload = serde_json::json!({
            "outcome_id": outcome_id,
            "domain": request.domain,
            "predictions_resolved": predictions_resolved,
            "average_accuracy": average_accuracy,
            "slashings_applied": slashings_applied,
            "results": results.iter().map(|(stake_id, acc, slash)| {
                serde_json::json!({
                    "stake_id": stake_id,
                    "accuracy": acc,
                    "slashed": slash.is_some(),
                })
            }).collect::<Vec<_>>(),
        });
        let resolution_message = q_network::distributed_qno::QnoGossipMessage::new_resolution(
            hex::encode(state.node_id),
            state.libp2p_peer_info.read().await.0.clone(),
            &serde_json::to_vec(&resolution_payload).unwrap_or_default(),
            None,
            None,
        );
        let _ = cmd_tx.send(q_network::NetworkCommand::PublishQnoOperation {
            topic: q_network::distributed_qno::TOPIC_QNO_RESOLUTION.to_string(),
            message: resolution_message,
        });

        // Broadcast individual slashing events
        for (stake_id, _, slash_opt) in &results {
            if let Some(slash_record) = slash_opt {
                let slashing_payload = serde_json::json!({
                    "slash_id": slash_record.id,
                    "stake_id": stake_id,
                    "domain": slash_record.domain,
                    "slash_amount": slash_record.slash_amount,
                    "slash_percentage": slash_record.slash_percentage,
                    "reason": format!("{:?}", slash_record.slash_reason),
                    "timestamp": slash_record.timestamp,
                });
                let slashing_message = q_network::distributed_qno::QnoGossipMessage::new_slashing(
                    hex::encode(state.node_id),
                    state.libp2p_peer_info.read().await.0.clone(),
                    &serde_json::to_vec(&slashing_payload).unwrap_or_default(),
                    None,
                    None,
                );
                let _ = cmd_tx.send(q_network::NetworkCommand::PublishQnoOperation {
                    topic: q_network::distributed_qno::TOPIC_QNO_RESOLUTION.to_string(),
                    message: slashing_message,
                });
            }
        }

        debug!("📤 [QNO P2P] Broadcast oracle outcome and {} resolution results", predictions_resolved);
    }

    Ok(Json(ApiResponse::success(OracleOutcomeResponse {
        outcome_id,
        domain: request.domain,
        predictions_resolved,
        average_accuracy: average_accuracy * 100.0,
        slashings_applied,
    })))
}

/// GET /api/v1/qno/slashing/history - Get slashing history for authenticated wallet
pub async fn get_slashing_history(
    auth: AuthenticatedWallet,
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<SlashingHistoryResponse>>, StatusCode> {
    let address_hex = hex::encode(auth.address);
    info!("⚡ [QNO] Fetching slashing history for authenticated wallet");

    let qno = state.qno_storage.read().await;
    let storage = match qno.as_ref() {
        Some(s) => s,
        None => {
            return Ok(Json(ApiResponse::error("QNO system not initialized".to_string())));
        }
    };

    let records = storage.get_slashing_history(&address_hex).await;

    let total_slashed: u128 = records.iter().map(|r| r.slash_amount).sum();
    let display_records: Vec<SlashingRecordDisplay> = records.iter().map(|r| {
        SlashingRecordDisplay {
            id: r.id.clone(),
            stake_id: r.stake_id.clone(),
            domain: r.domain.clone(),
            reason: format!("{:?}", r.slash_reason),
            slash_amount: format_amount(r.slash_amount),
            slash_percentage: r.slash_percentage * 100.0,
            timestamp: chrono::DateTime::from_timestamp(r.timestamp as i64, 0)
                .map(|dt| dt.format("%Y-%m-%dT%H:%M:%SZ").to_string())
                .unwrap_or_else(|| "Unknown".to_string()),
        }
    }).collect();

    Ok(Json(ApiResponse::success(SlashingHistoryResponse {
        total_slashings: records.len() as u32,
        total_slashed_amount: format_amount(total_slashed),
        records: display_records,
    })))
}

/// GET /api/v1/qno/prediction/:stake_id - Get prediction status for a stake
pub async fn get_prediction_status(
    auth: AuthenticatedWallet,
    Path(stake_id): Path<String>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<PredictionStatusResponse>>, StatusCode> {
    let address_hex = hex::encode(auth.address);
    info!("🔍 [QNO] Fetching prediction status for stake: {}", stake_id);

    let qno = state.qno_storage.read().await;
    let storage = match qno.as_ref() {
        Some(s) => s,
        None => {
            return Ok(Json(ApiResponse::error("QNO system not initialized".to_string())));
        }
    };

    // Verify the stake belongs to this wallet
    let positions = storage.get_positions(&address_hex).await;
    if !positions.iter().any(|p| p.id == stake_id) {
        return Ok(Json(ApiResponse::error("Stake not found".to_string())));
    }

    // Get prediction record
    match storage.get_prediction_record(&stake_id).await {
        Some(record) => {
            let failure_count = storage.get_failure_count(&address_hex).await;

            Ok(Json(ApiResponse::success(PredictionStatusResponse {
                stake_id: record.stake_id,
                domain: record.domain,
                prediction_value: record.prediction_value,
                resolved: record.resolved,
                accuracy_score: record.accuracy_score.map(|a| a * 100.0),
                failure_count,
            })))
        }
        None => {
            Ok(Json(ApiResponse::error("Prediction record not found".to_string())))
        }
    }
}

/// GET /api/v1/qno/resolution/config - Get resolution configuration
pub async fn get_resolution_config(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let qno = state.qno_storage.read().await;
    if let Some(storage) = qno.as_ref() {
        let config = storage.get_resolution_config();

        let config_json = serde_json::json!({
            "accuracy_threshold": config.accuracy_threshold * 100.0,
            "slash_after_failures": config.slash_after_failures,
            "base_slash_percentage": config.base_slash_percentage * 100.0,
            "max_slash_percentage": config.max_slash_percentage * 100.0,
            "accuracy_bonus_multiplier": config.accuracy_bonus_multiplier,
            "inaccuracy_penalty_multiplier": config.inaccuracy_penalty_multiplier,
        });

        Ok(Json(ApiResponse::success(config_json)))
    } else {
        Ok(Json(ApiResponse::error("QNO system not initialized".to_string())))
    }
}

// ============================================================================
// Resolution Background Task
// ============================================================================

/// Start background resolution task that processes oracle outcomes
/// Uses OracleService to fetch real oracle data and resolve predictions
pub fn start_resolution_task(qno_storage: Arc<RwLock<Option<Arc<QnoStorage>>>>) {
    tokio::spawn(async move {
        // Wait for system to stabilize
        tokio::time::sleep(tokio::time::Duration::from_secs(30)).await;

        // Initialize oracle service (development mode for now)
        let is_production = std::env::var("Q_ORACLE_PRODUCTION").unwrap_or_default() == "true";
        let oracle_service = OracleService::new(is_production);

        info!("🔮 [QNO Resolution] Started with oracle service (production={})", is_production);

        // Resolution interval: 1 hour in production, 5 minutes in dev
        let interval_secs = if is_production { 3600 } else { 300 };
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(interval_secs));

        info!("🔮 [QNO Resolution] Resolution interval: {}s", interval_secs);

        loop {
            interval.tick().await;

            let storage_guard = qno_storage.read().await;
            if let Some(storage) = storage_guard.as_ref() {
                info!("🔮 [QNO Resolution] Fetching oracle data for all domains...");

                // Step 1: Update all oracle feeds
                let feed_results = oracle_service.update_all_feeds().await;
                let successful_feeds = feed_results.iter()
                    .filter(|(_, r)| r.is_ok())
                    .count();

                info!("📊 [QNO Resolution] Updated {}/{} oracle feeds",
                      successful_feeds, feed_results.len());

                // Step 2: Create prediction outcomes from oracle data
                let outcomes = oracle_service.create_all_outcomes().await;

                if outcomes.is_empty() {
                    warn!("⚠️ [QNO Resolution] No valid outcomes to resolve");
                    continue;
                }

                info!("🔮 [QNO Resolution] Resolving {} domain predictions...", outcomes.len());

                // Step 3: Resolve predictions for each domain
                let mut total_resolved = 0;
                let mut total_slashed = 0;

                for outcome in outcomes {
                    match storage.resolve_domain_predictions(&outcome).await {
                        Ok(results) => {
                            // Results are tuples: (stake_id: String, accuracy_score: f64, slashing: Option<SlashingRecord>)
                            let resolved_count = results.len();
                            let slashed_count = results.iter()
                                .filter(|(_, _, slashing)| slashing.is_some())
                                .count();

                            total_resolved += resolved_count;
                            total_slashed += slashed_count;

                            if resolved_count > 0 {
                                info!("✅ [QNO Resolution] Domain '{}': {} resolved, {} slashed | Actual: {:.4}",
                                      outcome.domain, resolved_count, slashed_count, outcome.actual_value);
                            }

                            // Log individual resolution details for debugging
                            for (stake_id, accuracy_score, slashing) in results {
                                let slash_amount = slashing.map(|s| s.slash_amount).unwrap_or(0);
                                debug!("📊 [QNO Resolution] Stake {} -> accuracy: {:.2}%, slash: {}",
                                       stake_id,
                                       accuracy_score * 100.0,
                                       slash_amount);
                            }
                        }
                        Err(e) => {
                            error!("❌ [QNO Resolution] Failed to resolve '{}': {}", outcome.domain, e);
                        }
                    }
                }

                info!("🔮 [QNO Resolution] Cycle complete: {} predictions resolved, {} slashed",
                      total_resolved, total_slashed);
            } else {
                debug!("🔮 [QNO Resolution] Storage not initialized yet, waiting...");
            }
        }
    });
}
