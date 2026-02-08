//! QNO (Quantum Neural Oracle) Storage Module
//!
//! Provides persistent storage for QNO prediction staking:
//! - Staking positions
//! - Domain statistics
//! - Global stats
//! - P2P gossip integration for decentralized validation
//!
//! v3.2.2: Added u128_serde for MessagePack P2P compatibility

use anyhow::{anyhow, Result};
#[cfg(not(target_os = "windows"))]
use rocksdb::DB;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

// Import u128_serde for MessagePack compatibility
use q_types::{u128_serde, option_u128_serde};

use crate::{CF_QNO_DOMAINS, CF_QNO_STAKES, CF_QNO_STATS};

// ============================================================================
// Types (shared with qno_api.rs)
// ============================================================================

/// Prediction domain for staking
/// v3.2.2: Added u128_serde for MessagePack P2P compatibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionDomain {
    pub id: String,
    pub name: String,
    pub description: String,
    pub apy: f64,
    pub risk_level: String,
    pub total_staked: u128,
    pub validator_count: u32,
    pub accuracy_30d: f64,
    pub min_stake: u128,
    pub max_stake: u128,
    pub active: bool,
}

/// Staking position for a user
/// v3.2.2: Added u128_serde for MessagePack P2P compatibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StakingPosition {
    pub id: String,
    pub wallet_address: String,  // Added for P2P sync
    pub domain: String,
    pub domain_name: String,
    pub amount: u128,
    pub confidence: f64,
    pub lock_days: u32,
    pub lock_multiplier: f64,
    pub staked_at: u64,
    pub unlocks_at: u64,
    pub reward: u128,
    pub accrued_reward: u128,    // Continuously accrued reward
    pub status: String,
    pub prediction_accuracy: f64,
}

/// Overall staking statistics
/// v3.2.2: Added u128_serde for MessagePack P2P compatibility
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StakingStats {
    #[serde(default)]
    pub total_staked: u128,
    pub total_stakers: u32,
    pub average_apy: f64,
    #[serde(default)]
    pub total_rewards_paid: u128,
    pub active_domains: u32,
    pub prediction_accuracy_global: f64,
}

/// QNO operation for P2P gossip
/// v3.2.2: Added u128_serde for MessagePack P2P compatibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QnoOperation {
    Stake {
        position: StakingPosition,
        signature: Vec<u8>,
        timestamp: u64,
    },
    Unstake {
        wallet_address: String,
        stake_id: String,
        penalty_amount: u128,
        signature: Vec<u8>,
        timestamp: u64,
    },
    Claim {
        wallet_address: String,
        stake_id: String,
        reward_amount: u128,
        principal_returned: u128,
        signature: Vec<u8>,
        timestamp: u64,
    },
    RewardAccrual {
        stake_id: String,
        accrued_amount: u128,
        timestamp: u64,
    },
}

// ============================================================================
// Prediction Resolution Types
// ============================================================================

/// Represents an oracle-provided outcome for a prediction domain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionOutcome {
    pub id: String,                     // Unique outcome ID
    pub domain: String,                 // Domain this outcome belongs to
    pub outcome_type: OutcomeType,      // Type of outcome measurement
    pub predicted_value: f64,           // What was predicted (e.g., gas price)
    pub actual_value: f64,              // What actually happened
    pub timestamp: u64,                 // When outcome was recorded
    pub confidence_threshold: f64,      // Threshold for "correct" prediction
    pub oracle_signature: Vec<u8>,      // Oracle's signature for verification
}

/// Types of prediction outcomes
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OutcomeType {
    GasFee,           // Network gas fees prediction
    BlockTime,        // Block confirmation time
    NetworkLoad,      // Network congestion level
    ValidatorUptime,  // Validator availability
    CrossChain,       // Cross-chain transaction success
    DefiTvl,          // DeFi Total Value Locked trends
    Custom(String),   // Custom domain
}

/// Record linking a stake to its prediction performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionRecord {
    pub stake_id: String,
    pub domain: String,
    pub wallet_address: String,
    pub prediction_value: f64,          // What the staker predicted
    pub confidence: f64,                // Staker's confidence (from stake)
    pub created_at: u64,
    pub resolved: bool,                 // Whether outcome has been resolved
    pub resolution_timestamp: Option<u64>,
    pub accuracy_score: Option<f64>,    // 0.0 to 1.0, None if unresolved
    pub outcome_id: Option<String>,     // Reference to PredictionOutcome
}

/// Slashing event record
/// v3.2.2: Added u128_serde for MessagePack P2P compatibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlashingRecord {
    pub id: String,
    pub stake_id: String,
    pub wallet_address: String,
    pub domain: String,
    pub slash_reason: SlashReason,
    pub slash_amount: u128,             // Amount slashed (in base units)
    pub slash_percentage: f64,          // Percentage of stake slashed
    pub consecutive_failures: u32,      // How many failures led to this
    pub timestamp: u64,
}

/// Reasons for slashing
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SlashReason {
    ConsecutiveWrongPredictions(u32),   // N consecutive wrong predictions
    SevereInaccuracy,                   // Prediction was way off (>50% error)
    MaliciousPattern,                   // Detected gaming/manipulation
    ValidatorMisbehavior,               // General misbehavior
}

/// Resolution configuration
#[derive(Debug, Clone)]
pub struct ResolutionConfig {
    /// Accuracy threshold for "correct" prediction (default: 0.90 = 10% error margin)
    pub accuracy_threshold: f64,
    /// Consecutive failures before slashing
    pub slash_after_failures: u32,
    /// Base slash percentage for wrong predictions
    pub base_slash_percentage: f64,
    /// Maximum slash percentage cap
    pub max_slash_percentage: f64,
    /// Bonus multiplier for highly accurate predictions
    pub accuracy_bonus_multiplier: f64,
    /// Penalty multiplier for inaccurate predictions
    pub inaccuracy_penalty_multiplier: f64,
}

impl Default for ResolutionConfig {
    fn default() -> Self {
        Self {
            accuracy_threshold: 0.90,           // 10% error margin
            slash_after_failures: 3,            // Slash after 3 consecutive failures
            base_slash_percentage: 0.05,        // 5% base slash
            max_slash_percentage: 0.25,         // Max 25% slash
            accuracy_bonus_multiplier: 1.5,     // 50% bonus for accurate predictions
            inaccuracy_penalty_multiplier: 0.5, // 50% penalty for inaccurate
        }
    }
}

// ============================================================================
// QNO Storage Engine
// ============================================================================

pub struct QnoStorage {
    db: Arc<DB>,
    // In-memory cache for fast reads
    positions_cache: RwLock<HashMap<String, Vec<StakingPosition>>>,
    domains_cache: RwLock<Vec<PredictionDomain>>,
    stats_cache: RwLock<StakingStats>,
    initialized: RwLock<bool>,

    // Prediction Resolution caches
    /// Prediction outcomes from oracle (domain -> Vec<outcome>)
    outcomes_cache: RwLock<HashMap<String, Vec<PredictionOutcome>>>,
    /// Prediction records (stake_id -> record)
    prediction_records_cache: RwLock<HashMap<String, PredictionRecord>>,
    /// Consecutive failure count per wallet (wallet -> count)
    failure_counts: RwLock<HashMap<String, u32>>,
    /// Slashing records (wallet -> Vec<slashing>)
    slashing_history: RwLock<HashMap<String, Vec<SlashingRecord>>>,
    /// Resolution configuration
    resolution_config: ResolutionConfig,
}

impl QnoStorage {
    pub fn new(db: Arc<DB>) -> Self {
        Self {
            db,
            positions_cache: RwLock::new(HashMap::new()),
            domains_cache: RwLock::new(Vec::new()),
            stats_cache: RwLock::new(StakingStats::default()),
            initialized: RwLock::new(false),
            // Prediction Resolution
            outcomes_cache: RwLock::new(HashMap::new()),
            prediction_records_cache: RwLock::new(HashMap::new()),
            failure_counts: RwLock::new(HashMap::new()),
            slashing_history: RwLock::new(HashMap::new()),
            resolution_config: ResolutionConfig::default(),
        }
    }

    /// Create with custom resolution config
    pub fn new_with_config(db: Arc<DB>, config: ResolutionConfig) -> Self {
        Self {
            db,
            positions_cache: RwLock::new(HashMap::new()),
            domains_cache: RwLock::new(Vec::new()),
            stats_cache: RwLock::new(StakingStats::default()),
            initialized: RwLock::new(false),
            outcomes_cache: RwLock::new(HashMap::new()),
            prediction_records_cache: RwLock::new(HashMap::new()),
            failure_counts: RwLock::new(HashMap::new()),
            slashing_history: RwLock::new(HashMap::new()),
            resolution_config: config,
        }
    }

    /// Initialize QNO storage - load from disk or create defaults
    pub async fn initialize(&self) -> Result<()> {
        let mut initialized = self.initialized.write().await;
        if *initialized {
            return Ok(());
        }

        info!("🔮 [QNO Storage] Initializing QNO storage...");

        // Load domains from disk or create defaults
        self.load_or_create_domains().await?;

        // Load all positions from disk
        self.load_all_positions().await?;

        // Load or compute stats
        self.load_or_compute_stats().await?;

        *initialized = true;
        info!("✅ [QNO Storage] Initialization complete");
        Ok(())
    }

    /// Load domains from disk or create default domains
    async fn load_or_create_domains(&self) -> Result<()> {
        let cf = self.db.cf_handle(CF_QNO_DOMAINS)
            .ok_or_else(|| anyhow!("CF_QNO_DOMAINS not found"))?;

        let mut domains = Vec::new();
        let iter = self.db.iterator_cf(&cf, rocksdb::IteratorMode::Start);

        for item in iter {
            match item {
                Ok((_, value)) => {
                    if let Ok(domain) = serde_json::from_slice::<PredictionDomain>(&value) {
                        domains.push(domain);
                    }
                }
                Err(e) => {
                    warn!("Error reading domain: {}", e);
                }
            }
        }

        if domains.is_empty() {
            info!("🔮 [QNO Storage] No domains found, creating defaults...");
            domains = self.create_default_domains();

            // Persist default domains
            for domain in &domains {
                self.save_domain(domain).await?;
            }
        }

        info!("🔮 [QNO Storage] Loaded {} prediction domains", domains.len());
        *self.domains_cache.write().await = domains;
        Ok(())
    }

    /// Create default prediction domains
    fn create_default_domains(&self) -> Vec<PredictionDomain> {
        vec![
            PredictionDomain {
                id: "gas-fees".to_string(),
                name: "Gas Fee Prediction".to_string(),
                description: "Predict network gas fees for next 24 hours".to_string(),
                apy: 12.5,
                risk_level: "low".to_string(),
                total_staked: 0,
                validator_count: 0,
                accuracy_30d: 0.0,
                min_stake: 10_000_000,     // 0.1 QUG
                max_stake: 10_000_000_000_000,
                active: true,
            },
            PredictionDomain {
                id: "block-time".to_string(),
                name: "Block Time Estimation".to_string(),
                description: "Predict average block confirmation times".to_string(),
                apy: 18.3,
                risk_level: "medium".to_string(),
                total_staked: 0,
                validator_count: 0,
                accuracy_30d: 0.0,
                min_stake: 50_000_000,
                max_stake: 50_000_000_000_000,
                active: true,
            },
            PredictionDomain {
                id: "network-load".to_string(),
                name: "Network Load Forecast".to_string(),
                description: "Forecast network congestion levels".to_string(),
                apy: 15.7,
                risk_level: "medium".to_string(),
                total_staked: 0,
                validator_count: 0,
                accuracy_30d: 0.0,
                min_stake: 25_000_000,
                max_stake: 25_000_000_000_000,
                active: true,
            },
            PredictionDomain {
                id: "validator-uptime".to_string(),
                name: "Validator Uptime".to_string(),
                description: "Predict validator availability and performance".to_string(),
                apy: 22.1,
                risk_level: "high".to_string(),
                total_staked: 0,
                validator_count: 0,
                accuracy_30d: 0.0,
                min_stake: 100_000_000,
                max_stake: 100_000_000_000_000,
                active: true,
            },
            PredictionDomain {
                id: "cross-chain".to_string(),
                name: "Cross-Chain Bridge".to_string(),
                description: "Predict cross-chain transaction success rates".to_string(),
                apy: 28.5,
                risk_level: "high".to_string(),
                total_staked: 0,
                validator_count: 0,
                accuracy_30d: 0.0,
                min_stake: 250_000_000,
                max_stake: 200_000_000_000_000,
                active: true,
            },
            PredictionDomain {
                id: "defi-tvl".to_string(),
                name: "DeFi TVL Trends".to_string(),
                description: "Predict Total Value Locked movements".to_string(),
                apy: 35.2,
                risk_level: "high".to_string(),
                total_staked: 0,
                validator_count: 0,
                accuracy_30d: 0.0,
                min_stake: 500_000_000,
                max_stake: 500_000_000_000_000,
                active: true,
            },
        ]
    }

    /// Load all positions from disk
    async fn load_all_positions(&self) -> Result<()> {
        let cf = self.db.cf_handle(CF_QNO_STAKES)
            .ok_or_else(|| anyhow!("CF_QNO_STAKES not found"))?;

        let mut positions: HashMap<String, Vec<StakingPosition>> = HashMap::new();
        let iter = self.db.iterator_cf(&cf, rocksdb::IteratorMode::Start);

        let mut count = 0;
        for item in iter {
            match item {
                Ok((key, value)) => {
                    if let Ok(position) = serde_json::from_slice::<StakingPosition>(&value) {
                        let wallet = position.wallet_address.clone();
                        positions.entry(wallet).or_insert_with(Vec::new).push(position);
                        count += 1;
                    }
                }
                Err(e) => {
                    warn!("Error reading position: {}", e);
                }
            }
        }

        info!("🔮 [QNO Storage] Loaded {} staking positions for {} wallets",
              count, positions.len());
        *self.positions_cache.write().await = positions;
        Ok(())
    }

    /// Load or compute global stats
    async fn load_or_compute_stats(&self) -> Result<()> {
        let cf = self.db.cf_handle(CF_QNO_STATS)
            .ok_or_else(|| anyhow!("CF_QNO_STATS not found"))?;

        // Try to load from disk
        if let Some(data) = self.db.get_cf(&cf, b"global")? {
            if let Ok(stats) = serde_json::from_slice::<StakingStats>(&data) {
                *self.stats_cache.write().await = stats;
                return Ok(());
            }
        }

        // Compute from positions
        self.recompute_stats().await?;
        Ok(())
    }

    /// Recompute stats from current positions
    pub async fn recompute_stats(&self) -> Result<()> {
        let positions = self.positions_cache.read().await;
        let domains = self.domains_cache.read().await;

        let total_stakers = positions.len() as u32;
        let total_staked: u128 = positions
            .values()
            .flat_map(|p| p.iter())
            .filter(|p| p.status == "active")
            .map(|p| p.amount)
            .sum();

        let active_domains: Vec<_> = domains.iter().filter(|d| d.active).collect();
        let average_apy = if active_domains.is_empty() {
            0.0
        } else {
            active_domains.iter().map(|d| d.apy).sum::<f64>() / active_domains.len() as f64
        };

        let stats = {
            let current = self.stats_cache.read().await;
            StakingStats {
                total_staked,
                total_stakers,
                average_apy,
                total_rewards_paid: current.total_rewards_paid,
                active_domains: active_domains.len() as u32,
                prediction_accuracy_global: current.prediction_accuracy_global,
            }
        };

        *self.stats_cache.write().await = stats.clone();
        self.save_stats(&stats).await?;
        Ok(())
    }

    // ========================================================================
    // Persistence Operations
    // ========================================================================

    /// Save a staking position to disk
    pub async fn save_position(&self, position: &StakingPosition) -> Result<()> {
        let cf = self.db.cf_handle(CF_QNO_STAKES)
            .ok_or_else(|| anyhow!("CF_QNO_STAKES not found"))?;

        let key = format!("{}:{}", position.wallet_address, position.id);
        let value = serde_json::to_vec(position)?;
        self.db.put_cf(&cf, key.as_bytes(), &value)?;

        debug!("💾 [QNO] Saved position {} for wallet {}",
               position.id, &position.wallet_address[..8.min(position.wallet_address.len())]);
        Ok(())
    }

    /// Delete a staking position from disk
    pub async fn delete_position(&self, wallet_address: &str, stake_id: &str) -> Result<()> {
        let cf = self.db.cf_handle(CF_QNO_STAKES)
            .ok_or_else(|| anyhow!("CF_QNO_STAKES not found"))?;

        let key = format!("{}:{}", wallet_address, stake_id);
        self.db.delete_cf(&cf, key.as_bytes())?;

        debug!("🗑️ [QNO] Deleted position {} for wallet {}", stake_id, wallet_address);
        Ok(())
    }

    /// Save a domain to disk
    pub async fn save_domain(&self, domain: &PredictionDomain) -> Result<()> {
        let cf = self.db.cf_handle(CF_QNO_DOMAINS)
            .ok_or_else(|| anyhow!("CF_QNO_DOMAINS not found"))?;

        let value = serde_json::to_vec(domain)?;
        self.db.put_cf(&cf, domain.id.as_bytes(), &value)?;
        Ok(())
    }

    /// Save global stats to disk
    pub async fn save_stats(&self, stats: &StakingStats) -> Result<()> {
        let cf = self.db.cf_handle(CF_QNO_STATS)
            .ok_or_else(|| anyhow!("CF_QNO_STATS not found"))?;

        let value = serde_json::to_vec(stats)?;
        self.db.put_cf(&cf, b"global", &value)?;
        Ok(())
    }

    // ========================================================================
    // Cache Operations (for API handlers)
    // ========================================================================

    /// Get all domains
    pub async fn get_domains(&self) -> Vec<PredictionDomain> {
        self.domains_cache.read().await.clone()
    }

    /// Get a specific domain
    pub async fn get_domain(&self, domain_id: &str) -> Option<PredictionDomain> {
        self.domains_cache.read().await
            .iter()
            .find(|d| d.id == domain_id)
            .cloned()
    }

    /// Update domain stats
    pub async fn update_domain_stats(&self, domain_id: &str, staked_delta: i64, validator_delta: i32) -> Result<()> {
        let mut domains = self.domains_cache.write().await;
        if let Some(domain) = domains.iter_mut().find(|d| d.id == domain_id) {
            if staked_delta >= 0 {
                domain.total_staked = domain.total_staked.saturating_add(staked_delta as u128);
            } else {
                domain.total_staked = domain.total_staked.saturating_sub((-staked_delta) as u128);
            }

            if validator_delta >= 0 {
                domain.validator_count = domain.validator_count.saturating_add(validator_delta as u32);
            } else {
                domain.validator_count = domain.validator_count.saturating_sub((-validator_delta) as u32);
            }

            let domain_clone = domain.clone();
            drop(domains);
            self.save_domain(&domain_clone).await?;
        }
        Ok(())
    }

    /// Get positions for a wallet
    pub async fn get_positions(&self, wallet_address: &str) -> Vec<StakingPosition> {
        self.positions_cache.read().await
            .get(wallet_address)
            .cloned()
            .unwrap_or_default()
    }

    /// Get all positions (for leaderboard, stats)
    pub async fn get_all_positions(&self) -> HashMap<String, Vec<StakingPosition>> {
        self.positions_cache.read().await.clone()
    }

    /// Add a new position
    pub async fn add_position(&self, position: StakingPosition) -> Result<()> {
        // Save to disk first
        self.save_position(&position).await?;

        // Update cache
        let wallet = position.wallet_address.clone();
        let mut cache = self.positions_cache.write().await;
        cache.entry(wallet).or_insert_with(Vec::new).push(position);

        // Recompute stats
        drop(cache);
        self.recompute_stats().await?;
        Ok(())
    }

    /// Update an existing position
    pub async fn update_position(&self, wallet_address: &str, stake_id: &str,
                                  update_fn: impl FnOnce(&mut StakingPosition)) -> Result<bool> {
        let mut cache = self.positions_cache.write().await;

        if let Some(positions) = cache.get_mut(wallet_address) {
            if let Some(position) = positions.iter_mut().find(|p| p.id == stake_id) {
                update_fn(position);
                let position_clone = position.clone();
                drop(cache);
                self.save_position(&position_clone).await?;
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Remove a position (for unstake)
    pub async fn remove_position(&self, wallet_address: &str, stake_id: &str) -> Result<Option<StakingPosition>> {
        // Delete from disk
        self.delete_position(wallet_address, stake_id).await?;

        // Remove from cache
        let mut cache = self.positions_cache.write().await;
        if let Some(positions) = cache.get_mut(wallet_address) {
            if let Some(idx) = positions.iter().position(|p| p.id == stake_id) {
                let removed = positions.remove(idx);
                drop(cache);
                self.recompute_stats().await?;
                return Ok(Some(removed));
            }
        }
        Ok(None)
    }

    /// Get global stats
    pub async fn get_stats(&self) -> StakingStats {
        self.stats_cache.read().await.clone()
    }

    /// Update total rewards paid
    pub async fn add_rewards_paid(&self, amount: u128) -> Result<()> {
        let mut stats = self.stats_cache.write().await;
        stats.total_rewards_paid = stats.total_rewards_paid.saturating_add(amount);
        let stats_clone = stats.clone();
        drop(stats);
        self.save_stats(&stats_clone).await?;
        Ok(())
    }

    // ========================================================================
    // Reward Accrual
    // ========================================================================

    /// Accrue rewards for all active positions
    /// Called periodically (e.g., every minute) by background task
    pub async fn accrue_rewards(&self) -> Result<u32> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let domains = self.get_domains().await;
        let mut updated_count = 0;

        let mut cache = self.positions_cache.write().await;

        for (wallet, positions) in cache.iter_mut() {
            for position in positions.iter_mut() {
                if position.status != "active" {
                    continue;
                }

                // Find domain APY
                let apy = domains
                    .iter()
                    .find(|d| d.id == position.domain)
                    .map(|d| d.apy)
                    .unwrap_or(15.0);

                // Calculate accrued reward since last update
                // Reward per second = (amount * APY) / (365 * 24 * 60 * 60)
                let seconds_staked = now.saturating_sub(position.staked_at);
                // Use high-precision basis points to avoid f64 precision loss
                let rate_bps = ((apy / 100.0) * position.confidence * position.lock_multiplier * 1_000_000_000.0) as u128;
                let seconds_per_year: u128 = 365 * 24 * 60 * 60;
                // expected_reward = amount * rate_bps * seconds / (1e9 * seconds_per_year)
                let expected_reward = position.amount
                    .saturating_mul(rate_bps)
                    .saturating_mul(seconds_staked as u128)
                    / (1_000_000_000 * seconds_per_year);

                if expected_reward > position.accrued_reward {
                    position.accrued_reward = expected_reward;
                    updated_count += 1;
                }

                // Update status if unlocked
                if now >= position.unlocks_at && position.status == "active" {
                    position.status = "unlocked".to_string();
                }
            }
        }

        // Persist updated positions
        if updated_count > 0 {
            for (wallet, positions) in cache.iter() {
                for position in positions {
                    if let Err(e) = self.save_position(position).await {
                        error!("Failed to save position during accrual: {}", e);
                    }
                }
            }
        }

        Ok(updated_count)
    }

    // ========================================================================
    // P2P Gossip Validation
    // ========================================================================

    /// Validate and apply a QNO operation from P2P gossip
    pub async fn validate_and_apply_operation(&self, operation: &QnoOperation) -> Result<bool> {
        match operation {
            QnoOperation::Stake { position, signature, timestamp } => {
                // Verify signature matches wallet address
                // In production, verify Ed25519 signature

                // Check timestamp is recent (within 5 minutes)
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs();

                if now.abs_diff(*timestamp) > 300 {
                    warn!("[QNO P2P] Stake operation too old: {}", timestamp);
                    return Ok(false);
                }

                // Check if position already exists (dedup)
                let existing = self.get_positions(&position.wallet_address).await;
                if existing.iter().any(|p| p.id == position.id) {
                    debug!("[QNO P2P] Position already exists: {}", position.id);
                    return Ok(false);
                }

                // Apply the stake
                self.add_position(position.clone()).await?;
                info!("✅ [QNO P2P] Applied stake from peer: {} in {}",
                      position.id, position.domain);
                Ok(true)
            }

            QnoOperation::Unstake { wallet_address, stake_id, penalty_amount, .. } => {
                if let Some(_) = self.remove_position(wallet_address, stake_id).await? {
                    info!("✅ [QNO P2P] Applied unstake from peer: {}", stake_id);
                    Ok(true)
                } else {
                    Ok(false)
                }
            }

            QnoOperation::Claim { wallet_address, stake_id, reward_amount, .. } => {
                let updated = self.update_position(wallet_address, stake_id, |p| {
                    p.status = "claimed".to_string();
                    p.reward = *reward_amount;
                }).await?;

                if updated {
                    self.add_rewards_paid(*reward_amount).await?;
                    info!("✅ [QNO P2P] Applied claim from peer: {}", stake_id);
                }
                Ok(updated)
            }

            QnoOperation::RewardAccrual { stake_id, accrued_amount, .. } => {
                // Find the position and update accrued reward
                let cache = self.positions_cache.read().await;
                let mut found_wallet: Option<String> = None;
                for (wallet, positions) in cache.iter() {
                    if positions.iter().any(|p| p.id == *stake_id) {
                        found_wallet = Some(wallet.clone());
                        break;
                    }
                }
                drop(cache);

                if let Some(wallet) = found_wallet {
                    return self.update_position(&wallet, stake_id, |p| {
                        p.accrued_reward = p.accrued_reward.max(*accrued_amount);
                    }).await;
                }
                Ok(false)
            }
        }
    }

    // ========================================================================
    // Prediction Resolution System
    // ========================================================================

    /// Store a prediction outcome from the oracle
    pub async fn store_prediction_outcome(&self, outcome: PredictionOutcome) -> Result<()> {
        info!("📊 [QNO Resolution] Storing outcome {} for domain {} (predicted={:.4}, actual={:.4})",
              outcome.id, outcome.domain, outcome.predicted_value, outcome.actual_value);

        let mut cache = self.outcomes_cache.write().await;
        cache.entry(outcome.domain.clone())
            .or_insert_with(Vec::new)
            .push(outcome);
        Ok(())
    }

    /// Get prediction outcomes for a domain
    pub async fn get_prediction_outcomes(&self, domain: &str) -> Vec<PredictionOutcome> {
        self.outcomes_cache.read().await
            .get(domain)
            .cloned()
            .unwrap_or_default()
    }

    /// Create a prediction record linking a stake to its prediction
    pub async fn create_prediction_record(&self, stake_id: &str, domain: &str,
                                           wallet_address: &str, prediction_value: f64,
                                           confidence: f64) -> Result<PredictionRecord> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let record = PredictionRecord {
            stake_id: stake_id.to_string(),
            domain: domain.to_string(),
            wallet_address: wallet_address.to_string(),
            prediction_value,
            confidence,
            created_at: now,
            resolved: false,
            resolution_timestamp: None,
            accuracy_score: None,
            outcome_id: None,
        };

        let mut cache = self.prediction_records_cache.write().await;
        cache.insert(stake_id.to_string(), record.clone());

        info!("📝 [QNO Resolution] Created prediction record for stake {} in domain {}",
              stake_id, domain);
        Ok(record)
    }

    /// Get a prediction record by stake ID
    pub async fn get_prediction_record(&self, stake_id: &str) -> Option<PredictionRecord> {
        self.prediction_records_cache.read().await
            .get(stake_id)
            .cloned()
    }

    /// Calculate accuracy score comparing predicted vs actual value
    /// Returns a score from 0.0 (completely wrong) to 1.0 (perfect prediction)
    pub fn calculate_accuracy_score(&self, predicted: f64, actual: f64) -> f64 {
        if actual == 0.0 {
            if predicted == 0.0 {
                return 1.0; // Both zero = perfect
            }
            return 0.0; // Predicted non-zero but actual was zero
        }

        // Calculate percentage error
        let error = ((predicted - actual) / actual).abs();

        // Convert error to accuracy score (exponential decay)
        // 0% error = 1.0, 10% error ≈ 0.9, 50% error ≈ 0.61, 100% error ≈ 0.37
        let accuracy = (-error).exp();

        accuracy.clamp(0.0, 1.0)
    }

    /// Check if a prediction is considered "correct" based on accuracy threshold
    pub fn is_prediction_correct(&self, accuracy_score: f64) -> bool {
        accuracy_score >= self.resolution_config.accuracy_threshold
    }

    /// Get consecutive failure count for a wallet
    pub async fn get_failure_count(&self, wallet_address: &str) -> u32 {
        self.failure_counts.read().await
            .get(wallet_address)
            .copied()
            .unwrap_or(0)
    }

    /// Increment failure count for a wallet
    pub async fn increment_failure_count(&self, wallet_address: &str) -> u32 {
        let mut counts = self.failure_counts.write().await;
        let count = counts.entry(wallet_address.to_string()).or_insert(0);
        *count += 1;
        *count
    }

    /// Reset failure count for a wallet (on successful prediction)
    pub async fn reset_failure_count(&self, wallet_address: &str) {
        let mut counts = self.failure_counts.write().await;
        counts.remove(wallet_address);
    }

    /// Calculate slash amount based on consecutive failures
    fn calculate_slash_amount(&self, stake_amount: u128, consecutive_failures: u32) -> (u128, f64) {
        // Escalating slash: base% + 5% per additional failure, capped at max%
        let failure_multiplier = (consecutive_failures.saturating_sub(
            self.resolution_config.slash_after_failures
        )) as f64;

        let slash_percentage = (self.resolution_config.base_slash_percentage
            + failure_multiplier * 0.05)
            .min(self.resolution_config.max_slash_percentage);

        // Use basis points to avoid f64 precision loss with large amounts
        let slash_bps = (slash_percentage * 10_000.0) as u128;
        let slash_amount = stake_amount.saturating_mul(slash_bps) / 10_000;
        (slash_amount, slash_percentage)
    }

    /// Apply slashing to a stake for wrong predictions
    pub async fn apply_slashing(&self, stake_id: &str, wallet_address: &str,
                                 domain: &str, slash_reason: SlashReason) -> Result<Option<SlashingRecord>> {
        let consecutive_failures = self.get_failure_count(wallet_address).await;

        // Only slash after threshold failures
        if consecutive_failures < self.resolution_config.slash_after_failures {
            debug!("[QNO Slashing] Not slashing {} - failures {} < threshold {}",
                   wallet_address, consecutive_failures, self.resolution_config.slash_after_failures);
            return Ok(None);
        }

        // Get current stake amount
        let positions = self.get_positions(wallet_address).await;
        let position = positions.iter().find(|p| p.id == stake_id);

        let stake_amount = match position {
            Some(p) => p.amount,
            None => {
                warn!("[QNO Slashing] Position {} not found for wallet {}", stake_id, wallet_address);
                return Ok(None);
            }
        };

        let (slash_amount, slash_percentage) = self.calculate_slash_amount(stake_amount, consecutive_failures);

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let record = SlashingRecord {
            id: format!("slash-{}-{}", stake_id, now),
            stake_id: stake_id.to_string(),
            wallet_address: wallet_address.to_string(),
            domain: domain.to_string(),
            slash_reason: slash_reason.clone(),
            slash_amount,
            slash_percentage,
            consecutive_failures,
            timestamp: now,
        };

        // Apply the slash to the position
        // Convert slash_percentage to basis points to avoid f64 precision loss
        let slash_bps_for_reward = (slash_percentage * 10_000.0) as u128;
        let slashed = self.update_position(wallet_address, stake_id, |p| {
            p.amount = p.amount.saturating_sub(slash_amount);
            // Also reduce accrued reward proportionally using integer math
            let reward_slash = p.accrued_reward.saturating_mul(slash_bps_for_reward) / 10_000;
            p.accrued_reward = p.accrued_reward.saturating_sub(reward_slash);
        }).await?;

        if slashed {
            // Store slashing record
            let mut history = self.slashing_history.write().await;
            history.entry(wallet_address.to_string())
                .or_insert_with(Vec::new)
                .push(record.clone());

            warn!("⚡ [QNO Slashing] Slashed {} ({:.1}%) from stake {} for {:?}",
                  slash_amount, slash_percentage * 100.0, stake_id, slash_reason);

            Ok(Some(record))
        } else {
            Ok(None)
        }
    }

    /// Get slashing history for a wallet
    pub async fn get_slashing_history(&self, wallet_address: &str) -> Vec<SlashingRecord> {
        self.slashing_history.read().await
            .get(wallet_address)
            .cloned()
            .unwrap_or_default()
    }

    /// Adjust reward based on prediction accuracy
    /// Returns the adjusted reward amount
    pub fn adjust_reward_for_accuracy(&self, base_reward: u128, accuracy_score: f64) -> u128 {
        let is_correct = self.is_prediction_correct(accuracy_score);

        if is_correct {
            // Bonus for accurate predictions
            // Higher accuracy = higher bonus (up to 50% extra)
            let bonus_multiplier = 1.0 + (accuracy_score - self.resolution_config.accuracy_threshold)
                .max(0.0) * self.resolution_config.accuracy_bonus_multiplier;
            // Use basis points to avoid f64 precision loss with large amounts
            let multiplier_bps = (bonus_multiplier * 10_000.0) as u128;
            base_reward.saturating_mul(multiplier_bps) / 10_000
        } else {
            // Penalty for inaccurate predictions
            let penalty = self.resolution_config.inaccuracy_penalty_multiplier;
            let final_multiplier = 1.0 - penalty * (1.0 - accuracy_score);
            // Use basis points to avoid f64 precision loss with large amounts
            let multiplier_bps = (final_multiplier * 10_000.0) as u128;
            base_reward.saturating_mul(multiplier_bps) / 10_000
        }
    }

    /// Resolve a prediction against an oracle outcome
    /// This is the main resolution function that:
    /// 1. Calculates accuracy score
    /// 2. Updates the prediction record
    /// 3. Adjusts rewards
    /// 4. Applies slashing if necessary
    pub async fn resolve_prediction(&self, stake_id: &str, outcome: &PredictionOutcome) -> Result<(f64, Option<SlashingRecord>)> {
        // Get the prediction record
        let record = match self.get_prediction_record(stake_id).await {
            Some(r) => r,
            None => {
                return Err(anyhow!("Prediction record not found for stake {}", stake_id));
            }
        };

        if record.resolved {
            return Err(anyhow!("Prediction {} already resolved", stake_id));
        }

        // Calculate accuracy score
        let accuracy = self.calculate_accuracy_score(record.prediction_value, outcome.actual_value);
        let is_correct = self.is_prediction_correct(accuracy);

        info!("🎯 [QNO Resolution] Resolving stake {} - predicted={:.4}, actual={:.4}, accuracy={:.2}%, correct={}",
              stake_id, record.prediction_value, outcome.actual_value, accuracy * 100.0, is_correct);

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Update prediction record
        {
            let mut cache = self.prediction_records_cache.write().await;
            if let Some(rec) = cache.get_mut(stake_id) {
                rec.resolved = true;
                rec.resolution_timestamp = Some(now);
                rec.accuracy_score = Some(accuracy);
                rec.outcome_id = Some(outcome.id.clone());
            }
        }

        // Handle failure tracking and slashing
        let slashing = if is_correct {
            // Reset failure count on success
            self.reset_failure_count(&record.wallet_address).await;
            None
        } else {
            // Increment failure count
            let failures = self.increment_failure_count(&record.wallet_address).await;

            // Check if slashing is needed
            if failures >= self.resolution_config.slash_after_failures {
                let reason = if accuracy < 0.5 {
                    SlashReason::SevereInaccuracy
                } else {
                    SlashReason::ConsecutiveWrongPredictions(failures)
                };

                self.apply_slashing(stake_id, &record.wallet_address, &record.domain, reason).await?
            } else {
                None
            }
        };

        // Update position with accuracy and adjusted rewards
        self.update_position(&record.wallet_address, stake_id, |p| {
            p.prediction_accuracy = accuracy;

            // Adjust accrued reward based on accuracy
            p.accrued_reward = self.adjust_reward_for_accuracy(p.accrued_reward, accuracy);
        }).await?;

        // Update domain accuracy stats
        self.update_domain_accuracy(&record.domain, accuracy).await?;

        Ok((accuracy, slashing))
    }

    /// Update domain accuracy (rolling average)
    async fn update_domain_accuracy(&self, domain_id: &str, new_accuracy: f64) -> Result<()> {
        let mut domains = self.domains_cache.write().await;
        if let Some(domain) = domains.iter_mut().find(|d| d.id == domain_id) {
            // Simple rolling average (could use exponential moving average)
            let current = domain.accuracy_30d;
            domain.accuracy_30d = if current == 0.0 {
                new_accuracy
            } else {
                current * 0.95 + new_accuracy * 0.05  // 5% weight to new values
            };

            let domain_clone = domain.clone();
            drop(domains);
            self.save_domain(&domain_clone).await?;
        }
        Ok(())
    }

    /// Get resolution configuration
    pub fn get_resolution_config(&self) -> &ResolutionConfig {
        &self.resolution_config
    }

    /// Resolve all pending predictions for a domain when oracle provides outcome
    pub async fn resolve_domain_predictions(&self, outcome: &PredictionOutcome) -> Result<Vec<(String, f64, Option<SlashingRecord>)>> {
        let records = self.prediction_records_cache.read().await;

        // Find all unresolved predictions for this domain
        let pending: Vec<_> = records.iter()
            .filter(|(_, r)| r.domain == outcome.domain && !r.resolved)
            .map(|(id, _)| id.clone())
            .collect();
        drop(records);

        info!("🔄 [QNO Resolution] Resolving {} pending predictions for domain {}",
              pending.len(), outcome.domain);

        let mut results = Vec::new();
        for stake_id in pending {
            match self.resolve_prediction(&stake_id, outcome).await {
                Ok((accuracy, slashing)) => {
                    results.push((stake_id, accuracy, slashing));
                }
                Err(e) => {
                    warn!("Failed to resolve prediction {}: {}", stake_id, e);
                }
            }
        }

        Ok(results)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn create_test_db() -> Arc<DB> {
        let dir = tempdir().unwrap();
        let mut opts = rocksdb::Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);

        let cfs = vec![
            rocksdb::ColumnFamilyDescriptor::new(CF_QNO_STAKES, rocksdb::Options::default()),
            rocksdb::ColumnFamilyDescriptor::new(CF_QNO_DOMAINS, rocksdb::Options::default()),
            rocksdb::ColumnFamilyDescriptor::new(CF_QNO_STATS, rocksdb::Options::default()),
        ];

        Arc::new(DB::open_cf_descriptors(&opts, dir.path(), cfs).unwrap())
    }

    #[tokio::test]
    async fn test_initialize_creates_default_domains() {
        let db = create_test_db();
        let storage = QnoStorage::new(db);

        storage.initialize().await.unwrap();

        let domains = storage.get_domains().await;
        assert_eq!(domains.len(), 6);
        assert!(domains.iter().any(|d| d.id == "gas-fees"));
    }

    #[tokio::test]
    async fn test_add_and_get_position() {
        let db = create_test_db();
        let storage = QnoStorage::new(db);
        storage.initialize().await.unwrap();

        let position = StakingPosition {
            id: "test-stake-1".to_string(),
            wallet_address: "abc123".to_string(),
            domain: "gas-fees".to_string(),
            domain_name: "Gas Fee Prediction".to_string(),
            amount: 1_000_000_000,
            confidence: 0.8,
            lock_days: 30,
            lock_multiplier: 1.5,
            staked_at: 1700000000,
            unlocks_at: 1702592000,
            reward: 0,
            accrued_reward: 0,
            status: "active".to_string(),
            prediction_accuracy: 0.0,
        };

        storage.add_position(position.clone()).await.unwrap();

        let positions = storage.get_positions("abc123").await;
        assert_eq!(positions.len(), 1);
        assert_eq!(positions[0].id, "test-stake-1");
    }

    // ========================================================================
    // Prediction Resolution Tests
    // ========================================================================

    #[tokio::test]
    async fn test_accuracy_score_calculation() {
        let db = create_test_db();
        let storage = QnoStorage::new(db);

        // Perfect prediction
        assert_eq!(storage.calculate_accuracy_score(100.0, 100.0), 1.0);

        // 10% error -> ~0.90
        let accuracy_10_pct = storage.calculate_accuracy_score(110.0, 100.0);
        assert!(accuracy_10_pct > 0.89 && accuracy_10_pct < 0.92);

        // 50% error -> ~0.61
        let accuracy_50_pct = storage.calculate_accuracy_score(150.0, 100.0);
        assert!(accuracy_50_pct > 0.58 && accuracy_50_pct < 0.65);

        // Zero actual value handling
        assert_eq!(storage.calculate_accuracy_score(0.0, 0.0), 1.0);
        assert_eq!(storage.calculate_accuracy_score(50.0, 0.0), 0.0);
    }

    #[tokio::test]
    async fn test_prediction_is_correct() {
        let db = create_test_db();
        let storage = QnoStorage::new(db);

        // Default threshold is 0.90
        assert!(storage.is_prediction_correct(0.95));
        assert!(storage.is_prediction_correct(0.90));
        assert!(!storage.is_prediction_correct(0.89));
        assert!(!storage.is_prediction_correct(0.50));
    }

    #[tokio::test]
    async fn test_failure_count_tracking() {
        let db = create_test_db();
        let storage = QnoStorage::new(db);

        let wallet = "test-wallet-123";

        // Initial count should be 0
        assert_eq!(storage.get_failure_count(wallet).await, 0);

        // Increment failures
        assert_eq!(storage.increment_failure_count(wallet).await, 1);
        assert_eq!(storage.increment_failure_count(wallet).await, 2);
        assert_eq!(storage.increment_failure_count(wallet).await, 3);

        // Verify count
        assert_eq!(storage.get_failure_count(wallet).await, 3);

        // Reset on successful prediction
        storage.reset_failure_count(wallet).await;
        assert_eq!(storage.get_failure_count(wallet).await, 0);
    }

    #[tokio::test]
    async fn test_reward_adjustment_for_accuracy() {
        let db = create_test_db();
        let storage = QnoStorage::new(db);

        let base_reward = 1_000_000;

        // High accuracy (>0.90 threshold) -> bonus
        let high_accuracy_reward = storage.adjust_reward_for_accuracy(base_reward, 0.95);
        assert!(high_accuracy_reward > base_reward);

        // Low accuracy -> penalty
        let low_accuracy_reward = storage.adjust_reward_for_accuracy(base_reward, 0.50);
        assert!(low_accuracy_reward < base_reward);

        // Perfect accuracy -> max bonus
        let perfect_reward = storage.adjust_reward_for_accuracy(base_reward, 1.0);
        assert!(perfect_reward > high_accuracy_reward);
    }

    #[tokio::test]
    async fn test_create_prediction_record() {
        let db = create_test_db();
        let storage = QnoStorage::new(db);
        storage.initialize().await.unwrap();

        let record = storage.create_prediction_record(
            "stake-001",
            "gas-fees",
            "wallet-abc",
            42.5,
            0.85,
        ).await.unwrap();

        assert_eq!(record.stake_id, "stake-001");
        assert_eq!(record.domain, "gas-fees");
        assert_eq!(record.prediction_value, 42.5);
        assert!(!record.resolved);

        // Retrieve
        let retrieved = storage.get_prediction_record("stake-001").await;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().confidence, 0.85);
    }

    #[tokio::test]
    async fn test_prediction_resolution() {
        let db = create_test_db();
        let storage = QnoStorage::new(db);
        storage.initialize().await.unwrap();

        // Create a position
        let position = StakingPosition {
            id: "stake-res-001".to_string(),
            wallet_address: "wallet-res-test".to_string(),
            domain: "gas-fees".to_string(),
            domain_name: "Gas Fee Prediction".to_string(),
            amount: 1_000_000_000,
            confidence: 0.8,
            lock_days: 30,
            lock_multiplier: 1.5,
            staked_at: 1700000000,
            unlocks_at: 1702592000,
            reward: 0,
            accrued_reward: 100_000,
            status: "active".to_string(),
            prediction_accuracy: 0.0,
        };
        storage.add_position(position).await.unwrap();

        // Create prediction record with predicted value of 100
        storage.create_prediction_record(
            "stake-res-001",
            "gas-fees",
            "wallet-res-test",
            100.0,
            0.8,
        ).await.unwrap();

        // Create oracle outcome with actual value of 95 (5% error)
        let outcome = PredictionOutcome {
            id: "outcome-001".to_string(),
            domain: "gas-fees".to_string(),
            outcome_type: OutcomeType::GasFee,
            predicted_value: 100.0,
            actual_value: 95.0,
            timestamp: 1700086400,
            confidence_threshold: 0.90,
            oracle_signature: vec![],
        };

        // Resolve
        let (accuracy, slashing) = storage.resolve_prediction("stake-res-001", &outcome).await.unwrap();

        // 5% error should give ~0.95 accuracy
        assert!(accuracy > 0.94);
        assert!(slashing.is_none()); // No slashing for accurate prediction

        // Verify position was updated
        let positions = storage.get_positions("wallet-res-test").await;
        assert_eq!(positions[0].prediction_accuracy, accuracy);
    }

    #[tokio::test]
    async fn test_slashing_after_consecutive_failures() {
        let db = create_test_db();
        let storage = QnoStorage::new(db);
        storage.initialize().await.unwrap();

        let wallet = "wallet-slash-test";

        // Create position
        let position = StakingPosition {
            id: "stake-slash-001".to_string(),
            wallet_address: wallet.to_string(),
            domain: "gas-fees".to_string(),
            domain_name: "Gas Fee Prediction".to_string(),
            amount: 1_000_000_000,
            confidence: 0.8,
            lock_days: 30,
            lock_multiplier: 1.5,
            staked_at: 1700000000,
            unlocks_at: 1702592000,
            reward: 0,
            accrued_reward: 100_000,
            status: "active".to_string(),
            prediction_accuracy: 0.0,
        };
        storage.add_position(position).await.unwrap();

        // Simulate 3 consecutive failures (threshold is 3)
        storage.increment_failure_count(wallet).await;
        storage.increment_failure_count(wallet).await;
        storage.increment_failure_count(wallet).await;

        // Apply slashing
        let result = storage.apply_slashing(
            "stake-slash-001",
            wallet,
            "gas-fees",
            SlashReason::ConsecutiveWrongPredictions(3),
        ).await.unwrap();

        assert!(result.is_some());
        let slash_record = result.unwrap();
        assert_eq!(slash_record.consecutive_failures, 3);
        assert!(slash_record.slash_amount > 0);
        assert!(slash_record.slash_percentage <= 0.25); // Max 25%

        // Verify position amount was reduced
        let positions = storage.get_positions(wallet).await;
        assert!(positions[0].amount < 1_000_000_000);

        // Verify slashing history
        let history = storage.get_slashing_history(wallet).await;
        assert_eq!(history.len(), 1);
    }

    #[tokio::test]
    async fn test_no_slashing_before_threshold() {
        let db = create_test_db();
        let storage = QnoStorage::new(db);
        storage.initialize().await.unwrap();

        let wallet = "wallet-no-slash";

        // Create position
        let position = StakingPosition {
            id: "stake-no-slash-001".to_string(),
            wallet_address: wallet.to_string(),
            domain: "gas-fees".to_string(),
            domain_name: "Gas Fee Prediction".to_string(),
            amount: 1_000_000_000,
            confidence: 0.8,
            lock_days: 30,
            lock_multiplier: 1.5,
            staked_at: 1700000000,
            unlocks_at: 1702592000,
            reward: 0,
            accrued_reward: 100_000,
            status: "active".to_string(),
            prediction_accuracy: 0.0,
        };
        storage.add_position(position).await.unwrap();

        // Only 2 failures (threshold is 3)
        storage.increment_failure_count(wallet).await;
        storage.increment_failure_count(wallet).await;

        // Should not slash
        let result = storage.apply_slashing(
            "stake-no-slash-001",
            wallet,
            "gas-fees",
            SlashReason::ConsecutiveWrongPredictions(2),
        ).await.unwrap();

        assert!(result.is_none());

        // Position should be unchanged
        let positions = storage.get_positions(wallet).await;
        assert_eq!(positions[0].amount, 1_000_000_000);
    }
}
