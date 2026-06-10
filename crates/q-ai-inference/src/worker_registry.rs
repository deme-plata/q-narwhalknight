//! Worker Registry + On-Chain Staking for Decentralized AI Inference
//!
//! v6.0.0: Implements permissionless worker registration with economic skin-in-the-game.
//! Workers stake QUG to join the inference network. Staking makes cheating irrational
//! because the slashing penalty exceeds any gain from serving garbage.
//!
//! ## Staking Tiers (from staking_security.rs)
//!
//! | Tier     | Minimum Stake | Task Priority | Max Concurrent |
//! |----------|---------------|---------------|----------------|
//! | Bronze   | 1,000 QUG     | Low           | 2              |
//! | Silver   | 10,000 QUG    | Medium        | 5              |
//! | Gold     | 100,000 QUG   | High          | 10             |
//! | Diamond  | 1,000,000 QUG | Highest       | 20             |
//!
//! ## Staking Flow
//!
//! ```text
//! 1. Worker calls POST /api/v1/ai/stake { amount: 1000 }
//! 2. amount deducted from wallet_balances → credited to staked_balances
//! 3. Worker registered in WorkerRegistry with capabilities
//! 4. Worker can now accept inference requests
//! 5. If slashed: stake reduced, reputation damaged
//! 6. Unstake: 7-day cooldown, then stake returned to wallet
//! ```

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

/// Minimum stake to register as an inference worker (1,000 QUG in 24-decimal)
pub const MIN_STAKE_BRONZE: u128 = 1_000 * 10u128.pow(24);
/// Silver tier minimum (10,000 QUG)
pub const MIN_STAKE_SILVER: u128 = 10_000 * 10u128.pow(24);
/// Gold tier minimum (100,000 QUG)
pub const MIN_STAKE_GOLD: u128 = 100_000 * 10u128.pow(24);
/// Diamond tier minimum (1,000,000 QUG)
pub const MIN_STAKE_DIAMOND: u128 = 1_000_000 * 10u128.pow(24);

/// Unstaking cooldown period in blocks (~7 days at 10s/block)
pub const UNSTAKE_COOLDOWN_BLOCKS: u64 = 60_480;

/// Slash percentage (10% of stake per offense)
pub const SLASH_PERCENTAGE: u64 = 10;

/// Staking tier determines task priority and max concurrent requests
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum StakingTier {
    None = 0,
    Bronze = 1,
    Silver = 2,
    Gold = 3,
    Diamond = 4,
}

impl StakingTier {
    pub fn from_stake(amount: u128) -> Self {
        if amount >= MIN_STAKE_DIAMOND {
            StakingTier::Diamond
        } else if amount >= MIN_STAKE_GOLD {
            StakingTier::Gold
        } else if amount >= MIN_STAKE_SILVER {
            StakingTier::Silver
        } else if amount >= MIN_STAKE_BRONZE {
            StakingTier::Bronze
        } else {
            StakingTier::None
        }
    }

    pub fn max_concurrent(&self) -> usize {
        match self {
            StakingTier::None => 0,
            StakingTier::Bronze => 2,
            StakingTier::Silver => 5,
            StakingTier::Gold => 10,
            StakingTier::Diamond => 20,
        }
    }

    /// Stake-weighted score bonus for task routing (log2 scale)
    pub fn score_bonus(&self) -> u64 {
        match self {
            StakingTier::None => 0,
            StakingTier::Bronze => 0,
            StakingTier::Silver => 10,
            StakingTier::Gold => 20,
            StakingTier::Diamond => 30,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            StakingTier::None => "None",
            StakingTier::Bronze => "Bronze",
            StakingTier::Silver => "Silver",
            StakingTier::Gold => "Gold",
            StakingTier::Diamond => "Diamond",
        }
    }
}

/// On-chain worker registration with staking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerRegistration {
    /// Worker's wallet address
    pub worker_address: [u8; 32],
    /// Amount of QUG staked (24-decimal)
    pub stake_amount: u128,
    /// Current staking tier
    pub tier: StakingTier,
    /// SHA3-256 hash of the model this worker serves
    pub model_hash: [u8; 32],
    /// Human-readable model name
    pub model_name: String,
    /// Block height when registered
    pub registered_at: u64,
    /// Reputation score (0.0 to 1.0)
    pub reputation: f64,
    /// Total inferences completed
    pub total_inferences: u64,
    /// Total verification challenges passed
    pub total_challenges_passed: u64,
    /// Total verification challenges failed
    pub total_challenges_failed: u64,
    /// Total amount slashed from this worker
    pub total_slashed: u128,
    /// Estimated inference latency (ms)
    pub avg_latency_ms: u64,
    /// Worker's announced price per token (24-decimal QUG)
    pub price_per_token: u128,
    /// Whether the worker is currently active (responding to heartbeats)
    pub is_active: bool,
    /// Last heartbeat block height
    pub last_heartbeat: u64,
    /// Peer ID for P2P communication
    pub peer_id: String,
}

/// Pending unstake request with cooldown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnstakeRequest {
    pub worker_address: [u8; 32],
    pub amount: u128,
    pub requested_at_block: u64,
    pub available_at_block: u64,
}

/// Worker Registry managing all registered inference workers
pub struct WorkerRegistry {
    /// Registered workers: worker_address -> registration
    workers: Arc<RwLock<HashMap<[u8; 32], WorkerRegistration>>>,
    /// Staked balances: worker_address -> staked amount
    staked_balances: Arc<RwLock<HashMap<[u8; 32], u128>>>,
    /// Pending unstake requests
    unstake_requests: Arc<RwLock<Vec<UnstakeRequest>>>,
    /// Current block height (updated by block producer)
    current_height: Arc<RwLock<u64>>,
}

impl WorkerRegistry {
    pub fn new() -> Self {
        info!("🏭 Initializing Worker Registry for decentralized AI inference");
        Self {
            workers: Arc::new(RwLock::new(HashMap::new())),
            staked_balances: Arc::new(RwLock::new(HashMap::new())),
            unstake_requests: Arc::new(RwLock::new(Vec::new())),
            current_height: Arc::new(RwLock::new(0)),
        }
    }

    /// Update the current block height
    pub async fn set_height(&self, height: u64) {
        *self.current_height.write().await = height;
    }

    /// Register a worker with a stake. Returns the worker's tier.
    ///
    /// The caller is responsible for debiting the wallet balance BEFORE calling this.
    pub async fn register_worker(
        &self,
        worker_address: [u8; 32],
        stake_amount: u128,
        model_hash: [u8; 32],
        model_name: String,
        peer_id: String,
        price_per_token: u128,
    ) -> Result<StakingTier> {
        let tier = StakingTier::from_stake(stake_amount);
        if tier == StakingTier::None {
            return Err(anyhow!(
                "Stake amount {} below minimum {} (Bronze tier)",
                stake_amount,
                MIN_STAKE_BRONZE
            ));
        }

        let height = *self.current_height.read().await;

        let registration = WorkerRegistration {
            worker_address,
            stake_amount,
            tier,
            model_hash,
            model_name: model_name.clone(),
            registered_at: height,
            reputation: 1.0, // Start with perfect reputation
            total_inferences: 0,
            total_challenges_passed: 0,
            total_challenges_failed: 0,
            total_slashed: 0,
            avg_latency_ms: 0,
            price_per_token,
            is_active: true,
            last_heartbeat: height,
            peer_id: peer_id.clone(),
        };

        self.workers.write().await.insert(worker_address, registration);
        *self.staked_balances.write().await.entry(worker_address).or_insert(0) += stake_amount;

        info!(
            "🔒 Worker registered: peer={}, model={}, stake={} QUG, tier={}",
            peer_id,
            model_name,
            stake_amount / 10u128.pow(24),
            tier.name()
        );

        Ok(tier)
    }

    /// Add more stake to an existing worker registration
    pub async fn add_stake(&self, worker_address: [u8; 32], additional: u128) -> Result<StakingTier> {
        let mut workers = self.workers.write().await;
        let worker = workers.get_mut(&worker_address)
            .ok_or_else(|| anyhow!("Worker not registered"))?;

        worker.stake_amount += additional;
        worker.tier = StakingTier::from_stake(worker.stake_amount);

        *self.staked_balances.write().await.entry(worker_address).or_insert(0) += additional;

        Ok(worker.tier)
    }

    /// Request unstaking (starts cooldown period)
    pub async fn request_unstake(&self, worker_address: [u8; 32], amount: u128) -> Result<UnstakeRequest> {
        let mut workers = self.workers.write().await;
        let worker = workers.get_mut(&worker_address)
            .ok_or_else(|| anyhow!("Worker not registered"))?;

        if amount > worker.stake_amount {
            return Err(anyhow!("Unstake amount {} exceeds stake {}", amount, worker.stake_amount));
        }

        let remaining = worker.stake_amount - amount;
        if remaining > 0 && StakingTier::from_stake(remaining) == StakingTier::None {
            return Err(anyhow!(
                "Remaining stake {} below minimum. Unstake full amount or keep at least {} (Bronze)",
                remaining, MIN_STAKE_BRONZE
            ));
        }

        let height = *self.current_height.read().await;
        let request = UnstakeRequest {
            worker_address,
            amount,
            requested_at_block: height,
            available_at_block: height + UNSTAKE_COOLDOWN_BLOCKS,
        };

        worker.stake_amount -= amount;
        worker.tier = StakingTier::from_stake(worker.stake_amount);
        if worker.stake_amount == 0 {
            worker.is_active = false;
        }

        self.unstake_requests.write().await.push(request.clone());

        info!(
            "🔓 Unstake requested: {} QUG, available at block {}",
            amount / 10u128.pow(24),
            request.available_at_block
        );

        Ok(request)
    }

    /// Process matured unstake requests. Returns (worker_address, amount) pairs to credit.
    pub async fn process_unstakes(&self) -> Vec<([u8; 32], u128)> {
        let height = *self.current_height.read().await;
        let mut requests = self.unstake_requests.write().await;
        let mut staked = self.staked_balances.write().await;

        let mut matured = Vec::new();
        requests.retain(|req| {
            if req.available_at_block <= height {
                if let Some(balance) = staked.get_mut(&req.worker_address) {
                    *balance = balance.saturating_sub(req.amount);
                }
                matured.push((req.worker_address, req.amount));
                false // Remove from pending
            } else {
                true // Keep
            }
        });

        matured
    }

    /// Slash a worker's stake (called by opML verifier on dispute loss)
    pub async fn slash_worker(&self, worker_address: [u8; 32], reason: &str) -> Result<u128> {
        let mut workers = self.workers.write().await;
        let worker = workers.get_mut(&worker_address)
            .ok_or_else(|| anyhow!("Worker not registered"))?;

        let slash_amount = worker.stake_amount * SLASH_PERCENTAGE as u128 / 100;
        worker.stake_amount -= slash_amount;
        worker.total_slashed += slash_amount;
        worker.total_challenges_failed += 1;
        worker.tier = StakingTier::from_stake(worker.stake_amount);

        // Decay reputation on slash
        worker.reputation = (worker.reputation * 0.8).max(0.0);

        if let Some(balance) = self.staked_balances.write().await.get_mut(&worker_address) {
            *balance = balance.saturating_sub(slash_amount);
        }

        warn!(
            "⚡ Worker slashed: {} QUG (reason: {}), remaining stake: {} QUG, reputation: {:.2}",
            slash_amount / 10u128.pow(24),
            reason,
            worker.stake_amount / 10u128.pow(24),
            worker.reputation
        );

        Ok(slash_amount)
    }

    /// Record a successful inference completion
    pub async fn record_inference_success(&self, worker_address: [u8; 32], latency_ms: u64) {
        if let Some(worker) = self.workers.write().await.get_mut(&worker_address) {
            worker.total_inferences += 1;
            // Exponential moving average for latency
            if worker.avg_latency_ms == 0 {
                worker.avg_latency_ms = latency_ms;
            } else {
                worker.avg_latency_ms = (worker.avg_latency_ms * 9 + latency_ms) / 10;
            }
            // Slowly recover reputation (max 1.0)
            worker.reputation = (worker.reputation + 0.001).min(1.0);
        }
    }

    /// Record a passed verification challenge
    pub async fn record_challenge_passed(&self, worker_address: [u8; 32]) {
        if let Some(worker) = self.workers.write().await.get_mut(&worker_address) {
            worker.total_challenges_passed += 1;
            worker.reputation = (worker.reputation + 0.005).min(1.0);
        }
    }

    /// Update worker heartbeat
    pub async fn heartbeat(&self, worker_address: [u8; 32]) {
        let height = *self.current_height.read().await;
        if let Some(worker) = self.workers.write().await.get_mut(&worker_address) {
            worker.last_heartbeat = height;
            worker.is_active = true;
        }
    }

    /// Mark workers as inactive if no heartbeat for 100 blocks
    pub async fn prune_inactive(&self) {
        let height = *self.current_height.read().await;
        let mut workers = self.workers.write().await;
        for worker in workers.values_mut() {
            if worker.is_active && height.saturating_sub(worker.last_heartbeat) > 100 {
                worker.is_active = false;
            }
        }
    }

    /// Get all active workers for a specific model
    pub async fn get_workers_for_model(&self, model_hash: &[u8; 32]) -> Vec<WorkerRegistration> {
        self.workers.read().await.values()
            .filter(|w| w.is_active && &w.model_hash == model_hash)
            .cloned()
            .collect()
    }

    /// Get all active workers regardless of model
    pub async fn get_active_workers(&self) -> Vec<WorkerRegistration> {
        self.workers.read().await.values()
            .filter(|w| w.is_active)
            .cloned()
            .collect()
    }

    /// Get a specific worker's registration
    pub async fn get_worker(&self, address: &[u8; 32]) -> Option<WorkerRegistration> {
        self.workers.read().await.get(address).cloned()
    }

    /// Get the staked balance for a worker
    pub async fn get_staked_balance(&self, address: &[u8; 32]) -> u128 {
        self.staked_balances.read().await.get(address).copied().unwrap_or(0)
    }

    /// Get total staked across all workers
    pub async fn total_staked(&self) -> u128 {
        self.staked_balances.read().await.values().sum()
    }

    /// Get registry stats for API response
    pub async fn get_stats(&self) -> WorkerRegistryStats {
        let workers = self.workers.read().await;
        let active = workers.values().filter(|w| w.is_active).count();
        let total_staked: u128 = workers.values().map(|w| w.stake_amount).sum();
        let total_inferences: u64 = workers.values().map(|w| w.total_inferences).sum();

        WorkerRegistryStats {
            total_workers: workers.len(),
            active_workers: active,
            total_staked,
            total_inferences,
            tier_distribution: TierDistribution {
                bronze: workers.values().filter(|w| w.tier == StakingTier::Bronze).count(),
                silver: workers.values().filter(|w| w.tier == StakingTier::Silver).count(),
                gold: workers.values().filter(|w| w.tier == StakingTier::Gold).count(),
                diamond: workers.values().filter(|w| w.tier == StakingTier::Diamond).count(),
            },
        }
    }
}

/// Registry statistics for monitoring/API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerRegistryStats {
    pub total_workers: usize,
    pub active_workers: usize,
    pub total_staked: u128,
    pub total_inferences: u64,
    pub tier_distribution: TierDistribution,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierDistribution {
    pub bronze: usize,
    pub silver: usize,
    pub gold: usize,
    pub diamond: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_staking_tiers() {
        assert_eq!(StakingTier::from_stake(0), StakingTier::None);
        assert_eq!(StakingTier::from_stake(999 * 10u128.pow(24)), StakingTier::None);
        assert_eq!(StakingTier::from_stake(1_000 * 10u128.pow(24)), StakingTier::Bronze);
        assert_eq!(StakingTier::from_stake(10_000 * 10u128.pow(24)), StakingTier::Silver);
        assert_eq!(StakingTier::from_stake(100_000 * 10u128.pow(24)), StakingTier::Gold);
        assert_eq!(StakingTier::from_stake(1_000_000 * 10u128.pow(24)), StakingTier::Diamond);
    }

    #[tokio::test]
    async fn test_register_and_slash() {
        let registry = WorkerRegistry::new();
        let addr = [1u8; 32];
        let model = [2u8; 32];

        let tier = registry.register_worker(
            addr, MIN_STAKE_BRONZE, model, "test-model".into(), "peer1".into(), 100
        ).await.unwrap();
        assert_eq!(tier, StakingTier::Bronze);

        let worker = registry.get_worker(&addr).await.unwrap();
        assert_eq!(worker.reputation, 1.0);

        let slashed = registry.slash_worker(addr, "test").await.unwrap();
        assert_eq!(slashed, MIN_STAKE_BRONZE / 10); // 10%

        let worker = registry.get_worker(&addr).await.unwrap();
        assert!(worker.reputation < 1.0);
        assert_eq!(worker.stake_amount, MIN_STAKE_BRONZE - slashed);
    }

    #[tokio::test]
    async fn test_unstake_cooldown() {
        let registry = WorkerRegistry::new();
        let addr = [1u8; 32];
        let model = [2u8; 32];

        registry.register_worker(
            addr, MIN_STAKE_BRONZE, model, "test".into(), "peer1".into(), 100
        ).await.unwrap();

        let request = registry.request_unstake(addr, MIN_STAKE_BRONZE).await.unwrap();
        assert_eq!(request.available_at_block, UNSTAKE_COOLDOWN_BLOCKS);

        // Before cooldown: nothing to process
        let matured = registry.process_unstakes().await;
        assert!(matured.is_empty());

        // After cooldown
        registry.set_height(UNSTAKE_COOLDOWN_BLOCKS).await;
        let matured = registry.process_unstakes().await;
        assert_eq!(matured.len(), 1);
        assert_eq!(matured[0].1, MIN_STAKE_BRONZE);
    }
}
