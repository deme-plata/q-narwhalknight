//! PPLNS (Pay Per Last N Shares) reward calculator

use chrono::{DateTime, Utc};
use indexmap::IndexMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;

use crate::config::PPLNSConfig;
use crate::share::Share;
use crate::worker::WorkerId;
use crate::{DEV_FEE_BPS, DEFAULT_POOL_FEE_BPS};

/// PPLNS share entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PPLNSShare {
    /// Worker who submitted the share
    pub worker_id: WorkerId,

    /// Share difficulty
    pub difficulty: f64,

    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Reward payout entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardEntry {
    /// Worker ID
    pub worker_id: WorkerId,

    /// Wallet address
    pub wallet_address: String,

    /// Reward amount (atomic units)
    pub amount: u64,

    /// Proportion of total (0.0 to 1.0)
    pub proportion: f64,

    /// Difficulty contribution
    pub difficulty_contribution: f64,
}

/// Round information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Round {
    /// Round number
    pub round_id: u64,

    /// Block height found
    pub block_height: u64,

    /// Block hash
    pub block_hash: [u8; 32],

    /// Total block reward
    pub block_reward: u64,

    /// Dev fee
    pub dev_fee: u64,

    /// Pool fee
    pub pool_fee: u64,

    /// Miner rewards
    pub miner_rewards: u64,

    /// Individual payouts
    pub payouts: Vec<RewardEntry>,

    /// Worker who found the block
    pub found_by: WorkerId,

    /// Timestamp
    pub timestamp: DateTime<Utc>,

    /// Total shares in round
    pub total_shares: u64,

    /// Total difficulty in round
    pub total_difficulty: f64,
}

/// PPLNS calculator
pub struct PPLNSCalculator {
    /// Configuration
    config: PPLNSConfig,

    /// Pool fee in basis points
    pool_fee_bps: u64,

    /// Share window (rolling)
    shares: RwLock<VecDeque<PPLNSShare>>,

    /// Total difficulty in window
    total_difficulty: RwLock<f64>,

    /// Current network difficulty
    network_difficulty: RwLock<f64>,

    /// Round counter
    round_counter: RwLock<u64>,

    /// Round history
    rounds: RwLock<VecDeque<Round>>,

    /// Maximum rounds to keep in history
    max_round_history: usize,
}

impl PPLNSCalculator {
    /// Create new PPLNS calculator
    pub fn new(config: PPLNSConfig, pool_fee_bps: u64) -> Self {
        Self {
            config,
            pool_fee_bps,
            shares: RwLock::new(VecDeque::new()),
            total_difficulty: RwLock::new(0.0),
            network_difficulty: RwLock::new(1.0),
            round_counter: RwLock::new(0),
            rounds: RwLock::new(VecDeque::new()),
            max_round_history: 100,
        }
    }

    /// Add share to window
    pub fn add_share(&self, share: Share) {
        let pplns_share = PPLNSShare {
            worker_id: share.worker_id,
            difficulty: share.difficulty,
            timestamp: share.timestamp,
        };

        let mut shares = self.shares.write();
        let mut total_diff = self.total_difficulty.write();

        shares.push_back(pplns_share);
        *total_diff += share.difficulty;

        // Trim window if needed
        self.trim_window(&mut shares, &mut total_diff);
    }

    /// Trim share window to N shares worth of difficulty
    fn trim_window(&self, shares: &mut VecDeque<PPLNSShare>, total_diff: &mut f64) {
        let network_diff = *self.network_difficulty.read();
        let window_size = network_diff * self.config.n_factor;

        // Remove oldest shares until within window
        while *total_diff > window_size && !shares.is_empty() {
            if let Some(old_share) = shares.pop_front() {
                *total_diff -= old_share.difficulty;
            }
        }

        // Also enforce memory limit
        while shares.len() > self.config.max_shares_in_memory {
            if let Some(old_share) = shares.pop_front() {
                *total_diff -= old_share.difficulty;
            }
        }
    }

    /// Update network difficulty
    pub fn update_network_difficulty(&self, difficulty: f64) {
        *self.network_difficulty.write() = difficulty;
    }

    /// Calculate rewards for a block
    pub fn calculate_rewards(
        &self,
        block_reward: u64,
        block_height: u64,
        block_hash: [u8; 32],
        found_by: WorkerId,
        wallet_lookup: impl Fn(&WorkerId) -> String,
    ) -> Round {
        let shares = self.shares.read();
        let total_diff = *self.total_difficulty.read();

        // Calculate fees
        let dev_fee = block_reward * DEV_FEE_BPS / 10_000;
        let pool_fee = block_reward * self.pool_fee_bps / 10_000;
        let miner_rewards = block_reward - dev_fee - pool_fee;

        // Aggregate difficulty per worker
        let mut worker_difficulty: IndexMap<WorkerId, f64> = IndexMap::new();
        for share in shares.iter() {
            *worker_difficulty.entry(share.worker_id.clone()).or_insert(0.0) += share.difficulty;
        }

        // Calculate proportional rewards
        let mut payouts = Vec::new();
        for (worker_id, difficulty) in &worker_difficulty {
            let proportion = if total_diff > 0.0 {
                difficulty / total_diff
            } else {
                0.0
            };

            let amount = (miner_rewards as f64 * proportion) as u64;

            if amount > 0 {
                payouts.push(RewardEntry {
                    worker_id: worker_id.clone(),
                    wallet_address: wallet_lookup(worker_id),
                    amount,
                    proportion,
                    difficulty_contribution: *difficulty,
                });
            }
        }

        // Sort by amount descending
        payouts.sort_by(|a, b| b.amount.cmp(&a.amount));

        // Increment round counter
        let round_id = {
            let mut counter = self.round_counter.write();
            *counter += 1;
            *counter
        };

        let round = Round {
            round_id,
            block_height,
            block_hash,
            block_reward,
            dev_fee,
            pool_fee,
            miner_rewards,
            payouts,
            found_by,
            timestamp: Utc::now(),
            total_shares: shares.len() as u64,
            total_difficulty: total_diff,
        };

        // Store in history
        {
            let mut rounds = self.rounds.write();
            rounds.push_back(round.clone());
            while rounds.len() > self.max_round_history {
                rounds.pop_front();
            }
        }

        tracing::info!(
            round = round_id,
            height = block_height,
            reward = block_reward,
            miners = round.payouts.len(),
            "Round completed, rewards calculated"
        );

        round
    }

    /// Start new round (after block found)
    pub fn new_round(&self) {
        // Don't clear shares - PPLNS uses rolling window
        // Just log the event
        tracing::info!("New PPLNS round started, window size preserved");
    }

    /// Get share proportions for current window (no fee deduction).
    /// Returns Vec<(wallet_address, proportion)> where proportions sum to ~1.0.
    /// Used by block producer for PPLNS coinbase distribution — block producer
    /// handles dev fee separately, so this method must NOT deduct any fees.
    pub fn get_share_proportions(
        &self,
        wallet_lookup: impl Fn(&WorkerId) -> String,
    ) -> Vec<(String, f64)> {
        let shares = self.shares.read();
        let total_diff = *self.total_difficulty.read();
        if total_diff <= 0.0 || shares.is_empty() {
            return Vec::new();
        }
        // Aggregate difficulty per worker
        let mut worker_difficulty: IndexMap<WorkerId, f64> = IndexMap::new();
        for share in shares.iter() {
            *worker_difficulty.entry(share.worker_id.clone()).or_insert(0.0) += share.difficulty;
        }
        // Convert to wallet proportions (merge workers with same wallet)
        let mut wallet_proportions: IndexMap<String, f64> = IndexMap::new();
        for (worker_id, difficulty) in &worker_difficulty {
            let wallet = wallet_lookup(worker_id);
            *wallet_proportions.entry(wallet).or_insert(0.0) += difficulty / total_diff;
        }
        wallet_proportions.into_iter().collect()
    }

    /// Get current share count
    pub fn share_count(&self) -> usize {
        self.shares.read().len()
    }

    /// Get total difficulty in window
    pub fn window_difficulty(&self) -> f64 {
        *self.total_difficulty.read()
    }

    /// Get worker statistics
    pub fn worker_stats(&self) -> Vec<(WorkerId, f64, f64)> {
        let shares = self.shares.read();
        let total_diff = *self.total_difficulty.read();

        let mut worker_difficulty: IndexMap<WorkerId, f64> = IndexMap::new();
        for share in shares.iter() {
            *worker_difficulty.entry(share.worker_id.clone()).or_insert(0.0) += share.difficulty;
        }

        worker_difficulty
            .into_iter()
            .map(|(id, diff)| {
                let proportion = if total_diff > 0.0 { diff / total_diff } else { 0.0 };
                (id, diff, proportion)
            })
            .collect()
    }

    /// Get round history
    pub fn round_history(&self) -> Vec<Round> {
        self.rounds.read().iter().cloned().collect()
    }

    /// Get last round
    pub fn last_round(&self) -> Option<Round> {
        self.rounds.read().back().cloned()
    }

    /// Get pool statistics
    pub fn stats(&self) -> PPLNSStats {
        let shares = self.shares.read();
        let network_diff = *self.network_difficulty.read();

        PPLNSStats {
            n_factor: self.config.n_factor,
            window_size: network_diff * self.config.n_factor,
            current_difficulty: *self.total_difficulty.read(),
            share_count: shares.len(),
            worker_count: shares.iter()
                .map(|s| &s.worker_id)
                .collect::<std::collections::HashSet<_>>()
                .len(),
            oldest_share: shares.front().map(|s| s.timestamp),
            newest_share: shares.back().map(|s| s.timestamp),
        }
    }

    /// Get current round ID
    pub fn current_round_id(&self) -> u64 {
        *self.round_counter.read()
    }
}

/// PPLNS statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PPLNSStats {
    pub n_factor: f64,
    pub window_size: f64,
    pub current_difficulty: f64,
    pub share_count: usize,
    pub worker_count: usize,
    pub oldest_share: Option<DateTime<Utc>>,
    pub newest_share: Option<DateTime<Utc>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> PPLNSConfig {
        PPLNSConfig {
            n_factor: 2.0,
            max_shares_in_memory: 1000,
            persist_shares: false,
        }
    }

    #[test]
    fn test_add_share() {
        let calculator = PPLNSCalculator::new(test_config(), 150);
        calculator.update_network_difficulty(1.0);

        let share = Share::new(
            WorkerId::new("qnk123", "rig1"),
            "job1".to_string(),
            0.5,
            [0; 32],
            12345,
            false,
        );

        calculator.add_share(share);

        assert_eq!(calculator.share_count(), 1);
        assert_eq!(calculator.window_difficulty(), 0.5);
    }

    #[test]
    fn test_calculate_rewards() {
        let calculator = PPLNSCalculator::new(test_config(), 150);
        // n_factor=2.0 so window = network_diff * 2.0; set 6.0 → window=12 covers all 10 shares
        calculator.update_network_difficulty(6.0);

        // Add shares from two workers
        for i in 0..10 {
            let worker = if i < 7 { "worker1" } else { "worker2" };
            let share = Share::new(
                WorkerId::new("qnk123", worker),
                "job1".to_string(),
                1.0,
                [0; 32],
                i,
                false,
            );
            calculator.add_share(share);
        }

        let round = calculator.calculate_rewards(
            2_000_000_000, // 2 QUG
            100,
            [1; 32],
            WorkerId::new("qnk123", "worker1"),
            |id| id.wallet().to_string(),
        );

        // Check fee calculations (v8.6.0: DEV_FEE_BPS=175 = 1.75%, pool_fee_bps=150 = 1.5%)
        assert_eq!(round.dev_fee, 35_000_000); // 1.75%
        assert_eq!(round.pool_fee, 30_000_000); // 1.5%
        assert_eq!(round.miner_rewards, 1_935_000_000); // 96.75%

        // Worker1 has 70% of shares
        assert_eq!(round.payouts.len(), 2);
        assert!(round.payouts[0].proportion > 0.6); // ~70%
    }

    #[test]
    fn test_window_trimming() {
        let calculator = PPLNSCalculator::new(test_config(), 150);
        calculator.update_network_difficulty(1.0); // Window = 2.0 difficulty

        // Add 5 shares with difficulty 1.0 each
        for i in 0..5 {
            let share = Share::new(
                WorkerId::new("qnk123", "rig1"),
                "job1".to_string(),
                1.0,
                [0; 32],
                i,
                false,
            );
            calculator.add_share(share);
        }

        // Window should be trimmed to ~2.0 difficulty (N=2)
        assert!(calculator.window_difficulty() <= 2.5);
    }
}
