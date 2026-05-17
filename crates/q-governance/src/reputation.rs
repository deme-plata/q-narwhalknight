//! Miner reputation system for long-term contributors

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

/// Reputation system tracking long-term miner contributions
pub struct ReputationSystem {
    /// Reputation scores by miner address
    reputations: Arc<RwLock<HashMap<[u8; 32], MinerReputation>>>,
}

/// Miner reputation data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinerReputation {
    /// Miner address
    pub address: [u8; 32],

    /// Total blocks mined historically
    pub lifetime_blocks_mined: u64,

    /// Total hashes contributed historically
    pub lifetime_hashes: u128,

    /// Time when first block was mined (Unix timestamp)
    pub first_block_timestamp: u64,

    /// Years active (calculated from first_block_timestamp)
    pub years_active: f64,

    /// Trust score (0.0 - 1.0)
    pub trust_score: f64,

    /// Reputation multiplier for voting (1.0 - 2.0)
    pub reputation_multiplier: f64,
}

impl ReputationSystem {
    /// Create new reputation system
    pub fn new() -> Self {
        Self {
            reputations: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Update reputation for a miner
    pub async fn update_reputation(
        &self,
        address: [u8; 32],
        blocks_mined: u64,
        hashes: u128,
    ) {
        let mut reps = self.reputations.write().await;

        let rep = reps.entry(address).or_insert_with(|| {
            MinerReputation {
                address,
                lifetime_blocks_mined: 0,
                lifetime_hashes: 0,
                first_block_timestamp: chrono::Utc::now().timestamp() as u64,
                years_active: 0.0,
                trust_score: 0.5,
                reputation_multiplier: 1.0,
            }
        });

        rep.lifetime_blocks_mined += blocks_mined;
        rep.lifetime_hashes += hashes;

        // Calculate years active
        let now = chrono::Utc::now().timestamp() as u64;
        let seconds_active = now - rep.first_block_timestamp;
        rep.years_active = seconds_active as f64 / (365.25 * 24.0 * 3600.0);

        // Calculate trust score
        rep.trust_score = self.calculate_trust_score(rep);

        // Calculate reputation multiplier
        rep.reputation_multiplier = self.calculate_reputation_multiplier(rep);

        info!(
            "✅ Updated reputation: address={}, trust_score={:.2}, multiplier={:.2}x",
            hex::encode(address),
            rep.trust_score,
            rep.reputation_multiplier
        );
    }

    /// Calculate trust score based on history
    fn calculate_trust_score(&self, rep: &MinerReputation) -> f64 {
        // Factors:
        // 1. Years active (max 5 years = 1.0)
        let years_factor = (rep.years_active / 5.0).min(1.0);

        // 2. Lifetime blocks (log scale)
        let blocks_factor = if rep.lifetime_blocks_mined > 0 {
            (rep.lifetime_blocks_mined as f64).log10() / 6.0 // 1M blocks = 1.0
        } else {
            0.0
        }
        .min(1.0);

        // 3. Consistency (blocks per year)
        let consistency_factor = if rep.years_active > 0.0 {
            let blocks_per_year = rep.lifetime_blocks_mined as f64 / rep.years_active;
            (blocks_per_year / 10000.0).min(1.0) // 10K blocks/year = 1.0
        } else {
            0.0
        };

        // Weighted average
        let trust = (years_factor * 0.4) + (blocks_factor * 0.3) + (consistency_factor * 0.3);

        trust.max(0.0).min(1.0)
    }

    /// Calculate reputation multiplier for voting power
    ///
    /// Long-term miners get bonus voting power:
    /// - 0 years: 1.0x
    /// - 1 year: 1.2x
    /// - 3 years: 1.5x
    /// - 5+ years: 2.0x (max)
    fn calculate_reputation_multiplier(&self, rep: &MinerReputation) -> f64 {
        let base_multiplier = 1.0;
        let max_multiplier = 2.0;

        // Logarithmic scaling based on years active
        let years_bonus = if rep.years_active > 0.0 {
            (rep.years_active.log2() / 2.0).min(1.0)
        } else {
            0.0
        };

        base_multiplier + (years_bonus * (max_multiplier - base_multiplier))
    }

    /// Get reputation for a miner
    pub async fn get_reputation(&self, address: &[u8; 32]) -> Option<MinerReputation> {
        let reps = self.reputations.read().await;
        reps.get(address).cloned()
    }

    /// Get all reputations (for leaderboard)
    pub async fn get_all_reputations(&self) -> Vec<MinerReputation> {
        let reps = self.reputations.read().await;
        let mut all: Vec<_> = reps.values().cloned().collect();
        all.sort_by(|a, b| {
            b.trust_score
                .partial_cmp(&a.trust_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all
    }

    /// Get top miners by reputation
    pub async fn get_top_miners(&self, count: usize) -> Vec<MinerReputation> {
        let mut all = self.get_all_reputations().await;
        all.truncate(count);
        all
    }
}

impl Default for ReputationSystem {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_new_miner_reputation() {
        let system = ReputationSystem::new();
        let address = [1u8; 32];

        system.update_reputation(address, 1, 1_000_000).await;

        let rep = system.get_reputation(&address).await.unwrap();
        assert_eq!(rep.lifetime_blocks_mined, 1);
        assert_eq!(rep.lifetime_hashes, 1_000_000);
        assert_eq!(rep.reputation_multiplier, 1.0); // New miner, no bonus
    }

    #[tokio::test]
    async fn test_veteran_miner_bonus() {
        let system = ReputationSystem::new();
        let address = [2u8; 32];

        // Simulate veteran miner (set first block to 3 years ago)
        {
            let mut reps = system.reputations.write().await;
            let three_years_ago =
                chrono::Utc::now().timestamp() as u64 - (3 * 365 * 24 * 3600);

            reps.insert(
                address,
                MinerReputation {
                    address,
                    lifetime_blocks_mined: 10000,
                    lifetime_hashes: 10_000_000_000,
                    first_block_timestamp: three_years_ago,
                    years_active: 3.0,
                    trust_score: 0.0,
                    reputation_multiplier: 1.0,
                },
            );
        }

        system.update_reputation(address, 1, 1_000_000).await;

        let rep = system.get_reputation(&address).await.unwrap();
        assert!(rep.reputation_multiplier > 1.0); // Veteran gets bonus
        assert!(rep.reputation_multiplier < 2.0); // But not max yet
        assert!(rep.trust_score > 0.5); // High trust score
    }
}
