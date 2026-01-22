//! Staking & Slashing Security Model for Q-NarwhalKnight
//!
//! v1.4.4-beta: Economic security layer that makes attacks irrational
//!
//! ## Design Goals
//!
//! 1. **Make attacks economically irrational**: Even if technically possible,
//!    attackers should lose more money than they can steal.
//!
//! 2. **Protect retail instant payments**: Merchants accepting 0-conf payments
//!    are protected by an insurance pool funded by staking rewards.
//!
//! 3. **Incentivize honest behavior**: Miners who stake collateral earn
//!    higher rewards but lose their stake if caught misbehaving.
//!
//! ## Staking Tiers
//!
//! | Tier | Stake Required | Reward Multiplier | Slashing Risk |
//! |------|---------------|-------------------|---------------|
//! | 🥉 Bronze | 1,000 QUG | 1.0x | 10% of stake |
//! | 🥈 Silver | 10,000 QUG | 1.2x | 25% of stake |
//! | 🥇 Gold | 100,000 QUG | 1.5x | 50% of stake |
//! | 💎 Diamond | 1,000,000 QUG | 2.0x | 100% of stake |
//!
//! ## Slashable Offenses
//!
//! 1. **Equivocation**: Mining conflicting blocks at same height
//! 2. **Double Signing**: Signing conflicting DAG vertices
//! 3. **Orphan Spam**: Deliberately creating orphan blocks
//! 4. **VDF Manipulation**: Submitting invalid VDF proofs
//!
//! ## Insurance Pool
//!
//! - 5% of all mining rewards go to the insurance pool
//! - Covers merchants who accept instant (0-conf) payments
//! - If double-spend on instant payment, merchant refunded from pool
//! - Pool size target: 1% of circulating supply

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Staking tier definitions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StakingTier {
    /// No stake - basic miner
    Unstaked,
    /// 1,000 QUG stake - 1.0x rewards, 10% slashing
    Bronze,
    /// 10,000 QUG stake - 1.2x rewards, 25% slashing
    Silver,
    /// 100,000 QUG stake - 1.5x rewards, 50% slashing
    Gold,
    /// 1,000,000 QUG stake - 2.0x rewards, 100% slashing
    Diamond,
}

/// Base units constant for 24-decimal precision
const ONE_QUG: u128 = 1_000_000_000_000_000_000_000_000; // 10^24

impl StakingTier {
    /// Minimum stake required for this tier (in base units, 24 decimals)
    pub fn min_stake(&self) -> u128 {
        match self {
            StakingTier::Unstaked => 0,
            StakingTier::Bronze => 1_000 * ONE_QUG,      // 1,000 QUG
            StakingTier::Silver => 10_000 * ONE_QUG,     // 10,000 QUG
            StakingTier::Gold => 100_000 * ONE_QUG,      // 100,000 QUG
            StakingTier::Diamond => 1_000_000 * ONE_QUG, // 1,000,000 QUG
        }
    }

    /// Reward multiplier for this tier
    pub fn reward_multiplier(&self) -> f64 {
        match self {
            StakingTier::Unstaked => 0.8, // Unstaked miners get 80% rewards
            StakingTier::Bronze => 1.0,
            StakingTier::Silver => 1.2,
            StakingTier::Gold => 1.5,
            StakingTier::Diamond => 2.0,
        }
    }

    /// Percentage of stake slashed on offense (0.0 - 1.0)
    pub fn slashing_percentage(&self) -> f64 {
        match self {
            StakingTier::Unstaked => 0.0, // Can't slash what you don't have
            StakingTier::Bronze => 0.10,
            StakingTier::Silver => 0.25,
            StakingTier::Gold => 0.50,
            StakingTier::Diamond => 1.00, // Full stake at risk
        }
    }

    /// Determine tier from stake amount
    pub fn from_stake(stake: u128) -> Self {
        if stake >= StakingTier::Diamond.min_stake() {
            StakingTier::Diamond
        } else if stake >= StakingTier::Gold.min_stake() {
            StakingTier::Gold
        } else if stake >= StakingTier::Silver.min_stake() {
            StakingTier::Silver
        } else if stake >= StakingTier::Bronze.min_stake() {
            StakingTier::Bronze
        } else {
            StakingTier::Unstaked
        }
    }

    /// Get tier emoji for display
    pub fn emoji(&self) -> &'static str {
        match self {
            StakingTier::Unstaked => "⚪",
            StakingTier::Bronze => "🥉",
            StakingTier::Silver => "🥈",
            StakingTier::Gold => "🥇",
            StakingTier::Diamond => "💎",
        }
    }
}

/// Slashable offenses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SlashableOffense {
    /// Mining conflicting blocks at the same height
    Equivocation {
        block_hash_1: String,
        block_hash_2: String,
        height: u64,
    },
    /// Signing conflicting DAG vertices
    DoubleSigning {
        vertex_id_1: String,
        vertex_id_2: String,
    },
    /// Deliberately creating orphan blocks
    OrphanSpam {
        orphan_count: u64,
        time_window_seconds: u64,
    },
    /// Submitting invalid VDF proofs
    VdfManipulation {
        block_hash: String,
        expected_iterations: u64,
        actual_iterations: u64,
    },
    /// Attempted double-spend on instant payment
    InstantPaymentDoubleSpend {
        tx_hash: String,
        merchant_address: String,
        amount: u128,
    },
}

impl SlashableOffense {
    /// Severity multiplier for slashing calculation
    pub fn severity_multiplier(&self) -> f64 {
        match self {
            SlashableOffense::Equivocation { .. } => 1.0,      // Full slashing
            SlashableOffense::DoubleSigning { .. } => 1.0,     // Full slashing
            SlashableOffense::OrphanSpam { orphan_count, .. } => {
                (*orphan_count as f64 / 10.0).min(1.0) // Scales with severity
            }
            SlashableOffense::VdfManipulation { .. } => 0.5,   // Half slashing
            SlashableOffense::InstantPaymentDoubleSpend { .. } => 2.0, // Double slashing + refund
        }
    }
}

/// Miner stake record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinerStake {
    /// Miner's wallet address
    pub address: String,
    /// Staked amount in base units (24 decimals)
    #[serde(serialize_with = "q_types::u128_serde::serialize", deserialize_with = "q_types::u128_serde::deserialize")]
    pub staked_amount: u128,
    /// Current tier
    pub tier: StakingTier,
    /// When stake was locked
    pub stake_timestamp: u64,
    /// Minimum unlock time (stake must be locked for 30 days)
    pub unlock_timestamp: u64,
    /// Total rewards earned while staking (24 decimals)
    #[serde(serialize_with = "q_types::u128_serde::serialize", deserialize_with = "q_types::u128_serde::deserialize")]
    pub rewards_earned: u128,
    /// Any pending slashing (24 decimals)
    #[serde(serialize_with = "q_types::u128_serde::serialize", deserialize_with = "q_types::u128_serde::deserialize")]
    pub pending_slash: u128,
    /// Offense history
    pub offense_history: Vec<SlashRecord>,
}

/// Record of a slashing event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlashRecord {
    pub timestamp: u64,
    pub offense: SlashableOffense,
    #[serde(serialize_with = "q_types::u128_serde::serialize", deserialize_with = "q_types::u128_serde::deserialize")]
    pub slashed_amount: u128,
    #[serde(serialize_with = "q_types::u128_serde::serialize", deserialize_with = "q_types::u128_serde::deserialize")]
    pub remaining_stake: u128,
}

/// Insurance pool for instant payments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsurancePool {
    /// Total pool balance in base units (24 decimals)
    #[serde(serialize_with = "q_types::u128_serde::serialize", deserialize_with = "q_types::u128_serde::deserialize")]
    pub balance: u128,
    /// Target balance (1% of circulating supply)
    #[serde(serialize_with = "q_types::u128_serde::serialize", deserialize_with = "q_types::u128_serde::deserialize")]
    pub target_balance: u128,
    /// Claims paid out (24 decimals)
    #[serde(serialize_with = "q_types::u128_serde::serialize", deserialize_with = "q_types::u128_serde::deserialize")]
    pub total_claims_paid: u128,
    /// Number of claims
    pub claim_count: u64,
    /// Pending claims
    pub pending_claims: Vec<InsuranceClaim>,
}

/// Insurance claim for double-spend on instant payment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsuranceClaim {
    pub claim_id: String,
    pub merchant_address: String,
    pub tx_hash: String,
    #[serde(serialize_with = "q_types::u128_serde::serialize", deserialize_with = "q_types::u128_serde::deserialize")]
    pub amount: u128,
    pub timestamp: u64,
    pub status: ClaimStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ClaimStatus {
    Pending,
    UnderReview,
    Approved,
    Paid,
    Rejected,
}

/// Staking security manager
pub struct StakingSecurityManager {
    /// Miner stakes indexed by address
    stakes: Arc<RwLock<HashMap<String, MinerStake>>>,
    /// Insurance pool
    insurance_pool: Arc<RwLock<InsurancePool>>,
    /// Percentage of rewards going to insurance (5%)
    insurance_contribution_rate: f64,
    /// Minimum stake lock period (30 days in seconds)
    min_lock_period: u64,
}

impl Default for StakingSecurityManager {
    fn default() -> Self {
        Self::new()
    }
}

impl StakingSecurityManager {
    pub fn new() -> Self {
        Self {
            stakes: Arc::new(RwLock::new(HashMap::new())),
            insurance_pool: Arc::new(RwLock::new(InsurancePool {
                balance: 0,
                target_balance: 21_000_000 * ONE_QUG / 100, // 1% of max supply (24 decimals)
                total_claims_paid: 0,
                claim_count: 0,
                pending_claims: Vec::new(),
            })),
            insurance_contribution_rate: 0.05, // 5%
            min_lock_period: 30 * 24 * 60 * 60, // 30 days
        }
    }

    /// Stake QUG for a miner
    pub async fn stake(&self, address: &str, amount: u128) -> Result<StakingTier> {
        let mut stakes = self.stakes.write().await;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs();

        let stake = stakes.entry(address.to_string()).or_insert(MinerStake {
            address: address.to_string(),
            staked_amount: 0,
            tier: StakingTier::Unstaked,
            stake_timestamp: now,
            unlock_timestamp: now + self.min_lock_period,
            rewards_earned: 0,
            pending_slash: 0,
            offense_history: Vec::new(),
        });

        stake.staked_amount += amount;
        stake.tier = StakingTier::from_stake(stake.staked_amount);
        stake.stake_timestamp = now;
        stake.unlock_timestamp = now + self.min_lock_period;

        info!(
            "💰 Miner {} staked {} QUG, now at {} tier",
            address,
            amount as f64 / ONE_QUG as f64,
            stake.tier.emoji()
        );

        Ok(stake.tier)
    }

    /// Unstake QUG (if lock period has passed)
    pub async fn unstake(&self, address: &str, amount: u128) -> Result<u128> {
        let mut stakes = self.stakes.write().await;

        let stake = stakes
            .get_mut(address)
            .ok_or_else(|| anyhow!("No stake found for address"))?;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs();

        if now < stake.unlock_timestamp {
            return Err(anyhow!(
                "Stake locked until {} (in {} seconds)",
                stake.unlock_timestamp,
                stake.unlock_timestamp - now
            ));
        }

        if stake.pending_slash > 0 {
            return Err(anyhow!(
                "Cannot unstake while {} pending slash",
                stake.pending_slash
            ));
        }

        let unstake_amount = amount.min(stake.staked_amount);
        stake.staked_amount -= unstake_amount;
        stake.tier = StakingTier::from_stake(stake.staked_amount);

        info!(
            "💸 Miner {} unstaked {} QUG, now at {} tier",
            address,
            unstake_amount as f64 / ONE_QUG as f64,
            stake.tier.emoji()
        );

        Ok(unstake_amount)
    }

    /// Process a slashing offense
    pub async fn slash(&self, address: &str, offense: SlashableOffense) -> Result<u128> {
        let mut stakes = self.stakes.write().await;

        let stake = stakes
            .get_mut(address)
            .ok_or_else(|| anyhow!("No stake found for address"))?;

        // Calculate slash amount
        let base_slash = (stake.staked_amount as f64 * stake.tier.slashing_percentage()) as u128;
        let severity_adjusted = (base_slash as f64 * offense.severity_multiplier()) as u128;
        let slash_amount = severity_adjusted.min(stake.staked_amount);

        stake.staked_amount -= slash_amount;
        stake.tier = StakingTier::from_stake(stake.staked_amount);

        // Record the offense
        stake.offense_history.push(SlashRecord {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs(),
            offense: offense.clone(),
            slashed_amount: slash_amount,
            remaining_stake: stake.staked_amount,
        });

        error!(
            "⚔️ SLASHED: Miner {} lost {} QUG for {:?}",
            address,
            slash_amount as f64 / ONE_QUG as f64,
            offense
        );

        // If instant payment double-spend, add slashed amount to insurance pool
        if let SlashableOffense::InstantPaymentDoubleSpend { amount, .. } = &offense {
            let mut pool = self.insurance_pool.write().await;
            pool.balance += slash_amount;
            info!(
                "🛡️ Insurance pool increased by {} QUG from slashing",
                slash_amount as f64 / ONE_QUG as f64
            );
        }

        Ok(slash_amount)
    }

    /// Calculate adjusted mining reward based on stake tier
    pub async fn calculate_reward(&self, address: &str, base_reward: u128) -> Result<(u128, u128)> {
        let stakes = self.stakes.read().await;

        let (multiplier, tier) = if let Some(stake) = stakes.get(address) {
            (stake.tier.reward_multiplier(), stake.tier)
        } else {
            (StakingTier::Unstaked.reward_multiplier(), StakingTier::Unstaked)
        };

        let adjusted_reward = (base_reward as f64 * multiplier) as u128;
        let insurance_contribution = (adjusted_reward as f64 * self.insurance_contribution_rate) as u128;
        let final_reward = adjusted_reward - insurance_contribution;

        debug!(
            "💎 {} tier miner {} gets {} QUG ({:.0}% of {}), {} to insurance",
            tier.emoji(),
            address,
            final_reward as f64 / ONE_QUG as f64,
            multiplier * 100.0,
            base_reward as f64 / ONE_QUG as f64,
            insurance_contribution as f64 / ONE_QUG as f64
        );

        // Add to insurance pool
        {
            let mut pool = self.insurance_pool.write().await;
            pool.balance += insurance_contribution;
        }

        Ok((final_reward, insurance_contribution))
    }

    /// File an insurance claim for instant payment double-spend
    pub async fn file_claim(
        &self,
        merchant_address: &str,
        tx_hash: &str,
        amount: u128,
    ) -> Result<String> {
        let mut pool = self.insurance_pool.write().await;

        let claim_id = format!("CLAIM-{}", uuid::Uuid::new_v4());

        pool.pending_claims.push(InsuranceClaim {
            claim_id: claim_id.clone(),
            merchant_address: merchant_address.to_string(),
            tx_hash: tx_hash.to_string(),
            amount,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs(),
            status: ClaimStatus::Pending,
        });

        warn!(
            "🆘 Insurance claim filed: {} for {} QUG by merchant {}",
            claim_id,
            amount as f64 / ONE_QUG as f64,
            merchant_address
        );

        Ok(claim_id)
    }

    /// Process approved insurance claim
    /// v3.0.4: Returns u128 for 24-decimal precision
    pub async fn pay_claim(&self, claim_id: &str) -> Result<u128> {
        let mut pool = self.insurance_pool.write().await;

        let claim_idx = pool
            .pending_claims
            .iter()
            .position(|c| c.claim_id == claim_id)
            .ok_or_else(|| anyhow!("Claim not found"))?;

        // Extract values first to avoid borrow conflicts
        let claim_status = pool.pending_claims[claim_idx].status.clone();
        let claim_amount = pool.pending_claims[claim_idx].amount;

        if claim_status != ClaimStatus::Approved {
            return Err(anyhow!("Claim not approved"));
        }

        if pool.balance < claim_amount {
            return Err(anyhow!("Insufficient insurance pool balance"));
        }

        // Now modify
        pool.balance -= claim_amount;
        pool.total_claims_paid += claim_amount;
        pool.claim_count += 1;
        pool.pending_claims[claim_idx].status = ClaimStatus::Paid;

        info!(
            "✅ Insurance claim {} paid: {} QUG to merchant",
            claim_id,
            claim_amount / 1_000_000_000_000_000_000_000_000u128
        );

        Ok(claim_amount)
    }

    /// Get staking statistics
    pub async fn get_stats(&self) -> serde_json::Value {
        let stakes = self.stakes.read().await;
        let pool = self.insurance_pool.read().await;

        // v3.0.4: staked_amount is now u128
        let total_staked: u128 = stakes.values().map(|s| s.staked_amount).sum();
        let tier_counts: HashMap<String, usize> = stakes
            .values()
            .fold(HashMap::new(), |mut acc, s| {
                *acc.entry(format!("{:?}", s.tier)).or_insert(0) += 1;
                acc
            });

        serde_json::json!({
            "staking": {
                "total_staked_qug": total_staked / 1_000_000_000_000_000_000_000_000u128,
                "total_stakers": stakes.len(),
                "tier_distribution": tier_counts,
                "tiers": {
                    "bronze": { "min_stake": 1000, "reward_multiplier": "1.0x", "slash_risk": "10%" },
                    "silver": { "min_stake": 10000, "reward_multiplier": "1.2x", "slash_risk": "25%" },
                    "gold": { "min_stake": 100000, "reward_multiplier": "1.5x", "slash_risk": "50%" },
                    "diamond": { "min_stake": 1000000, "reward_multiplier": "2.0x", "slash_risk": "100%" }
                }
            },
            "insurance_pool": {
                "balance_qug": pool.balance / 1_000_000_000_000_000_000_000_000u128,
                "target_balance_qug": pool.target_balance / 1_000_000_000_000_000_000_000_000u128,
                "fill_percentage": (pool.balance as f64 / pool.target_balance as f64 * 100.0),
                "total_claims_paid_qug": pool.total_claims_paid / 1_000_000_000_000_000_000_000_000u128,
                "claim_count": pool.claim_count,
                "pending_claims": pool.pending_claims.len(),
                "contribution_rate": "5% of mining rewards"
            },
            "security_model": {
                "slashable_offenses": [
                    "Equivocation (mining conflicting blocks)",
                    "Double Signing (conflicting DAG vertices)",
                    "Orphan Spam (deliberate orphan creation)",
                    "VDF Manipulation (invalid proofs)",
                    "Instant Payment Double-Spend (2x slashing)"
                ],
                "instant_payment_protection": "Merchants accepting 0-conf are insured up to pool balance"
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_staking_tiers() {
        let manager = StakingSecurityManager::new();

        // Stake bronze tier
        let tier = manager.stake("miner1", 1_000 * 1_000_000_000_000_000_000_000_000u128).await.unwrap();
        assert_eq!(tier, StakingTier::Bronze);

        // Stake more to reach silver
        let tier = manager.stake("miner1", 9_000 * 1_000_000_000_000_000_000_000_000u128).await.unwrap();
        assert_eq!(tier, StakingTier::Silver);
    }

    #[tokio::test]
    async fn test_reward_calculation() {
        let manager = StakingSecurityManager::new();

        // Unstaked miner gets 80% * 95% (after insurance)
        let (reward, insurance) = manager
            .calculate_reward("unstaked_miner", 100 * 1_000_000_000_000_000_000_000_000u128)
            .await
            .unwrap();
        assert_eq!(insurance, 4 * 1_000_000_000_000_000_000_000_000u128); // 5% of 80 = 4
        assert_eq!(reward, 76 * 1_000_000_000_000_000_000_000_000u128);   // 80 - 4 = 76

        // Stake diamond tier
        manager.stake("diamond_miner", 1_000_000 * 1_000_000_000_000_000_000_000_000u128).await.unwrap();

        // Diamond miner gets 200% * 95% (after insurance)
        let (reward, insurance) = manager
            .calculate_reward("diamond_miner", 100 * 1_000_000_000_000_000_000_000_000u128)
            .await
            .unwrap();
        assert_eq!(insurance, 10 * 1_000_000_000_000_000_000_000_000u128); // 5% of 200 = 10
        assert_eq!(reward, 190 * 1_000_000_000_000_000_000_000_000u128);   // 200 - 10 = 190
    }

    #[tokio::test]
    async fn test_slashing() {
        let manager = StakingSecurityManager::new();

        // Stake gold tier
        manager.stake("miner1", 100_000 * 1_000_000_000_000_000_000_000_000u128).await.unwrap();

        // Slash for equivocation (50% of stake)
        let slashed = manager
            .slash(
                "miner1",
                SlashableOffense::Equivocation {
                    block_hash_1: "hash1".to_string(),
                    block_hash_2: "hash2".to_string(),
                    height: 100,
                },
            )
            .await
            .unwrap();

        assert_eq!(slashed, 50_000 * 1_000_000_000_000_000_000_000_000u128); // 50% of 100k
    }
}
