//! Prediction Markets
//!
//! AMM for prediction shares and staking on neural predictions.

use std::collections::HashMap;
use tracing::{debug, info, warn};

/// Prediction markets container
pub struct PredictionMarkets {
    /// Prediction share AMM
    pub amm: PredictionAMM,

    /// Prediction staking pool
    pub staking: PredictionStakingPool,
}

impl PredictionMarkets {
    /// Create new prediction markets
    pub fn new() -> Self {
        Self {
            amm: PredictionAMM::new(),
            staking: PredictionStakingPool::new(),
        }
    }
}

impl Default for PredictionMarkets {
    fn default() -> Self {
        Self::new()
    }
}

/// Automated Market Maker for prediction shares
pub struct PredictionAMM {
    /// Liquidity pools per prediction type
    pools: HashMap<PredictionType, PredictionPool>,

    /// Swap fee (basis points)
    swap_fee_bps: u32,

    /// Total fees collected
    total_fees: u64,
}

/// Prediction type identifier
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct PredictionType {
    pub domain: crate::PredictionDomain,
    pub target_block: u64,
}

/// Liquidity pool for prediction shares
#[derive(Clone, Debug)]
pub struct PredictionPool {
    /// Share reserves: (YES shares, NO shares)
    pub reserves: (u64, u64),

    /// LP token supply
    pub lp_supply: u64,

    /// Pool creation block
    pub created_at: u64,

    /// Resolution block
    pub resolves_at: u64,

    /// Resolution outcome (set after resolution)
    pub outcome: Option<bool>,

    /// Total volume
    pub volume: u64,
}

/// Share type (YES or NO)
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ShareType {
    Yes,
    No,
}

impl PredictionAMM {
    /// Create new AMM
    pub fn new() -> Self {
        Self {
            pools: HashMap::new(),
            swap_fee_bps: 30, // 0.30% fee
            total_fees: 0,
        }
    }

    /// Create a new prediction market
    pub fn create_market(
        &mut self,
        prediction_type: PredictionType,
        initial_liquidity: u64,
        resolves_at: u64,
    ) -> Result<u64, MarketError> {
        if self.pools.contains_key(&prediction_type) {
            return Err(MarketError::MarketExists);
        }

        if initial_liquidity < 1_000_000 { // Minimum 0.01 QUG
            return Err(MarketError::InsufficientLiquidity);
        }

        // Split initial liquidity into YES and NO shares
        let shares_each = initial_liquidity / 2;

        let pool = PredictionPool {
            reserves: (shares_each, shares_each),
            lp_supply: shares_each, // 1:1 LP tokens
            created_at: 0, // Would be current block
            resolves_at,
            outcome: None,
            volume: 0,
        };

        self.pools.insert(prediction_type, pool);
        info!("Created prediction market for {:?}", prediction_type.domain);

        Ok(shares_each) // Return LP tokens minted
    }

    /// Buy prediction shares
    pub fn buy_shares(
        &mut self,
        prediction_type: PredictionType,
        share_type: ShareType,
        qug_amount: u64,
    ) -> Result<BuyResult, MarketError> {
        let pool = self.pools.get_mut(&prediction_type)
            .ok_or(MarketError::MarketNotFound)?;

        if pool.outcome.is_some() {
            return Err(MarketError::MarketResolved);
        }

        // Get reserves based on share type
        let (reserve_in, reserve_out) = match share_type {
            ShareType::Yes => (pool.reserves.1, pool.reserves.0),
            ShareType::No => (pool.reserves.0, pool.reserves.1),
        };

        // Apply fee (v1.4.5-beta: saturating add to prevent overflow)
        let fee = (qug_amount as u128 * self.swap_fee_bps as u128 / 10000) as u64;
        let amount_after_fee = qug_amount.saturating_sub(fee);
        self.total_fees = self.total_fees.saturating_add(fee);

        // Constant product: x * y = k
        let k = reserve_in as u128 * reserve_out as u128;
        let new_reserve_in = reserve_in + amount_after_fee;
        let new_reserve_out = (k / new_reserve_in as u128) as u64;
        let shares_out = reserve_out.saturating_sub(new_reserve_out);

        // Update reserves
        match share_type {
            ShareType::Yes => {
                pool.reserves.0 = new_reserve_out;
                pool.reserves.1 = new_reserve_in;
            }
            ShareType::No => {
                pool.reserves.0 = new_reserve_in;
                pool.reserves.1 = new_reserve_out;
            }
        }

        pool.volume += qug_amount;

        // Calculate price before dropping mutable borrow
        let total = pool.reserves.0 + pool.reserves.1;
        let new_price = if total == 0 { 0.5 } else { pool.reserves.1 as f64 / total as f64 };

        Ok(BuyResult {
            shares_received: shares_out,
            price_paid: qug_amount as f64 / shares_out.max(1) as f64,
            fee_paid: fee,
            new_price,
        })
    }

    /// Sell prediction shares
    pub fn sell_shares(
        &mut self,
        prediction_type: PredictionType,
        share_type: ShareType,
        shares_amount: u64,
    ) -> Result<SellResult, MarketError> {
        let pool = self.pools.get_mut(&prediction_type)
            .ok_or(MarketError::MarketNotFound)?;

        if pool.outcome.is_some() {
            return Err(MarketError::MarketResolved);
        }

        let (reserve_in, reserve_out) = match share_type {
            ShareType::Yes => (pool.reserves.0, pool.reserves.1),
            ShareType::No => (pool.reserves.1, pool.reserves.0),
        };

        let k = reserve_in as u128 * reserve_out as u128;
        let new_reserve_in = reserve_in + shares_amount;
        let new_reserve_out = (k / new_reserve_in as u128) as u64;
        let qug_out = reserve_out.saturating_sub(new_reserve_out);

        // Apply fee (v1.4.5-beta: saturating add to prevent overflow)
        let fee = (qug_out as u128 * self.swap_fee_bps as u128 / 10000) as u64;
        let qug_after_fee = qug_out.saturating_sub(fee);
        self.total_fees = self.total_fees.saturating_add(fee);

        // Update reserves
        match share_type {
            ShareType::Yes => {
                pool.reserves.0 = new_reserve_in;
                pool.reserves.1 = new_reserve_out;
            }
            ShareType::No => {
                pool.reserves.1 = new_reserve_in;
                pool.reserves.0 = new_reserve_out;
            }
        }

        // Calculate price before dropping mutable borrow
        let total = pool.reserves.0 + pool.reserves.1;
        let new_price = if total == 0 { 0.5 } else { pool.reserves.1 as f64 / total as f64 };

        Ok(SellResult {
            qug_received: qug_after_fee,
            price_received: qug_after_fee as f64 / shares_amount.max(1) as f64,
            fee_paid: fee,
            new_price,
        })
    }

    /// Get current price (probability of YES)
    pub fn get_price(&self, prediction_type: &PredictionType) -> Option<f64> {
        self.pools.get(prediction_type).map(|p| self.calculate_price(p))
    }

    /// Calculate price from pool
    fn calculate_price(&self, pool: &PredictionPool) -> f64 {
        let total = pool.reserves.0 + pool.reserves.1;
        if total == 0 {
            return 0.5;
        }
        pool.reserves.1 as f64 / total as f64
    }

    /// Resolve market
    pub fn resolve_market(
        &mut self,
        prediction_type: PredictionType,
        outcome: bool,
    ) -> Result<(), MarketError> {
        let pool = self.pools.get_mut(&prediction_type)
            .ok_or(MarketError::MarketNotFound)?;

        if pool.outcome.is_some() {
            return Err(MarketError::AlreadyResolved);
        }

        pool.outcome = Some(outcome);
        info!("Resolved market {:?}: outcome={}", prediction_type.domain, outcome);

        Ok(())
    }

    /// Claim winnings after resolution
    pub fn claim_winnings(
        &self,
        prediction_type: &PredictionType,
        share_type: ShareType,
        shares: u64,
    ) -> Result<u64, MarketError> {
        let pool = self.pools.get(prediction_type)
            .ok_or(MarketError::MarketNotFound)?;

        let outcome = pool.outcome.ok_or(MarketError::NotResolved)?;

        let is_winner = match (share_type, outcome) {
            (ShareType::Yes, true) => true,
            (ShareType::No, false) => true,
            _ => false,
        };

        if is_winner {
            Ok(shares) // 1 QUG per winning share
        } else {
            Ok(0)
        }
    }
}

impl Default for PredictionAMM {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of buying shares
#[derive(Clone, Debug)]
pub struct BuyResult {
    pub shares_received: u64,
    pub price_paid: f64,
    pub fee_paid: u64,
    pub new_price: f64,
}

/// Result of selling shares
#[derive(Clone, Debug)]
pub struct SellResult {
    pub qug_received: u64,
    pub price_received: f64,
    pub fee_paid: u64,
    pub new_price: f64,
}

/// Prediction staking pool
pub struct PredictionStakingPool {
    /// Active stakes
    stakes: HashMap<u64, Vec<PredictionStake>>,

    /// Stake ID counter
    next_stake_id: u64,

    /// Minimum stake
    min_stake: u64,

    /// Maximum confidence boost
    max_confidence_boost: f64,
}

/// Individual prediction stake
#[derive(Clone, Debug)]
pub struct PredictionStake {
    pub stake_id: u64,
    pub staker: String,
    pub amount: u64,
    pub predicted_value: f64,
    pub confidence: f64,
    pub staked_at: u64,
}

/// Stake receipt
#[derive(Clone, Debug)]
pub struct StakeReceipt {
    pub stake_id: u64,
    pub amount: u64,
    pub potential_reward: u64,
    pub potential_loss: u64,
}

impl PredictionStakingPool {
    /// Create new staking pool
    pub fn new() -> Self {
        Self {
            stakes: HashMap::new(),
            next_stake_id: 1,
            min_stake: 1_000_000, // 0.01 QUG minimum
            max_confidence_boost: 2.0,
        }
    }

    /// Stake on a prediction
    pub async fn stake_prediction(
        &self,
        prediction_id: u64,
        predicted_value: f64,
        amount: u64,
        confidence: f64,
    ) -> anyhow::Result<StakeReceipt> {
        // Validate confidence
        if !(0.0..=1.0).contains(&confidence) {
            return Err(anyhow::anyhow!("Confidence must be between 0 and 1"));
        }

        // Validate stake amount
        let min_stake = self.calculate_min_stake(confidence);
        if amount < min_stake {
            return Err(anyhow::anyhow!(
                "Insufficient stake: {} required for confidence {}",
                min_stake, confidence
            ));
        }

        let stake_id = 0; // Would increment next_stake_id

        let potential_reward = self.calculate_potential_reward(amount, confidence);
        let potential_loss = self.calculate_potential_loss(amount, confidence);

        Ok(StakeReceipt {
            stake_id,
            amount,
            potential_reward,
            potential_loss,
        })
    }

    /// Calculate minimum stake based on confidence
    fn calculate_min_stake(&self, confidence: f64) -> u64 {
        // Higher confidence requires more stake
        let base = self.min_stake;
        let max = self.min_stake * 100; // 1 QUG at 100% confidence

        (base as f64 + (max - base) as f64 * confidence.powi(2)) as u64
    }

    /// Calculate potential reward
    fn calculate_potential_reward(&self, amount: u64, confidence: f64) -> u64 {
        // Reward = stake × confidence × boost (max 2x)
        let boost = 1.0 + confidence * (self.max_confidence_boost - 1.0);
        (amount as f64 * boost * 0.5) as u64 // 50% max return
    }

    /// Calculate potential loss
    fn calculate_potential_loss(&self, amount: u64, confidence: f64) -> u64 {
        // Loss = stake × confidence
        (amount as f64 * confidence) as u64
    }

    /// Resolve stakes and distribute rewards
    pub fn resolve_stakes(
        &mut self,
        prediction_id: u64,
        actual_value: f64,
    ) -> ResolutionSummary {
        let stakes = match self.stakes.remove(&prediction_id) {
            Some(s) => s,
            None => return ResolutionSummary::default(),
        };

        let mut winners = Vec::new();
        let mut losers = Vec::new();
        let mut total_rewards = 0u64;
        let mut total_slashed = 0u64;

        for stake in stakes {
            let error = (stake.predicted_value - actual_value).abs()
                / actual_value.abs().max(0.01);

            // Threshold based on confidence
            let threshold = 0.1 / stake.confidence.max(0.1);

            if error <= threshold {
                // Winner
                let reward = self.calculate_reward(&stake, error);
                total_rewards += reward;
                winners.push((stake.staker.clone(), reward));
            } else {
                // Loser
                let slash = self.calculate_slash(&stake, error);
                total_slashed += slash;
                losers.push((stake.staker.clone(), slash));
            }
        }

        ResolutionSummary {
            prediction_id,
            actual_value,
            winners,
            losers,
            total_rewards,
            total_slashed,
        }
    }

    fn calculate_reward(&self, stake: &PredictionStake, error: f64) -> u64 {
        let accuracy = 1.0 - error.min(1.0);
        (stake.amount as f64 * accuracy * stake.confidence * 0.5) as u64
    }

    fn calculate_slash(&self, stake: &PredictionStake, error: f64) -> u64 {
        let error_penalty = error.min(1.0);
        (stake.amount as f64 * error_penalty * stake.confidence) as u64
    }
}

impl Default for PredictionStakingPool {
    fn default() -> Self {
        Self::new()
    }
}

/// Resolution summary
#[derive(Clone, Debug, Default)]
pub struct ResolutionSummary {
    pub prediction_id: u64,
    pub actual_value: f64,
    pub winners: Vec<(String, u64)>,
    pub losers: Vec<(String, u64)>,
    pub total_rewards: u64,
    pub total_slashed: u64,
}

/// Market errors
#[derive(Debug, Clone, thiserror::Error)]
pub enum MarketError {
    #[error("Market already exists")]
    MarketExists,

    #[error("Market not found")]
    MarketNotFound,

    #[error("Market already resolved")]
    MarketResolved,

    #[error("Market not yet resolved")]
    NotResolved,

    #[error("Already resolved")]
    AlreadyResolved,

    #[error("Insufficient liquidity")]
    InsufficientLiquidity,

    #[error("Insufficient balance")]
    InsufficientBalance,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PredictionDomain;

    #[test]
    fn test_create_market() {
        let mut amm = PredictionAMM::new();

        let pred_type = PredictionType {
            domain: PredictionDomain::FeeForecasting,
            target_block: 100_000,
        };

        let lp_tokens = amm.create_market(pred_type, 100_000_000, 200_000).unwrap();
        assert!(lp_tokens > 0);

        // Initial price should be 0.5 (50% probability)
        let price = amm.get_price(&pred_type).unwrap();
        assert!((price - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_buy_shares() {
        let mut amm = PredictionAMM::new();

        let pred_type = PredictionType {
            domain: PredictionDomain::FeeForecasting,
            target_block: 100_000,
        };

        amm.create_market(pred_type, 100_000_000, 200_000).unwrap();

        // Buy YES shares
        let result = amm.buy_shares(pred_type, ShareType::Yes, 10_000_000).unwrap();
        assert!(result.shares_received > 0);

        // Price should have moved up (more YES buying = higher YES price)
        let new_price = amm.get_price(&pred_type).unwrap();
        assert!(new_price > 0.5);
    }

    #[test]
    fn test_staking_pool() {
        let pool = PredictionStakingPool::new();

        // High confidence requires more stake
        let min_low = pool.calculate_min_stake(0.1);
        let min_high = pool.calculate_min_stake(0.9);
        assert!(min_high > min_low);
    }
}
