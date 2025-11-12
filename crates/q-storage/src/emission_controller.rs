//! Adaptive Block Reward System - Emission Controller
//!
//! Implements throughput-independent emission by dynamically adjusting
//! block rewards based on actual network throughput.
//!
//! **Core Innovation**: Reward ∝ 1/Throughput → Annual emission is constant
//!
//! At 10 blocks/sec: 0.00026 QUG/block → 82,031 QUG/year
//! At 10,000 blocks/sec: 0.00000026 QUG/block → 82,031 QUG/year
//!
//! This enables 10,000+ blocks/second while maintaining 256-year emission schedule.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use tracing::{debug, info, warn};

/// Genesis timestamp: Oct 26, 2025 00:00 UTC
pub const GENESIS_TIMESTAMP: u64 = 1761436800;

/// Seconds per halving era (4 years)
pub const SECONDS_PER_HALVING: u64 = 126_144_000; // 365.25 * 4 * 24 * 60 * 60

/// Seconds per year (365.25 days for leap years)
pub const SECONDS_PER_YEAR: f64 = 31_557_600.0;

/// Base annual emission for Era 1 (82,031 QUG with 8 decimals)
/// Calculated as: 21M ÷ 256 ÷ 1 = 82,031 QUG/year
pub const BASE_ANNUAL_EMISSION: u64 = 82_031_000_000_000;

/// Minimum reward per block (0.000001 QUG) - prevents division by zero
pub const MIN_REWARD: u64 = 100;

/// Maximum reward per block (1 QUG) - safety cap
pub const MAX_REWARD_PER_BLOCK: u64 = 100_000_000;

/// Fixed-point precision for intermediate calculations (6 decimal places)
const PRECISION: u128 = 1_000_000;

/// Number of blocks to track for rate calculation
const RATE_WINDOW_SIZE: usize = 1000;

/// Emission phase (for dual-track emission)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmissionPhase {
    /// Bootstrap phase (Years 0-4): Base reward + adaptive subsidy
    Bootstrap,
    /// Mature phase (Years 4+): Pure adaptive reward
    Mature,
}

/// Block window for tracking throughput
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockWindow {
    /// Start height of window
    pub start_height: u64,
    /// End height of window
    pub end_height: u64,
    /// Start timestamp
    pub start_timestamp: u64,
    /// End timestamp
    pub end_timestamp: u64,
    /// Number of blocks in window (can be > end - start in DAG)
    pub block_count: u64,
    /// Number of non-empty blocks (for spam filtering)
    pub non_empty_blocks: u64,
}

impl BlockWindow {
    /// Calculate block rate (blocks per second)
    pub fn block_rate(&self) -> f64 {
        let time_elapsed = (self.end_timestamp - self.start_timestamp) as f64;
        if time_elapsed == 0.0 {
            return 0.0;
        }
        self.block_count as f64 / time_elapsed
    }

    /// Calculate economic block rate (non-empty blocks only)
    pub fn economic_rate(&self) -> f64 {
        let time_elapsed = (self.end_timestamp - self.start_timestamp) as f64;
        if time_elapsed == 0.0 {
            return 0.0;
        }
        self.non_empty_blocks as f64 / time_elapsed
    }
}

/// Emission controller - manages adaptive block rewards
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmissionController {
    /// Recent block windows for rate tracking
    block_windows: VecDeque<BlockWindow>,

    /// Current halving era (0 = 2025-2029, 1 = 2029-2033, etc.)
    current_era: u64,

    /// Total emitted this era (for cap enforcement)
    total_emitted_this_era: u64,

    /// Target emission for current era
    era_target_emission: u64,

    /// Current emission phase
    phase: EmissionPhase,

    /// Genesis timestamp
    genesis_timestamp: u64,
}

impl Default for EmissionController {
    fn default() -> Self {
        Self::new(GENESIS_TIMESTAMP)
    }
}

impl EmissionController {
    /// Create new emission controller
    pub fn new(genesis_timestamp: u64) -> Self {
        Self {
            block_windows: VecDeque::with_capacity(RATE_WINDOW_SIZE),
            current_era: 0,
            total_emitted_this_era: 0,
            era_target_emission: BASE_ANNUAL_EMISSION * 4, // 4 years per era
            phase: EmissionPhase::Bootstrap,
            genesis_timestamp,
        }
    }

    /// Add a new block to tracking
    pub fn add_block(&mut self, height: u64, timestamp: u64, has_transactions: bool) {
        // Create or update current window
        if let Some(last_window) = self.block_windows.back_mut() {
            // Extend existing window
            last_window.end_height = height;
            last_window.end_timestamp = timestamp;
            last_window.block_count += 1;
            if has_transactions {
                last_window.non_empty_blocks += 1;
            }
        } else {
            // Create first window
            self.block_windows.push_back(BlockWindow {
                start_height: height,
                end_height: height,
                start_timestamp: timestamp,
                end_timestamp: timestamp,
                block_count: 1,
                non_empty_blocks: if has_transactions { 1 } else { 0 },
            });
        }

        // Limit window size
        if self.block_windows.len() > RATE_WINDOW_SIZE {
            self.block_windows.pop_front();
        }
    }

    /// Calculate smoothed block rate (weighted recent blocks more)
    pub fn calculate_smoothed_rate(&self) -> f64 {
        if self.block_windows.is_empty() {
            return 0.166; // Default: 6-second blocks (current assumption)
        }

        // Weighted average: recent windows have higher weight
        let mut total_weight = 0.0;
        let mut weighted_rate = 0.0;

        for (i, window) in self.block_windows.iter().enumerate() {
            let weight = (i + 1) as f64; // Linear weighting (recent = higher)
            weighted_rate += window.block_rate() * weight;
            total_weight += weight;
        }

        if total_weight == 0.0 {
            return 0.166;
        }

        (weighted_rate / total_weight).max(0.001) // Minimum 0.001 blocks/sec
    }

    /// Calculate economic block rate (exclude empty spam blocks)
    pub fn calculate_economic_rate(&self) -> f64 {
        if self.block_windows.is_empty() {
            return 0.166;
        }

        let mut total_weight = 0.0;
        let mut weighted_rate = 0.0;

        for (i, window) in self.block_windows.iter().enumerate() {
            let weight = (i + 1) as f64;
            weighted_rate += window.economic_rate() * weight;
            total_weight += weight;
        }

        if total_weight == 0.0 {
            return 0.166;
        }

        (weighted_rate / total_weight).max(0.001)
    }

    /// Update era based on current timestamp
    pub fn update_era(&mut self, current_timestamp: u64) {
        let elapsed_seconds = current_timestamp.saturating_sub(self.genesis_timestamp);
        let new_era = elapsed_seconds / SECONDS_PER_HALVING;

        if new_era > self.current_era {
            info!(
                "📅 Era transition: {} → {} (halving emission)",
                self.current_era, new_era
            );

            self.current_era = new_era;
            self.total_emitted_this_era = 0;

            // Halve emission target each era
            self.era_target_emission = (BASE_ANNUAL_EMISSION * 4) >> new_era;

            // Transition to mature phase after first era
            if new_era >= 1 {
                self.phase = EmissionPhase::Mature;
            }
        }
    }

    /// Calculate adaptive block reward
    ///
    /// Uses integer arithmetic with 128-bit precision to avoid rounding errors.
    ///
    /// Formula: reward = TARGET_ANNUAL_EMISSION / blocks_expected_this_year
    pub fn calculate_adaptive_reward(
        &self,
        current_timestamp: u64,
        recent_block_rate: f64,
        total_supply: u64,
    ) -> Result<u64> {
        const QUG_MAX_SUPPLY: u64 = 2_100_000_000_000_000; // 21M QUG

        // Safety 1: Hard cap enforcement
        if total_supply >= QUG_MAX_SUPPLY {
            debug!("🛑 Max supply reached - no more rewards");
            return Ok(0);
        }

        // Safety 2: Era check (64 halvings = 256 years)
        if self.current_era >= 64 {
            debug!("🛑 Era 64+ reached - emission complete");
            return Ok(0);
        }

        // Safety 3: Sane block rate (0.001 to 100,000 blocks/sec)
        let sane_block_rate = recent_block_rate.clamp(0.001, 100_000.0);

        // Calculate annual target for current era
        let annual_target = self.era_target_emission / 4; // Split 4-year era into annual segments

        // Calculate expected blocks this year
        let expected_blocks_this_year = (sane_block_rate * SECONDS_PER_YEAR) as u128;

        if expected_blocks_this_year == 0 {
            warn!("⚠️  Expected blocks is zero - using minimum reward");
            return Ok(MIN_REWARD);
        }

        // Integer arithmetic with 128-bit precision
        // reward = (annual_target * PRECISION) / expected_blocks / PRECISION
        let reward_fp = (annual_target as u128 * PRECISION) / expected_blocks_this_year;
        let mut reward = (reward_fp / PRECISION) as u64;

        // Safety 4: Enforce bounds
        reward = reward.clamp(MIN_REWARD, MAX_REWARD_PER_BLOCK);

        // Safety 5: Don't exceed remaining era emission
        let remaining_era_emission = self.era_target_emission.saturating_sub(self.total_emitted_this_era);
        let conservative_max = remaining_era_emission / 1000; // Conservative estimate
        reward = reward.min(conservative_max);

        debug!(
            "💰 Adaptive reward calculated: {} QUG (rate: {:.2} blocks/sec, era: {})",
            reward as f64 / 100_000_000.0,
            sane_block_rate,
            self.current_era
        );

        Ok(reward)
    }

    /// Calculate block reward with dual-phase emission
    ///
    /// **Bootstrap Phase (Era 0)**: Base reward + adaptive subsidy
    /// **Mature Phase (Era 1+)**: Pure adaptive reward
    pub fn calculate_block_reward(
        &mut self,
        current_timestamp: u64,
        total_supply: u64,
    ) -> Result<u64> {
        // Update era based on timestamp
        self.update_era(current_timestamp);

        // Calculate recent block rate
        let block_rate = self.calculate_economic_rate();

        match self.phase {
            EmissionPhase::Bootstrap => {
                // Bootstrap: 0.01 QUG base + adaptive subsidy
                const BASE_REWARD: u64 = 1_000_000; // 0.01 QUG (8 decimals)

                let adaptive_reward = self.calculate_adaptive_reward(
                    current_timestamp,
                    block_rate,
                    total_supply,
                )?;

                // Subtract base from adaptive to get subsidy
                let subsidy = adaptive_reward.saturating_sub(BASE_REWARD);
                let total = BASE_REWARD + subsidy;

                info!(
                    "🎁 Bootstrap reward: {:.8} QUG (base: 0.01, subsidy: {:.8})",
                    total as f64 / 100_000_000.0,
                    subsidy as f64 / 100_000_000.0
                );

                Ok(total)
            }
            EmissionPhase::Mature => {
                // Mature: Pure adaptive reward
                let reward = self.calculate_adaptive_reward(
                    current_timestamp,
                    block_rate,
                    total_supply,
                )?;

                info!(
                    "💎 Mature reward: {:.8} QUG (pure adaptive)",
                    reward as f64 / 100_000_000.0
                );

                Ok(reward)
            }
        }
    }

    /// Record emission (update total emitted this era)
    pub fn record_emission(&mut self, amount: u64) {
        self.total_emitted_this_era += amount;
    }

    /// Get current emission statistics
    pub fn get_stats(&self) -> EmissionStats {
        EmissionStats {
            current_era: self.current_era,
            era_target_emission: self.era_target_emission,
            total_emitted_this_era: self.total_emitted_this_era,
            phase: self.phase,
            current_block_rate: self.calculate_economic_rate(),
            window_count: self.block_windows.len(),
        }
    }

    /// Get current emission phase
    pub fn phase(&self) -> EmissionPhase {
        self.phase
    }

    /// Get current era
    pub fn current_era(&self) -> u64 {
        self.current_era
    }
}

/// Emission statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmissionStats {
    pub current_era: u64,
    pub era_target_emission: u64,
    pub total_emitted_this_era: u64,
    pub phase: EmissionPhase,
    pub current_block_rate: f64,
    pub window_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emission_controller_creation() {
        let controller = EmissionController::new(GENESIS_TIMESTAMP);
        assert_eq!(controller.current_era, 0);
        assert_eq!(controller.phase, EmissionPhase::Bootstrap);
    }

    #[test]
    fn test_block_rate_tracking() {
        let mut controller = EmissionController::new(GENESIS_TIMESTAMP);

        // Simulate 10 blocks at 1 block/second
        for i in 0..10 {
            controller.add_block(i, GENESIS_TIMESTAMP + i, true);
        }

        let rate = controller.calculate_smoothed_rate();
        assert!(rate > 0.5 && rate < 2.0, "Rate should be ~1 block/sec");
    }

    #[test]
    fn test_adaptive_reward_scaling() {
        let controller = EmissionController::new(GENESIS_TIMESTAMP);

        // Test at different throughputs
        let reward_at_10 = controller.calculate_adaptive_reward(
            GENESIS_TIMESTAMP + 1000,
            10.0, // 10 blocks/sec
            0,
        ).unwrap();

        let reward_at_10000 = controller.calculate_adaptive_reward(
            GENESIS_TIMESTAMP + 1000,
            10_000.0, // 10,000 blocks/sec
            0,
        ).unwrap();

        // Reward should scale inversely with throughput
        assert!(reward_at_10 > reward_at_10000);

        // Check ratio is approximately 1000:1
        let ratio = reward_at_10 as f64 / reward_at_10000 as f64;
        assert!(ratio > 900.0 && ratio < 1100.0, "Ratio should be ~1000");
    }

    #[test]
    fn test_era_transition() {
        let mut controller = EmissionController::new(GENESIS_TIMESTAMP);

        // Simulate 4 years passing
        let future_timestamp = GENESIS_TIMESTAMP + SECONDS_PER_HALVING;
        controller.update_era(future_timestamp);

        assert_eq!(controller.current_era, 1);
        assert_eq!(controller.phase, EmissionPhase::Mature);
    }

    #[test]
    fn test_supply_cap_enforcement() {
        let controller = EmissionController::new(GENESIS_TIMESTAMP);

        const MAX_SUPPLY: u64 = 2_100_000_000_000_000;

        let reward = controller.calculate_adaptive_reward(
            GENESIS_TIMESTAMP + 1000,
            10.0,
            MAX_SUPPLY, // At max supply
        ).unwrap();

        assert_eq!(reward, 0, "Reward should be 0 at max supply");
    }

    #[test]
    fn test_annual_emission_invariance() {
        let controller = EmissionController::new(GENESIS_TIMESTAMP);

        // Test that annual emission is approximately constant across throughputs
        for throughput in [1.0, 10.0, 100.0, 1000.0, 10_000.0] {
            let reward_per_block = controller.calculate_adaptive_reward(
                GENESIS_TIMESTAMP + 1000,
                throughput,
                0,
            ).unwrap();

            let blocks_per_year = throughput * SECONDS_PER_YEAR;
            let annual_emission = (reward_per_block as f64) * blocks_per_year;

            // Should be close to BASE_ANNUAL_EMISSION (within 1%)
            let target = BASE_ANNUAL_EMISSION as f64;
            let error = (annual_emission - target).abs() / target;

            assert!(
                error < 0.01,
                "Annual emission should be constant: throughput={}, emission={}, error={}%",
                throughput, annual_emission, error * 100.0
            );
        }
    }
}
