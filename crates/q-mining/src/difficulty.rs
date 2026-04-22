/// Q-Mining Difficulty Adjustment Module — LWMA (v10.3.0)
///
/// Linearly Weighted Moving Average difficulty adjustment.
/// Weights recent blocks more heavily for fast response to hashrate changes.
/// Resistant to timestamp manipulation and oscillation.
///
/// Why LWMA over Dark Gravity Well:
/// - DAG chains have inherently variable block times (concurrent miners)
/// - DGW's 24-block window oscillates on DAG concurrency noise
/// - LWMA's 60-block window smooths it out
/// - Proven on Monero forks with similar hashrate volatility
///
/// Formula:
///   sum_weighted = Σ(i=1..N) [ i × clamp(solvetime_i, 1, target×6) ]
///   sum_weights = N × (N+1) / 2
///   adjustment = (target × sum_weights) / sum_weighted
///   new_difficulty = current × clamp(adjustment, 0.5, 2.0)
///   new_difficulty = max(new_difficulty, MIN_DIFFICULTY)
///
/// Tests: crates/q-mining/tests/mining_fairness_tests.rs (7 LWMA tests passing)

use anyhow::Result;
use std::time::Duration;
use tracing::{info, warn};

/// Minimum difficulty in leading zero bits — never go below this
const MIN_DIFFICULTY_BITS: u32 = 16;

/// Default LWMA window size (number of recent blocks to consider)
const DEFAULT_WINDOW_SIZE: u64 = 60;

/// Maximum solvetime multiplier for clamping outliers (6× target)
const MAX_SOLVETIME_MULTIPLIER: f64 = 6.0;

/// Maximum single-step adjustment factor (prevents oscillation)
const MAX_ADJUSTMENT_FACTOR: f64 = 2.0;

/// Minimum single-step adjustment factor
const MIN_ADJUSTMENT_FACTOR: f64 = 0.5;

/// Minimum number of blocks needed before adjusting
const MIN_BLOCKS_FOR_ADJUSTMENT: usize = 10;

/// Difficulty adjuster using LWMA algorithm
#[derive(Debug, Clone)]
pub struct DifficultyAdjuster {
    pub target_block_time: Duration,
    pub adjustment_window: u64,
}

impl DifficultyAdjuster {
    /// Create new LWMA difficulty adjuster
    ///
    /// target_block_time: desired block interval (e.g., 1 second for 1 bps)
    /// adjustment_window: number of recent blocks to consider (default: 60)
    pub fn new(target_block_time: Duration, adjustment_window: u64) -> Self {
        Self {
            target_block_time,
            adjustment_window: if adjustment_window == 0 { DEFAULT_WINDOW_SIZE } else { adjustment_window },
        }
    }

    /// Calculate next difficulty using LWMA (Linearly Weighted Moving Average)
    ///
    /// current_difficulty: current difficulty in leading-zero-bits
    /// recent_block_times: solve times of recent blocks (oldest first)
    ///
    /// Returns: new difficulty in leading-zero-bits
    pub fn calculate_next_difficulty(
        &self,
        current_difficulty: u32,
        recent_block_times: &[Duration],
    ) -> Result<u32> {
        let n = recent_block_times.len().min(self.adjustment_window as usize);

        // Need minimum data points before adjusting
        if n < MIN_BLOCKS_FOR_ADJUSTMENT {
            return Ok(current_difficulty);
        }

        let target_ms = self.target_block_time.as_millis() as f64;
        if target_ms <= 0.0 {
            return Ok(current_difficulty);
        }

        // LWMA: sum of (weight × clamped_solvetime) where weight = position (1..N)
        // Recent blocks have higher weight
        let sum_weights = (n * (n + 1) / 2) as f64;
        let max_solvetime = target_ms * MAX_SOLVETIME_MULTIPLIER;

        let mut sum_weighted = 0.0;
        for (i, duration) in recent_block_times.iter().take(n).enumerate() {
            let solvetime_ms = duration.as_millis() as f64;
            // Clamp: prevent extreme outliers from skewing the average
            // Min 1ms (prevents division by zero), Max 6× target (prevents time warp)
            let clamped = solvetime_ms.max(1.0).min(max_solvetime);
            sum_weighted += (i as f64 + 1.0) * clamped;
        }

        if sum_weighted <= 0.0 {
            return Ok(current_difficulty);
        }

        // adjustment > 1.0 → blocks too slow → decrease difficulty
        // adjustment < 1.0 → blocks too fast → increase difficulty
        let adjustment = (target_ms * sum_weights) / sum_weighted;

        // Clamp to prevent wild oscillation (max 2× change per adjustment)
        let clamped_adjustment = adjustment.max(MIN_ADJUSTMENT_FACTOR).min(MAX_ADJUSTMENT_FACTOR);

        // Apply adjustment
        let new_difficulty = (current_difficulty as f64 * clamped_adjustment) as u32;

        // Enforce minimum difficulty floor
        let final_difficulty = new_difficulty.max(MIN_DIFFICULTY_BITS);

        if final_difficulty != current_difficulty {
            info!(
                "⚙️ [LWMA] Difficulty adjusted: {} → {} bits (adjustment: {:.4}×, window: {} blocks, avg_solvetime: {:.0}ms, target: {:.0}ms)",
                current_difficulty, final_difficulty, clamped_adjustment, n,
                sum_weighted / sum_weights, target_ms
            );
        }

        Ok(final_difficulty)
    }

    /// Calculate LWMA with full diagnostics (for dashboard/API)
    pub fn calculate_with_diagnostics(
        &self,
        current_difficulty: u32,
        recent_block_times: &[Duration],
    ) -> LwmaDiagnostics {
        let n = recent_block_times.len().min(self.adjustment_window as usize);
        let target_ms = self.target_block_time.as_millis() as f64;

        if n < MIN_BLOCKS_FOR_ADJUSTMENT || target_ms <= 0.0 {
            return LwmaDiagnostics {
                current_difficulty,
                proposed_difficulty: current_difficulty,
                adjustment_factor: 1.0,
                window_size: n,
                avg_solvetime_ms: 0.0,
                target_solvetime_ms: target_ms,
                blocks_too_fast: false,
                blocks_too_slow: false,
                clamped: false,
                insufficient_data: n < MIN_BLOCKS_FOR_ADJUSTMENT,
            };
        }

        let sum_weights = (n * (n + 1) / 2) as f64;
        let max_solvetime = target_ms * MAX_SOLVETIME_MULTIPLIER;

        let mut sum_weighted = 0.0;
        for (i, duration) in recent_block_times.iter().take(n).enumerate() {
            let clamped = (duration.as_millis() as f64).max(1.0).min(max_solvetime);
            sum_weighted += (i as f64 + 1.0) * clamped;
        }

        let avg_solvetime = sum_weighted / sum_weights;
        let raw_adjustment = (target_ms * sum_weights) / sum_weighted;
        let clamped_adjustment = raw_adjustment.max(MIN_ADJUSTMENT_FACTOR).min(MAX_ADJUSTMENT_FACTOR);
        let proposed = ((current_difficulty as f64) * clamped_adjustment) as u32;
        let final_diff = proposed.max(MIN_DIFFICULTY_BITS);

        LwmaDiagnostics {
            current_difficulty,
            proposed_difficulty: final_diff,
            adjustment_factor: clamped_adjustment,
            window_size: n,
            avg_solvetime_ms: avg_solvetime,
            target_solvetime_ms: target_ms,
            blocks_too_fast: avg_solvetime < target_ms * 0.8,
            blocks_too_slow: avg_solvetime > target_ms * 1.2,
            clamped: (raw_adjustment - clamped_adjustment).abs() > 0.001,
            insufficient_data: false,
        }
    }
}

/// Diagnostics for LWMA calculation (for dashboard display)
#[derive(Debug, Clone)]
pub struct LwmaDiagnostics {
    pub current_difficulty: u32,
    pub proposed_difficulty: u32,
    pub adjustment_factor: f64,
    pub window_size: usize,
    pub avg_solvetime_ms: f64,
    pub target_solvetime_ms: f64,
    pub blocks_too_fast: bool,
    pub blocks_too_slow: bool,
    pub clamped: bool,
    pub insufficient_data: bool,
}

/// Calculate difficulty for the next block — pure function of chain history.
/// Same inputs → same output on every node. No stored mutable state.
/// Called at: challenge endpoint, block template creation, block validation.
///
/// Mirrors the emission controller pattern: deterministic, chain-derived,
/// with hard caps and fallback behavior.
///
/// # Arguments
/// * `previous_difficulty_bits` - Current difficulty from the previous block header
/// * `recent_timestamps` - Last N block timestamps (unix seconds, oldest first)
/// * `activation_height` - Height at which LWMA activates
/// * `next_height` - Height of the block being produced
/// * `target_block_time_secs` - Target block interval in seconds (e.g., 1 for 1 bps)
///
/// # Returns
/// Difficulty in leading zero bits for the next block
pub fn calculate_difficulty_for_next_block(
    previous_difficulty_bits: u32,
    recent_timestamps: &[u64],
    activation_height: u64,
    next_height: u64,
    target_block_time_secs: u64,
) -> u32 {
    // Before activation: legacy fixed difficulty
    if next_height < activation_height {
        return LEGACY_DIFFICULTY_BITS;
    }

    let n = recent_timestamps.len();

    // Need one full window of post-activation data before adjusting
    // Per reviewer feedback: require full window (120 blocks), not partial
    if n < LWMA_WINDOW_SIZE {
        return previous_difficulty_bits.max(MIN_DIFFICULTY_BITS);
    }

    let target_ms = (target_block_time_secs * 1000) as f64;
    if target_ms <= 0.0 {
        return previous_difficulty_bits.max(MIN_DIFFICULTY_BITS);
    }

    // LWMA: compute from the last N block timestamps
    // Recent blocks weighted more heavily (linear: 1,2,3...N)
    let window = n.min(LWMA_WINDOW_SIZE);
    let sum_weights = (window * (window + 1) / 2) as f64;
    let max_solvetime = target_ms * MAX_SOLVETIME_MULTIPLIER;

    let mut sum_weighted = 0.0;
    let start = n.saturating_sub(window);
    for i in 1..window {
        let idx = start + i;
        if idx >= n || idx == 0 {
            continue;
        }
        // Solvetime in milliseconds (timestamps are in seconds)
        let solvetime_ms = (recent_timestamps[idx].saturating_sub(recent_timestamps[idx - 1])) as f64 * 1000.0;
        let clamped = solvetime_ms.max(1.0).min(max_solvetime);
        sum_weighted += (i as f64) * clamped;
    }

    if sum_weighted <= 0.0 {
        return previous_difficulty_bits.max(MIN_DIFFICULTY_BITS);
    }

    // adjustment > 1.0 → blocks too slow → decrease difficulty
    // adjustment < 1.0 → blocks too fast → increase difficulty
    let adjustment = (target_ms * sum_weights) / sum_weighted;

    // Clamp: max 2× change per step (prevents oscillation)
    let clamped = adjustment.max(MIN_ADJUSTMENT_FACTOR).min(MAX_ADJUSTMENT_FACTOR);

    // Apply adjustment to previous difficulty
    let new_difficulty = (previous_difficulty_bits as f64 * clamped) as u32;

    // Floor + ceiling: never below minimum, never above maximum
    // MAX_DIFFICULTY_BITS prevents LWMA runaway from turbo-sync burst blocks
    let final_difficulty = new_difficulty.max(MIN_DIFFICULTY_BITS).min(MAX_DIFFICULTY_BITS);

    if final_difficulty != previous_difficulty_bits {
        info!(
            "⚙️ [LWMA-PURE] Difficulty: {} → {} bits (adjustment: {:.4}×, window: {}, avg_solvetime: {:.0}ms, target: {:.0}ms, height: {})",
            previous_difficulty_bits, final_difficulty, clamped, window,
            sum_weighted / sum_weights, target_ms, next_height
        );
    }

    final_difficulty
}

/// LWMA window size (number of recent blocks to consider)
/// Per v2 review: 120 blocks (up from 60 in v1)
const LWMA_WINDOW_SIZE: usize = 120;

/// Legacy difficulty before LWMA activation
const LEGACY_DIFFICULTY_BITS: u32 = 16;

/// Maximum difficulty LWMA can set — prevents runaway from turbo-sync burst blocks.
/// 32 bits ≈ solvable at ~5 GH/s in ~1 second. Handlers.rs has Q_MAX_DIFFICULTY_BITS override.
const MAX_DIFFICULTY_BITS: u32 = 32;

/// Count leading zero bits in a 32-byte hash
pub fn count_leading_zero_bits(hash: &[u8; 32]) -> u32 {
    let mut count = 0u32;
    for &byte in hash.iter() {
        if byte == 0 {
            count += 8;
        } else {
            count += byte.leading_zeros();
            break;
        }
    }
    count
}

/// Difficulty target for mining (byte-level representation)
#[derive(Debug, Clone, Copy)]
pub struct DifficultyTarget {
    pub leading_zeros: u32,
    pub target_hash: [u8; 32],
}

impl DifficultyTarget {
    /// Create difficulty target from leading zeros requirement
    pub fn from_leading_zeros(leading_zeros: u32) -> Self {
        let mut target_hash = [0xFF; 32];
        let full_bytes = (leading_zeros / 8) as usize;
        let remaining_bits = leading_zeros % 8;

        for i in 0..full_bytes.min(32) {
            target_hash[i] = 0x00;
        }

        if full_bytes < 32 && remaining_bits > 0 {
            target_hash[full_bytes] = 0xFF >> remaining_bits;
        }

        Self {
            leading_zeros,
            target_hash,
        }
    }

    /// Check if a hash meets this difficulty target
    pub fn meets_target(&self, hash: &[u8; 32]) -> bool {
        for i in 0..32 {
            if hash[i] > self.target_hash[i] {
                return false;
            } else if hash[i] < self.target_hash[i] {
                return true;
            }
        }
        true // Equal counts as meeting target
    }
}
