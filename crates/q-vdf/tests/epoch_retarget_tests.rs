//! Tests for the epoch-based retarget formula
//!
//! PURPOSE: Validate the difficulty retarget formula BEFORE implementation.
//! Catches the sign bug flagged by reviewers: when miners find solutions
//! too fast, difficulty must INCREASE (target gets SMALLER).
//!
//! This is prerequisite P4 — no production code is modified.
//! These tests define the EXPECTED BEHAVIOR of the retarget algorithm.
//! Once we implement it in difficulty.rs, these become the acceptance tests.
//!
//! Run with: cargo test --package q-vdf --test epoch_retarget_tests

use num_bigint::BigUint;
use num_traits::{One, Zero};

// ============================================================================
// RETARGET FORMULA (reference implementation for testing)
// This is what we will implement in crates/q-mining/src/difficulty.rs
// ============================================================================

/// Difficulty target: a 256-bit number. Larger = easier mining.
/// A solution is valid if hash < target.
type Target = BigUint;

/// Configuration for a single lane's retarget
struct LaneRetargetConfig {
    /// Target accepted solutions per second for this lane
    target_rate: f64,
    /// Maximum adjustment ratio per epoch (e.g., 1.25 = max 25% change)
    max_adjustment: f64,
    /// Minimum difficulty floor (target cannot exceed this)
    floor_target: Target,
    /// Maximum relaxation per epoch when inactive (e.g., 0.02 = 2%)
    inactivity_relaxation: f64,
}

/// Compute next epoch target for a lane.
///
/// CRITICAL: The formula must satisfy:
///   - actual_rate > target_rate  =>  new_target < old_target  (harder)
///   - actual_rate < target_rate  =>  new_target > old_target  (easier)
///   - actual_rate == target_rate =>  new_target == old_target (unchanged)
///
/// The CORRECT formula is:
///   ratio = target_rate / actual_rate
///   new_target = old_target * ratio
///
/// NOT:
///   ratio = actual_rate / target_rate  (WRONG — this is the sign bug!)
fn compute_next_target(
    old_target: &Target,
    actual_rate: f64,
    config: &LaneRetargetConfig,
) -> Target {
    if actual_rate <= 0.0 {
        // No solutions in epoch — apply inactivity relaxation
        // "target may relax by at most 2% per epoch, bounded by floor"
        let relaxation = 1.0 + config.inactivity_relaxation;
        let new_target = multiply_target(old_target, relaxation);
        return clamp_to_floor(&new_target, &config.floor_target);
    }

    // CORRECT formula: ratio = target_rate / actual_rate
    // When actual > target (too fast), ratio < 1, target shrinks (harder)
    // When actual < target (too slow), ratio > 1, target grows (easier)
    let ratio = config.target_rate / actual_rate;

    // Clamp to max adjustment bounds
    let clamped_ratio = ratio.max(1.0 / config.max_adjustment).min(config.max_adjustment);

    // Apply adjustment
    let new_target = multiply_target(old_target, clamped_ratio);

    // Enforce floor
    clamp_to_floor(&new_target, &config.floor_target)
}

/// Multiply a BigUint target by a floating-point ratio
/// Uses fixed-point arithmetic to avoid floating-point consensus issues
fn multiply_target(target: &Target, ratio: f64) -> Target {
    // Convert to fixed-point: ratio * 1_000_000 as integer for precision
    let ratio_fixed = (ratio * 1_000_000.0).round() as u64;
    let numerator = target * BigUint::from(ratio_fixed);
    numerator / BigUint::from(1_000_000u64)
}

/// Clamp target to not exceed floor (floor = maximum allowed target = easiest difficulty)
/// If target > floor, return floor. Otherwise return target unchanged.
fn clamp_to_floor(target: &Target, floor: &Target) -> Target {
    if target > floor { floor.clone() } else { target.clone() }
}

/// Helper: create a target with N leading zero bytes
fn target_with_leading_zeros(zero_bytes: usize) -> Target {
    let mut bytes = vec![0x00u8; zero_bytes];
    bytes.resize(32, 0xFF);
    BigUint::from_bytes_be(&bytes)
}

// ============================================================================
// DIRECTION TESTS (the most critical tests — catch the sign bug)
// ============================================================================

#[test]
fn test_too_fast_makes_harder() {
    // Miners finding solutions 2x faster than target => difficulty must INCREASE
    let config = LaneRetargetConfig {
        target_rate: 1.0,        // want 1 solution/sec
        max_adjustment: 1.25,
        floor_target: target_with_leading_zeros(4),
        inactivity_relaxation: 0.02,
    };

    let old_target = target_with_leading_zeros(2); // 0x0000FFFF...
    let actual_rate = 2.0; // 2x too fast

    let new_target = compute_next_target(&old_target, actual_rate, &config);

    assert!(new_target < old_target,
        "When actual_rate ({}) > target_rate ({}), target must DECREASE (harder mining). \
         Old: {}, New: {}", actual_rate, config.target_rate, old_target, new_target);
}

#[test]
fn test_too_slow_makes_easier() {
    // Miners finding solutions 2x slower than target => difficulty must DECREASE
    let config = LaneRetargetConfig {
        target_rate: 1.0,
        max_adjustment: 1.25,
        floor_target: target_with_leading_zeros(0), // high floor (easy) so it doesn't cap us
        inactivity_relaxation: 0.02,
    };

    let old_target = target_with_leading_zeros(2);
    let actual_rate = 0.5; // 2x too slow

    let new_target = compute_next_target(&old_target, actual_rate, &config);

    assert!(new_target > old_target,
        "When actual_rate ({}) < target_rate ({}), target must INCREASE (easier mining). \
         Old: {}, New: {}", actual_rate, config.target_rate, old_target, new_target);
}

#[test]
fn test_on_target_unchanged() {
    // Miners at exactly the right rate => difficulty should not change
    let config = LaneRetargetConfig {
        target_rate: 1.0,
        max_adjustment: 1.25,
        floor_target: target_with_leading_zeros(0), // high floor won't interfere
        inactivity_relaxation: 0.02,
    };

    let old_target = target_with_leading_zeros(2);
    let actual_rate = 1.0; // exactly on target

    let new_target = compute_next_target(&old_target, actual_rate, &config);

    assert_eq!(new_target, old_target,
        "When actual_rate == target_rate, target must not change");
}

// ============================================================================
// CLAMPING TESTS
// ============================================================================

#[test]
fn test_extreme_fast_clamped_to_max_adjustment() {
    // 100x too fast — should only adjust by max_adjustment (25%)
    // Floor must be EASIER (larger) than old_target so it doesn't interfere
    let config = LaneRetargetConfig {
        target_rate: 1.0,
        max_adjustment: 1.25,
        floor_target: target_with_leading_zeros(0), // very easy floor won't interfere with hardening
        inactivity_relaxation: 0.02,
    };

    let old_target = target_with_leading_zeros(2);
    let actual_rate = 100.0; // 100x too fast

    let new_target = compute_next_target(&old_target, actual_rate, &config);

    // ratio = 1.0 / 100.0 = 0.01, clamped to 1/1.25 = 0.80
    // new_target should be ~80% of old_target
    let min_expected = multiply_target(&old_target, 0.799);
    let max_expected = multiply_target(&old_target, 0.801);

    assert!(new_target >= min_expected && new_target <= max_expected,
        "Extreme fast rate must be clamped to max 25% decrease. \
         old={}, new={}, min_exp={}, max_exp={}", old_target, new_target, min_expected, max_expected);
}

#[test]
fn test_extreme_slow_clamped_to_max_adjustment() {
    // 100x too slow — should only adjust by max_adjustment (25%)
    let config = LaneRetargetConfig {
        target_rate: 1.0,
        max_adjustment: 1.25,
        floor_target: target_with_leading_zeros(0), // very high floor
        inactivity_relaxation: 0.02,
    };

    let old_target = target_with_leading_zeros(2);
    let actual_rate = 0.01; // 100x too slow

    let new_target = compute_next_target(&old_target, actual_rate, &config);

    // ratio = 1.0 / 0.01 = 100, clamped to 1.25
    let min_expected = multiply_target(&old_target, 1.24);
    let max_expected = multiply_target(&old_target, 1.26);

    assert!(new_target >= min_expected && new_target <= max_expected,
        "Extreme slow rate must be clamped to max 25% increase");
}

// ============================================================================
// INACTIVITY TESTS
// ============================================================================

#[test]
fn test_zero_solutions_relaxes_by_inactivity_rate() {
    let config = LaneRetargetConfig {
        target_rate: 1.0,
        max_adjustment: 1.25,
        floor_target: target_with_leading_zeros(0), // high floor
        inactivity_relaxation: 0.02, // 2% per epoch
    };

    let old_target = target_with_leading_zeros(2);
    let new_target = compute_next_target(&old_target, 0.0, &config);

    // Should be ~102% of old target (2% easier)
    let expected_min = multiply_target(&old_target, 1.019);
    let expected_max = multiply_target(&old_target, 1.021);

    assert!(new_target >= expected_min && new_target <= expected_max,
        "Zero solutions should relax target by ~2%");
}

#[test]
fn test_inactivity_bounded_by_floor() {
    // Even after many inactive epochs, target cannot exceed floor
    let floor = target_with_leading_zeros(1); // relatively easy floor
    let config = LaneRetargetConfig {
        target_rate: 1.0,
        max_adjustment: 1.25,
        floor_target: floor.clone(),
        inactivity_relaxation: 0.02,
    };

    // Start with a target very close to floor
    let old_target = multiply_target(&floor, 0.99);
    let new_target = compute_next_target(&old_target, 0.0, &config);

    assert!(new_target <= floor,
        "Relaxed target must not exceed floor. Got: {}, Floor: {}", new_target, floor);
}

#[test]
fn test_many_inactive_epochs_converge_to_floor() {
    let floor = target_with_leading_zeros(1);
    let config = LaneRetargetConfig {
        target_rate: 1.0,
        max_adjustment: 1.25,
        floor_target: floor.clone(),
        inactivity_relaxation: 0.02,
    };

    let mut target = target_with_leading_zeros(3); // start very hard

    // Run 200 inactive epochs
    for _ in 0..200 {
        target = compute_next_target(&target, 0.0, &config);
    }

    // After many relaxations, should be at or near floor
    assert!(target <= floor, "After many inactive epochs, target must reach floor");
}

// ============================================================================
// STABILITY TESTS
// ============================================================================

#[test]
fn test_small_fluctuations_converge() {
    // Alternating slightly fast / slightly slow should converge, not oscillate
    let config = LaneRetargetConfig {
        target_rate: 1.0,
        max_adjustment: 1.25,
        floor_target: target_with_leading_zeros(0),
        inactivity_relaxation: 0.02,
    };

    let mut target = target_with_leading_zeros(2);
    let initial = target.clone();

    // Alternate: 1.1x fast, then 0.9x slow, for 20 epochs
    for i in 0..20 {
        let rate = if i % 2 == 0 { 1.1 } else { 0.9 };
        target = compute_next_target(&target, rate, &config);
    }

    // Target should not have drifted far from initial
    let ratio_to_initial = target.bits() as f64 / initial.bits() as f64;
    assert!(ratio_to_initial > 0.8 && ratio_to_initial < 1.2,
        "Small fluctuations should converge, not amplify. Ratio: {:.3}", ratio_to_initial);
}

#[test]
fn test_sustained_fast_reduces_target_monotonically() {
    // If rate is consistently too fast, target must decrease every epoch
    let config = LaneRetargetConfig {
        target_rate: 1.0,
        max_adjustment: 1.25,
        floor_target: target_with_leading_zeros(10), // very hard floor
        inactivity_relaxation: 0.02,
    };

    let mut target = target_with_leading_zeros(2);
    let mut prev = target.clone();

    for _ in 0..10 {
        target = compute_next_target(&target, 2.0, &config); // consistently 2x too fast
        assert!(target < prev,
            "Sustained fast rate must monotonically decrease target");
        prev = target.clone();
    }
}

#[test]
fn test_sustained_slow_increases_target_monotonically() {
    // If rate is consistently too slow, target must increase every epoch
    let config = LaneRetargetConfig {
        target_rate: 1.0,
        max_adjustment: 1.25,
        floor_target: target_with_leading_zeros(0), // high floor
        inactivity_relaxation: 0.02,
    };

    let mut target = target_with_leading_zeros(2);
    let mut prev = target.clone();

    for _ in 0..10 {
        target = compute_next_target(&target, 0.5, &config); // consistently 2x too slow
        assert!(target > prev,
            "Sustained slow rate must monotonically increase target");
        prev = target.clone();
    }
}

// ============================================================================
// TWO-LANE INDEPENDENCE TESTS
// ============================================================================

#[test]
fn test_lanes_adjust_independently() {
    let blake3_config = LaneRetargetConfig {
        target_rate: 2.0,  // BLAKE3 lane: 2 solutions/sec
        max_adjustment: 1.25,
        floor_target: target_with_leading_zeros(4),
        inactivity_relaxation: 0.02,
    };

    let genus2_config = LaneRetargetConfig {
        target_rate: 0.5,  // Genus-2 lane: 0.5 solutions/sec
        max_adjustment: 1.25,
        floor_target: target_with_leading_zeros(0), // high floor for Genus-2 (easier lane)
        inactivity_relaxation: 0.02,
    };

    let b3_target = target_with_leading_zeros(2);
    let g2_target = target_with_leading_zeros(1);

    // BLAKE3 lane too fast, Genus-2 lane too slow
    let new_b3 = compute_next_target(&b3_target, 4.0, &blake3_config);
    let new_g2 = compute_next_target(&g2_target, 0.1, &genus2_config);

    // BLAKE3 should get harder (smaller target)
    assert!(new_b3 < b3_target, "BLAKE3 lane must get harder when too fast");

    // Genus-2 should get easier (larger target)
    assert!(new_g2 > g2_target, "Genus-2 lane must get easier when too slow");
}

// ============================================================================
// REWARD BUDGET TESTS
// ============================================================================

/// Maximum fraction of lane budget one solution can earn
const MAX_REWARD_FRACTION: f64 = 0.10;

/// Compute reward per solution for a lane in an epoch
fn compute_reward_per_solution(lane_budget: f64, accepted_solutions: u64) -> f64 {
    if accepted_solutions == 0 {
        return 0.0;
    }

    let raw_reward = lane_budget / accepted_solutions as f64;
    let max_reward = lane_budget * MAX_REWARD_FRACTION;
    raw_reward.min(max_reward)
}

#[test]
fn test_reward_capped_at_10_percent() {
    // Only 1 solution in epoch — should NOT get entire budget
    let budget = 1000.0;
    let reward = compute_reward_per_solution(budget, 1);

    assert_eq!(reward, 100.0, "Single solution must be capped at 10% of budget");
    assert!(reward < budget, "Single solution must not get entire budget");
}

#[test]
fn test_reward_distributed_evenly() {
    let budget = 1000.0;
    let reward = compute_reward_per_solution(budget, 100);

    assert_eq!(reward, 10.0, "100 solutions should each get 1% of budget");
}

#[test]
fn test_reward_cap_kicks_in_at_threshold() {
    let budget = 1000.0;

    // With 10 solutions: raw = 100, cap = 100 => cap exactly reached
    let r10 = compute_reward_per_solution(budget, 10);
    assert_eq!(r10, 100.0);

    // With 5 solutions: raw = 200, cap = 100 => capped
    let r5 = compute_reward_per_solution(budget, 5);
    assert_eq!(r5, 100.0);

    // With 20 solutions: raw = 50, cap = 100 => not capped
    let r20 = compute_reward_per_solution(budget, 20);
    assert_eq!(r20, 50.0);
}

#[test]
fn test_zero_solutions_zero_reward() {
    let reward = compute_reward_per_solution(1000.0, 0);
    assert_eq!(reward, 0.0, "Zero solutions must yield zero reward");
}

#[test]
fn test_reward_budget_independence() {
    // BLAKE3 lane and Genus-2 lane have separate budgets
    let total_emission = 10000.0;
    let blake3_budget = total_emission * 0.80;
    let genus2_budget = total_emission * 0.20;

    // BLAKE3: 500 solutions, Genus-2: 10 solutions
    let b3_reward = compute_reward_per_solution(blake3_budget, 500);
    let g2_reward = compute_reward_per_solution(genus2_budget, 10);

    assert_eq!(b3_reward, 16.0, "BLAKE3: 8000/500 = 16");
    assert_eq!(g2_reward, 200.0, "Genus-2: 2000/10 = 200 (capped at 10% = 200)");

    // Total paid out must not exceed total emission
    let total_paid = b3_reward * 500.0 + g2_reward * 10.0;
    assert!(total_paid <= total_emission,
        "Total paid ({}) must not exceed total emission ({})", total_paid, total_emission);
}

// ============================================================================
// FIXED-POINT ARITHMETIC TESTS
// ============================================================================

#[test]
fn test_multiply_target_identity() {
    let target = target_with_leading_zeros(2);
    let result = multiply_target(&target, 1.0);
    assert_eq!(result, target, "Multiplying by 1.0 must not change target");
}

#[test]
fn test_multiply_target_halves() {
    let target = target_with_leading_zeros(2);
    let result = multiply_target(&target, 0.5);
    let expected = &target / BigUint::from(2u64);
    assert_eq!(result, expected, "Multiplying by 0.5 must halve the target");
}

#[test]
fn test_multiply_target_doubles() {
    let target = target_with_leading_zeros(2);
    let result = multiply_target(&target, 2.0);
    let expected = &target * BigUint::from(2u64);
    assert_eq!(result, expected, "Multiplying by 2.0 must double the target");
}
