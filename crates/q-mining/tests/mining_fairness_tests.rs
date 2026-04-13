/// Mining Fairness Tests — v10.3.0
///
/// Comprehensive tests for difficulty-weighted rewards, LWMA difficulty adjustment,
/// and dual-lane hybrid mining. Written BEFORE implementation as acceptance criteria.
///
/// Run: cargo test --package q-mining --test mining_fairness_tests
///
/// These tests verify the mathematical properties that must hold on a $1B mainnet.
/// Every test documents WHY the property matters.

use std::time::Duration;

// ============================================================================
// PHASE A: Difficulty-Weighted Reward Distribution
// ============================================================================

/// Calculate the number of leading zero BITS in a 32-byte hash
fn leading_zero_bits(hash: &[u8; 32]) -> u32 {
    let mut zeros = 0u32;
    for byte in hash.iter() {
        if *byte == 0 {
            zeros += 8;
        } else {
            zeros += byte.leading_zeros();
            break;
        }
    }
    zeros
}

/// Calculate difficulty weight from leading zero bits
/// Weight = 2^(leading_zeros) — exponential scaling
fn difficulty_weight(hash: &[u8; 32]) -> f64 {
    let zeros = leading_zero_bits(hash);
    2.0_f64.powf(zeros as f64)
}

/// Calculate difficulty-weighted rewards for a set of solutions
/// Returns reward per solution (in atomic units)
fn calculate_weighted_rewards(total_reward: u128, hashes: &[[u8; 32]]) -> Vec<u128> {
    if hashes.is_empty() {
        return vec![];
    }
    if hashes.len() == 1 {
        return vec![total_reward];
    }

    let weights: Vec<f64> = hashes.iter().map(|h| difficulty_weight(h)).collect();
    let total_weight: f64 = weights.iter().sum();

    if total_weight == 0.0 {
        // Fallback: equal split
        let per = total_reward / hashes.len() as u128;
        return vec![per; hashes.len()];
    }

    // Integer-only proportional distribution to avoid f64 precision loss
    // on u128 values (critical for $1B mainnet — no dust creation/destruction)
    //
    // Algorithm: scale weights to u128, compute share as (total * weight_i) / total_weight
    // Assign remainder to highest-weight miner (deterministic, no rounding leak)
    let scale = 1_000_000_000_u128; // Scale factor for integer precision
    let int_weights: Vec<u128> = weights.iter()
        .map(|w| (*w * scale as f64) as u128)
        .collect();
    let int_total_weight: u128 = int_weights.iter().sum();

    if int_total_weight == 0 {
        let per = total_reward / hashes.len() as u128;
        return vec![per; hashes.len()];
    }

    let mut rewards: Vec<u128> = int_weights.iter()
        .map(|w| {
            // Use u128 division-first to avoid overflow:
            // reward = (total / total_weight) * w + ((total % total_weight) * w) / total_weight
            let quotient = total_reward / int_total_weight;
            let remainder = total_reward % int_total_weight;
            quotient * w + (remainder * w) / int_total_weight
        })
        .collect();

    // Assign remainder to highest-weight miner (deterministic)
    let distributed: u128 = rewards.iter().sum();
    let remainder = total_reward.saturating_sub(distributed);
    if remainder > 0 {
        let max_idx = int_weights.iter()
            .enumerate()
            .max_by_key(|(_, w)| *w)
            .map(|(i, _)| i)
            .unwrap_or(0);
        rewards[max_idx] += remainder;
    }

    rewards
}

/// Create a mock hash with exactly N leading zero bits
fn mock_hash_with_zeros(leading_zeros: u32) -> [u8; 32] {
    let mut hash = [0xFFu8; 32];
    let full_bytes = (leading_zeros / 8) as usize;
    let remaining_bits = leading_zeros % 8;

    for i in 0..full_bytes.min(32) {
        hash[i] = 0x00;
    }
    if full_bytes < 32 && remaining_bits > 0 {
        hash[full_bytes] = 0xFF >> remaining_bits;
    }
    hash
}

// --- Phase A Tests ---

#[test]
fn test_leading_zero_bits_basic() {
    let h16 = mock_hash_with_zeros(16);
    assert_eq!(leading_zero_bits(&h16), 16);

    let h24 = mock_hash_with_zeros(24);
    assert_eq!(leading_zero_bits(&h24), 24);

    let h0 = [0xFF; 32];
    assert_eq!(leading_zero_bits(&h0), 0);

    let h256 = [0x00; 32];
    assert_eq!(leading_zero_bits(&h256), 256);
}

#[test]
fn test_single_solution_gets_all_reward() {
    let hashes = vec![mock_hash_with_zeros(16)];
    let rewards = calculate_weighted_rewards(1_000_000_000, &hashes);
    assert_eq!(rewards.len(), 1);
    assert_eq!(rewards[0], 1_000_000_000);
}

#[test]
fn test_equal_difficulty_equal_rewards() {
    // Three miners with identical difficulty → equal split
    let hashes = vec![
        mock_hash_with_zeros(16),
        mock_hash_with_zeros(16),
        mock_hash_with_zeros(16),
    ];
    let rewards = calculate_weighted_rewards(1_000_000_000, &hashes);
    assert_eq!(rewards.len(), 3);
    // Allow 1 unit rounding tolerance
    let expected = 1_000_000_000 / 3;
    for r in &rewards {
        assert!((*r as i128 - expected as i128).abs() <= 1,
            "Expected ~{}, got {}", expected, r);
    }
}

#[test]
fn test_harder_solution_gets_more_reward() {
    // Miner A: 16 leading zeros (minimum)
    // Miner B: 24 leading zeros (256x harder)
    let hashes = vec![
        mock_hash_with_zeros(16),
        mock_hash_with_zeros(24),
    ];
    let rewards = calculate_weighted_rewards(1_000_000_000, &hashes);
    assert_eq!(rewards.len(), 2);
    // B has 2^24 / 2^16 = 256x the weight
    // B should get ~256/257 = 99.6% of reward
    assert!(rewards[1] > rewards[0] * 200,
        "Harder solution should get >>200x more: A={}, B={}", rewards[0], rewards[1]);
}

#[test]
fn test_no_reward_leak() {
    // Sum of all rewards must equal total (conservation of coins)
    // This is critical for a $1B mainnet — any leak is money creation/destruction
    for n in 1..=250 {
        let hashes: Vec<[u8; 32]> = (0..n)
            .map(|i| mock_hash_with_zeros(16 + (i % 8) as u32))
            .collect();
        let total = 82_031_250_000_000_000_000_000_u128; // Realistic block reward
        let rewards = calculate_weighted_rewards(total, &hashes);
        let sum: u128 = rewards.iter().sum();
        assert_eq!(sum, total,
            "Reward leak! n={}, total={}, sum={}, diff={}", n, total, sum, total as i128 - sum as i128);
    }
}

#[test]
fn test_empty_solutions_no_panic() {
    let rewards = calculate_weighted_rewards(1_000_000_000, &[]);
    assert!(rewards.is_empty());
}

#[test]
fn test_extreme_difficulty_spread() {
    // GPU miner with minimum difficulty vs CPU miner with 32 extra zero bits
    // This is the real-world scenario we're trying to fix
    let hashes = vec![
        mock_hash_with_zeros(16), // GPU: barely meets target
        mock_hash_with_zeros(48), // CPU: found something 2^32 times harder
    ];
    let rewards = calculate_weighted_rewards(1_000_000_000, &hashes);
    // CPU miner should get almost everything
    assert!(rewards[1] > 999_000_000,
        "CPU miner with 48 zeros should get >99.9% vs GPU with 16 zeros: CPU={}, GPU={}",
        rewards[1], rewards[0]);
}

#[test]
fn test_many_equal_vs_one_hard() {
    // 100 GPU miners with minimum difficulty vs 1 CPU miner with 24 zeros
    let mut hashes: Vec<[u8; 32]> = (0..100)
        .map(|_| mock_hash_with_zeros(16))
        .collect();
    hashes.push(mock_hash_with_zeros(24)); // CPU miner

    let rewards = calculate_weighted_rewards(1_000_000_000, &hashes);
    // CPU miner (24 zeros) has 2^24 weight vs 100 × 2^16 = 100 × 65536 = 6,553,600
    // CPU weight: 16,777,216 vs GPU total: 6,553,600
    // CPU share: 16,777,216 / (16,777,216 + 6,553,600) = 71.9%
    assert!(rewards[100] > 700_000_000,
        "CPU miner should get >70%: CPU={}, total_GPU={}",
        rewards[100], rewards[..100].iter().sum::<u128>());
}

#[test]
fn test_reward_monotonicity() {
    // Harder solutions must always get >= softer solutions
    let hashes = vec![
        mock_hash_with_zeros(16),
        mock_hash_with_zeros(18),
        mock_hash_with_zeros(20),
        mock_hash_with_zeros(22),
        mock_hash_with_zeros(24),
    ];
    let rewards = calculate_weighted_rewards(1_000_000_000, &hashes);
    for i in 1..rewards.len() {
        assert!(rewards[i] >= rewards[i-1],
            "Reward not monotonic: rewards[{}]={} < rewards[{}]={}",
            i, rewards[i], i-1, rewards[i-1]);
    }
}

// ============================================================================
// PHASE B: LWMA Difficulty Adjustment
// ============================================================================

/// LWMA (Linearly Weighted Moving Average) difficulty adjustment
/// Returns new difficulty based on recent block solve times
fn lwma_next_difficulty(
    current_difficulty: u32,
    recent_solve_times_ms: &[u64],
    target_block_time_ms: u64,
    window_size: usize,
) -> u32 {
    let n = recent_solve_times_ms.len().min(window_size);
    if n < 10 {
        return current_difficulty; // Not enough data
    }

    let target = target_block_time_ms as f64;
    let sum_weights = (n * (n + 1) / 2) as f64;
    let max_solvetime = target * 6.0; // Clamp outliers

    let mut sum_weighted = 0.0;
    for (i, &st) in recent_solve_times_ms.iter().take(n).enumerate() {
        let clamped = (st as f64).max(1.0).min(max_solvetime);
        sum_weighted += (i as f64 + 1.0) * clamped;
    }

    let adjustment = (target * sum_weights) / sum_weighted;
    let clamped = adjustment.max(0.5).min(2.0); // Max 2x change per adjustment

    let new_diff = (current_difficulty as f64 * clamped) as u32;
    new_diff.max(16) // Never below 16 leading zero bits
}

#[test]
fn test_lwma_stable_at_target() {
    // If all blocks arrive at exactly the target time, difficulty shouldn't change
    let solve_times: Vec<u64> = vec![1000; 60]; // 60 blocks, all 1000ms (1 bps target)
    let new_diff = lwma_next_difficulty(16, &solve_times, 1000, 60);
    assert_eq!(new_diff, 16, "Difficulty should not change when at target");
}

#[test]
fn test_lwma_increases_on_fast_blocks() {
    // Blocks arriving too fast → difficulty should increase
    let solve_times: Vec<u64> = vec![200; 60]; // 200ms blocks (5x too fast)
    let new_diff = lwma_next_difficulty(16, &solve_times, 1000, 60);
    assert!(new_diff > 16,
        "Difficulty should increase when blocks are fast: got {}", new_diff);
}

#[test]
fn test_lwma_decreases_on_slow_blocks() {
    // Blocks arriving too slow → difficulty should decrease
    let solve_times: Vec<u64> = vec![5000; 60]; // 5000ms blocks (5x too slow)
    let new_diff = lwma_next_difficulty(20, &solve_times, 1000, 60);
    assert!(new_diff < 20,
        "Difficulty should decrease when blocks are slow: got {}", new_diff);
}

#[test]
fn test_lwma_clamp_prevents_oscillation() {
    // Even with extreme block times, adjustment clamped to [0.5x, 2.0x]
    let solve_times: Vec<u64> = vec![1; 60]; // Near-instant blocks
    let new_diff = lwma_next_difficulty(16, &solve_times, 1000, 60);
    assert!(new_diff <= 32,
        "Difficulty should be clamped to max 2x: got {}", new_diff);

    let solve_times_slow: Vec<u64> = vec![100_000; 60]; // 100s blocks
    let new_diff_slow = lwma_next_difficulty(32, &solve_times_slow, 1000, 60);
    assert!(new_diff_slow >= 16,
        "Difficulty should be clamped to min 0.5x: got {}", new_diff_slow);
}

#[test]
fn test_lwma_minimum_difficulty_floor() {
    // Difficulty should never go below 16 (our absolute minimum)
    let solve_times: Vec<u64> = vec![100_000; 60]; // Very slow
    let new_diff = lwma_next_difficulty(16, &solve_times, 1000, 60);
    assert!(new_diff >= 16,
        "Difficulty should never go below floor: got {}", new_diff);
}

#[test]
fn test_lwma_needs_minimum_data() {
    // With fewer than 10 blocks, return current difficulty unchanged
    let solve_times: Vec<u64> = vec![200; 5]; // Only 5 blocks
    let new_diff = lwma_next_difficulty(16, &solve_times, 1000, 60);
    assert_eq!(new_diff, 16, "Should return current diff with insufficient data");
}

#[test]
fn test_lwma_weights_recent_more() {
    // Recent blocks should have more influence than old blocks
    // Scenario: first 30 blocks normal, last 30 blocks very fast
    let mut solve_times: Vec<u64> = vec![1000; 30]; // Normal
    solve_times.extend(vec![200; 30]); // Then fast

    let new_diff = lwma_next_difficulty(16, &solve_times, 1000, 60);
    // Should increase because recent blocks are fast (LWMA weights them more)
    assert!(new_diff > 16,
        "LWMA should respond to recent fast blocks: got {}", new_diff);

    // Compare with reversed order (fast first, then normal)
    let mut solve_times_rev: Vec<u64> = vec![200; 30]; // Fast first
    solve_times_rev.extend(vec![1000; 30]); // Then normal

    let new_diff_rev = lwma_next_difficulty(16, &solve_times_rev, 1000, 60);
    // Should increase LESS because the fast blocks are older
    assert!(new_diff > new_diff_rev,
        "Recent-fast should produce higher diff than old-fast: recent={}, old={}", new_diff, new_diff_rev);
}

#[test]
fn test_lwma_resistant_to_timestamp_manipulation() {
    // Single extreme outlier shouldn't drastically change difficulty
    let mut solve_times: Vec<u64> = vec![1000; 59]; // 59 normal blocks
    solve_times.push(1); // 1 suspicious near-instant block

    let new_diff = lwma_next_difficulty(16, &solve_times, 1000, 60);
    // Should barely change — one outlier out of 60 blocks
    assert!(new_diff <= 17,
        "Single outlier shouldn't cause large change: got {}", new_diff);
}

// ============================================================================
// PHASE C-D: Dual-Lane Hybrid Mining
// ============================================================================

/// Simulates dual-lane reward distribution
fn dual_lane_rewards(
    total_reward: u128,
    fast_lane_hashes: &[[u8; 32]],  // BLAKE3 (GPU)
    fair_lane_hashes: &[[u8; 32]],  // Genus-2 VDF (CPU)
    fast_lane_share_bps: u128,       // e.g., 5000 = 50%
) -> (Vec<u128>, Vec<u128>) {
    let bps_divisor: u128 = 10_000;
    let fast_budget = total_reward * fast_lane_share_bps / bps_divisor;
    let fair_budget = total_reward - fast_budget;

    let fast_rewards = if fast_lane_hashes.is_empty() {
        vec![] // Rolls over (not redistributed to fair lane)
    } else {
        calculate_weighted_rewards(fast_budget, fast_lane_hashes)
    };

    let fair_rewards = if fair_lane_hashes.is_empty() {
        vec![]
    } else {
        calculate_weighted_rewards(fair_budget, fair_lane_hashes)
    };

    (fast_rewards, fair_rewards)
}

#[test]
fn test_dual_lane_50_50_split() {
    let fast = vec![mock_hash_with_zeros(16)];
    let fair = vec![mock_hash_with_zeros(16)];
    let (fr, fir) = dual_lane_rewards(1_000_000, &fast, &fair, 5000);
    assert_eq!(fr[0], 500_000, "Fast lane should get 50%");
    assert_eq!(fir[0], 500_000, "Fair lane should get 50%");
}

#[test]
fn test_dual_lane_empty_fast_lane() {
    // No GPU miners this block — fast lane reward rolls over (not given to fair lane)
    let fair = vec![mock_hash_with_zeros(16)];
    let (fr, fir) = dual_lane_rewards(1_000_000, &[], &fair, 5000);
    assert!(fr.is_empty(), "Empty fast lane should have no rewards");
    assert_eq!(fir[0], 500_000, "Fair lane gets its 50%, not the other lane's share");
}

#[test]
fn test_dual_lane_empty_fair_lane() {
    // No CPU miners — fair lane reward rolls over
    let fast = vec![mock_hash_with_zeros(16)];
    let (fr, fir) = dual_lane_rewards(1_000_000, &fast, &[], 5000);
    assert_eq!(fr[0], 500_000, "Fast lane gets its 50%");
    assert!(fir.is_empty(), "Empty fair lane should have no rewards");
}

#[test]
fn test_dual_lane_no_cross_contamination() {
    // GPU miner in fast lane should NOT affect CPU miner reward in fair lane
    let fast = vec![
        mock_hash_with_zeros(16),
        mock_hash_with_zeros(16),
        mock_hash_with_zeros(16),
    ];
    let fair = vec![mock_hash_with_zeros(20)];
    let (fr, fir) = dual_lane_rewards(1_000_000, &fast, &fair, 5000);

    // Fast lane: 3 equal miners split 500,000
    assert_eq!(fr.len(), 3);
    // Fair lane: 1 miner gets full 500,000
    assert_eq!(fir[0], 500_000, "Fair lane miner should get full fair budget");
}

#[test]
fn test_dual_lane_conservation() {
    // Total distributed must equal total reward (when both lanes active)
    let fast = vec![mock_hash_with_zeros(16), mock_hash_with_zeros(18)];
    let fair = vec![mock_hash_with_zeros(20), mock_hash_with_zeros(22)];
    let total = 1_000_000_000u128;
    let (fr, fir) = dual_lane_rewards(total, &fast, &fair, 5000);

    let fast_sum: u128 = fr.iter().sum();
    let fair_sum: u128 = fir.iter().sum();
    assert_eq!(fast_sum + fair_sum, total,
        "Total must be conserved: fast={} + fair={} = {} (expected {})",
        fast_sum, fair_sum, fast_sum + fair_sum, total);
}

#[test]
fn test_dual_lane_difficulty_weighted_within_lanes() {
    // Within each lane, harder solutions should get more
    let fast = vec![mock_hash_with_zeros(16), mock_hash_with_zeros(24)];
    let fair = vec![mock_hash_with_zeros(16), mock_hash_with_zeros(24)];
    let (fr, fir) = dual_lane_rewards(1_000_000_000, &fast, &fair, 5000);

    // In both lanes, the 24-zero miner should dominate
    assert!(fr[1] > fr[0] * 200, "Fast lane: harder solution should get more");
    assert!(fir[1] > fir[0] * 200, "Fair lane: harder solution should get more");
}

// ============================================================================
// INTEGRATION: Full mining pipeline simulation
// ============================================================================

#[test]
fn test_realistic_mining_scenario() {
    // Simulate a real block with:
    // - 30 GPU miners (16-17 zeros, fast lane)
    // - 5 CPU miners (20-24 zeros, fair lane)
    // - 1,000,000,000,000,000 atomic units block reward
    let total_reward = 82_031_250_000_000_000_000_000_u128; // ~0.082 QUG

    let fast_hashes: Vec<[u8; 32]> = (0..30)
        .map(|i| mock_hash_with_zeros(16 + (i % 2) as u32))
        .collect();

    let fair_hashes: Vec<[u8; 32]> = (0..5)
        .map(|i| mock_hash_with_zeros(20 + (i as u32)))
        .collect();

    let (fr, fir) = dual_lane_rewards(total_reward, &fast_hashes, &fair_hashes, 5000);

    // Basic sanity
    assert_eq!(fr.len(), 30);
    assert_eq!(fir.len(), 5);

    let fast_total: u128 = fr.iter().sum();
    let fair_total: u128 = fir.iter().sum();

    // Conservation
    assert_eq!(fast_total + fair_total, total_reward);

    // Each CPU miner should get more than each GPU miner
    let avg_gpu = fast_total / 30;
    let avg_cpu = fair_total / 5;
    assert!(avg_cpu > avg_gpu,
        "Average CPU reward ({}) should exceed average GPU reward ({})",
        avg_cpu, avg_gpu);

    // Print distribution for manual inspection
    println!("=== Realistic Mining Scenario ===");
    println!("Total reward: {} atomic units", total_reward);
    println!("Fast lane (30 GPU): {} total, {} avg per miner", fast_total, avg_gpu);
    println!("Fair lane (5 CPU): {} total, {} avg per miner", fair_total, avg_cpu);
    println!("CPU/GPU reward ratio: {:.1}x", avg_cpu as f64 / avg_gpu as f64);
}

// ============================================================================
// PHASE B.2: LWMA Pure Function (Emission Controller Pattern)
// ============================================================================

use q_mining::difficulty::calculate_difficulty_for_next_block;
use q_mining::difficulty::count_leading_zero_bits;

/// Generate timestamps for N blocks at a given rate (blocks per second)
fn generate_timestamps(start: u64, count: usize, bps: f64) -> Vec<u64> {
    let interval_secs = 1.0 / bps;
    (0..count)
        .map(|i| start + (i as f64 * interval_secs) as u64)
        .collect()
}

#[test]
fn test_pure_fn_before_activation_returns_legacy() {
    // Before activation height: always returns legacy 16 bits
    let timestamps = generate_timestamps(1000000, 200, 3.46);
    let result = calculate_difficulty_for_next_block(16, &timestamps, 999999, 500, 1);
    assert_eq!(result, 16, "Before activation height, should return legacy difficulty");
}

#[test]
fn test_pure_fn_at_activation_insufficient_data() {
    // At activation height but not enough timestamps: return previous difficulty
    let timestamps = generate_timestamps(1000000, 50, 1.0); // Only 50, need 120
    let result = calculate_difficulty_for_next_block(16, &timestamps, 100, 200, 1);
    assert_eq!(result, 16, "With insufficient data, should return previous difficulty");
}

#[test]
fn test_pure_fn_fast_blocks_increase_difficulty() {
    // Blocks at 3.46 bps (too fast, target is 1 bps) → difficulty should increase
    let timestamps = generate_timestamps(1000000, 130, 3.46);
    let result = calculate_difficulty_for_next_block(16, &timestamps, 100, 200, 1);
    assert!(result > 16,
        "Fast blocks (3.46 bps) should increase difficulty: got {} (expected > 16)", result);
}

#[test]
fn test_pure_fn_slow_blocks_decrease_difficulty() {
    // Blocks at 0.3 bps (too slow, target is 1 bps) → difficulty should decrease
    let timestamps = generate_timestamps(1000000, 130, 0.3);
    let result = calculate_difficulty_for_next_block(20, &timestamps, 100, 200, 1);
    assert!(result < 20,
        "Slow blocks (0.3 bps) should decrease difficulty: got {} (expected < 20)", result);
}

#[test]
fn test_pure_fn_stable_blocks_no_change() {
    // Blocks at exactly 1 bps (on target) → difficulty should stay roughly the same
    let timestamps = generate_timestamps(1000000, 130, 1.0);
    let result = calculate_difficulty_for_next_block(16, &timestamps, 100, 200, 1);
    assert_eq!(result, 16,
        "At target rate, difficulty should remain stable: got {} (expected 16)", result);
}

#[test]
fn test_pure_fn_deterministic() {
    // Same inputs → same output (consensus safety)
    let timestamps = generate_timestamps(1000000, 130, 2.5);
    let r1 = calculate_difficulty_for_next_block(16, &timestamps, 100, 200, 1);
    let r2 = calculate_difficulty_for_next_block(16, &timestamps, 100, 200, 1);
    let r3 = calculate_difficulty_for_next_block(16, &timestamps, 100, 200, 1);
    assert_eq!(r1, r2, "Must be deterministic: {} != {}", r1, r2);
    assert_eq!(r2, r3, "Must be deterministic: {} != {}", r2, r3);
}

#[test]
fn test_pure_fn_floor_never_below_16() {
    // Even with extremely slow blocks, should never go below 16
    let timestamps = generate_timestamps(1000000, 130, 0.01); // Very slow
    let result = calculate_difficulty_for_next_block(16, &timestamps, 100, 200, 1);
    assert!(result >= 16,
        "Difficulty floor: got {} (expected >= 16)", result);
}

#[test]
fn test_pure_fn_clamp_max_2x() {
    // Even with extremely fast blocks, max 2× increase per step
    let timestamps = generate_timestamps(1000000, 130, 100.0); // 100× too fast
    let result = calculate_difficulty_for_next_block(16, &timestamps, 100, 200, 1);
    assert!(result <= 32, // 16 * 2.0 = 32
        "Difficulty should be clamped to max 2×: got {} (expected <= 32)", result);
}

#[test]
fn test_pure_fn_convergence_simulation() {
    // Simulate 500 blocks starting at 3.46 bps
    // LWMA should converge block rate toward 1.0 bps
    let mut difficulty = 16u32;
    let mut timestamps: Vec<u64> = vec![1000000];
    let activation = 0u64;
    let target_bps = 1.0;

    for i in 1..500 {
        // Simulate: higher difficulty → slower blocks
        // Approximate: actual_bps = base_bps / (2^(difficulty - 16))
        let difficulty_factor = 2.0f64.powi((difficulty as i32 - 16).min(10));
        let actual_bps = 3.46 / difficulty_factor;
        let interval = (1.0 / actual_bps).max(0.1);
        let next_ts = timestamps.last().unwrap() + interval as u64;
        timestamps.push(next_ts);

        if timestamps.len() > 120 {
            difficulty = calculate_difficulty_for_next_block(
                difficulty, &timestamps, activation, i as u64, 1,
            );
        }
    }

    // After 500 blocks, difficulty should have increased from 16
    println!("=== Convergence Simulation ===");
    println!("Start: difficulty=16, rate=3.46 bps");
    println!("End: difficulty={}, timestamps={}", difficulty, timestamps.len());

    assert!(difficulty > 16,
        "After 500 blocks at 3.46 bps, difficulty should increase from 16: got {}", difficulty);
}

#[test]
fn test_count_leading_zero_bits() {
    let mut h16 = [0xFF; 32];
    h16[0] = 0x00;
    h16[1] = 0x00;
    assert_eq!(count_leading_zero_bits(&h16), 16);

    let mut h24 = [0xFF; 32];
    h24[0] = 0x00;
    h24[1] = 0x00;
    h24[2] = 0x00;
    assert_eq!(count_leading_zero_bits(&h24), 24);

    let h0 = [0xFF; 32];
    assert_eq!(count_leading_zero_bits(&h0), 0);

    let h256 = [0x00; 32];
    assert_eq!(count_leading_zero_bits(&h256), 256);

    // Partial byte: 0b00001111 = 4 leading zeros in byte
    let mut h20 = [0xFF; 32];
    h20[0] = 0x00;
    h20[1] = 0x00;
    h20[2] = 0x0F; // 4 leading zeros
    assert_eq!(count_leading_zero_bits(&h20), 20);
}
