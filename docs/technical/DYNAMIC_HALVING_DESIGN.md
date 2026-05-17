# Dynamic Halving System Design

## Problem Statement

**Current Issue:** Halving based on fixed block count assumes constant BPS
```rust
const HALVING_INTERVAL: u64 = 3_153_600_000; // Assumes 100 BPS forever
```

**Why This Breaks:**
- ❌ Performance optimizations change actual BPS
- ❌ Network conditions vary block production
- ❌ Cannot adapt to real-world dynamics
- ❌ Hardcoded assumptions become outdated

## Solution Options

### Option 1: Time-Based Halving (Bitcoin Model)

**Use timestamps, not block counts:**

```rust
pub fn calculate_block_reward_time_based(
    genesis_timestamp: u64,
    current_timestamp: u64,
) -> u64 {
    const SECONDS_PER_YEAR: u64 = 31_536_000;
    const BASE_REWARD: u64 = 100_000; // 0.001 QNK

    let elapsed_seconds = current_timestamp - genesis_timestamp;
    let halving_count = elapsed_seconds / SECONDS_PER_YEAR;

    if halving_count >= 64 {
        return 0;
    }

    BASE_REWARD >> halving_count
}
```

**Pros:**
- ✅ Independent of BPS
- ✅ Predictable calendar halvings (every 365 days)
- ✅ Simple to implement
- ✅ Works with any block production rate

**Cons:**
- ⚠️ Miners could game timestamps slightly
- ⚠️ Need timestamp validation

### Option 2: Difficulty-Adjusted Halving

**Adjust based on actual network performance:**

```rust
pub fn calculate_block_reward_difficulty_adjusted(
    block_height: u64,
    avg_block_time_ms: u64,
) -> u64 {
    const TARGET_BLOCK_TIME_MS: u64 = 10; // Target: 100 BPS
    const BLOCKS_PER_ERA: u64 = 3_153_600_000; // Nominal at target speed
    const BASE_REWARD: u64 = 100_000;

    // Adjust halving interval based on actual block time
    let adjusted_interval = if avg_block_time_ms > 0 {
        (BLOCKS_PER_ERA * TARGET_BLOCK_TIME_MS) / avg_block_time_ms
    } else {
        BLOCKS_PER_ERA
    };

    let halving_count = block_height / adjusted_interval;

    if halving_count >= 64 {
        return 0;
    }

    BASE_REWARD >> halving_count
}
```

**Pros:**
- ✅ Adapts to real performance
- ✅ Self-correcting
- ✅ Maintains emission schedule regardless of BPS

**Cons:**
- ⚠️ More complex
- ⚠️ Need accurate difficulty tracking
- ⚠️ Could be manipulated

### Option 3: Hybrid (Recommended)

**Combine time-based epochs with difficulty adjustment:**

```rust
pub struct DynamicRewardCalculator {
    genesis_timestamp: u64,
    target_annual_emission: Vec<u64>, // Decreasing schedule
}

impl DynamicRewardCalculator {
    pub fn calculate_reward(
        &self,
        current_timestamp: u64,
        blocks_in_current_epoch: u64,
    ) -> u64 {
        const SECONDS_PER_YEAR: u64 = 31_536_000;

        // Determine which epoch we're in (time-based)
        let elapsed_seconds = current_timestamp - self.genesis_timestamp;
        let epoch = (elapsed_seconds / SECONDS_PER_YEAR) as usize;

        if epoch >= self.target_annual_emission.len() {
            return 0; // All rewards distributed
        }

        // Calculate reward to hit target emission for this epoch
        let target_emission = self.target_annual_emission[epoch];
        let seconds_in_epoch = elapsed_seconds % SECONDS_PER_YEAR;
        let remaining_emission = if blocks_in_current_epoch > 0 {
            let emission_so_far = (target_emission * seconds_in_epoch) / SECONDS_PER_YEAR;
            target_emission.saturating_sub(emission_so_far)
        } else {
            target_emission
        };

        // Distribute remaining emission across remaining blocks
        let seconds_remaining = SECONDS_PER_YEAR - seconds_in_epoch;
        if seconds_remaining > 0 && blocks_in_current_epoch > 0 {
            let estimated_blocks_remaining =
                (blocks_in_current_epoch * seconds_remaining) / seconds_in_epoch;

            if estimated_blocks_remaining > 0 {
                return remaining_emission / estimated_blocks_remaining;
            }
        }

        // Fallback to standard halving
        self.target_annual_emission[epoch] / 3_153_600_000
    }
}
```

**Pros:**
- ✅ Time-based epochs (predictable calendar)
- ✅ Adapts to actual BPS
- ✅ Hits emission targets regardless of performance
- ✅ Most robust solution

**Cons:**
- ⚠️ Most complex implementation
- ⚠️ Requires tracking epoch statistics

### Option 4: Target-Based Emission (Simplest)

**Set annual emission targets, let rewards float:**

```rust
pub fn calculate_block_reward_target_based(
    genesis_timestamp: u64,
    current_timestamp: u64,
    total_emitted_so_far: u64,
) -> u64 {
    const SECONDS_PER_YEAR: u64 = 31_536_000;

    // Emission targets by year
    const EMISSION_SCHEDULE: [u64; 8] = [
        3_153_600_00_000_000, // Year 1: 3.15M QNK
        1_576_800_00_000_000, // Year 2: 1.58M QNK
        788_400_00_000_000,   // Year 3: 788K QNK
        394_200_00_000_000,   // Year 4: 394K QNK
        197_100_00_000_000,   // Year 5: 197K QNK
        98_550_00_000_000,    // Year 6: 98.5K QNK
        49_275_00_000_000,    // Year 7: 49.3K QNK
        24_637_50_000_000,    // Year 8: 24.6K QNK
    ];

    let elapsed_seconds = current_timestamp - genesis_timestamp;
    let year = (elapsed_seconds / SECONDS_PER_YEAR) as usize;

    if year >= EMISSION_SCHEDULE.len() {
        return 0;
    }

    // Calculate cumulative target up to current year
    let mut target_supply = 0u64;
    for i in 0..=year.min(EMISSION_SCHEDULE.len() - 1) {
        target_supply += EMISSION_SCHEDULE[i];
    }

    // Adjust for partial year
    let seconds_in_year = elapsed_seconds % SECONDS_PER_YEAR;
    let partial_year_target =
        (EMISSION_SCHEDULE[year] * seconds_in_year) / SECONDS_PER_YEAR;
    target_supply = target_supply - EMISSION_SCHEDULE[year] + partial_year_target;

    // Calculate reward needed to hit target
    if total_emitted_so_far < target_supply {
        let remaining = target_supply - total_emitted_so_far;
        // Distribute over next 100 blocks (smoothing)
        remaining / 100
    } else {
        // We're ahead of schedule, reduce rewards temporarily
        0
    }
}
```

**Pros:**
- ✅ Guarantees hitting emission targets
- ✅ Self-correcting (if too fast, reduces rewards)
- ✅ Works with any BPS
- ✅ Simple conceptually

**Cons:**
- ⚠️ Rewards vary block-to-block
- ⚠️ Need accurate total_emitted tracking
- ⚠️ Could confuse miners with variable rewards

## Recommended Solution

**Option 3: Hybrid Time-Based Epochs**

### Implementation Plan

1. **Genesis Timestamp**: Store blockchain start time
2. **Epoch Tracking**: Track blocks produced per calendar year
3. **Difficulty Adjustment**: Adjust rewards to hit annual targets
4. **Fallback**: Use standard halving if tracking fails

### Architecture

```rust
pub struct BlockchainState {
    genesis_timestamp: u64,
    epochs: Vec<EpochStats>,
}

pub struct EpochStats {
    year: u64,
    start_timestamp: u64,
    blocks_produced: u64,
    total_emission: u64,
    target_emission: u64,
}

impl BlockchainState {
    pub fn calculate_dynamic_reward(
        &self,
        current_timestamp: u64,
    ) -> u64 {
        let current_epoch = self.get_current_epoch(current_timestamp);

        // Time-based halving schedule
        let base_target = match current_epoch {
            0 => 3_153_600_00_000_000, // 3.15M QNK
            1 => 1_576_800_00_000_000, // 1.58M QNK
            2 => 788_400_00_000_000,   // 788K QNK
            n => 3_153_600_00_000_000 >> n, // Continue halving
        };

        // Get actual stats for current epoch
        if let Some(stats) = self.epochs.get(current_epoch) {
            let elapsed = current_timestamp - stats.start_timestamp;
            let progress = elapsed as f64 / 31_536_000.0; // Fraction of year

            // Calculate expected emission by now
            let expected_emission = (base_target as f64 * progress) as u64;

            // Adjust reward based on actual vs expected
            if stats.total_emission < expected_emission {
                // Behind schedule, increase reward
                let deficit = expected_emission - stats.total_emission;
                let remaining_time = 31_536_000 - elapsed;
                let estimated_blocks = stats.blocks_produced * remaining_time / elapsed;

                if estimated_blocks > 0 {
                    return deficit / estimated_blocks;
                }
            } else {
                // Ahead of schedule, decrease reward
                let remaining = base_target.saturating_sub(stats.total_emission);
                let remaining_time = 31_536_000 - elapsed;
                let estimated_blocks = stats.blocks_produced * remaining_time / elapsed;

                if estimated_blocks > 0 {
                    return remaining / estimated_blocks;
                }
            }
        }

        // Fallback: standard halving
        100_000 >> current_epoch // 0.001 QNK halved by epoch
    }
}
```

## Difficulty Adjustment Algorithm

**Separate from rewards, but complementary:**

```rust
pub fn calculate_target_difficulty(
    recent_blocks: &[Block],
    target_block_time_ms: u64,
) -> u256 {
    const ADJUSTMENT_WINDOW: usize = 2016; // Like Bitcoin

    if recent_blocks.len() < ADJUSTMENT_WINDOW {
        return current_difficulty(); // Not enough data
    }

    let window = &recent_blocks[recent_blocks.len() - ADJUSTMENT_WINDOW..];
    let time_taken = window.last().unwrap().timestamp
                    - window.first().unwrap().timestamp;

    let expected_time = ADJUSTMENT_WINDOW as u64 * target_block_time_ms;

    // Adjust difficulty to maintain target block time
    let current_diff = window.last().unwrap().difficulty;
    let new_diff = (current_diff * expected_time) / time_taken;

    // Limit adjustment to 4x per period (prevents wild swings)
    new_diff.clamp(current_diff / 4, current_diff * 4)
}
```

## Migration Path

### Phase 1: Add Infrastructure
- [ ] Add genesis_timestamp to blockchain state
- [ ] Create EpochStats tracking
- [ ] Implement dynamic reward calculator
- [ ] Keep old calculator as fallback

### Phase 2: Soft Fork
- [ ] Deploy new code
- [ ] Run both calculators in parallel
- [ ] Compare outputs
- [ ] Verify correctness

### Phase 3: Hard Fork
- [ ] Switch to dynamic calculator
- [ ] Remove old calculator
- [ ] Full production deployment

## Testing Strategy

```rust
#[test]
fn test_dynamic_rewards_various_bps() {
    let genesis = 1700000000;
    let calc = DynamicRewardCalculator::new(genesis);

    // Test at 50 BPS (slow)
    let slow_reward = calc.calculate_reward(
        genesis + 100, // 100 seconds later
        5000, // 5000 blocks in 100 seconds = 50 BPS
    );

    // Test at 100 BPS (target)
    let target_reward = calc.calculate_reward(
        genesis + 100,
        10000, // 10000 blocks = 100 BPS
    );

    // Test at 200 BPS (fast)
    let fast_reward = calc.calculate_reward(
        genesis + 100,
        20000, // 20000 blocks = 200 BPS
    );

    // Slower BPS should give higher per-block rewards
    assert!(slow_reward > target_reward);
    assert!(target_reward > fast_reward);

    // But total emission should be similar over time
    assert_eq!(slow_reward * 5000, target_reward * 10000);
}
```

## Recommendation

**Implement Option 3 (Hybrid)** because:

1. ✅ **Time-based epochs** = Predictable calendar halvings
2. ✅ **Difficulty adjustment** = Adapts to actual BPS
3. ✅ **Emission targets** = Guaranteed distribution schedule
4. ✅ **Future-proof** = Works at any performance level
5. ✅ **Fair** = Can't game the system by changing BPS

This allows you to optimize performance indefinitely without breaking tokenomics!

---

**Next Steps:**
1. Review this design
2. Confirm approach
3. Implement dynamic calculator
4. Add epoch tracking to blockchain state
5. Test with variable BPS scenarios
