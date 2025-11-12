# 🎯 Adaptive Block Reward System - Proposal for Centuries of Emission

## 🔍 Problem Statement

**Current System:**
- Fixed reward: **0.05 QUG per block**
- Assumed throughput: **0.166 blocks/sec** (6-second blocks)
- Actual throughput: **2-10 blocks/sec** (DAG-BFT high-throughput)
- **Result**: 21M cap reached in 1.3-6.7 years, not 256 years!

**Your Vision:**
- Optimize to **10,000+ blocks/second**
- Emission should last **centuries**
- Time-based halving every 4 years

**The Contradiction:**
Time-based halving + fixed reward + variable throughput = unpredictable emission timeline!

---

## ✅ Solution: Adaptive Block Reward

### **Core Concept**

**Reward scales INVERSELY with network throughput!**

```
Reward per block = TARGET_ANNUAL_EMISSION / blocks_produced_this_year
```

### **How It Works**

1. **Time-Based Halving** controls the **target annual emission**
2. **Adaptive Reward** adjusts reward per block based on **actual throughput**
3. **Supply Cap** enforces hard 21M limit as final safety net

---

## 📊 The Mathematics

### **Target Emission Schedule (Time-Based Halving)**

| Era | Years | Target Annual Emission | Total Emitted |
|-----|-------|------------------------|---------------|
| 1 | 2025-2029 | 82,031 QUG/year | 328,125 QUG |
| 2 | 2029-2033 | 41,016 QUG/year | 164,063 QUG |
| 3 | 2033-2037 | 20,508 QUG/year | 82,031 QUG |
| ... | ... | ... | ... |
| 64+ | 2281+ | <1 QUG/year | → 21M QUG |

**Calculation**:
```
Era 1 target: 21M ÷ (2^6) ÷ 4 years = 82,031 QUG/year
(Using geometric series: first era gets 1/64 of total over 4 years)
```

### **Adaptive Reward Formula**

```rust
fn calculate_adaptive_reward(
    current_timestamp: u64,
    genesis_timestamp: u64,
    recent_block_rate: f64, // blocks per second (trailing average)
) -> u64 {
    // Step 1: Determine current era and target annual emission
    let elapsed_seconds = current_timestamp - genesis_timestamp;
    let halving_count = elapsed_seconds / SECONDS_PER_HALVING; // 126,144,000 sec = 4 years

    const BASE_ANNUAL_EMISSION: u64 = 82_031_000_000_000; // 82,031 QUG (8 decimals)
    let target_annual_emission = BASE_ANNUAL_EMISSION >> halving_count; // Halve each era

    // Step 2: Calculate expected blocks this year
    const SECONDS_PER_YEAR: f64 = 31_557_600.0; // 365.25 days
    let expected_blocks_this_year = recent_block_rate * SECONDS_PER_YEAR;

    // Step 3: Calculate reward per block
    let reward_per_block = (target_annual_emission as f64 / expected_blocks_this_year) as u64;

    // Step 4: Enforce minimum reward (prevent division by zero or too-small rewards)
    const MIN_REWARD: u64 = 100; // 0.000001 QUG minimum
    reward_per_block.max(MIN_REWARD)
}
```

### **Example Calculations**

#### **Scenario 1: Current (2-10 blocks/sec)**

```
Era 1 target: 82,031 QUG/year

At 10 blocks/sec:
- Blocks/year: 10 × 31,557,600 = 315,576,000 blocks
- Reward/block: 82,031 ÷ 315,576,000 = 0.00026 QUG/block
- Actual emission: 315,576,000 × 0.00026 = 82,031 QUG ✅

At 2 blocks/sec:
- Blocks/year: 2 × 31,557,600 = 63,115,200 blocks
- Reward/block: 82,031 ÷ 63,115,200 = 0.0013 QUG/block
- Actual emission: 63,115,200 × 0.0013 = 82,031 QUG ✅
```

#### **Scenario 2: Future (10,000 blocks/sec)**

```
Era 1 target: 82,031 QUG/year

At 10,000 blocks/sec:
- Blocks/year: 10,000 × 31,557,600 = 315,576,000,000 blocks
- Reward/block: 82,031 ÷ 315,576,000,000 = 0.00000026 QUG/block
- Actual emission: 315,576,000,000 × 0.00000026 = 82,031 QUG ✅
```

**No matter the throughput, annual emission stays at target!**

---

## 🔧 Implementation Strategy

### **Phase 1: Block Rate Tracking**

```rust
struct BlockRateTracker {
    recent_blocks: VecDeque<(u64, u64)>, // (height, timestamp)
    window_size: usize, // Number of blocks to average (e.g., 1000)
}

impl BlockRateTracker {
    fn calculate_recent_rate(&self) -> f64 {
        if self.recent_blocks.len() < 2 {
            return 0.166; // Default to 6-second blocks
        }

        let (first_height, first_time) = self.recent_blocks.front().unwrap();
        let (last_height, last_time) = self.recent_blocks.back().unwrap();

        let blocks_produced = last_height - first_height;
        let time_elapsed = last_time - first_time;

        blocks_produced as f64 / time_elapsed as f64
    }

    fn add_block(&mut self, height: u64, timestamp: u64) {
        self.recent_blocks.push_back((height, timestamp));
        if self.recent_blocks.len() > self.window_size {
            self.recent_blocks.pop_front();
        }
    }
}
```

### **Phase 2: Adaptive Reward Calculation**

Update `balance_consensus.rs`:

```rust
fn calculate_block_reward(
    &self,
    current_timestamp: u64,
    recent_block_rate: f64, // NEW PARAMETER
) -> Result<u64, BalanceConsensusError> {
    // Time-based halving (existing logic)
    const SECONDS_PER_HALVING: u64 = 126_144_000; // 4 years
    const BASE_ANNUAL_EMISSION: u64 = 82_031_000_000_000; // 82,031 QUG

    let elapsed_seconds = current_timestamp - self.genesis_timestamp;
    let halving_count = elapsed_seconds / SECONDS_PER_HALVING;

    if halving_count >= 64 {
        return Ok(0); // After 256 years
    }

    // Calculate target annual emission for current era
    let target_annual_emission = BASE_ANNUAL_EMISSION >> halving_count;

    // Adaptive reward based on throughput
    const SECONDS_PER_YEAR: f64 = 31_557_600.0;
    let expected_blocks_this_year = recent_block_rate * SECONDS_PER_YEAR;

    let reward_per_block = (target_annual_emission as f64 / expected_blocks_this_year) as u64;

    // Enforce minimum reward
    const MIN_REWARD: u64 = 100; // 0.000001 QUG
    Ok(reward_per_block.max(MIN_REWARD))
}
```

### **Phase 3: Update Block Producer**

Update `block_producer.rs`:

```rust
// Remove FIXED_BLOCK_REWARD constant
// const FIXED_BLOCK_REWARD: u64 = 5_000_000; // DELETE THIS

fn create_coinbase_transactions(
    solutions: &[MiningSolution],
    recent_block_rate: f64, // NEW PARAMETER
    current_timestamp: u64,  // NEW PARAMETER
) -> Vec<Transaction> {
    // Calculate adaptive reward
    let adaptive_reward = calculate_adaptive_reward(
        current_timestamp,
        GENESIS_TIMESTAMP,
        recent_block_rate,
    );

    let total_reward = adaptive_reward; // Now adaptive!
    let dev_fee_amount = (total_reward as f64 * DEV_FEE_PERCENT) as u64;
    let miner_reward_per_solution = (total_reward - dev_fee_amount) / solutions.len() as u64;

    // Rest of logic unchanged...
}
```

---

## 📈 Benefits

### **1. Throughput-Independent Emission**

| Throughput | Reward/Block | Annual Emission | Time to 21M |
|------------|--------------|-----------------|-------------|
| 0.1 blocks/sec | 0.026 QUG | 82,031 QUG | 256 years ✅ |
| 1 blocks/sec | 0.0026 QUG | 82,031 QUG | 256 years ✅ |
| 10 blocks/sec | 0.00026 QUG | 82,031 QUG | 256 years ✅ |
| 100 blocks/sec | 0.000026 QUG | 82,031 QUG | 256 years ✅ |
| 10,000 blocks/sec | 0.00000026 QUG | 82,031 QUG | 256 years ✅ |

**Emission is ALWAYS 82,031 QUG/year regardless of throughput!**

### **2. Predictable Economic Schedule**

- Oct 26, 2029: Emission halves to 41,016 QUG/year
- Oct 26, 2033: Emission halves to 20,508 QUG/year
- Oct 26, 2037: Emission halves to 10,254 QUG/year
- ...
- Oct 26, 2281: Negligible emission, approaching 21M cap

### **3. No Throttling Required**

You can optimize to 10,000+ blocks/second without worrying about hitting the 21M cap too early!

### **4. Fair Miner Distribution**

- More throughput = More competition = Lower reward/block
- Less throughput = Less competition = Higher reward/block
- **Total annual payout is always the same!**

---

## 🎯 Migration Strategy

### **Testnet Phase**

1. **Deploy to testnet-phase9** with adaptive rewards
2. Monitor block rate tracking accuracy
3. Verify emission stays within target range
4. Test with synthetic high-throughput scenarios

### **Mainnet Migration**

1. **Announce transition** with clear timeline
2. **Grace period** for miners to update software
3. **Activate at specific block height** (e.g., height 100,000)
4. **Monitor closely** for first few weeks

### **Backward Compatibility**

```rust
fn calculate_block_reward(&self, current_timestamp: u64, recent_block_rate: f64) -> u64 {
    const ADAPTIVE_ACTIVATION_HEIGHT: u64 = 100_000;

    if self.current_height < ADAPTIVE_ACTIVATION_HEIGHT {
        // Use old fixed reward system
        return FIXED_BLOCK_REWARD;
    } else {
        // Use new adaptive system
        return calculate_adaptive_reward(current_timestamp, recent_block_rate);
    }
}
```

---

## 💡 Alternative: Difficulty Adjustment Instead

**Different Approach**: Instead of adjusting REWARD, adjust DIFFICULTY to control block production rate.

```rust
// Target 1 block every 6 seconds
const TARGET_BLOCK_TIME: f64 = 6.0;

fn adjust_difficulty(current_difficulty: u128, recent_block_rate: f64) -> u128 {
    let actual_block_time = 1.0 / recent_block_rate;
    let adjustment_factor = actual_block_time / TARGET_BLOCK_TIME;

    // Increase difficulty if blocks too fast, decrease if too slow
    let new_difficulty = (current_difficulty as f64 * adjustment_factor) as u128;

    // Limit adjustment to prevent wild swings
    let max_adjustment = 1.1; // Max 10% change per adjustment
    new_difficulty.clamp(
        (current_difficulty as f64 / max_adjustment) as u128,
        (current_difficulty as f64 * max_adjustment) as u128,
    )
}
```

**But this contradicts your vision of 10,000+ blocks/sec!**

So **adaptive reward** is better than difficulty adjustment for your use case.

---

## 🏆 Recommendation

**Implement Adaptive Block Reward System!**

**Why**:
- ✅ Supports your vision of 10,000+ blocks/sec
- ✅ Maintains predictable 256-year emission schedule
- ✅ Works with time-based halving (complements it!)
- ✅ No throttling required
- ✅ Fair distribution based on competition

**Implementation Steps**:
1. Add `BlockRateTracker` to consensus layer
2. Update `calculate_block_reward()` to accept throughput parameter
3. Update `create_coinbase_transactions()` to use adaptive reward
4. Test on testnet with varying throughput
5. Deploy to mainnet with activation height

---

## 📝 Code Locations to Update

| File | Function | Change |
|------|----------|--------|
| `q-storage/src/balance_consensus.rs` | `calculate_block_reward()` | Add throughput parameter, implement adaptive formula |
| `q-api-server/src/block_producer.rs` | `create_coinbase_transactions()` | Replace FIXED_BLOCK_REWARD with adaptive calculation |
| `q-storage/src/lib.rs` | Add `BlockRateTracker` struct | Track recent block production rate |
| `q-types/src/lib.rs` | Add constants | `BASE_ANNUAL_EMISSION`, `ADAPTIVE_ACTIVATION_HEIGHT` |

---

## ✅ Summary

**Your Question**: "what can we do to fix this so it lasts for centuries even though im upgrading and optimizing so we get 10000 blocks pr second"

**The Answer**:

**Implement ADAPTIVE BLOCK REWARD that scales inversely with throughput!**

```
Reward per block = TARGET_ANNUAL_EMISSION / blocks_produced_this_year
```

**Result**:
- 🎯 Emission predictably reaches 21M over 256 years
- 🚀 Supports 10,000+ blocks/second without hitting cap early
- 📅 Time-based halving still controls the schedule (every 4 years)
- ⚖️ Fair miner distribution based on competition
- 🔒 Supply cap enforcement as final safety net

**This gives you BOTH**:
1. Centuries of emission (predictable economics)
2. High throughput optimization (10,000+ blocks/sec)

---

**Generated**: November 11, 2025
**Status**: 🎯 Solution proposed, ready for implementation
