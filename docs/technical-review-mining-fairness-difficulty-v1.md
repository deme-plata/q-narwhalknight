# Technical Review: Mining Fairness, Difficulty Adjustment & Dual-Lane Hybrid Mining

**Date:** 2026-04-12  
**Status:** Pre-implementation — DeepSeek peer review requested  
**Network:** $1B mainnet — all changes must be height-gated and tested on Delta Docker first  

---

## 1. Current State (The Problem)

### 1.1 No Difficulty Adjustment — At All

**File:** `crates/q-mining/src/difficulty.rs` (line 29)

```rust
pub fn calculate_next_difficulty(&self, current_difficulty: u32, _recent_block_times: &[Duration]) -> Result<u32> {
    // TODO: Implement actual difficulty adjustment algorithm
    Ok(current_difficulty)  // RETURNS INPUT UNCHANGED
}
```

**File:** `crates/q-api-server/src/handlers.rs` (lines 9474-9476)

```rust
let mut difficulty_target = [0xffu8; 32];
difficulty_target[0] = 0x00;
difficulty_target[1] = 0x00;  // HARDCODED: 16 leading zero bits, forever
```

There is no Dark Gravity Well. No LWMA. No Kimoto. No DigiShield. No EMA. No anything. Difficulty is a constant `0x00 0x00 0xFF...` — 16 leading zero bits — hardcoded since genesis. It never changes regardless of hashrate.

**Consequence:** Block rate is 3.46 bps instead of target 1.0 bps because hashrate grew but difficulty didn't. The emission controller compensates by reducing reward per block, but the block rate itself is unregulated.

### 1.2 Equal Reward Per Solution (No Difficulty Weighting)

**File:** `crates/q-api-server/src/block_producer.rs` (line 1286)

```rust
let miner_reward_per_solution = (total_reward.saturating_sub(dev_fee_amount)) / solutions.len() as u128;
```

If a block has 250 solutions, each miner gets exactly 1/250th of the reward regardless of:
- How hard their solution was (how many leading zeros)
- How much hashrate they contributed
- Whether they're CPU or GPU
- How long they spent finding the solution

A GPU miner submitting an easy solution (barely meets target) gets the same reward as a CPU miner who found a hash with 10 extra leading zeros. There's no incentive to find better solutions.

### 1.3 GPU Miners Displaced CPU Miners

BLAKE3×100 is embarrassingly parallel:
- Each nonce is independent — no sequential dependency between nonces
- 100 BLAKE3 hashes per nonce takes ~300ns — trivial even sequentially
- GPUs try millions of nonces/sec vs thousands for CPU
- No memory-hardness barrier

Result: GPU miners earn 100-1000x more than CPU miners. CPU miners left.

---

## 2. Difficulty Adjustment — What To Implement

### 2.1 Algorithm: LWMA (Linearly Weighted Moving Average)

**Why LWMA over Dark Gravity Well (DGW):**

| Algorithm | Strengths | Weaknesses | Used By |
|-----------|-----------|------------|---------|
| Dark Gravity Well (DGW) | Fast response to hashrate changes | Oscillation on high-variance networks | Dash |
| Kimoto Gravity Well (KGW) | Simple | Slow response, vulnerable to time warp | Early altcoins |
| DigiShield | Battle-tested | Slow response to rapid hashrate drops | Dogecoin, ZCash |
| **LWMA** | **Fast response, oscillation-resistant, simple** | Slightly higher variance than DGW | Monero forks, many DAG chains |
| EMA | Smooth | Slow to respond to flash crashes | Ethereum (pre-PoS) |

**LWMA is the right choice for DAG-Knight because:**
1. DAG chains have inherently variable block times (multiple miners produce concurrent blocks)
2. LWMA weights recent blocks more heavily — responds fast to hashrate changes
3. Simple formula — easy to audit, hard to game
4. Resistant to time warp attacks (bounded timestamp adjustment)
5. Proven on networks with similar characteristics (Monero forks)

### 2.2 LWMA Formula

```
For a window of N blocks (N = 60 recommended for 1 bps target):

  sum_weighted_solvetimes = Σ(i=1 to N) [ i × solvetime_i ]
  sum_weights = N × (N + 1) / 2

  target_solvetime = 1.0 seconds (1 bps)

  next_difficulty = previous_difficulty × 
                    (target_solvetime × sum_weights) / 
                    (sum_weighted_solvetimes)

Clamps:
  - Solvetime per block: max(1, min(solvetime, 6 × target)) — prevents extreme outliers
  - Difficulty change per adjustment: max(0.5x, min(2.0x, adjustment)) — prevents oscillation
  - Minimum difficulty: 16 leading zero bits (current hardcoded value)
```

**Why N=60:** At 1 bps target, that's a 1-minute window. Fast enough to respond to hashrate changes, slow enough to avoid noise.

### 2.3 Implementation Plan

**File to modify:** `crates/q-mining/src/difficulty.rs`

Replace the stub with:

```rust
pub fn calculate_next_difficulty(
    &self,
    current_difficulty: u32,
    recent_block_times: &[Duration],
) -> Result<u32> {
    let n = recent_block_times.len().min(self.adjustment_window as usize);
    if n < 10 { return Ok(current_difficulty); } // Need minimum data

    let target_ms = self.target_block_time.as_millis() as f64;
    
    // LWMA: weight recent blocks more heavily
    let mut sum_weighted = 0.0;
    let sum_weights = (n * (n + 1) / 2) as f64;
    
    for (i, duration) in recent_block_times.iter().take(n).enumerate() {
        let solvetime = duration.as_millis() as f64;
        // Clamp individual solvetimes to prevent manipulation
        let clamped = solvetime.max(1.0).min(target_ms * 6.0);
        sum_weighted += (i as f64 + 1.0) * clamped;
    }
    
    let adjustment = (target_ms * sum_weights) / sum_weighted;
    // Clamp adjustment to prevent oscillation
    let clamped_adjustment = adjustment.max(0.5).min(2.0);
    
    let new_difficulty = (current_difficulty as f64 * clamped_adjustment) as u32;
    Ok(new_difficulty.max(16)) // Never below 16 leading zero bits
}
```

**File to modify:** `crates/q-api-server/src/handlers.rs` (line 9474)

Replace hardcoded difficulty with dynamic:
```rust
let difficulty_target = state.current_difficulty_target.load(Ordering::Relaxed);
```

**New atomic in AppState:** `current_difficulty_target: AtomicU32` — updated every N blocks by the difficulty adjuster.

**Risk: MEDIUM** — This is a consensus change. Block validity now depends on difficulty. Requires height-gated activation.

---

## 3. Difficulty-Weighted Rewards (Phase A — Quick Win)

### 3.1 The Change

**File:** `crates/q-api-server/src/block_producer.rs` (line 1286)

Replace:
```rust
let miner_reward_per_solution = (total_reward - dev_fee) / solutions.len();
```

With:
```rust
// Calculate difficulty achieved by each solution
// Lower hash = harder solution = more weight
let weights: Vec<f64> = solutions.iter().map(|s| {
    // Count leading zero bits in the solution hash
    let leading_zeros = s.hash.iter()
        .take_while(|&&b| b == 0)
        .count() as f64 * 8.0
        + s.hash.iter()
            .find(|&&b| b != 0)
            .map(|b| b.leading_zeros() as f64)
            .unwrap_or(0.0);
    // Weight = 2^(leading_zeros) — exponential scaling
    2.0_f64.powf(leading_zeros)
}).collect();

let total_weight: f64 = weights.iter().sum();

// Distribute reward proportional to difficulty achieved
for (i, solution) in solutions.iter().enumerate() {
    let proportion = weights[i] / total_weight;
    let reward = ((miner_total as f64) * proportion) as u128;
    // ... create coinbase tx for this miner with `reward`
}
```

### 3.2 What This Means For Miners

**Example with current equal split:**

| Miner | Solution Hash | Leading Zeros | Reward (equal) |
|-------|--------------|---------------|----------------|
| GPU-1 | `0000FF...` | 16 bits | 1/3 |
| GPU-2 | `0000AB...` | 16 bits | 1/3 |
| CPU-1 | `000000FF...` | 24 bits | 1/3 |

**With difficulty-weighted:**

| Miner | Leading Zeros | Weight (2^n) | Share | Reward |
|-------|--------------|-------------|-------|--------|
| GPU-1 | 16 bits | 65,536 | 0.4% | 0.4% |
| GPU-2 | 16 bits | 65,536 | 0.4% | 0.4% |
| CPU-1 | 24 bits | 16,777,216 | 99.2% | 99.2% |

A CPU miner who finds a significantly harder solution (more leading zeros) earns proportionally more. This doesn't eliminate GPU advantage (GPUs still submit more solutions), but it rewards quality over quantity.

### 3.3 Risk Assessment

**Risk: LOW**
- Total emission unchanged — same total reward per block
- No consensus change — other nodes don't validate reward distribution within coinbase (they validate total ≤ allowed reward)
- Only affects which miner addresses receive how much
- Backward compatible — old miners see the same challenges, submit the same solutions

**Testing:**
- Property test: sum of all weighted rewards = total miner allocation (no rounding leak)
- Property test: miner with 2x difficulty gets ~2x reward
- Edge case: single solution in block → gets 100% (same as today)
- Edge case: all solutions have identical difficulty → equal split (same as today)

---

## 4. Dual-Lane Hybrid Mining (The Full Solution)

### 4.1 Architecture

```
                    Block Reward (100%)
                         |
              ┌──────────┴──────────┐
              │                     │
         Fast Lane (50%)       Fair Lane (50%)
         BLAKE3 × 100         Genus-2 VDF
         (GPU-friendly)       (Sequential, CPU-fair)
              │                     │
         difficulty-            difficulty-
         weighted               weighted
         within lane            within lane
              │                     │
              └──────────┬──────────┘
                         │
                   Block Producer
                   (merge both lanes)
```

### 4.2 How It Works

**Fast Lane (existing BLAKE3×100):**
- Same algorithm as today
- GPU miners continue as-is
- 50% of block reward allocated to this lane
- Rewards difficulty-weighted within the lane
- Difficulty adjusted by LWMA independently

**Fair Lane (Genus-2 VDF — new):**
- Sequential computation — each VDF step depends on the previous
- GPUs have no advantage over CPUs (sequential = no parallelism)
- 50% of block reward allocated to this lane
- Rewards difficulty-weighted within the lane
- Difficulty adjusted by LWMA independently (different window, different target)

**Block production:**
- Collect solutions from both lanes
- Each lane gets its 50% share
- If one lane has no solutions in a block, its share rolls over to the next block (doesn't transfer to the other lane — prevents gaming)

### 4.3 Why This Works For Everyone

**For GPU miners:** Nothing changes. They keep mining BLAKE3, keep their hashrate investment, keep earning. Their total reward decreases by ~50% per block, but they face no new competition in their lane.

**For CPU miners:** A new lane where they can compete fairly. Genus-2 VDF takes ~2-4 seconds per evaluation regardless of hardware — no GPU speedup possible. CPU miners compete on luck and uptime, not hardware budget.

**For the network:** Two independent proof systems = higher security. An attacker would need to dominate BOTH lanes to control block production. Miner diversity improves (more independent miners = better decentralization).

**For the exchange listing:** Miner count increases (CPU miners return). Community diversity improves.

### 4.4 Implementation Phases

| Phase | Change | Risk | Effort | Prerequisite |
|-------|--------|------|--------|-------------|
| **A** | Difficulty-weighted rewards (within existing single lane) | LOW | 3-5 days | None — can ship now |
| **B** | LWMA difficulty adjustment for BLAKE3 lane | MEDIUM | 5-8 days | Tests for Phase A |
| **C** | Genus-2 VDF in miner binary (CPU fair lane) | MEDIUM | 8-12 days | Phase B |
| **D** | Dual-lane reward split (50/50) in block producer | MEDIUM | 5-8 days | Phase C |
| **E** | Independent difficulty adjustment per lane | MEDIUM | 3-5 days | Phase D |
| **F** | Height-gated activation on mainnet | LOW | 2 days | All above tested |

**Total: 26-40 days for the full hybrid system**

### 4.5 Critical Design Decisions

**Q: What if nobody mines the VDF lane?**
A: The 50% share rolls over. After 10 empty blocks, the VDF lane's accumulated reward becomes attractive enough that someone will mine it.

**Q: Can a GPU miner game the VDF lane?**
A: No. Genus-2 VDF is inherently sequential. A GPU evaluating `D → 2D → 4D → ...` on the Jacobian is no faster than a CPU. The computation is a chain of dependent multiplications — parallelism doesn't help.

**Q: What about FPGAs/ASICs?**
A: Genus-2 Jacobian arithmetic is field multiplication heavy. An ASIC could gain ~2-5x over CPU, but not 100-1000x like BLAKE3. The sequential nature is the key constraint — clock speed matters, not core count.

**Q: What's the 50/50 split based on?**
A: It's a starting point. The split could be adjusted via governance or emission controller. 50/50 is the simplest defensible choice — neither lane subsidises the other.

---

## 5. Testing Strategy

### 5.1 Phase A Tests (Before Implementation)

```rust
#[test]
fn test_difficulty_weighted_rewards_proportional() {
    // Miner A: 16 leading zeros (minimum)
    // Miner B: 24 leading zeros (8 more = 256x harder)
    // Miner B should get ~256x more reward
    let solutions = vec![
        mock_solution(16_leading_zeros),
        mock_solution(24_leading_zeros),
    ];
    let rewards = calculate_weighted_rewards(1_000_000, &solutions);
    assert!(rewards[1] > rewards[0] * 200); // B gets >200x A
    assert_eq!(rewards[0] + rewards[1], 1_000_000); // No leak
}

#[test]
fn test_equal_difficulty_equal_rewards() {
    let solutions = vec![
        mock_solution(16_leading_zeros),
        mock_solution(16_leading_zeros),
        mock_solution(16_leading_zeros),
    ];
    let rewards = calculate_weighted_rewards(1_000_000, &solutions);
    // All equal difficulty → equal rewards
    assert_eq!(rewards[0], rewards[1]);
    assert_eq!(rewards[1], rewards[2]);
}

#[test]
fn test_single_solution_gets_all() {
    let solutions = vec![mock_solution(16_leading_zeros)];
    let rewards = calculate_weighted_rewards(1_000_000, &solutions);
    assert_eq!(rewards[0], 1_000_000);
}

#[test]
fn test_no_rounding_leak() {
    // Property test: for any N solutions with any difficulties,
    // sum(rewards) must equal total (within 1 atomic unit tolerance)
    proptest!(|(n in 1..250usize)| {
        let solutions: Vec<_> = (0..n).map(|_| mock_solution(rand_zeros())).collect();
        let rewards = calculate_weighted_rewards(1_000_000_000, &solutions);
        let total: u128 = rewards.iter().sum();
        assert!(total >= 999_999_999 && total <= 1_000_000_000);
    });
}
```

### 5.2 Docker Integration Test

```bash
# On Delta: Run 3-miner simulation for 50 blocks
docker-compose -f docker-compose-mining-fairness.yml up
# Verify:
# - High-difficulty solutions get proportionally more reward
# - Total emission per block matches expected
# - No consensus divergence between nodes
```

---

## 6. Priority Order

```
Week 1: Write tests for Phase A (difficulty-weighted rewards)
Week 1: Implement Phase A (3-5 days, LOW risk)
Week 2: Test Phase A on Delta Docker
Week 2: Deploy Phase A to Delta canary → production

Week 3: Implement Phase B (LWMA difficulty adjustment)  
Week 3: Test on Delta Docker (multi-miner simulation)

Week 4-5: Implement Phases C-D (Genus-2 in miner + dual-lane split)
Week 5-6: Test full hybrid system on Delta Docker

Week 6: Height-gated activation vote
Week 7+: Mainnet activation at announced height
```

---

## 7. Comparison: Before vs After

| Metric | Current | After Phase A | After Full Hybrid |
|--------|---------|--------------|-------------------|
| Difficulty | Fixed 16 bits forever | Fixed 16 bits | LWMA per lane |
| Block rate | 3.46 bps (unregulated) | 3.46 bps | ~1.0 bps (target) |
| Reward distribution | Equal per solution | Proportional to difficulty | Proportional per lane |
| CPU miner viability | None (GPU dominates) | Slightly better (hard solutions rewarded) | **Fair** (own VDF lane) |
| GPU miner impact | 100% of rewards | Same total, less if easy solutions | 50% of rewards (BLAKE3 lane) |
| Security model | Single PoW | Single PoW, better incentives | **Dual PoW** (harder to 51%) |
| Miner count | Low (GPU only) | Low (but better incentives) | **High** (CPU miners return) |
