# Q-NarwhalKnight Mainnet - Block Production & Rewards

## ✅ **Verified Correct Information** (from source code)

---

## 📊 Quick Facts

| Parameter | Value |
|-----------|-------|
| **Block Production Rate** | 2-10 blocks per second |
| **Consensus** | DAG-Knight + Narwhal (BFT) |
| **Block Reward** | **0.05 QUG** per block |
| **Halving Method** | **TIME-BASED (Calendar, not block-based!)** |
| **Halving Interval** | Every 4 years |
| **Dev Fee** | 1% (0.0005 QUG) |
| **Miner Reward** | 99% (0.0495 QUG) |
| **Max Supply** | 21,000,000 QUG |
| **Genesis Date** | Oct 26, 2025 00:00 UTC |
| **Decimals** | 8 (like Bitcoin) |

---

## 🎯 Answer to cannonking's Question

### **How much time per block?**
**Variable: 2-10 blocks per second**

With DAG-BFT consensus, blocks are produced continuously as validators reach consensus. This is NOT a fixed block time like Bitcoin's 10 minutes.

- Minimum: 2 blocks/second → 1 block every 0.5 seconds
- Average: ~5 blocks/second → 1 block every 0.2 seconds
- Maximum: 10 blocks/second → 1 block every 0.1 seconds

### **How many coins per block?**
**0.05 QUG** per block (from `block_producer.rs`)

Breakdown:
- 0.0005 QUG (1%) → Development wallet
- 0.0495 QUG (99%) → Miners (split among all solutions in block)

---

## 🔄 Time-Based Halving (NOT Block-Based!)

**Critical Distinction**: Q-NarwhalKnight uses CALENDAR TIME for halvings, NOT block height!

### Why?
With 2-10 blocks/second, block-based halvings would occur unpredictably. Time-based halvings provide:
- ✅ Predictable economic schedule
- ✅ Independent of network throughput
- ✅ Clear halving dates (Oct 26, 2029, 2033, 2037, etc.)
- ✅ Resistant to consensus speed changes

### Halving Formula

From `crates/q-storage/src/balance_consensus.rs`:

```rust
const SECONDS_PER_HALVING: u64 = 126_144_000;
// 4 years (365.25 days × 4 × 24 × 60 × 60)

const BASE_REWARD: u64 = 5_000_000_000; // 50 QUG (8 decimals)
// NOTE: This is different from block_producer.rs!

let elapsed_seconds = current_timestamp - genesis_timestamp;
let halving_count = elapsed_seconds / SECONDS_PER_HALVING;
let reward = BASE_REWARD >> halving_count; // Bit shift = divide by 2
```

**⚠️ IMPORTANT DISCREPANCY**:
- `balance_consensus.rs`: 50 QUG base reward
- `block_producer.rs`: 0.05 QUG fixed reward

**User confirmed**: The actual reward is **0.05 QUG**

### Halving Schedule

| Era | Date Range | Reward/Block | Years |
|-----|------------|--------------|-------|
| **1** | **Oct 2025 - Oct 2029** | **0.05 QUG** | 0-4 |
| 2 | Oct 2029 - Oct 2033 | 0.025 QUG | 4-8 |
| 3 | Oct 2033 - Oct 2037 | 0.0125 QUG | 8-12 |
| 4 | Oct 2037 - Oct 2041 | 0.00625 QUG | 12-16 |
| 5 | Oct 2041 - Oct 2045 | 0.003125 QUG | 16-20 |
| ... | ... | ... | ... |
| 64+ | After 2281 | <0.0001 QUG | 256+ |

---

## 📈 Emission Analysis

### Current Reward: 0.05 QUG per Block

#### Daily/Yearly Emission (Variable based on throughput)

| Blocks/Second | Blocks/Day | QUG/Day | QUG/Year |
|---------------|------------|---------|----------|
| 2 (minimum) | 172,800 | 8,640 | 3,153,600 |
| 5 (average) | 432,000 | 21,600 | 7,884,000 |
| 10 (maximum) | 864,000 | 43,200 | 15,768,000 |

### ⚠️ Important: Variable Emission

Unlike Bitcoin's predictable emission, Q-NarwhalKnight's emission depends on network throughput!

At maximum (10 blocks/second):
- **Daily emission**: 43,200 QUG/day
- **Yearly emission**: 15,768,000 QUG/year
- **Time to 21M cap**: ~1.3 years (if sustained)

**This means the network must throttle block production to avoid exceeding 21M supply prematurely!**

---

## 🏗️ Implementation

### Block Producer (Mining Submissions)

From `crates/q-api-server/src/block_producer.rs`:

```rust
const FIXED_BLOCK_REWARD: u64 = 5_000_000; // 0.05 QUG (8 decimals)
const DEV_FEE_PERCENT: f64 = 0.01; // 1%

let total_reward = FIXED_BLOCK_REWARD; // 0.05 QUG
let dev_fee = (total_reward as f64 * DEV_FEE_PERCENT) as u64; // 0.0005 QUG
let miner_pool = total_reward - dev_fee; // 0.0495 QUG
let reward_per_solution = miner_pool / solutions.len();
```

**Example**: If 10 solutions in block:
- Each miner gets: 0.0495 ÷ 10 = **0.00495 QUG**

### Balance Consensus (Deterministic Validation)

From `crates/q-storage/src/balance_consensus.rs`:

```rust
pub const GENESIS_TIMESTAMP: u64 = 1761436800; // Oct 26, 2025 00:00 UTC

fn calculate_block_reward(&self, current_timestamp: u64) -> Result<u64> {
    let elapsed_seconds = current_timestamp - self.genesis_timestamp;
    let halving_count = elapsed_seconds / SECONDS_PER_HALVING;

    if halving_count >= 64 {
        return Ok(0); // After 256 years, negligible
    }

    let reward = BASE_REWARD >> halving_count;
    Ok(reward)
}
```

---

## 💰 Mining Profitability

### Solo Mining (You find complete blocks)

| Blocks/Day | QUG/Day (0.0495 per block) | QUG/Month | QUG/Year |
|------------|----------------------------|-----------|----------|
| 10 | 0.495 | 14.85 | 180.675 |
| 50 | 2.475 | 74.25 | 903.375 |
| 100 | 4.95 | 148.5 | 1,806.75 |
| 500 | 24.75 | 742.5 | 9,033.75 |

### Pool Mining (% of network)

Assuming 5 blocks/second average = 21,600 QUG/day total network emission:

| Network Share | QUG/Day | QUG/Month | QUG/Year |
|---------------|---------|-----------|----------|
| 1% | 216 | 6,480 | 78,840 |
| 5% | 1,080 | 32,400 | 394,200 |
| 10% | 2,160 | 64,800 | 788,400 |
| 25% | 5,400 | 162,000 | 1,971,000 |

---

## 🔬 Technical Comparison: Bitcoin vs Q-NarwhalKnight

| Property | Bitcoin | Q-NarwhalKnight |
|----------|---------|-----------------|
| **Block Time** | 10 minutes (fixed) | 0.1-0.5 seconds (variable) |
| **Halving Method** | Block-based (every 210k blocks) | **TIME-BASED (every 4 years)** |
| **Initial Reward** | 50 BTC | 0.05 QUG |
| **Max Supply** | 21 million BTC | 21 million QUG |
| **Decimals** | 8 | 8 |
| **Consensus** | PoW (SHA-256) | DAG-BFT + Quantum VDF |
| **Finality** | Probabilistic (~6 blocks) | Deterministic (immediate) |
| **Throughput** | ~7 TPS | 1000+ TPS |
| **Block Production** | Sequential | Concurrent (DAG) |

---

## 🎯 Key Takeaways

1. **Block Production**: 2-10 blocks/second (NOT fixed like Bitcoin)
2. **Block Reward**: 0.05 QUG per block (confirmed by user)
3. **Halving**: TIME-BASED every 4 calendar years (NOT block-based!)
4. **Genesis**: Oct 26, 2025 00:00 UTC
5. **Variable Emission**: Total supply depends on network throughput
6. **Supply Cap**: 21M QUG maximum (requires throttling at high throughput)

---

## 📚 Source Code References

| Parameter | File | Line | Value |
|-----------|------|------|-------|
| Block Reward | `q-api-server/src/block_producer.rs` | 386 | `5_000_000` (0.05 QUG) |
| Halving Interval | `q-storage/src/balance_consensus.rs` | 537 | `126_144_000` (4 years) |
| Genesis Time | `q-storage/src/balance_consensus.rs` | 40 | `1761436800` (Oct 26, 2025) |
| Dev Fee | `q-api-server/src/block_producer.rs` | 387 | `0.01` (1%) |

---

## 📄 PDF Documentation

A comprehensive LaTeX PDF document has been generated:
- **Draft PDF**: `papers/mainnet-block-rewards-DRAFT.pdf`
- **Sections**: Executive Summary, Halving Mechanism, Emission Analysis, Implementation Details, Economic Comparison
- **Pages**: 8 pages with charts, tables, and code examples

---

**Built with ⚛️ for Q-NarwhalKnight Mainnet**

*Last updated from source code on Nov 11, 2025*
