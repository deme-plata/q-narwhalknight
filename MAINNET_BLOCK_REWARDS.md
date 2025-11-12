# 🌐 Q-NarwhalKnight Mainnet - Block Time & Rewards

## ⏱️ Block Time

**Target Block Time: 15 seconds** (from block_producer.rs)

- One new block every ~15 seconds
- **~5,760 blocks per day** (24 hours × 60 minutes × 4 blocks/minute)
- **~40,320 blocks per week**
- **~2,102,400 blocks per year**

---

## 💰 Current Block Rewards (Phase 7)

### **FIXED Block Reward: 0.05 QUG per block**

From `crates/q-api-server/src/block_producer.rs`:
```rust
const FIXED_BLOCK_REWARD: u64 = 5_000_000; // 0.05 QUG per BLOCK (8 decimals)
```

### Reward Split:
```
Total Block Reward:    0.05 QUG (100%)
├─ Development Fee:    0.0005 QUG (1%)
└─ Miner Rewards:      0.0495 QUG (99%)
```

**Important**: The 0.0495 QUG miner reward is **split equally among ALL solutions in the block**

Example:
- If 1 solution in block: 0.0495 QUG to that miner
- If 10 solutions in block: 0.00495 QUG per miner (0.0495 ÷ 10)
- If 100 solutions in block: 0.000495 QUG per miner (0.0495 ÷ 100)

---

## 📊 Daily Emission

With **15-second blocks** and **0.05 QUG per block**:

```
Blocks per day:     5,760 blocks
Block reward:       0.05 QUG
Daily emission:     5,760 × 0.05 = 288 QUG/day
```

### Annual Supply:
```
Days per year:      365
Daily emission:     288 QUG
Annual emission:    288 × 365 = 105,120 QUG/year
```

### Time to 21 Million Cap:
```
Max supply:         21,000,000 QUG
Annual emission:    105,120 QUG/year
Time to cap:        21,000,000 ÷ 105,120 ≈ 200 years
```

**Note**: This assumes **constant** 0.05 QUG reward. Halving schedule may reduce this (see below).

---

## 🔄 Halving Schedule

From block_producer.rs comments:
> "Time-based halving handled in balance_consensus (every 4 years)"

### Estimated Halving Schedule:

| Era | Years | Reward/Block | Blocks/Year | QUG/Year | Total QUG |
|-----|-------|--------------|-------------|----------|-----------|
| **1** | 0-4 | **0.05 QUG** | 2,102,400 | 105,120 | 420,480 |
| 2 | 4-8 | 0.025 QUG | 2,102,400 | 52,560 | 210,240 |
| 3 | 8-12 | 0.0125 QUG | 2,102,400 | 26,280 | 105,120 |
| 4 | 12-16 | 0.00625 QUG | 2,102,400 | 13,140 | 52,560 |
| 5+ | 16+ | Decreasing | 2,102,400 | Decreasing | Tail emission |

**Total Supply**: Approaches 21,000,000 QUG asymptotically

---

## 💸 Mining Profitability Calculator

### Assumptions:
- Block time: 15 seconds
- Block reward: 0.05 QUG
- Your solutions vs total solutions in blocks determines your share

### Example Scenarios:

#### Solo Mining (You get ALL solutions in YOUR blocks)
If you mine **X blocks per day** with **only your solutions**:

| Blocks/Day | QUG/Day | QUG/Month | QUG/Year |
|------------|---------|-----------|----------|
| 10 | 0.495 | 14.85 | 180.675 |
| 50 | 2.475 | 74.25 | 903.375 |
| 100 | 4.95 | 148.5 | 1,806.75 |
| 500 | 24.75 | 742.5 | 9,033.75 |

*Each block pays 0.0495 QUG to miners (after 1% dev fee)*

#### Pool Mining (You contribute % of solutions)
If you contribute **X% of total network solutions**:

```
Network daily emission: 288 QUG/day
Your 1% of network:     2.88 QUG/day
Your 5% of network:     14.4 QUG/day
Your 10% of network:    28.8 QUG/day
```

---

## 🎯 How Mining Works (Current Implementation)

### Block Producer Logic:
1. **Collect solutions** from miners (queue_solution())
2. **Every 15 seconds**, produce a block
3. **Drain all pending solutions** (up to max_solutions_per_block)
4. **Split 0.0495 QUG** equally among all solutions in block
5. **Pay 0.0005 QUG** to development wallet

### Key Configuration:
```rust
block_interval_secs: 15           // 15 second blocks
max_solutions_per_block: 100      // Up to 100 solutions per block
min_solutions_per_block: 1        // Minimum 1 solution required
```

### Reward Calculation:
```rust
total_reward = 5_000_000          // 0.05 QUG (fixed)
dev_fee = total_reward * 0.01     // 1% = 0.0005 QUG
miner_pool = total_reward - dev_fee // 0.0495 QUG
reward_per_solution = miner_pool / num_solutions
```

---

## ⚠️ Important Notes

### 1. **No Hybrid CPU+GPU Split in Current Implementation**
The `hybrid_mining.rs` module exists in `q-mining` crate but is **NOT currently used** by the block producer.

Current system:
- All miners submit "solutions" (PoW hashes)
- Solutions are treated equally
- No separate CPU (VDF) vs GPU (SHA-3) distinction
- All solutions share miner reward pool equally

### 2. **Fixed Block Reward (Bitcoin-style)**
```rust
// From block_producer.rs line 386:
const FIXED_BLOCK_REWARD: u64 = 5_000_000; // 0.05 QUG per BLOCK
```

This is a **FIXED** reward per block, preventing hyperinflation from unlimited solutions.

### 3. **Development Fee**
```rust
const DEV_FEE_PERCENT: f64 = 0.01; // 1%
```
Goes to: `efca1e8c1f46e91013b4073898c771bb3d566453537ccf87e834505925e50723`

---

## 📈 Supply Projection

### Year 1:
```
Blocks: 2,102,400
Reward: 0.05 QUG/block
Supply: 105,120 QUG
```

### Year 4 (First Halving):
```
Total supply: ~420,480 QUG
New reward: 0.025 QUG/block
```

### Year 8 (Second Halving):
```
Total supply: ~630,720 QUG
New reward: 0.0125 QUG/block
```

### Long-term (Asymptotic):
```
Max supply: 21,000,000 QUG
Reached in: ~200 years (with halvings)
```

---

## 🚀 Quick Summary for cannonking

### **How much time per block?**
**15 seconds** (4 blocks per minute)

### **How many coins per block?**
**0.05 QUG** total
- 0.0005 QUG (1%) → Development
- 0.0495 QUG (99%) → Miners (split among all solutions in block)

### **Daily emission:**
**288 QUG/day** (5,760 blocks × 0.05 QUG)

### **When does it halve?**
**Every 4 years** (handled in balance_consensus)

### **Max supply:**
**21,000,000 QUG** (like Bitcoin)

---

## 🔧 Future: Hybrid Mining Integration

The hybrid CPU+GPU system in `crates/q-mining/src/hybrid_mining.rs` is designed but **not yet integrated** with the block producer.

To integrate:
1. Modify block producer to accept both VDF proofs (CPU) and PoW solutions (GPU)
2. Split 0.0495 QUG miner reward 50/50:
   - 0.02475 QUG to CPU miner (VDF proof)
   - 0.02475 QUG to GPU miner (PoW solution)
3. Require both components for valid block

---

**Built with ⚛️ for Q-NarwhalKnight Mainnet**

*Last updated from block_producer.rs at block interval: 15s, reward: 0.05 QUG*
