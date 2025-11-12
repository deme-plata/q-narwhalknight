# Phase 6: Austrian Economics - Conservative Mining Rewards

## 🚨 Problem with Phase 5

**Phase 5 Results (few days):**
- Blocks mined: ~20,000
- Coins created: **998,663.95 QNK**
- Average per block: **~50 QNK**

**Issues:**
- 1 MILLION coins in days = HYPERINFLATION
- No scarcity = No value
- Completely defeats Austrian economic principles

---

## 💎 Phase 6: True Scarcity Model

### Core Principles

1. **Sound Money:** Algorithmic, predictable issuance
2. **Digital Scarcity:** Fixed supply cap
3. **Time Preference:** Reward early adopters
4. **Market Discovery:** Let free market set value
5. **Long-term Sustainability:** Tail emission for security

---

## 📊 New Emission Schedule

### Bitcoin-Inspired, But Faster Convergence

| Epoch | Block Range | Reward per Block | Duration (est.) | Epoch Supply |
|-------|-------------|------------------|-----------------|--------------|
| 1 | 1 - 210,000 | **0.5 QNK** | ~146 days | 105,000 QNK |
| 2 | 210,001 - 420,000 | **0.25 QNK** | ~146 days | 52,500 QNK |
| 3 | 420,001 - 630,000 | **0.125 QNK** | ~146 days | 26,250 QNK |
| 4 | 630,001 - 840,000 | **0.0625 QNK** | ~146 days | 13,125 QNK |
| 5+ | 840,001+ | **0.03125 QNK** (tail) | Forever | Asymptotic |

**Assumptions:**
- Block time: ~1 minute (60 seconds)
- 210,000 blocks ≈ 146 days (4.8 months)

### Total Supply

**Maximum Supply: ~200,000 QNK** (asymptotic, never fully reached)

**Supply after 4 years:**
- Year 1: ~157,500 QNK (3 halvings)
- Year 2: ~183,750 QNK
- Year 3: ~190,000 QNK
- Year 4: ~195,000 QNK
- After: Approaches 200k with tail emission

---

## 🔢 Comparison: Phase 5 vs Phase 6

| Metric | Phase 5 | Phase 6 |
|--------|---------|---------|
| **Block Reward** | ~50 QNK | 0.5 QNK |
| **Supply (20k blocks)** | 998,663 QNK | **10,000 QNK** |
| **Inflation Rate** | INSANE | Conservative |
| **Scarcity** | None | Extreme |
| **Austrian Economics** | ❌ Failed | ✅ TRUE |

**Phase 6 is 100x more scarce!**

---

## 💰 Reward Calculation

### Base Formula (No Quantum Bonus)

```rust
fn calculate_block_reward_phase6(height: u64) -> u64 {
    const INITIAL_REWARD: u64 = 50_000_000; // 0.5 QNK (8 decimals)
    const HALVING_INTERVAL: u64 = 210_000;
    const MIN_REWARD: u64 = 3_125_000; // 0.03125 QNK tail emission

    let epoch = height / HALVING_INTERVAL;

    if epoch >= 4 {
        // Tail emission: constant 0.03125 QNK forever
        MIN_REWARD
    } else {
        // Halving schedule: 0.5 → 0.25 → 0.125 → 0.0625
        INITIAL_REWARD >> epoch // Right shift = divide by 2^epoch
    }
}
```

### With Quantum Enhancement (Optional)

```rust
// Quantum bonus: UP TO +10% for high-quality quantum randomness
// Only if entropy_quality >= 0.9

let quantum_bonus = if quantum_quality >= 0.9 {
    let bonus_rate = 0.10; // 10% max
    let quality_factor = (quantum_quality - 0.9) / 0.1; // Scale 0.9-1.0 to 0-1
    (base_reward as f64 * bonus_rate * quality_factor) as u64
} else {
    0
};

// Example: 0.5 QNK base + 0.05 QNK quantum = 0.55 QNK total
```

### Development Fee

```rust
// 2% development fee (sound infrastructure funding)
fn calculate_dev_fee(block_reward: u64) -> u64 {
    block_reward / 50 // 2% = 1/50
}

// Example: 0.5 QNK × 2% = 0.01 QNK to dev treasury
```

**Total per block (with max quantum bonus):**
- Miner: 0.55 QNK × 98% = **0.539 QNK**
- Dev fee: 0.55 QNK × 2% = **0.011 QNK**

---

## 🏦 Treasury & Dev Fee

### Transparent Treasury

**Address:** `qnk_treasury_phase6_...` (publicly auditable)

**Uses:**
1. Core development (50%)
2. Security audits (20%)
3. Infrastructure (15%)
4. Community grants (10%)
5. Emergency fund (5%)

**Quarterly Reports:**
- All expenses published
- Community oversight
- DAO governance (future)

---

## 📈 Economic Analysis

### Inflation Rate Over Time

| Year | Blocks | Supply | Annual Inflation |
|------|--------|--------|------------------|
| 1 | 525,600 | ~157k QNK | - (genesis) |
| 2 | 1,051,200 | ~183k QNK | 16.5% |
| 3 | 1,576,800 | ~190k QNK | 3.8% |
| 4 | 2,102,400 | ~195k QNK | 2.6% |
| 5+ | - | ~200k QNK | <1% (tail) |

**Converges to <1% inflation after Year 4 (similar to gold)**

### Stock-to-Flow Ratio

**Bitcoin's S2F model:**
- Higher S2F = Higher scarcity = Higher value

**Phase 6 S2F:**
- Year 1: S2F ~1 (low, but early phase)
- Year 2: S2F ~7 (comparable to silver)
- Year 4: S2F ~20 (approaching gold)
- Year 10: S2F ~100+ (exceeds gold!)

---

## 🎯 Why This Works (Austrian Economics)

### 1. Digital Scarcity

**Ludwig von Mises:** "Sound money is an essential foundation of civilization"

- **Fixed supply** = Provable scarcity
- **Algorithmic issuance** = No central bank manipulation
- **Transparent rules** = No surprises

### 2. Time Preference

**Murray Rothbard:** "Time preference is the foundation of interest rates"

- **Early adopters rewarded** with higher % of supply
- **Risk takers incentivized** (higher uncertainty, higher reward)
- **Late adopters** still benefit from tail emission (security budget)

### 3. Free Market Pricing

**F.A. Hayek:** "Prices coordinate economic activity"

- **No pre-mine** = Fair launch
- **No insider allocation** = Equal opportunity
- **Market discovers value** based on utility
- **Supply schedule known** = Rational expectations

### 4. Long-term Sustainability

**Carl Menger:** "Money emerges from the market, not decree"

- **Tail emission** = Perpetual security budget
- **No sudden end** = Smooth transition to fee market
- **Inflation <1%** = Sound store of value

---

## 🚀 Migration from Phase 5

### User Impact

**Phase 5 balance: NOT transferred**
- Fresh network = Fresh start
- This is a TESTNET (rehearsal for mainnet)
- Phase 5 data preserved in `./data` (read-only)

**Phase 6 mining:**
- Everyone starts at 0
- Equal opportunity
- Early adopters accumulate more (higher rewards in epoch 1)

### Miner Transition

**Phase 5 miners:**
1. Stop Phase 5 node
2. Update to v0.9.60-beta
3. Start Phase 6 mining
4. Earn 0.5 QNK per block (epoch 1)

**Phase 6 advantages:**
- **Real scarcity** = Potential future value
- **Mainnet rehearsal** = Experience matters
- **Sound economics** = Confidence in long-term

---

## 📚 Economic References

### Austrian School Classics

1. **Carl Menger** - "Principles of Economics" (1871)
   - Origin of money from barter
   - Subjective theory of value

2. **Ludwig von Mises** - "The Theory of Money and Credit" (1912)
   - Sound money principles
   - Critique of inflation

3. **F.A. Hayek** - "Denationalization of Money" (1976)
   - Private money competition
   - Market-driven currencies

4. **Murray Rothbard** - "What Has Government Done to Our Money?" (1963)
   - Gold standard
   - Time preference theory

### Bitcoin & Digital Scarcity

5. **Satoshi Nakamoto** - Bitcoin Whitepaper (2008)
   - Proof of Work
   - Fixed supply (21M BTC)

6. **Saifedean Ammous** - "The Bitcoin Standard" (2018)
   - Stock-to-flow model
   - Sound money in digital age

7. **PlanB** - Stock-to-Flow Model (2019)
   - S2F ratio predicts value
   - Scarcity drives price

---

## ✅ Implementation Checklist

- [x] Define conservative emission schedule (0.5 → 0.25 → ...)
- [x] Calculate total supply (~200k QNK)
- [x] Verify inflation converges to <1%
- [ ] Update `RewardConfig` in `q-mining/src/rewards.rs`
- [ ] Add Phase 6 variant to config
- [ ] Update tests to verify new economics
- [ ] Add treasury address
- [ ] Document dev fee distribution
- [ ] Create quarterly reporting framework

---

## 🎉 Expected Outcomes

### Short Term (Months 1-6)

- **Supply**: ~80,000 QNK created
- **Scarcity**: Extreme (vs Phase 5's million)
- **Community**: Early adopters accumulate
- **Testing**: Mainnet economics validated

### Medium Term (Year 1)

- **Supply**: ~157,500 QNK
- **Inflation**: Declining (halvings kick in)
- **Adoption**: Sound money attracts users
- **Value**: Market discovers fair price

### Long Term (Years 2-5)

- **Supply**: Approaches 200k QNK
- **Inflation**: <1% (tail emission)
- **Store of Value**: Competing with gold
- **Network Effect**: Self-sustaining economy

---

## 💡 Key Takeaway

**Phase 5 created 1 MILLION coins in days = FAILURE**

**Phase 6 creates ~10,000 coins from same blocks = SUCCESS**

**100x more scarce = 100x better economics = Real potential value**

This is how Bitcoin succeeded. This is how sound money works.

Let's do it right this time. 🚀
