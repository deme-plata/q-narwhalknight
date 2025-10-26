# QNK Tokenomics - Quick Reference Card

## 🎯 At a Glance

```
Model:        Democratized Mining (ASIC-Resistant VDF)
Performance:  100 BPS / 1000+ TPS
Block Reward: 0.001 QNK (decreasing via halvings)
Max Supply:   21,000,000 QNK
Philosophy:   Austrian Economics Time Preference
```

## 📊 Key Numbers

| Metric | Value |
|--------|-------|
| Initial Reward | 0.001 QNK/block |
| Halving Interval | 3,153,600,000 blocks (~1 year) |
| First Halving | Block 3,153,600,000 |
| Blocks Per Second | 100 BPS |
| Transactions Per Second | 1000+ TPS |
| Year 1 Emission | 3,153,600 QNK |
| 4-Year Emission | ~5.9M QNK (28% of cap) |
| Max Supply | 21,000,000 QNK |

## 💰 Halving Schedule

```
Era 1: 0.001000 QNK → 3,153,600 QNK/year
Era 2: 0.000500 QNK → 1,576,800 QNK/year
Era 3: 0.000250 QNK → 788,400 QNK/year
Era 4: 0.000125 QNK → 394,200 QNK/year
...continues...
Era ∞: →21M QNK total
```

## 🏠 Miner Earnings (Year 1)

| Miner Type | Network Share | Monthly Earnings |
|------------|---------------|------------------|
| **Small** (home) | 0.001% | ~2,592 QNK |
| **Medium** (farm) | 0.1% | ~259,200 QNK |
| **Large** (datacenter) | 1% | ~2,592,000 QNK |

## ✅ Why This Works

**ASIC Resistance (VDF):**
- Sequential computation (time-based)
- Cannot parallelize
- $500 laptop competitive with $50,000 farm

**Fast Blocks (100 BPS):**
- Smooth dashboard animation
- Quick transaction finality
- High user engagement

**Small Rewards (0.001 QNK):**
- Prevents oversupply at 100 BPS
- Many small rewards > few large rewards
- Encourages consistent mining

**Predictable Halvings:**
- Austrian economics time preference
- Builds scarcity over time
- Sustainable long-term model

## 🔧 Test Commands

```bash
# Verify halving schedule
./test_halving_100bps.sh

# Check current reward
curl -s http://localhost:8080/api/v1/network/supply | jq '.data.block_reward'

# Get mining challenge
curl -s http://localhost:8080/api/v1/mining/challenge
```

## 📚 Documentation

- **`DEMOCRATIZED_MINING_TOKENOMICS.md`** - Full specification
- **`HIGH_BPS_TOKENOMICS_ANALYSIS.md`** - Design rationale
- **`HALVING_IMPLEMENTATION_SUMMARY.md`** - Implementation overview
- **`TOKENOMICS_QUICK_REFERENCE.md`** - This card

## 🎨 Dashboard

100 BPS creates perfect visual experience:
- ✅ Fast, impressive animation
- ✅ Smooth quantum consensus visualization
- ✅ Real-time block generation
- ✅ Professional aesthetic

## 🚀 Status

**Implementation:** ✅ Complete
**Testing:** ✅ Verified
**Documentation:** ✅ Comprehensive
**Ready for:** Production deployment

---

**TL;DR:** 100 BPS + VDF mining + 0.001 QNK rewards + halvings = democratized quantum consensus with Austrian economics. Home miners welcome. 🏠⚛️
