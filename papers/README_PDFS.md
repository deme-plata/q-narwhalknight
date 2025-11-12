# Q-NarwhalKnight Mainnet Documentation

## PDF Documents

### `mainnet-rewards.pdf` (220KB, 9 pages) ✅ **COMPREHENSIVE FINAL VERSION**

**Complete emission control system analysis with correct values:**
- Block Reward: **0.05 QUG** per block
- Block Production: **2-10 blocks/second** (DAG-BFT consensus)
- Halving: **TIME-BASED** (every 4 calendar years, independent of block count)
- Genesis: October 26, 2025 00:00 UTC
- Max Supply: **21,000,000 QUG** (hard cap enforced)
- Target Throughput: **~1.66 blocks/sec** for 256-year emission timeline

**Contents:**
1. Quick Reference (complete specs table)
2. Block Production (variable block time: 0.1-0.5 seconds)
3. Block Rewards (0.05 QUG breakdown with dev fee)
4. Time-Based Halving (calendar-based with full code examples)
5. **Two-Layer Emission Control System** ⭐ NEW
   - Layer 1: Time-based halving (controls schedule)
   - Layer 2: Supply cap enforcement (hard 21M limit)
6. Emission Analysis (with sustainability scenarios)
7. **Why Block-Based Halving Fails with DAG-BFT** ⭐ NEW
8. Mining Profitability (solo & pool calculations)
9. Bitcoin Comparison (detailed economic differences)
10. Key Takeaways (updated with two-layer protection)
11. Source Code References (with verification code)

### `mainnet-block-rewards-DRAFT.pdf` (232KB, 8 pages) ⚠️ **DRAFT**

Earlier version with some formatting issues from sed replacements.
Use `mainnet-rewards.pdf` instead.

## Source Files

- `mainnet-rewards.tex` - Clean LaTeX source (final version)
- `mainnet-block-rewards.tex` - Draft LaTeX with formatting issues

## Key Information Summary

### Block Time
**Variable: 0.1-0.5 seconds** (2-10 blocks per second)

### Block Reward
**0.05 QUG** total per block:
- 1% (0.0005 QUG) → Dev wallet
- 99% (0.0495 QUG) → Miners (split among solutions)

### Halving
**TIME-BASED**: Every 4 calendar years (not block-based!)
- Era 1 (2025-2029): 0.05 QUG/block
- Era 2 (2029-2033): 0.025 QUG/block
- Era 3 (2033-2037): 0.0125 QUG/block
- ...continues every 4 years

### Emission
**Variable** (depends on throughput):
- At 2 blocks/sec: 8,640 QUG/day
- At 5 blocks/sec: 21,600 QUG/day
- At 10 blocks/sec: 43,200 QUG/day

### Maximum Supply
**21,000,000 QUG** (requires production throttling)

---

**Generated:** November 11, 2025
**Source Code:** `q-api-server/src/block_producer.rs`, `q-storage/src/balance_consensus.rs`
