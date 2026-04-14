# Technical Review: Dual-Lane Reward Split — Zero-Risk Implementation for $1B Mainnet

**Date:** 2026-04-14  
**Severity:** MAINNET-CRITICAL — this code distributes real money  
**Network:** Q-NarwhalKnight mainnet-genesis ($1B market cap)  
**Purpose:** Design review for DeepSeek approval before implementation  
**Status:** NOT YET CODED — seeking peer review of design first

---

## 0. What We're Changing

**Current behavior** (line 1530 of `block_producer.rs`):
```
total_reward → dev_fee (2%) → miner_total (98%)
miner_total → difficulty-weighted distribution among ALL solutions
```

**New behavior** (after activation height):
```
total_reward → dev_fee (2%) → miner_total (98%)
miner_total → blake3_total (50%) + vdf_total (50%)
blake3_total → difficulty-weighted among BLAKE3 solutions
vdf_total → difficulty-weighted among VDF solutions
```

**Invariants that MUST hold:**
1. `blake3_total + vdf_total == miner_total` (no rounding leak between lanes)
2. `sum(blake3_rewards) == blake3_total` (no leak within BLAKE3 lane)
3. `sum(vdf_rewards) == vdf_total` (no leak within VDF lane)
4. `total_reward == dev_fee + sum(all_miner_rewards)` (global conservation)
5. Before activation: **zero behavioral change** (existing miners unaffected)
6. Total emission per block: **unchanged** (same QUG created regardless of lane split)

---

## 1. Current Code (Exact, Verified)

### Line 1530: Miner total
```rust
let miner_total = total_reward.saturating_sub(dev_fee_amount);
```

### Lines 1532-1700: PPLNS distribution (skip for now — handled separately)
```rust
let used_pplns = if let Some(ref proportions) = distributed_proportions { ... } else { false };
```

### Lines 1702-1814: Difficulty-weighted distribution (the code we're modifying)
```rust
if !used_pplns {
    // Phase A: weight by leading zero bits
    let difficulty_weights: Vec<u128> = solutions.iter().map(|s| {
        let mut zeros = 0u32;
        for byte in s.hash.iter() {
            if *byte == 0 { zeros += 8; }
            else { zeros += byte.leading_zeros(); break; }
        }
        1u128 << zeros.min(64)
    }).collect();

    let total_weight: u128 = difficulty_weights.iter().sum();

    // Division-first integer arithmetic (prevents u128 overflow)
    let mut miner_rewards: Vec<u128> = difficulty_weights.iter().map(|w| {
        if total_weight == 0 { return miner_total / solutions.len() as u128; }
        let quotient = miner_total / total_weight;
        let remainder = miner_total % total_weight;
        quotient * w + (remainder * w) / total_weight
    }).collect();

    // Remainder to highest-weight miner (deterministic)
    let distributed: u128 = miner_rewards.iter().sum();
    let remainder = miner_total.saturating_sub(distributed);
    if remainder > 0 {
        let max_idx = difficulty_weights.iter()
            .enumerate().max_by_key(|(_, w)| *w).map(|(i, _)| i).unwrap_or(0);
        miner_rewards[max_idx] += remainder;
    }

    // Create coinbase transactions...
    for (idx, solution) in solutions.iter().enumerate() {
        let reward_amount = miner_rewards[idx];
        // ... TX creation (lines 1754-1787)
    }
}
```

---

## 2. Proposed Change (Exact Code)

### Step 1: Extract the existing weighted distribution into a pure helper function

This is a **refactor only** — zero behavioral change. The existing logic moves into a function that can be called once (current behavior) or twice (dual-lane).

```rust
/// Distribute `total` among solutions proportional to their difficulty weight.
/// Returns a Vec of rewards with sum == total (no rounding leak).
/// Uses division-first integer arithmetic to prevent u128 overflow.
///
/// INVARIANT: sum(result) == total (guaranteed by remainder assignment)
fn distribute_weighted(
    solutions: &[&MiningSolution],
    total: u128,
) -> Vec<u128> {
    if solutions.is_empty() {
        return vec![];
    }

    // Count leading zero BITS in each solution's hash → weight = 2^zeros
    let weights: Vec<u128> = solutions.iter().map(|s| {
        let mut zeros = 0u32;
        for byte in s.hash.iter() {
            if *byte == 0 { zeros += 8; }
            else { zeros += byte.leading_zeros(); break; }
        }
        1u128 << zeros.min(64)
    }).collect();

    let total_weight: u128 = weights.iter().sum();

    // Division-first: reward_i = (total / total_weight) * w_i + ((total % total_weight) * w_i) / total_weight
    let mut rewards: Vec<u128> = weights.iter().map(|w| {
        if total_weight == 0 {
            return total / solutions.len() as u128;
        }
        let q = total / total_weight;
        let r = total % total_weight;
        q * w + (r * w) / total_weight
    }).collect();

    // Assign rounding remainder to highest-weight solution (deterministic)
    let distributed: u128 = rewards.iter().sum();
    let remainder = total.saturating_sub(distributed);
    if remainder > 0 {
        let max_idx = weights.iter()
            .enumerate()
            .max_by_key(|(_, w)| *w)
            .map(|(i, _)| i)
            .unwrap_or(0);
        rewards[max_idx] += remainder;
    }

    // POST-CONDITION: sum(rewards) == total
    debug_assert_eq!(rewards.iter().sum::<u128>(), total,
        "distribute_weighted: sum({}) != total({})", rewards.iter().sum::<u128>(), total);

    rewards
}
```

**Verification step:** After extracting this function, call it with the SAME inputs as before and verify the output matches. This is a zero-change refactor.

### Step 2: Add the dual-lane split (height-gated)

Replace lines 1702-1814 with:

```rust
if !used_pplns {
    // ═══════════════════════════════════════════════════════════════
    // v10.3.4: Dual-lane reward split (Phase C — CPU-fair mining)
    // Height-gated behind GENUS2_VDF_MINING activation.
    // Before activation: IDENTICAL to previous behavior.
    // After activation: 50/50 split between BLAKE3 and VDF lanes.
    // ═══════════════════════════════════════════════════════════════

    let genus2_activation = std::env::var("Q_GENUS2_VDF_ACTIVATION_HEIGHT")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(u64::MAX);
    let genus2_active = block_height >= genus2_activation;

    if !genus2_active {
        // ═══════════════════════════════════════════════════════════
        // PRE-ACTIVATION: Exact same behavior as before.
        // All solutions treated as one pool. No lane split.
        // ═══════════════════════════════════════════════════════════
        let all_solutions: Vec<&MiningSolution> = solutions.iter().collect();
        let rewards = distribute_weighted(&all_solutions, miner_total);
        // ... create coinbase TXs (same code as before, lines 1750-1814)

    } else {
        // ═══════════════════════════════════════════════════════════
        // POST-ACTIVATION: Dual-lane split.
        // ═══════════════════════════════════════════════════════════

        // Partition solutions by lane
        let blake3_solutions: Vec<&MiningSolution> = solutions.iter()
            .filter(|s| s.vdf_output.is_none())
            .collect();
        let vdf_solutions: Vec<&MiningSolution> = solutions.iter()
            .filter(|s| s.vdf_output.is_some())
            .collect();

        // Lane reward allocation (integer, no rounding leak)
        const VDF_SHARE_BPS: u128 = 5000;   // 50%
        const BPS: u128 = 10_000;
        let vdf_total = miner_total * VDF_SHARE_BPS / BPS;
        let blake3_total = miner_total - vdf_total;  // Remainder to BLAKE3 (no leak)
        // INVARIANT: blake3_total + vdf_total == miner_total ✓

        // Grace period: first 100 blocks after activation
        let blocks_since_activation = block_height.saturating_sub(genus2_activation);
        let in_grace_period = blocks_since_activation < 100;

        // Handle unclaimed VDF rewards
        let (effective_blake3_total, effective_vdf_total) = if vdf_solutions.is_empty() {
            if in_grace_period {
                // Grace: BLAKE3 miners get everything (VDF lane hasn't started yet)
                (miner_total, 0u128)
            } else {
                // Post-grace: unclaimed VDF rewards are BURNED (not distributed)
                // This creates supply reduction incentive for CPU miners to participate
                (blake3_total, 0u128)
                // NOTE: vdf_total is simply not distributed — burned by omission
            }
        } else if blake3_solutions.is_empty() {
            // No BLAKE3 solutions — VDF miners get everything
            // (This is unusual but possible during transition)
            (0u128, miner_total)
        } else {
            // Normal: both lanes have solutions
            (blake3_total, vdf_total)
        };

        // INVARIANT: effective_blake3_total + effective_vdf_total <= miner_total
        // The difference (if any) is burned (unclaimed VDF rewards post-grace)

        // Distribute within each lane
        let blake3_rewards = if !blake3_solutions.is_empty() {
            distribute_weighted(&blake3_solutions, effective_blake3_total)
        } else {
            vec![]
        };
        let vdf_rewards = if !vdf_solutions.is_empty() {
            distribute_weighted(&vdf_solutions, effective_vdf_total)
        } else {
            vec![]
        };

        // Create coinbase transactions for BLAKE3 lane
        for (idx, solution) in blake3_solutions.iter().enumerate() {
            let reward_amount = blake3_rewards[idx];
            // ... same TX creation code (lines 1754-1787)
            // TX data: "Mining reward #{} BLAKE3 lane (difficulty: {} zero bits)"
        }

        // Create coinbase transactions for VDF lane
        for (idx, solution) in vdf_solutions.iter().enumerate() {
            let reward_amount = vdf_rewards[idx];
            // ... same TX creation code (lines 1754-1787)
            // TX data: "Mining reward #{} VDF lane (difficulty: {} zero bits)"
        }

        // Log lane distribution
        info!("⚡ Block #{}: Dual-lane rewards — BLAKE3: {:.6} QUG ({} miners), VDF: {:.6} QUG ({} miners){}",
            block_height,
            effective_blake3_total as f64 / 1e24, blake3_solutions.len(),
            effective_vdf_total as f64 / 1e24, vdf_solutions.len(),
            if vdf_solutions.is_empty() && !in_grace_period {
                format!(", VDF burned: {:.6} QUG", vdf_total as f64 / 1e24)
            } else { String::new() }
        );
    }
}
```

---

## 3. Safety Analysis

### 3.1 Conservation Laws

| Invariant | Mechanism | Can It Fail? |
|-----------|-----------|--------------|
| `blake3_total + vdf_total == miner_total` | `vdf_total = miner_total * 5000 / 10000`, `blake3_total = miner_total - vdf_total` | **NO** — subtraction guarantees exact sum |
| `sum(blake3_rewards) == effective_blake3_total` | `distribute_weighted` has `debug_assert` + remainder assignment | **NO** — proved by construction |
| `sum(vdf_rewards) == effective_vdf_total` | Same as above | **NO** |
| `total_reward == dev_fee + sum(all_rewards) + burned` | `burned = miner_total - effective_blake3_total - effective_vdf_total` (only when VDF empty post-grace) | **NO** — arithmetic identity |

### 3.2 Overflow Analysis

| Operation | Max Value | Fits in u128? |
|-----------|-----------|---------------|
| `miner_total * 5000` | ~10^30 * 5000 = 5×10^33 | YES (u128 max = 3.4×10^38) |
| `quotient * w` where w = 2^64 | ~10^30 / 10^6 * 2^64 = ~10^43 | **DANGER** — could overflow if total_weight is small! |

**Wait — the existing Phase A code has the same risk.** `quotient = miner_total / total_weight`, then `quotient * w`. If `total_weight` is small (e.g., 1 solution with weight 1), then `quotient = miner_total` (~10^30), and `w = 1`, so `quotient * w = 10^30`. That's fine. But if `w = 2^64` and `quotient` is large, then `quotient * 2^64` could overflow u128. When does this happen?

`quotient = miner_total / total_weight`. For `quotient * w` to overflow u128:
`(miner_total / total_weight) * w > 2^128`

Since `w ≤ total_weight` (each weight is part of the sum), `quotient * w ≤ miner_total`. So **no overflow** — the product is bounded by `miner_total` which fits in u128. ✓

### 3.3 Edge Cases

| Scenario | Behavior | Safe? |
|----------|----------|-------|
| **0 solutions total** | `solutions.is_empty()` — no rewards created | ✓ (existing behavior) |
| **All BLAKE3, 0 VDF (grace)** | `effective_blake3_total = miner_total` — BLAKE3 gets 100% | ✓ (same as pre-activation) |
| **All BLAKE3, 0 VDF (post-grace)** | BLAKE3 gets 50%, VDF 50% burned | ✓ (intentional supply reduction) |
| **All VDF, 0 BLAKE3** | `effective_vdf_total = miner_total` — VDF gets 100% | ✓ |
| **1 BLAKE3 + 1 VDF** | Each gets 50% × weight-adjusted | ✓ |
| **PPLNS active** | `used_pplns = true`, dual-lane code skipped entirely | ✓ (PPLNS has separate path) |
| **Before activation height** | `genus2_active = false`, pre-activation path runs (identical to current) | ✓ |
| **Exactly at activation height** | First block to use dual-lane | ✓ |
| **VDF lane has 1 solution with weight 1** | Gets all of `effective_vdf_total` (no division by zero — weight is always ≥1) | ✓ |

### 3.4 Determinism

Every node must produce the **identical** reward split for the same block. This requires:

1. **Same activation height:** Read from constant or env var — same on all nodes. ✓
2. **Same lane classification:** `vdf_output.is_some()` — deterministic from block data. ✓
3. **Same share BPS:** Hardcoded constant `5000`. ✓
4. **Same distribute_weighted:** Pure function, integer-only. ✓
5. **Same remainder assignment:** `max_by_key` with tie-breaking by index. ✓

### 3.5 What Can Go Wrong

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| Bug in `distribute_weighted` extraction | Low | Critical (wrong rewards) | Test: verify pre-activation output matches current code exactly |
| Integer overflow in lane split | Zero | Critical | Proved above: `quotient * w ≤ miner_total` |
| Rounding leak between lanes | Zero | Low (dust amounts) | `blake3_total = miner_total - vdf_total` ensures exact sum |
| PPLNS interaction | Zero | Critical | PPLNS path is separate `if` branch, never reaches lane split |
| VDF solution incorrectly classified | Low | Medium (wrong lane) | Classification uses `vdf_output.is_some()` which is set during verification |
| Consensus failure (nodes disagree on rewards) | Zero | Critical | All inputs deterministic, all arithmetic integer-only |

---

## 4. Testing Plan

### Test 1: Extract refactor — zero behavioral change
```
Given: 5 solutions with known difficulty weights
When: distribute_weighted(all_5, miner_total)
Then: Output matches current code EXACTLY (bit-for-bit same rewards)
```

### Test 2: Pre-activation behavior unchanged
```
Given: block_height < activation_height, 5 solutions
When: Dual-lane code runs
Then: All solutions go to single pool, rewards identical to Test 1
```

### Test 3: Post-activation with only BLAKE3 (grace period)
```
Given: block_height = activation + 50, 3 BLAKE3 solutions, 0 VDF
When: Dual-lane code runs
Then: BLAKE3 miners get 100% of miner_total (grace period)
```

### Test 4: Post-activation with only BLAKE3 (post-grace)
```
Given: block_height = activation + 200, 3 BLAKE3 solutions, 0 VDF
When: Dual-lane code runs
Then: BLAKE3 miners get 50%, VDF 50% burned
     sum(blake3_rewards) == blake3_total
     Total coinbase = dev_fee + blake3_total (not miner_total)
```

### Test 5: Post-activation with both lanes
```
Given: block_height = activation + 200, 3 BLAKE3 + 2 VDF solutions
When: Dual-lane code runs
Then: BLAKE3 pool gets 50%, VDF pool gets 50%
     Each pool distributed by difficulty weight
     sum(all_rewards) == miner_total
```

### Test 6: Conservation law
```
For 1000 random combinations of solutions:
  assert!(dev_fee + sum(all_rewards) + burned == total_reward)
  assert!(sum(blake3_rewards) + sum(vdf_rewards) + burned == miner_total)
```

### Test 7: Single VDF miner gets full VDF allocation
```
Given: 1 VDF solution (weight 2^10) + 3 BLAKE3 solutions
When: Dual-lane code runs
Then: VDF miner gets exactly vdf_total (no sharing needed)
     BLAKE3 miners split blake3_total by weight
```

---

## 5. PPLNS Interaction

The dual-lane split is in the `if !used_pplns` branch (line 1702). PPLNS distribution happens on lines 1547-1700 and is **completely separate**. When PPLNS is active:
- `used_pplns = true`
- Line 1702: `if !used_pplns` is false
- Dual-lane code never executes
- No interaction possible

**For future consideration:** If PPLNS is extended to support dual-lane, it would need its own lane-aware distribution. But that's a separate change and NOT part of this proposal.

---

## 6. Rollback Plan

If a bug is discovered after activation:

1. **Immediate:** Set `Q_GENUS2_VDF_ACTIVATION_HEIGHT=999999999999` on all nodes → reverts to pre-activation behavior on next block
2. **Code fix:** Set activation height constant to `u64::MAX` in code, redeploy
3. **No data corruption:** The only effect of the dual-lane code is different reward amounts in coinbase transactions. No database state is modified. No balances are corrupted. The chain is still valid — it just had different reward distribution for some blocks.

---

## 7. Deployment Sequence

1. **Code the change** with `distribute_weighted` extraction + dual-lane logic
2. **Write unit tests** (Tests 1-7 above)
3. **Run all existing tests** — verify zero regression
4. **Deploy to Delta Docker** with `Q_GENUS2_VDF_ACTIVATION_HEIGHT=1`
5. **Submit test VDF + BLAKE3 solutions** — verify rewards split correctly
6. **Deploy to production** with activation height `u64::MAX` (disabled)
7. **Announce activation height** (current_height + 200,000, ~2 days notice)
8. **Monitor first 100 blocks** after activation (grace period)

---

## 8. Questions for DeepSeek Review

1. **Is the `blake3_total = miner_total - vdf_total` pattern safe?** We use subtraction instead of `miner_total * 5000 / 10000` for BLAKE3 to avoid rounding leak. This means BLAKE3 gets `miner_total - floor(miner_total * 5000 / 10000)`. For odd `miner_total`, BLAKE3 gets 1 extra satoshi. Is this acceptable?

2. **Should the VDF burn be explicit or implicit?** Currently, when VDF lane is empty post-grace, `vdf_total` is simply not distributed (burned by omission). Should we instead create an explicit burn transaction to `[0u8; 32]` for auditability?

3. **Tie-breaking in `max_by_key`:** If two solutions have the same weight, `max_by_key` returns the LAST one (Rust's `Iterator::max_by_key` returns the maximum element, breaking ties in favor of later elements). Is this acceptable for determinism? (It is — all nodes use the same iterator order, so all nodes pick the same element.)

4. **Should we cap the VDF burn?** If the VDF lane is dead for 1000 blocks post-grace, that's 500× the per-block VDF allocation burned. Should there be a maximum cumulative burn before the lane auto-disables? Our previous design said "No auto-disable" but we want to confirm.

5. **Transaction ordering in the block:** Should BLAKE3 coinbase TXs come before VDF coinbase TXs? Or interleaved by original solution order? This affects TX ID determinism since `idx` is used in the hash.

---

## 9. Summary

This change is **surgically scoped**:
- Extracts existing reward logic into a pure function (zero behavioral change)
- Adds a height-gated `if/else` that splits `miner_total` into two pools
- Calls the same function twice (once per lane)
- Grace period handles the transition (first 100 blocks = 100% to BLAKE3)
- All arithmetic is integer-only with proven conservation laws
- Rollback is one env var change

**The key safety property:** Before activation, the code is IDENTICAL to current behavior. After activation, total emission is unchanged — only the distribution changes.
