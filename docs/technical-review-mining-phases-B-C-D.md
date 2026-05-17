# Technical Review: Mining Fairness Phases B, C, D — LWMA Difficulty + Dual-Lane Hybrid Mining

**Date:** 2026-04-13  
**Network:** Q-NarwhalKnight mainnet-genesis ($1B market cap)  
**Current Status:** Phase A (difficulty-weighted rewards) deployed on Delta canary  
**Prepared for:** DeepSeek + ChatGPT peer review  
**Constraint:** All changes must be height-gated and tested on Delta Docker. Never on Epsilon without 48h canary soak.

---

## Current State (After Phase A)

Phase A changed **how** rewards are split within a block. The mining algorithm, difficulty target, and block rate are unchanged.

| Metric | Before Phase A | After Phase A |
|--------|---------------|---------------|
| Reward distribution | Equal per solution (1/N) | Weighted by leading zero bits (2^zeros) |
| Difficulty target | Fixed 16 bits | Fixed 16 bits (unchanged) |
| Block rate | 3.46 bps (unregulated) | 3.46 bps (unchanged) |
| Miner binary | No change needed | No change needed |
| CPU miner viability | Poor (quantity wins) | Better (quality rewarded) |

**What Phase A does NOT fix:** Block rate is still 3.46× target. Difficulty never adjusts. GPU miners still dominate by volume. CPU miners have no dedicated lane.

---

## Phase B: LWMA Difficulty Adjustment

### B.1 — Algorithm (DONE, committed)

LWMA is implemented in `crates/q-mining/src/difficulty.rs`. 7 tests passing.

```
Window:     60 blocks
Weights:    Linear (1,2,3...60) — recent blocks weighted more
Solvetime:  Clamped to [1ms, 6×target] per block
Adjustment: Clamped to [0.5×, 2.0×] per step
Floor:      16 leading zero bits minimum
Bootstrap:  Needs ≥10 blocks before adjusting
```

### B.2 — Wiring to Challenge Endpoint (THIS REVIEW)

**What needs to change:**

1. **New atomic in AppState** (`crates/q-api-server/src/lib.rs`):
   ```rust
   pub current_difficulty_bits: AtomicU32  // starts at 16
   ```

2. **Periodic difficulty recalculation task** (`crates/q-api-server/src/main.rs`):
   - Every N blocks (N = LWMA window = 60), read recent block timestamps
   - Call `DifficultyAdjuster::calculate_next_difficulty()`
   - Store result in `current_difficulty_bits` atomic
   - Log the adjustment: `"⚙️ [LWMA] Difficulty: 16 → 18 bits (blocks too fast)"`

3. **Challenge endpoint reads dynamic difficulty** (`crates/q-api-server/src/handlers.rs:9474`):
   ```rust
   // BEFORE (hardcoded):
   difficulty_target[0] = 0x00;
   difficulty_target[1] = 0x00;
   
   // AFTER (dynamic):
   let bits = state.current_difficulty_bits.load(Ordering::Relaxed);
   let target = DifficultyTarget::from_leading_zeros(bits);
   difficulty_target = target.target_hash;
   ```

4. **Height-gated activation** (`crates/q-types/src/upgrades.rs`):
   ```rust
   pub const LWMA_DIFFICULTY_ADJUSTMENT: NetworkUpgrade = NetworkUpgrade {
       name: "lwma_difficulty_adjustment",
       activation_height: TBD,  // Announced 1 week before activation
       description: "LWMA difficulty adjustment targeting 1.0 bps",
   };
   ```
   Before activation height: hardcoded 16 bits (current behavior).
   After activation height: LWMA dynamic.

### B.2 — What Miners Experience

- The `difficulty_target` field in `/api/v1/mining/challenge` starts changing between blocks
- Miners already handle this — they fetch a new challenge every 50s and use whatever target the server gives
- No miner binary update needed
- If difficulty increases: fewer solutions meet the target → fewer accepted → block rate drops toward 1.0 bps
- If difficulty decreases: more solutions accepted → block rate increases toward 1.0 bps
- Convergence to target: ~60 blocks (~60 seconds at 1 bps, ~17 seconds at current 3.46 bps)

### B.2 — Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Difficulty spikes too high → no blocks produced | HIGH | Clamp adjustment to max 2× per step; minimum floor of 16 bits |
| Difficulty drops too low → block flood | MEDIUM | Clamp adjustment to min 0.5× per step; emission controller compensates |
| Old miners ignore new difficulty | NONE | Miners use server's challenge target — no client-side difficulty |
| Oscillation (yo-yo effect) | MEDIUM | LWMA's 60-block window dampens this; tested against fast/slow/mixed scenarios |
| Consensus disagreement on difficulty | MEDIUM | All nodes run same LWMA with same inputs (block timestamps from chain) |
| Time warp attack (manipulated timestamps) | LOW | Solvetime clamped to [1ms, 6×target]; outliers have limited impact |

### B.2 — Deployment Sequence

```
Day 1:   Implement B.2, compile, test on Delta Docker
Day 2-3: Deploy to Delta canary, monitor 48 hours
Day 3:   If stable → announce activation height (current + 50,000 blocks)
Day 4:   Deploy to all production nodes via ha-deploy.sh
Day 10:  Activation height reached → LWMA goes live
Day 11:  Monitor block rate converge from 3.46 → ~1.0 bps
```

### B.2 — Testing Requirements

```rust
// Integration test: simulate 200 blocks at 3.46 bps, verify difficulty increases
#[test]
fn test_lwma_reduces_fast_block_rate() {
    let adjuster = DifficultyAdjuster::new(Duration::from_secs(1), 60);
    let fast_times: Vec<Duration> = vec![Duration::from_millis(289); 60]; // 3.46 bps
    let new_diff = adjuster.calculate_next_difficulty(16, &fast_times).unwrap();
    assert!(new_diff > 16, "Difficulty should increase for fast blocks");
}

// Integration test: verify convergence over 500 blocks
#[test]
fn test_lwma_converges_to_target() {
    // Simulate: start at 3.46 bps, apply LWMA every 60 blocks
    // Verify block rate converges to 1.0 bps within 300 blocks
}

// Docker test: 3 nodes, 1 miner, verify all nodes agree on difficulty
```

### B.2 — Open Questions for Reviewers

1. Should the LWMA window be 60 blocks (60s at target rate) or longer (120, 180)?
2. Should the activation be gradual (linear ramp from 16 bits over 1000 blocks) or instant at activation height?
3. The emission controller already compensates for block rate deviation. When LWMA brings block rate to 1.0 bps, the emission correction factor should converge to 1.0. Should we verify this convergence or trust the existing controller?
4. At 3.46 bps with LWMA targeting 1.0 bps, difficulty needs to roughly triple (16 → ~18 bits). Is 2× max adjustment per step sufficient, or will convergence be too slow?

---

## Phase C: Genus-2 VDF Fair Lane for CPU Miners

### What Exists

A full Genus-2 hyperelliptic Jacobian VDF implementation sits in `crates/q-vdf/src/genus2_vdf.rs` (622 lines):

- Genus-2 curve: `y² = x⁵ + a₄x⁴ + a₃x³ + a₂x² + a₁x + a₀`
- Jacobian elements in Mumford form: `D = (u(x), v(x))`
- Doubling via Cantor's algorithm: `D → 2D` — **inherently sequential**
- Three security levels: pq128 (256-bit), pq192 (384-bit), pq256 (512-bit)
- Wesolowski proof for efficient verification

**The server already handles dual-path verification** (`main.rs:15095-15235`):
- PATH A: If `genus2_vdf_output` present → Genus-2 verification
- PATH B: Otherwise → BLAKE3×100 (current behavior)

**The MiningSolution struct already has VDF fields** (all `Option`, all `None` today):
- `vdf_output`, `vdf_proof`, `vdf_checkpoints`, `vdf_iterations_count`

### What Needs to Be Built

1. **Genus-2 VDF computation in the miner** — CPU miners compute `D → 2D → 4D → ... → 2^T·D` and submit the output + Wesolowski proof
2. **Challenge endpoint returns VDF parameters** — curve coefficients, iteration count T, security level
3. **Independent difficulty for the VDF lane** — separate LWMA adjuster targeting the VDF evaluation time
4. **Reward split between lanes** — 50% BLAKE3 (GPU), 50% Genus-2 VDF (CPU)

### Why GPU Miners Cannot Dominate the VDF Lane

Each VDF step `D → 2D` requires:
1. Compute the resultant of two degree-2 polynomials over GF(p) — field multiplications
2. Cantor's reduction algorithm — sequential polynomial operations
3. Step T depends on step T-1 — **no parallelism possible**

A GPU with 10,000 cores evaluates one VDF step at the same speed as one CPU core. The sequential dependency is the fundamental constraint. Clock speed matters; core count does not.

**Expected performance:**
- CPU (3.5 GHz): ~500 VDF doublings/second at pq128 (256-bit field)
- GPU (1.5 GHz core): ~200 VDF doublings/second (slower clock, same sequential ops)
- Target T = 1000 iterations → ~2 seconds on CPU, ~5 seconds on GPU

CPU miners actually have an **advantage** in the VDF lane due to higher single-thread clock speeds.

### Phase C — What Miners Experience

**Existing GPU miners:** Nothing changes. They keep mining BLAKE3 in the "fast lane" and earn 50% of block reward.

**New/returning CPU miners:** Download updated miner binary (auto-updater handles this). The miner detects the VDF lane is active (challenge endpoint includes VDF parameters after activation height) and starts computing Genus-2 VDF proofs. Each VDF proof takes ~2 seconds on a modern CPU. Submit via the same `/api/v1/mining/submit` endpoint with the `vdf_output` and `vdf_proof` fields populated.

**Miner update is backward-compatible:** Old miners that don't populate VDF fields continue mining BLAKE3 only. They miss out on the VDF lane's 50% reward share, but their BLAKE3 mining still works.

### Phase C — Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Genus-2 VDF has a cryptographic weakness | MEDIUM | Use pq128 (conservative); VDF labeled "conjectured" quantum-safe |
| VDF computation too slow on low-end CPUs | LOW | Adjust T downward; minimum is profitable at any T |
| Nobody mines the VDF lane | LOW | Unclaimed 50% accumulates → becomes increasingly attractive |
| VDF verification too slow on server | MEDIUM | Wesolowski proof verification is O(log T) — fast even for large T |
| Miner update fails to auto-deploy | LOW | Old miners keep working on BLAKE3; they just miss VDF rewards |
| 50/50 split is unfair to GPU miners | MEDIUM | Adjustable via governance; 50/50 is the starting point |

### Phase C — Deployment Sequence

```
Week 1:  Implement Genus-2 VDF in CPU miner (crates/q-miner/)
Week 2:  Add VDF parameters to challenge endpoint (height-gated)
Week 2:  Test on Delta Docker — full mining pipeline
Week 3:  Deploy miner update to downloads (auto-updater picks it up)
Week 3:  Announce VDF lane activation height (current + 100,000 blocks)
Week 4:  Deploy server to all nodes via ha-deploy.sh
Week 5+: Activation height → VDF lane goes live
```

### Phase C — Open Questions for Reviewers

1. What should the initial VDF iteration count T be? 500 (1s CPU time), 1000 (2s), 2000 (4s)?
2. Should the 50/50 split be hardcoded or configurable via an env var?
3. If nobody mines the VDF lane for 100 blocks, should the unclaimed reward accumulate or redistribute to BLAKE3 lane?
4. The Genus-2 VDF's quantum resistance is "conjectured" (per DeepSeek review). Should we present the VDF lane as "CPU-fair" rather than "quantum-resistant" to avoid overclaiming?
5. Should the miner support both lanes simultaneously (mine BLAKE3 on GPU while computing VDF on CPU)?

---

## Phase D: Per-Lane LWMA Difficulty Adjustment

### What It Does

After Phase C activates, each lane needs its own difficulty adjustment:

- **BLAKE3 lane:** LWMA adjusts BLAKE3 difficulty to target X blocks/second from GPU miners
- **VDF lane:** LWMA adjusts VDF iteration count T to target Y VDF proofs per block from CPU miners

The two lanes are independent — GPU hashrate changes don't affect VDF difficulty, and vice versa.

### What Needs to Be Built

1. **Two DifficultyAdjuster instances** — one per lane, each with its own window and target
2. **VDF difficulty = iteration count T** — higher T = harder VDF = fewer proofs per block
3. **Block producer tracks per-lane solution counts** — to feed into each LWMA adjuster
4. **Two difficulty fields in challenge response** — `blake3_difficulty_target` + `vdf_iterations`

### Phase D — Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Lanes interact unexpectedly | MEDIUM | Complete independence — no shared state between adjusters |
| One lane has 0 miners → difficulty drops to floor | LOW | Minimum difficulty floor per lane; unclaimed rewards don't leak |
| Difficulty adjustment race between block producers | LOW | Deterministic — all producers use same chain data for LWMA |

### Phase D — Open Questions for Reviewers

1. Should both lanes target the same block rate (e.g., 0.5 bps each → 1.0 bps total), or should they be asymmetric?
2. How quickly should VDF difficulty respond? Shorter LWMA window (30 blocks) for faster response, or same 60-block window as BLAKE3?
3. If a lane's difficulty floor is reached and stays there for 1000+ blocks, should the system auto-disable that lane and redirect its reward share?

---

## Dependency Graph

```
Phase A (DONE) ─── reward weighting
       │
Phase B.2 ──────── wire LWMA to challenge endpoint
       │               (height-gated activation)
       │
Phase C ────────── Genus-2 VDF in miner + dual-lane split
       │               (requires Phase B working)
       │               (miner binary update)
       │               (height-gated activation)
       │
Phase D ────────── per-lane difficulty adjustment
                       (requires Phase C working)
```

Each phase is independently deployable and reversible. Each has its own activation height. Each can be delayed without affecting the others (Phase A is already live; Phase B can go live independently of Phase C).

---

## Safety Invariants (Must Hold Across All Phases)

1. **Total emission per block never changes** — same QUG created regardless of which lane or how rewards are split
2. **Sum of all miner rewards = total_reward - dev_fee** — no rounding leak, no extra coins
3. **Old miners always work** — BLAKE3 mining path never removed, just potentially less profitable
4. **Height-gated activation** — every consensus change has an announced activation height
5. **Rollback possible** — if a phase fails, set activation height to `u64::MAX` (effectively disable)

---

## Timeline Summary

| Phase | What | Risk | Effort | Dependencies |
|-------|------|------|--------|-------------|
| **A** | Difficulty-weighted rewards | LOW | **DONE** | None |
| **B.2** | LWMA wired to challenge endpoint | MEDIUM | 3-5 days | Phase A stable on canary |
| **C** | Genus-2 VDF lane + dual-lane split | MEDIUM | 2-3 weeks | Phase B.2 activated |
| **D** | Per-lane difficulty adjustment | MEDIUM | 1 week | Phase C activated |

**Total to full dual-lane mining: ~4-6 weeks of careful, incremental deployment.**

---

## What We're Explicitly NOT Doing

- **No memory-hard PoW** (Ethash/ProgPoW) — adds complexity, doesn't solve GPU dominance for non-memory-hard algos
- **No ASIC resistance claims** — VDF lane has ~2-5× ASIC advantage ceiling (clock speed, not parallelism), which is acceptable
- **No proof-of-stake hybrid** — out of scope, different security model
- **No changing total emission** — mining fairness is about distribution, not inflation
- **No breaking existing miners** — old binaries always work, just earn less if they don't adopt new features
