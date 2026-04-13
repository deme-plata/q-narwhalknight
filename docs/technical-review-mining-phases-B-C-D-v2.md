# Technical Review v2: Mining Fairness Phases B–D — Revised After Peer Review

**Date:** 2026-04-13  
**Network:** Q-NarwhalKnight mainnet-genesis ($1B market cap)  
**Revision:** v2 — incorporates DeepSeek + ChatGPT peer review feedback  
**Status:** Phase A deployed on Delta canary, Phase B redesigned per review  

---

## What Changed From v1

The v1 proposal had a critical design flaw identified independently by both reviewers:

> **"Do not make `current_difficulty_bits` a periodically updated `AtomicU32`. Difficulty is consensus-relevant behavior — it should be a pure function of chain-visible data, not a background timer."**

We agreed. The redesign follows the **emission controller pattern** — the most battle-tested economic controller in the codebase (running correctly on $1B mainnet for 50+ days with zero emission errors).

### Key Changes From v1

| Item | v1 (Rejected) | v2 (This Document) |
|------|--------------|---------------------|
| Difficulty derivation | Background task updates AtomicU32 every 60 blocks | **Pure function of chain history, computed on demand** |
| Recompute frequency | Every 60 blocks | **Every block (on demand at template creation)** |
| LWMA window | 60 blocks | **120 blocks** |
| Activation | Optional ramp | **Instant at fixed height** |
| VDF lane framing | "Quantum-resistant" | **"CPU-fair experimental lane"** |
| Reward split | Env var | **Consensus parameter in height-gated upgrade** |
| GPU claim | "GPU same speed as CPU per VDF step" | **"Extra cores don't help; specialized hardware may reduce per-step latency"** |

---

## The Emission Controller Pattern (What We're Copying)

The emission controller in `crates/q-storage/src/emission_controller.rs` has run correctly for 50+ days on a $1B mainnet. Its design principles are:

### 1. Budget-based error correction

Doesn't just look at instantaneous rate. Tracks **cumulative actual vs cumulative target**. If it over-emitted yesterday, today's reward is lower. Self-correcting over time.

```rust
correction_factor = f(actual_cumulative_emission, target_cumulative_at_time)
// Not: f(block_rate_last_60_seconds)
```

### 2. Multiple fallback rate estimators

Three levels, each used when the higher one lacks data:
- Primary: 60-second windowed rate (smooth)
- Secondary: wall-clock cumulative rate (fresh starts)
- Tertiary: block-timestamp global rate (cold start)
- Default: 1.0 bps (conservative)

### 3. Hard caps at every level

- Block rate: `[0.001, 100000]` bps
- Correction factor: `[0.01, 5.0]`
- Reward: `[MIN_REWARD, dynamic_max]`
- Never exceeds remaining supply or era budget

### 4. PID controller with quadratic boost

- Small errors (<10%): proportional correction (gentle)
- Large errors (>10%): quadratic acceleration (aggressive convergence)
- Halves convergence time from ~13h to ~6h

### 5. Pure function of observable state

```rust
calculate_adaptive_reward(timestamp, recent_block_rate, total_supply) → u128
```
Given the same inputs, always produces the same output. No background timer. No mutable state that drifts. Called at block production and block validation — same formula, same result.

**This is the pattern we apply to difficulty adjustment.**

---

## Phase B Redesigned: LWMA as Pure Function

### The Core Function

```rust
/// Calculate difficulty for the next block — pure function of chain history.
/// Same inputs → same output on every node. No stored mutable state.
/// Called at: challenge endpoint, block template creation, block validation.
///
/// Mirrors the emission controller pattern: deterministic, chain-derived,
/// with hard caps and fallback behavior.
fn calculate_difficulty_for_next_block(
    recent_timestamps: &[u64],     // Last 120 block timestamps (unix ms)
    activation_height: u64,         // Height gate
    next_height: u64,               // Block being produced
    target_block_time_ms: u64,      // 1000ms = 1 bps target
) -> u32 {
    // Before activation: legacy fixed difficulty
    if next_height < activation_height {
        return 16; // Current hardcoded value
    }

    let n = recent_timestamps.len();
    
    // Need one full window of post-activation data before adjusting
    // (Reviewer feedback: ≥10 blocks is insufficient; require full window)
    if n < 120 {
        return 16; // Conservative default until enough data
    }

    // LWMA: compute from the last 120 block timestamps
    // Recent blocks weighted more heavily (linear: 1,2,3...120)
    let mut sum_weighted: f64 = 0.0;
    let sum_weights: f64 = (n * (n + 1) / 2) as f64;
    let target_ms = target_block_time_ms as f64;
    let max_solvetime = target_ms * 6.0;

    for i in 1..n {
        let solvetime = (recent_timestamps[i] - recent_timestamps[i-1]) as f64;
        let clamped = solvetime.max(1.0).min(max_solvetime);
        sum_weighted += (i as f64) * clamped;
    }

    if sum_weighted <= 0.0 {
        return 16;
    }

    // adjustment > 1.0 → blocks too slow → decrease difficulty
    // adjustment < 1.0 → blocks too fast → increase difficulty
    let adjustment = (target_ms * sum_weights) / sum_weighted;

    // Clamp: max 2× change from current (prevents oscillation)
    let clamped = adjustment.max(0.5).min(2.0);

    // Current difficulty is derived from the last block in the window
    // (In practice, read from the previous block header)
    let current_difficulty = 16u32; // TODO: read from chain
    let new_difficulty = (current_difficulty as f64 * clamped) as u32;

    // Floor: never below 16 bits
    new_difficulty.max(16)
}
```

### Where It's Called

| Call Site | When | Purpose |
|-----------|------|---------|
| Challenge endpoint (`handlers.rs`) | Miner requests work | Provide current target |
| Block template creation (`block_producer.rs`) | Node produces block | Embed difficulty in block header |
| Block validation (`balance_consensus.rs`) | Node receives block | Verify block meets expected difficulty |

All three call sites use the **same function** with the **same chain inputs**. No disagreement possible.

### How It Mirrors the Emission Controller

| Emission Controller | Difficulty Controller |
|---|---|
| `calculate_adaptive_reward(timestamp, rate, supply)` | `calculate_difficulty_for_next_block(timestamps, activation, height, target)` |
| Inputs: chain timestamps + cumulative emission | Inputs: chain timestamps + activation height |
| Output: u128 reward | Output: u32 difficulty bits |
| Correction: PID with quadratic boost | Correction: LWMA with linear weighting |
| Clamp: `[0.01, 5.0]` correction factor | Clamp: `[0.5, 2.0]` adjustment factor |
| Floor: MIN_REWARD | Floor: 16 bits |
| Called at: block production + block validation | Called at: challenge + block production + validation |
| Pure function: same inputs → same output | Pure function: same inputs → same output |

### Activation

**Instant at a fixed height.** No ramp. Before activation height: return 16 (current behavior). At and after activation height: LWMA.

Per reviewer feedback: "A ramp complicates verification, testing, and incident analysis for little gain."

### Required Tests (Per Reviewer Feedback)

```
1. Determinism: same chain history on 3 nodes → same next difficulty
2. Restart: node restart at height h → same next difficulty as before restart
3. Reorg: reorg of depth k → recomputed difficulty matches new canonical tip
4. Activation boundary: h=H-1 (legacy), h=H (LWMA), h=H+1 (LWMA)
5. Convergence: simulate 500 blocks starting at 3.46 bps → verify rate reaches 1.0±0.2 bps
6. Emission interaction: verify emission correction_factor converges to 1.0 after LWMA stabilizes
```

---

## Phase C Revised: CPU-Fair Experimental Lane

### Revised Framing

**v1 said:** "Quantum-resistant VDF lane"  
**v2 says:** "CPU-fair experimental lane using sequential computation"

Per both reviewers: the VDF lane's primary value is **sequential work** (extra cores don't help), not proven quantum safety. The Genus-2 Jacobian DLP has known quantum attack vectors (Regev's attack, IACR 2024/2004).

### Revised Claims

| Claim | v1 | v2 |
|-------|----|----|
| GPU resistance | "GPU same speed as CPU per step" | **"Extra parallel cores do not linearly speed up one VDF instance. Specialized hardware may reduce per-step latency."** |
| CPU advantage | "CPU miners have an advantage" | **"Pending benchmarks on desktop CPU, laptop CPU, consumer GPU, and FPGA"** |
| Quantum safety | "No known quantum speedup" | **"Conjectured. Recent work (IACR 2024/2004) shows improved quantum attacks on genus-2 HECC."** |
| Purpose | "Post-quantum mining" | **"Reserved sequential-work lane intended to improve CPU participation"** |

### Reward Split: Consensus Parameter

Per reviewer feedback: the 50/50 split must be a **consensus parameter encoded in the upgrade activation**, not an env var.

```rust
pub const DUAL_LANE_MINING: NetworkUpgrade = NetworkUpgrade {
    name: "dual_lane_mining",
    activation_height: TBD,
    description: "Dual-lane mining: BLAKE3 (GPU) + Genus-2 VDF (CPU)",
    params: DualLaneParams {
        blake3_reward_share_bps: 5000,  // 50%
        vdf_reward_share_bps: 5000,     // 50%
        initial_vdf_iterations: 1000,   // ~2s on mid-range CPU
    },
};
```

Changeable only via a future height-gated upgrade, not at runtime.

### Unclaimed VDF Rewards

Per reviewer consensus: **accumulate for a bounded interval, then burn.**

```
Blocks 1-100 after activation: unclaimed VDF rewards redistributed to BLAKE3 (grace period)
Blocks 101+: unclaimed VDF rewards burned (permanent supply reduction)
```

This creates strong incentive for CPU miners to participate without permanently subsidizing GPU miners.

### Both Lanes Simultaneously: Yes

Per both reviewers: a single miner should mine both lanes — GPU thread does BLAKE3, CPU thread does VDF. This is the natural deployment and helps adoption.

### Initial VDF Parameters

Per reviewer feedback: benchmark before setting T.

**Plan:**
1. Benchmark Genus-2 VDF doubling on: desktop CPU (i7), laptop CPU (i5), consumer GPU (RTX 3060), server CPU (Xeon)
2. Set T such that mid-range desktop CPU takes ~2 seconds per VDF proof
3. Provide `q-miner calibrate` command that measures local performance
4. Publish benchmark results before announcing activation height

### Standalone VDF Prover

Per reviewer feedback: provide a standalone tool to avoid linking the entire VDF library into the miner's hot path.

```bash
# Miner calls this as subprocess:
q-vdf-prover --challenge <hex> --iterations 1000 --security pq128
# Returns: vdf_output, vdf_proof (hex)
```

---

## Phase D Revised: Per-Lane Difficulty

### VDF Difficulty = Iteration Count T

Per reviewer feedback: "Changing T changes latency in a coarse, user-visible way. Use slower adjustment, wider window, smaller per-step change."

| Parameter | BLAKE3 Lane | VDF Lane |
|-----------|-------------|----------|
| LWMA window | 120 blocks | **240 blocks** (longer — VDF has higher variance) |
| Max adjustment per step | 2.0× | **1.5×** (gentler — T changes are more visible to miners) |
| Target rate | To be determined | To be determined |
| Difficulty unit | Leading zero bits | **VDF iteration count T** |

### Deterministic Derivation

Same pattern as Phase B: per-lane difficulty is a **pure function of the canonical chain**. Each node counts VDF solutions and BLAKE3 solutions in the last N blocks and computes the next target independently. Same chain → same result.

### Lane Target Rates

**Open question for this review:**

The reviewers disagreed:
- DeepSeek: "Asymmetric (0.9/0.1 bps) mirrors real-world mining power distribution"
- ChatGPT: "Start 50/50 reward and roughly 50/50 expected contribution unless you have an explicit economic reason for asymmetry"

**Our current thinking:** Start with 50/50 reward split but **allow the difficulty adjusters to find the natural rate** per lane based on actual participation. If 90% of mining power is GPUs, the BLAKE3 lane will naturally produce 90% of solutions at its equilibrium difficulty. The VDF lane will find its own equilibrium based on CPU participation.

The 50% reward allocation to VDF lane is what creates the CPU incentive — it's the guaranteed "reserved seat" regardless of how many or few CPU miners participate.

**We'd like reviewer feedback on this approach.**

### Auto-Disable Dead Lane

Per reviewer consensus: **No.** Auto-disable adds complexity and could be exploited. Keep the floor, let the lane self-correct. If a lane is dead at floor difficulty for extended periods, address via governance (future height-gated upgrade), not automatic logic.

---

## Safety Invariants (Unchanged From v1)

1. **Total emission per block never changes** — same QUG created regardless of lane or split
2. **Sum of all miner rewards = total_reward - dev_fee** — no rounding leak
3. **Old miners always work** — BLAKE3 path never removed
4. **Height-gated activation** — every change has an announced activation height
5. **Rollback possible** — set activation height to `u64::MAX` to disable

**New invariant (v2):**

6. **Difficulty is a pure function of canonical chain state** — no mutable server state, no background timers, deterministic across all nodes

---

## Revised Timeline

| Phase | What | Status | Next Step |
|-------|------|--------|-----------|
| **A** | Difficulty-weighted rewards | **Deployed on Delta canary** | Monitor 48h, then production |
| **B** | LWMA as pure function of chain | **Redesigned per review** | Implement, test determinism |
| **C** | CPU-fair VDF lane | **Revised framing + claims** | Benchmark VDF on real hardware |
| **D** | Per-lane difficulty | **Revised parameters** | Only after B+C validated |

---

## Open Questions for This Round of Review

1. **Lane target rates:** Should both lanes target an equal block rate, or let difficulty adjusters find the natural equilibrium while the reward split (50/50) provides the incentive? 

2. **Emission controller interaction:** When LWMA brings block rate from 3.46→1.0 bps, the emission controller's correction factor should converge to 1.0. Should we add a safety check that alerts if both controllers are making large corrections simultaneously (potential feedback loop)?

3. **VDF lane as optional feature:** Should the VDF lane be compile-time gated (like `advanced-crypto`) so nodes can run without VDF support? Or must all nodes support both lanes after activation?

4. **Regev's attack (IACR 2024/2004):** The paper shows improved quantum attacks on genus-2 HECC. Does this affect the VDF's sequential-work property (which is what we care about), or only the discrete log security (which we don't rely on for the mining lane)?

5. **The "jackpot problem":** If VDF rewards accumulate for 100 blocks then burn, a strategic miner could wait until block 99 (maximum accumulated reward) then submit one proof and claim the entire jackpot. Should the accumulation be capped (e.g., max 5× the per-block VDF allocation)?

---

## References

- [Zawy12 LWMA difficulty algorithms](https://github.com/zawy12/difficulty-algorithms/issues/3) — Window size analysis
- [Zawy12 EMA discussion](https://github.com/zawy12/difficulty-algorithms/issues/62) — N=60 vs N=120 tradeoffs  
- [Wesolowski VDF](https://eprint.iacr.org/2018/601.pdf) — Verifiable Delay Functions definition
- [Regev's attack on hyperelliptic cryptosystems](https://eprint.iacr.org/2024/2004.pdf) — Genus-2 quantum security analysis
- Q-NarwhalKnight emission controller: `crates/q-storage/src/emission_controller.rs` — the pattern we're copying
