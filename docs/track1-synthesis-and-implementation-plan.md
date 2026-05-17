# Track 1: Dual-Lane Mining — Synthesis & Implementation Plan

**Date:** 2026-04-07
**Reviewers:** DeepSeek (Cryptography), ChatGPT (Systems Engineering), Gemma 4 (Architecture)
**Status:** All three reviewers approve Option B direction with corrections

---

## Reviewer Consensus Matrix

| Topic | DeepSeek | ChatGPT | Gemma 4 | Resolution |
|-------|----------|---------|---------|------------|
| **Approach** | Option B approved | Option B approved, lane-bucketed | Dual-lane approved | **Option B: Lane-Bucketed Dual-Proof** |
| **Curve** | pq192 (384-bit) mandatory | Agreed | Agreed + memory-hard consideration | **Use pq192 (384-bit prime)** |
| **Difficulty** | Per-proof LWMA | Per-proof LWMA, N=60-120 | Volatility-adjusted LWMA (V-LWMA) | **Per-lane LWMA, window=120, volatility-adjusted** |
| **Reward split** | Not addressed directly | 80/20 start, separate buckets | SCR-based (security contribution ratio) | **80/20 initial, SCR-tuned at H2** |
| **VDF duration** | 2-4s on ref CPU | 2-4s, benchmark GPU first | May be insufficient without memory-hardness | **2-4s initial, increase if GPU >20x faster** |
| **Activation** | Height-gated, 2-week notice | 3-stage: H0/H1/H2 | VDF must anchor into finality | **3-stage activation (H0/H1/H2)** |
| **Verification** | Wesolowski O(log N) is fast | Isolated queues per proof type | Agreed | **Separate verification pools** |
| **Test vectors** | SageMath verification needed | Not addressed | Not addressed | **SageMath test vectors before activation** |

---

## Critical Corrections From Reviews

### 1. DO NOT use pq128 curve (DeepSeek)
The pq128 curve (256-bit prime) only provides ~128-bit classical security due to Gaudry's index calculus on genus-2 Jacobians. **Must use pq192 (384-bit prime) minimum.**

### 2. DO NOT use equal shared rewards (ChatGPT)
Naive "same reward for either proof" causes oscillation. **Lane-bucketed rewards with separate issuance caps per proof type.**

### 3. DO NOT activate without difficulty adjustment (All three)
Prerequisite. The current `difficulty.rs` is a stub. No mining changes until this works.

### 4. DO NOT skip GPU benchmarks (All three)
GPU can run many independent VDF chains in parallel. If GPU achieves >20x advantage via thousands of concurrent chains, increase VDF iterations or add memory-hardness.

### 5. Integrate VDF into finality, not just PoW (Gemma 4)
VDF output should generate a cryptographic timestamp anchor required for block commitment, tying it into DAG-Knight consensus directly.

---

## Implementation Plan

### Prerequisites (Must complete FIRST)

#### P1: Implement Per-Lane LWMA Difficulty Adjustment
**File:** `crates/q-mining/src/difficulty.rs` (currently a stub)
**Risk:** HIGH — consensus change, needs height gate

**Design:**
```
For each proof type p in {blake3, genus2}:
  - Maintain rolling window of last 120 accepted solution timestamps
  - solve_time_i = clamp(t_i - t_{i-1}, target/4, target*4)
  - LWMA = sum(weight_i * solve_time_i) / sum(weight_i)
    where weight_i = i (linear, recent weighted heavier)
  - new_target_p = old_target_p * (LWMA / target_interval_p)
  - Clamp: max ±8% adjustment per update
  - Minimum sample threshold: 20 events before first adjustment
  - Decay: if lane inactive >10 minutes, increase difficulty slowly (prevent trivial exploit)
```

**Key decisions:**
- Control variable: accepted solution inter-arrival time (not block timestamps)
- Window: 120 accepted solutions per proof type
- Damping: ±8% max per adjustment, ±4x total range clamp
- Anti-collapse: minimum difficulty floor per lane (prevents Genus-2 from becoming trivially easy when no miners present)

#### P2: Benchmark Genus-2 VDF on GPU
**Before any code activation, must answer:**
- How fast is a single Genus-2 doubling on CPU (pq192, 384-bit field)?
- How many concurrent VDF chains can a GPU run?
- What is the GPU:CPU throughput ratio for many-chains-in-parallel?
- Target: GPU advantage must be <20x for the system to be meaningful

**Method:**
```bash
# CPU benchmark (single core)
cargo bench --package q-vdf --bench genus2_doubling

# GPU benchmark: simulate many independent chains
# (write a test that spawns N threads each doing sequential doublings)
cargo bench --package q-vdf --bench genus2_parallel_chains
```

#### P3: SageMath Test Vectors for Genus-2
Generate reference test vectors from SageMath for:
- `double_jacobian()` on pq192 curve (10,000 random inputs)
- Full VDF evaluation (1000 doublings from known seed)
- Mumford representation serialization round-trip
- Edge cases: identity element, degree-1 divisors, degenerate inputs

---

### Stage H0: Telemetry-Only Acceptance (No Economic Effect)
**Activation:** `current_height + 20,000` (~2 days at 3 blocks/sec)

**What changes:**
- Server accepts Genus-2 VDF fields in mining submissions
- Server verifies them if present (dual-path already exists in main.rs:15095)
- Verification results logged but DO NOT affect rewards
- Metrics added: submissions/sec by proof type, verify CPU time, valid ratio

**What doesn't change:**
- Only BLAKE3 solutions earn rewards
- Existing miners unaffected
- No difficulty adjustment active yet (still static)

**Purpose:** Validate verification throughput, catch bugs, measure Genus-2 submission patterns

**Code changes:**
1. Add proof-type metrics counters (main.rs shard consumer)
2. Add `proof_type` field to `MiningSubmission` struct
3. Log Genus-2 verification results without reward credit
4. Add `/api/v1/mining/stats` endpoint showing per-lane telemetry

---

### Stage H1: Rewarded Lane Activation
**Activation:** `H0 + 40,000` (~3.5 days after H0, with tuning time)

**What changes:**
- Genus-2 solutions now earn rewards from a SEPARATE budget
- Per-lane difficulty adjustment activates
- Reward split: 80% BLAKE3 budget, 20% Genus-2 budget (of total mining issuance)
- Each lane has independent difficulty target

**Reward budget mechanics:**
```
Per epoch (e.g., 10,000 blocks):
  total_mining_reward = emission_rate * epoch_duration
  blake3_budget = total_mining_reward * 0.80
  genus2_budget = total_mining_reward * 0.20
  
  blake3_reward_per_solution = blake3_budget / accepted_blake3_solutions_in_epoch
  genus2_reward_per_solution = genus2_budget / accepted_genus2_solutions_in_epoch
```

**If a lane has zero participation:**
- Its budget rolls to next epoch (does NOT transfer to other lane)
- Difficulty decays slowly (max 2% per adjustment period)
- Minimum difficulty floor prevents trivial exploit when first miner appears

**Code changes:**
1. Wire `compute_genus2_vdf()` into slint-wallet CPU miner (activate at H1)
2. Add lane-aware reward calculation in `block_producer.rs`
3. Activate per-lane LWMA in `difficulty.rs`
4. Update challenge response to include per-lane difficulty targets
5. Add `proof_type` to block solutions and P2P block propagation

---

### Stage H2: Policy Tuning
**Activation:** `H1 + 100,000` (~9 days after H1)

**Based on observed data:**
- Adjust reward split (could move to 70/30 or 60/40 if CPU participation low)
- Adjust VDF iteration count if GPU advantage measured >20x
- Potentially move to SCR-based split (Gemma 4's suggestion)
- Tighten Genus-2 validation (minimum iteration requirement)

---

## Architecture: Verification Queue Isolation

```
[HTTP Handler] ──> classify by proof_type
     │                    │
     ├─ BLAKE3 ──> blake3_shards[0..7] ──> rayon par_iter ──> block producer
     │             (existing 8-shard pipeline, unchanged)
     │
     └─ Genus-2 ─> genus2_shards[0..3] ──> rayon par_iter ──> block producer
                   (new 4-shard pipeline, bounded concurrency)
                   (per-miner rate limit: max 10 Genus-2/min)
                   (cheap structural validation BEFORE expensive verify)
```

**Queue limits:**
- BLAKE3: existing 50K buffer (unchanged)
- Genus-2: 5K buffer (smaller — fewer submissions expected)
- Per-miner Genus-2 rate limit: 10/minute (VDF takes 2-4s, so honest miner submits ~15-30/min max)

**Admission filter (cheap, before expensive verify):**
1. Check proof field lengths (VDF output, proof, checkpoints)
2. Check canonical encoding (no malleability)
3. Check nonce not recently seen (dedup cache)
4. Only then: expensive Wesolowski verification

---

## Genus-2 VDF Parameter Selection

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Curve | pq192 (`y^2 = x^5 + x^3 - 2x + 1`) | 384-bit prime, ~192-bit quantum security (DeepSeek) |
| Field size | 384-bit prime | Gaudry's algorithm: classical security ~2^192 |
| Iterations | TBD (benchmark first) | Target: 2-4s on Intel i9-13900K single core |
| Proof | Wesolowski (O(log N) verify) | Server verification fast, miner computation sequential |
| Checkpoints | Every 100 iterations | Enables parallel-safe verification from nearest checkpoint |
| Seed | BLAKE3(challenge \|\| nonce) | Domain-separated from BLAKE3-only path |
| Output hash | SHA3-256(vdf_output) | Different hash family from BLAKE3 for domain separation |

---

## Risk Register

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Genus-2 has implementation bug | MEDIUM | HIGH (invalid proofs accepted) | SageMath test vectors, fuzz testing |
| GPU achieves >20x on parallel VDF chains | MEDIUM | MEDIUM (CPU still uncompetitive) | Benchmark first; increase iterations; add memory-hardness later |
| Difficulty oscillation between lanes | LOW | MEDIUM (unstable block times) | Lane-bucketed rewards + LWMA damping |
| Genus-2 verification DoS | LOW | MEDIUM (starves BLAKE3 verification) | Isolated queues, per-miner rate limit, cheap pre-validation |
| Old miners break at activation | LOW | HIGH (network disruption) | BLAKE3 path unchanged; old miners work forever |
| Reward budget exploit (inactive lane) | LOW | LOW (small budget wasted) | Floor difficulty + budget rollover, no cross-lane transfer |

---

## Implementation Order

```
Week 1: Prerequisites
  ├─ P1: LWMA difficulty adjustment (difficulty.rs)
  ├─ P2: Genus-2 CPU/GPU benchmarks (q-vdf bench)
  └─ P3: SageMath test vectors

Week 2: Stage H0
  ├─ Add proof_type classification
  ├─ Add per-lane metrics
  ├─ Activate H0 (telemetry-only)
  └─ Monitor for 2-3 days

Week 3: Stage H1
  ├─ Wire Genus-2 into slint-wallet CPU miner
  ├─ Implement lane-bucketed rewards
  ├─ Implement isolated verification queues
  ├─ Activate H1 (rewarded lanes)
  └─ Monitor for 1 week

Week 4+: Stage H2
  ├─ Analyze participation data
  ├─ Tune reward split
  ├─ Adjust VDF iterations if needed
  └─ Consider SCR-based dynamic split
```

---

## What We Are NOT Doing

- NOT replacing BLAKE3 (GPU miners keep working)
- NOT using equal shared rewards (lane-bucketed instead)
- NOT using pq128 curve (pq192 minimum)
- NOT activating without benchmarks and test vectors
- NOT changing DAG-Knight block ordering
- NOT requiring a hard fork (height-gated activation)
- NOT touching the 8-shard BLAKE3 pipeline (proven at 320K/sec)

---

*"Three reviewers, one direction: lane-bucketed dual-proof mining with separate economics. Implement difficulty adjustment first. Benchmark before committing. Keep the chain alive."*
