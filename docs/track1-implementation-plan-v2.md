# Track 1: Dual-Lane Mining — Implementation Plan v2

**Date:** 2026-04-07
**Version:** 2.0 (incorporates all reviewer corrections)
**Reviewers:** DeepSeek (Crypto), ChatGPT (Systems), Gemma 4 (Architecture), plus final-round corrections
**Status:** APPROVED — ready for implementation

---

## Design Principles

1. **The chain stays live.** Every change is height-gated. Old miners work forever.
2. **Lane-bucketed economics.** Two independent mining lanes, each with own difficulty and reward budget.
3. **Deterministic consensus.** Difficulty is computed from on-chain data only — never from server-side timestamps.
4. **Shadow before activate.** H0 has zero consensus and zero economic effect.
5. **Wider rollout gaps.** At least 1 week between stages.

---

## Final Corrections Applied (v1 -> v2)

| # | v1 Problem | v2 Fix |
|---|------------|--------|
| 1 | H0 ambiguous — "accepts and verifies" could be consensus | H0 is explicitly **shadow-only**: off-chain metrics, zero consensus/reward effect |
| 2 | LWMA on "accepted solution timestamps" — not deterministic across nodes | **Epoch-based retargeting** from on-chain included solutions per lane |
| 3 | `reward = budget / solutions` spikes when few solutions in epoch | **Max payout cap** per solution + **trailing-epoch smoothing** + unspent budget rolls forward with decay |
| 4 | "increase difficulty slowly when inactive" — sign ambiguous | Rewritten: "lane target may relax by at most 2% per retarget, bounded by minimum-difficulty floor" |
| 5 | H0/H1/H2 spacing too aggressive (2-3 days) | H0: 1 week shadow. H1: after benchmarks + wallet release. H2: 3-4 weeks after H1 |
| 6 | "old miners work forever" — needs protocol versioning | `proof_type` defaults to `blake3_legacy`. All new fields optional. Wire format backward-compatible |

---

## Architecture Overview

```
                    ┌─────────────────────────────────┐
                    │      Mining Challenge API        │
                    │  GET /api/v1/mining/challenge    │
                    │                                   │
                    │  Returns:                         │
                    │    blake3_target: [u8; 32]        │
                    │    genus2_target: [u8; 32]        │
                    │    vdf_iterations: u32            │
                    │    epoch_number: u64              │
                    │    lane_rewards: {b3: f64, g2: f64}│
                    └────────────┬──────────────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                                      ▼
    ┌──────────────────┐                   ┌──────────────────┐
    │   GPU Miner      │                   │   CPU Miner      │
    │   (BLAKE3 x100)  │                   │   (Genus-2 VDF)  │
    │                   │                   │                   │
    │  Fast parallel    │                   │  Sequential       │
    │  ~millions/sec    │                   │  ~1 every 2-4s    │
    │  proof_type: 0    │                   │  proof_type: 1    │
    └────────┬─────────┘                   └────────┬─────────┘
             │                                       │
             ▼                                       ▼
    ┌──────────────────────────────────────────────────────┐
    │              POST /api/v1/mining/submit               │
    │                                                       │
    │  Classify by proof_type                               │
    │    ├─ type 0 (BLAKE3) ──> blake3_shards[0..7]        │
    │    └─ type 1 (Genus2) ──> genus2_shards[0..3]        │
    │                                                       │
    │  Per-shard: batch verify via rayon                    │
    │  Pass verified solutions to block producer            │
    └──────────────────────────────────────────────────────┘
                                 │
                                 ▼
    ┌──────────────────────────────────────────────────────┐
    │              Block Producer                           │
    │                                                       │
    │  Collects verified solutions (both lanes)            │
    │  Tags each with proof_type                           │
    │  Includes in block (up to 250 solutions)             │
    │  Reward calculated per-lane from epoch budget         │
    └──────────────────────────────────────────────────────┘
```

---

## Epoch-Based Lane Retargeting (Deterministic)

### Why epoch-based, not continuous LWMA

Continuous LWMA on server-seen acceptance times is **not consensus-deterministic**. Different nodes see different arrival times. Epoch-based retargeting uses only on-chain data (included solutions with block heights/timestamps), which every node can verify identically.

### Epoch Definition

```
EPOCH_LENGTH = 10,000 blocks (~55 minutes at 3 blocks/sec)

At the end of each epoch E:
  For each lane p in {blake3, genus2}:

    included_solutions_p = count of solutions with proof_type=p in epoch E
    epoch_duration = last_block_timestamp - first_block_timestamp (clamped)

    actual_rate_p = included_solutions_p / epoch_duration
    target_rate_p = configured target (e.g., blake3: 2.0/sec, genus2: 0.5/sec)

    ratio = actual_rate_p / target_rate_p
    ratio = clamp(ratio, 0.75, 1.25)      // max ±25% per epoch

    new_target_p = old_target_p * ratio    // larger target = easier
    new_target_p = max(new_target_p, FLOOR_TARGET_p)  // minimum difficulty
```

### Inactivity Rule

When a lane has zero included solutions in an epoch:
- Target may relax by at most **2% per epoch**, bounded by `FLOOR_TARGET_p`
- This prevents an inactive lane from becoming trivially easy
- When a miner appears, difficulty is still meaningful

### Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Epoch length | 10,000 blocks | ~55 min. Long enough for stable measurement |
| Max adjustment | ±25% per epoch | Responsive but not spiky |
| Floor target (BLAKE3) | `[0x00, 0x00, 0xFF, ...]` | Current difficulty (2 leading zero bytes) |
| Floor target (Genus-2) | TBD after benchmark | Must be achievable by 1 CPU miner |
| Inactivity relaxation | 2% per epoch | Slow decay, prevents trivial exploit |
| Target rate (BLAKE3) | ~2.0 solutions/sec | Sustains current block production |
| Target rate (Genus-2) | ~0.5 solutions/sec | Realistic for CPU miners at 2-4s VDF |

---

## Lane-Bucketed Reward System

### Per-Epoch Budget Allocation

```
total_emission = emission_rate * epoch_duration

blake3_budget = total_emission * BLAKE3_SHARE    // initially 0.80
genus2_budget = total_emission * GENUS2_SHARE    // initially 0.20
```

### Per-Solution Reward (Anti-Spike Controls)

```
For lane p in epoch E:
  accepted_p = count of rewarded solutions for lane p in epoch E
  
  if accepted_p == 0:
    // No solutions — budget rolls forward (with 10% decay cap per epoch)
    rollover_p = min(unspent_p, lane_budget_p * 0.5)
    // Rolled-over budget NEVER exceeds 50% of one epoch's lane budget
  
  else:
    raw_reward = lane_budget_p / accepted_p
    max_reward = lane_budget_p * MAX_REWARD_FRACTION  // e.g., 0.10 = max 10% of budget per solution
    reward_per_solution = min(raw_reward, max_reward)
    
    // Unspent budget (from cap) rolls forward with same decay rule
    spent = reward_per_solution * accepted_p
    unspent = lane_budget_p - spent
    rollover_p = min(unspent, lane_budget_p * 0.5)
```

### Anti-Gaming Properties

| Attack | Defense |
|--------|---------|
| Submit 1 solution at epoch start, claim entire budget | `MAX_REWARD_FRACTION` caps at 10% per solution |
| Hold solutions until epoch boundary | Reward is per included solution, not per submission time |
| Flood cheap invalid solutions | Verification rejects invalids before reward counting |
| Mine both lanes to capture both budgets | Each lane has independent difficulty; mining both is valid but expensive |

### Initial Reward Split

| Lane | Share | Rationale |
|------|-------|-----------|
| BLAKE3 | 80% | Existing GPU miners, dominant hashrate |
| Genus-2 | 20% | Bootstrap CPU participation |

Adjustable at H2 based on observed participation. If CPU miners are still underrepresented, increase to 30% or 40%.

---

## Protocol Versioning & Backward Compatibility

### Proof Type Encoding

```rust
#[repr(u8)]
pub enum ProofType {
    Blake3Legacy = 0,   // Default if field absent
    Genus2Jacobian = 1,
}
```

### Mining Submission (Backward-Compatible)

```rust
pub struct MiningSolutionRequest {
    // --- Existing fields (unchanged) ---
    pub miner_address: String,
    pub nonce: u64,
    pub hash: String,
    pub difficulty_target: String,
    pub challenge_hash: Option<String>,
    pub hash_rate: Option<f64>,
    pub miner_id: Option<String>,
    pub worker_name: Option<String>,
    pub miner_version: Option<String>,

    // --- New fields (all Optional, backward-compatible) ---
    #[serde(default)]                              // defaults to 0 (Blake3Legacy)
    pub proof_type: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vdf_output: Option<String>,                // Genus-2 Mumford representation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vdf_proof: Option<String>,                 // Wesolowski proof
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vdf_iterations_count: Option<u64>,
}
```

**Old miners** send `proof_type: 0` (or omit it entirely — defaults to 0). All new fields are `None`. Server treats as BLAKE3 legacy. **No change required for old miners.**

### Block Solution Encoding (P2P Wire Format)

```rust
pub struct MiningSolution {
    // --- Existing fields ---
    pub nonce: u64,
    pub hash: [u8; 32],
    pub difficulty_target: [u8; 32],
    pub miner_address: [u8; 32],
    pub timestamp: u64,
    // ... existing optional fields ...

    // --- New fields (serde optional, backward-compatible) ---
    #[serde(default)]
    pub proof_type: u8,                            // 0 = BLAKE3, 1 = Genus-2
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vdf_output: Option<Vec<u8>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vdf_proof: Option<Vec<u8>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vdf_iterations_count: Option<u64>,
}
```

**Old nodes** deserializing new blocks will ignore unknown fields (serde `deny_unknown_fields` is NOT set). New solutions with `proof_type: 1` will be treated as opaque data by old nodes — they can relay blocks but cannot verify Genus-2 proofs. This is acceptable because:
- Old nodes still verify BLAKE3 solutions in the same block
- Block validity requires at least one valid solution (BLAKE3 suffices)
- Genus-2 solutions are "bonus" — their reward is computed only by upgraded nodes

### Challenge Response (Extended)

```rust
pub struct MiningChallengeResponse {
    // --- Existing fields (unchanged) ---
    pub challenge_hash: String,
    pub difficulty_target: String,        // BLAKE3 target (legacy field name)
    pub block_height: u64,
    pub vdf_iterations: u32,
    pub block_reward: f64,
    pub expires_at: String,

    // --- New fields (all Optional) ---
    #[serde(skip_serializing_if = "Option::is_none")]
    pub genus2_target: Option<String>,    // Genus-2 lane difficulty target
    #[serde(skip_serializing_if = "Option::is_none")]
    pub genus2_iterations: Option<u32>,   // Required VDF iterations for Genus-2
    #[serde(skip_serializing_if = "Option::is_none")]
    pub epoch_number: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lane_rewards: Option<LaneRewards>,
}

pub struct LaneRewards {
    pub blake3_reward_per_solution: f64,
    pub genus2_reward_per_solution: f64,
}
```

**Old miners** ignore unknown fields. They see `difficulty_target` and `vdf_iterations` as before.

---

## Genus-2 VDF Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Curve** | pq192 | 384-bit prime, ~192-bit quantum security |
| **Equation** | `y^2 = x^5 + x^3 - 2x + 1` | From existing `Genus2CurveParams::pq192()` |
| **Iterations** | TBD (benchmark) | Target: 2-4 seconds on Intel i9-13900K (single core) |
| **Proof** | Wesolowski | O(log N) verification |
| **Seed** | `BLAKE3(challenge_hash \|\| nonce)` | Domain-separated from BLAKE3-only path |
| **Output hash** | `SHA3-256(vdf_output)` | Different hash family for domain separation |
| **Checkpoints** | NOT in Phase 1 | Simplicity first (reviewer correction) |

---

## Verification Queue Architecture

### BLAKE3 Lane (Unchanged)

- 8 sharded mpsc channels (existing pipeline)
- Batch size: 2000 or 50ms timeout
- rayon par_iter verification
- Throughput: ~320K verifications/sec

### Genus-2 Lane (New, Isolated)

- 4 sharded mpsc channels
- Batch size: 100 or 200ms timeout (fewer, heavier submissions)
- rayon par_iter Wesolowski verification
- Per-miner rate limit: **60/minute** (above honest peak of ~15-30/min)
- Queue depth: 5K max (vs 50K for BLAKE3)
- **Global CPU budget**: max 4 cores dedicated to Genus-2 verification

### Admission Filter (Cheap Pre-Validation)

Before expensive Wesolowski verification, reject if:
1. `vdf_output` length != expected Mumford serialization size for pq192
2. `vdf_proof` length < 32 bytes
3. `proof_type` not in {0, 1}
4. Nonce in recent dedup cache (last 10K nonces per miner)
5. Miner over rate limit

### Invalid Proof Fingerprint Cache

Cache the `SHA3(vdf_output || vdf_proof)` of recently rejected Genus-2 proofs. If a miner resubmits the same invalid proof, reject without re-verifying. Cache size: 100K entries, LRU eviction.

---

## Rollout Timeline

### Pre-H0: Prerequisites (Week 1-2)

| Task | Description | Risk |
|------|-------------|------|
| **P1: Epoch-based difficulty** | Implement in `difficulty.rs`. Per-lane, deterministic, from on-chain data | HIGH — consensus change |
| **P2: GPU benchmark** | Measure Genus-2 doubling speed on GPU vs CPU. Kill/go decision | MEDIUM |
| **P3: SageMath vectors** | 10,000 random doubling tests + full VDF evaluation on pq192 | MEDIUM |
| **P4: Protocol versioning** | Add `proof_type` field to submission/block/challenge structs | LOW |
| **P5: Queue isolation** | Add Genus-2 verification shards with rate limiting | LOW |

### H0: Shadow Mode (1 week minimum observation)

**Height:** `current_height + 60,000` (~5.5 days, announce 1 week before)

**What happens:**
- Server accepts `proof_type: 1` submissions
- Server verifies Genus-2 proofs in isolated queue
- Results logged to metrics only
- **Zero consensus effect** — Genus-2 proofs do not affect block validity
- **Zero economic effect** — Genus-2 proofs do not earn rewards
- BLAKE3 mining completely unchanged

**Metrics collected:**
- Genus-2 submissions/sec
- Verification CPU time histogram
- Valid/invalid ratio
- Queue depth and backpressure events
- Proof sizes (bytes)

**Exit criteria for H1:**
- Verification throughput sufficient (no bottleneck)
- No crash or memory leak from Genus-2 verification path
- GPU benchmark complete (GPU advantage <20x confirmed)
- At least 3 days of clean shadow operation

### H1: Economic Activation (2-3 weeks after H0)

**Height:** announced after H0 exit criteria met

**What happens:**
- Genus-2 solutions earn rewards from separate 20% budget
- Per-lane epoch-based difficulty retargeting activates
- Slint wallet CPU miner wired to compute Genus-2 VDF
- Challenge response includes `genus2_target` and `genus2_iterations`

**Old miners:** continue on BLAKE3 at 80% budget. No software update needed.
**New CPU miners:** download updated slint-wallet, select Genus-2 mining mode.

### H2: Policy Tuning (3-4 weeks after H1)

**Height:** announced after sufficient H1 data

**Based on observed data:**
- Adjust reward split if CPU participation still low (→ 70/30 or 60/40)
- Adjust VDF iteration count if needed
- Evaluate SCR-based dynamic split
- Consider adding memory-hardness if GPU advantage is borderline

---

## Risk Register (Updated)

| Risk | Prob | Impact | Mitigation | Status |
|------|------|--------|------------|--------|
| Genus-2 implementation bug | MED | HIGH | SageMath vectors + fuzz before H0 | P3 |
| GPU >20x on parallel VDF | MED | MED | Benchmark before H1; increase iterations | P2 |
| Difficulty desynchronization | LOW | HIGH | Epoch-based, deterministic, on-chain only | P1 |
| Reward spike at epoch boundary | LOW | MED | Max 10% per solution cap + rollover decay | Design |
| Verification DoS | LOW | MED | Isolated queues, rate limit, fingerprint cache | P5 |
| Old miners break | VERY LOW | HIGH | All new fields Optional, defaults to blake3 | P4 |
| Inactive lane exploit | LOW | LOW | 2% max relaxation per epoch + floor target | Design |

---

## What We Are NOT Doing

- NOT replacing BLAKE3 (GPU miners keep working, indefinitely)
- NOT using equal shared rewards (lane-bucketed with caps)
- NOT using pq128 curve (pq192 minimum for quantum security)
- NOT using server-side timestamps for difficulty (epoch-based, on-chain only)
- NOT activating without benchmarks and test vectors
- NOT using checkpoints in Phase 1 (simplicity first)
- NOT changing DAG-Knight block ordering or finality rules
- NOT requiring a hard fork (height-gated, backward-compatible wire format)
- NOT rushing the rollout (1 week minimum between stages)

---

## File Change Map

| File | Change | Stage |
|------|--------|-------|
| `crates/q-mining/src/difficulty.rs` | Replace stub with epoch-based per-lane LWMA | P1 |
| `crates/q-api-server/src/lib.rs` | Add `proof_type` to `MiningSubmission`, `CachedChallenge` | P4 |
| `crates/q-api-server/src/handlers.rs` | Extend challenge response, classify submissions by lane | P4/H0 |
| `crates/q-api-server/src/main.rs` | Add Genus-2 verification shards, metrics, shadow mode | P5/H0 |
| `crates/q-api-server/src/block_producer.rs` | Lane-aware reward calculation, epoch budgets | H1 |
| `crates/q-types/src/block.rs` | Add `proof_type` to `MiningSolution` (Optional, default 0) | P4 |
| `crates/q-vdf/src/genus2_vdf.rs` | Switch default curve from pq128 to pq192 | P1 |
| `gui/slint-wallet/src/miner.rs` | Wire `compute_genus2_vdf()`, add proof_type selection | H1 |
| `gui/slint-wallet/src/models.rs` | Add `proof_type` to `MiningSubmission` | H1 |
| `crates/q-storage/src/lib.rs` | Store per-lane solution counts per epoch for retargeting | P1 |

---

*"Three reviewers, two rounds, one clean plan. Lane-bucketed dual-proof mining with deterministic epoch-based difficulty. Shadow first, activate second, tune third."*
