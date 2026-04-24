# Technical Review: Emission Controller State Recovery
**Date:** 2026-04-24  
**Author:** Server Beta (Claude Code)  
**External Review:** DeepSeek AI (2026-04-24)  
**Status:** Design approved — ready for phased implementation  
**Severity:** HIGH — affects economic correctness of $1.3B mainnet  
**Incident Trigger:** Epsilon ran on wrong DB for ~3 days (2026-04-21 to 2026-04-24)

---

## 1. The Problem

The `EmissionController` (`crates/q-storage/src/emission_controller.rs`) is the economic brain of the chain. It controls block reward sizes, cumulative supply tracking, and PID rate correction. It is persisted as a single JSON blob in RocksDB at CF_MANIFEST key `"emission_controller_state"`.

**The vulnerability**: This state is **not replicated over P2P**. If a node boots with a fresh or wrong DB, the emission controller initializes to `total_cumulative_emission = 0`, producing silent economic errors that persist for days until the PID self-corrects.

**This happened**: Epsilon ran on a freshly re-synced 7.1GB DB for 3 days (Apr 21–24). `total_cumulative_emission` started at 0, `correction_factor` at 1.0 (wrong), `daily_records` empty. The node issued block rewards from a wrong baseline for the entire period.

**Current broken code path** (`main.rs:3282`):
```rust
match state.storage_engine.load_emission_state().await {
    Some(bytes) => balance_engine.restore_emission_state(&bytes).await,
    None => { info!("💰 No persisted emission state (first boot)"); }
    // ↑ falls through to EmissionController::new() → total_cumulative_emission = 0 — BUG
}
```

---

## 2. The EmissionController Struct

`crates/q-storage/src/emission_controller.rs:660–721`

```rust
pub struct EmissionController {
    block_windows: VecDeque<BlockWindow>,
    current_era: u64,                              // halving era 0–63
    total_emitted_this_era: u128,
    era_target_emission: u128,
    phase: EmissionPhase,
    genesis_timestamp: u64,                        // GENESIS_TIMESTAMP = 1771761600
    daily_records: BTreeMap<String, DailyEmissionRecord>,  // 90-day audit trail
    total_cumulative_emission: u128,               // critical: stored in base units (10^24 per QUG)
    daily_rate_samples: Vec<f64>,
    correction_factor: f64,                        // PID feedback, range [0.01, 3.0]
    total_blocks_tracked: u64,
    last_tracked_height: u64,
    wallclock_start_epoch: u64,
    wallclock_blocks_tracked: u64,
    wallclock_windows: VecDeque<(u64, u64)>,       // 30-min sliding window (180 × 10s buckets)
}
```

Units: `QUG_MAX_SUPPLY = 21_000_000 × 10^24` base units. After 64 days at ~7,185 QUG/day, expected `total_cumulative_emission` ≈ `460,000 × 10^24` base units.

---

## 3. Field Recovery Classification

### Category A — Deterministically Reconstructable Locally
| Field | Method |
|-------|--------|
| `genesis_timestamp` | Hardcoded `GENESIS_TIMESTAMP = 1771761600` |
| `current_era` | `era_at_time(elapsed_secs)` |
| `era_target_emission` | `era_emission(current_era)` |
| `phase` | Derived from era + elapsed time |

### Category B — Close Approximation from Time Formula
| Field | Method | Error |
|-------|--------|-------|
| `total_cumulative_emission` | `target_cumulative_at_time(elapsed)` at line 583 | ≤ 0.15% over 3-day PID convergence |

`target_cumulative_at_time()` uses pure integer arithmetic (no floating point). It computes Σ(complete eras) + partial era fraction. DeepSeek confirmed this error is "a few basis points that the PID compensates within days" — negligible economically.

### Category C — NOT Recoverable from Block History
| Field | Why Unrecoverable |
|-------|------------------|
| `correction_factor` | PID history — stochastic, depends on wall-clock timing of live block arrivals |
| `wallclock_windows` | Ring buffer of live arrivals. Turbo-synced blocks are filtered by `LIVE_BLOCK_THRESHOLD_SECS = 120` — none of them populate this buffer |
| `wallclock_start_epoch` | Wall-clock reference for this node instance |
| `daily_records` | 90-day BTreeMap requiring live observation timing — permanently lost if DB lost |
| `daily_rate_samples`, `wallclock_blocks_tracked` | Historical counters |

**Root cause of unrecoverability**: The emission controller measures wall-clock block *arrival* rate, not block timestamps. Turbo-sync at 1,100 blocks/sec delivers historical blocks all >120s old by wall-clock → none populate `wallclock_windows`. Rate windows only fill from genuinely live blocks arriving at ~1/sec.

---

## 4. Why Previous Approaches Were Insufficient

**Starting from zero (current):** Critical bug. Total supply = 0 = wrong era, wrong correction baseline. Should be removed immediately.

**Time-formula only (Option B):** Fixes `total_cumulative_emission` to within 0.15%. `correction_factor` starts at 1.0 (neutral, not actual value). `daily_records` lost permanently. DeepSeek: "The economic error is negligible — a few basis points the PID compensates within days." Safe standalone fix.

**HTTP 2-of-3 (Option A):** DeepSeek identified the attack: a malicious bootstrap peer can inject arbitrary `correction_factor` (e.g., 0.5), halving block rewards and causing miners to produce invalid coinbase blocks. The 2-of-3 majority check does not constrain `correction_factor`. No BFT guarantee.

**The right solution**: Use the same Bracha Byzantine Reliable Broadcast protocol already proven in `q-narwhal-core/src/reliable_broadcast.rs` for DAG-Knight vertex propagation.

---

## 5. Approved Solution: Bracha BRB for Emission State Sync

### Design

Adapt the existing SEND → ECHO → READY → DELIVER protocol for emission state agreement. Key differences from vertex BRB:
- Emission state is time-varying: honest nodes legitimately differ in `correction_factor` and `wallclock_windows`. Agreement is on a **quantized commitment** of `total_cumulative_emission`, not the full state hash.
- Multiple nodes may offer simultaneously (no single designated sender). The requester uses a deterministic tie-breaker (see §5.4).
- BRB runs **after** gossipsub mesh is established (not at process start), with time-formula fallback always available.

### New Gossipsub Topics

```
/qnk/mainnet-genesis/emission-sync-req     ← fresh node broadcasts request
/qnk/mainnet-genesis/emission-sync-offer   ← peers offer state (SEND phase)
/qnk/mainnet-genesis/emission-sync-echo    ← nodes echo valid offers (ECHO phase)
/qnk/mainnet-genesis/emission-sync-ready   ← ready phase
```

These are additive — existing nodes ignore them. No protocol version bump.

### New Message Types (`q-types/src/lib.rs`)

```rust
pub struct EmissionSyncRequest {
    pub requester_id: Vec<u8>,     // PeerId bytes
    pub request_id: [u8; 16],      // random nonce
    pub formula_total: u128,       // target_cumulative_at_time(now-genesis) — sanity anchor
    pub timestamp: u64,            // wall clock (bootstrap peers reject stale requests > 60s)
    pub signature: Vec<u8>,        // sign(request_id || formula_total) with requester's PeerId key
}

pub struct EmissionSyncOffer {
    pub responder_id: Vec<u8>,     // PeerId — MUST be in bootstrap peer list to be accepted
    pub request_id: [u8; 16],
    pub total_cumulative_emission: u128,
    pub total_quantized: u128,     // total / 10^22 (tolerance bucket: nearest 100 QUG)
    pub current_era: u64,
    pub correction_factor: f64,    // clamped to [0.5, 2.0] before sending
    pub height: u64,               // chain tip height at time of offer
    pub state_bytes: Vec<u8>,      // serde_json serialized EmissionController
    pub state_hash: [u8; 32],      // sha3_256(state_bytes) — integrity anchor
    pub signature: Vec<u8>,        // sign(request_id || state_hash || total_quantized.to_le_bytes())
}

pub struct EmissionSyncEcho {
    pub echoer_id: Vec<u8>,
    pub request_id: [u8; 16],
    pub state_hash: [u8; 32],
    pub total_quantized: u128,
}

pub struct EmissionSyncReady {
    pub sender_id: Vec<u8>,
    pub request_id: [u8; 16],
    pub state_hash: [u8; 32],
    pub total_quantized: u128,
}
```

### Protocol Flow

```
STARTUP SEQUENCE (triggered after gossipsub mesh established, not at process start):

  load_emission_state() returns None?
    YES → formula_total = target_cumulative_at_time(wall_now - GENESIS_TIMESTAMP)
          Try Bracha BRB (30-second window):
            Phase 1: Publish EmissionSyncRequest (signed with PeerId key)
            Phase 2: Bootstrap peers respond with EmissionSyncOffer (signed)
            Phase 3: Any node validates + echoes valid offers
            Phase 4: Ready phase with amplification
            Phase 5: Deliver → apply state
          If BRB fails/times out → from_time_based_fallback() (Option B)
    NO  → restore from local DB as before

BRB PHASES:

  PHASE 2 — OFFER (= SEND in Bracha):
    Bootstrap peer receives Request:
      Validate: request.timestamp within 60s of wall-clock
      Validate: requester signature is valid
      Clamp: correction_factor to [0.5, 2.0] before serializing
      Sign: (request_id || state_hash || total_quantized.to_le_bytes())
      Publish EmissionSyncOffer

  PHASE 3 — ECHO:
    Any node receives Offer:
      ① Verify responder_id is in configured bootstrap peer list (mandatory — DeepSeek recommendation)
      ② Verify signature against responder's PeerId key
      ③ sha3_256(state_bytes) == state_hash (integrity)
      ④ Try EmissionController::restore_from_bytes(state_bytes) → Err = reject (safety)
      ⑤ total_quantized within 1% of formula_total / 10^22
      If all pass: publish EmissionSyncEcho with {state_hash, total_quantized}
      Track: echo_votes[{state_hash, total_quantized}] → HashSet<PeerId>

  PHASE 4 — READY:
    When echo_votes for a bucket reaches threshold_2f_plus_1 (≥3, with f=1):
      Publish EmissionSyncReady for that bucket
    READY AMPLIFICATION (as in existing Bracha code):
      When ready_votes reaches threshold_f_plus_1 (≥2) without having sent Ready:
        Publish EmissionSyncReady (amplification)
    Track: ready_votes[{state_hash, total_quantized}] → HashSet<PeerId>

  PHASE 5 — DELIVER:
    When ready_votes for a bucket reaches threshold_2f_plus_1 (≥3):
      If multiple buckets reach threshold simultaneously:
        TIE-BREAKER: Accept offer with highest `height` (chain tip), break ties by state_hash lexicographic (DeepSeek recommendation)
      Retrieve state_bytes from stored Offer with matching state_hash
      Final validation before apply:
        sha3_256(state_bytes) == state_hash
        total_cumulative_emission within 1% of formula_total
        correction_factor ∈ [0.8, 1.2] (tighter clamp at delivery — limits damage even if manipulated)
      Apply EmissionController::restore_from_bytes(state_bytes)
      Save to local DB immediately
      Log: "✅ [EMISSION BRB] Delivered: total={} QUG, correction={:.4}, height={}, from {} ready votes"
```

### Byzantine Fault Tolerance Properties

For n=4 bootstrap nodes (Beta, Gamma, Epsilon, Delta), f=1:
- `threshold_2f_plus_1 = 3` (echo/ready/deliver quorum)
- `threshold_f_plus_1 = 2` (amplification threshold)

An attacker controlling 1 bootstrap node cannot:
- Cause delivery of a forged state (needs f+1=2 ready votes)
- Inject a malicious `correction_factor` (offer will not accumulate enough echoes from honest nodes)
- Prevent delivery (3 honest nodes sufficient for quorum)

### DeepSeek-Recommended Safeguards (all incorporated above)

1. **Deterministic tie-breaker**: highest `height`, then `state_hash` lexicographic order
2. **Bootstrap-peer-list enforcement**: offers from nodes NOT in the configured bootstrap list are rejected at echo phase — closes the "rogue network peer" attack vector
3. **Deserialization safety**: `restore_from_bytes()` failure = treat offer as invalid, no echo
4. **Rate-limiting**: bootstrap nodes accept at most 1 active sync session per requester PeerId; ignore `/emission-sync-req` messages older than 60s
5. **BRB after mesh established**: sync triggered only after first gossipsub peers connected, not at process start

---

## 6. Files to Modify

| File | Change | Lines |
|------|--------|-------|
| `crates/q-storage/src/emission_controller.rs` | Add `from_time_based_fallback()` after `new()` (~line 751) | ~20 |
| `crates/q-types/src/lib.rs` | Add 4 message types above | ~60 |
| `crates/q-narwhal-core/src/reliable_broadcast.rs` | Add `EmissionSyncBrb` struct (adapts existing logic) | ~150 |
| `crates/q-network/src/unified_network_manager.rs` | Subscribe + dispatch 4 new topics | ~40 |
| `crates/q-api-server/src/main.rs` | Replace `None` branch with BRB + fallback sequence | ~50 |

Total: ~320 lines. No changes to consensus, block validation, or existing gossipsub behavior.

---

## 7. What Changes vs. What Doesn't

**Changes:**
- `main.rs` None branch: zero → time-formula fallback → BRB on fresh DB
- 4 new gossipsub topics (additive, existing nodes ignore)
- 5 new message types in q-types

**Unchanged:**
- Block reward formula
- Block validation rules
- Existing gossipsub topics and message types
- Any running node with persisted emission state (BRB only fires when `load_emission_state() == None`)
- Consensus rules

---

## 8. Phased Implementation (Approved)

| Phase | What | Status | Notes |
|-------|------|--------|-------|
| 1 ✅ | Cron checkpoint on Epsilon every 6h | **Done** | `emission-20260424-0833-initial.json` captured |
| 2A | `from_time_based_fallback()` — 20 lines, standalone patch | Next binary | Ship this first, independently of BRB. Eliminates total=0 bug permanently. |
| 2B | `EmissionSyncBrb` + 4 topics + startup integration | Next binary after 2A | Testnet validation required. Byzantine node + partition tests before mainnet. |
| 3 | Embed `emission_cumulative` in block headers | 6+ weeks | Hard fork, requires height-gated upgrade, not critical |

**DeepSeek final recommendation (endorsed):**  
> "Deploy Option B immediately (zero-day patch) and then roll out the BRB sync in a subsequent release after adequate testing. No production-critical economic safety depends on the peer-sourced state; it's a convenience, not a necessity. The BRB design is sound, but its complexity warrants a slower, careful rollout."

---

## 9. Testing Requirements for Phase 2B (Before Mainnet)

- [ ] Fresh DB boot with all 4 bootstrap peers online → BRB delivers within 30s
- [ ] Fresh DB boot with 1 Byzantine peer sending malicious `correction_factor` → honest state delivered
- [ ] Fresh DB boot with 2 peers offline → BRB times out → time-formula fallback applied
- [ ] Fresh DB boot with network partition → BRB times out → fallback applied
- [ ] Rogue non-bootstrap peer sends offers → rejected at echo phase
- [ ] Stale request replay (>60s old) → rejected by bootstrap peers
- [ ] Concurrent fresh nodes → each independently reaches same delivered state
- [ ] Deserialization bomb (malformed state_bytes, valid signature) → offer rejected, no echo

---

## 10. Operational State

As of 2026-04-24:
- Epsilon: running on correct 219GB home DB, height 16,141,830+
- Emission checkpoints: `/home/orobit/emission-checkpoints/`, every 6 hours
- First checkpoint: `emission-20260424-0833-initial.json` (13KB, correct state)
- Beta and Gamma: unaffected, state authoritative

The April 21–24 wrong-DB period is resolved. Phase 2A (`from_time_based_fallback()`) will ensure it can never produce `total=0` again even if the incident recurs.
