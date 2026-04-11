# Technical Review v4: Epsilon Sync Stall — Uptime Hardening Plan

**Date:** 2026-04-11 (v4.1 — final after 3 peer review passes)  
**Severity:** Critical (mainnet, $1B+ market cap)  
**Prepared for:** Internal review + DeepSeek peer review  
**Status:** Incident resolved (restart), root causes identified, phased fixes proposed  
**Approach:** LOW-RISK ONLY — defensive additions, no refactors, no behavioral changes to consensus/validation  
**Peer Review:** v3 reviewed by DeepSeek (2 independent passes). v4 reviewed (final pass). All feedback incorporated.  

---

## 1. Incident Summary

On 2026-04-11 ~00:00 UTC, Epsilon (89.149.241.126) — the 10Gbit primary supernode — stalled at height 14,170,303 while the network tip was 14,225,000+. The node was **completely stuck for 10+ minutes** with zero height advancement.

**Resolution:** Manual intervention — stop Docker test container, kill -9 node, restart. Node caught up 55,000 blocks in ~10 seconds after restart, confirming the problem was entirely in the stalled node's internal state, not network or data availability.

| Server | Height at Incident | Network Tip | Gap | Status |
|--------|-------------------|-------------|-----|--------|
| Epsilon | 14,170,303 | 14,225,000+ | ~55K | **STUCK** — zero blocks for 10+ min |
| Beta | 14,224,981 | 14,225,000+ | ~250 | Producing blocks normally |
| Delta | — | — | — | Not checked |
| Gamma | — | — | — | Not checked |

**System state at time of incident (Epsilon):**
- **RAM:** 34GB/62GB used, **swap 8.0GB/8.0GB (100% exhausted, 8KB free)**
- **TCP sockets:** 60,106 total, **52,330 TIME_WAIT**, 4,860 established
- **Docker:** `q-sync-v10.2.9-test` container consuming 3.8GB RAM + 229% CPU + 1.28TB disk I/O
- **Sync state:** Chunk 14,177,940–14,180,439 timing out every 21.4s in infinite retry loop
- **AEGIS trust:** Beta peer at 0.00% trust (4 consecutive data failures)
- **HTTP connections:** 1,964 active (warned as "high")

**After restart (Docker stopped + node restarted):**
- **RAM:** 3.6GB/62GB used, swap 660MB/8.0GB
- **Height:** Jumped to 14,225,531 within 10 seconds
- Batch sync committing 200 blocks/batch with balance updates

---

## 2. Root Cause Analysis — 4 Cascading Failures

This was not a single bug. Four independent weaknesses compounded into a full stall:

```
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│  Docker container   │───►│  Swap exhaustion     │───►│  I/O thrashing      │
│  3.8GB + 229% CPU   │    │  8.0GB/8.0GB (100%)  │    │  All disk ops slow  │
└─────────────────────┘    └──────────────────────┘    └─────────┬───────────┘
                                                                  │
                                                                  ▼
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│  52K TIME_WAIT      │───►│  P2P responses slow  │───►│  Chunk timeouts     │
│  sockets (TCP)      │    │  (network saturated)  │    │  21.4s per attempt  │
└─────────────────────┘    └──────────────────────┘    └─────────┬───────────┘
                                                                  │
                                                                  ▼
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│  AEGIS trust → 0%   │───►│  Same peer retried   │───►│  PERMANENT STALL    │
│  (no recovery)      │    │  (no fallback logic)  │    │  (restart required) │
└─────────────────────┘    └──────────────────────┘    └─────────────────────┘
```

### 2.1 Failure #1: AEGIS Trust Death Spiral (No Recovery)

**File:** `crates/q-storage/src/aegis_sync.rs` lines 220–252  
**Risk to mainnet:** None (trust system is informational only, filter is already disabled)

The AEGIS-QL peer trust system has a fundamental flaw: **trust can only decrease, never recover.**

```rust
// aegis_sync.rs:220-252 — record_data_failure()
pub fn record_data_failure(&self, peer_id: &str) {
    // ...
    entry.data_failures += 1;
    // Data failures are weighted 2x to quickly identify incompatible nodes
    let weighted_valid = entry.valid_packs as f64;
    let weighted_failures = (entry.invalid_signatures + entry.merkle_failures
                            + entry.data_failures * 2) as f64;
    entry.trust_score = weighted_valid / (weighted_valid + weighted_failures).max(1.0);
}
```

**The math:** A fresh peer starts with `valid_packs=0, trust=0.5`. After a single data failure:
- `weighted_valid = 0.0`
- `weighted_failures = 0 + 0 + 1*2 = 2.0`
- `trust = 0.0 / 2.0 = 0.00` — **permanently blacklisted after ONE failure**

There is no:
- Time-decay of failure counters
- Trust floor (minimum score)
- Forgiveness timer
- Success-based recovery (successful chunks don't reduce failure counters)
- Stale peer cleanup

The `last_seen` timestamp field exists (line 117) but is **never used for recovery** — only updated.

**Current mitigation (insufficient):** The trust filter was disabled entirely in v2.1.5:

```rust
// turbo_sync.rs:3434-3437
// 🛡️ v2.1.5-DELTA-V: DISABLED trust filter - was causing all peers to be rejected
// The trust system has a chicken-egg problem: peers get penalized for failed syncs,
// but syncs fail because no peers are available!
let min_trust_threshold = 0.0f64;  // Was 0.1, now disabled
```

Disabling the filter means 0% trust peers are still used, but `record_data_failure()` is still called on every retry failure (turbo_sync.rs:5461), polluting logs and metrics with misleading "trust: 0.00%" warnings.

**Known gap (from peer review):** If a peer is continuously retried (e.g., the only available peer), its `last_seen` is updated on every failure. Time-based decay alone won't help because `last_seen` never ages past the threshold. This requires **success-based decay** as a complement — see Fix 3.1.

### 2.2 Failure #2: Turbo Sync Timeout Convergence + No Circuit Breaker

**Files:**  
- `crates/q-storage/src/turbo_sync.rs` lines 5424–5470 (retry loop)  
- `crates/q-storage/src/crypto_enhanced_sync.rs` lines 707–730, 826–833 (timeout calc)  
- `crates/q-storage/src/kalman_predictor.rs` lines 64–74 (Kalman timeout)  
**Risk to mainnet:** None (sync speed only, no consensus/validation)

The sync stall was a **timeout convergence loop**:

**Step 1: Both timeout predictors converge to the same high value.**

```rust
// crypto_enhanced_sync.rs:826-833 — Adaptive timeout
// Timeout = median(RTT) + 3*MAD + 100ms buffer
let calculated = (median as f64 + 3.0 * mad + 100.0) as u64;
self.current_timeout_ms = calculated.max(self.min_timeout_ms).min(self.max_timeout_ms);
```

```rust
// kalman_predictor.rs:64-74 — Kalman timeout
let timeout_ms = self.latency_ms + 4.0 * self.jitter_ms;
Duration::from_millis(timeout_ms as u64).clamp(min_timeout, max_timeout)
```

The blended timeout (70% adaptive + 30% Kalman) converged to 21.4s because:
- Under swap pressure, P2P responses genuinely took 5–7s
- Both predictors recorded these slow RTTs as "normal"
- median(7s) + 3 * MAD(2s) + 0.1s = 13.1s → escalated to 21.4s as variance grew
- Kalman: latency(7s) + 4 * jitter(3.6s) = 21.4s — **identical convergence**

**Step 2: DoS protection resets but doesn't fix the problem.**

```rust
// crypto_enhanced_sync.rs:707-730
pub fn record_timeout(&mut self) {
    self.consecutive_timeouts += 1;
    if self.consecutive_timeouts >= self.max_consecutive_timeouts {  // 5
        self.current_timeout_ms = self.min_timeout_ms * 3;  // 15s
        self.consecutive_timeouts = 0;
        self.rtt_samples.clear();
        return;
    }
    // 1.5x exponential backoff
    self.current_timeout_ms = std::cmp::min(
        self.current_timeout_ms.saturating_mul(3) / 2,
        self.max_timeout_ms,
    );
}
```

After 5 consecutive timeouts, it resets to `min * 3 = 15s` and clears samples. But the next slow RTT immediately repollutes the median. Within 2–3 measurements, the timeout climbs back to 21.4s.

**Step 3: Retry loop hammers the same peer.**

```rust
// turbo_sync.rs:5424-5470
let max_retries = 3;
loop {
    let peer_idx = (chunk_idx + retry_count as usize) % peer_count;
    let peer = if retry_count == 0 {
        ordered_peers[0]  // Gravity-assist peer
    } else {
        ordered_peers[peer_idx]  // Round-robin
    };
    match self_clone.download_and_apply_chunk(peer, start, end, retry_count).await {
        Err(e) => {
            retry_count += 1;
            if retry_count >= max_retries { return Err(e); }
            self_clone.peer_trust.record_data_failure(&peer_str);
            tokio::time::sleep(Duration::from_millis(50 * retry_count as u64)).await;
        }
        // ...
    }
}
```

When `peer_count = 1`, the round-robin `(chunk_idx + retry_count) % 1 = 0` selects the **same peer every time**. Three retries × 21.4s = **64.2 seconds burned per chunk**, then the chunk fails, and the outer sync loop restarts the same chunk.

**What's missing:**
1. No per-chunk wall-clock timeout (could stall for minutes)
2. No peer exclusion per chunk (same slow peer retried)
3. No "abandon and reschedule" — failed chunk blocks the entire batch

### 2.3 Failure #3: TIME_WAIT Socket Explosion (52K+)

**Files:**  
- `crates/q-flux/src/proxy.rs` lines 893, 1095 (raw TCP per WebSocket/SSE)  
- `crates/q-flux/src/acceptor.rs` lines 213–249 (no SO_LINGER)  
- `crates/q-api-server/src/miner_link_api.rs` lines 76–199 (miner churn)  
**Risk to mainnet:** None (socket tuning, no protocol changes)

The 52,330 TIME_WAIT sockets came from two sources:

**Source A: q-flux creates a raw TCP socket for every WebSocket/SSE upstream connection.**

Regular HTTP uses hyper's connection pool (upstream.rs:245–260) with 32 idle connections per host and 30s keepalive. **WebSocket and SSE bypass this pool entirely** — each upgrade/SSE creates a fresh `TcpStream::connect()`.

**Source B: Rapid miner connect/disconnect churn.**

Logs showed 30+ miners connecting and disconnecting per second. Each connection creates two sockets (client→q-flux + q-flux→backend). At 30 conn/sec × 2 sockets × 60s TIME_WAIT = **3,600 TIME_WAIT at steady state**. Under sustained load, this exceeds 50K.

**Clarification (from peer review):** `tcp_fin_timeout` governs `FIN_WAIT2` state, **NOT** `TIME_WAIT` duration. The Linux kernel hardcodes TIME_WAIT at 60 seconds (defined as `TCP_TIMEWAIT_LEN` in `include/net/tcp.h`). There is no sysctl to reduce it. The primary mitigation levers are:
- `tcp_tw_reuse=1` — allows reuse of TIME_WAIT sockets for new outgoing connections (already set on Epsilon)
- Reducing connection churn (the real fix — requires pooling or rate limiting)
- Expanding ephemeral port range to tolerate higher TIME_WAIT counts

### 2.4 Failure #4: Health Endpoint Blind to Memory Pressure

**File:** `crates/q-api-server/src/handlers.rs` lines 265–320  
**Risk to mainnet:** None (monitoring only)

The `/api/v1/health` endpoint is purely height-based:

```rust
// handlers.rs:289-320
let status = if current_height == 0 {
    "starting".to_string()
} else if network_height > 0 && current_height + 10 < network_height {
    "syncing".to_string()
} else {
    "ready".to_string()
};
```

**HealthStatus struct (lines 265–275):**
```rust
pub struct HealthStatus {
    pub status: String,          // "starting" | "syncing" | "ready"
    pub height: u64,
    pub network_height: u64,
    pub peers: usize,
    pub version: String,
    pub uptime_secs: u64,
    pub balance_state_hash: String,
    pub wallet_count: usize,
    pub total_supply_qug: String,
}
```

No memory metrics, no swap usage, no connection count, no sync stall detection.

During the incident, `/health` returned `200 OK status: "syncing"` — technically correct but giving **zero indication** of catastrophic resource exhaustion.

**Critical gap (from peer review):** The `MemoryLimiter` defines Critical pressure as >90% RSS. During this incident, RSS was 34GB/62GB (55%) — **not critical by RSS alone**. The problem was **swap exhaustion** (8GB/8GB = 100%). The pressure detection must include swap, or it will miss the exact failure mode that caused this incident.

---

## 3. Proposed Fixes — Phased Approach

### Phase 1: Ship Now (Low Risk, High Impact)

These changes are additive, well-understood, and independently revertible. They address the direct causes of the stall.

### Phase 2: Ship After Soak (Medium Risk or Needs Validation)

These changes are directionally correct but need load testing or have subtle edge cases identified during peer review.

### Risk Classification

Every proposed fix is evaluated against:

| Criterion | Requirement |
|-----------|-------------|
| **Consensus** | ZERO changes to block validation, signature verification, or consensus rules |
| **State** | ZERO changes to database schema, block format, or balance logic |
| **Protocol** | ZERO changes to P2P wire format, gossipsub topics, or handshake |
| **Behavior at tip** | Node at network tip producing blocks must be UNAFFECTED |
| **Rollback** | Every change must be safe to revert without data migration |

---

### PHASE 1 — Ship Now

---

### 3.1 Fix: AEGIS Trust Time-Decay + Success-Based Recovery

**Risk: VERY LOW** — Trust scores are informational only (filter disabled since v2.1.5). No code path uses trust scores for consensus, validation, or peer selection decisions. The filter threshold is 0.0 and remains 0.0.

**Change:** Add two recovery mechanisms:
1. Periodic time-decay of failure counters for peers that have not failed recently
2. Consecutive-success-based decay for actively-used peers (addresses the peer review gap where continuously-retried peers never recover because `last_seen` is always fresh)

**Struct change:** Add two fields to `PeerTrustMetrics`:
- `last_failure_at: i64` — timestamp of most recent failure (distinct from `last_seen` which updates on any interaction)
- `consecutive_successes: u64` — resets to 0 on any failure

**File:** `crates/q-storage/src/aegis_sync.rs`

```rust
// ADD to PeerTrustMetrics struct:
pub last_failure_at: i64,        // v10.2.10: timestamp of last failure
pub consecutive_successes: u64,  // v10.2.10: resets on failure

// NEW: Add to PeerTrustRegistry impl

/// v10.2.10: Time-decay failure counters every 5 minutes.
/// Peers that have not failed recently should not be permanently blacklisted.
/// Halves failure counters for peers whose last failure was >5 minutes ago.
/// Note: uses `last_failure_at` (not `last_seen`) so that actively-used
/// peers with no recent failures also benefit from decay.
pub fn apply_time_decay(&self) {
    let now = chrono::Utc::now().timestamp();
    let decay_threshold_secs = 300; // 5 minutes

    for mut entry in self.peers.iter_mut() {
        let since_last_failure = now - entry.last_failure_at;
        if since_last_failure > decay_threshold_secs && entry.trust_score < 0.5 {
            entry.invalid_signatures /= 2;
            entry.merkle_failures /= 2;
            entry.data_failures /= 2;
            Self::recalculate_trust(&mut entry);
        }
    }
}

/// v10.2.10: Consecutive-success-based recovery.
/// After 5 consecutive successful chunk downloads from a peer,
/// reduce its failure counters by 1. The `consecutive_successes` counter
/// resets to 0 on any failure, so this only triggers for peers that have
/// genuinely recovered (not peers with intermittent failures).
pub fn record_successful_chunk(&self, peer_id: &str) {
    if let Some(mut entry) = self.peers.get_mut(peer_id) {
        entry.valid_packs += 1;
        entry.consecutive_successes += 1;

        if entry.consecutive_successes >= 5 && entry.trust_score < 0.5 {
            entry.invalid_signatures = entry.invalid_signatures.saturating_sub(1);
            entry.merkle_failures = entry.merkle_failures.saturating_sub(1);
            entry.data_failures = entry.data_failures.saturating_sub(1);
            entry.consecutive_successes = 0; // Reset — next recovery needs 5 more
        }
        Self::recalculate_trust(&mut entry);
    }
}

/// Update record_data_failure to reset consecutive_successes and set last_failure_at:
pub fn record_data_failure(&self, peer_id: &str) {
    // ... existing logic ...
    entry.consecutive_successes = 0;  // v10.2.10: reset on failure
    entry.last_failure_at = chrono::Utc::now().timestamp();
    // ... rest unchanged ...
}

fn recalculate_trust(entry: &mut PeerTrustMetrics) {
    let weighted_valid = entry.valid_packs as f64;
    let weighted_failures = (entry.invalid_signatures
        + entry.merkle_failures
        + entry.data_failures * 2) as f64;
    entry.trust_score = if weighted_valid + weighted_failures > 0.0 {
        weighted_valid / (weighted_valid + weighted_failures)
    } else {
        0.5 // Reset to neutral if all counters decayed to 0
    };
}

/// v10.2.10: Remove peers not seen in over 1 hour.
pub fn cleanup_stale_peers(&self) {
    let now = chrono::Utc::now().timestamp();
    self.peers.retain(|_, entry| (now - entry.last_seen) < 3600);
}
```

**Caller:** 
- `apply_time_decay()` and `cleanup_stale_peers()`: 5-minute tokio interval in sync loop (background maintenance).
- `record_successful_chunk()`: after every successful `download_and_apply_chunk()` in the retry loop (turbo_sync.rs, next to existing `record_data_failure()`).

**Why this is safe:**
- Trust scores are currently unused (threshold = 0.0)
- Only affects logging output (trust percentage in warn messages)
- DashMap iteration is lock-free (no contention with sync)
- Consecutive-success recovery requires 5 unbroken successes — intermittent failures reset the counter
- `last_failure_at` is a new field with no external consumers; default to `0` for existing peers (immediately eligible for decay, which is correct — old peers should benefit)
- Worst case: a truly malicious peer gets a second chance — will fail again and reset `consecutive_successes` to 0

### 3.2 Fix: Sync Stall Circuit Breaker (Wall-Clock Deadline + Peer Exclusion)

**Risk: LOW** — Changes only affect the sync-behind path. A node at network tip never enters this code. No consensus, validation, or block production code is touched.

**Change:** Two defensive additions to the retry loop. (Chunk splitting is deferred to Phase 2 — see 3.8.)

**File:** `crates/q-storage/src/turbo_sync.rs`

#### 3.2a: Per-Chunk Wall-Clock Deadline

If a chunk cannot be fetched in 120s across all retries, **abandon the current attempt and let the outer sync loop reschedule it**. The chunk is not "skipped" in the sense of leaving a gap — the outer loop will re-evaluate gaps and retry the range on the next iteration.

```rust
// Define a typed error variant (NOT string matching):
#[derive(Debug)]
pub enum ChunkError {
    WallClockTimeout { start: u64, end: u64, elapsed: Duration },
    TransportTimeout { peer: String, duration: Duration },
    ValidationFailure { peer: String, reason: String },
    Other(anyhow::Error),
}

// In the retry loop (line ~5424), add:
let chunk_deadline = tokio::time::Instant::now() + Duration::from_secs(120);

loop {
    if tokio::time::Instant::now() >= chunk_deadline {
        warn!("⏰ [CIRCUIT BREAKER] Chunk {}-{} exceeded 120s wall-clock limit — \
               abandoning attempt, will be rescheduled by outer sync loop",
              start, end);
        return Err(ChunkError::WallClockTimeout {
            start, end,
            elapsed: Duration::from_secs(120),
        });
    }
    // ... existing retry logic
}
```

**Backoff on abandoned chunks (peer review feedback):** To prevent immediate re-attempt of a chunk that just timed out 3 times, add a 60-second cooldown in the outer sync loop when a chunk returns a wall-clock timeout error. Uses typed error variant (not string matching — per final peer review).

```rust
// In the outer sync loop, after collecting chunk results:
let has_wall_clock_timeout = failed_chunks.iter()
    .any(|e| matches!(e, ChunkError::WallClockTimeout { .. }));
if has_wall_clock_timeout {
    info!("⏳ [CIRCUIT BREAKER] Cooling down 60s before retrying timed-out chunks");
    tokio::time::sleep(Duration::from_secs(60)).await;
}
```

**Why this is safe:**
- Abandoned chunks will be retried on the next sync iteration (outer loop re-evaluates height gaps)
- Block application is contiguous — the outer loop requests blocks from current_height to target, so abandoned chunks don't create sparse gaps; they simply mean the batch ends at the last successfully applied height
- 120s is generous — normal chunks complete in 1–5 seconds
- 60s cooldown prevents hot-loop retries of stuck chunks
- This only triggers under pathological conditions (like this incident)

#### 3.2b: Per-Chunk Peer Exclusion

Track which peers have timed out on this specific chunk and prefer different peers for retries. **Exclusion applies only to retryable transport/timeout failures** — validation failures (invalid signature, bad Merkle) continue through existing AEGIS integrity-handling paths and should not merely exclude the peer from one chunk.

```rust
// In the retry loop, maintain a per-chunk exclusion set:
let mut excluded_peers: HashSet<String> = HashSet::new();

loop {
    let available: Vec<_> = ordered_peers.iter()
        .filter(|p| !excluded_peers.contains(&p.to_string()))
        .collect();

    let peer = if available.is_empty() {
        // All peers failed on this chunk — last resort, use any peer
        // (better than deadlocking; the wall-clock deadline will catch us)
        ordered_peers[retry_count as usize % peer_count]
    } else if retry_count == 0 {
        *available[0]
    } else {
        *available[retry_count as usize % available.len()]
    };

    match download_and_apply_chunk(peer, start, end, retry_count).await {
        Err(ChunkError::TransportTimeout { .. }) => {
            // Transport/timeout: exclude this peer for THIS chunk only
            excluded_peers.insert(peer.to_string());
            // ... existing retry logic
        }
        Err(ChunkError::ValidationFailure { .. }) => {
            // Validation: do NOT exclude — pass through to AEGIS integrity handlers
            // which apply broader penalties (record_invalid_signature, record_merkle_failure)
            // ... existing retry logic
        }
        Err(e) => {
            // Other errors: exclude peer (defensive)
            excluded_peers.insert(peer.to_string());
            // ... existing retry logic
        }
        // ...
    }
}
```

**Why this is safe:**
- Only affects peer selection within a single chunk's retry loop
- Falls back to any peer if all are excluded (no deadlock possible)
- No persistent state changes — exclusion set is per-chunk, per-iteration
- Validation failures continue through AEGIS integrity paths (no behavioral change)
- Wall-clock deadline (3.2a) provides the ultimate escape hatch

### 3.3 Fix: Timeout Convergence Ceiling (15s)

**Risk: VERY LOW** — Only affects timeout duration for sync requests. Worst case: slightly more retries on legitimately slow networks (handled by the retry loop). No impact at network tip.

**Change:** Cap the blended timeout at 15s instead of 30s.

**File:** `crates/q-storage/src/crypto_enhanced_sync.rs`

```rust
// In timeout calculation (line ~826-833):
let calculated = (median as f64 + 3.0 * mad + 100.0) as u64;

// v10.2.10: Hard ceiling at 15s.
// The previous 30s max allowed timeout convergence to 21.4s which caused
// the 2026-04-11 sync stall. 15s is still 3x typical chunk RTT (1-5s).
// Chunks needing >15s will fail and be retried with a different peer.
let effective_max = 15_000u64.min(self.max_timeout_ms);

self.current_timeout_ms = calculated
    .max(self.min_timeout_ms)
    .min(effective_max);
```

**File:** `crates/q-storage/src/kalman_predictor.rs`

```rust
// In optimal_timeout() (line ~64-74):
let timeout_ms = self.latency_ms + 4.0 * self.jitter_ms;
// v10.2.10: Reduced from 60s to 15s to match adaptive timeout ceiling.
// Kalman's latency + 4*jitter can exceed 15s; clamping is intentional.
let max_timeout = Duration::from_secs(15);
Duration::from_millis(timeout_ms as u64).clamp(min_timeout, max_timeout)
```

**Why this is safe:**
- 15s is 3x typical chunk RTT on 10Gbit
- Chunks needing >15s fail → retry with different peer → retry loop handles it
- The clamp will truncate high Kalman estimates — that is the intent
- No impact at network tip (sync code inactive)

### 3.4 Fix: Health Endpoint Enrichment

**Risk: VERY LOW** — Read-only additions to health response. Existing fields unchanged. New fields use `skip_serializing_if` so existing API consumers see no change.

**Change:** Add resource metrics and degradation signaling. Split into liveness vs. degradation (per peer review feedback — don't overload readiness with raw memory pressure).

**File:** `crates/q-api-server/src/handlers.rs`

```rust
// Extend HealthStatus struct (all new fields are Option, skip if None):
pub struct HealthStatus {
    // ... existing 9 fields unchanged ...

    // v10.2.10: Resource observability
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_rss_mb: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub swap_used_percent: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active_connections: Option<u64>,

    // v10.2.10: Degradation reasons (empty = healthy)
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub degraded_reasons: Vec<String>,

    // v10.2.10: Sync stall detection
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sync_stalled_secs: Option<u64>,  // seconds since last height change, if stalled
}
```

**Degradation logic (NOT 503 — informational only for Phase 1):**

```rust
// In health_check handler:
let mut degraded_reasons = Vec::new();

// Check swap exhaustion (the actual trigger in this incident)
if swap_used_percent > 95 {
    degraded_reasons.push("swap_exhausted".to_string());
}
// Check memory pressure from MemoryLimiter
if memory_pressure == MemoryPressure::Critical {
    degraded_reasons.push("memory_critical".to_string());
}
// Check sync stall
if let Some(stall_secs) = sync_stalled_secs {
    if stall_secs > 300 {
        degraded_reasons.push(format!("sync_stalled_{}s", stall_secs));
    }
}
// Check connection pressure
if active_connections > max_connections * 80 / 100 {
    degraded_reasons.push("connection_pressure".to_string());
}

// Status remains 200 OK — degraded_reasons is informational.
// Phase 2 may add a separate /ready endpoint that returns 503.
```

**Why NOT returning 503 in Phase 1 (peer review feedback):**
- `MemoryPressure::Critical` is based on RSS (>90%), but the incident had 55% RSS with 100% swap — **it would not have triggered**.
- Returning 503 based on a metric that misses the failure mode is worse than no 503 (false sense of safety).
- The `MemoryLimiter` pressure calculation needs to incorporate swap before we can trust it for 503 decisions.
- Phase 1 adds the fields for monitoring; Phase 2 adds the swap-aware 503 after validation.

**File:** `crates/q-storage/src/memory_limiter.rs`

```rust
// v10.2.10: Include swap in pressure calculation.
// The 2026-04-11 incident had 55% RSS but 100% swap — pure RSS thresholds
// would have missed it.
//
// Note: MemoryPressure must derive Ord with variants ordered by severity:
//   Low < Medium < High < Critical
// This is satisfied by the existing enum definition order (derive(Ord)
// uses declaration order).

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]  // ADD Ord derivation
pub enum MemoryPressure { Low, Medium, High, Critical }

pub async fn get_memory_pressure(&self) -> MemoryPressure {
    let rss_pressure = self.get_rss_pressure();
    let swap_pressure = self.get_swap_pressure(); // NEW

    // Return the more severe of the two
    std::cmp::max(rss_pressure, swap_pressure)
}

fn get_swap_pressure(&self) -> MemoryPressure {
    // Read /proc/meminfo for swap usage
    let (swap_total, swap_used) = Self::read_swap_usage();
    if swap_total == 0 { return MemoryPressure::Low; }
    let percent = (swap_used * 100) / swap_total;
    match percent {
        0..=60 => MemoryPressure::Low,
        61..=80 => MemoryPressure::Medium,
        81..=95 => MemoryPressure::High,
        _ => MemoryPressure::Critical,
    }
}
```

**Why this is safe:**
- `skip_serializing_if` means existing consumers see no change
- `degraded_reasons` is informational — no behavioral change (no 503)
- Swap pressure detection is read-only (reads `/proc/meminfo`)
- Swap-aware pressure feeds into existing `should_pause_sync()` — a bonus: the sync pause will now also trigger on swap exhaustion, not just RSS

### 3.5 Fix: AEGIS Trust Decay Caller + Operational Sysctl

**Risk: OPERATIONAL** — No code changes for sysctl. Decay caller is a simple tokio interval.

#### 3.5a: Sysctl Tuning (Epsilon) — Already Applied

```bash
# /etc/sysctl.d/99-qnk.conf — current state:
net.ipv4.ip_local_port_range = 32768 60999
# TIME_WAIT is kernel-hardcoded 60s (TCP_TIMEWAIT_LEN); no sysctl changes it.
# tw_reuse allows reuse of TIME_WAIT sockets for new outbound connections (RFC 6191).
net.ipv4.tcp_tw_reuse = 1
net.ipv4.tcp_rmem = 4096 87380 1048576
net.ipv4.tcp_wmem = 4096 87380 1048576
```

**Correction from v3 (peer review):** `tcp_fin_timeout` governs `FIN_WAIT2`, **not** `TIME_WAIT`. The TIME_WAIT duration on Linux is hardcoded at 60s in the kernel (`TCP_TIMEWAIT_LEN`). There is no sysctl to change it. `tcp_tw_reuse=1` (already set) is the correct mitigation — it allows the kernel to reuse TIME_WAIT sockets for new outbound connections to the same destination.

The 52K TIME_WAIT buildup was primarily caused by the node being in a degraded state for hours (poisoned sync, high churn). After restart, TIME_WAIT dropped to 703 within minutes. The systemic fix is reducing connection churn (Phase 2: connection pooling for WebSocket/SSE).

#### 3.5b: Docker Resource Limits (Operational Rule)

**All Docker test containers on Epsilon must use resource limits:**
```bash
docker run --memory=4g --cpus=4 --memory-swap=6g ...
```

Without limits, a Docker container can consume unbounded RAM and push the host node into swap. This was the initial trigger for the cascade.

---

### PHASE 2 — Ship After Soak Testing

---

### 3.6 Fix: Sync Stall Watchdog (Auto-Recovery)

**Risk: LOW** — But requires tighter trigger conditions than proposed in v3. Deferred to Phase 2 per peer review.

**Change:** Background watchdog that resets sync predictor state when the node is genuinely stalled.

**Peer review improvements incorporated:**
1. **Tighter trigger conditions** — only fires when ALL of these are true:
   - Node is behind tip by >100 blocks (not just `height > 0`)
   - Sync mode is active (not idle/paused)
   - No committed height progress for 5 minutes
   - At least one sync request has been attempted in the last 60 seconds (active stall, not idle)
2. **Cooldown** — do not reset more than once per 30 minutes to prevent oscillation
3. **Targeted reset** — reset sync predictors and current stuck work, NOT global trust

```rust
/// v10.2.10: Sync stall watchdog — Phase 2 (requires soak testing).
pub async fn sync_stall_watchdog(
    current_height: Arc<AtomicU64>,
    network_height: Arc<AtomicU64>,
    sync_active: Arc<AtomicBool>,       // true when sync loop is running
    last_sync_attempt: Arc<AtomicU64>,   // epoch seconds of last request
    adaptive_timeout: Arc<Mutex<AdaptiveTimeout>>,
    kalman: Arc<Mutex<KalmanPredictor>>,
    peer_trust: Arc<PeerTrustRegistry>,
) {
    let mut last_height = 0u64;
    let mut stall_start: Option<Instant> = None;
    let mut last_reset: Option<Instant> = None;
    let stall_threshold = Duration::from_secs(300);     // 5 min
    let cooldown = Duration::from_secs(1800);            // 30 min between resets

    loop {
        tokio::time::sleep(Duration::from_secs(30)).await;

        let height = current_height.load(Ordering::Relaxed);
        let net_height = network_height.load(Ordering::Relaxed);
        let is_syncing = sync_active.load(Ordering::Relaxed);
        let last_attempt = last_sync_attempt.load(Ordering::Relaxed);
        let now_epoch = chrono::Utc::now().timestamp() as u64;

        // Condition 1: Behind by >100 blocks
        let behind = net_height > height + 100;
        // Condition 2: Sync mode is active
        // Condition 3: No height progress
        let stuck = height == last_height && height > 0;
        // Condition 4: Active stall (sync attempted in last 60s)
        let recently_attempted = now_epoch.saturating_sub(last_attempt) < 60;

        if behind && is_syncing && stuck && recently_attempted {
            let stall_time = stall_start.get_or_insert(Instant::now());

            if stall_time.elapsed() >= stall_threshold {
                // Check cooldown — no reset within 30 min of last reset
                if let Some(lr) = last_reset {
                    if lr.elapsed() < cooldown {
                        warn!("🚨 [STALL WATCHDOG] Stall detected but in cooldown \
                               (last reset {:?} ago, need {:?})",
                               lr.elapsed(), cooldown);
                        continue;
                    }
                }

                warn!("🚨 [STALL WATCHDOG] Height {} unchanged for {:?}, \
                       behind by {} blocks, sync active — resetting predictors",
                       height, stall_time.elapsed(), net_height - height);

                // Reset adaptive timeout samples
                if let Ok(mut timeout) = adaptive_timeout.lock() {
                    timeout.reset_to_default();
                }
                // Reset Kalman predictor
                if let Ok(mut k) = kalman.lock() {
                    k.reset();
                }
                // Single trust decay (not double — peer review noted implicit doubling)
                peer_trust.apply_time_decay();

                last_reset = Some(Instant::now());
                stall_start = None;

                warn!("✅ [STALL WATCHDOG] Reset complete — next sync iteration \
                       will use fresh timeouts");
            }
        } else {
            stall_start = None;
            last_height = height;
        }
    }
}
```

**Why deferred to Phase 2:**
- The watchdog interacts with the timeout predictor → needs to be tested under real network conditions
- If the stall is caused by genuinely slow network (not poisoned state), resetting predictors will cause rapid re-convergence back to the same timeout → reset → stall → reset loop (mitigated by 30-min cooldown, but needs validation)
- The Phase 1 fixes (timeout ceiling, circuit breaker, peer exclusion) should prevent most stalls without the watchdog

### 3.7 Fix: SO_LINGER=0 on q-flux Upstream Sockets

**Risk: MEDIUM** — Converts graceful close (FIN) to abortive close (RST). May change error semantics visible to the peer. Deferred to Phase 2 per peer review.

**File:** `crates/q-flux/src/proxy.rs`

```rust
// In handle_websocket_upgrade() after TcpStream::connect (line ~893):
let stream = tokio::net::TcpStream::connect(backend).await?;
// v10.2.10 Phase 2: Set SO_LINGER=0 to avoid TIME_WAIT on upstream close.
// Converts FIN to RST on active close. Only for localhost connections.
// CAUTION: Does NOT help when backend initiates close (FIN_WAIT2 from backend,
// TIME_WAIT from q-flux passive close). Only effective when q-flux actively
// closes the upstream socket (client disconnected → q-flux closes upstream).
let sock_ref = socket2::SockRef::from(&stream);
sock_ref.set_linger(Some(Duration::from_secs(0)))?;
```

**Why deferred to Phase 2:**
- SO_LINGER=0 can truncate unread buffered data on the peer side
- For long-lived WebSocket/SSE streams, the close lifecycle is complex — needs packet capture validation
- Only helps when q-flux initiates close; when backend closes (e.g., idle timeout), q-flux still accumulates TIME_WAIT
- The real systemic fix is connection pooling for WebSocket/SSE (rejected for hotfix, but should be the Phase 3 target)
- `tcp_tw_reuse=1` (already set) provides the primary mitigation

### 3.8 Fix: Adaptive Chunk Splitting

**Risk: MEDIUM** — The v3 pseudocode was too naive (both reviewers flagged this). Splitting inline without routing halves through the retry/scheduler machinery creates subtle issues. Deferred to Phase 2.

**Issues with v3 approach (peer review):**
1. Both halves used the same `peer` — defeats the purpose
2. Halves ran sequentially inline, not through the retry loop
3. Did not use the peer exclusion or timeout machinery
4. If a single corrupted block exists in the range, both halves may still fail

**Correct approach (Phase 2):** Replace one failed work item in the scheduler with two smaller work items, letting normal peer selection and retry logic handle them. This requires changes to the sync batch planner, not the inner retry loop.

```rust
// Phase 2: In the OUTER sync loop (batch planner), not the inner retry loop:
// After a chunk fails all retries, if chunk_size > 500:
//   - Remove the failed (start, end) from the work queue
//   - Insert (start, mid) and (mid+1, end) into the work queue
//   - Both halves go through normal scheduling with full retry + peer exclusion
//
// This is ~30 lines of change in the batch planner, not in download_and_apply_chunk.
```

**Why deferred to Phase 2:**
- Requires understanding the batch planner's work queue semantics
- Needs testing with concurrent chunk downloads to verify no ordering issues
- Phase 1's wall-clock deadline + 60s cooldown prevents infinite stall without splitting

---

## 4. Changes NOT Proposed (Too Risky for Mainnet)

| Change | Why Rejected |
|--------|-------------|
| Re-enable AEGIS trust filter (threshold > 0.0) | Could cause peer starvation — the original chicken-egg problem |
| Change chunk size defaults | Affects all nodes, hard to predict interaction with different network conditions |
| Modify P2P wire format for block-pack responses | Protocol change — requires coordinated upgrade across all nodes |
| Add connection pooling for WebSocket/SSE in q-flux | Significant refactor of proxy.rs — correct but too many moving parts for a hotfix |
| Change block-pack semaphore limits | Current limits (4 concurrent) are battle-tested; changing risks OOM regression |
| Auto-kill Docker containers on memory pressure | Operational risk — could kill legitimate test containers |
| Reduce `max_retries` from 3 to 2 | Less resilient to transient failures; the problem is timeout duration, not retry count |
| Return 503 from `/health` on memory pressure (Phase 1) | Current `MemoryPressure::Critical` is RSS-only; would not have triggered during this incident (55% RSS, 100% swap). Must add swap awareness first (done in 3.4), then validate before enabling 503. Target for Phase 2. |

---

## 5. Implementation Priority

### Phase 1: Ship Now

| Order | Fix | Files Changed | Lines Changed | Risk | Impact |
|-------|-----|--------------|---------------|------|--------|
| 1 | 3.3: Timeout ceiling 15s | `crypto_enhanced_sync.rs`, `kalman_predictor.rs` | ~6 lines | **Very Low** | Prevents 21.4s convergence trap |
| 2 | 3.2a: Chunk wall-clock deadline | `turbo_sync.rs` | ~15 lines | **Low** | Abandons stuck chunks after 120s |
| 3 | 3.2b: Per-chunk peer exclusion | `turbo_sync.rs` | ~20 lines | **Low** | Avoids retrying same slow peer |
| 4 | 3.4: Health endpoint + swap pressure | `handlers.rs`, `memory_limiter.rs` | ~50 lines | **Very Low** | Operational visibility |
| 5 | 3.1: AEGIS trust decay + success recovery | `aegis_sync.rs` | ~50 lines | **Very Low** | Clean metrics, peer recovery |
| 6 | 3.5b: Docker resource limits | Operational (no code) | 0 lines | **Operational** | Prevent resource starvation |

**Phase 1 total:** ~140 lines of additions across 5 files. No deletions.

### Phase 2: Ship After Soak

| Order | Fix | Risk | Why Deferred |
|-------|-----|------|-------------|
| 7 | 3.6: Stall watchdog | **Low** | Needs validation of reset-oscillation behavior |
| 8 | 3.8: Adaptive chunk splitting | **Medium** | Needs scheduler-level implementation, not inline |
| 9 | 3.7: SO_LINGER=0 on q-flux | **Medium** | Needs packet capture validation for long-lived streams |
| 10 | Health 503 on degradation | **Low** | Needs swap-aware pressure validated in production first |

---

## 6. Testing Strategy

### Pre-deployment (Beta, before ha-deploy.sh):
```bash
# 1. Existing test suites MUST pass (zero regressions)
cargo test --package q-storage --test mainnet_critical_tests
cargo test --package q-storage --test balance_propagation_tests
cargo test --package q-storage --test sync_down_protection_tests
cargo test --package q-types --test signature_verification_tests

# 2. New unit tests for added code
cargo test --package q-storage aegis_time_decay
cargo test --package q-storage aegis_success_recovery
cargo test --package q-storage chunk_wall_clock_deadline
cargo test --package q-storage peer_exclusion_per_chunk
cargo test --package q-storage swap_pressure_detection
```

### Docker soak test (Epsilon, 24 hours — BEFORE Phase 2):
```bash
# Build Debian 12 binary via Docker
# Run with memory-constrained container (4GB limit, 2GB swap)
# Verify: sync completes without stalls
# Target: TIME_WAIT remains below ~5K under steady-state load; investigate sustained excursions
# Monitor: /health endpoint shows swap_used_percent field
# Simulate: tc qdisc add latency 5000ms on peer interface → verify timeout ceiling holds at 15s
# Simulate: kill -STOP peer response process → verify wall-clock deadline triggers at 120s
# Simulate: single peer with intermittent 50% packet loss → verify peer exclusion selects alt peer
```

### Chaos test plan (Phase 2 validation):
```
Scenario: Reproduce exact 2026-04-11 failure cascade
1. Start node with single peer, 8GB memory limit
2. Run competing Docker container consuming 4GB
3. Inject 5-7s latency on P2P responses (tc qdisc)
4. Verify:
   - Timeout ceiling prevents convergence above 15s ✓
   - Wall-clock deadline abandons chunks after 120s ✓
   - Peer exclusion avoids re-selecting failed peer ✓
   - Health endpoint reports swap_used_percent > 95 ✓
   - (Phase 2) Watchdog triggers at 5 min and resets predictors ✓
   - (Phase 2) Watchdog does NOT trigger again within 30 min ✓
   - Node recovers sync without manual restart ✓
```

### Post-deployment verification:
```bash
# On Epsilon:
journalctl -u q-api-server --since "5 minutes ago" | grep -E "CIRCUIT BREAKER|AEGIS.*decay|AEGIS.*success"
# Should see: no circuit breaker triggers (healthy), periodic decay, success recovery

# Verify health endpoint:
curl -s localhost:8080/api/v1/health | jq '.data.swap_used_percent, .data.degraded_reasons'
# Should see: swap_used_percent < 50, degraded_reasons empty

# Monitor TIME_WAIT (target: < 5K under steady-state load; investigate sustained excursions above):
ss -s | grep TIME-WAIT
```

---

## 7. Operational Recommendations (Immediate, No Code)

| # | Action | Command | Effect |
|---|--------|---------|--------|
| 1 | Limit Docker test containers | Always use `--memory=4g --cpus=4` | Prevent resource starvation |
| 2 | Add swap monitoring alert | `if swap_used > 90%; alert "Epsilon swap critical"` | Early warning before stall |
| 3 | Verify sysctl persisted | `cat /etc/sysctl.d/99-qnk.conf` | tcp_tw_reuse=1 survives reboot |
| 4 | Set Docker OOM score adj | `--oom-score-adj=500` on test containers | Kernel kills Docker before node. **Ensure the node service has a lower (more protected) score** — verify with `cat /proc/$(pgrep -f q-api-server)/oom_score_adj` |

---

## 8. Peer Review Feedback Integration Log

| Reviewer Point | v3 State | v4/v4.1 Change |
|---------------|----------|-----------|
| `tcp_fin_timeout` does NOT control TIME_WAIT | Incorrectly claimed "reduces TIME_WAIT from 60s to 15s" | v4: Corrected — TIME_WAIT is kernel-hardcoded 60s; `tcp_fin_timeout` only affects FIN_WAIT2 |
| SO_LINGER=0 is not "very low risk" | Classified as Very Low | v4: Reclassified as Medium, deferred to Phase 2 |
| Watchdog may oscillate (reset → re-converge → stall → reset) | No cooldown, no trigger conditions beyond `height unchanged` | v4: Added 30-min cooldown + 4 trigger conditions (behind, syncing, stuck, recently attempted) |
| Chunk splitting pseudocode too naive | Inline splitting with same peer | v4: Deferred to Phase 2; correct approach is scheduler-level, not inline |
| "Skip and backfill" semantics unclear | Said "skip and continue syncing" | v4: Clarified: "abandon and reschedule" — outer loop re-evaluates gaps, no sparse state |
| Active peer never recovers (last_seen always fresh) | Time-decay only, requires `last_seen` age > 5min | v4.1: Added `last_failure_at` field for decay gating; `consecutive_successes` counter (resets on failure) for success-based recovery. Prose and code now match. |
| Health 503 on memory pressure misses swap | RSS-only Critical threshold | v4: Added swap-aware pressure detection; deferred 503 to Phase 2 until validated |
| Backoff on timed-out chunks | No backoff — immediate retry | v4: Added 60s cooldown after wall-clock timeout |
| Chunk splitting min size 500 may still contain bad block | Min 500 could still fail | v4: Acknowledged; wall-clock deadline is the escape hatch, not splitting |
| Docker OOM score adjustment | Not mentioned | v4: Added. v4.1: Added note to verify node has lower (protected) OOM score. |
| Success recovery uses lifetime modulo, not consecutive | `valid_packs % 5` — prose said "consecutive" but code used lifetime counter | v4.1: Replaced with `consecutive_successes` counter that resets to 0 on failure. 5 consecutive successes → reduce failures by 1, then counter resets. |
| `last_seen` wrong field for decay gating | `last_seen` updated on any interaction, not just failures | v4.1: Added `last_failure_at` field. Time-decay now uses `last_failure_at` so actively-used peers with no recent failures benefit from decay. |
| Error-string matching is fragile | `e.to_string().contains("wall-clock timeout")` | v4.1: Replaced with typed `ChunkError` enum and `matches!()` pattern. |
| Peer exclusion should distinguish timeout vs validation | Excluded peer on any `Err` uniformly | v4.1: Exclusion only on `ChunkError::TransportTimeout`; validation failures pass through AEGIS integrity handlers. |
| TIME_WAIT < 5K assertion too strong | Stated as verified postcondition | v4.1: Softened to "target under steady-state load; investigate sustained excursions." |
| `std::cmp::max` assumes `MemoryPressure` has `Ord` | Implicit assumption | v4.1: Explicit `derive(Ord)` with note that declaration order matches severity. |

---

## 9. Conclusion

The Epsilon stall was a **preventable cascade** of resource exhaustion → network degradation → sync state poisoning → permanent stall. Each individual weakness is minor; the danger is their interaction.

**Phase 1 fixes** (shipping now) are **purely defensive additions**:
- No consensus or validation changes
- No P2P protocol changes
- No database schema changes
- No changes to block production, mining, or balance logic
- All changes are in the sync-behind path (inactive at network tip)
- All changes are additive (no existing code removed)
- All changes are independently revertible
- ~140 lines of additions across 5 files

**Phase 2 fixes** (after soak testing) address deeper issues:
- Stall watchdog for auto-recovery (needs oscillation validation)
- Chunk splitting at scheduler level (needs ordering validation)
- SO_LINGER on upstream sockets (needs packet capture validation)
- Health 503 on degradation (needs swap-aware pressure validated first)

**Expected outcome after Phase 1:**
- Timeout cannot converge above 15s (ceiling)
- Stuck chunks abandoned after 120s + 60s cooldown (no infinite spin)
- Failed peers excluded per-chunk (different peer on each retry)
- Health endpoint reports swap pressure and degradation reasons
- AEGIS trust recovers via both time-decay and success-based mechanisms

**Expected outcome after Phase 2:**
- Sync stalls auto-recover within 5 minutes (watchdog) without manual restart
- TIME_WAIT further reduced by connection lifecycle improvements
- Failed chunks automatically split into smaller work items

**Estimated uptime improvement:**
- Phase 1: 99.5% → ~99.9% (prevents most stalls, still requires restart for edge cases)
- Phase 1 + Phase 2: → ~99.95% (auto-recovery for remaining edge cases)
