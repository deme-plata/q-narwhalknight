# MINING STALL - COMPREHENSIVE ACTION PLAN v1.0.5-beta

**Date**: 2025-11-12 17:15 CET (16:15 UTC)
**Status**: ACTIONABLE ROADMAP
**Version**: v1.0.5-beta (Next Release)
**Based On**: External AI Expert Reviews (KIMIAI + ChatGPT)

---

## EXECUTIVE SUMMARY

**Current Status**:
- v1.0.4-beta deployed with Phase 0 challenge caching (WORKING)
- Root cause identified: NO ACTIVE MINERS on network
- Network health: 0 peers, 0 external miners, solo bootstrap node
- Stall frequency: Every 13-30 minutes (internal miner exhaustion)

**Expert Consensus**:
Both external AI reviews identified critical architectural gaps beyond the "no miners" symptom:
1. Challenge generation not consensus-bound (vulnerable to precompute/divergence)
2. No P2P peer discovery working (0 peers = bootstrap failure)
3. No mining telemetry or health metrics
4. No solution deduplication or miner identity tracking
5. Bootstrap node as single point of failure
6. No emergency fallback mechanisms

---

## PRIORITY BREAKDOWN

### 🔴 CRITICAL (Deploy in v1.0.5-beta - Next 48 Hours)

**1. Consensus-Bound Challenge Generation**
- **Current**: `blake3::hash("block_{height}_time_{timestamp}")`
- **Issue**: Non-deterministic, not cryptographically bound to blockchain state
- **Fix**: Bind challenge to (`prev_hash`, `height`, `difficulty`, `vdf_iters`, `lane_id`, `version`)
- **File**: `crates/q-api-server/src/handlers.rs:4442`
- **Impact**: Prevents cross-lane solution reuse, precompute attacks, challenge divergence

**Implementation**:
```rust
// Consensus-bound challenge input
struct ChallengeInput {
    version: &'static [u8],    // b"QNK/1"
    parent_hash: [u8; 32],     // Previous block hash
    height: u64,
    difficulty: [u8; 32],
    vdf_iters: u32,
    lane_id: u8,               // 0-7 for 8 producers
}

fn make_challenge(input: &ChallengeInput) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(input.version);
    h.update(&input.parent_hash);
    h.update(&input.height.to_le_bytes());
    h.update(&input.difficulty);
    h.update(&input.vdf_iters.to_le_bytes());
    h.update(&[input.lane_id]);
    *h.finalize().as_bytes()
}
```

**Cache Key**: Change from `(height)` to `(height, parent_hash)` to handle reorgs correctly

---

**2. Fix P2P Bootstrap - "0 Peers" is CATASTROPHIC**
- **Current**: Bootstrap node in complete network isolation
- **Issue**: No peer discovery, no external connectivity, single point of failure
- **Files**: `crates/q-network/src/unified_network_manager.rs` (libp2p config)
- **Impact**: WITHOUT peers, external miners can't connect even if they exist

**Diagnostic Checklist**:
```bash
# 1. Verify port exposure
ss -tlnp | grep 9001  # Should show LISTEN on 0.0.0.0:9001

# 2. External reachability test
nc -vz 185.182.185.227 9001  # From remote host

# 3. Check libp2p config
journalctl -u q-api-server | grep -E "(libp2p|peer_id|listen_addr|bootstrap)"

# 4. Verify bootstrap peer list
# Should include self-bootstrap for testing

# 5. Check multiaddr publication
curl -s https://quillon.xyz/api/v1/network/info | jq '.data.listen_addresses'
```

**Required Fixes**:
- Expose port 9001 TCP/UDP on firewall
- Configure external IP in libp2p multiaddr
- Add DNS-based peer discovery
- Implement peer persistence (save on shutdown, load on start)
- Add AutoNAT for NAT traversal
- Log peer connection/disconnection events

---

**3. Solution Deduplication by (height, lane_id, solution_hash)**
- **Current**: No deduplication, solutions can be double-applied
- **Issue**: Retries and network latency can cause duplicate solution submission
- **File**: `crates/q-api-server/src/handlers.rs` (`submit_solution`)
- **Impact**: Prevents accidental double-spending of solutions

**Implementation**:
```rust
pub struct SolutionDedupCache {
    // Key: (height, lane_id, solution_hash)
    cache: Arc<RwLock<HashMap<(u64, u8, [u8; 32]), chrono::DateTime<Utc>>>>,
}

impl SolutionDedupCache {
    pub fn insert(&self, height: u64, lane_id: u8, hash: [u8; 32]) -> bool {
        let mut cache = self.cache.write();

        // Clean expired entries (>5 minutes old)
        let now = chrono::Utc::now();
        cache.retain(|_, timestamp| now.signed_duration_since(*timestamp).num_seconds() < 300);

        // Check if already exists
        if cache.contains_key(&(height, lane_id, hash)) {
            return false; // Duplicate
        }

        cache.insert((height, lane_id, hash), now);
        true // New solution
    }
}
```

---

**4. Prometheus Metrics - CRITICAL OBSERVABILITY**
- **Current**: No mining health metrics, blind debugging via journalctl
- **Issue**: Can't measure network hashrate, solution rate, miner activity
- **File**: New `crates/q-api-server/src/metrics.rs`
- **Impact**: Enables data-driven debugging and alerting

**Metrics to Add**:
```rust
// Mining metrics
pub mining_solutions_total: IntCounterVec,           // Labels: miner_id, lane_id
pub mining_solutions_rate: Gauge,                   // Solutions/second
pub mining_challenge_age_seconds: Gauge,             // Time since last challenge update
pub mining_challenge_requests_total: IntCounter,     // API call count
pub mining_cache_hits_total: IntCounter,             // Cache hit count
pub mining_cache_misses_total: IntCounter,           // Cache miss count

// Network metrics
pub p2p_peers_gauge: Gauge,                          // Current peer count
pub p2p_peer_connections_total: IntCounterVec,       // Labels: peer_id, event (connected/disconnected)

// Block production metrics
pub block_production_seconds: Histogram,              // Block time distribution
pub blocks_minted_total: IntCounterVec,               // Labels: lane_id
pub internal_miner_queue_depth: Gauge,                // Remaining internal capacity

// Health metrics
pub time_since_last_solution: Gauge,                  // Seconds since last solution
pub time_since_last_block: Gauge,                     // Seconds since last block
```

**Alerting Rules**:
```yaml
- alert: MiningStalled
  expr: time_since_last_solution > 300  # 5 minutes
  severity: critical

- alert: NoPeers
  expr: p2p_peers_gauge < 1
  severity: warning

- alert: ChallengeStale
  expr: mining_challenge_age_seconds > 120
  severity: warning
```

---

### 🟡 HIGH (Deploy in v1.0.6-beta - Next 7 Days)

**5. Emergency Internal Miner Refill**
- **Current**: Internal miners exhaust after 13-30 min, blockchain halts
- **Issue**: No fallback mechanism when external miners unavailable
- **File**: `crates/q-api-server/src/main.rs` (watchdog loop)
- **Impact**: Provides graceful degradation instead of hard stall

**Implementation**:
```rust
// Watchdog loop (run every 60 seconds)
loop {
    tokio::time::sleep(Duration::seconds(60)).await;

    let time_since_solution = (Utc::now() - last_solution_time).num_seconds();

    if time_since_solution > 300 {  // 5 minutes without solution
        warn!("⚠️ No mining solutions for {} seconds - activating emergency internal miners", time_since_solution);

        // Activate emergency mining (devnet only)
        #[cfg(feature = "devnet")]
        {
            internal_miner_pool.enable_emergency_mode(duration: 600).await;  // 10 min burst
        }

        #[cfg(not(feature = "devnet"))]
        {
            error!("🚨 CRITICAL: No mining solutions for {} seconds on mainnet - manual intervention required", time_since_solution);
        }
    }
}
```

**Design Principles**:
- Emergency mode ONLY on devnet (feature flag gated)
- Temporary boost (10-minute burst, then wait again)
- Logged loudly for diagnostics
- Does NOT hide the underlying "no miners" problem

---

**6. Miner Identity & Telemetry**
- **Current**: Solutions accepted anonymously, no tracking
- **Issue**: Can't identify which miners are active, measure their hashrate, or debug issues
- **File**: `crates/q-types/src/lib.rs` (MiningSolution struct)
- **Impact**: Enables miner troubleshooting and reward attribution

**Extended Solution Format**:
```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MiningSolution {
    pub miner_id: String,              // Public key or deterministic ID
    pub signature: String,              // Sign(miner_privkey, challenge_hash || nonce || height)
    pub challenge_hash: String,
    pub nonce: String,
    pub solution_hash: String,
    pub timestamp: i64,
    pub miner_version: String,          // e.g., "q-miner/v1.0.5-beta"
    pub submitted_at: DateTime<Utc>,
}
```

**Verification**:
```rust
// In submit_solution handler
fn verify_miner_signature(solution: &MiningSolution) -> Result<bool> {
    let message = format!("{}{}{}", solution.challenge_hash, solution.nonce, solution.height);
    let signature_bytes = hex::decode(&solution.signature)?;
    let pubkey = PublicKey::from_miner_id(&solution.miner_id)?;

    Ok(pubkey.verify(message.as_bytes(), &signature_bytes))
}
```

**Telemetry Tracking**:
```rust
pub struct MinerStats {
    pub miner_id: String,
    pub last_seen: DateTime<Utc>,
    pub solutions_submitted: u64,
    pub solutions_accepted: u64,
    pub estimated_hashrate: f64,       // Solutions/second * difficulty
    pub version: String,
}

// Update on every submission
miner_stats.insert(solution.miner_id, MinerStats { /* ... */ });
```

---

**7. Network Health Dashboard**
- **Current**: No visual monitoring, CLI-only debugging
- **Issue**: Slow incident response, no real-time visibility
- **File**: New `crates/q-api-server/src/handlers/dashboard.rs`
- **Impact**: Enables proactive monitoring instead of reactive debugging

**Dashboard Endpoint**: `GET /api/v1/dashboard`

**Response**:
```json
{
  "network_status": {
    "peers": 0,
    "status": "CRITICAL",
    "message": "No P2P peers connected - network isolated"
  },
  "mining_status": {
    "active_miners": 0,
    "last_solution": "2025-11-12T16:00:00Z",
    "seconds_since_solution": 900,
    "status": "CRITICAL",
    "message": "No solutions for 15 minutes - network stalled"
  },
  "block_production": {
    "current_height": 45292,
    "blocks_per_second": 0.0,
    "last_block": "2025-11-12T16:00:00Z",
    "status": "WARNING",
    "internal_miner_buffer": "12%",
    "projected_stall": "18 minutes"
  },
  "recommended_actions": [
    "Deploy external miners immediately",
    "Fix P2P bootstrap (0 peers critical)",
    "Verify port 9001 TCP/UDP open"
  ]
}
```

**Web UI**: Simple React dashboard at `https://quillon.xyz/dashboard`

---

### 🟢 MEDIUM (Deploy in v1.0.7-beta - Weeks 2-4)

**8. Dynamic Difficulty Adjustment**
- **Current**: Static `0000ffff...` difficulty (2 leading zero bytes)
- **Issue**: No adjustment as network grows/shrinks
- **File**: `crates/q-mining/src/difficulty.rs`
- **Impact**: Keeps block time stable as hashrate changes

**Algorithm**:
```rust
pub fn calculate_next_difficulty(
    current_difficulty: [u8; 32],
    target_block_time: u64,  // e.g., 60 seconds
    actual_block_times: &[u64],  // Last N blocks
) -> [u8; 32] {
    let avg_block_time = actual_block_times.iter().sum::<u64>() / actual_block_times.len() as u64;

    // Adjustment factor (max ±25% per window)
    let adjustment = if avg_block_time > target_block_time {
        // Blocks too slow → reduce difficulty
        (target_block_time as f64 / avg_block_time as f64).clamp(0.75, 1.0)
    } else {
        // Blocks too fast → increase difficulty
        (target_block_time as f64 / avg_block_time as f64).clamp(1.0, 1.25)
    };

    adjust_difficulty(current_difficulty, adjustment)
}
```

**Window**: 100 blocks (~100 minutes at 1 block/min target)

---

**9. Slot-Based Deterministic Challenges (Phase 1)**
- **Current**: Single global challenge for all lanes
- **Issue**: Solution contention between lanes, no fairness guarantees
- **File**: `crates/q-api-server/src/handlers.rs` + `crates/q-mining/src/slots.rs`
- **Impact**: Fair mining, prevents solution reuse, enables work-weighted rewards

**Design**:
```rust
pub struct SlotChallenge {
    pub slot_id: u64,              // Time-based or round-based
    pub lane_id: u8,               // 0-7
    pub challenge_hash: [u8; 32],  // Unique per (slot, lane)
    pub reserved_for: Option<String>, // Miner who solved it
    pub expires_at: DateTime<Utc>,
}

// Generate per-lane challenges
fn generate_slot_challenges(
    slot_id: u64,
    parent_hash: [u8; 32],
    height: u64,
) -> Vec<SlotChallenge> {
    (0..8).map(|lane_id| {
        SlotChallenge {
            slot_id,
            lane_id,
            challenge_hash: make_challenge(&ChallengeInput {
                version: b"QNK/1",
                parent_hash,
                height,
                difficulty,
                vdf_iters,
                lane_id, // ← Unique per lane
            }),
            reserved_for: None,
            expires_at: Utc::now() + Duration::seconds(120),
        }
    }).collect()
}
```

**Solution Reservation**:
- When miner solves challenge for lane X, mark it reserved
- Other lanes ignore that solution
- Prevents cross-lane solution reuse

---

**10. "Hello Miner" Distribution Package**
- **Current**: No easy way for external miners to connect
- **Issue**: High barrier to entry, no documentation
- **File**: New `hello-miner/` binary + README
- **Impact**: Enables community mining participation

**Package Contents**:
```
q-hello-miner-v1.0.5.tar.gz
├── q-miner (binary)
├── config.toml (pre-configured for quillon.xyz)
├── start-miner.sh
└── README.md
```

**README.md**:
```markdown
# Quillon Testnet Miner - Start Mining in 60 Seconds

## Quick Start
```bash
chmod +x start-miner.sh
./start-miner.sh
```

## Configuration
Edit `config.toml`:
```toml
[network]
bootstrap_node = "185.182.185.227:9001"
api_endpoint = "https://quillon.xyz"

[mining]
threads = 4  # CPU threads to use
wallet = "qnk..."  # Your wallet address

[telemetry]
enabled = true
report_interval = 60  # seconds
```

## System Requirements
- CPU: 2+ cores
- RAM: 512 MB
- Network: Open port 30333 (optional but recommended)

## Expected Performance
- CPU Mining: ~1-10 solutions/hour (varies by difficulty)
- Reward per solution: ~1.33 QNK (current rate)

## Troubleshooting
Check logs: `./q-miner --check-connectivity`
View stats: `curl https://quillon.xyz/api/v1/mining/stats`
```

---

## DEPLOYMENT SEQUENCE

### v1.0.5-beta (Emergency Release - 48 Hours)
**Goal**: Fix critical consensus/P2P issues

1. Consensus-bound challenge generation ✅
2. Solution deduplication ✅
3. P2P bootstrap fix ✅
4. Prometheus metrics ✅
5. Compile with 10-hour timeout
6. Deploy to production
7. Monitor for 24 hours

**Success Criteria**:
- ✅ Peers > 0
- ✅ Challenge hash cryptographically bound
- ✅ No duplicate solutions accepted
- ✅ Metrics dashboard available

---

### v1.0.6-beta (Stability Release - 7 Days)
**Goal**: Add emergency fallbacks and telemetry

1. Emergency internal miner refill (devnet only)
2. Miner identity & signature verification
3. Telemetry tracking
4. Network health dashboard
5. Miner stats API

**Success Criteria**:
- ✅ Stalls self-recover on devnet
- ✅ Miner activity visible in dashboard
- ✅ Per-miner hashrate tracked
- ✅ Alerting functional

---

### v1.0.7-beta (Feature Release - Weeks 2-4)
**Goal**: Production-ready mining ecosystem

1. Dynamic difficulty adjustment
2. Slot-based challenges (Phase 1)
3. Solution reservation
4. Hello-miner distribution package
5. Mining rewards transparency

**Success Criteria**:
- ✅ External miners actively mining
- ✅ Difficulty adjusts with network growth
- ✅ Fair work-weighted rewards
- ✅ Community participation

---

## TESTING PROTOCOL

### Pre-Deployment
```bash
# 1. Consensus-bound challenge test
# Generate 2 challenges for same height, different parent_hash
# Verify they produce DIFFERENT hashes

# 2. P2P connectivity test
nc -vz 185.182.185.227 9001  # From remote host
curl -s https://quillon.xyz/api/v1/network/info

# 3. Solution deduplication test
# Submit same solution twice
# Verify second submission rejected

# 4. Metrics endpoint test
curl -s https://quillon.xyz/metrics | grep mining_
```

### Post-Deployment
```bash
# 5. Monitor for 4 hours
watch -n 60 'curl -s https://quillon.xyz/api/v1/dashboard | jq .'

# 6. Verify no stalls
journalctl -u q-api-server -f | grep -E "(STALLED|Mining still stalled)"

# 7. Check peer count
watch -n 10 'curl -s https://quillon.xyz/metrics | grep p2p_peers_gauge'

# 8. Validate metrics
curl -s https://quillon.xyz/metrics | grep -E "(mining_solutions_total|time_since_last_solution)"
```

---

## RISK ASSESSMENT

### v1.0.5-beta Risks

**HIGH RISK**: Consensus-bound challenge changes mining protocol
- **Mitigation**: Deploy on devnet first, test with canary miner
- **Rollback**: Revert to v1.0.4-beta if miners reject new challenges

**MEDIUM RISK**: P2P config changes may break existing connections
- **Mitigation**: Gradual rollout, keep old bootstrap nodes
- **Rollback**: Keep v1.0.4 running in parallel during transition

**LOW RISK**: Metrics overhead
- **Mitigation**: Async metric updates, no blocking I/O
- **Rollback**: Feature flag to disable metrics

---

## SUCCESS METRICS

### Network Health (Target: v1.0.7-beta)
| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Peers | 0 | 3-5 | ❌ CRITICAL |
| Active Miners | 0 | 5-10 | ❌ CRITICAL |
| Solutions/Hour | 0 | 100+ | ❌ CRITICAL |
| Stall Frequency | 13-30 min | <1 per week | ❌ CRITICAL |
| Max Stall Duration | 242 min | <5 min | ❌ CRITICAL |

### Code Quality
| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Challenge Consistency | ✅ | ✅ | ✅ DONE |
| Consensus Binding | ❌ | ✅ | 🟡 PLANNED |
| Solution Dedup | ❌ | ✅ | 🟡 PLANNED |
| Metrics Coverage | 0% | 80% | 🟡 PLANNED |
| P2P Connectivity | ❌ | ✅ | 🟡 PLANNED |

---

## FILES TO MODIFY

### v1.0.5-beta
```
crates/q-api-server/src/handlers.rs          (consensus-bound challenges + dedup)
crates/q-api-server/src/lib.rs               (add dedup cache to AppState)
crates/q-api-server/src/metrics.rs           (NEW - Prometheus metrics)
crates/q-network/src/unified_network_manager.rs  (P2P bootstrap fix)
crates/q-types/src/lib.rs                    (extend MiningSolution struct)
```

### v1.0.6-beta
```
crates/q-api-server/src/main.rs              (emergency miner watchdog)
crates/q-api-server/src/handlers/dashboard.rs  (NEW - health dashboard)
crates/q-api-server/src/telemetry.rs         (NEW - miner stats tracking)
```

### v1.0.7-beta
```
crates/q-mining/src/difficulty.rs            (dynamic adjustment)
crates/q-mining/src/slots.rs                 (NEW - slot-based challenges)
hello-miner/                                 (NEW - miner distribution package)
```

---

## DEPENDENCIES

### Rust Crates (Add to Cargo.toml)
```toml
[dependencies]
prometheus = "0.13"           # Metrics
lazy_static = "1.4"           # Static metrics registry
ed25519-dalek = "2.0"         # Miner signature verification
```

### Infrastructure
- Prometheus server (for metrics collection)
- Grafana (for dashboard visualization)
- Port 9001 open on firewall (P2P)
- Port 9090 open for Prometheus scraping

---

## REFERENCES

### Expert Reviews
- `MINING_STALL_ROOT_CAUSE_TECHNICAL_REVIEW.md` - This session's diagnosis
- KIMIAI Analysis (2025-11-12) - Comprehensive architectural review
- ChatGPT Technical Review (2025-11-12) - Consensus-binding guidance

### Related Documents
- `PHASE0_EMERGENCY_FIX_DEPLOYED_v1.0.4-beta.md` - Challenge caching implementation
- `SECOND_STALL_INCIDENT_13_MINUTES.md` - Recurring stall pattern
- `SERVICE_RESTART_SUCCESS_BUT_PHASE0_REQUIRED.md` - Root cause analysis

---

## CONCLUSION

**Key Insight**: Phase 0 fixed the symptom (challenge consistency), but the disease (no miners + architectural gaps) remains.

**v1.0.5-beta Priority**: Fix consensus binding + P2P + metrics FIRST. Miners can't connect if P2P is broken, and we can't debug without metrics.

**Long-term Vision**: Transform from "solo bootstrap node" to "distributed mining network" with:
- Real P2P connectivity
- External miner participation
- Consensus-safe challenge generation
- Fair reward distribution
- Self-healing fallbacks

**Timeline**:
- v1.0.5: 48 hours (critical fixes)
- v1.0.6: 7 days (stability)
- v1.0.7: 2-4 weeks (production-ready)

---

**Prepared By**: Server Beta (Claude Code)
**Report Date**: 2025-11-12 17:15 CET (16:15 UTC)
**Status**: 📋 **ACTION PLAN READY** - Begin v1.0.5-beta implementation
**Next Step**: Implement consensus-bound challenge generation
**Confidence**: **95%** - Expert-validated roadmap with clear priorities
