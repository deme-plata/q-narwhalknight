# Zero-Downtime Mining Supercluster: Technical Review

**Date:** 2026-04-04
**Author:** Claude Code (Server Beta)
**Reviewed By:** Nemotron Cascade Two (Epsilon), DeepSeek, Human Operator
**Status:** ACCEPTED WITH CONDITIONS — All P0/P1 items implemented

## Review Outcomes (2026-04-04)

All reviewer feedback has been incorporated:

- **Nemotron P0:** `is_admin_drained` flag on BackendHealth (no health mutation race). Auto-clear after 2 consecutive health check successes or 300s timeout. Safety check rejects drain if no healthy cluster peers.
- **DeepSeek P0:** Backend selector skips `is_admin_drained` backends in all paths (`next_backend`, `next_backend_excluding`, `pick_backend`).
- **Human P1:** `deploy-epsilon` command added to ha-deploy.sh with drain/deploy/recover sequence. Delta canary approach documented.

---

## 1. Problem Statement

### Observed Behavior
- Network hashrate dropped from **~1 TH/s to ~50 GH/s** (95% loss) after a deploy restart
- CPU miners vanish during the 30-90 second deploy window and **never return**
- This happens every time `ha-deploy.sh full` is executed

### Root Cause Analysis

The deploy pipeline has **three critical gaps** that compound into miner exodus:

#### Gap 1: Hard Kill Without Pre-Drain (30-60s impact)
```
ha-deploy.sh line 346:
    pgrep -f "$BINARY_NAME" | xargs -I{} kill -9 {} 2>/dev/null || true
    sleep 2
```
- `kill -9` (SIGKILL) terminates the process instantly — no graceful shutdown
- All in-flight mining submissions are dropped
- SSE connections severed without close frame
- q-flux health check takes up to **15 seconds** (5s interval x 3 failure threshold) to detect the backend is down
- During this window: miners get connection refused errors, TCP timeouts, or 502s from q-flux

#### Gap 2: q-flux Health Check Latency (15s detection delay)
```toml
# q-flux-epsilon.toml
[upstream]
health_check_interval = "5s"    # Check every 5 seconds
failure_threshold = 3            # DEFAULT (not in config, but code default)
```
- With `failure_threshold=3` and `interval=5s`, it takes **15 seconds** for q-flux to mark the local backend unhealthy
- Only THEN does it route to cluster peers (Beta at 185.182.185.227, Delta at 5.79.79.158)
- Miners experience 15 seconds of errors before failover kicks in

#### Gap 3: External Profit-Switching Miners Leave Permanently
- Auto-mine scripts (NiceHash, profit switchers) detect 15-30s of no rewards
- They switch to another chain (e.g., Monero, Kaspa) that's still responding
- **They don't come back** because Q-NarwhalKnight is a small chain — profit switchers only return if profitability exceeds threshold again
- Our own q-miner binary handles this fine (retries indefinitely), but external miners don't

### Impact Quantification
- 1 TH/s -> 50 GH/s = **950 GH/s lost** per deploy
- At current block rewards, this represents significant reduction in network security
- Recovery takes days as miners gradually rediscover the chain

---

## 2. Current Architecture

### q-flux Reverse Proxy (Already Deployed on Epsilon + Delta)

```
                    Internet (miners connect to quillon.xyz)
                              |
                    [q-flux on Epsilon :443]
                         /          \
                   [local :8080]   [cluster peers]
                   (q-api-server)   Beta:8080, Delta:8080
                                    (failover only)
```

**q-flux capabilities already implemented:**
- Super-cluster failover (upstream.rs:327-408): local -> half-open -> cluster peers -> degraded
- Health state machine (health.rs): Healthy -> Unhealthy (3 failures) -> HalfOpen (1 success) -> Healthy (2 successes)
- Inline recovery: successful request to unhealthy backend immediately marks it half-open
- DashMap-based health tracking (lock-free, 48-core safe)
- Admin API on 127.0.0.1:9090 with /health, /metrics, /status, /backends, /peers, /tls-reload

**What's missing:**
- No `/admin/drain` endpoint to pre-drain before deploy
- Health check thresholds too conservative for deploy scenarios
- No mechanism to signal miners about backup servers

### Miner-Side Failover (Already Implemented)

**Standalone q-miner** (`crates/q-miner/src/main.rs`):
- `ServerSelector` with multi-server support: `--server "host1,host2,host3"`
- Health check every 15 seconds per server
- Auto-elect primary by lowest latency
- `fetch_with_failover()` tries all healthy servers

**Slint wallet miner** (`gui/slint-wallet/src/miner.rs`):
- `FALLBACK_BOOTSTRAP_URL = "https://quillon.xyz"` hardcoded
- SSE listener with 3-failure switchover to fallback
- `submit_with_fallback()` retries on quillon.xyz

**What's missing:**
- Mining challenge response doesn't advertise backup servers
- Miners that connect directly (not through our miner binary) have no failover info

### Deploy Pipeline (ha-deploy.sh)

```
Current sequence:
1. Build binary (cargo build --release)
2. Verify on Alpha (Docker canary — non-blocking)
3. Verify on Gamma (SCP, restart, health check)
4. Promote Gamma (nginx weight=10, Beta weight=1)
5. Deploy Beta:
   a. Backup current binary
   b. kill -9 q-api-server        <-- HARD KILL, NO DRAIN
   c. sleep 2
   d. Copy new binary
   e. systemctl start q-api-server
   f. Wait for health (600s timeout, 90s stability)
6. Restore weights (Beta=10, Gamma=1)
```

**For Epsilon** (which is the primary mining endpoint via quillon.xyz):
- Epsilon is NOT in the ha-deploy.sh pipeline
- Binary is copied via SCP after Beta deploy succeeds
- Epsilon restart is manual or via separate process
- **This is the most critical gap**: Epsilon restart has NO orchestration

---

## 3. Proposed Solution

### Architecture After Changes

```
                    Internet (miners connect to quillon.xyz)
                              |
                    [q-flux on Epsilon :443]
                    /          |          \
             [local :8080]  [Beta:8080]  [Delta:8080]
             (primary)      (cluster)    (cluster)
                 |
            PRE-DRAIN before deploy
            (0ms failover to cluster)
```

### Phase 1: q-flux Admin Drain Endpoint

**New endpoint:** `POST /admin/drain`

**Behavior:**
1. Immediately marks ALL local backends as unhealthy in the DashMap
2. All subsequent requests route to cluster peers (Beta, Delta)
3. Returns JSON confirmation with peer status

**New endpoint:** `POST /admin/undrain`

**Behavior:**
1. Marks local backends as half-open (so health checker can promote them)
2. Used for manual recovery if needed

**Implementation location:** `crates/q-flux/src/admin.rs`

**Code sketch:**
```rust
// In handle_admin_request():
("POST", "/admin/drain") => handle_drain(&state, true).await,
("POST", "/admin/undrain") => handle_drain(&state, false).await,

async fn handle_drain(state: &AdminState, drain: bool) -> Response<Full<Bytes>> {
    if let Some(ref health_map) = state.health_map {
        for backend in &state.local_backends {
            if let Some(mut entry) = health_map.get_mut(backend) {
                if drain {
                    entry.is_healthy = false;
                    entry.half_open = false;
                    entry.consecutive_failures = 999; // Prevent auto-recovery
                } else {
                    entry.half_open = true;
                    entry.consecutive_failures = 0;
                }
            }
        }
    }
    // Return cluster peer health status
    json_response(200, &DrainResponse {
        drained: drain,
        cluster_peers_healthy: count_healthy_peers(state),
    })
}
```

**Risk assessment:** LOW
- Only affects routing, not data
- Admin API is localhost-only (127.0.0.1:9090)
- Undrain is automatic via health checker once backend restarts
- Cluster peers are already health-checked and ready

### Phase 2: Backup Servers in Mining Challenge Response

**Change:** Add `backup_servers` field to `MiningChallengeResponse`

```rust
// In crates/q-api-server/src/handlers.rs
pub struct MiningChallengeResponse {
    // ... existing fields ...
    
    /// v1.0.5: Backup mining servers for failover during rolling deploys
    /// Miners should try these servers if primary becomes unresponsive
    #[serde(skip_serializing_if = "Option::is_none")]
    pub backup_servers: Option<Vec<String>>,
}
```

**Server returns:** `["https://dl.quillon.xyz", "http://185.182.185.227:8080"]`

**Miner behavior:** On connection failure, try backup servers before giving up

**Risk assessment:** LOW
- Optional field, old miners ignore it
- New miners get better failover
- No consensus impact

### Phase 3: Tuned Health Check Thresholds

**Change q-flux config for deploy-optimized failover:**

```toml
# q-flux-epsilon.toml — PROPOSED CHANGES
[upstream]
health_check_interval = "2s"    # Was: 5s (faster detection)
health_check_timeout = "2s"     # Was: 3s (faster timeout)
failure_threshold = 1            # Was: 3 (immediate failover on first failure)
healthy_threshold = 1            # Was: 2 (faster recovery)

[cluster]
health_check_interval = "5s"    # Was: 10s (faster cluster peer checks)
health_check_timeout = "3s"     # Was: 5s
```

**Effect:** Failover detection drops from **15 seconds to 2 seconds**

**Risk assessment:** MEDIUM
- `failure_threshold=1` means a single health check timeout triggers failover
- Could cause unnecessary failover on transient network blips
- Mitigation: local health check is TCP+HTTP to localhost — virtually never has transient failures
- For cluster peers, keep `failure_threshold=2` (cross-network is less reliable)

### Phase 4: Deploy Script Pre-Drain Integration

**Change ha-deploy.sh to drain before killing:**

```bash
# BEFORE (current):
pgrep -f "$BINARY_NAME" | xargs -I{} kill -9 {} 2>/dev/null || true
sleep 2

# AFTER (proposed):
# 1. Pre-drain: tell q-flux to route away from local backend
curl -s -X POST http://127.0.0.1:9090/admin/drain || true
sleep 3  # Allow in-flight requests to complete

# 2. Now safe to kill — no traffic reaching this backend
pgrep -f "$BINARY_NAME" | xargs -I{} kill -9 {} 2>/dev/null || true
sleep 1

# 3. Deploy new binary
cp "$BINARY_PATH" "$BETA_BINARY"
chmod +x "$BETA_BINARY"

# 4. Start new binary
systemctl start "$SERVICE_NAME"

# 5. Wait for health — q-flux auto-recovers when health check passes
# (undrain is automatic — health checker promotes half-open -> healthy)
```

**Risk assessment:** LOW
- `curl || true` means drain failure doesn't block deploy
- 3-second drain window allows in-flight completions
- q-flux auto-recovers via health checker (no manual undrain needed)

---

## 4. Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| Miner-visible downtime during deploy | 15-60 seconds | **0 seconds** |
| Health check failover time | 15 seconds | 2 seconds |
| Pre-drain time | N/A (hard kill) | 3 seconds (graceful) |
| Miner hashrate loss per deploy | ~95% (1 TH/s -> 50 GH/s) | **~0%** |
| Time to full recovery | Days | Instant |
| SSE reconnections during deploy | All miners | **0** (q-flux routes to peers) |

---

## 5. Implementation Order and Dependencies

```
Phase 1 (q-flux drain) ──────────────> Phase 4 (ha-deploy.sh integration)
                                              |
Phase 2 (backup_servers) ─── independent ─────┤
                                              |
Phase 3 (health thresholds) ─ independent ────┘
                                              |
                                              v
                                     TESTING & DEPLOY
```

**Phase 1 MUST come before Phase 4** (deploy script needs drain endpoint)
**Phases 2 and 3 are independent** and can be done in parallel

---

## 6. Testing Plan

### Unit Tests
- Drain endpoint marks all local backends unhealthy
- Undrain endpoint marks local backends half-open
- Health checker auto-recovers drained backend after restart
- `backup_servers` field serializes correctly in challenge response

### Integration Tests
1. Start q-flux with 1 local backend + 2 cluster peers
2. Call `/admin/drain` — verify all requests route to cluster peers
3. Restart local backend — verify auto-recovery within 2 health check intervals
4. Measure zero-downtime: no 5xx errors during drain->restart->recover cycle

### Production Validation
1. Deploy to Epsilon with new drain sequence
2. Monitor q-flux logs for "Super-cluster failover" messages
3. Monitor mining stats: hashrate should remain constant during deploy
4. Check SSE streams: no disconnections visible to miners

---

## 7. Questions for Reviewers

### For Nemotron Cascade Two (Epsilon):
1. Do you see any race conditions in the drain -> kill -> restart -> auto-recover sequence?
2. Should we add a "drain timeout" that auto-undrains after N seconds as a safety valve?
3. Is `failure_threshold=1` too aggressive for the local health check? Could a GC pause or RocksDB compaction cause a false positive?

### For DeepSeek:
1. Are there better algorithms for miner-side server selection when multiple servers are available?
2. Should the `backup_servers` field include latency hints or weights?
3. Any concerns about the DashMap concurrent access pattern for the drain flag?

### For Human Operator:
1. Should we deploy q-flux changes to Delta first as canary before Epsilon?
2. Do we need a separate "Epsilon deploy" section in ha-deploy.sh?
3. Timeline: Deploy this before or after the VDF mining upgrade?

---

## 8. Files to Modify

| File | Change | Risk |
|------|--------|------|
| `crates/q-flux/src/admin.rs` | Add `/admin/drain` and `/admin/undrain` endpoints | LOW |
| `crates/q-flux/src/health.rs` | Add `is_drained` flag to BackendHealth | LOW |
| `crates/q-flux/q-flux-epsilon.toml` | Tune health check thresholds | MEDIUM |
| `crates/q-flux/q-flux-delta.toml` | Same threshold tuning | MEDIUM |
| `crates/q-api-server/src/handlers.rs` | Add `backup_servers` to MiningChallengeResponse | LOW |
| `scripts/ha-deploy.sh` | Add pre-drain curl before kill | LOW |

---

## 9. Appendix: Current Config Files

### q-flux-epsilon.toml (current)
```toml
[upstream]
backends = ["127.0.0.1:8080"]
health_check_interval = "5s"
health_check_path = "/api/v1/status"

[cluster]
peers = ["185.182.185.227:8080", "5.79.79.158:8080"]
health_check_interval = "10s"
```

### Key Code Locations
- q-flux admin server: `crates/q-flux/src/admin.rs` (lines 44-160)
- Health map type: `crates/q-flux/src/health.rs:51` — `Arc<DashMap<String, BackendHealth>>`
- Backend selection: `crates/q-flux/src/upstream.rs:335-408`
- Deploy kill: `scripts/ha-deploy.sh:346`
- Miner ServerSelector: `crates/q-miner/src/main.rs:384-530`
- Challenge response: `crates/q-api-server/src/handlers.rs:9572-9600`
