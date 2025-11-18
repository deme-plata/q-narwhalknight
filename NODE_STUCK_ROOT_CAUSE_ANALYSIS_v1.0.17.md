# Node Stuck & Network Isolation Root Cause Analysis v1.0.17-beta

**Date**: 2025-11-18
**Version**: v1.0.17-beta
**Critical Severity**: P0 - Network Split / Consensus Failure

---

## Executive Summary

Two critical issues identified:

1. **Server Beta (185.182.185.227)**: Node gets stuck in deadlock/busy loop, stops producing blocks
2. **Server Alpha (Docker nodes)**: Nodes create own blockchain branch instead of syncing with network

Both issues stem from **async lock acquisition deadlocks** and **libp2p connection failures**.

---

## Issue #1: Node Stuck at Height (Server Beta)

### Symptoms
- Block production stops completely
- Process shows 115% CPU usage (busy loop)
- HTTP API remains responsive
- Mining submissions still accepted
- `produce_blocks()` function never called

### Timeline (2025-11-18)
```
11:00:23 - Last blocks produced at height 14286
11:00:30 - Block production loop stops executing
12:00:00 - Node still stuck, CPU at 115%
12:05:16 - Service restart attempted (takes 5+ minutes to stop)
12:10:33 - Service forcefully restarted, recovery at height 14301
```

### Root Cause
**Deadlock in `discovery.lock().await`** - Fixed in v1.0.17-beta but likely exists in ANOTHER location.

Location (already fixed): `crates/q-api-server/src/main.rs:5761`

```rust
// ❌ WRONG (v1.0.16-beta) - Blocks main thread
let discovered_peers_arc = if let Some(ref discovery) = app_state.libp2p_discovery {
    let manager = discovery.lock().await;  // DEADLOCK!
    Some(manager.get_discovered_peers_arc())
} else {
    None
};

tokio::spawn(async move {
    info!("🔄 Starting OPTIMIZED active block sync loop...");
    // ...
});

// ✅ FIXED (v1.0.17-beta) - Lock inside spawn
tokio::spawn(async move {
    info!("🔄 Starting OPTIMIZED active block sync loop...");

    let discovered_peers_arc = if let Some(ref discovery) = discovery_clone {
        let manager = discovery.lock().await;  // Non-blocking to main
        Some(manager.get_discovered_peers_arc())
    } else {
        None
    };
    // ...
});
```

### Evidence
```
# Service logs show block production stopped
12:00:23 - ✅ produce_blocks() completed in 2ms, produced 8 blocks
12:00:25 - [Last sync messages at height 14286]
12:00:30+ - NO MORE produce_blocks() calls
12:09:51 - ERROR: IMMEDIATE ACTION REQUIRED: Service needs restart
```

### Impact
- ✅ HTTP server starts successfully (v1.0.17-beta fix)
- ❌ Node still gets stuck after ~15-20 minutes of operation
- ❌ Requires manual service restart to recover
- ❌ Blocks lost during downtime

---

## Issue #2: Network Isolation - Nodes Create Own Chain (Server Alpha)

### Symptoms
- Bootstrap discovery SUCCEEDS (finds 2 peers via HTTP)
- libp2p connections FAIL completely
- Node produces own blockchain branch
- Sync rate: ~60 blocks/minute (local production, not network sync)
- AutoNAT errors: "NoAddresses", "NoServer"
- All gossipsub publish attempts fail: "InsufficientPeers"

### Evidence from Server Alpha Logs

**1. Bootstrap Discovery Success:**
```
✅ Discovered 2 bootstrap peer(s) automatically
📍 Added testnet-phase12 bootstrap peer: 12D3KooWC688bzHi7djbkensGQMABzX9tY41LNasgd3g3FdwqQn7
```

**2. Network Isolation:**
```
⚠️  [P2P HEALTH] NO CONNECTIONS - Network isolated!
AutoNAT event: OutboundProbe(Error { probe_id: ProbeId(0), peer: None, error: NoAddresses })
AutoNAT event: OutboundProbe(Error { probe_id: ProbeId(1), peer: None, error: NoServer })
```

**3. Publishing Failures:**
```
❌ Failed to publish block 1 to topic /qnk/testnet-phase12/blocks: InsufficientPeers
❌ Failed to publish AI message to topic qnk/ai/heartbeat/v1: InsufficientPeers
```

**4. Single Validator Mode:**
```
✅ Single validator mode (default)
Minimum peers: 1
✅ LOCK-FREE Block Production initialized with 8 producers
```

### Root Cause

**Design Flaw**: Single validator mode allows block production WITHOUT peer connections.

**Architecture Issue**:
1. HTTP bootstrap discovery works (finds peers via REST API)
2. libp2p connection establishment FAILS (AutoNAT/NAT traversal issues)
3. Block production continues despite network isolation
4. Node builds own chain thinking it's the only validator

**Why libp2p Connections Fail:**

1. **Docker Networking**: Container may not have proper port forwarding
2. **Firewall Rules**: libp2p ports (9001/tcp) may be blocked
3. **NAT Traversal**: AutoNAT requires both nodes to support hole punching
4. **Multiaddress Issues**: Peer multiaddrs may not be reachable from Docker
5. **Protocol Mismatch**: Bootstrap peer may be using different libp2p protocols

### Network Architecture Comparison

```
┌─────────────────────────────────────────────────────────────┐
│ CURRENT (BROKEN) - Split Brain Scenario                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Server Beta (185.182.185.227)                              │
│  ├─ Height: 14,286+                                         │
│  ├─ Peers: 0                                                │
│  └─ Chain: Main                                             │
│                                                              │
│  Server Alpha (Docker)                                      │
│  ├─ Height: 1→500 (own chain!)                             │
│  ├─ Peers: 0 (despite finding bootstrap)                   │
│  └─ Chain: Isolated branch                                 │
│                                                              │
│  Result: Two separate blockchains! ❌                       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ EXPECTED (CORRECT) - Synchronized Network                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Server Beta ◄──────libp2p──────► Server Alpha              │
│  ├─ Height: 14,286              ├─ Height: 14,286           │
│  ├─ Peers: 1+                   ├─ Peers: 1+                │
│  └─ Chain: Main                 └─ Chain: Main (synced)     │
│                                                              │
│  Result: Consensus achieved! ✅                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Critical Fixes Required

### Fix #1: Prevent Block Production Until Peer Connections Established

**Location**: `crates/q-api-server/src/main.rs:5200+`

```rust
// Current (WRONG):
if network_height > 0 {
    let gap = network_height.saturating_sub(current_height);
    if gap > CATCHUP_DISABLE_THRESHOLD {
        // Only checks if already syncing
    }
}

// Should be (CORRECT):
let connected_peers = app_state.libp2p_manager
    .map(|m| m.connected_peers_count())
    .unwrap_or(0);

if connected_peers == 0 {
    if loop_iteration % 30 == 0 {
        warn!("🚫 [NETWORK ISOLATION] Block production DISABLED: No peer connections");
        warn!("   Waiting for libp2p connections before producing blocks");
        warn!("   This prevents creating an isolated blockchain branch");
    }
    continue;  // Skip block production
}
```

### Fix #2: Debug libp2p Connection Failures

**Add Enhanced Logging**:

```rust
// In unified_network_manager.rs
impl UnifiedNetworkManager {
    pub async fn run(&mut self) {
        loop {
            match self.swarm.select_next_some().await {
                SwarmEvent::NewListenAddr { address, .. } => {
                    info!("🎧 [LIBP2P] Listening on: {}", address);
                }
                SwarmEvent::Dialing { peer_id, connection_id } => {
                    info!("📞 [LIBP2P] Dialing peer: {:?} (conn: {:?})", peer_id, connection_id);
                }
                SwarmEvent::ConnectionEstablished { peer_id, endpoint, .. } => {
                    info!("✅ [LIBP2P] Connection established with: {}", peer_id);
                    info!("   Endpoint: {:?}", endpoint);
                }
                SwarmEvent::OutgoingConnectionError { peer_id, error, .. } => {
                    error!("❌ [LIBP2P] Outgoing connection failed:");
                    error!("   Peer: {:?}", peer_id);
                    error!("   Error: {:?}", error);
                    error!("   DIAGNOSTIC: This is why nodes can't connect!");
                }
                _ => {}
            }
        }
    }
}
```

### Fix #3: Docker Network Configuration

**Ensure proper port forwarding**:

```bash
# In docker-compose.yml or docker run
ports:
  - "8080:8080"   # HTTP API
  - "9001:9001"   # libp2p P2P

# Environment
environment:
  - Q_P2P_PORT=9001
  - Q_EXTERNAL_ADDRESS=/ip4/<HOST_IP>/tcp/9001  # CRITICAL!
```

### Fix #4: Bootstrap Peer Validation

```rust
// Validate bootstrap peer is actually reachable
async fn validate_bootstrap_peer(peer_addr: &str) -> Result<bool> {
    // Try TCP connection first
    match TcpStream::connect(peer_addr).await {
        Ok(_) => {
            info!("✅ Bootstrap peer TCP reachable: {}", peer_addr);
            Ok(true)
        }
        Err(e) => {
            error!("❌ Bootstrap peer TCP unreachable: {}", peer_addr);
            error!("   Error: {}", e);
            Ok(false)
        }
    }
}
```

---

## Immediate Actions Required

### Priority 0 (Critical - Must Fix Now):
1. ✅ Fix HTTP server deadlock (DONE in v1.0.17-beta)
2. ⏳ Find and fix remaining `.lock().await` deadlocks
3. ⏳ Add peer connection requirement before block production
4. ⏳ Fix libp2p connection establishment

### Priority 1 (High - Fix Today):
1. Add comprehensive libp2p connection diagnostics
2. Fix Docker network configuration for proper P2P
3. Implement connection health monitoring
4. Add automatic service restart on deadlock detection

### Priority 2 (Medium - Fix This Week):
1. Implement distributed lock-free architecture
2. Add circuit breakers for busy loops
3. Implement peer connection quality metrics
4. Build automated deadlock detection and recovery

---

## Testing Checklist

Before deploying any fix:

- [ ] Test with 0 peers (should NOT produce blocks)
- [ ] Test with 1 peer (should sync, then produce)
- [ ] Test with 3+ peers (full consensus)
- [ ] Test Docker networking (port forwarding working)
- [ ] Test AutoNAT (both nodes can hole-punch)
- [ ] Test graceful degradation (peer loss scenarios)
- [ ] Test recovery from network isolation
- [ ] Monitor for deadlocks (CPU usage, heartbeats)
- [ ] Verify no split-brain scenarios possible

---

## Monitoring & Alerts

**Add These Metrics**:

```rust
// Prometheus metrics
- libp2p_connected_peers_total
- libp2p_connection_attempts_total
- libp2p_connection_failures_total
- block_production_enabled (0 or 1)
- network_isolation_detected (0 or 1)
- deadlock_suspected (based on CPU + heartbeat)
```

**Alert Rules**:

```yaml
- alert: NetworkIsolation
  expr: libp2p_connected_peers_total == 0
  for: 5m

- alert: DeadlockSuspected
  expr: rate(block_production_heartbeat[5m]) == 0 AND cpu_usage > 100%
  for: 2m

- alert: SplitBrainDetected
  expr: abs(local_height - network_height) > 100 AND libp2p_connected_peers_total == 0
  for: 10m
```

---

## Conclusion

The node stuck issue is caused by **async lock acquisition deadlocks** that prevent the block production loop from executing. The network isolation issue is caused by **libp2p connection failures** combined with **block production without peer validation**.

**The fix requires**:
1. Moving ALL `.lock().await` calls inside `tokio::spawn`
2. Requiring peer connections before allowing block production
3. Fixing Docker networking and AutoNAT configuration
4. Adding comprehensive diagnostics and monitoring

**Critical**: Without these fixes, nodes will continue to get stuck and create isolated blockchain branches, leading to total consensus failure.
