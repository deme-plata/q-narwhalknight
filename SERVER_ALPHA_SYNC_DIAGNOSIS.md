# Server Alpha Sync Failure Diagnosis

**Date:** November 2, 2025, 06:40 CET
**Issue:** Server Alpha Docker container slow sync + progress bar not showing
**Status:** 🔍 ROOT CAUSE IDENTIFIED

---

## 🔍 Problem Summary

**Server Alpha (161.35.219.10)** Docker container `q-v0.6.7-beta`:
- Reports network height: **13 blocks** (WRONG)
- Actual Server Beta height: **5,817 blocks**
- Sync mode: Individual blocks only (~1 block every 2-3 seconds)
- Progress bar: NOT visible
- Turbo sync: NOT triggering

---

## 📊 Root Cause Analysis

### Timeline of Events:

```
05:26:21 ✅ Successfully connected to Server Beta (12D3KooWRX3GGK9...)
05:26:21 📢 Server Beta subscribed to /qnk/testnet-phase2/peer-heights
05:26:22 📨 Received ONE peer-heights message from Server Beta
05:27:02 👋 DISCONNECTION from Server Beta (after 41 seconds)
05:27:09 📊 Network height updated to 5 (from local Docker peer)
05:27:44 📊 Network height updated to 13 (from local Docker peer)
```

**Critical Finding:**
Server Alpha **DID** successfully connect to Server Beta's P2P network at 185.182.185.227:9001, but the connection **closed after only 41 seconds**. After disconnection, Server Alpha only receives peer height announcements from the local Docker peer (`12D3KooWCJJxJASS...` at 172.17.0.2).

---

## 🚨 Root Cause

### Connection Instability

**Primary Issue:** libp2p connection to Server Beta is unstable and disconnects prematurely

**Evidence:**
```
2025-11-02T05:26:21.661513Z ✅ [CONNECTION] Successfully connected to peer: 12D3KooWRX3GGK9Fs3iM3BfqYNNJiHBDujac7EHwqWjaK1n1kzPN
2025-11-02T05:26:21.661523Z 📍 [CONNECTION] Endpoint: Dialer { address: /ip4/185.182.185.227/tcp/9001/p2p/12D3KooWRX3GGK9Fs3iM3BfqYNNJiHBDujac7EHwqWjaK1n1kzPN, role_override: Dialer }
2025-11-02T05:26:21.686860Z 📢 Peer 12D3KooWRX3GGK9Fs3iM3BfqYNNJiHBDujac7EHwqWjaK1n1kzPN subscribed to topic: /qnk/testnet-phase2/peer-heights
2025-11-02T05:26:22.625620Z 📨 Gossipsub message from 12D3KooWRX3GGK9Fs3iM3BfqYNNJiHBDujac7EHwqWjaK1n1kzPN: topic=/qnk/testnet-phase2/peer-heights
...
2025-11-02T05:27:02.796563Z 👋 [DISCONNECTION] Connection closed with peer: 12D3KooWRX3GGK9Fs3iM3BfqYNNJiHBDujac7EHwqWjaK1n1kzPN (remaining peers: 1)
```

**Connection Duration:** 41 seconds (05:26:21 → 05:27:02)

**Why Connection Disconnects:**

Possible causes:
1. **Network timeout** - Connection idle timeout or keepalive failure
2. **Firewall/NAT** - Connection tracking timeout in firewall
3. **libp2p keepalive** - Ping/keepalive interval too long
4. **Resource limits** - Docker container resource constraints
5. **P2P protocol issue** - Gossipsub mesh maintenance problem

---

## 💡 Proposed Solutions

### Solution 1: Increase libp2p Keepalive Frequency (RECOMMENDED)

**File:** `crates/q-network/src/unified_network_manager.rs`

**Current (suspected):**
```rust
let swarm = SwarmBuilder::with_tokio_executor(transport, behaviour, local_peer_id)
    .build();
// Default ping interval: 15 seconds
// Default keepalive: connection_idle_timeout 10 seconds
```

**Fix:**
```rust
use libp2p::swarm::SwarmBuilder;
use libp2p::swarm::ConnectionLimits;
use libp2p::swarm::Config;

let swarm_config = Config::with_tokio_executor()
    .with_idle_connection_timeout(Duration::from_secs(300)) // 5 minutes instead of 10 seconds
    .build();

let swarm = SwarmBuilder::with_existing_identity(local_key_pair)
    .with_tokio()
    .with_transport(transport)
    .with_behaviour(|_| behaviour)?
    .with_swarm_config(|_| swarm_config)
    .build();
```

**Benefits:**
- ✅ Prevents premature connection closure
- ✅ Maintains stable gossipsub mesh
- ✅ Continuous peer height announcements
- ✅ No code changes needed on Server Beta

### Solution 2: Add Automatic Reconnection Logic

**File:** `crates/q-network/src/unified_network_manager.rs`

**Add to connection event handler:**
```rust
SwarmEvent::ConnectionClosed { peer_id, cause, .. } => {
    warn!("👋 [DISCONNECTION] Connection closed with peer: {}", peer_id);

    // If disconnected from bootstrap peer, immediately reconnect
    if bootstrap_peers.contains(&peer_id) {
        warn!("🔄 [AUTO-RECONNECT] Bootstrap peer disconnected - reconnecting...");
        if let Err(e) = swarm.dial(peer_id) {
            error!("❌ [AUTO-RECONNECT] Failed to redial {}: {}", peer_id, e);
        } else {
            info!("✅ [AUTO-RECONNECT] Redialing bootstrap peer {}", peer_id);
        }
    }
}
```

**Benefits:**
- ✅ Automatic recovery from disconnections
- ✅ Maintains network connectivity
- ✅ Resilient to temporary network issues

### Solution 3: Enable libp2p Ping Keepalive

**File:** `crates/q-network/src/unified_network_manager.rs`

**Current ping behavior:**
```rust
let ping_behaviour = Ping::new(PingConfig::new());
```

**Enhanced ping with keepalive:**
```rust
use libp2p::ping::{Ping, PingConfig};

let ping_config = PingConfig::new()
    .with_interval(Duration::from_secs(10))  // Ping every 10 seconds
    .with_timeout(Duration::from_secs(20));   // Timeout after 20 seconds

let ping_behaviour = Ping::new(ping_config);
```

**Benefits:**
- ✅ Active keepalive mechanism
- ✅ Early detection of dead connections
- ✅ Connection health monitoring

---

## 🎯 Recommended Implementation Plan

### Phase 1: Increase Connection Timeout (Quick Fix)

1. Modify `unified_network_manager.rs` swarm configuration
2. Set `idle_connection_timeout` to 300 seconds (5 minutes)
3. Rebuild and deploy to Server Alpha Docker container
4. Test: Monitor connection duration to Server Beta

**Expected Result:**
Connection stays alive, continuous peer height announcements received

### Phase 2: Add Auto-Reconnect (Robustness)

1. Add bootstrap peer tracking
2. Implement automatic reconnection on disconnection
3. Add backoff strategy (exponential backoff)
4. Deploy and test

**Expected Result:**
Even if connection drops, automatic recovery within seconds

### Phase 3: Enhanced Monitoring (Production)

1. Add connection duration metrics
2. Log disconnection reasons
3. Track reconnection attempts
4. Alert on repeated disconnections

---

## 📋 Testing Plan

### Test 1: Connection Stability
```bash
# Monitor Server Alpha Docker container logs
docker logs -f q-v0.6.7-beta | grep -E "CONNECTION|DISCONNECTION|12D3KooWRX3GGK9"

# Expected: Connection stays alive >5 minutes
# Expected: No disconnection events
```

### Test 2: Peer Height Reception
```bash
# Monitor peer height announcements
docker logs -f q-v0.6.7-beta | grep "peer-heights"

# Expected: Regular messages from Server Beta (every 30 seconds)
# Expected: Network height updates to 5,817+
```

### Test 3: Turbo Sync Activation
```bash
# Monitor turbo sync triggering
docker logs -f q-v0.6.7-beta | grep "TURBO SYNC"

# Expected: "Syncing to height 5817" message
# Expected: Progress bar appears
# Expected: ~10,000 blocks/min sync speed
```

---

## 📊 Success Criteria

- [x] Server Beta P2P port (9001) is accessible
- [x] Server Alpha connects to Server Beta successfully
- [ ] Connection stays alive >5 minutes
- [ ] Server Alpha receives peer heights from Server Beta (5,817+)
- [ ] Network height correctly reported as 5,817+
- [ ] Turbo sync triggers automatically
- [ ] Progress bar appears during sync
- [ ] Sync completes in <1 minute (5,817 blocks @ ~10k blocks/min)

---

## 🔧 Files to Modify

1. **`crates/q-network/src/unified_network_manager.rs`**
   - Line ~100-150: Swarm initialization
   - Add connection timeout configuration
   - Add automatic reconnection logic
   - Enhance ping keepalive

2. **`crates/q-network/Cargo.toml`**
   - Verify libp2p version supports connection timeout configuration
   - Update if necessary

---

**Status:** Ready to implement fix

**Next Steps:**
1. Implement Solution 1 (increase connection timeout)
2. Build and deploy to Server Alpha
3. Monitor connection stability
4. Verify turbo sync triggers
5. If successful, implement Solution 2 for robustness
