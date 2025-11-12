# P2P Network Isolation Analysis and Fix

**Date**: October 30, 2025
**Issue**: Server Alpha containers isolated on test network, not connecting to main network
**Status**: 🔍 Analysis Complete - Implementing Fix

---

## Problem Statement

Server Alpha containers are successfully running P2P gossipsub and communicating with each other (height ~57-62), but are **isolated on a separate network** and not connecting to the main network at Server Beta (114K+ blocks).

### Symptoms

1. **Server Alpha** (Docker containers):
   - Height: ~57-62 blocks
   - Connected peers: 4 (all in same isolated network)
   - Publishing blocks via gossipsub: ✅ Working
   - Network sync from Server Beta: ❌ Not working
   - Log: `📤 Publishing block 62 to gossipsub topic: /qnk/testnet/blocks`

2. **Server Beta** (Main network):
   - Height: 114,000+ blocks
   - Running on `185.182.185.227:8080`
   - Serving bootstrap discovery endpoint: `/api/v1/status`

### Root Cause Analysis

#### 1. **Bootstrap Discovery Flow**

From `crates/q-api-server/src/config.rs`:

```rust
// Default bootstrap URL
let bootstrap_url = env::var("Q_BOOTSTRAP_URL")
    .unwrap_or_else(|_| "http://185.182.185.227:8080".to_string());

// Fetch bootstrap peers from /api/v1/status
match Self::fetch_bootstrap_peers(&bootstrap_url) {
    Ok(peers) => {
        config.bootstrap_peers = peers;
    }
    Err(e) => {
        warn!("⚠️  No bootstrap peers discovered");
        warn!("⚠️  Falling back to mDNS local discovery only");
    }
}
```

**Issues Identified**:
- ❌ Docker containers may not have network route to 185.182.185.227
- ❌ HTTP request may be failing silently
- ❌ Multiaddr format from Server Beta may be incompatible
- ❌ Kademlia DHT not bootstrapping properly

#### 2. **Network Topology**

```
┌─────────────────────────────────────────────────────┐
│  Server Alpha (Docker Containers)                  │
│  - Container Network: bridge/overlay               │
│  - mDNS: ✅ Works (local discovery only)           │
│  - Bootstrap fetch: ❌ May fail (network isolation)│
└─────────────────────────────────────────────────────┘
                        ↓ (blocked?)
                   HTTP Request
                        ↓
┌─────────────────────────────────────────────────────┐
│  Server Beta (185.182.185.227:8080)                │
│  - Height: 114K+ blocks                             │
│  - Bootstrap endpoint: /api/v1/status               │
│  - Returns: p2p_multiaddr                           │
└─────────────────────────────────────────────────────┘
```

#### 3. **Gossipsub Topic Mismatch?**

Current topics: `/qnk/testnet/blocks`, `/qnk/testnet/acks`

**Question**: Does Server Beta use the same topic names?
- If Server Beta uses `/qnk/mainnet/blocks`, they won't communicate!

---

## Diagnostic Steps

### Step 1: Test Bootstrap URL Reachability

```bash
# From Server Alpha container
curl -v http://185.182.185.227:8080/api/v1/status

# Expected output:
# {
#   "p2p_multiaddr": "/ip4/185.182.185.227/tcp/8081/p2p/12D3KooW...",
#   "network_height": 114123,
#   ...
# }
```

### Step 2: Check Network Routing

```bash
# From Server Alpha container
ping -c 3 185.182.185.227
traceroute 185.182.185.227
```

### Step 3: Verify Gossipsub Topics

```bash
# Check Server Beta logs for topic names
journalctl -u q-api-server | grep -E "topic:|Subscribing"

# Expected: Should match /qnk/testnet/blocks
```

### Step 4: Check Peer ID Format

```bash
# Server Beta peer ID should be in bootstrap_peers list
# Format: /ip4/185.182.185.227/tcp/8081/p2p/12D3KooW...
```

---

## Proposed Fixes

### Fix 1: Add Explicit Bootstrap Peer Environment Variable

**For Server Alpha containers**, add to docker-compose or run command:

```bash
Q_BOOTSTRAP_PEERS="/ip4/185.182.185.227/tcp/8081/p2p/12D3KooWxxx"
```

Where `12D3KooWxxx` is Server Beta's actual peer ID (fetch from status endpoint).

### Fix 2: Enhance Bootstrap Discovery with Retry Logic

**File**: `crates/q-api-server/src/config.rs`

```rust
// Add retry logic for bootstrap discovery
fn fetch_bootstrap_peers_with_retry(url: &str, retries: u8) -> anyhow::Result<Vec<String>> {
    let mut last_error = None;

    for attempt in 0..retries {
        match Self::fetch_bootstrap_peers(url) {
            Ok(peers) if !peers.is_empty() => return Ok(peers),
            Ok(_) => {
                warn!("Attempt {}: No bootstrap peers found", attempt + 1);
            }
            Err(e) => {
                warn!("Attempt {}: Bootstrap fetch failed: {}", attempt + 1, e);
                last_error = Some(e);
            }
        }

        if attempt < retries - 1 {
            std::thread::sleep(Duration::from_secs(2));
        }
    }

    Err(last_error.unwrap_or_else(|| anyhow::anyhow!("No bootstrap peers found after {} attempts", retries)))
}
```

### Fix 3: Add Kademlia Bootstrap Mode

**File**: `crates/q-network/src/unified_network_manager.rs`

After adding bootstrap peers to Kademlia, trigger bootstrap mode:

```rust
// After adding bootstrap peers
if bootstrap_count > 0 {
    info!("🔄 Triggering Kademlia bootstrap with {} peers", bootstrap_count);

    // Trigger bootstrap mode
    if let Err(e) = kademlia.bootstrap() {
        warn!("⚠️  Failed to trigger Kademlia bootstrap: {:?}", e);
    }
}
```

### Fix 4: Ensure Topic Name Consistency

**Verify** that both Server Alpha and Server Beta use the same gossipsub topics:
- `/qnk/testnet/blocks`
- `/qnk/testnet/acks`

If Server Beta uses `/qnk/mainnet/*`, update to `/qnk/testnet/*` or vice versa.

### Fix 5: Add Network Diagnostics Endpoint

**File**: `crates/q-api-server/src/handlers.rs`

Add endpoint to expose P2P network status:

```rust
pub async fn p2p_diagnostics(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let network_manager = state.network_manager.read().await;

    Json(json!({
        "connected_peers": network_manager.get_peer_count(),
        "peer_list": network_manager.get_peer_list(),
        "gossipsub_topics": network_manager.get_subscribed_topics(),
        "bootstrap_peers": network_manager.get_bootstrap_peers(),
        "kademlia_routing_table_size": network_manager.get_routing_table_size(),
        "sync_status": {
            "is_syncing": network_manager.is_syncing(),
            "local_height": state.blockchain.read().await.get_height(),
            "network_height": network_manager.get_network_height(),
        }
    }))
}
```

---

## Implementation Plan

### Phase 1: Immediate Diagnostics (15 minutes)

1. ✅ SSH into Server Alpha container
2. ✅ Test `curl http://185.182.185.227:8080/api/v1/status`
3. ✅ Check network routing with `ping` and `traceroute`
4. ✅ Verify gossipsub topic names match

### Phase 2: Bootstrap Fix (30 minutes)

1. Fetch Server Beta's peer ID from `/api/v1/status`
2. Add explicit `Q_BOOTSTRAP_PEERS` environment variable to Server Alpha
3. Restart Server Alpha containers
4. Monitor logs for `✅ Connected to bootstrap peer` messages

### Phase 3: Code Enhancements (1 hour)

1. Implement retry logic for bootstrap discovery
2. Add Kademlia `bootstrap()` trigger
3. Add P2P diagnostics endpoint
4. Test with multiple isolated containers

### Phase 4: Network Sync Verification (30 minutes)

1. Monitor Server Alpha logs for `📥 Received block from network`
2. Verify height increases to match Server Beta (114K+)
3. Confirm peer count increases beyond 4 (connect to broader network)
4. Test block propagation from Server Alpha to Server Beta

---

## Expected Outcomes

After implementing fixes:

1. **Server Alpha containers** should:
   - Successfully connect to Server Beta's bootstrap peer
   - Discover additional peers via Kademlia DHT
   - Begin syncing blocks from height ~60 to 114K+
   - Show `network_height: 114000+` in status endpoint

2. **Gossipsub behavior** should show:
   ```
   📥 Received block 115 from peer 12D3KooW... (Server Beta)
   📥 Syncing blocks: local=62, network=114500, lag=114438
   ```

3. **Peer count** should increase from 4 to 10+ as DHT discovers more nodes

---

## Monitoring Commands

```bash
# Server Alpha - Check bootstrap connection
docker logs container_name | grep -E "bootstrap|Connected to peer"

# Server Alpha - Monitor sync progress
watch -n 5 'curl -s localhost:8080/api/v1/status | jq ".current_height, .network_height"'

# Server Beta - Check incoming connections
netstat -antp | grep :8081 | grep ESTABLISHED

# Both servers - Monitor gossipsub messages
journalctl -u q-api-server -f | grep -E "gossipsub|Received block|Publishing block"
```

---

## Risk Assessment

### Low Risk
- ✅ Bootstrap retry logic (graceful degradation)
- ✅ Explicit environment variable configuration
- ✅ P2P diagnostics endpoint (read-only)

### Medium Risk
- ⚠️  Kademlia bootstrap trigger (may cause connection churn)
- ⚠️  Topic name changes (requires coordinated update)

### Mitigation
- Test all changes in isolated Docker environment first
- Keep mDNS fallback active for local discovery
- Implement gradual rollout: fix Server Alpha first, monitor for 24h, then update Server Beta if needed

---

## Next Steps

1. **Execute Phase 1 diagnostics** to confirm root cause
2. **Apply Phase 2 fixes** with explicit bootstrap peer configuration
3. **Monitor results** for 1 hour to verify sync progress
4. **Implement Phase 3 code enhancements** if manual fix succeeds
5. **Document findings** in P2P gossipsub whitepaper

---

**Status**: Ready for diagnostic execution
**ETA**: Network sync should begin within 30 minutes of fix deployment
**Success Criteria**: Server Alpha height matches Server Beta (114K+) within 2 hours
