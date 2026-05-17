# Peer Connection Issue Analysis & Workaround

## Date: 2025-10-23

## 🐛 Issue Reported

The console and frontend UI are showing **0 peers** connected, even though a masternode is running on port 8080 and briefly showed 1 peer before dropping back to 0.

---

## 🔍 Root Cause Analysis

### Problem 1: Peer Count Not Being Updated

**Location**: `crates/q-api-server/src/main.rs:568-571`

The code has comments indicating that `node_status.connected_peers` should be updated by network events:

```rust
// Note: node_status.connected_peers is updated by:
// - libp2p peer discovery events
// - Connection manager events
// Stats loop reads from node_status.connected_peers
```

However, **ALL peer connection update code is DEACTIVATED**:

- Line 1181: `// DEACTIVATED: status.connected_peers += 1;` (peer connected)
- Line 1209: `// DEACTIVATED: status.connected_peers = status.connected_peers.saturating_sub(1);` (peer disconnected)
- Line 1449: `// status.connected_peers += 1;` (BEP-44 discovery)
- Line 1515: `// status.connected_peers = discovered_peers.len() as u32;` (production discovery)

**Result**: `connected_peers` stays at 0 permanently.

---

### Problem 2: libp2p Swarm Thread-Safety

The `UnifiedNetworkManager` contains a libp2p `Swarm` which includes:
- `dyn Abstract<(PeerId, StreamMuxerBox)>` - NOT `Sync`
- `dyn Executor + Send` - NOT `Sync`
- `FuturesUnordered<Pin<Box<dyn Future>>>` - NOT `Sync`
- `dyn Stream<Item = Result<RtnlMessage>>` - NOT `Sync`

This makes it **impossible to call `get_peer_count()` from a separate tokio task** because the method requires `&self` which is not `Send + Sync`.

**Attempted Fix (Failed)**:
```rust
// This DOES NOT compile due to Send/Sync constraints
let peer_count = {
    let discovery = libp2p_discovery.lock().await;
    discovery.get_peer_count().await  // ❌ Cannot send &UnifiedNetworkManager across threads
};
```

**Compiler Errors**:
```
error[E0277]: `dyn Abstract<(PeerId, StreamMuxerBox)> + Send + Unpin` cannot be shared between threads safely
error[E0277]: `(dyn Executor + Send + 'static)` cannot be shared between threads safely
error[E0277]: `(dyn Future<Output = ()> + Send + 'static)` cannot be shared between threads safely
error[E0277]: `dyn Stream<Item = Result<RtnlMessage, Error>> + Send` cannot be shared between threads safely
```

---

## 💡 Solution Options

### Option 1: Add Atomic Peer Counter (Recommended)

Modify `UnifiedNetworkManager` to maintain an `Arc<AtomicUsize>` for peer count:

**In `crates/q-network/src/unified_network_manager.rs`**:
```rust
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct UnifiedNetworkManager {
    swarm: Swarm<QNarwhalBehaviour>,
    discovered_peers: Arc<RwLock<HashMap<PeerId, PeerDiscoveryInfo>>>,
    peer_count: Arc<AtomicUsize>,  // ✅ NEW: Thread-safe peer counter
    // ... other fields
}

impl UnifiedNetworkManager {
    // Update peer count whenever a peer connects/disconnects
    async fn handle_swarm_event(&mut self, event: SwarmEvent<QNarwhalBehaviourEvent>) {
        match event {
            SwarmEvent::ConnectionEstablished { peer_id, .. } => {
                self.peer_count.fetch_add(1, Ordering::SeqCst);
                info!("✅ Peer connected: {:?} (total: {})", peer_id, self.peer_count.load(Ordering::SeqCst));
            }
            SwarmEvent::ConnectionClosed { peer_id, .. } => {
                self.peer_count.fetch_sub(1, Ordering::SeqCst);
                info!("❌ Peer disconnected: {:?} (total: {})", peer_id, self.peer_count.load(Ordering::SeqCst));
            }
            // ... other events
        }
    }

    // Thread-safe getter
    pub fn get_peer_count_atomic(&self) -> Arc<AtomicUsize> {
        self.peer_count.clone()
    }
}
```

**In `crates/q-api-server/src/main.rs`**:
```rust
// Store the atomic peer counter in AppState
let peer_count_atomic = if let Some(libp2p_discovery) = &app_state.libp2p_discovery {
    let discovery = libp2p_discovery.lock().await;
    Some(discovery.get_peer_count_atomic())
} else {
    None
};

// In the stats update loop
loop {
    interval.tick().await;

    // Read peer count from atomic (thread-safe!)
    let connected_peers = if let Some(peer_count) = &peer_count_atomic {
        peer_count.load(Ordering::SeqCst)
    } else {
        0
    };

    // Update node_status
    {
        let mut status = app_state.node_status.write().await;
        status.connected_peers = connected_peers as u32;
    }

    // ... rest of stats update
}
```

---

### Option 2: Use Channels for Peer Count Updates

Have the libp2p event loop send peer count updates via a channel:

**In `crates/q-network/src/unified_network_manager.rs`**:
```rust
pub struct UnifiedNetworkManager {
    swarm: Swarm<QNarwhalBehaviour>,
    peer_count_tx: Option<tokio::sync::mpsc::UnboundedSender<usize>>,
    // ... other fields
}

impl UnifiedNetworkManager {
    pub fn set_peer_count_channel(&mut self, tx: tokio::sync::mpsc::UnboundedSender<usize>) {
        self.peer_count_tx = Some(tx);
    }

    async fn handle_swarm_event(&mut self, event: SwarmEvent<QNarwhalBehaviourEvent>) {
        match event {
            SwarmEvent::ConnectionEstablished { .. } => {
                let count = self.swarm.connected_peers().count();
                if let Some(tx) = &self.peer_count_tx {
                    let _ = tx.send(count);
                }
            }
            SwarmEvent::ConnectionClosed { .. } => {
                let count = self.swarm.connected_peers().count();
                if let Some(tx) = &self.peer_count_tx {
                    let _ = tx.send(count);
                }
            }
            // ... other events
        }
    }
}
```

**In `crates/q-api-server/src/main.rs`**:
```rust
// Create channel for peer count updates
let (peer_count_tx, mut peer_count_rx) = tokio::sync::mpsc::unbounded_channel();

// Give the channel to libp2p manager
if let Some(libp2p_discovery) = &app_state.libp2p_discovery {
    let mut discovery = libp2p_discovery.lock().await;
    discovery.set_peer_count_channel(peer_count_tx);
}

// Spawn task to receive peer count updates
let state_clone = app_state.clone();
tokio::spawn(async move {
    while let Some(peer_count) = peer_count_rx.recv().await {
        let mut status = state_clone.node_status.write().await;
        status.connected_peers = peer_count as u32;
        info!("🌐 Peer count updated: {} peers", peer_count);
    }
});
```

---

### Option 3: Re-activate Existing Event Handlers

The simplest solution is to **un-comment the deactivated peer tracking code**:

**Lines 1175-1182** (Peer Discovered):
```rust
// Change from:
// DEACTIVATED:                         {
// DEACTIVATED:                             let mut status = state_clone.node_status.write().await;
// DEACTIVATED:                             status.connected_peers += 1;
// DEACTIVATED:                         }

// To:
{
    let mut status = state_clone.node_status.write().await;
    status.connected_peers += 1;
}
```

**Lines 1206-1210** (Peer Disconnected):
```rust
// Change from:
// DEACTIVATED:                         {
// DEACTIVATED:                             let mut status = state_clone.node_status.write().await;
// DEACTIVATED:                             status.connected_peers = status.connected_peers.saturating_sub(1);
// DEACTIVATED:                         }

// To:
{
    let mut status = state_clone.node_status.write().await;
    status.connected_peers = status.connected_peers.saturating_sub(1);
}
```

**Problem**: These sections are inside `// DEACTIVATED:` blocks for network managers that are currently disabled (Arc<()> placeholders).

---

## 🎯 Recommended Action Plan

### Immediate Fix (Option 1 - Best)

1. **Modify `UnifiedNetworkManager`** to add `Arc<AtomicUsize>` for peer count
2. **Update peer count** in `SwarmEvent::ConnectionEstablished` and `SwarmEvent::ConnectionClosed`
3. **Expose atomic counter** via `get_peer_count_atomic()`
4. **Read atomic value** in stats loop without locking the manager

### Benefits
- ✅ Thread-safe (no Send/Sync issues)
- ✅ Real-time updates
- ✅ No additional allocations
- ✅ Minimal performance impact

---

## 🐛 Why Peers Are Dropping

The brief appearance of 1 peer suggests that:

1. **Peer discovery is working** (mDNS or gossipsub found the masternode)
2. **Connection is established** briefly
3. **Connection drops** immediately (possibly due to handshake failure)
4. **No reconnection** attempts

### Possible Causes

1. **Protocol Mismatch**: Node on port 8080 uses different protocol version
2. **Timeout**: Connection times out before handshake completes
3. **Firewall**: Firewall drops incoming connections after initial SYN/ACK
4. **libp2p Configuration**: Transport or security mismatch between nodes

### Debug Steps

```bash
# Check if masternode is listening
ss -tulpn | grep 8080

# Check libp2p logs for connection errors
# Look for:
# - "Connection refused"
# - "Handshake failed"
# - "Protocol negotiation failed"
# - "Timeout"

# Test connectivity
curl http://localhost:8080/api/v1/status

# Check firewall rules
sudo iptables -L -n | grep 8080
```

---

## 📊 Current State

- **Peer Discovery**: ✅ Working (briefly finds peers)
- **Connection Establishment**: ⚠️ Partially working (connects but drops)
- **Peer Count Tracking**: ❌ Broken (all update code deactivated)
- **Frontend Display**: ❌ Shows 0 peers (reads from broken tracking)

---

## 🔮 Next Steps

1. **Implement Option 1** (Atomic peer counter) in `q-network` crate
2. **Add debug logging** to libp2p connection events
3. **Investigate why connections drop** after brief success
4. **Test with two nodes** on same machine
5. **Verify peer count updates** in console and frontend

---

**Status**: ⚠️ Issue Identified - Requires `q-network` Crate Modification

**Blocking**: Cannot update peer count from separate thread due to libp2p Swarm thread-safety constraints

**Solution**: Add `Arc<AtomicUsize>` peer counter to `UnifiedNetworkManager` for thread-safe access
