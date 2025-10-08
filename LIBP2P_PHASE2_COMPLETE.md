# libp2p Phase 2 Implementation Status

**Date**: October 6, 2025
**Status**: ⚠️ **IN PROGRESS** - Partial implementation with threading issues

---

## What Was Implemented

### 1. Peer Address Tracking
✅ Modified `UnifiedNetworkManager` to store discovered peer addresses:
- Added `peer_addresses: Arc<RwLock<HashMap<PeerId, Vec<Multiaddr>>>>` field
- Updated mDNS event handler to store addresses when peers are discovered
- Added cleanup when peers expire

### 2. Bridging Methods
✅ Added `get_discovered_peer_addresses()` method to return `Vec<PeerInfo>`
✅ Added `multiaddr_to_socket_addr()` helper to parse libp2p addresses
✅ Properly converts libp2p PeerId + Multiaddr → ConnectionManager PeerInfo

### 3. Bridge Task (Attempted)
❌ Attempted to spawn periodic task in main.rs to bridge discoveries
❌ Hit Rust threading limitation: `Swarm` is not `Sync`

---

## Compilation Errors Encountered

### Error 1: ServerRole Privacy
**Problem**: `ServerRole` enum was private
**Fix**: Changed import from `crate::connection_manager::ServerRole` to `crate::handshake::ServerRole`
**Status**: ✅ Fixed

### Error 2: Optional ConnectionManager
**Problem**: `connection_manager` field is `Option<Arc<ConnectionManager>>`, not `Arc<ConnectionManager>`
**Fix**: Wrapped bridge spawn in `if let Some(connection_manager) = &app_state.connection_manager`
**Status**: ✅ Fixed

### Error 3: Swarm Not Sync (CURRENT BLOCKER)
**Problem**:
```rust
error[E0277]: `dyn Abstract<(PeerId, StreamMuxerBox)> + Send + Unpin` cannot be shared between threads safely
 --> crates/q-api-server/src/main.rs:1347:13
  |
  = help: the trait `Sync` is not implemented for `dyn Abstract<...>`
```

**Root Cause**: libp2p's `Swarm` contains non-`Sync` types, so it can't be shared across threads via `Arc<Mutex<UnifiedNetworkManager>>`.

**Attempted Approach**:
```rust
// This doesn't work because Swarm is not Sync:
let discovery_bridge = libp2p_discovery.clone(); // Arc<Mutex<UnifiedNetworkManager>>
tokio::spawn(async move {
    let discovery = discovery_bridge.lock().await; // Fails - not Sync
    let peer_infos = discovery.get_discovered_peer_addresses().await;
    ...
});
```

---

## Solution Options

### Option A: Channel-Based Communication (Recommended)
libp2p best practice is to use channels to communicate with the Swarm:

```rust
// In UnifiedNetworkManager
pub struct UnifiedNetworkManager {
    swarm: Swarm<QNarwhalBehaviour>,
    peer_tx: mpsc::Sender<PeerInfo>, // Send discovered peers via channel
    ...
}

impl UnifiedNetworkManager {
    async fn handle_behaviour_event(&mut self, event: QNarwhalEvent) -> Result<()> {
        match event {
            QNarwhalEvent::Mdns(MdnsEvent::Discovered(peers)) => {
                for (peer_id, addr) in peers {
                    if let Some(peer_info) = self.convert_to_peer_info(peer_id, addr) {
                        let _ = self.peer_tx.send(peer_info).await; // Send to ConnectionManager
                    }
                }
            }
            ...
        }
    }
}

// In main.rs
let (peer_tx, mut peer_rx) = mpsc::channel(100);
let mut discovery = UnifiedNetworkManager::new(peer_tx).await?;

// Spawn discovery event loop
tokio::spawn(async move {
    discovery.run().await
});

// Spawn peer receiver task
if let Some(connection_manager) = &app_state.connection_manager {
    let conn_mgr = connection_manager.clone();
    tokio::spawn(async move {
        while let Some(peer_info) = peer_rx.recv().await {
            conn_mgr.add_discovered_peer(peer_info).await;
        }
    });
}
```

### Option B: Move get_discovered_peer_addresses() to Separate Thread-Safe Struct
Create a separate `PeerRegistry` that is `Sync`:

```rust
pub struct PeerRegistry {
    peers: Arc<RwLock<HashMap<PeerId, Vec<Multiaddr>>>>,
}

impl PeerRegistry {
    pub async fn add_peer(&self, peer_id: PeerId, addr: Multiaddr) { ... }
    pub async fn get_discovered_peer_addresses(&self) -> Vec<PeerInfo> { ... }
}

pub struct UnifiedNetworkManager {
    swarm: Swarm<QNarwhalBehaviour>,
    peer_registry: Arc<PeerRegistry>, // Can be cloned and shared
    ...
}
```

Then in main.rs, clone the `peer_registry` separately from the swarm.

### Option C: Abandon Periodic Polling, Use Events Only
Remove the periodic bridge task entirely. Instead, send PeerInfo directly to ConnectionManager in the discovery event handler via a channel (same as Option A but simplified).

---

## Recommended Next Steps

1. **Implement Option A (Channel-Based)** - aligns with libp2p best practices
2. **Test 2-node discovery with bridging** - verify ConnectionManager receives discovered peers
3. **Document Phase 2 completion** in LIBP2P_INTEGRATION_COMPLETE.md
4. **Move to Phase 3**: Add Gossipsub for consensus messages

---

## Files Modified (Current State)

1. **crates/q-network/src/unified_network_manager.rs**:
   - Lines 24-25: Added imports for `PeerInfo`, `DiscoveryMethod`, `ServerRole`
   - Line 70: Added `peer_addresses` HashMap field
   - Lines 128, 173-176, 185-186: Store/remove peer addresses in event handlers
   - Lines 205-257: Added `get_discovered_peer_addresses()` and `multiaddr_to_socket_addr()`

2. **crates/q-api-server/src/main.rs**:
   - Lines 1343-1367: Attempted bridge task (DOES NOT COMPILE - Sync issue)

---

## Time Spent

- Phase 2 Investigation: 30 min
- Implementation Attempt: 1 hour
- Debugging threading issues: 30 min
- **Total**: ~2 hours

---

## Status Summary

| Component | Status | Notes |
|-----------|--------|-------|
| Peer Address Storage | ✅ Complete | Addresses tracked in HashMap |
| Multiaddr Parsing | ✅ Complete | Converts libp2p → SocketAddr |
| PeerInfo Conversion | ✅ Complete | Proper metadata mapping |
| Bridge Task | ❌ Blocked | Swarm not Sync, need channel approach |
| **Overall Phase 2** | **70% Complete** | Need to refactor to channel-based design |

---

**Next Action**: Implement channel-based communication (Option A) to complete Phase 2.
