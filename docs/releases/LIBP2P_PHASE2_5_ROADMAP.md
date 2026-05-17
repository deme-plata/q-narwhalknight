# libp2p Integration Phases 2-5 Roadmap

**Date**: October 6, 2025
**Status**: Phase 1 Complete ✅ | Phases 2-5 Planned

---

## ✅ Phase 1 Complete: mDNS Discovery

**Achieved**:
- Zero-config peer discovery via mDNS
- Auto-dial discovered peers
- Identify protocol for peer exchange
- Ping keepalive
- **Test Result**: 2 nodes discovered each other in ~5 seconds

**Integration**:
- crates/q-network/src/lib.rs (exports)
- crates/q-api-server/src/lib.rs (initialization)
- crates/q-api-server/src/main.rs (event loop)

---

## 🔄 Phase 2: Bridge libp2p to P2P Connection Manager

**Goal**: Connect libp2p discovered peers to existing ConnectionManager

**Current State**:
- `UnifiedNetworkManager` discovers peers via mDNS
- `ConnectionManager` handles actual consensus connections
- **Gap**: libp2p peers not bridged to ConnectionManager

**Implementation Options**:

### Option A: Extract Addresses from libp2p (Recommended)
```rust
impl UnifiedNetworkManager {
    /// Get discovered peer addresses for connection manager
    pub async fn get_discovered_peer_addresses(&self) -> Vec<(PeerId, Vec<Multiaddr>)> {
        let addresses = self.peer_addresses.read().await;
        addresses.iter()
            .map(|(peer_id, addrs)| (*peer_id, addrs.clone()))
            .collect()
    }

    /// Convert libp2p multiaddr to SocketAddr for TCP connection
    fn multiaddr_to_socket_addr(addr: &Multiaddr) -> Option<SocketAddr> {
        // Parse /ip4/X.X.X.X/tcp/PORT format
        // Return Some(SocketAddr) if valid TCP address
    }
}
```

### Option B: Direct Integration
Modify `UnifiedNetworkManager::handle_behaviour_event()` to:
1. Extract IP:PORT from Multiaddr
2. Create PeerInfo struct
3. Call `connection_manager.add_discovered_peer(peer_info)`

**Estimated Time**: 2-3 hours

---

## 📡 Phase 3: Add Gossipsub for Consensus Messages

**Goal**: Use libp2p Gossipsub for consensus message propagation

**Benefits**:
- Efficient pub/sub messaging
- Topic-based routing (/qnk/blocks, /qnk/votes)
- Automatic message deduplication
- Peer scoring for Byzantine resistance

**Implementation**:
```rust
#[derive(NetworkBehaviour)]
pub struct QNarwhalBehaviour {
    mdns: mdns::tokio::Behaviour,
    identify: libp2p::identify::Behaviour,
    ping: libp2p::ping::Behaviour,
    gossipsub: gossipsub::Behaviour,  // NEW
}

// Topics:
// - /qnk/blocks/1.0.0
// - /qnk/votes/1.0.0
// - /qnk/ack/1.0.0
```

**Integration with Consensus**:
- DAG-Knight blocks → Gossipsub publish
- Mempool transactions → Gossipsub broadcast
- Vote aggregation → Gossipsub subscribe

**Estimated Time**: 4-6 hours

---

## 🕸️ Phase 4: Multi-Node Mesh Network Testing

**Goal**: Verify libp2p scales to 4-10+ nodes

**Test Scenarios**:

### Test 4.1: 4-Node Local Network
```bash
# Launch 4 nodes with mDNS
for i in {1..4}; do
  Q_DB_PATH=./data-mesh-node$i Q_P2P_PORT=$((9210+i)) \
  ./q-api-server --port $((9100+i)) &
done

# Expected: Full mesh (each node connected to 3 others)
```

### Test 4.2: 10-Node Stress Test
- Verify discovery time stays <10s
- Check connection quality
- Monitor Gossipsub propagation delay

### Test 4.3: Cross-Server Discovery
- Node on Server Alpha discovers Node on Server Beta via mDNS
- Test with Tor + libp2p (both running)

**Success Criteria**:
- ✅ All nodes discover each other
- ✅ Messages propagate to all nodes
- ✅ No connection failures
- ✅ Discovery time scales linearly

**Estimated Time**: 3-4 hours

---

## ⚡ Phase 5: Performance Optimization

**Goal**: Minimize libp2p overhead, maximize throughput

**Optimizations**:

### 5.1: Connection Limits
```rust
let swarm_config = Config::with_tokio_executor()
    .with_idle_connection_timeout(Duration::from_secs(30))
    .with_max_negotiating_inbound_streams(128);
```

### 5.2: Gossipsub Tuning
```rust
let gossipsub_config = gossipsub::ConfigBuilder::default()
    .heartbeat_interval(Duration::from_millis(100))  // Fast propagation
    .mesh_n_low(4)     // Min peers in mesh
    .mesh_n_high(12)   // Max peers in mesh
    .build()?;
```

### 5.3: Message Batching
- Batch multiple consensus messages into single Gossipsub publish
- Reduces network overhead for high TPS

### 5.4: Metrics & Monitoring
```rust
pub struct LibP2pMetrics {
    discovered_peers: usize,
    active_connections: usize,
    messages_sent: u64,
    messages_received: u64,
    gossipsub_mesh_size: usize,
}
```

**Estimated Time**: 2-3 hours

---

## 📋 Implementation Summary

| Phase | Status | Time Estimate | Complexity |
|-------|--------|---------------|------------|
| Phase 1: mDNS Discovery | ✅ Complete | ~2 hours | Low |
| Phase 2: Bridge to ConnectionManager | 📋 Planned | 2-3 hours | Medium |
| Phase 3: Gossipsub Integration | 📋 Planned | 4-6 hours | High |
| Phase 4: Multi-Node Testing | 📋 Planned | 3-4 hours | Medium |
| Phase 5: Optimization | 📋 Planned | 2-3 hours | Medium |
| **Total** | **20% Done** | **~15 hours** | **Medium-High** |

---

## 🎯 Recommended Next Steps

1. **Validate Phase 1** with multi-node test (4 nodes)
2. **Decide on bridging strategy** for Phase 2
3. **Implement Phase 2** (connection manager bridge)
4. **Test Phase 2** with 2-4 nodes
5. **Proceed to Phase 3** (Gossipsub) if Phase 2 successful

---

## 🔧 Current Architecture

```
┌─────────────────────────────────────────────────────┐
│              Q-NarwhalKnight Node                   │
│                                                     │
│  ┌──────────────┐         ┌──────────────────┐    │
│  │   libp2p     │         │  ConnectionManager│    │
│  │   (mDNS)     │ ----?-> │   (TCP Streams)   │    │
│  └──────────────┘         └──────────────────┘    │
│         ↓                           ↓              │
│  Discover Peers              Consensus Messages    │
└─────────────────────────────────────────────────────┘

Current: libp2p discovers, but doesn't connect to ConnectionManager
Phase 2: Bridge the gap (marked with ?)
Phase 3: Add Gossipsub for consensus messaging
```

---

## 📊 Expected Performance After All Phases

| Metric | Current | After Phase 5 |
|--------|---------|---------------|
| **Peer Discovery** | mDNS only | mDNS + DHT |
| **Message Propagation** | Direct TCP | Gossipsub pub/sub |
| **Network Topology** | Partial mesh | Full mesh |
| **Consensus Integration** | None | Full integration |
| **Throughput** | ~350k TPS | Same (network not bottleneck) |
| **Discovery Time** | <5s for 2 nodes | <10s for 10+ nodes |

---

**Status**: Phase 1 complete and working! Ready to proceed with Phase 2 when approved.
