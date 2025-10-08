# 🌐 Dual-Stack Discovery Architecture for Q-NarwhalKnight

**Date**: October 6, 2025
**Status**: 🔨 **IN PROGRESS** - Phase 5a: Kademlia DHT
**Goal**: Enable both **clearnet (Kademlia DHT)** and **Tor onion** discovery

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│           Q-NarwhalKnight Discovery Architecture             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  LOCAL NETWORK (Phases 1-4: ✅ COMPLETE)                    │
│    ├── mDNS Discovery (~50ms)                               │
│    ├── Gossipsub mesh formation (~150ms)                    │
│    └── ConnectionManager bridge (channel-based)             │
│                                                              │
│  CLEARNET DISCOVERY (Phase 5a: 🔨 IN PROGRESS)              │
│    ├── Kademlia DHT for global peer discovery               │
│    ├── Bootstrap nodes for initial connectivity             │
│    ├── DHT queries for peer routing (5-30s)                 │
│    └── NAT traversal via relay nodes                        │
│                                                              │
│  TOR DISCOVERY (Phase 6: 📋 PLANNED per CLAUDE.md)          │
│    ├── .qnk onion addresses                                 │
│    ├── 4 dedicated circuits per validator                   │
│    ├── Tor-only mode for complete anonymity                 │
│    └── Quantum-enhanced circuit seeding (QRNG)              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 5a: Kademlia DHT Implementation

### Current Progress: Type Definitions ✅

**Files Modified**:
- `crates/q-network/src/unified_network_manager.rs`
  - Line 9: Added Kademlia imports
  - Line 36: Added `kademlia: Kademlia<MemoryStore>` to QNarwhalBehaviour
  - Line 48: Added `Kademlia(KademliaEvent)` to QNarwhalEvent
  - Lines 60-64: Added `From<KademliaEvent>` implementation

### Remaining Implementation Steps

#### 1. Initialize Kademlia DHT (crates/q-network/src/unified_network_manager.rs)

Add after line 125 (after ping configuration):

```rust
// Configure Kademlia DHT for global internet discovery (Phase 5a)
let mut kademlia_config = KademliaConfig::default();
kademlia_config.set_query_timeout(Duration::from_secs(60)); // Global DHT queries
kademlia_config.set_replication_factor(20.try_into().unwrap()); // High redundancy
kademlia_config.set_publication_interval(Some(Duration::from_secs(3600))); // Hourly refresh

let kademlia_store = MemoryStore::new(local_peer_id);
let mut kademlia = Kademlia::with_config(local_peer_id, kademlia_store, kademlia_config);

info!("🌍 Kademlia DHT initialized for clearnet discovery");
```

#### 2. Add Bootstrap Nodes

Add bootstrap node configuration (environment variable or config file):

```rust
// Bootstrap nodes for clearnet discovery
let bootstrap_peers = std::env::var("Q_BOOTSTRAP_PEERS")
    .unwrap_or_else(|_| {
        // Default bootstrap nodes (to be deployed)
        "/dns4/bootstrap1.q-narwhalknight.io/tcp/9000/p2p/12D3Koo...,\
         /dns4/bootstrap2.q-narwhalknight.io/tcp/9000/p2p/12D3Koo...,\
         /dns4/bootstrap3.q-narwhalknight.io/tcp/9000/p2p/12D3Koo...".to_string()
    });

// Add bootstrap peers to Kademlia
for addr_str in bootstrap_peers.split(',') {
    if let Ok(addr) = addr_str.trim().parse::<Multiaddr>() {
        if let Some(Protocol::P2p(peer_id_hash)) = addr.iter().last() {
            if let Ok(peer_id) = PeerId::from_multihash(peer_id_hash) {
                kademlia.add_address(&peer_id, addr.clone());
                info!("📍 Added bootstrap peer: {} at {}", peer_id, addr);
            }
        }
    }
}

// Bootstrap the DHT
kademlia.bootstrap().expect("Failed to bootstrap Kademlia DHT");
info!("🚀 Kademlia DHT bootstrap initiated");
```

#### 3. Update QNarwhalBehaviour Initialization

Modify line 158-163:

```rust
// Combine all behaviors
let behaviour = QNarwhalBehaviour {
    mdns,
    kademlia,      // ADD THIS
    identify,
    ping,
    gossipsub,
};
```

#### 4. Update Info Logging

Modify line 178 to include Kademlia:

```rust
info!("  • Kademlia DHT (clearnet discovery, bootstrap + periodic queries)");
```

#### 5. Add Kademlia Event Handlers

In `handle_behaviour_event()` method (after Gossipsub handlers), add:

```rust
QNarwhalEvent::Kademlia(kad_event) => {
    match kad_event {
        KademliaEvent::OutboundQueryProgressed {
            id,
            result,
            ..
        } => {
            match result {
                kad::QueryResult::GetClosestPeers(Ok(ok)) => {
                    info!("🌍 DHT query {}: Found {} peers", id, ok.peers.len());
                    for peer in ok.peers {
                        // Add discovered peers to Kademlia routing table
                        debug!("🔍 DHT peer discovered: {}", peer);
                    }
                }
                kad::QueryResult::GetClosestPeers(Err(err)) => {
                    warn!("⚠️ DHT query {} failed: {:?}", id, err);
                }
                kad::QueryResult::Bootstrap(Ok(ok)) => {
                    info!("✅ DHT bootstrap complete: {} peers in routing table", ok.num_remaining);
                }
                kad::QueryResult::Bootstrap(Err(err)) => {
                    error!("❌ DHT bootstrap failed: {:?}", err);
                }
                _ => {
                    debug!("🌍 Kademlia event: {:?}", result);
                }
            }
        }
        KademliaEvent::RoutingUpdated {
            peer,
            is_new_peer,
            addresses,
            ..
        } => {
            if is_new_peer {
                info!("🆕 New DHT peer added to routing table: {}", peer);

                // Bridge to ConnectionManager
                if let Some(ref tx) = self.peer_tx {
                    for addr in addresses.iter() {
                        if let Some(socket_addr) = Self::multiaddr_to_socket_addr(addr) {
                            let peer_info = PeerInfo {
                                address: socket_addr,
                                node_id: peer.to_string(),
                                server_role: ServerRole::Alpha,
                                discovered_via: DiscoveryMethod::DHT, // NEW: Add DHT to DiscoveryMethod enum
                                timestamp: SystemTime::now(),
                                onion_address: None,
                            };
                            let _ = tx.send(peer_info);
                            info!("🌉 Bridged DHT peer {} to ConnectionManager", peer);
                        }
                    }
                }
            }
        }
        _ => {
            debug!("🌍 Kademlia event: {:?}", kad_event);
        }
    }
}
```

#### 6. Add Periodic DHT Queries

Add method to UnifiedNetworkManager for periodic peer discovery:

```rust
/// Start periodic DHT queries for peer discovery
pub fn start_dht_discovery(&mut self) {
    // Query for random peer IDs every 60 seconds
    // This populates the routing table with global peers
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(60));
        loop {
            interval.tick().await;
            // Trigger DHT query via swarm command
            // (Implementation requires refactoring to send commands to swarm)
        }
    });
}
```

---

## Phase 5b: Bootstrap Node Support

### Bootstrap Node Deployment

**Requirements**:
1. Deploy 3-5 bootstrap nodes on different cloud providers
2. Stable public IPs with DNS records
3. High uptime (99.9%+)
4. Geographic distribution (US East, EU, Asia)

**Bootstrap Node Configuration**:
```toml
# /etc/q-narwhalknight/bootstrap.toml
[bootstrap]
mode = "bootstrap_node"
enable_mdns = false  # Don't use mDNS on public nodes
enable_kademlia = true
enable_gossipsub = true

[network]
listen_addr = "/ip4/0.0.0.0/tcp/9000"
announce_addr = "/ip4/PUBLIC_IP/tcp/9000"
max_connections = 1000  # Bootstrap nodes handle many connections

[kademlia]
server_mode = true  # Always respond to DHT queries
```

### Bootstrap DNS Records

```bash
# DNS TXT records for Q-NarwhalKnight bootstrap
_dnsaddr.q-narwhalknight.io. IN TXT "dnsaddr=/dns4/bootstrap1.q-narwhalknight.io/tcp/9000/p2p/12D3Koo..."
_dnsaddr.q-narwhalknight.io. IN TXT "dnsaddr=/dns4/bootstrap2.q-narwhalknight.io/tcp/9000/p2p/12D3Koo..."
_dnsaddr.q-narwhalknight.io. IN TXT "dnsaddr=/dns4/bootstrap3.q-narwhalknight.io/tcp/9000/p2p/12D3Koo..."
```

---

## Phase 5c: NAT Traversal via Relay Nodes

### Circuit Relay v2 Integration

```rust
use libp2p::relay;

// In UnifiedNetworkManager::new()
let relay = relay::v2::client::Behaviour::new(local_peer_id);

// Add to QNarwhalBehaviour
pub struct QNarwhalBehaviour {
    mdns: mdns::tokio::Behaviour,
    kademlia: Kademlia<MemoryStore>,
    relay: relay::v2::client::Behaviour,  // NEW
    identify: libp2p::identify::Behaviour,
    ping: libp2p::ping::Behaviour,
    gossipsub: gossipsub::Behaviour,
}
```

### Relay Node Discovery

```rust
// Discover relay nodes via Kademlia
// Connect to relay when behind restrictive NAT
// Use relay for hole-punching attempts
```

---

## Phase 5d: Performance Tuning & Metrics

### Connection Limits

```rust
// In SwarmConfig
let config = Config::with_tokio_executor()
    .with_idle_connection_timeout(Duration::from_secs(60))
    .with_max_negotiating_inbound_streams(256)
    .with_notify_handler_buffer_size(32);
```

### Gossipsub Mesh Tuning

```rust
let gossipsub_config = gossipsub::ConfigBuilder::default()
    .heartbeat_interval(Duration::from_millis(100))
    .validation_mode(ValidationMode::Strict)
    .mesh_n_low(4)        // Minimum peers in mesh
    .mesh_n(6)            // Target peers in mesh
    .mesh_n_high(12)      // Maximum peers in mesh
    .mesh_outbound_min(2) // Minimum outbound connections
    .build()?;
```

### Prometheus Metrics

```rust
use prometheus::{Counter, Gauge, Histogram, Registry};

pub struct NetworkMetrics {
    peers_discovered_total: Counter,
    active_connections: Gauge,
    dht_query_duration: Histogram,
    gossipsub_messages_sent: Counter,
}

// Integrate with libp2p swarm events
```

---

## Configuration: Dual-Stack Discovery Modes

### Mode 1: Clearnet Only (Default)
```bash
Q_DISCOVERY_MODE=clearnet \
Q_BOOTSTRAP_PEERS="..." \
./q-api-server
```

**Uses**: mDNS + Kademlia DHT + Bootstrap nodes

### Mode 2: Tor Only (Anonymous)
```bash
Q_DISCOVERY_MODE=tor \
Q_TOR_SOCKS5_PROXY=127.0.0.1:9050 \
./q-api-server
```

**Uses**: Tor onion discovery (Phase 6)

### Mode 3: Hybrid (Best of Both Worlds)
```bash
Q_DISCOVERY_MODE=hybrid \
Q_BOOTSTRAP_PEERS="..." \
Q_TOR_SOCKS5_PROXY=127.0.0.1:9050 \
./q-api-server
```

**Uses**: All discovery mechanisms simultaneously

---

## Testing Plan

### Phase 5a Test: Kademlia DHT Discovery

**Test Scenario**: 2 nodes on different networks (simulate clearnet)

```bash
# Node 1 (acts as bootstrap)
Q_DB_PATH=./data-dht-node1 Q_P2P_PORT=9001 \
RUST_LOG=info,q_network::unified_network_manager=debug \
./target/x86_64-unknown-linux-gnu/release/q-api-server --port 8001

# Get Node 1's multiaddr
# Example: /ip4/PUBLIC_IP/tcp/9001/p2p/12D3Koo...

# Node 2 (connects via DHT)
Q_BOOTSTRAP_PEERS="/ip4/NODE1_IP/tcp/9001/p2p/NODE1_PEER_ID" \
Q_DB_PATH=./data-dht-node2 Q_P2P_PORT=9002 \
RUST_LOG=info,q_network::unified_network_manager=debug \
./target/x86_64-unknown-linux-gnu/release/q-api-server --port 8002
```

**Success Criteria**:
- ✅ Node 2 bootstraps DHT from Node 1
- ✅ Nodes discover each other via Kademlia
- ✅ Gossipsub mesh forms across DHT connection
- ✅ Messages propagate between nodes
- ✅ Discovery time <30 seconds

---

## Migration Path from Phase 4

### Step 1: Add Kademlia to Existing Setup ✅ (In Progress)
- Type definitions added
- Kademlia behavior added to struct
- Event handling prepared

### Step 2: Enable Kademlia (Next)
- Initialize Kademlia in `new()`
- Add bootstrap node support
- Test 2-node DHT discovery

### Step 3: Production Bootstrap Nodes
- Deploy 3-5 bootstrap nodes
- Configure DNS records
- Update default bootstrap list

### Step 4: Add Relay Support
- Integrate Circuit Relay v2
- Enable NAT traversal
- Test from restrictive NAT

### Step 5: Performance Tuning
- Optimize connection limits
- Tune Gossipsub parameters
- Add Prometheus metrics

---

## Summary

| Discovery Method | Range | Latency | Anonymity | Status |
|-----------------|-------|---------|-----------|---------|
| **mDNS** | Local network | ~50ms | None | ✅ Phase 1-4 Complete |
| **Kademlia DHT** | Global clearnet | 5-30s | IP exposed | 🔨 Phase 5a In Progress |
| **Bootstrap Nodes** | Global clearnet | <5s | IP exposed | 📋 Phase 5b Planned |
| **Relay Nodes** | Global (NAT) | 100-500ms | IP exposed | 📋 Phase 5c Planned |
| **Tor Onion** | Global anonymous | 300-1000ms | Full anonymity | 📋 Phase 6 per CLAUDE.md |

**Current Implementation**: ~30% complete for full dual-stack architecture

**Next Immediate Step**: Complete Kademlia DHT initialization and test 2-node clearnet discovery
