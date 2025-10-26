# libp2p Integration Roadmap - Enable Distributed Networking

**Objective**: Enable libp2p gossipsub networking without Tor dependency for distributed consensus testing

**Timeline**: 2-3 days for core implementation

---

## Phase 1: Decouple NetworkManager from Tor (Day 1)

### 1.1 Add Configuration Flag

**File**: `crates/q-network/src/lib.rs`

```rust
#[derive(Clone)]
pub struct NetworkConfig {
    pub node_id: NodeId,
    pub listen_port: u16,
    pub enable_tor: bool,  // ← NEW
    pub bootstrap_peers: Vec<Multiaddr>,
}

impl NetworkConfig {
    pub fn from_env() -> Self {
        let enable_tor = std::env::var("Q_ENABLE_TOR")
            .unwrap_or_else(|_| "false".to_string())
            .parse::<bool>()
            .unwrap_or(false);

        Self {
            node_id: NodeId::generate(),
            listen_port: std::env::var("Q_P2P_PORT")
                .unwrap_or("9050".to_string())
                .parse()
                .unwrap_or(9050),
            enable_tor,
            bootstrap_peers: vec![],
        }
    }
}
```

### 1.2 Modify NetworkManager Initialization

**File**: `crates/q-network/src/lib.rs`

```rust
pub struct NetworkManager {
    libp2p_bridge: Option<LibP2PBridge>,
    tor_client: Option<TorClient>,
    config: NetworkConfig,
}

impl NetworkManager {
    pub async fn new(config: NetworkConfig) -> Result<Self> {
        let (libp2p_bridge, tor_client) = if config.enable_tor {
            // Tor-enabled path (existing behavior)
            match TorClient::new().await {
                Ok(tor) => {
                    let bridge = LibP2PBridge::new_with_tor(config.clone(), &tor).await?;
                    (Some(bridge), Some(tor))
                }
                Err(e) => {
                    error!("Failed to initialize Tor: {}, falling back to direct libp2p", e);
                    let bridge = LibP2PBridge::new_direct(config.clone()).await?;
                    (Some(bridge), None)
                }
            }
        } else {
            // Direct libp2p path (NEW)
            info!("🌐 Starting libp2p in direct mode (Tor disabled)");
            let bridge = LibP2PBridge::new_direct(config.clone()).await?;
            (Some(bridge), None)
        };

        Ok(Self {
            libp2p_bridge,
            tor_client,
            config,
        })
    }
}
```

### 1.3 Update LibP2PBridge

**File**: `crates/q-network/src/libp2p_bridge.rs`

Add new initialization method:

```rust
impl LibP2PBridge {
    /// Initialize libp2p without Tor (direct TCP/IP)
    pub async fn new_direct(config: NetworkConfig) -> Result<Self> {
        let local_key = identity::Keypair::generate_ed25519();
        let local_peer_id = PeerId::from(local_key.public());

        info!("🆔 libp2p Peer ID: {}", local_peer_id);

        // Transport: TCP + Noise + Yamux (no Tor)
        let transport = tcp::tokio::Transport::new(tcp::Config::default())
            .upgrade(libp2p::core::upgrade::Version::V1Lazy)
            .authenticate(noise::Config::new(&local_key)?)
            .multiplex(yamux::Config::default())
            .boxed();

        // Gossipsub configuration
        let gossipsub_config = gossipsub::ConfigBuilder::default()
            .heartbeat_interval(Duration::from_secs(5))
            .validation_mode(gossipsub::ValidationMode::Strict)
            .message_id_fn(message_id_fn)
            .build()?;

        let mut gossipsub = gossipsub::Behaviour::new(
            gossipsub::MessageAuthenticity::Signed(local_key.clone()),
            gossipsub_config,
        )?;

        // Subscribe to consensus topics
        gossipsub.subscribe(&gossipsub::IdentTopic::new("/qnk/consensus/v1"))?;
        gossipsub.subscribe(&gossipsub::IdentTopic::new("/qnk/blocks/v1"))?;
        gossipsub.subscribe(&gossipsub::IdentTopic::new("/qnk/transactions/v1"))?;

        info!("📡 Subscribed to gossipsub topics");

        // mDNS for local peer discovery
        let mdns = mdns::tokio::Behaviour::new(
            mdns::Config::default(),
            local_peer_id,
        )?;

        info!("🔍 mDNS discovery enabled");

        // Identify protocol
        let identify = identify::Behaviour::new(identify::Config::new(
            "/qnk/1.0.0".to_string(),
            local_key.public(),
        ));

        // Create behavior
        let behaviour = QnkBehaviour {
            gossipsub,
            mdns,
            identify,
        };

        // Build swarm
        let mut swarm = SwarmBuilder::with_tokio_executor(
            transport,
            behaviour,
            local_peer_id,
        ).build();

        // Listen on all interfaces
        swarm.listen_on(format!("/ip4/0.0.0.0/tcp/{}", config.listen_port).parse()?)?;

        info!("✅ libp2p swarm listening on port {}", config.listen_port);

        Ok(Self {
            swarm: Arc::new(Mutex::new(swarm)),
            local_peer_id,
            config,
        })
    }

    /// Initialize libp2p with Tor (existing method - keep for later)
    pub async fn new_with_tor(config: NetworkConfig, tor: &TorClient) -> Result<Self> {
        // ... existing Tor initialization
        todo!("Tor integration - future work")
    }
}
```

---

## Phase 2: Update API Server Integration (Day 1)

### 2.1 Modify main.rs

**File**: `crates/q-api-server/src/main.rs`

```rust
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ... existing setup ...

    // Network configuration from environment
    let network_config = NetworkConfig::from_env();

    info!("🌐 Network configuration:");
    info!("  Tor enabled: {}", network_config.enable_tor);
    info!("  P2P port: {}", network_config.listen_port);

    // Initialize NetworkManager with new config
    let network_manager = match NetworkManager::new(network_config).await {
        Ok(nm) => {
            info!("✅ NetworkManager initialized successfully");
            Some(nm)
        }
        Err(e) => {
            error!("❌ NetworkManager initialization failed: {}", e);
            error!("⚠️ Continuing without P2P networking");
            None
        }
    };

    // ... rest of setup ...
}
```

### 2.2 Add Event Loop for libp2p

**File**: `crates/q-api-server/src/main.rs`

```rust
// Spawn libp2p event processing task
if let Some(ref nm) = network_manager {
    let nm_clone = nm.clone();
    tokio::spawn(async move {
        nm_clone.run_event_loop().await;
    });
}
```

**File**: `crates/q-network/src/lib.rs`

```rust
impl NetworkManager {
    /// Main event loop for processing libp2p events
    pub async fn run_event_loop(&self) {
        if let Some(ref bridge) = self.libp2p_bridge {
            loop {
                let event = bridge.swarm.lock().await.select_next_some().await;
                self.handle_swarm_event(event).await;
            }
        }
    }

    async fn handle_swarm_event(&self, event: SwarmEvent<QnkBehaviourEvent>) {
        match event {
            SwarmEvent::Behaviour(QnkBehaviourEvent::Mdns(mdns::Event::Discovered(peers))) => {
                for (peer_id, addr) in peers {
                    info!("🔍 mDNS discovered peer: {} at {}", peer_id, addr);
                    if let Some(ref bridge) = self.libp2p_bridge {
                        bridge.swarm.lock().await.dial(addr.clone()).ok();
                    }
                }
            }
            SwarmEvent::Behaviour(QnkBehaviourEvent::Gossipsub(gossipsub::Event::Message {
                propagation_source,
                message,
                ..
            })) => {
                info!("📨 Received gossip message from {}: {} bytes",
                    propagation_source, message.data.len());
                // TODO: Process consensus messages
            }
            SwarmEvent::Behaviour(QnkBehaviourEvent::Identify(identify::Event::Received {
                peer_id,
                info,
            })) => {
                info!("🆔 Identified peer {}: agent={}", peer_id, info.agent_version);
            }
            SwarmEvent::ConnectionEstablished { peer_id, endpoint, .. } => {
                info!("🔗 Connection established with peer: {} via {}", peer_id, endpoint);
            }
            SwarmEvent::ConnectionClosed { peer_id, cause, .. } => {
                warn!("❌ Connection closed with peer: {} (cause: {:?})", peer_id, cause);
            }
            _ => {}
        }
    }
}
```

---

## Phase 3: Testing & Validation (Day 2)

### 3.1 Update Test Script

**File**: `test_real_libp2p_network.sh`

```bash
#!/bin/bash

echo "🌐 Q-NarwhalKnight libp2p Network Test (Tor Disabled)"
echo "================================================================"

# Kill existing nodes
killall q-api-server 2>/dev/null
sleep 2

# Clean data directories
rm -rf ./data-libp2p-test-node*
for i in {0..3}; do
    mkdir -p ./data-libp2p-test-node$i
done

echo "📋 Node Configuration:"
echo "  Node 0 (Bootstrap): HTTP=9110, P2P=9210"
echo "  Node 1:             HTTP=9111, P2P=9211"
echo "  Node 2:             HTTP=9112, P2P=9212"
echo "  Node 3:             HTTP=9113, P2P=9213"
echo ""

# Launch bootstrap node
echo "🚀 Launching Bootstrap Node..."
Q_DB_PATH=./data-libp2p-test-node0 \
Q_P2P_PORT=9210 \
Q_ENABLE_TOR=false \
RUST_LOG=info,libp2p=debug,q_network=debug \
./target/x86_64-unknown-linux-gnu/release/q-api-server --port 9110 \
> libp2p-node0.log 2>&1 &

echo "  Bootstrap PID: $!"
sleep 3

# Launch peer nodes
echo ""
echo "🚀 Launching Peer Nodes..."
for i in 1 2 3; do
    HTTP_PORT=$((9110 + i))
    P2P_PORT=$((9210 + i))

    echo "  Starting Node $i (HTTP: $HTTP_PORT, P2P: $P2P_PORT)..."

    Q_DB_PATH=./data-libp2p-test-node$i \
    Q_P2P_PORT=$P2P_PORT \
    Q_ENABLE_TOR=false \
    RUST_LOG=info,libp2p=debug,q_network=debug \
    ./target/x86_64-unknown-linux-gnu/release/q-api-server --port $HTTP_PORT \
    > libp2p-node$i.log 2>&1 &

    echo "    PID: $!"
    sleep 2
done

echo ""
echo "✅ All 4 nodes launched"
echo ""

# Wait for peer discovery
echo "⏳ Waiting 15 seconds for mDNS peer discovery..."
for i in {1..15}; do
    echo -n "."
    sleep 1
done
echo ""
echo ""

# Check for peer discovery
echo "================================================================"
echo "📊 Peer Discovery Status:"
echo ""

for i in {0..3}; do
    echo "Node $i:"
    echo "  mDNS discoveries:"
    grep -c "mDNS discovered peer" libp2p-node$i.log 2>/dev/null || echo "    0"
    echo "  Connections established:"
    grep -c "Connection established" libp2p-node$i.log 2>/dev/null || echo "    0"
    echo "  Peer IDs:"
    grep "libp2p Peer ID:" libp2p-node$i.log 2>/dev/null | awk '{print "    " $NF}'
    echo ""
done

# Check gossipsub subscriptions
echo "================================================================"
echo "📡 Gossipsub Topic Subscriptions:"
echo ""
grep "Subscribed to gossipsub topics" libp2p-node*.log 2>/dev/null | wc -l | \
    awk '{print "  " $1 " nodes subscribed successfully"}'

# Test transaction gossip
echo ""
echo "================================================================"
echo "🧪 Testing Transaction Gossip..."
echo ""

# Submit transaction to Node 0
echo "Submitting test transaction to Node 0..."
curl -s -X POST http://localhost:9110/api/v1/transactions \
  -H "Content-Type: application/json" \
  -d '{
    "from": "test-sender",
    "to": "test-receiver",
    "amount": 100,
    "data": "libp2p-gossip-test"
  }' | jq '.'

sleep 2

# Check if transaction propagated via gossip
echo ""
echo "Checking transaction propagation:"
for i in {1..3}; do
    HTTP_PORT=$((9110 + i))
    echo -n "  Node $i: "
    if curl -s http://localhost:$HTTP_PORT/api/v1/mempool | grep -q "libp2p-gossip-test"; then
        echo "✅ Received"
    else
        echo "❌ Not received"
    fi
done

echo ""
echo "================================================================"
echo "📝 Summary:"
echo ""
echo "Logs available:"
echo "  tail -f libp2p-node0.log  # Bootstrap node"
echo "  tail -f libp2p-node1.log  # Peer node 1"
echo "  tail -f libp2p-node2.log  # Peer node 2"
echo "  tail -f libp2p-node3.log  # Peer node 3"
echo ""
echo "Stop nodes:"
echo "  killall q-api-server"
echo ""
echo "================================================================"
```

### 3.2 Expected Output

**Successful libp2p Discovery**:
```
[2025-10-06] INFO q_network: 🌐 Network configuration:
[2025-10-06] INFO q_network:   Tor enabled: false
[2025-10-06] INFO q_network:   P2P port: 9210
[2025-10-06] INFO q_network::libp2p_bridge: 🆔 libp2p Peer ID: 12D3KooWABC...
[2025-10-06] INFO q_network::libp2p_bridge: 📡 Subscribed to gossipsub topics
[2025-10-06] INFO q_network::libp2p_bridge: 🔍 mDNS discovery enabled
[2025-10-06] INFO q_network::libp2p_bridge: ✅ libp2p swarm listening on port 9210
[2025-10-06] INFO q_network: 🔍 mDNS discovered peer: 12D3KooWDEF... at /ip4/192.168.1.100/tcp/9211
[2025-10-06] INFO q_network: 🔗 Connection established with peer: 12D3KooWDEF...
[2025-10-06] INFO q_network: 🆔 Identified peer 12D3KooWDEF...: agent=q-narwhalknight/0.1.0
```

---

## Phase 4: Gossipsub Message Integration (Day 3)

### 4.1 Add Consensus Message Publishing

**File**: `crates/q-network/src/lib.rs`

```rust
impl NetworkManager {
    /// Publish consensus message via gossipsub
    pub async fn publish_consensus_message(&self, msg: ConsensusMessage) -> Result<()> {
        if let Some(ref bridge) = self.libp2p_bridge {
            let topic = gossipsub::IdentTopic::new("/qnk/consensus/v1");
            let data = bincode::serialize(&msg)?;

            bridge.swarm.lock().await
                .behaviour_mut()
                .gossipsub
                .publish(topic, data)?;

            info!("📤 Published consensus message to gossipsub");
        }
        Ok(())
    }

    /// Publish transaction to mempool topic
    pub async fn publish_transaction(&self, tx: Transaction) -> Result<()> {
        if let Some(ref bridge) = self.libp2p_bridge {
            let topic = gossipsub::IdentTopic::new("/qnk/transactions/v1");
            let data = bincode::serialize(&tx)?;

            bridge.swarm.lock().await
                .behaviour_mut()
                .gossipsub
                .publish(topic, data)?;

            info!("📤 Published transaction to gossipsub");
        }
        Ok(())
    }
}
```

### 4.2 Process Received Messages

```rust
async fn handle_swarm_event(&self, event: SwarmEvent<QnkBehaviourEvent>) {
    match event {
        SwarmEvent::Behaviour(QnkBehaviourEvent::Gossipsub(gossipsub::Event::Message {
            message,
            ..
        })) => {
            match message.topic.as_str() {
                "/qnk/consensus/v1" => {
                    if let Ok(msg) = bincode::deserialize::<ConsensusMessage>(&message.data) {
                        info!("📨 Received consensus message: {:?}", msg);
                        // Forward to consensus engine
                        // self.consensus_tx.send(msg).await?;
                    }
                }
                "/qnk/transactions/v1" => {
                    if let Ok(tx) = bincode::deserialize::<Transaction>(&message.data) {
                        info!("📨 Received transaction: {}", tx.hash);
                        // Forward to mempool
                        // self.mempool_tx.send(tx).await?;
                    }
                }
                _ => {}
            }
        }
        // ... other events
    }
}
```

---

## Implementation Checklist

### Day 1: Core Implementation
- [ ] Add `enable_tor` field to `NetworkConfig`
- [ ] Implement `NetworkConfig::from_env()` with `Q_ENABLE_TOR`
- [ ] Add `LibP2PBridge::new_direct()` method
- [ ] Modify `NetworkManager::new()` for conditional Tor
- [ ] Update `main.rs` to use new config
- [ ] Add libp2p event loop task
- [ ] Implement `run_event_loop()` and `handle_swarm_event()`

### Day 2: Testing
- [ ] Update test script with `Q_ENABLE_TOR=false`
- [ ] Test 4-node network launch
- [ ] Verify mDNS peer discovery
- [ ] Verify connection establishment
- [ ] Test gossipsub topic subscriptions
- [ ] Validate Identify protocol

### Day 3: Message Integration
- [ ] Implement `publish_consensus_message()`
- [ ] Implement `publish_transaction()`
- [ ] Add message deserialization in event handler
- [ ] Test transaction gossip propagation
- [ ] Benchmark message latency
- [ ] Document gossipsub performance

---

## Success Criteria

### Functional Requirements
✅ NetworkManager initializes without Tor when `Q_ENABLE_TOR=false`
✅ libp2p swarm listens on configured port
✅ mDNS discovers peers on local network
✅ Nodes establish peer connections automatically
✅ Gossipsub topics are subscribed correctly
✅ Messages propagate across all connected peers

### Performance Targets
- **Peer Discovery**: <5 seconds via mDNS
- **Connection Establishment**: <2 seconds per peer
- **Message Propagation**: <100ms for 4-node network
- **Gossip Overhead**: <5% of consensus bandwidth

### Testing Validation
- 4-node network with full mesh connectivity
- Transaction gossip reaches all nodes
- Consensus messages propagate correctly
- No Tor dependency errors in logs
- Identify protocol shows correct agent version

---

## Future Enhancements (Post-MVP)

### Tor Re-integration (Week 2)
- [ ] Implement `LibP2PBridge::new_with_tor()`
- [ ] Test libp2p over Tor SOCKS5 proxy
- [ ] Register .onion addresses for nodes
- [ ] Benchmark Tor vs direct performance

### Advanced P2P Features (Week 3+)
- [ ] DHT for global peer discovery
- [ ] Circuit relay for NAT traversal
- [ ] Peer scoring and reputation
- [ ] Network partition detection
- [ ] Adaptive gossip (Episub)

### Production Readiness (Week 4+)
- [ ] Connection pooling and limits
- [ ] Rate limiting per peer
- [ ] Message validation and filtering
- [ ] Metrics and monitoring (Prometheus)
- [ ] Configuration profiles (dev/prod)

---

## Risk Mitigation

### Risk: mDNS May Not Work Across Subnets
**Mitigation**: Add bootstrap peer list in config for manual peering

### Risk: Gossipsub Message Flooding
**Mitigation**: Implement message deduplication and rate limiting

### Risk: Connection Stability Issues
**Mitigation**: Add reconnection logic with exponential backoff

### Risk: Performance Degradation
**Mitigation**: Benchmark at each step, optimize critical paths

---

## References

- **libp2p Rust Docs**: https://docs.rs/libp2p
- **Gossipsub Spec**: https://github.com/libp2p/specs/tree/master/pubsub/gossipsub
- **mDNS Discovery**: https://docs.rs/libp2p-mdns
- **Code Locations**:
  - `crates/q-network/src/lib.rs` - NetworkManager
  - `crates/q-network/src/libp2p_bridge.rs` - libp2p integration
  - `crates/q-api-server/src/main.rs` - API server entry point

---

**Next Step**: Begin Day 1 implementation by adding `NetworkConfig::from_env()` and modifying `NetworkManager::new()` to support Tor-optional initialization.
