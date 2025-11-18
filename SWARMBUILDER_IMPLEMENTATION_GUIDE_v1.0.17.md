# SwarmBuilder Implementation Guide - v1.0.17-beta

## 🎯 OBJECTIVE
Replace the current "dummy" relay client initialization with proper SwarmBuilder pattern to enable functional NAT traversal.

---

## 📋 IMPLEMENTATION CHECKLIST

### Step 1: Add connection-limits feature to Cargo.toml ✅ TODO

**File**: `crates/q-network/Cargo.toml`

```toml
libp2p = { version = "0.53", features = [
    "noise", "yamux", "tcp", "gossipsub", "identify", "ping", "kad", "upnp",
    "macros", "tokio", "request-response",
    "autonat", "relay", "dcutr", "quic", "dns",
    "connection-limits",  # 🔥 ADD THIS
] }
```

### Step 2: Update QNarwhalBehaviour struct ✅ TODO

**File**: `crates/q-network/src/unified_network_manager.rs:50-95`

Add `connection_limits` field:

```rust
#[derive(NetworkBehaviour)]
#[behaviour(to_swarm = "QNarwhalEvent")]
pub struct QNarwhalBehaviour {
    #[cfg(not(target_os = "windows"))]
    mdns: mdns::tokio::Behaviour,
    kademlia: Kademlia<MemoryStore>,
    identify: libp2p::identify::Behaviour,
    ping: libp2p::ping::Behaviour,
    gossipsub: gossipsub::Behaviour,
    block_sync: libp2p::request_response::Behaviour<q_types::BlockPackCodec>,
    handshake: libp2p::request_response::Behaviour<crate::handshake_validator::HandshakeCodec>,

    // NAT traversal
    autonat: libp2p::autonat::Behaviour,
    relay: libp2p::relay::client::Behaviour,
    dcutr: libp2p::dcutr::Behaviour,

    // 🔥 ADD THIS:
    connection_limits: libp2p::connection_limits::Behaviour,
}
```

### Step 3: Add ConnectionLimits event variant ✅ TODO

**File**: `crates/q-network/src/unified_network_manager.rs:81-100`

```rust
pub enum QNarwhalEvent {
    // ... existing variants
    AutoNat(libp2p::autonat::Event),
    Relay(libp2p::relay::client::Event),
    Dcutr(libp2p::dcutr::Event),

    // 🔥 ADD THIS:
    ConnectionLimits(libp2p::connection_limits::Event),
}
```

### Step 4: Add From implementation for ConnectionLimits ✅ TODO

**File**: `crates/q-network/src/unified_network_manager.rs:140-170`

```rust
impl From<libp2p::connection_limits::Event> for QNarwhalEvent {
    fn from(event: libp2p::connection_limits::Event) -> Self {
        QNarwhalEvent::ConnectionLimits(event)
    }
}
```

### Step 5: Replace swarm construction with SwarmBuilder ✅ CRITICAL

**File**: `crates/q-network/src/unified_network_manager.rs:334-650`

**DELETE these lines (334-339)**:
```rust
// ❌ DELETE
let transport = tcp::tokio::Transport::new(tcp::Config::default())
    .upgrade(upgrade::Version::V1)
    .authenticate(noise::Config::new(&keypair)?)
    .multiplex(yamux::Config::default())
    .boxed();
```

**DELETE behaviour initialization section (595-618)**:
```rust
// ❌ DELETE (entire section from "// 🔥 v1.0.17-beta: Initialize NAT Traversal"
// through "let behaviour = QNarwhalBehaviour { ... }")
```

**DELETE swarm creation (620-630)**:
```rust
// ❌ DELETE
let config = Config::with_tokio_executor()
    .with_idle_connection_timeout(Duration::from_secs(300));
let mut swarm = Swarm::new(transport, behaviour, local_peer_id, config);
```

**REPLACE WITH** (insert at line 595, before "// Listen on configured port"):

```rust
// 🔥 v1.0.17-beta: Proper SwarmBuilder pattern with relay client
use libp2p::{SwarmBuilder, connection_limits::{ConnectionLimits, Behaviour as ConnLimitBehaviour}};

info!("🔧 Building swarm with SwarmBuilder pattern (NAT traversal enabled)");

// Configure connection limits
let limits = ConnectionLimits::default()
    .with_max_pending_incoming(Some(64))
    .with_max_pending_outgoing(Some(64))
    .with_max_established_incoming(Some(256))
    .with_max_established_outgoing(Some(256))
    .with_max_established_per_peer(Some(8));

// Build swarm using SwarmBuilder
let mut swarm = SwarmBuilder::with_existing_identity(keypair.clone())
    .with_tokio()
    .with_tcp(
        tcp::Config::default().port_reuse(true).nodelay(true),
        noise::Config::new,
        yamux::Config::default,
    )?
    .with_quic()  // QUIC transport for better NAT traversal
    .with_dns()?  // DNS resolution
    .with_relay_client(  // 🔥 CRITICAL: Proper relay client bound to transport
        noise::Config::new,
        yamux::Config::default,
    )?
    .with_behaviour(|keypair, relay_client| {
        let local_peer_id = keypair.public().to_peer_id();

        // mDNS for local discovery
        #[cfg(not(target_os = "windows"))]
        let mdns = mdns::tokio::Behaviour::new(mdns::Config::default(), local_peer_id)
            .expect("mDNS initialization failed");

        // Kademlia DHT (keep existing config from outer scope)
        let mut kad_config = KademliaConfig::default();
        kad_config.set_query_timeout(Duration::from_secs(60));
        let kad_store = MemoryStore::new(local_peer_id);
        let mut kademlia = Kademlia::with_config(local_peer_id, kad_store, kad_config);

        // Add bootstrap peers to Kademlia (copy from outer scope bootstrap logic)
        // NOTE: Bootstrap peer loop from lines 370-440 should run BEFORE this,
        // and the bootstrap_peer_map should be passed into with_behaviour closure

        // Identify
        let identify = libp2p::identify::Behaviour::new(
            libp2p::identify::Config::new("/qnarwhal/1.0.0".to_string(), keypair.public())
                .with_push_listen_addr_updates(true),
        );

        // Ping
        let ping = libp2p::ping::Behaviour::new(libp2p::ping::Config::new());

        // Gossipsub (keep existing config)
        let gossipsub_config = gossipsub::ConfigBuilder::default()
            .heartbeat_interval(Duration::from_millis(100))
            .validation_mode(ValidationMode::Strict)
            .max_transmit_size(50 * 1024 * 1024)
            .flood_publish(true)
            .mesh_outbound_min(1)
            .mesh_n_low(1)
            .mesh_n(2)
            .mesh_n_high(4)
            .message_id_fn(|message| {
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};

                let mut hasher = DefaultHasher::new();
                if let Some(source) = &message.source {
                    source.hash(&mut hasher);
                }
                message.data.hash(&mut hasher);
                if let Some(seq) = message.sequence_number {
                    seq.hash(&mut hasher);
                }
                MessageId::from(hasher.finish().to_le_bytes().to_vec())
            })
            .build()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        let mut gossipsub = gossipsub::Behaviour::new(
            gossipsub::MessageAuthenticity::Signed(keypair.clone()),
            gossipsub_config,
        )
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        // Subscribe to topics (copy from lines 515-562)
        // ... (topic subscription code here)

        // Request-response protocols
        use libp2p::request_response::{self, ProtocolSupport};
        use q_types::{BlockPackCodec, BlockPackProtocol};

        let block_sync_protocols = std::iter::once((BlockPackProtocol, ProtocolSupport::Full));
        let block_sync_config = request_response::Config::default();
        let block_sync = request_response::Behaviour::with_codec(
            BlockPackCodec::default(),
            block_sync_protocols,
            block_sync_config,
        );

        use crate::handshake_validator::{HandshakeCodec, HANDSHAKE_PROTOCOL};
        let handshake_protocols = std::iter::once((
            HANDSHAKE_PROTOCOL,
            ProtocolSupport::Full
        ));
        let handshake_config = request_response::Config::default();
        let handshake = request_response::Behaviour::with_codec(
            HandshakeCodec::default(),
            handshake_protocols,
            handshake_config,
        );

        // 🔥 NAT traversal - relay_client from closure parameter
        let autonat = libp2p::autonat::Behaviour::new(local_peer_id, Default::default());
        let relay = relay_client;  // ✅ CRITICAL: Use the provided relay client
        let dcutr = libp2p::dcutr::Behaviour::new(local_peer_id);

        // 🔒 Connection limits
        let connection_limits = ConnLimitBehaviour::new(limits.clone());

        Ok::<QNarwhalBehaviour, std::io::Error>(QNarwhalBehaviour {
            #[cfg(not(target_os = "windows"))]
            mdns,
            kademlia,
            identify,
            ping,
            gossipsub,
            block_sync,
            handshake,
            autonat,
            relay,
            dcutr,
            connection_limits,
        })
    })?
    .with_swarm_config(|c| {
        c.with_idle_connection_timeout(Duration::from_secs(30 * 60))
         .with_notify_handler_buffer_size(32)
         .with_per_connection_event_buffer_size(64)
    })
    .build();

info!("✅ Swarm built successfully with NAT traversal enabled");
```

### Step 6: Add ConnectionLimits event handler ✅ TODO

**File**: `crates/q-network/src/unified_network_manager.rs:1515` (after DCUtR handler)

```rust
QNarwhalEvent::ConnectionLimits(event) => {
    use libp2p::connection_limits::Event as ConnLimitEvent;
    match event {
        ConnLimitEvent::ConnectionEstablished { peer_id, .. } => {
            debug!("🔗 Connection established to: {}", peer_id);
        }
        ConnLimitEvent::ConnectionClosed { peer_id, .. } => {
            debug!("🔌 Connection closed to: {}", peer_id);
        }
        _ => {}
    }
}
```

---

## ⚠️ CRITICAL NOTES

### Bootstrap Peer Logic Restructuring

The bootstrap peer loop (lines 370-440) currently runs BEFORE swarm creation. With SwarmBuilder, you need to:

1. **Extract bootstrap peer discovery** into a separate function that returns `Vec<(PeerId, Multiaddr)>`
2. **Pass bootstrap peers** into the `.with_behaviour()` closure
3. **Add peers to Kademlia** inside the closure

Example:

```rust
// Before SwarmBuilder
let bootstrap_peers = discover_bootstrap_peers(&network_config).await?;

// Inside with_behaviour closure
for (peer_id, addr) in bootstrap_peers {
    kademlia.add_address(&peer_id, addr);
}
```

### Gossipsub Topic Subscriptions

Topics are subscribed in lines 527-562. Move this logic into the closure or do it AFTER swarm creation:

```rust
// After swarm.build()
for topic in &topics {
    swarm.behaviour_mut().gossipsub.subscribe(topic)?;
}
```

---

## 🧪 TESTING PROCEDURE

### 1. Compilation
```bash
cargo build --release --package q-network
cargo build --release --package q-api-server
```

### 2. Bootstrap Node Setup (Server Beta)

Add relay + AutoNAT server behaviours to bootstrap node. This requires a SEPARATE implementation on the bootstrap node side.

### 3. Home Network Test

```bash
# On home network node:
RUST_LOG=info,q_network=debug cargo run --bin q-api-server

# Expected logs:
# [INFO] 🔍 AutoNAT status changed: Unknown → Private
# [INFO] ✅ Relay reservation accepted from: 12D3KooW...
# [INFO] 🎉 DCUtR hole-punching success! Direct connection to: ...
```

---

## 📊 EXPECTED OUTCOMES

### Before Fix (Current State):
- AutoNAT: ✅ Works (client-only)
- Relay: ❌ Silently fails (no transport binding)
- DCUtR: ❌ Cannot work (no relay circuits)
- Home nodes: ❌ Unreachable behind NAT

### After Fix (SwarmBuilder):
- AutoNAT: ✅ Detects NAT status within 60s
- Relay: ✅ Establishes working circuits
- DCUtR: ✅ Punches holes when possible
- Home nodes: ✅ Reachable via relay + hole-punching

---

## 🎯 ESTIMATED EFFORT

- **Code changes**: 1-2 hours
- **Testing**: 2 hours
- **Total**: 3-4 hours to production-ready NAT traversal

---

## 📝 FINAL CHECKLIST

- [ ] Add `connection-limits` feature to Cargo.toml
- [ ] Update `QNarwhalBehaviour` struct with `connection_limits` field
- [ ] Add `ConnectionLimits` event variant and From impl
- [ ] Replace transport creation with SwarmBuilder
- [ ] Move behaviour initialization into `.with_behaviour()` closure
- [ ] Add ConnectionLimits event handler
- [ ] Refactor bootstrap peer logic
- [ ] Move gossipsub subscriptions after swarm creation
- [ ] Compile successfully
- [ ] Test on home network
- [ ] Verify relay circuits work
- [ ] Confirm DCUtR hole-punching

---

**Document Status**: IMPLEMENTATION READY
**Last Updated**: 2025-11-18 03:15 CET
**Next Action**: Begin implementation following this guide

---

**Generated by**: Claude Code (Server Beta)
**Context**: SwarmBuilder Pattern Implementation for NAT Traversal
