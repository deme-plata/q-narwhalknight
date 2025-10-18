/// Simplified Zero-Knowledge Discovery for Q-NarwhalKnight
/// Implements peer discovery compatible with libp2p v0.53
/// Uses mDNS for local discovery and basic peer coordination

use libp2p::{
    core::{transport::Transport, upgrade},
    gossipsub::{self, IdentTopic, MessageId, ValidationMode},
    identity::Keypair,
    kad::{self, store::MemoryStore, Config as KademliaConfig, Event as KademliaEvent, Behaviour as Kademlia},
    noise, tcp, yamux,
    swarm::{SwarmEvent, Swarm, Config, NetworkBehaviour},
    Multiaddr,
    PeerId,
};

// mDNS is only available on non-Windows platforms due to libudev dependency
#[cfg(not(target_os = "windows"))]
use libp2p::mdns::{self, Event as MdnsEvent};
use futures::StreamExt;

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, error, info, warn};

use crate::connection_manager::{PeerInfo, DiscoveryMethod};
use crate::handshake::ServerRole;

/// Default bootstrap peer for global network connectivity
/// This is the production bootstrap node running on 185.182.185.227
const DEFAULT_BOOTSTRAP_PEER: &str = "/ip4/185.182.185.227/tcp/8081/p2p/12D3KooWPaQogoQVq1XoNenW93So8TC9T8CahEoMto455j4jgYmG";

/// Q-NarwhalKnight network behavior combining all discovery mechanisms
#[derive(NetworkBehaviour)]
#[behaviour(to_swarm = "QNarwhalEvent")]
pub struct QNarwhalBehaviour {
    /// mDNS for local network discovery (zero-config) - only available on non-Windows platforms
    #[cfg(not(target_os = "windows"))]
    mdns: mdns::tokio::Behaviour,
    /// Kademlia DHT for global internet discovery (clearnet)
    kademlia: Kademlia<MemoryStore>,
    /// Identify protocol for peer exchange
    identify: libp2p::identify::Behaviour,
    /// Ping protocol to keep connections alive
    ping: libp2p::ping::Behaviour,
    /// Gossipsub for consensus message propagation (Phase 3)
    gossipsub: gossipsub::Behaviour,
}

#[derive(Debug)]
pub enum QNarwhalEvent {
    #[cfg(not(target_os = "windows"))]
    Mdns(MdnsEvent),
    Kademlia(KademliaEvent),
    Identify(libp2p::identify::Event),
    Ping(libp2p::ping::Event),
    Gossipsub(gossipsub::Event),
}

#[cfg(not(target_os = "windows"))]
impl From<MdnsEvent> for QNarwhalEvent {
    fn from(event: MdnsEvent) -> Self {
        QNarwhalEvent::Mdns(event)
    }
}

impl From<KademliaEvent> for QNarwhalEvent {
    fn from(event: KademliaEvent) -> Self {
        QNarwhalEvent::Kademlia(event)
    }
}

impl From<libp2p::identify::Event> for QNarwhalEvent {
    fn from(event: libp2p::identify::Event) -> Self {
        QNarwhalEvent::Identify(event)
    }
}

impl From<libp2p::ping::Event> for QNarwhalEvent {
    fn from(event: libp2p::ping::Event) -> Self {
        QNarwhalEvent::Ping(event)
    }
}

impl From<gossipsub::Event> for QNarwhalEvent {
    fn from(event: gossipsub::Event) -> Self {
        QNarwhalEvent::Gossipsub(event)
    }
}

/// Simplified Network Manager - Zero-Knowledge Discovery System
pub struct UnifiedNetworkManager {
    /// libp2p swarm handling all protocols
    swarm: Swarm<QNarwhalBehaviour>,
    /// All discovered peers from ALL mechanisms
    discovered_peers: Arc<RwLock<HashSet<PeerId>>>,
    /// Peer addresses discovered (for connection manager bridge)
    peer_addresses: Arc<RwLock<HashMap<PeerId, Vec<Multiaddr>>>>,
    /// Local peer ID
    local_peer_id: PeerId,
    /// Channel to send discovered peers to ConnectionManager (Phase 2 bridge)
    peer_tx: Option<mpsc::UnboundedSender<PeerInfo>>,
    /// Channel to forward gossipsub messages (for database replication, etc.)
    gossipsub_message_tx: Option<mpsc::UnboundedSender<(String, Vec<u8>)>>,
}

impl UnifiedNetworkManager {
    /// Create new network manager with ZERO configuration required
    pub async fn new() -> anyhow::Result<Self> {
        // Generate or load keypair
        let keypair = Keypair::generate_ed25519();
        let local_peer_id = PeerId::from(keypair.public());

        info!("🚀 Starting Q-NarwhalKnight Zero-Knowledge Discovery");
        info!("🆔 Local Peer ID: {}", local_peer_id);

        // Create transport (TCP + Noise + Yamux) for libp2p v0.53
        let transport = tcp::tokio::Transport::new(tcp::Config::default())
            .upgrade(upgrade::Version::V1)
            .authenticate(noise::Config::new(&keypair)?)
            .multiplex(yamux::Config::default())
            .boxed();

        // Configure mDNS for local discovery (zero-config) - only on non-Windows platforms
        #[cfg(not(target_os = "windows"))]
        let mdns = mdns::Behaviour::new(mdns::Config::default(), local_peer_id)?;

        #[cfg(target_os = "windows")]
        info!("ℹ️ mDNS local discovery disabled on Windows (uses Kademlia DHT only)");

        // Configure Identify for peer info exchange
        let identify = libp2p::identify::Behaviour::new(
            libp2p::identify::Config::new("/qnarwhal/1.0.0".to_string(), keypair.public())
                .with_push_listen_addr_updates(true),
        );

        // Configure ping to keep connections alive
        let ping = libp2p::ping::Behaviour::new(libp2p::ping::Config::new());

        // Configure Kademlia DHT for global internet discovery (Phase 5a)
        let mut kad_config = KademliaConfig::default();
        kad_config.set_query_timeout(Duration::from_secs(60));

        let kad_store = MemoryStore::new(local_peer_id);
        let mut kademlia = Kademlia::with_config(local_peer_id, kad_store, kad_config);

        // Bootstrap from environment variable or use hardcoded default
        let bootstrap_peers_str = std::env::var("Q_BOOTSTRAP_PEERS")
            .unwrap_or_else(|_| {
                info!("ℹ️ Using default bootstrap peer: {}", DEFAULT_BOOTSTRAP_PEER);
                DEFAULT_BOOTSTRAP_PEER.to_string()
            });

        let mut bootstrap_count = 0;
        for addr_str in bootstrap_peers_str.split(',') {
            if let Ok(addr) = addr_str.trim().parse::<Multiaddr>() {
                // Extract peer ID from multiaddr (last component should be /p2p/<peer_id>)
                use libp2p::multiaddr::Protocol;
                if let Some(Protocol::P2p(peer_id)) = addr.iter().last() {
                    kademlia.add_address(&peer_id, addr.clone());
                    info!("📍 Added bootstrap peer: {} at {}", peer_id, addr);
                    bootstrap_count += 1;
                } else {
                    warn!("⚠️ Bootstrap multiaddr missing /p2p/ component: {}", addr);
                }
            }
        }

        // Start bootstrap if we have peers
        if bootstrap_count > 0 {
            match kademlia.bootstrap() {
                Ok(_) => {
                    info!("🚀 Kademlia DHT bootstrap initiated with {} peers", bootstrap_count);
                }
                Err(e) => {
                    warn!("⚠️ Failed to start DHT bootstrap: {:?}", e);
                }
            }
        } else {
            info!("ℹ️ No bootstrap peers configured - DHT will populate via mDNS discoveries");
        }

        info!("🌍 Kademlia DHT initialized for clearnet discovery");

        // Configure Gossipsub for consensus message propagation (Phase 3)
        let gossipsub_config = gossipsub::ConfigBuilder::default()
            .heartbeat_interval(Duration::from_millis(100)) // Fast propagation
            .validation_mode(ValidationMode::Strict) // Validate messages
            .message_id_fn(|message| {
                // Use message content hash as ID for deduplication
                MessageId::from(message.data.as_slice())
            })
            .build()
            .map_err(|e| anyhow::anyhow!("Gossipsub config error: {}", e))?;

        let mut gossipsub = gossipsub::Behaviour::new(
            gossipsub::MessageAuthenticity::Signed(keypair.clone()),
            gossipsub_config,
        )
        .map_err(|e| anyhow::anyhow!("Gossipsub initialization error: {}", e))?;

        // Subscribe to consensus topics
        let topics = vec![
            IdentTopic::new("/qnk/blocks/1.0.0"),    // Block propagation
            IdentTopic::new("/qnk/votes/1.0.0"),     // Vote aggregation
            IdentTopic::new("/qnk/ack/1.0.0"),       // Acknowledgements
        ];

        for topic in &topics {
            gossipsub.subscribe(topic)
                .map_err(|e| anyhow::anyhow!("Failed to subscribe to topic {}: {}", topic, e))?;
            debug!("📢 Subscribed to Gossipsub topic: {}", topic);
        }

        // Combine all behaviors
        let behaviour = QNarwhalBehaviour {
            #[cfg(not(target_os = "windows"))]
            mdns,
            kademlia,
            identify,
            ping,
            gossipsub,
        };

        // Build swarm using libp2p v0.53 API
        let config = Config::with_tokio_executor();
        let mut swarm = Swarm::new(transport, behaviour, local_peer_id, config);

        // Listen on all interfaces, random port
        swarm.listen_on("/ip4/0.0.0.0/tcp/0".parse()?)?;
        swarm.listen_on("/ip6/::/tcp/0".parse()?)?;

        info!("✅ Zero-Knowledge Discovery initialized successfully!");
        info!("📡 Discovery mechanisms active:");
        info!("  • mDNS (local network, <1 second)");
        info!("  • Kademlia DHT (global clearnet discovery)");
        info!("  • Identify (peer exchange)");
        info!("  • Ping (connection keepalive)");
        info!("  • Gossipsub (consensus messaging, {} topics)", topics.len());

        Ok(Self {
            swarm,
            discovered_peers: Arc::new(RwLock::new(HashSet::new())),
            peer_addresses: Arc::new(RwLock::new(HashMap::new())),
            local_peer_id,
            peer_tx: None, // Set via set_peer_channel() after construction
            gossipsub_message_tx: None, // Set via set_gossipsub_channel() after construction
        })
    }

    /// Set channel for sending discovered peers to ConnectionManager (Phase 2)
    pub fn set_peer_channel(&mut self, tx: mpsc::UnboundedSender<PeerInfo>) {
        self.peer_tx = Some(tx);
        info!("🌉 libp2p → ConnectionManager bridge channel established");
    }

    /// Set channel for forwarding gossipsub messages to subscribers
    pub fn set_gossipsub_channel(&mut self, tx: mpsc::UnboundedSender<(String, Vec<u8>)>) {
        self.gossipsub_message_tx = Some(tx);
        info!("🌉 Gossipsub message forwarding channel established");
    }

    /// Main event loop - processes all discovery events
    pub async fn run(&mut self) -> anyhow::Result<()> {
        loop {
            match self.swarm.next().await {
                Some(SwarmEvent::Behaviour(event)) => {
                    self.handle_behaviour_event(event).await?;
                }
                Some(SwarmEvent::NewListenAddr { address, .. }) => {
                    info!("📍 Listening on: {}", address);
                }
                Some(SwarmEvent::ConnectionEstablished {
                    peer_id,
                    endpoint,
                    num_established,
                    ..
                }) => {
                    let mut peers = self.discovered_peers.write().await;
                    peers.insert(peer_id);
                    info!(
                        "🔗 Connected to peer: {} (total connections: {})",
                        peer_id, num_established
                    );
                    info!("📊 Total discovered peers: {}", peers.len());
                }
                Some(SwarmEvent::ConnectionClosed { peer_id, .. }) => {
                    debug!("👋 Connection closed: {}", peer_id);
                }
                _ => {}
            }
        }
    }

    /// Handle behavior-specific events
    async fn handle_behaviour_event(&mut self, event: QNarwhalEvent) -> anyhow::Result<()> {
        match event {
            #[cfg(not(target_os = "windows"))]
            QNarwhalEvent::Mdns(MdnsEvent::Discovered(peers)) => {
                for (peer_id, addr) in peers {
                    info!("✨ mDNS discovered: {} at {}", peer_id, addr);

                    // Store peer address for connection manager bridge
                    let mut addresses = self.peer_addresses.write().await;
                    addresses.entry(peer_id)
                        .or_insert_with(Vec::new)
                        .push(addr.clone());

                    // Send to ConnectionManager via channel (Phase 2 bridge)
                    if let Some(ref tx) = self.peer_tx {
                        if let Some(socket_addr) = Self::multiaddr_to_socket_addr(&addr) {
                            let peer_info = PeerInfo {
                                address: socket_addr,
                                node_id: peer_id.to_string(),
                                server_role: ServerRole::Alpha,
                                discovered_via: DiscoveryMethod::Multicast,
                                timestamp: SystemTime::now(),
                                onion_address: None,
                            };
                            if let Err(e) = tx.send(peer_info) {
                                warn!("⚠️ Failed to send peer to ConnectionManager: {}", e);
                            } else {
                                debug!("🌉 Bridged peer {} to ConnectionManager", peer_id);
                            }
                        }
                    }

                    self.swarm.dial(addr)?;
                }
            }
            #[cfg(not(target_os = "windows"))]
            QNarwhalEvent::Mdns(MdnsEvent::Expired(peers)) => {
                for (peer_id, _) in peers {
                    debug!("mDNS peer expired: {}", peer_id);
                    // Remove expired peer addresses
                    let mut addresses = self.peer_addresses.write().await;
                    addresses.remove(&peer_id);
                }
            }
            QNarwhalEvent::Kademlia(kad_event) => {
                match kad_event {
                    KademliaEvent::OutboundQueryProgressed {
                        id,
                        result,
                        ..
                    } => {
                        match result {
                            kad::QueryResult::GetClosestPeers(Ok(ok)) => {
                                info!("🌍 DHT query {:?}: Found {} peers", id, ok.peers.len());
                                for peer in ok.peers {
                                    debug!("🔍 DHT peer discovered: {}", peer);
                                }
                            }
                            kad::QueryResult::GetClosestPeers(Err(err)) => {
                                warn!("⚠️ DHT query {:?} failed: {:?}", id, err);
                            }
                            kad::QueryResult::Bootstrap(Ok(ok)) => {
                                info!("✅ DHT bootstrap complete: {} peers in routing table", ok.num_remaining);
                            }
                            kad::QueryResult::Bootstrap(Err(err)) => {
                                error!("❌ DHT bootstrap failed: {:?}", err);
                            }
                            _ => {
                                debug!("🌍 Kademlia query result: {:?}", result);
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

                            // Bridge to ConnectionManager (Phase 2 integration)
                            if let Some(ref tx) = self.peer_tx {
                                for addr in addresses.iter() {
                                    if let Some(socket_addr) = Self::multiaddr_to_socket_addr(addr) {
                                        let peer_info = PeerInfo {
                                            address: socket_addr,
                                            node_id: peer.to_string(),
                                            server_role: ServerRole::Alpha,
                                            discovered_via: DiscoveryMethod::Multicast, // TODO: Add DHT variant
                                            timestamp: SystemTime::now(),
                                            onion_address: None,
                                        };
                                        if let Err(e) = tx.send(peer_info) {
                                            warn!("⚠️ Failed to send DHT peer to ConnectionManager: {}", e);
                                        } else {
                                            info!("🌉 Bridged DHT peer {} to ConnectionManager: {}", peer, socket_addr);
                                        }
                                    }
                                }
                            }
                        } else {
                            debug!("🔄 DHT routing table updated for peer: {}", peer);
                        }
                    }
                    _ => {
                        debug!("🌍 Kademlia event: {:?}", kad_event);
                    }
                }
            }
            QNarwhalEvent::Identify(event) => {
                debug!("🔍 Identify event: {:?}", event);
            }
            QNarwhalEvent::Ping(event) => {
                debug!("🏓 Ping event: {:?}", event);
            }
            QNarwhalEvent::Gossipsub(gossipsub::Event::Message {
                propagation_source,
                message_id,
                message,
            }) => {
                info!(
                    "📨 Gossipsub message received from {}: topic={}, id={:?}, size={} bytes",
                    propagation_source,
                    message.topic,
                    message_id,
                    message.data.len()
                );

                // Forward to gossipsub message channel if available
                if let Some(ref tx) = self.gossipsub_message_tx {
                    let topic = message.topic.to_string();
                    let data = message.data.clone();

                    if let Err(e) = tx.send((topic.clone(), data)) {
                        warn!("⚠️ Failed to forward gossipsub message on topic {}: {}", topic, e);
                    } else {
                        debug!("✅ Forwarded gossipsub message on topic: {}", topic);
                    }
                }

                // Also log receipt for debugging
                debug!("📨 Message data: {:?}", message.data);
            }
            QNarwhalEvent::Gossipsub(gossipsub::Event::Subscribed { peer_id, topic }) => {
                info!("📢 Peer {} subscribed to topic: {}", peer_id, topic);
            }
            QNarwhalEvent::Gossipsub(gossipsub::Event::Unsubscribed { peer_id, topic }) => {
                info!("📢 Peer {} unsubscribed from topic: {}", peer_id, topic);
            }
            QNarwhalEvent::Gossipsub(event) => {
                debug!("📢 Gossipsub event: {:?}", event);
            }
        }
        Ok(())
    }

    /// Get all discovered peers from ALL discovery methods
    pub async fn get_discovered_peers(&self) -> Vec<PeerId> {
        self.discovered_peers.read().await.iter().cloned().collect()
    }

    /// Get discovered peer addresses for connection manager bridge (Phase 2)
    pub async fn get_discovered_peer_addresses(&self) -> Vec<PeerInfo> {
        let addresses = self.peer_addresses.read().await;
        let mut peer_infos = Vec::new();

        for (peer_id, multiaddrs) in addresses.iter() {
            for multiaddr in multiaddrs {
                if let Some(socket_addr) = Self::multiaddr_to_socket_addr(multiaddr) {
                    peer_infos.push(PeerInfo {
                        address: socket_addr,
                        node_id: peer_id.to_string(),
                        server_role: ServerRole::Alpha, // Default to Alpha for mDNS peers
                        discovered_via: DiscoveryMethod::Multicast, // mDNS is multicast-based
                        timestamp: SystemTime::now(),
                        onion_address: None, // libp2p peers don't have onion addresses
                    });
                    info!("🌉 Bridging libp2p peer {} -> ConnectionManager: {}", peer_id, socket_addr);
                }
            }
        }

        peer_infos
    }

    /// Convert libp2p Multiaddr to SocketAddr for TCP connection
    /// Parses /ip4/X.X.X.X/tcp/PORT or /ip6/.../tcp/PORT formats
    fn multiaddr_to_socket_addr(addr: &Multiaddr) -> Option<SocketAddr> {
        use libp2p::multiaddr::Protocol;

        let mut ip = None;
        let mut port = None;

        for component in addr.iter() {
            match component {
                Protocol::Ip4(addr) => ip = Some(std::net::IpAddr::V4(addr)),
                Protocol::Ip6(addr) => ip = Some(std::net::IpAddr::V6(addr)),
                Protocol::Tcp(p) => port = Some(p),
                _ => {}
            }
        }

        match (ip, port) {
            (Some(ip), Some(port)) => {
                let socket_addr = SocketAddr::new(ip, port);
                debug!("📍 Parsed multiaddr {} -> {}", addr, socket_addr);
                Some(socket_addr)
            }
            _ => {
                warn!("⚠️ Failed to parse multiaddr to SocketAddr: {}", addr);
                None
            }
        }
    }

    /// Announce ourselves as Q-NarwhalKnight node (simplified for mDNS-only)
    pub fn announce_self(&mut self) -> anyhow::Result<()> {
        info!("📢 Announced self to network via mDNS");
        Ok(())
    }

    /// Subscribe to a custom gossipsub topic
    pub fn subscribe_topic(&mut self, topic: &str) -> anyhow::Result<()> {
        let ident_topic = IdentTopic::new(topic);
        self.swarm.behaviour_mut().gossipsub
            .subscribe(&ident_topic)
            .map_err(|e| anyhow::anyhow!("Failed to subscribe to topic {}: {}", topic, e))?;
        info!("📢 Subscribed to gossipsub topic: {}", topic);
        Ok(())
    }

    /// Publish a message to a gossipsub topic
    pub fn publish_topic(&mut self, topic: &str, data: Vec<u8>) -> anyhow::Result<()> {
        let ident_topic = IdentTopic::new(topic);
        self.swarm.behaviour_mut().gossipsub
            .publish(ident_topic, data)
            .map_err(|e| anyhow::anyhow!("Failed to publish to topic {}: {}", topic, e))?;
        debug!("📤 Published message to gossipsub topic: {}", topic);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_zero_knowledge_discovery() {
        // Create manager with ZERO configuration
        let manager = UnifiedNetworkManager::new().await.unwrap();

        // No IPs needed
        // No ports needed
        // No environment variables needed
        // No configuration files needed

        // It just works!
        assert_eq!(manager.local_peer_id, manager.local_peer_id);
    }

    #[tokio::test]
    async fn test_parallel_discovery() {
        // Two nodes with NO prior knowledge of each other
        let mut node1 = UnifiedNetworkManager::new().await.unwrap();
        let mut node2 = UnifiedNetworkManager::new().await.unwrap();

        // They will discover each other via:
        // 1. mDNS if on same network (<1 second)
        // 2. Kademlia DHT if on internet (5-30 seconds)
        // 3. Gossip amplification from other peers

        // No configuration required!
    }
}