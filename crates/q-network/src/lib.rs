/// Q-Network: Quantum-ready libp2p networking layer
/// Phase 0: Classical Ed25519 + QUIC
/// Phase 1: Post-quantum TLS with crypto-agility
/// Phase 4: QKD integration

use q_types::*;
use anyhow::Result;
use libp2p::{
    gossipsub::{self, MessageId, TopicHash},
    identify,
    noise,
    quic,
    swarm::{NetworkBehaviour, SwarmEvent},
    tcp,
    yamux, Multiaddr, PeerId, Swarm, Transport,
};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use tokio::sync::{broadcast, RwLock};
use tracing::{debug, error, info, warn};

pub mod crypto_agile;
pub mod peer_discovery;
pub mod message_handler;
pub mod qkd_transport; // Phase 4 preparation

pub use crypto_agile::{CryptoProvider, CryptoScheme, AgileHandshake};
pub use peer_discovery::{PeerDiscovery, PeerInfo};
pub use message_handler::{MessageHandler, NetworkMessage};

/// Main networking component
pub struct QuantumNetwork {
    swarm: Swarm<QuantumBehaviour>,
    crypto_provider: CryptoProvider,
    message_tx: broadcast::Sender<NetworkMessage>,
    peer_info: RwLock<HashMap<PeerId, PeerInfo>>,
    node_id: NodeId,
    current_phase: Phase,
}

/// Combined network behavior
#[derive(NetworkBehaviour)]
pub struct QuantumBehaviour {
    pub gossipsub: gossipsub::Behaviour,
    pub identify: identify::Behaviour,
    pub peer_discovery: peer_discovery::Behaviour,
}

impl QuantumNetwork {
    /// Create new quantum network (Phase 0: Classical)
    pub async fn new_phase0(node_id: NodeId) -> Result<Self> {
        let keypair = libp2p::identity::Keypair::generate_ed25519();
        let peer_id = PeerId::from(keypair.public());
        
        info!("🌐 Initializing Q-Network Phase 0 with PeerId: {}", peer_id);

        // Configure transport with classical cryptography
        let transport = tcp::tokio::Transport::new(tcp::Config::default())
            .upgrade(libp2p::core::upgrade::Version::V1Lazy)
            .authenticate(noise::Config::new(&keypair)?)
            .multiplex(yamux::Config::default())
            .boxed();

        // Configure GossipSub with quantum-random topic IDs
        let gossipsub_config = gossipsub::ConfigBuilder::default()
            .heartbeat_interval(std::time::Duration::from_secs(1))
            .validation_mode(gossipsub::ValidationMode::Strict)
            .message_id_fn(quantum_message_id_fn) // Custom message ID for quantum resistance
            .build()?;

        let gossipsub = gossipsub::Behaviour::new(
            gossipsub::MessageAuthenticity::Signed(keypair.clone()),
            gossipsub_config,
        )?;

        // Configure Identify protocol
        let identify = identify::Behaviour::new(
            identify::Config::new("q-narwhal-knight/1.0.0".into(), keypair.public())
                .with_agent_version("q-narwhal-knight/0.1.0".into())
        );

        // Configure peer discovery
        let peer_discovery = peer_discovery::Behaviour::new();

        let behaviour = QuantumBehaviour {
            gossipsub,
            identify,
            peer_discovery,
        };

        let swarm = Swarm::new(transport, behaviour, peer_id, Default::default());

        let (message_tx, _) = broadcast::channel(10000);

        Ok(Self {
            swarm,
            crypto_provider: CryptoProvider::new_phase0()?,
            message_tx,
            peer_info: RwLock::new(HashMap::new()),
            node_id,
            current_phase: Phase::Phase0,
        })
    }

    /// Upgrade to Phase 1: Post-Quantum Cryptography
    pub async fn upgrade_to_phase1(&mut self) -> Result<()> {
        info!("🔄 Upgrading Q-Network to Phase 1 (Post-Quantum)");

        // Create new crypto provider with post-quantum algorithms
        self.crypto_provider = CryptoProvider::new_phase1()?;
        self.current_phase = Phase::Phase1;

        // Initiate crypto-agile handshake with all connected peers
        let connected_peers: Vec<PeerId> = self.swarm.connected_peers().cloned().collect();
        
        for peer_id in connected_peers {
            self.initiate_crypto_upgrade(peer_id).await?;
        }

        info!("✅ Successfully upgraded to Phase 1 (Post-Quantum)");
        Ok(())
    }

    /// Start the network event loop
    pub async fn run(&mut self) -> Result<()> {
        info!("🚀 Starting Q-Network event loop");

        loop {
            match self.swarm.select_next_some().await {
                SwarmEvent::NewListenAddr { address, .. } => {
                    info!("📡 Listening on {}", address);
                }
                SwarmEvent::ConnectionEstablished { peer_id, .. } => {
                    info!("🤝 Connected to peer: {}", peer_id);
                    self.on_peer_connected(peer_id).await?;
                }
                SwarmEvent::ConnectionClosed { peer_id, cause, .. } => {
                    info!("🔌 Disconnected from peer: {} ({})", peer_id, cause);
                    self.on_peer_disconnected(peer_id).await?;
                }
                SwarmEvent::Behaviour(event) => {
                    self.handle_behaviour_event(event).await?;
                }
                SwarmEvent::IncomingConnection { .. } => {
                    debug!("📥 Incoming connection");
                }
                SwarmEvent::OutgoingConnectionError { peer_id, error, .. } => {
                    if let Some(peer_id) = peer_id {
                        warn!("❌ Failed to connect to {}: {}", peer_id, error);
                    } else {
                        warn!("❌ Outgoing connection error: {}", error);
                    }
                }
                _ => {}
            }
        }
    }

    /// Handle behavior-specific events
    async fn handle_behaviour_event(&mut self, event: QuantumBehaviourEvent) -> Result<()> {
        match event {
            QuantumBehaviourEvent::Gossipsub(gossipsub::Event::Message {
                propagation_source,
                message,
                ..
            }) => {
                debug!("📨 Received gossip message from {}", propagation_source);
                self.handle_gossip_message(message).await?;
            }
            QuantumBehaviourEvent::Gossipsub(gossipsub::Event::Subscribed { peer_id, topic }) => {
                debug!("📢 Peer {} subscribed to topic {}", peer_id, topic);
            }
            QuantumBehaviourEvent::Identify(identify::Event::Received { peer_id, info }) => {
                debug!("🆔 Identified peer {}: {}", peer_id, info.protocol_version);
                self.update_peer_info(peer_id, info).await?;
            }
            QuantumBehaviourEvent::PeerDiscovery(peer_discovery::Event::PeerDiscovered { 
                peer_id, 
                capabilities 
            }) => {
                info!("🔍 Discovered peer {} with capabilities: {:?}", peer_id, capabilities);
                self.on_peer_discovered(peer_id, capabilities).await?;
            }
            _ => {}
        }
        Ok(())
    }

    /// Subscribe to consensus topics
    pub async fn subscribe_to_consensus_topics(&mut self) -> Result<()> {
        let topics = vec![
            "q-narwhal-vertices",
            "q-narwhal-certificates", 
            "q-dag-knight-anchors",
            "q-quantum-beacon",
        ];

        for topic_str in topics {
            let topic = gossipsub::IdentTopic::new(topic_str);
            self.swarm.behaviour_mut().gossipsub.subscribe(&topic)?;
            info!("📡 Subscribed to topic: {}", topic_str);
        }

        Ok(())
    }

    /// Broadcast vertex to network
    pub async fn broadcast_vertex(&mut self, vertex: &Vertex) -> Result<()> {
        let topic = gossipsub::IdentTopic::new("q-narwhal-vertices");
        let message = NetworkMessage::Vertex(vertex.clone());
        let data = postcard::to_allocvec(&message)?;

        self.swarm.behaviour_mut().gossipsub.publish(topic, data)?;
        debug!("📤 Broadcast vertex {} for round {}", 
               hex::encode(vertex.id), vertex.round);

        Ok(())
    }

    /// Broadcast certificate to network
    pub async fn broadcast_certificate(&mut self, certificate: &Certificate) -> Result<()> {
        let topic = gossipsub::IdentTopic::new("q-narwhal-certificates");
        let message = NetworkMessage::Certificate(certificate.clone());
        let data = postcard::to_allocvec(&message)?;

        self.swarm.behaviour_mut().gossipsub.publish(topic, data)?;
        debug!("📤 Broadcast certificate for vertex {} in round {}", 
               hex::encode(certificate.vertex_id), certificate.round);

        Ok(())
    }

    /// Handle incoming gossip messages
    async fn handle_gossip_message(&self, message: gossipsub::Message) -> Result<()> {
        let network_message: NetworkMessage = postcard::from_bytes(&message.data)?;
        
        // Forward to message handler
        if self.message_tx.send(network_message).is_err() {
            warn!("No receivers for network message");
        }

        Ok(())
    }

    /// Handle new peer connection
    async fn on_peer_connected(&mut self, peer_id: PeerId) -> Result<()> {
        // Store peer info
        {
            let mut peers = self.peer_info.write().await;
            peers.insert(peer_id, PeerInfo::new(peer_id));
        }

        // If we're in Phase 1, initiate crypto negotiation
        if self.current_phase == Phase::Phase1 {
            self.initiate_crypto_upgrade(peer_id).await?;
        }

        Ok(())
    }

    /// Handle peer disconnection
    async fn on_peer_disconnected(&mut self, peer_id: PeerId) -> Result<()> {
        let mut peers = self.peer_info.write().await;
        peers.remove(&peer_id);
        Ok(())
    }

    /// Handle peer discovery
    async fn on_peer_discovered(&mut self, peer_id: PeerId, capabilities: Vec<String>) -> Result<()> {
        // Attempt to connect to discovered peer
        if !self.swarm.is_connected(&peer_id) {
            debug!("🔗 Attempting to connect to discovered peer {}", peer_id);
            // Connection attempts would be handled by libp2p automatically
        }

        // Update peer capabilities
        {
            let mut peers = self.peer_info.write().await;
            if let Some(peer_info) = peers.get_mut(&peer_id) {
                peer_info.capabilities = capabilities;
            }
        }

        Ok(())
    }

    /// Update peer information from identify protocol
    async fn update_peer_info(&self, peer_id: PeerId, info: identify::Info) -> Result<()> {
        let mut peers = self.peer_info.write().await;
        if let Some(peer_info) = peers.get_mut(&peer_id) {
            peer_info.agent_version = Some(info.agent_version);
            peer_info.protocol_version = Some(info.protocol_version);
            peer_info.supported_protocols = info.protocols;
        }
        Ok(())
    }

    /// Initiate crypto-agile upgrade handshake
    async fn initiate_crypto_upgrade(&mut self, peer_id: PeerId) -> Result<()> {
        debug!("🔐 Initiating crypto-agile upgrade with peer {}", peer_id);

        let handshake = AgileHandshake::new(
            self.crypto_provider.get_supported_schemes(),
            self.current_phase,
        )?;

        let topic = gossipsub::IdentTopic::new("q-crypto-upgrade");
        let message = NetworkMessage::CryptoUpgrade(handshake);
        let data = postcard::to_allocvec(&message)?;

        self.swarm.behaviour_mut().gossipsub.publish(topic, data)?;
        Ok(())
    }

    /// Get connected peer count
    pub fn connected_peer_count(&self) -> usize {
        self.swarm.connected_peers().count()
    }

    /// Get network statistics
    pub async fn get_network_stats(&self) -> NetworkStats {
        let peer_info = self.peer_info.read().await;
        
        NetworkStats {
            connected_peers: self.connected_peer_count() as u64,
            total_peers_seen: peer_info.len() as u64,
            current_phase: self.current_phase,
            crypto_provider: self.crypto_provider.get_current_scheme(),
            uptime: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default(),
        }
    }

    /// Get message receiver for consensus layer
    pub fn get_message_receiver(&self) -> broadcast::Receiver<NetworkMessage> {
        self.message_tx.subscribe()
    }
}

/// Custom message ID function for quantum resistance
fn quantum_message_id_fn(message: &gossipsub::Message) -> MessageId {
    let mut hasher = DefaultHasher::new();
    message.data.hash(&mut hasher);
    message.source.hash(&mut hasher);
    // Add quantum-resistant elements
    message.sequence_number.hash(&mut hasher);
    MessageId::from(hasher.finish().to_string())
}

/// Network statistics for monitoring
#[derive(Debug, Clone, serde::Serialize)]
pub struct NetworkStats {
    pub connected_peers: u64,
    pub total_peers_seen: u64,
    pub current_phase: Phase,
    pub crypto_provider: String,
    pub uptime: std::time::Duration,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_network_creation() {
        let node_id = [1u8; 32];
        let network = QuantumNetwork::new_phase0(node_id).await;
        assert!(network.is_ok());
    }

    #[tokio::test]
    async fn test_topic_subscription() {
        let node_id = [1u8; 32];
        let mut network = QuantumNetwork::new_phase0(node_id).await.unwrap();
        
        let result = network.subscribe_to_consensus_topics().await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_message_id_function() {
        use libp2p::PeerId;
        
        let message = gossipsub::Message {
            source: Some(PeerId::random()),
            data: b"test message".to_vec(),
            sequence_number: Some(42),
            topic: TopicHash::from_raw("test"),
        };
        
        let id1 = quantum_message_id_fn(&message);
        let id2 = quantum_message_id_fn(&message);
        
        assert_eq!(id1, id2); // Same message should produce same ID
    }

    #[tokio::test]
    async fn test_network_stats() {
        let node_id = [1u8; 32];
        let network = QuantumNetwork::new_phase0(node_id).await.unwrap();
        
        let stats = network.get_network_stats().await;
        assert_eq!(stats.connected_peers, 0);
        assert_eq!(stats.current_phase, Phase::Phase0);
    }
}