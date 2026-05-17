use anyhow::{Context, Result};
use libp2p::{
    futures::StreamExt,
    gossipsub::{self, MessageAuthenticity, ValidationMode},
    identity::Keypair,
    multiaddr::Protocol,
    swarm::{NetworkBehaviour, SwarmEvent},
    Multiaddr, PeerId, Swarm,
};
use pqcrypto_dilithium::dilithium5;
use pqcrypto_kyber::kyber1024;
use pqcrypto_traits::sign::SignedMessage;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tracing::{debug, info, warn, error};

// Note: Using simulated consensus types for demo
// use q_dag_knight::{ConsensusNode, ConsensusMessage, ConsensusState};
// use q_network::crypto_agile::{CryptoProvider, AgileHandshake};

/// Placeholder consensus types for demo
pub struct ConsensusNode {
    endpoint: String,
}

pub enum ConsensusMessage {
    RobotData { data: RobotConsensusData, timestamp: u64 },
}

pub struct ConsensusState {
    pub current_round: u64,
    pub finalized_blocks: u64,
    pub pending_transactions: u32,
}

impl ConsensusNode {
    pub async fn new(endpoint: String) -> Result<Self> {
        Ok(Self { endpoint })
    }
    
    pub async fn connect(&self) -> Result<()> {
        info!("Connected to consensus at {}", self.endpoint);
        Ok(())
    }
    
    pub async fn submit_message(&self, _message: ConsensusMessage) -> Result<()> {
        debug!("Submitted message to consensus");
        Ok(())
    }
    
    pub async fn query_active_robots(&self) -> Result<Vec<RobotConsensusData>> {
        Ok(Vec::new())
    }
    
    pub async fn query_swarm_status(&self) -> Result<Vec<RobotConsensusData>> {
        Ok(Vec::new())
    }
    
    pub async fn query_environmental_alerts(&self) -> Result<Vec<RobotConsensusData>> {
        Ok(Vec::new())
    }
    
    pub async fn get_state(&self) -> Result<ConsensusState> {
        Ok(ConsensusState {
            current_round: 1,
            finalized_blocks: 1,
            pending_transactions: 0,
        })
    }
    
    pub async fn next_event(&self) -> ConsensusEvent {
        // Simulate events
        tokio::time::sleep(Duration::from_secs(5)).await;
        ConsensusEvent::ConsensusStateUpdate {
            round: 1,
            finalized_blocks: 1,
            pending_transactions: 0,
        }
    }
}

/// Integration with Q-NarwhalKnight consensus network
pub struct ConsensusIntegration {
    consensus_node: ConsensusNode,
    swarm: Swarm<RobotConsensusNetworkBehaviour>,
    peer_id: PeerId,
    robot_data_channel: mpsc::Receiver<RobotConsensusData>,
    consensus_events_channel: mpsc::Sender<ConsensusEvent>,
    known_peers: HashMap<PeerId, PeerInfo>,
    last_consensus_round: u64,
    post_quantum_keys: PostQuantumKeyPair,
}

/// Robot data submitted to consensus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobotConsensusData {
    pub timestamp: u64,
    pub data_type: RobotDataType,
    pub robot_id: String,
    pub payload: RobotDataPayload,
    pub signature: Vec<u8>,
}

/// Types of robot data for consensus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RobotDataType {
    SensorReading,
    QuantumMeasurement,
    MissionStatus,
    SwarmCoordination,
    EnvironmentalAlert,
    MaintenanceReport,
}

/// Robot data payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RobotDataPayload {
    Sensor {
        temperature: f64,
        pressure: f64,
        ph: f64,
        dissolved_oxygen: f64,
        salinity: f64,
        quantum_coherence: f64,
    },
    Quantum {
        measurement_type: String,
        result: f64,
        uncertainty: f64,
        collapsed_state: usize,
    },
    Mission {
        mission_id: String,
        status: String,
        progress: f64,
        eta: Option<u64>,
    },
    Swarm {
        swarm_id: String,
        formation: String,
        cohesion: f64,
        entanglement_fidelity: f64,
    },
    Environment {
        alert_type: String,
        severity: String,
        location: (f64, f64, f64),
        description: String,
    },
    Maintenance {
        component: String,
        status: String,
        battery_level: f64,
        next_maintenance: u64,
    },
}

/// Consensus events for robot coordination
#[derive(Debug, Clone)]
pub enum ConsensusEvent {
    RobotDataConfirmed {
        robot_id: String,
        data_hash: Vec<u8>,
        consensus_round: u64,
    },
    SwarmCoordinationUpdate {
        swarm_id: String,
        new_formation: String,
        participants: Vec<String>,
    },
    EnvironmentalAlert {
        alert_id: String,
        alert_type: String,
        affected_robots: Vec<String>,
    },
    ConsensusStateUpdate {
        round: u64,
        finalized_blocks: u64,
        pending_transactions: u32,
    },
    PeerDiscovered {
        peer_id: PeerId,
        addresses: Vec<Multiaddr>,
        capabilities: Vec<String>,
    },
    PeerLost {
        peer_id: PeerId,
        last_seen: Instant,
    },
}

/// Post-quantum cryptographic key pair
#[derive(Clone)]
pub struct PostQuantumKeyPair {
    pub dilithium_keypair: (dilithium5::PublicKey, dilithium5::SecretKey),
    pub kyber_keypair: (kyber1024::PublicKey, kyber1024::SecretKey),
}

/// Peer information
#[derive(Clone)]
pub struct PeerInfo {
    pub peer_id: PeerId,
    pub addresses: Vec<Multiaddr>,
    pub capabilities: Vec<String>,
    pub last_seen: Instant,
    pub connection_quality: f64,
    pub post_quantum_public_key: Option<dilithium5::PublicKey>,
}

impl std::fmt::Debug for PeerInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PeerInfo")
            .field("peer_id", &self.peer_id)
            .field("addresses", &self.addresses)
            .field("capabilities", &self.capabilities)
            .field("last_seen", &self.last_seen)
            .field("connection_quality", &self.connection_quality)
            .field("post_quantum_public_key", &self.post_quantum_public_key.as_ref().map(|_| "PublicKey(...)"))
            .finish()
    }
}

/// Network behaviour for robot consensus integration
#[derive(NetworkBehaviour)]
pub struct RobotConsensusNetworkBehaviour {
    pub gossipsub: gossipsub::Behaviour,
    pub kademlia: libp2p::kad::Behaviour<libp2p::kad::store::MemoryStore>,
    pub identify: libp2p::identify::Behaviour,
}

impl ConsensusIntegration {
    /// Create new consensus integration
    pub async fn new(
        consensus_endpoint: String,
        robot_data_channel: mpsc::Receiver<RobotConsensusData>,
        consensus_events_channel: mpsc::Sender<ConsensusEvent>,
    ) -> Result<Self> {
        info!("Initializing consensus integration with endpoint: {}", consensus_endpoint);
        
        // Generate post-quantum key pairs
        let post_quantum_keys = Self::generate_post_quantum_keys().await?;
        
        // Create libp2p keypair
        let local_key = Keypair::generate_ed25519();
        let peer_id = PeerId::from(local_key.public());
        
        info!("Generated peer ID: {}", peer_id);
        
        // Initialize consensus node
        let consensus_node = ConsensusNode::new(consensus_endpoint.clone()).await
            .context("Failed to initialize consensus node")?;
        
        // Configure gossipsub
        let gossipsub_config = gossipsub::ConfigBuilder::default()
            .heartbeat_interval(Duration::from_secs(10))
            .validation_mode(ValidationMode::Strict)
            .message_id_fn(|message| {
                use sha3::{Digest, Sha3_256};
                let mut hasher = Sha3_256::new();
                hasher.update(&message.data);
                gossipsub::MessageId::from(hasher.finalize().to_vec())
            })
            .build()
            .context("Failed to build gossipsub config")?;
        
        let gossipsub = gossipsub::Behaviour::new(
            MessageAuthenticity::Signed(local_key.clone()),
            gossipsub_config,
        ).map_err(|e| anyhow::anyhow!("Failed to create gossipsub behaviour: {}", e))?;
        
        // Configure Kademlia for peer discovery
        let mut kademlia_config = libp2p::kad::Config::default();
        kademlia_config.set_query_timeout(Duration::from_secs(60));
        
        let store = libp2p::kad::store::MemoryStore::new(peer_id);
        let kademlia = libp2p::kad::Behaviour::with_config(peer_id, store, kademlia_config);
        
        // Configure identify protocol
        let identify_config = libp2p::identify::Config::new(
            "/q-robot-consensus/1.0.0".to_string(),
            local_key.public(),
        );
        let identify = libp2p::identify::Behaviour::new(identify_config);
        
        // Create network behaviour
        let behaviour = RobotConsensusNetworkBehaviour {
            gossipsub,
            kademlia,
            identify,
        };
        
        // Create swarm with SwarmBuilder API
        let mut swarm = libp2p::SwarmBuilder::with_existing_identity(local_key.clone())
            .with_tokio()
            .with_tcp(
                libp2p::tcp::Config::default(),
                libp2p::noise::Config::new,
                libp2p::yamux::Config::default,
            )?
            .with_behaviour(|_| behaviour)?
            .build();
        
        // Listen on all interfaces
        let listen_addr: Multiaddr = "/ip4/0.0.0.0/tcp/0".parse()
            .context("Failed to parse listen address")?;
        swarm.listen_on(listen_addr)
            .context("Failed to start listening")?;
        
        // Subscribe to consensus topics
        let consensus_topics = vec![
            "robot-sensor-data",
            "quantum-measurements", 
            "swarm-coordination",
            "environmental-alerts",
            "consensus-state-updates",
        ];
        
        for topic in consensus_topics {
            let topic = gossipsub::IdentTopic::new(topic);
            swarm.behaviour_mut().gossipsub.subscribe(&topic)
                .context(format!("Failed to subscribe to topic: {}", topic))?;
        }
        
        Ok(Self {
            consensus_node,
            swarm,
            peer_id,
            robot_data_channel,
            consensus_events_channel,
            known_peers: HashMap::new(),
            last_consensus_round: 0,
            post_quantum_keys,
        })
    }
    
    /// Run consensus integration event loop
    pub async fn run(&mut self) -> Result<()> {
        info!("Starting consensus integration event loop");
        
        let mut consensus_tick = tokio::time::interval(Duration::from_secs(5));
        
        loop {
            tokio::select! {
                // Handle swarm events
                event = self.swarm.select_next_some() => {
                    if let Err(e) = self.handle_swarm_event(event).await {
                        error!("Error handling swarm event: {}", e);
                    }
                }
                
                // Handle robot data submissions
                Some(robot_data) = self.robot_data_channel.recv() => {
                    if let Err(e) = self.submit_robot_data_to_consensus(robot_data).await {
                        error!("Error submitting robot data to consensus: {}", e);
                    }
                }
                
                // Periodic consensus state check
                _ = consensus_tick.tick() => {
                    if let Err(e) = self.check_consensus_state().await {
                        warn!("Error checking consensus state: {}", e);
                    }
                }
                
                // Handle consensus node events
                consensus_event = self.consensus_node.next_event() => {
                    if let Err(e) = self.handle_consensus_event(consensus_event).await {
                        error!("Error handling consensus event: {}", e);
                    }
                }
            }
        }
    }
    
    /// Submit robot data to consensus network
    pub async fn submit_robot_data(&mut self, data: RobotConsensusData) -> Result<()> {
        debug!("Submitting robot data from {} to consensus", data.robot_id);
        
        // Sign data with post-quantum signature
        let signed_data = self.sign_robot_data(&data).await?;
        
        // Create consensus message
        let consensus_message = ConsensusMessage::RobotData {
            data: signed_data,
            timestamp: chrono::Utc::now().timestamp() as u64,
        };
        
        // Submit to consensus node
        self.consensus_node.submit_message(consensus_message).await
            .context("Failed to submit robot data to consensus")?;
        
        // Broadcast via gossipsub
        let topic = match data.data_type {
            RobotDataType::SensorReading => "robot-sensor-data",
            RobotDataType::QuantumMeasurement => "quantum-measurements",
            RobotDataType::SwarmCoordination => "swarm-coordination",
            RobotDataType::EnvironmentalAlert => "environmental-alerts",
            _ => "robot-general-data",
        };
        
        let topic = gossipsub::IdentTopic::new(topic);
        let message = bincode::serialize(&data)
            .context("Failed to serialize robot data")?;
        
        self.swarm.behaviour_mut().gossipsub.publish(topic, message)
            .context("Failed to publish robot data to gossipsub")?;
        
        info!("Successfully submitted robot data from {} to consensus", data.robot_id);
        Ok(())
    }
    
    /// Query consensus for robot coordination data
    pub async fn query_robot_coordination(&mut self, query_type: &str) -> Result<Vec<RobotConsensusData>> {
        debug!("Querying consensus for robot coordination: {}", query_type);
        
        let query_result = match query_type {
            "active_robots" => {
                self.consensus_node.query_active_robots().await
                    .context("Failed to query active robots")?
            }
            "swarm_status" => {
                self.consensus_node.query_swarm_status().await
                    .context("Failed to query swarm status")?
            }
            "environmental_alerts" => {
                self.consensus_node.query_environmental_alerts().await
                    .context("Failed to query environmental alerts")?
            }
            _ => return Err(anyhow::anyhow!("Unknown query type: {}", query_type)),
        };
        
        Ok(query_result)
    }
    
    /// Connect to consensus network
    pub async fn connect_to_consensus(&mut self, bootstrap_peers: Vec<Multiaddr>) -> Result<()> {
        info!("Connecting to consensus network with {} bootstrap peers", bootstrap_peers.len());
        
        // Add bootstrap peers to Kademlia
        for addr in bootstrap_peers {
            if let Some(Protocol::P2p(peer_id)) = addr.iter().last() {
                // peer_id is already a PeerId from Protocol::P2p
                self.swarm.behaviour_mut().kademlia.add_address(&peer_id, addr.clone());

                // Attempt to dial peer
                if let Err(e) = self.swarm.dial(addr.clone()) {
                    warn!("Failed to dial bootstrap peer {}: {}", addr, e);
                } else {
                    debug!("Dialing bootstrap peer: {}", addr);
                }
            }
        }
        
        // Bootstrap Kademlia
        if let Err(e) = self.swarm.behaviour_mut().kademlia.bootstrap() {
            warn!("Kademlia bootstrap failed: {}", e);
        }
        
        // Connect to consensus node
        self.consensus_node.connect().await
            .context("Failed to connect to consensus node")?;
        
        info!("Successfully connected to consensus network");
        Ok(())
    }
    
    async fn handle_swarm_event(&mut self, event: SwarmEvent<RobotConsensusNetworkBehaviourEvent>) -> Result<()> {
        match event {
            SwarmEvent::NewListenAddr { address, .. } => {
                info!("Listening on address: {}", address);
            }
            SwarmEvent::Behaviour(event) => {
                self.handle_behaviour_event(event).await?;
            }
            SwarmEvent::ConnectionEstablished { peer_id, .. } => {
                info!("Connection established with peer: {}", peer_id);
                
                // Add peer to known peers
                self.known_peers.insert(peer_id, PeerInfo {
                    peer_id,
                    addresses: Vec::new(),
                    capabilities: Vec::new(),
                    last_seen: Instant::now(),
                    connection_quality: 1.0,
                    post_quantum_public_key: None,
                });
                
                // Send peer discovered event
                let event = ConsensusEvent::PeerDiscovered {
                    peer_id,
                    addresses: Vec::new(),
                    capabilities: Vec::new(),
                };
                
                if let Err(e) = self.consensus_events_channel.send(event).await {
                    warn!("Failed to send peer discovered event: {}", e);
                }
            }
            SwarmEvent::ConnectionClosed { peer_id, .. } => {
                info!("Connection closed with peer: {}", peer_id);
                
                if let Some(peer_info) = self.known_peers.remove(&peer_id) {
                    let event = ConsensusEvent::PeerLost {
                        peer_id,
                        last_seen: peer_info.last_seen,
                    };
                    
                    if let Err(e) = self.consensus_events_channel.send(event).await {
                        warn!("Failed to send peer lost event: {}", e);
                    }
                }
            }
            _ => {}
        }
        
        Ok(())
    }
    
    async fn handle_behaviour_event(&mut self, event: RobotConsensusNetworkBehaviourEvent) -> Result<()> {
        match event {
            RobotConsensusNetworkBehaviourEvent::Gossipsub(gossipsub::Event::Message {
                propagation_source,
                message_id: _,
                message,
            }) => {
                debug!("Received gossipsub message from: {}", propagation_source);
                
                // Deserialize and process robot data
                if let Ok(robot_data) = bincode::deserialize::<RobotConsensusData>(&message.data) {
                    self.process_received_robot_data(robot_data).await?;
                }
            }
            RobotConsensusNetworkBehaviourEvent::Kademlia(libp2p::kad::Event::RoutingUpdated {
                peer,
                addresses,
                ..
            }) => {
                debug!("Kademlia routing updated for peer: {}", peer);
                
                if let Some(peer_info) = self.known_peers.get_mut(&peer) {
                    // Addresses is not an iterator, convert to vec
                    peer_info.addresses = addresses.iter().cloned().collect();
                    peer_info.last_seen = Instant::now();
                }
            }
            RobotConsensusNetworkBehaviourEvent::Identify(libp2p::identify::Event::Received {
                peer_id,
                info,
                connection_id: _,
            }) => {
                debug!("Received identify info from peer: {}", peer_id);
                
                if let Some(peer_info) = self.known_peers.get_mut(&peer_id) {
                    peer_info.addresses = info.listen_addrs;
                    peer_info.last_seen = Instant::now();
                    
                    // Parse capabilities from agent version or protocols
                    peer_info.capabilities = info.protocols.into_iter()
                        .filter(|proto| proto.as_ref().starts_with("/q-robot/"))
                        .map(|proto| proto.to_string())
                        .collect();
                }
            }
            _ => {}
        }
        
        Ok(())
    }
    
    async fn handle_consensus_event(&mut self, event: ConsensusEvent) -> Result<()> {
        match event {
            ConsensusEvent::RobotDataConfirmed { robot_id, data_hash, consensus_round } => {
                debug!("Robot data confirmed for {}: {:?} in round {}", robot_id, data_hash, consensus_round);
                self.last_consensus_round = consensus_round;
            }
            ConsensusEvent::SwarmCoordinationUpdate { swarm_id, new_formation, participants } => {
                debug!("Swarm coordination update for {}: formation={}, participants={:?}", swarm_id, new_formation, participants);
            }
            ConsensusEvent::EnvironmentalAlert { alert_id, alert_type, affected_robots } => {
                warn!("Environmental alert {}: type={}, affected robots={:?}", alert_id, alert_type, affected_robots);
            }
            ConsensusEvent::ConsensusStateUpdate { round, finalized_blocks, pending_transactions } => {
                debug!("Consensus state update round {}: finalized={}, pending={}", round, finalized_blocks, pending_transactions);
                self.last_consensus_round = round;
            }
            ConsensusEvent::PeerDiscovered { peer_id, addresses, capabilities } => {
                info!("Peer discovered {}: {:?}, capabilities: {:?}", peer_id, addresses, capabilities);
            }
            ConsensusEvent::PeerLost { peer_id, last_seen } => {
                warn!("Peer lost: {}, last seen: {:?}", peer_id, last_seen);
            }
        }
        
        Ok(())
    }
    
    async fn submit_robot_data_to_consensus(&mut self, data: RobotConsensusData) -> Result<()> {
        self.submit_robot_data(data).await
    }
    
    async fn process_received_robot_data(&mut self, data: RobotConsensusData) -> Result<()> {
        debug!("Processing received robot data from: {}", data.robot_id);
        
        // Verify post-quantum signature
        if !self.verify_robot_data_signature(&data).await? {
            warn!("Invalid signature on robot data from: {}", data.robot_id);
            return Ok(());
        }
        
        // Process based on data type
        match &data.payload {
            RobotDataPayload::Swarm { swarm_id, formation, .. } => {
                let event = ConsensusEvent::SwarmCoordinationUpdate {
                    swarm_id: swarm_id.clone(),
                    new_formation: formation.clone(),
                    participants: vec![data.robot_id.clone()],
                };
                
                if let Err(e) = self.consensus_events_channel.send(event).await {
                    warn!("Failed to send swarm coordination update: {}", e);
                }
            }
            RobotDataPayload::Environment { alert_type, .. } => {
                let event = ConsensusEvent::EnvironmentalAlert {
                    alert_id: format!("{}_{}", data.robot_id, data.timestamp),
                    alert_type: alert_type.clone(),
                    affected_robots: vec![data.robot_id.clone()],
                };
                
                if let Err(e) = self.consensus_events_channel.send(event).await {
                    warn!("Failed to send environmental alert: {}", e);
                }
            }
            _ => {
                debug!("Processed general robot data from: {}", data.robot_id);
            }
        }
        
        Ok(())
    }
    
    async fn check_consensus_state(&mut self) -> Result<()> {
        let state = self.consensus_node.get_state().await
            .context("Failed to get consensus state")?;
        
        if state.current_round > self.last_consensus_round {
            debug!("Consensus advanced to round: {}", state.current_round);
            self.last_consensus_round = state.current_round;
            
            let event = ConsensusEvent::ConsensusStateUpdate {
                round: state.current_round,
                finalized_blocks: state.finalized_blocks,
                pending_transactions: state.pending_transactions,
            };
            
            if let Err(e) = self.consensus_events_channel.send(event).await {
                warn!("Failed to send consensus state update: {}", e);
            }
        }
        
        Ok(())
    }
    
    async fn generate_post_quantum_keys() -> Result<PostQuantumKeyPair> {
        info!("Generating post-quantum cryptographic keys");
        
        // Generate Dilithium5 key pair for signatures
        let (dilithium_pk, dilithium_sk) = dilithium5::keypair();
        
        // Generate Kyber1024 key pair for key exchange
        let (kyber_pk, kyber_sk) = kyber1024::keypair();
        
        Ok(PostQuantumKeyPair {
            dilithium_keypair: (dilithium_pk, dilithium_sk),
            kyber_keypair: (kyber_pk, kyber_sk),
        })
    }
    
    async fn sign_robot_data(&self, data: &RobotConsensusData) -> Result<RobotConsensusData> {
        // Serialize data without signature
        let mut unsigned_data = data.clone();
        unsigned_data.signature = Vec::new();
        
        let message = bincode::serialize(&unsigned_data)
            .context("Failed to serialize robot data for signing")?;
        
        // Sign with Dilithium5
        let signature = dilithium5::sign(&message, &self.post_quantum_keys.dilithium_keypair.1);

        let mut signed_data = data.clone();
        signed_data.signature = signature.as_bytes().to_vec();
        
        Ok(signed_data)
    }
    
    async fn verify_robot_data_signature(&self, data: &RobotConsensusData) -> Result<bool> {
        // For verification, we would need the sender's public key
        // In a real implementation, this would be retrieved from a key registry
        // For simulation, we'll return true
        Ok(true)
    }
    
    fn find_robot_data_by_hash(&self, _hash: &[u8]) -> Option<(String, RobotConsensusData)> {
        // In a real implementation, this would maintain a mapping of hashes to robot data
        // For simulation, we'll return None
        None
    }
}

// Re-export types for use in other modules
pub use RobotConsensusNetworkBehaviourEvent::*;