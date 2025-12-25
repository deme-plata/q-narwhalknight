/// Real Production DHT Implementation using libp2p Kademlia
///
/// This replaces all mock implementations with actual network connectivity.
use anyhow::{anyhow, Result};
use futures::prelude::*;
use libp2p::{
    core::upgrade,
    identify, identity,
    kad::{
        self, Behaviour as KademliaBehaviour, Config as KademliaConfig,
        Event as KademliaEvent, QueryResult, Record, RecordKey,
        GetRecordOk, GetProvidersOk
    },
    multiaddr::Protocol,
    noise, ping,
    swarm::{SwarmEvent, NetworkBehaviour},
    SwarmBuilder,
    tcp, yamux, Multiaddr, PeerId, Swarm, Transport,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    error::Error,
    time::Duration,
};
use tokio::{
    sync::{broadcast, mpsc},
    time::{interval, sleep},
};
use tracing::{debug, error, info, warn};

/// Real peer information structure
#[derive(Debug, Clone)]
pub struct PeerInfo {
    pub peer_id: PeerId,
    pub addresses: Vec<Multiaddr>,
    pub onion_address: Option<String>,
    pub capabilities: Vec<String>,
    pub last_seen: std::time::SystemTime,
    pub rtt: Option<Duration>,
}

/// DHT events for the application layer
#[derive(Debug, Clone)]
pub enum DhtEvent {
    PeerDiscovered(PeerInfo),
    PeerConnected(PeerId),
    PeerDisconnected(PeerId),
    RecordFound { key: String, value: Vec<u8> },
    RecordStored { key: String },
}

/// Production DHT configuration
#[derive(Debug, Clone)]
pub struct RealDhtConfig {
    pub listen_addresses: Vec<Multiaddr>,
    pub bootstrap_peers: Vec<(PeerId, Multiaddr)>,
    pub enable_mdns: bool,
    pub kad_protocol_name: Vec<u8>,
    pub record_ttl: Duration,
    pub query_timeout: Duration,
    pub connection_idle_timeout: Duration,
}

impl Default for RealDhtConfig {
    fn default() -> Self {
        Self {
            listen_addresses: vec![
                "/ip4/0.0.0.0/tcp/0".parse().unwrap(),
                "/ip6/::/tcp/0".parse().unwrap(),
            ],
            bootstrap_peers: vec![],
            enable_mdns: true,
            kad_protocol_name: b"/q-narwhal/kad/1.0.0".to_vec(),
            record_ttl: Duration::from_secs(3600), // 1 hour
            query_timeout: Duration::from_secs(10),
            connection_idle_timeout: Duration::from_secs(60),
        }
    }
}

/// Real production DHT implementation
pub struct RealDht {
    swarm: Swarm<DhtBehaviour>,
    event_sender: broadcast::Sender<DhtEvent>,
    command_receiver: mpsc::Receiver<DhtCommand>,
    command_sender: mpsc::Sender<DhtCommand>,
    config: RealDhtConfig,
    connected_peers: HashMap<PeerId, PeerInfo>,
    stored_records: HashMap<String, Vec<u8>>,
}

/// Commands that can be sent to the DHT
#[derive(Debug)]
pub enum DhtCommand {
    StartListening,
    Bootstrap,
    FindPeer(PeerId),
    GetRecord(String),
    PutRecord { key: String, value: Vec<u8> },
    GetProviders(String),
    StartProviding(String),
    ConnectToPeer { peer_id: PeerId, address: Multiaddr },
}

/// Combined network behavior for the DHT
#[derive(libp2p::swarm::NetworkBehaviour)]
#[behaviour(to_swarm = "DhtBehaviourEvent")]
struct DhtBehaviour {
    kademlia: KademliaBehaviour<libp2p::kad::store::MemoryStore>,
    identify: identify::Behaviour,
    ping: ping::Behaviour,
}

#[derive(Debug)]
enum DhtBehaviourEvent {
    Kademlia(KademliaEvent),
    Identify(identify::Event),
    Ping(ping::Event),
}

impl From<KademliaEvent> for DhtBehaviourEvent {
    fn from(event: KademliaEvent) -> Self {
        DhtBehaviourEvent::Kademlia(event)
    }
}

impl From<identify::Event> for DhtBehaviourEvent {
    fn from(event: identify::Event) -> Self {
        DhtBehaviourEvent::Identify(event)
    }
}

impl From<ping::Event> for DhtBehaviourEvent {
    fn from(event: ping::Event) -> Self {
        DhtBehaviourEvent::Ping(event)
    }
}

impl RealDht {
    /// Create a new real DHT instance
    pub async fn new(config: RealDhtConfig) -> Result<Self> {
        // Generate a new identity for this node
        let local_key = identity::Keypair::generate_ed25519();
        let local_peer_id = PeerId::from(local_key.public());

        info!("Creating real DHT node with peer ID: {}", local_peer_id);

        // Create transport
        // v1.4.14-beta: Enable TCP_NODELAY for lower latency (disables Nagle's algorithm)
        let transport = tcp::tokio::Transport::new(tcp::Config::default().nodelay(true))
            .upgrade(upgrade::Version::V1Lazy)
            .authenticate(noise::Config::new(&local_key)?)
            .multiplex(yamux::Config::default())
            .timeout(Duration::from_secs(20))
            .boxed();

        // Configure Kademlia DHT
        let mut kad_config = KademliaConfig::default();
        kad_config.set_query_timeout(config.query_timeout);
        kad_config.set_record_ttl(Some(config.record_ttl));
        kad_config.set_provider_record_ttl(Some(config.record_ttl));
        kad_config.set_publication_interval(Some(Duration::from_secs(600))); // 10 minutes

        let store = libp2p::kad::store::MemoryStore::new(local_peer_id);
        let mut kademlia = KademliaBehaviour::with_config(
            local_peer_id,
            store,
            kad_config,
        );

        // Note: Protocol names are set during construction in newer libp2p versions
        // kademlia.set_protocol_names(vec![std::str::from_utf8(&config.kad_protocol_name)?]);
        debug!("Using default Kademlia protocol names");

        // Create network behavior
        let behaviour = DhtBehaviour {
            kademlia,
            identify: identify::Behaviour::new(identify::Config::new(
                "/qnarwhal/1.0.0".to_string(),  // v1.0.2-beta: Standardized protocol ID for P2P compatibility
                local_key.public(),
            )),
            ping: ping::Behaviour::new(ping::Config::new()),
        };

        // Create swarm using libp2p 0.53 API
        // v1.4.14-beta: Enable TCP_NODELAY for lower latency
        let transport = tcp::tokio::Transport::new(tcp::Config::default().nodelay(true))
            .upgrade(libp2p::core::upgrade::Version::V1)
            .authenticate(noise::Config::new(&local_key)?)
            .multiplex(yamux::Config::default())
            .boxed();

        let swarm_config = libp2p::swarm::Config::with_tokio_executor()
            .with_idle_connection_timeout(config.connection_idle_timeout);

        let swarm = Swarm::new(transport, behaviour, local_key.public().to_peer_id(), swarm_config);

        // Create communication channels
        let (event_sender, _) = broadcast::channel(1000);
        let (command_sender, command_receiver) = mpsc::channel(100);

        Ok(Self {
            swarm,
            event_sender,
            command_receiver,
            command_sender,
            config,
            connected_peers: HashMap::new(),
            stored_records: HashMap::new(),
        })
    }

    /// Get a handle for sending commands to the DHT
    pub fn command_sender(&self) -> mpsc::Sender<DhtCommand> {
        self.command_sender.clone()
    }

    /// Subscribe to DHT events
    pub fn subscribe_events(&self) -> broadcast::Receiver<DhtEvent> {
        self.event_sender.subscribe()
    }

    /// Get the local peer ID
    pub fn local_peer_id(&self) -> PeerId {
        *self.swarm.local_peer_id()
    }

    /// Get listening addresses
    pub fn listening_addresses(&self) -> Vec<Multiaddr> {
        self.swarm.listeners().cloned().collect()
    }

    /// Get connected peers
    pub fn connected_peers(&self) -> &HashMap<PeerId, PeerInfo> {
        &self.connected_peers
    }

    /// Run the DHT event loop
    pub async fn run(&mut self) -> Result<()> {
        info!("Starting real DHT node");

        // Start listening on configured addresses
        for addr in &self.config.listen_addresses {
            match self.swarm.listen_on(addr.clone()) {
                Ok(_) => info!("Listening on: {}", addr),
                Err(e) => warn!("Failed to listen on {}: {}", addr, e),
            }
        }

        // Bootstrap with configured peers
        for (peer_id, addr) in &self.config.bootstrap_peers {
            self.swarm.behaviour_mut().kademlia.add_address(peer_id, addr.clone());
            info!("Added bootstrap peer: {} at {}", peer_id, addr);
        }

        let mut bootstrap_timer = interval(Duration::from_secs(60));
        let mut maintenance_timer = interval(Duration::from_secs(300)); // 5 minutes

        loop {
            tokio::select! {
                // Handle swarm events
                event = self.swarm.select_next_some() => {
                    self.handle_swarm_event(event).await?;
                }

                // Handle commands
                command = self.command_receiver.recv() => {
                    if let Some(cmd) = command {
                        self.handle_command(cmd).await?;
                    }
                }

                // Periodic bootstrap
                _ = bootstrap_timer.tick() => {
                    if !self.config.bootstrap_peers.is_empty() {
                        debug!("Running periodic bootstrap");
                        if let Err(e) = self.swarm.behaviour_mut().kademlia.bootstrap() {
                            warn!("Bootstrap failed: {}", e);
                        }
                    }
                }

                // Maintenance tasks
                _ = maintenance_timer.tick() => {
                    self.run_maintenance().await?;
                }
            }
        }
    }

    async fn handle_swarm_event(&mut self, event: SwarmEvent<DhtBehaviourEvent>) -> Result<()> {
        match event {
            SwarmEvent::NewListenAddr { address, .. } => {
                info!("DHT listening on: {}", address);
            }

            SwarmEvent::ConnectionEstablished { peer_id, endpoint, .. } => {
                info!("Connected to peer: {} via {}", peer_id, endpoint.get_remote_address());
                
                // Create peer info
                let peer_info = PeerInfo {
                    peer_id,
                    addresses: vec![endpoint.get_remote_address().clone()],
                    onion_address: None,
                    capabilities: vec![],
                    last_seen: std::time::SystemTime::now(),
                    rtt: None,
                };

                self.connected_peers.insert(peer_id, peer_info.clone());
                let _ = self.event_sender.send(DhtEvent::PeerConnected(peer_id));
            }

            SwarmEvent::ConnectionClosed { peer_id, cause, .. } => {
                info!("Disconnected from peer: {} (cause: {:?})", peer_id, cause);
                self.connected_peers.remove(&peer_id);
                let _ = self.event_sender.send(DhtEvent::PeerDisconnected(peer_id));
            }

            SwarmEvent::Behaviour(event) => {
                // Handle behavior events using enum pattern matching
                match event {
                    DhtBehaviourEvent::Kademlia(kad_event) => {
                        self.handle_kademlia_event(kad_event).await?;
                    }
                    DhtBehaviourEvent::Identify(identify_event) => {
                        self.handle_identify_event(identify_event).await?;
                    }
                    DhtBehaviourEvent::Ping(ping_event) => {
                        self.handle_ping_event(ping_event).await?;
                    }
                }
            }

            SwarmEvent::OutgoingConnectionError { peer_id, error, connection_id: _ } => {
                if let Some(peer_id) = peer_id {
                    warn!("Failed to connect to peer {}: {}", peer_id, error);
                }
            }

            _ => {}
        }

        Ok(())
    }

    async fn handle_kademlia_event(&mut self, event: KademliaEvent) -> Result<()> {
        match event {
            KademliaEvent::RoutingUpdated { peer, .. } => {
                debug!("Routing table updated with peer: {}", peer);
            }

            KademliaEvent::UnroutablePeer { peer } => {
                warn!("Peer is unroutable: {}", peer);
            }

            KademliaEvent::RoutablePeer { peer, address } => {
                info!("Discovered routable peer: {} at {}", peer, address);
                
                let peer_info = PeerInfo {
                    peer_id: peer,
                    addresses: vec![address],
                    onion_address: None,
                    capabilities: vec![],
                    last_seen: std::time::SystemTime::now(),
                    rtt: None,
                };

                let _ = self.event_sender.send(DhtEvent::PeerDiscovered(peer_info));
            }

            KademliaEvent::OutboundQueryProgressed { result, .. } => {
                match result {
                    QueryResult::GetRecord(Ok(record)) => {
                        // Note: GetRecordOk structure changed in newer libp2p versions
                        info!("Record retrieval succeeded");
                        // TODO: Update to use correct GetRecordOk API once libp2p is stable
                    }

                    QueryResult::GetRecord(Err(e)) => {
                        warn!("Failed to get record: {:?}", e);
                    }

                    QueryResult::PutRecord(Ok(_)) => {
                        info!("Successfully stored record in DHT");
                    }

                    QueryResult::PutRecord(Err(e)) => {
                        warn!("Failed to store record: {:?}", e);
                    }

                    QueryResult::GetClosestPeers(Ok(peers)) => {
                        info!("Found {} closest peers", peers.peers.len());
                        // 🔥 v2.0.0: PeerInfo doesn't impl Display, use Debug
                        for peer in peers.peers {
                            debug!("Closest peer: {:?}", peer);
                        }
                    }

                    QueryResult::GetProviders(Ok(providers)) => {
                        info!("Provider discovery succeeded");
                        // TODO: Update to use correct GetProvidersOk API once libp2p is stable
                    }

                    QueryResult::StartProviding(Ok(_)) => {
                        info!("Successfully started providing");
                    }

                    _ => {}
                }
            }

            _ => {}
        }

        Ok(())
    }

    async fn handle_identify_event(&mut self, event: identify::Event) -> Result<()> {
        match event {
            identify::Event::Received { peer_id, info, connection_id: _ } => {
                info!("Identified peer: {} - {}", peer_id, info.protocol_version);
                
                // Update peer info with identification data
                if let Some(peer_info) = self.connected_peers.get_mut(&peer_id) {
                    peer_info.addresses = info.listen_addrs.clone();
                    peer_info.capabilities = vec![info.protocol_version];
                }

                // Add addresses to Kademlia
                for addr in &info.listen_addrs {
                    self.swarm.behaviour_mut().kademlia.add_address(&peer_id, addr.clone());
                }
            }

            identify::Event::Sent { .. } => {
                debug!("Sent identify info");
            }

            identify::Event::Error { peer_id, error, connection_id: _ } => {
                warn!("Identify error with peer {:?}: {}", peer_id, error);
            }

            identify::Event::Pushed { .. } => {
                debug!("Pushed identify update");
            }
        }

        Ok(())
    }

    async fn handle_ping_event(&mut self, event: ping::Event) -> Result<()> {
        match event {
            ping::Event { peer, result, connection: _ } => {
                match result {
                    Ok(rtt) => {
                        debug!("Ping to {} succeeded: {:?}", peer, rtt);
                        // Update RTT in peer info
                        if let Some(peer_info) = self.connected_peers.get_mut(&peer) {
                            peer_info.rtt = Some(rtt);
                            peer_info.last_seen = std::time::SystemTime::now();
                        }
                    }
                    Err(failure) => {
                        warn!("Ping to {} failed: {:?}", peer, failure);
                    }
                }
            }
        }

        Ok(())
    }

    async fn handle_command(&mut self, command: DhtCommand) -> Result<()> {
        match command {
            DhtCommand::StartListening => {
                // Already listening, but could restart if needed
                info!("DHT already listening");
            }

            DhtCommand::Bootstrap => {
                info!("Starting DHT bootstrap");
                if let Err(e) = self.swarm.behaviour_mut().kademlia.bootstrap() {
                    error!("Bootstrap failed: {}", e);
                    return Err(anyhow!("Bootstrap failed: {}", e));
                }
            }

            DhtCommand::FindPeer(peer_id) => {
                info!("Finding peer: {}", peer_id);
                self.swarm.behaviour_mut().kademlia.get_closest_peers(peer_id);
            }

            DhtCommand::GetRecord(key) => {
                info!("Getting record: {}", key);
                let record_key = libp2p::kad::RecordKey::new(&key);
                self.swarm.behaviour_mut().kademlia.get_record(record_key);
            }

            DhtCommand::PutRecord { key, value } => {
                info!("Storing record: {} ({} bytes)", key, value.len());
                let record_key = libp2p::kad::RecordKey::new(&key);
                let record = Record::new(record_key, value.clone());
                if let Err(e) = self.swarm.behaviour_mut().kademlia.put_record(record, libp2p::kad::Quorum::One) {
                    error!("Failed to store record: {}", e);
                } else {
                    self.stored_records.insert(key.clone(), value);
                    let _ = self.event_sender.send(DhtEvent::RecordStored { key });
                }
            }

            DhtCommand::GetProviders(key) => {
                info!("Getting providers for: {}", key);
                let record_key = libp2p::kad::RecordKey::new(&key);
                self.swarm.behaviour_mut().kademlia.get_providers(record_key);
            }

            DhtCommand::StartProviding(key) => {
                info!("Starting to provide: {}", key);
                let record_key = libp2p::kad::RecordKey::new(&key);
                if let Err(e) = self.swarm.behaviour_mut().kademlia.start_providing(record_key) {
                    error!("Failed to start providing: {}", e);
                }
            }

            DhtCommand::ConnectToPeer { peer_id, address } => {
                info!("Connecting to peer: {} at {}", peer_id, address);
                self.swarm.behaviour_mut().kademlia.add_address(&peer_id, address.clone());
                if let Err(e) = self.swarm.dial(address) {
                    error!("Failed to dial peer: {}", e);
                }
            }
        }

        Ok(())
    }

    async fn run_maintenance(&mut self) -> Result<()> {
        debug!("Running DHT maintenance tasks");

        // Clean up old peer information
        let cutoff = std::time::SystemTime::now() - Duration::from_secs(3600); // 1 hour
        let mut to_remove = Vec::new();

        for (peer_id, peer_info) in &self.connected_peers {
            if peer_info.last_seen < cutoff {
                to_remove.push(*peer_id);
            }
        }

        for peer_id in to_remove {
            self.connected_peers.remove(&peer_id);
            debug!("Removed stale peer: {}", peer_id);
        }

        // Re-publish our stored records
        for (key, value) in &self.stored_records {
            let record_key = libp2p::kad::RecordKey::new(key);
            let record = Record::new(record_key, value.clone());
            if let Err(e) = self.swarm.behaviour_mut().kademlia.put_record(record, libp2p::kad::Quorum::One) {
                warn!("Failed to re-publish record {}: {}", key, e);
            }
        }

        Ok(())
    }
}

/// Helper function to create a DHT with bootstrap peers
pub async fn create_production_dht(
    bootstrap_peers: Vec<String>, 
    listen_port: Option<u16>
) -> Result<RealDht> {
    let mut config = RealDhtConfig::default();
    
    // Set custom listen port if provided
    if let Some(port) = listen_port {
        config.listen_addresses = vec![
            format!("/ip4/0.0.0.0/tcp/{}", port).parse()?,
            format!("/ip6/::/tcp/{}", port).parse()?,
        ];
    }

    // Parse bootstrap peers
    for peer_str in bootstrap_peers {
        // Expected format: "peer_id@multiaddr"
        if let Some((peer_id_str, addr_str)) = peer_str.split_once('@') {
            if let (Ok(peer_id), Ok(addr)) = (peer_id_str.parse::<PeerId>(), addr_str.parse::<Multiaddr>()) {
                config.bootstrap_peers.push((peer_id, addr));
            } else {
                warn!("Invalid bootstrap peer format: {}", peer_str);
            }
        }
    }

    RealDht::new(config).await
}

/// Production DHT manager that handles multiple DHT instances
pub struct DhtManager {
    dhts: HashMap<String, mpsc::Sender<DhtCommand>>,
    event_receiver: broadcast::Receiver<DhtEvent>,
}

impl DhtManager {
    pub fn new() -> Self {
        let (_, event_receiver) = broadcast::channel(1000);
        Self {
            dhts: HashMap::new(),
            event_receiver,
        }
    }

    pub async fn add_dht(&mut self, name: String, dht: RealDht) -> Result<()> {
        let command_sender = dht.command_sender();
        self.dhts.insert(name.clone(), command_sender);
        
        // Spawn the DHT event loop
        tokio::spawn(async move {
            let mut dht = dht;
            if let Err(e) = dht.run().await {
                error!("DHT {} failed: {}", name, e);
            }
        });
        
        Ok(())
    }

    pub async fn send_command(&self, dht_name: &str, command: DhtCommand) -> Result<()> {
        if let Some(sender) = self.dhts.get(dht_name) {
            sender.send(command).await?;
            Ok(())
        } else {
            Err(anyhow!("DHT {} not found", dht_name))
        }
    }
}