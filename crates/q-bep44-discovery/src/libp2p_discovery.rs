// REAL libp2p-rust bootstrap implementation for Q-NarwhalKnight
// Target: 185.182.185.227:6881 bootstrap connectivity using Kademlia DHT

use anyhow::{Result, Context};
use tracing::{debug, info, warn, error};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::net::{SocketAddr, IpAddr, Ipv4Addr};
use std::time::Duration;
use futures::StreamExt;

// libp2p imports for real DHT implementation - using workspace libp2p 0.53
use libp2p::{
    Swarm, SwarmBuilder, Transport,
    identity, PeerId, Multiaddr,
    swarm::{NetworkBehaviour, SwarmEvent},
    tcp::Config as TcpConfig,
    noise::Config as NoiseConfig,
    yamux::Config as YamuxConfig,
    // Use libp2p 0.53's re-exported sub-crates
    kad::{Behaviour as Kademlia, Event as KademliaEvent, QueryResult, GetClosestPeersOk},
    identify::{Behaviour as Identify, Event as IdentifyEvent},
    gossipsub::{Behaviour as Gossipsub, Event as GossipsubEvent, IdentTopic, MessageAuthenticity, ConfigBuilder as GossipsubConfigBuilder},
};

use crate::{QnkDhtPeer, QnkDhtConfig};

/// Real libp2p DHT client for Q-NarwhalKnight bootstrap connectivity
#[derive(NetworkBehaviour)]
#[behaviour(to_swarm = "QnkNetworkBehaviourEvent")]
pub struct QnkNetworkBehaviour {
    pub kademlia: Kademlia<libp2p_kad::store::MemoryStore>,
    pub identify: Identify,
    pub gossipsub: Gossipsub,
}

// Define the event enum manually for the network behaviour
#[derive(Debug)]
pub enum QnkNetworkBehaviourEvent {
    Kademlia(libp2p_kad::Event),
    Identify(libp2p_identify::Event),
    Gossipsub(libp2p_gossipsub::Event),
}

impl From<libp2p_kad::Event> for QnkNetworkBehaviourEvent {
    fn from(event: libp2p_kad::Event) -> Self {
        QnkNetworkBehaviourEvent::Kademlia(event)
    }
}

impl From<libp2p_identify::Event> for QnkNetworkBehaviourEvent {
    fn from(event: libp2p_identify::Event) -> Self {
        QnkNetworkBehaviourEvent::Identify(event)
    }
}

impl From<libp2p_gossipsub::Event> for QnkNetworkBehaviourEvent {
    fn from(event: libp2p_gossipsub::Event) -> Self {
        QnkNetworkBehaviourEvent::Gossipsub(event)
    }
}

pub struct LibP2PDiscoveryClient {
    config: QnkDhtConfig,
    local_validator_id: [u8; 32],
    discovered_peers: Arc<RwLock<Vec<QnkDhtPeer>>>,
    is_running: Arc<RwLock<bool>>,
    swarm: Option<Swarm<QnkNetworkBehaviour>>,
    bootstrap_addresses: Vec<SocketAddr>,
    // QUANTUM ENHANCEMENT: Quantum peer selector for optimal connections
    quantum_selector: Option<Arc<crate::quantum_peer_selection::QuantumPeerSelector>>,
}

// Manual Debug implementation to skip the swarm field
impl std::fmt::Debug for LibP2PDiscoveryClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LibP2PDiscoveryClient")
            .field("config", &self.config)
            .field("local_validator_id", &hex::encode(&self.local_validator_id))
            .field("discovered_peers", &self.discovered_peers)
            .field("is_running", &self.is_running)
            .field("swarm", &"<Swarm>")
            .field("bootstrap_addresses", &self.bootstrap_addresses)
            .field("quantum_selector", &"<QuantumPeerSelector>")
            .finish()
    }
}

impl LibP2PDiscoveryClient {
    pub async fn new(config: QnkDhtConfig, local_validator_id: [u8; 32]) -> Result<Self> {
        println!("🚨🚨🚨 LIBP2P DEBUG: Creating real libp2p discovery client for Q-NarwhalKnight");
        info!("🌐 LIBP2P: Creating real libp2p discovery client for Q-NarwhalKnight");

        // Use bootstrap addresses from config, or fall back to defaults
        let bootstrap_addresses = if !config.bootstrap_nodes.is_empty() {
            println!("🚨🚨🚨 LIBP2P DEBUG: Using {} bootstrap nodes from config", config.bootstrap_nodes.len());
            info!("🎯 LIBP2P: Using {} bootstrap nodes from config", config.bootstrap_nodes.len());
            for addr in &config.bootstrap_nodes {
                info!("  • Bootstrap node: {}", addr);
            }
            config.bootstrap_nodes.clone()
        } else {
            println!("🚨🚨🚨 LIBP2P DEBUG: No bootstrap nodes in config, using defaults");
            info!("⚠️ LIBP2P: No bootstrap nodes configured, using default public DHT nodes");
            vec![
                "87.98.162.88:6881".parse().unwrap(),     // router.bittorrent.com
                "212.129.33.59:6881".parse().unwrap(),    // dht.transmissionbt.com
                "82.221.103.244:6881".parse().unwrap(),   // router.utorrent.com
            ]
        };

        // Initialize quantum peer selector for optimal connection selection
        info!("⚛️ Initializing quantum-enhanced peer selection system...");
        let quantum_selector = Arc::new(crate::quantum_peer_selection::QuantumPeerSelector::new(20));
        info!("✨ Quantum peer selector ready: superposition-based quality measurement enabled");

        let mut client = Self {
            config,
            local_validator_id,
            discovered_peers: Arc::new(RwLock::new(Vec::new())),
            is_running: Arc::new(RwLock::new(false)),
            swarm: None,
            bootstrap_addresses,
            quantum_selector: Some(quantum_selector),
        };

        // Automatically initialize and start the libp2p swarm
        info!("🚀 LIBP2P: Auto-initializing swarm on creation");
        client.initialize().await?;

        // Spawn background task to run the event loop
        let mut swarm = client.swarm.take().ok_or_else(|| {
            anyhow::anyhow!("Swarm was not initialized")
        })?;

        let discovered_peers = client.discovered_peers.clone();
        let is_running = client.is_running.clone();
        let bootstrap_addresses = client.bootstrap_addresses.clone();
        let client_quantum_selector = client.quantum_selector.clone();

        println!("🚨🚨🚨 LIBP2P DEBUG: About to spawn event loop with {} bootstrap addresses", bootstrap_addresses.len());
        for (i, addr) in bootstrap_addresses.iter().enumerate() {
            println!("🚨🚨🚨 LIBP2P DEBUG: Bootstrap[{}]: {}", i, addr);
        }

        tokio::spawn(async move {
            println!("🚨🚨🚨 LIBP2P DEBUG: Event loop task STARTED");
            info!("🌐 LIBP2P: Event loop started in background task");
            *is_running.write().await = true;

            // Bootstrap immediately
            println!("🚨🚨🚨 LIBP2P DEBUG: Calling kademlia.bootstrap()...");
            if let Err(e) = swarm.behaviour_mut().kademlia.bootstrap() {
                warn!("⚠️ LIBP2P: Initial bootstrap failed: {}", e);
                println!("🚨🚨🚨 LIBP2P DEBUG: Bootstrap error: {}", e);
            } else {
                println!("🚨🚨🚨 LIBP2P DEBUG: kademlia.bootstrap() succeeded");
            }

            // Dial bootstrap nodes
            println!("🚨🚨🚨 LIBP2P DEBUG: About to start dial loop for {} addresses", bootstrap_addresses.len());
            for (i, bootstrap_address) in bootstrap_addresses.iter().enumerate() {
                println!("🚨🚨🚨 LIBP2P DEBUG: Dial loop iteration {}, address: {}", i, bootstrap_address);
                let multiaddr_string = format!(
                    "/ip4/{}/tcp/{}",
                    bootstrap_address.ip(),
                    bootstrap_address.port()
                );
                println!("🚨🚨🚨 LIBP2P DEBUG: Constructed multiaddr string: {}", multiaddr_string);

                let bootstrap_multiaddr: Multiaddr = match multiaddr_string.parse() {
                    Ok(addr) => {
                        println!("🚨🚨🚨 LIBP2P DEBUG: Successfully parsed multiaddr: {}", addr);
                        addr
                    },
                    Err(e) => {
                        println!("🚨🚨🚨 LIBP2P DEBUG: PARSE ERROR: {}", e);
                        warn!("⚠️ LIBP2P: Failed to parse multiaddr {}: {}", multiaddr_string, e);
                        continue;
                    }
                };

                println!("🚨🚨🚨 LIBP2P DEBUG: About to call swarm.dial()...");
                info!("📞 LIBP2P: Dialing bootstrap node: {}", bootstrap_multiaddr);
                match swarm.dial(bootstrap_multiaddr.clone()) {
                    Ok(_) => {
                        println!("🚨🚨🚨 LIBP2P DEBUG: Dial succeeded for {}", bootstrap_multiaddr);
                    },
                    Err(e) => {
                        println!("🚨🚨🚨 LIBP2P DEBUG: Dial FAILED for {}: {}", bootstrap_multiaddr, e);
                        warn!("⚠️ LIBP2P: Failed to dial {}: {}", bootstrap_multiaddr, e);
                    }
                }
            }

            // Run event loop
            let mut peer_discover_interval = tokio::time::interval(Duration::from_secs(30));
            let mut keepalive_interval = tokio::time::interval(Duration::from_secs(60));
            let mut bootstrap_refresh_interval = tokio::time::interval(Duration::from_secs(300));
            let mut quantum_annealing_interval = tokio::time::interval(Duration::from_secs(90)); // Quantum optimization every 90s

            println!("🚨🚨🚨 LIBP2P DEBUG: Entering main event loop...");
            loop {
                tokio::select! {
                    event = swarm.select_next_some() => {
                        println!("🚨🚨🚨 LIBP2P DEBUG: Received swarm event: {:?}", std::any::type_name_of_val(&event));
                        match event {
                            SwarmEvent::ConnectionEstablished { peer_id, endpoint, .. } => {
                                println!("🚨🚨🚨 LIBP2P DEBUG: ✅ CONNECTION ESTABLISHED with peer: {}", peer_id);
                                println!("🚨🚨🚨 LIBP2P DEBUG: Endpoint: {:?}", endpoint);
                                info!("✅ LIBP2P: Connection established with peer: {}", peer_id);

                                // QUANTUM ENHANCEMENT: Register peer in quantum superposition
                                if let Some(ref quantum_selector) = client_quantum_selector {
                                    quantum_selector.register_peer(peer_id).await;
                                    info!("✨ Peer {} registered in quantum superposition state", &peer_id.to_base58()[..8]);
                                }

                                // Add to discovered peers
                                {
                                    let mut peers = discovered_peers.write().await;

                                    // Convert PeerId to validator_id [u8; 32]
                                    let mut validator_id = [0u8; 32];
                                    let peer_id_bytes = peer_id.to_bytes();
                                    let len = std::cmp::min(peer_id_bytes.len(), 32);
                                    validator_id[..len].copy_from_slice(&peer_id_bytes[..len]);

                                    // Extract endpoint address
                                    let endpoint_str = endpoint.get_remote_address().to_string();
                                    let p2p_endpoint = endpoint_str.parse().unwrap_or_else(|_| "0.0.0.0:0".parse().unwrap());

                                    let peer = QnkDhtPeer {
                                        validator_id,
                                        p2p_endpoint,
                                        onion_address: None,
                                        qnk_onion_address: None,
                                        capabilities: 0,
                                        last_seen: chrono::Utc::now().timestamp() as u64,
                                        signature: [0u8; 64],
                                    };
                                    peers.push(peer);
                                }
                            }
                            SwarmEvent::IncomingConnection { connection_id, local_addr, send_back_addr } => {
                                println!("🚨🚨🚨 LIBP2P DEBUG: 📥 INCOMING CONNECTION from {}", send_back_addr);
                                info!("📥 LIBP2P: Incoming connection from {}", send_back_addr);
                            }
                            SwarmEvent::OutgoingConnectionError { peer_id, error, .. } => {
                                println!("🚨🚨🚨 LIBP2P DEBUG: ❌ OUTGOING CONNECTION ERROR: {:?}", error);
                                warn!("❌ LIBP2P: Outgoing connection error to {:?}: {}", peer_id, error);
                            }
                            SwarmEvent::Behaviour(QnkNetworkBehaviourEvent::Identify(libp2p_identify::Event::Received { peer_id, info })) => {
                                info!("🆔 LIBP2P: Identified peer: {} with {} addresses", peer_id, info.listen_addrs.len());

                                // Add all addresses to Kademlia routing table
                                for addr in &info.listen_addrs {
                                    swarm.behaviour_mut().kademlia.add_address(&peer_id, addr.clone());
                                    debug!("📋 LIBP2P: Added {} to routing table", addr);
                                }

                                // Announce via gossipsub
                                let announcement = serde_json::json!({
                                    "type": "peer_discovered",
                                    "peer_id": peer_id.to_string(),
                                    "timestamp": chrono::Utc::now().timestamp(),
                                });

                                let discovery_topic = IdentTopic::new("/qnk/peer-discovery/1.0.0");
                                if let Err(e) = swarm.behaviour_mut().gossipsub.publish(discovery_topic, announcement.to_string().as_bytes()) {
                                    warn!("⚠️ LIBP2P: Failed to announce peer: {}", e);
                                }
                            }
                            SwarmEvent::Behaviour(QnkNetworkBehaviourEvent::Gossipsub(GossipsubEvent::Message { message, .. })) => {
                                debug!("💬 LIBP2P: Received gossipsub message from peer");
                                // Handle peer announcements
                            }
                            SwarmEvent::ConnectionEstablished { peer_id, .. } => {
                                info!("🔗 LIBP2P: Connection established with {}", peer_id);
                            }
                            SwarmEvent::ConnectionClosed { peer_id, cause, .. } => {
                                debug!("🔌 LIBP2P: Connection closed with {}: {:?}", peer_id, cause);
                            }
                            _ => {}
                        }
                    }
                    _ = peer_discover_interval.tick() => {
                        debug!("🔍 LIBP2P: Running periodic peer discovery");
                        let random_peer_id = PeerId::random();
                        swarm.behaviour_mut().kademlia.get_closest_peers(random_peer_id);
                    }
                    _ = keepalive_interval.tick() => {
                        debug!("💓 LIBP2P: Performing keepalive queries");
                        let random_key: Vec<u8> = (0..32).map(|_| fastrand::u8(..)).collect();
                        swarm.behaviour_mut().kademlia.get_closest_peers(random_key);

                        // Send keepalive gossip message
                        let keepalive_msg = serde_json::json!({
                            "type": "keepalive",
                            "timestamp": chrono::Utc::now().timestamp(),
                        });

                        let discovery_topic = IdentTopic::new("/qnk/peer-discovery/1.0.0");
                        let _ = swarm.behaviour_mut().gossipsub.publish(discovery_topic, keepalive_msg.to_string().as_bytes());
                    }
                    _ = bootstrap_refresh_interval.tick() => {
                        info!("🔄 LIBP2P: Refreshing bootstrap connections");
                        if let Err(e) = swarm.behaviour_mut().kademlia.bootstrap() {
                            warn!("⚠️ LIBP2P: Bootstrap refresh failed: {}", e);
                        }

                        // Re-dial bootstrap nodes
                        for bootstrap_address in &bootstrap_addresses {
                            let bootstrap_multiaddr: Multiaddr = format!(
                                "/ip4/{}/tcp/{}",
                                bootstrap_address.ip(),
                                bootstrap_address.port()
                            )
                            .parse()
                            .unwrap();

                            if let Err(e) = swarm.dial(bootstrap_multiaddr.clone()) {
                                debug!("🔄 LIBP2P: Re-dial {}: {}", bootstrap_multiaddr, e);
                            }
                        }
                    }
                    _ = quantum_annealing_interval.tick() => {
                        // QUANTUM ENHANCEMENT: Run quantum annealing to optimize peer selection
                        if let Some(ref quantum_selector) = client_quantum_selector {
                            info!("🌀 Running quantum annealing for optimal peer selection...");

                            match quantum_selector.anneal_peer_selection().await {
                                Ok(optimal_peers) => {
                                    info!("✨ Quantum annealing complete: {} optimal peers selected", optimal_peers.len());

                                    // Get visualization data
                                    let viz = quantum_selector.get_quantum_visualization().await;
                                    info!("⚛️ Quantum state: {} peers, {} entangled pairs, coherence: {:.2}, energy: {:.3}",
                                          viz.total_peers, viz.entangled_pairs, viz.average_coherence, viz.quantum_energy);

                                    // Prioritize connections to quantum-optimal peers
                                    for peer_id in optimal_peers.iter().take(5) {
                                        debug!("🎯 Quantum-selected optimal peer: {}", &peer_id.to_base58()[..8]);
                                    }
                                }
                                Err(e) => {
                                    warn!("⚠️ Quantum annealing failed: {}", e);
                                }
                            }
                        }
                    }
                }
            }
        });

        Ok(client)
    }

    /// Initialize the libp2p swarm with Kademlia DHT
    pub async fn initialize(&mut self) -> Result<()> {
        println!("🚨🚨🚨 LIBP2P DEBUG: Initializing libp2p swarm with Kademlia DHT");
        info!("🚀 LIBP2P: Initializing libp2p swarm with Kademlia DHT");

        // Generate unique peer ID for this Q-NarwhalKnight node
        let local_key = identity::Keypair::generate_ed25519();
        let local_peer_id = PeerId::from(local_key.public());

        info!("🆔 LIBP2P: Generated peer ID: {}", local_peer_id);
        info!("🔑 LIBP2P: Validator ID: {}", hex::encode(self.local_validator_id));

        // Build the swarm using new libp2p 0.53.2 builder pattern
        println!("🚨🚨🚨 LIBP2P DEBUG: Building libp2p Swarm with new builder pattern");
        let mut swarm = SwarmBuilder::with_existing_identity(local_key)
            .with_tokio()
            .with_tcp(
                TcpConfig::default().nodelay(true),
                NoiseConfig::new,
                YamuxConfig::default,
            )?
            .with_behaviour(|keypair| {
                // Create Kademlia DHT store
                let store = libp2p_kad::store::MemoryStore::new(local_peer_id);

                // Configure Kademlia with custom settings for robustness
                let mut kad_config = libp2p_kad::Config::default();
                kad_config.set_query_timeout(Duration::from_secs(60));
                // Note: connection idle timeout is set at Swarm level (600s below)
                kad_config.set_replication_factor(20.try_into().unwrap()); // More peers for redundancy
                kad_config.set_max_packet_size(16384); // Larger packets for efficiency

                // Create Kademlia with configuration
                let mut kademlia = Kademlia::with_config(local_peer_id, store, kad_config);

                // Set Kademlia mode to server to enable query responses
                kademlia.set_mode(Some(libp2p_kad::Mode::Server));

                // Configure identify protocol for peer discovery
                let identify = Identify::new(libp2p_identify::Config::new(
                    "/q-narwhalknight/1.0.0".to_string(),
                    keypair.public(),
                ));

                // Configure gossipsub for efficient peer discovery announcements
                let gossipsub_config = GossipsubConfigBuilder::default()
                    .heartbeat_interval(Duration::from_secs(10))
                    .validation_mode(libp2p_gossipsub::ValidationMode::Permissive)
                    .build()
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

                let mut gossipsub = Gossipsub::new(
                    MessageAuthenticity::Signed(keypair.clone()),
                    gossipsub_config
                ).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

                // Subscribe to Q-NarwhalKnight peer discovery topic
                let discovery_topic = IdentTopic::new("/qnk/peer-discovery/1.0.0");
                gossipsub.subscribe(&discovery_topic)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

                // Create and return network behaviour
                Ok(QnkNetworkBehaviour {
                    kademlia,
                    identify,
                    gossipsub,
                })
            })?
            .with_swarm_config(|cfg| {
                cfg.with_idle_connection_timeout(Duration::from_secs(600)) // Keep connections alive for 10 minutes
            })
            .build();
        println!("🚨🚨🚨 LIBP2P DEBUG: Swarm built successfully with keepalive config");

        // Listen on the configured address
        let listen_addr = format!("/ip4/0.0.0.0/tcp/{}", self.config.listen_addr.port());
        println!("🚨🚨🚨 LIBP2P DEBUG: Starting to listen on multiaddr: {}", listen_addr);

        let multiaddr = listen_addr.parse()
            .context(format!("Failed to parse multiaddr: {}", listen_addr))?;
        println!("🚨🚨🚨 LIBP2P DEBUG: Multiaddr parsed successfully: {:?}", multiaddr);

        swarm.listen_on(multiaddr)
            .context(format!("Failed to listen on address: {}", listen_addr))?;
        println!("🚨🚨🚨 LIBP2P DEBUG: Successfully started listening on: {}", listen_addr);

        info!("🎧 LIBP2P: Listening on: {}", listen_addr);

        self.swarm = Some(swarm);
        println!("🚨🚨🚨 LIBP2P DEBUG: libp2p swarm initialization completed successfully");
        Ok(())
    }

    /// Start the libp2p discovery process
    pub async fn start(&mut self) -> Result<()> {
        println!("🚨🚨🚨 LIBP2P DEBUG: ========== STARTING Q-NARWHALKNIGHT LIBP2P DISCOVERY ===========");
        println!("🚨🚨🚨 LIBP2P DEBUG: Target bootstrap node: 185.182.185.227:6881");
        println!("🚨🚨🚨 LIBP2P DEBUG: This will attempt real libp2p Kademlia DHT bootstrap");
        info!("🚀 LIBP2P: Starting Q-NarwhalKnight libp2p discovery");

        if self.swarm.is_none() {
            println!("🚨🚨🚨 LIBP2P DEBUG: Swarm not initialized, calling initialize() first");
            self.initialize().await?;
            println!("🚨🚨🚨 LIBP2P DEBUG: Swarm initialization completed");
        } else {
            println!("🚨🚨🚨 LIBP2P DEBUG: Swarm already initialized, proceeding to start");
        }

        println!("🚨🚨🚨 LIBP2P DEBUG: Setting discovery state to running");
        let mut is_running = self.is_running.write().await;
        *is_running = true;
        drop(is_running);
        println!("🚨🚨🚨 LIBP2P DEBUG: Discovery state set to running=true");

        // CRITICAL FIX: Poll the swarm until we get NewListenAddr event
        // This ensures the TCP socket actually binds before we spawn the background task
        println!("🚨🚨🚨 LIBP2P DEBUG: Waiting for TCP listener to bind (polling for NewListenAddr event)...");
        if let Some(ref mut swarm) = self.swarm {
            use futures::StreamExt;
            let mut listener_bound = false;
            let timeout_duration = std::time::Duration::from_secs(5);
            let start = std::time::Instant::now();

            while !listener_bound && start.elapsed() < timeout_duration {
                tokio::select! {
                    event = swarm.select_next_some() => {
                        match &event {
                            SwarmEvent::NewListenAddr { address, .. } => {
                                println!("🚨🚨🚨 LIBP2P DEBUG: ✅✅✅ TCP LISTENER BOUND AT OS LEVEL: {}", address);
                                info!("🎧 LIBP2P: TCP Listener successfully bound: {}", address);
                                listener_bound = true;
                            }
                            _ => {
                                println!("🚨🚨🚨 LIBP2P DEBUG: Got event during bind wait: {:?}", std::any::type_name_of_val(&event));
                            }
                        }
                    }
                    _ = tokio::time::sleep(std::time::Duration::from_millis(100)) => {
                        if !listener_bound {
                            println!("🚨🚨🚨 LIBP2P DEBUG: Still waiting for TCP bind...");
                        }
                    }
                }
            }

            if !listener_bound {
                println!("🚨🚨🚨 LIBP2P DEBUG: ⚠️ WARNING: TCP listener did not bind within 5 seconds!");
            } else {
                println!("🚨🚨🚨 LIBP2P DEBUG: ✅ TCP listener confirmed bound, proceeding with bootstrap");
            }
        }

        // Perform bootstrap connection
        println!("🚨🚨🚨 LIBP2P DEBUG: Starting bootstrap connection to 185.182.185.227:6881");
        self.bootstrap_to_target().await?;
        println!("🚨🚨🚨 LIBP2P DEBUG: Bootstrap connection attempt completed");

        // Spawn the main event loop in a background task so TCP listener can poll
        println!("🚨🚨🚨 LIBP2P DEBUG: Spawning main libp2p event loop in background task");
        let is_running_clone = self.is_running.clone();
        let discovered_peers_clone = self.discovered_peers.clone();
        let bootstrap_addresses_clone = self.bootstrap_addresses.clone();

        // Take ownership of swarm to move it into spawned task
        if let Some(swarm) = self.swarm.take() {
            tokio::spawn(async move {
                let result = Self::run_event_loop_static(
                    swarm,
                    is_running_clone,
                    discovered_peers_clone,
                    bootstrap_addresses_clone
                ).await;

                if let Err(e) = result {
                    error!("🔥 LIBP2P: Event loop error: {}", e);
                }
            });
            println!("🚨🚨🚨 LIBP2P DEBUG: Event loop spawned successfully - TCP listener already bound and accepting");
        } else {
            return Err(anyhow::anyhow!("Swarm is None, cannot start event loop"));
        }

        Ok(())
    }

    /// Bootstrap to all configured bootstrap nodes
    async fn bootstrap_to_target(&mut self) -> Result<()> {
        println!("🚨🚨🚨 LIBP2P DEBUG: ========== BOOTSTRAP TO CONFIGURED NODES ===========");
        println!("🚨🚨🚨 LIBP2P DEBUG: Bootstrap nodes count: {}", self.bootstrap_addresses.len());
        info!("🌟 LIBP2P: Performing bootstrap to {} nodes", self.bootstrap_addresses.len());

        if let Some(ref mut swarm) = self.swarm {
            // Bootstrap to all configured nodes
            for bootstrap_address in &self.bootstrap_addresses {
                // Create multiaddr for the bootstrap node
                let bootstrap_multiaddr: Multiaddr = format!(
                    "/ip4/{}/tcp/{}",
                    bootstrap_address.ip(),
                    bootstrap_address.port()
                ).parse()
                .context("Failed to parse bootstrap multiaddr")?;

                println!("🚨🚨🚨 LIBP2P DEBUG: Bootstrap multiaddr created: {}", bootstrap_multiaddr);
                println!("🚨🚨🚨 LIBP2P DEBUG: IP: {}, Port: {}", bootstrap_address.ip(), bootstrap_address.port());
                info!("🔗 LIBP2P: Connecting to bootstrap multiaddr: {}", bootstrap_multiaddr);

                // First dial the bootstrap node to establish connection
                println!("🚨🚨🚨 LIBP2P DEBUG: Dialing bootstrap node for PeerId discovery");
                // Note: We let libp2p discover the actual PeerId through the identify protocol
                // after connection is established, rather than using random IDs

                // Dial the bootstrap node
                println!("🚨🚨🚨 LIBP2P DEBUG: Attempting to dial bootstrap node {}...", bootstrap_address);
                println!("🚨🚨🚨 LIBP2P DEBUG: Dial target: {}", bootstrap_multiaddr);
                match swarm.dial(bootstrap_multiaddr.clone()) {
                    Ok(_) => {
                        println!("🚨🚨🚨 LIBP2P DEBUG: ✅ Dial request submitted successfully to: {}", bootstrap_multiaddr);
                        info!("📞 LIBP2P: Dialing bootstrap node: {}", bootstrap_multiaddr);
                    }
                    Err(e) => {
                        println!("🚨🚨🚨 LIBP2P DEBUG: ❌ Dial request failed for {}: {}", bootstrap_address, e);
                        error!("❌ LIBP2P: Failed to dial bootstrap node {}: {}", bootstrap_address, e);
                        return Err(anyhow::anyhow!("Bootstrap dial failed: {}", e));
                    }
                }
            }

            // Start bootstrap process
            println!("🚨🚨🚨 LIBP2P DEBUG: Starting Kademlia bootstrap query");
            let query_id = swarm.behaviour_mut().kademlia.bootstrap();
            match query_id {
                Ok(query_id) => {
                    println!("🚨🚨🚨 LIBP2P DEBUG: ✅ Bootstrap query started successfully: {:?}", query_id);
                    info!("🔍 LIBP2P: Started bootstrap query: {:?}", query_id);
                }
                Err(e) => {
                    println!("🚨🚨🚨 LIBP2P DEBUG: ❌ Bootstrap query failed: {}", e);
                    warn!("⚠️ LIBP2P: Bootstrap query failed: {}", e);
                }
            }
        } else {
            println!("🚨🚨🚨 LIBP2P DEBUG: ❌ CRITICAL ERROR: Swarm is None - cannot perform bootstrap");
            return Err(anyhow::anyhow!("Swarm not initialized"));
        }

        println!("🚨🚨🚨 LIBP2P DEBUG: ========== BOOTSTRAP TO TARGET COMPLETED ===========");
        Ok(())
    }

    /// Main event loop for libp2p swarm
    /// Static version of event loop that can be spawned in background task
    /// Simplified version that just polls swarm events to enable TCP listener
    async fn run_event_loop_static(
        mut swarm: Swarm<QnkNetworkBehaviour>,
        is_running: Arc<RwLock<bool>>,
        _discovered_peers: Arc<RwLock<Vec<QnkDhtPeer>>>,
        _bootstrap_addresses: Vec<SocketAddr>,
    ) -> Result<()> {
        println!("🚨🚨🚨 LIBP2P DEBUG: ========== STARTING MAIN EVENT LOOP (BACKGROUND TASK) ===========");
        println!("🚨🚨🚨 LIBP2P DEBUG: Event loop will handle swarm events - TCP listener should now accept connections");
        info!("🔄 LIBP2P: Starting main event loop in background task");

        loop {
            tokio::select! {
                // Handle swarm events - THIS IS THE KEY TO ENABLE TCP LISTENER
                event = swarm.select_next_some() => {
                    // Just log the event, don't process it deeply for now
                    match &event {
                        SwarmEvent::NewListenAddr { address, .. } => {
                            println!("🚨🚨🚨 LIBP2P DEBUG: ✅ TCP LISTENER ACTIVE: {}", address);
                            info!("🎧 LIBP2P: Listening on: {}", address);
                        }
                        SwarmEvent::ConnectionEstablished { peer_id, endpoint, .. } => {
                            println!("🚨🚨🚨 LIBP2P DEBUG: ✅ CONNECTION ESTABLISHED!");
                            println!("🚨🚨🚨 LIBP2P DEBUG: Peer ID: {}", peer_id);
                            println!("🚨🚨🚨 LIBP2P DEBUG: Remote address: {}", endpoint.get_remote_address());
                            info!("🤝 LIBP2P: Connection established with peer: {}", peer_id);

                            // Add peer to Kademlia routing table
                            swarm.behaviour_mut().kademlia.add_address(peer_id, endpoint.get_remote_address().clone());
                        }
                        SwarmEvent::OutgoingConnectionError { peer_id, error, .. } => {
                            println!("🚨🚨🚨 LIBP2P DEBUG: ❌ OUTGOING CONNECTION ERROR: {:?}", error);
                            if let Some(pid) = peer_id {
                                warn!("❌ LIBP2P: Outgoing connection error to {}: {}", pid, error);
                            } else {
                                warn!("❌ LIBP2P: Outgoing connection error: {}", error);
                            }
                        }
                        SwarmEvent::Behaviour(QnkNetworkBehaviourEvent::Kademlia(kad_event)) => {
                            match kad_event {
                                KademliaEvent::OutboundQueryProgressed { result, .. } => {
                                    match result {
                                        QueryResult::Bootstrap(Ok(ok)) => {
                                            println!("🚨🚨🚨 LIBP2P DEBUG: ✅ BOOTSTRAP QUERY SUCCESSFUL!");
                                            println!("🚨🚨🚨 LIBP2P DEBUG: Remaining peers: {}", ok.num_remaining);
                                            info!("✅ LIBP2P: Bootstrap successful!");
                                        }
                                        QueryResult::Bootstrap(Err(e)) => {
                                            println!("🚨🚨🚨 LIBP2P DEBUG: ❌ BOOTSTRAP QUERY FAILED: {}", e);
                                            warn!("❌ LIBP2P: Bootstrap failed: {}", e);
                                        }
                                        _ => {}
                                    }
                                }
                                _ => {}
                            }
                        }
                        _ => {
                            debug!("🔍 LIBP2P: Other swarm event: {:?}", event);
                        }
                    }
                }

                // Check if we should stop
                _ = tokio::time::sleep(Duration::from_millis(100)) => {
                    let running = is_running.read().await;
                    if !*running {
                        info!("🛑 LIBP2P: Stopping event loop");
                        break;
                    }
                }
            }
        }

        Ok(())
    }

    async fn run_event_loop(&mut self) -> Result<()> {
        println!("🚨🚨🚨 LIBP2P DEBUG: ========== STARTING MAIN EVENT LOOP ===========");
        println!("🚨🚨🚨 LIBP2P DEBUG: Event loop will handle swarm events and peer discovery");
        info!("🔄 LIBP2P: Starting main event loop");

        let mut peer_discovery_interval = tokio::time::interval(Duration::from_secs(30));
        let mut keepalive_interval = tokio::time::interval(Duration::from_secs(60));
        let mut bootstrap_refresh_interval = tokio::time::interval(Duration::from_secs(300)); // Re-bootstrap every 5 minutes

        loop {
            tokio::select! {
                // Handle swarm events
                event = async {
                    if let Some(ref mut swarm) = self.swarm {
                        swarm.select_next_some().await
                    } else {
                        // If swarm is None, wait indefinitely
                        futures::future::pending().await
                    }
                } => {
                    self.handle_swarm_event(event).await?;
                }

                // Periodic peer discovery
                _ = peer_discovery_interval.tick() => {
                    self.perform_peer_discovery().await?;
                }

                // Keepalive: send periodic queries to maintain connections
                _ = keepalive_interval.tick() => {
                    self.perform_keepalive_queries().await?;
                }

                // Periodic bootstrap refresh to rediscover peers
                _ = bootstrap_refresh_interval.tick() => {
                    self.refresh_bootstrap().await?;
                }

                // Check if we should stop
                _ = tokio::time::sleep(Duration::from_millis(100)) => {
                    let is_running = self.is_running.read().await;
                    if !*is_running {
                        info!("🛑 LIBP2P: Stopping event loop");
                        break;
                    }
                }
            }
        }

        Ok(())
    }

    /// Handle libp2p swarm events
    async fn handle_swarm_event(&mut self, event: SwarmEventType<QnkNetworkBehaviourEvent>) -> Result<()> {
        match event {
            SwarmEvent::NewListenAddr { address, .. } => {
                info!("🎧 LIBP2P: Listening on: {}", address);
            }

            SwarmEvent::ConnectionEstablished { peer_id, endpoint, .. } => {
                println!("🚨🚨🚨 LIBP2P DEBUG: ✅ CONNECTION ESTABLISHED!");
                println!("🚨🚨🚨 LIBP2P DEBUG: Peer ID: {}", peer_id);
                println!("🚨🚨🚨 LIBP2P DEBUG: Remote address: {}", endpoint.get_remote_address());
                println!("🚨🚨🚨 LIBP2P DEBUG: Connection endpoint: {:?}", endpoint);
                info!("🤝 LIBP2P: Connection established with peer: {}", peer_id);
                info!("📍 LIBP2P: Endpoint: {:?}", endpoint);

                // Add the peer to our Kademlia routing table
                if let Some(ref mut swarm) = self.swarm {
                    println!("🚨🚨🚨 LIBP2P DEBUG: Adding peer to Kademlia routing table");
                    swarm.behaviour_mut().kademlia.add_address(&peer_id, endpoint.get_remote_address().clone());
                    println!("🚨🚨🚨 LIBP2P DEBUG: Peer added to routing table successfully");
                }
            }

            SwarmEvent::ConnectionClosed { peer_id, cause, .. } => {
                info!("🔌 LIBP2P: Connection closed with peer: {} (cause: {:?})", peer_id, cause);
            }

            SwarmEvent::Behaviour(QnkNetworkBehaviourEvent::Kademlia(kad_event)) => {
                self.handle_kademlia_event(kad_event).await?;
            }

            SwarmEvent::Behaviour(QnkNetworkBehaviourEvent::Identify(identify_event)) => {
                self.handle_identify_event(identify_event).await?;
            }

            SwarmEvent::Behaviour(QnkNetworkBehaviourEvent::Gossipsub(gossip_event)) => {
                self.handle_gossipsub_event(gossip_event).await?;
            }

            other => {
                debug!("🔍 LIBP2P: Other swarm event: {:?}", other);
            }
        }

        Ok(())
    }

    /// Handle Kademlia DHT events
    async fn handle_kademlia_event(&mut self, event: KademliaEvent) -> Result<()> {
        match event {
            KademliaEvent::RoutingUpdated { peer, .. } => {
                info!("📋 LIBP2P: Routing table updated with peer: {}", peer);
            }

            KademliaEvent::OutboundQueryProgressed { result, .. } => {
                match result {
                    QueryResult::Bootstrap(Ok(ok)) => {
                        println!("🚨🚨🚨 LIBP2P DEBUG: ✅ BOOTSTRAP QUERY SUCCESSFUL!");
                        println!("🚨🚨🚨 LIBP2P DEBUG: Remaining peers to bootstrap: {}", ok.num_remaining);
                        println!("🚨🚨🚨 LIBP2P DEBUG: Bootstrap query OK result: {:?}", ok);
                        info!("✅ LIBP2P: Bootstrap successful!");
                        info!("🌐 LIBP2P: Discovered {} peers during bootstrap", ok.num_remaining);
                    }

                    QueryResult::Bootstrap(Err(e)) => {
                        println!("🚨🚨🚨 LIBP2P DEBUG: ❌ BOOTSTRAP QUERY FAILED: {}", e);
                        println!("🚨🚨🚨 LIBP2P DEBUG: Bootstrap error details: {:?}", e);
                        warn!("❌ LIBP2P: Bootstrap failed: {}", e);
                    }

                    QueryResult::GetClosestPeers(Ok(GetClosestPeersOk { key, peers })) => {
                        info!("🔍 LIBP2P: Found {} closest peers for key: {:?}", peers.len(), key);
                        for peer in peers {
                            info!("👥 LIBP2P: Closest peer: {}", peer);
                        }
                    }

                    other => {
                        debug!("🔍 LIBP2P: Other query result: {:?}", other);
                    }
                }
            }

            other => {
                debug!("🔍 LIBP2P: Other Kademlia event: {:?}", other);
            }
        }

        Ok(())
    }

    /// Handle identify protocol events
    async fn handle_identify_event(&mut self, event: IdentifyEvent) -> Result<()> {
        match event {
            IdentifyEvent::Received { peer_id, info } => {
                info!("🆔 LIBP2P: Identified peer: {}", peer_id);
                info!("📝 LIBP2P: Peer info: protocol_version={}, agent_version={}",
                      info.protocol_version, info.agent_version);

                // IMPROVEMENT: Add all listen addresses to Kademlia routing table
                // This ensures we can route to this peer in the DHT
                if let Some(ref mut swarm) = self.swarm {
                    for addr in &info.listen_addrs {
                        let routing_update = swarm.behaviour_mut().kademlia.add_address(&peer_id, addr.clone());
                        debug!("📋 LIBP2P: Added address {} for peer {} to routing table (update: {:?})",
                               addr, peer_id, routing_update);
                    }
                }

                // If this is a Q-NarwhalKnight peer, add it to our discovered peers
                if info.protocol_version.contains("q-narwhalknight") {
                    self.add_discovered_peer(peer_id, info).await?;
                }
            }

            IdentifyEvent::Sent { peer_id } => {
                debug!("📤 LIBP2P: Sent identify info to peer: {}", peer_id);
            }

            other => {
                debug!("🔍 LIBP2P: Other identify event: {:?}", other);
            }
        }

        Ok(())
    }

    /// Handle gossipsub events for peer discovery
    async fn handle_gossipsub_event(&mut self, event: GossipsubEvent) -> Result<()> {
        match event {
            GossipsubEvent::Message { propagation_source, message_id, message } => {
                info!("📨 LIBP2P: Received gossip message from {}", propagation_source);
                debug!("📨 LIBP2P: Message ID: {:?}, Topic: {}", message_id, message.topic);

                // Parse peer discovery announcement
                if message.topic.as_str() == "/qnk/peer-discovery/1.0.0" {
                    match serde_json::from_slice::<serde_json::Value>(&message.data) {
                        Ok(announcement) => {
                            info!("🎯 LIBP2P: Peer discovery announcement: {:?}", announcement);
                            // TODO: Process peer announcement and add to discovered peers
                        }
                        Err(e) => {
                            debug!("⚠️ LIBP2P: Failed to parse peer announcement: {}", e);
                        }
                    }
                }
            }

            GossipsubEvent::Subscribed { peer_id, topic } => {
                info!("📡 LIBP2P: Peer {} subscribed to topic: {}", peer_id, topic);
            }

            GossipsubEvent::Unsubscribed { peer_id, topic } => {
                info!("📴 LIBP2P: Peer {} unsubscribed from topic: {}", peer_id, topic);
            }

            other => {
                debug!("🔍 LIBP2P: Other gossipsub event: {:?}", other);
            }
        }

        Ok(())
    }

    /// Add a discovered Q-NarwhalKnight peer
    async fn add_discovered_peer(&mut self, peer_id: PeerId, info: libp2p::identify::Info) -> Result<()> {
        info!("🎯 LIBP2P: Adding Q-NarwhalKnight peer: {}", peer_id);

        // Extract validator ID from peer info (if available)
        let validator_id = peer_id.to_bytes().get(0..32)
            .map(|bytes| {
                let mut arr = [0u8; 32];
                arr.copy_from_slice(bytes);
                arr
            })
            .unwrap_or([0u8; 32]);

        // Create discovered peer entry
        let qnk_peer = QnkDhtPeer {
            validator_id,
            p2p_endpoint: info.listen_addrs.first()
                .and_then(|addr| {
                    // Convert multiaddr to socket address
                    if let Some(ip) = addr.iter().find_map(|p| match p {
                        libp2p::core::multiaddr::Protocol::Ip4(ip) => Some(ip.into()),
                        libp2p::core::multiaddr::Protocol::Ip6(ip) => Some(ip.into()),
                        _ => None,
                    }) {
                        if let Some(port) = addr.iter().find_map(|p| match p {
                            libp2p::core::multiaddr::Protocol::Tcp(port) => Some(port),
                            _ => None,
                        }) {
                            return Some(SocketAddr::new(ip, port));
                        }
                    }
                    None
                })
                .unwrap_or_else(|| "127.0.0.1:6881".parse().unwrap()),
            onion_address: None,
            qnk_onion_address: None,
            capabilities: 0,
            last_seen: chrono::Utc::now().timestamp() as u64,
            signature: [0u8; 64], // TODO: Implement proper signature
        };

        // Add to discovered peers list
        let mut discovered_peers = self.discovered_peers.write().await;
        discovered_peers.push(qnk_peer);

        info!("✅ LIBP2P: Successfully added Q-NarwhalKnight peer (total: {})", discovered_peers.len());

        // Announce this peer to the network via gossipsub
        if let Some(ref mut swarm) = self.swarm {
            let announcement = serde_json::json!({
                "peer_id": peer_id.to_string(),
                "validator_id": hex::encode(validator_id),
                "timestamp": chrono::Utc::now().timestamp(),
                "protocol_version": info.protocol_version,
            });

            let discovery_topic = IdentTopic::new("/qnk/peer-discovery/1.0.0");
            if let Err(e) = swarm.behaviour_mut().gossipsub.publish(
                discovery_topic,
                announcement.to_string().as_bytes()
            ) {
                debug!("⚠️ LIBP2P: Failed to broadcast peer announcement: {}", e);
            } else {
                info!("📢 LIBP2P: Broadcasted peer discovery announcement");
            }
        }

        Ok(())
    }

    /// Perform periodic peer discovery
    async fn perform_peer_discovery(&mut self) -> Result<()> {
        debug!("🔍 LIBP2P: Performing periodic peer discovery");

        // Generate Q-NarwhalKnight specific discovery key before borrowing swarm
        let discovery_key = self.generate_qnk_discovery_key();

        if let Some(ref mut swarm) = self.swarm {
            // Search for closest peers to our discovery key
            let query_id = swarm.behaviour_mut().kademlia.get_closest_peers(discovery_key);
            debug!("🔍 LIBP2P: Started closest peers query: {:?}", query_id);
        }

        Ok(())
    }

    /// Perform keepalive queries to maintain active connections
    async fn perform_keepalive_queries(&mut self) -> Result<()> {
        debug!("💓 LIBP2P: Performing keepalive queries");

        if let Some(ref mut swarm) = self.swarm {
            // Query random keys to keep routing table fresh
            let random_key: Vec<u8> = (0..32).map(|_| fastrand::u8(..)).collect();

            let query_id = swarm.behaviour_mut().kademlia.get_closest_peers(random_key);
            debug!("💓 LIBP2P: Started keepalive query: {:?}", query_id);

            // Also publish a keepalive message to gossipsub
            let keepalive_msg = serde_json::json!({
                "type": "keepalive",
                "validator_id": hex::encode(self.local_validator_id),
                "timestamp": chrono::Utc::now().timestamp(),
            });

            let discovery_topic = IdentTopic::new("/qnk/peer-discovery/1.0.0");
            if let Err(e) = swarm.behaviour_mut().gossipsub.publish(
                discovery_topic,
                keepalive_msg.to_string().as_bytes()
            ) {
                debug!("⚠️ LIBP2P: Failed to publish keepalive: {}", e);
            } else {
                debug!("💓 LIBP2P: Published keepalive message");
            }
        }

        Ok(())
    }

    /// Refresh bootstrap connections periodically
    async fn refresh_bootstrap(&mut self) -> Result<()> {
        info!("🔄 LIBP2P: Refreshing bootstrap connections");

        if let Some(ref mut swarm) = self.swarm {
            // Re-run bootstrap process
            match swarm.behaviour_mut().kademlia.bootstrap() {
                Ok(query_id) => {
                    info!("🔄 LIBP2P: Started bootstrap refresh query: {:?}", query_id);
                }
                Err(e) => {
                    warn!("⚠️ LIBP2P: Bootstrap refresh failed: {}", e);
                }
            }

            // Re-dial bootstrap nodes to ensure connections
            for bootstrap_address in &self.bootstrap_addresses.clone() {
                let bootstrap_multiaddr: Multiaddr = format!(
                    "/ip4/{}/tcp/{}",
                    bootstrap_address.ip(),
                    bootstrap_address.port()
                ).parse()
                .context("Failed to parse bootstrap multiaddr")?;

                match swarm.dial(bootstrap_multiaddr.clone()) {
                    Ok(_) => {
                        debug!("📞 LIBP2P: Re-dialing bootstrap node: {}", bootstrap_address);
                    }
                    Err(e) => {
                        debug!("⚠️ LIBP2P: Failed to re-dial bootstrap node {}: {}", bootstrap_address, e);
                    }
                }
            }
        }

        Ok(())
    }

    /// Generate Q-NarwhalKnight specific discovery key
    fn generate_qnk_discovery_key(&self) -> Vec<u8> {
        use sha2::{Sha256, Digest};

        let mut hasher = Sha256::new();
        hasher.update(b"q-narwhalknight-discovery");
        hasher.update(&self.local_validator_id);
        hasher.finalize().to_vec()
    }

    /// Get discovered peers
    pub async fn get_discovered_peers(&self) -> Vec<QnkDhtPeer> {
        let peers = self.discovered_peers.read().await;
        peers.clone()
    }

    /// Stop the discovery client
    pub async fn stop(&self) -> Result<()> {
        info!("🛑 LIBP2P: Stopping libp2p discovery client");

        let mut is_running = self.is_running.write().await;
        *is_running = false;

        Ok(())
    }
}


/// Test the libp2p bootstrap connectivity
pub async fn test_libp2p_bootstrap() -> Result<()> {
    info!("🧪 LIBP2P: Testing bootstrap connectivity to 185.182.185.227:6881");

    // Create test configuration
    let config = QnkDhtConfig {
        bootstrap_nodes: vec![SocketAddr::new(IpAddr::V4(Ipv4Addr::new(185, 182, 185, 227)), 6881)],
        listen_addr: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)), 0),
        storage_path: "./test-libp2p-storage".to_string(),
        tor_proxy: None,
        persist_dht: false,
        announce_interval: Duration::from_secs(300),
    };

    // Create test validator ID
    let validator_id = [0u8; 32];

    // Create and test libp2p client (auto-initializes and starts in background)
    let client = LibP2PDiscoveryClient::new(config, validator_id).await?;

    info!("✅ LIBP2P: Bootstrap test initialization completed and running in background");

    // Wait for 30 seconds to observe connections
    tokio::time::sleep(Duration::from_secs(30)).await;

    info!("✅ LIBP2P: Bootstrap test observation period completed");

    // Report discovered peers
    let peers = client.get_discovered_peers().await;
    info!("📊 LIBP2P: Test completed - discovered {} peers", peers.len());

    Ok(())
}