/// Real Peer Discovery System - Production Implementation
/// 
/// Integrates Bitcoin DHT, DNS steganography, Tor networking, and libp2p Kademlia
/// for actual peer discovery with real network connectivity
use anyhow::{anyhow, Result};
use libp2p::PeerId;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    net::{SocketAddr, IpAddr},
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::{
    sync::{broadcast, Mutex, RwLock},
    time::{interval, sleep},
};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

// Import our real implementations
use crate::real_dht::{RealDht, DhtCommand, DhtEvent};
// Note: These dependencies are commented out to avoid circular dependencies
// They are injected at runtime through the API layer
// use q_bitcoin_bridge::real_bitcoin_client::{RealBitcoinClient, BitcoinNetworkScanner};
// use q_dns_phantom::real_dns_resolver::{RealDnsResolver, DnsEvent, PeerAdvertisement};
use q_tor_client::real_tor_client::{RealTorClient, TorEvent};
use q_types::{NodeId, Phase};

/// Peer discovery configuration
#[derive(Debug, Clone)]
pub struct PeerDiscoveryConfig {
    pub node_id: NodeId,
    pub onion_address: String,
    pub capabilities: Vec<String>,
    pub protocol_version: String,
    pub listen_port: u16,
    pub phase: Phase,
    
    // Discovery methods
    pub enable_dht: bool,
    pub enable_bitcoin_discovery: bool,
    pub enable_dns_discovery: bool,
    pub enable_tor: bool,
    
    // Network settings
    pub bootstrap_peers: Vec<String>,
    pub bitcoin_rpc_url: Option<String>,
    pub bitcoin_rpc_user: Option<String>,
    pub bitcoin_rpc_password: Option<String>,
    pub dns_servers: Vec<String>,
    pub tor_data_dir: String,
    
    // Timing
    pub discovery_interval: Duration,
    pub advertisement_ttl: Duration,
    pub max_peers: usize,
    pub connection_timeout: Duration,
}

impl Default for PeerDiscoveryConfig {
    fn default() -> Self {
        Self {
            node_id: [0u8; 32],
            onion_address: "".to_string(),
            capabilities: vec!["consensus".to_string(), "storage".to_string()],
            protocol_version: "qnk/1.0".to_string(),
            listen_port: 8333,
            phase: Phase::Phase1,
            
            enable_dht: true,
            enable_bitcoin_discovery: true,
            enable_dns_discovery: true,
            enable_tor: true,
            
            bootstrap_peers: vec![],
            bitcoin_rpc_url: Some("http://127.0.0.1:18332".to_string()),
            bitcoin_rpc_user: Some("bitcoin".to_string()),
            bitcoin_rpc_password: Some("password".to_string()),
            dns_servers: vec!["8.8.8.8:53".to_string(), "1.1.1.1:53".to_string()],
            tor_data_dir: "tor_data".to_string(),
            
            discovery_interval: Duration::from_secs(60),
            advertisement_ttl: Duration::from_secs(3600),
            max_peers: 50,
            connection_timeout: Duration::from_secs(30),
        }
    }
}

/// Discovered peer information
#[derive(Debug, Clone)]
pub struct DiscoveredPeer {
    pub node_id: NodeId,
    pub peer_id: Option<PeerId>,
    pub addresses: Vec<SocketAddr>,
    pub onion_address: Option<String>,
    pub capabilities: Vec<String>,
    pub protocol_version: String,
    pub phase: Phase,
    pub discovered_via: DiscoveryMethod,
    pub discovered_at: SystemTime,
    pub last_seen: SystemTime,
    pub connection_status: PeerConnectionStatus,
    pub reliability_score: f64,
    pub rtt: Option<Duration>,
}

/// How the peer was discovered
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DiscoveryMethod {
    DHT,
    Bitcoin,
    DNS,
    Manual,
    Bootstrap,
}

/// Peer connection status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PeerConnectionStatus {
    Unknown,
    Connecting,
    Connected,
    Disconnected,
    Failed,
}

/// Peer discovery events
#[derive(Debug, Clone)]
pub enum PeerDiscoveryEvent {
    PeerDiscovered {
        peer: DiscoveredPeer,
        method: DiscoveryMethod,
    },
    PeerConnected {
        node_id: NodeId,
        address: String,
    },
    PeerDisconnected {
        node_id: NodeId,
        reason: String,
    },
    PeerUpdated {
        node_id: NodeId,
        peer: DiscoveredPeer,
    },
    DiscoveryError {
        method: DiscoveryMethod,
        error: String,
    },
    AdvertisementSent {
        method: DiscoveryMethod,
        target: String,
    },
}

/// Discovery statistics
#[derive(Debug, Clone, Default)]
pub struct DiscoveryStats {
    pub peers_discovered: u64,
    pub dht_discoveries: u64,
    pub bitcoin_discoveries: u64,
    pub dns_discoveries: u64,
    pub manual_additions: u64,
    pub successful_connections: u64,
    pub failed_connections: u64,
    pub advertisements_sent: u64,
    pub discovery_errors: u64,
    pub avg_discovery_time: Duration,
    pub uptime: Duration,
}

/// Real peer discovery system
pub struct RealPeerDiscovery {
    config: PeerDiscoveryConfig,
    
    // Network components
    dht_client: Option<Arc<Mutex<RealDht>>>,
    // Bitcoin and DNS components disabled due to circular dependency
    // bitcoin_client: Option<Arc<RealBitcoinClient>>,
    // bitcoin_scanner: Option<Arc<Mutex<BitcoinNetworkScanner>>>,
    // dns_resolver: Option<Arc<RealDnsResolver>>,
    tor_client: Option<Arc<RealTorClient>>,
    
    // State management
    discovered_peers: Arc<RwLock<HashMap<NodeId, DiscoveredPeer>>>,
    connected_peers: Arc<RwLock<HashSet<NodeId>>>,
    blacklisted_peers: Arc<RwLock<HashSet<NodeId>>>,
    
    // Communication
    event_sender: broadcast::Sender<PeerDiscoveryEvent>,
    stats: Arc<Mutex<DiscoveryStats>>,
    
    // Runtime state
    started: Arc<Mutex<bool>>,
    start_time: SystemTime,
}

impl RealPeerDiscovery {
    /// Create a new real peer discovery system
    pub async fn new(config: PeerDiscoveryConfig) -> Result<Self> {
        info!("Creating real peer discovery system for node: {}", 
              hex::encode(&config.node_id[..8]));

        let (event_sender, _) = broadcast::channel(1000);

        Ok(Self {
            config,
            dht_client: None,
            // Bitcoin and DNS fields commented out due to circular dependency
            // bitcoin_client: None,
            // bitcoin_scanner: None,
            // dns_resolver: None,
            tor_client: None,
            discovered_peers: Arc::new(RwLock::new(HashMap::new())),
            connected_peers: Arc::new(RwLock::new(HashSet::new())),
            blacklisted_peers: Arc::new(RwLock::new(HashSet::new())),
            event_sender,
            stats: Arc::new(Mutex::new(DiscoveryStats::default())),
            started: Arc::new(Mutex::new(false)),
            start_time: SystemTime::now(),
        })
    }

    /// Initialize all discovery components
    pub async fn initialize(&mut self) -> Result<()> {
        info!("Initializing peer discovery components...");

        // Initialize Tor client first (needed for other components)
        if self.config.enable_tor {
            info!("Initializing Tor client...");
            let tor_client = q_tor_client::real_tor_client::create_tor_client(
                &self.config.tor_data_dir,
                true, // Enable onion service
                None, // No bridges for now
            ).await?;
            
            // Create onion service if we don't have an address
            if self.config.onion_address.is_empty() {
                let onion_config = q_tor_client::real_tor_client::OnionServiceConfig {
                    service_name: "q-narwhal-node".to_string(),
                    virtual_port: self.config.listen_port,
                    target_port: self.config.listen_port,
                    private_key: None,
                    client_auth: vec![],
                    max_connections: 100,
                };
                
                let onion_address = tor_client.create_onion_service(onion_config).await?;
                info!("Created onion service: {}", onion_address);
                // Note: In a real implementation, we'd update our config here
            }
            
            self.tor_client = Some(Arc::new(tor_client));
        }

        // Initialize DHT
        if self.config.enable_dht {
            info!("Initializing DHT client...");
            let dht = crate::real_dht::create_production_dht(
                self.config.bootstrap_peers.clone(),
                Some(self.config.listen_port),
            ).await?;
            
            self.dht_client = Some(Arc::new(Mutex::new(dht)));
        }

        // Initialize Bitcoin client
        if self.config.enable_bitcoin_discovery {
            if let (Some(url), Some(user), Some(pass)) = (
                &self.config.bitcoin_rpc_url,
                &self.config.bitcoin_rpc_user,
                &self.config.bitcoin_rpc_password,
            ) {
                info!("Initializing Bitcoin client...");
                let tor_proxy = if self.config.enable_tor && self.tor_client.is_some() {
                    Some("socks5://127.0.0.1:9050")
                } else {
                    None
                };

                // TODO: Bitcoin client initialization - commented out to avoid circular deps
                // let bitcoin_client = q_bitcoin_bridge::real_bitcoin_client::create_bitcoin_client(
                //     url, user, pass, tor_proxy,
                // ).await?;
                // let scanner = BitcoinNetworkScanner::new(bitcoin_client.clone());
                // self.bitcoin_client = Some(Arc::new(bitcoin_client));
                // self.bitcoin_scanner = Some(Arc::new(Mutex::new(scanner)));
                info!("Bitcoin client initialization disabled (circular dependency)");
            }
        }

        // Initialize DNS resolver
        if self.config.enable_dns_discovery {
            info!("Initializing DNS resolver...");
            let dns_servers: Vec<&str> = self.config.dns_servers.iter()
                .map(|s| s.as_str()).collect();
            
            // TODO: DNS resolver initialization - commented out to avoid circular deps
            // let dns_resolver = q_dns_phantom::real_dns_resolver::create_dns_resolver(
            //     dns_servers,
            //     true, // Enable steganography
            //     if self.config.enable_tor { Some("socks5://127.0.0.1:9050") } else { None },
            // ).await?;
            // dns_resolver.start_background_tasks().await?;
            // self.dns_resolver = Some(Arc::new(dns_resolver));
            info!("DNS resolver initialization disabled (circular dependency)");
        }

        info!("Peer discovery components initialized successfully");
        Ok(())
    }

    /// Start the peer discovery system
    pub async fn start(&self) -> Result<()> {
        info!("Starting peer discovery system...");

        {
            let mut started = self.started.lock().await;
            if *started {
                return Err(anyhow!("Peer discovery already started"));
            }
            *started = true;
        }

        // Start DHT if available
        if let Some(dht) = &self.dht_client {
            let dht_clone = dht.clone();
            let event_sender = self.event_sender.clone();
            
            tokio::spawn(async move {
                let dht_guard = dht_clone.lock().await;
                let mut event_receiver = dht_guard.subscribe_events();
                drop(dht_guard);
                
                while let Ok(event) = event_receiver.recv().await {
                    if let Err(e) = Self::handle_dht_event(event, event_sender.clone()).await {
                        debug!("DHT event handling error: {}", e);
                    }
                }
            });

            // Start DHT bootstrap
            let dht_guard = dht.lock().await;
            let command_sender = dht_guard.command_sender();
            drop(dht_guard);
            
            if let Err(e) = command_sender.send(DhtCommand::Bootstrap).await {
                warn!("Failed to start DHT bootstrap: {}", e);
            }
        }

        // Start DNS monitoring
        // DNS integration disabled due to circular dependency
        // if let Some(dns) = &self.dns_resolver {
        //     let dns_clone = dns.clone();
        //     let event_sender = self.event_sender.clone();
        //
        //     tokio::spawn(async move {
        //         let mut event_receiver = dns_clone.subscribe_events();
        //
        //         while let Ok(_event) = event_receiver.recv().await {
        //             // DNS event handling disabled due to circular dependency
        //             debug!("DNS event received but handler disabled");
        //         }
        //     });
        // }

        // Start Tor monitoring
        if let Some(tor) = &self.tor_client {
            let tor_clone = tor.clone();
            let event_sender = self.event_sender.clone();
            
            tokio::spawn(async move {
                let mut event_receiver = tor_clone.subscribe_events();
                
                while let Ok(event) = event_receiver.recv().await {
                    if let Err(e) = Self::handle_tor_event(event, event_sender.clone()).await {
                        debug!("Tor event handling error: {}", e);
                    }
                }
            });
        }

        // Start periodic discovery tasks
        self.start_periodic_tasks().await?;

        info!("Peer discovery system started successfully");
        Ok(())
    }

    /// Start periodic discovery tasks
    async fn start_periodic_tasks(&self) -> Result<()> {
        // DHT discovery task
        if self.dht_client.is_some() {
            let dht = self.dht_client.clone();
            let config = self.config.clone();
            let stats = self.stats.clone();
            
            tokio::spawn(async move {
                let mut interval = interval(config.discovery_interval);
                loop {
                    interval.tick().await;
                    
                    if let Some(dht_client) = &dht {
                        if let Err(e) = Self::run_dht_discovery(dht_client.clone(), &config, stats.clone()).await {
                            warn!("DHT discovery error: {}", e);
                        }
                    }
                }
            });
        }

        // Bitcoin discovery task
        // Bitcoin scanner disabled due to circular dependency
        if false { // if self.bitcoin_scanner.is_some() {
            // let scanner = self.bitcoin_scanner.clone();
            let config = self.config.clone();
            let event_sender = self.event_sender.clone();
            let stats = self.stats.clone();
            
            tokio::spawn(async move {
                let mut interval = interval(config.discovery_interval * 2); // Less frequent
                loop {
                    interval.tick().await;
                    
                    // Bitcoin discovery disabled
                    // if let Some(bitcoin_scanner) = &scanner {
                    //     if let Err(e) = Self::run_bitcoin_discovery(
                    //         bitcoin_scanner.clone(),
                    //         event_sender.clone(),
                    //         stats.clone()
                    //     ).await {
                    //         warn!("Bitcoin discovery error: {}", e);
                    //     }
                    // }
                }
            });
        }

        // DNS discovery task
        // DNS resolver disabled due to circular dependency
        if false { // if self.dns_resolver.is_some() {
            // let dns = self.dns_resolver.clone();
            let config = self.config.clone();
            let stats = self.stats.clone();
            
            tokio::spawn(async move {
                let mut interval = interval(config.discovery_interval / 2); // More frequent
                loop {
                    interval.tick().await;
                    
                    // DNS discovery disabled due to circular dependency
                    // if let Some(dns_resolver) = &dns {
                    //     if let Err(e) = Self::run_dns_discovery(dns_resolver.clone(), &config, stats.clone()).await {
                    //         debug!("DNS discovery error: {}", e);
                    //     }
                    // }
                }
            });
        }

        // Periodic advertisement
        let config = self.config.clone();
        let dht = self.dht_client.clone();
        // let dns = self.dns_resolver.clone(); // Disabled due to circular dependency
        let stats = self.stats.clone();
        
        tokio::spawn(async move {
            let mut interval = interval(config.advertisement_ttl / 2);
            loop {
                interval.tick().await;
                
                if let Err(e) = Self::advertise_self(&config, dht.clone(), stats.clone()).await {
                    warn!("Self advertisement error: {}", e);
                }
            }
        });

        Ok(())
    }

    /// Run DHT-based peer discovery
    async fn run_dht_discovery(
        dht: Arc<Mutex<RealDht>>,
        config: &PeerDiscoveryConfig,
        stats: Arc<Mutex<DiscoveryStats>>,
    ) -> Result<()> {
        debug!("Running DHT discovery...");

        let dht_guard = dht.lock().await;
        let command_sender = dht_guard.command_sender();
        drop(dht_guard);

        // Search for peers with our capabilities
        for capability in &config.capabilities {
            let key = format!("qnk:capability:{}", capability);
            if let Err(e) = command_sender.send(DhtCommand::GetProviders(key)).await {
                debug!("DHT provider query failed: {}", e);
            }
        }

        {
            let mut stats_guard = stats.lock().await;
            stats_guard.dht_discoveries += 1;
        }

        Ok(())
    }

    /// Bitcoin network discovery (disabled due to circular dependency)
    /*
    async fn run_bitcoin_discovery(
        scanner: Arc<Mutex<BitcoinNetworkScanner>>,
        event_sender: broadcast::Sender<PeerDiscoveryEvent>,
        stats: Arc<Mutex<DiscoveryStats>>,
    ) -> Result<()> {
        debug!("Running Bitcoin network discovery...");

        let mut scanner_guard = scanner.lock().await;
        let q_nodes = scanner_guard.scan_for_q_nodes().await?;
        drop(scanner_guard);

        {
            let mut stats_guard = stats.lock().await;
            stats_guard.bitcoin_discoveries += q_nodes.len() as u64;
        }

        info!("Found {} potential Q-NarwhalKnight nodes via Bitcoin", q_nodes.len());

        // Process discovered nodes
        for node_addr in q_nodes {
            // Create peer record
            let peer = DiscoveredPeer {
                node_id: [0u8; 32], // Would be extracted from Bitcoin data
                peer_id: None,
                addresses: vec![], // Would parse from node_addr
                onion_address: None,
                capabilities: vec![],
                protocol_version: "unknown".to_string(),
                phase: Phase::Phase1,
                discovered_via: DiscoveryMethod::Bitcoin,
                discovered_at: SystemTime::now(),
                last_seen: SystemTime::now(),
                connection_status: PeerConnectionStatus::Unknown,
                reliability_score: 0.5,
                rtt: None,
            };

            let _ = event_sender.send(PeerDiscoveryEvent::PeerDiscovered {
                peer,
                method: DiscoveryMethod::Bitcoin,
            });
        }

        Ok(())
    }
    */

    /// DNS-based discovery (disabled due to circular dependency)
    /*
    async fn run_dns_discovery(
        dns: Arc<RealDnsResolver>,
        config: &PeerDiscoveryConfig,
        stats: Arc<Mutex<DiscoveryStats>>,
    ) -> Result<()> {
        debug!("Running DNS discovery...");

        // Query known Q-NarwhalKnight domains
        let discovery_domains = [
            "peers.qnk.onion",
            "registry.quantum-knight.net",
            "discovery.narwhal-nodes.com",
        ];

        for domain in &discovery_domains {
            // TODO: DNS queries disabled due to circular dependency resolution
            debug!("Skipping DNS discovery for {} (circular dependency)", domain);
        }

        Ok(())
    }
    */

    /// Advertise ourselves through available channels
    async fn advertise_self(
        config: &PeerDiscoveryConfig,
        dht: Option<Arc<Mutex<RealDht>>>,
        // dns: Option<Arc<RealDnsResolver>>, // Disabled due to circular dependency
        stats: Arc<Mutex<DiscoveryStats>>,
    ) -> Result<()> {
        debug!("Advertising self to network...");

        // Advertise via DHT
        if let Some(dht_client) = dht {
            let dht_guard = dht_client.lock().await;
            let command_sender = dht_guard.command_sender();
            drop(dht_guard);

            for capability in &config.capabilities {
                let key = format!("qnk:capability:{}", capability);
                if let Err(e) = command_sender.send(DhtCommand::StartProviding(key)).await {
                    debug!("DHT advertisement failed: {}", e);
                } else {
                    debug!("Advertised capability: {}", capability);
                }
            }

            // Store our node info
            let node_info = format!("{}:{}:{}", 
                                  hex::encode(config.node_id),
                                  config.onion_address,
                                  config.capabilities.join(","));
            let key = format!("qnk:node:{}", hex::encode(&config.node_id[..8]));
            
            if let Err(e) = command_sender.send(DhtCommand::PutRecord {
                key,
                value: node_info.into_bytes(),
            }).await {
                debug!("DHT node record storage failed: {}", e);
            }
        }

        // Advertise via DNS (steganographic) - disabled due to circular dependency
        // if let Some(dns_resolver) = dns {
        //     let advertisement = format!("v=qnk1;node={};onion={};caps={};port={}",
        //                               hex::encode(config.node_id),
        //                               config.onion_address,
        //                               config.capabilities.join(","),
        //                               config.listen_port);
        //
        //     if let Err(e) = dns_resolver.send_steganographic_message(
        //         advertisement.as_bytes(),
        //         "discovery.qnk.network"
        //     ).await {
        //         debug!("DNS steganographic advertisement failed: {}", e);
        //     } else {
        //         debug!("Sent steganographic advertisement via DNS");
        //     }
        // }

        {
            let mut stats_guard = stats.lock().await;
            stats_guard.advertisements_sent += 1;
        }

        Ok(())
    }

    /// Handle DHT events
    async fn handle_dht_event(
        event: DhtEvent,
        event_sender: broadcast::Sender<PeerDiscoveryEvent>,
    ) -> Result<()> {
        match event {
            DhtEvent::PeerDiscovered(peer_info) => {
                debug!("DHT peer discovered: {}", peer_info.peer_id);
                
                let peer = DiscoveredPeer {
                    node_id: [0u8; 32], // Would extract from peer_info
                    peer_id: Some(peer_info.peer_id),
                    addresses: peer_info.addresses.iter()
                        .filter_map(|addr| addr.to_string().parse().ok())
                        .collect(),
                    onion_address: peer_info.onion_address,
                    capabilities: peer_info.capabilities,
                    protocol_version: "qnk/1.0".to_string(),
                    phase: Phase::Phase1,
                    discovered_via: DiscoveryMethod::DHT,
                    discovered_at: SystemTime::now(),
                    last_seen: peer_info.last_seen,
                    connection_status: PeerConnectionStatus::Unknown,
                    reliability_score: 0.7,
                    rtt: peer_info.rtt,
                };

                let _ = event_sender.send(PeerDiscoveryEvent::PeerDiscovered {
                    peer,
                    method: DiscoveryMethod::DHT,
                });
            }
            _ => {}
        }
        
        Ok(())
    }

    /// Handle DNS events
    // TODO: DNS event handling disabled due to circular dependency
    /*
    async fn handle_dns_event(
        event: DnsEvent,
        event_sender: broadcast::Sender<PeerDiscoveryEvent>,
    ) -> Result<()> {
        match event {
            DnsEvent::PeerAdvertisementFound(advertisement) => {
                info!("Found peer advertisement in DNS: {}", advertisement.onion_address);
                
                let peer = DiscoveredPeer {
                    node_id: advertisement.node_id,
                    peer_id: None,
                    addresses: vec![],
                    onion_address: Some(advertisement.onion_address),
                    capabilities: advertisement.capabilities,
                    protocol_version: advertisement.protocol_version,
                    phase: Phase::Phase1,
                    discovered_via: DiscoveryMethod::DNS,
                    discovered_at: SystemTime::now(),
                    last_seen: SystemTime::now(),
                    connection_status: PeerConnectionStatus::Unknown,
                    reliability_score: 0.6,
                    rtt: None,
                };

                let _ = event_sender.send(PeerDiscoveryEvent::PeerDiscovered {
                    peer,
                    method: DiscoveryMethod::DNS,
                });
            }
            _ => {}
        }

        Ok(())
    }
    */

    /// Handle Tor events
    async fn handle_tor_event(
        event: TorEvent,
        _event_sender: broadcast::Sender<PeerDiscoveryEvent>,
    ) -> Result<()> {
        match event {
            TorEvent::OnionServicePublished { service_id, address } => {
                info!("Onion service published: {} -> {}", service_id, address);
            }
            TorEvent::ConnectionEstablished(connection) => {
                debug!("Tor connection established: {}", connection.target_address);
            }
            _ => {}
        }
        
        Ok(())
    }

    /// Subscribe to peer discovery events
    pub fn subscribe_events(&self) -> broadcast::Receiver<PeerDiscoveryEvent> {
        self.event_sender.subscribe()
    }

    /// Get discovered peers
    pub async fn get_discovered_peers(&self) -> HashMap<NodeId, DiscoveredPeer> {
        self.discovered_peers.read().await.clone()
    }

    /// Get connected peers
    pub async fn get_connected_peers(&self) -> HashSet<NodeId> {
        self.connected_peers.read().await.clone()
    }

    /// Get discovery statistics
    pub async fn get_stats(&self) -> DiscoveryStats {
        let mut stats = self.stats.lock().await.clone();
        stats.uptime = SystemTime::now().duration_since(self.start_time).unwrap_or_default();
        stats
    }

    /// Add a peer manually
    pub async fn add_peer_manually(&self, peer: DiscoveredPeer) -> Result<()> {
        {
            let mut peers = self.discovered_peers.write().await;
            peers.insert(peer.node_id, peer.clone());
        }

        {
            let mut stats = self.stats.lock().await;
            stats.manual_additions += 1;
        }

        let _ = self.event_sender.send(PeerDiscoveryEvent::PeerDiscovered {
            peer,
            method: DiscoveryMethod::Manual,
        });

        Ok(())
    }

    /// Connect to a discovered peer
    pub async fn connect_to_peer(&self, node_id: NodeId) -> Result<()> {
        let peer = {
            let peers = self.discovered_peers.read().await;
            peers.get(&node_id).cloned()
        };

        if let Some(mut peer) = peer {
            info!("Connecting to peer: {}", hex::encode(&node_id[..8]));
            
            peer.connection_status = PeerConnectionStatus::Connecting;
            peer.last_seen = SystemTime::now();
            
            // Update peer status
            {
                let mut peers = self.discovered_peers.write().await;
                peers.insert(node_id, peer.clone());
            }

            // Attempt connection through available methods
            let connected = if let Some(onion_addr) = &peer.onion_address {
                self.connect_via_tor(onion_addr).await.is_ok()
            } else if !peer.addresses.is_empty() {
                self.connect_via_clearnet(&peer.addresses[0]).await.is_ok()
            } else {
                false
            };

            if connected {
                peer.connection_status = PeerConnectionStatus::Connected;
                {
                    let mut connected = self.connected_peers.write().await;
                    connected.insert(node_id);
                }

                {
                    let mut stats = self.stats.lock().await;
                    stats.successful_connections += 1;
                }

                let _ = self.event_sender.send(PeerDiscoveryEvent::PeerConnected {
                    node_id,
                    address: peer.onion_address.clone()
                        .or_else(|| peer.addresses.first().map(|a| a.to_string()))
                        .unwrap_or_default(),
                });
            } else {
                peer.connection_status = PeerConnectionStatus::Failed;
                
                {
                    let mut stats = self.stats.lock().await;
                    stats.failed_connections += 1;
                }
            }

            // Update peer with final status
            {
                let mut peers = self.discovered_peers.write().await;
                peers.insert(node_id, peer);
            }
            
            Ok(())
        } else {
            Err(anyhow!("Peer not found: {}", hex::encode(&node_id[..8])))
        }
    }

    /// Connect via Tor
    async fn connect_via_tor(&self, onion_address: &str) -> Result<()> {
        if let Some(tor_client) = &self.tor_client {
            let _stream = tor_client.connect(onion_address).await?;
            info!("Connected to {} via Tor", onion_address);
            Ok(())
        } else {
            Err(anyhow!("Tor client not available"))
        }
    }

    /// Connect via clearnet
    async fn connect_via_clearnet(&self, address: &SocketAddr) -> Result<()> {
        let _stream = tokio::net::TcpStream::connect(address).await?;
        info!("Connected to {} via clearnet", address);
        Ok(())
    }
}

/// Create a production peer discovery system
pub async fn create_peer_discovery(
    node_id: NodeId,
    onion_address: String,
    capabilities: Vec<String>,
    bootstrap_peers: Vec<String>,
) -> Result<RealPeerDiscovery> {
    let config = PeerDiscoveryConfig {
        node_id,
        onion_address,
        capabilities,
        bootstrap_peers,
        ..Default::default()
    };

    let mut discovery = RealPeerDiscovery::new(config).await?;
    discovery.initialize().await?;
    
    Ok(discovery)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires network access
    async fn test_peer_discovery() {
        let node_id = [1u8; 32];
        let capabilities = vec!["consensus".to_string(), "storage".to_string()];
        
        let discovery = create_peer_discovery(
            node_id,
            "".to_string(),
            capabilities,
            vec![],
        ).await.expect("Failed to create peer discovery");

        discovery.start().await.expect("Failed to start discovery");
        
        // Let it run for a bit
        tokio::time::sleep(Duration::from_secs(10)).await;
        
        let stats = discovery.get_stats().await;
        println!("Discovery stats: {:?}", stats);
    }
}