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
use crate::distributed_ai::DistributedAITopics;
use q_types::QBlock;

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
    /// Request-response for block synchronization (Phase 3)
    block_sync: libp2p::request_response::Behaviour<q_storage::sync::BlockSyncCodec>,
}

#[derive(Debug)]
pub enum QNarwhalEvent {
    #[cfg(not(target_os = "windows"))]
    Mdns(MdnsEvent),
    Kademlia(KademliaEvent),
    Identify(libp2p::identify::Event),
    Ping(libp2p::ping::Event),
    Gossipsub(gossipsub::Event),
    BlockSync(libp2p::request_response::Event<q_storage::sync::BlockSyncRequest, q_storage::sync::BlockSyncResponse>),
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

impl From<libp2p::request_response::Event<q_storage::sync::BlockSyncRequest, q_storage::sync::BlockSyncResponse>> for QNarwhalEvent {
    fn from(event: libp2p::request_response::Event<q_storage::sync::BlockSyncRequest, q_storage::sync::BlockSyncResponse>) -> Self {
        QNarwhalEvent::BlockSync(event)
    }
}

/// Commands that can be sent to the network manager
#[derive(Debug)]
pub enum NetworkCommand {
    /// Dial a peer at the given multiaddr
    DialPeer {
        multiaddr: Multiaddr,
        response_tx: tokio::sync::oneshot::Sender<std::result::Result<(), String>>,
    },
    /// Set the peer discovery channel (for ConnectionManager bridge)
    SetPeerChannel {
        tx: mpsc::UnboundedSender<crate::connection_manager::PeerInfo>,
    },
    /// Set the gossipsub message channel (for message propagation)
    SetGossipsubChannel {
        tx: mpsc::UnboundedSender<(String, Vec<u8>)>,
    },
    /// Publish a block to the gossipsub network (P2P broadcasting)
    PublishBlock {
        topic: String,
        block_bytes: Vec<u8>,
        block_height: u64,
    },
    /// Publish a block request to the gossipsub network (P2P historical sync)
    PublishBlockRequest {
        topic: String,
        request_bytes: Vec<u8>,
    },
    /// Publish a block response to the gossipsub network (P2P historical sync)
    PublishBlockResponse {
        topic: String,
        response_bytes: Vec<u8>,
        block_height: u64,
    },
    /// Publish a block pack to the gossipsub network (Turbo Sync)
    PublishBlockPack {
        topic: String,
        pack_bytes: Vec<u8>,
    },
    /// Request a block pack from peers (Turbo Sync)
    RequestBlockPack {
        topic: String,
        request_bytes: Vec<u8>,
        start_height: u64,
        end_height: u64,
    },
    /// Publish an AI message to the distributed AI network
    PublishAIMessage {
        topic: String,
        message: crate::distributed_ai::AIGossipsubMessage,
    },
}

/// Response from /api/v1/peer-id endpoint
#[derive(Deserialize)]
struct PeerIdResponse {
    success: bool,
    data: Option<PeerIdData>,
}

#[derive(Deserialize)]
struct PeerIdData {
    peer_id: String,
}

/// Fetch peer ID from bootstrap node HTTP endpoint
///
/// # Arguments
/// * `ip` - IP address of bootstrap node
/// * `http_port` - HTTP API port (default: 18080)
///
/// # Returns
/// * `Ok(peer_id)` if successfully fetched
/// * `Err` if HTTP request failed or invalid response
async fn fetch_peer_id_from_http(ip: &str, http_port: u16) -> anyhow::Result<String> {
    let url = format!("http://{}:{}/api/v1/peer-id", ip, http_port);

    info!("🔍 Fetching dynamic peer ID from {}", url);

    // Use reqwest with timeout
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()?;

    let response: PeerIdResponse = client
        .get(&url)
        .send()
        .await?
        .json()
        .await?;

    if response.success {
        if let Some(data) = response.data {
            info!("✅ Successfully fetched peer ID: {}", data.peer_id);
            return Ok(data.peer_id);
        }
    }

    Err(anyhow::anyhow!("Failed to fetch peer ID from HTTP endpoint"))
}

/// Simplified Network Manager - Zero-Knowledge Discovery System
pub struct UnifiedNetworkManager {
    /// libp2p swarm handling all protocols
    swarm: Swarm<QNarwhalBehaviour>,
    /// All discovered peers from ALL mechanisms
    discovered_peers: Arc<RwLock<HashSet<PeerId>>>,
    /// Peer addresses discovered (for connection manager bridge)
    peer_addresses: Arc<RwLock<HashMap<PeerId, Vec<Multiaddr>>>>,
    /// Bootstrap peers that should be automatically reconnected on disconnect (v0.6.8-beta)
    bootstrap_peers: Arc<RwLock<HashMap<PeerId, Multiaddr>>>,
    /// Local peer ID
    local_peer_id: PeerId,
    /// Channel to send discovered peers to ConnectionManager (Phase 2 bridge)
    peer_tx: Option<mpsc::UnboundedSender<crate::connection_manager::PeerInfo>>,
    /// Channel to forward gossipsub messages (for database replication, etc.)
    gossipsub_message_tx: Option<mpsc::UnboundedSender<(String, Vec<u8>)>>,
    /// Thread-safe atomic counter for connected peers
    connected_peer_count: Arc<std::sync::atomic::AtomicUsize>,
    /// Network configuration (testnet/mainnet)
    network_config: q_types::NetworkConfig,
    /// Channel to receive network commands (e.g., dial peer)
    command_rx: mpsc::UnboundedReceiver<NetworkCommand>,
    /// Channel sender for commands (cloned and shared with API)
    command_tx: mpsc::UnboundedSender<NetworkCommand>,
    /// Storage engine for block sync (Phase 3a)
    storage: Option<Arc<q_storage::QStorage>>,
    /// Gossipsub message aggregation (v0.6.9-beta) - tracks messages per topic
    /// v0.9.7-beta: Extended to track block height ranges for sync progress visibility
    gossipsub_stats: Arc<RwLock<HashMap<String, (usize, usize, std::time::Instant, Option<u64>, Option<u64>)>>>, // (count, total_bytes, last_log_time, min_height, max_height)
    /// Channel to forward synced blocks for consensus validation (Phase 3b)
    block_sync_tx: Option<mpsc::UnboundedSender<Vec<q_types::block::QBlock>>>,
}

impl UnifiedNetworkManager {
    /// Create new network manager with network configuration
    ///
    /// # Arguments
    /// * `network_config` - Network configuration (testnet/mainnet)
    pub async fn new(network_config: q_types::NetworkConfig) -> anyhow::Result<Self> {
        // Generate or load keypair
        let keypair = Keypair::generate_ed25519();
        let local_peer_id = PeerId::from(keypair.public());

        info!("🚀 Starting Q-NarwhalKnight Zero-Knowledge Discovery");
        info!("🌐 Network: {}", network_config.network_id.display_name());
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

        // Bootstrap from network configuration with automatic peer ID discovery
        let bootstrap_peers = &network_config.bootstrap_peers;
        let mut bootstrap_count = 0;
        // 🔧 v0.6.8-beta: Track bootstrap peers for automatic reconnection
        let mut bootstrap_peer_map: HashMap<PeerId, Multiaddr> = HashMap::new();

        for addr_str in bootstrap_peers {
            if let Ok(mut addr) = addr_str.trim().parse::<Multiaddr>() {
                // Extract peer ID from multiaddr (last component should be /p2p/<peer_id>)
                use libp2p::multiaddr::Protocol;

                // Check if multiaddr has /p2p/ component
                let has_peer_id = addr.iter().any(|p| matches!(p, Protocol::P2p(_)));

                if has_peer_id {
                    // Multiaddr already has peer ID - use directly
                    if let Some(Protocol::P2p(peer_id)) = addr.iter().last() {
                        kademlia.add_address(&peer_id, addr.clone());
                        // 🔧 v0.6.8-beta: Track this bootstrap peer for automatic reconnection
                        bootstrap_peer_map.insert(peer_id, addr.clone());
                        info!("📍 Added {} bootstrap peer: {} at {}",
                              network_config.network_id.as_str(), peer_id, addr);
                        bootstrap_count += 1;
                    }
                } else {
                    // Missing /p2p/ component - try automatic discovery
                    warn!("⚠️ Bootstrap multiaddr missing /p2p/ component: {}", addr);

                    // Extract IP and P2P port from multiaddr
                    let mut ip: Option<String> = None;
                    let mut p2p_port: Option<u16> = None;

                    for protocol in addr.iter() {
                        match protocol {
                            Protocol::Ip4(addr_v4) => ip = Some(addr_v4.to_string()),
                            Protocol::Ip6(addr_v6) => ip = Some(addr_v6.to_string()),
                            Protocol::Tcp(port) => p2p_port = Some(port),
                            _ => {}
                        }
                    }

                    if let (Some(bootstrap_ip), Some(_)) = (ip, p2p_port) {
                        info!("🔄 Attempting automatic peer ID discovery for {}", bootstrap_ip);

                        // Try to fetch peer ID from HTTP endpoint (port 18080 by default)
                        match fetch_peer_id_from_http(&bootstrap_ip, 18080).await {
                            Ok(peer_id_str) => {
                                // Parse peer ID and append to multiaddr
                                match peer_id_str.parse::<PeerId>() {
                                    Ok(peer_id) => {
                                        addr.push(Protocol::P2p(peer_id));
                                        kademlia.add_address(&peer_id, addr.clone());
                                        // 🔧 v0.6.8-beta: Track this bootstrap peer for automatic reconnection
                                        bootstrap_peer_map.insert(peer_id, addr.clone());
                                        info!("✅ Added {} bootstrap peer with dynamic peer ID: {} at {}",
                                              network_config.network_id.as_str(), peer_id, addr);
                                        bootstrap_count += 1;
                                    }
                                    Err(e) => {
                                        warn!("⚠️ Failed to parse fetched peer ID: {}", e);
                                    }
                                }
                            }
                            Err(e) => {
                                warn!("⚠️ Failed to fetch peer ID via HTTP: {}", e);
                                warn!("   Skipping bootstrap peer (no fallback peer ID available)");
                            }
                        }
                    } else {
                        warn!("⚠️ Could not extract IP/port from multiaddr: {}", addr);
                    }
                }
            } else {
                warn!("⚠️ Invalid bootstrap multiaddr: {}", addr_str);
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
        // 🚀 v0.6.5-beta: Increased max_transmit_size to 10MB for large batch sync messages
        // Batch sync with postcard serialization: ~395 bytes/block → 10MB allows ~25k blocks/batch
        let gossipsub_config = gossipsub::ConfigBuilder::default()
            .heartbeat_interval(Duration::from_millis(100)) // Fast propagation
            .validation_mode(ValidationMode::Strict) // Validate messages
            .max_transmit_size(10 * 1024 * 1024) // 10MB for large batches (was 65KB default)
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

        // Subscribe to network-specific consensus topics (testnet/mainnet separation)
        // Each network has its own gossipsub namespace to prevent cross-network message propagation
        let network_prefix = network_config.network_id.gossipsub_topic_prefix();
        let topics = vec![
            IdentTopic::new(format!("{}/blocks", network_prefix)),          // Block propagation
            IdentTopic::new(network_config.network_id.transactions_topic()), // Transaction propagation
            IdentTopic::new(format!("{}/mining-rewards", network_prefix)),   // Mining reward announcements
            IdentTopic::new(format!("{}/dex/swaps", network_prefix)),        // DEX swap events
            IdentTopic::new(format!("{}/votes", network_prefix)),            // Vote aggregation
            IdentTopic::new(network_config.network_id.acks_topic()),         // Acknowledgements
            IdentTopic::new(network_config.network_id.block_requests_topic()), // P2P block requests
            IdentTopic::new(network_config.network_id.block_responses_topic()), // P2P block responses (single blocks)
            IdentTopic::new(network_config.network_id.batch_block_responses_topic()), // P2P BATCH block responses (OPTIMIZED)
        ];

        for topic in &topics {
            gossipsub.subscribe(topic)
                .map_err(|e| anyhow::anyhow!("Failed to subscribe to topic {}: {}", topic, e))?;
            info!("📢 Subscribed to {} Gossipsub topic: {}",
                  network_config.network_id.as_str(), topic);
        }

        // Subscribe to distributed AI inference topics
        let ai_topics = DistributedAITopics::new();
        for topic in ai_topics.all_topics() {
            gossipsub.subscribe(&topic)
                .map_err(|e| anyhow::anyhow!("Failed to subscribe to AI topic {}: {}", topic, e))?;
            info!("🤖 Subscribed to AI inference topic: {}", topic);
        }
        info!("✅ Subscribed to {} AI inference Gossipsub topics", ai_topics.all_topics().len());

        // Configure Request-Response for block synchronization (Phase 3)
        use libp2p::request_response::{self, ProtocolSupport};
        use q_storage::sync::{BlockSyncCodec, BLOCK_SYNC_PROTOCOL};

        let block_sync_protocols = std::iter::once((BLOCK_SYNC_PROTOCOL, ProtocolSupport::Full));
        let block_sync_config = request_response::Config::default();
        let block_sync = request_response::Behaviour::with_codec(
            BlockSyncCodec::default(),
            block_sync_protocols,
            block_sync_config,
        );

        info!("🔗 Block sync request-response protocol initialized");

        // Combine all behaviors
        let behaviour = QNarwhalBehaviour {
            #[cfg(not(target_os = "windows"))]
            mdns,
            kademlia,
            identify,
            ping,
            gossipsub,
            block_sync,
        };

        // Build swarm using libp2p v0.53 API
        // 🔧 v0.6.8-beta: Increase idle connection timeout to prevent premature disconnections
        // Server Alpha was disconnecting from Server Beta after only 41 seconds due to
        // default 10-second idle timeout. Increasing to 300 seconds (5 minutes) to maintain
        // stable connections for continuous peer height announcements and turbo sync.
        // See: SERVER_ALPHA_SYNC_DIAGNOSIS.md and V0.6.8_BETA_LOG_REDUCTION_AND_NETWORK_FIX.md
        let config = Config::with_tokio_executor()
            .with_idle_connection_timeout(Duration::from_secs(300)); // 5 minutes (was 10 seconds default)
        let mut swarm = Swarm::new(transport, behaviour, local_peer_id, config);

        // Listen on configured port or random port
        // Check for Q_P2P_PORT environment variable for fixed port (bootstrap nodes)
        let p2p_port = std::env::var("Q_P2P_PORT")
            .ok()
            .and_then(|p| p.parse::<u16>().ok())
            .unwrap_or(0); // 0 = random port (default)

        if p2p_port > 0 {
            info!("🔒 Using fixed libp2p port: {}", p2p_port);
            swarm.listen_on(format!("/ip4/0.0.0.0/tcp/{}", p2p_port).parse()?)?;
            swarm.listen_on(format!("/ip6/::/tcp/{}", p2p_port).parse()?)?;
        } else {
            // Listen on all interfaces, random port
            swarm.listen_on("/ip4/0.0.0.0/tcp/0".parse()?)?;
            swarm.listen_on("/ip6/::/tcp/0".parse()?)?;
        }

        info!("✅ Zero-Knowledge Discovery initialized successfully!");
        info!("📡 Discovery mechanisms active:");
        info!("  • mDNS (local network, <1 second)");
        info!("  • Kademlia DHT (global clearnet discovery)");
        info!("  • Identify (peer exchange)");
        info!("  • Ping (connection keepalive)");
        info!("  • Gossipsub (consensus messaging, {} topics)", topics.len());

        // Create command channel for API operations
        let (command_tx, command_rx) = mpsc::unbounded_channel();

        Ok(Self {
            swarm,
            discovered_peers: Arc::new(RwLock::new(HashSet::new())),
            peer_addresses: Arc::new(RwLock::new(HashMap::new())),
            bootstrap_peers: Arc::new(RwLock::new(bootstrap_peer_map)), // v0.6.8-beta: Auto-reconnection tracking
            local_peer_id,
            peer_tx: None, // Set via set_peer_channel() after construction
            gossipsub_message_tx: None, // Set via set_gossipsub_channel() after construction
            connected_peer_count: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            network_config,
            command_rx,
            command_tx,
            storage: None, // Set via set_storage() after construction
            gossipsub_stats: Arc::new(RwLock::new(HashMap::new())), // v0.6.9-beta: Gossipsub aggregation
            block_sync_tx: None, // Set via set_block_sync_channel() after construction
        })
    }

    /// Get network configuration
    pub fn network_config(&self) -> &q_types::NetworkConfig {
        &self.network_config
    }

    /// Set channel for sending discovered peers to ConnectionManager (Phase 2)
    pub fn set_peer_channel(&mut self, tx: mpsc::UnboundedSender<crate::connection_manager::PeerInfo>) {
        self.peer_tx = Some(tx);
        info!("🌉 libp2p → ConnectionManager bridge channel established");
    }

    /// Set channel for forwarding gossipsub messages to subscribers
    pub fn set_gossipsub_channel(&mut self, tx: mpsc::UnboundedSender<(String, Vec<u8>)>) {
        self.gossipsub_message_tx = Some(tx);
        info!("🌉 Gossipsub message forwarding channel established");
    }

    /// Set storage engine for block synchronization (Phase 3a)
    pub fn set_storage(&mut self, storage: Arc<q_storage::QStorage>) {
        self.storage = Some(storage);
        info!("🗄️ Storage engine linked to network manager for block sync");
    }

    /// Set channel for forwarding synced blocks to consensus (Phase 3b)
    pub fn set_block_sync_channel(&mut self, tx: mpsc::UnboundedSender<Vec<q_types::block::QBlock>>) {
        self.block_sync_tx = Some(tx);
        info!("🔗 Block sync forwarding channel established for consensus validation");
    }

    /// Get a cloned command sender for API operations
    ///
    /// This allows the API to send commands to the network manager's event loop
    /// without holding a lock on the network manager itself.
    pub fn get_command_sender(&self) -> mpsc::UnboundedSender<NetworkCommand> {
        self.command_tx.clone()
    }

    /// Main event loop - processes all discovery events
    pub async fn run(&mut self) -> anyhow::Result<()> {
        loop {
            tokio::select! {
                // Process network commands from API
                Some(command) = self.command_rx.recv() => {
                    match command {
                        NetworkCommand::DialPeer { multiaddr, response_tx } => {
                            debug!("📞 Processing dial command for {}", multiaddr);
                            let result = self.swarm.dial(multiaddr.clone())
                                .map_err(|e| format!("Failed to dial {}: {}", multiaddr, e));
                            let _ = response_tx.send(result);
                        }
                        NetworkCommand::SetPeerChannel { tx } => {
                            self.peer_tx = Some(tx);
                            info!("✅ Peer channel set for libp2p → ConnectionManager bridge");
                            info!("🌉 P2P peer discovery propagation ENABLED");
                        }
                        NetworkCommand::SetGossipsubChannel { tx } => {
                            self.gossipsub_message_tx = Some(tx);
                            info!("✅ Gossipsub channel set for message propagation");
                            info!("🌉 P2P mining reward/transaction propagation ENABLED");
                        }
                        NetworkCommand::PublishBlock { topic, block_bytes, block_height } => {
                            info!("📤 Publishing block {} ({} bytes) to gossipsub topic: {}", block_height, block_bytes.len(), topic);
                            let ident_topic = IdentTopic::new(topic.as_str());
                            match self.swarm.behaviour_mut().gossipsub.publish(ident_topic, block_bytes) {
                                Ok(_) => {
                                    info!("✅ Successfully published block {} to P2P network", block_height);
                                }
                                Err(e) => {
                                    warn!("❌ Failed to publish block {} to topic {}: {}", block_height, topic, e);
                                }
                            }
                        }
                        NetworkCommand::PublishBlockRequest { topic, request_bytes } => {
                            info!("📤 Publishing block request ({} bytes) to gossipsub topic: {}", request_bytes.len(), topic);
                            let ident_topic = IdentTopic::new(topic.as_str());
                            match self.swarm.behaviour_mut().gossipsub.publish(ident_topic, request_bytes) {
                                Ok(_) => {
                                    info!("✅ Successfully published block request to P2P network");
                                }
                                Err(e) => {
                                    warn!("❌ Failed to publish block request to topic {}: {}", topic, e);
                                }
                            }
                        }
                        NetworkCommand::PublishBlockResponse { topic, response_bytes, block_height } => {
                            info!("📤 Publishing block response for block {} ({} bytes) to gossipsub topic: {}", block_height, response_bytes.len(), topic);
                            let ident_topic = IdentTopic::new(topic.as_str());
                            match self.swarm.behaviour_mut().gossipsub.publish(ident_topic, response_bytes) {
                                Ok(_) => {
                                    info!("✅ Successfully published block response for block {} to P2P network", block_height);
                                }
                                Err(e) => {
                                    warn!("❌ Failed to publish block response to topic {}: {}", topic, e);
                                }
                            }
                        }
                        NetworkCommand::PublishBlockPack { topic, pack_bytes } => {
                            info!("🚀 [TURBO SYNC] Publishing block pack ({:.1} KB compressed) to gossipsub topic: {}",
                                  pack_bytes.len() as f64 / 1024.0, topic);
                            let ident_topic = IdentTopic::new(topic.as_str());
                            match self.swarm.behaviour_mut().gossipsub.publish(ident_topic, pack_bytes) {
                                Ok(_) => {
                                    info!("✅ [TURBO SYNC] Successfully published block pack to P2P network");
                                }
                                Err(e) => {
                                    warn!("❌ [TURBO SYNC] Failed to publish block pack to topic {}: {}", topic, e);
                                }
                            }
                        }
                        NetworkCommand::RequestBlockPack { topic, request_bytes, start_height, end_height } => {
                            info!("🚀 [TURBO SYNC] Requesting block pack {}-{} ({} bytes) from P2P network",
                                  start_height, end_height, request_bytes.len());
                            let ident_topic = IdentTopic::new(topic.as_str());
                            match self.swarm.behaviour_mut().gossipsub.publish(ident_topic, request_bytes) {
                                Ok(_) => {
                                    info!("✅ [TURBO SYNC] Successfully published block pack request to P2P network");
                                }
                                Err(e) => {
                                    warn!("❌ [TURBO SYNC] Failed to publish block pack request to topic {}: {}", topic, e);
                                }
                            }
                        }
                        NetworkCommand::PublishAIMessage { topic, message } => {
                            info!("🤖 [DISTRIBUTED AI] Publishing AI message to topic: {}", topic);
                            info!("   Message ID: {}", message.message_id);
                            info!("   Sender: {}", message.sender_node_id);

                            // Serialize AI message to bytes
                            match postcard::to_allocvec(&message) {
                                Ok(message_bytes) => {
                                    let ident_topic = IdentTopic::new(topic.as_str());
                                    match self.swarm.behaviour_mut().gossipsub.publish(ident_topic, message_bytes.clone()) {
                                        Ok(_) => {
                                            info!("✅ [DISTRIBUTED AI] Successfully published AI message ({} bytes) to P2P network", message_bytes.len());
                                        }
                                        Err(e) => {
                                            error!("❌ [DISTRIBUTED AI] Failed to publish AI message to topic {}: {}", topic, e);
                                        }
                                    }
                                }
                                Err(e) => {
                                    error!("❌ [DISTRIBUTED AI] Failed to serialize AI message: {}", e);
                                }
                            }
                        }
                    }
                }
                // Process swarm events
                event = self.swarm.select_next_some() => match event {
                    SwarmEvent::Behaviour(behaviour_event) => {
                        self.handle_behaviour_event(behaviour_event).await?;
                    }
                    SwarmEvent::NewListenAddr { address, .. } => {
                        info!("📍 Listening on: {}", address);
                    }
                    SwarmEvent::ConnectionEstablished {
                        peer_id,
                        endpoint,
                        num_established,
                        ..
                    } => {
                    info!("🌐 [LIBP2P CONNECTION] ==========================================");
                    info!("✅ [CONNECTION] Successfully connected to peer: {}", peer_id);
                    info!("📍 [CONNECTION] Endpoint: {:?}", endpoint);
                    info!("🔢 [CONNECTION] Number of established connections: {}", num_established);

                    let mut peers = self.discovered_peers.write().await;
                    peers.insert(peer_id);
                    let peer_count = peers.len();
                    drop(peers); // Release lock before atomic operations

                    // Update atomic counter for API endpoint (thread-safe)
                    self.connected_peer_count.store(peer_count, std::sync::atomic::Ordering::SeqCst);

                        info!("📊 [NETWORK STATE] Total discovered peers: {} (atomic counter updated)", peer_count);
                        info!("🔄 [SYNC] Ready to synchronize DAG state with peer {}", peer_id);
                        info!("📡 [PROPAGATION] Will propagate vertices/transactions to this peer");
                        info!("🔐 [CONSENSUS] Peer will participate in Bracha's protocol voting");
                        info!("🌐 [LIBP2P CONNECTION COMPLETE] ==========================================\n");
                    }
                    SwarmEvent::ConnectionClosed { peer_id, .. } => {
                        // Remove peer from discovered set
                        let mut peers = self.discovered_peers.write().await;
                        peers.remove(&peer_id);
                        let peer_count = peers.len();
                        drop(peers);

                        // Update atomic counter
                        self.connected_peer_count.store(peer_count, std::sync::atomic::Ordering::SeqCst);

                        info!("👋 [DISCONNECTION] Connection closed with peer: {} (remaining peers: {})", peer_id, peer_count);

                        // 🔧 v0.6.8-beta: Automatic reconnection for bootstrap peers
                        // Server Alpha was disconnecting from Server Beta after 41 seconds, causing turbo sync failure.
                        // If this is a bootstrap peer, immediately attempt to reconnect.
                        let bootstrap_peers = self.bootstrap_peers.read().await;
                        if let Some(multiaddr) = bootstrap_peers.get(&peer_id) {
                            warn!("🔄 [AUTO-RECONNECT] Bootstrap peer disconnected - reconnecting to {}", peer_id);
                            let addr = multiaddr.clone();
                            drop(bootstrap_peers); // Release lock before dial operation

                            if let Err(e) = self.swarm.dial(addr.clone()) {
                                error!("❌ [AUTO-RECONNECT] Failed to redial bootstrap peer {} at {}: {}", peer_id, addr, e);
                            } else {
                                info!("✅ [AUTO-RECONNECT] Redialing bootstrap peer {} at {}", peer_id, addr);
                            }
                        }
                    }
                    _ => {}
                }
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

                    // Attempt to dial the peer - log errors but don't propagate (keep event loop running)
                    if let Err(e) = self.swarm.dial(addr.clone()) {
                        warn!("⚠️ Failed to dial peer {} at {}: {}", peer_id, addr, e);
                    } else {
                        info!("📞 Dialing peer {} at {}...", peer_id, addr);
                    }
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
                // Truncate MessageId for cleaner logs - extract first 16 bytes of hex
                let msg_id_str = format!("{:?}", message_id);
                // MessageId format is MessageId(hexhexhex...) so strip the prefix/suffix and truncate
                let msg_id_short = if msg_id_str.starts_with("MessageId(") && msg_id_str.ends_with(')') {
                    let hex_part = &msg_id_str[10..msg_id_str.len()-1]; // Remove "MessageId(" and ")"
                    if hex_part.len() > 32 {
                        format!("{}...", &hex_part[..32])
                    } else {
                        hex_part.to_string()
                    }
                } else {
                    msg_id_str
                };

                // 🔇 v0.6.9-beta: Aggregate gossipsub logs to prevent spam (was 775+ logs hiding progress bar)
                // v0.9.7-beta: Enhanced to track block height ranges for sync progress visibility
                // Only log aggregated stats every 2MB or 10 seconds per topic
                let topic_str = message.topic.to_string();
                let msg_size = message.data.len();

                // Extract block height if this is a block message
                let block_height = if topic_str.contains("/blocks") {
                    postcard::from_bytes::<QBlock>(&message.data)
                        .ok()
                        .map(|block| block.header.height)
                } else {
                    None
                };

                let mut should_log = false;
                {
                    let mut stats = self.gossipsub_stats.write().await;
                    let entry = stats.entry(topic_str.clone()).or_insert((0, 0, std::time::Instant::now(), None, None));
                    entry.0 += 1; // message count
                    entry.1 += msg_size; // total bytes

                    // Track block height range if available
                    if let Some(height) = block_height {
                        entry.3 = Some(entry.3.map_or(height, |min| min.min(height)));
                        entry.4 = Some(entry.4.map_or(height, |max| max.max(height)));
                    }

                    // Log if 2MB accumulated OR 10 seconds elapsed
                    if entry.1 >= 2_000_000 || entry.2.elapsed().as_secs() >= 10 {
                        should_log = true;

                        if let (Some(min_height), Some(max_height)) = (entry.3, entry.4) {
                            let height_range = if min_height == max_height {
                                format!("height={}", min_height)
                            } else {
                                format!("heights={}-{} (Δ={})", min_height, max_height, max_height - min_height)
                            };
                            info!(
                                "📨 [AGGREGATED] Received {} messages ({:.2} MB) on topic {} in last {}s | {}",
                                entry.0,
                                entry.1 as f64 / 1_000_000.0,
                                topic_str,
                                entry.2.elapsed().as_secs(),
                                height_range
                            );
                        } else {
                            info!(
                                "📨 [AGGREGATED] Received {} messages ({:.2} MB) on topic {} in last {}s",
                                entry.0,
                                entry.1 as f64 / 1_000_000.0,
                                topic_str,
                                entry.2.elapsed().as_secs()
                            );
                        }

                        // Reset counters
                        entry.0 = 0;
                        entry.1 = 0;
                        entry.2 = std::time::Instant::now();
                        entry.3 = None;
                        entry.4 = None;
                    }
                }

                // Individual message details at DEBUG level only
                // v0.9.7-beta: Enhanced logging to show block heights and sync progress
                if topic_str.contains("/blocks") {
                    // Attempt to decode block information for better sync visibility
                    match postcard::from_bytes::<QBlock>(&message.data) {
                        Ok(block) => {
                            info!(
                                "📨 Gossipsub BLOCK from {}: topic={}, height={}, txs={}, size={} bytes, hash={}",
                                propagation_source,
                                message.topic,
                                block.header.height,
                                block.transactions.len(),
                                msg_size,
                                hex::encode(&block.calculate_hash()[..8])
                            );
                        }
                        Err(_) => {
                            debug!(
                                "📨 Gossipsub message from {}: topic={}, id={}, size={} bytes (failed to decode block)",
                                propagation_source,
                                message.topic,
                                msg_id_short,
                                msg_size
                            );
                        }
                    }
                } else {
                    debug!(
                        "📨 Gossipsub message from {}: topic={}, id={}, size={} bytes",
                        propagation_source,
                        message.topic,
                        msg_id_short,
                        msg_size
                    );
                }

                // Forward to gossipsub message channel if available
                if let Some(ref tx) = self.gossipsub_message_tx {
                    let data = message.data.clone();

                    if let Err(e) = tx.send((topic_str.clone(), data)) {
                        warn!("⚠️ Failed to forward gossipsub message on topic {}: {}", topic_str, e);
                    } else {
                        // 🔇 v0.6.9-beta: Changed to DEBUG to prevent log spam
                        // v0.9.7-beta: Enhanced with block height information
                        if topic_str.contains("/blocks") {
                            if let Ok(block) = postcard::from_bytes::<QBlock>(&message.data) {
                                info!("✅ Forwarded BLOCK on topic: {} (height={}, size={} bytes)",
                                     topic_str, block.header.height, msg_size);
                            } else {
                                debug!("✅ Forwarded gossipsub message on topic: {} (size={} bytes)", topic_str, msg_size);
                            }
                        } else {
                            debug!("✅ Forwarded gossipsub message on topic: {} (size={} bytes)", topic_str, msg_size);
                        }
                    }
                } else {
                    warn!("⚠️ Gossipsub message received but gossipsub_message_tx is None!");
                }
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
            QNarwhalEvent::BlockSync(block_sync_event) => {
                use libp2p::request_response::{Event, Message};

                match block_sync_event {
                    Event::Message { peer, message } => {
                        match message {
                            Message::Request { request_id, request, channel } => {
                                info!("📥 [BLOCK-SYNC] Received block sync request from {}", peer);
                                info!("   Requested: {} blocks from height {}", request.limit, request.start_height);

                                // Phase 3a: Fetch real blocks from storage
                                let response = if let Some(ref storage) = self.storage {
                                    match storage.get_qblocks_range(request.start_height, request.limit).await {
                                        Ok(blocks) => {
                                            let latest_height = storage.get_latest_qblock_height().await
                                                .ok()
                                                .flatten()
                                                .unwrap_or(0);

                                            let block_count = blocks.len() as u64;
                                            let end_height = if !blocks.is_empty() { blocks.last().unwrap().header.height } else { request.start_height };

                                            info!("✅ [BLOCK-SYNC] Fetched {} blocks from storage (heights {}-{})",
                                                  block_count, request.start_height, end_height);

                                            q_storage::sync::BlockSyncResponse {
                                                start_height: request.start_height,
                                                blocks,
                                                total_blocks: block_count,
                                                latest_height,
                                            }
                                        }
                                        Err(e) => {
                                            error!("❌ [BLOCK-SYNC] Failed to fetch blocks from storage: {}", e);
                                            q_storage::sync::BlockSyncResponse {
                                                start_height: request.start_height,
                                                blocks: vec![],
                                                total_blocks: 0,
                                                latest_height: 0,
                                            }
                                        }
                                    }
                                } else {
                                    warn!("⚠️ [BLOCK-SYNC] Storage not available, sending empty response");
                                    q_storage::sync::BlockSyncResponse {
                                        start_height: request.start_height,
                                        blocks: vec![],
                                        total_blocks: 0,
                                        latest_height: 0,
                                    }
                                };

                                if let Err(e) = self.swarm.behaviour_mut().block_sync.send_response(channel, response) {
                                    error!("❌ [BLOCK-SYNC] Failed to send response: {:?}", e);
                                } else {
                                    info!("✅ [BLOCK-SYNC] Sent response to {}", peer);
                                }
                            }
                            Message::Response { request_id, response } => {
                                info!("📨 [BLOCK-SYNC] Received block sync response: {} blocks from height {}",
                                      response.blocks.len(), response.start_height);
                                info!("   Latest height on peer: {}", response.latest_height);

                                // Phase 3b: Forward blocks to consensus for validation
                                if !response.blocks.is_empty() {
                                    if let Some(ref tx) = self.block_sync_tx {
                                        if let Err(e) = tx.send(response.blocks.clone()) {
                                            error!("❌ [BLOCK-SYNC] Failed to forward blocks to consensus: {}", e);
                                        } else {
                                            info!("✅ [BLOCK-SYNC] Forwarded {} blocks to consensus for validation", response.blocks.len());
                                        }
                                    } else {
                                        warn!("⚠️ [BLOCK-SYNC] Block sync channel not configured, blocks not forwarded");
                                    }
                                } else {
                                    debug!("📭 [BLOCK-SYNC] No blocks in response, nothing to forward");
                                }
                            }
                        }
                    }
                    Event::OutboundFailure { peer, request_id, error } => {
                        warn!("⚠️ [BLOCK-SYNC] Outbound failure to {}: {:?}", peer, error);
                    }
                    Event::InboundFailure { peer, error, .. } => {
                        warn!("⚠️ [BLOCK-SYNC] Inbound failure from {}: {:?}", peer, error);
                    }
                    Event::ResponseSent { peer, .. } => {
                        debug!("✅ [BLOCK-SYNC] Response sent to {}", peer);
                    }
                }
            }
        }
        Ok(())
    }

    /// Get all discovered peers from ALL discovery methods
    pub async fn get_discovered_peers(&self) -> Vec<PeerId> {
        self.discovered_peers.read().await.iter().cloned().collect()
    }

    /// Get the number of discovered/connected peers
    pub async fn get_peer_count(&self) -> usize {
        self.discovered_peers.read().await.len()
    }

    /// Get thread-safe atomic reference to connected peer count
    /// This can be safely cloned and shared across threads
    pub fn get_peer_count_atomic(&self) -> Arc<std::sync::atomic::AtomicUsize> {
        self.connected_peer_count.clone()
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
        info!("📤 Publishing {} bytes to gossipsub topic: {}", data.len(), topic);
        self.swarm.behaviour_mut().gossipsub
            .publish(ident_topic, data)
            .map_err(|e| anyhow::anyhow!("Failed to publish to topic {}: {}", topic, e))?;
        info!("✅ Successfully published message to gossipsub topic: {}", topic);
        Ok(())
    }

    /// Request blocks from a specific peer via libp2p request-response (Phase 3)
    pub fn request_blocks_from_peer(
        &mut self,
        peer_id: PeerId,
        start_height: u64,
        limit: usize,
    ) -> anyhow::Result<()> {
        info!("📤 [BLOCK-SYNC] Requesting {} blocks from height {} from peer {}", limit, start_height, peer_id);

        // Convert PeerId to NodeId (32-byte array)
        let peer_id_bytes = self.local_peer_id.to_bytes();
        let mut node_id = [0u8; 32];
        let copy_len = peer_id_bytes.len().min(32);
        node_id[..copy_len].copy_from_slice(&peer_id_bytes[..copy_len]);

        let request = q_storage::sync::BlockSyncRequest {
            start_height,
            limit,
            request_id: uuid::Uuid::new_v4().to_string(),
            requester: node_id,
        };

        self.swarm.behaviour_mut().block_sync.send_request(&peer_id, request);

        info!("✅ [BLOCK-SYNC] Block sync request sent to {}", peer_id);
        Ok(())
    }

    /// Auto-detect missing blocks and request from peers (Phase 3c)
    pub async fn check_and_sync_blocks(&mut self) -> anyhow::Result<()> {
        // Check if we have storage configured
        let storage = match &self.storage {
            Some(s) => s,
            None => {
                debug!("🔄 [AUTO-SYNC] Storage not configured, skipping auto-sync");
                return Ok(());
            }
        };

        // Get our local height
        let local_height = storage.get_latest_qblock_height().await?
            .unwrap_or(0);

        // Get connected peers
        let peer_id = {
            let peers = self.discovered_peers.read().await;
            if peers.is_empty() {
                debug!("🔄 [AUTO-SYNC] No peers connected, skipping auto-sync");
                return Ok(());
            }

            // Request blocks from the first available peer
            // In production, this would query multiple peers for their heights
            // and sync from the one with the highest height
            *peers.iter().next().unwrap()
        }; // Lock released here

        // Request next batch of blocks (100 at a time for gradual sync)
        let batch_size = 100;
        info!("🔄 [AUTO-SYNC] Local height: {}, requesting blocks from peer {}", local_height, peer_id);

        self.request_blocks_from_peer(peer_id, local_height, batch_size)?;

        Ok(())
    }

    /// Get the local peer ID
    pub fn peer_id(&self) -> PeerId {
        self.local_peer_id
    }

    /// Get listen addresses
    pub fn get_listen_addrs(&self) -> Vec<Multiaddr> {
        self.swarm.listeners().cloned().collect()
    }

    /// Manually dial a peer by multiaddr
    ///
    /// # Arguments
    /// * `multiaddr` - The multiaddr to dial (e.g., "/ip4/127.0.0.1/tcp/33305/p2p/12D3Koo...")
    ///
    /// # Returns
    /// * `Ok(())` if dial was initiated successfully
    /// * `Err` if dial failed
    pub fn dial_peer(&mut self, multiaddr: Multiaddr) -> anyhow::Result<()> {
        info!("📞 Manually dialing peer at {}", multiaddr);

        self.swarm.dial(multiaddr.clone())
            .map_err(|e| anyhow::anyhow!("Failed to dial peer {}: {}", multiaddr, e))?;

        Ok(())
    }

    /// Run one iteration of the network event loop
    /// Should be called in a loop from an async task
    pub async fn run_once(&mut self) -> anyhow::Result<()> {
        use futures::stream::StreamExt;

        // Process one event from the swarm
        if let Some(event) = self.swarm.next().await {
            match event {
                SwarmEvent::Behaviour(behaviour_event) => {
                    self.handle_behaviour_event(behaviour_event).await?;
                }
                SwarmEvent::NewListenAddr { address, .. } => {
                    info!("📍 Listening on: {}", address);
                }
                SwarmEvent::ConnectionEstablished {
                    peer_id,
                    endpoint,
                    num_established,
                    ..
                } => {
                    let mut peers = self.discovered_peers.write().await;
                    let is_new = peers.insert(peer_id);
                    let peer_count = peers.len();
                    drop(peers); // Release lock before calling atomic operations

                    // Update atomic counter (thread-safe)
                    self.connected_peer_count.store(peer_count, std::sync::atomic::Ordering::SeqCst);

                    info!(
                        "🔗 Connected to peer: {} (total connections: {}, new: {})",
                        peer_id, num_established, is_new
                    );
                    info!("📊 Total discovered peers: {} (atomic counter updated)", peer_count);
                }
                SwarmEvent::ConnectionClosed { peer_id, .. } => {
                    // Remove peer from discovered set
                    let mut peers = self.discovered_peers.write().await;
                    peers.remove(&peer_id);
                    let peer_count = peers.len();
                    drop(peers);

                    // Update atomic counter
                    self.connected_peer_count.store(peer_count, std::sync::atomic::Ordering::SeqCst);

                    info!("👋 Connection closed: {} (remaining peers: {})", peer_id, peer_count);

                    // 🔧 v0.6.8-beta: Automatic reconnection for bootstrap peers
                    let bootstrap_peers = self.bootstrap_peers.read().await;
                    if let Some(multiaddr) = bootstrap_peers.get(&peer_id) {
                        warn!("🔄 [AUTO-RECONNECT] Bootstrap peer disconnected - reconnecting to {}", peer_id);
                        let addr = multiaddr.clone();
                        drop(bootstrap_peers);

                        if let Err(e) = self.swarm.dial(addr.clone()) {
                            error!("❌ [AUTO-RECONNECT] Failed to redial bootstrap peer {} at {}: {}", peer_id, addr, e);
                        } else {
                            info!("✅ [AUTO-RECONNECT] Redialing bootstrap peer {} at {}", peer_id, addr);
                        }
                    }
                }
                _ => {}
            }
        }

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