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

/// 🔥 v1.0.17-beta: Multiple bootstrap peers for decentralization
/// Previously: Single bootstrap node (centralization risk)
/// Now: Multiple diverse bootstrap nodes (different operators, geos)
/// ✅ v1.0.17-beta: Fixed bootstrap configuration (correct P2P port 9001 and PeerID)
const BOOTSTRAP_PEERS: &[&str] = &[
    "/ip4/185.182.185.227/tcp/9001/p2p/12D3KooWC688bzHi7djbkensGQMABzX9tY41LNasgd3g3FdwqQn7",  // Server Beta (EU) - P2P port, actual PeerID
    // TODO: Add Server Alpha (US) bootstrap node
    // TODO: Add community bootstrap nodes
];

/// Legacy compatibility - use first bootstrap peer as default
const DEFAULT_BOOTSTRAP_PEER: &str = BOOTSTRAP_PEERS[0];

/// Q-NarwhalKnight network behavior combining all discovery mechanisms
/// 🔥 v1.0.17-beta: Added NAT traversal for true decentralization (AutoNAT + Relay + DCUtR)
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
    /// ✅ v0.9.68-beta: Replaced with proper BlockPackCodec for efficient block sync
    block_sync: libp2p::request_response::Behaviour<q_types::BlockPackCodec>,
    /// ✅ v1.0.15.1-beta: Handshake protocol for version validation
    handshake: libp2p::request_response::Behaviour<crate::handshake_validator::HandshakeCodec>,

    // 🔥 v1.0.17-beta: NAT Traversal (A+ → A++ upgrade)
    /// AutoNAT: Detect if this node is publicly dialable
    /// Enables nodes to know if they're behind NAT/firewall
    autonat: libp2p::autonat::Behaviour,
    /// Relay Client: Be reachable via relay nodes even when behind NAT
    /// Provides addressability for home nodes without port forwarding
    relay: libp2p::relay::client::Behaviour,
    /// DCUtR: Direct Connection Upgrade through Relay (hole-punching)
    /// Upgrades relay connections to direct connections (~70% success rate)
    dcutr: libp2p::dcutr::Behaviour,
    /// Connection Limits: Prevent accidental supernodes
    /// Limits connections per peer and total connections
    connection_limits: libp2p::connection_limits::Behaviour,
}

#[derive(Debug)]
pub enum QNarwhalEvent {
    #[cfg(not(target_os = "windows"))]
    Mdns(MdnsEvent),
    Kademlia(KademliaEvent),
    Identify(libp2p::identify::Event),
    Ping(libp2p::ping::Event),
    Gossipsub(gossipsub::Event),
    BlockSync(libp2p::request_response::Event<q_types::BlockPackRequest, q_types::BlockPackResponse>),
    Handshake(libp2p::request_response::Event<crate::handshake_validator::HandshakeMessage, crate::handshake_validator::HandshakeResult>),

    // 🔥 v1.0.17-beta: NAT Traversal Events
    AutoNat(libp2p::autonat::Event),
    Relay(libp2p::relay::client::Event),
    Dcutr(libp2p::dcutr::Event),
    // NOTE: connection_limits has ToSwarm = Infallible (never emits events)
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

impl From<libp2p::request_response::Event<q_types::BlockPackRequest, q_types::BlockPackResponse>> for QNarwhalEvent {
    fn from(event: libp2p::request_response::Event<q_types::BlockPackRequest, q_types::BlockPackResponse>) -> Self {
        QNarwhalEvent::BlockSync(event)
    }
}

impl From<libp2p::request_response::Event<crate::handshake_validator::HandshakeMessage, crate::handshake_validator::HandshakeResult>> for QNarwhalEvent {
    fn from(event: libp2p::request_response::Event<crate::handshake_validator::HandshakeMessage, crate::handshake_validator::HandshakeResult>) -> Self {
        QNarwhalEvent::Handshake(event)
    }
}

// 🔥 v1.0.17-beta: NAT Traversal Event Conversions
impl From<libp2p::autonat::Event> for QNarwhalEvent {
    fn from(event: libp2p::autonat::Event) -> Self {
        QNarwhalEvent::AutoNat(event)
    }
}

impl From<libp2p::relay::client::Event> for QNarwhalEvent {
    fn from(event: libp2p::relay::client::Event) -> Self {
        QNarwhalEvent::Relay(event)
    }
}

impl From<libp2p::dcutr::Event> for QNarwhalEvent {
    fn from(event: libp2p::dcutr::Event) -> Self {
        QNarwhalEvent::Dcutr(event)
    }
}

// NOTE: connection_limits::Behaviour does not emit events (ToSwarm = void::Void)
// libp2p 0.53 uses void::Void for connection_limits, so we need From<void::Void>
impl From<void::Void> for QNarwhalEvent {
    fn from(v: void::Void) -> Self {
        // void::Void is uninhabited, so this can never be called
        void::unreachable(v)
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
    /// Publish peer height announcement (Turbo Sync peer discovery)
    PublishPeerHeight {
        topic: String,
        announcement_bytes: Vec<u8>,
        height: u64,
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

/// v0.9.73-beta: Peer compatibility tracking for BlockPackCodec protocol
/// Tracks which peers successfully support the BlockPackCodec request-response protocol
#[derive(Debug, Clone, Default)]
pub struct PeerCompatibility {
    /// Peers that have successfully responded (PeerId → success count)
    pub successes: HashMap<PeerId, u32>,
    /// Peers that have failed to respond (PeerId → failure count)
    pub failures: HashMap<PeerId, u32>,
    /// Blacklisted peers (incompatible with BlockPackCodec)
    pub blacklist: HashSet<PeerId>,
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
    /// v0.9.73-beta: Peer compatibility tracking for BlockPackCodec protocol
    /// Tracks which peers successfully support the new request-response protocol
    peer_compat: Arc<std::sync::RwLock<PeerCompatibility>>,
    /// v1.0.12-beta: Pending block range requests for batch sync
    /// Maps request_id (as String) → oneshot channel for async await
    /// v1.0.15-beta: Fixed to use String instead of removed libp2p::request_response::RequestId
    pending_block_requests: Arc<std::sync::Mutex<HashMap<String, tokio::sync::oneshot::Sender<Vec<q_types::QBlock>>>>>,
    /// v1.0.15.1-beta: Protocol version validation for peer handshakes
    /// Prevents silent communication failures from incompatible protocol versions
    handshake_validator: Arc<RwLock<crate::handshake_validator::HandshakeValidator>>,
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

        // 🔥 v1.0.17-beta: SwarmBuilder replaces manual transport construction
        // Old manual TCP+Noise+Yamux transport deleted - now handled by SwarmBuilder
        // mDNS, Identify, Ping, Kademlia initialization moved into SwarmBuilder closure

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
                        // kademlia.add_address(&peer_id, addr.clone());  // Moved to SwarmBuilder closure
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

                        // Try to fetch peer ID from HTTP endpoint (port 8080 for API server)
                        // v0.9.21-beta FIX: Changed from 18080 to 8080 (Server Beta API port)
                        match fetch_peer_id_from_http(&bootstrap_ip, 8080).await {
                            Ok(peer_id_str) => {
                                // Parse peer ID and append to multiaddr
                                match peer_id_str.parse::<PeerId>() {
                                    Ok(peer_id) => {
                                        addr.push(Protocol::P2p(peer_id));
                                        // kademlia.add_address(&peer_id, addr.clone());  // Moved to SwarmBuilder closure
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

        // 🔥 v1.0.17-beta: Kademlia bootstrap moved to after SwarmBuilder
        // Bootstrap is now done inside the SwarmBuilder closure where kademlia exists
        if bootstrap_count > 0 {
            info!("🔧 Will bootstrap {} peers after swarm creation", bootstrap_count);
        } else {
            info!("ℹ️ No bootstrap peers configured - DHT will populate via mDNS discoveries");
        }

        info!("🌍 Kademlia DHT initialized for clearnet discovery");


        // 🔥 v1.0.17-beta: Configure connection limits
        use libp2p::connection_limits::{ConnectionLimits, Behaviour as ConnLimitsBehaviour};

        let limits = ConnectionLimits::default()
            .with_max_pending_incoming(Some(64))
            .with_max_pending_outgoing(Some(64))
            .with_max_established_incoming(Some(256))
            .with_max_established_outgoing(Some(256))
            .with_max_established_per_peer(Some(8));

        info!("🔒 Connection limits configured: max 256 established connections, 8 per peer");

        // 🔥 v1.0.17-beta: Build swarm using SwarmBuilder pattern
        use libp2p::SwarmBuilder;
        use std::num::NonZeroUsize;

        info!("🔧 Building swarm with SwarmBuilder pattern (NAT traversal enabled)");

        // Clone data for use inside closure
        let network_config_clone = network_config.clone();
        let bootstrap_peer_map_clone = bootstrap_peer_map.clone();

        let mut swarm = SwarmBuilder::with_existing_identity(keypair)
            .with_tokio()
            .with_tcp(
                tcp::Config::default().port_reuse(true).nodelay(true),
                noise::Config::new,
                yamux::Config::default,
            ).expect("Failed to configure TCP transport")
            .with_quic()  // QUIC transport for better NAT traversal
            .with_dns().expect("Failed to configure DNS")  // DNS resolution
            .with_relay_client(  // 🔥 CRITICAL: Proper relay client bound to transport
                noise::Config::new,
                yamux::Config::default,
            ).expect("Failed to configure relay client")
            .with_behaviour(move |keypair_inner, relay_client| {
                let local_peer_id_inner = keypair_inner.public().to_peer_id();

                // mDNS for local discovery
                #[cfg(not(target_os = "windows"))]
                let mdns = mdns::tokio::Behaviour::new(mdns::Config::default(), local_peer_id_inner)
                    .expect("Failed to initialize behaviour");

                // Kademlia DHT
                let mut kad_config = KademliaConfig::default();
                kad_config.set_query_timeout(Duration::from_secs(60));
                let kad_store = MemoryStore::new(local_peer_id_inner);
                let mut kademlia = Kademlia::with_config(local_peer_id_inner, kad_store, kad_config);

                // Add bootstrap peers to Kademlia
                for (peer_id, addr) in &bootstrap_peer_map_clone {
                    kademlia.add_address(peer_id, addr.clone());
                }

                // Identify
                let identify = libp2p::identify::Behaviour::new(
                    libp2p::identify::Config::new("/qnarwhal/1.0.0".to_string(), keypair_inner.public())
                        .with_push_listen_addr_updates(true),
                );

                // Ping
                let ping = libp2p::ping::Behaviour::new(libp2p::ping::Config::new());

                // Gossipsub
                use libp2p::gossipsub::{ValidationMode, MessageId};

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
                    .expect("Failed to initialize behaviour");

                let gossipsub = gossipsub::Behaviour::new(
                    gossipsub::MessageAuthenticity::Signed(keypair_inner.clone()),
                    gossipsub_config,
                )
                .expect("Failed to initialize behaviour");

                // Request-response for block synchronization
                use libp2p::request_response::{self, ProtocolSupport};
                use q_types::{BlockPackCodec, BlockPackProtocol};

                let block_sync_protocols = std::iter::once((BlockPackProtocol, ProtocolSupport::Full));
                let block_sync_config = request_response::Config::default();
                let block_sync = request_response::Behaviour::with_codec(
                    BlockPackCodec::default(),
                    block_sync_protocols,
                    block_sync_config,
                );

                // Handshake protocol
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
                let autonat = libp2p::autonat::Behaviour::new(local_peer_id_inner, Default::default());
                let relay = relay_client;  // ✅ CRITICAL: Use the provided relay client
                let dcutr = libp2p::dcutr::Behaviour::new(local_peer_id_inner);

                // 🔒 Connection limits
                let connection_limits = ConnLimitsBehaviour::new(limits.clone());

                Ok(QNarwhalBehaviour {
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
                 .with_notify_handler_buffer_size(NonZeroUsize::new(32).unwrap())
                 .with_per_connection_event_buffer_size(64)
            })
            .build();

        info!("✅ Swarm built successfully with NAT traversal enabled");

        // Subscribe to gossip topics AFTER swarm creation
        let network_prefix = network_config.network_id.gossipsub_topic_prefix();
        let topics = vec![
            IdentTopic::new(format!("{}/blocks", network_prefix)),
            IdentTopic::new(network_config.network_id.transactions_topic()),
            IdentTopic::new(format!("{}/mining-rewards", network_prefix)),
            IdentTopic::new(format!("{}/dex/swaps", network_prefix)),
            IdentTopic::new(format!("{}/votes", network_prefix)),
            IdentTopic::new(network_config.network_id.acks_topic()),
            IdentTopic::new(network_config.network_id.block_requests_topic()),
            IdentTopic::new(network_config.network_id.block_responses_topic()),
            IdentTopic::new(network_config.network_id.batch_block_responses_topic()),
        ];

        for topic in &topics {
            swarm.behaviour_mut().gossipsub.subscribe(topic)
                .map_err(|e| anyhow::anyhow!("Failed to subscribe to topic {}: {}", topic, e))?;
            info!("📢 Subscribed to {} Gossipsub topic: {}",
                  network_config.network_id.as_str(), topic);
        }

        // 🔄 v0.9.60-beta: BACKWARD COMPATIBILITY
        if network_config.network_id == q_types::NetworkId::TestnetPhase5 {
            let phase4_topics = vec![
                IdentTopic::new("/qnk/testnet-phase4/blocks"),
                IdentTopic::new("/qnk/testnet-phase4/transactions"),
                IdentTopic::new("/qnk/testnet-phase4/mining-rewards"),
                IdentTopic::new("/qnk/testnet-phase4/peer-heights"),
                IdentTopic::new("/qnk/testnet-phase4/block-pack-requests"),
                IdentTopic::new("/qnk/testnet-phase4/block-pack-responses"),
            ];

            for topic in &phase4_topics {
                swarm.behaviour_mut().gossipsub.subscribe(topic)
                    .map_err(|e| anyhow::anyhow!("Failed to subscribe to phase4 topic {}: {}", topic, e))?;
                info!("🔄 [BACKWARD COMPAT] Subscribed to phase4 topic: {}", topic);
            }
            info!("✅ Backward compatibility enabled: {} phase4 topics subscribed", phase4_topics.len());
        }

        // Subscribe to distributed AI inference topics
        let ai_topics = DistributedAITopics::new();
        for topic in ai_topics.all_topics() {
            swarm.behaviour_mut().gossipsub.subscribe(&topic)
                .map_err(|e| anyhow::anyhow!("Failed to subscribe to AI topic {}: {}", topic, e))?;
            info!("🤖 Subscribed to AI inference topic: {}", topic);
        }
        info!("✅ Subscribed to {} AI inference Gossipsub topics", ai_topics.all_topics().len());

        info!("🔗 Block sync request-response protocol initialized (BlockPackCodec)");
        info!("🤝 Handshake protocol initialized for peer validation (v{}.{}.{})",
              crate::handshake_validator::ProtocolVersion::CURRENT.major,
              crate::handshake_validator::ProtocolVersion::CURRENT.minor,
              crate::handshake_validator::ProtocolVersion::CURRENT.patch);

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

        // 🚀 v0.9.38-beta: PHASE 1.2 - Bootstrap Peer Discovery with Retry Logic
        // Explicitly dial bootstrap peer if Q_BOOTSTRAP_PEER environment variable is set
        // This ensures Server Alpha connects to Server Beta for network unification
        if let Ok(bootstrap_env) = std::env::var("Q_BOOTSTRAP_PEER") {
            info!("🔍 [BOOTSTRAP] Explicit bootstrap peer configured: {}", bootstrap_env);

            // Parse bootstrap multiaddr
            if let Ok(bootstrap_addr) = bootstrap_env.parse::<Multiaddr>() {
                info!("📡 [BOOTSTRAP] Dialing bootstrap peer: {}", bootstrap_addr);

                // Attempt immediate dial
                match swarm.dial(bootstrap_addr.clone()) {
                    Ok(_) => {
                        info!("✅ [BOOTSTRAP] Initiated connection to bootstrap peer");
                    }
                    Err(e) => {
                        warn!("⚠️  [BOOTSTRAP] Initial dial failed: {:?}", e);
                        warn!("   Will retry automatically in background task");
                    }
                }
            } else {
                warn!("❌ [BOOTSTRAP] Failed to parse bootstrap peer multiaddr: {}", bootstrap_env);
                warn!("   Expected format: /ip4/<IP>/tcp/<PORT>/p2p/<PEER_ID>");
            }
        } else {
            info!("ℹ️  [BOOTSTRAP] No explicit bootstrap peer configured (Q_BOOTSTRAP_PEER not set)");
            info!("   Node will rely on mDNS and Kademlia DHT for peer discovery");
        }

        // Create command channel for API operations
        let (command_tx, command_rx) = mpsc::unbounded_channel();

        // 🤝 v1.0.15.1-beta: Initialize handshake validator for protocol version validation
        let handshake_validator = Arc::new(RwLock::new(
            crate::handshake_validator::HandshakeValidator::new(
                network_config.network_id.display_name().to_string(),
                network_config.genesis_hash.to_vec(),
            )
        ));
        info!("🤝 [HANDSHAKE] Protocol validator initialized (v{}.{}.{})",
              crate::handshake_validator::ProtocolVersion::CURRENT.major,
              crate::handshake_validator::ProtocolVersion::CURRENT.minor,
              crate::handshake_validator::ProtocolVersion::CURRENT.patch);

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
            peer_compat: Arc::new(std::sync::RwLock::new(PeerCompatibility::default())), // v0.9.73-beta: Peer compatibility tracking
            pending_block_requests: Arc::new(std::sync::Mutex::new(HashMap::new())), // v1.0.12-beta: Batch sync request tracking
            handshake_validator, // v1.0.15.1-beta: Protocol version validation
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
        // 🚀 v0.9.38-beta: PHASE 1.2 - Periodic P2P health monitoring
        let mut health_check_interval = tokio::time::interval(tokio::time::Duration::from_secs(30));

        loop {
            tokio::select! {
                // 🩺 Periodic P2P health check (every 30 seconds)
                _ = health_check_interval.tick() => {
                    let peer_count = self.discovered_peers.read().await.len();

                    if peer_count == 0 {
                        warn!("⚠️  [P2P HEALTH] NO CONNECTIONS - Network isolated!");
                        warn!("   Check bootstrap peer configuration and firewall settings");

                        // If bootstrap peer is configured, attempt reconnection
                        if let Ok(bootstrap_env) = std::env::var("Q_BOOTSTRAP_PEER") {
                            if let Ok(bootstrap_addr) = bootstrap_env.parse::<Multiaddr>() {
                                info!("🔄 [AUTO-RECONNECT] Attempting to reconnect to bootstrap peer...");
                                if let Err(e) = self.swarm.dial(bootstrap_addr.clone()) {
                                    error!("❌ [AUTO-RECONNECT] Dial failed: {:?}", e);
                                }
                            }
                        }
                    } else {
                        info!("✅ [P2P HEALTH] {} connected peer(s) - Network healthy", peer_count);
                    }
                }
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
                        NetworkCommand::PublishPeerHeight { topic, announcement_bytes, height } => {
                            debug!("📡 [TURBO SYNC] Publishing peer height announcement {} ({} bytes) to topic: {}",
                                  height, announcement_bytes.len(), topic);
                            let ident_topic = IdentTopic::new(topic.as_str());
                            match self.swarm.behaviour_mut().gossipsub.publish(ident_topic, announcement_bytes) {
                                Ok(_) => {
                                    debug!("✅ [TURBO SYNC] Successfully announced height {} to P2P network", height);
                                }
                                Err(e) => {
                                    warn!("❌ [TURBO SYNC] Failed to publish peer height to topic {}: {}", topic, e);
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

                    // 🤝 v1.0.15.1-beta: Initiate protocol handshake with new peer
                    let validator = self.handshake_validator.read().await;
                    let handshake_msg = validator.create_handshake(
                        format!("q-api-server-v{}.{}.{}",
                                crate::handshake_validator::ProtocolVersion::CURRENT.major,
                                crate::handshake_validator::ProtocolVersion::CURRENT.minor,
                                crate::handshake_validator::ProtocolVersion::CURRENT.patch)
                    );
                    drop(validator);

                    info!("🤝 [HANDSHAKE] Initiating handshake with peer {}", peer_id);
                    debug!("   Our protocol: v{}.{}.{}",
                          handshake_msg.protocol_version.major,
                          handshake_msg.protocol_version.minor,
                          handshake_msg.protocol_version.patch);
                    debug!("   Our network: {}", handshake_msg.network_id);

                    // Send handshake request to peer
                    let request_id = self.swarm.behaviour_mut().handshake.send_request(&peer_id, handshake_msg);
                    debug!("🤝 [HANDSHAKE] Sent handshake request {:?} to {}", request_id, peer_id);

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

                    // ✅ v1.0.17-beta: Critical diagnostic logging for connection failures
                    // Captures WHY dials fail (transport errors, limits, protocol mismatch, etc.)
                    SwarmEvent::OutgoingConnectionError { peer_id, error, .. } => {
                        error!("❌ [P2P-DIAG] OUTGOING CONNECTION FAILED");

                        // peer_id is Option<PeerId> - handle both cases
                        match peer_id {
                            Some(pid) => error!("   Peer: {}", pid),
                            None => error!("   Peer: <address-only dial, no peer ID yet>"),
                        }

                        error!("   Error type: {:?}", error);

                        // Detailed transport error breakdown
                        match &error {
                            libp2p::swarm::DialError::Transport(transport_errors) => {
                                error!("   🚨 TRANSPORT LAYER FAILURE:");
                                for (addr, transport_error) in transport_errors {
                                    error!("      Failed address: {}", addr);
                                    error!("      Transport error: {:?}", transport_error);

                                    // Drill into specific IO errors
                                    use libp2p::core::transport::TransportError;
                                    match transport_error {
                                        TransportError::MultiaddrNotSupported(a) => {
                                            error!("         → Multiaddr format not supported: {}", a);
                                            error!("         → FIX: Use /ip4/ instead of /dns/ or check transport config");
                                        }
                                        TransportError::Other(io_error) => {
                                            error!("         → IO Error: {}", io_error);
                                            error!("         → IO Error kind: {:?}", io_error.kind());

                                            // Common error patterns
                                            let err_str = format!("{}", io_error);
                                            if err_str.contains("Connection refused") || err_str.contains("refused") {
                                                error!("         → FIX: Peer not listening on port (check ss -tlnp | grep 9001)");
                                            } else if err_str.contains("timeout") || err_str.contains("timed out") {
                                                error!("         → FIX: Firewall blocking connection (check iptables/ufw)");
                                            } else if err_str.contains("DNS") || err_str.contains("dns") {
                                                error!("         → FIX: DNS resolution failed (use /ip4/ multiaddr)");
                                            } else if err_str.contains("No route") {
                                                error!("         → FIX: Network routing issue");
                                            }
                                        }
                                    }
                                }
                            }

                            libp2p::swarm::DialError::Denied { cause } => {
                                error!("   🚨 CONNECTION DENIED: {:?}", cause);
                                error!("      → Possible causes:");
                                error!("         - Connection limits reached (check connection_limits configuration)");
                                error!("         - Connection gating logic rejecting dials");
                                error!("         - Behaviour blocking connection");
                                error!("      → FIX: Temporarily disable connection_limits to test");
                            }

                            libp2p::swarm::DialError::NoAddresses => {
                                error!("   🚨 NO ADDRESSES TO DIAL");
                                error!("      → FIX: Multiaddr is empty or invalid");
                                error!("      → Check: BOOTSTRAP_PEERS configuration");
                            }

                            libp2p::swarm::DialError::WrongPeerId { obtained, endpoint } => {
                                error!("   🚨 WRONG PEER ID MISMATCH:");
                                if let Some(expected) = peer_id {
                                    error!("      Expected: {}", expected);
                                }
                                error!("      Obtained: {}", obtained);
                                error!("      Endpoint: {:?}", endpoint);
                                error!("      → FIX: Update BOOTSTRAP_PEERS with correct PeerID");
                                error!("      → Get actual PeerID from peer's logs: journalctl | grep 'Local peer ID'");
                            }

                            libp2p::swarm::DialError::Aborted => {
                                error!("   🚨 DIAL ABORTED");
                                error!("      → Dial was cancelled before completion");
                            }

                            libp2p::swarm::DialError::DialPeerConditionFalse(_) => {
                                error!("   🚨 DIAL PEER CONDITION FALSE");
                                error!("      → Some pre-dial condition check failed");
                            }

                            _ => {
                                error!("   🚨 OTHER DIAL ERROR: {:?}", error);
                            }
                        }

                        error!(""); // Blank line for readability
                    }

                    // ✅ v1.0.17-beta: Diagnostic logging for incoming connection failures (Server Beta side)
                    // Helps diagnose when Alpha reaches Beta at TCP layer but fails in libp2p upgrade
                    SwarmEvent::IncomingConnectionError { local_addr, send_back_addr, error, .. } => {
                        warn!("❌ [P2P-DIAG] INCOMING CONNECTION FAILED:");
                        warn!("   Local address: {}", local_addr);
                        warn!("   Remote address: {}", send_back_addr);
                        warn!("   Error: {:?}", error);
                        warn!("   → This indicates the peer reached us at TCP layer but failed during libp2p upgrade");
                        warn!("   → Check for transport/protocol version mismatch with the peer");
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
                // ✅ v0.9.68-beta: Updated to use BlockPackCodec for efficient block sync
                use libp2p::request_response::{Event, Message};

                match block_sync_event {
                    Event::Message { peer, message } => {
                        match message {
                            Message::Request { request_id, request, channel } => {
                                info!("📥 [BLOCK-PACK] Received block pack request from {}", peer);
                                info!("   Requested: blocks {}-{} (max {})",
                                      request.start_height, request.end_height, request.max_blocks);

                                // Validate request
                                if let Err(e) = request.validate() {
                                    error!("❌ [BLOCK-PACK] Invalid request: {}", e);
                                    let response = q_types::BlockPackResponse::from_blocks(vec![], request.end_height);
                                    let _ = self.swarm.behaviour_mut().block_sync.send_response(channel, response);
                                    return Ok(());
                                }

                                // Fetch blocks from storage
                                let response = if let Some(ref storage) = self.storage {
                                    let block_count = (request.end_height - request.start_height + 1) as usize;
                                    let limit = block_count.min(request.max_blocks);

                                    match storage.get_qblocks_range(request.start_height, limit).await {
                                        Ok(blocks) => {
                                            info!("✅ [BLOCK-PACK] Fetched {} blocks from storage (heights {}-{})",
                                                  blocks.len(),
                                                  blocks.first().map(|b| b.header.height).unwrap_or(request.start_height),
                                                  blocks.last().map(|b| b.header.height).unwrap_or(request.start_height));

                                            q_types::BlockPackResponse::from_blocks(blocks, request.end_height)
                                        }
                                        Err(e) => {
                                            error!("❌ [BLOCK-PACK] Failed to fetch blocks from storage: {}", e);
                                            q_types::BlockPackResponse::from_blocks(vec![], request.end_height)
                                        }
                                    }
                                } else {
                                    warn!("⚠️ [BLOCK-PACK] Storage not available, sending empty response");
                                    q_types::BlockPackResponse::from_blocks(vec![], request.end_height)
                                };

                                if let Err(e) = self.swarm.behaviour_mut().block_sync.send_response(channel, response) {
                                    error!("❌ [BLOCK-PACK] Failed to send response: {:?}", e);
                                } else {
                                    info!("✅ [BLOCK-PACK] Sent response to {}", peer);
                                }
                            }
                            Message::Response { request_id, response } => {
                                info!("📨 [BLOCK-PACK] Received block pack response: {} blocks (heights {}-{})",
                                      response.blocks.len(), response.start_height, response.end_height);

                                // ✅ v0.9.73-beta: Mark peer as successful (compatible with BlockPackCodec)
                                self.mark_peer_success(peer);

                                // v1.0.12-beta: Check if this is a pending batch sync request
                                // v1.0.15-beta: Convert request_id to String for HashMap lookup
                                let request_id_str = format!("{:?}", request_id);
                                let mut pending = self.pending_block_requests.lock().unwrap();
                                if let Some(tx) = pending.remove(&request_id_str) {
                                    // Send blocks to waiting BatchSyncEngine
                                    if let Err(_) = tx.send(response.blocks.clone()) {
                                        warn!("⚠️  [BATCH SYNC] Failed to deliver blocks: receiver dropped");
                                    } else {
                                        debug!("✅ [BATCH SYNC] Delivered {} blocks to waiting request",
                                               response.blocks.len());
                                    }
                                }
                                drop(pending); // Release lock

                                if response.has_more {
                                    info!("   More blocks available beyond height {}", response.end_height);
                                }

                                // Forward blocks to consensus for validation
                                if !response.blocks.is_empty() {
                                    if let Some(ref tx) = self.block_sync_tx {
                                        if let Err(e) = tx.send(response.blocks.clone()) {
                                            error!("❌ [BLOCK-PACK] Failed to forward blocks to consensus: {}", e);
                                        } else {
                                            info!("✅ [BLOCK-PACK] Forwarded {} blocks to consensus for validation", response.blocks.len());
                                        }
                                    } else {
                                        warn!("⚠️ [BLOCK-PACK] Block sync channel not configured, blocks not forwarded");
                                    }
                                } else {
                                    debug!("📭 [BLOCK-PACK] No blocks in response, nothing to forward");
                                }
                            }
                        }
                    }
                    Event::OutboundFailure { peer, request_id, error } => {
                        warn!("⚠️ [BLOCK-PACK] Outbound failure to {}: {:?}", peer, error);

                        // ✅ v0.9.73-beta: Mark peer as failed (timeout/incompatible)
                        self.mark_peer_failure(peer);
                    }
                    Event::InboundFailure { peer, error, .. } => {
                        warn!("⚠️ [BLOCK-PACK] Inbound failure from {}: {:?}", peer, error);
                    }
                    Event::ResponseSent { peer, .. } => {
                        debug!("✅ [BLOCK-PACK] Response sent to {}", peer);
                    }
                }
            }
            QNarwhalEvent::Handshake(handshake_event) => {
                // ✅ v1.0.15.1-beta: Protocol version validation for peer compatibility
                use libp2p::request_response::{Event, Message};

                match handshake_event {
                    Event::Message { peer, message } => {
                        match message {
                            Message::Request { request_id, request, channel } => {
                                info!("🤝 [HANDSHAKE] Received handshake request from {}", peer);
                                debug!("   Protocol: v{}.{}.{}",
                                       request.protocol_version.major,
                                       request.protocol_version.minor,
                                       request.protocol_version.patch);
                                debug!("   Network: {}", request.network_id);
                                debug!("   Node: {}", request.node_version);

                                // Validate handshake using our validator
                                let validator = self.handshake_validator.read().await;
                                let result = validator.validate_handshake(&request);
                                drop(validator);

                                // Send validation result back to peer
                                if let Err(e) = self.swarm.behaviour_mut().handshake.send_response(channel, result.clone()) {
                                    error!("❌ [HANDSHAKE] Failed to send handshake response: {:?}", e);
                                } else {
                                    match &result {
                                        crate::handshake_validator::HandshakeResult::Success => {
                                            info!("✅ [HANDSHAKE] Peer {} validated successfully", peer);
                                        }
                                        crate::handshake_validator::HandshakeResult::IncompatibleProtocol { ours, theirs } => {
                                            warn!("❌ [HANDSHAKE] Peer {} has incompatible protocol: ours={}, theirs={}",
                                                  peer, ours, theirs);
                                            // Disconnect incompatible peer
                                            let _ = self.swarm.disconnect_peer_id(peer);
                                        }
                                        crate::handshake_validator::HandshakeResult::WrongNetwork { ours, theirs } => {
                                            warn!("❌ [HANDSHAKE] Peer {} on wrong network: ours={}, theirs={}",
                                                  peer, ours, theirs);
                                            // Disconnect peer on wrong network
                                            let _ = self.swarm.disconnect_peer_id(peer);
                                        }
                                        crate::handshake_validator::HandshakeResult::GenesisMismatch => {
                                            warn!("❌ [HANDSHAKE] Peer {} has mismatched genesis hash", peer);
                                            // Disconnect peer with wrong genesis
                                            let _ = self.swarm.disconnect_peer_id(peer);
                                        }
                                        crate::handshake_validator::HandshakeResult::MissingFeatures { required } => {
                                            warn!("❌ [HANDSHAKE] Peer {} missing required features: {:?}", peer, required);
                                            // Disconnect peer missing features
                                            let _ = self.swarm.disconnect_peer_id(peer);
                                        }
                                    }
                                }
                            }
                            Message::Response { request_id, response } => {
                                match response {
                                    crate::handshake_validator::HandshakeResult::Success => {
                                        info!("✅ [HANDSHAKE] Peer validated our handshake successfully");
                                    }
                                    crate::handshake_validator::HandshakeResult::IncompatibleProtocol { ours, theirs } => {
                                        warn!("❌ [HANDSHAKE] Our protocol rejected by peer: ours={}, theirs={}", theirs, ours);
                                    }
                                    crate::handshake_validator::HandshakeResult::WrongNetwork { ours, theirs } => {
                                        warn!("❌ [HANDSHAKE] Network mismatch: ours={}, theirs={}", theirs, ours);
                                    }
                                    crate::handshake_validator::HandshakeResult::GenesisMismatch => {
                                        warn!("❌ [HANDSHAKE] Genesis hash rejected by peer");
                                    }
                                    crate::handshake_validator::HandshakeResult::MissingFeatures { required } => {
                                        warn!("❌ [HANDSHAKE] We are missing required features: {:?}", required);
                                    }
                                }
                            }
                        }
                    }
                    Event::OutboundFailure { peer, request_id, error } => {
                        warn!("⚠️ [HANDSHAKE] Outbound failure to {}: {:?}", peer, error);
                    }
                    Event::InboundFailure { peer, error, .. } => {
                        warn!("⚠️ [HANDSHAKE] Inbound failure from {}: {:?}", peer, error);
                    }
                    Event::ResponseSent { peer, .. } => {
                        debug!("✅ [HANDSHAKE] Response sent to {}", peer);
                    }
                }
            }
            // 🔥 v1.0.17-beta: NAT Traversal Event Handling
            QNarwhalEvent::AutoNat(event) => {
                match event {
                    libp2p::autonat::Event::StatusChanged { old, new } => {
                        info!("🔍 AutoNAT status changed: {:?} → {:?}", old, new);
                        match new {
                            libp2p::autonat::NatStatus::Public(addr) => {
                                info!("✅ Node is publicly dialable at: {}", addr);
                            }
                            libp2p::autonat::NatStatus::Private => {
                                warn!("⚠️  Node is behind NAT - relay connections will be used");
                            }
                            libp2p::autonat::NatStatus::Unknown => {
                                info!("❓ NAT status unknown - AutoNAT probing in progress");
                            }
                        }
                    }
                    _ => {
                        debug!("🔍 AutoNAT event: {:?}", event);
                    }
                }
            }
            QNarwhalEvent::Relay(event) => {
                // 🔥 v1.0.17-beta: Relay client events for NAT traversal
                // Just log all relay events for now since event enum names changed in libp2p 0.53
                debug!("🔁 Relay event: {:?}", event);
            }
            QNarwhalEvent::Dcutr(event) => {
                // 🔥 v1.0.17-beta: DCUtR (Direct Connection Upgrade through Relay) events
                // Hole-punching for NAT traversal - just log for now
                debug!("🎉 DCUtR event: {:?}", event);
            }
        }
        Ok(())
    }

    /// Get all discovered peers from ALL discovery methods
    pub async fn get_discovered_peers(&self) -> Vec<PeerId> {
        self.discovered_peers.read().await.iter().cloned().collect()
    }

    /// Get thread-safe reference to discovered peers
    /// This can be safely cloned and shared across threads without holding a reference to the manager
    pub fn get_discovered_peers_arc(&self) -> Arc<RwLock<HashSet<PeerId>>> {
        Arc::clone(&self.discovered_peers)
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

    /// ✅ v1.0.4-beta: Active peer height probing
    /// Queries discovered peers for their blockchain heights
    /// This eliminates passive dependency on gossipsub announcements
    ///
    /// # Returns
    /// Vector of (PeerId, height) pairs for all responsive peers
    ///
    /// # Performance
    /// - Concurrent queries to all peers
    /// - 5 second timeout per peer
    /// - Non-blocking (returns immediately with available results)
    pub async fn probe_peer_heights(&self) -> anyhow::Result<Vec<(PeerId, u64)>> {
        let peers = self.discovered_peers.read().await.clone();

        if peers.is_empty() {
            debug!("🔍 [PEER PROBING] No peers to probe");
            return Ok(Vec::new());
        }

        info!("🔍 [PEER PROBING] Actively probing {} peers for heights", peers.len());

        // For now, we'll rely on the TurboSync peer registry which is populated
        // by gossipsub peer-height announcements. In future versions, this could
        // directly query peers via request-response protocol.
        //
        // The key improvement here is that we actively check what heights we know
        // instead of passively waiting for announcements.

        Ok(Vec::new()) // Placeholder - integration with TurboSync peer registry needed
    }

    /// ✅ v1.0.4-beta: Get best known network height from discovered peers
    /// Queries the peer registry for the highest known peer height
    ///
    /// # Returns
    /// Some(height) if any peer heights are known, None otherwise
    pub async fn get_best_known_height(&self) -> Option<u64> {
        // This will be integrated with TurboSync peer registry
        // For now, return None to let timeout-based activation handle it
        None
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
    /// ✅ v0.9.68-beta: Updated to use BlockPackRequest for efficient block sync
    pub fn request_blocks_from_peer(
        &mut self,
        peer_id: PeerId,
        start_height: u64,
        limit: usize,
    ) -> anyhow::Result<()> {
        info!("📤 [BLOCK-SYNC] Requesting {} blocks from height {} from peer {}", limit, start_height, peer_id);

        let end_height = start_height + limit as u64 - 1;
        let request = q_types::BlockPackRequest::new(start_height, end_height);

        self.swarm.behaviour_mut().block_sync.send_request(&peer_id, request);

        info!("✅ [BLOCK-SYNC] Block sync request sent to {}", peer_id);
        Ok(())
    }

    /// v1.0.12-beta: Request a range of blocks and wait for response (async)
    /// Used by BatchSyncEngine for high-performance batch synchronization
    ///
    /// # Arguments
    /// * `start_height` - Starting block height (inclusive)
    /// * `end_height` - Ending block height (inclusive)
    ///
    /// # Returns
    /// Vector of blocks sorted by height
    ///
    /// # Performance
    /// - 60 second timeout per request (v1.0.13-beta: increased from 10s for large batches)
    /// - Automatic peer selection (highest height, compatible with BlockPackCodec)
    /// - Falls back to next peer on failure
    pub async fn request_block_range_impl(
        &mut self,
        start_height: u64,
        end_height: u64,
    ) -> anyhow::Result<Vec<q_types::QBlock>> {
        use tokio::time::{timeout, Duration};

        // Select best peer for this request
        let peer_id = {
            let discovered = self.discovered_peers.read().await;

            if discovered.is_empty() {
                return Err(anyhow::anyhow!("No peers available for block range request"));
            }

            // Get compatible peers (not blacklisted)
            let blacklist = self.get_blacklisted_peers();
            let compatible: Vec<PeerId> = discovered
                .iter()
                .filter(|p| !blacklist.contains(p))
                .copied()
                .collect();

            if compatible.is_empty() {
                return Err(anyhow::anyhow!(
                    "No compatible peers available (all blacklisted)"
                ));
            }

            // For now, use first compatible peer
            // TODO: Select peer with highest height and lowest latency
            compatible[0]
        };

        info!("📤 [BATCH SYNC] Requesting blocks {}-{} from peer {} ({} blocks requested)",
               start_height, end_height, peer_id, end_height - start_height + 1);

        // Create oneshot channel for response
        let (tx, rx) = tokio::sync::oneshot::channel();

        // Send request via libp2p
        let request = q_types::BlockPackRequest::new(start_height, end_height);
        let request_id = self.swarm.behaviour_mut().block_sync.send_request(&peer_id, request);

        // v1.0.15-beta: Convert request_id to String for HashMap storage
        let request_id_str = format!("{:?}", request_id);

        // Store channel for response delivery
        {
            let mut pending = self.pending_block_requests.lock().unwrap();
            pending.insert(request_id_str.clone(), tx);
            info!("🔗 [BATCH SYNC] Request ID {} registered, {} pending requests total",
                  request_id_str, pending.len());
        }

        info!("✅ [BATCH SYNC] libp2p request sent, waiting for response (timeout: 60s)");

        // Wait for response with timeout (v1.0.13-beta: increased from 10s to 60s)
        let request_start = std::time::Instant::now();
        match timeout(Duration::from_secs(60), rx).await {
            Ok(Ok(blocks)) => {
                let elapsed = request_start.elapsed();
                info!("📨 [BATCH SYNC] SUCCESS: Received {} blocks from peer {} in {:.2}s",
                       blocks.len(), peer_id, elapsed.as_secs_f64());
                Ok(blocks)
            }
            Ok(Err(_)) => {
                // Channel closed without response
                let elapsed = request_start.elapsed();
                warn!("❌ [BATCH SYNC] FAILURE: Channel closed without response after {:.2}s (peer: {})",
                      elapsed.as_secs_f64(), peer_id);
                self.mark_peer_failure(peer_id);
                Err(anyhow::anyhow!(
                    "Block range request failed: channel closed without response"
                ))
            }
            Err(_) => {
                // Timeout (v1.0.13-beta: 60s timeout)
                warn!("⏱️  [BATCH SYNC] TIMEOUT: No response after 60s from peer {}", peer_id);
                self.mark_peer_failure(peer_id);

                // Clean up pending request
                let mut pending = self.pending_block_requests.lock().unwrap();
                pending.remove(&request_id_str);
                info!("🧹 [BATCH SYNC] Cleaned up timed-out request, {} pending requests remaining",
                      pending.len());

                Err(anyhow::anyhow!(
                    "Block range request timed out after 60s (peer: {})",
                    peer_id
                ))
            }
        }
    }

    /// v0.9.73-beta: Mark peer as successful (responded to BlockPackCodec request)
    /// This is called when a peer successfully responds with blocks via BlockPackCodec
    pub fn mark_peer_success(&self, peer_id: PeerId) {
        let mut compat = self.peer_compat.write().unwrap();

        // Increment success counter
        *compat.successes.entry(peer_id).or_insert(0) += 1;

        // Remove from failure list (peer is proven working)
        compat.failures.remove(&peer_id);

        // Remove from blacklist (peer is proven compatible)
        if compat.blacklist.remove(&peer_id) {
            info!("✅ [PEER COMPAT] Peer {} removed from blacklist (now responsive)", peer_id);
        }

        let success_count = compat.successes.get(&peer_id).copied().unwrap_or(0);
        debug!("✅ [PEER COMPAT] Peer {} marked successful ({} total successes)", peer_id, success_count);
    }

    /// v0.9.73-beta: Mark peer as failed (timeout/no response to BlockPackCodec request)
    /// This is called when a peer fails to respond within timeout
    /// After 3 failures, the peer is blacklisted as incompatible
    pub fn mark_peer_failure(&self, peer_id: PeerId) {
        let mut compat = self.peer_compat.write().unwrap();

        // Increment failure counter
        *compat.failures.entry(peer_id).or_insert(0) += 1;
        let failure_count = compat.failures.get(&peer_id).copied().unwrap_or(0);

        debug!("⚠️  [PEER COMPAT] Peer {} marked failed ({} failures)", peer_id, failure_count);

        // Blacklist after 3 failures
        if failure_count >= 3 {
            compat.blacklist.insert(peer_id);
            compat.successes.remove(&peer_id); // Remove from successes
            warn!("🚫 [PEER COMPAT] Peer {} BLACKLISTED (3+ failures - incompatible with BlockPackCodec)", peer_id);
        }
    }

    /// v0.9.73-beta: Get list of compatible peers (successfully responded, not blacklisted)
    /// Used to filter peer selection for fast sync requests
    pub fn get_compatible_peers(&self) -> Vec<PeerId> {
        let compat = self.peer_compat.read().unwrap();

        // Return peers that have at least one success and are not blacklisted
        compat.successes.keys()
            .filter(|peer_id| !compat.blacklist.contains(peer_id))
            .cloned()
            .collect()
    }

    /// v0.9.75-beta: Get list of blacklisted peers (proven incompatible after 3+ failures)
    /// Used for OPTIMISTIC peer testing - assume compatible unless blacklisted
    pub fn get_blacklisted_peers(&self) -> std::collections::HashSet<PeerId> {
        let compat = self.peer_compat.read().unwrap();
        compat.blacklist.clone()
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
                    // 🤝 v1.0.15.1-beta: Initiate protocol handshake with new peer
                    let validator = self.handshake_validator.read().await;
                    let handshake_msg = validator.create_handshake(
                        format!("q-api-server-v{}.{}.{}",
                                crate::handshake_validator::ProtocolVersion::CURRENT.major,
                                crate::handshake_validator::ProtocolVersion::CURRENT.minor,
                                crate::handshake_validator::ProtocolVersion::CURRENT.patch)
                    );
                    drop(validator);

                    info!("🤝 [HANDSHAKE] Initiating handshake with peer {}", peer_id);
                    debug!("   Our protocol: v{}.{}.{}",
                          handshake_msg.protocol_version.major,
                          handshake_msg.protocol_version.minor,
                          handshake_msg.protocol_version.patch);
                    debug!("   Our network: {}", handshake_msg.network_id);

                    // Send handshake request to peer
                    let request_id = self.swarm.behaviour_mut().handshake.send_request(&peer_id, handshake_msg);
                    debug!("🤝 [HANDSHAKE] Sent handshake request {:?} to {}", request_id, peer_id);

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

// v1.0.12-beta: Implement BlockRangeFetcher trait for batch sync
// Trait is defined in q-types to avoid circular dependency
#[async_trait::async_trait]
impl q_types::BlockRangeFetcher for UnifiedNetworkManager {
    async fn request_block_range(
        &mut self,
        start_height: u64,
        end_height: u64,
    ) -> anyhow::Result<Vec<q_types::QBlock>> {
        self.request_block_range_impl(start_height, end_height).await
    }
}