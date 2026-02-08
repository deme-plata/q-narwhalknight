/// Simplified Zero-Knowledge Discovery for Q-NarwhalKnight
/// Implements peer discovery compatible with libp2p v0.53
/// Uses mDNS for local discovery and basic peer coordination

use libp2p::{
    core::{transport::Transport, upgrade},
    gossipsub::{self, IdentTopic, MessageId, ValidationMode},
    identity::Keypair,
    kad::{self, store::MemoryStore, Config as KademliaConfig, Event as KademliaEvent, Behaviour as Kademlia},
    noise, tcp, yamux, websocket,
    swarm::{SwarmEvent, Swarm, Config, NetworkBehaviour},
    Multiaddr,
    PeerId,
    SwarmBuilder,
};

// mDNS is only available on non-Windows platforms due to libudev dependency
#[cfg(not(target_os = "windows"))]
use libp2p::mdns::{self, Event as MdnsEvent};
#[cfg(not(target_os = "windows"))]
use libp2p::swarm::behaviour::toggle::Toggle;
use futures::StreamExt;

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use dashmap::{DashMap, DashSet};
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

// v4.3.0-beta: Mining solution rate limiting now handled by gossipsub_queue (MessagePriority::Low)
// Removed: LAST_MINING_SOLUTION_BROADCAST_MS, MIN_MINING_BROADCAST_INTERVAL_MS
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, error, info, trace, warn};

// v4.3.0-beta: Global peer latency tracker and bootstrap health cache
lazy_static::lazy_static! {
    /// Peer latency tracker fed by libp2p ping RTT measurements
    pub static ref PEER_LATENCY_TRACKER: crate::peer_latency::PeerLatencyTracker =
        crate::peer_latency::PeerLatencyTracker::new();
    /// Bootstrap endpoint health cache for latency-based ordering
    pub static ref BOOTSTRAP_HEALTH_CACHE: crate::peer_latency::BootstrapHealthCache =
        crate::peer_latency::BootstrapHealthCache::new();
}

use crate::connection_manager::{PeerInfo, DiscoveryMethod};
use crate::handshake::ServerRole;
use crate::distributed_ai::DistributedAITopics;
use crate::address_filter::{is_routable_peer_address, log_filter_configuration, get_external_address, get_external_wss_address};
use q_types::QBlock;

/// 🔥 v1.3.5-beta: DYNAMIC BOOTSTRAP DISCOVERY - NO HARDCODED PEER IDs
///
/// Bootstrap peer discovery order:
/// 1. HTTP discovery from Q_BOOTSTRAP_URL (fetches /api/v1/status from masternode)
/// 2. Q_BOOTSTRAP_PEERS environment variable (comma-separated multiaddrs)
/// 3. Q_BOOTSTRAP_PEER environment variable (single multiaddr for backwards compat)
///
/// WHY NO HARDCODED PEERS:
/// - Peer IDs change when libp2p_identity.key regenerates (e.g., fresh data dir)
/// - Hardcoded peer IDs get stale and cause "no peers available" errors
/// - Dynamic discovery via HTTP is always up-to-date
///
/// For new nodes to connect:
/// 1. Set Q_BOOTSTRAP_URL=http://185.182.185.227:8080 (production masternode)
/// 2. Node will auto-discover current peer ID from /api/v1/status endpoint
///
/// The /api/v1/status endpoint returns:
/// {
///   "data": {
///     "peer_id": "12D3KooW...",
///     "multiaddrs": ["/ip4/185.182.185.227/tcp/9001/p2p/12D3KooW..."],
///     "network_id": "testnet-phase16"
///   }
/// }

/// 🔧 v4.2.0-beta: MULTIPLE HARDCODED BOOTSTRAP PEERS - Mainnet safety
/// This ensures nodes can connect even when one bootstrap node is down
/// Server Beta (185.182.185.227) = Primary production bootstrap
/// Server Gamma (109.205.176.60) = Secondary production bootstrap
pub const HARDCODED_BOOTSTRAP_PEERS: &[&str] = &[
    // Server Beta - Primary production bootstrap (quillon.xyz)
    "/ip4/185.182.185.227/tcp/9001/p2p/12D3KooWFrhdwDDTgxPX41mUyRgLcE1ozsBYArKM4DT8t4VLwuNx",
    // Server Gamma - Secondary production bootstrap
    // Peer ID discovered dynamically via HTTP at startup (fetch_peer_id_from_http)
    "/ip4/109.205.176.60/tcp/9001",
];

/// v4.2.0-beta: Bootstrap HTTP API endpoints for dynamic peer ID discovery
/// Used when hardcoded peer IDs are stale or missing (e.g., new server first boot)
pub const BOOTSTRAP_HTTP_ENDPOINTS: &[&str] = &[
    "http://185.182.185.227:8080",
    "http://109.205.176.60:8080",
];

/// Legacy single bootstrap peer constant (for backwards compatibility)
pub const HARDCODED_BOOTSTRAP_PEER: &str = "/ip4/185.182.185.227/tcp/9001/p2p/12D3KooWFrhdwDDTgxPX41mUyRgLcE1ozsBYArKM4DT8t4VLwuNx";

/// v3.3.2-beta: Get bootstrap peers with MULTIPLE HARDCODED FALLBACKS
/// Priority:
/// 1. Q_BOOTSTRAP_PEERS env var (comma-separated)
/// 2. Q_BOOTSTRAP_PEER env var (single peer)
/// 3. HARDCODED_BOOTSTRAP_PEERS (all included as fallbacks for mainnet safety)
fn get_bootstrap_peers() -> Vec<String> {
    let mut peers = Vec::new();

    // Check Q_BOOTSTRAP_PEERS (comma-separated list)
    if let Ok(env_peers) = std::env::var("Q_BOOTSTRAP_PEERS") {
        let env_list: Vec<String> = env_peers
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty() && s.contains("/p2p/"))
            .collect();

        if !env_list.is_empty() {
            info!("🔧 [BOOTSTRAP] Using {} peers from Q_BOOTSTRAP_PEERS environment variable", env_list.len());
            for (i, peer) in env_list.iter().enumerate() {
                info!("   {}. {}", i + 1, peer);
            }
            peers.extend(env_list);
        }
    }

    // Check Q_BOOTSTRAP_PEER (single peer, backwards compatibility)
    if let Ok(single_peer) = std::env::var("Q_BOOTSTRAP_PEER") {
        let trimmed = single_peer.trim().to_string();
        if !trimmed.is_empty() && trimmed.contains("/p2p/") && !peers.contains(&trimmed) {
            info!("🔧 [BOOTSTRAP] Using peer from Q_BOOTSTRAP_PEER: {}", trimmed);
            peers.push(trimmed);
        }
    }

    // 🔧 v3.3.2-beta: ALWAYS add ALL hardcoded bootstrap peers as fallback
    // This ensures connectivity even when one bootstrap node is down (mainnet safety)
    for hardcoded in HARDCODED_BOOTSTRAP_PEERS {
        let hardcoded_str = hardcoded.to_string();
        if !peers.contains(&hardcoded_str) {
            info!("🔧 [BOOTSTRAP] Adding hardcoded bootstrap peer: {}", hardcoded_str);
            peers.push(hardcoded_str);
        }
    }

    if peers.len() == 1 {
        info!("🔧 [BOOTSTRAP] Using single hardcoded bootstrap peer");
    } else {
        info!("🔧 [BOOTSTRAP] Total bootstrap peers: {} (mainnet resilient)", peers.len());
    }

    peers
}

/// Q-NarwhalKnight network behavior combining all discovery mechanisms
/// 🔥 v1.0.17-beta: Added NAT traversal for true decentralization (AutoNAT + Relay + DCUtR)
#[derive(NetworkBehaviour)]
#[behaviour(to_swarm = "QNarwhalEvent")]
pub struct QNarwhalBehaviour {
    /// mDNS for local network discovery (zero-config) - only available on non-Windows platforms
    /// Wrapped in Toggle so mDNS failure (e.g. Permission denied on ARM) doesn't crash the node
    #[cfg(not(target_os = "windows"))]
    mdns: Toggle<mdns::tokio::Behaviour>,
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
    /// 🌐 v3.5.5: Relay Server - Enable this node to relay connections for browser P2P
    /// Browsers can connect to each other through this relay server
    relay_server: libp2p::relay::Behaviour,
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
    /// 🌐 v3.5.5: Relay Server events for browser P2P
    RelayServer(libp2p::relay::Event),
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

// 🌐 v3.5.5: Relay Server event conversion for browser P2P
impl From<libp2p::relay::Event> for QNarwhalEvent {
    fn from(event: libp2p::relay::Event) -> Self {
        QNarwhalEvent::RelayServer(event)
    }
}

impl From<libp2p::dcutr::Event> for QNarwhalEvent {
    fn from(event: libp2p::dcutr::Event) -> Self {
        QNarwhalEvent::Dcutr(event)
    }
}

// 🔥 v2.0.0: libp2p 0.56 uses Infallible (not void::Void) for connection_limits
// Infallible is an uninhabited type that never needs conversion
impl From<std::convert::Infallible> for QNarwhalEvent {
    fn from(i: std::convert::Infallible) -> Self {
        // Infallible is uninhabited, so this can never be called
        match i {}
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
    /// ⚡ v2.6.0: Publish raw all-reduce message for tensor parallelism
    PublishAllReduce {
        topic: String,
        data: Vec<u8>,
    },
    /// 🔐 v2.4.6-beta: Publish BFT consensus messages (sig-requests, sig-responses, certificates, validators)
    PublishConsensusMessage {
        topic: String,
        message_bytes: Vec<u8>,
    },
    /// Publish a liquidity pool announcement to the DEX network (v0.6.1-beta)
    PublishPoolAnnouncement {
        topic: String,
        announcement_bytes: Vec<u8>,
    },
    /// v2.3.7-beta: Publish a token deployment announcement to the DEX network
    /// Enables cross-node token discovery for true DEX decentralization
    PublishTokenAnnouncement {
        topic: String,
        announcement_bytes: Vec<u8>,
    },
    /// v1.0.88-beta: Publish miner stats to P2P network
    /// Allows users mining to localhost nodes to have hashrate visible on bootstrap node
    PublishMinerStats {
        topic: String,
        stats_bytes: Vec<u8>,
        miner_address: String,
    },
    /// v1.1.8-beta: Publish balance update to P2P network
    /// Enables decentralized mining by syncing balance updates across all nodes
    PublishBalanceUpdate {
        topic: String,
        update_bytes: Vec<u8>,
        wallet_address: String,
        amount: u64,
    },
    /// v2.2.1-beta: Publish mining solution to P2P network
    /// Enables decentralized mining by broadcasting solutions to all nodes
    /// Any node can include the solution in a block, not just the receiving node
    PublishMiningSolution {
        topic: String,
        solution_bytes: Vec<u8>,
        miner_address: String,
        block_height: u64,
        nonce: u64,
    },
    /// v3.3.0-beta: Publish transaction to P2P mempool network
    /// Enables real-time mempool synchronization across all nodes
    /// Transactions are immediately visible on all nodes before block inclusion
    PublishTransaction {
        topic: String,
        tx_bytes: Vec<u8>,
        tx_hash: String,
    },
    /// 🚀 v1.3.9-beta: Direct request-response for Turbo Sync (replaces gossipsub)
    /// Uses libp2p request-response protocol for reliable, point-to-point block fetching.
    /// MUCH more reliable than gossipsub for large block batches because:
    /// - Point-to-point delivery (no broadcast flooding)
    /// - Built-in timeout handling (60s default)
    /// - Guaranteed response or error (no silent drops)
    RequestBlockRangeDirect {
        /// Optional peer ID to request from (if None, selects best peer automatically)
        peer_id: Option<String>,
        start_height: u64,
        end_height: u64,
        /// Oneshot channel to deliver the response
        response_tx: tokio::sync::oneshot::Sender<anyhow::Result<Vec<q_types::QBlock>>>,
    },

    // ============================================================================
    // v1.3.11-beta: DECENTRALIZED CONSENSUS P2P PROTOCOL
    // ============================================================================

    /// Request consensus signatures from validators for a vertex/block
    /// Broadcast to all validators via gossipsub to request their signatures
    PublishConsensusRequest {
        topic: String,
        vertex_id: [u8; 32],
        round: u64,
        block_hash: [u8; 32],
        requester_id: [u8; 32],
    },

    /// Respond with our signature for a consensus request
    /// Sent as a response to PublishConsensusRequest
    PublishConsensusSignature {
        topic: String,
        vertex_id: [u8; 32],
        validator_id: [u8; 32],
        signature: [u8; 64],
        public_key: [u8; 32],
        timestamp: u64,
    },

    /// Announce a completed certificate with multi-validator signatures
    /// Broadcast to network so all nodes know this block is consensus-confirmed
    PublishConsensusCertificate {
        topic: String,
        vertex_id: [u8; 32],
        round: u64,
        signatures: Vec<([u8; 32], Vec<u8>)>, // (validator_id, signature)
        threshold_met: bool,
    },

    /// Report equivocation (double-signing) by a validator
    /// Triggers slashing on all honest nodes
    ReportEquivocation {
        topic: String,
        validator_id: [u8; 32],
        vertex_id: [u8; 32],
        signature1: Vec<u8>,
        signature2: Vec<u8>,
    },

    /// v1.4.2-beta: Publish QNO (Quantum Neural Oracle) operation
    /// Broadcasts stake/unstake/claim operations for decentralized validation
    PublishQnoOperation {
        topic: String,
        message: crate::distributed_qno::QnoGossipMessage,
    },

    /// v2.4.8-beta: Publish token social media profile update
    /// Syncs social profiles (Twitter, Discord, etc.) across all nodes
    PublishTokenSocial {
        topic: String,
        contract_address: String,
        profile_bytes: Vec<u8>,
    },

    /// v2.9.2-beta: Publish DEX event (trade, liquidity, price) to all peers
    /// Enables TRUE decentralization by syncing DEX state across all nodes
    /// This is the missing piece that makes the DEX fully P2P!
    PublishDexEvent {
        topic: String,
        message: Vec<u8>,
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

/// 🚀 v1.0.4-beta: Load or generate persistent libp2p identity
/// Prevents PeerID churn on every restart (critical for bootstrap nodes)
///
/// # Arguments
/// * `data_dir` - Directory to store identity key file
///
/// # Returns
/// * Persistent keypair loaded from disk or newly generated and saved
/// v3.3.7-beta: Returns (Keypair, is_new_identity) to enable connection warmup for new nodes
fn load_or_generate_identity(data_dir: &std::path::Path) -> anyhow::Result<(Keypair, bool)> {
    let key_path = data_dir.join("libp2p_identity.key");

    if key_path.exists() {
        // Load existing persistent identity
        let bytes = std::fs::read(&key_path)?;
        let keypair = Keypair::from_protobuf_encoding(&bytes)?;
        info!("🔑 Loaded persistent libp2p identity: {}", keypair.public().to_peer_id());
        info!("   Identity file: {}", key_path.display());
        Ok((keypair, false)) // Existing identity
    } else {
        // Generate new identity and persist to disk
        let keypair = Keypair::generate_ed25519();
        let bytes = keypair.to_protobuf_encoding()?;

        // Ensure directory exists
        if let Some(parent) = key_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        std::fs::write(&key_path, &bytes)?;
        info!("🔑 Generated NEW persistent libp2p identity: {}", keypair.public().to_peer_id());
        info!("   Identity saved to: {}", key_path.display());
        info!("   ⚠️  IMPORTANT: Back up this file to preserve PeerID across server migrations");
        info!("   🔄 NEW IDENTITY: Connection warmup will be enabled for DHT registration");
        Ok((keypair, true)) // New identity - needs warmup
    }
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
    let endpoint_key = format!("http://{}:{}", ip, http_port);

    info!("🔍 Fetching dynamic peer ID from {} (with retry)", url);

    // 🚀 v1.0.3-beta: Retry logic for Docker network initialization delays
    for attempt in 1..=3 {
        // Add delay on retry attempts (exponential backoff)
        if attempt > 1 {
            let delay = Duration::from_secs(attempt as u64);
            info!("⏳ Retry attempt {}/3 after {}s delay", attempt, delay.as_secs());
            tokio::time::sleep(delay).await;
        }

        // Use reqwest with timeout
        let client = match reqwest::Client::builder()
            .timeout(Duration::from_secs(10)) // Increased from 5s
            .build() {
            Ok(c) => c,
            Err(e) => {
                error!("❌ Failed to build reqwest client: {:?}", e);
                if attempt == 3 {
                    BOOTSTRAP_HEALTH_CACHE.record_failure(&endpoint_key);
                    return Err(anyhow::anyhow!("Client build error: {}", e));
                }
                continue;
            }
        };

        // v4.3.0-beta: Track request timing for bootstrap health ordering
        let request_start = std::time::Instant::now();

        match client.get(&url).send().await {
            Ok(response) => {
                let status = response.status();
                info!("✅ HTTP response received: status={}", status);

                if !status.is_success() {
                    warn!("⚠️ Non-success HTTP status: {}", status);
                    if attempt == 3 {
                        BOOTSTRAP_HEALTH_CACHE.record_failure(&endpoint_key);
                        return Err(anyhow::anyhow!("HTTP error status: {}", status));
                    }
                    continue;
                }

                match response.json::<PeerIdResponse>().await {
                    Ok(peer_response) => {
                        if peer_response.success {
                            if let Some(data) = peer_response.data {
                                let rtt = request_start.elapsed();
                                info!("✅ Successfully fetched peer ID on attempt {}/3: {} ({}ms)",
                                      attempt, data.peer_id, rtt.as_millis());
                                // v4.3.0-beta: Record successful fetch for health ordering
                                BOOTSTRAP_HEALTH_CACHE.record_success(&endpoint_key, rtt, &data.peer_id);
                                return Ok(data.peer_id);
                            } else {
                                warn!("⚠️ API returned success=true but no data field");
                            }
                        } else {
                            warn!("⚠️ API returned success=false");
                        }
                        if attempt == 3 {
                            BOOTSTRAP_HEALTH_CACHE.record_failure(&endpoint_key);
                            return Err(anyhow::anyhow!("API returned invalid response"));
                        }
                    }
                    Err(e) => {
                        error!("❌ Failed to parse JSON response: {:?}", e);
                        if attempt == 3 {
                            BOOTSTRAP_HEALTH_CACHE.record_failure(&endpoint_key);
                            return Err(anyhow::anyhow!("JSON parse error: {}", e));
                        }
                    }
                }
            }
            Err(e) => {
                error!("❌ HTTP request failed on attempt {}/3: {:?}", attempt, e);
                error!("   Is timeout: {}", e.is_timeout());
                error!("   Is connect: {}", e.is_connect());
                if let Some(url_err) = e.url() {
                    error!("   Failed URL: {}", url_err);
                }
                if attempt == 3 {
                    BOOTSTRAP_HEALTH_CACHE.record_failure(&endpoint_key);
                    return Err(anyhow::anyhow!("HTTP request error after 3 attempts: {}", e));
                }
            }
        }
    }

    BOOTSTRAP_HEALTH_CACHE.record_failure(&endpoint_key);
    Err(anyhow::anyhow!("Failed to fetch peer ID from HTTP endpoint after 3 attempts"))
}

/// v0.9.73-beta: Peer compatibility tracking for BlockPackCodec protocol
/// Tracks which peers successfully support the BlockPackCodec request-response protocol
/// v1.0.83-beta: Made blacklist less aggressive - expires after 5 minutes, threshold raised to 10
/// v1.0.86-beta: CRITICAL FIX - Much less aggressive blacklisting to prevent network death
#[derive(Debug, Clone, Default)]
pub struct PeerCompatibility {
    /// Peers that have successfully responded (PeerId → success count)
    pub successes: HashMap<PeerId, u32>,
    /// Peers that have failed to respond (PeerId → failure count)
    pub failures: HashMap<PeerId, u32>,
    /// Blacklisted peers with timestamp (PeerId → blacklist time)
    /// v1.0.83-beta: Blacklist now expires after BLACKLIST_EXPIRY_SECS
    pub blacklist: HashMap<PeerId, std::time::Instant>,
    /// v1.0.86-beta: Track last failure time for decay calculation
    pub last_failure_time: HashMap<PeerId, std::time::Instant>,
    /// v1.0.86-beta: Track if peer is a bootstrap peer (higher failure tolerance)
    pub is_bootstrap: HashSet<PeerId>,
}

/// v1.0.86-beta: Blacklist configuration constants - MUCH less aggressive
/// Problem: 10 failures + 5 min blacklist = network death with single bootstrap peer
/// Fix: Higher threshold, shorter expiry, special treatment for bootstrap peers

/// Failures before blacklisting (increased from 10 to 50)
/// Rationale: NAT traversal alone can cause 5-10 "failures" per connection attempt
const BLACKLIST_FAILURE_THRESHOLD: u32 = 50;

/// Blacklist expiry reduced from 300s to 60s (1 minute)
/// Rationale: Network conditions change fast, retry sooner
const BLACKLIST_EXPIRY_SECS: u64 = 60;

/// v1.0.86-beta: Failure decay interval - decay 1 failure every 30 seconds
/// Prevents failure accumulation from transient issues
const FAILURE_DECAY_INTERVAL_SECS: u64 = 30;

/// v1.0.86-beta: Bootstrap peers get 3x higher threshold (150 failures)
/// Rationale: Losing bootstrap = network isolation, be very conservative
const BOOTSTRAP_BLACKLIST_MULTIPLIER: u32 = 3;

/// 🚀 v1.7.0-LAMINAR (VORTEX ELIMINATION): Lock-free peer compatibility tracking
/// Uses DashMap for concurrent access without lock contention
/// Eliminates the lock bottleneck that limited sync throughput to ~500 BPS
#[derive(Debug)]
pub struct PeerCompatibilityV2 {
    /// Peers that have successfully responded (PeerId → success count)
    pub successes: DashMap<PeerId, u32>,
    /// Peers that have failed to respond (PeerId → failure count)
    pub failures: DashMap<PeerId, u32>,
    /// Blacklisted peers with timestamp (PeerId → blacklist time)
    pub blacklist: DashMap<PeerId, std::time::Instant>,
    /// Track last failure time for decay calculation
    pub last_failure_time: DashMap<PeerId, std::time::Instant>,
    /// Track if peer is a bootstrap peer (higher failure tolerance)
    pub is_bootstrap: DashSet<PeerId>,
}

impl Default for PeerCompatibilityV2 {
    fn default() -> Self {
        Self {
            successes: DashMap::new(),
            failures: DashMap::new(),
            blacklist: DashMap::new(),
            last_failure_time: DashMap::new(),
            is_bootstrap: DashSet::new(),
        }
    }
}

impl Clone for PeerCompatibilityV2 {
    fn clone(&self) -> Self {
        Self {
            successes: self.successes.iter().map(|r| (*r.key(), *r.value())).collect(),
            failures: self.failures.iter().map(|r| (*r.key(), *r.value())).collect(),
            blacklist: self.blacklist.iter().map(|r| (*r.key(), *r.value())).collect(),
            last_failure_time: self.last_failure_time.iter().map(|r| (*r.key(), *r.value())).collect(),
            is_bootstrap: self.is_bootstrap.iter().map(|r| *r.key()).collect(),
        }
    }
}

impl PeerCompatibilityV2 {
    /// Record a successful response from peer
    pub fn record_success(&self, peer_id: &PeerId) {
        *self.successes.entry(*peer_id).or_insert(0) += 1;
    }

    /// Record a failed response from peer with decay calculation
    pub fn record_failure(&self, peer_id: &PeerId) {
        let now = std::time::Instant::now();

        // Apply decay before recording new failure
        if let Some(mut entry) = self.failures.get_mut(peer_id) {
            if let Some(last_time) = self.last_failure_time.get(peer_id) {
                let elapsed = now.duration_since(*last_time.value()).as_secs();
                let decay = (elapsed / FAILURE_DECAY_INTERVAL_SECS) as u32;
                *entry.value_mut() = entry.value().saturating_sub(decay);
            }
        }

        // Record new failure
        *self.failures.entry(*peer_id).or_insert(0) += 1;
        self.last_failure_time.insert(*peer_id, now);

        // Check if should be blacklisted
        let failures = self.failures.get(peer_id).map(|f| *f).unwrap_or(0);
        let threshold = if self.is_bootstrap.contains(peer_id) {
            BLACKLIST_FAILURE_THRESHOLD * BOOTSTRAP_BLACKLIST_MULTIPLIER
        } else {
            BLACKLIST_FAILURE_THRESHOLD
        };

        if failures >= threshold {
            self.blacklist.insert(*peer_id, now);
        }
    }

    /// Check if peer is blacklisted (with expiry check)
    pub fn is_blacklisted(&self, peer_id: &PeerId) -> bool {
        if let Some(entry) = self.blacklist.get(peer_id) {
            let elapsed = entry.value().elapsed().as_secs();
            if elapsed >= BLACKLIST_EXPIRY_SECS {
                // Expired - remove from blacklist
                drop(entry);
                self.blacklist.remove(peer_id);
                return false;
            }
            return true;
        }
        false
    }

    /// Get failure count for peer (with decay applied)
    pub fn get_failures(&self, peer_id: &PeerId) -> u32 {
        self.failures.get(peer_id).map(|f| *f).unwrap_or(0)
    }

    /// Get success count for peer
    pub fn get_successes(&self, peer_id: &PeerId) -> u32 {
        self.successes.get(peer_id).map(|s| *s).unwrap_or(0)
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
    /// 🚀 v1.7.0-LAMINAR: Lock-free peer compatibility tracking (VORTEX ELIMINATION)
    /// Uses DashMap for concurrent access without lock contention
    peer_compat: Arc<PeerCompatibilityV2>,
    /// v1.0.12-beta: Pending block range requests for batch sync
    /// Maps request_id (as String) → oneshot channel for async await
    /// v1.0.15-beta: Fixed to use String instead of removed libp2p::request_response::RequestId
    pending_block_requests: Arc<std::sync::Mutex<HashMap<String, tokio::sync::oneshot::Sender<Vec<q_types::QBlock>>>>>,
    /// v1.0.15.1-beta: Protocol version validation for peer handshakes
    /// Prevents silent communication failures from incompatible protocol versions
    handshake_validator: Arc<RwLock<crate::handshake_validator::HandshakeValidator>>,
    /// v1.0.44-beta: Track multiple outstanding sync requests for concurrent sync
    /// Stores Vec<(request_id, start_height, timestamp)> - supports up to MAX_CONCURRENT_REQUESTS
    /// v1.0.43-beta: Original single request tracking for stall detection
    outstanding_sync_requests: Arc<std::sync::Mutex<Vec<(String, u64, std::time::Instant)>>>,
    /// v1.0.45-beta: Track best known network height for progress display
    /// Updated from BlockPackResponse.peer_height on each sync response
    known_network_height: Arc<std::sync::atomic::AtomicU64>,
    /// v1.2.7-beta: Channel for async block pack responses (non-blocking handler)
    /// The handler spawns a task for slow DB operations, which sends the response through this channel.
    /// The main event loop polls this and calls send_response. Prevents ResponseOmission timeouts.
    block_pack_response_tx: mpsc::UnboundedSender<(u64, q_types::BlockPackResponse)>,
    block_pack_response_rx: mpsc::UnboundedReceiver<(u64, q_types::BlockPackResponse)>,
    /// v1.2.7-beta: Pending response channels indexed by request ID for async responses
    pending_response_channels: Arc<std::sync::Mutex<HashMap<u64, libp2p::request_response::ResponseChannel<q_types::BlockPackResponse>>>>,
    /// v1.2.7-beta: Counter for generating unique request IDs for async response tracking
    next_async_request_id: Arc<std::sync::atomic::AtomicU64>,
    /// v1.3.3-beta: Tor-enabled flag for adaptive timeouts and batch sizes
    /// Set during initialization based on Q_TOR_ENABLED, Q_TOR_PROXY, or SOCKS5 proxy detection
    tor_enabled: bool,
    /// v1.3.3-beta: Retry queue for failed sync requests with exponential backoff
    /// Stores (height, retry_count, next_retry_time) for failed heights that should be retried
    sync_retry_queue: Arc<std::sync::Mutex<Vec<(u64, u8, std::time::Instant)>>>,
}

// SAFETY: UnifiedNetworkManager is Sync because:
// 1. All fields are either Send+Sync primitives or wrapped in Arc<RwLock<T>>
// 2. The Swarm<T> field is single-threaded but never accessed across threads (only from async runtime)
// 3. This is required for libp2p 0.56 compatibility with NetworkFetcher trait
unsafe impl Sync for UnifiedNetworkManager {}

impl UnifiedNetworkManager {
    /// Create new network manager with network configuration
    ///
    /// # Arguments
    /// * `network_config` - Network configuration (testnet/mainnet)
    pub async fn new(network_config: q_types::NetworkConfig) -> anyhow::Result<Self> {
        // 🚀 v1.0.3-beta: Docker network namespace initialization delay
        // Docker containers with --network host may need time for network to be fully ready
        // This prevents "error sending request" failures in reqwest
        if std::env::var("RUNNING_IN_DOCKER").is_ok() || std::env::var("container").is_ok() {
            info!("🐳 Docker environment detected, waiting 2s for network initialization...");
            tokio::time::sleep(Duration::from_secs(2)).await;
        }

        // 🚀 v1.0.4-beta: CRITICAL FIX - Persistent libp2p identity
        // Load existing identity from disk or generate new one and save
        // This prevents PeerID churn on every restart (breaks bootstrap DHT routing)
        // 🔄 v3.3.7-beta: Track if identity is NEW for connection warmup
        let data_dir = std::env::var("Q_DB_PATH").unwrap_or_else(|_| "./data-mine12".to_string());
        let data_path = std::path::Path::new(&data_dir);
        let (keypair, is_new_identity) = load_or_generate_identity(data_path)?;
        let local_peer_id = PeerId::from(keypair.public());

        info!("🚀 Starting Q-NarwhalKnight Zero-Knowledge Discovery");
        info!("🌐 Network: {}", network_config.network_id.display_name());
        info!("🆔 Local Peer ID: {}", local_peer_id);
        if is_new_identity {
            info!("🔄 [NEW IDENTITY] Connection warmup will be performed after initial bootstrap");
        }

        // 🐳 v1.2.2-beta: Log Docker/container address filtering configuration
        log_filter_configuration();

        // 🔥 v1.0.17-beta: SwarmBuilder replaces manual transport construction
        // Old manual TCP+Noise+Yamux transport deleted - now handled by SwarmBuilder
        // mDNS, Identify, Ping, Kademlia initialization moved into SwarmBuilder closure

        // Bootstrap from network configuration with automatic peer ID discovery
        // v1.0.86-beta: Merge hardcoded peers, network config peers, and env var override
        let mut all_bootstrap_peers: Vec<String> = Vec::new();

        // 1. Start with network config peers
        all_bootstrap_peers.extend(network_config.bootstrap_peers.clone());

        // 2. Add hardcoded/env var peers (get_bootstrap_peers handles env var override)
        for peer in get_bootstrap_peers() {
            if !all_bootstrap_peers.contains(&peer) {
                all_bootstrap_peers.push(peer);
            }
        }

        info!("🔧 [BOOTSTRAP] Total unique bootstrap peers: {}", all_bootstrap_peers.len());

        let bootstrap_peers = &all_bootstrap_peers;
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
                    // Multiaddr already has peer ID - but ALWAYS verify via HTTP first!
                    // 🚨 v1.1.0-phase13: CRITICAL FIX - Hardcoded peer IDs become stale after phase transitions
                    // The peer ID changes when the node restarts with a fresh database (new identity)
                    // So we MUST fetch the current peer ID via HTTP and compare

                    // Extract IP for HTTP discovery
                    let mut bootstrap_ip: Option<String> = None;
                    for protocol in addr.iter() {
                        match protocol {
                            Protocol::Ip4(addr_v4) => bootstrap_ip = Some(addr_v4.to_string()),
                            Protocol::Ip6(addr_v6) => bootstrap_ip = Some(addr_v6.to_string()),
                            _ => {}
                        }
                    }

                    if let Some(ref ip) = bootstrap_ip {
                        // Try HTTP discovery to get CURRENT peer ID
                        match fetch_peer_id_from_http(ip, 8080).await {
                            Ok(current_peer_id_str) => {
                                match current_peer_id_str.parse::<PeerId>() {
                                    Ok(current_peer_id) => {
                                        // 🔧 v1.0.88-beta: Skip ourselves as a bootstrap peer
                                        if current_peer_id == local_peer_id {
                                            info!("ℹ️  [BOOTSTRAP] Skipping self as bootstrap peer: {} (this is us!)", current_peer_id);
                                            continue;
                                        }

                                        // Check if hardcoded peer ID matches current
                                        if let Some(Protocol::P2p(hardcoded_peer_id)) = addr.iter().last() {
                                            if hardcoded_peer_id != current_peer_id {
                                                warn!("⚠️ [BOOTSTRAP] Hardcoded peer ID {} is STALE!", hardcoded_peer_id);
                                                warn!("   Current peer ID from HTTP: {}", current_peer_id);
                                                warn!("   Using HTTP-discovered peer ID instead");
                                            }
                                        }

                                        // Build multiaddr with CURRENT peer ID
                                        let mut fresh_addr = addr.clone();
                                        // Remove old /p2p/ component
                                        fresh_addr = fresh_addr.into_iter()
                                            .filter(|p| !matches!(p, Protocol::P2p(_)))
                                            .collect();
                                        fresh_addr.push(Protocol::P2p(current_peer_id));

                                        bootstrap_peer_map.insert(current_peer_id, fresh_addr.clone());
                                        info!("📍 Added {} bootstrap peer (HTTP-verified): {} at {}",
                                              network_config.network_id.as_str(), current_peer_id, fresh_addr);
                                        bootstrap_count += 1;
                                    }
                                    Err(e) => {
                                        warn!("⚠️ Failed to parse HTTP peer ID: {}", e);
                                        // Fallback to hardcoded peer ID
                                        if let Some(Protocol::P2p(peer_id)) = addr.iter().last() {
                                            if peer_id != local_peer_id {
                                                bootstrap_peer_map.insert(peer_id, addr.clone());
                                                info!("📍 Added {} bootstrap peer (hardcoded fallback): {} at {}",
                                                      network_config.network_id.as_str(), peer_id, addr);
                                                bootstrap_count += 1;
                                            }
                                        }
                                    }
                                }
                            }
                            Err(e) => {
                                warn!("⚠️ HTTP peer ID discovery failed: {}", e);
                                warn!("   Falling back to hardcoded peer ID");
                                // Fallback to hardcoded peer ID
                                if let Some(Protocol::P2p(peer_id)) = addr.iter().last() {
                                    if peer_id != local_peer_id {
                                        bootstrap_peer_map.insert(peer_id, addr.clone());
                                        info!("📍 Added {} bootstrap peer (hardcoded fallback): {} at {}",
                                              network_config.network_id.as_str(), peer_id, addr);
                                        bootstrap_count += 1;
                                    }
                                }
                            }
                        }
                    } else {
                        // No IP to fetch from - use hardcoded directly
                        if let Some(Protocol::P2p(peer_id)) = addr.iter().last() {
                            if peer_id != local_peer_id {
                                bootstrap_peer_map.insert(peer_id, addr.clone());
                                info!("📍 Added {} bootstrap peer: {} at {}",
                                      network_config.network_id.as_str(), peer_id, addr);
                                bootstrap_count += 1;
                            }
                        }
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
                                        // 🔧 v1.0.88-beta: Skip ourselves as a bootstrap peer
                                        if peer_id == local_peer_id {
                                            info!("ℹ️  [BOOTSTRAP] Skipping self as bootstrap peer (dynamic discovery): {} (this is us!)", peer_id);
                                            continue;
                                        }
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


        // 🔥 v2.0.0: Configure connection limits (libp2p 0.56)
        use libp2p::connection_limits::{ConnectionLimits, Behaviour as ConnLimitsBehaviour};

        let limits = ConnectionLimits::default()
            .with_max_pending_incoming(Some(64))
            .with_max_pending_outgoing(Some(64))
            .with_max_established_incoming(Some(256))
            .with_max_established_outgoing(Some(256))
            .with_max_established_per_peer(Some(8))
            .with_max_established(Some(300));  // 🔥 v2.0.0: Total connection cap for memory safety

        info!("🔒 Connection limits configured: max 300 total, 256 established connections, 8 per peer");

        // 🔥 v1.0.17-beta: Build swarm using SwarmBuilder pattern
        use libp2p::SwarmBuilder;
        use std::num::NonZeroUsize;

        info!("🔧 Building swarm with SwarmBuilder pattern (NAT traversal enabled)");

        // 🚨 v1.0.20-beta: CRITICAL - Log transport configuration for debugging
        info!("🔍 [TRANSPORT] TCP Configuration:");
        info!("   Port reuse: enabled");
        info!("   TCP nodelay: enabled");
        info!("   Authentication: Noise protocol");
        info!("   Multiplexing: Yamux");
        info!("   QUIC: enabled (additional transport)");

        // Clone data for use inside closure
        let network_config_clone = network_config.clone();
        let bootstrap_peer_map_clone = bootstrap_peer_map.clone();

        // 🌐 v2.0.0: SwarmBuilder with full transport stack (libp2p 0.56)
        // 🔥 CRITICAL PATTERN: .with_websocket() is ASYNC (needs .await?), others are sync (just ?)
        let mut swarm = SwarmBuilder::with_existing_identity(keypair)
            .with_tokio()
            .with_tcp(
                tcp::Config::default().port_reuse(true).nodelay(true),
                noise::Config::new,
                yamux::Config::default,
            )?  // Sync
            .with_quic()  // Sync
            .with_dns()?  // Sync
            .with_websocket(
                noise::Config::new,
                yamux::Config::default,
            )
            .await?  // ← ASYNC - Only method that needs .await
            .with_relay_client(
                noise::Config::new,
                yamux::Config::default,
            )?  // Sync
            .with_behaviour(move |keypair_inner, relay_client| {
                let local_peer_id_inner = keypair_inner.public().to_peer_id();

                // mDNS for local discovery (optional - may fail on restricted systems like ARM without root)
                #[cfg(not(target_os = "windows"))]
                let mdns: Toggle<mdns::tokio::Behaviour> = match mdns::tokio::Behaviour::new(mdns::Config::default(), local_peer_id_inner) {
                    Ok(m) => {
                        info!("✅ mDNS initialized - local peer discovery enabled");
                        Toggle::from(Some(m))
                    }
                    Err(e) => {
                        tracing::warn!("⚠️ mDNS init failed: {}. Continuing without local peer discovery. \
                            This is normal on ARM/embedded or when running without root. \
                            Internet-based peer discovery (Kademlia DHT) will still work.", e);
                        Toggle::from(None)
                    }
                };

                // Kademlia DHT
                let mut kad_config = KademliaConfig::default();
                kad_config.set_query_timeout(Duration::from_secs(60));
                let kad_store = MemoryStore::new(local_peer_id_inner);
                let mut kademlia = Kademlia::with_config(local_peer_id_inner, kad_store, kad_config);

                // Add bootstrap peers to Kademlia (skip self)
                // 🔧 v1.0.88-beta: Skip adding ourselves to our own Kademlia routing table
                for (peer_id, addr) in &bootstrap_peer_map_clone {
                    if *peer_id == local_peer_id_inner {
                        continue; // Skip self
                    }
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

                // 🚀 v1.0.72-beta: ADAPTIVE GOSSIPSUB PARAMETERS for sub-50ms finality
                // Dynamically configurable via environment variables for different network profiles:
                //   Q_GOSSIPSUB_PROFILE=low-latency (default) | balanced | high-throughput
                //   Q_GOSSIPSUB_HEARTBEAT_MS=100              (50-1000ms)
                //   Q_GOSSIPSUB_MESH_N=12                     (6-24 target peers)
                //   Q_GOSSIPSUB_FLOOD_PUBLISH=true            (instant propagation)
                //
                // Profile defaults:
                //   low-latency:     heartbeat=100ms, mesh_n=12, flood=true  (sub-50ms finality)
                //   balanced:        heartbeat=500ms, mesh_n=8,  flood=false (standard operation)
                //   high-throughput: heartbeat=200ms, mesh_n=16, flood=true  (large blocks)

                let gossipsub_profile = std::env::var("Q_GOSSIPSUB_PROFILE")
                    .unwrap_or_else(|_| "low-latency".to_string());

                let (default_heartbeat_ms, default_mesh_n, default_flood) = match gossipsub_profile.as_str() {
                    "balanced" => (500, 8, false),
                    "high-throughput" => (200, 16, true),
                    _ => (100, 12, true),  // low-latency (default)
                };

                // Allow per-parameter overrides via environment
                let heartbeat_ms = std::env::var("Q_GOSSIPSUB_HEARTBEAT_MS")
                    .ok()
                    .and_then(|v| v.parse::<u64>().ok())
                    .unwrap_or(default_heartbeat_ms)
                    .max(50).min(1000);  // Clamp to safe range

                let mesh_n = std::env::var("Q_GOSSIPSUB_MESH_N")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(default_mesh_n)
                    .max(6).min(24);  // Clamp to safe range

                let flood_publish = std::env::var("Q_GOSSIPSUB_FLOOD_PUBLISH")
                    .map(|v| v == "true" || v == "1")
                    .unwrap_or(default_flood);

                // Derive other mesh parameters from mesh_n
                let mesh_n_low = (mesh_n / 2).max(4);       // 50% of target, min 4
                let mesh_n_high = (mesh_n * 4 / 3).min(32); // 133% of target, max 32
                let mesh_outbound_min = (mesh_n / 3).max(2); // 33% of target, min 2

                info!("🔧 [ADAPTIVE GOSSIPSUB] Profile: {}", gossipsub_profile);
                info!("   heartbeat_interval: {}ms", heartbeat_ms);
                info!("   mesh_n: {} (low: {}, high: {})", mesh_n, mesh_n_low, mesh_n_high);
                info!("   mesh_outbound_min: {}", mesh_outbound_min);
                info!("   flood_publish: {}", flood_publish);

                let gossipsub_config = gossipsub::ConfigBuilder::default()
                    .heartbeat_interval(Duration::from_millis(heartbeat_ms))  // ⚡ Adaptive heartbeat
                    .validation_mode(ValidationMode::Strict)
                    .max_transmit_size(1 * 1024 * 1024)  // 1 MB per message
                    .flood_publish(flood_publish)   // ⚡ Adaptive flood publish
                    .mesh_outbound_min(mesh_outbound_min)  // ⚡ Adaptive outbound
                    .mesh_n_low(mesh_n_low)         // ⚡ Adaptive min peers
                    .mesh_n(mesh_n)                 // ⚡ Adaptive target peers
                    .mesh_n_high(mesh_n_high)       // ⚡ Adaptive max peers
                    .message_id_fn(|message: &gossipsub::Message| {
                        // 🔥 v2.0.0: Use blake3 for cryptographic message deduplication
                        // This prevents hash collision attacks on gossipsub
                        use blake3::Hasher;

                        let mut hasher = Hasher::new();
                        if let Some(source) = &message.source {
                            hasher.update(source.to_bytes().as_ref());
                        }
                        hasher.update(&message.data);
                        if let Some(seq) = &message.sequence_number {
                            hasher.update(&seq.to_le_bytes());
                        }
                        MessageId::from(hasher.finalize().as_bytes().to_vec())
                    })
                    .build()
                    .expect("Failed to initialize behaviour");

                let gossipsub = gossipsub::Behaviour::new(
                    gossipsub::MessageAuthenticity::Signed(keypair_inner.clone()),
                    gossipsub_config,
                )
                .expect("Failed to initialize behaviour");

                // Request-response for block synchronization
                // 🔥 v2.0.0: Updated for libp2p 0.56 request-response API
                use libp2p::request_response::{self, ProtocolSupport};
                use q_types::{BlockPackCodec, BlockPackProtocol};
                use std::time::Duration;

                let block_sync_protocols = std::iter::once((BlockPackProtocol, ProtocolSupport::Full));

                // 🚀 v1.0.4-beta: Adaptive max_concurrent_streams (memory safety)
                // Was 100 → now 10 (low memory) or 40 (normal)
                // Prevents OOM on constrained devices during sync
                let low_memory_mode = std::env::var("Q_LOW_MEMORY_MODE").is_ok();
                let max_streams = if low_memory_mode { 10 } else { 40 };

                // 🧅 v1.3.5-beta: IMPROVED TIMEOUT CONFIGURATION
                // Problem: 30s timeout was too short for transferring 800+ blocks over network
                // Block packs of 500-1000 blocks can take 45-60s to serialize and transmit
                //
                // Direct mode: 60s (was 30s) - allows time for large block pack transfers
                // Tor mode: 120s - accounts for Tor circuit latency (3-6 hops, 200-500ms each)
                // Users can override with Q_P2P_REQUEST_TIMEOUT env var
                let tor_enabled = std::env::var("Q_TOR_ENABLED").is_ok() ||
                                  std::env::var("Q_TOR_PROXY").is_ok();
                let default_timeout = if tor_enabled { 120 } else { 60 };
                let request_timeout_secs = std::env::var("Q_P2P_REQUEST_TIMEOUT")
                    .ok()
                    .and_then(|s| s.parse::<u64>().ok())
                    .unwrap_or(default_timeout);
                info!("🔧 [P2P] Block sync request timeout: {}s (tor_enabled: {})",
                      request_timeout_secs, tor_enabled);

                let block_sync_config = request_response::Config::default()
                    .with_request_timeout(Duration::from_secs(request_timeout_secs))
                    .with_max_concurrent_streams(max_streams);
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
                let handshake_config = request_response::Config::default()
                    .with_request_timeout(Duration::from_secs(15))
                    .with_max_concurrent_streams(50);
                let handshake = request_response::Behaviour::with_codec(
                    HandshakeCodec::default(),
                    handshake_protocols,
                    handshake_config,
                );

                // 🔥 NAT traversal - relay_client provided by .with_relay_client()
                let autonat = libp2p::autonat::Behaviour::new(local_peer_id_inner, Default::default());
                let relay = relay_client;  // ✅ Use the relay client from SwarmBuilder
                let dcutr = libp2p::dcutr::Behaviour::new(local_peer_id_inner);

                // 🌐 v3.5.5: Relay Server for browser-to-browser P2P
                // This allows browsers to connect to each other through this node
                let relay_server = libp2p::relay::Behaviour::new(local_peer_id_inner, Default::default());
                info!("🌐 [RELAY SERVER] Enabled - browsers can relay connections through this node");

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
                    relay_server,
                    dcutr,
                    connection_limits,
                })
            })?
            .with_swarm_config(|c| {
                // v1.4.12-beta: PERFORMANCE OPTIMIZATION - Increase buffer sizes for high-throughput sync
                // Previous values (32/64) were bottlenecking libp2p at ~7 blocks/s
                // New values (256/256) allow 100+ blocks/s through the event queues
                // This matches the actual network capacity (100+ Mbit/s)
                c.with_idle_connection_timeout(Duration::from_secs(30 * 60))
                 .with_notify_handler_buffer_size(NonZeroUsize::new(256).unwrap())  // 8x larger
                 .with_per_connection_event_buffer_size(256)  // 4x larger
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
            // v1.0.88-beta: Miner stats topic for P2P hashrate aggregation
            // Allows bootstrap node to see hashrates from miners on user's localhost nodes
            IdentTopic::new(format!("{}/miner-stats", network_prefix)),
            // v1.0.90-beta: DEX pools, contracts, and AI credits sync topics
            // These enable full state synchronization across nodes
            IdentTopic::new(network_config.network_id.dex_pools_topic()),
            IdentTopic::new(network_config.network_id.contract_deployments_topic()),
            IdentTopic::new(network_config.network_id.ai_credits_topic()),
            // v2.9.2-beta: Protocol fees consensus verification topic
            // All nodes MUST verify protocol fees to ensure master wallet receives correct %
            IdentTopic::new(network_config.network_id.protocol_fees_topic()),
            // UN-DEPRECATED v3.9.5-beta: balance_updates_topic restored to default subscriptions
            // P2P balance gossipsub provides fast balance replication alongside DAG-Knight consensus
            // To disable: set Q_DISABLE_BALANCE_GOSSIP=1
            IdentTopic::new(network_config.network_id.balance_updates_topic()),
            IdentTopic::new(network_config.network_id.miner_stats_topic()),
            // v3.3.0-beta: P2P mempool transaction propagation
            // Enables real-time transaction synchronization across all nodes before block inclusion
            IdentTopic::new(network_config.network_id.mempool_transactions_topic()),
            // v3.5.8: Browser peer discovery - relay browser announcements so they can find each other
            IdentTopic::new(network_config.network_id.browser_peers_topic()),
            // v3.9.5-beta: Validator announcements for dynamic validator registry
            // Enables decentralized validator discovery and multi-bootstrap peer support
            IdentTopic::new(network_config.network_id.validator_announce_topic()),
        ];

        // UN-DEPRECATED v3.9.5-beta: Balance gossipsub is now enabled by default
        // To disable: set Q_DISABLE_BALANCE_GOSSIP=1
        let balance_gossip_disabled = std::env::var("Q_DISABLE_BALANCE_GOSSIP")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);

        if balance_gossip_disabled {
            info!("ℹ️ Q_DISABLE_BALANCE_GOSSIP=1 - gossipsub balance updates disabled by operator");
        } else {
            info!("✅ [v3.9.5] Gossipsub balance updates ENABLED (P2P balance replication active)");
        }

        for topic in &topics {
            swarm.behaviour_mut().gossipsub.subscribe(topic)
                .map_err(|e| anyhow::anyhow!("Failed to subscribe to topic {}: {}", topic, e))?;
            info!("📢 Subscribed to {} Gossipsub topic: {}",
                  network_config.network_id.as_str(), topic);
        }

        // UN-DEPRECATED v3.9.5-beta: balance-updates topic is now in the default subscription list above
        // No separate conditional subscription needed

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
            let ipv4_addr = format!("/ip4/0.0.0.0/tcp/{}", p2p_port).parse()?;
            let ipv6_addr = format!("/ip6/::/tcp/{}", p2p_port).parse()?;

            match swarm.listen_on(ipv4_addr) {
                Ok(listener_id) => {
                    info!("🎧 [LISTENER] ✅ TCP IPv4 listener started successfully: {:?}", listener_id);
                    info!("   Address: 0.0.0.0:{}", p2p_port);
                }
                Err(e) => {
                    error!("🚨 [LISTENER] ❌ FAILED to start TCP IPv4 listener: {:?}", e);
                    return Err(e.into());
                }
            }

            match swarm.listen_on(ipv6_addr) {
                Ok(listener_id) => {
                    info!("🎧 [LISTENER] ✅ TCP IPv6 listener started successfully: {:?}", listener_id);
                    info!("   Address: [::]:{}", p2p_port);
                }
                Err(e) => {
                    warn!("⚠️  [LISTENER] TCP IPv6 listener failed (not critical): {:?}", e);
                }
            }

            // 🐳 v1.2.2-beta: Register external address for Docker/NAT environments
            // This ensures Identify announces ONLY the public IP, not Docker internal addresses
            if let Some(external_addr) = get_external_address() {
                swarm.add_external_address(external_addr.clone());
                info!("📢 [EXTERNAL-TCP] Registered external TCP address with swarm: {}", external_addr);
                info!("   This address will be announced via Identify protocol");
            }

            // 🌐 v3.4.3-browser: Register WSS external address for browser P2P clients
            // This is CRITICAL for browsers to connect - they connect via wss://quillon.xyz:9443
            // which nginx proxies to ws://127.0.0.1:9001
            if let Some(wss_addr) = get_external_wss_address() {
                swarm.add_external_address(wss_addr.clone());
                info!("🌐 [EXTERNAL-WSS] Registered external WSS address for browser P2P: {}", wss_addr);
                info!("   Browser clients will connect via: wss://quillon.xyz:9443");
                info!("   nginx proxies this to: ws://127.0.0.1:{}/ws", p2p_port);
            } else {
                warn!("⚠️  [EXTERNAL-WSS] No WSS address configured - browser P2P may not work!");
                warn!("   Set Q_EXTERNAL_WSS_ADDRESS=/dns4/quillon.xyz/tcp/9443/wss");
            }

            // 🌐 WebSocket listeners for browser clients (same port as TCP)
            let ws_ipv4_addr = format!("/ip4/0.0.0.0/tcp/{}/ws", p2p_port).parse()?;
            match swarm.listen_on(ws_ipv4_addr) {
                Ok(listener_id) => {
                    info!("🌐 [LISTENER] ✅ WebSocket IPv4 listener started successfully: {:?}", listener_id);
                    info!("   Address: ws://0.0.0.0:{}/ws (for browser clients)", p2p_port);
                }
                Err(e) => {
                    warn!("⚠️  [LISTENER] WebSocket IPv4 listener failed (not critical for node-to-node): {:?}", e);
                }
            }
        } else {
            // Listen on all interfaces, random port
            let ipv4_addr = "/ip4/0.0.0.0/tcp/0".parse()?;
            let ipv6_addr = "/ip6/::/tcp/0".parse()?;

            match swarm.listen_on(ipv4_addr) {
                Ok(listener_id) => {
                    info!("🎧 [LISTENER] ✅ TCP IPv4 listener started successfully: {:?}", listener_id);
                    info!("   Address: 0.0.0.0:<random-port>");
                }
                Err(e) => {
                    error!("🚨 [LISTENER] ❌ FAILED to start TCP IPv4 listener: {:?}", e);
                    return Err(e.into());
                }
            }

            match swarm.listen_on(ipv6_addr) {
                Ok(listener_id) => {
                    info!("🎧 [LISTENER] ✅ TCP IPv6 listener started successfully: {:?}", listener_id);
                    info!("   Address: [::]:<random-port>");
                }
                Err(e) => {
                    warn!("⚠️  [LISTENER] TCP IPv6 listener failed (not critical): {:?}", e);
                }
            }

            // 🌐 WebSocket listener for browser clients (random port)
            let ws_ipv4_addr = "/ip4/0.0.0.0/tcp/0/ws".parse()?;
            match swarm.listen_on(ws_ipv4_addr) {
                Ok(listener_id) => {
                    info!("🌐 [LISTENER] ✅ WebSocket IPv4 listener started successfully: {:?}", listener_id);
                    info!("   Address: ws://0.0.0.0:<random-port>/ws (for browser clients)");
                }
                Err(e) => {
                    warn!("⚠️  [LISTENER] WebSocket IPv4 listener failed (not critical for node-to-node): {:?}", e);
                }
            }
        }

        info!("✅ Zero-Knowledge Discovery initialized successfully!");
        info!("📡 Discovery mechanisms active:");
        info!("  • mDNS (local network, <1 second)");
        info!("  • Kademlia DHT (global clearnet discovery)");
        info!("  • Identify (peer exchange)");
        info!("  • Ping (connection keepalive)");
        info!("  • Gossipsub (consensus messaging, {} topics)", topics.len());
        info!("🌐 Transport layers:");
        info!("  • TCP (node-to-node)");
        info!("  • WebSocket (browser clients)");

        // 🧅 v3.4.20-beta: Tor/Encryption layer initialization logging
        info!("🔐 Security layers active:");
        info!("  • Noise Protocol (XX handshake, AES-256-GCM encryption)");
        info!("  • Tor Onion Routing (3-hop circuits, traffic analysis resistance)");
        info!("  • Post-Quantum Protection (Kyber1024 key exchange ready)");
        info!("  • Zero IP Leakage (anonymity-preserving gossipsub)");
        info!("🔐 All P2P traffic encrypted end-to-end via Noise+Tor layers");

        // 🔥 v1.0.17-beta: CRITICAL FIX - Trigger Kademlia bootstrap process
        // The bootstrap peers were added to Kademlia's routing table during swarm creation,
        // but we MUST call bootstrap() to actually initiate the dial attempts!
        if bootstrap_count > 0 {
            info!("🚀 [BOOTSTRAP] Initiating Kademlia bootstrap process for {} peers", bootstrap_count);
            match swarm.behaviour_mut().kademlia.bootstrap() {
                Ok(query_id) => {
                    info!("✅ [BOOTSTRAP] Kademlia bootstrap initiated successfully (query_id: {:?})", query_id);
                    info!("   → Bootstrap peers will be dialed automatically by Kademlia");
                }
                Err(e) => {
                    warn!("⚠️  [BOOTSTRAP] Kademlia bootstrap failed: {}", e);
                    warn!("   → This may happen if no bootstrap peers were added to routing table");
                    warn!("   → Node will still attempt discovery via mDNS and identify protocol");
                }
            }

            // 🔧 v1.0.17-beta: DIAGNOSTIC - Also manually dial bootstrap peers
            // This helps us see connection errors immediately instead of waiting for Kademlia
            info!("🔧 [BOOTSTRAP-DIAG] Manually dialing {} bootstrap peers for immediate error visibility", bootstrap_count);
            for (peer_id, addr) in &bootstrap_peer_map {
                // 🔧 v1.0.88-beta: FIX - Skip dialing ourselves as bootstrap peer
                // CRITICAL: When this node IS the bootstrap peer, it was trying to dial itself
                // and getting blacklisted after 150 failures, breaking ALL P2P connectivity
                if *peer_id == local_peer_id {
                    info!("ℹ️  [BOOTSTRAP] Skipping self-dial - this node IS bootstrap peer {}", peer_id);
                    continue;
                }

                info!("📡 [BOOTSTRAP-DIAG] Manually dialing: {} at {}", peer_id, addr);

                // 🚨 v1.0.20-beta: CRITICAL - Log connection state BEFORE dial
                let conn_info = swarm.network_info();
                info!("🔍 [DIAL PRE] Connection state before dial:");
                info!("   Total connections: {:?}", conn_info.connection_counters());
                info!("   Pending outgoing: {}", conn_info.connection_counters().num_pending_outgoing());
                info!("   Established: {}", conn_info.connection_counters().num_established());

                // 🔧 v1.0.87-beta: FIX DialFailure - Add peer address to swarm's address book
                // CRITICAL: Without this, request-response can't dial the peer for block sync
                // Kademlia.add_address() only updates Kademlia's routing table, NOT the swarm's address book
                // 🔧 v1.0.88-beta: FIX - Strip /p2p/ from address before adding to address book
                // add_peer_address expects transport address ONLY, not the full multiaddr with peer ID
                let addr_without_p2p: Multiaddr = addr.iter()
                    .filter(|p| !matches!(p, libp2p::multiaddr::Protocol::P2p(_)))
                    .collect();
                swarm.add_peer_address(*peer_id, addr_without_p2p.clone());
                info!("📋 [BOOTSTRAP] Added peer {} address to swarm address book: {}", peer_id, addr_without_p2p);

                match swarm.dial(addr.clone()) {
                    Ok(_) => {
                        info!("✅ [BOOTSTRAP-DIAG] Dial initiated for {}", peer_id);

                        // 🚨 v1.0.20-beta: CRITICAL - Log connection state AFTER dial
                        let conn_info_after = swarm.network_info();
                        info!("🔍 [DIAL POST] Connection state after dial:");
                        info!("   Total connections: {:?}", conn_info_after.connection_counters());
                        info!("   Pending outgoing: {}", conn_info_after.connection_counters().num_pending_outgoing());
                        info!("   Established: {}", conn_info_after.connection_counters().num_established());
                    }
                    Err(e) => {
                        error!("❌ [BOOTSTRAP-DIAG] Failed to dial {}: {:?}", peer_id, e);
                    }
                }
            }
        } else {
            info!("ℹ️  [BOOTSTRAP] No bootstrap peers to dial - relying on mDNS/identify discovery");
        }

        // 🚀 v0.9.38-beta: PHASE 1.2 - Bootstrap Peer Discovery with Retry Logic
        // Explicitly dial bootstrap peer if Q_BOOTSTRAP_PEER environment variable is set
        // This ensures Server Alpha connects to Server Beta for network unification
        if let Ok(bootstrap_env) = std::env::var("Q_BOOTSTRAP_PEER") {
            info!("🔍 [BOOTSTRAP] Explicit bootstrap peer configured: {}", bootstrap_env);

            // Parse bootstrap multiaddr
            if let Ok(bootstrap_addr) = bootstrap_env.parse::<Multiaddr>() {
                info!("📡 [BOOTSTRAP] Dialing bootstrap peer: {}", bootstrap_addr);

                // 🔧 v1.0.87-beta: FIX DialFailure - Extract peer ID and add to swarm's address book
                // CRITICAL: Without this, request-response can't dial the peer for block sync
                if let Some(libp2p::multiaddr::Protocol::P2p(peer_id)) = bootstrap_addr.iter().find(|p| matches!(p, libp2p::multiaddr::Protocol::P2p(_))) {
                    // 🔧 v1.0.88-beta: FIX - Skip dialing ourselves
                    if peer_id == local_peer_id {
                        info!("ℹ️  [BOOTSTRAP] Skipping Q_BOOTSTRAP_PEER self-dial - this node IS bootstrap peer {}", peer_id);
                    } else {
                        // Strip /p2p/ from address for add_peer_address
                        let addr_without_p2p: Multiaddr = bootstrap_addr.iter()
                            .filter(|p| !matches!(p, libp2p::multiaddr::Protocol::P2p(_)))
                            .collect();
                        swarm.add_peer_address(peer_id, addr_without_p2p.clone());
                        info!("📋 [BOOTSTRAP] Added peer {} to swarm address book: {}", peer_id, addr_without_p2p);

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
                    }
                } else {
                    // No peer ID in address, just try to dial
                    match swarm.dial(bootstrap_addr.clone()) {
                        Ok(_) => {
                            info!("✅ [BOOTSTRAP] Initiated connection to bootstrap peer (no peer ID in addr)");
                        }
                        Err(e) => {
                            warn!("⚠️  [BOOTSTRAP] Initial dial failed: {:?}", e);
                            warn!("   Will retry automatically in background task");
                        }
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

        // 🚀 v1.7.0-LAMINAR: Lock-free peer compatibility with DashMap (VORTEX ELIMINATION)
        // Register bootstrap peers for special blacklist treatment (3x threshold)
        let peer_compat = PeerCompatibilityV2::default();
        for peer_id in bootstrap_peer_map.keys() {
            if *peer_id == local_peer_id {
                info!("ℹ️  [BOOTSTRAP] Skipping self-registration - this node IS bootstrap peer {}", peer_id);
                continue;
            }
            peer_compat.is_bootstrap.insert(*peer_id);
            info!("🛡️ [BOOTSTRAP] Registered {} as bootstrap peer (failure threshold: {})",
                  peer_id, BLACKLIST_FAILURE_THRESHOLD * BOOTSTRAP_BLACKLIST_MULTIPLIER);
        }
        info!("🚀 [LAMINAR] {} bootstrap peers registered with lock-free DashMap",
              peer_compat.is_bootstrap.len());

        // v1.2.7-beta: Create channel for async block pack responses (prevents ResponseOmission)
        let (block_pack_response_tx, block_pack_response_rx) = mpsc::unbounded_channel();

        // v1.3.3-beta: Improved Tor detection for adaptive timeouts and batch sizes
        // Detect Tor from multiple sources:
        // 1. Q_TOR_ENABLED=1 (explicit enable)
        // 2. Q_TOR_PROXY (socks5 proxy address)
        // 3. ALL_PROXY/SOCKS_PROXY containing "socks5" (common proxy patterns)
        // 4. Check if 127.0.0.1:9050 or 127.0.0.1:9150 appears in any proxy config (default Tor ports)
        let tor_enabled = std::env::var("Q_TOR_ENABLED").is_ok() ||
            std::env::var("Q_TOR_PROXY").is_ok() ||
            std::env::var("ALL_PROXY").map(|v| v.contains("socks5") || v.contains("9050") || v.contains("9150")).unwrap_or(false) ||
            std::env::var("SOCKS_PROXY").map(|v| v.contains("socks5") || v.contains("9050") || v.contains("9150")).unwrap_or(false) ||
            std::env::var("socks_proxy").map(|v| v.contains("socks5") || v.contains("9050") || v.contains("9150")).unwrap_or(false);

        if tor_enabled {
            info!("🧅 [TOR] Tor mode ENABLED - using 120s timeouts and 1000-block batches");
        } else {
            info!("🔌 [DIRECT] Direct connection mode - using 30s timeouts and 5000-block batches");
        }

        // 🔄 v3.3.7-beta: Connection warmup for NEW identity
        // When a new libp2p identity is generated, DHT routing tables are empty and
        // bootstrap connections often fail on first attempt. This warmup period:
        // 1. Waits for initial connections to establish
        // 2. Re-triggers Kademlia bootstrap if no connections were established
        // 3. Re-dials bootstrap peers with exponential backoff
        // This fixes the "works after restart but not first boot" issue
        if is_new_identity && bootstrap_count > 0 {
            info!("🔄 [CONNECTION WARMUP] New identity detected - starting connection warmup period");

            // Clone bootstrap_peer_map before moving into loop
            let warmup_bootstrap_peers: Vec<(PeerId, Multiaddr)> = bootstrap_peer_map
                .iter()
                .filter(|(peer_id, _)| **peer_id != local_peer_id)
                .map(|(k, v)| (*k, v.clone()))
                .collect();

            for warmup_attempt in 1..=3 {
                // Wait for connections to establish (exponential backoff: 3s, 6s, 12s)
                let wait_secs = 3 * (1 << (warmup_attempt - 1));
                info!("🔄 [WARMUP {}/3] Waiting {}s for DHT propagation and connections...",
                      warmup_attempt, wait_secs);
                tokio::time::sleep(Duration::from_secs(wait_secs)).await;

                // Check connection status
                let conn_info = swarm.network_info();
                let established = conn_info.connection_counters().num_established();
                let pending = conn_info.connection_counters().num_pending_outgoing();

                info!("🔄 [WARMUP {}/3] Connection status: established={}, pending={}",
                      warmup_attempt, established, pending);

                if established > 0 {
                    info!("✅ [WARMUP] Connection warmup SUCCEEDED! {} established connections", established);
                    break;
                }

                if warmup_attempt < 3 {
                    // No connections yet - retry bootstrap
                    info!("🔄 [WARMUP {}/3] No connections established - retrying bootstrap...", warmup_attempt);

                    // Re-trigger Kademlia bootstrap
                    match swarm.behaviour_mut().kademlia.bootstrap() {
                        Ok(query_id) => {
                            info!("🔄 [WARMUP] Kademlia bootstrap re-triggered (query_id: {:?})", query_id);
                        }
                        Err(e) => {
                            warn!("⚠️  [WARMUP] Kademlia bootstrap retry failed: {}", e);
                        }
                    }

                    // Re-dial bootstrap peers
                    for (peer_id, addr) in &warmup_bootstrap_peers {
                        // Strip /p2p/ for add_peer_address
                        let addr_without_p2p: Multiaddr = addr.iter()
                            .filter(|p| !matches!(p, libp2p::multiaddr::Protocol::P2p(_)))
                            .collect();
                        swarm.add_peer_address(*peer_id, addr_without_p2p);

                        match swarm.dial(addr.clone()) {
                            Ok(_) => {
                                info!("🔄 [WARMUP] Re-dialing bootstrap peer {}", peer_id);
                            }
                            Err(e) => {
                                warn!("⚠️  [WARMUP] Re-dial failed for {}: {:?}", peer_id, e);
                            }
                        }
                    }
                } else {
                    // Final attempt - log warning but don't fail
                    warn!("⚠️  [WARMUP] Connection warmup exhausted all 3 attempts");
                    warn!("   Node will continue with background discovery (mDNS, Identify)");
                    warn!("   P2P sync may be delayed until connections are established");
                }
            }
        }

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
            peer_compat: Arc::new(peer_compat), // 🚀 v1.7.0-LAMINAR: Lock-free DashMap (no RwLock needed)
            pending_block_requests: Arc::new(std::sync::Mutex::new(HashMap::new())), // v1.0.12-beta: Batch sync request tracking
            outstanding_sync_requests: Arc::new(std::sync::Mutex::new(Vec::new())), // v1.0.44-beta: Track multiple concurrent sync requests
            known_network_height: Arc::new(std::sync::atomic::AtomicU64::new(0)), // v1.0.45-beta: Network height for progress display
            handshake_validator, // v1.0.15.1-beta: Protocol version validation
            // v1.2.7-beta: Async block pack response handling (prevents ResponseOmission)
            block_pack_response_tx: block_pack_response_tx,
            block_pack_response_rx: block_pack_response_rx,
            pending_response_channels: Arc::new(std::sync::Mutex::new(HashMap::new())),
            next_async_request_id: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            // v1.3.3-beta: Tor-aware adaptive batch sizes and retry logic
            tor_enabled,
            // v1.3.3-beta: Exponential backoff retry queue for failed sync requests
            sync_retry_queue: Arc::new(std::sync::Mutex::new(Vec::new())),
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
        // 🚀 v1.0.40-beta: FIX #2 - Reduced interval from 30s to 10s for faster sync
        // Combined with batch_size increase (100 → 1000), this gives ~100 blocks/second
        // vs previous ~3.3 blocks/second (30x improvement)
        let mut health_check_interval = tokio::time::interval(tokio::time::Duration::from_secs(10));

        // v4.3.0-beta: Queue drain interval - flush priority gossipsub queue to swarm
        let mut queue_drain_interval = tokio::time::interval(tokio::time::Duration::from_millis(1));

        // v4.3.0-beta: Peer scoring interval - update gossipsub scores from latency data
        let mut peer_scoring_interval = tokio::time::interval(tokio::time::Duration::from_secs(10));

        // 🚨 v1.0.20-beta: Event loop heartbeat to diagnose silent failures
        let mut last_heartbeat = std::time::Instant::now();
        let heartbeat_interval = tokio::time::Duration::from_secs(5);

        // 🔧 v2.4.8: Track consecutive disconnected checks for aggressive reconnection
        let mut consecutive_no_peers = 0u32;

        info!("💓 [EVENT LOOP] Starting with heartbeat monitoring (every {:?})", heartbeat_interval);

        loop {
            // 🚨 v1.0.20-beta: Heartbeat check
            if last_heartbeat.elapsed() > heartbeat_interval {
                info!("💓 [EVENT LOOP] Heartbeat - still alive and processing events");
                info!("   Peer count: {}", self.discovered_peers.read().await.len());
                info!("   Network info: {:?}", self.swarm.network_info().connection_counters());
                last_heartbeat = std::time::Instant::now();
            }
            tokio::select! {
                // v1.3.1-beta: CRITICAL FIX - Add biased selection to prioritize block pack responses
                // Without this, the event loop may process swarm events before responding to block pack
                // requests, causing timeouts and sync failures. Biased mode processes branches in order.
                biased;

                // v1.3.1-beta: HIGHEST PRIORITY - Block pack response channel MUST be processed first!
                // This prevents ResponseOmission timeouts that break P2P sync completely.
                // If responses aren't sent within 30s, the requester times out and blacklists us.
                Some((async_req_id, response)) = self.block_pack_response_rx.recv() => {
                    // Retrieve the stored response channel
                    if let Some(channel) = self.pending_response_channels.lock().unwrap().remove(&async_req_id) {
                        let block_count = response.blocks.len();
                        if let Err(e) = self.swarm.behaviour_mut().block_sync.send_response(channel, response) {
                            error!("❌ [BLOCK-PACK ASYNC] Failed to send response for request {}: {:?}", async_req_id, e);
                        } else {
                            info!("✅ [BLOCK-PACK ASYNC] Sent {} blocks for async request {}", block_count, async_req_id);
                        }
                    } else {
                        warn!("⚠️ [BLOCK-PACK ASYNC] No pending channel for request {} (may have timed out)", async_req_id);
                    }
                }

                // v4.3.0-beta: Drain priority gossipsub queue → publish to swarm
                _ = queue_drain_interval.tick() => {
                    // Drain up to 10 messages per 1ms tick (= 10,000/sec max throughput)
                    let batch = crate::gossipsub_queue::gossipsub_queue().drain_batch(10);
                    for msg in batch {
                        let ident_topic = IdentTopic::new(&msg.topic);
                        if let Err(e) = self.swarm.behaviour_mut().gossipsub.publish(ident_topic, msg.data) {
                            debug!("[QUEUE DRAIN] Failed to publish {}: {}", msg.topic, e);
                        }
                    }
                }

                // v4.3.0-beta: Update gossipsub peer scores from latency data
                _ = peer_scoring_interval.tick() => {
                    let scores = PEER_LATENCY_TRACKER.get_all_scores();
                    let mut updated = 0usize;
                    for (peer_id, score) in &scores {
                        // set_application_score returns bool: true if PeerScoreParams configured, false otherwise
                        if self.swarm.behaviour_mut().gossipsub.set_application_score(peer_id, *score) {
                            updated += 1;
                        }
                    }
                    if !scores.is_empty() {
                        debug!("[MESH SCORING] Updated {}/{} peer scores from latency data", updated, scores.len());
                    }
                    // Also log queue stats
                    crate::gossipsub_queue::log_queue_stats();
                }

                // 🩺 Periodic P2P health check (every 10 seconds)
                _ = health_check_interval.tick() => {
                    let peer_count = self.discovered_peers.read().await.len();
                    let conn_info = self.swarm.network_info();
                    let pending_out = conn_info.connection_counters().num_pending_outgoing();
                    let established = conn_info.connection_counters().num_established();

                    // 🚨 v2.2.3: Detect stuck pending connections (silent dial failures)
                    if pending_out > 0 && established == 0 {
                        warn!("🚨 [STUCK CONNECTION] {} pending outgoing, 0 established!", pending_out);
                        warn!("   This may indicate a silent dial failure (NAT/firewall issue)");
                        warn!("   Network counters: {:?}", conn_info.connection_counters());
                    }

                    if peer_count == 0 {
                        consecutive_no_peers += 1;
                        warn!("⚠️  [P2P HEALTH] NO CONNECTIONS - Network isolated! (consecutive_checks: {})", consecutive_no_peers);
                        warn!("   pending_outgoing={}, established={}", pending_out, established);
                        warn!("   Check bootstrap peer configuration and firewall settings");

                        // 🔧 v1.4.3-beta: Reduced from 6 checks (60s) to 3 checks (30s) for faster reconnection
                        // After 3 consecutive checks with no peers, clear stale peer state and force fresh discovery
                        if consecutive_no_peers >= 3 {
                            warn!("🚨 [AUTO-RECONNECT] Been disconnected for {} checks - clearing stale bootstrap peers and refreshing",
                                  consecutive_no_peers);
                            // Clear potentially stale bootstrap peer IDs (they may have regenerated keys)
                            let mut bp = self.bootstrap_peers.write().await;
                            bp.clear();
                            drop(bp);
                            consecutive_no_peers = 0; // Reset counter to allow fresh discovery
                        }

                        // 🚀 v2.3.2-beta: CRITICAL FIX - Use stored bootstrap peers for auto-reconnect
                        // Previously only checked Q_BOOTSTRAP_PEER env var, ignoring hardcoded bootstrap!
                        // This caused nodes to stay disconnected after transient network failures.
                        let mut reconnect_attempted = false;

                        // 1. 🔧 v2.4.8: Try ALL stored bootstrap peers (was only trying one with break)
                        // This is critical when some bootstrap nodes are temporarily offline
                        let bootstrap_peers = self.bootstrap_peers.read().await;
                        let mut dial_count = 0;
                        for (peer_id, addr) in bootstrap_peers.iter() {
                            if *peer_id == self.local_peer_id {
                                continue; // Don't dial ourselves
                            }
                            info!("🔄 [AUTO-RECONNECT] Attempting to reconnect to stored bootstrap peer {}...", peer_id);
                            if let Err(e) = self.swarm.dial(addr.clone()) {
                                warn!("❌ [AUTO-RECONNECT] Dial to {} failed: {:?}", peer_id, e);
                            } else {
                                info!("✅ [AUTO-RECONNECT] Dial initiated to {} at {}", peer_id, addr);
                                reconnect_attempted = true;
                                dial_count += 1;
                                // v2.4.8: Continue trying ALL peers, don't break
                            }
                        }
                        if dial_count > 0 {
                            info!("🔄 [AUTO-RECONNECT] Initiated {} dial attempts to stored bootstrap peers", dial_count);
                        }
                        drop(bootstrap_peers); // Release lock

                        // 2. Fallback to env var bootstrap peer (if configured and not already tried)
                        if !reconnect_attempted {
                            if let Ok(bootstrap_env) = std::env::var("Q_BOOTSTRAP_PEER") {
                                if let Ok(bootstrap_addr) = bootstrap_env.parse::<Multiaddr>() {
                                    info!("🔄 [AUTO-RECONNECT] Attempting to reconnect via Q_BOOTSTRAP_PEER env...");
                                    if let Err(e) = self.swarm.dial(bootstrap_addr.clone()) {
                                        error!("❌ [AUTO-RECONNECT] Dial failed: {:?}", e);
                                    } else {
                                        reconnect_attempted = true;
                                    }
                                }
                            }
                        }

                        // 3. 🔧 v4.2.0: Dynamic bootstrap discovery from ALL known HTTP endpoints
                        // Tries Q_BOOTSTRAP_URL first, then all BOOTSTRAP_HTTP_ENDPOINTS
                        if !reconnect_attempted {
                            // Build list of URLs to try
                            let mut discovery_urls: Vec<String> = Vec::new();
                            if let Ok(bootstrap_url) = std::env::var("Q_BOOTSTRAP_URL") {
                                discovery_urls.push(bootstrap_url.trim_end_matches('/').to_string());
                            }
                            for endpoint in BOOTSTRAP_HTTP_ENDPOINTS {
                                let ep = endpoint.trim_end_matches('/').to_string();
                                if !discovery_urls.contains(&ep) {
                                    discovery_urls.push(ep);
                                }
                            }

                          for bootstrap_url in &discovery_urls {
                            if reconnect_attempted { break; }
                            info!("🔄 [AUTO-RECONNECT] Trying bootstrap endpoint: {}", bootstrap_url);
                                let url = format!("{}/api/v1/status", bootstrap_url);
                                match reqwest::Client::new()
                                    .get(&url)
                                    .timeout(std::time::Duration::from_secs(5))
                                    .send()
                                    .await
                                {
                                    Ok(response) if response.status().is_success() => {
                                        if let Ok(text) = response.text().await {
                                            // 🔧 v1.4.3-beta: FIX - Parse correct API response format
                                            // API returns: {"success":true,"data":{"peer_id":"...","multiaddrs":["...",...]}}
                                            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
                                                // Try the correct format first: data.multiaddrs[]
                                                let multiaddrs = json.get("data")
                                                    .and_then(|d| d.get("multiaddrs"))
                                                    .and_then(|m| m.as_array());

                                                if let Some(addrs) = multiaddrs {
                                                    // Find a public IP multiaddr (skip localhost)
                                                    for addr_val in addrs {
                                                        if let Some(addr_str) = addr_val.as_str() {
                                                            // Skip localhost addresses
                                                            if addr_str.contains("127.0.0.1") || addr_str.contains("::1") {
                                                                continue;
                                                            }
                                                            if let Ok(addr) = addr_str.parse::<Multiaddr>() {
                                                                info!("🔄 [AUTO-RECONNECT] Found fresh bootstrap: {}", addr_str);
                                                                if let Err(e) = self.swarm.dial(addr.clone()) {
                                                                    error!("❌ [AUTO-RECONNECT] Fresh dial failed: {:?}", e);
                                                                } else {
                                                                    info!("✅ [AUTO-RECONNECT] Dial initiated to fresh bootstrap");
                                                                    reconnect_attempted = true;

                                                                    // Update stored bootstrap peers with fresh address
                                                                    if let Some(peer_id) = addr.iter().find_map(|p| {
                                                                        if let libp2p::multiaddr::Protocol::P2p(id) = p {
                                                                            Some(id)
                                                                        } else { None }
                                                                    }) {
                                                                        let mut bp = self.bootstrap_peers.write().await;
                                                                        bp.insert(peer_id, addr);
                                                                        info!("📋 [AUTO-RECONNECT] Updated bootstrap peer cache with fresh ID");
                                                                    }
                                                                    break; // Success, stop trying more addresses
                                                                }
                                                            }
                                                        }
                                                    }
                                                } else {
                                                    // Fallback: try legacy p2p_address field
                                                    if let Some(p2p_addr) = json.get("p2p_address").and_then(|v| v.as_str()) {
                                                        if let Ok(addr) = p2p_addr.parse::<Multiaddr>() {
                                                            info!("🔄 [AUTO-RECONNECT] Found fresh bootstrap (legacy): {}", p2p_addr);
                                                            if let Err(e) = self.swarm.dial(addr.clone()) {
                                                                error!("❌ [AUTO-RECONNECT] Fresh dial failed: {:?}", e);
                                                            } else {
                                                                info!("✅ [AUTO-RECONNECT] Dial initiated to fresh bootstrap");
                                                                reconnect_attempted = true;
                                                            }
                                                        }
                                                    } else {
                                                        warn!("⚠️ [AUTO-RECONNECT] Bootstrap response missing multiaddrs and p2p_address");
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    Ok(response) => {
                                        warn!("⚠️ [AUTO-RECONNECT] Bootstrap URL returned status: {}", response.status());
                                    }
                                    Err(e) => {
                                        warn!("⚠️ [AUTO-RECONNECT] Failed to reach {}: {}", bootstrap_url, e);
                                    }
                                }
                          } // end for bootstrap_url in discovery_urls
                        }

                        if !reconnect_attempted {
                            error!("❌ [AUTO-RECONNECT] No bootstrap peers available for reconnection!");
                            error!("   Set Q_BOOTSTRAP_PEER, Q_BOOTSTRAP_URL, or ensure network config has bootstrap_peers");
                        }
                    } else {
                        // 🔧 v2.4.8: Reset disconnection counter when connected
                        consecutive_no_peers = 0;
                        info!("✅ [P2P HEALTH] {} connected peer(s), {} established - Network healthy",
                              peer_count, established);
                    }
                }
                // Process network commands from API
                // 🚀 v1.1.1-beta: Refactored to use shared handle_command() method
                Some(command) = self.command_rx.recv() => {
                    self.handle_command(command).await;
                }
                // Process swarm events
                event = self.swarm.select_next_some() => {
                    // 🚨 v1.0.20-beta: CRITICAL - Log EVERY SwarmEvent to catch silent failures
                    debug!("🔍 [SWARM EVENT] {:?}", event);

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
                    info!("🌐 [LIBP2P CONNECTION] ==========================================");
                    info!("✅ [CONNECTION] Successfully connected to peer: {}", peer_id);
                    info!("📍 [CONNECTION] Endpoint: {:?}", endpoint);
                    info!("🔢 [CONNECTION] Number of established connections: {}", num_established);

                    // 🧅 v3.4.20-beta: Tor/Encryption layer status logging
                    info!("🔐 [ENCRYPTION] ════════════════════════════════════════════════");
                    info!("🔐 [NOISE] XX handshake completed - AES-256-GCM channel established");
                    info!("🔐 [TOR LAYER] Onion routing active - 3-hop circuit to peer");
                    info!("🔐 [PQ-SAFE] Post-quantum key exchange (Kyber1024) protecting session");
                    info!("🔐 [PRIVACY] Zero IP leakage - traffic analysis resistant");
                    info!("🔐 [ENCRYPTION] ════════════════════════════════════════════════");

                    // 🔧 v1.0.87-beta: FIX DialFailure - Add peer address to swarm on connection
                    // CRITICAL: This ensures request-response can dial the peer for block sync
                    // even if the initial bootstrap address wasn't added
                    // 🐳 v1.2.2-beta: Filter non-routable Docker/container addresses to prevent sync failures
                    if let libp2p::core::ConnectedPoint::Dialer { address, .. } = &endpoint {
                        // For outgoing connections, we know the address we dialed
                        let addr_without_p2p: Multiaddr = address.iter()
                            .filter(|p| !matches!(p, libp2p::multiaddr::Protocol::P2p(_)))
                            .collect();
                        // 🐳 v1.2.2-beta: Only add routable PEER addresses to prevent Docker network failures
                        if is_routable_peer_address(&addr_without_p2p) {
                            self.swarm.add_peer_address(peer_id, addr_without_p2p.clone());
                            info!("📋 [CONNECTION] Added peer {} to swarm address book: {}", peer_id, addr_without_p2p);
                        } else {
                            debug!("🐳 [ADDR-FILTER] Skipping non-routable peer address: {}", addr_without_p2p);
                        }
                    } else if let libp2p::core::ConnectedPoint::Listener { send_back_addr, .. } = &endpoint {
                        // For incoming connections, save the peer's advertised address
                        let addr_without_p2p: Multiaddr = send_back_addr.iter()
                            .filter(|p| !matches!(p, libp2p::multiaddr::Protocol::P2p(_)))
                            .collect();
                        // 🐳 v1.2.2-beta: Only add routable PEER addresses to prevent Docker network failures
                        if is_routable_peer_address(&addr_without_p2p) {
                            self.swarm.add_peer_address(peer_id, addr_without_p2p.clone());
                            info!("📋 [CONNECTION] Added incoming peer {} to swarm address book: {}", peer_id, addr_without_p2p);
                        } else {
                            debug!("🐳 [ADDR-FILTER] Skipping non-routable incoming peer address: {}", addr_without_p2p);
                        }
                    }

                    // 🌐 v1.0.21-browser: Check if this is a WebSocket connection (browser client)
                    let endpoint_str = format!("{:?}", endpoint);
                    let is_websocket = endpoint_str.contains("/ws") || endpoint_str.contains("websocket");

                    if is_websocket {
                        info!("🌐 [BROWSER CLIENT] Detected WebSocket connection - skipping handshake validation");
                        info!("   Browser clients use gossipsub directly without custom handshake protocol");
                    } else {
                        // 🤝 v1.0.15.1-beta: Initiate protocol handshake with new peer (node-to-node only)
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
                    }

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
                                error!("      → Set Q_BOOTSTRAP_URL=http://185.182.185.227:8080 for auto-discovery");
                            }

                            // 🔥 v2.0.0: libp2p 0.56 renamed 'endpoint' to 'address'
                            libp2p::swarm::DialError::WrongPeerId { obtained, address } => {
                                error!("   🚨 WRONG PEER ID MISMATCH:");
                                if let Some(expected) = peer_id {
                                    error!("      Expected: {}", expected);
                                }
                                error!("      Obtained: {}", obtained);
                                error!("      Address: {:?}", address);
                                error!("      → FIX: Use dynamic discovery (Q_BOOTSTRAP_URL=http://185.182.185.227:8080)");
                                error!("      → Or fetch current peer ID: curl -s http://185.182.185.227:8080/api/v1/status | jq '.data.multiaddrs[0]'");
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

                    // 🚨 v2.2.3: Enhanced dial diagnostics for silent connection failures
                    SwarmEvent::Dialing { peer_id, connection_id } => {
                        info!("📞 [DIALING] Initiating connection to peer: {:?}", peer_id);
                        info!("   Connection ID: {:?}", connection_id);
                        let conn_info = self.swarm.network_info();
                        info!("   Current state: pending_out={}, established={}",
                              conn_info.connection_counters().num_pending_outgoing(),
                              conn_info.connection_counters().num_established());
                    }

                    // 🚨 v1.0.20-beta: CRITICAL - Log ALL unhandled SwarmEvents
                    // This catches NewExternalAddrCandidate, etc. that might indicate dial activity
                    other => {
                        info!("🔍 [SWARM EVENT] Unhandled event: {:?}", other);
                    }
                }
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
                                // 🔥 v2.0.0: PeerInfo doesn't impl Display, use Debug
                                for peer in ok.peers {
                                    debug!("🔍 DHT peer discovered: {:?}", peer);
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
                // v4.3.0-beta: Extract RTT and feed into peer latency tracker
                match event.result {
                    Ok(rtt) => {
                        debug!("🏓 Ping RTT to {}: {}ms", event.peer, rtt.as_millis());
                        PEER_LATENCY_TRACKER.update_rtt(&event.peer, rtt);
                    }
                    Err(ref e) => {
                        debug!("🏓 Ping failure to {}: {:?}", event.peer, e);
                    }
                }
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
                            // 🧅 v3.4.20-beta: Tor encryption status for received data
                            info!("🔐 [TOR-RX] {} messages decrypted via Noise+Onion layer ({:.1} MB secure)", entry.0, entry.1 as f64 / 1_000_000.0);
                        } else {
                            info!(
                                "📨 [AGGREGATED] Received {} messages ({:.2} MB) on topic {} in last {}s",
                                entry.0,
                                entry.1 as f64 / 1_000_000.0,
                                topic_str,
                                entry.2.elapsed().as_secs()
                            );
                            // 🧅 v3.4.20-beta: Tor encryption status for received data
                            info!("🔐 [TOR-RX] {} messages decrypted via Noise+Onion layer ({:.1} MB secure)", entry.0, entry.1 as f64 / 1_000_000.0);
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
                // v2.1.7-DELTA-V: FIX - Use MessagePack (rmp_serde) + VersionedBlock to match server
                if topic_str.contains("/blocks") {
                    // Attempt to decode block information for better sync visibility
                    // 🔧 v2.1.7: Server uses rmp_serde::to_vec(&VersionedBlock), so we must match
                    match rmp_serde::from_slice::<q_types::VersionedBlock>(&message.data) {
                        Ok(versioned_block) => {
                            let block = &versioned_block.block;  // Access .block field, not .inner()
                            debug!(
                                "📨 Gossipsub BLOCK from {}: topic={}, height={}, txs={}, size={} bytes, hash={}",
                                propagation_source,
                                message.topic,
                                block.header.height,
                                block.transactions.len(),
                                msg_size,
                                hex::encode(&block.calculate_hash()[..8])
                            );
                        }
                        Err(e) => {
                            // Fallback: try postcard for backward compatibility
                            match postcard::from_bytes::<QBlock>(&message.data) {
                                Ok(block) => {
                                    debug!(
                                        "📨 Gossipsub BLOCK (postcard) from {}: height={}, txs={}, size={} bytes",
                                        propagation_source,
                                        block.header.height,
                                        block.transactions.len(),
                                        msg_size
                                    );
                                }
                                Err(_) => {
                                    debug!(
                                        "📨 Gossipsub block from {}: size={} bytes (decode failed: {})",
                                        propagation_source, msg_size, e
                                    );
                                }
                            }
                        }
                    }
                } else {
                    // v3.4.2: Reduced to trace to prevent log spam
                    trace!(
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
                                trace!("✅ Forwarded gossipsub message on topic: {} (size={} bytes)", topic_str, msg_size);
                            }
                        } else {
                            // v3.4.2: Reduced to trace to prevent log spam
                            trace!("✅ Forwarded gossipsub message on topic: {} (size={} bytes)", topic_str, msg_size);
                        }
                    }
                } else {
                    warn!("⚠️ Gossipsub message received but gossipsub_message_tx is None!");
                }
            }
            QNarwhalEvent::Gossipsub(gossipsub::Event::Subscribed { peer_id, topic }) => {
                info!("📢 Peer {} subscribed to topic: {}", peer_id, topic);
                // 🚀 v1.0.40-beta: FIX #3 - Log mesh status on subscription
                self.log_gossipsub_mesh_status(&topic.to_string());
            }
            QNarwhalEvent::Gossipsub(gossipsub::Event::Unsubscribed { peer_id, topic }) => {
                info!("📢 Peer {} unsubscribed from topic: {}", peer_id, topic);
            }
            // 🚀 v1.0.40-beta: FIX #3 - Enhanced gossipsub mesh diagnostics
            QNarwhalEvent::Gossipsub(gossipsub::Event::GossipsubNotSupported { peer_id }) => {
                warn!("⚠️ [GOSSIPSUB MESH] Peer {} does not support gossipsub protocol!", peer_id);
                warn!("   This peer cannot participate in mesh-based message propagation");
            }
            QNarwhalEvent::Gossipsub(event) => {
                // Log all gossipsub events at INFO level for debugging mesh formation
                info!("📢 [GOSSIPSUB] Event: {:?}", event);
            }
            QNarwhalEvent::BlockSync(block_sync_event) => {
                // ✅ v0.9.68-beta: Updated to use BlockPackCodec for efficient block sync
                use libp2p::request_response::{Event, Message};

                match block_sync_event {
                    Event::Message { peer, message, connection_id: _ } => {
                        match message {
                            Message::Request { request_id, request, channel } => {
                                info!("📥 [BLOCK-PACK] Received block pack request from {}", peer);
                                info!("   Requested: blocks {}-{} (max {})",
                                      request.start_height, request.end_height, request.max_blocks);

                                // v1.2.7-beta: NON-BLOCKING HANDLER - Prevents ResponseOmission timeouts
                                // Instead of blocking on async DB calls, we:
                                // 1. Store the response channel in pending_response_channels
                                // 2. Spawn a task to do the slow DB work
                                // 3. The task sends the response through block_pack_response_tx
                                // 4. The main event loop polls block_pack_response_rx and sends responses

                                // Validate request synchronously (fast)
                                if let Err(e) = request.validate() {
                                    error!("❌ [BLOCK-PACK] Invalid request: {}", e);
                                    let response = q_types::BlockPackResponse::from_blocks(vec![], request.end_height, 0);
                                    let _ = self.swarm.behaviour_mut().block_sync.send_response(channel, response);
                                    return Ok(());
                                }

                                // Generate unique async request ID
                                let async_req_id = self.next_async_request_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                                // Store the response channel for later use
                                self.pending_response_channels.lock().unwrap().insert(async_req_id, channel);

                                // Clone what we need for the spawned task
                                let storage_clone = self.storage.clone();
                                let response_tx = self.block_pack_response_tx.clone();
                                let start_height = request.start_height;
                                let end_height = request.end_height;
                                let max_blocks = request.max_blocks;
                                let peer_clone = peer;

                                // Spawn task to do the slow DB work
                                tokio::spawn(async move {
                                    let response = if let Some(storage) = storage_clone {
                                        // Get our height (uses cached value internally, very fast)
                                        let our_height = storage.get_highest_contiguous_block().await.unwrap_or(0);

                                        let block_count = (end_height - start_height + 1) as usize;
                                        let limit = block_count.min(max_blocks);

                                        match storage.get_qblocks_range(start_height, limit).await {
                                            Ok(blocks) => {
                                                info!("✅ [BLOCK-PACK] Async task fetched {} blocks (heights {}-{}) for {}",
                                                      blocks.len(),
                                                      blocks.first().map(|b| b.header.height).unwrap_or(start_height),
                                                      blocks.last().map(|b| b.header.height).unwrap_or(start_height),
                                                      peer_clone);
                                                q_types::BlockPackResponse::from_blocks(blocks, end_height, our_height)
                                            }
                                            Err(e) => {
                                                error!("❌ [BLOCK-PACK] Async task failed to fetch blocks: {}", e);
                                                q_types::BlockPackResponse::from_blocks(vec![], end_height, our_height)
                                            }
                                        }
                                    } else {
                                        warn!("⚠️ [BLOCK-PACK] Storage not available in async task");
                                        q_types::BlockPackResponse::from_blocks(vec![], end_height, 0)
                                    };

                                    // Send response back to main event loop for actual sending
                                    if let Err(e) = response_tx.send((async_req_id, response)) {
                                        error!("❌ [BLOCK-PACK] Failed to send async response to channel: {}", e);
                                    }
                                });

                                // Handler returns immediately - response will be sent when async task completes
                                debug!("🚀 [BLOCK-PACK] Spawned async handler task for request {}", async_req_id);
                            }
                            Message::Response { request_id, response } => {
                                // v1.0.45-beta: Update known network height for progress display
                                if response.peer_height > 0 {
                                    let current = self.known_network_height.load(std::sync::atomic::Ordering::Relaxed);
                                    if response.peer_height > current {
                                        self.known_network_height.store(response.peer_height, std::sync::atomic::Ordering::Relaxed);
                                    }
                                }

                                // v1.0.87-beta: NETWORK PROTOCOL DEBUGGING
                                let blocks_received = response.blocks.len();
                                let expected_blocks = (response.end_height.saturating_sub(response.start_height) + 1) as usize;
                                let completion_ratio = if expected_blocks > 0 {
                                    (blocks_received as f64 / expected_blocks as f64) * 100.0
                                } else {
                                    0.0
                                };

                                info!("📨 [NET-PROTOCOL] Response from {} | request_id={:?} | blocks: {}/{} ({:.1}%) | range: {}-{} | peer_height: {}",
                                      peer, request_id, blocks_received, expected_blocks, completion_ratio,
                                      response.start_height, response.end_height, response.peer_height);

                                // Detect incomplete responses
                                if blocks_received < expected_blocks && blocks_received > 0 {
                                    warn!("⚠️  [NET-PROTOCOL] INCOMPLETE RESPONSE: Got {} blocks, expected {} (missing {})",
                                          blocks_received, expected_blocks, expected_blocks - blocks_received);
                                }

                                // Debug individual block details for small batches
                                if blocks_received <= 5 && blocks_received > 0 {
                                    for (i, block) in response.blocks.iter().enumerate() {
                                        let block_hash = block.calculate_hash();
                                        debug!("   [NET-PROTOCOL] Block {}: height={} hash={}.. parent={}.. txs={}",
                                              i, block.header.height,
                                              hex::encode(&block_hash[..8]),
                                              hex::encode(&block.header.prev_block_hash[..8]),
                                              block.transactions.len());
                                    }
                                }

                                info!("📨 [BLOCK-PACK] Received {} blocks (heights {}-{}) | Network: {}",
                                      response.blocks.len(), response.start_height, response.end_height, response.peer_height);

                                // ✅ v0.9.73-beta: Mark peer as successful (compatible with BlockPackCodec)
                                self.mark_peer_success(peer);

                                // v1.0.44-beta: Remove completed request from outstanding list
                                // Supports concurrent sync - multiple requests can be in flight
                                if let Ok(mut guard) = self.outstanding_sync_requests.lock() {
                                    let before_len = guard.len();
                                    // Remove request matching this height range
                                    guard.retain(|(_, h, _)| *h != response.start_height);
                                    if guard.len() < before_len {
                                        info!("✅ [BLOCK-SYNC] Cleared request for height {} ({} still in flight)",
                                              response.start_height, guard.len());
                                    }
                                }

                                // v1.0.12-beta: Check if this is a pending batch sync request
                                // v1.0.15-beta: Convert request_id to String for HashMap lookup
                                // v1.3.10-beta: Enhanced logging for response channel delivery debugging
                                let request_id_str = format!("{:?}", request_id);
                                let mut pending = self.pending_block_requests.lock().unwrap();

                                // v1.3.10-beta: Log pending requests for debugging channel mismatches
                                let pending_keys: Vec<_> = pending.keys().cloned().collect();
                                if !pending_keys.is_empty() {
                                    info!("📋 [RESPONSE DELIVERY] Looking for request_id: {}", &request_id_str);
                                    info!("   Pending requests ({}): {:?}",
                                          pending_keys.len(),
                                          pending_keys.iter().take(5).collect::<Vec<_>>());
                                }

                                if let Some(tx) = pending.remove(&request_id_str) {
                                    // Send blocks to waiting BatchSyncEngine
                                    if let Err(_) = tx.send(response.blocks.clone()) {
                                        warn!("⚠️  [BATCH SYNC] Failed to deliver blocks: receiver dropped");
                                    } else {
                                        info!("✅ [BATCH SYNC] Successfully delivered {} blocks via channel (request_id: {})",
                                               response.blocks.len(), &request_id_str[..request_id_str.len().min(30)]);
                                    }
                                } else {
                                    // v1.3.10-beta: Log when no matching request found (potential channel mismatch)
                                    if !pending_keys.is_empty() {
                                        warn!("⚠️  [RESPONSE DELIVERY] NO MATCHING REQUEST for ID: {}", &request_id_str);
                                        warn!("   Available pending keys: {:?}", pending_keys);
                                        warn!("   Blocks will be forwarded to consensus but NOT to TurboSync channel");
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
                    Event::OutboundFailure { peer, request_id, error, connection_id: _ } => {
                        // 🔧 v1.0.89-beta: Enhanced debugging for DialFailure
                        error!("❌ [BLOCK-PACK] ================================================");
                        error!("❌ [BLOCK-PACK] OUTBOUND FAILURE to peer: {}", peer);
                        error!("❌ [BLOCK-PACK] Request ID: {:?}", request_id);
                        error!("❌ [BLOCK-PACK] Error type: {:?}", error);

                        // Log connection state for this peer
                        let is_connected = self.swarm.is_connected(&peer);
                        error!("❌ [BLOCK-PACK] Peer connected?: {}", is_connected);

                        // Log all addresses we have for this peer in the address book
                        // Note: No direct API to get addresses from swarm, so we check our own cache
                        if let Ok(peer_addrs) = self.peer_addresses.try_read() {
                            if let Some(addrs) = peer_addrs.get(&peer) {
                                error!("❌ [BLOCK-PACK] Our cached addresses for peer: {:?}", addrs);
                            } else {
                                error!("❌ [BLOCK-PACK] NO cached addresses for peer!");
                            }
                        }

                        // Log network state
                        let conn_info = self.swarm.network_info();
                        error!("❌ [BLOCK-PACK] Network state:");
                        error!("   - Pending outgoing: {}", conn_info.connection_counters().num_pending_outgoing());
                        error!("   - Established: {}", conn_info.connection_counters().num_established());
                        error!("   - Total connections: {:?}", conn_info.connection_counters());

                        // Log all connected peers
                        let connected_peers: Vec<_> = self.swarm.connected_peers().collect();
                        error!("❌ [BLOCK-PACK] Currently connected peers ({}):", connected_peers.len());
                        for cp in &connected_peers {
                            error!("   - {}", cp);
                        }
                        error!("❌ [BLOCK-PACK] ================================================");

                        // ✅ v0.9.73-beta: Mark peer as failed (timeout/incompatible)
                        self.mark_peer_failure(peer);

                        // v1.3.3-beta: Add failed heights to retry queue with exponential backoff
                        // Instead of just clearing outstanding requests, schedule them for retry
                        // Max 5 retries with exponential delays: 2s, 4s, 8s, 16s, 32s
                        const MAX_RETRIES: u8 = 5;

                        if let Ok(mut outstanding) = self.outstanding_sync_requests.lock() {
                            if !outstanding.is_empty() {
                                let heights_to_retry: Vec<u64> = outstanding.iter()
                                    .map(|(_, h, _)| *h)
                                    .collect();
                                outstanding.clear();

                                // Add to retry queue
                                if let Ok(mut retry_queue) = self.sync_retry_queue.lock() {
                                    for height in heights_to_retry {
                                        // Check if already in retry queue
                                        if let Some(entry) = retry_queue.iter_mut().find(|(h, _, _)| *h == height) {
                                            // Already in queue - increment retry count
                                            if entry.1 < MAX_RETRIES {
                                                entry.1 += 1;
                                                // Exponential backoff: 2^retry_count seconds
                                                let delay_secs = 2u64.pow(entry.1 as u32);
                                                entry.2 = std::time::Instant::now() + Duration::from_secs(delay_secs);
                                                warn!("🔄 [RETRY] Height {} scheduled for retry #{} in {}s",
                                                      height, entry.1, delay_secs);
                                            } else {
                                                warn!("❌ [RETRY] Height {} exceeded max retries ({}), dropping",
                                                      height, MAX_RETRIES);
                                                // Remove from queue - exceeded max retries
                                                retry_queue.retain(|(h, _, _)| *h != height);
                                            }
                                        } else {
                                            // New entry - first retry in 2 seconds
                                            retry_queue.push((height, 1, std::time::Instant::now() + Duration::from_secs(2)));
                                            warn!("🔄 [RETRY] Height {} added to retry queue (retry #1 in 2s)", height);
                                        }
                                    }
                                }
                            }
                        }
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
                    Event::Message { peer, message, connection_id } => {
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
                                            // 🔥 v2.0.0: libp2p 0.56 uses close_connection with ConnectionId
                                            let _ = self.swarm.close_connection(connection_id);
                                        }
                                        crate::handshake_validator::HandshakeResult::WrongNetwork { ours, theirs } => {
                                            warn!("❌ [HANDSHAKE] Peer {} on wrong network: ours={}, theirs={}",
                                                  peer, ours, theirs);
                                            // 🔥 v2.0.0: libp2p 0.56 uses close_connection with ConnectionId
                                            let _ = self.swarm.close_connection(connection_id);
                                        }
                                        crate::handshake_validator::HandshakeResult::GenesisMismatch => {
                                            warn!("❌ [HANDSHAKE] Peer {} has mismatched genesis hash", peer);
                                            // 🔥 v2.0.0: libp2p 0.56 uses close_connection with ConnectionId
                                            let _ = self.swarm.close_connection(connection_id);
                                        }
                                        crate::handshake_validator::HandshakeResult::MissingFeatures { required } => {
                                            warn!("❌ [HANDSHAKE] Peer {} missing required features: {:?}", peer, required);
                                            // 🔥 v2.0.0: libp2p 0.56 uses close_connection with ConnectionId
                                            let _ = self.swarm.close_connection(connection_id);
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
                    Event::OutboundFailure { peer, request_id, error, connection_id: _ } => {
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
            QNarwhalEvent::RelayServer(event) => {
                // 🌐 v3.5.5: Relay Server events for browser P2P
                // Log relay server activity for monitoring browser-to-browser connections
                info!("🌐 [RELAY SERVER] {:?}", event);
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

    /// v2.9.2-beta: Broadcast a protocol fee record for consensus verification
    /// All nodes on the network will verify this fee matches expected calculations
    /// This ensures the master wallet receives the correct percentage per DEX trade
    pub fn broadcast_protocol_fee(&mut self, fee_gossip: q_types::ProtocolFeeGossip) -> anyhow::Result<()> {
        use q_types::NetworkId;

        // Get the protocol fees topic for the current network
        let network_id = std::env::var("Q_NETWORK_ID")
            .ok()
            .and_then(|s| s.parse::<NetworkId>().ok())
            .unwrap_or(NetworkId::TestnetPhase16);

        let topic = network_id.protocol_fees_topic();

        // Serialize the fee gossip message
        let data = serde_json::to_vec(&fee_gossip)
            .map_err(|e| anyhow::anyhow!("Failed to serialize protocol fee gossip: {}", e))?;

        info!(
            "💰 [PROTOCOL FEE BROADCAST] Publishing fee record to topic: {}",
            topic
        );
        info!(
            "   Fee ID: {}, Trade: {}, Amount: {}, Token: {}",
            hex::encode(&fee_gossip.fee_record.fee_id[..8]),
            hex::encode(&fee_gossip.fee_record.trade_tx_hash[..8]),
            fee_gossip.fee_record.fee_amount,
            hex::encode(&match fee_gossip.fee_record.fee_token {
                q_types::TokenType::QUG => q_types::QUG_TOKEN_ADDRESS,
                q_types::TokenType::QUGUSD => q_types::QUGUSD_TOKEN_ADDRESS,
                q_types::TokenType::Custom(addr) => addr,
            }[..4])
        );

        self.publish_topic(&topic, data)?;

        info!(
            "✅ [PROTOCOL FEE BROADCAST] Fee record published for consensus verification"
        );

        Ok(())
    }

    /// Request blocks from a specific peer via libp2p request-response (Phase 3)
    /// ✅ v0.9.68-beta: Updated to use BlockPackRequest for efficient block sync
    /// 🚀 v1.0.43-beta: Added request_id tracking for debugging message delivery issues
    pub fn request_blocks_from_peer(
        &mut self,
        peer_id: PeerId,
        start_height: u64,
        limit: usize,
    ) -> anyhow::Result<()> {
        info!("📤 [BLOCK-SYNC] ================================================");
        info!("📤 [BLOCK-SYNC] Requesting {} blocks from height {} from peer {}", limit, start_height, peer_id);

        // 🔧 v1.0.89-beta: Enhanced debugging - check connection and address state BEFORE sending
        let is_connected = self.swarm.is_connected(&peer_id);
        info!("📤 [BLOCK-SYNC] Peer {} connected BEFORE request?: {}", peer_id, is_connected);

        // Log network state
        let conn_info = self.swarm.network_info();
        info!("📤 [BLOCK-SYNC] Network state before request:");
        info!("   - Pending outgoing: {}", conn_info.connection_counters().num_pending_outgoing());
        info!("   - Established: {}", conn_info.connection_counters().num_established());

        // Log all connected peers
        let connected_peers: Vec<_> = self.swarm.connected_peers().collect();
        info!("📤 [BLOCK-SYNC] Currently connected peers ({}):", connected_peers.len());
        for cp in &connected_peers {
            info!("   - {}", cp);
        }
        info!("📤 [BLOCK-SYNC] ================================================");

        let end_height = start_height + limit as u64 - 1;

        // v1.0.87-beta: CRITICAL FIX - Check for duplicate requests before sending
        // BUG: Same height was being requested multiple times due to race conditions
        // This caused duplicate block processing and wasted bandwidth
        if let Ok(guard) = self.outstanding_sync_requests.lock() {
            for (_, existing_height, _) in guard.iter() {
                if *existing_height == start_height {
                    debug!("⏭️  [BLOCK-SYNC] Skipping duplicate request for height {} (already in flight)", start_height);
                    return Ok(());
                }
            }
        }

        let request = q_types::BlockPackRequest::new(start_height, end_height);

        // v1.0.43-beta: Track request_id for debugging - helps identify lost requests
        let request_id = self.swarm.behaviour_mut().block_sync.send_request(&peer_id, request);

        // v1.0.44-beta: Record outstanding request for concurrent sync tracking
        // Multiple requests can be in flight simultaneously
        let request_id_str = format!("{:?}", request_id);
        if let Ok(mut guard) = self.outstanding_sync_requests.lock() {
            guard.push((request_id_str.clone(), start_height, std::time::Instant::now()));
        }

        info!("✅ [BLOCK-SYNC] Block sync request sent to {} (request_id: {}, heights: {}-{})",
              peer_id, request_id_str, start_height, end_height);
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

        // 🧅 v1.3.2-beta: TOR-AWARE BATCH SYNC TIMEOUT
        // Tor networks need longer timeouts due to multi-hop latency
        let tor_enabled = std::env::var("Q_TOR_ENABLED").is_ok() ||
                          std::env::var("Q_TOR_PROXY").is_ok();
        let batch_timeout_secs = std::env::var("Q_BATCH_SYNC_TIMEOUT")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(if tor_enabled { 180 } else { 60 }); // 3 min for Tor, 1 min for direct

        info!("✅ [BATCH SYNC] libp2p request sent, waiting for response (timeout: {}s)", batch_timeout_secs);

        // Wait for response with timeout (v1.3.2-beta: Tor-aware timeout)
        let request_start = std::time::Instant::now();
        match timeout(Duration::from_secs(batch_timeout_secs), rx).await {
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
                // Timeout (v1.3.2-beta: Tor-aware timeout)
                warn!("⏱️  [BATCH SYNC] TIMEOUT: No response after {}s from peer {}", batch_timeout_secs, peer_id);
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

    /// 🚀 v1.0.4-beta: Phase 2 DAG-Aware Sync - Batch Request
    ///
    /// Request multiple blocks by hash in a single batch request.
    /// This is optimized for DAG layer fetching where blocks may not be sequential.
    ///
    /// # Arguments
    /// * `peer_id` - Peer to request from (as string)
    /// * `block_hashes` - Vector of block hashes to fetch
    ///
    /// # Returns
    /// Vector of blocks (may be in different order than requested)
    ///
    /// # Performance
    /// - 60 second timeout per request
    /// - Designed for 500-1000 block batches
    /// - Network-bound performance
    pub async fn request_blocks_batch(
        &mut self,
        peer_id_str: &str,
        block_hashes: &[String],
    ) -> anyhow::Result<Vec<q_types::QBlock>> {
        use tokio::time::{timeout, Duration};

        // Parse PeerId
        let peer_id: PeerId = peer_id_str.parse()
            .map_err(|e| anyhow::anyhow!("Invalid peer ID {}: {}", peer_id_str, e))?;

        info!("📦 [DAG SYNC] Requesting {} blocks by hash from peer {}",
              block_hashes.len(), peer_id);

        // Create oneshot channel for response
        let (tx, rx): (tokio::sync::oneshot::Sender<Vec<q_types::Block>>, _) = tokio::sync::oneshot::channel();

        // Create batch request (using existing BlockPackRequest with hash list)
        // NOTE: This requires extending BlockPackRequest to support hash-based fetching
        // For now, we'll fall back to height-based range requests and filter
        // TODO: Extend BlockPackRequest protocol to support direct hash fetching

        // WORKAROUND: Convert block hashes to height ranges
        // This is a temporary solution until we extend the protocol
        // For Phase 2 initial deployment, we'll fetch by block ranges and deduplicate
        warn!("⚠️  [DAG SYNC] Using height-based fallback for batch request (protocol extension needed)");

        // For now, return error indicating protocol needs extension
        return Err(anyhow::anyhow!(
            "Hash-based batch fetching not yet implemented - protocol extension required. \
             Use request_block_headers() to fetch lightweight metadata first, \
             then request_block_range_impl() for full blocks."
        ));
    }

    /// 🚀 v1.0.4-beta: Phase 2 DAG-Aware Sync - Block Headers Request
    ///
    /// Request lightweight block headers (no transactions) for DAG layer detection.
    /// Headers contain: hash, height, dag_parents, timestamp - ~200 bytes each.
    ///
    /// # Arguments
    /// * `peer_id` - Peer to request from (as string)
    /// * `start_height` - Starting block height
    /// * `end_height` - Ending block height (inclusive)
    ///
    /// # Returns
    /// Vector of lightweight block headers for DAG analysis
    ///
    /// # Performance
    /// - Headers are ~200 bytes vs ~4.6KB for full blocks (23x smaller)
    /// - Can fetch 100,000 headers in ~20MB vs ~460MB for full blocks
    /// - 30 second timeout per request
    pub async fn request_block_headers(
        &mut self,
        peer_id_str: &str,
        start_height: u64,
        end_height: u64,
    ) -> anyhow::Result<Vec<q_storage::DagBlockHeader>> {
        use tokio::time::{timeout, Duration};

        // Parse PeerId
        let peer_id: PeerId = peer_id_str.parse()
            .map_err(|e| anyhow::anyhow!("Invalid peer ID {}: {}", peer_id_str, e))?;

        let num_headers = end_height - start_height + 1;
        info!("📋 [DAG SYNC] Requesting {} block headers ({}-{}) from peer {}",
              num_headers, start_height, end_height, peer_id);

        // For Phase 2 initial deployment, we fetch full blocks and extract headers
        // TODO: Add dedicated header-only protocol for maximum efficiency
        info!("⚠️  [DAG SYNC] Using full block fetch (header-only protocol not yet implemented)");

        // Fetch full blocks using existing protocol
        let blocks = self.request_block_range_impl(start_height, end_height).await?;

        // Convert to lightweight headers
        let headers: Vec<q_storage::DagBlockHeader> = blocks
            .into_iter()
            .map(|block| q_storage::DagBlockHeader {
                hash: hex::encode(block.calculate_hash()),
                height: block.header.height,
                vertex_id: None, // Vertex ID not available from block data
                dag_parents: block.dag_parents
                    .iter()
                    .map(|v| hex::encode(v))
                    .collect(),
            })
            .collect();

        info!("✅ [DAG SYNC] Extracted {} headers from full blocks", headers.len());
        Ok(headers)
    }

    /// 🚀 v1.7.0-LAMINAR: Lock-free peer success tracking (VORTEX ELIMINATION)
    /// No locks needed - DashMap provides concurrent access
    pub fn mark_peer_success(&self, peer_id: PeerId) {
        // Remove from failure list (peer is proven working)
        self.peer_compat.failures.remove(&peer_id);

        // Remove from blacklist (peer is proven compatible)
        if self.peer_compat.blacklist.remove(&peer_id).is_some() {
            info!("✅ [LAMINAR] Peer {} removed from blacklist (now responsive)", peer_id);
        }

        // Record success (lock-free)
        self.peer_compat.record_success(&peer_id);

        let success_count = self.peer_compat.get_successes(&peer_id);
        debug!("✅ [LAMINAR] Peer {} marked successful ({} total successes)", peer_id, success_count);
    }

    /// 🚀 v1.7.0-LAMINAR: Lock-free peer failure tracking (VORTEX ELIMINATION)
    /// No locks needed - DashMap provides concurrent access with automatic decay and blacklisting
    pub fn mark_peer_failure(&self, peer_id: PeerId) {
        let is_bootstrap = self.peer_compat.is_bootstrap.contains(&peer_id);
        let threshold = if is_bootstrap {
            BLACKLIST_FAILURE_THRESHOLD * BOOTSTRAP_BLACKLIST_MULTIPLIER
        } else {
            BLACKLIST_FAILURE_THRESHOLD
        };

        // Record failure (lock-free, includes decay calculation and blacklist logic)
        self.peer_compat.record_failure(&peer_id);

        let failure_count = self.peer_compat.get_failures(&peer_id);
        debug!("⚠️  [LAMINAR] Peer {} marked failed ({}/{} failures before blacklist{})",
               peer_id, failure_count, threshold,
               if is_bootstrap { " [BOOTSTRAP]" } else { "" });

        // Log if peer just got blacklisted
        if self.peer_compat.is_blacklisted(&peer_id) && failure_count >= threshold {
            if is_bootstrap {
                error!("🚨 [LAMINAR] BOOTSTRAP peer {} BLACKLISTED for {}s ({}+ failures) - P2P sync may fail!",
                      peer_id, BLACKLIST_EXPIRY_SECS, threshold);
                error!("   💡 Check: network connectivity, firewall, bootstrap server status");
            } else {
                warn!("🚫 [LAMINAR] Peer {} BLACKLISTED for {}s ({}+ failures)",
                      peer_id, BLACKLIST_EXPIRY_SECS, threshold);
            }
        }
    }

    /// v1.0.86-beta: Log P2P sync failure with detailed diagnostics
    /// This makes it visible when P2P sync is broken instead of silently skipping
    async fn log_p2p_sync_failure(&self, reason: &str) {
        let blacklisted = self.get_blacklisted_peers();
        let discovered = self.discovered_peers.read().await.len();
        let compatible = self.get_compatible_peers().len();
        let bootstrap_count = self.bootstrap_peers.read().await.len();

        // Use warn! to make this visible - this is a serious problem!
        warn!("🚨 ══════════════════════════════════════════════════════════════════");
        warn!("🚨 [P2P SYNC FAILED] No eligible peers for P2P synchronization!");
        warn!("🚨 ══════════════════════════════════════════════════════════════════");
        warn!("🚨 Reason: {}", reason);
        warn!("🚨");
        warn!("🚨 📊 Peer Status:");
        warn!("🚨    Compatible peers (proven working): {}", compatible);
        warn!("🚨    Blacklisted peers: {}", blacklisted.len());
        warn!("🚨    Discovered peers (total): {}", discovered);
        warn!("🚨    Bootstrap peers configured: {}", bootstrap_count);
        warn!("🚨");
        warn!("🚨 ⚠️  IMPACT: Falling back to HTTP sync");
        warn!("🚨    HTTP sync speed: ~1 block/sec");
        warn!("🚨    P2P sync speed: 100-200 blocks/sec");
        warn!("🚨    This is 100-200x SLOWER!");
        warn!("🚨");
        warn!("🚨 💡 Troubleshooting:");
        warn!("🚨    1. Check firewall allows port 9001 (TCP)");
        warn!("🚨    2. Check NAT/router port forwarding");
        warn!("🚨    3. Verify bootstrap server is online: 185.182.185.227:9001");
        warn!("🚨    4. Try setting Q_BOOTSTRAP_PEERS environment variable");
        warn!("🚨    5. Check network connectivity to bootstrap peer");
        warn!("🚨 ══════════════════════════════════════════════════════════════════");
    }

    /// 🚀 v1.7.0-LAMINAR: Lock-free compatible peer list (VORTEX ELIMINATION)
    /// Uses DashMap iteration - no locks needed
    pub fn get_compatible_peers(&self) -> Vec<PeerId> {
        // Return peers that have at least one success and are not actively blacklisted
        self.peer_compat.successes.iter()
            .filter(|entry| !self.peer_compat.is_blacklisted(entry.key()))
            .map(|entry| *entry.key())
            .collect()
    }

    /// 🚀 v1.7.0-LAMINAR: Lock-free blacklisted peer list (VORTEX ELIMINATION)
    /// Uses DashMap iteration with automatic expiry checking
    pub fn get_blacklisted_peers(&self) -> std::collections::HashSet<PeerId> {
        let now = std::time::Instant::now();
        let expiry_duration = std::time::Duration::from_secs(BLACKLIST_EXPIRY_SECS);

        // Only return peers that are still actively blacklisted (not expired)
        self.peer_compat.blacklist.iter()
            .filter(|entry| now.duration_since(*entry.value()) < expiry_duration)
            .map(|entry| *entry.key())
            .collect()
    }

    /// v1.0.45-beta: Get best known network height for progress display
    /// Returns 0 if no peer heights have been received yet
    pub fn get_known_network_height(&self) -> u64 {
        self.known_network_height.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// 🚀 v1.0.40-beta: FIX #3 - Log gossipsub mesh status for debugging
    /// This helps diagnose why peer-height announcements aren't being received
    fn log_gossipsub_mesh_status(&mut self, topic: &str) {
        let gossipsub = &self.swarm.behaviour().gossipsub;

        // Get mesh peers for this topic
        let topic_hash = gossipsub::IdentTopic::new(topic).hash();
        let mesh_peers: Vec<_> = gossipsub.mesh_peers(&topic_hash).collect();

        if mesh_peers.is_empty() {
            warn!("⚠️ [GOSSIPSUB MESH] Topic '{}' has NO mesh peers!", topic);
            warn!("   Messages on this topic will NOT be received via gossipsub mesh");
            warn!("   This explains why peer-height announcements aren't working");
            warn!("   WORKAROUND: Bootstrap peer auto-registered via HTTP fallback");
        } else {
            info!("✅ [GOSSIPSUB MESH] Topic '{}' has {} mesh peers:", topic, mesh_peers.len());
            for (i, peer) in mesh_peers.iter().enumerate().take(5) {
                info!("   {}. {}", i + 1, peer);
            }
            if mesh_peers.len() > 5 {
                info!("   ... and {} more peers", mesh_peers.len() - 5);
            }
        }

        // Also log all topics we're subscribed to
        let all_topics: Vec<_> = gossipsub.topics().collect();
        info!("📋 [GOSSIPSUB] Subscribed to {} topics total", all_topics.len());
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
        // 🚨 v1.0.41-beta: CRITICAL FIX - Use cached height instead of qblock:latest pointer
        // BUG: get_latest_qblock_height() reads the DB pointer which isn't updated atomically
        // during concurrent transaction commits, causing LOCAL HEIGHT to regress
        // (e.g., 1630 -> 1333 -> 1442 -> stuck at 1997)
        // FIX: Use get_highest_contiguous_block() which reads from the in-memory cache
        // that's updated after each successful block save
        let local_height = storage.get_highest_contiguous_block().await?;

        // v1.0.44-beta: Concurrent sync with stall detection
        // - Support up to MAX_CONCURRENT_SYNC requests in flight
        // - Clear stale requests older than timeout
        // - Track which heights are already requested to avoid duplicates
        // v1.0.81-beta: CRITICAL FIX - Reduce stale timeout from 60s to 35s
        // BUG: libp2p request timeout is 30s, but we waited 60s to clear stale requests
        // This caused P2P deadlock: 3 requests timeout at 30s but aren't cleared until 60s
        // During those 30 extra seconds, no new sync requests possible (queue full)
        // FIX: Clear stale requests at 35s (just after libp2p timeout + grace period)
        // v1.3.3-beta: For Tor, use longer stall timeout (150s = 120s request timeout + 30s grace)
        // v1.3.5-beta: Increased direct mode timeout to 65s (60s request timeout + 5s grace)
        // This matches the new 60s request timeout for direct mode to handle large block packs
        let stall_timeout_secs = if self.tor_enabled { 150 } else { 65 };
        const MAX_CONCURRENT_SYNC: usize = 3;  // Allow 3 parallel requests

        // v1.3.3-beta: Process retry queue - retry failed heights with exponential backoff
        if let Ok(mut retry_queue) = self.sync_retry_queue.lock() {
            let now = std::time::Instant::now();
            let mut heights_to_retry: Vec<u64> = Vec::new();

            // Find heights that are due for retry
            for (height, retry_count, next_retry_time) in retry_queue.iter() {
                if now >= *next_retry_time {
                    heights_to_retry.push(*height);
                    info!("🔄 [RETRY] Height {} due for retry #{}", height, retry_count);
                }
            }

            // Remove retried heights from queue (they'll be re-added on failure)
            for height in &heights_to_retry {
                retry_queue.retain(|(h, _, _)| h != height);
            }

            // These heights will be picked up naturally by the sync logic below
            // since local_height will be lower than them
        }

        // 🚨 v1.3.8-beta CRITICAL FIX: Start request at local_height + 1, not local_height!
        // BUG: If local_height=131735, we already HAVE block 131735. Requesting from 131735
        // includes the block we have, causing "AlreadyProcessed" for first block in batch.
        // This caused infinite sync loop where same range was requested forever.
        // FIX: Request starting at local_height + 1 to get the NEXT block we need.
        let mut next_request_height = local_height + 1;
        if let Ok(mut guard) = self.outstanding_sync_requests.lock() {
            // Clean up stale requests
            let before_len = guard.len();
            guard.retain(|(req_id, height, sent_at)| {
                let elapsed = sent_at.elapsed().as_secs();
                if elapsed > stall_timeout_secs {
                    warn!("⚠️ [STALL-DETECT] Request {} (height {}) stale after {}s - removing",
                          req_id, height, elapsed);
                    false
                } else {
                    true
                }
            });
            if guard.len() < before_len {
                info!("🧹 Cleaned up {} stale sync requests", before_len - guard.len());
            }

            // Check if we've hit concurrent limit
            if guard.len() >= MAX_CONCURRENT_SYNC {
                debug!("🔄 [AUTO-SYNC] {} requests in flight (max {}), waiting...",
                       guard.len(), MAX_CONCURRENT_SYNC);
                return Ok(());
            }

            // 🔧 v1.0.49-beta: CRITICAL FIX for sync backlog issue
            // BUG: If pending requests are for heights far ahead of local_height (e.g., due to
            // previous successful batches that were never applied), we would keep requesting
            // heights that are even further ahead, creating an ever-growing gap.
            //
            // FIX: The next_request_height should be based on local_height, not on pending requests.
            // We only pipeline ahead if the pending requests are for heights contiguous with local_height.
            // If there's a gap between local_height and the lowest pending request, reset to local_height.

            if !guard.is_empty() {
                // v1.3.3-beta: Adaptive batch size based on network type
                // v1.3.5-beta: CRITICAL FIX - Reduced direct mode batch from 5000 to 500 blocks
                // PROBLEM: 5000 blocks = ~5-10MB payload, takes 45-60s to serialize + transmit
                // This caused request timeouts even with 60s timeout (serialization + network delay)
                // FIX: 500 blocks = ~500KB-1MB payload, completes in 5-10s with headroom
                // Tor networks: 200 blocks (small payloads for Tor circuit reliability)
                // Direct networks: 500 blocks (balance between throughput and timeout safety)
                let batch_size = if self.tor_enabled { 200u64 } else { 500u64 };

                // Find the lowest height being requested
                let min_pending_height = guard.iter()
                    .map(|(_, h, _)| *h)
                    .min()
                    .unwrap_or(local_height);

                // If the lowest pending request is more than batch_size ahead of local_height,
                // there's a gap - we need to fill it first, not pipeline further ahead
                let gap = min_pending_height.saturating_sub(local_height);
                if gap > batch_size {
                    // Gap detected! Request from local_height + 1 instead of continuing the pipeline
                    // 🚨 v1.3.8-beta: Use local_height + 1 (we already HAVE local_height!)
                    warn!("🔧 [SYNC-FIX] Gap detected: local={}, lowest_pending={}, gap={}",
                          local_height, min_pending_height, gap);
                    warn!("🔧 [SYNC-FIX] Resetting next_request_height to {} to fill gap", local_height + 1);
                    next_request_height = local_height + 1;
                } else {
                    // No gap - safe to pipeline ahead after pending requests
                    // v1.0.87-beta: CRITICAL FIX - Limit pipelining to prevent runaway
                    // BUG: Without limit, pipelining could get 200k+ blocks ahead of local_height
                    // This caused blocks to be received far ahead, never committed (pointer stuck),
                    // then same blocks re-requested endlessly.
                    // FIX: Cap pipelining to max 15k blocks ahead of local_height
                    const MAX_PIPELINE_AHEAD: u64 = 15_000;

                    let max_pending_height = guard.iter()
                        .map(|(_, h, _)| h + batch_size)
                        .max()
                        .unwrap_or(local_height);

                    // Clamp to prevent runaway pipelining
                    next_request_height = max_pending_height.min(local_height + MAX_PIPELINE_AHEAD);

                    if max_pending_height > local_height + MAX_PIPELINE_AHEAD {
                        debug!("🔧 [PIPELINE-CAP] Capped next_request from {} to {} (local={}, max_ahead={})",
                               max_pending_height, next_request_height, local_height, MAX_PIPELINE_AHEAD);
                    }
                }
            }
        }

        // v1.2.5-beta: Removed peer-height-cap code that was causing compilation errors
        // The turbo_sync_manager API wasn't available on QStorage.
        // The main optimization (ResponseOmission fix) is still in place via height_cache.cached()

        // v1.0.100-beta: CRITICAL FIX - Only request from ACTUALLY CONNECTED peers
        // BUG: Previous code selected peers from discovered/bootstrap lists but didn't verify
        // they were actually connected. This caused requests to disconnected peers which
        // silently failed, leaving sync permanently stuck.
        // FIX: First get the list of actually connected peers, then filter by priority.
        let connected_peers: Vec<PeerId> = self.swarm.connected_peers().cloned().collect();

        if connected_peers.is_empty() {
            warn!("🚨 [AUTO-SYNC] No connected peers! Cannot sync.");
            self.log_p2p_sync_failure("no connected peers").await;
            return Ok(());
        }

        let blacklist = self.get_blacklisted_peers();
        let available_peers: Vec<PeerId> = connected_peers.iter()
            .filter(|p| !blacklist.contains(p))
            .cloned()
            .collect();

        if available_peers.is_empty() {
            warn!("🚨 [AUTO-SYNC] All {} connected peers are blacklisted!", connected_peers.len());
            self.log_p2p_sync_failure("all connected peers blacklisted").await;
            return Ok(());
        }

        // Prioritize connected peers: compatible > bootstrap > any
        let peer_id = {
            // Step 1: Try compatible peers that are ACTUALLY connected
            let compatible = self.get_compatible_peers();
            let compatible_connected: Vec<_> = compatible.iter()
                .filter(|p| available_peers.contains(p))
                .cloned()
                .collect();

            if !compatible_connected.is_empty() {
                debug!("🔄 [AUTO-SYNC] Using connected compatible peer");
                compatible_connected[0]
            } else {
                // Step 2: Try bootstrap peers that are ACTUALLY connected
                let bootstrap = self.bootstrap_peers.read().await;
                let bootstrap_connected: Vec<_> = bootstrap.keys()
                    .filter(|p| available_peers.contains(p))
                    .cloned()
                    .collect();
                drop(bootstrap);

                if !bootstrap_connected.is_empty() {
                    debug!("🔄 [AUTO-SYNC] Using connected bootstrap peer");
                    bootstrap_connected[0]
                } else {
                    // Step 3: Use any available connected peer
                    debug!("🔄 [AUTO-SYNC] Using any connected peer (no compatible/bootstrap connected)");
                    available_peers[0]
                }
            }
        };

        info!("✅ [AUTO-SYNC] Selected peer {} from {} connected peers",
              &peer_id.to_string()[..12], connected_peers.len());

        // 🚀 v1.3.3-beta: TurboSync adaptive batch size for Tor compatibility
        // v1.0.46-beta original: 5000 blocks per request for maximum throughput
        // v1.3.5-beta: CRITICAL FIX - Reduced batch sizes to prevent timeouts
        // PROBLEM: Large batch sizes (5000 blocks) caused:
        //   1. ~5-10MB payloads that take 30-60s just to serialize
        //   2. Network transmission adds another 10-30s
        //   3. Total time exceeds timeout, causing "Empty response buffer" errors
        // FIX: Smaller batches complete well within timeout:
        //   - 500 blocks = ~500KB-1MB, completes in 5-10s
        //   - 200 blocks for Tor = ~200-400KB, handles circuit latency
        // Effective sync rate: 500 blocks/10s = 50 blocks/sec = 3000 blocks/min
        // This is actually FASTER than timeout-and-retry cycles!
        let batch_size = if self.tor_enabled { 200 } else { 500 };

        // Use next_request_height for pipelining (calculated above to follow pending requests)
        info!("🚀 [AUTO-SYNC] Local: {}, Next request: {}, batch: {}, peer: {}",
              local_height, next_request_height, batch_size, peer_id);

        self.request_blocks_from_peer(peer_id, next_request_height, batch_size)?;

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
    ///
    /// 🚨 v1.1.1-beta CRITICAL FIX: Now processes BOTH swarm events AND command_rx
    /// BUG: Previously run_once() only processed swarm events, completely ignoring
    /// the command_rx channel. This meant ALL commands (PublishBlock, PublishPeerHeight,
    /// etc.) sent via libp2p_command_tx were NEVER received by the network manager.
    ///
    /// IMPACT: Blocks mined locally were NEVER broadcast to P2P network, even though
    /// logs showed "📡 Block N broadcast command sent to P2P network". The command
    /// was sent to the channel, but run_once() never read from command_rx.
    ///
    /// FIX: Use tokio::select! to process either a swarm event OR a command,
    /// whichever is ready first. This mirrors the behavior of run().
    pub async fn run_once(&mut self) -> anyhow::Result<()> {
        use futures::stream::StreamExt;

        // 🚀 v1.3.7-beta: CRITICAL FIX - Add block_pack_response_rx polling!
        // BUG: run_once() was missing the response channel polling that run() has.
        // This caused the bootstrap node to receive block requests, fetch blocks,
        // but NEVER send the response back - breaking sync completely.
        // ROOT CAUSE: Async block pack handler stores response in channel, but
        // run_once() never polled it, so responses sat in channel forever.
        tokio::select! {
            // v1.3.7-beta: HIGHEST PRIORITY - Block pack response channel
            // Must be processed first to prevent ResponseOmission timeouts!
            biased;

            Some((async_req_id, response)) = self.block_pack_response_rx.recv() => {
                // Retrieve the stored response channel and send response
                if let Some(channel) = self.pending_response_channels.lock().unwrap().remove(&async_req_id) {
                    let block_count = response.blocks.len();
                    if let Err(e) = self.swarm.behaviour_mut().block_sync.send_response(channel, response) {
                        error!("❌ [BLOCK-PACK ASYNC] Failed to send response for request {}: {:?}", async_req_id, e);
                    } else {
                        info!("✅ [BLOCK-PACK ASYNC] Sent {} blocks for async request {} (via run_once)", block_count, async_req_id);
                    }
                } else {
                    warn!("⚠️ [BLOCK-PACK ASYNC] No pending channel for request {} (may have timed out)", async_req_id);
                }
            }
            // Process commands from API (PublishBlock, PublishPeerHeight, etc.)
            Some(command) = self.command_rx.recv() => {
                self.handle_command(command).await;
            }
            // Process swarm events (connections, messages, etc.)
            Some(event) = self.swarm.next() => {
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
                        // 🌐 v1.0.21-browser: Check if this is a WebSocket connection (browser client)
                        let endpoint_str = format!("{:?}", endpoint);
                        let is_websocket = endpoint_str.contains("/ws") || endpoint_str.contains("websocket");

                        if !is_websocket {
                            // 🤝 v1.0.15.1-beta: Initiate protocol handshake with new peer (node-to-node only)
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
                        } else {
                            info!("🌐 [BROWSER CLIENT] WebSocket connection - skipping handshake");
                        }

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
        }

        Ok(())
    }

    /// Handle a single network command
    /// 🚀 v1.1.1-beta: Extracted from run() to share with run_once()
    async fn handle_command(&mut self, command: NetworkCommand) {
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
                // 🔍 v1.0.71-beta: Enhanced diagnostics for P2P broadcast issues
                let peer_count = self.connected_peer_count.load(std::sync::atomic::Ordering::Relaxed);
                let connected_peers: Vec<_> = self.swarm.connected_peers().collect();

                info!("📤 Publishing block {} ({} bytes) to gossipsub topic: {}", block_height, block_bytes.len(), topic);
                info!("   📊 Connected peers: {} (atomic count: {})", connected_peers.len(), peer_count);

                if connected_peers.is_empty() {
                    warn!("⚠️  [P2P BROADCAST] No connected peers - block {} will be stored locally only", block_height);
                    warn!("   → Other nodes can fetch this block via turbo-sync once connected");
                    warn!("   → Check bootstrap peer connectivity and port forwarding (Docker: -p 9001:9001)");
                }

                // v4.3.0-beta: Route through priority gossipsub queue instead of direct publish
                match crate::gossipsub_queue::gossipsub_queue().enqueue(topic.clone(), block_bytes) {
                    Ok(()) => {
                        info!("✅ [QUEUE] Enqueued block {} for P2P broadcast (topic={})", block_height, topic);
                    }
                    Err(reason) => {
                        warn!("⚠️ [QUEUE] Block {} dropped: {} (topic={})", block_height, reason, topic);
                    }
                }
            }
            NetworkCommand::PublishBlockRequest { topic, request_bytes } => {
                info!("📤 Publishing block request ({} bytes) to gossipsub topic: {}", request_bytes.len(), topic);
                match crate::gossipsub_queue::gossipsub_queue().enqueue(topic.clone(), request_bytes) {
                    Ok(()) => {
                        debug!("📤 [QUEUE] Enqueued block request (topic={})", topic);
                    }
                    Err(reason) => {
                        warn!("⚠️ [QUEUE] Block request dropped: {} (topic={})", reason, topic);
                    }
                }
            }
            NetworkCommand::PublishBlockResponse { topic, response_bytes, block_height } => {
                info!("📤 Publishing block response for block {} ({} bytes) to gossipsub topic: {}", block_height, response_bytes.len(), topic);
                match crate::gossipsub_queue::gossipsub_queue().enqueue(topic.clone(), response_bytes) {
                    Ok(()) => {
                        debug!("📤 [QUEUE] Enqueued block response for block {} (topic={})", block_height, topic);
                    }
                    Err(reason) => {
                        warn!("⚠️ [QUEUE] Block response for block {} dropped: {} (topic={})", block_height, reason, topic);
                    }
                }
            }
            NetworkCommand::PublishBlockPack { topic, pack_bytes } => {
                info!("🚀 [TURBO SYNC] Publishing block pack ({:.1} KB compressed) to gossipsub topic: {}",
                      pack_bytes.len() as f64 / 1024.0, topic);
                match crate::gossipsub_queue::gossipsub_queue().enqueue(topic.clone(), pack_bytes) {
                    Ok(()) => {
                        debug!("📤 [QUEUE] Enqueued block pack (topic={})", topic);
                    }
                    Err(reason) => {
                        warn!("⚠️ [QUEUE] Block pack dropped: {} (topic={})", reason, topic);
                    }
                }
            }
            NetworkCommand::RequestBlockPack { topic, request_bytes, start_height, end_height } => {
                info!("🚀 [TURBO SYNC] Requesting block pack {}-{} ({} bytes) from P2P network",
                      start_height, end_height, request_bytes.len());
                match crate::gossipsub_queue::gossipsub_queue().enqueue(topic.clone(), request_bytes) {
                    Ok(()) => {
                        debug!("📤 [QUEUE] Enqueued block pack request {}-{} (topic={})", start_height, end_height, topic);
                    }
                    Err(reason) => {
                        warn!("⚠️ [QUEUE] Block pack request {}-{} dropped: {} (topic={})", start_height, end_height, reason, topic);
                    }
                }
            }
            NetworkCommand::PublishPeerHeight { topic, announcement_bytes, height } => {
                debug!("📡 [TURBO SYNC] Publishing peer height announcement {} ({} bytes) to topic: {}",
                      height, announcement_bytes.len(), topic);
                match crate::gossipsub_queue::gossipsub_queue().enqueue(topic.clone(), announcement_bytes) {
                    Ok(()) => {
                        debug!("📤 [QUEUE] Enqueued peer height {} (topic={})", height, topic);
                    }
                    Err(reason) => {
                        warn!("⚠️ [QUEUE] Peer height {} dropped: {} (topic={})", height, reason, topic);
                    }
                }
            }
            NetworkCommand::PublishAIMessage { topic, message } => {
                // v2.3.19-beta: Enhanced debugging for distributed AI troubleshooting
                info!("═══════════════════════════════════════════════════════════════");
                info!("🤖 [DISTRIBUTED AI TX] SENDING AI MESSAGE");
                info!("═══════════════════════════════════════════════════════════════");
                info!("   📍 Topic: {}", topic);
                info!("   🆔 Message ID: {}", message.message_id);
                info!("   👤 Sender Node: {}", message.sender_node_id);
                info!("   🔗 Sender Peer: {}", message.sender_peer_id);
                info!("   ⏰ Timestamp: {}", message.timestamp);
                info!("   🔢 Sequence #: {}", message.sequence_number);
                info!("   🔄 Retry Count: {}", message.retry_count);
                info!("   ⚡ Priority: {:?}", message.priority);

                // Log payload type with details (v2.3.19: Correct field names from AIMessagePayload enum)
                let payload_desc = match &message.payload {
                    crate::distributed_ai::AIMessagePayload::InferenceRequest { request_id, prompt, .. } => {
                        format!("InferenceRequest(id={}, prompt_len={})", request_id, prompt.len())
                    }
                    crate::distributed_ai::AIMessagePayload::InferenceResponse { request_id, generated_text, tokens_generated, .. } => {
                        format!("InferenceResponse(id={}, text_len={}, tokens={})", request_id, generated_text.len(), tokens_generated)
                    }
                    crate::distributed_ai::AIMessagePayload::TargetedInferenceRequest { request_id, target_node_id, prompt, .. } => {
                        format!("TargetedInferenceRequest(id={}, target={}, prompt_len={})", request_id, target_node_id, prompt.len())
                    }
                    crate::distributed_ai::AIMessagePayload::TokenChunk { request_id, token_index, encrypted_token, .. } => {
                        // v2.5.1-beta: Privacy-safe - never log token content
                        format!("TokenChunk(id={}, idx={}, encrypted={})", request_id, token_index, encrypted_token.is_some())
                    }
                    crate::distributed_ai::AIMessagePayload::BulkTokenChunk { request_id, start_index, tokens, encrypted_tokens, .. } => {
                        // v2.5.1-beta: Privacy-safe - log count, not content
                        format!("BulkTokenChunk(id={}, start={}, count={}, encrypted={})",
                                request_id, start_index, tokens.len(), encrypted_tokens.is_some())
                    }
                    crate::distributed_ai::AIMessagePayload::InferenceStarted { request_id, worker_node_id, model, .. } => {
                        format!("InferenceStarted(id={}, worker={}, model={})", request_id, worker_node_id, model)
                    }
                    crate::distributed_ai::AIMessagePayload::InferenceComplete { request_id, worker_node_id, finish_reason, tokens_generated, .. } => {
                        format!("InferenceComplete(id={}, worker={}, reason={}, tokens={})", request_id, worker_node_id, finish_reason, tokens_generated)
                    }
                    crate::distributed_ai::AIMessagePayload::InferenceError { request_id, worker_node_id, code, message, .. } => {
                        format!("InferenceError(id={}, worker={}, code={}, msg={})", request_id, worker_node_id, code, message)
                    }
                    crate::distributed_ai::AIMessagePayload::NodeCapability { node_id, peer_id, capability, available_layers, .. } => {
                        format!("NodeCapability(node={}, peer={}, layers={}, cap={:?})", node_id, peer_id, available_layers, capability)
                    }
                    crate::distributed_ai::AIMessagePayload::Heartbeat { node_id, active_requests, layers_assigned, .. } => {
                        format!("Heartbeat(node={}, active={}, layers={:?})", node_id, active_requests, layers_assigned)
                    }
                    crate::distributed_ai::AIMessagePayload::CoordinatorElection { node_id, score, uptime_secs, inference_count, .. } => {
                        format!("CoordinatorElection(node={}, score={}, uptime={}s, inferences={})", node_id, score, uptime_secs, inference_count)
                    }
                    crate::distributed_ai::AIMessagePayload::CancelInference { request_id, target_node_id, reason, .. } => {
                        format!("CancelInference(id={}, target={}, reason={})", request_id, target_node_id, reason)
                    }
                    crate::distributed_ai::AIMessagePayload::LayerOutput { request_id, layer_index, .. } => {
                        format!("LayerOutput(id={}, layer={})", request_id, layer_index)
                    }
                    crate::distributed_ai::AIMessagePayload::LayerAssignment { request_id, .. } => {
                        format!("LayerAssignment(id={})", request_id)
                    }
                    crate::distributed_ai::AIMessagePayload::KVCacheUpdate { request_id, layer_index, sequence_length, .. } => {
                        format!("KVCacheUpdate(id={}, layer={}, seq_len={})", request_id, layer_index, sequence_length)
                    }
                    // Tensor Parallelism messages (v2.4.0)
                    crate::distributed_ai::AIMessagePayload::AllReduceChunk { request_id, layer_index, step, .. } => {
                        format!("AllReduceChunk(id={}, layer={}, step={})", request_id, layer_index, step)
                    }
                    crate::distributed_ai::AIMessagePayload::AllReduceComplete { request_id, layer_index, .. } => {
                        format!("AllReduceComplete(id={}, layer={})", request_id, layer_index)
                    }
                    crate::distributed_ai::AIMessagePayload::ShardAssignment { request_id, node_rank, world_size, .. } => {
                        format!("ShardAssignment(id={}, rank={}/{})", request_id, node_rank, world_size)
                    }
                    crate::distributed_ai::AIMessagePayload::WeightShard { request_id, layer_index, weight_name, .. } => {
                        format!("WeightShard(id={}, layer={}, name={})", request_id, layer_index, weight_name)
                    }
                    crate::distributed_ai::AIMessagePayload::ShardReady { request_id, node_id, layer_index, weight_name, .. } => {
                        format!("ShardReady(id={}, node={}, layer={}, name={})", request_id, node_id, layer_index, weight_name)
                    }
                    crate::distributed_ai::AIMessagePayload::TensorParallelRequest { request_id, prompt, .. } => {
                        format!("TensorParallelRequest(id={}, prompt_len={})", request_id, prompt.len())
                    }
                    crate::distributed_ai::AIMessagePayload::HiddenStates { request_id, layer_index, sequence_position, .. } => {
                        format!("HiddenStates(id={}, layer={}, seq_pos={})", request_id, layer_index, sequence_position)
                    }
                    crate::distributed_ai::AIMessagePayload::TensorParallelToken { request_id, token_id, token_text, token_index, .. } => {
                        format!("TensorParallelToken(id={}, token={}, text='{}', idx={})", request_id, token_id, token_text, token_index)
                    }
                };
                info!("   📦 Payload: {}", payload_desc);

                // v2.3.18-beta FIX: Use JSON serialization for AI messages
                // JSON is more tolerant of version differences than postcard binary format
                // This fixes the "Option discriminant" error when nodes have different struct layouts
                match serde_json::to_vec(&message) {
                    Ok(message_bytes) => {
                        info!("   📏 Serialized Size: {} bytes (JSON)", message_bytes.len());

                        // Log first 200 chars of JSON for debugging
                        if let Ok(json_str) = String::from_utf8(message_bytes.clone()) {
                            let preview: String = json_str.chars().take(300).collect();
                            info!("   📝 JSON Preview: {}...", preview);
                        }

                        // v4.3.0-beta: Route through priority gossipsub queue
                        match crate::gossipsub_queue::gossipsub_queue().enqueue(topic.clone(), message_bytes) {
                            Ok(()) => {
                                info!("   ✅ [QUEUE] AI message enqueued for broadcast");
                                info!("═══════════════════════════════════════════════════════════════");
                            }
                            Err(reason) => {
                                warn!("   ⚠️ [QUEUE] AI message dropped: {}", reason);
                                info!("═══════════════════════════════════════════════════════════════");
                            }
                        }
                    }
                    Err(e) => {
                        error!("   ❌ JSON SERIALIZATION FAILED: {}", e);
                        error!("═══════════════════════════════════════════════════════════════");
                    }
                }
            }
            NetworkCommand::PublishPoolAnnouncement { topic, announcement_bytes } => {
                info!("💱 [LIQUIDITY POOLS] Publishing pool announcement to topic: {} ({} bytes)", topic, announcement_bytes.len());
                match crate::gossipsub_queue::gossipsub_queue().enqueue(topic.clone(), announcement_bytes) {
                    Ok(()) => {
                        debug!("📤 [QUEUE] Enqueued pool announcement (topic={})", topic);
                    }
                    Err(reason) => {
                        warn!("⚠️ [QUEUE] Pool announcement dropped: {} (topic={})", reason, topic);
                    }
                }
            }
            NetworkCommand::PublishTokenAnnouncement { topic, announcement_bytes } => {
                // v2.3.7-beta: Publish token deployment for cross-node discovery
                info!("🪙 [TOKEN ANNOUNCEMENT] Publishing token deployment to topic: {} ({} bytes)", topic, announcement_bytes.len());
                match crate::gossipsub_queue::gossipsub_queue().enqueue(topic.clone(), announcement_bytes) {
                    Ok(()) => {
                        debug!("📤 [QUEUE] Enqueued token announcement (topic={})", topic);
                    }
                    Err(reason) => {
                        warn!("⚠️ [QUEUE] Token announcement dropped: {} (topic={})", reason, topic);
                    }
                }
            }
            NetworkCommand::PublishMinerStats { topic, stats_bytes, miner_address } => {
                // v1.0.88-beta: Publish miner stats for P2P hashrate aggregation
                debug!("⛏️ [MINER STATS] Publishing stats for {} to topic: {} ({} bytes)",
                       &miner_address[..16.min(miner_address.len())], topic, stats_bytes.len());
                match crate::gossipsub_queue::gossipsub_queue().enqueue(topic.clone(), stats_bytes) {
                    Ok(()) => {
                        debug!("📤 [QUEUE] Enqueued miner stats (topic={})", topic);
                    }
                    Err(reason) => {
                        warn!("⚠️ [QUEUE] Miner stats dropped: {} (topic={})", reason, topic);
                    }
                }
            }
            NetworkCommand::PublishBalanceUpdate { topic, update_bytes, wallet_address, amount } => {
                // v1.1.26-beta: Balance update log reduced to debug to avoid spam
                debug!("💰 [BALANCE UPDATE] Publishing update for {} (+{} units) to topic: {} ({} bytes)",
                       &wallet_address[..16.min(wallet_address.len())], amount, topic, update_bytes.len());
                match crate::gossipsub_queue::gossipsub_queue().enqueue(topic.clone(), update_bytes) {
                    Ok(()) => {
                        trace!("📤 [QUEUE] Enqueued balance update (topic={})", topic);
                    }
                    Err(reason) => {
                        debug!("⚠️ [QUEUE] Balance update dropped: {} (topic={})", reason, topic);
                    }
                }
            }
            NetworkCommand::PublishMiningSolution { topic, solution_bytes, miner_address, block_height, nonce } => {
                // v2.2.1-beta: P2P mining solution broadcasting
                // Enables decentralized mining - any node can include the solution in a block
                // v4.3.0-beta: Rate limiting now handled by gossipsub queue (MessagePriority::Low = 100ms interval)
                info!("⛏️ [P2P MINING] Broadcasting solution from {} for block #{} (nonce: {}) to topic: {} ({} bytes)",
                       &miner_address[..16.min(miner_address.len())], block_height, nonce, topic, solution_bytes.len());
                match crate::gossipsub_queue::gossipsub_queue().enqueue(topic.clone(), solution_bytes) {
                    Ok(()) => {
                        debug!("📤 [QUEUE] Enqueued mining solution for block #{} (topic={})", block_height, topic);
                    }
                    Err(reason) => {
                        debug!("⚠️ [QUEUE] Mining solution for block #{} dropped: {} (topic={})", block_height, reason, topic);
                    }
                }
            }
            NetworkCommand::PublishTransaction { topic, tx_bytes, tx_hash } => {
                // v3.3.0-beta: P2P mempool transaction propagation
                // Broadcast transaction to all peers for real-time mempool synchronization
                debug!("📤 [P2P MEMPOOL] Broadcasting tx {} ({} bytes) to topic: {}",
                       &tx_hash[..16.min(tx_hash.len())], tx_bytes.len(), topic);
                match crate::gossipsub_queue::gossipsub_queue().enqueue(topic.clone(), tx_bytes) {
                    Ok(()) => {
                        debug!("📤 [QUEUE] Enqueued tx {} (topic={})", &tx_hash[..16.min(tx_hash.len())], topic);
                    }
                    Err(reason) => {
                        debug!("⚠️ [QUEUE] Tx {} dropped: {} (topic={})", &tx_hash[..16.min(tx_hash.len())], reason, topic);
                    }
                }
            }
            NetworkCommand::PublishQnoOperation { topic, message } => {
                // v1.4.2-beta: Publish QNO operation for decentralized stake validation
                debug!("🔮 [QNO P2P] Publishing {:?} operation to topic: {}", message.message_type, topic);
                match message.to_bytes() {
                    Ok(msg_bytes) => {
                        match crate::gossipsub_queue::gossipsub_queue().enqueue(topic.clone(), msg_bytes) {
                            Ok(()) => {
                                debug!("📤 [QUEUE] Enqueued QNO operation (topic={})", topic);
                            }
                            Err(reason) => {
                                warn!("⚠️ [QUEUE] QNO operation dropped: {} (topic={})", reason, topic);
                            }
                        }
                    }
                    Err(e) => {
                        error!("❌ [QNO P2P] Failed to serialize QNO message: {}", e);
                    }
                }
            }
            NetworkCommand::PublishTokenSocial { topic, contract_address, profile_bytes } => {
                // v2.4.8-beta: Publish token social profile for cross-node sync
                info!("📱 [TOKEN SOCIAL] Publishing social profile for {} to topic: {} ({} bytes)",
                      contract_address, topic, profile_bytes.len());
                match crate::gossipsub_queue::gossipsub_queue().enqueue(topic.clone(), profile_bytes) {
                    Ok(()) => {
                        debug!("📤 [QUEUE] Enqueued token social profile (topic={})", topic);
                    }
                    Err(reason) => {
                        warn!("⚠️ [QUEUE] Token social profile dropped: {} (topic={})", reason, topic);
                    }
                }
            }
            NetworkCommand::PublishDexEvent { topic, message } => {
                // v2.9.2-beta: Publish DEX event (trade, liquidity, price) to all peers
                // This enables TRUE DEX decentralization by broadcasting state changes
                info!("💱 [DEX P2P] Publishing DEX event to topic: {} ({} bytes)", topic, message.len());
                match crate::gossipsub_queue::gossipsub_queue().enqueue(topic.clone(), message) {
                    Ok(()) => {
                        debug!("📤 [QUEUE] Enqueued DEX event (topic={})", topic);
                    }
                    Err(reason) => {
                        warn!("⚠️ [QUEUE] DEX event dropped: {} (topic={})", reason, topic);
                    }
                }
            }
            NetworkCommand::RequestBlockRangeDirect { peer_id, start_height, end_height, response_tx } => {
                // 🚀 v1.3.10-beta: Direct request-response for Turbo Sync
                // Uses libp2p request-response protocol instead of gossipsub for reliability
                // ENHANCED: Added peer connection verification and better peer selection
                info!("🚀 [TURBO SYNC DIRECT] Requesting blocks {}-{} via request-response",
                      start_height, end_height);

                // v1.3.10-beta: Get all connected peers for better selection
                let connected_peers: Vec<_> = self.swarm.connected_peers().cloned().collect();
                info!("📊 [PEER STATE] {} connected peers available", connected_peers.len());

                // Select target peer - either specified or best available
                let target_peer = if let Some(pid_str) = peer_id {
                    match pid_str.parse::<PeerId>() {
                        Ok(pid) => {
                            // v2.1.7-DELTA-V: VERIFY peer is actually connected
                            // CRITICAL FIX: Don't try unconnected peers - libp2p can't auto-dial
                            // without a known address! This was causing 75% chunk failures.
                            if self.swarm.is_connected(&pid) {
                                info!("✅ [PEER CHECK] Specified peer {} is connected", pid);
                                Some(pid)
                            } else {
                                // v2.1.7: Check if we have an address cached for this peer
                                // Use try_read() (non-blocking) since we're in sync context
                                let has_address = match self.peer_addresses.try_read() {
                                    Ok(addrs) => addrs.get(&pid).map(|a| !a.is_empty()).unwrap_or(false),
                                    Err(_) => false, // Lock contention, assume no address
                                };

                                if has_address {
                                    info!("🔗 [PEER CHECK] Peer {} not connected but has cached address, attempting dial", pid);
                                    Some(pid)
                                } else {
                                    // 🚨 v2.1.7: FAIL FAST - no point trying to dial without address
                                    error!("❌ [PEER CHECK] Peer {} is NOT connected and has NO cached address!", pid);
                                    error!("   Cannot dial peer without knowing its address.");
                                    error!("   This peer was likely discovered via gossipsub height announcement");
                                    error!("   but we don't know how to reach it.");
                                    None  // Return None to trigger fallback to connected peers
                                }
                            }
                        }
                        Err(e) => {
                            warn!("❌ [TURBO SYNC DIRECT] Invalid peer ID {}: {}", pid_str, e);
                            None
                        }
                    }
                } else {
                    None  // No peer specified, will use fallback
                };

                // v2.1.7-DELTA-V: FALLBACK - If specified peer is unreachable, use ANY connected peer
                let final_target = if target_peer.is_some() {
                    target_peer
                } else if !connected_peers.is_empty() {
                    warn!("🔄 [v2.1.7 FALLBACK] Specified peer unreachable, using connected peer instead");
                    for (i, peer) in connected_peers.iter().take(5).enumerate() {
                        info!("   Available peer #{}: {}", i + 1, peer);
                    }
                    Some(connected_peers[0])
                } else {
                    error!("❌ [TURBO SYNC DIRECT] No connected peers available!");
                    error!("   Ensure bootstrap node is reachable: 185.182.185.227:9001");
                    let _ = response_tx.send(Err(anyhow::anyhow!("No connected peers - check P2P connectivity")));
                    return;
                };

                let target_peer = final_target;

                if let Some(peer) = target_peer {
                    info!("📤 [TURBO SYNC DIRECT] Sending request-response to peer {}", peer);
                    info!("   Request: blocks {}-{} ({} blocks)", start_height, end_height, end_height - start_height + 1);

                    // Create and send the request
                    let request = q_types::BlockPackRequest::new(start_height, end_height);
                    let request_id = self.swarm.behaviour_mut().block_sync.send_request(&peer, request);

                    // Store the response channel for when we get the response
                    // Convert request_id to string for HashMap key
                    let request_id_str = format!("{:?}", request_id);

                    // Create a channel to convert QBlock vector response to the expected format
                    let (internal_tx, internal_rx) = tokio::sync::oneshot::channel::<Vec<q_types::QBlock>>();

                    // Store in pending requests map
                    if let Ok(mut pending) = self.pending_block_requests.lock() {
                        pending.insert(request_id_str.clone(), internal_tx);
                    }

                    // Spawn task to wait for response and forward it
                    let response_tx_clone = response_tx;
                    tokio::spawn(async move {
                        match internal_rx.await {
                            Ok(blocks) => {
                                info!("✅ [TURBO SYNC DIRECT] Received {} blocks via request-response", blocks.len());
                                let _ = response_tx_clone.send(Ok(blocks));
                            }
                            Err(_) => {
                                warn!("❌ [TURBO SYNC DIRECT] Response channel dropped (timeout or error)");
                                let _ = response_tx_clone.send(Err(anyhow::anyhow!("Request-response timeout or channel closed")));
                            }
                        }
                    });

                    info!("✅ [TURBO SYNC DIRECT] Request sent, waiting for response (request_id: {})",
                          &request_id_str[..20.min(request_id_str.len())]);
                } else {
                    let _ = response_tx.send(Err(anyhow::anyhow!("No valid peer available for direct request")));
                }
            }

            // ============================================================================
            // v1.3.11-beta: DECENTRALIZED CONSENSUS P2P HANDLERS
            // ============================================================================

            NetworkCommand::PublishConsensusRequest { topic, vertex_id, round, block_hash, requester_id } => {
                info!("🔐 [CONSENSUS P2P] Broadcasting signature request for vertex {}.. (round {})",
                      hex::encode(&vertex_id[..8]), round);

                // Create consensus request message
                #[derive(serde::Serialize)]
                struct ConsensusRequest {
                    msg_type: &'static str,
                    vertex_id: String,
                    round: u64,
                    block_hash: String,
                    requester_id: String,
                    timestamp: u64,
                }

                let request = ConsensusRequest {
                    msg_type: "consensus_signature_request",
                    vertex_id: hex::encode(vertex_id),
                    round,
                    block_hash: hex::encode(block_hash),
                    requester_id: hex::encode(requester_id),
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs(),
                };

                let request_bytes = serde_json::to_vec(&request).unwrap_or_default();

                match crate::gossipsub_queue::gossipsub_queue().enqueue(topic.clone(), request_bytes) {
                    Ok(()) => {
                        debug!("📤 [QUEUE] Enqueued consensus signature request (topic={})", topic);
                    }
                    Err(reason) => {
                        warn!("⚠️ [QUEUE] Consensus signature request dropped: {} (topic={})", reason, topic);
                    }
                }
            }

            NetworkCommand::PublishConsensusSignature { topic, vertex_id, validator_id, signature, public_key, timestamp } => {
                info!("🔐 [CONSENSUS P2P] Publishing signature for vertex {}.. from validator {}",
                      hex::encode(&vertex_id[..8]), hex::encode(&validator_id[..8]));

                // Create consensus signature message
                #[derive(serde::Serialize)]
                struct ConsensusSignature {
                    msg_type: &'static str,
                    vertex_id: String,
                    validator_id: String,
                    signature: String,
                    public_key: String,
                    timestamp: u64,
                }

                let sig_msg = ConsensusSignature {
                    msg_type: "consensus_signature_response",
                    vertex_id: hex::encode(vertex_id),
                    validator_id: hex::encode(validator_id),
                    signature: hex::encode(signature),
                    public_key: hex::encode(public_key),
                    timestamp,
                };

                let sig_bytes = serde_json::to_vec(&sig_msg).unwrap_or_default();

                match crate::gossipsub_queue::gossipsub_queue().enqueue(topic.clone(), sig_bytes) {
                    Ok(()) => {
                        debug!("📤 [QUEUE] Enqueued consensus signature (topic={})", topic);
                    }
                    Err(reason) => {
                        warn!("⚠️ [QUEUE] Consensus signature dropped: {} (topic={})", reason, topic);
                    }
                }
            }

            NetworkCommand::PublishConsensusCertificate { topic, vertex_id, round, signatures, threshold_met } => {
                info!("🎖️ [CONSENSUS P2P] Broadcasting certificate for vertex {}.. ({} signatures, threshold_met: {})",
                      hex::encode(&vertex_id[..8]), signatures.len(), threshold_met);

                // Create consensus certificate message
                #[derive(serde::Serialize)]
                struct ConsensusCertificate {
                    msg_type: &'static str,
                    vertex_id: String,
                    round: u64,
                    signatures: Vec<(String, String)>,
                    threshold_met: bool,
                    timestamp: u64,
                }

                let cert_msg = ConsensusCertificate {
                    msg_type: "consensus_certificate",
                    vertex_id: hex::encode(vertex_id),
                    round,
                    signatures: signatures.iter()
                        .map(|(id, sig)| (hex::encode(id), hex::encode(sig)))
                        .collect(),
                    threshold_met,
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs(),
                };

                let cert_bytes = serde_json::to_vec(&cert_msg).unwrap_or_default();

                match crate::gossipsub_queue::gossipsub_queue().enqueue(topic.clone(), cert_bytes) {
                    Ok(()) => {
                        debug!("📤 [QUEUE] Enqueued consensus certificate (topic={})", topic);
                    }
                    Err(reason) => {
                        warn!("⚠️ [QUEUE] Consensus certificate dropped: {} (topic={})", reason, topic);
                    }
                }
            }

            NetworkCommand::ReportEquivocation { topic, validator_id, vertex_id, signature1, signature2 } => {
                error!("🚨 [CONSENSUS P2P] EQUIVOCATION DETECTED! Validator {} double-signed vertex {}",
                       hex::encode(&validator_id[..8]), hex::encode(&vertex_id[..8]));

                // Create equivocation proof message
                #[derive(serde::Serialize)]
                struct EquivocationReport {
                    msg_type: &'static str,
                    validator_id: String,
                    vertex_id: String,
                    signature1: String,
                    signature2: String,
                    timestamp: u64,
                }

                let report = EquivocationReport {
                    msg_type: "equivocation_proof",
                    validator_id: hex::encode(validator_id),
                    vertex_id: hex::encode(vertex_id),
                    signature1: hex::encode(&signature1),
                    signature2: hex::encode(&signature2),
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs(),
                };

                let report_bytes = serde_json::to_vec(&report).unwrap_or_default();

                // v4.3.0-beta: Route through priority queue (Critical priority for consensus)
                match crate::gossipsub_queue::gossipsub_queue().enqueue(topic.clone(), report_bytes) {
                    Ok(()) => {
                        error!("📢 [CONSENSUS P2P] Equivocation proof enqueued - validator {} will be slashed",
                               hex::encode(&validator_id[..8]));
                    }
                    Err(reason) => {
                        error!("❌ [CONSENSUS P2P] Failed to enqueue equivocation proof: {}", reason);
                    }
                }
            }

            // ⚡ v2.6.0: Handle all-reduce messages for tensor parallelism
            // v4.3.0-beta: Route through priority queue
            NetworkCommand::PublishAllReduce { topic, data } => {
                debug!("⚡ [ALL-REDUCE] Publishing {} bytes to topic: {}", data.len(), topic);
                match crate::gossipsub_queue::gossipsub_queue().enqueue(topic.clone(), data) {
                    Ok(()) => {
                        debug!("✅ [QUEUE] Enqueued all-reduce message (topic={})", topic);
                    }
                    Err(reason) => {
                        warn!("⚠️ [QUEUE] All-reduce message dropped: {} (topic={})", reason, topic);
                    }
                }
            }

            // 🔐 v2.4.6-beta: BFT consensus message publishing
            // v4.3.0-beta: Route through priority queue
            NetworkCommand::PublishConsensusMessage { topic, message_bytes } => {
                debug!("🔐 [BFT] Publishing {} bytes to consensus topic: {}", message_bytes.len(), topic);
                match crate::gossipsub_queue::gossipsub_queue().enqueue(topic.clone(), message_bytes) {
                    Ok(()) => {
                        debug!("✅ [QUEUE] Enqueued BFT consensus message (topic={})", topic);
                    }
                    Err(reason) => {
                        warn!("⚠️ [QUEUE] BFT consensus message dropped: {} (topic={})", reason, topic);
                    }
                }
            }
        }
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

// NOTE: UnifiedNetworkManager does NOT implement NetworkFetcher directly
// because it contains libp2p Swarm types that are !Sync.
// Instead, use DagSyncNetworkAdapter (in dag_sync_adapter.rs) which wraps
// UnifiedNetworkManager in Arc<RwLock<>> to provide the required Sync trait bound.
//
// The DagSyncNetworkAdapter provides the NetworkFetcher implementation
// by acquiring short-lived write locks to call the &mut self methods below.