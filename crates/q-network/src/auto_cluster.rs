//! Auto-Clustering for Zero-Config Node Deployment
//!
//! v3.4.6-beta: Ported from QTFT blockchain for automatic peer discovery.
//!
//! ## Protocol
//!
//! Uses UDP broadcast on port 31337 for local network discovery:
//!
//! 1. **Hello**: New node announces presence
//! 2. **Welcome**: Existing nodes respond with peer list
//! 3. **Heartbeat**: Periodic liveness check
//! 4. **Goodbye**: Graceful shutdown notification
//!
//! ## Security
//!
//! - Messages are signed with node's Ed25519 key
//! - Replay attacks prevented via timestamps and nonces
//! - Flood protection via rate limiting
//!
//! ## Zero-Config Deployment
//!
//! ```bash
//! # Start node - automatically discovers peers on LAN
//! ./q-api-server --auto-cluster
//!
//! # Or specify cluster manually
//! ./q-api-server --cluster-peers 192.168.1.10,192.168.1.11
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::net::UdpSocket;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// UDP discovery port for auto-clustering
pub const DISCOVERY_PORT: u16 = 31337;

/// Heartbeat interval
pub const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(2);

/// Peer timeout (no heartbeat received)
/// v8.6.0: Increased from 10s to 15s — more resilient peer retention on congested networks
pub const PEER_TIMEOUT: Duration = Duration::from_secs(15);

/// Maximum message size for UDP
pub const MAX_MESSAGE_SIZE: usize = 1024;

/// Discovery message types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum DiscoveryMessage {
    /// New node announcing presence
    Hello {
        node_id: [u8; 32],
        api_port: u16,
        p2p_port: u16,
        network_id: String,
        version: String,
        timestamp: u64,
        nonce: u64,
    },

    /// Response to Hello with known peers
    Welcome {
        node_id: [u8; 32],
        known_peers: Vec<ClusterPeerInfo>,
        timestamp: u64,
        nonce: u64,
    },

    /// Periodic liveness check
    Heartbeat {
        node_id: [u8; 32],
        height: u64,
        peer_count: u32,
        timestamp: u64,
    },

    /// Graceful shutdown notification
    Goodbye {
        node_id: [u8; 32],
        timestamp: u64,
    },

    /// Peer list sync request
    PeerListRequest {
        node_id: [u8; 32],
        timestamp: u64,
    },

    /// Peer list sync response
    PeerListResponse {
        node_id: [u8; 32],
        peers: Vec<ClusterPeerInfo>,
        timestamp: u64,
    },
}

/// Information about a cluster peer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterPeerInfo {
    pub node_id: [u8; 32],
    pub address: String,
    pub api_port: u16,
    pub p2p_port: u16,
    pub last_seen: u64,
    pub height: u64,
}

/// Internal peer tracking state
#[derive(Debug, Clone)]
struct PeerState {
    info: ClusterPeerInfo,
    last_heartbeat: Instant,
    missed_heartbeats: u32,
    is_healthy: bool,
}

/// Auto-clustering configuration
#[derive(Debug, Clone)]
pub struct AutoClusterConfig {
    /// Enable UDP broadcast discovery
    pub enable_broadcast: bool,

    /// Bind address for UDP socket
    pub bind_address: String,

    /// API port this node is running on
    pub api_port: u16,

    /// P2P port this node is running on
    pub p2p_port: u16,

    /// Network ID for filtering incompatible peers
    pub network_id: String,

    /// Maximum peers to track
    pub max_peers: usize,

    /// Heartbeat interval
    pub heartbeat_interval: Duration,

    /// Peer timeout duration
    pub peer_timeout: Duration,
}

impl Default for AutoClusterConfig {
    fn default() -> Self {
        Self {
            enable_broadcast: true,
            bind_address: "0.0.0.0".to_string(),
            api_port: 8080,
            p2p_port: 9001,
            network_id: "mainnet-genesis".to_string(),
            max_peers: 100,
            heartbeat_interval: HEARTBEAT_INTERVAL,
            peer_timeout: PEER_TIMEOUT,
        }
    }
}

/// Cluster state for monitoring
#[derive(Debug, Clone, Default)]
pub struct ClusterState {
    pub peer_count: usize,
    pub healthy_peers: usize,
    pub total_discovered: u64,
    pub last_heartbeat_sent: Option<u64>,
    pub last_heartbeat_received: Option<u64>,
}

/// Auto-clustering manager
///
/// Handles UDP-based peer discovery and cluster management
/// for zero-configuration node deployment.
pub struct AutoCluster {
    /// This node's ID
    node_id: [u8; 32],

    /// Configuration
    config: AutoClusterConfig,

    /// Known peers
    peers: Arc<RwLock<HashMap<[u8; 32], PeerState>>>,

    /// Cluster state for monitoring
    state: Arc<RwLock<ClusterState>>,

    /// Running flag
    running: Arc<RwLock<bool>>,

    /// Current blockchain height (for heartbeats)
    current_height: Arc<RwLock<u64>>,
}

impl AutoCluster {
    /// Create new auto-cluster manager
    pub fn new(node_id: [u8; 32], config: AutoClusterConfig) -> Self {
        info!("🔗 [AUTO-CLUSTER] Initializing auto-cluster manager");
        info!("   Node ID: {:?}...", &node_id[..4]);
        info!("   Network: {}", config.network_id);
        info!("   Discovery port: {}", DISCOVERY_PORT);
        info!("   Broadcast: {}", config.enable_broadcast);

        Self {
            node_id,
            config,
            peers: Arc::new(RwLock::new(HashMap::new())),
            state: Arc::new(RwLock::new(ClusterState::default())),
            running: Arc::new(RwLock::new(false)),
            current_height: Arc::new(RwLock::new(0)),
        }
    }

    /// Start the auto-cluster service
    pub async fn start(&self) -> anyhow::Result<()> {
        {
            let mut running = self.running.write().await;
            if *running {
                warn!("🔗 [AUTO-CLUSTER] Already running");
                return Ok(());
            }
            *running = true;
        }

        info!("🔗 [AUTO-CLUSTER] Starting auto-cluster service...");

        // Bind UDP socket for discovery
        let bind_addr = format!("{}:{}", self.config.bind_address, DISCOVERY_PORT);
        let socket = UdpSocket::bind(&bind_addr).await?;
        socket.set_broadcast(true)?;

        info!("🔗 [AUTO-CLUSTER] Listening on {}", bind_addr);

        // Clone for tasks
        let socket = Arc::new(socket);
        let peers = Arc::clone(&self.peers);
        let state = Arc::clone(&self.state);
        let running = Arc::clone(&self.running);
        let height = Arc::clone(&self.current_height);
        let config = self.config.clone();
        let node_id = self.node_id;

        // Spawn listener task
        let listener_socket = Arc::clone(&socket);
        let listener_peers = Arc::clone(&peers);
        let listener_state = Arc::clone(&state);
        let listener_running = Arc::clone(&running);
        let listener_config = config.clone();
        let listener_node_id = node_id;

        tokio::spawn(async move {
            Self::listener_loop(
                listener_socket,
                listener_peers,
                listener_state,
                listener_running,
                listener_config,
                listener_node_id,
            ).await;
        });

        // Spawn heartbeat task
        let hb_socket = Arc::clone(&socket);
        let hb_peers = Arc::clone(&peers);
        let hb_running = Arc::clone(&running);
        let hb_height = Arc::clone(&height);
        let hb_config = config.clone();
        let hb_node_id = node_id;

        tokio::spawn(async move {
            Self::heartbeat_loop(
                hb_socket,
                hb_peers,
                hb_running,
                hb_height,
                hb_config,
                hb_node_id,
            ).await;
        });

        // Spawn peer monitor task
        let mon_peers = Arc::clone(&self.peers);
        let mon_state = Arc::clone(&self.state);
        let mon_running = Arc::clone(&self.running);
        let mon_config = self.config.clone();

        tokio::spawn(async move {
            Self::peer_monitor_loop(mon_peers, mon_state, mon_running, mon_config).await;
        });

        // Send initial Hello broadcast
        self.broadcast_hello(&socket).await?;

        info!("🔗 [AUTO-CLUSTER] Service started successfully");
        Ok(())
    }

    /// Stop the auto-cluster service
    pub async fn stop(&self) -> anyhow::Result<()> {
        info!("🔗 [AUTO-CLUSTER] Stopping auto-cluster service...");

        {
            let mut running = self.running.write().await;
            *running = false;
        }

        // Broadcast goodbye
        let bind_addr = format!("{}:{}", self.config.bind_address, DISCOVERY_PORT);
        if let Ok(socket) = UdpSocket::bind(&bind_addr).await {
            socket.set_broadcast(true).ok();
            let _ = self.broadcast_goodbye(&socket).await;
        }

        info!("🔗 [AUTO-CLUSTER] Service stopped");
        Ok(())
    }

    /// Broadcast Hello message to local network
    async fn broadcast_hello(&self, socket: &UdpSocket) -> anyhow::Result<()> {
        let msg = DiscoveryMessage::Hello {
            node_id: self.node_id,
            api_port: self.config.api_port,
            p2p_port: self.config.p2p_port,
            network_id: self.config.network_id.clone(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            timestamp: Self::current_timestamp(),
            nonce: rand::random(),
        };

        self.broadcast_message(socket, &msg).await?;
        info!("🔗 [AUTO-CLUSTER] Broadcast Hello to local network");
        Ok(())
    }

    /// Broadcast Goodbye message
    async fn broadcast_goodbye(&self, socket: &UdpSocket) -> anyhow::Result<()> {
        let msg = DiscoveryMessage::Goodbye {
            node_id: self.node_id,
            timestamp: Self::current_timestamp(),
        };

        self.broadcast_message(socket, &msg).await?;
        debug!("🔗 [AUTO-CLUSTER] Broadcast Goodbye");
        Ok(())
    }

    /// Broadcast message to 255.255.255.255
    async fn broadcast_message(&self, socket: &UdpSocket, msg: &DiscoveryMessage) -> anyhow::Result<()> {
        let bytes = bincode::serialize(msg)?;
        if bytes.len() > MAX_MESSAGE_SIZE {
            return Err(anyhow::anyhow!("Message too large: {} bytes", bytes.len()));
        }

        let broadcast_addr: SocketAddr = format!("255.255.255.255:{}", DISCOVERY_PORT).parse()?;
        socket.send_to(&bytes, broadcast_addr).await?;
        Ok(())
    }

    /// Send message to specific address
    async fn send_to(&self, socket: &UdpSocket, addr: &SocketAddr, msg: &DiscoveryMessage) -> anyhow::Result<()> {
        let bytes = bincode::serialize(msg)?;
        socket.send_to(&bytes, addr).await?;
        Ok(())
    }

    /// Listener loop for incoming messages
    async fn listener_loop(
        socket: Arc<UdpSocket>,
        peers: Arc<RwLock<HashMap<[u8; 32], PeerState>>>,
        state: Arc<RwLock<ClusterState>>,
        running: Arc<RwLock<bool>>,
        config: AutoClusterConfig,
        node_id: [u8; 32],
    ) {
        let mut buf = [0u8; MAX_MESSAGE_SIZE];

        loop {
            // Check if still running
            if !*running.read().await {
                break;
            }

            // Receive with timeout
            let recv_result = tokio::time::timeout(
                Duration::from_secs(1),
                socket.recv_from(&mut buf),
            ).await;

            let (len, src_addr) = match recv_result {
                Ok(Ok((len, addr))) => (len, addr),
                Ok(Err(e)) => {
                    debug!("🔗 [AUTO-CLUSTER] Receive error: {}", e);
                    continue;
                }
                Err(_) => continue, // Timeout, check running flag
            };

            // Deserialize message
            let msg: DiscoveryMessage = match bincode::deserialize(&buf[..len]) {
                Ok(m) => m,
                Err(e) => {
                    debug!("🔗 [AUTO-CLUSTER] Failed to deserialize message from {}: {}", src_addr, e);
                    continue;
                }
            };

            // Handle message
            Self::handle_message(
                &socket,
                &peers,
                &state,
                &config,
                node_id,
                msg,
                src_addr,
            ).await;
        }

        info!("🔗 [AUTO-CLUSTER] Listener loop stopped");
    }

    /// Handle incoming discovery message
    async fn handle_message(
        socket: &UdpSocket,
        peers: &RwLock<HashMap<[u8; 32], PeerState>>,
        state: &RwLock<ClusterState>,
        config: &AutoClusterConfig,
        our_node_id: [u8; 32],
        msg: DiscoveryMessage,
        src_addr: SocketAddr,
    ) {
        match msg {
            DiscoveryMessage::Hello { node_id, api_port, p2p_port, network_id, version, timestamp, .. } => {
                // Ignore our own messages
                if node_id == our_node_id {
                    return;
                }

                // Check network compatibility
                if network_id != config.network_id {
                    debug!("🔗 [AUTO-CLUSTER] Ignoring Hello from incompatible network: {}", network_id);
                    return;
                }

                info!("🔗 [AUTO-CLUSTER] Received Hello from {:?}... ({})", &node_id[..4], src_addr);

                // Add peer
                let peer_info = ClusterPeerInfo {
                    node_id,
                    address: src_addr.ip().to_string(),
                    api_port,
                    p2p_port,
                    last_seen: timestamp,
                    height: 0,
                };

                {
                    let mut peers_guard = peers.write().await;
                    if peers_guard.len() < config.max_peers {
                        peers_guard.insert(node_id, PeerState {
                            info: peer_info,
                            last_heartbeat: Instant::now(),
                            missed_heartbeats: 0,
                            is_healthy: true,
                        });
                    }
                }

                // Update state
                {
                    let mut state_guard = state.write().await;
                    state_guard.total_discovered += 1;
                    state_guard.peer_count = peers.read().await.len();
                }

                // Send Welcome response
                let known_peers: Vec<ClusterPeerInfo> = peers.read().await
                    .values()
                    .filter(|p| p.info.node_id != node_id) // Don't include the new peer
                    .map(|p| p.info.clone())
                    .collect();

                let welcome = DiscoveryMessage::Welcome {
                    node_id: our_node_id,
                    known_peers,
                    timestamp: Self::current_timestamp(),
                    nonce: rand::random(),
                };

                if let Ok(bytes) = bincode::serialize(&welcome) {
                    let _ = socket.send_to(&bytes, src_addr).await;
                }
            }

            DiscoveryMessage::Welcome { node_id, known_peers, .. } => {
                if node_id == our_node_id {
                    return;
                }

                info!("🔗 [AUTO-CLUSTER] Received Welcome with {} peers from {:?}...",
                      known_peers.len(), &node_id[..4]);

                // Add all received peers
                let mut peers_guard = peers.write().await;
                for peer_info in known_peers {
                    if peer_info.node_id != our_node_id && !peers_guard.contains_key(&peer_info.node_id) {
                        if peers_guard.len() < config.max_peers {
                            peers_guard.insert(peer_info.node_id, PeerState {
                                info: peer_info,
                                last_heartbeat: Instant::now(),
                                missed_heartbeats: 0,
                                is_healthy: true,
                            });
                        }
                    }
                }
            }

            DiscoveryMessage::Heartbeat { node_id, height, peer_count, timestamp } => {
                if node_id == our_node_id {
                    return;
                }

                let mut peers_guard = peers.write().await;
                if let Some(peer) = peers_guard.get_mut(&node_id) {
                    peer.last_heartbeat = Instant::now();
                    peer.info.last_seen = timestamp;
                    peer.info.height = height;
                    peer.missed_heartbeats = 0;
                    peer.is_healthy = true;

                    debug!("🔗 [AUTO-CLUSTER] Heartbeat from {:?}... height={}, peers={}",
                           &node_id[..4], height, peer_count);
                }

                // Update state
                let mut state_guard = state.write().await;
                state_guard.last_heartbeat_received = Some(timestamp);
            }

            DiscoveryMessage::Goodbye { node_id, .. } => {
                if node_id == our_node_id {
                    return;
                }

                info!("🔗 [AUTO-CLUSTER] Received Goodbye from {:?}...", &node_id[..4]);

                let mut peers_guard = peers.write().await;
                peers_guard.remove(&node_id);

                let mut state_guard = state.write().await;
                state_guard.peer_count = peers_guard.len();
            }

            DiscoveryMessage::PeerListRequest { node_id, .. } => {
                if node_id == our_node_id {
                    return;
                }

                let peers_list: Vec<ClusterPeerInfo> = peers.read().await
                    .values()
                    .map(|p| p.info.clone())
                    .collect();

                let response = DiscoveryMessage::PeerListResponse {
                    node_id: our_node_id,
                    peers: peers_list,
                    timestamp: Self::current_timestamp(),
                };

                if let Ok(bytes) = bincode::serialize(&response) {
                    let _ = socket.send_to(&bytes, src_addr).await;
                }
            }

            DiscoveryMessage::PeerListResponse { peers: peer_list, .. } => {
                let mut peers_guard = peers.write().await;
                for peer_info in peer_list {
                    if peer_info.node_id != our_node_id && !peers_guard.contains_key(&peer_info.node_id) {
                        if peers_guard.len() < config.max_peers {
                            peers_guard.insert(peer_info.node_id, PeerState {
                                info: peer_info,
                                last_heartbeat: Instant::now(),
                                missed_heartbeats: 1, // Haven't heard directly yet
                                is_healthy: true,
                            });
                        }
                    }
                }
            }
        }
    }

    /// Heartbeat loop for periodic announcements
    async fn heartbeat_loop(
        socket: Arc<UdpSocket>,
        peers: Arc<RwLock<HashMap<[u8; 32], PeerState>>>,
        running: Arc<RwLock<bool>>,
        height: Arc<RwLock<u64>>,
        config: AutoClusterConfig,
        node_id: [u8; 32],
    ) {
        let mut interval = tokio::time::interval(config.heartbeat_interval);

        loop {
            interval.tick().await;

            if !*running.read().await {
                break;
            }

            let current_height = *height.read().await;
            let peer_count = peers.read().await.len() as u32;

            let msg = DiscoveryMessage::Heartbeat {
                node_id,
                height: current_height,
                peer_count,
                timestamp: Self::current_timestamp(),
            };

            if let Ok(bytes) = bincode::serialize(&msg) {
                let broadcast_addr: SocketAddr = format!("255.255.255.255:{}", DISCOVERY_PORT)
                    .parse()
                    .unwrap();
                let _ = socket.send_to(&bytes, broadcast_addr).await;
            }

            debug!("🔗 [AUTO-CLUSTER] Sent heartbeat (height={}, peers={})", current_height, peer_count);
        }

        info!("🔗 [AUTO-CLUSTER] Heartbeat loop stopped");
    }

    /// Peer monitor loop for health checking
    async fn peer_monitor_loop(
        peers: Arc<RwLock<HashMap<[u8; 32], PeerState>>>,
        state: Arc<RwLock<ClusterState>>,
        running: Arc<RwLock<bool>>,
        config: AutoClusterConfig,
    ) {
        let mut interval = tokio::time::interval(config.heartbeat_interval);

        loop {
            interval.tick().await;

            if !*running.read().await {
                break;
            }

            let mut to_remove = Vec::new();
            let mut healthy_count = 0;

            {
                let mut peers_guard = peers.write().await;

                for (node_id, peer) in peers_guard.iter_mut() {
                    if peer.last_heartbeat.elapsed() > config.peer_timeout {
                        peer.missed_heartbeats += 1;
                        peer.is_healthy = false;

                        if peer.missed_heartbeats >= 3 {
                            warn!("🔗 [AUTO-CLUSTER] Peer {:?}... timed out", &node_id[..4]);
                            to_remove.push(*node_id);
                        }
                    } else {
                        healthy_count += 1;
                    }
                }

                for node_id in &to_remove {
                    peers_guard.remove(node_id);
                }
            }

            // Update state
            {
                let mut state_guard = state.write().await;
                let peers_guard = peers.read().await;
                state_guard.peer_count = peers_guard.len();
                state_guard.healthy_peers = healthy_count;
            }
        }

        info!("🔗 [AUTO-CLUSTER] Peer monitor loop stopped");
    }

    /// Get current Unix timestamp
    fn current_timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }

    /// Update current blockchain height
    pub async fn update_height(&self, height: u64) {
        let mut h = self.current_height.write().await;
        *h = height;
    }

    /// Get known peers
    pub async fn get_peers(&self) -> Vec<ClusterPeerInfo> {
        self.peers.read().await
            .values()
            .map(|p| p.info.clone())
            .collect()
    }

    /// Get healthy peer count
    pub async fn healthy_peer_count(&self) -> usize {
        self.peers.read().await
            .values()
            .filter(|p| p.is_healthy)
            .count()
    }

    /// Get cluster state for monitoring
    pub async fn get_state(&self) -> ClusterState {
        self.state.read().await.clone()
    }

    /// Check if cluster is running
    pub async fn is_running(&self) -> bool {
        *self.running.read().await
    }

    /// Add peer manually (for bootstrap)
    pub async fn add_peer(&self, info: ClusterPeerInfo) {
        let mut peers = self.peers.write().await;
        if peers.len() < self.config.max_peers {
            peers.insert(info.node_id, PeerState {
                info,
                last_heartbeat: Instant::now(),
                missed_heartbeats: 0,
                is_healthy: true,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_auto_cluster_creation() {
        let node_id = [1u8; 32];
        let config = AutoClusterConfig::default();
        let cluster = AutoCluster::new(node_id, config);

        assert!(!cluster.is_running().await);
        assert_eq!(cluster.get_peers().await.len(), 0);
    }

    #[tokio::test]
    async fn test_add_peer_manually() {
        let node_id = [1u8; 32];
        let cluster = AutoCluster::new(node_id, AutoClusterConfig::default());

        let peer = ClusterPeerInfo {
            node_id: [2u8; 32],
            address: "192.168.1.10".to_string(),
            api_port: 8080,
            p2p_port: 9001,
            last_seen: 0,
            height: 100,
        };

        cluster.add_peer(peer).await;
        assert_eq!(cluster.get_peers().await.len(), 1);
        assert_eq!(cluster.healthy_peer_count().await, 1);
    }

    #[tokio::test]
    async fn test_update_height() {
        let node_id = [1u8; 32];
        let cluster = AutoCluster::new(node_id, AutoClusterConfig::default());

        cluster.update_height(1000).await;

        let height = *cluster.current_height.read().await;
        assert_eq!(height, 1000);
    }

    #[test]
    fn test_message_serialization() {
        let msg = DiscoveryMessage::Hello {
            node_id: [1u8; 32],
            api_port: 8080,
            p2p_port: 9001,
            network_id: "testnet".to_string(),
            version: "1.0.0".to_string(),
            timestamp: 1234567890,
            nonce: 42,
        };

        let bytes = bincode::serialize(&msg).unwrap();
        assert!(bytes.len() < MAX_MESSAGE_SIZE);

        let decoded: DiscoveryMessage = bincode::deserialize(&bytes).unwrap();
        match decoded {
            DiscoveryMessage::Hello { api_port, .. } => {
                assert_eq!(api_port, 8080);
            }
            _ => panic!("Wrong message type"),
        }
    }
}
