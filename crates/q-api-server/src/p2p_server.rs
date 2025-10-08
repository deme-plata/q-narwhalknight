/// P2P connection server for cross-server mesh networking
///
/// This module handles incoming P2P connections from discovered peers
/// and manages the peer-to-peer network for quantum consensus.
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::SystemTime;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

// Import our networking modules
use q_network::connection_manager::{ActiveConnection, PeerInfo};
use q_network::handshake::{
    perform_server_handshake, HandshakeError, LocalNodeInfo, RemotePeerInfo, ServerRole,
};

/// P2P server for handling incoming connections
pub struct P2PServer {
    listener: TcpListener,
    local_node: LocalNodeInfo,
    active_peers: Arc<RwLock<HashMap<String, RemotePeerInfo>>>,
    connection_stats: Arc<RwLock<ConnectionStats>>,
}

/// Connection statistics
#[derive(Debug, Default)]
pub struct ConnectionStats {
    pub total_connections: u64,
    pub successful_handshakes: u64,
    pub failed_handshakes: u64,
    pub active_peers: u64,
    pub alpha_connections: u64,
    pub beta_connections: u64,
}

impl P2PServer {
    /// Create new P2P server
    pub async fn new(port: u16, server_role: ServerRole) -> Result<Self> {
        let listener = TcpListener::bind(format!("0.0.0.0:{}", port)).await?;
        let local_addr = listener.local_addr()?;

        let local_node = match server_role {
            ServerRole::Alpha => LocalNodeInfo::new_alpha_node(),
            ServerRole::Beta => LocalNodeInfo::new_beta_node(),
            _ => LocalNodeInfo::new_beta_node(), // Default to Beta
        };

        info!(
            "🌐 P2P server started on {} as {} node: {}",
            local_addr,
            format!("{:?}", local_node.server_role),
            local_node.node_id
        );

        Ok(Self {
            listener,
            local_node,
            active_peers: Arc::new(RwLock::new(HashMap::new())),
            connection_stats: Arc::new(RwLock::new(ConnectionStats::default())),
        })
    }

    /// Start accepting P2P connections
    pub async fn start(&self) -> Result<()> {
        info!("🚀 P2P server listening for cross-server connections...");

        loop {
            match self.listener.accept().await {
                Ok((stream, addr)) => {
                    info!("📡 Incoming P2P connection from: {}", addr);

                    // Update connection stats
                    {
                        let mut stats = self.connection_stats.write().await;
                        stats.total_connections += 1;
                    }

                    // Handle connection in background task
                    let local_node = self.local_node.clone();
                    let active_peers = self.active_peers.clone();
                    let connection_stats = self.connection_stats.clone();

                    tokio::spawn(async move {
                        if let Err(e) = Self::handle_p2p_connection(
                            stream,
                            addr,
                            local_node,
                            active_peers,
                            connection_stats,
                        )
                        .await
                        {
                            warn!("❌ P2P connection handling failed for {}: {}", addr, e);
                        }
                    });
                }
                Err(e) => {
                    error!("🚫 Failed to accept P2P connection: {}", e);
                    // Continue accepting other connections
                }
            }
        }
    }

    /// Handle individual P2P connection
    async fn handle_p2p_connection(
        mut stream: TcpStream,
        addr: SocketAddr,
        local_node: LocalNodeInfo,
        active_peers: Arc<RwLock<HashMap<String, RemotePeerInfo>>>,
        connection_stats: Arc<RwLock<ConnectionStats>>,
    ) -> Result<()> {
        info!("🤝 Handling P2P connection from: {}", addr);

        // Perform handshake
        match perform_server_handshake(&mut stream, &local_node).await {
            Ok(peer_info) => {
                info!(
                    "✅ P2P handshake successful with {} node: {}",
                    format!("{:?}", peer_info.server_role),
                    peer_info.node_id
                );

                // Update statistics
                {
                    let mut stats = connection_stats.write().await;
                    stats.successful_handshakes += 1;
                    stats.active_peers += 1;

                    match peer_info.server_role {
                        ServerRole::Alpha => stats.alpha_connections += 1,
                        ServerRole::Beta => stats.beta_connections += 1,
                        _ => {}
                    }
                }

                // Add to active peer list
                {
                    let mut peers = active_peers.write().await;
                    peers.insert(peer_info.node_id.clone(), peer_info.clone());

                    info!(
                        "👥 Active peers: {} (Alpha: {}, Beta: {})",
                        peers.len(),
                        peers
                            .values()
                            .filter(|p| matches!(p.server_role, ServerRole::Alpha))
                            .count(),
                        peers
                            .values()
                            .filter(|p| matches!(p.server_role, ServerRole::Beta))
                            .count()
                    );
                }

                // Log successful cross-server connection
                if matches!(peer_info.server_role, ServerRole::Alpha) {
                    info!("🌟 CROSS-SERVER CONNECTION ESTABLISHED!");
                    info!("   ✅ Alpha node connected: {}", peer_info.node_id);
                    info!("   📡 Connection from: {}", addr);
                    info!("   🎯 Zero-configuration discovery successful!");
                }

                // Keep connection alive (simplified for demonstration)
                Self::maintain_connection(stream, peer_info, active_peers).await?;
            }
            Err(e) => {
                error!("❌ P2P handshake failed with {}: {}", addr, e);

                // Update failure statistics
                {
                    let mut stats = connection_stats.write().await;
                    stats.failed_handshakes += 1;
                }

                return Err(anyhow!("Handshake failed: {}", e));
            }
        }

        Ok(())
    }

    /// Maintain active connection
    async fn maintain_connection(
        stream: TcpStream,
        peer_info: RemotePeerInfo,
        active_peers: Arc<RwLock<HashMap<String, RemotePeerInfo>>>,
    ) -> Result<()> {
        info!("🔗 Maintaining connection with: {}", peer_info.node_id);

        // Simple keepalive loop
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(30));

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    // Send keepalive ping
                    let ping = b"ping\n";
                    if let Err(e) = stream.try_write(ping) {
                        warn!("❌ Failed to send keepalive to {}: {}", peer_info.node_id, e);
                        break;
                    }
                    debug!("📍 Sent keepalive to: {}", peer_info.node_id);
                }
                _ = tokio::time::sleep(tokio::time::Duration::from_secs(300)) => {
                    // Connection timeout
                    info!("⏰ Connection timeout with: {}", peer_info.node_id);
                    break;
                }
            }
        }

        // Remove from active peers on disconnect
        {
            let mut peers = active_peers.write().await;
            peers.remove(&peer_info.node_id);
            info!("📤 Removed disconnected peer: {}", peer_info.node_id);
        }

        Ok(())
    }

    /// Get current connection statistics
    pub async fn get_stats(&self) -> ConnectionStats {
        let stats = self.connection_stats.read().await;
        ConnectionStats {
            total_connections: stats.total_connections,
            successful_handshakes: stats.successful_handshakes,
            failed_handshakes: stats.failed_handshakes,
            active_peers: stats.active_peers,
            alpha_connections: stats.alpha_connections,
            beta_connections: stats.beta_connections,
        }
    }

    /// Get active peer count
    pub async fn get_active_peer_count(&self) -> usize {
        let peers = self.active_peers.read().await;
        peers.len()
    }

    /// Get active peer list
    pub async fn get_active_peers(&self) -> Vec<RemotePeerInfo> {
        let peers = self.active_peers.read().await;
        peers.values().cloned().collect()
    }
}

/// Start P2P server as background task
pub async fn start_p2p_server(port: u16, server_role: ServerRole) -> Result<Arc<P2PServer>> {
    let server = Arc::new(P2PServer::new(port, server_role).await?);

    // Start server in background
    let server_clone = server.clone();
    tokio::spawn(async move {
        if let Err(e) = server_clone.start().await {
            error!("💥 P2P server crashed: {}", e);
        }
    });

    // Give server time to start
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    info!("✅ P2P server started and running in background");
    Ok(server)
}
