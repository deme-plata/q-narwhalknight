use crate::NodeStatus;
use q_types::NodeId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, error, info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandshakeMessage {
    pub node_id: String,
    pub server_role: String,
    pub protocol_version: u32,
    pub discovery_method: String,
    pub timestamp: u64,
}

#[derive(Debug, Clone)]
pub struct PeerConnection {
    pub node_id: String,
    pub address: SocketAddr,
    pub server_role: String,
    pub connected_at: std::time::SystemTime,
}

pub type ActivePeers = Arc<RwLock<HashMap<String, PeerConnection>>>;

pub async fn start_p2p_listener(
    port: u16,
    local_node_id: NodeId,
    active_peers: ActivePeers,
    node_status: Arc<RwLock<NodeStatus>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let p2p_address = format!("0.0.0.0:{}", port + 1); // Use port+1 for P2P
    let listener = TcpListener::bind(&p2p_address).await?;

    info!("🔗 P2P Connection Listener started on {}", p2p_address);
    info!("📡 Ready to accept peer connections from Alpha nodes");

    while let Ok((stream, addr)) = listener.accept().await {
        info!("📥 Incoming P2P connection from: {}", addr);

        let local_node_id_clone = local_node_id;
        let active_peers_clone = active_peers.clone();
        let node_status_clone = node_status.clone();

        tokio::spawn(async move {
            if let Err(e) = handle_p2p_connection(
                stream,
                addr,
                local_node_id_clone,
                active_peers_clone,
                node_status_clone,
            )
            .await
            {
                warn!("❌ P2P connection handling failed for {}: {}", addr, e);
            }
        });
    }

    Ok(())
}

async fn handle_p2p_connection(
    mut stream: TcpStream,
    addr: SocketAddr,
    local_node_id: NodeId,
    active_peers: ActivePeers,
    node_status: Arc<RwLock<NodeStatus>>,
) -> Result<(), Box<dyn std::error::Error>> {
    debug!("🤝 Starting handshake with {}", addr);

    // Read handshake message
    let mut buffer = vec![0; 1024];
    let n = stream.read(&mut buffer).await?;

    if n == 0 {
        return Err("Connection closed during handshake".into());
    }

    let handshake_data = String::from_utf8_lossy(&buffer[..n]);
    debug!("📨 Received handshake data: {}", handshake_data);

    // Try to parse as JSON handshake
    let peer_handshake: HandshakeMessage = match serde_json::from_str(&handshake_data) {
        Ok(msg) => msg,
        Err(_) => {
            // Fallback: treat as simple connection attempt
            let peer_id = format!("peer-{}-{}", addr.ip(), chrono::Utc::now().timestamp());
            HandshakeMessage {
                node_id: peer_id,
                server_role: "alpha".to_string(),
                protocol_version: 1,
                discovery_method: "dns-phantom".to_string(),
                timestamp: chrono::Utc::now().timestamp_millis() as u64,
            }
        }
    };

    info!(
        "✅ Handshake received from {} node: {}",
        peer_handshake.server_role, peer_handshake.node_id
    );

    // Send our handshake response
    let our_handshake = HandshakeMessage {
        node_id: hex::encode(local_node_id),
        server_role: "beta".to_string(),
        protocol_version: 1,
        discovery_method: "server-listener".to_string(),
        timestamp: chrono::Utc::now().timestamp_millis() as u64,
    };

    let handshake_json = serde_json::to_string(&our_handshake)?;
    stream.write_all(handshake_json.as_bytes()).await?;
    stream.flush().await?;

    info!("📤 Sent handshake response to {}", addr);

    // Create peer connection record
    let peer_connection = PeerConnection {
        node_id: peer_handshake.node_id.clone(),
        address: addr,
        server_role: peer_handshake.server_role.clone(),
        connected_at: std::time::SystemTime::now(),
    };

    // Add to active peers
    {
        let mut peers = active_peers.write().await;
        peers.insert(peer_handshake.node_id.clone(), peer_connection);
        let peer_count = peers.len();
        info!(
            "👥 Added peer {} to active connections (total: {})",
            peer_handshake.node_id, peer_count
        );

        // Update node_status with current peer count
        {
            let mut status = node_status.write().await;
            status.connected_peers = peer_count as u32;
        }
    }

    // Keep connection alive and handle messages
    loop {
        let mut buffer = vec![0; 1024];
        match stream.read(&mut buffer).await {
            Ok(0) => {
                info!("🔌 Peer {} disconnected", peer_handshake.node_id);
                break;
            }
            Ok(n) => {
                let message = String::from_utf8_lossy(&buffer[..n]);
                debug!("📨 Message from {}: {}", peer_handshake.node_id, message);

                // Echo back a response
                let response = format!("Server Beta received: {}", message);
                if let Err(e) = stream.write_all(response.as_bytes()).await {
                    warn!(
                        "Failed to send response to {}: {}",
                        peer_handshake.node_id, e
                    );
                    break;
                }
            }
            Err(e) => {
                warn!("Error reading from {}: {}", peer_handshake.node_id, e);
                break;
            }
        }
    }

    // Remove from active peers when connection closes
    {
        let mut peers = active_peers.write().await;
        peers.remove(&peer_handshake.node_id);
        let peer_count = peers.len();
        info!(
            "👥 Removed peer {} from active connections (remaining: {})",
            peer_handshake.node_id, peer_count
        );

        // Update node_status with current peer count
        {
            let mut status = node_status.write().await;
            status.connected_peers = peer_count as u32;
        }
    }

    Ok(())
}
