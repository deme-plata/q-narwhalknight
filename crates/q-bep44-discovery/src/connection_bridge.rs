use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{Mutex, RwLock};
use tracing::{error, info, warn, debug};
use serde::{Deserialize, Serialize};
use chrono::Utc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleHandshake {
    pub node_id: String,
    pub server_role: String,
    pub discovery_method: String,
    pub timestamp: i64,
    pub message: String,
}

#[derive(Debug, Clone)]
pub struct ConnectedPeer {
    pub node_id: String,
    pub address: SocketAddr,
    pub server_role: String,
    pub connected_at: i64,
}

pub async fn start_connection_bridge(port: u16) -> Result<(), Box<dyn std::error::Error>> {
    let address = format!("0.0.0.0:{}", port);
    let listener = TcpListener::bind(&address).await?;
    
    info!("🌉 Connection Bridge started on {}", address);
    info!("🎯 Ready to accept discovered peer connections from Alpha nodes");
    
    let connected_peers = Arc::new(RwLock::new(HashMap::<String, ConnectedPeer>::new()));

    while let Ok((stream, addr)) = listener.accept().await {
        info!("📥 Incoming connection from: {}", addr);
        
        let peers = connected_peers.clone();
        tokio::spawn(async move {
            if let Err(e) = handle_connection(stream, addr, peers).await {
                warn!("Connection handling failed for {}: {}", addr, e);
            }
        });
    }

    Ok(())
}

async fn handle_connection(
    mut stream: TcpStream,
    addr: SocketAddr,
    peers: Arc<RwLock<HashMap<String, ConnectedPeer>>>,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("🤝 Handling connection from {}", addr);

    // Send welcome message
    let welcome = "Q-NarwhalKnight Server Beta P2P Bridge Ready\n";
    stream.write_all(welcome.as_bytes()).await?;
    stream.flush().await?;
    
    // Try to read handshake or any data
    let mut buffer = vec![0; 1024];
    let n = stream.read(&mut buffer).await?;
    
    if n == 0 {
        info!("Connection closed immediately by {}", addr);
        return Ok(());
    }
    
    let received_data = String::from_utf8_lossy(&buffer[..n]);
    info!("📨 Received from {}: {}", addr, received_data.trim());
    
    // Create peer record
    let peer_id = format!("alpha-peer-{}-{}", addr.ip(), Utc::now().timestamp());
    let peer = ConnectedPeer {
        node_id: peer_id.clone(),
        address: addr,
        server_role: "alpha".to_string(),
        connected_at: Utc::now().timestamp(),
    };
    
    // Add to connected peers
    {
        let mut peer_list = peers.write().await;
        peer_list.insert(peer_id.clone(), peer.clone());
        info!("✅ Added peer {} to active connections (total: {})", peer_id, peer_list.len());
    }
    
    // Send confirmation
    let response = SimpleHandshake {
        node_id: "server-beta-1".to_string(),
        server_role: "beta".to_string(),
        discovery_method: "dns-phantom-bridge".to_string(),
        timestamp: Utc::now().timestamp(),
        message: format!("Connected! You are peer #{}", {
            let peer_count = peers.read().await.len();
            peer_count
        }),
    };
    
    let response_json = serde_json::to_string(&response)?;
    stream.write_all(response_json.as_bytes()).await?;
    stream.write_all(b"\n").await?;
    stream.flush().await?;
    
    info!("🎉 Successfully established connection with {}", peer_id);
    
    // Keep connection alive and handle messages
    loop {
        let mut buffer = vec![0; 1024];
        match stream.read(&mut buffer).await {
            Ok(0) => {
                info!("🔌 Peer {} disconnected", peer_id);
                break;
            }
            Ok(n) => {
                let message = String::from_utf8_lossy(&buffer[..n]);
                debug!("📨 Message from {}: {}", peer_id, message.trim());
                
                // Echo response
                let echo = format!("Server Beta received: {}", message.trim());
                if let Err(e) = stream.write_all(echo.as_bytes()).await {
                    warn!("Failed to send response to {}: {}", peer_id, e);
                    break;
                }
                if let Err(e) = stream.write_all(b"\n").await {
                    warn!("Failed to send newline to {}: {}", peer_id, e);
                    break;
                }
                if let Err(e) = stream.flush().await {
                    warn!("Failed to flush to {}: {}", peer_id, e);
                    break;
                }
            }
            Err(e) => {
                warn!("Error reading from {}: {}", peer_id, e);
                break;
            }
        }
    }
    
    // Remove from connected peers
    {
        let mut peer_list = peers.write().await;
        peer_list.remove(&peer_id);
        info!("👥 Removed peer {} from connections (remaining: {})", peer_id, peer_list.len());
    }
    
    Ok(())
}