use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

#[derive(Debug, Clone)]
pub struct ConnectedPeer {
    pub node_id: String,
    pub address: SocketAddr,
    pub server_role: String,
    pub connected_at: i64,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    info!("🌉 Starting Q-NarwhalKnight Connection Bridge for Server Beta");
    info!("🎯 Purpose: Accept connections from Alpha nodes after DNS-Phantom discovery");

    let port = 8081; // Use port 8081 for P2P connections
    start_connection_bridge(port).await?;

    Ok(())
}

pub async fn start_connection_bridge(port: u16) -> Result<(), Box<dyn std::error::Error>> {
    let address = format!("0.0.0.0:{}", port);
    let listener = TcpListener::bind(&address).await?;

    info!("🌉 Connection Bridge started on {}", address);
    info!("🎯 Ready to accept discovered peer connections from Alpha nodes");

    let connected_peers = Arc::new(RwLock::new(HashMap::<String, ConnectedPeer>::new()));

    while let Ok((stream, addr)) = listener.accept().await {
        info!("📥 ALPHA CONNECTION DETECTED from: {}", addr);

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
    info!("🎉 SUCCESS: Alpha node connected from {}", addr);

    // Send welcome message
    let welcome = "🎯 Q-NarwhalKnight Server Beta P2P Bridge - Connection Successful!\n";
    stream.write_all(welcome.as_bytes()).await?;
    stream.flush().await?;

    // Try to read any data
    let mut buffer = vec![0; 1024];
    let n = stream.read(&mut buffer).await?;

    if n > 0 {
        let received_data = String::from_utf8_lossy(&buffer[..n]);
        info!("📨 Received from Alpha {}: {}", addr, received_data.trim());
    }

    // Create peer record
    let peer_id = format!("alpha-peer-{}", addr.port());
    let peer = ConnectedPeer {
        node_id: peer_id.clone(),
        address: addr,
        server_role: "alpha".to_string(),
        connected_at: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64,
    };

    // Add to connected peers
    {
        let mut peer_list = peers.write().await;
        peer_list.insert(peer_id.clone(), peer.clone());
        info!(
            "✅ SUCCESS: Added Alpha peer {} (total connections: {})",
            peer_id,
            peer_list.len()
        );
    }

    // Send success confirmation
    let response = format!(
        "{{\"status\":\"connected\",\"server\":\"beta\",\"peer_id\":\"{}\",\"total_peers\":{}}}\n",
        peer_id,
        peers.read().await.len()
    );

    stream.write_all(response.as_bytes()).await?;
    stream.flush().await?;

    info!(
        "🎉 COLLABORATION SUCCESS: Alpha node {} established connection",
        peer_id
    );

    // Keep connection alive
    loop {
        let mut buffer = vec![0; 512];
        match stream.read(&mut buffer).await {
            Ok(0) => {
                info!("🔌 Alpha peer {} disconnected", peer_id);
                break;
            }
            Ok(n) => {
                let message = String::from_utf8_lossy(&buffer[..n]);
                debug!("📨 Message from Alpha {}: {}", peer_id, message.trim());

                // Send acknowledgment
                let ack = format!("Beta received: {}", message.trim());
                if stream.write_all(ack.as_bytes()).await.is_err() {
                    break;
                }
                if stream.write_all(b"\n").await.is_err() {
                    break;
                }
                if stream.flush().await.is_err() {
                    break;
                }
            }
            Err(_) => break,
        }
    }

    // Remove from connected peers
    {
        let mut peer_list = peers.write().await;
        peer_list.remove(&peer_id);
        info!(
            "👥 Removed Alpha peer {} (remaining: {})",
            peer_id,
            peer_list.len()
        );
    }

    Ok(())
}
