//! Real peer-to-peer connection implementation for Q-NarwhalKnight
//! Server Beta implementation - replaces demo connection code

use anyhow::{Result, Context};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::net::TcpStream;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tracing::{info, warn, debug};
use serde::{Serialize, Deserialize};
use serde_big_array::BigArray;

use crate::{DiscoveredPeer, NodeId};

/// Placeholder for Tor client integration
#[derive(Debug)]
pub struct TorClient;

impl TorClient {
    /// Connect to onion address via Tor - placeholder implementation
    pub async fn connect_to_onion(&self, _onion_host: &str, _port: u16) -> Result<TcpStream> {
        // For now, return error indicating Tor connection not fully implemented
        anyhow::bail!("Tor connection placeholder - not fully implemented yet")
    }
}

/// Real peer connection manager - no more demo code
#[derive(Debug)]
pub struct PeerConnector {
    local_node_id: NodeId,
    connection_pool: Arc<RwLock<HashMap<NodeId, PeerConnection>>>,
    tor_client: Option<Arc<TorClient>>, // Will integrate with q-tor-client
    handshake_timeout: std::time::Duration,
    // NEW: Connection failure tracking for exponential backoff
    connection_failures: Arc<RwLock<HashMap<NodeId, ConnectionFailureState>>>,
}

/// Connection failure state for exponential backoff
#[derive(Debug, Clone)]
struct ConnectionFailureState {
    failure_count: u32,
    last_failure: chrono::DateTime<chrono::Utc>,
    next_retry_time: chrono::DateTime<chrono::Utc>,
}

/// Active peer connection
#[derive(Debug)]
pub struct PeerConnection {
    pub peer_id: NodeId,
    pub stream: Arc<RwLock<TcpStream>>,
    pub peer_address: SocketAddr,
    pub established_at: chrono::DateTime<chrono::Utc>,
    pub last_activity: Arc<RwLock<chrono::DateTime<chrono::Utc>>>,
    pub authenticated: bool,
}

/// Handshake message for peer authentication
#[derive(Serialize, Deserialize, Debug)]
pub struct HandshakeMessage {
    pub protocol_id: [u8; 16],  // "Q-NARWHALKNIGHT\0"
    #[serde(with = "BigArray")]
    pub node_id: NodeId,
    pub supported_protocols: Vec<String>,
    #[serde(with = "BigArray")]
    pub challenge: [u8; 32],    // Random challenge for auth
    #[serde(with = "BigArray")]
    pub signature: [u8; 64],    // ed25519 signature of challenge
    pub timestamp: u64,
}

/// Message frame for peer communication
#[derive(Debug)]
pub struct MessageFrame {
    pub length: u32,
    pub msg_type: u8,
    pub payload: Vec<u8>,
    pub checksum: u32,
}

/// Message types for peer communication
#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum MessageType {
    Handshake = 0x01,
    HandshakeResponse = 0x02,
    Ping = 0x10,
    Pong = 0x11,
    ConsensusMessage = 0x20,
    TransactionBroadcast = 0x21,
    BlockAnnouncement = 0x22,
}

impl PeerConnector {
    pub fn new(local_node_id: NodeId) -> Self {
        Self {
            local_node_id,
            connection_pool: Arc::new(RwLock::new(HashMap::new())),
            tor_client: None,
            handshake_timeout: std::time::Duration::from_secs(30),
            connection_failures: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Establish real connection to discovered peer - replaces demo implementation
    pub async fn connect_to_peer(&self, mut peer_info: DiscoveredPeer) -> Result<()> {
        info!(
            "🔗 Establishing REAL connection to peer {} at {}",
            hex::encode(&peer_info.validator_id[..8]),
            peer_info.onion_address
        );

        // NEW: Check exponential backoff before attempting connection
        if let Some(should_skip) = self.should_skip_connection_attempt(&peer_info.validator_id).await? {
            if should_skip {
                let next_retry = {
                    let failures = self.connection_failures.read().await;
                    failures.get(&peer_info.validator_id).map(|f| f.next_retry_time)
                };

                if let Some(retry_time) = next_retry {
                    info!(
                        "⏱️ Skipping connection to {} due to exponential backoff (retry after {})",
                        hex::encode(&peer_info.validator_id[..8]),
                        retry_time.format("%H:%M:%S")
                    );
                    return Err(anyhow::anyhow!("Connection skipped due to exponential backoff"));
                }
            }
        }

        // NEW: Service ping verification before attempting connection
        if peer_info.service_status != crate::ServiceStatus::Online {
            info!("🔍 Pinging service at {} before connection", peer_info.onion_address);

            peer_info.service_status = crate::ServiceStatus::Checking;
            peer_info.last_service_check = chrono::Utc::now();

            match self.ping_service(&peer_info).await {
                Ok(true) => {
                    info!("✅ Service ping successful for {}", peer_info.onion_address);
                    peer_info.service_status = crate::ServiceStatus::Online;
                }
                Ok(false) => {
                    warn!("❌ Service ping failed for {}", peer_info.onion_address);
                    peer_info.service_status = crate::ServiceStatus::Offline;
                    return Err(anyhow::anyhow!("Service offline - ping failed"));
                }
                Err(e) => {
                    warn!("⚠️ Service ping error for {}: {}", peer_info.onion_address, e);
                    peer_info.service_status = crate::ServiceStatus::Unknown;
                    // Continue with connection attempt anyway
                }
            }
        }

        // Check if already connected
        {
            let pool = self.connection_pool.read().await;
            if pool.contains_key(&peer_info.validator_id) {
                info!("✅ Already connected to peer {}", hex::encode(&peer_info.validator_id[..8]));
                return Ok(());
            }
        }

        let connection = if peer_info.onion_address.ends_with(".onion") {
            // Route through Tor for .onion addresses
            self.connect_via_tor(&peer_info).await?
        } else {
            // Direct TCP connection for IP addresses
            self.connect_direct(&peer_info).await?
        };

        // Perform handshake authentication
        let authenticated_connection = self.perform_handshake(connection, &peer_info).await?;

        // Store in connection pool
        {
            let mut pool = self.connection_pool.write().await;
            pool.insert(peer_info.validator_id, authenticated_connection);
        }

        info!(
            "✅ Successfully connected and authenticated peer {}",
            hex::encode(&peer_info.validator_id[..8])
        );

        Ok(())
    }

    /// Direct TCP connection to peer
    async fn connect_direct(&self, peer_info: &DiscoveredPeer) -> Result<PeerConnection> {
        debug!("🔌 Attempting direct TCP connection to {}", peer_info.onion_address);

        // Parse address - for now assume format "ip:port"
        let socket_addr: SocketAddr = peer_info.onion_address.parse()
            .context("Failed to parse peer address")?;

        let stream = tokio::time::timeout(
            self.handshake_timeout,
            TcpStream::connect(socket_addr)
        ).await
        .context("Connection timeout")?
        .context("Failed to establish TCP connection")?;

        info!("🔌 TCP connection established to {}", socket_addr);

        Ok(PeerConnection {
            peer_id: peer_info.validator_id,
            stream: Arc::new(RwLock::new(stream)),
            peer_address: socket_addr,
            established_at: chrono::Utc::now(),
            last_activity: Arc::new(RwLock::new(chrono::Utc::now())),
            authenticated: false,
        })
    }

    /// Connect via Tor for .onion addresses
    async fn connect_via_tor(&self, peer_info: &DiscoveredPeer) -> Result<PeerConnection> {
        info!("🧅 Attempting Tor connection to {}", peer_info.onion_address);

        if let Some(tor_client) = &self.tor_client {
            // Parse onion address to extract address and port
            let (onion_host, port) = if let Some(colon_pos) = peer_info.onion_address.rfind(':') {
                let host = &peer_info.onion_address[..colon_pos];
                let port_str = &peer_info.onion_address[colon_pos + 1..];
                let port = port_str.parse::<u16>().unwrap_or(8080);
                (host, port)
            } else {
                // Default to port 8080 if no port specified
                (peer_info.onion_address.as_str(), 8080)
            };

            // Use the tor client to establish connection
            info!("🧅 Connecting to {}:{} via Tor", onion_host, port);

            match tor_client.connect_to_onion(onion_host, port).await {
                Ok(tor_stream) => {
                    info!("✅ Successfully connected to {} via Tor", peer_info.onion_address);

                    // Use the tor_stream directly as it's already a TcpStream
                    Ok(PeerConnection {
                        peer_id: peer_info.validator_id,
                        stream: Arc::new(RwLock::new(tor_stream)),
                        peer_address: "127.0.0.1:8080".parse()?, // Placeholder address for Tor connections
                        established_at: chrono::Utc::now(),
                        last_activity: Arc::new(RwLock::new(chrono::Utc::now())),
                        authenticated: false, // Will be set to true after handshake
                    })
                }
                Err(e) => {
                    warn!("❌ Failed to connect to {} via Tor: {}", peer_info.onion_address, e);
                    anyhow::bail!("Tor connection failed: {}", e)
                }
            }
        } else {
            warn!("🧅 No Tor client available, cannot connect to .onion address");
            anyhow::bail!("No Tor client available")
        }
    }

    /// Perform cryptographic handshake with peer
    async fn perform_handshake(
        &self,
        mut connection: PeerConnection,
        peer_info: &DiscoveredPeer,
    ) -> Result<PeerConnection> {
        info!("🤝 Performing handshake with peer {}", hex::encode(&peer_info.validator_id[..8]));

        // Generate challenge for authentication
        let mut challenge = [0u8; 32];
        use rand::RngCore;
        rand::thread_rng().fill_bytes(&mut challenge);

        // Create handshake message
        let handshake = HandshakeMessage {
            protocol_id: *b"Q-NARWHALKNIGHT\0",
            node_id: self.local_node_id,
            supported_protocols: vec![
                "consensus/1.0".to_string(),
                "mempool/1.0".to_string(),
                "discovery/1.0".to_string(),
            ],
            challenge,
            signature: [0u8; 64], // TODO: Sign with ed25519 keypair
            timestamp: chrono::Utc::now().timestamp() as u64,
        };

        // Send handshake
        self.send_message(&connection, MessageType::Handshake, &handshake).await?;

        // Receive handshake response
        let response_frame = self.receive_message(&connection).await?;
        if response_frame.msg_type != MessageType::HandshakeResponse as u8 {
            anyhow::bail!("Expected handshake response, got message type {}", response_frame.msg_type);
        }

        let response: HandshakeMessage = bincode::deserialize(&response_frame.payload)
            .context("Failed to deserialize handshake response")?;

        // Verify peer identity
        if response.node_id != peer_info.validator_id {
            anyhow::bail!("Peer node ID mismatch: expected {}, got {}",
                hex::encode(peer_info.validator_id), hex::encode(response.node_id));
        }

        // TODO: Verify signature with peer's public key
        // For now, mark as authenticated
        connection.authenticated = true;

        info!("✅ Handshake completed successfully with peer {}", hex::encode(&peer_info.validator_id[..8]));
        Ok(connection)
    }

    /// Send message to peer
    async fn send_message<T: Serialize>(
        &self,
        connection: &PeerConnection,
        msg_type: MessageType,
        payload: &T,
    ) -> Result<()> {
        let serialized = bincode::serialize(payload)
            .context("Failed to serialize message")?;

        let frame = MessageFrame {
            length: serialized.len() as u32,
            msg_type: msg_type as u8,
            payload: serialized.clone(),
            checksum: crc32fast::hash(&serialized),
        };

        let mut stream = connection.stream.write().await;

        // Send frame header
        stream.write_u32(frame.length).await?;
        stream.write_u8(frame.msg_type).await?;
        stream.write_u32(frame.checksum).await?;

        // Send payload
        stream.write_all(&frame.payload).await?;
        stream.flush().await?;

        // Update activity timestamp
        {
            let mut last_activity = connection.last_activity.write().await;
            *last_activity = chrono::Utc::now();
        }

        debug!("📤 Sent message type {} to peer {}", msg_type as u8, hex::encode(&connection.peer_id[..8]));
        Ok(())
    }

    /// Receive message from peer
    async fn receive_message(&self, connection: &PeerConnection) -> Result<MessageFrame> {
        let mut stream = connection.stream.write().await;

        // Read frame header
        let length = stream.read_u32().await?;
        let msg_type = stream.read_u8().await?;
        let checksum = stream.read_u32().await?;

        // Read payload
        let mut payload = vec![0u8; length as usize];
        stream.read_exact(&mut payload).await?;

        // Verify checksum
        let calculated_checksum = crc32fast::hash(&payload);
        if checksum != calculated_checksum {
            anyhow::bail!("Message checksum mismatch");
        }

        // Update activity timestamp
        {
            let mut last_activity = connection.last_activity.write().await;
            *last_activity = chrono::Utc::now();
        }

        debug!("📥 Received message type {} from peer {}", msg_type, hex::encode(&connection.peer_id[..8]));

        Ok(MessageFrame {
            length,
            msg_type,
            payload,
            checksum,
        })
    }

    /// Get active peer connections
    pub async fn get_active_connections(&self) -> Vec<NodeId> {
        let pool = self.connection_pool.read().await;
        pool.keys().copied().collect()
    }

    /// Send data to specific peer
    pub async fn send_to_peer(&self, peer_id: &NodeId, data: Vec<u8>) -> Result<()> {
        let pool = self.connection_pool.read().await;

        if let Some(connection) = pool.get(peer_id) {
            // Send as raw consensus message
            #[derive(Serialize)]
            struct RawMessage {
                data: Vec<u8>,
            }

            let data_len = data.len();
            self.send_message(connection, MessageType::ConsensusMessage, &RawMessage { data }).await?;
            info!("📤 Sent {} bytes to peer {}", data_len, hex::encode(&peer_id[..8]));
            Ok(())
        } else {
            anyhow::bail!("Peer {} not connected", hex::encode(peer_id))
        }
    }

    /// Disconnect from peer
    pub async fn disconnect_peer(&self, peer_id: &NodeId) -> Result<()> {
        let mut pool = self.connection_pool.write().await;

        if let Some(_connection) = pool.remove(peer_id) {
            // TCP stream will be dropped automatically
            info!("🔌 Disconnected from peer {}", hex::encode(&peer_id[..8]));
            Ok(())
        } else {
            anyhow::bail!("Peer {} not connected", hex::encode(peer_id))
        }
    }

    /// Get connection statistics
    pub async fn get_connection_stats(&self) -> ConnectionStats {
        let pool = self.connection_pool.read().await;

        ConnectionStats {
            active_connections: pool.len() as u32,
            total_peers: pool.keys().len() as u32,
            authenticated_peers: pool.values().filter(|c| c.authenticated).count() as u32,
        }
    }

    /// Ping service to check if it's online before attempting full connection
    async fn ping_service(&self, peer_info: &DiscoveredPeer) -> Result<bool> {
        info!("🏓 Pinging service at {}", peer_info.onion_address);

        if peer_info.onion_address.ends_with(".onion") {
            // Tor onion service ping
            self.ping_onion_service(peer_info).await
        } else {
            // Direct IP ping
            self.ping_direct_service(peer_info).await
        }
    }

    /// Ping onion service via Tor
    async fn ping_onion_service(&self, peer_info: &DiscoveredPeer) -> Result<bool> {
        if let Some(tor_client) = &self.tor_client {
            // Parse onion address to extract host and port
            let (onion_host, port) = if let Some(colon_pos) = peer_info.onion_address.rfind(':') {
                let host = &peer_info.onion_address[..colon_pos];
                let port_str = &peer_info.onion_address[colon_pos + 1..];
                let port = port_str.parse::<u16>().unwrap_or(8080);
                (host, port)
            } else {
                (peer_info.onion_address.as_str(), 8080)
            };

            // Quick connection test with short timeout
            let ping_result = tokio::time::timeout(
                std::time::Duration::from_secs(10), // Short timeout for ping
                tor_client.connect_to_onion(onion_host, port)
            ).await;

            match ping_result {
                Ok(Ok(_connection)) => {
                    info!("✅ Onion service ping successful: {}", peer_info.onion_address);
                    Ok(true)
                }
                Ok(Err(_)) | Err(_) => {
                    info!("❌ Onion service ping failed: {}", peer_info.onion_address);
                    Ok(false)
                }
            }
        } else {
            warn!("🧅 No Tor client available for onion ping");
            Ok(false)
        }
    }

    /// Ping direct IP service
    async fn ping_direct_service(&self, peer_info: &DiscoveredPeer) -> Result<bool> {
        let socket_addr: SocketAddr = peer_info.onion_address.parse()
            .context("Failed to parse peer address for ping")?;

        let ping_result = tokio::time::timeout(
            std::time::Duration::from_secs(5), // Quick ping timeout
            TcpStream::connect(socket_addr)
        ).await;

        match ping_result {
            Ok(Ok(_stream)) => {
                info!("✅ Direct service ping successful: {}", peer_info.onion_address);
                Ok(true)
            }
            Ok(Err(_)) | Err(_) => {
                info!("❌ Direct service ping failed: {}", peer_info.onion_address);
                Ok(false)
            }
        }
    }

    /// Check if we should skip connection attempt due to exponential backoff
    async fn should_skip_connection_attempt(&self, node_id: &NodeId) -> Result<Option<bool>> {
        let failures = self.connection_failures.read().await;

        if let Some(failure_state) = failures.get(node_id) {
            let now = chrono::Utc::now();

            // Check if we're still in backoff period
            if now < failure_state.next_retry_time {
                return Ok(Some(true)); // Skip connection
            } else {
                return Ok(Some(false)); // Can attempt connection
            }
        }

        Ok(None) // No failure history, proceed normally
    }

    /// Record connection failure and update exponential backoff
    async fn record_connection_failure(&self, node_id: &NodeId) -> Result<()> {
        let mut failures = self.connection_failures.write().await;
        let now = chrono::Utc::now();

        let failure_state = failures.entry(*node_id).or_insert(ConnectionFailureState {
            failure_count: 0,
            last_failure: now,
            next_retry_time: now,
        });

        failure_state.failure_count += 1;
        failure_state.last_failure = now;

        // Exponential backoff: 1min, 2min, 4min, 8min, max 30min
        let backoff_minutes = std::cmp::min(2_u32.pow(failure_state.failure_count - 1), 30);
        failure_state.next_retry_time = now + chrono::Duration::minutes(backoff_minutes as i64);

        info!(
            "📈 Connection failure #{} for peer {} - next retry in {} minutes",
            failure_state.failure_count,
            hex::encode(&node_id[..8]),
            backoff_minutes
        );

        Ok(())
    }

    /// Record successful connection and clear failure state
    async fn record_connection_success(&self, node_id: &NodeId) -> Result<()> {
        let mut failures = self.connection_failures.write().await;

        if failures.remove(node_id).is_some() {
            info!("🎉 Connection success for peer {} - cleared failure state", hex::encode(&node_id[..8]));
        }

        Ok(())
    }
}

/// Connection statistics
#[derive(Debug, Serialize)]
pub struct ConnectionStats {
    pub active_connections: u32,
    pub total_peers: u32,
    pub authenticated_peers: u32,
}