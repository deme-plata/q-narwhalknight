/// P2P handshake protocol for quantum consensus network
///
/// Implements the handshake protocol for establishing authenticated connections
/// between Q-NarwhalKnight nodes across different servers.
use anyhow::{anyhow, Result};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::time::{Duration, SystemTime};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::time::timeout;
use tracing::{debug, error, info, warn};

/// Protocol version
pub const PROTOCOL_VERSION: u32 = 1;

/// Handshake message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandshakeMessage {
    pub message_type: String,
    pub node_id: String,
    pub server_role: ServerRole,
    pub protocol_version: u32,
    pub capabilities: Vec<String>,
    pub challenge: Vec<u8>,
    pub timestamp: u64,
    pub signature: Option<String>,
}

/// Server role
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ServerRole {
    Alpha,
    Beta,
    Gamma,
    Unknown,
}

/// Node capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Capability {
    DnsPhantom,
    TorSupport,
    QuantumConsensus,
    DagBft,
    PostQuantumCrypto,
}

/// Remote peer information after handshake
#[derive(Debug, Clone)]
pub struct RemotePeerInfo {
    pub node_id: String,
    pub server_role: ServerRole,
    pub protocol_version: u32,
    pub capabilities: Vec<String>,
    pub connected_at: SystemTime,
}

/// Handshake error types
#[derive(Debug, thiserror::Error)]
pub enum HandshakeError {
    #[error("Incompatible protocol version: expected {expected}, got {received}")]
    IncompatibleVersion { expected: u32, received: u32 },

    #[error("Authentication failed")]
    AuthenticationFailed,

    #[error("Timeout during handshake")]
    Timeout,

    #[error("Invalid handshake message: {0}")]
    InvalidMessage(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}

/// Local node information
#[derive(Debug, Clone)]
pub struct LocalNodeInfo {
    pub node_id: String,
    pub server_role: ServerRole,
    pub capabilities: Vec<String>,
}

impl LocalNodeInfo {
    pub fn new_alpha_node() -> Self {
        Self {
            node_id: format!("alpha-{}", generate_node_id()),
            server_role: ServerRole::Alpha,
            capabilities: vec![
                "dns-phantom".to_string(),
                "tor-support".to_string(),
                "quantum-consensus".to_string(),
                "zero-config-discovery".to_string(),
            ],
        }
    }

    pub fn new_beta_node() -> Self {
        Self {
            node_id: format!("beta-{}", generate_node_id()),
            server_role: ServerRole::Beta,
            capabilities: vec![
                "dns-phantom".to_string(),
                "tor-support".to_string(),
                "quantum-consensus".to_string(),
                "mesh-coordination".to_string(),
            ],
        }
    }
}

/// Perform handshake as initiator (Alpha node)
pub async fn perform_client_handshake(
    stream: &mut TcpStream,
    local_node: &LocalNodeInfo,
) -> Result<RemotePeerInfo, HandshakeError> {
    info!("🤝 Starting client handshake as: {}", local_node.node_id);

    // Generate challenge
    let challenge = generate_challenge();

    // Create handshake message
    let handshake = HandshakeMessage {
        message_type: "handshake_request".to_string(),
        node_id: local_node.node_id.clone(),
        server_role: local_node.server_role.clone(),
        protocol_version: PROTOCOL_VERSION,
        capabilities: local_node.capabilities.clone(),
        challenge: challenge.clone(),
        timestamp: SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        signature: None, // TODO: Implement cryptographic signature
    };

    // Send handshake request
    send_handshake_message(stream, &handshake).await?;
    info!("📤 Sent handshake request to peer");

    // Receive handshake response
    let response = receive_handshake_message(stream).await?;
    info!("📨 Received handshake response from: {}", response.node_id);

    // Verify handshake response
    verify_handshake_response(&handshake, &response)?;

    // Send handshake acknowledgment
    let ack = HandshakeMessage {
        message_type: "handshake_ack".to_string(),
        node_id: local_node.node_id.clone(),
        server_role: local_node.server_role.clone(),
        protocol_version: PROTOCOL_VERSION,
        capabilities: local_node.capabilities.clone(),
        challenge: response.challenge.clone(), // Echo their challenge
        timestamp: SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        signature: None,
    };

    send_handshake_message(stream, &ack).await?;
    info!("📤 Sent handshake acknowledgment");

    info!(
        "✅ Client handshake completed successfully with {} node: {}",
        format!("{:?}", response.server_role),
        response.node_id
    );

    Ok(RemotePeerInfo {
        node_id: response.node_id,
        server_role: response.server_role,
        protocol_version: response.protocol_version,
        capabilities: response.capabilities,
        connected_at: SystemTime::now(),
    })
}

/// Perform handshake as responder (Server Beta)
pub async fn perform_server_handshake(
    stream: &mut TcpStream,
    local_node: &LocalNodeInfo,
) -> Result<RemotePeerInfo, HandshakeError> {
    info!("🤝 Starting server handshake as: {}", local_node.node_id);

    // Receive handshake request
    let request = receive_handshake_message(stream).await?;
    info!("📨 Received handshake request from: {}", request.node_id);

    // Verify handshake request
    verify_handshake_request(&request)?;

    // Generate response challenge
    let response_challenge = generate_challenge();

    // Create handshake response
    let response = HandshakeMessage {
        message_type: "handshake_response".to_string(),
        node_id: local_node.node_id.clone(),
        server_role: local_node.server_role.clone(),
        protocol_version: PROTOCOL_VERSION,
        capabilities: local_node.capabilities.clone(),
        challenge: response_challenge.clone(),
        timestamp: SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        signature: None,
    };

    // Send handshake response
    send_handshake_message(stream, &response).await?;
    info!("📤 Sent handshake response");

    // Receive handshake acknowledgment
    let ack = receive_handshake_message(stream).await?;
    info!("📨 Received handshake acknowledgment");

    // Verify acknowledgment
    verify_handshake_ack(&response, &ack)?;

    info!(
        "✅ Server handshake completed successfully with {} node: {}",
        format!("{:?}", request.server_role),
        request.node_id
    );

    Ok(RemotePeerInfo {
        node_id: request.node_id,
        server_role: request.server_role,
        protocol_version: request.protocol_version,
        capabilities: request.capabilities,
        connected_at: SystemTime::now(),
    })
}

/// Send handshake message
async fn send_handshake_message(
    stream: &mut TcpStream,
    message: &HandshakeMessage,
) -> Result<(), HandshakeError> {
    let json = serde_json::to_string(message)?;
    let message_bytes = format!("{}\n", json).into_bytes();

    stream.write_all(&message_bytes).await?;
    stream.flush().await?;

    debug!("📤 Sent handshake message: {}", message.message_type);
    Ok(())
}

/// Receive handshake message
async fn receive_handshake_message(
    stream: &mut TcpStream,
) -> Result<HandshakeMessage, HandshakeError> {
    // Read with timeout
    let mut buffer = Vec::new();
    let mut single_byte = [0u8; 1];

    let receive_future = async {
        loop {
            stream.read_exact(&mut single_byte).await?;
            let byte = single_byte[0];

            if byte == b'\n' {
                break;
            }

            buffer.push(byte);

            // Prevent excessive memory usage
            if buffer.len() > 8192 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Message too long",
                ));
            }
        }
        Ok::<(), std::io::Error>(())
    };

    timeout(Duration::from_secs(30), receive_future)
        .await
        .map_err(|_| HandshakeError::Timeout)??;

    let message_str = String::from_utf8_lossy(&buffer);
    let message: HandshakeMessage = serde_json::from_str(&message_str)?;

    debug!("📨 Received handshake message: {}", message.message_type);
    Ok(message)
}

/// Verify handshake request
fn verify_handshake_request(request: &HandshakeMessage) -> Result<(), HandshakeError> {
    if request.protocol_version != PROTOCOL_VERSION {
        return Err(HandshakeError::IncompatibleVersion {
            expected: PROTOCOL_VERSION,
            received: request.protocol_version,
        });
    }

    if request.message_type != "handshake_request" {
        return Err(HandshakeError::InvalidMessage(format!(
            "Expected handshake_request, got: {}",
            request.message_type
        )));
    }

    Ok(())
}

/// Verify handshake response
fn verify_handshake_response(
    request: &HandshakeMessage,
    response: &HandshakeMessage,
) -> Result<(), HandshakeError> {
    if response.protocol_version != PROTOCOL_VERSION {
        return Err(HandshakeError::IncompatibleVersion {
            expected: PROTOCOL_VERSION,
            received: response.protocol_version,
        });
    }

    if response.message_type != "handshake_response" {
        return Err(HandshakeError::InvalidMessage(format!(
            "Expected handshake_response, got: {}",
            response.message_type
        )));
    }

    Ok(())
}

/// Verify handshake acknowledgment
fn verify_handshake_ack(
    response: &HandshakeMessage,
    ack: &HandshakeMessage,
) -> Result<(), HandshakeError> {
    if ack.message_type != "handshake_ack" {
        return Err(HandshakeError::InvalidMessage(format!(
            "Expected handshake_ack, got: {}",
            ack.message_type
        )));
    }

    // Verify challenge echo
    if ack.challenge != response.challenge {
        return Err(HandshakeError::AuthenticationFailed);
    }

    Ok(())
}

/// Generate random challenge
fn generate_challenge() -> Vec<u8> {
    let mut rng = rand::thread_rng();
    (0..32).map(|_| rng.gen::<u8>()).collect()
}

/// Generate node ID
fn generate_node_id() -> String {
    let mut rng = rand::thread_rng();
    format!("{:08x}", rng.gen::<u32>())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_handshake_message_serialization() {
        let message = HandshakeMessage {
            message_type: "test".to_string(),
            node_id: "test-node".to_string(),
            server_role: ServerRole::Alpha,
            protocol_version: 1,
            capabilities: vec!["test".to_string()],
            challenge: vec![1, 2, 3, 4],
            timestamp: 1234567890,
            signature: None,
        };

        let json = serde_json::to_string(&message).unwrap();
        let deserialized: HandshakeMessage = serde_json::from_str(&json).unwrap();

        assert_eq!(message.node_id, deserialized.node_id);
        assert_eq!(message.protocol_version, deserialized.protocol_version);
    }
}
