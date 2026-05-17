/// P2P handshake protocol for quantum consensus network
///
/// Implements the handshake protocol for establishing authenticated connections
/// between Q-NarwhalKnight nodes across different servers.
///
/// v2.5.0-beta: Added Ed25519 signature support for authenticated handshakes
use anyhow::{anyhow, Result};
use ed25519_dalek::{Signature, SigningKey, Signer, VerifyingKey, Verifier};
use rand::Rng;
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::sync::Arc;
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
    /// v8.6.2: Server Delta (1Gbit dedicated, bootstrap node)
    Delta,
    /// v8.6.2: Server Epsilon (10Gbit supernode, priority bootstrap)
    Epsilon,
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
#[derive(Clone)]
pub struct LocalNodeInfo {
    pub node_id: String,
    pub server_role: ServerRole,
    pub capabilities: Vec<String>,
    /// v2.5.0-beta: Ed25519 signing key for authenticated handshakes
    pub signing_key: Option<Arc<SigningKey>>,
}

impl std::fmt::Debug for LocalNodeInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalNodeInfo")
            .field("node_id", &self.node_id)
            .field("server_role", &self.server_role)
            .field("capabilities", &self.capabilities)
            .field("has_signing_key", &self.signing_key.is_some())
            .finish()
    }
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
            signing_key: None,
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
            signing_key: None,
        }
    }

    /// v2.5.0-beta: Create node with signing key for authenticated handshakes
    pub fn with_signing_key(mut self, signing_key: Arc<SigningKey>) -> Self {
        self.signing_key = Some(signing_key);
        self
    }

    /// v2.5.0-beta: Generate a new signing key for this node
    pub fn generate_signing_key(&mut self) {
        use sha3::{Digest, Sha3_256};
        let mut hasher = Sha3_256::new();
        hasher.update(b"handshake-signing-key-v2.5.0");
        hasher.update(self.node_id.as_bytes());
        hasher.update(&std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
            .to_le_bytes());
        let key_bytes: [u8; 32] = hasher.finalize().into();
        self.signing_key = Some(Arc::new(SigningKey::from_bytes(&key_bytes)));
    }
}

/// v2.5.0-beta: Sign a handshake message using Ed25519
fn sign_handshake_message(
    message_type: &str,
    node_id: &str,
    challenge: &[u8],
    timestamp: u64,
    signing_key: Option<&Arc<SigningKey>>,
) -> Option<String> {
    if let Some(key) = signing_key {
        // Create canonical data to sign: message_type + node_id + challenge + timestamp
        let mut sign_data = Vec::with_capacity(128);
        sign_data.extend_from_slice(message_type.as_bytes());
        sign_data.extend_from_slice(node_id.as_bytes());
        sign_data.extend_from_slice(challenge);
        sign_data.extend_from_slice(&timestamp.to_le_bytes());

        let signature = key.sign(&sign_data);
        Some(hex::encode(signature.to_bytes()))
    } else {
        warn!("⚠️ No signing key provided for handshake - message will be unsigned");
        None
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
    let timestamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    // v2.5.0-beta: Sign the handshake message
    let signature = sign_handshake_message(
        "handshake_request",
        &local_node.node_id,
        &challenge,
        timestamp,
        local_node.signing_key.as_ref(),
    );

    // Create handshake message
    let handshake = HandshakeMessage {
        message_type: "handshake_request".to_string(),
        node_id: local_node.node_id.clone(),
        server_role: local_node.server_role.clone(),
        protocol_version: PROTOCOL_VERSION,
        capabilities: local_node.capabilities.clone(),
        challenge: challenge.clone(),
        timestamp,
        signature, // v2.5.0-beta: Now signed if signing key available
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
    let ack_timestamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    // v2.5.0-beta: Sign the acknowledgment
    let ack_signature = sign_handshake_message(
        "handshake_ack",
        &local_node.node_id,
        &response.challenge,
        ack_timestamp,
        local_node.signing_key.as_ref(),
    );

    let ack = HandshakeMessage {
        message_type: "handshake_ack".to_string(),
        node_id: local_node.node_id.clone(),
        server_role: local_node.server_role.clone(),
        protocol_version: PROTOCOL_VERSION,
        capabilities: local_node.capabilities.clone(),
        challenge: response.challenge.clone(), // Echo their challenge
        timestamp: ack_timestamp,
        signature: ack_signature, // v2.5.0-beta: Signed
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
    let response_timestamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    // v2.5.0-beta: Sign the response
    let response_signature = sign_handshake_message(
        "handshake_response",
        &local_node.node_id,
        &response_challenge,
        response_timestamp,
        local_node.signing_key.as_ref(),
    );

    // Create handshake response
    let response = HandshakeMessage {
        message_type: "handshake_response".to_string(),
        node_id: local_node.node_id.clone(),
        server_role: local_node.server_role.clone(),
        protocol_version: PROTOCOL_VERSION,
        capabilities: local_node.capabilities.clone(),
        challenge: response_challenge.clone(),
        timestamp: response_timestamp,
        signature: response_signature, // v2.5.0-beta: Signed
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
