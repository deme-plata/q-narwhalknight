/// Handshake Validator Module - v1.0.15.1-beta
///
/// Implements protocol version validation for network handshakes.
/// Kimi AI Recommendation: Reject peers with incompatible protocol versions to prevent silent failures.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::io;
use tracing::{debug, info, warn};

// libp2p imports for request-response protocol
use async_trait::async_trait;
use futures::prelude::*;
use futures::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use libp2p::request_response::Codec;
use unsigned_varint::{aio, encode};  // For length-prefixed encoding

/// Protocol version for Q-NarwhalKnight network
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ProtocolVersion {
    pub major: u16,
    pub minor: u16,
    pub patch: u16,
}

impl ProtocolVersion {
    /// Current protocol version for v1.0.15.1-beta
    pub const CURRENT: ProtocolVersion = ProtocolVersion {
        major: 1,
        minor: 0,
        patch: 15,
    };

    /// Create a new protocol version
    pub const fn new(major: u16, minor: u16, patch: u16) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Check if this version is compatible with another version
    /// Compatible if major versions match and minor is within 1 step
    pub fn is_compatible_with(&self, other: &ProtocolVersion) -> bool {
        // Major version must match exactly
        if self.major != other.major {
            return false;
        }

        // Minor version can differ by at most 1
        let minor_diff = if self.minor > other.minor {
            self.minor - other.minor
        } else {
            other.minor - self.minor
        };

        minor_diff <= 1
    }

    /// Parse version from string like "1.0.15"
    pub fn from_string(s: &str) -> Result<Self> {
        let parts: Vec<&str> = s.split('.').collect();
        if parts.len() != 3 {
            return Err(anyhow!("Invalid version format: {}", s));
        }

        Ok(Self {
            major: parts[0].parse()?,
            minor: parts[1].parse()?,
            patch: parts[2].parse()?,
        })
    }
}

impl fmt::Display for ProtocolVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// Handshake message exchanged during peer connection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandshakeMessage {
    /// Protocol version
    pub protocol_version: ProtocolVersion,

    /// Network ID (e.g., "mainnet", "testnet-phase8")
    pub network_id: String,

    /// Node version string
    pub node_version: String,

    /// Features supported by this node
    pub features: Vec<String>,

    /// Genesis block hash for network verification
    pub genesis_hash: Vec<u8>,

    /// v8.4.0: Self-reported bandwidth tier (Mbps) for sync peer selection
    /// 0 = unknown. Used by gravity-assist to prefer high-bandwidth peers.
    #[serde(default)]
    pub bandwidth_tier_mbps: u32,
}

impl HandshakeMessage {
    /// Create a new handshake message
    pub fn new(network_id: String, node_version: String, genesis_hash: Vec<u8>) -> Self {
        Self {
            protocol_version: ProtocolVersion::CURRENT,
            network_id,
            node_version,
            features: vec![
                "turbo-sync".to_string(),
                "batch-sync".to_string(),
                "pointer-integrity".to_string(),
                "memory-limiting".to_string(),
            ],
            genesis_hash,
            // v8.4.0: Self-reported bandwidth for sync peer selection
            bandwidth_tier_mbps: std::env::var("Q_BANDWIDTH_MBPS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
        }
    }
}

/// Handshake validation result
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HandshakeResult {
    /// Handshake successful
    Success,

    /// Incompatible protocol version
    IncompatibleProtocol {
        ours: ProtocolVersion,
        theirs: ProtocolVersion,
    },

    /// Wrong network ID
    WrongNetwork {
        ours: String,
        theirs: String,
    },

    /// Genesis hash mismatch
    GenesisMismatch,

    /// Missing required features
    MissingFeatures {
        required: Vec<String>,
    },
}

/// Handshake validator for protocol compatibility
pub struct HandshakeValidator {
    /// Our protocol version
    our_version: ProtocolVersion,

    /// Our network ID
    our_network_id: String,

    /// Our genesis hash
    our_genesis_hash: Vec<u8>,

    /// Required features for peers
    required_features: Vec<String>,
}

impl HandshakeValidator {
    /// Create a new handshake validator
    pub fn new(network_id: String, genesis_hash: Vec<u8>) -> Self {
        info!("🤝 [HANDSHAKE] Validator initialized");
        info!("   Protocol: v{}", ProtocolVersion::CURRENT);
        info!("   Network: {}", network_id);
        info!("   Genesis: {}", hex::encode(&genesis_hash[..8.min(genesis_hash.len())]));

        Self {
            our_version: ProtocolVersion::CURRENT,
            our_network_id: network_id,
            our_genesis_hash: genesis_hash,
            required_features: vec![], // No features are strictly required for now
        }
    }

    /// Validate an incoming handshake message
    pub fn validate_handshake(&self, peer_handshake: &HandshakeMessage) -> HandshakeResult {
        // Check protocol version compatibility
        if !self.our_version.is_compatible_with(&peer_handshake.protocol_version) {
            warn!(
                "❌ [HANDSHAKE] Incompatible protocol version: ours={}, theirs={}",
                self.our_version, peer_handshake.protocol_version
            );
            return HandshakeResult::IncompatibleProtocol {
                ours: self.our_version,
                theirs: peer_handshake.protocol_version,
            };
        }

        // Check network ID
        if self.our_network_id != peer_handshake.network_id {
            warn!(
                "❌ [HANDSHAKE] Wrong network: ours={}, theirs={}",
                self.our_network_id, peer_handshake.network_id
            );
            return HandshakeResult::WrongNetwork {
                ours: self.our_network_id.clone(),
                theirs: peer_handshake.network_id.clone(),
            };
        }

        // Check genesis hash
        if self.our_genesis_hash != peer_handshake.genesis_hash {
            warn!("❌ [HANDSHAKE] Genesis hash mismatch");
            return HandshakeResult::GenesisMismatch;
        }

        // Check required features
        let missing_features: Vec<String> = self
            .required_features
            .iter()
            .filter(|f| !peer_handshake.features.contains(f))
            .cloned()
            .collect();

        if !missing_features.is_empty() {
            warn!(
                "❌ [HANDSHAKE] Missing required features: {:?}",
                missing_features
            );
            return HandshakeResult::MissingFeatures {
                required: missing_features,
            };
        }

        debug!(
            "✅ [HANDSHAKE] Success: peer v{} on {}",
            peer_handshake.protocol_version, peer_handshake.network_id
        );

        HandshakeResult::Success
    }

    /// Generate our handshake message
    pub fn create_handshake(&self, node_version: String) -> HandshakeMessage {
        HandshakeMessage::new(
            self.our_network_id.clone(),
            node_version,
            self.our_genesis_hash.clone(),
        )
    }

    /// Update required features
    pub fn set_required_features(&mut self, features: Vec<String>) {
        self.required_features = features;
    }
}

/// libp2p Protocol for handshake exchange
pub type HandshakeProtocol = &'static str;

/// Protocol name constant
pub const HANDSHAKE_PROTOCOL: HandshakeProtocol = "/qnk/handshake/1.0.0";

/// Codec for encoding/decoding handshake messages over libp2p
#[derive(Clone, Default)]
pub struct HandshakeCodec;

#[async_trait]
impl Codec for HandshakeCodec {
    type Protocol = &'static str;
    type Request = HandshakeMessage;
    type Response = HandshakeResult;

    async fn read_request<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
    ) -> io::Result<Self::Request>
    where
        T: AsyncRead + Unpin + Send,
    {
        // Read length prefix
        let len = aio::read_usize(&mut *io).await
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        // Read message bytes
        let mut buf = vec![0u8; len];
        io.read_exact(&mut buf).await?;

        // Deserialize message
        bincode::deserialize(&buf)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    async fn read_response<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
    ) -> io::Result<Self::Response>
    where
        T: AsyncRead + Unpin + Send,
    {
        // Read length prefix
        let len = aio::read_usize(&mut *io).await
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        // Read message bytes
        let mut buf = vec![0u8; len];
        io.read_exact(&mut buf).await?;

        // Deserialize response
        bincode::deserialize(&buf)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    async fn write_request<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
        req: Self::Request,
    ) -> io::Result<()>
    where
        T: AsyncWrite + Unpin + Send,
    {
        // Serialize request
        let msg_bytes = bincode::serialize(&req)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        // Encode length as varint
        let mut len_buf = encode::usize_buffer();
        let len_bytes = encode::usize(msg_bytes.len(), &mut len_buf);

        // Write length prefix and message
        io.write_all(len_bytes).await?;
        io.write_all(&msg_bytes).await?;
        io.flush().await?;
        Ok(())
    }

    async fn write_response<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
        res: Self::Response,
    ) -> io::Result<()>
    where
        T: AsyncWrite + Unpin + Send,
    {
        // Serialize response
        let msg_bytes = bincode::serialize(&res)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        // Encode length as varint
        let mut len_buf = encode::usize_buffer();
        let len_bytes = encode::usize(msg_bytes.len(), &mut len_buf);

        // Write length prefix and message
        io.write_all(len_bytes).await?;
        io.write_all(&msg_bytes).await?;
        io.flush().await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_protocol_version_compatibility() {
        let v1_0_15 = ProtocolVersion::new(1, 0, 15);
        let v1_0_14 = ProtocolVersion::new(1, 0, 14);
        let v1_1_0 = ProtocolVersion::new(1, 1, 0);
        let v2_0_0 = ProtocolVersion::new(2, 0, 0);

        // Same major, minor within 1 step
        assert!(v1_0_15.is_compatible_with(&v1_0_14));
        assert!(v1_0_15.is_compatible_with(&v1_1_0));

        // Different major version
        assert!(!v1_0_15.is_compatible_with(&v2_0_0));

        // Same version
        assert!(v1_0_15.is_compatible_with(&v1_0_15));
    }

    #[test]
    fn test_protocol_version_parsing() {
        let v = ProtocolVersion::from_string("1.0.15").unwrap();
        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 0);
        assert_eq!(v.patch, 15);

        assert!(ProtocolVersion::from_string("invalid").is_err());
    }

    #[test]
    fn test_handshake_validation_success() {
        let genesis = vec![1, 2, 3, 4];
        let validator = HandshakeValidator::new("testnet".to_string(), genesis.clone());

        let peer_handshake = HandshakeMessage::new(
            "testnet".to_string(),
            "v1.0.15-beta".to_string(),
            genesis,
        );

        assert_eq!(
            validator.validate_handshake(&peer_handshake),
            HandshakeResult::Success
        );
    }

    #[test]
    fn test_handshake_validation_wrong_network() {
        let genesis = vec![1, 2, 3, 4];
        let validator = HandshakeValidator::new("mainnet-genesis".to_string(), genesis.clone());

        let peer_handshake = HandshakeMessage::new(
            "testnet".to_string(),
            "v1.0.15-beta".to_string(),
            genesis,
        );

        match validator.validate_handshake(&peer_handshake) {
            HandshakeResult::WrongNetwork { .. } => {},
            _ => panic!("Expected WrongNetwork result"),
        }
    }

    #[test]
    fn test_handshake_validation_genesis_mismatch() {
        let validator = HandshakeValidator::new("testnet".to_string(), vec![1, 2, 3, 4]);

        let peer_handshake = HandshakeMessage::new(
            "testnet".to_string(),
            "v1.0.15-beta".to_string(),
            vec![5, 6, 7, 8],
        );

        assert_eq!(
            validator.validate_handshake(&peer_handshake),
            HandshakeResult::GenesisMismatch
        );
    }

    #[test]
    fn test_handshake_validation_incompatible_protocol() {
        let genesis = vec![1, 2, 3, 4];
        let validator = HandshakeValidator::new("testnet".to_string(), genesis.clone());

        let mut peer_handshake = HandshakeMessage::new(
            "testnet".to_string(),
            "v2.0.0".to_string(),
            genesis,
        );
        peer_handshake.protocol_version = ProtocolVersion::new(2, 0, 0);

        match validator.validate_handshake(&peer_handshake) {
            HandshakeResult::IncompatibleProtocol { .. } => {},
            _ => panic!("Expected IncompatibleProtocol result"),
        }
    }
}
