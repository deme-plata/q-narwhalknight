/// Production Loopix Protocol Definitions
/// 
/// Defines secure network protocols for the Loopix mix network:
/// - Fixed-size cell protocol (traffic analysis resistance)
/// - Directory protocol with signed epochs
/// - Node information and routing structures

use libp2p::{
    request_response::{Codec, ResponseChannel},
    StreamProtocol, PeerId,
};
use serde::{Deserialize, Serialize};
use std::time::{Duration, SystemTime};
use futures::prelude::*;
use std::io;
use async_trait::async_trait;
use ring::rand::SecureRandom;
use crate::loopix_crypto::padding::CELL_SIZE;

/// Protocol identifiers
pub const LOOPIX_CELL_PROTO: &str = "/loopix/cell/1.0.0";
pub const LOOPIX_DIR_PROTO: &str = "/loopix/dir/1.0.0";

/// Fixed-size cell for traffic analysis resistance
/// 
/// All Loopix messages are transmitted in fixed-size cells to prevent
/// size-based traffic analysis. Cells are padded with cryptographically
/// secure random data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoopixCell {
    /// Cell payload (already padded to CELL_SIZE)
    pub bytes: Vec<u8>,
}

impl LoopixCell {
    /// Create a new cell from payload with secure padding
    pub fn new(payload: Vec<u8>) -> Result<Self, anyhow::Error> {
        if payload.len() > CELL_SIZE - 2 {
            return Err(anyhow::anyhow!("Payload too large for cell"));
        }
        
        let padded = crate::loopix_crypto::padding::pad_to_cell(&payload)?;
        Ok(LoopixCell { bytes: padded })
    }
    
    /// Extract payload from cell, removing padding
    pub fn extract_payload(&self) -> Result<Vec<u8>, anyhow::Error> {
        crate::loopix_crypto::padding::unpad_from_cell(&self.bytes)
            .map_err(|e| anyhow::anyhow!("Failed to unpad cell: {}", e))
    }
}

/// Directory protocol requests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DirectoryRequest {
    /// Register a node with the directory
    Register(NodeInfo),
    /// Fetch the current signed epoch descriptor
    FetchEpoch,
    /// Fetch nodes of a specific type
    FetchNodesByType(NodeType),
}

/// Directory protocol responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DirectoryResponse {
    /// Registration acknowledgment
    Ack,
    /// Current epoch descriptor with signature
    Epoch(EpochDescriptor),
    /// List of nodes
    Nodes(Vec<NodeInfo>),
    /// Error response
    Error(String),
}

/// Node type in the Loopix network
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeType {
    /// Mix node (routes and delays messages)
    Mix,
    /// Provider node (entry/exit point for clients)
    Provider,
    /// Client node (sends real messages and cover traffic)
    Client,
    /// Directory server (maintains network state)
    Directory,
}

/// Node information for directory registration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    /// Peer ID in the libp2p network
    pub peer_id: PeerId,
    /// Node type
    pub node_type: NodeType,
    /// Listen addresses
    pub listen_addrs: Vec<String>,
    /// Public key for this epoch (for client verification)
    pub public_key: Vec<u8>,
    /// Node capabilities/version info
    pub capabilities: Vec<String>,
    /// Registration timestamp
    #[serde(skip, default = "SystemTime::now")]
    pub registered_at: SystemTime,
}

/// Signed epoch descriptor from directory
/// 
/// Contains the current network topology signed by the directory authority.
/// Clients verify this signature before trusting mix node keys.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpochDescriptor {
    /// Mix nodes available in this epoch
    pub mix_nodes: Vec<NodeInfo>,
    /// Provider nodes available in this epoch  
    pub providers: Vec<NodeInfo>,
    /// Epoch sequence number
    pub epoch_id: u64,
    /// Epoch start time
    pub started_at: SystemTime,
    /// Epoch expiration time
    pub expires_at: SystemTime,
    /// Directory's signature over this epoch
    pub signature: Vec<u8>,
}

impl EpochDescriptor {
    /// Check if this epoch is currently valid
    pub fn is_valid(&self) -> bool {
        let now = SystemTime::now();
        now >= self.started_at && now < self.expires_at
    }
    
    /// Get all mix nodes suitable for routing
    pub fn available_mixes(&self) -> &[NodeInfo] {
        &self.mix_nodes
    }
    
    /// Get all provider nodes suitable for client connections
    pub fn available_providers(&self) -> &[NodeInfo] {
        &self.providers
    }
}

/// Loopix message types for different phases of communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoopixMessage {
    /// Real user message sent through the mix network
    UserMessage {
        /// Final recipient identifier
        recipient: String,
        /// Layered encrypted message payload
        encrypted_payload: Vec<u8>,
        /// Number of mix layers to traverse
        layer_count: u8,
    },
    
    /// Message being routed through mix nodes
    MixMessage {
        /// Next hop in the path
        next_hop: PeerId,
        /// Encrypted payload for next hop
        encrypted_payload: Vec<u8>,
        /// Processing delay hint (milliseconds)
        delay_hint: u64,
    },
    
    /// Cover traffic (dummy message for traffic analysis protection)
    CoverTraffic {
        /// Random payload indistinguishable from real traffic
        dummy_payload: Vec<u8>,
        /// Target path length to mimic
        path_length: u8,
    },
    
    /// Heartbeat for connection liveness
    Heartbeat {
        /// Timestamp
        timestamp: SystemTime,
    },
}

/// Fixed-size cell codec implementation
/// 
/// Ensures all network traffic uses identical cell sizes to prevent
/// traffic analysis based on message sizes.
#[derive(Clone, Default)]
pub struct LoopixCellCodec;

#[async_trait]
impl Codec for LoopixCellCodec {
    type Protocol = StreamProtocol;
    type Request = LoopixCell;
    type Response = (); // Empty response for one-way messaging

    async fn read_request<T>(&mut self, _: &Self::Protocol, io: &mut T) -> io::Result<Self::Request>
    where
        T: futures::AsyncRead + Unpin + Send,
    {
        let mut cell_bytes = vec![0u8; CELL_SIZE];
        io.read_exact(&mut cell_bytes).await?;
        
        Ok(LoopixCell { bytes: cell_bytes })
    }

    async fn read_response<T>(
        &mut self,
        _: &Self::Protocol,
        _io: &mut T
    ) -> io::Result<Self::Response>
    where
        T: futures::AsyncRead + Unpin + Send,
    {
        Ok(()) // Empty response
    }

    async fn write_request<T>(
        &mut self,
        _: &Self::Protocol,
        io: &mut T,
        req: Self::Request
    ) -> io::Result<()>
    where
        T: futures::AsyncWrite + Unpin + Send,
    {
        if req.bytes.len() != CELL_SIZE {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Cell size must be exactly {} bytes", CELL_SIZE)
            ));
        }

        io.write_all(&req.bytes).await
    }

    async fn write_response<T>(
        &mut self,
        _: &Self::Protocol,
        _io: &mut T,
        _res: Self::Response
    ) -> io::Result<()>
    where
        T: futures::AsyncWrite + Unpin + Send,
    {
        Ok(()) // Empty response, nothing to write
    }
}

/// Directory protocol codec
/// 
/// Handles variable-size directory requests/responses but still uses
/// fixed-size cells for transport.
#[derive(Clone, Default)]
pub struct DirectoryCodec;

#[async_trait]
impl Codec for DirectoryCodec {
    type Protocol = StreamProtocol;
    type Request = DirectoryRequest;
    type Response = DirectoryResponse;

    async fn read_request<T>(
        &mut self,
        _: &Self::Protocol,
        io: &mut T
    ) -> io::Result<Self::Request>
    where
        T: futures::AsyncRead + Unpin + Send,
    {
        let mut cell_bytes = vec![0u8; CELL_SIZE];
        io.read_exact(&mut cell_bytes).await?;
        
        // Extract length prefix
        let payload_len = u16::from_be_bytes([cell_bytes[0], cell_bytes[1]]) as usize;
        if payload_len > CELL_SIZE - 2 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Payload length exceeds cell capacity"
            ));
        }
        
        // Deserialize payload
        let payload = &cell_bytes[2..2 + payload_len];
        bincode::deserialize(payload)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    async fn read_response<T>(
        &mut self,
        _: &Self::Protocol,
        io: &mut T
    ) -> io::Result<Self::Response>
    where
        T: futures::AsyncRead + Unpin + Send,
    {
        let mut cell_bytes = vec![0u8; CELL_SIZE];
        io.read_exact(&mut cell_bytes).await?;
        
        let payload_len = u16::from_be_bytes([cell_bytes[0], cell_bytes[1]]) as usize;
        if payload_len > CELL_SIZE - 2 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Payload length exceeds cell capacity"
            ));
        }
        
        let payload = &cell_bytes[2..2 + payload_len];
        bincode::deserialize(payload)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    async fn write_request<T>(
        &mut self,
        _: &Self::Protocol,
        io: &mut T,
        req: Self::Request
    ) -> io::Result<()>
    where
        T: futures::AsyncWrite + Unpin + Send,
    {
        let payload = bincode::serialize(&req)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        self.write_cell_with_payload(io, payload).await
    }

    async fn write_response<T>(
        &mut self,
        _: &Self::Protocol,
        io: &mut T,
        res: Self::Response
    ) -> io::Result<()>
    where
        T: futures::AsyncWrite + Unpin + Send,
    {
        let payload = bincode::serialize(&res)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        self.write_cell_with_payload(io, payload).await
    }
}

impl DirectoryCodec {
    /// Write payload as a fixed-size cell with random padding
    async fn write_cell_with_payload<T>(
        &self,
        io: &mut T,
        payload: Vec<u8>
    ) -> io::Result<()>
    where
        T: futures::AsyncWrite + Unpin + Send,
    {
        if payload.len() > CELL_SIZE - 2 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Payload too large for cell"
            ));
        }
        
        let mut cell = vec![0u8; CELL_SIZE];
        
        // Write length prefix
        let len = payload.len() as u16;
        cell[0..2].copy_from_slice(&len.to_be_bytes());
        
        // Write payload
        cell[2..2 + payload.len()].copy_from_slice(&payload);
        
        // Fill remainder with cryptographically secure random padding
        let mut rng = ring::rand::SystemRandom::new();
        rng.fill(&mut cell[2 + payload.len()..])
            .map_err(|_| std::io::Error::new(
                std::io::ErrorKind::Other, 
                "Failed to generate random padding"
            ))?;
        
        io.write_all(&cell).await
    }
}

/// Path selection utilities for clients
pub mod path_selection {
    use super::*;
    use rand::{seq::SliceRandom, Rng};
    
    /// Configuration for path selection
    #[derive(Debug, Clone)]
    pub struct PathConfig {
        pub min_hops: usize,
        pub max_hops: usize,
        pub max_same_node_reuse: usize,
    }
    
    impl Default for PathConfig {
        fn default() -> Self {
            Self {
                min_hops: 2,
                max_hops: 5,
                max_same_node_reuse: 1,
            }
        }
    }
    
    /// Select a random path through the mix network
    pub fn select_random_path(
        available_mixes: &[NodeInfo],
        config: &PathConfig,
    ) -> Result<Vec<PeerId>, anyhow::Error> {
        if available_mixes.is_empty() {
            return Err(anyhow::anyhow!("No mix nodes available"));
        }
        
        let mut rng = rand::thread_rng();
        let path_length = rng.gen_range(config.min_hops..=config.max_hops);
        
        let mut path = Vec::with_capacity(path_length);
        let mut available: Vec<_> = available_mixes.iter().collect();
        
        for _ in 0..path_length {
            if available.is_empty() {
                // If we run out of unique nodes, allow reuse
                available = available_mixes.iter().collect();
            }
            
            if let Some(selected) = available.choose(&mut rng) {
                path.push(selected.peer_id);
                
                // Remove from available to avoid immediate reuse
                available.retain(|node| node.peer_id != selected.peer_id);
            } else {
                return Err(anyhow::anyhow!("Failed to select mix node"));
            }
        }
        
        Ok(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cell_creation_and_extraction() {
        let payload = b"Test message for Loopix cell".to_vec();
        
        // Create cell
        let cell = LoopixCell::new(payload.clone()).unwrap();
        assert_eq!(cell.bytes.len(), CELL_SIZE);
        
        // Extract payload
        let extracted = cell.extract_payload().unwrap();
        assert_eq!(extracted, payload);
    }
    
    #[test]
    fn test_epoch_validity() {
        let now = SystemTime::now();
        let hour = Duration::from_secs(3600);
        
        let valid_epoch = EpochDescriptor {
            mix_nodes: vec![],
            providers: vec![],
            epoch_id: 1,
            started_at: now - hour,
            expires_at: now + hour,
            signature: vec![],
        };
        
        assert!(valid_epoch.is_valid());
        
        let expired_epoch = EpochDescriptor {
            mix_nodes: vec![],
            providers: vec![],
            epoch_id: 2,
            started_at: now - hour * 2,
            expires_at: now - hour,
            signature: vec![],
        };
        
        assert!(!expired_epoch.is_valid());
    }
    
    #[test]
    fn test_path_selection() {
        use path_selection::*;
        
        let mixes: Vec<NodeInfo> = (0..10)
            .map(|i| NodeInfo {
                peer_id: PeerId::random(),
                node_type: NodeType::Mix,
                listen_addrs: vec![format!("127.0.0.1:{}", 8000 + i)],
                public_key: vec![i as u8; 32],
                capabilities: vec![],
                registered_at: SystemTime::now(),
            })
            .collect();
        
        let config = PathConfig::default();
        let path = select_random_path(&mixes, &config).unwrap();
        
        assert!(path.len() >= config.min_hops);
        assert!(path.len() <= config.max_hops);
        
        // Should not have immediate duplicates (unless forced by limited nodes)
        for window in path.windows(2) {
            if mixes.len() > 1 {
                assert_ne!(window[0], window[1]);
            }
        }
    }
}