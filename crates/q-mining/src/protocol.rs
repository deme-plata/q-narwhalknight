use crate::block::{QuantumPoWBlock, MiningTemplate, BlockHeader};
use crate::rewards::RewardResult;
use crate::errors::MiningError;
use q_types::{NodeId, VertexId};
use serde::{Serialize, Deserialize};
use sha3::{Digest, Sha3_256};
use std::collections::HashMap;
use tokio::sync::mpsc;

/// Mining network protocol for Q-NarwhalKnight quantum-enhanced mining
/// 
/// This protocol defines the communication between miners, mining pools,
/// and the DAG-Knight consensus network for coordinated mining operations.
#[derive(Debug, Clone)]
pub struct MiningNetworkProtocol {
    /// Protocol version for backward compatibility
    pub version: ProtocolVersion,
    /// Supported mining algorithms
    pub supported_algorithms: Vec<MiningAlgorithm>,
    /// Network capabilities
    pub capabilities: NetworkCapabilities,
}

/// Protocol version for mining network
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProtocolVersion {
    pub major: u8,
    pub minor: u8,
    pub patch: u8,
}

/// Mining algorithm identifiers for network negotiation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MiningAlgorithm {
    /// SHA-3-256 with quantum enhancements
    QuantumSHA3_256 = 0x01,
    /// Classical SHA-3-256 fallback
    ClassicalSHA3_256 = 0x02,
    /// Future: Memory-hard quantum algorithm
    QuantumArgon2 = 0x03,
}

/// Network capabilities for mining coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkCapabilities {
    /// Maximum hash rate (hashes per second)
    pub max_hash_rate: u64,
    /// GPU mining support
    pub gpu_mining: bool,
    /// Quantum VDF integration
    pub quantum_vdf: bool,
    /// Mining pool support
    pub pool_mining: bool,
    /// Stratum protocol support
    pub stratum_support: bool,
    /// DAG commitment integration
    pub dag_commitment: bool,
    /// Maximum block size handling
    pub max_block_size: u32,
}

/// Mining network message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MiningMessage {
    // Peer discovery and handshake
    /// Initial handshake with capability negotiation
    Handshake {
        protocol_version: ProtocolVersion,
        node_id: NodeId,
        capabilities: NetworkCapabilities,
        chain_id: [u8; 32],
    },
    
    /// Handshake acknowledgment
    HandshakeAck {
        protocol_version: ProtocolVersion,
        accepted_capabilities: NetworkCapabilities,
        peer_info: PeerMiningInfo,
    },
    
    // Mining coordination
    /// Request mining template
    GetMiningTemplate {
        miner_id: [u8; 20],
        preferred_algorithm: MiningAlgorithm,
        quantum_capability: f64, // 0.0-1.0 quantum enhancement level
    },
    
    /// Mining template response
    MiningTemplate {
        template: MiningTemplate,
        difficulty_target: [u8; 32],
        quantum_seed: Option<[u8; 32]>,
        vdf_challenge: Option<[u8; 64]>,
        template_id: u64,
        expires_at: u64, // Unix timestamp
    },
    
    /// Submit mined block
    SubmitBlock {
        template_id: u64,
        miner_id: [u8; 20],
        block: QuantumPoWBlock,
        mining_stats: MiningStats,
    },
    
    /// Block submission response
    BlockSubmissionResult {
        template_id: u64,
        accepted: bool,
        reason: String,
        reward: Option<RewardResult>,
        next_template: Option<MiningTemplate>,
    },
    
    // Network synchronization
    /// Broadcast new block to network
    NewBlock {
        block: QuantumPoWBlock,
        propagation_path: Vec<NodeId>, // For Dandelion++ support
        timestamp: u64,
    },
    
    /// Request block headers for synchronization
    GetBlockHeaders {
        start_height: u64,
        count: u32,
        reverse: bool,
    },
    
    /// Block headers response
    BlockHeaders {
        headers: Vec<BlockHeader>,
        total_blocks: u64,
    },
    
    /// Request full block data
    GetBlock {
        block_hash: [u8; 32],
        include_transactions: bool,
    },
    
    /// Block data response
    Block {
        block: QuantumPoWBlock,
        confirmations: u32,
    },
    
    // Mining pool protocol (Stratum-like)
    /// Mining pool work assignment
    PoolWorkAssignment {
        job_id: String,
        template: MiningTemplate,
        difficulty: u32,
        quantum_target: f64,
        pool_fee: f64, // 0.0-1.0
    },
    
    /// Pool work submission
    PoolWorkSubmission {
        job_id: String,
        nonce: u64,
        timestamp: u64,
        quantum_proof: Option<Vec<u8>>,
    },
    
    /// Pool submission result
    PoolSubmissionResult {
        job_id: String,
        accepted: bool,
        share_difficulty: u32,
        estimated_reward: u64,
    },
    
    // Network maintenance
    /// Ping for connection health
    Ping {
        timestamp: u64,
        chain_height: u64,
    },
    
    /// Pong response
    Pong {
        timestamp: u64,
        latency_ms: u32,
    },
    
    /// Request peer list
    GetPeers,
    
    /// Peer list response
    Peers {
        peers: Vec<PeerMiningInfo>,
    },
    
    /// Error message
    Error {
        code: u32,
        message: String,
        details: Option<String>,
    },
}

/// Peer mining information for discovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerMiningInfo {
    pub node_id: NodeId,
    pub address: String,
    pub port: u16,
    pub capabilities: NetworkCapabilities,
    pub hash_rate: u64,
    pub last_seen: u64,
    pub reputation: f64, // 0.0-1.0 trust score
}

/// Mining statistics for performance tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiningStats {
    pub hash_rate: f64,
    pub mining_duration_ms: u64,
    pub quantum_utilization: f64,
    pub gpu_utilization: Option<f64>,
    pub error_rate: f64,
    pub efficiency_percent: f64,
}

/// Mining network error codes
#[derive(Debug, Clone, Copy)]
pub enum MiningErrorCode {
    // Protocol errors
    UnsupportedProtocolVersion = 1000,
    InvalidHandshake = 1001,
    CapabilityMismatch = 1002,
    
    // Mining errors
    InvalidTemplate = 2000,
    TemplateParsing = 2001,
    TemplateExpired = 2002,
    InvalidDifficulty = 2003,
    
    // Block errors
    InvalidBlock = 3000,
    BlockValidation = 3001,
    InvalidQuantumProof = 3002,
    DuplicateBlock = 3003,
    
    // Network errors
    PeerTimeout = 4000,
    PeerDisconnected = 4001,
    MessageTooLarge = 4002,
    RateLimitExceeded = 4003,
    
    // Pool errors
    InvalidJob = 5000,
    PoolOverloaded = 5001,
    InsufficientShares = 5002,
    PayoutThresholdNotMet = 5003,
}

/// Mining network event types
#[derive(Debug, Clone)]
pub enum MiningNetworkEvent {
    /// New peer connected
    PeerConnected(NodeId, PeerMiningInfo),
    /// Peer disconnected
    PeerDisconnected(NodeId),
    /// New mining template available
    NewTemplate(MiningTemplate),
    /// Block mined successfully
    BlockMined(QuantumPoWBlock, RewardResult),
    /// Network difficulty adjustment
    DifficultyAdjusted { old: u32, new: u32, height: u64 },
    /// Pool work assignment received
    PoolWorkReceived(String, MiningTemplate),
    /// Error occurred
    NetworkError(MiningErrorCode, String),
}

/// Mining network protocol implementation
#[derive(Debug)]
pub struct MiningNetworkHandler {
    /// Local node ID
    pub node_id: NodeId,
    /// Protocol version
    pub protocol_version: ProtocolVersion,
    /// Local capabilities
    pub capabilities: NetworkCapabilities,
    /// Connected peers
    pub peers: HashMap<NodeId, PeerMiningInfo>,
    /// Event sender for upper layers
    pub event_sender: mpsc::UnboundedSender<MiningNetworkEvent>,
    /// Current mining template
    pub current_template: Option<MiningTemplate>,
    /// Message statistics
    pub message_stats: MessageStatistics,
}

/// Message handling statistics
#[derive(Debug, Default)]
pub struct MessageStatistics {
    pub messages_sent: u64,
    pub messages_received: u64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub errors: u64,
    pub peer_connections: u32,
    pub active_templates: u32,
}

impl Default for ProtocolVersion {
    fn default() -> Self {
        Self {
            major: 1,
            minor: 0,
            patch: 0,
        }
    }
}

impl Default for NetworkCapabilities {
    fn default() -> Self {
        Self {
            max_hash_rate: 10_000, // 10K H/s default
            gpu_mining: true,
            quantum_vdf: true,
            pool_mining: true,
            stratum_support: true,
            dag_commitment: true,
            max_block_size: 2_097_152, // 2MB
        }
    }
}

impl MiningNetworkHandler {
    /// Create new mining network handler
    pub fn new(
        node_id: NodeId,
        capabilities: NetworkCapabilities,
    ) -> (Self, mpsc::UnboundedReceiver<MiningNetworkEvent>) {
        let (event_sender, event_receiver) = mpsc::unbounded_channel();
        
        let handler = Self {
            node_id,
            protocol_version: ProtocolVersion::default(),
            capabilities,
            peers: HashMap::new(),
            event_sender,
            current_template: None,
            message_stats: MessageStatistics::default(),
        };
        
        (handler, event_receiver)
    }
    
    /// Handle incoming mining network message
    pub async fn handle_message(
        &mut self,
        peer_id: NodeId,
        message: MiningMessage,
    ) -> Result<Option<MiningMessage>, MiningError> {
        self.message_stats.messages_received += 1;
        
        match message {
            MiningMessage::Handshake { protocol_version, node_id, capabilities, chain_id } => {
                self.handle_handshake(peer_id, protocol_version, node_id, capabilities, chain_id).await
            }
            
            MiningMessage::GetMiningTemplate { miner_id, preferred_algorithm, quantum_capability } => {
                self.handle_get_mining_template(peer_id, miner_id, preferred_algorithm, quantum_capability).await
            }
            
            MiningMessage::SubmitBlock { template_id, miner_id, block, mining_stats } => {
                self.handle_submit_block(peer_id, template_id, miner_id, block, mining_stats).await
            }
            
            MiningMessage::NewBlock { block, propagation_path, timestamp } => {
                self.handle_new_block(peer_id, block, propagation_path, timestamp).await
            }
            
            MiningMessage::Ping { timestamp, chain_height } => {
                Ok(Some(MiningMessage::Pong {
                    timestamp,
                    latency_ms: self.calculate_latency(timestamp),
                }))
            }
            
            MiningMessage::GetPeers => {
                Ok(Some(MiningMessage::Peers {
                    peers: self.peers.values().cloned().collect(),
                }))
            }
            
            _ => {
                tracing::debug!("Unhandled mining message type from peer {}", peer_id);
                Ok(None)
            }
        }
    }
    
    /// Handle mining handshake
    async fn handle_handshake(
        &mut self,
        peer_id: NodeId,
        protocol_version: ProtocolVersion,
        node_id: NodeId,
        capabilities: NetworkCapabilities,
        chain_id: [u8; 32],
    ) -> Result<Option<MiningMessage>, MiningError> {
        // Validate protocol compatibility
        if protocol_version.major != self.protocol_version.major {
            return Ok(Some(MiningMessage::Error {
                code: MiningErrorCode::UnsupportedProtocolVersion as u32,
                message: format!("Unsupported protocol version {}.{}.{}", 
                               protocol_version.major, protocol_version.minor, protocol_version.patch),
                details: None,
            }));
        }
        
        // Create peer info
        let peer_info = PeerMiningInfo {
            node_id,
            address: format!("peer-{}", peer_id), // Simplified for now
            port: 9944,
            capabilities: capabilities.clone(),
            hash_rate: capabilities.max_hash_rate,
            last_seen: chrono::Utc::now().timestamp() as u64,
            reputation: 1.0, // Start with full trust
        };
        
        // Store peer
        self.peers.insert(peer_id, peer_info.clone());
        self.message_stats.peer_connections += 1;
        
        // Notify upper layers
        let _ = self.event_sender.send(MiningNetworkEvent::PeerConnected(peer_id, peer_info.clone()));
        
        // Send handshake acknowledgment
        Ok(Some(MiningMessage::HandshakeAck {
            protocol_version: self.protocol_version,
            accepted_capabilities: self.negotiate_capabilities(&capabilities),
            peer_info: peer_info,
        }))
    }
    
    /// Handle mining template request
    async fn handle_get_mining_template(
        &mut self,
        peer_id: NodeId,
        miner_id: [u8; 20],
        preferred_algorithm: MiningAlgorithm,
        quantum_capability: f64,
    ) -> Result<Option<MiningMessage>, MiningError> {
        // Generate mining template
        let template = self.generate_mining_template(miner_id, preferred_algorithm, quantum_capability).await?;
        let template_id = self.generate_template_id(&template);
        
        // Calculate difficulty target
        let difficulty_target = self.calculate_difficulty_target(&template);
        
        // Generate quantum seed if supported
        let quantum_seed = if quantum_capability > 0.0 {
            Some(self.generate_quantum_seed())
        } else {
            None
        };
        
        // Generate VDF challenge if quantum VDF enabled
        let vdf_challenge = if self.capabilities.quantum_vdf && quantum_seed.is_some() {
            Some(self.generate_vdf_challenge(&template))
        } else {
            None
        };
        
        self.current_template = Some(template.clone());
        self.message_stats.active_templates += 1;
        
        let _ = self.event_sender.send(MiningNetworkEvent::NewTemplate(template.clone()));
        
        Ok(Some(MiningMessage::MiningTemplate {
            template,
            difficulty_target,
            quantum_seed,
            vdf_challenge,
            template_id,
            expires_at: chrono::Utc::now().timestamp() as u64 + 300, // 5 minutes
        }))
    }
    
    /// Handle block submission
    async fn handle_submit_block(
        &mut self,
        peer_id: NodeId,
        template_id: u64,
        miner_id: [u8; 20],
        block: QuantumPoWBlock,
        mining_stats: MiningStats,
    ) -> Result<Option<MiningMessage>, MiningError> {
        // Validate block
        let validation_result = self.validate_submitted_block(&block, template_id).await?;
        
        if !validation_result.valid {
            return Ok(Some(MiningMessage::BlockSubmissionResult {
                template_id,
                accepted: false,
                reason: validation_result.error_reason.unwrap_or("Invalid block".to_string()),
                reward: None,
                next_template: None,
            }));
        }
        
        // Calculate reward
        let reward_result = self.calculate_mining_reward(&block).await?;
        
        // Update peer reputation based on submission quality
        if let Some(peer_info) = self.peers.get_mut(&peer_id) {
            peer_info.reputation = (peer_info.reputation + 0.1).min(1.0);
            peer_info.hash_rate = mining_stats.hash_rate as u64;
            peer_info.last_seen = chrono::Utc::now().timestamp() as u64;
        }
        
        // Notify upper layers
        let _ = self.event_sender.send(MiningNetworkEvent::BlockMined(block.clone(), reward_result.clone()));
        
        // Generate next template
        let next_template = self.generate_next_mining_template(miner_id).await?;
        
        Ok(Some(MiningMessage::BlockSubmissionResult {
            template_id,
            accepted: true,
            reason: "Block accepted".to_string(),
            reward: Some(reward_result),
            next_template: Some(next_template),
        }))
    }
    
    /// Handle new block broadcast
    async fn handle_new_block(
        &mut self,
        peer_id: NodeId,
        block: QuantumPoWBlock,
        propagation_path: Vec<NodeId>,
        timestamp: u64,
    ) -> Result<Option<MiningMessage>, MiningError> {
        // Validate block
        let validation_result = self.validate_new_block(&block).await?;
        
        if validation_result.valid {
            // Forward block to other peers (Dandelion++ propagation)
            let new_propagation_path = self.update_propagation_path(propagation_path, self.node_id);
            
            // Broadcast to peers (except the sender)
            for (&other_peer_id, _) in self.peers.iter() {
                if other_peer_id != peer_id {
                    // In real implementation, would send directly to peer
                    tracing::debug!("Would forward block to peer {}", other_peer_id);
                }
            }
        }
        
        Ok(None) // No direct response needed for broadcasts
    }
    
    /// Generate mining template for miner
    async fn generate_mining_template(
        &self,
        miner_id: [u8; 20],
        algorithm: MiningAlgorithm,
        quantum_capability: f64,
    ) -> Result<MiningTemplate, MiningError> {
        // In real implementation, would fetch from DAG consensus
        Ok(MiningTemplate {
            parent_hash: [0; 32], // Placeholder
            height: 1000, // Placeholder
            timestamp: chrono::Utc::now().timestamp() as u64,
            difficulty: self.calculate_current_difficulty(),
            reward_address: miner_id,
            transactions: vec![], // Placeholder
            dag_commitment_height: 100, // Placeholder
        })
    }
    
    /// Calculate current mining difficulty
    fn calculate_current_difficulty(&self) -> u32 {
        // Simplified difficulty calculation
        1000 // Placeholder
    }
    
    /// Generate template ID
    fn generate_template_id(&self, template: &MiningTemplate) -> u64 {
        let mut hasher = Sha3_256::new();
        hasher.update(&template.parent_hash);
        hasher.update(&template.height.to_be_bytes());
        hasher.update(&template.timestamp.to_be_bytes());
        let hash = hasher.finalize();
        u64::from_be_bytes(hash[..8].try_into().unwrap())
    }
    
    /// Calculate difficulty target for template
    fn calculate_difficulty_target(&self, template: &MiningTemplate) -> [u8; 32] {
        let mut target = [0xFF; 32];
        // Simple difficulty to target conversion
        let leading_zeros = template.difficulty / 256;
        for i in 0..leading_zeros.min(32) as usize {
            target[i] = 0x00;
        }
        target
    }
    
    /// Generate quantum seed
    fn generate_quantum_seed(&self) -> [u8; 32] {
        let mut seed = [0u8; 32];
        // In real implementation, would use quantum RNG
        let random_data = uuid::Uuid::new_v4();
        seed[..16].copy_from_slice(random_data.as_bytes());
        seed[16..].copy_from_slice(random_data.as_bytes());
        seed
    }
    
    /// Generate VDF challenge
    fn generate_vdf_challenge(&self, template: &MiningTemplate) -> [u8; 64] {
        let mut challenge = [0u8; 64];
        let mut hasher = Sha3_256::new();
        hasher.update(&template.parent_hash);
        hasher.update(&template.height.to_be_bytes());
        let hash = hasher.finalize();
        challenge[..32].copy_from_slice(&hash);
        challenge[32..].copy_from_slice(&hash);
        challenge
    }
    
    /// Negotiate capabilities with peer
    fn negotiate_capabilities(&self, peer_capabilities: &NetworkCapabilities) -> NetworkCapabilities {
        NetworkCapabilities {
            max_hash_rate: self.capabilities.max_hash_rate.min(peer_capabilities.max_hash_rate),
            gpu_mining: self.capabilities.gpu_mining && peer_capabilities.gpu_mining,
            quantum_vdf: self.capabilities.quantum_vdf && peer_capabilities.quantum_vdf,
            pool_mining: self.capabilities.pool_mining && peer_capabilities.pool_mining,
            stratum_support: self.capabilities.stratum_support && peer_capabilities.stratum_support,
            dag_commitment: self.capabilities.dag_commitment && peer_capabilities.dag_commitment,
            max_block_size: self.capabilities.max_block_size.min(peer_capabilities.max_block_size),
        }
    }
    
    /// Calculate network latency
    fn calculate_latency(&self, timestamp: u64) -> u32 {
        let now = chrono::Utc::now().timestamp() as u64;
        if now > timestamp {
            ((now - timestamp) * 1000) as u32 // Convert to milliseconds
        } else {
            0
        }
    }
    
    /// Validate submitted block
    async fn validate_submitted_block(
        &self,
        block: &QuantumPoWBlock,
        template_id: u64,
    ) -> Result<BlockValidationResult, MiningError> {
        // Simplified validation
        Ok(BlockValidationResult {
            valid: true,
            error_reason: None,
        })
    }
    
    /// Validate new block broadcast
    async fn validate_new_block(&self, block: &QuantumPoWBlock) -> Result<BlockValidationResult, MiningError> {
        // Simplified validation
        Ok(BlockValidationResult {
            valid: true,
            error_reason: None,
        })
    }
    
    /// Calculate mining reward for block
    async fn calculate_mining_reward(&self, block: &QuantumPoWBlock) -> Result<RewardResult, MiningError> {
        // Simplified reward calculation - in real implementation would use RewardCalculator
        Ok(RewardResult {
            base_reward: 2_000_000_000, // 2.0 QNK
            quantum_bonus: 200_000_000, // 0.2 QNK bonus
            total_reward: 2_200_000_000,
            burn_amount: 550_000_000, // 25% burn
            final_reward: 1_650_000_000, // Net reward
            quantum_quality: 0.95,
            calculation_reason: "High-quality quantum mining".to_string(),
        })
    }
    
    /// Generate next mining template
    async fn generate_next_mining_template(&self, miner_id: [u8; 20]) -> Result<MiningTemplate, MiningError> {
        self.generate_mining_template(miner_id, MiningAlgorithm::QuantumSHA3_256, 0.8).await
    }
    
    /// Update propagation path for Dandelion++
    fn update_propagation_path(&self, mut path: Vec<NodeId>, node_id: NodeId) -> Vec<NodeId> {
        path.push(node_id);
        // Limit path length for privacy
        if path.len() > 10 {
            path.drain(0..1);
        }
        path
    }
    
    /// Get network statistics
    pub fn get_statistics(&self) -> &MessageStatistics {
        &self.message_stats
    }
    
    /// Get connected peers count
    pub fn peer_count(&self) -> usize {
        self.peers.len()
    }
    
    /// Get current network hash rate estimate
    pub fn network_hash_rate(&self) -> u64 {
        self.peers.values().map(|p| p.hash_rate).sum()
    }
}

/// Block validation result
#[derive(Debug)]
struct BlockValidationResult {
    valid: bool,
    error_reason: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_mining_network_handler_creation() {
        let node_id = [1u8; 32];
        let capabilities = NetworkCapabilities::default();
        
        let (handler, _receiver) = MiningNetworkHandler::new(node_id, capabilities);
        
        assert_eq!(handler.node_id, node_id);
        assert_eq!(handler.peers.len(), 0);
        assert_eq!(handler.protocol_version.major, 1);
    }
    
    #[tokio::test]
    async fn test_handshake_handling() {
        let node_id = [1u8; 32];
        let peer_id = [2u8; 32];
        let capabilities = NetworkCapabilities::default();
        
        let (mut handler, _receiver) = MiningNetworkHandler::new(node_id, capabilities.clone());
        
        let handshake = MiningMessage::Handshake {
            protocol_version: ProtocolVersion::default(),
            node_id: peer_id,
            capabilities: capabilities.clone(),
            chain_id: [0; 32],
        };
        
        let response = handler.handle_message(peer_id, handshake).await.unwrap();
        
        assert!(response.is_some());
        assert!(handler.peers.contains_key(&peer_id));
        
        if let Some(MiningMessage::HandshakeAck { .. }) = response {
            // Handshake acknowledgment received
        } else {
            panic!("Expected HandshakeAck response");
        }
    }
    
    #[tokio::test]
    async fn test_mining_template_generation() {
        let node_id = [1u8; 32];
        let capabilities = NetworkCapabilities::default();
        
        let (handler, _receiver) = MiningNetworkHandler::new(node_id, capabilities);
        
        let miner_id = [3u8; 20];
        let template = handler.generate_mining_template(
            miner_id,
            MiningAlgorithm::QuantumSHA3_256,
            0.8
        ).await.unwrap();
        
        assert_eq!(template.reward_address, miner_id);
        assert!(template.height > 0);
        assert!(template.difficulty > 0);
    }
    
    #[tokio::test]
    async fn test_capability_negotiation() {
        let node_id = [1u8; 32];
        let local_caps = NetworkCapabilities {
            max_hash_rate: 20_000,
            gpu_mining: true,
            quantum_vdf: true,
            ..Default::default()
        };
        
        let peer_caps = NetworkCapabilities {
            max_hash_rate: 15_000,
            gpu_mining: false,
            quantum_vdf: true,
            ..Default::default()
        };
        
        let (handler, _receiver) = MiningNetworkHandler::new(node_id, local_caps);
        let negotiated = handler.negotiate_capabilities(&peer_caps);
        
        assert_eq!(negotiated.max_hash_rate, 15_000); // Minimum of both
        assert!(!negotiated.gpu_mining); // Logical AND
        assert!(negotiated.quantum_vdf); // Both support
    }
    
    #[test]
    fn test_protocol_version_compatibility() {
        let v1 = ProtocolVersion { major: 1, minor: 0, patch: 0 };
        let v2 = ProtocolVersion { major: 1, minor: 1, patch: 0 };
        let v3 = ProtocolVersion { major: 2, minor: 0, patch: 0 };
        
        assert_eq!(v1.major, v2.major); // Compatible
        assert_ne!(v1.major, v3.major); // Incompatible
    }
    
    #[test]
    fn test_mining_algorithm_serialization() {
        let algo = MiningAlgorithm::QuantumSHA3_256;
        let serialized = bincode::serialize(&algo).unwrap();
        let deserialized: MiningAlgorithm = bincode::deserialize(&serialized).unwrap();
        
        assert_eq!(algo, deserialized);
    }
}