/// Message handler for Q-Network consensus messages
/// Handles vertex, certificate, and quantum protocol messages

use q_types::*;
use q_narwhal_core::{Certificate, Vertex};
use q_dag_knight::{CommitDecision, AnchorElectionResult};
use super::crypto_agile::AgileHandshake;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::sync::{broadcast, RwLock};
use tracing::{debug, info, warn, error};

/// Network message types for Q-NarwhalKnight
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkMessage {
    // Core consensus messages
    Vertex(Vertex),
    Certificate(Certificate),
    CommitNotification(CommitDecision),
    
    // DAG-Knight specific
    AnchorElection(AnchorElectionResult),
    OrderingProof { round: Round, ordering: Vec<VertexId> },
    
    // Quantum protocol messages
    QuantumBeacon { round: Round, entropy: [u8; 32], signature: Vec<u8> },
    VDFProof { challenge: [u8; 32], proof: [u8; 32], difficulty: u64 },
    
    // Crypto-agile protocol
    CryptoUpgrade(AgileHandshake),
    AlgorithmNegotiation { schemes: Vec<String>, nonce: [u8; 32] },
    
    // Phase 2+ quantum messages
    QRNGSeed { round: Round, seed: [u8; 32], qrng_proof: Vec<u8> },
    QuantumEntanglement { qubits: Vec<u8>, bell_state: [u8; 4] },
    
    // Synchronization and health
    SyncRequest { from_round: Round, to_round: Round },
    SyncResponse { vertices: Vec<Vertex>, certificates: Vec<Certificate> },
    HealthCheck { timestamp: u64, node_id: NodeId },
    
    // Error handling
    Error { code: u32, message: String },
}

impl NetworkMessage {
    /// Get the message type as string
    pub fn message_type(&self) -> &'static str {
        match self {
            NetworkMessage::Vertex(_) => "vertex",
            NetworkMessage::Certificate(_) => "certificate",
            NetworkMessage::CommitNotification(_) => "commit",
            NetworkMessage::AnchorElection(_) => "anchor_election",
            NetworkMessage::OrderingProof { .. } => "ordering_proof",
            NetworkMessage::QuantumBeacon { .. } => "quantum_beacon",
            NetworkMessage::VDFProof { .. } => "vdf_proof",
            NetworkMessage::CryptoUpgrade(_) => "crypto_upgrade",
            NetworkMessage::AlgorithmNegotiation { .. } => "algorithm_negotiation",
            NetworkMessage::QRNGSeed { .. } => "qrng_seed",
            NetworkMessage::QuantumEntanglement { .. } => "quantum_entanglement",
            NetworkMessage::SyncRequest { .. } => "sync_request",
            NetworkMessage::SyncResponse { .. } => "sync_response",
            NetworkMessage::HealthCheck { .. } => "health_check",
            NetworkMessage::Error { .. } => "error",
        }
    }

    /// Get the round number if applicable
    pub fn round(&self) -> Option<Round> {
        match self {
            NetworkMessage::Vertex(v) => Some(v.round),
            NetworkMessage::Certificate(c) => Some(c.round),
            NetworkMessage::CommitNotification(d) => Some(d.round),
            NetworkMessage::AnchorElection(r) => Some(r.round),
            NetworkMessage::OrderingProof { round, .. } => Some(*round),
            NetworkMessage::QuantumBeacon { round, .. } => Some(*round),
            NetworkMessage::QRNGSeed { round, .. } => Some(*round),
            NetworkMessage::SyncRequest { from_round, .. } => Some(*from_round),
            _ => None,
        }
    }

    /// Check if message is quantum-related
    pub fn is_quantum_message(&self) -> bool {
        matches!(self,
            NetworkMessage::QuantumBeacon { .. } |
            NetworkMessage::VDFProof { .. } |
            NetworkMessage::QRNGSeed { .. } |
            NetworkMessage::QuantumEntanglement { .. }
        )
    }

    /// Validate message integrity
    pub fn validate(&self) -> Result<()> {
        match self {
            NetworkMessage::Vertex(vertex) => {
                if vertex.id == [0u8; 32] {
                    return Err(anyhow::anyhow!("Invalid vertex ID"));
                }
                if vertex.signature.is_empty() {
                    return Err(anyhow::anyhow!("Missing vertex signature"));
                }
            }
            NetworkMessage::Certificate(cert) => {
                if cert.vertex_id == [0u8; 32] {
                    return Err(anyhow::anyhow!("Invalid certificate vertex ID"));
                }
                if !cert.threshold_met {
                    return Err(anyhow::anyhow!("Certificate threshold not met"));
                }
            }
            NetworkMessage::VDFProof { challenge, proof, .. } => {
                if *challenge == [0u8; 32] || *proof == [0u8; 32] {
                    return Err(anyhow::anyhow!("Invalid VDF proof"));
                }
            }
            NetworkMessage::Error { code, message } => {
                if message.is_empty() {
                    return Err(anyhow::anyhow!("Empty error message"));
                }
            }
            _ => {}
        }
        Ok(())
    }
}

/// Message handler for processing network messages
pub struct MessageHandler {
    node_id: NodeId,
    current_phase: Phase,
    message_stats: RwLock<MessageStats>,
    message_cache: RwLock<HashMap<[u8; 32], NetworkMessage>>,
    
    // Event channels
    vertex_tx: broadcast::Sender<Vertex>,
    certificate_tx: broadcast::Sender<Certificate>,
    commit_tx: broadcast::Sender<CommitDecision>,
    quantum_tx: broadcast::Sender<NetworkMessage>,
}

#[derive(Debug, Clone)]
pub struct MessageStats {
    pub total_messages: u64,
    pub vertices_received: u64,
    pub certificates_received: u64,
    pub commits_received: u64,
    pub quantum_messages: u64,
    pub errors_received: u64,
    pub bytes_transferred: u64,
    pub processing_errors: u64,
}

impl MessageHandler {
    pub fn new(node_id: NodeId, current_phase: Phase) -> Self {
        let (vertex_tx, _) = broadcast::channel(10000);
        let (certificate_tx, _) = broadcast::channel(10000);
        let (commit_tx, _) = broadcast::channel(10000);
        let (quantum_tx, _) = broadcast::channel(10000);

        Self {
            node_id,
            current_phase,
            message_stats: RwLock::new(MessageStats {
                total_messages: 0,
                vertices_received: 0,
                certificates_received: 0,
                commits_received: 0,
                quantum_messages: 0,
                errors_received: 0,
                bytes_transferred: 0,
                processing_errors: 0,
            }),
            message_cache: RwLock::new(HashMap::new()),
            vertex_tx,
            certificate_tx,
            commit_tx,
            quantum_tx,
        }
    }

    /// Process incoming network message
    pub async fn handle_message(&self, message: NetworkMessage, source: libp2p::PeerId) -> Result<()> {
        // Update statistics
        {
            let mut stats = self.message_stats.write().await;
            stats.total_messages += 1;
            
            // Estimate message size (simplified)
            let estimated_size = match &message {
                NetworkMessage::Vertex(_) => 1024,
                NetworkMessage::Certificate(_) => 512,
                NetworkMessage::SyncResponse { vertices, certificates } => 
                    (vertices.len() * 1024 + certificates.len() * 512) as u64,
                _ => 256,
            };
            stats.bytes_transferred += estimated_size;
        }

        // Validate message
        if let Err(e) = message.validate() {
            warn!("Invalid message from {}: {}", source, e);
            self.increment_error_count().await;
            return Ok(());
        }

        debug!("Processing {} message from {}", message.message_type(), source);

        // Process message by type
        match message {
            NetworkMessage::Vertex(vertex) => {
                self.handle_vertex(vertex, source).await?;
            }
            NetworkMessage::Certificate(certificate) => {
                self.handle_certificate(certificate, source).await?;
            }
            NetworkMessage::CommitNotification(commit) => {
                self.handle_commit_notification(commit, source).await?;
            }
            NetworkMessage::AnchorElection(election) => {
                self.handle_anchor_election(election, source).await?;
            }
            NetworkMessage::QuantumBeacon { round, entropy, signature } => {
                self.handle_quantum_beacon(round, entropy, signature, source).await?;
            }
            NetworkMessage::VDFProof { challenge, proof, difficulty } => {
                self.handle_vdf_proof(challenge, proof, difficulty, source).await?;
            }
            NetworkMessage::CryptoUpgrade(handshake) => {
                self.handle_crypto_upgrade(handshake, source).await?;
            }
            NetworkMessage::SyncRequest { from_round, to_round } => {
                self.handle_sync_request(from_round, to_round, source).await?;
            }
            NetworkMessage::HealthCheck { timestamp, node_id } => {
                self.handle_health_check(timestamp, node_id, source).await?;
            }
            NetworkMessage::Error { code, message } => {
                self.handle_error_message(code, message, source).await?;
            }
            msg if msg.is_quantum_message() => {
                self.handle_quantum_message(msg, source).await?;
            }
            _ => {
                debug!("Unhandled message type: {}", message.message_type());
            }
        }

        Ok(())
    }

    /// Handle vertex message
    async fn handle_vertex(&self, vertex: Vertex, source: libp2p::PeerId) -> Result<()> {
        info!("Received vertex {} from {} for round {}", 
              hex::encode(vertex.id), source, vertex.round);

        // Update stats
        {
            let mut stats = self.message_stats.write().await;
            stats.vertices_received += 1;
        }

        // Cache the vertex
        {
            let mut cache = self.message_cache.write().await;
            let message_hash = self.compute_message_hash(&NetworkMessage::Vertex(vertex.clone()))?;
            cache.insert(message_hash, NetworkMessage::Vertex(vertex.clone()));
        }

        // Forward to consensus layer
        if self.vertex_tx.send(vertex).is_err() {
            warn!("No receivers for vertex messages");
        }

        Ok(())
    }

    /// Handle certificate message
    async fn handle_certificate(&self, certificate: Certificate, source: libp2p::PeerId) -> Result<()> {
        info!("Received certificate for vertex {} from {} in round {}", 
              hex::encode(certificate.vertex_id), source, certificate.round);

        // Update stats
        {
            let mut stats = self.message_stats.write().await;
            stats.certificates_received += 1;
        }

        // Forward to consensus layer
        if self.certificate_tx.send(certificate).is_err() {
            warn!("No receivers for certificate messages");
        }

        Ok(())
    }

    /// Handle commit notification
    async fn handle_commit_notification(&self, commit: CommitDecision, source: libp2p::PeerId) -> Result<()> {
        info!("Received commit notification for vertex {} from {} in round {}", 
              hex::encode(commit.vertex_id), source, commit.round);

        // Update stats
        {
            let mut stats = self.message_stats.write().await;
            stats.commits_received += 1;
        }

        // Forward to consensus layer
        if self.commit_tx.send(commit).is_err() {
            warn!("No receivers for commit messages");
        }

        Ok(())
    }

    /// Handle anchor election result
    async fn handle_anchor_election(&self, election: AnchorElectionResult, source: libp2p::PeerId) -> Result<()> {
        info!("Received anchor election result for round {} from {}", 
              election.round, source);

        if let Some(anchor_id) = election.anchor_vertex_id {
            info!("Anchor elected: {} with strength {:.4}", 
                  hex::encode(anchor_id), election.election_strength);
        }

        // Forward as quantum message
        if self.quantum_tx.send(NetworkMessage::AnchorElection(election)).is_err() {
            warn!("No receivers for anchor election messages");
        }

        Ok(())
    }

    /// Handle quantum beacon
    async fn handle_quantum_beacon(&self, round: Round, entropy: [u8; 32], signature: Vec<u8>, source: libp2p::PeerId) -> Result<()> {
        debug!("Received quantum beacon for round {} from {}", round, source);

        // Update quantum message stats
        {
            let mut stats = self.message_stats.write().await;
            stats.quantum_messages += 1;
        }

        // Forward quantum beacon
        let beacon_msg = NetworkMessage::QuantumBeacon { round, entropy, signature };
        if self.quantum_tx.send(beacon_msg).is_err() {
            warn!("No receivers for quantum beacon messages");
        }

        Ok(())
    }

    /// Handle VDF proof
    async fn handle_vdf_proof(&self, challenge: [u8; 32], proof: [u8; 32], difficulty: u64, source: libp2p::PeerId) -> Result<()> {
        debug!("Received VDF proof from {} (difficulty: {})", source, difficulty);

        // Forward VDF proof
        let vdf_msg = NetworkMessage::VDFProof { challenge, proof, difficulty };
        if self.quantum_tx.send(vdf_msg).is_err() {
            warn!("No receivers for VDF proof messages");
        }

        Ok(())
    }

    /// Handle crypto upgrade handshake
    async fn handle_crypto_upgrade(&self, handshake: AgileHandshake, source: libp2p::PeerId) -> Result<()> {
        info!("Received crypto upgrade handshake from {} (target phase: {:?})", 
              source, handshake.target_phase);

        // Only accept if we support the target phase
        if handshake.target_phase > self.current_phase {
            warn!("Peer {} requesting unsupported phase {:?}", source, handshake.target_phase);
            return Ok(());
        }

        // Forward crypto upgrade message
        let upgrade_msg = NetworkMessage::CryptoUpgrade(handshake);
        if self.quantum_tx.send(upgrade_msg).is_err() {
            warn!("No receivers for crypto upgrade messages");
        }

        Ok(())
    }

    /// Handle sync request
    async fn handle_sync_request(&self, from_round: Round, to_round: Round, source: libp2p::PeerId) -> Result<()> {
        debug!("Received sync request from {} for rounds {}-{}", source, from_round, to_round);

        // TODO: Implement sync response by gathering vertices and certificates
        // This would integrate with the vertex store and certificate storage
        info!("Sync request handling not fully implemented yet");

        Ok(())
    }

    /// Handle health check
    async fn handle_health_check(&self, timestamp: u64, node_id: NodeId, source: libp2p::PeerId) -> Result<()> {
        debug!("Received health check from {} (node: {})", source, hex::encode(node_id));

        // TODO: Respond with our health status
        Ok(())
    }

    /// Handle error message
    async fn handle_error_message(&self, code: u32, message: String, source: libp2p::PeerId) -> Result<()> {
        warn!("Received error from {}: {} (code: {})", source, message, code);

        // Update error stats
        {
            let mut stats = self.message_stats.write().await;
            stats.errors_received += 1;
        }

        Ok(())
    }

    /// Handle quantum-specific messages
    async fn handle_quantum_message(&self, message: NetworkMessage, source: libp2p::PeerId) -> Result<()> {
        debug!("Processing quantum message: {} from {}", message.message_type(), source);

        // Update quantum message stats
        {
            let mut stats = self.message_stats.write().await;
            stats.quantum_messages += 1;
        }

        // Forward to quantum handlers
        if self.quantum_tx.send(message).is_err() {
            warn!("No receivers for quantum messages");
        }

        Ok(())
    }

    /// Get event receivers for different message types
    pub fn get_vertex_receiver(&self) -> broadcast::Receiver<Vertex> {
        self.vertex_tx.subscribe()
    }

    pub fn get_certificate_receiver(&self) -> broadcast::Receiver<Certificate> {
        self.certificate_tx.subscribe()
    }

    pub fn get_commit_receiver(&self) -> broadcast::Receiver<CommitDecision> {
        self.commit_tx.subscribe()
    }

    pub fn get_quantum_receiver(&self) -> broadcast::Receiver<NetworkMessage> {
        self.quantum_tx.subscribe()
    }

    /// Get message statistics
    pub async fn get_stats(&self) -> MessageStats {
        self.message_stats.read().await.clone()
    }

    /// Clear message cache
    pub async fn clear_cache(&self) {
        let mut cache = self.message_cache.write().await;
        cache.clear();
        debug!("Cleared message cache");
    }

    /// Helper methods
    async fn increment_error_count(&self) {
        let mut stats = self.message_stats.write().await;
        stats.processing_errors += 1;
    }

    fn compute_message_hash(&self, message: &NetworkMessage) -> Result<[u8; 32]> {
        use sha3::{Digest, Sha3_256};
        
        let data = postcard::to_allocvec(message)?;
        let mut hasher = Sha3_256::new();
        hasher.update(&data);
        Ok(hasher.finalize().into())
    }

    /// Update current phase capabilities
    pub fn update_phase(&mut self, phase: Phase) {
        self.current_phase = phase;
        info!("Updated message handler to phase: {:?}", phase);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use libp2p::PeerId;
    use chrono::Utc;

    fn create_test_vertex() -> Vertex {
        Vertex {
            id: [1u8; 32],
            round: 1,
            author: [2u8; 32],
            tx_root: [3u8; 32],
            parents: vec![],
            transactions: vec![],
            signature: vec![1, 2, 3],
            timestamp: Utc::now(),
        }
    }

    #[tokio::test]
    async fn test_message_handler_creation() {
        let node_id = [1u8; 32];
        let handler = MessageHandler::new(node_id, Phase::Phase0);
        
        let stats = handler.get_stats().await;
        assert_eq!(stats.total_messages, 0);
    }

    #[tokio::test]
    async fn test_message_validation() {
        let vertex = create_test_vertex();
        let message = NetworkMessage::Vertex(vertex);
        
        assert!(message.validate().is_ok());
        assert_eq!(message.message_type(), "vertex");
        assert_eq!(message.round(), Some(1));
    }

    #[tokio::test]
    async fn test_vertex_handling() {
        let node_id = [1u8; 32];
        let handler = MessageHandler::new(node_id, Phase::Phase0);
        let source = PeerId::random();
        
        let vertex = create_test_vertex();
        let message = NetworkMessage::Vertex(vertex.clone());
        
        let result = handler.handle_message(message, source).await;
        assert!(result.is_ok());
        
        let stats = handler.get_stats().await;
        assert_eq!(stats.total_messages, 1);
        assert_eq!(stats.vertices_received, 1);
    }

    #[tokio::test]
    async fn test_quantum_message_detection() {
        let beacon_msg = NetworkMessage::QuantumBeacon {
            round: 1,
            entropy: [42u8; 32],
            signature: vec![1, 2, 3],
        };
        
        assert!(beacon_msg.is_quantum_message());
        
        let vertex_msg = NetworkMessage::Vertex(create_test_vertex());
        assert!(!vertex_msg.is_quantum_message());
    }
}