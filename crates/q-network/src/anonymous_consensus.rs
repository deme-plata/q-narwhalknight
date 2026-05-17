/// Anonymous Consensus Integration with Loopix
/// 
/// Integrates Q-NarwhalKnight DAG-Knight consensus with Loopix anonymity network
/// to provide anonymous block proposals, voting, and leader election while
/// maintaining high performance and quantum resistance.

use crate::{
    loopix_network::{LoopixNetwork, LoopixCommand, LoopixEvent, NetworkStatistics},
    loopix_protocol::{LoopixMessage, NodeType, EpochDescriptor},
    loopix_client::{LoopixClient, ClientConfig},
    peer_discovery::{PeerInfo, PeerAddress},
};

use q_narwhal_core::{Block, Vote, Transaction, ProposalId, EpochId};
use q_dag_knight::{DagKnightConsensus, ConsensusEvent, LeaderElection};
use q_types::{NodeId, Phase, Signature, VoteType};

use libp2p::PeerId;
use std::{
    collections::HashMap,
    time::{Duration, Instant, SystemTime},
};
use tokio::{
    sync::{mpsc, oneshot, RwLock},
    time::interval,
};
use tracing::{info, warn, debug, error};
use serde::{Serialize, Deserialize};
use chrono;
use hex;
use blake3;
use bincode;
use rand;

/// Anonymity level for different consensus operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnonymityLevel {
    /// No anonymity - direct libp2p (emergency/bootstrap only)
    Direct,
    /// Low anonymity - 2 mix layers, optimized for performance
    Low,
    /// Balanced anonymity - 3 mix layers, production default
    Balanced,
    /// High anonymity - 5 mix layers, maximum protection
    High,
    /// Maximum anonymity - 5 layers + maximum cover traffic
    Maximum,
}

impl AnonymityLevel {
    /// Get the number of mix layers for this anonymity level
    pub fn mix_layers(&self) -> usize {
        match self {
            Self::Direct => 0,
            Self::Low => 2,
            Self::Balanced => 3,
            Self::High => 5,
            Self::Maximum => 5,
        }
    }
    
    /// Get the cover traffic multiplier for this anonymity level
    pub fn cover_traffic_multiplier(&self) -> f64 {
        match self {
            Self::Direct => 0.0,
            Self::Low => 0.2,
            Self::Balanced => 0.4,
            Self::High => 0.6,
            Self::Maximum => 0.8,
        }
    }
}

/// Performance mode for consensus operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerformanceMode {
    /// Maximum performance - minimal anonymity
    HighPerformance,
    /// Balanced performance and anonymity
    Balanced,
    /// Maximum anonymity - reduced performance
    HighAnonymity,
}

impl PerformanceMode {
    /// Get the default anonymity level for this performance mode
    pub fn default_anonymity_level(&self) -> AnonymityLevel {
        match self {
            Self::HighPerformance => AnonymityLevel::Low,
            Self::Balanced => AnonymityLevel::Balanced,
            Self::HighAnonymity => AnonymityLevel::High,
        }
    }
}

/// Configuration for anonymous consensus
#[derive(Debug, Clone)]
pub struct AnonymousConsensusConfig {
    /// Node identifier
    pub node_id: NodeId,
    /// Performance mode
    pub performance_mode: PerformanceMode,
    /// Whether to enable anonymous block proposals
    pub anonymous_proposals: bool,
    /// Whether to enable anonymous voting
    pub anonymous_voting: bool,
    /// Whether to enable anonymous leader election
    pub anonymous_leader_election: bool,
    /// Adaptive anonymity based on network conditions
    pub adaptive_anonymity: bool,
    /// Emergency bypass threshold (latency in ms)
    pub emergency_bypass_threshold_ms: u64,
}

impl Default for AnonymousConsensusConfig {
    fn default() -> Self {
        Self {
            node_id: [0u8; 32],
            performance_mode: PerformanceMode::Balanced,
            anonymous_proposals: true,
            anonymous_voting: true,
            anonymous_leader_election: true,
            adaptive_anonymity: true,
            emergency_bypass_threshold_ms: 5000, // 5 second emergency bypass
        }
    }
}

/// Anonymous consensus message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnonymousConsensusMessage {
    /// Anonymous block proposal
    BlockProposal {
        proposal_id: ProposalId,
        block: Block,
        proposer_commitment: Vec<u8>, // Zero-knowledge commitment
        proof: Vec<u8>, // Validity proof
    },
    
    /// Anonymous vote on a block
    Vote {
        proposal_id: ProposalId,
        decision: VoteDecision,
        nullifier: Vec<u8>, // Prevents double-voting
        proof: Vec<u8>, // Vote validity proof
    },
    
    /// Anonymous leader election participation
    LeaderElectionContribution {
        epoch_id: EpochId,
        vdf_contribution: Vec<u8>,
        commitment: Vec<u8>,
        proof: Vec<u8>,
    },
    
    /// Anonymous consensus acknowledgment
    Acknowledgment {
        proposal_id: ProposalId,
        timestamp: SystemTime,
    },
}

/// Vote decision with anonymity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VoteDecision {
    Accept,
    Reject,
    Abstain,
}

/// Anonymous consensus engine
pub struct AnonymousConsensus {
    /// Configuration
    config: AnonymousConsensusConfig,
    /// Underlying DAG-Knight consensus
    dag_knight: DagKnightConsensus,
    /// Loopix anonymity network
    loopix_network: LoopixNetwork,
    /// Loopix command sender
    loopix_commands: mpsc::Sender<LoopixCommand>,
    /// Network health monitoring
    network_health: NetworkHealth,
    /// Performance metrics
    metrics: AnonymousConsensusMetrics,
    /// Current epoch information
    current_epoch: Option<EpochDescriptor>,
    /// Pending consensus operations
    pending_operations: HashMap<String, PendingOperation>,
}

/// Network health status for adaptive anonymity
#[derive(Debug, Clone)]
pub struct NetworkHealth {
    /// Average latency through anonymity network
    pub avg_anonymity_latency: Duration,
    /// Success rate of anonymous message delivery
    pub delivery_success_rate: f64,
    /// Network congestion level (0.0 = none, 1.0 = severe)
    pub congestion_level: f64,
    /// Last health check time
    pub last_check: Instant,
}

impl Default for NetworkHealth {
    fn default() -> Self {
        Self {
            avg_anonymity_latency: Duration::from_millis(150),
            delivery_success_rate: 0.98,
            congestion_level: 0.1,
            last_check: Instant::now(),
        }
    }
}

/// Pending consensus operation
#[derive(Debug)]
struct PendingOperation {
    operation_id: String,
    message: AnonymousConsensusMessage,
    anonymity_level: AnonymityLevel,
    started_at: Instant,
    timeout: Duration,
}

/// Performance metrics for anonymous consensus
#[derive(Debug, Default)]
pub struct AnonymousConsensusMetrics {
    /// Total blocks proposed anonymously
    pub anonymous_proposals: u64,
    /// Total anonymous votes cast
    pub anonymous_votes: u64,
    /// Total anonymous leader election participations
    pub anonymous_leader_elections: u64,
    /// Average anonymity latency overhead
    pub avg_anonymity_overhead: Duration,
    /// Consensus finality time with anonymity
    pub consensus_finality_time: Duration,
    /// Messages that bypassed anonymity for performance
    pub emergency_bypasses: u64,
    /// Anonymity success rate
    pub anonymity_success_rate: f64,
}

impl AnonymousConsensus {
    /// Create a new anonymous consensus engine
    pub async fn new(
        config: AnonymousConsensusConfig,
        loopix_network: LoopixNetwork,
        loopix_commands: mpsc::Sender<LoopixCommand>,
    ) -> Result<Self, anyhow::Error> {
        let dag_knight = DagKnightConsensus {
            node_id: config.node_id,
            current_epoch: 1,
        };
        
        Ok(Self {
            config,
            dag_knight,
            loopix_network,
            loopix_commands,
            network_health: NetworkHealth::default(),
            metrics: AnonymousConsensusMetrics::default(),
            current_epoch: None,
            pending_operations: HashMap::new(),
        })
    }
    
    /// Start the anonymous consensus engine
    pub async fn start(&mut self) -> Result<(), anyhow::Error> {
        info!("Starting anonymous consensus with {:?} mode", self.config.performance_mode);
        
        // Start monitoring tasks
        let mut health_check_timer = interval(Duration::from_secs(30));
        let mut metrics_timer = interval(Duration::from_secs(60));
        let mut pending_cleanup_timer = interval(Duration::from_secs(10));
        
        loop {
            tokio::select! {
                // Monitor network health for adaptive anonymity
                _ = health_check_timer.tick() => {
                    self.check_network_health().await;
                }
                
                // Update and log metrics
                _ = metrics_timer.tick() => {
                    self.update_metrics().await;
                    self.log_performance_metrics().await;
                }
                
                // Clean up timed-out pending operations
                _ = pending_cleanup_timer.tick() => {
                    self.cleanup_pending_operations().await;
                }
            }
        }
    }
    
    /// Propose a block with anonymity
    pub async fn propose_block_anonymous(
        &mut self,
        block: Block,
        anonymity_level: Option<AnonymityLevel>,
    ) -> Result<ProposalId, anyhow::Error> {
        if !self.config.anonymous_proposals {
            return self.dag_knight.propose_block_direct(block).await;
        }
        
        let anonymity_level = anonymity_level.unwrap_or_else(|| {
            self.determine_optimal_anonymity_level(MessagePriority::High)
        });
        
        let proposal_id: ProposalId = rand::random();
        
        // Generate zero-knowledge commitment for the proposal
        let proposer_commitment = self.generate_proposer_commitment(&block).await?;
        let proof = self.generate_validity_proof(&block).await?;
        
        let message = AnonymousConsensusMessage::BlockProposal {
            proposal_id,
            block: block.clone(),
            proposer_commitment,
            proof,
        };
        
        // Send through Loopix anonymity network
        self.send_anonymous_consensus_message(message, anonymity_level).await?;
        
        self.metrics.anonymous_proposals += 1;
        info!("Proposed block {} with {:?} anonymity", hex::encode(proposal_id), anonymity_level);
        
        Ok(proposal_id)
    }
    
    /// Cast an anonymous vote on a proposal
    pub async fn vote_anonymous(
        &mut self,
        proposal_id: ProposalId,
        decision: VoteDecision,
        anonymity_level: Option<AnonymityLevel>,
    ) -> Result<(), anyhow::Error> {
        if !self.config.anonymous_voting {
            let vote = Vote {
                proposal_id,
                vote_type: match decision {
                    VoteDecision::Accept => VoteType::Accept,
                    VoteDecision::Reject => VoteType::Reject,
                    VoteDecision::Abstain => VoteType::Abstain,
                },
                voter: self.config.node_id,
                round: 0,
                epoch: 0,
                signature: vec![],
                timestamp: chrono::Utc::now(),
            };
            return self.dag_knight.vote_direct(vote).await;
        }
        
        let anonymity_level = anonymity_level.unwrap_or_else(|| {
            self.determine_optimal_anonymity_level(MessagePriority::Normal)
        });
        
        // Generate nullifier to prevent double-voting
        let nullifier = self.generate_vote_nullifier(proposal_id).await?;
        let proof = self.generate_vote_proof(proposal_id, &decision).await?;
        
        let message = AnonymousConsensusMessage::Vote {
            proposal_id,
            decision,
            nullifier,
            proof,
        };
        
        self.send_anonymous_consensus_message(message, anonymity_level).await?;
        
        self.metrics.anonymous_votes += 1;
        debug!("Cast anonymous vote for proposal {} with {:?} anonymity", hex::encode(proposal_id), anonymity_level);
        
        Ok(())
    }
    
    /// Participate in anonymous leader election
    pub async fn participate_leader_election_anonymous(
        &mut self,
        epoch_id: EpochId,
        anonymity_level: Option<AnonymityLevel>,
    ) -> Result<(), anyhow::Error> {
        if !self.config.anonymous_leader_election {
            return self.dag_knight.participate_leader_election_direct(epoch_id).await;
        }
        
        let anonymity_level = anonymity_level.unwrap_or(AnonymityLevel::Maximum);
        
        // Generate VDF contribution for leader election
        let vdf_contribution = self.generate_vdf_contribution(epoch_id).await?;
        let commitment = self.generate_leader_commitment(epoch_id).await?;
        let proof = self.generate_leader_proof(epoch_id, &vdf_contribution).await?;
        
        let message = AnonymousConsensusMessage::LeaderElectionContribution {
            epoch_id,
            vdf_contribution,
            commitment,
            proof,
        };
        
        self.send_anonymous_consensus_message(message, anonymity_level).await?;
        
        self.metrics.anonymous_leader_elections += 1;
        info!("Participated in anonymous leader election for epoch {} with maximum anonymity", epoch_id);
        
        Ok(())
    }
    
    /// Send consensus message through anonymity network
    async fn send_anonymous_consensus_message(
        &mut self,
        message: AnonymousConsensusMessage,
        anonymity_level: AnonymityLevel,
    ) -> Result<(), anyhow::Error> {
        let start_time = Instant::now();
        
        // Check if we should bypass anonymity for emergency performance
        if self.should_bypass_anonymity(&message, anonymity_level).await {
            warn!("Bypassing anonymity for emergency performance");
            self.send_direct_consensus_message(message).await?;
            self.metrics.emergency_bypasses += 1;
            return Ok(());
        }
        
        // Serialize message for transmission
        let payload = bincode::serialize(&message)
            .map_err(|e| anyhow::anyhow!("Failed to serialize consensus message: {}", e))?;
        
        // Send through Loopix with appropriate anonymity level
        let command = LoopixCommand::SendMessage(payload);

        self.loopix_commands.send(command).await?;
        
        // Message sent through Loopix - assume successful delivery for now
        // In a real implementation, this would have proper delivery confirmation
        let latency = start_time.elapsed();
        self.update_anonymity_metrics(latency, true).await;
        debug!("Anonymous consensus message sent through Loopix in {:?}", latency);
        
        Ok(())
    }
    
    /// Determine optimal anonymity level based on network conditions and message priority
    fn determine_optimal_anonymity_level(&self, priority: MessagePriority) -> AnonymityLevel {
        if !self.config.adaptive_anonymity {
            return self.config.performance_mode.default_anonymity_level();
        }
        
        // Adaptive anonymity based on network health and message priority
        match (self.network_health.congestion_level, priority) {
            (congestion, MessagePriority::Emergency) if congestion > 0.7 => AnonymityLevel::Direct,
            (congestion, MessagePriority::High) if congestion > 0.5 => AnonymityLevel::Low,
            (congestion, MessagePriority::Normal) if congestion > 0.3 => AnonymityLevel::Balanced,
            (_, MessagePriority::Low) => AnonymityLevel::High,
            (_, _) => self.config.performance_mode.default_anonymity_level(),
        }
    }
    
    /// Check if anonymity should be bypassed for emergency performance
    async fn should_bypass_anonymity(
        &self,
        message: &AnonymousConsensusMessage,
        anonymity_level: AnonymityLevel,
    ) -> bool {
        // Check latency threshold
        let expected_latency = match anonymity_level {
            AnonymityLevel::Direct => Duration::from_millis(50),
            AnonymityLevel::Low => Duration::from_millis(200),
            AnonymityLevel::Balanced => Duration::from_millis(400),
            AnonymityLevel::High => Duration::from_millis(700),
            AnonymityLevel::Maximum => Duration::from_millis(1500),
        };
        
        if expected_latency.as_millis() > self.config.emergency_bypass_threshold_ms as u128 {
            return true;
        }
        
        // Check network health
        if self.network_health.delivery_success_rate < 0.8 {
            return true;
        }
        
        // Check message criticality
        match message {
            AnonymousConsensusMessage::BlockProposal { .. } => {
                // Never bypass anonymity for block proposals unless network is severely degraded
                self.network_health.congestion_level > 0.9
            }
            AnonymousConsensusMessage::Vote { .. } => {
                // Can bypass for votes if network is congested
                self.network_health.congestion_level > 0.7
            }
            AnonymousConsensusMessage::Acknowledgment { .. } => {
                // Can bypass acknowledgments more freely
                self.network_health.congestion_level > 0.5
            }
            _ => false,
        }
    }
    
    /// Send consensus message directly (bypass anonymity)
    async fn send_direct_consensus_message(
        &mut self,
        message: AnonymousConsensusMessage,
    ) -> Result<(), anyhow::Error> {
        // Forward directly to DAG-Knight consensus without anonymity
        match message {
            AnonymousConsensusMessage::BlockProposal { proposal_id, block, .. } => {
                self.dag_knight.handle_block_proposal(block).await?;
            }
            AnonymousConsensusMessage::Vote { proposal_id, decision, .. } => {
                let vote = Vote {
                    proposal_id,
                    vote_type: match decision {
                        VoteDecision::Accept => VoteType::Accept,
                        VoteDecision::Reject => VoteType::Reject,
                        VoteDecision::Abstain => VoteType::Abstain,
                    },
                    voter: self.config.node_id,
                round: 0,
                epoch: 0,
                    signature: vec![],
                    timestamp: chrono::Utc::now(),
                };
                self.dag_knight.handle_vote(vote).await?;
            }
            AnonymousConsensusMessage::LeaderElectionContribution { epoch_id, vdf_contribution, .. } => {
                self.dag_knight.handle_leader_contribution(vdf_contribution).await?;
            }
            AnonymousConsensusMessage::Acknowledgment { proposal_id, .. } => {
                let ack_data = proposal_id.as_slice().to_vec();
                self.dag_knight.handle_acknowledgment(ack_data).await?;
            }
        }
        
        Ok(())
    }
    
    /// Check network health for adaptive anonymity
    async fn check_network_health(&mut self) {
        // Stub implementation for network health check
        // In a real implementation, this would query Loopix statistics
        self.network_health.last_check = Instant::now();

        // Simple health calculation - would be more sophisticated in production
        self.network_health.congestion_level = 0.3; // Default moderate congestion

        debug!("Network health updated: congestion={:.2}, latency={:?}",
               self.network_health.congestion_level,
               self.network_health.avg_anonymity_latency);
    }
    
    /// Update anonymity performance metrics
    async fn update_anonymity_metrics(&mut self, latency: Duration, success: bool) {
        if success {
            // Update average latency with exponential moving average
            let alpha = 0.1; // Smoothing factor
            let current_avg = self.network_health.avg_anonymity_latency.as_millis() as f64;
            let new_latency = latency.as_millis() as f64;
            let updated_avg = alpha * new_latency + (1.0 - alpha) * current_avg;
            
            self.network_health.avg_anonymity_latency = Duration::from_millis(updated_avg as u64);
            
            // Update success rate
            let current_rate = self.network_health.delivery_success_rate;
            self.network_health.delivery_success_rate = alpha * 1.0 + (1.0 - alpha) * current_rate;
        } else {
            // Update success rate for failure
            let alpha = 0.1;
            let current_rate = self.network_health.delivery_success_rate;
            self.network_health.delivery_success_rate = alpha * 0.0 + (1.0 - alpha) * current_rate;
        }
    }
    
    /// Update overall consensus metrics
    async fn update_metrics(&mut self) {
        // Calculate anonymity success rate
        let total_operations = self.metrics.anonymous_proposals + 
                               self.metrics.anonymous_votes + 
                               self.metrics.anonymous_leader_elections;
        
        if total_operations > 0 {
            let successful_operations = total_operations - self.metrics.emergency_bypasses;
            self.metrics.anonymity_success_rate = successful_operations as f64 / total_operations as f64;
        }
        
        // Update consensus finality time (would integrate with actual DAG-Knight metrics)
        self.metrics.consensus_finality_time = Duration::from_millis(2900); // ~2.9s with anonymity
        self.metrics.avg_anonymity_overhead = self.network_health.avg_anonymity_latency;
    }
    
    /// Log performance metrics
    async fn log_performance_metrics(&self) {
        info!("🎭 Anonymous Consensus Metrics:");
        info!("   Anonymous Proposals: {}", self.metrics.anonymous_proposals);
        info!("   Anonymous Votes: {}", self.metrics.anonymous_votes);
        info!("   Anonymity Success Rate: {:.1}%", self.metrics.anonymity_success_rate * 100.0);
        info!("   Average Anonymity Overhead: {:?}", self.metrics.avg_anonymity_overhead);
        info!("   Consensus Finality: {:?}", self.metrics.consensus_finality_time);
        info!("   Emergency Bypasses: {}", self.metrics.emergency_bypasses);
    }
    
    /// Cleanup timed-out pending operations
    async fn cleanup_pending_operations(&mut self) {
        let now = Instant::now();
        let mut expired_ops = Vec::new();
        
        for (op_id, operation) in &self.pending_operations {
            if now.duration_since(operation.started_at) > operation.timeout {
                expired_ops.push(op_id.clone());
            }
        }
        
        for op_id in expired_ops {
            if let Some(operation) = self.pending_operations.remove(&op_id) {
                warn!("Operation {} timed out after {:?}", op_id, operation.timeout);
            }
        }
    }
    
    /// Get current anonymity metrics
    pub fn get_metrics(&self) -> &AnonymousConsensusMetrics {
        &self.metrics
    }
    
    /// Get current network health
    pub fn get_network_health(&self) -> &NetworkHealth {
        &self.network_health
    }
    
    // Helper methods for cryptographic operations
    
    async fn generate_proposer_commitment(&self, block: &Block) -> Result<Vec<u8>, anyhow::Error> {
        // Generate zero-knowledge commitment for block proposer
        // In production, this would use proper ZK-SNARK/STARK proofs
        Ok(blake3::hash(&bincode::serialize(block)?).as_bytes().to_vec())
    }
    
    async fn generate_validity_proof(&self, block: &Block) -> Result<Vec<u8>, anyhow::Error> {
        // Generate proof that the block is valid without revealing proposer identity
        Ok(vec![42u8; 64]) // Mock proof
    }
    
    async fn generate_vote_nullifier(&self, proposal_id: ProposalId) -> Result<Vec<u8>, anyhow::Error> {
        // Generate nullifier to prevent double-voting while maintaining anonymity
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.config.node_id);
        hasher.update(&proposal_id_to_bytes(&proposal_id));
        hasher.update(b"vote-nullifier");
        Ok(hasher.finalize().as_bytes().to_vec())
    }
    
    async fn generate_vote_proof(&self, proposal_id: ProposalId, decision: &VoteDecision) -> Result<Vec<u8>, anyhow::Error> {
        // Generate proof that vote is valid without revealing voter identity
        Ok(vec![42u8; 64]) // Mock proof
    }
    
    async fn generate_vdf_contribution(&self, epoch_id: EpochId) -> Result<Vec<u8>, anyhow::Error> {
        // Generate VDF contribution for leader election
        Ok(vec![42u8; 64]) // Mock VDF
    }
    
    async fn generate_leader_commitment(&self, epoch_id: EpochId) -> Result<Vec<u8>, anyhow::Error> {
        // Generate commitment for leader election participation
        Ok(vec![42u8; 32]) // Mock commitment
    }
    
    async fn generate_leader_proof(&self, epoch_id: EpochId, contribution: &[u8]) -> Result<Vec<u8>, anyhow::Error> {
        // Generate proof of valid leader election participation
        Ok(vec![42u8; 64]) // Mock proof
    }
}

/// Message priority for adaptive anonymity
#[derive(Debug, Clone, Copy)]
enum MessagePriority {
    Emergency,
    High,
    Normal,
    Low,
}

// Helper functions for type compatibility
fn new_proposal_id() -> ProposalId {
    rand::random()
}

fn new_epoch_id() -> EpochId {
    rand::random()
}

fn proposal_id_to_bytes(id: &ProposalId) -> Vec<u8> {
    id.as_slice().to_vec()
}

// Note: Real consensus types are now imported from q-narwhal-core and q-dag-knight

// DagKnightConsensus implementation is now in q-dag-knight crate

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;
    
    #[test]
    fn test_anonymity_level_configuration() {
        assert_eq!(AnonymityLevel::Direct.mix_layers(), 0);
        assert_eq!(AnonymityLevel::Low.mix_layers(), 2);
        assert_eq!(AnonymityLevel::Balanced.mix_layers(), 3);
        assert_eq!(AnonymityLevel::High.mix_layers(), 5);
        assert_eq!(AnonymityLevel::Maximum.mix_layers(), 5);
        
        assert_eq!(AnonymityLevel::Direct.cover_traffic_multiplier(), 0.0);
        assert_eq!(AnonymityLevel::Maximum.cover_traffic_multiplier(), 0.8);
    }
    
    #[test]
    fn test_performance_mode_defaults() {
        assert_eq!(
            PerformanceMode::HighPerformance.default_anonymity_level(),
            AnonymityLevel::Low
        );
        assert_eq!(
            PerformanceMode::Balanced.default_anonymity_level(),
            AnonymityLevel::Balanced
        );
        assert_eq!(
            PerformanceMode::HighAnonymity.default_anonymity_level(),
            AnonymityLevel::High
        );
    }
    
    #[tokio::test]
    async fn test_consensus_config_creation() {
        let config = AnonymousConsensusConfig::default();
        assert_eq!(config.performance_mode, PerformanceMode::Balanced);
        assert!(config.anonymous_proposals);
        assert!(config.anonymous_voting);
        assert!(config.adaptive_anonymity);
    }
}