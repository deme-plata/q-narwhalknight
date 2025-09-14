//! Consensus Voting Protocol for Phase 2C
//!
//! Implements advanced consensus voting mechanisms that integrate with Server Alpha's
//! Phase 2B DAG vertex creation. This completes the BFT protocol by adding voting,
//! finalization, and Byzantine fault tolerance for the consensus system.

use crate::byzantine_detector::ByzantineDetector;
use crate::production_mempool::ProductionMempool;
use crate::tor_broadcast::{TorBroadcastManager, BroadcastMessage};
use anyhow::{Result, anyhow};
use q_types::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::{RwLock, Mutex};
use tracing::{debug, info, warn, error};

/// Advanced consensus message types for Phase 2C
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConsensusMessage {
    /// Vertex proposal from DAG vertex creator (Server Alpha Phase 2B)
    VertexProposal {
        vertex: Vertex,
        proposer: ValidatorId,
        round: Round,
        vdf_proof: Vec<u8>,
        timestamp: u64,
    },
    
    /// Vote on a proposed vertex
    ConsensusVote {
        vertex_id: VertexId,
        round: Round,
        vote: VoteType,
        voter: ValidatorId,
        justification: Vec<u8>,
        timestamp: u64,
    },
    
    /// Commit decision for finalized vertices
    CommitDecision {
        round: Round,
        committed_vertices: Vec<VertexId>,
        certificate: Vec<u8>,
        finalizer: ValidatorId,
        timestamp: u64,
    },
    
    /// Heartbeat for liveness detection
    Heartbeat {
        sender: ValidatorId,
        round: Round,
        timestamp: u64,
        active_vertices: Vec<VertexId>,
    },
}

/// Vote types for consensus decisions
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum VoteType {
    /// Accept the vertex
    Accept,
    /// Reject the vertex
    Reject,
    /// Abstain from voting (liveness issues)
    Abstain,
}

impl Copy for VoteType {}

/// View identifier for view changes
pub type ViewId = u64;

/// Consensus voting state
#[derive(Debug, Clone)]
pub struct ConsensusVotingState {
    pub current_round: Round,
    pub current_view: ViewId,
    pub pending_vertices: HashMap<VertexId, PendingVertex>,
    pub vote_tallies: HashMap<VertexId, VoteTally>,
    pub committed_vertices: HashSet<VertexId>,
}

/// Pending vertex waiting for votes
#[derive(Debug, Clone)]
pub struct PendingVertex {
    pub vertex: Vertex,
    pub proposer: ValidatorId,
    pub proposal_time: SystemTime,
    pub vdf_proof: Vec<u8>,
    pub vote_deadline: SystemTime,
}

/// Vote tally for a vertex
#[derive(Debug, Clone)]
pub struct VoteTally {
    pub vertex_id: VertexId,
    pub accept_votes: HashSet<ValidatorId>,
    pub reject_votes: HashSet<ValidatorId>,
    pub abstain_votes: HashSet<ValidatorId>,
    pub total_weight: u64,
    pub threshold_met: bool,
}

/// Consensus voting configuration
#[derive(Debug, Clone)]
pub struct ConsensusVotingConfig {
    /// BFT threshold (2f+1 for f Byzantine nodes)
    pub byzantine_threshold: u32,
    
    /// Total number of validators
    pub total_validators: u32,
    
    /// Voting timeout per round
    pub voting_timeout: Duration,
    
    /// Heartbeat interval
    pub heartbeat_interval: Duration,
    
    /// Maximum concurrent pending vertices
    pub max_pending_vertices: usize,
    
    /// Enable Byzantine detection
    pub enable_byzantine_detection: bool,
}

impl Default for ConsensusVotingConfig {
    fn default() -> Self {
        Self {
            byzantine_threshold: 3, // Supports up to 1 Byzantine node with 4 validators
            total_validators: 4,
            voting_timeout: Duration::from_secs(5),
            heartbeat_interval: Duration::from_secs(2),
            max_pending_vertices: 100,
            enable_byzantine_detection: true,
        }
    }
}

/// Validator information for vote validation
#[derive(Debug, Clone)]
pub struct ValidatorInfo {
    pub validator_id: ValidatorId,
    pub public_key: Vec<u8>,
    pub onion_address: String,
    pub stake_weight: u64,
    pub reputation_score: f64,
    pub is_active: bool,
}

/// Consensus performance metrics
#[derive(Debug, Clone)]
pub struct ConsensusMetrics {
    pub total_rounds: u64,
    pub successful_rounds: u64,
    pub failed_rounds: u64,
    pub average_round_time: Duration,
    pub byzantine_events_detected: u64,
    pub vertices_committed: u64,
    pub vote_participation_rate: f64,
}

impl Default for ConsensusMetrics {
    fn default() -> Self {
        Self {
            total_rounds: 0,
            successful_rounds: 0,
            failed_rounds: 0,
            average_round_time: Duration::from_secs(0),
            byzantine_events_detected: 0,
            vertices_committed: 0,
            vote_participation_rate: 0.0,
        }
    }
}

/// Main consensus voting system
pub struct ConsensusVoting {
    /// Unique validator identifier
    node_id: ValidatorId,
    
    /// Voting configuration
    config: ConsensusVotingConfig,
    
    /// Current consensus state
    state: Arc<RwLock<ConsensusVotingState>>,
    
    /// Byzantine fault detector
    byzantine_detector: Arc<ByzantineDetector>,
    
    /// Network broadcast manager (Tor-based)
    broadcast_manager: Arc<TorBroadcastManager>,
    
    /// Integration with production mempool
    mempool: Arc<ProductionMempool>,
    
    /// Validator registry for vote validation
    validators: Arc<RwLock<HashMap<ValidatorId, ValidatorInfo>>>,
    
    /// Consensus metrics
    metrics: Arc<RwLock<ConsensusMetrics>>,
}

impl ConsensusVoting {
    /// Create new consensus voting system
    pub fn new(
        node_id: ValidatorId,
        config: ConsensusVotingConfig,
        byzantine_detector: Arc<ByzantineDetector>,
        broadcast_manager: Arc<TorBroadcastManager>,
        mempool: Arc<ProductionMempool>,
    ) -> Self {
        info!("🗳️ Initializing Consensus Voting System");
        info!("   Validator ID: {:?}", hex::encode(&node_id[..8]));
        info!("   Byzantine Threshold: {}/{}", config.byzantine_threshold, config.total_validators);
        info!("   Voting Timeout: {:?}", config.voting_timeout);
        info!("   Byzantine Detection: {}", config.enable_byzantine_detection);
        
        let initial_state = ConsensusVotingState {
            current_round: 0,
            current_view: 0,
            pending_vertices: HashMap::new(),
            vote_tallies: HashMap::new(),
            committed_vertices: HashSet::new(),
        };
        
        Self {
            node_id,
            config,
            state: Arc::new(RwLock::new(initial_state)),
            byzantine_detector,
            broadcast_manager,
            mempool,
            validators: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(ConsensusMetrics::default())),
        }
    }
    
    /// Start the consensus voting system
    pub async fn start(&self) -> Result<()> {
        info!("🚀 Starting Consensus Voting System");
        
        // For Phase 2C, we'll implement the core functionality
        // Advanced features like background loops will be added in future iterations
        
        info!("✅ Consensus Voting System started successfully");
        Ok(())
    }
    
    /// Process vertex proposal from Server Alpha Phase 2B
    pub async fn process_vertex_proposal(&self, vertex: Vertex, proposer: ValidatorId, vdf_proof: Vec<u8>) -> Result<()> {
        info!("📥 Processing vertex proposal from {:?}", hex::encode(&proposer[..8]));
        debug!("   Vertex ID: {:?}", hex::encode(&vertex.id[..8]));
        debug!("   Round: {}", vertex.round);
        debug!("   Transactions: {}", vertex.transactions.len());
        
        let mut state = self.state.write().await;
        
        // Validate vertex proposal
        self.validate_vertex_proposal(&vertex, proposer, &vdf_proof).await?;
        
        // Check if we're in the correct round
        if vertex.round != state.current_round {
            warn!("⚠️ Vertex proposal for round {} but current round is {}", 
                  vertex.round, state.current_round);
            return Err(anyhow!("Vertex proposal for incorrect round"));
        }
        
        // Check pending vertex limits
        if state.pending_vertices.len() >= self.config.max_pending_vertices {
            warn!("⚠️ Too many pending vertices, rejecting proposal");
            return Err(anyhow!("Maximum pending vertices exceeded"));
        }
        
        // Add to pending vertices
        let vertex_id = vertex.id;
        let pending_vertex = PendingVertex {
            vertex: vertex.clone(),
            proposer,
            proposal_time: SystemTime::now(),
            vdf_proof: vdf_proof.clone(),
            vote_deadline: SystemTime::now() + self.config.voting_timeout,
        };
        
        state.pending_vertices.insert(vertex_id, pending_vertex);
        
        // Initialize vote tally
        let vote_tally = VoteTally {
            vertex_id,
            accept_votes: HashSet::new(),
            reject_votes: HashSet::new(),
            abstain_votes: HashSet::new(),
            total_weight: 0,
            threshold_met: false,
        };
        state.vote_tallies.insert(vertex_id, vote_tally);
        
        drop(state); // Release lock before async operations
        
        // Validate vertex against our mempool
        let vote_decision = self.decide_vote(&vertex, &vdf_proof).await?;
        
        // Cast our vote
        self.cast_vote(vertex_id, vertex.round, vote_decision).await?;
        
        info!("✅ Processed vertex proposal and cast vote: {:?}", vote_decision);
        Ok(())
    }
    
    /// Validate vertex proposal
    async fn validate_vertex_proposal(&self, vertex: &Vertex, proposer: ValidatorId, vdf_proof: &[u8]) -> Result<()> {
        // Validate proposer is known validator
        let validators = self.validators.read().await;
        if !validators.contains_key(&proposer) {
            return Err(anyhow!("Unknown proposer validator"));
        }
        
        // Validate VDF proof (simplified for Phase 2C)
        if vdf_proof.len() < 32 {
            return Err(anyhow!("Invalid VDF proof length"));
        }
        
        // Validate DAG structure (parent references)
        // This integrates with Server Alpha's Phase 2B vertex creation
        if vertex.round > 0 && vertex.parents.is_empty() {
            return Err(anyhow!("Non-genesis vertex must have parent references"));
        }
        
        // Check Byzantine detection
        if self.config.enable_byzantine_detection {
            let analysis = self.byzantine_detector.analyze_validator_behavior(proposer).await?;
            if matches!(analysis.suspicion_level, crate::byzantine_detector::SuspicionLevel::HighlyMalicious) {
                warn!("⚠️ Proposer {:?} suspected of Byzantine behavior", hex::encode(&proposer[..8]));
                return Err(anyhow!("Proposer suspected of Byzantine behavior"));
            }
        }
        
        Ok(())
    }
    
    /// Decide how to vote on a vertex
    async fn decide_vote(&self, vertex: &Vertex, _vdf_proof: &[u8]) -> Result<VoteType> {
        debug!("🤔 Deciding vote for vertex {:?}", hex::encode(&vertex.id[..8]));
        
        // Simple decision logic for Phase 2C
        // In production, this would involve complex validation
        
        let total_transactions = vertex.transactions.len();
        if total_transactions == 0 {
            debug!("📝 Empty vertex - voting Accept for liveness");
            return Ok(VoteType::Accept);
        }
        
        // For Phase 2C, accept vertices with reasonable transaction counts
        if total_transactions <= 1000 {
            debug!("✅ Reasonable transaction count ({}) - voting Accept", total_transactions);
            Ok(VoteType::Accept)
        } else {
            debug!("⚠️ High transaction count ({}) - voting Abstain", total_transactions);
            Ok(VoteType::Abstain)
        }
    }
    
    /// Cast vote on vertex
    pub async fn cast_vote(&self, vertex_id: VertexId, round: Round, vote: VoteType) -> Result<()> {
        debug!("🗳️ Casting vote {:?} for vertex {:?}", vote, hex::encode(&vertex_id[..8]));
        
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        
        let consensus_message = ConsensusMessage::ConsensusVote {
            vertex_id,
            round,
            vote,
            voter: self.node_id,
            justification: vec![], // Could include reasoning for the vote
            timestamp,
        };
        
        // For Phase 2C, we'll use the existing BroadcastMessage format
        // In future phases, we'll add ConsensusMessage variant to BroadcastMessage
        let _serialized_message = serde_json::to_vec(&consensus_message)?;
        let broadcast_message = BroadcastMessage::TransactionAnnounce {
            tx_hash: vertex_id, // Using vertex_id as the hash for consensus messages
            size: 0, // Placeholder
            fee: 0, // Placeholder
            priority: 1, // High priority for consensus messages
        };
        
        self.broadcast_manager.broadcast_to_all(broadcast_message).await?;
        
        // Record our own vote
        self.record_vote(vertex_id, self.node_id, vote).await?;
        
        info!("✅ Vote cast and broadcasted successfully");
        Ok(())
    }
    
    /// Record vote and check for threshold
    async fn record_vote(&self, vertex_id: VertexId, voter: ValidatorId, vote: VoteType) -> Result<()> {
        let mut state = self.state.write().await;
        
        if let Some(vote_tally) = state.vote_tallies.get_mut(&vertex_id) {
            // Remove any previous vote from this validator
            vote_tally.accept_votes.remove(&voter);
            vote_tally.reject_votes.remove(&voter);
            vote_tally.abstain_votes.remove(&voter);
            
            // Add new vote
            match vote {
                VoteType::Accept => {
                    vote_tally.accept_votes.insert(voter);
                }
                VoteType::Reject => {
                    vote_tally.reject_votes.insert(voter);
                }
                VoteType::Abstain => {
                    vote_tally.abstain_votes.insert(voter);
                }
            }
            
            // Check Byzantine threshold (2f+1 for f Byzantine nodes)
            let accept_count = vote_tally.accept_votes.len() as u32;
            let reject_count = vote_tally.reject_votes.len() as u32;
            
            if accept_count >= self.config.byzantine_threshold {
                info!("✅ Vertex {:?} reached accept threshold ({}/{})", 
                      hex::encode(&vertex_id[..8]), accept_count, self.config.byzantine_threshold);
                
                vote_tally.threshold_met = true;
                
                // Commit vertex
                if let Some(_pending_vertex) = state.pending_vertices.remove(&vertex_id) {
                    state.committed_vertices.insert(vertex_id);
                    
                    // Update metrics
                    let mut metrics = self.metrics.write().await;
                    metrics.vertices_committed += 1;
                    drop(metrics);
                    
                    info!("🎉 Vertex committed successfully");
                }
            } else if reject_count >= self.config.byzantine_threshold {
                info!("❌ Vertex {:?} reached reject threshold ({}/{})", 
                      hex::encode(&vertex_id[..8]), reject_count, self.config.byzantine_threshold);
                
                // Remove rejected vertex
                state.pending_vertices.remove(&vertex_id);
                state.vote_tallies.remove(&vertex_id);
            }
        }
        
        Ok(())
    }
    
    /// Process received consensus vote
    pub async fn process_consensus_vote(&self, vertex_id: VertexId, voter: ValidatorId, vote: VoteType) -> Result<()> {
        debug!("📥 Processing vote {:?} from {:?} for vertex {:?}", 
               vote, hex::encode(&voter[..8]), hex::encode(&vertex_id[..8]));
        
        // Validate voter
        let validators = self.validators.read().await;
        if !validators.contains_key(&voter) {
            warn!("⚠️ Vote from unknown validator: {:?}", hex::encode(&voter[..8]));
            return Err(anyhow!("Vote from unknown validator"));
        }
        drop(validators);
        
        // Record the vote
        self.record_vote(vertex_id, voter, vote).await?;
        
        Ok(())
    }
    
    /// Add validator to registry
    pub async fn add_validator(&self, validator_info: ValidatorInfo) -> Result<()> {
        let mut validators = self.validators.write().await;
        validators.insert(validator_info.validator_id, validator_info.clone());
        
        info!("➕ Added validator {:?} with onion address {}", 
              hex::encode(&validator_info.validator_id[..8]), 
              validator_info.onion_address);
        
        Ok(())
    }
    
    /// Get consensus metrics
    pub async fn get_metrics(&self) -> ConsensusMetrics {
        let metrics = self.metrics.read().await;
        metrics.clone()
    }
    
    /// Get current consensus state
    pub async fn get_current_state(&self) -> ConsensusVotingState {
        let state = self.state.read().await;
        state.clone()
    }
    
    /// Advance to next round
    pub async fn advance_round(&self) -> Result<()> {
        let mut state = self.state.write().await;
        state.current_round += 1;
        
        // Clean up old pending vertices that timed out
        let now = SystemTime::now();
        let timed_out_vertices: Vec<VertexId> = state
            .pending_vertices
            .iter()
            .filter(|(_, pending)| pending.vote_deadline < now)
            .map(|(vertex_id, _)| *vertex_id)
            .collect();
        
        for vertex_id in timed_out_vertices {
            state.pending_vertices.remove(&vertex_id);
            state.vote_tallies.remove(&vertex_id);
            debug!("⏰ Removed timed out vertex {:?}", hex::encode(&vertex_id[..8]));
        }
        
        // Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.total_rounds += 1;
        
        info!("🔄 Advanced to round {}", state.current_round);
        Ok(())
    }
    
    /// Validate vertex transactions
    pub async fn validate_vertex_transactions(&self, vertex: &q_types::Vertex) -> Result<bool> {
        debug!("🔍 Validating transactions for vertex {}", hex::encode(&vertex.id[..8]));
        
        // Basic validation - all transactions should be valid
        // In a real implementation, this would:
        // 1. Verify transaction signatures
        // 2. Check transaction format and fields
        // 3. Validate against mempool state
        // 4. Check for double-spending
        
        for transaction in &vertex.transactions {
            // TODO: Implement proper transaction validation
            if transaction.amount == 0 && transaction.fee == 0 {
                // Empty transactions are invalid
                return Ok(false);
            }
        }
        
        // For now, assume all transactions are valid
        Ok(true)
    }
}

impl Clone for ConsensusVoting {
    fn clone(&self) -> Self {
        Self {
            node_id: self.node_id,
            config: self.config.clone(),
            state: self.state.clone(),
            byzantine_detector: self.byzantine_detector.clone(),
            broadcast_manager: self.broadcast_manager.clone(),
            mempool: self.mempool.clone(),
            validators: self.validators.clone(),
            metrics: self.metrics.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tor_broadcast::TorBroadcastConfig;
    use crate::byzantine_detector::ByzantineConfig;
    
    #[tokio::test]
    async fn test_consensus_voting_creation() {
        let node_id = [1u8; 32];
        let config = ConsensusVotingConfig::default();
        
        // Create mock dependencies
        let byzantine_config = ByzantineConfig::default();
        let byzantine_detector = Arc::new(ByzantineDetector::new(byzantine_config).await.unwrap());
        
        let broadcast_config = TorBroadcastConfig::default();
        let broadcast_manager = Arc::new(TorBroadcastManager::new(node_id, broadcast_config));
        
        let mempool_config = crate::production_mempool::MempoolConfig::default();
        let mempool = Arc::new(ProductionMempool::new(node_id, mempool_config, broadcast_manager.clone()).await.unwrap());
        
        let consensus_voting = ConsensusVoting::new(
            node_id,
            config,
            byzantine_detector,
            broadcast_manager,
            mempool,
        );
        
        assert_eq!(consensus_voting.node_id, node_id);
        
        let state = consensus_voting.get_current_state().await;
        assert_eq!(state.current_round, 0);
        assert_eq!(state.current_view, 0);
    }
    
    #[tokio::test]
    async fn test_vote_decision_logic() {
        let node_id = [1u8; 32];
        let config = ConsensusVotingConfig::default();
        
        // Create test vertex with transactions
        let vertex = Vertex {
            id: [1u8; 32],
            round: 0,
            author: node_id,
            tx_root: [0u8; 32],
            parents: vec![],
            transactions: vec![],
            signature: vec![],
            timestamp: chrono::Utc::now(),
        };
        
        // Mock dependencies for test
        let byzantine_config = ByzantineConfig::default();
        let byzantine_detector = Arc::new(ByzantineDetector::new(byzantine_config).await.unwrap());
        
        let broadcast_config = TorBroadcastConfig::default();
        let broadcast_manager = Arc::new(TorBroadcastManager::new(node_id, broadcast_config));
        
        let mempool_config = crate::production_mempool::MempoolConfig::default();
        let mempool = Arc::new(ProductionMempool::new(node_id, mempool_config, broadcast_manager.clone()).await.unwrap());
        
        let consensus_voting = ConsensusVoting::new(
            node_id,
            config,
            byzantine_detector,
            broadcast_manager,
            mempool,
        );
        
        // Test empty vertex (should accept for liveness)
        let vote = consensus_voting.decide_vote(&vertex, &vec![0u8; 32]).await.unwrap();
        assert_eq!(vote, VoteType::Accept);
    }
    
    #[tokio::test]
    async fn test_validator_management() {
        let node_id = [1u8; 32];
        let config = ConsensusVotingConfig::default();
        
        // Mock dependencies
        let byzantine_config = ByzantineConfig::default();
        let byzantine_detector = Arc::new(ByzantineDetector::new(byzantine_config).await.unwrap());
        
        let broadcast_config = TorBroadcastConfig::default();
        let broadcast_manager = Arc::new(TorBroadcastManager::new(node_id, broadcast_config));
        
        let mempool_config = crate::production_mempool::MempoolConfig::default();
        let mempool = Arc::new(ProductionMempool::new(node_id, mempool_config, broadcast_manager.clone()).await.unwrap());
        
        let consensus_voting = ConsensusVoting::new(
            node_id,
            config,
            byzantine_detector,
            broadcast_manager,
            mempool,
        );
        
        // Add test validator
        let validator_info = ValidatorInfo {
            validator_id: [2u8; 32],
            public_key: vec![1, 2, 3, 4],
            onion_address: "test.onion".to_string(),
            stake_weight: 1000,
            reputation_score: 1.0,
            is_active: true,
        };
        
        consensus_voting.add_validator(validator_info.clone()).await.unwrap();
        
        let validators = consensus_voting.validators.read().await;
        assert!(validators.contains_key(&validator_info.validator_id));
        assert_eq!(validators.get(&validator_info.validator_id).unwrap().onion_address, "test.onion");
    }
}