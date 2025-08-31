/// Q-DAG-Knight: Quantum-enhanced Bullshark consensus with DAG-Knight ordering
/// Zero-message complexity asynchronous BFT with quantum anchor election

use q_types::*;
use q_narwhal_core::{Certificate, VertexStore, reliable_broadcast::BroadcastStats};
use anyhow::Result;
use async_trait::async_trait;
use std::collections::{HashMap, HashSet, BTreeSet};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

pub mod anchor_election;
pub mod ordering_rules;
pub mod quantum_beacon;
pub mod commit_logic;

pub use anchor_election::{QuantumAnchorElection, AnchorElectionResult};
pub use ordering_rules::OrderingEngine;
pub use quantum_beacon::{QuantumBeacon, BeaconState};
pub use commit_logic::CommitProtocol;

/// Main DAG-Knight consensus engine
pub struct DAGKnightConsensus {
    pub node_id: NodeId,
    pub vertex_store: VertexStore,
    pub anchor_election: QuantumAnchorElection,
    pub ordering_engine: OrderingEngine,
    pub quantum_beacon: QuantumBeacon,
    pub commit_protocol: CommitProtocol,
    
    // State tracking
    pub committed_vertices: RwLock<HashMap<Round, Vec<VertexId>>>,
    pub pending_commits: RwLock<HashMap<Round, Vec<VertexId>>>,
    pub current_round: RwLock<Round>,
    pub last_commit_round: RwLock<Round>,
    
    // Configuration
    pub f: usize, // Number of Byzantine nodes (2f+1 = total nodes)
    pub delta: u64, // Rounds to look back for commit decision
}

impl DAGKnightConsensus {
    pub fn new(node_id: NodeId, f: usize) -> Result<Self> {
        Ok(Self {
            node_id,
            vertex_store: VertexStore::new(),
            anchor_election: QuantumAnchorElection::new(f)?,
            ordering_engine: OrderingEngine::new()?,
            quantum_beacon: QuantumBeacon::new()?,
            commit_protocol: CommitProtocol::new(f)?,
            committed_vertices: RwLock::new(HashMap::new()),
            pending_commits: RwLock::new(HashMap::new()),
            current_round: RwLock::new(0),
            last_commit_round: RwLock::new(0),
            f,
            delta: 4, // Conservative default
        })
    }

    /// Process a certificate from the Narwhal layer
    pub async fn process_certificate(&self, certificate: Certificate) -> Result<Vec<CommitDecision>> {
        debug!("Processing certificate for vertex {} in round {}", 
               hex::encode(certificate.vertex_id), certificate.round);

        let mut commit_decisions = Vec::new();

        // Update current round if necessary
        {
            let mut current = self.current_round.write().await;
            if certificate.round > *current {
                *current = certificate.round;
                info!("Advanced to round {}", certificate.round);
            }
        }

        // Store vertex if we have it
        if let Some(vertex) = self.vertex_store.get_vertex(&certificate.vertex_id).await {
            // Process through ordering engine
            let ordering_result = self.ordering_engine
                .process_vertex(&vertex, &certificate).await?;

            // Check for anchor election in even rounds
            if certificate.round % 2 == 0 {
                let election_result = self.anchor_election
                    .elect_anchor(certificate.round, &vertex).await?;
                
                if let Some(anchor_vertex_id) = election_result.anchor_vertex_id {
                    info!("Elected anchor {} for round {} with VDF output {:?}", 
                          hex::encode(anchor_vertex_id), certificate.round,
                          election_result.vdf_output);
                    
                    // Apply commit rule
                    let commit_result = self.commit_protocol
                        .evaluate_commit(certificate.round, anchor_vertex_id).await?;
                    
                    if let Some(decision) = commit_result {
                        commit_decisions.push(decision);
                        
                        // Update committed vertices
                        {
                            let mut committed = self.committed_vertices.write().await;
                            committed.entry(certificate.round)
                                .or_insert_with(Vec::new)
                                .push(anchor_vertex_id);
                        }
                        
                        {
                            let mut last_commit = self.last_commit_round.write().await;
                            *last_commit = certificate.round;
                        }
                    }
                }
            }

            // Check for delayed commit decisions (δ rounds back)
            let current_round = *self.current_round.read().await;
            if current_round >= self.delta {
                let check_round = current_round - self.delta;
                let delayed_commits = self.commit_protocol
                    .check_delayed_commits(check_round).await?;
                commit_decisions.extend(delayed_commits);
            }
        }

        Ok(commit_decisions)
    }

    /// Advance to the next round and update quantum beacon
    pub async fn advance_round(&self) -> Result<()> {
        let new_round = {
            let mut current = self.current_round.write().await;
            *current += 1;
            *current
        };

        // Generate new quantum beacon for the round
        let beacon_state = self.quantum_beacon.generate_beacon_state(new_round).await?;
        
        info!("Advanced to round {} with quantum beacon entropy: {}", 
              new_round, hex::encode(&beacon_state.entropy_seed[..8]));

        Ok(())
    }

    /// Get current consensus status
    pub async fn get_status(&self) -> ConsensusStatus {
        let current_round = *self.current_round.read().await;
        let last_commit_round = *self.last_commit_round.read().await;
        let committed_vertices = self.committed_vertices.read().await;
        let pending_commits = self.pending_commits.read().await;

        let total_committed: usize = committed_vertices.values()
            .map(|v| v.len())
            .sum();

        ConsensusStatus {
            current_round,
            last_commit_round,
            committed_vertices: total_committed as u64,
            pending_commits: pending_commits.len() as u64,
            beacon_health: self.quantum_beacon.get_health_metrics().await,
        }
    }

    /// Get vertices committed in a specific round
    pub async fn get_committed_vertices(&self, round: Round) -> Vec<VertexId> {
        let committed = self.committed_vertices.read().await;
        committed.get(&round).cloned().unwrap_or_else(Vec::new)
    }

    /// Check if consensus has progressed beyond a round
    pub async fn is_round_committed(&self, round: Round) -> bool {
        let last_commit = *self.last_commit_round.read().await;
        round <= last_commit
    }

    /// Get ordering of transactions from committed vertices
    pub async fn get_transaction_ordering(&self, from_round: Round, to_round: Round) -> Result<Vec<Transaction>> {
        let mut ordered_transactions = Vec::new();

        for round in from_round..=to_round {
            let committed_vertices = self.get_committed_vertices(round).await;
            
            for vertex_id in committed_vertices {
                if let Some(vertex) = self.vertex_store.get_vertex(&vertex_id).await {
                    // Add transactions in deterministic order
                    let mut txs = vertex.transactions;
                    txs.sort_by(|a, b| a.id.cmp(&b.id)); // Deterministic ordering
                    ordered_transactions.extend(txs);
                }
            }
        }

        Ok(ordered_transactions)
    }

    /// Performance and health metrics
    pub async fn get_metrics(&self) -> ConsensusMetrics {
        let status = self.get_status().await;
        let vertex_stats = self.vertex_store.get_stats().await;
        let anchor_stats = self.anchor_election.get_statistics().await;
        let commit_stats = self.commit_protocol.get_statistics().await;

        ConsensusMetrics {
            current_round: status.current_round,
            commit_latency_rounds: status.current_round - status.last_commit_round,
            total_vertices: vertex_stats.total_vertices,
            committed_vertices: status.committed_vertices,
            anchor_elections: anchor_stats.total_elections,
            successful_commits: commit_stats.successful_commits,
            throughput_tps: commit_stats.average_tps,
            quantum_entropy_quality: status.beacon_health.entropy_quality,
        }
    }
}

/// Result of a commit decision
#[derive(Debug, Clone)]
pub struct CommitDecision {
    pub round: Round,
    pub vertex_id: VertexId,
    pub commit_type: CommitType,
    pub transactions: Vec<Transaction>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone)]
pub enum CommitType {
    AnchorCommit,   // Committed due to anchor election
    DelayedCommit,  // Committed after δ rounds
    ChainCommit,    // Committed due to causal dependency
}

/// Consensus engine status
#[derive(Debug, Clone, serde::Serialize)]
pub struct ConsensusStatus {
    pub current_round: Round,
    pub last_commit_round: Round,
    pub committed_vertices: u64,
    pub pending_commits: u64,
    pub beacon_health: quantum_beacon::BeaconHealth,
}

/// Performance metrics for monitoring
#[derive(Debug, Clone, serde::Serialize)]
pub struct ConsensusMetrics {
    pub current_round: Round,
    pub commit_latency_rounds: u64,
    pub total_vertices: u64,
    pub committed_vertices: u64,
    pub anchor_elections: u64,
    pub successful_commits: u64,
    pub throughput_tps: f64,
    pub quantum_entropy_quality: f64,
}

/// Trait for consensus event handlers
#[async_trait]
pub trait ConsensusEventHandler: Send + Sync {
    async fn on_vertex_committed(&self, decision: &CommitDecision) -> Result<()>;
    async fn on_anchor_elected(&self, round: Round, vertex_id: VertexId) -> Result<()>;
    async fn on_round_advanced(&self, round: Round) -> Result<()>;
    async fn on_consensus_stall(&self, round: Round) -> Result<()>;
}

/// Default event handler that logs events
pub struct LoggingEventHandler;

#[async_trait]
impl ConsensusEventHandler for LoggingEventHandler {
    async fn on_vertex_committed(&self, decision: &CommitDecision) -> Result<()> {
        info!("Committed vertex {} in round {} ({:?})", 
              hex::encode(decision.vertex_id), decision.round, decision.commit_type);
        Ok(())
    }

    async fn on_anchor_elected(&self, round: Round, vertex_id: VertexId) -> Result<()> {
        info!("Anchor elected for round {}: {}", round, hex::encode(vertex_id));
        Ok(())
    }

    async fn on_round_advanced(&self, round: Round) -> Result<()> {
        debug!("Advanced to round {}", round);
        Ok(())
    }

    async fn on_consensus_stall(&self, round: Round) -> Result<()> {
        warn!("Consensus appears to be stalled at round {}", round);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_vertex(id: u8, round: Round) -> Vertex {
        Vertex {
            id: [id; 32],
            round,
            author: [1; 32],
            tx_root: [0; 32],
            parents: vec![],
            transactions: vec![],
            signature: vec![],
            timestamp: Utc::now(),
        }
    }

    fn create_test_certificate(vertex_id: VertexId, round: Round) -> Certificate {
        Certificate {
            vertex_id,
            round,
            signatures: std::collections::BTreeMap::new(),
            threshold_met: true,
        }
    }

    #[tokio::test]
    async fn test_consensus_creation() {
        let node_id = [1u8; 32];
        let consensus = DAGKnightConsensus::new(node_id, 1);
        assert!(consensus.is_ok());
        
        let consensus = consensus.unwrap();
        assert_eq!(consensus.node_id, node_id);
        assert_eq!(consensus.f, 1);
    }

    #[tokio::test]
    async fn test_round_advancement() {
        let node_id = [1u8; 32];
        let consensus = DAGKnightConsensus::new(node_id, 1).unwrap();
        
        let initial_round = *consensus.current_round.read().await;
        assert_eq!(initial_round, 0);
        
        consensus.advance_round().await.unwrap();
        
        let new_round = *consensus.current_round.read().await;
        assert_eq!(new_round, 1);
    }

    #[tokio::test]
    async fn test_certificate_processing() {
        let node_id = [1u8; 32];
        let consensus = DAGKnightConsensus::new(node_id, 1).unwrap();
        
        // Create and store a vertex
        let vertex = create_test_vertex(42, 1);
        consensus.vertex_store.store_vertex(vertex.clone()).await.unwrap();
        
        // Create certificate
        let certificate = create_test_certificate(vertex.id, vertex.round);
        
        // Process certificate
        let decisions = consensus.process_certificate(certificate).await.unwrap();
        
        // Should have advanced round
        let current_round = *consensus.current_round.read().await;
        assert!(current_round >= 1);
    }

    #[tokio::test]
    async fn test_status_reporting() {
        let node_id = [1u8; 32];
        let consensus = DAGKnightConsensus::new(node_id, 1).unwrap();
        
        let status = consensus.get_status().await;
        assert_eq!(status.current_round, 0);
        assert_eq!(status.last_commit_round, 0);
        assert_eq!(status.committed_vertices, 0);
    }

    #[tokio::test]
    async fn test_metrics_collection() {
        let node_id = [1u8; 32];
        let consensus = DAGKnightConsensus::new(node_id, 1).unwrap();
        
        let metrics = consensus.get_metrics().await;
        assert_eq!(metrics.current_round, 0);
        assert_eq!(metrics.committed_vertices, 0);
        assert!(metrics.quantum_entropy_quality >= 0.0);
    }
}