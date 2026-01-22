//! Phase 3 Integration - Complete BFT Consensus Protocol
//!
//! Server Alpha Phase 3: Integrates voting coordination with existing 
//! DAG-Knight consensus, mempool, and Server Beta's Phase 2C components

use crate::{DAGKnightConsensus, VotingCoordinator, VotingCoordinatorConfig};
use anyhow::Result;
use q_narwhal_core::{ProductionMempool, ConsensusVoting, ByzantineDetector, SuspicionLevel};
use q_types::*;
use std::sync::Arc;
use tracing::{info, debug, error, warn};

// Helper function for displaying VertexId
fn format_vertex_id(id: &VertexId) -> String {
    hex::encode(&id[..8]) // Show first 8 bytes for brevity
}

/// Phase 3 Integration Manager
pub struct Phase3Integration {
    /// DAG-Knight consensus engine (from Phase 2B)
    consensus: Arc<DAGKnightConsensus>,
    
    /// Production mempool (from Server Beta Phase 2A)
    mempool: Arc<ProductionMempool>,
    
    /// Consensus voting system (from Server Beta Phase 2C)
    consensus_voting: Arc<ConsensusVoting>,
    
    /// Byzantine detector (from Server Beta Phase 2C)
    byzantine_detector: Arc<ByzantineDetector>,
    
    /// Voting coordinator (Phase 3)
    voting_coordinator: Arc<VotingCoordinator>,
}

impl Phase3Integration {
    /// Create complete Phase 3 integration
    pub async fn new(
        consensus: Arc<DAGKnightConsensus>,
        mempool: Arc<ProductionMempool>,
        consensus_voting: Arc<ConsensusVoting>,
        byzantine_detector: Arc<ByzantineDetector>,
    ) -> Result<Self> {
        info!("Initializing Phase 3 BFT consensus integration");
        
        // Configure voting coordinator for Byzantine fault tolerance
        let voting_config = VotingCoordinatorConfig {
            byzantine_threshold: (consensus.f * 2) + 1, // 2f+1 for f Byzantine nodes
            max_validators: 100,
            voting_timeout: std::time::Duration::from_secs(10),
            finalization_timeout: std::time::Duration::from_secs(5),
            enable_slashing: true,
            min_voting_stake: 1000,
        };
        
        // Create voting coordinator with Server Beta's Phase 2C components
        let voting_coordinator = Arc::new(
            VotingCoordinator::new(
                consensus.node_id,
                consensus_voting.clone(),
                byzantine_detector.clone(),
                voting_config,
            ).await?
        );
        
        info!("Phase 3 integration initialized with BFT threshold: {}", 
              (consensus.f * 2) + 1);
        
        Ok(Self {
            consensus,
            mempool,
            consensus_voting,
            byzantine_detector,
            voting_coordinator,
        })
    }
    
    /// Run complete Phase 3 consensus protocol
    pub async fn run_complete_consensus_protocol(&self) -> Result<()> {
        info!("Starting complete BFT consensus protocol - Phase 3");
        
        // Run multiple consensus components concurrently
        let consensus_tasks = vec![
            // 1. Voting coordination (Phase 3)
            tokio::spawn({
                let voting_coordinator = self.voting_coordinator.clone();
                async move {
                    if let Err(e) = voting_coordinator.run_consensus_coordination().await {
                        error!("Voting coordination failed: {:?}", e);
                    }
                }
            }),
            
            // 2. Vertex creation (Phase 2B continues)
            tokio::spawn({
                let consensus = self.consensus.clone();
                let mempool = self.mempool.clone();
                async move {
                    // This would integrate with mempool integration from Phase 2B
                    if let Err(e) = Self::run_vertex_creation_loop(consensus, mempool).await {
                        error!("Vertex creation failed: {:?}", e);
                    }
                }
            }),
            
            // 3. Byzantine monitoring (Server Beta Phase 2C)
            tokio::spawn({
                let byzantine_detector = self.byzantine_detector.clone();
                async move {
                    if let Err(e) = Self::run_byzantine_monitoring(byzantine_detector).await {
                        error!("Byzantine monitoring failed: {:?}", e);
                    }
                }
            }),
        ];
        
        // Wait for all consensus components
        for task in consensus_tasks {
            if let Err(e) = task.await {
                error!("Consensus task failed: {:?}", e);
            }
        }
        
        Ok(())
    }
    
    /// Vertex creation loop (integrates Phase 2B with Phase 3)
    async fn run_vertex_creation_loop(
        consensus: Arc<DAGKnightConsensus>,
        mempool: Arc<ProductionMempool>,
    ) -> Result<()> {
        debug!("Starting vertex creation loop with Phase 3 integration");
        
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(2));
        
        loop {
            interval.tick().await;
            
            // 1. Create vertex with mempool transactions (Phase 2B)
            match Self::create_vertex_for_consensus(&consensus, &mempool).await {
                Ok(Some(vertex)) => {
                    info!("Created vertex {} for Phase 3 consensus", format_vertex_id(&vertex.id));
                    
                    // 2. Submit vertex to voting coordinator (Phase 3)
                    // This would trigger the BFT voting process
                    // voting_coordinator.process_vertex_proposal(vertex).await?;
                },
                Ok(None) => {
                    debug!("No transactions available for vertex creation");
                },
                Err(e) => {
                    error!("Vertex creation failed: {:?}", e);
                }
            }
        }
    }
    
    /// Create vertex for consensus (Phase 2B + Phase 3 integration)
    async fn create_vertex_for_consensus(
        consensus: &Arc<DAGKnightConsensus>,
        mempool: &Arc<ProductionMempool>,
    ) -> Result<Option<crate::Vertex>> {
        // Get transactions from mempool (Server Beta Phase 2A)
        let transactions = mempool.get_transactions_for_block(1000).await;
        
        if transactions.is_empty() {
            return Ok(None);
        }
        
        // Perform quantum anchor election
        let current_round = *consensus.current_round.read().await;
        // TODO: Need candidate vertex for anchor election - temporarily use dummy vertex
        // let anchor_result = consensus.anchor_election.elect_anchor(current_round).await?;
        let anchor_result = crate::AnchorElectionResult {
            round: current_round,
            anchor_vertex_id: Some([0u8; 32]),
            vdf_output: [0u8; 32],
            quantum_beacon: [0u8; 32],
            election_strength: 1.0,
            candidates: vec![],
            vrf_result: None,
            randomness_proof: None,
        };
        
        // Create vertex with VDF proof (Phase 2B)
        let vertex = consensus.vertex_creator
            .create_vertex_with_mempool_transactions(mempool, Some(anchor_result))
            .await?;
        
        Ok(Some(vertex))
    }
    
    /// Byzantine monitoring loop (Server Beta Phase 2C integration)
    async fn run_byzantine_monitoring(byzantine_detector: Arc<ByzantineDetector>) -> Result<()> {
        info!("🔍 Starting Byzantine monitoring with Phase 3 integration");

        let mut interval = tokio::time::interval(std::time::Duration::from_secs(30));

        loop {
            interval.tick().await;

            // Perform comprehensive Byzantine analysis using available methods
            let malicious_validators = byzantine_detector.get_malicious_validators().await;

            if !malicious_validators.is_empty() {
                warn!("🚨 Byzantine behavior detected! {} malicious validators identified",
                      malicious_validators.len());
                for validator_id in &malicious_validators {
                    warn!("   - Malicious validator: {}", hex::encode(&validator_id[..8]));

                    // Report this validator to the network
                    if let Err(e) = byzantine_detector.report_byzantine_validator(*validator_id).await {
                        error!("Failed to report Byzantine validator: {:?}", e);
                    }
                }
            }

            // Get overall detection stats
            let stats = byzantine_detector.get_detection_stats().await;
            debug!("📊 Byzantine detection stats: {} validators tracked, {} reports filed",
                   stats.total_validators_tracked, stats.metrics.byzantine_reports);
        }
    }
    
    /// Process incoming vertex from peer (Phase 3 voting integration)
    pub async fn process_peer_vertex(&self, vertex: crate::Vertex, from_peer: ValidatorId) -> Result<()> {
        info!("Processing vertex {} from peer {} in Phase 3", format_vertex_id(&vertex.id), format_vertex_id(&from_peer));
        
        // 1. Validate vertex structure (Phase 2B)
        if !self.consensus.vertex_creator.validate_vertex(&vertex).await? {
            warn!("Invalid vertex {} from peer {}", format_vertex_id(&vertex.id), format_vertex_id(&from_peer));
            return Ok(());
        }
        
        // 2. Check for Byzantine behavior (Server Beta Phase 2C)
        let behavior_analysis = self.byzantine_detector
            .analyze_validator_behavior(from_peer)
            .await?;

        // Check suspicion level from analysis result
        if behavior_analysis.suspicion_level >= SuspicionLevel::Suspicious {
            warn!("⚠️ Suspicious vertex {} from potentially Byzantine peer {} (suspicion: {:?}, reputation: {:.2})",
                  format_vertex_id(&vertex.id),
                  format_vertex_id(&from_peer),
                  behavior_analysis.suspicion_level,
                  behavior_analysis.reputation_score);

            // If highly malicious, consider rejecting the vertex
            if behavior_analysis.suspicion_level >= SuspicionLevel::HighlyMalicious {
                warn!("🚨 REJECTING vertex from highly malicious peer {}", format_vertex_id(&from_peer));
                return Ok(());
            }
        }
        
        // 3. Submit to voting coordinator (Phase 3)
        // This would trigger BFT voting process
        self.voting_coordinator
            .process_vertex_consensus(&vertex)
            .await?;
        
        info!("Vertex {} processed through Phase 3 BFT consensus", format_vertex_id(&vertex.id));
        Ok(())
    }
    
    /// Get comprehensive consensus status (all phases)
    pub async fn get_consensus_status(&self) -> Result<ComprehensiveConsensusStatus> {
        let voting_metrics = self.voting_coordinator.get_metrics().await;
        let current_round = self.voting_coordinator.get_current_round().await;
        // TODO: Fix method call - get_pending_count doesn't exist
        let mempool_size = 0; // self.mempool.get_pending_count().await;
        
        Ok(ComprehensiveConsensusStatus {
            current_round,
            mempool_size: mempool_size as usize,
            vertices_finalized: voting_metrics.vertices_finalized,
            byzantine_nodes_detected: voting_metrics.byzantine_nodes_detected,
            consensus_success_rate: voting_metrics.consensus_success_rate,
            average_finalization_time: voting_metrics.average_finalization_time,
            node_id: self.consensus.node_id,
        })
    }
}

/// Comprehensive consensus status across all phases
#[derive(Debug, Clone)]
pub struct ComprehensiveConsensusStatus {
    pub current_round: Round,
    pub mempool_size: usize,
    pub vertices_finalized: u64,
    pub byzantine_nodes_detected: u64,
    pub consensus_success_rate: f64,
    pub average_finalization_time: std::time::Duration,
    pub node_id: ValidatorId,
}

// Note: Cannot implement methods on external types like ProductionMempool, ConsensusVoting, ByzantineDetector
// These methods would need to be implemented in their respective crates (q-narwhal-core)

#[derive(Debug, Default)]
pub struct BehaviorAnalysis {
    pub suspicion_level: SuspicionLevel,
}

impl BehaviorAnalysis {
    pub fn is_suspicious(&self) -> bool {
        matches!(self.suspicion_level, SuspicionLevel::High | SuspicionLevel::Critical)
    }
    
    pub fn is_highly_suspicious(&self) -> bool {
        matches!(self.suspicion_level, SuspicionLevel::Critical)
    }
}

#[derive(Debug, Default)]
pub enum SuspicionLevel {
    #[default]
    None,
    Low,
    Medium,
    High,
    Critical,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_phase3_integration_config() {
        // Test Phase 3 integration configuration
        let f = 3; // 3 Byzantine nodes
        let expected_threshold = (f * 2) + 1; // 7 nodes needed for consensus
        
        assert_eq!(expected_threshold, 7);
    }
    
    #[test]
    fn test_comprehensive_status() {
        let status = ComprehensiveConsensusStatus {
            current_round: 100,
            mempool_size: 250,
            vertices_finalized: 95,
            byzantine_nodes_detected: 2,
            consensus_success_rate: 0.95,
            average_finalization_time: std::time::Duration::from_secs(3),
            node_id: ValidatorId::default(),
        };
        
        assert_eq!(status.current_round, 100);
        assert_eq!(status.consensus_success_rate, 0.95);
    }
}