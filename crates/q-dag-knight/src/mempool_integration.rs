//! Mempool Integration for DAG-Knight Consensus
//!
//! Phase 2B: Integrates Server Beta's production mempool with Server Alpha's
//! DAG vertex creation and consensus protocol.

use crate::{AnchorElectionResult, DAGKnightConsensus, QuantumVDFProof, Vertex, VertexCreator};
use anyhow::Result;
use q_narwhal_core::production_mempool::ProductionMempool;
use q_narwhal_core::tor_broadcast::BroadcastMessage;
use q_types::*;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Convert vertex_creator::Vertex to q_types::Vertex
fn convert_vertex_to_q_types(vertex: &crate::Vertex) -> q_types::Vertex {
    q_types::Vertex {
        id: vertex.id,
        round: vertex.round,
        author: vertex.proposer,
        tx_root: [0u8; 32], // TODO: Compute actual transaction root
        parents: vertex.parents.clone(),
        transactions: Vec::new(), // TODO: Get actual transactions from hashes
        signature: vertex.signature.clone(),
        timestamp: chrono::DateTime::from_timestamp(vertex.timestamp as i64, 0)
            .unwrap_or_else(|| chrono::Utc::now()),
    }
}

/// Mempool-DAG integration coordinator
pub struct MempoolDAGIntegration {
    /// DAG-Knight consensus engine
    consensus: Arc<DAGKnightConsensus>,

    /// Production mempool (from Server Beta Phase 2A)
    mempool: Arc<ProductionMempool>,

    /// Configuration for integration
    config: IntegrationConfig,

    /// Integration metrics
    metrics: Arc<RwLock<IntegrationMetrics>>,
}

/// Configuration for mempool-DAG integration
#[derive(Debug, Clone)]
pub struct IntegrationConfig {
    /// How often to create new vertices (block time)
    pub vertex_creation_interval: Duration,

    /// Maximum transactions per vertex
    pub max_transactions_per_vertex: usize,

    /// Timeout for VDF computation
    pub vdf_timeout: Duration,

    /// Whether to use quantum anchor election
    pub quantum_anchor_enabled: bool,
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            vertex_creation_interval: Duration::from_secs(2), // 2-second block time
            max_transactions_per_vertex: 1000,
            vdf_timeout: Duration::from_secs(3),
            quantum_anchor_enabled: true,
        }
    }
}

/// Metrics for mempool-DAG integration
#[derive(Debug, Default, Clone)]
pub struct IntegrationMetrics {
    pub vertices_created: u64,
    pub transactions_processed: u64,
    pub average_vertex_creation_time: Duration,
    pub average_vdf_computation_time: Duration,
    pub failed_vertex_creations: u64,
    pub mempool_sync_time: Duration,
}

impl MempoolDAGIntegration {
    /// Create new mempool-DAG integration
    pub async fn new(
        consensus: Arc<DAGKnightConsensus>,
        mempool: Arc<ProductionMempool>,
        config: IntegrationConfig,
    ) -> Result<Self> {
        Ok(Self {
            consensus,
            mempool,
            config,
            metrics: Arc::new(RwLock::new(IntegrationMetrics::default())),
        })
    }

    /// Main consensus loop - creates vertices with mempool transactions
    pub async fn run_consensus_loop(&self) -> Result<()> {
        info!("Starting mempool-DAG consensus loop");

        let mut interval = tokio::time::interval(self.config.vertex_creation_interval);

        loop {
            interval.tick().await;

            match self.create_vertex_from_mempool().await {
                Ok(vertex) => {
                    info!(
                        "Created vertex {:?} with {} transactions in round {}",
                        vertex.id,
                        vertex.transactions.len(),
                        vertex.round
                    );

                    // Update metrics
                    let mut metrics = self.metrics.write().await;
                    metrics.vertices_created += 1;
                    metrics.transactions_processed += vertex.transactions.len() as u64;
                }
                Err(e) => {
                    error!("Failed to create vertex from mempool: {:?}", e);
                    self.metrics.write().await.failed_vertex_creations += 1;
                }
            }
        }
    }

    /// Create a DAG vertex using transactions from the production mempool
    pub async fn create_vertex_from_mempool(&self) -> Result<Vertex> {
        let start_time = Instant::now();

        // 1. Get transactions from Server Beta's production mempool
        debug!("Fetching transactions from production mempool...");
        let mempool_start = Instant::now();

        let transactions = self
            .mempool
            .get_transactions_for_block(self.config.max_transactions_per_vertex)
            .await;

        let mempool_time = mempool_start.elapsed();
        self.metrics.write().await.mempool_sync_time = mempool_time;

        info!(
            "Retrieved {} transactions from mempool in {:?}",
            transactions.len(),
            mempool_time
        );

        // 2. Perform quantum anchor election if enabled
        let anchor_result = if self.config.quantum_anchor_enabled {
            // Create a placeholder vertex for anchor election
            let placeholder_vertex = Vertex {
                id: [0u8; 32],
                round: *self.consensus.current_round.read().await,
                proposer: self.consensus.node_id,
                transactions: Vec::new(),
                parents: Vec::new(),
                vdf_proof: QuantumVDFProof {
                    challenge: [0u8; 32],
                    proof: [0u8; 64],
                    quantum_seed: Some([0u8; 32]),
                    computation_time: Duration::from_millis(0),
                    difficulty: 0,
                    entropy_estimate: 0.0,
                    parallel_witnesses: Vec::new(),
                },
                timestamp: chrono::Utc::now().timestamp() as u64,
                signature: vec![],
            };
            Some(
                self.consensus
                    .anchor_election
                    .elect_anchor(
                        *self.consensus.current_round.read().await,
                        &convert_vertex_to_q_types(&placeholder_vertex),
                    )
                    .await?,
            )
        } else {
            None
        };

        // 3. Create vertex using DAG-Knight vertex creator
        debug!("Creating DAG vertex with VDF proof...");
        let vdf_start = Instant::now();

        let vertex = self
            .consensus
            .vertex_creator
            .create_vertex_with_mempool_transactions(&*self.mempool, anchor_result)
            .await?;

        let vdf_time = vdf_start.elapsed();
        let total_time = start_time.elapsed();

        // 4. Broadcast vertex proposal to network
        self.broadcast_vertex_proposal(&vertex).await?;

        // 5. Remove included transactions from mempool
        let tx_hashes: Vec<_> = transactions.iter().map(|tx| tx.hash()).collect();
        self.mempool.remove_included_transactions(&tx_hashes).await;

        // 6. Update consensus state
        self.update_consensus_state(&vertex).await?;

        // 7. Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.average_vdf_computation_time = vdf_time;
        metrics.average_vertex_creation_time = total_time;

        info!(
            "Vertex creation complete: {} transactions, VDF: {:?}, Total: {:?}",
            vertex.transactions.len(),
            vdf_time,
            total_time
        );

        Ok(vertex)
    }

    /// Broadcast vertex proposal using Server Beta's infrastructure
    async fn broadcast_vertex_proposal(&self, vertex: &Vertex) -> Result<()> {
        debug!("Broadcasting vertex proposal {:?}", vertex.id);

        // Use Server Beta's message infrastructure
        let proposal = BroadcastMessage::BlockProposal {
            vertex_id: vertex.id,
            height: vertex.round,                      // Use round as height
            transactions: vertex.transactions.clone(), // These are already [u8; 32]
            vdf_proof: vec![0u8; 32],                  // TODO: Serialize VDF proof properly
            parent_vertices: vertex.parents.clone(),
            proposer: vertex.proposer,
        };

        // Broadcast via production mempool's network infrastructure
        self.mempool.broadcast_to_all_peers(proposal).await?;

        info!("Vertex proposal {:?} broadcast to network", vertex.id);
        Ok(())
    }

    /// Update consensus state after vertex creation
    async fn update_consensus_state(&self, vertex: &Vertex) -> Result<()> {
        // Add vertex to consensus engine's vertex store
        self.consensus
            .vertex_store
            .add_vertex(convert_vertex_to_q_types(&vertex))
            .await?;

        // Advance consensus round
        let new_round = self.consensus.vertex_creator.advance_round().await?;
        *self.consensus.current_round.write().await = new_round;

        info!("Advanced consensus to round {}", new_round);
        Ok(())
    }

    /// Handle incoming vertex from peer
    pub async fn handle_peer_vertex(&self, vertex: Vertex, from_peer: NodeId) -> Result<()> {
        info!("Received vertex {:?} from peer {:?}", vertex.id, from_peer);

        // 1. Validate vertex structure and VDF proof
        if !self
            .consensus
            .vertex_creator
            .validate_vertex(&vertex)
            .await?
        {
            warn!("Invalid vertex {:?} from peer {:?}", vertex.id, from_peer);
            return Ok(()); // Ignore invalid vertices
        }

        // 2. Add vertex to consensus state
        self.consensus
            .vertex_store
            .add_vertex(convert_vertex_to_q_types(&vertex))
            .await?;

        // 3. Process any new transactions in the vertex
        // (These would be transactions we haven't seen before)
        for tx_hash in &vertex.transactions {
            if !self.mempool.has_transaction(tx_hash).await.unwrap_or(false) {
                // Request transaction data from peer if we don't have it
                self.request_transaction_from_peer(tx_hash, from_peer)
                    .await?;
            }
        }

        // 4. Update commit logic based on new vertex
        self.consensus
            .commit_protocol
            .process_vertex(&convert_vertex_to_q_types(&vertex))
            .await?;

        info!("Processed vertex {:?} from peer {:?}", vertex.id, from_peer);
        Ok(())
    }

    /// Request transaction data from peer
    async fn request_transaction_from_peer(
        &self,
        tx_hash: &TxHash,
        from_peer: NodeId,
    ) -> Result<()> {
        debug!(
            "Requesting transaction {:?} from peer {:?}",
            tx_hash, from_peer
        );

        let request = BroadcastMessage::TransactionRequest {
            tx_hashes: vec![format!("{:?}", tx_hash)],
            requestor: format!("{:?}", self.consensus.node_id),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs(),
        };

        self.mempool.send_to_peer(from_peer, request).await?;
        Ok(())
    }

    /// Get current integration metrics
    pub async fn get_metrics(&self) -> IntegrationMetrics {
        (*self.metrics.read().await).clone()
    }

    /// Get consensus status
    pub async fn get_consensus_status(&self) -> ConsensusStatus {
        let current_round = *self.consensus.current_round.read().await;
        let last_commit = *self.consensus.last_commit_round.read().await;
        let mempool_size = self.mempool.get_pending_count().await;

        ConsensusStatus {
            current_round,
            last_commit_round: last_commit,
            pending_vertices: (current_round - last_commit) as usize,
            mempool_size: mempool_size as usize,
            node_id: self.consensus.node_id,
        }
    }
}

/// Current consensus status
#[derive(Debug, Clone)]
pub struct ConsensusStatus {
    pub current_round: Round,
    pub last_commit_round: Round,
    pub pending_vertices: usize,
    pub mempool_size: usize,
    pub node_id: NodeId,
}

#[cfg(test)]
mod tests {
    use super::*;
    use q_types::Phase;

    #[tokio::test]
    async fn test_mempool_dag_integration() {
        // This would test the full integration between Server Beta's mempool
        // and Server Alpha's DAG vertex creation

        // For now, we'll create a minimal test
        let node_id = NodeId::default();
        let consensus = Arc::new(DAGKnightConsensus::new(node_id, 1).await.unwrap());

        // In a real test, we'd create a mempool with test transactions
        // and verify vertex creation works correctly

        assert_eq!(consensus.node_id, node_id);
    }
}
