/// DAG State Synchronization Protocol for Q-NarwhalKnight
/// Ensures all validators have consistent view of the DAG
use anyhow::{Context, Result};
use q_types::{Certificate, NodeId, Round, Transaction, ValidatorId, Vertex, VertexId};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, error, info, warn};

use super::peer_registry::PeerRegistry;
use super::persistent_channels::{
    ChannelMessage, MessagePriority, MessageType, PersistentChannelManager,
};

/// DAG synchronization request message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DagSyncRequest {
    pub request_id: [u8; 16],
    pub requester: ValidatorId,
    pub sync_type: SyncType,
    pub from_round: Round,
    pub to_round: Option<Round>,
    pub missing_vertices: Vec<VertexId>,
    pub timestamp: u64,
}

/// DAG synchronization response message  
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DagSyncResponse {
    pub request_id: [u8; 16],
    pub responder: ValidatorId,
    pub vertices: Vec<Vertex>,
    pub certificates: Vec<Certificate>,
    pub dag_summary: DagStateSummary,
    pub timestamp: u64,
}

/// Types of DAG synchronization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SyncType {
    /// Sync recent rounds
    RecentRounds { last_n_rounds: u64 },
    /// Sync specific round range
    RoundRange { from: Round, to: Round },
    /// Sync missing specific vertices
    MissingVertices { vertex_ids: Vec<VertexId> },
    /// Full DAG state sync (for new validators)
    FullSync,
    /// Heartbeat sync (lightweight status check)
    HeartbeatSync,
}

/// Summary of DAG state for consistency checking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DagStateSummary {
    pub current_round: Round,
    pub total_vertices: u64,
    pub total_certificates: u64,
    pub vertices_per_round: BTreeMap<Round, u64>,
    pub state_hash: [u8; 32],
    pub last_finalized_round: Round,
    pub validator_weights: HashMap<ValidatorId, u64>,
}

/// DAG state synchronization manager
pub struct DagSyncManager {
    local_validator_id: ValidatorId,
    peer_registry: Arc<PeerRegistry>,
    channel_manager: Arc<PersistentChannelManager>,

    // Local DAG state tracking
    local_dag_summary: RwLock<Option<DagStateSummary>>,
    pending_requests: RwLock<HashMap<[u8; 16], DagSyncRequest>>,
    sync_metrics: RwLock<SyncMetrics>,

    // Sync configuration
    max_sync_batch_size: usize,
    sync_timeout: Duration,
    heartbeat_interval: Duration,
}

impl DagSyncManager {
    pub fn new(
        local_validator_id: ValidatorId,
        peer_registry: Arc<PeerRegistry>,
        channel_manager: Arc<PersistentChannelManager>,
    ) -> Self {
        Self {
            local_validator_id,
            peer_registry,
            channel_manager,
            local_dag_summary: RwLock::new(None),
            pending_requests: RwLock::new(HashMap::new()),
            sync_metrics: RwLock::new(SyncMetrics::new()),
            max_sync_batch_size: 1000,
            sync_timeout: Duration::from_secs(30),
            heartbeat_interval: Duration::from_secs(10),
        }
    }

    /// Update local DAG state summary
    pub async fn update_local_dag_summary(&self, summary: DagStateSummary) -> Result<()> {
        debug!(
            "📊 Updating local DAG summary: round {}, {} vertices",
            summary.current_round, summary.total_vertices
        );

        {
            let mut local_summary = self.local_dag_summary.write().await;
            *local_summary = Some(summary);
        }

        // Trigger sync if we're behind
        self.check_and_trigger_sync().await?;

        Ok(())
    }

    /// Request DAG synchronization with peers
    pub async fn request_dag_sync(&self, sync_type: SyncType) -> Result<Vec<DagSyncResponse>> {
        let request_id = self.generate_request_id();

        info!("🔄 Requesting DAG sync: {:?}", sync_type);

        let sync_request = DagSyncRequest {
            request_id,
            requester: self.local_validator_id,
            sync_type,
            from_round: self.get_current_round().await,
            to_round: None,
            missing_vertices: vec![],
            timestamp: chrono::Utc::now().timestamp() as u64,
        };

        // Store pending request
        {
            let mut pending = self.pending_requests.write().await;
            pending.insert(request_id, sync_request.clone());
        }

        // Send sync request to all connected peers
        let connected_peers = self.peer_registry.get_connected_peers().await;
        let mut responses = Vec::new();

        for peer_id in connected_peers {
            match self.send_sync_request_to_peer(peer_id, &sync_request).await {
                Ok(response) => {
                    if let Some(response) = response {
                        responses.push(response);
                    }
                }
                Err(e) => {
                    warn!(
                        "Failed to get sync response from peer {}: {}",
                        hex::encode(peer_id),
                        e
                    );
                }
            }
        }

        // Remove from pending requests
        {
            let mut pending = self.pending_requests.write().await;
            pending.remove(&request_id);
        }

        // Update metrics
        {
            let mut metrics = self.sync_metrics.write().await;
            metrics.total_sync_requests += 1;
            metrics.successful_sync_responses += responses.len() as u64;
        }

        info!(
            "✅ DAG sync completed: received {} responses",
            responses.len()
        );
        Ok(responses)
    }

    /// Send sync request to a specific peer
    async fn send_sync_request_to_peer(
        &self,
        peer_id: ValidatorId,
        sync_request: &DagSyncRequest,
    ) -> Result<Option<DagSyncResponse>> {
        debug!("📤 Sending sync request to peer {}", hex::encode(peer_id));

        let peer_info = self
            .peer_registry
            .get_peer(&peer_id)
            .await
            .ok_or_else(|| anyhow::anyhow!("Peer not found in registry"))?;

        // Serialize request
        let request_data =
            serde_json::to_vec(sync_request).context("Failed to serialize sync request")?;

        let message = ChannelMessage {
            data: request_data,
            message_type: MessageType::StateSync,
            priority: MessagePriority::High,
            created_at: Instant::now(),
        };

        // Send request through persistent channel
        self.channel_manager
            .send_message(peer_id, message)
            .await
            .context("Failed to send sync request")?;

        // Wait for response (simplified - in production would use proper request/response correlation)
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Simulate response (in production, this would come through message handling)
        let mock_response = self
            .create_mock_sync_response(sync_request, peer_id)
            .await?;

        Ok(Some(mock_response))
    }

    /// Create a mock sync response for testing
    async fn create_mock_sync_response(
        &self,
        request: &DagSyncRequest,
        responder: ValidatorId,
    ) -> Result<DagSyncResponse> {
        // This is a simplified mock response
        // In production, this would be generated by the receiving peer
        let response = DagSyncResponse {
            request_id: request.request_id,
            responder,
            vertices: vec![],     // Would contain actual vertices
            certificates: vec![], // Would contain actual certificates
            dag_summary: DagStateSummary {
                current_round: request.from_round + 1,
                total_vertices: 1000,
                total_certificates: 500,
                vertices_per_round: BTreeMap::new(),
                state_hash: [0u8; 32], // Would be actual state hash
                last_finalized_round: request.from_round,
                validator_weights: HashMap::new(),
            },
            timestamp: chrono::Utc::now().timestamp() as u64,
        };

        Ok(response)
    }

    /// Handle incoming DAG sync request
    pub async fn handle_sync_request(&self, request: DagSyncRequest) -> Result<DagSyncResponse> {
        debug!(
            "📥 Handling sync request {} from {}",
            hex::encode(request.request_id),
            hex::encode(request.requester)
        );

        let response_vertices = self.get_vertices_for_sync(&request).await?;
        let response_certificates = self.get_certificates_for_sync(&request).await?;
        let local_summary = self
            .get_local_dag_summary()
            .await
            .ok_or_else(|| anyhow::anyhow!("Local DAG summary not available"))?;

        let response = DagSyncResponse {
            request_id: request.request_id,
            responder: self.local_validator_id,
            vertices: response_vertices,
            certificates: response_certificates,
            dag_summary: local_summary,
            timestamp: chrono::Utc::now().timestamp() as u64,
        };

        // Update metrics
        {
            let mut metrics = self.sync_metrics.write().await;
            metrics.sync_requests_handled += 1;
        }

        debug!(
            "✅ Generated sync response with {} vertices",
            response.vertices.len()
        );
        Ok(response)
    }

    /// Get vertices to include in sync response
    async fn get_vertices_for_sync(&self, request: &DagSyncRequest) -> Result<Vec<Vertex>> {
        // This would query the actual DAG store
        // For now, return empty vector as placeholder
        match &request.sync_type {
            SyncType::RecentRounds { last_n_rounds } => {
                debug!("Getting vertices for last {} rounds", last_n_rounds);
                Ok(vec![])
            }
            SyncType::RoundRange { from, to } => {
                debug!("Getting vertices for rounds {} to {}", from, to);
                Ok(vec![])
            }
            SyncType::MissingVertices { vertex_ids } => {
                debug!("Getting {} specific vertices", vertex_ids.len());
                Ok(vec![])
            }
            SyncType::FullSync => {
                debug!("Getting all vertices for full sync");
                Ok(vec![])
            }
            SyncType::HeartbeatSync => {
                debug!("Heartbeat sync - no vertices needed");
                Ok(vec![])
            }
        }
    }

    /// Get certificates to include in sync response
    async fn get_certificates_for_sync(
        &self,
        request: &DagSyncRequest,
    ) -> Result<Vec<Certificate>> {
        // This would query the actual certificate store
        // For now, return empty vector as placeholder
        debug!("Getting certificates for sync request");
        Ok(vec![])
    }

    /// Check if we need to sync and trigger if necessary
    async fn check_and_trigger_sync(&self) -> Result<()> {
        let local_summary = self.get_local_dag_summary().await;
        let connected_peers = self.peer_registry.get_connected_peers().await;

        if connected_peers.is_empty() {
            return Ok(());
        }

        // Check if we're significantly behind
        // This would involve comparing our state with peer states
        // For now, just log the check
        debug!("🔍 Checking if DAG sync is needed");

        Ok(())
    }

    /// Start periodic heartbeat sync
    pub async fn start_heartbeat_sync(self: Arc<Self>) -> Result<()> {
        info!("💓 Starting DAG heartbeat sync");

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(self.heartbeat_interval);

            loop {
                interval.tick().await;

                if let Err(e) = self.perform_heartbeat_sync().await {
                    warn!("Heartbeat sync failed: {}", e);
                }
            }
        });

        Ok(())
    }

    /// Perform heartbeat sync with peers
    async fn perform_heartbeat_sync(&self) -> Result<()> {
        debug!("💓 Performing heartbeat sync");

        let sync_type = SyncType::HeartbeatSync;
        let _responses = self.request_dag_sync(sync_type).await?;

        // Process heartbeat responses to detect inconsistencies
        // This would trigger full sync if major differences detected

        Ok(())
    }

    /// Detect DAG inconsistencies from sync responses
    pub async fn detect_inconsistencies(
        &self,
        responses: &[DagSyncResponse],
    ) -> Vec<DagInconsistency> {
        let mut inconsistencies = Vec::new();
        let local_summary = self.get_local_dag_summary().await;

        if let Some(local) = local_summary {
            for response in responses {
                let peer_summary = &response.dag_summary;

                // Check round differences
                if peer_summary.current_round > local.current_round + 5 {
                    inconsistencies.push(DagInconsistency::RoundGap {
                        peer: response.responder,
                        local_round: local.current_round,
                        peer_round: peer_summary.current_round,
                    });
                }

                // Check state hash differences
                if peer_summary.state_hash != local.state_hash {
                    inconsistencies.push(DagInconsistency::StateHashMismatch {
                        peer: response.responder,
                        local_hash: local.state_hash,
                        peer_hash: peer_summary.state_hash,
                    });
                }

                // Check vertex count differences
                let vertex_diff = if peer_summary.total_vertices > local.total_vertices {
                    peer_summary.total_vertices - local.total_vertices
                } else {
                    local.total_vertices - peer_summary.total_vertices
                };

                if vertex_diff > 100 {
                    // Significant difference threshold
                    inconsistencies.push(DagInconsistency::VertexCountMismatch {
                        peer: response.responder,
                        local_count: local.total_vertices,
                        peer_count: peer_summary.total_vertices,
                    });
                }
            }
        }

        if !inconsistencies.is_empty() {
            warn!("⚠️ Detected {} DAG inconsistencies", inconsistencies.len());
        }

        inconsistencies
    }

    /// Get current round from local DAG summary
    async fn get_current_round(&self) -> Round {
        if let Some(summary) = self.get_local_dag_summary().await {
            summary.current_round
        } else {
            0
        }
    }

    /// Get local DAG summary
    async fn get_local_dag_summary(&self) -> Option<DagStateSummary> {
        let summary = self.local_dag_summary.read().await;
        summary.clone()
    }

    /// Generate unique request ID
    fn generate_request_id(&self) -> [u8; 16] {
        use rand::RngCore;
        let mut rng = rand::thread_rng();
        let mut id = [0u8; 16];
        rng.fill_bytes(&mut id);
        id
    }

    /// Get sync metrics
    pub async fn get_sync_metrics(&self) -> SyncMetrics {
        self.sync_metrics.read().await.clone()
    }
}

/// Types of DAG inconsistencies
#[derive(Debug, Clone)]
pub enum DagInconsistency {
    RoundGap {
        peer: ValidatorId,
        local_round: Round,
        peer_round: Round,
    },
    StateHashMismatch {
        peer: ValidatorId,
        local_hash: [u8; 32],
        peer_hash: [u8; 32],
    },
    VertexCountMismatch {
        peer: ValidatorId,
        local_count: u64,
        peer_count: u64,
    },
}

/// Synchronization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncMetrics {
    pub total_sync_requests: u64,
    pub successful_sync_responses: u64,
    pub sync_requests_handled: u64,
    pub last_sync_timestamp: Option<u64>,
    pub average_sync_duration_ms: u64,
    pub inconsistencies_detected: u64,
}

impl SyncMetrics {
    pub fn new() -> Self {
        Self {
            total_sync_requests: 0,
            successful_sync_responses: 0,
            sync_requests_handled: 0,
            last_sync_timestamp: None,
            average_sync_duration_ms: 0,
            inconsistencies_detected: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::peer_registry::PeerRegistry;
    use crate::persistent_channels::PersistentChannelManager;
    use q_tor_client::QTorClient;

    #[tokio::test]
    async fn test_dag_sync_manager_creation() {
        let validator_id = [1u8; 32];
        let peer_registry = Arc::new(PeerRegistry::new(validator_id));
        let tor_client = Arc::new(QTorClient::mock());
        let channel_manager = Arc::new(PersistentChannelManager::new(tor_client, validator_id, 24));

        let sync_manager = DagSyncManager::new(validator_id, peer_registry, channel_manager);

        let metrics = sync_manager.get_sync_metrics().await;
        assert_eq!(metrics.total_sync_requests, 0);
    }

    #[tokio::test]
    async fn test_dag_summary_update() {
        let validator_id = [1u8; 32];
        let peer_registry = Arc::new(PeerRegistry::new(validator_id));
        let tor_client = Arc::new(QTorClient::mock());
        let channel_manager = Arc::new(PersistentChannelManager::new(tor_client, validator_id, 24));

        let sync_manager = DagSyncManager::new(validator_id, peer_registry, channel_manager);

        let summary = DagStateSummary {
            current_round: 10,
            total_vertices: 1000,
            total_certificates: 500,
            vertices_per_round: BTreeMap::new(),
            state_hash: [1u8; 32],
            last_finalized_round: 9,
            validator_weights: HashMap::new(),
        };

        sync_manager
            .update_local_dag_summary(summary)
            .await
            .unwrap();

        let local_summary = sync_manager.get_local_dag_summary().await.unwrap();
        assert_eq!(local_summary.current_round, 10);
        assert_eq!(local_summary.total_vertices, 1000);
    }

    #[tokio::test]
    async fn test_inconsistency_detection() {
        let validator_id = [1u8; 32];
        let peer_registry = Arc::new(PeerRegistry::new(validator_id));
        let tor_client = Arc::new(QTorClient::mock());
        let channel_manager = Arc::new(PersistentChannelManager::new(tor_client, validator_id, 24));

        let sync_manager = DagSyncManager::new(validator_id, peer_registry, channel_manager);

        // Set local summary
        let local_summary = DagStateSummary {
            current_round: 10,
            total_vertices: 1000,
            total_certificates: 500,
            vertices_per_round: BTreeMap::new(),
            state_hash: [1u8; 32],
            last_finalized_round: 9,
            validator_weights: HashMap::new(),
        };

        sync_manager
            .update_local_dag_summary(local_summary)
            .await
            .unwrap();

        // Create response with different state
        let peer_id = [2u8; 32];
        let response = DagSyncResponse {
            request_id: [0u8; 16],
            responder: peer_id,
            vertices: vec![],
            certificates: vec![],
            dag_summary: DagStateSummary {
                current_round: 20, // Significant gap
                total_vertices: 1000,
                total_certificates: 500,
                vertices_per_round: BTreeMap::new(),
                state_hash: [2u8; 32], // Different hash
                last_finalized_round: 19,
                validator_weights: HashMap::new(),
            },
            timestamp: 0,
        };

        let inconsistencies = sync_manager.detect_inconsistencies(&[response]).await;
        assert_eq!(inconsistencies.len(), 2); // Round gap + hash mismatch
    }
}
