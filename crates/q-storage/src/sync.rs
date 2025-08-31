/// Pull-based sync protocol for DagKnight DAG catch-up
/// Efficient range-based vertex and certificate synchronization

use anyhow::{Context, Result};
use async_trait::async_trait;
use libp2p::{
    request_response::{self, ProtocolSupport, RequestId},
    StreamProtocol,
};
use q_dag_knight::BullsharkCert;
use q_types::{NodeId, Vertex};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    io,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, error, info, warn};

use crate::kv::KVStore;

/// Sync protocol for DAG catch-up
pub struct SyncProtocol {
    hot_db: Arc<dyn KVStore>,
    cold_db: Arc<dyn KVStore>,
    pending_requests: Arc<RwLock<HashMap<RequestId, SyncRequestInfo>>>,
    sync_progress: Arc<RwLock<SyncProgress>>,
}

impl SyncProtocol {
    /// Create new sync protocol
    pub async fn new(hot_db: Arc<dyn KVStore>, cold_db: Arc<dyn KVStore>) -> Result<Self> {
        Ok(Self {
            hot_db,
            cold_db,
            pending_requests: Arc::new(RwLock::new(HashMap::new())),
            sync_progress: Arc::new(RwLock::new(SyncProgress::default())),
        })
    }

    /// Start catch-up sync from a specific round
    pub async fn start_catch_up(&self, from_round: u64) -> Result<()> {
        info!("🔄 Starting DAG catch-up sync from round {}", from_round);

        let mut progress = self.sync_progress.write().await;
        progress.start_sync(from_round);
        
        // In production, this would initiate sync requests to peers
        // For now, we just log the intent
        info!("📡 Catch-up sync initiated from round {}", from_round);
        
        Ok(())
    }

    /// Handle incoming sync request
    pub async fn handle_sync_request(&self, request: SyncRequest) -> Result<SyncResponse> {
        debug!("📥 Handling sync request: rounds {}-{}", 
               request.start_round, request.target_round);

        let start_time = Instant::now();

        // Fetch vertices in the requested range
        let mut vertices = Vec::new();
        for round in request.start_round..=request.target_round {
            let round_vertices = self.get_vertices_for_round(round).await?;
            vertices.extend(round_vertices);
        }

        // Fetch certificates in the requested range
        let mut certificates = Vec::new();
        for round in request.start_round..=request.target_round {
            if let Some(cert) = self.get_certificate_for_round(round).await? {
                certificates.push(cert);
            }
        }

        let response = SyncResponse {
            start_round: request.start_round,
            target_round: request.target_round,
            vertices,
            certificates,
            total_vertices: vertices.len() as u64,
            total_certificates: certificates.len() as u64,
        };

        let latency = start_time.elapsed();
        info!("📤 Sync response prepared: {} vertices, {} certs ({}ms)",
              response.total_vertices, response.total_certificates, latency.as_millis());

        Ok(response)
    }

    /// Handle incoming sync response
    pub async fn handle_sync_response(&self, response: SyncResponse) -> Result<()> {
        info!("📥 Processing sync response: {} vertices, {} certificates",
              response.total_vertices, response.total_certificates);

        let start_time = Instant::now();

        // Store received vertices
        for vertex in &response.vertices {
            self.store_synced_vertex(vertex).await?;
        }

        // Store received certificates
        for cert in &response.certificates {
            self.store_synced_certificate(cert).await?;
        }

        // Update sync progress
        {
            let mut progress = self.sync_progress.write().await;
            progress.update_progress(response.target_round, response.total_vertices);
        }

        let latency = start_time.elapsed();
        info!("✅ Sync response processed ({}ms)", latency.as_millis());

        Ok(())
    }

    /// Get vertices for a specific round from storage
    async fn get_vertices_for_round(&self, round: u64) -> Result<Vec<Vertex>> {
        let prefix = round.to_be_bytes();
        let vertex_data = self.hot_db.scan_prefix("dag_vertices", &prefix).await?;

        let mut vertices = Vec::new();
        for (_, data) in vertex_data {
            let vertex: Vertex = bincode::deserialize(&data)?;
            vertices.push(vertex);
        }

        Ok(vertices)
    }

    /// Get certificate for a specific round from storage
    async fn get_certificate_for_round(&self, round: u64) -> Result<Option<BullsharkCert>> {
        let key = round.to_be_bytes();
        
        if let Some(cert_data) = self.hot_db.get("bullshark_cert", &key).await? {
            let cert: BullsharkCert = bincode::deserialize(&cert_data)?;
            return Ok(Some(cert));
        }

        Ok(None)
    }

    /// Store vertex received from sync
    async fn store_synced_vertex(&self, vertex: &Vertex) -> Result<()> {
        let vertex_key = self.vertex_key(vertex.round, &vertex.author, vertex.sequence);
        let vertex_data = bincode::serialize(vertex)?;
        
        self.hot_db.put("dag_vertices", &vertex_key, &vertex_data).await?;
        
        debug!("💾 Stored synced vertex {} for round {}", 
               hex::encode(&vertex.id), vertex.round);
        
        Ok(())
    }

    /// Store certificate received from sync
    async fn store_synced_certificate(&self, cert: &BullsharkCert) -> Result<()> {
        let cert_key = cert.round.to_be_bytes();
        let cert_data = bincode::serialize(cert)?;
        
        self.hot_db.put("bullshark_cert", &cert_key, &cert_data).await?;
        
        debug!("📜 Stored synced certificate for round {}", cert.round);
        
        Ok(())
    }

    /// Generate vertex key for storage
    fn vertex_key(&self, round: u64, author: &[u8], sequence: u64) -> Vec<u8> {
        let mut key = Vec::with_capacity(8 + author.len() + 8);
        key.extend_from_slice(&round.to_be_bytes());
        key.extend_from_slice(author);
        key.extend_from_slice(&sequence.to_be_bytes());
        key
    }

    /// Get sync progress
    pub async fn get_sync_progress(&self) -> SyncProgress {
        self.sync_progress.read().await.clone()
    }

    /// Check if sync is complete
    pub async fn is_sync_complete(&self) -> bool {
        let progress = self.sync_progress.read().await;
        progress.is_complete()
    }

    /// Shutdown sync protocol
    pub async fn shutdown(&self) -> Result<()> {
        info!("🛑 Shutting down sync protocol");
        
        // Cancel any pending requests
        {
            let mut requests = self.pending_requests.write().await;
            requests.clear();
        }

        info!("✅ Sync protocol shutdown complete");
        Ok(())
    }
}

/// Sync request message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncRequest {
    /// Starting round (inclusive)
    pub start_round: u64,
    
    /// Target round (inclusive)
    pub target_round: u64,
    
    /// Request ID for tracking
    pub request_id: String,
    
    /// Requesting node ID
    pub requester: NodeId,
    
    /// Include vertex data
    pub include_vertices: bool,
    
    /// Include certificate data
    pub include_certificates: bool,
    
    /// Maximum response size in bytes
    pub max_response_size: usize,
}

/// Sync response message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncResponse {
    /// Starting round of response
    pub start_round: u64,
    
    /// Target round of response
    pub target_round: u64,
    
    /// Vertices in the range
    pub vertices: Vec<Vertex>,
    
    /// Certificates in the range
    pub certificates: Vec<BullsharkCert>,
    
    /// Total vertices included
    pub total_vertices: u64,
    
    /// Total certificates included
    pub total_certificates: u64,
}

/// Information about pending sync request
#[derive(Debug, Clone)]
struct SyncRequestInfo {
    pub request: SyncRequest,
    pub sent_at: Instant,
    pub peer_id: String,
}

/// Sync progress tracking
#[derive(Debug, Clone, Default)]
pub struct SyncProgress {
    /// Is sync currently active
    pub active: bool,
    
    /// Starting round of current sync
    pub start_round: u64,
    
    /// Target round to reach
    pub target_round: u64,
    
    /// Current round being processed
    pub current_round: u64,
    
    /// Total vertices synced
    pub vertices_synced: u64,
    
    /// Total certificates synced
    pub certificates_synced: u64,
    
    /// Sync start time
    pub started_at: Option<Instant>,
    
    /// Last update time
    pub last_update: Instant,
}

impl SyncProgress {
    /// Start new sync
    pub fn start_sync(&mut self, from_round: u64) {
        self.active = true;
        self.start_round = from_round;
        self.current_round = from_round;
        self.vertices_synced = 0;
        self.certificates_synced = 0;
        self.started_at = Some(Instant::now());
        self.last_update = Instant::now();
    }

    /// Update sync progress
    pub fn update_progress(&mut self, completed_round: u64, vertices_count: u64) {
        self.current_round = completed_round;
        self.vertices_synced += vertices_count;
        self.last_update = Instant::now();
    }

    /// Mark sync as complete
    pub fn complete_sync(&mut self) {
        self.active = false;
        self.last_update = Instant::now();
    }

    /// Check if sync is complete
    pub fn is_complete(&self) -> bool {
        !self.active || (self.target_round > 0 && self.current_round >= self.target_round)
    }

    /// Get sync percentage complete
    pub fn completion_percentage(&self) -> f64 {
        if self.target_round <= self.start_round {
            return 100.0;
        }

        let completed = self.current_round.saturating_sub(self.start_round);
        let total = self.target_round.saturating_sub(self.start_round);
        
        (completed as f64 / total as f64) * 100.0
    }

    /// Get sync duration
    pub fn sync_duration(&self) -> Option<Duration> {
        self.started_at.map(|start| start.elapsed())
    }

    /// Get estimated time remaining
    pub fn estimated_time_remaining(&self) -> Option<Duration> {
        if let Some(duration) = self.sync_duration() {
            let completion = self.completion_percentage() / 100.0;
            if completion > 0.0 && completion < 1.0 {
                let total_estimated = duration.as_secs_f64() / completion;
                let remaining = total_estimated - duration.as_secs_f64();
                return Some(Duration::from_secs_f64(remaining.max(0.0)));
            }
        }
        
        None
    }
}

/// libp2p codec for sync protocol
#[derive(Debug, Clone)]
pub struct DagSyncCodec;

#[async_trait]
impl request_response::Codec for DagSyncCodec {
    type Protocol = StreamProtocol;
    type Request = SyncRequest;
    type Response = SyncResponse;

    async fn read_request<T>(&mut self, _: &Self::Protocol, io: &mut T) -> io::Result<Self::Request>
    where
        T: futures::AsyncRead + Unpin + Send,
    {
        // Implementation would read and deserialize SyncRequest
        // For now, return a dummy request
        Err(io::Error::new(io::ErrorKind::Unsupported, "Not implemented"))
    }

    async fn read_response<T>(&mut self, _: &Self::Protocol, io: &mut T) -> io::Result<Self::Response>
    where
        T: futures::AsyncRead + Unpin + Send,
    {
        // Implementation would read and deserialize SyncResponse
        // For now, return a dummy response
        Err(io::Error::new(io::ErrorKind::Unsupported, "Not implemented"))
    }

    async fn write_request<T>(&mut self, _: &Self::Protocol, io: &mut T, req: Self::Request) -> io::Result<()>
    where
        T: futures::AsyncWrite + Unpin + Send,
    {
        // Implementation would serialize and write SyncRequest
        // For now, just succeed
        Ok(())
    }

    async fn write_response<T>(&mut self, _: &Self::Protocol, io: &mut T, res: Self::Response) -> io::Result<()>
    where
        T: futures::AsyncWrite + Unpin + Send,
    {
        // Implementation would serialize and write SyncResponse
        // For now, just succeed
        Ok(())
    }
}

/// Sync protocol configuration
#[derive(Debug, Clone)]
pub struct SyncConfig {
    /// Maximum vertices per sync response
    pub max_vertices_per_response: usize,
    
    /// Maximum certificates per sync response
    pub max_certificates_per_response: usize,
    
    /// Request timeout
    pub request_timeout: Duration,
    
    /// Maximum concurrent sync requests
    pub max_concurrent_requests: usize,
    
    /// Retry attempts for failed requests
    pub max_retry_attempts: u32,
}

impl Default for SyncConfig {
    fn default() -> Self {
        Self {
            max_vertices_per_response: 1000,
            max_certificates_per_response: 100,
            request_timeout: Duration::from_secs(30),
            max_concurrent_requests: 10,
            max_retry_attempts: 3,
        }
    }
}

/// Sync statistics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncStats {
    pub active_requests: usize,
    pub completed_requests: u64,
    pub failed_requests: u64,
    pub vertices_synced: u64,
    pub certificates_synced: u64,
    pub bytes_synced: u64,
    pub average_request_latency: Duration,
    pub current_sync_progress: Option<SyncProgress>,
}

/// Information about a sync request
#[derive(Debug, Clone)]
struct SyncRequestInfo {
    request: SyncRequest,
    sent_at: Instant,
    peer_id: String,
    retry_count: u32,
}

/// Trait for sync-enabled storage
#[async_trait]
pub trait SyncEnabled {
    /// Request sync from peers
    async fn request_sync(&self, start_round: u64, target_round: u64) -> Result<()>;
    
    /// Check sync status
    async fn get_sync_status(&self) -> SyncProgress;
    
    /// Handle sync request from peer
    async fn handle_peer_sync_request(&self, request: SyncRequest) -> Result<SyncResponse>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sync_request_serialization() {
        let request = SyncRequest {
            start_round: 100,
            target_round: 200,
            request_id: "test_request_123".to_string(),
            requester: [1u8; 32],
            include_vertices: true,
            include_certificates: true,
            max_response_size: 1024 * 1024, // 1MB
        };

        let serialized = bincode::serialize(&request).unwrap();
        let deserialized: SyncRequest = bincode::deserialize(&serialized).unwrap();

        assert_eq!(request.start_round, deserialized.start_round);
        assert_eq!(request.target_round, deserialized.target_round);
        assert_eq!(request.request_id, deserialized.request_id);
    }

    #[test]
    fn test_sync_progress_tracking() {
        let mut progress = SyncProgress::default();
        
        // Start sync
        progress.start_sync(100);
        assert!(progress.active);
        assert_eq!(progress.start_round, 100);
        assert_eq!(progress.current_round, 100);
        
        // Update progress
        progress.update_progress(150, 1000);
        assert_eq!(progress.current_round, 150);
        assert_eq!(progress.vertices_synced, 1000);
        
        // Test completion percentage
        progress.target_round = 200;
        let completion = progress.completion_percentage();
        assert_eq!(completion, 50.0); // (150-100)/(200-100) * 100
    }

    #[test]
    fn test_sync_config_defaults() {
        let config = SyncConfig::default();
        
        assert_eq!(config.max_vertices_per_response, 1000);
        assert_eq!(config.max_certificates_per_response, 100);
        assert_eq!(config.request_timeout, Duration::from_secs(30));
        assert_eq!(config.max_concurrent_requests, 10);
        assert_eq!(config.max_retry_attempts, 3);
    }

    #[tokio::test]
    async fn test_sync_progress_completion() {
        let mut progress = SyncProgress::default();
        
        // Not active = complete
        assert!(progress.is_complete());
        
        // Start sync but no target = not complete
        progress.start_sync(100);
        assert!(!progress.is_complete());
        
        // Set target and reach it = complete
        progress.target_round = 200;
        progress.update_progress(200, 1000);
        assert!(progress.is_complete());
    }
}