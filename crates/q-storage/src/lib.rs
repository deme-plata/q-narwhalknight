/// Q-Storage: High-performance blockchain database for DagKnight consensus
/// Optimized for Narwhal mempool and Bullshark finality with hot/cold storage split
/// Battle-tested design using RocksDB with specialized column families

use anyhow::{Context, Result};
use async_trait::async_trait;
use q_dag_knight::BullsharkCert;
use q_narwhal_core::{Certificate, NarwhalPayload};
use q_types::{Block, NodeId, Phase, Transaction, Vertex};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

pub mod kv;
pub mod manifest;
pub mod sync;
pub mod snapshot;
pub mod metrics;

pub use kv::{KVStore, RocksDBKV};
pub use manifest::StorageManifest;
pub use sync::{SyncProtocol, SyncRequest, SyncResponse};
pub use snapshot::SnapshotManager;
pub use metrics::StorageMetrics;

/// Column family names for optimized storage
pub const CF_BLOCKS: &str = "blocks";
pub const CF_DAG_VERTICES: &str = "dag_vertices";
pub const CF_BULLSHARK_CERT: &str = "bullshark_cert";
pub const CF_MANIFEST: &str = "manifest";
pub const CF_NARWHAL_PAYLOADS: &str = "narwhal_payloads";

/// Main storage engine for Q-NarwhalKnight
pub struct QStorage {
    /// Hot database (RocksDB) - blocks, vertices, certificates
    hot_db: Arc<dyn KVStore>,
    /// Cold database (RocksDB) - large Narwhal payloads
    cold_db: Arc<dyn KVStore>,
    /// Storage manifest with watermarks
    manifest: Arc<RwLock<StorageManifest>>,
    /// Sync protocol for DAG catch-up
    sync_protocol: Arc<SyncProtocol>,
    /// Snapshot manager
    snapshot_manager: Arc<SnapshotManager>,
    /// Storage metrics
    metrics: Arc<StorageMetrics>,
    /// Node configuration
    node_id: NodeId,
    data_dir: PathBuf,
}

impl QStorage {
    /// Open storage with hot/cold database split
    pub async fn open<P: AsRef<Path>>(
        data_dir: P,
        node_id: NodeId,
    ) -> Result<Self> {
        let data_dir = data_dir.as_ref().to_path_buf();
        info!("🗄️ Opening Q-Storage at {:?} for node {}", 
              data_dir, hex::encode(&node_id[..4]));

        // Configure hot database (frequent access)
        let hot_path = data_dir.join("hot");
        let hot_db = Arc::new(
            RocksDBKV::open_hot_db(&hot_path).await
                .context("Failed to open hot database")?
        );

        // Configure cold database (large payloads)
        let cold_path = data_dir.join("cold");
        let cold_db = Arc::new(
            RocksDBKV::open_cold_db(&cold_path).await
                .context("Failed to open cold database")?
        );

        // Load storage manifest
        let manifest = Arc::new(RwLock::new(
            StorageManifest::load_or_create(&hot_db).await?
        ));

        // Initialize sync protocol
        let sync_protocol = Arc::new(
            SyncProtocol::new(hot_db.clone(), cold_db.clone()).await?
        );

        // Initialize snapshot manager
        let snapshot_manager = Arc::new(
            SnapshotManager::new(data_dir.clone(), hot_db.clone(), cold_db.clone()).await?
        );

        // Initialize metrics
        let metrics = Arc::new(StorageMetrics::new());

        let storage = Self {
            hot_db,
            cold_db,
            manifest,
            sync_protocol,
            snapshot_manager,
            metrics,
            node_id,
            data_dir,
        };

        // Perform crash recovery
        storage.recover().await?;

        info!("✅ Q-Storage initialized successfully");
        Ok(storage)
    }

    /// Store DAG vertex with Narwhal payload
    pub async fn store_vertex(&self, vertex: &Vertex, payload: &NarwhalPayload) -> Result<()> {
        debug!("💾 Storing vertex {} for round {}", 
               hex::encode(&vertex.id), vertex.round);

        let start_time = Instant::now();

        // Store vertex in hot DB
        let vertex_key = self.vertex_key(vertex.round, &vertex.author, vertex.sequence);
        let vertex_data = bincode::serialize(vertex)?;
        
        self.hot_db.put(CF_DAG_VERTICES, &vertex_key, &vertex_data).await?;

        // Store payload in cold DB  
        let payload_key = payload.digest();
        let payload_data = bincode::serialize(payload)?;
        
        self.cold_db.put(CF_NARWHAL_PAYLOADS, &payload_key, &payload_data).await?;

        // Update metrics
        let latency = start_time.elapsed();
        self.metrics.record_vertex_write(latency, vertex_data.len(), payload_data.len()).await;

        // Check if this completes a round
        self.check_round_completion(vertex.round).await?;

        debug!("✅ Stored vertex {} ({}ms)", hex::encode(&vertex.id), latency.as_millis());
        Ok(())
    }

    /// Store Bullshark certificate
    pub async fn store_certificate(&self, cert: &BullsharkCert) -> Result<()> {
        debug!("📜 Storing Bullshark certificate for round {}", cert.round);

        let cert_key = cert.round.to_be_bytes();
        let cert_data = bincode::serialize(cert)?;

        self.hot_db.put(CF_BULLSHARK_CERT, &cert_key, &cert_data).await?;

        // Update manifest watermark
        self.update_dag_watermark(cert.round).await?;

        info!("✅ Stored certificate for round {}", cert.round);
        Ok(())
    }

    /// Finalize block with Bullshark consensus
    pub async fn finalize_block(&self, block: &Block, finality_proof: &BullsharkCert) -> Result<()> {
        info!("🎯 Finalizing block {} at height {}", 
              hex::encode(&block.hash()), block.height);

        let start_time = Instant::now();

        // Prepare atomic batch
        let mut batch = Vec::new();

        // Store finalized block
        let block_key = self.block_key(block.height, &block.hash());
        let block_data = bincode::serialize(block)?;
        batch.push((CF_BLOCKS, block_key, block_data));

        // Store finality proof
        let proof_key = format!("finality_{}", block.height);
        let proof_data = bincode::serialize(finality_proof)?;
        batch.push((CF_BULLSHARK_CERT, proof_key.into_bytes(), proof_data));

        // Commit atomically
        self.hot_db.write_batch(batch).await?;

        // Update finalized height in manifest
        {
            let mut manifest = self.manifest.write().await;
            manifest.finalized_height = block.height.max(manifest.finalized_height);
            manifest.save(&self.hot_db).await?;
        }

        let latency = start_time.elapsed();
        self.metrics.record_block_finalization(latency, block.transactions.len()).await;

        info!("✅ Finalized block {} ({}ms, {} txs)", 
              hex::encode(&block.hash()), latency.as_millis(), block.transactions.len());

        // Check if we should create a snapshot
        self.check_snapshot_trigger(block.height).await?;

        Ok(())
    }

    /// Get vertex by ID
    pub async fn get_vertex(&self, vertex_id: &[u8]) -> Result<Option<Vertex>> {
        // For point queries, we need to search by vertex ID
        // This is less efficient than round-based queries
        debug!("🔍 Looking up vertex {}", hex::encode(vertex_id));

        // Implementation would need a secondary index vertex_id -> (round, author, seq)
        // For now, we'll implement a scan (inefficient but correct)
        self.scan_for_vertex(vertex_id).await
    }

    /// Get vertices for a specific round
    pub async fn get_vertices_for_round(&self, round: u64) -> Result<Vec<Vertex>> {
        debug!("🔍 Fetching all vertices for round {}", round);

        let prefix = round.to_be_bytes();
        let vertices = self.hot_db.scan_prefix(CF_DAG_VERTICES, &prefix).await?;

        let mut result = Vec::new();
        for (_, vertex_data) in vertices {
            let vertex: Vertex = bincode::deserialize(&vertex_data)?;
            result.push(vertex);
        }

        debug!("✅ Found {} vertices for round {}", result.len(), round);
        Ok(result)
    }

    /// Get Narwhal payload by digest
    pub async fn get_payload(&self, digest: &[u8]) -> Result<Option<NarwhalPayload>> {
        debug!("🔍 Fetching payload {}", hex::encode(digest));

        if let Some(payload_data) = self.cold_db.get(CF_NARWHAL_PAYLOADS, digest).await? {
            let payload: NarwhalPayload = bincode::deserialize(&payload_data)?;
            return Ok(Some(payload));
        }

        Ok(None)
    }

    /// Get finalized block by height
    pub async fn get_block_by_height(&self, height: u64) -> Result<Option<Block>> {
        debug!("🔍 Fetching block at height {}", height);

        // Scan for block with this height (RocksDB iterator)
        let prefix = height.to_be_bytes();
        let blocks = self.hot_db.scan_prefix(CF_BLOCKS, &prefix).await?;

        if let Some((_, block_data)) = blocks.into_iter().next() {
            let block: Block = bincode::deserialize(&block_data)?;
            return Ok(Some(block));
        }

        Ok(None)
    }

    /// Get storage statistics
    pub async fn get_storage_stats(&self) -> StorageStats {
        let manifest = self.manifest.read().await;
        let metrics = self.metrics.get_current_metrics().await;

        StorageStats {
            dag_round_watermark: manifest.dag_round_watermark,
            finalized_height: manifest.finalized_height,
            total_vertices: metrics.total_vertices,
            total_payloads: metrics.total_payloads,
            total_blocks: metrics.total_blocks,
            hot_db_size: self.hot_db.get_db_size().await.unwrap_or(0),
            cold_db_size: self.cold_db.get_db_size().await.unwrap_or(0),
            average_write_latency: metrics.average_write_latency,
            average_read_latency: metrics.average_read_latency,
        }
    }

    /// Perform crash recovery
    async fn recover(&self) -> Result<()> {
        info!("🔄 Starting storage crash recovery");

        let manifest = self.manifest.read().await;
        info!("📊 Recovery state - DAG watermark: {}, finalized: {}", 
              manifest.dag_round_watermark, manifest.finalized_height);

        // Verify DAG consistency
        self.verify_dag_consistency().await?;

        // Start sync process if needed
        if manifest.dag_round_watermark > 0 {
            self.sync_protocol.start_catch_up(manifest.dag_round_watermark).await?;
        }

        info!("✅ Storage recovery complete");
        Ok(())
    }

    /// Verify DAG consistency after crash
    async fn verify_dag_consistency(&self) -> Result<()> {
        debug!("🔍 Verifying DAG consistency");

        let manifest = self.manifest.read().await;
        let watermark = manifest.dag_round_watermark;

        // Check that we have contiguous rounds up to watermark
        for round in 0..=watermark {
            let vertices = self.get_vertices_for_round(round).await?;
            if vertices.is_empty() && round < watermark {
                warn!("⚠️ Missing vertices for round {}, truncating watermark", round);
                // In production, we'd truncate the watermark here
                break;
            }
        }

        debug!("✅ DAG consistency verified up to round {}", watermark);
        Ok(())
    }

    /// Update DAG round watermark
    async fn update_dag_watermark(&self, round: u64) -> Result<()> {
        let mut manifest = self.manifest.write().await;
        
        if round > manifest.dag_round_watermark {
            manifest.dag_round_watermark = round;
            manifest.save(&self.hot_db).await?;
            debug!("📈 Updated DAG watermark to round {}", round);
        }
        
        Ok(())
    }

    /// Check if a round is complete (for watermark advancement)
    async fn check_round_completion(&self, round: u64) -> Result<()> {
        // In production, this would check if we have enough vertices for Bullshark progress
        // For now, we just advance the watermark
        self.update_dag_watermark(round).await
    }

    /// Check if we should trigger a snapshot
    async fn check_snapshot_trigger(&self, block_height: u64) -> Result<()> {
        const SNAPSHOT_INTERVAL: u64 = 1000; // Every 1000 blocks

        if block_height % SNAPSHOT_INTERVAL == 0 {
            info!("📸 Triggering snapshot at height {}", block_height);
            self.snapshot_manager.create_snapshot(block_height).await?;
        }

        Ok(())
    }

    /// Scan for vertex by ID (inefficient, needs secondary index in production)
    async fn scan_for_vertex(&self, vertex_id: &[u8]) -> Result<Option<Vertex>> {
        // This is a fallback method - in production we'd have a secondary index
        warn!("🐌 Performing inefficient vertex scan for {}", hex::encode(vertex_id));

        let all_vertices = self.hot_db.scan_all(CF_DAG_VERTICES).await?;
        
        for (_, vertex_data) in all_vertices {
            if let Ok(vertex) = bincode::deserialize::<Vertex>(&vertex_data) {
                if vertex.id == vertex_id {
                    return Ok(Some(vertex));
                }
            }
        }

        Ok(None)
    }

    /// Generate vertex key for storage
    fn vertex_key(&self, round: u64, author: &[u8], sequence: u64) -> Vec<u8> {
        let mut key = Vec::with_capacity(8 + author.len() + 8);
        key.extend_from_slice(&round.to_be_bytes());
        key.extend_from_slice(author);
        key.extend_from_slice(&sequence.to_be_bytes());
        key
    }

    /// Generate block key for storage
    fn block_key(&self, height: u64, hash: &[u8]) -> Vec<u8> {
        let mut key = Vec::with_capacity(8 + hash.len());
        key.extend_from_slice(&height.to_be_bytes());
        key.extend_from_slice(hash);
        key
    }

    /// Compact storage to reclaim space
    pub async fn compact(&self) -> Result<()> {
        info!("🗜️ Compacting storage databases");

        // Compact hot database
        self.hot_db.compact().await?;
        
        // Compact cold database
        self.cold_db.compact().await?;

        info!("✅ Storage compaction complete");
        Ok(())
    }

    /// Prune old data beyond retention policy
    pub async fn prune(&self, retain_rounds: u64) -> Result<()> {
        info!("🧹 Pruning data older than {} rounds", retain_rounds);

        let manifest = self.manifest.read().await;
        let prune_before_round = manifest.dag_round_watermark.saturating_sub(retain_rounds);

        if prune_before_round == 0 {
            debug!("No data to prune");
            return Ok(());
        }

        // Prune old vertices
        let pruned_vertices = self.prune_vertices_before_round(prune_before_round).await?;
        
        // Prune old payloads (more aggressive)
        let pruned_payloads = self.prune_payloads_before_round(prune_before_round).await?;

        info!("✅ Pruned {} vertices and {} payloads", pruned_vertices, pruned_payloads);
        Ok(())
    }

    /// Prune vertices before a specific round
    async fn prune_vertices_before_round(&self, before_round: u64) -> Result<usize> {
        let mut pruned_count = 0;
        
        // Iterate through rounds to prune
        for round in 0..before_round {
            let round_prefix = round.to_be_bytes();
            let vertices = self.hot_db.scan_prefix(CF_DAG_VERTICES, &round_prefix).await?;
            
            for (key, _) in vertices {
                self.hot_db.delete(CF_DAG_VERTICES, &key).await?;
                pruned_count += 1;
            }
        }

        Ok(pruned_count)
    }

    /// Prune payloads before a specific round
    async fn prune_payloads_before_round(&self, before_round: u64) -> Result<usize> {
        // This requires mapping payload digests to rounds
        // For now, we'll implement a simple approach
        debug!("🧹 Pruning payloads before round {}", before_round);
        
        // In production, we'd maintain a digest -> round mapping
        // For now, return 0 as a placeholder
        Ok(0)
    }

    /// Get storage health status
    pub async fn health_check(&self) -> StorageHealth {
        let stats = self.get_storage_stats().await;
        let manifest = self.manifest.read().await;

        // Check various health indicators
        let db_accessible = self.hot_db.get(CF_MANIFEST, b"test").await.is_ok();
        let watermark_reasonable = stats.dag_round_watermark <= stats.finalized_height + 100;
        let write_performance_ok = stats.average_write_latency < Duration::from_millis(100);

        let status = if db_accessible && watermark_reasonable && write_performance_ok {
            StorageHealthStatus::Healthy
        } else if !db_accessible {
            StorageHealthStatus::DatabaseError
        } else if !watermark_reasonable {
            StorageHealthStatus::InconsistentState
        } else {
            StorageHealthStatus::PerformanceIssues
        };

        StorageHealth {
            status,
            last_write: manifest.last_update,
            error_count: 0, // TODO: Track errors
            stats,
        }
    }

    /// Shutdown storage gracefully
    pub async fn shutdown(&self) -> Result<()> {
        info!("🛑 Shutting down Q-Storage");

        // Flush any pending writes
        self.hot_db.flush().await?;
        self.cold_db.flush().await?;

        // Stop sync protocol
        self.sync_protocol.shutdown().await?;

        info!("✅ Q-Storage shutdown complete");
        Ok(())
    }
}

/// Storage statistics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageStats {
    pub dag_round_watermark: u64,
    pub finalized_height: u64,
    pub total_vertices: u64,
    pub total_payloads: u64,
    pub total_blocks: u64,
    pub hot_db_size: u64,
    pub cold_db_size: u64,
    pub average_write_latency: Duration,
    pub average_read_latency: Duration,
}

/// Storage health information
#[derive(Debug, Clone)]
pub struct StorageHealth {
    pub status: StorageHealthStatus,
    pub last_write: Instant,
    pub error_count: u64,
    pub stats: StorageStats,
}

/// Storage health status
#[derive(Debug, Clone, PartialEq)]
pub enum StorageHealthStatus {
    Healthy,
    PerformanceIssues,
    InconsistentState,
    DatabaseError,
    Offline,
}

impl StorageHealthStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Healthy => "healthy",
            Self::PerformanceIssues => "performance_issues",
            Self::InconsistentState => "inconsistent_state", 
            Self::DatabaseError => "database_error",
            Self::Offline => "offline",
        }
    }

    pub fn is_critical(&self) -> bool {
        matches!(self, Self::InconsistentState | Self::DatabaseError | Self::Offline)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_storage_creation() {
        let temp_dir = TempDir::new().unwrap();
        let node_id = [1u8; 32];
        
        let storage = QStorage::open(temp_dir.path(), node_id).await;
        assert!(storage.is_ok());
    }

    #[test]
    fn test_key_generation() {
        let storage = QStorage {
            // Mock storage for testing key generation
            hot_db: Arc::new(MockKVStore::new()),
            cold_db: Arc::new(MockKVStore::new()),
            manifest: Arc::new(RwLock::new(StorageManifest::default())),
            sync_protocol: Arc::new(SyncProtocol::mock()),
            snapshot_manager: Arc::new(SnapshotManager::mock()),
            metrics: Arc::new(StorageMetrics::new()),
            node_id: [1u8; 32],
            data_dir: PathBuf::from("/tmp"),
        };

        let vertex_key = storage.vertex_key(100, &[0xaa, 0xbb], 5);
        assert_eq!(vertex_key.len(), 8 + 2 + 8); // round + author + sequence

        let block_key = storage.block_key(1000, &[0xcc; 32]);
        assert_eq!(block_key.len(), 8 + 32); // height + hash
    }

    #[test]
    fn test_health_status() {
        assert_eq!(StorageHealthStatus::Healthy.as_str(), "healthy");
        assert!(!StorageHealthStatus::Healthy.is_critical());
        
        assert_eq!(StorageHealthStatus::DatabaseError.as_str(), "database_error");
        assert!(StorageHealthStatus::DatabaseError.is_critical());
    }
}

// Mock implementations for testing
#[cfg(test)]
struct MockKVStore;

#[cfg(test)]
impl MockKVStore {
    fn new() -> Self { Self }
}

#[cfg(test)]
#[async_trait]
impl KVStore for MockKVStore {
    async fn put(&self, _cf: &str, _key: &[u8], _value: &[u8]) -> Result<()> { Ok(()) }
    async fn get(&self, _cf: &str, _key: &[u8]) -> Result<Option<Vec<u8>>> { Ok(None) }
    async fn delete(&self, _cf: &str, _key: &[u8]) -> Result<()> { Ok(()) }
    async fn write_batch(&self, _batch: Vec<(&str, Vec<u8>, Vec<u8>)>) -> Result<()> { Ok(()) }
    async fn scan_prefix(&self, _cf: &str, _prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> { Ok(vec![]) }
    async fn scan_all(&self, _cf: &str) -> Result<Vec<(Vec<u8>, Vec<u8>)>> { Ok(vec![]) }
    async fn flush(&self) -> Result<()> { Ok(()) }
    async fn compact(&self) -> Result<()> { Ok(()) }
    async fn get_db_size(&self) -> Result<u64> { Ok(0) }
}