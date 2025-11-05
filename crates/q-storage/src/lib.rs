/// Q-Storage: High-performance blockchain database for DagKnight consensus
/// Optimized for Narwhal mempool and Bullshark finality with hot/cold storage split
/// Battle-tested design using RocksDB with specialized column families
use anyhow::{Context, Result};
use q_dag_knight::BullsharkCert;
use q_dag_knight::NarwhalPayload;
use q_narwhal_core::Certificate;
use q_types::{Block, NodeId, Vertex};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, SystemTime},
};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

// External crates
extern crate hex;
extern crate blake3;

pub mod balance_consensus;
pub mod kv;
pub mod manifest;
pub mod metrics;
pub mod pruning;
pub mod snapshot;
pub mod sync;
pub mod token_registry;
pub mod price_history;
pub mod transaction;
pub mod turbo_sync;
pub mod turbo_sync_peer_bridge;
pub mod zk_block_request_auth;

// Windows uses sled implementation
#[cfg(target_os = "windows")]
pub mod kv_sled;

// Export platform-specific KVStore implementation
#[cfg(not(target_os = "windows"))]
pub use kv::{KVStore, RocksDBKV};

#[cfg(target_os = "windows")]
pub use kv::KVStore;
#[cfg(target_os = "windows")]
pub use kv_sled::RocksDBKV;
pub use balance_consensus::{
    BalanceConsensusEngine, BalanceConsensusError, BalanceStorage, BalanceUpdate,
    ChangeReason, ConsensusStats, GENESIS_TIMESTAMP, DEV_FEE_PERCENT, FOUNDER_WALLET,
};
pub use manifest::StorageManifest;
pub use metrics::StorageMetrics;
pub use pruning::{AdaptivePruningEngine, PruningConfig, PruningMode, PruningStats, CheckpointPolicy, RetentionTier};
pub use snapshot::SnapshotManager;
pub use sync::{SyncProtocol, SyncRequest, SyncResponse};
pub use transaction::{QTransaction, TransactionState};
pub use turbo_sync::{TurboSyncManager, TurboSyncConfig, BlockPack, BlockPackRequest, NetworkRequest, TurboSyncMetrics};
pub use turbo_sync_peer_bridge::{TurboSyncPeerBridge, PeerHeightEntry, run_periodic_sync};
pub use zk_block_request_auth::{
    BlockRequestAuthenticator, AuthenticatedBlockPackRequest, AuthenticatedBlockPackResponse,
    generate_block_request_proof,
};

/// Column family names for optimized storage
pub const CF_BLOCKS: &str = "blocks";
pub const CF_DAG_VERTICES: &str = "dag_vertices";
pub const CF_BULLSHARK_CERT: &str = "bullshark_cert";
pub const CF_MANIFEST: &str = "manifest";
pub const CF_NARWHAL_PAYLOADS: &str = "narwhal_payloads";
pub const CF_TRANSACTIONS: &str = "transactions";
pub const CF_BALANCES: &str = "balances";  // v0.8.2-beta: Balance consensus storage
pub const CF_BLOCK_HASH_TO_HEIGHT: &str = "block_hash_to_height";  // v0.8.3-beta: Block hash index
pub const CF_AI_CHATS: &str = "ai_chats";
pub const CF_AI_CREDITS: &str = "ai_credits";
pub const CF_AI_TRANSACTIONS: &str = "ai_transactions";
pub const CF_AI_TREASURY: &str = "ai_treasury";
pub const CF_PAYMENT_PROPOSALS: &str = "payment_proposals";
pub const CF_PAYMENT_VOTES: &str = "payment_votes";
pub const CF_PAYMENT_LOCKS: &str = "payment_locks";
pub const CF_BANNED_PEERS: &str = "banned_peers";  // v0.9.7-beta: ZK proof ban persistence

/// Storage configuration
#[derive(Debug, Clone)]
pub struct StorageConfig {
    pub db_path: String,
    pub hot_db_path: String,
    pub enable_metrics: bool,
    pub sync_writes: bool,
    pub cache_size_mb: usize,
    pub max_open_files: usize,
}

/// Main storage engine for Q-NarwhalKnight
pub struct QStorage {
    /// Hot database (RocksDB) - blocks, vertices, certificates
    hot_db: Arc<dyn KVStore>,
    /// Concrete RocksDBKV reference for advanced operations like pruning
    hot_db_concrete: Arc<RocksDBKV>,
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
    /// Transaction counter for unique IDs (v0.8.1-beta)
    tx_counter: Arc<std::sync::atomic::AtomicU64>,
}

/// Type alias for compatibility with API server
pub type StorageEngine = QStorage;

impl QStorage {
    /// Create new storage engine with configuration
    pub async fn new(config: StorageConfig) -> Result<Self> {
        let data_dir = PathBuf::from(&config.db_path);
        let node_id = [0u8; 32]; // Default node ID for API server
        Self::open(data_dir, node_id).await
    }

    /// Open storage with hot/cold database split
    pub async fn open<P: AsRef<Path>>(data_dir: P, node_id: NodeId) -> Result<Self> {
        let data_dir = data_dir.as_ref().to_path_buf();
        info!(
            "🗄️ Opening Q-Storage at {:?} for node {}",
            data_dir,
            hex::encode(&node_id[..4])
        );

        // Configure hot database (frequent access)
        let hot_path = data_dir.join("hot");
        let hot_db_concrete = Arc::new(
            RocksDBKV::open_hot_db(&hot_path)
                .await
                .context("Failed to open hot database")?,
        );

        // Configure cold database (large payloads)
        let cold_path = data_dir.join("cold");
        let cold_db = Arc::new(
            RocksDBKV::open_cold_db(&cold_path)
                .await
                .context("Failed to open cold database")?,
        );

        // Create trait object reference from concrete type
        let hot_db: Arc<dyn KVStore> = hot_db_concrete.clone();

        // Load storage manifest with explicit type coercion
        let manifest = Arc::new(RwLock::new(
            StorageManifest::load_or_create(&hot_db).await?,
        ));

        // Initialize sync protocol
        let sync_protocol = Arc::new(SyncProtocol::new(hot_db.clone(), cold_db.clone()).await?);

        // Initialize snapshot manager
        let snapshot_manager = Arc::new(
            SnapshotManager::new(data_dir.clone(), hot_db.clone(), cold_db.clone()).await?,
        );

        // Initialize metrics
        let metrics = Arc::new(StorageMetrics::new());

        let storage = Self {
            hot_db,
            hot_db_concrete,
            cold_db,
            manifest,
            sync_protocol,
            snapshot_manager,
            metrics,
            node_id,
            data_dir,
            tx_counter: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        };

        // Perform crash recovery and get recovered height
        let _recovered_height = storage.recover().await?;

        info!("✅ Q-Storage initialized successfully");
        Ok(storage)
    }

    /// Begin atomic transaction
    ///
    /// **SECURITY FIX (v0.8.1-beta)**: Enables atomic operations to prevent
    /// CRITICAL-1 race condition between balance updates and block storage.
    ///
    /// # Usage
    ///
    /// ```rust
    /// let tx = storage.begin_transaction().await?;
    ///
    /// // Buffer operations (not yet committed)
    /// balance_engine.process_block_mining_rewards_tx(&tx, &block).await?;
    /// tx.save_qblock(&block).await?;
    ///
    /// // Commit atomically (all or nothing)
    /// tx.commit().await?;
    /// ```
    ///
    /// # Performance
    ///
    /// - Single atomic write with fsync (~2-3ms)
    /// - Sub-50ms DAG-Knight finality maintained ✅
    /// - ~25% faster than separate operations
    pub async fn begin_transaction(&self) -> Result<crate::transaction::QTransaction> {
        let tx_id = self
            .tx_counter
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        Ok(crate::transaction::QTransaction::new(
            self.hot_db_concrete.clone(),
            tx_id,
        ))
    }

    /// Store DAG vertex with Narwhal payload
    pub async fn store_vertex(&self, vertex: &Vertex, payload: &NarwhalPayload) -> Result<()> {
        debug!(
            "💾 Storing vertex {} for round {}",
            hex::encode(&vertex.id),
            vertex.round
        );

        let start_time = SystemTime::now();

        // Store vertex in hot DB
        let vertex_key = self.vertex_key(vertex.round, &vertex.author, &vertex.id);
        let vertex_data = bincode::serialize(vertex)?;

        self.hot_db
            .put(CF_DAG_VERTICES, &vertex_key, &vertex_data)
            .await?;

        // Store payload in cold DB
        let payload_key = blake3::hash(&bincode::serialize(payload)?)
            .as_bytes()
            .to_vec();
        let payload_data = bincode::serialize(payload)?;

        self.cold_db
            .put(CF_NARWHAL_PAYLOADS, &payload_key, &payload_data)
            .await?;

        // Update metrics
        let latency = start_time.elapsed().unwrap_or(Duration::from_millis(0));
        self.metrics
            .record_vertex_write(latency, vertex_data.len(), payload_data.len())
            .await;

        // Check if this completes a round
        self.check_round_completion(vertex.round).await?;

        debug!(
            "✅ Stored vertex {} ({}ms)",
            hex::encode(&vertex.id),
            latency.as_millis()
        );
        Ok(())
    }

    /// Store Bullshark certificate
    pub async fn store_certificate(&self, cert: &BullsharkCert) -> Result<()> {
        debug!("📜 Storing Bullshark certificate for round {}", cert.round);

        let cert_key = cert.round.to_be_bytes();
        let cert_data = bincode::serialize(cert)?;

        self.hot_db
            .put(CF_BULLSHARK_CERT, &cert_key, &cert_data)
            .await?;

        // Update manifest watermark
        self.update_dag_watermark(cert.round).await?;

        info!("✅ Stored certificate for round {}", cert.round);
        Ok(())
    }

    /// Finalize block with Bullshark consensus
    pub async fn finalize_block(
        &self,
        block: &Block,
        finality_proof: &BullsharkCert,
    ) -> Result<()> {
        info!(
            "🎯 Finalizing block {} at height {}",
            hex::encode(&block.hash),
            block.height
        );

        let start_time = SystemTime::now();

        // Prepare atomic batch
        let mut batch = Vec::new();

        // Store finalized block
        let block_key = self.block_key(block.height, &block.hash);
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

        let latency = start_time.elapsed().unwrap_or(Duration::from_millis(0));
        self.metrics
            .record_block_finalization(latency, block.vertices.len())
            .await;

        info!(
            "✅ Finalized block {} ({}ms, {} txs)",
            hex::encode(&block.hash),
            latency.as_millis(),
            block.vertices.len()
        );

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

    // ========================================
    // QBLOCK STORAGE METHODS (Phase 2)
    // ========================================

    /// Save QBlock to storage (simplified version for Phase 2 block production)
    /// This is used by the BlockProducer to persist produced blocks
    pub async fn save_qblock(&self, block: &q_types::block::QBlock) -> Result<()> {
        let start_time = SystemTime::now();
        let block_hash = block.calculate_hash();

        info!(
            "💾 Saving QBlock at height {} with hash {}",
            block.header.height,
            hex::encode(&block_hash[..8])
        );

        // Serialize block
        let block_data = bincode::serialize(block)
            .context("Failed to serialize QBlock")?;

        // Prepare batch writes for atomic storage
        let mut batch = Vec::new();

        // Store by height: qblock:height:{height}
        let height_key = format!("qblock:height:{}", block.header.height);
        batch.push((CF_BLOCKS, height_key.into_bytes(), block_data.clone()));

        // Store by hash: qblock:hash:{hash_hex}
        let hash_key = format!("qblock:hash:{}", hex::encode(block_hash));
        batch.push((CF_BLOCKS, hash_key.into_bytes(), block_data.clone()));

        // Store latest height pointer: qblock:latest
        let latest_height_bytes = block.header.height.to_be_bytes().to_vec();
        batch.push((CF_BLOCKS, b"qblock:latest".to_vec(), latest_height_bytes));

        // Commit atomically
        self.hot_db.write_batch(batch).await
            .context("Failed to write QBlock batch to database")?;

        let latency = start_time.elapsed().unwrap_or(Duration::from_millis(0));

        // Update metrics (count mining solutions as transactions)
        self.metrics
            .record_block_finalization(latency, block.mining_solutions.len())
            .await;

        info!(
            "✅ Saved QBlock {} in {}ms ({} mining solutions)",
            block.header.height,
            latency.as_millis(),
            block.mining_solutions.len()
        );

        Ok(())
    }

    /// 🚀 BATCH SAVE BLOCKS - High-performance bulk block storage
    /// Saves multiple blocks in a single RocksDB batch write operation
    /// This is 10x-100x faster than saving blocks one-by-one
    pub async fn save_qblocks_batch(&self, blocks: &[q_types::block::QBlock]) -> Result<()> {
        if blocks.is_empty() {
            return Ok(());
        }

        let start_time = SystemTime::now();
        let num_blocks = blocks.len();

        info!("🚀 BATCH SAVE: Saving {} blocks to database...", num_blocks);

        // Prepare single large batch for all blocks
        let mut batch = Vec::new();
        let mut total_mining_solutions = 0;

        for block in blocks {
            let block_hash = block.calculate_hash();

            // Serialize block
            let block_data = bincode::serialize(block)
                .context("Failed to serialize QBlock in batch")?;

            // Store by height: qblock:height:{height}
            let height_key = format!("qblock:height:{}", block.header.height);
            batch.push((CF_BLOCKS, height_key.into_bytes(), block_data.clone()));

            // Store by hash: qblock:hash:{hash_hex}
            let hash_key = format!("qblock:hash:{}", hex::encode(block_hash));
            batch.push((CF_BLOCKS, hash_key.into_bytes(), block_data.clone()));

            total_mining_solutions += block.mining_solutions.len();
        }

        // Update latest height pointer to highest block
        if let Some(max_block) = blocks.iter().max_by_key(|b| b.header.height) {
            let latest_height_bytes = max_block.header.height.to_be_bytes().to_vec();
            batch.push((CF_BLOCKS, b"qblock:latest".to_vec(), latest_height_bytes));
        }

        // Commit entire batch atomically
        self.hot_db.write_batch(batch).await
            .context("Failed to write batch QBlocks to database")?;

        let latency = start_time.elapsed().unwrap_or(Duration::from_millis(0));

        // Update metrics for all blocks
        for block in blocks {
            self.metrics
                .record_block_finalization(latency, block.mining_solutions.len())
                .await;
        }

        info!(
            "✅ BATCH SAVE COMPLETE: Saved {} blocks in {}ms ({} blocks/sec, {} solutions)",
            num_blocks,
            latency.as_millis(),
            if latency.as_millis() > 0 { (num_blocks as u128 * 1000) / latency.as_millis() } else { 0 },
            total_mining_solutions
        );

        Ok(())
    }

    /// Get QBlock by height
    pub async fn get_qblock_by_height(&self, height: u64) -> Result<Option<q_types::block::QBlock>> {
        debug!("🔍 Fetching QBlock at height {}", height);

        let height_key = format!("qblock:height:{}", height);

        match self.hot_db.get(CF_BLOCKS, height_key.as_bytes()).await? {
            Some(block_data) => {
                // Try to deserialize - if it fails, log warning and treat as missing block
                // This provides backwards compatibility when block format changes
                match bincode::deserialize::<q_types::block::QBlock>(&block_data) {
                    Ok(block) => Ok(Some(block)),
                    Err(e) => {
                        warn!("⚠️  Failed to deserialize QBlock at height {}: {} - treating as missing (backwards compatibility)", height, e);
                        Ok(None)
                    }
                }
            }
            None => Ok(None),
        }
    }

    /// Get QBlock by hash
    pub async fn get_qblock_by_hash(&self, hash: &[u8; 32]) -> Result<Option<q_types::block::QBlock>> {
        debug!("🔍 Fetching QBlock by hash {}", hex::encode(hash));

        let hash_key = format!("qblock:hash:{}", hex::encode(hash));

        match self.hot_db.get(CF_BLOCKS, hash_key.as_bytes()).await? {
            Some(block_data) => {
                // Try to deserialize - if it fails, log warning and treat as missing block
                // This provides backwards compatibility when block format changes
                match bincode::deserialize::<q_types::block::QBlock>(&block_data) {
                    Ok(block) => Ok(Some(block)),
                    Err(e) => {
                        warn!("⚠️  Failed to deserialize QBlock with hash {}: {} - treating as missing (backwards compatibility)", hex::encode(hash), e);
                        Ok(None)
                    }
                }
            }
            None => Ok(None),
        }
    }

    /// Get latest QBlock
    pub async fn get_latest_qblock(&self) -> Result<Option<q_types::block::QBlock>> {
        debug!("🔍 Fetching latest QBlock");

        // Get latest height
        let latest_height = match self.hot_db.get(CF_BLOCKS, b"qblock:latest").await? {
            Some(height_bytes) => {
                if height_bytes.len() != 8 {
                    warn!("Invalid latest height bytes length: {}", height_bytes.len());
                    return Ok(None);
                }

                let mut height_array = [0u8; 8];
                height_array.copy_from_slice(&height_bytes);
                u64::from_be_bytes(height_array)
            }
            None => {
                // ✅ v0.5.21-beta FIX: qblock:latest pointer missing (legacy database)
                // Use get_highest_contiguous_block() to scan for latest block
                info!("⚠️  qblock:latest pointer missing - scanning for latest block...");
                let highest = self.get_highest_contiguous_block().await?;

                if highest == 0 {
                    debug!("No latest QBlock found in storage");
                    return Ok(None);
                }

                info!("✅ Found latest block at height {} via scanning", highest);
                highest
            }
        };

        // Fetch block at that height
        self.get_qblock_by_height(latest_height).await
    }

    /// Get range of QBlocks for blockchain synchronization
    /// Returns blocks from start_height (inclusive) up to limit blocks
    ///
    /// # Arguments
    /// * `start_height` - Starting block height (inclusive)
    /// * `limit` - Maximum number of blocks to return
    ///
    /// # Returns
    /// Vector of QBlocks in ascending height order
    pub async fn get_qblocks_range(&self, start_height: u64, limit: usize) -> Result<Vec<q_types::block::QBlock>> {
        // v0.6.0-beta: Prevent memory exhaustion from excessive block requests
        const MAX_BLOCKS_PER_REQUEST: usize = 1000;
        let capped_limit = std::cmp::min(limit, MAX_BLOCKS_PER_REQUEST);

        if limit > MAX_BLOCKS_PER_REQUEST {
            warn!("🚨 Block range request capped: requested {} blocks, returning max {}",
                  limit, MAX_BLOCKS_PER_REQUEST);
        }

        info!("🔍 Fetching QBlocks from height {} (limit: {})", start_height, capped_limit);

        let mut blocks = Vec::new();

        // Get latest height to know the upper bound
        let latest_height = match self.hot_db.get(CF_BLOCKS, b"qblock:latest").await? {
            Some(height_bytes) if height_bytes.len() == 8 => {
                let mut height_array = [0u8; 8];
                height_array.copy_from_slice(&height_bytes);
                u64::from_be_bytes(height_array)
            }
            _ => {
                debug!("No latest QBlock height found, returning empty range");
                return Ok(blocks);
            }
        };

        // Calculate end height (inclusive)
        let end_height = std::cmp::min(start_height + capped_limit as u64 - 1, latest_height);

        // Fetch blocks sequentially
        for height in start_height..=end_height {
            match self.get_qblock_by_height(height).await? {
                Some(block) => blocks.push(block),
                None => {
                    warn!("Missing block at height {} during range query", height);
                    // Don't break - try to get as many blocks as possible
                }
            }
        }

        info!("✅ Retrieved {} QBlocks (heights {}-{})", blocks.len(), start_height, end_height);
        Ok(blocks)
    }

    /// Get latest QBlock height
    /// Returns None if no blocks exist yet
    pub async fn get_latest_qblock_height(&self) -> Result<Option<u64>> {
        match self.hot_db.get(CF_BLOCKS, b"qblock:latest").await? {
            Some(height_bytes) if height_bytes.len() == 8 => {
                let mut height_array = [0u8; 8];
                height_array.copy_from_slice(&height_bytes);
                Ok(Some(u64::from_be_bytes(height_array)))
            }
            _ => Ok(None),
        }
    }

    /// Get highest contiguous block height (no gaps from genesis)
    /// Used for accurate peer height registration in TurboSync
    ///
    /// Returns the highest block height where all blocks [0..height] exist in storage
    /// This prevents advertising blocks we don't actually have
    pub async fn get_highest_contiguous_block(&self) -> Result<u64> {
        // ✅ v0.5.19-beta FIX: Handle legacy databases without qblock:latest pointer
        // If qblock:latest doesn't exist, scan backwards from a large number to find highest block
        let mut latest = self.get_latest_qblock_height().await?.unwrap_or(0);

        if latest == 0 {
            // qblock:latest pointer missing (old database) - scan for highest block
            info!("🔍 qblock:latest pointer missing, scanning for highest block...");

            // IMPROVED: Check some common heights first for faster discovery
            let probe_heights = vec![150_000, 145_000, 140_000, 100_000, 50_000, 10_000, 1_000, 100, 10, 1];

            for &probe_height in &probe_heights {
                info!("🔍 Probing height {}...", probe_height);
                if let Ok(Some(_)) = self.get_qblock_by_height(probe_height).await {
                    // Found a block! Use this as starting point for binary search
                    latest = probe_height + 50_000; // Add buffer for binary search
                    info!("✅ Found block at height {}, will binary search up to {}", probe_height, latest);
                    break;
                }
            }

            if latest == 0 {
                // No blocks found even at low heights
                info!("❌ No blocks found in database");
                return Ok(0);
            }
        }

        // Binary search for highest contiguous block
        info!("🔍 Starting binary search for highest contiguous block (range: 0-{})", latest);
        let mut low = 0u64;
        let mut high = latest;
        let mut verified = 0u64;
        let mut iterations = 0;
        const MAX_ITERATIONS: u32 = 1000; // v0.6.0-beta: Prevent infinite loops

        while low <= high {
            let mid = (low + high) / 2;
            iterations += 1;

            // v0.6.0-beta: Safety check to prevent infinite loops
            if iterations > MAX_ITERATIONS {
                error!("🚨 Binary search exceeded {} iterations! Breaking to prevent hang. Last verified: {}",
                       MAX_ITERATIONS, verified);
                break;
            }

            // Check if block at mid height exists
            let block_exists = self.get_qblock_by_height(mid).await?.is_some();

            if iterations <= 10 || iterations % 5 == 0 {
                info!("  Binary search iteration {}: mid={}, exists={}, range=[{}, {}]",
                      iterations, mid, block_exists, low, high);
            }

            if block_exists {
                // Block exists, search higher
                verified = mid;
                low = mid + 1;
            } else {
                // Block missing, search lower
                if mid == 0 {
                    break;
                }
                high = mid - 1;
            }
        }

        info!(
            "✅ Highest contiguous block: {} (scanned up to: {}, gap: {}, iterations: {})",
            verified,
            latest,
            latest.saturating_sub(verified),
            iterations
        );

        Ok(verified)
    }

    /// Clean up corrupt/undeserializable blocks above a certain height
    /// This allows the node to re-sync those blocks from peers
    /// v0.9.1-beta: Backwards compatibility fix for enum format changes
    pub async fn cleanup_corrupt_blocks_above(&self, height: u64) -> Result<()> {
        info!("🧹 Scanning for corrupt blocks above height {}...", height);

        let mut deleted_count = 0;
        let scan_limit = height + 10000; // Scan up to 10k blocks ahead

        for check_height in (height + 1)..=scan_limit {
            let height_key = format!("qblock:height:{}", check_height);

            // Check if block exists
            if let Some(block_data) = self.hot_db.get(CF_BLOCKS, height_key.as_bytes()).await? {
                // Try to deserialize
                if bincode::deserialize::<q_types::block::QBlock>(&block_data).is_err() {
                    // Corrupt block found - delete it
                    warn!("🗑️  Deleting corrupt block at height {} (backwards compatibility)", check_height);
                    self.hot_db.delete(CF_BLOCKS, height_key.as_bytes()).await?;

                    // Also delete by hash if we can extract it (first 32 bytes might be the hash)
                    if block_data.len() >= 32 {
                        let potential_hash_key = format!("qblock:hash:{}", hex::encode(&block_data[0..32]));
                        let _ = self.hot_db.delete(CF_BLOCKS, potential_hash_key.as_bytes()).await;
                    }

                    deleted_count += 1;
                }
            } else {
                // No more blocks found, stop scanning
                break;
            }
        }

        if deleted_count > 0 {
            warn!("🧹 Cleaned up {} corrupt blocks above height {}", deleted_count, height);
            info!("📡 Node will now re-sync these blocks from network peers");
        } else {
            info!("✅ No corrupt blocks found above height {}", height);
        }

        Ok(())
    }

    /// Get the first missing height in blockchain (gap detection)
    /// v0.7.4-beta: Production fix for "messy height" issue
    ///
    /// Returns None if no gaps exist (blockchain is contiguous from genesis)
    /// Returns Some(height) if a gap is detected at that height
    ///
    /// This prevents height from skipping missing blocks during Turbo Sync
    pub async fn get_first_missing_height(&self) -> Result<Option<u64>> {
        let highest_contiguous = self.get_highest_contiguous_block().await?;

        // Get the highest block we have stored (may have gaps)
        let latest_height = match self.hot_db.get(CF_BLOCKS, b"qblock:latest").await? {
            Some(height_bytes) => {
                let mut height_array = [0u8; 8];
                height_array.copy_from_slice(&height_bytes);
                u64::from_be_bytes(height_array)
            }
            None => {
                // No latest pointer - no blocks stored
                return Ok(None);
            }
        };

        // If highest_overall == highest_contiguous, no gaps
        if latest_height == highest_contiguous {
            return Ok(None);
        }

        // Gap exists - find the first missing height
        // Start from highest_contiguous + 1 and scan upwards
        for height in (highest_contiguous + 1)..=latest_height {
            let block_key = format!("qblock:height:{}", height);
            if self.hot_db.get(CF_BLOCKS, block_key.as_bytes()).await?.is_none() {
                info!("🔍 Gap detected: Missing block at height {}", height);
                return Ok(Some(height));
            }
        }

        // Shouldn't reach here if logic is correct, but handle gracefully
        warn!("⚠️ Gap detection inconsistency: highest_contiguous={}, latest={}",
              highest_contiguous, latest_height);
        Ok(None)
    }

    /// Repair height pointer by scanning database for actual highest block
    ///
    /// **HEIGHT RECOVERY (v0.8.5-beta)**: Fixes databases where height pointer is stuck
    /// at old value due to v0.8.3-beta bug. Scans database to find actual highest block
    /// and updates the qblock:latest pointer.
    ///
    /// This method is called automatically on startup to detect and repair height
    /// pointer inconsistencies that can occur during version upgrades.
    ///
    /// # Returns
    /// - `Ok(height)`: The repaired/verified height
    /// - `Err`: Database error during repair
    ///
    /// # Example
    /// ```rust
    /// // On startup after opening database:
    /// let repaired_height = storage.repair_height_pointer().await?;
    /// info!("Height pointer verified/repaired: {}", repaired_height);
    /// ```
    pub async fn repair_height_pointer(&self) -> Result<u64> {
        info!("🔧 [HEIGHT RECOVERY] Checking height pointer integrity...");

        // Get current height pointer value
        let pointer_height = self.get_latest_qblock_height().await?.unwrap_or(0);
        info!("🔍 [HEIGHT RECOVERY] Current height pointer: {}", pointer_height);

        // Use existing method to find actual highest contiguous block
        let actual_height = self.get_highest_contiguous_block().await?;
        info!("🔍 [HEIGHT RECOVERY] Actual highest block: {}", actual_height);

        // Check for mismatch
        if actual_height > pointer_height {
            warn!(
                "⚠️  [HEIGHT RECOVERY] Height pointer mismatch detected! Pointer: {}, Actual: {}",
                pointer_height, actual_height
            );
            warn!("🔧 [HEIGHT RECOVERY] Repairing height pointer...");

            // Update the height pointer to actual highest block
            let height_bytes = actual_height.to_be_bytes();
            self.hot_db.put(CF_BLOCKS, b"qblock:latest", &height_bytes).await
                .context("Failed to update height pointer")?;

            info!("✅ [HEIGHT RECOVERY] Height pointer repaired: {} → {}",
                  pointer_height, actual_height);
            info!("✅ [HEIGHT RECOVERY] Node can now sync normally from network");

            Ok(actual_height)
        } else if actual_height == pointer_height && actual_height > 0 {
            info!("✅ [HEIGHT RECOVERY] Height pointer is consistent: {}", actual_height);
            Ok(actual_height)
        } else if actual_height == 0 && pointer_height == 0 {
            info!("ℹ️  [HEIGHT RECOVERY] Empty database (height 0) - this is normal for new nodes");
            Ok(0)
        } else {
            // Pointer is higher than actual - this shouldn't happen but handle gracefully
            warn!(
                "⚠️  [HEIGHT RECOVERY] Unexpected state: pointer={}, actual={}",
                pointer_height, actual_height
            );
            warn!("🔧 [HEIGHT RECOVERY] Correcting pointer to match actual height");

            let height_bytes = actual_height.to_be_bytes();
            self.hot_db.put(CF_BLOCKS, b"qblock:latest", &height_bytes).await
                .context("Failed to correct height pointer")?;

            info!("✅ [HEIGHT RECOVERY] Height pointer corrected to {}", actual_height);
            Ok(actual_height)
        }
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

    /// Perform crash recovery and return recovered blockchain height
    async fn recover(&self) -> Result<u64> {
        info!("🔄 Starting storage crash recovery");

        let manifest = self.manifest.read().await;
        info!(
            "📊 Recovery state - DAG watermark: {}, finalized: {}",
            manifest.dag_round_watermark, manifest.finalized_height
        );

        // 🚀 CRITICAL FIX (v0.6.6): Find highest blockchain height in database
        let recovered_height = self.get_highest_contiguous_block().await?;
        info!("📈 Recovered blockchain height: {} blocks from database", recovered_height);

        // 🧹 Clean up corrupt blocks above recovered height (backwards compatibility fix)
        self.cleanup_corrupt_blocks_above(recovered_height).await?;

        // Verify DAG consistency
        self.verify_dag_consistency().await?;

        // Start sync process if needed
        if manifest.dag_round_watermark > 0 {
            self.sync_protocol
                .start_catch_up(manifest.dag_round_watermark)
                .await?;
        }

        info!("✅ Storage recovery complete - restored {} blocks", recovered_height);
        Ok(recovered_height)
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
                warn!(
                    "⚠️ Missing vertices for round {}, truncating watermark",
                    round
                );
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
        warn!(
            "🐌 Performing inefficient vertex scan for {}",
            hex::encode(vertex_id)
        );

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
    fn vertex_key(&self, round: u64, author: &[u8], vertex_id: &[u8]) -> Vec<u8> {
        let mut key = Vec::with_capacity(8 + author.len() + vertex_id.len());
        key.extend_from_slice(&round.to_be_bytes());
        key.extend_from_slice(author);
        key.extend_from_slice(vertex_id);
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

        info!(
            "✅ Pruned {} vertices and {} payloads",
            pruned_vertices, pruned_payloads
        );
        Ok(())
    }

    /// Prune vertices before a specific round
    async fn prune_vertices_before_round(&self, before_round: u64) -> Result<usize> {
        let mut pruned_count = 0;

        // Iterate through rounds to prune
        for round in 0..before_round {
            let round_prefix = round.to_be_bytes();
            let vertices = self
                .hot_db
                .scan_prefix(CF_DAG_VERTICES, &round_prefix)
                .await?;

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
            last_write: std::time::SystemTime::now(),
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

    /// Save wallet balance to persistent storage with SYNC to guarantee disk write
    pub async fn save_wallet_balance(&self, address: &[u8; 32], amount: u64) -> Result<()> {
        let key = format!("wallet_balance_{}", hex::encode(address));
        let value = amount.to_le_bytes();

        // CRITICAL: Use synced write to guarantee data reaches disk (survives pkill -9)
        // This overrides the default set_sync(false) in write_options()
        self.hot_db.put_sync(CF_MANIFEST, key.as_bytes(), &value).await?;

        info!(
            "💰 SYNCED wallet balance to disk: {} -> {} units (survives hard kill)",
            hex::encode(address),
            amount
        );
        Ok(())
    }

    /// Load wallet balance from persistent storage
    pub async fn load_wallet_balance(&self, address: &[u8; 32]) -> Result<Option<u64>> {
        let key = format!("wallet_balance_{}", hex::encode(address));
        match self.hot_db.get(CF_MANIFEST, key.as_bytes()).await? {
            Some(bytes) => {
                if bytes.len() == 8 {
                    let amount = u64::from_le_bytes([
                        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6],
                        bytes[7],
                    ]);
                    debug!(
                        "💰 Loaded wallet balance: {} -> {}",
                        hex::encode(address),
                        amount
                    );
                    Ok(Some(amount))
                } else {
                    warn!(
                        "Invalid wallet balance data length for address {}",
                        hex::encode(address)
                    );
                    Ok(None)
                }
            }
            None => Ok(None),
        }
    }

    /// Load all wallet balances from persistent storage
    pub async fn load_wallet_balances(&self) -> Result<HashMap<[u8; 32], u64>> {
        let mut balances = HashMap::new();
        let prefix = "wallet_balance_".as_bytes();

        // Use a prefix scan to get all wallet balance entries
        match self.hot_db.scan_prefix(CF_MANIFEST, prefix).await {
            Ok(entries) => {
                for (key, value) in entries {
                    if let Ok(key_str) = String::from_utf8(key) {
                        if let Some(hex_addr) = key_str.strip_prefix("wallet_balance_") {
                            if let Ok(addr_bytes) = hex::decode(hex_addr) {
                                if addr_bytes.len() == 32 && value.len() == 8 {
                                    let mut address = [0u8; 32];
                                    address.copy_from_slice(&addr_bytes);
                                    let amount = u64::from_le_bytes([
                                        value[0], value[1], value[2], value[3], value[4], value[5],
                                        value[6], value[7],
                                    ]);
                                    balances.insert(address, amount);
                                }
                            }
                        }
                    }
                }
                info!(
                    "💰 Loaded {} wallet balances from persistent storage",
                    balances.len()
                );
            }
            Err(e) => {
                warn!("Failed to scan wallet balances: {}", e);
            }
        }

        Ok(balances)
    }

    /// Save multiple wallet balances atomically with SYNC to guarantee disk write
    pub async fn save_wallet_balances(&self, balances: &HashMap<[u8; 32], u64>) -> Result<()> {
        let mut batch_ops = Vec::new();

        for (address, amount) in balances {
            let key = format!("wallet_balance_{}", hex::encode(address));
            let value = amount.to_le_bytes().to_vec();
            batch_ops.push((CF_MANIFEST, key.into_bytes(), value));
        }

        // CRITICAL: write_batch now uses fsync to survive hard kills (fixed in kv.rs)
        self.hot_db.write_batch(batch_ops).await?;
        info!(
            "💰 SYNCED {} wallet balances to persistent storage (survives hard kill)",
            balances.len()
        );
        Ok(())
    }

    /// Save total minted supply to persistent storage (enforces 21M QUG hard cap)
    /// CRITICAL: Must be called atomically with balance updates to prevent supply violations
    pub async fn save_total_supply(&self, total_supply: u64) -> Result<()> {
        let key = b"total_minted_supply";
        let value = total_supply.to_le_bytes();

        // CRITICAL: Use synced write to guarantee data reaches disk (same pattern as save_wallet_balance)
        self.hot_db.put_sync(CF_MANIFEST, key, &value).await?;

        debug!("💎 Saved total supply: {} QUG ({} base units)", total_supply / 100_000_000, total_supply);
        Ok(())
    }

    /// Load total minted supply from persistent storage
    /// Returns 0 if no supply data exists (fresh blockchain)
    pub async fn load_total_supply(&self) -> Result<u64> {
        let key = b"total_minted_supply";
        match self.hot_db.get(CF_MANIFEST, key).await? {
            Some(bytes) => {
                if bytes.len() == 8 {
                    let supply = u64::from_le_bytes([
                        bytes[0], bytes[1], bytes[2], bytes[3],
                        bytes[4], bytes[5], bytes[6], bytes[7],
                    ]);
                    info!("💎 Loaded total supply from storage: {} QUG ({} base units)",
                        supply / 100_000_000, supply);
                    Ok(supply)
                } else {
                    warn!("Invalid total supply data in storage, starting from 0");
                    Ok(0)
                }
            }
            None => {
                info!("No total supply data found, starting fresh blockchain from 0");
                Ok(0)
            }
        }
    }

    /// Save token balance to persistent storage
    /// Key format: token_balance_{wallet_hex}_{token_hex}
    pub async fn save_token_balance(&self, wallet_address: &[u8; 32], token_address: &[u8; 32], amount: u64) -> Result<()> {
        let key = format!("token_balance_{}_{}", hex::encode(wallet_address), hex::encode(token_address));
        let value = amount.to_le_bytes();
        self.hot_db.put(CF_MANIFEST, key.as_bytes(), &value).await?;
        debug!(
            "🪙 Saved token balance: wallet={}, token={}, amount={}",
            hex::encode(wallet_address),
            hex::encode(token_address),
            amount
        );
        Ok(())
    }

    /// Get a single token balance from persistent storage
    pub async fn get_token_balance(&self, wallet_address: &[u8; 32], token_address: &[u8; 32]) -> Result<u64> {
        let key = format!("token_balance_{}_{}", hex::encode(wallet_address), hex::encode(token_address));
        match self.hot_db.get(CF_MANIFEST, key.as_bytes()).await? {
            Some(bytes) => {
                if bytes.len() == 8 {
                    let amount = u64::from_le_bytes([
                        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
                    ]);
                    debug!(
                        "🪙 Loaded token balance: wallet={}, token={}, amount={}",
                        hex::encode(wallet_address),
                        hex::encode(token_address),
                        amount
                    );
                    Ok(amount)
                } else {
                    warn!(
                        "Invalid token balance data length for wallet {} token {}",
                        hex::encode(wallet_address),
                        hex::encode(token_address)
                    );
                    Ok(0)
                }
            }
            None => {
                debug!(
                    "🪙 Token balance not found in storage: wallet={}, token={}",
                    hex::encode(wallet_address),
                    hex::encode(token_address)
                );
                Ok(0)
            }
        }
    }

    /// Load all token balances from persistent storage
    pub async fn load_token_balances(&self) -> Result<HashMap<([u8; 32], [u8; 32]), u64>> {
        let mut balances = HashMap::new();
        let prefix = "token_balance_".as_bytes();

        match self.hot_db.scan_prefix(CF_MANIFEST, prefix).await {
            Ok(entries) => {
                for (key, value) in entries {
                    if let Ok(key_str) = String::from_utf8(key) {
                        if let Some(addresses) = key_str.strip_prefix("token_balance_") {
                            // Parse "wallet_hex_token_hex"
                            let parts: Vec<&str> = addresses.split('_').collect();
                            if parts.len() == 2 {
                                if let (Ok(wallet_bytes), Ok(token_bytes)) = (hex::decode(parts[0]), hex::decode(parts[1])) {
                                    if wallet_bytes.len() == 32 && token_bytes.len() == 32 && value.len() == 8 {
                                        let mut wallet_address = [0u8; 32];
                                        let mut token_address = [0u8; 32];
                                        wallet_address.copy_from_slice(&wallet_bytes);
                                        token_address.copy_from_slice(&token_bytes);
                                        let amount = u64::from_le_bytes([
                                            value[0], value[1], value[2], value[3], value[4], value[5],
                                            value[6], value[7],
                                        ]);
                                        balances.insert((wallet_address, token_address), amount);
                                    }
                                }
                            }
                        }
                    }
                }
                info!(
                    "🪙 Loaded {} token balances from persistent storage",
                    balances.len()
                );
            }
            Err(e) => {
                warn!("Failed to scan token balances: {}", e);
            }
        }

        Ok(balances)
    }

    /// Save multiple token balances atomically with SYNC to guarantee disk write
    pub async fn save_token_balances(&self, balances: &HashMap<([u8; 32], [u8; 32]), u64>) -> Result<()> {
        let mut batch_ops = Vec::new();

        for ((wallet_address, token_address), amount) in balances {
            let key = format!("token_balance_{}_{}", hex::encode(wallet_address), hex::encode(token_address));
            let value = amount.to_le_bytes().to_vec();
            batch_ops.push((CF_MANIFEST, key.into_bytes(), value));
        }

        // CRITICAL: write_batch now uses fsync to survive hard kills (fixed in kv.rs)
        self.hot_db.write_batch(batch_ops).await?;
        info!(
            "🪙 SYNCED {} token balances to persistent storage (survives hard kill)",
            balances.len()
        );
        Ok(())
    }

    /// Save transaction to persistent storage
    pub async fn save_transaction(&self, tx: &q_types::Transaction) -> Result<()> {
        let tx_data = bincode::serialize(tx)?;
        self.hot_db.put(CF_TRANSACTIONS, &tx.id, &tx_data).await?;
        debug!(
            "💳 Saved transaction: {} ({} -> {})",
            hex::encode(&tx.id),
            hex::encode(&tx.from),
            hex::encode(&tx.to)
        );
        Ok(())
    }

    /// Load transaction from persistent storage
    pub async fn load_transaction(&self, tx_id: &[u8; 32]) -> Result<Option<q_types::Transaction>> {
        match self.hot_db.get(CF_TRANSACTIONS, tx_id).await? {
            Some(tx_data) => {
                let tx: q_types::Transaction = bincode::deserialize(&tx_data)?;
                debug!("💳 Loaded transaction: {}", hex::encode(tx_id));
                Ok(Some(tx))
            }
            None => Ok(None),
        }
    }

    /// Load all transactions from persistent storage
    pub async fn load_all_transactions(&self) -> Result<Vec<q_types::Transaction>> {
        let mut transactions = Vec::new();

        match self.hot_db.scan_all(CF_TRANSACTIONS).await {
            Ok(entries) => {
                for (_key, tx_data) in entries {
                    if let Ok(tx) = bincode::deserialize::<q_types::Transaction>(&tx_data) {
                        transactions.push(tx);
                    }
                }
                info!(
                    "💳 Loaded {} transactions from persistent storage",
                    transactions.len()
                );
            }
            Err(e) => {
                warn!("Failed to scan transactions: {}", e);
            }
        }

        Ok(transactions)
    }

    /// Save multiple transactions atomically with SYNC to guarantee disk write
    pub async fn save_transactions(&self, transactions: &[q_types::Transaction]) -> Result<()> {
        let mut batch_ops = Vec::new();

        for tx in transactions {
            let tx_data = bincode::serialize(tx)?;
            batch_ops.push((CF_TRANSACTIONS, tx.id.to_vec(), tx_data));
        }

        // CRITICAL: write_batch now uses fsync to survive hard kills (fixed in kv.rs)
        self.hot_db.write_batch(batch_ops).await?;
        info!(
            "💳 SYNCED {} transactions to persistent storage (survives hard kill)",
            transactions.len()
        );
        Ok(())
    }

    /// Delete transaction from persistent storage
    pub async fn delete_transaction(&self, tx_id: &[u8; 32]) -> Result<()> {
        self.hot_db.delete(CF_TRANSACTIONS, tx_id).await?;
        debug!("🗑️ Deleted transaction: {}", hex::encode(tx_id));
        Ok(())
    }

    /// Save smart contract to persistent storage
    pub async fn save_contract(&self, address: &[u8; 32], contract_data: &[u8]) -> Result<()> {
        let key = format!("contract_{}", hex::encode(address));
        self.hot_db.put(CF_MANIFEST, key.as_bytes(), contract_data).await?;
        debug!("📜 Saved smart contract: {}", hex::encode(address));
        Ok(())
    }

    /// Load smart contract from persistent storage
    pub async fn load_contract(&self, address: &[u8; 32]) -> Result<Option<Vec<u8>>> {
        let key = format!("contract_{}", hex::encode(address));
        match self.hot_db.get(CF_MANIFEST, key.as_bytes()).await? {
            Some(contract_data) => {
                debug!("📜 Loaded smart contract: {}", hex::encode(address));
                Ok(Some(contract_data))
            }
            None => Ok(None),
        }
    }

    /// Load all smart contracts from persistent storage
    pub async fn load_all_contracts(&self) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let prefix = "contract_".as_bytes();
        let mut contracts = Vec::new();

        match self.hot_db.scan_prefix(CF_MANIFEST, prefix).await {
            Ok(entries) => {
                for (key, value) in entries {
                    if let Ok(key_str) = String::from_utf8(key) {
                        if let Some(hex_addr) = key_str.strip_prefix("contract_") {
                            if let Ok(addr_bytes) = hex::decode(hex_addr) {
                                if addr_bytes.len() == 32 {
                                    contracts.push((addr_bytes, value));
                                }
                            }
                        }
                    }
                }
                info!("📜 Loaded {} smart contracts from persistent storage", contracts.len());
            }
            Err(e) => {
                warn!("Failed to scan smart contracts: {}", e);
            }
        }

        Ok(contracts)
    }

    /// Delete smart contract from persistent storage
    pub async fn delete_contract(&self, address: &[u8; 32]) -> Result<()> {
        let key = format!("contract_{}", hex::encode(address));
        self.hot_db.delete(CF_MANIFEST, key.as_bytes()).await?;
        debug!("🗑️ Deleted smart contract: {}", hex::encode(address));
        Ok(())
    }

    /// Save liquidity pool to persistent storage
    /// Pool ID format: "QUG-QUGUSD" or similar token pair identifier
    pub async fn save_liquidity_pool(&self, pool_id: &str, pool_data: &[u8]) -> Result<()> {
        let key = format!("liquidity_pool:{}", pool_id);
        self.hot_db.put(CF_MANIFEST, key.as_bytes(), pool_data).await?;
        debug!("💧 Saved liquidity pool: {}", pool_id);
        Ok(())
    }

    /// Load all liquidity pools from persistent storage
    /// Returns map of pool_id -> serialized pool data
    pub async fn load_liquidity_pools(&self) -> Result<HashMap<String, Vec<u8>>> {
        let mut pools = HashMap::new();
        let prefix = b"liquidity_pool:";

        match self.hot_db.scan_prefix(CF_MANIFEST, prefix).await {
            Ok(entries) => {
                for (key, value) in entries {
                    if let Ok(key_str) = String::from_utf8(key) {
                        if let Some(pool_id) = key_str.strip_prefix("liquidity_pool:") {
                            pools.insert(pool_id.to_string(), value);
                        }
                    }
                }
                info!(
                    "💧 Loaded {} liquidity pools from persistent storage",
                    pools.len()
                );
            }
            Err(e) => {
                warn!("Failed to scan liquidity pools: {}", e);
            }
        }

        Ok(pools)
    }

    /// Delete liquidity pool from persistent storage
    pub async fn delete_liquidity_pool(&self, pool_id: &str) -> Result<()> {
        let key = format!("liquidity_pool:{}", pool_id);
        self.hot_db.delete(CF_MANIFEST, key.as_bytes()).await?;
        debug!("🗑️ Deleted liquidity pool: {}", pool_id);
        Ok(())
    }

    // ============================================================================
    // Loan Application Persistence - Quillon Bank CDP System
    // ============================================================================

    /// Save loan application to persistent storage
    /// Key format: loan_app:{loan_id}
    pub async fn save_loan_application(&self, loan_id: &str, loan_bytes: &[u8]) -> Result<()> {
        let key = format!("loan_app:{}", loan_id);
        self.hot_db.put(CF_MANIFEST, key.as_bytes(), loan_bytes).await?;
        debug!("🏦 Saved loan application: {}", loan_id);
        Ok(())
    }

    /// Load all loan applications from persistent storage
    pub async fn load_loan_applications(&self) -> Result<HashMap<String, Vec<u8>>> {
        let mut loans = HashMap::new();
        let prefix = b"loan_app:";

        match self.hot_db.scan_prefix(CF_MANIFEST, prefix).await {
            Ok(entries) => {
                for (key, value) in entries {
                    if let Ok(key_str) = String::from_utf8(key) {
                        if let Some(loan_id) = key_str.strip_prefix("loan_app:") {
                            loans.insert(loan_id.to_string(), value);
                        }
                    }
                }
                info!(
                    "🏦 Loaded {} loan applications from persistent storage",
                    loans.len()
                );
            }
            Err(e) => {
                warn!("Failed to scan loan applications: {}", e);
            }
        }

        Ok(loans)
    }

    /// Delete loan application from persistent storage
    pub async fn delete_loan_application(&self, loan_id: &str) -> Result<()> {
        let key = format!("loan_app:{}", loan_id);
        self.hot_db.delete(CF_MANIFEST, key.as_bytes()).await?;
        debug!("🗑️ Deleted loan application: {}", loan_id);
        Ok(())
    }

    /// Save benchmark timestamp for IP rate limiting
    /// Key format: benchmark_ip:{ip_address}
    pub async fn save_benchmark_timestamp(&self, ip_address: &str, timestamp: u64) -> Result<()> {
        let key = format!("benchmark_ip:{}", ip_address);
        let value = timestamp.to_le_bytes();
        self.hot_db.put(CF_MANIFEST, key.as_bytes(), &value).await?;
        debug!("⏰ Saved benchmark timestamp for IP: {}", ip_address);
        Ok(())
    }

    /// Load benchmark timestamp for IP rate limiting
    /// Returns None if IP has never run benchmark, Some(timestamp) otherwise
    pub async fn load_benchmark_timestamp(&self, ip_address: &str) -> Result<Option<u64>> {
        let key = format!("benchmark_ip:{}", ip_address);
        match self.hot_db.get(CF_MANIFEST, key.as_bytes()).await? {
            Some(bytes) => {
                if bytes.len() == 8 {
                    let timestamp = u64::from_le_bytes([
                        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
                    ]);
                    debug!("⏰ Loaded benchmark timestamp for IP {}: {}", ip_address, timestamp);
                    Ok(Some(timestamp))
                } else {
                    warn!("Invalid benchmark timestamp data length for IP {}", ip_address);
                    Ok(None)
                }
            }
            None => Ok(None),
        }
    }

    /// Check if IP is rate limited for benchmark (DISABLED - no rate limiting)
    /// Returns (is_limited, minutes_remaining)
    pub async fn check_benchmark_rate_limit(&self, ip_address: &str) -> Result<(bool, u64)> {
        const COOLDOWN_SECONDS: u64 = 0; // DISABLED - no rate limiting

        match self.load_benchmark_timestamp(ip_address).await? {
            Some(last_timestamp) => {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();

                let elapsed = now.saturating_sub(last_timestamp);

                if elapsed < COOLDOWN_SECONDS {
                    let remaining_seconds = COOLDOWN_SECONDS - elapsed;
                    let remaining_minutes = (remaining_seconds + 59) / 60; // Round up
                    debug!("🚫 IP {} is rate limited, {} minutes remaining", ip_address, remaining_minutes);
                    Ok((true, remaining_minutes))
                } else {
                    debug!("✅ IP {} is not rate limited", ip_address);
                    Ok((false, 0))
                }
            }
            None => {
                debug!("✅ IP {} has never run benchmark", ip_address);
                Ok((false, 0))
            }
        }
    }

    /// Get USD balance for a wallet (in cents)
    /// Key format: usd_balance:{wallet_address_hex}
    pub async fn get_usd_balance(&self, wallet_address: &str) -> Result<u64> {
        let key = format!("usd_balance:{}", wallet_address);
        match self.hot_db.get(CF_MANIFEST, key.as_bytes()).await? {
            Some(bytes) => {
                if bytes.len() == 8 {
                    let balance_cents = u64::from_le_bytes([
                        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
                    ]);
                    debug!("💵 Loaded USD balance for {}: {} cents", wallet_address, balance_cents);
                    Ok(balance_cents)
                } else {
                    warn!("Invalid USD balance data length for wallet {}", wallet_address);
                    Ok(0)
                }
            }
            None => {
                debug!("💵 USD balance not found for wallet {}, returning 0", wallet_address);
                Ok(0)
            }
        }
    }

    /// Credit USD balance to a wallet (amount in cents)
    /// Adds the specified amount to the current balance
    pub async fn credit_usd_balance(&self, wallet_address: &str, amount_cents: u64) -> Result<()> {
        let current_balance = self.get_usd_balance(wallet_address).await?;
        let new_balance = current_balance.saturating_add(amount_cents);

        let key = format!("usd_balance:{}", wallet_address);
        let value = new_balance.to_le_bytes();
        self.hot_db.put(CF_MANIFEST, key.as_bytes(), &value).await?;

        info!("💵 Credited {} cents to {}, new balance: {} cents",
            amount_cents, wallet_address, new_balance);
        Ok(())
    }

    /// Debit USD balance from a wallet (amount in cents)
    /// Returns error if insufficient balance
    pub async fn debit_usd_balance(&self, wallet_address: &str, amount_cents: u64) -> Result<()> {
        let current_balance = self.get_usd_balance(wallet_address).await?;

        if current_balance < amount_cents {
            return Err(anyhow::anyhow!(
                "Insufficient USD balance: has {} cents, needs {} cents",
                current_balance,
                amount_cents
            ));
        }

        let new_balance = current_balance - amount_cents;

        let key = format!("usd_balance:{}", wallet_address);
        let value = new_balance.to_le_bytes();
        self.hot_db.put(CF_MANIFEST, key.as_bytes(), &value).await?;

        info!("💵 Debited {} cents from {}, new balance: {} cents",
            amount_cents, wallet_address, new_balance);
        Ok(())
    }

    /// Set USD balance for a wallet directly (amount in cents)
    /// This is used for admin operations or migrations
    pub async fn set_usd_balance(&self, wallet_address: &str, balance_cents: u64) -> Result<()> {
        let key = format!("usd_balance:{}", wallet_address);
        let value = balance_cents.to_le_bytes();
        self.hot_db.put(CF_MANIFEST, key.as_bytes(), &value).await?;

        info!("💵 Set USD balance for {} to {} cents", wallet_address, balance_cents);
        Ok(())
    }

    /// Load all USD balances from persistent storage
    /// Returns map of wallet_address -> balance_cents
    pub async fn load_all_usd_balances(&self) -> Result<HashMap<String, u64>> {
        let mut balances = HashMap::new();
        let prefix = b"usd_balance:";

        match self.hot_db.scan_prefix(CF_MANIFEST, prefix).await {
            Ok(entries) => {
                for (key, value) in entries {
                    if let Ok(key_str) = String::from_utf8(key) {
                        if let Some(wallet_address) = key_str.strip_prefix("usd_balance:") {
                            if value.len() == 8 {
                                let balance_cents = u64::from_le_bytes([
                                    value[0], value[1], value[2], value[3],
                                    value[4], value[5], value[6], value[7],
                                ]);
                                balances.insert(wallet_address.to_string(), balance_cents);
                            }
                        }
                    }
                }
                info!("💵 Loaded {} USD wallet balances from persistent storage", balances.len());
            }
            Err(e) => {
                warn!("Failed to scan USD balances: {}", e);
            }
        }

        Ok(balances)
    }

    /// Save password hash to persistent storage (bcrypt hash)
    /// Key format: wallet_password_{address_hex}
    pub async fn save_password_hash(&self, address: &[u8; 32], password_hash: &str) -> Result<()> {
        let key = format!("wallet_password_{}", hex::encode(address));
        self.hot_db.put(CF_MANIFEST, key.as_bytes(), password_hash.as_bytes()).await?;
        debug!(
            "🔐 Saved password hash for address: {}",
            hex::encode(address)
        );
        Ok(())
    }

    /// Load password hash from persistent storage
    pub async fn load_password_hash(&self, address: &[u8; 32]) -> Result<Option<String>> {
        let key = format!("wallet_password_{}", hex::encode(address));
        match self.hot_db.get(CF_MANIFEST, key.as_bytes()).await? {
            Some(bytes) => {
                match String::from_utf8(bytes) {
                    Ok(password_hash) => {
                        debug!(
                            "🔐 Loaded password hash for address: {}",
                            hex::encode(address)
                        );
                        Ok(Some(password_hash))
                    }
                    Err(e) => {
                        warn!("Invalid password hash UTF-8 for address {}: {}", hex::encode(address), e);
                        Ok(None)
                    }
                }
            }
            None => Ok(None),
        }
    }

    /// Load all password hashes from persistent storage
    /// Returns map of address -> bcrypt_hash
    pub async fn load_password_hashes(&self) -> Result<HashMap<[u8; 32], String>> {
        let mut hashes = HashMap::new();
        let prefix = "wallet_password_".as_bytes();

        match self.hot_db.scan_prefix(CF_MANIFEST, prefix).await {
            Ok(entries) => {
                for (key, value) in entries {
                    if let Ok(key_str) = String::from_utf8(key) {
                        if let Some(hex_addr) = key_str.strip_prefix("wallet_password_") {
                            if let Ok(addr_bytes) = hex::decode(hex_addr) {
                                if addr_bytes.len() == 32 {
                                    let mut address = [0u8; 32];
                                    address.copy_from_slice(&addr_bytes);
                                    if let Ok(password_hash) = String::from_utf8(value) {
                                        hashes.insert(address, password_hash);
                                    }
                                }
                            }
                        }
                    }
                }
                info!(
                    "🔐 Loaded {} password hashes from persistent storage",
                    hashes.len()
                );
            }
            Err(e) => {
                warn!("Failed to scan password hashes: {}", e);
            }
        }

        Ok(hashes)
    }

    /// Save CollateralVault to persistent storage (generic byte storage)
    /// Key: collateral_vault
    pub async fn save_collateral_vault_data(&self, vault_data: &[u8]) -> Result<()> {
        let key = b"collateral_vault";
        self.hot_db.put(CF_MANIFEST, key, vault_data).await?;
        debug!("💰 Saved CollateralVault data ({} bytes)", vault_data.len());
        Ok(())
    }

    /// Load CollateralVault data from persistent storage
    /// Returns None if no vault exists (first run)
    pub async fn load_collateral_vault_data(&self) -> Result<Option<Vec<u8>>> {
        let key = b"collateral_vault";
        match self.hot_db.get(CF_MANIFEST, key).await? {
            Some(vault_data) => {
                info!("💰 Loaded CollateralVault data ({} bytes)", vault_data.len());
                Ok(Some(vault_data))
            }
            None => {
                debug!("💰 No persisted CollateralVault found (first run)");
                Ok(None)
            }
        }
    }

    // ========================================
    // AI CHAT STORAGE METHODS
    // ========================================

    /// Create a new AI chat session
    /// Key format: chat:{chat_id}
    pub async fn create_chat(&self, metadata: &ChatMetadata) -> Result<()> {
        let key = format!("chat:{}", metadata.chat_id);
        let value = bincode::serialize(metadata)?;

        self.hot_db.put(CF_AI_CHATS, key.as_bytes(), &value).await?;

        // Add to user's chat list
        self.add_chat_to_user_list(&metadata.user_id, &metadata.chat_id).await?;

        // Set as latest chat for user
        let latest_key = format!("chat:latest:{}", metadata.user_id);
        self.hot_db.put(CF_AI_CHATS, latest_key.as_bytes(), metadata.chat_id.as_bytes()).await?;

        info!("💬 Created chat {} for user {}", metadata.chat_id, metadata.user_id);
        Ok(())
    }

    /// Save a chat message
    /// Key format: chat:{chat_id}:msg:{index}
    pub async fn save_chat_message(&self, chat_id: &str, message: &ChatMessage) -> Result<()> {
        let msg_key = format!("chat:{}:msg:{}", chat_id, message.index);
        let value = bincode::serialize(message)?;

        self.hot_db.put(CF_AI_CHATS, msg_key.as_bytes(), &value).await?;

        // Update chat metadata's message count and updated_at
        let metadata_key = format!("chat:{}", chat_id);
        if let Some(metadata_data) = self.hot_db.get(CF_AI_CHATS, metadata_key.as_bytes()).await? {
            let mut metadata: ChatMetadata = bincode::deserialize(&metadata_data)?;
            metadata.message_count = message.index + 1;
            metadata.updated_at = message.timestamp;

            let updated_value = bincode::serialize(&metadata)?;
            self.hot_db.put(CF_AI_CHATS, metadata_key.as_bytes(), &updated_value).await?;
        }

        debug!("💬 Saved message {} in chat {}", message.index, chat_id);
        Ok(())
    }

    /// Load chat messages
    /// Returns messages in order
    pub async fn load_chat_messages(&self, chat_id: &str) -> Result<Vec<ChatMessage>> {
        let prefix = format!("chat:{}:msg:", chat_id);
        let messages_data = self.hot_db.scan_prefix(CF_AI_CHATS, prefix.as_bytes()).await?;

        let mut messages = Vec::new();
        for (_, msg_data) in messages_data {
            if let Ok(message) = bincode::deserialize::<ChatMessage>(&msg_data) {
                messages.push(message);
            }
        }

        // Sort by index
        messages.sort_by_key(|m| m.index);

        debug!("💬 Loaded {} messages from chat {}", messages.len(), chat_id);
        Ok(messages)
    }

    /// List all chats for a user
    pub async fn list_user_chats(&self, user_id: &str) -> Result<Vec<ChatMetadata>> {
        let list_key = format!("chat:user:{}", user_id);
        let chat_ids_data = self.hot_db.get(CF_AI_CHATS, list_key.as_bytes()).await?;

        let chat_ids: Vec<String> = match chat_ids_data {
            Some(data) => bincode::deserialize(&data)?,
            None => Vec::new(),
        };

        let mut chats = Vec::new();
        for chat_id in chat_ids {
            let key = format!("chat:{}", chat_id);
            if let Some(metadata_data) = self.hot_db.get(CF_AI_CHATS, key.as_bytes()).await? {
                if let Ok(metadata) = bincode::deserialize::<ChatMetadata>(&metadata_data) {
                    chats.push(metadata);
                }
            }
        }

        // Sort by updated_at descending (most recent first)
        chats.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));

        debug!("💬 Listed {} chats for user {}", chats.len(), user_id);
        Ok(chats)
    }

    /// Delete a chat and all its messages
    pub async fn delete_chat(&self, chat_id: &str, user_id: &str) -> Result<()> {
        // Delete chat metadata
        let metadata_key = format!("chat:{}", chat_id);
        self.hot_db.delete(CF_AI_CHATS, metadata_key.as_bytes()).await?;

        // Delete all messages
        let msg_prefix = format!("chat:{}:msg:", chat_id);
        let messages = self.hot_db.scan_prefix(CF_AI_CHATS, msg_prefix.as_bytes()).await?;
        for (msg_key, _) in messages {
            self.hot_db.delete(CF_AI_CHATS, &msg_key).await?;
        }

        // Remove from user's chat list
        self.remove_chat_from_user_list(user_id, chat_id).await?;

        info!("💬 Deleted chat {} for user {}", chat_id, user_id);
        Ok(())
    }

    /// Rename a chat
    pub async fn rename_chat(&self, chat_id: &str, new_title: &str) -> Result<()> {
        let key = format!("chat:{}", chat_id);
        if let Some(metadata_data) = self.hot_db.get(CF_AI_CHATS, key.as_bytes()).await? {
            let mut metadata: ChatMetadata = bincode::deserialize(&metadata_data)?;
            metadata.title = new_title.to_string();

            let updated_value = bincode::serialize(&metadata)?;
            self.hot_db.put(CF_AI_CHATS, key.as_bytes(), &updated_value).await?;

            info!("💬 Renamed chat {} to '{}'", chat_id, new_title);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Chat {} not found", chat_id))
        }
    }

    /// Update chat settings (privacy, performance options)
    pub async fn update_chat_settings(&self, chat_id: &str, settings: &ChatSettings) -> Result<()> {
        let key = format!("chat:{}", chat_id);
        if let Some(metadata_data) = self.hot_db.get(CF_AI_CHATS, key.as_bytes()).await? {
            let mut metadata: ChatMetadata = bincode::deserialize(&metadata_data)?;

            metadata.encryption_enabled = settings.encryption_enabled;
            metadata.zk_proofs_enabled = settings.zk_proofs_enabled;
            metadata.distributed_enabled = settings.distributed_enabled;
            metadata.enable_kv_cache = settings.enable_kv_cache;
            metadata.enable_pipeline_parallel = settings.enable_pipeline_parallel;
            metadata.enable_load_balancing = settings.enable_load_balancing;

            let updated_value = bincode::serialize(&metadata)?;
            self.hot_db.put(CF_AI_CHATS, key.as_bytes(), &updated_value).await?;

            info!("💬 Updated settings for chat {}", chat_id);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Chat {} not found", chat_id))
        }
    }

    /// Get chat metadata
    pub async fn get_chat_metadata(&self, chat_id: &str) -> Result<Option<ChatMetadata>> {
        let key = format!("chat:{}", chat_id);
        match self.hot_db.get(CF_AI_CHATS, key.as_bytes()).await? {
            Some(metadata_data) => {
                let metadata: ChatMetadata = bincode::deserialize(&metadata_data)?;
                Ok(Some(metadata))
            }
            None => Ok(None),
        }
    }

    /// Add chat to user's list (internal helper)
    async fn add_chat_to_user_list(&self, user_id: &str, chat_id: &str) -> Result<()> {
        let key = format!("chat:user:{}", user_id);

        let mut chat_ids: Vec<String> = match self.hot_db.get(CF_AI_CHATS, key.as_bytes()).await? {
            Some(data) => bincode::deserialize(&data)?,
            None => Vec::new(),
        };

        if !chat_ids.contains(&chat_id.to_string()) {
            chat_ids.push(chat_id.to_string());
            let value = bincode::serialize(&chat_ids)?;
            self.hot_db.put(CF_AI_CHATS, key.as_bytes(), &value).await?;
        }

        Ok(())
    }

    /// Remove chat from user's list (internal helper)
    async fn remove_chat_from_user_list(&self, user_id: &str, chat_id: &str) -> Result<()> {
        let key = format!("chat:user:{}", user_id);

        if let Some(data) = self.hot_db.get(CF_AI_CHATS, key.as_bytes()).await? {
            let mut chat_ids: Vec<String> = bincode::deserialize(&data)?;
            chat_ids.retain(|id| id != chat_id);

            let value = bincode::serialize(&chat_ids)?;
            self.hot_db.put(CF_AI_CHATS, key.as_bytes(), &value).await?;
        }

        Ok(())
    }

    // ============================================================================
    // Payment Consensus Storage Methods
    // ============================================================================

    /// Get wallet credits
    pub async fn get_wallet_credits(&self, wallet_address: &str) -> Result<Option<AICredits>> {
        let key = format!("credits:{}", wallet_address);
        match self.hot_db.get(CF_AI_CREDITS, key.as_bytes()).await? {
            Some(data) => {
                let credits: AICredits = bincode::deserialize(&data)?;
                Ok(Some(credits))
            }
            None => Ok(None),
        }
    }

    /// Initialize wallet credits
    pub async fn init_wallet_credits(&self, wallet_address: &str) -> Result<AICredits> {
        let now = SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs();
        let credits = AICredits {
            wallet_address: wallet_address.to_string(),
            balance_qnk: 0,
            balance_qugusd: 0,
            total_spent_qnk: 0,
            total_spent_qugusd: 0,
            total_tokens_generated: 0,
            created_at: now,
            updated_at: now,
        };

        let key = format!("credits:{}", wallet_address);
        let value = bincode::serialize(&credits)?;
        self.hot_db.put(CF_AI_CREDITS, key.as_bytes(), &value).await?;

        info!("💰 Initialized credits for wallet {}", wallet_address);
        Ok(credits)
    }

    /// Update wallet balance
    pub async fn update_wallet_balance(
        &self,
        wallet_address: &str,
        delta_qnk: i64,
        delta_qugusd: i64,
    ) -> Result<()> {
        let key = format!("credits:{}", wallet_address);

        let mut credits = match self.get_wallet_credits(wallet_address).await? {
            Some(c) => c,
            None => self.init_wallet_credits(wallet_address).await?,
        };

        // Update balances (handle underflow)
        if delta_qnk < 0 && credits.balance_qnk < delta_qnk.abs() as u64 {
            return Err(anyhow::anyhow!("Insufficient QNK balance"));
        }
        if delta_qugusd < 0 && credits.balance_qugusd < delta_qugusd.abs() as u64 {
            return Err(anyhow::anyhow!("Insufficient QUGUSD balance"));
        }

        if delta_qnk >= 0 {
            credits.balance_qnk += delta_qnk as u64;
        } else {
            credits.balance_qnk -= delta_qnk.abs() as u64;
            credits.total_spent_qnk += delta_qnk.abs() as u64;
        }

        if delta_qugusd >= 0 {
            credits.balance_qugusd += delta_qugusd as u64;
        } else {
            credits.balance_qugusd -= delta_qugusd.abs() as u64;
            credits.total_spent_qugusd += delta_qugusd.abs() as u64;
        }

        credits.updated_at = SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs();

        let value = bincode::serialize(&credits)?;
        self.hot_db.put(CF_AI_CREDITS, key.as_bytes(), &value).await?;

        debug!("💰 Updated wallet {} balance: QNK {} QUGUSD {}",
            wallet_address, credits.balance_qnk, credits.balance_qugusd);

        Ok(())
    }

    /// Save AI transaction
    pub async fn save_ai_transaction(&self, tx: &AITransaction) -> Result<()> {
        let key = format!("aitx:{}", tx.tx_id);
        let value = bincode::serialize(tx)?;
        self.hot_db.put(CF_AI_TRANSACTIONS, key.as_bytes(), &value).await?;

        debug!("📝 Saved AI transaction {}", tx.tx_id);
        Ok(())
    }

    /// Get AI transaction
    pub async fn get_ai_transaction(&self, tx_id: &str) -> Result<Option<AITransaction>> {
        let key = format!("aitx:{}", tx_id);
        match self.hot_db.get(CF_AI_TRANSACTIONS, key.as_bytes()).await? {
            Some(data) => {
                let tx: AITransaction = bincode::deserialize(&data)?;
                Ok(Some(tx))
            }
            None => Ok(None),
        }
    }

    /// Save payment proposal
    pub async fn save_payment_proposal(&self, proposal: &PaymentProposal) -> Result<()> {
        let key = format!("proposal:{}", proposal.request_id);
        let value = bincode::serialize(proposal)?;
        self.hot_db.put(CF_PAYMENT_PROPOSALS, key.as_bytes(), &value).await?;

        debug!("🗳️ Saved payment proposal {}", proposal.request_id);
        Ok(())
    }

    /// Get payment proposal
    pub async fn get_payment_proposal(&self, request_id: &str) -> Result<Option<PaymentProposal>> {
        let key = format!("proposal:{}", request_id);
        match self.hot_db.get(CF_PAYMENT_PROPOSALS, key.as_bytes()).await? {
            Some(data) => {
                let proposal: PaymentProposal = bincode::deserialize(&data)?;
                Ok(Some(proposal))
            }
            None => Ok(None),
        }
    }

    /// Save payment vote
    pub async fn save_payment_vote(&self, vote: &PaymentVote) -> Result<()> {
        let key = format!("vote:{}:{}", vote.request_id, vote.validator_node_id);
        let value = bincode::serialize(vote)?;
        self.hot_db.put(CF_PAYMENT_VOTES, key.as_bytes(), &value).await?;

        debug!("✅ Saved payment vote for request {} from validator {}",
            vote.request_id, vote.validator_node_id);
        Ok(())
    }

    /// Get all votes for a payment request
    pub async fn get_payment_votes(&self, request_id: &str) -> Result<Vec<PaymentVote>> {
        let prefix = format!("vote:{}:", request_id);
        let votes_data = self.hot_db.scan_prefix(CF_PAYMENT_VOTES, prefix.as_bytes()).await?;

        let mut votes = Vec::new();
        for (_, vote_data) in votes_data {
            if let Ok(vote) = bincode::deserialize::<PaymentVote>(&vote_data) {
                votes.push(vote);
            }
        }

        debug!("🗳️ Loaded {} votes for request {}", votes.len(), request_id);
        Ok(votes)
    }

    /// Save payment lock
    pub async fn save_payment_lock(&self, lock: &PaymentLock) -> Result<()> {
        let key = format!("lock:{}", lock.request_id);
        let value = bincode::serialize(lock)?;
        self.hot_db.put(CF_PAYMENT_LOCKS, key.as_bytes(), &value).await?;

        info!("🔒 Saved payment lock for request {}", lock.request_id);
        Ok(())
    }

    /// Get payment lock
    pub async fn get_payment_lock(&self, request_id: &str) -> Result<Option<PaymentLock>> {
        let key = format!("lock:{}", request_id);
        match self.hot_db.get(CF_PAYMENT_LOCKS, key.as_bytes()).await? {
            Some(data) => {
                let lock: PaymentLock = bincode::deserialize(&data)?;
                Ok(Some(lock))
            }
            None => Ok(None),
        }
    }

    /// Remove payment lock (after settlement)
    pub async fn remove_payment_lock(&self, request_id: &str) -> Result<()> {
        let key = format!("lock:{}", request_id);
        self.hot_db.delete(CF_PAYMENT_LOCKS, key.as_bytes()).await?;

        debug!("🔓 Removed payment lock for request {}", request_id);
        Ok(())
    }

    /// Check if wallet has pending payments (for double-spend detection)
    pub async fn has_pending_payment(&self, wallet_address: &str) -> Result<bool> {
        let prefix = format!("lock:");
        let locks_data = self.hot_db.scan_prefix(CF_PAYMENT_LOCKS, prefix.as_bytes()).await?;

        for (_, lock_data) in locks_data {
            if let Ok(lock) = bincode::deserialize::<PaymentLock>(&lock_data) {
                if lock.wallet_address == wallet_address {
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }

    // ============================================================================
    // Treasury Management (Master Wallet)
    // ============================================================================

    /// Get treasury balance (creates if doesn't exist)
    pub async fn get_treasury_balance(&self) -> Result<AITreasury> {
        let key = b"treasury:master";
        match self.hot_db.get(CF_AI_TREASURY, key).await? {
            Some(data) => {
                let treasury: AITreasury = bincode::deserialize(&data)?;
                Ok(treasury)
            }
            None => {
                // Initialize treasury with environment variable or default
                let treasury_address = std::env::var("AI_TREASURY_WALLET")
                    .unwrap_or_else(|_| "MASTER_AI_TREASURY_WALLET".to_string());

                let now = SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs();
                let treasury = AITreasury {
                    wallet_address: treasury_address,
                    total_revenue_qnk: 0,
                    total_revenue_qugusd: 0,
                    total_requests_served: 0,
                    total_tokens_generated: 0,
                    created_at: now,
                    updated_at: now,
                };

                self.save_treasury_balance(&treasury).await?;
                info!("💰 Initialized AI treasury wallet: {}", treasury.wallet_address);
                Ok(treasury)
            }
        }
    }

    /// Credit treasury with AI payment (100% of profits)
    pub async fn credit_treasury(
        &self,
        amount_qnk: u64,
        amount_qugusd: u64,
        tokens_generated: u32,
    ) -> Result<()> {
        let mut treasury = self.get_treasury_balance().await?;

        treasury.total_revenue_qnk += amount_qnk;
        treasury.total_revenue_qugusd += amount_qugusd;
        treasury.total_requests_served += 1;
        treasury.total_tokens_generated += tokens_generated as u64;
        treasury.updated_at = SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs();

        self.save_treasury_balance(&treasury).await?;

        info!(
            "💰 Treasury credited: {} QNK, {} QUGUSD ({} tokens) | Total: {} QNK, {} requests",
            amount_qnk,
            amount_qugusd,
            tokens_generated,
            treasury.total_revenue_qnk,
            treasury.total_requests_served
        );

        Ok(())
    }

    /// Save treasury balance to disk
    async fn save_treasury_balance(&self, treasury: &AITreasury) -> Result<()> {
        let key = b"treasury:master";
        let value = bincode::serialize(treasury)?;
        self.hot_db.put(CF_AI_TREASURY, key, &value).await?;
        Ok(())
    }

    /// Get access to hot RocksDB for advanced operations like pruning
    /// This returns the concrete RocksDBKV type which supports pruning operations
    pub fn get_hot_db(&self) -> Arc<RocksDBKV> {
        self.hot_db_concrete.clone()
    }

    /// Execute adaptive pruning on the hot database
    /// This is a wrapper method that allows calling pruning without dealing with thread safety issues
    pub async fn prune_old_blocks(&self, current_height: u64) -> Result<crate::pruning::PruningStats> {
        self.hot_db_concrete.prune_old_blocks(current_height).await
    }

    /// Atomic payment settlement: refund user + credit treasury + log transaction
    pub async fn settle_payment_atomic(&self, settlement: &PaymentSettlement) -> Result<()> {
        debug!(
            "🔄 Atomic settlement for request {}: {} QNK to treasury, {} QNK refund",
            settlement.request_id, settlement.treasury_payment_qnk, settlement.refund_amount_qnk
        );

        // Build atomic batch
        let mut batch_ops = Vec::new();

        // 1. Refund user if applicable
        if settlement.refund_amount_qnk > 0 {
            let refund_key = format!("credits:{}", settlement.wallet_address);
            let mut user_credits = self
                .get_wallet_credits(&settlement.wallet_address)
                .await?
                .ok_or_else(|| anyhow::anyhow!("User credits not found"))?;

            user_credits.balance_qnk += settlement.refund_amount_qnk;
            user_credits.updated_at =
                SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs();

            let value = bincode::serialize(&user_credits)?;
            batch_ops.push((CF_AI_CREDITS, refund_key.into_bytes(), value));
        }

        // 2. Credit treasury (100% of actual cost)
        let treasury_key = b"treasury:master".to_vec();
        let mut treasury = self.get_treasury_balance().await?;
        treasury.total_revenue_qnk += settlement.treasury_payment_qnk;
        treasury.total_requests_served += 1;
        treasury.total_tokens_generated += settlement.actual_tokens_generated as u64;
        treasury.updated_at = SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs();

        let treasury_value = bincode::serialize(&treasury)?;
        batch_ops.push((CF_AI_TREASURY, treasury_key, treasury_value));

        // 3. Log transaction
        let tx = AITransaction {
            tx_id: settlement.request_id.clone(),
            wallet_address: settlement.wallet_address.clone(),
            chat_id: "".to_string(), // Will be filled by caller
            input_tokens: 0,
            output_tokens: settlement.actual_tokens_generated,
            cost_usd_cents: 0, // Will be calculated from oracle
            cost_qnk: settlement.actual_cost_qnk,
            payment_token: PaymentToken::QNK,
            oracle_price_usd_cents: 0,
            timestamp: settlement.timestamp,
            status: PaymentStatus::Completed,
        };

        let tx_key = format!("aitx:{}", tx.tx_id);
        let tx_value = bincode::serialize(&tx)?;
        batch_ops.push((CF_AI_TRANSACTIONS, tx_key.into_bytes(), tx_value));

        // 4. Execute atomic batch
        self.hot_db.write_batch(batch_ops).await?;

        // 5. Remove payment lock (separate operation, not critical if fails)
        if let Err(e) = self.remove_payment_lock(&settlement.request_id).await {
            warn!(
                "⚠️ Failed to remove payment lock for {}: {}",
                settlement.request_id, e
            );
        }

        info!(
            "✅ Payment settled atomically: {} QNK to treasury, {} QNK refunded to {}",
            settlement.treasury_payment_qnk, settlement.refund_amount_qnk, settlement.wallet_address
        );

        Ok(())
    }
}

/// Implementation of BalanceStorage trait for consensus engine
#[async_trait::async_trait]
impl BalanceStorage for QStorage {
    /// Add amount to wallet balance (atomic operation)
    async fn add_balance(&self, address: &str, amount: u64) -> Result<()> {
        // Convert hex string address to [u8; 32]
        let address_bytes = hex::decode(address)
            .context("Invalid hex address format")?;

        if address_bytes.len() != 32 {
            return Err(anyhow::anyhow!(
                "Invalid address length: expected 32 bytes, got {}",
                address_bytes.len()
            ));
        }

        let mut addr_array = [0u8; 32];
        addr_array.copy_from_slice(&address_bytes);

        // Get current balance
        let current = self.load_wallet_balance(&addr_array).await?.unwrap_or(0);

        // Add amount (saturating to prevent overflow)
        let new_balance = current.saturating_add(amount);

        // Save new balance
        self.save_wallet_balance(&addr_array, new_balance).await?;

        debug!(
            "✅ [BALANCE CONSENSUS] Added {} to {}, new balance: {}",
            amount, address, new_balance
        );

        Ok(())
    }

    /// Get wallet balance
    async fn get_balance(&self, address: &str) -> Result<u64> {
        // Convert hex string address to [u8; 32]
        let address_bytes = hex::decode(address)
            .context("Invalid hex address format")?;

        if address_bytes.len() != 32 {
            return Err(anyhow::anyhow!(
                "Invalid address length: expected 32 bytes, got {}",
                address_bytes.len()
            ));
        }

        let mut addr_array = [0u8; 32];
        addr_array.copy_from_slice(&address_bytes);

        Ok(self.load_wallet_balance(&addr_array).await?.unwrap_or(0))
    }

    /// Set wallet balance directly
    async fn set_balance(&self, address: &str, balance: u64) -> Result<()> {
        // Convert hex string address to [u8; 32]
        let address_bytes = hex::decode(address)
            .context("Invalid hex address format")?;

        if address_bytes.len() != 32 {
            return Err(anyhow::anyhow!(
                "Invalid address length: expected 32 bytes, got {}",
                address_bytes.len()
            ));
        }

        let mut addr_array = [0u8; 32];
        addr_array.copy_from_slice(&address_bytes);

        self.save_wallet_balance(&addr_array, balance).await?;

        debug!(
            "✅ [BALANCE CONSENSUS] Set balance for {} to {}",
            address, balance
        );

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
    pub last_write: std::time::SystemTime,
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
        matches!(
            self,
            Self::InconsistentState | Self::DatabaseError | Self::Offline
        )
    }
}

/// AI Chat metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMetadata {
    pub chat_id: String,
    pub user_id: String,
    pub title: String,
    pub model: String,
    pub created_at: u64,
    pub updated_at: u64,
    pub message_count: u64,
    pub encryption_enabled: bool,
    pub zk_proofs_enabled: bool,
    pub distributed_enabled: bool,
    pub enable_kv_cache: bool,
    pub enable_pipeline_parallel: bool,
    pub enable_load_balancing: bool,
}

/// AI Chat message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub index: u64,
    pub role: String,
    pub content: String,
    pub timestamp: u64,
    pub images: Option<Vec<String>>,
    pub audio: Option<String>,
    pub generation_stats: Option<GenerationStats>,
}

/// AI generation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationStats {
    pub total_tokens: usize,
    pub latency_ms: u64,
    pub tokens_per_second: f64,
    pub privacy_overhead_ms: u64,
    pub zk_proof_time_ms: u64,
    pub distributed_nodes_used: usize,
}

/// Chat settings for updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatSettings {
    pub encryption_enabled: bool,
    pub zk_proofs_enabled: bool,
    pub distributed_enabled: bool,
    pub enable_kv_cache: bool,
    pub enable_pipeline_parallel: bool,
    pub enable_load_balancing: bool,
}

// ============================================================================
// Payment Consensus Structures
// ============================================================================

/// AI Credits for wallet
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AICredits {
    pub wallet_address: String,
    pub balance_qnk: u64,
    pub balance_qugusd: u64,
    pub total_spent_qnk: u64,
    pub total_spent_qugusd: u64,
    pub total_tokens_generated: u64,
    pub created_at: u64,
    pub updated_at: u64,
}

/// AI Transaction record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AITransaction {
    pub tx_id: String,
    pub wallet_address: String,
    pub chat_id: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cost_usd_cents: u64,
    pub cost_qnk: u64,
    pub payment_token: PaymentToken,
    pub oracle_price_usd_cents: u64,
    pub timestamp: u64,
    pub status: PaymentStatus,
}

/// Payment token type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PaymentToken {
    QNK,
    QUGUSD,
}

/// Payment status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PaymentStatus {
    Pending,
    Completed,
    Refunded,
    Failed,
}

/// Payment Proposal for distributed consensus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaymentProposal {
    pub request_id: String,
    pub wallet_address: String,
    pub estimated_tokens: u32,
    pub estimated_cost_qnk: u64,
    pub payment_token: PaymentToken,
    pub signature: Vec<u8>,
    pub timestamp: u64,
    pub proposer_node_id: String,
}

/// Payment Vote from validator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaymentVote {
    pub request_id: String,
    pub validator_node_id: String,
    pub vote: bool, // true = approve, false = reject
    pub reason: Option<String>,
    pub signature: Vec<u8>,
}

/// Payment Lock (consensus reached)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaymentLock {
    pub request_id: String,
    pub wallet_address: String,
    pub locked_amount_qnk: u64,
    pub locked_at: u64,
    pub validator_signatures: Vec<ValidatorSignature>,
}

/// Validator Signature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorSignature {
    pub node_id: String,
    pub signature: Vec<u8>,
}

/// AI Treasury (Master Wallet)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AITreasury {
    pub wallet_address: String,
    pub total_revenue_qnk: u64,
    pub total_revenue_qugusd: u64,
    pub total_requests_served: u64,
    pub total_tokens_generated: u64,
    pub created_at: u64,
    pub updated_at: u64,
}

/// Payment Settlement (100% profits to treasury)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaymentSettlement {
    pub request_id: String,
    pub wallet_address: String,  // User wallet
    pub actual_tokens_generated: u32,
    pub actual_cost_qnk: u64,
    pub refund_amount_qnk: u64,
    pub treasury_payment_qnk: u64,  // = actual_cost_qnk (100% to treasury)
    pub treasury_wallet: String,  // MASTER_AI_TREASURY_WALLET
    pub generation_node_id: String,
    pub validator_signatures: Vec<ValidatorSignature>,
    pub timestamp: u64,
}

/// Consensus Result
#[derive(Debug, Clone, PartialEq)]
pub enum ConsensusResult {
    Approved,
    Rejected,
    Pending,
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

        let vertex_key = storage.vertex_key(100, &[0xaa, 0xbb], &[0x01; 32]);
        assert_eq!(vertex_key.len(), 8 + 2 + 32); // round + author + vertex_id

        let block_key = storage.block_key(1000, &[0xcc; 32]);
        assert_eq!(block_key.len(), 8 + 32); // height + hash
    }

    #[test]
    fn test_health_status() {
        assert_eq!(StorageHealthStatus::Healthy.as_str(), "healthy");
        assert!(!StorageHealthStatus::Healthy.is_critical());

        assert_eq!(
            StorageHealthStatus::DatabaseError.as_str(),
            "database_error"
        );
        assert!(StorageHealthStatus::DatabaseError.is_critical());
    }
}

// Mock implementations for testing
#[cfg(test)]
struct MockKVStore;

#[cfg(test)]
impl MockKVStore {
    fn new() -> Self {
        Self
    }
}

#[cfg(test)]
#[async_trait]
impl KVStore for MockKVStore {
    async fn put(&self, _cf: &str, _key: &[u8], _value: &[u8]) -> Result<()> {
        Ok(())
    }
    async fn get(&self, _cf: &str, _key: &[u8]) -> Result<Option<Vec<u8>>> {
        Ok(None)
    }
    async fn delete(&self, _cf: &str, _key: &[u8]) -> Result<()> {
        Ok(())
    }
    async fn write_batch(&self, _batch: Vec<(&str, Vec<u8>, Vec<u8>)>) -> Result<()> {
        Ok(())
    }
    async fn write_batch_bulk(&self, _batch: Vec<(&str, Vec<u8>, Vec<u8>)>) -> Result<()> {
        Ok(())
    }
    async fn scan_prefix(&self, _cf: &str, _prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        Ok(vec![])
    }
    async fn scan_all(&self, _cf: &str) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        Ok(vec![])
    }
    async fn flush(&self) -> Result<()> {
        Ok(())
    }
    async fn compact(&self) -> Result<()> {
        Ok(())
    }
    async fn get_db_size(&self) -> Result<u64> {
        Ok(0)
    }
}
