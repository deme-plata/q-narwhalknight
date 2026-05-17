/// 🚀 Project APOLLO Phase 3: HOHMANN - Staged Sync (ORBITAL INSERTION)
///
/// Header-first sync strategy for 10-50x faster initial sync:
/// - Stage 1: Headers only (2KB/block) - Chain skeleton verification
/// - Stage 2: Transaction hashes (for Merkle verification)
/// - Stage 3: Full transactions (on-demand or batch)
/// - Stage 4: State reconstruction (parallel balance computation)
///
/// Aerospace analogy:
/// - ORBITAL INSERTION: Establish basic trajectory (headers)
/// - COAST PHASE: Drift towards target (tx hashes)
/// - FINAL APPROACH: Precision adjustments (full txs)
/// - LANDING: Commit to surface (state)
///
/// Performance target: Initial sync from hours to minutes

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, RwLock, Semaphore};
use tracing::{debug, error, info, warn};

/// Sync stage progression (HOHMANN orbital phases)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SyncStage {
    /// Stage 1: Headers only - Chain skeleton
    /// ~2KB per block header, fastest sync phase
    /// Validates: height continuity, timestamps, DAG structure
    OrbitalInsertion,

    /// Stage 2: Transaction hashes for Merkle verification
    /// ~32 bytes per tx hash, verifies tx inclusion
    CoastPhase,

    /// Stage 3: Full transactions - On-demand or batch
    /// ~100-500 bytes per tx, full transaction data
    FinalApproach,

    /// Stage 4: State reconstruction - Balance computation
    /// Parallel processing of all transactions to compute state
    Landing,

    /// Sync complete - Normal operation
    Orbiting,
}

impl SyncStage {
    pub fn next(&self) -> Option<SyncStage> {
        match self {
            SyncStage::OrbitalInsertion => Some(SyncStage::CoastPhase),
            SyncStage::CoastPhase => Some(SyncStage::FinalApproach),
            SyncStage::FinalApproach => Some(SyncStage::Landing),
            SyncStage::Landing => Some(SyncStage::Orbiting),
            SyncStage::Orbiting => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            SyncStage::OrbitalInsertion => "ORBITAL_INSERTION",
            SyncStage::CoastPhase => "COAST_PHASE",
            SyncStage::FinalApproach => "FINAL_APPROACH",
            SyncStage::Landing => "LANDING",
            SyncStage::Orbiting => "ORBITING",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            SyncStage::OrbitalInsertion => "Syncing block headers (chain skeleton)",
            SyncStage::CoastPhase => "Verifying transaction Merkle proofs",
            SyncStage::FinalApproach => "Downloading full transaction data",
            SyncStage::Landing => "Reconstructing account states",
            SyncStage::Orbiting => "Fully synced, normal operation",
        }
    }
}

/// Minimal block header for fast sync (Stage 1: ORBITAL INSERTION)
/// ~200-500 bytes vs ~50-100KB for full block
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HeaderOnly {
    /// Block height
    pub height: u64,
    /// Block hash (32 bytes)
    pub hash: [u8; 32],
    /// Previous block hash (32 bytes)
    pub prev_hash: [u8; 32],
    /// Timestamp (8 bytes)
    pub timestamp: u64,
    /// Merkle root of transactions (32 bytes)
    pub tx_merkle_root: [u8; 32],
    /// State root after this block (32 bytes)
    pub state_root: [u8; 32],
    /// Number of transactions in block
    pub tx_count: u32,
    /// Total block size (for bandwidth estimation)
    pub block_size: u32,
    /// DAG parent hashes (for DAG-Knight consensus)
    pub dag_parents: Vec<[u8; 32]>,
}

impl HeaderOnly {
    /// Estimated size in bytes for bandwidth calculations
    pub fn estimated_size(&self) -> usize {
        // Base: 32*5 (hashes) + 8 (timestamp) + 4*2 (counts) + 8 (height)
        // + parents: 32 * parent_count
        200 + (self.dag_parents.len() * 32)
    }
}

/// Staged sync configuration
#[derive(Clone, Debug)]
pub struct StagedSyncConfig {
    /// Maximum headers to request per batch
    pub header_batch_size: usize,
    /// Maximum tx hashes per batch
    pub tx_hash_batch_size: usize,
    /// Maximum full transactions per batch
    pub tx_batch_size: usize,
    /// Parallel state reconstruction workers
    pub state_workers: usize,
    /// Skip state reconstruction (for light clients)
    pub light_mode: bool,
    /// Trust checkpoints for faster sync
    pub trust_checkpoints: bool,
    /// Maximum concurrent header requests
    pub header_parallelism: usize,
}

impl Default for StagedSyncConfig {
    fn default() -> Self {
        Self {
            header_batch_size: 10_000,  // 10k headers = ~2-5MB per batch
            tx_hash_batch_size: 50_000, // 50k hashes = ~1.6MB per batch
            tx_batch_size: 1_000,       // 1k txs = ~500KB per batch
            state_workers: std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4),
            light_mode: false,
            trust_checkpoints: true,
            header_parallelism: 32,
        }
    }
}

/// Staged sync progress metrics
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct StagedSyncMetrics {
    /// Current sync stage
    pub stage: String,
    /// Headers synced
    pub headers_synced: u64,
    /// Target header height
    pub headers_target: u64,
    /// Transaction hashes verified
    pub tx_hashes_verified: u64,
    /// Full transactions downloaded
    pub txs_downloaded: u64,
    /// Accounts reconstructed
    pub accounts_processed: u64,
    /// Total bytes downloaded
    pub bytes_downloaded: u64,
    /// Time spent in current stage
    pub stage_duration_ms: u64,
    /// Estimated time remaining (ms)
    pub eta_ms: u64,
    /// Headers per second
    pub headers_per_second: f64,
    /// Transactions per second
    pub txs_per_second: f64,
}

/// Main staged sync orchestrator (HOHMANN mission control)
pub struct StagedSyncManager {
    /// Current sync stage
    stage: SyncStage,
    /// Configuration
    config: StagedSyncConfig,
    /// Headers synced so far (height -> header)
    headers: HashMap<u64, HeaderOnly>,
    /// Highest header height we have
    headers_height: u64,
    /// Target network height
    target_height: u64,
    /// Transaction hashes verified (height -> verified)
    tx_hashes_verified: HashMap<u64, bool>,
    /// Full blocks downloaded (height -> downloaded)
    blocks_downloaded: HashMap<u64, bool>,
    /// State reconstruction progress
    state_progress: u64,
    /// Metrics
    metrics: Arc<RwLock<StagedSyncMetrics>>,
    /// Stage start time
    stage_start: Instant,
    /// Parallelism limiter for header downloads
    header_semaphore: Arc<Semaphore>,
}

impl StagedSyncManager {
    /// Create new staged sync manager
    pub fn new(config: StagedSyncConfig, target_height: u64) -> Self {
        let header_parallelism = config.header_parallelism;
        Self {
            stage: SyncStage::OrbitalInsertion,
            config,
            headers: HashMap::new(),
            headers_height: 0,
            target_height,
            tx_hashes_verified: HashMap::new(),
            blocks_downloaded: HashMap::new(),
            state_progress: 0,
            metrics: Arc::new(RwLock::new(StagedSyncMetrics::default())),
            stage_start: Instant::now(),
            header_semaphore: Arc::new(Semaphore::new(header_parallelism)),
        }
    }

    /// Get current sync stage
    pub fn current_stage(&self) -> SyncStage {
        self.stage
    }

    /// Get sync progress as percentage (0.0 - 100.0)
    pub fn progress_percent(&self) -> f64 {
        if self.target_height == 0 {
            return 100.0;
        }

        let stage_weight = match self.stage {
            SyncStage::OrbitalInsertion => 0.0,
            SyncStage::CoastPhase => 25.0,
            SyncStage::FinalApproach => 50.0,
            SyncStage::Landing => 75.0,
            SyncStage::Orbiting => 100.0,
        };

        let stage_progress = match self.stage {
            SyncStage::OrbitalInsertion => {
                (self.headers_height as f64 / self.target_height as f64) * 25.0
            }
            SyncStage::CoastPhase => {
                let verified = self.tx_hashes_verified.len() as f64;
                (verified / self.target_height as f64) * 25.0
            }
            SyncStage::FinalApproach => {
                let downloaded = self.blocks_downloaded.len() as f64;
                (downloaded / self.target_height as f64) * 25.0
            }
            SyncStage::Landing => {
                (self.state_progress as f64 / self.target_height as f64) * 25.0
            }
            SyncStage::Orbiting => 0.0,
        };

        stage_weight + stage_progress
    }

    /// Check if sync is complete
    pub fn is_complete(&self) -> bool {
        self.stage == SyncStage::Orbiting
    }

    /// Add headers from peer response (Stage 1)
    pub async fn add_headers(&mut self, headers: Vec<HeaderOnly>) -> Result<()> {
        if self.stage != SyncStage::OrbitalInsertion {
            bail!("Cannot add headers in stage {:?}", self.stage);
        }

        let count = headers.len();
        let start = Instant::now();

        for header in headers {
            // Verify header chain continuity
            if header.height > 1 {
                if let Some(prev) = self.headers.get(&(header.height - 1)) {
                    if header.prev_hash != prev.hash {
                        warn!(
                            "Header {} prev_hash mismatch! Expected {:?}, got {:?}",
                            header.height,
                            hex::encode(prev.hash),
                            hex::encode(header.prev_hash)
                        );
                        continue; // Skip invalid header
                    }
                }
            }

            // Store header and update height
            let height = header.height;
            self.headers.insert(height, header);
            if height > self.headers_height {
                self.headers_height = height;
            }
        }

        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.headers_synced = self.headers_height;
            metrics.headers_target = self.target_height;
            let elapsed = self.stage_start.elapsed().as_secs_f64();
            if elapsed > 0.0 {
                metrics.headers_per_second = self.headers_height as f64 / elapsed;
            }
        }

        debug!(
            "🛰️ [ORBITAL INSERTION] Added {} headers in {:?}, now at height {} / {}",
            count,
            start.elapsed(),
            self.headers_height,
            self.target_height
        );

        // Check if we should transition to next stage
        if self.headers_height >= self.target_height {
            self.transition_to_next_stage().await?;
        }

        Ok(())
    }

    /// Mark transaction hashes as verified (Stage 2)
    pub async fn verify_tx_hashes(&mut self, height: u64, verified: bool) -> Result<()> {
        if self.stage != SyncStage::CoastPhase {
            bail!("Cannot verify tx hashes in stage {:?}", self.stage);
        }

        self.tx_hashes_verified.insert(height, verified);

        // Check if all blocks have verified tx hashes
        let verified_count = self.tx_hashes_verified.values().filter(|v| **v).count() as u64;
        if verified_count >= self.headers_height {
            self.transition_to_next_stage().await?;
        }

        Ok(())
    }

    /// Mark full block as downloaded (Stage 3)
    pub async fn mark_block_downloaded(&mut self, height: u64) -> Result<()> {
        if self.stage != SyncStage::FinalApproach {
            bail!("Cannot mark blocks in stage {:?}", self.stage);
        }

        self.blocks_downloaded.insert(height, true);

        // Check if all blocks are downloaded
        let downloaded_count = self.blocks_downloaded.len() as u64;
        if downloaded_count >= self.headers_height {
            self.transition_to_next_stage().await?;
        }

        Ok(())
    }

    /// Update state reconstruction progress (Stage 4)
    pub async fn update_state_progress(&mut self, processed_height: u64) -> Result<()> {
        if self.stage != SyncStage::Landing {
            bail!("Cannot update state in stage {:?}", self.stage);
        }

        self.state_progress = processed_height;

        // Check if state reconstruction is complete
        if self.state_progress >= self.headers_height {
            self.transition_to_next_stage().await?;
        }

        Ok(())
    }

    /// Transition to next sync stage
    async fn transition_to_next_stage(&mut self) -> Result<()> {
        let old_stage = self.stage;
        let stage_duration = self.stage_start.elapsed();

        if let Some(next) = old_stage.next() {
            info!(
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            );
            info!(
                "🚀 [HOHMANN] Stage transition: {} → {}",
                old_stage.name(),
                next.name()
            );
            info!("   Previous stage completed in {:?}", stage_duration);
            info!("   {} → {}", old_stage.description(), next.description());
            info!(
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            );

            self.stage = next;
            self.stage_start = Instant::now();

            // Update metrics
            {
                let mut metrics = self.metrics.write().await;
                metrics.stage = next.name().to_string();
                metrics.stage_duration_ms = 0;
            }
        }

        Ok(())
    }

    /// Get current metrics
    pub async fn get_metrics(&self) -> StagedSyncMetrics {
        let metrics = self.metrics.read().await;
        let mut result = metrics.clone();
        result.stage_duration_ms = self.stage_start.elapsed().as_millis() as u64;
        result
    }

    /// Get headers in height range for peer serving
    pub fn get_headers(&self, start_height: u64, count: usize) -> Vec<&HeaderOnly> {
        (start_height..start_height + count as u64)
            .filter_map(|h| self.headers.get(&h))
            .collect()
    }

    /// Check if we have header at height
    pub fn has_header(&self, height: u64) -> bool {
        self.headers.contains_key(&height)
    }

    /// Get highest header height
    pub fn highest_header(&self) -> u64 {
        self.headers_height
    }

    /// Get header at height
    pub fn get_header(&self, height: u64) -> Option<&HeaderOnly> {
        self.headers.get(&height)
    }

    /// Generate header request ranges for parallel download
    /// Returns Vec of (start_height, count) tuples
    pub fn get_header_request_ranges(&self) -> Vec<(u64, usize)> {
        let mut ranges = Vec::new();
        let batch_size = self.config.header_batch_size;

        let mut current = self.headers_height + 1;
        while current <= self.target_height {
            let remaining = (self.target_height - current + 1) as usize;
            let count = remaining.min(batch_size);
            ranges.push((current, count));
            current += count as u64;
        }

        // Limit to parallelism level
        ranges.truncate(self.config.header_parallelism);
        ranges
    }
}

/// Header-only sync protocol messages
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum StagedSyncMessage {
    /// Request headers from start_height, up to count
    HeaderRequest {
        start_height: u64,
        count: usize,
    },
    /// Response with headers
    HeaderResponse {
        headers: Vec<HeaderOnly>,
    },
    /// Request transaction Merkle proofs for heights
    TxHashRequest {
        heights: Vec<u64>,
    },
    /// Response with tx Merkle proofs
    TxHashResponse {
        proofs: HashMap<u64, Vec<[u8; 32]>>,
    },
    /// Request full blocks by height
    BlockRequest {
        heights: Vec<u64>,
    },
    /// Request current sync stage from peer
    StageQuery,
    /// Response with peer's current stage
    StageResponse {
        stage: SyncStage,
        headers_height: u64,
        target_height: u64,
    },
}

/// Trait for staged sync peer communication
#[async_trait::async_trait]
pub trait StagedSyncPeer: Send + Sync {
    /// Request headers from peer
    async fn request_headers(&self, start: u64, count: usize) -> Result<Vec<HeaderOnly>>;

    /// Request tx hash proofs from peer
    async fn request_tx_proofs(&self, heights: &[u64]) -> Result<HashMap<u64, Vec<[u8; 32]>>>;

    /// Request full blocks from peer
    async fn request_blocks(&self, heights: &[u64]) -> Result<Vec<Vec<u8>>>;

    /// Get peer's sync stage
    async fn get_peer_stage(&self) -> Result<(SyncStage, u64, u64)>;
}

/// Light client mode - headers only, no state
pub struct LightSyncManager {
    /// Headers (chain skeleton)
    pub headers: HashMap<u64, HeaderOnly>,
    /// Highest verified header
    pub verified_height: u64,
    /// Checkpoint heights for fast verification
    pub checkpoints: Vec<u64>,
}

impl LightSyncManager {
    pub fn new() -> Self {
        Self {
            headers: HashMap::new(),
            verified_height: 0,
            checkpoints: Vec::new(),
        }
    }

    /// Verify a transaction exists in block using Merkle proof
    pub fn verify_tx_inclusion(
        &self,
        height: u64,
        tx_hash: [u8; 32],
        merkle_proof: &[[u8; 32]],
    ) -> Result<bool> {
        let header = self
            .headers
            .get(&height)
            .context("Header not found for height")?;

        // Verify Merkle proof against header's tx_merkle_root
        let computed_root = compute_merkle_root(tx_hash, merkle_proof);
        Ok(computed_root == header.tx_merkle_root)
    }

    /// Verify state root using checkpoint
    pub fn verify_state_at_checkpoint(&self, height: u64, state_root: [u8; 32]) -> Result<bool> {
        if !self.checkpoints.contains(&height) {
            bail!("Height {} is not a checkpoint", height);
        }

        let header = self
            .headers
            .get(&height)
            .context("Checkpoint header not found")?;
        Ok(header.state_root == state_root)
    }
}

impl Default for LightSyncManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute Merkle root from leaf and proof
fn compute_merkle_root(leaf: [u8; 32], proof: &[[u8; 32]]) -> [u8; 32] {
    let mut current = leaf;
    for sibling in proof {
        // Sort to ensure consistent ordering
        let (left, right) = if current < *sibling {
            (current, *sibling)
        } else {
            (*sibling, current)
        };

        // Hash left || right
        use sha3::{Digest, Sha3_256};
        let mut hasher = Sha3_256::new();
        hasher.update(left);
        hasher.update(right);
        let result = hasher.finalize();
        current.copy_from_slice(&result);
    }
    current
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sync_stage_progression() {
        assert_eq!(
            SyncStage::OrbitalInsertion.next(),
            Some(SyncStage::CoastPhase)
        );
        assert_eq!(SyncStage::CoastPhase.next(), Some(SyncStage::FinalApproach));
        assert_eq!(SyncStage::FinalApproach.next(), Some(SyncStage::Landing));
        assert_eq!(SyncStage::Landing.next(), Some(SyncStage::Orbiting));
        assert_eq!(SyncStage::Orbiting.next(), None);
    }

    #[test]
    fn test_header_only_size() {
        let header = HeaderOnly {
            height: 100,
            hash: [0u8; 32],
            prev_hash: [0u8; 32],
            timestamp: 1234567890,
            tx_merkle_root: [0u8; 32],
            state_root: [0u8; 32],
            tx_count: 100,
            block_size: 50000,
            dag_parents: vec![[0u8; 32], [0u8; 32]],
        };

        // Base 200 + 2*32 = 264 bytes
        assert_eq!(header.estimated_size(), 264);
    }

    #[tokio::test]
    async fn test_staged_sync_manager_creation() {
        let config = StagedSyncConfig::default();
        let manager = StagedSyncManager::new(config, 100_000);

        assert_eq!(manager.current_stage(), SyncStage::OrbitalInsertion);
        assert_eq!(manager.progress_percent(), 0.0);
        assert!(!manager.is_complete());
    }

    #[test]
    fn test_merkle_proof_verification() {
        let leaf = [1u8; 32];
        let sibling = [2u8; 32];
        let proof = vec![sibling];

        let root = compute_merkle_root(leaf, &proof);
        assert_ne!(root, leaf);
        assert_ne!(root, sibling);
    }
}
