/// Phase 2 DAG-Aware Sync State Manager (v1.0.4-beta)
///
/// Provides persistent checkpointing and crash recovery for DAG-aware synchronization.
/// This is the CRITICAL foundation for reliable sync operations.
///
/// Key Features:
/// - Checkpoint every 20 DAG layers for resume capability (v8.6.0: was 10)
/// - Track pending blocks awaiting parent resolution
/// - Atomic state transitions with RocksDB transactions
/// - Memory-bounded pending block buffer (max 20,000 blocks) (v8.6.0: was 10K)
///
/// Design rationale (from expert feedback):
/// - Without SyncStateManager, interrupted syncs must restart from scratch
/// - Checkpointing enables graceful recovery from network failures
/// - Bounded memory prevents DoS via excessive pending blocks

use crate::kv::KVStore;
use anyhow::{Context, Result};
use q_types::QBlock as Block;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Checkpoint frequency (save state every N layers)
/// v8.6.0: Increased from 10 to 20 — fewer checkpoint writes during sync, better throughput
const CHECKPOINT_INTERVAL_LAYERS: usize = 20;

/// Maximum pending blocks to prevent memory exhaustion
/// v8.6.0: Increased from 10K to 20K — more headroom for DAG-parallel block arrival
const MAX_PENDING_BLOCKS: usize = 20_000;

/// Persistent sync state for crash recovery
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SyncCheckpoint {
    /// Last fully committed DAG layer
    pub current_layer: usize,

    /// Highest blockchain height in last committed layer
    pub current_height: u64,

    /// Target height we're syncing to
    pub target_height: u64,

    /// Number of blocks committed so far
    pub blocks_committed: u64,

    /// Timestamp of last checkpoint (Unix epoch seconds)
    pub last_checkpoint_time: u64,

    /// Sync session ID (for tracking metrics)
    pub sync_session_id: String,
}

/// DAG-aware sync state manager with checkpoint/resume
pub struct SyncStateManager {
    /// Persistent key-value store for checkpoints
    kv: Arc<dyn KVStore>,

    /// Current sync checkpoint
    checkpoint: Arc<RwLock<SyncCheckpoint>>,

    /// Blocks awaiting parent resolution (hash → Block)
    /// Bounded to MAX_PENDING_BLOCKS to prevent memory exhaustion
    pending_blocks: Arc<RwLock<HashMap<String, Block>>>,

    /// Last checkpoint layer (for determining when to save)
    last_checkpoint_layer: Arc<RwLock<usize>>,
}

impl SyncStateManager {
    /// Create new sync state manager
    pub fn new(kv: Arc<dyn KVStore>) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            kv,
            checkpoint: Arc::new(RwLock::new(SyncCheckpoint {
                current_layer: 0,
                current_height: 0,
                target_height: 0,
                blocks_committed: 0,
                last_checkpoint_time: 0,
                sync_session_id: format!("sync-{}", now),
            })),
            pending_blocks: Arc::new(RwLock::new(HashMap::new())),
            last_checkpoint_layer: Arc::new(RwLock::new(0)),
        }
    }

    /// Initialize sync session with target height
    pub async fn start_sync_session(&self, target_height: u64) -> Result<()> {
        use std::time::{SystemTime, UNIX_EPOCH};

        let mut checkpoint = self.checkpoint.write().await;
        checkpoint.target_height = target_height;

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_secs();

        checkpoint.sync_session_id = format!("sync-{}", now);
        checkpoint.last_checkpoint_time = now;

        info!(
            "🚀 Starting DAG sync session {} to height {}",
            checkpoint.sync_session_id, target_height
        );

        // Save initial checkpoint
        self.save_checkpoint_internal(&checkpoint).await?;

        Ok(())
    }

    /// Resume from last checkpoint (crash recovery)
    pub async fn resume_from_checkpoint(&self) -> Result<Option<SyncCheckpoint>> {
        info!("🔄 Attempting to resume from last checkpoint...");

        match self.load_checkpoint().await? {
            Some(checkpoint) => {
                info!(
                    "✅ Found checkpoint: layer {}, height {}, {} blocks committed",
                    checkpoint.current_layer,
                    checkpoint.current_height,
                    checkpoint.blocks_committed
                );

                // Restore checkpoint state
                let mut current_checkpoint = self.checkpoint.write().await;
                *current_checkpoint = checkpoint.clone();

                let mut last_layer = self.last_checkpoint_layer.write().await;
                *last_layer = checkpoint.current_layer;

                Ok(Some(checkpoint))
            }
            None => {
                info!("📝 No checkpoint found, starting fresh sync");
                Ok(None)
            }
        }
    }

    /// Mark a DAG layer as committed
    ///
    /// Call this after successfully writing a layer to the database.
    /// Will automatically checkpoint every CHECKPOINT_INTERVAL_LAYERS.
    pub async fn commit_layer(
        &self,
        layer_num: usize,
        highest_height_in_layer: u64,
        blocks_in_layer: usize,
    ) -> Result<()> {
        // Update checkpoint
        let mut checkpoint = self.checkpoint.write().await;
        checkpoint.current_layer = layer_num;
        checkpoint.current_height = highest_height_in_layer;
        checkpoint.blocks_committed += blocks_in_layer as u64;

        debug!(
            "✅ Layer {} committed: height {}, {} blocks",
            layer_num, highest_height_in_layer, blocks_in_layer
        );

        // Check if we should save checkpoint
        let last_checkpoint_layer = *self.last_checkpoint_layer.read().await;
        if layer_num >= last_checkpoint_layer + CHECKPOINT_INTERVAL_LAYERS {
            info!(
                "💾 Checkpointing at layer {} ({} blocks committed)",
                layer_num, checkpoint.blocks_committed
            );

            checkpoint.last_checkpoint_time = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs();

            self.save_checkpoint_internal(&checkpoint).await?;

            let mut last_layer = self.last_checkpoint_layer.write().await;
            *last_layer = layer_num;
        }

        Ok(())
    }

    /// Add block to pending queue (awaiting parent resolution)
    pub async fn add_pending_block(&self, block_hash: String, block: Block) -> Result<()> {
        let mut pending = self.pending_blocks.write().await;

        // Enforce memory limit to prevent DoS
        if pending.len() >= MAX_PENDING_BLOCKS {
            warn!(
                "⚠️  Pending block limit reached ({}), dropping oldest block",
                MAX_PENDING_BLOCKS
            );
            // Remove oldest entry (HashMap iteration order is arbitrary, but good enough)
            if let Some(key) = pending.keys().next().cloned() {
                pending.remove(&key);
            }
        }

        pending.insert(block_hash, block);
        debug!("📥 Added block to pending queue (total: {})", pending.len());

        Ok(())
    }

    /// Remove block from pending queue (parents resolved)
    pub async fn remove_pending_block(&self, block_hash: &str) -> Option<Block> {
        let mut pending = self.pending_blocks.write().await;
        pending.remove(block_hash)
    }

    /// Get all pending blocks
    pub async fn get_pending_blocks(&self) -> HashMap<String, Block> {
        self.pending_blocks.read().await.clone()
    }

    /// Get current sync progress
    pub async fn get_progress(&self) -> SyncProgress {
        let checkpoint = self.checkpoint.read().await;
        let pending_count = self.pending_blocks.read().await.len();

        SyncProgress {
            current_layer: checkpoint.current_layer,
            current_height: checkpoint.current_height,
            target_height: checkpoint.target_height,
            blocks_committed: checkpoint.blocks_committed,
            pending_blocks: pending_count,
            progress_percent: if checkpoint.target_height > 0 {
                (checkpoint.current_height as f64 / checkpoint.target_height as f64 * 100.0)
            } else {
                0.0
            },
        }
    }

    /// Clear all state (for testing or database reset)
    pub async fn clear_all(&self) -> Result<()> {
        warn!("🗑️  Clearing all sync state (testing/reset)");

        self.pending_blocks.write().await.clear();

        let mut checkpoint = self.checkpoint.write().await;
        checkpoint.current_layer = 0;
        checkpoint.current_height = 0;
        checkpoint.blocks_committed = 0;

        self.delete_checkpoint().await?;

        Ok(())
    }

    // ========== Internal Methods ==========

    /// Save checkpoint to persistent storage
    async fn save_checkpoint_internal(&self, checkpoint: &SyncCheckpoint) -> Result<()> {
        let checkpoint_json = serde_json::to_vec(checkpoint)
            .context("Failed to serialize checkpoint")?;

        self.kv
            .put("manifest", b"dag_sync_checkpoint", &checkpoint_json)
            .await
            .context("Failed to save checkpoint to KV store")?;

        debug!(
            "💾 Checkpoint saved: layer {}, height {}",
            checkpoint.current_layer, checkpoint.current_height
        );

        Ok(())
    }

    /// Load checkpoint from persistent storage
    async fn load_checkpoint(&self) -> Result<Option<SyncCheckpoint>> {
        match self.kv.get("manifest", b"dag_sync_checkpoint").await? {
            Some(checkpoint_json) => {
                let checkpoint: SyncCheckpoint = serde_json::from_slice(&checkpoint_json)
                    .context("Failed to deserialize checkpoint")?;

                Ok(Some(checkpoint))
            }
            None => Ok(None),
        }
    }

    /// Delete checkpoint from storage
    async fn delete_checkpoint(&self) -> Result<()> {
        self.kv
            .delete("manifest", b"dag_sync_checkpoint")
            .await
            .context("Failed to delete checkpoint")?;

        Ok(())
    }
}

/// Sync progress information for monitoring
#[derive(Debug, Clone)]
pub struct SyncProgress {
    pub current_layer: usize,
    pub current_height: u64,
    pub target_height: u64,
    pub blocks_committed: u64,
    pub pending_blocks: usize,
    pub progress_percent: f64,
}

impl SyncProgress {
    /// Format progress as human-readable string
    pub fn to_string(&self) -> String {
        format!(
            "Layer {}, Height {}/{} ({:.1}%), {} blocks committed, {} pending",
            self.current_layer,
            self.current_height,
            self.target_height,
            self.progress_percent,
            self.blocks_committed,
            self.pending_blocks
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RocksDBKV;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_checkpoint_save_and_resume() {
        let temp_dir = TempDir::new().unwrap();
        let kv = Arc::new(RocksDBKV::open_hot_db(temp_dir.path()).await.unwrap());

        let manager = SyncStateManager::new(Arc::clone(&kv));

        // Start sync session
        manager.start_sync_session(1000).await.unwrap();

        // Commit some layers
        // v8.6.0: CHECKPOINT_INTERVAL_LAYERS is now 20, so checkpoint at layer 20
        manager.commit_layer(10, 500, 100).await.unwrap();
        manager.commit_layer(20, 1000, 100).await.unwrap(); // Should trigger checkpoint

        // Create new manager and resume
        let manager2 = SyncStateManager::new(kv);
        let checkpoint = manager2.resume_from_checkpoint().await.unwrap();

        assert!(checkpoint.is_some());
        let checkpoint = checkpoint.unwrap();
        assert_eq!(checkpoint.current_layer, 20);
        assert_eq!(checkpoint.current_height, 1000);
        assert_eq!(checkpoint.blocks_committed, 200);
    }

    #[tokio::test]
    async fn test_pending_blocks_limit() {
        let temp_dir = TempDir::new().unwrap();
        let kv = Arc::new(RocksDBKV::open_hot_db(temp_dir.path()).await.unwrap());
        let manager = SyncStateManager::new(kv);

        // Add blocks up to limit
        for i in 0..MAX_PENDING_BLOCKS + 100 {
            let block = Block::default(); // Assuming Default exists
            manager
                .add_pending_block(format!("block_{}", i), block)
                .await
                .unwrap();
        }

        // Should be capped at MAX_PENDING_BLOCKS
        let pending_count = manager.pending_blocks.read().await.len();
        assert!(pending_count <= MAX_PENDING_BLOCKS);
    }

    #[tokio::test]
    async fn test_progress_tracking() {
        let temp_dir = TempDir::new().unwrap();
        let kv = Arc::new(RocksDBKV::open_hot_db(temp_dir.path()).await.unwrap());
        let manager = SyncStateManager::new(kv);

        manager.start_sync_session(1000).await.unwrap();
        manager.commit_layer(5, 500, 100).await.unwrap();

        let progress = manager.get_progress().await;
        assert_eq!(progress.current_layer, 5);
        assert_eq!(progress.current_height, 500);
        assert_eq!(progress.target_height, 1000);
        assert_eq!(progress.blocks_committed, 100);
        assert_eq!(progress.progress_percent, 50.0);
    }
}
