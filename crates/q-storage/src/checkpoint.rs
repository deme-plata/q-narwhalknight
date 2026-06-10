//! Height Checkpoint System
//!
//! Writes periodic checkpoint files outside of RocksDB for recovery validation.
//! If RocksDB loses data due to WAL truncation, the checkpoint file provides
//! an external reference for the last known good height.
//!
//! Checkpoint file format (JSON):
//! {
//!   "height": 300000,
//!   "block_hash": "abc123...",
//!   "timestamp": "2025-12-01T12:00:00Z",
//!   "version": "1.0.79-beta"
//! }

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn, error};
use serde::{Serialize, Deserialize};

/// Checkpoint interval in blocks
const CHECKPOINT_INTERVAL: u64 = 1000;

/// Checkpoint file data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Block height at checkpoint
    pub height: u64,
    /// Hash of the block at this height
    pub block_hash: String,
    /// Timestamp when checkpoint was created
    pub timestamp: String,
    /// Version of the software that created this checkpoint
    pub version: String,
}

/// Height checkpoint manager
pub struct CheckpointManager {
    /// Path to checkpoint file
    checkpoint_path: PathBuf,
    /// Last checkpointed height
    last_checkpoint: AtomicU64,
    /// Lock for checkpoint writes
    write_lock: RwLock<()>,
}

impl CheckpointManager {
    /// Create new checkpoint manager
    pub fn new(data_dir: &str) -> Self {
        let checkpoint_path = PathBuf::from(data_dir).join("checkpoint.json");

        // Try to load existing checkpoint
        let last_checkpoint = if checkpoint_path.exists() {
            match Self::load_checkpoint_sync(&checkpoint_path) {
                Ok(cp) => {
                    info!("📍 Loaded existing checkpoint: height={}, hash={}",
                          cp.height, &cp.block_hash[..16.min(cp.block_hash.len())]);
                    cp.height
                }
                Err(e) => {
                    warn!("⚠️  Failed to load checkpoint file: {}", e);
                    0
                }
            }
        } else {
            info!("📍 No existing checkpoint found, starting fresh");
            0
        };

        Self {
            checkpoint_path,
            last_checkpoint: AtomicU64::new(last_checkpoint),
            write_lock: RwLock::new(()),
        }
    }

    /// Load checkpoint synchronously (for initialization)
    fn load_checkpoint_sync(path: &PathBuf) -> Result<Checkpoint, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let checkpoint: Checkpoint = serde_json::from_str(&content)?;
        Ok(checkpoint)
    }

    /// Check if we should create a checkpoint at this height
    pub fn should_checkpoint(&self, height: u64) -> bool {
        // Checkpoint every CHECKPOINT_INTERVAL blocks
        if height % CHECKPOINT_INTERVAL != 0 {
            return false;
        }

        // Only checkpoint if we've advanced
        let last = self.last_checkpoint.load(Ordering::Relaxed);
        height > last
    }

    /// Create a checkpoint at the given height
    pub async fn create_checkpoint(&self, height: u64, block_hash: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let _lock = self.write_lock.write().await;

        let checkpoint = Checkpoint {
            height,
            block_hash: block_hash.to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            version: "1.0.79-beta".to_string(),
        };

        // Write checkpoint atomically (write to temp, then rename)
        let temp_path = self.checkpoint_path.with_extension("json.tmp");
        let content = serde_json::to_string_pretty(&checkpoint)?;

        // Write to temp file
        tokio::fs::write(&temp_path, &content).await?;

        // Sync to disk
        let file = tokio::fs::File::open(&temp_path).await?;
        file.sync_all().await?;

        // Atomic rename
        tokio::fs::rename(&temp_path, &self.checkpoint_path).await?;

        // Update last checkpoint
        self.last_checkpoint.store(height, Ordering::Relaxed);

        info!("📍 [CHECKPOINT] Created at height {} (hash: {})",
              height, &block_hash[..16.min(block_hash.len())]);

        Ok(())
    }

    /// Get the last checkpointed height
    pub fn last_checkpointed_height(&self) -> u64 {
        self.last_checkpoint.load(Ordering::Relaxed)
    }

    /// Load the current checkpoint
    pub async fn load_checkpoint(&self) -> Option<Checkpoint> {
        if !self.checkpoint_path.exists() {
            return None;
        }

        match tokio::fs::read_to_string(&self.checkpoint_path).await {
            Ok(content) => {
                match serde_json::from_str(&content) {
                    Ok(cp) => Some(cp),
                    Err(e) => {
                        warn!("⚠️  Failed to parse checkpoint: {}", e);
                        None
                    }
                }
            }
            Err(e) => {
                warn!("⚠️  Failed to read checkpoint: {}", e);
                None
            }
        }
    }

    /// Verify that the database height matches the checkpoint
    /// Returns: (is_valid, checkpoint_height, db_height)
    pub async fn verify_against_db(&self, db_height: u64) -> (bool, Option<u64>, u64) {
        let checkpoint = self.load_checkpoint().await;

        match checkpoint {
            Some(cp) => {
                if db_height < cp.height {
                    error!("🚨 DATABASE HEIGHT REGRESSION DETECTED!");
                    error!("   Checkpoint: {} blocks", cp.height);
                    error!("   Database:   {} blocks", db_height);
                    error!("   Missing:    {} blocks", cp.height - db_height);
                    error!("");
                    error!("   This indicates data loss! Possible causes:");
                    error!("   - RocksDB WAL corruption (kill -9)");
                    error!("   - Manual database deletion");
                    error!("   - Storage failure");
                    (false, Some(cp.height), db_height)
                } else {
                    debug!("✅ Database height {} matches/exceeds checkpoint {}", db_height, cp.height);
                    (true, Some(cp.height), db_height)
                }
            }
            None => {
                // No checkpoint exists yet
                (true, None, db_height)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_checkpoint_creation() {
        let temp_dir = TempDir::new().unwrap();
        let manager = CheckpointManager::new(temp_dir.path().to_str().unwrap());

        // Should checkpoint at multiples of CHECKPOINT_INTERVAL
        assert!(manager.should_checkpoint(CHECKPOINT_INTERVAL));
        assert!(manager.should_checkpoint(CHECKPOINT_INTERVAL * 2));
        assert!(!manager.should_checkpoint(CHECKPOINT_INTERVAL + 1));

        // Create checkpoint
        manager.create_checkpoint(CHECKPOINT_INTERVAL, "abc123def456").await.unwrap();

        // Verify it was saved
        assert_eq!(manager.last_checkpointed_height(), CHECKPOINT_INTERVAL);

        // Load and verify
        let cp = manager.load_checkpoint().await.unwrap();
        assert_eq!(cp.height, CHECKPOINT_INTERVAL);
        assert_eq!(cp.block_hash, "abc123def456");
    }

    #[tokio::test]
    async fn test_checkpoint_verification() {
        let temp_dir = TempDir::new().unwrap();
        let manager = CheckpointManager::new(temp_dir.path().to_str().unwrap());

        // No checkpoint yet - should be valid
        let (valid, _, _) = manager.verify_against_db(100).await;
        assert!(valid);

        // Create checkpoint at height 1000
        manager.create_checkpoint(1000, "hash1000").await.unwrap();

        // DB at same height - valid
        let (valid, _, _) = manager.verify_against_db(1000).await;
        assert!(valid);

        // DB at higher height - valid
        let (valid, _, _) = manager.verify_against_db(1500).await;
        assert!(valid);

        // DB at lower height - INVALID (data loss detected)
        let (valid, cp_height, db_height) = manager.verify_against_db(500).await;
        assert!(!valid);
        assert_eq!(cp_height, Some(1000));
        assert_eq!(db_height, 500);
    }
}
