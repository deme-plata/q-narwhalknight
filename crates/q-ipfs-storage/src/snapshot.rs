use crate::{IpfsStorageError, Result};
#[cfg(not(target_os = "windows"))]
use rocksdb::{checkpoint::Checkpoint, DB};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

/// Type of snapshot to create
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SnapshotType {
    /// Full snapshot of entire database
    Full,
    /// Incremental snapshot (only changed SST files)
    Incremental,
}

/// Metadata about a database snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotMetadata {
    /// Unique identifier for this snapshot
    pub id: String,
    /// Type of snapshot
    pub snapshot_type: SnapshotType,
    /// Timestamp when snapshot was created
    pub timestamp: i64,
    /// Size of snapshot in bytes
    pub size_bytes: u64,
    /// Number of files in snapshot
    pub file_count: usize,
    /// Parent snapshot ID (for incremental snapshots)
    pub parent_id: Option<String>,
    /// Database path that was snapshotted
    pub db_path: PathBuf,
    /// Checkpoint path where snapshot was created
    pub checkpoint_path: PathBuf,
}

/// Manages RocksDB snapshots for IPFS storage
pub struct SnapshotManager {
    /// Base directory for storing checkpoints
    checkpoint_dir: PathBuf,
    /// Last snapshot metadata (for incremental snapshots)
    last_snapshot: Option<SnapshotMetadata>,
}

impl SnapshotManager {
    /// Create a new snapshot manager
    pub fn new<P: AsRef<Path>>(checkpoint_dir: P) -> Result<Self> {
        let checkpoint_dir = checkpoint_dir.as_ref().to_path_buf();

        // Create checkpoint directory if it doesn't exist
        std::fs::create_dir_all(&checkpoint_dir)?;

        Ok(Self {
            checkpoint_dir,
            last_snapshot: None,
        })
    }

    /// Create a snapshot of the given RocksDB database
    #[cfg(not(target_os = "windows"))]
    pub async fn create_snapshot<P: AsRef<Path>>(
        &mut self,
        db_path: P,
        snapshot_type: SnapshotType,
    ) -> Result<SnapshotMetadata> {
        let db_path = db_path.as_ref();

        info!(
            "Creating {:?} snapshot of database: {}",
            snapshot_type,
            db_path.display()
        );

        // Open the database (read-only to avoid lock issues)
        let db = DB::open_for_read_only(
            &rocksdb::Options::default(),
            db_path,
            false, // do not fail if wal files are missing
        )?;

        // Generate snapshot ID
        let snapshot_id = uuid::Uuid::new_v4().to_string();
        let checkpoint_path = self.checkpoint_dir.join(&snapshot_id);

        // Create checkpoint
        let checkpoint = Checkpoint::new(&db)?;
        checkpoint.create_checkpoint(&checkpoint_path)?;

        info!(
            "Checkpoint created at: {}",
            checkpoint_path.display()
        );

        // Calculate snapshot size
        let (size_bytes, file_count) = self.calculate_snapshot_size(&checkpoint_path)?;

        let metadata = SnapshotMetadata {
            id: snapshot_id,
            snapshot_type,
            timestamp: chrono::Utc::now().timestamp(),
            size_bytes,
            file_count,
            parent_id: if snapshot_type == SnapshotType::Incremental {
                self.last_snapshot.as_ref().map(|s| s.id.clone())
            } else {
                None
            },
            db_path: db_path.to_path_buf(),
            checkpoint_path: checkpoint_path.clone(),
        };

        // Update last snapshot
        self.last_snapshot = Some(metadata.clone());

        info!(
            "Snapshot created: {} files, {} bytes",
            file_count, size_bytes
        );

        Ok(metadata)
    }

    /// Stub for Windows (RocksDB not available)
    #[cfg(target_os = "windows")]
    pub async fn create_snapshot<P: AsRef<Path>>(
        &mut self,
        _db_path: P,
        _snapshot_type: SnapshotType,
    ) -> Result<SnapshotMetadata> {
        Err(IpfsStorageError::Ipfs("Snapshots not supported on Windows (no RocksDB)".into()))
    }

    /// Calculate total size and file count of a snapshot
    fn calculate_snapshot_size(&self, checkpoint_path: &Path) -> Result<(u64, usize)> {
        let mut total_size = 0u64;
        let mut file_count = 0usize;

        for entry in std::fs::read_dir(checkpoint_path)? {
            let entry = entry?;
            let metadata = entry.metadata()?;

            if metadata.is_file() {
                total_size += metadata.len();
                file_count += 1;
            }
        }

        Ok((total_size, file_count))
    }

    /// List all files in a snapshot
    pub fn list_snapshot_files(&self, snapshot: &SnapshotMetadata) -> Result<Vec<PathBuf>> {
        let mut files = Vec::new();

        for entry in std::fs::read_dir(&snapshot.checkpoint_path)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() {
                files.push(path);
            }
        }

        files.sort();
        Ok(files)
    }

    /// Clean up old snapshots
    pub async fn cleanup_snapshots(&self, keep_count: usize) -> Result<Vec<String>> {
        debug!("Cleaning up old snapshots, keeping {} most recent", keep_count);

        let mut snapshots: Vec<_> = std::fs::read_dir(&self.checkpoint_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir())
            .collect();

        // Sort by modification time (oldest first)
        snapshots.sort_by_key(|e| {
            e.metadata()
                .and_then(|m| m.modified())
                .ok()
        });

        let mut removed = Vec::new();

        // Remove oldest snapshots if we have more than keep_count
        if snapshots.len() > keep_count {
            for snapshot in snapshots.iter().take(snapshots.len() - keep_count) {
                let path = snapshot.path();
                if let Some(id) = path.file_name().and_then(|n| n.to_str()) {
                    info!("Removing old snapshot: {}", id);
                    std::fs::remove_dir_all(&path)?;
                    removed.push(id.to_string());
                }
            }
        }

        Ok(removed)
    }

    /// Delete a specific snapshot
    pub async fn delete_snapshot(&self, snapshot_id: &str) -> Result<()> {
        let snapshot_path = self.checkpoint_dir.join(snapshot_id);

        if snapshot_path.exists() {
            info!("Deleting snapshot: {}", snapshot_id);
            std::fs::remove_dir_all(&snapshot_path)?;
            Ok(())
        } else {
            warn!("Snapshot not found: {}", snapshot_id);
            Err(IpfsStorageError::SnapshotNotFound(snapshot_id.to_string()))
        }
    }

    /// Get checkpoint directory
    pub fn checkpoint_dir(&self) -> &Path {
        &self.checkpoint_dir
    }

    /// Get last snapshot metadata
    pub fn last_snapshot(&self) -> Option<&SnapshotMetadata> {
        self.last_snapshot.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_snapshot_manager_creation() {
        let temp_dir = TempDir::new().unwrap();
        let manager = SnapshotManager::new(temp_dir.path()).unwrap();

        assert_eq!(manager.checkpoint_dir(), temp_dir.path());
        assert!(manager.last_snapshot().is_none());
    }

    #[tokio::test]
    async fn test_snapshot_cleanup() {
        let temp_dir = TempDir::new().unwrap();
        let manager = SnapshotManager::new(temp_dir.path()).unwrap();

        // Create some dummy snapshot directories
        for i in 0..5 {
            std::fs::create_dir(temp_dir.path().join(format!("snapshot-{}", i))).unwrap();
        }

        let removed = manager.cleanup_snapshots(3).await.unwrap();
        assert_eq!(removed.len(), 2);
    }
}
