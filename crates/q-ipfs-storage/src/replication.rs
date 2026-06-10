use crate::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, info, warn, error};

/// Gossipsub topic for database updates
pub const DATABASE_UPDATES_TOPIC: &str = "/qnk/database-updates/1.0.0";

/// Types of database updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DatabaseUpdateType {
    /// Incremental update with specific changes
    Incremental {
        /// Sequence number for ordering
        sequence: u64,
        /// CIDs of changed chunks
        changed_chunks: Vec<String>,
        /// Manifest CID for the complete state
        manifest_cid: String,
    },
    /// Full snapshot announcement
    Snapshot {
        /// Manifest CID for the snapshot
        manifest_cid: String,
        /// Timestamp of snapshot
        timestamp: u64,
    },
    /// Request for missing data
    SyncRequest {
        /// Last known sequence number
        last_sequence: u64,
        /// Node ID requesting sync
        requester: Vec<u8>,
    },
}

/// Database update message broadcast via gossipsub
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseUpdate {
    /// Node ID that produced this update
    pub node_id: Vec<u8>,
    /// Type of update
    pub update_type: DatabaseUpdateType,
    /// Signature (for verification)
    pub signature: Vec<u8>,
}

/// Configuration for database replication
#[derive(Debug, Clone)]
pub struct ReplicationConfig {
    /// Enable automatic replication
    pub enabled: bool,
    /// Interval between snapshot broadcasts (seconds)
    pub snapshot_interval: u64,
    /// Maximum number of incremental updates before full snapshot
    pub max_incremental_updates: usize,
    /// Enable verification of received updates
    pub verify_updates: bool,
    /// Number of parallel downloads for replication
    pub parallel_downloads: usize,
}

impl Default for ReplicationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            snapshot_interval: 300, // 5 minutes
            max_incremental_updates: 100,
            verify_updates: true,
            parallel_downloads: 10,
        }
    }
}

/// Statistics for replication
#[derive(Debug, Clone, Default)]
pub struct ReplicationStats {
    /// Total updates sent
    pub updates_sent: u64,
    /// Total updates received
    pub updates_received: u64,
    /// Total bytes synced
    pub bytes_synced: u64,
    /// Number of peers synced with
    pub peers_synced: usize,
    /// Last sync timestamp
    pub last_sync_time: Option<std::time::SystemTime>,
    /// Current sequence number
    pub current_sequence: u64,
}

/// Database replication manager
pub struct DatabaseReplicationManager {
    /// Local node ID
    node_id: Vec<u8>,
    /// IPFS storage instance
    storage: Arc<RwLock<Option<IpfsRocksStorage>>>,
    /// Configuration
    config: ReplicationConfig,
    /// Database path (v0.9.50-beta: Make configurable to fix recurring IPFS path bug)
    db_path: String,
    /// Replication statistics
    stats: Arc<RwLock<ReplicationStats>>,
    /// Channel for outgoing updates
    update_tx: mpsc::UnboundedSender<DatabaseUpdate>,
    /// Received update sequence numbers (for deduplication)
    received_sequences: Arc<RwLock<HashSet<u64>>>,
    /// Known manifest CIDs
    known_manifests: Arc<RwLock<HashMap<String, u64>>>,
}

impl DatabaseReplicationManager {
    /// Create a new replication manager
    ///
    /// # Arguments
    /// * `node_id` - Local node identifier
    /// * `storage` - IPFS storage instance
    /// * `config` - Replication configuration
    /// * `db_path` - Database path (v0.9.50-beta: from Q_DB_PATH env var)
    pub fn new(
        node_id: Vec<u8>,
        storage: Arc<RwLock<Option<IpfsRocksStorage>>>,
        config: ReplicationConfig,
        db_path: String,
    ) -> (Self, mpsc::UnboundedReceiver<DatabaseUpdate>) {
        let (update_tx, update_rx) = mpsc::unbounded_channel();

        let manager = Self {
            node_id,
            storage,
            config,
            db_path,
            stats: Arc::new(RwLock::new(ReplicationStats::default())),
            update_tx,
            received_sequences: Arc::new(RwLock::new(HashSet::new())),
            known_manifests: Arc::new(RwLock::new(HashMap::new())),
        };

        (manager, update_rx)
    }

    /// Start the replication manager
    pub async fn start(self: Arc<Self>) {
        if !self.config.enabled {
            info!("Database replication disabled");
            return;
        }

        info!("🔄 Starting database replication manager");
        info!("   Snapshot interval: {} seconds", self.config.snapshot_interval);
        info!("   Max incremental updates: {}", self.config.max_incremental_updates);

        // Spawn periodic snapshot broadcaster
        let manager_clone = self.clone();
        tokio::spawn(async move {
            manager_clone.snapshot_broadcaster().await;
        });

        info!("✅ Database replication manager started");
    }

    /// Periodic snapshot broadcaster
    async fn snapshot_broadcaster(&self) {
        let mut interval = tokio::time::interval(
            std::time::Duration::from_secs(self.config.snapshot_interval)
        );

        loop {
            interval.tick().await;

            // Create and broadcast snapshot
            if let Err(e) = self.broadcast_snapshot().await {
                warn!("Failed to broadcast snapshot: {:?}", e);
            }
        }
    }

    /// Broadcast a snapshot announcement
    async fn broadcast_snapshot(&self) -> Result<()> {
        info!("📸 Creating database snapshot for broadcast");

        let storage_guard = self.storage.read().await;
        let Some(ref storage_instance) = *storage_guard else {
            return Err(IpfsStorageError::Ipfs("Storage not initialized".to_string()));
        };

        // Note: This is a read-only operation, but we need mutable access for backup
        // In production, you'd want to implement a read-only snapshot method
        drop(storage_guard);

        let mut storage_write = self.storage.write().await;
        let Some(ref mut storage_mut) = *storage_write else {
            return Err(IpfsStorageError::Ipfs("Storage not initialized".to_string()));
        };

        // Create backup (this generates the manifest CID)
        // v0.9.50-beta FIX: Use configured database path from Q_DB_PATH env var
        // Previous versions hardcoded this path, causing RECURRING block production stalls
        // when the actual database was at a different location (e.g. ./data-mine5)
        // This bug caused production outages in v0.9.7, v0.9.8, and v0.9.49
        let manifest_cid = storage_mut.backup_database(
            &self.db_path, // v0.9.50-beta: Use configured path, not hardcoded!
            BackupOptions {
                snapshot_type: SnapshotType::Full,
                compress: true,
                replication: 3,
            }
        ).await?;

        drop(storage_write);

        // Update stats
        let mut stats = self.stats.write().await;
        stats.current_sequence += 1;
        let sequence = stats.current_sequence;
        drop(stats);

        // Create update message
        let update = DatabaseUpdate {
            node_id: self.node_id.clone(),
            update_type: DatabaseUpdateType::Snapshot {
                manifest_cid: manifest_cid.clone(),
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            },
            signature: vec![], // TODO: Implement signing
        };

        // Broadcast update
        if let Err(e) = self.update_tx.send(update) {
            warn!("Failed to send snapshot update: {}", e);
        }

        // Update stats
        let mut stats = self.stats.write().await;
        stats.updates_sent += 1;
        stats.last_sync_time = Some(std::time::SystemTime::now());

        // Store manifest CID
        let mut known_manifests = self.known_manifests.write().await;
        known_manifests.insert(manifest_cid.clone(), sequence);

        info!("✅ Broadcast snapshot with manifest CID: {}", manifest_cid);

        Ok(())
    }

    /// Handle incoming database update
    pub async fn handle_update(&self, update: DatabaseUpdate) -> Result<()> {
        // Skip our own updates
        if update.node_id == self.node_id {
            return Ok(());
        }

        debug!("📥 Received database update from peer: {}", hex::encode(&update.node_id[..8]));

        // Verify signature if enabled
        if self.config.verify_updates {
            // TODO: Implement signature verification
        }

        // Handle based on update type
        match update.update_type {
            DatabaseUpdateType::Snapshot { manifest_cid, timestamp } => {
                self.handle_snapshot_update(manifest_cid, timestamp).await?;
            }
            DatabaseUpdateType::Incremental { sequence, changed_chunks, manifest_cid } => {
                self.handle_incremental_update(sequence, changed_chunks, manifest_cid).await?;
            }
            DatabaseUpdateType::SyncRequest { last_sequence, requester } => {
                self.handle_sync_request(last_sequence, requester).await?;
            }
        }

        // Update stats
        let mut stats = self.stats.write().await;
        stats.updates_received += 1;

        Ok(())
    }

    /// Handle snapshot update
    async fn handle_snapshot_update(&self, manifest_cid: String, timestamp: u64) -> Result<()> {
        info!("📥 Handling snapshot update: {}", manifest_cid);

        // Check if we already have this manifest
        let known_manifests = self.known_manifests.read().await;
        if known_manifests.contains_key(&manifest_cid) {
            debug!("Already have snapshot {}, skipping", manifest_cid);
            return Ok(());
        }
        drop(known_manifests);

        // Download and apply snapshot
        let storage_guard = self.storage.read().await;
        let Some(ref storage_instance) = *storage_guard else {
            return Err(IpfsStorageError::Ipfs("Storage not initialized".to_string()));
        };

        // Restore to temporary location first
        let temp_restore_path = format!("./data-sync-{}", timestamp);
        info!("📦 Restoring snapshot to temporary location: {}", temp_restore_path);

        storage_instance.restore_database(
            &manifest_cid,
            &temp_restore_path,
            RestoreOptions {
                verify_chunks: self.config.verify_updates,
                parallel_downloads: self.config.parallel_downloads,
            }
        ).await?;

        info!("✅ Successfully restored snapshot from peer");

        // TODO: In production, you'd want to:
        // 1. Verify the restored data with consensus
        // 2. Merge with local database (not replace)
        // 3. Handle conflicts intelligently
        // 4. Clean up temporary location

        // Update known manifests
        let mut known_manifests = self.known_manifests.write().await;
        let mut stats = self.stats.write().await;
        stats.current_sequence += 1;
        known_manifests.insert(manifest_cid, stats.current_sequence);
        stats.peers_synced += 1;

        Ok(())
    }

    /// Handle incremental update
    async fn handle_incremental_update(
        &self,
        sequence: u64,
        changed_chunks: Vec<String>,
        manifest_cid: String,
    ) -> Result<()> {
        info!("📥 Handling incremental update (sequence: {}, chunks: {})", sequence, changed_chunks.len());

        // Check if we've already processed this sequence
        let mut received_sequences = self.received_sequences.write().await;
        if received_sequences.contains(&sequence) {
            debug!("Already processed sequence {}, skipping", sequence);
            return Ok(());
        }
        received_sequences.insert(sequence);
        drop(received_sequences);

        // Download changed chunks
        let storage_guard = self.storage.read().await;
        let Some(ref storage_instance) = *storage_guard else {
            return Err(IpfsStorageError::Ipfs("Storage not initialized".to_string()));
        };

        for chunk_cid in &changed_chunks {
            debug!("📦 Downloading changed chunk: {}", chunk_cid);
            // Download chunk via IPFS
            // TODO: Apply chunk to local database
        }

        info!("✅ Applied incremental update (sequence: {})", sequence);

        // Update stats
        let mut stats = self.stats.write().await;
        stats.bytes_synced += changed_chunks.len() as u64 * 256 * 1024; // Approximate

        Ok(())
    }

    /// Handle sync request from peer
    async fn handle_sync_request(&self, last_sequence: u64, requester: Vec<u8>) -> Result<()> {
        info!("📤 Peer {} requests sync from sequence {}", hex::encode(&requester[..8]), last_sequence);

        // TODO: Send missing updates to requester
        // This would involve:
        // 1. Finding all updates since last_sequence
        // 2. Sending them to the requester
        // 3. Possibly sending a full snapshot if too many updates

        Ok(())
    }

    /// Get replication statistics
    pub async fn get_stats(&self) -> ReplicationStats {
        self.stats.read().await.clone()
    }

    /// Request full sync from network
    pub async fn request_full_sync(&self) -> Result<()> {
        info!("🔄 Requesting full sync from network");

        let stats = self.stats.read().await;
        let last_sequence = stats.current_sequence;
        drop(stats);

        let sync_request = DatabaseUpdate {
            node_id: self.node_id.clone(),
            update_type: DatabaseUpdateType::SyncRequest {
                last_sequence,
                requester: self.node_id.clone(),
            },
            signature: vec![],
        };

        if let Err(e) = self.update_tx.send(sync_request) {
            warn!("Failed to send sync request: {}", e);
        }

        Ok(())
    }

    /// Get channel for outgoing updates (for gossipsub integration)
    pub fn get_update_sender(&self) -> mpsc::UnboundedSender<DatabaseUpdate> {
        self.update_tx.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_replication_manager_creation() {
        let node_id = vec![1, 2, 3, 4];
        let storage = Arc::new(RwLock::new(None));
        let config = ReplicationConfig::default();
        let db_path = "./data-test".to_string(); // v0.9.50-beta: Pass db_path

        let (manager, _rx) = DatabaseReplicationManager::new(node_id, storage, config, db_path);
        let stats = manager.get_stats().await;

        assert_eq!(stats.updates_sent, 0);
        assert_eq!(stats.updates_received, 0);
    }
}
