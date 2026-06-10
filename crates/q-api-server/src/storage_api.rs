use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Json},
};
use q_ipfs_storage::{BackupOptions, IpfsRocksStorage, RestoreOptions, StorageConfig};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{error, info};

/// Shared storage state
pub type StorageState = Arc<RwLock<Option<IpfsRocksStorage>>>;

/// Request to create a backup
#[derive(Debug, Deserialize)]
pub struct BackupRequest {
    /// Path to the database to backup
    pub db_path: String,
    /// Enable compression (default: true)
    #[serde(default = "default_true")]
    pub compress: bool,
    /// Replication factor (default: 3)
    #[serde(default = "default_replication")]
    pub replication: usize,
}

fn default_true() -> bool {
    true
}

fn default_replication() -> usize {
    3
}

/// Response from backup operation
#[derive(Debug, Serialize)]
pub struct BackupResponse {
    /// Success status
    pub success: bool,
    /// Manifest CID for the backup
    pub manifest_cid: Option<String>,
    /// Error message if failed
    pub error: Option<String>,
}

/// Request to restore from backup
#[derive(Debug, Deserialize)]
pub struct RestoreRequest {
    /// Manifest CID to restore from
    pub manifest_cid: String,
    /// Path where to restore the database
    pub output_path: String,
    /// Verify chunks (default: true)
    #[serde(default = "default_true")]
    pub verify_chunks: bool,
}

/// Response from restore operation
#[derive(Debug, Serialize)]
pub struct RestoreResponse {
    /// Success status
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
}

/// Initialize the storage system
pub async fn initialize_storage() -> Result<IpfsRocksStorage, String> {
    info!("Initializing IPFS-RocksDB storage system");

    let config = StorageConfig::default();

    IpfsRocksStorage::new(config)
        .await
        .map_err(|e| format!("Failed to initialize storage: {:?}", e))
}

/// POST /api/storage/backup - Create a database backup
pub async fn backup_database(
    State(storage): State<StorageState>,
    Json(request): Json<BackupRequest>,
) -> impl IntoResponse {
    info!("Backup request for: {}", request.db_path);

    // Get storage instance
    let storage_lock = storage.read().await;
    let Some(ref storage_instance) = *storage_lock else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(BackupResponse {
                success: false,
                manifest_cid: None,
                error: Some("Storage system not initialized".to_string()),
            }),
        );
    };

    // Create backup options
    let options = BackupOptions {
        snapshot_type: q_ipfs_storage::SnapshotType::Full,
        compress: request.compress,
        replication: request.replication,
    };

    // Need to drop the read lock before we can get a write lock
    drop(storage_lock);

    // Perform backup (requires mutable access)
    let mut storage_write = storage.write().await;
    let Some(ref mut storage_mut) = *storage_write else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(BackupResponse {
                success: false,
                manifest_cid: None,
                error: Some("Storage system not initialized".to_string()),
            }),
        );
    };

    match storage_mut.backup_database(&request.db_path, options).await {
        Ok(manifest_cid) => {
            info!("Backup successful: {}", manifest_cid);
            (
                StatusCode::OK,
                Json(BackupResponse {
                    success: true,
                    manifest_cid: Some(manifest_cid),
                    error: None,
                }),
            )
        }
        Err(e) => {
            error!("Backup failed: {:?}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(BackupResponse {
                    success: false,
                    manifest_cid: None,
                    error: Some(format!("Backup failed: {:?}", e)),
                }),
            )
        }
    }
}

/// POST /api/storage/restore - Restore database from backup
pub async fn restore_database(
    State(storage): State<StorageState>,
    Json(request): Json<RestoreRequest>,
) -> impl IntoResponse {
    info!("Restore request from CID: {}", request.manifest_cid);

    // Get storage instance
    let storage_lock = storage.read().await;
    let Some(ref storage_instance) = *storage_lock else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(RestoreResponse {
                success: false,
                error: Some("Storage system not initialized".to_string()),
            }),
        );
    };

    // Create restore options
    let options = RestoreOptions {
        verify_chunks: request.verify_chunks,
        parallel_downloads: 10,
    };

    // Perform restore
    match storage_instance
        .restore_database(&request.manifest_cid, &request.output_path, options)
        .await
    {
        Ok(_) => {
            info!("Restore successful to: {}", request.output_path);
            (
                StatusCode::OK,
                Json(RestoreResponse {
                    success: true,
                    error: None,
                }),
            )
        }
        Err(e) => {
            error!("Restore failed: {:?}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(RestoreResponse {
                    success: false,
                    error: Some(format!("Restore failed: {:?}", e)),
                }),
            )
        }
    }
}

/// GET /api/storage/status - Get storage system status
pub async fn storage_status(State(storage): State<StorageState>) -> impl IntoResponse {
    let storage_lock = storage.read().await;
    let initialized = storage_lock.is_some();

    Json(serde_json::json!({
        "initialized": initialized,
        "system": "IPFS-RocksDB Decentralized Storage",
        "version": "1.0.0"
    }))
}
