/*!
# Q-IPFS-Storage: Decentralized RocksDB Storage via IPFS

This crate provides distributed storage for RocksDB databases using IPFS,
integrated with the Q-NarwhalKnight network infrastructure.

## Features

- **Automatic Snapshots**: Create incremental and full snapshots
- **IPFS Storage**: Store database snapshots on IPFS with content addressing
- **Distributed Pinning**: Pin data across multiple network nodes
- **Compression**: Zstd/LZ4 compression for efficient storage
- **Restoration**: Restore databases from IPFS CIDs
- **Network Integration**: Leverages existing q-network libp2p infrastructure

## Architecture

```
┌─────────────────┐
│   RocksDB       │
│   Database      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Snapshot       │
│  Manager        │  ← Creates checkpoints
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Chunker        │  ← Splits into 256KB chunks
│  + Compressor   │  ← Zstd compression
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  IPFS Client    │  ← Content-addressed storage
│  (via libp2p)   │  ← Distributed pinning
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Q-Network      │  ← Gossip manifest CIDs
│  Gossip Layer   │  ← Multi-node pinning
└─────────────────┘
```

## Usage

```rust
use q_ipfs_storage::{IpfsRocksStorage, StorageConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = StorageConfig::default();
    let mut storage = IpfsRocksStorage::new(config).await?;

    // Backup database to IPFS
    let cid = storage.backup_database("/path/to/rocksdb").await?;
    println!("Database backed up to IPFS: {}", cid);

    // Restore from IPFS
    storage.restore_database(&cid, "/path/to/restore").await?;

    Ok(())
}
```
*/

pub mod chunker;
pub mod compression;
pub mod ipfs_client;
pub mod manifest;
pub mod pinning;
pub mod snapshot;
pub mod storage;
pub mod replication;

// Re-exports
pub use storage::{IpfsRocksStorage, StorageConfig, StorageStats, BackupOptions, RestoreOptions};
pub use snapshot::{SnapshotManager, SnapshotMetadata, SnapshotType};
pub use chunker::{ChunkManager, ChunkMetadata};
pub use compression::{Compressor, CompressionType};
pub use ipfs_client::{IpfsClient, IpfsConfig};
pub use manifest::{StorageManifest, ChunkInfo};
pub use pinning::{PinningStrategy, PinningManager, PinStatus};
pub use replication::{DatabaseReplicationManager, ReplicationConfig, ReplicationStats, DatabaseUpdate, DATABASE_UPDATES_TOPIC};

use thiserror::Error;

/// Errors that can occur during IPFS storage operations
#[derive(Error, Debug)]
pub enum IpfsStorageError {
    #[cfg(not(target_os = "windows"))]
    #[error("RocksDB error: {0}")]
    RocksDb(#[from] rocksdb::Error),

    #[error("IPFS operation failed: {0}")]
    Ipfs(String),

    #[error("Network error: {0}")]
    Network(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Compression error: {0}")]
    Compression(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Snapshot not found: {0}")]
    SnapshotNotFound(String),

    #[error("Invalid CID: {0}")]
    InvalidCid(String),

    #[error("Chunk verification failed: expected {expected}, got {actual}")]
    ChunkVerification { expected: String, actual: String },

    #[error("Manifest validation failed: {0}")]
    ManifestValidation(String),

    #[error("Pinning failed: {0}")]
    Pinning(String),
}

pub type Result<T> = std::result::Result<T, IpfsStorageError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_types() {
        let err = IpfsStorageError::InvalidCid("test".to_string());
        assert!(err.to_string().contains("Invalid CID"));
    }
}
