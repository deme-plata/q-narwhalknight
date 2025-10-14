use crate::*;
use bytes::Bytes;
use std::path::Path;
use tracing::info;

/// Configuration for the storage system
#[derive(Debug, Clone)]
pub struct StorageConfig {
    /// Directory for RocksDB checkpoints
    pub checkpoint_dir: String,
    /// IPFS client configuration
    pub ipfs_config: ipfs_client::IpfsConfig,
    /// Enable compression
    pub enable_compression: bool,
    /// Compression type
    pub compression_type: compression::CompressionType,
    /// Chunk size in bytes
    pub chunk_size: usize,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            checkpoint_dir: "./checkpoints".to_string(),
            ipfs_config: ipfs_client::IpfsConfig::default(),
            enable_compression: true,
            compression_type: compression::CompressionType::Zstd,
            chunk_size: chunker::DEFAULT_CHUNK_SIZE,
        }
    }
}

/// Statistics about storage operations
#[derive(Debug, Clone)]
pub struct StorageStats {
    /// Total number of snapshots created
    pub total_snapshots: usize,
    /// Total size of all snapshots in bytes
    pub total_size_bytes: u64,
    /// Total compressed size in bytes
    pub total_compressed_bytes: u64,
    /// Number of chunks uploaded
    pub total_chunks: usize,
    /// Compression ratio (0.0 - 1.0)
    pub compression_ratio: f64,
}

/// Options for backup operation
#[derive(Debug, Clone)]
pub struct BackupOptions {
    /// Type of snapshot to create
    pub snapshot_type: SnapshotType,
    /// Enable compression
    pub compress: bool,
    /// Replication factor
    pub replication: usize,
}

impl Default for BackupOptions {
    fn default() -> Self {
        Self {
            snapshot_type: SnapshotType::Full,
            compress: true,
            replication: 3,
        }
    }
}

/// Options for restore operation
#[derive(Debug, Clone)]
pub struct RestoreOptions {
    /// Verify chunks after download
    pub verify_chunks: bool,
    /// Number of parallel downloads
    pub parallel_downloads: usize,
}

impl Default for RestoreOptions {
    fn default() -> Self {
        Self {
            verify_chunks: true,
            parallel_downloads: 10,
        }
    }
}

/// Main orchestrator for IPFS-RocksDB storage
pub struct IpfsRocksStorage {
    /// Snapshot manager
    snapshot_manager: SnapshotManager,
    /// Chunk manager
    chunk_manager: ChunkManager,
    /// Compressor
    compressor: Compressor,
    /// IPFS client
    ipfs_client: IpfsClient,
    /// Pinning manager
    pinning_manager: PinningManager,
    /// Configuration
    config: StorageConfig,
}

impl IpfsRocksStorage {
    /// Create a new storage system
    pub async fn new(config: StorageConfig) -> Result<Self> {
        info!("Initializing IPFS-RocksDB storage system");

        let snapshot_manager = SnapshotManager::new(&config.checkpoint_dir)?;
        let chunk_manager = ChunkManager::with_chunk_size(config.chunk_size);
        let compressor = Compressor::with_config(config.compression_type, 3);
        let ipfs_client = IpfsClient::new(config.ipfs_config.clone()).await?;
        let pinning_manager = PinningManager::new(PinningStrategy::Replicated(
            config.ipfs_config.replication_factor,
        ));

        info!("Storage system initialized successfully");

        Ok(Self {
            snapshot_manager,
            chunk_manager,
            compressor,
            ipfs_client,
            pinning_manager,
            config,
        })
    }

    /// Backup a RocksDB database to IPFS
    pub async fn backup_database<P: AsRef<Path>>(
        &mut self,
        db_path: P,
        options: BackupOptions,
    ) -> Result<String> {
        let db_path = db_path.as_ref();
        info!("Starting backup of database: {}", db_path.display());

        // Step 1: Create snapshot
        info!("Creating snapshot...");
        let snapshot = self
            .snapshot_manager
            .create_snapshot(db_path, options.snapshot_type)
            .await?;

        info!(
            "Snapshot created: {} files, {} bytes",
            snapshot.file_count, snapshot.size_bytes
        );

        // Step 2: Create manifest
        let mut manifest = StorageManifest::new(snapshot.clone());

        // Step 3: Process each file in snapshot
        let files = self.snapshot_manager.list_snapshot_files(&snapshot)?;
        let mut total_compressed = 0u64;

        for file in files {
            info!("Processing file: {}", file.display());

            // Chunk the file
            let chunks = self.chunk_manager.chunk_file(&file).await?;

            for (chunk_meta, chunk_data) in chunks {
                // Compress if enabled
                let data_to_upload = if options.compress {
                    let compressed = self.compressor.compress(&chunk_data)?;
                    total_compressed += compressed.len() as u64;
                    compressed
                } else {
                    total_compressed += chunk_data.len() as u64;
                    chunk_data
                };

                // Upload to IPFS
                let cid = self.ipfs_client.put_chunk(&data_to_upload).await?;

                info!(
                    "Uploaded chunk {} with CID: {}",
                    chunk_meta.index, cid
                );

                // Pin locally if enabled
                if self.config.ipfs_config.enable_local_pinning {
                    self.ipfs_client.pin_local(&cid).await?;
                }

                // Add to manifest
                manifest.add_chunk(cid.clone(), chunk_meta);

                // Request remote pinning
                self.pinning_manager.pin_chunk(&cid).await?;
            }
        }

        // Step 4: Upload manifest to IPFS
        let manifest_json = manifest.to_json()?;
        let manifest_cid = self.ipfs_client.put_chunk(manifest_json.as_bytes()).await?;

        info!("Manifest uploaded with CID: {}", manifest_cid);

        // Pin manifest
        if self.config.ipfs_config.enable_local_pinning {
            self.ipfs_client.pin_local(&manifest_cid).await?;
        }

        info!(
            "Backup complete: {} chunks, {:.2}% compression",
            manifest.total_chunks,
            ((snapshot.size_bytes - total_compressed) as f64 / snapshot.size_bytes as f64) * 100.0
        );

        // Step 5: Cleanup old snapshots (keep last 10)
        self.snapshot_manager.cleanup_snapshots(10).await?;

        Ok(manifest_cid)
    }

    /// Restore a database from IPFS
    pub async fn restore_database<P: AsRef<Path>>(
        &self,
        manifest_cid: &str,
        output_path: P,
        options: RestoreOptions,
    ) -> Result<()> {
        let output_path = output_path.as_ref();
        info!(
            "Starting restore from manifest: {} to {}",
            manifest_cid,
            output_path.display()
        );

        // Step 1: Download manifest
        info!("Downloading manifest...");
        let manifest_data = self.ipfs_client.get_chunk(manifest_cid).await?;
        let manifest_json = String::from_utf8(manifest_data.to_vec())
            .map_err(|e| IpfsStorageError::Ipfs(format!("Invalid UTF-8 in manifest: {}", e)))?;
        let manifest = StorageManifest::from_json(&manifest_json)?;

        info!(
            "Manifest loaded: {} chunks to download",
            manifest.total_chunks
        );

        // Step 2: Download all chunks and organize by file
        info!("Downloading {} chunks...", manifest.total_chunks);

        // Group chunks by file path with metadata
        use std::collections::HashMap;
        let mut file_chunks: HashMap<String, Vec<(ChunkMetadata, Bytes)>> = HashMap::new();

        for chunk_info in &manifest.chunks {
            info!("Downloading chunk {}: {}", chunk_info.metadata.index, chunk_info.cid);

            // Download chunk
            let compressed_data = self.ipfs_client.get_chunk(&chunk_info.cid).await?;

            // Decompress
            let data = if self.config.enable_compression {
                self.compressor.decompress(&compressed_data)?
            } else {
                compressed_data
            };

            // Verify chunk if enabled
            if options.verify_chunks {
                if !self
                    .chunk_manager
                    .verify_chunk(&chunk_info.metadata, &data)?
                {
                    return Err(IpfsStorageError::ChunkVerification {
                        expected: chunk_info.metadata.hash.clone(),
                        actual: "mismatch".to_string(),
                    });
                }
            }

            // Group by file path with full metadata
            file_chunks
                .entry(chunk_info.metadata.file_path.clone())
                .or_insert_with(Vec::new)
                .push((chunk_info.metadata.clone(), data));
        }

        info!("All chunks downloaded and verified");

        // Step 3: Reassemble files
        info!("Reassembling {} files...", file_chunks.len());
        std::fs::create_dir_all(output_path)?;

        for (file_path, mut chunks) in file_chunks {
            // Sort chunks by index
            chunks.sort_by_key(|(meta, _)| meta.index);

            // Chunks are already in correct format for reassemble_chunks
            let chunk_data = chunks;

            // Determine output file path
            let file_name = std::path::Path::new(&file_path)
                .file_name()
                .ok_or_else(|| {
                    IpfsStorageError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "Invalid file path in manifest",
                    ))
                })?;

            let output_file = (output_path.as_ref() as &Path).join(file_name);

            // Reassemble file
            info!("Reassembling file: {:?}", output_file);
            self.chunk_manager
                .reassemble_chunks(chunk_data, &output_file)
                .await?;
        }

        info!("File reassembly complete");

        info!("Restore complete: {}", output_path.display());

        Ok(())
    }

    /// Get storage statistics
    pub async fn get_stats(&self) -> Result<StorageStats> {
        // TODO: Implement actual stats collection
        Ok(StorageStats {
            total_snapshots: 0,
            total_size_bytes: 0,
            total_compressed_bytes: 0,
            total_chunks: 0,
            compression_ratio: 0.0,
        })
    }

    /// List all available backups
    pub async fn list_backups(&self) -> Result<Vec<String>> {
        // TODO: Implement backup listing
        Ok(Vec::new())
    }

    /// Delete a backup
    pub async fn delete_backup(&self, _manifest_cid: &str) -> Result<()> {
        // TODO: Implement backup deletion
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_storage_creation() {
        let temp_dir = TempDir::new().unwrap();
        let mut config = StorageConfig::default();
        config.checkpoint_dir = temp_dir.path().to_string_lossy().to_string();

        let storage = IpfsRocksStorage::new(config).await;
        assert!(storage.is_ok());
    }
}
