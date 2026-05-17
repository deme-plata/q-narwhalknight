//! Multi-path and S3 overflow storage for Q-NarwhalKnight
//!
//! # Overview
//! Enables storing database across multiple folders/disks including S3 storage.
//! When primary disk is full, new blocks are written to overflow paths.
//!
//! # Environment Variables
//! - `Q_STORAGE_OVERFLOW_PATHS`: Comma-separated list of overflow paths
//!   Example: `/mnt/disk2/q-data,/mnt/s3-storage/q-data`
//! - `Q_STORAGE_S3_BUCKET`: S3 bucket for archival storage (optional)
//! - `Q_STORAGE_S3_PREFIX`: S3 key prefix (default: "q-narwhalknight")
//! - `Q_STORAGE_S3_REGION`: S3 region (default: "eu-west-1")
//! - `Q_STORAGE_PRIMARY_MIN_FREE_GB`: Minimum free space on primary before overflow (default: 10)
//! - `Q_STORAGE_ARCHIVE_AFTER_BLOCKS`: Archive blocks older than N blocks to S3 (default: 100000)
//!
//! # S3 Compatibility Notes
//! S3 has high latency (100-300ms) and is not suitable for hot data.
//! This implementation uses S3 ONLY for:
//! - Archival/cold blocks (older than Q_STORAGE_ARCHIVE_AFTER_BLOCKS)
//! - Backup files
//!
//! Hot data (recent blocks, balances, state) always stays on local/fast storage.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Storage overflow configuration
#[derive(Debug, Clone)]
pub struct OverflowConfig {
    /// Primary storage path (fastest storage)
    pub primary_path: PathBuf,
    /// Overflow paths for when primary is full (local disks)
    pub overflow_paths: Vec<PathBuf>,
    /// S3 bucket for archival (optional)
    pub s3_bucket: Option<String>,
    /// S3 key prefix
    pub s3_prefix: String,
    /// S3 region
    pub s3_region: String,
    /// Minimum free GB on primary before using overflow
    pub primary_min_free_gb: u64,
    /// Archive blocks older than this to S3
    pub archive_after_blocks: u64,
    /// Enable S3 for backups
    pub s3_backups_enabled: bool,
}

impl Default for OverflowConfig {
    fn default() -> Self {
        Self {
            primary_path: PathBuf::from("./data"),
            overflow_paths: Vec::new(),
            s3_bucket: None,
            s3_prefix: "q-narwhalknight".to_string(),
            s3_region: "eu-west-1".to_string(),
            primary_min_free_gb: 10,
            archive_after_blocks: 100_000,
            s3_backups_enabled: false,
        }
    }
}

impl OverflowConfig {
    /// Load configuration from environment variables
    pub fn from_env() -> Self {
        let mut config = Self::default();

        // Parse overflow paths
        if let Ok(paths) = std::env::var("Q_STORAGE_OVERFLOW_PATHS") {
            config.overflow_paths = paths
                .split(',')
                .map(|s| PathBuf::from(s.trim()))
                .filter(|p| !p.as_os_str().is_empty())
                .collect();

            if !config.overflow_paths.is_empty() {
                info!("📂 [OVERFLOW] Configured {} overflow paths", config.overflow_paths.len());
                for path in &config.overflow_paths {
                    info!("   └─ {:?}", path);
                }
            }
        }

        // S3 configuration
        if let Ok(bucket) = std::env::var("Q_STORAGE_S3_BUCKET") {
            config.s3_bucket = Some(bucket.clone());
            config.s3_backups_enabled = true;
            info!("☁️  [S3] Configured bucket: {}", bucket);
        }

        if let Ok(prefix) = std::env::var("Q_STORAGE_S3_PREFIX") {
            config.s3_prefix = prefix;
        }

        if let Ok(region) = std::env::var("Q_STORAGE_S3_REGION") {
            config.s3_region = region;
        }

        if let Ok(min_free) = std::env::var("Q_STORAGE_PRIMARY_MIN_FREE_GB") {
            if let Ok(gb) = min_free.parse() {
                config.primary_min_free_gb = gb;
            }
        }

        if let Ok(archive_after) = std::env::var("Q_STORAGE_ARCHIVE_AFTER_BLOCKS") {
            if let Ok(blocks) = archive_after.parse() {
                config.archive_after_blocks = blocks;
            }
        }

        config
    }
}

/// Manages storage across multiple paths with S3 support
pub struct OverflowManager {
    config: OverflowConfig,
    /// Current active storage path (may change when primary fills up)
    active_path: Arc<RwLock<PathBuf>>,
    /// Block height ranges stored in each path: (path, start_height, end_height)
    path_ranges: Arc<RwLock<Vec<(PathBuf, u64, u64)>>>,
}

impl OverflowManager {
    /// Create new overflow manager
    pub fn new(config: OverflowConfig) -> Self {
        let active_path = config.primary_path.clone();

        Self {
            config,
            active_path: Arc::new(RwLock::new(active_path)),
            path_ranges: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Get the current active storage path for new writes
    pub async fn get_active_path(&self) -> PathBuf {
        self.active_path.read().await.clone()
    }

    /// Check if we need to switch to overflow storage
    pub async fn check_and_switch_if_needed(&self) -> Result<bool> {
        let current_path = self.active_path.read().await.clone();

        // Check free space on current path
        let free_gb = self.get_free_space_gb(&current_path).await?;

        if free_gb >= self.config.primary_min_free_gb {
            return Ok(false); // Enough space
        }

        warn!(
            "⚠️  [OVERFLOW] Primary storage low: {} GB free (min: {} GB)",
            free_gb, self.config.primary_min_free_gb
        );

        // Find an overflow path with space
        for overflow_path in &self.config.overflow_paths {
            let overflow_free = self.get_free_space_gb(overflow_path).await?;

            if overflow_free >= self.config.primary_min_free_gb {
                info!(
                    "📂 [OVERFLOW] Switching to overflow path: {:?} ({} GB free)",
                    overflow_path, overflow_free
                );

                // Ensure directory exists
                tokio::fs::create_dir_all(overflow_path).await?;

                // Switch active path
                *self.active_path.write().await = overflow_path.clone();

                return Ok(true);
            }
        }

        error!("🚨 [OVERFLOW] All storage paths are full!");
        Err(anyhow::anyhow!("All storage paths are full"))
    }

    /// Get free space in GB for a path
    async fn get_free_space_gb(&self, path: &Path) -> Result<u64> {
        // Handle S3 paths (always return "unlimited")
        if self.is_s3_path(path) {
            return Ok(u64::MAX / (1024 * 1024 * 1024));
        }

        let path = path.to_path_buf();
        let free_bytes = tokio::task::spawn_blocking(move || -> Result<u64> {
            // Use statvfs on Unix
            #[cfg(unix)]
            {
                use std::os::unix::fs::MetadataExt;
                let metadata = std::fs::metadata(&path).unwrap_or_else(|_| {
                    // If path doesn't exist, check parent
                    std::fs::metadata(path.parent().unwrap_or(&path)).unwrap()
                });

                // Get filesystem stats
                let output = std::process::Command::new("df")
                    .arg("-B1")
                    .arg(&path)
                    .output();

                if let Ok(output) = output {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    // Parse df output (second line, fourth column is available)
                    if let Some(line) = stdout.lines().nth(1) {
                        let parts: Vec<&str> = line.split_whitespace().collect();
                        if parts.len() >= 4 {
                            if let Ok(avail) = parts[3].parse::<u64>() {
                                return Ok(avail);
                            }
                        }
                    }
                }

                // Fallback: assume 100GB free
                Ok(100 * 1024 * 1024 * 1024)
            }

            #[cfg(not(unix))]
            {
                // Fallback for non-Unix: assume 100GB free
                Ok(100 * 1024 * 1024 * 1024)
            }
        })
        .await??;

        Ok(free_bytes / (1024 * 1024 * 1024))
    }

    /// Check if a path is an S3 path
    fn is_s3_path(&self, path: &Path) -> bool {
        path.to_string_lossy().starts_with("s3://")
    }

    /// Get the path where a specific block height should be stored/read from
    pub async fn get_path_for_height(&self, height: u64) -> PathBuf {
        let ranges = self.path_ranges.read().await;

        // Check if we have a recorded range for this height
        for (path, start, end) in ranges.iter() {
            if height >= *start && height <= *end {
                return path.clone();
            }
        }

        // Default to primary
        self.config.primary_path.clone()
    }

    /// Record that a height range is stored in a specific path
    pub async fn record_height_range(&self, path: PathBuf, start: u64, end: u64) {
        let mut ranges = self.path_ranges.write().await;
        ranges.push((path, start, end));
    }

    /// Archive old blocks to S3 (if configured)
    pub async fn archive_old_blocks_to_s3(
        &self,
        current_height: u64,
        export_fn: impl Fn(u64, u64) -> Result<Vec<u8>>,
    ) -> Result<Option<String>> {
        let bucket = match &self.config.s3_bucket {
            Some(b) => b,
            None => return Ok(None), // S3 not configured
        };

        let archive_threshold = current_height.saturating_sub(self.config.archive_after_blocks);
        if archive_threshold == 0 {
            return Ok(None);
        }

        // This is a placeholder - actual S3 upload would use aws-sdk-s3
        // For S3 to work well with blockchain:
        // 1. Use large objects (batch multiple blocks)
        // 2. Use multipart upload for reliability
        // 3. Never read individual blocks from S3 (too slow)
        // 4. Only use for archival/backup

        info!(
            "☁️  [S3] Would archive blocks 0-{} to s3://{}/{}",
            archive_threshold, bucket, self.config.s3_prefix
        );

        Ok(Some(format!(
            "s3://{}/{}/archive_0_{}.bin",
            bucket, self.config.s3_prefix, archive_threshold
        )))
    }

    /// Upload backup file to S3 (if configured)
    /// Returns S3 URL if successful
    pub async fn upload_backup_to_s3(&self, local_path: &Path, backup_name: &str) -> Result<Option<String>> {
        let bucket = match &self.config.s3_bucket {
            Some(b) if self.config.s3_backups_enabled => b,
            _ => return Ok(None),
        };

        let s3_key = format!("{}/backups/{}", self.config.s3_prefix, backup_name);

        info!("☁️  [S3] Uploading backup to s3://{}/{}", bucket, s3_key);

        // Read the file
        let data = tokio::fs::read(local_path).await
            .context("Failed to read backup file for S3 upload")?;

        // S3 upload using aws CLI (simpler than SDK for this use case)
        // In production, use aws-sdk-s3 crate
        let s3_url = format!("s3://{}/{}", bucket, s3_key);

        let output = tokio::process::Command::new("aws")
            .args([
                "s3", "cp",
                local_path.to_string_lossy().as_ref(),
                &s3_url,
                "--region", &self.config.s3_region,
            ])
            .output()
            .await;

        match output {
            Ok(output) if output.status.success() => {
                info!("✅ [S3] Backup uploaded: {}", s3_url);
                Ok(Some(s3_url))
            }
            Ok(output) => {
                let stderr = String::from_utf8_lossy(&output.stderr);
                warn!("⚠️  [S3] Upload failed: {}", stderr);
                Ok(None)
            }
            Err(e) => {
                warn!("⚠️  [S3] AWS CLI not available: {}", e);
                Ok(None)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_from_env() {
        std::env::set_var("Q_STORAGE_OVERFLOW_PATHS", "/tmp/overflow1,/tmp/overflow2");
        std::env::set_var("Q_STORAGE_PRIMARY_MIN_FREE_GB", "5");

        let config = OverflowConfig::from_env();

        assert_eq!(config.overflow_paths.len(), 2);
        assert_eq!(config.primary_min_free_gb, 5);

        std::env::remove_var("Q_STORAGE_OVERFLOW_PATHS");
        std::env::remove_var("Q_STORAGE_PRIMARY_MIN_FREE_GB");
    }
}
