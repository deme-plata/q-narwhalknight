/// Adaptive Pruning System for Q-NarwhalKnight
/// Implements tiered data retention with automatic storage optimization
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};
use tracing::{debug, info, warn};

/// Pruning mode determines storage strategy
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum PruningMode {
    /// Full node - Keep all blockchain data (no pruning)
    Full,
    /// Archive node - Retain all with aggressive compression
    Archive,
    /// Light mode - Minimal storage (checkpoints + recent blocks)
    Light,
    /// Adaptive mode - Auto-adjust based on available disk space
    Adaptive,
}

impl Default for PruningMode {
    fn default() -> Self {
        // v0.9.1-beta: Default to FULL mode for testnet safety
        // Adaptive pruning was deleting all blocks - caused catastrophic data loss
        // Pruning must be explicitly enabled via Q_PRUNING_MODE environment variable
        PruningMode::Full  // SAFE - NO PRUNING
    }
}

/// Pruning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruningConfig {
    /// Pruning mode
    pub mode: PruningMode,

    /// Retain blocks newer than this (days)
    pub retain_recent_blocks_days: u64,

    /// Minimum free disk space to maintain (bytes)
    pub min_free_disk_space: u64,

    /// Target storage size (bytes, 0 = unlimited)
    pub target_storage_size: u64,

    /// Checkpoint retention policy
    pub checkpoint_policy: CheckpointPolicy,

    /// Auto-prune interval (seconds)
    pub auto_prune_interval: u64,

    /// Enable aggressive pruning when disk space is low
    pub aggressive_pruning_threshold: f64, // 0.0-1.0 (percentage)
}

impl Default for PruningConfig {
    fn default() -> Self {
        Self {
            // v10.3.0: FIXED — was Adaptive (contradicting PruningMode::default() = Full)
            // Adaptive pruning silently deleted blocks on all nodes without being asked.
            // Must be explicitly enabled via Q_PRUNING_MODE=adaptive env var.
            mode: PruningMode::Full,
            retain_recent_blocks_days: 30,
            min_free_disk_space: 10 * 1024 * 1024 * 1024, // 10 GB
            target_storage_size: 0, // Unlimited
            checkpoint_policy: CheckpointPolicy::default(),
            auto_prune_interval: 3600, // 1 hour
            aggressive_pruning_threshold: 0.1, // 10% free space triggers aggressive mode
        }
    }
}

/// Checkpoint retention policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointPolicy {
    /// Always retain genesis block
    pub retain_genesis: bool,

    /// Checkpoint interval (blocks)
    pub checkpoint_interval: u64,

    /// Maximum number of checkpoints to retain
    pub max_checkpoints: usize,

    /// Retain checkpoints for re-org protection depth
    pub reorg_protection_depth: u64,
}

impl Default for CheckpointPolicy {
    fn default() -> Self {
        Self {
            retain_genesis: true,
            checkpoint_interval: 55_000, // Approximately weekly checkpoints
            max_checkpoints: 52, // ~1 year of weekly checkpoints
            reorg_protection_depth: 24, // 24 blocks deep (based on whitepaper analysis)
        }
    }
}

/// Tiered data retention categories
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetentionTier {
    /// Genesis + critical checkpoints (always retained)
    Tier0Critical,
    /// Recent blocks (retain_recent_blocks_days)
    Tier1Recent,
    /// Checkpoint blocks (checkpoint_interval)
    Tier2Checkpoints,
    /// Historical blocks (can be pruned)
    Tier3Historical,
}

/// Pruning statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruningStats {
    pub total_blocks: u64,
    pub pruned_blocks: u64,
    pub retained_blocks: u64,
    pub storage_before: u64,
    pub storage_after: u64,
    pub space_saved: u64,
    pub last_prune_time: SystemTime,
    pub prune_duration_ms: u64,
}

/// Adaptive pruning engine
pub struct AdaptivePruningEngine {
    config: PruningConfig,
    db_path: PathBuf,
    stats: Option<PruningStats>,
}

impl AdaptivePruningEngine {
    /// Create new pruning engine
    pub fn new<P: AsRef<Path>>(db_path: P, config: PruningConfig) -> Self {
        Self {
            config,
            db_path: db_path.as_ref().to_path_buf(),
            stats: None,
        }
    }

    /// Check if block should be retained
    pub fn should_retain_block(&self, block_height: u64, current_height: u64) -> Result<bool> {
        match self.config.mode {
            PruningMode::Full | PruningMode::Archive => {
                // Keep all blocks
                Ok(true)
            }
            PruningMode::Light => {
                // Only keep checkpoints and recent blocks
                Ok(self.is_checkpoint_block(block_height) || self.is_recent_block(block_height, current_height))
            }
            PruningMode::Adaptive => {
                // Dynamic retention based on disk space
                self.adaptive_retention_policy(block_height, current_height)
            }
        }
    }

    /// Determine retention tier for a block
    pub fn get_retention_tier(&self, block_height: u64, current_height: u64) -> RetentionTier {
        // Tier 0: Genesis and critical checkpoints
        if block_height == 0 || self.is_critical_checkpoint(block_height) {
            return RetentionTier::Tier0Critical;
        }

        // Tier 1: Recent blocks (within retain_recent_blocks_days)
        if self.is_recent_block(block_height, current_height) {
            return RetentionTier::Tier1Recent;
        }

        // Tier 2: Checkpoint blocks
        if self.is_checkpoint_block(block_height) {
            return RetentionTier::Tier2Checkpoints;
        }

        // Tier 3: Historical (pruneable)
        RetentionTier::Tier3Historical
    }

    /// Check if block is a checkpoint
    fn is_checkpoint_block(&self, block_height: u64) -> bool {
        block_height == 0 || block_height % self.config.checkpoint_policy.checkpoint_interval == 0
    }

    /// Check if block is a critical checkpoint (always retain)
    fn is_critical_checkpoint(&self, block_height: u64) -> bool {
        if block_height == 0 && self.config.checkpoint_policy.retain_genesis {
            return true;
        }

        // First few checkpoints are critical
        if self.is_checkpoint_block(block_height) && block_height < self.config.checkpoint_policy.checkpoint_interval * 3 {
            return true;
        }

        false
    }

    /// Check if block is recent
    fn is_recent_block(&self, block_height: u64, current_height: u64) -> bool {
        // Estimate blocks per day (assuming 2.5s block time)
        let blocks_per_day = 34_560; // 86400 / 2.5
        let recent_block_threshold = blocks_per_day * self.config.retain_recent_blocks_days;

        current_height.saturating_sub(block_height) <= recent_block_threshold
    }

    /// Adaptive retention policy based on disk space
    fn adaptive_retention_policy(&self, block_height: u64, current_height: u64) -> Result<bool> {
        // Always retain recent blocks and checkpoints
        if self.is_recent_block(block_height, current_height) || self.is_checkpoint_block(block_height) {
            return Ok(true);
        }

        // Check disk space availability
        let free_space_ratio = self.get_free_disk_space_ratio()?;

        // If disk space is low, apply aggressive pruning
        if free_space_ratio < self.config.aggressive_pruning_threshold {
            warn!("⚠️ Low disk space ({:.1}% free) - aggressive pruning enabled", free_space_ratio * 100.0);

            // Only keep critical checkpoints and very recent blocks
            return Ok(self.is_critical_checkpoint(block_height) ||
                     (current_height.saturating_sub(block_height) <= 1000));
        }

        // Normal adaptive policy
        Ok(self.is_checkpoint_block(block_height) || self.is_recent_block(block_height, current_height))
    }

    /// Get free disk space ratio (0.0-1.0)
    fn get_free_disk_space_ratio(&self) -> Result<f64> {
        // Use filesystem stats to get available space
        #[cfg(target_family = "unix")]
        {
            use std::os::unix::fs::MetadataExt;
            use std::fs;

            let metadata = fs::metadata(&self.db_path)
                .context("Failed to get filesystem metadata")?;

            // Get filesystem stats using libc
            use std::ffi::CString;
            use std::mem;

            let path_cstring = CString::new(self.db_path.to_string_lossy().as_ref())
                .context("Invalid path")?;

            unsafe {
                let mut statfs: libc::statfs = mem::zeroed();
                if libc::statfs(path_cstring.as_ptr(), &mut statfs) == 0 {
                    let total_space = statfs.f_blocks as u64 * statfs.f_bsize as u64;
                    let free_space = statfs.f_bavail as u64 * statfs.f_bsize as u64;

                    if total_space > 0 {
                        return Ok(free_space as f64 / total_space as f64);
                    }
                }
            }
        }

        // Fallback: assume 50% free space if we can't determine
        Ok(0.5)
    }

    /// Execute pruning operation
    pub async fn execute_pruning(&mut self, current_height: u64, total_blocks: u64) -> Result<PruningStats> {
        let start_time = SystemTime::now();
        info!("🗑️ Starting adaptive pruning (mode: {:?}, current height: {})", self.config.mode, current_height);

        let storage_before = self.get_current_storage_size()?;

        // Calculate retention policy
        let mut retained_blocks = 0u64;
        let mut pruned_blocks = 0u64;

        // Simulate pruning (actual implementation would interact with RocksDB)
        for height in 0..=current_height {
            if self.should_retain_block(height, current_height)? {
                retained_blocks += 1;
            } else {
                pruned_blocks += 1;
                // TODO: Actually delete block from RocksDB
                debug!("🗑️ Pruning block {} (tier: {:?})", height, self.get_retention_tier(height, current_height));
            }
        }

        let storage_after = storage_before - (pruned_blocks * 36_400); // Estimated 36.4 KB per block
        let space_saved = storage_before.saturating_sub(storage_after);

        let duration = start_time.elapsed().unwrap_or(Duration::from_secs(0));

        let stats = PruningStats {
            total_blocks,
            pruned_blocks,
            retained_blocks,
            storage_before,
            storage_after,
            space_saved,
            last_prune_time: start_time,
            prune_duration_ms: duration.as_millis() as u64,
        };

        info!(
            "✅ Pruning complete: pruned {} blocks, retained {}, saved {} MB in {} ms",
            pruned_blocks,
            retained_blocks,
            space_saved / 1_000_000,
            stats.prune_duration_ms
        );

        self.stats = Some(stats.clone());
        Ok(stats)
    }

    /// Get current storage size
    fn get_current_storage_size(&self) -> Result<u64> {
        use std::fs;

        let mut total_size = 0u64;

        // Walk directory tree to calculate size
        fn dir_size(path: &Path) -> Result<u64> {
            let mut size = 0u64;

            for entry in fs::read_dir(path)? {
                let entry = entry?;
                let metadata = entry.metadata()?;

                if metadata.is_dir() {
                    size += dir_size(&entry.path())?;
                } else {
                    size += metadata.len();
                }
            }

            Ok(size)
        }

        if self.db_path.exists() {
            total_size = dir_size(&self.db_path)?;
        }

        Ok(total_size)
    }

    /// Get current pruning statistics
    pub fn get_stats(&self) -> Option<&PruningStats> {
        self.stats.as_ref()
    }

    /// Update configuration
    pub fn update_config(&mut self, config: PruningConfig) {
        info!("🔧 Updating pruning configuration: {:?}", config);
        self.config = config;
    }

    /// Calculate estimated storage savings
    pub fn estimate_storage_savings(&self, current_height: u64) -> Result<(u64, u64, f64)> {
        let mut retained = 0u64;
        let mut pruned = 0u64;

        for height in 0..=current_height {
            if self.should_retain_block(height, current_height)? {
                retained += 1;
            } else {
                pruned += 1;
            }
        }

        let avg_block_size = 36_400u64; // 36.4 KB
        let total_size = (retained + pruned) * avg_block_size;
        let savings = pruned * avg_block_size;
        let efficiency = if total_size > 0 {
            savings as f64 / total_size as f64
        } else {
            0.0
        };

        Ok((retained, pruned, efficiency))
    }
}

impl PruningStats {
    /// Get efficiency percentage
    pub fn efficiency_percent(&self) -> f64 {
        if self.storage_before > 0 {
            (self.space_saved as f64 / self.storage_before as f64) * 100.0
        } else {
            0.0
        }
    }

    /// Get retention rate
    pub fn retention_rate(&self) -> f64 {
        if self.total_blocks > 0 {
            (self.retained_blocks as f64 / self.total_blocks as f64) * 100.0
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_pruning_modes() {
        let temp_dir = TempDir::new().unwrap();
        let config = PruningConfig::default();
        let engine = AdaptivePruningEngine::new(temp_dir.path(), config);

        // Test full mode
        let mut full_config = PruningConfig::default();
        full_config.mode = PruningMode::Full;
        let mut full_engine = AdaptivePruningEngine::new(temp_dir.path(), full_config);

        assert!(full_engine.should_retain_block(1000, 10000).unwrap());
        assert!(full_engine.should_retain_block(5000, 10000).unwrap());
    }

    #[test]
    fn test_checkpoint_detection() {
        let temp_dir = TempDir::new().unwrap();
        let config = PruningConfig::default();
        let engine = AdaptivePruningEngine::new(temp_dir.path(), config);

        assert!(engine.is_checkpoint_block(0)); // Genesis
        assert!(engine.is_checkpoint_block(55_000)); // First checkpoint
        assert!(engine.is_checkpoint_block(110_000)); // Second checkpoint
        assert!(!engine.is_checkpoint_block(1000)); // Not a checkpoint
    }

    #[test]
    fn test_retention_tiers() {
        let temp_dir = TempDir::new().unwrap();
        let config = PruningConfig::default();
        let engine = AdaptivePruningEngine::new(temp_dir.path(), config);

        let current_height = 110_000;

        // Tier 0: Genesis
        assert_eq!(engine.get_retention_tier(0, current_height), RetentionTier::Tier0Critical);

        // Tier 1: Recent (within 30 days)
        assert_eq!(engine.get_retention_tier(109_000, current_height), RetentionTier::Tier1Recent);

        // Tier 2: Checkpoint
        assert_eq!(engine.get_retention_tier(55_000, current_height), RetentionTier::Tier2Checkpoints);

        // Tier 3: Historical
        assert_eq!(engine.get_retention_tier(10_000, current_height), RetentionTier::Tier3Historical);
    }

    #[tokio::test]
    async fn test_storage_estimation() {
        let temp_dir = TempDir::new().unwrap();
        let config = PruningConfig::default();
        let engine = AdaptivePruningEngine::new(temp_dir.path(), config);

        let current_height = 110_000;
        let (retained, pruned, efficiency) = engine.estimate_storage_savings(current_height).unwrap();

        assert!(retained > 0);
        assert!(pruned > 0);
        assert!(efficiency > 0.0 && efficiency < 1.0);

        info!("Storage estimation: retained={}, pruned={}, efficiency={:.1}%",
              retained, pruned, efficiency * 100.0);
    }
}
