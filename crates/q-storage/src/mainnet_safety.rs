//! Mainnet Safety Infrastructure
//!
//! v1.1.24-beta: Production-grade database safety for mainnet deployment
//!
//! ## Components
//!
//! 1. **Enhanced Checkpoint System** - Hourly auto-checkpoints with 24-hour history
//! 2. **Background Integrity Monitor** - Continuous verification every 60 seconds
//! 3. **IPFS Backup System** - Incremental hourly backups with one-command restore
//! 4. **Pre-commit Integrity Checks** - Verify before every atomic commit
//!
//! ## Design Goals
//!
//! - Zero database corruptions (automated detection + repair)
//! - Point-in-time recovery (hourly checkpoints)
//! - Off-site backups (IPFS)
//! - One-command restore capability

use anyhow::{anyhow, Context, Result};
use hex;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

// ============================================================================
// ENHANCED CHECKPOINT SYSTEM
// ============================================================================

/// Enhanced checkpoint with full state information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedCheckpoint {
    /// Block height at checkpoint
    pub height: u64,
    /// Hash of the block at this height
    pub block_hash: String,
    /// State root (Merkle root of all balances)
    pub state_root: Option<String>,
    /// Timestamp when checkpoint was created
    pub timestamp: u64,
    /// Version of the software
    pub version: String,
    /// IPFS CID if backed up
    pub ipfs_cid: Option<String>,
    /// Checkpoint type (hourly, block-interval, manual)
    pub checkpoint_type: CheckpointType,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CheckpointType {
    /// Created every 1000 blocks
    BlockInterval,
    /// Created every hour
    Hourly,
    /// Created manually (before upgrade, etc.)
    Manual,
    /// Pre-deployment checkpoint
    PreDeployment,
}

/// Enhanced checkpoint manager with hourly auto-checkpointing
pub struct EnhancedCheckpointManager {
    /// Data directory
    data_dir: PathBuf,
    /// Recent checkpoints (keep last 24 for 1-day history)
    checkpoints: RwLock<VecDeque<EnhancedCheckpoint>>,
    /// Maximum checkpoints to keep
    max_checkpoints: usize,
    /// Last hourly checkpoint timestamp
    last_hourly_checkpoint: AtomicU64,
    /// Storage reference for reading blocks
    storage: Arc<dyn CheckpointStorage + Send + Sync>,
    /// Whether auto-checkpointing is running
    auto_checkpoint_running: AtomicBool,
}

/// Trait for checkpoint storage operations
#[async_trait::async_trait]
pub trait CheckpointStorage {
    async fn get_latest_height(&self) -> Result<u64>;
    async fn get_block_hash(&self, height: u64) -> Result<Option<String>>;
    async fn compute_state_root(&self) -> Result<Option<String>>;
    async fn export_blocks_range(&self, start: u64, end: u64) -> Result<Vec<u8>>;
    async fn import_blocks(&self, data: &[u8]) -> Result<u64>;
    async fn truncate_after_height(&self, height: u64) -> Result<()>;
}

impl EnhancedCheckpointManager {
    /// Create new enhanced checkpoint manager
    pub fn new(data_dir: PathBuf, storage: Arc<dyn CheckpointStorage + Send + Sync>) -> Self {
        let checkpoints_dir = data_dir.join("checkpoints");
        std::fs::create_dir_all(&checkpoints_dir).ok();

        // Load existing checkpoints
        let checkpoints = Self::load_checkpoints(&checkpoints_dir);

        info!(
            "📍 [CHECKPOINT] Initialized with {} existing checkpoints",
            checkpoints.len()
        );

        Self {
            data_dir,
            checkpoints: RwLock::new(checkpoints),
            max_checkpoints: 24, // 24 hours of hourly checkpoints
            last_hourly_checkpoint: AtomicU64::new(0),
            storage,
            auto_checkpoint_running: AtomicBool::new(false),
        }
    }

    /// Load existing checkpoints from disk
    fn load_checkpoints(dir: &PathBuf) -> VecDeque<EnhancedCheckpoint> {
        let mut checkpoints = VecDeque::new();

        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                if entry.path().extension().map_or(false, |e| e == "json") {
                    if let Ok(content) = std::fs::read_to_string(entry.path()) {
                        if let Ok(cp) = serde_json::from_str::<EnhancedCheckpoint>(&content) {
                            checkpoints.push_back(cp);
                        }
                    }
                }
            }
        }

        // Sort by height
        let mut vec: Vec<_> = checkpoints.into_iter().collect();
        vec.sort_by_key(|cp| cp.height);
        vec.into_iter().collect()
    }

    /// Start automatic hourly checkpointing
    pub async fn start_auto_checkpointing(self: Arc<Self>) {
        if self
            .auto_checkpoint_running
            .swap(true, Ordering::SeqCst)
        {
            warn!("⚠️ [CHECKPOINT] Auto-checkpointing already running");
            return;
        }

        info!("🕐 [CHECKPOINT] Starting automatic hourly checkpointing");

        let manager = self.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(3600)); // 1 hour

            loop {
                interval.tick().await;

                if let Err(e) = manager.create_checkpoint(CheckpointType::Hourly).await {
                    error!("❌ [CHECKPOINT] Hourly checkpoint failed: {}", e);
                }
            }
        });
    }

    /// Create a checkpoint
    pub async fn create_checkpoint(&self, checkpoint_type: CheckpointType) -> Result<EnhancedCheckpoint> {
        let height = self.storage.get_latest_height().await?;
        let block_hash = self
            .storage
            .get_block_hash(height)
            .await?
            .unwrap_or_else(|| "unknown".to_string());
        let state_root = self.storage.compute_state_root().await?;

        let checkpoint = EnhancedCheckpoint {
            height,
            block_hash: block_hash.clone(),
            state_root,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            ipfs_cid: None,
            checkpoint_type: checkpoint_type.clone(),
        };

        // Save to disk
        self.save_checkpoint(&checkpoint).await?;

        // Add to in-memory list
        let mut checkpoints = self.checkpoints.write().await;
        checkpoints.push_back(checkpoint.clone());

        // Prune old checkpoints
        while checkpoints.len() > self.max_checkpoints {
            if let Some(old) = checkpoints.pop_front() {
                self.delete_checkpoint_file(old.height).await.ok();
            }
        }

        info!(
            "📍 [CHECKPOINT] Created {:?} checkpoint at height {} (hash: {})",
            checkpoint_type,
            height,
            &block_hash[..16.min(block_hash.len())]
        );

        Ok(checkpoint)
    }

    /// Save checkpoint to disk atomically
    async fn save_checkpoint(&self, checkpoint: &EnhancedCheckpoint) -> Result<()> {
        let checkpoints_dir = self.data_dir.join("checkpoints");
        let filename = format!("checkpoint_{}.json", checkpoint.height);
        let path = checkpoints_dir.join(&filename);
        let temp_path = checkpoints_dir.join(format!("{}.tmp", filename));

        let content = serde_json::to_string_pretty(checkpoint)?;

        // Write to temp file first
        tokio::fs::write(&temp_path, &content).await?;

        // Sync to disk
        let file = tokio::fs::File::open(&temp_path).await?;
        file.sync_all().await?;

        // Atomic rename
        tokio::fs::rename(&temp_path, &path).await?;

        Ok(())
    }

    /// Delete checkpoint file
    async fn delete_checkpoint_file(&self, height: u64) -> Result<()> {
        let path = self
            .data_dir
            .join("checkpoints")
            .join(format!("checkpoint_{}.json", height));

        if path.exists() {
            tokio::fs::remove_file(path).await?;
        }

        Ok(())
    }

    /// Get the latest checkpoint
    pub async fn get_latest_checkpoint(&self) -> Option<EnhancedCheckpoint> {
        let checkpoints = self.checkpoints.read().await;
        checkpoints.back().cloned()
    }

    /// Get checkpoint at or before a specific height
    pub async fn get_checkpoint_at_or_before(&self, height: u64) -> Option<EnhancedCheckpoint> {
        let checkpoints = self.checkpoints.read().await;
        checkpoints
            .iter()
            .filter(|cp| cp.height <= height)
            .max_by_key(|cp| cp.height)
            .cloned()
    }

    /// List all checkpoints
    pub async fn list_checkpoints(&self) -> Vec<EnhancedCheckpoint> {
        let checkpoints = self.checkpoints.read().await;
        checkpoints.iter().cloned().collect()
    }

    /// Restore to a checkpoint
    pub async fn restore_to_checkpoint(&self, height: u64) -> Result<()> {
        let checkpoint = self
            .get_checkpoint_at_or_before(height)
            .await
            .ok_or_else(|| anyhow!("No checkpoint found at or before height {}", height))?;

        warn!(
            "🔄 [RESTORE] Rolling back to checkpoint at height {}",
            checkpoint.height
        );

        // Truncate blocks after checkpoint
        self.storage
            .truncate_after_height(checkpoint.height)
            .await?;

        info!(
            "✅ [RESTORE] Successfully restored to height {}",
            checkpoint.height
        );

        Ok(())
    }
}

// ============================================================================
// BACKGROUND INTEGRITY MONITOR
// ============================================================================

/// Background integrity monitor with continuous verification
pub struct BackgroundIntegrityMonitor {
    /// Storage reference
    storage: Arc<dyn IntegrityCheckable + Send + Sync>,
    /// Check interval
    check_interval: Duration,
    /// Auto-repair enabled
    auto_repair: bool,
    /// Whether monitor is running
    running: AtomicBool,
    /// Last check timestamp
    last_check: AtomicU64,
    /// Issues found count
    issues_found: AtomicU64,
    /// v10.5.0: Gap-fill trigger channel.
    /// When BlockGaps are detected, sends (first_gap_height, last_gap_height) to this sender.
    /// The TurboSyncManager or main loop owns the receiver and dispatches a targeted fetch.
    /// Uses OnceLock so it can be set after Arc<BackgroundIntegrityMonitor> is created.
    pub gap_fill_tx: std::sync::OnceLock<tokio::sync::mpsc::UnboundedSender<(u64, u64)>>,
}

/// Trait for integrity checking operations
#[async_trait::async_trait]
pub trait IntegrityCheckable {
    async fn get_pointer_height(&self) -> Result<u64>;
    async fn scan_actual_highest_block(&self) -> Result<u64>;
    async fn find_block_gaps(&self, start: u64, end: u64) -> Result<Vec<u64>>;
    async fn verify_parent_chain(&self, height: u64) -> Result<Vec<u64>>;
    async fn fix_pointer(&self, correct_height: u64) -> Result<()>;
}

/// Integrity check report
#[derive(Debug, Clone)]
pub struct IntegrityCheckReport {
    pub is_healthy: bool,
    pub pointer_height: u64,
    pub actual_height: u64,
    pub gaps: Vec<u64>,
    pub broken_parent_links: Vec<u64>,
    pub issues: Vec<IntegrityIssue>,
    pub check_duration_ms: u64,
}

#[derive(Debug, Clone)]
pub enum IntegrityIssue {
    PointerMismatch { claimed: u64, actual: u64 },
    BlockGaps { count: usize, first_gap: u64 },
    BrokenParentChain { broken_links: Vec<u64> },
}

impl BackgroundIntegrityMonitor {
    /// Create new background integrity monitor
    pub fn new(
        storage: Arc<dyn IntegrityCheckable + Send + Sync>,
        check_interval: Duration,
        auto_repair: bool,
    ) -> Self {
        Self {
            storage,
            check_interval,
            auto_repair,
            running: AtomicBool::new(false),
            last_check: AtomicU64::new(0),
            issues_found: AtomicU64::new(0),
            gap_fill_tx: std::sync::OnceLock::new(),
        }
    }

    /// v10.5.0: Wire a gap-fill channel so auto-repair can actually trigger fetches.
    /// Safe to call after Arc<BackgroundIntegrityMonitor> is created — uses OnceLock.
    pub fn set_gap_fill_channel(&self, tx: tokio::sync::mpsc::UnboundedSender<(u64, u64)>) {
        let _ = self.gap_fill_tx.set(tx);
    }

    /// Start background monitoring
    pub async fn start(self: Arc<Self>) {
        if self.running.swap(true, Ordering::SeqCst) {
            warn!("⚠️ [INTEGRITY] Monitor already running");
            return;
        }

        info!(
            "🔍 [INTEGRITY] Starting background monitor (interval: {:?}, auto-repair: {})",
            self.check_interval, self.auto_repair
        );

        let monitor = self.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(monitor.check_interval);

            loop {
                interval.tick().await;

                match monitor.run_integrity_check().await {
                    Ok(report) => {
                        if !report.is_healthy {
                            error!(
                                "🚨 [INTEGRITY] Issues detected: {} issues found",
                                report.issues.len()
                            );

                            if monitor.auto_repair {
                                monitor.auto_repair_issues(&report).await;
                            }
                        } else {
                            debug!(
                                "✅ [INTEGRITY] Check passed (pointer: {}, duration: {}ms)",
                                report.pointer_height, report.check_duration_ms
                            );
                        }
                    }
                    Err(e) => {
                        error!("❌ [INTEGRITY] Check failed: {}", e);
                    }
                }
            }
        });
    }

    /// Run a single integrity check
    pub async fn run_integrity_check(&self) -> Result<IntegrityCheckReport> {
        let start = std::time::Instant::now();
        let mut issues = Vec::new();

        // Check 1: Pointer matches actual highest block
        let pointer_height = self.storage.get_pointer_height().await?;
        let actual_height = self.storage.scan_actual_highest_block().await?;

        if pointer_height != actual_height {
            issues.push(IntegrityIssue::PointerMismatch {
                claimed: pointer_height,
                actual: actual_height,
            });
        }

        // Check 2: No gaps in block chain (sample check for performance)
        // v7.3.7: Start from max(1, height-1000) — height 0 never exists (genesis is at height 1+)
        // Scanning from 0 when height < 1000 causes permanent false "gap at 0" → spurious P2P syncs
        let check_range = actual_height.saturating_sub(1000).max(1);
        let gaps = self
            .storage
            .find_block_gaps(check_range, actual_height)
            .await?;

        if !gaps.is_empty() {
            issues.push(IntegrityIssue::BlockGaps {
                count: gaps.len(),
                first_gap: gaps[0],
            });
        }

        // Check 3: Parent chain valid (sample check)
        let broken_links = self.storage.verify_parent_chain(actual_height).await?;

        if !broken_links.is_empty() {
            issues.push(IntegrityIssue::BrokenParentChain {
                broken_links: broken_links.clone(),
            });
        }

        let is_healthy = issues.is_empty();
        let check_duration_ms = start.elapsed().as_millis() as u64;

        // Update stats
        self.last_check.store(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            Ordering::SeqCst,
        );

        if !is_healthy {
            self.issues_found.fetch_add(1, Ordering::SeqCst);
        }

        Ok(IntegrityCheckReport {
            is_healthy,
            pointer_height,
            actual_height,
            gaps,
            broken_parent_links: broken_links,
            issues,
            check_duration_ms,
        })
    }

    /// Auto-repair detected issues
    async fn auto_repair_issues(&self, report: &IntegrityCheckReport) {
        for issue in &report.issues {
            match issue {
                IntegrityIssue::PointerMismatch { actual, .. } => {
                    warn!(
                        "🔧 [AUTO-REPAIR] Fixing pointer mismatch: setting to {}",
                        actual
                    );
                    if let Err(e) = self.storage.fix_pointer(*actual).await {
                        error!("❌ [AUTO-REPAIR] Failed to fix pointer: {}", e);
                    } else {
                        info!("✅ [AUTO-REPAIR] Pointer fixed successfully");
                    }
                }
                IntegrityIssue::BlockGaps { count, first_gap } => {
                    // v10.5.0: Actually trigger the gap fill instead of just logging.
                    // Scan for the last gap height so we pass a meaningful range.
                    let last_gap = report.gaps.last().copied().unwrap_or(*first_gap);
                    warn!(
                        "⚠️ [AUTO-REPAIR] {} gaps detected ({}-{}), dispatching gap-fill fetch",
                        count, first_gap, last_gap
                    );
                    if let Some(tx) = self.gap_fill_tx.get() {
                        if let Err(e) = tx.send((*first_gap, last_gap)) {
                            error!("❌ [AUTO-REPAIR] Gap-fill channel send failed: {}", e);
                        } else {
                            info!("📨 [AUTO-REPAIR] Gap-fill request sent for {}-{}", first_gap, last_gap);
                        }
                    } else {
                        warn!("⚠️ [AUTO-REPAIR] No gap-fill channel wired — gaps will persist until restart. \
                               Call set_gap_fill_channel() during setup.");
                    }
                }
                IntegrityIssue::BrokenParentChain { .. } => {
                    error!("🚨 [AUTO-REPAIR] Broken parent chain detected - manual intervention required");
                }
            }
        }
    }

    /// Get monitor status
    pub fn status(&self) -> MonitorStatus {
        MonitorStatus {
            running: self.running.load(Ordering::SeqCst),
            last_check: self.last_check.load(Ordering::SeqCst),
            issues_found: self.issues_found.load(Ordering::SeqCst),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MonitorStatus {
    pub running: bool,
    pub last_check: u64,
    pub issues_found: u64,
}

// ============================================================================
// PRE-COMMIT INTEGRITY CHECK
// ============================================================================

/// Pre-commit integrity check for atomic transactions
pub struct PreCommitIntegrityCheck {
    /// Checks to run before commit
    checks: Vec<Box<dyn Fn() -> Result<()> + Send + Sync>>,
}

impl PreCommitIntegrityCheck {
    pub fn new() -> Self {
        Self { checks: Vec::new() }
    }

    /// Add a check to run before commit
    pub fn add_check<F>(&mut self, check: F)
    where
        F: Fn() -> Result<()> + Send + Sync + 'static,
    {
        self.checks.push(Box::new(check));
    }

    /// Run all pre-commit checks
    pub fn run_checks(&self) -> Result<()> {
        for (i, check) in self.checks.iter().enumerate() {
            check().with_context(|| format!("Pre-commit check {} failed", i))?;
        }
        Ok(())
    }
}

impl Default for PreCommitIntegrityCheck {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// IPFS BACKUP SYSTEM
// ============================================================================

/// IPFS backup manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupManifest {
    pub start_height: u64,
    pub end_height: u64,
    pub ipfs_cid: String,
    pub timestamp: u64,
    pub size_bytes: usize,
    pub blocks_count: u64,
    pub checksum: String,
}

/// IPFS backup system for off-site storage
pub struct IpfsBackupSystem {
    /// Data directory
    data_dir: PathBuf,
    /// Storage reference
    storage: Arc<dyn CheckpointStorage + Send + Sync>,
    /// IPFS client (using q-ipfs-storage)
    ipfs_enabled: bool,
    /// Last backup height
    last_backup_height: AtomicU64,
    /// Backup manifests
    manifests: RwLock<Vec<BackupManifest>>,
}

impl IpfsBackupSystem {
    /// Create new IPFS backup system
    pub fn new(data_dir: PathBuf, storage: Arc<dyn CheckpointStorage + Send + Sync>) -> Self {
        let manifests_path = data_dir.join("backup_manifests.json");
        let manifests = Self::load_manifests(&manifests_path);

        let last_height = manifests.iter().map(|m| m.end_height).max().unwrap_or(0);

        Self {
            data_dir,
            storage,
            ipfs_enabled: false, // Will be enabled when IPFS is available
            last_backup_height: AtomicU64::new(last_height),
            manifests: RwLock::new(manifests),
        }
    }

    /// Load backup manifests from disk
    fn load_manifests(path: &PathBuf) -> Vec<BackupManifest> {
        if path.exists() {
            if let Ok(content) = std::fs::read_to_string(path) {
                if let Ok(manifests) = serde_json::from_str(&content) {
                    return manifests;
                }
            }
        }
        Vec::new()
    }

    /// Save manifests to disk
    async fn save_manifests(&self) -> Result<()> {
        let manifests = self.manifests.read().await;
        let path = self.data_dir.join("backup_manifests.json");
        let content = serde_json::to_string_pretty(&*manifests)?;
        tokio::fs::write(path, content).await?;
        Ok(())
    }

    /// Start automatic hourly backups
    pub async fn start_auto_backup(self: Arc<Self>) {
        info!("📦 [BACKUP] Starting automatic hourly backup system");

        let backup_system = self.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(3600)); // 1 hour

            loop {
                interval.tick().await;

                match backup_system.create_incremental_backup().await {
                    Ok(Some(manifest)) => {
                        info!(
                            "✅ [BACKUP] Incremental backup created: {} blocks ({} bytes)",
                            manifest.blocks_count, manifest.size_bytes
                        );
                    }
                    Ok(None) => {
                        debug!("📦 [BACKUP] No new blocks to backup");
                    }
                    Err(e) => {
                        error!("❌ [BACKUP] Failed: {}", e);
                    }
                }
            }
        });
    }

    /// Create incremental backup (only new blocks since last backup)
    pub async fn create_incremental_backup(&self) -> Result<Option<BackupManifest>> {
        let last_backup_height = self.last_backup_height.load(Ordering::SeqCst);
        let current_height = self.storage.get_latest_height().await?;

        if current_height <= last_backup_height {
            return Ok(None); // No new blocks
        }

        info!(
            "📦 [BACKUP] Creating incremental backup: {} -> {} ({} blocks)",
            last_backup_height,
            current_height,
            current_height - last_backup_height
        );

        // Export blocks
        let blocks_data = self
            .storage
            .export_blocks_range(last_backup_height + 1, current_height)
            .await?;

        // Calculate checksum
        let checksum = hex::encode(blake3::hash(&blocks_data).as_bytes());

        // Save locally first
        let backup_filename = format!(
            "backup_{}_{}.bin",
            last_backup_height + 1,
            current_height
        );
        let backup_path = self.data_dir.join("backups").join(&backup_filename);
        std::fs::create_dir_all(backup_path.parent().unwrap())?;
        tokio::fs::write(&backup_path, &blocks_data).await?;

        // Create manifest (IPFS CID would be added if IPFS is enabled)
        let manifest = BackupManifest {
            start_height: last_backup_height + 1,
            end_height: current_height,
            ipfs_cid: format!("local://{}", backup_filename), // Local path for now
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            size_bytes: blocks_data.len(),
            blocks_count: current_height - last_backup_height,
            checksum,
        };

        // Save manifest
        {
            let mut manifests = self.manifests.write().await;
            manifests.push(manifest.clone());
        }
        self.save_manifests().await?;

        // Update last backup height
        self.last_backup_height
            .store(current_height, Ordering::SeqCst);

        // v3.9.3-beta: Automatic backup cleanup to prevent disk bloat
        if let Err(e) = self.cleanup_old_backups().await {
            warn!("⚠️  [BACKUP] Cleanup failed: {}", e);
        }

        Ok(Some(manifest))
    }

    /// v3.9.3-beta: Automatically clean up old backups to prevent disk bloat
    /// Keeps backups from last 24 hours + one backup per day for last 7 days
    async fn cleanup_old_backups(&self) -> Result<()> {
        let backups_dir = self.data_dir.join("backups");

        if !backups_dir.exists() {
            return Ok(());
        }

        let now = SystemTime::now();
        let one_day = Duration::from_secs(24 * 60 * 60);
        let seven_days = Duration::from_secs(7 * 24 * 60 * 60);

        let mut entries: Vec<(std::path::PathBuf, SystemTime)> = Vec::new();

        // Collect all backup files with their modification times
        let mut dir = tokio::fs::read_dir(&backups_dir).await?;
        while let Some(entry) = dir.next_entry().await? {
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "bin") {
                if let Ok(metadata) = entry.metadata().await {
                    if let Ok(modified) = metadata.modified() {
                        entries.push((path, modified));
                    }
                }
            }
        }

        if entries.is_empty() {
            return Ok(());
        }

        // Sort by modification time (newest first)
        entries.sort_by(|a, b| b.1.cmp(&a.1));

        let mut deleted_count = 0;
        let mut freed_bytes: u64 = 0;
        let mut kept_days: std::collections::HashSet<u64> = std::collections::HashSet::new();

        for (path, modified) in entries {
            let age = now.duration_since(modified).unwrap_or(Duration::ZERO);

            // Keep all backups from last 24 hours
            if age < one_day {
                continue;
            }

            // For backups older than 24 hours, keep one per day (up to 7 days)
            if age < seven_days {
                let day_number = age.as_secs() / (24 * 60 * 60);
                if !kept_days.contains(&day_number) {
                    kept_days.insert(day_number);
                    continue; // Keep this backup (first one for this day)
                }
            }

            // Delete this backup (older than 7 days, or duplicate for a day)
            if let Ok(metadata) = tokio::fs::metadata(&path).await {
                freed_bytes += metadata.len();
            }

            if let Err(e) = tokio::fs::remove_file(&path).await {
                warn!("⚠️  [BACKUP] Failed to delete {:?}: {}", path, e);
            } else {
                deleted_count += 1;
            }
        }

        if deleted_count > 0 {
            info!(
                "🧹 [BACKUP] Cleaned up {} old backups, freed {:.2} GB",
                deleted_count,
                freed_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
            );
        }

        // Also clean up manifests for deleted backups
        self.cleanup_orphaned_manifests().await?;

        Ok(())
    }

    /// Remove manifests for backups that no longer exist
    async fn cleanup_orphaned_manifests(&self) -> Result<()> {
        let backups_dir = self.data_dir.join("backups");
        let mut manifests = self.manifests.write().await;
        let original_count = manifests.len();

        manifests.retain(|m| {
            if m.ipfs_cid.starts_with("local://") {
                let filename = &m.ipfs_cid[8..];
                let path = backups_dir.join(filename);
                path.exists()
            } else {
                true // Keep IPFS/S3 manifests
            }
        });

        let removed = original_count - manifests.len();
        if removed > 0 {
            debug!("🧹 [BACKUP] Removed {} orphaned manifests", removed);
        }

        drop(manifests);
        self.save_manifests().await?;

        Ok(())
    }

    /// Restore from backup to a specific height
    pub async fn restore_to_height(&self, target_height: u64) -> Result<()> {
        let manifests = self.manifests.read().await;

        // Find all manifests needed to reach target height
        let mut needed_manifests: Vec<_> = manifests
            .iter()
            .filter(|m| m.start_height <= target_height)
            .cloned()
            .collect();

        needed_manifests.sort_by_key(|m| m.start_height);

        info!(
            "📥 [RESTORE] Restoring to height {} using {} backup(s)",
            target_height,
            needed_manifests.len()
        );

        for manifest in needed_manifests {
            // Load from local backup (or IPFS if enabled)
            let backup_path = if manifest.ipfs_cid.starts_with("local://") {
                self.data_dir
                    .join("backups")
                    .join(&manifest.ipfs_cid[8..])
            } else {
                // Would fetch from IPFS here
                return Err(anyhow!("IPFS restore not yet implemented"));
            };

            let data = tokio::fs::read(&backup_path).await?;

            // Verify checksum
            let checksum = hex::encode(blake3::hash(&data).as_bytes());
            if checksum != manifest.checksum {
                return Err(anyhow!(
                    "Backup checksum mismatch: expected {}, got {}",
                    manifest.checksum,
                    checksum
                ));
            }

            // Import blocks
            let imported = self.storage.import_blocks(&data).await?;
            info!(
                "📥 [RESTORE] Imported {} blocks from backup {}",
                imported, manifest.ipfs_cid
            );
        }

        info!("✅ [RESTORE] Successfully restored to height {}", target_height);
        Ok(())
    }
}

// ============================================================================
// MAINNET SAFETY MANAGER (Orchestrator)
// ============================================================================

/// Mainnet Safety Manager - orchestrates all safety components
pub struct MainnetSafetyManager {
    /// Enhanced checkpoint manager
    pub checkpoint_manager: Arc<EnhancedCheckpointManager>,
    /// Background integrity monitor
    pub integrity_monitor: Arc<BackgroundIntegrityMonitor>,
    /// IPFS backup system
    pub backup_system: Arc<IpfsBackupSystem>,
    /// Running state
    running: AtomicBool,
}

impl MainnetSafetyManager {
    /// Create new mainnet safety manager (all components)
    pub fn new<S>(data_dir: PathBuf, storage: Arc<S>) -> Self
    where
        S: CheckpointStorage + IntegrityCheckable + Send + Sync + 'static,
    {
        let checkpoint_manager = Arc::new(EnhancedCheckpointManager::new(
            data_dir.clone(),
            storage.clone(),
        ));

        let integrity_monitor = Arc::new(BackgroundIntegrityMonitor::new(
            storage.clone(),
            Duration::from_secs(60), // Check every 60 seconds
            true,                    // Auto-repair enabled
        ));

        let backup_system = Arc::new(IpfsBackupSystem::new(data_dir, storage));

        Self {
            checkpoint_manager,
            integrity_monitor,
            backup_system,
            running: AtomicBool::new(false),
        }
    }

    /// Start all safety systems
    pub async fn start(&self) -> Result<()> {
        if self.running.swap(true, Ordering::SeqCst) {
            return Err(anyhow!("Safety systems already running"));
        }

        info!("🛡️ [MAINNET-SAFETY] Starting all safety systems...");

        // Start auto-checkpointing
        self.checkpoint_manager.clone().start_auto_checkpointing().await;

        // Start integrity monitoring
        self.integrity_monitor.clone().start().await;

        // Start auto-backup
        self.backup_system.clone().start_auto_backup().await;

        info!("✅ [MAINNET-SAFETY] All safety systems started");
        info!("   📍 Hourly checkpoints: ACTIVE");
        info!("   🔍 Integrity monitor: ACTIVE (60s interval)");
        info!("   📦 Auto-backup: ACTIVE (hourly)");

        Ok(())
    }

    /// Create pre-deployment checkpoint
    pub async fn create_pre_deployment_checkpoint(&self) -> Result<EnhancedCheckpoint> {
        info!("📍 [MAINNET-SAFETY] Creating pre-deployment checkpoint...");
        self.checkpoint_manager
            .create_checkpoint(CheckpointType::PreDeployment)
            .await
    }

    /// Emergency restore to last known good state
    pub async fn emergency_restore(&self) -> Result<()> {
        warn!("🚨 [MAINNET-SAFETY] Emergency restore initiated!");

        // Get latest checkpoint
        let checkpoint = self
            .checkpoint_manager
            .get_latest_checkpoint()
            .await
            .ok_or_else(|| anyhow!("No checkpoint available for restore"))?;

        info!(
            "📍 [RESTORE] Restoring to checkpoint at height {}",
            checkpoint.height
        );

        // Restore
        self.checkpoint_manager
            .restore_to_checkpoint(checkpoint.height)
            .await?;

        info!("✅ [RESTORE] Emergency restore complete");
        Ok(())
    }

    /// Get overall safety status
    pub async fn status(&self) -> SafetyStatus {
        let monitor_status = self.integrity_monitor.status();
        let latest_checkpoint = self.checkpoint_manager.get_latest_checkpoint().await;

        SafetyStatus {
            systems_running: self.running.load(Ordering::SeqCst),
            integrity_monitor: monitor_status,
            latest_checkpoint_height: latest_checkpoint.map(|c| c.height),
            last_backup_height: self.backup_system.last_backup_height.load(Ordering::SeqCst),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SafetyStatus {
    pub systems_running: bool,
    pub integrity_monitor: MonitorStatus,
    pub latest_checkpoint_height: Option<u64>,
    pub last_backup_height: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_type_serialization() {
        let checkpoint = EnhancedCheckpoint {
            height: 1000,
            block_hash: "abc123".to_string(),
            state_root: Some("def456".to_string()),
            timestamp: 1234567890,
            version: "1.1.24-beta".to_string(),
            ipfs_cid: None,
            checkpoint_type: CheckpointType::Hourly,
        };

        let json = serde_json::to_string(&checkpoint).unwrap();
        let decoded: EnhancedCheckpoint = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.height, 1000);
        assert_eq!(decoded.checkpoint_type, CheckpointType::Hourly);
    }
}
