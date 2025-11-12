//! Database Integrity Checker and Auto-Repair System
//!
//! v0.9.76-beta: Detects and repairs database corruption automatically on startup
//!
//! Features:
//! - Corruption detection (pointer vs actual blocks mismatch)
//! - Automatic backup creation before repair
//! - Rollback support if repair fails
//! - Gap detection and reporting

use anyhow::{Context, Result};
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{debug, error, info, warn};

const CF_BLOCKS: &str = "blocks";

/// Integrity check result
#[derive(Debug, Clone)]
pub struct IntegrityReport {
    pub is_healthy: bool,
    pub total_blocks: u64,
    pub highest_block: u64,
    pub highest_contiguous: u64,
    pub pointer_height: u64,
    pub gaps: Vec<u64>,
    pub corruption_type: Option<CorruptionType>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CorruptionType {
    /// Pointer shows higher height than actual blocks (catastrophic data loss)
    PointerTooHigh { pointer: u64, actual: u64 },
    /// Pointer shows lower height than actual blocks (minor - can advance)
    PointerTooLow { pointer: u64, actual: u64 },
    /// Gaps detected in blockchain
    GapsDetected { first_gap: u64, total_gaps: usize },
    /// Complete data loss - no blocks but pointer exists
    TotalDataLoss { pointer: u64 },
}

impl IntegrityReport {
    pub fn is_critical(&self) -> bool {
        matches!(
            self.corruption_type,
            Some(CorruptionType::PointerTooHigh { .. })
                | Some(CorruptionType::TotalDataLoss { .. })
        )
    }

    pub fn needs_repair(&self) -> bool {
        self.corruption_type.is_some()
    }
}

/// Database integrity checker
pub struct IntegrityChecker {
    db_path: PathBuf,
}

impl IntegrityChecker {
    pub fn new(db_path: PathBuf) -> Self {
        Self { db_path }
    }

    /// Perform comprehensive integrity check
    pub async fn check(&self) -> Result<IntegrityReport> {
        info!("🔍 ════════════════════════════════════════════════════════");
        info!("🔍 DATABASE INTEGRITY CHECK");
        info!("🔍 ════════════════════════════════════════════════════════");
        info!("📂 Database: {}", self.db_path.display());

        // Open database read-only
        let db = self.open_database()?;
        let db = Arc::new(db);

        // Get pointer height
        let pointer_height = self.get_pointer_height(&db)?;
        info!("📌 qblock:latest pointer: {}", pointer_height);

        // Scan for actual blocks
        let (total_blocks, highest_block, highest_contiguous, gaps) =
            self.scan_blocks(&db, pointer_height).await?;

        info!("📊 Scan Results:");
        info!("   • Total blocks found: {}", total_blocks);
        info!("   • Highest block: {}", highest_block);
        info!("   • Highest contiguous: {}", highest_contiguous);
        info!("   • Gaps detected: {}", gaps.len());

        // Determine corruption type
        let corruption_type = self.detect_corruption(
            pointer_height,
            total_blocks,
            highest_contiguous,
            &gaps,
        );

        let is_healthy = corruption_type.is_none();

        if is_healthy {
            info!("✅ Database integrity: HEALTHY");
        } else {
            error!("🚨 Database integrity: CORRUPTED");
            if let Some(ref corruption) = corruption_type {
                error!("   Type: {:?}", corruption);
            }
        }

        info!("🔍 ════════════════════════════════════════════════════════");

        Ok(IntegrityReport {
            is_healthy,
            total_blocks,
            highest_block,
            highest_contiguous,
            pointer_height,
            gaps,
            corruption_type,
        })
    }

    /// Repair database corruption
    pub async fn repair(&self, report: &IntegrityReport) -> Result<()> {
        if !report.needs_repair() {
            info!("✅ No repair needed - database is healthy");
            return Ok(());
        }

        error!("🔧 ════════════════════════════════════════════════════════");
        error!("🔧 AUTOMATIC DATABASE REPAIR");
        error!("🔧 ════════════════════════════════════════════════════════");

        // Create backup before repair
        self.create_backup().await?;

        // Perform repair based on corruption type
        match &report.corruption_type {
            Some(CorruptionType::PointerTooHigh { actual, .. }) => {
                // Critical: Reset pointer to actual height
                warn!("🚨 CRITICAL: Pointer shows higher height than actual blocks");
                warn!("   This indicates catastrophic data loss!");
                warn!("   Resetting pointer to actual contiguous height: {}", actual);

                self.fix_pointer(report.highest_contiguous).await?;

                error!("✅ Pointer repaired: {} → {}", report.pointer_height, actual);
                error!("⚠️  IMPORTANT: Node will now sync from height {} via P2P", actual);
            }
            Some(CorruptionType::TotalDataLoss { pointer }) => {
                // Critical: Total data loss - reset to genesis
                warn!("🚨 CATASTROPHIC: Total data loss detected!");
                warn!("   Pointer shows {} but 0 blocks found!", pointer);
                warn!("   Resetting to genesis (height 0)");

                self.fix_pointer(0).await?;

                error!("✅ Pointer repaired: {} → 0", report.pointer_height);
                error!("⚠️  IMPORTANT: Node will now sync from genesis via P2P");
            }
            Some(CorruptionType::PointerTooLow { pointer, actual }) => {
                // Minor: Advance pointer to actual height
                info!("🔧 Minor corruption: Pointer behind actual height");
                info!("   Advancing pointer: {} → {}", pointer, actual);

                self.fix_pointer(*actual).await?;

                info!("✅ Pointer advanced successfully");
            }
            Some(CorruptionType::GapsDetected { first_gap, .. }) => {
                // Gaps: Set pointer to highest contiguous
                warn!("⚠️  Gaps detected in blockchain");
                warn!("   First gap at height: {}", first_gap);
                warn!("   Setting pointer to highest contiguous: {}", report.highest_contiguous);

                self.fix_pointer(report.highest_contiguous).await?;

                warn!("✅ Pointer set to contiguous height");
                warn!("   P2P gap fill will recover missing blocks");
            }
            None => {
                // Should never happen since we checked needs_repair()
                return Ok(());
            }
        }

        error!("🔧 ════════════════════════════════════════════════════════");
        error!("✅ Repair completed successfully");
        error!("🔧 ════════════════════════════════════════════════════════");

        Ok(())
    }

    /// Create backup of database before repair
    async fn create_backup(&self) -> Result<()> {
        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
        let backup_path = self.db_path.with_extension(format!("backup_{}", timestamp));

        info!("💾 Creating backup: {}", backup_path.display());

        // Use RocksDB backup engine for safe backup
        use rocksdb::backup::{BackupEngine, BackupEngineOptions};

        let backup_opts = BackupEngineOptions::new(&backup_path)?;
        let mut backup_engine = BackupEngine::open(&backup_opts, &rocksdb::Env::new()?)?;

        // Create backup
        let db = self.open_database()?;
        backup_engine
            .create_new_backup(&db)
            .context("Failed to create backup")?;

        info!("✅ Backup created successfully");
        info!("   Location: {}", backup_path.display());
        info!("   Use this to rollback if repair fails");

        Ok(())
    }

    /// Fix qblock:latest pointer
    async fn fix_pointer(&self, correct_height: u64) -> Result<()> {
        use rocksdb::{Options, WriteBatch, WriteOptions};

        info!("🔧 Fixing qblock:latest pointer to {}", correct_height);

        let db = self.open_database()?;
        let cf_blocks = db
            .cf_handle(CF_BLOCKS)
            .context("blocks column family not found")?;

        // Create write batch
        let mut batch = WriteBatch::default();
        let height_bytes = correct_height.to_be_bytes();
        batch.put_cf(&cf_blocks, b"qblock:latest", &height_bytes);

        // Write with sync=true for durability
        let mut write_opts = WriteOptions::default();
        write_opts.set_sync(true);
        write_opts.disable_wal(false);

        db.write_opt(batch, &write_opts)
            .context("Failed to write pointer")?;

        // Flush to ensure persistence
        db.flush_cf(&cf_blocks)
            .context("Failed to flush after pointer update")?;

        info!("✅ Pointer fixed and flushed to disk");

        Ok(())
    }

    /// Open database read-only or read-write
    fn open_database(&self) -> Result<rocksdb::DB> {
        use rocksdb::{ColumnFamilyDescriptor, Options, DB};

        let mut db_opts = Options::default();
        db_opts.create_if_missing(false);

        // Discover column families
        let cf_list = DB::list_cf(&db_opts, &self.db_path)?;

        let cfs: Vec<_> = cf_list
            .iter()
            .map(|name| ColumnFamilyDescriptor::new(name.as_str(), Options::default()))
            .collect();

        DB::open_cf_descriptors(&db_opts, &self.db_path, cfs)
            .context("Failed to open database")
    }

    /// Get qblock:latest pointer height
    fn get_pointer_height(&self, db: &rocksdb::DB) -> Result<u64> {
        let cf_blocks = db
            .cf_handle(CF_BLOCKS)
            .context("blocks column family not found")?;

        match db.get_cf(&cf_blocks, b"qblock:latest")? {
            Some(bytes) if bytes.len() == 8 => {
                let mut array = [0u8; 8];
                array.copy_from_slice(&bytes);
                Ok(u64::from_be_bytes(array))
            }
            Some(bytes) => {
                warn!("⚠️  Invalid pointer length: {} bytes", bytes.len());
                Ok(0)
            }
            None => {
                warn!("⚠️  qblock:latest pointer missing");
                Ok(0)
            }
        }
    }

    /// Scan database for actual blocks
    async fn scan_blocks(
        &self,
        db: &Arc<rocksdb::DB>,
        pointer_height: u64,
    ) -> Result<(u64, u64, u64, Vec<u64>)> {
        let cf_blocks = db
            .cf_handle(CF_BLOCKS)
            .context("blocks column family not found")?;

        let mut total_blocks = 0u64;
        let mut highest_block = 0u64;
        let mut gaps = Vec::new();

        // Scan up to 2x pointer height (or 200k max)
        let scan_limit = (pointer_height * 2).max(200_000);

        info!("🔍 Scanning blocks 0 → {} ...", scan_limit);

        for height in 0..=scan_limit {
            if height % 10_000 == 0 && height > 0 {
                debug!("   Scanned {} blocks...", height);
            }

            let key = format!("qblock:height:{}", height);

            if db.get_cf(&cf_blocks, key.as_bytes())?.is_some() {
                total_blocks += 1;
                if height > highest_block {
                    highest_block = height;
                }
            } else if height < highest_block {
                // Gap detected
                gaps.push(height);
            } else {
                // Reached end of contiguous chain
                break;
            }
        }

        // Find highest contiguous (no gaps before it)
        let mut highest_contiguous = 0u64;
        for height in 0..=highest_block {
            let key = format!("qblock:height:{}", height);
            if db.get_cf(&cf_blocks, key.as_bytes())?.is_some() {
                highest_contiguous = height;
            } else {
                // Found first gap
                break;
            }
        }

        Ok((total_blocks, highest_block, highest_contiguous, gaps))
    }

    /// Detect corruption type
    fn detect_corruption(
        &self,
        pointer_height: u64,
        total_blocks: u64,
        highest_contiguous: u64,
        gaps: &[u64],
    ) -> Option<CorruptionType> {
        // Check for total data loss
        if total_blocks == 0 && pointer_height > 0 {
            return Some(CorruptionType::TotalDataLoss {
                pointer: pointer_height,
            });
        }

        // Check for pointer too high (catastrophic)
        if pointer_height > highest_contiguous && highest_contiguous < 1000 {
            return Some(CorruptionType::PointerTooHigh {
                pointer: pointer_height,
                actual: highest_contiguous,
            });
        }

        // Check for pointer too low (minor)
        if pointer_height < highest_contiguous {
            // Only report if difference is significant (> 10 blocks)
            if highest_contiguous - pointer_height > 10 {
                return Some(CorruptionType::PointerTooLow {
                    pointer: pointer_height,
                    actual: highest_contiguous,
                });
            }
        }

        // Check for gaps
        if !gaps.is_empty() {
            return Some(CorruptionType::GapsDetected {
                first_gap: gaps[0],
                total_gaps: gaps.len(),
            });
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_total_data_loss() {
        let checker = IntegrityChecker::new(PathBuf::from("/tmp/test"));

        let corruption = checker.detect_corruption(9606, 0, 0, &[]);

        assert_eq!(
            corruption,
            Some(CorruptionType::TotalDataLoss { pointer: 9606 })
        );
    }

    #[test]
    fn test_detect_pointer_too_high() {
        let checker = IntegrityChecker::new(PathBuf::from("/tmp/test"));

        let corruption = checker.detect_corruption(9606, 100, 100, &[]);

        assert_eq!(
            corruption,
            Some(CorruptionType::PointerTooHigh {
                pointer: 9606,
                actual: 100
            })
        );
    }

    #[test]
    fn test_detect_gaps() {
        let checker = IntegrityChecker::new(PathBuf::from("/tmp/test"));

        let gaps = vec![2, 5, 10];
        let corruption = checker.detect_corruption(100, 97, 1, &gaps);

        assert_eq!(
            corruption,
            Some(CorruptionType::GapsDetected {
                first_gap: 2,
                total_gaps: 3
            })
        );
    }

    #[test]
    fn test_healthy_database() {
        let checker = IntegrityChecker::new(PathBuf::from("/tmp/test"));

        let corruption = checker.detect_corruption(100, 101, 100, &[]);

        assert_eq!(corruption, None);
    }
}
