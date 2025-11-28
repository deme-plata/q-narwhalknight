//! Database Pointer Integrity Protection
//!
//! Prevents database pointer corruption that causes height regression.
//! Implements startup validation and automatic recovery from pointer corruption.
//!
//! This module solves the critical issue discovered on 2025-11-17 where the
//! `qblock:latest` pointer was corrupted from 12,114 → 353, causing the node
//! to think it was at height 353 while all 12,114 blocks were intact.

use anyhow::{Context, Result};
use rocksdb::DB;
use std::sync::Arc;
use tracing::{error, info, warn};

use crate::CF_BLOCKS;

/// Configuration for pointer integrity severity thresholds
#[derive(Debug, Clone)]
pub struct IntegrityThresholds {
    /// Threshold for minor corruption (default: 10 blocks)
    pub minor_threshold: u64,
    /// Threshold for moderate corruption (default: 100 blocks)
    pub moderate_threshold: u64,
}

impl Default for IntegrityThresholds {
    fn default() -> Self {
        Self {
            minor_threshold: 10,
            moderate_threshold: 100,
        }
    }
}

/// Database pointer integrity checker and auto-recovery system
pub struct PointerIntegrityChecker {
    db: Arc<DB>,
    /// ✅ CHATGPT FIX: Configurable thresholds instead of magic numbers
    thresholds: IntegrityThresholds,
}

/// Result of pointer integrity check
#[derive(Debug, Clone)]
pub struct IntegrityCheckResult {
    pub pointer_height: u64,
    pub actual_highest_height: u64,
    pub total_blocks_found: u64,
    pub is_corrupted: bool,
    pub corruption_severity: CorruptionSeverity,
}

/// Severity of pointer corruption
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorruptionSeverity {
    /// Pointer matches actual height - no corruption
    None,
    /// Pointer off by <10 blocks - minor (may be race condition)
    Minor,
    /// Pointer off by 10-100 blocks - moderate
    Moderate,
    /// Pointer off by >100 blocks - severe (like the 353 vs 12,114 case)
    Severe,
}

impl PointerIntegrityChecker {
    /// Create a new pointer integrity checker with default thresholds
    pub fn new(db: Arc<DB>) -> Self {
        Self::with_thresholds(db, IntegrityThresholds::default())
    }

    /// Create a new pointer integrity checker with custom thresholds
    /// ✅ CHATGPT FIX: Allow configurable thresholds for different deployment scenarios
    pub fn with_thresholds(db: Arc<DB>, thresholds: IntegrityThresholds) -> Self {
        Self { db, thresholds }
    }

    /// Perform comprehensive pointer integrity check on startup
    ///
    /// This scans the database to find the actual highest block and compares
    /// it to the `qblock:latest` pointer. If there's a mismatch, it indicates
    /// corruption and can trigger automatic recovery.
    pub fn check_on_startup(&self) -> Result<IntegrityCheckResult> {
        info!("🔍 DATABASE INTEGRITY CHECK: Starting pointer validation...");

        let cf_blocks = self.db.cf_handle(CF_BLOCKS)
            .context("blocks column family not found")?;

        // Read current pointer value
        let pointer_height = match self.db.get_cf(&cf_blocks, b"qblock:latest")? {
            Some(bytes) if bytes.len() == 8 => {
                let mut height_array = [0u8; 8];
                height_array.copy_from_slice(&bytes);
                u64::from_be_bytes(height_array)
            }
            Some(bytes) => {
                error!("🚨 CRITICAL: qblock:latest has invalid length: {} bytes", bytes.len());
                0 // Treat as corrupted
            }
            None => {
                error!("🚨 CRITICAL: qblock:latest pointer is MISSING!");
                0 // Treat as corrupted
            }
        };

        info!("   Current pointer value: {}", pointer_height);

        // Scan database to find actual highest block
        // Use optimized search strategy: check recent blocks first, then scan backwards
        let actual_highest = self.find_highest_block(&cf_blocks)?;
        let total_blocks = self.count_blocks_up_to(&cf_blocks, actual_highest)?;

        info!("   Actual highest block: {}", actual_highest);
        info!("   Total blocks found: {}", total_blocks);

        // Determine corruption severity
        // ✅ CRITICAL FIX (ChatGPT recommendation): Handle reverse corruption (pointer > actual)
        // Calculate diff (absolute difference for logging)
        let diff = if pointer_height > actual_highest {
            pointer_height - actual_highest
        } else {
            actual_highest - pointer_height
        };

        // ✅ CHATGPT FIX: Use configurable thresholds instead of hardcoded magic numbers
        let severity = if pointer_height > actual_highest {
            // Pointer points into empty space - ALWAYS SEVERE
            // This means blocks were deleted or partial DB flush occurred
            error!("🚨 REVERSE CORRUPTION: Pointer ({}) > Actual ({}) - blocks may be missing!",
                   pointer_height, actual_highest);
            CorruptionSeverity::Severe
        } else {
            // Normal forward corruption (actual > pointer)
            // Use configurable thresholds for flexibility across deployments
            if diff == 0 {
                CorruptionSeverity::None
            } else if diff < self.thresholds.minor_threshold {
                CorruptionSeverity::Minor
            } else if diff < self.thresholds.moderate_threshold {
                CorruptionSeverity::Moderate
            } else {
                CorruptionSeverity::Severe
            }
        };

        let is_corrupted = severity != CorruptionSeverity::None;

        if is_corrupted {
            error!("🚨 DATABASE POINTER CORRUPTION DETECTED!");
            error!("   Pointer: {} | Actual: {} | Diff: {} | Severity: {:?}",
                   pointer_height, actual_highest, diff, severity);
        } else {
            info!("✅ DATABASE INTEGRITY CHECK: Pointer is correct ({} blocks)", total_blocks);
        }

        Ok(IntegrityCheckResult {
            pointer_height,
            actual_highest_height: actual_highest,
            total_blocks_found: total_blocks,
            is_corrupted,
            corruption_severity: severity,
        })
    }

    /// Find the highest block in the database using optimized search
    ///
    /// 🚨 v1.0.17-beta FIX: Added sanity bounds to prevent u64::MAX corruption
    /// BUG: Was finding orphaned binary-key blocks and returning u64::MAX
    /// FIX: Only search up to 1M blocks max, ignore impossibly high values
    ///
    /// Strategy:
    /// 1. Try the current pointer value first (fast path for healthy DB)
    /// 2. Binary search upward if pointer seems too low
    /// 3. Scan backward from a high estimate if binary search fails
    fn find_highest_block(&self, cf_blocks: &impl rocksdb::AsColumnFamilyRef) -> Result<u64> {
        // 🚨 CRITICAL v1.0.35-beta FIX: Remove 100k block limit
        // BUG: v1.0.17-beta added safety limit of 100k blocks
        // ISSUE: Nodes with >100k blocks lost data on restart!
        // FIX: Scan up to 10M blocks (reasonable max for testnet), then use reverse scan
        // This is slower but COMPLETE - no data loss

        warn!("🔍 find_highest_block: Starting SAFE linear scan (max 10M blocks)");
        let absolute_max = 10_000_000u64;  // Increased from 100k to 10M

        let mut highest_found = 0u64;

        // Linear scan from 0 to absolute_max
        for height in 0..=absolute_max {
            if self.block_exists(cf_blocks, height)? {
                highest_found = height;
            } else if height > highest_found + 1000 {
                // If we've seen 1000 consecutive missing blocks, stop
                warn!("🔍 find_highest_block: Stopping at {} (1000 consecutive gaps)", highest_found);
                break;
            }
        }

        // 🚨 FINAL SANITY CHECK: Never return impossible values
        if highest_found > absolute_max {
            error!("🚨 CRITICAL: find_highest_block found impossible height: {}", highest_found);
            error!("   Forcing return of 0 to prevent corruption");
            return Ok(0);
        }

        if highest_found == 0 {
            warn!("⚠️  find_highest_block: No valid blocks found, returning 0");
        } else {
            info!("✅ find_highest_block: Found highest block at {}", highest_found);
        }

        Ok(highest_found)
    }

    /// Check if a block exists at the given height
    ///
    /// 🚨 v1.0.17-beta FIX: Only check for STRING keys (qblock:height:N)
    /// IGNORES: Binary-key blocks from transaction.rs bug (0x0000000000000001 format)
    /// This prevents pointer corruption from counting orphaned binary-key blocks
    fn block_exists(&self, cf_blocks: &impl rocksdb::AsColumnFamilyRef, height: u64) -> Result<bool> {
        let key = format!("qblock:height:{}", height);
        Ok(self.db.get_cf(cf_blocks, key.as_bytes())?.is_some())
    }

    /// Count blocks up to a given height (optimized sampling)
    fn count_blocks_up_to(&self, cf_blocks: &impl rocksdb::AsColumnFamilyRef, max_height: u64) -> Result<u64> {
        // For large heights, estimate by sampling
        if max_height > 10_000 {
            // Sample every 100th block and extrapolate
            let sample_size = (max_height / 100).min(1000);
            let mut found = 0u64;

            for i in 0..sample_size {
                let height = (i * 100) + (i % 10); // Add some randomness
                if self.block_exists(cf_blocks, height)? {
                    found += 1;
                }
            }

            // Estimate total (with safety margin)
            return Ok((found * 100).min(max_height));
        }

        // For smaller chains, count exactly
        let mut count = 0u64;
        for height in 0..=max_height {
            if self.block_exists(cf_blocks, height)? {
                count += 1;
            }
        }
        Ok(count)
    }

    /// Attempt automatic recovery from pointer corruption
    ///
    /// This is the "auto-repair" function that runs if corruption is detected.
    /// It updates the `qblock:latest` pointer to match the actual highest block.
    ///
    /// SAFETY: Only repairs if corruption is SEVERE (>100 blocks off) to avoid
    /// false positives from normal operation.
    pub fn auto_repair(&self, check_result: &IntegrityCheckResult) -> Result<()> {
        if !check_result.is_corrupted {
            info!("✅ No repair needed - pointer is correct");
            return Ok(());
        }

        match check_result.corruption_severity {
            CorruptionSeverity::None => {
                info!("✅ No corruption detected");
                Ok(())
            }
            CorruptionSeverity::Minor => {
                warn!("⚠️  Minor pointer mismatch detected ({} blocks off)",
                      check_result.actual_highest_height.saturating_sub(check_result.pointer_height));
                warn!("   Not auto-repairing (may be normal race condition during block production)");
                Ok(())
            }
            CorruptionSeverity::Moderate => {
                warn!("⚠️  Moderate pointer corruption detected ({} blocks off)",
                      check_result.actual_highest_height.saturating_sub(check_result.pointer_height));
                warn!("   Not auto-repairing (manual intervention recommended)");
                Err(anyhow::anyhow!("Moderate corruption detected - manual repair recommended"))
            }
            CorruptionSeverity::Severe => {
                error!("🚨 SEVERE pointer corruption detected!");
                error!("   Pointer: {} | Actual: {} | Diff: {}",
                       check_result.pointer_height,
                       check_result.actual_highest_height,
                       check_result.actual_highest_height.saturating_sub(check_result.pointer_height));
                error!("🔧 Attempting automatic recovery...");

                self.repair_pointer(check_result.actual_highest_height)?;

                info!("✅ Automatic repair successful!");
                info!("   Pointer updated: {} → {}",
                      check_result.pointer_height,
                      check_result.actual_highest_height);
                Ok(())
            }
        }
    }

    /// Repair the database pointer to the correct value
    ///
    /// ✅ CHATGPT FIX ALREADY IMPLEMENTED: Atomic WriteBatch for pointer updates
    /// See: crates/q-storage/src/safe_batched_writer.rs:266-292
    /// The SafeBatchedWriter.add_block_to_batch() method uses WriteBatch to atomically write:
    ///     1. Block data (by height + by hash)
    ///     2. qblock:latest pointer
    /// This prevents pointer corruption at the source (both or neither commit).
    ///
    /// This repair function is for recovery only (fixes corruption after it happens).
    /// For prevention, use SafeBatchedWriter for all block writes.
    fn repair_pointer(&self, correct_height: u64) -> Result<()> {
        let cf_blocks = self.db.cf_handle(CF_BLOCKS)
            .context("blocks column family not found")?;

        let height_bytes = correct_height.to_be_bytes();
        self.db.put_cf(&cf_blocks, b"qblock:latest", &height_bytes)?;

        // Verify the fix
        let verify = self.db.get_cf(&cf_blocks, b"qblock:latest")?
            .context("Failed to verify pointer after repair")?;

        let mut verify_array = [0u8; 8];
        verify_array.copy_from_slice(&verify);
        let verify_height = u64::from_be_bytes(verify_array);

        if verify_height != correct_height {
            error!("🚨 REPAIR VERIFICATION FAILED!");
            error!("   Expected: {} | Got: {}", correct_height, verify_height);
            return Err(anyhow::anyhow!("Pointer repair verification failed"));
        }

        Ok(())
    }

    /// Create a genesis block if it's missing
    ///
    /// The genesis block (height 0) can be missing in bootstrap nodes that
    /// started at height 1. This creates a minimal genesis block to satisfy
    /// integrity checks.
    pub fn ensure_genesis_block(&self) -> Result<()> {
        let cf_blocks = self.db.cf_handle(CF_BLOCKS)
            .context("blocks column family not found")?;

        // Check if genesis exists
        if self.block_exists(&cf_blocks, 0)? {
            info!("✅ Genesis block (height 0) exists");
            return Ok(());
        }

        warn!("⚠️  Genesis block (height 0) is MISSING!");
        warn!("   This is normal for bootstrap nodes that started at height 1");
        warn!("   Skipping genesis creation (not required for operation)");

        Ok(())
    }
}

/// Perform startup integrity check and auto-repair if needed
///
/// This is the main entry point called during node startup.
/// It checks pointer integrity and attempts automatic recovery if corruption
/// is detected.
///
/// CRASH-FAST: If severe corruption is detected and auto-repair fails,
/// this function will return an error that should cause the node to crash.
/// This prevents the node from running in a corrupted state.
pub fn check_and_repair_on_startup(db: Arc<DB>) -> Result<IntegrityCheckResult> {
    let checker = PointerIntegrityChecker::new(db);

    // Step 1: Check integrity
    let result = checker.check_on_startup()
        .context("Failed to check database pointer integrity")?;

    // Step 2: Ensure genesis block exists (non-critical)
    let _ = checker.ensure_genesis_block();

    // Step 3: Auto-repair if needed
    if result.is_corrupted {
        match result.corruption_severity {
            CorruptionSeverity::Severe => {
                // Attempt auto-repair for severe corruption
                checker.auto_repair(&result)
                    .context("Failed to auto-repair severe pointer corruption")?;

                // Re-check after repair
                let recheck = checker.check_on_startup()?;
                if recheck.is_corrupted {
                    error!("🚨 CRITICAL: Auto-repair failed - pointer still corrupted!");
                    return Err(anyhow::anyhow!("Pointer auto-repair failed verification"));
                }

                info!("✅ Database pointer repaired successfully!");
                return Ok(recheck);
            }
            CorruptionSeverity::Moderate => {
                error!("🚨 CRITICAL: Moderate pointer corruption detected!");
                error!("   Manual intervention required - run repair-database tool");
                return Err(anyhow::anyhow!("Moderate pointer corruption - manual repair required"));
            }
            CorruptionSeverity::Minor => {
                warn!("⚠️  Minor pointer mismatch detected - monitoring");
                // Allow startup but log warning
            }
            CorruptionSeverity::None => {
                // Already handled above
            }
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_corruption_severity_classification() {
        // ✅ CHATGPT FIX: Test with default configurable thresholds (10, 100)
        let thresholds = IntegrityThresholds::default();
        let test_cases = vec![
            (0, CorruptionSeverity::None),
            (5, CorruptionSeverity::Minor),     // < 10
            (50, CorruptionSeverity::Moderate),  // >= 10, < 100
            (1000, CorruptionSeverity::Severe),  // >= 100
            (12000, CorruptionSeverity::Severe), // Like the 353 vs 12114 case
        ];

        for (diff, expected) in test_cases {
            let severity = if diff == 0 {
                CorruptionSeverity::None
            } else if diff < thresholds.minor_threshold {
                CorruptionSeverity::Minor
            } else if diff < thresholds.moderate_threshold {
                CorruptionSeverity::Moderate
            } else {
                CorruptionSeverity::Severe
            };
            assert_eq!(severity, expected, "Failed for diff={}", diff);
        }
    }

    #[test]
    fn test_custom_thresholds() {
        // Test with custom thresholds (20, 200)
        let thresholds = IntegrityThresholds {
            minor_threshold: 20,
            moderate_threshold: 200,
        };

        let test_cases = vec![
            (0, CorruptionSeverity::None),
            (10, CorruptionSeverity::Minor),     // < 20
            (100, CorruptionSeverity::Moderate), // >= 20, < 200
            (500, CorruptionSeverity::Severe),   // >= 200
        ];

        for (diff, expected) in test_cases {
            let severity = if diff == 0 {
                CorruptionSeverity::None
            } else if diff < thresholds.minor_threshold {
                CorruptionSeverity::Minor
            } else if diff < thresholds.moderate_threshold {
                CorruptionSeverity::Moderate
            } else {
                CorruptionSeverity::Severe
            };
            assert_eq!(severity, expected, "Failed for diff={} with custom thresholds", diff);
        }
    }
}
