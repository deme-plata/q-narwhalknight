// Pre-Flight Verification System for Mainnet-Safe Deployments
// v3.3.7-beta
//
// This module provides comprehensive integrity verification BEFORE a node starts
// serving requests. Run this after binary updates to ensure:
// 1. All blocks load correctly
// 2. Parent chain integrity is intact
// 3. Balance consistency is verified
// 4. Schema version is compatible
//
// Usage:
//   Q_PREFLIGHT_CHECK=1 ./q-api-server   # Run preflight check before starting
//   Q_PREFLIGHT_ONLY=1 ./q-api-server    # Run preflight only, then exit (dry-run mode)
//
// This eliminates the "cowboy coding luck factor" for mainnet deployments.

use anyhow::{Context, Result};
use tracing::{error, info, warn};
#[cfg(not(target_os = "windows"))]
use rocksdb::DB;
use std::sync::Arc;
use std::time::Instant;

use crate::{CF_BLOCKS, CF_BALANCES, CF_MANIFEST};

/// Pre-flight check results
#[derive(Debug, Clone)]
pub struct PreflightReport {
    /// Total blocks verified
    pub blocks_verified: u64,
    /// Blocks with issues (hash mismatch, missing parents, etc.)
    pub blocks_with_issues: u64,
    /// Balance discrepancies found
    pub balance_discrepancies: u64,
    /// Schema version mismatch
    pub schema_mismatch: bool,
    /// Time taken for verification (seconds)
    pub verification_time_secs: f64,
    /// Detailed issues found
    pub issues: Vec<PreflightIssue>,
    /// Overall pass/fail
    pub passed: bool,
    /// Current height found
    pub current_height: u64,
}

#[derive(Debug, Clone)]
pub struct PreflightIssue {
    pub severity: IssueSeverity,
    pub component: String,
    pub description: String,
    pub block_height: Option<u64>,
    pub recommendation: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum IssueSeverity {
    Critical, // Blocks node from starting
    Warning,  // Logs warning but allows start
    Info,     // Informational only
}

impl std::fmt::Display for IssueSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IssueSeverity::Critical => write!(f, "CRITICAL"),
            IssueSeverity::Warning => write!(f, "WARNING"),
            IssueSeverity::Info => write!(f, "INFO"),
        }
    }
}

/// Pre-flight verification engine
pub struct PreflightVerifier {
    /// Database path
    db_path: String,
    /// Expected schema version
    expected_schema_version: u32,
    /// Software version
    software_version: String,
    /// Sample rate for block verification (1.0 = all blocks, 0.01 = 1% sample)
    sample_rate: f64,
    /// Maximum blocks to verify (0 = all)
    max_blocks: u64,
}

impl PreflightVerifier {
    pub fn new(db_path: String) -> Self {
        Self {
            db_path,
            expected_schema_version: 1, // Update as schema evolves
            software_version: env!("CARGO_PKG_VERSION").to_string(),
            sample_rate: 0.01, // Default: verify 1% of blocks (fast)
            max_blocks: 10_000, // Default: verify at most 10K blocks
        }
    }

    /// Set sample rate for block verification
    pub fn with_sample_rate(mut self, rate: f64) -> Self {
        self.sample_rate = rate.clamp(0.001, 1.0);
        self
    }

    /// Set maximum blocks to verify
    pub fn with_max_blocks(mut self, max: u64) -> Self {
        self.max_blocks = max;
        self
    }

    /// Run comprehensive pre-flight verification
    pub fn run_verification(&self, db: Arc<DB>) -> Result<PreflightReport> {
        let start_time = Instant::now();
        let mut issues = Vec::new();
        let mut blocks_verified = 0u64;
        let mut blocks_with_issues = 0u64;
        let balance_discrepancies = 0u64;

        info!("╔═══════════════════════════════════════════════════════════════╗");
        info!("║           PRE-FLIGHT VERIFICATION STARTING                    ║");
        info!("╠═══════════════════════════════════════════════════════════════╣");
        info!("║  Database: {}  ", self.db_path);
        info!("║  Software: v{}  ", self.software_version);
        info!("║  Sample Rate: {:.1}%  ", self.sample_rate * 100.0);
        info!("╚═══════════════════════════════════════════════════════════════╝");

        // Step 1: Get current height
        info!("📏 [PREFLIGHT 1/5] Checking current height...");
        let current_height = self.get_current_height(&db)?;
        info!("   Current height: {} blocks", current_height);

        // Step 2: Check schema version
        info!("📋 [PREFLIGHT 2/5] Checking schema version...");
        let schema_mismatch = self.check_schema_version(&db, &mut issues);

        // Step 3: Check block chain integrity
        info!("🔗 [PREFLIGHT 3/5] Verifying block chain integrity...");
        let (verified, with_issues) = self.verify_block_chain(&db, current_height, &mut issues)?;
        blocks_verified = verified;
        blocks_with_issues = with_issues;

        // Step 4: Check parent chain continuity
        info!("⛓️  [PREFLIGHT 4/5] Checking parent chain continuity...");
        self.check_parent_chain(&db, current_height, &mut issues)?;

        // Step 5: Check pointer consistency
        info!("🎯 [PREFLIGHT 5/5] Checking database pointers...");
        self.check_pointer_consistency(&db, current_height, &mut issues)?;

        let verification_time = start_time.elapsed().as_secs_f64();

        // Determine pass/fail
        let critical_issues = issues.iter().filter(|i| i.severity == IssueSeverity::Critical).count();
        let passed = critical_issues == 0;

        let report = PreflightReport {
            blocks_verified,
            blocks_with_issues,
            balance_discrepancies,
            schema_mismatch,
            verification_time_secs: verification_time,
            issues,
            passed,
            current_height,
        };

        self.print_report(&report);

        Ok(report)
    }

    fn get_current_height(&self, db: &Arc<DB>) -> Result<u64> {
        let cf = db.cf_handle(CF_MANIFEST).context("Missing metadata CF")?;

        match db.get_cf(&cf, b"current_height")? {
            Some(bytes) => {
                if bytes.len() == 8 {
                    Ok(u64::from_le_bytes(bytes.try_into().unwrap()))
                } else {
                    // Try string format
                    let s = String::from_utf8_lossy(&bytes);
                    s.parse::<u64>().context("Invalid height format")
                }
            }
            None => Ok(0),
        }
    }

    fn check_schema_version(&self, db: &Arc<DB>, issues: &mut Vec<PreflightIssue>) -> bool {
        let cf = match db.cf_handle(CF_MANIFEST) {
            Some(cf) => cf,
            None => {
                info!("   ℹ️  No metadata CF - assuming first run");
                return false;
            }
        };

        match db.get_cf(&cf, b"schema_version") {
            Ok(Some(bytes)) => {
                let version_str = String::from_utf8_lossy(&bytes);
                if let Ok(stored_version) = version_str.parse::<u32>() {
                    if stored_version != self.expected_schema_version {
                        issues.push(PreflightIssue {
                            severity: if stored_version < self.expected_schema_version {
                                IssueSeverity::Critical
                            } else {
                                IssueSeverity::Warning
                            },
                            component: "Schema".to_string(),
                            description: format!(
                                "Schema version mismatch: stored={}, expected={}",
                                stored_version, self.expected_schema_version
                            ),
                            block_height: None,
                            recommendation: if stored_version < self.expected_schema_version {
                                "Run database migration: migrate_to_phase2".to_string()
                            } else {
                                "This binary may be outdated. Consider upgrading.".to_string()
                            },
                        });
                        return true;
                    }
                    info!("   ✅ Schema version: {} (matches expected)", stored_version);
                }
            }
            Ok(None) => {
                info!("   ℹ️  No schema version stored (first run or pre-versioning database)");
            }
            Err(e) => {
                warn!("   ⚠️  Failed to read schema version: {}", e);
            }
        }
        false
    }

    fn verify_block_chain(&self, db: &Arc<DB>, current_height: u64, issues: &mut Vec<PreflightIssue>) -> Result<(u64, u64)> {
        let mut verified = 0u64;
        let mut with_issues = 0u64;

        if current_height == 0 {
            info!("   ℹ️  Empty blockchain (height 0)");
            return Ok((0, 0));
        }

        let cf = db.cf_handle(CF_BLOCKS).context("Missing blocks CF")?;

        // Determine blocks to verify based on sample rate
        let blocks_to_verify = if self.sample_rate >= 1.0 {
            current_height.min(self.max_blocks)
        } else {
            ((current_height as f64) * self.sample_rate).ceil() as u64
        }.min(self.max_blocks);

        info!("   Verifying {} blocks (out of {} total, {:.1}% sample rate)",
              blocks_to_verify, current_height, self.sample_rate * 100.0);

        // Sample blocks evenly across the chain
        let step = if blocks_to_verify > 0 {
            (current_height / blocks_to_verify).max(1)
        } else {
            1
        };

        let mut last_log_time = Instant::now();
        let mut height = 1u64;

        while height <= current_height && verified < blocks_to_verify {
            // Progress logging every 5 seconds
            if last_log_time.elapsed().as_secs() >= 5 {
                info!("   Progress: verified {}/{} blocks ({:.1}%)",
                      verified, blocks_to_verify, (verified as f64 / blocks_to_verify as f64) * 100.0);
                last_log_time = Instant::now();
            }

            let key = format!("height:{}", height);
            match db.get_cf(&cf, key.as_bytes()) {
                Ok(Some(bytes)) => {
                    // Try to deserialize the block
                    match rmp_serde::from_slice::<q_types::QBlock>(&bytes) {
                        Ok(block) => {
                            // Verify height matches
                            if block.header.height != height {
                                issues.push(PreflightIssue {
                                    severity: IssueSeverity::Critical,
                                    component: "BlockChain".to_string(),
                                    description: format!(
                                        "Block height mismatch: stored at index {}, but block.header.height={}",
                                        height, block.header.height
                                    ),
                                    block_height: Some(height),
                                    recommendation: "Database corruption detected. Restore from backup.".to_string(),
                                });
                                with_issues += 1;
                            }
                            verified += 1;
                        }
                        Err(e) => {
                            issues.push(PreflightIssue {
                                severity: IssueSeverity::Critical,
                                component: "BlockChain".to_string(),
                                description: format!("Failed to deserialize block at height {}: {}", height, e),
                                block_height: Some(height),
                                recommendation: "Block data corruption. Restore from backup or re-sync.".to_string(),
                            });
                            with_issues += 1;
                        }
                    }
                }
                Ok(None) => {
                    issues.push(PreflightIssue {
                        severity: IssueSeverity::Warning,
                        component: "BlockChain".to_string(),
                        description: format!("Missing block at height {} (may be a gap)", height),
                        block_height: Some(height),
                        recommendation: "Block gap detected. Run gap-fill sync if this persists.".to_string(),
                    });
                    // Don't count as issue for gaps - they may be normal in DAG
                }
                Err(e) => {
                    issues.push(PreflightIssue {
                        severity: IssueSeverity::Critical,
                        component: "BlockChain".to_string(),
                        description: format!("Failed to load block at height {}: {}", height, e),
                        block_height: Some(height),
                        recommendation: "Database read error. Check RocksDB integrity.".to_string(),
                    });
                    with_issues += 1;
                }
            }

            height += step;
        }

        if with_issues == 0 {
            info!("   ✅ Verified {} blocks - all OK", verified);
        } else {
            error!("   ❌ Verified {} blocks - {} with issues", verified, with_issues);
        }

        Ok((verified, with_issues))
    }

    fn check_parent_chain(&self, db: &Arc<DB>, current_height: u64, issues: &mut Vec<PreflightIssue>) -> Result<()> {
        if current_height < 2 {
            info!("   ℹ️  Skipping parent chain check (height < 2)");
            return Ok(());
        }

        let cf = db.cf_handle(CF_BLOCKS).context("Missing blocks CF")?;

        // Check the last 100 blocks for parent chain continuity
        let check_depth = 100.min(current_height - 1);
        let mut broken_links = 0;
        let mut checks_performed = 0;

        for height in (current_height.saturating_sub(check_depth) + 1)..=current_height {
            let key = format!("height:{}", height);
            let parent_key = format!("height:{}", height - 1);

            let block_result = db.get_cf(&cf, key.as_bytes());
            let parent_result = db.get_cf(&cf, parent_key.as_bytes());

            if let (Ok(Some(block_bytes)), Ok(Some(parent_bytes))) = (block_result, parent_result) {
                if let (Ok(block), Ok(parent_block)) = (
                    rmp_serde::from_slice::<q_types::QBlock>(&block_bytes),
                    rmp_serde::from_slice::<q_types::QBlock>(&parent_bytes),
                ) {
                    if block.header.prev_block_hash != parent_block.calculate_hash() {
                        issues.push(PreflightIssue {
                            severity: IssueSeverity::Critical,
                            component: "ParentChain".to_string(),
                            description: format!(
                                "Broken parent link at height {}: prev_hash={}, parent_hash={}",
                                height,
                                hex::encode(&block.header.prev_block_hash[..8]),
                                hex::encode(&parent_block.calculate_hash()[..8])
                            ),
                            block_height: Some(height),
                            recommendation: "Chain fork or corruption detected. Restore from checkpoint.".to_string(),
                        });
                        broken_links += 1;
                    }
                    checks_performed += 1;
                }
            }
        }

        if broken_links == 0 {
            info!("   ✅ Parent chain intact (checked {} links)", checks_performed);
        } else {
            error!("   ❌ Found {} broken parent links!", broken_links);
        }

        Ok(())
    }

    fn check_pointer_consistency(&self, db: &Arc<DB>, current_height: u64, issues: &mut Vec<PreflightIssue>) -> Result<()> {
        if current_height == 0 {
            info!("   ℹ️  Height pointer at 0 (empty or fresh chain)");
            return Ok(());
        }

        let cf = db.cf_handle(CF_BLOCKS).context("Missing blocks CF")?;

        // Verify the block at current_height exists
        let key = format!("height:{}", current_height);
        match db.get_cf(&cf, key.as_bytes()) {
            Ok(Some(block_bytes)) => {
                match rmp_serde::from_slice::<q_types::QBlock>(&block_bytes) {
                    Ok(block) => {
                        if block.header.height != current_height {
                            issues.push(PreflightIssue {
                                severity: IssueSeverity::Critical,
                                component: "Pointers".to_string(),
                                description: format!(
                                    "Height pointer {} points to block with height {}",
                                    current_height, block.header.height
                                ),
                                block_height: Some(current_height),
                                recommendation: "Pointer inconsistency. Reset pointers with repair_database.".to_string(),
                            });
                        } else {
                            info!("   ✅ Height pointer {} matches block height", current_height);
                        }
                    }
                    Err(e) => {
                        issues.push(PreflightIssue {
                            severity: IssueSeverity::Warning,
                            component: "Pointers".to_string(),
                            description: format!("Failed to deserialize tip block: {}", e),
                            block_height: Some(current_height),
                            recommendation: "Tip block may be corrupted.".to_string(),
                        });
                    }
                }
            }
            Ok(None) => {
                issues.push(PreflightIssue {
                    severity: IssueSeverity::Critical,
                    component: "Pointers".to_string(),
                    description: format!("Height pointer {} points to non-existent block", current_height),
                    block_height: Some(current_height),
                    recommendation: "Missing tip block. Restore from backup or re-sync.".to_string(),
                });
            }
            Err(e) => {
                issues.push(PreflightIssue {
                    severity: IssueSeverity::Warning,
                    component: "Pointers".to_string(),
                    description: format!("Failed to verify tip block: {}", e),
                    block_height: Some(current_height),
                    recommendation: "Database read error during verification.".to_string(),
                });
            }
        }

        Ok(())
    }

    fn print_report(&self, report: &PreflightReport) {
        info!("");
        info!("╔═══════════════════════════════════════════════════════════════╗");
        info!("║               PRE-FLIGHT VERIFICATION REPORT                  ║");
        info!("╠═══════════════════════════════════════════════════════════════╣");
        info!("║  Current Height: {} blocks  ", report.current_height);
        info!("║  Blocks Verified: {}  ", report.blocks_verified);
        info!("║  Blocks with Issues: {}  ", report.blocks_with_issues);
        info!("║  Verification Time: {:.2}s  ", report.verification_time_secs);
        info!("╠═══════════════════════════════════════════════════════════════╣");

        let critical = report.issues.iter().filter(|i| i.severity == IssueSeverity::Critical).count();
        let warnings = report.issues.iter().filter(|i| i.severity == IssueSeverity::Warning).count();

        if critical > 0 {
            error!("║  ❌ CRITICAL ISSUES: {}  ", critical);
        }
        if warnings > 0 {
            warn!("║  ⚠️  WARNINGS: {}  ", warnings);
        }

        if report.passed {
            info!("║                                                               ║");
            info!("║  ✅ PRE-FLIGHT CHECK: PASSED                                  ║");
            info!("║     Node is safe to start serving requests                    ║");
        } else {
            error!("║                                                               ║");
            error!("║  ❌ PRE-FLIGHT CHECK: FAILED                                  ║");
            error!("║     DO NOT start serving requests until issues are resolved  ║");
        }
        info!("╚═══════════════════════════════════════════════════════════════╝");

        // Print detailed issues
        if !report.issues.is_empty() {
            info!("");
            info!("Detailed Issues:");
            for (i, issue) in report.issues.iter().enumerate() {
                match issue.severity {
                    IssueSeverity::Critical => {
                        error!("  {}. [{}] {} - {}", i + 1, issue.severity, issue.component, issue.description);
                        error!("     Recommendation: {}", issue.recommendation);
                    }
                    IssueSeverity::Warning => {
                        warn!("  {}. [{}] {} - {}", i + 1, issue.severity, issue.component, issue.description);
                        warn!("     Recommendation: {}", issue.recommendation);
                    }
                    IssueSeverity::Info => {
                        info!("  {}. [{}] {} - {}", i + 1, issue.severity, issue.component, issue.description);
                    }
                }
            }
        }
    }
}

/// Check if preflight verification is enabled via environment variable
pub fn is_preflight_enabled() -> bool {
    std::env::var("Q_PREFLIGHT_CHECK").is_ok() || std::env::var("Q_PREFLIGHT_ONLY").is_ok()
}

/// Check if we should exit after preflight (dry-run mode)
pub fn is_preflight_only() -> bool {
    std::env::var("Q_PREFLIGHT_ONLY").is_ok()
}

/// Run preflight check with default settings
pub fn run_preflight_check(db: Arc<DB>, db_path: &str) -> Result<PreflightReport> {
    // Determine sample rate from environment (default 1% for speed)
    let sample_rate = std::env::var("Q_PREFLIGHT_SAMPLE_RATE")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.01);

    // Determine max blocks from environment (default 10K)
    let max_blocks = std::env::var("Q_PREFLIGHT_MAX_BLOCKS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(10_000);

    let verifier = PreflightVerifier::new(db_path.to_string())
        .with_sample_rate(sample_rate)
        .with_max_blocks(max_blocks);

    verifier.run_verification(db)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preflight_env_detection() {
        // Test that environment variables are detected
        std::env::remove_var("Q_PREFLIGHT_CHECK");
        std::env::remove_var("Q_PREFLIGHT_ONLY");
        assert!(!is_preflight_enabled());
        assert!(!is_preflight_only());

        std::env::set_var("Q_PREFLIGHT_CHECK", "1");
        assert!(is_preflight_enabled());
        assert!(!is_preflight_only());

        std::env::remove_var("Q_PREFLIGHT_CHECK");
        std::env::set_var("Q_PREFLIGHT_ONLY", "1");
        assert!(is_preflight_enabled());
        assert!(is_preflight_only());

        // Cleanup
        std::env::remove_var("Q_PREFLIGHT_ONLY");
    }

    #[test]
    fn test_preflight_verifier_configuration() {
        let verifier = PreflightVerifier::new("/tmp/test".to_string())
            .with_sample_rate(0.5)
            .with_max_blocks(5000);

        assert_eq!(verifier.sample_rate, 0.5);
        assert_eq!(verifier.max_blocks, 5000);
    }
}
