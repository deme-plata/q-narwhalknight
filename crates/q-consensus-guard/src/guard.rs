//! Main Consensus Guard
//!
//! This is the single entry point for all consensus safety.
//! Call `ConsensusGuard::verify_or_panic()` at node startup.

use crate::golden_blocks::{GoldenBlockRegistry, GoldenBlockMismatch, BlockFingerprint};
use crate::serialization_guard::{SerializationGuard, SerializationMismatch};
use crate::determinism::{DeterminismChecker, DeterminismViolation};
use crate::upgrade_gate::{UpgradeGate, init_upgrade_gate};

use anyhow::{anyhow, Result};
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{error, info, warn};

/// Configuration for ConsensusGuard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardConfig {
    /// Is this mainnet?
    pub is_mainnet: bool,

    /// Fail on any violation (recommended for mainnet)
    pub strict_mode: bool,

    /// Enable golden block verification
    pub verify_golden_blocks: bool,

    /// Enable serialization fingerprint checks
    pub verify_serialization: bool,

    /// Enable determinism checking
    pub verify_determinism: bool,

    /// Path to store/load recorded transitions
    pub transitions_path: Option<String>,
}

impl Default for GuardConfig {
    fn default() -> Self {
        Self {
            is_mainnet: false,
            strict_mode: true,
            verify_golden_blocks: true,
            verify_serialization: true,
            verify_determinism: true,
            transitions_path: None,
        }
    }
}

impl GuardConfig {
    /// Create mainnet configuration (maximum safety)
    pub fn mainnet() -> Self {
        Self {
            is_mainnet: true,
            strict_mode: true,
            verify_golden_blocks: true,
            verify_serialization: true,
            verify_determinism: true,
            transitions_path: Some("./data/determinism_log.bin".to_string()),
        }
    }

    /// Create testnet configuration (some flexibility)
    pub fn testnet() -> Self {
        Self {
            is_mainnet: false,
            strict_mode: true,
            verify_golden_blocks: true,
            verify_serialization: true,
            verify_determinism: true,
            transitions_path: None,
        }
    }

    /// Create development configuration (warnings only)
    pub fn development() -> Self {
        Self {
            is_mainnet: false,
            strict_mode: false,
            verify_golden_blocks: true,
            verify_serialization: false,
            verify_determinism: false,
            transitions_path: None,
        }
    }
}

/// All possible consensus violations
#[derive(Debug, Clone, thiserror::Error)]
pub enum ConsensusViolation {
    #[error("Golden block verification failed: {0}")]
    GoldenBlockFailed(#[from] GoldenBlockMismatch),

    #[error("Serialization format changed: {0}")]
    SerializationChanged(#[from] SerializationMismatch),

    #[error("Non-deterministic behavior detected: {0}")]
    NonDeterministic(#[from] DeterminismViolation),

    #[error("Upgrade gate misconfigured: {0}")]
    UpgradeGateError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),
}

impl ConsensusViolation {
    /// Print detailed error message with fix instructions
    pub fn print_detailed(&self) {
        match self {
            ConsensusViolation::GoldenBlockFailed(e) => e.print_detailed(),
            ConsensusViolation::SerializationChanged(e) => e.print_detailed(),
            ConsensusViolation::NonDeterministic(e) => e.print_detailed(),
            _ => {
                error!("🚨 CONSENSUS VIOLATION: {}", self);
            }
        }
    }
}

/// The main consensus guard
///
/// Verifies all safety invariants at startup and during runtime.
pub struct ConsensusGuard {
    config: GuardConfig,
    golden_blocks: GoldenBlockRegistry,
    serialization: SerializationGuard,
    determinism: DeterminismChecker,
    upgrade_gate: Arc<UpgradeGate>,
    verified: bool,
}

impl ConsensusGuard {
    /// Create a new consensus guard
    pub fn new(config: GuardConfig) -> Result<Self> {
        info!("🛡️ [CONSENSUS GUARD] Initializing...");
        info!("   Mode: {}", if config.is_mainnet { "MAINNET" } else { "TESTNET" });
        info!("   Strict: {}", config.strict_mode);

        // Initialize upgrade gate
        init_upgrade_gate(config.is_mainnet);

        let upgrade_gate = Arc::new(UpgradeGate::new(config.is_mainnet));
        let golden_blocks = GoldenBlockRegistry::new(config.is_mainnet);
        let serialization = SerializationGuard::new();
        let determinism = DeterminismChecker::new(config.strict_mode);

        Ok(Self {
            config,
            golden_blocks,
            serialization,
            determinism,
            upgrade_gate,
            verified: false,
        })
    }

    /// Verify all safety invariants
    ///
    /// Call this at node startup. Panics if any check fails in strict mode.
    pub fn verify<F, S>(
        &mut self,
        get_block_fingerprint: F,
        get_serialization_canonical: S,
    ) -> Result<(), ConsensusViolation>
    where
        F: Fn(u64) -> Option<BlockFingerprint>,
        S: Fn(&str) -> Option<Vec<u8>>,
    {
        info!("🛡️ [CONSENSUS GUARD] Running verification suite...");

        // 1. Verify golden blocks
        if self.config.verify_golden_blocks {
            self.golden_blocks.verify_all(get_block_fingerprint)?;
        }

        // 2. Verify serialization fingerprints
        if self.config.verify_serialization {
            self.serialization.verify_all(get_serialization_canonical)?;
        }

        // 3. Log upgrade gate status
        self.upgrade_gate.log_status(0);

        self.verified = true;
        info!("🛡️ [CONSENSUS GUARD] All verifications passed! ✅");

        Ok(())
    }

    /// Quick verification without block/serialization callbacks
    ///
    /// Use this for minimal startup verification.
    pub fn verify_minimal(&mut self) -> Result<(), ConsensusViolation> {
        info!("🛡️ [CONSENSUS GUARD] Running minimal verification...");

        // Just log upgrade gate status
        self.upgrade_gate.log_status(0);

        self.verified = true;
        info!("🛡️ [CONSENSUS GUARD] Minimal verification passed! ✅");

        Ok(())
    }

    /// Verify or panic (for mainnet)
    ///
    /// Call this at startup. Node will not start if verification fails.
    pub fn verify_or_panic<F, S>(
        &mut self,
        get_block_fingerprint: F,
        get_serialization_canonical: S,
    ) where
        F: Fn(u64) -> Option<BlockFingerprint>,
        S: Fn(&str) -> Option<Vec<u8>>,
    {
        if let Err(violation) = self.verify(get_block_fingerprint, get_serialization_canonical) {
            violation.print_detailed();

            if self.config.strict_mode {
                panic!(
                    "\n\n🚨 CONSENSUS GUARD VERIFICATION FAILED 🚨\n\n{}\n\nNode refusing to start to prevent mainnet corruption.\n\n",
                    violation
                );
            } else {
                warn!("⚠️ [CONSENSUS GUARD] Violation detected but strict mode disabled");
            }
        }
    }

    /// Check if a block height has a specific upgrade active
    #[inline]
    pub fn is_upgrade_active(&self, upgrade: crate::upgrade_gate::Upgrade, height: u64) -> bool {
        self.upgrade_gate.is_active(upgrade, height)
    }

    /// Record a state transition for determinism checking
    pub fn record_transition(
        &mut self,
        input_state: [u8; 32],
        operation: &str,
        params: &[u8],
        output_state: [u8; 32],
        height: u64,
    ) {
        if self.config.verify_determinism {
            self.determinism.record_transition(input_state, operation, params, output_state, height);
        }
    }

    /// Verify a transition matches recorded behavior
    pub fn verify_transition(
        &self,
        input_state: [u8; 32],
        operation: &str,
        params: &[u8],
        actual_output: [u8; 32],
    ) -> Result<(), ConsensusViolation> {
        if self.config.verify_determinism {
            self.determinism.verify_transition(input_state, operation, params, actual_output)?;
        }
        Ok(())
    }

    /// Get the upgrade gate for height-gated features
    pub fn upgrade_gate(&self) -> &UpgradeGate {
        &self.upgrade_gate
    }

    /// Check if guard has been verified
    pub fn is_verified(&self) -> bool {
        self.verified
    }
}

/// Global consensus guard instance
static GLOBAL_GUARD: Lazy<RwLock<Option<ConsensusGuard>>> = Lazy::new(|| RwLock::new(None));

/// Initialize the global consensus guard
pub fn init_global_guard(config: GuardConfig) -> Result<()> {
    let guard = ConsensusGuard::new(config)?;
    *GLOBAL_GUARD.write() = Some(guard);
    info!("🛡️ [CONSENSUS GUARD] Global guard initialized");
    Ok(())
}

/// Get the global consensus guard
pub fn global_guard() -> parking_lot::RwLockReadGuard<'static, Option<ConsensusGuard>> {
    GLOBAL_GUARD.read()
}

/// Check if upgrade is active (convenience function using global guard)
#[inline]
pub fn is_upgrade_active(upgrade: crate::upgrade_gate::Upgrade, height: u64) -> bool {
    if let Some(guard) = GLOBAL_GUARD.read().as_ref() {
        guard.is_upgrade_active(upgrade, height)
    } else {
        // Fall back to direct upgrade gate check
        crate::upgrade_gate::is_upgrade_active(upgrade, height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_guard_creation() {
        let guard = ConsensusGuard::new(GuardConfig::default());
        assert!(guard.is_ok());
    }

    #[test]
    fn test_minimal_verification() {
        let mut guard = ConsensusGuard::new(GuardConfig::development()).unwrap();
        assert!(guard.verify_minimal().is_ok());
        assert!(guard.is_verified());
    }

    #[test]
    fn test_config_presets() {
        let mainnet = GuardConfig::mainnet();
        assert!(mainnet.is_mainnet);
        assert!(mainnet.strict_mode);

        let testnet = GuardConfig::testnet();
        assert!(!testnet.is_mainnet);
        assert!(testnet.strict_mode);

        let dev = GuardConfig::development();
        assert!(!dev.strict_mode);
    }
}
