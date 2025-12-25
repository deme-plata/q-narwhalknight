//! # Q-NarwhalKnight Upgrade Framework
//!
//! Block-height activated upgrades for safe mainnet evolution.
//! This allows deploying new binaries without coordinated restarts.
//!
//! ## How it works:
//! 1. New feature is implemented behind a height check
//! 2. Activation height is set (e.g., 2 weeks in the future)
//! 3. Node operators upgrade binaries at their convenience
//! 4. At activation height, all nodes switch to new rules simultaneously
//!
//! ## Safety guarantees:
//! - Old blocks always validate with old rules (immutable history)
//! - New rules only apply to blocks >= activation height
//! - If bug found: announce delay, nodes can downgrade before activation

use std::sync::atomic::{AtomicU64, Ordering};

/// Network upgrade definitions
///
/// Each upgrade has:
/// - Unique name for logging/debugging
/// - Activation height (when it takes effect)
/// - Description of what changes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NetworkUpgrade {
    pub name: &'static str,
    pub activation_height: u64,
    pub description: &'static str,
}

/// All network upgrades in chronological order
///
/// IMPORTANT: Never remove or reorder entries!
/// Only append new upgrades at the end.
pub mod upgrades {
    use super::NetworkUpgrade;

    /// Genesis - the beginning
    pub const GENESIS: NetworkUpgrade = NetworkUpgrade {
        name: "genesis",
        activation_height: 0,
        description: "Network launch",
    };

    /// Phase 16 - Current testnet
    pub const PHASE_16: NetworkUpgrade = NetworkUpgrade {
        name: "phase_16",
        activation_height: 0,
        description: "Testnet Phase 16 - P2P fixes, DAG-Knight stability",
    };

    /// ML Batch Optimizer (v1.4.0)
    pub const ML_BATCH_OPTIMIZER: NetworkUpgrade = NetworkUpgrade {
        name: "ml_batch_optimizer",
        activation_height: 0, // Already active (testnet feature)
        description: "ML-driven adaptive batch size for sync",
    };

    /// Post-Quantum Signatures Required
    /// Set this to a future height when ready for mainnet
    pub const PQ_SIGNATURES_REQUIRED: NetworkUpgrade = NetworkUpgrade {
        name: "pq_signatures_required",
        activation_height: u64::MAX, // Not yet activated
        description: "Require Dilithium signatures on all transactions",
    };

    /// Example future upgrade (template)
    pub const FUTURE_UPGRADE_TEMPLATE: NetworkUpgrade = NetworkUpgrade {
        name: "future_upgrade",
        activation_height: u64::MAX, // Set to specific height when ready
        description: "Description of the upgrade",
    };
}

/// Upgrade manager - checks if upgrades are active at given height
#[derive(Debug)]
pub struct UpgradeManager {
    /// Current chain height (updated as blocks are processed)
    current_height: AtomicU64,

    /// Network type (mainnet activations differ from testnet)
    is_mainnet: bool,
}

impl UpgradeManager {
    /// Create new upgrade manager
    pub fn new(is_mainnet: bool) -> Self {
        Self {
            current_height: AtomicU64::new(0),
            is_mainnet,
        }
    }

    /// Update current height
    pub fn set_height(&self, height: u64) {
        self.current_height.store(height, Ordering::SeqCst);
    }

    /// Get current height
    pub fn height(&self) -> u64 {
        self.current_height.load(Ordering::SeqCst)
    }

    /// Check if an upgrade is active at the current height
    pub fn is_active(&self, upgrade: &NetworkUpgrade) -> bool {
        self.is_active_at_height(upgrade, self.height())
    }

    /// Check if an upgrade is active at a specific height
    pub fn is_active_at_height(&self, upgrade: &NetworkUpgrade, height: u64) -> bool {
        height >= upgrade.activation_height
    }

    /// Get all upgrades that activated between two heights
    pub fn upgrades_between(&self, from_height: u64, to_height: u64) -> Vec<&'static NetworkUpgrade> {
        let all_upgrades = [
            &upgrades::GENESIS,
            &upgrades::PHASE_16,
            &upgrades::ML_BATCH_OPTIMIZER,
            &upgrades::PQ_SIGNATURES_REQUIRED,
        ];

        all_upgrades
            .iter()
            .filter(|u| u.activation_height > from_height && u.activation_height <= to_height)
            .copied()
            .collect()
    }

    /// Log active upgrades at current height
    pub fn log_active_upgrades(&self) {
        let height = self.height();
        tracing::info!("📋 Active upgrades at height {}:", height);

        let all_upgrades = [
            &upgrades::GENESIS,
            &upgrades::PHASE_16,
            &upgrades::ML_BATCH_OPTIMIZER,
            &upgrades::PQ_SIGNATURES_REQUIRED,
        ];

        for upgrade in all_upgrades.iter() {
            if self.is_active_at_height(upgrade, height) {
                tracing::info!("  ✅ {} (height {}): {}",
                    upgrade.name, upgrade.activation_height, upgrade.description);
            } else {
                tracing::info!("  ⏳ {} (height {}): {} [PENDING]",
                    upgrade.name, upgrade.activation_height, upgrade.description);
            }
        }
    }
}

/// Macro for height-gated features
///
/// Usage:
/// ```rust
/// if_upgrade_active!(manager, PQ_SIGNATURES_REQUIRED, {
///     // New code path
///     validate_pq_signature(tx)?;
/// } else {
///     // Old code path (for backward compatibility)
///     validate_ed25519_signature(tx)?;
/// });
/// ```
#[macro_export]
macro_rules! if_upgrade_active {
    ($manager:expr, $upgrade:ident, $then:block else $else:block) => {
        if $manager.is_active(&$crate::upgrades::upgrades::$upgrade) {
            $then
        } else {
            $else
        }
    };
    ($manager:expr, $upgrade:ident, $then:block) => {
        if $manager.is_active(&$crate::upgrades::upgrades::$upgrade) {
            $then
        }
    };
}

/// Database schema version tracking
///
/// When you need to change database schema:
/// 1. Increment CURRENT_SCHEMA_VERSION
/// 2. Add migration in migrations list
/// 3. Migration runs automatically on startup
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SchemaVersion(pub u32);

impl SchemaVersion {
    /// Current schema version
    pub const CURRENT: SchemaVersion = SchemaVersion(1);

    /// Minimum supported version (for migration)
    pub const MINIMUM_SUPPORTED: SchemaVersion = SchemaVersion(1);
}

/// Database migration definition
pub struct Migration {
    pub from_version: SchemaVersion,
    pub to_version: SchemaVersion,
    pub description: &'static str,
    // Migration function would go here
}

/// All database migrations
pub const MIGRATIONS: &[Migration] = &[
    // Example migration (add more as needed):
    // Migration {
    //     from_version: SchemaVersion(1),
    //     to_version: SchemaVersion(2),
    //     description: "Add index on block timestamp",
    // },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_upgrade_activation() {
        let manager = UpgradeManager::new(false);

        // Genesis is always active
        assert!(manager.is_active_at_height(&upgrades::GENESIS, 0));
        assert!(manager.is_active_at_height(&upgrades::GENESIS, 1_000_000));

        // PQ signatures not yet active (height = MAX)
        assert!(!manager.is_active_at_height(&upgrades::PQ_SIGNATURES_REQUIRED, 0));
        assert!(!manager.is_active_at_height(&upgrades::PQ_SIGNATURES_REQUIRED, 1_000_000));
    }

    #[test]
    fn test_upgrade_between_heights() {
        let manager = UpgradeManager::new(false);

        // No upgrades between 0 and 100 (all activate at 0)
        let between = manager.upgrades_between(0, 100);
        assert!(between.is_empty());
    }
}
