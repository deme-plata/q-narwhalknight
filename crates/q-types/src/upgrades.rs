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

    // =========================================================================
    // CRITICAL SECURITY FIXES (v2.3.1-beta)
    // =========================================================================

    /// SQIsign Signature Verification Fix
    ///
    /// CRITICAL SECURITY FIX: Previous versions accepted ALL SQIsign signatures
    /// without actual cryptographic verification. This upgrade enables proper
    /// commitment comparison in SQIsign verification.
    ///
    /// - Before: Any SQIsign signature accepted (security theater)
    /// - After: Only cryptographically valid signatures accepted
    ///
    /// Set to 0 for testnet (immediate activation).
    /// For mainnet: Set to current_height + 20000 before launch.
    pub const SQISIGN_VERIFICATION_FIX: NetworkUpgrade = NetworkUpgrade {
        name: "sqisign_verification_fix",
        activation_height: 0, // TESTNET: Immediate activation
        description: "CRITICAL: Enable actual SQIsign signature verification",
    };

    /// Hybrid Key Separation Requirement
    ///
    /// CRITICAL SECURITY FIX: Previous versions would silently use the same key
    /// for both Ed25519 and SQIsign in hybrid mode, breaking the "both must break"
    /// security guarantee.
    ///
    /// - Before: Hybrid mode could use same key for both algorithms
    /// - After: Hybrid mode requires separate key material (32 + 64 bytes)
    ///
    /// Set to 0 for testnet (immediate activation).
    pub const HYBRID_KEY_SEPARATION: NetworkUpgrade = NetworkUpgrade {
        name: "hybrid_key_separation",
        activation_height: 0, // TESTNET: Immediate activation
        description: "CRITICAL: Require separate keys for hybrid Ed25519+SQIsign",
    };

    /// Phase 1 (Dilithium5) Deprecation
    ///
    /// Dilithium5 signatures deprecated in v1.0.86-beta due to size (4,627 bytes).
    /// At activation: New blocks MUST NOT use Phase1Dilithium5.
    /// Historical Phase1 blocks remain valid for chain consistency.
    pub const PHASE1_DILITHIUM5_DEPRECATED: NetworkUpgrade = NetworkUpgrade {
        name: "phase1_dilithium5_deprecated",
        activation_height: 1_000_000, // Future activation
        description: "Deprecate Dilithium5 signatures - use SQIsign instead",
    };

    /// Phase 2 (SQIsign) Mandatory
    ///
    /// At activation: New blocks MUST use Phase2SQIsign or HybridEd25519SQIsign.
    /// Phase0Ed25519 no longer accepted for NEW blocks (historical blocks valid).
    pub const PHASE2_SQISIGN_MANDATORY: NetworkUpgrade = NetworkUpgrade {
        name: "phase2_sqisign_mandatory",
        activation_height: 2_000_000, // Future activation
        description: "Require SQIsign or hybrid signatures for new blocks",
    };

    // =========================================================================
    // U128 AMOUNT UPGRADE (v2.5.0)
    // =========================================================================

    /// U128 Token Amounts
    ///
    /// Upgrades token amounts from u64 to u128 for:
    /// - Token supplies up to 10^38 (u128 max: ~3.4 × 10^38)
    /// - 24 decimals for native coin (extreme precision)
    /// - Smart contracts with massive token supplies (10^30+)
    ///
    /// Before activation:
    /// - Blocks use u64 amounts (8 decimals)
    /// - P2PBalanceUpdate version 2
    ///
    /// After activation:
    /// - New blocks use u128 amounts (24 decimals)
    /// - P2PBalanceUpdate version 3
    /// - Legacy u64 values automatically converted to u128
    ///
    /// Set to 0 for testnet (immediate activation for testing).
    /// For mainnet: Set to current_height + 50000 (~1 week notice).
    pub const U128_AMOUNTS: NetworkUpgrade = NetworkUpgrade {
        name: "u128_amounts",
        activation_height: 0, // TESTNET: Immediate activation
        description: "Upgrade token amounts to u128 for 10^38 supply support",
    };

    // =========================================================================
    // FEE REDUCTION UPGRADE (v3.4.0-beta)
    // =========================================================================

    /// 10x Transaction Fee Reduction
    ///
    /// Reduces minimum transaction fees by 10x to improve user experience:
    /// - Simple transfer: 0.00021 QUG → 0.000021 QUG
    /// - Token transfer: 0.00042 QUG → 0.000042 QUG
    /// - Swap: 0.00063 QUG → 0.000063 QUG
    /// - Contract call: 0.00105 QUG → 0.000105 QUG
    ///
    /// Before activation:
    /// - MIN_FEE_PER_GAS = 1 (legacy fees)
    /// - Minimum transfer fee = 21,000 gas * 1 = 0.00021 QUG
    ///
    /// After activation:
    /// - get_min_fee_per_gas() returns 0.1 (10x reduction)
    /// - Minimum transfer fee = 21,000 gas * 0.1 = 0.000021 QUG
    ///
    /// Mainnet safety: Height-gated so old blocks validate with old rules.
    /// Set to 350,000 for testnet (~2 weeks notice from current ~300k).
    /// For mainnet: Set to current_height + 20000 before launch.
    pub const REDUCED_FEES_V1: NetworkUpgrade = NetworkUpgrade {
        name: "reduced_fees_v1",
        activation_height: 350_000, // TESTNET: ~2 weeks from now
        description: "10x reduction in transaction fees for better UX",
    };

    // =========================================================================
    // GENUS-2 VDF MINING UPGRADE (v1.0.5)
    // =========================================================================

    /// Genus-2 Jacobian VDF Mining
    ///
    /// Replaces BLAKE3×100 PoW with real Genus-2 hyperelliptic curve VDF:
    /// - Challenge: x = BLAKE3(prev_hash || merkle_root || miner_address || nonce)
    /// - VDF Eval: y = x^(2^T) via sequential squaring in Jacobian J(C)
    /// - Hash check: h = SHA3-256(y), accept if h < difficulty_target
    /// - Proof: Wesolowski proof π for O(log T) verification
    ///
    /// Before activation:
    /// - Mining uses BLAKE3×100 iterated hashing (GPU-parallelizable)
    /// - Server recomputes all 100 BLAKE3 rounds to verify
    ///
    /// After activation:
    /// - Mining uses Genus-2 sequential squaring (inherently sequential = ASIC/GPU resistant)
    /// - Server verifies via O(log T) Wesolowski proof (much faster than recompute)
    /// - MiningSolution includes vdf_output, vdf_proof, vdf_checkpoints fields
    /// - GPU becomes challenge pre-filter; VDF runs on CPU
    ///
    /// Performance (per whitepaper):
    /// - ~1,200 squarings/sec on i7-12700K (128-bit security)
    /// - With T=5,000 min iterations: ~4.2s per VDF eval per nonce
    /// - Target block time adjusts for VDF throughput
    ///
    /// Mainnet safety: Height-gated. Old BLAKE3 blocks validate with old rules.
    /// For mainnet: Set to current_height + 40000 (~4 weeks notice).
    pub const GENUS2_VDF_MINING: NetworkUpgrade = NetworkUpgrade {
        name: "genus2_vdf_mining",
        activation_height: u64::MAX, // NOT YET ACTIVATED — set after testing
        description: "Replace BLAKE3 PoW with Genus-2 Jacobian VDF mining",
    };

    /// Phase B.2: LWMA Difficulty Adjustment (v10.3.0)
    ///
    /// Replaces fixed 16-bit difficulty with LWMA (Linearly Weighted Moving Average)
    /// targeting 1.0 blocks/second. Pure function of chain history — no background
    /// timer, no mutable state. Same inputs → same output on every node.
    ///
    /// Follows the emission controller pattern:
    /// - Called at: challenge endpoint, block template, block validation
    /// - Inputs: recent 120 block timestamps from canonical chain
    /// - Output: difficulty in leading zero bits
    /// - Clamp: [0.5×, 2.0×] per step, floor at 16 bits
    ///
    /// Activation: Instant at fixed height. Before: hardcoded 16 bits.
    /// After: LWMA dynamic, recalculated every block.
    ///
    /// Mainnet safety: Height-gated. Set to current_height + 50000 (~14h at 1bps).
    /// Announce to miners 1 week before activation.
    pub const LWMA_DIFFICULTY_ADJUSTMENT: NetworkUpgrade = NetworkUpgrade {
        name: "lwma_difficulty_adjustment",
        activation_height: 14_900_000, // Activates ~1 hour from height 14887114 (2026-04-13)
        description: "LWMA difficulty adjustment targeting 1.0 bps (Phase B.2)",
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
