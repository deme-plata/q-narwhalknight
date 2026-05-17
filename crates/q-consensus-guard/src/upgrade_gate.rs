//! Height-Gated Upgrade System
//!
//! ALL consensus changes MUST go through this system. No exceptions.
//!
//! ## The Rule
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    THE GOLDEN RULE                               │
//! │                                                                  │
//! │   Old blocks MUST validate with OLD rules.                      │
//! │   New blocks MUST validate with NEW rules.                      │
//! │   There is NO other option.                                     │
//! │                                                                  │
//! │   ❌ WRONG: if use_new_validation { ... }                       │
//! │   ✅ RIGHT: if block.height >= UPGRADE_HEIGHT { ... }           │
//! │                                                                  │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

use once_cell::sync::Lazy;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{info, warn};

/// All possible consensus upgrades
///
/// ADD NEW UPGRADES HERE. Never remove or reorder existing ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u32)]
pub enum Upgrade {
    /// Genesis - no upgrade, always active
    Genesis = 0,

    /// Phase 1: Post-quantum signatures (Dilithium)
    PostQuantumSignatures = 1,

    /// Phase 2: Enhanced block validation
    EnhancedBlockValidation = 2,

    /// Phase 3: New transaction format
    TransactionV2 = 3,

    /// Phase 4: DAG consensus improvements
    DAGConsensusV2 = 4,

    /// Phase 5: Privacy layer
    PrivacyLayer = 5,

    /// Phase 6: Smart contracts V2
    SmartContractsV2 = 6,

    /// Phase 7: State root computation in block headers
    StateRootV1 = 7,

    /// Phase 8: Block evidence required for P2P balance updates
    BlockEvidenceRequired = 8,

    /// Phase 9: Balance state root enforced in block headers
    BalanceRootV1 = 9,

    /// Phase 10: Hybrid Ed25519 + Dilithium5 block signatures.
    ///
    /// Producers MAY emit `SpectralSignature` with
    /// `crypto_phase = HybridEd25519Dilithium5` (both signatures present) once
    /// active. Before activation, producers fall back to Phase0Ed25519 even if
    /// their `ValidatorKeypair::preferred_phase` is set to Hybrid.
    /// Verifiers always accept whatever phase a block carries; this gate only
    /// restricts the producer side so transitional behavior is coordinated.
    HybridSignaturesV1 = 10,

    /// Phase 11: balance_root_v2 — Sparse Merkle Tree balance commitment.
    ///
    /// PRE-ACTIVATION (shadow mode): when env var `Q_BALANCE_ROOT_V2_SHADOW=1`
    /// the node maintains a side-computed SMT root via
    /// `crates/q-storage/src/balance_smt.rs::BalanceSmt`, logs MISMATCH if it
    /// disagrees with the canonical v1 flat hash on each block, but does NOT
    /// enforce the SMT root in headers. v1 stays canonical.
    ///
    /// POST-ACTIVATION (height >= activation_height):
    ///   - `BlockHeader::balance_state_root` MUST equal the SMT root.
    ///   - Producers compute it via `BalanceSmt::apply_to_batch` in the same
    ///     WriteBatch as `save_wallet_balances`.
    ///   - Verifiers re-derive locally and reject mismatches.
    ///
    /// Activation is dormant on mainnet (u64::MAX) and only flips once a soak
    /// period with zero MISMATCH log lines has demonstrated cross-node root
    /// agreement. See docs/deepseek-handoff-balance-root-v2-2026-05-14.md.
    BalanceRootV2 = 11,

    // Add more as needed - NEVER REMOVE OR REORDER
}

impl Upgrade {
    /// Get upgrade name for logging
    pub fn name(&self) -> &'static str {
        match self {
            Upgrade::Genesis => "Genesis",
            Upgrade::PostQuantumSignatures => "PostQuantumSignatures",
            Upgrade::EnhancedBlockValidation => "EnhancedBlockValidation",
            Upgrade::TransactionV2 => "TransactionV2",
            Upgrade::DAGConsensusV2 => "DAGConsensusV2",
            Upgrade::PrivacyLayer => "PrivacyLayer",
            Upgrade::SmartContractsV2 => "SmartContractsV2",
            Upgrade::StateRootV1 => "StateRootV1",
            Upgrade::BlockEvidenceRequired => "BlockEvidenceRequired",
            Upgrade::BalanceRootV1 => "BalanceRootV1",
            Upgrade::HybridSignaturesV1 => "HybridSignaturesV1",
            Upgrade::BalanceRootV2 => "BalanceRootV2",
        }
    }
}

/// Configuration for an upgrade
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpgradeConfig {
    /// Block height at which upgrade activates
    pub activation_height: u64,

    /// Human-readable description
    pub description: String,

    /// Whether this upgrade is mandatory (node must support it)
    pub mandatory: bool,

    /// Minimum node version required
    pub min_version: String,
}

/// Mainnet upgrade schedule
///
/// THIS IS THE SOURCE OF TRUTH. Update carefully!
pub static MAINNET_UPGRADES: Lazy<HashMap<Upgrade, UpgradeConfig>> = Lazy::new(|| {
    let mut upgrades = HashMap::new();

    // Genesis - always active
    upgrades.insert(Upgrade::Genesis, UpgradeConfig {
        activation_height: 0,
        description: "Genesis block".to_string(),
        mandatory: true,
        min_version: "0.0.1".to_string(),
    });

    // Post-quantum signatures - not yet scheduled
    upgrades.insert(Upgrade::PostQuantumSignatures, UpgradeConfig {
        activation_height: u64::MAX, // Not scheduled yet
        description: "Enable Dilithium post-quantum signatures".to_string(),
        mandatory: false,
        min_version: "2.0.0".to_string(),
    });

    // State root computation - not yet scheduled for mainnet
    upgrades.insert(Upgrade::StateRootV1, UpgradeConfig {
        activation_height: u64::MAX,
        description: "Compute real state root in block headers".to_string(),
        mandatory: false,
        min_version: "5.1.0".to_string(),
    });

    // Block evidence required - not yet scheduled for mainnet
    upgrades.insert(Upgrade::BlockEvidenceRequired, UpgradeConfig {
        activation_height: u64::MAX,
        description: "Require block hash evidence for P2P balance updates".to_string(),
        mandatory: false,
        min_version: "5.1.0".to_string(),
    });

    // Balance state root enforcement - bumped to 20,000,000 to allow shadow-mode soak (was 18,600,000)
    // Shadow mode must prove root agreement across all nodes before enforcement activates.
    upgrades.insert(Upgrade::BalanceRootV1, UpgradeConfig {
        activation_height: 20_000_000, // ~15 days from tip ~17.67M; shadow soak first
        description: "Enforce balance state root in block headers".to_string(),
        mandatory: true,
        min_version: "10.6.0".to_string(),
    });

    // Hybrid Ed25519 + Dilithium5 producer-side gate.
    // Dormant on mainnet (u64::MAX) until a concrete activation height is
    // chosen after canary soak. Until then, producers with
    // preferred_phase = Hybrid fall back to Phase0Ed25519.
    upgrades.insert(Upgrade::HybridSignaturesV1, UpgradeConfig {
        activation_height: u64::MAX,
        description: "Allow producers to emit Hybrid Ed25519+Dilithium5 signatures".to_string(),
        mandatory: false,
        min_version: "10.9.20".to_string(),
    });

    // balance_root_v2 — Sparse Merkle Tree balance commitment.
    // DORMANT on mainnet (u64::MAX). Before flipping, we need:
    //   (1) Shadow mode (Q_BALANCE_ROOT_V2_SHADOW=1) running across all nodes
    //       for at least one week with zero MISMATCH log entries.
    //   (2) Cross-node determinism test green (multi-node simulation produces
    //       byte-identical SMT roots at every height).
    //   (3) Reorg-correctness test green (SMT matches canonical chain after
    //       a multi-block reorg).
    //   (4) Activation rebuild path verified — every node successfully
    //       `BalanceSmt::rebuild_from_balances(&wallet_table)` at the
    //       activation height with deterministic root agreement.
    upgrades.insert(Upgrade::BalanceRootV2, UpgradeConfig {
        activation_height: u64::MAX,
        description: "Enforce balance_root_v2 (Sparse Merkle Tree) in block headers".to_string(),
        mandatory: true,
        min_version: "10.9.22".to_string(),
    });

    // Add more upgrades here as they are scheduled

    upgrades
});

/// Testnet upgrade schedule (faster activation for testing)
pub static TESTNET_UPGRADES: Lazy<HashMap<Upgrade, UpgradeConfig>> = Lazy::new(|| {
    let mut upgrades = HashMap::new();

    upgrades.insert(Upgrade::Genesis, UpgradeConfig {
        activation_height: 0,
        description: "Genesis block".to_string(),
        mandatory: true,
        min_version: "0.0.1".to_string(),
    });

    // Post-quantum - activate at block 100000 on testnet
    upgrades.insert(Upgrade::PostQuantumSignatures, UpgradeConfig {
        activation_height: 100_000,
        description: "Enable Dilithium post-quantum signatures".to_string(),
        mandatory: false,
        min_version: "2.0.0".to_string(),
    });

    // State root computation - activate immediately on testnet for testing
    upgrades.insert(Upgrade::StateRootV1, UpgradeConfig {
        activation_height: 0,
        description: "Compute real state root in block headers".to_string(),
        mandatory: false,
        min_version: "5.1.0".to_string(),
    });

    // Block evidence required for P2P balance updates - activate immediately on testnet
    upgrades.insert(Upgrade::BlockEvidenceRequired, UpgradeConfig {
        activation_height: 0,
        description: "Require block hash evidence for P2P balance updates".to_string(),
        mandatory: false,
        min_version: "5.1.0".to_string(),
    });

    // Balance state root enforcement - immediate on testnet for testing
    upgrades.insert(Upgrade::BalanceRootV1, UpgradeConfig {
        activation_height: 0, // Immediate for testnet
        description: "Enforce balance state root in block headers".to_string(),
        mandatory: true,
        min_version: "10.6.0".to_string(),
    });

    // Hybrid signatures - immediate on testnet so canaries exercise the path
    upgrades.insert(Upgrade::HybridSignaturesV1, UpgradeConfig {
        activation_height: 0,
        description: "Allow producers to emit Hybrid Ed25519+Dilithium5 signatures".to_string(),
        mandatory: false,
        min_version: "10.9.20".to_string(),
    });

    // balance_root_v2 — immediate on testnet so the SMT path is exercised on
    // every block, the shadow-mode plumbing is hot from genesis, and any
    // determinism issues surface in CI before mainnet activation.
    upgrades.insert(Upgrade::BalanceRootV2, UpgradeConfig {
        activation_height: 0,
        description: "Enforce balance_root_v2 (Sparse Merkle Tree) in block headers".to_string(),
        mandatory: true,
        min_version: "10.9.22".to_string(),
    });

    upgrades
});

/// The Upgrade Gate - controls which features are active at which height
pub struct UpgradeGate {
    /// Current network (mainnet/testnet)
    is_mainnet: bool,

    /// Override heights for testing
    overrides: RwLock<HashMap<Upgrade, u64>>,
}

impl UpgradeGate {
    /// Create new upgrade gate
    pub fn new(is_mainnet: bool) -> Self {
        Self {
            is_mainnet,
            overrides: RwLock::new(HashMap::new()),
        }
    }

    /// Check if an upgrade is active at a given height
    ///
    /// THIS IS THE FUNCTION TO USE IN ALL CONSENSUS CODE
    #[inline]
    pub fn is_active(&self, upgrade: Upgrade, block_height: u64) -> bool {
        // Check overrides first (for testing)
        if let Some(&override_height) = self.overrides.read().get(&upgrade) {
            return block_height >= override_height;
        }

        // Get from schedule
        let schedule = if self.is_mainnet {
            &MAINNET_UPGRADES
        } else {
            &TESTNET_UPGRADES
        };

        schedule
            .get(&upgrade)
            .map(|config| block_height >= config.activation_height)
            .unwrap_or(false)
    }

    /// Get activation height for an upgrade
    pub fn activation_height(&self, upgrade: Upgrade) -> Option<u64> {
        // Check overrides first
        if let Some(&override_height) = self.overrides.read().get(&upgrade) {
            return Some(override_height);
        }

        let schedule = if self.is_mainnet {
            &MAINNET_UPGRADES
        } else {
            &TESTNET_UPGRADES
        };

        schedule.get(&upgrade).map(|c| c.activation_height)
    }

    /// Override activation height (FOR TESTING ONLY)
    #[cfg(any(test, feature = "testing"))]
    pub fn override_height(&self, upgrade: Upgrade, height: u64) {
        warn!(
            "⚠️ [UPGRADE GATE] Overriding {} activation to height {} (TESTING ONLY)",
            upgrade.name(), height
        );
        self.overrides.write().insert(upgrade, height);
    }

    /// List all pending upgrades
    pub fn pending_upgrades(&self, current_height: u64) -> Vec<(Upgrade, u64)> {
        let schedule = if self.is_mainnet {
            &MAINNET_UPGRADES
        } else {
            &TESTNET_UPGRADES
        };

        schedule
            .iter()
            .filter(|(_, config)| config.activation_height > current_height)
            .filter(|(_, config)| config.activation_height != u64::MAX)
            .map(|(upgrade, config)| (*upgrade, config.activation_height))
            .collect()
    }

    /// Log upgrade status at startup
    pub fn log_status(&self, current_height: u64) {
        let schedule = if self.is_mainnet {
            &MAINNET_UPGRADES
        } else {
            &TESTNET_UPGRADES
        };

        info!("🔐 [UPGRADE GATE] Status at height {}:", current_height);

        for (upgrade, config) in schedule.iter() {
            let status = if config.activation_height == u64::MAX {
                "NOT SCHEDULED".to_string()
            } else if current_height >= config.activation_height {
                format!("✅ ACTIVE (since {})", config.activation_height)
            } else {
                format!("⏳ PENDING (at {})", config.activation_height)
            };

            info!("   {} - {}: {}", upgrade.name(), config.description, status);
        }
    }
}

/// Global upgrade gate instance
static GLOBAL_GATE: Lazy<RwLock<Option<UpgradeGate>>> = Lazy::new(|| RwLock::new(None));

/// Initialize the global upgrade gate (call once at startup)
pub fn init_upgrade_gate(is_mainnet: bool) {
    let gate = UpgradeGate::new(is_mainnet);
    *GLOBAL_GATE.write() = Some(gate);
    info!(
        "🔐 [UPGRADE GATE] Initialized for {}",
        if is_mainnet { "MAINNET" } else { "TESTNET" }
    );
}

/// Check if upgrade is active (convenience function)
///
/// # Panics
/// Panics if upgrade gate not initialized
#[inline]
pub fn is_upgrade_active(upgrade: Upgrade, block_height: u64) -> bool {
    GLOBAL_GATE
        .read()
        .as_ref()
        .expect("Upgrade gate not initialized! Call init_upgrade_gate() first")
        .is_active(upgrade, block_height)
}

/// Example of how to use in consensus code:
///
/// ```rust,ignore
/// use q_consensus_guard::{Upgrade, is_upgrade_active};
///
/// fn validate_signature(block: &Block) -> Result<()> {
///     if is_upgrade_active(Upgrade::PostQuantumSignatures, block.height) {
///         // New rule: require post-quantum signatures
///         verify_dilithium(block)?;
///     } else {
///         // Old rule: Ed25519 still valid
///         verify_ed25519(block)?;
///     }
///     Ok(())
/// }
/// ```

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_upgrade_activation() {
        let gate = UpgradeGate::new(false); // testnet

        // Genesis always active
        assert!(gate.is_active(Upgrade::Genesis, 0));
        assert!(gate.is_active(Upgrade::Genesis, 1_000_000));

        // PQ sigs active at 100000 on testnet
        assert!(!gate.is_active(Upgrade::PostQuantumSignatures, 99_999));
        assert!(gate.is_active(Upgrade::PostQuantumSignatures, 100_000));
        assert!(gate.is_active(Upgrade::PostQuantumSignatures, 100_001));
    }

    #[test]
    fn test_mainnet_not_scheduled() {
        let gate = UpgradeGate::new(true); // mainnet

        // PQ sigs NOT scheduled on mainnet (u64::MAX)
        assert!(!gate.is_active(Upgrade::PostQuantumSignatures, 1_000_000));
        assert!(!gate.is_active(Upgrade::PostQuantumSignatures, u64::MAX - 1));
    }

    #[test]
    fn test_hybrid_signatures_dormant_on_mainnet() {
        // v10.9.20: HybridSignaturesV1 must stay dormant on mainnet until a
        // concrete activation height is chosen. Until then, producers fall
        // back to Phase0Ed25519 even if their preferred_phase is Hybrid.
        let mainnet = UpgradeGate::new(true);
        assert!(!mainnet.is_active(Upgrade::HybridSignaturesV1, 0));
        assert!(!mainnet.is_active(Upgrade::HybridSignaturesV1, 17_700_000));
        assert!(!mainnet.is_active(Upgrade::HybridSignaturesV1, u64::MAX - 1));

        // Testnet activates immediately so canary nodes exercise the path.
        let testnet = UpgradeGate::new(false);
        assert!(testnet.is_active(Upgrade::HybridSignaturesV1, 0));
        assert!(testnet.is_active(Upgrade::HybridSignaturesV1, 1_000_000));
    }

    #[test]
    fn test_balance_root_v2_dormant_on_mainnet() {
        // v10.9.22: BalanceRootV2 must stay dormant on mainnet until at least
        // one week of green shadow-mode soak demonstrates cross-node SMT root
        // agreement. Activation is a separate, deliberate operator decision —
        // it is NOT controlled by this test. Until then, the SMT module sits
        // unwired and balance_root_v1 stays canonical.
        let mainnet = UpgradeGate::new(true);
        assert!(!mainnet.is_active(Upgrade::BalanceRootV2, 0));
        assert!(!mainnet.is_active(Upgrade::BalanceRootV2, 17_700_000));
        assert!(!mainnet.is_active(Upgrade::BalanceRootV2, 20_000_000));
        assert!(!mainnet.is_active(Upgrade::BalanceRootV2, u64::MAX - 1));

        // Testnet activates immediately so the SMT path is exercised on
        // every block from genesis and any determinism issues surface in CI.
        let testnet = UpgradeGate::new(false);
        assert!(testnet.is_active(Upgrade::BalanceRootV2, 0));
        assert!(testnet.is_active(Upgrade::BalanceRootV2, 1));
        assert!(testnet.is_active(Upgrade::BalanceRootV2, 1_000_000));
    }

    #[test]
    fn test_balance_root_v2_is_distinct_from_v1() {
        // The v2 SMT upgrade is independent of the v1 flat-hash upgrade.
        // Don't accidentally couple them — operators should be able to
        // activate v1 enforcement without flipping v2, and vice versa.
        let mainnet = UpgradeGate::new(true);
        // BalanceRootV1 has a real activation height (20,000,000); V2 is u64::MAX.
        assert!(!mainnet.is_active(Upgrade::BalanceRootV2, 20_000_000));
        // But BalanceRootV1 *is* active at that height.
        assert!(mainnet.is_active(Upgrade::BalanceRootV1, 20_000_000));
    }
}
