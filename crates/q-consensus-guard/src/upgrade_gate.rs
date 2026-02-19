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
}
