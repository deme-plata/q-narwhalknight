//! # Q-Consensus-Guard: Automatic Mainnet Safety
//!
//! This crate makes mainnet data corruption **impossible** through:
//!
//! 1. **Immutable Block Validation** - Old blocks ALWAYS validate the same way
//! 2. **Height-Gated Upgrades** - All consensus changes require activation height
//! 3. **Golden Block Tests** - Known blocks must always pass validation
//! 4. **Serialization Fingerprints** - Detect breaking serialization changes
//! 5. **State Transition Determinism** - Same inputs = same outputs, always
//!
//! ## How It Works
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                    CONSENSUS GUARD ARCHITECTURE                      │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │                                                                      │
//! │   COMPILE TIME                        RUNTIME                        │
//! │   ────────────                        ───────                        │
//! │                                                                      │
//! │   ┌──────────────┐                   ┌──────────────────────┐       │
//! │   │ #[consensus] │                   │ ConsensusGuard       │       │
//! │   │ attribute    │──────────────────▶│ - validates blocks   │       │
//! │   │ marks code   │                   │ - checks heights     │       │
//! │   └──────────────┘                   │ - verifies hashes    │       │
//! │                                      └──────────────────────┘       │
//! │   ┌──────────────┐                            │                     │
//! │   │ Golden Tests │                            ▼                     │
//! │   │ CI/CD fails  │◀──────────────────┌──────────────────────┐       │
//! │   │ if mismatch  │                   │ GoldenBlockRegistry  │       │
//! │   └──────────────┘                   │ - canonical blocks   │       │
//! │                                      │ - expected hashes    │       │
//! │   ┌──────────────┐                   │ - validation rules   │       │
//! │   │ Serialization│                   └──────────────────────┘       │
//! │   │ Fingerprints │                            │                     │
//! │   │ detect breaks│◀───────────────────────────┘                     │
//! │   └──────────────┘                                                  │
//! │                                                                      │
//! │   IF ANY CHECK FAILS → BUILD FAILS / NODE REFUSES TO START          │
//! │                                                                      │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use q_consensus_guard::{ConsensusGuard, UpgradeGate, GoldenBlock};
//!
//! // At node startup - MUST pass or node won't start
//! ConsensusGuard::verify_or_panic();
//!
//! // For consensus-critical code
//! fn validate_block(block: &Block) -> Result<()> {
//!     // Height-gated upgrade - safe!
//!     if UpgradeGate::is_active(Upgrade::PostQuantumSigs, block.height) {
//!         verify_dilithium(block)?;
//!     } else {
//!         verify_ed25519(block)?;
//!     }
//!     Ok(())
//! }
//! ```

pub mod golden_blocks;
pub mod upgrade_gate;
pub mod serialization_guard;
pub mod determinism;
pub mod guard;

pub use golden_blocks::{GoldenBlock, GoldenBlockRegistry, BlockFingerprint};
pub use upgrade_gate::{Upgrade, UpgradeGate, UpgradeConfig, MAINNET_UPGRADES, is_upgrade_active};
pub use serialization_guard::{SerializationGuard, TypeFingerprint};
pub use determinism::{DeterminismChecker, StateTransition};
pub use guard::{ConsensusGuard, ConsensusViolation, GuardConfig};

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::{
        ConsensusGuard, ConsensusViolation, GuardConfig,
        GoldenBlock, GoldenBlockRegistry,
        Upgrade, UpgradeGate,
        SerializationGuard,
        DeterminismChecker,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_guard_compiles() {
        // Basic smoke test
        let guard = ConsensusGuard::new(GuardConfig::default());
        assert!(guard.is_ok());
    }
}
