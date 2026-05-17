//! Golden Block Registry
//!
//! Golden blocks are known-good blocks that MUST always validate.
//! If a golden block fails validation, the node REFUSES TO START.
//!
//! This catches consensus-breaking changes at startup, not in production.
//!
//! ## How It Works
//!
//! 1. Developer captures blocks at important heights (genesis, upgrades, etc.)
//! 2. Block hash and validation fingerprint are stored in code
//! 3. At node startup, these blocks are re-validated
//! 4. If ANY golden block fails → node crashes with clear error
//!
//! ## Adding a Golden Block
//!
//! ```rust,ignore
//! // In golden_blocks.rs
//! GoldenBlock {
//!     height: 1000000,
//!     hash: "abc123...",
//!     fingerprint: BlockFingerprint { ... },
//!     added_in_version: "1.5.0",
//! }
//! ```

use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::collections::HashMap;
use once_cell::sync::Lazy;
use tracing::{error, info};

/// Fingerprint of a block's validation result
///
/// This captures HOW a block was validated, not just IF it was valid.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockFingerprint {
    /// SHA3-256 of the serialized block
    pub block_hash: [u8; 32],

    /// Hash of the validation rules applied
    pub validation_rules_hash: [u8; 32],

    /// State root after applying this block
    pub post_state_root: [u8; 32],

    /// Active upgrades at this height
    pub active_upgrades: Vec<u32>,
}

impl BlockFingerprint {
    /// Create fingerprint from components
    pub fn new(
        block_bytes: &[u8],
        validation_rules: &str,
        post_state_root: [u8; 32],
        active_upgrades: Vec<u32>,
    ) -> Self {
        let mut block_hasher = Sha3_256::new();
        block_hasher.update(block_bytes);
        let block_hash: [u8; 32] = block_hasher.finalize().into();

        let mut rules_hasher = Sha3_256::new();
        rules_hasher.update(validation_rules.as_bytes());
        let validation_rules_hash: [u8; 32] = rules_hasher.finalize().into();

        Self {
            block_hash,
            validation_rules_hash,
            post_state_root,
            active_upgrades,
        }
    }

    /// Compute a single hash representing the entire fingerprint
    pub fn digest(&self) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(&self.block_hash);
        hasher.update(&self.validation_rules_hash);
        hasher.update(&self.post_state_root);
        for upgrade in &self.active_upgrades {
            hasher.update(&upgrade.to_le_bytes());
        }
        hasher.finalize().into()
    }
}

/// A golden block that must always validate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldenBlock {
    /// Block height
    pub height: u64,

    /// Block hash (hex string)
    pub hash: String,

    /// Expected fingerprint
    pub fingerprint: BlockFingerprint,

    /// Version when this golden block was added
    pub added_in_version: String,

    /// Description of why this block is golden
    pub description: String,
}

/// Registry of all golden blocks
pub struct GoldenBlockRegistry {
    blocks: HashMap<u64, GoldenBlock>,
    is_mainnet: bool,
}

impl GoldenBlockRegistry {
    /// Create new registry
    pub fn new(is_mainnet: bool) -> Self {
        let mut registry = Self {
            blocks: HashMap::new(),
            is_mainnet,
        };

        // Load golden blocks
        if is_mainnet {
            registry.load_mainnet_blocks();
        } else {
            registry.load_testnet_blocks();
        }

        registry
    }

    /// Load mainnet golden blocks
    fn load_mainnet_blocks(&mut self) {
        // Genesis block
        self.add_golden_block(GoldenBlock {
            height: 0,
            hash: "0000000000000000000000000000000000000000000000000000000000000000".to_string(),
            fingerprint: BlockFingerprint {
                block_hash: [0u8; 32],
                validation_rules_hash: [0u8; 32],
                post_state_root: [0u8; 32],
                active_upgrades: vec![0], // Genesis upgrade only
            },
            added_in_version: "1.0.0".to_string(),
            description: "Genesis block - network beginning".to_string(),
        });

        // Add more mainnet golden blocks here as the network grows
        // Example:
        // self.add_golden_block(GoldenBlock {
        //     height: 1_000_000,
        //     hash: "actual_hash_here",
        //     fingerprint: actual_fingerprint,
        //     added_in_version: "1.5.0",
        //     description: "First million blocks milestone",
        // });
    }

    /// Load testnet golden blocks
    fn load_testnet_blocks(&mut self) {
        // Testnet genesis
        self.add_golden_block(GoldenBlock {
            height: 0,
            hash: "0000000000000000000000000000000000000000000000000000000000000000".to_string(),
            fingerprint: BlockFingerprint {
                block_hash: [0u8; 32],
                validation_rules_hash: [0u8; 32],
                post_state_root: [0u8; 32],
                active_upgrades: vec![0],
            },
            added_in_version: "1.0.0".to_string(),
            description: "Testnet genesis".to_string(),
        });
    }

    /// Add a golden block
    fn add_golden_block(&mut self, block: GoldenBlock) {
        self.blocks.insert(block.height, block);
    }

    /// Get a golden block by height
    pub fn get(&self, height: u64) -> Option<&GoldenBlock> {
        self.blocks.get(&height)
    }

    /// Get all golden block heights
    pub fn heights(&self) -> Vec<u64> {
        let mut heights: Vec<_> = self.blocks.keys().copied().collect();
        heights.sort();
        heights
    }

    /// Verify a block matches its golden fingerprint
    ///
    /// Returns error with detailed mismatch info if validation fails.
    pub fn verify_block(
        &self,
        height: u64,
        computed_fingerprint: &BlockFingerprint,
    ) -> Result<(), GoldenBlockMismatch> {
        let golden = match self.blocks.get(&height) {
            Some(g) => g,
            None => return Ok(()), // Not a golden block, no verification needed
        };

        // Compare fingerprints
        if golden.fingerprint.block_hash != computed_fingerprint.block_hash {
            return Err(GoldenBlockMismatch::BlockHashMismatch {
                height,
                expected: hex::encode(golden.fingerprint.block_hash),
                actual: hex::encode(computed_fingerprint.block_hash),
            });
        }

        if golden.fingerprint.validation_rules_hash != computed_fingerprint.validation_rules_hash {
            return Err(GoldenBlockMismatch::ValidationRulesMismatch {
                height,
                expected: hex::encode(golden.fingerprint.validation_rules_hash),
                actual: hex::encode(computed_fingerprint.validation_rules_hash),
            });
        }

        if golden.fingerprint.post_state_root != computed_fingerprint.post_state_root {
            return Err(GoldenBlockMismatch::StateRootMismatch {
                height,
                expected: hex::encode(golden.fingerprint.post_state_root),
                actual: hex::encode(computed_fingerprint.post_state_root),
            });
        }

        if golden.fingerprint.active_upgrades != computed_fingerprint.active_upgrades {
            return Err(GoldenBlockMismatch::UpgradesMismatch {
                height,
                expected: golden.fingerprint.active_upgrades.clone(),
                actual: computed_fingerprint.active_upgrades.clone(),
            });
        }

        Ok(())
    }

    /// Verify all golden blocks in a database
    ///
    /// Called at startup. Panics if any golden block fails.
    pub fn verify_all<F>(&self, get_block_fingerprint: F) -> Result<(), GoldenBlockMismatch>
    where
        F: Fn(u64) -> Option<BlockFingerprint>,
    {
        info!("🔒 [GOLDEN BLOCKS] Verifying {} golden blocks...", self.blocks.len());

        for (height, golden) in &self.blocks {
            match get_block_fingerprint(*height) {
                Some(computed) => {
                    self.verify_block(*height, &computed)?;
                    info!("   ✅ Height {}: {}", height, golden.description);
                }
                None => {
                    // Block not in database yet - OK for initial sync
                    info!("   ⏭️ Height {}: not yet synced", height);
                }
            }
        }

        info!("🔒 [GOLDEN BLOCKS] All verifications passed!");
        Ok(())
    }
}

/// Error when a golden block doesn't match
#[derive(Debug, Clone, thiserror::Error)]
pub enum GoldenBlockMismatch {
    #[error("Block hash mismatch at height {height}: expected {expected}, got {actual}")]
    BlockHashMismatch {
        height: u64,
        expected: String,
        actual: String,
    },

    #[error("Validation rules changed at height {height}: expected {expected}, got {actual}")]
    ValidationRulesMismatch {
        height: u64,
        expected: String,
        actual: String,
    },

    #[error("State root mismatch at height {height}: expected {expected}, got {actual}")]
    StateRootMismatch {
        height: u64,
        expected: String,
        actual: String,
    },

    #[error("Active upgrades mismatch at height {height}: expected {expected:?}, got {actual:?}")]
    UpgradesMismatch {
        height: u64,
        expected: Vec<u32>,
        actual: Vec<u32>,
    },
}

impl GoldenBlockMismatch {
    /// Print detailed error message
    pub fn print_detailed(&self) {
        error!("╔════════════════════════════════════════════════════════════╗");
        error!("║           🚨 GOLDEN BLOCK VERIFICATION FAILED 🚨           ║");
        error!("╠════════════════════════════════════════════════════════════╣");
        error!("║                                                            ║");
        error!("║  A consensus-critical change was detected!                 ║");
        error!("║                                                            ║");
        error!("║  This means your code changes would cause old blocks       ║");
        error!("║  to validate differently than before.                      ║");
        error!("║                                                            ║");
        error!("║  THIS WOULD CORRUPT THE BLOCKCHAIN ON MAINNET.            ║");
        error!("║                                                            ║");
        error!("╠════════════════════════════════════════════════════════════╣");
        error!("║  Error: {}  ║", self);
        error!("╠════════════════════════════════════════════════════════════╣");
        error!("║                                                            ║");
        error!("║  HOW TO FIX:                                              ║");
        error!("║  1. Wrap your change in a height check:                    ║");
        error!("║     if block.height >= UPGRADE_HEIGHT {{ new_rule }}       ║");
        error!("║                                                            ║");
        error!("║  2. Use the UpgradeGate system for all consensus changes   ║");
        error!("║                                                            ║");
        error!("║  3. NEVER change validation for historical blocks          ║");
        error!("║                                                            ║");
        error!("╚════════════════════════════════════════════════════════════╝");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fingerprint_digest() {
        let fp1 = BlockFingerprint::new(b"block1", "rules1", [1u8; 32], vec![0]);
        let fp2 = BlockFingerprint::new(b"block1", "rules1", [1u8; 32], vec![0]);
        let fp3 = BlockFingerprint::new(b"block2", "rules1", [1u8; 32], vec![0]);

        assert_eq!(fp1.digest(), fp2.digest());
        assert_ne!(fp1.digest(), fp3.digest());
    }

    #[test]
    fn test_golden_block_verification() {
        let registry = GoldenBlockRegistry::new(false);

        // Genesis should exist
        assert!(registry.get(0).is_some());

        // Matching fingerprint should pass
        let genesis_fp = BlockFingerprint {
            block_hash: [0u8; 32],
            validation_rules_hash: [0u8; 32],
            post_state_root: [0u8; 32],
            active_upgrades: vec![0],
        };
        assert!(registry.verify_block(0, &genesis_fp).is_ok());

        // Mismatched fingerprint should fail
        let bad_fp = BlockFingerprint {
            block_hash: [1u8; 32], // Wrong!
            validation_rules_hash: [0u8; 32],
            post_state_root: [0u8; 32],
            active_upgrades: vec![0],
        };
        assert!(registry.verify_block(0, &bad_fp).is_err());
    }
}
