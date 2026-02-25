// ============================================================================
// v1.1.21-beta: Genesis Checkpoint Validator (Phase 15 Fork Prevention)
// ============================================================================
//
// PROBLEM SOLVED:
// ---------------
// When nodes mine independently (due to connectivity failures), they create
// different blockchain chains from different starting points. When sync is
// restored, these chains cannot merge because they have different genesis blocks.
//
// SOLUTION (Inspired by Kaspa DAG-Knight):
// ----------------------------------------
// 1. Hardcode genesis block hash for each network phase into the binary
// 2. On first sync, validate that blocks chain back to the hardcoded genesis
// 3. Reject blocks from peers on incompatible chains
// 4. Use checkpoint hashes at known heights as additional validation
//
// REFERENCES:
// -----------
// - Kaspa DAG KNIGHT: https://eprint.iacr.org/2022/1494.pdf
// - Kaspa fork recovery (Nov 2021): Hardwired new genesis block
// - GhostDAG pruning point checkpoints for chain validation
//
// ============================================================================

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, error, info, warn};

use q_types::block::{BlockHash, QBlock};
use q_types::NetworkId;

/// Genesis block checkpoint for fork prevention
/// Hardcoded into binary - cannot be changed without recompilation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenesisCheckpoint {
    /// Network phase this checkpoint applies to
    pub network_id: NetworkId,

    /// The genesis block hash (block at height 0 or 1)
    pub genesis_block_hash: BlockHash,

    /// Height of the genesis block (usually 0 or 1)
    pub genesis_height: u64,

    /// Previous block hash in genesis (all zeros for true genesis)
    pub genesis_prev_hash: BlockHash,

    /// Network name for display
    pub network_name: String,

    /// Description of this phase
    pub description: String,
}

/// Additional checkpoint at a known height (pruning point equivalent)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeightCheckpoint {
    /// Block height
    pub height: u64,

    /// Expected block hash at this height
    pub block_hash: BlockHash,

    /// Previous block hash (for chain validation)
    pub prev_hash: BlockHash,
}

/// Chain validation result
#[derive(Debug, Clone)]
pub enum ChainValidationResult {
    /// Block is on the correct chain
    Valid,

    /// Block's genesis doesn't match our genesis
    GenesisHashMismatch {
        expected: BlockHash,
        found: BlockHash,
    },

    /// Block's prev_hash doesn't match known block
    ParentHashMismatch {
        height: u64,
        expected_parent: BlockHash,
        found_parent: BlockHash,
    },

    /// Block height 1's prev_hash doesn't match genesis
    GenesisChainBroken {
        expected_genesis: BlockHash,
        found_prev_hash: BlockHash,
    },

    /// Block failed checkpoint validation
    CheckpointMismatch {
        checkpoint_height: u64,
        expected_hash: BlockHash,
        found_hash: BlockHash,
    },

    /// Cannot validate - need more blocks to trace back to genesis
    NeedMoreBlocks {
        lowest_height_needed: u64,
    },
}

/// Genesis checkpoint validator (Kaspa-style fork prevention)
pub struct GenesisCheckpointValidator {
    /// Genesis checkpoints per network phase
    genesis_checkpoints: HashMap<NetworkId, GenesisCheckpoint>,

    /// Height checkpoints (pruning points) per network phase
    height_checkpoints: HashMap<NetworkId, Vec<HeightCheckpoint>>,

    /// Cache of validated block hashes (height -> hash)
    validated_chain_cache: HashMap<u64, BlockHash>,

    /// Current network ID
    current_network: NetworkId,
}

impl GenesisCheckpointValidator {
    /// Create a new validator with hardcoded genesis checkpoints
    ///
    /// CRITICAL: These values are compiled into the binary.
    /// To change them, you must recompile and redeploy.
    /// This is INTENTIONAL - prevents malicious genesis tampering.
    pub fn new(network_id: NetworkId) -> Self {
        let mut genesis_checkpoints = HashMap::new();
        let height_checkpoints: HashMap<NetworkId, Vec<HeightCheckpoint>> = HashMap::new();

        // ========================================================================
        // PHASE 14: Database Durability & P2P Security (v1.1.9-beta to v1.1.20)
        // Genesis: First block mined on bootstrap node after Phase 14 database reset
        // ========================================================================
        genesis_checkpoints.insert(
            NetworkId::TestnetPhase14,
            GenesisCheckpoint {
                network_id: NetworkId::TestnetPhase14,
                // ⚠️ This hash will be set when Phase 14 chain is confirmed stable
                // For now, use placeholder that accepts any genesis during transition
                genesis_block_hash: [0u8; 32], // Placeholder - to be hardcoded after chain stabilization
                genesis_height: 1, // Phase 14 starts at height 1 (not 0)
                genesis_prev_hash: [0u8; 32], // Genesis has no parent
                network_name: "testnet-phase14".to_string(),
                description: "Phase 14: Database Durability & P2P Security with WAL guarantees".to_string(),
            },
        );

        // ========================================================================
        // PHASE 15: Safe Batched Sync & Genesis Checkpoint (v1.1.22-beta)
        // Genesis: First block mined on bootstrap node after Phase 15 database reset
        // Key features: Connected peer selection, genesis checkpoint validation
        // ========================================================================
        genesis_checkpoints.insert(
            NetworkId::TestnetPhase15,
            GenesisCheckpoint {
                network_id: NetworkId::TestnetPhase15,
                // ⚠️ This hash will be set when Phase 15 chain is confirmed stable
                // For now, use placeholder that accepts any genesis during transition
                genesis_block_hash: [0u8; 32], // Placeholder - to be hardcoded after chain stabilization
                genesis_height: 1, // Phase 15 starts at height 1 (not 0)
                genesis_prev_hash: [0u8; 32], // Genesis has no parent
                network_name: "testnet-phase15".to_string(),
                description: "Phase 15: Safe Batched Sync with Genesis Checkpoint - fork prevention via connected peer selection".to_string(),
            },
        );

        // ========================================================================
        // MAINNET: (FUTURE)
        // Genesis: Will be hardcoded before mainnet launch
        // ========================================================================
        genesis_checkpoints.insert(
            NetworkId::Mainnet2026,
            GenesisCheckpoint {
                network_id: NetworkId::Mainnet2026,
                genesis_block_hash: [0u8; 32],
                genesis_height: 1,
                genesis_prev_hash: [0u8; 32],
                network_name: "mainnet2026".to_string(),
                description: "Q-NarwhalKnight Mainnet 2026 (deprecated)".to_string(),
            },
        );

        // v7.1.2: Mainnet 2026.1 - Clean relaunch with emission fix + data isolation
        genesis_checkpoints.insert(
            NetworkId::Mainnet2026_1,
            GenesisCheckpoint {
                network_id: NetworkId::Mainnet2026_1,
                genesis_block_hash: [0u8; 32],
                genesis_height: 1,
                genesis_prev_hash: [0u8; 32],
                network_name: "mainnet2026.1".to_string(),
                description: "Q-NarwhalKnight Mainnet 2026.1 - Production network (deprecated)".to_string(),
            },
        );

        // v7.3.2: Mainnet 2026.1.1 - 4-day rehearsal chain (Feb 18-22, 2026)
        genesis_checkpoints.insert(
            NetworkId::Mainnet2026_1_1,
            GenesisCheckpoint {
                network_id: NetworkId::Mainnet2026_1_1,
                genesis_block_hash: [0u8; 32],
                genesis_height: 1,
                genesis_prev_hash: [0u8; 32],
                network_name: "mainnet2026.1.1".to_string(),
                description: "Q-NarwhalKnight Mainnet 2026.1.1 - Rehearsal chain".to_string(),
            },
        );

        // v8.0.1: Mainnet 2026.1.3 - Emission rehearsal with rate fix
        genesis_checkpoints.insert(
            NetworkId::Mainnet2026_1_3,
            GenesisCheckpoint {
                network_id: NetworkId::Mainnet2026_1_3,
                genesis_block_hash: [0u8; 32],
                genesis_height: 1,
                genesis_prev_hash: [0u8; 32],
                network_name: "mainnet2026.1.3".to_string(),
                description: "Q-NarwhalKnight Mainnet 2026.1.3 - Emission rehearsal with rate fix".to_string(),
            },
        );

        // v7.3.0: Mainnet 2026.2 - Fresh directory relaunch with zero contamination
        genesis_checkpoints.insert(
            NetworkId::MainnetGenesis,
            GenesisCheckpoint {
                network_id: NetworkId::MainnetGenesis,
                genesis_block_hash: [0u8; 32],
                genesis_height: 1,
                genesis_prev_hash: [0u8; 32],
                network_name: "mainnet-genesis".to_string(),
                description: "Q-NarwhalKnight Mainnet 2026.2 - Production network".to_string(),
            },
        );

        Self {
            genesis_checkpoints,
            height_checkpoints,
            validated_chain_cache: HashMap::new(),
            current_network: network_id,
        }
    }

    /// Set the genesis block hash for Phase 15 at runtime
    /// This should only be called ONCE during Phase 15 transition
    pub fn set_phase15_genesis(&mut self, genesis_hash: BlockHash) {
        warn!("🔐 [GENESIS] Setting Phase 15 genesis checkpoint: {:?}...",
              hex::encode(&genesis_hash[..8]));

        // Note: In production, this should be hardcoded, not set at runtime
        // This method exists only for the Phase 14 -> 15 transition period
        if let Some(checkpoint) = self.genesis_checkpoints.get_mut(&self.current_network) {
            checkpoint.genesis_block_hash = genesis_hash;
        }
    }

    /// Add a height checkpoint (pruning point)
    pub fn add_height_checkpoint(&mut self, network: NetworkId, checkpoint: HeightCheckpoint) {
        self.height_checkpoints
            .entry(network)
            .or_insert_with(Vec::new)
            .push(checkpoint);
    }

    /// Get the genesis checkpoint for the current network
    pub fn get_genesis_checkpoint(&self) -> Option<&GenesisCheckpoint> {
        self.genesis_checkpoints.get(&self.current_network)
    }

    /// Check if genesis hash is set (not placeholder zeros)
    pub fn is_genesis_hardcoded(&self) -> bool {
        if let Some(checkpoint) = self.get_genesis_checkpoint() {
            checkpoint.genesis_block_hash != [0u8; 32]
        } else {
            false
        }
    }

    /// Validate that a block is on the correct chain
    ///
    /// This is the core fork prevention check:
    /// 1. If block is at genesis height, verify its hash matches our genesis
    /// 2. If block is at height 1, verify prev_hash is our genesis
    /// 3. If we have the parent block in cache, verify chain continuity
    /// 4. If block matches a height checkpoint, verify hash
    pub fn validate_block_chain(&mut self, block: &QBlock) -> ChainValidationResult {
        let block_hash = block.calculate_hash();
        let height = block.header.height;
        let prev_hash = block.header.prev_block_hash;

        // Skip validation if genesis is not hardcoded (transition period)
        if !self.is_genesis_hardcoded() {
            debug!("⚠️ [GENESIS] Skipping chain validation - genesis not yet hardcoded");
            // Still cache the block for future validation
            self.validated_chain_cache.insert(height, block_hash);
            return ChainValidationResult::Valid;
        }

        let genesis = self.get_genesis_checkpoint().unwrap();

        // =====================================================================
        // Check 1: Genesis block validation
        // =====================================================================
        if height == genesis.genesis_height {
            if block_hash != genesis.genesis_block_hash {
                error!("🚨 [GENESIS] FORK DETECTED: Block at genesis height {} has wrong hash!",
                       height);
                error!("   Expected: {}...", hex::encode(&genesis.genesis_block_hash[..16]));
                error!("   Found:    {}...", hex::encode(&block_hash[..16]));
                return ChainValidationResult::GenesisHashMismatch {
                    expected: genesis.genesis_block_hash,
                    found: block_hash,
                };
            }

            info!("✅ [GENESIS] Block at height {} matches hardcoded genesis checkpoint", height);
            self.validated_chain_cache.insert(height, block_hash);
            return ChainValidationResult::Valid;
        }

        // =====================================================================
        // Check 2: Block at height 2 must chain to genesis
        // =====================================================================
        if height == genesis.genesis_height + 1 {
            if prev_hash != genesis.genesis_block_hash {
                error!("🚨 [GENESIS] FORK DETECTED: Block at height {} doesn't chain to genesis!",
                       height);
                error!("   Expected prev_hash: {}...", hex::encode(&genesis.genesis_block_hash[..16]));
                error!("   Found prev_hash:    {}...", hex::encode(&prev_hash[..16]));
                return ChainValidationResult::GenesisChainBroken {
                    expected_genesis: genesis.genesis_block_hash,
                    found_prev_hash: prev_hash,
                };
            }

            info!("✅ [GENESIS] Block at height {} correctly chains to genesis", height);
            self.validated_chain_cache.insert(height, block_hash);
            return ChainValidationResult::Valid;
        }

        // =====================================================================
        // Check 3: Parent hash validation against cached blocks
        // =====================================================================
        if height > 1 {
            let parent_height = height - 1;
            if let Some(&cached_parent_hash) = self.validated_chain_cache.get(&parent_height) {
                if prev_hash != cached_parent_hash {
                    error!("🚨 [GENESIS] CHAIN BREAK: Block {} prev_hash doesn't match block {}!",
                           height, parent_height);
                    error!("   Cached parent:  {}...", hex::encode(&cached_parent_hash[..16]));
                    error!("   Block prev:     {}...", hex::encode(&prev_hash[..16]));
                    return ChainValidationResult::ParentHashMismatch {
                        height,
                        expected_parent: cached_parent_hash,
                        found_parent: prev_hash,
                    };
                }

                debug!("✅ [GENESIS] Block {} chains correctly to validated block {}", height, parent_height);
            } else if parent_height > genesis.genesis_height {
                // We don't have the parent cached - request more blocks
                debug!("⏳ [GENESIS] Need parent block {} to validate block {}", parent_height, height);
                return ChainValidationResult::NeedMoreBlocks {
                    lowest_height_needed: parent_height,
                };
            }
        }

        // =====================================================================
        // Check 4: Height checkpoint validation
        // =====================================================================
        if let Some(checkpoints) = self.height_checkpoints.get(&self.current_network) {
            for checkpoint in checkpoints {
                if checkpoint.height == height {
                    if block_hash != checkpoint.block_hash {
                        error!("🚨 [CHECKPOINT] Block {} doesn't match height checkpoint!",
                               height);
                        error!("   Expected: {}...", hex::encode(&checkpoint.block_hash[..16]));
                        error!("   Found:    {}...", hex::encode(&block_hash[..16]));
                        return ChainValidationResult::CheckpointMismatch {
                            checkpoint_height: height,
                            expected_hash: checkpoint.block_hash,
                            found_hash: block_hash,
                        };
                    }
                    info!("✅ [CHECKPOINT] Block {} matches height checkpoint", height);
                }
            }
        }

        // Block passed all checks
        self.validated_chain_cache.insert(height, block_hash);
        ChainValidationResult::Valid
    }

    /// Validate a batch of blocks in order (lowest height first)
    /// Returns the first validation failure, or Valid if all pass
    pub fn validate_block_batch(&mut self, blocks: &[QBlock]) -> Result<()> {
        // Sort by height (lowest first)
        let mut sorted_blocks: Vec<&QBlock> = blocks.iter().collect();
        sorted_blocks.sort_by_key(|b| b.header.height);

        for block in sorted_blocks {
            match self.validate_block_chain(block) {
                ChainValidationResult::Valid => continue,
                ChainValidationResult::NeedMoreBlocks { lowest_height_needed } => {
                    // This is OK during sync - we'll validate when we get more blocks
                    debug!("⏳ [GENESIS] Deferred validation for block {} (need height {})",
                           block.header.height, lowest_height_needed);
                    continue;
                }
                ChainValidationResult::GenesisHashMismatch { expected, found } => {
                    return Err(anyhow!(
                        "FORK REJECTED: Genesis hash mismatch (expected {}..., found {}...)",
                        hex::encode(&expected[..8]), hex::encode(&found[..8])
                    ));
                }
                ChainValidationResult::GenesisChainBroken { expected_genesis, found_prev_hash } => {
                    return Err(anyhow!(
                        "FORK REJECTED: Block doesn't chain to genesis (expected {}..., found {}...)",
                        hex::encode(&expected_genesis[..8]), hex::encode(&found_prev_hash[..8])
                    ));
                }
                ChainValidationResult::ParentHashMismatch { height, expected_parent, found_parent } => {
                    return Err(anyhow!(
                        "FORK REJECTED: Chain break at height {} (expected parent {}..., found {}...)",
                        height, hex::encode(&expected_parent[..8]), hex::encode(&found_parent[..8])
                    ));
                }
                ChainValidationResult::CheckpointMismatch { checkpoint_height, expected_hash, found_hash } => {
                    return Err(anyhow!(
                        "FORK REJECTED: Checkpoint mismatch at height {} (expected {}..., found {}...)",
                        checkpoint_height, hex::encode(&expected_hash[..8]), hex::encode(&found_hash[..8])
                    ));
                }
            }
        }

        Ok(())
    }

    /// Cache a known-good block hash
    /// Call this for blocks that come from our own chain
    pub fn cache_local_block(&mut self, height: u64, hash: BlockHash) {
        self.validated_chain_cache.insert(height, hash);
    }

    /// Clear the validation cache (useful after database reset)
    pub fn clear_cache(&mut self) {
        self.validated_chain_cache.clear();
    }

    /// Get statistics about validation state
    pub fn get_stats(&self) -> (usize, Option<u64>, Option<u64>) {
        let cache_size = self.validated_chain_cache.len();
        let min_height = self.validated_chain_cache.keys().min().copied();
        let max_height = self.validated_chain_cache.keys().max().copied();
        (cache_size, min_height, max_height)
    }
}

// ============================================================================
// Helper functions for genesis hash management
// ============================================================================

/// Compute genesis block hash from a QBlock
/// This should be called with block at height 1 from the bootstrap node
/// to get the hash that will be hardcoded into the Phase 15 binary
pub fn compute_genesis_hash_from_block(block: &QBlock) -> BlockHash {
    let genesis_hash = block.calculate_hash();

    info!("🔐 [GENESIS] Computed genesis hash from block:");
    info!("   Height: {}", block.header.height);
    info!("   Hash: {}", hex::encode(&genesis_hash));
    info!("   Prev: {}", hex::encode(&block.header.prev_block_hash));

    genesis_hash
}

/// Print a block hash as a Rust array literal for hardcoding into source
pub fn format_hash_as_rust_array(hash: &BlockHash) -> String {
    let mut result = String::from("[\n");
    for (i, byte) in hash.iter().enumerate() {
        if i % 8 == 0 {
            result.push_str("    ");
        }
        result.push_str(&format!("0x{:02x}, ", byte));
        if i % 8 == 7 {
            result.push('\n');
        }
    }
    result.push(']');
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genesis_validator_creation() {
        let validator = GenesisCheckpointValidator::new(NetworkId::TestnetPhase14);
        assert!(validator.get_genesis_checkpoint().is_some());

        // Phase 14 should have placeholder genesis (not hardcoded yet)
        assert!(!validator.is_genesis_hardcoded());
    }

    #[test]
    fn test_validation_skipped_when_not_hardcoded() {
        let mut validator = GenesisCheckpointValidator::new(NetworkId::TestnetPhase14);

        // Create a mock block
        let block = create_test_block(1, [0u8; 32], [0xab; 32]);

        // Should return Valid because genesis is not hardcoded (transition period)
        match validator.validate_block_chain(&block) {
            ChainValidationResult::Valid => {}
            other => panic!("Expected Valid during transition period, got {:?}", other),
        }
    }

    #[test]
    fn test_genesis_mismatch_detection() {
        let mut validator = GenesisCheckpointValidator::new(NetworkId::TestnetPhase14);

        // Hardcode a test genesis
        let expected_genesis: BlockHash = [0x42; 32];
        validator.set_phase15_genesis(expected_genesis);

        // Create a block at genesis height with wrong hash
        let wrong_block = create_test_block(1, [0u8; 32], [0xde; 32]);

        match validator.validate_block_chain(&wrong_block) {
            ChainValidationResult::GenesisHashMismatch { expected, found } => {
                assert_eq!(expected, expected_genesis);
                assert_ne!(found, expected_genesis);
            }
            other => panic!("Expected GenesisHashMismatch, got {:?}", other),
        }
    }

    // Helper function to create test blocks
    fn create_test_block(height: u64, prev_hash: BlockHash, hash_seed: [u8; 32]) -> QBlock {
        use q_types::block::{BlockHeader, QuantumMetadata};

        QBlock {
            header: BlockHeader {
                height,
                phase: 14,
                network_id: "testnet-phase14".to_string(),
                prev_block_hash: prev_hash,
                merkle_root: hash_seed,
                timestamp: std::time::SystemTime::now(),
                difficulty: 1000,
                nonce: 0,
                miner_address: "test_miner".to_string(),
                total_mined_coins: 0,
            },
            mining_solutions: vec![],
            dag_parents: vec![],
            quantum_metadata: QuantumMetadata::default(),
            transactions: vec![],
            balance_updates: vec![],
            size_bytes: 0,
        }
    }
}
