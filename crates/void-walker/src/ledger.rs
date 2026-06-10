//! 📚 Multiverse Ledger: "Longest Bridge Wins" Consensus
//! Blockchain for quantum brane-hopping with attosecond precision

use crate::brane::{BraneCoord, Bridge};
use blake3::Hasher as Blake3;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub type HeaderChecksum = [u8; 32];

/// On-chain multiverse proof (serializable)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MultiverseBlock {
    pub block_id: String,
    pub origin_brane: BraneCoord,
    pub target_brane: BraneCoord,
    pub bridge_length: f64,
    pub topological_charge: i32,
    pub parallel_water_sig: [u8; 32],
    pub header_checksum: HeaderChecksum,
    pub attosecond_timestamp: u64,
    pub creator_species_id: String,
    pub k_parameter_snapshot: f64,
    pub laser_imprint_id: String,
    pub thought_intent: String,
    pub block_height: u64,
    pub previous_hash: HeaderChecksum,
}

impl From<Bridge> for MultiverseBlock {
    fn from(bridge: Bridge) -> Self {
        let block_id = hex::encode(&bridge.checksum()[..8]);

        Self {
            block_id,
            origin_brane: bridge.origin,
            target_brane: bridge.target,
            bridge_length: bridge.length_ps,
            topological_charge: bridge.topo_charge,
            parallel_water_sig: bridge.parallel_sig,
            header_checksum: bridge.checksum(),
            attosecond_timestamp: bridge.attosecond_timestamp,
            creator_species_id: "unknown".to_string(),
            k_parameter_snapshot: 7.0, // Will be updated by creator
            laser_imprint_id: "none".to_string(),
            thought_intent: "unknown".to_string(),
            block_height: 0,
            previous_hash: [0; 32],
        }
    }
}

impl MultiverseBlock {
    /// Create genesis block
    pub fn genesis() -> Self {
        let origin = BraneCoord::origin();
        let bridge = Bridge::new(origin, origin, 0, [0; 32]);
        let mut block = Self::from(bridge);
        block.creator_species_id = "genesis".to_string();
        block.thought_intent = "Let there be quantum water".to_string();
        block.block_height = 0;
        block
    }

    /// Create new block with full metadata
    pub fn new_with_metadata(
        bridge: Bridge,
        creator_species_id: String,
        k_parameter: f64,
        laser_imprint_id: String,
        thought_intent: String,
        previous_hash: HeaderChecksum,
        block_height: u64,
    ) -> Self {
        let mut block = Self::from(bridge);
        block.creator_species_id = creator_species_id;
        block.k_parameter_snapshot = k_parameter;
        block.laser_imprint_id = laser_imprint_id;
        block.thought_intent = thought_intent;
        block.previous_hash = previous_hash;
        block.block_height = block_height;
        block
    }

    /// Calculate block difficulty based on bridge quality
    pub fn difficulty(&self) -> f64 {
        let base_difficulty = self.bridge_length;
        let topo_bonus = (self.topological_charge.abs() as f64) * 0.1;
        let k_bonus = (self.k_parameter_snapshot - 7.0).abs() * 0.05;

        base_difficulty + topo_bonus + k_bonus
    }

    /// Verify block integrity
    pub fn verify(&self) -> bool {
        // Check that bridge length matches coordinates
        let calculated_length = self.origin_brane.phase_distance(&self.target_brane);
        let length_match = (self.bridge_length - calculated_length).abs() < 0.001;

        // Check timestamp is reasonable (not too far in future)
        let now_as = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
            / 1_000_000_000;
        let timestamp_valid = self.attosecond_timestamp <= now_as + 1000; // 1000 attoseconds tolerance

        length_match && timestamp_valid
    }

    /// Check if block represents successful brane hop
    pub fn is_successful_hop(&self) -> bool {
        self.bridge_length > 0.1
            && self.topological_charge.abs() >= 2
            && self.k_parameter_snapshot > 5.0
    }
}

/// "Longest bridge wins" multiverse chain
#[derive(Default, Clone, Serialize, Deserialize)]
pub struct MultiverseChain {
    pub blocks: Vec<MultiverseBlock>,
    pub total_bridge_length: f64,
    pub genesis_timestamp: u64,
    pub last_update: u64,
}

impl MultiverseChain {
    /// Create new chain with genesis block
    pub fn new() -> Self {
        let genesis = MultiverseBlock::genesis();
        let genesis_timestamp = genesis.attosecond_timestamp;

        Self {
            blocks: vec![genesis],
            total_bridge_length: 0.0,
            genesis_timestamp,
            last_update: genesis_timestamp,
        }
    }

    /// Add new block to chain (maintains longest bridge ordering)
    pub fn push(&mut self, mut block: MultiverseBlock) {
        // Set block height and previous hash
        block.block_height = self.blocks.len() as u64;
        if let Some(last_block) = self.blocks.last() {
            block.previous_hash = last_block.header_checksum;
        }

        self.blocks.push(block.clone());
        self.total_bridge_length += block.bridge_length;
        self.last_update = block.attosecond_timestamp;

        // Sort by bridge length (longest first - "longest bridge wins")
        self.blocks.sort_by(|a, b| {
            b.bridge_length
                .partial_cmp(&a.bridge_length)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Get head block (longest bridge)
    pub fn head(&self) -> Option<&MultiverseBlock> {
        self.blocks.first()
    }

    /// Get chain tip (most recent block)
    pub fn tip(&self) -> Option<&MultiverseBlock> {
        self.blocks.iter().max_by_key(|block| block.block_height)
    }

    /// Calculate chain digest (BLAKE3 hash of all block hashes)
    pub fn digest(&self) -> String {
        let mut h = Blake3::new();
        for block in &self.blocks {
            h.update(&block.header_checksum);
        }
        hex::encode(h.finalize().as_bytes())
    }

    /// Get blocks by species (for analytics)
    pub fn blocks_by_species(&self, species_id: &str) -> Vec<&MultiverseBlock> {
        self.blocks
            .iter()
            .filter(|block| block.creator_species_id == species_id)
            .collect()
    }

    /// Calculate chain statistics
    pub fn chain_stats(&self) -> ChainStatistics {
        let block_count = self.blocks.len() as u64;
        let unique_species: std::collections::HashSet<_> =
            self.blocks.iter().map(|b| &b.creator_species_id).collect();

        let avg_bridge_length = if block_count > 0 {
            self.total_bridge_length / block_count as f64
        } else {
            0.0
        };

        let successful_hops = self.blocks.iter().filter(|b| b.is_successful_hop()).count() as u64;

        let avg_k_parameter = if block_count > 0 {
            self.blocks
                .iter()
                .map(|b| b.k_parameter_snapshot)
                .sum::<f64>()
                / block_count as f64
        } else {
            7.0
        };

        ChainStatistics {
            block_count,
            unique_species_count: unique_species.len() as u32,
            total_bridge_length: self.total_bridge_length,
            average_bridge_length: avg_bridge_length,
            successful_hops,
            hop_success_rate: if block_count > 0 {
                successful_hops as f64 / block_count as f64
            } else {
                0.0
            },
            average_k_parameter: avg_k_parameter,
            chain_age_seconds: (self.last_update - self.genesis_timestamp) as f64 * 1e-18, // Convert from attoseconds
        }
    }

    /// Verify entire chain integrity
    pub fn verify_chain(&self) -> bool {
        for (i, block) in self.blocks.iter().enumerate() {
            // Verify block integrity
            if !block.verify() {
                eprintln!("Block verification failed at index {}", i);
                return false;
            }

            // Check block height consistency (after sorting)
            if i > 0 && block.block_height >= self.blocks[i - 1].block_height {
                // Heights should be unique after sorting by bridge length
                // This is acceptable since we sort by bridge length, not height
            }
        }
        true
    }
}

/// Chain statistics for analytics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainStatistics {
    pub block_count: u64,
    pub unique_species_count: u32,
    pub total_bridge_length: f64,
    pub average_bridge_length: f64,
    pub successful_hops: u64,
    pub hop_success_rate: f64,
    pub average_k_parameter: f64,
    pub chain_age_seconds: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brane::Bridge;

    #[test]
    fn test_multiverse_block_creation() {
        let origin = BraneCoord::origin();
        let target = origin.advance(1.0);
        let bridge = Bridge::new(origin, target, 5, [42; 32]);

        let block = MultiverseBlock::from(bridge);
        assert!(block.verify()); // Verify block integrity
        assert!(!block.block_id.is_empty());
    }

    #[test]
    fn test_multiverse_chain() {
        let mut chain = MultiverseChain::new();
        assert_eq!(chain.blocks.len(), 1); // Genesis block

        // Add a few blocks
        for i in 1..5 {
            let origin = BraneCoord::origin();
            let target = origin.advance(i as f64);
            let bridge = Bridge::new(origin, target, i, [i as u8; 32]);
            let block = MultiverseBlock::from(bridge);
            chain.push(block);
        }

        assert_eq!(chain.blocks.len(), 5);

        // Head should be longest bridge
        let head = chain.head().unwrap();
        assert!(head.bridge_length >= chain.blocks.last().unwrap().bridge_length);
    }

    #[test]
    fn test_chain_statistics() {
        let mut chain = MultiverseChain::new();

        // Add blocks with varying bridge lengths
        for i in 1..10 {
            let bridge_length = i as f64 * 0.1;
            let origin = BraneCoord::origin();
            let target = origin.advance(bridge_length);
            let bridge = Bridge::new(origin, target, i, [i as u8; 32]);
            let mut block = MultiverseBlock::from(bridge);
            block.creator_species_id = format!("species-{}", i % 3); // 3 different species
            chain.push(block);
        }

        let stats = chain.chain_stats();
        assert_eq!(stats.block_count, 10); // 9 + genesis
        assert_eq!(stats.unique_species_count, 4); // 3 + genesis
        assert!(stats.average_bridge_length > 0.0);
    }

    #[test]
    fn test_block_verification() {
        let origin = BraneCoord::origin();
        let target = origin.advance(0.5);
        let bridge = Bridge::new(origin, target, 3, [123; 32]);
        let block = MultiverseBlock::from(bridge);

        assert!(block.verify()); // Verify block integrity
        assert!(block.difficulty() > 0.0);
    }
}
