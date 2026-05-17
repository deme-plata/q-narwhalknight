/// v1.4.11-beta: Stake-Weighted Finality
///
/// Provides enhanced finality guarantees by combining:
/// - Block confirmations (PoW security)
/// - Staker attestations (PoS security)
/// - Economic finality thresholds
///
/// Finality Formula:
/// effective_confirmations = raw_confirmations × (1 + stake_weight_factor)
///
/// Where stake_weight_factor = (attesting_stake / total_stake) × MAX_STAKE_BONUS
/// MAX_STAKE_BONUS = 3.0 (stakers can triple effective confirmations) // v8.6.0
///
/// A block is considered:
/// - Probabilistic Finality: 1+ confirmations (can be reorged)
/// - Economic Finality: 3 effective confirmations OR 33% stake attestation
/// - Absolute Finality: 6 effective confirmations AND 67% stake attestation

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Maximum bonus from stake attestations (3x = 300%)
/// v8.6.0: Increased from 2.0 to 3.0 — stronger finality boost for stakers accelerates
/// economic finality without reducing security (more stake = more skin in the game)
pub const MAX_STAKE_BONUS: f64 = 3.0;

/// Minimum stake ratio for economic finality
pub const ECONOMIC_FINALITY_THRESHOLD: f64 = 0.33;

/// Minimum stake ratio for absolute finality
pub const ABSOLUTE_FINALITY_THRESHOLD: f64 = 0.67;

/// Effective confirmations for economic finality
pub const ECONOMIC_CONFIRMATION_TARGET: u64 = 3; // v8.6.0: Halved for 2x faster finality

/// Effective confirmations for absolute finality
pub const ABSOLUTE_CONFIRMATION_TARGET: u64 = 6; // v8.6.0: Halved for 2x faster finality

/// Finality level for a block
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FinalityLevel {
    /// Just mined, not yet confirmed
    Pending,
    /// 1+ confirmations, can be reorged
    Probabilistic,
    /// 3 effective confirmations OR 33% stake
    Economic,
    /// 6 effective confirmations AND 67% stake
    Absolute,
}

impl FinalityLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            FinalityLevel::Pending => "pending",
            FinalityLevel::Probabilistic => "probabilistic",
            FinalityLevel::Economic => "economic",
            FinalityLevel::Absolute => "absolute",
        }
    }

    pub fn security_bits(&self) -> u32 {
        match self {
            FinalityLevel::Pending => 0,
            FinalityLevel::Probabilistic => 16,
            FinalityLevel::Economic => 40,
            FinalityLevel::Absolute => 80,
        }
    }
}

/// Staker attestation for a block
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockAttestation {
    /// Block hash being attested
    pub block_hash: [u8; 32],
    /// Block height
    pub block_height: u64,
    /// Staker's wallet address
    pub staker_address: String,
    /// Amount staked by this staker
    pub staked_amount: u64,
    /// Attestation timestamp
    pub timestamp: u64,
    /// Signature of attestation (hex-encoded)
    pub signature: Option<String>,
}

/// Block finality record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockFinalityRecord {
    /// Block hash
    pub block_hash: [u8; 32],
    /// Block height
    pub block_height: u64,
    /// Raw confirmations (blocks on top)
    pub raw_confirmations: u64,
    /// Effective confirmations (with stake bonus)
    pub effective_confirmations: f64,
    /// Total stake that has attested this block
    pub attesting_stake: u64,
    /// Total stake in the network
    pub total_stake: u64,
    /// Stake ratio (attesting_stake / total_stake)
    pub stake_ratio: f64,
    /// Current finality level
    pub finality_level: FinalityLevel,
    /// Stakers who have attested
    pub attesters: HashSet<String>,
    /// Timestamp of first attestation
    pub first_attestation: Option<u64>,
    /// Timestamp finality was achieved
    pub finality_achieved: Option<u64>,
}

/// Stake-Weighted Finality Manager
pub struct StakeWeightedFinalityManager {
    /// Block finality records: block_hash -> BlockFinalityRecord
    block_records: Arc<RwLock<HashMap<[u8; 32], BlockFinalityRecord>>>,
    /// Block height to hash mapping
    height_to_hash: Arc<RwLock<HashMap<u64, [u8; 32]>>>,
    /// Current network total stake
    total_network_stake: Arc<RwLock<u64>>,
    /// Current chain height
    chain_height: Arc<RwLock<u64>>,
}

impl Default for StakeWeightedFinalityManager {
    fn default() -> Self {
        Self::new()
    }
}

impl StakeWeightedFinalityManager {
    /// Create new manager
    pub fn new() -> Self {
        Self {
            block_records: Arc::new(RwLock::new(HashMap::new())),
            height_to_hash: Arc::new(RwLock::new(HashMap::new())),
            total_network_stake: Arc::new(RwLock::new(0)),
            chain_height: Arc::new(RwLock::new(0)),
        }
    }

    /// Update total network stake
    pub async fn update_total_stake(&self, total_stake: u64) {
        *self.total_network_stake.write().await = total_stake;
    }

    /// Update chain height (triggers confirmation updates)
    pub async fn update_chain_height(&self, new_height: u64) {
        let old_height = *self.chain_height.read().await;
        if new_height <= old_height {
            return;
        }

        *self.chain_height.write().await = new_height;

        // Update confirmations for all tracked blocks
        let mut records = self.block_records.write().await;
        let total_stake = *self.total_network_stake.read().await;

        for record in records.values_mut() {
            if record.block_height <= new_height {
                record.raw_confirmations = new_height - record.block_height + 1;
                Self::recalculate_finality(record, total_stake);
            }
        }
    }

    /// Register a new block for finality tracking
    pub async fn register_block(&self, block_hash: [u8; 32], block_height: u64) {
        let current_height = *self.chain_height.read().await;
        let total_stake = *self.total_network_stake.read().await;

        let raw_confirmations = if block_height <= current_height {
            current_height - block_height + 1
        } else {
            0
        };

        let record = BlockFinalityRecord {
            block_hash,
            block_height,
            raw_confirmations,
            effective_confirmations: raw_confirmations as f64,
            attesting_stake: 0,
            total_stake,
            stake_ratio: 0.0,
            finality_level: if raw_confirmations >= 1 {
                FinalityLevel::Probabilistic
            } else {
                FinalityLevel::Pending
            },
            attesters: HashSet::new(),
            first_attestation: None,
            finality_achieved: None,
        };

        let mut records = self.block_records.write().await;
        records.insert(block_hash, record);

        let mut height_map = self.height_to_hash.write().await;
        height_map.insert(block_height, block_hash);

        debug!(
            "📋 [FINALITY] Registered block {} at height {} ({} confirmations)",
            hex::encode(&block_hash[..8]),
            block_height,
            raw_confirmations
        );
    }

    /// Process a staker attestation
    pub async fn process_attestation(
        &self,
        attestation: BlockAttestation,
    ) -> Result<FinalityLevel, String> {
        let mut records = self.block_records.write().await;
        let total_stake = *self.total_network_stake.read().await;

        let record = records
            .get_mut(&attestation.block_hash)
            .ok_or("Block not found in finality tracking")?;

        // Check if already attested by this staker
        if record.attesters.contains(&attestation.staker_address) {
            return Err("Staker already attested this block".to_string());
        }

        // Add attestation
        record.attesters.insert(attestation.staker_address.clone());
        record.attesting_stake = record
            .attesting_stake
            .saturating_add(attestation.staked_amount);

        if record.first_attestation.is_none() {
            record.first_attestation = Some(attestation.timestamp);
        }

        // Recalculate finality
        Self::recalculate_finality(record, total_stake);

        let old_level = record.finality_level.clone();
        let new_level = record.finality_level.clone();

        if new_level != old_level {
            info!(
                "⬆️ [FINALITY] Block {} upgraded: {:?} -> {:?}",
                hex::encode(&attestation.block_hash[..8]),
                old_level,
                new_level
            );

            if matches!(new_level, FinalityLevel::Economic | FinalityLevel::Absolute) {
                record.finality_achieved = Some(attestation.timestamp);
            }
        }

        Ok(record.finality_level.clone())
    }

    /// Recalculate finality level for a block
    fn recalculate_finality(record: &mut BlockFinalityRecord, total_stake: u64) {
        // Calculate stake ratio
        record.total_stake = total_stake;
        record.stake_ratio = if total_stake > 0 {
            record.attesting_stake as f64 / total_stake as f64
        } else {
            0.0
        };

        // Calculate stake weight bonus
        let stake_bonus = record.stake_ratio.min(1.0) * MAX_STAKE_BONUS;

        // Effective confirmations = raw × (1 + bonus)
        record.effective_confirmations =
            record.raw_confirmations as f64 * (1.0 + stake_bonus);

        // Determine finality level
        record.finality_level = if record.effective_confirmations >= ABSOLUTE_CONFIRMATION_TARGET as f64
            && record.stake_ratio >= ABSOLUTE_FINALITY_THRESHOLD
        {
            FinalityLevel::Absolute
        } else if record.effective_confirmations >= ECONOMIC_CONFIRMATION_TARGET as f64
            || record.stake_ratio >= ECONOMIC_FINALITY_THRESHOLD
        {
            FinalityLevel::Economic
        } else if record.raw_confirmations >= 1 {
            FinalityLevel::Probabilistic
        } else {
            FinalityLevel::Pending
        };
    }

    /// Get finality status for a block
    pub async fn get_block_finality(&self, block_hash: &[u8; 32]) -> Option<BlockFinalityRecord> {
        self.block_records.read().await.get(block_hash).cloned()
    }

    /// Get finality status by height
    pub async fn get_finality_by_height(&self, height: u64) -> Option<BlockFinalityRecord> {
        let height_map = self.height_to_hash.read().await;
        let block_hash = height_map.get(&height)?;
        self.block_records.read().await.get(block_hash).cloned()
    }

    /// Get blocks with specific finality level
    pub async fn get_blocks_by_finality(&self, level: FinalityLevel) -> Vec<BlockFinalityRecord> {
        self.block_records
            .read()
            .await
            .values()
            .filter(|r| r.finality_level == level)
            .cloned()
            .collect()
    }

    /// Check if a block has achieved at least economic finality
    pub async fn is_final(&self, block_hash: &[u8; 32]) -> bool {
        self.block_records
            .read()
            .await
            .get(block_hash)
            .map(|r| matches!(r.finality_level, FinalityLevel::Economic | FinalityLevel::Absolute))
            .unwrap_or(false)
    }

    /// Get finality statistics
    pub async fn get_stats(&self) -> FinalityStats {
        let records = self.block_records.read().await;
        let chain_height = *self.chain_height.read().await;
        let total_stake = *self.total_network_stake.read().await;

        let mut pending = 0;
        let mut probabilistic = 0;
        let mut economic = 0;
        let mut absolute = 0;

        for record in records.values() {
            match record.finality_level {
                FinalityLevel::Pending => pending += 1,
                FinalityLevel::Probabilistic => probabilistic += 1,
                FinalityLevel::Economic => economic += 1,
                FinalityLevel::Absolute => absolute += 1,
            }
        }

        FinalityStats {
            total_tracked: records.len(),
            pending_finality: pending,
            probabilistic_finality: probabilistic,
            economic_finality: economic,
            absolute_finality: absolute,
            chain_height,
            total_network_stake: total_stake,
        }
    }

    /// Cleanup old records (keep last N blocks)
    pub async fn cleanup_old_records(&self, keep_last_n: u64) {
        let chain_height = *self.chain_height.read().await;
        if chain_height <= keep_last_n {
            return;
        }

        let cutoff = chain_height - keep_last_n;
        let mut to_remove = Vec::new();

        {
            let records = self.block_records.read().await;
            for (hash, record) in records.iter() {
                if record.block_height < cutoff {
                    to_remove.push(*hash);
                }
            }
        }

        if !to_remove.is_empty() {
            let mut records = self.block_records.write().await;
            let mut height_map = self.height_to_hash.write().await;

            for hash in &to_remove {
                if let Some(record) = records.remove(hash) {
                    height_map.remove(&record.block_height);
                }
            }

            info!(
                "🧹 [FINALITY] Cleaned up {} old finality records",
                to_remove.len()
            );
        }
    }
}

/// Finality statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinalityStats {
    pub total_tracked: usize,
    pub pending_finality: usize,
    pub probabilistic_finality: usize,
    pub economic_finality: usize,
    pub absolute_finality: usize,
    pub chain_height: u64,
    pub total_network_stake: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_finality_progression() {
        let manager = StakeWeightedFinalityManager::new();

        // Set up network state
        manager.update_total_stake(1_000_000).await;
        manager.update_chain_height(100).await;

        // Register a block
        let block_hash = [1u8; 32];
        manager.register_block(block_hash, 95).await;

        // Check initial finality (5 confirmations = probabilistic)
        let record = manager.get_block_finality(&block_hash).await.unwrap();
        assert_eq!(record.finality_level, FinalityLevel::Probabilistic);
        assert_eq!(record.raw_confirmations, 6); // 100 - 95 + 1

        // Add stake attestations (33% = economic finality)
        let attestation = BlockAttestation {
            block_hash,
            block_height: 95,
            staker_address: "qnk123".to_string(),
            staked_amount: 330_000, // 33%
            timestamp: 1000,
            signature: None,
        };

        let level = manager.process_attestation(attestation).await.unwrap();
        assert_eq!(level, FinalityLevel::Economic);

        // Add more stake (67% = absolute finality with enough confirmations)
        manager.update_chain_height(107).await; // 12+ raw confirmations

        let attestation2 = BlockAttestation {
            block_hash,
            block_height: 95,
            staker_address: "qnk456".to_string(),
            staked_amount: 340_000, // Total now 67%
            timestamp: 1001,
            signature: None,
        };

        let level = manager.process_attestation(attestation2).await.unwrap();
        assert_eq!(level, FinalityLevel::Absolute);
    }
}
