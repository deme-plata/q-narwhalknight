use crate::block::QuantumPoWBlock;
use q_precision::{QAmount, gas_optimization::GasCosts};
// use dilithium::verify_signature;  // TODO: Use q-aegis-ql for signature verification
use sha3::{Digest, Sha3_256};
use std::collections::HashMap;
use tokio::sync::RwLock;
use std::sync::Arc;
use anyhow::Result;
use thiserror::Error;

/// Mining error types
#[derive(Debug, Error)]
pub enum MiningError {
    #[error("Invalid quantum quality: {0}")]
    InvalidQuantumQuality(f64),
    #[error("Signature verification failed")]
    SignatureVerificationFailed,
    #[error("Invalid signature")]
    InvalidSignature,
    #[error("Invalid block structure: {0}")]
    InvalidBlock(String),
    #[error("Reward calculation failed: {0}")]
    RewardCalculationFailed(String),
}

/// Quantum-enhanced mining rewards calculator and validator
#[derive(Debug, Clone)]
pub struct RewardCalculator {
    /// Current mining configuration
    config: RewardConfig,
    /// Block height to reward mapping cache
    reward_cache: Arc<RwLock<HashMap<u64, u64>>>,
    /// Burn address for deflationary mechanism
    burn_address: [u8; 32],
    /// Statistics tracking
    stats: Arc<RwLock<RewardStats>>,
}

/// Mining reward configuration with ultra-precision
#[derive(Debug, Clone)]
pub struct RewardConfig {
    /// Base reward amount (2.0 QNK with ultra-precision)
    pub base_reward: QAmount,
    /// Halving interval in blocks (1M blocks ≈ 1 year)
    pub halving_interval: u64,
    /// Maximum total supply (21M QNK with ultra-precision)
    pub max_supply: QAmount,
    /// Quantum enhancement bonus percentage (0-100)
    pub quantum_bonus_percent: u8,
    /// Burn rate percentage (25%)
    pub burn_rate_percent: u8,
    /// Minimum quantum quality for bonus (0.0-1.0)
    pub min_quantum_quality: f64,
    /// Gas costs for mining operations
    pub gas_costs: GasCosts,
}

/// Reward calculation result with ultra-precision
#[derive(Debug, Clone)]
pub struct RewardResult {
    /// Base mining reward (ultra-precision QAmount)
    pub base_reward: QAmount,
    /// Quantum enhancement bonus (ultra-precision QAmount)
    pub quantum_bonus: QAmount,
    /// Total reward before burn (ultra-precision QAmount)
    pub total_reward: QAmount,
    /// Amount to burn (deflationary)
    pub burn_amount: QAmount,
    /// Final reward to miner (ultra-precision QAmount)
    pub final_reward: QAmount,
    /// Gas cost for reward calculation
    pub gas_cost: QAmount,
    /// Quantum quality assessment
    pub quantum_quality: f64,
    /// Reason for reward amount
    pub calculation_reason: String,
}

/// Reward validation statistics
#[derive(Debug, Clone)]
pub struct RewardStats {
    /// Total rewards distributed
    pub total_distributed: QAmount,
    /// Total amount burned
    pub total_burned: QAmount,
    /// Quantum bonuses awarded
    pub quantum_bonuses: QAmount,
    /// Invalid reward attempts
    pub invalid_attempts: u64,
    /// Average quantum quality
    pub avg_quantum_quality: f64,
    /// Blocks processed
    pub blocks_processed: u64,
}

impl Default for RewardStats {
    fn default() -> Self {
        Self {
            total_distributed: QAmount::ZERO,
            total_burned: QAmount::ZERO,
            quantum_bonuses: QAmount::ZERO,
            invalid_attempts: 0,
            avg_quantum_quality: 0.0,
            blocks_processed: 0,
        }
    }
}

/// Reward validation error types
#[derive(Debug, Clone)]
pub enum RewardValidationError {
    /// Reward amount exceeds allowed maximum
    ExcessiveReward { claimed: u64, maximum: u64 },
    /// Invalid quantum quality score
    InvalidQuantumQuality(f64),
    /// Signature verification failed
    InvalidSignature,
    /// Block height invalid
    InvalidHeight(u64),
    /// Quantum bonus not justified
    UnauthorizedQuantumBonus,
    /// Burn calculation error
    BurnCalculationError,
}

impl Default for RewardConfig {
    fn default() -> Self {
        // ✅ v0.9.60-beta Phase 6: AUSTRIAN ECONOMICS - TRUE SCARCITY
        //
        // Phase 5 Problem: 998,663 QNK mined in days = HYPERINFLATION
        // Phase 6 Solution: 0.5 QNK per block = 100x MORE SCARCE
        //
        // Emission Schedule (Bitcoin-inspired):
        // - Epoch 1 (0-210k): 0.5 QNK/block → 105,000 QNK
        // - Epoch 2 (210k-420k): 0.25 QNK/block → 52,500 QNK
        // - Epoch 3 (420k-630k): 0.125 QNK/block → 26,250 QNK
        // - Epoch 4 (630k-840k): 0.0625 QNK/block → 13,125 QNK
        // - Epoch 5+ (840k+): 0.03125 QNK/block (tail emission)
        //
        // Max Supply: ~200,000 QNK (vs Phase 5's million in days!)
        // Inflation Year 4+: <1% (sound money, comparable to gold)
        //
        // See PHASE6_AUSTRIAN_ECONOMICS.md for full analysis
        Self {
            base_reward: QAmount::from_qwei(50_000_000), // 0.5 QNK (8 decimals) - 100x less than Phase 5!
            halving_interval: 210_000, // Bitcoin-style 210k blocks (~146 days)
            max_supply: QAmount::from_qwei(200_000_000_000_000), // 200k QNK total (asymptotic)
            quantum_bonus_percent: 10, // 10% max bonus for high-quality quantum randomness
            burn_rate_percent: 0, // NO BURN in Phase 6 (scarcity via low emission)
            min_quantum_quality: 0.9, // 90% entropy quality required for bonus
            gas_costs: GasCosts::default(),
        }
    }
}

impl RewardCalculator {
    /// Create new reward calculator with configuration
    pub fn new(config: RewardConfig) -> Self {
        // Generate deterministic burn address from genesis
        let mut hasher = Sha3_256::new();
        hasher.update(b"QNK_BURN_ADDRESS_V1");
        let burn_hash = hasher.finalize();
        let mut burn_address = [0u8; 32];
        burn_address.copy_from_slice(&burn_hash);

        Self {
            config,
            reward_cache: Arc::new(RwLock::new(HashMap::new())),
            burn_address,
            stats: Arc::new(RwLock::new(RewardStats::default())),
        }
    }

    /// Calculate quantum-enhanced reward for a mined block
    pub async fn calculate_reward(&self, block: &QuantumPoWBlock) -> Result<RewardResult, MiningError> {
        // Check cache first for performance
        let cache_key = block.header.height;
        {
            let cache = self.reward_cache.read().await;
            if let Some(&cached_reward) = cache.get(&cache_key) {
                return self.build_cached_result(cached_reward, block).await;
            }
        }

        // Calculate base reward with halving
        let halving_count = block.header.height / self.config.halving_interval;
        let base_reward = if halving_count >= 64 {
            QAmount::ZERO // After 64 halvings, no more rewards
        } else {
            self.config.base_reward >> halving_count
        };

        // Assess quantum quality for bonus calculation
        let quantum_quality = self.assess_quantum_quality(block)?;

        // Calculate quantum enhancement bonus
        let quantum_bonus = if quantum_quality >= self.config.min_quantum_quality {
            let bonus_rate = self.config.quantum_bonus_percent as f64 / 100.0;
            // Scale bonus based on quality (linear from min_quality to 1.0)
            let quality_factor = (quantum_quality - self.config.min_quantum_quality)
                / (1.0 - self.config.min_quantum_quality);
            let bonus_qwei = (base_reward.to_qwei() as f64 * bonus_rate * quality_factor) as i128;
            QAmount::from_qwei(bonus_qwei)
        } else {
            QAmount::ZERO
        };

        let total_reward = base_reward + quantum_bonus;

        // Calculate burn amount (deflationary mechanism)
        let burn_rate_qamount = QAmount::from_qwei(self.config.burn_rate_percent as i128);
        let hundred = QAmount::from_qwei(100);
        let burn_amount = (total_reward * burn_rate_qamount) / hundred;
        let final_reward = total_reward - burn_amount;

        // Calculate gas cost for this reward calculation
        let gas_cost = QAmount::from_qwei(self.config.gas_costs.reward_calculation as i128);

        // Build detailed result
        let result = RewardResult {
            base_reward,
            quantum_bonus,
            total_reward,
            burn_amount,
            final_reward,
            gas_cost,
            quantum_quality,
            calculation_reason: format!(
                "Height: {}, Halving: {}, Quality: {:.3}, Bonus: {}%",
                block.header.height,
                halving_count,
                quantum_quality,
                if quantum_bonus > QAmount::ZERO { self.config.quantum_bonus_percent } else { 0 }
            ),
        };

        // Cache the result (convert QAmount to u64 for storage)
        {
            let mut cache = self.reward_cache.write().await;
            cache.insert(cache_key, final_reward.to_qwei() as u64);
        }

        // Update statistics
        self.update_stats(&result).await;

        Ok(result)
    }

    /// Validate a mining reward claim
    pub async fn validate_reward_claim(
        &self,
        block: &QuantumPoWBlock,
        claimed_reward: u64,
    ) -> Result<(), RewardValidationError> {
        // Calculate expected reward
        let expected = self.calculate_reward(block).await
            .map_err(|_| RewardValidationError::InvalidHeight(block.header.height))?;

        // Convert claimed reward to QAmount for comparison
        let claimed_qamount = QAmount::from_qwei(claimed_reward as i128);

        // Verify claimed amount matches expected
        if claimed_qamount > expected.final_reward {
            return Err(RewardValidationError::ExcessiveReward {
                claimed: claimed_reward,
                maximum: expected.final_reward.to_qwei() as u64,
            });
        }

        // Verify quantum quality assessment
        let quantum_quality = self.assess_quantum_quality(block)
            .map_err(|_| RewardValidationError::InvalidQuantumQuality(0.0))?;

        if quantum_quality < 0.0 || quantum_quality > 1.0 {
            return Err(RewardValidationError::InvalidQuantumQuality(quantum_quality));
        }

        // Verify quantum bonus justification
        if expected.quantum_bonus > QAmount::ZERO && quantum_quality < self.config.min_quantum_quality {
            return Err(RewardValidationError::UnauthorizedQuantumBonus);
        }

        // Verify miner signature
        self.verify_miner_signature(block)
            .map_err(|_| RewardValidationError::InvalidSignature)?;

        // Update validation statistics
        {
            let mut stats = self.stats.write().await;
            stats.blocks_processed += 1;
        }

        Ok(())
    }

    /// Assess quantum enhancement quality for bonus calculation
    fn assess_quantum_quality(&self, block: &QuantumPoWBlock) -> Result<f64, MiningError> {
        let quantum_data = &block.quantum_data;

        // Base quality from VDF proof entropy
        let mut quality = quantum_data.entropy_quality;

        // Boost quality if quantum seed was successfully injected
        if quantum_data.quantum_seed.is_some() {
            quality = (quality + 0.1).min(1.0);
        }

        // Boost quality based on VDF proof complexity
        if let Some(vdf_proof) = &quantum_data.vdf_proof {
            // VDF proof quality assessment would go here
            // For now, just use a simple boost
            quality = (quality + 0.05).min(1.0);
        }

        Ok(quality)
    }

    /// Assess VDF proof quality for quantum enhancement
    fn assess_vdf_proof_quality(&self, vdf_proof: &[u8]) -> f64 {
        // Simple entropy-based quality assessment
        let mut entropy = 0.0;
        let mut byte_counts = [0u32; 256];
        
        for &byte in vdf_proof {
            byte_counts[byte as usize] += 1;
        }

        let total_bytes = vdf_proof.len() as f64;
        for count in byte_counts.iter() {
            if *count > 0 {
                let probability = *count as f64 / total_bytes;
                entropy -= probability * probability.log2();
            }
        }

        // Normalize entropy to 0-1 range (8 bits = perfect entropy)
        (entropy / 8.0).min(1.0)
    }

    /// Verify miner signature for reward claim
    fn verify_miner_signature(&self, block: &QuantumPoWBlock) -> Result<(), MiningError> {
        // Create message for signature verification
        let mut message = Vec::new();
        message.extend_from_slice(&block.header.parent_hash);
        message.extend_from_slice(&block.header.height.to_be_bytes());
        message.extend_from_slice(&block.header.timestamp.to_be_bytes());
        message.extend_from_slice(&block.header.nonce.to_be_bytes());
        message.extend_from_slice(&block.header.miner_address);

        // Verify signature length (Dilithium5 signatures are ~4595 bytes)
        if block.signature.len() < 100 {
            return Err(MiningError::InvalidSignature);
        }

        // TODO: Implement full Dilithium5 signature verification when q-aegis-ql provides verify_signature
        // For now, verify basic signature structure
        Ok(())
    }

    /// Build cached reward result
    async fn build_cached_result(&self, cached_reward: u64, block: &QuantumPoWBlock) -> Result<RewardResult, MiningError> {
        let quantum_quality = self.assess_quantum_quality(block)?;

        let reward_amount = QAmount::from_qwei(cached_reward as i128);

        Ok(RewardResult {
            base_reward: reward_amount,
            quantum_bonus: QAmount::ZERO,
            total_reward: reward_amount,
            burn_amount: QAmount::ZERO,
            final_reward: reward_amount,
            gas_cost: QAmount::from_qwei(self.config.gas_costs.reward_calculation as i128),
            quantum_quality,
            calculation_reason: "Cached result".to_string(),
        })
    }

    /// Update reward statistics
    async fn update_stats(&self, result: &RewardResult) {
        let mut stats = self.stats.write().await;
        stats.total_distributed += result.final_reward;
        stats.total_burned += result.burn_amount;
        stats.quantum_bonuses += result.quantum_bonus;
        stats.blocks_processed += 1;
        
        // Update rolling average quantum quality
        let total_quality = stats.avg_quantum_quality * (stats.blocks_processed - 1) as f64;
        stats.avg_quantum_quality = (total_quality + result.quantum_quality) / stats.blocks_processed as f64;
    }

    /// Get current reward statistics
    pub async fn get_statistics(&self) -> RewardStats {
        self.stats.read().await.clone()
    }

    /// Get burn address for deflationary mechanism
    pub fn get_burn_address(&self) -> [u8; 32] {
        self.burn_address
    }

    /// Calculate total supply at given height
    pub fn calculate_total_supply(&self, height: u64) -> QAmount {
        let mut total_supply = QAmount::ZERO;
        let mut current_reward = self.config.base_reward;
        let mut blocks_processed = 0u64;

        while blocks_processed < height && current_reward > QAmount::ZERO {
            let blocks_until_halving = self.config.halving_interval.min(height - blocks_processed);
            let supply_this_period = current_reward * QAmount::from_qwei(blocks_until_halving as i128);

            // Apply burn rate (rewards are net of burn)
            let burn_rate = QAmount::from_qwei((100 - self.config.burn_rate_percent as i128));
            let net_supply = (supply_this_period * burn_rate) / QAmount::from_qwei(100);
            total_supply = total_supply + net_supply;

            blocks_processed += blocks_until_halving;
            current_reward >>= 1; // Halve reward
        }

        if total_supply > self.config.max_supply {
            self.config.max_supply
        } else {
            total_supply
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::{BlockHeader, MiningData, QuantumData};

    fn create_test_block(height: u64, quantum_quality: Option<f64>) -> QuantumPoWBlock {
        QuantumPoWBlock {
            header: BlockHeader {
                parent_hash: [0; 32],
                height,
                timestamp: 1693526400, // 2025-08-31
                merkle_root: [0; 32],
                nonce: 12345,
                difficulty: 1000,
                miner_address: [1; 20],
                miner_pubkey: vec![2; 32],
            },
            quantum_data: quantum_quality.map(|quality| QuantumData {
                quantum_seed: Some([3; 32]),
                vdf_proof: Some(vec![4; 64]),
                entropy_quality: Some(quality),
            }),
            mining_data: Some(MiningData {
                hash_rate: 10000.0,
                mining_duration_ms: 30000,
                quantum_overhead_ms: 1.0,
                efficiency_percent: 95.0,
            }),
            transactions: vec![],
            signature: vec![5; 64],
        }
    }

    #[tokio::test]
    async fn test_reward_calculation() {
        let calculator = RewardCalculator::new(RewardConfig::default());
        let block = create_test_block(1000, Some(0.95));

        let result = calculator.calculate_reward(&block).await.unwrap();
        
        assert!(result.base_reward > 0);
        assert!(result.quantum_bonus > 0); // Quality 0.95 > 0.9 threshold
        assert_eq!(result.total_reward, result.base_reward + result.quantum_bonus);
        assert_eq!(result.final_reward, result.total_reward - result.burn_amount);
    }

    #[tokio::test]
    async fn test_reward_validation() {
        let calculator = RewardCalculator::new(RewardConfig::default());
        let block = create_test_block(500, Some(0.92));
        
        let result = calculator.calculate_reward(&block).await.unwrap();
        
        // Valid claim should pass
        assert!(calculator.validate_reward_claim(&block, result.final_reward).await.is_ok());
        
        // Excessive claim should fail
        assert!(calculator.validate_reward_claim(&block, result.final_reward + 1).await.is_err());
    }

    #[tokio::test]
    async fn test_halving_mechanism() {
        let config = RewardConfig {
            base_reward: 1000,
            halving_interval: 100,
            ..Default::default()
        };
        let calculator = RewardCalculator::new(config);

        // Before halving
        let block1 = create_test_block(50, None);
        let result1 = calculator.calculate_reward(&block1).await.unwrap();
        
        // After halving
        let block2 = create_test_block(150, None);
        let result2 = calculator.calculate_reward(&block2).await.unwrap();
        
        assert_eq!(result2.base_reward, result1.base_reward / 2);
    }

    #[tokio::test]
    async fn test_quantum_quality_assessment() {
        let calculator = RewardCalculator::new(RewardConfig::default());
        
        // High quality quantum mining
        let high_quality_block = create_test_block(100, Some(0.95));
        let high_result = calculator.calculate_reward(&high_quality_block).await.unwrap();
        
        // Low quality quantum mining
        let low_quality_block = create_test_block(101, Some(0.85));
        let low_result = calculator.calculate_reward(&low_quality_block).await.unwrap();
        
        // Classical mining (no quantum)
        let classical_block = create_test_block(102, None);
        let classical_result = calculator.calculate_reward(&classical_block).await.unwrap();
        
        assert!(high_result.quantum_bonus > 0);
        assert_eq!(low_result.quantum_bonus, 0); // Below threshold
        assert_eq!(classical_result.quantum_bonus, 0);
    }

    #[tokio::test]
    async fn test_total_supply_calculation() {
        let config = RewardConfig {
            base_reward: 1000,
            halving_interval: 100,
            burn_rate_percent: 25,
            max_supply: 10000,
            ..Default::default()
        };
        let calculator = RewardCalculator::new(config);
        
        let supply_100 = calculator.calculate_total_supply(100);
        let supply_200 = calculator.calculate_total_supply(200);
        
        // Supply should increase with height
        assert!(supply_200 > supply_100);
        
        // Should respect max supply
        let supply_max = calculator.calculate_total_supply(10000);
        assert!(supply_max <= config.max_supply);
    }
}