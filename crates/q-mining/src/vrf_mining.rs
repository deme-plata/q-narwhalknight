//! Post-Quantum VRF Mining Integration
//!
//! This module integrates the Ring-LWE based VRF with the mining system for:
//! - Fair block lottery (verifiable random leader election)
//! - Mining reward randomization (bonus for lucky VRF outputs)
//! - Post-quantum secure leader selection (resistant to Shor's algorithm)
//!
//! ## Why Ring-LWE VRF?
//!
//! The original X-VRF (IACR 2021/302) was BROKEN in 2024. An attack allows
//! doubling selection probability by exploiting WOTS+ collision vulnerabilities.
//! This implementation uses Ring-LWE which provides:
//! - Proper uniqueness guarantees (no double-selection attack)
//! - Post-quantum security based on lattice hardness
//! - Efficient verification for high TPS
//!
//! ## Integration with Mining
//!
//! ```ignore
//! use q_mining::vrf_mining::{VrfMiningEngine, VrfMiningConfig};
//!
//! let config = VrfMiningConfig::default();
//! let engine = VrfMiningEngine::new(config)?;
//!
//! // Check if this miner is eligible for block at height 1000
//! let (output, proof) = engine.evaluate_block_lottery(1000, &prev_hash)?;
//! if engine.is_lottery_winner(&output, &difficulty_threshold) {
//!     // This miner can propose a block!
//! }
//! ```

use anyhow::Result;
use serde::{Deserialize, Serialize};
use sha3::Sha3_256;

// Digest trait needed for Sha3_256::new(), update(), finalize()
#[cfg(not(feature = "mining-vrf"))]
use sha3::Digest;
use tracing::{debug, info, warn};

#[cfg(feature = "mining-vrf")]
use q_rlwe_vrf::{VrfSecretKey, VrfPublicKey, VrfOutput, VrfProof, MiningVrf, SecurityLevel};

/// Configuration for VRF-based mining
#[derive(Debug, Clone)]
pub struct VrfMiningConfig {
    /// Security level for Ring-LWE parameters
    pub security_level: VrfSecurityLevel,
    /// Enable VRF for block lottery
    pub lottery_enabled: bool,
    /// Enable VRF for reward randomization
    pub reward_randomization_enabled: bool,
    /// Minimum VRF iterations for timing resistance
    pub min_vrf_iterations: u32,
}

/// Security levels for VRF mining
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VrfSecurityLevel {
    /// 128-bit post-quantum security
    Standard,
    /// 192-bit post-quantum security
    High,
    /// 256-bit post-quantum security
    Maximum,
}

impl Default for VrfMiningConfig {
    fn default() -> Self {
        Self {
            security_level: VrfSecurityLevel::Standard,
            lottery_enabled: true,
            reward_randomization_enabled: true,
            min_vrf_iterations: 1000,
        }
    }
}

/// VRF output for mining decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiningVrfOutput {
    /// Raw VRF output bytes
    pub output: [u8; 32],
    /// VRF proof for verification
    pub proof: Vec<u8>,
    /// Block height this was generated for
    pub block_height: u64,
    /// Previous block hash used as input
    pub prev_hash: [u8; 32],
    /// Whether this output won the lottery
    pub is_winner: bool,
    /// Reward multiplier (1.0 - 2.0 based on output)
    pub reward_multiplier: f64,
}

/// VRF mining engine for post-quantum leader election
#[cfg(feature = "mining-vrf")]
pub struct VrfMiningEngine {
    config: VrfMiningConfig,
    mining_vrf: MiningVrf,
    secret_key: VrfSecretKey,
    public_key: VrfPublicKey,
}

#[cfg(feature = "mining-vrf")]
impl VrfMiningEngine {
    /// Create new VRF mining engine
    pub fn new(config: VrfMiningConfig) -> Result<Self> {
        let security_level = match config.security_level {
            VrfSecurityLevel::Standard => SecurityLevel::Standard,
            VrfSecurityLevel::High => SecurityLevel::High,
            VrfSecurityLevel::Maximum => SecurityLevel::Maximum,
        };

        let mining_vrf = MiningVrf::new(security_level)?;
        let (secret_key, public_key) = mining_vrf.keypair();

        info!(
            "✨ VRF mining engine initialized with {:?} security",
            config.security_level
        );

        Ok(Self {
            config,
            mining_vrf,
            secret_key,
            public_key,
        })
    }

    /// Create engine with existing keypair
    pub fn with_keypair(
        config: VrfMiningConfig,
        secret_key: VrfSecretKey,
        public_key: VrfPublicKey,
    ) -> Result<Self> {
        let security_level = match config.security_level {
            VrfSecurityLevel::Standard => SecurityLevel::Standard,
            VrfSecurityLevel::High => SecurityLevel::High,
            VrfSecurityLevel::Maximum => SecurityLevel::Maximum,
        };

        let mining_vrf = MiningVrf::with_keypair(security_level, secret_key.clone(), public_key.clone())?;

        Ok(Self {
            config,
            mining_vrf,
            secret_key,
            public_key,
        })
    }

    /// Evaluate block lottery for a given height
    ///
    /// Returns VRF output and proof that can be verified by others.
    /// The output determines if this miner is eligible to propose a block.
    pub fn evaluate_block_lottery(
        &self,
        block_height: u64,
        prev_block_hash: &[u8; 32],
    ) -> Result<MiningVrfOutput> {
        if !self.config.lottery_enabled {
            // Return dummy output when lottery disabled
            return Ok(MiningVrfOutput {
                output: [0u8; 32],
                proof: vec![],
                block_height,
                prev_hash: *prev_block_hash,
                is_winner: true, // Everyone wins when lottery disabled
                reward_multiplier: 1.0,
            });
        }

        let (output, proof) = self.mining_vrf.evaluate_for_block(block_height, prev_block_hash)?;

        let output_bytes = output.as_bytes();
        let proof_bytes = proof.to_bytes();

        // Calculate reward multiplier (1.0 - 2.0 based on output entropy)
        let reward_multiplier = self.calculate_reward_multiplier(&output_bytes);

        debug!(
            "VRF evaluation for block {}: output={}, multiplier={:.3}",
            block_height,
            hex::encode(&output_bytes[..8]),
            reward_multiplier
        );

        Ok(MiningVrfOutput {
            output: output_bytes,
            proof: proof_bytes,
            block_height,
            prev_hash: *prev_block_hash,
            is_winner: false, // Caller should check with is_lottery_winner
            reward_multiplier,
        })
    }

    /// Check if VRF output wins the lottery
    ///
    /// The difficulty threshold determines how many miners can win.
    /// Lower threshold = fewer winners = harder to win.
    pub fn is_lottery_winner(&self, output: &MiningVrfOutput, difficulty_threshold: &[u8; 32]) -> bool {
        if !self.config.lottery_enabled {
            return true;
        }

        self.mining_vrf.is_winner_with_threshold(&output.output, difficulty_threshold)
    }

    /// Verify someone else's VRF lottery result
    pub fn verify_lottery_result(
        &self,
        miner_public_key: &VrfPublicKey,
        block_height: u64,
        prev_block_hash: &[u8; 32],
        output: &MiningVrfOutput,
    ) -> Result<bool> {
        if !self.config.lottery_enabled {
            return Ok(true);
        }

        // Reconstruct VRF output and proof from bytes
        let vrf_output = VrfOutput::from_bytes(&output.output)?;
        let vrf_proof = VrfProof::from_bytes(&output.proof)?;

        Ok(self.mining_vrf.verify_block_lottery(
            miner_public_key,
            block_height,
            prev_block_hash,
            &vrf_output,
            &vrf_proof,
        )?)
    }

    /// Calculate reward multiplier from VRF output
    ///
    /// Returns value between 1.0 and 2.0:
    /// - 1.0 = base reward
    /// - 2.0 = maximum luck bonus (very rare)
    fn calculate_reward_multiplier(&self, output: &[u8; 32]) -> f64 {
        if !self.config.reward_randomization_enabled {
            return 1.0;
        }

        // Use first 8 bytes to determine luck
        let luck_value = u64::from_be_bytes([
            output[0], output[1], output[2], output[3],
            output[4], output[5], output[6], output[7],
        ]);

        // Map to 1.0 - 2.0 range
        // Most values will cluster near 1.0 due to exponential distribution
        let normalized = (luck_value as f64) / (u64::MAX as f64);

        // Exponential distribution: most miners get ~1.0, few get up to 2.0
        let multiplier = 1.0 + normalized.powf(4.0);

        multiplier.min(2.0)
    }

    /// Get miner's public key for sharing with network
    pub fn public_key(&self) -> &VrfPublicKey {
        &self.public_key
    }

    /// Serialize secret key for storage
    pub fn export_secret_key(&self) -> Vec<u8> {
        self.secret_key.to_bytes()
    }

    /// Get current configuration
    pub fn config(&self) -> &VrfMiningConfig {
        &self.config
    }
}

/// Calculate lottery difficulty threshold based on network parameters
///
/// # Arguments
/// * `total_hashrate` - Network total hashrate
/// * `target_winners` - Target number of winning miners per block
/// * `block_time` - Target time between blocks in seconds
///
/// # Returns
/// Difficulty threshold as 32 bytes (lower = harder)
pub fn calculate_lottery_threshold(
    total_hashrate: f64,
    target_winners: u32,
    block_time: u32,
) -> [u8; 32] {
    // Calculate target probability
    let target_probability = (target_winners as f64) / (total_hashrate * block_time as f64);

    // Convert to threshold bytes
    let threshold_value = (target_probability * (u64::MAX as f64)) as u64;

    let mut threshold = [0u8; 32];
    threshold[..8].copy_from_slice(&threshold_value.to_be_bytes());

    threshold
}

// Fallback implementation when mining-vrf feature is disabled
#[cfg(not(feature = "mining-vrf"))]
pub struct VrfMiningEngine;

#[cfg(not(feature = "mining-vrf"))]
impl VrfMiningEngine {
    pub fn new(_config: VrfMiningConfig) -> Result<Self> {
        warn!("VRF mining requires the 'mining-vrf' feature. Using fallback mode.");
        Ok(Self)
    }

    pub fn evaluate_block_lottery(
        &self,
        block_height: u64,
        prev_block_hash: &[u8; 32],
    ) -> Result<MiningVrfOutput> {
        // Fallback: use deterministic hash-based lottery
        let mut hasher = Sha3_256::new();
        hasher.update(b"fallback-lottery");
        hasher.update(&block_height.to_be_bytes());
        hasher.update(prev_block_hash);
        let hash = hasher.finalize();

        let mut output = [0u8; 32];
        output.copy_from_slice(&hash);

        Ok(MiningVrfOutput {
            output,
            proof: vec![],
            block_height,
            prev_hash: *prev_block_hash,
            is_winner: true,
            reward_multiplier: 1.0,
        })
    }

    pub fn is_lottery_winner(&self, _output: &MiningVrfOutput, _difficulty_threshold: &[u8; 32]) -> bool {
        true // Everyone wins in fallback mode
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = VrfMiningConfig::default();
        assert!(config.lottery_enabled);
        assert!(config.reward_randomization_enabled);
        assert_eq!(config.security_level, VrfSecurityLevel::Standard);
    }

    #[test]
    fn test_lottery_threshold_calculation() {
        // 1 TH/s network, 1 winner, 30 second blocks
        let threshold = calculate_lottery_threshold(1e12, 1, 30);

        // Should produce a valid threshold
        assert!(!threshold.iter().all(|&b| b == 0));
    }

    #[cfg(feature = "mining-vrf")]
    #[test]
    fn test_vrf_mining_creation() {
        let config = VrfMiningConfig::default();
        let engine = VrfMiningEngine::new(config);
        assert!(engine.is_ok());
    }

    #[cfg(feature = "mining-vrf")]
    #[test]
    fn test_block_lottery_evaluation() {
        let config = VrfMiningConfig::default();
        let engine = VrfMiningEngine::new(config).unwrap();

        let prev_hash = [42u8; 32];
        let result = engine.evaluate_block_lottery(1000, &prev_hash);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.block_height, 1000);
        assert_eq!(output.prev_hash, prev_hash);
        assert!(output.reward_multiplier >= 1.0);
        assert!(output.reward_multiplier <= 2.0);
    }

    #[cfg(feature = "mining-vrf")]
    #[test]
    fn test_deterministic_output() {
        let config = VrfMiningConfig::default();
        let engine = VrfMiningEngine::new(config).unwrap();

        let prev_hash = [1u8; 32];

        // Same inputs should produce same outputs
        let output1 = engine.evaluate_block_lottery(100, &prev_hash).unwrap();
        let output2 = engine.evaluate_block_lottery(100, &prev_hash).unwrap();

        assert_eq!(output1.output, output2.output);
        assert_eq!(output1.reward_multiplier, output2.reward_multiplier);
    }

    #[cfg(feature = "mining-vrf")]
    #[test]
    fn test_different_heights_different_outputs() {
        let config = VrfMiningConfig::default();
        let engine = VrfMiningEngine::new(config).unwrap();

        let prev_hash = [1u8; 32];

        let output1 = engine.evaluate_block_lottery(100, &prev_hash).unwrap();
        let output2 = engine.evaluate_block_lottery(101, &prev_hash).unwrap();

        // Different heights should produce different outputs
        assert_ne!(output1.output, output2.output);
    }
}
