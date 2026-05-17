//! Bulletproofs v2 Range Proofs for Mining Rewards
//!
//! Implements efficient zero-knowledge range proofs based on IACR 2024/313
//! for confidential mining rewards.
//!
//! ## Why Bulletproofs v2?
//!
//! Mining rewards need to be:
//! - **Verifiable**: Prove rewards are within valid range (0 to max supply)
//! - **Private**: Don't reveal exact reward amounts (optional privacy)
//! - **Efficient**: 2x faster verification than Bulletproofs v1
//! - **Aggregatable**: Batch-verify multiple rewards efficiently
//!
//! ## Integration with Mining
//!
//! ```ignore
//! use q_mining::reward_proofs::{RewardProof, RewardProver};
//!
//! // Create prover
//! let prover = RewardProver::new(64)?; // 64-bit range (0 to 2^64-1)
//!
//! // Prove reward is valid
//! let reward = 50_000_000; // 50 QNK block reward
//! let proof = prover.prove_reward(reward)?;
//!
//! // Verify (done by other nodes)
//! assert!(proof.verify()?);
//!
//! // Aggregate multiple proofs for batch verification
//! let aggregated = RewardAggregator::aggregate(&[proof1, proof2, proof3])?;
//! assert!(aggregated.verify_all()?);
//! ```

#[cfg(feature = "advanced-crypto")]
use q_crypto_advanced::bulletproofs_v2::{
    Scalar, Point, RangeProof, BulletproofsProver, BulletproofsVerifier,
    AggregatedRangeProof, AggregatedProver, InnerProductProof, DEFAULT_RANGE_BITS,
};

use anyhow::{anyhow, Result};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::collections::HashMap;
use tracing::{debug, info, warn};

/// Configuration for reward proof generation
#[derive(Debug, Clone)]
pub struct RewardProofConfig {
    /// Number of bits for range proof (default: 64)
    pub range_bits: usize,
    /// Maximum reward value (for validation)
    pub max_reward: u64,
    /// Minimum reward value
    pub min_reward: u64,
    /// Enable aggregated proofs
    pub enable_aggregation: bool,
}

impl Default for RewardProofConfig {
    fn default() -> Self {
        Self {
            range_bits: 64,
            max_reward: u64::MAX,
            min_reward: 0,
            enable_aggregation: true,
        }
    }
}

impl RewardProofConfig {
    /// Standard configuration for block rewards
    pub fn for_block_rewards() -> Self {
        Self {
            range_bits: 64,
            max_reward: 100_000_000_000, // 100B max supply
            min_reward: 0,
            enable_aggregation: true,
        }
    }

    /// Configuration for transaction fees
    pub fn for_fees() -> Self {
        Self {
            range_bits: 64,
            max_reward: 1_000_000_000, // 1B max fee
            min_reward: 0,
            enable_aggregation: true,
        }
    }
}

/// A confidential reward with Pedersen commitment and range proof
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidentialReward {
    /// Pedersen commitment to the reward amount: C = g^amount * h^blinding
    #[serde(with = "hex_serde")]
    pub commitment: Vec<u8>,
    /// Range proof showing commitment is to a value in [0, 2^n)
    #[serde(with = "hex_serde")]
    pub range_proof: Vec<u8>,
    /// Block height this reward is for
    pub block_height: u64,
    /// Miner address (can be disclosed)
    pub miner_address: [u8; 32],
    /// Number of bits in range proof
    pub range_bits: usize,
}

impl ConfidentialReward {
    /// Get proof size in bytes
    pub fn proof_size(&self) -> usize {
        self.range_proof.len()
    }

    /// Serialize for storage/transmission
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        bincode::serialize(self).map_err(|e| anyhow!("Serialization failed: {}", e))
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        bincode::deserialize(bytes).map_err(|e| anyhow!("Deserialization failed: {}", e))
    }
}

mod hex_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(bytes: &Vec<u8>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        hex::encode(bytes).serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<u8>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        hex::decode(&s).map_err(serde::de::Error::custom)
    }
}

/// Prover for mining reward range proofs
#[cfg(feature = "advanced-crypto")]
pub struct RewardProver {
    config: RewardProofConfig,
    prover: BulletproofsProver,
}

#[cfg(feature = "advanced-crypto")]
impl RewardProver {
    /// Create a new reward prover
    pub fn new(config: RewardProofConfig) -> Result<Self> {
        let prover = BulletproofsProver::new(config.range_bits)?;

        info!(
            "Reward prover initialized with {}-bit range proofs",
            config.range_bits
        );

        Ok(Self { config, prover })
    }

    /// Create with default configuration
    pub fn default_prover() -> Result<Self> {
        Self::new(RewardProofConfig::default())
    }

    /// Prove that a reward amount is within the valid range
    pub fn prove_reward(
        &self,
        amount: u64,
        block_height: u64,
        miner_address: [u8; 32],
    ) -> Result<ConfidentialReward> {
        // Validate amount
        if amount > self.config.max_reward {
            return Err(anyhow!(
                "Reward {} exceeds maximum {}",
                amount,
                self.config.max_reward
            ));
        }
        if amount < self.config.min_reward {
            return Err(anyhow!(
                "Reward {} below minimum {}",
                amount,
                self.config.min_reward
            ));
        }

        // SECURITY FIX: Use ChaCha20Rng with proper entropy source
        // Previously used Scalar::random() without specifying entropy source,
        // which could use weak system RNG on some platforms.
        // ChaCha20Rng from_entropy() uses OS-level cryptographic randomness.
        let mut rng = ChaCha20Rng::from_entropy();
        let blinding = Scalar::random_with_rng(&mut rng);

        // Create Pedersen commitment and range proof
        let (commitment, range_proof) = self.prover.prove(amount, &blinding)?;

        debug!(
            "Generated range proof for reward {} at height {}",
            amount, block_height
        );

        Ok(ConfidentialReward {
            commitment: commitment.to_bytes(),
            range_proof: range_proof.to_bytes(),
            block_height,
            miner_address,
            range_bits: self.config.range_bits,
        })
    }

    /// Prove reward and keep blinding factor (for later disclosure)
    pub fn prove_with_blinding(
        &self,
        amount: u64,
        block_height: u64,
        miner_address: [u8; 32],
    ) -> Result<(ConfidentialReward, Scalar)> {
        if amount > self.config.max_reward {
            return Err(anyhow!(
                "Reward {} exceeds maximum {}",
                amount,
                self.config.max_reward
            ));
        }

        let blinding = Scalar::random();
        let (commitment, range_proof) = self.prover.prove(amount, &blinding)?;

        let reward = ConfidentialReward {
            commitment: commitment.to_bytes(),
            range_proof: range_proof.to_bytes(),
            block_height,
            miner_address,
            range_bits: self.config.range_bits,
        };

        Ok((reward, blinding))
    }

    /// Get configuration
    pub fn config(&self) -> &RewardProofConfig {
        &self.config
    }
}

/// Verifier for mining reward range proofs
#[cfg(feature = "advanced-crypto")]
pub struct RewardVerifier {
    config: RewardProofConfig,
    verifier: BulletproofsVerifier,
}

#[cfg(feature = "advanced-crypto")]
impl RewardVerifier {
    /// Create a new reward verifier
    pub fn new(config: RewardProofConfig) -> Result<Self> {
        let verifier = BulletproofsVerifier::new(config.range_bits)?;

        Ok(Self { config, verifier })
    }

    /// Create with default configuration
    pub fn default_verifier() -> Result<Self> {
        Self::new(RewardProofConfig::default())
    }

    /// Verify a confidential reward
    pub fn verify(&self, reward: &ConfidentialReward) -> Result<bool> {
        if reward.range_bits != self.config.range_bits {
            return Err(anyhow!(
                "Range bits mismatch: expected {}, got {}",
                self.config.range_bits,
                reward.range_bits
            ));
        }

        let commitment = Point::from_bytes(&reward.commitment)?;
        let range_proof = RangeProof::from_bytes(&reward.range_proof)?;

        self.verifier.verify(&commitment, &range_proof)
    }

    /// Verify and return detailed result
    pub fn verify_detailed(&self, reward: &ConfidentialReward) -> VerificationResult {
        let start = std::time::Instant::now();

        let result = self.verify(reward);
        let verification_time = start.elapsed();

        VerificationResult {
            valid: result.unwrap_or(false),
            error: result.err().map(|e| e.to_string()),
            verification_time_us: verification_time.as_micros() as u64,
            block_height: reward.block_height,
            proof_size_bytes: reward.proof_size(),
        }
    }
}

/// Detailed verification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    pub valid: bool,
    pub error: Option<String>,
    pub verification_time_us: u64,
    pub block_height: u64,
    pub proof_size_bytes: usize,
}

/// Aggregator for multiple reward proofs (2x faster batch verification)
#[cfg(feature = "advanced-crypto")]
pub struct RewardAggregator {
    config: RewardProofConfig,
    aggregator: AggregatedProver,
    pending_rewards: Vec<(u64, Scalar)>, // (amount, blinding)
}

#[cfg(feature = "advanced-crypto")]
impl RewardAggregator {
    /// Create a new reward aggregator
    pub fn new(config: RewardProofConfig) -> Result<Self> {
        let aggregator = AggregatedProver::new(config.range_bits)?;

        Ok(Self {
            config,
            aggregator,
            pending_rewards: Vec::new(),
        })
    }

    /// Add a reward to aggregate
    pub fn add_reward(&mut self, amount: u64, blinding: Scalar) -> Result<()> {
        if amount > self.config.max_reward {
            return Err(anyhow!("Reward exceeds maximum"));
        }

        self.pending_rewards.push((amount, blinding));
        Ok(())
    }

    /// Generate aggregated proof for all pending rewards
    pub fn finalize(self) -> Result<AggregatedRewardProof> {
        let mut commitments = Vec::with_capacity(self.pending_rewards.len());
        let mut proofs = Vec::new();

        for (amount, blinding) in &self.pending_rewards {
            let (commitment, proof) = self.aggregator.prove(*amount, blinding)?;
            commitments.push(commitment.to_bytes());
            proofs.push(proof.to_bytes());
        }

        // Aggregate the individual proofs
        let aggregated_proof = self.aggregator.aggregate(&proofs)?;

        info!(
            "Aggregated {} reward proofs (2x faster verification)",
            commitments.len()
        );

        Ok(AggregatedRewardProof {
            commitments,
            aggregated_proof: aggregated_proof.to_bytes(),
            reward_count: self.pending_rewards.len() as u32,
            range_bits: self.config.range_bits,
        })
    }

    /// Get pending reward count
    pub fn pending_count(&self) -> usize {
        self.pending_rewards.len()
    }
}

/// Aggregated proof for multiple rewards
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedRewardProof {
    /// Individual Pedersen commitments
    pub commitments: Vec<Vec<u8>>,
    /// Single aggregated range proof
    #[serde(with = "hex_serde")]
    pub aggregated_proof: Vec<u8>,
    /// Number of rewards in this proof
    pub reward_count: u32,
    /// Range bits used
    pub range_bits: usize,
}

impl AggregatedRewardProof {
    /// Calculate space savings vs individual proofs
    pub fn space_savings(&self) -> f64 {
        // Individual proof would be ~688 bytes each (Bulletproofs v2)
        let individual_size = self.reward_count as usize * 688;
        let aggregated_size = self.aggregated_proof.len();

        if individual_size > 0 {
            1.0 - (aggregated_size as f64 / individual_size as f64)
        } else {
            0.0
        }
    }

    /// Serialize for storage
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        bincode::serialize(self).map_err(|e| anyhow!("Serialization failed: {}", e))
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        bincode::deserialize(bytes).map_err(|e| anyhow!("Deserialization failed: {}", e))
    }
}

/// Batch verifier for aggregated reward proofs
#[cfg(feature = "advanced-crypto")]
pub struct AggregatedRewardVerifier {
    verifier: BulletproofsVerifier,
}

#[cfg(feature = "advanced-crypto")]
impl AggregatedRewardVerifier {
    /// Create new aggregated verifier
    pub fn new(range_bits: usize) -> Result<Self> {
        let verifier = BulletproofsVerifier::new(range_bits)?;
        Ok(Self { verifier })
    }

    /// Verify an aggregated reward proof (2x faster than individual)
    pub fn verify(&self, proof: &AggregatedRewardProof) -> Result<bool> {
        let mut commitments = Vec::with_capacity(proof.commitments.len());
        for c in &proof.commitments {
            commitments.push(Point::from_bytes(c)?);
        }

        let aggregated = AggregatedRangeProof::from_bytes(&proof.aggregated_proof)?;
        self.verifier.verify_aggregated(&commitments, &aggregated)
    }
}

/// Statistics for reward proof operations
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RewardProofStats {
    pub total_proofs_generated: u64,
    pub total_proofs_verified: u64,
    pub total_aggregations: u64,
    pub average_proof_size_bytes: usize,
    pub average_verification_time_us: u64,
    pub average_generation_time_us: u64,
}

// Fallback for when advanced-crypto is disabled
#[cfg(not(feature = "advanced-crypto"))]
pub struct RewardProver;

#[cfg(not(feature = "advanced-crypto"))]
impl RewardProver {
    pub fn new(_config: RewardProofConfig) -> Result<Self> {
        Err(anyhow!(
            "Bulletproofs v2 requires the 'advanced-crypto' feature. Enable it in Cargo.toml."
        ))
    }
}

#[cfg(all(test, feature = "advanced-crypto"))]
mod tests {
    use super::*;

    #[test]
    fn test_reward_prover_creation() {
        let config = RewardProofConfig::for_block_rewards();
        let prover = RewardProver::new(config);
        assert!(prover.is_ok());
    }

    #[test]
    fn test_single_reward_proof() {
        let config = RewardProofConfig::for_block_rewards();
        let prover = RewardProver::new(config.clone()).unwrap();
        let verifier = RewardVerifier::new(config).unwrap();

        let reward = 50_000_000u64; // 50 QNK
        let miner = [42u8; 32];

        let proof = prover.prove_reward(reward, 1000, miner).unwrap();

        assert!(proof.commitment.len() > 0);
        assert!(proof.range_proof.len() > 0);

        let is_valid = verifier.verify(&proof).unwrap();
        assert!(is_valid);

        println!("Proof size: {} bytes", proof.proof_size());
    }

    #[test]
    fn test_reward_exceeds_max() {
        let config = RewardProofConfig {
            max_reward: 1000,
            ..Default::default()
        };
        let prover = RewardProver::new(config).unwrap();

        let result = prover.prove_reward(2000, 1, [0u8; 32]);
        assert!(result.is_err());
    }

    #[test]
    fn test_verification_timing() {
        let config = RewardProofConfig::for_block_rewards();
        let prover = RewardProver::new(config.clone()).unwrap();
        let verifier = RewardVerifier::new(config).unwrap();

        let proof = prover.prove_reward(100_000, 1, [1u8; 32]).unwrap();
        let result = verifier.verify_detailed(&proof);

        assert!(result.valid);
        println!(
            "Verification time: {} us",
            result.verification_time_us
        );
    }

    #[test]
    fn test_aggregated_proofs() {
        let config = RewardProofConfig::for_block_rewards();
        let mut aggregator = RewardAggregator::new(config.clone()).unwrap();

        // Add multiple rewards
        let blinding1 = Scalar::random();
        let blinding2 = Scalar::random();
        let blinding3 = Scalar::random();

        aggregator.add_reward(50_000_000, blinding1).unwrap();
        aggregator.add_reward(25_000_000, blinding2).unwrap();
        aggregator.add_reward(75_000_000, blinding3).unwrap();

        let aggregated = aggregator.finalize().unwrap();

        assert_eq!(aggregated.reward_count, 3);
        println!(
            "Aggregated {} proofs with {:.1}% space savings",
            aggregated.reward_count,
            aggregated.space_savings() * 100.0
        );

        // Verify
        let verifier = AggregatedRewardVerifier::new(config.range_bits).unwrap();
        let is_valid = verifier.verify(&aggregated).unwrap();
        assert!(is_valid);
    }
}
