/// Hybrid CPU+GPU Mining System
///
/// **User Explanation:**
/// CPU miners solve VDF proofs (memory-based), GPU miners solve SHA-3 hashes (compute-based).
/// Both are required for a valid block and both earn 50% of the block reward - keeping CPU mining profitable!
///
/// This ensures:
/// - CPUs remain profitable (they're naturally good at VDF)
/// - GPUs are efficient (they're naturally good at SHA-3)
/// - Network stays decentralized (needs both types of miners)
/// - No need to track hardware prices or adjust manually

use anyhow::Result;
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::{QuantumPoWBlock, MiningTemplate};
use q_dag_knight::{VDFProof, QuantumVDF};
use q_types::Address;

/// Hybrid mining block that requires both CPU and GPU work
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridMiningBlock {
    /// Block height
    pub height: u64,

    /// Timestamp
    pub timestamp: u64,

    /// Previous block hash
    pub previous_hash: [u8; 32],

    /// CPU Component: VDF Proof (memory-bound, CPU-optimized)
    pub vdf_proof: VDFProof,
    pub cpu_miner_address: Address,
    pub vdf_difficulty: u64,

    /// GPU Component: SHA-3 PoW (compute-bound, GPU-optimized)
    pub pow_hash: [u8; 32],
    pub pow_nonce: u64,
    pub gpu_miner_address: Address,
    pub pow_difficulty: u32,

    /// Merkle root of transactions
    pub merkle_root: [u8; 32],

    /// Transaction data
    pub transactions: Vec<Vec<u8>>,
}

impl HybridMiningBlock {
    /// Create new hybrid mining block
    pub fn new(
        height: u64,
        previous_hash: [u8; 32],
        merkle_root: [u8; 32],
        vdf_proof: VDFProof,
        cpu_miner: Address,
        pow_hash: [u8; 32],
        pow_nonce: u64,
        gpu_miner: Address,
    ) -> Self {
        Self {
            height,
            timestamp: chrono::Utc::now().timestamp() as u64,
            previous_hash,
            vdf_proof,
            cpu_miner_address: cpu_miner,
            vdf_difficulty: 1000,
            pow_hash,
            pow_nonce,
            gpu_miner_address: gpu_miner,
            pow_difficulty: 4,
            merkle_root,
            transactions: Vec::new(),
        }
    }

    /// Validate both CPU and GPU components
    pub fn validate(&self) -> Result<bool> {
        // Validate VDF proof (CPU work)
        let vdf_valid = self.validate_vdf_proof()?;

        // Validate PoW hash (GPU work)
        let pow_valid = self.validate_pow_hash()?;

        Ok(vdf_valid && pow_valid)
    }

    /// Validate VDF proof component
    fn validate_vdf_proof(&self) -> Result<bool> {
        // VDF proof should be valid and meet difficulty requirement
        // TODO: Implement proper VDF verification
        Ok(self.vdf_proof.proof.len() > 0)
    }

    /// Validate PoW hash component
    fn validate_pow_hash(&self) -> Result<bool> {
        // Recompute hash and verify it meets difficulty
        let computed_hash = self.compute_pow_hash();

        // Check hash matches
        if computed_hash != self.pow_hash {
            return Ok(false);
        }

        // Check difficulty (leading zeros)
        let leading_zeros = computed_hash.iter().take_while(|&&b| b == 0).count();
        Ok(leading_zeros >= self.pow_difficulty as usize)
    }

    /// Compute PoW hash for validation
    fn compute_pow_hash(&self) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(&self.height.to_le_bytes());
        hasher.update(&self.previous_hash);
        hasher.update(&self.merkle_root);
        hasher.update(&self.pow_nonce.to_le_bytes());
        hasher.update(&self.vdf_proof.proof);

        hasher.finalize().into()
    }

    /// Calculate rewards for both CPU and GPU miners (50/50 split)
    pub fn calculate_rewards(&self, total_block_reward: u64) -> HybridRewards {
        let cpu_reward = total_block_reward / 2;
        let gpu_reward = total_block_reward / 2;

        HybridRewards {
            cpu_miner: self.cpu_miner_address.clone(),
            cpu_reward,
            gpu_miner: self.gpu_miner_address.clone(),
            gpu_reward,
            total_reward: total_block_reward,
        }
    }

    /// Get block hash
    pub fn hash(&self) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(&self.height.to_le_bytes());
        hasher.update(&self.timestamp.to_le_bytes());
        hasher.update(&self.previous_hash);
        hasher.update(&self.vdf_proof.proof);
        hasher.update(&self.pow_hash);

        hasher.finalize().into()
    }
}

/// Hybrid mining rewards (50/50 split)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridRewards {
    pub cpu_miner: Address,
    pub cpu_reward: u64,
    pub gpu_miner: Address,
    pub gpu_reward: u64,
    pub total_reward: u64,
}

/// CPU Mining Pool - Manages VDF proof submissions
#[derive(Debug)]
pub struct CPUMiningPool {
    /// Pending VDF proofs waiting for GPU PoW
    pending_vdf_proofs: Arc<RwLock<Vec<PendingVDFProof>>>,

    /// Active CPU miners
    active_miners: Arc<RwLock<Vec<Address>>>,
}

#[derive(Debug, Clone)]
pub struct PendingVDFProof {
    pub proof: VDFProof,
    pub miner_address: Address,
    pub submitted_at: u64,
    pub height: u64,
}

impl CPUMiningPool {
    pub fn new() -> Self {
        Self {
            pending_vdf_proofs: Arc::new(RwLock::new(Vec::new())),
            active_miners: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Submit VDF proof from CPU miner
    pub async fn submit_vdf_proof(
        &self,
        proof: VDFProof,
        miner_address: Address,
        height: u64,
    ) -> Result<()> {
        let pending = PendingVDFProof {
            proof,
            miner_address,
            submitted_at: chrono::Utc::now().timestamp() as u64,
            height,
        };

        self.pending_vdf_proofs.write().await.push(pending);

        tracing::info!(
            "📊 CPU miner {} submitted VDF proof for height {}",
            hex::encode(&miner_address),
            height
        );

        Ok(())
    }

    /// Get pending VDF proof for a given height
    pub async fn get_pending_vdf(&self, height: u64) -> Option<PendingVDFProof> {
        let proofs = self.pending_vdf_proofs.read().await;
        proofs.iter().find(|p| p.height == height).cloned()
    }

    /// Remove used VDF proof
    pub async fn consume_vdf_proof(&self, height: u64) -> Option<PendingVDFProof> {
        let mut proofs = self.pending_vdf_proofs.write().await;
        if let Some(pos) = proofs.iter().position(|p| p.height == height) {
            Some(proofs.remove(pos))
        } else {
            None
        }
    }
}

/// GPU Mining Pool - Manages PoW hash submissions
#[derive(Debug)]
pub struct GPUMiningPool {
    /// CPU pool reference to fetch VDF proofs
    cpu_pool: Arc<CPUMiningPool>,

    /// Active GPU miners
    active_miners: Arc<RwLock<Vec<Address>>>,
}

impl GPUMiningPool {
    pub fn new(cpu_pool: Arc<CPUMiningPool>) -> Self {
        Self {
            cpu_pool,
            active_miners: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Submit PoW solution from GPU miner
    pub async fn submit_pow_solution(
        &self,
        pow_hash: [u8; 32],
        pow_nonce: u64,
        gpu_miner_address: Address,
        height: u64,
        previous_hash: [u8; 32],
        merkle_root: [u8; 32],
    ) -> Result<Option<HybridMiningBlock>> {
        // Try to find matching VDF proof from CPU pool
        if let Some(vdf_pending) = self.cpu_pool.consume_vdf_proof(height).await {
            // Create complete hybrid block
            let block = HybridMiningBlock::new(
                height,
                previous_hash,
                merkle_root,
                vdf_pending.proof,
                vdf_pending.miner_address,
                pow_hash,
                pow_nonce,
                gpu_miner_address,
            );

            // Validate block
            if block.validate()? {
                tracing::info!(
                    "✅ Hybrid block created! Height: {}, CPU miner: {}, GPU miner: {}",
                    height,
                    hex::encode(&block.cpu_miner_address),
                    hex::encode(&block.gpu_miner_address)
                );

                Ok(Some(block))
            } else {
                tracing::warn!("❌ Hybrid block validation failed");
                Ok(None)
            }
        } else {
            tracing::debug!(
                "⏳ No matching VDF proof for height {}, GPU solution queued",
                height
            );
            Ok(None)
        }
    }
}

/// Hybrid Mining Coordinator
#[derive(Debug)]
pub struct HybridMiningCoordinator {
    cpu_pool: Arc<CPUMiningPool>,
    gpu_pool: Arc<GPUMiningPool>,
    block_reward: u64,
}

impl HybridMiningCoordinator {
    pub fn new(block_reward: u64) -> Self {
        let cpu_pool = Arc::new(CPUMiningPool::new());
        let gpu_pool = Arc::new(GPUMiningPool::new(cpu_pool.clone()));

        Self {
            cpu_pool,
            gpu_pool,
            block_reward,
        }
    }

    /// Submit CPU work (VDF proof)
    pub async fn submit_cpu_work(
        &self,
        proof: VDFProof,
        miner_address: Address,
        height: u64,
    ) -> Result<()> {
        self.cpu_pool.submit_vdf_proof(proof, miner_address, height).await
    }

    /// Submit GPU work (PoW solution)
    pub async fn submit_gpu_work(
        &self,
        pow_hash: [u8; 32],
        pow_nonce: u64,
        gpu_miner_address: Address,
        height: u64,
        previous_hash: [u8; 32],
        merkle_root: [u8; 32],
    ) -> Result<Option<HybridMiningBlock>> {
        self.gpu_pool.submit_pow_solution(
            pow_hash,
            pow_nonce,
            gpu_miner_address,
            height,
            previous_hash,
            merkle_root,
        ).await
    }

    /// Calculate and distribute rewards for a hybrid block
    pub async fn distribute_rewards(&self, block: &HybridMiningBlock) -> Result<HybridRewards> {
        let rewards = block.calculate_rewards(self.block_reward);

        tracing::info!(
            "💰 Hybrid block rewards distributed:\n  \
             CPU miner {}: {} QNK\n  \
             GPU miner {}: {} QNK",
            hex::encode(&rewards.cpu_miner),
            rewards.cpu_reward as f64 / 1_000_000_000.0,
            hex::encode(&rewards.gpu_miner),
            rewards.gpu_reward as f64 / 1_000_000_000.0,
        );

        Ok(rewards)
    }

    /// Get pending VDF proof count
    pub async fn pending_vdf_count(&self) -> usize {
        self.cpu_pool.pending_vdf_proofs.read().await.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_hybrid_mining() {
        let coordinator = HybridMiningCoordinator::new(2_000_000_000); // 2 QNK

        let cpu_miner = Address::from([1u8; 20]);
        let gpu_miner = Address::from([2u8; 20]);

        // Create dummy VDF proof
        let vdf_proof = VDFProof {
            proof: vec![1, 2, 3, 4],
            iterations: 1000,
            quantum_quality: 0.8,
        };

        // CPU miner submits VDF proof
        coordinator.submit_cpu_work(vdf_proof, cpu_miner.clone(), 1).await.unwrap();

        assert_eq!(coordinator.pending_vdf_count().await, 1);

        // GPU miner submits PoW solution
        let pow_hash = [0u8; 32];
        let result = coordinator.submit_gpu_work(
            pow_hash,
            12345,
            gpu_miner.clone(),
            1,
            [0u8; 32],
            [0u8; 32],
        ).await.unwrap();

        // Should create hybrid block
        assert!(result.is_some());

        let block = result.unwrap();
        let rewards = coordinator.distribute_rewards(&block).await.unwrap();

        // Verify 50/50 split
        assert_eq!(rewards.cpu_reward, 1_000_000_000); // 1 QNK
        assert_eq!(rewards.gpu_reward, 1_000_000_000); // 1 QNK
        assert_eq!(rewards.total_reward, 2_000_000_000); // 2 QNK
    }

    #[test]
    fn test_reward_calculation() {
        let block = HybridMiningBlock {
            height: 1,
            timestamp: 0,
            previous_hash: [0u8; 32],
            vdf_proof: VDFProof {
                proof: vec![],
                iterations: 0,
                quantum_quality: 0.0,
            },
            cpu_miner_address: Address::from([1u8; 20]),
            vdf_difficulty: 0,
            pow_hash: [0u8; 32],
            pow_nonce: 0,
            gpu_miner_address: Address::from([2u8; 20]),
            pow_difficulty: 0,
            merkle_root: [0u8; 32],
            transactions: vec![],
        };

        let rewards = block.calculate_rewards(2_000_000_000);

        // Verify 50/50 split
        assert_eq!(rewards.cpu_reward, 1_000_000_000);
        assert_eq!(rewards.gpu_reward, 1_000_000_000);
    }
}
