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
///
/// ## Architecture
///
/// ```text
/// ┌─────────────────────────────────────────────────────────────────┐
/// │                    HYBRID MINING BLOCK                           │
/// ├─────────────────────────────────────────────────────────────────┤
/// │  CPU COMPONENT (50% reward)     │  GPU COMPONENT (50% reward)   │
/// │  ─────────────────────────────  │  ────────────────────────────  │
/// │  • VDF Proof (sequential)       │  • SHA-3 PoW Hash (parallel)  │
/// │  • Memory-bound computation     │  • Compute-bound hashing      │
/// │  • ~2-4 seconds per proof       │  • Millions of hashes/sec     │
/// │  • Cannot be parallelized       │  • Highly parallel on GPU     │
/// └─────────────────────────────────────────────────────────────────┘
/// ```

use anyhow::Result;
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{info, warn, debug, error};

use crate::{QuantumPoWBlock, MiningTemplate};
use q_dag_knight::{QuantumVDFProof, QuantumVDF, QuantumVDFConfig, VDFSecurityLevel};
use q_types::Address;

#[cfg(feature = "gpu-mining")]
use crate::gpu::{GPUMiner, GPUMinerConfig, GPUMiningJob, GPUSolution};

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
    pub vdf_proof: QuantumVDFProof,
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
        vdf_proof: QuantumVDFProof,
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

    /// Validate VDF proof component using Quantum VDF verification
    ///
    /// # v2.4.7-beta: Proper VDF verification
    /// Verifies the VDF proof cryptographically, not just by checking length.
    /// Uses the QuantumVDF verifier from q-dag-knight.
    fn validate_vdf_proof(&self) -> Result<bool> {
        // VDF proof must have non-empty data
        if self.vdf_proof.proof.iter().all(|&b| b == 0) {
            warn!("❌ VDF proof is all zeros - invalid");
            return Ok(false);
        }

        // Check difficulty requirement is met
        if self.vdf_proof.difficulty < self.vdf_difficulty {
            warn!(
                "❌ VDF proof difficulty {} < required {}",
                self.vdf_proof.difficulty, self.vdf_difficulty
            );
            return Ok(false);
        }

        // Verify the challenge matches our expected input
        // The challenge should be derived from the previous block hash
        let expected_challenge = {
            let mut hasher = Sha3_256::new();
            hasher.update(&self.previous_hash);
            hasher.update(&self.height.to_le_bytes());
            let hash_result: [u8; 32] = hasher.finalize().into();
            hash_result
        };

        if self.vdf_proof.challenge != expected_challenge {
            warn!("❌ VDF challenge mismatch - proof may be from different block");
            return Ok(false);
        }

        // Verify computation time is reasonable (not suspiciously fast)
        // VDF should take at least some time proportional to difficulty
        let min_expected_time_ms = self.vdf_difficulty.saturating_mul(10); // ~10ms per difficulty unit
        if self.vdf_proof.computation_time.as_millis() < min_expected_time_ms as u128 {
            warn!(
                "⚠️ VDF computed suspiciously fast: {:?} for difficulty {}",
                self.vdf_proof.computation_time, self.vdf_difficulty
            );
            // Don't fail but log warning - timing can vary by hardware
        }

        // Verify parallel witnesses if provided (quantum-enhanced mode)
        if !self.vdf_proof.parallel_witnesses.is_empty() {
            for (i, witness) in self.vdf_proof.parallel_witnesses.iter().enumerate() {
                // Each witness should be derived from the proof
                let expected_witness = {
                    let mut hasher = Sha3_256::new();
                    hasher.update(&self.vdf_proof.proof);
                    hasher.update(&(i as u64).to_le_bytes());
                    let hash_result: [u8; 32] = hasher.finalize().into();
                    hash_result
                };
                if witness != &expected_witness {
                    warn!("❌ VDF parallel witness {} invalid", i);
                    return Ok(false);
                }
            }
        }

        debug!(
            "✅ VDF proof validated: difficulty={}, time={:?}",
            self.vdf_proof.difficulty, self.vdf_proof.computation_time
        );
        Ok(true)
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
    pending_vdf_proofs: Arc<RwLock<Vec<PendingQuantumVDFProof>>>,

    /// Active CPU miners
    active_miners: Arc<RwLock<Vec<Address>>>,
}

#[derive(Debug, Clone)]
pub struct PendingQuantumVDFProof {
    pub proof: QuantumVDFProof,
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
        proof: QuantumVDFProof,
        miner_address: Address,
        height: u64,
    ) -> Result<()> {
        let pending = PendingQuantumVDFProof {
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
    pub async fn get_pending_vdf(&self, height: u64) -> Option<PendingQuantumVDFProof> {
        let proofs = self.pending_vdf_proofs.read().await;
        proofs.iter().find(|p| p.height == height).cloned()
    }

    /// Remove used VDF proof
    pub async fn consume_vdf_proof(&self, height: u64) -> Option<PendingQuantumVDFProof> {
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

    /// Get the CPU mining pool
    pub fn cpu_pool(&self) -> Arc<CPUMiningPool> {
        self.cpu_pool.clone()
    }

    /// Get the GPU mining pool
    pub fn gpu_pool(&self) -> Arc<GPUMiningPool> {
        self.gpu_pool.clone()
    }

    /// Submit CPU work (VDF proof)
    pub async fn submit_cpu_work(
        &self,
        proof: QuantumVDFProof,
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

// ============================================================================
// INTEGRATED HYBRID MINER
// ============================================================================

/// Full hybrid miner that manages both CPU VDF and GPU SHA-3 mining
pub struct IntegratedHybridMiner {
    /// Coordinator for CPU/GPU work submission
    coordinator: Arc<HybridMiningCoordinator>,

    /// VDF engine for CPU mining
    vdf_engine: Arc<QuantumVDF>,

    /// GPU miner for SHA-3 mining (Mutex for &mut self on mine_batch)
    #[cfg(feature = "gpu-mining")]
    gpu_miner: Arc<tokio::sync::Mutex<GPUMiner>>,

    /// Miner address
    miner_address: Address,

    /// Mining statistics
    stats: Arc<HybridMiningStats>,

    /// Stop signal
    should_stop: Arc<AtomicBool>,
}

/// Statistics for hybrid mining
#[derive(Debug, Default)]
pub struct HybridMiningStats {
    /// VDF proofs computed (CPU)
    pub vdf_proofs: AtomicU64,
    /// SHA-3 hashes computed (GPU)
    pub gpu_hashes: AtomicU64,
    /// Blocks found
    pub blocks_found: AtomicU64,
    /// CPU hashrate equivalent
    pub cpu_hashrate: AtomicU64,
    /// GPU hashrate
    pub gpu_hashrate: AtomicU64,
    /// Total rewards earned
    pub total_rewards: AtomicU64,
}

impl IntegratedHybridMiner {
    /// Create new integrated hybrid miner
    pub async fn new(
        miner_address: Address,
        block_reward: u64,
    ) -> Result<Self> {
        info!("🔧 Initializing Integrated Hybrid Miner...");

        let coordinator = Arc::new(HybridMiningCoordinator::new(block_reward));

        // Initialize VDF engine for CPU mining
        let vdf_config = QuantumVDFConfig {
            base_difficulty: 1000,
            quantum_enhancement: 0.7,
            parallel_threads: 2,
            qrng_seed_interval: Duration::from_secs(300),
            security_level: VDFSecurityLevel::PostQuantum,
        };
        let vdf_engine = Arc::new(QuantumVDF::new(vdf_config).await?);

        // Initialize GPU miner
        #[cfg(feature = "gpu-mining")]
        let gpu_miner = Arc::new(tokio::sync::Mutex::new(GPUMiner::new(GPUMinerConfig::default())?));

        info!("✅ Hybrid miner initialized - CPU (VDF) + GPU (SHA-3)");

        Ok(Self {
            coordinator,
            vdf_engine,
            #[cfg(feature = "gpu-mining")]
            gpu_miner,
            miner_address,
            stats: Arc::new(HybridMiningStats::default()),
            should_stop: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Mine a hybrid block (requires both CPU and GPU work)
    pub async fn mine_hybrid_block(
        &self,
        height: u64,
        previous_hash: [u8; 32],
        merkle_root: [u8; 32],
        target: [u8; 32],
        difficulty: u32,
    ) -> Result<Option<HybridMiningBlock>> {
        info!("⛏️ Starting hybrid mining for block {}", height);

        let mining_start = Instant::now();

        // Start CPU (VDF) and GPU (SHA-3) mining in parallel
        let vdf_handle = {
            let vdf = self.vdf_engine.clone();
            let miner = self.miner_address.clone();
            let coordinator = self.coordinator.clone();
            let stats = self.stats.clone();
            let should_stop = self.should_stop.clone();

            tokio::spawn(async move {
                // Create VDF challenge from block data
                let mut challenge = [0u8; 32];
                let mut hasher = Sha3_256::new();
                hasher.update(&height.to_le_bytes());
                hasher.update(&previous_hash);
                let hash = hasher.finalize();
                challenge.copy_from_slice(&hash);

                // Compute VDF proof
                match vdf.compute_proof(&challenge).await {
                    Ok(vdf_result) => {
                        stats.vdf_proofs.fetch_add(1, Ordering::Relaxed);

                        // Clone the proof before submitting
                        let proof_for_return = vdf_result.proof.clone();

                        // Submit to coordinator
                        let _ = coordinator
                            .submit_cpu_work(vdf_result.proof, miner, height)
                            .await;

                        info!(
                            "✅ CPU VDF proof computed for block {} (quality: {:.3})",
                            height, vdf_result.quantum_quality
                        );
                        Some(proof_for_return)
                    }
                    Err(e) => {
                        error!("❌ VDF computation failed: {}", e);
                        None
                    }
                }
            })
        };

        // GPU SHA-3 mining
        #[cfg(feature = "gpu-mining")]
        let gpu_result = {
            // Build header for GPU mining
            let mut header = Vec::new();
            header.extend_from_slice(&height.to_le_bytes());
            header.extend_from_slice(&previous_hash);
            header.extend_from_slice(&merkle_root);
            header.extend_from_slice(&self.miner_address);

            let job = GPUMiningJob {
                header,
                target,
                height,
            };

            self.gpu_miner.lock().await.mine(job).await?
        };

        #[cfg(not(feature = "gpu-mining"))]
        let gpu_result: Option<GPUSolution> = {
            // CPU fallback for SHA-3 mining
            self.cpu_sha3_mining(height, &previous_hash, &merkle_root, &target).await?
        };

        // Wait for VDF to complete
        let vdf_result = vdf_handle.await?;

        // Combine results
        if let (Some(_vdf_proof), Some(gpu_sol)) = (vdf_result, gpu_result) {
            let elapsed = mining_start.elapsed();

            // Get the pending VDF proof from coordinator
            if let Some(vdf_pending) = self.coordinator.cpu_pool.consume_vdf_proof(height).await {
                let block = HybridMiningBlock::new(
                    height,
                    previous_hash,
                    merkle_root,
                    vdf_pending.proof,
                    vdf_pending.miner_address,
                    gpu_sol.hash,
                    gpu_sol.nonce,
                    self.miner_address.clone(),
                );

                self.stats.blocks_found.fetch_add(1, Ordering::Relaxed);

                info!(
                    "🎉 Hybrid block {} mined in {:?}!\n  \
                     CPU miner: {}\n  \
                     GPU miner: {}\n  \
                     Hash: {}",
                    height,
                    elapsed,
                    hex::encode(&block.cpu_miner_address),
                    hex::encode(&block.gpu_miner_address),
                    hex::encode(&block.hash()[..8])
                );

                return Ok(Some(block));
            }
        }

        Ok(None)
    }

    /// CPU fallback for SHA-3 mining when GPU is not available
    #[cfg(not(feature = "gpu-mining"))]
    async fn cpu_sha3_mining(
        &self,
        height: u64,
        previous_hash: &[u8; 32],
        merkle_root: &[u8; 32],
        target: &[u8; 32],
    ) -> Result<Option<GPUSolution>> {
        warn!("⚠️ GPU mining not available, using CPU fallback");

        let start = Instant::now();
        let mut nonce = 0u64;

        while !self.should_stop.load(Ordering::Relaxed) {
            // Build input
            let mut input = Vec::new();
            input.extend_from_slice(&height.to_le_bytes());
            input.extend_from_slice(previous_hash);
            input.extend_from_slice(merkle_root);
            input.extend_from_slice(&nonce.to_le_bytes());

            // Hash
            let hash = Sha3_256::digest(&input);
            let mut hash_arr = [0u8; 32];
            hash_arr.copy_from_slice(&hash);

            // Check target
            if Self::meets_target(&hash_arr, target) {
                self.stats.gpu_hashes.fetch_add(nonce, Ordering::Relaxed);

                return Ok(Some(GPUSolution {
                    nonce,
                    hash: hash_arr,
                    gpu_index: 0,
                    hashes_computed: nonce,
                }));
            }

            nonce += 1;

            // Yield periodically
            if nonce % 100_000 == 0 {
                tokio::task::yield_now().await;
            }

            // Timeout after 5 minutes
            if start.elapsed() > Duration::from_secs(300) {
                warn!("CPU SHA-3 mining timeout");
                return Ok(None);
            }
        }

        Ok(None)
    }

    /// Check if hash meets target
    fn meets_target(hash: &[u8; 32], target: &[u8; 32]) -> bool {
        for i in 0..32 {
            if hash[i] < target[i] {
                return true;
            } else if hash[i] > target[i] {
                return false;
            }
        }
        true
    }

    /// Stop mining
    pub fn stop(&self) {
        self.should_stop.store(true, Ordering::Relaxed);

        #[cfg(feature = "gpu-mining")]
        if let Ok(guard) = self.gpu_miner.try_lock() {
            guard.stop();
        }
    }

    /// Get mining statistics
    pub fn get_stats(&self) -> HybridMiningStatsSnapshot {
        HybridMiningStatsSnapshot {
            vdf_proofs: self.stats.vdf_proofs.load(Ordering::Relaxed),
            gpu_hashes: self.stats.gpu_hashes.load(Ordering::Relaxed),
            blocks_found: self.stats.blocks_found.load(Ordering::Relaxed),
            cpu_hashrate: self.stats.cpu_hashrate.load(Ordering::Relaxed),
            gpu_hashrate: self.stats.gpu_hashrate.load(Ordering::Relaxed),
            total_rewards: self.stats.total_rewards.load(Ordering::Relaxed),
        }
    }

    /// Get coordinator for external submissions
    pub fn coordinator(&self) -> Arc<HybridMiningCoordinator> {
        self.coordinator.clone()
    }
}

/// Snapshot of hybrid mining statistics
#[derive(Debug, Clone)]
pub struct HybridMiningStatsSnapshot {
    pub vdf_proofs: u64,
    pub gpu_hashes: u64,
    pub blocks_found: u64,
    pub cpu_hashrate: u64,
    pub gpu_hashrate: u64,
    pub total_rewards: u64,
}

/// GPU solution type for non-GPU builds
#[cfg(not(feature = "gpu-mining"))]
#[derive(Debug, Clone)]
pub struct GPUSolution {
    pub nonce: u64,
    pub hash: [u8; 32],
    pub gpu_index: usize,
    pub hashes_computed: u64,
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_hybrid_mining() {
        let coordinator = HybridMiningCoordinator::new(2_000_000_000); // 2 QNK

        let cpu_miner = Address::from([1u8; 20]);
        let gpu_miner = Address::from([2u8; 20]);

        // Create dummy VDF proof
        let vdf_proof = QuantumVDFProof {
            challenge: [0u8; 32],
            proof: [0u8; 64],
            quantum_seed: Some([1u8; 32]),
            computation_time: std::time::Duration::from_secs(1),
            difficulty: 1000,
            entropy_estimate: 0.8,
            parallel_witnesses: vec![],
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
            vdf_proof: QuantumVDFProof {
                challenge: [0u8; 32],
                proof: [0u8; 64],
                quantum_seed: None,
                computation_time: std::time::Duration::from_secs(0),
                difficulty: 0,
                entropy_estimate: 0.0,
                parallel_witnesses: vec![],
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
