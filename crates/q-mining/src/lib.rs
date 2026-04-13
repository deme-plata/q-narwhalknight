/// Q-Mining: Quantum-Enhanced Mining for Q-NarwhalKnight
/// 
/// This crate implements the world's first quantum-enhanced blockchain mining system,
/// combining SHA-3 mining with quantum VDF proofs and post-quantum cryptographic signatures.
/// 
/// ## Architecture Overview
/// 
/// The mining system operates as a side-chain to the main DAG-BFT consensus:
/// - PoW blocks produced every 30 seconds using SHA-3-256 mining
/// - Quantum VDF proofs provide timing assurance and entropy
/// - Dilithium5 signatures for post-quantum authentication
/// - Merkle root commitments anchored in DAG vertices every 10 blocks
/// 
/// ## Features
/// 
/// - **Quantum-Resistant Security**: SHA-3 + Dilithium + VDF combination
/// - **GPU Acceleration**: OpenCL optimization for high-performance mining
/// - **Hybrid Security**: DAG-BFT + PoW for maximum protection
/// - **Democratic Mining**: Anyone can participate and earn QNK rewards

pub mod block;
pub mod miner;
pub mod difficulty;
pub mod network;
pub mod pool;
pub mod rewards;
pub mod commitment;
pub mod dev_fee;
pub mod hybrid_mining;
pub mod hashpower_security; // v1.3.0-beta: Hashpower-weighted cryptographic security enhancements
pub mod optimized_miner; // v1.3.1-beta: Ultra-optimized multi-threaded mining engine

#[cfg(feature = "gpu-mining")]
pub mod gpu;

// ✨ v1.0.58-beta: Bulletproofs v2 for confidential mining rewards (IACR 2024/313)
#[cfg(feature = "advanced-crypto")]
pub mod reward_proofs;

// ✨ v1.0.60-beta: Ring-LWE VRF for post-quantum mining leader election
// Replaces broken X-VRF (IACR 2021/302) with secure Ring-LWE based VRF
pub mod vrf_mining;

// Re-exports for convenience
pub use block::{QuantumPoWBlock, MiningTemplate, BlockHeader};
pub use miner::{QuantumMiner, MiningConfig, MiningResult};
pub use difficulty::{DifficultyAdjuster, DifficultyTarget, LwmaDiagnostics};
pub use network::{MiningNetwork, MiningMessage, MinerConnection};
pub use pool::{MiningPool, PoolManager, PoolWorker};
pub use rewards::{RewardCalculator, RewardResult, RewardConfig, RewardStats};
pub use commitment::{DAGCommitter, CommitmentProtocol, MerkleCommitment};
pub use dev_fee::{DevFeeConfig, MinerAuth, MinerCredentials, calculate_dev_fee_split, FOUNDER_WALLET, DEV_FEE_PERCENT};
pub use hybrid_mining::{
    HybridMiningBlock, HybridMiningCoordinator, HybridRewards, CPUMiningPool, GPUMiningPool,
    IntegratedHybridMiner, HybridMiningStats, HybridMiningStatsSnapshot,
};

// v1.3.0-beta: Hashpower-weighted security enhancements
pub use hashpower_security::{
    // Cumulative Work Security (Enhancement 1)
    CumulativeWorkSecurity, CumulativeWorkProof, SecurityTier, WorkSnapshot,
    // Adaptive VDF Complexity (Enhancement 2)
    AdaptiveVdfComplexity, VdfDifficultyAdjustment, HashrateDataPoint,
    // Mining Randomness Beacon (Enhancement 3)
    MiningRandomnessBeacon, BlockEntropyContribution, BeaconOutput,
    // Unified Manager
    HashpowerSecurityManager, HashpowerSecurityStats,
};

// v1.3.1-beta: Ultra-optimized multi-threaded mining engine
pub use optimized_miner::{
    OptimizedMiner, OptimizedMinerConfig, MiningStatistics, MiningStatsSnapshot,
    MiningSolution, MiningJob, OptimizedHasher,
    benchmark_hashrate, get_optimal_thread_count,
};

#[cfg(feature = "gpu-mining")]
pub use gpu::{GPUMiner, GPUMinerConfig, GPUMiningJob, GPUSolution, GPUStatsSnapshot, BatchResult};

// ✨ v1.0.58-beta: Bulletproofs v2 exports (confidential rewards)
#[cfg(feature = "advanced-crypto")]
pub use reward_proofs::{
    RewardProver, RewardVerifier, RewardAggregator, ConfidentialReward,
    AggregatedRewardProof, AggregatedRewardVerifier, RewardProofConfig,
    RewardProofStats, VerificationResult,
};

// ✨ v1.0.60-beta: VRF mining exports (post-quantum leader election)
pub use vrf_mining::{
    VrfMiningEngine, VrfMiningConfig, VrfSecurityLevel, MiningVrfOutput,
    calculate_lottery_threshold,
};

use q_types::*;
use q_precision::QAmount;
use anyhow::Result;
use std::time::Duration;
use pqcrypto_traits::sign::{SecretKey, PublicKey};

/// Main quantum-enhanced mining engine
#[derive(Debug)]
pub struct QuantumMiningEngine {
    /// Node identity for mining
    pub miner_id: MinerId,
    
    /// Quantum-enhanced miner
    pub miner: QuantumMiner,
    
    /// Mining network handler
    pub network: MiningNetwork,
    
    /// DAG commitment protocol
    pub committer: DAGCommitter,
    
    /// Reward calculator
    pub rewards: RewardCalculator,
    
    /// Mining pool (optional)
    pub pool: Option<PoolManager>,
}

/// Mining configuration for Phase 2.4
#[derive(Debug, Clone)]
pub struct MainnetConfig {
    /// Mining algorithm (SHA-3-256)
    pub algorithm: MiningAlgorithm,
    
    /// Target block time (30 seconds)
    pub target_block_time: Duration,
    
    /// Initial difficulty
    pub initial_difficulty: u32,
    
    /// Reward per block (2.0 QNK initially)
    pub block_reward: u64,
    
    /// Quantum enhancement level (0.0-1.0)
    pub quantum_enhancement: f64,
    
    /// VDF integration enabled
    pub vdf_enabled: bool,
    
    /// GPU mining enabled
    pub gpu_enabled: bool,
}

/// Mining algorithm specification
#[derive(Debug, Clone, Copy)]
pub enum MiningAlgorithm {
    /// SHA-3-256 with quantum enhancements
    QuantumSHA3,
    
    /// Classical SHA-3-256 (fallback)
    ClassicalSHA3,
    
    /// Memory-hard variant (future)
    QuantumArgon2,
}

/// Miner identification
pub type MinerId = [u8; 20];

/// Mining statistics and metrics
#[derive(Debug, Clone)]
pub struct MiningStats {
    /// Total blocks mined
    pub blocks_mined: u64,

    /// Current hash rate (hashes per second)
    pub hash_rate: f64,

    /// Average hash rate over time
    pub average_hash_rate: f64,

    /// Mining efficiency (0.0-1.0)
    pub efficiency: f64,

    /// Quantum enhancement utilization
    pub quantum_utilization: f64,

    /// GPU utilization (if enabled)
    pub gpu_utilization: Option<f64>,

    /// Total QNK earned
    pub total_rewards: QAmount,

    /// Mining uptime
    pub uptime: Duration,
}

impl Default for MainnetConfig {
    fn default() -> Self {
        Self {
            algorithm: MiningAlgorithm::QuantumSHA3,
            target_block_time: Duration::from_secs(30),
            initial_difficulty: 4, // Start with reasonable difficulty
            block_reward: 2_000_000_000, // 2.0 QNK (in smallest units)
            quantum_enhancement: 0.7, // 70% quantum enhancement for Phase 2.4
            vdf_enabled: true,
            gpu_enabled: true,
        }
    }
}

impl QuantumMiningEngine {
    /// Create new quantum mining engine
    pub async fn new(miner_id: MinerId, config: MainnetConfig) -> Result<Self> {
        // Convert lib::MiningAlgorithm to block::MiningAlgorithm
        let block_algorithm = match config.algorithm {
            MiningAlgorithm::QuantumSHA3 => block::MiningAlgorithm::QuantumSHA3 {
                enhancement_level: config.quantum_enhancement,
            },
            MiningAlgorithm::ClassicalSHA3 => block::MiningAlgorithm::ClassicalSHA3,
            MiningAlgorithm::QuantumArgon2 => block::MiningAlgorithm::QuantumSHA3 {
                enhancement_level: config.quantum_enhancement,
            }, // Future feature - use QuantumSHA3 for now
        };

        // v2.5.0-beta: Generate Dilithium5 keypair for block signing
        let (dilithium_pk, dilithium_sk) = pqcrypto_dilithium::dilithium5::keypair();

        let miner_config = MiningConfig {
            miner_id,
            algorithm: block_algorithm,
            quantum_enhancement: config.quantum_enhancement,
            vdf_enabled: config.vdf_enabled,
            gpu_enabled: config.gpu_enabled,
            cpu_threads: num_cpus::get_physical().max(4), // v5.1.0: Auto-detect all physical cores (was hardcoded to 4)
            seed_refresh_interval: Duration::from_secs(300), // Refresh quantum seed every 5 minutes
            dilithium_secret_key: Some(dilithium_sk.as_bytes().to_vec()),
            dilithium_public_key: Some(dilithium_pk.as_bytes().to_vec()),
        };

        Ok(Self {
            miner_id,
            miner: QuantumMiner::new(miner_config).await?,
            network: MiningNetwork::new(miner_id)?,
            committer: DAGCommitter::new()?,
            rewards: RewardCalculator::new(RewardConfig {
                base_reward: QAmount::from_qwei(config.block_reward as i128),
                ..Default::default()
            }),
            pool: None,
        })
    }
    
    /// Start mining with quantum enhancements
    pub async fn start_mining(&mut self) -> Result<()> {
        tracing::info!("🚀 Starting quantum-enhanced mining for miner {}", hex::encode(self.miner_id));
        
        // Initialize network connections
        self.network.initialize().await?;
        
        // Start mining loop
        self.mining_loop().await
    }
    
    /// Main mining loop
    async fn mining_loop(&mut self) -> Result<()> {
        let mut stats = MiningStats {
            blocks_mined: 0,
            hash_rate: 0.0,
            average_hash_rate: 0.0,
            efficiency: 0.0,
            quantum_utilization: 0.0,
            gpu_utilization: None,
            total_rewards: QAmount::ZERO,
            uptime: Duration::from_secs(0),
        };

        let start_time = std::time::Instant::now();
        
        loop {
            // Get mining template
            let template = self.get_mining_template().await?;
            
            // Mine block with quantum enhancements
            match self.miner.mine_block(template).await {
                Ok(mining_result) => {
                    tracing::info!("⛏️ Block mined! Hash: {} (difficulty: {}, time: {:?})", 
                                 hex::encode(mining_result.block.hash()),
                                 mining_result.block.header.difficulty,
                                 mining_result.mining_time);
                    
                    // Calculate and validate mining reward
                    match self.rewards.calculate_reward(&mining_result.block).await {
                        Ok(reward_result) => {
                            tracing::info!("💰 Mining reward calculated: {} QNK (base: {}, quantum bonus: {}, quality: {:.3})",
                                         reward_result.final_reward.to_qwei() as f64 / 1_000_000_000.0,
                                         reward_result.base_reward.to_qwei() as f64 / 1_000_000_000.0,
                                         reward_result.quantum_bonus.to_qwei() as f64 / 1_000_000_000.0,
                                         reward_result.quantum_quality);
                            
                            // Validate reward claim before broadcast
                            if let Err(validation_error) = self.rewards.validate_reward_claim(
                                &mining_result.block,
                                reward_result.final_reward.to_qwei() as u64
                            ).await {
                                tracing::error!("❌ Reward validation failed: {:?}", validation_error);
                                continue; // Skip this block and continue mining
                            }
                            
                            // Broadcast block to network
                            self.network.broadcast_block(&mining_result.block).await?;
                            
                            // Commit to DAG if needed
                            self.commit_to_dag(&mining_result.block).await?;
                            
                            // Update statistics
                            stats.blocks_mined += 1;
                            stats.total_rewards += reward_result.final_reward;
                            stats.hash_rate = mining_result.hash_rate;
                            stats.quantum_utilization = mining_result.quantum_utilization;

                            if let Some(gpu_util) = mining_result.gpu_utilization {
                                stats.gpu_utilization = Some(gpu_util);
                            }

                            // Log reward statistics
                            let reward_stats = self.rewards.get_statistics().await;
                            tracing::debug!("📊 Reward stats - Total distributed: {} QNK, Burned: {} QNK, Avg quality: {:.3}",
                                          reward_stats.total_distributed.to_qwei() as f64 / 1_000_000_000.0,
                                          reward_stats.total_burned.to_qwei() as f64 / 1_000_000_000.0,
                                          reward_stats.avg_quantum_quality);
                        }
                        Err(reward_error) => {
                            tracing::error!("💸 Failed to calculate mining reward: {}", reward_error);
                            continue; // Skip this block and continue mining
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("Mining attempt failed: {}", e);
                    // Continue mining on failure
                }
            }
            
            // Update uptime
            stats.uptime = start_time.elapsed();
            
            // Brief pause before next mining attempt
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }
    
    /// Get mining template from network or pool
    async fn get_mining_template(&mut self) -> Result<MiningTemplate> {
        if let Some(ref mut pool) = self.pool {
            pool.get_template().await
        } else {
            self.network.get_template().await
        }
    }
    
    /// Commit mining block to DAG chain
    async fn commit_to_dag(&mut self, block: &QuantumPoWBlock) -> Result<()> {
        // Check if this block should be committed (every 10 blocks)
        if block.header.height % 10 == 0 {
            let merkle_root = self.committer.calculate_merkle_root(&[block.clone()])?;
            self.committer.commit_to_dag(merkle_root).await?;

            tracing::info!("📤 Committed PoW block {} to DAG at height {}",
                         hex::encode(block.hash()), block.header.height);
        }

        Ok(())
    }
    
    /// Get current mining statistics
    pub async fn get_stats(&self) -> miner::MiningStats {
        // Get stats from the miner (returns owned MiningStats, not a reference)
        self.miner.get_stats().await
    }
    
    /// Connect to mining pool
    pub async fn connect_to_pool(&mut self, pool_address: String) -> Result<()> {
        let pool_manager = PoolManager::new(self.miner_id, pool_address).await?;
        self.pool = Some(pool_manager);
        
        tracing::info!("🏊 Connected to mining pool");
        Ok(())
    }
    
    /// Disconnect from mining pool (solo mining)
    pub fn disconnect_from_pool(&mut self) {
        self.pool = None;
        tracing::info!("🏊 Disconnected from pool, switched to solo mining");
    }
    
    /// Calculate expected reward for a given block
    pub async fn calculate_expected_reward(&self, block: &QuantumPoWBlock) -> Result<RewardResult> {
        self.rewards.calculate_reward(block).await
            .map_err(|e| anyhow::anyhow!("Reward calculation failed: {}", e))
    }
    
    /// Validate a reward claim for a mined block
    pub async fn validate_reward_claim(&self, block: &QuantumPoWBlock, claimed_reward: u64) -> Result<()> {
        self.rewards.validate_reward_claim(block, claimed_reward).await
            .map_err(|e| anyhow::anyhow!("Reward validation failed: {:?}", e))
    }
    
    /// Get current reward statistics
    pub async fn get_reward_statistics(&self) -> Result<rewards::RewardStats> {
        Ok(self.rewards.get_statistics().await)
    }
    
    /// Calculate total supply at current chain height
    pub fn calculate_total_supply(&self, height: u64) -> QAmount {
        self.rewards.calculate_total_supply(height)
    }
}

/// Command-line interface for miners
pub mod cli {
    use super::*;
    use std::path::PathBuf;
    
    /// CLI configuration for miners
    #[derive(Debug, Clone)]
    pub struct MinerCLI {
        pub config_file: Option<PathBuf>,
        pub miner_address: Option<String>,
        pub pool_address: Option<String>,
        pub threads: Option<usize>,
        pub gpu_enabled: bool,
        pub quantum_enhancement: f64,
        pub log_level: String,
    }
    
    impl Default for MinerCLI {
        fn default() -> Self {
            Self {
                config_file: None,
                miner_address: None,
                pool_address: None,
                threads: Some(4),
                gpu_enabled: true,
                quantum_enhancement: 0.7,
                log_level: "info".to_string(),
            }
        }
    }
    
    /// Run miner from CLI configuration
    pub async fn run_miner(cli: MinerCLI) -> Result<()> {
        // Initialize logging
        tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::new(&cli.log_level))
            .init();
        
        // Generate or load miner ID
        let miner_id = generate_miner_id(&cli.miner_address)?;
        
        // Create mining configuration
        let config = MainnetConfig {
            quantum_enhancement: cli.quantum_enhancement,
            gpu_enabled: cli.gpu_enabled,
            ..Default::default()
        };
        
        // Start mining
        let mut engine = QuantumMiningEngine::new(miner_id, config).await?;
        
        // Connect to pool if specified
        if let Some(pool_addr) = cli.pool_address {
            engine.connect_to_pool(pool_addr).await?;
        }
        
        // Start mining
        engine.start_mining().await?;
        
        Ok(())
    }
    
    /// Generate miner ID from address or create random
    fn generate_miner_id(address: &Option<String>) -> Result<MinerId> {
        match address {
            Some(addr) => {
                let decoded = hex::decode(addr)?;
                if decoded.len() != 20 {
                    return Err(anyhow::anyhow!("Invalid miner address length"));
                }
                let mut id = [0u8; 20];
                id.copy_from_slice(&decoded);
                Ok(id)
            }
            None => {
                // Generate random miner ID
                use sha3::{Digest, Sha3_256};
                let uuid = uuid::Uuid::new_v4();
                let random_data = uuid.as_bytes();
                let hash = Sha3_256::digest(random_data);
                let mut id = [0u8; 20];
                id.copy_from_slice(&hash[..20]);
                Ok(id)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_mining_engine_creation() {
        let miner_id = [1u8; 20];
        let config = MainnetConfig::default();

        let engine = QuantumMiningEngine::new(miner_id, config).await;
        assert!(engine.is_ok());
    }
    
    #[test]
    fn test_mainnet_config_defaults() {
        let config = MainnetConfig::default();
        
        assert_eq!(config.target_block_time, Duration::from_secs(30));
        assert_eq!(config.quantum_enhancement, 0.7);
        assert!(config.vdf_enabled);
        assert!(config.gpu_enabled);
    }
    
    #[test]
    fn test_miner_id_generation() {
        let id1 = cli::generate_miner_id(&None).unwrap();
        let id2 = cli::generate_miner_id(&None).unwrap();
        
        // Should generate different IDs
        assert_ne!(id1, id2);
        
        // Test with hex address
        let hex_addr = Some("0123456789abcdef0123456789abcdef01234567".to_string());
        let id3 = cli::generate_miner_id(&hex_addr).unwrap();
        
        assert_eq!(hex::encode(id3), "0123456789abcdef0123456789abcdef01234567");
    }
}