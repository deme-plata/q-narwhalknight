pub mod config;
pub mod cpu;
pub mod gpu;
pub mod network;
pub mod ui;
pub mod utils;

pub use config::MinerConfig;
pub use cpu::CpuMiner;
pub use gpu::{CudaMiner, OpenClMiner};
pub use network::{PoolClient, StratumClient};
pub use ui::Dashboard;

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Core mining engine trait
#[async_trait::async_trait]
pub trait MiningEngine: Send + Sync {
    /// Start the mining engine
    async fn start(&mut self) -> Result<()>;

    /// Stop the mining engine
    async fn stop(&mut self) -> Result<()>;

    /// Get current hash rate
    async fn get_hash_rate(&self) -> f64;

    /// Get mining statistics
    async fn get_stats(&self) -> MiningStats;
}

/// Mining statistics for a single device
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MiningStats {
    pub hash_rate: f64,
    pub accepted_shares: u64,
    pub rejected_shares: u64,
    pub power_usage: f64,
    pub temperature: f64,
    pub uptime: chrono::Duration,
}

/// Core mining algorithm trait
#[async_trait::async_trait]
pub trait MiningAlgorithm: Send + Sync {
    /// Algorithm name
    fn name(&self) -> &str;
    
    /// Compute hash for given input
    async fn compute_hash(&self, input: &[u8], nonce: u64) -> Result<[u8; 32]>;
    
    /// Verify solution meets difficulty target
    async fn verify_solution(&self, hash: &[u8; 32], target: &[u8; 32]) -> bool;
    
    /// Get algorithm-specific parameters
    fn get_parameters(&self) -> AlgorithmParameters;
}

/// Algorithm configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmParameters {
    pub memory_requirement: u64,
    pub compute_intensity: u8,
    pub parallelization_factor: u32,
    pub quantum_resistance: bool,
}

/// Mining work unit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkUnit {
    pub job_id: String,
    pub previous_hash: [u8; 32],
    pub merkle_root: [u8; 32],
    pub timestamp: u64,
    pub difficulty_target: [u8; 32],
    pub nonce_range: (u64, u64),
    pub extra_data: Vec<u8>,
}

/// Mining solution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Solution {
    pub job_id: String,
    pub nonce: u64,
    pub hash: [u8; 32],
    pub timestamp: u64,
    pub worker_id: String,
}

/// Mining statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalMiningStats {
    pub total_hash_rate: f64,
    pub accepted_shares: u64,
    pub rejected_shares: u64,
    pub efficiency: f64,
    pub uptime: chrono::Duration,
    pub power_usage: f64,
    pub devices: Vec<DeviceStats>,
}

impl Default for GlobalMiningStats {
    fn default() -> Self {
        Self {
            total_hash_rate: 0.0,
            accepted_shares: 0,
            rejected_shares: 0,
            efficiency: 0.0,
            uptime: chrono::Duration::zero(),
            power_usage: 0.0,
            devices: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceStats {
    pub device_id: String,
    pub device_type: DeviceType,
    pub hash_rate: f64,
    pub temperature: f64,
    pub power_usage: f64,
    pub memory_usage: f64,
    pub utilization: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeviceType {
    CPU,
    CUDA(String),  // GPU model name
    OpenCL(String),
    Vulkan(String),
}

/// Mining event for real-time updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MiningEvent {
    /// New work received from pool
    NewWork(WorkUnit),
    
    /// Solution found and submitted
    SolutionFound {
        device_id: String,
        hash_rate: f64,
        nonce: u64,
    },
    
    /// Share accepted by pool
    ShareAccepted {
        job_id: String,
        difficulty: f64,
        reward: f64,
    },
    
    /// Share rejected by pool
    ShareRejected {
        job_id: String,
        reason: String,
    },
    
    /// Device status update
    DeviceUpdate {
        device_id: String,
        stats: DeviceStats,
    },
    
    /// Network status change
    NetworkEvent {
        connected: bool,
        peer_count: u32,
        pool_latency: f64,
    },
}

pub mod algorithms {
    use super::*;
    
    /// DAG-Knight VDF mining algorithm
    pub struct DagKnightVDF {
        difficulty: u64,
        vdf_iterations: u64,
    }
    
    impl DagKnightVDF {
        pub fn new(difficulty: u64) -> Self {
            Self {
                difficulty,
                vdf_iterations: difficulty * 1000,
            }
        }
    }
    
    #[async_trait::async_trait]
    impl MiningAlgorithm for DagKnightVDF {
        fn name(&self) -> &str {
            "dag-knight-vdf"
        }
        
        async fn compute_hash(&self, input: &[u8], nonce: u64) -> Result<[u8; 32]> {
            // Combine input with nonce
            let mut hasher_input = Vec::with_capacity(input.len() + 8);
            hasher_input.extend_from_slice(input);
            hasher_input.extend_from_slice(&nonce.to_le_bytes());
            
            // Initial hash
            let initial_hash = blake3::hash(&hasher_input);
            
            // VDF computation
            let mut current = initial_hash.as_bytes().to_vec();
            for _ in 0..self.vdf_iterations {
                current = blake3::hash(&current).as_bytes().to_vec();
            }
            
            let mut result = [0u8; 32];
            result.copy_from_slice(&current[..32]);
            Ok(result)
        }
        
        async fn verify_solution(&self, hash: &[u8; 32], target: &[u8; 32]) -> bool {
            // Check if hash meets difficulty target (hash < target)
            hash < target
        }
        
        fn get_parameters(&self) -> AlgorithmParameters {
            AlgorithmParameters {
                memory_requirement: 1024 * 1024, // 1MB
                compute_intensity: 8,
                parallelization_factor: 1,
                quantum_resistance: true,
            }
        }
    }
    
    /// Quantum-enhanced Blake3 mining
    pub struct QuantumBlake3 {
        rounds: u32,
    }
    
    impl QuantumBlake3 {
        pub fn new(rounds: u32) -> Self {
            Self { rounds }
        }
    }
    
    #[async_trait::async_trait]
    impl MiningAlgorithm for QuantumBlake3 {
        fn name(&self) -> &str {
            "quantum-blake3"
        }
        
        async fn compute_hash(&self, input: &[u8], nonce: u64) -> Result<[u8; 32]> {
            let mut hasher_input = Vec::with_capacity(input.len() + 8);
            hasher_input.extend_from_slice(input);
            hasher_input.extend_from_slice(&nonce.to_le_bytes());
            
            let mut hash = blake3::hash(&hasher_input);
            
            // Multiple rounds for increased security
            for _ in 1..self.rounds {
                hash = blake3::hash(hash.as_bytes());
            }
            
            Ok(*hash.as_bytes())
        }
        
        async fn verify_solution(&self, hash: &[u8; 32], target: &[u8; 32]) -> bool {
            hash < target
        }
        
        fn get_parameters(&self) -> AlgorithmParameters {
            AlgorithmParameters {
                memory_requirement: 512 * 1024, // 512KB
                compute_intensity: 6,
                parallelization_factor: 4,
                quantum_resistance: true,
            }
        }
    }
}