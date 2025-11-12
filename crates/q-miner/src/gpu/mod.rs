pub mod cuda;
pub mod opencl;
pub mod vulkan;
pub mod multi_gpu;

#[cfg(feature = "cuda-mining")]
pub use cuda::CudaMiner;
#[cfg(not(feature = "cuda-mining"))]
pub use cuda::CudaMinerStub as CudaMiner;

pub use opencl::OpenClMiner;
pub use vulkan::VulkanMiner;
pub use multi_gpu::{MultiGPUMiner, GPUDeviceInfo, LoadBalancingStrategy};

use crate::{MiningAlgorithm, MiningEngine, MiningStats, WorkUnit, Solution, GlobalMiningStats};
use anyhow::Result;
use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, debug, error};

/// GPU device information
#[derive(Debug, Clone)]
pub struct GpuDeviceInfo {
    pub device_id: u32,
    pub name: String,
    pub vendor: String,
    pub compute_capability: Option<String>,
    pub memory_total: u64,
    pub memory_free: u64,
    pub core_count: u32,
    pub max_threads_per_block: u32,
    pub max_shared_memory: u32,
}

/// Common GPU mining interface
#[async_trait]
pub trait GpuMiner: MiningEngine {
    /// Initialize GPU device
    async fn initialize_device(&mut self, device_id: u32) -> Result<()>;

    /// Allocate GPU memory for mining
    async fn allocate_memory(&mut self, size: usize) -> Result<()>;

    /// Load mining kernel onto GPU
    async fn load_kernel(&mut self, algorithm: Arc<dyn MiningAlgorithm>) -> Result<()>;

    /// Submit work to GPU for processing
    async fn submit_work(&mut self, work: WorkUnit) -> Result<()>;

    /// Check for completed solutions
    async fn check_solutions(&mut self) -> Result<Vec<Solution>>;

    /// Get device temperature
    async fn get_temperature(&self) -> Result<f64>;

    /// Get device power usage
    async fn get_power_usage(&self) -> Result<f64>;

    /// Get device utilization percentage
    async fn get_utilization(&self) -> Result<f64>;
}

/// Simplified GPU mining backend for multi-GPU coordination
#[async_trait]
pub trait GpuMiningBackend: Send + Sync {
    /// Get device ID
    fn device_id(&self) -> usize;

    /// Submit mining work
    async fn submit_work(&mut self, work: WorkUnit) -> Result<()>;

    /// Check for solutions
    async fn check_solutions(&mut self) -> Result<Vec<Solution>>;

    /// Get current hash rate
    async fn get_hash_rate(&self) -> Result<f64>;

    /// Get device stats
    async fn get_stats(&self) -> Result<crate::DeviceStats>;
}

/// GPU mining configuration
#[derive(Debug, Clone)]
pub struct GpuMiningConfig {
    pub device_ids: Vec<u32>,
    pub threads_per_block: u32,
    pub blocks_per_grid: u32,
    pub memory_limit: u64,
    pub intensity: u8,
    pub enable_monitoring: bool,
}

impl Default for GpuMiningConfig {
    fn default() -> Self {
        Self {
            device_ids: vec![0],
            threads_per_block: 256,
            blocks_per_grid: 1024,
            memory_limit: 1024 * 1024 * 1024, // 1GB
            intensity: 7,
            enable_monitoring: true,
        }
    }
}

/// Shared GPU mining utilities
pub mod utils {
    use super::*;
    
    /// Detect available GPU devices
    pub async fn detect_gpu_devices() -> Result<Vec<GpuDeviceInfo>> {
        let mut devices = Vec::new();
        
        // CUDA device detection
        #[cfg(feature = "cuda-mining")]
        {
            if let Ok(cuda_devices) = cuda::detect_cuda_devices().await {
                devices.extend(cuda_devices);
            }
        }
        
        // OpenCL device detection
        #[cfg(feature = "opencl-mining")]
        {
            if let Ok(opencl_devices) = opencl::detect_opencl_devices().await {
                devices.extend(opencl_devices);
            }
        }
        
        Ok(devices)
    }
    
    /// Calculate optimal GPU mining parameters
    pub fn calculate_optimal_params(device: &GpuDeviceInfo, intensity: u8) -> GpuMiningConfig {
        let memory_limit = (device.memory_total as f64 * 0.8) as u64; // Use 80% of GPU memory
        let intensity_factor = intensity as f64 / 10.0;
        
        let threads_per_block = match device.vendor.as_str() {
            "NVIDIA" => 256,
            "AMD" => 64,
            _ => 128,
        };
        
        let blocks_per_grid = ((device.core_count as f64 * intensity_factor) as u32).max(32);
        
        GpuMiningConfig {
            device_ids: vec![device.device_id],
            threads_per_block,
            blocks_per_grid,
            memory_limit,
            intensity,
            enable_monitoring: true,
        }
    }
    
    /// Convert difficulty target to GPU format
    pub fn difficulty_to_target(difficulty: f64) -> [u8; 32] {
        let target_value = (u64::MAX as f64 / difficulty) as u64;
        let mut target = [0u8; 32];
        target[24..32].copy_from_slice(&target_value.to_be_bytes());
        target
    }
}

/// GPU mining kernel interface
pub trait MiningKernel: Send + Sync {
    /// Kernel name
    fn name(&self) -> &str;
    
    /// Launch kernel with given parameters
    fn launch(&self, work: &WorkUnit, nonce_start: u64, nonce_count: u64) -> Result<()>;
    
    /// Check if kernel execution is complete
    fn is_complete(&self) -> bool;
    
    /// Retrieve results from kernel
    fn get_results(&self) -> Result<Vec<Solution>>;
    
    /// Get kernel performance metrics
    fn get_metrics(&self) -> KernelMetrics;
}

#[derive(Debug, Clone)]
pub struct KernelMetrics {
    pub execution_time_ms: f64,
    pub hash_rate: f64,
    pub memory_bandwidth: f64,
    pub compute_utilization: f64,
}

/// Multi-GPU mining coordinator
pub struct MultiGpuCoordinator {
    miners: Vec<Box<dyn GpuMiner>>,
    work_distributor: WorkDistributor,
    solution_collector: SolutionCollector,
    stats: Arc<RwLock<GlobalMiningStats>>,
}

impl MultiGpuCoordinator {
    pub async fn new(devices: Vec<u32>, intensity: u8) -> Result<Self> {
        let mut miners: Vec<Box<dyn GpuMiner>> = Vec::new();
        
        for &device_id in &devices {
            #[cfg(feature = "cuda-mining")]
            {
                if cuda::is_cuda_device(device_id).await? {
                    let miner = CudaMiner::new(vec![device_id], intensity).await?;
                    miners.push(Box::new(miner));
                    continue;
                }
            }
            
            #[cfg(feature = "opencl-mining")]
            {
                let miner = OpenClMiner::new(vec![device_id], intensity).await?;
                miners.push(Box::new(miner));
            }
        }
        
        let work_distributor = WorkDistributor::new(miners.len());
        let solution_collector = SolutionCollector::new();
        let stats = Arc::new(RwLock::new(GlobalMiningStats::default()));
        
        Ok(Self {
            miners,
            work_distributor,
            solution_collector,
            stats,
        })
    }
    
    pub async fn start_mining(&mut self, work: WorkUnit) -> Result<()> {
        info!("🚀 Starting multi-GPU mining for job {}", work.job_id);
        
        // Distribute work across GPUs
        let work_units = self.work_distributor.distribute_work(work).await?;
        
        // Start mining on each GPU
        for (miner, work_unit) in self.miners.iter_mut().zip(work_units.iter()) {
            miner.submit_work(work_unit.clone()).await?;
        }
        
        Ok(())
    }
    
    pub async fn collect_solutions(&mut self) -> Result<Vec<Solution>> {
        let mut all_solutions = Vec::new();
        
        for miner in &mut self.miners {
            let solutions = miner.check_solutions().await?;
            all_solutions.extend(solutions);
        }
        
        Ok(all_solutions)
    }
}

/// Work distribution for multiple mining devices
pub struct WorkDistributor {
    device_count: usize,
}

impl WorkDistributor {
    pub fn new(device_count: usize) -> Self {
        Self { device_count }
    }
    
    pub async fn distribute_work(&self, work: WorkUnit) -> Result<Vec<WorkUnit>> {
        let nonce_range_size = work.nonce_range.1 - work.nonce_range.0;
        let nonces_per_device = nonce_range_size / self.device_count as u64;
        
        let mut work_units = Vec::with_capacity(self.device_count);
        
        for i in 0..self.device_count {
            let start_nonce = work.nonce_range.0 + (i as u64 * nonces_per_device);
            let end_nonce = if i == self.device_count - 1 {
                work.nonce_range.1 // Last device gets remaining nonces
            } else {
                start_nonce + nonces_per_device
            };
            
            let mut device_work = work.clone();
            device_work.nonce_range = (start_nonce, end_nonce);
            work_units.push(device_work);
        }
        
        Ok(work_units)
    }
}

/// Solution collection and validation
pub struct SolutionCollector {
    pending_solutions: Vec<Solution>,
}

impl SolutionCollector {
    pub fn new() -> Self {
        Self {
            pending_solutions: Vec::new(),
        }
    }
    
    pub async fn add_solution(&mut self, solution: Solution) {
        self.pending_solutions.push(solution);
    }
    
    pub async fn get_best_solutions(&mut self, count: usize) -> Vec<Solution> {
        // Sort by hash value (lowest first)
        self.pending_solutions.sort_by(|a, b| a.hash.cmp(&b.hash));
        
        let best_solutions = self.pending_solutions
            .drain(..count.min(self.pending_solutions.len()))
            .collect();
            
        best_solutions
    }
}