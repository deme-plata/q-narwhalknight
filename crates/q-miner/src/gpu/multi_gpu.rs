/// Multi-GPU Mining Support (Inspired by mistral.rs device_map.rs)
///
/// This module implements multi-GPU mining coordination similar to how mistral.rs
/// distributes AI inference across multiple GPUs. We use the same device mapping
/// patterns but apply them to mining workload distribution.

use anyhow::{Context, Result};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::mpsc;

#[cfg(feature = "cuda-mining")]
use cudarc::driver::CudaDevice;

#[cfg(feature = "opencl-mining")]
use opencl3::device::{Device as OpenCLDevice, CL_DEVICE_TYPE_GPU};

use super::GpuMiningBackend;
use crate::{DeviceStats, DeviceType, MiningEvent, WorkUnit};

/// GPU Device Information (like mistral.rs DeviceLayerMapMetadata)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPUDeviceInfo {
    /// Device ordinal (0, 1, 2, ...)
    pub ordinal: usize,

    /// Device name/model
    pub name: String,

    /// Device type
    pub device_type: GPUDeviceType,

    /// Compute capability (CUDA) or version (OpenCL)
    pub compute_version: String,

    /// Total memory in GB
    pub memory_gb: f64,

    /// Estimated hash capacity (H/s)
    pub estimated_hash_rate: f64,

    /// Maximum power draw in watts
    pub max_power_watts: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum GPUDeviceType {
    CUDA,
    OpenCL,
    Vulkan,
    Metal,
}

/// Multi-GPU Device Mapping (inspired by mistral.rs DeviceMapMetadata)
#[derive(Debug, Clone)]
pub struct MultiGPUDeviceMap {
    /// Detected GPU devices
    devices: Vec<GPUDeviceInfo>,

    /// Device to work unit mapping
    work_distribution: Vec<WorkDistribution>,

    /// Load balancing strategy
    load_strategy: LoadBalancingStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkDistribution {
    pub device_id: usize,
    pub work_percentage: f64,
    pub priority: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    /// Equal distribution across all GPUs
    Equal,

    /// Distribute based on hash capacity
    CapacityBased,

    /// Dynamic adjustment based on real-time performance
    Dynamic,

    /// Manual assignment per device
    Manual(Vec<f64>),
}

/// Multi-GPU Mining Coordinator
pub struct MultiGPUMiner {
    /// Device map
    device_map: Arc<MultiGPUDeviceMap>,

    /// Individual GPU miners
    gpu_miners: Vec<Box<dyn GpuMiningBackend>>,

    /// Metrics aggregator
    metrics: Arc<RwLock<MultiGPUMetrics>>,

    /// Event broadcast channel
    event_tx: mpsc::UnboundedSender<MiningEvent>,

    /// Work distribution queue
    work_queue: Arc<RwLock<Vec<WorkUnit>>>,

    /// Active status
    active: Arc<RwLock<bool>>,
}

#[derive(Debug, Clone, Default)]
pub struct MultiGPUMetrics {
    pub total_hash_rate: f64,
    pub device_metrics: Vec<DeviceMetrics>,
    pub work_distribution_efficiency: f64,
    pub gpu_sync_overhead_ms: f64,
}

#[derive(Debug, Clone)]
pub struct DeviceMetrics {
    pub device_id: usize,
    pub hash_rate: f64,
    pub temperature: f64,
    pub power_usage: f64,
    pub memory_used_gb: f64,
    pub utilization_percent: f64,
    pub shares_accepted: u64,
    pub shares_rejected: u64,
}

impl MultiGPUMiner {
    /// Auto-detect and initialize all available GPUs (like mistral.rs get_all_similar_devices)
    pub async fn auto_detect() -> Result<Self> {
        tracing::info!("🔍 Auto-detecting GPUs for mining...");

        let mut detected_devices: Vec<GPUDeviceInfo> = Vec::new();

        // Detect CUDA devices
        #[cfg(feature = "cuda-mining")]
        {
            if let Ok(cuda_devices) = Self::detect_cuda_devices() {
                detected_devices.extend(cuda_devices);
                tracing::info!("✅ Detected {} CUDA device(s)", detected_devices.len());
            }
        }

        // Detect OpenCL devices
        #[cfg(feature = "opencl-mining")]
        {
            if let Ok(opencl_devices) = Self::detect_opencl_devices() {
                let opencl_count = opencl_devices.len();
                detected_devices.extend(opencl_devices);
                tracing::info!("✅ Detected {} OpenCL device(s)", opencl_count);
            }
        }

        if detected_devices.is_empty() {
            anyhow::bail!("No compatible GPU devices detected. Please install CUDA/OpenCL drivers.");
        }

        tracing::info!("📊 Total GPUs detected: {}", detected_devices.len());
        for (idx, device) in detected_devices.iter().enumerate() {
            tracing::info!(
                "  GPU {}: {} ({} GB, est. {:.2} MH/s)",
                idx,
                device.name,
                device.memory_gb,
                device.estimated_hash_rate / 1_000_000.0
            );
        }

        // Create device map with capacity-based load balancing
        let device_map = Arc::new(Self::create_device_map(
            detected_devices,
            LoadBalancingStrategy::CapacityBased,
        )?);

        let (event_tx, _event_rx) = mpsc::unbounded_channel();

        Ok(Self {
            device_map,
            gpu_miners: Vec::new(),
            metrics: Arc::new(RwLock::new(MultiGPUMetrics::default())),
            event_tx,
            work_queue: Arc::new(RwLock::new(Vec::new())),
            active: Arc::new(RwLock::new(false)),
        })
    }

    /// Detect CUDA devices (similar to mistral.rs Device::new_cuda)
    #[cfg(feature = "cuda-mining")]
    fn detect_cuda_devices() -> Result<Vec<GPUDeviceInfo>> {
        use cudarc::driver::safe::CudaDevice;

        let mut devices = Vec::new();
        let mut ordinal = 0;

        loop {
            match CudaDevice::new(ordinal) {
                Ok(device) => {
                    let name = device.name()?;
                    let memory_bytes = device.total_memory()?;
                    let memory_gb = memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0);

                    // Estimate hash rate based on GPU generation/memory
                    let estimated_hash_rate = Self::estimate_cuda_hash_rate(&name, memory_gb);

                    devices.push(GPUDeviceInfo {
                        ordinal,
                        name: name.clone(),
                        device_type: GPUDeviceType::CUDA,
                        compute_version: format!("CUDA {}.{}", device.compute_cap().0, device.compute_cap().1),
                        memory_gb,
                        estimated_hash_rate,
                        max_power_watts: Self::estimate_power_draw(&name),
                    });

                    ordinal += 1;
                }
                Err(_) => break,
            }
        }

        Ok(devices)
    }

    /// Detect OpenCL devices
    #[cfg(feature = "opencl-mining")]
    fn detect_opencl_devices() -> Result<Vec<GPUDeviceInfo>> {
        use opencl3::platform::get_platforms;

        let mut devices = Vec::new();
        let mut ordinal = 0;

        let platforms = get_platforms()?;
        for platform in platforms {
            if let Ok(platform_devices) = platform.get_devices(CL_DEVICE_TYPE_GPU) {
                for device in platform_devices {
                    if let Ok(name) = device.name() {
                        let memory_bytes = device.global_mem_size()? as u64;
                        let memory_gb = memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0);

                        let estimated_hash_rate = Self::estimate_opencl_hash_rate(&name, memory_gb);

                        devices.push(GPUDeviceInfo {
                            ordinal,
                            name: name.clone(),
                            device_type: GPUDeviceType::OpenCL,
                            compute_version: device.opencl_c_version()?,
                            memory_gb,
                            estimated_hash_rate,
                            max_power_watts: Self::estimate_power_draw(&name),
                        });

                        ordinal += 1;
                    }
                }
            }
        }

        Ok(devices)
    }

    /// Create device map with load balancing strategy (like mistral.rs DeviceMapSetting::into_mapper)
    fn create_device_map(
        devices: Vec<GPUDeviceInfo>,
        strategy: LoadBalancingStrategy,
    ) -> Result<MultiGPUDeviceMap> {
        let work_distribution: Vec<WorkDistribution> = match strategy {
            LoadBalancingStrategy::Equal => {
                let percentage = 1.0 / devices.len() as f64;
                devices
                    .iter()
                    .enumerate()
                    .map(|(idx, _)| WorkDistribution {
                        device_id: idx,
                        work_percentage: percentage,
                        priority: 1,
                    })
                    .collect()
            }
            LoadBalancingStrategy::CapacityBased => {
                let total_capacity: f64 = devices.iter().map(|d| d.estimated_hash_rate).sum();
                devices
                    .iter()
                    .enumerate()
                    .map(|(idx, device)| WorkDistribution {
                        device_id: idx,
                        work_percentage: device.estimated_hash_rate / total_capacity,
                        priority: 1,
                    })
                    .collect()
            }
            LoadBalancingStrategy::Dynamic => {
                // Start with capacity-based, will adjust dynamically
                let total_capacity: f64 = devices.iter().map(|d| d.estimated_hash_rate).sum();
                devices
                    .iter()
                    .enumerate()
                    .map(|(idx, device)| WorkDistribution {
                        device_id: idx,
                        work_percentage: device.estimated_hash_rate / total_capacity,
                        priority: 1,
                    })
                    .collect()
            }
            LoadBalancingStrategy::Manual(ref percentages) => {
                if percentages.len() != devices.len() {
                    anyhow::bail!("Manual percentages must match device count");
                }
                devices
                    .iter()
                    .enumerate()
                    .map(|(idx, _)| WorkDistribution {
                        device_id: idx,
                        work_percentage: percentages[idx],
                        priority: 1,
                    })
                    .collect()
            }
        };

        tracing::info!("📊 Work distribution:");
        for dist in &work_distribution {
            let device_id: usize = dist.device_id;
            tracing::info!(
                "  GPU {}: {:.1}% of work",
                device_id,
                dist.work_percentage * 100.0
            );
        }

        Ok(MultiGPUDeviceMap {
            devices,
            work_distribution,
            load_strategy: strategy.clone(),
        })
    }

    /// Distribute work across GPUs (like mistral.rs LayerDeviceMapper::map)
    pub async fn distribute_work(&self, work: WorkUnit) -> Result<()> {
        let device_map = self.device_map.clone();
        let work_queue = self.work_queue.clone();

        // Split work based on device distribution
        for dist in &device_map.work_distribution {
            let device_work = self.create_device_work(&work, dist)?;
            work_queue.write().push(device_work);
        }

        // Notify GPU miners of new work
        self.event_tx.send(MiningEvent::NewWork(work))?;

        Ok(())
    }

    /// Create work unit for specific device
    fn create_device_work(&self, work: &WorkUnit, dist: &WorkDistribution) -> Result<WorkUnit> {
        // Split nonce range based on work percentage
        let total_nonce_range = work.nonce_range.1 - work.nonce_range.0;
        let device_nonce_range = (total_nonce_range as f64 * dist.work_percentage) as u64;

        let device_nonce_start = work.nonce_range.0 + (dist.device_id as u64 * device_nonce_range);
        let device_nonce_end = device_nonce_start + device_nonce_range;

        Ok(WorkUnit {
            job_id: format!("{}-gpu{}", work.job_id, dist.device_id),
            previous_hash: work.previous_hash,
            merkle_root: work.merkle_root,
            timestamp: work.timestamp,
            difficulty_target: work.difficulty_target,
            nonce_range: (device_nonce_start, device_nonce_end),
            extra_data: work.extra_data.clone(),
        })
    }

    /// Get aggregated metrics from all GPUs
    pub fn get_metrics(&self) -> MultiGPUMetrics {
        self.metrics.read().clone()
    }

    /// Update metrics (called periodically by each GPU miner)
    pub fn update_device_metrics(&self, device_id: usize, stats: DeviceStats) {
        let mut metrics = self.metrics.write();

        // Update or insert device metrics
        if let Some(device_metrics) = metrics
            .device_metrics
            .iter_mut()
            .find(|m| m.device_id == device_id)
        {
            device_metrics.hash_rate = stats.hash_rate;
            device_metrics.temperature = stats.temperature;
            device_metrics.power_usage = stats.power_usage;
            device_metrics.utilization_percent = stats.utilization;
        } else {
            metrics.device_metrics.push(DeviceMetrics {
                device_id,
                hash_rate: stats.hash_rate,
                temperature: stats.temperature,
                power_usage: stats.power_usage,
                memory_used_gb: stats.memory_usage,
                utilization_percent: stats.utilization,
                shares_accepted: 0,
                shares_rejected: 0,
            });
        }

        // Recalculate total hash rate
        metrics.total_hash_rate = metrics.device_metrics.iter().map(|m| m.hash_rate).sum();
    }

    /// Estimate CUDA hash rate based on GPU model
    fn estimate_cuda_hash_rate(name: &str, memory_gb: f64) -> f64 {
        // Rough estimates based on common mining benchmarks
        let name_lower = name.to_lowercase();

        if name_lower.contains("4090") {
            120_000_000.0 // 120 MH/s for RTX 4090
        } else if name_lower.contains("4080") {
            100_000_000.0 // 100 MH/s for RTX 4080
        } else if name_lower.contains("3090") {
            110_000_000.0 // 110 MH/s for RTX 3090
        } else if name_lower.contains("3080") {
            95_000_000.0 // 95 MH/s for RTX 3080
        } else if name_lower.contains("3070") {
            60_000_000.0 // 60 MH/s for RTX 3070
        } else if name_lower.contains("a100") {
            150_000_000.0 // 150 MH/s for A100
        } else {
            // Fallback: estimate based on memory
            memory_gb * 5_000_000.0
        }
    }

    /// Estimate OpenCL hash rate
    fn estimate_opencl_hash_rate(name: &str, memory_gb: f64) -> f64 {
        // Similar estimates for AMD/Intel/other GPUs
        let name_lower = name.to_lowercase();

        if name_lower.contains("7900") {
            90_000_000.0 // AMD RX 7900 XT
        } else if name_lower.contains("6900") {
            80_000_000.0 // AMD RX 6900 XT
        } else if name_lower.contains("6800") {
            70_000_000.0 // AMD RX 6800 XT
        } else {
            memory_gb * 4_000_000.0
        }
    }

    /// Estimate power draw based on GPU model
    fn estimate_power_draw(name: &str) -> f64 {
        let name_lower = name.to_lowercase();

        if name_lower.contains("4090") {
            450.0
        } else if name_lower.contains("4080") {
            320.0
        } else if name_lower.contains("3090") {
            350.0
        } else if name_lower.contains("3080") {
            320.0
        } else if name_lower.contains("a100") {
            400.0
        } else {
            250.0 // Conservative default
        }
    }

    /// Get device information
    pub fn get_devices(&self) -> &[GPUDeviceInfo] {
        &self.device_map.devices
    }

    /// Start all GPU miners
    pub async fn start(&mut self) -> Result<()> {
        *self.active.write() = true;
        tracing::info!("🚀 Starting multi-GPU mining with {} devices", self.device_map.devices.len());
        Ok(())
    }

    /// Stop all GPU miners
    pub async fn stop(&mut self) -> Result<()> {
        *self.active.write() = false;
        tracing::info!("⏸️  Stopped multi-GPU mining");
        Ok(())
    }

    /// Check if actively mining
    pub fn is_active(&self) -> bool {
        *self.active.read()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_work_distribution_equal() {
        let devices = vec![
            GPUDeviceInfo {
                ordinal: 0,
                name: "GPU 0".to_string(),
                device_type: GPUDeviceType::CUDA,
                compute_version: "8.9".to_string(),
                memory_gb: 24.0,
                estimated_hash_rate: 100_000_000.0,
                max_power_watts: 350.0,
            },
            GPUDeviceInfo {
                ordinal: 1,
                name: "GPU 1".to_string(),
                device_type: GPUDeviceType::CUDA,
                compute_version: "8.9".to_string(),
                memory_gb: 24.0,
                estimated_hash_rate: 100_000_000.0,
                max_power_watts: 350.0,
            },
        ];

        let map = MultiGPUMiner::create_device_map(devices, LoadBalancingStrategy::Equal).unwrap();

        assert_eq!(map.work_distribution.len(), 2);
        assert!((map.work_distribution[0].work_percentage - 0.5).abs() < 0.001);
        assert!((map.work_distribution[1].work_percentage - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_work_distribution_capacity() {
        let devices = vec![
            GPUDeviceInfo {
                ordinal: 0,
                name: "Fast GPU".to_string(),
                device_type: GPUDeviceType::CUDA,
                compute_version: "8.9".to_string(),
                memory_gb: 24.0,
                estimated_hash_rate: 120_000_000.0,
                max_power_watts: 400.0,
            },
            GPUDeviceInfo {
                ordinal: 1,
                name: "Slow GPU".to_string(),
                device_type: GPUDeviceType::CUDA,
                compute_version: "7.5".to_string(),
                memory_gb: 8.0,
                estimated_hash_rate: 60_000_000.0,
                max_power_watts: 200.0,
            },
        ];

        let map = MultiGPUMiner::create_device_map(devices, LoadBalancingStrategy::CapacityBased).unwrap();

        // Fast GPU should get ~66.67% of work
        assert!((map.work_distribution[0].work_percentage - 0.6667).abs() < 0.01);
        // Slow GPU should get ~33.33% of work
        assert!((map.work_distribution[1].work_percentage - 0.3333).abs() < 0.01);
    }
}
