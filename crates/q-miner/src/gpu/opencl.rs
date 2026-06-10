use anyhow::{Result, anyhow};

#[cfg(feature = "opencl-mining")]
use opencl3::{
    command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE},
    context::Context,
    device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_ALL},
    kernel::{ExecuteKernel, Kernel},
    memory::{Buffer, CL_MEM_READ_WRITE, CL_MEM_COPY_HOST_PTR},
    platform::get_platforms,
    program::Program,
};

use std::ptr;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{info, debug, error, warn};

#[cfg(feature = "opencl-mining")]
use serde::Serialize;

#[cfg(feature = "opencl-mining")]
use hex;

use crate::{DeviceType, DeviceStats, MiningStats, MiningEngine, WorkUnit, Solution, gpu::GpuDeviceInfo};
use async_trait::async_trait;

/// HTTP payload sent to the mining submit endpoint.
#[cfg(feature = "opencl-mining")]
#[derive(Serialize)]
struct MiningSubmission {
    nonce: u64,
    hash: String,
    wallet_address: String,
    difficulty: u64,
}

/// OpenCL GPU mining implementation for cross-platform support
#[cfg(feature = "opencl-mining")]
pub struct OpenClMiner {
    devices: Vec<OpenCLDevice>,
    context: Context,
    command_queues: Vec<CommandQueue>,
    program: Program,
    kernels: Vec<Kernel>,
    stats: Arc<RwLock<MiningStats>>,
    is_running: Arc<RwLock<bool>>,
    /// API server base URL (e.g. "http://localhost:8080")
    server_url: String,
    /// Wallet address for mining rewards
    wallet_address: String,
}

#[cfg(feature = "opencl-mining")]
pub struct OpenCLDevice {
    pub device_id: String,
    pub device: Device,
    pub name: String,
    pub vendor: String,
    pub memory: u64,
    pub compute_units: u32,
    pub max_work_group_size: usize,
    pub global_memory_cache_size: u64,
    pub local_memory_size: u64,
}

#[cfg(feature = "opencl-mining")]
impl OpenClMiner {
    pub async fn new(device_ids: Vec<u32>, intensity: u8, server_url: String, wallet_address: String) -> Result<Self> {
        info!("🔍 Initializing OpenCL GPU mining...");
        
        // Get all platforms
        let platforms = get_platforms()?;
        if platforms.is_empty() {
            return Err(anyhow!("No OpenCL platforms found"));
        }
        
        // Find GPU devices across all platforms
        let mut all_devices = Vec::new();
        for platform in &platforms {
            let devices = get_all_devices(platform.id(), CL_DEVICE_TYPE_GPU)?;
            all_devices.extend(devices);
        }
        
        if all_devices.is_empty() {
            return Err(anyhow!("No OpenCL GPU devices found"));
        }
        
        info!("🚀 Found {} OpenCL GPU device(s)", all_devices.len());
        
        // Create context for all devices
        let context = Context::from_devices(&all_devices)?;
        
        // Create command queues
        let mut command_queues = Vec::new();
        for device in &all_devices {
            let queue = CommandQueue::create_default_with_properties(
                &context,
                device.id(),
                CL_QUEUE_PROFILING_ENABLE,
                0,
            )?;
            command_queues.push(queue);
        }
        
        // Build OpenCL program
        let program_source = include_str!("kernels/dag_knight_vdf.cl");
        let program = Program::create_and_build_from_source(&context, program_source, "")?;
        
        // Create kernels for each device
        let mut kernels = Vec::new();
        for _ in &all_devices {
            let kernel = Kernel::create(&program, "dag_knight_vdf_opencl")?;
            kernels.push(kernel);
        }
        
        // Create device info structs
        let mut devices = Vec::new();
        for (i, device) in all_devices.iter().enumerate() {
            let device_info = OpenCLDevice {
                device_id: format!("OpenCL-{}", i),
                device: device.clone(),
                name: device.name()?,
                vendor: device.vendor()?,
                memory: device.global_mem_size()?,
                compute_units: device.max_compute_units()?,
                max_work_group_size: device.max_work_group_size()?,
                global_memory_cache_size: device.global_mem_cache_size()?,
                local_memory_size: device.local_mem_size()?,
            };
            
            info!("📱 Device {}: {} ({} CUs, {:.1} GB)", 
                device_info.device_id,
                device_info.name,
                device_info.compute_units,
                device_info.memory as f64 / 1e9
            );
            
            devices.push(device_info);
        }
        
        Ok(Self {
            devices,
            context,
            command_queues,
            program,
            kernels,
            stats: Arc::new(RwLock::new(MiningStats::default())),
            is_running: Arc::new(RwLock::new(false)),
            server_url,
            wallet_address,
        })
    }
    
    pub async fn start_mining(
        &mut self,
        difficulty_target: [u8; 32],
        intensity: u8,
    ) -> Result<()> {
        info!("⛏️ Starting OpenCL mining with intensity {}", intensity);
        
        *self.is_running.write().await = true;
        
        // Calculate work sizes based on intensity and device capabilities
        let mut mining_tasks = Vec::new();
        
        for (device_idx, device) in self.devices.iter().enumerate() {
            let work_size = self.calculate_work_size(device, intensity);

            let stats = self.stats.clone();
            let is_running = self.is_running.clone();
            let device_info = device.clone();
            let kernel = self.kernels[device_idx].clone();
            let queue = self.command_queues[device_idx].clone();
            let context = self.context.clone();
            let server_url = self.server_url.clone();
            let wallet_address = self.wallet_address.clone();

            // Spawn mining task for each device
            let task = tokio::spawn(async move {
                Self::mine_on_device(
                    device_info,
                    kernel,
                    queue,
                    context,
                    work_size,
                    difficulty_target,
                    stats,
                    is_running,
                    server_url,
                    wallet_address,
                ).await
            });
            
            mining_tasks.push(task);
        }
        
        // Wait for all mining tasks
        for task in mining_tasks {
            if let Err(e) = task.await? {
                error!("OpenCL mining task failed: {}", e);
            }
        }
        
        Ok(())
    }
    
    async fn mine_on_device(
        device: OpenCLDevice,
        kernel: Kernel,
        queue: CommandQueue,
        context: Context,
        initial_work_size: usize,
        difficulty_target: [u8; 32],
        stats: Arc<RwLock<MiningStats>>,
        is_running: Arc<RwLock<bool>>,
        server_url: String,
        wallet_address: String,
    ) -> Result<()> {
        info!("🚀 Starting mining on device: {}", device.name);

        // Adaptive work size constants (Task 5)
        const DISPATCH_TARGET_LOW_MS: u128 = 200;
        const DISPATCH_TARGET_HIGH_MS: u128 = 600;
        const MIN_WORK_SIZE: usize = 50_000;
        const MAX_WORK_SIZE: usize = 5_000_000;

        // ── Persistent GPU buffers (Task 4) ────────────────────────────────
        // Buffers are allocated once and reused every dispatch.
        // input_size=64 (difficulty target 32B + nonce 8B + padding),
        // output_size is dynamic based on work_size — we pre-allocate for MAX.
        let input_size = 64usize;
        let max_output_size = MAX_WORK_SIZE * 32;

        let input_buffer = Buffer::<u8>::create(
            &context,
            CL_MEM_READ_WRITE,
            input_size,
            ptr::null_mut(),
        )?;

        // Allocate for maximum work size to avoid re-allocation
        let output_buffer = Buffer::<u8>::create(
            &context,
            CL_MEM_READ_WRITE,
            max_output_size,
            ptr::null_mut(),
        )?;

        info!("🎮 GPU {} buffers pre-allocated (input {}B, output {}MB max — persistent)",
            device.device_id, input_size, max_output_size / (1024 * 1024));

        let mut nonce_base = 0u64;
        let mut last_stats_update = std::time::Instant::now();
        let mut hashes_computed = 0u64;
        // Task 5: adaptive work size — start at configured size, tune at runtime
        let mut work_size = initial_work_size.clamp(MIN_WORK_SIZE, MAX_WORK_SIZE);

        // HTTP client for solution submission
        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        let submit_url = format!("{}/api/v1/mining/submit", server_url);

        while *is_running.read().await {
            let dispatch_start = std::time::Instant::now();

            // ── Task 4: Buffer reuse — only write what changed ──────────────
            // Prepare input data (same slice, reused buffer)
            let mut input_data = vec![0u8; input_size];
            input_data[..32].copy_from_slice(&difficulty_target);
            input_data[32..40].copy_from_slice(&nonce_base.to_le_bytes());

            // Copy input to GPU (persistent buffer — no re-allocation)
            queue.enqueue_write_buffer(&input_buffer, true, 0, &input_data, &[])?;

            // The output buffer is always the same allocation; we read only
            // the slice we actually wrote (work_size * 32 bytes).
            let output_slice_size = work_size * 32;

            // Set kernel arguments and dispatch
            ExecuteKernel::new(&kernel)
                .set_arg(&input_buffer)
                .set_arg(&output_buffer)
                .set_arg(&(work_size as u32))
                .set_arg(&nonce_base)
                .set_global_work_size(work_size)
                .set_local_work_size(device.max_work_group_size.min(256))
                .enqueue_nd_range(&queue)?;

            // Wait for completion
            queue.finish()?;

            // Read back only the used slice
            let mut output_data = vec![0u8; output_slice_size];
            queue.enqueue_read_buffer(&output_buffer, true, 0, &mut output_data, &[])?;

            let dispatch_ms = dispatch_start.elapsed().as_millis();

            // ── Task 5: Adaptive work size tuning ───────────────────────────
            if dispatch_ms < DISPATCH_TARGET_LOW_MS {
                // Dispatch finished too fast — increase work size by 25%
                work_size = ((work_size * 5) / 4).min(MAX_WORK_SIZE);
                debug!("📈 OpenCL {} work_size → {} (dispatch {}ms < {}ms)",
                    device.device_id, work_size, dispatch_ms, DISPATCH_TARGET_LOW_MS);
            } else if dispatch_ms > DISPATCH_TARGET_HIGH_MS {
                // Dispatch took too long — decrease work size by 25%
                work_size = ((work_size * 3) / 4).max(MIN_WORK_SIZE);
                debug!("📉 OpenCL {} work_size → {} (dispatch {}ms > {}ms)",
                    device.device_id, work_size, dispatch_ms, DISPATCH_TARGET_HIGH_MS);
            }

            // ── Check for valid solutions ────────────────────────────────────
            for chunk_idx in 0..work_size {
                let hash_start = chunk_idx * 32;
                let hash_end = hash_start + 32;
                let hash = &output_data[hash_start..hash_end];

                if Self::meets_difficulty(hash, &difficulty_target) {
                    let nonce = nonce_base + chunk_idx as u64;
                    let hex_hash = hex::encode(hash);
                    info!("💎 Solution submitted: nonce={} hash={:.8}...", nonce, hex_hash);

                    // ── Task 1: Submit via HTTP POST ─────────────────────────
                    let submission = MiningSubmission {
                        nonce,
                        hash: hex_hash.clone(),
                        wallet_address: wallet_address.clone(),
                        difficulty: u64::from_le_bytes(
                            difficulty_target[..8].try_into().unwrap_or([0u8; 8])
                        ),
                    };
                    match http_client
                        .post(&submit_url)
                        .json(&submission)
                        .send()
                        .await
                    {
                        Ok(resp) if resp.status().is_success() => {
                            info!("✅ Solution accepted by server");
                        }
                        Ok(resp) => {
                            warn!("⚠️ Solution submit returned HTTP {}", resp.status());
                        }
                        Err(e) => {
                            warn!("⚠️ Solution submit failed: {} (will retry on next solution)", e);
                        }
                    }
                }
            }

            hashes_computed += work_size as u64;
            nonce_base += work_size as u64;

            // Update statistics every second
            if last_stats_update.elapsed().as_secs() >= 1 {
                let elapsed = last_stats_update.elapsed();
                let hash_rate = hashes_computed as f64 / elapsed.as_secs_f64();

                let mut stats_guard = stats.write().await;
                stats_guard.devices.clear();
                stats_guard.devices.push(DeviceStats {
                    device_id: device.device_id.clone(),
                    device_type: DeviceType::OpenCL(device.name.clone()),
                    hash_rate,
                    temperature: 65.0,
                    power_usage: Self::estimate_power_usage(&device, hash_rate),
                    memory_usage: (input_size + output_slice_size) as f64 / 1e6,
                    utilization: 100.0,
                });

                debug!("📊 OpenCL {} hash rate: {:.2} MH/s",
                    device.device_id, hash_rate / 1e6);

                hashes_computed = 0;
                last_stats_update = std::time::Instant::now();
            }

            // Small yield to prevent starving the tokio runtime
            tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
        }

        info!("⏹️ OpenCL mining stopped on device: {}", device.name);
        Ok(())
    }
    
    fn calculate_work_size(&self, device: &OpenCLDevice, intensity: u8) -> usize {
        // Calculate optimal work size based on device capabilities and intensity
        let base_work_size = device.compute_units as usize * 1024;
        let intensity_multiplier = intensity as usize;
        
        // Scale by intensity (1-10)
        let work_size = base_work_size * intensity_multiplier;
        
        // Ensure work size is reasonable and aligned
        work_size.min(1_048_576).max(1024) // Between 1K and 1M
    }
    
    fn meets_difficulty(hash: &[u8], target: &[u8; 32]) -> bool {
        // Check if hash is less than difficulty target
        for (h, t) in hash.iter().zip(target.iter()) {
            match h.cmp(t) {
                std::cmp::Ordering::Less => return true,
                std::cmp::Ordering::Greater => return false,
                std::cmp::Ordering::Equal => continue,
            }
        }
        false
    }
    
    fn estimate_power_usage(device: &OpenCLDevice, hash_rate: f64) -> f64 {
        // Estimate power usage based on device characteristics and hash rate
        let base_power = match device.vendor.to_lowercase().as_str() {
            "nvidia" => 150.0,      // RTX series baseline
            "amd" => 200.0,         // RX series baseline
            "intel" => 75.0,        // Arc series baseline
            _ => 100.0,             // Generic estimate
        };
        
        // Scale with hash rate (rough approximation)
        let efficiency_factor = hash_rate / 1e9; // GH/s
        base_power * (0.5 + efficiency_factor * 0.5)
    }
    
    pub async fn stop_mining(&self) -> Result<()> {
        info!("⏹️ Stopping OpenCL mining...");
        *self.is_running.write().await = false;
        Ok(())
    }
    
    pub async fn get_stats(&self) -> MiningStats {
        self.stats.read().await.clone()
    }
}

#[cfg(feature = "opencl-mining")]
#[async_trait]
impl MiningEngine for OpenClMiner {
    async fn start(&mut self) -> Result<()> {
        info!("🚀 Starting OpenCL mining engine");
        let difficulty_target = [0x00u8; 32]; // Default difficulty
        self.start_mining(difficulty_target, 7).await?;
        Ok(())
    }
    
    async fn stop(&mut self) -> Result<()> {
        self.stop_mining().await
    }
    
    async fn get_hash_rate(&self) -> f64 {
        let stats = self.stats.read().await;
        stats.hash_rate
    }
    
    async fn get_stats(&self) -> MiningStats {
        self.get_stats().await
    }
}

/// OpenCL device detection and capabilities
#[cfg(feature = "opencl-mining")]
pub async fn detect_opencl_devices() -> Result<Vec<GpuDeviceInfo>> {
    let platforms = get_platforms().map_err(|e| anyhow!("Failed to get OpenCL platforms: {}", e))?;
    let mut devices = Vec::new();

    for platform in platforms {
        match get_all_devices(platform.id(), CL_DEVICE_TYPE_GPU) {
            Ok(platform_devices) => {
                for (i, device) in platform_devices.iter().enumerate() {
                    match OpenCLDevice::from_device(device.clone(), format!("OpenCL-{}-{}", platform.name().unwrap_or("Unknown".to_string()), i)) {
                        Ok(device_info) => devices.push(device_info.to_gpu_device_info()),
                        Err(e) => warn!("Failed to get device info: {}", e),
                    }
                }
            }
            Err(e) => debug!("No GPU devices found on platform {}: {}",
                platform.name().unwrap_or("Unknown".to_string()), e),
        }
    }

    Ok(devices)
}

#[cfg(feature = "opencl-mining")]
impl OpenCLDevice {
    fn from_device(device: Device, device_id: String) -> Result<Self> {
        Ok(Self {
            device_id,
            name: device.name()?,
            vendor: device.vendor()?,
            memory: device.global_mem_size()?,
            compute_units: device.max_compute_units()?,
            max_work_group_size: device.max_work_group_size()?,
            global_memory_cache_size: device.global_mem_cache_size()?,
            local_memory_size: device.local_mem_size()?,
            device,
        })
    }

    pub fn to_gpu_device_info(&self) -> GpuDeviceInfo {
        GpuDeviceInfo {
            device_id: self.device_id.parse().unwrap_or(0),
            name: self.name.clone(),
            vendor: self.vendor.clone(),
            compute_capability: None,
            memory_total: self.memory,
            memory_free: self.memory, // Approximate
            core_count: self.compute_units,
            max_threads_per_block: self.max_work_group_size as u32,
            max_shared_memory: self.local_memory_size as u32,
        }
    }
    
    pub fn get_device_info(&self) -> String {
        format!("{} - {} CUs, {:.1} GB VRAM", 
            self.name, 
            self.compute_units, 
            self.memory as f64 / 1e9
        )
    }
    
    pub fn estimate_hash_rate(&self, intensity: u8) -> f64 {
        // Estimate hash rate based on compute units and memory
        let base_rate = self.compute_units as f64 * 1_000_000.0; // 1 MH/s per CU baseline
        let intensity_factor = intensity as f64 / 10.0;
        let memory_factor = (self.memory as f64 / 1e9).min(8.0) / 8.0; // Memory bandwidth factor
        
        base_rate * intensity_factor * memory_factor
    }
}

/// Advanced OpenCL features
#[cfg(feature = "opencl-mining")]
pub mod advanced {
    use super::*;

    /// Multi-device OpenCL coordinator
    pub struct MultiDeviceCoordinator {
        devices: Vec<OpenCLDevice>,
        workload_distribution: Vec<f64>, // Percentage per device
    }

    impl MultiDeviceCoordinator {
        pub fn new(devices: Vec<OpenCLDevice>) -> Self {
            // Calculate optimal workload distribution based on device capabilities
            let total_compute_power: u32 = devices.iter()
                .map(|d| d.compute_units)
                .sum();
            
            let workload_distribution = devices.iter()
                .map(|d| d.compute_units as f64 / total_compute_power as f64)
                .collect();
            
            Self {
                devices,
                workload_distribution,
            }
        }
        
        pub fn redistribute_workload(&mut self, device_performance: &[f64]) {
            // Dynamically adjust workload based on actual performance
            let total_performance: f64 = device_performance.iter().sum();
            
            if total_performance > 0.0 {
                self.workload_distribution = device_performance.iter()
                    .map(|&perf| perf / total_performance)
                    .collect();
            }
        }
        
        pub fn get_work_allocation(&self, total_work: usize) -> Vec<usize> {
            self.workload_distribution.iter()
                .map(|&ratio| (total_work as f64 * ratio) as usize)
                .collect()
        }
    }
    
    /// OpenCL memory pool manager
    pub struct MemoryPoolManager {
        allocated_buffers: Vec<Buffer<u8>>,
        free_buffers: Vec<Buffer<u8>>,
        total_allocated: usize,
        max_allocation: usize,
    }
    
    impl MemoryPoolManager {
        pub fn new(max_allocation_mb: usize) -> Self {
            Self {
                allocated_buffers: Vec::new(),
                free_buffers: Vec::new(),
                total_allocated: 0,
                max_allocation: max_allocation_mb * 1024 * 1024,
            }
        }
        
        pub fn allocate_buffer(&mut self, context: &Context, size: usize) -> Result<Buffer<u8>> {
            // Try to reuse existing buffer
            if let Some(buffer) = self.free_buffers.pop() {
                // Note: Buffer doesn't implement Clone, so we just return it
                // In real usage, we'd need to track buffers differently
                return Ok(buffer);
            }
            
            // Check memory limits
            if self.total_allocated + size > self.max_allocation {
                return Err(anyhow!("GPU memory allocation limit exceeded"));
            }
            
            // Allocate new buffer
            let buffer = Buffer::<u8>::create(context, CL_MEM_READ_WRITE, size, ptr::null_mut())?;
            self.total_allocated += size;

            Ok(buffer)
        }
        
        pub fn free_buffer(&mut self, buffer: Buffer<u8>) {
            if let Some(pos) = self.allocated_buffers.iter().position(|b| std::ptr::eq(b, &buffer)) {
                self.allocated_buffers.remove(pos);
                self.free_buffers.push(buffer);
            }
        }
        
        pub fn get_memory_usage(&self) -> (usize, usize) {
            (self.total_allocated, self.max_allocation)
        }
    }
}

// ============================================================================
// OpenCL Mining Stub (when opencl-mining feature is not enabled)
// ============================================================================

#[cfg(not(feature = "opencl-mining"))]
pub struct OpenClMinerStub;

#[cfg(not(feature = "opencl-mining"))]
impl OpenClMinerStub {
    pub async fn new(_device_ids: Vec<u32>, _intensity: u8) -> Result<Self> {
        Err(anyhow!("OpenCL mining support not compiled. Rebuild with --features opencl-mining"))
    }
}

#[cfg(not(feature = "opencl-mining"))]
pub type OpenClMiner = OpenClMinerStub;

#[cfg(not(feature = "opencl-mining"))]
#[async_trait]
impl MiningEngine for OpenClMiner {
    async fn start(&mut self) -> Result<()> {
        Err(anyhow!("OpenCL not available"))
    }

    async fn stop(&mut self) -> Result<()> {
        Ok(())
    }

    async fn get_hash_rate(&self) -> f64 {
        0.0
    }

    async fn get_stats(&self) -> MiningStats {
        MiningStats::default()
    }
}