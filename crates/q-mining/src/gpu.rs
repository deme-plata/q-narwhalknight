//! GPU Mining Module for Q-NarwhalKnight
//!
//! This module implements high-performance GPU mining using OpenCL for SHA-3-256
//! proof-of-work computation. Designed to leverage massively parallel GPU architectures
//! for maximum hashrate.
//!
//! ## Features
//!
//! - **OpenCL Acceleration**: Cross-platform GPU support (NVIDIA, AMD, Intel)
//! - **Kernel Optimization**: Hand-tuned SHA-3-256 kernel for GPUs
//! - **Batch Processing**: Process millions of nonces per kernel dispatch
//! - **Memory Optimization**: Efficient buffer management and data transfer
//! - **Multi-GPU Support**: Use all available GPUs in parallel
//!
//! ## GPU Component in Hybrid Mining
//!
//! The GPU component handles the compute-bound SHA-3 PoW mining while the
//! CPU handles the sequential VDF proofs. This creates an optimal split:
//! - GPU: High parallelism SHA-3 hashing (thousands of threads)
//! - CPU: Sequential VDF computation (memory-bound, not parallelizable)

use anyhow::{anyhow, Result};
use sha3::{Digest, Sha3_256};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn, error};

#[cfg(feature = "gpu-mining")]
use opencl3::{
    command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE},
    context::Context,
    device::{Device, CL_DEVICE_TYPE_GPU},
    kernel::{ExecuteKernel, Kernel},
    memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY},
    program::Program,
    types::{cl_uchar, cl_uint, cl_ulong},
};

// ============================================================================
// SHA-3-256 OpenCL Kernel
// ============================================================================

/// OpenCL kernel source for SHA-3-256 mining
/// This is a highly optimized implementation for GPU execution
pub const SHA3_KERNEL_SOURCE: &str = r#"
// Keccak-f[1600] round constants
__constant ulong KECCAK_RC[24] = {
    0x0000000000000001UL, 0x0000000000008082UL, 0x800000000000808aUL,
    0x8000000080008000UL, 0x000000000000808bUL, 0x0000000080000001UL,
    0x8000000080008081UL, 0x8000000000008009UL, 0x000000000000008aUL,
    0x0000000000000088UL, 0x0000000080008009UL, 0x000000008000000aUL,
    0x000000008000808bUL, 0x800000000000008bUL, 0x8000000000008089UL,
    0x8000000000008003UL, 0x8000000000008002UL, 0x8000000000000080UL,
    0x000000000000800aUL, 0x800000008000000aUL, 0x8000000080008081UL,
    0x8000000000008080UL, 0x0000000080000001UL, 0x8000000080008008UL
};

// Rotation offsets
__constant uint KECCAK_ROT[24] = {
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14,
    27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44
};

// Pi lane indices
__constant uint KECCAK_PI[24] = {
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4,
    15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1
};

// Rotate left
inline ulong rotl64(ulong x, uint n) {
    return (x << n) | (x >> (64 - n));
}

// Keccak-f[1600] permutation
void keccak_f1600(__private ulong state[25]) {
    for (int round = 0; round < 24; round++) {
        // Theta
        ulong C[5], D[5];
        for (int x = 0; x < 5; x++) {
            C[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
        }
        for (int x = 0; x < 5; x++) {
            D[x] = C[(x + 4) % 5] ^ rotl64(C[(x + 1) % 5], 1);
        }
        for (int x = 0; x < 5; x++) {
            for (int y = 0; y < 5; y++) {
                state[x + 5 * y] ^= D[x];
            }
        }

        // Rho and Pi
        ulong t = state[1];
        for (int i = 0; i < 24; i++) {
            uint j = KECCAK_PI[i];
            ulong temp = state[j];
            state[j] = rotl64(t, KECCAK_ROT[i]);
            t = temp;
        }

        // Chi
        for (int y = 0; y < 5; y++) {
            ulong row[5];
            for (int x = 0; x < 5; x++) {
                row[x] = state[x + 5 * y];
            }
            for (int x = 0; x < 5; x++) {
                state[x + 5 * y] = row[x] ^ ((~row[(x + 1) % 5]) & row[(x + 2) % 5]);
            }
        }

        // Iota
        state[0] ^= KECCAK_RC[round];
    }
}

// SHA3-256 hash function
void sha3_256(__private ulong state[25], __private const uchar* input, uint input_len, __private uchar output[32]) {
    // Initialize state to zero
    for (int i = 0; i < 25; i++) {
        state[i] = 0;
    }

    // Absorb input (rate = 136 bytes for SHA3-256)
    const uint rate = 136;
    uint offset = 0;

    // Process full blocks
    while (input_len >= rate) {
        for (int i = 0; i < rate / 8; i++) {
            ulong lane = 0;
            for (int j = 0; j < 8; j++) {
                lane |= ((ulong)input[offset + i * 8 + j]) << (j * 8);
            }
            state[i] ^= lane;
        }
        keccak_f1600(state);
        offset += rate;
        input_len -= rate;
    }

    // Final block with padding
    uchar final_block[136];
    for (int i = 0; i < 136; i++) {
        final_block[i] = 0;
    }
    for (uint i = 0; i < input_len; i++) {
        final_block[i] = input[offset + i];
    }
    final_block[input_len] = 0x06;  // SHA3 domain separator
    final_block[rate - 1] |= 0x80;  // Final padding bit

    // XOR final block
    for (int i = 0; i < rate / 8; i++) {
        ulong lane = 0;
        for (int j = 0; j < 8; j++) {
            lane |= ((ulong)final_block[i * 8 + j]) << (j * 8);
        }
        state[i] ^= lane;
    }
    keccak_f1600(state);

    // Squeeze output (32 bytes)
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            output[i * 8 + j] = (uchar)(state[i] >> (j * 8));
        }
    }
}

// Check if hash meets difficulty target
bool meets_target(__private const uchar hash[32], __global const uchar* target) {
    for (int i = 0; i < 32; i++) {
        if (hash[i] < target[i]) return true;
        if (hash[i] > target[i]) return false;
    }
    return true;
}

// Main mining kernel
__kernel void sha3_mine(
    __global const uchar* header,      // Block header (without nonce)
    const uint header_len,              // Header length
    __global const uchar* target,       // Difficulty target (32 bytes)
    const ulong nonce_start,            // Starting nonce for this dispatch
    __global ulong* found_nonce,        // Output: found nonce (0 if none)
    __global uchar* found_hash,         // Output: found hash (32 bytes)
    __global uint* found_flag           // Output: 1 if solution found
) {
    uint gid = get_global_id(0);
    ulong nonce = nonce_start + gid;

    // Build input: header + nonce (little-endian)
    uchar input[256];
    for (uint i = 0; i < header_len; i++) {
        input[i] = header[i];
    }
    for (int i = 0; i < 8; i++) {
        input[header_len + i] = (uchar)(nonce >> (i * 8));
    }

    // Compute SHA3-256
    ulong state[25];
    uchar hash[32];
    sha3_256(state, input, header_len + 8, hash);

    // Check if meets target
    if (meets_target(hash, target)) {
        // Atomic to prevent race conditions
        uint old = atomic_cmpxchg(found_flag, 0, 1);
        if (old == 0) {
            *found_nonce = nonce;
            for (int i = 0; i < 32; i++) {
                found_hash[i] = hash[i];
            }
        }
    }
}
"#;

// ============================================================================
// GPU DEVICE INFO
// ============================================================================

/// Information about an available GPU device
#[derive(Debug, Clone)]
pub struct GPUDeviceInfo {
    /// Device index
    pub index: usize,

    /// Device name
    pub name: String,

    /// Vendor name
    pub vendor: String,

    /// Compute units (cores)
    pub compute_units: u32,

    /// Max work group size
    pub max_work_group_size: usize,

    /// Global memory size (bytes)
    pub global_memory: u64,

    /// Local memory size (bytes)
    pub local_memory: u64,

    /// Max clock frequency (MHz)
    pub max_clock_freq: u32,

    /// OpenCL version
    pub opencl_version: String,
}

// ============================================================================
// GPU MINER
// ============================================================================

/// GPU miner using OpenCL for SHA-3 mining
pub struct GPUMiner {
    /// Configuration
    config: GPUMinerConfig,

    /// Stop signal
    should_stop: Arc<AtomicBool>,

    /// Mining statistics
    stats: Arc<GPUMiningStats>,

    /// Available devices
    devices: Vec<GPUDeviceInfo>,

    #[cfg(feature = "gpu-mining")]
    /// OpenCL contexts (one per device)
    contexts: Vec<GPUContext>,
}

/// GPU miner configuration
#[derive(Debug, Clone)]
pub struct GPUMinerConfig {
    /// Use all available GPUs
    pub use_all_gpus: bool,

    /// Specific GPU indices to use
    pub gpu_indices: Vec<usize>,

    /// Work items per dispatch (global work size)
    pub work_size: usize,

    /// Local work group size
    pub local_work_size: usize,

    /// Mining intensity (1-100)
    pub intensity: u32,

    /// Stats reporting interval
    pub stats_interval: Duration,
}

impl Default for GPUMinerConfig {
    fn default() -> Self {
        Self {
            use_all_gpus: true,
            gpu_indices: vec![],
            work_size: 1 << 22, // ~4M work items
            local_work_size: 256,
            intensity: 80,
            stats_interval: Duration::from_secs(5),
        }
    }
}

/// GPU-specific mining context
#[cfg(feature = "gpu-mining")]
struct GPUContext {
    device: Device,
    context: Context,
    queue: CommandQueue,
    program: Program,
    kernel: Kernel,
}

/// GPU mining statistics
#[derive(Debug, Default)]
pub struct GPUMiningStats {
    /// Total hashes computed
    pub total_hashes: AtomicU64,

    /// Current hash rate (H/s)
    pub current_hashrate: AtomicU64,

    /// Peak hash rate
    pub peak_hashrate: AtomicU64,

    /// Blocks found
    pub blocks_found: AtomicU64,

    /// Kernel dispatches
    pub dispatches: AtomicU64,

    /// GPU temperature (if available)
    pub temperature: AtomicU64,

    /// GPU power draw (watts, if available)
    pub power_draw: AtomicU64,
}

/// GPU mining solution
#[derive(Debug, Clone)]
pub struct GPUSolution {
    pub nonce: u64,
    pub hash: [u8; 32],
    pub gpu_index: usize,
    pub hashes_computed: u64,
}

/// GPU mining job
#[derive(Clone)]
pub struct GPUMiningJob {
    pub header: Vec<u8>,
    pub target: [u8; 32],
    pub height: u64,
}

impl GPUMiner {
    /// Create new GPU miner
    pub fn new(config: GPUMinerConfig) -> Result<Self> {
        info!("🎮 Initializing GPU miner...");

        let devices = Self::enumerate_devices()?;

        if devices.is_empty() {
            return Err(anyhow!("No OpenCL-capable GPU devices found"));
        }

        info!("🎮 Found {} GPU device(s):", devices.len());
        for dev in &devices {
            info!(
                "  [{}] {} - {} CUs, {} MB VRAM",
                dev.index,
                dev.name,
                dev.compute_units,
                dev.global_memory / (1024 * 1024)
            );
        }

        #[cfg(feature = "gpu-mining")]
        let contexts = Self::initialize_contexts(&devices, &config)?;

        Ok(Self {
            config,
            should_stop: Arc::new(AtomicBool::new(false)),
            stats: Arc::new(GPUMiningStats::default()),
            devices,
            #[cfg(feature = "gpu-mining")]
            contexts,
        })
    }

    /// Enumerate available GPU devices
    fn enumerate_devices() -> Result<Vec<GPUDeviceInfo>> {
        #[cfg(feature = "gpu-mining")]
        {
            let platforms = opencl3::platform::get_platforms()?;
            let mut devices = Vec::new();
            let mut index = 0;

            for platform in platforms {
                // Get devices from this platform
                if let Ok(device_ids) = platform.get_devices(CL_DEVICE_TYPE_GPU) {
                    for device_id in device_ids {
                        let device = Device::new(device_id);

                        let info = GPUDeviceInfo {
                            index,
                            name: device.name().unwrap_or_default(),
                            vendor: device.vendor().unwrap_or_default(),
                            compute_units: device.max_compute_units().unwrap_or(0),
                            max_work_group_size: device.max_work_group_size().unwrap_or(256),
                            global_memory: device.global_mem_size().unwrap_or(0),
                            local_memory: device.local_mem_size().unwrap_or(0),
                            max_clock_freq: device.max_clock_frequency().unwrap_or(0),
                            opencl_version: device.opencl_c_version().unwrap_or_default(),
                        };
                        devices.push(info);
                        index += 1;
                    }
                }
            }

            Ok(devices)
        }

        #[cfg(not(feature = "gpu-mining"))]
        {
            warn!("GPU mining not enabled. Compile with --features gpu-mining");
            Ok(vec![])
        }
    }

    /// Initialize OpenCL contexts for selected devices
    #[cfg(feature = "gpu-mining")]
    fn initialize_contexts(devices: &[GPUDeviceInfo], config: &GPUMinerConfig) -> Result<Vec<GPUContext>> {
        let mut contexts = Vec::new();

        let indices: Vec<usize> = if config.use_all_gpus {
            (0..devices.len()).collect()
        } else {
            config.gpu_indices.clone()
        };

        for &idx in &indices {
            if idx >= devices.len() {
                warn!("GPU index {} out of range, skipping", idx);
                continue;
            }

            info!("🎮 Initializing GPU {} ({})", idx, devices[idx].name);

            // Get device
            let platforms = opencl3::platform::get_platforms()?;
            let mut target_device = None;

            'outer: for platform in &platforms {
                if let Ok(device_ids) = platform.get_devices(CL_DEVICE_TYPE_GPU) {
                    let mut current_idx = 0;
                    for device_id in device_ids {
                        let device = Device::new(device_id);
                        if current_idx == idx {
                            target_device = Some(device);
                            break 'outer;
                        }
                        current_idx += 1;
                    }
                }
            }

            let device = target_device.ok_or_else(|| anyhow!("Failed to find GPU {}", idx))?;

            // Create context
            let context = Context::from_device(&device)?;

            // Create command queue
            let queue = CommandQueue::create_default(&context, CL_QUEUE_PROFILING_ENABLE)?;

            // Build program
            let program = Program::create_and_build_from_source(&context, SHA3_KERNEL_SOURCE, "")
                .map_err(|e| anyhow!("Failed to build OpenCL program: {}", e))?;

            // Create kernel
            let kernel = Kernel::create(&program, "sha3_mine")?;

            contexts.push(GPUContext {
                device,
                context,
                queue,
                program,
                kernel,
            });
        }

        Ok(contexts)
    }

    /// Start mining with given job
    #[cfg(feature = "gpu-mining")]
    pub async fn mine(&self, job: GPUMiningJob) -> Result<Option<GPUSolution>> {
        if self.contexts.is_empty() {
            return Err(anyhow!("No GPU contexts initialized"));
        }

        let start_time = Instant::now();
        let mut nonce_offset = 0u64;
        let work_size = self.config.work_size;

        info!(
            "🎮 GPU mining started on {} device(s) | Work size: {}",
            self.contexts.len(),
            work_size
        );

        while !self.should_stop.load(Ordering::Relaxed) {
            // Dispatch to all GPUs
            for (gpu_idx, ctx) in self.contexts.iter().enumerate() {
                let result = self.dispatch_kernel(
                    ctx,
                    &job.header,
                    &job.target,
                    nonce_offset,
                    work_size,
                )?;

                self.stats.dispatches.fetch_add(1, Ordering::Relaxed);
                self.stats.total_hashes.fetch_add(work_size as u64, Ordering::Relaxed);

                if let Some((nonce, hash)) = result {
                    info!(
                        "🎉 GPU {} found solution! Nonce: {} | Hash: {}",
                        gpu_idx,
                        nonce,
                        hex::encode(&hash[..8])
                    );

                    self.stats.blocks_found.fetch_add(1, Ordering::Relaxed);

                    return Ok(Some(GPUSolution {
                        nonce,
                        hash,
                        gpu_index: gpu_idx,
                        hashes_computed: self.stats.total_hashes.load(Ordering::Relaxed),
                    }));
                }

                nonce_offset += work_size as u64;
            }

            // Update hashrate
            let elapsed = start_time.elapsed().as_secs_f64();
            if elapsed > 0.0 {
                let hashrate = (self.stats.total_hashes.load(Ordering::Relaxed) as f64 / elapsed) as u64;
                self.stats.current_hashrate.store(hashrate, Ordering::Relaxed);

                let peak = self.stats.peak_hashrate.load(Ordering::Relaxed);
                if hashrate > peak {
                    self.stats.peak_hashrate.store(hashrate, Ordering::Relaxed);
                }
            }

            // Yield to prevent blocking
            tokio::task::yield_now().await;
        }

        Ok(None)
    }

    /// Dispatch mining kernel to GPU
    #[cfg(feature = "gpu-mining")]
    fn dispatch_kernel(
        &self,
        ctx: &GPUContext,
        header: &[u8],
        target: &[u8; 32],
        nonce_start: u64,
        work_size: usize,
    ) -> Result<Option<(u64, [u8; 32])>> {
        // CL_TRUE for blocking operations
        const CL_TRUE: cl_uint = 1;

        // Create buffers
        let mut header_buffer = unsafe {
            Buffer::<cl_uchar>::create(&ctx.context, CL_MEM_READ_ONLY, header.len(), std::ptr::null_mut())?
        };
        let mut target_buffer = unsafe {
            Buffer::<cl_uchar>::create(&ctx.context, CL_MEM_READ_ONLY, 32, std::ptr::null_mut())?
        };
        let found_nonce_buffer = unsafe {
            Buffer::<cl_ulong>::create(&ctx.context, CL_MEM_WRITE_ONLY, 1, std::ptr::null_mut())?
        };
        let found_hash_buffer = unsafe {
            Buffer::<cl_uchar>::create(&ctx.context, CL_MEM_WRITE_ONLY, 32, std::ptr::null_mut())?
        };
        let mut found_flag_buffer = unsafe {
            Buffer::<cl_uint>::create(&ctx.context, CL_MEM_READ_WRITE, 1, std::ptr::null_mut())?
        };

        // Upload data
        unsafe {
            ctx.queue.enqueue_write_buffer(&mut header_buffer, CL_TRUE, 0, header, &[])?;
            ctx.queue.enqueue_write_buffer(&mut target_buffer, CL_TRUE, 0, target, &[])?;

            let zero_flag: [u32; 1] = [0];
            ctx.queue.enqueue_write_buffer(&mut found_flag_buffer, CL_TRUE, 0, &zero_flag, &[])?;
        }

        // Set kernel arguments
        let header_len = header.len() as u32;

        unsafe {
            ctx.kernel.set_arg(0, &header_buffer)?;
            ctx.kernel.set_arg(1, &header_len)?;
            ctx.kernel.set_arg(2, &target_buffer)?;
            ctx.kernel.set_arg(3, &nonce_start)?;
            ctx.kernel.set_arg(4, &found_nonce_buffer)?;
            ctx.kernel.set_arg(5, &found_hash_buffer)?;
            ctx.kernel.set_arg(6, &found_flag_buffer)?;
        }

        // Execute kernel
        let global_work_size = [work_size];
        let local_work_size = [self.config.local_work_size];

        unsafe {
            ctx.queue.enqueue_nd_range_kernel(
                ctx.kernel.get(),
                1,
                std::ptr::null(),
                global_work_size.as_ptr(),
                local_work_size.as_ptr(),
                &[],
            )?;
        }

        ctx.queue.finish()?;

        // Read results
        let mut found_flag: [u32; 1] = [0];
        unsafe {
            ctx.queue.enqueue_read_buffer(&found_flag_buffer, CL_TRUE, 0, &mut found_flag, &[])?;
        }

        if found_flag[0] != 0 {
            let mut found_nonce: [u64; 1] = [0];
            let mut found_hash: [u8; 32] = [0; 32];

            unsafe {
                ctx.queue.enqueue_read_buffer(&found_nonce_buffer, CL_TRUE, 0, &mut found_nonce, &[])?;
                ctx.queue.enqueue_read_buffer(&found_hash_buffer, CL_TRUE, 0, &mut found_hash, &[])?;
            }

            return Ok(Some((found_nonce[0], found_hash)));
        }

        Ok(None)
    }

    /// Fallback CPU mining when GPU is not available
    #[cfg(not(feature = "gpu-mining"))]
    pub async fn mine(&self, job: GPUMiningJob) -> Result<Option<GPUSolution>> {
        warn!("GPU mining not available, falling back to CPU");

        let mut nonce = 0u64;
        let start_time = Instant::now();

        while !self.should_stop.load(Ordering::Relaxed) {
            // Build input
            let mut input = job.header.clone();
            input.extend_from_slice(&nonce.to_le_bytes());

            // Hash
            let hash = Sha3_256::digest(&input);
            let mut hash_arr = [0u8; 32];
            hash_arr.copy_from_slice(&hash);

            // Check target
            if Self::meets_target(&hash_arr, &job.target) {
                return Ok(Some(GPUSolution {
                    nonce,
                    hash: hash_arr,
                    gpu_index: 0,
                    hashes_computed: nonce,
                }));
            }

            nonce += 1;
            self.stats.total_hashes.fetch_add(1, Ordering::Relaxed);

            // Update stats periodically
            if nonce % 1_000_000 == 0 {
                let elapsed = start_time.elapsed().as_secs_f64();
                if elapsed > 0.0 {
                    let hashrate = (nonce as f64 / elapsed) as u64;
                    self.stats.current_hashrate.store(hashrate, Ordering::Relaxed);
                }
                tokio::task::yield_now().await;
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
    }

    /// Get mining statistics
    pub fn get_stats(&self) -> GPUStatsSnapshot {
        GPUStatsSnapshot {
            total_hashes: self.stats.total_hashes.load(Ordering::Relaxed),
            current_hashrate: self.stats.current_hashrate.load(Ordering::Relaxed),
            peak_hashrate: self.stats.peak_hashrate.load(Ordering::Relaxed),
            blocks_found: self.stats.blocks_found.load(Ordering::Relaxed),
            dispatches: self.stats.dispatches.load(Ordering::Relaxed),
            num_devices: self.devices.len(),
        }
    }

    /// Get available devices
    pub fn get_devices(&self) -> &[GPUDeviceInfo] {
        &self.devices
    }

    /// Format hashrate for display
    pub fn format_hashrate(hashrate: u64) -> String {
        if hashrate >= 1_000_000_000_000 {
            format!("{:.2} TH/s", hashrate as f64 / 1_000_000_000_000.0)
        } else if hashrate >= 1_000_000_000 {
            format!("{:.2} GH/s", hashrate as f64 / 1_000_000_000.0)
        } else if hashrate >= 1_000_000 {
            format!("{:.2} MH/s", hashrate as f64 / 1_000_000.0)
        } else if hashrate >= 1_000 {
            format!("{:.2} KH/s", hashrate as f64 / 1_000.0)
        } else {
            format!("{} H/s", hashrate)
        }
    }
}

/// Snapshot of GPU mining statistics
#[derive(Debug, Clone)]
pub struct GPUStatsSnapshot {
    pub total_hashes: u64,
    pub current_hashrate: u64,
    pub peak_hashrate: u64,
    pub blocks_found: u64,
    pub dispatches: u64,
    pub num_devices: usize,
}

// ============================================================================
// OpenCL Context wrapper (for export)
// ============================================================================

/// OpenCL context wrapper for external use
pub struct OpenCLContext {
    #[cfg(feature = "gpu-mining")]
    inner: Context,
}

/// SHA-3 kernel wrapper for external use
pub struct SHA3Kernel {
    #[cfg(feature = "gpu-mining")]
    inner: Kernel,
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_meets_target() {
        let easy_target = [0xFF; 32];
        let hard_target = [0x00; 32];
        let hash = [0x0F; 32];

        assert!(GPUMiner::meets_target(&hash, &easy_target));
        assert!(!GPUMiner::meets_target(&hash, &hard_target));
    }

    #[test]
    fn test_format_hashrate() {
        assert_eq!(GPUMiner::format_hashrate(500), "500 H/s");
        assert_eq!(GPUMiner::format_hashrate(1_500_000), "1.50 MH/s");
        assert_eq!(GPUMiner::format_hashrate(1_500_000_000), "1.50 GH/s");
    }

    #[tokio::test]
    async fn test_gpu_enumeration() {
        let devices = GPUMiner::enumerate_devices();
        assert!(devices.is_ok());
        // May be empty if no GPU is available
    }
}
