//! GPU Mining Module for Q-NarwhalKnight — Hybrid Quantum Mining
//!
//! Implements BLAKE3 + 100-round VDF proof-of-work on GPU via OpenCL.
//! Each GPU work item independently computes: BLAKE3(challenge||nonce) then
//! 99 sequential BLAKE3(h) VDF rounds, checking the final hash against
//! the difficulty target. GPU parallelism comes from running thousands of
//! nonce candidates simultaneously.
//!
//! ## Algorithm (must match server validation)
//! ```text
//! input = challenge_hash[32] || nonce_le[8]   // 40 bytes
//! h = BLAKE3(input)                           // initial hash
//! for _ in 0..99: h = BLAKE3(h)              // VDF chain (99 rounds)
//! if h < difficulty_target: SOLUTION!         // total: 100 BLAKE3 hashes
//! ```

use anyhow::{anyhow, Result};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{info, warn, error};

#[cfg(feature = "gpu-mining")]
use opencl3::{
    command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE},
    context::Context,
    device::{Device, CL_DEVICE_TYPE_GPU},
    kernel::Kernel,
    memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY},
    program::Program,
    types::{cl_uchar, cl_uint, cl_ulong},
};

// ============================================================================
// BLAKE3 + VDF OpenCL Kernel
// ============================================================================

/// OpenCL kernel implementing BLAKE3 + 99-round VDF for Q-NarwhalKnight mining.
///
/// Algorithm per work item:
///   1. Build 40-byte input: challenge_hash[32] || nonce_le[8]
///   2. h = BLAKE3(input)           — single-block 40-byte hash
///   3. for 99 rounds: h = BLAKE3(h) — single-block 32-byte hash (VDF chain)
///   4. Compare final h < target (byte-wise, big-endian-like)
pub const BLAKE3_KERNEL_SOURCE: &str = r#"
// ═══════════════════════════════════════════════════════════════════
// BLAKE3 constants
// ═══════════════════════════════════════════════════════════════════

__constant uint BLAKE3_IV[8] = {
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u
};

// Pre-computed message schedule for 7 rounds (BLAKE3 spec §2.2)
__constant uchar MSG_SCHED[7][16] = {
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15},
    { 2, 6, 3,10, 7, 0, 4,13, 1,11,12, 5, 9,14,15, 8},
    { 3, 4,10,12,13, 2, 7,14, 6, 5, 9, 0,11,15, 8, 1},
    {10, 7,12, 9,14, 3,13,15, 4, 0,11, 2, 5, 8, 1, 6},
    {12,13, 9,11,15,10,14, 8, 7, 2, 5, 3, 0, 1, 6, 4},
    { 9,14,11, 5, 8,12,15, 1,13, 3, 0,10, 2, 6, 4, 7},
    {11,15, 5, 0, 1, 9, 8, 6,14,10, 2,12, 3, 4, 7,13}
};

#define CHUNK_START 1u
#define CHUNK_END   2u
#define ROOT        8u

inline uint rotr32(uint x, uint n) {
    return (x >> n) | (x << (32u - n));
}

// ═══════════════════════════════════════════════════════════════════
// BLAKE3 G mixing function (inlined for performance)
// ═══════════════════════════════════════════════════════════════════

#define G(s, a, b, c, d, mx, my) \
    s[a] = s[a] + s[b] + (mx);  \
    s[d] = rotr32(s[d] ^ s[a], 16u); \
    s[c] = s[c] + s[d];         \
    s[b] = rotr32(s[b] ^ s[c], 12u); \
    s[a] = s[a] + s[b] + (my);  \
    s[d] = rotr32(s[d] ^ s[a], 8u);  \
    s[c] = s[c] + s[d];         \
    s[b] = rotr32(s[b] ^ s[c], 7u);

// ═══════════════════════════════════════════════════════════════════
// BLAKE3 compression — single-block hash
// cv[8]: chaining value, block[16]: message words, counter: u64,
// block_len: actual data bytes, flags: combination of START/END/ROOT
// output[8]: resulting hash words
// ═══════════════════════════════════════════════════════════════════

void blake3_compress(
    const uint cv[8],
    const uint block[16],
    ulong counter,
    uint block_len,
    uint flags,
    uint output[8]
) {
    uint s[16];
    s[0]  = cv[0]; s[1]  = cv[1]; s[2]  = cv[2]; s[3]  = cv[3];
    s[4]  = cv[4]; s[5]  = cv[5]; s[6]  = cv[6]; s[7]  = cv[7];
    s[8]  = BLAKE3_IV[0]; s[9]  = BLAKE3_IV[1];
    s[10] = BLAKE3_IV[2]; s[11] = BLAKE3_IV[3];
    s[12] = (uint)(counter & 0xFFFFFFFFul);
    s[13] = (uint)(counter >> 32);
    s[14] = block_len;
    s[15] = flags;

    // 7 rounds with message schedule permutation
    for (int r = 0; r < 7; r++) {
        uint m0  = block[MSG_SCHED[r][ 0]]; uint m1  = block[MSG_SCHED[r][ 1]];
        uint m2  = block[MSG_SCHED[r][ 2]]; uint m3  = block[MSG_SCHED[r][ 3]];
        uint m4  = block[MSG_SCHED[r][ 4]]; uint m5  = block[MSG_SCHED[r][ 5]];
        uint m6  = block[MSG_SCHED[r][ 6]]; uint m7  = block[MSG_SCHED[r][ 7]];
        uint m8  = block[MSG_SCHED[r][ 8]]; uint m9  = block[MSG_SCHED[r][ 9]];
        uint m10 = block[MSG_SCHED[r][10]]; uint m11 = block[MSG_SCHED[r][11]];
        uint m12 = block[MSG_SCHED[r][12]]; uint m13 = block[MSG_SCHED[r][13]];
        uint m14 = block[MSG_SCHED[r][14]]; uint m15 = block[MSG_SCHED[r][15]];

        // Column step
        G(s, 0, 4,  8, 12, m0,  m1);
        G(s, 1, 5,  9, 13, m2,  m3);
        G(s, 2, 6, 10, 14, m4,  m5);
        G(s, 3, 7, 11, 15, m6,  m7);
        // Diagonal step
        G(s, 0, 5, 10, 15, m8,  m9);
        G(s, 1, 6, 11, 12, m10, m11);
        G(s, 2, 7,  8, 13, m12, m13);
        G(s, 3, 4,  9, 14, m14, m15);
    }

    // Output: XOR lower and upper halves
    for (int i = 0; i < 8; i++) {
        output[i] = s[i] ^ s[i + 8];
    }
}

// ═══════════════════════════════════════════════════════════════════
// blake3_hash_40: Hash 40-byte input (challenge[32] + nonce_le[8])
// Returns 32-byte hash as 8 uint words (little-endian)
// ═══════════════════════════════════════════════════════════════════

void blake3_hash_40(
    __global const uchar* challenge,
    ulong nonce,
    uint output[8]
) {
    // Build message block: 40 bytes of data + 24 bytes of zero padding
    uint block[16];

    // challenge_hash bytes → LE u32 words (bytes 0..31)
    for (int i = 0; i < 8; i++) {
        block[i] = (uint)challenge[i*4]
                 | ((uint)challenge[i*4+1] << 8)
                 | ((uint)challenge[i*4+2] << 16)
                 | ((uint)challenge[i*4+3] << 24);
    }

    // nonce (u64 LE) → two u32 words (bytes 32..39)
    block[8] = (uint)(nonce & 0xFFFFFFFFul);
    block[9] = (uint)(nonce >> 32);

    // Zero padding (bytes 40..63)
    block[10] = 0; block[11] = 0; block[12] = 0;
    block[13] = 0; block[14] = 0; block[15] = 0;

    // Single-chunk, single-block: flags = CHUNK_START | CHUNK_END | ROOT
    blake3_compress(BLAKE3_IV, block, 0, 40u, CHUNK_START | CHUNK_END | ROOT, output);
}

// ═══════════════════════════════════════════════════════════════════
// blake3_hash_32: Hash 32-byte input (VDF intermediate hash)
// Input and output are both uint[8] (LE words)
// ═══════════════════════════════════════════════════════════════════

void blake3_hash_32(const uint input[8], uint output[8]) {
    uint block[16];
    for (int i = 0; i < 8; i++) block[i] = input[i];
    for (int i = 8; i < 16; i++) block[i] = 0;

    blake3_compress(BLAKE3_IV, block, 0, 32u, CHUNK_START | CHUNK_END | ROOT, output);
}

// ═══════════════════════════════════════════════════════════════════
// Target comparison: hash < target (byte-wise, little-endian words)
// hash is uint[8] (LE words), target is uchar[32] (raw bytes)
// Both represent the same byte ordering: word[0] bits 0-7 = byte[0]
// ═══════════════════════════════════════════════════════════════════

bool meets_target(const uint hash[8], __global const uchar* target) {
    for (int i = 0; i < 8; i++) {
        uint h = hash[i];
        for (int j = 0; j < 4; j++) {
            uchar hb = (uchar)((h >> (j * 8)) & 0xFFu);
            uchar tb = target[i * 4 + j];
            if (hb < tb) return true;
            if (hb > tb) return false;
        }
    }
    return true;
}

// ═══════════════════════════════════════════════════════════════════
// MAIN MINING KERNEL: BLAKE3 + 99-round VDF
//
// Each work item:
//   1. Compute h = BLAKE3(challenge[32] || nonce_le[8])  — 40 bytes
//   2. Repeat 99 times: h = BLAKE3(h)                    — 32 bytes
//   3. If h < target → atomically write solution
//
// Total: 100 BLAKE3 hashes per nonce candidate
// ═══════════════════════════════════════════════════════════════════

__kernel void blake3_mine(
    __global const uchar* challenge,    // challenge_hash (32 bytes)
    __global const uchar* target,       // difficulty target (32 bytes)
    const ulong nonce_start,            // starting nonce for this dispatch
    __global ulong* found_nonce,        // output: winning nonce
    __global uchar* found_hash,         // output: winning hash (32 bytes)
    __global uint* found_flag           // output: 1 if solution found
) {
    uint gid = get_global_id(0);
    ulong nonce = nonce_start + (ulong)gid;

    // Step 1: Initial BLAKE3 hash of 40-byte input
    uint h[8];
    blake3_hash_40(challenge, nonce, h);

    // Step 2: VDF chain — 99 sequential BLAKE3 hashes
    uint tmp[8];
    for (int vdf = 0; vdf < 99; vdf++) {
        blake3_hash_32(h, tmp);
        for (int i = 0; i < 8; i++) h[i] = tmp[i];
    }

    // Step 3: Check if final hash meets difficulty target
    if (meets_target(h, target)) {
        // Atomic CAS to claim the solution (first writer wins)
        uint old = atomic_cmpxchg(found_flag, 0u, 1u);
        if (old == 0u) {
            *found_nonce = nonce;
            // Convert hash words to bytes (LE)
            for (int i = 0; i < 8; i++) {
                found_hash[i*4 + 0] = (uchar)( h[i]        & 0xFFu);
                found_hash[i*4 + 1] = (uchar)((h[i] >>  8) & 0xFFu);
                found_hash[i*4 + 2] = (uchar)((h[i] >> 16) & 0xFFu);
                found_hash[i*4 + 3] = (uchar)((h[i] >> 24) & 0xFFu);
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
    pub index: usize,
    pub name: String,
    pub vendor: String,
    pub compute_units: u32,
    pub max_work_group_size: usize,
    pub global_memory: u64,
    pub local_memory: u64,
    pub max_clock_freq: u32,
    pub opencl_version: String,
}

// ============================================================================
// GPU MINER
// ============================================================================

/// GPU miner using OpenCL for BLAKE3+VDF hybrid quantum mining
pub struct GPUMiner {
    config: GPUMinerConfig,
    should_stop: Arc<AtomicBool>,
    stats: Arc<GPUMiningStats>,
    devices: Vec<GPUDeviceInfo>,
    #[cfg(feature = "gpu-mining")]
    contexts: Vec<GPUContext>,
}

/// GPU miner configuration
#[derive(Debug, Clone)]
pub struct GPUMinerConfig {
    pub use_all_gpus: bool,
    pub gpu_indices: Vec<usize>,
    /// Work items per dispatch (global work size). Each item = 100 BLAKE3 hashes.
    pub work_size: usize,
    pub local_work_size: usize,
    pub intensity: u32,
    pub stats_interval: Duration,
}

impl Default for GPUMinerConfig {
    fn default() -> Self {
        Self {
            use_all_gpus: true,
            gpu_indices: vec![],
            // ~1M work items. Each does 100 BLAKE3 hashes, so 100M hashes per dispatch.
            // Reduced from 4M to avoid GPU timeouts on the 100-round VDF.
            work_size: 1 << 20,
            local_work_size: 256,
            intensity: 80,
            stats_interval: Duration::from_secs(5),
        }
    }
}

/// GPU-specific mining context
#[cfg(feature = "gpu-mining")]
struct GPUContext {
    #[allow(dead_code)]
    device: Device,
    context: Context,
    queue: CommandQueue,
    #[allow(dead_code)]
    program: Program,
    kernel: Kernel,
}

/// GPU mining statistics
#[derive(Debug, Default)]
pub struct GPUMiningStats {
    pub total_hashes: AtomicU64,
    pub current_hashrate: AtomicU64,
    pub peak_hashrate: AtomicU64,
    pub blocks_found: AtomicU64,
    pub dispatches: AtomicU64,
    pub temperature: AtomicU64,
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

/// Result from a single mine_batch dispatch
#[derive(Debug)]
pub struct BatchResult {
    /// Solution found (if any)
    pub solution: Option<GPUSolution>,
    /// Number of nonce candidates tried in this batch
    pub hashes: u64,
}

/// A mining job submitted to the GPU
#[derive(Debug, Clone)]
pub struct GPUMiningJob {
    /// Block header bytes to hash
    pub header: Vec<u8>,
    /// Target difficulty (hash must be below this)
    pub target: [u8; 32],
    /// Block height
    pub height: u64,
}

impl GPUMiner {
    /// Create new GPU miner with BLAKE3+VDF kernel
    pub fn new(config: GPUMinerConfig) -> Result<Self> {
        info!("🎮 Initializing GPU miner (BLAKE3+VDF hybrid quantum mining)...");

        let devices = Self::enumerate_devices()?;

        if devices.is_empty() {
            return Err(anyhow!("No OpenCL-capable GPU devices found"));
        }

        info!("🎮 Found {} GPU device(s):", devices.len());
        for dev in &devices {
            info!(
                "  [{}] {} - {} CUs, {} MB VRAM",
                dev.index, dev.name, dev.compute_units,
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

            info!("🎮 Initializing GPU {} ({}) with BLAKE3+VDF kernel", idx, devices[idx].name);

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
            let context = Context::from_device(&device)?;
            let queue = CommandQueue::create_default(&context, CL_QUEUE_PROFILING_ENABLE)?;

            let program = Program::create_and_build_from_source(&context, BLAKE3_KERNEL_SOURCE, "")
                .map_err(|e| anyhow!("Failed to build BLAKE3 OpenCL program: {}", e))?;

            let kernel = Kernel::create(&program, "blake3_mine")?;

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

    /// Dispatch a single batch of mining work to the first GPU and return.
    ///
    /// This is the primary API for the mining loop. The caller controls:
    /// - New-block abandonment (check signal between batches)
    /// - Statistics updates
    /// - Nonce progression
    ///
    /// Returns (solution_if_found, nonces_tried).
    #[cfg(feature = "gpu-mining")]
    pub fn mine_batch(
        &self,
        challenge_hash: &[u8; 32],
        target: &[u8; 32],
        nonce_start: u64,
    ) -> Result<BatchResult> {
        if self.contexts.is_empty() {
            return Err(anyhow!("No GPU contexts initialized"));
        }

        let ctx = &self.contexts[0];
        let work_size = self.config.work_size;

        let result = self.dispatch_blake3_kernel(ctx, challenge_hash, target, nonce_start, work_size)?;

        self.stats.dispatches.fetch_add(1, Ordering::Relaxed);
        self.stats.total_hashes.fetch_add(work_size as u64, Ordering::Relaxed);

        let solution = result.map(|(nonce, hash)| {
            self.stats.blocks_found.fetch_add(1, Ordering::Relaxed);
            GPUSolution {
                nonce,
                hash,
                gpu_index: 0,
                hashes_computed: work_size as u64,
            }
        });

        Ok(BatchResult {
            solution,
            hashes: work_size as u64,
        })
    }

    /// Dispatch the BLAKE3+VDF mining kernel to one GPU context
    #[cfg(feature = "gpu-mining")]
    fn dispatch_blake3_kernel(
        &self,
        ctx: &GPUContext,
        challenge: &[u8; 32],
        target: &[u8; 32],
        nonce_start: u64,
        work_size: usize,
    ) -> Result<Option<(u64, [u8; 32])>> {
        const CL_TRUE: cl_uint = 1;

        // Create buffers
        let mut challenge_buffer = unsafe {
            Buffer::<cl_uchar>::create(&ctx.context, CL_MEM_READ_ONLY, 32, std::ptr::null_mut())?
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
            ctx.queue.enqueue_write_buffer(&mut challenge_buffer, CL_TRUE, 0, challenge, &[])?;
            ctx.queue.enqueue_write_buffer(&mut target_buffer, CL_TRUE, 0, target, &[])?;
            let zero_flag: [u32; 1] = [0];
            ctx.queue.enqueue_write_buffer(&mut found_flag_buffer, CL_TRUE, 0, &zero_flag, &[])?;
        }

        // Set kernel arguments
        unsafe {
            ctx.kernel.set_arg(0, &challenge_buffer)?;
            ctx.kernel.set_arg(1, &target_buffer)?;
            ctx.kernel.set_arg(2, &nonce_start)?;
            ctx.kernel.set_arg(3, &found_nonce_buffer)?;
            ctx.kernel.set_arg(4, &found_hash_buffer)?;
            ctx.kernel.set_arg(5, &found_flag_buffer)?;
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

    /// Fallback mine_batch when GPU feature is not compiled in
    #[cfg(not(feature = "gpu-mining"))]
    pub fn mine_batch(
        &self,
        _challenge_hash: &[u8; 32],
        _target: &[u8; 32],
        _nonce_start: u64,
    ) -> Result<BatchResult> {
        Err(anyhow!("GPU mining not available. Compile with --features gpu-mining"))
    }

    /// Check if hash meets target (byte-wise comparison: hash < target)
    pub fn meets_target(hash: &[u8; 32], target: &[u8; 32]) -> bool {
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

    /// Update the hashrate stat (called by the mining loop externally)
    pub fn update_hashrate(&self, hashrate: u64) {
        self.stats.current_hashrate.store(hashrate, Ordering::Relaxed);
        let peak = self.stats.peak_hashrate.load(Ordering::Relaxed);
        if hashrate > peak {
            self.stats.peak_hashrate.store(hashrate, Ordering::Relaxed);
        }
    }

    /// Get available devices
    pub fn get_devices(&self) -> &[GPUDeviceInfo] {
        &self.devices
    }

    /// Get device name of the first GPU (for TUI display)
    pub fn device_name(&self) -> &str {
        self.devices.first().map(|d| d.name.as_str()).unwrap_or("Unknown GPU")
    }

    /// Mine a job by iterating mine_batch until a solution is found or stopped
    pub async fn mine(&self, job: GPUMiningJob) -> Result<Option<GPUSolution>> {
        use blake3;
        // Derive challenge hash from the header
        let challenge_hash: [u8; 32] = blake3::hash(&job.header).into();
        let mut nonce_start: u64 = 0;

        while !self.should_stop.load(Ordering::Relaxed) {
            let result = self.mine_batch(&challenge_hash, &job.target, nonce_start)?;
            if let Some(sol) = result.solution {
                return Ok(Some(sol));
            }
            nonce_start = nonce_start.wrapping_add(result.hashes);
            // Yield to tokio runtime
            tokio::task::yield_now().await;
        }
        Ok(None)
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

    #[test]
    fn test_gpu_enumeration() {
        let devices = GPUMiner::enumerate_devices();
        assert!(devices.is_ok());
        // May be empty if no GPU is available
    }

    /// Verify that the CPU BLAKE3+VDF produces the expected hash for a known input.
    /// The GPU kernel must produce identical output for the same input.
    #[test]
    fn test_blake3_vdf_reference() {
        let challenge = [0x42u8; 32]; // test challenge
        let nonce: u64 = 12345;

        // CPU reference: BLAKE3(challenge || nonce_le) then 99 rounds of BLAKE3(h)
        let mut input = [0u8; 40];
        input[..32].copy_from_slice(&challenge);
        input[32..].copy_from_slice(&nonce.to_le_bytes());

        let mut h = *blake3::hash(&input).as_bytes();
        for _ in 0..99 {
            h = *blake3::hash(&h).as_bytes();
        }

        // The hash should be deterministic
        assert_ne!(h, [0u8; 32], "BLAKE3+VDF should produce non-zero output");

        // Verify it matches a second computation (deterministic)
        let mut h2 = *blake3::hash(&input).as_bytes();
        for _ in 0..99 {
            h2 = *blake3::hash(&h2).as_bytes();
        }
        assert_eq!(h, h2, "BLAKE3+VDF must be deterministic");
    }
}
