//! GPU Mining Module for Slint Wallet — OpenCL BLAKE3+VDF
//!
//! Copied and adapted from `crates/q-mining/src/gpu.rs`. The OpenCL kernel
//! (`BLAKE3_KERNEL_SOURCE`) is identical to the standalone miner — solutions
//! from this module are accepted by the server with no difference.
//!
//! Differences from q-mining:
//! - No CUDA (OpenCL covers NVIDIA + AMD + Intel)
//! - std::thread (not tokio tasks) to match wallet's existing miner model
//! - Shares challenge RwLock + new_block_signal with CPU miner threads
//! - Uses eprintln! instead of tracing

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::api_client::ApiClient;
use crate::miner::{MinerState, SharedChallenge};
use crate::models::MiningSubmission;

// ============================================================================
// BLAKE3 + VDF OpenCL Kernel (MUST match server validation exactly)
// Copied verbatim from crates/q-mining/src/gpu.rs
// ============================================================================

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
// blake3_hash_40: Hash 40-byte input (challenge_words[8] + nonce_le[8])
// ═══════════════════════════════════════════════════════════════════

void blake3_hash_40(
    __constant uint* challenge_words,
    ulong nonce,
    uint output[8]
) {
    uint block[16];

    block[0] = challenge_words[0]; block[1] = challenge_words[1];
    block[2] = challenge_words[2]; block[3] = challenge_words[3];
    block[4] = challenge_words[4]; block[5] = challenge_words[5];
    block[6] = challenge_words[6]; block[7] = challenge_words[7];

    block[8] = (uint)(nonce & 0xFFFFFFFFul);
    block[9] = (uint)(nonce >> 32);

    block[10] = 0; block[11] = 0; block[12] = 0;
    block[13] = 0; block[14] = 0; block[15] = 0;

    uint iv[8];
    for (int i = 0; i < 8; i++) iv[i] = BLAKE3_IV[i];

    blake3_compress(iv, block, 0, 40u, CHUNK_START | CHUNK_END | ROOT, output);
}

// ═══════════════════════════════════════════════════════════════════
// blake3_hash_32: Hash 32-byte input (VDF intermediate hash)
// ═══════════════════════════════════════════════════════════════════

void blake3_hash_32(const uint input[8], uint output[8]) {
    uint block[16];
    for (int i = 0; i < 8; i++) block[i] = input[i];
    for (int i = 8; i < 16; i++) block[i] = 0;

    uint iv[8];
    for (int i = 0; i < 8; i++) iv[i] = BLAKE3_IV[i];

    blake3_compress(iv, block, 0, 32u, CHUNK_START | CHUNK_END | ROOT, output);
}

// ═══════════════════════════════════════════════════════════════════
// Target comparison
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
// ═══════════════════════════════════════════════════════════════════

__attribute__((reqd_work_group_size(256,1,1)))
__kernel void blake3_mine(
    __constant uint* challenge_words,
    __global const uchar* target,
    const ulong nonce_start,
    __global ulong* found_nonce,
    __global uchar* found_hash,
    __global uint* found_flag
) {
    uint gid = get_global_id(0);
    ulong nonce = nonce_start + (ulong)gid;

    uint h[8];
    blake3_hash_40(challenge_words, nonce, h);

    uint tmp[8];
    #pragma unroll 3
    for (int vdf = 0; vdf < 99; vdf++) {
        blake3_hash_32(h, tmp);
        for (int i = 0; i < 8; i++) h[i] = tmp[i];
    }

    if (meets_target(h, target)) {
        uint old = atomic_cmpxchg(found_flag, 0u, 1u);
        if (old == 0u) {
            *found_nonce = nonce;
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
// GPU Device Info
// ============================================================================

/// Information about a detected GPU device
#[derive(Debug, Clone)]
pub struct GpuDevice {
    pub index: usize,
    pub name: String,
    pub vendor: String,
    pub memory_mb: u64,
    pub compute_units: u32,
    pub enabled: bool,
}

// ============================================================================
// Adaptive work size constants (from q-mining)
// ============================================================================

const MIN_WORK_SIZE: usize = 1 << 16;   // 65K
const MAX_WORK_SIZE: usize = 1 << 23;   // 8M
const DISPATCH_TARGET_LOW_MS: u128 = 100;
const DISPATCH_TARGET_HIGH_MS: u128 = 400;
const LOCAL_WORK_SIZE: usize = 256;
const DEFAULT_WORK_SIZE: usize = 1 << 20; // 1M

/// Convert challenge bytes [u8; 32] to LE u32 words [u32; 8] for GPU upload.
#[inline]
fn challenge_bytes_to_words(challenge: &[u8; 32]) -> [u32; 8] {
    std::array::from_fn(|i| {
        u32::from_le_bytes([
            challenge[i * 4],
            challenge[i * 4 + 1],
            challenge[i * 4 + 2],
            challenge[i * 4 + 3],
        ])
    })
}

// ============================================================================
// Kernel cache (adapted from q-mining)
// ============================================================================

#[cfg(feature = "gpu-opencl")]
fn kernel_cache_dir() -> std::path::PathBuf {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| ".".to_string());
    std::path::PathBuf::from(home)
        .join(".config")
        .join("slint-wallet")
        .join("kernel-cache")
}

#[cfg(feature = "gpu-opencl")]
fn kernel_cache_key(device_name: &str, driver_version: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    BLAKE3_KERNEL_SOURCE.hash(&mut hasher);
    device_name.hash(&mut hasher);
    driver_version.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

#[cfg(feature = "gpu-opencl")]
fn load_cached_kernel(cache_key: &str) -> Option<Vec<u8>> {
    let path = kernel_cache_dir().join(format!("{}.clbin", cache_key));
    std::fs::read(&path).ok()
}

#[cfg(feature = "gpu-opencl")]
fn save_kernel_cache(cache_key: &str, binary: &[u8]) {
    let dir = kernel_cache_dir();
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join(format!("{}.clbin", cache_key));
    let _ = std::fs::write(&path, binary);
}

// ============================================================================
// GPU Detection
// ============================================================================

/// Detect all OpenCL GPU devices on the system.
/// Returns empty vec if no GPUs found or OpenCL not available.
pub fn detect_gpu_devices() -> Vec<GpuDevice> {
    #[cfg(feature = "gpu-opencl")]
    {
        _detect_gpu_devices_opencl()
    }
    #[cfg(not(feature = "gpu-opencl"))]
    {
        Vec::new()
    }
}

#[cfg(feature = "gpu-opencl")]
fn _detect_gpu_devices_opencl() -> Vec<GpuDevice> {
    use opencl3::device::{Device, CL_DEVICE_TYPE_GPU};
    use opencl3::platform::get_platforms;

    let platforms = match get_platforms() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("[GPU] OpenCL platform query failed: {}", e);
            return Vec::new();
        }
    };

    let mut devices = Vec::new();
    let mut index = 0;

    for platform in &platforms {
        let plat_name = platform.name().unwrap_or_else(|_| "Unknown".into());
        if let Ok(device_ids) = platform.get_devices(CL_DEVICE_TYPE_GPU) {
            eprintln!("[GPU] Platform: {} — {} GPU(s)", plat_name, device_ids.len());
            for device_id in device_ids {
                let device = Device::new(device_id);
                let name = device.name().unwrap_or_default();
                let vendor = device.vendor().unwrap_or_default();
                let memory = device.global_mem_size().unwrap_or(0);
                let cus = device.max_compute_units().unwrap_or(0);

                eprintln!("[GPU]   [{}] {} — {} CUs, {} MB VRAM",
                    index, name, cus, memory / (1024 * 1024));

                devices.push(GpuDevice {
                    index,
                    name,
                    vendor,
                    memory_mb: memory / (1024 * 1024),
                    compute_units: cus,
                    enabled: true,
                });
                index += 1;
            }
        }
    }

    devices
}

// ============================================================================
// GPU Mining Entry Points
// ============================================================================

/// Start GPU mining threads. One std::thread per GPU device.
/// Shares the same challenge RwLock and new_block_signal as CPU threads.
///
/// Returns immediately; mining happens in background threads.
pub fn start_gpu_mining(
    state: Arc<MinerState>,
    api_client: Arc<ApiClient>,
    miner_address: String,
    rt: tokio::runtime::Handle,
    pool_mode: bool,
    challenge: Arc<std::sync::RwLock<Option<SharedChallenge>>>,
    new_block_signal: Arc<AtomicU64>,
) {
    let devices = detect_gpu_devices();
    if devices.is_empty() {
        eprintln!("[GPU] No GPU devices detected — GPU mining disabled");
        state.set_gpu_status("No GPUs detected");
        return;
    }

    let num_gpus = devices.len();
    state.gpu_device_count.store(num_gpus as u64, Ordering::SeqCst);
    state.gpu_enabled.store(true, Ordering::SeqCst);
    state.set_gpu_status(&format!("{} GPU(s) starting...", num_gpus));
    eprintln!("[GPU] Starting GPU mining on {} device(s)", num_gpus);

    for device in devices {
        let state = state.clone();
        let api_client = api_client.clone();
        let miner_address = miner_address.clone();
        let rt = rt.clone();
        let challenge = challenge.clone();
        let new_block_signal = new_block_signal.clone();

        std::thread::Builder::new()
            .name(format!("gpu-miner-{}", device.index))
            .spawn(move || {
                gpu_worker_loop(
                    device,
                    state,
                    api_client,
                    miner_address,
                    rt,
                    pool_mode,
                    challenge,
                    new_block_signal,
                );
            })
            .expect("failed to spawn GPU mining thread");
    }
}

/// Stop GPU mining (GPU threads check gpu_enabled and exit).
pub fn stop_gpu_mining(state: &MinerState) {
    state.gpu_enabled.store(false, Ordering::SeqCst);
    state.gpu_hashrate.store(0, Ordering::SeqCst);
}

// ============================================================================
// Per-GPU Worker Loop
// ============================================================================

fn gpu_worker_loop(
    device: GpuDevice,
    state: Arc<MinerState>,
    api_client: Arc<ApiClient>,
    miner_address: String,
    rt: tokio::runtime::Handle,
    pool_mode: bool,
    challenge: Arc<std::sync::RwLock<Option<SharedChallenge>>>,
    new_block_signal: Arc<AtomicU64>,
) {
    #[cfg(feature = "gpu-opencl")]
    {
        _gpu_worker_loop_opencl(
            device, state, api_client, miner_address, rt, pool_mode,
            challenge, new_block_signal,
        );
    }
    #[cfg(not(feature = "gpu-opencl"))]
    {
        let _ = (device, state, api_client, miner_address, rt, pool_mode, challenge, new_block_signal);
        eprintln!("[GPU] OpenCL not compiled in — GPU worker exiting");
    }
}

#[cfg(feature = "gpu-opencl")]
fn _gpu_worker_loop_opencl(
    device: GpuDevice,
    state: Arc<MinerState>,
    api_client: Arc<ApiClient>,
    miner_address: String,
    rt: tokio::runtime::Handle,
    pool_mode: bool,
    challenge: Arc<std::sync::RwLock<Option<SharedChallenge>>>,
    new_block_signal: Arc<AtomicU64>,
) {
    use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE};
    use opencl3::context::Context;
    use opencl3::device::{Device, CL_DEVICE_TYPE_GPU};
    use opencl3::kernel::Kernel;
    use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY};
    use opencl3::program::Program;
    use opencl3::types::{cl_uchar, cl_uint, cl_ulong};

    let gpu_idx = device.index;

    // Find the OpenCL device by global index (matching detect_gpu_devices ordering)
    let platforms = match opencl3::platform::get_platforms() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("[GPU-{}] Failed to get platforms: {}", gpu_idx, e);
            return;
        }
    };

    let mut target_device: Option<Device> = None;
    let mut global_idx = 0usize;
    'find: for platform in &platforms {
        if let Ok(device_ids) = platform.get_devices(CL_DEVICE_TYPE_GPU) {
            for device_id in device_ids {
                if global_idx == gpu_idx {
                    target_device = Some(Device::new(device_id));
                    break 'find;
                }
                global_idx += 1;
            }
        }
    }

    let cl_device = match target_device {
        Some(d) => d,
        None => {
            eprintln!("[GPU-{}] Device not found", gpu_idx);
            return;
        }
    };

    let context = match Context::from_device(&cl_device) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[GPU-{}] Failed to create context: {}", gpu_idx, e);
            return;
        }
    };

    let queue = match CommandQueue::create_default(&context, CL_QUEUE_PROFILING_ENABLE) {
        Ok(q) => q,
        Err(e) => {
            eprintln!("[GPU-{}] Failed to create command queue: {}", gpu_idx, e);
            return;
        }
    };

    // Compile kernel (with cache)
    let dev_name = cl_device.name().unwrap_or_default();
    let dev_version = cl_device.version().unwrap_or_default();
    let cache_key = kernel_cache_key(&dev_name, &dev_version);

    let program = if let Some(binary) = load_cached_kernel(&cache_key) {
        match Program::create_and_build_from_binary(&context, &[&binary], "") {
            Ok(prog) => {
                eprintln!("[GPU-{}] Kernel loaded from cache", gpu_idx);
                prog
            }
            Err(_) => {
                let prog = match Program::create_and_build_from_source(&context, BLAKE3_KERNEL_SOURCE, "") {
                    Ok(p) => p,
                    Err(e) => {
                        eprintln!("[GPU-{}] Kernel compile failed: {}", gpu_idx, e);
                        return;
                    }
                };
                if let Ok(binaries) = prog.get_binaries() {
                    if let Some(bin) = binaries.first() {
                        save_kernel_cache(&cache_key, bin);
                    }
                }
                prog
            }
        }
    } else {
        eprintln!("[GPU-{}] Compiling kernel (first run — will be cached)...", gpu_idx);
        let t0 = Instant::now();
        let prog = match Program::create_and_build_from_source(&context, BLAKE3_KERNEL_SOURCE, "") {
            Ok(p) => p,
            Err(e) => {
                eprintln!("[GPU-{}] Kernel compile failed: {}", gpu_idx, e);
                return;
            }
        };
        eprintln!("[GPU-{}] Kernel compiled in {:.1}s", gpu_idx, t0.elapsed().as_secs_f64());
        if let Ok(binaries) = prog.get_binaries() {
            if let Some(bin) = binaries.first() {
                save_kernel_cache(&cache_key, bin);
            }
        }
        prog
    };

    let kernel = match Kernel::create(&program, "blake3_mine") {
        Ok(k) => k,
        Err(e) => {
            eprintln!("[GPU-{}] Kernel creation failed: {}", gpu_idx, e);
            return;
        }
    };

    // Pre-allocate persistent buffers
    let mut challenge_buf = match unsafe {
        Buffer::<cl_uint>::create(&context, CL_MEM_READ_ONLY, 8, std::ptr::null_mut())
    } {
        Ok(b) => b,
        Err(e) => { eprintln!("[GPU-{}] Buffer alloc failed: {}", gpu_idx, e); return; }
    };
    let mut target_buf = match unsafe {
        Buffer::<cl_uchar>::create(&context, CL_MEM_READ_ONLY, 32, std::ptr::null_mut())
    } {
        Ok(b) => b,
        Err(e) => { eprintln!("[GPU-{}] Buffer alloc failed: {}", gpu_idx, e); return; }
    };
    let found_nonce_buf = match unsafe {
        Buffer::<cl_ulong>::create(&context, CL_MEM_WRITE_ONLY, 1, std::ptr::null_mut())
    } {
        Ok(b) => b,
        Err(e) => { eprintln!("[GPU-{}] Buffer alloc failed: {}", gpu_idx, e); return; }
    };
    let found_hash_buf = match unsafe {
        Buffer::<cl_uchar>::create(&context, CL_MEM_WRITE_ONLY, 32, std::ptr::null_mut())
    } {
        Ok(b) => b,
        Err(e) => { eprintln!("[GPU-{}] Buffer alloc failed: {}", gpu_idx, e); return; }
    };
    let mut found_flag_buf = match unsafe {
        Buffer::<cl_uint>::create(&context, CL_MEM_READ_WRITE, 1, std::ptr::null_mut())
    } {
        Ok(b) => b,
        Err(e) => { eprintln!("[GPU-{}] Buffer alloc failed: {}", gpu_idx, e); return; }
    };

    eprintln!("[GPU-{}] {} ready — buffers allocated", gpu_idx, device.name);
    state.set_gpu_status(&format!("{} GPU(s) active", state.gpu_device_count.load(Ordering::Relaxed)));

    // Mining state
    let mut nonce: u64 = rand::random::<u64>().wrapping_add(gpu_idx as u64 * 10_000_000_000);
    let mut adaptive_work_size: usize = DEFAULT_WORK_SIZE;
    let mut cached_challenge = [0u8; 32];
    let mut cached_target = [0u8; 32];
    let mut last_hashrate_time = Instant::now();
    let mut hashes_since_last: u64 = 0;
    let mut last_known_block_signal = new_block_signal.load(Ordering::Relaxed);
    let server_url = api_client.base_url().to_string();

    const CL_TRUE: cl_uint = 1;
    const CL_FALSE: cl_uint = 0;

    while state.running.load(Ordering::SeqCst) && state.gpu_enabled.load(Ordering::SeqCst) {
        // Read current challenge
        let (ch_bytes, tg_bytes, ch_hash, diff_target, block_height) = {
            let lock = challenge.read().unwrap();
            match lock.as_ref() {
                Some(sc) => (
                    sc.challenge_bytes,
                    sc.target_bytes,
                    sc.challenge_hash.clone(),
                    sc.difficulty_target.clone(),
                    sc.height,
                ),
                None => {
                    drop(lock);
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    continue;
                }
            }
        };

        last_known_block_signal = new_block_signal.load(Ordering::Relaxed);

        // Round work size to LOCAL_WORK_SIZE multiple
        let work_size = (adaptive_work_size / LOCAL_WORK_SIZE) * LOCAL_WORK_SIZE;
        let work_size = work_size.max(LOCAL_WORK_SIZE);

        // Conditional upload (only when challenge/target changes)
        unsafe {
            if cached_challenge != ch_bytes {
                let words = challenge_bytes_to_words(&ch_bytes);
                if let Err(e) = queue.enqueue_write_buffer(&mut challenge_buf, CL_FALSE, 0, &words, &[]) {
                    eprintln!("[GPU-{}] Challenge upload failed: {}", gpu_idx, e);
                    std::thread::sleep(std::time::Duration::from_secs(1));
                    continue;
                }
                cached_challenge = ch_bytes;
            }
            if cached_target != tg_bytes {
                if let Err(e) = queue.enqueue_write_buffer(&mut target_buf, CL_FALSE, 0, &tg_bytes, &[]) {
                    eprintln!("[GPU-{}] Target upload failed: {}", gpu_idx, e);
                    std::thread::sleep(std::time::Duration::from_secs(1));
                    continue;
                }
                cached_target = tg_bytes;
            }
            // Zero the found_flag before dispatch
            let zero_flag: [u32; 1] = [0];
            if let Err(e) = queue.enqueue_write_buffer(&mut found_flag_buf, CL_FALSE, 0, &zero_flag, &[]) {
                eprintln!("[GPU-{}] Flag zero failed: {}", gpu_idx, e);
                std::thread::sleep(std::time::Duration::from_secs(1));
                continue;
            }
        }

        // Set kernel args
        if let Err(e) = (|| -> Result<(), opencl3::error_codes::ClError> {
            unsafe {
                kernel.set_arg(0, &challenge_buf)?;
                kernel.set_arg(1, &target_buf)?;
                kernel.set_arg(2, &nonce)?;
                kernel.set_arg(3, &found_nonce_buf)?;
                kernel.set_arg(4, &found_hash_buf)?;
                kernel.set_arg(5, &found_flag_buf)?;
            }
            Ok(())
        })() {
            eprintln!("[GPU-{}] Set kernel args failed: {}", gpu_idx, e);
            std::thread::sleep(std::time::Duration::from_secs(1));
            continue;
        }

        // Dispatch
        let dispatch_start = Instant::now();
        let global_ws = [work_size];
        let local_ws = [LOCAL_WORK_SIZE];

        if let Err(e) = unsafe {
            queue.enqueue_nd_range_kernel(
                kernel.get(),
                1,
                std::ptr::null(),
                global_ws.as_ptr(),
                local_ws.as_ptr(),
                &[],
            )
        } {
            eprintln!("[GPU-{}] Kernel dispatch failed: {}", gpu_idx, e);
            std::thread::sleep(std::time::Duration::from_secs(1));
            continue;
        }

        if let Err(e) = queue.finish() {
            eprintln!("[GPU-{}] Queue finish failed: {}", gpu_idx, e);
            std::thread::sleep(std::time::Duration::from_secs(1));
            continue;
        }

        let dispatch_ms = dispatch_start.elapsed().as_millis();

        // Adaptive work size tuning
        if dispatch_ms < DISPATCH_TARGET_LOW_MS {
            adaptive_work_size = (adaptive_work_size * 3 / 2).min(MAX_WORK_SIZE);
        } else if dispatch_ms > DISPATCH_TARGET_HIGH_MS {
            adaptive_work_size = (adaptive_work_size * 2 / 3).max(MIN_WORK_SIZE);
        }

        hashes_since_last += work_size as u64;

        // Update hashrate every ~2 seconds
        let elapsed = last_hashrate_time.elapsed();
        if elapsed.as_millis() >= 2000 {
            let rate = (hashes_since_last as f64 / elapsed.as_secs_f64()) as u64;
            state.gpu_hashrate.store(rate, Ordering::SeqCst);
            hashes_since_last = 0;
            last_hashrate_time = Instant::now();
        }

        // Check for solution
        let mut found_flag: [u32; 1] = [0];
        unsafe {
            if let Err(e) = queue.enqueue_read_buffer(&found_flag_buf, CL_TRUE, 0, &mut found_flag, &[]) {
                eprintln!("[GPU-{}] Read flag failed: {}", gpu_idx, e);
            }
        }

        if found_flag[0] != 0 {
            let mut found_nonce_val: [u64; 1] = [0];
            let mut found_hash_val: [u8; 32] = [0; 32];

            unsafe {
                let _ = queue.enqueue_read_buffer(&found_nonce_buf, CL_TRUE, 0, &mut found_nonce_val, &[]);
                let _ = queue.enqueue_read_buffer(&found_hash_buf, CL_TRUE, 0, &mut found_hash_val, &[]);
            }

            let sol_nonce = found_nonce_val[0];
            let sol_hash = found_hash_val;

            eprintln!("[GPU-{}] Found valid hash! Nonce: {} — submitting...", gpu_idx, sol_nonce);

            let hr = state.gpu_hashrate.load(Ordering::Relaxed) + state.hashrate.load(Ordering::Relaxed);
            let submission = MiningSubmission {
                miner_address: miner_address.clone(),
                nonce: sol_nonce,
                hash: hex::encode(sol_hash),
                difficulty_target: diff_target.clone(),
                challenge_hash: Some(ch_hash.clone()),
                hash_rate: if hr > 0 { Some(hr as f64 / 1000.0) } else { None },
                miner_id: Some(format!("slint-gpu{}-{}", gpu_idx, &miner_address[3..11])),
                worker_name: Some("slint-wallet-gpu".to_string()),
                miner_version: Some(env!("CARGO_PKG_VERSION").to_string()),
                // v1.0.5: Genus-2 VDF fields (None until activation height reached)
                vdf_output: None,
                vdf_proof: None,
                vdf_checkpoints: None,
                vdf_iterations_count: None,
            };

            let client = api_client.clone();
            let sub = submission.clone();
            let srv = server_url.clone();
            match rt.block_on(crate::miner::submit_with_fallback(&client, &sub, &srv)) {
                Ok(resp) => {
                    eprintln!("[GPU-{}] Block accepted! {:?}", gpu_idx, resp);
                    state.gpu_blocks_found.fetch_add(1, Ordering::SeqCst);
                    state.set_status("GPU block found!");
                }
                Err(msg) => {
                    eprintln!("[GPU-{}] Submit error: {}", gpu_idx, msg);
                }
            }

            // Pool share
            if pool_mode {
                let share_id = hex::encode(&sol_hash[..16]);
                let diff = hr as f64;
                let client = api_client.clone();
                let addr = miner_address.clone();
                match rt.block_on(client.submit_pool_share(
                    &addr,
                    "slint-wallet-gpu",
                    &share_id,
                    diff,
                    block_height,
                    sol_nonce,
                )) {
                    Ok(_) => eprintln!("[GPU-{}] Pool share submitted", gpu_idx),
                    Err(e) => eprintln!("[GPU-{}] Pool share error: {}", gpu_idx, e),
                }
            }
        }

        // Advance nonce
        nonce = nonce.wrapping_add(work_size as u64);

        // Check new-block signal — if changed, loop back immediately to pick up new challenge
        let sig = new_block_signal.load(Ordering::Relaxed);
        if sig != last_known_block_signal {
            last_known_block_signal = sig;
            // Reset cached challenge to force re-upload on next iteration
            cached_challenge = [0u8; 32];
        }
    }

    eprintln!("[GPU-{}] Worker stopped", gpu_idx);
}
