use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use crate::api_client::ApiClient;
use crate::models::MiningSubmission;

/// Background miner state shared between the mining threads and the UI.
pub struct MinerState {
    pub running: AtomicBool,
    pub hashrate: AtomicU64,
    pub blocks_found: AtomicU64,
    pub total_hashes: AtomicU64,
    pub active_threads: AtomicU64,
}

impl MinerState {
    pub fn new() -> Self {
        Self {
            running: AtomicBool::new(false),
            hashrate: AtomicU64::new(0),
            blocks_found: AtomicU64::new(0),
            total_hashes: AtomicU64::new(0),
            active_threads: AtomicU64::new(0),
        }
    }
}

/// Shared challenge data that all mining threads read from.
struct SharedChallenge {
    challenge_bytes: [u8; 32],
    target_bytes: [u8; 32],
    vdf_iterations: u32,
    challenge_hash: String,
    difficulty_target: String,
}

/// BLAKE3 VDF mining: hash(challenge || nonce), then iterate N times.
/// Marked #[inline(always)] to avoid function call overhead in hot loop.
#[inline(always)]
fn mine_hash(challenge_bytes: &[u8; 32], nonce: u64, vdf_iterations: u32) -> [u8; 32] {
    let mut input = [0u8; 40];
    input[..32].copy_from_slice(challenge_bytes);
    input[32..40].copy_from_slice(&nonce.to_le_bytes());

    let mut current = *blake3::hash(&input).as_bytes();

    for _ in 0..vdf_iterations {
        current = *blake3::hash(&current).as_bytes();
    }

    current
}

/// Check if hash is below difficulty target.
#[inline(always)]
fn meets_difficulty(hash: &[u8; 32], target: &[u8; 32]) -> bool {
    // Compare byte-by-byte (big-endian) - early exit on first difference
    for i in 0..32 {
        if hash[i] < target[i] { return true; }
        if hash[i] > target[i] { return false; }
    }
    true
}

/// Detect the optimal number of mining threads for this system.
/// On Windows, uses GetActiveProcessorCount for high-core-count CPUs (EPYC 9654 etc.)
/// which correctly reports >64 cores across processor groups.
fn detect_mining_threads() -> usize {
    // Method 1: std::thread::available_parallelism (may cap at 64 on older Windows)
    let std_count = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    // Method 2: num_cpus crate (cross-platform, better Windows support)
    let num_cpus_logical = num_cpus::get();
    let num_cpus_physical = num_cpus::get_physical();

    // Method 3: On Windows, use GetActiveProcessorCount(ALL_PROCESSOR_GROUPS) for >64 cores
    #[cfg(target_os = "windows")]
    let win_count = {
        // ALL_PROCESSOR_GROUPS = 0xFFFF
        extern "system" {
            fn GetActiveProcessorCount(GroupNumber: u16) -> u32;
        }
        let count = unsafe { GetActiveProcessorCount(0xFFFF) } as usize;
        if count > 0 { count } else { 0 }
    };
    #[cfg(not(target_os = "windows"))]
    let win_count = 0usize;

    // Method 4: raw_cpuid for x86_64 - detect CPU brand and expected core count
    #[cfg(target_arch = "x86_64")]
    let cpuid_info = {
        let cpuid = raw_cpuid::CpuId::new();
        let brand = cpuid.get_processor_brand_string()
            .map(|b| b.as_str().to_string())
            .unwrap_or_default();
        let has_avx2 = cpuid.get_extended_feature_info()
            .map(|ef| ef.has_avx2())
            .unwrap_or(false);
        let has_avx512 = cpuid.get_extended_feature_info()
            .map(|ef| ef.has_avx512f())
            .unwrap_or(false);
        eprintln!("[MINER] CPU: {} | AVX2={} AVX512={}", brand, has_avx2, has_avx512);
        brand
    };
    #[cfg(not(target_arch = "x86_64"))]
    let cpuid_info = String::new();

    // Take the maximum of all detection methods
    let detected = std_count.max(num_cpus_logical).max(win_count);

    // Reserve 2 threads for OS/UI/network (minimum 1 mining thread)
    let mining_threads = if detected > 4 { detected - 2 } else { detected.max(1) };

    eprintln!(
        "[MINER] Thread detection: std={}, num_cpus={}/{} (logical/physical), win_api={} → using {} mining threads",
        std_count, num_cpus_logical, num_cpus_physical, win_count, mining_threads
    );

    if cpuid_info.contains("9654") && detected < 96 {
        eprintln!(
            "[MINER] WARNING: AMD EPYC 9654 detected but only {} threads visible! Expected 96-192.",
            detected
        );
        eprintln!("[MINER] Check: Windows processor group affinity, BIOS NUMA settings");
    }

    mining_threads
}

/// Pin a thread to a specific CPU core for cache locality.
/// On NUMA systems (EPYC), this prevents threads from migrating between sockets.
fn pin_thread_to_core(thread_id: usize) {
    let core_ids = core_affinity::get_core_ids().unwrap_or_default();
    if thread_id < core_ids.len() {
        if core_affinity::set_for_current(core_ids[thread_id]) {
            // Successfully pinned
            return;
        }
    }

    // Fallback for Windows: use SetThreadAffinityMask directly
    #[cfg(target_os = "windows")]
    {
        use std::os::raw::c_ulong;
        extern "system" {
            fn GetCurrentThread() -> *mut std::ffi::c_void;
            fn SetThreadAffinityMask(hThread: *mut std::ffi::c_void, dwThreadAffinityMask: c_ulong) -> c_ulong;
        }
        // Pin to core within first 64 (processor group 0)
        if thread_id < 64 {
            unsafe {
                let handle = GetCurrentThread();
                let mask: c_ulong = 1 << thread_id;
                SetThreadAffinityMask(handle, mask);
            }
        }
    }
}

/// Start the mining loop with multiple threads (one per CPU core).
pub fn start_mining(
    state: Arc<MinerState>,
    api_client: Arc<ApiClient>,
    miner_address: String,
    rt: tokio::runtime::Handle,
) {
    state.running.store(true, Ordering::SeqCst);
    state.hashrate.store(0, Ordering::SeqCst);

    let num_threads = detect_mining_threads();
    state.active_threads.store(num_threads as u64, Ordering::SeqCst);

    // Shared challenge behind RwLock so all threads read the same challenge
    let challenge: Arc<std::sync::RwLock<Option<SharedChallenge>>> =
        Arc::new(std::sync::RwLock::new(None));

    // Per-thread hashrate counters, summed by the coordinator
    let thread_hashes: Arc<Vec<AtomicU64>> = Arc::new(
        (0..num_threads).map(|_| AtomicU64::new(0)).collect(),
    );

    // Coordinator thread: fetches challenges and aggregates hashrate
    {
        let state = state.clone();
        let api_client = api_client.clone();
        let rt = rt.clone();
        let challenge = challenge.clone();
        let thread_hashes = thread_hashes.clone();

        std::thread::Builder::new()
            .name("miner-coordinator".into())
            .spawn(move || {
                let mut last_hashrate_calc = std::time::Instant::now();

                while state.running.load(Ordering::SeqCst) {
                    // Fetch new challenge
                    let client = api_client.clone();
                    match rt.block_on(client.get_mining_challenge()) {
                        Ok(c) => {
                            if let (Ok(ch), Ok(tg)) = (
                                hex::decode(&c.challenge_hash),
                                hex::decode(&c.difficulty_target),
                            ) {
                                if ch.len() == 32 && tg.len() == 32 {
                                    let mut cb = [0u8; 32];
                                    let mut tb = [0u8; 32];
                                    cb.copy_from_slice(&ch);
                                    tb.copy_from_slice(&tg);
                                    let sc = SharedChallenge {
                                        challenge_bytes: cb,
                                        target_bytes: tb,
                                        vdf_iterations: c.vdf_iterations,
                                        challenge_hash: c.challenge_hash,
                                        difficulty_target: c.difficulty_target,
                                    };
                                    *challenge.write().unwrap() = Some(sc);
                                }
                            }
                        }
                        Err(_) => {}
                    }

                    // Wait 30 seconds, updating hashrate every second
                    for _ in 0..30 {
                        if !state.running.load(Ordering::Relaxed) {
                            return;
                        }
                        std::thread::sleep(std::time::Duration::from_secs(1));

                        // Sum all thread hash counters
                        let elapsed = last_hashrate_calc.elapsed();
                        if elapsed.as_millis() >= 900 {
                            let total: u64 = thread_hashes
                                .iter()
                                .map(|h| h.swap(0, Ordering::Relaxed))
                                .sum();
                            let rate = (total as f64 / elapsed.as_secs_f64()) as u64;
                            state.hashrate.store(rate, Ordering::SeqCst);
                            state.total_hashes.fetch_add(total, Ordering::Relaxed);
                            last_hashrate_calc = std::time::Instant::now();
                        }
                    }
                }
            })
            .expect("failed to spawn miner coordinator");
    }

    // Spawn mining worker threads
    for thread_id in 0..num_threads {
        let state = state.clone();
        let api_client = api_client.clone();
        let miner_address = miner_address.clone();
        let rt = rt.clone();
        let challenge = challenge.clone();
        let my_counter = thread_hashes.clone();

        std::thread::Builder::new()
            .name(format!("miner-{}", thread_id))
            .spawn(move || {
                // Pin this thread to a CPU core for NUMA locality
                pin_thread_to_core(thread_id);

                // Each thread starts at a different nonce offset (random + spread)
                let mut nonce: u64 = rand::random::<u64>().wrapping_add(thread_id as u64 * 1_000_000_000);

                // Larger batch size = less overhead from RwLock reads and atomic ops
                // 10K nonces per batch is optimal: ~1ms per batch at 10M H/s
                const BATCH_SIZE: u64 = 10_000;

                while state.running.load(Ordering::SeqCst) {
                    // Read current challenge (one RwLock read per batch, not per nonce)
                    let ch = {
                        let lock = challenge.read().unwrap();
                        match lock.as_ref() {
                            Some(sc) => (
                                sc.challenge_bytes,
                                sc.target_bytes,
                                sc.vdf_iterations,
                                sc.challenge_hash.clone(),
                                sc.difficulty_target.clone(),
                            ),
                            None => {
                                drop(lock);
                                std::thread::sleep(std::time::Duration::from_millis(100));
                                continue;
                            }
                        }
                    };

                    let (challenge_bytes, target_bytes, vdf_iters, challenge_hash, difficulty_target) = ch;

                    // Mine a batch - tight inner loop with no allocations
                    for _ in 0..BATCH_SIZE {
                        let hash = mine_hash(&challenge_bytes, nonce, vdf_iters);

                        if meets_difficulty(&hash, &target_bytes) {
                            let submission = MiningSubmission {
                                miner_address: miner_address.clone(),
                                nonce,
                                hash: hex::encode(hash),
                                difficulty_target: difficulty_target.clone(),
                                challenge_hash: challenge_hash.clone(),
                                hash_rate: state.hashrate.load(Ordering::Relaxed) as f64,
                            };

                            let client = api_client.clone();
                            let _ = rt.block_on(client.submit_mining_solution(&submission));
                            state.blocks_found.fetch_add(1, Ordering::SeqCst);
                        }

                        nonce = nonce.wrapping_add(1);
                    }

                    // Report hashes to our counter (once per batch)
                    my_counter[thread_id].fetch_add(BATCH_SIZE, Ordering::Relaxed);
                }
            })
            .expect("failed to spawn mining thread");
    }
}

/// Stop the mining loop.
pub fn stop_mining(state: &MinerState) {
    state.running.store(false, Ordering::SeqCst);
}
