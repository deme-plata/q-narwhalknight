use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use crate::api_client::ApiClient;
use crate::models::MiningSubmission;

/// Server-side VDF verification uses exactly 100 iterations.
/// This MUST match the server's recomputation in handlers.rs submit_mining_solution.
const VDF_ITERATIONS: u32 = 100;

/// Background miner state shared between the mining threads and the UI.
pub struct MinerState {
    pub running: AtomicBool,
    pub hashrate: AtomicU64,
    pub blocks_found: AtomicU64,
    pub total_hashes: AtomicU64,
    pub active_threads: AtomicU64,
    /// Last status message for UI display
    pub last_status: std::sync::Mutex<String>,
}

impl MinerState {
    pub fn new() -> Self {
        Self {
            running: AtomicBool::new(false),
            hashrate: AtomicU64::new(0),
            blocks_found: AtomicU64::new(0),
            total_hashes: AtomicU64::new(0),
            active_threads: AtomicU64::new(0),
            last_status: std::sync::Mutex::new(String::new()),
        }
    }

    pub fn set_status(&self, msg: &str) {
        if let Ok(mut s) = self.last_status.lock() {
            *s = msg.to_string();
        }
    }
}

/// Shared challenge data that all mining threads read from.
struct SharedChallenge {
    challenge_bytes: [u8; 32],
    target_bytes: [u8; 32],
    challenge_hash: String,
    difficulty_target: String,
    height: u64,
}

/// BLAKE3 VDF mining: hash(challenge || nonce), then iterate 100 times.
/// Must match server-side verification exactly (100 VDF iterations).
#[inline(always)]
fn mine_hash(challenge_bytes: &[u8; 32], nonce: u64) -> [u8; 32] {
    let mut input = [0u8; 40];
    input[..32].copy_from_slice(challenge_bytes);
    input[32..40].copy_from_slice(&nonce.to_le_bytes());

    let mut current = *blake3::hash(&input).as_bytes();

    for _ in 0..VDF_ITERATIONS {
        current = *blake3::hash(&current).as_bytes();
    }

    current
}

/// Check if hash is below difficulty target.
#[inline(always)]
fn meets_difficulty(hash: &[u8; 32], target: &[u8; 32]) -> bool {
    for i in 0..32 {
        if hash[i] < target[i] { return true; }
        if hash[i] > target[i] { return false; }
    }
    true
}

/// Detect the optimal number of mining threads for this system.
fn detect_mining_threads() -> usize {
    let std_count = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    let num_cpus_logical = num_cpus::get();
    let num_cpus_physical = num_cpus::get_physical();

    #[cfg(target_os = "windows")]
    let win_count = {
        extern "system" {
            fn GetActiveProcessorCount(GroupNumber: u16) -> u32;
        }
        let count = unsafe { GetActiveProcessorCount(0xFFFF) } as usize;
        if count > 0 { count } else { 0 }
    };
    #[cfg(not(target_os = "windows"))]
    let win_count = 0usize;

    #[cfg(target_arch = "x86_64")]
    {
        let cpuid = raw_cpuid::CpuId::new();
        let brand = cpuid.get_processor_brand_string()
            .map(|b| b.as_str().to_string())
            .unwrap_or_default();
        let has_avx2 = cpuid.get_extended_feature_info()
            .map(|ef| ef.has_avx2())
            .unwrap_or(false);
        eprintln!("[MINER] CPU: {} | AVX2={}", brand, has_avx2);
    }

    let detected = std_count.max(num_cpus_logical).max(win_count);
    // Reserve 2 threads for OS/UI/network (minimum 1 mining thread)
    let mining_threads = if detected > 4 { detected - 2 } else { detected.max(1) };

    eprintln!(
        "[MINER] Threads: std={}, logical={}, physical={}, win_api={} → {} mining threads",
        std_count, num_cpus_logical, num_cpus_physical, win_count, mining_threads
    );

    mining_threads
}

/// Pin a thread to a specific CPU core for cache locality.
fn pin_thread_to_core(thread_id: usize) {
    let core_ids = core_affinity::get_core_ids().unwrap_or_default();
    if thread_id < core_ids.len() {
        if core_affinity::set_for_current(core_ids[thread_id]) {
            return;
        }
    }

    #[cfg(target_os = "windows")]
    {
        use std::os::raw::c_ulong;
        extern "system" {
            fn GetCurrentThread() -> *mut std::ffi::c_void;
            fn SetThreadAffinityMask(hThread: *mut std::ffi::c_void, dwThreadAffinityMask: c_ulong) -> c_ulong;
        }
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
    pool_mode: bool,
) {
    state.running.store(true, Ordering::SeqCst);
    state.hashrate.store(0, Ordering::SeqCst);
    state.set_status("Starting...");

    // Validate miner address format before starting
    if !miner_address.starts_with("qnk") || miner_address.len() != 67 {
        eprintln!("[MINER] ERROR: Invalid miner address format: '{}' (need qnk + 64 hex = 67 chars)", miner_address);
        state.set_status("Error: invalid address");
        state.running.store(false, Ordering::SeqCst);
        return;
    }
    eprintln!("[MINER] Address: {}...{}", &miner_address[..10], &miner_address[miner_address.len()-6..]);

    let num_threads = detect_mining_threads();
    state.active_threads.store(num_threads as u64, Ordering::SeqCst);

    // Shared challenge behind RwLock so all threads read the same challenge
    let challenge: Arc<std::sync::RwLock<Option<SharedChallenge>>> =
        Arc::new(std::sync::RwLock::new(None));

    // Per-thread hashrate counters
    let thread_hashes: Arc<Vec<AtomicU64>> = Arc::new(
        (0..num_threads).map(|_| AtomicU64::new(0)).collect(),
    );

    // Coordinator thread: fetches challenges every 10s, aggregates hashrate every second
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
                let mut consecutive_errors = 0u32;
                let mut last_height = 0u64;

                while state.running.load(Ordering::SeqCst) {
                    // Fetch new challenge
                    let client = api_client.clone();
                    match rt.block_on(client.get_mining_challenge()) {
                        Ok(c) => {
                            consecutive_errors = 0;
                            if let (Ok(ch), Ok(tg)) = (
                                hex::decode(&c.challenge_hash),
                                hex::decode(&c.difficulty_target),
                            ) {
                                if ch.len() == 32 && tg.len() == 32 {
                                    let mut cb = [0u8; 32];
                                    let mut tb = [0u8; 32];
                                    cb.copy_from_slice(&ch);
                                    tb.copy_from_slice(&tg);

                                    let height_changed = c.block_height != last_height;
                                    last_height = c.block_height;

                                    let sc = SharedChallenge {
                                        challenge_bytes: cb,
                                        target_bytes: tb,
                                        challenge_hash: c.challenge_hash.clone(),
                                        difficulty_target: c.difficulty_target.clone(),
                                        height: c.block_height,
                                    };
                                    *challenge.write().unwrap() = Some(sc);

                                    if height_changed {
                                        eprintln!("[MINER] New challenge at height {} (reward: {:.4} QUG, VDF: {}→100 fixed)",
                                            c.block_height, c.block_reward, c.vdf_iterations);
                                    }
                                    state.set_status(&format!("Mining at height {}", c.block_height));
                                } else {
                                    eprintln!("[MINER] Bad challenge data: ch_len={}, tg_len={}", ch.len(), tg.len());
                                    state.set_status("Error: bad challenge data");
                                }
                            } else {
                                eprintln!("[MINER] Failed to decode challenge hex");
                                state.set_status("Error: decode failed");
                            }
                        }
                        Err(e) => {
                            consecutive_errors += 1;
                            let msg = e.to_string();
                            if msg.contains("503") || msg.contains("SERVICE_UNAVAILABLE") {
                                state.set_status("Waiting for sync...");
                                eprintln!("[MINER] Node syncing — mining paused (attempt {})", consecutive_errors);
                            } else if msg.contains("No peers") || msg.contains("discovering") {
                                state.set_status("Connecting to network...");
                                eprintln!("[MINER] No peers yet — waiting (attempt {})", consecutive_errors);
                            } else {
                                state.set_status(&format!("Error (retry {})", consecutive_errors));
                                eprintln!("[MINER] Challenge error (attempt {}): {}", consecutive_errors, msg);
                            }
                        }
                    }

                    // Wait 10 seconds between challenge fetches, updating hashrate every second
                    let wait_secs = if consecutive_errors > 5 { 15 } else { 10 };
                    for _ in 0..wait_secs {
                        if !state.running.load(Ordering::Relaxed) {
                            return;
                        }
                        std::thread::sleep(std::time::Duration::from_secs(1));

                        // Sum all thread hash counters for hashrate
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
                pin_thread_to_core(thread_id);

                let mut nonce: u64 = rand::random::<u64>().wrapping_add(thread_id as u64 * 1_000_000_000);
                const BATCH_SIZE: u64 = 10_000;

                while state.running.load(Ordering::SeqCst) {
                    // Read current challenge (one RwLock read per batch)
                    let ch = {
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

                    let (challenge_bytes, target_bytes, challenge_hash, difficulty_target, block_height) = ch;

                    // Mine a batch — tight inner loop
                    for _ in 0..BATCH_SIZE {
                        let hash = mine_hash(&challenge_bytes, nonce);

                        if meets_difficulty(&hash, &target_bytes) {
                            let hr = state.hashrate.load(Ordering::Relaxed);
                            let submission = MiningSubmission {
                                miner_address: miner_address.clone(),
                                nonce,
                                hash: hex::encode(hash),
                                difficulty_target: difficulty_target.clone(),
                                challenge_hash: Some(challenge_hash.clone()),
                                hash_rate: if hr > 0 { Some(hr as f64 / 1000.0) } else { None }, // Server expects KH/s
                                miner_id: Some(format!("slint-{}", &miner_address[3..11])),
                                worker_name: Some("slint-wallet".to_string()),
                                miner_version: Some(env!("CARGO_PKG_VERSION").to_string()),
                            };

                            eprintln!("[MINER] Found valid hash! Nonce: {} — submitting...", nonce);
                            let client = api_client.clone();
                            match rt.block_on(client.submit_mining_solution(&submission)) {
                                Ok(resp) => {
                                    eprintln!("[MINER] Block accepted! Response: {:?}", resp);
                                    state.blocks_found.fetch_add(1, Ordering::SeqCst);
                                    state.set_status("Block found!");
                                }
                                Err(e) => {
                                    let msg = e.to_string();
                                    if msg.contains("Hash verification failed") {
                                        eprintln!("[MINER] HASH MISMATCH — server rejected (VDF issue?)");
                                    } else if msg.contains("Duplicate nonce") {
                                        eprintln!("[MINER] Duplicate nonce — already submitted");
                                    } else if msg.contains("does not meet") {
                                        eprintln!("[MINER] Below difficulty — stale challenge?");
                                    } else {
                                        eprintln!("[MINER] Submit error: {}", msg);
                                    }
                                }
                            }

                            // Pool mode: also submit as pool share for PPLNS tracking
                            if pool_mode {
                                let share_id = hex::encode(&hash[..16]);
                                let diff = hr as f64;
                                let client = api_client.clone();
                                let addr = miner_address.clone();
                                match rt.block_on(client.submit_pool_share(
                                    &addr,
                                    "slint-wallet",
                                    &share_id,
                                    diff,
                                    block_height,
                                    nonce,
                                )) {
                                    Ok(_) => eprintln!("[MINER] Pool share submitted"),
                                    Err(e) => eprintln!("[MINER] Pool share error: {}", e),
                                }
                            }
                        }

                        nonce = nonce.wrapping_add(1);
                    }

                    my_counter[thread_id].fetch_add(BATCH_SIZE, Ordering::Relaxed);
                }
            })
            .expect("failed to spawn mining thread");
    }
}

/// Stop the mining loop.
pub fn stop_mining(state: &MinerState) {
    state.running.store(false, Ordering::SeqCst);
    state.set_status("Stopped");
}
