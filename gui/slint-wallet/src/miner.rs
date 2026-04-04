use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use crate::api_client::ApiClient;
#[cfg(feature = "gpu-opencl")]
use crate::gpu_miner;
use crate::models::MiningSubmission;

/// Server-side VDF verification uses exactly 100 iterations.
/// This MUST match the server's recomputation in handlers.rs submit_mining_solution.
const VDF_ITERATIONS: u32 = 100;

/// v8.5.2: Fallback bootstrap server (same as standalone miner).
/// If the primary server goes down, challenge fetch + solution submit automatically
/// fail over to this URL, preventing missed blocks.
const FALLBACK_BOOTSTRAP_URL: &str = "https://quillon.xyz";

/// Background miner state shared between the mining threads and the UI.
pub struct MinerState {
    pub running: AtomicBool,
    pub hashrate: AtomicU64,
    pub blocks_found: AtomicU64,
    pub total_hashes: AtomicU64,
    pub active_threads: AtomicU64,
    /// Last status message for UI display
    pub last_status: std::sync::Mutex<String>,
    // GPU mining fields
    pub gpu_hashrate: AtomicU64,
    pub gpu_blocks_found: AtomicU64,
    pub gpu_enabled: AtomicBool,
    pub gpu_device_count: AtomicU64,
    pub gpu_status: std::sync::Mutex<String>,
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
            gpu_hashrate: AtomicU64::new(0),
            gpu_blocks_found: AtomicU64::new(0),
            gpu_enabled: AtomicBool::new(false),
            gpu_device_count: AtomicU64::new(0),
            gpu_status: std::sync::Mutex::new(String::new()),
        }
    }

    pub fn set_status(&self, msg: &str) {
        if let Ok(mut s) = self.last_status.lock() {
            *s = msg.to_string();
        }
    }

    pub fn set_gpu_status(&self, msg: &str) {
        if let Ok(mut s) = self.gpu_status.lock() {
            *s = msg.to_string();
        }
    }
}

/// Shared challenge data that all mining threads (CPU + GPU) read from.
pub(crate) struct SharedChallenge {
    pub challenge_bytes: [u8; 32],
    pub target_bytes: [u8; 32],
    pub challenge_hash: String,
    pub difficulty_target: String,
    pub height: u64,
}

/// BLAKE3 VDF mining: hash(challenge || nonce), then iterate 100 times.
/// Must match server-side verification exactly (100 VDF iterations).
/// v8.5.2: Zero-allocation — pre-allocated 40-byte buffer (from standalone miner).
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

/// v8.5.2: SSE listener for new-block events — instant challenge refresh.
/// Ported from standalone miner's start_sse_listener with fallback support.
async fn start_sse_listener(
    wallet: String,
    server_url: String,
    is_running: Arc<AtomicBool>,
    new_block_signal: Arc<AtomicU64>,
    state: Arc<MinerState>,
) {
    use eventsource_client::{self as eventsource, Client as _};
    use futures::StreamExt;

    let primary_url = format!("{}/api/v1/events?wallet_address={}", server_url, wallet);
    let fallback_url = format!("{}/api/v1/events?wallet_address={}", FALLBACK_BOOTSTRAP_URL, wallet);
    let mut use_fallback = false;
    let mut primary_fail_count = 0u32;

    loop {
        if !is_running.load(Ordering::SeqCst) {
            break;
        }

        let url = if use_fallback { &fallback_url } else { &primary_url };

        let client = match eventsource::ClientBuilder::for_url(url) {
            Ok(builder) => builder.build(),
            Err(e) => {
                eprintln!("[MINER-SSE] Failed to create SSE client for {}: {}", url, e);
                if !use_fallback {
                    primary_fail_count += 1;
                    if primary_fail_count >= 3 {
                        eprintln!("[MINER-SSE] Switching to fallback {}", FALLBACK_BOOTSTRAP_URL);
                        use_fallback = true;
                        primary_fail_count = 0;
                    }
                }
                tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
                continue;
            }
        };

        let mut stream = client.stream();
        eprintln!("[MINER-SSE] Connected to {}", url);

        while is_running.load(Ordering::SeqCst) {
            match stream.next().await {
                Some(Ok(eventsource::SSE::Event(ev))) => {
                    // New block → signal mining threads to refresh challenge immediately
                    if ev.event_type == "new-block" {
                        if let Ok(data) = serde_json::from_str::<serde_json::Value>(&ev.data) {
                            if let Some(height) = data.get("height").and_then(|v| v.as_u64()) {
                                let sig = new_block_signal.fetch_add(1, Ordering::SeqCst) + 1;
                                eprintln!("[MINER-SSE] New block #{} — signaling threads (sig={})", height, sig);
                            }
                        }
                        // Reset fail counter on successful events
                        primary_fail_count = 0;
                    }

                    // Mining reward notification
                    if ev.event_type == "mining_reward" {
                        if let Ok(data) = serde_json::from_str::<serde_json::Value>(&ev.data) {
                            if let Some(miner_addr) = data.get("miner_address").and_then(|v| v.as_str()) {
                                if miner_addr == wallet {
                                    let reward = data.get("reward_qnk").and_then(|v| v.as_f64()).unwrap_or(0.0);
                                    let height = data.get("block_height").and_then(|v| v.as_u64()).unwrap_or(0);
                                    eprintln!("[MINER-SSE] Mining reward! +{:.8} QUG at block #{}", reward, height);
                                    state.set_status(&format!("Reward! +{:.4} QUG", reward));
                                }
                            }
                        }
                    }

                    // Balance update for mining rewards
                    if ev.event_type == "balance_updated" {
                        if let Ok(data) = serde_json::from_str::<serde_json::Value>(&ev.data) {
                            if let Some(addr) = data.get("wallet_address").and_then(|v| v.as_str()) {
                                if addr == wallet {
                                    if let Some(reason) = data.get("change_reason").and_then(|v| v.as_str()) {
                                        if reason == "mining_reward" {
                                            let bal = data.get("new_balance").and_then(|v| v.as_f64()).unwrap_or(0.0);
                                            eprintln!("[MINER-SSE] Balance: {:.8} QUG", bal);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                Some(Ok(eventsource::SSE::Comment(_))) => {}
                Some(Err(e)) => {
                    eprintln!("[MINER-SSE] Stream error: {}", e);
                    break;
                }
                None => {
                    eprintln!("[MINER-SSE] Stream ended");
                    break;
                }
            }
        }

        // v1.0.2: Drop stream explicitly to trigger clean TCP close (prevent FIN-WAIT-1 zombies)
        drop(stream);

        if is_running.load(Ordering::SeqCst) {
            if !use_fallback {
                primary_fail_count += 1;
                if primary_fail_count >= 3 {
                    eprintln!("[MINER-SSE] Primary failed {} times, switching to fallback", primary_fail_count);
                    use_fallback = true;
                    primary_fail_count = 0;
                }
            } else {
                primary_fail_count += 1;
                if primary_fail_count >= 3 {
                    eprintln!("[MINER-SSE] Fallback failed, retrying primary...");
                    use_fallback = false;
                    primary_fail_count = 0;
                }
            }
            tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
        }
    }

    eprintln!("[MINER-SSE] Listener stopped");
}

/// v8.5.2: Fetch challenge with automatic fallback to quillon.xyz.
/// Ported from standalone miner's fetch_with_fallback pattern.
async fn fetch_challenge_with_fallback(
    api_client: &ApiClient,
    server_url: &str,
) -> Result<crate::models::MiningChallenge, String> {
    // Try primary server first
    match api_client.get_mining_challenge().await {
        Ok(c) => return Ok(c),
        Err(e) => {
            let msg = e.to_string();
            // Don't fallback for auth/sync errors — those are local issues
            if msg.contains("503") || msg.contains("SERVICE_UNAVAILABLE")
                || msg.contains("No peers") || msg.contains("discovering") {
                return Err(msg);
            }
            eprintln!("[MINER] Primary challenge failed ({}), trying fallback...", msg);
        }
    }

    // Fallback: direct GET to quillon.xyz (no auth needed for challenge)
    let fallback_url = format!("{}/api/v1/mining/challenge", FALLBACK_BOOTSTRAP_URL);
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new());

    match client.get(&fallback_url).send().await {
        Ok(resp) if resp.status().is_success() => {
            #[derive(serde::Deserialize)]
            struct Wrapper { data: Option<crate::models::MiningChallenge> }
            match resp.json::<Wrapper>().await {
                Ok(w) if w.data.is_some() => {
                    eprintln!("[MINER] Using fallback challenge from {}", FALLBACK_BOOTSTRAP_URL);
                    Ok(w.data.unwrap())
                }
                _ => Err("Fallback: no data in response".to_string()),
            }
        }
        Ok(resp) => Err(format!("Fallback HTTP {}", resp.status())),
        Err(e) => Err(format!("Fallback unreachable: {}", e)),
    }
}

/// v8.5.2: Submit solution with automatic fallback.
/// If primary fails, tries fallback server to avoid missing block rewards.
/// Public so GPU miner can also use it.
pub(crate) async fn submit_with_fallback(
    api_client: &ApiClient,
    submission: &MiningSubmission,
    server_url: &str,
) -> Result<serde_json::Value, String> {
    // Try primary
    match api_client.submit_mining_solution(submission).await {
        Ok(resp) => return Ok(resp),
        Err(e) => {
            eprintln!("[MINER] Primary submit failed: {} — trying fallback", e);
        }
    }

    // Fallback: POST directly to quillon.xyz
    let fallback_url = format!("{}/api/v1/mining/submit", FALLBACK_BOOTSTRAP_URL);
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new());

    match client.post(&fallback_url).json(submission).send().await {
        Ok(resp) if resp.status().is_success() => {
            eprintln!("[MINER] Fallback submit succeeded!");
            resp.json().await.map_err(|e| format!("Fallback parse: {}", e))
        }
        Ok(resp) => {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            Err(format!("Fallback HTTP {}: {}", status, body))
        }
        Err(e) => Err(format!("Fallback unreachable: {}", e)),
    }
}

/// Start the mining loop with multiple threads (one per CPU core).
/// v8.5.2: Major networking improvements ported from standalone miner:
///   - SSE new-block listener for instant challenge refresh (no more 10s stale work)
///   - Fallback server (quillon.xyz) for challenge fetch + solution submit
///   - Stale work detection every 4096 hashes (from standalone miner v7.4.3)
///   - Thread-local hash counting (batch 1024 before atomic add)
///   - TCP keepalive on HTTP client (15s, prevents socket exhaustion)
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

    // v8.5.2: New-block signal from SSE — mining threads check this to abandon stale work
    let new_block_signal = Arc::new(AtomicU64::new(0));

    // v8.5.2: Spawn SSE listener for instant new-block notifications
    {
        let wallet = miner_address.clone();
        let server_url = api_client.base_url().to_string();
        let is_running = Arc::new(AtomicBool::new(true));
        let signal = new_block_signal.clone();
        let state_clone = state.clone();
        // Store is_running so we can stop SSE when mining stops
        let is_running_for_stop = is_running.clone();

        rt.spawn(async move {
            start_sse_listener(wallet, server_url, is_running, signal, state_clone).await;
        });

        // Monitor mining state and stop SSE when mining stops
        let state_monitor = state.clone();
        rt.spawn(async move {
            loop {
                tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                if !state_monitor.running.load(Ordering::SeqCst) {
                    is_running_for_stop.store(false, Ordering::SeqCst);
                    break;
                }
            }
        });
    }

    // Coordinator thread: fetches challenges, aggregates hashrate
    // v8.5.2: Thread 0 refreshes every 50s (like standalone miner), others rely on SSE signal.
    // Also refreshes immediately on new_block_signal change.
    {
        let state = state.clone();
        let api_client = api_client.clone();
        let rt = rt.clone();
        let challenge = challenge.clone();
        let thread_hashes = thread_hashes.clone();
        let new_block_signal = new_block_signal.clone();

        std::thread::Builder::new()
            .name("miner-coordinator".into())
            .spawn(move || {
                let mut last_hashrate_calc = std::time::Instant::now();
                let mut consecutive_errors = 0u32;
                let mut last_height = 0u64;
                let mut last_block_signal = 0u64;
                let server_url = api_client.base_url().to_string();

                while state.running.load(Ordering::SeqCst) {
                    // Fetch new challenge (with fallback)
                    let client = api_client.clone();
                    match rt.block_on(fetch_challenge_with_fallback(&client, &server_url)) {
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
                        Err(msg) => {
                            consecutive_errors += 1;
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

                    // v8.5.2: Wait up to 50s between challenge fetches (like standalone miner thread 0),
                    // but wake up immediately if SSE signals a new block.
                    // Update hashrate every second within the wait loop.
                    let wait_secs = if consecutive_errors > 5 { 15 } else { 50 };
                    for _tick in 0..wait_secs {
                        if !state.running.load(Ordering::Relaxed) {
                            return;
                        }
                        std::thread::sleep(std::time::Duration::from_secs(1));

                        // Check if SSE signaled a new block — refresh challenge immediately
                        let current_signal = new_block_signal.load(Ordering::Relaxed);
                        if current_signal != last_block_signal {
                            last_block_signal = current_signal;
                            break; // Exit wait loop to fetch new challenge now
                        }

                        // Sum all thread hash counters for hashrate (every second)
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

    // Start GPU mining threads (share same challenge RwLock + new_block_signal as CPU)
    #[cfg(feature = "gpu-opencl")]
    {
        gpu_miner::start_gpu_mining(
            state.clone(),
            api_client.clone(),
            miner_address.clone(),
            rt.clone(),
            pool_mode,
            challenge.clone(),
            new_block_signal.clone(),
        );
    }

    // Spawn CPU mining worker threads
    for thread_id in 0..num_threads {
        let state = state.clone();
        let api_client = api_client.clone();
        let miner_address = miner_address.clone();
        let rt = rt.clone();
        let challenge = challenge.clone();
        let my_counter = thread_hashes.clone();
        let new_block_signal = new_block_signal.clone();

        std::thread::Builder::new()
            .name(format!("miner-{}", thread_id))
            .spawn(move || {
                pin_thread_to_core(thread_id);

                let mut nonce: u64 = rand::random::<u64>().wrapping_add(thread_id as u64 * 1_000_000_000);
                const BATCH_SIZE: u64 = 10_000;
                let server_url = api_client.base_url().to_string();

                // v8.5.2: Track last known block signal for stale work detection
                let mut last_known_block_signal = new_block_signal.load(Ordering::Relaxed);

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

                    // Update signal baseline when reading fresh challenge
                    last_known_block_signal = new_block_signal.load(Ordering::Relaxed);

                    // v8.5.2: Thread-local hash counting — batch 1024 before atomic add
                    // Reduces cache-line contention in multi-thread scenarios
                    let mut local_hash_count: u64 = 0;

                    // Mine a batch — tight inner loop
                    for i in 0..BATCH_SIZE {
                        let hash = mine_hash(&challenge_bytes, nonce);
                        local_hash_count += 1;

                        // v8.5.2: Batch atomic update every 1024 hashes (from standalone miner)
                        if local_hash_count >= 1024 {
                            my_counter[thread_id].fetch_add(local_hash_count, Ordering::Relaxed);
                            local_hash_count = 0;
                        }

                        // v8.5.2 (from standalone v7.4.3): Check for new block every 4096 hashes
                        // Before: only checked between batches (up to 7s of wasted work)
                        // After: max ~2ms wasted on stale challenge
                        if i & 4095 == 0 && i > 0 {
                            let sig = new_block_signal.load(Ordering::Relaxed);
                            if sig != last_known_block_signal {
                                break; // New block arrived — refresh challenge immediately
                            }
                        }

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
                                // v1.0.5: Genus-2 VDF fields (None until activation height reached)
                                vdf_output: None,
                                vdf_proof: None,
                                vdf_checkpoints: None,
                                vdf_iterations_count: None,
                            };

                            eprintln!("[MINER] Found valid hash! Nonce: {} — submitting...", nonce);

                            // v8.5.2: Submit with fallback (from standalone miner)
                            let client = api_client.clone();
                            let sub = submission.clone();
                            let srv = server_url.clone();
                            match rt.block_on(submit_with_fallback(&client, &sub, &srv)) {
                                Ok(resp) => {
                                    eprintln!("[MINER] Block accepted! Response: {:?}", resp);
                                    state.blocks_found.fetch_add(1, Ordering::SeqCst);
                                    state.set_status("Block found!");
                                }
                                Err(msg) => {
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

                    // Flush remaining local hash count
                    if local_hash_count > 0 {
                        my_counter[thread_id].fetch_add(local_hash_count, Ordering::Relaxed);
                    }
                }
            })
            .expect("failed to spawn mining thread");
    }
}

/// Stop the mining loop (CPU + GPU).
pub fn stop_mining(state: &MinerState) {
    state.running.store(false, Ordering::SeqCst);
    state.set_status("Stopped");
    #[cfg(feature = "gpu-opencl")]
    crate::gpu_miner::stop_gpu_mining(state);
}
