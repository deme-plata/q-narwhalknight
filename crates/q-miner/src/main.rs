use anyhow::Result;
use clap::Parser;
use console::style;
use std::sync::{Arc, atomic::{AtomicU64, AtomicBool, Ordering}};
use tokio::signal;
use tracing::{error, info, warn};
use chrono::{DateTime, Utc};

// Simplified command-line arguments
#[derive(Parser)]
#[command(name = "q-miner")]
#[command(about = "Q-NarwhalKnight High-Performance Miner")]
#[command(version = "1.0.0")]
struct Args {
    /// Mining mode: solo, pool, benchmark
    #[arg(short, long, default_value = "benchmark")]
    mode: String,

    /// Wallet address for mining rewards
    #[arg(short, long)]
    wallet: Option<String>,

    /// Number of CPU threads (0 = auto-detect)
    #[arg(short, long, default_value = "0")]
    threads: usize,

    /// Enable GPU mining
    #[arg(long)]
    gpu: bool,

    /// Mining intensity (1-10)
    #[arg(short, long, default_value = "7")]
    intensity: u8,

    /// Enable benchmarking mode
    #[arg(long)]
    benchmark: bool,

    /// Duration in seconds for benchmark
    #[arg(long, default_value = "30")]
    duration: u64,

    /// API server URL (e.g., http://185.182.185.227:8080)
    #[arg(short, long, default_value = "http://localhost:8080")]
    server: String,
}

// Simplified hardware info structure
pub struct HardwareInfo {
    pub cpu_cores: usize,
    pub cpu_threads: usize,
    pub cuda_devices: usize,
    pub opencl_devices: usize,
}

// Mining challenge from API server
#[derive(Debug, Clone, serde::Deserialize)]
pub struct MiningChallenge {
    pub challenge_hash: String,
    pub difficulty_target: String,
    pub block_height: u64,
    pub vdf_iterations: u32,
    pub block_reward: f64,
    pub expires_at: DateTime<Utc>,
}

// API response wrapper
#[derive(Debug, serde::Deserialize)]
struct ApiResponse<T> {
    success: bool,
    data: Option<T>,
    error: Option<String>,
}

/// Helper function to normalize server URL (remove trailing slash)
fn normalize_server_url(url: &str) -> String {
    url.trim_end_matches('/').to_string()
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("q_miner=info,q_dag_knight=info")
        .init();

    let args = Args::parse();

    // Print banner
    print_banner();

    // Hardware detection
    info!("🔍 Detecting hardware capabilities...");
    let hardware_info = detect_hardware().await?;

    println!("{}", style("💻 Hardware Detection Results:").cyan().bold());
    println!(
        "   CPU: {} cores, {} threads",
        hardware_info.cpu_cores, hardware_info.cpu_threads
    );

    if hardware_info.cuda_devices > 0 {
        println!("   CUDA: {} devices detected", hardware_info.cuda_devices);
    }
    if hardware_info.opencl_devices > 0 {
        println!(
            "   OpenCL: {} devices detected",
            hardware_info.opencl_devices
        );
    }

    // Determine mining configuration
    let cpu_threads = if args.threads == 0 {
        hardware_info.cpu_threads
    } else {
        args.threads
    };

    if args.mode == "benchmark" || args.benchmark {
        info!("🏁 Running benchmark mode for {} seconds...", args.duration);
        run_benchmark(cpu_threads, args.intensity, args.duration).await?;
    } else {
        // Validate wallet address for non-benchmark modes
        let wallet = match args.wallet {
            Some(w) => w,
            None => {
                error!("❌ Wallet address required for {} mode. Use --wallet <address>", args.mode);
                std::process::exit(1);
            }
        };

        // Validate wallet format - support both QUG (qnk + 64 hex) and AQUA (qnka + 62 hex) wallets
        let is_qug_wallet = wallet.starts_with("qnk") && wallet.len() == 67;
        let is_aqua_wallet = wallet.starts_with("qnka") && wallet.len() == 66;

        if !is_qug_wallet && !is_aqua_wallet {
            error!("❌ Invalid wallet address format.");
            error!("   QUG wallet: 'qnk' + 64 hex chars (67 total)");
            error!("   AQUA wallet: 'qnka' + 62 hex chars (66 total)");
            std::process::exit(1);
        }

        info!("⛏️  Starting Q-NarwhalKnight mining...");
        info!("💰 Mining to wallet: {}", wallet);
        info!("🌐 Connecting to server: {}", args.server);
        run_mining(cpu_threads, args.intensity, args.gpu, &wallet, &args.server).await?;
    }

    Ok(())
}

async fn detect_hardware() -> Result<HardwareInfo> {
    let cpu_cores = num_cpus::get_physical();
    let cpu_threads = num_cpus::get();
    
    // Simplified GPU detection (placeholder)
    let cuda_devices = if cfg!(feature = "cuda-mining") { 1 } else { 0 };
    let opencl_devices = if cfg!(feature = "opencl-mining") { 1 } else { 0 };
    
    Ok(HardwareInfo {
        cpu_cores,
        cpu_threads,
        cuda_devices,
        opencl_devices,
    })
}

async fn run_benchmark(threads: usize, intensity: u8, duration: u64) -> Result<()> {
    let hash_counter = Arc::new(AtomicU64::new(0));
    let is_running = Arc::new(AtomicBool::new(true));
    
    let start_time = std::time::Instant::now();
    let benchmark_duration = std::time::Duration::from_secs(duration);
    
    info!("🔥 Starting {} mining threads for benchmark", threads);
    
    let handles: Vec<_> = (0..threads)
        .map(|thread_id| {
            let hash_counter = hash_counter.clone();
            let is_running = is_running.clone();
            
            tokio::spawn(async move {
                benchmark_mining_thread(thread_id, hash_counter, is_running, intensity, benchmark_duration).await
            })
        })
        .collect();
    
    // Wait for benchmark completion
    for handle in handles {
        let _ = handle.await;
    }
    
    let elapsed = start_time.elapsed();
    let total_hashes = hash_counter.load(Ordering::Relaxed);
    let hash_rate = total_hashes as f64 / elapsed.as_secs_f64();
    
    info!("🏁 Benchmark Results:");
    info!("   Duration: {:.2}s", elapsed.as_secs_f64());
    info!("   Total Hashes: {}", total_hashes);
    info!("   Hash Rate: {:.2} H/s", hash_rate);
    info!("   Per Thread: {:.2} H/s", hash_rate / threads as f64);
    
    println!("\n{}", style("🎯 Q-NarwhalKnight Mining Benchmark Complete!").green().bold());
    println!("📊 Final Hash Rate: {:.2} H/s ({:.2} MH/s)", hash_rate, hash_rate / 1_000_000.0);
    
    Ok(())
}

async fn run_mining(threads: usize, intensity: u8, gpu_enabled: bool, wallet: &str, server_url: &str) -> Result<()> {
    let hash_counter = Arc::new(AtomicU64::new(0));
    let is_running = Arc::new(AtomicBool::new(true));
    let wallet = wallet.to_string();
    let server_url = server_url.to_string();

    // CRITICAL FIX: Shared signal for when a new block is produced
    // All mining threads will check this and immediately fetch new challenge
    let new_block_signal = Arc::new(AtomicU64::new(0)); // Increments when new block arrives

    // Shared current hashrate for network statistics (in KH/s for compatibility with API)
    let current_hashrate_khs = Arc::new(tokio::sync::RwLock::new(0.0f64));

    info!("🔥 Starting {} CPU mining threads", threads);

    let handles: Vec<_> = (0..threads)
        .map(|thread_id| {
            let hash_counter = hash_counter.clone();
            let is_running = is_running.clone();
            let wallet = wallet.clone();
            let server_url = server_url.clone();
            let new_block_signal = new_block_signal.clone();
            let hashrate_khs = current_hashrate_khs.clone();

            tokio::spawn(async move {
                mining_thread(thread_id, hash_counter, is_running, intensity, wallet, server_url, new_block_signal, hashrate_khs).await
            })
        })
        .collect();

    // Start hash rate monitor
    let monitor_counter = hash_counter.clone();
    let monitor_running = is_running.clone();
    let monitor_hashrate = current_hashrate_khs.clone();
    let monitor_handle = tokio::spawn(async move {
        hash_rate_monitor(monitor_counter, monitor_running, monitor_hashrate).await;
    });

    // Start SSE listener for real-time mining rewards AND new blocks
    let sse_wallet = wallet.clone();
    let sse_server_url = server_url.clone();
    let sse_running = is_running.clone();
    let sse_new_block_signal = new_block_signal.clone();
    let sse_handle = tokio::spawn(async move {
        start_sse_listener(sse_wallet, sse_server_url, sse_running, sse_new_block_signal).await;
    });

    if gpu_enabled {
        info!("🚀 GPU mining would be enabled (placeholder)");
    }

    info!("✅ Q-NarwhalKnight miner started successfully!");
    info!("🎧 Connected to SSE stream for real-time block updates");
    info!("Press Ctrl+C to stop mining...");

    // Wait for shutdown signal
    signal::ctrl_c().await?;

    info!("🛑 Shutdown signal received, stopping mining...");
    is_running.store(false, Ordering::SeqCst);

    // Wait for all threads to stop
    for handle in handles {
        let _ = handle.await;
    }
    monitor_handle.abort();
    sse_handle.abort();

    let total_hashes = hash_counter.load(Ordering::Relaxed);
    info!("👋 Q-NarwhalKnight miner stopped. Total hashes: {}", total_hashes);

    Ok(())
}

async fn benchmark_mining_thread(
    thread_id: usize,
    hash_counter: Arc<AtomicU64>,
    is_running: Arc<AtomicBool>,
    intensity: u8,
    duration: std::time::Duration,
) {
    let start_time = std::time::Instant::now();
    let mut nonce = thread_id as u64 * 10_000;
    let batch_size = (intensity as u64) * 1000;
    
    while start_time.elapsed() < duration && is_running.load(Ordering::SeqCst) {
        // Mine a batch of nonces using DAG-Knight VDF algorithm
        for _ in 0..batch_size {
            let _hash = compute_dag_knight_hash(&[0u8; 32], nonce);
            hash_counter.fetch_add(1, Ordering::Relaxed);
            nonce += 1;
        }
    }
    
    info!("🛑 Benchmark thread {} completed", thread_id);
}

async fn mining_thread(
    thread_id: usize,
    hash_counter: Arc<AtomicU64>,
    is_running: Arc<AtomicBool>,
    intensity: u8,
    wallet: String,
    server_url: String,
    new_block_signal: Arc<AtomicU64>,
    current_hashrate_khs: Arc<tokio::sync::RwLock<f64>>,
) {
    info!("🔥 CPU mining thread {} started", thread_id);

    let mut nonce = thread_id as u64 * 1_000_000;
    // OPTIMIZED: Increased batch size for maximum CPU utilization (10x increase)
    // Larger batches = fewer context switches = more CPU time spent hashing
    let batch_size = (intensity as u64) * 100_000; // Was 10_000, now 100_000 for 99% CPU usage
    let api_url = &server_url;

    let client = reqwest::Client::new();

    // Check if server is syncing before starting to mine
    match check_server_sync_status(api_url).await {
        Ok((is_syncing, blocks_behind)) if is_syncing => {
            info!("⏸️  Thread {} waiting: Server is syncing ({} blocks behind network)", thread_id, blocks_behind);
            info!("   Mining will start automatically when sync is complete");
            // Wait for sync to complete before fetching challenge
            loop {
                tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
                if !is_running.load(Ordering::SeqCst) {
                    return;
                }
                match check_server_sync_status(api_url).await {
                    Ok((false, _)) => {
                        info!("✅ Thread {} detected sync complete - starting mining", thread_id);
                        break;
                    }
                    Ok((true, behind)) => {
                        info!("⏸️  Thread {} still waiting: {} blocks behind", thread_id, behind);
                    }
                    Err(_) => {
                        // Connection error, will retry
                    }
                }
            }
        }
        Ok(_) => {
            // Not syncing, proceed normally
        }
        Err(e) => {
            warn!("⚠️  Thread {} couldn't check sync status: {} - proceeding anyway", thread_id, e);
        }
    }

    // Fetch initial mining challenge
    let mut current_challenge = match fetch_mining_challenge(api_url).await {
        Ok(challenge) => {
            info!("📋 Thread {} fetched challenge: block #{}, reward: {} QNK",
                 thread_id, challenge.block_height, challenge.block_reward);
            challenge
        }
        Err(e) => {
            error!("❌ Thread {} failed to fetch initial challenge: {}", thread_id, e);
            error!("   Make sure q-api-server is running on {}", api_url);
            return;
        }
    };

    let mut challenge_hash = match hex_to_bytes(&current_challenge.challenge_hash) {
        Ok(hash) => hash,
        Err(e) => {
            error!("❌ Thread {} failed to decode challenge hash: {}", thread_id, e);
            return;
        }
    };

    let mut target = match hex_to_bytes(&current_challenge.difficulty_target) {
        Ok(t) => t,
        Err(e) => {
            error!("❌ Thread {} failed to decode difficulty target: {}", thread_id, e);
            return;
        }
    };

    let mut last_challenge_refresh = std::time::Instant::now();
    let challenge_refresh_interval = std::time::Duration::from_secs(50); // Refresh before 60s expiry
    let mut last_known_block_signal = new_block_signal.load(Ordering::Relaxed);

    while is_running.load(Ordering::SeqCst) {
        // CRITICAL FIX: Check if new block arrived via SSE
        let current_block_signal = new_block_signal.load(Ordering::Relaxed);
        let should_refresh_immediately = current_block_signal != last_known_block_signal;

        // Refresh challenge if expired, near expiration, OR new block arrived
        if should_refresh_immediately || last_challenge_refresh.elapsed() >= challenge_refresh_interval {
            match fetch_mining_challenge(api_url).await {
                Ok(new_challenge) => {
                    if new_challenge.block_height != current_challenge.block_height {
                        if should_refresh_immediately {
                            info!("🔄 Thread {} IMMEDIATELY updated challenge (new block signal): block #{} -> #{}",
                                 thread_id, current_challenge.block_height, new_challenge.block_height);
                        } else {
                            info!("🔄 Thread {} updated challenge (periodic): block #{} -> #{}",
                                 thread_id, current_challenge.block_height, new_challenge.block_height);
                        }
                    }
                    current_challenge = new_challenge;

                    // Decode new challenge hash and difficulty
                    if let Ok(hash) = hex_to_bytes(&current_challenge.challenge_hash) {
                        challenge_hash = hash;
                    }
                    if let Ok(t) = hex_to_bytes(&current_challenge.difficulty_target) {
                        target = t;
                    }

                    last_challenge_refresh = std::time::Instant::now();
                    last_known_block_signal = current_block_signal;
                }
                Err(e) => {
                    warn!("⚠️  Thread {} failed to refresh challenge: {}", thread_id, e);
                    // Continue with existing challenge
                }
            }
        }

        // Mine a batch of nonces with MAXIMUM CPU utilization
        // Pre-allocate buffer for hash input (40 bytes: 32 for challenge + 8 for nonce)
        let mut hash_input = [0u8; 40];
        hash_input[..32].copy_from_slice(&challenge_hash);

        for _ in 0..batch_size {
            // Update nonce in pre-allocated buffer (zero-copy, maximum performance)
            hash_input[32..].copy_from_slice(&nonce.to_le_bytes());

            let hash = compute_dag_knight_hash_optimized(&hash_input);
            hash_counter.fetch_add(1, Ordering::Relaxed);

            // Check if solution meets difficulty target
            if hash < target {
                // Clean output: just show the solution found without verbose hash bytes
                info!("💎 Solution found! Block #{}, Thread {}",
                     current_challenge.block_height, thread_id);

                // Get current hashrate for network statistics
                let hashrate_khs = *current_hashrate_khs.read().await;

                // Submit solution to the network with challenge_hash for server-side verification
                let solution = serde_json::json!({
                    "miner_address": wallet,
                    "nonce": nonce,
                    "hash": hex::encode(hash),
                    "difficulty_target": hex::encode(target),
                    "challenge_hash": hex::encode(challenge_hash),
                    "hash_rate": hashrate_khs  // Send hashrate in KH/s for network statistics
                });

                // CRITICAL: Submit solution in background to avoid blocking mining thread
                // The mining thread must continue immediately to maintain hash rate
                let normalized_url = normalize_server_url(api_url);
                let submit_url = format!("{}/api/v1/mining/submit", normalized_url);
                let client_clone = client.clone();
                tokio::spawn(async move {
                    match client_clone.post(&submit_url)
                        .json(&solution)
                        .send()
                        .await
                    {
                        Ok(resp) => {
                            if resp.status().is_success() {
                                if let Ok(result) = resp.json::<serde_json::Value>().await {
                                    if let Some(data) = result.get("data") {
                                        if let Some(reward) = data.get("reward_qnk") {
                                            info!("✅ Solution accepted! Earned {} QNK", reward);
                                        }
                                    }
                                }
                            } else {
                                warn!("❌ Solution rejected: HTTP {}", resp.status());
                            }
                        }
                        Err(e) => {
                            warn!("Failed to submit solution: {}", e);
                        }
                    }
                });
            }

            nonce += 1;
        }

        // REMOVED: tokio::task::yield_now() - This was throttling CPU usage!
        // Mining thread now runs at 100% CPU utilization for maximum performance
    }

    info!("🛑 CPU mining thread {} stopped", thread_id);
}

async fn hash_rate_monitor(
    hash_counter: Arc<AtomicU64>,
    is_running: Arc<AtomicBool>,
    current_hashrate_khs: Arc<tokio::sync::RwLock<f64>>,
) {
    let mut last_hash_count = 0u64;
    let mut last_time = std::time::Instant::now();
    let start_time = std::time::Instant::now();

    while is_running.load(Ordering::SeqCst) {
        tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;

        let current_hash_count = hash_counter.load(Ordering::Relaxed);
        let current_time = std::time::Instant::now();

        let hashes_computed = current_hash_count - last_hash_count;
        let time_elapsed = current_time.duration_since(last_time).as_secs_f64();

        if time_elapsed > 0.0 {
            let hash_rate = hashes_computed as f64 / time_elapsed;
            let hash_rate_khs = hash_rate / 1_000.0; // Convert H/s to KH/s
            let tpm = (hash_rate * 60.0) / 1_000_000.0; // Tasks Per Minute in millions
            let uptime = current_time.duration_since(start_time).as_secs();
            let uptime_mins = uptime / 60;
            let uptime_secs = uptime % 60;

            // Update shared hashrate for network statistics
            *current_hashrate_khs.write().await = hash_rate_khs;

            // Clean status bar format
            info!("⛏️  Mining │ {:.2} MH/s │ {:.2}M TPM │ Uptime: {}m {}s │ Total: {:.2}M hashes",
                 hash_rate / 1_000_000.0, tpm, uptime_mins, uptime_secs, current_hash_count as f64 / 1_000_000.0);
        }

        last_hash_count = current_hash_count;
        last_time = current_time;
    }
}

/// SSE listener for real-time mining rewards AND new block notifications
async fn start_sse_listener(
    wallet: String,
    server_url: String,
    is_running: Arc<AtomicBool>,
    new_block_signal: Arc<AtomicU64>,
) {
    use eventsource_client::{self as eventsource, Client as _};
    use futures::StreamExt;

    // Normalize URL to prevent double slashes
    let normalized_url = normalize_server_url(&server_url);

    // Include wallet_address parameter for filtered SSE events
    let url = format!("{}/api/v1/events?wallet_address={}", normalized_url, wallet);

    loop {
        if !is_running.load(Ordering::SeqCst) {
            break;
        }

        let client = match eventsource::ClientBuilder::for_url(&url) {
            Ok(builder) => builder.build(),
            Err(e) => {
                warn!("Failed to create SSE client: {}", e);
                tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
                continue;
            }
        };

        let mut stream = client.stream();

        info!("🎧 Connected to SSE stream at {}", url);

        while is_running.load(Ordering::SeqCst) {
            match stream.next().await {
                Some(Ok(eventsource::SSE::Event(ev))) => {
                    // CRITICAL FIX: Handle new-block events for immediate challenge refresh
                    if ev.event_type == "new-block" {
                        match serde_json::from_str::<serde_json::Value>(&ev.data) {
                            Ok(data) => {
                                if let Some(block_height) = data.get("height").and_then(|v| v.as_u64()) {
                                    // Increment signal to notify all mining threads
                                    let new_signal = new_block_signal.fetch_add(1, Ordering::SeqCst) + 1;
                                    info!("🔔 NEW BLOCK #{} detected via SSE - signaling mining threads (signal: {})",
                                         block_height, new_signal);
                                }
                            }
                            Err(e) => {
                                warn!("Failed to parse new-block event: {}", e);
                            }
                        }
                    }

                    // Handle mining_reward events
                    if ev.event_type == "mining_reward" {
                        match serde_json::from_str::<serde_json::Value>(&ev.data) {
                            Ok(data) => {
                                if let Some(miner_address) = data.get("miner_address").and_then(|v| v.as_str()) {
                                    if miner_address == wallet {
                                        let reward_qnk = data.get("reward_qnk")
                                            .and_then(|v| v.as_f64())
                                            .unwrap_or(0.0);
                                        let block_height = data.get("block_height")
                                            .and_then(|v| v.as_u64())
                                            .unwrap_or(0);
                                        let nonce = data.get("nonce")
                                            .and_then(|v| v.as_u64())
                                            .unwrap_or(0);

                                        // Display celebratory reward notification
                                        info!("");
                                        info!("╔═══════════════════════════════════════════════════╗");
                                        info!("║   💎 MINING REWARD RECEIVED!                      ║");
                                        info!("╠═══════════════════════════════════════════════════╣");
                                        info!("║   Reward: {:<40} ║", format!("{:.8} QNK", reward_qnk));
                                        info!("║   Block:  {:<40} ║", format!("#{}", block_height));
                                        info!("║   Nonce:  {:<40} ║", nonce);
                                        info!("╚═══════════════════════════════════════════════════╝");
                                        info!("");
                                    }
                                }
                            }
                            Err(e) => {
                                warn!("Failed to parse mining_reward event: {}", e);
                            }
                        }
                    }

                    // Handle balance_updated events
                    if ev.event_type == "balance_updated" {
                        match serde_json::from_str::<serde_json::Value>(&ev.data) {
                            Ok(data) => {
                                if let Some(wallet_address) = data.get("wallet_address").and_then(|v| v.as_str()) {
                                    if wallet_address == wallet {
                                        if let Some(change_reason) = data.get("change_reason").and_then(|v| v.as_str()) {
                                            if change_reason == "mining_reward" {
                                                let new_balance = data.get("new_balance")
                                                    .and_then(|v| v.as_f64())
                                                    .unwrap_or(0.0);
                                                info!("💰 Balance Updated: {:.8} QNK", new_balance);
                                            }
                                        }
                                    }
                                }
                            }
                            Err(e) => {
                                warn!("Failed to parse balance_updated event: {}", e);
                            }
                        }
                    }
                }
                Some(Ok(eventsource::SSE::Comment(_))) => {
                    // Ignore comments
                }
                Some(Err(e)) => {
                    warn!("SSE stream error: {}", e);
                    break;
                }
                None => {
                    warn!("SSE stream ended");
                    break;
                }
            }
        }

        // Reconnect after delay if still running
        if is_running.load(Ordering::SeqCst) {
            warn!("Reconnecting to SSE stream in 5 seconds...");
            tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
        }
    }

    info!("🛑 SSE listener stopped");
}

/// Check if server is currently syncing (returns is_syncing, blocks_behind)
async fn check_server_sync_status(api_url: &str) -> Result<(bool, u64)> {
    let client = reqwest::Client::new();
    let normalized_url = normalize_server_url(api_url);
    let url = format!("{}/api/v1/status", normalized_url);

    let response = client.get(&url).send().await?;
    let api_response: ApiResponse<serde_json::Value> = response.json().await?;

    if !api_response.success {
        return Err(anyhow::anyhow!("API error: {:?}", api_response.error));
    }

    let data = api_response.data.ok_or_else(|| anyhow::anyhow!("No data in response"))?;

    let is_syncing = data.get("is_syncing")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let blocks_behind = data.get("blocks_behind")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    Ok((is_syncing, blocks_behind))
}

/// Fetch current mining challenge from API server
async fn fetch_mining_challenge(api_url: &str) -> Result<MiningChallenge> {
    let client = reqwest::Client::new();
    // Normalize URL to prevent double slashes
    let normalized_url = normalize_server_url(api_url);
    let url = format!("{}/api/v1/mining/challenge", normalized_url);

    let response = client.get(&url)
        .send()
        .await?;

    if !response.status().is_success() {
        anyhow::bail!("Failed to fetch mining challenge: HTTP {}", response.status());
    }

    let api_response: ApiResponse<MiningChallenge> = response.json().await?;

    if !api_response.success {
        let error_msg = api_response.error.unwrap_or_else(|| "Unknown error".to_string());
        anyhow::bail!("API returned error: {}", error_msg);
    }

    api_response.data.ok_or_else(|| anyhow::anyhow!("Missing challenge data in API response"))
}

/// Decode hex string to byte array
fn hex_to_bytes(hex_str: &str) -> Result<[u8; 32]> {
    let bytes = hex::decode(hex_str)?;
    if bytes.len() != 32 {
        anyhow::bail!("Expected 32 bytes, got {}", bytes.len());
    }
    let mut result = [0u8; 32];
    result.copy_from_slice(&bytes);
    Ok(result)
}

/// DAG-Knight VDF mining algorithm
/// OPTIMIZED: Original hash function (kept for compatibility with old code)
fn compute_dag_knight_hash(input: &[u8; 32], nonce: u64) -> [u8; 32] {
    let mut hasher_input = [0u8; 40];
    hasher_input[..32].copy_from_slice(input);
    hasher_input[32..].copy_from_slice(&nonce.to_le_bytes());
    compute_dag_knight_hash_optimized(&hasher_input)
}

/// MAXIMUM PERFORMANCE: Zero-allocation hash function for 100% CPU utilization
/// Optimizations:
/// - Pre-allocated fixed-size arrays (no heap allocations)
/// - In-place VDF computation
/// - Optimized for CPU cache efficiency
#[inline(always)]
fn compute_dag_knight_hash_optimized(hash_input: &[u8; 40]) -> [u8; 32] {
    // Initial hash with zero allocations
    let initial_hash = blake3::hash(hash_input);

    // VDF computation with fixed buffer (100 iterations)
    // Using array instead of Vec for zero allocations
    let mut current = *initial_hash.as_bytes();

    for _ in 0..100 {
        // In-place hashing for maximum performance
        current = *blake3::hash(&current).as_bytes();
    }

    current
}

fn print_banner() {
    println!(
        "{}",
        style(
            "
██████╗     ███╗   ██╗ █████╗ ██████╗ ██╗    ██╗██╗  ██╗ █████╗ ██╗     
██╔═══██╗    ████╗  ██║██╔══██╗██╔══██╗██║    ██║██║  ██║██╔══██╗██║     
██║   ██║    ██╔██╗ ██║███████║██████╔╝██║ █╗ ██║███████║███████║██║     
██║▄▄ ██║    ██║╚██╗██║██╔══██║██╔══██╗██║███╗██║██╔══██║██╔══██║██║     
╚██████╔╝    ██║ ╚████║██║  ██║██║  ██║╚███╔███╔╝██║  ██║██║  ██║███████╗
 ╚══▀▀═╝     ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝
                                                                          
██╗  ██╗███╗   ██╗██╗ ██████╗ ██╗  ██╗████████╗                        
██║ ██╔╝████╗  ██║██║██╔════╝ ██║  ██║╚══██╔══╝                        
█████╔╝ ██╔██╗ ██║██║██║  ███╗███████║   ██║                           
██╔═██╗ ██║╚██╗██║██║██║   ██║██╔══██║   ██║                           
██║  ██╗██║ ╚████║██║╚██████╔╝██║  ██║   ██║                           
╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝                           
"
        )
        .green()
        .bold()
    );

    println!(
        "{}",
        style("    🌟 Quantum-Enhanced Anonymous Consensus Mining").cyan()
    );
    println!(
        "{}",
        style("    ⚛️  DAG-Knight • VDF-Secure • Production Ready").dim()
    );
    println!();
}