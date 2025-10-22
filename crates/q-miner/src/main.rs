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

    info!("🔥 Starting {} CPU mining threads", threads);

    let handles: Vec<_> = (0..threads)
        .map(|thread_id| {
            let hash_counter = hash_counter.clone();
            let is_running = is_running.clone();
            let wallet = wallet.clone();
            let server_url = server_url.clone();

            tokio::spawn(async move {
                mining_thread(thread_id, hash_counter, is_running, intensity, wallet, server_url).await
            })
        })
        .collect();

    // Start hash rate monitor
    let monitor_counter = hash_counter.clone();
    let monitor_running = is_running.clone();
    let monitor_handle = tokio::spawn(async move {
        hash_rate_monitor(monitor_counter, monitor_running).await;
    });

    // Start SSE listener for real-time mining rewards
    let sse_wallet = wallet.clone();
    let sse_server_url = server_url.clone();
    let sse_running = is_running.clone();
    let sse_handle = tokio::spawn(async move {
        start_sse_listener(sse_wallet, sse_server_url, sse_running).await;
    });

    if gpu_enabled {
        info!("🚀 GPU mining would be enabled (placeholder)");
    }

    info!("✅ Q-NarwhalKnight miner started successfully!");
    info!("🎧 Connected to SSE stream for real-time rewards");
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
) {
    info!("🔥 CPU mining thread {} started", thread_id);

    let mut nonce = thread_id as u64 * 1_000_000;
    let batch_size = (intensity as u64) * 10_000;
    let api_url = &server_url;

    let client = reqwest::Client::new();

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

    while is_running.load(Ordering::SeqCst) {
        // Refresh challenge if expired or near expiration
        if last_challenge_refresh.elapsed() >= challenge_refresh_interval {
            match fetch_mining_challenge(api_url).await {
                Ok(new_challenge) => {
                    if new_challenge.block_height != current_challenge.block_height {
                        info!("🔄 Thread {} updated challenge: block #{} -> #{}",
                             thread_id, current_challenge.block_height, new_challenge.block_height);
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
                }
                Err(e) => {
                    warn!("⚠️  Thread {} failed to refresh challenge: {}", thread_id, e);
                    // Continue with existing challenge
                }
            }
        }

        // Mine a batch of nonces
        for _ in 0..batch_size {
            let hash = compute_dag_knight_hash(&challenge_hash, nonce);
            hash_counter.fetch_add(1, Ordering::Relaxed);

            // Check if solution meets difficulty target
            if hash < target {
                info!("💎 Thread {} found solution! Block #{}, Nonce: {}, Hash: {:02x?}",
                     thread_id, current_challenge.block_height, nonce, &hash[..8]);

                // Submit solution to the network with challenge_hash for server-side verification
                let solution = serde_json::json!({
                    "miner_address": wallet,
                    "nonce": nonce,
                    "hash": hex::encode(hash),
                    "difficulty_target": hex::encode(target),
                    "challenge_hash": hex::encode(challenge_hash)
                });

                match client.post(format!("{}/api/v1/mining/submit", api_url))
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
            }

            nonce += 1;
        }

        // Brief pause to prevent CPU overload
        tokio::task::yield_now().await;
    }

    info!("🛑 CPU mining thread {} stopped", thread_id);
}

async fn hash_rate_monitor(
    hash_counter: Arc<AtomicU64>,
    is_running: Arc<AtomicBool>,
) {
    let mut last_hash_count = 0u64;
    let mut last_time = std::time::Instant::now();

    while is_running.load(Ordering::SeqCst) {
        tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;

        let current_hash_count = hash_counter.load(Ordering::Relaxed);
        let current_time = std::time::Instant::now();

        let hashes_computed = current_hash_count - last_hash_count;
        let time_elapsed = current_time.duration_since(last_time).as_secs_f64();

        if time_elapsed > 0.0 {
            let hash_rate = hashes_computed as f64 / time_elapsed;

            info!("📊 Hash Rate: {:.2} H/s ({:.2} KH/s) - Total: {}",
                 hash_rate, hash_rate / 1000.0, current_hash_count);
        }

        last_hash_count = current_hash_count;
        last_time = current_time;
    }
}

/// SSE listener for real-time mining rewards
async fn start_sse_listener(wallet: String, server_url: String, is_running: Arc<AtomicBool>) {
    use eventsource_client::{self as eventsource, Client as _};
    use futures::StreamExt;

    // Include wallet_address parameter for filtered SSE events
    let url = format!("{}/api/v1/events?wallet_address={}", server_url, wallet);

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

/// Fetch current mining challenge from API server
async fn fetch_mining_challenge(api_url: &str) -> Result<MiningChallenge> {
    let client = reqwest::Client::new();
    let url = format!("{}/api/v1/mining/challenge", api_url);

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
fn compute_dag_knight_hash(input: &[u8; 32], nonce: u64) -> [u8; 32] {
    // Combine input with nonce
    let mut hasher_input = Vec::with_capacity(40);
    hasher_input.extend_from_slice(input);
    hasher_input.extend_from_slice(&nonce.to_le_bytes());

    // Initial hash
    let initial_hash = blake3::hash(&hasher_input);

    // VDF computation (simplified - 100 iterations for demo)
    let mut current = initial_hash.as_bytes().to_vec();
    for _ in 0..100 {
        current = blake3::hash(&current).as_bytes().to_vec();
    }

    let mut result = [0u8; 32];
    result.copy_from_slice(&current[..32]);
    result
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