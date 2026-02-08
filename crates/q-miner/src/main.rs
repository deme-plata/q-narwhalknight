use anyhow::Result;
use clap::Parser;
use console::style;
use std::sync::{Arc, atomic::{AtomicU64, AtomicBool, Ordering}};
use tokio::signal;
use tokio::sync::mpsc;
use tracing::{error, info, warn};
use chrono::{DateTime, Utc};
use core_affinity::CoreId;
#[cfg(target_arch = "x86_64")]
use raw_cpuid::CpuId;
use serde_json::Value;

// Simplified command-line arguments
#[derive(Parser)]
#[command(name = "q-miner")]
#[command(about = "Q-NarwhalKnight High-Performance Miner")]
#[command(version = "2.3.0")]
struct Args {
    /// Mining mode: solo, pool, decentralized, benchmark
    /// - solo: Mine directly to your local node
    /// - pool: Connect to centralized pool via Stratum protocol
    /// - decentralized: P2P pool mining with CRDT-based PPLNS (v2.3.0+)
    /// - benchmark: Test hashrate without submitting solutions
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

    /// Mining pool URL for pool mode (e.g., stratum+tcp://pool.quillon.xyz:3333)
    #[arg(long, default_value = "stratum+tcp://pool.quillon.xyz:3333")]
    pool_url: String,

    /// Worker name for pool mining (default: randomly generated)
    #[arg(long)]
    worker_name: Option<String>,

    /// v3.3.3-beta: Human-readable miner name for identification (e.g., "Server Alpha", "Mining Rig 1")
    /// Shows up in server logs to help distinguish between multiple miners
    #[arg(long, short = 'n')]
    miner_name: Option<String>,

    /// P2P bootstrap nodes for decentralized pool mode (comma-separated)
    #[arg(long, default_value = "http://quillon.xyz:8080")]
    bootstrap_nodes: String,

    /// Region for pool node discovery (e.g., us-east, eu-west, asia-pacific)
    #[arg(long, default_value = "global")]
    region: String,
}

// Hardware info structure with CPU optimization details
pub struct HardwareInfo {
    pub cpu_cores: usize,
    pub cpu_threads: usize,
    pub cuda_devices: usize,
    pub opencl_devices: usize,
    pub cpu_vendor: String,
    pub has_avx2: bool,
    pub has_avx512: bool,
    pub cache_line_size: usize,
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

/// Default fallback bootstrap server
const FALLBACK_BOOTSTRAP_URL: &str = "https://bootstrap1.quillon.xyz";

/// Try an HTTP GET request against the primary server, falling back to bootstrap1.quillon.xyz
/// Returns (response_body, actual_url_used) on success.
async fn fetch_with_fallback(
    client: &reqwest::Client,
    primary_base: &str,
    path: &str,
) -> Result<(String, String)> {
    let primary_url = format!("{}{}", normalize_server_url(primary_base), path);
    match client.get(&primary_url).timeout(std::time::Duration::from_secs(10)).send().await {
        Ok(resp) if resp.status().is_success() => {
            let body = resp.text().await.unwrap_or_default();
            Ok((body, primary_base.to_string()))
        }
        Ok(resp) => {
            warn!("⚠️  Primary server returned HTTP {} - trying fallback {}", resp.status(), FALLBACK_BOOTSTRAP_URL);
            let fallback_url = format!("{}{}", FALLBACK_BOOTSTRAP_URL, path);
            let resp = client.get(&fallback_url).timeout(std::time::Duration::from_secs(10)).send().await?;
            let body = resp.text().await.unwrap_or_default();
            Ok((body, FALLBACK_BOOTSTRAP_URL.to_string()))
        }
        Err(e) => {
            warn!("⚠️  Primary server {} unreachable: {} - trying fallback {}", primary_base, e, FALLBACK_BOOTSTRAP_URL);
            let fallback_url = format!("{}{}", FALLBACK_BOOTSTRAP_URL, path);
            let resp = client.get(&fallback_url).timeout(std::time::Duration::from_secs(10)).send().await?;
            let body = resp.text().await.unwrap_or_default();
            Ok((body, FALLBACK_BOOTSTRAP_URL.to_string()))
        }
    }
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
        "   CPU: {} ({}) - {} cores, {} threads",
        hardware_info.cpu_vendor,
        if hardware_info.has_avx512 {
            "AVX-512"
        } else if hardware_info.has_avx2 {
            "AVX2"
        } else {
            "SSE"
        },
        hardware_info.cpu_cores,
        hardware_info.cpu_threads
    );
    println!(
        "   Cache Line: {} bytes │ SIMD: {} │ Server-Optimized: {}",
        hardware_info.cache_line_size,
        if hardware_info.has_avx512 { "AVX-512" } else if hardware_info.has_avx2 { "AVX2" } else { "SSE" },
        if hardware_info.cpu_cores >= 16 { "✅" } else { "⚠️ Desktop CPU" }
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

        if args.mode == "pool" {
            // Pool mining mode - connect via Stratum protocol
            let worker_name = args.worker_name.unwrap_or_else(|| {
                format!("worker_{:08x}", rand::random::<u32>())
            });
            info!("⛏️  Starting Q-NarwhalKnight POOL mining...");
            info!("💰 Mining to wallet: {}", wallet);
            info!("🏊 Pool URL: {}", args.pool_url);
            info!("👷 Worker name: {}", worker_name);
            run_pool_mining(cpu_threads, args.intensity, &wallet, &worker_name, &args.pool_url).await?;
        } else if args.mode == "decentralized" {
            // Decentralized P2P pool mining mode - v2.3.0+
            // Uses CRDT-based PPLNS with gossipsub coordination
            let worker_name = args.worker_name.unwrap_or_else(|| {
                format!("worker_{:08x}", rand::random::<u32>())
            });
            info!("🌐 Starting Q-NarwhalKnight DECENTRALIZED POOL mining...");
            info!("💰 Mining to wallet: {}", wallet);
            info!("👷 Worker name: {}", worker_name);
            info!("📡 Bootstrap nodes: {}", args.bootstrap_nodes);
            info!("🗺️  Region: {}", args.region);
            info!("");
            info!("📊 Features:");
            info!("   ✅ CRDT-based PPLNS - No central pool needed");
            info!("   ✅ P2P share propagation via gossipsub");
            info!("   ✅ VDF anti-grinding proofs");
            info!("   ✅ Threshold signature payouts");
            info!("");
            run_decentralized_pool_mining(
                cpu_threads,
                args.intensity,
                &wallet,
                &worker_name,
                &args.bootstrap_nodes,
                &args.region,
            ).await?;
        } else {
            // Solo mining mode - connect directly to API server
            info!("⛏️  Starting Q-NarwhalKnight SOLO mining...");
            info!("💰 Mining to wallet: {}", wallet);
            info!("🌐 Primary server: {}", args.server);
            info!("🔄 Fallback server: {}", FALLBACK_BOOTSTRAP_URL);
            if let Some(ref name) = args.miner_name {
                info!("🏷️  Miner name: {}", name);
            }
            run_mining(cpu_threads, args.intensity, args.gpu, &wallet, &args.server, args.miner_name.as_deref()).await?;
        }
    }

    Ok(())
}

async fn detect_hardware() -> Result<HardwareInfo> {
    let cpu_cores = num_cpus::get_physical();
    let cpu_threads = num_cpus::get();

    // Simplified GPU detection (placeholder)
    let cuda_devices = if cfg!(feature = "cuda-mining") { 1 } else { 0 };
    let opencl_devices = if cfg!(feature = "opencl-mining") { 1 } else { 0 };

    // Detect CPU features (architecture-specific)
    #[cfg(target_arch = "x86_64")]
    let (cpu_vendor, has_avx2, has_avx512, cache_line_size) = {
        let cpuid = CpuId::new();
        let vendor = cpuid.get_vendor_info()
            .map(|v| v.as_str().to_string())
            .unwrap_or_else(|| "Unknown".to_string());
        let extended_features = cpuid.get_extended_feature_info();
        let avx2 = extended_features.as_ref().map(|ef| ef.has_avx2()).unwrap_or(false);
        let avx512 = extended_features.as_ref().map(|ef| ef.has_avx512f()).unwrap_or(false);
        let cache = cpuid.get_cache_parameters()
            .and_then(|mut params| params.next())
            .map(|info| info.coherency_line_size() as usize)
            .unwrap_or(64);
        (vendor, avx2, avx512, cache)
    };

    #[cfg(not(target_arch = "x86_64"))]
    let (cpu_vendor, has_avx2, has_avx512, cache_line_size) = {
        let vendor = if cfg!(target_arch = "aarch64") {
            "ARM".to_string()
        } else {
            "Unknown".to_string()
        };
        (vendor, false, false, 64usize)
    };

    Ok(HardwareInfo {
        cpu_cores,
        cpu_threads,
        cuda_devices,
        opencl_devices,
        cpu_vendor,
        has_avx2,
        has_avx512,
        cache_line_size,
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

async fn run_mining(threads: usize, intensity: u8, gpu_enabled: bool, wallet: &str, server_url: &str, miner_name: Option<&str>) -> Result<()> {
    let hash_counter = Arc::new(AtomicU64::new(0));
    let is_running = Arc::new(AtomicBool::new(true));
    let wallet = wallet.to_string();
    let server_url = server_url.to_string();

    // v3.3.3-beta: Generate unique miner ID for this instance
    let miner_id = format!("{:016x}", rand::random::<u64>());
    let miner_name = miner_name.map(|s| s.to_string());
    info!("🆔 Miner ID: {}", miner_id);

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
            let miner_id = miner_id.clone();
            let miner_name = miner_name.clone();

            tokio::spawn(async move {
                mining_thread(thread_id, hash_counter, is_running, intensity, wallet, server_url, new_block_signal, hashrate_khs, miner_id, miner_name).await
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

/// Pool mining mode using Stratum protocol
async fn run_pool_mining(
    threads: usize,
    intensity: u8,
    wallet: &str,
    worker_name: &str,
    pool_url: &str,
) -> Result<()> {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    use tokio::net::TcpStream;
    use tokio::sync::mpsc;
    use serde_json::{json, Value};

    let hash_counter = Arc::new(AtomicU64::new(0));
    let is_running = Arc::new(AtomicBool::new(true));
    let current_difficulty = Arc::new(tokio::sync::RwLock::new(1.0_f64));

    // Current job state shared between threads
    let current_job = Arc::new(tokio::sync::RwLock::new(Option::<PoolJob>::None));
    let job_signal = Arc::new(AtomicU64::new(0));

    // Channel for submitting shares to the pool
    let (share_tx, mut share_rx) = mpsc::channel::<ShareToSubmit>(1000);

    // Parse pool URL: stratum+tcp://host:port
    let url = pool_url
        .trim_start_matches("stratum+tcp://")
        .trim_start_matches("stratum://")
        .trim_start_matches("tcp://");

    let parts: Vec<&str> = url.split(':').collect();
    let host = parts.get(0).copied().unwrap_or("pool.quillon.xyz");
    let port: u16 = parts.get(1).and_then(|p| p.parse().ok()).unwrap_or(3333);

    info!("🔌 Connecting to pool {}:{}", host, port);

    let addr = format!("{}:{}", host, port);
    let stream = match TcpStream::connect(&addr).await {
        Ok(s) => s,
        Err(e) => {
            error!("❌ Failed to connect to pool: {}", e);
            return Err(e.into());
        }
    };

    let (reader, mut writer) = stream.into_split();
    let mut reader = BufReader::new(reader);

    // Generate subscription ID
    let sub_id = format!("{:08x}", rand::random::<u32>());

    // Send mining.subscribe
    let subscribe_msg = json!({
        "id": 1,
        "method": "mining.subscribe",
        "params": ["q-miner/1.0.0", sub_id]
    });
    let msg_str = format!("{}\n", serde_json::to_string(&subscribe_msg)?);
    writer.write_all(msg_str.as_bytes()).await?;
    info!("📤 Sent mining.subscribe");

    // Read subscribe response
    let mut line = String::new();
    reader.read_line(&mut line).await?;
    let response: Value = serde_json::from_str(line.trim())?;

    let extranonce1 = response.get("result")
        .and_then(|r| r.get(1))
        .and_then(|e| e.as_str())
        .unwrap_or("")
        .to_string();
    let extranonce2_size = response.get("result")
        .and_then(|r| r.get(2))
        .and_then(|s| s.as_u64())
        .unwrap_or(4) as usize;

    info!("📥 Subscribed: extranonce1={}, extranonce2_size={}", extranonce1, extranonce2_size);
    line.clear();

    // Send mining.authorize
    let worker_full = format!("{}.{}", wallet, worker_name);
    let authorize_msg = json!({
        "id": 2,
        "method": "mining.authorize",
        "params": [worker_full, "x"]  // password is typically "x" or empty
    });
    let msg_str = format!("{}\n", serde_json::to_string(&authorize_msg)?);
    writer.write_all(msg_str.as_bytes()).await?;
    info!("📤 Sent mining.authorize for {}", worker_full);

    // Read authorize response
    reader.read_line(&mut line).await?;
    let auth_response: Value = serde_json::from_str(line.trim())?;
    if auth_response.get("result").and_then(|r| r.as_bool()).unwrap_or(false) {
        info!("✅ Authorized successfully!");
    } else {
        let error = auth_response.get("error").cloned().unwrap_or(json!("Unknown error"));
        error!("❌ Authorization failed: {}", error);
        return Err(anyhow::anyhow!("Authorization failed"));
    }
    line.clear();

    info!("🔥 Starting {} pool mining threads", threads);

    // Spawn mining threads
    let handles: Vec<_> = (0..threads)
        .map(|thread_id| {
            let hash_counter = hash_counter.clone();
            let is_running = is_running.clone();
            let current_job = current_job.clone();
            let job_signal = job_signal.clone();
            let current_difficulty = current_difficulty.clone();
            let share_tx = share_tx.clone();
            let extranonce1 = extranonce1.clone();

            tokio::spawn(async move {
                pool_mining_thread(
                    thread_id, hash_counter, is_running, intensity,
                    current_job, job_signal, current_difficulty,
                    share_tx, extranonce1, extranonce2_size
                ).await
            })
        })
        .collect();

    // Drop original share_tx so channel closes when all mining threads stop
    drop(share_tx);

    // Start hash rate monitor
    let monitor_counter = hash_counter.clone();
    let monitor_running = is_running.clone();
    let monitor_hashrate = Arc::new(tokio::sync::RwLock::new(0.0_f64));
    let monitor_hashrate_clone = monitor_hashrate.clone();
    let monitor_handle = tokio::spawn(async move {
        hash_rate_monitor(monitor_counter, monitor_running, monitor_hashrate_clone).await;
    });

    // Stratum connection handler task (reading notifications and submitting shares)
    let stratum_running = is_running.clone();
    let stratum_job = current_job.clone();
    let stratum_job_signal = job_signal.clone();
    let stratum_difficulty = current_difficulty.clone();

    let stratum_handle = tokio::spawn(async move {
        let mut next_submit_id = 10u64;

        loop {
            tokio::select! {
                // Read notifications from pool
                result = reader.read_line(&mut line) => {
                    match result {
                        Ok(0) => {
                            warn!("⚠️ Pool connection closed");
                            break;
                        }
                        Ok(_) => {
                            if let Ok(msg) = serde_json::from_str::<Value>(line.trim()) {
                                // Handle notifications
                                if let Some(method) = msg.get("method").and_then(|m| m.as_str()) {
                                    match method {
                                        "mining.notify" => {
                                            if let Some(params) = msg.get("params").and_then(|p| p.as_array()) {
                                                if let Some(job) = parse_mining_notify(params) {
                                                    info!("📋 New job: id={}, clean={}", job.job_id, job.clean_jobs);
                                                    *stratum_job.write().await = Some(job);
                                                    stratum_job_signal.fetch_add(1, Ordering::SeqCst);
                                                }
                                            }
                                        }
                                        "mining.set_difficulty" => {
                                            if let Some(diff) = msg.get("params")
                                                .and_then(|p| p.get(0))
                                                .and_then(|d| d.as_f64())
                                            {
                                                info!("🎯 Difficulty updated: {}", diff);
                                                *stratum_difficulty.write().await = diff;
                                            }
                                        }
                                        _ => {
                                            warn!("Unknown method: {}", method);
                                        }
                                    }
                                }

                                // Handle responses to share submissions
                                if let Some(id) = msg.get("id").and_then(|i| i.as_u64()) {
                                    if id >= 10 {
                                        // This is a response to a share submission
                                        if msg.get("result").and_then(|r| r.as_bool()).unwrap_or(false) {
                                            info!("✅ Share accepted!");
                                        } else if let Some(error) = msg.get("error") {
                                            warn!("❌ Share rejected: {}", error);
                                        }
                                    }
                                }
                            }
                            line.clear();
                        }
                        Err(e) => {
                            error!("Failed to read from pool: {}", e);
                            break;
                        }
                    }
                }

                // Submit shares to pool
                Some(share) = share_rx.recv() => {
                    let submit_msg = json!({
                        "id": next_submit_id,
                        "method": "mining.submit",
                        "params": [
                            share.worker_name,
                            share.job_id,
                            share.extranonce2,
                            share.ntime,
                            share.nonce
                        ]
                    });
                    next_submit_id += 1;

                    if let Ok(msg_str) = serde_json::to_string(&submit_msg) {
                        if let Err(e) = writer.write_all(format!("{}\n", msg_str).as_bytes()).await {
                            error!("Failed to submit share: {}", e);
                        }
                    }
                }
            }

            if !stratum_running.load(Ordering::SeqCst) {
                break;
            }
        }
    });

    info!("✅ Q-NarwhalKnight pool miner started successfully!");
    info!("Press Ctrl+C to stop mining...");

    // Wait for shutdown signal
    signal::ctrl_c().await?;

    info!("🛑 Shutdown signal received, stopping pool mining...");
    is_running.store(false, Ordering::SeqCst);

    // Wait for all threads to stop
    for handle in handles {
        let _ = handle.await;
    }
    monitor_handle.abort();
    stratum_handle.abort();

    let total_hashes = hash_counter.load(Ordering::Relaxed);
    info!("👋 Q-NarwhalKnight pool miner stopped. Total hashes: {}", total_hashes);

    Ok(())
}

/// Job received from pool via mining.notify
#[derive(Debug, Clone)]
struct PoolJob {
    job_id: String,
    prevhash: [u8; 32],
    coinbase1: Vec<u8>,
    coinbase2: Vec<u8>,
    merkle_branches: Vec<[u8; 32]>,
    version: u32,
    nbits: u32,
    ntime: u32,
    clean_jobs: bool,
}

/// Share to submit to pool
#[derive(Debug, Clone)]
struct ShareToSubmit {
    worker_name: String,
    job_id: String,
    extranonce2: String,
    ntime: String,
    nonce: String,
}

fn parse_mining_notify(params: &[Value]) -> Option<PoolJob> {
    let job_id = params.get(0)?.as_str()?.to_string();
    let prevhash_hex = params.get(1)?.as_str()?;
    let coinbase1_hex = params.get(2)?.as_str()?;
    let coinbase2_hex = params.get(3)?.as_str()?;
    let merkle_array = params.get(4)?.as_array()?;
    let version_hex = params.get(5)?.as_str()?;
    let nbits_hex = params.get(6)?.as_str()?;
    let ntime_hex = params.get(7)?.as_str()?;
    let clean_jobs = params.get(8)?.as_bool().unwrap_or(false);

    // Parse prevhash
    let prevhash_bytes = hex::decode(prevhash_hex).ok()?;
    let mut prevhash = [0u8; 32];
    if prevhash_bytes.len() >= 32 {
        prevhash.copy_from_slice(&prevhash_bytes[..32]);
    }

    // Parse coinbase
    let coinbase1 = hex::decode(coinbase1_hex).ok()?;
    let coinbase2 = hex::decode(coinbase2_hex).ok()?;

    // Parse merkle branches
    let merkle_branches: Vec<[u8; 32]> = merkle_array.iter()
        .filter_map(|v| {
            let hex_str = v.as_str()?;
            let bytes = hex::decode(hex_str).ok()?;
            if bytes.len() >= 32 {
                let mut arr = [0u8; 32];
                arr.copy_from_slice(&bytes[..32]);
                Some(arr)
            } else {
                None
            }
        })
        .collect();

    // Parse version, nbits, ntime
    let version = u32::from_str_radix(version_hex, 16).ok()?;
    let nbits = u32::from_str_radix(nbits_hex, 16).ok()?;
    let ntime = u32::from_str_radix(ntime_hex, 16).ok()?;

    Some(PoolJob {
        job_id,
        prevhash,
        coinbase1,
        coinbase2,
        merkle_branches,
        version,
        nbits,
        ntime,
        clean_jobs,
    })
}

async fn pool_mining_thread(
    thread_id: usize,
    hash_counter: Arc<AtomicU64>,
    is_running: Arc<AtomicBool>,
    intensity: u8,
    current_job: Arc<tokio::sync::RwLock<Option<PoolJob>>>,
    job_signal: Arc<AtomicU64>,
    current_difficulty: Arc<tokio::sync::RwLock<f64>>,
    share_tx: mpsc::Sender<ShareToSubmit>,
    extranonce1: String,
    extranonce2_size: usize,
) {
    // Pin thread to CPU core for cache locality
    let core_ids = core_affinity::get_core_ids().unwrap_or_default();
    if thread_id < core_ids.len() {
        if core_affinity::set_for_current(core_ids[thread_id]) {
            info!("🔥 Pool mining thread {} started (pinned to core {})", thread_id, thread_id);
        } else {
            info!("🔥 Pool mining thread {} started (affinity pinning failed)", thread_id);
        }
    } else {
        info!("🔥 Pool mining thread {} started", thread_id);
    }

    let batch_size = (intensity as u64) * 100_000;
    let mut last_job_signal = 0u64;

    // Use thread ID to vary extranonce2 space across threads
    let mut nonce_base = (thread_id as u64) << 48;
    let mut extranonce2_counter: u32 = thread_id as u32;

    while is_running.load(Ordering::SeqCst) {
        // Wait for a job
        let job = {
            let job_opt = current_job.read().await;
            job_opt.clone()
        };

        let job = match job {
            Some(j) => j,
            None => {
                // No job yet, wait a bit
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                continue;
            }
        };

        let current_signal = job_signal.load(Ordering::Relaxed);
        if current_signal != last_job_signal {
            last_job_signal = current_signal;
            nonce_base = (thread_id as u64) << 48;
            extranonce2_counter = thread_id as u32;
        }

        let difficulty = *current_difficulty.read().await;

        // Calculate difficulty target (simplified for Q-NarwhalKnight)
        // Target = max_target / difficulty
        let max_target = [0xff_u8; 32]; // Maximum possible target
        let target = calculate_pool_target(difficulty);

        // Generate extranonce2
        let extranonce2 = format!("{:0>width$x}", extranonce2_counter, width = extranonce2_size * 2);
        extranonce2_counter = extranonce2_counter.wrapping_add(threads_count() as u32);

        // Build coinbase: coinbase1 + extranonce1 + extranonce2 + coinbase2
        let mut coinbase = job.coinbase1.clone();
        if let Ok(ext1) = hex::decode(&extranonce1) {
            coinbase.extend_from_slice(&ext1);
        }
        if let Ok(ext2) = hex::decode(&extranonce2) {
            coinbase.extend_from_slice(&ext2);
        }
        coinbase.extend_from_slice(&job.coinbase2);

        // Hash coinbase to get coinbase hash
        let coinbase_hash = blake3::hash(&coinbase);

        // Build merkle root
        let mut merkle_root = *coinbase_hash.as_bytes();
        for branch in &job.merkle_branches {
            let mut combined = [0u8; 64];
            combined[..32].copy_from_slice(&merkle_root);
            combined[32..].copy_from_slice(branch);
            merkle_root = *blake3::hash(&combined).as_bytes();
        }

        // Build block header (simplified for Q-NarwhalKnight)
        // header = version + prevhash + merkle_root + ntime + nbits + nonce
        let mut header_base = Vec::with_capacity(80);
        header_base.extend_from_slice(&job.version.to_le_bytes());
        header_base.extend_from_slice(&job.prevhash);
        header_base.extend_from_slice(&merkle_root);
        header_base.extend_from_slice(&job.ntime.to_le_bytes());
        header_base.extend_from_slice(&job.nbits.to_le_bytes());
        // Nonce will be appended during mining

        // Mine batch of nonces
        for _ in 0..batch_size {
            if !is_running.load(Ordering::SeqCst) {
                break;
            }

            // Check for new job
            if job_signal.load(Ordering::Relaxed) != last_job_signal {
                break;
            }

            // Build full header with nonce
            let mut header = header_base.clone();
            header.extend_from_slice(&nonce_base.to_le_bytes());

            // Compute hash using DAG-Knight VDF
            let hash = compute_dag_knight_hash_for_pool(&header);
            hash_counter.fetch_add(1, Ordering::Relaxed);

            // Check if meets target
            if hash_meets_target(&hash, &target) {
                info!("💎 Share found! Thread {}, nonce {}", thread_id, nonce_base);

                // Submit share
                let share = ShareToSubmit {
                    worker_name: format!("worker_{}", thread_id),
                    job_id: job.job_id.clone(),
                    extranonce2: extranonce2.clone(),
                    ntime: format!("{:08x}", job.ntime),
                    nonce: format!("{:016x}", nonce_base),
                };

                let _ = share_tx.send(share).await;
            }

            nonce_base = nonce_base.wrapping_add(1);
        }
    }

    info!("🛑 Pool mining thread {} stopped", thread_id);
}

fn threads_count() -> usize {
    num_cpus::get()
}

fn calculate_pool_target(difficulty: f64) -> [u8; 32] {
    // Pool difficulty 1 target (simplified)
    let diff1_target: u128 = 0x00000000ffff_0000_0000_0000_0000_0000_u128;
    let target_value = (diff1_target as f64 / difficulty) as u128;

    let mut target = [0u8; 32];
    // Set the target in the first 16 bytes (big-endian for comparison)
    for i in 0..16 {
        target[15 - i] = ((target_value >> (i * 8)) & 0xff) as u8;
    }
    target
}

fn hash_meets_target(hash: &[u8; 32], target: &[u8; 32]) -> bool {
    // Compare as big-endian (first bytes are most significant)
    for i in 0..32 {
        if hash[i] < target[i] {
            return true;
        } else if hash[i] > target[i] {
            return false;
        }
    }
    true // Equal counts as meeting target
}

fn compute_dag_knight_hash_for_pool(header: &[u8]) -> [u8; 32] {
    // Initial hash
    let initial_hash = blake3::hash(header);

    // VDF computation (100 iterations for consistency with solo mining)
    let mut current = *initial_hash.as_bytes();
    for _ in 0..100 {
        current = *blake3::hash(&current).as_bytes();
    }

    current
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
    miner_id: String,
    miner_name: Option<String>,
) {
    // OPTIMIZATION: Pin thread to specific CPU core for cache locality on multi-socket systems
    // This dramatically improves performance on AMD EPYC / Intel Xeon servers with NUMA
    let core_ids = core_affinity::get_core_ids().unwrap_or_default();
    if thread_id < core_ids.len() {
        if core_affinity::set_for_current(core_ids[thread_id]) {
            info!("🔥 CPU mining thread {} started (pinned to core {})", thread_id, thread_id);
        } else {
            info!("🔥 CPU mining thread {} started (affinity pinning failed, running unpinned)", thread_id);
        }
    } else {
        info!("🔥 CPU mining thread {} started (no core pinning - more threads than cores)", thread_id);
    }

    let mut nonce = thread_id as u64 * 1_000_000;
    // ARCHITECTURE-SPECIFIC OPTIMIZATION: Tune batch size for AMD EPYC vs Intel Xeon
    // AMD EPYC benefits from larger batches due to higher core count and larger L3 cache
    // Intel Xeon with AVX-512 benefits from slightly smaller batches due to higher single-thread performance
    //
    // NOTE: This is a simplified heuristic. For production, detect actual CPU model and tune accordingly.
    // AMD EPYC 7xx3/9xx4: 256MB L3 cache → larger batch_size
    // Intel Xeon Platinum 8xxx: 60MB L3 cache → medium batch_size
    let batch_size = (intensity as u64) * 100_000; // Base batch size
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
                // v3.3.3-beta: Include miner_id and miner_name for identification
                let solution = serde_json::json!({
                    "miner_address": wallet,
                    "nonce": nonce,
                    "hash": hex::encode(hash),
                    "difficulty_target": hex::encode(target),
                    "challenge_hash": hex::encode(challenge_hash),
                    "hash_rate": hashrate_khs,  // Send hashrate in KH/s for network statistics
                    "miner_id": miner_id,
                    "worker_name": miner_name
                });

                // CRITICAL: Submit solution in background to avoid blocking mining thread
                // The mining thread must continue immediately to maintain hash rate
                // v4.5.0: Falls back to bootstrap1.quillon.xyz if primary server fails
                let normalized_url = normalize_server_url(api_url);
                let submit_url = format!("{}/api/v1/mining/submit", normalized_url);
                let fallback_submit_url = format!("{}/api/v1/mining/submit", FALLBACK_BOOTSTRAP_URL);
                let client_clone = client.clone();
                tokio::spawn(async move {
                    let try_submit = |url: String, sol: serde_json::Value, cl: reqwest::Client| async move {
                        cl.post(&url)
                            .json(&sol)
                            .timeout(std::time::Duration::from_secs(10))
                            .send()
                            .await
                    };

                    match try_submit(submit_url, solution.clone(), client_clone.clone()).await {
                        Ok(resp) if resp.status().is_success() => {
                            if let Ok(result) = resp.json::<serde_json::Value>().await {
                                if let Some(data) = result.get("data") {
                                    if let Some(reward) = data.get("reward_qnk") {
                                        info!("✅ Solution accepted! Earned {} QNK", reward);
                                    }
                                }
                            }
                        }
                        Ok(resp) => {
                            warn!("❌ Solution rejected by primary (HTTP {}) - trying fallback...", resp.status());
                            match try_submit(fallback_submit_url, solution, client_clone).await {
                                Ok(resp2) if resp2.status().is_success() => {
                                    if let Ok(result) = resp2.json::<serde_json::Value>().await {
                                        if let Some(data) = result.get("data") {
                                            if let Some(reward) = data.get("reward_qnk") {
                                                info!("✅ Solution accepted via fallback! Earned {} QNK", reward);
                                            }
                                        }
                                    }
                                }
                                _ => { warn!("❌ Solution rejected by fallback too"); }
                            }
                        }
                        Err(e) => {
                            warn!("⚠️  Primary submit failed: {} - trying fallback...", e);
                            match try_submit(fallback_submit_url, solution, client_clone).await {
                                Ok(resp2) if resp2.status().is_success() => {
                                    if let Ok(result) = resp2.json::<serde_json::Value>().await {
                                        if let Some(data) = result.get("data") {
                                            if let Some(reward) = data.get("reward_qnk") {
                                                info!("✅ Solution accepted via fallback! Earned {} QNK", reward);
                                            }
                                        }
                                    }
                                }
                                _ => { warn!("❌ Failed to submit solution to both servers"); }
                            }
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
    let primary_url = format!("{}/api/v1/events?wallet_address={}", normalized_url, wallet);
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
                warn!("Failed to create SSE client for {}: {}", url, e);
                if !use_fallback {
                    primary_fail_count += 1;
                    if primary_fail_count >= 3 {
                        info!("🔄 Switching SSE to fallback server {}", FALLBACK_BOOTSTRAP_URL);
                        use_fallback = true;
                        primary_fail_count = 0;
                    }
                }
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
            if !use_fallback {
                primary_fail_count += 1;
                if primary_fail_count >= 3 {
                    info!("🔄 Primary SSE failed {} times, switching to fallback {}", primary_fail_count, FALLBACK_BOOTSTRAP_URL);
                    use_fallback = true;
                    primary_fail_count = 0;
                }
            } else {
                // If fallback also fails, try primary again
                primary_fail_count += 1;
                if primary_fail_count >= 3 {
                    info!("🔄 Fallback SSE failed, retrying primary server...");
                    use_fallback = false;
                    primary_fail_count = 0;
                }
            }
            warn!("Reconnecting to SSE stream in 5 seconds...");
            tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
        }
    }

    info!("🛑 SSE listener stopped");
}

/// Check if server is currently syncing (returns is_syncing, blocks_behind)
/// Falls back to bootstrap1.quillon.xyz if primary server is unreachable.
async fn check_server_sync_status(api_url: &str) -> Result<(bool, u64)> {
    let client = reqwest::Client::new();
    let path = "/api/v1/status";

    let (body, _used_url) = fetch_with_fallback(&client, api_url, path).await?;

    let api_response: ApiResponse<serde_json::Value> = serde_json::from_str(&body)
        .map_err(|e| anyhow::anyhow!("Failed to parse status response: {}", e))?;

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

/// Fetch current mining challenge from API server (with fallback to bootstrap1.quillon.xyz)
async fn fetch_mining_challenge(api_url: &str) -> Result<MiningChallenge> {
    let client = reqwest::Client::new();
    let path = "/api/v1/mining/challenge";

    let (body, _used_url) = fetch_with_fallback(&client, api_url, path).await?;

    let api_response: ApiResponse<MiningChallenge> = serde_json::from_str(&body)
        .map_err(|e| anyhow::anyhow!("Failed to parse mining challenge response: {}", e))?;

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

/// Decentralized P2P Pool Mining Mode (v2.3.0+)
///
/// Instead of connecting to a centralized pool server via Stratum,
/// this mode connects to any P2P node running the distributed pool coordinator.
/// Shares are submitted via HTTP API and propagated via gossipsub.
///
/// Features:
/// - CRDT-based PPLNS: No central pool state needed
/// - VDF anti-grinding proofs: Prevents share manipulation
/// - Threshold signature payouts: Consensus-based reward distribution
/// - P2P share propagation: Fully decentralized operation
async fn run_decentralized_pool_mining(
    threads: usize,
    intensity: u8,
    wallet: &str,
    worker_name: &str,
    bootstrap_nodes: &str,
    region: &str,
) -> Result<()> {
    use tokio::sync::mpsc;
    use serde_json::json;

    let hash_counter = Arc::new(AtomicU64::new(0));
    let is_running = Arc::new(AtomicBool::new(true));
    let shares_submitted = Arc::new(AtomicU64::new(0));
    let blocks_found = Arc::new(AtomicU64::new(0));

    // Parse bootstrap nodes
    let nodes: Vec<&str> = bootstrap_nodes.split(',').map(|s| s.trim()).collect();
    let primary_node = nodes.first().copied().unwrap_or("http://localhost:8080");

    info!("🔌 Connecting to P2P network via {}", primary_node);
    info!("🌍 Pool region: {}", region);

    // Check pool node status
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()?;

    let pool_status = client
        .get(&format!("{}/api/v1/pool/status", primary_node))
        .send()
        .await;

    match pool_status {
        Ok(resp) if resp.status().is_success() => {
            info!("✅ Connected to P2P pool node");
            if let Ok(status) = resp.json::<serde_json::Value>().await {
                if let Some(workers) = status.get("worker_count") {
                    info!("   Active workers: {}", workers);
                }
                if let Some(hashrate) = status.get("pool_hashrate") {
                    info!("   Pool hashrate: {} H/s", hashrate);
                }
            }
        }
        Ok(resp) => {
            warn!("⚠️  Pool API returned status: {} - continuing anyway", resp.status());
        }
        Err(e) => {
            warn!("⚠️  Could not connect to pool status API: {} - continuing anyway", e);
            info!("   Mining will work via standard mining challenge API");
        }
    }

    // Channel for submitting shares
    let (share_tx, mut share_rx) = mpsc::channel::<DecentralizedShare>(1000);

    // Spawn share submitter task
    let client_for_shares = client.clone();
    let node_for_shares = primary_node.to_string();
    let wallet_for_shares = wallet.to_string();
    let worker_for_shares = worker_name.to_string();
    let shares_counter = shares_submitted.clone();
    let blocks_counter = blocks_found.clone();

    tokio::spawn(async move {
        info!("📤 Share submitter started");
        while let Some(share) = share_rx.recv().await {
            // Submit share via API
            let submit_url = format!("{}/api/v1/pool/submit-share", node_for_shares);
            let share_data = json!({
                "wallet": wallet_for_shares,
                "worker": worker_for_shares,
                "share_id": hex::encode(&share.share_id),
                "difficulty": share.difficulty,
                "block_height": share.block_height,
                "nonce": share.nonce,
                "timestamp": share.timestamp,
            });

            match client_for_shares.post(&submit_url).json(&share_data).send().await {
                Ok(resp) if resp.status().is_success() => {
                    shares_counter.fetch_add(1, Ordering::Relaxed);

                    // Check if we found a block
                    if let Ok(result) = resp.json::<serde_json::Value>().await {
                        if result.get("block_found").and_then(|b| b.as_bool()).unwrap_or(false) {
                            blocks_counter.fetch_add(1, Ordering::Relaxed);
                            info!("🎉 BLOCK FOUND! Share accepted as block solution!");
                        }
                    }
                }
                Ok(resp) => {
                    warn!("⚠️  Share rejected: {}", resp.status());
                }
                Err(e) => {
                    warn!("❌ Share submission failed: {}", e);
                }
            }
        }
    });

    // Spawn hashrate display task
    let hash_display = hash_counter.clone();
    let shares_display = shares_submitted.clone();
    let blocks_display = blocks_found.clone();
    let is_running_display = is_running.clone();

    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(10));
        let mut last_hash_count = 0u64;

        loop {
            interval.tick().await;
            if !is_running_display.load(Ordering::Relaxed) {
                break;
            }

            let current = hash_display.load(Ordering::Relaxed);
            let delta = current.saturating_sub(last_hash_count);
            let hashrate = delta as f64 / 10.0;
            last_hash_count = current;

            let shares = shares_display.load(Ordering::Relaxed);
            let blocks = blocks_display.load(Ordering::Relaxed);

            info!(
                "⛏️  Hashrate: {:.2} KH/s | Total hashes: {} | Shares: {} | Blocks: {}",
                hashrate / 1000.0,
                current,
                shares,
                blocks
            );
        }
    });

    // Main mining loop - uses standard mining challenge API
    let server_url = normalize_server_url(primary_node);
    let wallet_clone = wallet.to_string();
    let worker_clone = worker_name.to_string();

    info!("⚡ Starting {} mining threads...", threads);

    // Spawn mining threads
    for thread_id in 0..threads {
        let hash_counter = hash_counter.clone();
        let is_running = is_running.clone();
        let share_tx = share_tx.clone();
        let server = server_url.clone();
        let wallet = wallet_clone.clone();
        let worker = worker_clone.clone();
        let client = client.clone();

        tokio::spawn(async move {
            let mut local_nonce = (thread_id as u64) << 56; // Thread-unique nonce range

            loop {
                if !is_running.load(Ordering::Relaxed) {
                    break;
                }

                // Get mining challenge from API
                let challenge_url = format!("{}/api/v1/mining/challenge?wallet={}", server, wallet);
                let challenge: MiningChallenge = match client.get(&challenge_url).send().await {
                    Ok(resp) => {
                        match resp.json::<ApiResponse<MiningChallenge>>().await {
                            Ok(api_resp) if api_resp.success => {
                                match api_resp.data {
                                    Some(c) => c,
                                    None => {
                                        tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                                        continue;
                                    }
                                }
                            }
                            _ => {
                                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                                continue;
                            }
                        }
                    }
                    Err(_) => {
                        tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                        continue;
                    }
                };

                // Parse challenge
                let challenge_bytes = match hex::decode(&challenge.challenge_hash) {
                    Ok(b) if b.len() >= 32 => {
                        let mut arr = [0u8; 32];
                        arr.copy_from_slice(&b[..32]);
                        arr
                    }
                    _ => continue,
                };

                let target_bytes = match hex::decode(&challenge.difficulty_target) {
                    Ok(b) if b.len() >= 32 => {
                        let mut arr = [0u8; 32];
                        arr.copy_from_slice(&b[..32]);
                        arr
                    }
                    _ => continue,
                };

                // Mining loop for this challenge
                let mining_start = std::time::Instant::now();
                let batch_size = 10000u64;

                while mining_start.elapsed() < std::time::Duration::from_secs(30) {
                    if !is_running.load(Ordering::Relaxed) {
                        break;
                    }

                    for _ in 0..batch_size {
                        local_nonce = local_nonce.wrapping_add(1);

                        // Compute hash
                        let mut hasher = blake3::Hasher::new();
                        hasher.update(&challenge_bytes);
                        hasher.update(&local_nonce.to_le_bytes());
                        hasher.update(wallet.as_bytes());
                        let hash = hasher.finalize();
                        let hash_bytes = hash.as_bytes();

                        hash_counter.fetch_add(1, Ordering::Relaxed);

                        // Check if solution meets target
                        if hash_bytes[..] < target_bytes[..] {
                            // Found a share!
                            let share = DecentralizedShare {
                                share_id: *hash_bytes,
                                difficulty: challenge.block_reward,
                                block_height: challenge.block_height,
                                nonce: local_nonce,
                                timestamp: chrono::Utc::now().timestamp_millis() as u64,
                            };

                            let _ = share_tx.send(share).await;
                            info!("💎 Share found! nonce={}, height={}", local_nonce, challenge.block_height);
                        }
                    }
                }
            }
        });
    }

    // Wait for shutdown signal
    signal::ctrl_c().await?;
    info!("🛑 Shutting down...");
    is_running.store(false, Ordering::Relaxed);

    let total_shares = shares_submitted.load(Ordering::Relaxed);
    let total_blocks = blocks_found.load(Ordering::Relaxed);
    let total_hashes = hash_counter.load(Ordering::Relaxed);

    info!("📊 Final statistics:");
    info!("   Total hashes: {}", total_hashes);
    info!("   Shares submitted: {}", total_shares);
    info!("   Blocks found: {}", total_blocks);

    Ok(())
}

/// Decentralized share data structure
struct DecentralizedShare {
    share_id: [u8; 32],
    difficulty: f64,
    block_height: u64,
    nonce: u64,
    timestamp: u64,
}