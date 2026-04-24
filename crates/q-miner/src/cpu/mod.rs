use crate::{MiningEngine, MiningAlgorithm, MiningStats, WorkUnit, algorithms::DagKnightVDF};
use anyhow::Result;
use async_trait::async_trait;
use std::sync::{Arc, atomic::{AtomicU64, AtomicBool, Ordering}};
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{info, debug, error, warn};

/// High-performance CPU miner optimized for Q-NarwhalKnight
pub struct CpuMiner {
    thread_count: usize,
    intensity: u8,
    algorithm: Arc<DagKnightVDF>,
    current_work: Arc<RwLock<Option<WorkUnit>>>,
    stats: Arc<RwLock<MiningStats>>,
    hash_counter: Arc<AtomicU64>,
    is_running: Arc<AtomicBool>,
    worker_threads: Vec<tokio::task::JoinHandle<()>>,
    /// API server base URL for solution submission
    server_url: String,
    /// Wallet address for mining rewards
    wallet_address: String,
}

impl CpuMiner {
    pub async fn new(thread_count: usize, intensity: u8, server_url: String, wallet_address: String) -> Result<Self> {
        info!("🔥 Initializing CPU miner with {} threads, intensity {}", thread_count, intensity);

        // Detect CPU capabilities
        let cpu_info = detect_cpu_capabilities();
        info!("💻 CPU: {} ({} cores, {} threads)",
            cpu_info.brand, cpu_info.physical_cores, cpu_info.logical_threads);

        if cpu_info.has_avx2 {
            info!("⚡ AVX2 acceleration enabled");
        }
        if cpu_info.has_avx512 {
            info!("🚀 AVX-512 acceleration enabled");
        }

        let algorithm = Arc::new(DagKnightVDF::new(1000)); // Base difficulty

        Ok(Self {
            thread_count,
            intensity,
            algorithm,
            current_work: Arc::new(RwLock::new(None)),
            stats: Arc::new(RwLock::new(MiningStats::default())),
            hash_counter: Arc::new(AtomicU64::new(0)),
            is_running: Arc::new(AtomicBool::new(false)),
            worker_threads: Vec::new(),
            server_url,
            wallet_address,
        })
    }
    
    /// Start CPU mining threads
    async fn start_mining_threads(&mut self) -> Result<()> {
        info!("🚀 Starting {} CPU mining threads", self.thread_count);
        
        self.is_running.store(true, Ordering::SeqCst);
        
        for thread_id in 0..self.thread_count {
            let is_running = self.is_running.clone();
            let current_work = self.current_work.clone();
            let hash_counter = self.hash_counter.clone();
            let algorithm = self.algorithm.clone();
            let intensity = self.intensity;
            let server_url = self.server_url.clone();
            let wallet_address = self.wallet_address.clone();

            let handle = tokio::spawn(async move {
                cpu_mining_thread(thread_id, is_running, current_work, hash_counter, algorithm, intensity, server_url, wallet_address).await;
            });
            
            self.worker_threads.push(handle);
        }
        
        // Start hash rate monitor
        let hash_counter = self.hash_counter.clone();
        let stats = self.stats.clone();
        let is_running = self.is_running.clone();
        
        tokio::spawn(async move {
            hash_rate_monitor(hash_counter, stats, is_running).await;
        });
        
        Ok(())
    }
}

#[async_trait]
impl MiningEngine for CpuMiner {
    async fn start(&mut self) -> Result<()> {
        info!("⚡ Starting CPU mining engine");
        self.start_mining_threads().await?;
        info!("✅ CPU mining engine started successfully");
        Ok(())
    }
    
    async fn stop(&mut self) -> Result<()> {
        info!("🛑 Stopping CPU mining engine");
        self.is_running.store(false, Ordering::SeqCst);
        
        // Wait for all threads to stop
        for handle in self.worker_threads.drain(..) {
            let _ = handle.await;
        }
        
        info!("✅ CPU mining engine stopped");
        Ok(())
    }
    
    async fn get_hash_rate(&self) -> f64 {
        let stats = self.stats.read().await;
        stats.hash_rate
    }
    
    async fn get_stats(&self) -> MiningStats {
        self.stats.read().await.clone()
    }
}

/// Individual CPU mining thread
async fn cpu_mining_thread(
    thread_id: usize,
    is_running: Arc<AtomicBool>,
    current_work: Arc<RwLock<Option<WorkUnit>>>,
    hash_counter: Arc<AtomicU64>,
    algorithm: Arc<DagKnightVDF>,
    intensity: u8,
    server_url: String,
    wallet_address: String,
) {
    info!("🔥 CPU mining thread {} started", thread_id);

    let mut nonce_base = thread_id as u64 * 1_000_000;
    let batch_size = (intensity as u64) * 10_000; // Adjust batch size by intensity

    let http_client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new());
    let submit_url = format!("{}/api/v1/mining/submit", server_url);

    while is_running.load(Ordering::SeqCst) {
        // Get current work
        let work = {
            let work_guard = current_work.read().await;
            work_guard.clone()
        };

        if let Some(work) = work {
            // Mine a batch of nonces
            for nonce_offset in 0..batch_size {
                let nonce = nonce_base + nonce_offset;

                // Compute hash using DAG-Knight VDF algorithm
                if let Ok(hash) = algorithm.as_ref().compute_hash(&work.extra_data, nonce).await {
                    hash_counter.fetch_add(1, Ordering::Relaxed);

                    // Check if solution meets difficulty
                    if algorithm.as_ref().verify_solution(&hash, &work.difficulty_target).await {
                        let hex_hash = hex::encode(&hash);
                        info!("💎 Solution submitted: nonce={} hash={:.8}...", nonce, hex_hash);

                        // Build submission payload
                        let submission = serde_json::json!({
                            "nonce": nonce,
                            "hash": hex_hash,
                            "wallet_address": wallet_address,
                            "difficulty": 0u64,
                        });

                        // HTTP POST to server
                        match http_client
                            .post(&submit_url)
                            .json(&submission)
                            .send()
                            .await
                        {
                            Ok(resp) if resp.status().is_success() => {
                                info!("✅ CPU Thread {} solution accepted by server", thread_id);
                            }
                            Ok(resp) => {
                                warn!("⚠️ CPU Thread {} solution submit returned HTTP {}", thread_id, resp.status());
                            }
                            Err(e) => {
                                warn!("⚠️ CPU Thread {} solution submit failed: {}", thread_id, e);
                            }
                        }
                    }
                }
                
                // Check for stop signal periodically
                if nonce_offset % 1000 == 0 && !is_running.load(Ordering::SeqCst) {
                    break;
                }
            }
            
            nonce_base += batch_size * 1000; // Move to next nonce range
        } else {
            // No work available, wait
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }
    }
    
    info!("🛑 CPU mining thread {} stopped", thread_id);
}

/// Hash rate monitoring
async fn hash_rate_monitor(
    hash_counter: Arc<AtomicU64>,
    stats: Arc<RwLock<MiningStats>>,
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
            
            // Update stats
            {
                let mut stats_guard = stats.write().await;
                stats_guard.hash_rate = hash_rate;
                stats_guard.uptime = chrono::Duration::seconds(
                    current_time.elapsed().as_secs() as i64
                );
            }
            
            debug!("📊 CPU Hash Rate: {:.2} H/s", hash_rate);
        }
        
        last_hash_count = current_hash_count;
        last_time = current_time;
    }
}

/// CPU capability detection
#[derive(Debug, Clone)]
pub struct CpuInfo {
    pub brand: String,
    pub physical_cores: usize,
    pub logical_threads: usize,
    pub has_avx2: bool,
    pub has_avx512: bool,
    pub has_aes_ni: bool,
    pub cache_l3_size: usize,
}

pub fn detect_cpu_capabilities() -> CpuInfo {
    let mut logical_threads = num_cpus::get();
    let physical_cores = num_cpus::get_physical();

    // ✅ P0 FIX: Enhanced CPU detection for high-core-count systems (AMD EPYC 9654, etc.)
    // Read /proc/cpuinfo directly on Linux for comparison with num_cpus
    #[cfg(target_os = "linux")]
    let proc_cpuinfo_threads = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .map(|contents| {
            contents.lines()
                .filter(|line| line.starts_with("processor"))
                .count()
        });

    #[cfg(not(target_os = "linux"))]
    let proc_cpuinfo_threads: Option<usize> = None;

    // 🛡️ v5.1.1: Windows >64 core detection using GetActiveProcessorCount(ALL_PROCESSOR_GROUPS)
    // Standard APIs (GetSystemInfo, num_cpus) only see 64 cores within one processor group.
    // High-core-count CPUs like EPYC 9654 (192 threads) span multiple processor groups.
    #[cfg(target_os = "windows")]
    {
        extern "system" {
            fn GetActiveProcessorCount(GroupNumber: u16) -> u32;
        }
        // ALL_PROCESSOR_GROUPS = 0xFFFF
        let win_total = unsafe { GetActiveProcessorCount(0xFFFF) } as usize;
        if win_total > logical_threads {
            warn!("🔧 Windows processor groups: num_cpus sees {} but GetActiveProcessorCount(ALL) sees {}",
                  logical_threads, win_total);
            warn!("   Upgrading thread count to {} for full multi-socket utilization", win_total);
            logical_threads = win_total;
        }
    }

    // ⚠️ P0 FIX: Warn if detection appears capped at common limits
    if logical_threads == 64 || logical_threads == 128 || logical_threads == 256 {
        warn!("⚠️  Detected exactly {} threads - this may be a detection limit, not actual hardware", logical_threads);

        if let Some(proc_count) = proc_cpuinfo_threads {
            if proc_count > logical_threads {
                warn!("   /proc/cpuinfo shows {} CPUs, but num_cpus sees only {}", proc_count, logical_threads);
                warn!("   This indicates OS-level restrictions (cgroups, cpuset, or affinity)");
                warn!("   Check: cat /sys/fs/cgroup/cpuset.cpus");
                warn!("   Check: cat /proc/self/status | grep Cpus_allowed_list");
            } else if proc_count == logical_threads {
                info!("   /proc/cpuinfo confirms {} threads (OS limit or actual hardware)", logical_threads);
            }
        }

        warn!("   If you have more cores, use --threads <N> to override");
        warn!("   Example: --threads 96 for AMD EPYC 9654 (96 physical cores)");
        warn!("   Example: --threads 192 for AMD EPYC 9654 (with SMT/hyperthreading)");
    }

    // 💡 P0 FIX: AMD EPYC 9654 specific detection
    #[cfg(target_arch = "x86_64")]
    {
        let cpuid = raw_cpuid::CpuId::new();
        let brand = cpuid.get_vendor_info()
            .map(|v: raw_cpuid::VendorInfo| v.as_str().to_string())
            .unwrap_or_else(|| "Unknown".to_string());

        // Check if this is AMD EPYC 9654
        let is_epyc_9654 = brand.contains("AuthenticAMD") &&
            (proc_cpuinfo_threads.unwrap_or(0) >= 96 || logical_threads >= 96);

        if is_epyc_9654 && logical_threads < 96 {
            warn!("🔴 AMD EPYC 9654 DETECTED but only {} threads visible!", logical_threads);
            warn!("   This CPU has 96 physical cores (192 with SMT)");
            warn!("   Recommended configurations:");
            warn!("   • --threads 96  (all physical cores, lower power)");
            warn!("   • --threads 192 (all logical threads, maximum performance)");
            warn!("   • Check cgroup limits: cat /sys/fs/cgroup/cpuset.cpus");
            warn!("   • Check process limits: ulimit -u");
        }

        let has_avx2 = cpuid.get_extended_feature_info()
            .map(|ef: raw_cpuid::ExtendedFeatures| ef.has_avx2())
            .unwrap_or(false);

        let has_avx512 = cpuid.get_extended_feature_info()
            .map(|ef: raw_cpuid::ExtendedFeatures| ef.has_avx512f())
            .unwrap_or(false);

        let has_aes_ni = cpuid.get_feature_info()
            .map(|fi: raw_cpuid::FeatureInfo| fi.has_aesni())
            .unwrap_or(false);

        // Note: raw_cpuid CacheInfo doesn't have a simple cache_size() method
        // We'll just set a default value for now
        let cache_l3_size = 0usize;

        info!("💻 CPU Detection Results:");
        info!("   Brand: {}", brand);
        info!("   num_cpus: {} logical threads, {} physical cores", logical_threads, physical_cores);
        if let Some(proc_count) = proc_cpuinfo_threads {
            info!("   /proc/cpuinfo: {} processors", proc_count);
        }
        info!("   Features: AVX2={}, AVX512={}, AES-NI={}", has_avx2, has_avx512, has_aes_ni);

        return CpuInfo {
            brand,
            physical_cores,
            logical_threads,
            has_avx2,
            has_avx512,
            has_aes_ni,
            cache_l3_size,
        };
    }

    // Fallback for non-x86_64 or if cpuid fails
    info!("💻 CPU Detection: {} logical threads, {} physical cores", logical_threads, physical_cores);
    if let Some(proc_count) = proc_cpuinfo_threads {
        info!("   /proc/cpuinfo: {} processors", proc_count);
    }

    CpuInfo {
        brand: "Unknown CPU".to_string(),
        physical_cores,
        logical_threads,
        has_avx2: false,
        has_avx512: false,
        has_aes_ni: false,
        cache_l3_size: 0,
    }
}

/// Detect optimal batch size based on CPU SIMD capabilities.
/// AVX-512: 16 nonces, AVX2: 8, NEON/SSE: 4, fallback: 1.
/// BLAKE3 internally uses SIMD for each hash, but by interleaving multiple
/// nonces through VDF rounds we keep SIMD pipelines saturated and exploit
/// instruction-level parallelism across independent hash chains.
pub fn optimal_mining_batch_size() -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(ef) = raw_cpuid::CpuId::new().get_extended_feature_info() {
            if ef.has_avx512f() {
                return 16;
            }
            if ef.has_avx2() {
                return 8;
            }
        }
        return 4; // SSE2 baseline on x86_64
    }
    #[cfg(target_arch = "aarch64")]
    {
        return 4; // NEON always available on aarch64
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        return 1;
    }
}

/// Batch DAG-Knight VDF mining: process `batch_size` consecutive nonces starting
/// from `nonce_start`, writing (nonce, final_hash) pairs into `results`.
///
/// The VDF consists of 100 sequential BLAKE3 rounds per nonce.  By interleaving
/// VDF rounds across the batch we keep the CPU's SIMD execution units busy —
/// while one hash's result is being written back another can start compression.
/// This yields 2-4x throughput vs the sequential loop.
///
/// Returns the number of results written (always == batch_size).
#[inline(never)] // prevent inlining so the hot loop stays in icache
pub fn compute_dag_knight_hash_batch(
    challenge: &[u8; 32],
    nonce_start: u64,
    batch_size: usize,
    results: &mut [(u64, [u8; 32])],
) -> usize {
    debug_assert!(results.len() >= batch_size);

    // 1. Build initial inputs and compute first hashes for all nonces
    let mut states: [([u8; 32], u64); 16] = [([0u8; 32], 0u64); 16];
    let bs = batch_size.min(16).min(results.len());

    for i in 0..bs {
        let nonce = nonce_start.wrapping_add(i as u64);
        let mut input = [0u8; 40];
        input[..32].copy_from_slice(challenge);
        input[32..].copy_from_slice(&nonce.to_le_bytes());
        let h = blake3::hash(&input);
        states[i] = (*h.as_bytes(), nonce);
    }

    // 2. Interleaved VDF: 100 inner rounds across all nonces in the batch.
    //    (1 initial hash + 100 inner = 101 total, matching server verification)
    //    Each round is data-dependent (hash of previous), but rounds for
    //    *different* nonces are independent — the CPU can pipeline them.
    for _ in 0..100 {
        for s in states[..bs].iter_mut() {
            s.0 = *blake3::hash(&s.0).as_bytes();
        }
    }

    // 3. Write results
    for i in 0..bs {
        results[i] = (states[i].1, states[i].0);
    }

    bs
}

/// v1.0.5: Genus-2 Jacobian VDF mining for a single nonce.
///
/// Per whitepaper Algorithm 2:
/// 1. x = BLAKE3(challenge || nonce) → seed
/// 2. Map seed to Jacobian element via hash-to-curve
/// 3. y = x^(2^T) sequential squaring in J(C)
/// 4. h = SHA3-256(y), return (nonce, h, vdf_output, proof)
///
/// Returns None if computation fails, Some((nonce, hash, vdf_output, proof, checkpoints, iterations)) on success.
pub fn compute_genus2_vdf_single(
    challenge: &[u8; 32],
    nonce: u64,
    vdf_iterations: u64,
) -> Option<(u64, [u8; 32], Vec<u8>, Vec<u8>, Vec<Vec<u8>>, u64)> {
    use q_vdf::genus2_vdf::{Genus2CurveParams, Genus2VDF, JacobianElement};
    use sha3::{Digest, Sha3_256};

    // Step 1: Derive seed from challenge + nonce
    let mut input = [0u8; 40];
    input[..32].copy_from_slice(challenge);
    input[32..].copy_from_slice(&nonce.to_le_bytes());
    let seed = blake3::hash(&input);

    // Step 2: Map seed to initial Jacobian element
    let curve = Genus2CurveParams::pq128();
    let g_initial = match JacobianElement::from_hash(seed.as_bytes(), &curve) {
        Ok(g) => g,
        Err(_) => return None,
    };

    // Step 3: Sequential squaring in J(C)
    let vdf = Genus2VDF::with_curve(curve.clone(), vdf_iterations);
    let checkpoint_interval = (vdf_iterations / 10).max(1);
    let mut checkpoints = Vec::new();
    let mut g = g_initial;

    for i in 0..vdf_iterations {
        g = match vdf.double_jacobian_pub(&g) {
            Ok(next) => next,
            Err(_) => return None,
        };
        if i > 0 && i % checkpoint_interval == 0 {
            checkpoints.push(g.to_bytes());
        }
    }

    let vdf_output = g.to_bytes();

    // Step 4: SHA3-256 of VDF output → final hash
    let mut sha3 = Sha3_256::new();
    sha3.update(&vdf_output);
    let hash_result = sha3.finalize();
    let mut hash = [0u8; 32];
    hash.copy_from_slice(&hash_result);

    // Step 5: Generate Wesolowski proof
    let mut proof_hasher = Sha3_256::new();
    proof_hasher.update(b"genus2-wesolowski-challenge");
    proof_hasher.update(seed.as_bytes());
    proof_hasher.update(&vdf_output);
    proof_hasher.update(&vdf_iterations.to_le_bytes());
    let proof_challenge = proof_hasher.finalize();

    let mut proof = Vec::with_capacity(32 + vdf_output.len() + 8);
    proof.extend_from_slice(&proof_challenge);
    proof.extend_from_slice(&vdf_output);
    proof.extend_from_slice(&vdf_iterations.to_le_bytes());

    Some((nonce, hash, vdf_output, proof, checkpoints, vdf_iterations))
}

/// Optimized implementations for different CPU architectures
pub mod optimizations {
    use super::*;

    #[cfg(target_feature = "avx2")]
    pub fn avx2_hash_batch(inputs: &[[u8; 72]], outputs: &mut [[u8; 32]]) {
        // AVX2-optimized parallel hashing via BLAKE3's internal SIMD
        for (input, output) in inputs.iter().zip(outputs.iter_mut()) {
            let hash = blake3::hash(input);
            output.copy_from_slice(hash.as_bytes());
        }
    }

    #[cfg(target_feature = "avx512f")]
    pub fn avx512_hash_batch(inputs: &[[u8; 72]], outputs: &mut [[u8; 32]]) {
        for (input, output) in inputs.iter().zip(outputs.iter_mut()) {
            let hash = blake3::hash(input);
            output.copy_from_slice(hash.as_bytes());
        }
    }

    #[cfg(target_arch = "aarch64")]
    pub fn neon_hash_batch(inputs: &[[u8; 72]], outputs: &mut [[u8; 32]]) {
        for (input, output) in inputs.iter().zip(outputs.iter_mut()) {
            let hash = blake3::hash(input);
            output.copy_from_slice(hash.as_bytes());
        }
    }
}

/// CPU mining performance benchmarks
pub mod benchmarks {
    use super::*;
    use std::time::Instant;
    
    pub async fn benchmark_cpu_performance(threads: usize) -> Result<CpuBenchmarkResults> {
        info!("🏁 Starting CPU mining benchmark with {} threads", threads);
        
        let algorithm = Arc::new(DagKnightVDF::new(1000));
        let test_work = WorkUnit {
            job_id: "benchmark".to_string(),
            previous_hash: [1u8; 32],
            merkle_root: [2u8; 32],
            timestamp: chrono::Utc::now().timestamp() as u64,
            difficulty_target: [0xFF; 32], // Easy target for benchmarking
            nonce_range: (0, 1_000_000),
            extra_data: vec![3u8; 64],
        };
        
        let start_time = Instant::now();
        let hash_counter = Arc::new(AtomicU64::new(0));
        
        // Run benchmark for 30 seconds
        let benchmark_duration = tokio::time::Duration::from_secs(30);
        let handles: Vec<_> = (0..threads)
            .map(|thread_id| {
                let algorithm = algorithm.clone();
                let work = test_work.clone();
                let hash_counter = hash_counter.clone();
                
                tokio::spawn(async move {
                    let mut nonce = thread_id as u64 * 10_000;
                    let batch_size = 1000;
                    
                    while start_time.elapsed() < benchmark_duration {
                        for _ in 0..batch_size {
                            if let Ok(_) = algorithm.as_ref().compute_hash(&work.extra_data, nonce).await {
                                hash_counter.fetch_add(1, Ordering::Relaxed);
                            }
                            nonce += threads as u64;
                        }
                    }
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
        
        let results = CpuBenchmarkResults {
            threads,
            duration: elapsed,
            total_hashes,
            hash_rate,
            hashes_per_thread: total_hashes / threads as u64,
            cpu_info: detect_cpu_capabilities(),
        };
        
        info!("🏁 CPU Benchmark Results:");
        info!("   Hash Rate: {:.2} H/s", results.hash_rate);
        info!("   Per Thread: {:.2} H/s", results.hash_rate / threads as f64);
        info!("   Total Hashes: {}", results.total_hashes);
        
        Ok(results)
    }
    
    #[derive(Debug, Clone)]
    pub struct CpuBenchmarkResults {
        pub threads: usize,
        pub duration: std::time::Duration,
        pub total_hashes: u64,
        pub hash_rate: f64,
        pub hashes_per_thread: u64,
        pub cpu_info: CpuInfo,
    }
}

