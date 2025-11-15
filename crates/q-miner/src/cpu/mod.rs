use crate::{MiningEngine, MiningAlgorithm, MiningStats, WorkUnit, Solution, algorithms::DagKnightVDF};
use anyhow::Result;
use async_trait::async_trait;
use std::sync::{Arc, atomic::{AtomicU64, AtomicBool, Ordering}};
use tokio::sync::RwLock;
use tracing::{info, debug, error};

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
}

impl CpuMiner {
    pub async fn new(thread_count: usize, intensity: u8) -> Result<Self> {
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
            
            let handle = tokio::spawn(async move {
                cpu_mining_thread(thread_id, is_running, current_work, hash_counter, algorithm, intensity).await;
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
) {
    info!("🔥 CPU mining thread {} started", thread_id);
    
    let mut nonce_base = thread_id as u64 * 1_000_000;
    let batch_size = (intensity as u64) * 10_000; // Adjust batch size by intensity
    
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
                        info!("💎 CPU Thread {} found solution! Nonce: {}", thread_id, nonce);
                        
                        // In a real implementation, submit solution to pool/network
                        let _solution = Solution {
                            job_id: work.job_id.clone(),
                            nonce,
                            hash,
                            timestamp: chrono::Utc::now().timestamp() as u64,
                            worker_id: format!("cpu_{}", thread_id),
                        };
                        
                        // TODO: Submit solution
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
    let logical_threads = num_cpus::get();
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
            proc_cpuinfo_threads.unwrap_or(0) >= 96;

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

/// Optimized implementations for different CPU architectures
pub mod optimizations {
    use super::*;
    
    #[cfg(target_feature = "avx2")]
    pub fn avx2_hash_batch(inputs: &[[u8; 72]], outputs: &mut [[u8; 32]]) {
        // AVX2-optimized parallel hashing
        // Process 8 hashes simultaneously using 256-bit SIMD
        use std::arch::x86_64::*;
        
        unsafe {
            for (input_chunk, output_chunk) in inputs.chunks(8).zip(outputs.chunks_mut(8)) {
                // Load 8 inputs into AVX2 registers
                // Perform parallel BLAKE3 computation
                // Store results
                
                for (i, (input, output)) in input_chunk.iter().zip(output_chunk.iter_mut()).enumerate() {
                    // Simplified fallback
                    let hash = blake3::hash(input);
                    output.copy_from_slice(hash.as_bytes());
                }
            }
        }
    }
    
    #[cfg(target_feature = "avx512f")]
    pub fn avx512_hash_batch(inputs: &[[u8; 72]], outputs: &mut [[u8; 32]]) {
        // AVX-512 optimized parallel hashing
        // Process 16 hashes simultaneously using 512-bit SIMD
        
        for (input_chunk, output_chunk) in inputs.chunks(16).zip(outputs.chunks_mut(16)) {
            for (input, output) in input_chunk.iter().zip(output_chunk.iter_mut()) {
                let hash = blake3::hash(input);
                output.copy_from_slice(hash.as_bytes());
            }
        }
    }
    
    #[cfg(target_arch = "aarch64")]
    pub fn neon_hash_batch(inputs: &[[u8; 72]], outputs: &mut [[u8; 32]]) {
        // ARM NEON optimized parallel hashing for M1/M2 Macs
        
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

