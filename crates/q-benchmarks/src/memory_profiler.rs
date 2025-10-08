//! Memory Performance Profiling Module
//!
//! Comprehensive memory usage analysis and optimization detection
//! for Q-NarwhalKnight consensus system.

use crate::{BenchmarkConfig, MemoryMetrics};
use anyhow::Result;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use sysinfo::{Pid, ProcessExt, System, SystemExt};
use tracing::{debug, info, warn};

/// Memory profiler for tracking system memory usage
pub struct MemoryProfiler {
    system: System,
    baseline_memory: u64,
    peak_memory: u64,
    memory_samples: Vec<(Instant, u64)>,
}

impl MemoryProfiler {
    pub fn new() -> Self {
        let mut system = System::new_all();
        system.refresh_all();

        Self {
            system,
            baseline_memory: 0,
            peak_memory: 0,
            memory_samples: Vec::new(),
        }
    }

    /// Start memory profiling session
    pub fn start_profiling(&mut self) -> Result<()> {
        self.system.refresh_all();
        self.baseline_memory = self.get_current_memory_usage();
        self.peak_memory = self.baseline_memory;
        self.memory_samples.clear();

        info!(
            "🧠 Starting memory profiling - Baseline: {:.1}MB",
            self.baseline_memory as f64 / 1_048_576.0
        );

        Ok(())
    }

    /// Record memory sample
    pub fn record_sample(&mut self) {
        self.system.refresh_all();
        let current_memory = self.get_current_memory_usage();

        if current_memory > self.peak_memory {
            self.peak_memory = current_memory;
        }

        self.memory_samples.push((Instant::now(), current_memory));

        debug!(
            "Memory sample: {:.1}MB",
            current_memory as f64 / 1_048_576.0
        );
    }

    /// Get current process memory usage in bytes
    fn get_current_memory_usage(&self) -> u64 {
        let current_pid = Pid::from(std::process::id() as usize);

        if let Some(process) = self.system.process(current_pid) {
            process.memory() * 1024 // Convert KB to bytes
        } else {
            // Fallback to system memory usage
            self.system.used_memory()
        }
    }

    /// Calculate memory efficiency metrics
    pub fn calculate_metrics(&self) -> MemoryMetrics {
        let heap_usage_mb = self.get_current_memory_usage() as f64 / 1_048_576.0;
        let peak_memory_mb = self.peak_memory as f64 / 1_048_576.0;
        let baseline_mb = self.baseline_memory as f64 / 1_048_576.0;

        // Calculate memory efficiency (lower growth = better)
        let memory_efficiency = if peak_memory_mb > baseline_mb {
            1.0 - ((peak_memory_mb - baseline_mb) / peak_memory_mb)
        } else {
            1.0
        };

        // Calculate allocation rate from samples
        let allocation_rate = self.calculate_allocation_rate();

        // Estimate GC pressure based on memory variance
        let gc_pressure = self.calculate_gc_pressure();

        MemoryMetrics {
            heap_usage_mb,
            peak_memory_mb,
            memory_efficiency: memory_efficiency.max(0.0),
            gc_pressure,
            allocation_rate,
        }
    }

    /// Calculate memory allocation rate (MB/s)
    fn calculate_allocation_rate(&self) -> f64 {
        if self.memory_samples.len() < 2 {
            return 0.0;
        }

        let start = self.memory_samples.first().unwrap();
        let end = self.memory_samples.last().unwrap();

        let memory_delta = end.1 as f64 - start.1 as f64;
        let time_delta = end.0.duration_since(start.0).as_secs_f64();

        if time_delta > 0.0 {
            (memory_delta / 1_048_576.0) / time_delta // MB/s
        } else {
            0.0
        }
    }

    /// Calculate GC pressure based on memory usage variance
    fn calculate_gc_pressure(&self) -> f64 {
        if self.memory_samples.len() < 10 {
            return 0.0;
        }

        let values: Vec<f64> = self
            .memory_samples
            .iter()
            .map(|(_, mem)| *mem as f64)
            .collect();

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;

        let std_dev = variance.sqrt();

        // Normalize GC pressure (higher variance = more GC activity)
        (std_dev / mean).min(1.0)
    }
}

/// Measure memory performance during benchmark
pub async fn measure_memory_performance(config: &BenchmarkConfig) -> Result<MemoryMetrics> {
    info!("💾 Starting memory performance measurement");

    let mut profiler = MemoryProfiler::new();
    profiler.start_profiling()?;

    // Clone config for spawned tasks to satisfy 'static lifetime requirement
    let config_clone = config.clone();
    let duration = config.duration_seconds;

    // Run memory-intensive workload simulation
    let memory_task = tokio::spawn(async move { simulate_consensus_memory_load(&config_clone).await });

    // Sample memory usage during the workload
    let sampling_task =
        tokio::spawn(
            async move { sample_memory_usage(&mut profiler, duration).await },
        );

    // Wait for both tasks
    let (_, mut profiler) = tokio::try_join!(memory_task, sampling_task)?;

    let metrics = profiler.calculate_metrics();

    info!("✅ Memory Performance Results:");
    info!("   Current Usage: {:.1}MB", metrics.heap_usage_mb);
    info!("   Peak Usage: {:.1}MB", metrics.peak_memory_mb);
    info!(
        "   Memory Efficiency: {:.1}%",
        metrics.memory_efficiency * 100.0
    );
    info!("   Allocation Rate: {:.2}MB/s", metrics.allocation_rate);
    info!("   GC Pressure: {:.3}", metrics.gc_pressure);

    if metrics.peak_memory_mb > 1000.0 {
        warn!("⚠️ High memory usage detected - optimization needed");
    }

    Ok(metrics)
}

/// Sample memory usage at regular intervals
async fn sample_memory_usage(
    profiler: &mut MemoryProfiler,
    duration_seconds: u64,
) -> MemoryProfiler {
    let sample_interval = Duration::from_millis(500); // Sample every 500ms
    let end_time = Instant::now() + Duration::from_secs(duration_seconds);

    while Instant::now() < end_time {
        profiler.record_sample();
        tokio::time::sleep(sample_interval).await;
    }

    // Take ownership to return from async context
    MemoryProfiler {
        system: System::new_all(),
        baseline_memory: profiler.baseline_memory,
        peak_memory: profiler.peak_memory,
        memory_samples: profiler.memory_samples.clone(),
    }
}

/// Simulate memory-intensive consensus operations
async fn simulate_consensus_memory_load(config: &BenchmarkConfig) -> Result<()> {
    info!("🔄 Simulating consensus memory workload");

    let mut vertex_storage = Vec::new();
    let mut state_cache = HashMap::new();

    let operations_per_second = config.target_tps as usize / 10; // Reduced for memory focus
    let total_operations = operations_per_second * config.duration_seconds as usize;

    for i in 0..total_operations {
        // Simulate vertex creation and storage
        let vertex = simulate_vertex_creation(i);
        vertex_storage.push(vertex);

        // Simulate state caching
        let cache_key = format!("state_{}", i % 1000);
        let cache_value = simulate_state_data(i);
        state_cache.insert(cache_key, cache_value);

        // Periodic cleanup to simulate GC behavior
        if i % 10000 == 0 {
            vertex_storage.truncate(5000); // Keep recent vertices
            if state_cache.len() > 5000 {
                state_cache.clear(); // Simulate cache eviction
            }
        }

        // Rate limiting
        if i % operations_per_second == 0 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    info!("✅ Completed consensus memory simulation");
    info!("   Vertices created: {}", total_operations);
    info!("   Final vertex storage: {}", vertex_storage.len());
    info!("   Final cache entries: {}", state_cache.len());

    Ok(())
}

/// Simulate vertex creation (memory allocation)
fn simulate_vertex_creation(id: usize) -> Vec<u8> {
    // Simulate realistic vertex size (~2KB)
    let vertex_size = 2048;
    let mut vertex = vec![0u8; vertex_size];

    // Add some realistic data patterns
    for (i, byte) in vertex.iter_mut().enumerate() {
        *byte = ((id + i) % 256) as u8;
    }

    vertex
}

/// Simulate state data creation
fn simulate_state_data(id: usize) -> HashMap<String, Vec<u8>> {
    let mut state = HashMap::new();

    // Simulate multiple state entries per operation
    for i in 0..5 {
        let key = format!("entry_{}_{}", id, i);
        let value = vec![(id + i) as u8; 256]; // 256 bytes per entry
        state.insert(key, value);
    }

    state
}

/// Analyze memory usage patterns and detect leaks
pub fn analyze_memory_patterns(samples: &[(Instant, u64)]) -> HashMap<String, f64> {
    let mut analysis = HashMap::new();

    if samples.len() < 2 {
        return analysis;
    }

    // Calculate memory growth trend
    let start_memory = samples.first().unwrap().1 as f64;
    let end_memory = samples.last().unwrap().1 as f64;
    let growth_rate = (end_memory - start_memory) / start_memory;

    analysis.insert("growth_rate".to_string(), growth_rate);

    // Detect potential memory leaks (consistent upward trend)
    let mut increases = 0;
    let mut decreases = 0;

    for window in samples.windows(2) {
        if window[1].1 > window[0].1 {
            increases += 1;
        } else {
            decreases += 1;
        }
    }

    let leak_indicator = increases as f64 / (increases + decreases) as f64;
    analysis.insert("leak_indicator".to_string(), leak_indicator);

    // Calculate memory stability (lower variance = more stable)
    let mean_memory: f64 =
        samples.iter().map(|(_, mem)| *mem as f64).sum::<f64>() / samples.len() as f64;
    let variance: f64 = samples
        .iter()
        .map(|(_, mem)| (*mem as f64 - mean_memory).powi(2))
        .sum::<f64>()
        / samples.len() as f64;

    let stability = 1.0 - (variance.sqrt() / mean_memory);
    analysis.insert("stability".to_string(), stability.max(0.0));

    analysis
}
