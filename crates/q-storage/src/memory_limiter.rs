/// Memory Limiter Module - v1.0.15.1-beta
///
/// Implements adaptive memory management for sync operations to prevent OOM crashes.
/// Kimi AI Recommendation: Cap batch sizes based on available RAM and detect memory pressure.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use sysinfo::{Pid, ProcessRefreshKind, System};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// v1.5.0-beta: Throughput sample for feedback-driven batch sizing
#[derive(Debug, Clone)]
pub struct ThroughputSample {
    /// Batch size used
    pub batch_size: usize,
    /// Blocks processed per second
    pub blocks_per_sec: f64,
    /// Memory used during batch (bytes)
    pub memory_bytes: u64,
    /// Timestamp of sample
    pub timestamp: Instant,
}

/// Default bytes per block estimate (4.6KB average for Q-NarwhalKnight)
const DEFAULT_BLOCK_BYTES: u64 = 4_700;

/// Maximum throughput samples to retain (sliding window)
const MAX_THROUGHPUT_SAMPLES: usize = 50;

/// Fast check interval during active sync (100ms)
const ACTIVE_SYNC_CHECK_INTERVAL: Duration = Duration::from_millis(100);

/// Maximum memory budget for sync operations (70% of available RAM)
const DEFAULT_SYNC_MEMORY_BUDGET_PCT: f64 = 0.70;

/// Memory pressure levels.
/// v10.2.10: Added PartialOrd, Ord — declaration order matches severity (Low < Critical).
/// This allows std::cmp::max(rss_pressure, swap_pressure) for combined pressure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MemoryPressure {
    Low,      // < 60% memory usage
    Medium,   // 60-80% memory usage
    High,     // 80-90% memory usage
    Critical, // > 90% memory usage
}

/// Configuration for memory limiter
#[derive(Debug, Clone)]
pub struct MemoryLimiterConfig {
    /// Low memory threshold (default: 60%)
    pub low_threshold: f64,
    /// Medium memory threshold (default: 80%)
    pub medium_threshold: f64,
    /// High memory threshold (default: 90%)
    pub high_threshold: f64,
    /// Minimum batch size (default: 10 blocks)
    pub min_batch_size: usize,
    /// Maximum batch size (default: 1000 blocks)
    pub max_batch_size: usize,
    /// Memory check interval (default: 5 seconds)
    pub check_interval: Duration,
}

impl Default for MemoryLimiterConfig {
    fn default() -> Self {
        // v6.0.5: RAM-aware batch sizes to prevent OOM on small nodes
        let ram_mb = {
            use sysinfo::System;
            let mut sys = System::new();
            sys.refresh_memory();
            (sys.total_memory() / (1024 * 1024)) as usize
        };
        // v9.2.0: Reduced medium tier batch sizes to prevent OOM on 16GB nodes
        // During sync, each block is ~50-200KB in memory (with Vecs, signatures, etc.)
        // 3000 blocks × 100KB = 300MB per batch — too much for 16GB with other overhead
        let (min_batch, max_batch) = match ram_mb {
            0..=3999     => (100, 500),     // micro: very conservative
            4000..=7999  => (200, 1000),    // small (Gamma 7.8GB): reduced from 5000
            8000..=15999 => (200, 1000),    // v9.2.0: was 500/3000 → OOM on 16GB nodes during sync
            _            => (500, 5000),    // large: original defaults
        };
        // v6.0.6: RAM-aware thresholds — lower for small nodes to trigger backpressure
        // before OOM kills the process
        // v9.2.0: Lowered medium tier thresholds (was 55/72/85 → OOM at 14.4/15.6GB)
        let (low_t, med_t, high_t) = match ram_mb {
            0..=3999     => (0.40, 0.55, 0.70),  // micro: aggressive backpressure
            4000..=7999  => (0.45, 0.60, 0.72),  // small (Gamma): trigger well before 6G cgroup limit
            8000..=15999 => (0.45, 0.60, 0.75),  // v9.2.0: was 55/72/85, too late for 16GB nodes
            _            => (0.60, 0.80, 0.90),  // large: original thresholds
        };
        Self {
            low_threshold: low_t,
            medium_threshold: med_t,
            high_threshold: high_t,
            min_batch_size: min_batch,
            max_batch_size: max_batch,
            check_interval: Duration::from_secs(5),
        }
    }
}

/// Memory limiter for adaptive batch sizing
///
/// v1.5.0-beta: Enhanced with actual block size tracking and throughput feedback
pub struct MemoryLimiter {
    config: MemoryLimiterConfig,
    system: Arc<RwLock<System>>,
    last_check: Arc<RwLock<Instant>>,
    current_pressure: Arc<RwLock<MemoryPressure>>,
    current_batch_size: AtomicUsize,
    total_memory_bytes: AtomicU64,
    available_memory_bytes: AtomicU64,

    // v1.5.0-beta: Dynamic block size tracking
    /// Observed average bytes per block (EMA with α=0.1)
    avg_block_bytes: AtomicU64,
    /// Last N throughput samples for feedback-driven adjustment
    throughput_samples: Arc<RwLock<VecDeque<ThroughputSample>>>,
    /// Whether we're in active sync mode (uses faster check intervals)
    active_sync_mode: Arc<std::sync::atomic::AtomicBool>,
    /// Process RSS bytes (more accurate than system memory for self-limiting)
    process_rss_bytes: AtomicU64,
    /// Target memory budget for sync operations (percentage of available)
    sync_memory_budget_pct: f64,
}

impl MemoryLimiter {
    /// Create a new memory limiter with default config
    pub fn new() -> Self {
        Self::with_config(MemoryLimiterConfig::default())
    }

    /// Create a new memory limiter with custom config
    pub fn with_config(config: MemoryLimiterConfig) -> Self {
        let mut system = System::new();
        system.refresh_memory();

        let total_memory = system.total_memory();
        let max_batch_size = config.max_batch_size;

        info!("🧠 [MEMORY LIMITER] Initialized");
        info!("   Total RAM: {} GB", total_memory / (1024 * 1024 * 1024));
        info!("   Low threshold: {:.0}%", config.low_threshold * 100.0);
        info!("   Medium threshold: {:.0}%", config.medium_threshold * 100.0);
        info!("   High threshold: {:.0}%", config.high_threshold * 100.0);
        info!("   Batch size range: {}-{}", config.min_batch_size, config.max_batch_size);

        Self {
            config,
            system: Arc::new(RwLock::new(system)),
            last_check: Arc::new(RwLock::new(Instant::now())),
            current_pressure: Arc::new(RwLock::new(MemoryPressure::Low)),
            current_batch_size: AtomicUsize::new(max_batch_size),
            total_memory_bytes: AtomicU64::new(total_memory),
            available_memory_bytes: AtomicU64::new(total_memory),
            // v1.5.0-beta: Initialize new dynamic sizing fields
            avg_block_bytes: AtomicU64::new(DEFAULT_BLOCK_BYTES),
            throughput_samples: Arc::new(RwLock::new(VecDeque::with_capacity(MAX_THROUGHPUT_SAMPLES))),
            active_sync_mode: Arc::new(AtomicBool::new(false)),
            process_rss_bytes: AtomicU64::new(0),
            sync_memory_budget_pct: DEFAULT_SYNC_MEMORY_BUDGET_PCT,
        }
    }

    /// Get current memory pressure level
    pub async fn get_memory_pressure(&self) -> MemoryPressure {
        // v1.5.0-beta: Use faster check intervals during active sync
        let check_interval = if self.active_sync_mode.load(Ordering::Relaxed) {
            ACTIVE_SYNC_CHECK_INTERVAL
        } else {
            self.config.check_interval
        };

        // Check if we need to update memory stats
        let should_update = {
            let last_check = self.last_check.read().await;
            last_check.elapsed() > check_interval
        };

        if should_update {
            self.update_memory_stats().await;
        }

        *self.current_pressure.read().await
    }

    /// Get recommended batch size based on current memory pressure
    pub async fn get_recommended_batch_size(&self) -> usize {
        let pressure = self.get_memory_pressure().await;

        let batch_size = match pressure {
            MemoryPressure::Low => self.config.max_batch_size,
            MemoryPressure::Medium => {
                // Scale down to 50% of max
                (self.config.max_batch_size / 2).max(self.config.min_batch_size)
            }
            MemoryPressure::High => {
                // Scale down to 25% of max
                (self.config.max_batch_size / 4).max(self.config.min_batch_size)
            }
            MemoryPressure::Critical => {
                // Use minimum batch size
                self.config.min_batch_size
            }
        };

        self.current_batch_size.store(batch_size, Ordering::Relaxed);
        batch_size
    }

    /// Update memory statistics and pressure level
    async fn update_memory_stats(&self) {
        let mut system = self.system.write().await;
        system.refresh_memory();
        // v1.5.0-beta: Also refresh processes for RSS tracking
        system.refresh_processes_specifics(ProcessRefreshKind::new().with_memory());

        let total = system.total_memory();
        let available = system.available_memory();
        let used = total - available;
        let usage_ratio = used as f64 / total as f64;

        // v1.5.0-beta: Track process RSS
        self.update_process_rss(&system);

        self.total_memory_bytes.store(total, Ordering::Relaxed);
        self.available_memory_bytes.store(available, Ordering::Relaxed);

        let ram_pressure = if usage_ratio < self.config.low_threshold {
            MemoryPressure::Low
        } else if usage_ratio < self.config.medium_threshold {
            MemoryPressure::Medium
        } else if usage_ratio < self.config.high_threshold {
            MemoryPressure::High
        } else {
            MemoryPressure::Critical
        };

        // v10.2.10: Include swap in pressure calculation.
        // The 2026-04-11 incident had 55% RSS but 100% swap exhaustion.
        //
        // v10.3.9: Skip host-level swap pressure when THIS process uses zero swap.
        // Running inside Docker, sysinfo reads HOST /proc/meminfo for swap stats.
        // If the host's swap is full (e.g. from other production processes) but
        // this container process itself has VmSwap=0, host swap exhaustion cannot
        // cause THIS process to crash — so don't throttle based on it.
        //
        // v10.4.8: Cap swap pressure at Medium when process has little of its own
        // footprint on disk (< 25% of RSS+swap). If a container process has 3 GB
        // RSS and only 200 MB swapped, it is NOT swap-thrashing — other processes
        // filled the host swap. Declaring Critical here stalls turbo sync forever
        // because the condition never self-heals. Only escalate beyond Medium when
        // the process itself is heavily swap-dependent.
        let process_vm_swap_kb = Self::read_process_vm_swap_kb();
        let swap_total = system.total_swap();
        let swap_used = system.used_swap();
        let swap_pressure = if swap_total == 0 {
            MemoryPressure::Low // No swap configured
        } else if process_vm_swap_kb == 0 {
            // This process uses no swap — host swap pressure is irrelevant
            MemoryPressure::Low
        } else {
            let swap_ratio = swap_used as f64 / swap_total as f64;
            let raw_pressure = if swap_ratio < 0.60 {
                MemoryPressure::Low
            } else if swap_ratio < 0.80 {
                MemoryPressure::Medium
            } else if swap_ratio < 0.95 {
                MemoryPressure::High
            } else {
                MemoryPressure::Critical
            };
            // v10.4.8: If process's own swap is < 25% of its total memory footprint
            // (RSS + swap), its active pages are mostly in RAM — host swap exhaustion
            // from other processes won't cause THIS process to crash. Cap at Medium.
            let process_rss_kb = self.process_rss_bytes.load(Ordering::Relaxed) / 1024;
            let process_total_kb = process_rss_kb + process_vm_swap_kb;
            let process_swap_pct = if process_total_kb > 0 {
                process_vm_swap_kb as f64 / process_total_kb as f64
            } else {
                0.0
            };
            if process_swap_pct < 0.25 {
                std::cmp::min(raw_pressure, MemoryPressure::Medium)
            } else {
                raw_pressure
            }
        };

        // Return the more severe of RAM and swap pressure
        let pressure = std::cmp::max(ram_pressure, swap_pressure);

        let old_pressure = *self.current_pressure.read().await;
        if pressure != old_pressure {
            warn!(
                "🧠 [MEMORY PRESSURE] Changed from {:?} to {:?} (RAM: {:.1}%, Swap: {:.1}%)",
                old_pressure,
                pressure,
                usage_ratio * 100.0,
                if swap_total > 0 { swap_used as f64 / swap_total as f64 * 100.0 } else { 0.0 }
            );
        } else {
            debug!(
                "🧠 [MEMORY] RAM: {:.1}%, Swap: {:.1}%, Pressure: {:?}, Available: {} GB",
                usage_ratio * 100.0,
                if swap_total > 0 { swap_used as f64 / swap_total as f64 * 100.0 } else { 0.0 },
                pressure,
                available / (1024 * 1024 * 1024)
            );
        }

        *self.current_pressure.write().await = pressure;
        *self.last_check.write().await = Instant::now();
    }

    /// Get memory usage statistics
    pub async fn get_memory_stats(&self) -> MemoryStats {
        // Ensure stats are fresh
        self.update_memory_stats().await;

        let total = self.total_memory_bytes.load(Ordering::Relaxed);
        let available = self.available_memory_bytes.load(Ordering::Relaxed);
        let used = total - available;
        let usage_ratio = used as f64 / total as f64;

        MemoryStats {
            total_bytes: total,
            used_bytes: used,
            available_bytes: available,
            usage_ratio,
            pressure: *self.current_pressure.read().await,
            current_batch_size: self.current_batch_size.load(Ordering::Relaxed),
        }
    }

    /// v10.2.10: Get swap usage as a percentage (0-100).
    /// Returns 0 if no swap configured.
    pub async fn get_swap_usage_percent(&self) -> u8 {
        let system = self.system.read().await;
        let swap_total = system.total_swap();
        if swap_total == 0 { return 0; }
        let swap_used = system.used_swap();
        ((swap_used as f64 / swap_total as f64) * 100.0).min(100.0) as u8
    }

    /// Check if sync operation should pause due to memory pressure
    pub async fn should_pause_sync(&self) -> bool {
        let pressure = self.get_memory_pressure().await;
        pressure == MemoryPressure::Critical
    }

    /// Wait until memory pressure decreases
    pub async fn wait_for_memory_relief(&self) {
        let mut backoff = Duration::from_millis(100);

        loop {
            let pressure = self.get_memory_pressure().await;

            if pressure != MemoryPressure::Critical {
                info!("✅ [MEMORY RELIEF] Memory pressure decreased to {:?}", pressure);
                break;
            }

            warn!(
                "⏸️  [MEMORY CRITICAL] Pausing sync operations, waiting {:?}",
                backoff
            );

            tokio::time::sleep(backoff).await;

            // Exponential backoff up to 10 seconds
            backoff = (backoff * 2).min(Duration::from_secs(10));
        }
    }

    // ========================================================================
    // v1.5.0-beta: Dynamic Memory-Based Batch Sizing
    // ========================================================================

    /// Enable or disable active sync mode (faster memory checks)
    pub fn set_active_sync_mode(&self, active: bool) {
        let was_active = self.active_sync_mode.swap(active, Ordering::SeqCst);
        if was_active != active {
            info!(
                "🧠 [MEMORY] Active sync mode: {} → {}",
                was_active, active
            );
        }
    }

    /// Record a throughput sample for feedback-driven adjustment
    pub async fn record_throughput_sample(&self, batch_size: usize, blocks_processed: usize, duration: Duration) {
        if duration.as_secs_f64() == 0.0 || blocks_processed == 0 {
            return;
        }

        let blocks_per_sec = blocks_processed as f64 / duration.as_secs_f64();
        let memory_bytes = self.process_rss_bytes.load(Ordering::Relaxed);

        let sample = ThroughputSample {
            batch_size,
            blocks_per_sec,
            memory_bytes,
            timestamp: Instant::now(),
        };

        let mut samples = self.throughput_samples.write().await;
        samples.push_back(sample);

        // Keep only last N samples
        while samples.len() > MAX_THROUGHPUT_SAMPLES {
            samples.pop_front();
        }

        debug!(
            "📊 [THROUGHPUT] Recorded: {} blocks in {:?} ({:.1} blocks/s), batch_size={}",
            blocks_processed, duration, blocks_per_sec, batch_size
        );
    }

    /// Update average block size based on observed data (EMA with α=0.1)
    pub fn update_block_size_estimate(&self, total_bytes: u64, num_blocks: usize) {
        if num_blocks == 0 {
            return;
        }

        let observed_avg = total_bytes / num_blocks as u64;
        let current_avg = self.avg_block_bytes.load(Ordering::Relaxed);

        // EMA: new_avg = α * observed + (1-α) * current
        // α = 0.1 for smooth adaptation
        let alpha = 0.1_f64;
        let new_avg = (alpha * observed_avg as f64 + (1.0 - alpha) * current_avg as f64) as u64;

        self.avg_block_bytes.store(new_avg, Ordering::Relaxed);

        debug!(
            "📏 [BLOCK SIZE] Updated: {} bytes → {} bytes (observed: {} bytes for {} blocks)",
            current_avg, new_avg, observed_avg, num_blocks
        );
    }

    /// Get optimal batch size based on available memory budget
    ///
    /// This calculates how many blocks can fit in the sync memory budget
    /// based on the observed average block size.
    pub async fn get_memory_budget_batch_size(&self) -> usize {
        // Ensure stats are fresh
        self.update_memory_stats().await;

        let available = self.available_memory_bytes.load(Ordering::Relaxed);
        let avg_block = self.avg_block_bytes.load(Ordering::Relaxed);

        // Calculate memory budget for sync
        let budget_bytes = (available as f64 * self.sync_memory_budget_pct) as u64;

        // Account for in-flight data (2x multiplier: compressed + decompressed)
        let effective_block_bytes = avg_block * 2;

        // Calculate max blocks that fit in budget
        let max_blocks = if effective_block_bytes > 0 {
            (budget_bytes / effective_block_bytes) as usize
        } else {
            self.config.max_batch_size
        };

        // Clamp to configured range
        let clamped = max_blocks
            .max(self.config.min_batch_size)
            .min(self.config.max_batch_size);

        debug!(
            "🧠 [BUDGET] Available: {:.1} GB, Budget: {:.1} MB, Block avg: {} B, Max blocks: {}",
            available as f64 / 1e9,
            budget_bytes as f64 / 1e6,
            avg_block,
            clamped
        );

        clamped
    }

    /// Get throughput-optimized batch size based on recent performance
    ///
    /// Analyzes recent throughput samples to find the batch size that
    /// maximizes blocks/second while staying within memory budget.
    pub async fn get_throughput_optimized_batch_size(&self) -> Option<usize> {
        let samples = self.throughput_samples.read().await;

        if samples.len() < 5 {
            // Not enough data for optimization
            return None;
        }

        // Find batch size with best throughput in recent samples (last 20)
        let recent_samples: Vec<_> = samples.iter().rev().take(20).collect();

        // Group by batch size ranges and find best performer
        let mut best_throughput = 0.0_f64;
        let mut best_batch_size = self.config.max_batch_size;

        for sample in &recent_samples {
            if sample.blocks_per_sec > best_throughput {
                best_throughput = sample.blocks_per_sec;
                best_batch_size = sample.batch_size;
            }
        }

        // Only recommend if we've seen good performance
        if best_throughput > 100.0 {
            info!(
                "📊 [THROUGHPUT OPT] Best: {} blocks/s at batch_size={}",
                best_throughput as usize, best_batch_size
            );
            Some(best_batch_size)
        } else {
            None
        }
    }

    /// Get dynamically optimized batch size combining all signals
    ///
    /// Priority order:
    /// 1. Memory pressure constraints (hard limit)
    /// 2. Memory budget calculation (soft limit)
    /// 3. Throughput optimization (preference)
    /// 4. Pressure-based scaling (fallback)
    pub async fn get_dynamic_batch_size(&self) -> usize {
        let pressure = self.get_memory_pressure().await;

        // Critical pressure: use minimum
        if pressure == MemoryPressure::Critical {
            return self.config.min_batch_size;
        }

        // Calculate memory budget limit
        let budget_limit = self.get_memory_budget_batch_size().await;

        // Try throughput optimization
        let throughput_opt = self.get_throughput_optimized_batch_size().await;

        // Get pressure-based recommendation
        let pressure_based = match pressure {
            MemoryPressure::Low => self.config.max_batch_size,
            MemoryPressure::Medium => self.config.max_batch_size / 2,
            MemoryPressure::High => self.config.max_batch_size / 4,
            MemoryPressure::Critical => self.config.min_batch_size,
        };

        // Combine: throughput preference capped by budget and pressure
        let optimal = throughput_opt.unwrap_or(pressure_based);
        let final_size = optimal.min(budget_limit).min(pressure_based)
            .max(self.config.min_batch_size);

        self.current_batch_size.store(final_size, Ordering::Relaxed);

        debug!(
            "🎯 [DYNAMIC BATCH] pressure={:?}, budget={}, throughput={:?}, final={}",
            pressure, budget_limit, throughput_opt, final_size
        );

        final_size
    }

    /// Update process RSS (resident set size) for more accurate memory tracking
    fn update_process_rss(&self, system: &System) {
        let pid = std::process::id();
        if let Some(process) = system.process(Pid::from_u32(pid)) {
            let rss = process.memory(); // in bytes
            self.process_rss_bytes.store(rss, Ordering::Relaxed);
        }
    }

    /// Get current process RSS in bytes
    pub fn get_process_rss(&self) -> u64 {
        self.process_rss_bytes.load(Ordering::Relaxed)
    }

    /// Read VmSwap from /proc/self/status (kB). Returns 0 on any error.
    /// Used to detect whether THIS process is actually using swap (vs host swap).
    fn read_process_vm_swap_kb() -> u64 {
        let Ok(status) = std::fs::read_to_string("/proc/self/status") else {
            return 0;
        };
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("VmSwap:") {
                return rest.split_whitespace()
                    .next()
                    .and_then(|v| v.parse::<u64>().ok())
                    .unwrap_or(0);
            }
        }
        0
    }

    /// Get average block size estimate in bytes
    pub fn get_avg_block_bytes(&self) -> u64 {
        self.avg_block_bytes.load(Ordering::Relaxed)
    }
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_bytes: u64,
    pub used_bytes: u64,
    pub available_bytes: u64,
    pub usage_ratio: f64,
    pub pressure: MemoryPressure,
    pub current_batch_size: usize,
}

impl MemoryStats {
    pub fn total_gb(&self) -> f64 {
        self.total_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    pub fn used_gb(&self) -> f64 {
        self.used_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    pub fn available_gb(&self) -> f64 {
        self.available_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    pub fn usage_percent(&self) -> f64 {
        self.usage_ratio * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_memory_limiter_initialization() {
        let limiter = MemoryLimiter::new();
        let stats = limiter.get_memory_stats().await;

        assert!(stats.total_bytes > 0);
        assert!(stats.usage_ratio >= 0.0 && stats.usage_ratio <= 1.0);
    }

    #[tokio::test]
    async fn test_batch_size_adaptation() {
        let config = MemoryLimiterConfig {
            low_threshold: 0.01,     // Very low threshold
            medium_threshold: 0.50,
            high_threshold: 0.90,
            min_batch_size: 10,
            max_batch_size: 1000,
            check_interval: Duration::from_millis(100),
        };

        let limiter = MemoryLimiter::with_config(config);
        let batch_size = limiter.get_recommended_batch_size().await;

        // Should adapt based on actual memory pressure
        assert!(batch_size >= 10 && batch_size <= 1000);
    }

    #[tokio::test]
    async fn test_memory_pressure_detection() {
        let limiter = MemoryLimiter::new();
        let pressure = limiter.get_memory_pressure().await;

        // Should return a valid pressure level
        match pressure {
            MemoryPressure::Low | MemoryPressure::Medium |
            MemoryPressure::High | MemoryPressure::Critical => {},
        }
    }
}
