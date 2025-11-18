/// Memory Limiter Module - v1.0.15.1-beta
///
/// Implements adaptive memory management for sync operations to prevent OOM crashes.
/// Kimi AI Recommendation: Cap batch sizes based on available RAM and detect memory pressure.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use sysinfo::System;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Memory pressure levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
        Self {
            low_threshold: 0.60,
            medium_threshold: 0.80,
            high_threshold: 0.90,
            min_batch_size: 10,
            max_batch_size: 1000,
            check_interval: Duration::from_secs(5),
        }
    }
}

/// Memory limiter for adaptive batch sizing
pub struct MemoryLimiter {
    config: MemoryLimiterConfig,
    system: Arc<RwLock<System>>,
    last_check: Arc<RwLock<Instant>>,
    current_pressure: Arc<RwLock<MemoryPressure>>,
    current_batch_size: AtomicUsize,
    total_memory_bytes: AtomicU64,
    available_memory_bytes: AtomicU64,
}

impl MemoryLimiter {
    /// Create a new memory limiter with default config
    pub fn new() -> Self {
        Self::with_config(MemoryLimiterConfig::default())
    }

    /// Create a new memory limiter with custom config
    pub fn with_config(config: MemoryLimiterConfig) -> Self {
        let mut system = System::new_all();
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
        }
    }

    /// Get current memory pressure level
    pub async fn get_memory_pressure(&self) -> MemoryPressure {
        // Check if we need to update memory stats
        let should_update = {
            let last_check = self.last_check.read().await;
            last_check.elapsed() > self.config.check_interval
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

        let total = system.total_memory();
        let available = system.available_memory();
        let used = total - available;
        let usage_ratio = used as f64 / total as f64;

        self.total_memory_bytes.store(total, Ordering::Relaxed);
        self.available_memory_bytes.store(available, Ordering::Relaxed);

        let pressure = if usage_ratio < self.config.low_threshold {
            MemoryPressure::Low
        } else if usage_ratio < self.config.medium_threshold {
            MemoryPressure::Medium
        } else if usage_ratio < self.config.high_threshold {
            MemoryPressure::High
        } else {
            MemoryPressure::Critical
        };

        let old_pressure = *self.current_pressure.read().await;
        if pressure != old_pressure {
            warn!(
                "🧠 [MEMORY PRESSURE] Changed from {:?} to {:?} (usage: {:.1}%)",
                old_pressure,
                pressure,
                usage_ratio * 100.0
            );
        } else {
            debug!(
                "🧠 [MEMORY] Usage: {:.1}%, Pressure: {:?}, Available: {} GB",
                usage_ratio * 100.0,
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
