/// Storage performance metrics and monitoring for Q-Storage
/// Tracks write/read latencies, throughput, and database health

use serde::{Deserialize, Serialize};
use std::{
    collections::VecDeque,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::RwLock;
use tracing::debug;

/// Storage performance metrics collector
#[derive(Debug)]
pub struct StorageMetrics {
    /// Write latency history (last 1000 operations)
    write_latency_history: Arc<RwLock<VecDeque<Duration>>>,
    
    /// Read latency history (last 1000 operations)
    read_latency_history: Arc<RwLock<VecDeque<Duration>>>,
    
    /// Total operations counters
    total_writes: Arc<RwLock<u64>>,
    total_reads: Arc<RwLock<u64>>,
    total_vertices: Arc<RwLock<u64>>,
    total_payloads: Arc<RwLock<u64>>,
    total_blocks: Arc<RwLock<u64>>,
    
    /// Throughput tracking
    bytes_written: Arc<RwLock<u64>>,
    bytes_read: Arc<RwLock<u64>>,
    
    /// Error counters
    write_errors: Arc<RwLock<u64>>,
    read_errors: Arc<RwLock<u64>>,
    
    /// Last update time
    last_update: Arc<RwLock<Instant>>,
}

impl StorageMetrics {
    /// Create new metrics collector
    pub fn new() -> Self {
        Self {
            write_latency_history: Arc::new(RwLock::new(VecDeque::with_capacity(1000))),
            read_latency_history: Arc::new(RwLock::new(VecDeque::with_capacity(1000))),
            total_writes: Arc::new(RwLock::new(0)),
            total_reads: Arc::new(RwLock::new(0)),
            total_vertices: Arc::new(RwLock::new(0)),
            total_payloads: Arc::new(RwLock::new(0)),
            total_blocks: Arc::new(RwLock::new(0)),
            bytes_written: Arc::new(RwLock::new(0)),
            bytes_read: Arc::new(RwLock::new(0)),
            write_errors: Arc::new(RwLock::new(0)),
            read_errors: Arc::new(RwLock::new(0)),
            last_update: Arc::new(RwLock::new(Instant::now())),
        }
    }

    /// Record vertex write operation
    pub async fn record_vertex_write(&self, latency: Duration, vertex_size: usize, payload_size: usize) {
        debug!("📊 Recording vertex write: {}ms, vertex: {}B, payload: {}B", 
               latency.as_millis(), vertex_size, payload_size);

        // Record write latency
        {
            let mut history = self.write_latency_history.write().await;
            if history.len() >= 1000 {
                history.pop_front();
            }
            history.push_back(latency);
        }

        // Update counters
        {
            let mut writes = self.total_writes.write().await;
            *writes += 1;
        }

        {
            let mut vertices = self.total_vertices.write().await;
            *vertices += 1;
        }

        {
            let mut payloads = self.total_payloads.write().await;
            *payloads += 1;
        }

        {
            let mut bytes = self.bytes_written.write().await;
            *bytes += vertex_size as u64 + payload_size as u64;
        }

        {
            let mut last_update = self.last_update.write().await;
            *last_update = Instant::now();
        }
    }

    /// Record block finalization
    pub async fn record_block_finalization(&self, latency: Duration, tx_count: usize) {
        debug!("📊 Recording block finalization: {}ms, {} transactions", 
               latency.as_millis(), tx_count);

        // Record write latency
        {
            let mut history = self.write_latency_history.write().await;
            if history.len() >= 1000 {
                history.pop_front();
            }
            history.push_back(latency);
        }

        // Update counters
        {
            let mut blocks = self.total_blocks.write().await;
            *blocks += 1;
        }

        {
            let mut writes = self.total_writes.write().await;
            *writes += 1;
        }

        {
            let mut last_update = self.last_update.write().await;
            *last_update = Instant::now();
        }
    }

    /// Record read operation
    pub async fn record_read(&self, latency: Duration, bytes_read: usize) {
        debug!("📊 Recording read: {}ms, {}B", latency.as_millis(), bytes_read);

        // Record read latency
        {
            let mut history = self.read_latency_history.write().await;
            if history.len() >= 1000 {
                history.pop_front();
            }
            history.push_back(latency);
        }

        // Update counters
        {
            let mut reads = self.total_reads.write().await;
            *reads += 1;
        }

        {
            let mut bytes = self.bytes_read.write().await;
            *bytes += bytes_read as u64;
        }

        {
            let mut last_update = self.last_update.write().await;
            *last_update = Instant::now();
        }
    }

    /// Record write error
    pub async fn record_write_error(&self) {
        let mut errors = self.write_errors.write().await;
        *errors += 1;
        debug!("❌ Recorded write error (total: {})", *errors);
    }

    /// Record read error  
    pub async fn record_read_error(&self) {
        let mut errors = self.read_errors.write().await;
        *errors += 1;
        debug!("❌ Recorded read error (total: {})", *errors);
    }

    /// Get current metrics snapshot
    pub async fn get_current_metrics(&self) -> StorageMetricsSnapshot {
        let write_history = self.write_latency_history.read().await;
        let read_history = self.read_latency_history.read().await;

        // Calculate average latencies
        let average_write_latency = if write_history.is_empty() {
            Duration::from_millis(0)
        } else {
            let total_ms: u64 = write_history.iter().map(|d| d.as_millis() as u64).sum();
            Duration::from_millis(total_ms / write_history.len() as u64)
        };

        let average_read_latency = if read_history.is_empty() {
            Duration::from_millis(0)
        } else {
            let total_ms: u64 = read_history.iter().map(|d| d.as_millis() as u64).sum();
            Duration::from_millis(total_ms / read_history.len() as u64)
        };

        // Calculate 95th percentile latencies
        let mut write_latencies: Vec<u64> = write_history.iter()
            .map(|d| d.as_millis() as u64)
            .collect();
        write_latencies.sort();
        
        let p95_write_latency = if write_latencies.is_empty() {
            Duration::from_millis(0)
        } else {
            let index = ((write_latencies.len() as f64) * 0.95) as usize;
            let index = index.min(write_latencies.len() - 1);
            Duration::from_millis(write_latencies[index])
        };

        let mut read_latencies: Vec<u64> = read_history.iter()
            .map(|d| d.as_millis() as u64)
            .collect();
        read_latencies.sort();
        
        let p95_read_latency = if read_latencies.is_empty() {
            Duration::from_millis(0)
        } else {
            let index = ((read_latencies.len() as f64) * 0.95) as usize;
            let index = index.min(read_latencies.len() - 1);
            Duration::from_millis(read_latencies[index])
        };

        StorageMetricsSnapshot {
            total_writes: *self.total_writes.read().await,
            total_reads: *self.total_reads.read().await,
            total_vertices: *self.total_vertices.read().await,
            total_payloads: *self.total_payloads.read().await,
            total_blocks: *self.total_blocks.read().await,
            bytes_written: *self.bytes_written.read().await,
            bytes_read: *self.bytes_read.read().await,
            write_errors: *self.write_errors.read().await,
            read_errors: *self.read_errors.read().await,
            average_write_latency,
            average_read_latency,
            p95_write_latency,
            p95_read_latency,
            last_update: *self.last_update.read().await,
        }
    }

    /// Get Prometheus-format metrics
    pub async fn get_prometheus_metrics(&self) -> String {
        let metrics = self.get_current_metrics().await;
        
        format!(
            "# HELP qstorage_writes_total Total storage write operations\n\
             # TYPE qstorage_writes_total counter\n\
             qstorage_writes_total {}\n\
             \n\
             # HELP qstorage_reads_total Total storage read operations\n\
             # TYPE qstorage_reads_total counter\n\
             qstorage_reads_total {}\n\
             \n\
             # HELP qstorage_vertices_total Total vertices stored\n\
             # TYPE qstorage_vertices_total counter\n\
             qstorage_vertices_total {}\n\
             \n\
             # HELP qstorage_payloads_total Total payloads stored\n\
             # TYPE qstorage_payloads_total counter\n\
             qstorage_payloads_total {}\n\
             \n\
             # HELP qstorage_blocks_total Total blocks finalized\n\
             # TYPE qstorage_blocks_total counter\n\
             qstorage_blocks_total {}\n\
             \n\
             # HELP qstorage_write_latency_ms_avg Average write latency in milliseconds\n\
             # TYPE qstorage_write_latency_ms_avg gauge\n\
             qstorage_write_latency_ms_avg {}\n\
             \n\
             # HELP qstorage_write_latency_ms_p95 95th percentile write latency in milliseconds\n\
             # TYPE qstorage_write_latency_ms_p95 gauge\n\
             qstorage_write_latency_ms_p95 {}\n\
             \n\
             # HELP qstorage_read_latency_ms_avg Average read latency in milliseconds\n\
             # TYPE qstorage_read_latency_ms_avg gauge\n\
             qstorage_read_latency_ms_avg {}\n\
             \n\
             # HELP qstorage_bytes_written_total Total bytes written to storage\n\
             # TYPE qstorage_bytes_written_total counter\n\
             qstorage_bytes_written_total {}\n\
             \n\
             # HELP qstorage_bytes_read_total Total bytes read from storage\n\
             # TYPE qstorage_bytes_read_total counter\n\
             qstorage_bytes_read_total {}\n\
             \n\
             # HELP qstorage_write_errors_total Total write errors\n\
             # TYPE qstorage_write_errors_total counter\n\
             qstorage_write_errors_total {}\n\
             \n\
             # HELP qstorage_read_errors_total Total read errors\n\
             # TYPE qstorage_read_errors_total counter\n\
             qstorage_read_errors_total {}\n",
            metrics.total_writes,
            metrics.total_reads,
            metrics.total_vertices,
            metrics.total_payloads,
            metrics.total_blocks,
            metrics.average_write_latency.as_millis(),
            metrics.p95_write_latency.as_millis(),
            metrics.average_read_latency.as_millis(),
            metrics.bytes_written,
            metrics.bytes_read,
            metrics.write_errors,
            metrics.read_errors
        )
    }

    /// Reset all metrics
    pub async fn reset_metrics(&self) {
        let mut write_history = self.write_latency_history.write().await;
        let mut read_history = self.read_latency_history.write().await;
        let mut total_writes = self.total_writes.write().await;
        let mut total_reads = self.total_reads.write().await;
        let mut total_vertices = self.total_vertices.write().await;
        let mut total_payloads = self.total_payloads.write().await;
        let mut total_blocks = self.total_blocks.write().await;
        let mut bytes_written = self.bytes_written.write().await;
        let mut bytes_read = self.bytes_read.write().await;
        let mut write_errors = self.write_errors.write().await;
        let mut read_errors = self.read_errors.write().await;

        write_history.clear();
        read_history.clear();
        *total_writes = 0;
        *total_reads = 0;
        *total_vertices = 0;
        *total_payloads = 0;
        *total_blocks = 0;
        *bytes_written = 0;
        *bytes_read = 0;
        *write_errors = 0;
        *read_errors = 0;

        debug!("🔄 Storage metrics reset");
    }

    /// Check if performance is within acceptable bounds
    pub async fn check_performance_health(&self) -> StoragePerformanceHealth {
        let metrics = self.get_current_metrics().await;

        // Define performance thresholds
        let write_latency_ok = metrics.average_write_latency < Duration::from_millis(100);
        let read_latency_ok = metrics.average_read_latency < Duration::from_millis(50);
        let error_rate_ok = {
            let total_ops = metrics.total_writes + metrics.total_reads;
            let total_errors = metrics.write_errors + metrics.read_errors;
            
            if total_ops == 0 {
                true
            } else {
                (total_errors as f64 / total_ops as f64) < 0.01 // <1% error rate
            }
        };

        if write_latency_ok && read_latency_ok && error_rate_ok {
            StoragePerformanceHealth::Healthy
        } else if !write_latency_ok && metrics.average_write_latency > Duration::from_millis(500) {
            StoragePerformanceHealth::SlowWrites
        } else if !read_latency_ok && metrics.average_read_latency > Duration::from_millis(200) {
            StoragePerformanceHealth::SlowReads
        } else if !error_rate_ok {
            StoragePerformanceHealth::HighErrorRate
        } else {
            StoragePerformanceHealth::Degraded
        }
    }

    /// Get write throughput (operations per second)
    pub async fn get_write_throughput(&self) -> f64 {
        let metrics = self.get_current_metrics().await;
        let duration_seconds = metrics.last_update.elapsed().as_secs_f64();
        
        if duration_seconds > 0.0 {
            metrics.total_writes as f64 / duration_seconds
        } else {
            0.0
        }
    }

    /// Get read throughput (operations per second)
    pub async fn get_read_throughput(&self) -> f64 {
        let metrics = self.get_current_metrics().await;
        let duration_seconds = metrics.last_update.elapsed().as_secs_f64();
        
        if duration_seconds > 0.0 {
            metrics.total_reads as f64 / duration_seconds
        } else {
            0.0
        }
    }

    /// Get bandwidth utilization
    pub async fn get_bandwidth_stats(&self) -> BandwidthStats {
        let metrics = self.get_current_metrics().await;
        let duration_seconds = metrics.last_update.elapsed().as_secs_f64();

        let write_bandwidth = if duration_seconds > 0.0 {
            metrics.bytes_written as f64 / duration_seconds / 1_000_000.0 // MB/s
        } else {
            0.0
        };

        let read_bandwidth = if duration_seconds > 0.0 {
            metrics.bytes_read as f64 / duration_seconds / 1_000_000.0 // MB/s
        } else {
            0.0
        };

        BandwidthStats {
            write_mbps: write_bandwidth,
            read_mbps: read_bandwidth,
            total_bytes_written: metrics.bytes_written,
            total_bytes_read: metrics.bytes_read,
        }
    }
}

/// Snapshot of storage metrics at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageMetricsSnapshot {
    pub total_writes: u64,
    pub total_reads: u64,
    pub total_vertices: u64,
    pub total_payloads: u64,
    pub total_blocks: u64,
    pub bytes_written: u64,
    pub bytes_read: u64,
    pub write_errors: u64,
    pub read_errors: u64,
    pub average_write_latency: Duration,
    pub average_read_latency: Duration,
    pub p95_write_latency: Duration,
    pub p95_read_latency: Duration,
    pub last_update: Instant,
}

/// Storage performance health status
#[derive(Debug, Clone, PartialEq)]
pub enum StoragePerformanceHealth {
    Healthy,
    Degraded,
    SlowWrites,
    SlowReads,
    HighErrorRate,
    Offline,
}

impl StoragePerformanceHealth {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Healthy => "healthy",
            Self::Degraded => "degraded",
            Self::SlowWrites => "slow_writes",
            Self::SlowReads => "slow_reads",
            Self::HighErrorRate => "high_error_rate",
            Self::Offline => "offline",
        }
    }

    pub fn requires_attention(&self) -> bool {
        !matches!(self, Self::Healthy)
    }
}

/// Bandwidth utilization statistics
#[derive(Debug, Clone, Serialize)]
pub struct BandwidthStats {
    pub write_mbps: f64,
    pub read_mbps: f64,
    pub total_bytes_written: u64,
    pub total_bytes_read: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_metrics_tracking() {
        let metrics = StorageMetrics::new();

        // Record some operations
        metrics.record_vertex_write(Duration::from_millis(50), 1024, 4096).await;
        metrics.record_vertex_write(Duration::from_millis(75), 2048, 8192).await;
        metrics.record_read(Duration::from_millis(25), 1024).await;

        let snapshot = metrics.get_current_metrics().await;

        assert_eq!(snapshot.total_writes, 2);
        assert_eq!(snapshot.total_reads, 1);
        assert_eq!(snapshot.total_vertices, 2);
        assert_eq!(snapshot.total_payloads, 2);
        assert_eq!(snapshot.bytes_written, 1024 + 4096 + 2048 + 8192);
        assert_eq!(snapshot.bytes_read, 1024);
        assert_eq!(snapshot.average_write_latency, Duration::from_millis(62)); // (50+75)/2
        assert_eq!(snapshot.average_read_latency, Duration::from_millis(25));
    }

    #[tokio::test]
    async fn test_error_tracking() {
        let metrics = StorageMetrics::new();

        // Record some operations with errors
        metrics.record_vertex_write(Duration::from_millis(50), 1024, 4096).await;
        metrics.record_write_error().await;
        metrics.record_read_error().await;

        let snapshot = metrics.get_current_metrics().await;

        assert_eq!(snapshot.total_writes, 1);
        assert_eq!(snapshot.write_errors, 1);
        assert_eq!(snapshot.read_errors, 1);
    }

    #[tokio::test]
    async fn test_performance_health() {
        let metrics = StorageMetrics::new();

        // Record healthy operations
        for _ in 0..10 {
            metrics.record_vertex_write(Duration::from_millis(20), 1024, 2048).await;
            metrics.record_read(Duration::from_millis(5), 1024).await;
        }

        let health = metrics.check_performance_health().await;
        assert_eq!(health, StoragePerformanceHealth::Healthy);

        // Record slow writes
        for _ in 0..20 {
            metrics.record_vertex_write(Duration::from_millis(600), 1024, 2048).await;
        }

        let health = metrics.check_performance_health().await;
        assert_eq!(health, StoragePerformanceHealth::SlowWrites);
    }

    #[tokio::test]
    async fn test_throughput_calculation() {
        let metrics = StorageMetrics::new();

        // Record operations
        for _ in 0..100 {
            metrics.record_vertex_write(Duration::from_millis(10), 1024, 2048).await;
        }

        // Wait a bit to establish time base
        tokio::time::sleep(Duration::from_millis(100)).await;

        let write_throughput = metrics.get_write_throughput().await;
        assert!(write_throughput > 0.0);
    }

    #[tokio::test]
    async fn test_bandwidth_stats() {
        let metrics = StorageMetrics::new();

        // Record bandwidth usage
        metrics.record_vertex_write(Duration::from_millis(10), 1024, 4096).await; // 5KB total
        metrics.record_read(Duration::from_millis(5), 2048).await; // 2KB read

        let bandwidth = metrics.get_bandwidth_stats().await;
        
        assert_eq!(bandwidth.total_bytes_written, 5120); // 1024 + 4096
        assert_eq!(bandwidth.total_bytes_read, 2048);
        assert!(bandwidth.write_mbps >= 0.0);
        assert!(bandwidth.read_mbps >= 0.0);
    }
}