//! Cache Metrics Collection and Monitoring
//!
//! Real-time performance monitoring and metrics collection for the
//! Q-NarwhalKnight hierarchical caching system.

use anyhow::Result;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::RwLock;
use tracing::{debug, info};

/// Cache metrics collector
#[derive(Debug)]
pub struct CacheMetricsCollector {
    access_history: Arc<RwLock<VecDeque<AccessMetric>>>,
    performance_windows: Arc<RwLock<HashMap<String, PerformanceWindow>>>,
    current_metrics: Arc<RwLock<RealTimeMetrics>>,
    collection_start: Instant,
}

/// Individual access metric
#[derive(Debug, Clone)]
pub struct AccessMetric {
    pub timestamp: SystemTime,
    pub access_time: Duration,
    pub cache_level: Option<crate::CacheLevel>,
    pub was_hit: bool,
    pub key_hash: u64, // For pattern analysis
    pub operation_type: OperationType,
}

/// Type of cache operation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OperationType {
    Read,
    Write,
    Eviction,
    Invalidation,
    Prefetch,
}

/// Performance measurement window
#[derive(Debug, Clone)]
pub struct PerformanceWindow {
    pub window_size: Duration,
    pub metrics: VecDeque<f64>,
    pub last_updated: Instant,
    pub min_value: f64,
    pub max_value: f64,
    pub avg_value: f64,
}

/// Real-time aggregated metrics
#[derive(Debug, Clone, Default)]
pub struct RealTimeMetrics {
    // Hit ratios by cache level
    pub l1_hit_ratio: f64,
    pub l2_hit_ratio: f64,
    pub l3_hit_ratio: f64,
    pub overall_hit_ratio: f64,

    // Performance metrics
    pub avg_access_time_ns: f64,
    pub p50_access_time_ns: f64,
    pub p95_access_time_ns: f64,
    pub p99_access_time_ns: f64,

    // Throughput metrics
    pub requests_per_second: f64,
    pub cache_effectiveness: f64,

    // Memory metrics
    pub memory_usage_mb: f64,
    pub memory_efficiency: f64,

    // System health indicators
    pub fragmentation_ratio: f64,
    pub gc_pressure: f64,

    // Advanced metrics
    pub prefetch_accuracy: f64,
    pub numa_locality: f64,
}

/// Performance summary for reporting
#[derive(Debug, Clone)]
pub struct PerformanceSummary {
    pub measurement_duration: Duration,
    pub total_requests: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub hit_ratio: f64,
    pub avg_latency_ms: f64,
    pub peak_throughput_rps: f64,
    pub memory_peak_mb: f64,
    pub performance_score: f64,
}

impl CacheMetricsCollector {
    /// Create new metrics collector
    pub fn new() -> Self {
        info!("Initializing cache metrics collector");

        Self {
            access_history: Arc::new(RwLock::new(VecDeque::with_capacity(10000))),
            performance_windows: Arc::new(RwLock::new(HashMap::new())),
            current_metrics: Arc::new(RwLock::new(RealTimeMetrics::default())),
            collection_start: Instant::now(),
        }
    }

    /// Record cache access with performance data
    pub async fn record_access(&self, access_time: Duration, was_hit: bool) {
        let metric = AccessMetric {
            timestamp: SystemTime::now(),
            access_time,
            cache_level: None, // Will be filled by caller if known
            was_hit,
            key_hash: 0, // Simplified for this example
            operation_type: OperationType::Read,
        };

        // Add to history
        {
            let mut history = self.access_history.write().await;

            // Keep history bounded
            if history.len() >= 10000 {
                history.pop_front();
            }

            history.push_back(metric);
        }

        // Update real-time metrics
        self.update_realtime_metrics().await;
    }

    /// Record cache operation with detailed information
    pub async fn record_operation(
        &self,
        operation: OperationType,
        cache_level: Option<crate::CacheLevel>,
        access_time: Duration,
        key_hash: u64,
        was_successful: bool,
    ) {
        let metric = AccessMetric {
            timestamp: SystemTime::now(),
            access_time,
            cache_level,
            was_hit: was_successful,
            key_hash,
            operation_type: operation,
        };

        {
            let mut history = self.access_history.write().await;
            if history.len() >= 10000 {
                history.pop_front();
            }
            history.push_back(metric);
        }

        // Update performance windows
        self.update_performance_window("access_time", access_time.as_nanos() as f64)
            .await;

        if was_successful {
            self.update_performance_window("hit_ratio", 1.0).await;
        } else {
            self.update_performance_window("hit_ratio", 0.0).await;
        }

        self.update_realtime_metrics().await;
    }

    /// Update performance window with new value
    async fn update_performance_window(&self, metric_name: &str, value: f64) {
        let mut windows = self.performance_windows.write().await;

        let window = windows.entry(metric_name.to_string()).or_insert_with(|| {
            PerformanceWindow {
                window_size: Duration::from_secs(60), // 1-minute windows
                metrics: VecDeque::with_capacity(60),
                last_updated: Instant::now(),
                min_value: f64::INFINITY,
                max_value: f64::NEG_INFINITY,
                avg_value: 0.0,
            }
        });

        // Add new value
        if window.metrics.len() >= 60 {
            window.metrics.pop_front();
        }
        window.metrics.push_back(value);

        // Update aggregates
        window.min_value = window.metrics.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        window.max_value = window
            .metrics
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        window.avg_value = window.metrics.iter().sum::<f64>() / window.metrics.len() as f64;
        window.last_updated = Instant::now();
    }

    /// Update real-time aggregated metrics
    async fn update_realtime_metrics(&self) {
        let history = self.access_history.read().await;

        if history.is_empty() {
            return;
        }

        // Calculate metrics from recent history
        let recent_window = Duration::from_secs(10);
        let cutoff_time = SystemTime::now() - recent_window;

        let recent_metrics: Vec<&AccessMetric> = history
            .iter()
            .filter(|m| m.timestamp > cutoff_time)
            .collect();

        if recent_metrics.is_empty() {
            return;
        }

        // Calculate hit ratios
        let total_requests = recent_metrics.len() as f64;
        let total_hits = recent_metrics.iter().filter(|m| m.was_hit).count() as f64;
        let overall_hit_ratio = total_hits / total_requests;

        // Level-specific hit ratios
        let l1_metrics: Vec<_> = recent_metrics
            .iter()
            .filter(|m| m.cache_level == Some(crate::CacheLevel::L1))
            .collect();
        let l2_metrics: Vec<_> = recent_metrics
            .iter()
            .filter(|m| m.cache_level == Some(crate::CacheLevel::L2))
            .collect();
        let l3_metrics: Vec<_> = recent_metrics
            .iter()
            .filter(|m| m.cache_level == Some(crate::CacheLevel::L3))
            .collect();

        let l1_hit_ratio = if !l1_metrics.is_empty() {
            l1_metrics.iter().filter(|m| m.was_hit).count() as f64 / l1_metrics.len() as f64
        } else {
            0.0
        };

        let l2_hit_ratio = if !l2_metrics.is_empty() {
            l2_metrics.iter().filter(|m| m.was_hit).count() as f64 / l2_metrics.len() as f64
        } else {
            0.0
        };

        let l3_hit_ratio = if !l3_metrics.is_empty() {
            l3_metrics.iter().filter(|m| m.was_hit).count() as f64 / l3_metrics.len() as f64
        } else {
            0.0
        };

        // Calculate latency percentiles
        let mut access_times: Vec<f64> = recent_metrics
            .iter()
            .map(|m| m.access_time.as_nanos() as f64)
            .collect();
        access_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let avg_access_time = access_times.iter().sum::<f64>() / access_times.len() as f64;
        let p50_access_time = Self::calculate_percentile(&access_times, 0.5);
        let p95_access_time = Self::calculate_percentile(&access_times, 0.95);
        let p99_access_time = Self::calculate_percentile(&access_times, 0.99);

        // Calculate throughput
        let requests_per_second = total_requests / recent_window.as_secs_f64();

        // Cache effectiveness (hit ratio weighted by access time reduction)
        let cache_effectiveness =
            overall_hit_ratio * 0.8 + (1.0 - (avg_access_time / 1_000_000.0).min(1.0)) * 0.2;

        // Update current metrics
        {
            let mut metrics = self.current_metrics.write().await;
            metrics.l1_hit_ratio = l1_hit_ratio;
            metrics.l2_hit_ratio = l2_hit_ratio;
            metrics.l3_hit_ratio = l3_hit_ratio;
            metrics.overall_hit_ratio = overall_hit_ratio;
            metrics.avg_access_time_ns = avg_access_time;
            metrics.p50_access_time_ns = p50_access_time;
            metrics.p95_access_time_ns = p95_access_time;
            metrics.p99_access_time_ns = p99_access_time;
            metrics.requests_per_second = requests_per_second;
            metrics.cache_effectiveness = cache_effectiveness;
        }

        debug!(
            "Updated real-time metrics: {:.1}% hit ratio, {:.1} RPS",
            overall_hit_ratio * 100.0,
            requests_per_second
        );
    }

    /// Calculate percentile from sorted array
    fn calculate_percentile(sorted_values: &[f64], percentile: f64) -> f64 {
        if sorted_values.is_empty() {
            return 0.0;
        }

        let index = ((sorted_values.len() - 1) as f64 * percentile).round() as usize;
        sorted_values[index.min(sorted_values.len() - 1)]
    }

    /// Get current real-time metrics
    pub async fn get_current_metrics(&self) -> RealTimeMetrics {
        self.current_metrics.read().await.clone()
    }

    /// Generate performance summary
    pub async fn generate_performance_summary(&self) -> Result<PerformanceSummary> {
        let history = self.access_history.read().await;
        let measurement_duration = self.collection_start.elapsed();

        if history.is_empty() {
            return Ok(PerformanceSummary {
                measurement_duration,
                total_requests: 0,
                cache_hits: 0,
                cache_misses: 0,
                hit_ratio: 0.0,
                avg_latency_ms: 0.0,
                peak_throughput_rps: 0.0,
                memory_peak_mb: 0.0,
                performance_score: 0.0,
            });
        }

        let total_requests = history.len() as u64;
        let cache_hits = history.iter().filter(|m| m.was_hit).count() as u64;
        let cache_misses = total_requests - cache_hits;
        let hit_ratio = cache_hits as f64 / total_requests as f64;

        // Calculate average latency
        let total_latency_ms: f64 = history
            .iter()
            .map(|m| m.access_time.as_millis() as f64)
            .sum();
        let avg_latency_ms = total_latency_ms / total_requests as f64;

        // Calculate peak throughput (max requests in any 1-second window)
        let peak_throughput_rps = self.calculate_peak_throughput(&history).await;

        // Estimate memory usage (simplified)
        let memory_peak_mb = (total_requests as f64 * 0.001).min(2048.0); // Max 2GB

        // Calculate performance score (0-100)
        let latency_score = (1.0 - (avg_latency_ms / 100.0).min(1.0)) * 100.0;
        let hit_ratio_score = hit_ratio * 100.0;
        let throughput_score = (peak_throughput_rps / 10000.0).min(1.0) * 100.0;

        let performance_score =
            (latency_score * 0.4 + hit_ratio_score * 0.4 + throughput_score * 0.2);

        info!("Performance summary: {:.1}% hit ratio, {:.2}ms avg latency, {:.0} peak RPS, score: {:.1}",
              hit_ratio * 100.0, avg_latency_ms, peak_throughput_rps, performance_score);

        Ok(PerformanceSummary {
            measurement_duration,
            total_requests,
            cache_hits,
            cache_misses,
            hit_ratio,
            avg_latency_ms,
            peak_throughput_rps,
            memory_peak_mb,
            performance_score,
        })
    }

    /// Calculate peak throughput in any 1-second window
    async fn calculate_peak_throughput(&self, history: &VecDeque<AccessMetric>) -> f64 {
        let mut max_throughput: f64 = 0.0;
        let window_size = Duration::from_secs(1);

        // Sliding window to find peak
        for i in 0..history.len() {
            let window_start = history[i].timestamp;
            let window_end = window_start + window_size;

            let requests_in_window = history
                .iter()
                .skip(i)
                .take_while(|m| m.timestamp <= window_end)
                .count();

            max_throughput = max_throughput.max(requests_in_window as f64);
        }

        max_throughput
    }

    /// Export metrics to JSON format
    pub async fn export_metrics_json(&self) -> Result<String> {
        let summary = self.generate_performance_summary().await?;
        let current_metrics = self.get_current_metrics().await;

        let export = serde_json::json!({
            "timestamp": SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            "summary": {
                "measurement_duration_secs": summary.measurement_duration.as_secs(),
                "total_requests": summary.total_requests,
                "cache_hits": summary.cache_hits,
                "cache_misses": summary.cache_misses,
                "hit_ratio": summary.hit_ratio,
                "avg_latency_ms": summary.avg_latency_ms,
                "peak_throughput_rps": summary.peak_throughput_rps,
                "memory_peak_mb": summary.memory_peak_mb,
                "performance_score": summary.performance_score,
            },
            "realtime_metrics": {
                "l1_hit_ratio": current_metrics.l1_hit_ratio,
                "l2_hit_ratio": current_metrics.l2_hit_ratio,
                "l3_hit_ratio": current_metrics.l3_hit_ratio,
                "overall_hit_ratio": current_metrics.overall_hit_ratio,
                "avg_access_time_ns": current_metrics.avg_access_time_ns,
                "p95_access_time_ns": current_metrics.p95_access_time_ns,
                "p99_access_time_ns": current_metrics.p99_access_time_ns,
                "requests_per_second": current_metrics.requests_per_second,
                "cache_effectiveness": current_metrics.cache_effectiveness,
            }
        });

        Ok(export.to_string())
    }

    /// Reset metrics collection
    pub async fn reset_metrics(&self) {
        {
            let mut history = self.access_history.write().await;
            history.clear();
        }

        {
            let mut windows = self.performance_windows.write().await;
            windows.clear();
        }

        {
            let mut metrics = self.current_metrics.write().await;
            *metrics = RealTimeMetrics::default();
        }

        info!("Cache metrics reset");
    }

    /// Get performance trend analysis
    pub async fn analyze_performance_trends(&self) -> Result<TrendAnalysis> {
        let windows = self.performance_windows.read().await;

        let mut trends = HashMap::new();

        for (metric_name, window) in windows.iter() {
            if window.metrics.len() >= 10 {
                let trend = self.calculate_trend(&window.metrics);
                trends.insert(metric_name.clone(), trend);
            }
        }

        Ok(TrendAnalysis {
            trends,
            analysis_timestamp: SystemTime::now(),
        })
    }

    /// Calculate trend direction and magnitude
    fn calculate_trend(&self, values: &VecDeque<f64>) -> TrendDirection {
        if values.len() < 2 {
            return TrendDirection::Stable;
        }

        let first_half: f64 = values.iter().take(values.len() / 2).sum();
        let second_half: f64 = values.iter().skip(values.len() / 2).sum();

        let first_avg = first_half / (values.len() / 2) as f64;
        let second_avg = second_half / (values.len() / 2) as f64;

        let change_ratio = (second_avg - first_avg) / first_avg.max(0.001);

        if change_ratio > 0.1 {
            TrendDirection::Increasing
        } else if change_ratio < -0.1 {
            TrendDirection::Decreasing
        } else {
            TrendDirection::Stable
        }
    }
}

/// Trend analysis results
#[derive(Debug, Clone)]
pub struct TrendAnalysis {
    pub trends: HashMap<String, TrendDirection>,
    pub analysis_timestamp: SystemTime,
}

/// Direction of performance trend
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_metrics_collector() {
        let collector = CacheMetricsCollector::new();

        // Record some access metrics
        collector
            .record_access(Duration::from_nanos(1000), true)
            .await;
        collector
            .record_access(Duration::from_nanos(2000), false)
            .await;

        let metrics = collector.get_current_metrics().await;
        assert!(metrics.overall_hit_ratio >= 0.0);
    }

    #[tokio::test]
    async fn test_performance_summary() {
        let collector = CacheMetricsCollector::new();

        // Record several metrics
        for i in 0..100 {
            let was_hit = i % 3 != 0; // 67% hit ratio
            collector
                .record_access(Duration::from_nanos(1000 + i * 100), was_hit)
                .await;
        }

        let summary = collector.generate_performance_summary().await.unwrap();
        assert_eq!(summary.total_requests, 100);
        assert!(summary.hit_ratio > 0.6 && summary.hit_ratio < 0.7);
    }
}
