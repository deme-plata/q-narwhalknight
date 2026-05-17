// Q-NarwhalKnight Sharding Metrics Collection
// Performance monitoring and analytics for sharded operations

use crate::{CrossShardResult, ShardId, ShardMetrics};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;

#[cfg(feature = "prometheus")]
use prometheus::{Counter, Gauge, Histogram, HistogramOpts, Opts, Registry};

/// Metrics collector for sharded operations
#[derive(Debug)]
pub struct ShardMetricsCollector {
    shard_metrics: Arc<RwLock<HashMap<ShardId, ShardMetrics>>>,
    historical_data: Arc<RwLock<VecDeque<MetricsSnapshot>>>,
    performance_aggregates: Arc<RwLock<PerformanceAggregates>>,
    cross_shard_stats: Arc<RwLock<CrossShardStats>>,
    collection_start: Instant,
    #[cfg(feature = "prometheus")]
    prometheus_metrics: PrometheusMetrics,
}

/// Historical snapshot of system metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MetricsSnapshot {
    timestamp: SystemTime,
    total_tps: f64,
    average_latency: f64,
    total_shards: usize,
    healthy_shards: usize,
    cross_shard_operations: u64,
    memory_usage_total: f64,
    cpu_utilization_avg: f64,
}

/// Performance aggregates across all shards
#[derive(Debug, Clone, Default)]
struct PerformanceAggregates {
    total_transactions_processed: u64,
    peak_tps: f64,
    average_tps: f64,
    min_latency_ms: f64,
    max_latency_ms: f64,
    average_latency_ms: f64,
    total_memory_usage_mb: f64,
    average_cpu_utilization: f64,
    uptime_seconds: f64,
}

/// Cross-shard operation statistics
#[derive(Debug, Clone, Default)]
struct CrossShardStats {
    operations_initiated: u64,
    operations_completed: u64,
    operations_failed: u64,
    average_operation_time_ms: f64,
    data_transferred_bytes: u64,
    cache_hit_ratio: f64,
}

/// Prometheus metrics integration
#[cfg(feature = "prometheus")]
#[derive(Debug)]
struct PrometheusMetrics {
    registry: Registry,
    shard_tps: Gauge,
    shard_latency: Histogram,
    cross_shard_operations: Counter,
    memory_usage: Gauge,
    cpu_utilization: Gauge,
    queue_depth: Gauge,
}

/// Detailed performance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    pub report_generated_at: SystemTime,
    pub collection_duration: Duration,
    pub system_overview: SystemOverview,
    pub shard_breakdown: Vec<ShardPerformance>,
    pub cross_shard_analysis: CrossShardAnalysis,
    pub performance_trends: Vec<TrendData>,
    pub recommendations: Vec<PerformanceRecommendation>,
}

/// System-level performance overview
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemOverview {
    pub total_tps: f64,
    pub peak_tps: f64,
    pub average_latency_ms: f64,
    pub total_shards: usize,
    pub healthy_shards: usize,
    pub total_memory_usage_mb: f64,
    pub average_cpu_utilization: f64,
    pub uptime_hours: f64,
}

/// Individual shard performance data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardPerformance {
    pub shard_id: ShardId,
    pub tps: f64,
    pub latency_ms: f64,
    pub cpu_utilization: f64,
    pub memory_usage_mb: f64,
    pub queue_depth: usize,
    pub efficiency_score: f64,
    pub health_status: HealthStatus,
}

/// Cross-shard operation analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossShardAnalysis {
    pub operations_per_second: f64,
    pub success_rate: f64,
    pub average_operation_time_ms: f64,
    pub data_transfer_rate_mbps: f64,
    pub bottleneck_shards: Vec<ShardId>,
}

/// Performance trend data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendData {
    pub metric_name: String,
    pub trend_direction: TrendDirection,
    pub change_percentage: f64,
    pub time_window: Duration,
}

/// Performance recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRecommendation {
    pub recommendation_type: RecommendationType,
    pub priority: Priority,
    pub description: String,
    pub expected_improvement: String,
    pub affected_shards: Vec<ShardId>,
}

/// Health status of a shard
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum HealthStatus {
    Excellent,
    Good,
    Fair,
    Poor,
    Critical,
}

/// Trend direction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Stable,
    Degrading,
}

/// Recommendation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationType {
    ScaleUp,
    ScaleDown,
    Rebalance,
    Optimize,
    Maintenance,
}

/// Recommendation priority
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Priority {
    Critical,
    High,
    Medium,
    Low,
}

impl ShardMetricsCollector {
    /// Create new metrics collector
    pub fn new() -> Self {
        tracing::info!("Creating shard metrics collector");

        #[cfg(feature = "prometheus")]
        let prometheus_metrics = Self::create_prometheus_metrics();

        Self {
            shard_metrics: Arc::new(RwLock::new(HashMap::new())),
            historical_data: Arc::new(RwLock::new(VecDeque::new())),
            performance_aggregates: Arc::new(RwLock::new(PerformanceAggregates::default())),
            cross_shard_stats: Arc::new(RwLock::new(CrossShardStats::default())),
            collection_start: Instant::now(),
            #[cfg(feature = "prometheus")]
            prometheus_metrics,
        }
    }

    #[cfg(feature = "prometheus")]
    fn create_prometheus_metrics() -> PrometheusMetrics {
        let registry = Registry::new();

        let shard_tps = Gauge::with_opts(Opts::new(
            "qnk_shard_tps",
            "Transactions per second per shard",
        ))
        .unwrap();

        let shard_latency = Histogram::with_opts(HistogramOpts::new(
            "qnk_shard_latency_ms",
            "Shard processing latency in milliseconds",
        ))
        .unwrap();

        let cross_shard_operations = Counter::with_opts(Opts::new(
            "qnk_cross_shard_operations_total",
            "Total cross-shard operations",
        ))
        .unwrap();

        let memory_usage = Gauge::with_opts(Opts::new(
            "qnk_shard_memory_mb",
            "Memory usage per shard in MB",
        ))
        .unwrap();

        let cpu_utilization = Gauge::with_opts(Opts::new(
            "qnk_shard_cpu_utilization",
            "CPU utilization per shard (0-1)",
        ))
        .unwrap();

        let queue_depth =
            Gauge::with_opts(Opts::new("qnk_shard_queue_depth", "Queue depth per shard")).unwrap();

        registry.register(Box::new(shard_tps.clone())).unwrap();
        registry.register(Box::new(shard_latency.clone())).unwrap();
        registry
            .register(Box::new(cross_shard_operations.clone()))
            .unwrap();
        registry.register(Box::new(memory_usage.clone())).unwrap();
        registry
            .register(Box::new(cpu_utilization.clone()))
            .unwrap();
        registry.register(Box::new(queue_depth.clone())).unwrap();

        PrometheusMetrics {
            registry,
            shard_tps,
            shard_latency,
            cross_shard_operations,
            memory_usage,
            cpu_utilization,
            queue_depth,
        }
    }

    /// Update metrics for a specific shard
    pub async fn update_shard_metrics(
        &self,
        shard_id: ShardId,
        metrics: ShardMetrics,
    ) -> Result<()> {
        {
            let mut shard_metrics = self.shard_metrics.write().await;
            shard_metrics.insert(shard_id, metrics.clone());
        }

        // Update Prometheus metrics if enabled
        #[cfg(feature = "prometheus")]
        {
            let shard_label = format!("{:?}", shard_id);
            self.prometheus_metrics
                .shard_tps
                .set(metrics.transactions_per_second);
            self.prometheus_metrics
                .shard_latency
                .observe(metrics.average_latency_ms);
            self.prometheus_metrics
                .memory_usage
                .set(metrics.memory_usage_mb);
            self.prometheus_metrics
                .cpu_utilization
                .set(metrics.cpu_utilization);
            self.prometheus_metrics
                .queue_depth
                .set(metrics.queue_depth as f64);
        }

        // Update performance aggregates
        self.update_performance_aggregates(&metrics).await;

        tracing::debug!(
            "Updated metrics for shard {:?}: TPS={:.1}, Latency={:.1}ms",
            shard_id,
            metrics.transactions_per_second,
            metrics.average_latency_ms
        );

        Ok(())
    }

    /// Update performance aggregates with new metrics
    async fn update_performance_aggregates(&self, metrics: &ShardMetrics) {
        let mut aggregates = self.performance_aggregates.write().await;

        aggregates.total_transactions_processed += metrics.transactions_per_second as u64;
        aggregates.peak_tps = aggregates.peak_tps.max(metrics.transactions_per_second);
        aggregates.total_memory_usage_mb += metrics.memory_usage_mb;

        // Update latency bounds
        if aggregates.min_latency_ms == 0.0
            || metrics.average_latency_ms < aggregates.min_latency_ms
        {
            aggregates.min_latency_ms = metrics.average_latency_ms;
        }
        aggregates.max_latency_ms = aggregates.max_latency_ms.max(metrics.average_latency_ms);

        // Update uptime
        aggregates.uptime_seconds = self.collection_start.elapsed().as_secs_f64();
    }

    /// Record cross-shard operation result
    pub async fn record_cross_shard_operation(&self, result: CrossShardResult) -> Result<()> {
        let mut stats = self.cross_shard_stats.write().await;

        stats.operations_initiated += 1;

        if result.success {
            stats.operations_completed += 1;
        } else {
            stats.operations_failed += 1;
        }

        // Update average operation time
        let total_ops = stats.operations_completed + stats.operations_failed;
        if total_ops > 0 {
            stats.average_operation_time_ms = (stats.average_operation_time_ms
                * (total_ops - 1) as f64
                + result.execution_time_ms as f64)
                / total_ops as f64;
        }

        stats.data_transferred_bytes += result.data_transferred_bytes;

        #[cfg(feature = "prometheus")]
        self.prometheus_metrics.cross_shard_operations.inc();

        tracing::debug!(
            "Recorded cross-shard operation: success={}, time={}ms, data={}B",
            result.success,
            result.execution_time_ms,
            result.data_transferred_bytes
        );

        Ok(())
    }

    /// Collect all current shard metrics
    pub async fn collect_all_shard_metrics(&self) -> Result<Vec<ShardMetrics>> {
        let metrics = self.shard_metrics.read().await;
        Ok(metrics.values().cloned().collect())
    }

    /// Take a snapshot of current system performance
    pub async fn take_snapshot(&self) -> Result<()> {
        let metrics = self.shard_metrics.read().await;
        let cross_shard_stats = self.cross_shard_stats.read().await;

        let total_tps = metrics.values().map(|m| m.transactions_per_second).sum();
        let average_latency = if !metrics.is_empty() {
            metrics.values().map(|m| m.average_latency_ms).sum::<f64>() / metrics.len() as f64
        } else {
            0.0
        };

        let healthy_shards = metrics.values().filter(|m| m.cpu_utilization < 0.8).count();
        let memory_total = metrics.values().map(|m| m.memory_usage_mb).sum();
        let cpu_avg = if !metrics.is_empty() {
            metrics.values().map(|m| m.cpu_utilization).sum::<f64>() / metrics.len() as f64
        } else {
            0.0
        };

        let snapshot = MetricsSnapshot {
            timestamp: SystemTime::now(),
            total_tps,
            average_latency,
            total_shards: metrics.len(),
            healthy_shards,
            cross_shard_operations: cross_shard_stats.operations_completed,
            memory_usage_total: memory_total,
            cpu_utilization_avg: cpu_avg,
        };

        let mut history = self.historical_data.write().await;
        history.push_back(snapshot);

        // Keep only recent history (last 1000 snapshots)
        while history.len() > 1000 {
            history.pop_front();
        }

        tracing::debug!(
            "Took performance snapshot: TPS={:.1}, Latency={:.1}ms, Shards={}/{}",
            total_tps,
            average_latency,
            healthy_shards,
            metrics.len()
        );

        Ok(())
    }

    /// Generate comprehensive performance report
    pub async fn generate_performance_report(&self) -> Result<PerformanceReport> {
        let metrics = self.shard_metrics.read().await;
        let history = self.historical_data.read().await;
        let aggregates = self.performance_aggregates.read().await;
        let cross_shard_stats = self.cross_shard_stats.read().await;

        // System overview
        let total_tps = metrics.values().map(|m| m.transactions_per_second).sum();
        let average_latency = if !metrics.is_empty() {
            metrics.values().map(|m| m.average_latency_ms).sum::<f64>() / metrics.len() as f64
        } else {
            0.0
        };

        let healthy_shards = metrics
            .values()
            .filter(|m| self.assess_health_status(m) != HealthStatus::Critical)
            .count();
        let total_memory = metrics.values().map(|m| m.memory_usage_mb).sum();
        let avg_cpu = if !metrics.is_empty() {
            metrics.values().map(|m| m.cpu_utilization).sum::<f64>() / metrics.len() as f64
        } else {
            0.0
        };

        let system_overview = SystemOverview {
            total_tps,
            peak_tps: aggregates.peak_tps,
            average_latency_ms: average_latency,
            total_shards: metrics.len(),
            healthy_shards,
            total_memory_usage_mb: total_memory,
            average_cpu_utilization: avg_cpu,
            uptime_hours: aggregates.uptime_seconds / 3600.0,
        };

        // Shard breakdown
        let shard_breakdown: Vec<ShardPerformance> = metrics
            .iter()
            .map(|(shard_id, metrics)| {
                let efficiency_score = self.calculate_efficiency_score(metrics);
                let health_status = self.assess_health_status(metrics);

                ShardPerformance {
                    shard_id: *shard_id,
                    tps: metrics.transactions_per_second,
                    latency_ms: metrics.average_latency_ms,
                    cpu_utilization: metrics.cpu_utilization,
                    memory_usage_mb: metrics.memory_usage_mb,
                    queue_depth: metrics.queue_depth,
                    efficiency_score,
                    health_status,
                }
            })
            .collect();

        // Cross-shard analysis
        let success_rate = if cross_shard_stats.operations_initiated > 0 {
            cross_shard_stats.operations_completed as f64
                / cross_shard_stats.operations_initiated as f64
        } else {
            1.0
        };

        let data_transfer_rate = if aggregates.uptime_seconds > 0.0 {
            (cross_shard_stats.data_transferred_bytes as f64 / (1024.0 * 1024.0))
                / aggregates.uptime_seconds
        } else {
            0.0
        };

        let bottleneck_shards = self.identify_bottleneck_shards(&metrics);

        let cross_shard_analysis = CrossShardAnalysis {
            operations_per_second: cross_shard_stats.operations_completed as f64
                / aggregates.uptime_seconds.max(1.0),
            success_rate,
            average_operation_time_ms: cross_shard_stats.average_operation_time_ms,
            data_transfer_rate_mbps: data_transfer_rate,
            bottleneck_shards,
        };

        // Performance trends
        let performance_trends = self.analyze_trends(&history);

        // Generate recommendations
        let recommendations =
            self.generate_recommendations(&shard_breakdown, &cross_shard_analysis);

        Ok(PerformanceReport {
            report_generated_at: SystemTime::now(),
            collection_duration: self.collection_start.elapsed(),
            system_overview,
            shard_breakdown,
            cross_shard_analysis,
            performance_trends,
            recommendations,
        })
    }

    /// Calculate efficiency score for a shard
    fn calculate_efficiency_score(&self, metrics: &ShardMetrics) -> f64 {
        // Efficiency = TPS / (CPU * Memory_Factor * Latency_Factor)
        let cpu_factor = metrics.cpu_utilization.max(0.01); // Avoid division by zero
        let memory_factor = (metrics.memory_usage_mb / 1000.0).max(0.01);
        let latency_factor = (metrics.average_latency_ms / 10.0).max(0.01);

        (metrics.transactions_per_second / (cpu_factor * memory_factor * latency_factor)).min(10.0)
    }

    /// Assess health status of a shard
    fn assess_health_status(&self, metrics: &ShardMetrics) -> HealthStatus {
        let cpu_score = if metrics.cpu_utilization > 0.9 {
            0
        } else if metrics.cpu_utilization > 0.8 {
            1
        } else if metrics.cpu_utilization > 0.7 {
            2
        } else {
            3
        };
        let memory_score = if metrics.memory_usage_mb > 2000.0 {
            0
        } else if metrics.memory_usage_mb > 1500.0 {
            1
        } else if metrics.memory_usage_mb > 1000.0 {
            2
        } else {
            3
        };
        let queue_score = if metrics.queue_depth > 5000 {
            0
        } else if metrics.queue_depth > 2000 {
            1
        } else if metrics.queue_depth > 1000 {
            2
        } else {
            3
        };
        let latency_score = if metrics.average_latency_ms > 100.0 {
            0
        } else if metrics.average_latency_ms > 50.0 {
            1
        } else if metrics.average_latency_ms > 20.0 {
            2
        } else {
            3
        };

        let total_score = cpu_score + memory_score + queue_score + latency_score;

        match total_score {
            0..=3 => HealthStatus::Critical,
            4..=6 => HealthStatus::Poor,
            7..=9 => HealthStatus::Fair,
            10..=11 => HealthStatus::Good,
            12 => HealthStatus::Excellent,
            _ => HealthStatus::Fair,
        }
    }

    /// Identify bottleneck shards
    fn identify_bottleneck_shards(&self, metrics: &HashMap<ShardId, ShardMetrics>) -> Vec<ShardId> {
        metrics
            .iter()
            .filter(|(_, m)| {
                m.cpu_utilization > 0.8 || m.queue_depth > 2000 || m.average_latency_ms > 50.0
            })
            .map(|(id, _)| *id)
            .collect()
    }

    /// Analyze performance trends
    fn analyze_trends(&self, history: &VecDeque<MetricsSnapshot>) -> Vec<TrendData> {
        let mut trends = Vec::new();

        if history.len() < 10 {
            return trends;
        }

        let recent: Vec<_> = history.iter().skip(history.len() - 5).collect();
        let older: Vec<_> = history.iter().skip(history.len() - 10).take(5).collect();

        let recent_avg_tps: f64 =
            recent.iter().map(|s| s.total_tps).sum::<f64>() / recent.len() as f64;
        let older_avg_tps: f64 =
            older.iter().map(|s| s.total_tps).sum::<f64>() / older.len() as f64;

        if older_avg_tps > 0.0 {
            let tps_change = ((recent_avg_tps - older_avg_tps) / older_avg_tps) * 100.0;
            let direction = if tps_change > 5.0 {
                TrendDirection::Improving
            } else if tps_change < -5.0 {
                TrendDirection::Degrading
            } else {
                TrendDirection::Stable
            };

            trends.push(TrendData {
                metric_name: "TPS".to_string(),
                trend_direction: direction,
                change_percentage: tps_change,
                time_window: Duration::from_secs(300), // 5 minute window
            });
        }

        trends
    }

    /// Generate performance recommendations
    fn generate_recommendations(
        &self,
        shard_breakdown: &[ShardPerformance],
        cross_shard_analysis: &CrossShardAnalysis,
    ) -> Vec<PerformanceRecommendation> {
        let mut recommendations = Vec::new();

        // Check for overloaded shards
        for shard in shard_breakdown {
            if shard.cpu_utilization > 0.8 {
                recommendations.push(PerformanceRecommendation {
                    recommendation_type: RecommendationType::ScaleUp,
                    priority: if shard.cpu_utilization > 0.9 { Priority::Critical } else { Priority::High },
                    description: format!("Shard {:?} is CPU overloaded ({:.1}%). Consider adding more processing capacity.", 
                                       shard.shard_id, shard.cpu_utilization * 100.0),
                    expected_improvement: "20-30% reduction in CPU utilization".to_string(),
                    affected_shards: vec![shard.shard_id],
                });
            }

            if shard.queue_depth > 2000 {
                recommendations.push(PerformanceRecommendation {
                    recommendation_type: RecommendationType::Rebalance,
                    priority: Priority::Medium,
                    description: format!(
                        "Shard {:?} has high queue depth ({}). Consider load rebalancing.",
                        shard.shard_id, shard.queue_depth
                    ),
                    expected_improvement: "50% reduction in queue depth".to_string(),
                    affected_shards: vec![shard.shard_id],
                });
            }
        }

        // Check cross-shard performance
        if cross_shard_analysis.success_rate < 0.95 {
            recommendations.push(PerformanceRecommendation {
                recommendation_type: RecommendationType::Optimize,
                priority: Priority::High,
                description: format!("Cross-shard success rate is low ({:.1}%). Review network reliability and timeout settings.", 
                                   cross_shard_analysis.success_rate * 100.0),
                expected_improvement: "Improve success rate to >98%".to_string(),
                affected_shards: cross_shard_analysis.bottleneck_shards.clone(),
            });
        }

        recommendations
    }

    #[cfg(feature = "prometheus")]
    pub fn get_prometheus_registry(&self) -> &Registry {
        &self.prometheus_metrics.registry
    }
}

impl Default for ShardMetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_metrics_collector_creation() {
        let collector = ShardMetricsCollector::new();

        let metrics = collector.collect_all_shard_metrics().await;
        assert!(metrics.is_ok());
        assert!(metrics.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_shard_metrics_update() {
        let collector = ShardMetricsCollector::new();

        let test_metrics = ShardMetrics {
            shard_id: ShardId::Consensus(0),
            transactions_per_second: 1000.0,
            average_latency_ms: 10.0,
            cpu_utilization: 0.5,
            memory_usage_mb: 500.0,
            active_connections: 10,
            queue_depth: 100,
        };

        let result = collector
            .update_shard_metrics(ShardId::Consensus(0), test_metrics)
            .await;
        assert!(result.is_ok());

        let collected = collector.collect_all_shard_metrics().await.unwrap();
        assert_eq!(collected.len(), 1);
        assert_eq!(collected[0].transactions_per_second, 1000.0);
    }

    #[tokio::test]
    async fn test_performance_report_generation() {
        let collector = ShardMetricsCollector::new();

        // Add some test data
        let test_metrics = ShardMetrics {
            shard_id: ShardId::Consensus(0),
            transactions_per_second: 1000.0,
            average_latency_ms: 10.0,
            cpu_utilization: 0.5,
            memory_usage_mb: 500.0,
            active_connections: 10,
            queue_depth: 100,
        };

        collector
            .update_shard_metrics(ShardId::Consensus(0), test_metrics)
            .await
            .unwrap();
        collector.take_snapshot().await.unwrap();

        let report = collector.generate_performance_report().await;
        assert!(report.is_ok());

        let report = report.unwrap();
        assert_eq!(report.system_overview.total_shards, 1);
        assert!(report.system_overview.total_tps > 0.0);
    }
}
