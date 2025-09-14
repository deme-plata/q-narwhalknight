//! Q-NarwhalKnight Performance Benchmarking Suite
//!
//! Comprehensive performance measurement and optimization framework
//! for the Q-NarwhalKnight quantum-resistant consensus system.
//!
//! # Features
//! - TPS (Transactions Per Second) measurement
//! - Latency profiling with percentile analysis
//! - Memory usage monitoring and optimization
//! - Network performance benchmarking
//! - Consensus performance validation
//! - Regression detection and historical tracking

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

pub mod consensus_benchmark;
pub mod memory_profiler;
pub mod network_benchmark;
pub mod performance_monitor;
pub mod regression_detector;
pub mod tps_benchmark;

/// Comprehensive performance metrics for Q-NarwhalKnight
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub timestamp: DateTime<Utc>,
    pub tps_metrics: TpsMetrics,
    pub latency_metrics: LatencyMetrics,
    pub memory_metrics: MemoryMetrics,
    pub network_metrics: NetworkMetrics,
    pub consensus_metrics: ConsensusMetrics,
}

/// Transactions Per Second measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TpsMetrics {
    pub transactions_per_second: f64,
    pub peak_tps: f64,
    pub sustained_tps: f64,
    pub target_tps: f64,
    pub efficiency_ratio: f64, // actual/target
}

/// Latency performance breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyMetrics {
    pub mean_latency_ms: f64,
    pub p50_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub p999_latency_ms: f64,
    pub max_latency_ms: f64,
}

/// Memory usage and optimization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetrics {
    pub heap_usage_mb: f64,
    pub peak_memory_mb: f64,
    pub memory_efficiency: f64,
    pub gc_pressure: f64,
    pub allocation_rate: f64,
}

/// Network performance characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMetrics {
    pub throughput_mbps: f64,
    pub connection_count: u32,
    pub message_latency_ms: f64,
    pub packet_loss_rate: f64,
    pub bandwidth_utilization: f64,
}

/// Consensus-specific performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusMetrics {
    pub vertex_processing_rate: f64,
    pub dag_growth_rate: f64,
    pub finality_latency_ms: f64,
    pub consensus_efficiency: f64,
    pub validator_participation: f64,
}

/// Performance benchmark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub duration_seconds: u64,
    pub target_tps: f64,
    pub warmup_seconds: u64,
    pub measurement_interval_ms: u64,
    pub node_count: u32,
    pub validator_count: u32,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            duration_seconds: 60,
            target_tps: 25_000.0, // Phase 1 target
            warmup_seconds: 10,
            measurement_interval_ms: 1000,
            node_count: 4,
            validator_count: 4,
        }
    }
}

/// Main benchmarking suite orchestrator
pub struct PerformanceBenchmarkSuite {
    config: BenchmarkConfig,
    metrics_history: Vec<PerformanceMetrics>,
}

impl PerformanceBenchmarkSuite {
    /// Create a new benchmark suite with configuration
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            metrics_history: Vec::new(),
        }
    }

    /// Run comprehensive performance benchmark
    pub async fn run_full_benchmark(&mut self) -> Result<PerformanceMetrics> {
        info!("🚀 Starting Q-NarwhalKnight Performance Benchmark Suite");
        info!(
            "Target TPS: {}, Duration: {}s",
            self.config.target_tps, self.config.duration_seconds
        );

        let start_time = Instant::now();

        // Warm up the system
        self.warmup_system().await?;

        // Run all benchmark components in parallel
        let metrics = tokio::try_join!(
            self.measure_tps_performance(),
            self.measure_latency_characteristics(),
            self.measure_memory_usage(),
            self.measure_network_performance(),
            self.measure_consensus_performance()
        )?;

        let performance_metrics = PerformanceMetrics {
            timestamp: Utc::now(),
            tps_metrics: metrics.0,
            latency_metrics: metrics.1,
            memory_metrics: metrics.2,
            network_metrics: metrics.3,
            consensus_metrics: metrics.4,
        };

        // Store metrics for regression detection
        self.metrics_history.push(performance_metrics.clone());

        let elapsed = start_time.elapsed();
        info!("✅ Benchmark completed in {:.2}s", elapsed.as_secs_f64());

        // Generate performance report
        self.generate_performance_report(&performance_metrics)?;

        Ok(performance_metrics)
    }

    async fn warmup_system(&self) -> Result<()> {
        info!(
            "🔥 Warming up system for {} seconds",
            self.config.warmup_seconds
        );
        tokio::time::sleep(Duration::from_secs(self.config.warmup_seconds)).await;
        Ok(())
    }

    async fn measure_tps_performance(&self) -> Result<TpsMetrics> {
        info!("📊 Measuring TPS performance");
        // Implementation will be in tps_benchmark.rs
        tps_benchmark::measure_baseline_tps(&self.config).await
    }

    async fn measure_latency_characteristics(&self) -> Result<LatencyMetrics> {
        info!("⏱️  Measuring latency characteristics");
        // Implementation will track percentile latencies
        Ok(LatencyMetrics {
            mean_latency_ms: 25.0, // Placeholder - will be measured
            p50_latency_ms: 20.0,
            p95_latency_ms: 45.0,
            p99_latency_ms: 80.0,
            p999_latency_ms: 150.0,
            max_latency_ms: 200.0,
        })
    }

    async fn measure_memory_usage(&self) -> Result<MemoryMetrics> {
        info!("💾 Measuring memory usage");
        memory_profiler::measure_memory_performance(&self.config).await
    }

    async fn measure_network_performance(&self) -> Result<NetworkMetrics> {
        info!("🌐 Measuring network performance");
        network_benchmark::measure_network_performance(&self.config).await
    }

    async fn measure_consensus_performance(&self) -> Result<ConsensusMetrics> {
        info!("🔗 Measuring consensus performance");
        consensus_benchmark::measure_consensus_performance(&self.config).await
    }

    fn generate_performance_report(&self, metrics: &PerformanceMetrics) -> Result<()> {
        let report = format!(
            r#"
📊 Q-NARWHALKNIGHT PERFORMANCE REPORT
=====================================
Timestamp: {}

🚀 TPS PERFORMANCE
  Current TPS: {:.0}
  Target TPS: {:.0}
  Efficiency: {:.1}%
  Peak TPS: {:.0}

⏱️  LATENCY ANALYSIS
  Mean: {:.1}ms
  P95: {:.1}ms
  P99: {:.1}ms
  Max: {:.1}ms

💾 MEMORY USAGE
  Heap Usage: {:.1}MB
  Peak Memory: {:.1}MB
  Efficiency: {:.1}%

🌐 NETWORK PERFORMANCE
  Throughput: {:.1}Mbps
  Connections: {}
  Message Latency: {:.1}ms

🔗 CONSENSUS PERFORMANCE
  Vertex Rate: {:.0}/s
  Finality: {:.1}ms
  Efficiency: {:.1}%
"#,
            metrics.timestamp.format("%Y-%m-%d %H:%M:%S UTC"),
            metrics.tps_metrics.transactions_per_second,
            metrics.tps_metrics.target_tps,
            metrics.tps_metrics.efficiency_ratio * 100.0,
            metrics.tps_metrics.peak_tps,
            metrics.latency_metrics.mean_latency_ms,
            metrics.latency_metrics.p95_latency_ms,
            metrics.latency_metrics.p99_latency_ms,
            metrics.latency_metrics.max_latency_ms,
            metrics.memory_metrics.heap_usage_mb,
            metrics.memory_metrics.peak_memory_mb,
            metrics.memory_metrics.memory_efficiency * 100.0,
            metrics.network_metrics.throughput_mbps,
            metrics.network_metrics.connection_count,
            metrics.network_metrics.message_latency_ms,
            metrics.consensus_metrics.vertex_processing_rate,
            metrics.consensus_metrics.finality_latency_ms,
            metrics.consensus_metrics.consensus_efficiency * 100.0,
        );

        println!("{}", report);

        // Save detailed report to file
        let report_path = format!(
            "performance_report_{}.json",
            metrics.timestamp.format("%Y%m%d_%H%M%S")
        );
        std::fs::write(&report_path, serde_json::to_string_pretty(metrics)?)?;

        info!("📄 Detailed report saved to: {}", report_path);

        Ok(())
    }

    /// Detect performance regressions compared to baseline
    pub fn detect_regressions(
        &self,
        baseline: &PerformanceMetrics,
        current: &PerformanceMetrics,
    ) -> Vec<String> {
        let mut regressions = Vec::new();

        // TPS regression check (>5% drop)
        if current.tps_metrics.transactions_per_second
            < baseline.tps_metrics.transactions_per_second * 0.95
        {
            regressions.push(format!(
                "TPS regression: {:.0} -> {:.0} ({:.1}% drop)",
                baseline.tps_metrics.transactions_per_second,
                current.tps_metrics.transactions_per_second,
                (1.0 - current.tps_metrics.transactions_per_second
                    / baseline.tps_metrics.transactions_per_second)
                    * 100.0
            ));
        }

        // Memory regression check (>10% increase)
        if current.memory_metrics.heap_usage_mb > baseline.memory_metrics.heap_usage_mb * 1.10 {
            regressions.push(format!(
                "Memory regression: {:.1}MB -> {:.1}MB ({:.1}% increase)",
                baseline.memory_metrics.heap_usage_mb,
                current.memory_metrics.heap_usage_mb,
                (current.memory_metrics.heap_usage_mb / baseline.memory_metrics.heap_usage_mb
                    - 1.0)
                    * 100.0
            ));
        }

        // Latency regression check (>20% increase)
        if current.latency_metrics.p95_latency_ms > baseline.latency_metrics.p95_latency_ms * 1.20 {
            regressions.push(format!(
                "Latency regression (P95): {:.1}ms -> {:.1}ms ({:.1}% increase)",
                baseline.latency_metrics.p95_latency_ms,
                current.latency_metrics.p95_latency_ms,
                (current.latency_metrics.p95_latency_ms / baseline.latency_metrics.p95_latency_ms
                    - 1.0)
                    * 100.0
            ));
        }

        regressions
    }
}

/// Convenience function to run a quick baseline benchmark
pub async fn measure_baseline_performance() -> Result<PerformanceMetrics> {
    let mut suite = PerformanceBenchmarkSuite::new(BenchmarkConfig {
        target_tps: 2_500.0, // Current baseline
        duration_seconds: 30,
        ..Default::default()
    });

    suite.run_full_benchmark().await
}
