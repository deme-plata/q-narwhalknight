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

use q_types::Digest; // needed for Sha3_256::new()

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
        info!("Measuring latency characteristics with real transaction processing");

        let num_samples = 500usize;
        let mut latencies_ms = Vec::with_capacity(num_samples);

        // Create a signing context for real crypto operations
        let mut seed = [0u8; 32];
        seed[0] = 0x1A;
        seed[1] = 0x7E;
        let signing_key = q_types::SecretKey::from_bytes(&seed);
        let from_address: q_types::Address = signing_key.verifying_key().to_bytes();

        for i in 0..num_samples {
            let start = Instant::now();

            // Full transaction lifecycle: sign, serialize, deserialize, verify
            let tx = {
                use ed25519_dalek::Signer;
                let to: q_types::Address = [0xEEu8; 32];
                let amount: q_types::Amount = 1_000_000;
                let fee: q_types::Amount = 100;
                let nonce = i as u64;

                let mut msg = Vec::with_capacity(32 + 32 + 16 + 16 + 8);
                msg.extend_from_slice(&from_address);
                msg.extend_from_slice(&to);
                msg.extend_from_slice(&amount.to_le_bytes());
                msg.extend_from_slice(&fee.to_le_bytes());
                msg.extend_from_slice(&nonce.to_le_bytes());

                let signature = signing_key.sign(&msg);

                let mut hasher = q_types::Sha3_256::new();
                hasher.update(&msg);
                let result = hasher.finalize();
                let mut id = [0u8; 32];
                id.copy_from_slice(&result);

                q_types::Transaction {
                    id,
                    from: from_address,
                    to,
                    amount,
                    fee,
                    nonce,
                    signature: signature.to_bytes().to_vec(),
                    timestamp: chrono::Utc::now(),
                    data: Vec::new(),
                    token_type: q_types::TokenType::QUG,
                    fee_token_type: q_types::TokenType::QUGUSD,
                    tx_type: q_types::TransactionType::Transfer,
                    pqc_signature: None,
                    signature_phase: q_types::TxSignaturePhase::Phase0Ed25519,
                    pqc_public_key: None,
                    zk_proof_bundle: None,
                    privacy_level: Default::default(),
                    bulletproof: None,
                    nullifier: None,
                    memo: None,
                }
            };

            // Serialize (network encoding)
            let encoded = bincode::serialize(&tx).unwrap();

            // Deserialize (network decoding)
            let decoded: q_types::Transaction = bincode::deserialize(&encoded).unwrap();

            // Verify signature
            {
                use ed25519_dalek::{Signature, Verifier, VerifyingKey};
                let vk = VerifyingKey::from_bytes(&decoded.from).unwrap();
                let mut msg = Vec::with_capacity(32 + 32 + 16 + 16 + 8);
                msg.extend_from_slice(&decoded.from);
                msg.extend_from_slice(&decoded.to);
                msg.extend_from_slice(&decoded.amount.to_le_bytes());
                msg.extend_from_slice(&decoded.fee.to_le_bytes());
                msg.extend_from_slice(&decoded.nonce.to_le_bytes());
                let mut sig_bytes = [0u8; 64];
                sig_bytes.copy_from_slice(&decoded.signature);
                let sig = Signature::from_bytes(&sig_bytes);
                vk.verify(&msg, &sig).unwrap();
            }

            std::hint::black_box(&encoded);

            let elapsed = start.elapsed();
            latencies_ms.push(elapsed.as_secs_f64() * 1000.0);
        }

        // Sort for percentile calculation
        latencies_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mean_latency_ms = latencies_ms.iter().sum::<f64>() / latencies_ms.len() as f64;
        let p50_latency_ms = percentile(&latencies_ms, 50.0);
        let p95_latency_ms = percentile(&latencies_ms, 95.0);
        let p99_latency_ms = percentile(&latencies_ms, 99.0);
        let p999_latency_ms = percentile(&latencies_ms, 99.9);
        let max_latency_ms = latencies_ms.last().copied().unwrap_or(0.0);

        info!("Latency results ({} samples):", num_samples);
        info!("  Mean:  {:.3}ms", mean_latency_ms);
        info!("  P50:   {:.3}ms", p50_latency_ms);
        info!("  P95:   {:.3}ms", p95_latency_ms);
        info!("  P99:   {:.3}ms", p99_latency_ms);
        info!("  P99.9: {:.3}ms", p999_latency_ms);
        info!("  Max:   {:.3}ms", max_latency_ms);

        Ok(LatencyMetrics {
            mean_latency_ms,
            p50_latency_ms,
            p95_latency_ms,
            p99_latency_ms,
            p999_latency_ms,
            max_latency_ms,
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

/// Compute a percentile value from a sorted slice of f64 values.
/// `p` is in the range [0.0, 100.0]. The slice must already be sorted ascending.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let rank = (p / 100.0) * (sorted.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    let frac = rank - lower as f64;
    if upper >= sorted.len() {
        sorted[sorted.len() - 1]
    } else {
        sorted[lower] * (1.0 - frac) + sorted[upper] * frac
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
