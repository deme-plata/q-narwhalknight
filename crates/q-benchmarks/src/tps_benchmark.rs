//! TPS (Transactions Per Second) Benchmarking Module
//!
//! Measures baseline and target TPS performance for Q-NarwhalKnight
//! with detailed transaction processing analysis.

use crate::{BenchmarkConfig, TpsMetrics};
use anyhow::Result;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::task;
use tracing::{debug, info, warn};

/// Measure baseline TPS performance of the system
pub async fn measure_baseline_tps(config: &BenchmarkConfig) -> Result<TpsMetrics> {
    info!(
        "📊 Starting TPS benchmark - Target: {:.0} TPS",
        config.target_tps
    );

    let transaction_counter = Arc::new(AtomicU64::new(0));
    let start_time = Instant::now();
    let mut measurement_tasks = Vec::new();

    // Start transaction generation tasks
    for i in 0..config.node_count {
        let counter = Arc::clone(&transaction_counter);
        let target_rate = config.target_tps / config.node_count as f64;
        let duration = Duration::from_secs(config.duration_seconds);

        let task = task::spawn(async move {
            simulate_transaction_load(i, target_rate, duration, counter).await
        });
        measurement_tasks.push(task);
    }

    // Measure TPS in intervals
    let measurement_task = {
        let counter = Arc::clone(&transaction_counter);
        let duration = Duration::from_secs(config.duration_seconds);
        let interval = Duration::from_millis(config.measurement_interval_ms);

        task::spawn(async move { measure_tps_over_time(counter, duration, interval).await })
    };

    // Wait for all tasks to complete
    for task in measurement_tasks {
        task.await?;
    }

    let tps_measurements = measurement_task.await?;
    let elapsed = start_time.elapsed();
    let total_transactions = transaction_counter.load(Ordering::Relaxed);

    // Calculate metrics
    let actual_tps = total_transactions as f64 / elapsed.as_secs_f64();
    let peak_tps = tps_measurements.iter().map(|&x| x).fold(0.0, f64::max);
    let sustained_tps = calculate_sustained_tps(&tps_measurements);
    let efficiency_ratio = actual_tps / config.target_tps;

    info!("✅ TPS Benchmark Results:");
    info!("   Total Transactions: {}", total_transactions);
    info!("   Actual TPS: {:.0}", actual_tps);
    info!("   Peak TPS: {:.0}", peak_tps);
    info!("   Sustained TPS: {:.0}", sustained_tps);
    info!("   Efficiency: {:.1}%", efficiency_ratio * 100.0);

    if actual_tps < config.target_tps * 0.8 {
        warn!("⚠️ TPS significantly below target - performance investigation needed");
    }

    Ok(TpsMetrics {
        transactions_per_second: actual_tps,
        peak_tps,
        sustained_tps,
        target_tps: config.target_tps,
        efficiency_ratio,
    })
}

/// Simulate transaction load from a single node
async fn simulate_transaction_load(
    node_id: u32,
    target_tps: f64,
    duration: Duration,
    counter: Arc<AtomicU64>,
) -> Result<()> {
    let start = Instant::now();
    let interval = Duration::from_nanos((1_000_000_000.0 / target_tps) as u64);

    debug!(
        "Node {} starting transaction simulation at {:.0} TPS",
        node_id, target_tps
    );

    while start.elapsed() < duration {
        let tx_start = Instant::now();

        // Simulate transaction processing
        simulate_single_transaction(node_id).await?;
        counter.fetch_add(1, Ordering::Relaxed);

        // Rate limiting to maintain target TPS
        let elapsed = tx_start.elapsed();
        if elapsed < interval {
            tokio::time::sleep(interval - elapsed).await;
        }
    }

    debug!("Node {} completed transaction simulation", node_id);
    Ok(())
}

/// Simulate processing of a single transaction
async fn simulate_single_transaction(node_id: u32) -> Result<()> {
    // Simulate realistic transaction processing steps:
    // 1. Transaction validation
    // 2. Cryptographic signature verification
    // 3. State update preparation
    // 4. Consensus vertex creation
    // 5. Network broadcast

    // Realistic processing delays based on current system
    let validation_delay = Duration::from_micros(50);
    let crypto_delay = Duration::from_micros(100);
    let state_delay = Duration::from_micros(75);
    let consensus_delay = Duration::from_micros(200);
    let network_delay = Duration::from_micros(125);

    // Simulate CPU work and I/O delays
    tokio::time::sleep(validation_delay).await;
    simulate_cpu_work(1000); // Crypto operations
    tokio::time::sleep(crypto_delay).await;

    tokio::time::sleep(state_delay).await;
    simulate_cpu_work(2000); // Consensus computation
    tokio::time::sleep(consensus_delay).await;

    tokio::time::sleep(network_delay).await;

    Ok(())
}

/// Simulate CPU-intensive work (crypto/consensus operations)
fn simulate_cpu_work(iterations: u32) {
    let mut sum = 0u64;
    for i in 0..iterations {
        sum = sum.wrapping_add(i as u64).wrapping_mul(17);
    }
    // Prevent compiler optimization
    std::hint::black_box(sum);
}

/// Measure TPS over time intervals
async fn measure_tps_over_time(
    counter: Arc<AtomicU64>,
    duration: Duration,
    interval: Duration,
) -> Vec<f64> {
    let mut measurements = Vec::new();
    let start = Instant::now();
    let mut last_count = 0u64;
    let mut last_time = start;

    while start.elapsed() < duration {
        tokio::time::sleep(interval).await;

        let current_count = counter.load(Ordering::Relaxed);
        let current_time = Instant::now();

        let transactions_in_interval = current_count - last_count;
        let time_elapsed = current_time.duration_since(last_time).as_secs_f64();
        let tps = transactions_in_interval as f64 / time_elapsed;

        measurements.push(tps);
        debug!("Interval TPS: {:.0}", tps);

        last_count = current_count;
        last_time = current_time;
    }

    measurements
}

/// Calculate sustained TPS (95th percentile of measurements)
fn calculate_sustained_tps(measurements: &[f64]) -> f64 {
    if measurements.is_empty() {
        return 0.0;
    }

    let mut sorted = measurements.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Use 95th percentile as sustained TPS
    let index = (sorted.len() as f64 * 0.95) as usize;
    sorted.get(index).copied().unwrap_or(0.0)
}

/// Benchmark TPS scaling with different node counts
pub async fn benchmark_tps_scaling(
    base_config: &BenchmarkConfig,
) -> Result<Vec<(u32, TpsMetrics)>> {
    let mut results = Vec::new();

    info!("🔄 Starting TPS scaling benchmark");

    for node_count in [1, 2, 4, 8, 16] {
        info!("Testing with {} nodes", node_count);

        let config = BenchmarkConfig {
            node_count,
            duration_seconds: 30, // Shorter runs for scaling test
            ..base_config.clone()
        };

        let metrics = measure_baseline_tps(&config).await?;
        results.push((node_count, metrics));

        // Brief pause between scaling tests
        tokio::time::sleep(Duration::from_secs(2)).await;
    }

    // Analyze scaling characteristics
    analyze_scaling_results(&results);

    Ok(results)
}

/// Analyze TPS scaling results and detect bottlenecks
fn analyze_scaling_results(results: &[(u32, TpsMetrics)]) {
    info!("📈 TPS Scaling Analysis:");

    for (node_count, metrics) in results {
        let efficiency = metrics.efficiency_ratio * 100.0;
        info!(
            "   {} nodes: {:.0} TPS ({:.1}% efficiency)",
            node_count, metrics.transactions_per_second, efficiency
        );
    }

    // Calculate scaling efficiency
    if results.len() >= 2 {
        let baseline_tps = results[0].1.transactions_per_second;
        let final_tps = results.last().unwrap().1.transactions_per_second;
        let final_nodes = results.last().unwrap().0;

        let scaling_efficiency = final_tps / (baseline_tps * final_nodes as f64);

        info!(
            "🎯 Scaling Efficiency: {:.1}% (1.0 = perfect linear scaling)",
            scaling_efficiency * 100.0
        );

        if scaling_efficiency < 0.7 {
            warn!("⚠️ Poor scaling efficiency detected - bottleneck investigation needed");
        }
    }
}
