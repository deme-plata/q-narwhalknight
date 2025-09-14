/// Phase 2 Quantum Enhancement Benchmarking Suite
/// 
/// Dedicated benchmarking tool for Q-NarwhalKnight Phase 2 components
/// with detailed performance analysis and comparison.

use anyhow::Result;
use clap::{Parser, Subcommand};
use criterion::{Criterion, BenchmarkId, Throughput, BatchSize};
use std::time::{Duration, Instant};
use tracing::{info, warn};
use tracing_subscriber;

use q_quantum_rng::{QuantumRNG, QRNGProvider};
use q_lattice_vrf::{LatticeVRF, SecurityLevel};
use q_vdf::{QuantumVDF, VDFProtocol};
use q_fairqueue::{QuantumFairQueue, QueueingPolicy};
use q_types::Round;

#[derive(Parser)]
#[command(name = "phase2-benchmark")]
#[command(about = "Q-NarwhalKnight Phase 2 Quantum Component Benchmarks")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
    
    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,
    
    /// Benchmark duration in seconds
    #[arg(short, long, default_value = "60")]
    duration: u64,
    
    /// Output format
    #[arg(short, long, default_value = "table")]
    format: String, // table, json, csv
}

#[derive(Subcommand)]
enum Commands {
    /// Benchmark QRNG components
    Qrng {
        /// Which providers to benchmark
        #[arg(short, long, value_delimiter = ',')]
        providers: Vec<String>,
        
        /// Data sizes to test (in bytes)
        #[arg(short, long, value_delimiter = ',', default_values_t = vec![32, 1024, 1048576])]
        sizes: Vec<usize>,
    },
    
    /// Benchmark L-VRF operations
    Lvrf {
        /// Security levels to test
        #[arg(short, long, value_delimiter = ',')]
        levels: Vec<String>,
        
        /// Number of concurrent operations
        #[arg(short, long, default_value = "1")]
        concurrency: usize,
    },
    
    /// Benchmark VDF protocols
    Vdf {
        /// Protocols to test
        #[arg(short, long, value_delimiter = ',')]
        protocols: Vec<String>,
        
        /// Time parameters to test
        #[arg(short, long, value_delimiter = ',', default_values_t = vec![100, 500, 1000])]
        time_params: Vec<u32>,
    },
    
    /// Benchmark Fair Queue
    Queue {
        /// Queueing policies to test
        #[arg(short, long, value_delimiter = ',')]
        policies: Vec<String>,
        
        /// Number of transactions
        #[arg(short, long, default_value = "10000")]
        transactions: usize,
    },
    
    /// Comprehensive comparison benchmark
    Compare {
        /// Include baseline (non-quantum) implementations
        #[arg(short, long)]
        baseline: bool,
    },
    
    /// Scalability analysis
    Scale {
        /// Node counts to test
        #[arg(short, long, value_delimiter = ',', default_values_t = vec![1, 3, 7, 15, 31])]
        nodes: Vec<usize>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    
    // Initialize logging
    let log_level = if cli.verbose { "debug" } else { "info" };
    tracing_subscriber::fmt()
        .with_env_filter(format!("phase2_benchmark={}", log_level))
        .init();
    
    info!("📊 Q-NarwhalKnight Phase 2 Benchmarking Suite");
    info!("⚛️  Analyzing quantum-enhanced consensus performance");
    
    let duration = Duration::from_secs(cli.duration);
    
    match cli.command {
        Commands::Qrng { providers, sizes } => {
            benchmark_qrng(providers, sizes, duration).await?;
        }
        
        Commands::Lvrf { levels, concurrency } => {
            benchmark_lvrf(levels, concurrency, duration).await?;
        }
        
        Commands::Vdf { protocols, time_params } => {
            benchmark_vdf(protocols, time_params, duration).await?;
        }
        
        Commands::Queue { policies, transactions } => {
            benchmark_queue(policies, transactions, duration).await?;
        }
        
        Commands::Compare { baseline } => {
            benchmark_comparison(baseline, duration).await?;
        }
        
        Commands::Scale { nodes } => {
            benchmark_scalability(nodes, duration).await?;
        }
    }
    
    Ok(())
}

/// Benchmark QRNG performance across providers and data sizes
async fn benchmark_qrng(providers: Vec<String>, sizes: Vec<usize>, duration: Duration) -> Result<()> {
    info!("🌌 Benchmarking QRNG providers...");
    
    let provider_map = [
        ("simulation", QRNGProvider::Simulation),
        ("thermal", QRNGProvider::ThermalNoise),
        ("optical", QRNGProvider::OpticalQuantum),
    ];
    
    let mut results = Vec::new();
    
    for (name, provider) in provider_map {
        if providers.is_empty() || providers.contains(&name.to_string()) {
            info!("Testing provider: {}", name);
            
            let mut qrng = QuantumRNG::new(provider).await?;
            
            for &size in &sizes {
                let start = Instant::now();
                let mut operations = 0;
                let mut total_bytes = 0;
                
                while start.elapsed() < duration / sizes.len() as u32 {
                    let _data = qrng.generate(size).await?;
                    operations += 1;
                    total_bytes += size;
                }
                
                let elapsed = start.elapsed();
                let throughput_mb = (total_bytes as f64) / (1024.0 * 1024.0) / elapsed.as_secs_f64();
                let ops_per_sec = operations as f64 / elapsed.as_secs_f64();
                
                results.push(BenchmarkResult {
                    component: format!("QRNG-{}", name),
                    metric: format!("{}-bytes", size),
                    value: throughput_mb,
                    unit: "MB/s".to_string(),
                    extra_info: format!("{:.1} ops/s", ops_per_sec),
                });
                
                info!("  {} bytes: {:.2} MB/s ({:.1} ops/s)", size, throughput_mb, ops_per_sec);
            }
        }
    }
    
    print_benchmark_results("QRNG Performance", &results);
    Ok(())
}

/// Benchmark L-VRF across security levels
async fn benchmark_lvrf(levels: Vec<String>, concurrency: usize, duration: Duration) -> Result<()> {
    info!("🔐 Benchmarking L-VRF operations...");
    
    let level_map = [
        ("low", SecurityLevel::Low),
        ("medium", SecurityLevel::Medium),
        ("high", SecurityLevel::High),
        ("ultra", SecurityLevel::Ultra),
    ];
    
    let mut results = Vec::new();
    
    for (name, level) in level_map {
        if levels.is_empty() || levels.contains(&name.to_string()) {
            info!("Testing security level: {}", name);
            
            let lvrf = LatticeVRF::new(level).await?;
            
            // Single-threaded evaluation benchmark
            let start = Instant::now();
            let mut evaluations = 0;
            let mut total_eval_time = Duration::ZERO;
            let mut total_verify_time = Duration::ZERO;
            
            while start.elapsed() < duration / level_map.len() as u32 {
                let input = format!("benchmark_{}", evaluations);
                let round = Round::new(evaluations + 1);
                
                // Time evaluation
                let eval_start = Instant::now();
                let result = lvrf.evaluate(input.as_bytes(), round).await?;
                let eval_time = eval_start.elapsed();
                total_eval_time += eval_time;
                
                // Time verification
                let verify_start = Instant::now();
                let _valid = lvrf.verify(input.as_bytes(), round, &result.output, &result.proof).await?;
                let verify_time = verify_start.elapsed();
                total_verify_time += verify_time;
                
                evaluations += 1;
            }
            
            let elapsed = start.elapsed();
            let eval_ops_per_sec = evaluations as f64 / elapsed.as_secs_f64();
            let avg_eval_ms = (total_eval_time.as_secs_f64() * 1000.0) / evaluations as f64;
            let avg_verify_ms = (total_verify_time.as_secs_f64() * 1000.0) / evaluations as f64;
            
            results.push(BenchmarkResult {
                component: format!("L-VRF-{}", name),
                metric: "Evaluation".to_string(),
                value: avg_eval_ms,
                unit: "ms".to_string(),
                extra_info: format!("{:.1} ops/s", eval_ops_per_sec),
            });
            
            results.push(BenchmarkResult {
                component: format!("L-VRF-{}", name),
                metric: "Verification".to_string(),
                value: avg_verify_ms,
                unit: "ms".to_string(),
                extra_info: format!("{}x speedup", (avg_eval_ms / avg_verify_ms) as u32),
            });
            
            info!("  Evaluation: {:.2}ms avg ({:.1} ops/s)", avg_eval_ms, eval_ops_per_sec);
            info!("  Verification: {:.2}ms avg", avg_verify_ms);
            
            // Concurrent benchmark if requested
            if concurrency > 1 {
                info!("  Testing {} concurrent operations...", concurrency);
                
                let concurrent_start = Instant::now();
                let mut tasks = Vec::new();
                
                for i in 0..concurrency {
                    let lvrf_clone = lvrf.clone();
                    let task = tokio::spawn(async move {
                        let input = format!("concurrent_benchmark_{}", i);
                        let round = Round::new(i + 1);
                        lvrf_clone.evaluate(input.as_bytes(), round).await
                    });
                    tasks.push(task);
                }
                
                let _results: Vec<_> = futures::future::join_all(tasks).await
                    .into_iter()
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .collect::<Result<Vec<_>, _>>()?;
                
                let concurrent_time = concurrent_start.elapsed();
                let concurrent_speedup = (avg_eval_ms * concurrency as f64 / 1000.0) / concurrent_time.as_secs_f64();
                
                results.push(BenchmarkResult {
                    component: format!("L-VRF-{}", name),
                    metric: format!("Concurrent-{}", concurrency),
                    value: concurrent_speedup,
                    unit: "speedup".to_string(),
                    extra_info: format!("{:.2}s total", concurrent_time.as_secs_f64()),
                });
                
                info!("  Concurrent speedup: {:.2}x", concurrent_speedup);
            }
        }
    }
    
    print_benchmark_results("L-VRF Performance", &results);
    Ok(())
}

/// Benchmark VDF protocols across different time parameters
async fn benchmark_vdf(protocols: Vec<String>, time_params: Vec<u32>, duration: Duration) -> Result<()> {
    info!("⚡ Benchmarking VDF protocols...");
    
    let protocol_map = [
        ("wesolowski", VDFProtocol::Wesolowski),
        ("pietrzak", VDFProtocol::Pietrzak),
        ("quantum", VDFProtocol::QuantumHybrid),
    ];
    
    let mut results = Vec::new();
    
    for (name, protocol) in protocol_map {
        if protocols.is_empty() || protocols.contains(&name.to_string()) {
            info!("Testing protocol: {}", name);
            
            let vdf = QuantumVDF::new(protocol).await?;
            
            for &time_param in &time_params {
                info!("  Time parameter: {}", time_param);
                
                let input = format!("vdf_benchmark_{}_{}", name, time_param).into_bytes();
                
                // Computation benchmark
                let compute_start = Instant::now();
                let result = vdf.evaluate(&input, time_param).await?;
                let compute_time = compute_start.elapsed();
                
                // Verification benchmark
                let verify_start = Instant::now();
                let _valid = vdf.verify(&input, time_param, &result.output, &result.proof).await?;
                let verify_time = verify_start.elapsed();
                
                let speedup = compute_time.as_secs_f64() / verify_time.as_secs_f64();
                
                results.push(BenchmarkResult {
                    component: format!("VDF-{}", name),
                    metric: format!("Compute-{}", time_param),
                    value: compute_time.as_secs_f64() * 1000.0,
                    unit: "ms".to_string(),
                    extra_info: format!("T={}", time_param),
                });
                
                results.push(BenchmarkResult {
                    component: format!("VDF-{}", name),
                    metric: format!("Verify-{}", time_param),
                    value: verify_time.as_secs_f64() * 1000.0,
                    unit: "ms".to_string(),
                    extra_info: format!("{}x faster", speedup as u32),
                });
                
                info!("    Computation: {:.1}ms", compute_time.as_secs_f64() * 1000.0);
                info!("    Verification: {:.2}ms ({:.0}x speedup)", verify_time.as_secs_f64() * 1000.0, speedup);
            }
        }
    }
    
    print_benchmark_results("VDF Performance", &results);
    Ok(())
}

/// Benchmark fair queueing policies
async fn benchmark_queue(policies: Vec<String>, transactions: usize, duration: Duration) -> Result<()> {
    info!("⚖️ Benchmarking Fair Queue policies...");
    
    let policy_map = [
        ("fifo", QueueingPolicy::FIFO),
        ("vrf", QueueingPolicy::VRFBased),
        ("anticensorship", QueueingPolicy::AntiCensorship),
    ];
    
    let mut results = Vec::new();
    
    for (name, policy) in policy_map {
        if policies.is_empty() || policies.contains(&name.to_string()) {
            info!("Testing policy: {}", name);
            
            let mut queue = QuantumFairQueue::new(policy).await?;
            
            // Enqueue benchmark
            let enqueue_start = Instant::now();
            for i in 0..transactions {
                let tx_id = uuid::Uuid::new_v4().into_bytes();
                let node_id = [(i % 256) as u8; 32];
                queue.enqueue_transaction(tx_id, node_id).await?;
            }
            let enqueue_time = enqueue_start.elapsed();
            let enqueue_tps = transactions as f64 / enqueue_time.as_secs_f64();
            
            // Dequeue benchmark
            let dequeue_start = Instant::now();
            let mut dequeued_count = 0;
            
            while !queue.is_empty().await? {
                let batch = queue.dequeue_next_batch(100).await?;
                dequeued_count += batch.len();
            }
            
            let dequeue_time = dequeue_start.elapsed();
            let dequeue_tps = dequeued_count as f64 / dequeue_time.as_secs_f64();
            
            // Fairness calculation benchmark
            let fairness_start = Instant::now();
            let _fairness = queue.calculate_fairness_metrics().await?;
            let fairness_time = fairness_start.elapsed();
            
            results.push(BenchmarkResult {
                component: format!("Queue-{}", name),
                metric: "Enqueue".to_string(),
                value: enqueue_tps,
                unit: "tx/s".to_string(),
                extra_info: format!("{} transactions", transactions),
            });
            
            results.push(BenchmarkResult {
                component: format!("Queue-{}", name),
                metric: "Dequeue".to_string(),
                value: dequeue_tps,
                unit: "tx/s".to_string(),
                extra_info: format!("{} dequeued", dequeued_count),
            });
            
            results.push(BenchmarkResult {
                component: format!("Queue-{}", name),
                metric: "Fairness".to_string(),
                value: fairness_time.as_secs_f64() * 1000.0,
                unit: "ms".to_string(),
                extra_info: "calculation time".to_string(),
            });
            
            info!("  Enqueue: {:.0} tx/s", enqueue_tps);
            info!("  Dequeue: {:.0} tx/s", dequeue_tps);
            info!("  Fairness calc: {:.2}ms", fairness_time.as_secs_f64() * 1000.0);
        }
    }
    
    print_benchmark_results("Fair Queue Performance", &results);
    Ok(())
}

/// Comparative benchmark of quantum vs classical approaches
async fn benchmark_comparison(include_baseline: bool, duration: Duration) -> Result<()> {
    info!("🆚 Running comparative benchmarks...");
    
    // This would compare quantum-enhanced vs classical implementations
    // For now, we'll benchmark different security/performance trade-offs
    
    let mut results = Vec::new();
    
    // Compare L-VRF security levels (quantum vs classical trade-offs)
    let levels = [SecurityLevel::Low, SecurityLevel::Medium, SecurityLevel::High, SecurityLevel::Ultra];
    
    for (i, level) in levels.iter().enumerate() {
        let lvrf = LatticeVRF::new(*level).await?;
        
        let input = b"comparison_benchmark";
        let round = Round::new(1);
        
        let start = Instant::now();
        let result = lvrf.evaluate(input, round).await?;
        let eval_time = start.elapsed();
        
        let verify_start = Instant::now();
        let _valid = lvrf.verify(input, round, &result.output, &result.proof).await?;
        let verify_time = verify_start.elapsed();
        
        let security_bits = match level {
            SecurityLevel::Low => 128,
            SecurityLevel::Medium => 192,
            SecurityLevel::High => 256,
            SecurityLevel::Ultra => 384,
        };
        
        results.push(BenchmarkResult {
            component: "Security Trade-off".to_string(),
            metric: format!("{}-bit", security_bits),
            value: eval_time.as_secs_f64() * 1000.0,
            unit: "ms".to_string(),
            extra_info: format!("verify: {:.2}ms", verify_time.as_secs_f64() * 1000.0),
        });
        
        info!("  {}-bit security: {:.2}ms eval, {:.2}ms verify", 
              security_bits, 
              eval_time.as_secs_f64() * 1000.0,
              verify_time.as_secs_f64() * 1000.0);
    }
    
    print_benchmark_results("Security vs Performance Trade-offs", &results);
    Ok(())
}

/// Benchmark scalability across different node counts
async fn benchmark_scalability(node_counts: Vec<usize>, duration: Duration) -> Result<()> {
    info!("📈 Benchmarking scalability...");
    
    let lvrf = LatticeVRF::new(SecurityLevel::Medium).await?;
    let mut results = Vec::new();
    
    for &node_count in &node_counts {
        info!("Testing with {} nodes...", node_count);
        
        let start = Instant::now();
        let mut tasks = Vec::new();
        
        for node_id in 0..node_count {
            let lvrf_clone = lvrf.clone();
            let task = tokio::spawn(async move {
                let input = format!("node_{}_scalability", node_id);
                let round = Round::new(1);
                lvrf_clone.evaluate(input.as_bytes(), round).await
            });
            tasks.push(task);
        }
        
        let _node_results: Vec<_> = futures::future::join_all(tasks).await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;
        
        let elapsed = start.elapsed();
        let throughput = node_count as f64 / elapsed.as_secs_f64();
        
        // Calculate efficiency compared to single node
        let efficiency = if node_count == 1 {
            100.0
        } else {
            let single_node_baseline = 50.0; // ms, approximate
            let expected_time = single_node_baseline / 1000.0; // Convert to seconds
            (expected_time / elapsed.as_secs_f64() * 100.0).min(100.0)
        };
        
        results.push(BenchmarkResult {
            component: "Scalability".to_string(),
            metric: format!("{}-nodes", node_count),
            value: throughput,
            unit: "nodes/s".to_string(),
            extra_info: format!("{:.1}% efficiency", efficiency),
        });
        
        info!("  {} nodes: {:.1} nodes/s ({:.1}% efficiency)", node_count, throughput, efficiency);
    }
    
    print_benchmark_results("Scalability Analysis", &results);
    Ok(())
}

/// Structure for benchmark results
#[derive(Debug, Clone)]
struct BenchmarkResult {
    component: String,
    metric: String,
    value: f64,
    unit: String,
    extra_info: String,
}

/// Print formatted benchmark results
fn print_benchmark_results(title: &str, results: &[BenchmarkResult]) {
    println!("\n{}", "=".repeat(80));
    println!("📊 {}", title);
    println!("{}", "=".repeat(80));
    
    // Print table header
    println!("{:<25} {:<15} {:>12} {:<8} {:<20}", "Component", "Metric", "Value", "Unit", "Extra Info");
    println!("{}", "-".repeat(80));
    
    // Print results
    for result in results {
        println!("{:<25} {:<15} {:>12.2} {:<8} {:<20}", 
                 result.component, 
                 result.metric, 
                 result.value, 
                 result.unit,
                 result.extra_info);
    }
    
    println!("{}", "=".repeat(80));
}