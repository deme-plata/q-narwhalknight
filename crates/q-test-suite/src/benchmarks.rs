/// Performance benchmarks for quantum-enhanced consensus components

use anyhow::Result;
use criterion::{Criterion, BenchmarkId, Throughput, BatchSize};
use std::time::{Duration, Instant};
use uuid::Uuid;
use futures::future::join_all;

use q_quantum_rng::{QuantumRNG, QRNGProvider};
use q_lattice_vrf::{LatticeVRF, SecurityLevel};
use q_vdf::{QuantumVDF, VDFProtocol};
use q_fairqueue::{QuantumFairQueue, QueueingPolicy};
use q_types::Round;

/// Benchmark results aggregation
#[derive(Debug, Default)]
pub struct BenchmarkResults {
    pub passed: bool,
    pub qrng_throughput: f64,      // MB/s
    pub lvrf_eval_time: Duration,   // per evaluation
    pub vdf_compute_time: Duration, // per evaluation  
    pub queue_throughput: f64,      // transactions/s
    pub consensus_latency: Duration, // end-to-end
    pub details: Vec<BenchmarkDetail>,
}

impl BenchmarkResults {
    pub fn summary(&self) -> String {
        format!(
            "QRNG: {:.1} MB/s | L-VRF: {:.1}ms | VDF: {:.1}ms | Queue: {:.0} tx/s | Consensus: {:.1}ms",
            self.qrng_throughput,
            self.lvrf_eval_time.as_secs_f64() * 1000.0,
            self.vdf_compute_time.as_secs_f64() * 1000.0,
            self.queue_throughput,
            self.consensus_latency.as_secs_f64() * 1000.0
        )
    }
}

#[derive(Debug)]
pub struct BenchmarkDetail {
    pub component: String,
    pub metric: String,
    pub value: f64,
    pub unit: String,
}

/// Run comprehensive benchmarks
pub async fn run_benchmarks() -> Result<BenchmarkResults> {
    let mut results = BenchmarkResults::default();
    
    println!("🚀 Running QRNG throughput benchmarks...");
    results.qrng_throughput = benchmark_qrng_throughput().await?;
    results.details.push(BenchmarkDetail {
        component: "QRNG".to_string(),
        metric: "Throughput".to_string(),
        value: results.qrng_throughput,
        unit: "MB/s".to_string(),
    });
    
    println!("🔐 Running L-VRF evaluation benchmarks...");
    results.lvrf_eval_time = benchmark_lvrf_evaluation().await?;
    results.details.push(BenchmarkDetail {
        component: "L-VRF".to_string(),
        metric: "Evaluation Time".to_string(),
        value: results.lvrf_eval_time.as_secs_f64() * 1000.0,
        unit: "ms".to_string(),
    });
    
    println!("⚡ Running quantum VDF benchmarks...");
    results.vdf_compute_time = benchmark_vdf_computation().await?;
    results.details.push(BenchmarkDetail {
        component: "VDF".to_string(),
        metric: "Computation Time".to_string(),
        value: results.vdf_compute_time.as_secs_f64() * 1000.0,
        unit: "ms".to_string(),
    });
    
    println!("⚖️ Running fair queue benchmarks...");
    results.queue_throughput = benchmark_fair_queue_throughput().await?;
    results.details.push(BenchmarkDetail {
        component: "Fair Queue".to_string(),
        metric: "Throughput".to_string(),
        value: results.queue_throughput,
        unit: "tx/s".to_string(),
    });
    
    println!("🎯 Running end-to-end consensus benchmarks...");
    results.consensus_latency = benchmark_consensus_latency().await?;
    results.details.push(BenchmarkDetail {
        component: "Consensus".to_string(),
        metric: "End-to-End Latency".to_string(),
        value: results.consensus_latency.as_secs_f64() * 1000.0,
        unit: "ms".to_string(),
    });
    
    // Evaluate if performance meets targets
    results.passed = 
        results.qrng_throughput > 10.0 &&              // > 10 MB/s
        results.lvrf_eval_time < Duration::from_millis(50) &&  // < 50ms
        results.vdf_compute_time < Duration::from_millis(200) && // < 200ms  
        results.queue_throughput > 1000.0 &&           // > 1000 tx/s
        results.consensus_latency < Duration::from_millis(300); // < 300ms
    
    Ok(results)
}

/// Benchmark QRNG throughput across different providers
async fn benchmark_qrng_throughput() -> Result<f64> {
    let mut total_throughput = 0.0;
    let providers = vec![
        QRNGProvider::Simulation,
        QRNGProvider::ThermalNoise,
        QRNGProvider::OpticalQuantum,
    ];
    
    for provider in providers {
        let mut qrng = QuantumRNG::new(provider).await?;
        
        let chunk_size = 1024 * 1024; // 1MB chunks
        let iterations = 10;
        let start = Instant::now();
        
        for _ in 0..iterations {
            let _data = qrng.generate(chunk_size).await?;
        }
        
        let duration = start.elapsed();
        let total_bytes = chunk_size * iterations;
        let throughput = (total_bytes as f64) / duration.as_secs_f64() / (1024.0 * 1024.0);
        
        total_throughput += throughput;
    }
    
    Ok(total_throughput / providers.len() as f64)
}

/// Benchmark L-VRF evaluation performance
async fn benchmark_lvrf_evaluation() -> Result<Duration> {
    let security_levels = vec![
        SecurityLevel::Low,
        SecurityLevel::Medium, 
        SecurityLevel::High,
        SecurityLevel::Ultra,
    ];
    
    let mut total_time = Duration::ZERO;
    let iterations_per_level = 20;
    
    for level in security_levels {
        let lvrf = LatticeVRF::new(level).await?;
        let input = b"benchmark_input_data";
        
        let start = Instant::now();
        
        for round in 1..=iterations_per_level {
            let _result = lvrf.evaluate(input, Round::new(round)).await?;
        }
        
        total_time += start.elapsed();
    }
    
    Ok(total_time / (security_levels.len() as u32 * iterations_per_level))
}

/// Benchmark VDF computation across protocols
async fn benchmark_vdf_computation() -> Result<Duration> {
    let protocols = vec![
        VDFProtocol::Wesolowski,
        VDFProtocol::Pietrzak,
        VDFProtocol::QuantumHybrid,
    ];
    
    let mut total_time = Duration::ZERO;
    let time_param = 1000; // Moderate difficulty
    let iterations = 5; // VDF is slow, fewer iterations
    
    for protocol in protocols {
        let vdf = QuantumVDF::new(protocol).await?;
        let input = b"vdf_benchmark_input";
        
        let start = Instant::now();
        
        for i in 0..iterations {
            let input_variant = [input, &i.to_le_bytes()].concat();
            let _result = vdf.evaluate(&input_variant, time_param).await?;
        }
        
        total_time += start.elapsed();
    }
    
    Ok(total_time / (protocols.len() as u32 * iterations))
}

/// Benchmark fair queue throughput
async fn benchmark_fair_queue_throughput() -> Result<f64> {
    let policies = vec![
        QueueingPolicy::FIFO,
        QueueingPolicy::VRFBased,
        QueueingPolicy::AntiCensorship,
    ];
    
    let mut total_throughput = 0.0;
    let transactions_count = 10000;
    
    for policy in policies {
        let mut queue = QuantumFairQueue::new(policy).await?;
        
        // Fill queue with transactions
        let start_enqueue = Instant::now();
        for i in 0..transactions_count {
            let tx_id = Uuid::new_v4().into_bytes();
            let node_id = [(i % 256) as u8; 32]; // Distribute across nodes
            queue.enqueue_transaction(tx_id, node_id).await?;
        }
        let enqueue_time = start_enqueue.elapsed();
        
        // Dequeue all transactions
        let start_dequeue = Instant::now();
        while !queue.is_empty().await? {
            let _batch = queue.dequeue_next_batch(100).await?;
        }
        let dequeue_time = start_dequeue.elapsed();
        
        let total_time = enqueue_time + dequeue_time;
        let throughput = transactions_count as f64 / total_time.as_secs_f64();
        total_throughput += throughput;
    }
    
    Ok(total_throughput / policies.len() as f64)
}

/// Benchmark end-to-end consensus latency
async fn benchmark_consensus_latency() -> Result<Duration> {
    // Initialize all components
    let mut qrng = QuantumRNG::new(QRNGProvider::Simulation).await?;
    let lvrf = LatticeVRF::new(SecurityLevel::Medium).await?;
    let vdf = QuantumVDF::new(VDFProtocol::QuantumHybrid).await?;
    let mut queue = QuantumFairQueue::new(QueueingPolicy::VRFBased).await?;
    
    let consensus_rounds = 10;
    let mut total_latency = Duration::ZERO;
    
    for round_num in 1..=consensus_rounds {
        let round = Round::new(round_num);
        let start = Instant::now();
        
        // Step 1: Generate quantum entropy
        let _entropy = qrng.generate(32).await?;
        
        // Step 2: VRF-based operations (anchor selection, etc.)
        let input = format!("consensus_round_{}", round_num);
        let _vrf_result = lvrf.evaluate(input.as_bytes(), round).await?;
        
        // Step 3: Process transactions through fair queue
        let tx_id = Uuid::new_v4().into_bytes();
        queue.enqueue_transaction(tx_id, [round_num as u8; 32]).await?;
        let _batch = queue.dequeue_next_batch(1).await?;
        
        // Step 4: Generate VDF proof (simulated with lower time parameter)
        let _vdf_result = vdf.evaluate(&tx_id, 100).await?;
        
        total_latency += start.elapsed();
    }
    
    Ok(total_latency / consensus_rounds)
}

/// Benchmark parallel processing capabilities
pub async fn benchmark_parallel_performance() -> Result<()> {
    println!("🔄 Running parallel processing benchmarks...");
    
    // Test parallel QRNG generation
    let parallel_qrng_start = Instant::now();
    let qrng_tasks: Vec<_> = (0..10).map(|_| {
        tokio::spawn(async move {
            let mut qrng = QuantumRNG::new(QRNGProvider::Simulation).await.unwrap();
            qrng.generate(1024).await.unwrap()
        })
    }).collect();
    
    let _results = join_all(qrng_tasks).await;
    let parallel_qrng_time = parallel_qrng_start.elapsed();
    
    // Test parallel L-VRF evaluation
    let parallel_lvrf_start = Instant::now();
    let lvrf = LatticeVRF::new(SecurityLevel::Medium).await?;
    let lvrf_tasks: Vec<_> = (0..5).map(|i| {
        let lvrf_clone = lvrf.clone();
        tokio::spawn(async move {
            let input = format!("parallel_test_{}", i);
            lvrf_clone.evaluate(input.as_bytes(), Round::new(i + 1)).await.unwrap()
        })
    }).collect();
    
    let _vrf_results = join_all(lvrf_tasks).await;
    let parallel_lvrf_time = parallel_lvrf_start.elapsed();
    
    println!("✅ Parallel QRNG (10 tasks): {:.2}ms", parallel_qrng_time.as_secs_f64() * 1000.0);
    println!("✅ Parallel L-VRF (5 tasks): {:.2}ms", parallel_lvrf_time.as_secs_f64() * 1000.0);
    
    Ok(())
}

/// Memory usage benchmarks
pub async fn benchmark_memory_usage() -> Result<()> {
    println!("💾 Running memory usage benchmarks...");
    
    // Benchmark component memory footprints
    let initial_memory = get_memory_usage()?;
    
    // QRNG memory usage
    let _qrng = QuantumRNG::new(QRNGProvider::Simulation).await?;
    let qrng_memory = get_memory_usage()? - initial_memory;
    
    // L-VRF memory usage  
    let _lvrf = LatticeVRF::new(SecurityLevel::High).await?;
    let lvrf_memory = get_memory_usage()? - initial_memory - qrng_memory;
    
    // VDF memory usage
    let _vdf = QuantumVDF::new(VDFProtocol::QuantumHybrid).await?;
    let vdf_memory = get_memory_usage()? - initial_memory - qrng_memory - lvrf_memory;
    
    println!("📊 Memory Usage:");
    println!("  QRNG: {:.1} MB", qrng_memory as f64 / (1024.0 * 1024.0));
    println!("  L-VRF: {:.1} MB", lvrf_memory as f64 / (1024.0 * 1024.0));
    println!("  VDF: {:.1} MB", vdf_memory as f64 / (1024.0 * 1024.0));
    
    Ok(())
}

/// Get current memory usage (simplified implementation)
fn get_memory_usage() -> Result<usize> {
    // In a real implementation, this would use system APIs
    // For now, return a placeholder
    Ok(0)
}

/// Stress test all components under load
pub async fn stress_test_components() -> Result<()> {
    println!("💪 Running stress tests...");
    
    let stress_duration = Duration::from_secs(30);
    let end_time = Instant::now() + stress_duration;
    
    let mut operations_count = 0u64;
    
    // Initialize components
    let mut qrng = QuantumRNG::new(QRNGProvider::Simulation).await?;
    let lvrf = LatticeVRF::new(SecurityLevel::Medium).await?;
    let mut queue = QuantumFairQueue::new(QueueingPolicy::VRFBased).await?;
    
    while Instant::now() < end_time {
        // Rapid fire operations
        let _entropy = qrng.generate(64).await?;
        
        let tx_id = Uuid::new_v4().into_bytes();
        queue.enqueue_transaction(tx_id, [operations_count as u8; 32]).await?;
        
        if operations_count % 10 == 0 {
            let _vrf_result = lvrf.evaluate(&tx_id, Round::new((operations_count % 1000) + 1)).await?;
        }
        
        if operations_count % 5 == 0 {
            let _batch = queue.dequeue_next_batch(1).await?;
        }
        
        operations_count += 1;
    }
    
    let ops_per_sec = operations_count as f64 / stress_duration.as_secs_f64();
    println!("✅ Stress test completed: {:.0} ops/sec", ops_per_sec);
    
    // Verify components still work correctly after stress
    let final_entropy = qrng.generate(32).await?;
    if final_entropy.len() != 32 {
        return Err(anyhow::anyhow!("QRNG failed after stress test"));
    }
    
    let final_vrf = lvrf.evaluate(b"stress_test_final", Round::new(1)).await?;
    let vrf_valid = lvrf.verify(b"stress_test_final", Round::new(1), &final_vrf.output, &final_vrf.proof).await?;
    if !vrf_valid {
        return Err(anyhow::anyhow!("L-VRF failed after stress test"));
    }
    
    Ok(())
}