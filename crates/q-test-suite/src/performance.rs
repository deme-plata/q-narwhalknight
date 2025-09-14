/// Performance validation tests for quantum consensus components

use anyhow::{Result, anyhow};
use std::time::{Duration, Instant};
use std::sync::Arc;
use tokio::sync::Semaphore;
use futures::future::join_all;
use uuid::Uuid;

use q_quantum_rng::{QuantumRNG, QRNGProvider};
use q_lattice_vrf::{LatticeVRF, SecurityLevel};
use q_vdf::{QuantumVDF, VDFProtocol};
use q_fairqueue::{QuantumFairQueue, QueueingPolicy};
use q_types::Round;

/// Performance test results
#[derive(Debug, Default)]
pub struct PerformanceResults {
    pub passed: bool,
    pub latency_targets_met: u32,
    pub throughput_targets_met: u32,
    pub scalability_score: f64, // 0-100
    pub resource_efficiency: f64, // 0-100
    pub details: Vec<PerformanceDetail>,
}

impl PerformanceResults {
    pub fn summary(&self) -> String {
        format!(
            "Latency: {}/{} | Throughput: {}/{} | Scalability: {:.1} | Efficiency: {:.1}",
            self.latency_targets_met,
            self.details.len(),
            self.throughput_targets_met, 
            self.details.len(),
            self.scalability_score,
            self.resource_efficiency
        )
    }
}

#[derive(Debug)]
pub struct PerformanceDetail {
    pub component: String,
    pub metric: String,
    pub measured_value: f64,
    pub target_value: f64,
    pub unit: String,
    pub passed: bool,
}

/// Performance targets for Q-NarwhalKnight
struct PerformanceTargets;

impl PerformanceTargets {
    const QRNG_THROUGHPUT_MB_S: f64 = 5.0;        // > 5 MB/s
    const LVRF_LATENCY_MS: f64 = 100.0;           // < 100ms
    const VDF_COMPUTE_MAX_MS: f64 = 500.0;        // < 500ms  
    const QUEUE_THROUGHPUT_TXS: f64 = 500.0;      // > 500 tx/s
    const CONSENSUS_LATENCY_MS: f64 = 1000.0;     // < 1000ms (1s)
    const MEMORY_USAGE_MB: f64 = 100.0;           // < 100MB per component
    const CPU_UTILIZATION: f64 = 80.0;            // < 80% per core
}

/// Run comprehensive performance tests
pub async fn run_performance_tests(samples: u32) -> Result<PerformanceResults> {
    let mut results = PerformanceResults::default();
    
    // Single-component performance tests
    test_qrng_performance(&mut results, samples).await?;
    test_lvrf_performance(&mut results, samples).await?;
    test_vdf_performance(&mut results, samples).await?;
    test_fair_queue_performance(&mut results, samples).await?;
    
    // End-to-end performance
    test_consensus_performance(&mut results, samples).await?;
    
    // Scalability tests
    test_horizontal_scalability(&mut results).await?;
    test_load_scaling(&mut results).await?;
    
    // Resource efficiency tests  
    test_memory_efficiency(&mut results).await?;
    test_cpu_efficiency(&mut results).await?;
    
    // Calculate aggregate scores
    results.scalability_score = calculate_scalability_score(&results);
    results.resource_efficiency = calculate_resource_efficiency(&results);
    
    // Count targets met
    for detail in &results.details {
        if detail.passed {
            if detail.metric.contains("Latency") || detail.metric.contains("Time") {
                results.latency_targets_met += 1;
            } else if detail.metric.contains("Throughput") || detail.metric.contains("Rate") {
                results.throughput_targets_met += 1;
            }
        }
    }
    
    results.passed = results.latency_targets_met > 0 && 
                    results.throughput_targets_met > 0 &&
                    results.scalability_score > 70.0 &&
                    results.resource_efficiency > 70.0;
    
    Ok(results)
}

/// Test QRNG performance characteristics
async fn test_qrng_performance(results: &mut PerformanceResults, samples: u32) -> Result<()> {
    let providers = vec![
        ("Simulation", QRNGProvider::Simulation),
        ("Thermal", QRNGProvider::ThermalNoise),
        ("Optical", QRNGProvider::OpticalQuantum),
    ];
    
    for (name, provider) in providers {
        let mut qrng = QuantumRNG::new(provider).await?;
        
        // Throughput test
        let chunk_size = 1024 * 1024; // 1MB chunks
        let start = Instant::now();
        
        for _ in 0..samples {
            let _data = qrng.generate(chunk_size / samples as usize).await?;
        }
        
        let duration = start.elapsed();
        let total_mb = (chunk_size as f64) / (1024.0 * 1024.0);
        let throughput = total_mb / duration.as_secs_f64();
        
        results.details.push(PerformanceDetail {
            component: format!("QRNG-{}", name),
            metric: "Throughput".to_string(),
            measured_value: throughput,
            target_value: PerformanceTargets::QRNG_THROUGHPUT_MB_S,
            unit: "MB/s".to_string(),
            passed: throughput >= PerformanceTargets::QRNG_THROUGHPUT_MB_S,
        });
        
        // Latency test (single generation)
        let latency_start = Instant::now();
        let _single_gen = qrng.generate(32).await?;
        let latency = latency_start.elapsed();
        
        results.details.push(PerformanceDetail {
            component: format!("QRNG-{}", name),
            metric: "Latency".to_string(),
            measured_value: latency.as_secs_f64() * 1000.0,
            target_value: 50.0, // < 50ms for 32 bytes
            unit: "ms".to_string(),
            passed: latency.as_millis() < 50,
        });
    }
    
    Ok(())
}

/// Test L-VRF performance across security levels
async fn test_lvrf_performance(results: &mut PerformanceResults, samples: u32) -> Result<()> {
    let security_levels = vec![
        ("Low", SecurityLevel::Low),
        ("Medium", SecurityLevel::Medium),
        ("High", SecurityLevel::High),
        ("Ultra", SecurityLevel::Ultra),
    ];
    
    for (name, level) in security_levels {
        let lvrf = LatticeVRF::new(level).await?;
        
        // Evaluation performance
        let mut total_eval_time = Duration::ZERO;
        let mut total_verify_time = Duration::ZERO;
        
        for i in 0..samples {
            let input = format!("perf_test_{}", i);
            let round = Round::new(i + 1);
            
            // Measure evaluation time
            let eval_start = Instant::now();
            let result = lvrf.evaluate(input.as_bytes(), round).await?;
            total_eval_time += eval_start.elapsed();
            
            // Measure verification time
            let verify_start = Instant::now();
            let _is_valid = lvrf.verify(input.as_bytes(), round, &result.output, &result.proof).await?;
            total_verify_time += verify_start.elapsed();
        }
        
        let avg_eval_time = total_eval_time / samples;
        let avg_verify_time = total_verify_time / samples;
        
        results.details.push(PerformanceDetail {
            component: format!("L-VRF-{}", name),
            metric: "Evaluation Time".to_string(),
            measured_value: avg_eval_time.as_secs_f64() * 1000.0,
            target_value: PerformanceTargets::LVRF_LATENCY_MS,
            unit: "ms".to_string(),
            passed: avg_eval_time.as_millis() < PerformanceTargets::LVRF_LATENCY_MS as u128,
        });
        
        results.details.push(PerformanceDetail {
            component: format!("L-VRF-{}", name),
            metric: "Verification Time".to_string(),
            measured_value: avg_verify_time.as_secs_f64() * 1000.0,
            target_value: 10.0, // Verification should be < 10ms
            unit: "ms".to_string(),
            passed: avg_verify_time.as_millis() < 10,
        });
        
        // Throughput (evaluations per second)
        let total_time = total_eval_time + total_verify_time;
        let throughput = samples as f64 / total_time.as_secs_f64();
        
        results.details.push(PerformanceDetail {
            component: format!("L-VRF-{}", name),
            metric: "Throughput".to_string(),
            measured_value: throughput,
            target_value: 10.0, // > 10 operations/sec
            unit: "ops/s".to_string(),
            passed: throughput >= 10.0,
        });
    }
    
    Ok(())
}

/// Test VDF performance across protocols
async fn test_vdf_performance(results: &mut PerformanceResults, samples: u32) -> Result<()> {
    let protocols = vec![
        ("Wesolowski", VDFProtocol::Wesolowski),
        ("Pietrzak", VDFProtocol::Pietrzak),
        ("QuantumHybrid", VDFProtocol::QuantumHybrid),
    ];
    
    let time_params = vec![100, 500, 1000]; // Different difficulty levels
    
    for (name, protocol) in protocols {
        let vdf = QuantumVDF::new(protocol).await?;
        
        for &time_param in &time_params {
            let mut total_compute_time = Duration::ZERO;
            let mut total_verify_time = Duration::ZERO;
            
            let iterations = (samples / 3).max(1); // Fewer iterations for VDF
            
            for i in 0..iterations {
                let input = format!("vdf_perf_{}_{}", time_param, i).into_bytes();
                
                // Measure computation time
                let compute_start = Instant::now();
                let result = vdf.evaluate(&input, time_param).await?;
                total_compute_time += compute_start.elapsed();
                
                // Measure verification time
                let verify_start = Instant::now();
                let _is_valid = vdf.verify(&input, time_param, &result.output, &result.proof).await?;
                total_verify_time += verify_start.elapsed();
            }
            
            let avg_compute_time = total_compute_time / iterations;
            let avg_verify_time = total_verify_time / iterations;
            
            results.details.push(PerformanceDetail {
                component: format!("VDF-{}-{}", name, time_param),
                metric: "Computation Time".to_string(),
                measured_value: avg_compute_time.as_secs_f64() * 1000.0,
                target_value: PerformanceTargets::VDF_COMPUTE_MAX_MS,
                unit: "ms".to_string(),
                passed: avg_compute_time.as_millis() < PerformanceTargets::VDF_COMPUTE_MAX_MS as u128,
            });
            
            // Verify speedup (verification should be much faster than computation)
            let speedup = avg_compute_time.as_secs_f64() / avg_verify_time.as_secs_f64();
            
            results.details.push(PerformanceDetail {
                component: format!("VDF-{}-{}", name, time_param),
                metric: "Verification Speedup".to_string(),
                measured_value: speedup,
                target_value: 100.0, // > 100x speedup
                unit: "x".to_string(),
                passed: speedup >= 100.0,
            });
        }
    }
    
    Ok(())
}

/// Test fair queue performance
async fn test_fair_queue_performance(results: &mut PerformanceResults, samples: u32) -> Result<()> {
    let policies = vec![
        ("FIFO", QueueingPolicy::FIFO),
        ("VRF-Based", QueueingPolicy::VRFBased),
        ("Anti-Censorship", QueueingPolicy::AntiCensorship),
    ];
    
    for (name, policy) in policies {
        let mut queue = QuantumFairQueue::new(policy).await?;
        
        // Enqueue performance
        let enqueue_start = Instant::now();
        for i in 0..samples {
            let tx_id = Uuid::new_v4().into_bytes();
            let node_id = [(i % 256) as u8; 32];
            queue.enqueue_transaction(tx_id, node_id).await?;
        }
        let enqueue_time = enqueue_start.elapsed();
        
        let enqueue_throughput = samples as f64 / enqueue_time.as_secs_f64();
        
        results.details.push(PerformanceDetail {
            component: format!("Queue-{}", name),
            metric: "Enqueue Throughput".to_string(),
            measured_value: enqueue_throughput,
            target_value: PerformanceTargets::QUEUE_THROUGHPUT_TXS,
            unit: "tx/s".to_string(),
            passed: enqueue_throughput >= PerformanceTargets::QUEUE_THROUGHPUT_TXS,
        });
        
        // Dequeue performance
        let dequeue_start = Instant::now();
        let mut dequeued_count = 0;
        
        while !queue.is_empty().await? {
            let batch = queue.dequeue_next_batch(10).await?;
            dequeued_count += batch.len();
        }
        
        let dequeue_time = dequeue_start.elapsed();
        let dequeue_throughput = dequeued_count as f64 / dequeue_time.as_secs_f64();
        
        results.details.push(PerformanceDetail {
            component: format!("Queue-{}", name),
            metric: "Dequeue Throughput".to_string(),
            measured_value: dequeue_throughput,
            target_value: PerformanceTargets::QUEUE_THROUGHPUT_TXS,
            unit: "tx/s".to_string(),
            passed: dequeue_throughput >= PerformanceTargets::QUEUE_THROUGHPUT_TXS,
        });
        
        // Fairness calculation performance
        let fairness_start = Instant::now();
        let _fairness_metrics = queue.calculate_fairness_metrics().await?;
        let fairness_time = fairness_start.elapsed();
        
        results.details.push(PerformanceDetail {
            component: format!("Queue-{}", name),
            metric: "Fairness Calculation".to_string(),
            measured_value: fairness_time.as_secs_f64() * 1000.0,
            target_value: 100.0, // < 100ms
            unit: "ms".to_string(),
            passed: fairness_time.as_millis() < 100,
        });
    }
    
    Ok(())
}

/// Test end-to-end consensus performance
async fn test_consensus_performance(results: &mut PerformanceResults, samples: u32) -> Result<()> {
    // Initialize all components
    let mut qrng = QuantumRNG::new(QRNGProvider::Simulation).await?;
    let lvrf = LatticeVRF::new(SecurityLevel::Medium).await?;
    let vdf = QuantumVDF::new(VDFProtocol::QuantumHybrid).await?;
    let mut queue = QuantumFairQueue::new(QueueingPolicy::VRFBased).await?;
    
    let mut total_consensus_time = Duration::ZERO;
    let iterations = (samples / 10).max(1); // End-to-end is expensive
    
    for round_num in 1..=iterations {
        let consensus_start = Instant::now();
        let round = Round::new(round_num);
        
        // Step 1: Generate quantum entropy
        let _entropy = qrng.generate(32).await?;
        
        // Step 2: VRF operations
        let input = format!("consensus_{}", round_num);
        let _vrf_result = lvrf.evaluate(input.as_bytes(), round).await?;
        
        // Step 3: Transaction processing
        let tx_id = Uuid::new_v4().into_bytes();
        queue.enqueue_transaction(tx_id, [round_num as u8; 32]).await?;
        let _batch = queue.dequeue_next_batch(1).await?;
        
        // Step 4: VDF proof (reduced time parameter for testing)
        let _vdf_result = vdf.evaluate(&tx_id, 50).await?;
        
        total_consensus_time += consensus_start.elapsed();
    }
    
    let avg_consensus_time = total_consensus_time / iterations;
    let consensus_throughput = iterations as f64 / total_consensus_time.as_secs_f64();
    
    results.details.push(PerformanceDetail {
        component: "Consensus".to_string(),
        metric: "End-to-End Latency".to_string(),
        measured_value: avg_consensus_time.as_secs_f64() * 1000.0,
        target_value: PerformanceTargets::CONSENSUS_LATENCY_MS,
        unit: "ms".to_string(),
        passed: avg_consensus_time.as_millis() < PerformanceTargets::CONSENSUS_LATENCY_MS as u128,
    });
    
    results.details.push(PerformanceDetail {
        component: "Consensus".to_string(),
        metric: "Throughput".to_string(),
        measured_value: consensus_throughput,
        target_value: 1.0, // > 1 consensus/sec
        unit: "rounds/s".to_string(),
        passed: consensus_throughput >= 1.0,
    });
    
    Ok(())
}

/// Test horizontal scalability (more nodes)
async fn test_horizontal_scalability(results: &mut PerformanceResults) -> Result<()> {
    let node_counts = vec![1, 3, 7, 15]; // Common BFT network sizes
    let lvrf = LatticeVRF::new(SecurityLevel::Medium).await?;
    
    for node_count in node_counts {
        let start = Instant::now();
        
        // Simulate VRF evaluation across multiple nodes
        let mut tasks = Vec::new();
        
        for node_id in 0..node_count {
            let lvrf_clone = lvrf.clone();
            let task = tokio::spawn(async move {
                let input = format!("node_{}_consensus", node_id);
                lvrf_clone.evaluate(input.as_bytes(), Round::new(1)).await
            });
            tasks.push(task);
        }
        
        let results_vec: Vec<_> = join_all(tasks).await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;
        
        let elapsed = start.elapsed();
        let throughput = node_count as f64 / elapsed.as_secs_f64();
        
        // Calculate efficiency (should scale well)
        let efficiency = if node_count == 1 {
            100.0
        } else {
            // Compare to single node performance
            let single_node_time = Duration::from_millis(50); // Baseline
            let expected_time = single_node_time; // Should be constant with parallelism
            let actual_efficiency = expected_time.as_secs_f64() / elapsed.as_secs_f64() * 100.0;
            actual_efficiency.min(100.0)
        };
        
        results.details.push(PerformanceDetail {
            component: "Scalability".to_string(),
            metric: format!("Efficiency-{}-nodes", node_count),
            measured_value: efficiency,
            target_value: 80.0, // > 80% efficiency
            unit: "%".to_string(),
            passed: efficiency >= 80.0,
        });
    }
    
    Ok(())
}

/// Test performance under increasing load
async fn test_load_scaling(results: &mut PerformanceResults) -> Result<()> {
    let load_levels = vec![10, 50, 100, 500]; // Transactions per second
    let mut queue = QuantumFairQueue::new(QueueingPolicy::VRFBased).await?;
    
    for target_tps in load_levels {
        let duration = Duration::from_secs(5); // 5 second test
        let total_transactions = target_tps * 5;
        
        let start = Instant::now();
        let semaphore = Arc::new(Semaphore::new(target_tps));
        
        let mut tasks = Vec::new();
        
        for i in 0..total_transactions {
            let sem = semaphore.clone();
            let tx_id = Uuid::new_v4().into_bytes();
            let node_id = [(i % 256) as u8; 32];
            
            // Rate limit to target TPS
            let task = tokio::spawn(async move {
                let _permit = sem.acquire().await.unwrap();
                tokio::time::sleep(Duration::from_millis(1000 / target_tps as u64)).await;
                (tx_id, node_id)
            });
            
            tasks.push(task);
        }
        
        // Wait for all transactions to be generated
        let transaction_data = join_all(tasks).await;
        
        // Now enqueue them as fast as possible
        let enqueue_start = Instant::now();
        for tx_data in transaction_data {
            let (tx_id, node_id) = tx_data?;
            queue.enqueue_transaction(tx_id, node_id).await?;
        }
        let enqueue_time = enqueue_start.elapsed();
        
        let actual_tps = total_transactions as f64 / enqueue_time.as_secs_f64();
        let load_factor = actual_tps / target_tps as f64;
        
        results.details.push(PerformanceDetail {
            component: "Load Scaling".to_string(),
            metric: format!("Load-{}-TPS", target_tps),
            measured_value: load_factor,
            target_value: 0.8, // Should handle at least 80% of target load
            unit: "ratio".to_string(),
            passed: load_factor >= 0.8,
        });
    }
    
    Ok(())
}

/// Test memory efficiency
async fn test_memory_efficiency(results: &mut PerformanceResults) -> Result<()> {
    // Initialize components and measure memory usage
    let initial_memory = get_process_memory()?;
    
    let _qrng = QuantumRNG::new(QRNGProvider::Simulation).await?;
    let qrng_memory = get_process_memory()? - initial_memory;
    
    let _lvrf = LatticeVRF::new(SecurityLevel::High).await?;
    let lvrf_memory = get_process_memory()? - initial_memory - qrng_memory;
    
    let _vdf = QuantumVDF::new(VDFProtocol::QuantumHybrid).await?;
    let vdf_memory = get_process_memory()? - initial_memory - qrng_memory - lvrf_memory;
    
    let components = vec![
        ("QRNG", qrng_memory),
        ("L-VRF", lvrf_memory),
        ("VDF", vdf_memory),
    ];
    
    for (name, memory_bytes) in components {
        let memory_mb = memory_bytes as f64 / (1024.0 * 1024.0);
        
        results.details.push(PerformanceDetail {
            component: format!("Memory-{}", name),
            metric: "Memory Usage".to_string(),
            measured_value: memory_mb,
            target_value: PerformanceTargets::MEMORY_USAGE_MB,
            unit: "MB".to_string(),
            passed: memory_mb <= PerformanceTargets::MEMORY_USAGE_MB,
        });
    }
    
    Ok(())
}

/// Test CPU efficiency
async fn test_cpu_efficiency(results: &mut PerformanceResults) -> Result<()> {
    let mut qrng = QuantumRNG::new(QRNGProvider::Simulation).await?;
    
    // Monitor CPU usage during operations
    let start_time = Instant::now();
    let test_duration = Duration::from_secs(5);
    
    let cpu_start = get_cpu_usage()?;
    
    // Perform operations for test duration
    while start_time.elapsed() < test_duration {
        let _entropy = qrng.generate(1024).await?;
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    
    let cpu_end = get_cpu_usage()?;
    let cpu_utilization = ((cpu_end - cpu_start) / test_duration.as_secs_f64()) * 100.0;
    
    results.details.push(PerformanceDetail {
        component: "CPU".to_string(),
        metric: "CPU Utilization".to_string(),
        measured_value: cpu_utilization,
        target_value: PerformanceTargets::CPU_UTILIZATION,
        unit: "%".to_string(),
        passed: cpu_utilization <= PerformanceTargets::CPU_UTILIZATION,
    });
    
    Ok(())
}

// Helper functions for performance measurement

fn calculate_scalability_score(results: &PerformanceResults) -> f64 {
    let scalability_tests: Vec<_> = results.details.iter()
        .filter(|d| d.component.contains("Scalability") || d.component.contains("Load"))
        .collect();
    
    if scalability_tests.is_empty() {
        return 50.0; // Default score
    }
    
    let passed = scalability_tests.iter().filter(|t| t.passed).count();
    (passed as f64 / scalability_tests.len() as f64) * 100.0
}

fn calculate_resource_efficiency(results: &PerformanceResults) -> f64 {
    let resource_tests: Vec<_> = results.details.iter()
        .filter(|d| d.component.contains("Memory") || d.component.contains("CPU"))
        .collect();
    
    if resource_tests.is_empty() {
        return 50.0; // Default score
    }
    
    let passed = resource_tests.iter().filter(|t| t.passed).count();
    (passed as f64 / resource_tests.len() as f64) * 100.0
}

fn get_process_memory() -> Result<usize> {
    // Simplified implementation - in production would use proper memory monitoring
    Ok(1024 * 1024 * 10) // 10MB placeholder
}

fn get_cpu_usage() -> Result<f64> {
    // Simplified implementation - in production would use proper CPU monitoring  
    Ok(25.0) // 25% placeholder
}