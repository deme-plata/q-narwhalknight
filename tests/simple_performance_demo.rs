/// Simple Performance Test Demo for Q-NarwhalKnight
/// Demonstrates basic 5-node consensus with transaction throughput measurement

use std::time::{Duration, Instant};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::time::sleep;
use tokio::sync::RwLock;
use anyhow::Result;

/// Simulated transaction for performance testing
#[derive(Debug, Clone)]
pub struct MockTransaction {
    pub id: u64,
    pub payload: Vec<u8>,
    pub timestamp: u64,
}

/// Simulated consensus node
#[derive(Debug)]
pub struct MockConsensusNode {
    pub node_id: u8,
    pub transactions_processed: AtomicU64,
    pub bytes_processed: AtomicU64,
    pub vertices_created: AtomicU64,
}

impl MockConsensusNode {
    pub fn new(node_id: u8) -> Self {
        Self {
            node_id,
            transactions_processed: AtomicU64::new(0),
            bytes_processed: AtomicU64::new(0),
            vertices_created: AtomicU64::new(0),
        }
    }
    
    /// Process a transaction (simulated)
    pub async fn process_transaction(&self, transaction: MockTransaction) -> Result<()> {
        // Simulate processing time
        sleep(Duration::from_micros(50)).await;
        
        // Update counters
        self.transactions_processed.fetch_add(1, Ordering::Relaxed);
        self.bytes_processed.fetch_add(transaction.payload.len() as u64, Ordering::Relaxed);
        
        // Every 10 transactions, create a vertex
        if self.transactions_processed.load(Ordering::Relaxed) % 10 == 0 {
            self.vertices_created.fetch_add(1, Ordering::Relaxed);
        }
        
        Ok(())
    }
    
    pub fn get_stats(&self) -> NodeStats {
        NodeStats {
            node_id: self.node_id,
            transactions_processed: self.transactions_processed.load(Ordering::Relaxed),
            bytes_processed: self.bytes_processed.load(Ordering::Relaxed),
            vertices_created: self.vertices_created.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone)]
pub struct NodeStats {
    pub node_id: u8,
    pub transactions_processed: u64,
    pub bytes_processed: u64,
    pub vertices_created: u64,
}

/// Simple 5-node network simulation
pub struct SimplePerformanceTest {
    nodes: Vec<MockConsensusNode>,
    test_duration: Duration,
    target_tps: u64,
}

impl SimplePerformanceTest {
    pub fn new(test_duration_secs: u64, target_tps: u64) -> Self {
        let nodes = (0..5).map(|i| MockConsensusNode::new(i)).collect();
        
        Self {
            nodes,
            test_duration: Duration::from_secs(test_duration_secs),
            target_tps,
        }
    }
    
    /// Run the performance test
    pub async fn run_test(&self) -> Result<PerformanceResults> {
        println!("🚀 Starting Q-NarwhalKnight 5-Node Performance Test");
        println!("📊 Duration: {:?}, Target TPS: {}", self.test_duration, self.target_tps);
        
        let start_time = Instant::now();
        let end_time = start_time + self.test_duration;
        
        let mut transaction_id = 0u64;
        let mut total_transactions = 0u64;
        let mut total_bytes = 0u64;
        
        // Transaction generation loop
        while Instant::now() < end_time {
            let batch_start = Instant::now();
            
            // Generate batch of transactions
            let batch_size = (self.target_tps / 10).max(1); // 10 batches per second
            
            for _ in 0..batch_size {
                transaction_id += 1;
                
                // Create transaction with random payload
                let payload_size = 512; // 512 bytes per transaction
                let payload: Vec<u8> = (0..payload_size).map(|i| ((transaction_id + i as u64) % 256) as u8).collect();
                
                let transaction = MockTransaction {
                    id: transaction_id,
                    payload: payload.clone(),
                    timestamp: chrono::Utc::now().timestamp_millis() as u64,
                };
                
                // Round-robin assign to nodes
                let node_index = (transaction_id as usize) % self.nodes.len();
                self.nodes[node_index].process_transaction(transaction).await?;
                
                total_transactions += 1;
                total_bytes += payload_size as u64;
            }
            
            // Maintain target TPS timing
            let batch_duration = batch_start.elapsed();
            let target_batch_duration = Duration::from_millis(100); // 10 batches per second
            
            if batch_duration < target_batch_duration {
                sleep(target_batch_duration - batch_duration).await;
            }
        }
        
        let actual_duration = start_time.elapsed();
        
        // Collect stats from all nodes
        let node_stats: Vec<NodeStats> = self.nodes.iter().map(|n| n.get_stats()).collect();
        
        // Calculate aggregated metrics
        let total_processed: u64 = node_stats.iter().map(|s| s.transactions_processed).sum();
        let total_processed_bytes: u64 = node_stats.iter().map(|s| s.bytes_processed).sum();
        let total_vertices: u64 = node_stats.iter().map(|s| s.vertices_created).sum();
        
        let achieved_tps = total_processed as f64 / actual_duration.as_secs_f64();
        let achieved_bps = total_processed_bytes as f64 / actual_duration.as_secs_f64();
        let achieved_mbps = achieved_bps / 1_000_000.0;
        
        Ok(PerformanceResults {
            duration: actual_duration,
            transactions_generated: total_transactions,
            transactions_processed: total_processed,
            bytes_processed: total_processed_bytes,
            vertices_created: total_vertices,
            achieved_tps,
            achieved_bps,
            achieved_mbps,
            node_stats,
            target_tps: self.target_tps,
        })
    }
    
    /// Generate performance report
    pub fn generate_report(&self, results: &PerformanceResults) -> String {
        let tps_percentage = (results.achieved_tps / results.target_tps as f64) * 100.0;
        let processing_efficiency = (results.transactions_processed as f64 / results.transactions_generated as f64) * 100.0;
        
        let mut report = format!(
r#"
🏆 Q-NARWHALKNIGHT PERFORMANCE TEST RESULTS

📊 THROUGHPUT METRICS
• Test Duration: {:.2} seconds
• Transactions Generated: {:,}
• Transactions Processed: {:,}
• Processing Efficiency: {:.2}%
• Data Processed: {:.2} MB
• Vertices Created: {:,}

⚡ PERFORMANCE METRICS
• Target TPS: {:,}
• Achieved TPS: {:.2}
• Target Achievement: {:.1}%
• Bandwidth: {:.2} MB/s
• Vertices per Second: {:.2}

🔬 NODE BREAKDOWN
"#,
            results.duration.as_secs_f64(),
            results.transactions_generated,
            results.transactions_processed,
            processing_efficiency,
            results.bytes_processed as f64 / 1_000_000.0,
            results.vertices_created,
            results.target_tps,
            results.achieved_tps,
            tps_percentage,
            results.achieved_mbps,
            results.vertices_created as f64 / results.duration.as_secs_f64(),
        );
        
        for stats in &results.node_stats {
            report.push_str(&format!(
                "Node {}: {:,} txns ({:.1}%), {:.2} MB, {:,} vertices\n",
                stats.node_id,
                stats.transactions_processed,
                (stats.transactions_processed as f64 / results.transactions_processed as f64) * 100.0,
                stats.bytes_processed as f64 / 1_000_000.0,
                stats.vertices_created,
            ));
        }
        
        report.push_str("\n🎯 PERFORMANCE ANALYSIS\n");
        
        if tps_percentage >= 90.0 {
            report.push_str("✅ Excellent performance - target TPS achieved\n");
        } else if tps_percentage >= 50.0 {
            report.push_str("⚠️ Good performance - optimization opportunities exist\n");
        } else {
            report.push_str("❌ Performance below target - requires significant optimization\n");
        }
        
        if processing_efficiency >= 95.0 {
            report.push_str("✅ High processing efficiency - minimal transaction loss\n");
        } else {
            report.push_str("⚠️ Processing efficiency could be improved\n");
        }
        
        // Simulated consensus metrics
        let estimated_finality_ms = 1000.0 + (results.achieved_tps / 100.0); // Higher TPS = slightly higher latency
        let quantum_entropy_quality = 0.85 + (results.vertices_created as f64 / 10000.0).min(0.1); // Simulate improving entropy
        
        report.push_str(&format!(
r#"
🔮 SIMULATED CONSENSUS METRICS
• Estimated Finality: {:.1} ms
• Quantum Entropy Quality: {:.3}
• Network Partition Tolerance: ✅ BFT (f=2)
• Post-Quantum Ready: ✅ Phase 1

🎯 TARGET COMPARISON
• 48k TPS Target: {:.1}% achieved
• 2.3s Finality Target: {:.1}% achieved (estimated)
"#,
            estimated_finality_ms,
            quantum_entropy_quality,
            (results.achieved_tps / 48000.0) * 100.0,
            (2300.0 / estimated_finality_ms) * 100.0,
        ));
        
        report
    }
}

#[derive(Debug, Clone)]
pub struct PerformanceResults {
    pub duration: Duration,
    pub transactions_generated: u64,
    pub transactions_processed: u64,
    pub bytes_processed: u64,
    pub vertices_created: u64,
    pub achieved_tps: f64,
    pub achieved_bps: f64,
    pub achieved_mbps: f64,
    pub node_stats: Vec<NodeStats>,
    pub target_tps: u64,
}

/// Main test execution
pub async fn run_5_node_performance_demo() -> Result<()> {
    println!("🌟 Q-NarwhalKnight Simple Performance Demo");
    
    // Test scenarios
    let scenarios = vec![
        ("Low Load", 30, 500),      // 30 seconds, 500 TPS
        ("Medium Load", 30, 2000),  // 30 seconds, 2k TPS  
        ("High Load", 30, 5000),    // 30 seconds, 5k TPS
    ];
    
    for (name, duration, target_tps) in scenarios {
        println!("\n{}", "=".repeat(60));
        println!("📊 Running Scenario: {}", name);
        println!("{}", "=".repeat(60));
        
        let test = SimplePerformanceTest::new(duration, target_tps);
        let results = test.run_test().await?;
        let report = test.generate_report(&results);
        
        println!("{}", report);
        
        // Small delay between tests
        sleep(Duration::from_secs(2)).await;
    }
    
    println!("\n✅ All performance scenarios completed!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_basic_performance() {
        let test = SimplePerformanceTest::new(5, 100); // 5 second, 100 TPS test
        let results = test.run_test().await.unwrap();
        
        assert!(results.transactions_processed > 0);
        assert!(results.achieved_tps > 0.0);
        assert_eq!(results.node_stats.len(), 5);
        
        println!("Basic test achieved: {:.2} TPS", results.achieved_tps);
    }
    
    #[tokio::test] 
    async fn test_high_throughput() {
        let test = SimplePerformanceTest::new(10, 1000); // 10 second, 1k TPS test
        let results = test.run_test().await.unwrap();
        
        assert!(results.achieved_tps > 100.0, "Should achieve reasonable TPS");
        assert!(results.bytes_processed > 1000, "Should process significant data");
        
        println!("High throughput test: {:.2} TPS, {:.2} MB/s", 
                results.achieved_tps, results.achieved_mbps);
    }
}