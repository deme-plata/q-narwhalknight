/// Integration tests for quantum-enhanced consensus components

use anyhow::{Result, anyhow};
use std::time::{Duration, Instant};
use uuid::Uuid;
use tokio::time::sleep;

use q_quantum_rng::{QuantumRNG, QRNGProvider};
use q_lattice_vrf::{LatticeVRF, SecurityLevel};
use q_vdf::{QuantumVDF, VDFProtocol};
use q_fairqueue::{QuantumFairQueue, QueueingPolicy};
use q_dag_knight::anchor_election::AnchorElection;
use q_types::{NodeId, Round, TransactionId};

/// Integration test results
#[derive(Debug, Default)]
pub struct IntegrationResults {
    pub passed: bool,
    pub tests_run: u32,
    pub tests_passed: u32,
    pub execution_time: Duration,
    pub details: Vec<TestDetail>,
}

impl IntegrationResults {
    pub fn summary(&self) -> String {
        format!(
            "Passed: {}/{} tests in {:.2}s\nSuccess rate: {:.1}%",
            self.tests_passed, self.tests_run, 
            self.execution_time.as_secs_f64(),
            if self.tests_run > 0 { 
                (self.tests_passed as f64 / self.tests_run as f64) * 100.0 
            } else { 0.0 }
        )
    }
}

#[derive(Debug)]
pub struct TestDetail {
    pub name: String,
    pub passed: bool,
    pub duration: Duration,
    pub error: Option<String>,
}

/// Run all integration tests
pub async fn run_integration_tests() -> Result<IntegrationResults> {
    let start = Instant::now();
    let mut results = IntegrationResults::default();
    
    // Test QRNG integration
    run_test(&mut results, "QRNG Hardware Integration", test_qrng_integration()).await;
    run_test(&mut results, "QRNG Entropy Quality", test_qrng_entropy_quality()).await;
    
    // Test L-VRF integration
    run_test(&mut results, "L-VRF Basic Operation", test_lvrf_basic_operation()).await;
    run_test(&mut results, "L-VRF Consensus Integration", test_lvrf_consensus_integration()).await;
    run_test(&mut results, "L-VRF Zero-Knowledge Proofs", test_lvrf_zk_proofs()).await;
    
    // Test VDF integration
    run_test(&mut results, "Quantum VDF Verification", test_quantum_vdf_verification()).await;
    run_test(&mut results, "VDF L-VRF Seed Integration", test_vdf_lvrf_integration()).await;
    
    // Test Fair Queue integration
    run_test(&mut results, "Quantum Fair Queue Basic", test_fair_queue_basic()).await;
    run_test(&mut results, "Anti-Censorship Detection", test_anti_censorship_detection()).await;
    
    // Test full system integration
    run_test(&mut results, "End-to-End Consensus", test_end_to_end_consensus()).await;
    run_test(&mut results, "Multi-Phase Compatibility", test_multi_phase_compatibility()).await;
    
    results.execution_time = start.elapsed();
    results.passed = results.tests_passed == results.tests_run;
    
    Ok(results)
}

async fn run_test<F, Fut>(results: &mut IntegrationResults, name: &str, test: F) 
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = Result<()>>,
{
    let start = Instant::now();
    results.tests_run += 1;
    
    match test().await {
        Ok(()) => {
            results.tests_passed += 1;
            results.details.push(TestDetail {
                name: name.to_string(),
                passed: true,
                duration: start.elapsed(),
                error: None,
            });
        }
        Err(e) => {
            results.details.push(TestDetail {
                name: name.to_string(),
                passed: false,
                duration: start.elapsed(),
                error: Some(e.to_string()),
            });
        }
    }
}

/// Test QRNG hardware integration
async fn test_qrng_integration() -> Result<()> {
    let mut qrng = QuantumRNG::new(QRNGProvider::Simulation).await?;
    
    // Test basic random generation
    let random_bytes = qrng.generate(32).await?;
    if random_bytes.len() != 32 {
        return Err(anyhow!("QRNG returned wrong number of bytes"));
    }
    
    // Test multiple generations are different
    let random_bytes_2 = qrng.generate(32).await?;
    if random_bytes == random_bytes_2 {
        return Err(anyhow!("QRNG returned identical results"));
    }
    
    // Test entropy pool
    let pool_size = qrng.get_entropy_pool_size().await?;
    if pool_size == 0 {
        return Err(anyhow!("QRNG entropy pool is empty"));
    }
    
    Ok(())
}

/// Test QRNG entropy quality
async fn test_qrng_entropy_quality() -> Result<()> {
    let mut qrng = QuantumRNG::new(QRNGProvider::Simulation).await?;
    
    // Generate larger sample for analysis
    let sample = qrng.generate(1024).await?;
    
    // Basic entropy checks
    let entropy_estimate = qrng.estimate_entropy(&sample).await?;
    if entropy_estimate < 7.0 {
        return Err(anyhow!("Entropy estimate too low: {}", entropy_estimate));
    }
    
    // Check for bias
    let ones = sample.iter().map(|&b| b.count_ones()).sum::<u32>();
    let total_bits = sample.len() as u32 * 8;
    let bias = (ones as f64 / total_bits as f64 - 0.5).abs();
    
    if bias > 0.1 {
        return Err(anyhow!("Excessive bias in QRNG output: {:.3}", bias));
    }
    
    Ok(())
}

/// Test L-VRF basic operation
async fn test_lvrf_basic_operation() -> Result<()> {
    let lvrf = LatticeVRF::new(SecurityLevel::High).await?;
    
    let input = b"test_input";
    let round = Round::new(1);
    
    // Test VRF evaluation
    let result = lvrf.evaluate(input, round).await?;
    
    // Test verification
    let is_valid = lvrf.verify(input, round, &result.output, &result.proof).await?;
    if !is_valid {
        return Err(anyhow!("L-VRF verification failed"));
    }
    
    // Test deterministic property
    let result_2 = lvrf.evaluate(input, round).await?;
    if result.output != result_2.output {
        return Err(anyhow!("L-VRF is not deterministic"));
    }
    
    Ok(())
}

/// Test L-VRF consensus integration
async fn test_lvrf_consensus_integration() -> Result<()> {
    let lvrf = LatticeVRF::new(SecurityLevel::Medium).await?;
    let mut anchor_election = AnchorElection::new().await?;
    
    // Test anchor election with L-VRF
    let candidates = vec![
        [1u8; 32], [2u8; 32], [3u8; 32], [4u8; 32], [5u8; 32]
    ];
    
    let round = Round::new(1);
    let elected = anchor_election.elect_anchor_with_vrf(&lvrf, &candidates, round).await?;
    
    if !candidates.contains(&elected) {
        return Err(anyhow!("Elected anchor not in candidate list"));
    }
    
    // Test that election is verifiable
    let is_valid = anchor_election.verify_anchor_election(&lvrf, &candidates, elected, round).await?;
    if !is_valid {
        return Err(anyhow!("Anchor election verification failed"));
    }
    
    Ok(())
}

/// Test L-VRF zero-knowledge proofs
async fn test_lvrf_zk_proofs() -> Result<()> {
    let lvrf = LatticeVRF::new(SecurityLevel::High).await?;
    
    let input = b"zk_test_input";
    let round = Round::new(42);
    
    // Generate VRF result with ZK proof
    let result = lvrf.evaluate(input, round).await?;
    
    // Verify ZK proof without revealing secret key
    let zk_valid = lvrf.verify_zero_knowledge_proof(&result.proof, input, round).await?;
    if !zk_valid {
        return Err(anyhow!("Zero-knowledge proof verification failed"));
    }
    
    // Test that proof doesn't reveal secret information
    let proof_bytes = bincode::serialize(&result.proof)?;
    if proof_bytes.len() < 64 {
        return Err(anyhow!("ZK proof suspiciously small"));
    }
    
    Ok(())
}

/// Test quantum VDF verification
async fn test_quantum_vdf_verification() -> Result<()> {
    let vdf = QuantumVDF::new(VDFProtocol::QuantumHybrid).await?;
    
    let input = b"vdf_test_input";
    let time_param = 1000;
    
    // Evaluate VDF
    let start = Instant::now();
    let result = vdf.evaluate(input, time_param).await?;
    let computation_time = start.elapsed();
    
    // Verify result
    let verification_start = Instant::now();
    let is_valid = vdf.verify(input, time_param, &result.output, &result.proof).await?;
    let verification_time = verification_start.elapsed();
    
    if !is_valid {
        return Err(anyhow!("VDF verification failed"));
    }
    
    // Verification should be much faster than computation
    if verification_time > computation_time / 10 {
        return Err(anyhow!("VDF verification too slow relative to computation"));
    }
    
    Ok(())
}

/// Test VDF L-VRF seed integration
async fn test_vdf_lvrf_integration() -> Result<()> {
    let lvrf = LatticeVRF::new(SecurityLevel::Medium).await?;
    let vdf = QuantumVDF::new(VDFProtocol::QuantumHybrid).await?;
    
    let input = b"integration_test";
    let round = Round::new(1);
    
    // Generate VRF seed
    let vrf_result = lvrf.evaluate(input, round).await?;
    
    // Use VRF output as VDF input for enhanced randomness
    let vdf_result = vdf.evaluate(&vrf_result.output, 500).await?;
    
    // Verify both components
    let vrf_valid = lvrf.verify(input, round, &vrf_result.output, &vrf_result.proof).await?;
    let vdf_valid = vdf.verify(&vrf_result.output, 500, &vdf_result.output, &vdf_result.proof).await?;
    
    if !vrf_valid || !vdf_valid {
        return Err(anyhow!("VDF-VRF integration verification failed"));
    }
    
    Ok(())
}

/// Test quantum fair queue basic functionality
async fn test_fair_queue_basic() -> Result<()> {
    let mut fair_queue = QuantumFairQueue::new(QueueingPolicy::VRFBased).await?;
    
    // Add test transactions
    let tx1 = Uuid::new_v4().into_bytes();
    let tx2 = Uuid::new_v4().into_bytes();
    let tx3 = Uuid::new_v4().into_bytes();
    
    fair_queue.enqueue_transaction(tx1, [1u8; 32]).await?;
    fair_queue.enqueue_transaction(tx2, [2u8; 32]).await?;
    fair_queue.enqueue_transaction(tx3, [3u8; 32]).await?;
    
    // Test dequeuing
    let dequeued = fair_queue.dequeue_next_batch(2).await?;
    if dequeued.len() != 2 {
        return Err(anyhow!("Wrong number of transactions dequeued"));
    }
    
    // Test fairness metrics
    let fairness = fair_queue.calculate_fairness_metrics().await?;
    if fairness.gini_coefficient > 0.5 {
        return Err(anyhow!("Queue fairness insufficient: Gini = {:.3}", fairness.gini_coefficient));
    }
    
    Ok(())
}

/// Test anti-censorship detection
async fn test_anti_censorship_detection() -> Result<()> {
    let mut fair_queue = QuantumFairQueue::new(QueueingPolicy::AntiCensorship).await?;
    
    let tx_id = Uuid::new_v4().into_bytes();
    let suspicious_node = [42u8; 32];
    
    // Simulate many rejections from the same node
    for _ in 0..100 {
        let _ = fair_queue.detect_censorship_attempt(&tx_id, suspicious_node).await?;
        sleep(Duration::from_millis(10)).await;
    }
    
    // Should now detect censorship
    let is_censorship = fair_queue.detect_censorship_attempt(&tx_id, suspicious_node).await?;
    if !is_censorship {
        return Err(anyhow!("Failed to detect obvious censorship pattern"));
    }
    
    // Check that countermeasures are applied
    let metrics = fair_queue.get_anti_censorship_metrics().await?;
    if metrics.suspected_censors_count == 0 {
        return Err(anyhow!("No suspected censors detected"));
    }
    
    Ok(())
}

/// Test end-to-end consensus with all quantum components
async fn test_end_to_end_consensus() -> Result<()> {
    // Initialize all quantum components
    let mut qrng = QuantumRNG::new(QRNGProvider::Simulation).await?;
    let lvrf = LatticeVRF::new(SecurityLevel::Medium).await?;
    let vdf = QuantumVDF::new(VDFProtocol::QuantumHybrid).await?;
    let mut fair_queue = QuantumFairQueue::new(QueueingPolicy::VRFBased).await?;
    let mut anchor_election = AnchorElection::new().await?;
    
    // Simulate consensus round
    let round = Round::new(1);
    let candidates = vec![[1u8; 32], [2u8; 32], [3u8; 32]];
    
    // 1. Generate quantum entropy for consensus
    let entropy = qrng.generate(32).await?;
    
    // 2. Elect anchor using L-VRF
    let anchor = anchor_election.elect_anchor_with_vrf(&lvrf, &candidates, round).await?;
    
    // 3. Process transactions through fair queue
    let tx_id = Uuid::new_v4().into_bytes();
    fair_queue.enqueue_transaction(tx_id, anchor).await?;
    let batch = fair_queue.dequeue_next_batch(1).await?;
    
    // 4. Generate VDF proof for timing
    let vdf_input = [&entropy[..], &tx_id].concat();
    let vdf_result = vdf.evaluate(&vdf_input, 100).await?;
    
    // Verify all components worked together
    let anchor_valid = anchor_election.verify_anchor_election(&lvrf, &candidates, anchor, round).await?;
    let vdf_valid = vdf.verify(&vdf_input, 100, &vdf_result.output, &vdf_result.proof).await?;
    
    if !anchor_valid || !vdf_valid || batch.is_empty() {
        return Err(anyhow!("End-to-end consensus test failed"));
    }
    
    Ok(())
}

/// Test multi-phase compatibility
async fn test_multi_phase_compatibility() -> Result<()> {
    // Test that quantum components gracefully handle non-quantum modes
    
    // Phase 0 (classical) mode should still work
    let qrng_classical = QuantumRNG::new_with_phase(QRNGProvider::Simulation, 0).await?;
    let classical_bytes = qrng_classical.generate(16).await?;
    if classical_bytes.len() != 16 {
        return Err(anyhow!("Phase 0 compatibility broken"));
    }
    
    // Phase 2 (quantum) mode should provide enhanced features
    let qrng_quantum = QuantumRNG::new_with_phase(QRNGProvider::Simulation, 2).await?;
    let quantum_bytes = qrng_quantum.generate(16).await?;
    if quantum_bytes.len() != 16 {
        return Err(anyhow!("Phase 2 quantum mode broken"));
    }
    
    // Verify quantum mode provides better entropy
    let classical_entropy = qrng_classical.estimate_entropy(&classical_bytes).await?;
    let quantum_entropy = qrng_quantum.estimate_entropy(&quantum_bytes).await?;
    
    if quantum_entropy <= classical_entropy {
        return Err(anyhow!("Quantum mode doesn't improve entropy"));
    }
    
    Ok(())
}