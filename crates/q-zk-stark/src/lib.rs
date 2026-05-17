//! Q-NarwhalKnight ZK-STARK Implementation
//!
//! This crate provides zero-knowledge Scalable Transparent Arguments of Knowledge (STARK)
//! implementation with GPU acceleration for the Q-NarwhalKnight quantum consensus system.
//!
//! Based on Server Beta's Phase 3 analysis and collaboration, this implementation targets:
//! - 50K+ TPS with zero-knowledge proofs
//! - <2s proof generation for complex circuits
//! - <10ms proof verification
//! - 10x-100x GPU acceleration for cryptographic operations

pub mod air;
pub mod batch_prover;
pub mod blockchain_state_circuit;
pub mod gpu;
pub mod nova_srs_generator_air; // v10.9.20 Job E: transparent-setup STARK attestation for Nova SRS
pub mod performance;
pub mod polynomials;
pub mod stark_prover;
pub mod stark_verifier;
pub mod wallet_privacy_stark;

// Re-export main types for convenience
pub use air::{AirConstraints, ExecutionTrace};
pub use batch_prover::{BatchConfig, BatchProvingStats, BatchStarkProof, BatchStarkProver};
pub use blockchain_state_circuit::{BlockPossessionCircuit, BlockchainStateProof, ProofMetadata};
pub use gpu::{FriProofGpu, GpuProvingMetrics, GpuStarkProver, StarkProofGpu};
pub use performance::{PerformanceTargets, StarkPerformanceBenchmark};
pub use stark_prover::{StarkProof, StarkProver};
pub use stark_verifier::{StarkVerifier, VerificationResult};
pub use wallet_privacy_stark::{
    StarkBalanceRangeProof, StarkTransactionPrivacyProof, StarkWalletOwnershipProof,
    WalletPrivacyStarkProver,
};

use anyhow::Result;

/// Phase 3 STARK system with GPU acceleration and CPU batching
pub struct StarkSystem {
    gpu_prover: Option<GpuStarkProver>,
    cpu_prover: StarkProver,
    batch_prover: BatchStarkProver,
    verifier: StarkVerifier,
    performance_monitor: gpu::performance_monitor::PerformanceMonitor,
}

impl StarkSystem {
    /// Create new STARK system with optional GPU acceleration
    pub async fn new(enable_gpu: bool) -> Result<Self> {
        Self::new_with_batch_config(enable_gpu, BatchConfig::default()).await
    }

    /// Create new STARK system with custom batch configuration
    pub async fn new_with_batch_config(enable_gpu: bool, batch_config: BatchConfig) -> Result<Self> {
        let cpu_prover = StarkProver::new();
        let batch_prover = BatchStarkProver::with_config(batch_config);
        let verifier = StarkVerifier::new();

        let gpu_prover = if enable_gpu {
            Some(GpuStarkProver::new().await?)
        } else {
            None
        };

        Ok(Self {
            gpu_prover,
            cpu_prover,
            batch_prover,
            verifier,
            performance_monitor: gpu::performance_monitor::PerformanceMonitor::new(),
        })
    }

    /// Get mutable reference to batch prover for direct access
    pub fn batch_prover_mut(&mut self) -> &mut BatchStarkProver {
        &mut self.batch_prover
    }

    /// Get batch proving statistics
    pub fn batch_stats(&self) -> &BatchProvingStats {
        self.batch_prover.stats()
    }

    /// Generate STARK proof using best available method (GPU if available, CPU fallback)
    pub async fn prove(&mut self, trace: &[Vec<u64>], constraints: &[u8]) -> Result<StarkProof> {
        if let Some(gpu_prover) = &mut self.gpu_prover {
            // Use GPU acceleration
            let start = std::time::Instant::now();
            let gpu_proof = gpu_prover.prove_stark_gpu(trace, constraints).await?;
            let duration = start.elapsed();

            // Record performance metrics
            self.performance_monitor
                .record_proving_time(trace.len(), duration);

            // Convert GPU proof to standard format
            Ok(StarkProof::from_gpu_proof(gpu_proof))
        } else {
            // Fallback to CPU
            self.cpu_prover.prove(trace, constraints).await
        }
    }

    /// Verify STARK proof
    pub async fn verify(&mut self, proof: &StarkProof, public_inputs: &[u64]) -> Result<bool> {
        let start = std::time::Instant::now();
        let is_valid = self.verifier.verify(proof, public_inputs).await?;
        let duration = start.elapsed();

        // Record verification performance
        self.performance_monitor
            .record_verification_time(proof.size_bytes(), duration);

        Ok(is_valid)
    }

    /// Get performance report for monitoring and optimization
    pub fn performance_report(&self) -> gpu::performance_monitor::PerformanceReport {
        self.performance_monitor.generate_performance_report()
    }

    /// Check if system meets Phase 3 performance targets
    pub fn meets_phase3_targets(&self) -> bool {
        let compliance = self
            .performance_monitor
            .generate_performance_report()
            .phase3_compliance;
        compliance.ready_for_production
    }
}

/// STARK system error types
#[derive(Debug, thiserror::Error)]
pub enum StarkError {
    #[error("STARK proving failed: {0}")]
    ProvingFailed(String),

    #[error("STARK verification failed: {0}")]
    VerificationFailed(String),

    #[error("GPU acceleration error: {0}")]
    GpuError(String),

    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),

    #[error("Performance target not met: {0}")]
    PerformanceTarget(String),
}

/// Phase 3 integration tests
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_stark_system_creation() {
        let system = StarkSystem::new(false).await;
        assert!(system.is_ok(), "Should create STARK system without GPU");
    }

    #[tokio::test]
    async fn test_basic_stark_proof() {
        let mut system = StarkSystem::new(false).await.unwrap();

        // Simple execution trace
        let trace = vec![vec![1, 2, 3, 4], vec![2, 3, 4, 5], vec![3, 4, 5, 6]];

        let constraints = vec![0u8; 10]; // Empty constraints for test

        let proof = system.prove(&trace, &constraints).await;
        assert!(proof.is_ok(), "Should generate STARK proof");

        if let Ok(proof) = proof {
            let is_valid = system.verify(&proof, &[1, 2, 3, 4]).await.unwrap();
            assert!(is_valid, "Proof should verify correctly");
        }
    }

    #[tokio::test]
    async fn test_performance_monitoring() {
        let system = StarkSystem::new(false).await.unwrap();
        let report = system.performance_report();

        // Performance report should be generated without error
        assert!(report.proving_performance.total_proofs_generated == 0);
    }

    #[tokio::test]
    async fn test_phase3_compliance() {
        let system = StarkSystem::new(false).await.unwrap();

        // Fresh system should not have compliance data yet
        // Real tests would run actual proofs to gather metrics
        let _meets_targets = system.meets_phase3_targets();
    }
}

/// Benchmark utilities for Phase 3 validation
pub mod benchmarks {
    use super::*;
    use std::time::Duration;

    /// Run comprehensive STARK benchmarks
    pub async fn run_phase3_benchmarks() -> Result<()> {
        println!("🚀 Q-NarwhalKnight STARK Phase 3 Benchmarks");
        println!("{}", "=".repeat(60));

        // Test CPU-only system
        println!("\n📊 CPU STARK Performance:");
        let mut cpu_system = StarkSystem::new(false).await?;
        benchmark_system(&mut cpu_system, "CPU").await?;

        // Test GPU-accelerated system if available
        if let Ok(mut gpu_system) = StarkSystem::new(true).await {
            println!("\n⚡ GPU STARK Performance:");
            benchmark_system(&mut gpu_system, "GPU").await?;

            // Compare performance
            let cpu_report = cpu_system.performance_report();
            let gpu_report = gpu_system.performance_report();

            print_performance_comparison(&cpu_report, &gpu_report);
        } else {
            println!("\n⚠️  GPU acceleration not available - skipping GPU benchmarks");
        }

        println!("\n✅ Phase 3 STARK benchmarks complete!");
        Ok(())
    }

    async fn benchmark_system(system: &mut StarkSystem, system_type: &str) -> Result<()> {
        println!("Testing {} STARK system...", system_type);

        // Test different trace sizes
        for &trace_size in &[100, 1000, 10000] {
            let trace: Vec<Vec<u64>> = (0..trace_size)
                .map(|i| vec![i as u64, (i * 2) as u64, (i * 3) as u64])
                .collect();

            let constraints = vec![0u8; 100]; // Mock constraints

            let start = std::time::Instant::now();
            let proof_result = system.prove(&trace, &constraints).await;
            let proving_time = start.elapsed();

            match proof_result {
                Ok(proof) => {
                    let verification_start = std::time::Instant::now();
                    let is_valid = system.verify(&proof, &[0]).await?;
                    let verification_time = verification_start.elapsed();

                    println!(
                        "  Trace size: {} | Proving: {:?} | Verification: {:?} | Valid: {}",
                        trace_size, proving_time, verification_time, is_valid
                    );

                    // Check Phase 3 targets
                    let meets_proving_target = proving_time <= Duration::from_millis(2000);
                    let meets_verification_target = verification_time <= Duration::from_millis(10);

                    if meets_proving_target && meets_verification_target {
                        println!("    ✅ Meets Phase 3 targets");
                    } else {
                        println!("    ⚠️  Performance optimization needed");
                    }
                }
                Err(e) => {
                    println!("  Trace size: {} | Error: {}", trace_size, e);
                }
            }
        }

        Ok(())
    }

    fn print_performance_comparison(
        cpu_report: &gpu::performance_monitor::PerformanceReport,
        gpu_report: &gpu::performance_monitor::PerformanceReport,
    ) {
        println!("\n📈 Performance Comparison:");
        println!("CPU vs GPU proving performance:");

        let cpu_avg = cpu_report
            .proving_performance
            .average_proving_time
            .as_millis();
        let gpu_avg = gpu_report
            .proving_performance
            .average_proving_time
            .as_millis();

        if gpu_avg > 0 {
            let speedup = cpu_avg as f64 / gpu_avg as f64;
            println!(
                "  Average proving time - CPU: {}ms | GPU: {}ms",
                cpu_avg, gpu_avg
            );
            println!("  GPU Speedup: {:.1}x", speedup);

            // Validate Server Beta's 10x-100x speedup projections
            if speedup >= 10.0 {
                println!("  ✅ Meets Server Beta's 10x+ speedup target");
            } else {
                println!("  ⚠️  Below 10x speedup target - further optimization needed");
            }
        }
    }
}
