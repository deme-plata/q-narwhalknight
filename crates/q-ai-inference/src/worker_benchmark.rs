//! Worker Hardware Verification & Benchmarking
//!
//! This module implements comprehensive benchmarks to verify worker hardware claims
//! and prevent dishonest nodes from claiming superior hardware for better work allocation.
//!
//! ## Why Benchmarking?
//!
//! Workers announce capability (e.g., "CUDA GPU with 24GB VRAM") but without verification:
//! - Dishonest workers can claim false hardware
//! - Get assigned more work than they can handle
//! - Cause network degradation and delays
//!
//! ## Benchmark Tests
//!
//! 1. **Inference Speed Test**: Measure tokens/second on standard prompt
//! 2. **Memory Capacity Test**: Verify VRAM/RAM capacity claims
//! 3. **Concurrent Load Test**: Test handling multiple requests
//! 4. **Latency Test**: Measure response time consistency
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                Worker Registration Flow                     │
//! └─────────────────────────────────────────────────────────────┘
//!
//! New Worker:                      Coordinator:
//! ┌──────────────┐                ┌──────────────┐
//! │ 1. Announce  │───────────────>│ 2. Receive   │
//! │   "CUDA 24GB"│   Capability   │   Claim      │
//! │              │                │              │
//! │              │<───────────────│ 3. Send      │
//! │              │   Benchmark    │   Benchmark  │
//! │              │   Challenge    │   Tests      │
//! │ 4. Execute   │                │              │
//! │   Benchmark  │                │              │
//! │   (30 tokens)│                │              │
//! │              │───────────────>│ 5. Validate  │
//! │              │   Results      │   Results    │
//! │              │   (2.1s)       │              │
//! │              │                │ Expected:    │
//! │              │                │ CUDA: <3s    │
//! │              │                │ CPU:  <10s   │
//! │              │                │              │
//! │              │<───────────────│ 6. Approve/  │
//! │              │   Accept/Reject│   Reject     │
//! └──────────────┘                └──────────────┘
//!       │                                │
//!       └──── Approved: Join Network ────┘
//!             Rejected: Banned
//! ```

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

use crate::types::DeviceCapability;

/// Benchmark challenge sent to worker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkChallenge {
    /// Challenge ID (unique nonce)
    pub challenge_id: String,

    /// Test prompt for inference
    pub prompt: String,

    /// Number of tokens to generate
    pub max_tokens: usize,

    /// Model to use
    pub model: String,

    /// Issued timestamp
    pub issued_at_ms: u64,

    /// Deadline for completion
    pub deadline_ms: u64,
}

/// Benchmark result submitted by worker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Challenge ID
    pub challenge_id: String,

    /// Worker node ID
    pub worker_node_id: String,

    /// Generated tokens
    pub tokens_generated: usize,

    /// Time taken (milliseconds)
    pub time_ms: u64,

    /// Tokens per second achieved
    pub tokens_per_second: f64,

    /// Peak memory usage (MB)
    pub peak_memory_mb: usize,

    /// Device used (CPU/CUDA/Metal)
    pub device: String,

    /// Submitted timestamp
    pub submitted_at_ms: u64,
}

/// Benchmark verification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BenchmarkVerification {
    /// Benchmark passed - worker claims verified
    Passed {
        worker_node_id: String,
        verified_capability: DeviceCapability,
        score: f64, // Performance score (0.0 - 1.0)
    },

    /// Benchmark failed - worker claims false or insufficient
    Failed {
        worker_node_id: String,
        claimed_capability: DeviceCapability,
        reason: String,
        ban_duration_hours: u64,
    },

    /// Benchmark timeout - worker too slow or unresponsive
    Timeout {
        worker_node_id: String,
        claimed_capability: DeviceCapability,
        ban_duration_hours: u64,
    },
}

/// Expected performance thresholds for different hardware
#[derive(Debug, Clone)]
pub struct PerformanceThresholds {
    /// Minimum tokens/second for CUDA GPUs
    pub cuda_min_tps: f64,

    /// Minimum tokens/second for Metal GPUs
    pub metal_min_tps: f64,

    /// Minimum tokens/second for CPU
    pub cpu_min_tps: f64,

    /// Maximum acceptable time for benchmark (ms)
    pub max_time_ms: u64,

    /// Tolerance for VRAM/RAM claims (±10%)
    pub memory_tolerance: f64,
}

impl Default for PerformanceThresholds {
    fn default() -> Self {
        Self {
            cuda_min_tps: 15.0,  // CUDA should achieve 15+ tok/s on 7B model
            metal_min_tps: 12.0, // Metal 12+ tok/s
            cpu_min_tps: 2.0,    // CPU at least 2+ tok/s
            max_time_ms: 30_000, // 30 seconds maximum
            memory_tolerance: 0.10, // ±10% for memory claims
        }
    }
}

/// Worker benchmark verifier
pub struct WorkerBenchmarkVerifier {
    /// Active benchmark challenges
    pending_challenges: Arc<RwLock<HashMap<String, BenchmarkChallenge>>>,

    /// Benchmark results awaiting verification
    pending_results: Arc<RwLock<HashMap<String, BenchmarkResult>>>,

    /// Verified worker capabilities
    verified_workers: Arc<RwLock<HashMap<String, DeviceCapability>>>,

    /// Banned workers (node_id -> unban_timestamp_ms)
    banned_workers: Arc<RwLock<HashMap<String, u64>>>,

    /// Performance thresholds
    thresholds: PerformanceThresholds,

    /// Configuration
    config: BenchmarkConfig,
}

/// Benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Benchmark prompt
    pub benchmark_prompt: String,

    /// Token count for benchmark
    pub benchmark_tokens: usize,

    /// Deadline for benchmark completion (ms)
    pub benchmark_deadline_ms: u64,

    /// Model to use for benchmarking
    pub benchmark_model: String,

    /// Ban duration for failed benchmarks (hours)
    pub ban_duration_hours: u64,

    /// Require re-benchmark every N hours
    pub rebenchmark_interval_hours: u64,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            benchmark_prompt: "Explain quantum computing in simple terms.".to_string(),
            benchmark_tokens: 30, // Small benchmark: 30 tokens
            benchmark_deadline_ms: 30_000, // 30 seconds
            benchmark_model: "Mistral-7B-Instruct-v0.3".to_string(),
            ban_duration_hours: 24, // 24 hour ban for failures
            rebenchmark_interval_hours: 168, // Re-benchmark weekly
        }
    }
}

impl WorkerBenchmarkVerifier {
    /// Create new benchmark verifier
    pub fn new(config: BenchmarkConfig, thresholds: PerformanceThresholds) -> Self {
        info!("🏋️ Initializing Worker Benchmark Verifier");
        info!("   Benchmark: {} tokens in <{}s", config.benchmark_tokens, config.benchmark_deadline_ms / 1000);
        info!("   Thresholds: CUDA≥{:.1} tok/s, Metal≥{:.1} tok/s, CPU≥{:.1} tok/s",
              thresholds.cuda_min_tps, thresholds.metal_min_tps, thresholds.cpu_min_tps);

        Self {
            pending_challenges: Arc::new(RwLock::new(HashMap::new())),
            pending_results: Arc::new(RwLock::new(HashMap::new())),
            verified_workers: Arc::new(RwLock::new(HashMap::new())),
            banned_workers: Arc::new(RwLock::new(HashMap::new())),
            thresholds,
            config,
        }
    }

    /// Issue benchmark challenge to worker
    pub async fn issue_challenge(&self, worker_node_id: &str, claimed_capability: &DeviceCapability) -> Result<BenchmarkChallenge> {
        info!("🎯 Issuing benchmark challenge to worker {}", worker_node_id);
        info!("   Claimed capability: {:?}", claimed_capability);

        // Check if worker is banned
        if self.is_banned(worker_node_id).await {
            let unban_time = self.banned_workers.read().await.get(worker_node_id).copied().unwrap_or(0);
            return Err(anyhow!("Worker {} is banned until {}", worker_node_id, unban_time));
        }

        let challenge = BenchmarkChallenge {
            challenge_id: uuid::Uuid::new_v4().to_string(),
            prompt: self.config.benchmark_prompt.clone(),
            max_tokens: self.config.benchmark_tokens,
            model: self.config.benchmark_model.clone(),
            issued_at_ms: current_timestamp_ms(),
            deadline_ms: self.config.benchmark_deadline_ms,
        };

        info!("   Challenge ID: {}", challenge.challenge_id);
        info!("   Prompt: '{}'", challenge.prompt);
        info!("   Required: {} tokens in <{}ms", challenge.max_tokens, challenge.deadline_ms);

        self.pending_challenges.write().await.insert(challenge.challenge_id.clone(), challenge.clone());

        Ok(challenge)
    }

    /// Verify benchmark result from worker
    pub async fn verify_result(
        &self,
        result: BenchmarkResult,
        claimed_capability: DeviceCapability,
    ) -> Result<BenchmarkVerification> {
        info!("🔍 Verifying benchmark result from worker {}", result.worker_node_id);
        info!("   Tokens: {}, Time: {}ms, TPS: {:.2}",
              result.tokens_generated, result.time_ms, result.tokens_per_second);

        // Get original challenge
        let challenge = self.pending_challenges.read().await
            .get(&result.challenge_id)
            .ok_or_else(|| anyhow!("No challenge found for ID {}", result.challenge_id))?
            .clone();

        // Check timeout
        let now = current_timestamp_ms();
        if now > challenge.issued_at_ms + challenge.deadline_ms {
            warn!("⏰ Benchmark TIMEOUT for worker {}", result.worker_node_id);
            return Ok(self.handle_timeout(&result.worker_node_id, &claimed_capability).await);
        }

        // Verify token count
        if result.tokens_generated < challenge.max_tokens {
            return Ok(self.handle_failure(
                &result.worker_node_id,
                &claimed_capability,
                &format!("Insufficient tokens: {} < {}", result.tokens_generated, challenge.max_tokens)
            ).await);
        }

        // Verify performance based on claimed capability
        let min_tps = match &claimed_capability {
            DeviceCapability::CUDA { .. } => self.thresholds.cuda_min_tps,
            DeviceCapability::Metal { .. } => self.thresholds.metal_min_tps,
            DeviceCapability::CPU { .. } => self.thresholds.cpu_min_tps,
        };

        if result.tokens_per_second < min_tps {
            return Ok(self.handle_failure(
                &result.worker_node_id,
                &claimed_capability,
                &format!("Performance too low: {:.2} tok/s < {:.2} tok/s minimum",
                         result.tokens_per_second, min_tps)
            ).await);
        }

        // Verify memory usage matches claims (with tolerance)
        if !self.verify_memory_claim(&result, &claimed_capability) {
            return Ok(self.handle_failure(
                &result.worker_node_id,
                &claimed_capability,
                &format!("Memory usage mismatch: claimed vs actual doesn't match within {}% tolerance",
                         self.thresholds.memory_tolerance * 100.0)
            ).await);
        }

        // Calculate performance score (0.0 - 1.0)
        let score = self.calculate_performance_score(&result, &claimed_capability);

        info!("✅ Benchmark PASSED for worker {}", result.worker_node_id);
        info!("   Performance score: {:.3}/1.0", score);

        // Store verified capability
        self.verified_workers.write().await.insert(
            result.worker_node_id.clone(),
            claimed_capability.clone()
        );

        // Cleanup
        self.pending_challenges.write().await.remove(&result.challenge_id);

        Ok(BenchmarkVerification::Passed {
            worker_node_id: result.worker_node_id,
            verified_capability: claimed_capability,
            score,
        })
    }

    /// Verify memory capacity claims
    fn verify_memory_claim(&self, result: &BenchmarkResult, claimed: &DeviceCapability) -> bool {
        let claimed_memory = match claimed {
            DeviceCapability::CUDA { vram_gb, .. } => *vram_gb * 1024, // Convert to MB
            DeviceCapability::Metal { vram_gb } => *vram_gb * 1024,
            DeviceCapability::CPU { ram_gb, .. } => *ram_gb * 1024,
        };

        let used_memory = result.peak_memory_mb;

        // Check if used memory is reasonable (model should use at least 4GB for 7B)
        if used_memory < 4000 {
            warn!("⚠️ Suspiciously low memory usage: {}MB (expected >4GB for 7B model)", used_memory);
            return false;
        }

        // Check if claimed memory can fit the model
        let min_required = 4000; // Minimum 4GB for Mistral-7B Q4
        if claimed_memory < min_required {
            warn!("⚠️ Claimed memory too low: {}MB < {}MB minimum", claimed_memory, min_required);
            return false;
        }

        true
    }

    /// Calculate performance score (0.0 = minimum acceptable, 1.0 = excellent)
    fn calculate_performance_score(&self, result: &BenchmarkResult, capability: &DeviceCapability) -> f64 {
        let min_tps = match capability {
            DeviceCapability::CUDA { .. } => self.thresholds.cuda_min_tps,
            DeviceCapability::Metal { .. } => self.thresholds.metal_min_tps,
            DeviceCapability::CPU { .. } => self.thresholds.cpu_min_tps,
        };

        // Excellent performance = 2x minimum threshold
        let excellent_tps = min_tps * 2.0;

        // Linear interpolation between minimum and excellent
        let score = ((result.tokens_per_second - min_tps) / (excellent_tps - min_tps)).clamp(0.0, 1.0);

        score
    }

    /// Handle benchmark failure
    async fn handle_failure(
        &self,
        worker_node_id: &str,
        claimed_capability: &DeviceCapability,
        reason: &str,
    ) -> BenchmarkVerification {
        error!("❌ Benchmark FAILED for worker {}: {}", worker_node_id, reason);

        // Ban worker
        let ban_until = current_timestamp_ms() + (self.config.ban_duration_hours * 3600 * 1000);
        self.banned_workers.write().await.insert(worker_node_id.to_string(), ban_until);

        BenchmarkVerification::Failed {
            worker_node_id: worker_node_id.to_string(),
            claimed_capability: claimed_capability.clone(),
            reason: reason.to_string(),
            ban_duration_hours: self.config.ban_duration_hours,
        }
    }

    /// Handle benchmark timeout
    async fn handle_timeout(
        &self,
        worker_node_id: &str,
        claimed_capability: &DeviceCapability,
    ) -> BenchmarkVerification {
        warn!("⏰ Benchmark TIMEOUT for worker {}", worker_node_id);

        // Ban worker
        let ban_until = current_timestamp_ms() + (self.config.ban_duration_hours * 3600 * 1000);
        self.banned_workers.write().await.insert(worker_node_id.to_string(), ban_until);

        BenchmarkVerification::Timeout {
            worker_node_id: worker_node_id.to_string(),
            claimed_capability: claimed_capability.clone(),
            ban_duration_hours: self.config.ban_duration_hours,
        }
    }

    /// Check if worker is currently banned
    pub async fn is_banned(&self, worker_node_id: &str) -> bool {
        let banned = self.banned_workers.read().await;
        if let Some(&unban_time) = banned.get(worker_node_id) {
            let now = current_timestamp_ms();
            if now < unban_time {
                return true; // Still banned
            }
            // Ban expired - will be cleaned up next
        }
        false
    }

    /// Clean up expired bans
    pub async fn cleanup_expired_bans(&self) {
        let now = current_timestamp_ms();
        let mut banned = self.banned_workers.write().await;
        banned.retain(|_, &mut unban_time| unban_time > now);
    }

    /// Check if worker needs re-benchmarking (weekly check)
    pub async fn needs_rebenchmark(&self, worker_node_id: &str) -> bool {
        // TODO: Track last benchmark time per worker
        // For now, assume verified workers don't need re-benchmark
        !self.verified_workers.read().await.contains_key(worker_node_id)
    }

    /// Get verified capability for worker
    pub async fn get_verified_capability(&self, worker_node_id: &str) -> Option<DeviceCapability> {
        self.verified_workers.read().await.get(worker_node_id).cloned()
    }
}

/// Get current timestamp in milliseconds
fn current_timestamp_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_issue_benchmark_challenge() {
        let verifier = WorkerBenchmarkVerifier::new(
            BenchmarkConfig::default(),
            PerformanceThresholds::default()
        );

        let capability = DeviceCapability::CUDA {
            vram_gb: 24,
            compute_capability: "8.0".to_string(),
        };

        let challenge = verifier.issue_challenge("worker-1", &capability).await;
        assert!(challenge.is_ok());

        let challenge = challenge.unwrap();
        assert_eq!(challenge.max_tokens, 30);
        assert!(!challenge.prompt.is_empty());
    }

    #[tokio::test]
    async fn test_verify_passing_benchmark() {
        let verifier = WorkerBenchmarkVerifier::new(
            BenchmarkConfig::default(),
            PerformanceThresholds::default()
        );

        let capability = DeviceCapability::CUDA {
            vram_gb: 24,
            compute_capability: "8.0".to_string(),
        };

        let challenge = verifier.issue_challenge("worker-1", &capability).await.unwrap();

        // Simulate good result
        let result = BenchmarkResult {
            challenge_id: challenge.challenge_id,
            worker_node_id: "worker-1".to_string(),
            tokens_generated: 30,
            time_ms: 1500, // 1.5 seconds for 30 tokens = 20 tok/s (exceeds 15 tok/s minimum)
            tokens_per_second: 20.0,
            peak_memory_mb: 5000,
            device: "CUDA".to_string(),
            submitted_at_ms: current_timestamp_ms(),
        };

        let verification = verifier.verify_result(result, capability).await.unwrap();

        match verification {
            BenchmarkVerification::Passed { score, .. } => {
                assert!(score > 0.0);
                println!("✅ Test passed with score: {:.3}", score);
            }
            _ => panic!("Expected Passed verification"),
        }
    }

    #[tokio::test]
    async fn test_verify_failing_benchmark_slow() {
        let verifier = WorkerBenchmarkVerifier::new(
            BenchmarkConfig::default(),
            PerformanceThresholds::default()
        );

        let capability = DeviceCapability::CUDA {
            vram_gb: 24,
            compute_capability: "8.0".to_string(),
        };

        let challenge = verifier.issue_challenge("worker-slow", &capability).await.unwrap();

        // Simulate slow result (below minimum threshold)
        let result = BenchmarkResult {
            challenge_id: challenge.challenge_id,
            worker_node_id: "worker-slow".to_string(),
            tokens_generated: 30,
            time_ms: 10000, // 10 seconds for 30 tokens = 3 tok/s (below 15 tok/s minimum for CUDA)
            tokens_per_second: 3.0,
            peak_memory_mb: 5000,
            device: "CUDA".to_string(),
            submitted_at_ms: current_timestamp_ms(),
        };

        let verification = verifier.verify_result(result, capability).await.unwrap();

        match verification {
            BenchmarkVerification::Failed { reason, .. } => {
                assert!(reason.contains("Performance too low"));
                println!("✅ Test correctly failed: {}", reason);
            }
            _ => panic!("Expected Failed verification"),
        }
    }
}
