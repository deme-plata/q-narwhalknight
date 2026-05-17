//! Automatic Failover & Retry Logic for Distributed AI
//!
//! This module provides robust fault tolerance for distributed inference:
//! - Automatic worker failure detection
//! - Request retry on worker failure
//! - Dynamic reassignment to healthy workers
//! - Circuit breaker for failing workers
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                   Failover Flow                             │
//! └─────────────────────────────────────────────────────────────┘
//!
//! Request Flow:                    Failover:
//! ┌──────────────┐                ┌──────────────┐
//! │ 1. Send to   │───────────────>│ Worker A     │
//! │   Worker A   │   Inference    │ (Processing) │
//! │              │                │              │
//! │              │<───────────────│ ❌ TIMEOUT   │
//! │              │    No Response │ (30s)        │
//! │              │                │              │
//! │ 2. Detect    │                └──────────────┘
//! │   Failure    │                       │
//! │              │                       ▼
//! │ 3. Retry     │                ┌──────────────┐
//! │   Worker B   │───────────────>│ Worker B     │
//! │              │   Inference    │ (Healthy)    │
//! │              │                │              │
//! │              │<───────────────│ ✅ Success   │
//! │ 4. Success!  │    Result      │              │
//! └──────────────┘                └──────────────┘
//! ```
//!
//! ## Circuit Breaker States
//!
//! ```text
//! CLOSED (Normal)
//!    │
//!    ├─ 3 failures → OPEN
//!    │
//! OPEN (Blocked)
//!    │
//!    ├─ Wait 60s → HALF_OPEN
//!    │
//! HALF_OPEN (Testing)
//!    │
//!    ├─ Success → CLOSED
//!    └─ Failure → OPEN
//! ```

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Worker health status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkerHealth {
    /// Worker is healthy and accepting requests
    Healthy,

    /// Worker is experiencing intermittent failures
    Degraded,

    /// Worker is failing consistently (circuit breaker OPEN)
    Unhealthy,

    /// Worker is being tested after recovery (circuit breaker HALF_OPEN)
    Testing,
}

/// Failure record for a worker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureRecord {
    /// Request ID that failed
    pub request_id: String,

    /// Failure type
    pub failure_type: FailureType,

    /// Timestamp when failure occurred
    pub failed_at_ms: u64,

    /// Error message
    pub error_message: String,
}

/// Types of failures
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FailureType {
    /// Request timeout (no response within deadline)
    Timeout,

    /// Invalid response (malformed data)
    InvalidResponse,

    /// Worker crashed/disconnected
    WorkerDisconnected,

    /// Inference error (model error, OOM, etc.)
    InferenceError,

    /// Network error (connection lost)
    NetworkError,
}

/// Circuit breaker for worker
#[derive(Debug, Clone)]
struct CircuitBreaker {
    /// Current state
    state: CircuitBreakerState,

    /// Failure count
    failure_count: usize,

    /// Last failure timestamp
    last_failure_ms: u64,

    /// Last state change timestamp
    last_state_change_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CircuitBreakerState {
    Closed,    // Normal operation
    Open,      // Blocking requests
    HalfOpen,  // Testing recovery
}

impl CircuitBreaker {
    fn new() -> Self {
        Self {
            state: CircuitBreakerState::Closed,
            failure_count: 0,
            last_failure_ms: 0,
            last_state_change_ms: current_timestamp_ms(),
        }
    }

    /// Record a failure
    fn record_failure(&mut self, failure_threshold: usize, timeout_ms: u64) {
        self.failure_count += 1;
        self.last_failure_ms = current_timestamp_ms();

        if self.failure_count >= failure_threshold {
            self.state = CircuitBreakerState::Open;
            self.last_state_change_ms = current_timestamp_ms();
            warn!("🔴 Circuit breaker OPEN (failures: {})", self.failure_count);
        }
    }

    /// Record a success
    fn record_success(&mut self) {
        match self.state {
            CircuitBreakerState::HalfOpen => {
                // Recovery successful!
                self.state = CircuitBreakerState::Closed;
                self.failure_count = 0;
                self.last_state_change_ms = current_timestamp_ms();
                info!("🟢 Circuit breaker CLOSED (recovered)");
            }
            CircuitBreakerState::Closed => {
                // Already healthy, reset failure count
                self.failure_count = 0;
            }
            CircuitBreakerState::Open => {
                // Shouldn't happen (requests blocked), but reset anyway
                self.failure_count = 0;
            }
        }
    }

    /// Check if circuit breaker allows requests
    fn is_available(&mut self, recovery_timeout_ms: u64) -> bool {
        let now = current_timestamp_ms();

        match self.state {
            CircuitBreakerState::Closed => true,
            CircuitBreakerState::HalfOpen => true, // Allow test request
            CircuitBreakerState::Open => {
                // Check if recovery timeout elapsed
                if now - self.last_state_change_ms > recovery_timeout_ms {
                    self.state = CircuitBreakerState::HalfOpen;
                    self.last_state_change_ms = now;
                    info!("🟡 Circuit breaker HALF_OPEN (testing recovery)");
                    true
                } else {
                    false // Still blocked
                }
            }
        }
    }

    fn get_health(&self) -> WorkerHealth {
        match self.state {
            CircuitBreakerState::Closed => {
                if self.failure_count > 0 {
                    WorkerHealth::Degraded
                } else {
                    WorkerHealth::Healthy
                }
            }
            CircuitBreakerState::Open => WorkerHealth::Unhealthy,
            CircuitBreakerState::HalfOpen => WorkerHealth::Testing,
        }
    }
}

/// Failover manager for distributed inference
pub struct FailoverManager {
    /// Worker health tracking
    worker_health: Arc<RwLock<HashMap<String, CircuitBreaker>>>,

    /// Failure history per worker
    failure_history: Arc<RwLock<HashMap<String, Vec<FailureRecord>>>>,

    /// Retry attempts per request
    retry_attempts: Arc<RwLock<HashMap<String, Vec<String>>>>, // request_id -> [worker_ids]

    /// Configuration
    config: FailoverConfig,
}

/// Failover configuration
#[derive(Debug, Clone)]
pub struct FailoverConfig {
    /// Maximum retry attempts per request
    pub max_retries: usize,

    /// Request timeout (milliseconds)
    pub request_timeout_ms: u64,

    /// Circuit breaker: failures before opening
    pub failure_threshold: usize,

    /// Circuit breaker: recovery timeout (milliseconds)
    pub recovery_timeout_ms: u64,

    /// Keep failure history for N hours
    pub failure_history_hours: u64,
}

impl Default for FailoverConfig {
    fn default() -> Self {
        Self {
            max_retries: 3, // Retry up to 3 times
            request_timeout_ms: 30_000, // 30 seconds per attempt
            failure_threshold: 3, // Open circuit after 3 consecutive failures
            recovery_timeout_ms: 60_000, // Wait 60s before testing recovery
            failure_history_hours: 24, // Keep 24 hours of failure history
        }
    }
}

impl FailoverManager {
    /// Create new failover manager
    pub fn new(config: FailoverConfig) -> Self {
        info!("🔄 Initializing Failover Manager");
        info!("   Max retries: {}", config.max_retries);
        info!("   Request timeout: {}s", config.request_timeout_ms / 1000);
        info!("   Failure threshold: {}", config.failure_threshold);
        info!("   Recovery timeout: {}s", config.recovery_timeout_ms / 1000);

        Self {
            worker_health: Arc::new(RwLock::new(HashMap::new())),
            failure_history: Arc::new(RwLock::new(HashMap::new())),
            retry_attempts: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Record successful inference
    pub async fn record_success(&self, worker_node_id: &str, request_id: &str) {
        debug!("✅ Recording success: worker={}, request={}", worker_node_id, request_id);

        // Update circuit breaker
        let mut health_map = self.worker_health.write().await;
        let breaker = health_map.entry(worker_node_id.to_string())
            .or_insert_with(CircuitBreaker::new);
        breaker.record_success();
    }

    /// Record inference failure and determine if retry is needed
    pub async fn record_failure(
        &self,
        worker_node_id: &str,
        request_id: &str,
        failure_type: FailureType,
        error_message: String,
    ) -> FailoverDecision {
        error!("❌ Recording failure: worker={}, request={}, type={:?}",
               worker_node_id, request_id, failure_type);

        // Record failure in history
        let failure_record = FailureRecord {
            request_id: request_id.to_string(),
            failure_type: failure_type.clone(),
            failed_at_ms: current_timestamp_ms(),
            error_message: error_message.clone(),
        };

        {
            let mut history = self.failure_history.write().await;
            history.entry(worker_node_id.to_string())
                .or_insert_with(Vec::new)
                .push(failure_record);
        }

        // Update circuit breaker
        {
            let mut health_map = self.worker_health.write().await;
            let breaker = health_map.entry(worker_node_id.to_string())
                .or_insert_with(CircuitBreaker::new);
            breaker.record_failure(self.config.failure_threshold, self.config.recovery_timeout_ms);
        }

        // Check retry count
        let mut retries = self.retry_attempts.write().await;
        let attempts = retries.entry(request_id.to_string())
            .or_insert_with(Vec::new);
        attempts.push(worker_node_id.to_string());

        let retry_count = attempts.len();

        if retry_count >= self.config.max_retries {
            warn!("⚠️ Max retries ({}) reached for request {}", self.config.max_retries, request_id);
            FailoverDecision::GiveUp {
                request_id: request_id.to_string(),
                total_attempts: retry_count,
                last_error: error_message,
            }
        } else {
            info!("🔄 Retry attempt {}/{} for request {}", retry_count + 1, self.config.max_retries, request_id);
            FailoverDecision::Retry {
                request_id: request_id.to_string(),
                attempt: retry_count + 1,
                avoid_workers: attempts.clone(),
            }
        }
    }

    /// Check if worker is healthy enough to accept work
    pub async fn is_worker_available(&self, worker_node_id: &str) -> bool {
        let mut health_map = self.worker_health.write().await;
        let breaker = health_map.entry(worker_node_id.to_string())
            .or_insert_with(CircuitBreaker::new);

        breaker.is_available(self.config.recovery_timeout_ms)
    }

    /// Get worker health status
    pub async fn get_worker_health(&self, worker_node_id: &str) -> WorkerHealth {
        let health_map = self.worker_health.read().await;
        health_map.get(worker_node_id)
            .map(|b| b.get_health())
            .unwrap_or(WorkerHealth::Healthy) // New workers start healthy
    }

    /// Get failure history for worker
    pub async fn get_failure_history(&self, worker_node_id: &str) -> Vec<FailureRecord> {
        let history = self.failure_history.read().await;
        history.get(worker_node_id)
            .cloned()
            .unwrap_or_default()
    }

    /// Get all worker health statuses
    pub async fn get_all_worker_health(&self) -> HashMap<String, WorkerHealth> {
        let health_map = self.worker_health.read().await;
        health_map.iter()
            .map(|(id, breaker)| (id.clone(), breaker.get_health()))
            .collect()
    }

    /// Clean up old failure records
    pub async fn cleanup_old_failures(&self) {
        let now = current_timestamp_ms();
        let cutoff = now - (self.config.failure_history_hours * 3600 * 1000);

        let mut history = self.failure_history.write().await;
        for failures in history.values_mut() {
            failures.retain(|f| f.failed_at_ms > cutoff);
        }
    }

    /// Clean up completed request retry tracking
    pub async fn cleanup_request(&self, request_id: &str) {
        self.retry_attempts.write().await.remove(request_id);
    }
}

/// Decision on how to handle a failure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FailoverDecision {
    /// Retry with different worker
    Retry {
        request_id: String,
        attempt: usize,
        avoid_workers: Vec<String>, // Workers to avoid in retry
    },

    /// Give up after max retries
    GiveUp {
        request_id: String,
        total_attempts: usize,
        last_error: String,
    },
}

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
    async fn test_record_success() {
        let manager = FailoverManager::new(FailoverConfig::default());

        manager.record_success("worker-1", "req-1").await;

        let health = manager.get_worker_health("worker-1").await;
        assert_eq!(health, WorkerHealth::Healthy);
    }

    #[tokio::test]
    async fn test_circuit_breaker_opens_after_failures() {
        let mut config = FailoverConfig::default();
        config.failure_threshold = 2; // Open after 2 failures

        let manager = FailoverManager::new(config);

        // First failure
        let decision1 = manager.record_failure(
            "worker-1",
            "req-1",
            FailureType::Timeout,
            "Timeout 1".to_string()
        ).await;
        assert!(matches!(decision1, FailoverDecision::Retry { .. }));

        // Second failure - should open circuit
        let decision2 = manager.record_failure(
            "worker-1",
            "req-2",
            FailureType::Timeout,
            "Timeout 2".to_string()
        ).await;

        let health = manager.get_worker_health("worker-1").await;
        assert_eq!(health, WorkerHealth::Unhealthy);
    }

    #[tokio::test]
    async fn test_max_retries_gives_up() {
        let mut config = FailoverConfig::default();
        config.max_retries = 2;

        let manager = FailoverManager::new(config);

        // Retry 1
        let decision1 = manager.record_failure(
            "worker-1",
            "req-1",
            FailureType::Timeout,
            "Error 1".to_string()
        ).await;
        assert!(matches!(decision1, FailoverDecision::Retry { attempt: 1, .. }));

        // Retry 2 - max reached, give up
        let decision2 = manager.record_failure(
            "worker-2",
            "req-1",
            FailureType::Timeout,
            "Error 2".to_string()
        ).await;

        match decision2 {
            FailoverDecision::GiveUp { total_attempts, .. } => {
                assert_eq!(total_attempts, 2);
            }
            _ => panic!("Expected GiveUp decision"),
        }
    }
}
