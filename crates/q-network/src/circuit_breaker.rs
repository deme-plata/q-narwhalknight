/// Circuit breaker for signature verification attack protection
///
/// This module implements Week 2, Day 3-4 of the security implementation plan:
/// Circuit breaker that automatically halts message processing when under attack.
///
/// Attack Detection:
/// - Threshold: 100 invalid signatures in 5 minutes
/// - Action: Open circuit breaker (halt all message processing)
/// - Recovery: Automatic decay on successful verifications
///
/// Security Impact:
/// - Prevents resource exhaustion during signature spam attacks
/// - Limits damage from coordinated forgery attempts
/// - Enables graceful degradation under attack

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{error, info, warn};

use super::security_metrics::SecurityMetrics;

/// Circuit breaker states
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitBreakerState {
    /// Normal operation - all messages processed
    Closed,

    /// Half-open - testing if attack has subsided
    /// Allows limited message processing to test recovery
    HalfOpen,

    /// Attack detected - emergency stop
    /// Rejects all messages immediately without verification
    Open,
}

/// Circuit breaker configuration
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Failure threshold to open circuit breaker
    /// Default: 100 invalid signatures in 5 minutes
    pub failure_threshold: usize,

    /// Time window for counting failures (seconds)
    /// Default: 300 seconds (5 minutes)
    pub failure_window_secs: u64,

    /// Number of successful verifications needed to close circuit
    /// Default: 10 consecutive successes
    pub success_threshold: usize,

    /// Timeout before attempting half-open state (seconds)
    /// Default: 60 seconds
    pub open_timeout_secs: u64,

    /// Maximum number of test requests in half-open state
    /// Default: 5 requests
    pub half_open_max_requests: usize,

    /// Decay rate for failures (per successful verification)
    /// Default: 0.1 (10% decay per success)
    pub decay_rate: f64,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 100,
            failure_window_secs: 300, // 5 minutes
            success_threshold: 10,
            open_timeout_secs: 60,
            half_open_max_requests: 5,
            decay_rate: 0.1,
        }
    }
}

/// Circuit breaker for signature verification
pub struct CircuitBreaker {
    /// Current circuit breaker state
    state: Arc<RwLock<CircuitBreakerState>>,

    /// Configuration
    config: CircuitBreakerConfig,

    /// Failure timestamps (for decay calculation)
    failures: Arc<RwLock<Vec<Instant>>>,

    /// Consecutive successes (for recovery)
    consecutive_successes: Arc<AtomicU64>,

    /// Time when circuit was opened
    opened_at: Arc<RwLock<Option<Instant>>>,

    /// Test requests in half-open state
    half_open_requests: Arc<AtomicU64>,

    /// Security metrics (for reporting)
    metrics: Arc<SecurityMetrics>,
}

impl CircuitBreaker {
    /// Create new circuit breaker with default config
    pub fn new(metrics: Arc<SecurityMetrics>) -> Self {
        Self::new_with_config(CircuitBreakerConfig::default(), metrics)
    }

    /// Create new circuit breaker with custom config
    pub fn new_with_config(config: CircuitBreakerConfig, metrics: Arc<SecurityMetrics>) -> Self {
        Self {
            state: Arc::new(RwLock::new(CircuitBreakerState::Closed)),
            config,
            failures: Arc::new(RwLock::new(Vec::new())),
            consecutive_successes: Arc::new(AtomicU64::new(0)),
            opened_at: Arc::new(RwLock::new(None)),
            half_open_requests: Arc::new(AtomicU64::new(0)),
            metrics,
        }
    }

    /// Check if message should be processed
    ///
    /// Returns:
    /// - Ok(true): Process message (circuit closed or half-open with capacity)
    /// - Ok(false): Reject message (circuit open)
    pub async fn should_process(&self) -> bool {
        let state = *self.state.read().await;

        match state {
            CircuitBreakerState::Closed => true,

            CircuitBreakerState::HalfOpen => {
                // Allow limited requests in half-open state
                let current = self.half_open_requests.fetch_add(1, Ordering::Relaxed);
                if current < self.config.half_open_max_requests as u64 {
                    true
                } else {
                    warn!("🔶 Circuit breaker: Half-open capacity exceeded, rejecting message");
                    false
                }
            }

            CircuitBreakerState::Open => {
                // Check if timeout has elapsed to transition to half-open
                let opened_at = self.opened_at.read().await;
                if let Some(opened_time) = *opened_at {
                    if opened_time.elapsed().as_secs() >= self.config.open_timeout_secs {
                        drop(opened_at); // Release read lock
                        warn!("🔶 Circuit breaker: Timeout elapsed, transitioning to half-open");
                        self.transition_to_half_open().await;
                        return true; // Allow first test request
                    }
                }
                false
            }
        }
    }

    /// Record a verification result
    ///
    /// This method:
    /// 1. Updates failure/success counters
    /// 2. Applies decay on success
    /// 3. Checks thresholds
    /// 4. Transitions states as needed
    pub async fn record_verification(&self, valid: bool) {
        if valid {
            self.record_success().await;
        } else {
            self.record_failure().await;
        }
    }

    /// Record a successful verification
    async fn record_success(&self) {
        // Increment consecutive successes
        let successes = self.consecutive_successes.fetch_add(1, Ordering::Relaxed) + 1;

        // Apply decay to failure count
        self.apply_decay().await;

        let state = *self.state.read().await;

        match state {
            CircuitBreakerState::HalfOpen => {
                // Check if enough successes to close circuit
                if successes >= self.config.success_threshold as u64 {
                    info!("✅ Circuit breaker: {} consecutive successes, closing circuit", successes);
                    self.close_circuit().await;
                }
            }

            CircuitBreakerState::Open => {
                // Should not receive verifications in open state
                warn!("⚠️  Circuit breaker: Received success in open state (unexpected)");
            }

            CircuitBreakerState::Closed => {
                // Normal operation, no action needed
            }
        }
    }

    /// Record a failed verification
    async fn record_failure(&self) {
        // Reset consecutive successes
        self.consecutive_successes.store(0, Ordering::Relaxed);

        // Add failure timestamp
        let mut failures = self.failures.write().await;
        failures.push(Instant::now());

        // Clean up old failures (outside time window)
        failures.retain(|&instant| {
            instant.elapsed().as_secs() < self.config.failure_window_secs
        });

        let failure_count = failures.len();
        drop(failures); // Release lock

        // Check if threshold exceeded
        if failure_count >= self.config.failure_threshold {
            let state = *self.state.read().await;
            if state != CircuitBreakerState::Open {
                error!("🚨 CRITICAL: Circuit breaker threshold exceeded!");
                error!("   Failures in last {} seconds: {}", self.config.failure_window_secs, failure_count);
                error!("   Threshold: {}", self.config.failure_threshold);
                error!("   Opening circuit breaker - halting message processing");
                drop(state); // Release read lock
                self.open_circuit().await;
            }
        } else if failure_count >= (self.config.failure_threshold * 3 / 4) {
            // Warning at 75% threshold
            warn!("⚠️  Circuit breaker: High failure rate detected");
            warn!("   Failures: {}/{}", failure_count, self.config.failure_threshold);
            warn!("   System under potential attack");
        }
    }

    /// Apply decay to failure count on successful verification
    async fn apply_decay(&self) {
        let mut failures = self.failures.write().await;

        if failures.is_empty() {
            return;
        }

        // Calculate how many failures to remove based on decay rate
        let decay_count = (failures.len() as f64 * self.config.decay_rate).ceil() as usize;

        if decay_count > 0 {
            // Remove oldest failures (FIFO decay)
            let remove_count = decay_count.min(failures.len());
            failures.drain(0..remove_count);
        }
    }

    /// Open circuit breaker (emergency stop)
    async fn open_circuit(&self) {
        let mut state = self.state.write().await;
        *state = CircuitBreakerState::Open;
        drop(state);

        // Record open time
        let mut opened_at = self.opened_at.write().await;
        *opened_at = Some(Instant::now());
        drop(opened_at);

        // Reset half-open counter
        self.half_open_requests.store(0, Ordering::Relaxed);

        // Update metrics
        self.metrics.open_circuit_breaker();

        error!("🚨 CIRCUIT BREAKER OPENED - Message processing halted");
        error!("   All incoming messages will be rejected for {} seconds", self.config.open_timeout_secs);
        error!("   This is an emergency protective measure");
    }

    /// Transition to half-open state (testing recovery)
    async fn transition_to_half_open(&self) {
        let mut state = self.state.write().await;
        *state = CircuitBreakerState::HalfOpen;
        drop(state);

        // Reset counters
        self.consecutive_successes.store(0, Ordering::Relaxed);
        self.half_open_requests.store(0, Ordering::Relaxed);

        warn!("🔶 Circuit breaker: Transitioning to HALF-OPEN");
        warn!("   Allowing {} test requests", self.config.half_open_max_requests);
        warn!("   Need {} consecutive successes to close circuit", self.config.success_threshold);
    }

    /// Close circuit breaker (resume normal operation)
    async fn close_circuit(&self) {
        let mut state = self.state.write().await;
        *state = CircuitBreakerState::Closed;
        drop(state);

        // Clear opened timestamp
        let mut opened_at = self.opened_at.write().await;
        *opened_at = None;
        drop(opened_at);

        // Reset counters
        self.consecutive_successes.store(0, Ordering::Relaxed);
        self.half_open_requests.store(0, Ordering::Relaxed);

        // Clear failures
        let mut failures = self.failures.write().await;
        failures.clear();
        drop(failures);

        // Update metrics
        self.metrics.close_circuit_breaker();

        info!("✅ Circuit breaker CLOSED - Resuming normal operation");
        info!("   Message processing restored");
    }

    /// Get current circuit breaker state
    pub async fn get_state(&self) -> CircuitBreakerState {
        *self.state.read().await
    }

    /// Get circuit breaker statistics
    pub async fn get_stats(&self) -> CircuitBreakerStats {
        let state = *self.state.read().await;
        let failures = self.failures.read().await;
        let failure_count = failures.len();
        drop(failures);

        let consecutive_successes = self.consecutive_successes.load(Ordering::Relaxed);
        let opened_at = self.opened_at.read().await;
        let time_in_open_state = opened_at.as_ref().map(|t| t.elapsed());
        drop(opened_at);

        CircuitBreakerStats {
            state,
            failure_count,
            failure_threshold: self.config.failure_threshold,
            consecutive_successes,
            success_threshold: self.config.success_threshold,
            time_in_open_state,
        }
    }

    /// Force close circuit breaker (manual override)
    pub async fn force_close(&self) {
        warn!("⚠️  Manual circuit breaker override: Forcing close");
        self.close_circuit().await;
    }

    /// Force open circuit breaker (manual emergency stop)
    pub async fn force_open(&self) {
        error!("🚨 Manual circuit breaker override: Forcing open");
        self.open_circuit().await;
    }
}

/// Circuit breaker statistics
#[derive(Debug, Clone)]
pub struct CircuitBreakerStats {
    pub state: CircuitBreakerState,
    pub failure_count: usize,
    pub failure_threshold: usize,
    pub consecutive_successes: u64,
    pub success_threshold: usize,
    pub time_in_open_state: Option<Duration>,
}

impl CircuitBreakerStats {
    /// Calculate failure percentage relative to threshold
    pub fn failure_percentage(&self) -> f64 {
        (self.failure_count as f64 / self.failure_threshold as f64) * 100.0
    }

    /// Check if circuit breaker is healthy
    pub fn is_healthy(&self) -> bool {
        self.state == CircuitBreakerState::Closed && self.failure_count < self.failure_threshold / 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn create_test_breaker() -> CircuitBreaker {
        let metrics = Arc::new(SecurityMetrics::new());
        let config = CircuitBreakerConfig {
            failure_threshold: 5, // Low threshold for testing
            failure_window_secs: 10,
            success_threshold: 3,
            open_timeout_secs: 2,
            half_open_max_requests: 2,
            decay_rate: 0.2,
        };
        CircuitBreaker::new_with_config(config, metrics)
    }

    #[tokio::test]
    async fn test_circuit_breaker_starts_closed() {
        let breaker = create_test_breaker().await;
        assert_eq!(breaker.get_state().await, CircuitBreakerState::Closed);
        assert!(breaker.should_process().await);
    }

    #[tokio::test]
    async fn test_circuit_opens_on_threshold() {
        let breaker = create_test_breaker().await;

        // Record failures below threshold
        for _ in 0..4 {
            breaker.record_verification(false).await;
        }

        assert_eq!(breaker.get_state().await, CircuitBreakerState::Closed);

        // Record failure that exceeds threshold
        breaker.record_verification(false).await;

        assert_eq!(breaker.get_state().await, CircuitBreakerState::Open);
        assert!(!breaker.should_process().await);
    }

    #[tokio::test]
    async fn test_circuit_transitions_to_half_open() {
        let breaker = create_test_breaker().await;

        // Open circuit
        for _ in 0..5 {
            breaker.record_verification(false).await;
        }
        assert_eq!(breaker.get_state().await, CircuitBreakerState::Open);

        // Wait for timeout
        tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;

        // Should transition to half-open on next check
        assert!(breaker.should_process().await);
        assert_eq!(breaker.get_state().await, CircuitBreakerState::HalfOpen);
    }

    #[tokio::test]
    async fn test_circuit_closes_on_success() {
        let breaker = create_test_breaker().await;

        // Open circuit
        for _ in 0..5 {
            breaker.record_verification(false).await;
        }

        // Wait and transition to half-open
        tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;
        breaker.should_process().await;

        // Record successful verifications
        for _ in 0..3 {
            breaker.record_verification(true).await;
        }

        assert_eq!(breaker.get_state().await, CircuitBreakerState::Closed);
        assert!(breaker.should_process().await);
    }

    #[tokio::test]
    async fn test_decay_on_success() {
        let breaker = create_test_breaker().await;

        // Record 3 failures
        for _ in 0..3 {
            breaker.record_verification(false).await;
        }

        let stats = breaker.get_stats().await;
        assert_eq!(stats.failure_count, 3);

        // Record success (should trigger decay)
        breaker.record_verification(true).await;

        let stats = breaker.get_stats().await;
        // With decay_rate=0.2, should remove 1 failure (20% of 3 = 0.6, ceil to 1)
        assert_eq!(stats.failure_count, 2);
    }

    #[tokio::test]
    async fn test_half_open_request_limit() {
        let breaker = create_test_breaker().await;

        // Open circuit
        for _ in 0..5 {
            breaker.record_verification(false).await;
        }

        // Wait and transition to half-open
        tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;

        // Should allow max_requests (2 in test config)
        assert!(breaker.should_process().await); // 1st request
        assert!(breaker.should_process().await); // 2nd request
        assert!(!breaker.should_process().await); // 3rd request rejected
    }

    #[tokio::test]
    async fn test_force_close() {
        let breaker = create_test_breaker().await;

        // Open circuit
        for _ in 0..5 {
            breaker.record_verification(false).await;
        }
        assert_eq!(breaker.get_state().await, CircuitBreakerState::Open);

        // Force close
        breaker.force_close().await;
        assert_eq!(breaker.get_state().await, CircuitBreakerState::Closed);
        assert!(breaker.should_process().await);
    }

    #[tokio::test]
    async fn test_stats_calculation() {
        let breaker = create_test_breaker().await;

        // Record 3 failures (threshold is 5)
        for _ in 0..3 {
            breaker.record_verification(false).await;
        }

        let stats = breaker.get_stats().await;
        assert_eq!(stats.failure_count, 3);
        assert_eq!(stats.failure_threshold, 5);
        assert_eq!(stats.failure_percentage(), 60.0);
        assert!(stats.is_healthy()); // 3 < 5/2, so healthy
    }

    #[tokio::test]
    async fn test_consecutive_success_reset_on_failure() {
        let breaker = create_test_breaker().await;

        // Record successes
        breaker.record_verification(true).await;
        breaker.record_verification(true).await;

        let stats = breaker.get_stats().await;
        assert_eq!(stats.consecutive_successes, 2);

        // Record failure (should reset)
        breaker.record_verification(false).await;

        let stats = breaker.get_stats().await;
        assert_eq!(stats.consecutive_successes, 0);
    }
}
