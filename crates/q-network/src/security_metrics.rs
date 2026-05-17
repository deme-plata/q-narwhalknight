/// Security metrics for distributed AI signature verification
///
/// This module implements Week 2, Day 1-2 of the security implementation plan:
/// Prometheus metrics for monitoring signature verification, cache performance,
/// and DHT operations.
///
/// Metrics Overview:
/// - Signature verification: Total, failed, duration
/// - Cache performance: Hits, misses, evictions, size
/// - DHT operations: Announcements, fetches
/// - Circuit breaker: Failure counts, state

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

/// Security metrics collector for distributed AI
pub struct SecurityMetrics {
    /// Total signature verifications performed
    pub signature_verifications_total: AtomicU64,

    /// Failed signature verifications (invalid signatures)
    pub signature_verifications_failed: AtomicU64,

    /// Signature verification duration histogram (in microseconds)
    /// Stores last 1000 samples for percentile calculation
    pub signature_verification_durations: Arc<RwLock<Vec<u64>>>,

    /// Maximum histogram samples (prevents unbounded growth)
    pub max_histogram_samples: usize,

    /// Signature cache hits
    pub signature_cache_hits: AtomicU64,

    /// Signature cache misses
    pub signature_cache_misses: AtomicU64,

    /// Signature cache evictions (LRU)
    pub signature_cache_evictions: AtomicU64,

    /// Current signature cache size
    pub signature_cache_size: AtomicU64,

    /// DHT public key announcements sent
    pub dht_pubkey_announcements: AtomicU64,

    /// DHT public key fetches (successful)
    pub dht_pubkey_fetches_success: AtomicU64,

    /// DHT public key fetches (failed)
    pub dht_pubkey_fetches_failed: AtomicU64,

    /// Circuit breaker: Invalid signatures in last 5 minutes
    /// This is a rolling counter that decays over time
    pub circuit_breaker_failures_5min: Arc<RwLock<Vec<Instant>>>,

    /// Circuit breaker state: 0 = closed (normal), 1 = open (attack detected)
    pub circuit_breaker_state: AtomicU64,
}

impl SecurityMetrics {
    /// Create new security metrics collector
    pub fn new() -> Self {
        Self {
            signature_verifications_total: AtomicU64::new(0),
            signature_verifications_failed: AtomicU64::new(0),
            signature_verification_durations: Arc::new(RwLock::new(Vec::with_capacity(1000))),
            max_histogram_samples: 1000,
            signature_cache_hits: AtomicU64::new(0),
            signature_cache_misses: AtomicU64::new(0),
            signature_cache_evictions: AtomicU64::new(0),
            signature_cache_size: AtomicU64::new(0),
            dht_pubkey_announcements: AtomicU64::new(0),
            dht_pubkey_fetches_success: AtomicU64::new(0),
            dht_pubkey_fetches_failed: AtomicU64::new(0),
            circuit_breaker_failures_5min: Arc::new(RwLock::new(Vec::new())),
            circuit_breaker_state: AtomicU64::new(0), // 0 = closed
        }
    }

    /// Record a signature verification
    pub async fn record_signature_verification(&self, valid: bool, duration_micros: u64) {
        // Increment total counter
        self.signature_verifications_total.fetch_add(1, Ordering::Relaxed);

        // Increment failed counter if invalid
        if !valid {
            self.signature_verifications_failed.fetch_add(1, Ordering::Relaxed);

            // Add to circuit breaker failure tracker
            let mut failures = self.circuit_breaker_failures_5min.write().await;
            failures.push(Instant::now());

            // Remove failures older than 5 minutes
            failures.retain(|&instant| instant.elapsed().as_secs() < 300);
        }

        // Record duration in histogram
        let mut durations = self.signature_verification_durations.write().await;
        durations.push(duration_micros);

        // Trim histogram if too large (keep most recent samples)
        if durations.len() > self.max_histogram_samples {
            let trim_count = durations.len() - self.max_histogram_samples;
            durations.drain(0..trim_count);
        }
    }

    /// Record a signature cache hit
    pub fn record_cache_hit(&self) {
        self.signature_cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a signature cache miss
    pub fn record_cache_miss(&self) {
        self.signature_cache_misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a cache eviction
    pub fn record_cache_eviction(&self) {
        self.signature_cache_evictions.fetch_add(1, Ordering::Relaxed);
    }

    /// Update current cache size
    pub fn set_cache_size(&self, size: u64) {
        self.signature_cache_size.store(size, Ordering::Relaxed);
    }

    /// Record a DHT public key announcement
    pub fn record_dht_announcement(&self) {
        self.dht_pubkey_announcements.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a DHT public key fetch
    pub fn record_dht_fetch(&self, success: bool) {
        if success {
            self.dht_pubkey_fetches_success.fetch_add(1, Ordering::Relaxed);
        } else {
            self.dht_pubkey_fetches_failed.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Get current circuit breaker failure count (last 5 minutes)
    pub async fn get_circuit_breaker_failures(&self) -> usize {
        let mut failures = self.circuit_breaker_failures_5min.write().await;

        // Clean up old entries
        failures.retain(|&instant| instant.elapsed().as_secs() < 300);

        failures.len()
    }

    /// Open circuit breaker (emergency stop on attack)
    pub fn open_circuit_breaker(&self) {
        self.circuit_breaker_state.store(1, Ordering::Relaxed);
    }

    /// Close circuit breaker (resume normal operation)
    pub fn close_circuit_breaker(&self) {
        self.circuit_breaker_state.store(0, Ordering::Relaxed);
    }

    /// Check if circuit breaker is open
    pub fn is_circuit_breaker_open(&self) -> bool {
        self.circuit_breaker_state.load(Ordering::Relaxed) == 1
    }

    /// Get signature verification statistics
    pub async fn get_signature_stats(&self) -> SignatureVerificationStats {
        let total = self.signature_verifications_total.load(Ordering::Relaxed);
        let failed = self.signature_verifications_failed.load(Ordering::Relaxed);

        let durations = self.signature_verification_durations.read().await;
        let (p50, p95, p99) = if !durations.is_empty() {
            let mut sorted = durations.clone();
            sorted.sort_unstable();

            let p50_idx = (sorted.len() as f64 * 0.50) as usize;
            let p95_idx = (sorted.len() as f64 * 0.95) as usize;
            let p99_idx = (sorted.len() as f64 * 0.99) as usize;

            (
                sorted[p50_idx.min(sorted.len() - 1)],
                sorted[p95_idx.min(sorted.len() - 1)],
                sorted[p99_idx.min(sorted.len() - 1)],
            )
        } else {
            (0, 0, 0)
        };

        SignatureVerificationStats {
            total_verifications: total,
            failed_verifications: failed,
            success_rate: if total > 0 {
                ((total - failed) as f64 / total as f64) * 100.0
            } else {
                100.0
            },
            duration_p50_micros: p50,
            duration_p95_micros: p95,
            duration_p99_micros: p99,
        }
    }

    /// Get cache performance statistics
    pub fn get_cache_stats(&self) -> CachePerformanceStats {
        let hits = self.signature_cache_hits.load(Ordering::Relaxed);
        let misses = self.signature_cache_misses.load(Ordering::Relaxed);
        let total = hits + misses;

        CachePerformanceStats {
            cache_hits: hits,
            cache_misses: misses,
            cache_hit_rate: if total > 0 {
                (hits as f64 / total as f64) * 100.0
            } else {
                0.0
            },
            cache_evictions: self.signature_cache_evictions.load(Ordering::Relaxed),
            cache_size: self.signature_cache_size.load(Ordering::Relaxed),
        }
    }

    /// Get DHT operation statistics
    pub fn get_dht_stats(&self) -> DhtOperationStats {
        let success = self.dht_pubkey_fetches_success.load(Ordering::Relaxed);
        let failed = self.dht_pubkey_fetches_failed.load(Ordering::Relaxed);
        let total = success + failed;

        DhtOperationStats {
            announcements: self.dht_pubkey_announcements.load(Ordering::Relaxed),
            fetches_success: success,
            fetches_failed: failed,
            fetch_success_rate: if total > 0 {
                (success as f64 / total as f64) * 100.0
            } else {
                100.0
            },
        }
    }

    /// Generate Prometheus-compatible metrics output
    pub async fn to_prometheus_format(&self) -> String {
        let sig_stats = self.get_signature_stats().await;
        let cache_stats = self.get_cache_stats();
        let dht_stats = self.get_dht_stats();
        let cb_failures = self.get_circuit_breaker_failures().await;
        let cb_state = if self.is_circuit_breaker_open() { 1 } else { 0 };

        format!(
            r#"# HELP signature_verifications_total Total number of signature verifications
# TYPE signature_verifications_total counter
signature_verifications_total {{result="valid"}} {}
signature_verifications_total {{result="invalid"}} {}

# HELP signature_verification_duration_seconds Signature verification duration histogram
# TYPE signature_verification_duration_seconds histogram
signature_verification_duration_seconds {{quantile="0.5"}} {}
signature_verification_duration_seconds {{quantile="0.95"}} {}
signature_verification_duration_seconds {{quantile="0.99"}} {}

# HELP signature_cache_hits_total Total signature cache hits
# TYPE signature_cache_hits_total counter
signature_cache_hits_total {}

# HELP signature_cache_misses_total Total signature cache misses
# TYPE signature_cache_misses_total counter
signature_cache_misses_total {}

# HELP signature_cache_evictions_total Total signature cache evictions
# TYPE signature_cache_evictions_total counter
signature_cache_evictions_total {}

# HELP signature_cache_size Current signature cache size
# TYPE signature_cache_size gauge
signature_cache_size {}

# HELP dht_pubkey_announcements_total Total DHT public key announcements
# TYPE dht_pubkey_announcements_total counter
dht_pubkey_announcements_total {}

# HELP dht_pubkey_fetches_total Total DHT public key fetches
# TYPE dht_pubkey_fetches_total counter
dht_pubkey_fetches_total {{result="success"}} {}
dht_pubkey_fetches_total {{result="error"}} {}

# HELP circuit_breaker_failures_5min Invalid signatures in last 5 minutes
# TYPE circuit_breaker_failures_5min gauge
circuit_breaker_failures_5min {}

# HELP circuit_breaker_state Circuit breaker state (0=closed, 1=open)
# TYPE circuit_breaker_state gauge
circuit_breaker_state {}
"#,
            sig_stats.total_verifications - sig_stats.failed_verifications,
            sig_stats.failed_verifications,
            sig_stats.duration_p50_micros as f64 / 1_000_000.0,
            sig_stats.duration_p95_micros as f64 / 1_000_000.0,
            sig_stats.duration_p99_micros as f64 / 1_000_000.0,
            cache_stats.cache_hits,
            cache_stats.cache_misses,
            cache_stats.cache_evictions,
            cache_stats.cache_size,
            dht_stats.announcements,
            dht_stats.fetches_success,
            dht_stats.fetches_failed,
            cb_failures,
            cb_state,
        )
    }
}

impl Default for SecurityMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Signature verification statistics
#[derive(Debug, Clone)]
pub struct SignatureVerificationStats {
    pub total_verifications: u64,
    pub failed_verifications: u64,
    pub success_rate: f64, // 0-100%
    pub duration_p50_micros: u64,
    pub duration_p95_micros: u64,
    pub duration_p99_micros: u64,
}

/// Cache performance statistics
#[derive(Debug, Clone)]
pub struct CachePerformanceStats {
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_hit_rate: f64, // 0-100%
    pub cache_evictions: u64,
    pub cache_size: u64,
}

/// DHT operation statistics
#[derive(Debug, Clone)]
pub struct DhtOperationStats {
    pub announcements: u64,
    pub fetches_success: u64,
    pub fetches_failed: u64,
    pub fetch_success_rate: f64, // 0-100%
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_signature_verification_tracking() {
        let metrics = SecurityMetrics::new();

        // Record some verifications
        metrics.record_signature_verification(true, 50).await;
        metrics.record_signature_verification(true, 60).await;
        metrics.record_signature_verification(false, 100).await;

        let stats = metrics.get_signature_stats().await;
        assert_eq!(stats.total_verifications, 3);
        assert_eq!(stats.failed_verifications, 1);
        assert_eq!(stats.success_rate, 66.66666666666666);
    }

    #[tokio::test]
    async fn test_cache_performance_tracking() {
        let metrics = SecurityMetrics::new();

        // Record cache operations
        metrics.record_cache_hit();
        metrics.record_cache_hit();
        metrics.record_cache_hit();
        metrics.record_cache_hit();
        metrics.record_cache_miss();

        let stats = metrics.get_cache_stats();
        assert_eq!(stats.cache_hits, 4);
        assert_eq!(stats.cache_misses, 1);
        assert_eq!(stats.cache_hit_rate, 80.0);
    }

    #[tokio::test]
    async fn test_dht_operation_tracking() {
        let metrics = SecurityMetrics::new();

        // Record DHT operations
        metrics.record_dht_announcement();
        metrics.record_dht_announcement();
        metrics.record_dht_fetch(true);
        metrics.record_dht_fetch(true);
        metrics.record_dht_fetch(false);

        let stats = metrics.get_dht_stats();
        assert_eq!(stats.announcements, 2);
        assert_eq!(stats.fetches_success, 2);
        assert_eq!(stats.fetches_failed, 1);
        assert_eq!(stats.fetch_success_rate, 66.66666666666666);
    }

    #[tokio::test]
    async fn test_circuit_breaker_failures() {
        let metrics = SecurityMetrics::new();

        // Record some failures
        metrics.record_signature_verification(false, 100).await;
        metrics.record_signature_verification(false, 100).await;
        metrics.record_signature_verification(false, 100).await;

        let failures = metrics.get_circuit_breaker_failures().await;
        assert_eq!(failures, 3);
    }

    #[tokio::test]
    async fn test_circuit_breaker_state() {
        let metrics = SecurityMetrics::new();

        assert!(!metrics.is_circuit_breaker_open());

        metrics.open_circuit_breaker();
        assert!(metrics.is_circuit_breaker_open());

        metrics.close_circuit_breaker();
        assert!(!metrics.is_circuit_breaker_open());
    }

    #[tokio::test]
    async fn test_prometheus_format() {
        let metrics = SecurityMetrics::new();

        // Record some data
        metrics.record_signature_verification(true, 50).await;
        metrics.record_signature_verification(false, 100).await;
        metrics.record_cache_hit();
        metrics.record_cache_miss();
        metrics.record_dht_announcement();

        let prometheus_output = metrics.to_prometheus_format().await;

        // Verify it contains expected metrics
        assert!(prometheus_output.contains("signature_verifications_total"));
        assert!(prometheus_output.contains("signature_cache_hits_total"));
        assert!(prometheus_output.contains("dht_pubkey_announcements_total"));
        assert!(prometheus_output.contains("circuit_breaker_state"));
    }

    #[tokio::test]
    async fn test_duration_percentiles() {
        let metrics = SecurityMetrics::new();

        // Record durations: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 microseconds
        for i in 1..=10 {
            metrics.record_signature_verification(true, i * 10).await;
        }

        let stats = metrics.get_signature_stats().await;

        // p50 should be around 50-60μs
        assert!(stats.duration_p50_micros >= 40 && stats.duration_p50_micros <= 60);

        // p95 should be around 90-100μs
        assert!(stats.duration_p95_micros >= 80 && stats.duration_p95_micros <= 100);

        // p99 should be around 100μs
        assert!(stats.duration_p99_micros >= 90 && stats.duration_p99_micros <= 100);
    }
}
