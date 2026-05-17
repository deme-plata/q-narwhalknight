use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

// ─── Latency Histogram (Issue #11) ────────────────────────────────────────

/// Lock-free latency histogram with fixed buckets.
/// All operations are atomic — safe to call from any thread.
#[derive(Debug)]
pub struct LatencyHistogram {
    /// Upper bounds in microseconds. The last bucket is +Inf.
    bounds_us: [u64; 10],
    /// Counts per bucket (cumulative — each bucket includes all smaller).
    buckets: [AtomicU64; 10],
    /// Total observations.
    pub count: AtomicU64,
    /// Sum of all observed latencies in microseconds.
    pub sum_us: AtomicU64,
}

impl Default for LatencyHistogram {
    fn default() -> Self {
        Self::new()
    }
}

impl LatencyHistogram {
    pub fn new() -> Self {
        Self {
            // 1ms, 5ms, 10ms, 25ms, 50ms, 100ms, 250ms, 500ms, 1s, 5s
            bounds_us: [1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000, 5_000_000],
            buckets: std::array::from_fn(|_| AtomicU64::new(0)),
            count: AtomicU64::new(0),
            sum_us: AtomicU64::new(0),
        }
    }

    /// Record a latency observation.
    #[inline]
    pub fn observe(&self, duration: Duration) {
        let us = duration.as_micros() as u64;
        self.sum_us.fetch_add(us, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
        for (i, &bound) in self.bounds_us.iter().enumerate() {
            if us <= bound {
                self.buckets[i].fetch_add(1, Ordering::Relaxed);
                return;
            }
        }
        // Exceeds all buckets — falls into +Inf (last bucket)
        self.buckets[9].fetch_add(1, Ordering::Relaxed);
    }

    /// Estimate a percentile (0.0–1.0) from the histogram using linear interpolation.
    /// Returns the estimated latency in seconds.
    pub fn percentile_seconds(&self, p: f64) -> f64 {
        let total = self.count.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        let target = (total as f64 * p).ceil() as u64;

        // Collect non-cumulative bucket counts
        let mut counts = [0u64; 10];
        for i in 0..10 {
            counts[i] = self.buckets[i].load(Ordering::Relaxed);
        }

        // Walk buckets to find which one contains the target observation
        let mut cumulative = 0u64;
        let lower_bounds_us: [u64; 10] = [0, 1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000];
        for i in 0..10 {
            cumulative += counts[i];
            if cumulative >= target {
                // Linear interpolation within this bucket
                let prev_cum = cumulative - counts[i];
                let lower = lower_bounds_us[i] as f64;
                let upper = self.bounds_us[i] as f64;
                let fraction = if counts[i] > 0 {
                    (target - prev_cum) as f64 / counts[i] as f64
                } else {
                    0.5
                };
                return (lower + fraction * (upper - lower)) / 1_000_000.0;
            }
        }
        // Beyond last bucket
        5.0
    }

    /// Format as Prometheus histogram lines.
    pub fn prometheus(&self, name: &str) -> String {
        let mut out = format!(
            "# HELP {name} Request latency in seconds\n# TYPE {name} histogram\n"
        );
        let mut cumulative = 0u64;
        let labels = ["0.001", "0.005", "0.01", "0.025", "0.05", "0.1", "0.25", "0.5", "1", "5"];
        for (i, label) in labels.iter().enumerate() {
            cumulative += self.buckets[i].load(Ordering::Relaxed);
            out.push_str(&format!("{name}_bucket{{le=\"{label}\"}} {cumulative}\n"));
        }
        let total = self.count.load(Ordering::Relaxed);
        out.push_str(&format!("{name}_bucket{{le=\"+Inf\"}} {total}\n"));
        let sum_s = self.sum_us.load(Ordering::Relaxed) as f64 / 1_000_000.0;
        out.push_str(&format!("{name}_sum {sum_s:.6}\n"));
        out.push_str(&format!("{name}_count {total}\n"));
        out
    }
}

// ─── Token Bucket Rate Limiter (Issue #12) ─────────────────────────────────

/// Per-IP token bucket rate limiter.
/// Lock-free: uses atomic CAS for token consumption.
#[derive(Debug)]
pub struct TokenBucket {
    /// Available tokens × 1000 (fixed-point for sub-token precision).
    tokens: AtomicU64,
    /// Last refill timestamp in microseconds.
    last_refill_us: AtomicU64,
    /// Tokens per second × 1000.
    rate_milli: u64,
    /// Max tokens × 1000.
    capacity_milli: u64,
}

impl TokenBucket {
    pub fn new(rate_per_sec: u64, burst: u64) -> Self {
        Self {
            tokens: AtomicU64::new(burst * 1000),
            last_refill_us: AtomicU64::new(0),
            rate_milli: rate_per_sec * 1000,
            capacity_milli: burst * 1000,
        }
    }

    /// Try to consume one token. Returns true if allowed, false if rate-limited.
    #[inline]
    pub fn try_acquire(&self, now_us: u64) -> bool {
        // Refill tokens based on elapsed time
        let last = self.last_refill_us.load(Ordering::Relaxed);
        let elapsed_us = now_us.saturating_sub(last);
        if elapsed_us > 1000 {
            // More than 1ms elapsed — refill
            let new_tokens = (elapsed_us as u128 * self.rate_milli as u128 / 1_000_000) as u64;
            if new_tokens > 0 {
                // CAS the timestamp to claim this refill window
                if self.last_refill_us.compare_exchange_weak(
                    last, now_us, Ordering::AcqRel, Ordering::Relaxed
                ).is_ok() {
                    // Atomically add tokens with CAS loop to avoid overwriting
                    // concurrent consumption
                    loop {
                        let current = self.tokens.load(Ordering::Relaxed);
                        let refilled = (current + new_tokens).min(self.capacity_milli);
                        if self.tokens.compare_exchange_weak(
                            current, refilled, Ordering::AcqRel, Ordering::Relaxed
                        ).is_ok() {
                            break;
                        }
                    }
                }
            }
        }

        // Try to consume one token (1000 milli-tokens)
        loop {
            let current = self.tokens.load(Ordering::Relaxed);
            if current < 1000 {
                return false; // Rate limited
            }
            if self.tokens.compare_exchange_weak(
                current, current - 1000, Ordering::AcqRel, Ordering::Relaxed
            ).is_ok() {
                return true;
            }
        }
    }
}

/// Per-IP rate limiter using DashMap of token buckets.
pub struct RateLimiter {
    buckets: dashmap::DashMap<std::net::IpAddr, TokenBucket>,
    rate_per_sec: u64,
    burst: u64,
    global: TokenBucket,
}

impl RateLimiter {
    pub fn new(rate_per_ip: u64, burst_per_ip: u64, global_rps: u64) -> Self {
        Self {
            buckets: dashmap::DashMap::with_capacity(1024),
            rate_per_sec: rate_per_ip,
            burst: burst_per_ip,
            global: TokenBucket::new(global_rps, global_rps * 2),
        }
    }

    /// Check if a request from this IP is allowed.
    pub fn check(&self, ip: std::net::IpAddr) -> bool {
        let now_us = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        // Global limit first
        if !self.global.try_acquire(now_us) {
            return false;
        }

        // Per-IP limit
        let bucket = self.buckets.entry(ip).or_insert_with(|| {
            TokenBucket::new(self.rate_per_sec, self.burst)
        });
        bucket.try_acquire(now_us)
    }

    /// Remove stale entries (IPs not seen recently). Call periodically.
    pub fn cleanup(&self) {
        let now_us = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;
        // Remove entries idle for >5 minutes
        let stale_threshold = 5 * 60 * 1_000_000u64;
        self.buckets.retain(|_, bucket| {
            now_us.saturating_sub(bucket.last_refill_us.load(Ordering::Relaxed)) < stale_threshold
        });
    }

    #[allow(dead_code)] // Public API for admin/monitoring
    pub fn active_ips(&self) -> usize {
        self.buckets.len()
    }
}

// ─── Core Metrics ──────────────────────────────────────────────────────────

/// Global metrics shared across all workers.
#[derive(Debug, Clone)]
pub struct Metrics {
    inner: Arc<MetricsInner>,
}

#[derive(Debug)]
struct MetricsInner {
    start_time: Instant,
    // Connection counters
    pub active_connections: AtomicU64,
    pub total_connections: AtomicU64,
    pub tls_handshakes: AtomicU64,
    pub tls_handshake_failures: AtomicU64,
    // Request counters
    pub total_requests: AtomicU64,
    pub requests_1xx: AtomicU64,
    pub requests_2xx: AtomicU64,
    pub requests_3xx: AtomicU64,
    pub requests_4xx: AtomicU64,
    pub requests_5xx: AtomicU64,
    // Upstream
    pub upstream_connect_failures: AtomicU64,
    pub upstream_timeouts: AtomicU64,
    pub upstream_active: AtomicU64,
    pub upstream_retries: AtomicU64,
    pub upstream_retry_successes: AtomicU64,
    /// Requests that timed out waiting for a semaphore permit (queued acquire).
    pub upstream_queue_timeouts: AtomicU64,
    // Rate limiting
    pub rate_limited: AtomicU64,
    // WebSocket
    pub websocket_upgrades: AtomicU64,
    pub active_websockets: AtomicU64,
    // Bytes
    pub bytes_received: AtomicU64,
    pub bytes_sent: AtomicU64,
    // Latency histogram (Issue #11)
    pub latency: LatencyHistogram,
    // Splice zero-copy (Issue #016)
    pub splice_connections_active: AtomicU64,
    pub splice_bytes_total: AtomicU64,
    pub splice_fallbacks_total: AtomicU64,
    // Connection draining (Issue #018)
    pub drain_active: AtomicU64,
    pub drain_completed_total: AtomicU64,
    pub drain_forced_total: AtomicU64,
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(MetricsInner {
                start_time: Instant::now(),
                active_connections: AtomicU64::new(0),
                total_connections: AtomicU64::new(0),
                tls_handshakes: AtomicU64::new(0),
                tls_handshake_failures: AtomicU64::new(0),
                total_requests: AtomicU64::new(0),
                requests_1xx: AtomicU64::new(0),
                requests_2xx: AtomicU64::new(0),
                requests_3xx: AtomicU64::new(0),
                requests_4xx: AtomicU64::new(0),
                requests_5xx: AtomicU64::new(0),
                upstream_connect_failures: AtomicU64::new(0),
                upstream_timeouts: AtomicU64::new(0),
                upstream_active: AtomicU64::new(0),
                upstream_retries: AtomicU64::new(0),
                upstream_retry_successes: AtomicU64::new(0),
                upstream_queue_timeouts: AtomicU64::new(0),
                rate_limited: AtomicU64::new(0),
                websocket_upgrades: AtomicU64::new(0),
                active_websockets: AtomicU64::new(0),
                bytes_received: AtomicU64::new(0),
                bytes_sent: AtomicU64::new(0),
                latency: LatencyHistogram::new(),
                splice_connections_active: AtomicU64::new(0),
                splice_bytes_total: AtomicU64::new(0),
                splice_fallbacks_total: AtomicU64::new(0),
                drain_active: AtomicU64::new(0),
                drain_completed_total: AtomicU64::new(0),
                drain_forced_total: AtomicU64::new(0),
            }),
        }
    }

    /// Record a request latency observation (Issue #11).
    #[inline]
    pub fn record_latency(&self, duration: Duration) {
        self.inner.latency.observe(duration);
    }

    /// Export just the latency histogram in Prometheus text format.
    pub fn prometheus_export_histogram(&self) -> String {
        self.inner.latency.prometheus("q_flux_request_duration_seconds")
    }

    /// P50 latency in seconds.
    pub fn latency_p50(&self) -> f64 {
        self.inner.latency.percentile_seconds(0.5)
    }

    /// P95 latency in seconds.
    pub fn latency_p95(&self) -> f64 {
        self.inner.latency.percentile_seconds(0.95)
    }

    /// P99 latency in seconds.
    pub fn latency_p99(&self) -> f64 {
        self.inner.latency.percentile_seconds(0.99)
    }

    /// Export all metrics in Prometheus text format.
    #[allow(dead_code)] // Standalone export; admin.rs builds its own for richer output
    pub fn prometheus_export(&self) -> String {
        let s = self.snapshot();
        let mut out = String::with_capacity(2048);

        out.push_str("# HELP q_flux_connections_active Current active connections\n");
        out.push_str("# TYPE q_flux_connections_active gauge\n");
        out.push_str(&format!("q_flux_connections_active {}\n", s.active_connections));

        out.push_str("# HELP q_flux_connections_total Total connections\n");
        out.push_str("# TYPE q_flux_connections_total counter\n");
        out.push_str(&format!("q_flux_connections_total {}\n", s.total_connections));

        out.push_str("# HELP q_flux_requests_total Total requests by status\n");
        out.push_str("# TYPE q_flux_requests_total counter\n");
        out.push_str(&format!("q_flux_requests_total{{status=\"2xx\"}} {}\n", s.requests_2xx));
        out.push_str(&format!("q_flux_requests_total{{status=\"4xx\"}} {}\n", s.requests_4xx));
        out.push_str(&format!("q_flux_requests_total{{status=\"5xx\"}} {}\n", s.requests_5xx));

        out.push_str("# HELP q_flux_tls_handshakes_total TLS handshakes\n");
        out.push_str("# TYPE q_flux_tls_handshakes_total counter\n");
        out.push_str(&format!("q_flux_tls_handshakes_total{{result=\"ok\"}} {}\n", s.tls_handshakes));
        out.push_str(&format!("q_flux_tls_handshakes_total{{result=\"fail\"}} {}\n", s.tls_handshake_failures));

        out.push_str("# HELP q_flux_bytes_received_total Bytes received\n");
        out.push_str("# TYPE q_flux_bytes_received_total counter\n");
        out.push_str(&format!("q_flux_bytes_received_total {}\n", s.bytes_received));

        out.push_str("# HELP q_flux_bytes_sent_total Bytes sent\n");
        out.push_str("# TYPE q_flux_bytes_sent_total counter\n");
        out.push_str(&format!("q_flux_bytes_sent_total {}\n", s.bytes_sent));

        out.push_str("# HELP q_flux_rate_limited_total Rate-limited requests\n");
        out.push_str("# TYPE q_flux_rate_limited_total counter\n");
        out.push_str(&format!("q_flux_rate_limited_total {}\n", s.rate_limited));

        out.push_str("# HELP q_flux_upstream_active Active upstream connections\n");
        out.push_str("# TYPE q_flux_upstream_active gauge\n");
        out.push_str(&format!("q_flux_upstream_active {}\n", s.upstream_active));

        out.push_str("# HELP q_flux_uptime_seconds Uptime\n");
        out.push_str("# TYPE q_flux_uptime_seconds gauge\n");
        out.push_str(&format!("q_flux_uptime_seconds {}\n", s.uptime_secs));

        // Latency histogram
        out.push_str(&self.inner.latency.prometheus("q_flux_request_duration_seconds"));

        out
    }

    // Connection tracking
    pub fn conn_opened(&self) {
        self.inner.active_connections.fetch_add(1, Ordering::Relaxed);
        self.inner.total_connections.fetch_add(1, Ordering::Relaxed);
    }

    pub fn conn_closed(&self) {
        self.inner.active_connections.fetch_sub(1, Ordering::Relaxed);
    }

    pub fn tls_handshake_ok(&self) {
        self.inner.tls_handshakes.fetch_add(1, Ordering::Relaxed);
    }

    pub fn tls_handshake_fail(&self) {
        self.inner.tls_handshake_failures.fetch_add(1, Ordering::Relaxed);
    }

    // Request tracking
    pub fn request(&self) {
        self.inner.total_requests.fetch_add(1, Ordering::Relaxed);
    }

    pub fn response_status(&self, status: u16) {
        match status / 100 {
            1 => { self.inner.requests_1xx.fetch_add(1, Ordering::Relaxed); }
            2 => { self.inner.requests_2xx.fetch_add(1, Ordering::Relaxed); }
            3 => { self.inner.requests_3xx.fetch_add(1, Ordering::Relaxed); }
            4 => { self.inner.requests_4xx.fetch_add(1, Ordering::Relaxed); }
            5 => { self.inner.requests_5xx.fetch_add(1, Ordering::Relaxed); }
            _ => {}
        }
    }

    // Upstream tracking
    pub fn upstream_connect_fail(&self) {
        self.inner.upstream_connect_failures.fetch_add(1, Ordering::Relaxed);
    }

    pub fn upstream_timeout(&self) {
        self.inner.upstream_timeouts.fetch_add(1, Ordering::Relaxed);
    }

    pub fn upstream_acquired(&self) {
        self.inner.upstream_active.fetch_add(1, Ordering::Relaxed);
    }

    pub fn upstream_released(&self) {
        self.inner.upstream_active.fetch_sub(1, Ordering::Relaxed);
    }

    pub fn upstream_retry(&self) {
        self.inner.upstream_retries.fetch_add(1, Ordering::Relaxed);
    }

    pub fn upstream_retry_success(&self) {
        self.inner.upstream_retry_successes.fetch_add(1, Ordering::Relaxed);
    }

    pub fn upstream_queue_timeout(&self) {
        self.inner.upstream_queue_timeouts.fetch_add(1, Ordering::Relaxed);
    }

    // Rate limiting
    pub fn rate_limited(&self) {
        self.inner.rate_limited.fetch_add(1, Ordering::Relaxed);
    }

    // WebSocket
    pub fn ws_upgrade(&self) {
        self.inner.websocket_upgrades.fetch_add(1, Ordering::Relaxed);
        self.inner.active_websockets.fetch_add(1, Ordering::Relaxed);
    }

    pub fn ws_closed(&self) {
        self.inner.active_websockets.fetch_sub(1, Ordering::Relaxed);
    }

    // Bytes
    pub fn bytes_rx(&self, n: u64) {
        self.inner.bytes_received.fetch_add(n, Ordering::Relaxed);
    }

    pub fn bytes_tx(&self, n: u64) {
        self.inner.bytes_sent.fetch_add(n, Ordering::Relaxed);
    }

    // Splice zero-copy (Issue #016)
    // These are used by try_splice_bidirectional when plain TCP splice succeeds.
    // Currently TLS connections always fall back, so only splice_fallback() is called.
    #[allow(dead_code)]
    pub fn splice_opened(&self) {
        self.inner.splice_connections_active.fetch_add(1, Ordering::Relaxed);
    }

    #[allow(dead_code)]
    pub fn splice_closed(&self) {
        self.inner.splice_connections_active.fetch_sub(1, Ordering::Relaxed);
    }

    #[allow(dead_code)]
    pub fn splice_bytes(&self, n: u64) {
        self.inner.splice_bytes_total.fetch_add(n, Ordering::Relaxed);
    }

    pub fn splice_fallback(&self) {
        self.inner.splice_fallbacks_total.fetch_add(1, Ordering::Relaxed);
    }

    // Connection draining (Issue #018)
    pub fn drain_start(&self) {
        self.inner.drain_active.fetch_add(1, Ordering::Relaxed);
    }

    pub fn drain_completed(&self) {
        self.inner.drain_active.fetch_sub(1, Ordering::Relaxed);
        self.inner.drain_completed_total.fetch_add(1, Ordering::Relaxed);
    }

    pub fn drain_forced(&self) {
        self.inner.drain_active.fetch_sub(1, Ordering::Relaxed);
        self.inner.drain_forced_total.fetch_add(1, Ordering::Relaxed);
    }

    // Snapshot for logging/reporting
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            uptime_secs: self.inner.start_time.elapsed().as_secs(),
            active_connections: self.inner.active_connections.load(Ordering::Relaxed),
            total_connections: self.inner.total_connections.load(Ordering::Relaxed),
            tls_handshakes: self.inner.tls_handshakes.load(Ordering::Relaxed),
            tls_handshake_failures: self.inner.tls_handshake_failures.load(Ordering::Relaxed),
            total_requests: self.inner.total_requests.load(Ordering::Relaxed),
            requests_2xx: self.inner.requests_2xx.load(Ordering::Relaxed),
            requests_4xx: self.inner.requests_4xx.load(Ordering::Relaxed),
            requests_5xx: self.inner.requests_5xx.load(Ordering::Relaxed),
            upstream_connect_failures: self.inner.upstream_connect_failures.load(Ordering::Relaxed),
            upstream_timeouts: self.inner.upstream_timeouts.load(Ordering::Relaxed),
            upstream_active: self.inner.upstream_active.load(Ordering::Relaxed),
            upstream_retries: self.inner.upstream_retries.load(Ordering::Relaxed),
            upstream_retry_successes: self.inner.upstream_retry_successes.load(Ordering::Relaxed),
            upstream_queue_timeouts: self.inner.upstream_queue_timeouts.load(Ordering::Relaxed),
            rate_limited: self.inner.rate_limited.load(Ordering::Relaxed),
            websocket_upgrades: self.inner.websocket_upgrades.load(Ordering::Relaxed),
            active_websockets: self.inner.active_websockets.load(Ordering::Relaxed),
            bytes_received: self.inner.bytes_received.load(Ordering::Relaxed),
            bytes_sent: self.inner.bytes_sent.load(Ordering::Relaxed),
            splice_connections_active: self.inner.splice_connections_active.load(Ordering::Relaxed),
            splice_bytes_total: self.inner.splice_bytes_total.load(Ordering::Relaxed),
            splice_fallbacks_total: self.inner.splice_fallbacks_total.load(Ordering::Relaxed),
            drain_active: self.inner.drain_active.load(Ordering::Relaxed),
            drain_completed_total: self.inner.drain_completed_total.load(Ordering::Relaxed),
            drain_forced_total: self.inner.drain_forced_total.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    pub uptime_secs: u64,
    pub active_connections: u64,
    pub total_connections: u64,
    pub tls_handshakes: u64,
    pub tls_handshake_failures: u64,
    pub total_requests: u64,
    pub requests_2xx: u64,
    pub requests_4xx: u64,
    pub requests_5xx: u64,
    pub upstream_connect_failures: u64,
    pub upstream_timeouts: u64,
    pub upstream_active: u64,
    pub upstream_retries: u64,
    pub upstream_retry_successes: u64,
    pub upstream_queue_timeouts: u64,
    pub rate_limited: u64,
    pub websocket_upgrades: u64,
    pub active_websockets: u64,
    pub bytes_received: u64,
    pub bytes_sent: u64,
    // Splice zero-copy (Issue #016)
    pub splice_connections_active: u64,
    pub splice_bytes_total: u64,
    pub splice_fallbacks_total: u64,
    // Connection draining (Issue #018)
    pub drain_active: u64,
    pub drain_completed_total: u64,
    pub drain_forced_total: u64,
}

impl std::fmt::Display for MetricsSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "uptime={}s conns={}/{} tls_ok={} tls_fail={} reqs={} 2xx={} 4xx={} 5xx={} \
             upstream_fail={} upstream_timeout={} upstream_active={} upstream_queue_timeout={} retries={}/{} rate_limited={} \
             ws={}/{} rx={}B tx={}B",
            self.uptime_secs,
            self.active_connections,
            self.total_connections,
            self.tls_handshakes,
            self.tls_handshake_failures,
            self.total_requests,
            self.requests_2xx,
            self.requests_4xx,
            self.requests_5xx,
            self.upstream_connect_failures,
            self.upstream_timeouts,
            self.upstream_active,
            self.upstream_queue_timeouts,
            self.upstream_retries,
            self.upstream_retry_successes,
            self.rate_limited,
            self.active_websockets,
            self.websocket_upgrades,
            self.bytes_received,
            self.bytes_sent,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_bucket_basic() {
        let bucket = TokenBucket::new(100, 10); // 100/s, burst 10
        // Bucket starts full (10 tokens = 10000 milli-tokens)
        let now = 1_000_000; // 1 second in microseconds
        // Should be able to consume 10 tokens
        for _ in 0..10 {
            assert!(bucket.try_acquire(now));
        }
        // 11th should be rate-limited
        assert!(!bucket.try_acquire(now));
    }

    #[test]
    fn test_token_bucket_refill() {
        let bucket = TokenBucket::new(100, 10);
        let t0 = 1_000_000;
        // Consume all tokens
        for _ in 0..10 {
            assert!(bucket.try_acquire(t0));
        }
        assert!(!bucket.try_acquire(t0));
        // Advance 100ms — should refill 10 tokens (100/s * 0.1s = 10)
        let t1 = t0 + 100_000;
        for _ in 0..10 {
            assert!(bucket.try_acquire(t1));
        }
        assert!(!bucket.try_acquire(t1));
    }

    #[test]
    fn test_token_bucket_concurrent_safety() {
        use std::sync::Arc;
        use std::thread;
        let bucket = Arc::new(TokenBucket::new(10_000, 1000));
        let acquired = Arc::new(AtomicU64::new(0));

        let mut handles = vec![];
        for _ in 0..4 {
            let b = Arc::clone(&bucket);
            let a = Arc::clone(&acquired);
            handles.push(thread::spawn(move || {
                let now = 1_000_000u64;
                for _ in 0..500 {
                    if b.try_acquire(now) {
                        a.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        // Total acquired should equal burst capacity (1000)
        let total = acquired.load(Ordering::Relaxed);
        assert_eq!(total, 1000, "concurrent consumers should acquire exactly burst capacity, got {}", total);
    }

    #[test]
    fn test_rate_limiter_per_ip() {
        let limiter = RateLimiter::new(5, 5, 1_000_000);
        let ip: std::net::IpAddr = "1.2.3.4".parse().unwrap();
        // Should allow burst of 5
        for _ in 0..5 {
            assert!(limiter.check(ip));
        }
        // 6th should be rate-limited (within same millisecond, no refill)
        assert!(!limiter.check(ip));
        assert_eq!(limiter.active_ips(), 1);
    }

    #[test]
    fn test_latency_histogram_observe() {
        let hist = LatencyHistogram::new();
        hist.observe(std::time::Duration::from_millis(1));
        hist.observe(std::time::Duration::from_millis(50));
        hist.observe(std::time::Duration::from_millis(500));
        let prom = hist.prometheus("test_latency");
        assert!(prom.contains("test_latency_bucket"));
        assert!(prom.contains("test_latency_count 3"));
    }

    #[test]
    fn test_drain_metrics_start_and_completed() {
        let m = Metrics::new();
        assert_eq!(m.snapshot().drain_active, 0);
        assert_eq!(m.snapshot().drain_completed_total, 0);

        m.drain_start();
        m.drain_start();
        assert_eq!(m.snapshot().drain_active, 2);

        m.drain_completed();
        assert_eq!(m.snapshot().drain_active, 1);
        assert_eq!(m.snapshot().drain_completed_total, 1);

        m.drain_completed();
        assert_eq!(m.snapshot().drain_active, 0);
        assert_eq!(m.snapshot().drain_completed_total, 2);
    }

    #[test]
    fn test_drain_metrics_forced() {
        let m = Metrics::new();
        m.drain_start();
        m.drain_start();
        m.drain_start();

        m.drain_forced();
        assert_eq!(m.snapshot().drain_active, 2);
        assert_eq!(m.snapshot().drain_forced_total, 1);

        m.drain_completed();
        m.drain_forced();
        assert_eq!(m.snapshot().drain_active, 0);
        assert_eq!(m.snapshot().drain_completed_total, 1);
        assert_eq!(m.snapshot().drain_forced_total, 2);
    }

    #[test]
    fn test_latency_percentile() {
        let hist = LatencyHistogram::new();
        // Add 100 observations at 1ms
        for _ in 0..100 {
            hist.observe(Duration::from_millis(1));
        }
        let p50 = hist.percentile_seconds(0.5);
        assert!(p50 > 0.0 && p50 <= 0.001, "P50 should be ~1ms, got {}", p50);
        let p99 = hist.percentile_seconds(0.99);
        assert!(p99 > 0.0 && p99 <= 0.001, "P99 should be ~1ms, got {}", p99);
    }

    #[test]
    fn test_latency_percentile_empty() {
        let hist = LatencyHistogram::new();
        assert_eq!(hist.percentile_seconds(0.5), 0.0);
    }

    #[test]
    fn test_latency_percentile_spread() {
        let hist = LatencyHistogram::new();
        for _ in 0..50 { hist.observe(Duration::from_millis(1)); }
        for _ in 0..50 { hist.observe(Duration::from_millis(100)); }
        let p50 = hist.percentile_seconds(0.5);
        // P50 should be around 1ms (50th of 100 observations, first 50 are 1ms)
        assert!(p50 <= 0.005, "P50 should be <=5ms, got {}", p50);
        let p99 = hist.percentile_seconds(0.99);
        // P99 should be around 100ms
        assert!(p99 >= 0.01, "P99 should be >=10ms, got {}", p99);
    }
}
