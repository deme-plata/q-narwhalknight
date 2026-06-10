use anyhow::Result;
use bytes::Bytes;
use http_body_util::Full;
use hyper::body::Incoming;
use hyper_util::client::legacy::Client;
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::rt::TokioExecutor;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;
use tokio::time::timeout;
use tracing::warn;

use crate::config::UpstreamConfig;
use crate::health::HealthMap;
use crate::metrics::Metrics;
use dashmap::DashMap;

/// Default max concurrent upstream requests per worker (fallback).
const DEFAULT_MAX_UPSTREAM_INFLIGHT: usize = 64;

/// Default global max concurrent upstream requests across ALL workers.
/// 2048 permits at 50ms avg response = 40K req/s throughput — 4× headroom for Epsilon.
const DEFAULT_MAX_UPSTREAM_GLOBAL: usize = 2048;

/// RAII guard for upstream_active metric tracking.
/// Ensures the counter is always decremented, even on task cancellation.
/// Without this, cancelled tokio tasks leak the counter indefinitely.
struct UpstreamActiveGuard<'a> {
    metrics: &'a Metrics,
}

impl<'a> UpstreamActiveGuard<'a> {
    #[inline]
    fn new(metrics: &'a Metrics) -> Self {
        metrics.upstream_acquired();
        Self { metrics }
    }
}

impl<'a> Drop for UpstreamActiveGuard<'a> {
    fn drop(&mut self) {
        self.metrics.upstream_released();
    }
}

/// Per-worker upstream connection pool.
/// Each worker gets its own hyper Client for connection pooling,
/// but shares a GLOBAL semaphore to precisely cap total backend load.
///
/// Super-cluster mode: when `cluster_peers` is non-empty, the pool tries local
/// backends first. If ALL local backends are unhealthy, it fails over to cluster
/// peers (remote q-api-server instances on other servers). Local always wins.
/// Adaptive concurrency controller using AIMD (Additive Increase, Multiplicative Decrease).
///
/// Tracks EWMA response latency per backend and adjusts the effective permit limit:
/// - Latency > 3× baseline → reduce effective permits by 25% (multiplicative decrease)
/// - Latency < 1.5× baseline → increase effective permits by 10% (additive increase)
/// - Floor: never below total_permits / 8
///
/// The actual semaphore size stays constant — adaptive concurrency works by holding
/// "phantom permits" that artificially reduce availability.
pub struct AdaptiveConcurrency {
    /// EWMA of response latency in microseconds (alpha = 0.1).
    ewma_latency_us: AtomicU64,
    /// Baseline latency captured from the first 100 responses (in microseconds).
    baseline_latency_us: AtomicU64,
    /// Number of responses seen (saturates at u64::MAX, used for baseline warm-up).
    response_count: AtomicU64,
    /// Sum of latencies during warm-up (first 100 responses), in microseconds.
    warmup_sum_us: AtomicU64,
    /// Current effective limit (may be lower than semaphore capacity).
    effective_limit: AtomicUsize,
    /// Maximum (configured) limit — the semaphore's actual capacity.
    max_limit: usize,
    /// Minimum limit floor (max_limit / 8).
    min_limit: usize,
}

impl AdaptiveConcurrency {
    pub fn new(max_limit: usize) -> Self {
        Self {
            ewma_latency_us: AtomicU64::new(0),
            baseline_latency_us: AtomicU64::new(0),
            response_count: AtomicU64::new(0),
            warmup_sum_us: AtomicU64::new(0),
            effective_limit: AtomicUsize::new(max_limit),
            max_limit,
            min_limit: (max_limit / 8).max(64),
        }
    }

    /// Record a response latency observation. Updates EWMA and baseline.
    #[inline]
    pub fn record_latency(&self, duration: Duration) {
        let us = duration.as_micros() as u64;
        let count = self.response_count.fetch_add(1, Ordering::Relaxed);

        // Warm-up: accumulate for baseline (first 100 responses)
        if count < 100 {
            self.warmup_sum_us.fetch_add(us, Ordering::Relaxed);
            if count == 99 {
                // 100th response — set baseline
                let sum = self.warmup_sum_us.load(Ordering::Relaxed);
                let baseline = sum / 100;
                self.baseline_latency_us.store(baseline, Ordering::Relaxed);
                self.ewma_latency_us.store(baseline, Ordering::Relaxed);
            }
            return;
        }

        // EWMA update: new = 0.1 * sample + 0.9 * old
        // Using integer math: new = (sample + 9 * old) / 10
        loop {
            let old = self.ewma_latency_us.load(Ordering::Relaxed);
            let new_val = (us + 9 * old) / 10;
            if self.ewma_latency_us.compare_exchange_weak(
                old, new_val, Ordering::Relaxed, Ordering::Relaxed
            ).is_ok() {
                break;
            }
        }
    }

    /// Periodic adjustment (call every ~1s). Returns the new effective limit.
    pub fn adjust(&self) -> usize {
        let baseline = self.baseline_latency_us.load(Ordering::Relaxed);
        if baseline == 0 {
            // Not enough data yet — keep max
            return self.max_limit;
        }

        let current_latency = self.ewma_latency_us.load(Ordering::Relaxed);
        let current_limit = self.effective_limit.load(Ordering::Relaxed);

        let new_limit = if current_latency > baseline * 3 {
            // Multiplicative decrease: reduce by 25%
            let reduced = current_limit * 3 / 4;
            reduced.max(self.min_limit)
        } else if current_latency < baseline * 3 / 2 {
            // Additive increase: +10%
            let increased = current_limit + (self.max_limit / 10).max(1);
            increased.min(self.max_limit)
        } else {
            // In the neutral zone — no change
            current_limit
        };

        if new_limit != current_limit {
            self.effective_limit.store(new_limit, Ordering::Relaxed);
            tracing::debug!(
                baseline_ms = baseline / 1000,
                current_ms = current_latency / 1000,
                old_limit = current_limit,
                new_limit = new_limit,
                "Adaptive concurrency adjusted",
            );
        }

        new_limit
    }

    /// Get the current effective limit.
    pub fn effective_limit(&self) -> usize {
        self.effective_limit.load(Ordering::Relaxed)
    }
}

/// Per-backend request counters for Prometheus metrics (Issue #026).
pub struct BackendCounters {
    pub total_requests: AtomicU64,
    pub failed_requests: AtomicU64,
}

impl BackendCounters {
    pub fn new() -> Self {
        Self {
            total_requests: AtomicU64::new(0),
            failed_requests: AtomicU64::new(0),
        }
    }
}

pub struct UpstreamPool {
    client: Client<HttpConnector, Full<Bytes>>,
    pub backends: Arc<Vec<String>>,
    /// Super-cluster: remote peer backends for cross-node failover.
    cluster_peers: Arc<Vec<String>>,
    response_timeout: Duration,
    /// How long to wait for a semaphore permit before returning 502.
    /// Duration::ZERO = old instant-reject behavior (try_acquire).
    acquire_timeout: Duration,
    metrics: Metrics,
    health_map: HealthMap,
    /// Round-robin index for local backends
    rr_index: AtomicUsize,
    /// Round-robin index for cluster peers
    cluster_rr_index: AtomicUsize,
    /// Per-worker semaphore (fallback if no global semaphore provided).
    per_worker_semaphore: Arc<Semaphore>,
    /// Global semaphore shared across ALL workers — preferred over per-worker.
    global_semaphore: Option<Arc<Semaphore>>,
    /// Max inflight limit (for error messages)
    max_inflight: usize,
    /// Adaptive concurrency controller (AIMD).
    pub adaptive: Arc<AdaptiveConcurrency>,
    /// Per-backend request counters (Issue #026).
    pub backend_counters: Arc<dashmap::DashMap<String, Arc<BackendCounters>>>,
}

impl UpstreamPool {
    #[allow(dead_code)]
    pub fn new(config: &UpstreamConfig, metrics: Metrics, health_map: HealthMap) -> Self {
        Self::new_with_cluster(config, metrics, health_map, vec![])
    }

    pub fn new_with_cluster(
        config: &UpstreamConfig,
        metrics: Metrics,
        health_map: HealthMap,
        cluster_peers: Vec<String>,
    ) -> Self {
        Self::new_full(config, metrics, health_map, cluster_peers, None)
    }

    pub fn new_full(
        config: &UpstreamConfig,
        metrics: Metrics,
        health_map: HealthMap,
        cluster_peers: Vec<String>,
        global_semaphore: Option<Arc<Semaphore>>,
    ) -> Self {
        let backend_counters: Arc<DashMap<String, Arc<BackendCounters>>> = Arc::new(DashMap::new());
        Self::new_with_counters(config, metrics, health_map, cluster_peers, global_semaphore, backend_counters)
    }

    pub fn new_with_counters(
        config: &UpstreamConfig,
        metrics: Metrics,
        health_map: HealthMap,
        cluster_peers: Vec<String>,
        global_semaphore: Option<Arc<Semaphore>>,
        backend_counters: Arc<DashMap<String, Arc<BackendCounters>>>,
    ) -> Self {
        let mut connector = HttpConnector::new();
        connector.set_nodelay(true);
        connector.set_keepalive(Some(config.keepalive_timeout));
        connector.set_connect_timeout(Some(config.connect_timeout));
        connector.enforce_http(false);

        // Pool is per-worker. With 128 idle conns × 48 workers = 6144 upstream conns max.
        // This is the key to high throughput: reuse TCP connections to upstream instead of
        // creating a new one per request (which was the bug that killed performance).
        let client = Client::builder(TokioExecutor::new())
            .pool_idle_timeout(config.keepalive_timeout)
            .pool_max_idle_per_host(config.max_conns_per_worker)
            .retry_canceled_requests(true)
            .set_host(true)
            .build(connector);

        if !cluster_peers.is_empty() {
            tracing::info!(
                local_backends = config.backends.len(),
                cluster_peers = cluster_peers.len(),
                "Super-cluster enabled: local-first, {} remote peer(s) as failover",
                cluster_peers.len(),
            );
        }

        let per_worker_max = if config.max_inflight_per_worker > 0 {
            config.max_inflight_per_worker
        } else {
            DEFAULT_MAX_UPSTREAM_INFLIGHT
        };

        // Effective limit for error messages: global if set, else per-worker
        let max_inflight = if let Some(ref sem) = global_semaphore {
            sem.available_permits()
        } else {
            per_worker_max
        };

        if global_semaphore.is_some() {
            tracing::info!(
                global_limit = max_inflight,
                per_worker_fallback = per_worker_max,
                acquire_timeout_ms = config.acquire_timeout.as_millis() as u64,
                "Using GLOBAL upstream semaphore (shared across all workers)"
            );
        }

        let adaptive = Arc::new(AdaptiveConcurrency::new(max_inflight));

        // Pre-populate per-backend counters (Issue #026) if not already present
        for b in &config.backends {
            backend_counters.entry(b.clone()).or_insert_with(|| Arc::new(BackendCounters::new()));
        }
        for p in &cluster_peers {
            backend_counters.entry(p.clone()).or_insert_with(|| Arc::new(BackendCounters::new()));
        }

        Self {
            client,
            backends: Arc::new(config.backends.clone()),
            cluster_peers: Arc::new(cluster_peers),
            response_timeout: config.response_timeout,
            acquire_timeout: config.acquire_timeout,
            metrics,
            health_map,
            rr_index: AtomicUsize::new(0),
            cluster_rr_index: AtomicUsize::new(0),
            per_worker_semaphore: Arc::new(Semaphore::new(per_worker_max)),
            global_semaphore,
            max_inflight,
            adaptive,
            backend_counters,
        }
    }

    /// Get the effective semaphore (global if available, else per-worker).
    #[inline]
    fn effective_semaphore(&self) -> &Arc<Semaphore> {
        self.global_semaphore.as_ref().unwrap_or(&self.per_worker_semaphore)
    }

    /// Pick the next healthy backend (round-robin, skipping unhealthy ones).
    ///
    /// Strategy (super-cluster aware, half-open aware):
    ///   1. Try local backends: prefer fully healthy, then half-open
    ///   2. If no healthy/half-open local backends AND cluster peers exist,
    ///      try cluster peers (same preference order)
    ///   3. If everything is unhealthy, fall back to first local backend
    ///      (degraded attempt is better than immediate 503)
    fn next_backend(&self) -> &str {
        let len = self.backends.len();
        let start = self.rr_index.fetch_add(1, Ordering::Relaxed);

        // First pass: look for a FULLY HEALTHY local backend (not half-open, not drained)
        let mut first_half_open: Option<&str> = None;
        for i in 0..len {
            let idx = (start + i) % len;
            let backend = &self.backends[idx];
            if let Some(entry) = self.health_map.get(backend.as_str()) {
                // v1.0.5: Skip admin-drained backends (deploy in progress)
                if entry.is_admin_drained {
                    continue;
                }
                if entry.is_healthy && !entry.half_open {
                    return backend;
                }
                if entry.is_healthy && entry.half_open && first_half_open.is_none() {
                    first_half_open = Some(backend);
                }
            } else {
                // No health entry means we haven't checked yet -- assume healthy
                return backend;
            }
        }

        // No fully healthy local backends — use half-open if available
        if let Some(ho_backend) = first_half_open {
            return ho_backend;
        }

        // All local backends unhealthy — try cluster peers if available
        if !self.cluster_peers.is_empty() {
            let clen = self.cluster_peers.len();
            let cstart = self.cluster_rr_index.fetch_add(1, Ordering::Relaxed);

            let mut cluster_half_open: Option<&str> = None;
            for i in 0..clen {
                let idx = (cstart + i) % clen;
                let peer = &self.cluster_peers[idx];
                if let Some(entry) = self.health_map.get(peer.as_str()) {
                    if entry.is_healthy && !entry.half_open {
                        warn!(
                            peer = peer.as_str(),
                            "Super-cluster failover: all local backends unhealthy, routing to cluster peer"
                        );
                        return peer;
                    }
                    if entry.is_healthy && entry.half_open && cluster_half_open.is_none() {
                        cluster_half_open = Some(peer);
                    }
                } else {
                    // No health entry — assume healthy (optimistic)
                    warn!(
                        peer = peer.as_str(),
                        "Super-cluster failover: routing to cluster peer (not yet health-checked)"
                    );
                    return peer;
                }
            }

            if let Some(ho_peer) = cluster_half_open {
                warn!(
                    peer = ho_peer,
                    "Super-cluster failover: routing to half-open cluster peer"
                );
                return ho_peer;
            }
        }

        // Everything unhealthy: fall through to the original local RR pick
        let fallback = &self.backends[start % len];
        warn!(
            backend = fallback.as_str(),
            "All backends unhealthy (local + cluster), attempting degraded fallback"
        );
        fallback
    }

    /// Pick the next healthy backend, skipping a specific address.
    /// Used for retry logic: after a failure on backend X, try a different one.
    /// Returns None if no alternative backend is available.
    fn next_backend_excluding(&self, exclude: &str) -> Option<&str> {
        let len = self.backends.len();
        let start = self.rr_index.fetch_add(1, Ordering::Relaxed);

        // Try local backends, skipping the excluded one and admin-drained ones
        for i in 0..len {
            let idx = (start + i) % len;
            let backend = &self.backends[idx];
            if backend == exclude {
                continue;
            }
            if let Some(entry) = self.health_map.get(backend.as_str()) {
                if entry.is_admin_drained {
                    continue; // v1.0.5: skip drained backends
                }
                if entry.is_healthy {
                    return Some(backend);
                }
            } else {
                return Some(backend);
            }
        }

        // Try cluster peers, skipping the excluded one
        if !self.cluster_peers.is_empty() {
            let clen = self.cluster_peers.len();
            let cstart = self.cluster_rr_index.fetch_add(1, Ordering::Relaxed);
            for i in 0..clen {
                let idx = (cstart + i) % clen;
                let peer = &self.cluster_peers[idx];
                if peer == exclude {
                    continue;
                }
                if let Some(entry) = self.health_map.get(peer.as_str()) {
                    if entry.is_healthy {
                        return Some(peer);
                    }
                } else {
                    return Some(peer);
                }
            }
        }

        None
    }

    /// Get next backend address for direct TCP connections (e.g. WebSocket, SSE).
    pub fn next_backend_addr(&self) -> &str {
        self.next_backend()
    }

    /// Pick a healthy backend address (for direct TCP streaming).
    /// Returns None if no backends are healthy.
    pub fn pick_backend(&self) -> Option<String> {
        // Round-robin through healthy backends (skipping admin-drained)
        let backends = &self.backends;
        let len = backends.len();
        if len == 0 { return None; }
        for _ in 0..len {
            let idx = self.rr_index.fetch_add(1, Ordering::Relaxed) % len;
            let addr = &backends[idx];
            if let Some(entry) = self.health_map.get(addr.as_str()) {
                if entry.is_admin_drained {
                    continue; // v1.0.5: skip drained backends
                }
                if entry.is_healthy {
                    return Some(addr.clone());
                }
            } else {
                // No health entry — assume healthy (optimistic)
                return Some(addr.clone());
            }
        }
        // No healthy local backends — try cluster failover
        for peer in self.cluster_peers.iter() {
            if let Some(entry) = self.health_map.get(peer.as_str()) {
                if entry.is_healthy {
                    return Some(peer.clone());
                }
            } else {
                return Some(peer.clone());
            }
        }
        None
    }

    /// Forward a request to the upstream and return the response along with
    /// the backend address that served it (for access logging).
    ///
    /// On failure, returns `(Err, tried_backend_addr)` so callers can retry
    /// on a different backend via `forward_excluding()`.
    pub async fn forward(
        &self,
        mut req: hyper::Request<Full<Bytes>>,
    ) -> std::result::Result<(hyper::Response<Incoming>, String), (anyhow::Error, String)> {
        // Backpressure: cap concurrent upstream requests.
        // Uses global semaphore (shared across all workers) when configured,
        // preventing the death spiral of 48 workers × N permits overwhelming
        // a single backend.
        //
        // Queued acquire with timeout: instead of instantly rejecting when all
        // permits are held (which caused thousands of 502s/sec during brief stalls),
        // we wait up to acquire_timeout for a permit to be freed by a completing
        // response. This dramatically reduces spurious 502s.
        let sem = self.effective_semaphore();
        let _permit = if self.acquire_timeout.is_zero() {
            // Legacy instant-reject mode
            match sem.try_acquire() {
                Ok(permit) => permit,
                Err(_) => {
                    self.metrics.upstream_queue_timeout();
                    return Err((anyhow::anyhow!("Upstream at capacity ({} in-flight)", self.max_inflight), String::new()));
                }
            }
        } else {
            match timeout(self.acquire_timeout, sem.acquire()).await {
                Ok(Ok(permit)) => permit,
                Ok(Err(_)) => {
                    // Semaphore closed — shouldn't happen in normal operation
                    self.metrics.upstream_queue_timeout();
                    return Err((anyhow::anyhow!("Upstream semaphore closed"), String::new()));
                }
                Err(_) => {
                    // Timed out waiting for a permit
                    self.metrics.upstream_queue_timeout();
                    return Err((anyhow::anyhow!(
                        "Upstream queue timeout ({:?}, {} in-flight)",
                        self.acquire_timeout, self.max_inflight
                    ), String::new()));
                }
            }
        };

        let backend = self.next_backend();
        let backend_addr = backend.to_string();

        // Rewrite the URI to point at the backend
        let path_and_query = req.uri().path_and_query()
            .map(|pq| pq.as_str())
            .unwrap_or("/");

        let uri = format!("http://{}{}", backend, path_and_query);
        *req.uri_mut() = match uri.parse() {
            Ok(u) => u,
            Err(e) => return Err((anyhow::anyhow!("Bad URI: {}", e), backend_addr)),
        };

        // Remove hop-by-hop headers that shouldn't be forwarded
        let headers = req.headers_mut();
        headers.remove(hyper::header::CONNECTION);
        headers.remove(hyper::header::TRANSFER_ENCODING);
        headers.remove("keep-alive");
        headers.remove("proxy-connection");

        // RAII guard: upstream_active metric is decremented even on task cancellation.
        // Without this, cancelled tasks leak the counter → inflated upstream_active.
        let _active_guard = UpstreamActiveGuard::new(&self.metrics);

        let req_start = Instant::now();
        let result = timeout(self.response_timeout, self.client.request(req)).await;

        // Feed response latency into adaptive concurrency controller
        let elapsed = req_start.elapsed();
        self.adaptive.record_latency(elapsed);

        // Issue #026: increment per-backend counters
        let counters = self.backend_counters
            .entry(backend_addr.clone())
            .or_insert_with(|| Arc::new(BackendCounters::new()))
            .clone();
        counters.total_requests.fetch_add(1, Ordering::Relaxed);

        match result {
            Ok(Ok(resp)) => {
                // Inline health recovery: if a request to this backend succeeded,
                // mark it healthy immediately. This is critical for recovery when
                // the health checker is starved by memory pressure — actual traffic
                // proves the backend is alive.
                if let Some(mut entry) = self.health_map.get_mut(backend_addr.as_str()) {
                    if !entry.is_healthy {
                        entry.is_healthy = true;
                        entry.half_open = true; // enter half-open, let health checker promote
                        entry.consecutive_failures = 0;
                        entry.consecutive_successes = 1;
                        entry.unhealthy_since = None;
                        entry.last_success = Some(std::time::Instant::now());
                        tracing::info!(
                            backend = backend_addr.as_str(),
                            "Backend auto-recovered via successful request (inline → half-open)"
                        );
                    }
                }
                Ok((resp, backend_addr))
            }
            Ok(Err(e)) => {
                counters.failed_requests.fetch_add(1, Ordering::Relaxed);
                self.metrics.upstream_connect_fail();
                Err((anyhow::anyhow!("Upstream error: {}", e), backend_addr))
            }
            Err(_) => {
                counters.failed_requests.fetch_add(1, Ordering::Relaxed);
                self.metrics.upstream_timeout();
                Err((anyhow::anyhow!("Upstream timeout after {:?}", self.response_timeout), backend_addr))
            }
        }
    }

    /// Expose health map for testing.
    #[cfg(test)]
    pub fn health_map(&self) -> &HealthMap {
        &self.health_map
    }

    /// Forward a request to an upstream backend, skipping the specified backend.
    /// Used for retry after a failed first attempt on an idempotent request.
    /// Returns None if no alternative backend is available.
    pub async fn forward_excluding(
        &self,
        mut req: hyper::Request<Full<Bytes>>,
        exclude_backend: &str,
    ) -> Option<Result<(hyper::Response<Incoming>, String)>> {
        let backend = self.next_backend_excluding(exclude_backend)?;
        let backend_addr = backend.to_string();

        let sem = self.effective_semaphore();
        let _permit = if self.acquire_timeout.is_zero() {
            match sem.try_acquire() {
                Ok(permit) => permit,
                Err(_) => {
                    self.metrics.upstream_queue_timeout();
                    return Some(Err(anyhow::anyhow!("Upstream at capacity (retry)")));
                }
            }
        } else {
            match timeout(self.acquire_timeout, sem.acquire()).await {
                Ok(Ok(permit)) => permit,
                Ok(Err(_)) => {
                    self.metrics.upstream_queue_timeout();
                    return Some(Err(anyhow::anyhow!("Upstream semaphore closed (retry)")));
                }
                Err(_) => {
                    self.metrics.upstream_queue_timeout();
                    return Some(Err(anyhow::anyhow!("Upstream queue timeout (retry)")));
                }
            }
        };

        let path_and_query = req.uri().path_and_query()
            .map(|pq| pq.as_str())
            .unwrap_or("/");

        let uri = format!("http://{}{}", backend_addr, path_and_query);
        match uri.parse() {
            Ok(parsed) => *req.uri_mut() = parsed,
            Err(e) => return Some(Err(anyhow::anyhow!("Bad URI on retry: {}", e))),
        }

        let headers = req.headers_mut();
        headers.remove(hyper::header::CONNECTION);
        headers.remove(hyper::header::TRANSFER_ENCODING);
        headers.remove("keep-alive");
        headers.remove("proxy-connection");

        // RAII guard for metrics (cancellation-safe)
        let _active_guard = UpstreamActiveGuard::new(&self.metrics);
        let req_start = Instant::now();
        let result = timeout(self.response_timeout, self.client.request(req)).await;
        self.adaptive.record_latency(req_start.elapsed());

        match result {
            Ok(Ok(resp)) => {
                if let Some(mut entry) = self.health_map.get_mut(backend_addr.as_str()) {
                    if !entry.is_healthy {
                        entry.is_healthy = true;
                        entry.half_open = true;
                        entry.consecutive_failures = 0;
                        entry.consecutive_successes = 1;
                        entry.unhealthy_since = None;
                        entry.last_success = Some(std::time::Instant::now());
                        tracing::info!(
                            backend = backend_addr.as_str(),
                            "Backend auto-recovered via retry request (inline → half-open)"
                        );
                    }
                }
                Some(Ok((resp, backend_addr)))
            }
            Ok(Err(e)) => {
                self.metrics.upstream_connect_fail();
                Some(Err(anyhow::anyhow!("Upstream retry error: {}", e)))
            }
            Err(_) => {
                self.metrics.upstream_timeout();
                Some(Err(anyhow::anyhow!("Upstream retry timeout after {:?}", self.response_timeout)))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::UpstreamConfig;
    use crate::health;
    use dashmap::DashMap;
    use std::collections::HashMap;
    use std::time::Duration;

    /// Build a minimal UpstreamConfig for testing.
    fn test_config(backends: Vec<&str>) -> UpstreamConfig {
        UpstreamConfig {
            backends: backends.into_iter().map(|s| s.to_string()).collect(),
            max_conns_per_worker: 32,
            keepalive_timeout: Duration::from_secs(30),
            connect_timeout: Duration::from_secs(5),
            response_timeout: Duration::from_secs(30),
            health_check_interval: Duration::from_secs(5),
            health_check_path: "/health".to_string(),
            health_check_timeout: Duration::from_secs(3),
            max_inflight_per_worker: 64,
            max_upstream_global: 0,
            acquire_timeout: Duration::from_millis(500),
            failure_threshold: 3,
            healthy_threshold: 2,
        }
    }

    /// Build an UpstreamPool from a list of backend addresses and optional cluster peers.
    fn make_pool(backends: Vec<&str>, cluster: Vec<&str>) -> UpstreamPool {
        let mut all: Vec<String> = backends.iter().map(|s| s.to_string()).collect();
        all.extend(cluster.iter().map(|s| s.to_string()));
        let health_map = health::new_health_map(&all);
        let config = test_config(backends);
        let metrics = Metrics::new();
        UpstreamPool::new_with_cluster(
            &config,
            metrics,
            health_map,
            cluster.into_iter().map(|s| s.to_string()).collect(),
        )
    }

    #[test]
    fn test_round_robin_distributes_evenly() {
        let pool = make_pool(vec!["A:80", "B:80", "C:80"], vec![]);
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for _ in 0..300 {
            let b = pool.next_backend();
            *counts.entry(b).or_insert(0) += 1;
        }
        // Each backend should get exactly 100 picks (300 / 3)
        assert_eq!(counts.get("A:80"), Some(&100));
        assert_eq!(counts.get("B:80"), Some(&100));
        assert_eq!(counts.get("C:80"), Some(&100));
    }

    #[test]
    fn test_skips_unhealthy_backend() {
        let pool = make_pool(vec!["A:80", "B:80", "C:80"], vec![]);
        // Mark B:80 as unhealthy
        if let Some(mut entry) = pool.health_map().get_mut("B:80") {
            entry.is_healthy = false;
        }
        for _ in 0..100 {
            let b = pool.next_backend();
            assert_ne!(b, "B:80", "Unhealthy backend B:80 should be skipped");
        }
    }

    #[test]
    fn test_prefers_healthy_over_half_open() {
        let pool = make_pool(vec!["A:80", "B:80", "C:80"], vec![]);
        // Mark B:80 as half-open (recovering)
        if let Some(mut entry) = pool.health_map().get_mut("B:80") {
            entry.half_open = true;
        }
        // A and C are fully healthy — B should be skipped
        for _ in 0..100 {
            let b = pool.next_backend();
            assert_ne!(b, "B:80", "Half-open B:80 should be skipped when healthy backends exist");
        }
    }

    #[test]
    fn test_half_open_used_when_no_fully_healthy() {
        let pool = make_pool(vec!["A:80", "B:80"], vec![]);
        // Mark A unhealthy, B half-open
        if let Some(mut entry) = pool.health_map().get_mut("A:80") {
            entry.is_healthy = false;
        }
        if let Some(mut entry) = pool.health_map().get_mut("B:80") {
            entry.half_open = true;
        }
        for _ in 0..50 {
            let b = pool.next_backend();
            assert_eq!(b, "B:80", "Half-open B:80 should be used when no fully healthy");
        }
    }

    #[test]
    fn test_cluster_failover_when_all_local_unhealthy() {
        let pool = make_pool(vec!["A:80", "B:80"], vec!["C:80", "D:80"]);
        // Mark all local backends unhealthy
        for backend in ["A:80", "B:80"] {
            if let Some(mut entry) = pool.health_map().get_mut(backend) {
                entry.is_healthy = false;
            }
        }
        for _ in 0..100 {
            let b = pool.next_backend();
            assert!(
                b == "C:80" || b == "D:80",
                "Should failover to cluster peer, got: {}",
                b,
            );
        }
    }

    #[test]
    fn test_local_preferred_over_cluster() {
        let pool = make_pool(vec!["A:80"], vec!["C:80"]);
        // Both healthy — should always pick local
        for _ in 0..100 {
            assert_eq!(pool.next_backend(), "A:80");
        }
    }

    #[test]
    fn test_fallback_when_all_unhealthy() {
        let pool = make_pool(vec!["A:80", "B:80"], vec!["C:80"]);
        // Mark everything unhealthy
        for backend in ["A:80", "B:80", "C:80"] {
            if let Some(mut entry) = pool.health_map().get_mut(backend) {
                entry.is_healthy = false;
            }
        }
        // Should fall back to a local backend (degraded)
        let b = pool.next_backend();
        assert!(b == "A:80" || b == "B:80", "Fallback should be a local backend, got: {}", b);
    }

    #[test]
    fn test_excluding_skips_specified_backend() {
        let pool = make_pool(vec!["A:80", "B:80", "C:80"], vec![]);
        for _ in 0..100 {
            let b = pool.next_backend_excluding("B:80");
            assert!(b.is_some());
            assert_ne!(b.unwrap(), "B:80");
        }
    }

    #[test]
    fn test_excluding_returns_none_when_no_alternative() {
        let pool = make_pool(vec!["A:80"], vec![]);
        // Only one backend — excluding it leaves nothing
        let b = pool.next_backend_excluding("A:80");
        assert!(b.is_none(), "Should return None when only backend is excluded");
    }

    #[test]
    fn test_excluding_tries_cluster_peers() {
        let pool = make_pool(vec!["A:80"], vec!["C:80"]);
        // Exclude the only local backend — should fall to cluster peer
        let b = pool.next_backend_excluding("A:80");
        assert_eq!(b, Some("C:80"));
    }

    #[test]
    fn test_no_health_entry_assumes_healthy() {
        // Create pool with backends NOT pre-registered in health map
        let config = test_config(vec!["X:80", "Y:80"]);
        let health_map = Arc::new(DashMap::new()); // empty — no entries
        let metrics = Metrics::new();
        let pool = UpstreamPool::new_with_cluster(&config, metrics, health_map, vec![]);
        // Should still pick backends (optimistic — no entry = healthy)
        let b = pool.next_backend();
        assert!(b == "X:80" || b == "Y:80");
    }

    #[test]
    fn test_single_backend_always_returned() {
        let pool = make_pool(vec!["A:80"], vec![]);
        for _ in 0..50 {
            assert_eq!(pool.next_backend(), "A:80");
        }
    }

    // ── AdaptiveConcurrency tests ─────────────────────────────────

    #[test]
    fn test_adaptive_concurrency_starts_at_max() {
        let ac = AdaptiveConcurrency::new(2048);
        assert_eq!(ac.effective_limit(), 2048);
        // Before baseline is set, adjust should return max
        assert_eq!(ac.adjust(), 2048);
    }

    #[test]
    fn test_adaptive_concurrency_baseline_warmup() {
        let ac = AdaptiveConcurrency::new(1024);
        // Feed 100 responses at ~10ms each
        for _ in 0..100 {
            ac.record_latency(Duration::from_millis(10));
        }
        let baseline = ac.baseline_latency_us.load(Ordering::Relaxed);
        // Should be ~10,000 us (10ms)
        assert!(baseline >= 9_000 && baseline <= 11_000, "baseline={}", baseline);
    }

    #[test]
    fn test_adaptive_concurrency_decrease_on_high_latency() {
        let ac = AdaptiveConcurrency::new(1024);
        // Warm up baseline at 10ms
        for _ in 0..100 {
            ac.record_latency(Duration::from_millis(10));
        }
        // Simulate high latency: feed 50ms responses to push EWMA above 3× baseline (30ms)
        for _ in 0..50 {
            ac.record_latency(Duration::from_millis(50));
        }
        let new_limit = ac.adjust();
        assert!(new_limit < 1024, "should decrease, got {}", new_limit);
        assert!(new_limit >= 1024 / 8, "should not go below floor, got {}", new_limit);
    }

    #[test]
    fn test_adaptive_concurrency_increase_on_low_latency() {
        let ac = AdaptiveConcurrency::new(1024);
        // Warm up baseline at 10ms
        for _ in 0..100 {
            ac.record_latency(Duration::from_millis(10));
        }
        // Manually reduce effective limit
        ac.effective_limit.store(512, Ordering::Relaxed);
        // Feed low-latency responses to keep EWMA below 1.5× baseline
        for _ in 0..20 {
            ac.record_latency(Duration::from_millis(10));
        }
        let new_limit = ac.adjust();
        assert!(new_limit > 512, "should increase from 512, got {}", new_limit);
        assert!(new_limit <= 1024, "should not exceed max, got {}", new_limit);
    }

    #[test]
    fn test_adaptive_concurrency_floor() {
        let ac = AdaptiveConcurrency::new(1024);
        // Floor = max(1024/8, 64) = 128
        assert_eq!(ac.min_limit, 128);
        // Force limit to floor
        ac.effective_limit.store(128, Ordering::Relaxed);
        // Warm up and push high latency
        for _ in 0..100 {
            ac.record_latency(Duration::from_millis(10));
        }
        for _ in 0..50 {
            ac.record_latency(Duration::from_millis(100));
        }
        let new_limit = ac.adjust();
        assert!(new_limit >= 128, "should not go below floor, got {}", new_limit);
    }
}
