use anyhow::Result;
use bytes::Bytes;
use http_body_util::Full;
use hyper::body::Incoming;
use hyper_util::client::legacy::Client;
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::rt::TokioExecutor;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
use tokio::sync::Semaphore;
use tokio::time::timeout;
use tracing::warn;

use crate::config::UpstreamConfig;
use crate::health::HealthMap;
use crate::metrics::Metrics;

/// Default max concurrent upstream requests per worker (fallback).
const DEFAULT_MAX_UPSTREAM_INFLIGHT: usize = 64;

/// Default global max concurrent upstream requests across ALL workers.
/// With a single backend, 512 concurrent requests is plenty for high throughput
/// without overwhelming q-api-server. At 20ms avg response: 512/0.02 = 25,600 req/s.
const DEFAULT_MAX_UPSTREAM_GLOBAL: usize = 512;

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
pub struct UpstreamPool {
    client: Client<HttpConnector, Full<Bytes>>,
    pub backends: Arc<Vec<String>>,
    /// Super-cluster: remote peer backends for cross-node failover.
    cluster_peers: Arc<Vec<String>>,
    response_timeout: Duration,
    metrics: Metrics,
    health_map: HealthMap,
    /// Round-robin index for local backends
    rr_index: AtomicUsize,
    /// Round-robin index for cluster peers
    cluster_rr_index: AtomicUsize,
    /// Per-worker semaphore (fallback if no global semaphore provided).
    per_worker_semaphore: Arc<Semaphore>,
    /// Global semaphore shared across ALL workers — preferred over per-worker.
    /// This prevents the death spiral where 48 workers × N permits each
    /// overwhelm a single backend with too many concurrent connections.
    global_semaphore: Option<Arc<Semaphore>>,
    /// Max inflight limit (for error messages)
    max_inflight: usize,
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
                "Using GLOBAL upstream semaphore (shared across all workers)"
            );
        }

        Self {
            client,
            backends: Arc::new(config.backends.clone()),
            cluster_peers: Arc::new(cluster_peers),
            response_timeout: config.response_timeout,
            metrics,
            health_map,
            rr_index: AtomicUsize::new(0),
            cluster_rr_index: AtomicUsize::new(0),
            per_worker_semaphore: Arc::new(Semaphore::new(per_worker_max)),
            global_semaphore,
            max_inflight,
        }
    }

    /// Get the effective semaphore (global if available, else per-worker).
    #[inline]
    fn effective_semaphore(&self) -> &Arc<Semaphore> {
        self.global_semaphore.as_ref().unwrap_or(&self.per_worker_semaphore)
    }

    /// Pick the next healthy backend (round-robin, skipping unhealthy ones).
    ///
    /// Strategy (super-cluster aware):
    ///   1. Try local backends first (round-robin, skip unhealthy)
    ///   2. If ALL local backends are unhealthy AND cluster peers exist,
    ///      try cluster peers (round-robin, skip unhealthy)
    ///   3. If everything is unhealthy, fall back to first local backend
    ///      (degraded attempt is better than immediate 503)
    fn next_backend(&self) -> &str {
        let len = self.backends.len();
        let start = self.rr_index.fetch_add(1, Ordering::Relaxed);

        // First pass: look for a healthy LOCAL backend starting at the RR index
        for i in 0..len {
            let idx = (start + i) % len;
            let backend = &self.backends[idx];
            if let Some(entry) = self.health_map.get(backend.as_str()) {
                if entry.is_healthy {
                    return backend;
                }
            } else {
                // No health entry means we haven't checked yet -- assume healthy
                return backend;
            }
        }

        // All local backends unhealthy — try cluster peers if available
        if !self.cluster_peers.is_empty() {
            let clen = self.cluster_peers.len();
            let cstart = self.cluster_rr_index.fetch_add(1, Ordering::Relaxed);

            for i in 0..clen {
                let idx = (cstart + i) % clen;
                let peer = &self.cluster_peers[idx];
                if let Some(entry) = self.health_map.get(peer.as_str()) {
                    if entry.is_healthy {
                        warn!(
                            peer = peer.as_str(),
                            "Super-cluster failover: all local backends unhealthy, routing to cluster peer"
                        );
                        return peer;
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

        // Try local backends, skipping the excluded one
        for i in 0..len {
            let idx = (start + i) % len;
            let backend = &self.backends[idx];
            if backend == exclude {
                continue;
            }
            if let Some(entry) = self.health_map.get(backend.as_str()) {
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
        let _permit = match self.effective_semaphore().try_acquire() {
            Ok(permit) => permit,
            Err(_) => {
                self.metrics.upstream_connect_fail();
                return Err((anyhow::anyhow!("Upstream at capacity ({} in-flight)", self.max_inflight), String::new()));
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

        let result = timeout(self.response_timeout, self.client.request(req)).await;

        match result {
            Ok(Ok(resp)) => {
                // Inline health recovery: if a request to this backend succeeded,
                // mark it healthy immediately. This is critical for recovery when
                // the health checker is starved by memory pressure — actual traffic
                // proves the backend is alive.
                if let Some(mut entry) = self.health_map.get_mut(backend_addr.as_str()) {
                    if !entry.is_healthy {
                        entry.is_healthy = true;
                        entry.consecutive_failures = 0;
                        entry.unhealthy_since = None;
                        entry.last_success = Some(std::time::Instant::now());
                        tracing::info!(
                            backend = backend_addr.as_str(),
                            "Backend auto-recovered via successful request (inline health)"
                        );
                    }
                }
                Ok((resp, backend_addr))
            }
            Ok(Err(e)) => {
                self.metrics.upstream_connect_fail();
                Err((anyhow::anyhow!("Upstream error: {}", e), backend_addr))
            }
            Err(_) => {
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

        let _permit = match self.effective_semaphore().try_acquire() {
            Ok(permit) => permit,
            Err(_) => {
                self.metrics.upstream_connect_fail();
                return Some(Err(anyhow::anyhow!("Upstream at capacity (retry)")));
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
        let result = timeout(self.response_timeout, self.client.request(req)).await;

        match result {
            Ok(Ok(resp)) => {
                if let Some(mut entry) = self.health_map.get_mut(backend_addr.as_str()) {
                    if !entry.is_healthy {
                        entry.is_healthy = true;
                        entry.consecutive_failures = 0;
                        entry.unhealthy_since = None;
                        entry.last_success = Some(std::time::Instant::now());
                        tracing::info!(
                            backend = backend_addr.as_str(),
                            "Backend auto-recovered via retry request (inline health)"
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
}
