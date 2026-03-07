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
use tokio::time::timeout;
use tracing::warn;

use crate::config::UpstreamConfig;
use crate::health::HealthMap;
use crate::metrics::Metrics;

/// Per-worker upstream connection pool.
/// Each worker gets its own pool to avoid cross-thread contention.
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
}

impl UpstreamPool {
    pub fn new(config: &UpstreamConfig, metrics: Metrics, health_map: HealthMap) -> Self {
        Self::new_with_cluster(config, metrics, health_map, vec![])
    }

    pub fn new_with_cluster(
        config: &UpstreamConfig,
        metrics: Metrics,
        health_map: HealthMap,
        cluster_peers: Vec<String>,
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

        Self {
            client,
            backends: Arc::new(config.backends.clone()),
            cluster_peers: Arc::new(cluster_peers),
            response_timeout: config.response_timeout,
            metrics,
            health_map,
            rr_index: AtomicUsize::new(0),
            cluster_rr_index: AtomicUsize::new(0),
        }
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

    /// Get next backend address for direct TCP connections (e.g. WebSocket).
    pub fn next_backend_addr(&self) -> &str {
        self.next_backend()
    }

    /// Forward a request to the upstream and return the response along with
    /// the backend address that served it (for access logging).
    pub async fn forward(
        &self,
        mut req: hyper::Request<Full<Bytes>>,
    ) -> Result<(hyper::Response<Incoming>, String)> {
        let backend = self.next_backend();
        let backend_addr = backend.to_string();

        // Rewrite the URI to point at the backend
        let path_and_query = req.uri().path_and_query()
            .map(|pq| pq.as_str())
            .unwrap_or("/");

        let uri = format!("http://{}{}", backend, path_and_query);
        *req.uri_mut() = uri.parse().map_err(|e| anyhow::anyhow!("Bad URI: {}", e))?;

        // Remove hop-by-hop headers that shouldn't be forwarded
        let headers = req.headers_mut();
        headers.remove(hyper::header::CONNECTION);
        headers.remove(hyper::header::TRANSFER_ENCODING);
        headers.remove("keep-alive");
        headers.remove("proxy-connection");

        self.metrics.upstream_acquired();

        let result = timeout(self.response_timeout, self.client.request(req)).await;

        self.metrics.upstream_released();

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
                Err(anyhow::anyhow!("Upstream error: {}", e))
            }
            Err(_) => {
                self.metrics.upstream_timeout();
                Err(anyhow::anyhow!("Upstream timeout after {:?}", self.response_timeout))
            }
        }
    }
}
