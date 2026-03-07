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

use crate::config::UpstreamConfig;
use crate::metrics::Metrics;

/// Per-worker upstream connection pool.
/// Each worker gets its own pool to avoid cross-thread contention.
pub struct UpstreamPool {
    client: Client<HttpConnector, Full<Bytes>>,
    pub backends: Arc<Vec<String>>,
    response_timeout: Duration,
    metrics: Metrics,
    /// Round-robin index
    rr_index: AtomicUsize,
}

impl UpstreamPool {
    pub fn new(config: &UpstreamConfig, metrics: Metrics) -> Self {
        let mut connector = HttpConnector::new();
        connector.set_nodelay(true);
        connector.set_keepalive(Some(config.keepalive_timeout));
        connector.set_connect_timeout(Some(config.connect_timeout));
        connector.enforce_http(false);

        let client = Client::builder(TokioExecutor::new())
            .pool_idle_timeout(config.keepalive_timeout)
            .pool_max_idle_per_host(config.max_conns_per_worker)
            .build(connector);

        Self {
            client,
            backends: Arc::new(config.backends.clone()),
            response_timeout: config.response_timeout,
            metrics,
            rr_index: AtomicUsize::new(0),
        }
    }

    /// Pick the next backend (round-robin).
    fn next_backend(&self) -> &str {
        let idx = self.rr_index.fetch_add(1, Ordering::Relaxed);
        &self.backends[idx % self.backends.len()]
    }

    /// Get next backend address for direct TCP connections (e.g. WebSocket).
    pub fn next_backend_addr(&self) -> &str {
        self.next_backend()
    }

    /// Forward a request to the upstream and return the response.
    pub async fn forward(
        &self,
        mut req: hyper::Request<Full<Bytes>>,
    ) -> Result<hyper::Response<Incoming>> {
        let backend = self.next_backend();

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
            Ok(Ok(resp)) => Ok(resp),
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
