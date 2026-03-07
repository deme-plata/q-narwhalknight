use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

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
    // Rate limiting
    pub rate_limited: AtomicU64,
    // WebSocket
    pub websocket_upgrades: AtomicU64,
    pub active_websockets: AtomicU64,
    // Bytes
    pub bytes_received: AtomicU64,
    pub bytes_sent: AtomicU64,
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
                rate_limited: AtomicU64::new(0),
                websocket_upgrades: AtomicU64::new(0),
                active_websockets: AtomicU64::new(0),
                bytes_received: AtomicU64::new(0),
                bytes_sent: AtomicU64::new(0),
            }),
        }
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
            rate_limited: self.inner.rate_limited.load(Ordering::Relaxed),
            active_websockets: self.inner.active_websockets.load(Ordering::Relaxed),
            bytes_received: self.inner.bytes_received.load(Ordering::Relaxed),
            bytes_sent: self.inner.bytes_sent.load(Ordering::Relaxed),
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
    pub rate_limited: u64,
    pub active_websockets: u64,
    pub bytes_received: u64,
    pub bytes_sent: u64,
}

impl std::fmt::Display for MetricsSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "uptime={}s conns={}/{} tls_ok={} tls_fail={} reqs={} 2xx={} 4xx={} 5xx={} \
             upstream_fail={} upstream_timeout={} upstream_active={} rate_limited={} \
             ws={} rx={}B tx={}B",
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
            self.rate_limited,
            self.active_websockets,
            self.bytes_received,
            self.bytes_sent,
        )
    }
}
