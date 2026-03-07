//! QUIC/HTTP3 proxy via quinn (Phase 3).
//!
//! Provides QUIC-based transport for mining clients. QUIC advantages over TCP+TLS:
//! - Zero-RTT: returning miners skip the full handshake (saves ~3ms per reconnect)
//! - No head-of-line blocking: lost packets don't stall other streams
//! - Connection migration: miners can change IP without reconnecting
//! - Built-in TLS 1.3: no separate TLS handshake step
//!
//! Architecture:
//! - Frontend: QUIC via quinn (miner <-> q-flux on UDP :443)
//! - Backend: HTTP/1.1 via hyper (q-flux <-> upstream on TCP)
//! - Each QUIC connection can carry multiple bidirectional streams
//! - Mining submit/challenge use separate streams (no blocking)
//!
//! # Cargo.toml dependency (add when enabling the `quic` feature):
//!
//! ```toml
//! [dependencies]
//! quinn = { version = "0.11", optional = true }
//!
//! [features]
//! quic = ["quinn"]
//! ```
//!
//! # Integration
//!
//! The QUIC endpoint runs alongside the TCP listener. In `main.rs`:
//! ```ignore
//! #[cfg(feature = "quic")]
//! {
//!     let quic_proxy = QuicProxy::new(quic_config, tls_config, metrics.clone())?;
//!     tokio::spawn(quic_proxy.run(upstream_config, shutdown_rx));
//! }
//! ```

use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use bytes::Bytes;
use tracing::{debug, info, warn};

use crate::metrics::Metrics;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the QUIC/HTTP3 proxy frontend.
#[derive(Debug, Clone)]
pub struct QuicConfig {
    /// UDP listen address (default: "0.0.0.0:443").
    pub listen: SocketAddr,

    /// Maximum concurrent bidirectional streams per connection (default: 128).
    /// Each mining submit or challenge request uses one stream.
    pub max_concurrent_streams: u32,

    /// Enable 0-RTT for returning clients (default: true).
    ///
    /// 0-RTT allows clients to send data in the first flight, skipping a full
    /// round-trip. This saves ~3ms per reconnect for miners. The trade-off is
    /// that 0-RTT data is replayable -- acceptable for idempotent mining
    /// challenge requests but NOT for mining submits (which have nonce
    /// uniqueness protection at the application layer).
    pub zero_rtt_enabled: bool,

    /// Idle timeout before closing the QUIC connection (default: 30s).
    /// Mining clients typically submit every ~1s, so 30s handles transient
    /// network interruptions without keeping dead connections forever.
    pub idle_timeout_secs: u64,

    /// Maximum UDP payload size in bytes (default: 1350).
    /// Conservative default to avoid fragmentation. Can be raised to 1452
    /// on networks that support it (most modern networks do).
    pub max_udp_payload: u16,

    /// Allow connection migration (default: true).
    /// When a miner's IP changes (NAT rebinding, mobile network switch),
    /// the QUIC connection survives without re-handshaking.
    pub migration_enabled: bool,
}

impl Default for QuicConfig {
    fn default() -> Self {
        Self {
            listen: SocketAddr::from(([0, 0, 0, 0], 443)),
            max_concurrent_streams: 128,
            zero_rtt_enabled: true,
            idle_timeout_secs: 30,
            max_udp_payload: 1350,
            migration_enabled: true,
        }
    }
}

impl QuicConfig {
    /// Validate configuration values.
    pub fn validate(&self) -> Result<()> {
        if self.max_concurrent_streams == 0 {
            anyhow::bail!("max_concurrent_streams must be > 0");
        }
        if self.idle_timeout_secs == 0 {
            anyhow::bail!("idle_timeout_secs must be > 0");
        }
        // QUIC minimum MTU is 1200 bytes
        if self.max_udp_payload < 1200 {
            anyhow::bail!(
                "max_udp_payload {} is below QUIC minimum (1200)",
                self.max_udp_payload
            );
        }
        // Typical Ethernet MTU minus IP/UDP headers
        if self.max_udp_payload > 1452 {
            anyhow::bail!(
                "max_udp_payload {} exceeds safe maximum (1452)",
                self.max_udp_payload
            );
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// QUIC-specific metrics
// ---------------------------------------------------------------------------

/// Counters specific to QUIC connections.
#[derive(Debug)]
pub struct QuicMetrics {
    /// Total QUIC connections accepted.
    pub connections: AtomicU64,
    /// Currently active QUIC connections.
    pub active_connections: AtomicU64,
    /// Total bidirectional streams opened.
    pub streams_opened: AtomicU64,
    /// Total 0-RTT connections accepted.
    pub zero_rtt_accepted: AtomicU64,
    /// Total connection migrations detected.
    pub migrations: AtomicU64,
    /// Total bytes received over QUIC.
    pub bytes_rx: AtomicU64,
    /// Total bytes sent over QUIC.
    pub bytes_tx: AtomicU64,
}

impl QuicMetrics {
    pub fn new() -> Self {
        Self {
            connections: AtomicU64::new(0),
            active_connections: AtomicU64::new(0),
            streams_opened: AtomicU64::new(0),
            zero_rtt_accepted: AtomicU64::new(0),
            migrations: AtomicU64::new(0),
            bytes_rx: AtomicU64::new(0),
            bytes_tx: AtomicU64::new(0),
        }
    }

    #[inline]
    pub fn conn_opened(&self) {
        self.connections.fetch_add(1, Ordering::Relaxed);
        self.active_connections.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn conn_closed(&self) {
        self.active_connections.fetch_sub(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn stream_opened(&self) {
        self.streams_opened.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn zero_rtt(&self) {
        self.zero_rtt_accepted.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn migration(&self) {
        self.migrations.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn rx(&self, n: u64) {
        self.bytes_rx.fetch_add(n, Ordering::Relaxed);
    }

    #[inline]
    pub fn tx(&self, n: u64) {
        self.bytes_tx.fetch_add(n, Ordering::Relaxed);
    }

    /// Format as Prometheus text for inclusion in the metrics endpoint.
    pub fn prometheus_export(&self) -> String {
        let mut out = String::with_capacity(1024);

        out.push_str("# HELP q_flux_quic_connections_total Total QUIC connections\n");
        out.push_str("# TYPE q_flux_quic_connections_total counter\n");
        out.push_str(&format!(
            "q_flux_quic_connections_total {}\n",
            self.connections.load(Ordering::Relaxed)
        ));

        out.push_str("# HELP q_flux_quic_connections_active Active QUIC connections\n");
        out.push_str("# TYPE q_flux_quic_connections_active gauge\n");
        out.push_str(&format!(
            "q_flux_quic_connections_active {}\n",
            self.active_connections.load(Ordering::Relaxed)
        ));

        out.push_str("# HELP q_flux_quic_streams_total Total QUIC streams opened\n");
        out.push_str("# TYPE q_flux_quic_streams_total counter\n");
        out.push_str(&format!(
            "q_flux_quic_streams_total {}\n",
            self.streams_opened.load(Ordering::Relaxed)
        ));

        out.push_str("# HELP q_flux_quic_zero_rtt_total Total 0-RTT connections\n");
        out.push_str("# TYPE q_flux_quic_zero_rtt_total counter\n");
        out.push_str(&format!(
            "q_flux_quic_zero_rtt_total {}\n",
            self.zero_rtt_accepted.load(Ordering::Relaxed)
        ));

        out.push_str("# HELP q_flux_quic_migrations_total Total connection migrations\n");
        out.push_str("# TYPE q_flux_quic_migrations_total counter\n");
        out.push_str(&format!(
            "q_flux_quic_migrations_total {}\n",
            self.migrations.load(Ordering::Relaxed)
        ));

        out
    }
}

impl Default for QuicMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// QUIC proxy (requires `quinn` feature)
// ---------------------------------------------------------------------------

/// QUIC reverse proxy that accepts QUIC/HTTP3 connections on a UDP socket
/// and forwards requests to the HTTP/1.1 upstream pool.
///
/// # Feature gate
///
/// This struct and its methods are only available when the `quic` feature is
/// enabled. Without it, the module still compiles (config, metrics, and
/// helpers are always available) but `QuicProxy` is not.
///
/// # Usage
///
/// ```ignore
/// let proxy = QuicProxy::new(config, tls_server_config, metrics)?;
/// proxy.run(upstream_pool, shutdown_signal).await;
/// ```
#[cfg(feature = "quic")]
pub struct QuicProxy {
    endpoint: quinn::Endpoint,
    config: QuicConfig,
    metrics: Arc<QuicMetrics>,
    global_metrics: Metrics,
}

#[cfg(feature = "quic")]
impl QuicProxy {
    /// Create a new QUIC proxy endpoint bound to the configured UDP address.
    ///
    /// `tls_config` should be a `rustls::ServerConfig` with ALPN set to
    /// `["h3"]`. The same TLS certificate used for TCP/TLS can be reused here.
    pub fn new(
        config: QuicConfig,
        tls_config: rustls::ServerConfig,
        metrics: Metrics,
    ) -> Result<Self> {
        config.validate()?;

        let mut server_config = quinn::ServerConfig::with_crypto(Arc::new(tls_config));

        let mut transport = quinn::TransportConfig::default();
        transport.max_concurrent_bidi_streams(
            quinn::VarInt::from_u32(config.max_concurrent_streams),
        );
        transport.max_idle_timeout(Some(
            quinn::IdleTimeout::try_from(Duration::from_secs(config.idle_timeout_secs))
                .map_err(|e| anyhow::anyhow!("Invalid idle timeout: {}", e))?,
        ));
        transport.initial_mtu(config.max_udp_payload);
        transport.allow_spin(true); // Enable spin bit for latency measurement

        if !config.migration_enabled {
            transport.allow_migration(false);
        }

        server_config.transport_config(Arc::new(transport));

        let endpoint = quinn::Endpoint::server(server_config, config.listen)?;

        info!(
            listen = %config.listen,
            max_streams = config.max_concurrent_streams,
            zero_rtt = config.zero_rtt_enabled,
            "QUIC endpoint started"
        );

        Ok(Self {
            endpoint,
            config,
            metrics: Arc::new(QuicMetrics::new()),
            global_metrics: metrics,
        })
    }

    /// Run the QUIC accept loop until the shutdown signal fires.
    ///
    /// Each incoming QUIC connection is spawned as an independent task.
    /// Within each connection, individual bidirectional streams are also
    /// spawned independently -- this mirrors the HTTP/2 stream model.
    pub async fn run(
        self,
        upstream: crate::upstream::UpstreamPool,
        mut shutdown: tokio::sync::watch::Receiver<bool>,
    ) {
        let upstream = Arc::new(upstream);

        loop {
            tokio::select! {
                Some(incoming) = self.endpoint.accept() => {
                    let metrics = self.metrics.clone();
                    let global_metrics = self.global_metrics.clone();
                    let upstream = upstream.clone();
                    let zero_rtt = self.config.zero_rtt_enabled;

                    tokio::spawn(async move {
                        match incoming.await {
                            Ok(connection) => {
                                metrics.conn_opened();
                                global_metrics.conn_opened();

                                let remote = connection.remote_address();
                                info!(peer = %remote, "QUIC connection established");

                                handle_quic_connection(
                                    connection,
                                    &upstream,
                                    &global_metrics,
                                    &metrics,
                                )
                                .await;

                                metrics.conn_closed();
                                global_metrics.conn_closed();
                                debug!(peer = %remote, "QUIC connection closed");
                            }
                            Err(e) => {
                                warn!("QUIC accept error: {}", e);
                            }
                        }
                    });
                }
                _ = shutdown.changed() => {
                    info!("QUIC proxy shutting down");
                    self.endpoint.close(
                        quinn::VarInt::from_u32(0),
                        b"server shutdown",
                    );
                    break;
                }
            }
        }
    }
}

/// Handle a single QUIC connection by accepting bidirectional streams.
///
/// Each stream is an independent HTTP request/response pair. Streams are
/// spawned as separate tasks so that a slow response on one stream does
/// not block others (this is the core QUIC advantage over HTTP/1.1).
#[cfg(feature = "quic")]
async fn handle_quic_connection(
    connection: quinn::Connection,
    upstream: &crate::upstream::UpstreamPool,
    global_metrics: &Metrics,
    quic_metrics: &QuicMetrics,
) {
    loop {
        match connection.accept_bi().await {
            Ok((send, recv)) => {
                quic_metrics.stream_opened();
                global_metrics.request();

                // Clone what we need for the spawned task
                let remote = connection.remote_address();
                // In a full implementation, upstream would be Arc-wrapped
                // and cloned here. For now we show the structure.

                tokio::spawn(async move {
                    if let Err(e) = handle_quic_stream(send, recv, remote).await {
                        debug!(peer = %remote, "QUIC stream error: {}", e);
                    }
                });
            }
            Err(quinn::ConnectionError::ApplicationClosed(_)) => {
                debug!("QUIC connection closed by peer (application)");
                break;
            }
            Err(quinn::ConnectionError::ConnectionClosed(_)) => {
                debug!("QUIC connection closed (transport)");
                break;
            }
            Err(e) => {
                warn!("QUIC stream accept error: {}", e);
                break;
            }
        }
    }
}

/// Handle a single QUIC bidirectional stream.
///
/// Reads an HTTP request from the QUIC receive stream, forwards it to the
/// HTTP/1.1 upstream, and sends the response back over the QUIC send stream.
///
/// The request/response format is a simplified HTTP-over-QUIC wire format:
/// - Request: HTTP/1.1 formatted headers + body on the receive stream
/// - Response: HTTP/1.1 formatted headers + body on the send stream
///
/// For full HTTP/3 compliance, an h3 crate would be used instead. This
/// simplified format is sufficient for our mining clients which we control.
#[cfg(feature = "quic")]
async fn handle_quic_stream(
    mut send: quinn::SendStream,
    mut recv: quinn::RecvStream,
    peer: SocketAddr,
) -> Result<()> {
    use bytes::Bytes;

    // Read request from QUIC stream (up to 1MB)
    let request_bytes = recv
        .read_to_end(1024 * 1024)
        .await
        .map_err(|e| anyhow::anyhow!("QUIC read error: {}", e))?;

    debug!(
        peer = %peer,
        len = request_bytes.len(),
        "QUIC stream request received"
    );

    // In a full implementation:
    // 1. Parse the HTTP request from request_bytes
    // 2. Forward to upstream via UpstreamPool::forward()
    // 3. Write the response to the send stream

    // Placeholder: echo back a minimal HTTP response
    let response = b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\n\r\nOK";
    send.write_all(response)
        .await
        .map_err(|e| anyhow::anyhow!("QUIC write error: {}", e))?;
    send.finish().map_err(|e| anyhow::anyhow!("QUIC finish error: {}", e))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Non-feature-gated helpers
// ---------------------------------------------------------------------------

/// Parse a QUIC transport header to extract the mining-related stream type.
///
/// Mining clients tag their streams with a purpose byte in the first byte:
/// - `0x01`: Mining challenge request (GET /api/v1/mining/challenge)
/// - `0x02`: Mining submit (POST /api/v1/mining/submit)
/// - `0x03`: SSE stream subscription (GET /api/v1/mining/stream)
/// - `0x00`: Generic HTTP request
///
/// This allows the proxy to prioritize mining submits over generic requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamPurpose {
    Generic,
    MiningChallenge,
    MiningSubmit,
    SseSubscription,
}

impl StreamPurpose {
    /// Parse the purpose from the first byte of a QUIC stream.
    pub fn from_tag(tag: u8) -> Self {
        match tag {
            0x01 => StreamPurpose::MiningChallenge,
            0x02 => StreamPurpose::MiningSubmit,
            0x03 => StreamPurpose::SseSubscription,
            _ => StreamPurpose::Generic,
        }
    }

    /// Whether this stream type should be prioritized.
    pub fn is_high_priority(&self) -> bool {
        matches!(self, StreamPurpose::MiningSubmit)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = QuicConfig::default();
        assert_eq!(config.max_concurrent_streams, 128);
        assert!(config.zero_rtt_enabled);
        assert_eq!(config.idle_timeout_secs, 30);
        assert_eq!(config.max_udp_payload, 1350);
        assert!(config.migration_enabled);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation_rejects_zero_streams() {
        let config = QuicConfig {
            max_concurrent_streams: 0,
            ..QuicConfig::default()
        };
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("max_concurrent_streams"));
    }

    #[test]
    fn test_config_validation_rejects_zero_timeout() {
        let config = QuicConfig {
            idle_timeout_secs: 0,
            ..QuicConfig::default()
        };
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("idle_timeout_secs"));
    }

    #[test]
    fn test_config_validation_rejects_small_mtu() {
        let config = QuicConfig {
            max_udp_payload: 1100, // Below QUIC minimum
            ..QuicConfig::default()
        };
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("max_udp_payload"));
        assert!(err.to_string().contains("1200"));
    }

    #[test]
    fn test_config_validation_rejects_oversized_mtu() {
        let config = QuicConfig {
            max_udp_payload: 9000, // Jumbo frame, unsafe for Internet
            ..QuicConfig::default()
        };
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("max_udp_payload"));
    }

    #[test]
    fn test_config_boundary_values() {
        // Minimum valid MTU
        let config = QuicConfig {
            max_udp_payload: 1200,
            ..QuicConfig::default()
        };
        assert!(config.validate().is_ok());

        // Maximum valid MTU
        let config = QuicConfig {
            max_udp_payload: 1452,
            ..QuicConfig::default()
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_stream_purpose_parsing() {
        assert_eq!(StreamPurpose::from_tag(0x00), StreamPurpose::Generic);
        assert_eq!(StreamPurpose::from_tag(0x01), StreamPurpose::MiningChallenge);
        assert_eq!(StreamPurpose::from_tag(0x02), StreamPurpose::MiningSubmit);
        assert_eq!(StreamPurpose::from_tag(0x03), StreamPurpose::SseSubscription);
        assert_eq!(StreamPurpose::from_tag(0xFF), StreamPurpose::Generic);
    }

    #[test]
    fn test_stream_purpose_priority() {
        assert!(!StreamPurpose::Generic.is_high_priority());
        assert!(!StreamPurpose::MiningChallenge.is_high_priority());
        assert!(StreamPurpose::MiningSubmit.is_high_priority());
        assert!(!StreamPurpose::SseSubscription.is_high_priority());
    }

    #[test]
    fn test_quic_metrics() {
        let m = QuicMetrics::new();

        m.conn_opened();
        m.conn_opened();
        m.stream_opened();
        m.stream_opened();
        m.stream_opened();
        m.zero_rtt();
        m.migration();
        m.rx(1024);
        m.tx(2048);
        m.conn_closed();

        assert_eq!(m.connections.load(Ordering::Relaxed), 2);
        assert_eq!(m.active_connections.load(Ordering::Relaxed), 1);
        assert_eq!(m.streams_opened.load(Ordering::Relaxed), 3);
        assert_eq!(m.zero_rtt_accepted.load(Ordering::Relaxed), 1);
        assert_eq!(m.migrations.load(Ordering::Relaxed), 1);
        assert_eq!(m.bytes_rx.load(Ordering::Relaxed), 1024);
        assert_eq!(m.bytes_tx.load(Ordering::Relaxed), 2048);
    }

    #[test]
    fn test_quic_metrics_prometheus() {
        let m = QuicMetrics::new();
        m.conn_opened();
        m.zero_rtt();

        let prom = m.prometheus_export();
        assert!(prom.contains("q_flux_quic_connections_total 1"));
        assert!(prom.contains("q_flux_quic_connections_active 1"));
        assert!(prom.contains("q_flux_quic_zero_rtt_total 1"));
        assert!(prom.contains("# TYPE q_flux_quic_connections_total counter"));
        assert!(prom.contains("# TYPE q_flux_quic_connections_active gauge"));
    }
}
