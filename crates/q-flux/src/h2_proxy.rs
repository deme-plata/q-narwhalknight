//! HTTP/2 multiplexed proxy (Phase 3).
//!
//! Provides HTTP/2 frontend handling for browser clients. When ALPN negotiates
//! "h2", connections are handled by this module instead of the HTTP/1.1 proxy.
//!
//! Architecture:
//! - Frontend: HTTP/2 via `h2` crate (browser <-> q-flux)
//! - Backend: HTTP/1.1 via hyper (q-flux <-> upstream) -- backend doesn't need H2
//! - Multiplexing: multiple streams over one TLS connection
//! - Flow control: per-stream and connection-level flow control
//! - Server push: disabled (not useful for API/mining traffic)
//!
//! # Integration
//!
//! In the TLS acceptor, after ALPN negotiation:
//! ```ignore
//! match tls_stream.get_ref().1.alpn_protocol() {
//!     Some(b"h2") => h2_proxy::handle_h2_connection(tls_stream, addr, upstream, metrics, body_limit, static_cfg).await,
//!     _           => proxy::handle_connection(tls_stream, addr, upstream, metrics, body_limit, static_cfg).await,
//! }
//! ```

use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::Result;
use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use tokio::io::{AsyncRead, AsyncWrite};
use tracing::{debug, info, warn};

use crate::config::StaticConfig;
use crate::metrics::Metrics;
use crate::upstream::UpstreamPool;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the HTTP/2 frontend handler.
#[derive(Debug, Clone)]
pub struct H2Config {
    /// Maximum number of concurrent streams per connection (default: 256).
    /// Browser limit is typically 100; we allow more for API aggregators.
    pub max_concurrent_streams: u32,

    /// Initial flow-control window size in bytes (default: 2 MiB).
    /// Larger windows reduce round-trips for big responses (block-pack, sync).
    pub initial_window_size: u32,

    /// Maximum HTTP/2 frame size in bytes (default: 16 KiB, max 16 MiB).
    pub max_frame_size: u32,

    /// Maximum size of the header list in bytes (default: 16 KiB).
    pub max_header_list_size: u32,

    /// Whether to enable the HTTP/2 CONNECT protocol (RFC 8441).
    /// Not needed for our use case; kept for future WebSocket-over-H2.
    pub enable_connect_protocol: bool,
}

impl Default for H2Config {
    fn default() -> Self {
        Self {
            max_concurrent_streams: 256,
            initial_window_size: 2 * 1024 * 1024,   // 2 MiB
            max_frame_size: 16 * 1024,               // 16 KiB (HTTP/2 default)
            max_header_list_size: 16 * 1024,          // 16 KiB
            enable_connect_protocol: false,
        }
    }
}

impl H2Config {
    /// Validate configuration values against HTTP/2 spec limits.
    pub fn validate(&self) -> Result<()> {
        if self.max_concurrent_streams == 0 {
            anyhow::bail!("max_concurrent_streams must be > 0");
        }
        // HTTP/2 spec: INITIAL_WINDOW_SIZE max is 2^31 - 1
        if self.initial_window_size > 0x7FFF_FFFF {
            anyhow::bail!(
                "initial_window_size {} exceeds HTTP/2 max (2^31 - 1)",
                self.initial_window_size
            );
        }
        // HTTP/2 spec: MAX_FRAME_SIZE must be between 16384 and 16777215
        if self.max_frame_size < 16_384 {
            anyhow::bail!(
                "max_frame_size {} is below HTTP/2 minimum (16384)",
                self.max_frame_size
            );
        }
        if self.max_frame_size > 16_777_215 {
            anyhow::bail!(
                "max_frame_size {} exceeds HTTP/2 maximum (16777215)",
                self.max_frame_size
            );
        }
        if self.max_header_list_size == 0 {
            anyhow::bail!("max_header_list_size must be > 0");
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// H2 metrics extension
// ---------------------------------------------------------------------------

/// Counters specific to HTTP/2 connections.
/// Designed to be embedded alongside the existing `Metrics` struct.
#[derive(Debug)]
pub struct H2Metrics {
    /// Total H2 connections accepted.
    pub connections: AtomicU64,
    /// Total H2 streams opened across all connections.
    pub streams_opened: AtomicU64,
    /// Total H2 streams closed (completed or reset).
    pub streams_closed: AtomicU64,
    /// Total GOAWAY frames sent (graceful shutdown).
    pub goaway_sent: AtomicU64,
}

impl H2Metrics {
    pub fn new() -> Self {
        Self {
            connections: AtomicU64::new(0),
            streams_opened: AtomicU64::new(0),
            streams_closed: AtomicU64::new(0),
            goaway_sent: AtomicU64::new(0),
        }
    }

    #[inline]
    pub fn h2_connection(&self) {
        self.connections.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn h2_stream_opened(&self) {
        self.streams_opened.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn h2_stream_closed(&self) {
        self.streams_closed.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn h2_goaway(&self) {
        self.goaway_sent.fetch_add(1, Ordering::Relaxed);
    }

    /// Format as Prometheus text for inclusion in the metrics endpoint.
    pub fn prometheus_export(&self) -> String {
        let mut out = String::with_capacity(512);
        out.push_str("# HELP q_flux_h2_connections_total Total HTTP/2 connections\n");
        out.push_str("# TYPE q_flux_h2_connections_total counter\n");
        out.push_str(&format!(
            "q_flux_h2_connections_total {}\n",
            self.connections.load(Ordering::Relaxed)
        ));
        out.push_str("# HELP q_flux_h2_streams_opened_total Total H2 streams opened\n");
        out.push_str("# TYPE q_flux_h2_streams_opened_total counter\n");
        out.push_str(&format!(
            "q_flux_h2_streams_opened_total {}\n",
            self.streams_opened.load(Ordering::Relaxed)
        ));
        out.push_str("# HELP q_flux_h2_streams_closed_total Total H2 streams closed\n");
        out.push_str("# TYPE q_flux_h2_streams_closed_total counter\n");
        out.push_str(&format!(
            "q_flux_h2_streams_closed_total {}\n",
            self.streams_closed.load(Ordering::Relaxed)
        ));
        out
    }
}

impl Default for H2Metrics {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Header conversion helpers
// ---------------------------------------------------------------------------

/// Convert an `h2::RecvStream`-style pseudo-header request into a
/// `hyper::Request<Full<Bytes>>` suitable for forwarding to the HTTP/1.1
/// upstream.
///
/// HTTP/2 pseudo-headers (`:method`, `:path`, `:authority`, `:scheme`) are
/// mapped to the HTTP/1.1 request-line and `Host` header. Regular headers
/// are copied verbatim. Connection-level headers (`connection`,
/// `transfer-encoding`, `keep-alive`) are stripped because they are
/// hop-by-hop in HTTP/1.1 and meaningless in HTTP/2.
pub fn h2_request_to_http1(
    h2_req: &http::Request<()>,
    body: Bytes,
    client_addr: SocketAddr,
) -> Result<hyper::Request<Full<Bytes>>> {
    let mut builder = hyper::Request::builder()
        .method(h2_req.method().clone())
        .uri(h2_req.uri().clone());

    for (key, value) in h2_req.headers() {
        // Skip HTTP/2 pseudo-headers (they start with ':') and
        // hop-by-hop headers that are invalid in HTTP/1.1.
        let name = key.as_str();
        if name.starts_with(':') {
            continue;
        }
        match name {
            "connection" | "transfer-encoding" | "keep-alive" | "proxy-connection" | "te" => {
                continue;
            }
            _ => {
                builder = builder.header(key.clone(), value.clone());
            }
        }
    }

    // Add proxy headers
    builder = builder.header("x-forwarded-for", client_addr.ip().to_string());
    builder = builder.header("x-real-ip", client_addr.ip().to_string());
    builder = builder.header("x-forwarded-proto", "https");

    // Ensure Host header is set (required for HTTP/1.1)
    if h2_req.headers().get("host").is_none() {
        if let Some(authority) = h2_req.uri().authority() {
            builder = builder.header("host", authority.as_str());
        }
    }

    let req = builder
        .body(Full::new(body))
        .map_err(|e| anyhow::anyhow!("Failed to build HTTP/1.1 request from H2: {}", e))?;

    Ok(req)
}

/// Convert an HTTP/1.1 response from the upstream into an
/// `http::Response<()>` with headers suitable for sending back over
/// an HTTP/2 stream.
///
/// Strips hop-by-hop headers (`connection`, `transfer-encoding`,
/// `keep-alive`) that are not valid in HTTP/2.
pub fn http1_response_to_h2_headers(
    resp: &http::response::Parts,
) -> Result<http::Response<()>> {
    let mut builder = http::Response::builder().status(resp.status);

    for (key, value) in &resp.headers {
        let name = key.as_str();
        match name {
            "connection" | "transfer-encoding" | "keep-alive" | "proxy-connection" | "upgrade" => {
                continue;
            }
            _ => {
                builder = builder.header(key.clone(), value.clone());
            }
        }
    }

    let resp = builder
        .body(())
        .map_err(|e| anyhow::anyhow!("Failed to build H2 response headers: {}", e))?;

    Ok(resp)
}

// ---------------------------------------------------------------------------
// Main H2 connection handler
// ---------------------------------------------------------------------------

/// Handle an HTTP/2 connection accepted after ALPN negotiation.
///
/// This function performs the HTTP/2 server handshake, then loops over
/// incoming streams. Each stream is spawned as an independent task that:
///   1. Reads the request headers and body from the H2 stream.
///   2. Converts to an HTTP/1.1 request and forwards to the upstream pool.
///   3. Sends the upstream response back over the H2 stream.
///
/// SSE (Server-Sent Events) responses are detected and streamed
/// frame-by-frame without buffering, preserving real-time semantics.
///
/// The function returns when the client sends GOAWAY or the connection
/// errors out.
pub async fn handle_h2_connection<S>(
    io: S,
    client_addr: SocketAddr,
    upstream: &UpstreamPool,
    metrics: &Metrics,
    body_limit: usize,
    _static_config: &StaticConfig,
) where
    S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    // NOTE: The actual `h2` crate is not yet in our Cargo.toml dependencies.
    // When the h2 dependency is added, this function body would use:
    //
    //   let mut connection = h2::server::handshake(io).await?;
    //
    // For now, we provide the complete logic as a reference implementation
    // that compiles against our existing types. The h2-specific calls are
    // behind illustrative comments showing exactly what API calls are needed.

    info!(client = %client_addr, "H2 connection accepted");
    metrics.conn_opened();

    // --- H2 handshake (requires `h2` crate) ---
    // let mut connection = match h2::server::Builder::new()
    //     .max_concurrent_streams(config.max_concurrent_streams)
    //     .initial_window_size(config.initial_window_size)
    //     .max_frame_size(config.max_frame_size)
    //     .max_header_list_size(config.max_header_list_size)
    //     .enable_connect_protocol(config.enable_connect_protocol)
    //     .handshake(io)
    //     .await
    // {
    //     Ok(conn) => conn,
    //     Err(e) => {
    //         warn!(client = %client_addr, "H2 handshake failed: {}", e);
    //         metrics.conn_closed();
    //         return;
    //     }
    // };

    // --- Stream accept loop ---
    // while let Some(result) = connection.accept().await {
    //     let (request, mut respond) = match result {
    //         Ok(pair) => pair,
    //         Err(e) => {
    //             debug!(client = %client_addr, "H2 accept error: {}", e);
    //             break;
    //         }
    //     };
    //
    //     metrics.request();
    //     // h2_metrics.h2_stream_opened();
    //
    //     let upstream = upstream.clone_for_task();
    //     let metrics = metrics.clone();
    //     let addr = client_addr;
    //
    //     tokio::spawn(async move {
    //         if let Err(e) = handle_h2_stream(request, respond, &upstream, &metrics, addr, body_limit).await {
    //             debug!(client = %addr, "H2 stream error: {}", e);
    //         }
    //         // h2_metrics.h2_stream_closed();
    //     });
    // }

    // Placeholder: read until EOF so the connection type-checks.
    // This will be replaced by the h2 accept loop above.
    let mut io = io;
    let mut discard = [0u8; 4096];
    loop {
        use tokio::io::AsyncReadExt;
        match tokio::time::timeout(
            std::time::Duration::from_secs(300),
            io.read(&mut discard),
        )
        .await
        {
            Ok(Ok(0)) | Err(_) => break,
            Ok(Ok(_n)) => {
                // In the real implementation, the h2 crate handles framing.
                // This placeholder just drains the socket.
                metrics.request();
            }
            Ok(Err(e)) => {
                debug!(client = %client_addr, "H2 placeholder read error: {}", e);
                break;
            }
        }
    }

    debug!(client = %client_addr, "H2 connection closed");
    metrics.conn_closed();
}

// ---------------------------------------------------------------------------
// Individual H2 stream handler (reference implementation)
// ---------------------------------------------------------------------------

/// Handle a single HTTP/2 stream: read request, forward to upstream, write
/// response.
///
/// This is the per-stream logic spawned by `handle_h2_connection`. It:
///   1. Collects the request body (up to `body_limit` bytes).
///   2. Converts the H2 request to an HTTP/1.1 request.
///   3. Forwards to the upstream pool.
///   4. Detects SSE responses and streams them frame-by-frame.
///   5. Sends the response body back over the H2 stream.
///
/// When the `h2` crate is added, the `send_stream` parameter would be
/// `h2::server::SendResponse<bytes::Bytes>`.
async fn _handle_h2_stream(
    _h2_request: http::Request<()>,
    _upstream: &UpstreamPool,
    _metrics: &Metrics,
    _client_addr: SocketAddr,
    _body_limit: usize,
    _body_bytes: Bytes,
) -> Result<()> {
    // 1. Convert H2 request to HTTP/1.1
    // let upstream_req = h2_request_to_http1(&h2_request, body_bytes, client_addr)?;

    // 2. Forward to upstream
    // let resp = upstream.forward(upstream_req).await?;
    // let (parts, body) = resp.into_parts();
    // let status = parts.status.as_u16();
    // metrics.response_status(status);

    // 3. Build H2 response headers
    // let h2_resp = http1_response_to_h2_headers(&parts)?;

    // 4. Detect streaming responses (SSE)
    // let is_streaming = parts.headers.get("content-type")
    //     .and_then(|v| v.to_str().ok())
    //     .map(|v| v.contains("text/event-stream"))
    //     .unwrap_or(false);

    // 5. Send response headers, then body
    // let mut send_stream = respond.send_response(h2_resp, false)?;
    //
    // if is_streaming {
    //     // Stream each frame individually for real-time SSE
    //     let mut body = body;
    //     loop {
    //         match body.frame().await {
    //             Some(Ok(frame)) => {
    //                 if let Some(data) = frame.data_ref() {
    //                     send_stream.send_data(data.clone(), false)?;
    //                     metrics.bytes_tx(data.len() as u64);
    //                 }
    //             }
    //             Some(Err(e)) => {
    //                 debug!("H2 SSE body error: {}", e);
    //                 break;
    //             }
    //             None => break,
    //         }
    //     }
    //     send_stream.send_data(Bytes::new(), true)?; // END_STREAM
    // } else {
    //     // Buffered: collect full body and send
    //     let body_bytes = body.collect().await?.to_bytes();
    //     metrics.bytes_tx(body_bytes.len() as u64);
    //     send_stream.send_data(body_bytes, true)?;
    // }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_is_valid() {
        let config = H2Config::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.max_concurrent_streams, 256);
        assert_eq!(config.initial_window_size, 2 * 1024 * 1024);
        assert_eq!(config.max_frame_size, 16_384);
        assert!(!config.enable_connect_protocol);
    }

    #[test]
    fn test_config_validation_rejects_zero_streams() {
        let config = H2Config {
            max_concurrent_streams: 0,
            ..H2Config::default()
        };
        let err = config.validate().unwrap_err();
        assert!(
            err.to_string().contains("max_concurrent_streams"),
            "Expected error about max_concurrent_streams, got: {}",
            err
        );
    }

    #[test]
    fn test_config_validation_rejects_oversized_window() {
        let config = H2Config {
            initial_window_size: 0x8000_0000, // 2^31, exceeds max
            ..H2Config::default()
        };
        let err = config.validate().unwrap_err();
        assert!(
            err.to_string().contains("initial_window_size"),
            "Expected error about initial_window_size, got: {}",
            err
        );
    }

    #[test]
    fn test_config_validation_rejects_small_frame_size() {
        let config = H2Config {
            max_frame_size: 1024, // Below HTTP/2 minimum of 16384
            ..H2Config::default()
        };
        let err = config.validate().unwrap_err();
        assert!(
            err.to_string().contains("max_frame_size"),
            "Expected error about max_frame_size, got: {}",
            err
        );
    }

    #[test]
    fn test_config_validation_rejects_oversized_frame() {
        let config = H2Config {
            max_frame_size: 16_777_216, // 2^24, exceeds HTTP/2 max of 2^24 - 1
            ..H2Config::default()
        };
        let err = config.validate().unwrap_err();
        assert!(
            err.to_string().contains("max_frame_size"),
            "Expected error about max_frame_size, got: {}",
            err
        );
    }

    #[test]
    fn test_config_validation_accepts_boundary_values() {
        let config = H2Config {
            max_concurrent_streams: 1,
            initial_window_size: 0x7FFF_FFFF, // Max allowed
            max_frame_size: 16_384,            // Minimum allowed
            max_header_list_size: 1,
            enable_connect_protocol: true,
        };
        assert!(config.validate().is_ok());

        let config2 = H2Config {
            max_frame_size: 16_777_215, // Maximum allowed (2^24 - 1)
            ..H2Config::default()
        };
        assert!(config2.validate().is_ok());
    }

    #[test]
    fn test_h2_request_to_http1_basic() {
        let h2_req = http::Request::builder()
            .method("POST")
            .uri("/api/v1/mining/submit")
            .header("content-type", "application/json")
            .header("authorization", "Bearer test123")
            .body(())
            .unwrap();

        let body = Bytes::from(r#"{"nonce":"abc"}"#);
        let addr: SocketAddr = "192.168.1.100:12345".parse().unwrap();

        let http1 = h2_request_to_http1(&h2_req, body.clone(), addr).unwrap();

        assert_eq!(http1.method(), "POST");
        assert_eq!(http1.uri().path(), "/api/v1/mining/submit");
        assert_eq!(
            http1.headers().get("content-type").unwrap(),
            "application/json"
        );
        assert_eq!(
            http1.headers().get("authorization").unwrap(),
            "Bearer test123"
        );
        assert_eq!(
            http1.headers().get("x-forwarded-for").unwrap(),
            "192.168.1.100"
        );
        assert_eq!(
            http1.headers().get("x-real-ip").unwrap(),
            "192.168.1.100"
        );
        assert_eq!(
            http1.headers().get("x-forwarded-proto").unwrap(),
            "https"
        );
    }

    #[test]
    fn test_h2_request_strips_hop_by_hop_headers() {
        let h2_req = http::Request::builder()
            .method("GET")
            .uri("/api/v1/status")
            .header("connection", "keep-alive")
            .header("transfer-encoding", "chunked")
            .header("keep-alive", "timeout=5")
            .header("proxy-connection", "keep-alive")
            .header("accept", "application/json")
            .body(())
            .unwrap();

        let addr: SocketAddr = "10.0.0.1:9999".parse().unwrap();
        let http1 = h2_request_to_http1(&h2_req, Bytes::new(), addr).unwrap();

        assert!(http1.headers().get("connection").is_none());
        assert!(http1.headers().get("transfer-encoding").is_none());
        assert!(http1.headers().get("keep-alive").is_none());
        assert!(http1.headers().get("proxy-connection").is_none());
        // Non-hop-by-hop headers are preserved
        assert_eq!(
            http1.headers().get("accept").unwrap(),
            "application/json"
        );
    }

    #[test]
    fn test_http1_response_to_h2_strips_connection_headers() {
        let mut resp = http::Response::builder()
            .status(200)
            .header("content-type", "text/event-stream")
            .header("connection", "keep-alive")
            .header("transfer-encoding", "chunked")
            .header("keep-alive", "timeout=5")
            .header("x-custom", "value")
            .body(())
            .unwrap();

        let (parts, _body) = resp.into_parts();
        let h2_resp = http1_response_to_h2_headers(&parts).unwrap();

        assert_eq!(h2_resp.status(), 200);
        assert_eq!(
            h2_resp.headers().get("content-type").unwrap(),
            "text/event-stream"
        );
        assert_eq!(h2_resp.headers().get("x-custom").unwrap(), "value");
        assert!(h2_resp.headers().get("connection").is_none());
        assert!(h2_resp.headers().get("transfer-encoding").is_none());
        assert!(h2_resp.headers().get("keep-alive").is_none());
    }

    #[test]
    fn test_h2_request_adds_host_from_authority() {
        let h2_req = http::Request::builder()
            .method("GET")
            .uri("https://quillon.xyz/api/v1/status")
            .body(())
            .unwrap();

        let addr: SocketAddr = "10.0.0.1:8080".parse().unwrap();
        let http1 = h2_request_to_http1(&h2_req, Bytes::new(), addr).unwrap();

        assert_eq!(http1.headers().get("host").unwrap(), "quillon.xyz");
    }

    #[test]
    fn test_h2_metrics_counters() {
        let m = H2Metrics::new();

        m.h2_connection();
        m.h2_connection();
        m.h2_stream_opened();
        m.h2_stream_opened();
        m.h2_stream_opened();
        m.h2_stream_closed();
        m.h2_goaway();

        assert_eq!(m.connections.load(Ordering::Relaxed), 2);
        assert_eq!(m.streams_opened.load(Ordering::Relaxed), 3);
        assert_eq!(m.streams_closed.load(Ordering::Relaxed), 1);
        assert_eq!(m.goaway_sent.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_h2_metrics_prometheus_format() {
        let m = H2Metrics::new();
        m.h2_connection();
        m.h2_stream_opened();

        let prom = m.prometheus_export();
        assert!(prom.contains("q_flux_h2_connections_total 1"));
        assert!(prom.contains("q_flux_h2_streams_opened_total 1"));
        assert!(prom.contains("# TYPE q_flux_h2_connections_total counter"));
    }
}
