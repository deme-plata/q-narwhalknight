//! HTTP/2 multiplexed proxy.
//!
//! Provides HTTP/2 frontend handling for browser clients. When ALPN negotiates
//! "h2", connections are handled by this module instead of the HTTP/1.1 proxy.
//!
//! Architecture:
//! - Frontend: HTTP/2 via hyper (browser <-> q-flux)
//! - Backend: HTTP/1.1 via hyper (q-flux <-> upstream) -- backend doesn't need H2
//! - Multiplexing: multiple streams over one TLS connection
//! - Flow control: per-stream and connection-level flow control
//! - Server push: disabled (not useful for API/mining traffic)

use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use bytes::Bytes;
use http_body_util::{BodyExt, Either, Full};
use hyper::body::Incoming;
use hyper::service::service_fn;
use hyper::{Request, Response};
use hyper_util::rt::TokioIo;
use tokio::io::{AsyncRead, AsyncWrite};
use tracing::{debug, info, warn};

use crate::access_log::{AccessLogger, log_access};
use crate::config::StaticConfig;
use crate::metrics::Metrics;
use crate::static_serve;
use crate::upstream::UpstreamPool;

/// Global H2-specific metrics instance (module-level, shared across connections).
pub static H2_METRICS: std::sync::LazyLock<H2Metrics> = std::sync::LazyLock::new(H2Metrics::new);

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

/// Export H2 metrics in Prometheus text format.
/// Called by the admin server's /metrics endpoint.
pub fn h2_prometheus_export() -> String {
    H2_METRICS.prometheus_export()
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
// Response body type
// ---------------------------------------------------------------------------

/// Response body for H2 handler: either a fully-buffered response (errors,
/// static files) or a pass-through upstream body (zero-copy streaming for SSE).
type H2Body = Either<Full<Bytes>, Incoming>;

// ---------------------------------------------------------------------------
// Main H2 connection handler
// ---------------------------------------------------------------------------

/// Handle an HTTP/2 connection accepted after ALPN negotiation.
///
/// Uses `hyper::server::conn::http2` to serve HTTP/2 on the TLS stream.
/// Each H2 stream/request is handled concurrently by hyper's built-in
/// multiplexing. Requests are forwarded to the upstream HTTP/1.1 pool.
///
/// SSE responses are streamed via `Incoming` body pass-through (zero-copy).
pub async fn handle_h2_connection<S>(
    io: S,
    client_addr: SocketAddr,
    upstream: Arc<UpstreamPool>,
    metrics: Metrics,
    body_limit: usize,
    static_config: Arc<StaticConfig>,
    access_logger: Option<AccessLogger>,
) where
    S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    let h2_metrics = &*H2_METRICS;
    h2_metrics.h2_connection();
    info!(client = %client_addr, "H2 connection accepted");

    let io = TokioIo::new(io);

    let service = service_fn(move |req: Request<Incoming>| {
        let upstream = upstream.clone();
        let metrics = metrics.clone();
        let static_config = static_config.clone();
        let access_logger = access_logger.clone();

        async move {
            let resp = handle_h2_request(
                req,
                client_addr,
                &upstream,
                &metrics,
                body_limit,
                &static_config,
                access_logger.as_ref(),
            )
            .await;
            Ok::<_, Infallible>(resp)
        }
    });

    let mut builder = hyper::server::conn::http2::Builder::new(
        hyper_util::rt::TokioExecutor::new(),
    );
    builder
        .max_concurrent_streams(256)
        .initial_stream_window_size(2 * 1024 * 1024) // 2 MiB
        .max_frame_size(16_384);               // 16 KiB

    if let Err(e) = builder.serve_connection(io, service).await {
        let msg = e.to_string();
        if !msg.contains("connection closed")
            && !msg.contains("broken pipe")
            && !msg.contains("reset by peer")
            && !msg.contains("stream error")
            && !msg.contains("GOAWAY")
        {
            debug!(client = %client_addr, "H2 connection error: {}", msg);
        }
    }

    debug!(client = %client_addr, "H2 connection closed");
}

// ---------------------------------------------------------------------------
// Per-request handler
// ---------------------------------------------------------------------------

/// Handle a single HTTP/2 request: static files, CORS, or proxy to upstream.
async fn handle_h2_request(
    req: Request<Incoming>,
    client_addr: SocketAddr,
    upstream: &UpstreamPool,
    metrics: &Metrics,
    body_limit: usize,
    static_config: &StaticConfig,
    access_logger: Option<&AccessLogger>,
) -> Response<H2Body> {
    let req_start = Instant::now();
    let h2_metrics = &*H2_METRICS;
    h2_metrics.h2_stream_opened();
    metrics.request();

    // Destructure request early so we own parts and body separately.
    let (parts, body) = req.into_parts();
    // CRITICAL: Use path_and_query() to preserve query string parameters!
    // .path() alone strips ?key=value which breaks OAuth2, search, etc.
    let req_path = parts.uri.path_and_query()
        .map(|pq| pq.as_str().to_string())
        .unwrap_or_else(|| parts.uri.path().to_string());
    let req_method = parts.method.clone();
    let user_agent = parts
        .headers
        .get(hyper::header::USER_AGENT)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    // 1. Static file routing
    if let static_serve::RouteResult::ServeFile(file_resp) =
        static_serve::route(&req_path, static_config)
    {
        if req_method == hyper::Method::OPTIONS {
            let resp = cors_preflight();
            let latency = req_start.elapsed();
            metrics.response_status(204);
            metrics.record_latency(latency);
            log_access(access_logger, client_addr, "OPTIONS", &req_path, 204, 0, 0, latency, user_agent.as_deref(), None, None);
            h2_metrics.h2_stream_closed();
            return resp;
        }

        let resp = serve_static_h2(&file_resp, &req_method, &parts.headers, metrics).await;
        let status = resp.status().as_u16();
        let latency = req_start.elapsed();
        metrics.response_status(status);
        metrics.record_latency(latency);
        log_access(access_logger, client_addr, req_method.as_str(), &req_path, status, 0, 0, latency, user_agent.as_deref(), None, None);
        h2_metrics.h2_stream_closed();
        return resp;
    }

    // 2. CORS preflight
    if req_method == hyper::Method::OPTIONS {
        let resp = cors_preflight();
        let latency = req_start.elapsed();
        metrics.response_status(204);
        metrics.record_latency(latency);
        log_access(access_logger, client_addr, "OPTIONS", &req_path, 204, 0, 0, latency, user_agent.as_deref(), None, None);
        h2_metrics.h2_stream_closed();
        return resp;
    }

    // 3. Collect request body (up to body_limit)
    let body_bytes = match collect_body(body, body_limit).await {
        Ok(b) => b,
        Err(_) => {
            let latency = req_start.elapsed();
            metrics.response_status(413);
            metrics.record_latency(latency);
            log_access(access_logger, client_addr, req_method.as_str(), &req_path, 413, 0, 0, latency, user_agent.as_deref(), None, None);
            h2_metrics.h2_stream_closed();
            return error_response(413, "Request body too large");
        }
    };

    let content_length = body_bytes.len() as u64;
    metrics.bytes_rx(content_length);

    // 4. Build upstream request (HTTP/1.1 to backend)
    let upstream_req = {
        let mut builder = hyper::Request::builder()
            .method(&req_method)
            .uri(&req_path);
        for (k, v) in &parts.headers {
            let name = k.as_str();
            match name {
                "connection" | "transfer-encoding" | "keep-alive" | "proxy-connection" | "te" => {
                    continue;
                }
                _ => {
                    builder = builder.header(k, v);
                }
            }
        }
        builder = builder.header("x-forwarded-for", client_addr.ip().to_string());
        builder = builder.header("x-real-ip", client_addr.ip().to_string());
        builder = builder.header("x-forwarded-proto", "https");
        builder.body(Full::new(body_bytes.clone())).unwrap()
    };

    // 5. Forward to upstream
    match upstream.forward(upstream_req).await {
        Ok((resp, backend_addr)) => {
            let status = resp.status().as_u16();

            // v9.2.6: Retry mining submissions on 503 — if primary backend is overloaded,
            // try alternate backend (e.g., Beta). Mining POSTs are idempotent (nonce-deduped).
            // Only retry POST /api/v1/mining/submit to avoid retrying non-idempotent requests.
            if status == 503 && req_method == hyper::Method::POST && req_path.contains("/mining/submit") {
                metrics.upstream_retry();
                // Drop the 503 response body, rebuild request for retry
                drop(resp);
                let retry_req = {
                    let mut builder = hyper::Request::builder()
                        .method(&req_method)
                        .uri(&req_path);
                    for (k, v) in &parts.headers {
                        let name = k.as_str();
                        match name {
                            "connection" | "transfer-encoding" | "keep-alive" | "proxy-connection" | "te" => continue,
                            _ => { builder = builder.header(k, v); }
                        }
                    }
                    builder = builder.header("x-forwarded-for", client_addr.ip().to_string());
                    builder = builder.header("x-real-ip", client_addr.ip().to_string());
                    builder = builder.header("x-forwarded-proto", "https");
                    builder.body(Full::new(body_bytes.clone())).unwrap()
                };
                if let Some(Ok((retry_resp, retry_backend))) = upstream.forward_excluding(retry_req, &backend_addr).await {
                    metrics.upstream_retry_success();
                    let retry_status = retry_resp.status().as_u16();
                    let latency = req_start.elapsed();
                    metrics.response_status(retry_status);
                    metrics.record_latency(latency);
                    log_access(access_logger, client_addr, req_method.as_str(), &req_path, retry_status, content_length, 0, latency, user_agent.as_deref(), Some(&retry_backend), Some("retry-503"));
                    h2_metrics.h2_stream_closed();

                    let (rp, rb) = retry_resp.into_parts();
                    let mut b = Response::builder().status(rp.status);
                    for (k, v) in &rp.headers {
                        let name = k.as_str();
                        match name {
                            "connection" | "transfer-encoding" | "keep-alive" | "proxy-connection" | "upgrade" => continue,
                            _ => { b = b.header(k, v); }
                        }
                    }
                    b = b.header("access-control-allow-origin", "*");
                    return b.body(Either::Right(rb)).unwrap_or_else(|_| error_response(500, "Internal proxy error"));
                }
                // Retry failed or no alternate backend — fall through to return 503
                let latency = req_start.elapsed();
                metrics.response_status(503);
                metrics.record_latency(latency);
                log_access(access_logger, client_addr, req_method.as_str(), &req_path, 503, content_length, 0, latency, user_agent.as_deref(), Some(&backend_addr), Some("503-no-retry"));
                h2_metrics.h2_stream_closed();
                return error_response(503, "Service Unavailable");
            }

            let latency = req_start.elapsed();
            metrics.response_status(status);
            metrics.record_latency(latency);
            log_access(access_logger, client_addr, req_method.as_str(), &req_path, status, content_length, 0, latency, user_agent.as_deref(), Some(&backend_addr), None);
            h2_metrics.h2_stream_closed();

            // Pass upstream response through with Incoming body (zero-copy).
            // hyper's HTTP/2 server handles framing + flow control automatically.
            let (resp_parts, resp_body) = resp.into_parts();
            let mut builder = Response::builder().status(resp_parts.status);
            let mut has_cors = false;
            for (k, v) in &resp_parts.headers {
                let name = k.as_str();
                match name {
                    "connection" | "transfer-encoding" | "keep-alive"
                    | "proxy-connection" | "upgrade" => continue,
                    _ => {
                        if name == "access-control-allow-origin" {
                            has_cors = true;
                        }
                        builder = builder.header(k, v);
                    }
                }
            }
            // Ensure CORS origin is always present for browser clients
            if !has_cors {
                builder = builder.header("access-control-allow-origin", "*");
            }
            builder
                .body(Either::Right(resp_body))
                .unwrap_or_else(|_| error_response(500, "Internal proxy error"))
        }
        Err((err, failed_backend)) => {
            // v9.2.6: Retry on connection failure for idempotent mining POSTs
            if req_method == hyper::Method::POST && req_path.contains("/mining/submit") && !failed_backend.is_empty() {
                let retry_req = {
                    let mut builder = hyper::Request::builder()
                        .method(&req_method)
                        .uri(&req_path);
                    for (k, v) in &parts.headers {
                        let name = k.as_str();
                        match name {
                            "connection" | "transfer-encoding" | "keep-alive" | "proxy-connection" | "te" => continue,
                            _ => { builder = builder.header(k, v); }
                        }
                    }
                    builder = builder.header("x-forwarded-for", client_addr.ip().to_string());
                    builder = builder.header("x-real-ip", client_addr.ip().to_string());
                    builder = builder.header("x-forwarded-proto", "https");
                    builder.body(Full::new(body_bytes.clone())).unwrap()
                };
                if let Some(Ok((retry_resp, retry_backend))) = upstream.forward_excluding(retry_req, &failed_backend).await {
                    let retry_status = retry_resp.status().as_u16();
                    let latency = req_start.elapsed();
                    metrics.response_status(retry_status);
                    metrics.record_latency(latency);
                    log_access(access_logger, client_addr, req_method.as_str(), &req_path, retry_status, content_length, 0, latency, user_agent.as_deref(), Some(&retry_backend), Some("retry-err"));
                    h2_metrics.h2_stream_closed();

                    let (rp, rb) = retry_resp.into_parts();
                    let mut b = Response::builder().status(rp.status);
                    for (k, v) in &rp.headers {
                        let name = k.as_str();
                        match name {
                            "connection" | "transfer-encoding" | "keep-alive" | "proxy-connection" | "upgrade" => continue,
                            _ => { b = b.header(k, v); }
                        }
                    }
                    b = b.header("access-control-allow-origin", "*");
                    return b.body(Either::Right(rb)).unwrap_or_else(|_| error_response(500, "Internal proxy error"));
                }
            }

            warn!(client = %client_addr, "H2 upstream error: {}", err);
            let latency = req_start.elapsed();
            metrics.response_status(502);
            metrics.record_latency(latency);
            log_access(access_logger, client_addr, req_method.as_str(), &req_path, 502, content_length, 0, latency, user_agent.as_deref(), None, None);
            h2_metrics.h2_stream_closed();
            error_response(502, "Bad Gateway")
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Collect request body up to `limit` bytes.
async fn collect_body(body: Incoming, limit: usize) -> Result<Bytes> {
    let collected = body
        .collect()
        .await
        .map_err(|e| anyhow::anyhow!("Body read error: {}", e))?;
    let bytes = collected.to_bytes();
    if bytes.len() > limit {
        anyhow::bail!("Body exceeds limit");
    }
    Ok(bytes)
}

/// Build an error response with a JSON body and CORS headers.
fn error_response(status: u16, msg: &str) -> Response<H2Body> {
    let body = format!("{{\"error\":\"{}\"}}", msg);
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .header("access-control-allow-origin", "*")
        .body(Either::Left(Full::new(Bytes::from(body))))
        .unwrap()
}

/// Build a 204 CORS preflight response.
fn cors_preflight() -> Response<H2Body> {
    Response::builder()
        .status(204)
        .header("access-control-allow-origin", "*")
        .header("access-control-allow-methods", "GET, POST, PUT, DELETE, OPTIONS")
        .header("access-control-allow-headers", "Content-Type, Authorization, X-Wallet-Address, X-Wallet-Signature, X-Wallet-Auth")
        .header("content-length", "0")
        .body(Either::Left(Full::new(Bytes::new())))
        .unwrap()
}

/// Serve a static file as an H2 response with gzip compression.
async fn serve_static_h2(
    file_resp: &static_serve::FileResponse,
    method: &hyper::Method,
    req_headers: &hyper::HeaderMap,
    metrics: &Metrics,
) -> Response<H2Body> {
    let metadata = match tokio::fs::metadata(&file_resp.path).await {
        Ok(m) => m,
        Err(_) => return error_response(404, "Not Found"),
    };
    let size = metadata.len();

    // ETag
    let etag = if let Ok(modified) = metadata.modified() {
        let dur = modified
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();
        format!("\"{:x}-{:x}\"", size, dur.as_secs())
    } else {
        format!("\"{:x}\"", size)
    };

    // 304 Not Modified
    if let Some(inm) = req_headers
        .get("if-none-match")
        .and_then(|v| v.to_str().ok())
    {
        if inm.trim() == etag || inm.contains(&etag) {
            return Response::builder()
                .status(304)
                .header("etag", &etag)
                .header("cache-control", file_resp.cache_control)
                .body(Either::Left(Full::new(Bytes::new())))
                .unwrap();
        }
    }

    // HEAD
    if *method == hyper::Method::HEAD {
        let mut builder = Response::builder()
            .status(200)
            .header("content-type", file_resp.mime)
            .header("content-length", size)
            .header("cache-control", file_resp.cache_control)
            .header("etag", &etag)
            .header("vary", "Accept-Encoding");
        if file_resp.is_download {
            let fname = file_resp.path.file_name().and_then(|n| n.to_str()).unwrap_or("download");
            builder = builder.header("content-disposition", format!("attachment; filename=\"{}\"", fname));
        }
        return builder.body(Either::Left(Full::new(Bytes::new()))).unwrap();
    }

    // Read file
    let body = match tokio::fs::read(&file_resp.path).await {
        Ok(b) => b,
        Err(e) => {
            warn!(path = %file_resp.path.display(), "Static file read error: {}", e);
            return error_response(500, "Internal Server Error");
        }
    };

    // Check if client accepts gzip and if this is a compressible type
    let accepts_gzip = req_headers.get("accept-encoding")
        .and_then(|v| v.to_str().ok())
        .map(|v| v.contains("gzip"))
        .unwrap_or(false);

    let is_compressible = file_resp.mime.starts_with("text/")
        || file_resp.mime.starts_with("application/javascript")
        || file_resp.mime.starts_with("application/json")
        || file_resp.mime.starts_with("application/xml")
        || file_resp.mime.starts_with("application/wasm")
        || file_resp.mime.starts_with("image/svg");

    // Gzip compress text assets
    if accepts_gzip && is_compressible && body.len() > 256 && !file_resp.is_download {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use std::io::Write;

        let mut encoder = GzEncoder::new(Vec::with_capacity(body.len() / 2), Compression::fast());
        if encoder.write_all(&body).is_ok() {
            if let Ok(compressed) = encoder.finish() {
                if compressed.len() < body.len() {
                    metrics.bytes_tx(compressed.len() as u64);
                    return Response::builder()
                        .status(200)
                        .header("content-type", file_resp.mime)
                        .header("content-length", compressed.len())
                        .header("content-encoding", "gzip")
                        .header("cache-control", file_resp.cache_control)
                        .header("etag", &etag)
                        .header("vary", "Accept-Encoding")
                        .body(Either::Left(Full::new(Bytes::from(compressed))))
                        .unwrap();
                }
            }
        }
    }

    // Uncompressed fallback
    metrics.bytes_tx(body.len() as u64);
    let mut builder = Response::builder()
        .status(200)
        .header("content-type", file_resp.mime)
        .header("content-length", body.len())
        .header("cache-control", file_resp.cache_control)
        .header("etag", &etag)
        .header("vary", "Accept-Encoding");
    if file_resp.is_download {
        let fname = file_resp.path.file_name().and_then(|n| n.to_str()).unwrap_or("download");
        builder = builder.header("content-disposition", format!("attachment; filename=\"{}\"", fname));
    }
    builder.body(Either::Left(Full::new(Bytes::from(body)))).unwrap()
}

// log_access is imported from access_log module (shared with proxy)

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
        let resp = http::Response::builder()
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
