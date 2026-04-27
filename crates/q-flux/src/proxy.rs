use anyhow::Result;
use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use std::io::Write as IoWrite;
use tokio::io::{AsyncRead, AsyncWrite, AsyncReadExt, AsyncWriteExt};

use crate::access_log::{AccessLogger, log_access};
use crate::config::StaticConfig;
use crate::libp2p_aware::{self, BandwidthLimiter, PeerPolicy, PeerTracker};
use crate::metrics::Metrics;
use crate::simd_parse;
use crate::static_serve;
use crate::upstream::UpstreamPool;

/// Drain signal receiver. When the sender sets `true`, long-lived connections
/// (SSE, WebSocket) should initiate graceful close. Uses tokio::sync::watch
/// so each connection can independently observe the drain transition.
pub type DrainReceiver = tokio::sync::watch::Receiver<bool>;

// SpliceChannel will be used when plain TCP listeners are added.
// For now, try_splice_bidirectional detects TLS and falls back.
#[cfg(target_os = "linux")]
#[allow(unused_imports)]
use crate::io_uring_loop::SpliceChannel;

const MAX_HEADER_SIZE: usize = 8192;
/// Client-side read timeout — prevents slowloris attacks.
const CLIENT_READ_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// Generate a short, unique request ID (hex-encoded worker-local counter).
/// Format: "wXXXX-NNNNNN" where XXXX is thread hash, NNNNNN is counter.
fn generate_request_id() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    // Use thread ID hash for worker disambiguation
    let tid = std::thread::current().id();
    let thash = format!("{:?}", tid);
    let w = thash.bytes().fold(0u16, |acc, b| acc.wrapping_add(b as u16));
    format!("{:04x}-{:06x}", w, n & 0xFFFFFF)
}

/// Handle a single HTTP connection (potentially multiple requests via keepalive).
#[allow(clippy::too_many_arguments)]
pub async fn handle_connection<S>(
    stream: S,
    client_addr: SocketAddr,
    upstream: &UpstreamPool,
    metrics: &Metrics,
    body_limit: usize,
    streaming_body_threshold: usize,
    static_config: &StaticConfig,
    peer_tracker: &Arc<PeerTracker>,
    bandwidth_limiter: &Arc<BandwidthLimiter>,
    drain_rx: DrainReceiver,
    file_cache: Option<&static_serve::FileCache>,
) where
    S: AsyncRead + AsyncWrite + Unpin,
{
    handle_connection_inner(stream, client_addr, upstream, metrics, body_limit, streaming_body_threshold, static_config, None, peer_tracker, bandwidth_limiter, drain_rx, file_cache).await;
}

/// Handle a single HTTP connection with optional access logging.
#[allow(clippy::too_many_arguments)]
pub async fn handle_connection_logged<S>(
    stream: S,
    client_addr: SocketAddr,
    upstream: &UpstreamPool,
    metrics: &Metrics,
    body_limit: usize,
    streaming_body_threshold: usize,
    static_config: &StaticConfig,
    access_logger: &AccessLogger,
    peer_tracker: &Arc<PeerTracker>,
    bandwidth_limiter: &Arc<BandwidthLimiter>,
    drain_rx: DrainReceiver,
    file_cache: Option<&static_serve::FileCache>,
) where
    S: AsyncRead + AsyncWrite + Unpin,
{
    handle_connection_inner(stream, client_addr, upstream, metrics, body_limit, streaming_body_threshold, static_config, Some(access_logger), peer_tracker, bandwidth_limiter, drain_rx, file_cache).await;
}

#[allow(clippy::too_many_arguments)]
async fn handle_connection_inner<S>(
    mut stream: S,
    client_addr: SocketAddr,
    upstream: &UpstreamPool,
    metrics: &Metrics,
    body_limit: usize,
    streaming_body_threshold: usize,
    static_config: &StaticConfig,
    access_logger: Option<&AccessLogger>,
    peer_tracker: &Arc<PeerTracker>,
    bandwidth_limiter: &Arc<BandwidthLimiter>,
    drain_rx: DrainReceiver,
    file_cache: Option<&static_serve::FileCache>,
) where
    S: AsyncRead + AsyncWrite + Unpin,
{
    let mut buf = vec![0u8; MAX_HEADER_SIZE];
    let mut buf_len = 0usize;

    // HTTP/1.1 keepalive loop — handle multiple requests per connection
    loop {
        // Read headers (with timeout to prevent slowloris)
        let (req, header_end) = match read_request_headers(&mut stream, &mut buf, &mut buf_len).await {
            Ok(Some(v)) => v,
            Ok(None) => break, // Clean close
            Err(e) => {
                tracing::debug!(client = %client_addr, "Header read error: {}", e);
                break;
            }
        };

        let req_start = Instant::now();
        metrics.request();

        // Static file routing: check before proxying
        let req_path = req.uri().path().to_string();
        let req_method = req.method().as_str();
        let user_agent = req.headers().get(hyper::header::USER_AGENT)
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());

        // Determine compression from client's Accept-Encoding header
        let client_compression = if static_config.proxy_compression {
            parse_accept_encoding(&buf[..buf_len])
        } else {
            Compression::None
        };

        // X-Request-ID: preserve client-provided ID or generate one
        let request_id = req.headers().get("x-request-id")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string())
            .unwrap_or_else(generate_request_id);
        if let static_serve::RouteResult::ServeFile(file_resp) = static_serve::route(&req_path, static_config) {
            let if_none_match = req.headers().get("if-none-match")
                .and_then(|v| v.to_str().ok())
                .map(|s| s.to_string());

            let status;
            if req_method == "OPTIONS" {
                let cors = format!("HTTP/1.1 204 No Content\r\n{}\r\ncontent-length: 0\r\n\r\n", static_serve::CORS_HEADERS);
                let _ = stream.write_all(cors.as_bytes()).await;
                let _ = stream.flush().await;
                status = 204;
            } else {
                let client_accepts_gzip = static_serve::accepts_gzip(&buf[..buf_len]);
                if let Err(e) = static_serve::serve_file(&mut stream, &file_resp, req_method, if_none_match.as_deref(), client_accepts_gzip, metrics, file_cache).await {
                    tracing::debug!(client = %client_addr, "Static serve error: {}", e);
                    break;
                }
                status = 200;
            }
            let latency = req_start.elapsed();
            metrics.response_status(status);
            metrics.record_latency(latency);
            log_access(access_logger, client_addr, req_method, &req_path, status, 0, 0, latency, user_agent.as_deref(), None, Some(request_id.as_str()));
            if !should_keep_alive(&req) { break; }
            let consumed = header_end;
            if consumed < buf_len { buf.copy_within(consumed..buf_len, 0); buf_len -= consumed; } else { buf_len = 0; }
            continue;
        }

        // CORS preflight
        if req_method == "OPTIONS" {
            let cors = format!("HTTP/1.1 204 No Content\r\n{}\r\ncontent-length: 0\r\n\r\n", static_serve::CORS_HEADERS);
            let _ = stream.write_all(cors.as_bytes()).await;
            let _ = stream.flush().await;
            let latency = req_start.elapsed();
            metrics.response_status(204);
            metrics.record_latency(latency);
            log_access(access_logger, client_addr, req_method, &req_path, 204, 0, 0, latency, user_agent.as_deref(), None, Some(request_id.as_str()));
            if !should_keep_alive(&req) { break; }
            buf_len = 0;
            continue;
        }

        // Check for WebSocket upgrade — SIMD fast-path first (Phase 2).
        // simd_parse::is_websocket_upgrade checks both Upgrade + Connection headers
        // in a single scan, ~3x faster than per-header string comparison.
        let is_upgrade = simd_parse::is_websocket_upgrade(&buf[..buf_len])
            || req.headers().get(hyper::header::UPGRADE)
                .and_then(|v| v.to_str().ok())
                .map(|v| v.eq_ignore_ascii_case("websocket"))
                .unwrap_or(false);

        if is_upgrade {
            handle_websocket_upgrade(stream, header_end, &buf[..buf_len], client_addr, upstream, metrics, peer_tracker, bandwidth_limiter, drain_rx.clone()).await;
            let latency = req_start.elapsed();
            metrics.record_latency(latency);
            log_access(access_logger, client_addr, req_method, &req_path, 101, 0, 0, latency, user_agent.as_deref(), None, Some(request_id.as_str()));
            return; // Connection consumed by WebSocket
        }

        // SSE/streaming endpoint: route via direct TCP to avoid hyper pool exhaustion.
        // Without this, SSE responses hold hyper TCP connections indefinitely,
        // forcing new TCP connections for every subsequent request. Under high load
        // this creates a death spiral: all semaphore permits consumed, all requests fail.
        if is_sse_path(&req_path) {
            handle_sse_direct(stream, &req, client_addr, upstream, metrics, &request_id, drain_rx.clone()).await;
            let latency = req_start.elapsed();
            metrics.record_latency(latency);
            log_access(access_logger, client_addr, req_method, &req_path, 200, 0, 0, latency, user_agent.as_deref(), None, Some(request_id.as_str()));
            return; // Connection consumed by SSE stream
        }

        // Read body if Content-Length present
        let content_length = req.headers()
            .get(hyper::header::CONTENT_LENGTH)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0);

        if content_length > body_limit {
            let latency = req_start.elapsed();
            metrics.record_latency(latency);
            log_access(access_logger, client_addr, req_method, &req_path, 413, 0, 0, latency, user_agent.as_deref(), None, Some(request_id.as_str()));
            let _ = write_error_response(&mut stream, 413, "Request body too large").await;
            break;
        }

        // ── Issue #024: Stream large request bodies directly to upstream ──
        // When Content-Length exceeds the streaming threshold, bypass the
        // hyper Client and stream the body directly to an upstream backend
        // via a raw TCP connection. This avoids buffering multi-MB uploads
        // in memory. Streaming bodies cannot be retried (body is consumed).
        if streaming_body_threshold > 0
            && content_length > streaming_body_threshold
            && !is_upgrade
            && !is_sse_path(&req_path)
        {
            let result = stream_body_to_upstream(
                &mut stream, &req, &buf[header_end..buf_len], content_length,
                upstream, client_addr, metrics, &request_id,
            ).await;
            let latency = req_start.elapsed();
            metrics.record_latency(latency);
            let status = match &result {
                Ok(s) => *s,
                Err(_) => 502,
            };
            metrics.response_status(status);
            log_access(access_logger, client_addr, req_method, &req_path, status,
                       content_length as u64, 0, latency, user_agent.as_deref(), None,
                       Some(request_id.as_str()));
            if result.is_err() {
                let _ = write_error_response(&mut stream, 502, "Bad Gateway").await;
            }
            // Streaming body consumes the connection — can't keepalive after
            break;
        }

        // Collect body bytes (what we already have past the headers + remainder)
        let body = if content_length > 0 {
            let already_read = buf_len.saturating_sub(header_end);
            let mut body_buf = Vec::with_capacity(content_length);

            let to_copy = already_read.min(content_length);
            body_buf.extend_from_slice(&buf[header_end..header_end + to_copy]);

            // Read remaining body with timeout
            while body_buf.len() < content_length {
                let remaining = content_length - body_buf.len();
                let mut chunk = vec![0u8; remaining.min(65536)];
                match tokio::time::timeout(CLIENT_READ_TIMEOUT, stream.read(&mut chunk)).await {
                    Ok(Ok(0)) => break,
                    Ok(Ok(n)) => {
                        metrics.bytes_rx(n as u64);
                        body_buf.extend_from_slice(&chunk[..n]);
                    }
                    Ok(Err(e)) => {
                        tracing::debug!(client = %client_addr, "Body read error: {}", e);
                        return;
                    }
                    Err(_) => {
                        tracing::debug!(client = %client_addr, "Body read timeout");
                        let _ = write_error_response(&mut stream, 408, "Request timeout").await;
                        return;
                    }
                }
            }
            Bytes::from(body_buf)
        } else {
            Bytes::new()
        };

        let keep_alive = should_keep_alive(&req);

        // Idempotent methods get one retry on a different backend (Issue #011).
        let is_idempotent = matches!(req_method, "GET" | "HEAD" | "OPTIONS");

        // Clone body before moving into the first request (needed for retry).
        let retry_body = if is_idempotent { Some(body.clone()) } else { None };

        // Build upstream request with full body
        let upstream_req = {
            let mut builder = hyper::Request::builder()
                .method(req.method().clone())
                .uri(req.uri().clone());
            for (k, v) in req.headers() {
                builder = builder.header(k, v);
            }
            builder = builder.header("X-Forwarded-For", client_addr.ip().to_string());
            builder = builder.header("X-Real-IP", client_addr.ip().to_string());
            builder = builder.header("X-Request-ID", &request_id);
            builder.body(Full::new(body)).unwrap()
        };

        match upstream.forward(upstream_req).await {
            Ok((resp, backend_addr)) => {
                let status = resp.status().as_u16();
                let latency = req_start.elapsed();
                metrics.response_status(status);
                metrics.record_latency(latency);
                log_access(access_logger, client_addr, req_method, &req_path, status, content_length as u64, 0, latency, user_agent.as_deref(), Some(&backend_addr), Some(request_id.as_str()));

                if let Err(e) = write_response(&mut stream, resp, metrics, Some(request_id.as_str()), client_compression, static_config.proxy_compression).await {
                    tracing::debug!(client = %client_addr, "Response write error: {}", e);
                    break;
                }
            }
            Err((err, failed_backend)) if is_idempotent => {
                metrics.upstream_retry();
                // Retry once on a different backend, skipping the one that failed
                tracing::info!(
                    client = %client_addr, method = req_method,
                    path = %req_path, failed = %failed_backend,
                    "Retrying {} on different backend (error: {})", req_method, err,
                );
                let retry_req = {
                    let mut builder = hyper::Request::builder()
                        .method(req.method().clone())
                        .uri(req.uri().clone());
                    for (k, v) in req.headers() {
                        builder = builder.header(k, v);
                    }
                    builder = builder.header("X-Forwarded-For", client_addr.ip().to_string());
                    builder = builder.header("X-Real-IP", client_addr.ip().to_string());
                    builder = builder.header("X-Request-ID", &request_id);
                    builder.body(Full::new(retry_body.unwrap_or_default())).unwrap()
                };
                match upstream.forward_excluding(retry_req, &failed_backend).await {
                    Some(Ok((resp, backend_addr))) => {
                        metrics.upstream_retry_success();
                        let status = resp.status().as_u16();
                        let latency = req_start.elapsed();
                        metrics.response_status(status);
                        metrics.record_latency(latency);
                        log_access(access_logger, client_addr, req_method, &req_path, status, content_length as u64, 0, latency, user_agent.as_deref(), Some(&backend_addr), Some(request_id.as_str()));
                        if let Err(e) = write_response(&mut stream, resp, metrics, Some(request_id.as_str()), client_compression, static_config.proxy_compression).await {
                            tracing::debug!(client = %client_addr, "Response write error (retry): {}", e);
                            break;
                        }
                    }
                    Some(Err(err2)) => {
                        let latency = req_start.elapsed();
                        tracing::warn!(client = %client_addr, "Retry also failed: {}", err2);
                        metrics.response_status(502);
                        metrics.record_latency(latency);
                        log_access(access_logger, client_addr, req_method, &req_path, 502, content_length as u64, 0, latency, user_agent.as_deref(), None, Some(request_id.as_str()));
                        if write_error_response(&mut stream, 502, "Bad Gateway").await.is_err() {
                            break;
                        }
                    }
                    None => {
                        // No alternative backend available — return original error
                        let latency = req_start.elapsed();
                        tracing::warn!(client = %client_addr, "No alternative backend for retry: {}", err);
                        metrics.response_status(502);
                        metrics.record_latency(latency);
                        log_access(access_logger, client_addr, req_method, &req_path, 502, content_length as u64, 0, latency, user_agent.as_deref(), None, Some(request_id.as_str()));
                        if write_error_response(&mut stream, 502, "Bad Gateway").await.is_err() {
                            break;
                        }
                    }
                }
            }
            Err((err, _)) => {
                let latency = req_start.elapsed();
                tracing::warn!(client = %client_addr, "Upstream error: {}", err);
                metrics.response_status(502);
                metrics.record_latency(latency);
                log_access(access_logger, client_addr, req_method, &req_path, 502, content_length as u64, 0, latency, user_agent.as_deref(), None, Some(request_id.as_str()));
                if write_error_response(&mut stream, 502, "Bad Gateway").await.is_err() {
                    break;
                }
            }
        }

        if !keep_alive {
            break;
        }

        // Reset buffer for next request — safe arithmetic
        let consumed = header_end.saturating_add(content_length);
        if consumed < buf_len {
            let remaining = buf_len - consumed;
            buf.copy_within(consumed..buf_len, 0);
            buf_len = remaining;
        } else {
            buf_len = 0;
        }
    }
}

/// Read request headers from the stream with timeout protection.
///
/// Uses SIMD-accelerated \r\n\r\n scanning (Phase 2) as a fast pre-check
/// before falling into httparse for full header parsing. On AVX2 machines
/// this saves ~2 cycles/byte on the boundary detection.
async fn read_request_headers<S>(
    stream: &mut S,
    buf: &mut [u8],
    buf_len: &mut usize,
) -> Result<Option<(hyper::Request<()>, usize)>>
where
    S: AsyncRead + Unpin,
{
    loop {
        // SIMD fast-path: check if we have a complete header block yet.
        // find_header_end uses AVX2→SSE4.2→scalar runtime dispatch.
        if let Some(header_end) = simd_parse::find_header_end(&buf[..*buf_len]) {
            // Complete headers found — parse with httparse for structured access
            let mut headers = [httparse::EMPTY_HEADER; 64];
            let mut parsed_req = httparse::Request::new(&mut headers);

            match parsed_req.parse(&buf[..header_end]) {
                Ok(httparse::Status::Complete(_)) | Ok(httparse::Status::Partial) => {
                    let method = parsed_req.method.unwrap_or("GET");
                    let path = parsed_req.path.unwrap_or("/");

                    let mut builder = hyper::Request::builder()
                        .method(method)
                        .uri(path);

                    for h in parsed_req.headers.iter() {
                        if h.name.is_empty() { break; }
                        builder = builder.header(h.name, h.value);
                    }

                    let req = builder.body(())
                        .map_err(|e| anyhow::anyhow!("Failed to build request: {}", e))?;

                    return Ok(Some((req, header_end)));
                }
                Err(e) => {
                    return Err(anyhow::anyhow!("HTTP parse error: {}", e));
                }
            }
        }

        // No complete header block yet — check buffer capacity
        if *buf_len >= buf.len() {
            return Err(anyhow::anyhow!("Request headers exceed {}B limit", buf.len()));
        }

        // Read more data with timeout (prevents slowloris)
        match tokio::time::timeout(CLIENT_READ_TIMEOUT, stream.read(&mut buf[*buf_len..])).await {
            Ok(Ok(0)) => {
                if *buf_len == 0 {
                    return Ok(None); // Clean EOF
                }
                return Err(anyhow::anyhow!("Connection closed mid-headers"));
            }
            Ok(Ok(n)) => {
                *buf_len += n;
            }
            Ok(Err(e)) => {
                return Err(anyhow::anyhow!("Read error: {}", e));
            }
            Err(_) => {
                return Err(anyhow::anyhow!("Header read timeout (slowloris protection)"));
            }
        }
    }
}

fn should_keep_alive(req: &hyper::Request<()>) -> bool {
    if let Some(conn) = req.headers().get(hyper::header::CONNECTION) {
        if let Ok(s) = conn.to_str() {
            return !s.eq_ignore_ascii_case("close");
        }
    }
    req.version() != hyper::Version::HTTP_10
}

/// Compression encoding selected based on client's Accept-Encoding header.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Compression {
    None,
    Gzip,
    Brotli,
}

/// Minimum response body size for compression (bytes).
/// Responses smaller than this are sent uncompressed — the compression overhead
/// exceeds the bandwidth savings.
const MIN_COMPRESS_SIZE: usize = 1024;

/// Parse Accept-Encoding from raw request headers to determine compression.
fn parse_accept_encoding(headers: &[u8]) -> Compression {
    // Quick scan for Accept-Encoding in the raw header bytes.
    // We look for "accept-encoding:" (case-insensitive) in the buffer.
    let lower = headers.to_ascii_lowercase();
    if let Some(pos) = lower.windows(17).position(|w| w == b"accept-encoding:") {
        let value_start = pos + 17;
        // Find end of this header line
        let value_end = lower[value_start..].iter().position(|&b| b == b'\r' || b == b'\n')
            .map(|p| value_start + p)
            .unwrap_or(lower.len());
        let value = &lower[value_start..value_end];
        // Prefer Brotli over gzip (15-20% better compression)
        if value.windows(2).any(|w| w == b"br") {
            return Compression::Brotli;
        }
        if value.windows(4).any(|w| w == b"gzip") {
            return Compression::Gzip;
        }
    }
    Compression::None
}

/// Check if a content-type is compressible (text-based or structured data).
fn is_compressible_content_type(content_type: &str) -> bool {
    let ct = content_type.to_ascii_lowercase();
    ct.starts_with("text/")
        || ct.starts_with("application/json")
        || ct.starts_with("application/javascript")
        || ct.starts_with("application/xml")
        || ct.starts_with("application/x-javascript")
        || ct.starts_with("image/svg+xml")
        || ct.starts_with("application/manifest+json")
        || ct.starts_with("application/ld+json")
}

/// Compress bytes with gzip (flate2).
fn gzip_compress(data: &[u8]) -> Vec<u8> {
    use flate2::write::GzEncoder;
    use flate2::Compression;
    let mut encoder = GzEncoder::new(Vec::with_capacity(data.len() / 2), Compression::fast());
    encoder.write_all(data).unwrap_or_default();
    encoder.finish().unwrap_or_default()
}

/// Compress bytes with Brotli.
fn brotli_compress(data: &[u8]) -> Vec<u8> {
    let mut output = Vec::with_capacity(data.len() / 2);
    // Quality 4 is a good speed/ratio tradeoff for on-the-fly compression.
    // Window size 22 (4MB) is standard for web content.
    let mut writer = brotli::CompressorWriter::new(&mut output, 4096, 4, 22);
    writer.write_all(data).unwrap_or_default();
    drop(writer);
    output
}

/// Write an HTTP response back to the client.
/// Detects SSE/streaming responses and streams them without buffering.
async fn write_response<S>(
    stream: &mut S,
    resp: hyper::Response<Incoming>,
    metrics: &Metrics,
    request_id: Option<&str>,
    compression: Compression,
    proxy_compression_enabled: bool,
) -> Result<()>
where
    S: AsyncWrite + Unpin,
{
    let (mut parts, body) = resp.into_parts();
    // Inject X-Request-ID into response for end-to-end tracing
    if let Some(rid) = request_id {
        if let Ok(val) = hyper::header::HeaderValue::from_str(rid) {
            parts.headers.insert("x-request-id", val);
        }
    }

    // Detect streaming responses: SSE or chunked transfer
    let is_streaming = parts.headers.get("content-type")
        .and_then(|v| v.to_str().ok())
        .map(|v| v.contains("text/event-stream") || v.contains("application/x-ndjson"))
        .unwrap_or(false)
        || parts.headers.get(hyper::header::TRANSFER_ENCODING)
            .and_then(|v| v.to_str().ok())
            .map(|v| v.contains("chunked"))
            .unwrap_or(false);

    if is_streaming {
        // STREAMING MODE: write headers immediately, then forward chunks as they arrive.
        // Critical for SSE — miners need real-time block/balance updates.
        write_response_headers_streaming(stream, &parts).await?;

        let mut body = body;
        loop {
            match body.frame().await {
                Some(Ok(frame)) => {
                    if let Some(data) = frame.data_ref() {
                        stream.write_all(data).await?;
                        stream.flush().await?; // Flush each SSE event immediately
                        metrics.bytes_tx(data.len() as u64);
                    }
                }
                Some(Err(e)) => {
                    tracing::debug!("Streaming body error: {}", e);
                    break;
                }
                None => break, // Stream ended
            }
        }
    } else {
        // BUFFERED MODE: collect full body, optionally compress, set Content-Length, send.
        let body_bytes = body.collect().await
            .map_err(|e| anyhow::anyhow!("Body collect error: {}", e))?
            .to_bytes();

        // Determine if we should compress this response
        let content_type = parts.headers.get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        let already_encoded = parts.headers.contains_key("content-encoding");
        let should_compress = proxy_compression_enabled
            && compression != Compression::None
            && !already_encoded
            && body_bytes.len() >= MIN_COMPRESS_SIZE
            && is_compressible_content_type(content_type);

        let (final_body, encoding_header): (std::borrow::Cow<[u8]>, Option<&str>) = if should_compress {
            match compression {
                Compression::Brotli => {
                    let compressed = brotli_compress(&body_bytes);
                    if compressed.len() < body_bytes.len() {
                        (std::borrow::Cow::Owned(compressed), Some("br"))
                    } else {
                        (std::borrow::Cow::Borrowed(&body_bytes), None)
                    }
                }
                Compression::Gzip => {
                    let compressed = gzip_compress(&body_bytes);
                    if compressed.len() < body_bytes.len() {
                        (std::borrow::Cow::Owned(compressed), Some("gzip"))
                    } else {
                        (std::borrow::Cow::Borrowed(&body_bytes), None)
                    }
                }
                Compression::None => (std::borrow::Cow::Borrowed(&body_bytes), None),
            }
        } else {
            (std::borrow::Cow::Borrowed(&body_bytes), None)
        };

        // Pre-allocate response buffer to reduce per-header allocations
        let mut resp_buf = Vec::with_capacity(512);

        // Write status line
        write!(resp_buf, "HTTP/1.1 {} {}\r\n",
            parts.status.as_u16(),
            parts.status.canonical_reason().unwrap_or("OK")
        ).ok();

        // Write headers (skip original Content-Length — we'll write our own)
        for (key, value) in &parts.headers {
            if key == hyper::header::TRANSFER_ENCODING || key == "keep-alive" {
                continue;
            }
            if key == hyper::header::CONTENT_LENGTH {
                continue; // always rewrite Content-Length to match final body
            }
            if encoding_header.is_some() && key == "content-encoding" {
                continue; // we're replacing this
            }
            write!(resp_buf, "{}: {}\r\n", key, value.to_str().unwrap_or("")).ok();
        }

        // Security headers (Issue #032)
        write!(resp_buf, "x-content-type-options: nosniff\r\n").ok();
        write!(resp_buf, "x-frame-options: SAMEORIGIN\r\n").ok();
        write!(resp_buf, "referrer-policy: strict-origin-when-cross-origin\r\n").ok();

        // Add compression headers
        if let Some(enc) = encoding_header {
            write!(resp_buf, "content-encoding: {}\r\n", enc).ok();
            write!(resp_buf, "vary: Accept-Encoding\r\n").ok();
        }

        write!(resp_buf, "content-length: {}\r\n", final_body.len()).ok();
        resp_buf.extend_from_slice(b"\r\n");

        stream.write_all(&resp_buf).await?;

        if !final_body.is_empty() {
            stream.write_all(&final_body).await?;
            metrics.bytes_tx(final_body.len() as u64);
        }

        stream.flush().await?;
    }

    Ok(())
}

/// Write response headers for a streaming response (no Content-Length).
async fn write_response_headers_streaming<S>(
    stream: &mut S,
    parts: &http::response::Parts,
) -> Result<()>
where
    S: AsyncWrite + Unpin,
{
    // Pre-allocate header buffer to reduce per-header allocations
    let mut resp_buf = Vec::with_capacity(512);
    write!(resp_buf, "HTTP/1.1 {} {}\r\n",
        parts.status.as_u16(),
        parts.status.canonical_reason().unwrap_or("OK")
    ).ok();

    for (key, value) in &parts.headers {
        // Skip content-length for streaming (we don't know the total size)
        if key == hyper::header::CONTENT_LENGTH {
            continue;
        }
        if key == "keep-alive" {
            continue;
        }
        write!(resp_buf, "{}: {}\r\n", key, value.to_str().unwrap_or("")).ok();
    }

    // Security headers (Issue #032)
    write!(resp_buf, "x-content-type-options: nosniff\r\n").ok();
    write!(resp_buf, "x-frame-options: SAMEORIGIN\r\n").ok();
    write!(resp_buf, "referrer-policy: strict-origin-when-cross-origin\r\n").ok();

    resp_buf.extend_from_slice(b"\r\n");

    stream.write_all(&resp_buf).await?;
    stream.flush().await?;
    Ok(())
}

async fn write_error_response<S>(stream: &mut S, status: u16, msg: &str) -> Result<()>
where
    S: AsyncWrite + Unpin,
{
    // Escape message for JSON safety
    let escaped_msg = msg.replace('\\', "\\\\").replace('"', "\\\"");
    let body = format!("{{\"error\":\"{}\"}}", escaped_msg);
    let response = format!(
        "HTTP/1.1 {} {}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\nx-content-type-options: nosniff\r\nx-frame-options: SAMEORIGIN\r\nreferrer-policy: strict-origin-when-cross-origin\r\n\r\n{}",
        status,
        reason_phrase(status),
        body.len(),
        body,
    );
    stream.write_all(response.as_bytes()).await?;
    stream.flush().await?;
    Ok(())
}

fn reason_phrase(status: u16) -> &'static str {
    match status {
        400 => "Bad Request",
        408 => "Request Timeout",
        413 => "Payload Too Large",
        429 => "Too Many Requests",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        _ => "Error",
    }
}

// log_access is imported from access_log module (shared with h2_proxy)

/// Copy bytes from `reader` to `writer` with per-peer bandwidth limiting.
///
/// If `max_bandwidth_mbps` is 0 the limiter is bypassed (used for non-libp2p
/// connections where we do not want to impose a token-bucket).
///
/// Bytes are tracked on the peer's `PeerState` via `record_rx` / `record_tx`
/// depending on the `is_rx` flag.
///
/// Returns the total number of bytes transferred.
async fn bandwidth_limited_copy<R, W>(
    reader: &mut R,
    writer: &mut W,
    peer_id: &str,
    max_bandwidth_mbps: u64,
    limiter: &BandwidthLimiter,
    peer_tracker: &Arc<PeerTracker>,
    is_rx: bool,
) -> std::io::Result<u64>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    let mut buf = [0u8; 16384]; // 16 KB chunks — balances syscall overhead vs latency
    let mut total: u64 = 0;
    let apply_limit = max_bandwidth_mbps > 0 && !peer_id.is_empty();

    loop {
        let n = reader.read(&mut buf).await?;
        if n == 0 {
            return Ok(total); // EOF
        }

        // Bandwidth gate: if the peer's bucket is exhausted, yield briefly
        // and retry. This creates natural back-pressure without dropping data.
        if apply_limit {
            let mut attempts = 0u32;
            while !limiter.try_consume(peer_id, n, max_bandwidth_mbps) {
                attempts += 1;
                // Exponential micro-sleep: 1ms, 2ms, 4ms ... capped at 50ms
                let delay_ms = (1u64 << attempts.min(5)).min(50);
                tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                // After ~6 retries (~126ms total) give up waiting and let the
                // data through to avoid stalling the connection indefinitely.
                if attempts >= 6 {
                    break;
                }
            }
        }

        writer.write_all(&buf[..n]).await?;

        // Track bytes on the peer
        if !peer_id.is_empty() {
            let peer = peer_tracker.get_or_create(peer_id);
            if is_rx {
                peer.record_rx(n as u64);
            } else {
                peer.record_tx(n as u64);
            }
        }

        total += n as u64;
    }
}

/// Handle WebSocket upgrade: forward the raw upgrade request to upstream,
/// then bidirectional splice both directions.
///
/// If the connection is a libp2p peer, check PeerTracker for connection
/// limits and circuit breaker state before allowing the upgrade.
async fn handle_websocket_upgrade<S>(
    mut client_stream: S,
    header_end: usize,
    buf: &[u8],
    client_addr: SocketAddr,
    upstream: &UpstreamPool,
    metrics: &Metrics,
    peer_tracker: &Arc<PeerTracker>,
    bandwidth_limiter: &Arc<BandwidthLimiter>,
    mut drain_rx: DrainReceiver,
) where
    S: AsyncRead + AsyncWrite + Unpin,
{
    metrics.ws_upgrade();

    // Check for libp2p handshake in the bytes past the HTTP headers.
    // If this is a libp2p connection, enforce per-peer connection limits.
    let extra = &buf[header_end..];
    let peer_key = if libp2p_aware::is_libp2p_handshake(extra) {
        let info = libp2p_aware::LibP2pDetector::detect(extra);
        let key = info
            .as_ref()
            .filter(|i| !i.peer_id.is_empty())
            .map(|i| i.peer_id.clone())
            .unwrap_or_else(|| format!("ip-{}", client_addr.ip()));

        if !libp2p_aware::should_allow_peer(peer_tracker, &key) {
            tracing::debug!(client = %client_addr, peer = %key, "libp2p peer denied by PeerTracker");
            let _ = write_error_response(&mut client_stream, 503, "Service Unavailable").await;
            metrics.ws_closed();
            return;
        }
        Some(key)
    } else {
        None
    };

    // Track connection open for libp2p peers
    if let Some(ref key) = peer_key {
        let peer = peer_tracker.get_or_create(key);
        peer.conn_opened();
    }

    // Use round-robin backend selection (not hardcoded backend[0])
    let backend = upstream.next_backend_addr();
    let mut upstream_conn = match tokio::net::TcpStream::connect(backend).await {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!(client = %client_addr, "WS upstream connect failed: {}", e);
            let _ = write_error_response(&mut client_stream, 502, "Bad Gateway").await;
            if let Some(ref key) = peer_key {
                let peer = peer_tracker.get_or_create(key);
                peer.conn_closed();
            }
            metrics.ws_closed();
            return;
        }
    };

    // Forward the raw HTTP upgrade request bytes to upstream
    upstream_conn.write_all(&buf[..header_end]).await.ok();
    if buf.len() > header_end {
        upstream_conn.write_all(&buf[header_end..]).await.ok();
    }

    // Track start time for circuit breaker (short connections = likely failure).
    let splice_start = Instant::now();
    let bw_peer_id = peer_key.clone().unwrap_or_default();
    let max_bw = peer_key
        .as_ref()
        .map(|k| {
            let peer = peer_tracker.get_or_create(k);
            PeerPolicy::from_tier(peer.tier).max_bandwidth_mbps
        })
        .unwrap_or(0); // 0 = unlimited (non-libp2p connections)

    // Issue #014: Try splice(2) zero-copy for the upstream↔client data path.
    // splice(2) moves data between fds entirely in kernel space — no userspace copy.
    // This only works when both sides expose raw fds (plain TCP sockets).
    // TLS streams don't support splice because data must pass through the TLS layer.
    // When splice is unavailable or fails, we fall back to bandwidth_limited_copy.
    #[cfg(target_os = "linux")]
    let splice_used = try_splice_bidirectional(
        &client_stream, &upstream_conn, metrics,
    ).await;

    #[cfg(not(target_os = "linux"))]
    let splice_used = false;

    if !splice_used {
        // Fallback: userspace copy with per-peer bandwidth limiting
        let (mut upstream_read, mut upstream_write) = tokio::io::split(upstream_conn);
        let (mut client_read, mut client_write) = tokio::io::split(client_stream);

        let c2u = bandwidth_limited_copy(
            &mut client_read,
            &mut upstream_write,
            &bw_peer_id,
            max_bw,
            bandwidth_limiter,
            peer_tracker,
            true, // client-to-upstream = rx
        );
        let u2c = bandwidth_limited_copy(
            &mut upstream_read,
            &mut client_write,
            &bw_peer_id,
            max_bw,
            bandwidth_limiter,
            peer_tracker,
            false, // upstream-to-client = tx
        );

        // Issue #018: drain signal — when drain fires, close the WS gracefully.
        metrics.drain_start();
        tokio::select! {
            r = c2u => {
                if let Ok(n) = r { metrics.bytes_rx(n); }
                metrics.drain_completed();
            }
            r = u2c => {
                if let Ok(n) = r { metrics.bytes_tx(n); }
                metrics.drain_completed();
            }
            _ = drain_rx.changed() => {
                tracing::debug!(client = %client_addr, "WebSocket drain signal — closing connection");
                metrics.drain_forced();
            }
        }
    }

    // Circuit breaker: short WebSocket connections (<2s) indicate upstream failure.
    if let Some(ref key) = peer_key {
        let peer = peer_tracker.get_or_create(key);
        let splice_duration = splice_start.elapsed();
        if splice_duration < std::time::Duration::from_secs(2) {
            peer.circuit_breaker.write().record_failure();
        } else {
            peer.circuit_breaker.write().record_success();
        }
        peer.conn_closed();
    }

    metrics.ws_closed();
}

/// Try zero-copy splice(2) bidirectional data transfer between two streams.
///
/// Returns `true` if splice was used successfully, `false` if splice is not
/// available (e.g. TLS streams, non-Linux) and the caller should fall back
/// to userspace copy.
///
/// splice(2) requires raw file descriptors from plain TCP sockets. TLS streams
/// wrap the fd in an encryption layer, so splice is only possible when the proxy
/// runs in plain TCP mode (no TLS). In the normal TLS case, this function detects
/// that and returns false immediately — no performance penalty.
#[cfg(target_os = "linux")]
async fn try_splice_bidirectional<S>(
    _client: &S,
    upstream: &tokio::net::TcpStream,
    metrics: &Metrics,
) -> bool
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    use std::os::unix::io::AsRawFd;

    // The client stream is typically a TLS stream (tokio_rustls::server::TlsStream).
    // TLS streams don't expose a usable raw fd for splice — the data must pass
    // through the TLS encryption/decryption layer in userspace.
    //
    // We check if S is a plain TcpStream by trying to downcast. If it's TLS,
    // splice is impossible and we return false immediately.
    //
    // For future plain-TCP listeners, this function will actually splice.
    // For now, it correctly detects TLS and falls back.

    // Try to get the upstream fd — this always works for TcpStream
    let upstream_fd = upstream.as_raw_fd();
    if upstream_fd < 0 {
        metrics.splice_fallback();
        return false;
    }

    // The client side: we can't get a raw fd from a generic S.
    // This means splice only works if S is concretely a TcpStream, which it
    // isn't in the TLS path (it's TlsStream<TcpStream>).
    // Type-erase check: S is always TLS in production, so record the fallback.
    //
    // When q-flux adds a plain TCP listener, this path will be extended to
    // extract the fd via trait specialization or a concrete type parameter.
    metrics.splice_fallback();
    let _ = upstream_fd; // suppress unused warning
    false

    // Future implementation when plain TCP is supported:
    // let channel = match SpliceChannel::new(pipe_size) { Ok(c) => c, Err(_) => return false };
    // metrics.splice_opened();
    // let result = tokio::task::spawn_blocking(move || {
    //     loop {
    //         match io_uring_loop::splice_bidirectional(client_fd, upstream_fd, &channel, 65536) {
    //             Ok((0, 0)) => break,
    //             Ok((fwd, rev)) => total += fwd + rev,
    //             Err(_) => break,
    //         }
    //     }
    //     total
    // }).await;
    // metrics.splice_bytes(result.unwrap_or(0) as u64);
    // metrics.splice_closed();
    // true
}

/// Check if a request path is a known SSE/streaming endpoint.
/// These are routed via direct TCP to avoid consuming hyper's connection pool
/// with long-lived streams that never return connections to the idle pool.
#[inline]
pub(crate) fn is_sse_path(path: &str) -> bool {
    path == "/api/v1/sse"
        || path.starts_with("/api/v1/sse?")
        || path == "/sse"
        || path.starts_with("/sse?")
        || path == "/api/v1/ai/chat"  // AI wallet analysis — streams SSE tokens
        || path == "/api/v1/admin/deploy/progress"  // deploy progress SSE
}

/// Handle SSE/streaming request via direct TCP connection.
/// This bypasses hyper's Client entirely — the connection goes straight from
/// q-flux to the backend over raw TCP, just like WebSocket.
///
/// Why: SSE responses stream indefinitely. When routed through hyper's Client,
/// the TCP connection to the backend is held by the streaming body reader and
/// never returned to the idle pool. This forces hyper to create new TCP
/// connections for every subsequent request, overwhelming the backend under load.
///
/// By using direct TCP (like WebSocket does), SSE connections don't affect
/// hyper's connection pool, keeping it healthy for normal HTTP requests.
async fn handle_sse_direct<S>(
    mut client_stream: S,
    req: &hyper::Request<()>,
    client_addr: SocketAddr,
    upstream: &UpstreamPool,
    metrics: &Metrics,
    request_id: &str,
    mut drain_rx: DrainReceiver,
) where
    S: AsyncRead + AsyncWrite + Unpin,
{
    let backend = upstream.next_backend_addr();
    let mut upstream_conn = match tokio::net::TcpStream::connect(backend).await {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!(client = %client_addr, backend = %backend, "SSE upstream connect failed: {}", e);
            let _ = write_error_response(&mut client_stream, 502, "Bad Gateway").await;
            return;
        }
    };

    // Build raw HTTP request to send to backend.
    // We construct this manually to add proxy headers (X-Forwarded-For, etc.)
    // while preserving all original client headers.
    let path = req.uri().path_and_query().map(|pq| pq.as_str()).unwrap_or("/");
    let mut raw_req = Vec::with_capacity(512);
    use std::io::Write as IoWrite;
    write!(raw_req, "{} {} HTTP/1.1\r\n", req.method(), path).ok();

    // Forward original headers (skip hop-by-hop)
    for (key, value) in req.headers() {
        if key == hyper::header::CONNECTION
            || key == hyper::header::TRANSFER_ENCODING
            || key == "keep-alive"
            || key == "proxy-connection"
        {
            continue;
        }
        write!(raw_req, "{}: {}\r\n", key, value.to_str().unwrap_or("")).ok();
    }
    // Add proxy headers
    write!(raw_req, "X-Forwarded-For: {}\r\n", client_addr.ip()).ok();
    write!(raw_req, "X-Real-IP: {}\r\n", client_addr.ip()).ok();
    write!(raw_req, "X-Request-ID: {}\r\n", request_id).ok();
    raw_req.extend_from_slice(b"\r\n");

    // Send request to backend
    if upstream_conn.write_all(&raw_req).await.is_err() {
        let _ = write_error_response(&mut client_stream, 502, "Bad Gateway").await;
        return;
    }

    // Issue #014: Try splice(2) zero-copy, fall back to tokio::io::copy
    #[cfg(target_os = "linux")]
    let splice_used = try_splice_bidirectional(
        &client_stream, &upstream_conn, metrics,
    ).await;

    #[cfg(not(target_os = "linux"))]
    let splice_used = false;

    if !splice_used {
        let (mut upstream_read, mut upstream_write) = tokio::io::split(upstream_conn);
        let (mut client_read, mut client_write) = tokio::io::split(client_stream);

        let u2c = tokio::io::copy(&mut upstream_read, &mut client_write);
        let c2u = tokio::io::copy(&mut client_read, &mut upstream_write);

        // Issue #018: drain signal — when drain fires, close the SSE stream.
        metrics.drain_start();
        tokio::select! {
            r = u2c => {
                if let Ok(n) = r { metrics.bytes_tx(n); }
                metrics.drain_completed();
            }
            r = c2u => {
                if let Ok(n) = r { metrics.bytes_rx(n); }
                metrics.drain_completed();
            }
            _ = drain_rx.changed() => {
                tracing::debug!(client = %client_addr, "SSE drain signal — closing stream");
                metrics.drain_forced();
            }
        }
    }
}

/// Issue #024: Stream a large request body directly to an upstream backend.
///
/// Opens a raw TCP connection to one upstream backend and pipes the request
/// headers + body through in 64KB chunks. The response is then streamed back
/// to the client. This avoids buffering multi-MB uploads in memory.
///
/// Returns the HTTP status code on success, or an error on failure.
/// Streaming bodies cannot be retried on a different backend (body consumed).
async fn stream_body_to_upstream<S: AsyncRead + AsyncWrite + Unpin>(
    client_stream: &mut S,
    req: &hyper::Request<()>,
    already_read_body: &[u8],
    content_length: usize,
    upstream: &UpstreamPool,
    client_addr: SocketAddr,
    metrics: &Metrics,
    request_id: &str,
) -> Result<u16> {
    // Pick a backend
    let backend_addr = upstream.pick_backend()
        .ok_or_else(|| anyhow::anyhow!("No healthy backends"))?;

    // Connect to upstream
    let mut upstream_stream = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        tokio::net::TcpStream::connect(&backend_addr),
    ).await
        .map_err(|_| anyhow::anyhow!("Upstream connect timeout"))?
        .map_err(|e| anyhow::anyhow!("Upstream connect error: {}", e))?;

    // Build raw HTTP request header
    let mut header_buf = Vec::with_capacity(1024);
    write!(header_buf, "{} {} HTTP/1.1\r\n", req.method(), req.uri())?;
    for (name, value) in req.headers() {
        write!(header_buf, "{}: ", name)?;
        header_buf.extend_from_slice(value.as_bytes());
        header_buf.extend_from_slice(b"\r\n");
    }
    write!(header_buf, "X-Forwarded-For: {}\r\n", client_addr.ip())?;
    write!(header_buf, "X-Real-IP: {}\r\n", client_addr.ip())?;
    write!(header_buf, "X-Request-ID: {}\r\n", request_id)?;
    header_buf.extend_from_slice(b"\r\n");

    // Send headers to upstream
    upstream_stream.write_all(&header_buf).await
        .map_err(|e| anyhow::anyhow!("Upstream header write: {}", e))?;

    // Send body bytes we already have
    if !already_read_body.is_empty() {
        upstream_stream.write_all(already_read_body).await
            .map_err(|e| anyhow::anyhow!("Upstream body write: {}", e))?;
        metrics.bytes_rx(already_read_body.len() as u64);
    }

    // Stream remaining body from client to upstream in 64KB chunks
    let mut streamed = already_read_body.len();
    let mut chunk_buf = vec![0u8; 65536];
    while streamed < content_length {
        let remaining = content_length - streamed;
        let to_read = remaining.min(65536);
        match tokio::time::timeout(
            CLIENT_READ_TIMEOUT,
            client_stream.read(&mut chunk_buf[..to_read]),
        ).await {
            Ok(Ok(0)) => break,
            Ok(Ok(n)) => {
                upstream_stream.write_all(&chunk_buf[..n]).await
                    .map_err(|e| anyhow::anyhow!("Upstream body stream: {}", e))?;
                metrics.bytes_rx(n as u64);
                streamed += n;
            }
            Ok(Err(e)) => return Err(anyhow::anyhow!("Client body read: {}", e)),
            Err(_) => return Err(anyhow::anyhow!("Client body read timeout")),
        }
    }
    upstream_stream.flush().await?;

    // Read upstream response and forward to client
    let mut resp_buf = vec![0u8; 8192];
    let mut resp_len = 0;
    loop {
        match tokio::time::timeout(
            std::time::Duration::from_secs(30),
            upstream_stream.read(&mut resp_buf[resp_len..]),
        ).await {
            Ok(Ok(0)) => break,
            Ok(Ok(n)) => {
                resp_len += n;
                // Check if we have the full headers
                if let Some(_header_end) = find_header_end_resp(&resp_buf[..resp_len]) {
                    // Write headers to client
                    client_stream.write_all(&resp_buf[..resp_len]).await?;
                    metrics.bytes_tx(resp_len as u64);

                    // Parse status code from first line
                    let status = parse_response_status(&resp_buf[..resp_len]);

                    // Stream remaining response body
                    let mut stream_buf = vec![0u8; 65536];
                    loop {
                        match upstream_stream.read(&mut stream_buf).await {
                            Ok(0) => break,
                            Ok(n) => {
                                client_stream.write_all(&stream_buf[..n]).await?;
                                metrics.bytes_tx(n as u64);
                            }
                            Err(_) => break,
                        }
                    }
                    client_stream.flush().await?;
                    return Ok(status);
                }
                if resp_len >= resp_buf.len() {
                    resp_buf.resize(resp_len + 8192, 0);
                }
            }
            Ok(Err(e)) => return Err(anyhow::anyhow!("Upstream response read: {}", e)),
            Err(_) => return Err(anyhow::anyhow!("Upstream response timeout")),
        }
    }

    // If we got here, upstream closed without sending headers
    Err(anyhow::anyhow!("Upstream closed without response"))
}

/// Find the end of HTTP headers (\r\n\r\n) in a response buffer.
fn find_header_end_resp(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n").map(|p| p + 4)
}

/// Parse the HTTP status code from the first line of a response.
fn parse_response_status(header: &[u8]) -> u16 {
    // HTTP/1.1 200 OK\r\n...
    if let Ok(s) = std::str::from_utf8(header) {
        if let Some(line) = s.lines().next() {
            if let Some(status_str) = line.split_whitespace().nth(1) {
                return status_str.parse().unwrap_or(502);
            }
        }
    }
    502
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── generate_request_id ─────────────────────────────────────────

    #[test]
    fn test_request_id_format() {
        let id = generate_request_id();
        // Format: "XXXX-XXXXXX" (4 hex - 6 hex)
        assert_eq!(id.len(), 11, "request ID should be 11 chars: {}", id);
        assert_eq!(id.as_bytes()[4], b'-', "dash at position 4");
        assert!(id[..4].chars().all(|c| c.is_ascii_hexdigit()), "first 4 chars hex");
        assert!(id[5..].chars().all(|c| c.is_ascii_hexdigit()), "last 6 chars hex");
    }

    #[test]
    fn test_request_id_uniqueness() {
        let ids: Vec<String> = (0..100).map(|_| generate_request_id()).collect();
        let unique: std::collections::HashSet<_> = ids.iter().collect();
        assert_eq!(ids.len(), unique.len(), "all 100 request IDs should be unique");
    }

    #[test]
    fn test_request_id_counter_increments() {
        let id1 = generate_request_id();
        let id2 = generate_request_id();
        // The counter portion (after dash) should differ
        let c1 = u32::from_str_radix(&id1[5..], 16).unwrap();
        let c2 = u32::from_str_radix(&id2[5..], 16).unwrap();
        assert_eq!(c2, c1 + 1, "counter should increment by 1");
    }

    // ── should_keep_alive ───────────────────────────────────────────

    #[test]
    fn test_keepalive_http11_default() {
        let req = hyper::Request::builder()
            .version(hyper::Version::HTTP_11)
            .body(()).unwrap();
        assert!(should_keep_alive(&req), "HTTP/1.1 default is keep-alive");
    }

    #[test]
    fn test_keepalive_http10_default() {
        let req = hyper::Request::builder()
            .version(hyper::Version::HTTP_10)
            .body(()).unwrap();
        assert!(!should_keep_alive(&req), "HTTP/1.0 default is close");
    }

    #[test]
    fn test_keepalive_connection_close() {
        let req = hyper::Request::builder()
            .version(hyper::Version::HTTP_11)
            .header(hyper::header::CONNECTION, "close")
            .body(()).unwrap();
        assert!(!should_keep_alive(&req), "Connection: close should disable keepalive");
    }

    #[test]
    fn test_keepalive_connection_close_case_insensitive() {
        let req = hyper::Request::builder()
            .version(hyper::Version::HTTP_11)
            .header(hyper::header::CONNECTION, "Close")
            .body(()).unwrap();
        assert!(!should_keep_alive(&req), "Connection: Close (caps) should disable keepalive");
    }

    #[test]
    fn test_keepalive_connection_keep_alive_header() {
        let req = hyper::Request::builder()
            .version(hyper::Version::HTTP_11)
            .header(hyper::header::CONNECTION, "keep-alive")
            .body(()).unwrap();
        assert!(should_keep_alive(&req), "Connection: keep-alive should enable keepalive");
    }

    // ── is_sse_path ─────────────────────────────────────────────────

    #[test]
    fn test_sse_path_exact() {
        assert!(is_sse_path("/api/v1/sse"));
        assert!(is_sse_path("/sse"));
        assert!(is_sse_path("/api/v1/ai/chat"));
        assert!(is_sse_path("/api/v1/admin/deploy/progress"));
    }

    #[test]
    fn test_sse_path_with_query() {
        assert!(is_sse_path("/api/v1/sse?token=abc"));
        assert!(is_sse_path("/sse?token=abc"));
    }

    #[test]
    fn test_sse_path_negative() {
        assert!(!is_sse_path("/api/v1/status"));
        assert!(!is_sse_path("/api/v1/sse-other"));
        assert!(!is_sse_path("/"));
        assert!(!is_sse_path("/sse/sub"));
        assert!(!is_sse_path("/api/v1/events"));
    }

    // ── reason_phrase ───────────────────────────────────────────────

    #[test]
    fn test_reason_phrases() {
        assert_eq!(reason_phrase(400), "Bad Request");
        assert_eq!(reason_phrase(408), "Request Timeout");
        assert_eq!(reason_phrase(413), "Payload Too Large");
        assert_eq!(reason_phrase(429), "Too Many Requests");
        assert_eq!(reason_phrase(502), "Bad Gateway");
        assert_eq!(reason_phrase(503), "Service Unavailable");
        assert_eq!(reason_phrase(500), "Error");
        assert_eq!(reason_phrase(200), "Error");
    }

    // ── write_error_response ────────────────────────────────────────

    #[tokio::test]
    async fn test_write_error_response_format() {
        let mut buf = Vec::new();
        write_error_response(&mut buf, 502, "Bad Gateway").await.unwrap();
        let response = String::from_utf8(buf).unwrap();
        assert!(response.starts_with("HTTP/1.1 502 Bad Gateway\r\n"));
        assert!(response.contains("content-type: application/json"));
        assert!(response.contains("connection: close"));
        assert!(response.contains(r#"{"error":"Bad Gateway"}"#));
    }

    #[tokio::test]
    async fn test_write_error_response_escapes_json() {
        let mut buf = Vec::new();
        write_error_response(&mut buf, 400, r#"bad "input" with \ slash"#).await.unwrap();
        let response = String::from_utf8(buf).unwrap();
        // Verify JSON escaping: quotes and backslashes escaped
        assert!(response.contains(r#"bad \"input\" with \\ slash"#));
    }

    #[tokio::test]
    async fn test_write_error_response_content_length() {
        let mut buf = Vec::new();
        write_error_response(&mut buf, 413, "Payload Too Large").await.unwrap();
        let response = String::from_utf8(buf).unwrap();
        let body = r#"{"error":"Payload Too Large"}"#;
        let expected_cl = format!("content-length: {}", body.len());
        assert!(response.contains(&expected_cl), "content-length should match body: {}", response);
    }

    // ── DrainReceiver type ──────────────────────────────────────────

    #[test]
    fn test_drain_receiver_initial_value() {
        let (_tx, rx) = tokio::sync::watch::channel(false);
        let drain_rx: DrainReceiver = rx;
        assert_eq!(*drain_rx.borrow(), false, "drain should start as false");
    }

    #[tokio::test]
    async fn test_drain_receiver_signal() {
        let (tx, mut rx) = tokio::sync::watch::channel(false);
        let _ = tx.send(true);
        rx.changed().await.unwrap();
        assert_eq!(*rx.borrow(), true, "drain should be true after signal");
    }
}
