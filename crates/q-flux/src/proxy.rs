use anyhow::Result;
use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use std::net::SocketAddr;
use tokio::io::{AsyncRead, AsyncWrite, AsyncReadExt, AsyncWriteExt};

use crate::config::StaticConfig;
use crate::metrics::Metrics;
use crate::static_serve;
use crate::upstream::UpstreamPool;

const MAX_HEADER_SIZE: usize = 8192;
/// Client-side read timeout — prevents slowloris attacks.
const CLIENT_READ_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// Handle a single HTTP connection (potentially multiple requests via keepalive).
pub async fn handle_connection<S>(
    mut stream: S,
    client_addr: SocketAddr,
    upstream: &UpstreamPool,
    metrics: &Metrics,
    body_limit: usize,
    static_config: &StaticConfig,
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

        metrics.request();

        // Static file routing: check before proxying
        let req_path = req.uri().path().to_string();
        let req_method = req.method().as_str().to_string();
        if let static_serve::RouteResult::ServeFile(file_resp) = static_serve::route(&req_path, static_config) {
            let if_none_match = req.headers().get("if-none-match")
                .and_then(|v| v.to_str().ok())
                .map(|s| s.to_string());

            if req_method == "OPTIONS" {
                let cors = format!("HTTP/1.1 204 No Content\r\n{}\r\ncontent-length: 0\r\n\r\n", static_serve::CORS_HEADERS);
                let _ = stream.write_all(cors.as_bytes()).await;
                let _ = stream.flush().await;
                metrics.response_status(204);
            } else {
                if let Err(e) = static_serve::serve_file(&mut stream, &file_resp, &req_method, if_none_match.as_deref(), metrics).await {
                    tracing::debug!(client = %client_addr, "Static serve error: {}", e);
                    break;
                }
                metrics.response_status(200);
            }
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
            metrics.response_status(204);
            if !should_keep_alive(&req) { break; }
            buf_len = 0;
            continue;
        }

        // Check for WebSocket upgrade
        let is_upgrade = req.headers().get(hyper::header::UPGRADE)
            .and_then(|v| v.to_str().ok())
            .map(|v| v.eq_ignore_ascii_case("websocket"))
            .unwrap_or(false);

        if is_upgrade {
            handle_websocket_upgrade(stream, header_end, &buf[..buf_len], client_addr, upstream, metrics).await;
            return; // Connection consumed by WebSocket
        }

        // Read body if Content-Length present
        let content_length = req.headers()
            .get(hyper::header::CONTENT_LENGTH)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0);

        if content_length > body_limit {
            let _ = write_error_response(&mut stream, 413, "Request body too large").await;
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
            builder.body(Full::new(body)).unwrap()
        };

        let keep_alive = should_keep_alive(&req);

        match upstream.forward(upstream_req).await {
            Ok(resp) => {
                let status = resp.status().as_u16();
                metrics.response_status(status);

                if let Err(e) = write_response(&mut stream, resp, metrics).await {
                    tracing::debug!(client = %client_addr, "Response write error: {}", e);
                    break;
                }
            }
            Err(e) => {
                tracing::warn!(client = %client_addr, "Upstream error: {}", e);
                metrics.response_status(502);
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
async fn read_request_headers<S>(
    stream: &mut S,
    buf: &mut Vec<u8>,
    buf_len: &mut usize,
) -> Result<Option<(hyper::Request<()>, usize)>>
where
    S: AsyncRead + Unpin,
{
    loop {
        let mut headers = [httparse::EMPTY_HEADER; 64];
        let mut parsed_req = httparse::Request::new(&mut headers);

        match parsed_req.parse(&buf[..*buf_len]) {
            Ok(httparse::Status::Complete(header_end)) => {
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
            Ok(httparse::Status::Partial) => {
                if *buf_len >= buf.len() {
                    return Err(anyhow::anyhow!("Request headers exceed {}B limit", buf.len()));
                }
            }
            Err(e) => {
                return Err(anyhow::anyhow!("HTTP parse error: {}", e));
            }
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

/// Write an HTTP response back to the client.
/// Detects SSE/streaming responses and streams them without buffering.
async fn write_response<S>(
    stream: &mut S,
    resp: hyper::Response<Incoming>,
    metrics: &Metrics,
) -> Result<()>
where
    S: AsyncWrite + Unpin,
{
    let (parts, body) = resp.into_parts();

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
        // BUFFERED MODE: collect full body, set Content-Length, send.
        let body_bytes = body.collect().await
            .map_err(|e| anyhow::anyhow!("Body collect error: {}", e))?
            .to_bytes();

        // Write status line
        let status_line = format!(
            "HTTP/1.1 {} {}\r\n",
            parts.status.as_u16(),
            parts.status.canonical_reason().unwrap_or("OK")
        );
        stream.write_all(status_line.as_bytes()).await?;

        // Write headers
        let mut wrote_content_length = false;
        for (key, value) in &parts.headers {
            if key == hyper::header::TRANSFER_ENCODING || key == "keep-alive" {
                continue;
            }
            if key == hyper::header::CONTENT_LENGTH {
                wrote_content_length = true;
            }
            let header_line = format!("{}: {}\r\n", key, value.to_str().unwrap_or(""));
            stream.write_all(header_line.as_bytes()).await?;
        }

        if !wrote_content_length {
            let cl = format!("content-length: {}\r\n", body_bytes.len());
            stream.write_all(cl.as_bytes()).await?;
        }

        stream.write_all(b"\r\n").await?;

        if !body_bytes.is_empty() {
            stream.write_all(&body_bytes).await?;
            metrics.bytes_tx(body_bytes.len() as u64);
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
    let status_line = format!(
        "HTTP/1.1 {} {}\r\n",
        parts.status.as_u16(),
        parts.status.canonical_reason().unwrap_or("OK")
    );
    stream.write_all(status_line.as_bytes()).await?;

    for (key, value) in &parts.headers {
        // Skip content-length for streaming (we don't know the total size)
        if key == hyper::header::CONTENT_LENGTH {
            continue;
        }
        if key == "keep-alive" {
            continue;
        }
        let header_line = format!("{}: {}\r\n", key, value.to_str().unwrap_or(""));
        stream.write_all(header_line.as_bytes()).await?;
    }

    stream.write_all(b"\r\n").await?;
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
        "HTTP/1.1 {} {}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
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

/// Handle WebSocket upgrade: forward the raw upgrade request to upstream,
/// then bidirectional splice both directions.
async fn handle_websocket_upgrade<S>(
    mut client_stream: S,
    header_end: usize,
    buf: &[u8],
    client_addr: SocketAddr,
    upstream: &UpstreamPool,
    metrics: &Metrics,
) where
    S: AsyncRead + AsyncWrite + Unpin,
{
    metrics.ws_upgrade();

    // Use round-robin backend selection (not hardcoded backend[0])
    let backend = upstream.next_backend_addr();
    let upstream_conn = match tokio::net::TcpStream::connect(backend).await {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!(client = %client_addr, "WS upstream connect failed: {}", e);
            let _ = write_error_response(&mut client_stream, 502, "Bad Gateway").await;
            metrics.ws_closed();
            return;
        }
    };

    let (mut upstream_read, mut upstream_write) = tokio::io::split(upstream_conn);

    // Forward the raw HTTP upgrade request bytes to upstream
    upstream_write.write_all(&buf[..header_end]).await.ok();
    if buf.len() > header_end {
        upstream_write.write_all(&buf[header_end..]).await.ok();
    }

    let (mut client_read, mut client_write) = tokio::io::split(client_stream);

    // Bidirectional splice
    let c2u = tokio::io::copy(&mut client_read, &mut upstream_write);
    let u2c = tokio::io::copy(&mut upstream_read, &mut client_write);

    tokio::select! {
        r = c2u => {
            if let Ok(n) = r { metrics.bytes_rx(n); }
        }
        r = u2c => {
            if let Ok(n) = r { metrics.bytes_tx(n); }
        }
    }

    metrics.ws_closed();
}
