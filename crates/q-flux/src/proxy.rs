use anyhow::Result;
use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use std::net::SocketAddr;
use tokio::io::{AsyncRead, AsyncWrite, AsyncReadExt, AsyncWriteExt};

use crate::metrics::Metrics;
use crate::upstream::UpstreamPool;

const MAX_HEADER_SIZE: usize = 8192;

/// Handle a single HTTP connection (potentially multiple requests via keepalive).
pub async fn handle_connection<S>(
    mut stream: S,
    client_addr: SocketAddr,
    upstream: &UpstreamPool,
    metrics: &Metrics,
    body_limit: usize,
) where
    S: AsyncRead + AsyncWrite + Unpin,
{
    let mut buf = vec![0u8; MAX_HEADER_SIZE];
    let mut buf_len = 0usize;

    // HTTP/1.1 keepalive loop — handle multiple requests per connection
    loop {
        // Read headers
        let (req, header_end) = match read_request_headers(&mut stream, &mut buf, &mut buf_len).await {
            Ok(Some(v)) => v,
            Ok(None) => break, // Clean close
            Err(e) => {
                tracing::debug!(client = %client_addr, "Header read error: {}", e);
                break;
            }
        };

        metrics.request();

        // Check for WebSocket upgrade
        let is_upgrade = req.headers().get(hyper::header::UPGRADE)
            .and_then(|v| v.to_str().ok())
            .map(|v| v.eq_ignore_ascii_case("websocket"))
            .unwrap_or(false);

        if is_upgrade {
            // WebSocket: forward the upgrade request, then splice
            handle_websocket_upgrade(stream, req, header_end, &buf[..buf_len], client_addr, upstream, metrics).await;
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

            // Copy any body bytes already in our header buffer
            let to_copy = already_read.min(content_length);
            body_buf.extend_from_slice(&buf[header_end..header_end + to_copy]);

            // Read remaining body
            while body_buf.len() < content_length {
                let remaining = content_length - body_buf.len();
                let mut chunk = vec![0u8; remaining.min(65536)];
                match stream.read(&mut chunk).await {
                    Ok(0) => break,
                    Ok(n) => {
                        metrics.bytes_rx(n as u64);
                        body_buf.extend_from_slice(&chunk[..n]);
                    }
                    Err(e) => {
                        tracing::debug!(client = %client_addr, "Body read error: {}", e);
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
            // Add X-Forwarded-For
            builder = builder.header("X-Forwarded-For", client_addr.ip().to_string());
            builder = builder.header("X-Real-IP", client_addr.ip().to_string());
            builder.body(Full::new(body)).unwrap()
        };

        // Forward to upstream
        let keep_alive = should_keep_alive(&req);

        match upstream.forward(upstream_req).await {
            Ok(resp) => {
                let status = resp.status().as_u16();
                metrics.response_status(status);

                // Write response back to client
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

        // Reset buffer for next request
        // Move any leftover bytes (past header+body) to front
        let consumed = header_end + content_length;
        if consumed < buf_len {
            let remaining = buf_len - consumed;
            buf.copy_within(consumed..buf_len, 0);
            buf_len = remaining;
        } else {
            buf_len = 0;
        }
    }
}

/// Read request headers from the stream into the buffer.
/// Returns the parsed request and the byte offset where headers end.
async fn read_request_headers<S>(
    stream: &mut S,
    buf: &mut Vec<u8>,
    buf_len: &mut usize,
) -> Result<Option<(hyper::Request<()>, usize)>>
where
    S: AsyncRead + Unpin,
{
    loop {
        // Try to parse what we have
        let mut headers = [httparse::EMPTY_HEADER; 64];
        let mut parsed_req = httparse::Request::new(&mut headers);

        match parsed_req.parse(&buf[..*buf_len]) {
            Ok(httparse::Status::Complete(header_end)) => {
                // Build hyper Request from parsed headers
                let method = parsed_req.method.unwrap_or("GET");
                let path = parsed_req.path.unwrap_or("/");

                let mut builder = hyper::Request::builder()
                    .method(method)
                    .uri(path);

                for h in parsed_req.headers.iter() {
                    if h.name.is_empty() { break; }
                    builder = builder.header(h.name, h.value);
                }

                // Detect Host header to reconstruct full URI if needed
                let req = builder.body(())
                    .map_err(|e| anyhow::anyhow!("Failed to build request: {}", e))?;

                return Ok(Some((req, header_end)));
            }
            Ok(httparse::Status::Partial) => {
                // Need more data
                if *buf_len >= buf.len() {
                    // Headers too large
                    return Err(anyhow::anyhow!("Request headers exceed {}B limit", buf.len()));
                }
            }
            Err(e) => {
                return Err(anyhow::anyhow!("HTTP parse error: {}", e));
            }
        }

        // Read more data
        let n = stream.read(&mut buf[*buf_len..]).await?;
        if n == 0 {
            if *buf_len == 0 {
                return Ok(None); // Clean EOF
            }
            return Err(anyhow::anyhow!("Connection closed mid-headers"));
        }
        *buf_len += n;
    }
}

fn should_keep_alive(req: &hyper::Request<()>) -> bool {
    // HTTP/1.1 defaults to keepalive; HTTP/1.0 requires explicit
    if let Some(conn) = req.headers().get(hyper::header::CONNECTION) {
        if let Ok(s) = conn.to_str() {
            return !s.eq_ignore_ascii_case("close");
        }
    }
    // Default: keepalive for 1.1, close for 1.0
    req.version() != hyper::Version::HTTP_10
}

/// Write an HTTP response from Incoming body back to the client.
async fn write_response<S>(
    stream: &mut S,
    resp: hyper::Response<Incoming>,
    metrics: &Metrics,
) -> Result<()>
where
    S: AsyncWrite + Unpin,
{
    let (parts, body) = resp.into_parts();

    // Write status line
    let status_line = format!(
        "HTTP/1.1 {} {}\r\n",
        parts.status.as_u16(),
        parts.status.canonical_reason().unwrap_or("OK")
    );
    stream.write_all(status_line.as_bytes()).await?;

    // Collect body to know content-length (for non-streaming responses)
    let body_bytes = body.collect().await
        .map_err(|e| anyhow::anyhow!("Body collect error: {}", e))?
        .to_bytes();

    // Write headers
    let mut wrote_content_length = false;
    for (key, value) in &parts.headers {
        // Skip hop-by-hop
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

    // End headers
    stream.write_all(b"\r\n").await?;

    // Write body
    if !body_bytes.is_empty() {
        stream.write_all(&body_bytes).await?;
        metrics.bytes_tx(body_bytes.len() as u64);
    }

    stream.flush().await?;
    Ok(())
}

async fn write_error_response<S>(stream: &mut S, status: u16, msg: &str) -> Result<()>
where
    S: AsyncWrite + Unpin,
{
    let body = format!("{{\"error\":\"{}\"}}", msg);
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
        413 => "Payload Too Large",
        429 => "Too Many Requests",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        _ => "Error",
    }
}

/// Handle WebSocket upgrade: forward the upgrade request to upstream,
/// then splice both directions.
async fn handle_websocket_upgrade<S>(
    mut client_stream: S,
    _req: hyper::Request<()>,
    header_end: usize,
    buf: &[u8],
    client_addr: SocketAddr,
    upstream: &UpstreamPool,
    metrics: &Metrics,
) where
    S: AsyncRead + AsyncWrite + Unpin,
{
    metrics.ws_upgrade();

    // Connect raw TCP to upstream for WebSocket
    let backend = &upstream.backends[0]; // TODO: round-robin
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
    // Forward any remaining bytes past headers
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
