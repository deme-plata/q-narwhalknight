//! Static file serving for q-flux.
//!
//! Serves files from a document root with:
//! - MIME type detection via file extension
//! - Cache-Control headers (long for hashed assets, no-cache for HTML)
//! - ETag / If-None-Match (304 Not Modified)
//! - SPA fallback: non-file, non-API paths serve index.html
//! - CORS headers for API paths
//! - `/downloads/` directory listing disabled, files served with Content-Disposition

use std::path::{Path, PathBuf};
use tokio::io::{AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader};

use crate::config::StaticConfig;
use crate::metrics::Metrics;

/// Size of the buffer used for streaming file content.
/// 64 KiB balances syscall overhead vs memory footprint per connection.
const STREAM_BUF_SIZE: usize = 64 * 1024;

/// Result of routing a request path.
pub enum RouteResult {
    /// Serve this file from disk.
    ServeFile(FileResponse),
    /// Forward to upstream (API, WebSocket, etc.).
    Proxy,
}

pub struct FileResponse {
    pub path: PathBuf,
    pub mime: &'static str,
    pub cache_control: &'static str,
    pub is_download: bool,
}

/// CORS headers applied to API responses.
pub const CORS_HEADERS: &str = "\
access-control-allow-origin: *\r\n\
access-control-allow-methods: GET, POST, PUT, DELETE, OPTIONS\r\n\
access-control-allow-headers: Content-Type, Authorization, X-Wallet-Address, X-Wallet-Signature, X-Wallet-Auth\r\n";

/// Decide whether a request path should be served as a static file or proxied.
pub fn route(path: &str, config: &StaticConfig) -> RouteResult {
    let root = match config.root.as_ref() {
        Some(r) => r,
        None => return RouteResult::Proxy, // No static config → proxy everything
    };

    // Normalise: strip query string and fragments
    let clean = path.split('?').next().unwrap_or(path);
    let clean = clean.split('#').next().unwrap_or(clean);

    // API, WebSocket, health → always proxy
    if clean.starts_with("/api/")
        || clean == "/api"
        || clean.starts_with("/ws")
        || clean == "/health"
        || clean.starts_with("/aioc/")
    {
        return RouteResult::Proxy;
    }

    // Downloads directory
    if clean.starts_with("/downloads/") {
        let rel = clean.strip_prefix('/').unwrap_or(clean);
        let candidate = root.join(rel);
        if candidate.is_file() && is_safe_path(&candidate, root) {
            return RouteResult::ServeFile(FileResponse {
                mime: mime_for_path(&candidate),
                cache_control: "no-cache",
                is_download: true,
                path: candidate,
            });
        }
    }

    // Static assets: match common extensions
    if has_static_extension(clean) {
        let rel = clean.strip_prefix('/').unwrap_or(clean);
        let candidate = root.join(rel);
        if candidate.is_file() && is_safe_path(&candidate, root) {
            // Hashed assets (e.g., index-CgzZl2jy.js) get long cache
            let cache = if is_hashed_asset(clean) {
                "public, max-age=31536000, immutable"
            } else {
                "public, max-age=86400"
            };
            return RouteResult::ServeFile(FileResponse {
                mime: mime_for_path(&candidate),
                cache_control: cache,
                is_download: false,
                path: candidate,
            });
        }
    }

    // Exact file match (e.g., /favicon.ico, /robots.txt)
    {
        let rel = clean.strip_prefix('/').unwrap_or(clean);
        if !rel.is_empty() && !rel.contains("..") {
            let candidate = root.join(rel);
            if candidate.is_file() && is_safe_path(&candidate, root) {
                return RouteResult::ServeFile(FileResponse {
                    mime: mime_for_path(&candidate),
                    cache_control: "public, max-age=3600",
                    is_download: false,
                    path: candidate,
                });
            }
        }
    }

    // SPA fallback: serve index.html for everything else
    if config.spa_fallback {
        let index = root.join("index.html");
        if index.is_file() {
            return RouteResult::ServeFile(FileResponse {
                mime: "text/html; charset=utf-8",
                cache_control: "no-cache",
                is_download: false,
                path: index,
            });
        }
    }

    // No static match → proxy to upstream
    RouteResult::Proxy
}

/// Serve a file response over a raw TCP/TLS stream.
pub async fn serve_file<S: AsyncWrite + Unpin>(
    stream: &mut S,
    resp: &FileResponse,
    method: &str,
    if_none_match: Option<&str>,
    metrics: &Metrics,
) -> std::io::Result<()> {
    let metadata = match tokio::fs::metadata(&resp.path).await {
        Ok(m) => m,
        Err(_) => {
            return write_raw(stream, 404, "text/plain", b"Not Found").await;
        }
    };

    let size = metadata.len();

    // Simple ETag: file size + modified time
    let etag = if let Ok(modified) = metadata.modified() {
        let dur = modified.duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
        format!("\"{:x}-{:x}\"", size, dur.as_secs())
    } else {
        format!("\"{:x}\"", size)
    };

    // 304 Not Modified?
    if let Some(inm) = if_none_match {
        if inm.trim() == etag || inm.contains(&etag) {
            let headers = format!(
                "HTTP/1.1 304 Not Modified\r\n\
                 etag: {}\r\n\
                 cache-control: {}\r\n\
                 \r\n",
                etag, resp.cache_control,
            );
            stream.write_all(headers.as_bytes()).await?;
            stream.flush().await?;
            return Ok(());
        }
    }

    // HEAD → headers only
    if method.eq_ignore_ascii_case("HEAD") {
        let headers = format!(
            "HTTP/1.1 200 OK\r\n\
             content-type: {}\r\n\
             content-length: {}\r\n\
             cache-control: {}\r\n\
             etag: {}\r\n\
             {}\
             \r\n",
            resp.mime,
            size,
            resp.cache_control,
            etag,
            if resp.is_download {
                format!("content-disposition: attachment; filename=\"{}\"\r\n",
                    resp.path.file_name().and_then(|n| n.to_str()).unwrap_or("download"))
            } else {
                String::new()
            },
        );
        stream.write_all(headers.as_bytes()).await?;
        stream.flush().await?;
        return Ok(());
    }

    // Build Content-Disposition header for downloads
    let disposition = if resp.is_download {
        format!(
            "content-disposition: attachment; filename=\"{}\"\r\n",
            resp.path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("download"),
        )
    } else {
        String::new()
    };

    // Open the file for streaming — never buffer the full body in memory.
    // Binary downloads (q-api-server, q-miner) are 50-100 MB each;
    // 10 concurrent downloads would spike 1 GB with tokio::fs::read().
    let file = match tokio::fs::File::open(&resp.path).await {
        Ok(f) => f,
        Err(e) => {
            tracing::warn!(path = %resp.path.display(), "File open error: {}", e);
            return write_raw(stream, 500, "text/plain", b"Internal Server Error").await;
        }
    };

    // Write HTTP response headers first (Content-Length from metadata).
    let headers = format!(
        "HTTP/1.1 200 OK\r\n\
         content-type: {}\r\n\
         content-length: {}\r\n\
         cache-control: {}\r\n\
         etag: {}\r\n\
         {}\
         \r\n",
        resp.mime, size, resp.cache_control, etag, disposition,
    );
    stream.write_all(headers.as_bytes()).await?;

    // Stream file body in 64 KiB chunks via BufReader.
    // Memory per connection is capped at ~64 KiB regardless of file size.
    let mut reader = BufReader::with_capacity(STREAM_BUF_SIZE, file);
    let mut bytes_sent: u64 = 0;
    let mut buf = [0u8; STREAM_BUF_SIZE];
    loop {
        let n = reader.read(&mut buf).await?;
        if n == 0 {
            break;
        }
        stream.write_all(&buf[..n]).await?;
        bytes_sent += n as u64;
    }
    stream.flush().await?;
    metrics.bytes_tx(bytes_sent);
    Ok(())
}

/// Write a simple HTTP error response.
async fn write_raw<S: AsyncWrite + Unpin>(
    stream: &mut S,
    status: u16,
    content_type: &str,
    body: &[u8],
) -> std::io::Result<()> {
    let reason = match status {
        404 => "Not Found",
        500 => "Internal Server Error",
        _ => "Error",
    };
    let header = format!(
        "HTTP/1.1 {} {}\r\ncontent-type: {}\r\ncontent-length: {}\r\n\r\n",
        status, reason, content_type, body.len()
    );
    stream.write_all(header.as_bytes()).await?;
    stream.write_all(body).await?;
    stream.flush().await
}

/// Prevent path traversal — resolved path must be under root.
fn is_safe_path(candidate: &Path, root: &Path) -> bool {
    match (candidate.canonicalize(), root.canonicalize()) {
        (Ok(c), Ok(r)) => c.starts_with(r),
        _ => false,
    }
}

/// Check if the path has a static asset extension.
fn has_static_extension(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    matches!(
        lower.rsplit('.').next(),
        Some("js" | "css" | "png" | "jpg" | "jpeg" | "gif" | "svg" | "ico"
            | "woff" | "woff2" | "ttf" | "eot" | "map" | "json" | "wasm"
            | "webp" | "avif" | "mp4" | "webm" | "pdf" | "txt" | "xml"
            | "html" | "htm")
    )
}

/// Detect hashed asset names (e.g., index-CgzZl2jy.js) for immutable caching.
fn is_hashed_asset(path: &str) -> bool {
    if let Some(stem) = path.rsplit('/').next() {
        if let Some(base) = stem.rsplit('.').nth(1) {
            if let Some(hash_part) = base.rsplit('-').next() {
                return hash_part.len() >= 8
                    && hash_part.chars().all(|c| c.is_ascii_alphanumeric());
            }
        }
    }
    false
}

/// Get MIME type from file extension.
fn mime_for_path(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).as_deref() {
        Some("html" | "htm") => "text/html; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("js" | "mjs") => "application/javascript; charset=utf-8",
        Some("json") => "application/json; charset=utf-8",
        Some("wasm") => "application/wasm",
        Some("png") => "image/png",
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("gif") => "image/gif",
        Some("svg") => "image/svg+xml",
        Some("ico") => "image/x-icon",
        Some("webp") => "image/webp",
        Some("avif") => "image/avif",
        Some("woff") => "font/woff",
        Some("woff2") => "font/woff2",
        Some("ttf") => "font/ttf",
        Some("eot") => "application/vnd.ms-fontobject",
        Some("map") => "application/json",
        Some("xml") => "application/xml; charset=utf-8",
        Some("txt") => "text/plain; charset=utf-8",
        Some("pdf") => "application/pdf",
        Some("mp4") => "video/mp4",
        Some("webm") => "video/webm",
        _ => "application/octet-stream",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mime_types() {
        assert_eq!(mime_for_path(Path::new("app.js")), "application/javascript; charset=utf-8");
        assert_eq!(mime_for_path(Path::new("style.css")), "text/css; charset=utf-8");
        assert_eq!(mime_for_path(Path::new("img.png")), "image/png");
        assert_eq!(mime_for_path(Path::new("font.woff2")), "font/woff2");
        assert_eq!(mime_for_path(Path::new("data.wasm")), "application/wasm");
        assert_eq!(mime_for_path(Path::new("unknown")), "application/octet-stream");
    }

    #[test]
    fn test_hashed_asset_detection() {
        assert!(is_hashed_asset("/assets/index-CgzZl2jy.js"));
        assert!(is_hashed_asset("/assets/vendor-AbCdEfGh.css"));
        assert!(!is_hashed_asset("/index.html"));
        assert!(!is_hashed_asset("/assets/app.js"));
        assert!(!is_hashed_asset("/assets/short-Ab.js")); // hash too short
    }

    #[test]
    fn test_static_extension_detection() {
        assert!(has_static_extension("/app.js"));
        assert!(has_static_extension("/style.CSS"));
        assert!(has_static_extension("/image.webp"));
        assert!(!has_static_extension("/api/v1/status"));
        assert!(!has_static_extension("/no-extension"));
    }

    #[test]
    fn test_api_paths_always_proxy() {
        let config = StaticConfig {
            root: Some(PathBuf::from("/tmp")),
            spa_fallback: true,
        };
        assert!(matches!(route("/api/v1/mining/challenge", &config), RouteResult::Proxy));
        assert!(matches!(route("/api/v1/events", &config), RouteResult::Proxy));
        assert!(matches!(route("/ws", &config), RouteResult::Proxy));
        assert!(matches!(route("/health", &config), RouteResult::Proxy));
        assert!(matches!(route("/aioc/test", &config), RouteResult::Proxy));
    }

    #[test]
    fn test_no_static_config_proxies_everything() {
        let config = StaticConfig { root: None, spa_fallback: false };
        assert!(matches!(route("/", &config), RouteResult::Proxy));
        assert!(matches!(route("/app.js", &config), RouteResult::Proxy));
    }
}
