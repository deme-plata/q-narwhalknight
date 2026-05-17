//! Static file serving for q-flux.
//!
//! Serves files from a document root with:
//! - **Gzip compression** for text assets (JS, CSS, HTML, JSON, SVG) — 4-5x smaller
//! - **In-memory file cache** with pre-compressed variants — zero disk I/O for hot assets
//! - MIME type detection via file extension
//! - Cache-Control headers (long for hashed assets, no-cache for HTML)
//! - ETag / If-None-Match (304 Not Modified)
//! - SPA fallback: non-file, non-API paths serve index.html
//! - `/downloads/` directory listing disabled, files served with Content-Disposition

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use flate2::write::GzEncoder;
use flate2::Compression;
use parking_lot::RwLock;
use tokio::io::{AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader};

use crate::config::StaticConfig;
use crate::metrics::Metrics;

/// Size of the buffer used for streaming file content (uncached files).
const STREAM_BUF_SIZE: usize = 64 * 1024;

/// Minimum file size worth compressing (below this, gzip overhead isn't worth it).
const MIN_COMPRESS_SIZE: u64 = 256;

/// Result of routing a request path.
pub enum RouteResult {
    /// Serve this file from disk (or cache).
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

// ── In-memory file cache ────────────────────────────────────────────────

/// Cached file entry: raw bytes + optional pre-compressed gzip variant.
struct CachedFile {
    raw: Vec<u8>,
    gzip: Option<Vec<u8>>,
    etag: String,
    mime: &'static str,
    cache_control: &'static str,
    mtime_secs: u64,
}

// Global cache metrics (aggregated across all workers)
static GLOBAL_CACHE_HITS: AtomicU64 = AtomicU64::new(0);
static GLOBAL_CACHE_MISSES: AtomicU64 = AtomicU64::new(0);

pub fn global_cache_hits() -> u64 {
    GLOBAL_CACHE_HITS.load(Ordering::Relaxed)
}

pub fn global_cache_misses() -> u64 {
    GLOBAL_CACHE_MISSES.load(Ordering::Relaxed)
}

/// Thread-safe in-memory file cache. Pre-compresses text assets on insert.
/// Keyed by relative path (e.g., "assets/index-CgzZl2jy.js").
pub struct FileCache {
    entries: RwLock<HashMap<String, Arc<CachedFile>>>,
    total_bytes: AtomicU64,
    max_file_size: usize,
    max_total: usize,
    gzip_enabled: bool,
    root: PathBuf,
    /// Canonicalized root for safe-path checks (computed once at startup).
    canonical_root: PathBuf,
    pub cache_hits: AtomicU64,
    pub cache_misses: AtomicU64,
}

impl FileCache {
    pub fn new(config: &StaticConfig) -> Option<Arc<Self>> {
        let root = config.root.as_ref()?;
        let canonical_root = root.canonicalize().ok()?;
        Some(Arc::new(Self {
            entries: RwLock::new(HashMap::with_capacity(128)),
            total_bytes: AtomicU64::new(0),
            max_file_size: config.cache_max_file_size,
            max_total: config.cache_max_total,
            gzip_enabled: config.gzip,
            root: root.clone(),
            canonical_root,
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
        }))
    }

    /// Look up a cached file. Returns None if not cached (caller falls back to disk).
    fn get(&self, rel_path: &str) -> Option<Arc<CachedFile>> {
        let entries = self.entries.read();
        let entry = entries.get(rel_path)?.clone();

        // Lazy staleness check: compare mtime. If file changed on disk, evict.
        // This is cheap because we only stat when we have a cache hit.
        let abs = self.root.join(rel_path);
        if let Ok(meta) = std::fs::metadata(&abs) {
            let disk_mtime = meta.modified().ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
                .unwrap_or(0);
            if disk_mtime != entry.mtime_secs {
                drop(entries);
                self.evict(rel_path);
                return None;
            }
        }
        Some(entry)
    }

    /// Insert a file into the cache. Pre-compresses with gzip if it's a text type.
    fn insert(&self, rel_path: &str, abs_path: &Path, mime: &'static str, cache_control: &'static str) -> Option<Arc<CachedFile>> {
        let meta = std::fs::metadata(abs_path).ok()?;
        let size = meta.len() as usize;

        // Don't cache files that are too large
        if size > self.max_file_size {
            return None;
        }

        // Don't exceed total cache budget
        let current_total = self.total_bytes.load(Ordering::Relaxed) as usize;
        if current_total + size > self.max_total {
            return None;
        }

        let raw = std::fs::read(abs_path).ok()?;
        let mtime_secs = meta.modified().ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let etag = format!("\"{:x}-{:x}\"", size, mtime_secs);

        // Pre-compress text assets with gzip
        let gzip = if self.gzip_enabled && is_compressible(mime) && size as u64 > MIN_COMPRESS_SIZE {
            let mut encoder = GzEncoder::new(Vec::with_capacity(size / 2), Compression::fast());
            use std::io::Write;
            if encoder.write_all(&raw).is_ok() {
                if let Ok(compressed) = encoder.finish() {
                    // Only use gzip if it's actually smaller
                    if compressed.len() < size {
                        Some(compressed)
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        let entry_size = raw.len() + gzip.as_ref().map(|g| g.len()).unwrap_or(0);
        let entry = Arc::new(CachedFile {
            raw,
            gzip,
            etag,
            mime,
            cache_control,
            mtime_secs,
        });

        self.entries.write().insert(rel_path.to_string(), entry.clone());
        self.total_bytes.fetch_add(entry_size as u64, Ordering::Relaxed);

        Some(entry)
    }

    fn evict(&self, rel_path: &str) {
        if let Some(entry) = self.entries.write().remove(rel_path) {
            let size = entry.raw.len() + entry.gzip.as_ref().map(|g| g.len()).unwrap_or(0);
            self.total_bytes.fetch_sub(size as u64, Ordering::Relaxed);
        }
    }

    /// Safe path check using pre-canonicalized root (no syscall for root).
    fn is_safe(&self, candidate: &Path) -> bool {
        match candidate.canonicalize() {
            Ok(c) => c.starts_with(&self.canonical_root),
            Err(_) => false,
        }
    }

    pub fn stats(&self) -> (usize, u64) {
        let entries = self.entries.read().len();
        let bytes = self.total_bytes.load(Ordering::Relaxed);
        (entries, bytes)
    }

    pub fn hits(&self) -> u64 {
        self.cache_hits.load(Ordering::Relaxed)
    }

    pub fn misses(&self) -> u64 {
        self.cache_misses.load(Ordering::Relaxed)
    }
}

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

    // Per-vhost additional proxy paths (e.g. "/v1" for bounty site)
    for prefix in &config.proxy_paths {
        if clean == prefix.as_str() || clean.starts_with(&format!("{}/", prefix)) {
            return RouteResult::Proxy;
        }
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

/// Serve a file response — uses cache + gzip when available.
pub async fn serve_file<S: AsyncWrite + Unpin>(
    stream: &mut S,
    resp: &FileResponse,
    method: &str,
    if_none_match: Option<&str>,
    accepts_gzip: bool,
    metrics: &Metrics,
    cache: Option<&FileCache>,
) -> std::io::Result<()> {
    // Try cache first
    if !resp.is_download {
        if let Some(file_cache) = cache {
            let rel_path = resp.path.strip_prefix(&file_cache.root)
                .ok()
                .and_then(|p| p.to_str())
                .map(|s| s.to_string());

            if let Some(rel) = rel_path {
                // Try cache lookup first
                if let Some(entry) = file_cache.get(&rel) {
                    // Cache hit
                    file_cache.cache_hits.fetch_add(1, Ordering::Relaxed);
                    GLOBAL_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
                    return serve_cached(stream, &entry, method, if_none_match, accepts_gzip, resp.cache_control, metrics).await;
                }

                // Cache miss — try to insert from disk
                file_cache.cache_misses.fetch_add(1, Ordering::Relaxed);
                GLOBAL_CACHE_MISSES.fetch_add(1, Ordering::Relaxed);

                if file_cache.is_safe(&resp.path) {
                    if let Some(entry) = file_cache.insert(&rel, &resp.path, resp.mime, resp.cache_control) {
                        return serve_cached(stream, &entry, method, if_none_match, accepts_gzip, resp.cache_control, metrics).await;
                    }
                }
            }
        }
    }

    // Cache miss or download — fall back to streaming from disk
    serve_from_disk(stream, resp, method, if_none_match, accepts_gzip, metrics, cache.map(|c| c.gzip_enabled).unwrap_or(false)).await
}

/// Serve from in-memory cache with optional gzip.
async fn serve_cached<S: AsyncWrite + Unpin>(
    stream: &mut S,
    entry: &CachedFile,
    method: &str,
    if_none_match: Option<&str>,
    accepts_gzip: bool,
    cache_control: &str,
    metrics: &Metrics,
) -> std::io::Result<()> {
    // 304 Not Modified?
    if let Some(inm) = if_none_match {
        if inm.trim() == entry.etag || inm.contains(&entry.etag) {
            let headers = format!(
                "HTTP/1.1 304 Not Modified\r\n\
                 etag: {}\r\n\
                 cache-control: {}\r\n\
                 \r\n",
                entry.etag, cache_control,
            );
            stream.write_all(headers.as_bytes()).await?;
            stream.flush().await?;
            return Ok(());
        }
    }

    // Pick gzip or raw body
    let (body, content_encoding) = if accepts_gzip {
        if let Some(ref gzip) = entry.gzip {
            (gzip.as_slice(), "content-encoding: gzip\r\nvary: Accept-Encoding\r\n")
        } else {
            (entry.raw.as_slice(), "vary: Accept-Encoding\r\n")
        }
    } else {
        (entry.raw.as_slice(), "vary: Accept-Encoding\r\n")
    };

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
            entry.mime, body.len(), cache_control, entry.etag, content_encoding,
        );
        stream.write_all(headers.as_bytes()).await?;
        stream.flush().await?;
        return Ok(());
    }

    let headers = format!(
        "HTTP/1.1 200 OK\r\n\
         content-type: {}\r\n\
         content-length: {}\r\n\
         cache-control: {}\r\n\
         etag: {}\r\n\
         {}\
         \r\n",
        entry.mime, body.len(), cache_control, entry.etag, content_encoding,
    );
    stream.write_all(headers.as_bytes()).await?;
    stream.write_all(body).await?;
    stream.flush().await?;
    metrics.bytes_tx(body.len() as u64);
    Ok(())
}

/// Serve from disk with streaming + optional on-the-fly gzip for small text files.
async fn serve_from_disk<S: AsyncWrite + Unpin>(
    stream: &mut S,
    resp: &FileResponse,
    method: &str,
    if_none_match: Option<&str>,
    _accepts_gzip: bool,
    metrics: &Metrics,
    _gzip_enabled: bool,
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

    let file = match tokio::fs::File::open(&resp.path).await {
        Ok(f) => f,
        Err(e) => {
            tracing::warn!(path = %resp.path.display(), "File open error: {}", e);
            return write_raw(stream, 500, "text/plain", b"Internal Server Error").await;
        }
    };

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

    // Stream file body in 64 KiB chunks
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

/// Check if a MIME type is compressible (text-based formats).
fn is_compressible(mime: &str) -> bool {
    mime.starts_with("text/")
        || mime.starts_with("application/javascript")
        || mime.starts_with("application/json")
        || mime.starts_with("application/xml")
        || mime.starts_with("application/wasm")
        || mime.starts_with("image/svg")
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

/// Parse Accept-Encoding header to check for gzip support.
pub fn accepts_gzip(headers_data: &[u8]) -> bool {
    // Fast scan for "gzip" in the raw header bytes
    // This avoids full header parsing for the common case
    if let Ok(s) = std::str::from_utf8(headers_data) {
        for line in s.split("\r\n") {
            if let Some(value) = line.strip_prefix("Accept-Encoding:").or_else(|| line.strip_prefix("accept-encoding:")) {
                return value.contains("gzip");
            }
        }
    }
    false
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
            ..Default::default()
        };
        assert!(matches!(route("/api/v1/mining/challenge", &config), RouteResult::Proxy));
        assert!(matches!(route("/api/v1/events", &config), RouteResult::Proxy));
        assert!(matches!(route("/ws", &config), RouteResult::Proxy));
        assert!(matches!(route("/health", &config), RouteResult::Proxy));
        assert!(matches!(route("/aioc/test", &config), RouteResult::Proxy));
    }

    #[test]
    fn test_no_static_config_proxies_everything() {
        let config = StaticConfig { root: None, ..Default::default() };
        assert!(matches!(route("/", &config), RouteResult::Proxy));
        assert!(matches!(route("/app.js", &config), RouteResult::Proxy));
    }

    #[test]
    fn test_compressible_types() {
        assert!(is_compressible("text/html; charset=utf-8"));
        assert!(is_compressible("text/css; charset=utf-8"));
        assert!(is_compressible("application/javascript; charset=utf-8"));
        assert!(is_compressible("application/json; charset=utf-8"));
        assert!(is_compressible("image/svg+xml"));
        assert!(is_compressible("application/wasm"));
        assert!(!is_compressible("image/png"));
        assert!(!is_compressible("image/jpeg"));
        assert!(!is_compressible("font/woff2"));
        assert!(!is_compressible("application/octet-stream"));
    }

    #[test]
    fn test_accepts_gzip_parsing() {
        assert!(accepts_gzip(b"GET / HTTP/1.1\r\nAccept-Encoding: gzip, deflate, br\r\n\r\n"));
        assert!(accepts_gzip(b"GET / HTTP/1.1\r\naccept-encoding: gzip\r\n\r\n"));
        assert!(!accepts_gzip(b"GET / HTTP/1.1\r\nAccept-Encoding: deflate, br\r\n\r\n"));
        assert!(!accepts_gzip(b"GET / HTTP/1.1\r\n\r\n"));
    }

    #[test]
    fn test_gzip_compression() {
        // Test that gzip actually compresses text
        let data = "function hello() { console.log('hello world'); }\n".repeat(100);
        let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
        use std::io::Write;
        encoder.write_all(data.as_bytes()).unwrap();
        let compressed = encoder.finish().unwrap();
        assert!(compressed.len() < data.len() / 2, "gzip should compress JS significantly");
    }

    #[test]
    fn test_mime_html() {
        assert_eq!(mime_for_path(Path::new("page.html")), "text/html; charset=utf-8");
        assert_eq!(mime_for_path(Path::new("page.htm")), "text/html; charset=utf-8");
    }

    #[test]
    fn test_mime_images() {
        assert_eq!(mime_for_path(Path::new("photo.jpg")), "image/jpeg");
        assert_eq!(mime_for_path(Path::new("photo.jpeg")), "image/jpeg");
        assert_eq!(mime_for_path(Path::new("icon.gif")), "image/gif");
        assert_eq!(mime_for_path(Path::new("logo.svg")), "image/svg+xml");
        assert_eq!(mime_for_path(Path::new("icon.ico")), "image/x-icon");
        assert_eq!(mime_for_path(Path::new("photo.webp")), "image/webp");
        assert_eq!(mime_for_path(Path::new("photo.avif")), "image/avif");
    }

    #[test]
    fn test_mime_fonts() {
        assert_eq!(mime_for_path(Path::new("f.woff")), "font/woff");
        assert_eq!(mime_for_path(Path::new("f.woff2")), "font/woff2");
        assert_eq!(mime_for_path(Path::new("f.ttf")), "font/ttf");
        assert_eq!(mime_for_path(Path::new("f.eot")), "application/vnd.ms-fontobject");
    }

    #[test]
    fn test_mime_video() {
        assert_eq!(mime_for_path(Path::new("v.mp4")), "video/mp4");
        assert_eq!(mime_for_path(Path::new("v.webm")), "video/webm");
    }

    #[test]
    fn test_mime_misc() {
        assert_eq!(mime_for_path(Path::new("data.json")), "application/json; charset=utf-8");
        assert_eq!(mime_for_path(Path::new("data.xml")), "application/xml; charset=utf-8");
        assert_eq!(mime_for_path(Path::new("readme.txt")), "text/plain; charset=utf-8");
        assert_eq!(mime_for_path(Path::new("doc.pdf")), "application/pdf");
        assert_eq!(mime_for_path(Path::new("bundle.map")), "application/json");
    }

    #[test]
    fn test_mime_case_insensitive() {
        assert_eq!(mime_for_path(Path::new("app.JS")), "application/javascript; charset=utf-8");
        assert_eq!(mime_for_path(Path::new("style.CSS")), "text/css; charset=utf-8");
        assert_eq!(mime_for_path(Path::new("IMAGE.PNG")), "image/png");
    }

    #[test]
    fn test_static_extensions_all() {
        let exts = ["js", "css", "png", "jpg", "jpeg", "gif", "svg", "ico",
                     "woff", "woff2", "ttf", "eot", "map", "json", "wasm",
                     "webp", "avif", "mp4", "webm", "pdf", "txt", "xml",
                     "html", "htm"];
        for ext in exts {
            assert!(has_static_extension(&format!("/file.{}", ext)),
                    "extension .{} should be static", ext);
        }
    }

    #[test]
    fn test_non_static_extensions() {
        assert!(!has_static_extension("/file.rs"));
        assert!(!has_static_extension("/file.toml"));
        assert!(!has_static_extension("/file.lock"));
        assert!(!has_static_extension("/file.md"));
    }

    #[test]
    fn test_hashed_asset_variations() {
        assert!(is_hashed_asset("/assets/vendor-ABCD1234.js"));
        assert!(is_hashed_asset("/assets/chunk-a1b2c3d4e5f6.css"));
        assert!(is_hashed_asset("/app-12345678.js"));
    }

    #[test]
    fn test_not_hashed_asset() {
        assert!(!is_hashed_asset("/index.html"));
        assert!(!is_hashed_asset("/assets/app.js"));
        assert!(!is_hashed_asset("/style.css"));
        assert!(!is_hashed_asset("/assets/app-abc.js"));
    }

    #[test]
    fn test_route_strips_query_string() {
        let config = StaticConfig {
            root: Some(PathBuf::from("/tmp")),
            spa_fallback: false,
            ..Default::default()
        };
        assert!(matches!(route("/api/v1/data?key=val", &config), RouteResult::Proxy));
        assert!(matches!(route("/api?x=1", &config), RouteResult::Proxy));
    }

    #[test]
    fn test_route_strips_fragment() {
        let config = StaticConfig {
            root: Some(PathBuf::from("/tmp")),
            spa_fallback: false,
            ..Default::default()
        };
        assert!(matches!(route("/api/v1/data#section", &config), RouteResult::Proxy));
    }

    #[test]
    fn test_ws_paths_always_proxy() {
        let config = StaticConfig {
            root: Some(PathBuf::from("/tmp")),
            spa_fallback: true,
            ..Default::default()
        };
        assert!(matches!(route("/ws", &config), RouteResult::Proxy));
        assert!(matches!(route("/ws/connect", &config), RouteResult::Proxy));
    }

    #[test]
    fn test_cors_headers_content() {
        assert!(CORS_HEADERS.contains("access-control-allow-origin: *"));
        assert!(CORS_HEADERS.contains("access-control-allow-methods:"));
        assert!(CORS_HEADERS.contains("access-control-allow-headers:"));
        assert!(CORS_HEADERS.contains("X-Wallet-Address"));
    }
}
