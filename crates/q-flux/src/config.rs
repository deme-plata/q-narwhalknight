use serde::Deserialize;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};

/// Virtual host configuration — per-domain TLS cert, backend, and static root.
///
/// Use `[[vhosts]]` in q-flux.toml to add entries. Each vhost must have
/// its own TLS cert/key. The `backend` and `static_root` fields override
/// the defaults for requests matching the listed `domains`.
///
/// Example:
/// ```toml
/// [[vhosts]]
/// domains = ["bounty.quillon.xyz"]
/// cert = "/etc/letsencrypt/live/bounty.quillon.xyz/fullchain.pem"
/// key  = "/etc/letsencrypt/live/bounty.quillon.xyz/privkey.pem"
/// backend     = "127.0.0.1:8083"
/// static_root = "/var/www/bounty.quillon.xyz"
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct VhostConfig {
    /// Domain names for SNI matching (e.g. `["bounty.quillon.xyz"]`).
    pub domains: Vec<String>,
    /// TLS certificate file (PEM, full chain).
    pub cert: PathBuf,
    /// TLS private key file (PEM).
    pub key: PathBuf,
    /// Backend address (`host:port`) to proxy API requests to.
    /// Falls back to the main `[upstream]` backends if absent.
    #[serde(default)]
    pub backend: Option<String>,
    /// Static files root directory.
    /// Falls back to `[static_files].root` if absent.
    #[serde(default)]
    pub static_root: Option<PathBuf>,
    /// Serve index.html for unmatched paths (SPA mode). Default: true.
    #[serde(default = "default_spa_fallback")]
    pub spa_fallback: bool,
    /// Additional path prefixes that must always be proxied to the backend,
    /// even when a static_root is set. Useful for API paths on vhosts that
    /// also serve static files (e.g. `["/v1", "/health"]` for bounty site).
    #[serde(default)]
    pub proxy_paths: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FluxConfig {
    pub server: ServerConfig,
    pub tls: TlsConfig,
    pub upstream: UpstreamConfig,
    #[serde(default)]
    pub limits: LimitsConfig,
    #[serde(default)]
    pub logging: LoggingConfig,
    #[serde(default)]
    pub static_files: StaticConfig,
    /// Super-cluster: cross-node failover backends.
    /// When all local upstream backends are unhealthy, q-flux routes to cluster
    /// peers instead of returning 503. Local backends always have priority.
    #[serde(default)]
    pub cluster: ClusterConfig,
    /// io_uring and splice(2) zero-copy configuration.
    #[serde(default)]
    pub io_uring: IoUringSection,
    /// IP allowlist/blocklist access control (Issue #027).
    #[serde(default)]
    pub access_control: AccessControlConfig,
    /// ACME certificate automation (Issue #021).
    #[serde(default)]
    pub acme: AcmeConfig,
    /// Virtual hosts: per-domain TLS cert + optional backend/static overrides.
    /// Uses SNI during TLS handshake to select the right certificate.
    #[serde(default)]
    pub vhosts: Vec<VhostConfig>,
    /// LibP2P WebSocket proxy: browser js-libp2p connects via WSS on a dedicated
    /// port (e.g. 9443), and q-flux does TLS termination + raw TCP proxy to the
    /// libp2p WebSocket listener (e.g. 127.0.0.1:9002). No HTTP parsing — just
    /// bidirectional byte passthrough after TLS.
    #[serde(default)]
    pub libp2p_ws: Libp2pWsConfig,
}

/// LibP2P WebSocket proxy configuration.
///
/// When enabled, connections arriving on the configured port are TLS-terminated
/// and then raw-proxied (bidirectional byte copy) to the libp2p WebSocket
/// listener backend. This avoids HTTP/2 ALPN negotiation issues — the browser
/// needs HTTP/1.1 for the WebSocket upgrade handshake, and the libp2p backend
/// handles the upgrade itself.
#[derive(Debug, Clone, Deserialize)]
pub struct Libp2pWsConfig {
    /// Enable the libp2p WebSocket proxy. Default: false.
    #[serde(default)]
    pub enabled: bool,
    /// Port that browser js-libp2p connects to (e.g. 9443).
    /// Must also be listed in server.listen.
    #[serde(default = "default_libp2p_ws_port")]
    pub port: u16,
    /// Backend address of the libp2p WebSocket listener (e.g. "127.0.0.1:9002").
    #[serde(default = "default_libp2p_ws_backend")]
    pub backend: String,
    /// Also proxy WebSocket upgrades arriving on the main HTTPS port (443) to
    /// the libp2p backend. This allows nodes behind restrictive NAT/firewalls
    /// to connect via port 443 (the only port guaranteed to be open).
    /// Default: false.
    #[serde(default)]
    pub proxy_on_main_port: bool,
}

impl Default for Libp2pWsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            port: 9443,
            backend: "127.0.0.1:9002".to_string(),
            proxy_on_main_port: false,
        }
    }
}

fn default_libp2p_ws_port() -> u16 { 9443 }
fn default_libp2p_ws_backend() -> String { "127.0.0.1:9002".to_string() }

/// io_uring and splice(2) zero-copy configuration.
///
/// When `splice_enabled` is true, WebSocket and SSE passthrough connections
/// use Linux splice(2) for zero-copy bidirectional data transfer. Data moves
/// directly between sockets in the kernel via a pipe, never touching userspace.
///
/// Falls back to normal tokio::io::copy on non-Linux or when splice fails.
#[derive(Debug, Clone, Deserialize)]
pub struct IoUringSection {
    /// Enable splice(2) zero-copy for WebSocket/SSE passthrough.
    /// Requires Linux. Falls back gracefully if splice fails at runtime.
    /// Default: true on Linux, false elsewhere.
    #[serde(default = "default_splice_enabled")]
    pub splice_enabled: bool,
    /// Pipe buffer size for splice channels (bytes).
    /// Larger pipes allow more data in-flight but consume kernel memory.
    /// Kernel rounds up to nearest page size. Default: 65536 (64KB).
    #[serde(default = "default_splice_pipe_size")]
    pub splice_pipe_size: usize,
    /// io_uring submission queue depth (power of two, >= 64).
    /// Only used if full io_uring event loop is enabled. Default: 4096.
    #[serde(default = "default_uring_queue_depth")]
    pub queue_depth: u32,
    /// Number of pre-allocated io_uring buffers. Default: 1024.
    #[serde(default = "default_uring_buffer_count")]
    pub buffer_count: u32,
    /// Size of each io_uring buffer in bytes. Default: 16384 (16KB).
    #[serde(default = "default_uring_buffer_size")]
    pub buffer_size: u32,
}

/// Super-cluster configuration for cross-node failover.
///
/// Cluster peers are remote q-api-server backends on other servers.
/// They are only used when ALL local backends (in `[upstream].backends`) are
/// unhealthy. This gives automatic failover without manual Nginx weight changes.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct ClusterConfig {
    /// Remote backend addresses (e.g. ["89.149.241.126:8080", "185.182.185.227:8080"]).
    #[serde(default)]
    pub peers: Vec<String>,
    /// Health check path for cluster peers (default: same as upstream).
    #[serde(default)]
    pub health_check_path: Option<String>,
    /// Health check interval for cluster peers (default: 10s, slower than local).
    #[serde(default = "default_cluster_health_interval", deserialize_with = "deserialize_duration")]
    pub health_check_interval: std::time::Duration,
    /// Health check timeout for cluster peers (default: 5s, longer than local 3s for cross-DC).
    #[serde(default = "default_cluster_health_timeout", deserialize_with = "deserialize_duration")]
    pub health_check_timeout: std::time::Duration,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StaticConfig {
    pub root: Option<PathBuf>,
    #[serde(default = "default_spa_fallback")]
    pub spa_fallback: bool,
    /// Enable gzip compression for text assets (JS, CSS, HTML, JSON, SVG).
    /// Reduces ~2.1MB index.js to ~550KB. Default: true.
    #[serde(default = "default_true")]
    pub gzip: bool,
    /// Max file size to cache in memory (bytes). Files larger than this are
    /// streamed from disk. Default: 4MB (covers all JS/CSS bundles).
    #[serde(default = "default_cache_max_file_size")]
    pub cache_max_file_size: usize,
    /// Max total memory for file cache (bytes). Default: 64MB.
    #[serde(default = "default_cache_max_total")]
    pub cache_max_total: usize,
    /// Enable gzip/Brotli compression for proxied API responses (JSON, text).
    /// Compresses responses >= 1KB with compressible content-types.
    /// Brotli preferred when client supports it, gzip as fallback.
    /// Default: true.
    #[serde(default = "default_true")]
    pub proxy_compression: bool,
    /// Additional path prefixes to always proxy, bypassing static file serving.
    /// Populated from VhostConfig.proxy_paths at startup.
    #[serde(default)]
    pub proxy_paths: Vec<String>,
}

/// IP access control configuration (Issue #027).
///
/// Modes:
/// - `disabled` (default): all IPs allowed
/// - `blocklist`: listed IPs/CIDRs blocked, all others allowed
/// - `allowlist`: only listed IPs/CIDRs allowed, all others blocked
///
/// Checked BEFORE TLS handshake — blocked IPs consume zero resources.
#[derive(Debug, Clone, Deserialize)]
pub struct AccessControlConfig {
    /// Access control mode: "disabled", "allowlist", or "blocklist".
    #[serde(default = "default_access_mode")]
    pub mode: String,
    /// IP addresses or CIDR ranges to allow (used in allowlist mode).
    /// Examples: ["10.0.0.0/8", "192.168.1.100"]
    #[serde(default)]
    pub allowlist: Vec<String>,
    /// IP addresses or CIDR ranges to block (used in blocklist mode).
    /// Examples: ["10.0.0.0/8", "192.168.1.0/24"]
    #[serde(default)]
    pub blocklist: Vec<String>,
}

impl Default for AccessControlConfig {
    fn default() -> Self {
        Self {
            mode: "disabled".to_string(),
            allowlist: Vec::new(),
            blocklist: Vec::new(),
        }
    }
}

fn default_access_mode() -> String { "disabled".to_string() }

/// ACME certificate automation configuration (Issue #021).
///
/// When enabled, q-flux automatically obtains and renews TLS certificates
/// from Let's Encrypt (or another ACME CA) using the HTTP-01 challenge.
#[derive(Debug, Clone, Deserialize)]
pub struct AcmeConfig {
    /// Enable ACME certificate automation. Default: false.
    #[serde(default)]
    pub enabled: bool,
    /// Domain names to obtain certificates for.
    /// Example: ["quillon.xyz", "www.quillon.xyz"]
    #[serde(default)]
    pub domains: Vec<String>,
    /// Contact email for the ACME account (used for expiry notifications).
    #[serde(default)]
    pub email: Option<String>,
    /// ACME directory URL. Default: Let's Encrypt production.
    #[serde(default = "default_acme_directory")]
    pub directory_url: String,
    /// Directory to store certificates and account keys.
    /// Default: "/etc/q-flux/acme"
    #[serde(default = "default_acme_cert_dir")]
    pub cert_dir: std::path::PathBuf,
    /// Days before expiry to trigger renewal. Default: 30.
    #[serde(default = "default_acme_renewal_days")]
    pub renewal_days: u32,
}

impl Default for AcmeConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            domains: Vec::new(),
            email: None,
            directory_url: default_acme_directory(),
            cert_dir: default_acme_cert_dir(),
            renewal_days: default_acme_renewal_days(),
        }
    }
}

fn default_acme_directory() -> String {
    "https://acme-v02.api.letsencrypt.org/directory".to_string()
}
fn default_acme_cert_dir() -> std::path::PathBuf {
    std::path::PathBuf::from("/etc/q-flux/acme")
}
fn default_acme_renewal_days() -> u32 { 30 }

fn default_spa_fallback() -> bool { true }
fn default_true() -> bool { true }
fn default_cache_max_file_size() -> usize { 4 * 1024 * 1024 } // 4MB
fn default_cache_max_total() -> usize { 64 * 1024 * 1024 } // 64MB

#[derive(Debug, Clone, Deserialize)]
pub struct ServerConfig {
    #[serde(default = "default_listen")]
    pub listen: Vec<String>,
    #[serde(default)]
    pub workers: usize, // 0 = auto-detect
    /// Admin HTTP server listen address (default: 127.0.0.1:9090).
    /// Set to "0.0.0.0:9090" to expose externally (not recommended).
    #[serde(default = "default_admin_listen")]
    pub admin_listen: SocketAddr,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TlsConfig {
    pub cert: PathBuf,
    pub key: PathBuf,
    /// Path to a DER-encoded OCSP response file for OCSP stapling.
    /// When set, the TLS handshake includes the stapled OCSP response,
    /// eliminating the 50-100ms OCSP lookup penalty for clients.
    #[serde(default)]
    pub ocsp_staple: Option<PathBuf>,
    /// Seconds to allow old TLS connections to drain after a certificate reload.
    /// This is informational -- old connections naturally use the old config via Arc,
    /// and new connections get the new config. Default: 30 seconds.
    #[serde(default = "default_drain_timeout")]
    pub drain_timeout_secs: u64,
    /// Enable kTLS (kernel TLS) offload on Linux 4.13+.
    /// When enabled, symmetric encryption/decryption is offloaded to the kernel,
    /// allowing zero-copy sendfile() for static files and reducing context switches.
    /// Falls back gracefully to userspace TLS if kTLS is not available.
    /// Default: false (requires Linux + kernel TLS module loaded).
    #[serde(default)]
    pub enable_ktls: bool,
}

fn default_drain_timeout() -> u64 { 30 }

#[derive(Debug, Clone, Deserialize)]
pub struct UpstreamConfig {
    pub backends: Vec<String>,
    #[serde(default = "default_max_conns_per_worker")]
    pub max_conns_per_worker: usize,
    #[serde(default = "default_keepalive_timeout", deserialize_with = "deserialize_duration")]
    pub keepalive_timeout: std::time::Duration,
    #[serde(default = "default_connect_timeout", deserialize_with = "deserialize_duration")]
    pub connect_timeout: std::time::Duration,
    #[serde(default = "default_response_timeout", deserialize_with = "deserialize_duration")]
    pub response_timeout: std::time::Duration,
    /// How often to probe each backend for health (default: 5s).
    #[serde(default = "default_health_check_interval", deserialize_with = "deserialize_duration")]
    pub health_check_interval: std::time::Duration,
    /// HTTP path to GET for health checks (default: "/api/v1/status").
    /// Set to "" for TCP-only checks.
    #[serde(default = "default_health_check_path")]
    pub health_check_path: String,
    /// Timeout for a single health probe including TCP connect + HTTP GET (default: 3s).
    #[serde(default = "default_health_check_timeout", deserialize_with = "deserialize_duration")]
    pub health_check_timeout: std::time::Duration,
    /// Max concurrent upstream requests per worker (default: 64).
    /// Only used as fallback if `max_upstream_global` is 0.
    #[serde(default = "default_max_inflight_per_worker")]
    pub max_inflight_per_worker: usize,
    /// Global max concurrent upstream requests across ALL workers (default: 2048).
    /// This is the preferred setting: a single shared semaphore prevents the death
    /// spiral where 48 workers × N permits each overwhelm a single backend.
    /// Set to 0 to fall back to per-worker limits (max_inflight_per_worker).
    /// Recommended: 512-1024 for single-backend, 1024-2048 for multi-backend.
    #[serde(default = "default_max_upstream_global")]
    pub max_upstream_global: usize,
    /// How long to wait for a semaphore permit before returning 502 (default: 500ms).
    /// With the old `try_acquire()` approach, permits were instant-reject on exhaustion,
    /// causing thousands of 502s/sec during brief backend stalls. Queued acquire with a
    /// timeout lets requests wait for permits freed by completing responses, dramatically
    /// reducing spurious 502s. Set to "0ms" to restore old instant-reject behavior.
    #[serde(default = "default_acquire_timeout", deserialize_with = "deserialize_duration")]
    pub acquire_timeout: std::time::Duration,
    /// Number of consecutive health check failures before marking backend unhealthy.
    /// Default: 3.
    #[serde(default = "default_failure_threshold")]
    pub failure_threshold: u32,
    /// Number of consecutive health check successes required to promote a backend
    /// from half-open (recovering) to fully healthy. Default: 2.
    #[serde(default = "default_healthy_threshold")]
    pub healthy_threshold: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LimitsConfig {
    #[serde(default = "default_max_connections")]
    pub max_connections: usize,
    #[serde(default = "default_max_conns_per_ip")]
    pub max_conns_per_ip: usize,
    #[serde(default = "default_request_body_limit")]
    pub request_body_limit: usize,
    /// Request body size threshold (bytes) above which the body is streamed
    /// directly to the upstream instead of being fully buffered first.
    /// Streaming bodies cannot be retried on a different backend (body is consumed).
    /// Default: 10MB. Set to 0 to disable streaming (always buffer).
    #[serde(default = "default_streaming_body_threshold")]
    pub streaming_body_threshold: usize,
    /// Token-bucket rate limit per IP (requests/sec). 0 = disabled.
    #[serde(default = "default_rate_limit_per_ip")]
    pub rate_limit_per_ip: usize,
    /// Token-bucket burst capacity per IP.
    #[serde(default = "default_rate_limit_burst")]
    pub rate_limit_burst: usize,
    /// Global token-bucket rate limit (requests/sec across all IPs).
    #[serde(default = "default_rate_limit_global_rps")]
    pub rate_limit_global_rps: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LoggingConfig {
    #[serde(default = "default_log_level")]
    pub level: String,
    pub access_log: Option<PathBuf>,
}

// Defaults
fn default_admin_listen() -> SocketAddr {
    SocketAddr::from(([127, 0, 0, 1], 9090))
}
fn default_listen() -> Vec<String> {
    vec!["0.0.0.0:443".into(), "0.0.0.0:80".into()]
}
fn default_max_conns_per_worker() -> usize { 32 }
fn default_keepalive_timeout() -> std::time::Duration { std::time::Duration::from_secs(30) }
fn default_connect_timeout() -> std::time::Duration { std::time::Duration::from_secs(5) }
fn default_response_timeout() -> std::time::Duration { std::time::Duration::from_secs(30) }
fn default_health_check_interval() -> std::time::Duration { std::time::Duration::from_secs(5) }
fn default_health_check_path() -> String { "/api/v1/status".to_string() }
fn default_health_check_timeout() -> std::time::Duration { std::time::Duration::from_secs(3) }
fn default_max_connections() -> usize { 10_000_000 }
fn default_max_conns_per_ip() -> usize { 500 }
fn default_request_body_limit() -> usize { 25 * 1024 * 1024 } // 25MB
fn default_log_level() -> String { "info".into() }

fn default_streaming_body_threshold() -> usize { 10 * 1024 * 1024 } // 10MB
fn default_rate_limit_per_ip() -> usize { 100 }
fn default_rate_limit_burst() -> usize { 200 }
fn default_rate_limit_global_rps() -> usize { 100_000 }
fn default_max_inflight_per_worker() -> usize { 64 }
fn default_max_upstream_global() -> usize { 2048 }
fn default_acquire_timeout() -> std::time::Duration { std::time::Duration::from_millis(500) }
fn default_failure_threshold() -> u32 { 3 }
fn default_healthy_threshold() -> u32 { 2 }
fn default_cluster_health_interval() -> std::time::Duration { std::time::Duration::from_secs(10) }
fn default_cluster_health_timeout() -> std::time::Duration { std::time::Duration::from_secs(5) }

#[cfg(target_os = "linux")]
fn default_splice_enabled() -> bool { true }
#[cfg(not(target_os = "linux"))]
fn default_splice_enabled() -> bool { false }
fn default_splice_pipe_size() -> usize { 65536 }
fn default_uring_queue_depth() -> u32 { 4096 }
fn default_uring_buffer_count() -> u32 { 1024 }
fn default_uring_buffer_size() -> u32 { 16384 }

impl Default for IoUringSection {
    fn default() -> Self {
        Self {
            splice_enabled: default_splice_enabled(),
            splice_pipe_size: default_splice_pipe_size(),
            queue_depth: default_uring_queue_depth(),
            buffer_count: default_uring_buffer_count(),
            buffer_size: default_uring_buffer_size(),
        }
    }
}

impl Default for StaticConfig {
    fn default() -> Self {
        Self {
            root: None,
            spa_fallback: default_spa_fallback(),
            gzip: default_true(),
            cache_max_file_size: default_cache_max_file_size(),
            cache_max_total: default_cache_max_total(),
            proxy_compression: default_true(),
            proxy_paths: Vec::new(),
        }
    }
}

impl Default for LimitsConfig {
    fn default() -> Self {
        Self {
            max_connections: default_max_connections(),
            max_conns_per_ip: default_max_conns_per_ip(),
            request_body_limit: default_request_body_limit(),
            streaming_body_threshold: default_streaming_body_threshold(),
            rate_limit_per_ip: default_rate_limit_per_ip(),
            rate_limit_burst: default_rate_limit_burst(),
            rate_limit_global_rps: default_rate_limit_global_rps(),
        }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: default_log_level(),
            access_log: None,
        }
    }
}

impl FluxConfig {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("Failed to read config {}: {}", path.display(), e))?;
        let config: FluxConfig = toml::from_str(&content)
            .map_err(|e| anyhow::anyhow!("Failed to parse config {}: {}", path.display(), e))?;
        config.validate()?;
        // File-system checks (only in load, not in validate — tests don't need real files)
        if !config.tls.cert.exists() {
            anyhow::bail!("TLS cert not found: {}", config.tls.cert.display());
        }
        if !config.tls.key.exists() {
            anyhow::bail!("TLS key not found: {}", config.tls.key.display());
        }
        if let Some(ref ocsp_path) = config.tls.ocsp_staple {
            if !ocsp_path.exists() {
                anyhow::bail!("OCSP staple file not found: {}", ocsp_path.display());
            }
        }
        if let Some(ref root) = config.static_files.root {
            if !root.exists() {
                anyhow::bail!("Static files root not found: {}", root.display());
            }
        }
        for (i, vhost) in config.vhosts.iter().enumerate() {
            if !vhost.cert.exists() {
                anyhow::bail!("vhosts[{}] cert not found: {}", i, vhost.cert.display());
            }
            if !vhost.key.exists() {
                anyhow::bail!("vhosts[{}] key not found: {}", i, vhost.key.display());
            }
            if let Some(ref root) = vhost.static_root {
                if !root.exists() {
                    anyhow::bail!("vhosts[{}] static_root not found: {}", i, root.display());
                }
            }
        }
        Ok(config)
    }

    /// Validate config values without checking filesystem.
    pub fn validate(&self) -> anyhow::Result<()> {
        // --- upstream ---
        if self.upstream.backends.is_empty() {
            anyhow::bail!("At least one upstream backend is required");
        }
        for b in &self.upstream.backends {
            Self::validate_host_port(b, "upstream backend")?;
        }
        for p in &self.cluster.peers {
            Self::validate_host_port(p, "cluster peer")?;
        }
        for (i, vhost) in self.vhosts.iter().enumerate() {
            if vhost.domains.is_empty() {
                anyhow::bail!("vhosts[{}] must have at least one domain", i);
            }
            if let Some(ref b) = vhost.backend {
                Self::validate_host_port(b, &format!("vhosts[{}] backend", i))?;
            }
        }

        // --- listen addresses ---
        if self.server.listen.is_empty() {
            anyhow::bail!("At least one listen address is required");
        }
        for addr in &self.server.listen {
            addr.parse::<SocketAddr>()
                .map_err(|_| anyhow::anyhow!("Invalid listen address: '{}' (expected host:port)", addr))?;
        }

        // --- timeouts ---
        if self.upstream.connect_timeout >= self.upstream.response_timeout {
            anyhow::bail!(
                "connect_timeout ({:?}) must be less than response_timeout ({:?})",
                self.upstream.connect_timeout, self.upstream.response_timeout
            );
        }
        if self.upstream.health_check_timeout >= self.upstream.health_check_interval {
            anyhow::bail!(
                "health_check_timeout ({:?}) must be less than health_check_interval ({:?})",
                self.upstream.health_check_timeout, self.upstream.health_check_interval
            );
        }

        // --- limits ---
        if self.upstream.max_conns_per_worker == 0 {
            anyhow::bail!("max_conns_per_worker must be > 0");
        }
        if self.limits.request_body_limit == 0 {
            anyhow::bail!("request_body_limit must be > 0");
        }

        Ok(())
    }

    /// Validate a "host:port" string (does not require DNS resolution, just format).
    fn validate_host_port(s: &str, label: &str) -> anyhow::Result<()> {
        if s.is_empty() {
            anyhow::bail!("Empty {} address", label);
        }
        // Try as SocketAddr first (covers IP:port)
        if s.parse::<SocketAddr>().is_ok() {
            return Ok(());
        }
        // Try as hostname:port (e.g. "backend.local:8080")
        match s.rsplit_once(':') {
            Some((host, port)) if !host.is_empty() && port.parse::<u16>().is_ok() => Ok(()),
            _ => anyhow::bail!("Invalid {} address: '{}' (expected host:port)", label, s),
        }
    }

    pub fn worker_count(&self) -> usize {
        if self.server.workers == 0 {
            num_cpus::get()
        } else {
            self.server.workers
        }
    }
}

fn deserialize_duration<'de, D>(deserializer: D) -> Result<std::time::Duration, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    parse_duration(&s).map_err(serde::de::Error::custom)
}

fn parse_duration(s: &str) -> Result<std::time::Duration, String> {
    let s = s.trim();
    // Check "ms" before 's' — "100ms" would otherwise match the 's' suffix first.
    if let Some(ms) = s.strip_suffix("ms") {
        ms.trim().parse::<u64>()
            .map(std::time::Duration::from_millis)
            .map_err(|e| format!("Invalid duration '{}': {}", s, e))
    } else if let Some(secs) = s.strip_suffix('s') {
        secs.trim().parse::<u64>()
            .map(std::time::Duration::from_secs)
            .map_err(|e| format!("Invalid duration '{}': {}", s, e))
    } else if let Some(mins) = s.strip_suffix('m') {
        mins.trim().parse::<u64>()
            .map(|m| std::time::Duration::from_secs(m * 60))
            .map_err(|e| format!("Invalid duration '{}': {}", s, e))
    } else {
        // Try parsing as raw seconds
        s.parse::<u64>()
            .map(std::time::Duration::from_secs)
            .map_err(|_| format!("Invalid duration '{}': use '30s', '100ms', or '5m'", s))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a minimal valid FluxConfig for testing validate().
    fn valid_config() -> FluxConfig {
        FluxConfig {
            server: ServerConfig {
                listen: vec!["0.0.0.0:443".into()],
                workers: 0,
                admin_listen: default_admin_listen(),
            },
            tls: TlsConfig {
                cert: PathBuf::from("/tmp/cert.pem"),
                key: PathBuf::from("/tmp/key.pem"),
                ocsp_staple: None,
                drain_timeout_secs: 30,
                enable_ktls: false,
            },
            upstream: UpstreamConfig {
                backends: vec!["127.0.0.1:8080".into()],
                max_conns_per_worker: default_max_conns_per_worker(),
                keepalive_timeout: default_keepalive_timeout(),
                connect_timeout: default_connect_timeout(),
                response_timeout: default_response_timeout(),
                health_check_interval: default_health_check_interval(),
                health_check_path: default_health_check_path(),
                health_check_timeout: default_health_check_timeout(),
                max_inflight_per_worker: default_max_inflight_per_worker(),
                max_upstream_global: default_max_upstream_global(),
                acquire_timeout: default_acquire_timeout(),
                failure_threshold: default_failure_threshold(),
                healthy_threshold: default_healthy_threshold(),
            },
            limits: LimitsConfig::default(),
            logging: LoggingConfig::default(),
            static_files: StaticConfig::default(),
            cluster: ClusterConfig::default(),
            io_uring: IoUringSection::default(),
            access_control: AccessControlConfig::default(),
            acme: AcmeConfig::default(),
            vhosts: Vec::new(),
            libp2p_ws: Libp2pWsConfig::default(),
        }
    }

    #[test]
    fn test_valid_config_passes() {
        valid_config().validate().unwrap();
    }

    #[test]
    fn test_empty_backends_rejected() {
        let mut cfg = valid_config();
        cfg.upstream.backends.clear();
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("upstream backend"), "{}", err);
    }

    #[test]
    fn test_invalid_backend_address_rejected() {
        let mut cfg = valid_config();
        cfg.upstream.backends = vec!["not-a-valid-address".into()];
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("Invalid upstream backend"), "{}", err);
    }

    #[test]
    fn test_hostname_port_backend_accepted() {
        let mut cfg = valid_config();
        cfg.upstream.backends = vec!["backend.local:8080".into()];
        cfg.validate().unwrap();
    }

    #[test]
    fn test_invalid_cluster_peer_rejected() {
        let mut cfg = valid_config();
        cfg.cluster.peers = vec!["bad-peer".into()];
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("cluster peer"), "{}", err);
    }

    #[test]
    fn test_empty_listen_rejected() {
        let mut cfg = valid_config();
        cfg.server.listen.clear();
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("listen address"), "{}", err);
    }

    #[test]
    fn test_invalid_listen_address_rejected() {
        let mut cfg = valid_config();
        cfg.server.listen = vec!["not-an-address".into()];
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("Invalid listen address"), "{}", err);
    }

    #[test]
    fn test_connect_timeout_gte_response_timeout_rejected() {
        let mut cfg = valid_config();
        cfg.upstream.connect_timeout = std::time::Duration::from_secs(30);
        cfg.upstream.response_timeout = std::time::Duration::from_secs(30);
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("connect_timeout"), "{}", err);
    }

    #[test]
    fn test_health_timeout_gte_interval_rejected() {
        let mut cfg = valid_config();
        cfg.upstream.health_check_timeout = std::time::Duration::from_secs(10);
        cfg.upstream.health_check_interval = std::time::Duration::from_secs(5);
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("health_check_timeout"), "{}", err);
    }

    #[test]
    fn test_zero_max_conns_rejected() {
        let mut cfg = valid_config();
        cfg.upstream.max_conns_per_worker = 0;
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("max_conns_per_worker"), "{}", err);
    }

    #[test]
    fn test_zero_body_limit_rejected() {
        let mut cfg = valid_config();
        cfg.limits.request_body_limit = 0;
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("request_body_limit"), "{}", err);
    }

    #[test]
    fn test_parse_duration_variants() {
        assert_eq!(parse_duration("30s").unwrap(), std::time::Duration::from_secs(30));
        assert_eq!(parse_duration("100ms").unwrap(), std::time::Duration::from_millis(100));
        assert_eq!(parse_duration("5m").unwrap(), std::time::Duration::from_secs(300));
        assert_eq!(parse_duration("60").unwrap(), std::time::Duration::from_secs(60));
        assert!(parse_duration("abc").is_err());
    }
}
