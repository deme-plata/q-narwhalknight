use serde::Deserialize;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};

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
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct StaticConfig {
    pub root: Option<PathBuf>,
    #[serde(default = "default_spa_fallback")]
    pub spa_fallback: bool,
}

fn default_spa_fallback() -> bool { true }

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
    /// Total max = workers × this value. With 48 workers and 64, total = 3072.
    /// Excess requests get an immediate 503. Prevents connection pileup on backend.
    #[serde(default = "default_max_inflight_per_worker")]
    pub max_inflight_per_worker: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LimitsConfig {
    #[serde(default = "default_max_connections")]
    pub max_connections: usize,
    #[serde(default = "default_max_conns_per_ip")]
    pub max_conns_per_ip: usize,
    #[serde(default = "default_request_body_limit")]
    pub request_body_limit: usize,
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

fn default_rate_limit_per_ip() -> usize { 100 }
fn default_rate_limit_burst() -> usize { 200 }
fn default_rate_limit_global_rps() -> usize { 100_000 }
fn default_max_inflight_per_worker() -> usize { 64 }
fn default_cluster_health_interval() -> std::time::Duration { std::time::Duration::from_secs(10) }

impl Default for LimitsConfig {
    fn default() -> Self {
        Self {
            max_connections: default_max_connections(),
            max_conns_per_ip: default_max_conns_per_ip(),
            request_body_limit: default_request_body_limit(),
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
        // Validate
        if config.upstream.backends.is_empty() {
            anyhow::bail!("At least one upstream backend is required");
        }
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
        Ok(config)
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
    if let Some(secs) = s.strip_suffix('s') {
        secs.trim().parse::<u64>()
            .map(std::time::Duration::from_secs)
            .map_err(|e| format!("Invalid duration '{}': {}", s, e))
    } else if let Some(ms) = s.strip_suffix("ms") {
        ms.trim().parse::<u64>()
            .map(std::time::Duration::from_millis)
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
