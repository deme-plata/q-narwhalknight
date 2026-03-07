use serde::Deserialize;
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
}

#[derive(Debug, Clone, Deserialize)]
pub struct ServerConfig {
    #[serde(default = "default_listen")]
    pub listen: Vec<String>,
    #[serde(default)]
    pub workers: usize, // 0 = auto-detect
}

#[derive(Debug, Clone, Deserialize)]
pub struct TlsConfig {
    pub cert: PathBuf,
    pub key: PathBuf,
}

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
}

#[derive(Debug, Clone, Deserialize)]
pub struct LimitsConfig {
    #[serde(default = "default_max_connections")]
    pub max_connections: usize,
    #[serde(default = "default_max_conns_per_ip")]
    pub max_conns_per_ip: usize,
    #[serde(default = "default_request_body_limit")]
    pub request_body_limit: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LoggingConfig {
    #[serde(default = "default_log_level")]
    pub level: String,
    pub access_log: Option<PathBuf>,
}

// Defaults
fn default_listen() -> Vec<String> {
    vec!["0.0.0.0:443".into(), "0.0.0.0:80".into()]
}
fn default_max_conns_per_worker() -> usize { 16 }
fn default_keepalive_timeout() -> std::time::Duration { std::time::Duration::from_secs(30) }
fn default_connect_timeout() -> std::time::Duration { std::time::Duration::from_secs(5) }
fn default_response_timeout() -> std::time::Duration { std::time::Duration::from_secs(30) }
fn default_max_connections() -> usize { 100_000 }
fn default_max_conns_per_ip() -> usize { 50 }
fn default_request_body_limit() -> usize { 25 * 1024 * 1024 } // 25MB
fn default_log_level() -> String { "info".into() }

impl Default for LimitsConfig {
    fn default() -> Self {
        Self {
            max_connections: default_max_connections(),
            max_conns_per_ip: default_max_conns_per_ip(),
            request_body_limit: default_request_body_limit(),
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
