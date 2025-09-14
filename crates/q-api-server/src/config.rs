use q_types::NodeId;
use serde::{Deserialize, Serialize};
use std::env;
use std::path::PathBuf;
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub port: u16,
    pub host: String,
    pub is_validator: bool,
    pub p2p_port: u16,
    pub bootstrap_peers: Vec<String>,
    pub database_url: Option<String>,
    pub db_path: Option<String>,
    pub hot_db_path: Option<String>,
    pub log_level: String,
    pub enable_metrics: bool,
    /// Node ID (optional, will be generated if not provided)
    pub node_id: Option<NodeId>,
    /// Tor configuration
    pub tor: TorConfig,
}

/// Tor-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TorConfig {
    /// Enable Tor networking
    pub enabled: bool,
    /// Number of dedicated circuits
    pub circuit_count: usize,
    /// Tor onion service port
    pub onion_port: u16,
    /// Tor data directory
    pub data_dir: Option<PathBuf>,
    /// Tor-only mode (no fallback to direct connections)
    pub tor_only: bool,
    /// Enable Dandelion++ for traffic analysis resistance
    pub enable_dandelion: bool,
    /// Latency target in milliseconds
    pub latency_target_ms: u16,
    /// Bootstrap onion addresses
    pub bootstrap_onions: Vec<String>,
    /// SOCKS5 proxy address
    pub socks5_addr: Option<String>,
}

impl Default for TorConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            circuit_count: 4,
            onion_port: 4001,
            data_dir: Some(PathBuf::from("/var/lib/qnk/tor")),
            tor_only: false,
            enable_dandelion: true,
            latency_target_ms: 300,
            bootstrap_onions: vec!["bootstrap.qnk.onion:4001".to_string()],
            socks5_addr: Some("127.0.0.1:9050".to_string()),
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            port: 8080,
            host: "0.0.0.0".to_string(),
            is_validator: false,
            p2p_port: 8081,
            bootstrap_peers: vec![],
            database_url: None,
            db_path: None,
            hot_db_path: None,
            log_level: "info".to_string(),
            enable_metrics: true,
            node_id: None,
            tor: TorConfig::default(),
        }
    }
}

impl Config {
    pub fn from_env() -> anyhow::Result<Self> {
        let mut config = Self::default();

        if let Ok(port) = env::var("Q_API_PORT") {
            config.port = port.parse()?;
        }

        if let Ok(host) = env::var("Q_API_HOST") {
            config.host = host;
        }

        if let Ok(is_validator) = env::var("Q_IS_VALIDATOR") {
            config.is_validator = is_validator.parse().unwrap_or(false);
        }

        if let Ok(p2p_port) = env::var("Q_P2P_PORT") {
            config.p2p_port = p2p_port.parse()?;
        }

        if let Ok(bootstrap_peers) = env::var("Q_BOOTSTRAP_PEERS") {
            config.bootstrap_peers = bootstrap_peers
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
        }

        if let Ok(database_url) = env::var("DATABASE_URL") {
            config.database_url = Some(database_url);
        }

        if let Ok(db_path) = env::var("Q_DB_PATH") {
            config.db_path = Some(db_path);
        }

        if let Ok(hot_db_path) = env::var("Q_HOT_DB_PATH") {
            config.hot_db_path = Some(hot_db_path);
        }

        if let Ok(log_level) = env::var("Q_LOG_LEVEL") {
            config.log_level = log_level;
        }

        if let Ok(enable_metrics) = env::var("Q_ENABLE_METRICS") {
            config.enable_metrics = enable_metrics.parse().unwrap_or(true);
        }

        // Tor configuration
        if let Ok(tor_enabled) = env::var("Q_TOR_ENABLED") {
            config.tor.enabled = tor_enabled.parse().unwrap_or(false);
        }

        if let Ok(tor_circuit_count) = env::var("Q_TOR_CIRCUIT_COUNT") {
            config.tor.circuit_count = tor_circuit_count.parse().unwrap_or(4);
        }

        if let Ok(tor_onion_port) = env::var("Q_TOR_ONION_PORT") {
            config.tor.onion_port = tor_onion_port.parse().unwrap_or(4001);
        }

        if let Ok(tor_data_dir) = env::var("Q_TOR_DATA_DIR") {
            config.tor.data_dir = Some(PathBuf::from(tor_data_dir));
        }

        if let Ok(tor_only) = env::var("Q_TOR_ONLY") {
            config.tor.tor_only = tor_only.parse().unwrap_or(false);
        }

        if let Ok(tor_dandelion) = env::var("Q_TOR_DANDELION") {
            config.tor.enable_dandelion = tor_dandelion.parse().unwrap_or(true);
        }

        if let Ok(tor_latency) = env::var("Q_TOR_LATENCY_TARGET_MS") {
            config.tor.latency_target_ms = tor_latency.parse().unwrap_or(300);
        }

        if let Ok(tor_bootstrap) = env::var("Q_TOR_BOOTSTRAP_ONIONS") {
            config.tor.bootstrap_onions = tor_bootstrap
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
        }

        if let Ok(tor_socks5) = env::var("Q_TOR_SOCKS5_ADDR") {
            config.tor.socks5_addr = Some(tor_socks5);
        }

        Ok(config)
    }
}
