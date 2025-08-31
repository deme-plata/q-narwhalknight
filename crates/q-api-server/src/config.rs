use serde::{Deserialize, Serialize};
use std::env;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub port: u16,
    pub host: String,
    pub is_validator: bool,
    pub p2p_port: u16,
    pub bootstrap_peers: Vec<String>,
    pub database_url: Option<String>,
    pub log_level: String,
    pub enable_metrics: bool,
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
            log_level: "info".to_string(),
            enable_metrics: true,
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

        if let Ok(log_level) = env::var("Q_LOG_LEVEL") {
            config.log_level = log_level;
        }

        if let Ok(enable_metrics) = env::var("Q_ENABLE_METRICS") {
            config.enable_metrics = enable_metrics.parse().unwrap_or(true);
        }

        Ok(config)
    }
}