use serde::Deserialize;
use std::net::SocketAddr;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub server:  ServerConfig,
    pub auth:    AuthConfig,
    pub relay:   RelayConfig,
    pub limits:  LimitsConfig,
    #[serde(default)]
    pub logging: LoggingConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ServerConfig {
    /// UDP + TCP listen address (e.g. "0.0.0.0:3478")
    pub bind: SocketAddr,
    /// STUN/TURN realm (sent to clients during 401 challenge)
    pub realm: String,
    /// Returned in SOFTWARE attribute
    #[serde(default = "default_software")]
    pub software: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AuthConfig {
    /// Shared secret — same value must be set in q-api-server for credential generation
    pub secret: String,
    /// Credential time-to-live in seconds (default 600)
    #[serde(default = "default_credential_ttl")]
    pub credential_ttl: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RelayConfig {
    /// First UDP port for relay sockets
    #[serde(default = "default_min_port")]
    pub min_port: u16,
    /// Last UDP port for relay sockets
    #[serde(default = "default_max_port")]
    pub max_port: u16,
    /// IP to bind relay sockets on (must be externally reachable)
    pub public_ip: std::net::IpAddr,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LimitsConfig {
    /// Maximum simultaneous TURN allocations
    #[serde(default = "default_max_allocs")]
    pub max_allocations: usize,
    /// Maximum allocation lifetime in seconds
    #[serde(default = "default_alloc_lifetime")]
    pub allocation_lifetime: u64,
    /// Maximum channels per allocation
    #[serde(default = "default_max_channels")]
    pub max_channels: usize,
    /// Maximum permissions per allocation
    #[serde(default = "default_max_perms")]
    pub max_permissions: usize,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct LoggingConfig {
    #[serde(default = "default_log_level")]
    pub level: String,
}

fn default_software() -> String  { "q-turn/1.0 (quillon.xyz)".into() }
fn default_credential_ttl() -> u64 { 600 }
fn default_min_port() -> u16    { 49152 }
fn default_max_port() -> u16    { 65535 }
fn default_max_allocs() -> usize { 1000 }
fn default_alloc_lifetime() -> u64 { 600 }
fn default_max_channels() -> usize { 16 }
fn default_max_perms() -> usize { 32 }
fn default_log_level() -> String { "info".into() }

impl Config {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        Ok(toml::from_str(&text)?)
    }

    pub fn default_for_testing(secret: &str, public_ip: std::net::IpAddr) -> Self {
        Self {
            server: ServerConfig {
                bind: "0.0.0.0:3478".parse().unwrap(),
                realm: "quillon.xyz".into(),
                software: default_software(),
            },
            auth: AuthConfig {
                secret: secret.into(),
                credential_ttl: default_credential_ttl(),
            },
            relay: RelayConfig {
                min_port: default_min_port(),
                max_port: default_max_port(),
                public_ip,
            },
            limits: LimitsConfig {
                max_allocations: default_max_allocs(),
                allocation_lifetime: default_alloc_lifetime(),
                max_channels: default_max_channels(),
                max_permissions: default_max_perms(),
            },
            logging: LoggingConfig::default(),
        }
    }
}
