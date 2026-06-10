use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

/// Configuration for Tor client
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TorConfig {
    /// Enable Tor networking
    pub enabled: bool,

    /// Number of dedicated circuits to maintain
    pub circuit_count: usize,

    /// RPC port for onion service
    pub rpc_port: u16,

    /// Data directory for Tor state
    pub data_dir: Option<PathBuf>,

    /// Onion key path
    pub onion_key_path: Option<PathBuf>,

    /// Bandwidth burst limit
    pub bandwidth_burst: String,

    /// Enable Dandelion++ for traffic analysis resistance
    pub enable_dandelion: bool,

    /// Latency target for adaptive QoS (milliseconds)
    pub latency_target_ms: Option<u16>,

    /// Tor-only mode (no fallback to direct connections)
    pub tor_only: bool,

    /// SOCKS proxy address (for Tor connection)
    pub socks_proxy_addr: Option<SocketAddr>,

    /// Bootstrap nodes as onion addresses
    pub bootstrap_onions: Vec<String>,

    /// Enable Prometheus metrics collection
    pub enable_prometheus_metrics: bool,

    /// Use embedded Arti client instead of external Tor daemon
    pub use_embedded_arti: bool,

    /// Cache directory for embedded Arti client
    pub cache_dir: Option<PathBuf>,
}

impl Default for TorConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            circuit_count: 4, // 1 control + 3 gossip
            rpc_port: 4001,
            data_dir: Some(PathBuf::from("/var/lib/qnk/tor")),
            onion_key_path: Some(PathBuf::from("/var/lib/qnk/hs_ed25519")),
            bandwidth_burst: "10 MB/s".to_string(),
            enable_dandelion: true,
            latency_target_ms: Some(300), // 300ms target
            tor_only: false,
            socks_proxy_addr: Some("127.0.0.1:9050".parse().unwrap()),
            bootstrap_onions: vec!["bootstrap.qnk.onion:4001".to_string()],
            enable_prometheus_metrics: true,
            use_embedded_arti: false,
            cache_dir: Some(PathBuf::from("/var/lib/qnk/tor_cache")),
        }
    }
}

impl TorConfig {
    /// Create configuration for stealth mode (Tor-only)
    pub fn stealth_mode() -> Self {
        Self {
            enabled: true,
            tor_only: true,
            enable_dandelion: true,
            latency_target_ms: Some(200), // More aggressive latency target
            ..Default::default()
        }
    }

    /// Create configuration for hybrid mode (Tor + direct fallback)
    pub fn hybrid_mode() -> Self {
        Self {
            enabled: true,
            tor_only: false,
            enable_dandelion: true,
            latency_target_ms: Some(300),
            ..Default::default()
        }
    }

    /// Create configuration for embedded Arti mode (no external Tor daemon needed)
    pub fn embedded_arti_mode() -> Self {
        Self {
            enabled: true,
            use_embedded_arti: true,
            tor_only: false,
            enable_dandelion: true,
            latency_target_ms: Some(300),
            socks_proxy_addr: None, // Not needed with embedded Arti
            data_dir: Some(PathBuf::from("/tmp/qnk_tor")),
            cache_dir: Some(PathBuf::from("/tmp/qnk_tor_cache")),
            ..Default::default()
        }
    }

    /// Create configuration for mandatory Tor mode with dedicated circuits (Arti 1.8.0)
    ///
    /// This is the recommended production configuration that:
    /// - Uses embedded Arti client (no external Tor daemon needed)
    /// - Enables mandatory Tor (no clearnet fallback allowed)
    /// - Uses dedicated circuits per operation type for traffic analysis resistance
    /// - Enables Dandelion++ for additional privacy
    /// - Uses 8 circuits (one per operation type)
    pub fn mandatory_dedicated_circuits() -> Self {
        Self {
            enabled: true,
            use_embedded_arti: true,
            tor_only: true, // NO CLEARNET FALLBACK - All traffic MUST go through Tor
            enable_dandelion: true,
            latency_target_ms: Some(300),
            circuit_count: 8, // One per operation type (see OperationType enum)
            socks_proxy_addr: None, // Not needed with embedded Arti
            data_dir: Some(PathBuf::from("/var/lib/qnk/tor")),
            cache_dir: Some(PathBuf::from("/var/cache/qnk/tor")),
            enable_prometheus_metrics: true,
            ..Default::default()
        }
    }

    /// Check if this configuration enforces mandatory Tor
    pub fn is_mandatory_tor(&self) -> bool {
        self.enabled && self.tor_only && self.use_embedded_arti
    }

    /// Convert to DedicatedCircuitConfig for the new circuit manager
    pub fn to_dedicated_circuit_config(&self) -> super::dedicated_circuits::DedicatedCircuitConfig {
        super::dedicated_circuits::DedicatedCircuitConfig {
            data_directory: self.data_dir
                .as_ref()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|| "/var/lib/qnk/tor".to_string()),
            cache_directory: self.cache_dir
                .as_ref()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|| "/var/cache/qnk/tor".to_string()),
            tor_mandatory: self.tor_only,
            // v8.6.0: reduced from 120s to 45s — faster startup; if Tor can't bootstrap
            // in 45s the network is likely unreachable and retrying is more productive
            bootstrap_timeout: Duration::from_secs(45),
            prewarm_circuits: true,
            log_level: "info".to_string(),
            adaptive_rotation: true,
            min_rotation_interval: Duration::from_secs(60),
            max_rotation_interval: Duration::from_secs(3600),
            auto_enforce_diversity: true,
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.enabled {
            if self.circuit_count == 0 {
                anyhow::bail!("Circuit count must be > 0 when Tor is enabled");
            }

            // v8.6.0: raised from 10 to 12 — mandatory_dedicated_circuits uses 8,
            // allow headroom for future operation types without hitting the limit
            if self.circuit_count > 12 {
                anyhow::bail!("Circuit count should not exceed 12 for performance reasons");
            }

            if let Some(target) = self.latency_target_ms {
                if target < 50 {
                    anyhow::bail!("Latency target too low (min 50ms)");
                }
                if target > 5000 {
                    anyhow::bail!("Latency target too high (max 5000ms)");
                }
            }
        }

        Ok(())
    }

    /// Get expected latency range based on configuration
    pub fn expected_latency_range(&self) -> (Duration, Duration) {
        if !self.enabled {
            return (Duration::from_millis(10), Duration::from_millis(50));
        }

        let base_latency = if self.tor_only { 200 } else { 150 };
        let target = self.latency_target_ms.unwrap_or(300);

        (
            Duration::from_millis(base_latency),
            Duration::from_millis(target as u64),
        )
    }
}
