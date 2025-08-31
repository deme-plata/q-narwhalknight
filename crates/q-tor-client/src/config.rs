use serde::{Deserialize, Serialize};
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
    
    /// SOCKS5 proxy address for client connections
    pub socks5_addr: Option<String>,
    
    /// Bootstrap nodes as onion addresses
    pub bootstrap_onions: Vec<String>,
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
            socks5_addr: Some("127.0.0.1:9050".to_string()),
            bootstrap_onions: vec![
                "bootstrap.qnk.onion:4001".to_string(),
            ],
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

    /// Validate configuration
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.enabled {
            if self.circuit_count == 0 {
                anyhow::bail!("Circuit count must be > 0 when Tor is enabled");
            }
            
            if self.circuit_count > 10 {
                anyhow::bail!("Circuit count should not exceed 10 for performance reasons");
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