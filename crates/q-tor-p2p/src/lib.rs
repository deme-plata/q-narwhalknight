//! Q-NarwhalKnight Tor-based P2P Network
//! Real-world peer discovery and connection through Tor

pub mod tor_client;
pub mod dht_discovery;
pub mod p2p_node;
pub mod onion_service;
pub mod consensus_transport;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::time::Duration;

/// Configuration for Tor P2P network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TorP2PConfig {
    /// Local SOCKS proxy port for Tor
    pub socks_port: u16,
    
    /// Control port for Tor daemon
    pub control_port: u16,
    
    /// Directory for Tor data
    pub data_dir: String,
    
    /// Bootstrap nodes (onion addresses)
    pub bootstrap_nodes: Vec<String>,
    
    /// Enable DHT-based discovery
    pub enable_dht: bool,
    
    /// Hidden service configuration
    pub hidden_service: HiddenServiceConfig,
    
    /// Network timeouts
    pub connection_timeout: Duration,
    pub request_timeout: Duration,
}

impl Default for TorP2PConfig {
    fn default() -> Self {
        Self {
            socks_port: 9050,
            control_port: 9051,
            data_dir: "/tmp/q-tor-data".to_string(),
            bootstrap_nodes: vec![
                // Real Tor v3 onion addresses would go here
                "qnkbootstrap3jklmnopqrstuvwxyzabcdefghijklmnopqrstuvwx.onion:8080".to_string(),
            ],
            enable_dht: true,
            hidden_service: HiddenServiceConfig::default(),
            connection_timeout: Duration::from_secs(30),
            request_timeout: Duration::from_secs(10),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiddenServiceConfig {
    pub enabled: bool,
    pub port: u16,
    pub private_key_path: Option<String>,
}

impl Default for HiddenServiceConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            port: 8080,
            private_key_path: None,
        }
    }
}

/// Peer information for DHT
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerInfo {
    pub peer_id: String,
    pub onion_address: String,
    pub port: u16,
    pub capabilities: Vec<String>,
    pub last_seen: u64,
    pub reputation: f64,
}

/// Network statistics
#[derive(Debug, Clone)]
pub struct NetworkStats {
    pub connected_peers: usize,
    pub total_discovered: usize,
    pub messages_sent: u64,
    pub messages_received: u64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub avg_latency_ms: f64,
}