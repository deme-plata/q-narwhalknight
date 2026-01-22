//! Pool Node Discovery via Kademlia DHT
//!
//! Enables miners to discover nearby pool nodes and
//! pool nodes to find each other for state synchronization.

use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::time::{Duration, Instant};

use super::PeerIdBytes;

/// Information about a pool node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolNodeInfo {
    /// Peer ID
    pub peer_id: PeerIdBytes,

    /// Stratum server port
    pub stratum_port: u16,

    /// libp2p multiaddresses
    pub multiaddrs: Vec<String>,

    /// Current worker count
    pub worker_count: u32,

    /// Current total hashrate (H/s)
    pub hashrate: f64,

    /// Uptime in seconds
    pub uptime_seconds: u64,

    /// Geographic region (for latency optimization)
    pub region: String,

    /// Version string
    pub version: String,

    /// Last seen timestamp (unix seconds)
    pub last_seen: u64,

    /// PPLNS state hash (for sync comparison)
    pub pplns_state_hash: [u8; 32],

    /// Is this node accepting new connections?
    pub accepting_connections: bool,

    /// Node's signature over this info
    #[serde(with = "BigArray")]
    pub signature: [u8; 64],
}

impl PoolNodeInfo {
    /// Create new pool node info
    pub fn new(
        peer_id: PeerIdBytes,
        stratum_port: u16,
        multiaddrs: Vec<String>,
        region: String,
    ) -> Self {
        Self {
            peer_id,
            stratum_port,
            multiaddrs,
            worker_count: 0,
            hashrate: 0.0,
            uptime_seconds: 0,
            region,
            version: env!("CARGO_PKG_VERSION").to_string(),
            last_seen: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            pplns_state_hash: [0u8; 32],
            accepting_connections: true,
            signature: [0u8; 64],
        }
    }

    /// Update dynamic fields
    pub fn update(&mut self, worker_count: u32, hashrate: f64, uptime_seconds: u64) {
        self.worker_count = worker_count;
        self.hashrate = hashrate;
        self.uptime_seconds = uptime_seconds;
        self.last_seen = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
    }

    /// Get Stratum connection address
    pub fn stratum_addr(&self) -> Option<SocketAddr> {
        // Try to extract IP from multiaddr
        for addr in &self.multiaddrs {
            if let Some(ip) = extract_ip_from_multiaddr(addr) {
                return Some(SocketAddr::new(ip, self.stratum_port));
            }
        }
        None
    }

    /// Check if node info is stale (not seen in 5 minutes)
    pub fn is_stale(&self) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        now - self.last_seen > 300
    }

    /// Serialize to bytes for DHT storage
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).unwrap_or_default()
    }

    /// Deserialize from bytes
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        bincode::deserialize(data).ok()
    }
}

/// Extract IP address from multiaddr string
fn extract_ip_from_multiaddr(addr: &str) -> Option<std::net::IpAddr> {
    // Parse formats like /ip4/192.168.1.1/tcp/9001
    let parts: Vec<&str> = addr.split('/').collect();
    for (i, part) in parts.iter().enumerate() {
        if *part == "ip4" || *part == "ip6" {
            if let Some(ip_str) = parts.get(i + 1) {
                return ip_str.parse().ok();
            }
        }
    }
    None
}

/// Pool node discovery manager
pub struct PoolNodeDiscovery {
    /// Known pool nodes
    known_nodes: HashMap<PeerIdBytes, PoolNodeInfo>,

    /// Our own node info
    our_info: Option<PoolNodeInfo>,

    /// Network ID
    network_id: String,

    /// DHT key prefix for pool nodes
    dht_prefix: String,

    /// Discovery interval
    discovery_interval: Duration,

    /// Last discovery run
    last_discovery: Option<Instant>,
}

impl PoolNodeDiscovery {
    /// Create new discovery manager
    pub fn new(network_id: &str) -> Self {
        Self {
            known_nodes: HashMap::new(),
            our_info: None,
            network_id: network_id.to_string(),
            dht_prefix: format!("/qnk/{}/pool/nodes/", network_id),
            discovery_interval: Duration::from_secs(60),
            last_discovery: None,
        }
    }

    /// Set our own node info
    pub fn set_our_info(&mut self, info: PoolNodeInfo) {
        self.our_info = Some(info);
    }

    /// Get DHT key for a pool node
    pub fn dht_key(&self, peer_id: &PeerIdBytes) -> Vec<u8> {
        format!("{}{}", self.dht_prefix, hex::encode(peer_id)).into_bytes()
    }

    /// Get DHT key for node list
    pub fn node_list_key(&self) -> Vec<u8> {
        format!("{}list", self.dht_prefix).into_bytes()
    }

    /// Add or update a known node
    pub fn add_node(&mut self, info: PoolNodeInfo) {
        if !info.is_stale() {
            self.known_nodes.insert(info.peer_id, info);
        }
    }

    /// Remove stale nodes
    pub fn remove_stale_nodes(&mut self) {
        self.known_nodes.retain(|_, info| !info.is_stale());
    }

    /// Get all known nodes
    pub fn get_all_nodes(&self) -> Vec<&PoolNodeInfo> {
        self.known_nodes.values().collect()
    }

    /// Get nodes sorted by estimated latency (based on region)
    pub fn get_nodes_by_region(&self, preferred_region: &str) -> Vec<&PoolNodeInfo> {
        let mut nodes: Vec<_> = self.known_nodes.values().collect();

        // Sort by region match, then by worker count (load balancing)
        nodes.sort_by(|a, b| {
            let a_region_match = a.region == preferred_region;
            let b_region_match = b.region == preferred_region;

            match (a_region_match, b_region_match) {
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                _ => a.worker_count.cmp(&b.worker_count), // Prefer less loaded
            }
        });

        nodes
    }

    /// Get nodes accepting connections, sorted by load
    pub fn get_available_nodes(&self) -> Vec<&PoolNodeInfo> {
        let mut nodes: Vec<_> = self
            .known_nodes
            .values()
            .filter(|n| n.accepting_connections && !n.is_stale())
            .collect();

        nodes.sort_by(|a, b| a.worker_count.cmp(&b.worker_count));
        nodes
    }

    /// Get node by peer ID
    pub fn get_node(&self, peer_id: &PeerIdBytes) -> Option<&PoolNodeInfo> {
        self.known_nodes.get(peer_id)
    }

    /// Get node count
    pub fn node_count(&self) -> usize {
        self.known_nodes.len()
    }

    /// Check if we should run discovery
    pub fn should_discover(&self) -> bool {
        match self.last_discovery {
            Some(last) => last.elapsed() >= self.discovery_interval,
            None => true,
        }
    }

    /// Mark discovery as complete
    pub fn discovery_complete(&mut self) {
        self.last_discovery = Some(Instant::now());
    }

    /// Get bootstrap nodes for initial connection
    pub fn bootstrap_nodes(&self) -> Vec<String> {
        // Return multiaddrs of known healthy nodes
        self.known_nodes
            .values()
            .filter(|n| !n.is_stale() && n.accepting_connections)
            .take(5)
            .flat_map(|n| n.multiaddrs.clone())
            .collect()
    }
}

/// Miner-side pool node selector
pub struct PoolNodeSelector {
    /// Available nodes
    nodes: Vec<PoolNodeInfo>,

    /// Currently connected node
    current_node: Option<PeerIdBytes>,

    /// Backup nodes (in case current fails)
    backup_nodes: Vec<PeerIdBytes>,

    /// Connection attempt counts
    attempt_counts: HashMap<PeerIdBytes, u32>,

    /// Max attempts before blacklisting
    max_attempts: u32,
}

impl PoolNodeSelector {
    /// Create new selector
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            current_node: None,
            backup_nodes: Vec::new(),
            attempt_counts: HashMap::new(),
            max_attempts: 3,
        }
    }

    /// Update available nodes
    pub fn update_nodes(&mut self, nodes: Vec<PoolNodeInfo>) {
        self.nodes = nodes
            .into_iter()
            .filter(|n| {
                let attempts = self.attempt_counts.get(&n.peer_id).copied().unwrap_or(0);
                attempts < self.max_attempts
            })
            .collect();

        // Sort by worker count (load balancing)
        self.nodes.sort_by(|a, b| a.worker_count.cmp(&b.worker_count));

        // Update backups
        self.backup_nodes = self
            .nodes
            .iter()
            .filter(|n| Some(n.peer_id) != self.current_node)
            .take(3)
            .map(|n| n.peer_id)
            .collect();
    }

    /// Select best node for connection
    pub fn select_node(&mut self) -> Option<&PoolNodeInfo> {
        // If we have a current node that's healthy, keep it
        if let Some(current) = self.current_node {
            if let Some(node) = self.nodes.iter().find(|n| n.peer_id == current) {
                if !node.is_stale() && node.accepting_connections {
                    return Some(node);
                }
            }
        }

        // Select least loaded available node
        let node = self
            .nodes
            .iter()
            .find(|n| n.accepting_connections && !n.is_stale())?;

        self.current_node = Some(node.peer_id);
        Some(node)
    }

    /// Report connection failure
    pub fn report_failure(&mut self, peer_id: &PeerIdBytes) {
        *self.attempt_counts.entry(*peer_id).or_insert(0) += 1;

        if Some(*peer_id) == self.current_node {
            self.current_node = None;
        }
    }

    /// Report connection success
    pub fn report_success(&mut self, peer_id: &PeerIdBytes) {
        self.attempt_counts.remove(peer_id);
        self.current_node = Some(*peer_id);
    }

    /// Get backup node for failover
    pub fn get_backup(&self) -> Option<&PoolNodeInfo> {
        for backup_id in &self.backup_nodes {
            if let Some(node) = self.nodes.iter().find(|n| &n.peer_id == backup_id) {
                if !node.is_stale() && node.accepting_connections {
                    return Some(node);
                }
            }
        }
        None
    }
}

impl Default for PoolNodeSelector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_node_info() {
        let peer_id = [1u8; 32];
        let info = PoolNodeInfo::new(
            peer_id,
            3333,
            vec!["/ip4/192.168.1.1/tcp/9001".to_string()],
            "us-east".to_string(),
        );

        assert_eq!(info.stratum_port, 3333);
        assert!(!info.is_stale());

        let addr = info.stratum_addr().unwrap();
        assert_eq!(addr.port(), 3333);
    }

    #[test]
    fn test_discovery_manager() {
        let mut discovery = PoolNodeDiscovery::new("testnet");

        let node1 = PoolNodeInfo::new(
            [1u8; 32],
            3333,
            vec!["/ip4/10.0.0.1/tcp/9001".to_string()],
            "us-east".to_string(),
        );

        let node2 = PoolNodeInfo::new(
            [2u8; 32],
            3334,
            vec!["/ip4/10.0.0.2/tcp/9001".to_string()],
            "eu-west".to_string(),
        );

        discovery.add_node(node1);
        discovery.add_node(node2);

        assert_eq!(discovery.node_count(), 2);

        let by_region = discovery.get_nodes_by_region("us-east");
        assert_eq!(by_region[0].region, "us-east");
    }

    #[test]
    fn test_node_selector() {
        let mut selector = PoolNodeSelector::new();

        let mut node1 = PoolNodeInfo::new(
            [1u8; 32],
            3333,
            vec![],
            "us-east".to_string(),
        );
        node1.worker_count = 100;

        let mut node2 = PoolNodeInfo::new(
            [2u8; 32],
            3334,
            vec![],
            "us-east".to_string(),
        );
        node2.worker_count = 50;

        selector.update_nodes(vec![node1, node2]);

        // Should select less loaded node
        let selected = selector.select_node().unwrap();
        assert_eq!(selected.worker_count, 50);
    }
}
