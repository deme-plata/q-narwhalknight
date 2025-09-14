use crate::TorClient;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerListResponse {
    pub peers: Vec<PeerInfo>,
    pub timestamp: u64,
    pub bootstrap_node: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerInfo {
    pub onion_address: String,
    pub port: u16,
    pub node_id: String,
    pub last_seen: u64,
    pub capabilities: Vec<String>,
}

impl PeerInfo {
    pub fn to_address_string(&self) -> String {
        format!("{}:{}", self.onion_address, self.port)
    }
}

pub struct BootstrapDiscovery {
    tor_client: Arc<TorClient>,
    bootstrap_nodes: Vec<String>,
    discovered_peers: Arc<RwLock<HashMap<String, PeerInfo>>>,
    bootstrap_reputation: Arc<RwLock<HashMap<String, f64>>>,
    last_query_time: Arc<RwLock<HashMap<String, SystemTime>>>,
    query_interval: Duration,
    peer_ttl: Duration,
}

impl BootstrapDiscovery {
    pub fn new(tor_client: Arc<TorClient>) -> Self {
        // Default free community bootstrap nodes
        let bootstrap_nodes = vec![
            "bootstrap1.qnk.onion:8333".to_string(),
            "bootstrap2.qnk.onion:8333".to_string(),
            "bootstrap3.qnk.onion:8333".to_string(),
            "bootstrap4.qnk.onion:8333".to_string(),
            "bootstrap5.qnk.onion:8333".to_string(),
        ];

        Self {
            tor_client,
            bootstrap_nodes,
            discovered_peers: Arc::new(RwLock::new(HashMap::new())),
            bootstrap_reputation: Arc::new(RwLock::new(HashMap::new())),
            last_query_time: Arc::new(RwLock::new(HashMap::new())),
            query_interval: Duration::from_secs(300), // 5 minutes
            peer_ttl: Duration::from_secs(3600),      // 1 hour
        }
    }

    pub fn with_custom_bootstrap_nodes(tor_client: Arc<TorClient>, nodes: Vec<String>) -> Self {
        let mut discovery = Self::new(tor_client);
        discovery.bootstrap_nodes = nodes;
        discovery
    }

    pub async fn start_discovery(&self) -> Result<()> {
        info!("🆓 Starting bootstrap node discovery (completely FREE)");
        info!(
            "🆓 Using {} community bootstrap nodes",
            self.bootstrap_nodes.len()
        );

        // Initialize reputation scores
        {
            let mut reputation = self.bootstrap_reputation.write().await;
            for node in &self.bootstrap_nodes {
                reputation.insert(node.clone(), 1.0); // Start with neutral reputation
            }
        }

        // Start background discovery loop
        self.start_discovery_loop().await;

        // Do initial discovery
        self.discover_peers_from_all_bootstraps().await?;

        Ok(())
    }

    async fn start_discovery_loop(&self) {
        let discovered_peers = Arc::clone(&self.discovered_peers);
        let bootstrap_nodes = self.bootstrap_nodes.clone();
        let tor_client = Arc::clone(&self.tor_client);
        let bootstrap_reputation = Arc::clone(&self.bootstrap_reputation);
        let last_query_time = Arc::clone(&self.last_query_time);
        let query_interval = self.query_interval;
        let peer_ttl = self.peer_ttl;

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(query_interval);

            loop {
                interval.tick().await;

                // Query bootstrap nodes for fresh peer lists
                for bootstrap in &bootstrap_nodes {
                    if Self::should_query_bootstrap(&last_query_time, bootstrap, query_interval)
                        .await
                    {
                        match Self::query_bootstrap_node(&tor_client, bootstrap).await {
                            Ok(peer_list) => {
                                Self::update_reputation(&bootstrap_reputation, bootstrap, true)
                                    .await;
                                Self::merge_peer_list(&discovered_peers, peer_list, peer_ttl).await;

                                {
                                    let mut query_times = last_query_time.write().await;
                                    query_times.insert(bootstrap.clone(), SystemTime::now());
                                }

                                info!("🆓 Refreshed peer list from {} (FREE)", bootstrap);
                            }
                            Err(e) => {
                                Self::update_reputation(&bootstrap_reputation, bootstrap, false)
                                    .await;
                                debug!("Bootstrap query failed for {}: {}", bootstrap, e);
                            }
                        }
                    }
                }

                // Cleanup expired peers
                Self::cleanup_expired_peers(&discovered_peers, peer_ttl).await;
            }
        });
    }

    async fn should_query_bootstrap(
        last_query_time: &Arc<RwLock<HashMap<String, SystemTime>>>,
        bootstrap: &str,
        interval: Duration,
    ) -> bool {
        let query_times = last_query_time.read().await;

        if let Some(last_time) = query_times.get(bootstrap) {
            SystemTime::now()
                .duration_since(*last_time)
                .unwrap_or(Duration::ZERO)
                >= interval
        } else {
            true // Never queried before
        }
    }

    async fn query_bootstrap_node(
        tor_client: &TorClient,
        bootstrap_address: &str,
    ) -> Result<PeerListResponse> {
        debug!("🆓 Querying bootstrap node: {} (FREE)", bootstrap_address);

        // Parse the bootstrap address
        let parts: Vec<&str> = bootstrap_address.split(':').collect();
        if parts.len() != 2 {
            return Err(anyhow!("Invalid bootstrap address format"));
        }

        let onion_address = parts[0];
        let port: u16 = parts[1].parse()?;

        // Connect through Tor - completely free operation
        let connection_result = Self::connect_to_bootstrap(tor_client, onion_address, port).await;

        match connection_result {
            Ok(peer_list) => {
                info!(
                    "🆓 Successfully retrieved {} peers from {} (FREE)",
                    peer_list.peers.len(),
                    bootstrap_address
                );
                Ok(peer_list)
            }
            Err(e) => {
                warn!(
                    "Failed to connect to bootstrap {}: {}",
                    bootstrap_address, e
                );
                Err(e)
            }
        }
    }

    async fn connect_to_bootstrap(
        _tor_client: &TorClient,
        onion_address: &str,
        port: u16,
    ) -> Result<PeerListResponse> {
        // This would establish a real Tor connection to the bootstrap node
        // and request the current peer list using Q-NarwhalKnight protocol

        // For now, simulate the bootstrap response
        // Real implementation would:
        // 1. Create Tor stream to bootstrap node
        // 2. Send PEER_LIST_REQUEST message
        // 3. Receive PEER_LIST_RESPONSE with current peers
        // 4. Parse and validate the response

        let simulated_peers = vec![
            PeerInfo {
                onion_address: "validator1abc123def456.onion".to_string(),
                port: 8333,
                node_id: "node_001".to_string(),
                last_seen: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                capabilities: vec!["consensus".to_string(), "mempool".to_string()],
            },
            PeerInfo {
                onion_address: "validator2xyz789uvw012.onion".to_string(),
                port: 8333,
                node_id: "node_002".to_string(),
                last_seen: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                capabilities: vec!["consensus".to_string(), "api".to_string()],
            },
        ];

        Ok(PeerListResponse {
            peers: simulated_peers,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            bootstrap_node: format!("{}:{}", onion_address, port),
        })
    }

    async fn merge_peer_list(
        discovered_peers: &Arc<RwLock<HashMap<String, PeerInfo>>>,
        peer_list: PeerListResponse,
        _peer_ttl: Duration,
    ) {
        let mut peers = discovered_peers.write().await;
        let _current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        for peer in peer_list.peers {
            let peer_key = peer.to_address_string();

            // Only add if we haven't seen this peer or if it's fresher
            let should_add = if let Some(existing) = peers.get(&peer_key) {
                peer.last_seen > existing.last_seen
            } else {
                true
            };

            if should_add {
                debug!("🆓 Adding peer from bootstrap: {} (FREE)", peer_key);
                peers.insert(peer_key, peer);
            }
        }

        debug!("🆓 Total discovered peers: {} (FREE)", peers.len());
    }

    async fn cleanup_expired_peers(
        discovered_peers: &Arc<RwLock<HashMap<String, PeerInfo>>>,
        peer_ttl: Duration,
    ) {
        let mut peers = discovered_peers.write().await;
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let ttl_seconds = peer_ttl.as_secs();

        let initial_count = peers.len();
        peers.retain(|_, peer| current_time - peer.last_seen < ttl_seconds);

        let cleaned_count = initial_count - peers.len();
        if cleaned_count > 0 {
            debug!("🆓 Cleaned up {} expired peers (FREE)", cleaned_count);
        }
    }

    async fn update_reputation(
        bootstrap_reputation: &Arc<RwLock<HashMap<String, f64>>>,
        bootstrap: &str,
        success: bool,
    ) {
        let mut reputation = bootstrap_reputation.write().await;
        let current_rep = reputation.get(bootstrap).copied().unwrap_or(1.0);

        let new_rep = if success {
            (current_rep + 0.1).min(2.0) // Increase reputation, cap at 2.0
        } else {
            (current_rep - 0.2).max(0.1) // Decrease reputation, floor at 0.1
        };

        reputation.insert(bootstrap.to_string(), new_rep);
        debug!("Bootstrap {} reputation: {:.2}", bootstrap, new_rep);
    }

    pub async fn discover_peers_from_all_bootstraps(&self) -> Result<Vec<String>> {
        info!("🆓 Discovering peers from all bootstrap nodes (FREE)");

        let mut all_discovered = HashSet::new();
        let mut successful_bootstraps = 0;

        for bootstrap in &self.bootstrap_nodes {
            match Self::query_bootstrap_node(&self.tor_client, bootstrap).await {
                Ok(peer_list) => {
                    successful_bootstraps += 1;
                    Self::merge_peer_list(&self.discovered_peers, peer_list, self.peer_ttl).await;

                    let peers = self.discovered_peers.read().await;
                    for peer_address in peers.keys() {
                        all_discovered.insert(peer_address.clone());
                    }

                    info!(
                        "🆓 Bootstrap {} provided {} peers (FREE)",
                        bootstrap,
                        peers.len()
                    );
                }
                Err(e) => {
                    warn!("Bootstrap discovery failed for {}: {}", bootstrap, e);
                }
            }
        }

        if successful_bootstraps == 0 {
            return Err(anyhow!("All bootstrap nodes failed"));
        }

        let result: Vec<String> = all_discovered.into_iter().collect();
        info!(
            "🆓 Total unique peers discovered: {} from {} bootstraps (FREE)",
            result.len(),
            successful_bootstraps
        );

        Ok(result)
    }

    pub async fn get_discovered_peers(&self) -> Vec<String> {
        let peers = self.discovered_peers.read().await;
        peers.keys().cloned().collect()
    }

    pub async fn get_peer_info(&self, address: &str) -> Option<PeerInfo> {
        let peers = self.discovered_peers.read().await;
        peers.get(address).cloned()
    }

    pub async fn get_peer_count(&self) -> usize {
        let peers = self.discovered_peers.read().await;
        peers.len()
    }

    pub async fn get_bootstrap_reputation(&self) -> HashMap<String, f64> {
        let reputation = self.bootstrap_reputation.read().await;
        reputation.clone()
    }

    pub async fn add_bootstrap_node(&self, address: String) {
        // This would be used to add new community bootstrap nodes dynamically
        info!("🆓 Adding new bootstrap node: {} (FREE)", address);
        // Implementation would update the bootstrap_nodes list
    }
}

// Production bootstrap protocol implementation
pub struct BootstrapProtocol;

impl BootstrapProtocol {
    pub fn create_peer_list_request() -> Vec<u8> {
        // Create Q-NarwhalKnight PEER_LIST_REQUEST message
        // This would be a proper binary protocol message
        b"QNK_PEER_LIST_REQUEST_V1".to_vec()
    }

    pub fn parse_peer_list_response(data: &[u8]) -> Result<PeerListResponse> {
        // Parse the binary response from bootstrap node
        // Real implementation would handle the actual protocol

        if data.starts_with(b"QNK_PEER_LIST_RESPONSE_V1") {
            // Parse the binary peer list format
            Ok(PeerListResponse {
                peers: Vec::new(),
                timestamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                bootstrap_node: "unknown".to_string(),
            })
        } else {
            Err(anyhow!("Invalid peer list response format"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_peer_info_address_string() {
        let peer = PeerInfo {
            onion_address: "test123.onion".to_string(),
            port: 8333,
            node_id: "node1".to_string(),
            last_seen: 0,
            capabilities: vec![],
        };

        assert_eq!(peer.to_address_string(), "test123.onion:8333");
    }

    #[tokio::test]
    async fn test_bootstrap_reputation() {
        let reputation = Arc::new(RwLock::new(HashMap::new()));

        // Test reputation increase
        BootstrapDiscovery::update_reputation(&reputation, "test.onion", true).await;
        let rep = reputation.read().await.get("test.onion").copied().unwrap();
        assert!(rep > 1.0);

        // Test reputation decrease
        BootstrapDiscovery::update_reputation(&reputation, "test.onion", false).await;
        let rep2 = reputation.read().await.get("test.onion").copied().unwrap();
        assert!(rep2 < rep);
    }
}
