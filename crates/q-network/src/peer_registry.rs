/// Peer Registry for Q-NarwhalKnight Network State Synchronization
/// Manages persistent peer information with real onion addresses
use anyhow::Result;
use q_types::{NodeId, ValidatorId};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Comprehensive peer information for production network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerInfo {
    pub validator_id: ValidatorId,
    pub onion_address: String,
    pub public_key: Vec<u8>,
    pub capabilities: HashSet<PeerCapability>,
    pub network_addresses: Vec<SocketAddr>,
    #[serde(
        serialize_with = "serialize_instant",
        deserialize_with = "deserialize_instant"
    )]
    pub last_seen: Instant,
    pub connection_quality: ConnectionQuality,
    pub protocol_version: String,
    pub stake: u64, // For consensus weight
    pub reputation_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Hash, Eq, PartialEq)]
pub enum PeerCapability {
    Consensus,
    Mempool,
    StateSync,
    ArchiveNode,
    BootstrapNode,
    TorRelay,
    QuantumReady,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionQuality {
    pub latency_ms: u32,
    pub bandwidth_mbps: u32,
    pub uptime_percentage: f64,
    pub failed_connections: u32,
    pub successful_connections: u32,
}

impl ConnectionQuality {
    pub fn new() -> Self {
        Self {
            latency_ms: 0,
            bandwidth_mbps: 0,
            uptime_percentage: 100.0,
            failed_connections: 0,
            successful_connections: 0,
        }
    }

    pub fn update_latency(&mut self, latency_ms: u32) {
        self.latency_ms = latency_ms;
    }

    pub fn record_connection_success(&mut self) {
        self.successful_connections += 1;
        self.uptime_percentage = (self.successful_connections as f64
            / (self.successful_connections + self.failed_connections) as f64)
            * 100.0;
    }

    pub fn record_connection_failure(&mut self) {
        self.failed_connections += 1;
        self.uptime_percentage = (self.successful_connections as f64
            / (self.successful_connections + self.failed_connections) as f64)
            * 100.0;
    }
}

/// Registry for managing validator peers in the network
pub struct PeerRegistry {
    peers: RwLock<HashMap<ValidatorId, PeerInfo>>,
    bootstrap_peers: RwLock<Vec<PeerInfo>>,
    connected_peers: RwLock<HashSet<ValidatorId>>,
    local_validator_id: ValidatorId,
    network_view_hash: RwLock<Option<[u8; 32]>>,
}

impl PeerRegistry {
    pub fn new(local_validator_id: ValidatorId) -> Self {
        Self {
            peers: RwLock::new(HashMap::new()),
            bootstrap_peers: RwLock::new(Vec::new()),
            connected_peers: RwLock::new(HashSet::new()),
            local_validator_id,
            network_view_hash: RwLock::new(None),
        }
    }

    /// Register a new peer or update existing peer information
    pub async fn register_peer(&self, peer_info: PeerInfo) -> Result<()> {
        let validator_id = peer_info.validator_id;

        info!(
            "📝 Registering peer {} with onion: {}",
            hex::encode(validator_id),
            peer_info.onion_address
        );

        {
            let mut peers = self.peers.write().await;
            peers.insert(validator_id, peer_info);
        }

        // Update network view hash
        self.update_network_view_hash().await?;

        debug!(
            "✅ Peer {} registered successfully",
            hex::encode(validator_id)
        );
        Ok(())
    }

    /// Get peer information by validator ID
    pub async fn get_peer(&self, validator_id: &ValidatorId) -> Option<PeerInfo> {
        let peers = self.peers.read().await;
        peers.get(validator_id).cloned()
    }

    /// Get all registered peers
    pub async fn get_all_peers(&self) -> Vec<PeerInfo> {
        let peers = self.peers.read().await;
        peers.values().cloned().collect()
    }

    /// Get peers with specific capability
    pub async fn get_peers_with_capability(&self, capability: PeerCapability) -> Vec<PeerInfo> {
        let peers = self.peers.read().await;
        peers
            .values()
            .filter(|peer| peer.capabilities.contains(&capability))
            .cloned()
            .collect()
    }

    /// Mark peer as connected
    pub async fn mark_peer_connected(&self, validator_id: ValidatorId) -> Result<()> {
        {
            let mut connected = self.connected_peers.write().await;
            connected.insert(validator_id);
        }

        // Update connection quality
        {
            let mut peers = self.peers.write().await;
            if let Some(peer) = peers.get_mut(&validator_id) {
                peer.connection_quality.record_connection_success();
                peer.last_seen = Instant::now();
            }
        }

        info!("🔗 Peer {} marked as connected", hex::encode(validator_id));
        Ok(())
    }

    /// Mark peer as disconnected
    pub async fn mark_peer_disconnected(&self, validator_id: &ValidatorId) -> Result<()> {
        {
            let mut connected = self.connected_peers.write().await;
            connected.remove(validator_id);
        }

        // Update connection quality
        {
            let mut peers = self.peers.write().await;
            if let Some(peer) = peers.get_mut(validator_id) {
                peer.connection_quality.record_connection_failure();
            }
        }

        warn!(
            "🔌 Peer {} marked as disconnected",
            hex::encode(*validator_id)
        );
        Ok(())
    }

    /// Get currently connected peers
    pub async fn get_connected_peers(&self) -> Vec<ValidatorId> {
        let connected = self.connected_peers.read().await;
        connected.iter().copied().collect()
    }

    /// Update peer latency measurement
    pub async fn update_peer_latency(
        &self,
        validator_id: &ValidatorId,
        latency_ms: u32,
    ) -> Result<()> {
        let mut peers = self.peers.write().await;
        if let Some(peer) = peers.get_mut(validator_id) {
            peer.connection_quality.update_latency(latency_ms);
            peer.last_seen = Instant::now();
            debug!(
                "📊 Updated latency for peer {}: {}ms",
                hex::encode(*validator_id),
                latency_ms
            );
        }
        Ok(())
    }

    /// Add bootstrap peers (known validators for network bootstrapping)
    pub async fn add_bootstrap_peers(&self, bootstrap_peers: Vec<PeerInfo>) -> Result<()> {
        info!("🚀 Adding {} bootstrap peers", bootstrap_peers.len());

        {
            let mut bootstrap = self.bootstrap_peers.write().await;
            bootstrap.extend(bootstrap_peers);
        }

        self.update_network_view_hash().await?;
        Ok(())
    }

    /// Get bootstrap peers for initial connection
    pub async fn get_bootstrap_peers(&self) -> Vec<PeerInfo> {
        let bootstrap = self.bootstrap_peers.read().await;
        bootstrap.clone()
    }

    /// Get network statistics
    pub async fn get_network_stats(&self) -> NetworkRegistryStats {
        let peers = self.peers.read().await;
        let connected = self.connected_peers.read().await;
        let bootstrap = self.bootstrap_peers.read().await;

        let total_peers = peers.len();
        let connected_count = connected.len();
        let bootstrap_count = bootstrap.len();

        let average_latency = if total_peers > 0 {
            peers
                .values()
                .map(|p| p.connection_quality.latency_ms)
                .sum::<u32>()
                / total_peers as u32
        } else {
            0
        };

        NetworkRegistryStats {
            total_peers,
            connected_peers: connected_count,
            bootstrap_peers: bootstrap_count,
            average_latency_ms: average_latency,
            network_view_hash: *self.network_view_hash.read().await,
        }
    }

    /// Update network view hash for consensus consistency
    async fn update_network_view_hash(&self) -> Result<()> {
        use sha3::{Digest, Sha3_256};

        let peers = self.peers.read().await;
        let bootstrap = self.bootstrap_peers.read().await;

        let mut hasher = Sha3_256::new();

        // Include all peer validator IDs in deterministic order
        let mut peer_ids: Vec<_> = peers.keys().collect();
        peer_ids.sort();

        for peer_id in peer_ids {
            hasher.update(peer_id);
        }

        // Include bootstrap peers
        for peer in bootstrap.iter() {
            hasher.update(peer.validator_id);
        }

        let hash: [u8; 32] = hasher.finalize().into();

        {
            let mut view_hash = self.network_view_hash.write().await;
            *view_hash = Some(hash);
        }

        debug!("🔄 Network view hash updated: {}", hex::encode(hash));
        Ok(())
    }

    /// Get network view hash for consistency checks
    pub async fn get_network_view_hash(&self) -> Option<[u8; 32]> {
        *self.network_view_hash.read().await
    }

    /// Remove stale peers (not seen for over threshold duration)
    pub async fn remove_stale_peers(&self, max_age: Duration) -> Result<Vec<ValidatorId>> {
        let mut removed_peers = Vec::new();
        let now = Instant::now();

        {
            let mut peers = self.peers.write().await;
            let mut connected = self.connected_peers.write().await;

            peers.retain(|validator_id, peer| {
                if now.duration_since(peer.last_seen) > max_age {
                    removed_peers.push(*validator_id);
                    connected.remove(validator_id);
                    false
                } else {
                    true
                }
            });
        }

        if !removed_peers.is_empty() {
            warn!("🧹 Removed {} stale peers", removed_peers.len());
            self.update_network_view_hash().await?;
        }

        Ok(removed_peers)
    }

    /// Get peer count by capability
    pub async fn get_capability_distribution(&self) -> HashMap<PeerCapability, usize> {
        let peers = self.peers.read().await;
        let mut distribution = HashMap::new();

        for peer in peers.values() {
            for capability in &peer.capabilities {
                *distribution.entry(capability.clone()).or_insert(0) += 1;
            }
        }

        distribution
    }
}

/// Network registry statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkRegistryStats {
    pub total_peers: usize,
    pub connected_peers: usize,
    pub bootstrap_peers: usize,
    pub average_latency_ms: u32,
    pub network_view_hash: Option<[u8; 32]>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_peer_registry_creation() {
        let validator_id = [1u8; 32];
        let registry = PeerRegistry::new(validator_id);

        let stats = registry.get_network_stats().await;
        assert_eq!(stats.total_peers, 0);
        assert_eq!(stats.connected_peers, 0);
    }

    #[tokio::test]
    async fn test_peer_registration() {
        let validator_id = [1u8; 32];
        let registry = PeerRegistry::new(validator_id);

        let peer_validator_id = [2u8; 32];
        let peer_info = PeerInfo {
            validator_id: peer_validator_id,
            onion_address: "validator2.qnk.onion".to_string(),
            public_key: vec![1, 2, 3, 4],
            capabilities: [PeerCapability::Consensus].into(),
            network_addresses: vec!["127.0.0.1:8080".parse().unwrap()],
            last_seen: Instant::now(),
            connection_quality: ConnectionQuality::new(),
            protocol_version: "qnk/1.0".to_string(),
            stake: 1000,
            reputation_score: 1.0,
        };

        registry.register_peer(peer_info).await.unwrap();

        let stats = registry.get_network_stats().await;
        assert_eq!(stats.total_peers, 1);

        let retrieved_peer = registry.get_peer(&peer_validator_id).await.unwrap();
        assert_eq!(retrieved_peer.onion_address, "validator2.qnk.onion");
    }

    #[tokio::test]
    async fn test_peer_connection_tracking() {
        let validator_id = [1u8; 32];
        let registry = PeerRegistry::new(validator_id);

        let peer_validator_id = [2u8; 32];
        let peer_info = PeerInfo {
            validator_id: peer_validator_id,
            onion_address: "validator2.qnk.onion".to_string(),
            public_key: vec![1, 2, 3, 4],
            capabilities: [PeerCapability::Consensus].into(),
            network_addresses: vec![],
            last_seen: Instant::now(),
            connection_quality: ConnectionQuality::new(),
            protocol_version: "qnk/1.0".to_string(),
            stake: 1000,
            reputation_score: 1.0,
        };

        registry.register_peer(peer_info).await.unwrap();
        registry
            .mark_peer_connected(peer_validator_id)
            .await
            .unwrap();

        let connected_peers = registry.get_connected_peers().await;
        assert_eq!(connected_peers.len(), 1);
        assert!(connected_peers.contains(&peer_validator_id));
    }
}

// Custom serialization for Instant
fn serialize_instant<S>(instant: &Instant, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let duration = instant.elapsed();
    let secs = duration.as_secs();
    serializer.serialize_u64(secs)
}

fn deserialize_instant<'de, D>(deserializer: D) -> Result<Instant, D::Error>
where
    D: Deserializer<'de>,
{
    let secs = u64::deserialize(deserializer)?;
    let duration = Duration::from_secs(secs);
    // Use checked_sub to avoid panic on Windows where Instant is based on uptime
    Ok(Instant::now().checked_sub(duration).unwrap_or(Instant::now()))
}
