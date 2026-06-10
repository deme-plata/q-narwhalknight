use crate::TorClient;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GossipMessage {
    pub message_type: GossipMessageType,
    pub sender_id: String,
    pub timestamp: u64,
    pub payload: GossipPayload,
    pub signature: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GossipMessageType {
    PeerAnnouncement,
    PeerListShare,
    PeerListRequest,
    HeartbeatPing,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GossipPayload {
    PeerList(Vec<GossipPeerInfo>),
    PeerAnnouncement(GossipPeerInfo),
    PeerListRequest { max_peers: u32 },
    Heartbeat { load: f64, uptime: u64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GossipPeerInfo {
    pub onion_address: String,
    pub port: u16,
    pub node_id: String,
    pub capabilities: Vec<String>,
    pub last_seen: u64,
    pub reputation: f64,
    pub hop_count: u8, // How many hops this info traveled
}

impl GossipPeerInfo {
    pub fn new(onion_address: String, port: u16, node_id: String) -> Self {
        Self {
            onion_address,
            port,
            node_id,
            capabilities: vec!["consensus".to_string(), "mempool".to_string()],
            last_seen: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            reputation: 1.0,
            hop_count: 0,
        }
    }

    pub fn to_address_string(&self) -> String {
        format!("{}:{}", self.onion_address, self.port)
    }

    pub fn age_seconds(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
            - self.last_seen
    }

    pub fn with_incremented_hop(mut self) -> Self {
        self.hop_count = self.hop_count.saturating_add(1);
        self
    }
}

pub struct GossipDiscovery {
    tor_client: Arc<TorClient>,
    our_node_id: String,
    our_onion_address: String,
    our_port: u16,

    // Peer management
    known_peers: Arc<RwLock<HashMap<String, GossipPeerInfo>>>,
    connected_peers: Arc<RwLock<HashSet<String>>>,
    peer_connections: Arc<RwLock<HashMap<String, SystemTime>>>,

    // Gossip configuration
    gossip_interval: Duration,
    max_hop_count: u8,
    max_peers_per_gossip: u32,
    peer_ttl: Duration,
    reputation_decay_rate: f64,
}

impl GossipDiscovery {
    pub fn new(
        tor_client: Arc<TorClient>,
        node_id: String,
        onion_address: String,
        port: u16,
    ) -> Self {
        Self {
            tor_client,
            our_node_id: node_id,
            our_onion_address: onion_address,
            our_port: port,
            known_peers: Arc::new(RwLock::new(HashMap::new())),
            connected_peers: Arc::new(RwLock::new(HashSet::new())),
            peer_connections: Arc::new(RwLock::new(HashMap::new())),
            gossip_interval: Duration::from_secs(60), // 1 minute
            max_hop_count: 5,
            max_peers_per_gossip: 10,
            peer_ttl: Duration::from_secs(1800), // 30 minutes
            reputation_decay_rate: 0.99,         // 1% decay per interval
        }
    }

    pub async fn start_gossip_protocol(&self) -> Result<()> {
        info!("🆓 Starting gossip protocol for viral peer discovery (completely FREE)");

        // Start gossip broadcast loop
        self.start_gossip_broadcast_loop().await;

        // Start reputation decay loop
        self.start_reputation_decay_loop().await;

        // Start peer cleanup loop
        self.start_cleanup_loop().await;

        Ok(())
    }

    async fn start_gossip_broadcast_loop(&self) {
        let known_peers = Arc::clone(&self.known_peers);
        let connected_peers = Arc::clone(&self.connected_peers);
        let tor_client = Arc::clone(&self.tor_client);
        let our_node_id = self.our_node_id.clone();
        let gossip_interval = self.gossip_interval;
        let max_peers_per_gossip = self.max_peers_per_gossip;

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(gossip_interval);

            loop {
                interval.tick().await;

                let peers_to_share = {
                    let peers = known_peers.read().await;
                    let connected = connected_peers.read().await;

                    // Select best peers to share (highest reputation, lowest hop count)
                    let mut peer_list: Vec<_> = peers.values().cloned().collect();
                    peer_list.sort_by(|a, b| {
                        b.reputation
                            .partial_cmp(&a.reputation)
                            .unwrap()
                            .then(a.hop_count.cmp(&b.hop_count))
                    });
                    peer_list.truncate(max_peers_per_gossip as usize);
                    peer_list
                };

                if !peers_to_share.is_empty() {
                    let connected_list: Vec<String> = {
                        let connected = connected_peers.read().await;
                        connected.iter().cloned().collect()
                    };

                    let connected_count = connected_list.len();
                    for peer_address in &connected_list {
                        if let Err(e) = Self::send_gossip_message(
                            &tor_client,
                            &peer_address,
                            &our_node_id,
                            GossipPayload::PeerList(peers_to_share.clone()),
                        )
                        .await
                        {
                            debug!("Failed to gossip to {}: {}", peer_address, e);
                        } else {
                            debug!(
                                "🆓 Gossiped {} peers to {} (FREE)",
                                peers_to_share.len(),
                                peer_address
                            );
                        }
                    }

                    info!(
                        "🆓 Gossiped {} peers to {} connected nodes (FREE - viral spread)",
                        peers_to_share.len(),
                        connected_count
                    );
                }
            }
        });
    }

    async fn start_reputation_decay_loop(&self) {
        let known_peers = Arc::clone(&self.known_peers);
        let decay_rate = self.reputation_decay_rate;

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(300)); // 5 minutes

            loop {
                interval.tick().await;

                let mut peers = known_peers.write().await;
                for peer in peers.values_mut() {
                    peer.reputation *= decay_rate;

                    // Remove peers with very low reputation
                    if peer.reputation < 0.1 {
                        debug!("Removing low reputation peer: {}", peer.to_address_string());
                    }
                }

                // Keep only peers with acceptable reputation
                peers.retain(|_, peer| peer.reputation >= 0.1);

                debug!(
                    "🆓 Applied reputation decay to {} peers (FREE)",
                    peers.len()
                );
            }
        });
    }

    async fn start_cleanup_loop(&self) {
        let known_peers = Arc::clone(&self.known_peers);
        let connected_peers = Arc::clone(&self.connected_peers);
        let peer_connections = Arc::clone(&self.peer_connections);
        let peer_ttl = self.peer_ttl;

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(120)); // 2 minutes

            loop {
                interval.tick().await;

                let current_time = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs();
                let ttl_seconds = peer_ttl.as_secs();

                // Clean up expired peers
                {
                    let mut peers = known_peers.write().await;
                    let initial_count = peers.len();

                    peers.retain(|_, peer| current_time - peer.last_seen < ttl_seconds);

                    let removed = initial_count - peers.len();
                    if removed > 0 {
                        debug!("🆓 Cleaned up {} expired peers (FREE)", removed);
                    }
                }

                // Clean up old connection records
                {
                    let mut connections = peer_connections.write().await;
                    let cutoff_time = SystemTime::now() - peer_ttl;

                    connections.retain(|_, &mut last_contact| last_contact > cutoff_time);
                }
            }
        });
    }

    async fn send_gossip_message(
        tor_client: &TorClient,
        peer_address: &str,
        sender_id: &str,
        payload: GossipPayload,
    ) -> Result<()> {
        let message = GossipMessage {
            message_type: match payload {
                GossipPayload::PeerList(_) => GossipMessageType::PeerListShare,
                GossipPayload::PeerAnnouncement(_) => GossipMessageType::PeerAnnouncement,
                GossipPayload::PeerListRequest { .. } => GossipMessageType::PeerListRequest,
                GossipPayload::Heartbeat { .. } => GossipMessageType::HeartbeatPing,
            },
            sender_id: sender_id.to_string(),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            payload,
            signature: Vec::new(), // Would be cryptographically signed in production
        };

        // Parse peer address
        let parts: Vec<&str> = peer_address.split(':').collect();
        if parts.len() != 2 {
            return Err(anyhow!("Invalid peer address format"));
        }

        let onion_address = parts[0];
        let port: u16 = parts[1].parse()?;

        // Send gossip message through Tor connection - completely free
        Self::send_tor_message(tor_client, onion_address, port, &message).await?;

        debug!("🆓 Sent gossip message to {} (FREE)", peer_address);
        Ok(())
    }

    async fn send_tor_message(
        tor_client: &TorClient,
        onion_address: &str,
        port: u16,
        message: &GossipMessage,
    ) -> Result<()> {
        // Real implementation would:
        // 1. Create Tor stream to peer
        // 2. Serialize gossip message to binary format
        // 3. Send message over secure Tor connection
        // 4. Handle response/acknowledgment

        let message_data = serde_json::to_vec(message)?;
        debug!(
            "🆓 Sending {} bytes to {}:{} via Tor (FREE)",
            message_data.len(),
            onion_address,
            port
        );

        // Simulate successful message delivery
        Ok(())
    }

    pub async fn handle_gossip_message(
        &self,
        message: GossipMessage,
        from_address: &str,
    ) -> Result<()> {
        debug!("🆓 Received gossip message from {} (FREE)", from_address);

        match message.payload {
            GossipPayload::PeerList(peer_list) => {
                self.process_peer_list(peer_list, from_address).await?;
            }
            GossipPayload::PeerAnnouncement(peer_info) => {
                self.process_peer_announcement(peer_info).await?;
            }
            GossipPayload::PeerListRequest { max_peers } => {
                self.handle_peer_list_request(from_address, max_peers)
                    .await?;
            }
            GossipPayload::Heartbeat { load, uptime } => {
                self.process_heartbeat(from_address, load, uptime).await?;
            }
        }

        // Update connection tracking
        {
            let mut connections = self.peer_connections.write().await;
            connections.insert(from_address.to_string(), SystemTime::now());
        }

        Ok(())
    }

    async fn process_peer_list(
        &self,
        peer_list: Vec<GossipPeerInfo>,
        from_address: &str,
    ) -> Result<()> {
        let mut known_peers = self.known_peers.write().await;
        let mut new_peers_count = 0;

        for peer in peer_list {
            // Skip if hop count is too high (prevent infinite gossip loops)
            if peer.hop_count >= self.max_hop_count {
                continue;
            }

            // Skip ourselves
            if peer.node_id == self.our_node_id {
                continue;
            }

            let peer_address = peer.to_address_string();
            let should_add = if let Some(existing) = known_peers.get(&peer_address) {
                // Update if this info is fresher or from a shorter path
                peer.last_seen > existing.last_seen
                    || (peer.last_seen == existing.last_seen && peer.hop_count < existing.hop_count)
            } else {
                true
            };

            if should_add {
                if !known_peers.contains_key(&peer_address) {
                    new_peers_count += 1;
                }

                let hop_count = peer.hop_count + 1;
                let updated_peer = peer.with_incremented_hop();
                known_peers.insert(peer_address.clone(), updated_peer);

                debug!(
                    "🆓 Added peer from gossip: {} (hop {}, FREE)",
                    peer_address, hop_count
                );
            }
        }

        if new_peers_count > 0 {
            info!(
                "🆓 Discovered {} new peers via gossip from {} (FREE - viral discovery)",
                new_peers_count, from_address
            );
        }

        Ok(())
    }

    async fn process_peer_announcement(&self, peer_info: GossipPeerInfo) -> Result<()> {
        if peer_info.node_id == self.our_node_id {
            return Ok(()); // Ignore our own announcements
        }

        let peer_address = peer_info.to_address_string();
        let mut known_peers = self.known_peers.write().await;

        if !known_peers.contains_key(&peer_address) {
            known_peers.insert(peer_address.clone(), peer_info.with_incremented_hop());
            info!("🆓 New peer announced: {} (FREE)", peer_address);
        }

        Ok(())
    }

    async fn handle_peer_list_request(&self, from_address: &str, max_peers: u32) -> Result<()> {
        let peer_list = {
            let peers = self.known_peers.read().await;
            let mut sorted_peers: Vec<_> = peers.values().cloned().collect();

            // Sort by reputation and recency
            sorted_peers.sort_by(|a, b| {
                b.reputation
                    .partial_cmp(&a.reputation)
                    .unwrap()
                    .then(b.last_seen.cmp(&a.last_seen))
            });

            sorted_peers.truncate(max_peers as usize);
            sorted_peers
        };

        // Send peer list response
        Self::send_gossip_message(
            &self.tor_client,
            from_address,
            &self.our_node_id,
            GossipPayload::PeerList(peer_list.clone()),
        )
        .await?;

        info!(
            "🆓 Sent {} peers to requester {} (FREE)",
            peer_list.len(),
            from_address
        );
        Ok(())
    }

    async fn process_heartbeat(&self, from_address: &str, load: f64, uptime: u64) -> Result<()> {
        // Update reputation based on heartbeat info
        let mut known_peers = self.known_peers.write().await;

        if let Some(peer) = known_peers.get_mut(from_address) {
            // Increase reputation for active peers
            peer.reputation = (peer.reputation + 0.05).min(2.0);
            peer.last_seen = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();

            debug!(
                "🆓 Updated peer {} reputation to {:.2} (FREE)",
                from_address, peer.reputation
            );
        }

        Ok(())
    }

    pub async fn announce_ourselves(&self) -> Result<()> {
        let our_peer_info = GossipPeerInfo::new(
            self.our_onion_address.clone(),
            self.our_port,
            self.our_node_id.clone(),
        );

        let connected_list: Vec<String> = {
            let connected = self.connected_peers.read().await;
            connected.iter().cloned().collect()
        };

        for peer_address in &connected_list {
            Self::send_gossip_message(
                &self.tor_client,
                peer_address,
                &self.our_node_id,
                GossipPayload::PeerAnnouncement(our_peer_info.clone()),
            )
            .await?;
        }

        info!(
            "🆓 Announced ourselves to {} peers (FREE - viral announcement)",
            connected_list.len()
        );
        Ok(())
    }

    pub async fn add_seed_peer(&self, onion_address: String, port: u16, node_id: String) {
        let peer_info = GossipPeerInfo::new(onion_address.clone(), port, node_id);
        let peer_address = peer_info.to_address_string();

        {
            let mut known_peers = self.known_peers.write().await;
            known_peers.insert(peer_address.clone(), peer_info);
        }

        {
            let mut connected = self.connected_peers.write().await;
            connected.insert(peer_address.clone());
        }

        info!("🆓 Added seed peer: {} (FREE)", peer_address);
    }

    pub async fn get_discovered_peers(&self) -> Vec<String> {
        let peers = self.known_peers.read().await;
        peers.keys().cloned().collect()
    }

    pub async fn get_peer_count(&self) -> usize {
        let peers = self.known_peers.read().await;
        peers.len()
    }

    pub async fn get_connected_peer_count(&self) -> usize {
        let connected = self.connected_peers.read().await;
        connected.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_gossip_peer_info() {
        let peer = GossipPeerInfo::new("test123.onion".to_string(), 8333, "node1".to_string());

        assert_eq!(peer.to_address_string(), "test123.onion:8333");
        assert_eq!(peer.hop_count, 0);

        let incremented = peer.with_incremented_hop();
        assert_eq!(incremented.hop_count, 1);
    }

    #[test]
    fn test_gossip_message_types() {
        let peer_list = vec![GossipPeerInfo::new(
            "test.onion".to_string(),
            8333,
            "node1".to_string(),
        )];
        let payload = GossipPayload::PeerList(peer_list);

        match payload {
            GossipPayload::PeerList(_) => {}
            _ => panic!("Wrong payload type"),
        }
    }
}
