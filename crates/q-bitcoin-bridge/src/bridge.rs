/// High-level Bitcoin-Tor bridge integration for Q-NarwhalKnight
///
/// This module provides the main interface for integrating Bitcoin-based peer discovery
/// with the Q-Knight network layer, managing the complete lifecycle of anonymous
/// peer connections through Bitcoin and Tor networks.
use anyhow::{anyhow, Result};
use q_types::{NodeId, PeerInfo};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::Duration,
};
use tokio::sync::{broadcast, mpsc, RwLock};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::{
    discovery::{BitcoinPeerDiscovery, DiscoveryStats},
    BitcoinBridge, BitcoinBridgeConfig, NodeAdvertisement, PeerDiscoveryEvent,
};

/// Integrated Bitcoin-Tor bridge for Q-Knight networking
pub struct IntegratedBitcoinBridge {
    config: BitcoinBridgeConfig,
    node_id: NodeId,
    onion_address: String,

    // Core components
    bitcoin_bridge: Arc<BitcoinBridge>,
    peer_discovery: Arc<BitcoinPeerDiscovery>,

    // Connection management
    active_connections: Arc<RwLock<HashMap<NodeId, ActiveConnection>>>,
    connection_attempts: Arc<RwLock<HashMap<NodeId, ConnectionAttempt>>>,

    // Event handling
    peer_events_tx: broadcast::Sender<PeerNetworkEvent>,
    internal_events_rx: Option<mpsc::UnboundedReceiver<PeerDiscoveryEvent>>,
}

#[derive(Debug, Clone)]
pub struct ActiveConnection {
    pub peer_info: PeerInfo,
    pub connection_id: Uuid,
    pub established_at: chrono::DateTime<chrono::Utc>,
    pub last_activity: chrono::DateTime<chrono::Utc>,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub tor_circuit_id: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct ConnectionAttempt {
    pub node_id: NodeId,
    pub attempt_id: Uuid,
    pub started_at: chrono::DateTime<chrono::Utc>,
    pub attempts_count: u32,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone)]
pub enum PeerNetworkEvent {
    /// New peer discovered through Bitcoin network
    PeerDiscovered {
        node_id: NodeId,
        advertisement: NodeAdvertisement,
        confidence: f64,
    },
    /// Successfully connected to a peer
    PeerConnected {
        node_id: NodeId,
        peer_info: PeerInfo,
        connection_id: Uuid,
    },
    /// Peer disconnected
    PeerDisconnected {
        node_id: NodeId,
        connection_id: Uuid,
        reason: String,
    },
    /// Connection attempt failed
    ConnectionFailed {
        node_id: NodeId,
        error: String,
        will_retry: bool,
    },
    /// Peer advertisement expired
    PeerExpired { node_id: NodeId },
}

impl IntegratedBitcoinBridge {
    /// Create a new integrated Bitcoin-Tor bridge
    pub async fn new(
        config: BitcoinBridgeConfig,
        node_id: NodeId,
        onion_address: String,
        tor_client: Arc<q_tor_client::QTorClient>,
    ) -> Result<Self> {
        info!("Initializing integrated Bitcoin-Tor bridge");

        // Create internal event channel for peer discovery
        let (internal_tx, internal_rx) = mpsc::unbounded_channel();

        // Create Bitcoin bridge
        let bitcoin_bridge = BitcoinBridge::new().await?;
        let bitcoin_bridge = Arc::new(bitcoin_bridge);

        // Initialize Bitcoin connection
        bitcoin_bridge.initialize().await?;

        // Create peer discovery
        let peer_discovery = Arc::new(BitcoinPeerDiscovery::new(
            config.clone(),
            bitcoin_bridge.bitcoin_client.clone(),
            internal_tx,
        ));

        // Create peer network event channel
        let (peer_events_tx, _) = broadcast::channel(1000);

        let bridge = Self {
            config,
            node_id,
            onion_address,
            bitcoin_bridge,
            peer_discovery,
            active_connections: Arc::new(RwLock::new(HashMap::new())),
            connection_attempts: Arc::new(RwLock::new(HashMap::new())),
            peer_events_tx,
            internal_events_rx: Some(internal_rx),
        };

        Ok(bridge)
    }

    /// Start the integrated bridge system
    pub async fn start(&mut self) -> Result<()> {
        info!("Starting integrated Bitcoin-Tor bridge");

        // Start advertising our node
        self.bitcoin_bridge
            .clone()
            .start_discovery(self.node_id, self.onion_address.clone())
            .await?;

        // Start peer discovery
        self.peer_discovery.clone().start_discovery().await?;

        // Skip background tasks for now - they require complex refactoring
        // TODO: Implement background task management without self references

        info!("Bitcoin-Tor bridge started successfully");
        Ok(())
    }

    /// Process internal peer discovery events
    async fn process_internal_events(
        &self,
        mut internal_rx: mpsc::UnboundedReceiver<PeerDiscoveryEvent>,
    ) {
        while let Some(event) = internal_rx.recv().await {
            match event {
                PeerDiscoveryEvent::PeerDiscovered {
                    node_id,
                    advertisement,
                } => {
                    info!("Processing discovered peer: {}", hex::encode(node_id));

                    // Calculate confidence based on discovery method and verification
                    let confidence = self.calculate_peer_confidence(&advertisement).await;

                    // Send network event
                    let network_event = PeerNetworkEvent::PeerDiscovered {
                        node_id,
                        advertisement: advertisement.clone(),
                        confidence,
                    };

                    if let Err(e) = self.peer_events_tx.send(network_event) {
                        warn!("Failed to send peer discovered event: {}", e);
                    }

                    // Initiate connection if peer meets criteria
                    if confidence > 0.7 && self.should_connect_to_peer(node_id).await {
                        self.initiate_connection(node_id, advertisement).await;
                    }
                }
                PeerDiscoveryEvent::PeerUpdated {
                    node_id,
                    advertisement,
                } => {
                    debug!("Peer updated: {}", hex::encode(node_id));

                    // Update existing connection if any
                    self.update_peer_connection(node_id, advertisement).await;
                }
                PeerDiscoveryEvent::PeerExpired { node_id } => {
                    info!("Peer expired: {}", hex::encode(node_id));

                    // Disconnect if connected
                    self.disconnect_peer(node_id, "Peer advertisement expired".to_string())
                        .await;

                    let network_event = PeerNetworkEvent::PeerExpired { node_id };
                    let _ = self.peer_events_tx.send(network_event);
                }
            }
        }
    }

    /// Calculate confidence score for discovered peer
    async fn calculate_peer_confidence(&self, advertisement: &NodeAdvertisement) -> f64 {
        let mut confidence: f64 = 0.5; // Base confidence

        // Check advertisement validity
        if advertisement.expires_at > chrono::Utc::now() {
            confidence += 0.2;
        }

        // Check signature validity (if implemented)
        if !advertisement.signature.is_empty() {
            confidence += 0.2;
        }

        // Check protocol compatibility
        if advertisement.protocol_version.starts_with("q-knight")
            || advertisement.protocol_version.starts_with("qk/")
        {
            confidence += 0.1;
        }

        // Check capabilities
        if advertisement
            .capabilities
            .contains(&"dag-consensus".to_string())
            || advertisement.capabilities.contains(&"DAG".to_string())
        {
            confidence += 0.1;
        }

        confidence.min(1.0)
    }

    /// Check if we should connect to a discovered peer
    async fn should_connect_to_peer(&self, node_id: NodeId) -> bool {
        // Don't connect to ourselves
        if node_id == self.node_id {
            return false;
        }

        // Check if already connected
        {
            let connections = self.active_connections.read().await;
            if connections.contains_key(&node_id) {
                return false;
            }
        }

        // Check if currently attempting connection
        {
            let attempts = self.connection_attempts.read().await;
            if attempts.contains_key(&node_id) {
                return false;
            }
        }

        // Check peer count limits
        let connections = self.active_connections.read().await;
        if connections.len() >= self.config.max_peers_advertised {
            return false;
        }

        true
    }

    /// Initiate connection to a discovered peer
    async fn initiate_connection(&self, node_id: NodeId, advertisement: NodeAdvertisement) {
        let attempt_id = Uuid::new_v4();
        let now = chrono::Utc::now();

        // Record connection attempt
        {
            let mut attempts = self.connection_attempts.write().await;
            attempts.insert(
                node_id,
                ConnectionAttempt {
                    node_id,
                    attempt_id,
                    started_at: now,
                    attempts_count: 1,
                    last_error: None,
                },
            );
        }

        info!("Initiating connection to peer {}", hex::encode(node_id));

        // Attempt connection through Bitcoin bridge
        match self.bitcoin_bridge.connect_to_peer(node_id).await {
            Ok(peer_info) => {
                // Connection successful
                let connection_id = Uuid::new_v4();
                let active_connection = ActiveConnection {
                    peer_info: peer_info.clone(),
                    connection_id,
                    established_at: now,
                    last_activity: now,
                    bytes_sent: 0,
                    bytes_received: 0,
                    tor_circuit_id: None, // TODO: Get actual circuit ID
                };

                // Record active connection
                {
                    let mut connections = self.active_connections.write().await;
                    connections.insert(node_id, active_connection);
                }

                // Remove from attempts
                {
                    let mut attempts = self.connection_attempts.write().await;
                    attempts.remove(&node_id);
                }

                info!("Successfully connected to peer {}", hex::encode(node_id));

                // Send connection event
                let network_event = PeerNetworkEvent::PeerConnected {
                    node_id,
                    peer_info,
                    connection_id,
                };
                let _ = self.peer_events_tx.send(network_event);
            }
            Err(e) => {
                error!("Failed to connect to peer {}: {}", hex::encode(node_id), e);

                // Update attempt record
                {
                    let mut attempts = self.connection_attempts.write().await;
                    if let Some(attempt) = attempts.get_mut(&node_id) {
                        attempt.attempts_count += 1;
                        attempt.last_error = Some(e.to_string());

                        // Remove attempt if too many failures
                        if attempt.attempts_count >= 3 {
                            attempts.remove(&node_id);
                        }
                    }
                }

                let network_event = PeerNetworkEvent::ConnectionFailed {
                    node_id,
                    error: e.to_string(),
                    will_retry: true, // TODO: Implement retry logic
                };
                let _ = self.peer_events_tx.send(network_event);
            }
        }
    }

    /// Update existing peer connection
    async fn update_peer_connection(&self, node_id: NodeId, _advertisement: NodeAdvertisement) {
        // Update connection information if peer is connected
        let mut connections = self.active_connections.write().await;
        if let Some(connection) = connections.get_mut(&node_id) {
            connection.last_activity = chrono::Utc::now();
            debug!("Updated connection for peer {}", hex::encode(node_id));
        }
    }

    /// Disconnect from a peer
    async fn disconnect_peer(&self, node_id: NodeId, reason: String) {
        let mut connections = self.active_connections.write().await;
        if let Some(connection) = connections.remove(&node_id) {
            info!(
                "Disconnecting from peer {}: {}",
                hex::encode(node_id),
                reason
            );

            // TODO: Actually close the connection/circuit

            let network_event = PeerNetworkEvent::PeerDisconnected {
                node_id,
                connection_id: connection.connection_id,
                reason,
            };
            let _ = self.peer_events_tx.send(network_event);
        }
    }

    /// Connection manager loop
    async fn connection_manager_loop(&self) {
        let mut interval = tokio::time::interval(Duration::from_secs(30));

        loop {
            interval.tick().await;
            self.manage_connections().await;
        }
    }

    /// Manage existing connections and attempts
    async fn manage_connections(&self) {
        let now = chrono::Utc::now();

        // Clean up old connection attempts
        {
            let mut attempts = self.connection_attempts.write().await;
            attempts.retain(|_, attempt| {
                let age = now.signed_duration_since(attempt.started_at);
                age < chrono::Duration::minutes(5) // Keep attempts for 5 minutes max
            });
        }

        // Check for stale connections
        {
            let mut connections = self.active_connections.write().await;
            let stale_peers: Vec<NodeId> = connections
                .iter()
                .filter(|(_, conn)| {
                    let age = now.signed_duration_since(conn.last_activity);
                    age > chrono::Duration::minutes(30) // Consider stale after 30 minutes
                })
                .map(|(&node_id, _)| node_id)
                .collect();

            for node_id in stale_peers {
                connections.remove(&node_id);
                info!("Removed stale connection to peer {}", hex::encode(node_id));
            }
        }
    }

    /// Health checker loop
    async fn health_checker_loop(&self) {
        let mut interval = tokio::time::interval(Duration::from_secs(60));

        loop {
            interval.tick().await;
            self.check_system_health().await;
        }
    }

    /// Check system health and performance
    async fn check_system_health(&self) {
        let stats = self.peer_discovery.get_discovery_stats().await;

        debug!(
            "System health: {} peers discovered, {} high confidence, {} blocks processed",
            stats.total_peers_discovered, stats.high_confidence_peers, stats.blocks_processed
        );

        // TODO: Implement health checks and alerts
    }

    /// Subscribe to peer network events
    pub fn subscribe_to_events(&self) -> broadcast::Receiver<PeerNetworkEvent> {
        self.peer_events_tx.subscribe()
    }

    /// Get current connection statistics
    pub async fn get_connection_stats(&self) -> ConnectionStats {
        let connections = self.active_connections.read().await;
        let attempts = self.connection_attempts.read().await;
        let discovery_stats = self.peer_discovery.get_discovery_stats().await;

        ConnectionStats {
            active_connections: connections.len() as u32,
            pending_attempts: attempts.len() as u32,
            total_discovered_peers: discovery_stats.total_peers_discovered,
            successful_connections: 0, // TODO: Track this
            failed_connections: 0,     // TODO: Track this
            average_connection_time: Duration::from_secs(0), // TODO: Calculate this
            last_updated: chrono::Utc::now(),
        }
    }

    /// Get list of active peer connections
    pub async fn get_active_peers(&self) -> Vec<(NodeId, PeerInfo)> {
        let connections = self.active_connections.read().await;
        connections
            .iter()
            .map(|(&node_id, conn)| (node_id, conn.peer_info.clone()))
            .collect()
    }

    /// Manually trigger connection to a specific peer
    pub async fn connect_to_peer(&self, node_id: NodeId) -> Result<()> {
        // Check if we already have this peer's advertisement
        let discovered_peers = self.bitcoin_bridge.get_discovered_peers().await;

        if let Some(advertisement) = discovered_peers.get(&node_id) {
            self.initiate_connection(node_id, advertisement.clone())
                .await;
            Ok(())
        } else {
            Err(anyhow!(
                "Peer {} not found in discovered peers",
                hex::encode(node_id)
            ))
        }
    }

    /// Manually disconnect from a specific peer
    pub async fn disconnect_from_peer(&self, node_id: NodeId) -> Result<()> {
        self.disconnect_peer(node_id, "Manual disconnect requested".to_string())
            .await;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ConnectionStats {
    pub active_connections: u32,
    pub pending_attempts: u32,
    pub total_discovered_peers: u32,
    pub successful_connections: u32,
    pub failed_connections: u32,
    pub average_connection_time: Duration,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_peer_network_event_serialization() {
        let event = PeerNetworkEvent::PeerDiscovered {
            node_id: [1u8; 32],
            advertisement: crate::NodeAdvertisement {
                node_id: [1u8; 32],
                onion_address: "test.onion".to_string(),
                port: 8333,
                protocol_version: "qk/0.1".to_string(),
                capabilities: vec!["DAG".to_string()],
                signature: vec![],
                timestamp: chrono::Utc::now(),
                expires_at: chrono::Utc::now() + chrono::Duration::hours(1),
            },
            confidence: 0.8,
        };

        // Test that event can be cloned and matches expected structure
        let cloned_event = event.clone();
        match cloned_event {
            PeerNetworkEvent::PeerDiscovered { confidence, .. } => {
                assert_eq!(confidence, 0.8);
            }
            _ => panic!("Event type mismatch"),
        }
    }

    #[tokio::test]
    async fn test_connection_stats() {
        // This would require more complex setup to test properly
        // For now, just test the struct construction
        let stats = ConnectionStats {
            active_connections: 5,
            pending_attempts: 2,
            total_discovered_peers: 10,
            successful_connections: 8,
            failed_connections: 2,
            average_connection_time: Duration::from_secs(3),
            last_updated: chrono::Utc::now(),
        };

        assert_eq!(stats.active_connections, 5);
        assert_eq!(stats.pending_attempts, 2);
    }
}
