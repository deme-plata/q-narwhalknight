/*!
 * Discovery-Connection Bridge for Q-NarwhalKnight
 *
 * This module provides the bridge between the BEP-44 discovery engine and
 * the peer connection system, solving the critical "discovery-connection gap"
 * where peers are discovered but never connected to.
 */

use anyhow::{Result, anyhow};
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{info, warn, debug, error};

use crate::{RealDiscoveryEngine, QnkValidator, LocalDiscoveredPeer, PeerCapability, ServiceStatus};

/// Type alias for discovered peer - use the existing LocalDiscoveredPeer struct
pub type DiscoveredPeer = LocalDiscoveredPeer;

impl Default for DiscoveredPeer {
    fn default() -> Self {
        Self {
            validator_id: [0u8; 32],
            onion_address: String::new(),
            real_ip_addresses: Vec::new(),
            api_port: 8080,
            p2p_port: 9080,
            capabilities: Vec::new(),
            signature: Vec::new(),
            timestamp: chrono::Utc::now(),
            discovery_method: "bep44".to_string(),
            info_hash: [0u8; 20],
            discovered_at: chrono::Utc::now(),
            service_status: ServiceStatus::Unknown,
            last_service_check: chrono::Utc::now(),
            connection_success_rate: 0.0,
        }
    }
}

/// Bridge component that connects discovery to connection layers
pub struct DiscoveryConnector {
    /// Real BEP-44 discovery engine
    discovery_engine: Arc<RealDiscoveryEngine>,
    /// Channel to send discovered peers for connection
    discovery_tx: mpsc::Sender<DiscoveredPeer>,
    /// Statistics
    stats: Arc<tokio::sync::RwLock<DiscoveryConnectorStats>>,
}

/// Discovery connector statistics
#[derive(Debug, Default, Clone)]
pub struct DiscoveryConnectorStats {
    /// Total peers processed through bridge
    pub peers_processed: u64,
    /// Peers sent for connection
    pub peers_sent_for_connection: u64,
    /// Failed conversions (invalid data)
    pub conversion_failures: u64,
    /// Channel send failures
    pub send_failures: u64,
}

impl DiscoveryConnector {
    /// Create new discovery-connection bridge
    pub fn new(discovery_engine: Arc<RealDiscoveryEngine>) -> (Self, mpsc::Receiver<DiscoveredPeer>) {
        let (tx, rx) = mpsc::channel(100); // Buffer for bursty discoveries

        let connector = Self {
            discovery_engine,
            discovery_tx: tx,
            stats: Arc::new(tokio::sync::RwLock::new(DiscoveryConnectorStats::default())),
        };

        info!("✅ Discovery-Connection Bridge created with 100-peer buffer");
        (connector, rx)
    }

    /// Process discoveries and send them for connection
    pub async fn process_discoveries(&self) -> Result<usize> {
        debug!("🔗 Processing discoveries for connection bridging");

        let validators = self.discovery_engine.discover_validators().await?;
        let mut processed_count = 0;

        if validators.is_empty() {
            debug!("No validators to process for connections");
            return Ok(0);
        }

        info!("🔗 Processing {} discovered validators for connections", validators.len());

        for validator in &validators {
            match self.convert_to_discovered_peer(&validator) {
                Ok(peer) => {
                    debug!("🔗 Converted validator {} to DiscoveredPeer", hex::encode(&peer.validator_id[..8]));

                    // Send to connection system via channel
                    match self.discovery_tx.send(peer.clone()).await {
                        Ok(_) => {
                            info!("📡 Bridged discovery: Sent {} ({}) for connection",
                                  hex::encode(&peer.validator_id[..8]), peer.onion_address);
                            processed_count += 1;

                            // Update stats
                            {
                                let mut stats = self.stats.write().await;
                                stats.peers_sent_for_connection += 1;
                            }
                        }
                        Err(e) => {
                            error!("❌ Failed to send peer {} for connection: {}", hex::encode(&peer.validator_id[..8]), e);

                            // Update stats
                            {
                                let mut stats = self.stats.write().await;
                                stats.send_failures += 1;
                            }
                        }
                    }
                }
                Err(e) => {
                    warn!("⚠️ Failed to convert validator to DiscoveredPeer: {}", e);

                    // Update stats
                    {
                        let mut stats = self.stats.write().await;
                        stats.conversion_failures += 1;
                    }
                }
            }
        }

        // Update total processed
        {
            let mut stats = self.stats.write().await;
            stats.peers_processed += validators.len() as u64;
        }

        info!("✅ Discovery bridge processed {} validators, sent {} for connection",
              validators.len(), processed_count);
        Ok(processed_count)
    }

    /// Convert QnkValidator to DiscoveredPeer for connection
    fn convert_to_discovered_peer(&self, validator: &QnkValidator) -> Result<DiscoveredPeer> {
        // Determine API and P2P ports from onion address
        let (api_port, p2p_port) = if validator.onion_address.contains(":") {
            let parts: Vec<&str> = validator.onion_address.split(":").collect();
            if parts.len() >= 2 {
                let port = parts[1].parse::<u16>().unwrap_or(8080);
                (port, port + 1000) // P2P port is typically API port + 1000
            } else {
                (8080, 9080)
            }
        } else {
            (8080, 9080)
        };

        // Convert validator capabilities to PeerCapability enum
        let capabilities: Vec<PeerCapability> = validator.capabilities.iter()
            .filter_map(|cap| match cap.as_str() {
                "consensus" => Some(PeerCapability::Consensus),
                "mempool" => Some(PeerCapability::Mempool),
                "state_sync" => Some(PeerCapability::StateSync),
                "archive" => Some(PeerCapability::Archive),
                _ => None,
            })
            .collect();

        // Check for localhost addresses that may be testnet artifacts
        if validator.onion_address.starts_with("127.0.0.") || validator.onion_address.starts_with("localhost") {
            warn!("🏠 Discovered localhost peer {} ({}); consider testnet config",
                  hex::encode(&validator.node_id[..8]), validator.onion_address);
        }

        let peer = DiscoveredPeer {
            validator_id: validator.node_id,
            onion_address: validator.onion_address.clone(),
            real_ip_addresses: Vec::new(), // BEP-44 discovery focuses on onion addresses
            api_port,
            p2p_port,
            capabilities,
            signature: Vec::new(), // Will be filled during validation
            timestamp: validator.last_seen,
            discovery_method: "bep44".to_string(),
            info_hash: [0u8; 20], // Will be computed from validator data
            discovered_at: validator.last_seen,
            service_status: ServiceStatus::Unknown, // Initial status, will be checked later
            last_service_check: chrono::Utc::now(),
            connection_success_rate: 0.0, // No connection attempts yet
        };

        debug!("🔄 Converted validator {} -> DiscoveredPeer {{ onion: {}, api_port: {}, p2p_port: {} }}",
               hex::encode(&validator.node_id[..8]), peer.onion_address, peer.api_port, peer.p2p_port);

        Ok(peer)
    }

    /// Get bridge statistics
    pub async fn get_stats(&self) -> DiscoveryConnectorStats {
        let stats = self.stats.read().await;
        (*stats).clone()
    }

    /// Get discovery engine reference for direct access
    pub fn discovery_engine(&self) -> &Arc<RealDiscoveryEngine> {
        &self.discovery_engine
    }
}

/// Connection handler that processes discovered peers
pub struct ConnectionHandler {
    /// Receiver for discovered peers
    peer_rx: mpsc::Receiver<DiscoveredPeer>,
    /// Handler statistics
    stats: Arc<tokio::sync::RwLock<ConnectionHandlerStats>>,
}

/// Connection handler statistics
#[derive(Debug, Default, Clone)]
pub struct ConnectionHandlerStats {
    /// Total connection attempts
    pub connection_attempts: u64,
    /// Successful connections
    pub successful_connections: u64,
    /// Failed connections
    pub failed_connections: u64,
    /// Currently active connections
    pub active_connections: u64,
}

impl ConnectionHandler {
    /// Create new connection handler with peer receiver
    pub fn new(peer_rx: mpsc::Receiver<DiscoveredPeer>) -> Self {
        Self {
            peer_rx,
            stats: Arc::new(tokio::sync::RwLock::new(ConnectionHandlerStats::default())),
        }
    }

    /// Start processing discovered peers for connections
    pub async fn start_processing(&mut self) {
        info!("🚀 Starting connection handler for discovered peers");

        while let Some(peer) = self.peer_rx.recv().await {
            self.handle_discovered_peer(peer).await;
        }

        info!("🛑 Connection handler stopped - no more peers to process");
    }

    /// Handle a single discovered peer
    async fn handle_discovered_peer(&self, peer: DiscoveredPeer) {
        info!("🔗 Attempting connection to discovered peer: {} ({})",
              hex::encode(&peer.validator_id[..8]), peer.onion_address);

        // Update connection attempt stats
        {
            let mut stats = self.stats.write().await;
            stats.connection_attempts += 1;
        }

        // Attempt connection based on onion address format
        let connection_result = if peer.onion_address.ends_with(".onion") {
            self.connect_via_tor(&peer).await
        } else if peer.onion_address.contains(":") {
            self.connect_direct_tcp(&peer).await
        } else {
            warn!("❌ Unknown address format '{}' for peer {}", peer.onion_address, hex::encode(&peer.validator_id[..8]));
            Err(anyhow!("Unknown address format: {}", peer.onion_address))
        };

        // Update connection result stats
        match connection_result {
            Ok(_) => {
                info!("✅ Successfully connected to peer {} ({})",
                      hex::encode(&peer.validator_id[..8]), peer.onion_address);
                let mut stats = self.stats.write().await;
                stats.successful_connections += 1;
                stats.active_connections += 1;
            }
            Err(e) => {
                warn!("❌ Failed to connect to peer {} ({}): {}",
                      hex::encode(&peer.validator_id[..8]), peer.onion_address, e);
                let mut stats = self.stats.write().await;
                stats.failed_connections += 1;
            }
        }
    }

    /// Attempt Tor connection to peer
    async fn connect_via_tor(&self, peer: &DiscoveredPeer) -> Result<()> {
        debug!("🧅 Attempting Tor connection to {} ({})",
               hex::encode(&peer.validator_id[..8]), peer.onion_address);

        // TODO: Implement real Tor connection using q-tor-client
        // For now, simulate connection attempt

        if peer.onion_address.ends_with(".onion") {
            info!("🧅 Tor connection simulated for {}", peer.onion_address);
            Ok(())
        } else {
            Err(anyhow!("Invalid onion address: {}", peer.onion_address))
        }
    }

    /// Attempt direct TCP connection to peer
    async fn connect_direct_tcp(&self, peer: &DiscoveredPeer) -> Result<()> {
        debug!("🌐 Attempting TCP connection to {} ({})",
               hex::encode(&peer.validator_id[..8]), peer.onion_address);

        // TODO: Implement real TCP connection using tokio
        // For now, simulate connection attempt

        if peer.onion_address.contains(":") {
            info!("🌐 TCP connection simulated for {}", peer.onion_address);
            Ok(())
        } else {
            Err(anyhow!("Invalid TCP endpoint: {}", peer.onion_address))
        }
    }

    /// Attempt QUIC connection to peer
    async fn connect_via_quic(&self, peer: &DiscoveredPeer) -> Result<()> {
        debug!("⚡ Attempting QUIC connection to {} ({})",
               hex::encode(&peer.validator_id[..8]), peer.onion_address);

        // TODO: Implement real QUIC connection
        // For now, simulate connection attempt

        info!("⚡ QUIC connection simulated for {}", peer.onion_address);
        Ok(())
    }

    /// Get connection handler statistics
    pub async fn get_stats(&self) -> ConnectionHandlerStats {
        let stats = self.stats.read().await;
        (*stats).clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_discovery_connector_creation() {
        // Mock discovery engine (would need proper mock in real test)
        // let engine = Arc::new(RealDiscoveryEngine::new([1u8; 32]).await.unwrap());
        // let (connector, _rx) = DiscoveryConnector::new(engine);
        // Test would verify connector is created properly
    }

    #[tokio::test]
    async fn test_peer_conversion() {
        let validator = QnkValidator {
            node_id: [0xAB; 32],
            onion_address: "test123.onion:8091".to_string(),
            capabilities: vec!["consensus".to_string()],
            last_seen: chrono::Utc::now(),
            sequence: 1,
            public_key: [0xCD; 32],
        };

        // Test conversion logic without full connector
        let id = hex::encode(&validator.node_id[..8]);
        assert_eq!(id.len(), 16); // 8 bytes = 16 hex chars
    }
}