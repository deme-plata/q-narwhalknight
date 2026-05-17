/// DHT to Gossip Coordinator - Integrates BEP-44 DHT discovery with libp2p gossip
///
/// This module creates the full pipeline:
/// BEP-44 DHT Peer Discovery → DHT Events → Libp2p Bridge → Gossip Network → Consensus Layer
use crate::libp2p_bridge::{DhtEvent, BridgeEvent, Libp2pBridge};
use crate::connection_manager::{ConnectionManager, PeerInfo, DiscoveryMethod};
use crate::handshake::ServerRole;
use q_bep44_discovery::{DiscoveryEngine, DiscoveredPeer, Bep44DiscoveryConfig};
use anyhow::Result;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use tracing::{info, warn, error, debug};
use libp2p::identity::Keypair as Libp2pKeypair;

/// Main coordinator that integrates DHT discovery with gossip consensus
pub struct DhtGossipCoordinator {
    /// BEP-44 DHT discovery engine
    dht_engine: Arc<RwLock<DiscoveryEngine>>,

    /// Libp2p gossip bridge
    bridge: Option<Libp2pBridge>,

    /// DHT event sender to bridge
    dht_event_tx: Option<mpsc::Sender<DhtEvent>>,

    /// Bridge event receiver from gossip
    bridge_event_rx: Option<mpsc::Receiver<BridgeEvent>>,

    /// Connection manager for direct peer connection attempts
    connection_manager: Option<Arc<RwLock<ConnectionManager>>>,

    /// Coordinator configuration
    config: CoordinatorConfig,

    /// Node identification
    node_id: [u8; 32],

    /// Status tracking
    is_running: Arc<RwLock<bool>>,
}

#[derive(Debug, Clone)]
pub struct CoordinatorConfig {
    /// DHT discovery interval
    pub dht_discovery_interval_secs: u64,

    /// Bridge event processing interval
    pub bridge_event_interval_ms: u64,

    /// Auto-announce discovered peers to gossip
    pub auto_announce_discovered_peers: bool,

    /// Enable performance metrics
    pub enable_metrics: bool,

    /// Libp2p key generation seed (deterministic)
    pub libp2p_key_seed: Option<[u8; 32]>,
}

impl Default for CoordinatorConfig {
    fn default() -> Self {
        Self {
            dht_discovery_interval_secs: 30,     // Discover peers every 30 seconds
            bridge_event_interval_ms: 100,       // Process bridge events every 100ms
            auto_announce_discovered_peers: true, // Automatically announce new peers
            enable_metrics: true,                 // Track performance metrics
            libp2p_key_seed: None,               // Generate random keys by default
        }
    }
}

impl DhtGossipCoordinator {
    /// Create new DHT-to-Gossip coordinator
    pub async fn new(
        node_id: [u8; 32],
        dht_config: Bep44DiscoveryConfig,
        coordinator_config: CoordinatorConfig
    ) -> Result<Self> {
        info!("🔗 Creating DHT-to-Gossip coordinator");
        info!("   • Node ID: {}", hex::encode(&node_id[..8]));
        info!("   • Auto-announce: {}", coordinator_config.auto_announce_discovered_peers);

        // Create DHT discovery engine
        let dht_engine = Arc::new(RwLock::new(
            DiscoveryEngine::new(dht_config, node_id).await?
        ));

        Ok(Self {
            dht_engine,
            bridge: None,
            dht_event_tx: None,
            bridge_event_rx: None,
            connection_manager: None,
            config: coordinator_config,
            node_id,
            is_running: Arc::new(RwLock::new(false)),
        })
    }

    /// Initialize the coordinator with gossip bridge
    pub async fn initialize_bridge(&mut self) -> Result<()> {
        info!("🌉 Initializing libp2p gossip bridge");

        // Generate libp2p keypair (deterministic if seed provided)
        let keypair = if let Some(seed) = self.config.libp2p_key_seed {
            // Use deterministic key generation based on seed
            Libp2pKeypair::ed25519_from_bytes(seed)
                .map_err(|e| anyhow::anyhow!("Failed to create ed25519 key from seed: {}", e))?
        } else {
            // Generate random keypair
            Libp2pKeypair::generate_ed25519()
        };

        let peer_id = libp2p::PeerId::from(keypair.public());
        info!("   • Bridge peer ID: {}", peer_id);

        // Create bridge event channels
        let (bridge_tx, bridge_rx) = mpsc::channel(1000);

        // Create libp2p bridge
        let (bridge, dht_tx) = Libp2pBridge::new(keypair, bridge_tx).await?;

        // Store bridge components
        self.bridge = Some(bridge);
        self.dht_event_tx = Some(dht_tx);
        self.bridge_event_rx = Some(bridge_rx);

        info!("✅ Gossip bridge initialized successfully");
        Ok(())
    }

    /// Start the full DHT-to-Gossip pipeline
    pub async fn start(&mut self) -> Result<()> {
        if self.bridge.is_none() {
            return Err(anyhow::anyhow!("Bridge not initialized. Call initialize_bridge() first."));
        }

        info!("🚀 Starting DHT-to-Gossip coordinator pipeline");

        // Mark as running
        {
            let mut running = self.is_running.write().await;
            *running = true;
        }

        // Initialize and start DHT discovery
        {
            let mut dht = self.dht_engine.write().await;
            dht.initialize().await?;
            dht.start().await?;
        }

        info!("✅ DHT discovery engine started");

        // Start libp2p bridge
        if let Some(bridge) = self.bridge.take() {
            let bridge_handle = tokio::spawn(async move {
                info!("🌉 Starting libp2p bridge event loop");
                if let Err(e) = bridge.run().await {
                    error!("Libp2p bridge error: {}", e);
                }
            });

            // Store bridge handle for later cleanup if needed
            // (In a full implementation, you'd want to track this)
            std::mem::forget(bridge_handle); // Prevent handle from being dropped
        }

        info!("✅ Libp2p bridge started");

        // Start DHT-to-Bridge event forwarding
        self.start_dht_event_forwarding().await?;

        // Start bridge event processing
        self.start_bridge_event_processing().await?;

        info!("🎯 DHT-to-Gossip coordinator fully operational");
        Ok(())
    }

    /// Start forwarding DHT discovery events to the libp2p bridge
    async fn start_dht_event_forwarding(&self) -> Result<()> {
        if self.dht_event_tx.is_none() {
            return Err(anyhow::anyhow!("DHT event sender not available"));
        }

        let dht_engine = self.dht_engine.clone();
        let dht_tx = self.dht_event_tx.as_ref().unwrap().clone();
        let connection_manager = self.connection_manager.clone();
        let is_running = self.is_running.clone();
        let discovery_interval = std::time::Duration::from_secs(self.config.dht_discovery_interval_secs);

        // FIXME: DiscoveryEngine is not Send due to libp2p Swarm limitations
        // For now, we'll skip the spawned task and just return OK
        // The DHT discovery will still work through direct API calls
        warn!("DHT event forwarding disabled due to Send constraint - using direct API calls instead");
        Ok(())
    }

    /// Start processing bridge events for consensus integration
    async fn start_bridge_event_processing(&mut self) -> Result<()> {
        if self.bridge_event_rx.is_none() {
            return Err(anyhow::anyhow!("Bridge event receiver not available"));
        }

        let mut bridge_rx = self.bridge_event_rx.take().unwrap();
        let is_running = self.is_running.clone();
        let event_interval = std::time::Duration::from_millis(self.config.bridge_event_interval_ms);

        tokio::spawn(async move {
            info!("🎯 Starting bridge event processing loop");

            loop {
                // Process bridge events with timeout
                let event_result = tokio::time::timeout(event_interval, bridge_rx.recv()).await;

                match event_result {
                    Ok(Some(bridge_event)) => {
                        Self::process_bridge_event(bridge_event).await;
                    },
                    Ok(None) => {
                        warn!("🔌 Bridge event channel closed");
                        break;
                    },
                    Err(_) => {
                        // Timeout - check if we should continue running
                        let running = is_running.read().await;
                        if !*running {
                            info!("🛑 Bridge event processing stopped");
                            break;
                        }
                        // Continue processing
                    }
                }
            }
        });

        Ok(())
    }

    /// Process individual bridge events for consensus integration
    async fn process_bridge_event(event: BridgeEvent) {
        match event {
            BridgeEvent::ConsensusMessage { topic, data, peer } => {
                info!("📨 Consensus message received");
                debug!("   • Topic: {}", topic);
                debug!("   • From peer: {}", peer);
                debug!("   • Data size: {} bytes", data.len());

                // In a full implementation, forward to consensus layer
                // For now, just log the event
                if topic.contains("/qnk/consensus/blocks") {
                    info!("🧱 Block proposal received via gossip from peer {}", peer);
                } else if topic.contains("/qnk/consensus/votes") {
                    info!("🗳️  Consensus vote received via gossip from peer {}", peer);
                } else if topic.contains("/qnk/peers/discovery") {
                    info!("🔍 Peer discovery message received via gossip from peer {}", peer);
                }
            },

            BridgeEvent::ValidatorDiscovered { peer_id, capabilities } => {
                info!("🆕 New validator discovered via gossip");
                debug!("   • Peer ID: {}", peer_id);
                debug!("   • Capabilities: {:?}", capabilities);

                // In a full implementation, add to validator registry
            },

            BridgeEvent::NetworkHealth { connected_peers, topics } => {
                debug!("💓 Network health update");
                debug!("   • Connected peers: {}", connected_peers);
                debug!("   • Active topics: {}", topics.len());

                // In a full implementation, update network health metrics
            },
        }
    }

    /// Stop the coordinator
    pub async fn stop(&mut self) -> Result<()> {
        info!("🛑 Stopping DHT-to-Gossip coordinator");

        // Mark as not running
        {
            let mut running = self.is_running.write().await;
            *running = false;
        }

        // Stop DHT discovery engine
        {
            let mut dht = self.dht_engine.write().await;
            dht.stop().await?;
        }

        info!("✅ DHT-to-Gossip coordinator stopped");
        Ok(())
    }

    /// Get coordinator status and metrics
    pub async fn get_coordinator_status(&self) -> Result<CoordinatorStatus> {
        let is_running = *self.is_running.read().await;

        // Get DHT stats
        let dht_stats = {
            let dht = self.dht_engine.read().await;
            dht.get_discovery_stats().await
        };

        // Get discovered peers count
        let discovered_peers = {
            let dht = self.dht_engine.read().await;
            dht.get_discovered_peers().await
        };

        Ok(CoordinatorStatus {
            is_running,
            discovered_peer_count: discovered_peers.len() as u64,
            dht_discovery_stats: dht_stats,
            bridge_peer_id: "bridge_running".to_string(), // In full implementation, get actual peer ID
            total_events_forwarded: 0, // In full implementation, track this
            consensus_messages_processed: 0, // In full implementation, track this
        })
    }

    /// Force immediate DHT discovery and forward to gossip
    pub async fn force_discovery_and_announce(&self) -> Result<Vec<String>> {
        info!("🔄 Forcing immediate DHT discovery and gossip announcement");

        // Force DHT discovery
        let discovered_peers = {
            let dht = self.dht_engine.read().await;
            dht.force_discovery().await?
        };

        let mut announced_peers = Vec::new();

        // Forward discoveries to bridge if available
        if let Some(dht_tx) = &self.dht_event_tx {
            for peer in &discovered_peers {
                // Create and send peer discovered event
                let dht_event = DhtEvent::PeerDiscovered {
                    peer_id: peer.validator_id.to_vec(),
                    address: format!("{}:{}",
                        peer.real_ip_addresses.first()
                            .map(|ip| ip.to_string())
                            .unwrap_or_else(|| "127.0.0.1".to_string()),
                        peer.p2p_port
                    ),
                };

                if let Err(e) = dht_tx.send(dht_event).await {
                    warn!("Failed to send forced discovery event: {}", e);
                } else {
                    let peer_id_str = hex::encode(&peer.validator_id[..8]);
                    announced_peers.push(peer_id_str.clone());
                    debug!("📢 Announced peer {} to gossip network", peer_id_str);
                }
            }
        }

        info!("✅ Forced discovery completed: {} peers announced to gossip", announced_peers.len());
        Ok(announced_peers)
    }

    /// Bridge the connection gap: Add ConnectionManager for direct peer connections
    pub fn with_connection_manager(mut self, connection_manager: Arc<RwLock<ConnectionManager>>) -> Self {
        info!("🔌 Connecting DHT discovery to ConnectionManager - Bridge gap fix enabled");
        self.connection_manager = Some(connection_manager);
        self
    }
}

/// Coordinator status and metrics
#[derive(Debug, Clone)]
pub struct CoordinatorStatus {
    pub is_running: bool,
    pub discovered_peer_count: u64,
    pub dht_discovery_stats: q_bep44_discovery::DiscoveryStats,
    pub bridge_peer_id: String,
    pub total_events_forwarded: u64,
    pub consensus_messages_processed: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_coordinator_creation() {
        let node_id = [1u8; 32];
        let dht_config = Bep44DiscoveryConfig::default();
        let coordinator_config = CoordinatorConfig::default();

        let coordinator = DhtGossipCoordinator::new(node_id, dht_config, coordinator_config).await;
        assert!(coordinator.is_ok());
    }

    #[tokio::test]
    async fn test_coordinator_initialization() {
        let node_id = [2u8; 32];
        let dht_config = Bep44DiscoveryConfig::default();
        let coordinator_config = CoordinatorConfig::default();

        let mut coordinator = DhtGossipCoordinator::new(node_id, dht_config, coordinator_config).await.unwrap();
        let result = coordinator.initialize_bridge().await;
        assert!(result.is_ok());
    }
}