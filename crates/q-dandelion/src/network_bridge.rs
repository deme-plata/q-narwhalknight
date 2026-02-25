//! Network Bridge for Dandelion++ Protocol
//!
//! Integrates Dandelion++ with q-network gossipsub for message propagation.
//! Provides priority-aware message publishing and peer discovery.
//!
//! 🌻 v2.5.0-beta: REAL LIBP2P INTEGRATION
//! This module now bridges Dandelion++ commands to q_network::NetworkCommand
//! for actual gossipsub message propagation.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, error, info, warn};

use crate::{DandelionMessage, DandelionPhase};

/// Message priority levels for gossipsub queue
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MessagePriority {
    /// Critical priority - never throttled
    Critical = 100,
    /// High priority - minimal throttling
    High = 75,
    /// Normal priority - standard rate limiting
    Normal = 50,
    /// Low priority - aggressive throttling
    Low = 25,
}

/// Network bridge configuration
#[derive(Debug, Clone)]
pub struct NetworkBridgeConfig {
    /// Network ID for topic construction
    pub network_id: String,
    /// Topic for fluff broadcast (transactions)
    pub fluff_topic: String,
    /// Topic for stem relay (direct P2P)
    pub stem_topic: String,
    /// Maximum messages per second for fluff
    pub fluff_rate_limit: u32,
    /// Enable Tor for all messages
    pub tor_enabled: bool,
    /// Peer refresh interval
    pub peer_refresh_interval: Duration,
}

impl Default for NetworkBridgeConfig {
    fn default() -> Self {
        Self {
            network_id: "mainnet-genesis".to_string(),
            fluff_topic: "/qnk/mainnet2026/mempool-txs".to_string(),
            stem_topic: "/qnk/mainnet2026/dandelion-stem".to_string(),
            fluff_rate_limit: 100,
            tor_enabled: true,
            peer_refresh_interval: Duration::from_secs(30),
        }
    }
}

/// Peer information for stem routing
#[derive(Debug, Clone)]
pub struct PeerInfo {
    /// Peer ID (NodeId)
    pub peer_id: [u8; 32],
    /// Multiaddress for direct connection
    pub multiaddr: String,
    /// Onion address if available
    pub onion_address: Option<String>,
    /// Last seen timestamp
    pub last_seen: Instant,
    /// Reputation score (0.0 - 1.0)
    pub reputation: f64,
    /// Is this peer a stem candidate
    pub stem_candidate: bool,
}

/// Network command for gossipsub interaction
#[derive(Debug, Clone)]
pub enum NetworkCommand {
    /// Publish message via gossipsub (fluff)
    PublishFluff {
        topic: String,
        message: Vec<u8>,
        priority: MessagePriority,
    },
    /// Send message to specific peer (stem)
    SendStem {
        peer_id: [u8; 32],
        message: Vec<u8>,
        via_tor: bool,
    },
    /// Request connected peers
    GetPeers,
    /// Subscribe to topic
    Subscribe { topic: String },
}

/// Network event from gossipsub
#[derive(Debug, Clone)]
pub enum NetworkEvent {
    /// Received Dandelion message
    MessageReceived {
        from_peer: [u8; 32],
        message: DandelionMessage,
    },
    /// Peer connected
    PeerConnected { peer: PeerInfo },
    /// Peer disconnected
    PeerDisconnected { peer_id: [u8; 32] },
    /// Publish confirmed
    PublishConfirmed { message_id: [u8; 32] },
    /// Publish failed
    PublishFailed { message_id: [u8; 32], error: String },
}

/// Network bridge for Dandelion++ integration
pub struct NetworkBridge {
    /// Configuration
    config: NetworkBridgeConfig,
    /// Command sender to network layer
    command_tx: mpsc::UnboundedSender<NetworkCommand>,
    /// Connected peers
    peers: Arc<RwLock<HashMap<[u8; 32], PeerInfo>>>,
    /// Stem peer candidates (subset of peers)
    stem_candidates: Arc<RwLock<Vec<[u8; 32]>>>,
    /// Message counter for stats
    messages_sent: AtomicU64,
    /// Fluff messages counter
    fluff_count: AtomicU64,
    /// Stem messages counter
    stem_count: AtomicU64,
    /// Last peer refresh
    last_peer_refresh: Arc<RwLock<Instant>>,
}

impl NetworkBridge {
    /// Create new network bridge
    pub fn new(
        config: NetworkBridgeConfig,
        command_tx: mpsc::UnboundedSender<NetworkCommand>,
    ) -> Self {
        Self {
            config,
            command_tx,
            peers: Arc::new(RwLock::new(HashMap::new())),
            stem_candidates: Arc::new(RwLock::new(Vec::new())),
            messages_sent: AtomicU64::new(0),
            fluff_count: AtomicU64::new(0),
            stem_count: AtomicU64::new(0),
            last_peer_refresh: Arc::new(RwLock::new(Instant::now())),
        }
    }

    /// Broadcast message in fluff phase via gossipsub
    pub async fn broadcast_fluff(&self, message: &DandelionMessage) -> Result<()> {
        debug!(
            "Broadcasting fluff message {} to gossipsub",
            hex::encode(message.id)
        );

        // Serialize message
        let message_bytes = bincode::serialize(message)
            .context("Failed to serialize Dandelion message")?;

        // Determine priority based on message type
        let priority = if message.hop_count == 0 {
            // First broadcast - high priority
            MessagePriority::High
        } else {
            MessagePriority::Normal
        };

        // Send to network layer
        self.command_tx
            .send(NetworkCommand::PublishFluff {
                topic: self.config.fluff_topic.clone(),
                message: message_bytes,
                priority,
            })
            .map_err(|e| anyhow::anyhow!("Failed to send network command: {}", e))?;

        self.messages_sent.fetch_add(1, Ordering::Relaxed);
        self.fluff_count.fetch_add(1, Ordering::Relaxed);

        info!(
            "Fluff broadcast initiated for message {}",
            hex::encode(message.id)
        );

        Ok(())
    }

    /// Send message to specific peer in stem phase
    pub async fn send_stem(
        &self,
        peer_id: &[u8; 32],
        message: &DandelionMessage,
    ) -> Result<()> {
        debug!(
            "Sending stem message {} to peer {}",
            hex::encode(message.id),
            hex::encode(peer_id)
        );

        // Serialize message
        let message_bytes = bincode::serialize(message)
            .context("Failed to serialize Dandelion message")?;

        // Send via Tor if enabled
        self.command_tx
            .send(NetworkCommand::SendStem {
                peer_id: *peer_id,
                message: message_bytes,
                via_tor: self.config.tor_enabled,
            })
            .map_err(|e| anyhow::anyhow!("Failed to send stem command: {}", e))?;

        self.messages_sent.fetch_add(1, Ordering::Relaxed);
        self.stem_count.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    /// Handle network event
    pub async fn handle_event(&self, event: NetworkEvent) -> Option<DandelionMessage> {
        match event {
            NetworkEvent::MessageReceived { from_peer, message } => {
                debug!(
                    "Received Dandelion message {} from peer {}",
                    hex::encode(message.id),
                    hex::encode(from_peer)
                );
                Some(message)
            }
            NetworkEvent::PeerConnected { peer } => {
                debug!("Peer connected: {}", hex::encode(peer.peer_id));
                let mut peers = self.peers.write().await;
                peers.insert(peer.peer_id, peer.clone());

                // Update stem candidates if this is a good peer
                if peer.stem_candidate && peer.reputation > 0.5 {
                    let mut candidates = self.stem_candidates.write().await;
                    if !candidates.contains(&peer.peer_id) {
                        candidates.push(peer.peer_id);
                    }
                }
                None
            }
            NetworkEvent::PeerDisconnected { peer_id } => {
                debug!("Peer disconnected: {}", hex::encode(peer_id));
                let mut peers = self.peers.write().await;
                peers.remove(&peer_id);

                let mut candidates = self.stem_candidates.write().await;
                candidates.retain(|id| id != &peer_id);
                None
            }
            NetworkEvent::PublishConfirmed { message_id } => {
                debug!("Publish confirmed: {}", hex::encode(message_id));
                None
            }
            NetworkEvent::PublishFailed { message_id, error } => {
                warn!(
                    "Publish failed for {}: {}",
                    hex::encode(message_id),
                    error
                );
                None
            }
        }
    }

    /// Get stem peer candidates
    pub async fn get_stem_candidates(&self) -> Vec<PeerInfo> {
        let candidates = self.stem_candidates.read().await;
        let peers = self.peers.read().await;

        candidates
            .iter()
            .filter_map(|id| peers.get(id).cloned())
            .collect()
    }

    /// Select random stem peer
    pub async fn select_stem_peer(&self) -> Option<PeerInfo> {
        let candidates = self.get_stem_candidates().await;
        if candidates.is_empty() {
            return None;
        }

        // Use random selection (quantum RNG would be integrated at higher level)
        use rand::Rng;
        let mut rng = rand::rng();
        let index = rng.random_range(0..candidates.len());
        candidates.get(index).cloned()
    }

    /// Update peer information
    pub async fn update_peer(&self, peer: PeerInfo) {
        let mut peers = self.peers.write().await;
        peers.insert(peer.peer_id, peer);
    }

    /// Get connected peer count
    pub async fn peer_count(&self) -> usize {
        self.peers.read().await.len()
    }

    /// Get stem candidate count
    pub async fn stem_candidate_count(&self) -> usize {
        self.stem_candidates.read().await.len()
    }

    /// Get statistics
    pub fn get_stats(&self) -> NetworkBridgeStats {
        NetworkBridgeStats {
            messages_sent: self.messages_sent.load(Ordering::Relaxed),
            fluff_messages: self.fluff_count.load(Ordering::Relaxed),
            stem_messages: self.stem_count.load(Ordering::Relaxed),
        }
    }

    /// Subscribe to Dandelion topics
    pub async fn subscribe_topics(&self) -> Result<()> {
        // Subscribe to fluff topic
        self.command_tx
            .send(NetworkCommand::Subscribe {
                topic: self.config.fluff_topic.clone(),
            })
            .map_err(|e| anyhow::anyhow!("Failed to subscribe: {}", e))?;

        // Subscribe to stem topic
        self.command_tx
            .send(NetworkCommand::Subscribe {
                topic: self.config.stem_topic.clone(),
            })
            .map_err(|e| anyhow::anyhow!("Failed to subscribe: {}", e))?;

        info!(
            "Subscribed to Dandelion topics: {}, {}",
            self.config.fluff_topic, self.config.stem_topic
        );

        Ok(())
    }

    /// Get fluff topic
    pub fn fluff_topic(&self) -> &str {
        &self.config.fluff_topic
    }

    /// Get stem topic
    pub fn stem_topic(&self) -> &str {
        &self.config.stem_topic
    }
}

/// Statistics from network bridge
#[derive(Debug, Clone)]
pub struct NetworkBridgeStats {
    pub messages_sent: u64,
    pub fluff_messages: u64,
    pub stem_messages: u64,
}

/// Helper to create topic string
pub fn create_topic(network_id: &str, suffix: &str) -> String {
    format!("/qnk/{}/{}", network_id, suffix)
}

// ================================================================================
// 🌻 v2.5.0-beta: LIBP2P BRIDGE - Connects Dandelion++ to q_network gossipsub
// ================================================================================

/// Bridge result for command processing
#[derive(Debug)]
pub enum BridgeResult {
    /// Command processed successfully
    Success,
    /// Command failed
    Failed(String),
    /// Channel closed
    ChannelClosed,
}

/// 🌻 v2.5.0-beta: Spawn a bridge task that forwards Dandelion commands to libp2p
///
/// This function creates a task that:
/// 1. Receives NetworkCommand messages from the Dandelion layer
/// 2. Translates them to q_network::NetworkCommand format
/// 3. Forwards to the libp2p command channel
///
/// # Arguments
/// * `dandelion_rx` - Receiver for Dandelion NetworkCommand messages
/// * `libp2p_tx` - Sender for q_network::NetworkCommand messages
/// * `network_id` - Network ID for topic construction (e.g., "mainnet")
///
/// # Returns
/// A JoinHandle for the bridge task
pub fn spawn_libp2p_bridge(
    mut dandelion_rx: mpsc::UnboundedReceiver<NetworkCommand>,
    libp2p_tx: mpsc::UnboundedSender<q_network::NetworkCommand>,
    network_id: String,
) -> tokio::task::JoinHandle<()> {
    info!("🌻 Starting Dandelion++ → libp2p bridge for network {}", network_id);

    tokio::spawn(async move {
        let mempool_topic = create_topic(&network_id, "mempool-txs");
        let mut commands_processed: u64 = 0;
        let mut commands_failed: u64 = 0;

        loop {
            match dandelion_rx.recv().await {
                Some(cmd) => {
                    let result = process_dandelion_command(
                        cmd,
                        &libp2p_tx,
                        &mempool_topic,
                    ).await;

                    match result {
                        BridgeResult::Success => {
                            commands_processed += 1;
                            if commands_processed % 100 == 0 {
                                debug!(
                                    "🌻 Dandelion bridge: {} commands processed, {} failed",
                                    commands_processed, commands_failed
                                );
                            }
                        }
                        BridgeResult::Failed(e) => {
                            commands_failed += 1;
                            warn!("🌻 Dandelion bridge command failed: {}", e);
                        }
                        BridgeResult::ChannelClosed => {
                            error!("🌻 Dandelion bridge: libp2p channel closed, shutting down");
                            break;
                        }
                    }
                }
                None => {
                    info!("🌻 Dandelion bridge: command channel closed, shutting down");
                    break;
                }
            }
        }

        info!(
            "🌻 Dandelion bridge shutdown: {} commands processed, {} failed",
            commands_processed, commands_failed
        );
    })
}

/// Process a single Dandelion command and forward to libp2p
async fn process_dandelion_command(
    cmd: NetworkCommand,
    libp2p_tx: &mpsc::UnboundedSender<q_network::NetworkCommand>,
    mempool_topic: &str,
) -> BridgeResult {
    match cmd {
        NetworkCommand::PublishFluff { topic, message, priority: _ } => {
            // Fluff phase: Broadcast transaction via gossipsub
            // Use the transaction topic (mempool-txs) for fluff
            let tx_topic = if topic.is_empty() {
                mempool_topic.to_string()
            } else {
                topic
            };

            // Compute transaction hash from message
            let tx_hash = {
                use blake3::Hasher;
                let mut hasher = Hasher::new();
                hasher.update(&message);
                hex::encode(&hasher.finalize().as_bytes()[..16])
            };

            debug!("🌻 Forwarding fluff message to gossipsub (topic: {}, hash: {})", tx_topic, tx_hash);

            if libp2p_tx.send(q_network::NetworkCommand::PublishTransaction {
                topic: tx_topic,
                tx_bytes: message,
                tx_hash,
            }).is_err() {
                return BridgeResult::ChannelClosed;
            }

            BridgeResult::Success
        }

        NetworkCommand::SendStem { peer_id, message, via_tor } => {
            // Stem phase: Send to specific peer
            // For stem, we can't use gossipsub directly - we need Tor relay
            // This is handled by TorBridge, so just log here
            if via_tor {
                debug!(
                    "🌻 Stem relay via Tor to peer {} ({} bytes)",
                    hex::encode(&peer_id[..8]),
                    message.len()
                );
                // Stem relay is handled by TorBridge.send_via_tor(), not gossipsub
                // This command type indicates Dandelion should handle it internally
            } else {
                // Non-Tor stem: Could use direct P2P if available
                debug!(
                    "🌻 Non-Tor stem relay to peer {} ({} bytes) - fallback to fluff",
                    hex::encode(&peer_id[..8]),
                    message.len()
                );
                // Fall back to gossipsub broadcast if no Tor
                let tx_hash = hex::encode(&peer_id[..8]);
                if libp2p_tx.send(q_network::NetworkCommand::PublishTransaction {
                    topic: mempool_topic.to_string(),
                    tx_bytes: message,
                    tx_hash,
                }).is_err() {
                    return BridgeResult::ChannelClosed;
                }
            }
            BridgeResult::Success
        }

        NetworkCommand::GetPeers => {
            // Peer list request - this would be handled differently
            // The peer list comes from the swarm, not via command
            debug!("🌻 GetPeers request (peers obtained from swarm state)");
            BridgeResult::Success
        }

        NetworkCommand::Subscribe { topic } => {
            // Subscribe to topic - gossipsub subscriptions are managed at swarm level
            debug!("🌻 Subscribe request for topic: {} (managed at swarm level)", topic);
            BridgeResult::Success
        }
    }
}

/// 🌻 v2.5.0-beta: Create a paired channel for Dandelion↔libp2p bridge
///
/// Returns (sender for Dandelion to use, receiver for bridge task)
pub fn create_bridge_channel() -> (
    mpsc::UnboundedSender<NetworkCommand>,
    mpsc::UnboundedReceiver<NetworkCommand>,
) {
    mpsc::unbounded_channel()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_topic() {
        let topic = create_topic("testnet", "mempool-txs");
        assert_eq!(topic, "/qnk/testnet/mempool-txs");
    }

    #[test]
    fn test_network_bridge_config_default() {
        let config = NetworkBridgeConfig::default();
        assert!(config.tor_enabled);
        assert_eq!(config.fluff_rate_limit, 100);
    }
}
