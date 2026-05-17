//! 🎻 Resonance Protocol Handler
//!
//! This module bridges the Quillon Resonance Consensus with libp2p gossipsub,
//! enabling resonance states to propagate across the network like sound waves.
//!
//! Philosophy: We don't broadcast votes - we broadcast vibrations.
//! The network is not a parliament - it's a symphony.

use anyhow::Context; // ✅ v0.9.98-beta: For error context in fail-fast pattern
use libp2p::gossipsub::{IdentTopic, Message, MessageId, Topic};
use q_resonance::{
    deserialize_resonance_message, serialize_resonance_message, ResonanceCoordinator,
    ResonanceMessage, RESONANCE_PROTOCOL,
};
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

/// 🎻 Resonance topic for libp2p gossipsub
pub fn resonance_topic() -> IdentTopic {
    IdentTopic::new(RESONANCE_PROTOCOL)
}

/// 🎻 Resonance Protocol Handler
///
/// Philosophy: This is the bridge between the distributed symphony (gossipsub)
/// and the resonance coordinator (consensus engine). It translates network
/// messages into resonance states and broadcasts local vibrations to peers.
pub struct ResonanceProtocolHandler {
    /// Coordinator for processing resonance consensus
    coordinator: Arc<ResonanceCoordinator>,

    /// Channel to receive messages from coordinator for broadcasting
    broadcast_rx: mpsc::UnboundedReceiver<ResonanceMessage>,

    /// Channel to send network messages to coordinator
    network_tx: mpsc::UnboundedSender<ResonanceMessage>,
}

impl ResonanceProtocolHandler {
    /// 🎻 Create new resonance protocol handler
    ///
    /// This connects the resonance coordinator to the network layer,
    /// enabling the distributed symphony to play.
    pub fn new(
        coordinator: Arc<ResonanceCoordinator>,
        broadcast_rx: mpsc::UnboundedReceiver<ResonanceMessage>,
        network_tx: mpsc::UnboundedSender<ResonanceMessage>,
    ) -> Self {
        info!("🎻 Initializing Resonance Protocol Handler");
        Self {
            coordinator,
            broadcast_rx,
            network_tx,
        }
    }

    /// 🎻 Create handler with new coordinator (convenience method)
    ///
    /// This is the primary entry point for setting up resonance gossip.
    pub fn with_new_coordinator(
        node_id: Vec<u8>,
    ) -> (
        Self,
        Arc<ResonanceCoordinator>,
        mpsc::UnboundedSender<ResonanceMessage>,
    ) {
        info!(
            "🎻 Creating Resonance Protocol Handler with new coordinator for node {:?}",
            node_id
        );

        // Create coordinator with gossip support
        let (coordinator, tx_from_network, rx_from_coordinator) =
            ResonanceCoordinator::new_with_gossip(node_id);

        let coordinator = Arc::new(coordinator);

        let handler = Self::new(
            Arc::clone(&coordinator),
            rx_from_coordinator,
            tx_from_network.clone(),
        );

        (handler, coordinator, tx_from_network)
    }

    /// 🎻 Process incoming gossip message from network
    ///
    /// Philosophy: Receive vibrations from other nodes and forward to coordinator
    /// for resonance processing.
    pub async fn handle_network_message(&self, data: &[u8]) -> anyhow::Result<()> {
        match deserialize_resonance_message(data) {
            Ok(msg) => {
                debug!(
                    "🎻 Received resonance message from network: {:?}",
                    std::mem::discriminant(&msg)
                );

                // ✅ v0.9.98-beta: FAIL FAST - Propagate errors (AI Expert Consensus)
                // ChatGPT, DeepSeek, Kimi AI all agree: "Propagate errors to gossipsub"
                // Previous pattern (warn!) made gossipsub think message was delivered

                // Forward to coordinator via channel
                self.network_tx.send(msg.clone())
                    .map_err(|e| anyhow::anyhow!("Failed to forward message to coordinator: {}", e))?;

                // Process directly in coordinator - MUST succeed for durability
                self.coordinator.handle_gossip_message(msg).await
                    .context("Coordinator failed to process gossip message - durability not guaranteed")?;

                Ok(())
            }
            Err(e) => {
                warn!("🎻 Failed to deserialize resonance message: {}", e);
                Err(anyhow::anyhow!("Deserialization failed: {}", e))
            }
        }
    }

    /// 🎻 Get next broadcast message from coordinator
    ///
    /// Philosophy: Listen for vibrations that the coordinator wants to broadcast
    /// to the network symphony.
    pub async fn next_broadcast(&mut self) -> Option<Vec<u8>> {
        if let Some(msg) = self.broadcast_rx.recv().await {
            debug!(
                "🎻 Broadcasting resonance message: {:?}",
                std::mem::discriminant(&msg)
            );

            match serialize_resonance_message(&msg) {
                Ok(data) => Some(data),
                Err(e) => {
                    error!("🎻 Failed to serialize broadcast message: {}", e);
                    None
                }
            }
        } else {
            None
        }
    }

    /// 🎻 Get coordinator reference for external operations
    pub fn coordinator(&self) -> &Arc<ResonanceCoordinator> {
        &self.coordinator
    }
}

/// 🎻 Resonance Gossip Manager
///
/// Philosophy: This manages the complete resonance gossip lifecycle,
/// integrating with libp2p gossipsub for network-wide harmony.
pub struct ResonanceGossipManager {
    handler: ResonanceProtocolHandler,
    topic: IdentTopic,
}

impl ResonanceGossipManager {
    /// 🎻 Create new resonance gossip manager
    pub fn new(handler: ResonanceProtocolHandler) -> Self {
        let topic = resonance_topic();
        info!("🎻 Creating Resonance Gossip Manager for topic: {}", topic);

        Self { handler, topic }
    }

    /// 🎻 Get the resonance topic for gossipsub subscription
    pub fn topic(&self) -> &IdentTopic {
        &self.topic
    }

    /// 🎻 Process incoming gossipsub message
    ///
    /// Call this when receiving messages from libp2p gossipsub on the resonance topic.
    pub async fn handle_gossip_message(&self, message: Message) -> anyhow::Result<()> {
        debug!(
            "🎻 Processing gossipsub message from peer: {:?}",
            message.source
        );

        self.handler.handle_network_message(&message.data).await
    }

    /// 🎻 Get next message to broadcast via gossipsub
    ///
    /// Call this in your gossipsub event loop to send coordinator broadcasts.
    pub async fn next_broadcast(&mut self) -> Option<Vec<u8>> {
        self.handler.next_broadcast().await
    }

    /// 🎻 Get coordinator for consensus operations
    pub fn coordinator(&self) -> &Arc<ResonanceCoordinator> {
        self.handler.coordinator()
    }

    /// 🎻 Spawn background task to handle broadcasts
    ///
    /// Philosophy: Run the broadcast loop in the background, automatically
    /// publishing coordinator messages to the network symphony.
    pub fn spawn_broadcast_task<F>(
        mut self,
        mut publish_fn: F,
    ) -> tokio::task::JoinHandle<()>
    where
        F: FnMut(IdentTopic, Vec<u8>) -> anyhow::Result<MessageId> + Send + 'static,
    {
        let topic = self.topic.clone();

        tokio::spawn(async move {
            info!("🎻 Starting resonance broadcast task");

            while let Some(data) = self.next_broadcast().await {
                if let Err(e) = publish_fn(topic.clone(), data) {
                    error!("🎻 Failed to publish resonance message: {}", e);
                } else {
                    debug!("🎻 Successfully published resonance broadcast");
                }
            }

            warn!("🎻 Resonance broadcast task terminated");
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resonance_topic() {
        let topic = resonance_topic();
        assert_eq!(topic.hash().as_str(), RESONANCE_PROTOCOL);
    }

    #[tokio::test]
    async fn test_protocol_handler_creation() {
        let node_id = vec![1, 2, 3];
        let (handler, coordinator, _tx) = ResonanceProtocolHandler::with_new_coordinator(node_id);

        // Verify coordinator is accessible
        assert_eq!(
            Arc::strong_count(handler.coordinator()),
            2 // handler + coordinator variable
        );
    }

    #[tokio::test]
    async fn test_gossip_manager_creation() {
        let node_id = vec![1, 2, 3];
        let (handler, _coordinator, _tx) = ResonanceProtocolHandler::with_new_coordinator(node_id);
        let manager = ResonanceGossipManager::new(handler);

        assert_eq!(manager.topic().hash().as_str(), RESONANCE_PROTOCOL);
    }
}
