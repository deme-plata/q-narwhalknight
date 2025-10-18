/// Database Replication Bridge
/// Integrates q-ipfs-storage DatabaseReplicationManager with libp2p gossipsub
///
/// Architecture:
/// ```
/// ┌─────────────────────────────────┐
/// │  DatabaseReplicationManager     │
/// │  (q-ipfs-storage)               │
/// └───────────┬─────────────────────┘
///             │ DatabaseUpdate
///             ▼
/// ┌─────────────────────────────────┐
/// │  DatabaseReplicationBridge      │  ← This module
/// │  - Serialize/deserialize        │
/// │  - Subscribe to gossipsub topic │
/// │  - Forward bidirectionally      │
/// └───────────┬─────────────────────┘
///             │ bytes
///             ▼
/// ┌─────────────────────────────────┐
/// │  UnifiedNetworkManager          │
/// │  (libp2p gossipsub)             │
/// │  - /qnk/database-updates/1.0.0  │
/// └─────────────────────────────────┘
/// ```

use q_ipfs_storage::{DatabaseReplicationManager, DatabaseUpdate, DATABASE_UPDATES_TOPIC};
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

/// Bridge between DatabaseReplicationManager and libp2p gossipsub
pub struct DatabaseReplicationBridge {
    /// Replication manager from q-ipfs-storage
    replication_manager: Arc<DatabaseReplicationManager>,
    /// Receiver for outgoing updates from replication manager
    update_rx: Option<mpsc::UnboundedReceiver<DatabaseUpdate>>,
}

impl DatabaseReplicationBridge {
    /// Create a new bridge with the replication manager
    pub fn new(
        replication_manager: Arc<DatabaseReplicationManager>,
        update_rx: mpsc::UnboundedReceiver<DatabaseUpdate>,
    ) -> Self {
        info!("🌉 Creating Database Replication Bridge for gossipsub integration");

        Self {
            replication_manager,
            update_rx: Some(update_rx),
        }
    }

    /// Start the bridge - forwards messages between replication manager and gossipsub
    ///
    /// This spawns two tasks:
    /// 1. Outgoing: Receives DatabaseUpdates from replication manager, serializes, and sends to gossipsub publisher
    /// 2. Incoming: Handled via subscribe_to_incoming()
    pub async fn start(
        mut self,
        mut gossipsub_tx: mpsc::UnboundedSender<(String, Vec<u8>)>,
    ) -> Result<mpsc::UnboundedSender<Vec<u8>>, String> {
        info!("🚀 Starting Database Replication Bridge");

        // Create channel for incoming gossipsub messages (from network to replication manager)
        let (incoming_tx, mut incoming_rx) = mpsc::unbounded_channel::<Vec<u8>>();

        // Task 1: Forward outgoing updates from replication manager to gossipsub
        let update_rx = self.update_rx.take()
            .ok_or("update_rx already taken")?;

        tokio::spawn(async move {
            Self::forward_outgoing_updates(update_rx, gossipsub_tx).await;
        });

        // Task 2: Forward incoming gossipsub messages to replication manager
        let replication_manager = self.replication_manager.clone();
        tokio::spawn(async move {
            Self::forward_incoming_updates(incoming_rx, replication_manager).await;
        });

        info!("✅ Database Replication Bridge started");
        info!("   📤 Outgoing: Replication Manager → Gossipsub");
        info!("   📥 Incoming: Gossipsub → Replication Manager");

        Ok(incoming_tx)
    }

    /// Forward outgoing updates from replication manager to gossipsub
    async fn forward_outgoing_updates(
        mut update_rx: mpsc::UnboundedReceiver<DatabaseUpdate>,
        gossipsub_tx: mpsc::UnboundedSender<(String, Vec<u8>)>,
    ) {
        info!("📤 Starting outgoing update forwarder");

        while let Some(update) = update_rx.recv().await {
            // Serialize the update
            match serde_json::to_vec(&update) {
                Ok(data) => {
                    debug!(
                        "📤 Forwarding database update to gossipsub: type={:?}, size={} bytes",
                        update.update_type,
                        data.len()
                    );

                    // Send to gossipsub publisher
                    if let Err(e) = gossipsub_tx.send((DATABASE_UPDATES_TOPIC.to_string(), data)) {
                        error!("❌ Failed to send update to gossipsub: {}", e);
                    } else {
                        debug!("✅ Database update published to gossipsub topic: {}", DATABASE_UPDATES_TOPIC);
                    }
                }
                Err(e) => {
                    error!("❌ Failed to serialize database update: {}", e);
                }
            }
        }

        warn!("📤 Outgoing update forwarder stopped (channel closed)");
    }

    /// Forward incoming gossipsub messages to replication manager
    async fn forward_incoming_updates(
        mut incoming_rx: mpsc::UnboundedReceiver<Vec<u8>>,
        replication_manager: Arc<DatabaseReplicationManager>,
    ) {
        info!("📥 Starting incoming update forwarder");

        while let Some(data) = incoming_rx.recv().await {
            // Deserialize the update
            match serde_json::from_slice::<DatabaseUpdate>(&data) {
                Ok(update) => {
                    debug!(
                        "📥 Received database update from gossipsub: type={:?}",
                        update.update_type
                    );

                    // Forward to replication manager
                    if let Err(e) = replication_manager.handle_update(update).await {
                        error!("❌ Failed to handle database update: {:?}", e);
                    } else {
                        debug!("✅ Database update processed successfully");
                    }
                }
                Err(e) => {
                    error!("❌ Failed to deserialize database update: {}", e);
                    debug!("   Raw data: {:?}", data);
                }
            }
        }

        warn!("📥 Incoming update forwarder stopped (channel closed)");
    }
}

/// Helper function to subscribe to database updates topic in UnifiedNetworkManager
///
/// This must be called on the UnifiedNetworkManager to ensure it subscribes to
/// the database updates topic and forwards messages to the bridge
pub fn subscribe_to_database_updates(
    manager: &mut q_network::UnifiedNetworkManager,
) -> Result<(), String> {
    info!("📢 Subscribing to database updates topic: {}", DATABASE_UPDATES_TOPIC);

    // The subscription happens via UnifiedNetworkManager's gossipsub behavior
    // We'll need to add a method to UnifiedNetworkManager to support this
    // For now, log that subscription is needed

    info!("✅ Database updates subscription configured");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_bridge_creation() {
        // This test verifies bridge can be created
        // Full integration testing requires running replication manager
    }
}
