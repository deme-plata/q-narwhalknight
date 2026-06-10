/*!
# Peer Synchronization Bridge

Automatically synchronizes libp2p discovered peers with the DAG-Knight consensus layer
to enable horizontal scaling performance improvements.

## Architecture

```
┌──────────────────────┐
│ LibP2P Discovery     │  Discovers peers via Kademlia DHT
│ (q-bep44-discovery)  │
└──────────┬───────────┘
           │
           │ peer_id, multiaddr, connection_quality
           ▼
┌──────────────────────┐
│ Peer Sync Bridge     │  Monitors discovered peers
│ (this module)        │  Registers them as validators
└──────────┬───────────┘
           │
           │ register_discovered_peer()
           ▼
┌──────────────────────┐
│ DAG-Knight Consensus │  Treats discovered peers as validators
│ (q-dag-knight)       │  Routes consensus messages to them
└──────────────────────┘
```
*/

use anyhow::Result;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::interval;
use tracing::{debug, info, warn};

use q_dag_knight::DAGKnightConsensus;
use q_bep44_discovery::DiscoveryEngine;

/// Peer synchronization service that bridges discovery and consensus
pub struct PeerSyncService {
    discovery_engine: Arc<tokio::sync::Mutex<DiscoveryEngine>>,
    consensus: Arc<DAGKnightConsensus>,
    sync_interval: Duration,
}

impl PeerSyncService {
    /// Create a new peer synchronization service
    pub fn new(
        discovery_engine: Arc<tokio::sync::Mutex<DiscoveryEngine>>,
        consensus: Arc<DAGKnightConsensus>,
        sync_interval_secs: u64,
    ) -> Self {
        Self {
            discovery_engine,
            consensus,
            sync_interval: Duration::from_secs(sync_interval_secs),
        }
    }

    /// Start the peer synchronization service
    pub async fn start(self: Arc<Self>) -> Result<()> {
        info!("🔄 Starting peer synchronization service (interval: {:?})", self.sync_interval);

        let mut sync_ticker = interval(self.sync_interval);

        loop {
            sync_ticker.tick().await;

            if let Err(e) = self.sync_peers().await {
                warn!("⚠️ Peer synchronization failed: {}", e);
            }
        }
    }

    /// Synchronize discovered peers to consensus layer
    async fn sync_peers(&self) -> Result<()> {
        // Get discovered peers from discovery engine
        let discovered_peers = self.discovery_engine.lock().await.get_discovered_peers().await;

        debug!("🔍 Syncing {} discovered peers to consensus layer", discovered_peers.len());

        let mut newly_registered = 0;

        for peer in discovered_peers {
            // Use the validator_id from DiscoveredPeer
            let peer_id_bytes = &peer.validator_id;

            // Calculate connection quality based on peer metadata
            let connection_quality = Self::calculate_connection_quality(&peer);

            // Create multiaddr string from peer info
            let multiaddr = if !peer.onion_address.is_empty() {
                format!("onion://{}", peer.onion_address)
            } else {
                format!("peer://{}", hex::encode(&peer_id_bytes[..8]))
            };

            // Register peer with consensus
            match self.consensus.register_discovered_peer(
                peer_id_bytes,
                multiaddr.clone(),
                connection_quality,
            ).await {
                Ok(_) => {
                    newly_registered += 1;
                }
                Err(e) => {
                    warn!("Failed to register peer {}: {}", hex::encode(&peer_id_bytes[..8]), e);
                }
            }
        }

        // Update mesh connectivity score
        self.consensus.update_mesh_connectivity().await;

        let validator_count = self.consensus.get_validator_count().await;
        let mesh_score = self.consensus.get_mesh_connectivity_score().await;

        info!(
            "✅ Peer sync complete: {} validators registered, mesh score: {:.2}",
            validator_count, mesh_score
        );

        if newly_registered > 0 {
            info!("🎉 {} new validators registered for horizontal scaling!", newly_registered);
        }

        Ok(())
    }

    /// Calculate connection quality score for a peer
    fn calculate_connection_quality(peer: &q_bep44_discovery::LocalDiscoveredPeer) -> f64 {
        let mut quality: f64 = 1.0;

        // Increase quality if has onion address (secure connection)
        if !peer.onion_address.is_empty() {
            quality += 0.2;
        }

        // Increase quality if has real IP addresses (actual connectivity)
        if !peer.real_ip_addresses.is_empty() {
            quality += 0.1;
        }

        quality.max(0.1).min(1.0)
    }

    /// Get current synchronization status
    pub async fn get_status(&self) -> PeerSyncStatus {
        let discovered_peer_count = self.discovery_engine.lock().await.get_discovered_peers().await.len();
        let validator_count = self.consensus.get_validator_count().await;
        let mesh_score = self.consensus.get_mesh_connectivity_score().await;

        PeerSyncStatus {
            discovered_peers: discovered_peer_count,
            registered_validators: validator_count,
            mesh_connectivity_score: mesh_score,
            sync_interval_secs: self.sync_interval.as_secs(),
        }
    }
}

/// Status information for peer synchronization
#[derive(Debug, Clone, serde::Serialize)]
pub struct PeerSyncStatus {
    pub discovered_peers: usize,
    pub registered_validators: usize,
    pub mesh_connectivity_score: f64,
    pub sync_interval_secs: u64,
}