/// Turbo Sync Peer Registry Bridge
///
/// **CRITICAL FIX v0.9.5-beta**: Bridge libp2p discovered peers to TurboSync peer registry
///
/// **ROOT CAUSE**: TurboSync maintained its own empty `peer_registry` separate from libp2p's
/// `discovered_peers`, causing "No peers available with target height" failures even when
/// peers were connected via libp2p.
///
/// **SOLUTION**: Automatically synchronize libp2p peer discoveries with TurboSync registry
/// using periodic sync + event-driven updates.

use anyhow::Result;
use libp2p::PeerId;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

use crate::turbo_sync::TurboSyncManager;

/// Peer height cache entry
#[derive(Clone, Debug)]
pub struct PeerHeightEntry {
    /// Peer ID
    pub peer_id: PeerId,
    /// Last announced height
    pub height: u64,
    /// Last update timestamp
    pub last_updated: Instant,
    /// Number of successful sync operations from this peer
    pub successful_syncs: u64,
    /// Number of failed sync operations from this peer
    pub failed_syncs: u64,
}

/// Peer registry bridge - synchronizes libp2p peers with TurboSync
pub struct TurboSyncPeerBridge {
    /// Cached peer heights
    peer_heights: Arc<RwLock<HashMap<PeerId, PeerHeightEntry>>>,

    /// Last sync timestamp
    last_sync: Arc<RwLock<Instant>>,

    /// Sync interval (how often to update TurboSync registry from cache)
    sync_interval: Duration,
}

impl TurboSyncPeerBridge {
    /// Create new peer registry bridge
    pub fn new() -> Self {
        Self {
            peer_heights: Arc::new(RwLock::new(HashMap::new())),
            last_sync: Arc::new(RwLock::new(Instant::now())),
            sync_interval: Duration::from_secs(5), // Update TurboSync registry every 5 seconds
        }
    }

    /// Update peer height from gossipsub peer-height announcement
    ///
    /// **Call this from gossipsub `/qnk/testnet-phase5/peer-heights` topic handler**
    pub async fn update_peer_height(&self, peer_id: PeerId, height: u64) {
        let mut peers = self.peer_heights.write().await;

        if let Some(entry) = peers.get_mut(&peer_id) {
            // Update existing peer
            if height > entry.height {
                debug!("🔄 [PEER BRIDGE] Updated peer {} height: {} → {}",
                       peer_id, entry.height, height);
                entry.height = height;
                entry.last_updated = Instant::now();
            }
        } else {
            // Register new peer
            info!("✅ [PEER BRIDGE] Registered new peer {} with height {}", peer_id, height);
            peers.insert(peer_id, PeerHeightEntry {
                peer_id,
                height,
                last_updated: Instant::now(),
                successful_syncs: 0,
                failed_syncs: 0,
            });
        }
    }

    /// Remove peer when disconnected from libp2p
    ///
    /// **Call this from libp2p ConnectionClosed event**
    pub async fn remove_peer(&self, peer_id: &PeerId) {
        let mut peers = self.peer_heights.write().await;
        if peers.remove(peer_id).is_some() {
            info!("👋 [PEER BRIDGE] Removed disconnected peer {}", peer_id);
        }
    }

    /// Record successful sync from peer (for reputation tracking)
    pub async fn record_sync_success(&self, peer_id: &PeerId) {
        let mut peers = self.peer_heights.write().await;
        if let Some(entry) = peers.get_mut(peer_id) {
            entry.successful_syncs += 1;
            debug!("✅ [PEER BRIDGE] Peer {} successful syncs: {}",
                   peer_id, entry.successful_syncs);
        }
    }

    /// Record failed sync from peer (for reputation tracking)
    pub async fn record_sync_failure(&self, peer_id: &PeerId) {
        let mut peers = self.peer_heights.write().await;
        if let Some(entry) = peers.get_mut(peer_id) {
            entry.failed_syncs += 1;
            warn!("❌ [PEER BRIDGE] Peer {} failed syncs: {}",
                  peer_id, entry.failed_syncs);

            // TODO: Ban peers with excessive failures (>10 consecutive failures)
        }
    }

    /// Synchronize cached peer heights to TurboSync registry
    ///
    /// **Call this periodically** (every 5 seconds) OR after receiving peer height announcement
    pub async fn sync_to_turbo_sync(&self, turbo_sync: &TurboSyncManager) -> Result<()> {
        let mut last_sync = self.last_sync.write().await;

        // Check if enough time has elapsed since last sync
        if last_sync.elapsed() < self.sync_interval {
            return Ok(());
        }

        let peers = self.peer_heights.read().await;
        let peer_count = peers.len();

        if peer_count == 0 {
            debug!("🔄 [PEER BRIDGE] No peers to sync to TurboSync registry");
            return Ok(());
        }

        info!("🔄 [PEER BRIDGE] Syncing {} peers to TurboSync registry", peer_count);

        // Register all cached peers with TurboSync
        for (peer_id, entry) in peers.iter() {
            turbo_sync.register_peer(*peer_id, entry.height).await;
        }

        // Update last sync timestamp
        *last_sync = Instant::now();

        info!("✅ [PEER BRIDGE] TurboSync registry updated with {} peers", peer_count);

        // Log peer registry info for debugging
        let registry_info = turbo_sync.get_peer_registry_info().await;
        if registry_info.is_empty() {
            error!("🚨 [PEER BRIDGE] WARNING: TurboSync registry is EMPTY after sync!");
            error!("   This indicates a critical bug in register_peer()");
        } else {
            debug!("📊 [PEER BRIDGE] TurboSync registry now has {} peers", registry_info.len());
            for (pid, height) in registry_info.iter().take(5) {
                debug!("   - Peer {}: height {}", pid, height);
            }
        }

        Ok(())
    }

    /// Get current peer count
    pub async fn peer_count(&self) -> usize {
        self.peer_heights.read().await.len()
    }

    /// Get all cached peers
    pub async fn get_all_peers(&self) -> Vec<PeerHeightEntry> {
        self.peer_heights.read().await.values().cloned().collect()
    }

    /// Get peers with height >= target (for Turbo Sync discovery)
    pub async fn get_peers_with_height(&self, target_height: u64) -> Vec<(PeerId, u64)> {
        let peers = self.peer_heights.read().await;

        peers.iter()
            .filter(|(_, entry)| entry.height >= target_height)
            .map(|(peer_id, entry)| (*peer_id, entry.height))
            .collect()
    }
}

impl Default for TurboSyncPeerBridge {
    fn default() -> Self {
        Self::new()
    }
}

/// Background task to periodically sync peers to TurboSync
///
/// **Usage**:
/// ```rust
/// let bridge = Arc::new(TurboSyncPeerBridge::new());
/// let turbo_sync = Arc::new(turbo_sync_manager);
///
/// tokio::spawn(run_periodic_sync(bridge.clone(), turbo_sync.clone()));
/// ```
pub async fn run_periodic_sync(
    bridge: Arc<TurboSyncPeerBridge>,
    turbo_sync: Arc<TurboSyncManager>,
) {
    info!("🔄 [PEER BRIDGE] Starting periodic sync task (every 5 seconds)");

    loop {
        tokio::time::sleep(Duration::from_secs(5)).await;

        if let Err(e) = bridge.sync_to_turbo_sync(&turbo_sync).await {
            error!("❌ [PEER BRIDGE] Failed to sync peers to TurboSync: {}", e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_peer_bridge_updates() {
        let bridge = TurboSyncPeerBridge::new();

        // Generate test peer IDs
        let peer1 = PeerId::random();
        let peer2 = PeerId::random();

        // Update peer heights
        bridge.update_peer_height(peer1, 100).await;
        bridge.update_peer_height(peer2, 200).await;

        // Verify peer count
        assert_eq!(bridge.peer_count().await, 2);

        // Verify height queries
        let peers_100 = bridge.get_peers_with_height(100).await;
        assert_eq!(peers_100.len(), 2); // Both have height >= 100

        let peers_150 = bridge.get_peers_with_height(150).await;
        assert_eq!(peers_150.len(), 1); // Only peer2 has height >= 150

        // Update peer1 to higher height
        bridge.update_peer_height(peer1, 250).await;
        let peers_200 = bridge.get_peers_with_height(200).await;
        assert_eq!(peers_200.len(), 2); // Both now have height >= 200

        // Remove peer
        bridge.remove_peer(&peer1).await;
        assert_eq!(bridge.peer_count().await, 1);
    }
}
