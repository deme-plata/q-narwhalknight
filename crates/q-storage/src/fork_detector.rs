// ✅ v0.9.67-beta: Comprehensive Fork Detection & Resolution System
// Automatically detects and resolves blockchain forks, including backward reorgs

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{error, info, warn};

/// Fork detection event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ForkEvent {
    /// Network consensus moved backward (reorg detected)
    BackwardReorg {
        our_height: u64,
        network_height: u64,
        reorg_depth: u64,
    },
    /// We're on a minority fork
    MinorityFork {
        our_height: u64,
        majority_height: u64,
        peer_count_majority: usize,
        peer_count_our_fork: usize,
    },
    /// Normal forward progression
    ForwardSync {
        our_height: u64,
        network_height: u64,
    },
    /// We're ahead of network (possible future fork)
    AheadOfNetwork {
        our_height: u64,
        network_height: u64,
    },
}

/// Peer height tracking for fork detection
#[derive(Debug, Clone)]
pub struct PeerHeightInfo {
    pub peer_id: String,
    pub height: u64,
    pub last_updated: std::time::Instant,
}

/// Fork detector with consensus tracking
pub struct ForkDetector {
    /// Tracked peer heights
    peer_heights: Arc<tokio::sync::RwLock<HashMap<String, PeerHeightInfo>>>,

    /// Minimum peers required for fork detection
    min_peers_for_consensus: usize,

    /// Fork detection threshold (% of peers that must agree)
    consensus_threshold: f64,

    /// Maximum allowed reorg depth before manual intervention
    max_auto_reorg_depth: u64,
}

impl ForkDetector {
    /// Create new fork detector
    pub fn new() -> Self {
        Self {
            peer_heights: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            min_peers_for_consensus: 3, // Need at least 3 peers
            consensus_threshold: 0.67,   // 67% of peers must agree
            max_auto_reorg_depth: 1000,  // Max 1000 blocks automatic reorg
        }
    }

    /// Update a peer's reported height
    pub async fn update_peer_height(&self, peer_id: String, height: u64) {
        let mut peers = self.peer_heights.write().await;
        peers.insert(
            peer_id.clone(),
            PeerHeightInfo {
                peer_id,
                height,
                last_updated: std::time::Instant::now(),
            },
        );
    }

    /// Remove stale peer heights (not updated in 60 seconds)
    async fn clean_stale_peers(&self) {
        let mut peers = self.peer_heights.write().await;
        let now = std::time::Instant::now();
        peers.retain(|_, info| now.duration_since(info.last_updated).as_secs() < 60);
    }

    /// Detect fork based on peer consensus
    pub async fn detect_fork(&self, our_height: u64) -> Result<ForkEvent> {
        // Clean stale peers first
        self.clean_stale_peers().await;

        let peers = self.peer_heights.read().await;

        // Not enough peers for fork detection
        if peers.len() < self.min_peers_for_consensus {
            warn!(
                "⚠️  [FORK DETECTOR] Only {} peers connected (need {}), cannot detect forks reliably",
                peers.len(),
                self.min_peers_for_consensus
            );
            return Ok(ForkEvent::ForwardSync {
                our_height,
                network_height: our_height,
            });
        }

        // Build height distribution
        let mut height_counts: HashMap<u64, usize> = HashMap::new();
        for info in peers.values() {
            *height_counts.entry(info.height).or_insert(0) += 1;
        }

        // Find consensus height (height with most peers)
        let (consensus_height, peer_count) = height_counts
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(h, c)| (*h, *c))
            .unwrap_or((our_height, 0));

        let consensus_ratio = peer_count as f64 / peers.len() as f64;

        info!(
            "🔍 [FORK DETECTOR] Our height: {}, Consensus height: {}, Peers agreeing: {}/{} ({:.1}%)",
            our_height,
            consensus_height,
            peer_count,
            peers.len(),
            consensus_ratio * 100.0
        );

        // ========================================
        // FORK DETECTION LOGIC
        // ========================================

        // Case 1: Backward reorg detected (CRITICAL!)
        if consensus_height < our_height && consensus_ratio >= self.consensus_threshold {
            let reorg_depth = our_height - consensus_height;

            error!(
                "🚨 [FORK DETECTOR] BACKWARD REORG DETECTED! {} peers agree network is at height {} (we're at {})",
                peer_count, consensus_height, our_height
            );
            error!(
                "🚨 [FORK DETECTOR] Reorg depth: {} blocks must be rolled back",
                reorg_depth
            );

            return Ok(ForkEvent::BackwardReorg {
                our_height,
                network_height: consensus_height,
                reorg_depth,
            });
        }

        // Case 2: We're on minority fork
        if consensus_height != our_height && consensus_ratio >= self.consensus_threshold {
            let our_fork_peers = height_counts.get(&our_height).copied().unwrap_or(0);

            warn!(
                "⚠️  [FORK DETECTOR] Minority fork detected! {}/{} peers on different chain (height {})",
                peer_count, peers.len(), consensus_height
            );

            return Ok(ForkEvent::MinorityFork {
                our_height,
                majority_height: consensus_height,
                peer_count_majority: peer_count,
                peer_count_our_fork: our_fork_peers,
            });
        }

        // Case 3: We're ahead of network (possible future fork or isolated)
        if our_height > consensus_height + 10 {
            warn!(
                "⚠️  [FORK DETECTOR] We're ahead of network: {} vs {}. Possibly isolated or on divergent fork.",
                our_height, consensus_height
            );

            return Ok(ForkEvent::AheadOfNetwork {
                our_height,
                network_height: consensus_height,
            });
        }

        // Case 4: Normal forward sync
        Ok(ForkEvent::ForwardSync {
            our_height,
            network_height: consensus_height,
        })
    }

    /// Get network consensus height (what most peers agree on)
    pub async fn get_consensus_height(&self) -> Option<u64> {
        self.clean_stale_peers().await;

        let peers = self.peer_heights.read().await;

        if peers.len() < self.min_peers_for_consensus {
            return None;
        }

        let mut height_counts: HashMap<u64, usize> = HashMap::new();
        for info in peers.values() {
            *height_counts.entry(info.height).or_insert(0) += 1;
        }

        height_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(height, _)| height)
    }

    /// Get all peer heights for debugging
    pub async fn get_peer_heights(&self) -> Vec<(String, u64)> {
        self.clean_stale_peers().await;

        let peers = self.peer_heights.read().await;
        peers
            .values()
            .map(|info| (info.peer_id.clone(), info.height))
            .collect()
    }

    /// Check if automatic reorg is safe
    pub fn is_safe_auto_reorg(&self, reorg_depth: u64) -> bool {
        reorg_depth <= self.max_auto_reorg_depth
    }
}

impl Default for ForkDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_backward_reorg_detection() {
        let detector = ForkDetector::new();

        // Simulate 5 peers all agreeing network is at height 2748
        for i in 0..5 {
            detector
                .update_peer_height(format!("peer{}", i), 2748)
                .await;
        }

        // Our node is at 4810
        let event = detector.detect_fork(4810).await.unwrap();

        match event {
            ForkEvent::BackwardReorg {
                our_height,
                network_height,
                reorg_depth,
            } => {
                assert_eq!(our_height, 4810);
                assert_eq!(network_height, 2748);
                assert_eq!(reorg_depth, 2062);
            }
            _ => panic!("Expected BackwardReorg, got {:?}", event),
        }
    }

    #[tokio::test]
    async fn test_forward_sync_normal() {
        let detector = ForkDetector::new();

        // Peers at various heights ahead of us
        detector.update_peer_height("peer1".to_string(), 5000).await;
        detector.update_peer_height("peer2".to_string(), 5001).await;
        detector.update_peer_height("peer3".to_string(), 5002).await;

        // We're at 4900
        let event = detector.detect_fork(4900).await.unwrap();

        match event {
            ForkEvent::ForwardSync { .. } => {
                // Expected
            }
            _ => panic!("Expected ForwardSync, got {:?}", event),
        }
    }

    #[tokio::test]
    async fn test_minority_fork() {
        let detector = ForkDetector::new();

        // 4 peers on height 3000
        for i in 0..4 {
            detector
                .update_peer_height(format!("peer{}", i), 3000)
                .await;
        }

        // 1 peer on our fork
        detector.update_peer_height("peer4".to_string(), 2900).await;

        // We're at 2900 (minority)
        let event = detector.detect_fork(2900).await.unwrap();

        match event {
            ForkEvent::MinorityFork {
                peer_count_majority,
                ..
            } => {
                assert_eq!(peer_count_majority, 4);
            }
            _ => panic!("Expected MinorityFork, got {:?}", event),
        }
    }
}
