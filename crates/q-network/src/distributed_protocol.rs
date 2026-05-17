/// Unified Distributed Protocol for Q-NarwhalKnight
/// Combines VM and DEX distribution into a single libp2p network behavior

use anyhow::Result;
use libp2p::PeerId;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

use super::distributed_dex::{
    DistributedDEXCoordinator, LiquidityPoolMessage, OrderBookMessage, TradeMessage,
    TOPIC_LIQUIDITY_POOL, TOPIC_ORDER_BOOK, TOPIC_TRADE_EXECUTION,
};
use super::distributed_vm::{
    ContractStateMessage, DistributedVMCoordinator, ExecutionResultMessage,
    TOPIC_CONTRACT_STATE, TOPIC_EXECUTION_RESULT, TOPIC_STATE_UPDATE,
};

/// Unified coordinator managing both VM and DEX
pub struct DistributedProtocolManager {
    pub vm_coordinator: Arc<DistributedVMCoordinator>,
    pub dex_coordinator: Arc<DistributedDEXCoordinator>,
    pub local_peer_id: PeerId,
}

impl DistributedProtocolManager {
    /// Create new distributed protocol manager
    pub async fn new(local_peer_id: PeerId) -> Result<Self> {
        info!("🌐 Initializing Distributed Protocol Manager for peer {}", local_peer_id);

        Ok(Self {
            vm_coordinator: Arc::new(DistributedVMCoordinator::new(local_peer_id)),
            dex_coordinator: Arc::new(DistributedDEXCoordinator::new(local_peer_id)),
            local_peer_id,
        })
    }

    // NOTE: Network publishing methods would be integrated with UnifiedNetworkManager
    // For now, we focus on the coordinator logic and stats

    /// Get combined network statistics
    pub async fn get_stats(&self) -> DistributedNetworkStats {
        let vm_stats = self.vm_coordinator.get_stats().await;
        let dex_stats = self.dex_coordinator.get_stats().await;

        DistributedNetworkStats {
            peer_id: self.local_peer_id.to_string(),
            vm_stats,
            dex_stats,
            uptime_secs: 0, // TODO: Track uptime
        }
    }
}

/// Combined network statistics
#[derive(Debug, Clone, serde::Serialize)]
pub struct DistributedNetworkStats {
    pub peer_id: String,
    pub vm_stats: super::distributed_vm::VMNetworkStats,
    pub dex_stats: super::distributed_dex::DEXStats,
    pub uptime_secs: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_protocol_manager_creation() {
        let peer_id = PeerId::random();
        let manager = DistributedProtocolManager::new(peer_id).await;
        assert!(manager.is_ok());
    }
}
