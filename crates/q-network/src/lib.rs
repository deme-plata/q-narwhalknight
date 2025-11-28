use anyhow::Result;
/// Q-Network: Quantum-ready libp2p networking layer
/// Phase 0: Classical Ed25519 + QUIC
/// Phase 1: Post-quantum TLS with crypto-agility
/// Phase 4: QKD integration
use q_types::*;
use std::collections::HashMap;
use tokio::sync::{broadcast, RwLock};
use tracing::{debug, error, info, warn};

pub mod crypto_agile;
pub mod dag_sync;
pub mod network_manager;
pub mod peer_registry;
pub mod persistent_channels;
pub mod protocol_handshake;
pub mod handshake_validator;  // ✅ v1.0.15.1-beta - Protocol version validation

// Export handshake components
pub use handshake_validator::{
    HandshakeValidator, HandshakeMessage, HandshakeResult, ProtocolVersion,
    HANDSHAKE_PROTOCOL, HandshakeCodec,
};

// Real network implementations (production-ready)
pub mod real_dht;
pub mod real_peer_discovery;

// libp2p-based peer discovery (zero-config mDNS + gossipsub)
pub mod unified_network_manager;
pub mod libp2p_bridge;
pub mod dag_sync_adapter; // 🚀 v1.0.4-beta: Phase 2 DAG-Aware Sync network adapter

// Resonance consensus protocol (Phase 3: String-theoretic consensus)
pub mod resonance_protocol;

// Transaction Tunneling - Ultra-low-latency fast path
pub mod transaction_tunneling;

// ZK Peer Height Proofs - Trustless P2P Sync (v0.9.6-beta)
pub mod zk_peer_height_proof;

// ✨ v1.0.58-beta: Lattice Aggregate Signatures for 98% bandwidth reduction (IACR 2025/1056)
#[cfg(feature = "advanced-crypto")]
pub mod lattice_gossip;

pub use crypto_agile::{AgileHandshake, CryptoProvider, CryptoScheme, Kyber1024KeyExchange};
pub use network_manager::{NetworkManager, NetworkManagerConfig};
pub use peer_registry::{PeerCapability, PeerInfo, PeerRegistry};
pub use persistent_channels::PersistentChannelManager;
pub use dag_sync::{DagSyncManager, DagSyncRequest, DagSyncResponse, SyncType, DagStateSummary};

// Export libp2p discovery components
pub use unified_network_manager::{UnifiedNetworkManager, NetworkCommand};
pub use libp2p_bridge::{Libp2pBridge, BridgeEvent, DhtEvent};

// Export resonance consensus protocol components
pub use resonance_protocol::{
    resonance_topic, ResonanceGossipManager, ResonanceProtocolHandler,
};

// Export transaction tunneling components
pub use transaction_tunneling::{
    TunnelingEngine, TunnelingConfig, TunnelingProfile, TunnelingResult,
    TunnelingStats, ConsensusMessageType,
};

// Export ZK peer height proof components (v0.9.6-beta)
pub use zk_peer_height_proof::{
    PeerHeightWithProof, PeerHeightVerifier, generate_height_proof,
};

// ✨ v1.0.58-beta: Lattice aggregate signature exports (98% bandwidth reduction)
#[cfg(feature = "advanced-crypto")]
pub use lattice_gossip::{
    GossipAggregator, GossipAggregatorConfig, AggregatedGossipMessage,
    GossipSigningKey, GossipBatchVerifier, LatticeSecurityLevel, AggregationStats,
};

// Export protocol handshake components (v0.9.57-beta)
pub use protocol_handshake::ProtocolHandshake;

// Distributed VM, DEX, and AI modules
pub mod distributed_vm;
pub mod distributed_dex;
pub mod distributed_ai;
pub mod distributed_ai_coordinator;
pub mod distributed_ai_worker; // FLAW #1 FIX: Worker node inference handler
pub mod distributed_protocol;
pub mod layer_forwarding;
pub mod distributed_inference_bridge;
pub mod kv_cache_manager;
pub mod distributed_mistralrs_bridge;
pub mod encrypted_tensor_forwarding; // PRIVACY: ZK + Aegis-QL encrypted tensors
pub mod failover_manager; // Automatic failover & retry logic
pub mod public_key_dht; // v1.0.3-beta: DHT-based public key distribution (Showstopper #2 fix)
pub mod signature_cache; // v1.0.3-beta: Signature verification cache with TOCTOU fix (Showstopper #3 fix)
pub mod security_metrics; // v1.0.3-beta: Prometheus metrics for signature verification (Week 2, Day 1-2)
pub mod circuit_breaker; // v1.0.3-beta: Circuit breaker for attack protection (Week 2, Day 3-4)

// Export distributed components
pub use distributed_vm::{
    DistributedVMCoordinator, ContractStateMessage, ExecutionRequest,
    ExecutionResponse, StateUpdate, MerkleProof, VMNetworkStats,
};
pub use distributed_dex::{
    DistributedDEXCoordinator, OrderBookMessage, TradeMessage,
    LiquidityPoolMessage, TradingPair, Order, OrderType, OrderSide,
    DEXStats, ArbitrageOpportunity,
};
pub use distributed_protocol::{
    DistributedProtocolManager, DistributedNetworkStats,
};
pub use distributed_ai::{
    DistributedAITopics, AIGossipsubMessage, AIMessagePayload, NodeCapability,
    CURRENT_PROTOCOL_VERSION, // v0.9.29+ FIX: Export protocol version constant
};
pub use distributed_ai_coordinator::{
    DistributedAICoordinator, AINode, DistributedInferenceRequest,
    DistributedAIStats, InferenceResponseChunk,
};
pub use distributed_ai_worker::{
    DistributedAIWorker, ActiveInferenceRequest,
};
pub use layer_forwarding::{
    LayerOutputManager, TensorData, TensorDType, LayerOutput, LayerForwardingStats,
};
pub use distributed_inference_bridge::{
    DistributedInferenceBridge, InferenceSession, SessionState, SessionStats,
};
pub use kv_cache_manager::{
    KVCacheManager, KVCacheEntry, SessionKVCache, KVCacheStats,
};
pub use distributed_mistralrs_bridge::{
    DistributedMistralRsBridge, DistributedMistralRsConfig,
};
pub use public_key_dht::{
    PublicKeyDhtManager, NodePublicKeyAnnouncement, CachedPublicKey,
};
pub use signature_cache::{
    SignatureCache, CacheStats,
};
pub use security_metrics::{
    SecurityMetrics, SignatureVerificationStats, CachePerformanceStats, DhtOperationStats,
};
pub use circuit_breaker::{
    CircuitBreaker, CircuitBreakerConfig, CircuitBreakerStats, CircuitBreakerState,
};
pub use failover_manager::{
    FailoverManager, FailoverConfig, FailoverDecision, WorkerHealth,
    FailureRecord, FailureType,
};

// Simplified network structure for compilation
pub struct QuantumNetwork {
    node_id: NodeId,
    current_phase: Phase,
    crypto_provider: CryptoProvider,
}

impl QuantumNetwork {
    /// Create new quantum network (Phase 0: Classical)
    pub async fn new_phase0(node_id: NodeId) -> Result<Self> {
        Self::new_with_phase(node_id, Phase::Phase0).await
    }

    /// Create new quantum network with specific phase
    pub async fn new_with_phase(node_id: NodeId, phase: Phase) -> Result<Self> {
        info!("🌐 Initializing Q-Network {:?}", phase);

        Ok(Self {
            node_id,
            current_phase: phase,
            crypto_provider: match phase {
                Phase::Phase0 => CryptoProvider::new_phase0()?,
                Phase::Phase1 | Phase::Phase2 | Phase::Phase3 | Phase::Phase4 => {
                    CryptoProvider::new_phase1()?
                }
            },
        })
    }

    /// Upgrade to Phase 1: Post-Quantum Cryptography
    pub async fn upgrade_to_phase1(&mut self) -> Result<()> {
        info!("🔄 Upgrading Q-Network to Phase 1 (Post-Quantum)");
        self.crypto_provider = CryptoProvider::new_phase1()?;
        self.current_phase = Phase::Phase1;
        info!("✅ Successfully upgraded to Phase 1 (Post-Quantum)");
        Ok(())
    }

    /// Get network statistics
    pub async fn get_network_stats(&self) -> NetworkStats {
        NetworkStats {
            connected_peers: 0,
            total_peers_seen: 0,
            current_phase: self.current_phase,
            crypto_provider: "placeholder".to_string(),
            uptime: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default(),
        }
    }
}

/// Network statistics for monitoring
#[derive(Debug, Clone, serde::Serialize)]
pub struct NetworkStats {
    pub connected_peers: u64,
    pub total_peers_seen: u64,
    pub current_phase: Phase,
    pub crypto_provider: String,
    pub uptime: std::time::Duration,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_network_creation() {
        let node_id = [1u8; 32];
        let network = QuantumNetwork::new_phase0(node_id).await;
        assert!(network.is_ok());
    }

    #[tokio::test]
    async fn test_network_stats() {
        let node_id = [1u8; 32];
        let network = QuantumNetwork::new_phase0(node_id).await.unwrap();

        let stats = network.get_network_stats().await;
        assert_eq!(stats.connected_peers, 0);
        assert_eq!(stats.current_phase, Phase::Phase0);
    }
}
pub mod connection_manager;
pub mod handshake;
