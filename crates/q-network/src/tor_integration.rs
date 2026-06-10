/// Tor Integration for Q-NarwhalKnight Network Layer
///
/// This module provides Tor transport integration using the q-tor-client
/// dedicated circuit architecture with Arti 1.8.0 (arti-client 0.37.0).
///
/// # Architecture
/// ```
/// ┌─────────────────┐    🧅 Tor Network    ┌─────────────────┐
/// │   Validator A   │◄──► 4+ Circuits   ◄──►│   Validator B   │
/// │ alice.qnk.onion │    (per operation)    │  bob.qnk.onion  │
/// └─────────────────┘                      └─────────────────┘
/// ```
///
/// # Operation Types
/// Each network operation type gets isolated Tor circuits:
/// - BlockPropagation: Consensus-critical block messages
/// - TransactionBroadcast: User transaction submission
/// - ConsensusVoting: DAG-Knight voting messages
/// - PeerDiscovery: Kademlia DHT queries
/// - StateSync: TurboSync block requests
/// - MetricsReporting: Prometheus telemetry (lowest priority)
/// - HeartbeatPing: Keep-alive messages
/// - ValidatorCommunication: Narwhal mempool messages

use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

// Re-export core Tor types from q-tor-client
pub use q_tor_client::{
    TorConfig,
    dedicated_circuits::{
        DedicatedCircuitConfig,
        DedicatedCircuitManager,
        OperationType,
        IsolatedCircuitStats,
        ManagerStats,
    },
    circuit_prewarming::{
        CircuitPrewarmingManager,
        PrewarmingConfig,
        CircuitHealth,
    },
    libp2p_transport::TorTransport,
};

/// Tor network configuration for Q-NarwhalKnight
///
/// This extends the base TorConfig with network-specific settings
/// for gossipsub topic routing and circuit selection.
#[derive(Clone, Debug)]
pub struct NetworkTorConfig {
    /// Base Tor configuration
    pub tor_config: TorConfig,
    /// Enable Tor for block propagation (gossipsub /qnk/blocks)
    pub tor_for_blocks: bool,
    /// Enable Tor for peer height announcements
    pub tor_for_peer_heights: bool,
    /// Enable Tor for turbo sync requests/responses
    pub tor_for_turbo_sync: bool,
    /// Enable Tor for Kademlia DHT queries
    pub tor_for_kademlia: bool,
    /// Enable Tor for miner stats broadcasting
    pub tor_for_miner_stats: bool,
    /// Enable Tor for balance update broadcasting
    pub tor_for_balance_updates: bool,
    /// Enable circuit prewarming for faster connections
    pub enable_prewarming: bool,
    /// Prewarmer configuration
    pub prewarmer_config: PrewarmingConfig,

    // v10.10.8 — three lifecycle phase flags (orthogonal to the per-topic flags above).
    // Q_TOR_LISTEN_ONION=1 → Phase A: secondary /onion3 listener via the embedded Arti
    // service. Other peers reach us anonymously; we still dial them clearnet.
    pub listen_onion: bool,
    // Q_TOR_STEM=1 → Phase B: Dandelion++ stem phase routes through dedicated Tor
    // circuits instead of plain gossipsub publish. TX broadcast only; block sync untouched.
    pub stem_via_tor: bool,
    // Q_TOR_OUTBOUND=1 → Phase C: wrap libp2p TCP transport with TorEnabledTransport.
    // ALL outbound dials route through Arti SOCKS5. Block sync slows 3-5×; pair with
    // Q_TOR_SKIP_SYNC=1 to keep block-pack on clearnet.
    pub outbound_via_tor: bool,
    // Q_TOR_SKIP_SYNC=1 escape hatch: when outbound_via_tor is true, block-pack and
    // turbo-sync requests still go clearnet. Prevents 17M-block bootstrap stall.
    pub skip_sync_via_tor: bool,

    // v10.10.15 — WireGuard fail-honest gate. When `require_wireguard` is true,
    // `outbound_via_tor` and `stem_via_tor` REFUSE to activate unless the
    // WireGuard interface (default `wg0`, override `Q_WIREGUARD_INTERFACE`) is
    // up. Stops the silent "ISP sees you're on Tor" failure mode when a VPN
    // kill-switch lapses. Operator must explicitly set Q_TOR_REQUIRE_WIREGUARD=0
    // to accept that risk. See `crate::wireguard_check`.
    pub require_wireguard: bool,
}

impl Default for NetworkTorConfig {
    fn default() -> Self {
        Self {
            tor_config: TorConfig::default(),
            // Default: Enable Tor for privacy-sensitive operations
            tor_for_blocks: true,
            tor_for_peer_heights: true,
            tor_for_turbo_sync: true,
            tor_for_kademlia: true,
            tor_for_miner_stats: true,
            tor_for_balance_updates: true,
            enable_prewarming: true,
            prewarmer_config: PrewarmingConfig::default(),
            // v10.10.8 lifecycle flags default OFF — code ships dead-disabled.
            listen_onion: false,
            stem_via_tor: false,
            outbound_via_tor: false,
            skip_sync_via_tor: true,
            // v10.10.15: gate defaults OFF (no behavior change for operators
            // who don't set it; with Tor outbound disabled by default anyway,
            // this is moot until they enable Phase B/C).
            require_wireguard: false,
        }
    }
}

impl NetworkTorConfig {
    /// Create a minimal Tor config (only critical operations)
    pub fn minimal() -> Self {
        Self {
            tor_config: TorConfig::default(),
            tor_for_blocks: true,
            tor_for_peer_heights: false,
            tor_for_turbo_sync: false, // Sync is high-bandwidth, use clearnet
            tor_for_kademlia: true,    // DHT queries reveal network topology
            tor_for_miner_stats: false,
            tor_for_balance_updates: true, // Privacy-sensitive
            enable_prewarming: false,
            prewarmer_config: PrewarmingConfig::default(),
            // v10.10.8 lifecycle flags default OFF in minimal mode too.
            listen_onion: false,
            stem_via_tor: false,
            outbound_via_tor: false,
            skip_sync_via_tor: true,
            // v10.10.15: gate defaults OFF (no behavior change for operators
            // who don't set it; with Tor outbound disabled by default anyway,
            // this is moot until they enable Phase B/C).
            require_wireguard: false,
        }
    }

    /// Create a maximum privacy config (all operations via Tor)
    pub fn maximum_privacy() -> Self {
        let mut config = Self::default();
        config.tor_config.tor_only = true;
        config
    }

    /// Load configuration from environment variables
    ///
    /// Environment variables:
    /// - Q_TOR_ENABLED: Enable Tor integration (default: false for now)
    /// - Q_TOR_MANDATORY: Require all traffic through Tor
    /// - Q_TOR_BLOCKS: Route blocks through Tor
    /// - Q_TOR_SYNC: Route sync through Tor
    /// - Q_TOR_DHT: Route DHT through Tor
    /// - Q_TOR_PREWARM: Enable circuit prewarming
    pub fn from_env() -> Self {
        let mut config = Self::default();

        // Check if Tor is enabled at all
        if std::env::var("Q_TOR_ENABLED").map(|v| v == "true" || v == "1").unwrap_or(false) {
            info!("🧅 [TOR] Tor integration enabled via Q_TOR_ENABLED");

            config.tor_config.tor_only = std::env::var("Q_TOR_MANDATORY")
                .map(|v| v == "true" || v == "1")
                .unwrap_or(false);

            config.tor_for_blocks = std::env::var("Q_TOR_BLOCKS")
                .map(|v| v == "true" || v == "1")
                .unwrap_or(true);

            config.tor_for_turbo_sync = std::env::var("Q_TOR_SYNC")
                .map(|v| v == "true" || v == "1")
                .unwrap_or(true);

            config.tor_for_kademlia = std::env::var("Q_TOR_DHT")
                .map(|v| v == "true" || v == "1")
                .unwrap_or(true);

            config.enable_prewarming = std::env::var("Q_TOR_PREWARM")
                .map(|v| v == "true" || v == "1")
                .unwrap_or(true);

            if config.tor_config.tor_only {
                info!("🧅 [TOR] MANDATORY TOR MODE - All traffic via Tor");
            }
        } else {
            // Tor disabled - set all flags to false
            config.tor_for_blocks = false;
            config.tor_for_peer_heights = false;
            config.tor_for_turbo_sync = false;
            config.tor_for_kademlia = false;
            config.tor_for_miner_stats = false;
            config.tor_for_balance_updates = false;
            config.enable_prewarming = false;
        }

        // v10.10.8 lifecycle flags read independently of Q_TOR_ENABLED so they can
        // be toggled per node without flipping the topic-level master switch.
        config.listen_onion = std::env::var("Q_TOR_LISTEN_ONION")
            .map(|v| v == "true" || v == "1")
            .unwrap_or(false);
        config.stem_via_tor = std::env::var("Q_TOR_STEM")
            .map(|v| v == "true" || v == "1")
            .unwrap_or(false);
        config.outbound_via_tor = std::env::var("Q_TOR_OUTBOUND")
            .map(|v| v == "true" || v == "1")
            .unwrap_or(false);
        config.skip_sync_via_tor = std::env::var("Q_TOR_SKIP_SYNC")
            .map(|v| v == "true" || v == "1")
            .unwrap_or(true);
        // v10.10.15: WireGuard fail-honest gate. Default OFF — operators who
        // care about the ISP-sees-Tor leak must opt in explicitly.
        config.require_wireguard = std::env::var("Q_TOR_REQUIRE_WIREGUARD")
            .map(|v| v == "true" || v == "1")
            .unwrap_or(false);

        if config.listen_onion {
            info!("🧅 [TOR] Phase A enabled — onion listener will be announced");
        }
        if config.stem_via_tor {
            info!("🧅 [TOR] Phase B enabled — Dandelion stems route via Tor circuits");
        }
        if config.outbound_via_tor {
            info!("🧅 [TOR] Phase C enabled — libp2p outbound dials route via Tor SOCKS5");
            if config.skip_sync_via_tor {
                info!("🧅 [TOR]   (block-pack stays clearnet — Q_TOR_SKIP_SYNC=1)");
            }
        }

        // v10.10.15: Run the WireGuard gate at config-load time so the result
        // (and any refusal) is visible in startup logs. If the gate refuses,
        // we DOWNGRADE the lifecycle flags to false rather than erroring
        // out — the node still boots on clearnet without Tor outbound. This
        // matches the "fail-honest" naming: operator sees the decision, can
        // act on it. Returning Err would crash-loop the node, which is worse
        // UX than a clearnet fallback with a loud warning.
        if config.stem_via_tor || config.outbound_via_tor {
            match crate::wireguard_check::activation_gate(config.require_wireguard) {
                Ok(_status) => { /* permitted */ }
                Err(_status) => {
                    warn!(
                        "🛡️  [WG GATE] Downgrading Tor Phase B/C OFF — \
                         require_wireguard=true but WG not up. Node will boot \
                         on clearnet. Set Q_TOR_REQUIRE_WIREGUARD=0 to override."
                    );
                    config.stem_via_tor = false;
                    config.outbound_via_tor = false;
                }
            }
        }

        config
    }
}

/// Tor-aware message router for gossipsub topics
///
/// Maps gossipsub topics to appropriate Tor circuit operation types
/// for traffic isolation and privacy.
pub struct TorMessageRouter {
    /// Network Tor configuration
    config: NetworkTorConfig,
    /// Circuit manager (if Tor enabled)
    circuit_manager: Option<Arc<DedicatedCircuitManager>>,
    /// Circuit prewarmer (if enabled)
    prewarmer: Option<Arc<CircuitPrewarmingManager>>,
    /// Statistics tracking
    stats: Arc<RwLock<TorRouterStats>>,
}

/// Statistics for Tor message routing
#[derive(Default, Debug, Clone)]
pub struct TorRouterStats {
    /// Messages routed through Tor
    pub messages_via_tor: u64,
    /// Messages routed via clearnet
    pub messages_via_clearnet: u64,
    /// Tor circuit errors
    pub tor_errors: u64,
    /// Average Tor latency (ms)
    pub avg_tor_latency_ms: f64,
    /// Last stats update time
    pub last_update: Option<std::time::Instant>,
}

impl TorMessageRouter {
    /// Create a new Tor message router
    pub async fn new(config: NetworkTorConfig) -> anyhow::Result<Self> {
        let (circuit_manager, prewarmer) = if config.tor_for_blocks
            || config.tor_for_peer_heights
            || config.tor_for_turbo_sync
            || config.tor_for_kademlia
        {
            info!("🧅 [TOR] Initializing dedicated circuit manager...");

            // Create DedicatedCircuitConfig from TorConfig
            let circuit_config = DedicatedCircuitConfig {
                data_directory: config.tor_config.data_dir.as_ref()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_else(|| "/var/lib/qnk/tor".to_string()),
                cache_directory: config.tor_config.cache_dir.as_ref()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_else(|| "/var/cache/qnk/tor".to_string()),
                tor_mandatory: config.tor_config.tor_only,
                ..DedicatedCircuitConfig::default()
            };

            let manager = Arc::new(
                DedicatedCircuitManager::new(circuit_config).await?
            );

            let prewarmer = if config.enable_prewarming {
                info!("🧅 [TOR] Starting circuit prewarmer...");
                Some(Arc::new(
                    CircuitPrewarmingManager::new(Arc::clone(&manager), config.prewarmer_config.clone())
                ))
            } else {
                None
            };

            (Some(manager), prewarmer)
        } else {
            info!("🧅 [TOR] Tor routing disabled - using clearnet only");
            (None, None)
        };

        Ok(Self {
            config,
            circuit_manager,
            prewarmer,
            stats: Arc::new(RwLock::new(TorRouterStats::default())),
        })
    }

    /// Determine the operation type for a gossipsub topic
    pub fn topic_to_operation(&self, topic: &str) -> OperationType {
        if topic.contains("/blocks") {
            OperationType::BlockPropagation
        } else if topic.contains("/peer-heights") {
            OperationType::PeerDiscovery
        } else if topic.contains("/turbo-sync") {
            OperationType::P2PSync
        } else if topic.contains("/tx") || topic.contains("/transactions") {
            OperationType::TransactionSubmission
        } else if topic.contains("/consensus") || topic.contains("/voting") {
            OperationType::ValidatorCommunication
        } else if topic.contains("/miner") {
            OperationType::General
        } else if topic.contains("/balance") {
            OperationType::TransactionSubmission // Privacy-sensitive
        } else if topic.contains("/ai") {
            OperationType::AIInference
        } else {
            OperationType::General // Default for unknown topics
        }
    }

    /// Check if a topic should be routed through Tor
    pub fn should_use_tor(&self, topic: &str) -> bool {
        if self.circuit_manager.is_none() {
            return false;
        }

        if topic.contains("/blocks") {
            self.config.tor_for_blocks
        } else if topic.contains("/peer-heights") {
            self.config.tor_for_peer_heights
        } else if topic.contains("/turbo-sync") {
            self.config.tor_for_turbo_sync
        } else if topic.contains("/miner") {
            self.config.tor_for_miner_stats
        } else if topic.contains("/balance") {
            self.config.tor_for_balance_updates
        } else {
            // Default: route through Tor if mandatory mode
            self.config.tor_config.tor_only
        }
    }

    /// Get circuit statistics
    pub async fn get_circuit_stats(&self) -> Option<ManagerStats> {
        if let Some(ref manager) = self.circuit_manager {
            Some(manager.get_manager_stats().await)
        } else {
            None
        }
    }

    /// Get router statistics
    pub async fn get_router_stats(&self) -> TorRouterStats {
        self.stats.read().await.clone()
    }

    /// Update statistics after routing a message
    pub async fn record_message(&self, via_tor: bool, latency_ms: Option<f64>) {
        let mut stats = self.stats.write().await;

        if via_tor {
            stats.messages_via_tor += 1;
            if let Some(latency) = latency_ms {
                // Running average
                let n = stats.messages_via_tor as f64;
                stats.avg_tor_latency_ms =
                    stats.avg_tor_latency_ms * (n - 1.0) / n + latency / n;
            }
        } else {
            stats.messages_via_clearnet += 1;
        }

        stats.last_update = Some(std::time::Instant::now());
    }

    /// Record a Tor error
    pub async fn record_tor_error(&self) {
        let mut stats = self.stats.write().await;
        stats.tor_errors += 1;
    }

    /// Get the circuit manager for direct access
    pub fn circuit_manager(&self) -> Option<Arc<DedicatedCircuitManager>> {
        self.circuit_manager.clone()
    }

    /// Check if Tor is enabled
    pub fn is_tor_enabled(&self) -> bool {
        self.circuit_manager.is_some()
    }

    /// Start the prewarmer background task
    pub async fn start_prewarmer(&self) {
        if let Some(ref prewarmer) = self.prewarmer {
            info!("🧅 [TOR] Starting circuit prewarmer background task...");
            prewarmer.start().await;
        }
    }

    /// Stop the prewarmer
    pub async fn stop_prewarmer(&self) {
        if let Some(ref prewarmer) = self.prewarmer {
            info!("🧅 [TOR] Stopping circuit prewarmer...");
            if let Err(e) = prewarmer.shutdown().await {
                warn!("🧅 [TOR] Error shutting down prewarmer: {}", e);
            }
        }
    }
}

/// Extension trait for UnifiedNetworkManager to add Tor support
#[async_trait::async_trait]
pub trait TorNetworkExt {
    /// Initialize Tor integration for the network manager
    async fn init_tor(&mut self, config: NetworkTorConfig) -> anyhow::Result<()>;

    /// Check if message should be sent via Tor
    fn should_route_via_tor(&self, topic: &str) -> bool;

    /// Get Tor statistics for metrics
    async fn get_tor_stats(&self) -> Option<TorRouterStats>;
}

/// Prometheus metrics for Tor integration
pub mod metrics {
    use prometheus::{IntCounter, IntGauge, Histogram, register_int_counter, register_int_gauge, register_histogram};
    use lazy_static::lazy_static;

    lazy_static! {
        /// Total messages routed through Tor
        pub static ref TOR_MESSAGES_TOTAL: IntCounter = register_int_counter!(
            "qnk_tor_messages_total",
            "Total messages routed through Tor circuits"
        ).expect("Failed to register metric");

        /// Total messages routed via clearnet
        pub static ref CLEARNET_MESSAGES_TOTAL: IntCounter = register_int_counter!(
            "qnk_clearnet_messages_total",
            "Total messages routed via clearnet"
        ).expect("Failed to register metric");

        /// Active Tor circuits
        pub static ref TOR_CIRCUITS_ACTIVE: IntGauge = register_int_gauge!(
            "qnk_tor_circuits_active",
            "Number of active Tor circuits"
        ).expect("Failed to register metric");

        /// Tor circuit errors
        pub static ref TOR_CIRCUIT_ERRORS: IntCounter = register_int_counter!(
            "qnk_tor_circuit_errors_total",
            "Total Tor circuit errors"
        ).expect("Failed to register metric");

        /// Tor message latency
        pub static ref TOR_LATENCY_HISTOGRAM: Histogram = register_histogram!(
            "qnk_tor_latency_seconds",
            "Tor message latency in seconds",
            vec![0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0]
        ).expect("Failed to register metric");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topic_to_operation() {
        let config = NetworkTorConfig::default();
        let router = TorMessageRouter {
            config,
            circuit_manager: None,
            prewarmer: None,
            stats: Arc::new(RwLock::new(TorRouterStats::default())),
        };

        assert!(matches!(
            router.topic_to_operation("/qnk/testnet/blocks"),
            OperationType::BlockPropagation
        ));

        assert!(matches!(
            router.topic_to_operation("/qnk/testnet/peer-heights"),
            OperationType::PeerDiscovery
        ));

        assert!(matches!(
            router.topic_to_operation("/qnk/testnet/turbo-sync"),
            OperationType::StateSync
        ));

        assert!(matches!(
            router.topic_to_operation("/qnk/testnet/tx"),
            OperationType::TransactionBroadcast
        ));
    }

    #[test]
    fn test_should_use_tor() {
        // With Tor disabled
        let config = NetworkTorConfig::from_env(); // Defaults to disabled
        let router = TorMessageRouter {
            config,
            circuit_manager: None,
            prewarmer: None,
            stats: Arc::new(RwLock::new(TorRouterStats::default())),
        };

        assert!(!router.should_use_tor("/qnk/testnet/blocks"));
    }

    #[test]
    fn test_config_profiles() {
        let minimal = NetworkTorConfig::minimal();
        assert!(minimal.tor_for_blocks);
        assert!(!minimal.tor_for_turbo_sync); // Disabled for performance

        let max_privacy = NetworkTorConfig::maximum_privacy();
        assert!(max_privacy.tor_config.tor_only);
    }
}
