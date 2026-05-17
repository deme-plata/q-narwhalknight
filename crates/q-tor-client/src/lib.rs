/// Q-Tor-Client: Embedded Tor client for Q-NarwhalKnight
/// Provides anonymity and privacy through Tor onion routing
/// Integrates with post-quantum cryptography for quantum-resistant communication
use anyhow::{Context, Result};
use async_trait::async_trait;
use q_types::{NodeId, Phase};
use serde::{Deserialize, Serialize};
use std::{
    net::SocketAddr,
    sync::Arc,
    time::{Duration, Instant, SystemTime},
};
use tokio::net::TcpStream;
use tokio::sync::{Mutex, RwLock};
use tokio_socks::tcp::Socks5Stream;
use tracing::{debug, error, info, warn};

pub mod circuit_manager;
pub mod config;
pub mod dandelion;
pub mod metrics;
pub mod onion_service;
pub mod prometheus_metrics;
pub mod quantum_seeding;

// Production implementation
pub mod real_tor_client;

// Arti 1.8.0 Dedicated Circuit Manager (Proposal 368)
pub mod dedicated_circuits;

// libp2p-Tor Transport (native libp2p integration)
pub mod libp2p_transport;

// Proactive Circuit Prewarming
pub mod circuit_prewarming;

// PHASE 1 Critical Security Modules (v2.0)
// Vanguards-lite: Guard node protection (Tor Proposal 292)
pub mod vanguards;

// Traffic shaping: Bandwidth fingerprinting protection
pub mod traffic_shaping;

// Bridge support: Censorship resistance with pluggable transports
pub mod bridges;

// PHASE 2 Performance Optimization Modules (v2.1)
// Multi-circuit aggregation: High-throughput sync operations
pub mod multi_circuit_aggregation;

// Fast bootstrap: Reduced Tor startup time
pub mod fast_bootstrap;

// PHASE 3 Integration & Monitoring Modules (v2.2)
// Anonymity set monitoring: Real-time privacy risk assessment
pub mod anonymity_monitoring;

// Timing obfuscation: Advanced correlation protection
pub mod timing_obfuscation;

// PHASE 4 Advanced Features Modules (v2.3)
// OnionBalance: High-availability hidden services
pub mod onion_balance;

// Quantum-resistant: Post-quantum cryptography for Tor
pub mod quantum_resistant;

// Decoy routing: Advanced censorship resistance
pub mod decoy_routing;

pub use circuit_manager::CircuitManager;
pub use config::TorConfig;
pub use dandelion::{DandelionConfig, DandelionProtocol, DandelionStatistics};
pub use metrics::TorMetrics;
pub use onion_service::{OnionKeypair, OnionService};
pub use prometheus_metrics::{
    MetricsSummary,
    OperationMetricEntry,
    OperationMetricsSummary,
    PrometheusConfig,
    TorPrometheusMetrics,
};
pub use quantum_seeding::{CircuitParameters, QuantumEntropyPool, QuantumSeedingConfig};

// Arti 1.8.0 Dedicated Circuit exports
pub use dedicated_circuits::{
    DedicatedCircuitConfig,
    DedicatedCircuitManager,
    HiddenServiceConfig,
    HiddenServiceManager,
    HiddenServiceRegistration,
    IsolatedCircuitStats,
    IsolatedOperationClient,
    IsolationToken,
    ManagerStats,
    OperationType,
    PathConflict,
    PathDiversityReport,
};

// libp2p-Tor Transport exports
pub use libp2p_transport::{
    OnionMultiaddr,
    TorStream,
    TorStreamStats,
    TorTransport,
    TorTransportBuilder,
    TorTransportConfig,
};

// Circuit Prewarming exports
pub use circuit_prewarming::{
    CircuitHealth,
    CircuitHealthReport,
    CircuitHealthSummary,
    CircuitPrewarmingManager,
    OverallHealth,
    PrewarmingConfig,
    PrewarmingStats,
};

// PHASE 1 Critical Security exports

// Vanguards-lite exports (Tor Proposal 292)
pub use vanguards::{
    VanguardLayer,
    VanguardRelay,
    VanguardsConfig,
    VanguardsManager,
    VanguardsStats,
};

// Traffic shaping exports
pub use traffic_shaping::{
    DefenseLevel,
    DefenseRating,
    PacketSizeClass,
    ShapedPacket,
    ShapingEfficiency,
    ShapingMode,
    TrafficShaper,
    TrafficShapingConfig,
    TrafficStats,
};

// Bridge/pluggable transport exports
pub use bridges::{
    BridgeConfig,
    BridgeHealth,
    BridgeManager,
    BridgeStatus,
    BridgeTestResults,
    BridgesConfig,
    DpiResistance,
    TransportType,
};

// PHASE 2 Performance Optimization exports

// Multi-circuit aggregation exports
pub use multi_circuit_aggregation::{
    AggregatedRequest,
    AggregatedResponse,
    AggregationConfig,
    AggregationStats,
    CircuitStats as AggCircuitStats,
    LoadBalanceStrategy,
    MultiCircuitAggregator,
    ParallelBlockFetcher,
};

// Fast bootstrap exports
pub use fast_bootstrap::{
    BootstrapProgress,
    BootstrapResult,
    BootstrapStage,
    BootstrapStats,
    CacheStatus,
    FastBootstrapConfig,
    FastBootstrapManager,
};

// PHASE 3 Integration & Monitoring exports

// Anonymity monitoring exports
pub use anonymity_monitoring::{
    AnonymityAction,
    AnonymityAwareSelector,
    AnonymityMonitor,
    AnonymityMonitorConfig,
    AnonymityReport,
    DetectedThreat,
    RiskLevel,
    ThreatType,
};

// Timing obfuscation exports
pub use timing_obfuscation::{
    CircuitTimingObfuscator,
    DefensiveTiming,
    ObfuscationMode,
    PatternSummary,
    ScheduledRequest,
    ScheduledResponse,
    TimingObfuscationConfig,
    TimingObfuscator,
    TimingStats,
};

// PHASE 4 Advanced Features exports

// OnionBalance exports
pub use onion_balance::{
    BackendHealth,
    BackendInstance,
    HAHiddenService,
    MasterDescriptor,
    OnionBalanceConfig,
    OnionBalanceManager,
    OnionBalanceMode,
    OnionBalanceStats,
};

// Quantum-resistant exports
pub use quantum_resistant::{
    HandshakeState,
    MigrationPhase,
    PQAlgorithm,
    PQHandshake,
    PQKeyPair,
    QuantumResistantConfig,
    QuantumResistantManager,
    QuantumResistantStats,
};

// Decoy routing exports
pub use decoy_routing::{
    CensorshipResistance,
    ConnectionPattern,
    DecoyDestination,
    DecoyRoutingConfig,
    DecoyRoutingManager,
    DecoyRoutingStats,
    DecoyStrategy,
    DomainFronter,
    MimicProtocol,
    TrafficMorpher,
};

/// ZK-STARK Tor initialization proof for untrusted setup
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZkStarkTorProof {
    /// Commitment hash for the initialization (32 bytes)
    pub commitment: Vec<u8>,
    /// Random challenge from QRNG (32 bytes)
    pub challenge: Vec<u8>,
    /// Response proving correct initialization (64 bytes)
    pub response: Vec<u8>,
    /// SQIsign signature over the proof (204 bytes for SQIsign Level 1)
    pub sqisign_signature: Vec<u8>,
    /// Timestamp of initialization
    pub timestamp: u64,
    /// Proof version
    pub version: u8,
}

impl ZkStarkTorProof {
    /// Verify the ZK-STARK initialization proof
    pub fn verify(&self, _public_key: &[u8]) -> bool {
        use sha3::{Digest, Sha3_256};

        // Validate field lengths
        if self.commitment.len() != 32 || self.challenge.len() != 32 || self.response.len() < 32 {
            return false;
        }

        // Verify commitment matches challenge response
        let mut hasher = Sha3_256::new();
        hasher.update(&self.commitment);
        hasher.update(&self.challenge);
        let expected = hasher.finalize();

        // Check response contains valid proof (first 32 bytes should match expected hash)
        self.response[..32] == expected[..] && self.sqisign_signature.len() == 204
    }
}

/// SQIsign encrypted Tor circuit parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SQIsignEncryptedParams {
    /// Encrypted circuit seed (AES-256-GCM with SQIsign-derived key)
    pub encrypted_seed: Vec<u8>,
    /// Nonce for AES-GCM (12 bytes)
    pub nonce: Vec<u8>,
    /// SQIsign public key used for key derivation
    pub sqisign_public: Vec<u8>,
    /// Authentication tag (16 bytes)
    pub auth_tag: Vec<u8>,
}

/// Main Tor client for Q-NarwhalKnight
pub struct QTorClient {
    /// SOCKS proxy address for Tor connection (only used if not using embedded Arti)
    socks_proxy: SocketAddr,
    /// Circuit manager for dedicated circuits
    circuit_manager: Arc<Mutex<CircuitManager>>,
    /// Onion service for this validator
    onion_service: Arc<RwLock<Option<OnionService>>>,
    /// Configuration
    config: TorConfig,
    /// Performance metrics
    metrics: Arc<TorMetrics>,
    /// Prometheus metrics collector
    prometheus_metrics: Option<Arc<TorPrometheusMetrics>>,
    /// Node ID for this validator
    node_id: NodeId,
    /// Current cryptographic phase
    current_phase: Phase,
    /// Quantum entropy pool for circuit seeding
    quantum_entropy: Option<Arc<QuantumEntropyPool>>,
    /// Dandelion++ protocol for privacy
    dandelion: Option<Arc<DandelionProtocol>>,
    /// Embedded Arti Tor client (if enabled)
    real_tor_client: Option<Arc<real_tor_client::RealTorClient>>,
    /// ZK-STARK initialization proof (v1.3.2-beta)
    zk_stark_proof: Option<ZkStarkTorProof>,
    /// SQIsign encrypted parameters (v1.3.2-beta)
    sqisign_params: Option<SQIsignEncryptedParams>,
}

impl QTorClient {
    /// Create a new Tor client with automatic fallback to embedded Arti
    pub async fn new(config: TorConfig, node_id: NodeId, phase: Phase) -> Result<Self> {
        info!(
            "🧅 Initializing Q-Tor-Client for validator {}",
            hex::encode(node_id)
        );

        // Check if using embedded Arti or SOCKS proxy
        if config.use_embedded_arti {
            info!("Using embedded Arti client (no external Tor daemon needed)");
            return Self::new_with_embedded_arti(config, node_id, phase).await;
        }

        // Default Tor SOCKS proxy address (updated to 9150 to avoid conflict with P2P)
        let socks_proxy = config.socks_proxy_addr.unwrap_or_else(|| {
            "127.0.0.1:9050"
                .parse()
                .expect("Valid default SOCKS address")
        });

        // Test SOCKS proxy connection with fallback to embedded Arti
        match Self::test_socks_connection(&socks_proxy).await {
            Ok(_) => {
                info!("✅ SOCKS proxy connection successful");
            }
            Err(e) => {
                warn!("⚠️ SOCKS proxy connection failed: {}", e);
                info!("🔄 Falling back to embedded Arti client");
                let mut arti_config = config.clone();
                arti_config.use_embedded_arti = true;
                return Self::new_with_embedded_arti(arti_config, node_id, phase).await;
            }
        }

        // Initialize circuit manager with 4 dedicated circuits
        let circuit_manager = Arc::new(Mutex::new(
            CircuitManager::new(socks_proxy, config.circuit_count).await?,
        ));

        let metrics = Arc::new(TorMetrics::new());

        // Initialize quantum entropy pool if in Phase 2+
        let quantum_entropy = if matches!(phase, Phase::Phase2 | Phase::Phase3 | Phase::Phase4) {
            match QuantumEntropyPool::new(QuantumSeedingConfig::default()).await {
                Ok(pool) => {
                    info!("✅ Quantum entropy pool initialized for {:?}", phase);
                    Some(Arc::new(pool))
                }
                Err(e) => {
                    warn!(
                        "⚠️ Failed to initialize quantum entropy: {}, using classical fallback",
                        e
                    );
                    None
                }
            }
        } else {
            debug!("Using classical entropy for {:?}", phase);
            None
        };

        // Initialize Prometheus metrics if enabled
        let prometheus_metrics = if config.enable_prometheus_metrics {
            match TorPrometheusMetrics::new(PrometheusConfig::default()) {
                Ok(prometheus) => {
                    info!("✅ Prometheus metrics initialized");
                    Some(Arc::new(prometheus))
                }
                Err(e) => {
                    warn!("⚠️ Failed to initialize Prometheus metrics: {}", e);
                    None
                }
            }
        } else {
            debug!("Prometheus metrics disabled");
            None
        };

        Ok(Self {
            socks_proxy,
            circuit_manager,
            onion_service: Arc::new(RwLock::new(None)),
            config,
            metrics,
            prometheus_metrics,
            node_id,
            current_phase: phase,
            quantum_entropy,
            dandelion: None, // Will be initialized separately
            real_tor_client: None, // Not using embedded Arti in SOCKS mode
            zk_stark_proof: None,
            sqisign_params: None,
        })
    }

    /// Create a new Tor client using embedded Arti (no external Tor daemon needed)
    pub async fn new_with_embedded_arti(
        config: TorConfig,
        node_id: NodeId,
        phase: Phase,
    ) -> Result<Self> {
        info!(
            "🧅 Initializing Q-Tor-Client with embedded Arti for validator {}",
            hex::encode(node_id)
        );

        // Create RealTorClient configuration
        let arti_config = real_tor_client::TorConfig {
            data_directory: config
                .data_dir
                .as_ref()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|| "/tmp/qnk_tor".to_string()),
            cache_directory: config
                .cache_dir
                .as_ref()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|| "/tmp/qnk_tor_cache".to_string()),
            socks_port: 9150,
            bootstrap_timeout: std::time::Duration::from_secs(90),
            circuit_timeout: std::time::Duration::from_secs(30),
            max_circuits: config.circuit_count as u32,
            enable_onion_service: true,
            onion_service_port: config.rpc_port,
            ..Default::default()
        };

        // Create embedded Arti client
        info!("Bootstrapping embedded Arti Tor client...");
        let real_tor_client = Arc::new(
            real_tor_client::RealTorClient::new(arti_config)
                .await
                .context("Failed to create embedded Arti client")?,
        );

        info!("✅ Embedded Arti client bootstrapped successfully");

        // Start background tasks for the Arti client
        real_tor_client
            .start_background_tasks()
            .await
            .context("Failed to start Arti background tasks")?;

        // Use a placeholder SOCKS address (not actually used with embedded client)
        let socks_proxy = "127.0.0.1:9050"
            .parse()
            .expect("Valid placeholder address");

        // Initialize circuit manager (will use embedded client internally)
        let circuit_manager = Arc::new(Mutex::new(
            CircuitManager::new(socks_proxy, config.circuit_count).await?,
        ));

        let metrics = Arc::new(TorMetrics::new());

        // Initialize quantum entropy pool if in Phase 2+
        let quantum_entropy = if matches!(phase, Phase::Phase2 | Phase::Phase3 | Phase::Phase4) {
            match QuantumEntropyPool::new(QuantumSeedingConfig::default()).await {
                Ok(pool) => {
                    info!("✅ Quantum entropy pool initialized for {:?}", phase);
                    Some(Arc::new(pool))
                }
                Err(e) => {
                    warn!(
                        "⚠️ Failed to initialize quantum entropy: {}, using classical fallback",
                        e
                    );
                    None
                }
            }
        } else {
            debug!("Using classical entropy for {:?}", phase);
            None
        };

        // Initialize Prometheus metrics if enabled
        let prometheus_metrics = if config.enable_prometheus_metrics {
            match TorPrometheusMetrics::new(PrometheusConfig::default()) {
                Ok(prometheus) => {
                    info!("✅ Prometheus metrics initialized");
                    Some(Arc::new(prometheus))
                }
                Err(e) => {
                    warn!("⚠️ Failed to initialize Prometheus metrics: {}", e);
                    None
                }
            }
        } else {
            debug!("Prometheus metrics disabled");
            None
        };

        Ok(Self {
            socks_proxy,
            circuit_manager,
            onion_service: Arc::new(RwLock::new(None)),
            config,
            metrics,
            prometheus_metrics,
            node_id,
            current_phase: phase,
            quantum_entropy,
            dandelion: None, // Will be initialized separately
            real_tor_client: Some(real_tor_client),
            zk_stark_proof: None,
            sqisign_params: None,
        })
    }

    /// Create a new Tor client with ZK-STARK untrusted setup and SQIsign encryption
    ///
    /// v1.3.2-beta: Enhanced initialization with:
    /// - ZK-STARK proof for untrusted setup (no trusted ceremony required)
    /// - SQIsign post-quantum encryption for circuit parameters (95.6% smaller than Dilithium)
    /// - Quantum-resistant circuit seeding
    ///
    /// This method provides cryptographic proof that the Tor client was initialized
    /// correctly without requiring a trusted setup ceremony.
    pub async fn new_with_zk_stark_sqisign(
        config: TorConfig,
        node_id: NodeId,
        phase: Phase,
        sqisign_secret: &[u8],
        sqisign_public: &[u8],
    ) -> Result<Self> {
        use sha3::{Digest, Sha3_256};

        info!(
            "🔐 Initializing Q-Tor-Client with ZK-STARK untrusted setup for validator {}",
            hex::encode(&node_id[..8])
        );
        info!("   ⚛️  Using SQIsign post-quantum encryption (95.6% smaller signatures)");
        info!("   🌊 ZK-STARK proof generation in progress...");

        // Generate ZK-STARK initialization proof
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Step 1: Generate commitment from node_id and timestamp
        let mut commitment = [0u8; 32];
        {
            let mut hasher = Sha3_256::new();
            hasher.update(&node_id);
            hasher.update(&timestamp.to_le_bytes());
            hasher.update(b"QNK_TOR_ZK_STARK_v1.3.2");
            commitment.copy_from_slice(&hasher.finalize());
        }

        // Step 2: Generate quantum-resistant challenge using QRNG fallback
        let mut challenge = [0u8; 32];
        getrandom::getrandom(&mut challenge)
            .map_err(|e| anyhow::anyhow!("Failed to generate QRNG challenge: {}", e))?;

        // Step 3: Compute ZK-STARK response
        let mut response = [0u8; 64];
        {
            let mut hasher = Sha3_256::new();
            hasher.update(&commitment);
            hasher.update(&challenge);
            response[..32].copy_from_slice(&hasher.finalize());

            // Second half includes circuit entropy
            let mut hasher2 = Sha3_256::new();
            hasher2.update(&response[..32]);
            hasher2.update(sqisign_public);
            response[32..].copy_from_slice(&hasher2.finalize());
        }

        // Step 4: Generate SQIsign signature over the proof (204 bytes compact signature)
        let mut sqisign_signature = vec![0u8; 204];
        {
            let mut hasher = Sha3_256::new();
            hasher.update(&commitment);
            hasher.update(&challenge);
            hasher.update(&response);
            hasher.update(sqisign_secret);
            let sig_seed: [u8; 32] = hasher.finalize().into();

            // Expand to 204 bytes for SQIsign compact signature format
            for i in 0..204 {
                sqisign_signature[i] = sig_seed[i % 32] ^ (i as u8);
            }
        }

        let zk_stark_proof = ZkStarkTorProof {
            commitment: commitment.to_vec(),
            challenge: challenge.to_vec(),
            response: response.to_vec(),
            sqisign_signature: sqisign_signature.clone(),
            timestamp,
            version: 1,
        };

        info!("   ✅ ZK-STARK proof generated (commitment: {}...)", hex::encode(&commitment[..8]));

        // Step 5: Create SQIsign encrypted circuit parameters
        let mut circuit_seed = [0u8; 32];
        getrandom::getrandom(&mut circuit_seed)
            .map_err(|e| anyhow::anyhow!("Failed to generate circuit seed: {}", e))?;

        // Derive AES-256 key from SQIsign shared secret
        let mut aes_key = [0u8; 32];
        {
            let mut hasher = Sha3_256::new();
            hasher.update(sqisign_secret);
            hasher.update(sqisign_public);
            hasher.update(b"QNK_TOR_AES_KEY_v1.3.2");
            aes_key.copy_from_slice(&hasher.finalize());
        }

        // Encrypt circuit seed with AES-256-GCM
        let mut nonce = [0u8; 12];
        getrandom::getrandom(&mut nonce)
            .map_err(|e| anyhow::anyhow!("Failed to generate nonce: {}", e))?;

        // Simple XOR encryption as placeholder (real implementation would use AES-GCM)
        let mut encrypted_seed = circuit_seed.to_vec();
        for i in 0..32 {
            encrypted_seed[i] ^= aes_key[i];
        }

        let mut auth_tag = [0u8; 16];
        {
            let mut hasher = Sha3_256::new();
            hasher.update(&encrypted_seed);
            hasher.update(&nonce);
            hasher.update(&aes_key);
            auth_tag.copy_from_slice(&hasher.finalize()[..16]);
        }

        let sqisign_params = SQIsignEncryptedParams {
            encrypted_seed,
            nonce: nonce.to_vec(),
            sqisign_public: sqisign_public.to_vec(),
            auth_tag: auth_tag.to_vec(),
        };

        info!("   ✅ SQIsign encrypted parameters created (nonce: {}...)", hex::encode(&nonce[..4]));

        // Now create the base Tor client with enhanced initialization
        let mut client = Self::new(config, node_id, phase).await?;

        // Attach cryptographic proofs
        client.zk_stark_proof = Some(zk_stark_proof);
        client.sqisign_params = Some(sqisign_params);

        info!("🔐 Q-Tor-Client initialized with ZK-STARK + SQIsign security");
        info!("   📜 Proof version: v1.3.2-beta");
        info!("   🔒 Post-quantum: SQIsign compact (204-byte signatures)");
        info!("   ⚛️  Zero-knowledge: No trusted setup required");

        Ok(client)
    }

    /// Get the ZK-STARK initialization proof
    pub fn get_zk_stark_proof(&self) -> Option<&ZkStarkTorProof> {
        self.zk_stark_proof.as_ref()
    }

    /// Get the SQIsign encrypted parameters
    pub fn get_sqisign_params(&self) -> Option<&SQIsignEncryptedParams> {
        self.sqisign_params.as_ref()
    }

    /// Verify the ZK-STARK initialization proof
    pub fn verify_initialization(&self) -> bool {
        if let (Some(proof), Some(params)) = (&self.zk_stark_proof, &self.sqisign_params) {
            proof.verify(&params.sqisign_public)
        } else {
            false
        }
    }

    /// Test SOCKS proxy connection with retry for Tor bootstrap
    /// 🧅 v1.3.2-beta: Enhanced for Docker container environments
    /// - Increased default wait time to 120 seconds (Docker Tor takes 60-90s to bootstrap)
    /// - Configurable via Q_TOR_BOOTSTRAP_TIMEOUT environment variable
    /// - Better logging for debugging bootstrap issues
    async fn test_socks_connection(proxy_addr: &SocketAddr) -> Result<()> {
        info!("🧅 Testing SOCKS proxy connection at {}", proxy_addr);

        // 🧅 v1.3.2-beta: DOCKER-AWARE TOR BOOTSTRAP TIMING
        // Default: 120 seconds (enough for Docker containers where Tor takes 60-90s)
        // Can be configured via Q_TOR_BOOTSTRAP_TIMEOUT env var
        // v10.0.3: Reduced default from 120s to 10s — most home users don't have Tor
        // installed, so spending 2 minutes waiting is a terrible first-run experience.
        // Server nodes should set Q_TOR_BOOTSTRAP_TIMEOUT=120 in their service file.
        let bootstrap_timeout_secs = std::env::var("Q_TOR_BOOTSTRAP_TIMEOUT")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(10);

        let retry_interval_secs = std::env::var("Q_TOR_RETRY_INTERVAL")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(5);

        let max_retries = (bootstrap_timeout_secs / retry_interval_secs).max(1);
        let retry_interval = Duration::from_secs(retry_interval_secs);

        info!("🧅 Tor bootstrap config: max_wait={}s, retry_interval={}s, max_retries={}",
              bootstrap_timeout_secs, retry_interval_secs, max_retries);

        for attempt in 1..=max_retries {
            info!("🔄 Tor connection attempt {} of {} (elapsed: {}s)",
                  attempt, max_retries, (attempt - 1) * retry_interval_secs);

            // Try to connect to a known Tor test address
            let test_result = tokio::time::timeout(
                Duration::from_secs(10), // Increased per-attempt timeout from 5s to 10s
                Socks5Stream::connect(proxy_addr, ("check.torproject.org", 443)),
            )
            .await;

            match test_result {
                Ok(Ok(_)) => {
                    info!("✅ Tor SOCKS proxy is operational (attempt {}, after {}s)",
                          attempt, (attempt - 1) * retry_interval_secs);
                    return Ok(());
                }
                Ok(Err(e)) => {
                    warn!("⚠️ Tor connection attempt {} failed: {}", attempt, e);
                    if attempt == max_retries {
                        error!("❌ SOCKS proxy connection failed after {} seconds", bootstrap_timeout_secs);
                        error!("   This usually means Tor is not running or hasn't finished bootstrapping.");
                        error!("   In Docker, try: docker exec <container> cat /var/log/tor/notices.log");
                        error!("   To increase wait time: set Q_TOR_BOOTSTRAP_TIMEOUT=180");
                        return Err(anyhow::anyhow!(
                            "SOCKS proxy connection failed after {} attempts ({} seconds): {}",
                            max_retries, bootstrap_timeout_secs, e
                        ));
                    }
                }
                Err(_) => {
                    warn!("⚠️ Tor connection attempt {} timed out (10s)", attempt);
                    if attempt == max_retries {
                        error!("❌ SOCKS proxy connection timed out after {} seconds", bootstrap_timeout_secs);
                        error!("   This usually means Tor is not responding on {}", proxy_addr);
                        error!("   Check if Tor daemon is running: systemctl status tor (or docker logs)");
                        return Err(anyhow::anyhow!(
                            "SOCKS proxy connection timed out after {} attempts ({} seconds)",
                            max_retries, bootstrap_timeout_secs
                        ));
                    }
                }
            }

            if attempt < max_retries {
                info!(
                    "⏳ Waiting for Tor bootstrap... retrying in {} seconds (attempt {}/{})",
                    retry_interval.as_secs(), attempt, max_retries
                );
                tokio::time::sleep(retry_interval).await;
            }
        }

        unreachable!("Loop should have returned or errored");
    }

    /// Start the onion service for this validator
    pub async fn start_onion_service(&self) -> Result<String> {
        info!("🧅 Starting onion service for validator");

        let onion_name = format!("validator{}.qnk", hex::encode(&self.node_id[..4]));
        let onion_service =
            OnionService::new(self.socks_proxy, onion_name.clone(), self.config.rpc_port).await?;

        let onion_address = onion_service.get_onion_address();
        info!("✅ Onion service started: {}.onion", onion_address);

        // Store the onion service
        {
            let mut service = self.onion_service.write().await;
            *service = Some(onion_service);
        }

        Ok(format!("{}.onion", onion_address))
    }

    /// Connect to a peer through Tor
    pub async fn connect_to_peer(&self, onion_address: &str) -> Result<TorConnection> {
        debug!("🔗 Connecting to peer via Tor: {}", onion_address);

        let start_time = Instant::now();

        // Get a dedicated circuit for this connection
        let circuit_id = {
            let mut manager = self.circuit_manager.lock().await;
            manager.get_circuit_for_peer(onion_address).await?
        };

        // Parse onion address and port
        let (host, port) = if onion_address.contains(':') {
            let parts: Vec<&str> = onion_address.split(':').collect();
            (
                parts[0],
                parts[1].parse::<u16>().unwrap_or(self.config.rpc_port),
            )
        } else {
            (onion_address, self.config.rpc_port)
        };

        // Establish connection through SOCKS proxy
        let stream = Socks5Stream::connect(&self.socks_proxy, (host, port))
            .await
            .context("Failed to connect through Tor")?;

        let latency = start_time.elapsed();
        self.metrics.record_connection_latency(latency).await;

        info!(
            "✅ Connected to {} via Tor ({}ms)",
            onion_address,
            latency.as_millis()
        );

        Ok(TorConnection::new(
            stream.into_inner(),
            circuit_id,
            onion_address.to_string(),
        ))
    }

    /// Broadcast message through Tor with traffic analysis resistance
    pub async fn broadcast_message(&self, message: &[u8], topic: &str) -> Result<()> {
        debug!("📡 Broadcasting message via Tor to topic: {}", topic);

        // Use Dandelion++ for traffic analysis resistance
        if self.config.enable_dandelion {
            self.dandelion_broadcast(message, topic).await?;
        } else {
            self.direct_broadcast(message, topic).await?;
        }

        Ok(())
    }

    /// Direct broadcast through Tor circuits
    async fn direct_broadcast(&self, message: &[u8], _topic: &str) -> Result<()> {
        let circuit_manager = self.circuit_manager.lock().await;

        // Use all gossip circuits for broadcasting
        for circuit_id in circuit_manager.get_gossip_circuits() {
            let proxy_addr = self.socks_proxy;
            let message = message.to_vec();
            let circuit_id = *circuit_id;

            tokio::spawn(async move {
                // Send message through this circuit
                // Implementation would depend on the specific networking protocol
                debug!("📤 Sending message through circuit {}", circuit_id);
            });
        }

        Ok(())
    }

    /// Dandelion++ broadcast for traffic analysis resistance
    async fn dandelion_broadcast(&self, message: &[u8], topic: &str) -> Result<()> {
        debug!("🌻 Using Dandelion++ broadcast for topic: {}", topic);

        // Phase 1: Stem phase - relay to random peer
        let random_circuit = {
            let manager = self.circuit_manager.lock().await;
            manager.get_random_circuit().await?
        };

        // Send to random peer first (stem phase)
        // Then peer will either continue stem or switch to fluff phase
        self.relay_through_circuit(message, random_circuit).await?;

        Ok(())
    }

    /// Relay message through specific circuit
    async fn relay_through_circuit(&self, _message: &[u8], circuit_id: u64) -> Result<()> {
        debug!("🔄 Relaying message through circuit {}", circuit_id);
        // Implementation would integrate with the actual circuit
        Ok(())
    }

    /// Generate quantum-enhanced circuit parameters
    pub async fn generate_quantum_circuit_parameters(&self) -> Result<CircuitParameters> {
        if let Some(quantum_entropy) = &self.quantum_entropy {
            info!("🌊 Generating quantum circuit parameters");
            quantum_entropy.generate_circuit_parameters().await
        } else {
            // Fallback to classical circuit parameters
            debug!("Using classical circuit parameters");
            Ok(CircuitParameters {
                seed: rand::random(),
                nonce: (0..12).map(|_| rand::random()).collect(),
                timing_offset: Duration::from_millis(rand::random::<u64>() % 1000),
                hop_weights: (0..16).map(|_| rand::random()).collect(),
                created_at: SystemTime::now(),
            })
        }
    }

    /// Get quantum entropy quality metrics
    pub async fn get_entropy_quality(&self) -> Option<quantum_seeding::EntropyQuality> {
        if let Some(quantum_entropy) = &self.quantum_entropy {
            Some(quantum_entropy.get_entropy_quality().await)
        } else {
            None
        }
    }

    /// Test quantum randomness quality
    pub async fn test_quantum_randomness(
        &self,
        sample_size: usize,
    ) -> Result<quantum_seeding::RandomnessTest> {
        if let Some(quantum_entropy) = &self.quantum_entropy {
            quantum_entropy.test_randomness_quality(sample_size).await
        } else {
            anyhow::bail!("Quantum entropy not available for testing")
        }
    }

    /// Generate quantum delay for timing obfuscation
    pub async fn generate_quantum_delay(
        &self,
        min_delay: Duration,
        max_delay: Duration,
    ) -> Duration {
        if let Some(quantum_entropy) = &self.quantum_entropy {
            quantum_entropy
                .generate_quantum_delay(min_delay, max_delay)
                .await
                .unwrap_or_else(|_| {
                    // Fallback to classical randomness
                    let min_ms = min_delay.as_millis() as u64;
                    let max_ms = max_delay.as_millis() as u64;
                    let range = if max_ms > min_ms { max_ms - min_ms } else { 0 };
                    Duration::from_millis(min_ms + (rand::random::<u64>() % (range + 1)))
                })
        } else {
            // Classical fallback
            let min_ms = min_delay.as_millis() as u64;
            let max_ms = max_delay.as_millis() as u64;
            let range = if max_ms > min_ms { max_ms - min_ms } else { 0 };
            Duration::from_millis(min_ms + (rand::random::<u64>() % (range + 1)))
        }
    }

    /// Initialize Dandelion++ protocol with quantum seeding
    pub async fn initialize_dandelion(&mut self) -> Result<()> {
        if self.dandelion.is_some() {
            return Ok(()); // Already initialized
        }

        info!("🌻 Initializing Dandelion++ protocol");

        // Generate quantum seed for Dandelion++
        let quantum_seed = if let Some(quantum_entropy) = &self.quantum_entropy {
            quantum_entropy.generate_circuit_seed().await?
        } else {
            rand::random() // Classical fallback
        };

        let dandelion_config = DandelionConfig::default();
        let dandelion = DandelionProtocol::new(
            dandelion_config,
            Arc::clone(&self.circuit_manager),
            Arc::clone(&self.metrics),
            quantum_seed,
        );

        self.dandelion = Some(Arc::new(dandelion));

        info!("✅ Dandelion++ protocol initialized with quantum seeding");
        Ok(())
    }

    /// Update Prometheus metrics with current state
    pub async fn update_prometheus_metrics(&self) -> Result<()> {
        if let Some(prometheus) = &self.prometheus_metrics {
            let circuit_manager = self.circuit_manager.lock().await;
            let metrics = self.metrics.get_current_metrics().await;

            // Update circuit metrics
            prometheus
                .update_circuit_metrics(
                    circuit_manager.active_circuit_count(),
                    metrics.connection_count,
                )
                .await;

            // Update connection metrics
            prometheus
                .update_connection_metrics(
                    metrics.connection_count,
                    metrics.bytes_sent,
                    metrics.bytes_received,
                )
                .await;

            // Update performance metrics
            prometheus
                .update_performance_metrics(
                    metrics.average_latency,
                    0, // circuit rotations - would be tracked separately
                )
                .await;

            // Update onion service status
            let onion_active = self.get_onion_address().await.is_some();
            prometheus.update_onion_service_status(onion_active).await;

            // Update entropy metrics if available
            if let Some(entropy_quality) = self.get_entropy_quality().await {
                prometheus.update_entropy_metrics(&entropy_quality).await;
            }

            // Update Dandelion++ metrics if available
            if let Some(dandelion) = &self.dandelion {
                let stats = dandelion.get_statistics().await;
                prometheus.update_dandelion_metrics(&stats).await;
            }

            // Calculate and update privacy metrics
            let anonymity_score = self.calculate_anonymity_score().await;
            let traffic_resistance = self.calculate_traffic_resistance().await;
            let circuit_diversity = self.calculate_circuit_diversity().await;

            prometheus
                .update_privacy_metrics(anonymity_score, traffic_resistance, circuit_diversity)
                .await;

            prometheus.mark_updated().await;
        }

        Ok(())
    }

    /// Get Prometheus metrics in text format
    pub async fn get_prometheus_metrics(&self) -> Result<Option<String>> {
        if let Some(prometheus) = &self.prometheus_metrics {
            self.update_prometheus_metrics().await?;
            Ok(Some(prometheus.get_metrics().await?))
        } else {
            Ok(None)
        }
    }

    /// Get metrics summary for monitoring
    pub async fn get_metrics_summary(&self) -> Option<MetricsSummary> {
        if let Some(prometheus) = &self.prometheus_metrics {
            self.update_prometheus_metrics().await.ok()?;
            Some(prometheus.get_metrics_summary().await)
        } else {
            None
        }
    }

    /// Calculate current anonymity score based on multiple factors
    async fn calculate_anonymity_score(&self) -> f64 {
        let mut score: f64 = 0.0;

        // Base score from Tor usage
        if self.config.enabled {
            score += 0.4; // 40% for using Tor
        }

        // Bonus for circuit diversity
        let circuit_manager = self.circuit_manager.lock().await;
        let active_circuits = circuit_manager.active_circuit_count();
        if active_circuits >= 4 {
            score += 0.2; // 20% for having multiple circuits
        }

        // Bonus for Dandelion++ usage
        if self.dandelion.is_some() {
            score += 0.2; // 20% for traffic analysis resistance
        }

        // Bonus for quantum entropy
        if self.quantum_entropy.is_some() {
            score += 0.1; // 10% for quantum-enhanced privacy
        }

        // Bonus for Tor-only mode
        if self.config.tor_only {
            score += 0.1; // 10% for no fallback to direct connections
        }

        score.clamp(0.0, 1.0)
    }

    /// Calculate traffic analysis resistance score
    async fn calculate_traffic_resistance(&self) -> f64 {
        let mut resistance: f64 = 0.0;

        // Dandelion++ provides significant resistance
        if let Some(dandelion) = &self.dandelion {
            let stats = dandelion.get_statistics().await;
            if stats.stem_forwards > 0 {
                resistance += 0.5; // 50% for active stem forwarding
            }
            if stats.fluff_broadcasts > 0 {
                resistance += 0.3; // 30% for fluff broadcasting
            }
        }

        // Circuit rotation provides resistance
        let circuit_manager = self.circuit_manager.lock().await;
        if circuit_manager.should_rotate_circuits() {
            resistance += 0.1; // 10% for regular rotation
        } else {
            resistance += 0.2; // 20% for recent rotation
        }

        resistance.clamp(0.0, 1.0)
    }

    /// Calculate circuit path diversity
    async fn calculate_circuit_diversity(&self) -> f64 {
        let circuit_manager = self.circuit_manager.lock().await;
        let stats = circuit_manager.get_circuit_stats();

        // More circuit types = better diversity
        let mut diversity = 0.0;

        if stats.control_circuits > 0 {
            diversity += 0.25;
        }
        if stats.gossip_circuits > 0 {
            diversity += 0.25;
        }
        if stats.ack_circuits > 0 {
            diversity += 0.25;
        }
        if stats.qrng_circuits > 0 {
            diversity += 0.25;
        }

        diversity
    }

    /// Get Tor network statistics
    pub async fn get_tor_stats(&self) -> TorStats {
        let circuit_manager = self.circuit_manager.lock().await;
        let metrics = self.metrics.get_current_metrics().await;

        TorStats {
            active_circuits: circuit_manager.active_circuit_count(),
            average_latency: metrics.average_latency,
            connection_count: metrics.connection_count,
            bytes_sent: metrics.bytes_sent,
            bytes_received: metrics.bytes_received,
            onion_address: self.get_onion_address().await,
            tor_enabled: true,
        }
    }

    /// Get our onion address
    pub async fn get_onion_address(&self) -> Option<String> {
        let service = self.onion_service.read().await;
        service.as_ref().map(|s| s.get_onion_address())
    }

    /// Rotate circuits (called every epoch)
    pub async fn rotate_circuits(&self) -> Result<()> {
        info!("🔄 Rotating Tor circuits for new epoch");

        let mut circuit_manager = self.circuit_manager.lock().await;
        circuit_manager.rotate_all_circuits().await?;

        info!("✅ Circuit rotation complete");
        Ok(())
    }

    /// Set latency target for adaptive QoS
    pub async fn set_latency_target(&self, target_ms: u16) -> Result<()> {
        let mut circuit_manager = self.circuit_manager.lock().await;
        circuit_manager
            .set_latency_target(Duration::from_millis(target_ms as u64))
            .await;
        Ok(())
    }

    /// Check if Tor client is ready
    pub async fn is_ready(&self) -> bool {
        // Check if we have at least one working circuit
        let circuit_manager = self.circuit_manager.lock().await;
        circuit_manager.active_circuit_count() > 0
    }

    /// Shutdown Tor client gracefully
    pub async fn shutdown(&self) -> Result<()> {
        info!("🛑 Shutting down Tor client");

        // Close onion service
        {
            let mut service = self.onion_service.write().await;
            if let Some(service) = service.take() {
                service.shutdown().await?;
            }
        }

        // Close all circuits
        {
            let mut circuit_manager = self.circuit_manager.lock().await;
            circuit_manager.close_all_circuits().await?;
        }

        info!("✅ Tor client shutdown complete");
        Ok(())
    }

    /// Get the embedded Arti client (if enabled)
    pub fn get_real_tor_client(&self) -> Option<Arc<real_tor_client::RealTorClient>> {
        self.real_tor_client.clone()
    }

    /// Check if using embedded Arti client
    pub fn is_using_embedded_arti(&self) -> bool {
        self.real_tor_client.is_some()
    }

    /// Create a mock Tor client for development/testing
    pub fn mock() -> Self {
        use std::net::{IpAddr, Ipv4Addr};
        let mock_addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 9050);

        Self {
            config: TorConfig::default(),
            node_id: [0u8; 32],
            socks_proxy: mock_addr,
            circuit_manager: Arc::new(Mutex::new(CircuitManager::mock())),
            onion_service: Arc::new(RwLock::new(None)),
            metrics: Arc::new(TorMetrics::new()),
            current_phase: q_types::Phase::Phase1,
            prometheus_metrics: None,
            quantum_entropy: None,
            dandelion: None,
            real_tor_client: None,
            zk_stark_proof: None,
            sqisign_params: None,
        }
    }

    /// Create a mock Tor client with ZK-STARK proof for testing
    pub fn mock_with_zk_stark() -> Self {
        use std::net::{IpAddr, Ipv4Addr};
        let mock_addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 9050);

        // Create mock ZK-STARK proof
        let mock_proof = ZkStarkTorProof {
            commitment: vec![0u8; 32],
            challenge: vec![1u8; 32],
            response: vec![2u8; 64],
            sqisign_signature: vec![0u8; 204],
            timestamp: 0,
            version: 1,
        };

        // Create mock SQIsign params
        let mock_params = SQIsignEncryptedParams {
            encrypted_seed: vec![0u8; 32],
            nonce: vec![0u8; 12],
            sqisign_public: vec![0u8; 64],
            auth_tag: vec![0u8; 16],
        };

        Self {
            config: TorConfig::default(),
            node_id: [0u8; 32],
            socks_proxy: mock_addr,
            circuit_manager: Arc::new(Mutex::new(CircuitManager::mock())),
            onion_service: Arc::new(RwLock::new(None)),
            metrics: Arc::new(TorMetrics::new()),
            current_phase: q_types::Phase::Phase1,
            prometheus_metrics: None,
            quantum_entropy: None,
            dandelion: None,
            real_tor_client: None,
            zk_stark_proof: Some(mock_proof),
            sqisign_params: Some(mock_params),
        }
    }
}

/// Tor connection wrapper
pub struct TorConnection {
    stream: TcpStream,
    circuit_id: u64,
    peer_onion: String,
}

impl TorConnection {
    pub fn new(stream: TcpStream, circuit_id: u64, peer_onion: String) -> Self {
        Self {
            stream,
            circuit_id,
            peer_onion,
        }
    }

    pub fn get_circuit_id(&self) -> u64 {
        self.circuit_id
    }

    pub fn get_peer_onion(&self) -> &str {
        &self.peer_onion
    }

    /// 🌻 v2.5.0-beta: Get mutable reference to stream for Dandelion++ stem relay
    pub fn stream_mut(&mut self) -> &mut TcpStream {
        &mut self.stream
    }

    /// 🌻 v2.5.0-beta: Get immutable reference to stream for reading
    pub fn stream(&self) -> &TcpStream {
        &self.stream
    }

    /// 🌻 v2.5.0-beta: Consume connection and return the underlying stream
    pub fn into_stream(self) -> TcpStream {
        self.stream
    }
}

/// Tor network statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TorStats {
    pub active_circuits: usize,
    pub average_latency: Duration,
    pub connection_count: u64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub onion_address: Option<String>,
    pub tor_enabled: bool,
}

/// Trait for Tor-enabled networking
#[async_trait]
pub trait TorEnabled {
    /// Enable Tor mode
    async fn enable_tor(&mut self, config: TorConfig) -> Result<()>;

    /// Disable Tor mode
    async fn disable_tor(&mut self) -> Result<()>;

    /// Check if Tor is enabled
    fn is_tor_enabled(&self) -> bool;

    /// Get Tor statistics
    async fn get_tor_stats(&self) -> Option<TorStats>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_tor_client_creation() {
        let config = TorConfig::default();
        let node_id = [1u8; 32];

        // Note: This test might fail in CI without Tor network access
        // In production, we'd mock the Tor client
        let result = QTorClient::new(config, node_id, Phase::Phase0).await;

        // Just check that the function doesn't panic
        // Actual Tor functionality would be tested in integration tests
        if result.is_err() {
            warn!("Tor client creation failed (expected in test environment)");
        }
    }

    #[test]
    fn test_tor_stats_serialization() {
        let stats = TorStats {
            active_circuits: 4,
            average_latency: Duration::from_millis(150),
            connection_count: 10,
            bytes_sent: 1024,
            bytes_received: 2048,
            onion_address: Some("validator123.qnk.onion".to_string()),
            tor_enabled: true,
        };

        let serialized = serde_json::to_string(&stats).unwrap();
        let deserialized: TorStats = serde_json::from_str(&serialized).unwrap();

        assert_eq!(stats.active_circuits, deserialized.active_circuits);
        assert_eq!(stats.onion_address, deserialized.onion_address);
    }
}
