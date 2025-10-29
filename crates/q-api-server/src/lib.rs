// DEACTIVATED: use q_bep44_discovery::DiscoveryEngine;
// DEACTIVATED: use q_bitcoin_bridge::bridge::IntegratedBitcoinBridge;
// DEACTIVATED: use q_dns_phantom::DNSPhantomNetwork;
use q_network::NetworkManager;
use q_storage::{StorageConfig, StorageEngine};
use q_tor_client::QTorClient; // Re-enabled for consensus integration
use q_types::*;
use q_wallet::{MemoryWalletStore, WalletManager};

// ZK Privacy Components - ✅ ENABLED
use q_zk_stark::StarkSystem;
use q_zk_snark::UniversalSNARK;

// Performance & Scaling Components
// use q_sharding::{ShardCoordinator, ShardManager}; // Temporarily disabled
// use q_cache::{HierarchicalCache, CacheManager}; // Temporarily disabled

// Consensus & DAG Components
use q_dag_knight::{DAGKnightConsensus, QuantumAnchorElection};
use q_narwhal_core::{NarwhalCore, ReliableBroadcast};
use q_narwhal_core::production_mempool::ProductionMempool;
use q_resonance::{KParameterAnalyzer, ResonanceCoordinator, KParameterMetrics, PhaseTransition};
use q_vdf::{QuantumVDF, VDFProof};

// Crypto & Security
use q_quantum_crypto::{BB84Protocol, QKDEngine, QuantumCryptoEngine};
use q_quantum_mixing::{QuantumMixingEngine, QuantumZKPProver};

// DeFi Components
// use q_dex::{DEXEngine, LiquidityPool}; // Temporarily disabled
// Quantum Oracle - AI-Enhanced Price Aggregation
// use q_oracle; // Temporarily disabled
// use q_stablecoin::{StablecoinManager, CollateralManager}; // Temporarily disabled
use q_quillon_bank::QuillonBankSystem; // ✅ ENABLED - Full quantum banking system

// Network & Infrastructure
use q_tor_circuit::{CircuitPool, DedicatedCircuitManager};
// use q_robot_control::{RobotFleet, SwarmIntelligence}; // Temporarily disabled
use q_network::{CryptoProvider, QuantumNetwork};
use libp2p::PeerId;

// Plugin System
use q_plugin_system::{PluginManager, PluginSystem, PluginSystemConfig};

// Sharding System
use q_sharding::{ShardConfig, ShardMetrics, ShardingEngine, ShardingStrategy};

// VM and Smart Contracts
use q_vm::contracts::{ContractRegistry, OrobitSmartContractEcosystem};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;
use uuid::Uuid;

pub mod config;
pub mod console_viz;  // Beautiful animated console visualization
pub mod dex_integration_api;
#[cfg(test)]
pub mod dex_integration_tests;
pub mod liquidity_api;  // Liquidity provision API
pub mod quillon_bank_api;  // ✅ ENABLED - Full Quillon Bank CDP system
pub mod cdp_simple;  // Simple CDP system for QUGUSD minting (fallback, can be removed)
pub mod handlers;
pub mod p2p_listener;
pub mod streaming;
pub mod binary_protocol;  // High-performance binary ingestion for 1M+ TPS
pub mod high_performance_server;  // HTTP/2 server optimized for 1M+ TPS
pub mod websocket_stream;  // WebSocket streaming for 1M+ TPS (zero HTTP overhead)
pub mod wallet_auth;  // Signature-based wallet authentication for privacy
pub mod storage_api;  // IPFS-RocksDB decentralized storage for database backups
pub mod database_replication_bridge;  // Bridge between IPFS replication and gossipsub
pub mod payment_api;  // ✅ ENABLED - Stripe payment processing with async-stripe
pub mod oauth2_provider;  // ✅ ENABLED - OAuth2 provider for third-party integrations
pub mod privacy_service_api;  // ✅ ENABLED - Privacy-as-a-Service (PaaS) enterprise API
pub mod paas_auth;  // ✅ ENABLED - PaaS authentication and rate limiting
pub mod paas_api_keys;  // ✅ ENABLED - API key management with argon2id hashing
pub mod paas_pricing;  // ✅ ENABLED - Dynamic USD pricing with oracle integration
pub mod paas_billing;  // ✅ ENABLED - Atomic billing with pre-charge and reserve
pub mod paas_billing_v2;  // ✅ ENABLED - Atomic billing v2 with Grok improvements
pub mod paas_idempotency;  // ✅ ENABLED - Idempotency support for safe retries
pub mod paas_audit;  // ✅ ENABLED - Audit logging and distributed tracing
pub mod paas_admin_api;  // ✅ ENABLED - PaaS admin endpoints for CLI management
pub mod aegis_auth_middleware;  // ✅ ENABLED - AEGIS-QL post-quantum authentication for founder operations
pub mod chat_api;  // ✅ ENABLED - AI chat API with privacy-first distributed inference
// pub mod supply_persistence;  // 🔒 DEACTIVATED - Will be implemented in v0.0.10
// io_uring is Linux kernel's async I/O interface (requires Linux kernel ≥5.1)
#[cfg(target_os = "linux")]
pub mod io_uring_adapter; // Safe io_uring wrapper to avoid runtime conflicts
pub mod parallel_workers; // 16x parallel worker pool for high TPS
pub mod block_producer;  // 🏗️ Block producer - aggregates mining solutions into QBlocks

pub use config::Config;
pub use console_viz::{ConsoleVisualizer, ConsensusStats, update_stats};
pub use streaming::{EventBroadcaster, HighPerformanceEmitter, StreamEvent};

/// Faucet request tracking for IP-based rate limiting
#[derive(Debug, Clone)]
pub struct FaucetRequestRecord {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub address: String,
    pub amount: u64,
}

/// Smart detection patterns for scripted abuse
#[derive(Debug, Clone)]
pub struct AbusePattern {
    pub requests_per_minute: u32,
    pub address_pattern_score: f64,
    pub timing_regularity_score: f64,
    pub user_agent_consistency: bool,
}

/// Daily faucet state with comprehensive protection
#[derive(Debug)]
pub struct FaucetState {
    /// Daily total coins distributed (resets at midnight UTC)
    pub daily_total_distributed: u64,
    /// Daily limit (1000 coins)
    pub daily_limit: u64,
    /// Last reset date for daily tracking
    pub last_reset_date: chrono::NaiveDate,
    /// IP-based request tracking (IP -> Vec<FaucetRequestRecord>)
    pub ip_requests: HashMap<String, Vec<FaucetRequestRecord>>,
    /// Address-based request tracking (address -> last_request_time)
    pub address_requests: HashMap<String, chrono::DateTime<chrono::Utc>>,
    /// Smart abuse detection patterns (IP -> AbusePattern)
    pub abuse_patterns: HashMap<String, AbusePattern>,
    /// Temporarily blacklisted IPs with expiration time
    pub blacklisted_ips: HashMap<String, chrono::DateTime<chrono::Utc>>,
}

/// Pending quantum mixing request
#[derive(Debug, Clone)]
pub struct PendingMixingRequest {
    pub participant_id: String,
    pub amount: u64,
    pub output_addresses: Vec<String>,
    pub privacy_level: q_types::PrivacyLevel,
    pub decoy_count: u32,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Liquidity pool structure
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LiquidityPool {
    pub pool_id: String,
    pub token0: String,  // Native QUG or token contract address
    pub token1: String,  // Token contract address
    pub reserve0: u64,
    pub reserve1: u64,
    pub provider: [u8; 32],  // Wallet address that provided liquidity
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Mining submission for async queue processing
#[derive(Debug, Clone)]
pub struct MiningSubmission {
    pub nonce: u64,
    pub hash: [u8; 32],
    pub difficulty_target: [u8; 32],
    pub miner_address: [u8; 32],
    pub miner_address_str: String,
    pub hash_rate: f64, // Hash rate in KH/s
}

impl Default for FaucetState {
    fn default() -> Self {
        Self {
            daily_total_distributed: 0,
            daily_limit: 1000_000_000_000, // 1000 QNK in base units
            last_reset_date: chrono::Utc::now().date_naive(),
            ip_requests: HashMap::new(),
            address_requests: HashMap::new(),
            abuse_patterns: HashMap::new(),
            blacklisted_ips: HashMap::new(),
        }
    }
}

impl FaucetState {
    /// Create a new FaucetState instance
    pub fn new() -> Self {
        Self::default()
    }
    /// Check if daily reset is needed and reset if so
    pub fn maybe_reset_daily(&mut self) {
        let today = chrono::Utc::now().date_naive();
        if today != self.last_reset_date {
            self.daily_total_distributed = 0;
            self.last_reset_date = today;
            // Clean up old IP requests (keep only last 24 hours)
            let cutoff = chrono::Utc::now() - chrono::Duration::hours(24);
            for requests in self.ip_requests.values_mut() {
                requests.retain(|req| req.timestamp > cutoff);
            }
            self.ip_requests.retain(|_, requests| !requests.is_empty());
            // Clean up expired blacklisted IPs
            let now = chrono::Utc::now();
            self.blacklisted_ips.retain(|_, expires| *expires > now);
        }
    }

    /// Check if IP is rate limited (max 10 requests per hour)
    pub fn is_ip_rate_limited(&self, ip: &str) -> bool {
        let one_hour_ago = chrono::Utc::now() - chrono::Duration::hours(1);
        if let Some(requests) = self.ip_requests.get(ip) {
            let recent_requests = requests.iter()
                .filter(|req| req.timestamp > one_hour_ago)
                .count();
            return recent_requests >= 10;
        }
        false
    }

    /// Check if address has requested within cooldown period (24 hours)
    pub fn is_address_in_cooldown(&self, address: &str) -> bool {
        let cooldown_period = chrono::Duration::hours(24);
        if let Some(last_request) = self.address_requests.get(address) {
            let now = chrono::Utc::now();
            return now.signed_duration_since(*last_request) < cooldown_period;
        }
        false
    }

    /// Check if IP is blacklisted due to abuse
    pub fn is_ip_blacklisted(&self, ip: &str) -> bool {
        if let Some(expires) = self.blacklisted_ips.get(ip) {
            return chrono::Utc::now() < *expires;
        }
        false
    }

    /// Analyze patterns to detect scripted abuse
    pub fn analyze_abuse_patterns(&mut self, ip: &str) -> bool {
        let one_hour_ago = chrono::Utc::now() - chrono::Duration::hours(1);
        if let Some(requests) = self.ip_requests.get(ip) {
            let recent_requests: Vec<_> = requests.iter()
                .filter(|req| req.timestamp > one_hour_ago)
                .collect();

            if recent_requests.len() >= 5 {
                let mut pattern = AbusePattern {
                    requests_per_minute: (recent_requests.len() * 60 / 3600) as u32,
                    address_pattern_score: 0.0,
                    timing_regularity_score: 0.0,
                    user_agent_consistency: true,
                };

                // Calculate address pattern score (random vs sequential patterns)
                let addresses: Vec<_> = recent_requests.iter().map(|r| &r.address).collect();
                let unique_addresses = addresses.iter().collect::<std::collections::HashSet<_>>().len();
                pattern.address_pattern_score = if unique_addresses > 1 {
                    // Check for sequential patterns in addresses
                    let mut sequential_score = 0.0;
                    for i in 1..addresses.len() {
                        if let (Ok(prev), Ok(curr)) = (
                            u64::from_str_radix(&addresses[i-1].replace("qnk", "").chars().take(16).collect::<String>(), 16),
                            u64::from_str_radix(&addresses[i].replace("qnk", "").chars().take(16).collect::<String>(), 16)
                        ) {
                            if curr.saturating_sub(prev) <= 10 {
                                sequential_score += 1.0;
                            }
                        }
                    }
                    sequential_score / (addresses.len() - 1) as f64
                } else {
                    0.0
                };

                // Calculate timing regularity score
                if recent_requests.len() >= 3 {
                    let mut intervals = Vec::new();
                    for i in 1..recent_requests.len() {
                        let interval = recent_requests[i].timestamp
                            .signed_duration_since(recent_requests[i-1].timestamp)
                            .num_seconds();
                        intervals.push(interval.abs());
                    }
                    let avg_interval = intervals.iter().sum::<i64>() as f64 / intervals.len() as f64;
                    let variance = intervals.iter()
                        .map(|&x| (x as f64 - avg_interval).powi(2))
                        .sum::<f64>() / intervals.len() as f64;
                    pattern.timing_regularity_score = 1.0 / (1.0 + variance / 100.0);
                }

                // Determine if this is likely scripted abuse
                let is_abuse = pattern.requests_per_minute > 5
                    || pattern.address_pattern_score > 0.7
                    || pattern.timing_regularity_score > 0.8;

                self.abuse_patterns.insert(ip.to_string(), pattern);

                if is_abuse {
                    // Blacklist IP for 24 hours
                    let expires = chrono::Utc::now() + chrono::Duration::hours(24);
                    self.blacklisted_ips.insert(ip.to_string(), expires);
                    return true;
                }
            }
        }
        false
    }

    /// Record a faucet request
    pub fn record_request(&mut self, ip: &str, address: &str, amount: u64) {
        let now = chrono::Utc::now();
        let request = FaucetRequestRecord {
            timestamp: now,
            address: address.to_string(),
            amount,
        };

        self.ip_requests.entry(ip.to_string())
            .or_insert_with(Vec::new)
            .push(request);
        
        self.address_requests.insert(address.to_string(), now);
        self.daily_total_distributed += amount;
    }
}

/// 🔒 Supply Consensus State - Post-Quantum Protected Max Supply Enforcement
/// This structure ensures that the 21M QNK supply cap is enforced across
/// the decentralized network using libp2p consensus and Dilithium5 signatures
#[derive(Debug, Clone)]
pub struct SupplyConsensusState {
    /// Last known supply from network consensus
    pub network_agreed_supply: u64,
    /// Timestamp of last consensus update
    pub last_consensus_timestamp: u64,
    /// Number of nodes that agreed on current supply
    pub consensus_node_count: usize,
    /// Dilithium5 signature from validator set
    pub validator_signature: Option<Vec<u8>>,
    /// libp2p peer IDs that validated this supply
    pub validating_peers: Vec<String>,
}

impl Default for SupplyConsensusState {
    fn default() -> Self {
        Self {
            network_agreed_supply: 0,
            last_consensus_timestamp: 0,
            consensus_node_count: 0,
            validator_signature: None,
            validating_peers: Vec::new(),
        }
    }
}

/// Mining statistics tracking for real-time network hash rate calculation
#[derive(Debug, Clone)]
pub struct MinerStats {
    pub address: String,
    pub last_hashrate: f64, // KH/s
    pub last_update: std::time::Instant,
    pub total_solutions: u64,
}

#[derive(Debug, Clone)]
pub struct MiningStatistics {
    pub total_solutions_submitted: u64,
    pub total_solutions_accepted: u64,
    pub active_miners: HashMap<String, MinerStats>,
    pub last_cleanup: std::time::Instant,
}

impl Default for MiningStatistics {
    fn default() -> Self {
        Self {
            total_solutions_submitted: 0,
            total_solutions_accepted: 0,
            active_miners: HashMap::new(),
            last_cleanup: std::time::Instant::now(),
        }
    }
}

impl MiningStatistics {
    /// Update miner statistics with new submission
    pub fn update_miner(&mut self, miner_address: String, hash_rate: f64) {
        let stats = self.active_miners.entry(miner_address.clone()).or_insert(MinerStats {
            address: miner_address,
            last_hashrate: 0.0,
            last_update: std::time::Instant::now(),
            total_solutions: 0,
        });

        stats.last_hashrate = hash_rate;
        stats.last_update = std::time::Instant::now();
        stats.total_solutions += 1;
        self.total_solutions_submitted += 1;
    }

    /// Calculate total network hash rate from active miners
    pub fn calculate_network_hashrate(&mut self) -> f64 {
        // Clean up stale miners (no activity in last 5 minutes)
        let now = std::time::Instant::now();
        if now.duration_since(self.last_cleanup).as_secs() > 60 {
            self.active_miners.retain(|_, stats| {
                now.duration_since(stats.last_update).as_secs() < 300
            });
            self.last_cleanup = now;
        }

        // Sum hash rates from all active miners
        self.active_miners.values()
            .map(|stats| stats.last_hashrate)
            .sum()
    }

    /// Get count of active miners
    pub fn active_miner_count(&self) -> usize {
        let now = std::time::Instant::now();
        self.active_miners.values()
            .filter(|stats| now.duration_since(stats.last_update).as_secs() < 300)
            .count()
    }
}

/// Application state shared across handlers
pub struct AppState {
    pub config: Config,
    pub node_id: NodeId,
    pub wallet_manager: WalletManager,
    pub node_status: Arc<RwLock<NodeStatus>>,
    // PERFORMANCE: Replaced RwLock<HashMap> with DashMap for lock-free concurrency (20-40K TPS target)
    pub tx_pool: Arc<dashmap::DashMap<TxHash, Transaction>>,
    pub tx_status: Arc<dashmap::DashMap<TxHash, TxStatus>>,
    pub blocks: Arc<RwLock<HashMap<Height, Vec<Transaction>>>>,
    pub wallet_balances: Arc<RwLock<HashMap<Address, Amount>>>, // Address -> Balance mapping
    // Password hashes: wallet_address -> bcrypt_hash (for secure login)
    pub wallet_password_hashes: Arc<RwLock<HashMap<Address, String>>>,
    // Token balances: (wallet_address, token_contract_address) -> token_amount
    pub token_balances: Arc<RwLock<HashMap<([u8; 32], [u8; 32]), u64>>>,
    // Liquidity pools: pool_id -> (token0, token1, reserve0, reserve1, provider)
    pub liquidity_pools: Arc<RwLock<HashMap<String, LiquidityPool>>>,
    // Nitro boosts: token_id -> total_boost_points (aggregated from all wallets)
    pub nitro_boosts: Arc<RwLock<HashMap<String, u64>>>,
    pub storage_engine: Arc<StorageEngine>, // Persistent storage for balances and state
    pub event_broadcaster: Arc<EventBroadcaster>,
    pub event_emitter: Arc<HighPerformanceEmitter>,

    // Faucet system with rate limiting and abuse protection
    pub faucet_state: Arc<RwLock<FaucetState>>,

    // 🔒 MAX SUPPLY ENFORCEMENT - Post-Quantum Consensus Protected
    // Total supply tracking with Dilithium5 signature verification
    pub total_minted_supply: Arc<RwLock<u64>>, // Total QNK minted across all wallets
    pub supply_consensus_state: Arc<RwLock<SupplyConsensusState>>, // libp2p consensus state

    // ⛏️ MINING STATISTICS - Real-time network hash rate tracking
    pub mining_statistics: Option<Arc<RwLock<MiningStatistics>>>, // Track active miners and hash rates

    // Quantum Privacy Mixer State
    pub mixing_requests: Arc<RwLock<HashMap<String, PendingMixingRequest>>>, // participant_id -> request
    pub quantum_mixer: Option<Arc<QuantumMixingEngine>>,
    pub zkp_prover: Option<Arc<QuantumZKPProver>>,

    // Network components
    // DEACTIVATED: pub bitcoin_bridge: Option<Arc<IntegratedBitcoinBridge>>,
    pub bitcoin_bridge: Option<Arc<()>>, // DEACTIVATED placeholder
    // DEACTIVATED: pub dns_phantom: Option<Arc<q_dns_phantom::node_integration::DNSPhantomNode>>,
    pub dns_phantom: Option<Arc<()>>, // DEACTIVATED placeholder
    // DEACTIVATED: pub bep44_discovery: Option<Arc<tokio::sync::Mutex<q_bep44_discovery::DiscoveryEngine>>>,
    pub bep44_discovery: Option<Arc<()>>, // DEACTIVATED placeholder
    pub tor_client: Option<Arc<QTorClient>>,
    pub network_manager: Option<Arc<q_network::NetworkManager>>,
    pub production_peer_discovery: Option<Arc<tokio::sync::Mutex<q_network::real_peer_discovery::RealPeerDiscovery>>>,

    // libp2p-based zero-config peer discovery (mDNS + Gossipsub)
    pub libp2p_discovery: Option<Arc<tokio::sync::Mutex<q_network::UnifiedNetworkManager>>>,

    // libp2p network command channel (for non-blocking P2P operations from API)
    pub libp2p_command_tx: Option<tokio::sync::mpsc::UnboundedSender<q_network::NetworkCommand>>,

    // Cached libp2p peer info for fast API access (updated by event loop)
    pub libp2p_peer_info: Arc<RwLock<(String, Vec<String>)>>, // (peer_id, listen_addresses)

    // Atomic peer count for fast lock-free access
    pub libp2p_peer_count: Option<Arc<std::sync::atomic::AtomicUsize>>,

    // Mining submission queue (async processing to prevent server overload)
    pub mining_submission_tx: Option<tokio::sync::mpsc::UnboundedSender<MiningSubmission>>,

    // BREAKTHROUGH: DNS-Phantom → Connection Integration
    pub connection_manager: Option<Arc<q_network::connection_manager::ConnectionManager>>,

    // DAG State Synchronization
    pub dag_sync_manager: Option<Arc<q_network::DagSyncManager>>,

    // ZK Privacy Components - ✅ ENABLED
    pub zk_stark_system: Option<Arc<tokio::sync::Mutex<StarkSystem>>>,
    pub zk_snark_system: Option<Arc<UniversalSNARK>>,

    // Performance & Scaling Optimizations
    pub simd_crypto_engine: Option<Arc<q_crypto_simd::SimdCryptoEngine>>,
    #[cfg(target_os = "linux")]
    pub kernel_io_engine: Option<Arc<crate::io_uring_adapter::IoUringAdapter>>,
    // pub shard_coordinator: Option<Arc<ShardCoordinator>>, // Temporarily disabled
    // pub shard_manager: Option<Arc<ShardManager>>, // Temporarily disabled
    // pub cache_manager: Option<Arc<CacheManager>>, // Temporarily disabled
    // pub hierarchical_cache: Option<Arc<HierarchicalCache>>, // Temporarily disabled

    // Consensus & DAG
    pub dag_knight: Option<Arc<DAGKnightConsensus>>,
    pub anchor_election: Option<Arc<QuantumAnchorElection>>,
    pub narwhal_core: Option<Arc<NarwhalCore>>,
    pub production_mempool: Option<Arc<ProductionMempool>>, // HIGH-PERFORMANCE MEMPOOL FOR 200K+ TPS
    pub reliable_broadcast: Option<Arc<ReliableBroadcast>>,
    pub quantum_vdf: Option<Arc<QuantumVDF>>,

    // PHASE 2: Parallel Block Production - Multiple producers for concurrent block creation
    pub block_producer_pool: Arc<crate::block_producer::ParallelBlockProducerPool>,

    // PHASE 3: DAG-Knight Consensus - Byzantine Fault-Tolerant Block Ordering
    pub consensus: Arc<RwLock<DAGKnightConsensus>>,

    // Quillon Resonance Consensus - K-Parameter Phase Analysis
    pub k_parameter_analyzer: Option<Arc<KParameterAnalyzer>>,
    pub resonance_coordinator: Option<Arc<ResonanceCoordinator>>,
    pub shadow_coordinator: Option<Arc<tokio::sync::Mutex<q_resonance::ShadowModeCoordinator>>>,

    // Quantum Cryptography
    pub quantum_crypto: Option<Arc<QuantumCryptoEngine>>,
    pub bb84_protocol: Option<Arc<BB84Protocol>>,
    pub qkd_engine: Option<Arc<QKDEngine>>,

    // DeFi Components
    // pub dex_engine: Option<Arc<DEXEngine>>, // Temporarily disabled
    // pub liquidity_pool: Option<Arc<LiquidityPool>>, // Temporarily disabled
    // pub oracle_network: Option<Arc<OracleNetwork>>, // Temporarily disabled
    // pub price_oracle: Option<Arc<PriceOracle>>, // Temporarily disabled
    // pub stablecoin_manager: Option<Arc<StablecoinManager>>, // Temporarily disabled
    // pub collateral_manager: Option<Arc<CollateralManager>>, // Temporarily disabled

    // Quantum Oracle - AI-Enhanced Price Aggregation (927k+ TPS)
    // pub quantum_oracle: Option<Arc<q_oracle::QuantumOracle>>, // Temporarily disabled

    // Quillon Bank - Full Quantum Banking System with CDP
    pub quillon_bank: Arc<RwLock<QuillonBankSystem>>, // ✅ ENABLED - Real banking system

    // AEGIS-QL Post-Quantum Authentication for Founder Operations
    pub aegis_auth_state: Arc<RwLock<aegis_auth_middleware::AegisAuthState>>, // ✅ ENABLED - Founder wallet verification

    // Privacy-as-a-Service (PaaS) Authentication & Rate Limiting
    pub paas_auth_manager: Arc<paas_auth::PaaSAuthManager>, // ✅ ENABLED - Hybrid signature auth
    pub paas_api_key_manager: Arc<paas_api_keys::PaaSApiKeyManager>, // ✅ ENABLED - API key management
    pub paas_pricing_manager: Arc<paas_pricing::PaaSPricingManager>, // ✅ ENABLED - Dynamic USD pricing
    pub paas_billing_manager: Arc<paas_billing::PaaSBillingManager>, // ✅ ENABLED - Atomic billing
    pub paas_idempotency_manager: Arc<paas_idempotency::PaaSIdempotencyManager>, // ✅ ENABLED - Idempotency
    pub paas_audit_manager: Arc<paas_audit::PaaSAuditManager>, // ✅ ENABLED - Audit logging & tracing

    // QUG/QUGUSD Stablecoin System - CollateralVault for over-collateralized minting
    pub collateral_vault: Arc<RwLock<q_vm::contracts::CollateralVault>>,

    // Advanced Infrastructure
    pub tor_circuit_manager: Option<Arc<DedicatedCircuitManager>>,
    pub circuit_pool: Option<Arc<CircuitPool>>,
    // pub robot_fleet: Option<Arc<RobotFleet>>, // Temporarily disabled
    // pub swarm_intelligence: Option<Arc<SwarmIntelligence>>, // Temporarily disabled
    pub p2p_network: Option<Arc<QuantumNetwork>>,
    pub crypto_provider: Option<Arc<CryptoProvider>>,

    // Plugin System
    pub plugin_system: Option<Arc<PluginSystem>>,
    pub plugin_manager: Option<Arc<PluginManager>>,

    // Sharding System
    pub sharding_engine: Option<Arc<ShardingEngine>>,
    pub shard_config: ShardConfig,

    // VM and Smart Contracts - Orobit Integration
    pub contract_registry: Arc<ContractRegistry>,
    pub orobit_ecosystem: Arc<OrobitSmartContractEcosystem>,

    // Distributed VM and DEX (Horizontal Scaling)
    pub distributed_protocol: Option<Arc<q_network::DistributedProtocolManager>>,

    // OAuth2 Provider for third-party integrations
    pub oauth2_storage: Arc<RwLock<oauth2_provider::OAuth2Storage>>,

    // AI Inference Engine - Privacy-first distributed inference with KV-cache (OLD - slow)
    pub inference_engine: Option<Arc<tokio::sync::Mutex<q_ai_inference::distributed_cache::DistributedInferenceWithCache>>>,

    // High-performance mistral.rs engine (10-100x faster, <2s first token)
    pub mistralrs_engine: Option<Arc<q_ai_inference::MistralRsEngine>>,
}

// SAFETY: AppState is safe to Send/Sync because:
// 1. All internal state is wrapped in Arc which is Send+Sync when T is Send+Sync
// 2. The KernelIoEngine contains tokio_uring Runtime which has Rc, but:
//    - It's wrapped in Arc<Mutex<>> which prevents actual cross-thread access to the Rc
//    - We never move the Runtime itself, only access it through the mutex
//    - All io_uring operations happen on the thread where the runtime was created
// 3. All RwLock and Mutex usage ensures proper synchronization
unsafe impl Send for AppState {}
unsafe impl Sync for AppState {}

impl AppState {
    /// Load founder's AEGIS-QL public key from file or environment
    fn load_founder_aegis_public_key() -> anyhow::Result<q_aegis_ql::PublicKey> {
        use std::path::PathBuf;
        use anyhow::Context;

        // Try environment variable first
        if let Ok(key_path_env) = std::env::var("QUILLON_FOUNDER_AEGIS_PUBKEY") {
            let key_path = PathBuf::from(key_path_env);
            return Self::load_aegis_key_from_file(&key_path);
        }

        // Fall back to default location
        let default_path = dirs::home_dir()
            .ok_or_else(|| anyhow::anyhow!("Could not determine home directory"))?
            .join(".quillon")
            .join("keys")
            .join("founder-aegis.pub");

        if default_path.exists() {
            return Self::load_aegis_key_from_file(&default_path);
        }

        // Last resort: use embedded development key (INSECURE - for testing only!)
        tracing::warn!("⚠️  SECURITY WARNING: Using embedded development AEGIS key!");
        tracing::warn!("   For production, set QUILLON_FOUNDER_AEGIS_PUBKEY environment variable");

        // Generate a deterministic key for development (NOT SECURE)
        let mut aegis = q_aegis_ql::AegisQL::new();
        let (public_key, _secret_key) = aegis.generate_keypair()
            .map_err(|e| anyhow::anyhow!("Failed to generate development key: {:?}", e))?;

        Ok(public_key)
    }

    /// Load AEGIS-QL public key from file
    fn load_aegis_key_from_file(path: &std::path::Path) -> anyhow::Result<q_aegis_ql::PublicKey> {
        use anyhow::Context;

        let key_bytes = std::fs::read(path)
            .with_context(|| format!("Failed to read AEGIS public key from {}", path.display()))?;

        let public_key: q_aegis_ql::PublicKey = bincode::deserialize(&key_bytes)
            .context("Failed to deserialize AEGIS public key")?;

        tracing::info!("✅ Loaded founder AEGIS-QL public key from {}", path.display());

        Ok(public_key)
    }

    pub async fn new(config: Config) -> anyhow::Result<Self> {
        // Extract config values before using config (v0.0.22-beta Quick Win #4)
        let is_validator = config.is_validator;
        let block_interval_secs = config.block_interval_secs;
        let max_solutions_per_block = config.max_solutions_per_block;
        let min_solutions_per_block = config.min_solutions_per_block;
        let validator_index = config.validator_index;
        let total_validators = config.total_validators;

        let _wallet_store = MemoryWalletStore::new();
        let wallet_manager = WalletManager::new();
        let node_id = [0u8; 32]; // Default node ID

        let node_status = NodeStatus {
            node_id,
            current_round: 0,
            current_height: 0,
            connected_peers: 0,
            tx_pool_size: 0,
            is_validator,
            uptime: std::time::Duration::from_secs(0),
        };

        // Initialize storage engine with wallet balances persistence
        let storage_config = StorageConfig {
            db_path: config
                .db_path
                .clone()
                .unwrap_or_else(|| "data/q-narwhal-db".to_string()),
            hot_db_path: config
                .hot_db_path
                .clone()
                .unwrap_or_else(|| "data/q-narwhal-hot".to_string()),
            enable_metrics: true,
            sync_writes: true,  // CRITICAL: Enable fsync() to survive hard kills (pkill -9)
            cache_size_mb: 256,
            max_open_files: 1000,
        };

        let storage_engine = Arc::new(StorageEngine::new(storage_config).await?);

        // Load existing wallet balances from storage
        let mut wallet_balances = HashMap::new();
        match storage_engine.load_wallet_balances().await {
            Ok(persisted_balances) => {
                wallet_balances = persisted_balances;
                tracing::info!(
                    "Loaded {} wallet balances from persistent storage",
                    wallet_balances.len()
                );
            }
            Err(e) => {
                tracing::warn!(
                    "Failed to load wallet balances from storage: {}, starting with empty balances",
                    e
                );
            }
        }

        // Load existing token balances from persistent storage
        let mut token_balances = HashMap::new();
        match storage_engine.load_token_balances().await {
            Ok(persisted_token_balances) => {
                token_balances = persisted_token_balances;
                tracing::info!(
                    "🪙 Loaded {} token balances from persistent storage",
                    token_balances.len()
                );
            }
            Err(e) => {
                tracing::warn!(
                    "Failed to load token balances from storage: {}, starting with empty token balances",
                    e
                );
            }
        }

        // Load existing password hashes from persistent storage
        let mut wallet_password_hashes = HashMap::new();
        match storage_engine.load_password_hashes().await {
            Ok(persisted_hashes) => {
                wallet_password_hashes = persisted_hashes;
                tracing::info!(
                    "🔐 Loaded {} password hashes from persistent storage",
                    wallet_password_hashes.len()
                );
            }
            Err(e) => {
                tracing::warn!(
                    "Failed to load password hashes from storage: {}, starting with empty hashes",
                    e
                );
            }
        }

        // Load existing liquidity pools from persistent storage
        let mut liquidity_pools_map = HashMap::new();
        match storage_engine.load_liquidity_pools().await {
            Ok(persisted_pools) => {
                // Deserialize each pool from bytes
                for (pool_id, pool_bytes) in persisted_pools {
                    match serde_json::from_slice::<LiquidityPool>(&pool_bytes) {
                        Ok(pool) => {
                            liquidity_pools_map.insert(pool_id.clone(), pool);
                        }
                        Err(e) => {
                            tracing::warn!(
                                "Failed to deserialize liquidity pool {}: {}, skipping",
                                pool_id,
                                e
                            );
                        }
                    }
                }
                tracing::info!(
                    "💧 Loaded {} liquidity pools from persistent storage",
                    liquidity_pools_map.len()
                );
            }
            Err(e) => {
                tracing::warn!(
                    "Failed to load liquidity pools from storage: {}, starting with empty pools",
                    e
                );
            }
        }

        // CRITICAL FIX: Do NOT load transactions back into mempool on startup
        // Transactions stored in RocksDB are historical/confirmed transactions
        // They should NOT be reprocessed as that causes balance corruption
        // Only new incoming transactions should go into the mempool
        let tx_pool = Arc::new(dashmap::DashMap::new());
        let tx_status = Arc::new(dashmap::DashMap::new());

        // Note: We still load transaction count for metrics, but don't add to mempool
        match storage_engine.load_all_transactions().await {
            Ok(persisted_transactions) => {
                tracing::info!(
                    "💳 Found {} historical transactions in storage (not added to mempool)",
                    persisted_transactions.len()
                );
                // Historical transactions are kept in storage for queries/history
                // but are NOT added to tx_pool to prevent reprocessing
            }
            Err(e) => {
                tracing::warn!(
                    "Failed to load historical transactions from storage: {}",
                    e
                );
            }
        }

        // Initialize real-time streaming
        let event_broadcaster = Arc::new(EventBroadcaster::new());
        let event_emitter = Arc::new(HighPerformanceEmitter::new(event_broadcaster.clone()));

        // Initialize VM and Smart Contract system with Orobit integration
        // IMPORTANT: Create ecosystem FIRST with storage, then pass to registry
        let orobit_ecosystem = Arc::new(OrobitSmartContractEcosystem::new_with_storage(Some(storage_engine.clone())).await?);
        let contract_registry = Arc::new(ContractRegistry::new_with_ecosystem(orobit_ecosystem.clone()));

        // NOTE: Token balances are now loaded from persistent storage above
        // No need to restore from deployed contracts - persistence handles it

        // Initialize Quillon Bank - Full Quantum Banking System with CDP
        let plugin_manager = Arc::new(PluginManager::new());
        let quillon_bank_system = QuillonBankSystem::new(
            node_id,
            q_types::Phase::Phase1,
            plugin_manager.clone(),
        ).await?;
        quillon_bank_system.initialize().await?;
        let quillon_bank = Arc::new(RwLock::new(quillon_bank_system));
        tracing::info!("🏦 Quillon Bank initialized - CDP and quantum banking ready");

        // Initialize AEGIS-QL Authentication for Founder Operations
        let aegis_auth_state = {
            tracing::info!("🔐 Initializing AEGIS-QL post-quantum authentication...");

            // Load founder AEGIS-QL public key from environment or file
            let founder_public_key = Self::load_founder_aegis_public_key()?;

            let auth_state = aegis_auth_middleware::AegisAuthState::new(founder_public_key);

            // Verify founder wallet matches expected constant
            let expected_wallet = hex::decode(aegis_auth_middleware::FOUNDER_WALLET)
                .expect("Invalid founder wallet hex constant");
            let mut expected_bytes = [0u8; 32];
            expected_bytes.copy_from_slice(&expected_wallet);

            if auth_state.founder_wallet == expected_bytes {
                tracing::info!("✅ Founder wallet verified: qnk{}", aegis_auth_middleware::FOUNDER_WALLET);
            } else {
                tracing::warn!("⚠️  Founder wallet mismatch - check AEGIS key configuration");
            }

            tracing::info!("✅ AEGIS-QL authentication initialized (post-quantum secure)");
            Arc::new(RwLock::new(auth_state))
        };

        // Initialize CollateralVault for QUG/QUGUSD stablecoin system
        // Load from persistent storage or create new if none exists
        let collateral_vault = match storage_engine.load_collateral_vault_data().await {
            Ok(Some(vault_bytes)) => {
                match bincode::deserialize::<q_vm::contracts::CollateralVault>(&vault_bytes) {
                    Ok(persisted_vault) => {
                        tracing::info!(
                            "💰 Loaded CollateralVault from storage: locked_qug={}, minted_qugusd={}",
                            persisted_vault.total_qug_locked,
                            persisted_vault.total_qugusd_minted
                        );
                        Arc::new(RwLock::new(persisted_vault))
                    }
                    Err(e) => {
                        tracing::warn!("Failed to deserialize CollateralVault: {}, creating new vault", e);
                        Arc::new(RwLock::new(q_vm::contracts::CollateralVault::new()))
                    }
                }
            }
            Ok(None) => {
                tracing::info!("💰 No persisted CollateralVault found, creating new vault");
                Arc::new(RwLock::new(q_vm::contracts::CollateralVault::new()))
            }
            Err(e) => {
                tracing::warn!(
                    "Failed to load CollateralVault from storage: {}, creating new vault",
                    e
                );
                Arc::new(RwLock::new(q_vm::contracts::CollateralVault::new()))
            }
        };
        tracing::info!("💰 CollateralVault initialized - QUG/QUGUSD stablecoin system ready");

        Ok(Self {
            config,
            node_id,
            wallet_manager,
            node_status: Arc::new(RwLock::new(node_status)),
            // PERFORMANCE: DashMap for lock-free transaction pool (20-40K TPS target)
            tx_pool,
            tx_status,
            blocks: Arc::new(RwLock::new(HashMap::new())),
            wallet_balances: Arc::new(RwLock::new(wallet_balances.clone())),
            wallet_password_hashes: Arc::new(RwLock::new(wallet_password_hashes)),
            token_balances: Arc::new(RwLock::new(token_balances)),
            liquidity_pools: Arc::new(RwLock::new(liquidity_pools_map)),
            nitro_boosts: Arc::new(RwLock::new(HashMap::new())),
            storage_engine: storage_engine.clone(),
            event_broadcaster,
            event_emitter,

            // Faucet system with rate limiting and abuse protection
            faucet_state: Arc::new(RwLock::new(FaucetState::default())),

            // 🔒 MAX SUPPLY ENFORCEMENT - Initialize supply tracking
            total_minted_supply: Arc::new(RwLock::new(0)), // Start at 0, will load from storage
            supply_consensus_state: Arc::new(RwLock::new(SupplyConsensusState::default())),
            mining_statistics: Some(Arc::new(RwLock::new(MiningStatistics::default()))),

            // Quantum Privacy Mixer State
            mixing_requests: Arc::new(RwLock::new(HashMap::new())),
            quantum_mixer: {
                // Create entropy pool for mixer initialization
                match q_quantum_mixing::quantum_entropy::QuantumEntropyPool::new().await {
                    Ok(entropy_pool) => {
                        let entropy_arc = Arc::new(entropy_pool);
                        match QuantumMixingEngine::new(entropy_arc).await {
                            Ok(mixer) => {
                                tracing::info!("✅ Quantum Mixing Engine initialized - Privacy mixer ready");
                                Some(Arc::new(mixer))
                            }
                            Err(e) => {
                                tracing::warn!("⚠️ Quantum Mixing Engine initialization failed: {}, mixer disabled", e);
                                None
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!("⚠️ Quantum Entropy Pool initialization failed: {}, mixer disabled", e);
                        None
                    }
                }
            },
            zkp_prover: {
                match q_quantum_mixing::quantum_entropy::QuantumEntropyPool::new().await {
                    Ok(entropy_pool) => {
                        let zkp_config = q_quantum_mixing::zkp_prover::ZKProofConfig::default();
                        match QuantumZKPProver::new(Arc::new(entropy_pool), zkp_config).await {
                            Ok(prover) => {
                                tracing::info!("✅ Quantum ZK Proof Engine initialized - ZK proofs ready");
                                Some(Arc::new(prover))
                            }
                            Err(e) => {
                                tracing::warn!("⚠️ ZK Proof Engine initialization failed: {}, proofs disabled", e);
                                None
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!("⚠️ Quantum Entropy Pool initialization failed: {}, ZK proofs disabled", e);
                        None
                    }
                }
            },

            // Network components
            bitcoin_bridge: None,
            dns_phantom: None,
            bep44_discovery: None,
            tor_client: None,
            network_manager: None,
            production_peer_discovery: None,
            libp2p_discovery: None,  // Disabled in test mode
            libp2p_command_tx: None,  // Disabled in test mode
            libp2p_peer_info: Arc::new(RwLock::new((String::new(), vec![]))), // Empty initially
            libp2p_peer_count: None, // Disabled in test mode
            mining_submission_tx: None,  // Disabled in test mode
            connection_manager: None,
            dag_sync_manager: None,  // Will be initialized with PeerRegistry

            // ZK Privacy Components - Initialize with None
            zk_stark_system: None,
            zk_snark_system: None,

            // Performance & Scaling Optimizations - Initialize for maximum TPS
            simd_crypto_engine: {
                let simd_config = q_crypto_simd::SimdCryptoConfig::default();
                match q_crypto_simd::SimdCryptoEngine::new(simd_config).await {
                    Ok(engine) => {
                        tracing::info!("✅ SIMD Crypto Engine initialized - Vectorized cryptography enabled");
                        Some(Arc::new(engine))
                    }
                    Err(e) => {
                        tracing::warn!("⚠️ SIMD Crypto Engine initialization failed: {}, using fallback", e);
                        None
                    }
                }
            },
            #[cfg(target_os = "linux")]
            kernel_io_engine: {
                match crate::io_uring_adapter::IoUringAdapter::new() {
                    Ok(adapter) => {
                        tracing::info!("✅ Kernel I/O Engine initialized with dedicated thread pool");
                        Some(Arc::new(adapter))
                    }
                    Err(e) => {
                        tracing::warn!("⚠️ Kernel I/O Engine failed to initialize: {}, using standard I/O", e);
                        None
                    }
                }
            },
            // shard_coordinator: None, // Temporarily disabled
            // shard_manager: None, // Temporarily disabled
            // cache_manager: None, // Temporarily disabled
            // hierarchical_cache: None, // Temporarily disabled

            // Consensus & DAG - Initialize with None, available on demand
            dag_knight: None,
            anchor_election: None,
            narwhal_core: None,
            production_mempool: None,  // Will be initialized in main.rs for high TPS
            reliable_broadcast: None,
            quantum_vdf: None,

            // PHASE 2: Parallel Block Producer Pool - 8 concurrent producers for exciting visualization
            block_producer_pool: {
                // Create base config for all producers
                let base_config = crate::block_producer::BlockProducerConfig {
                    block_interval_secs, // v0.0.22-beta: from extracted config (2s for fast visualization)
                    max_solutions_per_block, // v0.0.22-beta: from extracted config
                    min_solutions_per_block, // v0.0.22-beta: from extracted config
                    node_id,
                    is_validator,
                    validator_index: 0, // Will be overridden by pool for each producer
                    total_validators: 8, // Phase 2: 8 parallel producers
                };

                // Create pool with 8 parallel producers for true parallelism
                let num_producers = 8; // Phase 2: 8-way parallelism for exciting multi-lane visualization

                // CRITICAL FIX: Load blockchain state from storage to prevent data loss on restart
                let pool = crate::block_producer::ParallelBlockProducerPool::new_with_storage(
                    num_producers,
                    base_config,
                    &storage_engine, // Pass storage Arc to load blockchain state
                ).await?;

                info!("🚀 Phase 2: Parallel Block Production initialized with {} producers (LOADED FROM STORAGE)", num_producers);
                info!("⚡ Exciting visualization: Multiple blocks will appear simultaneously in different lanes!");

                Arc::new(pool)
            },

            // PHASE 3: DAG-Knight Consensus - Initialize with Byzantine fault tolerance
            consensus: {
                let consensus = DAGKnightConsensus::new(
                    node_id,
                    1, // f = 1 (supports 2f+1 = 3 nodes minimum for BFT)
                ).await?;

                info!("🎯 Initialized DAG-Knight consensus engine (f={}, min_nodes={})",
                    1, 3);

                Arc::new(RwLock::new(consensus))
            },

            // Quillon Resonance - Will be initialized in main.rs
            k_parameter_analyzer: None,
            resonance_coordinator: None,
            shadow_coordinator: None,

            // Quantum Cryptography - Initialize with None, available on demand
            quantum_crypto: None,
            bb84_protocol: None,
            qkd_engine: None,

            // DeFi Components - Initialize with None, available on demand
            // dex_engine: None, // Temporarily disabled
            // liquidity_pool: None, // Temporarily disabled
            // oracle_network: None, // Temporarily disabled
            // price_oracle: None, // Temporarily disabled
            // stablecoin_manager: None, // Temporarily disabled
            // collateral_manager: None, // Temporarily disabled

            // Quantum Oracle - AI-Enhanced Price Aggregation
            // quantum_oracle: None,  // Temporarily disabled

            // Advanced Infrastructure - Initialize with None, available on demand
            tor_circuit_manager: None,
            circuit_pool: None,
            // robot_fleet: None, // Temporarily disabled
            // swarm_intelligence: None, // Temporarily disabled
            p2p_network: None,
            crypto_provider: None,

            // Plugin System - Initialize with None, available on demand
            plugin_system: None,
            plugin_manager: None,

            // Sharding System - Initialize with default config
            sharding_engine: None,
            shard_config: ShardConfig::default(),

            // VM and Smart Contracts - Orobit Integration
            contract_registry,
            orobit_ecosystem,

            // Quillon Bank - Full Quantum Banking System with CDP
            quillon_bank,

            // AEGIS-QL Post-Quantum Authentication for Founder Operations
            aegis_auth_state,

            // QUG/QUGUSD Stablecoin System - CollateralVault
            collateral_vault,

            // Distributed VM and DEX - Initialize in production mode
            distributed_protocol: None, // Initialized separately

            // OAuth2 Provider - Initialize with empty storage
            oauth2_storage: Arc::new(RwLock::new(oauth2_provider::OAuth2Storage::new())),

            // Privacy-as-a-Service (PaaS) Components - Initialize with proper constructors
            paas_auth_manager: Arc::new(paas_auth::PaaSAuthManager::new()),
            paas_api_key_manager: Arc::new(paas_api_keys::PaaSApiKeyManager::new()),
            paas_pricing_manager: Arc::new(paas_pricing::PaaSPricingManager::new()),
            paas_billing_manager: Arc::new(paas_billing::PaaSBillingManager::new()),
            paas_idempotency_manager: Arc::new(paas_idempotency::PaaSIdempotencyManager::new()),
            paas_audit_manager: Arc::new(paas_audit::PaaSAuditManager::new(10_000)), // 10k records in memory

            // AI Inference Engine - Initialized in main.rs with auto-download
            inference_engine: None,
            mistralrs_engine: None,
        })
    }

    pub async fn new_with_networks(
        config: Config,
        node_id: NodeId,
        bitcoin_bridge: Option<Arc<()>>, // DEACTIVATED
        dns_phantom: Option<Arc<()>>, // DEACTIVATED
        bep44_discovery: Option<Arc<()>>, // DEACTIVATED
        tor_client: Option<Arc<QTorClient>>,
        __production_peer_discovery: Option<Arc<tokio::sync::Mutex<q_network::real_peer_discovery::RealPeerDiscovery>>>,
        libp2p_discovery: Option<Arc<tokio::sync::Mutex<q_network::UnifiedNetworkManager>>>,
        libp2p_command_tx: Option<tokio::sync::mpsc::UnboundedSender<q_network::NetworkCommand>>,
    ) -> anyhow::Result<Self> {
        // Extract config values before using config (v0.0.22-beta Quick Win #4)
        let is_validator = config.is_validator;
        let block_interval_secs = config.block_interval_secs;
        let max_solutions_per_block = config.max_solutions_per_block;
        let min_solutions_per_block = config.min_solutions_per_block;
        let validator_index = config.validator_index;
        let total_validators = config.total_validators;

        let _wallet_store = MemoryWalletStore::new();
        let wallet_manager = WalletManager::new();

        let node_status = NodeStatus {
            node_id,
            current_round: 0,
            current_height: 0,
            connected_peers: 0,
            tx_pool_size: 0,
            is_validator,
            uptime: std::time::Duration::from_secs(0),
        };

        // Initialize storage engine with wallet balances persistence
        let storage_config = StorageConfig {
            db_path: config
                .db_path
                .clone()
                .unwrap_or_else(|| "data/q-narwhal-db".to_string()),
            hot_db_path: config
                .hot_db_path
                .clone()
                .unwrap_or_else(|| "data/q-narwhal-hot".to_string()),
            enable_metrics: true,
            sync_writes: true,  // CRITICAL: Enable fsync() to survive hard kills (pkill -9)
            cache_size_mb: 256,
            max_open_files: 1000,
        };

        let storage_engine = Arc::new(StorageEngine::new(storage_config).await?);

        // Initialize NetworkManager to bridge DNS-phantom to libp2p
        let network_manager = {
            let mut tor_config = q_tor_client::TorConfig::default();
            tor_config.enabled = true;  // Enable Tor for NetworkManager

            let network_config = q_network::NetworkManagerConfig {
                local_validator_id: node_id,
                tor_config,
                phase: q_types::Phase::Phase1,
                channel_rotation_hours: 24,
                sync_enabled: true,
                heartbeat_interval_secs: 30,
                max_peers: 100,
            };

            match NetworkManager::new(network_config).await {
                Ok(nm) => {
                    tracing::info!("✅ NetworkManager initialized - DNS-phantom bridge ready");
                    Some(Arc::new(nm))
                }
                Err(e) => {
                    tracing::warn!("⚠️ NetworkManager initialization failed: {}, continuing without peer bridge", e);
                    None
                }
            }
        };

        // Use the libp2p discovery passed in from main.rs (which has topic subscriptions configured)
        // Or fall back to creating a basic one if None was passed
        let libp2p_discovery = if libp2p_discovery.is_some() {
            // Use the pre-configured manager from main.rs
            tracing::info!("✅ Using pre-configured libp2p manager from main.rs");
            libp2p_discovery
        } else {
            // Fallback: create a basic discovery manager with testnet config
            let fallback_config = q_types::NetworkConfig::testnet();
            match q_network::UnifiedNetworkManager::new(fallback_config).await {
                Ok(discovery) => {
                    tracing::info!("🚀 libp2p Zero-Knowledge Discovery initialized successfully (fallback testnet)!");
                    tracing::info!("📡 Active discovery mechanisms: mDNS (local network), Identify (peer exchange), Ping (keepalive)");
                    Some(Arc::new(tokio::sync::Mutex::new(discovery)))
                }
                Err(e) => {
                    tracing::warn!("⚠️ libp2p discovery initialization failed: {}, continuing without mDNS discovery", e);
                    None
                }
            }
        };

        // Load existing wallet balances from storage
        let mut wallet_balances = HashMap::new();
        match storage_engine.load_wallet_balances().await {
            Ok(persisted_balances) => {
                wallet_balances = persisted_balances;
                tracing::info!(
                    "Loaded {} wallet balances from persistent storage",
                    wallet_balances.len()
                );
            }
            Err(e) => {
                tracing::warn!(
                    "Failed to load wallet balances from storage: {}, starting with empty balances",
                    e
                );
            }
        }

        // Load existing token balances from persistent storage
        let mut token_balances = HashMap::new();
        match storage_engine.load_token_balances().await {
            Ok(persisted_token_balances) => {
                token_balances = persisted_token_balances;
                tracing::info!(
                    "🪙 Loaded {} token balances from persistent storage",
                    token_balances.len()
                );
            }
            Err(e) => {
                tracing::warn!(
                    "Failed to load token balances from storage: {}, starting with empty token balances",
                    e
                );
            }
        }

        // Load existing password hashes from persistent storage
        let mut wallet_password_hashes = HashMap::new();
        match storage_engine.load_password_hashes().await {
            Ok(persisted_hashes) => {
                wallet_password_hashes = persisted_hashes;
                tracing::info!(
                    "🔐 Loaded {} password hashes from persistent storage",
                    wallet_password_hashes.len()
                );
            }
            Err(e) => {
                tracing::warn!(
                    "Failed to load password hashes from storage: {}, starting with empty hashes",
                    e
                );
            }
        }

        // Load existing liquidity pools from persistent storage
        let mut liquidity_pools_map = HashMap::new();
        match storage_engine.load_liquidity_pools().await {
            Ok(persisted_pools) => {
                // Deserialize each pool from bytes
                for (pool_id, pool_bytes) in persisted_pools {
                    match serde_json::from_slice::<LiquidityPool>(&pool_bytes) {
                        Ok(pool) => {
                            liquidity_pools_map.insert(pool_id.clone(), pool);
                        }
                        Err(e) => {
                            tracing::warn!(
                                "Failed to deserialize liquidity pool {}: {}, skipping",
                                pool_id,
                                e
                            );
                        }
                    }
                }
                tracing::info!(
                    "💧 Loaded {} liquidity pools from persistent storage",
                    liquidity_pools_map.len()
                );
            }
            Err(e) => {
                tracing::warn!(
                    "Failed to load liquidity pools from storage: {}, starting with empty pools",
                    e
                );
            }
        }

        // CRITICAL FIX: Do NOT load transactions back into mempool on startup
        // Transactions stored in RocksDB are historical/confirmed transactions
        // They should NOT be reprocessed as that causes balance corruption
        // Only new incoming transactions should go into the mempool
        let tx_pool = Arc::new(dashmap::DashMap::new());
        let tx_status = Arc::new(dashmap::DashMap::new());

        // Note: We still load transaction count for metrics, but don't add to mempool
        match storage_engine.load_all_transactions().await {
            Ok(persisted_transactions) => {
                tracing::info!(
                    "💳 Found {} historical transactions in storage (not added to mempool)",
                    persisted_transactions.len()
                );
                // Historical transactions are kept in storage for queries/history
                // but are NOT added to tx_pool to prevent reprocessing
            }
            Err(e) => {
                tracing::warn!(
                    "Failed to load historical transactions from storage: {}",
                    e
                );
            }
        }

        // Initialize real-time streaming
        let event_broadcaster = Arc::new(EventBroadcaster::new());
        let event_emitter = Arc::new(HighPerformanceEmitter::new(event_broadcaster.clone()));

        // Initialize VM and Smart Contract system with Orobit integration
        // IMPORTANT: Create ecosystem FIRST with storage, then pass to registry
        let orobit_ecosystem = Arc::new(OrobitSmartContractEcosystem::new_with_storage(Some(storage_engine.clone())).await?);
        let contract_registry = Arc::new(ContractRegistry::new_with_ecosystem(orobit_ecosystem.clone()));

        // NOTE: Token balances are now loaded from persistent storage above
        // No need to restore from deployed contracts - persistence handles it

        // Initialize Quillon Bank - Full Quantum Banking System with CDP
        let plugin_manager = Arc::new(PluginManager::new());
        let quillon_bank_system = QuillonBankSystem::new(
            node_id,
            q_types::Phase::Phase1,
            plugin_manager.clone(),
        ).await?;
        quillon_bank_system.initialize().await?;
        let quillon_bank = Arc::new(RwLock::new(quillon_bank_system));
        tracing::info!("🏦 Quillon Bank initialized - CDP and quantum banking ready");

        // Initialize AEGIS-QL Authentication for Founder Operations
        let aegis_auth_state = {
            tracing::info!("🔐 Initializing AEGIS-QL post-quantum authentication...");

            // Load founder AEGIS-QL public key from environment or file
            let founder_public_key = Self::load_founder_aegis_public_key()?;

            let auth_state = aegis_auth_middleware::AegisAuthState::new(founder_public_key);

            // Verify founder wallet matches expected constant
            let expected_wallet = hex::decode(aegis_auth_middleware::FOUNDER_WALLET)
                .expect("Invalid founder wallet hex constant");
            let mut expected_bytes = [0u8; 32];
            expected_bytes.copy_from_slice(&expected_wallet);

            if auth_state.founder_wallet == expected_bytes {
                tracing::info!("✅ Founder wallet verified: qnk{}", aegis_auth_middleware::FOUNDER_WALLET);
            } else {
                tracing::warn!("⚠️  Founder wallet mismatch - check AEGIS key configuration");
            }

            tracing::info!("✅ AEGIS-QL authentication initialized (post-quantum secure)");
            Arc::new(RwLock::new(auth_state))
        };

        // Initialize CollateralVault for QUG/QUGUSD stablecoin system
        // Load from persistent storage or create new if none exists
        let collateral_vault = match storage_engine.load_collateral_vault_data().await {
            Ok(Some(vault_bytes)) => {
                match bincode::deserialize::<q_vm::contracts::CollateralVault>(&vault_bytes) {
                    Ok(persisted_vault) => {
                        tracing::info!(
                            "💰 Loaded CollateralVault from storage: locked_qug={}, minted_qugusd={}",
                            persisted_vault.total_qug_locked,
                            persisted_vault.total_qugusd_minted
                        );
                        Arc::new(RwLock::new(persisted_vault))
                    }
                    Err(e) => {
                        tracing::warn!("Failed to deserialize CollateralVault: {}, creating new vault", e);
                        Arc::new(RwLock::new(q_vm::contracts::CollateralVault::new()))
                    }
                }
            }
            Ok(None) => {
                tracing::info!("💰 No persisted CollateralVault found, creating new vault");
                Arc::new(RwLock::new(q_vm::contracts::CollateralVault::new()))
            }
            Err(e) => {
                tracing::warn!(
                    "Failed to load CollateralVault from storage: {}, creating new vault",
                    e
                );
                Arc::new(RwLock::new(q_vm::contracts::CollateralVault::new()))
            }
        };
        tracing::info!("💰 CollateralVault initialized - QUG/QUGUSD stablecoin system ready");

        Ok(Self {
            config,
            node_id,
            wallet_manager,
            node_status: Arc::new(RwLock::new(node_status)),
            // PERFORMANCE: DashMap for lock-free transaction pool (20-40K TPS target)
            tx_pool,
            tx_status,
            blocks: Arc::new(RwLock::new(HashMap::new())),
            wallet_balances: Arc::new(RwLock::new(wallet_balances.clone())),
            wallet_password_hashes: Arc::new(RwLock::new(wallet_password_hashes)),
            token_balances: Arc::new(RwLock::new(token_balances)),
            liquidity_pools: Arc::new(RwLock::new(liquidity_pools_map)),
            nitro_boosts: Arc::new(RwLock::new(HashMap::new())),
            storage_engine: storage_engine.clone(),
            event_broadcaster,
            event_emitter,

            // Faucet system
            faucet_state: Arc::new(RwLock::new(FaucetState::default())),

            // 🔒 MAX SUPPLY ENFORCEMENT - Load from storage or start at 0
            total_minted_supply: Arc::new(RwLock::new(0)), // TODO: Load from storage
            supply_consensus_state: Arc::new(RwLock::new(SupplyConsensusState::default())),
            mining_statistics: Some(Arc::new(RwLock::new(MiningStatistics::default()))),

            // Quantum Privacy Mixer State
            mixing_requests: Arc::new(RwLock::new(HashMap::new())),
            quantum_mixer: None, // Initialized later if needed
            zkp_prover: None, // Initialized later if needed

            // Network components with provided values
            bitcoin_bridge,
            dns_phantom,
            bep44_discovery,
            tor_client,
            network_manager,
            production_peer_discovery: None,

            // libp2p-based zero-config peer discovery
            libp2p_discovery,

            // libp2p network command channel
            libp2p_command_tx,

            // Cached libp2p peer info (will be populated after network starts)
            libp2p_peer_info: Arc::new(RwLock::new((String::new(), vec![]))),

            // Atomic peer count (will be populated from network manager)
            libp2p_peer_count: None,  // Will be initialized in main.rs after network manager creation

            // Mining submission async queue
            mining_submission_tx: None,  // Will be initialized in main.rs

            // BREAKTHROUGH: DNS-Phantom → Connection Integration
            connection_manager: {
                let cm = Arc::new(q_network::connection_manager::ConnectionManager::new());
                // Start the connection manager background tasks
                cm.start().await;
                tracing::info!(
                    "✅ Connection manager initialized for DNS-Phantom → Connection bridging"
                );
                Some(cm)
            },

            // DAG State Synchronization - will be initialized with PeerRegistry
            dag_sync_manager: None,

            // ZK Privacy Components - ✅ Initialize ZK systems
            zk_stark_system: {
                match StarkSystem::new(false).await {
                    Ok(system) => {
                        tracing::info!("✅ ZK-STARK System initialized - Zero-knowledge proofs enabled");
                        Some(Arc::new(tokio::sync::Mutex::new(system)))
                    }
                    Err(e) => {
                        tracing::warn!("⚠️ ZK-STARK System initialization failed: {}, ZK proofs unavailable", e);
                        None
                    }
                }
            },
            zk_snark_system: {
                let snark_config = q_zk_snark::SNARKConfig::default();
                let system = UniversalSNARK::new(snark_config);
                tracing::info!("✅ ZK-SNARK System initialized - Groth16/PLONK proofs enabled");
                Some(Arc::new(system))
            },

            // Performance & Scaling Optimizations - Initialize for maximum TPS
            simd_crypto_engine: {
                let simd_config = q_crypto_simd::SimdCryptoConfig::default();
                match q_crypto_simd::SimdCryptoEngine::new(simd_config).await {
                    Ok(engine) => {
                        tracing::info!("✅ SIMD Crypto Engine initialized - Vectorized cryptography enabled");
                        Some(Arc::new(engine))
                    }
                    Err(e) => {
                        tracing::warn!("⚠️ SIMD Crypto Engine initialization failed: {}, using fallback", e);
                        None
                    }
                }
            },
            #[cfg(target_os = "linux")]
            kernel_io_engine: {
                match crate::io_uring_adapter::IoUringAdapter::new() {
                    Ok(adapter) => {
                        tracing::info!("✅ Kernel I/O Engine initialized with dedicated thread pool");
                        Some(Arc::new(adapter))
                    }
                    Err(e) => {
                        tracing::warn!("⚠️ Kernel I/O Engine failed to initialize: {}, using standard I/O", e);
                        None
                    }
                }
            },
            // shard_coordinator: None, // Temporarily disabled
            // shard_manager: None, // Temporarily disabled
            // cache_manager: None, // Temporarily disabled
            // hierarchical_cache: None, // Temporarily disabled

            // Consensus & DAG - Initialize with None, available on demand
            dag_knight: None,
            anchor_election: None,
            narwhal_core: None,
            production_mempool: None,  // Will be initialized in main.rs for high TPS
            reliable_broadcast: None,
            quantum_vdf: None,

            // PHASE 2: Parallel Block Producer Pool - 8 concurrent producers for exciting visualization
            block_producer_pool: {
                // Create base config for all producers
                let base_config = crate::block_producer::BlockProducerConfig {
                    block_interval_secs, // v0.0.22-beta: from extracted config (2s for fast visualization)
                    max_solutions_per_block, // v0.0.22-beta: from extracted config
                    min_solutions_per_block, // v0.0.22-beta: from extracted config
                    node_id,
                    is_validator,
                    validator_index: 0, // Will be overridden by pool for each producer
                    total_validators: 8, // Phase 2: 8 parallel producers
                };

                // Create pool with 8 parallel producers for true parallelism
                let num_producers = 8; // Phase 2: 8-way parallelism for exciting multi-lane visualization

                // CRITICAL FIX: Load blockchain state from storage to prevent data loss on restart
                let pool = crate::block_producer::ParallelBlockProducerPool::new_with_storage(
                    num_producers,
                    base_config,
                    &storage_engine, // Pass storage Arc to load blockchain state
                ).await?;

                info!("🚀 Phase 2: Parallel Block Production initialized with {} producers (LOADED FROM STORAGE)", num_producers);
                info!("⚡ Exciting visualization: Multiple blocks will appear simultaneously in different lanes!");

                Arc::new(pool)
            },

            // PHASE 3: DAG-Knight Consensus - Initialize with Byzantine fault tolerance
            consensus: {
                let consensus = DAGKnightConsensus::new(
                    node_id,
                    1, // f = 1 (supports 2f+1 = 3 nodes minimum for BFT)
                ).await?;

                info!("🎯 Initialized DAG-Knight consensus engine (f={}, min_nodes={})",
                    1, 3);

                Arc::new(RwLock::new(consensus))
            },

            // Quillon Resonance - Will be initialized in main.rs
            k_parameter_analyzer: None,
            resonance_coordinator: None,
            shadow_coordinator: None,

            // Quantum Cryptography - Initialize with None, available on demand
            quantum_crypto: None,
            bb84_protocol: None,
            qkd_engine: None,

            // DeFi Components - Initialize with None, available on demand
            // dex_engine: None, // Temporarily disabled
            // liquidity_pool: None, // Temporarily disabled
            // oracle_network: None, // Temporarily disabled
            // price_oracle: None, // Temporarily disabled
            // stablecoin_manager: None, // Temporarily disabled
            // collateral_manager: None, // Temporarily disabled

            // Quantum Oracle - AI-Enhanced Price Aggregation
            // quantum_oracle: None,  // Temporarily disabled

            // Advanced Infrastructure - Initialize with None, available on demand
            tor_circuit_manager: None,
            circuit_pool: None,
            // robot_fleet: None, // Temporarily disabled
            // swarm_intelligence: None, // Temporarily disabled
            p2p_network: None,
            crypto_provider: None,

            // Plugin System - Initialize with None, available on demand
            plugin_system: None,
            plugin_manager: None,

            // Sharding System - Initialize with default config
            sharding_engine: None,
            shard_config: ShardConfig::default(),

            // VM and Smart Contracts - Orobit Integration
            contract_registry,
            orobit_ecosystem,

            // Quillon Bank - Full Quantum Banking System with CDP
            quillon_bank,

            // AEGIS-QL Post-Quantum Authentication for Founder Operations
            aegis_auth_state,

            // QUG/QUGUSD Stablecoin System - CollateralVault
            collateral_vault,

            // Distributed VM and DEX - Initialize in production mode
            distributed_protocol: None, // Initialized separately

            // OAuth2 Provider - Initialize with empty storage
            oauth2_storage: Arc::new(RwLock::new(oauth2_provider::OAuth2Storage::new())),

            // Privacy-as-a-Service (PaaS) Components - Initialize with proper constructors
            paas_auth_manager: Arc::new(paas_auth::PaaSAuthManager::new()),
            paas_api_key_manager: Arc::new(paas_api_keys::PaaSApiKeyManager::new()),
            paas_pricing_manager: Arc::new(paas_pricing::PaaSPricingManager::new()),
            paas_billing_manager: Arc::new(paas_billing::PaaSBillingManager::new()),
            paas_idempotency_manager: Arc::new(paas_idempotency::PaaSIdempotencyManager::new()),
            paas_audit_manager: Arc::new(paas_audit::PaaSAuditManager::new(10_000)), // 10k records in memory

            // AI Inference Engine - Initialized in main.rs with auto-download
            inference_engine: None,
            mistralrs_engine: None,
        })
    }

    /// Helper method to save wallet balance to persistent storage
    pub async fn save_wallet_balance(&self, address: &[u8; 32], amount: u64) -> anyhow::Result<()> {
        self.storage_engine
            .save_wallet_balance(address, amount)
            .await?;
        Ok(())
    }

    /// Helper method to save all wallet balances to persistent storage
    pub async fn save_all_wallet_balances(&self) -> anyhow::Result<()> {
        let balances = self.wallet_balances.read().await.clone();
        self.storage_engine.save_wallet_balances(&balances).await?;
        Ok(())
    }

    /// Initialize distributed VM and DEX protocol (for horizontal scaling)
    pub async fn init_distributed_protocol(&mut self, local_peer_id: libp2p::PeerId) -> anyhow::Result<()> {
        tracing::info!("🌐 Initializing distributed VM & DEX coordinators for horizontal scaling");

        let protocol = q_network::DistributedProtocolManager::new(local_peer_id).await?;

        self.distributed_protocol = Some(Arc::new(protocol));

        tracing::info!("✅ Distributed protocol coordinators initialized - ready for multi-node collaboration");
        tracing::info!("📝 Note: Network transport will be managed by UnifiedNetworkManager");

        Ok(())
    }

    /// Get distributed protocol stats
    pub async fn get_distributed_stats(&self) -> Option<q_network::DistributedNetworkStats> {
        if let Some(ref protocol) = self.distributed_protocol {
            Some(protocol.get_stats().await)
        } else {
            None
        }
    }
}
pub mod p2p_server;
