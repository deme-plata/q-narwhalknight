/// Q-NarwhalKnight API Server - Version v2.4.0-beta
/// v2.4.0-beta: Tensor Parallelism (Golden Standard) - Ring All-Reduce, Weight Sharding
/// v2.3.20-beta: Testnet signature bypass for distributed AI inference
/// v2.3.19-beta: Enhanced distributed AI debugging for troubleshooting P2P inference
/// v2.3.18-beta: Safe AI Intent Architecture - AI parses, Rust executes
/// v2.3.7-beta: TurboSync get_local_height() fix - use contiguous height, not latest stored
/// v2.3.6-beta: Sync cooldown fix - use contiguous height for gap detection post-sync
/// v2.3.5-beta: Sync activation fix + hashrate flickering fix
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Minimum miner version required to mine on this server.
/// Bump this when a miner update is mandatory (e.g. protocol change).
/// Miner versions are independent from server versions (q-miner v2.x vs q-api-server v8.x).
pub const MIN_MINER_VERSION: &str = "2.6.0";

/// v8.8.2: Migration flag for one-time bootstrap wallet balance sync.
/// Once set in RocksDB CF_MANIFEST, wallet balance import from peers is permanently disabled.
/// Also gates QUGUSD: before flag is set, QUGUSD flows through from peers (bootstrap).
/// After flag is set, QUGUSD is blocked (ghost prevention resumes).
pub const BOOTSTRAP_WALLET_SYNC_FLAG: &[u8] = b"migration_bootstrap_wallet_sync_v882_done";

// DEACTIVATED: use q_bep44_discovery::DiscoveryEngine;
// DEACTIVATED: use q_bitcoin_bridge::bridge::IntegratedBitcoinBridge;
// DEACTIVATED: use q_dns_phantom::DNSPhantomNetwork;
use q_network::NetworkManager;
use q_storage::{StorageConfig, StorageEngine};
use q_tor_client::QTorClient; // Re-enabled for consensus integration
// 🌻 v2.5.0-beta: Dandelion++ for mandatory Tor-based transaction anonymity
use q_dandelion::{QuantumDandelion, DandelionConfig, NetworkBridge, NetworkBridgeConfig};
use q_types::*;
use q_types::upgrades::UpgradeManager;
use q_wallet::{MemoryWalletStore, WalletManager};

// ZK Privacy Components - ✅ ENABLED
use q_zk_snark::UniversalSNARK;
use q_zk_stark::StarkSystem;

// Performance & Scaling Components
// use q_sharding::{ShardCoordinator, ShardManager}; // Temporarily disabled
// use q_cache::{HierarchicalCache, CacheManager}; // Temporarily disabled

// Consensus & DAG Components
use q_dag_knight::{DAGKnightConsensus, QuantumAnchorElection};
use q_narwhal_core::production_mempool::ProductionMempool;
use q_narwhal_core::{NarwhalCore, ReliableBroadcast};
#[cfg(feature = "resonance")]
use q_resonance::{KParameterAnalyzer, KParameterMetrics, PhaseTransition, ResonanceCoordinator};
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
use libp2p::PeerId;
use q_network::{CryptoProvider, QuantumNetwork};

// Plugin System
use q_plugin_system::{PluginManager, PluginSystem, PluginSystemConfig};

// ✨ v1.0.58-beta: FROST threshold signing for validator committees (IACR 2025/1024)
#[cfg(feature = "advanced-crypto")]
pub mod frost_committee;

// ✨ v1.0.51-beta: Crypto-enhanced instant mining rewards (AEGIS-256 authenticated)
pub mod instant_mining_rewards;

// ✨ v1.4.0-beta: Recursive SNARKs for eliminating weak subjectivity
// Post-quantum recursive proofs enable ~10ms trustless bootstrap for new nodes
pub mod recursive_proofs_api;

// ✨ v3.4.16-beta: Automatic ZK privacy proofs - ALL transactions get maximum privacy by default
// Users don't choose privacy levels - best privacy is always applied automatically
pub mod privacy_proof_generator;

// Sharding System
use q_sharding::{ShardConfig, ShardMetrics, ShardingEngine, ShardingStrategy};

// VM and Smart Contracts
use q_vm::contracts::{ContractRegistry, OrobitSmartContractEcosystem};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{error, info, trace};
use uuid::Uuid;

// ✅ v1.0.17-beta DEADLOCK FIX: Lock timeout helpers to prevent deadlocks
use anyhow::{Context, Result};
use tokio::time::{timeout, Duration};

/// Acquire a Mutex with timeout to prevent deadlocks
///
/// # Arguments
/// * `mutex` - The mutex to acquire
/// * `timeout_secs` - Timeout in seconds (recommended: 5-30s)
/// * `lock_name` - Human-readable lock name for diagnostics
///
/// # Returns
/// * `Ok(guard)` - Successfully acquired lock
/// * `Err(anyhow::Error)` - Timeout occurred, likely deadlock
///
/// # Panics
/// Does NOT panic - returns error for caller to handle
pub async fn lock_with_timeout<'a, T>(
    mutex: &'a tokio::sync::Mutex<T>,
    timeout_secs: u64,
    lock_name: &str,
) -> Result<tokio::sync::MutexGuard<'a, T>> {
    match timeout(Duration::from_secs(timeout_secs), mutex.lock()).await {
        Ok(guard) => {
            trace!("🔓 Acquired {} lock successfully", lock_name);
            Ok(guard)
        }
        Err(_) => {
            error!(
                "🚨 TIMEOUT: Failed to acquire {} lock within {}s",
                lock_name, timeout_secs
            );
            error!("💀 Likely deadlock detected - check lock ordering and hold durations");
            error!("📊 Lock timeout metrics should be monitored in production");
            anyhow::bail!(
                "Lock timeout on {} after {}s - potential deadlock",
                lock_name,
                timeout_secs
            );
        }
    }
}

/// Acquire a RwLock write lock with timeout
///
/// # Arguments
/// * `rwlock` - The RwLock to acquire
/// * `timeout_secs` - Timeout in seconds (recommended: 5-30s)
/// * `lock_name` - Human-readable lock name for diagnostics
///
/// # Returns
/// * `Ok(guard)` - Successfully acquired write lock
/// * `Err(anyhow::Error)` - Timeout occurred, likely deadlock
pub async fn write_lock_with_timeout<'a, T>(
    rwlock: &'a tokio::sync::RwLock<T>,
    timeout_secs: u64,
    lock_name: &str,
) -> Result<tokio::sync::RwLockWriteGuard<'a, T>> {
    match timeout(Duration::from_secs(timeout_secs), rwlock.write()).await {
        Ok(guard) => {
            trace!("🔓 Acquired {} write lock successfully", lock_name);
            Ok(guard)
        }
        Err(_) => {
            error!(
                "🚨 TIMEOUT: Failed to acquire {} write lock within {}s",
                lock_name, timeout_secs
            );
            error!("💀 Likely deadlock detected - check lock ordering and hold durations");
            error!("📊 Write lock timeout metrics should be monitored in production");
            anyhow::bail!(
                "Write lock timeout on {} after {}s - potential deadlock",
                lock_name,
                timeout_secs
            );
        }
    }
}

/// Acquire a RwLock read lock with timeout
///
/// # Arguments
/// * `rwlock` - The RwLock to acquire
/// * `timeout_secs` - Timeout in seconds (recommended: 5-30s)
/// * `lock_name` - Human-readable lock name for diagnostics
///
/// # Returns
/// * `Ok(guard)` - Successfully acquired read lock
/// * `Err(anyhow::Error)` - Timeout occurred, likely deadlock
pub async fn read_lock_with_timeout<'a, T>(
    rwlock: &'a tokio::sync::RwLock<T>,
    timeout_secs: u64,
    lock_name: &str,
) -> Result<tokio::sync::RwLockReadGuard<'a, T>> {
    match timeout(Duration::from_secs(timeout_secs), rwlock.read()).await {
        Ok(guard) => {
            trace!("🔓 Acquired {} read lock successfully", lock_name);
            Ok(guard)
        }
        Err(_) => {
            error!(
                "🚨 TIMEOUT: Failed to acquire {} read lock within {}s",
                lock_name, timeout_secs
            );
            error!("💀 Likely deadlock detected - check lock ordering and hold durations");
            error!("📊 Read lock timeout metrics should be monitored in production");
            anyhow::bail!(
                "Read lock timeout on {} after {}s - potential deadlock",
                lock_name,
                timeout_secs
            );
        }
    }
}

/// v1.4.10: Contract event record for persistent storage
/// NOTE: Defined early in lib.rs so contracts_api module can import it
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ContractEventRecord {
    pub id: String,
    pub event_type: String,  // "mint", "burn", "airdrop", "transfer", "pause", "unpause"
    pub amount: String,      // Display units (formatted string)
    pub from: Option<String>,
    pub to: Option<String>,
    pub recipients: Option<u32>,  // For airdrop events
    pub timestamp: u64,           // Unix timestamp
    pub tx_hash: String,
}

/// v9.1.0: Compute Power Layer — network-wide hashpower announcements via gossipsub.
/// Each node periodically broadcasts its aggregate mining hashrate so peers can
/// build a real-time picture of total network compute power.  This feeds into:
///   - Gravity-assist peer routing (Phase 3: high-hashpower peers get sync priority)
///   - Live security bits (Phase 5: security = f(cumulative work, live hashrate))
///   - PoW relay stamps (Phase 4: anti-spam requires proof of compute capability)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ComputePowerAnnouncement {
    /// Peer ID of the announcing node
    pub peer_id: String,
    /// Total hashrate in hashes/second aggregated across all connected miners
    pub total_hashrate_hs: f64,
    /// Number of active mining connections
    pub active_miners: u32,
    /// SIMD tier: "avx512", "avx2", "neon", "sse2", "scalar"
    pub simd_tier: String,
    /// Cumulative security bits from block work
    pub security_bits: f64,
    /// Unix timestamp (seconds) when this announcement was created
    pub timestamp: u64,
    /// Ed25519 signature over (peer_id || hashrate || timestamp) for authenticity
    pub signature: String,
}

pub mod config;
pub mod console_viz; // Beautiful animated console visualization
                     // v0.9.1-beta: DEX modules commented out (q_dex/q_oracle crates not yet implemented)
                     // pub mod dex_integration_api;
                     // #[cfg(test)]
                     // pub mod dex_integration_tests;
                     // pub mod dex_handlers;  // ✅ NEW - Dynamic token registry & price history API
                     // pub mod dex_initialization;  // ✅ NEW - DEX component initialization
                     // pub mod liquidity_api;  // Liquidity provision API
pub mod aegis_auth_middleware; // ✅ ENABLED - AEGIS-QL post-quantum authentication for founder operations
pub mod binary_protocol; // High-performance binary ingestion for 1M+ TPS
pub mod cdp_simple; // Simple CDP system for QUGUSD minting (fallback, can be removed)
pub mod chat_api; // ✅ ENABLED - AI chat API with privacy-first distributed inference
pub mod chat_persistence; // P2P peer-to-peer chat message persistence
pub mod group_chat_api; // ✅ ENABLED - Discord-like server-backed group chat
pub mod email_api; // ✅ v7.3.2 - Quillon Mail: decentralized email with crypto transfers
pub mod email_smtp; // ✅ v7.3.2 - Quillon Mail: SMTP server for inbound/outbound
pub mod calendar_api; // ✅ v7.3.3 - Blockchain Calendar: events, scheduled TXs, P2P sync
pub mod email_mta; // ✅ v7.3.2 - Quillon Mail: Mail Transport Agent for SMTP delivery
pub mod email_auth_verify; // ✅ v7.3.2 - Quillon Mail: SPF/DKIM/DMARC verification
pub mod ai_intent; // ✅ v2.3.18-beta - Safe AI intent schema (AI parses, Rust executes)
pub mod agent_panel;
pub mod ai_intent_parser; // ✅ v2.3.18-beta - Mistral 7B intent parsing with validation
pub mod ai_intent_executor; // ✅ v2.3.18-beta - Deterministic Rust intent execution
pub mod ai_intent_manager; // ✅ v2.3.18-beta - Unified intent pipeline with confirmation flow
pub mod ai_function_tools; // ✅ v2.5.0-beta - Ministral-3B native function calling for DEX
pub mod dex_market_analyzer; // ✅ v2.5.0-beta - Agentic market intelligence with Ministral-3B
pub mod verification_api; // ✅ NEW - Proof-of-inference verification monitoring (SSE)
pub mod database_replication_bridge; // Bridge between IPFS replication and gossipsub
pub mod dex_handlers; // ✅ ENABLED - DEX HTTP API handlers
pub mod dex_initialization; // ✅ ENABLED - DEX component initialization
pub mod governance_api; // ✅ v1.0.1 - Proof-of-Contribution governance with mining-weighted voting
pub mod dca_api; // ✅ v2.4.8-beta - Dollar Cost Averaging for automated recurring token purchases
pub mod limit_order_api; // ✅ v10.4.9 - Price-triggered one-shot swaps (limit orders)
pub mod perpetual_api; // ✅ v2.5.0-beta - Perpetual futures with 10x leverage (long/short)
pub mod handlers;
pub mod startup_progress; // ✅ v1.4.15-beta - Startup progress tracker for frontend UI
pub mod adaptive_confirmations; // ✅ v1.4.4-beta - ML-adaptive confirmation with retail-first instant finality
pub mod staking_security; // ✅ v1.4.4-beta - Staking/slashing security with insurance pool for instant payments
pub mod hashrate_tracker; // ✅ v1.0.16-beta - Network hashrate tracking for adaptive security
pub mod high_performance_server; // HTTP/2 server optimized for 1M+ TPS
pub mod oauth2_provider; // ✅ ENABLED - OAuth2 provider for third-party integrations
pub mod p2p_listener;
pub mod paas_admin_api; // ✅ ENABLED - PaaS admin endpoints for CLI management
pub mod paas_api_keys; // ✅ ENABLED - API key management with argon2id hashing
pub mod paas_audit; // ✅ ENABLED - Audit logging and distributed tracing
pub mod paas_auth; // ✅ ENABLED - PaaS authentication and rate limiting
pub mod paas_billing; // ✅ ENABLED - Atomic billing with pre-charge and reserve
pub mod paas_billing_v2; // ✅ ENABLED - Atomic billing v2 with Grok improvements
pub mod paas_idempotency; // ✅ ENABLED - Idempotency support for safe retries
pub mod paas_pricing; // ✅ ENABLED - Dynamic USD pricing with oracle integration
pub mod payment_api; // ✅ ENABLED - Stripe payment processing with async-stripe
pub mod payment_request_api; // ✅ v9.6.1 - QR code payment requests for brick-and-mortar POS
pub mod privacy_service_api; // ✅ ENABLED - Privacy-as-a-Service (PaaS) enterprise API
pub mod quillon_bank_api; // ✅ ENABLED - Full Quillon Bank CDP system
pub mod security_tier_governance; // ✅ v1.0.16-beta - Community governance for VDF security tiers
pub mod storage_api; // IPFS-RocksDB decentralized storage for database backups
pub mod streaming;
pub mod sync_activation; // ✅ v1.0.15-beta - Timeout-based sync activation
pub mod wallet_auth; // Signature-based wallet authentication for privacy
pub mod websocket_stream; // WebSocket streaming for 1M+ TPS (zero HTTP overhead)
pub mod call_manager; // Call lifecycle state machine (capacity + timeout enforcement)
pub mod signaling_server; // Chat/Voice/Video WebRTC signaling (SDP + ICE routing)
pub mod turn_credentials; // TURN credential endpoint (/api/v1/turn/credentials)
pub mod consensus_service; // ✅ v1.3.11-beta: TRUE DECENTRALIZED CONSENSUS with multi-validator signatures
pub mod oracle_integration; // ✅ v1.4.3-beta: Oracle feeds for QNO prediction resolution
pub mod zcash_api;
pub mod zcash_rpc; // ✅ v1.0.15-beta - Zcash RPC client for Zebra node integration // ✅ v1.0.15-beta - Zcash wallet API endpoints (address, balance, send)
pub mod privacy_service; // ✅ v2.5.0 - Privacy Layer with zk-STARK + AEGIS-QL
                                                                                    // pub mod sync_activation;  // ❌ DUPLICATE - Already declared on line 73
                                                                                    // pub mod supply_persistence;  // 🔒 DEACTIVATED - Will be implemented in v0.0.10
                                                                                    // io_uring is Linux kernel's async I/O interface (requires Linux kernel ≥5.1)
pub mod block_producer; // 🏗️ Block producer - aggregates mining solutions into QBlocks
#[cfg(target_os = "linux")]
pub mod io_uring_adapter; // Safe io_uring wrapper to avoid runtime conflicts
pub mod lockfree_producer;
pub mod parallel_workers; // 16x parallel worker pool for high TPS // 🔓 v0.9.92-beta: Lock-free producer - DEADLOCK FIX
pub mod transaction_utils; // ✅ v1.0.91-beta: Proper transaction handling with nonce management
pub mod contracts_api; // ✅ v2.4.8-beta - Smart contract deployment and social media profiles (AFTER transaction_utils!)
pub mod listing_api; // ✅ v6.5.0: Exchange Listing RWA packages (Gold/Silver/Bronze)
pub mod game_items_api; // ✅ v9.3.0: CS:GO2-style game items RWA (cases, skins, trade-up)
pub mod web_search_api; // ✅ v9.3.2: GLM-4-Flash web search with AI summaries + citations
pub mod compute_api; // ✅ v9.5.0: Starship Endgame — Compute Orchestrator API
pub mod ai_api; // ✅ v9.5.0: Starship Endgame — AI Inference Pool API
pub mod k_parameter_gauge; // ✅ v9.3.1: Lightweight K-parameter network health gauge (no q-resonance dep)
pub mod bitcoin_bridge_api; // ✅ v7.2.0: Bitcoin atomic swap bridge (QNK ↔ BTC)
pub mod bitcoin_deposit_api; // ✅ v10.2.11: Bitcoin deposit bridge — receive BTC, mint wBTC
pub mod bitcoin_lp_api; // ✅ v10.9.21: One-click "deposit BTC → become bridge LP"
pub mod bitcoin_rpc; // ✅ v9.6.5: Bitcoin Knots RPC client (balance, address, send, txs)
pub mod zcash_bridge_api; // ✅ v7.2.2: Zcash shielded atomic swap bridge (QNK ↔ ZEC)
pub mod ironfish_bridge_api; // ✅ v7.2.4: Iron Fish privacy atomic swap bridge (QNK ↔ IRON)
pub mod ethereum_bridge_api; // ✅ v7.3.0: Ethereum atomic swap bridge (QNK ↔ ETH)
pub mod bridge_committee; // ✅ v7.3.1: Multi-sig bridge validation with rotating 11-node committee
pub mod bridge_tokens; // ✅ v7.2.5: Wrapped bridge tokens (wBTC, wZEC, wIRON) mint/burn system
pub mod bridge_safety; // ✅ v9.4.0: Bridge safety layer — deposit verification, kill-switch, amount limits
pub mod integrity_api;  // ✅ v10.7.0: Data integrity & decentralization diagnostics API
pub mod sharkgod; // 🦈 SharkGod: Maximum power transaction beam (bypass all bottlenecks)
pub mod swap_indexer; // ✅ v2.4.0-beta: Consensus-verified swap history indexer
pub mod price_history_indexer; // ✅ v3.7.1-beta: Consensus-verified price history indexer
pub mod mining_commit_reveal; // ✅ v1.4.11-beta: Commit-reveal cryptographic time-locks for mining
pub mod stake_weighted_finality; // ✅ v1.4.11-beta: Stake-weighted finality with PoW+PoS hybrid security
pub mod pool_api; // ✅ v2.2.1-beta: Stratum mining pool HTTP API
pub mod temporal_api; // ✅ v2.3.5-beta: TemporalShield-STARK secret sharing with NO TRUSTED SETUP
pub mod trustee_manager; // ✅ v2.4.1-beta: HSM-backed trustee key management for TemporalShield
pub mod temporal_memo; // ✅ v2.4.1-beta: TemporalShield protection for private TX memos
pub mod validator_backup_api; // ✅ v2.7.0-beta: TemporalShield validator key backup (5-of-9 threshold)
pub mod chat_protector; // ✅ v2.7.0-beta: TemporalShield protection for AI chat content (3-of-5 threshold)
pub mod bootstrap_config; // ✅ v2.9.0-beta: Multi-bootstrap with automatic failover (decentralization)
pub mod upgrade_verifier; // ✅ v5.1.1: Safe rolling deployment verification
pub mod admin_settings_api; // ✅ v7.3.0: Node operator admin settings API (--admin-wallet)
pub mod deploy_admin_api; // ✅ v5.1.1: Deploy admin panel API (master-wallet-only)
pub mod node_auto_updater; // ✅ v8.5.0: P2P auto-update with Ed25519 quorum verification
pub mod state_sync_api; // ✅ v5.2.0: HTTP full state sync (contracts, pools, balances) from bootstrap peers
pub mod miner_link_api; // ✅ v7.2.0: WebSocket relay for wallet ↔ personal miner communication
pub mod node_setup; // v8.6.5: Automatic node setup wizard via OAuth2 device login
pub mod quorum_commit; // ✅ Phase 1: Multi-validator balance agreement (quorum commit broadcast)
pub mod equivocation_watcher; // 🛡️  Background anti-equivocation detector (double-signing alarm)

pub use config::Config;
pub use console_viz::{update_stats, ConsensusStats, ConsoleVisualizer};
pub use streaming::{EventBroadcaster, HighPerformanceEmitter, StreamEvent, SseQueryParams, WsQueryParams};
pub use contracts_api::TokenSocialProfile;

// v7.0.0: FaucetState, FaucetRequestRecord, AbusePattern removed — faucet eliminated

/// v9.4.0: Check if a wallet address is the master (founder) wallet
pub fn is_master_wallet(address: &[u8; 32]) -> bool {
    let founder_hex = aegis_auth_middleware::FOUNDER_WALLET;
    hex::encode(address) == founder_hex
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
/// v3.2.16-beta: Added token decimals for cross-decimal-base swap calculations
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LiquidityPool {
    pub pool_id: String,
    pub token0: String, // Native QUG or token contract address
    pub token1: String, // Token contract address
    #[serde(serialize_with = "q_types::u128_serde::serialize", deserialize_with = "q_types::u128_serde::deserialize")]
    pub reserve0: u128,
    #[serde(serialize_with = "q_types::u128_serde::serialize", deserialize_with = "q_types::u128_serde::deserialize")]
    pub reserve1: u128,
    pub provider: [u8; 32], // Wallet address that provided liquidity
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Total LP token supply (calculated using Uniswap V2 formula: sqrt(reserve0 * reserve1))
    /// For existing pools: proportional minting
    /// v0.6.0-beta: DEX Decentralization
    #[serde(default)]
    #[serde(serialize_with = "q_types::u128_serde::serialize", deserialize_with = "q_types::u128_serde::deserialize")]
    pub lp_token_supply: u128,
    /// v3.2.16-beta: Decimal places for token0 (QUG/QUGUSD = 24, custom tokens = 8)
    /// Used for normalizing reserves during swap calculations across different decimal bases
    #[serde(default = "default_token_decimals")]
    pub token0_decimals: u8,
    /// v3.2.16-beta: Decimal places for token1 (QUG/QUGUSD = 24, custom tokens = 8)
    #[serde(default = "default_token_decimals")]
    pub token1_decimals: u8,
}

/// Default token decimals for backwards compatibility with existing pools
fn default_token_decimals() -> u8 {
    24 // Default to 24 for QUG/QUGUSD (existing pools assumed to be QUG pairs)
}

/// v7.2.5: Bootstrap bridge token AMM pools (wBTC/QUG, wZEC/QUG, wIRON/QUG, wETH/QUG)
/// v8.2.7: Dynamic oracle prices — fetches live BTC/ETH/ZEC prices from CoinGecko/Binance
/// Called during AppState initialization to create cross-chain trading pairs
pub async fn bootstrap_bridge_pools(
    liquidity_pools_map: &mut HashMap<String, LiquidityPool>,
    storage_engine: &std::sync::Arc<q_storage::StorageEngine>,
    // v10.9.21: Kept in the signature for callsite compat. Empty-shell pools don't price
    // themselves — the first LP defines the QUG/wBTC ratio — so the bootstrap QUG price
    // is no longer needed here.
    _qug_price: f64,
    oracle: Option<&q_quillon_bank::oracle_integration::BankingOracleIntegration>,
) {
    // v9.0.4: Fetch LIVE prices from oracle (CoinGecko → Binance fallback)
    // Fallback prices updated 2026-03-04 to reflect current market
    use q_quillon_bank::AssetType;
    let btc_price = if let Some(orc) = oracle {
        let p = orc.get_price_f64(&AssetType::BTC).await;
        if p > 0.0 { p } else { 73_000.0 }
    } else { 73_000.0 };
    let eth_price = if let Some(orc) = oracle {
        let p = orc.get_price_f64(&AssetType::ETH).await;
        if p > 0.0 { p } else { 2_100.0 }
    } else { 2_100.0 };
    let zec_price = if let Some(orc) = oracle {
        let p = orc.get_price_f64(&AssetType::ZEC).await;
        if p > 0.0 { p } else { 237.0 }
    } else { 237.0 };
    let iron_price = 0.008; // Iron Fish not on major exchanges

    tracing::info!("💹 [ORACLE] Bridge pool prices: BTC=${:.0}, ETH=${:.0}, ZEC=${:.2}, IRON=${:.4}",
        btc_price, eth_price, zec_price, iron_price);

    // Bridge pool definitions: (pool_id, wrapped_symbol, native_price_usd, initial_amount_native)
    let bridge_pools: [(&str, &str, f64, f64); 4] = [
        ("pool-qug-wbtc-bridge", "wBTC", btc_price, 0.25),    // 0.25 BTC
        ("pool-qug-weth-bridge", "wETH", eth_price, 2.0),     // 2.0 ETH
        ("pool-qug-wzec-bridge", "wZEC", zec_price, 100.0),   // 100 ZEC
        ("pool-qug-wiron-bridge", "wIRON", iron_price, 50_000.0),  // 50K IRON
    ];

    for (pool_id, symbol, native_price_usd, _native_amount) in &bridge_pools {
        // v10.9.21: HONEST LIQUIDITY MODEL.
        //
        // Previously this function minted ~0.25 BTC worth of wBTC (and matching wZEC/wETH/wIRON)
        // *out of thin air* on every restart. The pool let users buy wBTC tokens, but the
        // bridge wallet on Delta held 0 BTC — so withdrawals to real Bitcoin always failed,
        // and users ended up holding unredeemable IOUs.
        //
        // New behaviour: pool exists as an empty shell. Both reserves start at 0. The pool
        // only gets liquidity from real user LP deposits (see /api/v1/bitcoin/lp/intent for
        // the one-click "deposit BTC + pair with QUG" flow). Until at least one LP arrives,
        // the swap handler returns "no liquidity yet — be the first LP" instead of trading
        // against fake supply.
        //
        // Invariant after this change: `total_wrapped_token_supply <= bridge_wallet_balance`
        // for each bridge token. Trading is honest; withdrawals always work.
        let empty_pool = LiquidityPool {
            pool_id: pool_id.to_string(),
            token0: "QUG".to_string(),
            token1: symbol.to_string(),
            reserve0: 0,
            reserve1: 0,
            provider: [0u8; 32], // System-owned shell; LPs are tracked via lp_token_supply
            created_at: chrono::Utc::now(),
            lp_token_supply: 0,
            token0_decimals: 24,
            token1_decimals: 24,
        };

        if let Some(existing) = liquidity_pools_map.get_mut(*pool_id) {
            // Detect legacy fake-seeded reserves (lp_token_supply > 0 but provider is the
            // system zero-address). Those were minted by the old bootstrap code and have no
            // real BTC backing. Drain them once on this upgrade so users can't trade against
            // unbacked supply.
            let is_legacy_fake_seed = existing.lp_token_supply > 0 && existing.provider == [0u8; 32];
            if is_legacy_fake_seed {
                tracing::warn!(
                    "🌉 [BRIDGE] {} pool had legacy fake-seeded reserves (r0={:.4}, r1={:.4}) — \
                     draining to zero. Real LPs must deposit via /api/v1/bitcoin/lp/intent.",
                    symbol,
                    existing.reserve0 as f64 / 1e24,
                    existing.reserve1 as f64 / 1e24,
                );
                *existing = empty_pool.clone();
            } else {
                tracing::info!(
                    "🌉 [BRIDGE] {} pool: live LP reserves preserved (r0={:.4} QUG, r1={:.4} {}, lp_supply={:.4})",
                    symbol,
                    existing.reserve0 as f64 / 1e24,
                    existing.reserve1 as f64 / 1e24,
                    symbol,
                    existing.lp_token_supply as f64 / 1e24,
                );
            }
        } else {
            liquidity_pools_map.insert(pool_id.to_string(), empty_pool.clone());
            tracing::info!(
                "🌉 [BRIDGE] Created empty {} pool shell @ ${:.0}/unit reference — awaiting first LP",
                symbol, native_price_usd
            );
        }
        // Persist updated/created pool to DB
        if let Some(p) = liquidity_pools_map.get(*pool_id) {
            if let Ok(pool_bytes) = serde_json::to_vec(p) {
                if let Err(e) = storage_engine.save_liquidity_pool(pool_id, &pool_bytes).await {
                    tracing::warn!("⚠️ Failed to persist bridge {} pool: {}", symbol, e);
                }
            }
        }
    }

    // v9.4.0: Bootstrap QUGUSD ↔ bridge pools (QUGUSD/wBTC, QUGUSD/wETH, etc.)
    // Enables direct QUGUSD stablecoin swaps to wrapped bridge tokens
    let qugusd_bridge_pools: [(&str, &str, f64, f64); 4] = [
        ("pool-qugusd-wbtc-bridge", "wBTC", btc_price, 0.25),
        ("pool-qugusd-weth-bridge", "wETH", eth_price, 2.0),
        ("pool-qugusd-wzec-bridge", "wZEC", zec_price, 100.0),
        ("pool-qugusd-wiron-bridge", "wIRON", iron_price, 50_000.0),
    ];

    for (pool_id, symbol, native_price_usd, _native_amount) in &qugusd_bridge_pools {
        // v10.9.21: Empty-shell pool (see QUG/wBTC explanation above). No fake seeding.
        let empty_pool = LiquidityPool {
            pool_id: pool_id.to_string(),
            token0: "QUGUSD".to_string(),
            token1: symbol.to_string(),
            reserve0: 0,
            reserve1: 0,
            provider: [0u8; 32],
            created_at: chrono::Utc::now(),
            lp_token_supply: 0,
            token0_decimals: 24,
            token1_decimals: 24,
        };

        if let Some(existing) = liquidity_pools_map.get_mut(*pool_id) {
            let is_legacy_fake_seed = existing.lp_token_supply > 0 && existing.provider == [0u8; 32];
            if is_legacy_fake_seed {
                tracing::warn!(
                    "🌉 [BRIDGE] QUGUSD/{} pool had legacy fake-seeded reserves — draining to zero.",
                    symbol
                );
                *existing = empty_pool.clone();
            } else {
                tracing::info!(
                    "🌉 [BRIDGE] QUGUSD/{} pool: live LP reserves preserved",
                    symbol
                );
            }
        } else {
            liquidity_pools_map.insert(pool_id.to_string(), empty_pool.clone());
            tracing::info!(
                "🌉 [BRIDGE] Created empty QUGUSD/{} pool shell @ ${:.0}/unit reference — awaiting first LP",
                symbol, native_price_usd
            );
        }
        if let Some(p) = liquidity_pools_map.get(*pool_id) {
            if let Ok(pool_bytes) = serde_json::to_vec(p) {
                if let Err(e) = storage_engine.save_liquidity_pool(pool_id, &pool_bytes).await {
                    tracing::warn!("⚠️ Failed to persist QUGUSD/{} pool: {}", symbol, e);
                }
            }
        }
    }
    tracing::info!("🌉 [BRIDGE v9.4.0] All bridge pools ready: QUG+QUGUSD ↔ wBTC/wETH/wZEC/wIRON");
}

/// Mining submission for async queue processing
/// v3.3.3-beta: Added miner_id and worker_name for miner identification
#[derive(Debug, Clone)]
pub struct MiningSubmission {
    pub nonce: u64,
    pub hash: [u8; 32],
    pub difficulty_target: [u8; 32],
    pub miner_address: [u8; 32],
    pub miner_address_str: String,
    pub hash_rate: f64, // Hash rate in KH/s
    /// v3.3.3-beta: Unique miner instance ID (auto-generated if not provided)
    pub miner_id: Option<String>,
    /// v3.3.3-beta: Human-readable miner name (e.g., "Server Alpha", "Mining Rig 1")
    pub worker_name: Option<String>,
    /// v1.0.2: Deferred VDF verification — challenge hash bytes for background validation
    /// Moved from HTTP handler to background processor for 10x throughput improvement
    pub challenge_hash_bytes: Option<[u8; 32]>,
    /// v1.0.2: Miner version for update check (moved from HTTP-only to background SSE)
    pub miner_version: Option<String>,
    /// v10.2.3: VDF iteration count from challenge — used for dynamic server-side verification.
    /// Defaults to 99 (100 total with initial hash) for backwards compatibility.
    pub vdf_iterations: u32,

    /// v1.0.5: Genus-2 VDF output (Mumford representation serialized)
    /// Present when miner uses real Genus-2 VDF (above GENUS2_VDF_MINING activation height)
    pub genus2_vdf_output: Option<Vec<u8>>,

    /// v1.0.5: Wesolowski proof for O(log T) VDF verification
    pub genus2_vdf_proof: Option<Vec<u8>>,

    /// v1.0.5: VDF intermediate checkpoints
    pub genus2_vdf_checkpoints: Option<Vec<Vec<u8>>>,

    /// v1.0.5: Number of Genus-2 VDF iterations (T)
    pub genus2_vdf_iterations: Option<u64>,
}

/// ⚡ v8.9.0: Lightweight SSE mining event for decoupled broadcast pipeline.
/// Batch processors send these to the SSE aggregator task at 10Hz.
#[derive(Debug, Clone)]
pub enum SseMiningEvent {
    /// A mining reward was accepted (pending consensus confirmation)
    MiningReward {
        wallet: String,
        hash_rate: f64,
        nonce: u64,
        miner_id: Option<String>,
        worker_name: Option<String>,
    },
    /// Balance update notification (pending, for UI feedback)
    BalanceUpdate {
        wallet: String,
        old_balance: u128,
        new_balance: u128,
        solution_count: usize,
    },
}

// v7.0.0: FaucetState impl removed — faucet eliminated

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
    pub last_hashrate: f64, // H/s (v3.5.6-beta: changed from KH/s to H/s for frontend compatibility)
    pub last_update: std::time::Instant,
    pub total_solutions: u64,
    /// v3.3.4-beta: Worker identifier to distinguish multiple miners to same wallet
    /// Format: "direct" for local submissions, "p2p:NODE_ID" for P2P relayed, or custom worker_name
    pub worker_id: String,
    /// v7.4.2: Human-readable miner name (e.g., "My Rig", "Server Alpha")
    /// Set from --miner-name CLI arg. Separate from worker_id which is used as internal key.
    pub worker_name: Option<String>,
    /// v3.5.3-beta: Track recent solution timestamps for accurate hashrate calculation
    /// Stores up to 100 recent solution timestamps for rolling hashrate computation
    pub solution_timestamps: Vec<std::time::Instant>,
    /// v3.5.7-beta: Track actual blocks found per worker (not just solutions submitted)
    /// Incremented when a mining solution results in a block being added to the chain
    pub blocks_found: u64,
    /// v3.5.7-beta: Track total rewards earned per worker in base units (1e-24 QUG)
    /// Allows comparing profitability between different mining rigs
    pub rewards_earned: u128,
}

/// v3.2.12-beta: Serde helper for Option<u128> to string serialization
/// JSON cannot handle integers larger than 2^53, and mining rewards are ~5*10^25 base units
/// This serializes Option<u128> as Option<String> for safe JSON transport
mod u128_string_option {
    use serde::{self, Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(value: &Option<u128>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match value {
            Some(v) => serializer.serialize_some(&v.to_string()),
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<u128>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let opt: Option<String> = Option::deserialize(deserializer)?;
        match opt {
            Some(s) => {
                // Try parsing as u128 string first
                if let Ok(v) = s.parse::<u128>() {
                    return Ok(Some(v));
                }
                // Fallback: try parsing as f64 for old JSON number format (lossy but backwards compatible)
                if let Ok(v) = s.parse::<f64>() {
                    return Ok(Some(v as u128));
                }
                // Last resort: try the raw string
                s.parse::<u128>().map(Some).map_err(serde::de::Error::custom)
            }
            None => Ok(None),
        }
    }
}

/// v1.0.88-beta: P2P Miner Stats Update
/// Serializable miner statistics for gossipsub broadcast
/// Allows users mining to localhost nodes to have their hashrate visible on bootstrap node dashboard
///
/// v1.3.8-beta: Added pending_reward for instant UI feedback
/// NOTE: pending_reward is INFORMATIONAL ONLY - it does NOT affect consensus or database.
/// The actual balance is only updated when the block containing the coinbase transaction
/// is committed to the DAG-Knight consensus. This field enables instant UI feedback
/// while maintaining consensus integrity.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct P2PMinerStatsUpdate {
    /// Miner wallet address (qnk...)
    pub miner_address: String,
    /// Current hashrate in H/s (v3.5.6-beta: changed from KH/s for frontend compatibility)
    pub hashrate_khs: f64,
    /// Total solutions found by this miner
    pub total_solutions: u64,
    /// Unix timestamp of this update
    pub timestamp: u64,
    /// Node ID that originated this update (for deduplication)
    pub origin_node_id: String,
    /// v3.5.6-beta: Worker ID to distinguish multiple miners to same wallet
    /// Format: "direct", "miner1", "p2p:NODE_ID", etc.
    #[serde(default)]
    pub worker_id: Option<String>,
    /// v1.3.8-beta: Pending reward from latest mining submission (QUG base units)
    /// This is for UI display ONLY - actual balance updates via DAG-Knight consensus
    /// when the block with coinbase transaction is committed.
    /// v3.2.12-beta: Use string serialization for u128 (JSON can't handle >2^53 integers)
    #[serde(default, with = "u128_string_option")]
    pub pending_reward: Option<u128>,
    /// v1.3.8-beta: Cumulative pending rewards this session (not yet in blocks)
    /// Resets when rewards are confirmed in committed blocks
    /// v3.2.12-beta: Use string serialization for u128 (JSON can't handle >2^53 integers)
    #[serde(default, with = "u128_string_option")]
    pub session_pending_total: Option<u128>,
    /// v3.5.7-beta: Actual blocks found by this worker (not just solutions submitted)
    #[serde(default)]
    pub blocks_found: u64,
    /// v3.5.7-beta: Total rewards earned by this worker in base units (1e-24 QUG)
    /// v3.5.7-beta: Use string serialization for u128 (JSON can't handle >2^53 integers)
    #[serde(default, with = "u128_string_option")]
    pub rewards_earned: Option<u128>,
}

/// v2.2.1: Batched miner stats update to prevent gossipsub queue saturation
/// Instead of sending one message per miner per submission, aggregate all updates
/// into a single batched message per broadcast interval (reduces P2P traffic 10-100x)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct P2PMinerStatsBatch {
    /// All miner updates in this batch
    pub updates: Vec<P2PMinerStatsUpdate>,
    /// Batch timestamp (Unix seconds)
    pub batch_timestamp: u64,
    /// Origin node ID
    pub origin_node_id: String,
    /// Batch sequence number (for ordering/dedup)
    pub batch_seq: u64,
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
    /// v3.3.4-beta: Added worker_id to distinguish multiple miners to same wallet
    pub fn update_miner(&mut self, miner_address: String, hash_rate: f64) {
        self.update_miner_with_worker(miner_address, hash_rate, "direct".to_string(), None);
    }

    /// v3.3.4-beta: Update miner with specific worker identifier
    /// worker_id: "direct" for local, "p2p:NODE_ID" for P2P relayed
    /// v3.5.3-beta: Calculate hashrate from solution submissions instead of client-reported values
    /// v3.5.4-beta: Returns the calculated hashrate for use in SSE events
    /// v3.9.3-beta: Added memory management - caps entries and resets counters periodically
    pub fn update_miner_with_worker(&mut self, miner_address: String, hash_rate: f64, worker_id: String, worker_name: Option<String>) -> f64 {
        // Use composite key: address:worker_id to track miners separately
        let key = format!("{}:{}", miner_address, worker_id);
        let now = std::time::Instant::now();

        // v3.9.3-beta: Periodic cleanup to prevent memory leak (every 30 seconds)
        if now.duration_since(self.last_cleanup).as_secs() > 30 {
            // Remove miners inactive for more than 5 minutes
            self.active_miners
                .retain(|_, stats| now.duration_since(stats.last_update).as_secs() < 300);

            // v3.9.3-beta: Reset total counters every hour to prevent u64 overflow and keep metrics fresh
            // Counters can be reconstructed from active_miners if needed
            if self.total_solutions_submitted > 10_000_000 {
                tracing::info!("🧹 [MEMORY] Resetting mining stats counters (submitted: {}, accepted: {})",
                    self.total_solutions_submitted, self.total_solutions_accepted);
                self.total_solutions_submitted = 0;
                self.total_solutions_accepted = 0;
                // Also reset per-miner total_solutions to prevent memory bloat in stats
                for stats in self.active_miners.values_mut() {
                    stats.total_solutions = stats.total_solutions.min(100_000); // Cap at 100K
                }
            }

            self.last_cleanup = now;
            tracing::debug!("🧹 [MEMORY] Mining stats cleanup: {} active miners", self.active_miners.len());
        }

        // v3.9.3-beta: Cap maximum number of tracked miners to prevent unbounded growth
        const MAX_TRACKED_MINERS: usize = 500;
        if self.active_miners.len() >= MAX_TRACKED_MINERS && !self.active_miners.contains_key(&key) {
            // Remove the oldest inactive miner to make room
            if let Some(oldest_key) = self.active_miners
                .iter()
                .min_by_key(|(_, stats)| stats.last_update)
                .map(|(k, _)| k.clone())
            {
                self.active_miners.remove(&oldest_key);
            }
        }

        let stats = self
            .active_miners
            .entry(key)
            .or_insert(MinerStats {
                address: miner_address,
                last_hashrate: 0.0,
                last_update: now,
                total_solutions: 0,
                worker_id: worker_id.clone(),
                worker_name: worker_name.clone(),
                solution_timestamps: Vec::with_capacity(120), // Pre-allocate for ~2 solutions/sec
                blocks_found: 0,      // v3.5.7-beta: Track actual blocks found
                rewards_earned: 0,    // v3.5.7-beta: Track rewards in base units
            });
        // v7.4.2: Update worker_name if provided (may not be set on first insert from stats event)
        if worker_name.is_some() {
            stats.worker_name = worker_name;
        }

        // v3.5.3-beta: Track solution timestamp for hashrate calculation
        stats.solution_timestamps.push(now);

        // Keep only solutions from last 60 seconds for hashrate calculation
        let sixty_secs_ago = now - std::time::Duration::from_secs(60);
        stats.solution_timestamps.retain(|&t| t > sixty_secs_ago);

        // v3.9.3-beta: Cap solution_timestamps to prevent unbounded growth (max ~120 for 2/sec)
        if stats.solution_timestamps.len() > 200 {
            stats.solution_timestamps.drain(0..100);
        }

        // v4.1.2: Use miner-reported hashrate directly (miner counts every hash attempt accurately)
        // Miner sends hashrate in KH/s, convert to H/s for display
        let hash_rate_hs = hash_rate * 1000.0; // Convert KH/s to H/s

        // Use client-reported hashrate if provided, otherwise estimate from solutions
        // The miner's own hash counter is the most accurate source
        if hash_rate_hs > 0.0 {
            stats.last_hashrate = hash_rate_hs;
        } else {
            // Fallback: estimate from solution rate (for miners that don't report hashrate)
            let solutions_in_window = stats.solution_timestamps.len() as f64;
            let time_window_secs = 60.0;
            let difficulty_factor = 65_536.0;
            stats.last_hashrate = (solutions_in_window * difficulty_factor) / time_window_secs;
        }

        stats.last_update = now;
        stats.total_solutions += 1;
        self.total_solutions_submitted += 1;

        // v3.5.4-beta: Return the calculated/effective hashrate for SSE events
        stats.last_hashrate
    }

    /// Calculate total network hash rate from active miners
    pub fn calculate_network_hashrate(&mut self) -> f64 {
        // Clean up stale miners (no activity in last 5 minutes)
        let now = std::time::Instant::now();
        if now.duration_since(self.last_cleanup).as_secs() > 60 {
            self.active_miners
                .retain(|_, stats| now.duration_since(stats.last_update).as_secs() < 300);
            self.last_cleanup = now;
        }

        // Sum hash rates from all active miners
        self.active_miners
            .values()
            .map(|stats| stats.last_hashrate)
            .sum()
    }

    /// Get count of active miners
    pub fn active_miner_count(&self) -> usize {
        let now = std::time::Instant::now();
        self.active_miners
            .values()
            .filter(|stats| now.duration_since(stats.last_update).as_secs() < 300)
            .count()
    }

    /// v1.0.88-beta: Update miner stats from P2P network
    /// Called when receiving miner stats from remote nodes (users mining to localhost)
    pub fn update_from_p2p(&mut self, update: &P2PMinerStatsUpdate) {
        // v3.5.6-beta: Use worker_id from P2P update if available (preserves original worker name)
        // Fallback to p2p:{node_id} for compatibility with older nodes
        let worker_id = update.worker_id.clone().unwrap_or_else(|| {
            format!("p2p:{}", &update.origin_node_id[..12.min(update.origin_node_id.len())])
        });
        let key = format!("{}:{}", update.miner_address, worker_id);
        let now = std::time::Instant::now();

        let stats = self
            .active_miners
            .entry(key)
            .or_insert(MinerStats {
                address: update.miner_address.clone(),
                last_hashrate: 0.0,
                last_update: now,
                total_solutions: 0,
                worker_id: worker_id.clone(),
                worker_name: None, // P2P stats don't include worker_name yet
                solution_timestamps: Vec::new(),
                blocks_found: 0,      // v3.5.7-beta: Track actual blocks found
                rewards_earned: 0,    // v3.5.7-beta: Track rewards in base units
            });

        // Update with P2P data - use max hashrate to avoid stale data overwriting
        if update.hashrate_khs > stats.last_hashrate ||
           now.duration_since(stats.last_update).as_secs() > 30 {
            stats.last_hashrate = update.hashrate_khs;
            stats.last_update = now;
        }

        // Track total solutions (use max to avoid counting same solutions twice)
        if update.total_solutions > stats.total_solutions {
            stats.total_solutions = update.total_solutions;
        }

        // v3.5.7-beta: Track blocks found and rewards earned from P2P
        if update.blocks_found > stats.blocks_found {
            stats.blocks_found = update.blocks_found;
        }
        if let Some(rewards) = update.rewards_earned {
            if rewards > stats.rewards_earned {
                stats.rewards_earned = rewards;
            }
        }
    }

    /// v3.5.7-beta: Record a block being found by a specific worker
    /// Called when a mining solution results in a block being added to the chain
    pub fn record_block_found(&mut self, miner_address: &str, worker_id: &str, reward_amount: u128) {
        let key = format!("{}:{}", miner_address, worker_id);
        if let Some(stats) = self.active_miners.get_mut(&key) {
            stats.blocks_found += 1;
            stats.rewards_earned += reward_amount;
            tracing::info!(
                "🏆 [MINING] Block found by {}:{} - total blocks: {}, total rewards: {} QUG",
                &miner_address[..16.min(miner_address.len())],
                worker_id,
                stats.blocks_found,
                stats.rewards_earned as f64 / 1e24
            );
        } else {
            // Worker not found in stats - try to find any worker for this address
            for (k, stats) in self.active_miners.iter_mut() {
                if k.starts_with(miner_address) {
                    stats.blocks_found += 1;
                    stats.rewards_earned += reward_amount;
                    tracing::info!(
                        "🏆 [MINING] Block found by {} (matched via address) - total blocks: {}, total rewards: {} QUG",
                        &miner_address[..16.min(miner_address.len())],
                        stats.blocks_found,
                        stats.rewards_earned as f64 / 1e24
                    );
                    return;
                }
            }
            tracing::warn!(
                "⚠️ [MINING] Block found but miner {}:{} not in active stats",
                miner_address, worker_id
            );
        }
    }

    /// v10.1.1: Record additional solution reward without incrementing blocks_found
    /// Used when a miner has multiple solutions in the same block
    pub fn record_solution_reward(&mut self, miner_address: &str, worker_id: &str, reward_amount: u128) {
        let key = format!("{}:{}", miner_address, worker_id);
        if let Some(stats) = self.active_miners.get_mut(&key) {
            stats.rewards_earned += reward_amount;
        } else {
            for (k, stats) in self.active_miners.iter_mut() {
                if k.starts_with(miner_address) {
                    stats.rewards_earned += reward_amount;
                    return;
                }
            }
        }
    }

    /// v3.3.4-beta: Get all miners for a given wallet address (across all workers)
    pub fn get_miners_for_address(&self, address: &str) -> Vec<&MinerStats> {
        self.active_miners
            .iter()
            .filter(|(key, _)| key.starts_with(address))
            .map(|(_, stats)| stats)
            .collect()
    }
}

// 🔧 v1.0.4-beta: Challenge caching for mining stall prevention
/// Cached mining challenge to ensure consistent challenge_hash across API requests for same height
#[derive(Clone, Debug)]
pub struct CachedChallenge {
    pub challenge_hash: String,
    pub difficulty_target: String,
    pub block_height: u64,
    pub vdf_iterations: u32,
    pub block_reward: f64,
    pub issued_at: chrono::DateTime<chrono::Utc>,
    pub expires_at: chrono::DateTime<chrono::Utc>,
}

/// 🔧 v1.0.5-beta Phase 2: Solution deduplication cache
/// Prevents duplicate solution submissions by tracking (height, solution_hash)
/// Auto-expires entries older than 5 minutes
#[derive(Debug, Clone)]
pub struct SolutionDedupEntry {
    pub height: u64,
    pub solution_hash: String,
    pub submitted_at: chrono::DateTime<chrono::Utc>,
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
    // v2.7.9-beta: Changed from u64 to u128 to support larger token supplies (up to 10^38)
    pub token_balances: Arc<RwLock<HashMap<([u8; 32], [u8; 32]), u128>>>,
    // Liquidity pools: pool_id -> (token0, token1, reserve0, reserve1, provider)
    pub liquidity_pools: Arc<RwLock<HashMap<String, LiquidityPool>>>,
    // Nitro boosts: token_id -> total_boost_points (aggregated from all wallets)
    pub nitro_boosts: Arc<RwLock<HashMap<String, u64>>>,
    pub storage_engine: Arc<StorageEngine>, // ✅ v0.9.27-beta: Persistent storage includes balance consensus

    // ✅ v1.0.91-beta: Nonce tracker for replay attack prevention
    // Each wallet has a monotonically increasing nonce
    pub nonce_tracker: Arc<transaction_utils::NonceTracker>,

    // ✅ v9.7.0: Cross-block transaction dedup cache — prevents replay of applied tx IDs
    // Maps tx_hash → block_height where it was applied. Pruned for entries >1000 blocks old.
    pub applied_tx_dedup: Arc<dashmap::DashMap<[u8; 32], u64>>,

    // ✅ v0.9.99-beta: Adaptive Block Rewards - Throughput-independent emission
    /// Balance consensus engine with adaptive reward calculation
    /// Ensures constant 2,625,000 QUG/year (Era 0) emission regardless of network throughput (1-10,000+ bps)
    pub balance_consensus_engine: Arc<q_storage::BalanceConsensusEngine>,

    pub event_broadcaster: Arc<EventBroadcaster>,
    pub event_emitter: Arc<HighPerformanceEmitter>,

    // v7.0.0: Faucet removed — all QUG earned through mining

    // 🔒 MAX SUPPLY ENFORCEMENT - Post-Quantum Consensus Protected
    // Total supply tracking with Dilithium5 signature verification
    pub total_minted_supply: Arc<RwLock<u128>>, // Total QNK minted across all wallets
    pub supply_consensus_state: Arc<RwLock<SupplyConsensusState>>, // libp2p consensus state

    // ⛏️ MINING STATISTICS - Real-time network hash rate tracking
    pub mining_statistics: Option<Arc<RwLock<MiningStatistics>>>, // Track active miners and hash rates

    // 💓 MINING HEARTBEAT MONITORING (v0.8.9-beta) - Detect mining stalls
    /// Timestamp of last mining solution received (Unix timestamp)
    pub last_mining_solution_time: Arc<std::sync::atomic::AtomicU64>,
    /// Flag indicating if mining is healthy (true = solutions arriving)
    pub mining_is_healthy: Arc<std::sync::atomic::AtomicBool>,

    // ⚡ v8.9.0: Lock-free solution counters (replaces RwLock for hot path stats)
    pub mining_solutions_submitted: Arc<std::sync::atomic::AtomicU64>,
    pub mining_solutions_accepted: Arc<std::sync::atomic::AtomicU64>,

    // ⚡ v8.9.0: Miner stats channel (batch processor → stats aggregator at 1Hz)
    pub miner_stats_tx: Option<tokio::sync::mpsc::Sender<(String, f64, Option<String>, Option<String>)>>,

    // ⚡ v8.9.0: SSE mining event channel (batch processor → SSE aggregator at 10Hz)
    pub sse_mining_event_tx: Option<tokio::sync::mpsc::Sender<SseMiningEvent>>,

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

    // 🌻 v2.5.0-beta: Dandelion++ for mandatory Tor-based transaction anonymity
    // All transactions route through stem→fluff phases for IP unlinkability
    // Tor is NOT opt-in - it's always enabled for transaction propagation
    pub dandelion: Option<Arc<QuantumDandelion>>,

    pub network_manager: Option<Arc<q_network::NetworkManager>>,
    pub production_peer_discovery:
        Option<Arc<tokio::sync::Mutex<q_network::real_peer_discovery::RealPeerDiscovery>>>,

    // libp2p-based zero-config peer discovery (mDNS + Gossipsub)
    pub libp2p_discovery: Option<Arc<tokio::sync::Mutex<q_network::UnifiedNetworkManager>>>,

    // libp2p network command channel (for non-blocking P2P operations from API)
    pub libp2p_command_tx: Option<tokio::sync::mpsc::UnboundedSender<q_network::NetworkCommand>>,

    // Cached libp2p peer info for fast API access (updated by event loop)
    pub libp2p_peer_info: Arc<RwLock<(String, Vec<String>)>>, // (peer_id, listen_addresses)

    // Atomic peer count for fast lock-free access
    pub libp2p_peer_count: Option<Arc<std::sync::atomic::AtomicUsize>>,

    // v10.9.27: Comprehensive network metrics — Prometheus-format /metrics
    // endpoint. Some(...) once UnifiedNetworkManager has been constructed
    // (it owns the Registry). The handler at handlers::metrics_endpoint
    // clones this Arc and serializes on each scrape.
    pub network_metrics: Option<Arc<q_network::NetworkMetrics>>,

    // v9.0.6: EMA-smoothed Decentralization Index (f64 bits stored as AtomicU64)
    pub di_ema: Arc<std::sync::atomic::AtomicU64>,

    // 💱 v0.6.1-beta: DEX DECENTRALIZATION PHASE 3 - Pool Announcement Signing
    // Node signing key for signing pool announcements broadcast to P2P network
    // Persisted to disk for consistency across restarts
    pub node_signing_key: Arc<ed25519_dalek::SigningKey>,

    // v7.2.12: Unified cryptographic engine (EternalCypher)
    pub node_cypher: Arc<q_eternal_cypher::NodeCypher>,

    // v7.2.13: Configurable admin wallet for node operator settings
    // Set via --admin-wallet CLI arg or Q_ADMIN_WALLET env var. Defaults to FOUNDER_WALLET.
    pub admin_wallet: String,

    // Stripe payment client - initialized once at startup from STRIPE_SECRET_KEY env var
    pub stripe_client: Option<stripe::Client>,

    // v1.0.2: P2P bandwidth tracking — cumulative byte counters for TUI dashboard
    pub p2p_bytes_in: Arc<std::sync::atomic::AtomicU64>,
    pub p2p_bytes_out: Arc<std::sync::atomic::AtomicU64>,

    // 🔥 Top Movers ring buffer (last 60 blocks of per-address balance deltas).
    // Each entry is a HashMap<Address, i128> for one block. Bounded to length 60
    // via push_back + pop_front in `update_tui_metrics`. Memory cost: roughly
    // 60 × ~100 addrs × (32 + 16) bytes ≈ 290 KB worst case.
    pub recent_balance_deltas:
        Arc<RwLock<std::collections::VecDeque<HashMap<Address, i128>>>>,
    // Highest block height already ingested into `recent_balance_deltas`.
    // Used to pull only the new blocks each TUI tick (typically 0-3 per tick at 1 bps).
    pub top_movers_last_ingested_height: Arc<std::sync::atomic::AtomicU64>,

    // v8.5.4: Network throttle mode (0=Conservative, 1=Normal, 2=Turbo) — set by TUI, read by sync loop
    // Conservative: 2 in-flight chunks, 200ms delay (SSD-friendly for cheap hardware)
    // Normal: half parallelism, 10ms delay (balanced)
    // Turbo: full parallelism, no delay (max sync speed, default)
    pub network_throttle_mode: Arc<std::sync::atomic::AtomicU8>,

    // SYNC MODE: Track highest block height seen from network to prevent mining during sync
    pub highest_network_height: Arc<std::sync::atomic::AtomicU64>,

    // v5.2.0: Timestamp of last peer height update (Unix secs) - for stale detection & decay
    pub last_peer_height_update: Arc<std::sync::atomic::AtomicU64>,

    // v5.2.0: Immediate sync trigger - wakes sync loop when peer announces higher height
    pub sync_trigger: Arc<tokio::sync::Notify>,

    // 🦈 SharkGod: Wake block producer immediately when a SharkGod tx is submitted
    pub sharkgod_block_wake: Option<Arc<tokio::sync::Notify>>,

    // v7.1.5: Configurable dev fee in basis points (100 = 1%, adjustable by master wallet)
    pub dev_fee_bps: Arc<std::sync::atomic::AtomicU64>,

    // v7.3.1: Node operator fee share in promille (1000 = 100%, 100 = 10%, 0 = disabled)
    // Controls what fraction of collected protocol fees (tx fees + DEX protocol fees) goes to admin_wallet
    // Remainder goes to FOUNDER_WALLET. Configurable via Q_NODE_OPERATOR_FEE_PROMILLE env or admin API.
    pub node_operator_fee_promille: Arc<std::sync::atomic::AtomicU64>,

    // v7.3.1: DEX protocol fee in basis points (out of the 30 bps LP fee)
    // Default 5 bps = 0.05% of swap amount extracted as protocol revenue
    pub dex_protocol_fee_bps: Arc<std::sync::atomic::AtomicU64>,

    // v8.1.1: Operator fee earnings tracking (atomic u64, in units of 1e-18 QUG for precision)
    // Tracks total fees earned by this node's operator wallet this session
    pub operator_fees_earned_session: Arc<std::sync::atomic::AtomicU64>,
    // Total lifetime earnings (loaded from DB on startup, persisted periodically)
    pub operator_fees_earned_total: Arc<std::sync::atomic::AtomicU64>,
    // Number of fee-generating transactions processed
    pub operator_fee_tx_count: Arc<std::sync::atomic::AtomicU64>,

    // ⚡ v0.9.66-beta: Lock-free current blockchain height for fast mining challenge generation
    // Updated atomically when blocks are produced, avoids RwLock contention on node_status.
    // SEMANTICS: max-seen height. Updated via fetch_max() whenever any block arrives via P2P
    // or batch sync. Used by sync logic to decide "am I behind?" relative to the network.
    // After Option A (v1.0.2) this can be higher than `contiguous_height_atomic` for a node
    // that has gaps in its archive history (e.g. fresh checkpoint-synced node still doing backfill).
    pub current_height_atomic: Arc<std::sync::atomic::AtomicU64>,

    // v1.0.2: Honest archive-height reporting. Reflects the highest height where every
    // block 1..=N is stored locally (contiguous storage), refreshed every 5s from
    // `get_highest_contiguous_block()`. Diverges from `current_height_atomic` whenever
    // the node has gaps. Used for API status, integrity reporting, and peer announcements
    // so other nodes don't try to fetch blocks from us that we don't actually have.
    pub contiguous_height_atomic: Arc<std::sync::atomic::AtomicU64>,

    // v8.2.9: Peak height — maximum height ever reached, never decreases
    // Prevents "rollback scare" in admin panel when node restarts and syncs back up
    pub peak_height_atomic: Arc<std::sync::atomic::AtomicU64>,

    // v10.9.19: API request counter — incremented by /api/v1/engine/pulse on every
    // call. Lets clients compute API throughput by polling pulse and computing the
    // delta. Initialized to 0 in both AppState constructors.
    pub api_requests_served: Arc<std::sync::atomic::AtomicU64>,

    // 🚀 v1.0.2-beta: HeightState cache - Eliminates binary search storm during shutdown
    // Cached height value (lock-free reads) with time-based freshness and shutdown mode
    pub height_state: q_storage::HeightState,

    // 🛑 v1.0.2-beta: Graceful shutdown broadcast channel
    // Signal handler broadcasts shutdown to all subsystems for clean termination
    pub shutdown_tx: tokio::sync::broadcast::Sender<()>,

    // 🔄 v8.5.0: Auto-update announcement forwarding channel
    // Gossipsub handler forwards raw announcement bytes to NodeAutoUpdater
    pub auto_update_tx: Option<tokio::sync::mpsc::UnboundedSender<Vec<u8>>>,

    // 🔄 v8.5.0: Auto-update state receiver (for SSE/API)
    pub auto_update_state: Option<tokio::sync::watch::Receiver<node_auto_updater::NodeUpdateState>>,

    // 🔄 v8.5.1: Runtime auto-update enabled toggle (admin-controllable via API)
    // Initialized from Q_AUTO_UPDATE env var, can be toggled via POST /api/v1/admin/update/toggle
    pub auto_update_enabled: Arc<std::sync::atomic::AtomicBool>,

    // 🚀 v8.8.2: One-time bootstrap wallet balance sync flag
    // Set to true after first successful wallet balance import from a trusted peer.
    // Once set, wallet balance import is permanently disabled (v8.5.4 safety preserved).
    // Also gates QUGUSD: captured BEFORE wallet import runs, so QUGUSD can flow in the same response.
    pub bootstrap_wallet_sync_done: Arc<std::sync::atomic::AtomicBool>,

    // 🤖 v9.3.3: AI inference active flag — when true, miners should throttle to 1 thread
    // Set by web_search_handler when Ollama LLM is streaming, cleared when done.
    // Mining challenge response includes `recommended_threads` so miners auto-throttle.
    pub ai_active: Arc<std::sync::atomic::AtomicBool>,

    // 🛡️ v10.3.1: DEX Safety Gate — blocks swaps until node is fully synced and reconciled.
    // Starts `false`, set to `true` ONLY after balance_consensus replay + apply_dex_qug_adjustments().
    // DeepSeek review: "node starts in DEX disabled / read-only mode; replay/reconciliation
    // finishes; then swap endpoint becomes available."
    // Uses AtomicBool (not RwLock) because execute_swap() is a hot path.
    pub dex_ready: Arc<std::sync::atomic::AtomicBool>,

    // 🛡️ v10.3.2: Startup sync complete — prevents ghost balance display.
    // False until authority sync (or first periodic sync) completes.
    // Balance API returns null/syncing when false to avoid showing stale 4200 QUG.
    pub startup_sync_complete: Arc<std::sync::atomic::AtomicBool>,

    // BFT-safe balance finalization engine (Bracha RB + DAG-Knight anchoring).
    // `None` on observer-only nodes; `Some` on validators.
    pub balance_finality_engine: Option<Arc<q_storage::balance_finality_engine::BalanceFinalityEngine>>,

    // 🔄 v8.5.1: Admin notification email for update alerts
    // Set via POST /api/v1/admin/update/notification-email or Q_ADMIN_NOTIFICATION_EMAIL env
    pub admin_notification_email: Arc<tokio::sync::RwLock<Option<String>>>,

    // 🔧 v1.0.4-beta: Challenge caching to prevent mining stalls
    // Ensures consistent challenge_hash across API requests for same height
    pub current_challenge: Arc<tokio::sync::RwLock<Option<CachedChallenge>>>,

    // 🔍 v0.9.67-beta: Comprehensive fork detection and automatic resolution
    // Tracks peer heights and detects backward reorgs, minority forks, network splits
    pub fork_detector: Arc<q_storage::fork_detector::ForkDetector>,

    // 🎨 v0.6.6-beta: Beautiful sync progress tracking for tqdm-style progress bar
    pub sync_start_time: Arc<std::sync::RwLock<Option<std::time::Instant>>>,
    pub sync_start_height: Arc<std::sync::atomic::AtomicU64>,

    // Mining submission queue (async processing to prevent server overload)
    // ✅ v1.0.2-beta Layer 3 FIX: Changed to bounded channel with 10,000 capacity
    // ⚡ v8.9.0: Sharded N-way parallel pipeline for 1M+ TPS
    pub mining_submission_tx: Option<tokio::sync::mpsc::Sender<MiningSubmission>>, // legacy single (unused if shards active)
    pub mining_submission_txs: Option<Arc<Vec<tokio::sync::mpsc::Sender<MiningSubmission>>>>,
    pub mining_shard_index: Option<Arc<std::sync::atomic::AtomicUsize>>,

    // 🔒 v4.1.3: Mining nonce deduplication — prevents double-reward attacks
    // Tracks (challenge_hash_prefix, nonce) pairs. Entries auto-expire when challenge rotates.
    // Uses DashMap for lock-free concurrent access from multiple mining submissions.
    pub mining_nonce_dedup: Arc<dashmap::DashMap<(u64, u64), u64>>, // (challenge_height, nonce) -> timestamp

    /// v5.1.0: Optimistic swap deduplication — prevents double-application of swap txs
    /// When a swap is executed locally, its tx.id is recorded here. When the same tx arrives
    /// in a P2P block, block state processing skips it to prevent double-credit.
    /// Key: tx_id ([u8; 32]), Value: timestamp (seconds since epoch) for expiry
    pub optimistic_applied_txs: Arc<dashmap::DashMap<[u8; 32], u64>>,

    // BREAKTHROUGH: DNS-Phantom → Connection Integration
    pub connection_manager: Option<Arc<q_network::connection_manager::ConnectionManager>>,

    // DAG State Synchronization
    pub dag_sync_manager: Option<Arc<q_network::DagSyncManager>>,

    // ZK Privacy Components - ✅ ENABLED
    pub zk_stark_system: Option<Arc<tokio::sync::Mutex<StarkSystem>>>,
    pub zk_snark_system: Option<Arc<UniversalSNARK>>,
    // ✅ Post-Quantum LatticeGuard SNARK (RLWE-based, no trusted setup)
    pub lattice_guard: Option<Arc<tokio::sync::Mutex<q_lattice_guard::LatticeGuard>>>,
    pub lattice_guard_srs: Option<Arc<q_lattice_guard::LatticeGuardSRS>>,

    // ✅ v2.4.1-beta: TemporalShield-STARK HSM-backed trustee management
    // Manages threshold secret sharing trustees with simulated HSM key storage
    pub temporal_trustee_manager: Option<Arc<tokio::sync::RwLock<trustee_manager::TrusteeManager>>>,

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

    // Phase 1: Quorum commit collector — tracks validator agreement on balance_root per height.
    // 3-of-4 validators signing the same (height, balance_root) means QUORUM VERIFIED.
    pub quorum_commit_collector: Arc<quorum_commit::QuorumCommitCollector>,
    pub production_mempool: Option<Arc<ProductionMempool>>, // HIGH-PERFORMANCE MEMPOOL FOR 200K+ TPS

    // ✅ v1.3.11-beta: TRUE DECENTRALIZED CONSENSUS SERVICE
    // Collects signatures from multiple validators (2/3+1 threshold) before finalizing
    // Without this, blocks are only locally validated without multi-party agreement
    pub consensus_service: Option<Arc<consensus_service::ConsensusService>>,
    pub reliable_broadcast: Option<Arc<ReliableBroadcast>>,
    pub quantum_vdf: Option<Arc<QuantumVDF>>,

    // PHASE 2: Parallel Block Production - Multiple producers for concurrent block creation
    // ✅ v0.9.92-beta DEADLOCK FIX: Lock-free producer pool (channel-based, zero RwLocks)
    pub block_producer_pool: Arc<crate::lockfree_producer::LockFreeProducerPool>,

    /// 📊 v1.0.72-beta: Finality Metrics - Sub-50ms latency tracking for consensus dashboard
    pub finality_metrics: Arc<crate::block_producer::FinalityMetrics>,

    /// Self-signed finality certificates for locally-produced blocks (item 8 / consensus foundation).
    /// Key = block height, value = FinalityCertificate signed by this node's Ed25519 key.
    /// Bounded ring: evicts entries for heights older than (tip - 10_000).
    pub finality_certs: Arc<std::sync::Mutex<std::collections::HashMap<u64, q_types::FinalityCertificate>>>,

    // AI Model Management - Lazy Loading with HTTP Download
    pub ai_model_manager: Option<Arc<q_ai_inference::ModelManager>>,

    // PHASE 3: DAG-Knight Consensus - Byzantine Fault-Tolerant Block Ordering
    pub consensus: Arc<RwLock<DAGKnightConsensus>>,

    // v9.3.1: Lightweight K-parameter network health gauge (always-on, no q-resonance dep)
    pub k_parameter_state: Arc<k_parameter_gauge::KParameterState>,

    // Quillon Resonance Consensus - K-Parameter Phase Analysis
    #[cfg(feature = "resonance")]
    pub k_parameter_analyzer: Option<Arc<KParameterAnalyzer>>,
    #[cfg(not(feature = "resonance"))]
    pub k_parameter_analyzer: Option<()>,

    #[cfg(feature = "resonance")]
    pub resonance_coordinator: Option<Arc<ResonanceCoordinator>>,
    #[cfg(not(feature = "resonance"))]
    pub resonance_coordinator: Option<()>,

    #[cfg(feature = "resonance")]
    pub shadow_coordinator: Option<Arc<tokio::sync::Mutex<q_resonance::ShadowModeCoordinator>>>,
    #[cfg(not(feature = "resonance"))]
    pub shadow_coordinator: Option<()>,

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

    // ============ DEX & TOKEN REGISTRY SYSTEM ============
    // Dynamic Token Registry - RocksDB-backed persistent storage for all tokens
    pub token_registry: Option<Arc<q_storage::token_registry::TokenRegistry>>,
    // Price History Manager - Time-series OHLCV candles for historical price tracking
    pub price_history: Option<Arc<q_storage::price_history::PriceHistoryManager>>,
    // v0.9.1-beta: DEX types enabled
    pub dex_manager: Option<Arc<q_dex::QuantumDexManager>>,
    pub price_bridge: Option<Arc<()>>, // Placeholder for oracle integration

    // v2.3.34-beta: Swap/Transaction History - In-memory cache + RocksDB persistence
    // Maps token_id -> Vec of swap transactions for that token
    pub swap_history: Arc<RwLock<HashMap<String, Vec<handlers::SwapHistoryRecord>>>>,

    // v2.4.0-beta: Consensus-Verified Swap Indexer - DAGKnight-verified transaction history
    // Indexes swap transactions from finalized blocks for trustless cross-node agreement
    pub swap_indexer: Arc<swap_indexer::SwapIndexer>,

    // v3.7.1-beta: Consensus-Verified Price History Indexer - Persistent price history
    // Derives prices from on-chain swap exchange rates, stored in RocksDB CF_PRICE_HISTORY
    // All nodes compute identical prices from identical blocks → Decentralized price consensus
    pub price_history_indexer: Arc<price_history_indexer::PriceHistoryIndexer>,

    // v2.3.8-beta: Volume Tracker - Rolling 24h volume per token (token_id -> (timestamp, volume))
    // Each entry is a tuple of (unix_timestamp_millis, volume_in_usd)
    pub volume_tracker: Arc<RwLock<HashMap<String, Vec<(i64, f64)>>>>,

    // v2.3.8-beta: Price Snapshot Cache - Historical prices for change calculation
    // Maps token_id -> Vec of (timestamp, price) sorted by timestamp descending
    pub price_snapshots: Arc<RwLock<HashMap<String, Vec<(i64, f64)>>>>,

    // Quillon Bank - Full Quantum Banking System with CDP
    pub quillon_bank: Arc<RwLock<QuillonBankSystem>>, // ✅ ENABLED - Real banking system

    // v2.4.0-beta: Governance Coordinator - PERSISTENT across restarts
    // Stores proposals and votes with RocksDB persistence + P2P gossipsub sync
    pub governance_coordinator: Arc<q_governance::GovernanceCoordinator>,

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

    // v8.5.5: QCREDIT Yield Vault — lock QUG, mint QCREDIT 1:1, earn tiered yield
    pub qcredit_vault: Arc<RwLock<q_vm::contracts::QCreditVault>>,

    // Quillon Bank Loan Applications - Pending loan applications with RocksDB persistence
    pub pending_loan_applications:
        Arc<RwLock<HashMap<String, crate::quillon_bank_api::LoanApplication>>>,

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

    // 🚀 v2.4.8: O(1) symbol-to-address lookup for oracle price resolution
    // DashMap for lock-free concurrent access - scales to millions of tokens
    // Key: UPPERCASE symbol (e.g., "MEME"), Value: contract address (e.g., "qnk542e85...")
    pub symbol_to_address: Arc<dashmap::DashMap<String, String>>,

    // v1.4.10: Contract event history for mint/burn/airdrop operations
    // Key: contract_address (hex string), Value: Vec of events
    pub contract_events: Arc<RwLock<HashMap<String, Vec<ContractEventRecord>>>>,

    /// v1.4.11: Commit-reveal mining protection
    /// Prevents front-running and MEV by requiring 2-phase commit/reveal for mining solutions
    pub commit_reveal_manager: Arc<mining_commit_reveal::CommitRevealManager>,

    /// v1.4.11: Stake-weighted finality manager
    /// Combines PoW confirmations with PoS attestations for hybrid security finality
    pub stake_finality_manager: Arc<stake_weighted_finality::StakeWeightedFinalityManager>,

    // Distributed VM and DEX (Horizontal Scaling)
    pub distributed_protocol: Option<Arc<q_network::DistributedProtocolManager>>,

    // OAuth2 Provider for third-party integrations
    pub oauth2_storage: Arc<oauth2_provider::OAuth2Storage>,

    // v8.1.7: OAuth2 Key Vault — Server-side encrypted signing keys for custodial transactions
    // When an OAuth2 user sends a tx, the server uses the vault key to sign automatically
    // (no mnemonic needed). Keys are AES-256-GCM encrypted with the server's node signing key.
    // Key: wallet address [u8;32], Value: encrypted Ed25519 private key bytes
    pub oauth2_key_vault: Arc<RwLock<HashMap<[u8; 32], Vec<u8>>>>,

    // v7.4.0: Peer JWT public keys for cross-node token verification
    pub peer_jwt_keys: Arc<dashmap::DashMap<String, oauth2_provider::PeerJwtKeyInfo>>,

    // v8.6.0: Distributed node operator fee — maps peer_id → operator_wallet_hex
    // Populated from PeerHeightAnnouncement gossipsub messages
    pub peer_operator_wallets: Arc<dashmap::DashMap<String, String>>,

    // AI Inference Engine - Privacy-first distributed inference with KV-cache (OLD - slow)
    pub inference_engine: Option<
        Arc<tokio::sync::Mutex<q_ai_inference::distributed_cache::DistributedInferenceWithCache>>,
    >,

    // High-performance mistral.rs engine (10-100x faster, <2s first token)
    pub mistralrs_engine: Option<Arc<q_ai_inference::MistralRsEngine>>,

    // Distributed AI Coordinator - Horizontal scaling across network nodes
    pub distributed_ai_coordinator: Option<Arc<q_network::DistributedAICoordinator>>,

    // 🔐 AI Verification System - Proof-of-inference and worker benchmarking
    pub proof_verifier: Option<Arc<q_ai_inference::ProofOfInferenceVerifier>>,
    pub benchmark_verifier: Option<Arc<q_ai_inference::WorkerBenchmarkVerifier>>,
    pub failover_manager: Option<Arc<q_network::FailoverManager>>,
    pub verification_events_tx: Option<tokio::sync::broadcast::Sender<crate::verification_api::VerificationEvent>>,

    // 🚀 TURBO SYNC - Git-Inspired 50-250x Faster Blockchain Synchronization
    pub turbo_sync: Option<Arc<q_storage::TurboSyncManager>>,

    // v1.0.2: Starship Flight Computer — central sync state machine
    pub flight_computer: Option<Arc<tokio::sync::RwLock<q_storage::FlightComputer>>>,

    // 🚀 v1.0.4-beta: PHASE 2 DAG-AWARE SYNC - 20-40x Faster with Parallel DAG Layer Fetching
    pub enable_dag_sync: bool, // Feature flag for Phase 2 (default: true)

    // 🌉 v0.9.6-beta: TURBO SYNC PEER BRIDGE - Synchronizes libp2p peers to TurboSync registry
    // Fixes: "No peers available with target height" even when peers connected via libp2p
    // TEMPORARILY DISABLED v1.0.15: Circular dependency with sync_activation module
    // pub peer_bridge: Option<Arc<q_storage::TurboSyncPeerBridge>>,

    // 🔐 AEGIS-KL MINER AUTHENTICATION - Post-Quantum Fork Protection (v0.5.7+)
    // Ensures only authorized miners with valid AEGIS-KL signatures can submit solutions
    // Prevents unauthorized forks and enforces 1% development fee at protocol level
    // TODO: Re-enable when q_mining::dev_fee is implemented
    // pub miner_auth: Option<Arc<q_mining::dev_fee::MinerAuth>>,

    // 🚀 v1.0.2-beta PHASE 1A: SAFE BATCHED SYNC - 150-250 BPS Performance
    // Expert-validated implementation with 0.0001% risk tolerance
    // Feature-flagged with --experimental-fast-sync
    pub fast_sync_enabled: bool,
    pub fast_sync_tx: Option<tokio::sync::mpsc::Sender<q_types::block::QBlock>>,
    #[cfg(not(target_os = "windows"))]
    pub fast_sync_metrics: Option<Arc<tokio::sync::Mutex<q_storage::BatchMetrics>>>,

    // ✅ v1.0.7-beta: AsyncStorageEngine - Permanent mining stall fix
    // Dedicated worker thread with micro-batching (512 blocks OR 2ms)
    // Eliminates RwLock contention and amortizes RocksDB compaction overhead
    // AI Consensus (5/5 experts): Root cause = blocking RocksDB I/O under async RwLock
    #[cfg(not(target_os = "windows"))]
    pub async_storage: Option<Arc<q_storage::AsyncStorageEngine>>,

    // ✨ v1.0.16-beta: PQC Validator Keypair - Post-Quantum Block Signing
    // Loaded from --validator-key CLI argument for PQC signature generation
    // Contains Ed25519 (classical) + Dilithium5 (post-quantum) keys
    pub validator_keypair: Option<Arc<q_types::ValidatorKeypair>>,

    // ✨ v1.0.16-beta: Validator Public Key Registry - For PQC signature verification
    // Maps NodeId → Public keys (Ed25519 + Dilithium5) for all known validators
    // Used to verify spectral signatures on incoming blocks
    pub validator_key_registry: Arc<RwLock<q_types::ValidatorKeyRegistry>>,

    // 🏛️ v3.9.5-beta: Dynamic Validator Registry for P2P decentralization
    // Tracks registered validators with stake, endpoints, and status
    // Used for: balance update verification, dynamic bootstrap peer discovery
    pub validator_registry: Arc<RwLock<q_types::validator_registry::ValidatorRegistry>>,

    // ⏰ v1.0.15-beta: Timeout-Based Sync Activation - Breaks "stuck at genesis" deadlock
    // Forces sync after timeout even when network_height=0 (no peer announcements received)
    // Solves: Node stuck at 12,923 waiting forever for gossipsub peer height announcements
    pub sync_activator: Option<Arc<crate::sync_activation::TimeoutBasedSyncActivation>>,

    // ✨ v1.4.0-beta: Recursive Proofs Service - Eliminates Weak Subjectivity
    // Post-quantum recursive SNARKs for ~10ms trustless light client bootstrap
    // New nodes can verify entire blockchain history without trusting checkpoints
    pub recursive_proofs_service: Option<Arc<crate::recursive_proofs_api::RecursiveProofsService>>,

    // ✨ v1.4.2-beta: Block-Height Activated Upgrade Manager
    // Enables safe mainnet evolution: deploy binaries anytime, features activate at height
    // Old blocks always validate with old rules (immutable history)
    pub upgrade_manager: Arc<UpgradeManager>,

    // 🔮 v1.4.2-beta: QNO (Quantum Neural Oracle) Prediction Staking
    // Persistent storage for prediction staking with P2P sync for decentralized validation
    #[cfg(not(target_os = "windows"))]
    pub qno_storage: Arc<RwLock<Option<Arc<q_storage::qno_storage::QnoStorage>>>>,

    // ⛏️ v2.2.1-beta: Stratum Mining Pool with PPLNS Rewards
    // Full-featured mining pool for external miner connectivity via Stratum V1 protocol
    // Always enabled - no feature flag required
    pub mining_pool: Option<Arc<q_mining_pool::MiningPool>>,

    // 🌐 v2.3.0-beta: Decentralized Mining Pool Coordinator
    // P2P-based mining pool with CRDT PPLNS, gossipsub coordination, and threshold payouts
    // Enables fully decentralized mining pool operation without central pool server
    pub distributed_pool_coordinator: Option<Arc<tokio::sync::RwLock<q_mining_pool::distributed::DistributedPoolCoordinator>>>,

    // 📡 v2.3.0-beta: Outbound message channel for distributed pool
    // Used to send pool messages to P2P network via gossipsub
    pub distributed_pool_outbound_tx: Option<tokio::sync::mpsc::Sender<q_mining_pool::distributed::coordinator::OutboundMessage>>,

    // 🏊 v10.0.0: Distributed PPLNS proportions for block producer coinbase distribution
    // Periodically synced from the distributed coordinator's CRDT state.
    // When present and non-empty, block producer uses these instead of local-only PPLNS.
    pub distributed_pplns_proportions: Arc<tokio::sync::RwLock<Option<Vec<(String, f64)>>>>,

    // 📊 v5.7.0: Pool hashrate history ring buffer (last 24h, sampled every 60s)
    pub pool_hashrate_history: Arc<tokio::sync::RwLock<Vec<pool_api::HashrateEntry>>>,

    // 💰 v2.4.8-beta: Dollar Cost Averaging (DCA) Storage
    // Enables users to automate recurring token purchases at configured intervals
    pub dca_storage: Option<Arc<dca_api::DcaStorage>>,

    // 🎯 v10.4.9: Limit Order Storage — price-triggered one-shot swaps
    pub limit_order_storage: Option<Arc<limit_order_api::LimitOrderStorage>>,

    // 📈 v2.5.0-beta: Perpetual Futures Storage
    // Enables leveraged long/short trading with up to 10x leverage
    pub perp_storage: Option<Arc<perpetual_api::PerpStorage>>,

    // ============ v2.4.2: TOKEN STAKING & FEE SYSTEM ============

    // 🔒 Token Fee Configurations - Fee settings per custom token contract
    // Enables reflection, burn, liquidity, and dev fees on token transfers
    pub token_fee_configs: Arc<RwLock<HashMap<String, q_storage::TokenFeeConfig>>>,

    // 🎯 Token Staking Positions - Active stakes per wallet+contract
    // Key format: "wallet_address:contract_address" (lowercase)
    pub token_staking_positions: Arc<RwLock<HashMap<String, q_storage::TokenStakePosition>>>,

    // 🔥 Token Burn Totals - Cumulative burned amounts per contract
    pub token_burn_totals: Arc<RwLock<HashMap<String, u128>>>,

    // 💎 Token Reflection Totals - Cumulative reflected amounts per contract
    pub token_reflection_totals: Arc<RwLock<HashMap<String, u128>>>,

    // 🌐 v2.4.8: Token Social Profiles - Decentralized social media links per contract
    // Key: contract_address (lowercase hex), Value: JSON-serialized social profile
    // Synced across nodes via gossipsub for decentralized token info
    pub token_social_profiles: Arc<RwLock<HashMap<String, TokenSocialProfile>>>,

    // 🚨 v3.3.3-beta: EMERGENCY PAUSE MECHANISM - Mainnet Kill Switch
    // When enabled: Block production pauses, transactions rejected, reads still work
    // Activation: POST /api/v1/admin/emergency-pause with founder signature
    // Resume: POST /api/v1/admin/emergency-resume with founder signature
    // CRITICAL: This is the last line of defense against catastrophic bugs
    pub emergency_paused: Arc<std::sync::atomic::AtomicBool>,
    pub emergency_pause_reason: Arc<RwLock<Option<String>>>,
    pub emergency_pause_timestamp: Arc<std::sync::atomic::AtomicU64>,

    // ⛏️ v9.1.4: Dynamic mining mode switch — admin can force all miners to solo/pool at runtime
    // 0 = no override (miners use their own --mode), 1 = force solo, 2 = force pool
    pub forced_mining_mode: Arc<std::sync::atomic::AtomicU8>,
    pub forced_pool_url: Arc<RwLock<Option<String>>>,

    // 📬 v3.9.1-beta: BANK MESSAGING SYSTEM - User-Bank Communication
    // Enables bidirectional messaging between loan holders and Quillon Bank
    // Messages are stored in-memory with RocksDB persistence via CF_MANIFEST
    pub bank_messages: Arc<RwLock<Vec<quillon_bank_api::BankMessage>>>,

    // 🪪 v3.9.1-beta: DECENTRALIZED IDENTITY SYSTEM - VM-Backed User Profiles
    // User identity records with KYC levels, beneficiary addresses, and death certificates
    // Enables account inheritance and estate planning on the blockchain
    pub user_identities: Arc<RwLock<Vec<quillon_bank_api::UserIdentity>>>,
    pub death_certificates: Arc<RwLock<Vec<quillon_bank_api::DeathCertificate>>>,

    // 🔐 v4.2.0-beta: VAULT RWA Token - Physical hardware wallet redemption tracking
    pub vault_redemptions: Arc<RwLock<Vec<contracts_api::VaultRedemption>>>,

    // v5.1.0: FORGE RWA Token - Physical mining machine redemption tracking
    pub forge_redemptions: Arc<RwLock<Vec<contracts_api::ForgeRedemption>>>,

    // v6.5.0: Exchange Listing RWA packages (Gold/Silver/Bronze)
    pub listing_orders: Arc<RwLock<Vec<listing_api::ListingOrder>>>,

    // v8.2.8: XLIST Crowdfunding campaigns
    pub listing_campaigns: Arc<RwLock<Vec<listing_api::ListingCampaign>>>,

    // v5.1.1: Node start time for uptime tracking in health endpoint
    pub start_time: std::time::Instant,

    // v5.1.1: Deploy admin state for rolling upgrade verification
    pub deploy_state: Arc<RwLock<deploy_admin_api::DeployState>>,

    // v7.2.0: Miner Link WebSocket relay — bridges wallet ↔ personal miner communication
    pub miner_link_registry: miner_link_api::MinerLinkRegistry,

    // v7.2.0: Bitcoin atomic swap manager (QNK ↔ BTC via HTLC)
    pub atomic_swap_manager: Option<Arc<q_bitcoin_bridge::atomic_swap::AtomicSwapManager>>,

    // v10.2.11: Bitcoin deposit bridge — receive BTC on-chain, mint wBTC
    pub deposit_bridge: Option<Arc<q_bitcoin_bridge::deposit_bridge::DepositBridge>>,

    // v9.6.5: Bitcoin Knots RPC client for wallet operations (balance, address, send)
    pub bitcoin_rpc_client: Option<Arc<bitcoin_rpc::BitcoinRpcClient>>,

    // v9.7.2: Zcash Zebra RPC client for real wallet operations (z-addresses, balance, send)
    pub zcash_rpc_client: Option<Arc<zcash_rpc::ZcashRpcClient>>,

    // v7.3.1: Multi-sig bridge validation committee (7-of-11 rotating attestations)
    pub bridge_committee: Arc<RwLock<bridge_committee::BridgeCommittee>>,

    // v9.4.0: Bridge safety controller — deposit verification, kill-switch, amount limits
    pub bridge_safety: Arc<bridge_safety::BridgeSafetyController>,

    // v9.5.0: Starship Endgame — 100% compute utilization orchestrator
    pub compute_orchestrator: Option<Arc<q_compute::orchestrator::Orchestrator>>,

    // v9.6.1: QR code payment requests for brick-and-mortar POS
    pub payment_requests: Arc<dashmap::DashMap<String, payment_request_api::PaymentRequest>>,

    // v10.2.0: Crown & Ash — Medieval grand strategy game state (on-chain WASM sim)
    pub crown_ash_state: crown_ash_api::SharedGameState,
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
        use anyhow::Context;
        use std::path::PathBuf;

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
        let (public_key, _secret_key) = aegis
            .generate_keypair()
            .map_err(|e| anyhow::anyhow!("Failed to generate development key: {:?}", e))?;

        Ok(public_key)
    }

    /// Load AEGIS-QL public key from file
    fn load_aegis_key_from_file(path: &std::path::Path) -> anyhow::Result<q_aegis_ql::PublicKey> {
        use anyhow::Context;

        let key_bytes = std::fs::read(path)
            .with_context(|| format!("Failed to read AEGIS public key from {}", path.display()))?;

        let public_key: q_aegis_ql::PublicKey =
            bincode::deserialize(&key_bytes).context("Failed to deserialize AEGIS public key")?;

        tracing::info!(
            "✅ Loaded founder AEGIS-QL public key from {}",
            path.display()
        );

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
            sync_writes: true, // CRITICAL: Enable fsync() to survive hard kills (pkill -9)
            cache_size_mb: 256,
            max_open_files: 1000,
        };

        // 🔍 v0.6.6-beta: Capture db_path before storage_config is moved
        let db_path_for_logging = storage_config.db_path.clone();

        let storage_engine = Arc::new(StorageEngine::new(storage_config).await?);

        // 🚨 v0.9.97-beta: CRITICAL DATABASE INTEGRITY CHECK
        // Unanimous AI Expert Recommendation (ChatGPT, DeepSeek, Kimi AI - 100% consensus)
        // MANDATORY on EVERY boot to prevent catastrophic data loss
        //
        // This check detects and repairs:
        // 1. Pointer-data mismatches (pointer=766, blocks=0)
        // 2. Gaps in blockchain
        // 3. Total data loss scenarios
        //
        // Performance: O(log N) - ~10 disk reads for 1M blocks, ~2 seconds for 10K blocks
        #[cfg(not(target_os = "windows"))]
        {
        tracing::info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        tracing::info!("🔍 v0.9.97-beta: COMPREHENSIVE DATABASE INTEGRITY CHECK");
        tracing::info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        tracing::info!("   AI Expert Consensus: MANDATORY safety check");
        tracing::info!("   Prevents: Data loss, pointer corruption, blockchain gaps");

        use q_storage::integrity::IntegrityChecker;
        use std::path::PathBuf;
        let hot_db_path = PathBuf::from(&db_path_for_logging).join("hot");
        let checker = IntegrityChecker::new(hot_db_path);

        match checker.check().await {
            Ok(report) => {
                if report.is_critical() {
                    // Catastrophic corruption - REFUSE TO START
                    tracing::error!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                    tracing::error!("💀 CRITICAL DATABASE CORRUPTION DETECTED!");
                    tracing::error!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                    tracing::error!("   Type: {:?}", report.corruption_type);
                    tracing::error!("   Pointer: {}", report.pointer_height);
                    tracing::error!("   Actual: {}", report.highest_contiguous);
                    tracing::error!("");
                    tracing::error!("🛠️  MANUAL INTERVENTION REQUIRED:");
                    tracing::error!("   1. Check disk health: smartctl -a /dev/sdX");
                    tracing::error!("   2. Review logs for crash/OOM events");
                    tracing::error!("   3. Restore from backup if available");
                    tracing::error!(
                        "   4. Or reset (testnet only): repair-database --reset-pointer=0"
                    );
                    tracing::error!("");
                    tracing::error!("   SERVICE WILL NOT START");
                    tracing::error!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

                    return Err(anyhow::anyhow!(
                        "Critical database corruption detected - refusing to start (see logs above)"
                    ));
                } else if report.needs_repair() {
                    // Minor corruption - AUTO-REPAIR
                    tracing::warn!("⚠️  Minor corruption detected - attempting auto-repair...");
                    tracing::warn!("   Type: {:?}", report.corruption_type);

                    match checker.repair(&report).await {
                        Ok(()) => {
                            tracing::info!("✅ Auto-repair completed successfully");
                            tracing::info!(
                                "   Database repaired: pointer {} → {}",
                                report.pointer_height,
                                report.highest_contiguous
                            );
                            tracing::info!(
                                "   Node will start from height {}",
                                report.highest_contiguous
                            );
                        }
                        Err(e) => {
                            tracing::error!("❌ Auto-repair failed: {}", e);
                            return Err(anyhow::anyhow!("Database repair failed: {}", e));
                        }
                    }
                } else {
                    // Database healthy
                    tracing::info!(
                        "✅ Database integrity verified: {} blocks",
                        report.highest_contiguous
                    );
                    tracing::info!("   No corruption detected");
                }
            }
            Err(e) => {
                tracing::error!("💀 Database integrity check failed: {}", e);
                tracing::error!("   SERVICE WILL NOT START");
                return Err(anyhow::anyhow!("Database integrity check failed: {}", e));
            }
        }

        tracing::info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        } // end #[cfg(not(target_os = "windows"))] integrity check block

        // ✅ HEIGHT RECOVERY FIX (v0.8.5-beta): Repair height pointer before loading height
        // NOTE: v0.9.97-beta comprehensive check above supersedes this, but kept for compatibility
        tracing::info!(
            "🔧 [v0.8.5] Running legacy height pointer check (superseded by v0.9.97)..."
        );
        match storage_engine.repair_height_pointer().await {
            Ok(repaired_height) => {
                tracing::info!(
                    "✅ [v0.8.5] Legacy check passed: height {}",
                    repaired_height
                );
            }
            Err(e) => {
                tracing::warn!("⚠️  [v0.8.5] Legacy check error (non-critical): {}", e);
            }
        }

        // ✅ CRITICAL FIX (v0.5.18-beta): Load initial blockchain height from RocksDB
        // This fixes the mainnet-critical bug where nodes would restart at height 0 instead of 145,000+
        // 🔍 v0.6.6-beta: Added comprehensive diagnostic logging to catch height recovery failures
        tracing::info!("🔍 [v0.6.6] Attempting to load blockchain height from database...");
        tracing::info!("🔍 [v0.6.6] Database path: {:?}", db_path_for_logging);

        let initial_height = match storage_engine.get_highest_contiguous_block().await {
            Ok(height) if height > 0 => {
                tracing::info!(
                    "✅ [v0.6.6] SUCCESS: Loaded blockchain state from database: height {}",
                    height
                );
                tracing::info!("✅ [v0.6.6] Node will resume from block {}", height);
                height
            }
            Ok(zero_height) => {
                tracing::warn!(
                    "⚠️  [v0.6.6] Database returned height {} - no blocks found!",
                    zero_height
                );
                tracing::warn!("⚠️  [v0.6.6] Database path: {:?}", db_path_for_logging);
                tracing::warn!("⚠️  [v0.6.6] Starting fresh blockchain at height 0");
                tracing::warn!(
                    "⚠️  [v0.6.6] If you expected blocks to be present, CHECK DATABASE INTEGRITY!"
                );
                0
            }
            Err(e) => {
                tracing::error!(
                    "🚨 [v0.6.6] CRITICAL ERROR: Failed to load blockchain height from database!"
                );
                tracing::error!("🚨 [v0.6.6] Error: {}", e);
                tracing::error!("🚨 [v0.6.6] Database path: {:?}", db_path_for_logging);
                tracing::error!(
                    "🚨 [v0.6.6] This will cause blockchain reset! Starting at height 0"
                );
                tracing::error!("🚨 [v0.6.6] Node will re-sync from peers (slow but safe)");
                0
            }
        };

        // 🔍 v0.6.6-beta: Verify loaded height makes sense
        tracing::info!("🔍 [v0.6.6] Final initial_height: {}", initial_height);
        if initial_height == 0 {
            tracing::warn!("⚠️  [v0.6.6] Blockchain starting at height 0");
            tracing::warn!(
                "⚠️  [v0.6.6] If this is unexpected, database may be empty or corrupted"
            );
        } else {
            tracing::info!(
                "✅ [v0.6.6] Blockchain initialized at height {}",
                initial_height
            );
        }

        let node_status = NodeStatus {
            node_id,
            current_round: 0,
            current_height: initial_height, // ✅ Load from database instead of hardcoded 0
            connected_peers: 0,
            tx_pool_size: 0,
            is_validator,
            uptime: std::time::Duration::from_secs(0),
        };

        // v7.2.4: Genesis filter — purge testnet BALANCES only (never delete blocks!)
        // The old code deleted blocks with timestamp < GENESIS_TIMESTAMP on every restart,
        // causing height drops from 295K to 77K. Fixed to only purge stale balances/state.
        match storage_engine.purge_pre_genesis_balances_only().await {
            Ok(true) => {
                tracing::info!("🧹 [GENESIS FILTER] Stale testnet balances purged — blocks preserved");
            }
            Ok(false) => {
                tracing::info!("✅ [GENESIS FILTER] No stale testnet data found — clean state");
            }
            Err(e) => {
                tracing::warn!("⚠️ [GENESIS FILTER] Error during balance purge: {} — continuing", e);
            }
        }

        // v7.2.6: Second purge — clears testnet balances re-synced via P2P
        match storage_engine.purge_testnet_balances_v726().await {
            Ok(true) => {
                tracing::info!("🧹 [GENESIS FILTER v7.2.6] Testnet balances purged from ALL nodes");
            }
            Ok(false) => {
                tracing::info!("✅ [GENESIS FILTER v7.2.6] Already purged — clean state");
            }
            Err(e) => {
                tracing::warn!("⚠️ [GENESIS FILTER v7.2.6] Error: {} — continuing", e);
            }
        }

        // v7.2.12: Third purge — nuclear option. Delete ALL wallet balances and rebuild from chain.
        // Previous purges (v7.2.5, v7.2.6) ran once but P2P re-introduced testnet balances.
        // This time we purge AND rebuild from on-chain data only.
        let wallet_purge_ran = match storage_engine.purge_testnet_wallets_v7212().await {
            Ok(true) => {
                tracing::info!("🧹 [GENESIS FILTER v7.2.12] Purged ALL wallet balances — will rebuild from chain");
                true
            }
            Ok(false) => {
                tracing::info!("✅ [GENESIS FILTER v7.2.12] Already purged — skip rebuild");
                false
            }
            Err(e) => {
                tracing::warn!("⚠️ [GENESIS FILTER v7.2.12] Purge error: {} — continuing", e);
                false
            }
        };

        // Load or rebuild wallet balances
        let mut wallet_balances = HashMap::new();
        let mut total_supply;

        if wallet_purge_ran {
            // Rebuild balances from chain (only mainnet blocks)
            match storage_engine.rebuild_balances_from_chain().await {
                Ok((rebuilt_balances, rebuilt_supply)) => {
                    tracing::info!(
                        "✅ [REBUILD v7.2.12] Rebuilt {} wallet balances, supply={} QUG",
                        rebuilt_balances.len(),
                        rebuilt_supply / 1_000_000_000_000_000_000_000_000u128
                    );
                    wallet_balances = rebuilt_balances;
                    total_supply = rebuilt_supply;
                }
                Err(e) => {
                    tracing::warn!("⚠️ [REBUILD v7.2.12] Failed: {} — starting with empty balances", e);
                    total_supply = 0;
                }
            }
        } else {
            // Normal path: load from storage
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

            // 💎 Load total minted supply from storage (max supply enforcement)
            total_supply = match storage_engine.load_total_supply().await {
                Ok(supply) => {
                    tracing::info!(
                        "💎 Loaded total minted supply: {} QUG (max: 21M QUG)",
                        supply / 1_000_000_000_000_000_000_000_000u128
                    );
                    supply
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to load total supply from storage: {}, starting from 0",
                        e
                    );
                    0
                }
            };
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

        // v8.5.6: ONE-TIME QUGUSD ghost cleanup (replaces v8.5.3 every-restart purge).
        // Root causes FIXED: (1) restore migration flag persists, (2) state_sync_api rejects QUGUSD from P2P.
        // After one-time purge, legitimate QUGUSD from swaps/mints/loans persists across restarts.
        {
            const QUGUSD_PURGE_FLAG: &[u8] = b"migration_qugusd_ghost_purge_v856_done";
            if !storage_engine.has_migration_flag(QUGUSD_PURGE_FLAG).await {
                use q_types::QUGUSD_TOKEN_ADDRESS;
                let qugusd_entries: Vec<([u8; 32], [u8; 32])> = token_balances
                    .iter()
                    .filter(|((_wallet, token_addr), _)| *token_addr == QUGUSD_TOKEN_ADDRESS)
                    .map(|((wallet, token), _)| (*wallet, *token))
                    .collect();
                let qugusd_count = qugusd_entries.len();
                if qugusd_count > 0 {
                    for (wallet, token) in &qugusd_entries {
                        token_balances.remove(&(*wallet, *token));
                    }
                    for (wallet, token) in &qugusd_entries {
                        let _ = storage_engine.delete_token_balance(wallet, token).await;
                    }
                    tracing::warn!(
                        "🧹 [v8.5.6] ONE-TIME purge: removed {} ghost QUGUSD entries (won't run again)",
                        qugusd_count
                    );
                }
                let _ = storage_engine.set_migration_flag(QUGUSD_PURGE_FLAG).await;
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
        // v7.1.6: Filter out pre-genesis (testnet) pools
        let pool_genesis_ts = q_storage::emission_controller::GENESIS_TIMESTAMP;
        let mut liquidity_pools_map = HashMap::new();
        match storage_engine.load_liquidity_pools().await {
            Ok(persisted_pools) => {
                let mut filtered_pool_count = 0u64;
                for (pool_id, pool_bytes) in persisted_pools {
                    match serde_json::from_slice::<LiquidityPool>(&pool_bytes) {
                        Ok(pool) => {
                            let pool_ts = pool.created_at.timestamp() as u64;
                            if pool_ts > 0 && pool_ts < pool_genesis_ts {
                                filtered_pool_count += 1;
                                let _ = storage_engine.delete_liquidity_pool(&pool_id).await;
                                continue;
                            }
                            // v10.2.2: Purge dust/broken pools on startup
                            const STARTUP_MIN_POOL_RESERVE: u128 = 10_000_000_000_000_000_000_000; // 10^22
                            if pool.reserve0 < STARTUP_MIN_POOL_RESERVE || pool.reserve1 < STARTUP_MIN_POOL_RESERVE {
                                filtered_pool_count += 1;
                                tracing::info!(
                                    "🧹 [POOL CLEANUP] Purging dust pool {} ({}/{}) — r0={}, r1={}",
                                    pool_id, pool.token0, pool.token1, pool.reserve0, pool.reserve1
                                );
                                let _ = storage_engine.delete_liquidity_pool(&pool_id).await;
                                continue;
                            }
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
                if filtered_pool_count > 0 {
                    tracing::info!(
                        "🧹 [GENESIS FILTER] Purged {} pre-genesis/dust pools from storage",
                        filtered_pool_count
                    );
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

        // v3.9.3-beta: SKIP loading all transactions on startup
        // Loading 30M+ transactions wastes memory and slows startup
        // Transactions remain in storage and can be queried on-demand
        tracing::info!("💳 Historical transactions available in storage (not loaded into memory)");

        // v7.1.2: Purge old testnet contracts when starting fresh mainnet
        // RocksDB CF_MANIFEST stores contracts from ALL networks without filtering.
        // When a fresh data-mainnet2026 directory is created, old testnet contracts
        // contaminate the contract list (1778+ old tokens appear).
        if initial_height == 0 {
            let network_id_str = std::env::var("Q_NETWORK_ID").unwrap_or_default();
            if network_id_str.contains("mainnet") {
                tracing::info!("🧹 [MAINNET SAFETY] Fresh mainnet detected (height=0) - purging old testnet contracts and pools");
                if let Err(e) = storage_engine.purge_dex_and_contracts().await {
                    tracing::warn!("⚠️  Failed to purge old contracts: {} (non-fatal)", e);
                } else {
                    tracing::info!("✅ [MAINNET SAFETY] Old testnet contracts purged successfully");
                }
            }
        }

        // Initialize real-time streaming
        let event_broadcaster = Arc::new(EventBroadcaster::new());
        let event_emitter = Arc::new(HighPerformanceEmitter::new(event_broadcaster.clone()));

        // Initialize VM and Smart Contract system with Orobit integration
        // IMPORTANT: Create ecosystem FIRST with storage, then pass to registry
        let orobit_ecosystem = Arc::new(
            OrobitSmartContractEcosystem::new_with_storage(Some(storage_engine.clone())).await?,
        );
        let contract_registry = Arc::new(ContractRegistry::new_with_ecosystem(
            orobit_ecosystem.clone(),
        ));

        // NOTE: Token balances are now loaded from persistent storage above
        // No need to restore from deployed contracts - persistence handles it

        // v7.1.7: Purge token balances for pre-genesis (testnet) contracts
        // orobit_ecosystem already filtered pre-genesis contracts at this point.
        // Remove token balances whose contract was purged (testnet orphans).
        // v7.2.12: Also delete from RocksDB so stablecoin_api won't resurrect them.
        {
            use q_types::{QUG_TOKEN_ADDRESS, QUGUSD_TOKEN_ADDRESS};
            let deployed = orobit_ecosystem.deployed_contracts.read().await;
            let before_count = token_balances.len();

            // Collect entries to remove BEFORE retain, for RocksDB cleanup
            let entries_to_remove: Vec<([u8; 32], [u8; 32])> = token_balances
                .iter()
                .filter(|((_wallet_addr, token_addr), _balance)| {
                    if *token_addr == QUG_TOKEN_ADDRESS || *token_addr == QUGUSD_TOKEN_ADDRESS {
                        return false; // keep native tokens
                    }
                    let ca = q_vm::contracts::orobit_smart_contracts::ContractAddress(*token_addr);
                    !deployed.contains_key(&ca)
                })
                .map(|((wallet_addr, token_addr), _)| (*wallet_addr, *token_addr))
                .collect();

            token_balances.retain(|(_wallet_addr, token_addr), _balance| {
                // Always keep native tokens
                if *token_addr == QUG_TOKEN_ADDRESS || *token_addr == QUGUSD_TOKEN_ADDRESS {
                    return true;
                }
                // Keep only if contract exists in post-genesis ecosystem
                let ca = q_vm::contracts::orobit_smart_contracts::ContractAddress(*token_addr);
                deployed.contains_key(&ca)
            });

            let removed = before_count - token_balances.len();

            // Delete from RocksDB so they don't reappear after restart
            let mut rocksdb_deleted = 0usize;
            for (wallet_addr, token_addr) in &entries_to_remove {
                if storage_engine.delete_token_balance(wallet_addr, token_addr).await.is_ok() {
                    rocksdb_deleted += 1;
                }
            }

            if removed > 0 {
                tracing::info!(
                    "🧹 [GENESIS FILTER] Purged {} testnet token balances (memory={}, RocksDB={}), {} remaining",
                    removed, removed, rocksdb_deleted, token_balances.len()
                );
            }
            drop(deployed);
        }

        // Initialize Quillon Bank - Full Quantum Banking System with CDP
        let plugin_manager = Arc::new(PluginManager::new());
        let quillon_bank_system =
            QuillonBankSystem::new(node_id, q_types::Phase::Phase1, plugin_manager.clone()).await?;
        quillon_bank_system.initialize().await?;
        let quillon_bank = Arc::new(RwLock::new(quillon_bank_system));
        tracing::info!("🏦 Quillon Bank initialized - CDP and quantum banking ready");

        // v2.4.0-beta: Initialize Governance Coordinator with RocksDB persistence
        let governance_coordinator = Arc::new(
            q_governance::GovernanceCoordinator::with_storage(storage_engine.clone()).await
        );
        tracing::info!("🏛️ Governance Coordinator initialized with RocksDB persistence");

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
                tracing::info!(
                    "✅ Founder wallet verified: qnk{}",
                    aegis_auth_middleware::FOUNDER_WALLET
                );
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
                    Ok(mut persisted_vault) => {
                        tracing::info!(
                            "💰 Loaded CollateralVault from storage: locked_qug={}, minted_qugusd={}, old_price=${}",
                            persisted_vault.total_qug_locked,
                            persisted_vault.total_qugusd_minted,
                            persisted_vault.qug_price_usd
                        );
                        // v4.0.4: Keep persisted vault price as-is (was updated from AMM during trading).
                        // v8.0.1: Migrate from old $42.50 default to $3000.00 default.
                        if persisted_vault.qug_price_usd <= 0.0 || persisted_vault.qug_price_usd < 100.0 {
                            tracing::warn!(
                                "💱 Vault QUG price was outdated/invalid (${:.6}), migrating to $3000.00",
                                persisted_vault.qug_price_usd
                            );
                            persisted_vault.qug_price_usd = 3000.00;
                        } else {
                            tracing::info!(
                                "💱 Loaded vault QUG price: ${:.4} (preserved from last session)",
                                persisted_vault.qug_price_usd
                            );
                        }

                        // v2.4.0: CRITICAL FIX - Detect and fix u64 underflow corruption
                        // v3.0.4: Updated to u128 for 24-decimal precision
                        // If total_qugusd_minted is impossibly large, this indicates corruption.
                        // Reset total_qugusd_minted to the actual sum of minted_qugusd values.
                        const IMPOSSIBLY_LARGE_SUPPLY: u128 = 1_000_000_000_000_000_000_000_000_000_000; // 10 billion with 24 decimals
                        if persisted_vault.total_qugusd_minted > IMPOSSIBLY_LARGE_SUPPLY {
                            let actual_sum: u128 = persisted_vault.minted_qugusd.values().sum();
                            tracing::error!(
                                "🚨 CORRUPTION DETECTED: total_qugusd_minted={} is impossibly large",
                                persisted_vault.total_qugusd_minted
                            );
                            tracing::warn!(
                                "🔧 FIXING: Resetting total_qugusd_minted from {} to actual sum {}",
                                persisted_vault.total_qugusd_minted,
                                actual_sum
                            );
                            persisted_vault.total_qugusd_minted = actual_sum;

                            // Also check and fix total_qug_locked if corrupted
                            let actual_qug_sum: u128 = persisted_vault.locked_qug.values().sum();
                            if persisted_vault.total_qug_locked > IMPOSSIBLY_LARGE_SUPPLY {
                                tracing::warn!(
                                    "🔧 FIXING: Resetting total_qug_locked from {} to actual sum {}",
                                    persisted_vault.total_qug_locked,
                                    actual_qug_sum
                                );
                                persisted_vault.total_qug_locked = actual_qug_sum;
                            }

                            tracing::info!("✅ Vault corruption fixed - totals now match actual balances");
                        }

                        // v8.1.2: REMOVED unconditional vault wipe (was v7.2.13)
                        // BUG FIX: The old code destroyed ALL CDP positions on every restart!
                        // Users who minted QUGUSD via collateral lost their positions.
                        if !persisted_vault.minted_qugusd.is_empty() {
                            let count = persisted_vault.minted_qugusd.len();
                            let total: u128 = persisted_vault.minted_qugusd.values().sum();
                            tracing::info!(
                                "✅ [v8.1.2] Preserved {} minted_qugusd entries (total={:.2} QUGUSD) in CollateralVault",
                                count, total as f64 / 1e24
                            );
                        }

                        Arc::new(RwLock::new(persisted_vault))
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Failed to deserialize CollateralVault: {}, creating new vault",
                            e
                        );
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
        // v4.0.7: Initialize vault QUG price from pool reserves on startup
        // Ensures price reflects actual market state even if vault persistence was missed
        {
            let mut vault_w = collateral_vault.write().await;
            let current_price = vault_w.qug_price_usd;
            let mut pool_price: f64 = 0.0;
            for p in liquidity_pools_map.values() {
                let t0 = p.token0.to_uppercase();
                let t1 = p.token1.to_uppercase();
                let t0_is_qug = t0 == "QUG" || t0 == "NATIVE-QUG"
                    || t0 == hex::encode([0u8; 32]).to_uppercase();
                let t1_is_qug = t1 == "QUG" || t1 == "NATIVE-QUG"
                    || t1 == hex::encode([0u8; 32]).to_uppercase();
                let qugusd_hex = hex::encode(q_types::QUGUSD_TOKEN_ADDRESS).to_uppercase();
                let t0_is_qugusd = t0 == "QUGUSD" || t0 == qugusd_hex
                    || t0 == format!("QNK{}", qugusd_hex);
                let t1_is_qugusd = t1 == "QUGUSD" || t1 == qugusd_hex
                    || t1 == format!("QNK{}", qugusd_hex);
                if (t0_is_qug && t1_is_qugusd) || (t0_is_qugusd && t1_is_qug) {
                    let (qug_r, usd_r) = if t0_is_qug {
                        (p.reserve0 as f64, p.reserve1 as f64)
                    } else {
                        (p.reserve1 as f64, p.reserve0 as f64)
                    };
                    if qug_r > 0.0 {
                        pool_price = usd_r / qug_r;
                    }
                    break;
                }
            }
            // v8.5.9: Force pool reserves to match $3000 target price on every startup.
            // Ghost trades from the supply inflation bug contaminated reserves — rather than
            // deleting the pool (which re-bootstraps with stale data), we directly correct
            // the reserve ratio to the authoritative oracle price.
            {
                let target_price = 3000.0_f64;
                if pool_price > 0.0 && (pool_price < target_price * 0.9 || pool_price > target_price * 1.1) {
                    // Pool ratio drifted >10% from target — reset reserves
                    let pool_id = "pool-qug-qugusd-bootstrap".to_string();
                    if let Some(mut pool_ref) = liquidity_pools_map.get_mut(&pool_id) {
                        // DashMap RefMut derefs directly to LiquidityPool
                        let qug_reserve_f64 = pool_ref.reserve0 as f64 / 1e24;
                        let new_qugusd_reserve = (qug_reserve_f64 * target_price * 1e24) as u128;
                        tracing::warn!(
                            "💱 [v8.5.9] Pool price ${:.2} drifted from target ${:.0} — resetting reserves (QUG={:.2}, QUGUSD: {:.0} → {:.0})",
                            pool_price, target_price, qug_reserve_f64,
                            pool_ref.reserve1 as f64 / 1e24, new_qugusd_reserve as f64 / 1e24
                        );
                        pool_ref.reserve1 = new_qugusd_reserve;
                        pool_ref.lp_token_supply = ((pool_ref.reserve0 as f64 * new_qugusd_reserve as f64).sqrt()) as u128;
                        // Serialize and persist
                        if let Ok(data) = serde_json::to_vec(&*pool_ref) {
                            let _ = storage_engine.save_liquidity_pool(&pool_id, &data).await;
                        }
                        pool_price = target_price;
                    }
                    vault_w.qug_price_usd = target_price;
                    vault_w.last_price_update = chrono::Utc::now().timestamp();
                }
            }
            if pool_price > 0.0 && pool_price < 1_000_000.0 {
                vault_w.qug_price_usd = pool_price;
                vault_w.last_price_update = chrono::Utc::now().timestamp();
                tracing::info!(
                    "💱 [STARTUP v4.0.7] Set vault QUG price from pool reserves: ${:.4} (was ${:.4})",
                    pool_price, current_price
                );
            } else {
                tracing::info!(
                    "💱 [STARTUP v4.0.7] No QUG/QUGUSD pool found, keeping vault price at ${:.4}",
                    current_price
                );

                // v4.0.7: Auto-create QUG/QUGUSD pool so AMM price discovery works
                // Without this pool, all QUG<->QUGUSD swaps use oracle (static price)
                let vault_price = current_price;
                let bootstrap_qug: u128 = 10_000 * 1_000_000_000_000_000_000_000_000u128; // 10k QUG in 24-decimal
                let bootstrap_qugusd: u128 = (10_000.0 * vault_price * 1e24) as u128;
                let pool_id = "pool-qug-qugusd-bootstrap".to_string();
                let pool = LiquidityPool {
                    pool_id: pool_id.clone(),
                    token0: "QUG".to_string(),
                    token1: "QUGUSD".to_string(),
                    reserve0: bootstrap_qug,
                    reserve1: bootstrap_qugusd,
                    provider: [0u8; 32],
                    created_at: chrono::Utc::now(),
                    lp_token_supply: ((bootstrap_qug as f64 * bootstrap_qugusd as f64).sqrt()) as u128,
                    token0_decimals: 24,
                    token1_decimals: 24,
                };
                liquidity_pools_map.insert(pool_id.clone(), pool.clone());
                if let Ok(pool_bytes) = serde_json::to_vec(&pool) {
                    if let Err(e) = storage_engine.save_liquidity_pool(&pool_id, &pool_bytes).await {
                        tracing::warn!("⚠️ Failed to persist bootstrap QUG/QUGUSD pool: {}", e);
                    }
                }
                tracing::info!(
                    "🏊 [STARTUP v4.0.7] Created bootstrap QUG/QUGUSD pool: 10,000 QUG / {:.0} QUGUSD @ ${:.2}/QUG",
                    bootstrap_qugusd as f64 / 1e24, vault_price
                );
            }
        }
        tracing::info!("💰 CollateralVault initialized - QUG/QUGUSD stablecoin system ready");

        // v8.2.7: Bootstrap bridge pools with LIVE oracle prices (CoinGecko/Binance)
        {
            let vault_r = collateral_vault.read().await;
            let qug_price = vault_r.qug_price_usd;
            drop(vault_r);
            let bank_r = quillon_bank.read().await;
            let oracle_ref = bank_r.oracle_integration.as_ref();
            bootstrap_bridge_pools(&mut liquidity_pools_map, &storage_engine, qug_price, Some(oracle_ref)).await;
            drop(bank_r);
        }

        // v8.5.5: Initialize QCREDIT Yield Vault from storage or create new
        let qcredit_vault = match storage_engine.load_qcredit_vault().await {
            Ok(Some(vault_bytes)) => {
                match serde_json::from_slice::<q_vm::contracts::QCreditVault>(&vault_bytes) {
                    Ok(persisted) => {
                        tracing::info!(
                            "💳 Loaded QCREDIT vault: total_locked={:.2}, positions={}, reserve={:.2}",
                            persisted.total_locked as f64 / 1e24,
                            persisted.positions.values().map(|v| v.len()).sum::<usize>(),
                            persisted.protocol_reserve as f64 / 1e24,
                        );
                        Arc::new(RwLock::new(persisted))
                    }
                    Err(e) => {
                        tracing::warn!("Failed to deserialize QCREDIT vault: {}, creating new", e);
                        Arc::new(RwLock::new(q_vm::contracts::QCreditVault::new()))
                    }
                }
            }
            Ok(None) => {
                tracing::info!("💳 No persisted QCREDIT vault found, creating new with seed reserve");
                let mut vault = q_vm::contracts::QCreditVault::new();
                // v9.5.1: Seed protocol reserve with 100K QUG to fund yield payouts
                let seed_reserve: u128 = 100_000 * 1_000_000_000_000_000_000_000_000u128;
                vault.fund_reserve(seed_reserve);
                tracing::info!("💳 Seeded QCREDIT protocol reserve with 100,000 QUG");
                Arc::new(RwLock::new(vault))
            }
            Err(e) => {
                tracing::warn!("Failed to load QCREDIT vault: {}, creating new with seed reserve", e);
                let mut vault = q_vm::contracts::QCreditVault::new();
                let seed_reserve: u128 = 100_000 * 1_000_000_000_000_000_000_000_000u128;
                vault.fund_reserve(seed_reserve);
                Arc::new(RwLock::new(vault))
            }
        };
        tracing::info!("💳 QCREDIT Yield Vault initialized");

        // v8.5.5: Bootstrap QUG/QCREDIT pool at 1:1 ratio
        {
            let qcredit_pool_id = "pool-qug-qcredit-bootstrap".to_string();
            if !liquidity_pools_map.contains_key(&qcredit_pool_id) {
                let bootstrap_amount: u128 = 10_000 * 1_000_000_000_000_000_000_000_000u128;
                let pool = LiquidityPool {
                    pool_id: qcredit_pool_id.clone(),
                    token0: format!("qnk{}", hex::encode(q_types::QUG_TOKEN_ADDRESS)),
                    token1: format!("qnk{}", hex::encode(q_types::QCREDIT_TOKEN_ADDRESS)),
                    reserve0: bootstrap_amount,
                    reserve1: bootstrap_amount,
                    provider: [0u8; 32],
                    created_at: chrono::Utc::now(),
                    lp_token_supply: bootstrap_amount,
                    token0_decimals: 24,
                    token1_decimals: 24,
                };
                liquidity_pools_map.insert(qcredit_pool_id.clone(), pool.clone());
                if let Ok(pool_bytes) = serde_json::to_vec(&pool) {
                    let _ = storage_engine.save_liquidity_pool(&qcredit_pool_id, &pool_bytes).await;
                }
                tracing::info!("🏊 [v8.5.5] Created bootstrap QUG/QCREDIT pool: 10K/10K @ 1:1");
            }
        }

        // Load existing loan applications from persistent storage
        let mut pending_loan_applications_map = HashMap::new();
        match storage_engine.load_loan_applications().await {
            Ok(persisted_loans) => {
                for (loan_id, loan_bytes) in persisted_loans {
                    match bincode::deserialize::<crate::quillon_bank_api::LoanApplication>(
                        &loan_bytes,
                    ) {
                        Ok(loan) => {
                            pending_loan_applications_map.insert(loan_id.clone(), loan);
                        }
                        Err(e) => {
                            tracing::warn!(
                                "Failed to deserialize loan application {}: {}, skipping",
                                loan_id,
                                e
                            );
                        }
                    }
                }
                tracing::info!(
                    "🏦 Loaded {} loan applications from persistent storage",
                    pending_loan_applications_map.len()
                );
            }
            Err(e) => {
                tracing::warn!(
                    "Failed to load loan applications from storage: {}, starting with empty loans",
                    e
                );
            }
        }

        // Initialize Adaptive Block Rewards System (for non-production new() path)
        let genesis_timestamp = q_storage::balance_consensus::active_genesis_timestamp();
        let dev_wallet = crate::aegis_auth_middleware::FOUNDER_WALLET.to_string();
        let balance_consensus_engine = Arc::new(q_storage::BalanceConsensusEngine::new(
            genesis_timestamp,
            dev_wallet,
        ));
        tracing::info!("✅ Adaptive Block Rewards initialized (new() path, genesis: {})", genesis_timestamp);

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

            // ✅ v1.0.91-beta: Initialize nonce tracker for replay attack prevention
            nonce_tracker: Arc::new(transaction_utils::NonceTracker::new()),
            // ✅ v9.7.0: Cross-block tx dedup cache
            applied_tx_dedup: Arc::new(dashmap::DashMap::new()),

            balance_consensus_engine: balance_consensus_engine.clone(),
            event_broadcaster,
            event_emitter,

            // Faucet system with rate limiting and abuse protection
            // v7.0.0: faucet_state removed

            // 🔒 MAX SUPPLY ENFORCEMENT - Initialize supply tracking
            total_minted_supply: Arc::new(RwLock::new(total_supply)), // Loaded from storage on startup
            supply_consensus_state: Arc::new(RwLock::new(SupplyConsensusState::default())),
            mining_statistics: Some(Arc::new(RwLock::new(MiningStatistics::default()))),

            // 💓 MINING HEARTBEAT MONITORING (v0.8.9-beta) - Detect mining stalls
            last_mining_solution_time: Arc::new(std::sync::atomic::AtomicU64::new(
                chrono::Utc::now().timestamp() as u64,
            )),
            mining_is_healthy: Arc::new(std::sync::atomic::AtomicBool::new(true)),

            // Quantum Privacy Mixer State
            mixing_requests: Arc::new(RwLock::new(HashMap::new())),
            quantum_mixer: {
                // Create entropy pool for mixer initialization
                match q_quantum_mixing::quantum_entropy::QuantumEntropyPool::new().await {
                    Ok(entropy_pool) => {
                        let entropy_arc = Arc::new(entropy_pool);
                        match QuantumMixingEngine::new(entropy_arc).await {
                            Ok(mixer) => {
                                tracing::info!(
                                    "✅ Quantum Mixing Engine initialized - Privacy mixer ready"
                                );
                                Some(Arc::new(mixer))
                            }
                            Err(e) => {
                                tracing::warn!("⚠️ Quantum Mixing Engine initialization failed: {}, mixer disabled", e);
                                None
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            "⚠️ Quantum Entropy Pool initialization failed: {}, mixer disabled",
                            e
                        );
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
                                tracing::info!(
                                    "✅ Quantum ZK Proof Engine initialized - ZK proofs ready"
                                );
                                Some(Arc::new(prover))
                            }
                            Err(e) => {
                                tracing::warn!(
                                    "⚠️ ZK Proof Engine initialization failed: {}, proofs disabled",
                                    e
                                );
                                None
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            "⚠️ Quantum Entropy Pool initialization failed: {}, ZK proofs disabled",
                            e
                        );
                        None
                    }
                }
            },

            // Network components
            bitcoin_bridge: None,
            dns_phantom: None,
            bep44_discovery: None,
            tor_client: None,
            // 🌻 v2.5.0-beta: Dandelion++ disabled in test/minimal mode
            dandelion: None,
            network_manager: None,
            consensus_service: None, // 🔐 v1.3.11-beta: Decentralized consensus (will be initialized in multi-node mode)
            production_peer_discovery: None,
            libp2p_discovery: None,  // Disabled in test mode
            libp2p_command_tx: None, // Disabled in test mode
            libp2p_peer_info: Arc::new(RwLock::new((String::new(), vec![]))), // Empty initially
            libp2p_peer_count: None, // Disabled in test mode
            network_metrics: None, // v10.9.27: wired in main.rs after UnifiedNetworkManager init
            di_ema: Arc::new(std::sync::atomic::AtomicU64::new(0)), // v9.0.6: DI EMA smoothing
            node_signing_key: Arc::new(ed25519_dalek::SigningKey::generate(&mut rand::rngs::OsRng)), // 💱 v0.6.1-beta: DEX pool signing key
            node_cypher: Arc::new(q_eternal_cypher::NodeCypher::from_ed25519_key(ed25519_dalek::SigningKey::generate(&mut rand::rngs::OsRng))), // v7.2.12: test dummy
            admin_wallet: crate::aegis_auth_middleware::FOUNDER_WALLET.to_string(),
            stripe_client: crate::payment_api::init_stripe_client().ok(),
            p2p_bytes_in: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            p2p_bytes_out: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            // 🔥 Top Movers ring buffer — empty deque + zero ingest height.
            recent_balance_deltas: Arc::new(RwLock::new(std::collections::VecDeque::new())),
            top_movers_last_ingested_height: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            network_throttle_mode: Arc::new(std::sync::atomic::AtomicU8::new(2)), // 2 = Turbo (default)
            highest_network_height: Arc::new(std::sync::atomic::AtomicU64::new(0)), // Sync mode tracking
            last_peer_height_update: Arc::new(std::sync::atomic::AtomicU64::new(0)), // v5.2.0: Peer height staleness
            sync_trigger: Arc::new(tokio::sync::Notify::new()), // v5.2.0: Immediate sync wake-up
            sharkgod_block_wake: Some(Arc::new(tokio::sync::Notify::new())), // 🦈 SharkGod block producer wake
            dev_fee_bps: Arc::new(std::sync::atomic::AtomicU64::new(190)), // v8.8.1: 190 bps = 1.9% mainnet dev fee
            node_operator_fee_promille: Arc::new(std::sync::atomic::AtomicU64::new(0)), // v7.3.1: disabled by default
            dex_protocol_fee_bps: Arc::new(std::sync::atomic::AtomicU64::new(5)), // v7.3.1: 5 bps = 0.05% protocol fee from swaps
            operator_fees_earned_session: Arc::new(std::sync::atomic::AtomicU64::new(0)), // v8.1.1
            operator_fees_earned_total: Arc::new(std::sync::atomic::AtomicU64::new(0)), // v8.1.1
            operator_fee_tx_count: Arc::new(std::sync::atomic::AtomicU64::new(0)), // v8.1.1
            current_height_atomic: Arc::new(std::sync::atomic::AtomicU64::new(initial_height)), // ⚡ v0.9.66-beta: Lock-free height (max-seen)
            contiguous_height_atomic: Arc::new(std::sync::atomic::AtomicU64::new(initial_height)), // v1.0.2: Honest contiguous height; refreshed every 5s
            peak_height_atomic: Arc::new(std::sync::atomic::AtomicU64::new(initial_height)), // v8.2.9: Peak height (never decreases)
            api_requests_served: Arc::new(std::sync::atomic::AtomicU64::new(0)), // v10.9.19: engine_pulse counter
            height_state: q_storage::HeightState::new(initial_height), // 🚀 v1.0.2-beta: HeightState cache - Eliminates binary search storm
            shutdown_tx: {
                let (tx, _rx) = tokio::sync::broadcast::channel(1);
                tx
            }, // 🛑 v1.0.2-beta: Graceful shutdown broadcast
            auto_update_tx: None, // 🔄 v8.5.0: Auto-update (set in main.rs after init)
            auto_update_state: None, // 🔄 v8.5.0: Auto-update state (set in main.rs after init)
            auto_update_enabled: Arc::new(std::sync::atomic::AtomicBool::new(
                std::env::var("Q_AUTO_UPDATE").unwrap_or_else(|_| "0".to_string()) == "1"
            )), // 🔄 v8.5.1: Runtime auto-update toggle
            bootstrap_wallet_sync_done: Arc::new(std::sync::atomic::AtomicBool::new(false)), // 🚀 v8.8.2: Bootstrap sync (test mode: not done)
            ai_active: Arc::new(std::sync::atomic::AtomicBool::new(false)), // 🤖 v9.3.3: AI inference throttle
            dex_ready: Arc::new(std::sync::atomic::AtomicBool::new(true)), // 🛡️ v10.3.1: DEX gate (test mode: always ready)
            startup_sync_complete: Arc::new(std::sync::atomic::AtomicBool::new(true)), // v10.3.2: test mode: always synced
            balance_finality_engine: None, // initialized in main.rs for validator nodes
            admin_notification_email: Arc::new(tokio::sync::RwLock::new(
                std::env::var("Q_ADMIN_NOTIFICATION_EMAIL").ok()
            )), // 🔄 v8.5.1: Admin notification email
            current_challenge: Arc::new(tokio::sync::RwLock::new(None)), // 🔧 v1.0.4-beta: Challenge caching
            fork_detector: Arc::new(q_storage::fork_detector::ForkDetector::new()), // 🔍 v0.9.67-beta: Comprehensive fork detection
            sync_start_time: Arc::new(std::sync::RwLock::new(None)), // 🎨 v0.6.6-beta: Progress bar sync tracking
            sync_start_height: Arc::new(std::sync::atomic::AtomicU64::new(0)), // 🎨 v0.6.6-beta: Progress bar sync tracking
            mining_submission_tx: None, // Disabled in test mode
            mining_submission_txs: None, // v8.9.0: Sharded pipeline (disabled in test mode)
            mining_shard_index: None,
            mining_solutions_submitted: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            mining_solutions_accepted: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            miner_stats_tx: None,
            sse_mining_event_tx: None,
            mining_nonce_dedup: Arc::new(dashmap::DashMap::new()),
            optimistic_applied_txs: Arc::new(dashmap::DashMap::new()),
            connection_manager: None,
            dag_sync_manager: None, // Will be initialized with PeerRegistry

            // ZK Privacy Components - Initialize with None
            zk_stark_system: None,
            zk_snark_system: None,
            // ✅ Post-Quantum LatticeGuard - Initialize with None in minimal mode
            lattice_guard: None,
            lattice_guard_srs: None,

            // ✅ v2.4.1-beta: TemporalShield - Initialize with None in minimal mode
            temporal_trustee_manager: None,

            // Performance & Scaling Optimizations - Initialize for maximum TPS
            simd_crypto_engine: {
                let simd_config = q_crypto_simd::SimdCryptoConfig::default();
                match q_crypto_simd::SimdCryptoEngine::new(simd_config).await {
                    Ok(engine) => {
                        tracing::info!(
                            "✅ SIMD Crypto Engine initialized - Vectorized cryptography enabled"
                        );
                        Some(Arc::new(engine))
                    }
                    Err(e) => {
                        tracing::warn!(
                            "⚠️ SIMD Crypto Engine initialization failed: {}, using fallback",
                            e
                        );
                        None
                    }
                }
            },
            #[cfg(target_os = "linux")]
            kernel_io_engine: {
                match crate::io_uring_adapter::IoUringAdapter::new() {
                    Ok(adapter) => {
                        tracing::info!(
                            "✅ Kernel I/O Engine initialized with dedicated thread pool"
                        );
                        Some(Arc::new(adapter))
                    }
                    Err(e) => {
                        tracing::warn!(
                            "⚠️ Kernel I/O Engine failed to initialize: {}, using standard I/O",
                            e
                        );
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
            production_mempool: None, // Will be initialized in main.rs for high TPS
            reliable_broadcast: None,
            quantum_vdf: None,
            quorum_commit_collector: Arc::new(quorum_commit::QuorumCommitCollector::new(3)), // 3-of-4 quorum (f=1)

            // PHASE 2: Parallel Block Producer Pool - 8 concurrent producers for exciting visualization
            // ✅ v0.9.92-beta: LOCK-FREE ARCHITECTURE - Channel-based message passing (DEADLOCK FIX)
            block_producer_pool: {
                // Create base config for all producers
                let base_config = crate::block_producer::BlockProducerConfig {
                    block_interval_secs, // v0.0.22-beta: from extracted config (2s for fast visualization)
                    max_solutions_per_block, // v0.0.22-beta: from extracted config
                    min_solutions_per_block, // v0.0.22-beta: from extracted config
                    node_id,
                    is_validator,
                    validator_index: 0, // Will be overridden by pool for each producer
                    total_validators: 8, // ✅ v1.0.17-beta: Restored to 8 (deadlock was NOT in producer contention)
                    network_id_str: std::env::var("Q_NETWORK_ID").unwrap_or_else(|_| {
                        let now = chrono::Utc::now().timestamp() as u64;
                        if now >= 1771761600 { "mainnet-genesis".to_string() } else { "mainnet2026.1.3".to_string() }
                    }),
                };

                // Create pool with 8 parallel producers for true parallelism
                let num_producers = 4; // 🔧 v8.0.9: Increased to 4 for 10+ bps target with interval=0

                // ✅ v0.9.92-beta DEADLOCK FIX: Use LOCK-FREE producer pool with channel-based architecture
                // This completely eliminates the RwLock deadlock that caused 9+ hour stalls
                // ✅ v0.9.99-beta: Now includes adaptive block rewards
                info!("🔓 Initializing LOCK-FREE Block Producer Pool (v0.9.92-beta DEADLOCK FIX)");
                let pool = crate::lockfree_producer::LockFreeProducerPool::new_with_storage(
                    num_producers,
                    base_config,
                    &storage_engine, // Pass storage Arc to load blockchain state
                    Some(balance_consensus_engine.clone()), // ✅ v0.9.99-beta: Adaptive rewards
                )
                .await?;

                info!(
                    "✅ LOCK-FREE Block Production initialized with {} producers",
                    num_producers
                );
                info!("   🔓 ZERO RwLocks - Channel-based architecture");
                info!("   ⚡ ZERO lock contention - Message passing only");
                info!("   🛡️  ZERO deadlock risk - No shared mutable state");
                info!("   ⚡ Exciting visualization: Multiple blocks will appear simultaneously in different lanes!");

                // 🚨 v0.9.9-beta CRITICAL FIX: Sync producers immediately after initialization
                // This ensures producers have the correct height even if crash recovery runs later
                // and updates the blockchain height. Without this, producers stay at stale height
                // from initialization and cause height regression errors after restart.
                if let Err(e) = pool.sync_from_storage(&storage_engine).await {
                    error!("❌ Failed to sync producers at startup: {}", e);
                }

                let pool_arc = Arc::new(pool);

                // 🚀 v1.0.3.9-beta: START STATE CONSISTENCY MONITOR (PRIMARY FIX)
                // Monitors database vs producer height every 10 seconds and auto-resyncs on divergence
                // This prevents the stale state bug where producers fall behind database state
                info!("🚀 [v1.0.3.9-beta] Starting state consistency monitor...");
                pool_arc
                    .clone()
                    .spawn_state_monitor(storage_engine.clone())
                    .await;
                info!("✅ [v1.0.3.9-beta] State monitor active - monitoring every 10s");

                pool_arc
            },

            /// 📊 v1.0.72-beta: Finality Metrics - Sub-50ms latency tracking
            finality_metrics: Arc::new(crate::block_producer::FinalityMetrics::default()),
            finality_certs: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),

            // AI Model Management - Lazy Loading (initialized later in main.rs if needed)
            ai_model_manager: None,

            // PHASE 3: DAG-Knight Consensus - Initialize with Byzantine fault tolerance
            consensus: {
                let consensus = DAGKnightConsensus::new(
                    node_id, 1, // f = 1 (supports 2f+1 = 3 nodes minimum for BFT)
                )
                .await?;

                info!(
                    "🎯 Initialized DAG-Knight consensus engine (f={}, min_nodes={})",
                    1, 3
                );

                Arc::new(RwLock::new(consensus))
            },

            // v9.3.1: K-parameter network health gauge (always-on)
            k_parameter_state: Arc::new(k_parameter_gauge::KParameterState::default()),

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
            orobit_ecosystem: orobit_ecosystem.clone(),
            symbol_to_address: Arc::new(dashmap::DashMap::new()), // v2.4.8: O(1) symbol lookup
            contract_events: Arc::new(RwLock::new(HashMap::new())), // v1.4.10: Contract event history

            // v1.4.11: Commit-reveal and stake-weighted finality for hybrid PoW/PoS security
            commit_reveal_manager: Arc::new(mining_commit_reveal::CommitRevealManager::new(true)),
            stake_finality_manager: Arc::new(stake_weighted_finality::StakeWeightedFinalityManager::new()),

            // Quillon Bank - Full Quantum Banking System with CDP
            quillon_bank,

            // v2.4.0-beta: Governance Coordinator (persistent across restarts)
            governance_coordinator,

            // AEGIS-QL Post-Quantum Authentication for Founder Operations
            aegis_auth_state,

            // QUG/QUGUSD Stablecoin System - CollateralVault
            collateral_vault,
            // v8.5.5: QCREDIT Yield Vault
            qcredit_vault,

            // Quillon Bank Loan Applications - Persistent storage
            pending_loan_applications: Arc::new(RwLock::new(pending_loan_applications_map)),

            // Distributed VM and DEX - Initialize in production mode
            distributed_protocol: None, // Initialized separately

            // OAuth2 Provider - v7.3.5: RocksDB-persisted clients (survive restarts)
            oauth2_storage: {
                let oauth2 = Arc::new(oauth2_provider::OAuth2Storage::with_storage(storage_engine.clone()));
                oauth2.load_clients_from_disk().await;
                oauth2
            },

            // v8.1.7: OAuth2 Key Vault — load encrypted signing keys from RocksDB
            oauth2_key_vault: {
                let vault_keys = storage_engine.load_vault_keys().await.unwrap_or_default();
                if !vault_keys.is_empty() {
                    tracing::info!("🔐 [VAULT] Loaded {} custodial signing keys from RocksDB", vault_keys.len());
                }
                Arc::new(RwLock::new(vault_keys))
            },

            // v7.4.0: Peer JWT public keys for cross-node token verification
            peer_jwt_keys: Arc::new(dashmap::DashMap::new()),
            peer_operator_wallets: Arc::new(dashmap::DashMap::new()),

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

            // Distributed AI Coordinator - Initialized separately
            distributed_ai_coordinator: None,

            // AI Verification System - Initialized in main.rs
            proof_verifier: None,
            benchmark_verifier: None,
            failover_manager: None,
            verification_events_tx: None,

            // Turbo Sync - Git-inspired fast blockchain synchronization (initialized in main.rs)
            turbo_sync: None,
            flight_computer: None,

            // 🚀 v1.0.4-beta: Phase 2 DAG-Aware Sync - Feature flag (set in main.rs)
            enable_dag_sync: false, // Will be set in main.rs based on Q_ENABLE_DAG_SYNC env var

            // v0.9.6-beta: Peer Registry Bridge (initialized in main.rs)
            // TEMPORARILY DISABLED v1.0.15: Circular dependency
            // peer_bridge: None,

            // AEGIS-KL Miner Authentication (initialized in main.rs)
            // miner_auth: None,

            // v0.9.1-beta: DeFi components (initialized later when needed)
            token_registry: None,
            price_history: None,
            // v0.9.1-beta: DEX components enabled
            dex_manager: None,
            price_bridge: None,

            // v2.3.34-beta: Swap history for Token Details Modal
            swap_history: Arc::new(RwLock::new(HashMap::new())),

            // v2.4.0-beta: Consensus-Verified Swap Indexer
            swap_indexer: Arc::new(swap_indexer::SwapIndexer::new(storage_engine.clone())),

            // v3.7.1-beta: Consensus-Verified Price History Indexer
            price_history_indexer: Arc::new(price_history_indexer::PriceHistoryIndexer::new(storage_engine.clone())),

            // v2.3.8-beta: Volume and price tracking for real oracle data
            volume_tracker: Arc::new(RwLock::new(HashMap::new())),
            price_snapshots: Arc::new(RwLock::new(HashMap::new())),

            // 🚀 v1.0.2-beta PHASE 1A: SAFE BATCHED SYNC - Initialized in main.rs
            fast_sync_enabled: false, // Will be set in main.rs based on CLI flag
            fast_sync_tx: None,       // Will be initialized in main.rs if enabled
            #[cfg(not(target_os = "windows"))]
            fast_sync_metrics: None,  // Will be initialized in main.rs if enabled

            // ✅ v1.0.7-beta: AsyncStorageEngine - Initialized in main.rs after DB setup
            #[cfg(not(target_os = "windows"))]
            async_storage: None, // Will be initialized in main.rs with DB handle

            // ✨ v1.0.16-beta: PQC Validator Keypair - Initialized in main.rs if --validator-key provided
            validator_keypair: None, // Will be set in main.rs after loading from file

            // ✨ v1.0.16-beta: Validator Key Registry - For PQC signature verification
            validator_key_registry: Arc::new(RwLock::new(q_types::ValidatorKeyRegistry::new())),
            validator_registry: Arc::new(RwLock::new(q_types::validator_registry::ValidatorRegistry::new())),

            // ⏰ v1.0.15-beta: Timeout-Based Sync Activation - Will be initialized in main.rs
            sync_activator: None, // Will be set in main.rs after AppState creation

            // ✨ v1.4.0-beta: Recursive Proofs Service - Will be initialized in main.rs
            recursive_proofs_service: None, // Will be set in main.rs after AppState creation

            // ✨ v1.4.2-beta: Block-Height Activated Upgrade Manager
            // Determines if network is mainnet from environment variable
            upgrade_manager: Arc::new(UpgradeManager::new(
                std::env::var("Q_MAINNET").map(|v| v == "1" || v.to_lowercase() == "true").unwrap_or(false)
            )),

            // 🔮 v1.4.2-beta: QNO Prediction Staking - Will be initialized after DB is ready
            #[cfg(not(target_os = "windows"))]
            qno_storage: Arc::new(RwLock::new(None)),

            // ⛏️ v2.3.0-beta: Stratum Mining Pool (initialized in main.rs)
            mining_pool: None,

            // 🌐 v2.3.0-beta: Decentralized Mining Pool (initialized in main.rs)
            distributed_pool_coordinator: None,
            distributed_pool_outbound_tx: None,
            distributed_pplns_proportions: Arc::new(tokio::sync::RwLock::new(None)),
            pool_hashrate_history: Arc::new(tokio::sync::RwLock::new(Vec::new())),

            // 💰 v2.4.8-beta: Dollar Cost Averaging (DCA) Storage
            dca_storage: Some(Arc::new(dca_api::DcaStorage::new())),

            // 🎯 v10.4.9: Limit Order Storage
            limit_order_storage: Some(Arc::new(limit_order_api::LimitOrderStorage::new())),

            // 📈 v2.5.0-beta: Perpetual Futures Storage
            perp_storage: Some(Arc::new(perpetual_api::PerpStorage::new())),

            // v2.4.2: Token staking & fee system - load from persistent storage
            token_fee_configs: {
                let configs = storage_engine.load_fee_configs().await.unwrap_or_default();
                if !configs.is_empty() {
                    tracing::info!("📊 Loaded {} token fee configurations from storage", configs.len());
                }
                Arc::new(RwLock::new(configs))
            },
            token_staking_positions: {
                let positions = storage_engine.load_stake_positions().await.unwrap_or_default();
                if !positions.is_empty() {
                    tracing::info!("🥩 Loaded {} staking positions from storage", positions.len());
                }
                Arc::new(RwLock::new(positions))
            },
            token_burn_totals: Arc::new(RwLock::new(HashMap::new())),
            token_reflection_totals: Arc::new(RwLock::new(HashMap::new())),
            token_social_profiles: Arc::new(RwLock::new(HashMap::new())),
            // 🚨 v3.3.3-beta: Emergency pause mechanism
            emergency_paused: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            emergency_pause_reason: Arc::new(RwLock::new(None)),
            emergency_pause_timestamp: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            // ⛏️ v9.1.4: Dynamic mining mode switch
            forced_mining_mode: Arc::new(std::sync::atomic::AtomicU8::new(0)),
            forced_pool_url: Arc::new(RwLock::new(None)),
            // 📬 v3.9.1-beta: Bank messaging and identity systems (loaded from RocksDB)
            bank_messages: {
                let mut loaded_messages = Vec::new();
                if let Ok(entries) = storage_engine.load_bank_messages().await {
                    for (_key, value) in entries {
                        if let Ok(msg) = serde_json::from_slice::<crate::quillon_bank_api::BankMessage>(&value) {
                            loaded_messages.push(msg);
                        }
                    }
                    if !loaded_messages.is_empty() {
                        tracing::info!("📬 Loaded {} bank messages from RocksDB", loaded_messages.len());
                    }
                }
                Arc::new(RwLock::new(loaded_messages))
            },
            user_identities: {
                let mut loaded_identities = Vec::new();
                if let Ok(entries) = storage_engine.load_user_identities().await {
                    for (_key, value) in entries {
                        if let Ok(id) = serde_json::from_slice::<crate::quillon_bank_api::UserIdentity>(&value) {
                            loaded_identities.push(id);
                        }
                    }
                    if !loaded_identities.is_empty() {
                        tracing::info!("🪪 Loaded {} user identities from RocksDB", loaded_identities.len());
                    }
                }
                Arc::new(RwLock::new(loaded_identities))
            },
            death_certificates: {
                let mut loaded_certs = Vec::new();
                if let Ok(entries) = storage_engine.load_death_certificates().await {
                    for (_key, value) in entries {
                        if let Ok(cert) = serde_json::from_slice::<crate::quillon_bank_api::DeathCertificate>(&value) {
                            loaded_certs.push(cert);
                        }
                    }
                    if !loaded_certs.is_empty() {
                        tracing::info!("💀 Loaded {} death certificates from RocksDB", loaded_certs.len());
                    }
                }
                Arc::new(RwLock::new(loaded_certs))
            },
            // 🔐 v4.2.0: VAULT RWA redemptions
            vault_redemptions: Arc::new(RwLock::new(Vec::new())),
            // v5.1.0: FORGE RWA redemptions
            forge_redemptions: Arc::new(RwLock::new(Vec::new())),
            // v6.5.0: Exchange Listing RWA orders
            listing_orders: Arc::new(RwLock::new(Vec::new())),
            // v8.2.8: XLIST Crowdfunding campaigns
            listing_campaigns: Arc::new(RwLock::new(Vec::new())),
            // v5.1.1: Node start time
            start_time: std::time::Instant::now(),
            // v5.1.1: Deploy admin state
            deploy_state: Arc::new(RwLock::new(deploy_admin_api::DeployState::new())),
            // v7.2.0: Miner link relay
            miner_link_registry: miner_link_api::new_registry(),
            // v7.2.0: Bitcoin bridge (initialized separately if configured)
            atomic_swap_manager: None,
            // v10.2.11: Bitcoin deposit bridge (initialized below from BTC_RPC_* env vars)
            deposit_bridge: None,
            // v9.6.5: Bitcoin Knots RPC client (initialized below)
            bitcoin_rpc_client: None,
            // v9.7.2: Zcash Zebra RPC client (initialized below)
            zcash_rpc_client: None,
            // v7.3.1: Bridge committee (peer ID set later)
            bridge_committee: Arc::new(RwLock::new(bridge_committee::BridgeCommittee::new(String::new()))),
            // v9.4.0: Bridge safety controller
            bridge_safety: Arc::new(bridge_safety::BridgeSafetyController::new()),
            // v9.5.0: Compute orchestrator (initialized later from CLI arg)
            compute_orchestrator: None,
            // v9.6.1: QR code payment requests for brick-and-mortar POS
            payment_requests: Arc::new(dashmap::DashMap::new()),
            // v10.2.0: Crown & Ash — Medieval grand strategy game state
            crown_ash_state: Arc::new(tokio::sync::RwLock::new(crown_ash_api::CrownAshGameState::empty())),
        })
    }

    pub async fn new_with_networks(
        config: Config,
        node_id: NodeId,
        bitcoin_bridge: Option<Arc<()>>,  // DEACTIVATED
        dns_phantom: Option<Arc<()>>,     // DEACTIVATED
        bep44_discovery: Option<Arc<()>>, // DEACTIVATED
        tor_client: Option<Arc<QTorClient>>,
        __production_peer_discovery: Option<
            Arc<tokio::sync::Mutex<q_network::real_peer_discovery::RealPeerDiscovery>>,
        >,
        libp2p_discovery: Option<Arc<tokio::sync::Mutex<q_network::UnifiedNetworkManager>>>,
        libp2p_command_tx: Option<tokio::sync::mpsc::UnboundedSender<q_network::NetworkCommand>>,
        // v8.0.2: Accept pre-created balance_consensus_engine from main.rs
        // This fixes the dual-controller bug where lib.rs and main.rs each created separate engines,
        // causing the block_producer_pool to use an empty/stale engine while the API used the correct one.
        balance_consensus_engine: Arc<q_storage::BalanceConsensusEngine>,
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

        let initial_height = 0; // Will be loaded from storage
        let node_status = NodeStatus {
            node_id,
            current_round: 0,
            current_height: initial_height,
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
            sync_writes: true, // CRITICAL: Enable fsync() to survive hard kills (pkill -9)
            cache_size_mb: 256,
            max_open_files: 1000,
        };

        let storage_engine = Arc::new(StorageEngine::new(storage_config).await?);

        // ✅ v0.9.10-beta CRITICAL FIX: Repair height pointer BEFORE loading blockchain state
        // This ensures crash recovery runs even in the `new_with_networks()` code path
        // which was missing the recovery call that exists in `new()`. Without this,
        // producers initialize at height=0 because storage appears empty on restart.
        tracing::info!("🔧 [v0.9.10] Running height pointer integrity check...");
        match storage_engine.repair_height_pointer().await {
            Ok(repaired_height) => {
                tracing::info!(
                    "✅ [v0.9.10] Height pointer verified/repaired: {}",
                    repaired_height
                );
            }
            Err(e) => {
                tracing::error!("❌ [v0.9.10] Height pointer repair failed: {}", e);
                tracing::error!("⚠️  Node will start from genesis - blockchain may need resync");
            }
        }

        // 🔀 v0.9.37-beta PHASE 3: Genesis block validation for fork detection
        // Validates that local genesis matches network consensus (prevents incompatible forks)
        tracing::info!("🔍 [v0.9.37] Validating genesis block against network consensus...");
        match storage_engine.validate_genesis_block(None).await {
            Ok(true) => {
                tracing::info!("✅ [v0.9.37] Genesis block validation passed");
            }
            Ok(false) => {
                tracing::error!("❌ [v0.9.37] CRITICAL: Genesis block mismatch detected!");
                tracing::error!("   This node is on an incompatible fork!");
                tracing::error!("   Local genesis differs from network consensus");
                tracing::error!(
                    "   Action required: Database reset or manual chain reorganization"
                );
                tracing::warn!("⚠️  Continuing despite genesis mismatch (allow fork debugging)");
            }
            Err(e) => {
                tracing::warn!("⚠️  [v0.9.37] Genesis validation failed: {}", e);
                tracing::warn!("   Continuing with startup (may be empty database)");
            }
        }

        // v1.3.4-beta: Skip NetworkManager when Tor is not enabled
        // NetworkManager uses Tor for DNS-phantom bridging, which blocks startup without Tor
        let network_manager = {
            let tor_enabled = std::env::var("Q_TOR_ENABLED").is_ok() ||
                std::env::var("Q_TOR_PROXY").is_ok();

            if tor_enabled {
                let mut tor_config = q_tor_client::TorConfig::default();
                tor_config.enabled = true; // Enable Tor for NetworkManager

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
                        tracing::info!("✅ NetworkManager initialized - DNS-phantom bridge ready (Tor enabled)");
                        Some(Arc::new(nm))
                    }
                    Err(e) => {
                        tracing::warn!("⚠️ NetworkManager initialization failed: {}, continuing without peer bridge", e);
                        None
                    }
                }
            } else {
                tracing::info!("🔌 Skipping NetworkManager (Tor not enabled) - using direct libp2p connections");
                None
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

        // v7.2.4: Genesis filter — purge testnet BALANCES only (never delete blocks!)
        // The old code deleted blocks with timestamp < GENESIS_TIMESTAMP on every restart,
        // causing height drops from 295K to 77K. Fixed to only purge stale balances/state.
        match storage_engine.purge_pre_genesis_balances_only().await {
            Ok(true) => {
                tracing::info!("🧹 [GENESIS FILTER] Stale testnet balances purged — blocks preserved");
            }
            Ok(false) => {
                tracing::info!("✅ [GENESIS FILTER] No stale testnet data found — clean state");
            }
            Err(e) => {
                tracing::warn!("⚠️ [GENESIS FILTER] Error during balance purge: {} — continuing", e);
            }
        }

        // v7.2.6: Second purge — clears testnet balances re-synced via P2P
        match storage_engine.purge_testnet_balances_v726().await {
            Ok(true) => {
                tracing::info!("🧹 [GENESIS FILTER v7.2.6] Testnet balances purged from ALL nodes");
            }
            Ok(false) => {
                tracing::info!("✅ [GENESIS FILTER v7.2.6] Already purged — clean state");
            }
            Err(e) => {
                tracing::warn!("⚠️ [GENESIS FILTER v7.2.6] Error: {} — continuing", e);
            }
        }

        // v7.2.12: Third purge — nuclear option. Delete ALL wallet balances and rebuild from chain.
        // Previous purges (v7.2.5, v7.2.6) ran once but P2P re-introduced testnet balances.
        // This time we purge AND rebuild from on-chain data only.
        let wallet_purge_ran = match storage_engine.purge_testnet_wallets_v7212().await {
            Ok(true) => {
                tracing::info!("🧹 [GENESIS FILTER v7.2.12] Purged ALL wallet balances — will rebuild from chain");
                true
            }
            Ok(false) => {
                tracing::info!("✅ [GENESIS FILTER v7.2.12] Already purged — skip rebuild");
                false
            }
            Err(e) => {
                tracing::warn!("⚠️ [GENESIS FILTER v7.2.12] Purge error: {} — continuing", e);
                false
            }
        };

        // Load or rebuild wallet balances
        let mut wallet_balances = HashMap::new();
        let mut total_supply;

        if wallet_purge_ran {
            // Rebuild balances from chain (only mainnet blocks)
            match storage_engine.rebuild_balances_from_chain().await {
                Ok((rebuilt_balances, rebuilt_supply)) => {
                    tracing::info!(
                        "✅ [REBUILD v7.2.12] Rebuilt {} wallet balances, supply={} QUG",
                        rebuilt_balances.len(),
                        rebuilt_supply / 1_000_000_000_000_000_000_000_000u128
                    );
                    wallet_balances = rebuilt_balances;
                    total_supply = rebuilt_supply;
                }
                Err(e) => {
                    tracing::warn!("⚠️ [REBUILD v7.2.12] Failed: {} — starting with empty balances", e);
                    total_supply = 0;
                }
            }
        } else {
            // Normal path: load from storage
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

            // 💎 Load total minted supply from storage (max supply enforcement)
            total_supply = match storage_engine.load_total_supply().await {
                Ok(supply) => {
                    tracing::info!(
                        "💎 Loaded total minted supply: {} QUG (max: 21M QUG)",
                        supply / 1_000_000_000_000_000_000_000_000u128
                    );
                    supply
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to load total supply from storage: {}, starting from 0",
                        e
                    );
                    0
                }
            };
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

        // v8.5.6: ONE-TIME QUGUSD ghost cleanup (replaces v8.5.3 every-restart purge).
        // Root causes FIXED: (1) restore migration flag persists, (2) state_sync_api rejects QUGUSD from P2P.
        // After one-time purge, legitimate QUGUSD from swaps/mints/loans persists across restarts.
        {
            const QUGUSD_PURGE_FLAG: &[u8] = b"migration_qugusd_ghost_purge_v856_done";
            if !storage_engine.has_migration_flag(QUGUSD_PURGE_FLAG).await {
                use q_types::QUGUSD_TOKEN_ADDRESS;
                let qugusd_entries: Vec<([u8; 32], [u8; 32])> = token_balances
                    .iter()
                    .filter(|((_wallet, token_addr), _)| *token_addr == QUGUSD_TOKEN_ADDRESS)
                    .map(|((wallet, token), _)| (*wallet, *token))
                    .collect();
                let qugusd_count = qugusd_entries.len();
                if qugusd_count > 0 {
                    for (wallet, token) in &qugusd_entries {
                        token_balances.remove(&(*wallet, *token));
                    }
                    for (wallet, token) in &qugusd_entries {
                        let _ = storage_engine.delete_token_balance(wallet, token).await;
                    }
                    tracing::warn!(
                        "🧹 [v8.5.6] ONE-TIME purge: removed {} ghost QUGUSD entries (won't run again)",
                        qugusd_count
                    );
                }
                let _ = storage_engine.set_migration_flag(QUGUSD_PURGE_FLAG).await;
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
        // v7.1.6: Filter out pre-genesis (testnet) pools
        let pool_genesis_ts = q_storage::emission_controller::GENESIS_TIMESTAMP;
        let mut liquidity_pools_map = HashMap::new();
        match storage_engine.load_liquidity_pools().await {
            Ok(persisted_pools) => {
                let mut filtered_pool_count = 0u64;
                for (pool_id, pool_bytes) in persisted_pools {
                    match serde_json::from_slice::<LiquidityPool>(&pool_bytes) {
                        Ok(pool) => {
                            let pool_ts = pool.created_at.timestamp() as u64;
                            if pool_ts > 0 && pool_ts < pool_genesis_ts {
                                filtered_pool_count += 1;
                                let _ = storage_engine.delete_liquidity_pool(&pool_id).await;
                                continue;
                            }
                            // v10.2.2: Purge dust/broken pools on startup
                            const STARTUP_MIN_POOL_RESERVE: u128 = 10_000_000_000_000_000_000_000; // 10^22
                            if pool.reserve0 < STARTUP_MIN_POOL_RESERVE || pool.reserve1 < STARTUP_MIN_POOL_RESERVE {
                                filtered_pool_count += 1;
                                tracing::info!(
                                    "🧹 [POOL CLEANUP] Purging dust pool {} ({}/{}) — r0={}, r1={}",
                                    pool_id, pool.token0, pool.token1, pool.reserve0, pool.reserve1
                                );
                                let _ = storage_engine.delete_liquidity_pool(&pool_id).await;
                                continue;
                            }
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
                if filtered_pool_count > 0 {
                    tracing::info!(
                        "🧹 [GENESIS FILTER] Purged {} pre-genesis/dust pools from storage",
                        filtered_pool_count
                    );
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

        // v3.9.3-beta: SKIP loading all transactions on startup
        // Loading 30M+ transactions wastes memory and slows startup
        // Transactions remain in storage and can be queried on-demand
        tracing::info!("💳 Historical transactions available in storage (not loaded into memory)");

        // Initialize real-time streaming
        let event_broadcaster = Arc::new(EventBroadcaster::new());
        let event_emitter = Arc::new(HighPerformanceEmitter::new(event_broadcaster.clone()));

        // Initialize VM and Smart Contract system with Orobit integration
        // IMPORTANT: Create ecosystem FIRST with storage, then pass to registry
        let orobit_ecosystem = Arc::new(
            OrobitSmartContractEcosystem::new_with_storage(Some(storage_engine.clone())).await?,
        );
        let contract_registry = Arc::new(ContractRegistry::new_with_ecosystem(
            orobit_ecosystem.clone(),
        ));

        // NOTE: Token balances are now loaded from persistent storage above
        // No need to restore from deployed contracts - persistence handles it

        // v7.1.7: Purge token balances for pre-genesis (testnet) contracts
        // orobit_ecosystem already filtered pre-genesis contracts at this point.
        // Remove token balances whose contract was purged (testnet orphans).
        // v7.2.12: Also delete from RocksDB so stablecoin_api won't resurrect them.
        {
            use q_types::{QUG_TOKEN_ADDRESS, QUGUSD_TOKEN_ADDRESS};
            let deployed = orobit_ecosystem.deployed_contracts.read().await;
            let before_count = token_balances.len();

            // Collect entries to remove BEFORE retain, for RocksDB cleanup
            let entries_to_remove: Vec<([u8; 32], [u8; 32])> = token_balances
                .iter()
                .filter(|((_wallet_addr, token_addr), _balance)| {
                    if *token_addr == QUG_TOKEN_ADDRESS || *token_addr == QUGUSD_TOKEN_ADDRESS {
                        return false; // keep native tokens
                    }
                    let ca = q_vm::contracts::orobit_smart_contracts::ContractAddress(*token_addr);
                    !deployed.contains_key(&ca)
                })
                .map(|((wallet_addr, token_addr), _)| (*wallet_addr, *token_addr))
                .collect();

            token_balances.retain(|(_wallet_addr, token_addr), _balance| {
                // Always keep native tokens
                if *token_addr == QUG_TOKEN_ADDRESS || *token_addr == QUGUSD_TOKEN_ADDRESS {
                    return true;
                }
                // Keep only if contract exists in post-genesis ecosystem
                let ca = q_vm::contracts::orobit_smart_contracts::ContractAddress(*token_addr);
                deployed.contains_key(&ca)
            });

            let removed = before_count - token_balances.len();

            // Delete from RocksDB so they don't reappear after restart
            let mut rocksdb_deleted = 0usize;
            for (wallet_addr, token_addr) in &entries_to_remove {
                if storage_engine.delete_token_balance(wallet_addr, token_addr).await.is_ok() {
                    rocksdb_deleted += 1;
                }
            }

            if removed > 0 {
                tracing::info!(
                    "🧹 [GENESIS FILTER] Purged {} testnet token balances (memory={}, RocksDB={}), {} remaining",
                    removed, removed, rocksdb_deleted, token_balances.len()
                );
            }
            drop(deployed);
        }

        // Initialize Quillon Bank - Full Quantum Banking System with CDP
        let plugin_manager = Arc::new(PluginManager::new());
        let quillon_bank_system =
            QuillonBankSystem::new(node_id, q_types::Phase::Phase1, plugin_manager.clone()).await?;
        quillon_bank_system.initialize().await?;
        let quillon_bank = Arc::new(RwLock::new(quillon_bank_system));
        tracing::info!("🏦 Quillon Bank initialized - CDP and quantum banking ready");

        // v2.4.0-beta: Initialize Governance Coordinator with RocksDB persistence
        let governance_coordinator = Arc::new(
            q_governance::GovernanceCoordinator::with_storage(storage_engine.clone()).await
        );
        tracing::info!("🏛️ Governance Coordinator initialized with RocksDB persistence");

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
                tracing::info!(
                    "✅ Founder wallet verified: qnk{}",
                    aegis_auth_middleware::FOUNDER_WALLET
                );
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
                    Ok(mut persisted_vault) => {
                        tracing::info!(
                            "💰 Loaded CollateralVault from storage: locked_qug={}, minted_qugusd={}, old_price=${}",
                            persisted_vault.total_qug_locked,
                            persisted_vault.total_qugusd_minted,
                            persisted_vault.qug_price_usd
                        );
                        // v4.0.4: Keep persisted vault price as-is (was updated from AMM during trading).
                        // v8.0.1: Migrate from old $42.50 default to $3000.00 default.
                        if persisted_vault.qug_price_usd <= 0.0 || persisted_vault.qug_price_usd < 100.0 {
                            tracing::warn!(
                                "💱 Vault QUG price was outdated/invalid (${:.6}), migrating to $3000.00",
                                persisted_vault.qug_price_usd
                            );
                            persisted_vault.qug_price_usd = 3000.00;
                        } else {
                            tracing::info!(
                                "💱 Loaded vault QUG price: ${:.4} (preserved from last session)",
                                persisted_vault.qug_price_usd
                            );
                        }

                        // v2.4.0: CRITICAL FIX - Detect and fix u64 underflow corruption
                        // v3.0.4: Updated to u128 for 24-decimal precision
                        // If total_qugusd_minted is impossibly large, this indicates corruption.
                        // Reset total_qugusd_minted to the actual sum of minted_qugusd values.
                        const IMPOSSIBLY_LARGE_SUPPLY: u128 = 1_000_000_000_000_000_000_000_000_000_000; // 10 billion with 24 decimals
                        if persisted_vault.total_qugusd_minted > IMPOSSIBLY_LARGE_SUPPLY {
                            let actual_sum: u128 = persisted_vault.minted_qugusd.values().sum();
                            tracing::error!(
                                "🚨 CORRUPTION DETECTED: total_qugusd_minted={} is impossibly large",
                                persisted_vault.total_qugusd_minted
                            );
                            tracing::warn!(
                                "🔧 FIXING: Resetting total_qugusd_minted from {} to actual sum {}",
                                persisted_vault.total_qugusd_minted,
                                actual_sum
                            );
                            persisted_vault.total_qugusd_minted = actual_sum;

                            // Also check and fix total_qug_locked if corrupted
                            let actual_qug_sum: u128 = persisted_vault.locked_qug.values().sum();
                            if persisted_vault.total_qug_locked > IMPOSSIBLY_LARGE_SUPPLY {
                                tracing::warn!(
                                    "🔧 FIXING: Resetting total_qug_locked from {} to actual sum {}",
                                    persisted_vault.total_qug_locked,
                                    actual_qug_sum
                                );
                                persisted_vault.total_qug_locked = actual_qug_sum;
                            }

                            tracing::info!("✅ Vault corruption fixed - totals now match actual balances");
                        }

                        // v8.1.2: REMOVED unconditional vault wipe (was v7.2.13)
                        // BUG FIX: The old code destroyed ALL CDP positions on every restart!
                        // Users who minted QUGUSD via collateral lost their positions.
                        if !persisted_vault.minted_qugusd.is_empty() {
                            let count = persisted_vault.minted_qugusd.len();
                            let total: u128 = persisted_vault.minted_qugusd.values().sum();
                            tracing::info!(
                                "✅ [v8.1.2] Preserved {} minted_qugusd entries (total={:.2} QUGUSD) in CollateralVault",
                                count, total as f64 / 1e24
                            );
                        }

                        Arc::new(RwLock::new(persisted_vault))
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Failed to deserialize CollateralVault: {}, creating new vault",
                            e
                        );
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
        // v4.0.7: Initialize vault QUG price from pool reserves on startup
        // Ensures price reflects actual market state even if vault persistence was missed
        {
            let mut vault_w = collateral_vault.write().await;
            let current_price = vault_w.qug_price_usd;
            let mut pool_price: f64 = 0.0;
            for p in liquidity_pools_map.values() {
                let t0 = p.token0.to_uppercase();
                let t1 = p.token1.to_uppercase();
                let t0_is_qug = t0 == "QUG" || t0 == "NATIVE-QUG"
                    || t0 == hex::encode([0u8; 32]).to_uppercase();
                let t1_is_qug = t1 == "QUG" || t1 == "NATIVE-QUG"
                    || t1 == hex::encode([0u8; 32]).to_uppercase();
                let qugusd_hex = hex::encode(q_types::QUGUSD_TOKEN_ADDRESS).to_uppercase();
                let t0_is_qugusd = t0 == "QUGUSD" || t0 == qugusd_hex
                    || t0 == format!("QNK{}", qugusd_hex);
                let t1_is_qugusd = t1 == "QUGUSD" || t1 == qugusd_hex
                    || t1 == format!("QNK{}", qugusd_hex);
                if (t0_is_qug && t1_is_qugusd) || (t0_is_qugusd && t1_is_qug) {
                    let (qug_r, usd_r) = if t0_is_qug {
                        (p.reserve0 as f64, p.reserve1 as f64)
                    } else {
                        (p.reserve1 as f64, p.reserve0 as f64)
                    };
                    if qug_r > 0.0 {
                        pool_price = usd_r / qug_r;
                    }
                    break;
                }
            }
            // v8.5.9: Force pool reserves to match $3000 target price on every startup.
            // Ghost trades from the supply inflation bug contaminated reserves — rather than
            // deleting the pool (which re-bootstraps with stale data), we directly correct
            // the reserve ratio to the authoritative oracle price.
            {
                let target_price = 3000.0_f64;
                if pool_price > 0.0 && (pool_price < target_price * 0.9 || pool_price > target_price * 1.1) {
                    // Pool ratio drifted >10% from target — reset reserves
                    let pool_id = "pool-qug-qugusd-bootstrap".to_string();
                    if let Some(mut pool_ref) = liquidity_pools_map.get_mut(&pool_id) {
                        // DashMap RefMut derefs directly to LiquidityPool
                        let qug_reserve_f64 = pool_ref.reserve0 as f64 / 1e24;
                        let new_qugusd_reserve = (qug_reserve_f64 * target_price * 1e24) as u128;
                        tracing::warn!(
                            "💱 [v8.5.9] Pool price ${:.2} drifted from target ${:.0} — resetting reserves (QUG={:.2}, QUGUSD: {:.0} → {:.0})",
                            pool_price, target_price, qug_reserve_f64,
                            pool_ref.reserve1 as f64 / 1e24, new_qugusd_reserve as f64 / 1e24
                        );
                        pool_ref.reserve1 = new_qugusd_reserve;
                        pool_ref.lp_token_supply = ((pool_ref.reserve0 as f64 * new_qugusd_reserve as f64).sqrt()) as u128;
                        // Serialize and persist
                        if let Ok(data) = serde_json::to_vec(&*pool_ref) {
                            let _ = storage_engine.save_liquidity_pool(&pool_id, &data).await;
                        }
                        pool_price = target_price;
                    }
                    vault_w.qug_price_usd = target_price;
                    vault_w.last_price_update = chrono::Utc::now().timestamp();
                }
            }
            if pool_price > 0.0 && pool_price < 1_000_000.0 {
                vault_w.qug_price_usd = pool_price;
                vault_w.last_price_update = chrono::Utc::now().timestamp();
                tracing::info!(
                    "💱 [STARTUP v4.0.7] Set vault QUG price from pool reserves: ${:.4} (was ${:.4})",
                    pool_price, current_price
                );
            } else {
                tracing::info!(
                    "💱 [STARTUP v4.0.7] No QUG/QUGUSD pool found, keeping vault price at ${:.4}",
                    current_price
                );

                // v4.0.7: Auto-create QUG/QUGUSD pool so AMM price discovery works
                // Without this pool, all QUG<->QUGUSD swaps use oracle (static price)
                let vault_price = current_price;
                let bootstrap_qug: u128 = 10_000 * 1_000_000_000_000_000_000_000_000u128; // 10k QUG in 24-decimal
                let bootstrap_qugusd: u128 = (10_000.0 * vault_price * 1e24) as u128;
                let pool_id = "pool-qug-qugusd-bootstrap".to_string();
                let pool = LiquidityPool {
                    pool_id: pool_id.clone(),
                    token0: "QUG".to_string(),
                    token1: "QUGUSD".to_string(),
                    reserve0: bootstrap_qug,
                    reserve1: bootstrap_qugusd,
                    provider: [0u8; 32],
                    created_at: chrono::Utc::now(),
                    lp_token_supply: ((bootstrap_qug as f64 * bootstrap_qugusd as f64).sqrt()) as u128,
                    token0_decimals: 24,
                    token1_decimals: 24,
                };
                liquidity_pools_map.insert(pool_id.clone(), pool.clone());
                if let Ok(pool_bytes) = serde_json::to_vec(&pool) {
                    if let Err(e) = storage_engine.save_liquidity_pool(&pool_id, &pool_bytes).await {
                        tracing::warn!("⚠️ Failed to persist bootstrap QUG/QUGUSD pool: {}", e);
                    }
                }
                tracing::info!(
                    "🏊 [STARTUP v4.0.7] Created bootstrap QUG/QUGUSD pool: 10,000 QUG / {:.0} QUGUSD @ ${:.2}/QUG",
                    bootstrap_qugusd as f64 / 1e24, vault_price
                );
            }
        }
        tracing::info!("💰 CollateralVault initialized - QUG/QUGUSD stablecoin system ready");

        // v8.2.7: Bootstrap bridge pools with LIVE oracle prices (CoinGecko/Binance)
        {
            let vault_r = collateral_vault.read().await;
            let qug_price = vault_r.qug_price_usd;
            drop(vault_r);
            let bank_r = quillon_bank.read().await;
            let oracle_ref = bank_r.oracle_integration.as_ref();
            bootstrap_bridge_pools(&mut liquidity_pools_map, &storage_engine, qug_price, Some(oracle_ref)).await;
            drop(bank_r);
        }

        // v8.5.5: Initialize QCREDIT Yield Vault from storage or create new
        let qcredit_vault = match storage_engine.load_qcredit_vault().await {
            Ok(Some(vault_bytes)) => {
                match serde_json::from_slice::<q_vm::contracts::QCreditVault>(&vault_bytes) {
                    Ok(persisted) => {
                        tracing::info!(
                            "💳 Loaded QCREDIT vault: total_locked={:.2}, positions={}, reserve={:.2}",
                            persisted.total_locked as f64 / 1e24,
                            persisted.positions.values().map(|v| v.len()).sum::<usize>(),
                            persisted.protocol_reserve as f64 / 1e24,
                        );
                        Arc::new(RwLock::new(persisted))
                    }
                    Err(e) => {
                        tracing::warn!("Failed to deserialize QCREDIT vault: {}, creating new", e);
                        Arc::new(RwLock::new(q_vm::contracts::QCreditVault::new()))
                    }
                }
            }
            Ok(None) => {
                tracing::info!("💳 No persisted QCREDIT vault found, creating new with seed reserve");
                let mut vault = q_vm::contracts::QCreditVault::new();
                // v9.5.1: Seed protocol reserve with 100K QUG to fund yield payouts
                let seed_reserve: u128 = 100_000 * 1_000_000_000_000_000_000_000_000u128;
                vault.fund_reserve(seed_reserve);
                tracing::info!("💳 Seeded QCREDIT protocol reserve with 100,000 QUG");
                Arc::new(RwLock::new(vault))
            }
            Err(e) => {
                tracing::warn!("Failed to load QCREDIT vault: {}, creating new with seed reserve", e);
                let mut vault = q_vm::contracts::QCreditVault::new();
                let seed_reserve: u128 = 100_000 * 1_000_000_000_000_000_000_000_000u128;
                vault.fund_reserve(seed_reserve);
                Arc::new(RwLock::new(vault))
            }
        };
        tracing::info!("💳 QCREDIT Yield Vault initialized");

        // v8.5.5: Bootstrap QUG/QCREDIT pool at 1:1 ratio
        {
            let qcredit_pool_id = "pool-qug-qcredit-bootstrap".to_string();
            if !liquidity_pools_map.contains_key(&qcredit_pool_id) {
                let bootstrap_amount: u128 = 10_000 * 1_000_000_000_000_000_000_000_000u128;
                let pool = LiquidityPool {
                    pool_id: qcredit_pool_id.clone(),
                    token0: format!("qnk{}", hex::encode(q_types::QUG_TOKEN_ADDRESS)),
                    token1: format!("qnk{}", hex::encode(q_types::QCREDIT_TOKEN_ADDRESS)),
                    reserve0: bootstrap_amount,
                    reserve1: bootstrap_amount,
                    provider: [0u8; 32],
                    created_at: chrono::Utc::now(),
                    lp_token_supply: bootstrap_amount,
                    token0_decimals: 24,
                    token1_decimals: 24,
                };
                liquidity_pools_map.insert(qcredit_pool_id.clone(), pool.clone());
                if let Ok(pool_bytes) = serde_json::to_vec(&pool) {
                    let _ = storage_engine.save_liquidity_pool(&qcredit_pool_id, &pool_bytes).await;
                }
                tracing::info!("🏊 [v8.5.5] Created bootstrap QUG/QCREDIT pool: 10K/10K @ 1:1");
            }
        }

        // Load existing loan applications from persistent storage
        let mut pending_loan_applications_map = HashMap::new();
        match storage_engine.load_loan_applications().await {
            Ok(persisted_loans) => {
                for (loan_id, loan_bytes) in persisted_loans {
                    match bincode::deserialize::<crate::quillon_bank_api::LoanApplication>(
                        &loan_bytes,
                    ) {
                        Ok(loan) => {
                            pending_loan_applications_map.insert(loan_id.clone(), loan);
                        }
                        Err(e) => {
                            tracing::warn!(
                                "Failed to deserialize loan application {}: {}, skipping",
                                loan_id,
                                e
                            );
                        }
                    }
                }
                tracing::info!(
                    "🏦 Loaded {} loan applications from persistent storage",
                    pending_loan_applications_map.len()
                );
            }
            Err(e) => {
                tracing::warn!(
                    "Failed to load loan applications from storage: {}, starting with empty loans",
                    e
                );
            }
        }

        // v8.0.2: balance_consensus_engine is now passed in from main.rs
        // This eliminates the dual-controller bug where lib.rs created its own engine
        // that diverged from main.rs's engine (which has restored emission state).
        // The block_producer_pool (LockFreeProducerPool) and API endpoints now use the SAME engine.
        tracing::info!("✅ v8.0.2: Using unified balance_consensus_engine from main.rs (no dual-controller)");

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

            // ✅ v1.0.91-beta: Initialize nonce tracker for replay attack prevention
            nonce_tracker: Arc::new(transaction_utils::NonceTracker::new()),
            // ✅ v9.7.0: Cross-block tx dedup cache
            applied_tx_dedup: Arc::new(dashmap::DashMap::new()),

            balance_consensus_engine: balance_consensus_engine.clone(),
            event_broadcaster,
            event_emitter,

            // Faucet system
            // v7.0.0: faucet_state removed

            // 🔒 MAX SUPPLY ENFORCEMENT - Initialize supply tracking
            total_minted_supply: Arc::new(RwLock::new(total_supply)), // Loaded from storage on startup
            supply_consensus_state: Arc::new(RwLock::new(SupplyConsensusState::default())),
            mining_statistics: Some(Arc::new(RwLock::new(MiningStatistics::default()))),

            // 💓 MINING HEARTBEAT MONITORING (v0.8.9-beta) - Detect mining stalls
            last_mining_solution_time: Arc::new(std::sync::atomic::AtomicU64::new(
                chrono::Utc::now().timestamp() as u64,
            )),
            mining_is_healthy: Arc::new(std::sync::atomic::AtomicBool::new(true)),

            // Quantum Privacy Mixer State
            mixing_requests: Arc::new(RwLock::new(HashMap::new())),
            quantum_mixer: None, // Initialized later if needed
            zkp_prover: None,    // Initialized later if needed

            // Network components with provided values
            bitcoin_bridge,
            dns_phantom,
            bep44_discovery,
            tor_client: tor_client.clone(),

            // 🌻 v2.5.0-beta: Dandelion++ for mandatory Tor-based transaction anonymity
            // Initialized to None here, will be set up in main.rs after AppState creation
            dandelion: None,

            network_manager,
            consensus_service: None, // 🔐 v1.3.11-beta: Decentralized consensus (will be initialized in multi-node mode)
            production_peer_discovery: None,

            // libp2p-based zero-config peer discovery
            libp2p_discovery,

            // libp2p network command channel
            libp2p_command_tx,

            // Cached libp2p peer info (will be populated after network starts)
            libp2p_peer_info: Arc::new(RwLock::new((String::new(), vec![]))),

            // Atomic peer count (will be populated from network manager)
            libp2p_peer_count: None, // Will be initialized in main.rs after network manager creation
            network_metrics: None, // v10.9.27: wired in main.rs once UnifiedNetworkManager is built
            di_ema: Arc::new(std::sync::atomic::AtomicU64::new(0)), // v9.0.6: DI EMA smoothing
            node_signing_key: Arc::new(ed25519_dalek::SigningKey::generate(&mut rand::rngs::OsRng)), // 💱 v0.6.1-beta: DEX pool signing key (will be replaced in main.rs)
            node_cypher: Arc::new(q_eternal_cypher::NodeCypher::from_ed25519_key(ed25519_dalek::SigningKey::generate(&mut rand::rngs::OsRng))), // v7.2.12: placeholder, replaced in main.rs
            admin_wallet: crate::aegis_auth_middleware::FOUNDER_WALLET.to_string(),
            stripe_client: crate::payment_api::init_stripe_client().ok(),
            p2p_bytes_in: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            p2p_bytes_out: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            // 🔥 Top Movers ring buffer — empty deque + zero ingest height.
            recent_balance_deltas: Arc::new(RwLock::new(std::collections::VecDeque::new())),
            top_movers_last_ingested_height: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            network_throttle_mode: Arc::new(std::sync::atomic::AtomicU8::new(2)), // 2 = Turbo (default)
            highest_network_height: Arc::new(std::sync::atomic::AtomicU64::new(0)), // Sync mode tracking
            last_peer_height_update: Arc::new(std::sync::atomic::AtomicU64::new(0)), // v5.2.0: Peer height staleness
            sync_trigger: Arc::new(tokio::sync::Notify::new()), // v5.2.0: Immediate sync wake-up
            sharkgod_block_wake: Some(Arc::new(tokio::sync::Notify::new())), // 🦈 SharkGod block producer wake
            dev_fee_bps: Arc::new(std::sync::atomic::AtomicU64::new(190)), // v8.8.1: 190 bps = 1.9% mainnet dev fee
            node_operator_fee_promille: Arc::new(std::sync::atomic::AtomicU64::new(0)), // v7.3.1: disabled by default
            dex_protocol_fee_bps: Arc::new(std::sync::atomic::AtomicU64::new(5)), // v7.3.1: 5 bps = 0.05% protocol fee from swaps
            operator_fees_earned_session: Arc::new(std::sync::atomic::AtomicU64::new(0)), // v8.1.1
            operator_fees_earned_total: Arc::new(std::sync::atomic::AtomicU64::new(0)), // v8.1.1
            operator_fee_tx_count: Arc::new(std::sync::atomic::AtomicU64::new(0)), // v8.1.1
            current_height_atomic: Arc::new(std::sync::atomic::AtomicU64::new(initial_height)), // ⚡ v0.9.66-beta: Lock-free height (max-seen)
            contiguous_height_atomic: Arc::new(std::sync::atomic::AtomicU64::new(initial_height)), // v1.0.2: Honest contiguous height; refreshed every 5s
            peak_height_atomic: Arc::new(std::sync::atomic::AtomicU64::new(initial_height)), // v8.2.9: Peak height (never decreases)
            api_requests_served: Arc::new(std::sync::atomic::AtomicU64::new(0)), // v10.9.19: engine_pulse counter
            height_state: q_storage::HeightState::new(initial_height), // 🚀 v1.0.2-beta: HeightState cache - Eliminates binary search storm
            shutdown_tx: {
                let (tx, _rx) = tokio::sync::broadcast::channel(1);
                tx
            }, // 🛑 v1.0.2-beta: Graceful shutdown broadcast
            auto_update_tx: None, // 🔄 v8.5.0: Auto-update (set in main.rs after init)
            auto_update_state: None, // 🔄 v8.5.0: Auto-update state (set in main.rs after init)
            auto_update_enabled: Arc::new(std::sync::atomic::AtomicBool::new(
                std::env::var("Q_AUTO_UPDATE").unwrap_or_else(|_| "0".to_string()) == "1"
            )), // 🔄 v8.5.1: Runtime auto-update toggle
            bootstrap_wallet_sync_done: Arc::new(std::sync::atomic::AtomicBool::new(
                storage_engine.has_migration_flag(crate::BOOTSTRAP_WALLET_SYNC_FLAG).await
            )), // 🚀 v8.8.2: One-time bootstrap wallet sync (loaded from RocksDB)
            ai_active: Arc::new(std::sync::atomic::AtomicBool::new(false)), // 🤖 v9.3.3: AI inference throttle
            dex_ready: Arc::new(std::sync::atomic::AtomicBool::new(false)), // 🛡️ v10.3.1: DEX gate (starts DISABLED — enabled after reconciliation in main.rs)
            startup_sync_complete: Arc::new(std::sync::atomic::AtomicBool::new(false)), // v10.3.2: starts false, set true after authority sync
            balance_finality_engine: None, // wired after AppState construction in main.rs
            admin_notification_email: Arc::new(tokio::sync::RwLock::new(
                std::env::var("Q_ADMIN_NOTIFICATION_EMAIL").ok()
            )), // 🔄 v8.5.1: Admin notification email
            current_challenge: Arc::new(tokio::sync::RwLock::new(None)), // 🔧 v1.0.4-beta: Challenge caching
            fork_detector: Arc::new(q_storage::fork_detector::ForkDetector::new()), // 🔍 v0.9.67-beta: Comprehensive fork detection
            sync_start_time: Arc::new(std::sync::RwLock::new(None)), // 🎨 v0.6.6-beta: Progress bar sync tracking
            sync_start_height: Arc::new(std::sync::atomic::AtomicU64::new(0)), // 🎨 v0.6.6-beta: Progress bar sync tracking

            // Mining submission async queue
            mining_submission_tx: None, // Will be initialized in main.rs
            mining_submission_txs: None, // v8.9.0: Sharded pipeline (initialized in main.rs)
            mining_shard_index: None,
            mining_solutions_submitted: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            mining_solutions_accepted: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            miner_stats_tx: None,
            sse_mining_event_tx: None,
            mining_nonce_dedup: Arc::new(dashmap::DashMap::new()),
            optimistic_applied_txs: Arc::new(dashmap::DashMap::new()),

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
                        tracing::info!(
                            "✅ ZK-STARK System initialized - Zero-knowledge proofs enabled"
                        );
                        Some(Arc::new(tokio::sync::Mutex::new(system)))
                    }
                    Err(e) => {
                        tracing::warn!(
                            "⚠️ ZK-STARK System initialization failed: {}, ZK proofs unavailable",
                            e
                        );
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
            // ✅ Post-Quantum LatticeGuard SNARK (RLWE-based, no trusted setup)
            lattice_guard: {
                match q_lattice_guard::LatticeGuard::new(q_lattice_guard::SecurityLevel::PQ128) {
                    Ok(guard) => {
                        tracing::info!(
                            "✅ LatticeGuard Post-Quantum SNARK initialized - RLWE-based proofs enabled"
                        );
                        Some(Arc::new(tokio::sync::Mutex::new(guard)))
                    }
                    Err(e) => {
                        tracing::warn!(
                            "⚠️ LatticeGuard initialization failed: {}, post-quantum SNARK unavailable",
                            e
                        );
                        None
                    }
                }
            },
            lattice_guard_srs: {
                // Generate or load SRS (Structured Reference String) for LatticeGuard
                // Uses caching to avoid regenerating on every startup
                let params = q_lattice_guard::RlweParams::pq128();
                let cache_path = std::env::var("Q_DB_PATH")
                    .map(|p| std::path::PathBuf::from(p).join("srs_cache"))
                    .unwrap_or_else(|_| std::path::PathBuf::from("/tmp/q-lattice-guard-srs"));
                let mut rng = rand::thread_rng();
                match q_lattice_guard::LatticeGuardSRS::generate_or_load(params, 10000, &cache_path, &mut rng) {
                    Ok(srs) => {
                        tracing::info!(
                            "✅ LatticeGuard SRS generated - transparent setup complete (10K constraints)"
                        );
                        Some(Arc::new(srs))
                    }
                    Err(e) => {
                        tracing::warn!(
                            "⚠️ LatticeGuard SRS generation failed: {}, using default SRS",
                            e
                        );
                        None
                    }
                }
            },

            // ✅ v2.4.1-beta: TemporalShield-STARK trustee management with HSM
            temporal_trustee_manager: {
                match trustee_manager::TrusteeManager::new() {
                    Ok(mut manager) => {
                        if let Err(e) = manager.initialize() {
                            tracing::warn!(
                                "⚠️ TrusteeManager initialization failed: {}, using uninitialized state",
                                e
                            );
                        } else {
                            tracing::info!(
                                "✅ TemporalShield TrusteeManager initialized - {} HSM-backed keys ready",
                                manager.total_keys()
                            );
                        }
                        Some(Arc::new(tokio::sync::RwLock::new(manager)))
                    }
                    Err(e) => {
                        tracing::warn!(
                            "⚠️ TrusteeManager creation failed: {}, TemporalShield unavailable",
                            e
                        );
                        None
                    }
                }
            },

            // Performance & Scaling Optimizations - Initialize for maximum TPS
            simd_crypto_engine: {
                let simd_config = q_crypto_simd::SimdCryptoConfig::default();
                match q_crypto_simd::SimdCryptoEngine::new(simd_config).await {
                    Ok(engine) => {
                        tracing::info!(
                            "✅ SIMD Crypto Engine initialized - Vectorized cryptography enabled"
                        );
                        Some(Arc::new(engine))
                    }
                    Err(e) => {
                        tracing::warn!(
                            "⚠️ SIMD Crypto Engine initialization failed: {}, using fallback",
                            e
                        );
                        None
                    }
                }
            },
            #[cfg(target_os = "linux")]
            kernel_io_engine: {
                match crate::io_uring_adapter::IoUringAdapter::new() {
                    Ok(adapter) => {
                        tracing::info!(
                            "✅ Kernel I/O Engine initialized with dedicated thread pool"
                        );
                        Some(Arc::new(adapter))
                    }
                    Err(e) => {
                        tracing::warn!(
                            "⚠️ Kernel I/O Engine failed to initialize: {}, using standard I/O",
                            e
                        );
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
            production_mempool: None, // Will be initialized in main.rs for high TPS
            reliable_broadcast: None,
            quantum_vdf: None,
            quorum_commit_collector: Arc::new(quorum_commit::QuorumCommitCollector::new(3)), // 3-of-4 quorum (f=1)

            // PHASE 2: Parallel Block Producer Pool - 8 concurrent producers for exciting visualization
            // ✅ v0.9.92-beta: LOCK-FREE ARCHITECTURE - Channel-based message passing (DEADLOCK FIX)
            block_producer_pool: {
                // Create base config for all producers
                let base_config = crate::block_producer::BlockProducerConfig {
                    block_interval_secs, // v0.0.22-beta: from extracted config (2s for fast visualization)
                    max_solutions_per_block, // v0.0.22-beta: from extracted config
                    min_solutions_per_block, // v0.0.22-beta: from extracted config
                    node_id,
                    is_validator,
                    validator_index: 0, // Will be overridden by pool for each producer
                    total_validators: 8, // ✅ v1.0.17-beta: Restored to 8 (deadlock was NOT in producer contention)
                    network_id_str: std::env::var("Q_NETWORK_ID").unwrap_or_else(|_| {
                        let now = chrono::Utc::now().timestamp() as u64;
                        if now >= 1771761600 { "mainnet-genesis".to_string() } else { "mainnet2026.1.3".to_string() }
                    }),
                };

                // Create pool with 8 parallel producers for true parallelism
                let num_producers = 4; // 🔧 v8.0.9: Increased to 4 for 10+ bps target with interval=0

                // ✅ v0.9.92-beta DEADLOCK FIX: Use LOCK-FREE producer pool with channel-based architecture
                // This completely eliminates the RwLock deadlock that caused 9+ hour stalls
                // ✅ v0.9.99-beta: Now includes adaptive block rewards
                info!("🔓 Initializing LOCK-FREE Block Producer Pool (v0.9.92-beta DEADLOCK FIX)");
                let pool = crate::lockfree_producer::LockFreeProducerPool::new_with_storage(
                    num_producers,
                    base_config,
                    &storage_engine, // Pass storage Arc to load blockchain state
                    Some(balance_consensus_engine.clone()), // ✅ v0.9.99-beta: Adaptive rewards
                )
                .await?;

                info!(
                    "✅ LOCK-FREE Block Production initialized with {} producers",
                    num_producers
                );
                info!("   🔓 ZERO RwLocks - Channel-based architecture");
                info!("   ⚡ ZERO lock contention - Message passing only");
                info!("   🛡️  ZERO deadlock risk - No shared mutable state");
                info!("   ⚡ Exciting visualization: Multiple blocks will appear simultaneously in different lanes!");

                // 🚨 v0.9.9-beta CRITICAL FIX: Sync producers immediately after initialization
                // This ensures producers have the correct height even if crash recovery runs later
                // and updates the blockchain height. Without this, producers stay at stale height
                // from initialization and cause height regression errors after restart.
                if let Err(e) = pool.sync_from_storage(&storage_engine).await {
                    error!("❌ Failed to sync producers at startup: {}", e);
                }

                let pool_arc = Arc::new(pool);

                // 🚀 v1.0.3.9-beta: START STATE CONSISTENCY MONITOR (PRIMARY FIX)
                // Monitors database vs producer height every 10 seconds and auto-resyncs on divergence
                // This prevents the stale state bug where producers fall behind database state
                info!("🚀 [v1.0.3.9-beta] Starting state consistency monitor...");
                pool_arc
                    .clone()
                    .spawn_state_monitor(storage_engine.clone())
                    .await;
                info!("✅ [v1.0.3.9-beta] State monitor active - monitoring every 10s");

                pool_arc
            },

            /// 📊 v1.0.72-beta: Finality Metrics - Sub-50ms latency tracking
            finality_metrics: Arc::new(crate::block_producer::FinalityMetrics::default()),
            finality_certs: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),

            // AI Model Management - Lazy Loading (initialized later in main.rs if needed)
            ai_model_manager: None,

            // PHASE 3: DAG-Knight Consensus - Initialize with Byzantine fault tolerance
            consensus: {
                let consensus = DAGKnightConsensus::new(
                    node_id, 1, // f = 1 (supports 2f+1 = 3 nodes minimum for BFT)
                )
                .await?;

                info!(
                    "🎯 Initialized DAG-Knight consensus engine (f={}, min_nodes={})",
                    1, 3
                );

                Arc::new(RwLock::new(consensus))
            },

            // v9.3.1: K-parameter network health gauge (always-on)
            k_parameter_state: Arc::new(k_parameter_gauge::KParameterState::default()),

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
            orobit_ecosystem: orobit_ecosystem.clone(),
            symbol_to_address: Arc::new(dashmap::DashMap::new()), // v2.4.8: O(1) symbol lookup
            contract_events: Arc::new(RwLock::new(HashMap::new())), // v1.4.10: Contract event history

            // v1.4.11: Commit-reveal and stake-weighted finality for hybrid PoW/PoS security
            commit_reveal_manager: Arc::new(mining_commit_reveal::CommitRevealManager::new(true)),
            stake_finality_manager: Arc::new(stake_weighted_finality::StakeWeightedFinalityManager::new()),

            // Quillon Bank - Full Quantum Banking System with CDP
            quillon_bank,

            // v2.4.0-beta: Governance Coordinator (persistent across restarts)
            governance_coordinator,

            // AEGIS-QL Post-Quantum Authentication for Founder Operations
            aegis_auth_state,

            // QUG/QUGUSD Stablecoin System - CollateralVault
            collateral_vault,
            // v8.5.5: QCREDIT Yield Vault
            qcredit_vault,

            // Quillon Bank Loan Applications - Persistent storage
            pending_loan_applications: Arc::new(RwLock::new(pending_loan_applications_map)),

            // Distributed VM and DEX - Initialize in production mode
            distributed_protocol: None, // Initialized separately

            // OAuth2 Provider - v7.3.5: RocksDB-persisted clients (survive restarts)
            oauth2_storage: {
                let oauth2 = Arc::new(oauth2_provider::OAuth2Storage::with_storage(storage_engine.clone()));
                oauth2.load_clients_from_disk().await;
                oauth2
            },

            // v8.1.7: OAuth2 Key Vault — load encrypted signing keys from RocksDB
            oauth2_key_vault: {
                let vault_keys = storage_engine.load_vault_keys().await.unwrap_or_default();
                if !vault_keys.is_empty() {
                    tracing::info!("🔐 [VAULT] Loaded {} custodial signing keys from RocksDB", vault_keys.len());
                }
                Arc::new(RwLock::new(vault_keys))
            },

            // v7.4.0: Peer JWT public keys for cross-node token verification
            peer_jwt_keys: Arc::new(dashmap::DashMap::new()),
            peer_operator_wallets: Arc::new(dashmap::DashMap::new()),

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

            // Distributed AI Coordinator - Initialized separately
            distributed_ai_coordinator: None,

            // AI Verification System - Initialized in main.rs
            proof_verifier: None,
            benchmark_verifier: None,
            failover_manager: None,
            verification_events_tx: None,

            // Turbo Sync - Git-inspired fast blockchain synchronization (initialized in main.rs)
            turbo_sync: None,
            flight_computer: None,

            // 🚀 v1.0.4-beta: Phase 2 DAG-Aware Sync - Feature flag (set in main.rs)
            enable_dag_sync: false, // Will be set in main.rs based on Q_ENABLE_DAG_SYNC env var

            // v0.9.6-beta: Peer Registry Bridge (initialized in main.rs)
            // TEMPORARILY DISABLED v1.0.15: Circular dependency
            // peer_bridge: None,

            // AEGIS-KL Miner Authentication (initialized in main.rs)
            // miner_auth: None,

            // v0.9.1-beta: DeFi components (initialized later when needed)
            token_registry: None,
            price_history: None,
            // v0.9.1-beta: DEX components enabled
            dex_manager: None,
            price_bridge: None,

            // v2.3.34-beta: Swap history for Token Details Modal
            swap_history: Arc::new(RwLock::new(HashMap::new())),

            // v2.4.0-beta: Consensus-Verified Swap Indexer
            swap_indexer: Arc::new(swap_indexer::SwapIndexer::new(storage_engine.clone())),

            // v3.7.1-beta: Consensus-Verified Price History Indexer
            price_history_indexer: Arc::new(price_history_indexer::PriceHistoryIndexer::new(storage_engine.clone())),

            // v2.3.8-beta: Volume and price tracking for real oracle data
            volume_tracker: Arc::new(RwLock::new(HashMap::new())),
            price_snapshots: Arc::new(RwLock::new(HashMap::new())),

            // 🚀 v1.0.2-beta PHASE 1A: SAFE BATCHED SYNC - Initialized in main.rs
            fast_sync_enabled: false, // Will be set in main.rs based on CLI flag
            fast_sync_tx: None,       // Will be initialized in main.rs if enabled
            #[cfg(not(target_os = "windows"))]
            fast_sync_metrics: None,  // Will be initialized in main.rs if enabled

            // ✅ v1.0.7-beta: AsyncStorageEngine - Initialized in main.rs after DB setup
            #[cfg(not(target_os = "windows"))]
            async_storage: None, // Will be initialized in main.rs with DB handle

            // ✨ v1.0.16-beta: PQC Validator Keypair - Initialized in main.rs if --validator-key provided
            validator_keypair: None, // Will be set in main.rs after loading from file

            // ✨ v1.0.16-beta: Validator Key Registry - For PQC signature verification
            validator_key_registry: Arc::new(RwLock::new(q_types::ValidatorKeyRegistry::new())),
            validator_registry: Arc::new(RwLock::new(q_types::validator_registry::ValidatorRegistry::new())),

            // ⏰ v1.0.15-beta: Timeout-Based Sync Activation - Will be initialized in main.rs
            sync_activator: None, // Will be set in main.rs after AppState creation

            // ✨ v1.4.0-beta: Recursive Proofs Service - Will be initialized in main.rs
            recursive_proofs_service: None, // Will be set in main.rs after AppState creation

            // ✨ v1.4.2-beta: Block-Height Activated Upgrade Manager
            // Determines if network is mainnet from environment variable
            upgrade_manager: Arc::new(UpgradeManager::new(
                std::env::var("Q_MAINNET").map(|v| v == "1" || v.to_lowercase() == "true").unwrap_or(false)
            )),

            // 🔮 v1.4.2-beta: QNO Prediction Staking - Will be initialized after DB is ready
            #[cfg(not(target_os = "windows"))]
            qno_storage: Arc::new(RwLock::new(None)),

            // ⛏️ v2.3.0-beta: Stratum Mining Pool (initialized in main.rs)
            mining_pool: None,

            // 🌐 v2.3.0-beta: Decentralized Mining Pool (initialized in main.rs)
            distributed_pool_coordinator: None,
            distributed_pool_outbound_tx: None,
            distributed_pplns_proportions: Arc::new(tokio::sync::RwLock::new(None)),
            pool_hashrate_history: Arc::new(tokio::sync::RwLock::new(Vec::new())),

            // 💰 v2.4.8-beta: Dollar Cost Averaging (DCA) Storage
            dca_storage: Some(Arc::new(dca_api::DcaStorage::new())),

            // 🎯 v10.4.9: Limit Order Storage
            limit_order_storage: Some(Arc::new(limit_order_api::LimitOrderStorage::new())),

            // 📈 v2.5.0-beta: Perpetual Futures Storage
            perp_storage: Some(Arc::new(perpetual_api::PerpStorage::new())),

            // v2.4.2: Token staking & fee system - load from persistent storage
            token_fee_configs: {
                let configs = storage_engine.load_fee_configs().await.unwrap_or_default();
                if !configs.is_empty() {
                    tracing::info!("📊 Loaded {} token fee configurations from storage", configs.len());
                }
                Arc::new(RwLock::new(configs))
            },
            token_staking_positions: {
                let positions = storage_engine.load_stake_positions().await.unwrap_or_default();
                if !positions.is_empty() {
                    tracing::info!("🥩 Loaded {} staking positions from storage", positions.len());
                }
                Arc::new(RwLock::new(positions))
            },
            token_burn_totals: Arc::new(RwLock::new(HashMap::new())),
            token_reflection_totals: Arc::new(RwLock::new(HashMap::new())),
            token_social_profiles: Arc::new(RwLock::new(HashMap::new())),
            // 🚨 v3.3.3-beta: Emergency pause mechanism
            emergency_paused: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            emergency_pause_reason: Arc::new(RwLock::new(None)),
            emergency_pause_timestamp: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            // ⛏️ v9.1.4: Dynamic mining mode switch
            forced_mining_mode: Arc::new(std::sync::atomic::AtomicU8::new(0)),
            forced_pool_url: Arc::new(RwLock::new(None)),
            // 📬 v3.9.1-beta: Bank messaging and identity systems (loaded from RocksDB)
            bank_messages: {
                let mut loaded_messages = Vec::new();
                if let Ok(entries) = storage_engine.load_bank_messages().await {
                    for (_key, value) in entries {
                        if let Ok(msg) = serde_json::from_slice::<crate::quillon_bank_api::BankMessage>(&value) {
                            loaded_messages.push(msg);
                        }
                    }
                    if !loaded_messages.is_empty() {
                        tracing::info!("📬 Loaded {} bank messages from RocksDB", loaded_messages.len());
                    }
                }
                Arc::new(RwLock::new(loaded_messages))
            },
            user_identities: {
                let mut loaded_identities = Vec::new();
                if let Ok(entries) = storage_engine.load_user_identities().await {
                    for (_key, value) in entries {
                        if let Ok(id) = serde_json::from_slice::<crate::quillon_bank_api::UserIdentity>(&value) {
                            loaded_identities.push(id);
                        }
                    }
                    if !loaded_identities.is_empty() {
                        tracing::info!("🪪 Loaded {} user identities from RocksDB", loaded_identities.len());
                    }
                }
                Arc::new(RwLock::new(loaded_identities))
            },
            death_certificates: {
                let mut loaded_certs = Vec::new();
                if let Ok(entries) = storage_engine.load_death_certificates().await {
                    for (_key, value) in entries {
                        if let Ok(cert) = serde_json::from_slice::<crate::quillon_bank_api::DeathCertificate>(&value) {
                            loaded_certs.push(cert);
                        }
                    }
                    if !loaded_certs.is_empty() {
                        tracing::info!("💀 Loaded {} death certificates from RocksDB", loaded_certs.len());
                    }
                }
                Arc::new(RwLock::new(loaded_certs))
            },
            // 🔐 v4.2.0: VAULT RWA redemptions
            vault_redemptions: Arc::new(RwLock::new(Vec::new())),
            // v5.1.0: FORGE RWA redemptions
            forge_redemptions: Arc::new(RwLock::new(Vec::new())),
            // v6.5.0: Exchange Listing RWA orders
            listing_orders: Arc::new(RwLock::new(Vec::new())),
            // v8.2.8: XLIST Crowdfunding campaigns
            listing_campaigns: Arc::new(RwLock::new(Vec::new())),
            // v5.1.1: Node start time
            start_time: std::time::Instant::now(),
            // v5.1.1: Deploy admin state
            deploy_state: Arc::new(RwLock::new(deploy_admin_api::DeployState::new())),
            // v7.2.0: Miner link relay
            miner_link_registry: miner_link_api::new_registry(),
            // v7.2.0: Bitcoin bridge (initialized separately if configured)
            atomic_swap_manager: None,
            // v10.2.11: Bitcoin deposit bridge (initialized below from BTC_RPC_* env vars)
            deposit_bridge: None,
            // v9.6.5: Bitcoin Knots RPC client (initialized below)
            bitcoin_rpc_client: None,
            // v9.7.2: Zcash Zebra RPC client (initialized below)
            zcash_rpc_client: None,
            // v7.3.1: Bridge committee (peer ID set later)
            bridge_committee: Arc::new(RwLock::new(bridge_committee::BridgeCommittee::new(String::new()))),
            // v9.4.0: Bridge safety controller
            bridge_safety: Arc::new(bridge_safety::BridgeSafetyController::new()),
            // v9.5.0: Compute orchestrator (initialized later from CLI arg)
            compute_orchestrator: None,
            // v9.6.1: QR code payment requests for brick-and-mortar POS
            payment_requests: Arc::new(dashmap::DashMap::new()),
            // v10.2.0: Crown & Ash — Medieval grand strategy game state
            crown_ash_state: Arc::new(tokio::sync::RwLock::new(crown_ash_api::CrownAshGameState::empty())),
        })
    }

    /// Helper method to save wallet balance to persistent storage
    pub async fn save_wallet_balance(&self, address: &[u8; 32], amount: u128) -> anyhow::Result<()> {
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
    pub async fn init_distributed_protocol(
        &mut self,
        local_peer_id: libp2p::PeerId,
    ) -> anyhow::Result<()> {
        tracing::info!("🌐 Initializing distributed VM & DEX coordinators for horizontal scaling");

        let protocol = q_network::DistributedProtocolManager::new(local_peer_id).await?;

        self.distributed_protocol = Some(Arc::new(protocol));

        tracing::info!(
            "✅ Distributed protocol coordinators initialized - ready for multi-node collaboration"
        );
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
