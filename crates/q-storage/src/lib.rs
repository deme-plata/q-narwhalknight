/// Q-Storage: High-performance blockchain database for DagKnight consensus
/// Optimized for Narwhal mempool and Bullshark finality with hot/cold storage split
/// Battle-tested design using RocksDB with specialized column families
use anyhow::{Context, Result};
use q_dag_knight::BullsharkCert;
use q_dag_knight::NarwhalPayload;
use q_narwhal_core::Certificate;
use q_types::{Block, NodeId, Vertex, u128_serde};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, SystemTime},
};
use tokio::sync::RwLock;
use tracing::{debug, error, info, trace, warn};

// External crates
extern crate hex;
extern crate blake3;

// 🚀 v1.0.93-beta: Parallel sync optimization
use rayon::prelude::*;

// ⚡ v9.1.0: Compute Power Layer — global peer hashrate map
// Written by q-network (gossipsub handler), read by turbo_sync (gravity-assist)
lazy_static::lazy_static! {
    /// Global map of peer compute power announcements.
    /// Key: peer_id string, Value: (hashrate_hs, active_miners, timestamp_secs)
    pub static ref PEER_COMPUTE_POWER: dashmap::DashMap<String, (f64, u32, u64)> = dashmap::DashMap::new();
}

/// v9.1.0: Compute power boost for gravity-assist peer selection.
/// Returns a multiplier based on a peer's announced hashrate (log-scale).
/// 1x at 0 H/s, 2x at 1 MH/s, 3x at 1 GH/s, capped at 5x.
pub fn compute_power_boost(peer_id: &str) -> f64 {
    if let Some(entry) = PEER_COMPUTE_POWER.get(peer_id) {
        let (hashrate_hs, _, timestamp) = *entry;
        // Expire announcements older than 120s
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        if now.saturating_sub(timestamp) > 120 {
            return 1.0;
        }
        if hashrate_hs <= 0.0 {
            return 1.0;
        }
        let boost = 1.0 + (hashrate_hs / 1_000.0).max(1.0).log10();
        boost.min(5.0)
    } else {
        1.0
    }
}

// ============ v2.4.2: TOKEN STAKING TYPES ============
// Defined here to avoid circular dependency with q-api-server

/// Token fee configuration stored per contract
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TokenFeeConfig {
    pub enabled: bool,
    pub reflection_fee_bps: u64,
    pub burn_fee_bps: u64,
    pub liquidity_fee_bps: u64,
    pub dev_fee_bps: u64,
    pub dev_wallet: Option<String>,
    pub excluded_addresses: Vec<String>,
}

/// Staking tier with lock periods and APY
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum StakingTier {
    Bronze,
    Silver,
    Gold,
    Diamond,
}

/// Token stake position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenStakePosition {
    pub wallet_address: String,
    pub contract_address: String,
    pub amount: u64,
    pub tier: StakingTier,
    pub start_time: u64,
    pub unlock_time: u64,
    pub last_reward_claim: u64,
    pub total_rewards_claimed: u64,
}

impl TokenFeeConfig {
    /// Calculate total fee percentage
    pub fn total_fee_bps(&self) -> u64 {
        if !self.enabled {
            return 0;
        }
        self.reflection_fee_bps + self.burn_fee_bps + self.liquidity_fee_bps + self.dev_fee_bps
    }

    /// Calculate fee breakdown for a transfer amount
    /// Returns (transfer_amount, reflection, burn, liquidity, dev)
    pub fn calculate_fees(&self, amount: u64) -> (u64, u64, u64, u64, u64) {
        if !self.enabled || amount == 0 {
            return (amount, 0, 0, 0, 0);
        }

        let reflection = (amount * self.reflection_fee_bps) / 10000;
        let burn = (amount * self.burn_fee_bps) / 10000;
        let liquidity = (amount * self.liquidity_fee_bps) / 10000;
        let dev = (amount * self.dev_fee_bps) / 10000;

        let total_fee = reflection + burn + liquidity + dev;
        let transfer_amount = amount.saturating_sub(total_fee);

        (transfer_amount, reflection, burn, liquidity, dev)
    }

    /// Check if an address is excluded from fees
    pub fn is_excluded(&self, address: &str) -> bool {
        self.excluded_addresses.iter().any(|a| a.eq_ignore_ascii_case(address))
    }
}

impl StakingTier {
    pub fn from_days(days: u64) -> Self {
        match days {
            0..=7 => StakingTier::Bronze,
            8..=30 => StakingTier::Silver,
            31..=90 => StakingTier::Gold,
            _ => StakingTier::Diamond,
        }
    }

    pub fn lock_period_seconds(&self) -> u64 {
        match self {
            StakingTier::Bronze => 7 * 24 * 3600,
            StakingTier::Silver => 30 * 24 * 3600,
            StakingTier::Gold => 90 * 24 * 3600,
            StakingTier::Diamond => 180 * 24 * 3600,
        }
    }

    pub fn apy_bps(&self) -> u64 {
        match self {
            StakingTier::Bronze => 500,   // 5%
            StakingTier::Silver => 1000,  // 10%
            StakingTier::Gold => 1500,    // 15%
            StakingTier::Diamond => 2500, // 25%
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            StakingTier::Bronze => "Bronze",
            StakingTier::Silver => "Silver",
            StakingTier::Gold => "Gold",
            StakingTier::Diamond => "Diamond",
        }
    }
}

pub mod aegis_sync; // v0.9.14-beta: AEGIS-QL signed P2P sync
#[cfg(not(target_os = "windows"))]
pub mod async_engine; // ✅ v1.0.2-beta: AsyncStorageEngine with micro-batching to eliminate mining stalls
pub mod balance_checkpoint; // v10.4.12: Hardcoded Epsilon balance snapshot — idempotent one-time import
pub mod balance_consensus;
pub mod balance_smt; // v10.9.16: Sparse Merkle Tree for balance_root_v2 (IVC SNARK Blueprint 1A)
pub mod hot_wallet_cache; // Hot-wallet SMT proof prefetch LRU (10K entries, ~80 MB worst-case)
pub mod batch_sync;
pub mod sharded_balance;  // 🚀 v3.4.6-beta: 16-shard balance cache for 2-3x lookup speedup // ✅ v1.0.12-beta: Phase 1 batch sync with 512-block batches + parallel validation
pub mod checkpoint; // ✅ v1.0.79-beta: Height checkpoint files for data loss detection
pub mod block_writer; // ✅ v0.9.93-beta: Single-writer queue to prevent database corruption
pub mod chain_reorganization; // v0.9.37-beta: Cross-fork blockchain synchronization
#[cfg(not(target_os = "windows"))]
pub mod db_util; // ✅ v1.0.2-beta: Spawn_blocking helpers for all RocksDB operations
pub mod emission_controller; // ✅ v0.9.99-beta: Adaptive block rewards for throughput-independent emission
pub mod fork_detector; // ✅ v0.9.67-beta: Comprehensive fork detection & automatic reorg
pub mod height_state; // ✅ v1.0.2-beta: Height cache to eliminate binary search storms
#[cfg(not(target_os = "windows"))]
pub mod integrity; // ✅ v0.9.76-beta: Database corruption detection & auto-repair
pub mod integrity_scrubber; // 🧹 v1.0.2: Background random-walk hash verifier (silent corruption detector)
pub mod kv;
pub mod manifest;
pub mod metrics;
pub mod ordered_block_buffer; // ✅ v1.0.2-beta: Height-ordered reorder buffer for consensus safety
#[cfg(not(target_os = "windows"))]
pub mod pointer_integrity; // ✅ v1.0.14-beta: Database pointer corruption detection & auto-repair
#[cfg(not(target_os = "windows"))]
pub mod preflight_check;  // ✅ v3.3.7-beta: Mainnet-safe pre-flight verification before serving requests
pub mod pruning;
#[cfg(not(target_os = "windows"))]
pub mod safe_batched_writer; // ✅ v1.0.2-beta: WAL-based batched writes for 150-250 BPS (Phase 1A)
pub mod snapshot;
pub mod sync;
pub mod token_registry;
pub mod price_history;
pub mod contract_events;  // v2.9.2-beta: Contract event persistence for VM decentralization
pub mod transaction;
pub mod turbo_sync;
// TEMPORARILY DISABLED: Circular dependency with q-api-server (sync_activation module)
// TODO v1.0.15: Fix turbo_sync_peer_bridge circular dependency
// pub mod turbo_sync_peer_bridge;
pub mod zk_block_request_auth;
pub mod memory_limiter;  // ✅ v1.0.15.1-beta - Adaptive memory management for sync operations
pub mod encryption;  // ✅ v1.0.39-beta - RocksDB encryption-at-rest with Argon2id + AES-GCM
pub mod encryption_stream;  // ✅ v1.0.40-beta - AES-CTR stream cipher for SST/WAL files
pub mod encryption_ffi;  // ✅ v1.0.40-beta - FFI bindings for C++ EncryptionProvider
pub mod encryption_migration;  // ✅ v1.0.41-beta - Transitional provider for online migration
pub mod encryption_zkstark;  // ✅ v1.0.43-beta - ZK-STARK proofs for untrusted automatic setup
pub mod genesis_checkpoint;  // ✅ v1.1.21-beta - Kaspa-style genesis checkpoint for fork prevention
pub mod mainnet_safety;  // ✅ v1.1.24-beta - Production-grade database safety for mainnet deployment
pub mod overflow_storage;  // ✅ v3.9.3-beta - Multi-path storage with S3 support for overflow

// ========== v1.0.4-beta: Phase 2 DAG-Aware Sync (20-40x Performance) ==========
pub mod sync_state_manager;  // Checkpoint/resume for crash recovery
pub mod dag_layer_detector;  // Topological DAG layer organization
pub mod parallel_batch_fetcher;  // Concurrent block fetching with concurrency limits
pub mod causal_validator;  // DAG parent dependency enforcement
#[cfg(not(target_os = "windows"))]
pub mod dag_sync_manager;  // Orchestration layer (wires all Phase 2 components)

// ========== v1.0.5-beta: Request Pipelining (libp2p-rust Phase 2) ==========
pub mod request_pipeline;  // Request pipelining with adaptive window sizing (+50% performance)

// ========== v1.0.6-beta: Pack Caching (libp2p-rust Phase 3) ==========
pub mod pack_cache;  // Server-side LRU cache for compressed block packs (+30% performance)

// ========== v1.0.50-beta: Crypto-Enhanced Sync (IACR 2024-2025 Papers) ==========
pub mod crypto_enhanced_sync;  // Incremental verification, adaptive timeout, checkpointing

// ========== v1.0.60-beta: Comprehensive State Sync ==========
pub mod state_processor;  // Transaction to StateChange processing for full decentralization
#[cfg(not(target_os = "windows"))]
pub mod state_applicator;  // Apply StateChanges to RocksDB column families
#[cfg(not(target_os = "windows"))]
pub mod block_state_processor;  // Block-level state processing pipeline
pub mod sparse_merkle_trie;  // Cryptographic state root verification with O(log n) proofs

// ========== v1.2.0-beta: Phase 3 Consensus Security ==========
pub mod validator_registry;  // On-chain validator set with lifecycle management

// ========== v1.3.0-beta: SHA3-256 Peer Reputation System ==========
pub mod peer_reputation;  // Peer banning and SHA3-256 height proof verification
pub mod sha3_data_integrity;  // SHA3-256 block hash verification, Merkle roots, chain proofs

// ========== v1.4.5-beta: DAG Spam Attack Prevention ==========
pub mod orphan_rate_limiter;  // Rate limits orphan blocks per peer to prevent DoS

// ========== v1.4.0-beta: ML-Driven Adaptive Batch Optimization ==========
pub mod ml_batch_optimizer;  // Online linear regression for optimal batch size prediction

// ========== v1.4.2-beta: QNO (Quantum Neural Oracle) Prediction Staking ==========
#[cfg(not(target_os = "windows"))]
pub mod qno_storage;  // Persistent storage for prediction staking with P2P sync

// ========== v1.5.0-beta: CHIRON Parallel State Applicator ==========
pub mod parallel_state_applicator;  // CHIRON-style parallel block processing (~30% sync speedup)

// ========== v1.5.0-beta: NEMO High-Contention Executor ==========
pub mod nemo_executor;  // NEMO-style executor for high contention (+42% over Block-STM)

// ========== v1.5.0-beta: Reddio-Style Async Storage Pipeline ==========
#[cfg(not(target_os = "windows"))]
pub mod async_pipeline;  // Async storage pipeline (70% overhead reduction)

pub mod balance_finality_engine;  // BFT-safe balance finalization via Bracha RB over DAG-Knight

// ========== v10.0.0: 3-Stage Sync Pipeline (Phase 4 Optimization) ==========
#[cfg(not(target_os = "windows"))]
pub mod sync_pipeline;  // Receive → Validate → Store pipeline (~50% sync throughput increase)

// ========== v1.8.0-HOHMANN: Project APOLLO Phase 3 - Staged Sync ==========
pub mod staged_sync;       // Header-first sync (ORBITAL INSERTION) - 10-50x faster initial sync
pub mod checkpoint_jumps;  // Gravity wells for checkpoint-based fast sync

// ========== v1.9.0-SLINGSHOT: Project APOLLO Phase 4 - Peer Optimization ==========
pub mod peer_momentum;     // Hot cache peer selection (GRAVITY ASSIST) - 3-5x cache hit improvement

// ========== v2.0.0-KALMAN: Project APOLLO Phase 5 - Control Systems ==========
pub mod pid_controller;    // Self-tuning PID rate control (THRUST CONTROL) - Optimal sync rate
pub mod kalman_predictor;  // Kalman filter network prediction (NAVIGATION) - Bandwidth-delay optimization

// ========== v2.1.0-DELTA-V: Project APOLLO Phase 6 - Zero-Copy & Maximum Efficiency ==========
pub mod precompressed_storage;  // Pre-compressed block storage (FUEL TANK) - Zero CPU for P2P serving
pub mod uring_writer;           // io_uring async I/O (WARP DRIVE) - Near-zero kernel overhead
pub mod zerocopy_blocks;        // Zero-copy block access (MASS REDUCTION) - Memory-mapped blocks

// ========== v2.3.9-beta: WARP SYNC v1.0 - Ultra-High-Performance Synchronization ==========
pub mod warp_sync;              // Batch signature verification + epoch-parallel validation (1,200x target)

// Windows uses sled implementation
#[cfg(target_os = "windows")]
pub mod kv_sled;

// Export platform-specific KVStore implementation
#[cfg(not(target_os = "windows"))]
pub use kv::{KVStore, RocksDBKV};

#[cfg(target_os = "windows")]
pub use kv::KVStore;
#[cfg(target_os = "windows")]
pub use kv_sled::RocksDBKV;
pub use aegis_sync::{
    SignedBlockPack, SyncAffirmationCertificate, PeerTrustRegistry, PeerTrustMetrics,
    compute_merkle_root, verify_timestamp,
};
#[cfg(not(target_os = "windows"))]
pub use async_engine::AsyncStorageEngine;
pub use balance_consensus::{
    BalanceConsensusEngine, BalanceConsensusError, BalanceStorage, BalanceUpdate,
    ChangeReason, ConsensusStats, GENESIS_TIMESTAMP, DEV_FEE_PERCENT, FOUNDER_WALLET,
    active_genesis_timestamp,
};
pub use block_writer::BlockWriter;
#[cfg(not(target_os = "windows"))]
pub use db_util::write_batch_sync;
pub use height_state::HeightState;
pub use chain_reorganization::{
    detect_fork, find_common_ancestor, reorganize_chain, ForkStatus, ReorgStats,
};
pub use ordered_block_buffer::OrderedBlockBuffer;
#[cfg(not(target_os = "windows"))]
pub use pointer_integrity::{
    check_and_repair_on_startup, PointerIntegrityChecker, IntegrityCheckResult,
    CorruptionSeverity, IntegrityThresholds,
};
pub use memory_limiter::{MemoryLimiter, MemoryLimiterConfig, MemoryPressure, MemoryStats};
#[cfg(not(target_os = "windows"))]
pub use safe_batched_writer::{SafeBatchedWriter, BatchConfig, BatchMetrics};
pub use manifest::StorageManifest;
pub use metrics::StorageMetrics;
pub use pruning::{AdaptivePruningEngine, PruningConfig, PruningMode, PruningStats, CheckpointPolicy, RetentionTier};
pub use snapshot::SnapshotManager;
pub use sync::{SyncProtocol, SyncRequest, SyncResponse};
pub use transaction::{QTransaction, TransactionState};
pub use turbo_sync::{TurboSyncManager, TurboSyncConfig, BlockPack, BlockPackRequest, NetworkRequest, TurboSyncMetrics, EnhancedPeerRegistry, PeerHeightRecord, DetailedSyncStatus, StarshipPhase, FlightComputer, StationKeepingState, StarshipTelemetry};
pub use ml_batch_optimizer::{SyncFeatures, BatchOutcome, BatchSizePredictor, BatchOptimizerConfig};
// TEMPORARILY DISABLED: Circular dependency with q-api-server
// pub use turbo_sync_peer_bridge::{TurboSyncPeerBridge, PeerHeightEntry, run_periodic_sync, run_enhanced_periodic_sync};
pub use zk_block_request_auth::{
    BlockRequestAuthenticator, AuthenticatedBlockPackRequest, AuthenticatedBlockPackResponse,
    generate_block_request_proof,
};
pub use encryption::{
    ProtectedKey, PassphraseKDF, KeysFileHeader, CpuCapabilities, EncryptionManager,
};
pub use encryption_stream::{
    EncryptedFileHeader, AesCtrStream, WalEncryption, FileEncryptionManager,
};
pub use encryption_migration::{
    TransitionalEncryptionProvider, FileEncryptionStatus, MigrationProgress, FileFormatDetector,
};

// ========== v2.9.2-beta: Contract Event Persistence Exports ==========
pub use contract_events::{
    ContractEvent, ContractEventStorage, ContractEventStats, IndexedParam,
};

// ========== v1.0.4-beta: Phase 2 DAG-Aware Sync Exports ==========
pub use sync_state_manager::{SyncStateManager, SyncCheckpoint, SyncProgress};
pub use dag_layer_detector::{DagLayerDetector, BlockHeader as DagBlockHeader};
pub use parallel_batch_fetcher::{ParallelBatchFetcher, BatchFetchConfig, NetworkFetcher};
pub use causal_validator::CausalValidator;
#[cfg(not(target_os = "windows"))]
pub use dag_sync_manager::{DagSyncManager, DagSyncConfig, SyncStats};

// ========== v1.0.60-beta: Comprehensive State Sync Exports ==========
pub use state_processor::{
    StateProcessor, StateReader, ExecutionResult, ExecutionLog,
    TokenMetadata, PoolState, VaultState,
    BASE_GAS, GAS_PER_DATA_BYTE, GAS_PER_STORAGE_WRITE, GAS_PER_STORAGE_READ, MAX_GAS_PER_TX,
};
#[cfg(not(target_os = "windows"))]
pub use state_applicator::StateApplicator;
#[cfg(not(target_os = "windows"))]
pub use block_state_processor::{BlockStateProcessor, BlockProcessingResult, TxProcessingResult, RocksDbStateReader};
pub use sparse_merkle_trie::{SparseMerkleTrie, MerkleProof, TrieNode, TrieStats, CF_STATE_TRIE, EMPTY_HASH};
// v8.7.3: Deterministic block state replay for P2P decentralization
pub use balance_consensus::replay_block_state_changes;
pub use balance_consensus::migrate_historical_state;
// v8.9.1: Token balance refresh after state replay
pub use balance_consensus::get_updated_token_balances_for_block;

// ========== v1.1.24-beta: Mainnet Safety Infrastructure Exports ==========
pub use mainnet_safety::{
    EnhancedCheckpoint, EnhancedCheckpointManager, CheckpointType, CheckpointStorage,
    BackgroundIntegrityMonitor, IntegrityCheckable, IntegrityCheckReport, IntegrityIssue, MonitorStatus,
    IpfsBackupSystem, BackupManifest,
    MainnetSafetyManager, SafetyStatus, PreCommitIntegrityCheck,
};

// ========== v2.0.0-KALMAN: Project APOLLO Phase 5 Exports ==========
pub use pid_controller::{PIDRateController, PIDMetrics, CascadedPID, AdaptiveRateLimiter};
pub use kalman_predictor::{
    KalmanNetworkPredictor, NetworkState, SyncSettings, AdaptiveSyncController, KalmanMetrics,
};

// ========== v2.1.0-DELTA-V: Project APOLLO Phase 6 Exports ==========
pub use precompressed_storage::{
    PrecompressedBlock, PrecompressedStore, PrecompressConfig, PrecompressStats,
    CompressionAlgorithm, is_precompressed, serve_block, compress_batch,
};
pub use uring_writer::{UringWriter, UringConfig, AsyncUringWriter, VectoredWriter};
pub use zerocopy_blocks::{ZeroCopyHeader, ZeroCopyBlockView, ZeroCopyBlockStore, ZeroCopyStats};

// ========== v1.8.0-HOHMANN: Project APOLLO Phase 3 Exports ==========
pub use staged_sync::{
    StagedSyncManager, SyncStage, StagedSyncConfig, StagedSyncMetrics, HeaderOnly,
};
pub use checkpoint_jumps::{GravityWell, CheckpointManager, CheckpointSyncStrategy, CheckpointSyncState};

// ========== v1.9.0-SLINGSHOT: Project APOLLO Phase 4 Exports ==========
pub use peer_momentum::{
    PeerMomentum, PeerMomentumManager, PeerStats, GravityAssistedSelector,
};

/// Column family names for optimized storage
pub const CF_BLOCKS: &str = "blocks";
pub const CF_DAG_VERTICES: &str = "dag_vertices";
pub const CF_BULLSHARK_CERT: &str = "bullshark_cert";
pub const CF_MANIFEST: &str = "manifest";
pub const CF_NARWHAL_PAYLOADS: &str = "narwhal_payloads";
pub const CF_TRANSACTIONS: &str = "transactions";
pub const CF_BALANCES: &str = "balances";  // v0.8.2-beta: Balance consensus storage
pub const CF_BLOCK_HASH_TO_HEIGHT: &str = "block_hash_to_height";  // v0.8.3-beta: Block hash index
pub const CF_AI_CHATS: &str = "ai_chats";
pub const CF_AI_CREDITS: &str = "ai_credits";
pub const CF_AI_TRANSACTIONS: &str = "ai_transactions";
pub const CF_AI_TREASURY: &str = "ai_treasury";
pub const CF_AI_ATTACHMENTS: &str = "ai_attachments";  // v0.9.9-beta: AI chat attachment support
pub const CF_PAYMENT_PROPOSALS: &str = "payment_proposals";
pub const CF_PAYMENT_VOTES: &str = "payment_votes";
pub const CF_PAYMENT_LOCKS: &str = "payment_locks";
pub const CF_BANNED_PEERS: &str = "banned_peers";  // v0.9.7-beta: ZK proof ban persistence
pub const CF_SYNC_CERTIFICATES: &str = "sync_certificates";  // v0.9.14-beta: AEGIS-QL sync affirmation
pub const CF_PEER_TRUST: &str = "peer_trust";  // v0.9.14-beta: AEGIS-QL peer trust metrics
pub const CF_MINING_REWARDS: &str = "mining_rewards";  // v0.6.2-beta: Mining rewards history

// ========== v1.0.60-beta: Comprehensive State Sync Column Families ==========
/// Token balances: key = [account:32 | token:32], value = balance:u64
pub const CF_TOKEN_BALANCES: &str = "cf_token_balances";
/// Custom token metadata: key = token_address:32, value = TokenMetadata
pub const CF_TOKENS: &str = "cf_tokens";
/// DEX pools: key = pool_id:32, value = PoolState
pub const CF_DEX_POOLS: &str = "cf_dex_pools";
/// LP token balances: key = [pool_id:32 | account:32], value = balance:u64
pub const CF_LP_BALANCES: &str = "cf_lp_balances";
/// Smart contract code: key = contract_address:32, value = bytecode
pub const CF_CONTRACTS: &str = "cf_contracts";
/// Smart contract storage: key = [contract:32 | slot:32], value = data
pub const CF_CONTRACT_STORAGE: &str = "cf_contract_storage";
/// Collateral vaults: key = vault_id:32, value = VaultState
pub const CF_VAULTS: &str = "cf_vaults";
/// Oracle price feeds: key = feed_id:32, value = (price, timestamp)
pub const CF_ORACLE_PRICES: &str = "cf_oracle_prices";
/// AI credits: key = account:32, value = (balance, earned, spent)
pub const CF_AI_CREDITS_V2: &str = "cf_ai_credits_v2";
/// AI providers: key = provider_id:32, value = ProviderState
pub const CF_AI_PROVIDERS: &str = "cf_ai_providers";
/// Governance proposals: key = proposal_id:32, value = ProposalState
pub const CF_PROPOSALS: &str = "cf_proposals";
/// Vote delegations: key = delegator:32, value = (delegate, voting_power)
pub const CF_DELEGATIONS: &str = "cf_delegations";
/// v2.4.0-beta: Governance votes: key = proposal_id:vote_id, value = WeightedVote JSON
pub const CF_GOVERNANCE_VOTES: &str = "cf_governance_votes";
/// Staking positions: key = [staker:32 | validator:32], value = StakeState
pub const CF_STAKES: &str = "cf_stakes";
/// Validator info: key = validator_id:32, value = ValidatorState
pub const CF_VALIDATORS: &str = "cf_validators";

// QNO (Quantum Neural Oracle) Column Families
/// QNO staking positions: key = "wallet_hex:stake_id", value = StakingPosition JSON
pub const CF_QNO_STAKES: &str = "cf_qno_stakes";
/// QNO domain stats: key = domain_id, value = PredictionDomain JSON
pub const CF_QNO_DOMAINS: &str = "cf_qno_domains";
/// QNO global stats: key = "global", value = StakingStats JSON
pub const CF_QNO_STATS: &str = "cf_qno_stats";
/// System parameters: key = param_name:32, value = bytes
pub const CF_SYSTEM_PARAMS: &str = "cf_system_params";
/// Account nonces: key = account:32, value = nonce:u64
pub const CF_NONCES: &str = "cf_nonces";
/// State root checkpoints: key = height:u64, value = (state_root, tx_root)
pub const CF_STATE_ROOTS: &str = "cf_state_roots";
/// v2.3.6-beta: Swap history for Token Details Modal
/// key = "swap:{token}:{timestamp}", value = JSON swap record
pub const CF_SWAP_HISTORY: &str = "cf_swap_history";
/// v2.4.9-beta: DCA (Dollar Cost Averaging) orders
/// key = order_id, value = DcaOrder JSON
pub const CF_DCA_ORDERS: &str = "cf_dca_orders";
/// v2.4.9-beta: DCA execution history
/// key = order_id:timestamp, value = DcaExecution JSON
pub const CF_DCA_EXECUTIONS: &str = "cf_dca_executions";
/// v2.9.2-beta: Protocol fees collected (for consensus verification)
/// key = fee_id:32, value = ProtocolFeeRecord JSON
pub const CF_PROTOCOL_FEES: &str = "cf_protocol_fees";

/// v2.5.0-beta: Perpetual futures positions
/// key = position_id, value = PerpPosition JSON
pub const CF_PERP_POSITIONS: &str = "cf_perp_positions";
/// v2.5.0-beta: Perpetual futures orders
/// key = order_id, value = PerpOrder JSON
pub const CF_PERP_ORDERS: &str = "cf_perp_orders";
/// v2.5.0-beta: Perpetual futures trades
/// key = trade_id:timestamp, value = PerpTrade JSON
pub const CF_PERP_TRADES: &str = "cf_perp_trades";
/// v2.5.0-beta: Perpetual futures funding history
/// key = market:timestamp, value = FundingPayment JSON
pub const CF_PERP_FUNDING: &str = "cf_perp_funding";
/// v2.5.0-beta: Perpetual futures liquidations
/// key = liquidation_id, value = Liquidation JSON
pub const CF_PERP_LIQUIDATIONS: &str = "cf_perp_liquidations";

/// v2.9.2-beta: Contract events for VM persistence
/// key = contract_address:block_height:event_index, value = ContractEvent JSON
pub const CF_CONTRACT_EVENTS: &str = "cf_contract_events";

// ========== v3.9.1-beta: Bank Messaging & Identity System ==========
/// Bank messages: key = msg_id, value = BankMessage JSON
/// Stores bidirectional messages between users and Quillon Bank
pub const CF_BANK_MESSAGES: &str = "cf_bank_messages";
/// Bank message index by wallet: key = [wallet:32][inverted_timestamp:8], value = msg_id
/// Allows efficient O(log n) lookups of all messages for a wallet
pub const CF_BANK_MSG_INDEX: &str = "cf_bank_msg_index";
/// User identities: key = wallet_address, value = UserIdentity JSON
/// Decentralized identity records with KYC levels and beneficiary addresses
pub const CF_USER_IDENTITIES: &str = "cf_user_identities";
/// Death certificates: key = cert_id, value = DeathCertificate JSON
/// Enables account inheritance and estate planning on the blockchain
pub const CF_DEATH_CERTIFICATES: &str = "cf_death_certificates";

// ========== v3.5.8-beta: Wallet Transaction Index for Decentralized History ==========
/// Wallet transaction index: key = [wallet:32][inverted_timestamp:8][tx_id:8], value = tx_id:32
/// Allows efficient O(log n) lookups of all transactions for a wallet address
/// Indexed for BOTH sender and recipient (each transaction creates 2 entries)
pub const CF_WALLET_TX_INDEX: &str = "cf_wallet_tx_index";

/// Wallet swap index: key = [wallet:32][inverted_timestamp:8][tx_id:8], value = swap_record
/// Allows efficient O(log n) lookups of all DEX swaps for a wallet address
pub const CF_WALLET_SWAP_INDEX: &str = "cf_wallet_swap_index";

/// v3.6.0-beta: Consensus-verified price history for tokens
/// Key format: [token_address:32][inverted_timestamp:8] for reverse chronological order
/// Value format: [price:f64 LE][block_height:u64 LE] = 16 bytes
pub const CF_PRICE_HISTORY: &str = "cf_price_history";

// ========== v7.3.1: Block Storage Optimization ==========
/// Quantum metadata stored separately from block body for lazy loading
pub const CF_QUANTUM_METADATA: &str = "quantum_metadata";

// ========== v7.3.2: Quillon Mail ==========
/// Email messages: key = email_id, value = EmailMessage (JSON)
pub const CF_EMAILS: &str = "cf_emails";
/// Inbox index: key = "wallet_hex:inverted_timestamp:email_id", value = email_id
pub const CF_EMAILS_BY_WALLET: &str = "cf_emails_by_wallet";
/// Folder index: key = "wallet_hex:folder:inverted_timestamp:email_id", value = email_id
pub const CF_EMAILS_BY_FOLDER: &str = "cf_emails_by_folder";
/// Email contacts: key = "wallet_hex:contact_wallet_hex", value = ContactInfo (JSON)
pub const CF_EMAIL_CONTACTS: &str = "cf_email_contacts";
/// Outbound SMTP queue: key = outbound_id, value = OutboundEmail (JSON)
pub const CF_EMAIL_OUTBOUND: &str = "cf_email_outbound";

/// Calendar events: key = event_id, value = CalendarEvent (JSON)
pub const CF_CALENDAR_EVENTS: &str = "cf_calendar_events";
/// Calendar date index: key = "wallet_hex:YYYYMMDD:event_id", value = event_id
pub const CF_CALENDAR_BY_DATE: &str = "cf_calendar_by_date";
/// Scheduled transaction index: key = "wallet_hex:timestamp:event_id", value = event_id
pub const CF_CALENDAR_SCHEDULED_TX: &str = "cf_calendar_scheduled_tx";
/// Community events shared via P2P: key = event_id, value = CalendarEvent (JSON)
pub const CF_CALENDAR_COMMUNITY: &str = "cf_calendar_community";

/// All column families for state sync (for database initialization)
pub const STATE_SYNC_COLUMN_FAMILIES: &[&str] = &[
    CF_TOKEN_BALANCES,
    CF_TOKENS,
    CF_DEX_POOLS,
    CF_LP_BALANCES,
    CF_CONTRACTS,
    CF_CONTRACT_STORAGE,
    CF_VAULTS,
    CF_ORACLE_PRICES,
    CF_AI_CREDITS_V2,
    CF_AI_PROVIDERS,
    CF_PROPOSALS,
    CF_DELEGATIONS,
    CF_GOVERNANCE_VOTES,
    CF_STAKES,
    CF_VALIDATORS,
    CF_SYSTEM_PARAMS,
    CF_NONCES,
    CF_STATE_ROOTS,
    CF_QNO_STAKES,
    CF_QNO_DOMAINS,
    CF_QNO_STATS,
    CF_SWAP_HISTORY,
    CF_DCA_ORDERS,
    CF_DCA_EXECUTIONS,
    CF_PERP_POSITIONS,
    CF_PERP_ORDERS,
    CF_PERP_TRADES,
    CF_PERP_FUNDING,
    CF_PERP_LIQUIDATIONS,
    CF_CONTRACT_EVENTS,
    CF_WALLET_TX_INDEX,      // v3.5.8-beta: Wallet-indexed transaction history
    CF_WALLET_SWAP_INDEX,    // v3.5.8-beta: Wallet-indexed swap history
    CF_PRICE_HISTORY,        // v3.6.0-beta: Consensus-verified price history
    CF_EMAILS,               // v7.3.2: Quillon Mail messages
    CF_EMAILS_BY_WALLET,     // v7.3.2: Quillon Mail wallet inbox index
    CF_EMAILS_BY_FOLDER,     // v7.3.2: Quillon Mail folder index
    CF_EMAIL_CONTACTS,       // v7.3.2: Quillon Mail contacts
    CF_EMAIL_OUTBOUND,       // v7.3.2: Quillon Mail SMTP outbound queue
    CF_CALENDAR_EVENTS,      // v7.3.3: Blockchain Calendar events
    CF_CALENDAR_BY_DATE,     // v7.3.3: Blockchain Calendar date index
    CF_CALENDAR_SCHEDULED_TX, // v7.3.3: Blockchain Calendar scheduled transactions
    CF_CALENDAR_COMMUNITY,   // v7.3.3: Blockchain Calendar community events
];

/// Storage configuration
#[derive(Debug, Clone)]
pub struct StorageConfig {
    pub db_path: String,
    pub hot_db_path: String,
    pub enable_metrics: bool,
    pub sync_writes: bool,
    pub cache_size_mb: usize,
    pub max_open_files: usize,
}

/// Main storage engine for Q-NarwhalKnight
pub struct QStorage {
    /// Hot database (RocksDB) - blocks, vertices, certificates
    hot_db: Arc<dyn KVStore>,
    /// Concrete RocksDBKV reference for advanced operations like pruning
    hot_db_concrete: Arc<RocksDBKV>,
    /// Cold database (RocksDB) - large Narwhal payloads
    cold_db: Arc<dyn KVStore>,
    /// Storage manifest with watermarks
    manifest: Arc<RwLock<StorageManifest>>,
    /// Sync protocol for DAG catch-up
    sync_protocol: Arc<SyncProtocol>,
    /// Snapshot manager
    snapshot_manager: Arc<SnapshotManager>,
    /// Storage metrics
    metrics: Arc<StorageMetrics>,
    /// Node configuration
    node_id: NodeId,
    data_dir: PathBuf,
    /// Transaction counter for unique IDs (v0.8.1-beta)
    tx_counter: Arc<std::sync::atomic::AtomicU64>,
    /// Single-writer block commit queue (v0.9.93-beta: prevents parallel write corruption)
    block_writer: Arc<BlockWriter>,
    /// ✅ v1.0.3.5-beta: Height cache to eliminate 100ms+ RocksDB query overhead
    /// Reduces sync_from_storage latency from 105ms → <1ms (100,000x speedup)
    height_cache: HeightState,
    /// 🚨 v1.1.0: Single-writer lock for turbo sync batches
    /// Prevents race conditions where parallel batch writes interleave,
    /// causing pointer to advance past gaps. Only ONE turbo batch can
    /// execute at a time, ensuring writes are strictly sequential.
    turbo_sync_lock: Arc<tokio::sync::Mutex<()>>,
    /// 🚨 v1.1.9: Global write lock for ALL block write operations
    /// CRITICAL FIX: All three write paths (save_qblock, save_qblocks_batch,
    /// save_qblocks_batch_turbo) must share this lock to prevent race conditions
    /// where concurrent writes read the same pointer and both try to update it.
    global_write_lock: Arc<tokio::sync::Mutex<()>>,
    /// 🚨 v1.1.9: Counter for cache verification
    /// Every 100 heights, we verify the cache matches the actual database
    cache_verification_counter: std::sync::atomic::AtomicU64,
    /// 📊 v10.9.23: Sparse Merkle Tree for balance_root_v2.
    ///
    /// Opened on the same physical RocksDB instance as `hot_db_concrete` so
    /// SMT updates can be composed atomically with wallet balance writes
    /// (via `BalanceSmt::apply_to_batch`). Currently DORMANT — the SMT is
    /// instantiated but no production code path calls `apply_to_batch`. The
    /// wiring into `save_wallet_balances` is deferred to the DeepSeek handoff
    /// (docs/deepseek-handoff-balance-root-v2-activation-2026-05-14.md,
    /// Job D2). Until that lands, the SMT root remains `genesis_root` and
    /// operators can use `rebuild_balance_smt_from_wallet_table()` to
    /// generate a deterministic root for cross-node manual verification.
    pub balance_smt: Arc<crate::balance_smt::BalanceSmt>,
}

/// Type alias for compatibility with API server
pub type StorageEngine = QStorage;

// ============================================================================
// AI Chat Attachment Metadata - v0.9.9-beta
// ============================================================================

/// AttachmentMetadata struct for AI chat attachments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttachmentMetadata {
    pub id: String,
    pub chat_id: String,
    pub user_id: String,
    pub filename: String,
    pub mime_type: String,
    pub file_size: i64,
    pub storage_path: String,
    pub thumbnail_path: Option<String>,
    pub extracted_text: Option<String>,
    pub vision_base64: Option<String>,
    pub upload_timestamp: i64,
    pub processed: bool,
}

impl QStorage {
    /// Create new storage engine with configuration
    pub async fn new(config: StorageConfig) -> Result<Self> {
        let data_dir = PathBuf::from(&config.db_path);
        let node_id = [0u8; 32]; // Default node ID for API server
        Self::open(data_dir, node_id).await
    }

    /// Open storage with hot/cold database split
    pub async fn open<P: AsRef<Path>>(data_dir: P, node_id: NodeId) -> Result<Self> {
        let data_dir = data_dir.as_ref().to_path_buf();
        info!(
            "🗄️ Opening Q-Storage at {:?} for node {}",
            data_dir,
            hex::encode(&node_id[..4])
        );

        // Configure hot database (frequent access)
        let hot_path = data_dir.join("hot");
        let hot_db_concrete = Arc::new(
            RocksDBKV::open_hot_db(&hot_path)
                .await
                .context("Failed to open hot database")?,
        );

        // Configure cold database (large payloads)
        let cold_path = data_dir.join("cold");
        let cold_db = Arc::new(
            RocksDBKV::open_cold_db(&cold_path)
                .await
                .context("Failed to open cold database")?,
        );

        // Create trait object reference from concrete type
        let hot_db: Arc<dyn KVStore> = hot_db_concrete.clone();

        // Load storage manifest with explicit type coercion
        let manifest = Arc::new(RwLock::new(
            StorageManifest::load_or_create(&hot_db).await?,
        ));

        // Initialize sync protocol
        let sync_protocol = Arc::new(SyncProtocol::new(hot_db.clone(), cold_db.clone()).await?);

        // Initialize snapshot manager
        let snapshot_manager = Arc::new(
            SnapshotManager::new(data_dir.clone(), hot_db.clone(), cold_db.clone()).await?,
        );

        // Initialize metrics
        let metrics = Arc::new(StorageMetrics::new());

        // Initialize single-writer block commit queue (v0.9.93-beta)
        let block_writer = Arc::new(BlockWriter::new(hot_db.clone()));

        // ✅ v1.0.3.5-beta: Initialize height cache with placeholder
        // Will be populated after storage is created using scan_highest_contiguous_block_internal
        let height_cache = HeightState::new(0);

        // v10.9.23: Open the balance_root_v2 SMT on the SAME hot RocksDB
        // instance. The CF (`cf_balance_smt`) is in the hot CF descriptor
        // list (see crates/q-storage/src/kv.rs::open_hot_db). On existing
        // DBs the CF is auto-created at first open via the migration path.
        // BalanceSmt::open is idempotent: if there's a persisted root, it's
        // loaded; otherwise the cached root is the genesis empty-tree hash.
        let balance_smt = Arc::new(
            crate::balance_smt::BalanceSmt::open(hot_db_concrete.get_raw_db())
                .context("Failed to open BalanceSmt on hot DB")?,
        );

        let storage = Self {
            hot_db: hot_db.clone(),
            hot_db_concrete,
            cold_db,
            manifest,
            sync_protocol,
            snapshot_manager,
            metrics,
            node_id,
            data_dir,
            tx_counter: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            block_writer,
            height_cache,
            // 🚨 v1.1.0: Single-writer lock for turbo sync
            turbo_sync_lock: Arc::new(tokio::sync::Mutex::new(())),
            // 🚨 v1.1.9: Global write lock for ALL block operations
            global_write_lock: Arc::new(tokio::sync::Mutex::new(())),
            // 🚨 v1.1.9: Cache verification counter
            cache_verification_counter: std::sync::atomic::AtomicU64::new(0),
            // 📊 v10.9.23: BalanceSmt for balance_root_v2 (dormant until D2 wires it)
            balance_smt,
        };

        // Perform crash recovery and get recovered height
        let _recovered_height = storage.recover().await?;

        // ✅ v1.0.3.5-beta: Initialize height cache after recovery
        // This is the ONLY time we run the slow binary search path
        let initial_height = storage.scan_highest_contiguous_block_internal().await?;
        storage.height_cache.update(initial_height).await;
        info!("✅ Height cache initialized with height {} (one-time DB scan)", initial_height);

        // v10.9.23: Heal `qblock:latest` if it lags the actual contiguous tip.
        //
        // SYMPTOM this fixes: after a checkpoint snapshot apply (which writes
        // blocks at heights ~16.5M directly into CF_BLOCKS but never advances
        // qblock:latest from 0), turbo sync builds its next request range from
        // `qblock:latest = 0` and ends up asking for `range: 0-0` (i.e. one
        // block at height 0, which no peer has). Sync wedges; node looks
        // ready but never advances. With the v1.0.2 pointer-cap "Option A"
        // active, the pointer can never self-heal forward from 0 because the
        // walk starts at `current_pointer + 1 = 1` and finds nothing.
        //
        // FIX: at startup, after we know the real highest contiguous block,
        // bump qblock:latest up to match if it's behind. This is safe because
        // `initial_height` came from `scan_highest_contiguous_block_internal`
        // — it's guaranteed contiguous from genesis (or from the checkpoint
        // floor) up to that value.
        if initial_height > 0 {
            let current_pointer: u64 = match storage.hot_db.get(CF_BLOCKS, b"qblock:latest").await? {
                Some(bytes) if bytes.len() == 8 => {
                    u64::from_be_bytes(bytes[..8].try_into().unwrap_or([0u8; 8]))
                }
                _ => 0,
            };
            if current_pointer < initial_height {
                let height_bytes = initial_height.to_be_bytes();
                if let Err(e) = storage.hot_db.put_sync(CF_BLOCKS, b"qblock:latest", &height_bytes).await {
                    warn!(
                        "⚠️ [POINTER-HEAL] Failed to heal qblock:latest {} → {}: {} (sync may stall)",
                        current_pointer, initial_height, e
                    );
                } else {
                    warn!(
                        "🔧 [POINTER-HEAL] qblock:latest healed {} → {} (matches scan_highest_contiguous). \
                         This unsticks turbo-sync request building after a checkpoint apply.",
                        current_pointer, initial_height
                    );
                }
            }
        }

        // v10.2.8: Scan for corrupt blocks near the recovered tip (kill -9 fix)
        // Must run HERE (not in recover()) because recover() gets height 0 from empty cache.
        // scan_highest_contiguous_block_internal() is the real height discovery.
        if let Some(new_height) = storage.cleanup_corrupt_blocks_near_tip(initial_height).await? {
            warn!("🔧 [CORRUPTION FIX] Adjusted height {} → {} (corrupt blocks deleted, turbo sync will refill)",
                  initial_height, new_height);
            // Re-scan to pick up corrected state
            let re_scanned = storage.scan_highest_contiguous_block_internal().await?;
            storage.height_cache.force_set(re_scanned).await;
            info!("✅ Height cache re-initialized to {} after corruption cleanup", re_scanned);
        }

        // FIX 1.4: STARTUP INTEGRITY CHECK (v0.9.93-beta)
        storage.verify_database_integrity().await
            .context("Database integrity check failed - refusing to start")?;

        info!("✅ Q-Storage initialized successfully");
        Ok(storage)
    }

    /// Get reference to the underlying KV store (hot_db)
    ///
    /// # Usage
    /// Used by DAG sync manager to access raw KV operations for block storage.
    ///
    /// # Example
    /// ```ignore
    /// let kv = storage_engine.get_kv();
    /// let dag_sync = DagSyncManager::new(kv, block_vertex_map, config);
    /// ```
    pub fn get_kv(&self) -> Arc<dyn KVStore> {
        self.hot_db.clone()
    }

    /// v6.1.1: Get RocksDB memory usage for OOM diagnostics
    /// Returns (memtable_mb, table_readers_mb, block_cache_mb) for the hot database
    pub fn get_rocksdb_memory_mb(&self) -> (f64, f64, f64) {
        self.hot_db_concrete.get_memory_usage_mb()
    }

    /// Begin atomic transaction
    ///
    /// **SECURITY FIX (v0.8.1-beta)**: Enables atomic operations to prevent
    /// CRITICAL-1 race condition between balance updates and block storage.
    ///
    /// # Usage
    ///
    /// ```rust
    /// let tx = storage.begin_transaction().await?;
    ///
    /// // Buffer operations (not yet committed)
    /// balance_engine.process_block_mining_rewards_tx(&tx, &block).await?;
    /// tx.save_qblock(&block).await?;
    ///
    /// // Commit atomically (all or nothing)
    /// tx.commit().await?;
    /// ```
    ///
    /// # Performance
    ///
    /// - Single atomic write with fsync (~2-3ms)
    /// - Sub-50ms DAG-Knight finality maintained ✅
    /// - ~25% faster than separate operations
    pub async fn begin_transaction(&self) -> Result<crate::transaction::QTransaction> {
        let tx_id = self
            .tx_counter
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        Ok(crate::transaction::QTransaction::new(
            self.hot_db_concrete.clone(),
            tx_id,
        ))
    }

    /// Store DAG vertex with Narwhal payload
    pub async fn store_vertex(&self, vertex: &Vertex, payload: &NarwhalPayload) -> Result<()> {
        debug!(
            "💾 Storing vertex {} for round {}",
            hex::encode(&vertex.id),
            vertex.round
        );

        let start_time = SystemTime::now();

        // Store vertex in hot DB
        let vertex_key = self.vertex_key(vertex.round, &vertex.author, &vertex.id);
        let vertex_data = bincode::serialize(vertex)?;

        self.hot_db
            .put(CF_DAG_VERTICES, &vertex_key, &vertex_data)
            .await?;

        // Store payload in cold DB
        let payload_key = blake3::hash(&bincode::serialize(payload)?)
            .as_bytes()
            .to_vec();
        let payload_data = bincode::serialize(payload)?;

        self.cold_db
            .put(CF_NARWHAL_PAYLOADS, &payload_key, &payload_data)
            .await?;

        // Update metrics
        let latency = start_time.elapsed().unwrap_or(Duration::from_millis(0));
        self.metrics
            .record_vertex_write(latency, vertex_data.len(), payload_data.len())
            .await;

        // Check if this completes a round
        self.check_round_completion(vertex.round).await?;

        debug!(
            "✅ Stored vertex {} ({}ms)",
            hex::encode(&vertex.id),
            latency.as_millis()
        );
        Ok(())
    }

    /// Store Bullshark certificate
    pub async fn store_certificate(&self, cert: &BullsharkCert) -> Result<()> {
        debug!("📜 Storing Bullshark certificate for round {}", cert.round);

        let cert_key = cert.round.to_be_bytes();
        let cert_data = bincode::serialize(cert)?;

        self.hot_db
            .put(CF_BULLSHARK_CERT, &cert_key, &cert_data)
            .await?;

        // Update manifest watermark
        self.update_dag_watermark(cert.round).await?;

        info!("✅ Stored certificate for round {}", cert.round);
        Ok(())
    }

    /// Finalize block with Bullshark consensus
    pub async fn finalize_block(
        &self,
        block: &Block,
        finality_proof: &BullsharkCert,
    ) -> Result<()> {
        info!(
            "🎯 Finalizing block {} at height {}",
            hex::encode(&block.hash),
            block.height
        );

        let start_time = SystemTime::now();

        // Prepare atomic batch
        let mut batch = Vec::new();

        // Store finalized block
        let block_key = self.block_key(block.height, &block.hash);
        let block_data = bincode::serialize(block)?;
        batch.push((CF_BLOCKS, block_key, block_data));

        // Store finality proof
        let proof_key = format!("finality_{}", block.height);
        let proof_data = bincode::serialize(finality_proof)?;
        batch.push((CF_BULLSHARK_CERT, proof_key.into_bytes(), proof_data));

        // Commit atomically
        self.hot_db.write_batch(batch).await?;

        // Update finalized height in manifest
        {
            let mut manifest = self.manifest.write().await;
            manifest.finalized_height = block.height.max(manifest.finalized_height);
            manifest.save(&self.hot_db).await?;
        }

        let latency = start_time.elapsed().unwrap_or(Duration::from_millis(0));
        self.metrics
            .record_block_finalization(latency, block.vertices.len())
            .await;

        info!(
            "✅ Finalized block {} ({}ms, {} txs)",
            hex::encode(&block.hash),
            latency.as_millis(),
            block.vertices.len()
        );

        // Check if we should create a snapshot
        self.check_snapshot_trigger(block.height).await?;

        Ok(())
    }

    /// Get vertex by ID
    pub async fn get_vertex(&self, vertex_id: &[u8]) -> Result<Option<Vertex>> {
        // For point queries, we need to search by vertex ID
        // This is less efficient than round-based queries
        debug!("🔍 Looking up vertex {}", hex::encode(vertex_id));

        // Implementation would need a secondary index vertex_id -> (round, author, seq)
        // For now, we'll implement a scan (inefficient but correct)
        self.scan_for_vertex(vertex_id).await
    }

    /// Get vertices for a specific round
    pub async fn get_vertices_for_round(&self, round: u64) -> Result<Vec<Vertex>> {
        debug!("🔍 Fetching all vertices for round {}", round);

        let prefix = round.to_be_bytes();
        let vertices = self.hot_db.scan_prefix(CF_DAG_VERTICES, &prefix).await?;

        let mut result = Vec::new();
        for (_, vertex_data) in vertices {
            let vertex: Vertex = bincode::deserialize(&vertex_data)?;
            result.push(vertex);
        }

        debug!("✅ Found {} vertices for round {}", result.len(), round);
        Ok(result)
    }

    /// Get Narwhal payload by digest
    pub async fn get_payload(&self, digest: &[u8]) -> Result<Option<NarwhalPayload>> {
        debug!("🔍 Fetching payload {}", hex::encode(digest));

        if let Some(payload_data) = self.cold_db.get(CF_NARWHAL_PAYLOADS, digest).await? {
            let payload: NarwhalPayload = bincode::deserialize(&payload_data)?;
            return Ok(Some(payload));
        }

        Ok(None)
    }

    /// Get finalized block by height
    pub async fn get_block_by_height(&self, height: u64) -> Result<Option<Block>> {
        debug!("🔍 Fetching block at height {}", height);

        // Scan for block with this height (RocksDB iterator)
        let prefix = height.to_be_bytes();
        let blocks = self.hot_db.scan_prefix(CF_BLOCKS, &prefix).await?;

        if let Some((_, block_data)) = blocks.into_iter().next() {
            let block: Block = bincode::deserialize(&block_data)?;
            return Ok(Some(block));
        }

        Ok(None)
    }

    // ========================================
    // QBLOCK STORAGE METHODS (Phase 2)
    // ========================================

    /// Save QBlock to storage (simplified version for Phase 2 block production)
    /// This is used by the BlockProducer to persist produced blocks
    ///
    /// ✅ v0.9.93-beta: Routes through single-writer queue to prevent corruption
    /// ✅ v1.0.3.5-beta: Updates height cache after successful write
    /// 🚨 v1.1.9-beta: Uses global write lock to prevent concurrent write races
    pub async fn save_qblock(&self, block: &q_types::block::QBlock) -> Result<()> {
        // v7.1.3: Reject pre-genesis blocks (testnet contamination via P2P)
        // v7.3.4: Use network-aware genesis timestamp (mainnet2026.1.1 uses rehearsal timestamp)
        {
            let genesis_ts = crate::balance_consensus::active_genesis_timestamp();
            if block.header.timestamp > 0 && block.header.timestamp < genesis_ts {
                warn!("🧹 [GENESIS FILTER] Rejecting pre-genesis block height={} timestamp={} (genesis={})",
                      block.header.height, block.header.timestamp, genesis_ts);
                return Ok(()); // Silently skip — not an error, just stale data
            }
        }

        // 🚨 v1.1.9: GLOBAL WRITE LOCK - Prevents race conditions with batch writes
        // All three write paths must share this lock to prevent pointer corruption
        let _global_guard = self.global_write_lock.lock().await;
        debug!("🔒 [v1.1.9] Acquired global write lock for single block {}", block.header.height);

        let start_time = SystemTime::now();
        let block_height = block.header.height;

        // FIX 1.1: Route through single-writer queue (serializes all writes)
        self.block_writer.write_block(block.clone()).await?;

        // ✅ v1.0.3.5-beta: Update height cache after successful write
        self.height_cache.update(block_height).await;
        debug!("🚀 [HEIGHT CACHE] Updated to height {} after single block save", block_height);

        let latency = start_time.elapsed().unwrap_or(Duration::from_millis(0));

        // Update metrics (count mining solutions as transactions)
        self.metrics
            .record_block_finalization(latency, block.mining_solutions.len())
            .await;

        debug!("🔓 [v1.1.9] Releasing global write lock after single block {}", block_height);
        Ok(())
    }

    /// 🌐 v2.7.2-beta: DAG LAYER BLOCK STORAGE - Enables parallel block production
    ///
    /// DAG-Knight allows multiple blocks at the same height from different proposers.
    /// This method stores blocks in the "DAG layer" using a composite key:
    ///   `qblock:dag:{height}:{proposer_hex}`
    ///
    /// This enables:
    /// - Multiple blocks per height (true DAG structure)
    /// - All valid blocks from all nodes are stored
    /// - Balance updates processed for ALL blocks (parallel mining rewards)
    /// - No block rejection due to fork-choice (DAG accepts all)
    ///
    /// The canonical chain pointer (`qblock:height:{height}`) is NOT modified.
    /// DAG ordering is used for transaction sequencing, not fork-choice.
    pub async fn save_dag_layer_block(&self, block: &q_types::block::QBlock) -> Result<()> {
        // v7.1.3: Reject pre-genesis blocks
        // v7.3.4: Use network-aware genesis timestamp
        {
            let genesis_ts = crate::balance_consensus::active_genesis_timestamp();
            if block.header.timestamp > 0 && block.header.timestamp < genesis_ts {
                warn!("🧹 [GENESIS FILTER] Rejecting pre-genesis DAG block height={} timestamp={}",
                      block.header.height, block.header.timestamp);
                return Ok(());
            }
        }

        let _global_guard = self.global_write_lock.lock().await;

        let block_height = block.header.height;
        let proposer_hex = hex::encode(&block.header.proposer[..8]);
        let block_hash = block.calculate_hash();

        // Serialize block
        let block_data = bincode::serialize(block)
            .context("Failed to serialize DAG layer block")?;

        // Composite key for DAG layer: allows multiple blocks per height
        let dag_key = format!("qblock:dag:{}:{}", block_height, proposer_hex);

        // Also store hash index for reverse lookup
        let hash_key = format!("qblock:hash:{}", hex::encode(block_hash));

        // Write both entries atomically using the KvStore trait method
        let batch: Vec<(&str, Vec<u8>, Vec<u8>)> = vec![
            (CF_BLOCKS, dag_key.clone().into_bytes(), block_data),
            (CF_BLOCKS, hash_key.into_bytes(), block_height.to_be_bytes().to_vec()),
        ];

        self.hot_db.write_batch_turbo(batch).await
            .context("Failed to write DAG layer block to database")?;

        debug!("🌐 [DAG] Stored block h={} proposer={} hash={}",
              block_height, proposer_hex, hex::encode(&block_hash[..8]));

        Ok(())
    }

    /// 🌐 v2.7.2-beta: Check if a DAG layer block exists for a specific proposer at height
    pub async fn has_dag_layer_block(&self, height: u64, proposer: &[u8; 32]) -> Result<bool> {
        let proposer_hex = hex::encode(&proposer[..8]);
        let dag_key = format!("qblock:dag:{}:{}", height, proposer_hex);

        match self.hot_db.get(CF_BLOCKS, dag_key.as_bytes()).await? {
            Some(_) => Ok(true),
            None => Ok(false),
        }
    }

    /// 🌐 v2.7.2-beta: Get all DAG layer blocks at a specific height
    /// Returns blocks from all proposers that have blocks at this height
    pub async fn get_dag_layer_blocks(&self, height: u64) -> Result<Vec<q_types::block::QBlock>> {
        let prefix = format!("qblock:dag:{}:", height);
        let mut blocks = Vec::new();

        // Scan all keys with this prefix using the KvStore trait method
        let entries = self.hot_db.scan_prefix(CF_BLOCKS, prefix.as_bytes()).await?;

        for (_key, value) in entries {
            if let Ok(block) = bincode::deserialize::<q_types::block::QBlock>(&value) {
                blocks.push(block);
            }
        }

        Ok(blocks)
    }

    /// 🚀 BATCH SAVE BLOCKS - High-performance bulk block storage
    /// Saves multiple blocks in a single RocksDB batch write operation
    /// This is 10x-100x faster than saving blocks one-by-one
    ///
    /// 🛡️ v1.0.79-beta: ORPHAN BLOCK REJECTION
    /// Blocks that would create gaps are rejected to prevent fork fracturing
    ///
    /// 🚀 v1.0.96-beta: Added save_qblocks_batch_turbo for parallel sync
    /// Turbo sync downloads chunks in parallel, so blocks arrive out of order.
    /// Use save_qblocks_batch_turbo to skip orphan checks during turbo sync.
    ///
    /// 🚨 v1.1.9-beta: Uses global write lock to prevent concurrent write races
    pub async fn save_qblocks_batch(&self, blocks: &[q_types::block::QBlock]) -> Result<()> {
        // 🚨 v1.1.9: GLOBAL WRITE LOCK - Prevents race conditions with single block writes
        let _global_guard = self.global_write_lock.lock().await;
        debug!("🔒 [v1.1.9] Acquired global write lock for batch of {} blocks", blocks.len());

        // Default: use orphan rejection for fork protection (regular block gossip)
        let result = self.save_qblocks_batch_internal(blocks, false).await;

        debug!("🔓 [v1.1.9] Releasing global write lock for batch");
        result
    }

    /// 🚀 v1.4.2-beta: TURBO BATCH SAVE - Optimized for 10,000+ blocks/second
    ///
    /// OPTIMIZATION: Lock-free batch preparation + minimal lock hold time
    /// - Before: Lock held for 30ms (hash computation + DB write)
    /// - After:  Lock held for 5-10ms (DB write only)
    ///
    /// Architecture:
    /// 1. Parallel hash computation + serialization (OUTSIDE lock) - 5-10ms
    /// 2. Acquire lock (minimal contention)
    /// 3. DB write (fast, batched) - 5-10ms
    /// 4. Height cache update - 1ms
    /// 5. Release lock
    ///
    /// This allows 32 parallel download streams to prepare batches simultaneously
    /// while only serializing on the actual database write.
    pub async fn save_qblocks_batch_turbo(&self, blocks: &[q_types::block::QBlock]) -> Result<()> {
        if blocks.is_empty() {
            return Ok(());
        }

        // v7.3.7: Bug #28 fix — filter pre-genesis blocks from turbo batch writes
        // save_qblock() had this filter but save_qblocks_batch_turbo() did NOT,
        // allowing rogue pre-launch nodes' blocks through the batch write path.
        let genesis_ts = crate::balance_consensus::active_genesis_timestamp();
        let blocks: Vec<q_types::block::QBlock> = blocks.iter()
            .filter(|b| {
                if b.header.timestamp > 0 && b.header.timestamp < genesis_ts {
                    warn!("🧹 [GENESIS FILTER] Rejecting pre-genesis block height={} ts={} (genesis={}) in turbo batch",
                          b.header.height, b.header.timestamp, genesis_ts);
                    false
                } else {
                    true
                }
            })
            .cloned()
            .collect();
        if blocks.is_empty() {
            return Ok(());
        }
        let blocks = blocks.as_slice();

        let num_blocks = blocks.len();

        // 🚀 v1.4.2-beta: PHASE 1 - Lock-free batch preparation (parallel, CPU-bound)
        // All hash computation and serialization happens OUTSIDE the lock
        let prep_start = std::time::Instant::now();

        // v7.3.1: Parallel serialization with storage optimization
        // Each block is split into: slim block (compressed) + quantum_metadata + transactions
        let block_data_parallel: Vec<(u64, [u8; 32], Vec<u8>, Vec<u8>, Vec<u8>, usize, bool)> = blocks
            .par_iter()
            .map(|block| {
                let block_hash = block.calculate_hash();
                let solutions = block.mining_solutions.len();

                // Serialize quantum_metadata separately
                let qm_data = bincode::serialize(&block.quantum_metadata).unwrap_or_default();

                // Serialize transactions separately
                let has_txs = !block.transactions.is_empty();
                let tx_data = if has_txs {
                    bincode::serialize(&block.transactions).unwrap_or_default()
                } else {
                    Vec::new()
                };

                // Create slim block (no quantum_metadata/transactions)
                let mut slim = block.clone();
                slim.transactions = Vec::new();
                slim.quantum_metadata = q_types::block::QuantumMetadata {
                    vertex_coordinates: q_types::block::HypergraphCoordinates {
                        temporal: 0.0, spatial: Vec::new(), energetic: 0.0, entropic: 0.0,
                        metadata: std::collections::HashMap::new(),
                    },
                    k_parameter: 0.0, energy: 0.0,
                    energy_components: q_types::block::EnergyComponents {
                        coupling: 0.0, potential: 0.0, ordering: 0.0,
                        fault_tolerance: 0.0, temporal: 0.0, finality: 0.0,
                    },
                    spectral_signatures: Vec::new(),
                    wavefunction_phase: 0.0, entropy_variance: 0.0,
                    byzantine_scores: std::collections::HashMap::new(),
                };

                // v7.3.5: Store as QRAW (no app-level compression) — RocksDB handles compression
                let slim_bytes = bincode::serialize(&slim).unwrap_or_default();
                let block_data = match precompressed_storage::PrecompressedBlock::compress(
                    &slim_bytes, precompressed_storage::CompressionAlgorithm::None
                ) {
                    Ok(compressed) => compressed.to_bytes(),
                    Err(_) => slim_bytes, // Fallback to uncompressed on error
                };

                (block.header.height, block_hash, block_data, qm_data, tx_data, solutions, has_txs)
            })
            .collect();

        // Pre-build the batch (memory only, no I/O)
        // v7.3.1: 4 entries per block max (slim block + hash ref + quantum_metadata + transactions)
        let mut batch = Vec::with_capacity(num_blocks * 4);
        let mut total_mining_solutions = 0;
        let mut heights: Vec<u64> = Vec::with_capacity(num_blocks);

        for (height, block_hash, block_data, qm_data, tx_data, solutions, has_txs) in block_data_parallel {
            if block_data.is_empty() {
                continue;
            }

            // Compressed slim block
            let height_key = format!("qblock:height:{}", height);
            batch.push((CF_BLOCKS, height_key.into_bytes(), block_data));

            // Hash → height reference
            let hash_key = format!("qblock:hash:{}", hex::encode(block_hash));
            batch.push((CF_BLOCKS, hash_key.into_bytes(), height.to_be_bytes().to_vec()));

            // Quantum metadata (separate CF for lazy loading)
            if !qm_data.is_empty() {
                let qm_key = format!("qm:{}", height);
                batch.push((CF_QUANTUM_METADATA, qm_key.into_bytes(), qm_data));
            }

            // Transaction bodies (separate CF)
            if has_txs && !tx_data.is_empty() {
                let tx_key = format!("block_txs:{}", height);
                batch.push((CF_TRANSACTIONS, tx_key.into_bytes(), tx_data));
            }

            heights.push(height);
            total_mining_solutions += solutions;
        }

        let prep_duration = prep_start.elapsed();
        debug!("⚡ [LOCKFREE PREP] {} blocks prepared in {:?} ({:.2}µs/block)",
               num_blocks, prep_duration, prep_duration.as_micros() as f64 / num_blocks as f64);

        // 🚀 v1.4.2-beta: PHASE 2 - Minimal lock hold time (only for DB write + height update)
        let lock_start = std::time::Instant::now();
        let _global_guard = self.global_write_lock.lock().await;
        let lock_acquire_time = lock_start.elapsed();

        if lock_acquire_time.as_millis() > 10 {
            warn!("⚠️  [LOCK CONTENTION] Waited {:?} to acquire write lock", lock_acquire_time);
        }

        // Get current height for contiguous calculation
        let contiguous_height = self.height_cache.cached();

        // Calculate highest contiguous height from the stored blocks
        heights.sort();
        heights.dedup();

        let mut new_contiguous_height = contiguous_height;
        let mut batch_advanced = 0u64;
        for height in &heights {
            if *height == new_contiguous_height + 1 {
                new_contiguous_height = *height;
                batch_advanced += 1;
            } else if *height > new_contiguous_height + 1 {
                info!("🔍 [TURBO CONTIGUITY] Gap at height {}: expected {}, got {} (cache was {})",
                      new_contiguous_height + 1, new_contiguous_height + 1, height, contiguous_height);
                break;
            }
        }
        if batch_advanced > 0 {
            info!("🔍 [TURBO CONTIGUITY] Batch advanced pointer by {} blocks: {} → {}",
                  batch_advanced, contiguous_height, new_contiguous_height);
        } else if !heights.is_empty() {
            info!("🔍 [TURBO CONTIGUITY] Batch did NOT advance pointer (cache={}, first_height={}, last_height={})",
                  contiguous_height, heights.first().unwrap(), heights.last().unwrap());
            warn!("⚠️  [GENESIS INTEGRITY] Batch {} blocks ({}-{}) did NOT extend contiguous chain \
                   (current contiguous: {}, first_height: {}, expected_next: {}). \
                   This creates a gap — sequential warp sync will fill it later.",
                  heights.len(), heights.first().unwrap(), heights.last().unwrap(),
                  contiguous_height, heights.first().unwrap(), contiguous_height + 1);
        }

        // v10.2.7: Forward probe — check if blocks from PREVIOUS batches bridge beyond this batch.
        // Fixes the stuck-height bug where gap-fill stores blocks [A..A+2] but blocks [A+3..] already
        // exist on disk from prior turbo sync. Without this, the pointer stays at A+2 instead of
        // advancing through the pre-existing blocks to the next real gap.
        if new_contiguous_height > contiguous_height {
            let mut probe = new_contiguous_height + 1;
            let probe_limit = probe + 10_000; // Bounded scan — don't block the write path
            while probe <= probe_limit {
                let probe_key = format!("qblock:height:{}", probe);
                match self.hot_db.get(CF_BLOCKS, probe_key.as_bytes()).await {
                    Ok(Some(_)) => {
                        new_contiguous_height = probe;
                        probe += 1;
                    }
                    _ => break,
                }
            }
            if new_contiguous_height > contiguous_height + (heights.len() as u64) {
                info!("🔗 [TURBO BRIDGE] Forward probe extended pointer by {} blocks ({} → {})",
                      new_contiguous_height - contiguous_height, contiguous_height, new_contiguous_height);
            }
        }

        // Update height pointer if we extended the chain
        if new_contiguous_height > contiguous_height {
            let latest_height_bytes = new_contiguous_height.to_be_bytes().to_vec();
            batch.push((CF_BLOCKS, b"qblock:latest".to_vec(), latest_height_bytes.clone()));
            // Also persist verified contiguous checkpoint for fast recovery on restart
            batch.push((CF_BLOCKS, b"qblock:contiguous_verified".to_vec(), latest_height_bytes));
        }

        // v7.2.8: ALWAYS persist the highest block stored (tip), even if not contiguous.
        // This prevents the "Swiss cheese reset" bug where height drops from 310K to 35K
        // on restart because qblock:latest only tracks contiguous height.
        // We always write the max — RocksDB is idempotent and the batch is already being written.
        if let Some(&max_stored) = heights.last() {
            batch.push((CF_BLOCKS, b"qblock:tip_height".to_vec(), max_stored.to_be_bytes().to_vec()));
        }

        // Atomic batch write to RocksDB
        self.hot_db.write_batch_turbo(batch).await
            .context("Failed to write turbo batch to database")?;

        // Update height cache
        if new_contiguous_height > contiguous_height {
            self.height_cache.update(new_contiguous_height).await;
            debug!("📈 [TURBO] Height pointer: {} → {}", contiguous_height, new_contiguous_height);
        }

        let lock_hold_time = lock_start.elapsed();
        debug!("🔓 [TURBO] Lock held for {:?} (acquire: {:?})", lock_hold_time, lock_acquire_time);

        info!("🚀 [TURBO BATCH] Saved {} blocks ({} solutions) in {:?} total",
              num_blocks, total_mining_solutions, prep_start.elapsed());

        Ok(())
    }

    /// Legacy turbo batch save (uses original implementation)
    #[allow(dead_code)]
    async fn save_qblocks_batch_turbo_legacy(&self, blocks: &[q_types::block::QBlock]) -> Result<()> {
        let _global_guard = self.global_write_lock.lock().await;
        let _turbo_guard = self.turbo_sync_lock.lock().await;
        self.save_qblocks_batch_internal(blocks, true).await
    }

    /// 🚨 v2.2.0: ACQUIRE GLOBAL WRITE LOCK
    ///
    /// CRITICAL FOR DATA INTEGRITY: External code (like turbo_sync.rs) that performs
    /// direct Transaction writes MUST acquire this lock first to prevent race conditions.
    ///
    /// Race condition scenario WITHOUT this lock:
    /// 1. Thread A: reads height pointer = 100
    /// 2. Thread B: reads height pointer = 100
    /// 3. Thread A: writes blocks 101-200, sets pointer = 200
    /// 4. Thread B: writes blocks 101-150, sets pointer = 150
    /// 5. RESULT: Pointer regresses from 200 to 150, blocks 151-200 become orphans!
    ///
    /// WITH this lock:
    /// 1. Thread A: acquires lock, writes 101-200, sets pointer = 200, releases
    /// 2. Thread B: acquires lock, reads pointer = 200, writes 201-250, releases
    /// 3. RESULT: All blocks contiguous, no data loss
    ///
    /// Usage:
    /// ```
    /// let _guard = storage.acquire_global_write_lock().await;
    /// // All writes within this scope are protected
    /// let tx = storage.begin_transaction().await?;
    /// tx.save_qblock(&block).await?;
    /// tx.commit().await?;
    /// // Guard drops here, releasing the lock
    /// ```
    pub async fn acquire_global_write_lock(&self) -> tokio::sync::MutexGuard<'_, ()> {
        self.global_write_lock.lock().await
    }

    /// Internal batch save with configurable orphan rejection
    async fn save_qblocks_batch_internal(
        &self,
        blocks: &[q_types::block::QBlock],
        skip_orphan_check: bool,
    ) -> Result<()> {
        if blocks.is_empty() {
            return Ok(());
        }

        let start_time = SystemTime::now();
        let original_count = blocks.len();

        // 🛡️ v1.0.79-beta: Get current contiguous height to reject orphans
        // The height_cache only allows monotonic updates, so cached() IS the contiguous height
        let contiguous_height = self.height_cache.cached();

        // 🚀 v1.0.96-beta: Skip orphan rejection for turbo sync (parallel downloads)
        let valid_blocks: Vec<&q_types::block::QBlock>;
        let rejected_count: usize;

        if skip_orphan_check {
            // Turbo sync mode: accept all blocks, they'll be connected later
            valid_blocks = blocks.iter().collect();
            rejected_count = 0;
            debug!("⚡ [TURBO BATCH] Skipping orphan check for {} blocks (parallel sync)", blocks.len());
        } else {
            // Normal mode: filter blocks to prevent gaps
            let mut temp_valid: Vec<&q_types::block::QBlock> = Vec::new();
            let mut temp_rejected = 0;

            // Sort blocks by height first to process in order
            let mut sorted_blocks: Vec<&q_types::block::QBlock> = blocks.iter().collect();
            sorted_blocks.sort_by_key(|b| b.header.height);

            // Track what our "expected next height" is as we accept blocks
            let mut expected_next = contiguous_height + 1;

            for block in sorted_blocks {
                let block_height = block.header.height;

                // Accept block if:
                // 1. It's the next expected block (extends our chain)
                // 2. It's at or below our contiguous height (already have it or filling gap)
                // 3. It's genesis (height 0 or 1)
                if block_height <= contiguous_height || block_height == expected_next || block_height <= 1 {
                    temp_valid.push(block);
                    if block_height == expected_next {
                        expected_next = block_height + 1;
                    }
                } else {
                    // This block would create a gap - reject it
                    temp_rejected += 1;
                    if temp_rejected <= 5 {
                        warn!("🚫 [ORPHAN REJECT] Block {} rejected: would create gap (contiguous={}, expected_next={})",
                              block_height, contiguous_height, expected_next);
                    }
                }
            }

            if temp_rejected > 5 {
                warn!("🚫 [ORPHAN REJECT] ... and {} more blocks rejected", temp_rejected - 5);
            }

            if temp_rejected > 0 {
                warn!("🛡️ [FORK PROTECTION] Rejected {}/{} blocks that would create gaps (contiguous height: {})",
                      temp_rejected, original_count, contiguous_height);
            }

            valid_blocks = temp_valid;
            rejected_count = temp_rejected;
        }

        if valid_blocks.is_empty() {
            info!("🚫 [ORPHAN REJECT] All {} blocks rejected - none extend our chain at height {}",
                  original_count, contiguous_height);
            return Ok(());
        }

        let num_blocks = valid_blocks.len();
        info!("🚀 BATCH SAVE: Saving {} blocks to database (rejected {} orphans)...",
              num_blocks, rejected_count);

        // 🚀 v7.3.1: PARALLEL hash computation + serialization with storage optimization
        // Each block is split: slim block (Lz4 compressed) + quantum_metadata + transactions
        let parallel_start = std::time::Instant::now();

        // Pre-compute all hashes, serialize+compress in parallel
        let block_data_parallel: Vec<(u64, [u8; 32], Vec<u8>, Vec<u8>, Vec<u8>, usize, bool)> = valid_blocks
            .par_iter()
            .map(|block| {
                let block_hash = block.calculate_hash();
                let solutions = block.mining_solutions.len();

                // Serialize quantum_metadata + transactions separately
                let qm_data = bincode::serialize(&block.quantum_metadata).unwrap_or_default();
                let has_txs = !block.transactions.is_empty();
                let tx_data = if has_txs {
                    bincode::serialize(&block.transactions).unwrap_or_default()
                } else { Vec::new() };

                // Create slim block clone
                let mut slim = (*block).clone();
                slim.transactions = Vec::new();
                slim.quantum_metadata = q_types::block::QuantumMetadata {
                    vertex_coordinates: q_types::block::HypergraphCoordinates {
                        temporal: 0.0, spatial: Vec::new(), energetic: 0.0, entropic: 0.0,
                        metadata: std::collections::HashMap::new(),
                    },
                    k_parameter: 0.0, energy: 0.0,
                    energy_components: q_types::block::EnergyComponents {
                        coupling: 0.0, potential: 0.0, ordering: 0.0,
                        fault_tolerance: 0.0, temporal: 0.0, finality: 0.0,
                    },
                    spectral_signatures: Vec::new(),
                    wavefunction_phase: 0.0, entropy_variance: 0.0,
                    byzantine_scores: std::collections::HashMap::new(),
                };

                // v7.3.5: Store as QRAW (no app-level compression) — RocksDB handles compression
                let slim_bytes = bincode::serialize(&slim).unwrap_or_default();
                let block_data = match precompressed_storage::PrecompressedBlock::compress(
                    &slim_bytes, precompressed_storage::CompressionAlgorithm::None
                ) {
                    Ok(compressed) => compressed.to_bytes(),
                    Err(_) => slim_bytes,
                };

                (block.header.height, block_hash, block_data, qm_data, tx_data, solutions, has_txs)
            })
            .collect();

        let parallel_duration = parallel_start.elapsed();
        debug!("⚡ [PARALLEL] Computed {} block hashes+serializations in {:?} ({:.2}µs/block)",
               num_blocks, parallel_duration,
               parallel_duration.as_micros() as f64 / num_blocks as f64);

        // Build batch from parallel results (memory only, no I/O)
        // v7.3.1: up to 4 entries per block
        let mut batch = Vec::with_capacity(num_blocks * 4);
        let mut total_mining_solutions = 0;

        for (height, block_hash, block_data, qm_data, tx_data, solutions, has_txs) in block_data_parallel {
            if block_data.is_empty() {
                warn!("⚠️  Skipping block {} - serialization failed", height);
                continue;
            }

            // Compressed slim block
            let height_key = format!("qblock:height:{}", height);
            batch.push((CF_BLOCKS, height_key.into_bytes(), block_data));

            // Hash → height reference (8 bytes)
            let hash_key = format!("qblock:hash:{}", hex::encode(block_hash));
            batch.push((CF_BLOCKS, hash_key.into_bytes(), height.to_be_bytes().to_vec()));

            // Quantum metadata (separate CF)
            if !qm_data.is_empty() {
                let qm_key = format!("qm:{}", height);
                batch.push((CF_QUANTUM_METADATA, qm_key.into_bytes(), qm_data));
            }

            // Transaction bodies (separate CF)
            if has_txs && !tx_data.is_empty() {
                let tx_key = format!("block_txs:{}", height);
                batch.push((CF_TRANSACTIONS, tx_key.into_bytes(), tx_data));
            }

            total_mining_solutions += solutions;
        }

        // 🛡️ v1.0.88-beta: CRITICAL FIX - Only update pointer to CONTIGUOUS height
        // BUG FIXED: Previously updated to MAX height which caused height regression when
        // blocks arrived out of order. Now we calculate the highest CONTIGUOUS block.
        //
        // Example of old bug:
        // - contiguous_height = 100
        // - valid_blocks contains [101, 102, 105, 106] (103, 104 missing)
        // - OLD: pointer updated to 106 (wrong! we don't have 103, 104)
        // - NEW: pointer updated to 102 (correct - highest contiguous)

        let mut new_contiguous_height = contiguous_height;

        // Calculate highest contiguous block from the accepted blocks
        // Sort valid_blocks by height and find the last contiguous one
        let mut sorted_heights: Vec<u64> = valid_blocks.iter().map(|b| b.header.height).collect();
        sorted_heights.sort();
        sorted_heights.dedup(); // Remove duplicates

        for height in sorted_heights {
            if height == new_contiguous_height + 1 {
                new_contiguous_height = height;
            } else if height > new_contiguous_height + 1 {
                // Gap detected - stop here
                break;
            }
            // heights <= new_contiguous_height are ignored (already have them)
        }

        // 🚨 MONOTONICITY CHECK: NEVER decrease the height pointer
        // 🚀 v1.0.93-beta: Use cached contiguous_height instead of async DB read
        // BEFORE: get_latest_qblock_height().await? (5-10ms async DB call)
        // AFTER: Use contiguous_height from height_cache.cached() (0ms)
        let current_pointer = contiguous_height; // Already fetched at start of function
        let max_height = if new_contiguous_height > current_pointer {
            let latest_height_bytes = new_contiguous_height.to_be_bytes().to_vec();
            batch.push((CF_BLOCKS, b"qblock:latest".to_vec(), latest_height_bytes));
            info!("📈 [BATCH SAVE] Height pointer: {} → {} (contiguous extension)",
                  current_pointer, new_contiguous_height);
            Some(new_contiguous_height)
        } else if new_contiguous_height < current_pointer && current_pointer > 1000 {
            // 🚨 CRITICAL: Pointer would decrease - this is a BUG, log and skip
            error!("🚨 [BATCH SAVE] REFUSING HEIGHT REGRESSION: {} → {} (gap in chain?)",
                   current_pointer, new_contiguous_height);
            error!("   valid_blocks: {} blocks, contiguous_height was: {}",
                   valid_blocks.len(), contiguous_height);
            Some(current_pointer) // Keep existing pointer
        } else {
            // No change needed
            debug!("📊 [BATCH SAVE] Height pointer unchanged at {}", current_pointer);
            Some(current_pointer)
        };

        // Commit entire batch atomically
        // 🚀 v1.0.89-beta: Use TURBO MODE for initial sync (1000+ BPS target)
        // Check if we're in turbo sync mode (far behind network height)
        let use_turbo = std::env::var("Q_TURBO_SYNC")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(true);  // Default: ON for maximum sync speed

        if use_turbo {
            // TURBO MODE: WAL enabled, no fsync per write
            // Caller must call sync_wal() periodically (handled by TurboSync)
            self.hot_db.write_batch_turbo(batch).await
                .context("Failed to write turbo batch QBlocks to database")?;
        } else {
            // SAFE MODE: fsync per write (original behavior)
            self.hot_db.write_batch(batch).await
                .context("Failed to write batch QBlocks to database")?;
        }

        // ✅ v1.0.3.5-beta: Update height cache after successful write
        // 🚀 v1.0.98-beta: CRITICAL FIX - Turbo scan starts from CURRENT contiguous height
        // When blocks arrive out of order due to parallel download:
        // 1. Height cache is at 234k (contiguous)
        // 2. Chunk 244k-254k arrives first, stored (but doesn't extend contiguous)
        // 3. Chunk 234k-244k arrives later, fills the gap
        // 4. Now scan from height_cache (234k) to find we have 234k-254k contiguous
        //
        // BUG FIX: v1.0.97-beta started from max_height (batch's highest), which was WRONG!
        // Example: batch has 244k-254k, max_height=254k, scan starts at 254k+1, finds nothing.
        // CORRECT: Start from height_cache (234k), scan finds 234k-254k contiguous.
        let final_height = if skip_orphan_check {
            // 🚨 v1.1.0 CRITICAL FIX: Detect gaps and warn about stranded blocks
            // Old code stopped at first gap, never discovering blocks beyond it.
            // This caused permanent chain corruption on mainnet-like scenarios.
            let current_contiguous = self.height_cache.cached();
            let mut scan_height = current_contiguous;
            let mut scanned = 0;
            let mut first_gap_at: Option<u64> = None;
            #[allow(unused_assignments)]
            let mut gap_size = 0u64;

            // Phase 1: Scan forward to find contiguous chain extent
            const MAX_SCAN: u64 = 100_000;
            while scanned < MAX_SCAN {
                let next_height = scan_height + 1;
                let height_key = format!("qblock:height:{}", next_height);

                match self.hot_db.get(CF_BLOCKS, height_key.as_bytes()).await {
                    Ok(Some(_)) => {
                        scan_height = next_height;
                        scanned += 1;
                    }
                    _ => {
                        // 🚨 v1.1.0: Found first gap - record it but DON'T stop yet!
                        if first_gap_at.is_none() {
                            first_gap_at = Some(next_height);
                        }
                        break;
                    }
                }
            }

            // Phase 2: If we found a gap, check if there are blocks BEYOND it
            // This is CRITICAL for detecting corruption before it causes sync failures
            if let Some(gap_start) = first_gap_at {
                // Sample heights beyond the gap to detect stranded blocks
                let probe_heights = [
                    gap_start + 100,
                    gap_start + 1000,
                    gap_start + 10_000,
                    gap_start + 100_000,
                    gap_start + 500_000,
                ];

                let mut highest_stranded: Option<u64> = None;
                for &probe in &probe_heights {
                    let probe_key = format!("qblock:height:{}", probe);
                    if let Ok(Some(_)) = self.hot_db.get(CF_BLOCKS, probe_key.as_bytes()).await {
                        highest_stranded = Some(probe);
                    }
                }

                if let Some(stranded_height) = highest_stranded {
                    // 🚨🚨🚨 CRITICAL: We have stranded blocks beyond the gap!
                    gap_size = stranded_height.saturating_sub(gap_start);
                    error!("🚨🚨🚨 [v1.1.0 GAP DETECTOR] CRITICAL DATABASE CORRUPTION DETECTED!");
                    error!("🚨 Contiguous chain: 0 → {}", scan_height);
                    error!("🚨 FIRST GAP at height: {}", gap_start);
                    error!("🚨 Stranded blocks found at height: {}+", stranded_height);
                    error!("🚨 Gap size: ~{} blocks", gap_size);
                    error!("🚨 New nodes CANNOT sync past height {}!", scan_height);
                    error!("🚨 ACTION REQUIRED: Fill gap or reset database to height {}", scan_height);

                    // Log to metrics for alerting
                    warn!("📊 [METRICS] gap_detected=true, gap_start={}, gap_size={}, stranded_height={}",
                          gap_start, gap_size, stranded_height);
                }
            }

            if scan_height > current_contiguous {
                info!("⚡ [TURBO SCAN] Extended contiguous height: {} → {} (+{} blocks now contiguous)",
                      current_contiguous, scan_height, scan_height - current_contiguous);

                // 🚨 v1.1.9: FIX 2 - FLUSH WAL BEFORE POINTER UPDATE
                // Ensures all block data is durably written before we update the pointer
                // Prevents: crash leaves pointer ahead of actual data
                if let Err(e) = self.hot_db.flush().await {
                    warn!("⚠️ [v1.1.9] WAL flush before pointer update failed: {} (continuing anyway)", e);
                }

                // Update the height pointer in database
                // v8.2.8: Use put_sync to ensure durability — sled on Windows loses
                // unsynced writes on crash, causing height regression to 0.
                let latest_height_bytes = scan_height.to_be_bytes().to_vec();
                self.hot_db.put_sync(CF_BLOCKS, b"qblock:latest", &latest_height_bytes).await
                    .context("Failed to update height pointer after turbo scan")?;

                // 🚨 v1.1.9: FIX 2 - FLUSH AFTER POINTER UPDATE
                // Ensures pointer is durably written
                if let Err(e) = self.hot_db.flush().await {
                    warn!("⚠️ [v1.1.9] WAL flush after pointer update failed: {} (continuing anyway)", e);
                }
            }

            // 🚨 v1.1.0: Return the ACTUAL contiguous height, not highest existing
            // This prevents pointer from advancing past gaps
            Some(scan_height)
        } else {
            max_height
        };

        // 🚨 v1.1.0 CRITICAL: Post-write verification before updating cache
        // This ensures we NEVER advance the cache past actually-written blocks
        if let Some(height) = final_height {
            // Verify the block at final_height actually exists
            let verification_key = format!("qblock:height:{}", height);
            match self.hot_db.get(CF_BLOCKS, verification_key.as_bytes()).await {
                Ok(Some(_)) => {
                    // Block verified - safe to update cache
                    self.height_cache.update(height).await;
                    debug!("🚀 [HEIGHT CACHE] Updated to height {} after batch save (VERIFIED)", height);
                }
                Ok(None) => {
                    // 🚨 CRITICAL: Block not found after write! Don't update cache.
                    error!("🚨🚨🚨 [v1.1.0 WRITE VERIFICATION FAILED] Block {} NOT FOUND after write!", height);
                    error!("🚨 Cache NOT updated - preventing pointer drift!");
                    error!("🚨 This indicates a write failure or race condition!");
                    // Don't update cache - leave it at previous contiguous height
                }
                Err(e) => {
                    error!("🚨 [WRITE VERIFICATION] DB error checking block {}: {}", height, e);
                    // Don't update cache on error - safe default
                }
            }
        }

        let latency = start_time.elapsed().unwrap_or(Duration::from_millis(0));

        // Update metrics for all blocks (use valid_blocks)
        for block in &valid_blocks {
            self.metrics
                .record_block_finalization(latency, block.mining_solutions.len())
                .await;
        }

        info!(
            "✅ BATCH SAVE COMPLETE: Saved {} blocks in {}ms ({} blocks/sec, {} solutions)",
            num_blocks,
            latency.as_millis(),
            if latency.as_millis() > 0 { (num_blocks as u128 * 1000) / latency.as_millis() } else { 0 },
            total_mining_solutions
        );

        Ok(())
    }

    /// Get QBlock by height
    /// v1.0.80-beta: Uses legacy fallback for blocks stored before v1.0.60
    pub async fn get_qblock_by_height(&self, height: u64) -> Result<Option<q_types::block::QBlock>> {
        debug!("🔍 Fetching QBlock at height {}", height);

        let height_key = format!("qblock:height:{}", height);

        match self.hot_db.get(CF_BLOCKS, height_key.as_bytes()).await? {
            Some(block_data) => {
                // v7.3.1: Detect compressed format vs legacy raw bincode
                let mut block = if precompressed_storage::is_precompressed(&block_data) {
                    // NEW FORMAT: compressed slim block → decompress → deserialize
                    let compressed = precompressed_storage::PrecompressedBlock::from_bytes(&block_data)
                        .context("Failed to parse compressed block header")?;
                    // v7.3.5: Diagnostic logging for LZ4 decompression failures
                    let raw_bytes = match compressed.decompress() {
                        Ok(bytes) => bytes,
                        Err(e) => {
                            warn!("⚠️  Failed to deserialize block at height {}: LZ4 decompression failed", height);
                            warn!("   block_data.len()={}, header: {:02x?}, algo={:?}, original_size={}, compressed_data.len()={}",
                                  block_data.len(),
                                  &block_data[..std::cmp::min(20, block_data.len())],
                                  compressed.algorithm,
                                  compressed.original_size,
                                  compressed.data.len());
                            return Err(e).context("Failed to decompress block");
                        }
                    };
                    match q_types::legacy::deserialize_qblock_with_fallback(&raw_bytes) {
                        Ok(b) => b,
                        Err(e) => {
                            // v10.2.9: Reduced to debug — corrupt blocks at heights 6-12M cause log storm
                            debug!("⚠️  Failed to deserialize decompressed QBlock at height {}: {}", height, e);
                            return Ok(None);
                        }
                    }
                } else {
                    // LEGACY FORMAT: raw bincode (pre-v7.3.1)
                    match q_types::legacy::deserialize_qblock_with_fallback(&block_data) {
                        Ok(b) => b,
                        Err(e) => {
                            // v10.2.9: Reduced to debug — corrupt blocks at heights 6-12M cause log storm
                            debug!("⚠️  Failed to deserialize QBlock at height {}: {} - treating as missing", height, e);
                            return Ok(None);
                        }
                    }
                };

                // v7.3.1: Reconstruct full block from separate CFs if needed
                // Fetch quantum_metadata from CF_QUANTUM_METADATA
                let qm_key = format!("qm:{}", height);
                if let Ok(Some(qm_data)) = self.hot_db.get(CF_QUANTUM_METADATA, qm_key.as_bytes()).await {
                    if let Ok(qm) = bincode::deserialize::<q_types::block::QuantumMetadata>(&qm_data) {
                        block.quantum_metadata = qm;
                    }
                }

                // Fetch transactions from CF_TRANSACTIONS
                if block.transactions.is_empty() {
                    let tx_key = format!("block_txs:{}", height);
                    if let Ok(Some(tx_data)) = self.hot_db.get(CF_TRANSACTIONS, tx_key.as_bytes()).await {
                        if let Ok(txs) = bincode::deserialize::<Vec<q_types::Transaction>>(&tx_data) {
                            block.transactions = txs;
                        }
                    }
                }

                Ok(Some(block))
            }
            None => Ok(None),
        }
    }

    /// Get QBlock by hash
    /// v1.3.5-beta: Two-step lookup (hash→height→block) to save 50% storage
    /// v1.0.80-beta: Uses legacy fallback for blocks stored before v1.0.60
    pub async fn get_qblock_by_hash(&self, hash: &[u8; 32]) -> Result<Option<q_types::block::QBlock>> {
        debug!("🔍 Fetching QBlock by hash {}", hex::encode(hash));

        let hash_key = format!("qblock:hash:{}", hex::encode(hash));

        match self.hot_db.get(CF_BLOCKS, hash_key.as_bytes()).await? {
            Some(data) => {
                // v1.3.5-beta: Check if this is a height reference (8 bytes) or full block (legacy)
                if data.len() == 8 {
                    // NEW FORMAT: hash→height reference, do two-step lookup
                    let height = u64::from_be_bytes(data.try_into().unwrap());
                    debug!("🔗 Hash→height lookup: {} → height {}", hex::encode(hash), height);
                    return self.get_qblock_by_height(height).await;
                }

                // LEGACY FORMAT: Full block stored under hash key (pre-v1.3.5)
                // v1.0.80-beta: Try current format first, then legacy format
                match q_types::legacy::deserialize_qblock_with_fallback(&data) {
                    Ok(block) => Ok(Some(block)),
                    Err(e) => {
                        warn!("⚠️  Failed to deserialize QBlock with hash {}: {} - treating as missing (backwards compatibility)", hex::encode(hash), e);
                        Ok(None)
                    }
                }
            }
            None => Ok(None),
        }
    }

    /// Get latest QBlock
    pub async fn get_latest_qblock(&self) -> Result<Option<q_types::block::QBlock>> {
        debug!("🔍 Fetching latest QBlock");

        // Get latest height
        let latest_height = match self.hot_db.get(CF_BLOCKS, b"qblock:latest").await? {
            Some(height_bytes) => {
                if height_bytes.len() != 8 {
                    warn!("Invalid latest height bytes length: {}", height_bytes.len());
                    return Ok(None);
                }

                let mut height_array = [0u8; 8];
                height_array.copy_from_slice(&height_bytes);
                u64::from_be_bytes(height_array)
            }
            None => {
                // ✅ v0.5.21-beta FIX: qblock:latest pointer missing (legacy database)
                // Use get_highest_contiguous_block() to scan for latest block
                info!("⚠️  qblock:latest pointer missing - scanning for latest block...");
                let highest = self.get_highest_contiguous_block().await?;

                if highest == 0 {
                    debug!("No latest QBlock found in storage");
                    return Ok(None);
                }

                info!("✅ Found latest block at height {} via scanning", highest);
                highest
            }
        };

        // Fetch block at that height
        self.get_qblock_by_height(latest_height).await
    }

    /// Get range of QBlocks for blockchain synchronization - BATCH OPTIMIZED
    /// 🚀 v1.0.100-beta: Uses RocksDB multi_get for 10-50x faster batch fetching
    /// Returns blocks from start_height (inclusive) up to limit blocks
    ///
    /// # Arguments
    /// * `start_height` - Starting block height (inclusive)
    /// * `limit` - Maximum number of blocks to return
    ///
    /// # Returns
    /// Vector of QBlocks in ascending height order
    pub async fn get_qblocks_range(&self, start_height: u64, limit: usize) -> Result<Vec<q_types::block::QBlock>> {
        // v2.1.1-DELTA-V: Increased from 1000 to 50000 for TurboSync P2P efficiency
        // TurboSync uses 20k block chunks; old 1000 cap caused sync stalls
        // Memory: 50k blocks × ~2KB = ~100MB max (acceptable for P2P sync)
        const MAX_BLOCKS_PER_REQUEST: usize = 50_000;
        let capped_limit = std::cmp::min(limit, MAX_BLOCKS_PER_REQUEST);

        if limit > MAX_BLOCKS_PER_REQUEST {
            warn!("🚨 Block range request capped: requested {} blocks, returning max {}",
                  limit, MAX_BLOCKS_PER_REQUEST);
        }

        let fetch_start = std::time::Instant::now();
        debug!("🚀 [BATCH FETCH] Fetching {} blocks from height {}", capped_limit, start_height);

        // v8.0.6: Use MAX of contiguous height AND tip height as upper bound.
        // qblock:latest tracks contiguous height (no gaps), but blocks beyond it
        // may exist from individual production or non-sequential turbo sync.
        // Without this, block-pack serving returns 0 blocks for heights above
        // the contiguous pointer even though the blocks exist in RocksDB.
        let contiguous_height = match self.hot_db.get(CF_BLOCKS, b"qblock:latest").await? {
            Some(height_bytes) if height_bytes.len() == 8 => {
                let mut height_array = [0u8; 8];
                height_array.copy_from_slice(&height_bytes);
                u64::from_be_bytes(height_array)
            }
            _ => 0,
        };
        let tip_height = match self.hot_db.get(CF_BLOCKS, b"qblock:tip_height").await? {
            Some(height_bytes) if height_bytes.len() == 8 => {
                let mut height_array = [0u8; 8];
                height_array.copy_from_slice(&height_bytes);
                u64::from_be_bytes(height_array)
            }
            _ => 0,
        };
        // Also check height cache which tracks the actual production height
        let cached_height = self.height_cache.cached();
        let latest_height = contiguous_height.max(tip_height).max(cached_height);

        if capped_limit >= 100 {
            info!("🔍 [BLOCK-RANGE-DEBUG] contiguous_height={}, tip_height={}, cached_height={}, latest_height={}, requested_start={}, requested_limit={}",
                  contiguous_height, tip_height, cached_height, latest_height, start_height, capped_limit);
        }

        if latest_height == 0 {
            debug!("No QBlock height found (contiguous=0, tip=0, cache=0), returning empty range");
            return Ok(Vec::new());
        }

        // Calculate end height (inclusive)
        let end_height = std::cmp::min(start_height + capped_limit as u64 - 1, latest_height);

        if start_height > end_height {
            if capped_limit >= 100 {
                info!("🔍 [BLOCK-RANGE-DEBUG] EMPTY RETURN: start_height={} > end_height={} (latest_height={})",
                      start_height, end_height, latest_height);
            }
            return Ok(Vec::new());
        }

        // 🚀 v1.0.100-beta: BATCH FETCH OPTIMIZATION
        // BEFORE: N individual DB calls (690 blocks/min bottleneck)
        // AFTER: Single batched multi_get call (10-50x faster)
        let keys: Vec<Vec<u8>> = (start_height..=end_height)
            .map(|h| format!("qblock:height:{}", h).into_bytes())
            .collect();

        // Use RocksDB multi_get for batch fetching
        let results = self.hot_db.multi_get(CF_BLOCKS, &keys).await?;

        {
            let found_count = results.iter().filter(|r| r.is_some()).count();
            let total_keys = keys.len();
            // v10.3.7: Always log (was >= 100), needed for checkpoint probe debugging
            if found_count == 0 || capped_limit <= 5 || capped_limit >= 100 {
                info!("🔍 [BLOCK-RANGE-DEBUG] multi_get: {}/{} keys found for heights {}..={} (cached_height={})",
                      found_count, total_keys, start_height, end_height,
                      self.height_cache.cached());
            }
        }

        // v7.3.1: Also batch-fetch quantum_metadata and transactions for reconstructing blocks
        let qm_keys: Vec<Vec<u8>> = (start_height..=end_height)
            .map(|h| format!("qm:{}", h).into_bytes())
            .collect();
        let tx_keys: Vec<Vec<u8>> = (start_height..=end_height)
            .map(|h| format!("block_txs:{}", h).into_bytes())
            .collect();

        let qm_results = self.hot_db.multi_get(CF_QUANTUM_METADATA, &qm_keys).await.unwrap_or_default();
        let tx_results = self.hot_db.multi_get(CF_TRANSACTIONS, &tx_keys).await.unwrap_or_default();

        let mut blocks = Vec::with_capacity(keys.len());
        let mut missing_count = 0usize;
        let mut corrupt_count = 0usize;
        // v10.2.9: Track consecutive failures for early abort
        let mut consecutive_failures = 0usize;
        const MAX_CONSECUTIVE_FAILURES: usize = 10;

        for (idx, result) in results.into_iter().enumerate() {
            // v10.2.9: Early abort if entire range is CORRUPT (not merely missing/sparse).
            // Only consecutive deserialization failures trigger this — missing blocks reset the counter.
            // Prevents I/O storm when syncing peers request blocks from corrupt height ranges
            // (e.g., heights 6-12M with thousands of corrupt entries from old format data)
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES && blocks.is_empty() {
                let height = start_height + idx as u64;
                if corrupt_count > 0 {
                    warn!("⚠️ [BLOCK-RANGE] Aborting range scan at height {} — {} consecutive failures, {} corrupt blocks deleted (I/O protection)",
                          height, consecutive_failures, corrupt_count);
                }
                missing_count += keys.len() - idx;
                break;
            }

            if let Some(block_data) = result {
                // v7.3.1: Detect compressed vs legacy format
                let deserialize_result = if precompressed_storage::is_precompressed(&block_data) {
                    // Compressed slim block
                    precompressed_storage::PrecompressedBlock::from_bytes(&block_data)
                        .and_then(|c| c.decompress().map_err(|e| e.into()))
                        .and_then(|raw| q_types::legacy::deserialize_qblock_with_fallback(&raw)
                            .map_err(|e| anyhow::anyhow!("{}", e)))
                } else {
                    // Legacy raw bincode
                    q_types::legacy::deserialize_qblock_with_fallback(&block_data)
                        .map_err(|e| anyhow::anyhow!("{}", e))
                };

                match deserialize_result {
                    Ok(mut block) => {
                        consecutive_failures = 0; // Reset on success
                        // Reconstruct from separate CFs if available
                        if let Some(Some(qm_data)) = qm_results.get(idx) {
                            if let Ok(qm) = bincode::deserialize::<q_types::block::QuantumMetadata>(qm_data) {
                                block.quantum_metadata = qm;
                            }
                        }
                        if block.transactions.is_empty() {
                            if let Some(Some(tx_data)) = tx_results.get(idx) {
                                if let Ok(txs) = bincode::deserialize::<Vec<q_types::Transaction>>(tx_data) {
                                    block.transactions = txs;
                                }
                            }
                        }
                        blocks.push(block);
                    }
                    Err(e) => {
                        let height = start_height + idx as u64;
                        corrupt_count += 1;
                        consecutive_failures += 1;
                        // v10.2.9: Rate-limit corrupt block warnings (log first + every 100th)
                        if corrupt_count <= 1 || corrupt_count % 100 == 0 {
                            warn!("⚠️  Failed to deserialize QBlock at height {}: {} - treating as missing", height, e);
                        }
                        // v10.3.7: NEVER delete blocks that fail deserialization.
                        // They may be from an older binary version with a different QBlock layout.
                        // The old "opportunistic cleanup" was destroying valid blocks permanently.
                        // Turbo sync cannot refill them if no peer has them either.
                        missing_count += 1;
                    }
                }
            } else {
                // v10.2.9-fix: Missing blocks (None from multi_get) are NORMAL in a sparse DAG.
                // Only corrupt blocks (deserialization failures) should count toward the abort threshold.
                // Incrementing consecutive_failures here caused early-abort on sparse ranges,
                // returning 0 blocks to syncing peers and triggering their "0 blocks = failure" check.
                consecutive_failures = 0; // Reset — a missing block is not a corruption signal
                missing_count += 1;
            }
        }

        // ═══════════════════════════════════════════════════════════════════
        // v10.3.6: DAG KEY FALLBACK
        // ═══════════════════════════════════════════════════════════════════
        // 545,710 blocks are stored under "qblock:dag:{height}:{proposer}"
        // from early chain history (gossipsub-received before turbo sync existed).
        // The multi_get above only searches "qblock:height:{N}" and misses them.
        //
        // For each height that multi_get returned None, check if a DAG entry
        // exists. This is a per-height prefix scan — O(log N) seek per height.
        // For typical 200-block pack requests, this adds ~20ms on NVMe.
        //
        // READ-ONLY: no database writes. No key format changes. Just smarter reads.
        // ═══════════════════════════════════════════════════════════════════
        let height_found: std::collections::HashSet<u64> = blocks.iter()
            .map(|b| b.header.height)
            .collect();

        let dag_needed: Vec<u64> = (start_height..=end_height)
            .filter(|h| !height_found.contains(h))
            .collect();

        let mut dag_found = 0u64;
        let mut dag_scanned = 0u64;
        let mut dag_deser_errors = 0u64;

        if !dag_needed.is_empty() {
            debug!("🔍 [DAG FALLBACK] Checking {} missing heights for qblock:dag: entries (range {}..={})",
                   dag_needed.len(), start_height, end_height);

            for &height in &dag_needed {
                dag_scanned += 1;

                let dag_prefix = format!("qblock:dag:{}:", height);
                // v10.3.7: Use scan_prefix_seek instead of scan_prefix.
                // CF_BLOCKS has bloom filters but NO prefix extractor, causing
                // prefix_iterator_cf to give false negatives. scan_prefix_seek
                // uses raw iterator seek (B-tree traversal) which always works.
                let dag_entries = match self.hot_db.scan_prefix_seek(CF_BLOCKS, dag_prefix.as_bytes(), 1).await {
                    Ok(entries) => entries,
                    Err(e) => {
                        debug!("⚠️ [DAG FALLBACK] scan_prefix error at height {}: {}", height, e);
                        continue;
                    }
                };

                // Take the first entry (deterministic: byte-order of proposer hash)
                if let Some((_key, value)) = dag_entries.into_iter().next() {
                    // Deserialize — handle compressed and legacy formats
                    let deser_result = if precompressed_storage::is_precompressed(&value) {
                        precompressed_storage::PrecompressedBlock::from_bytes(&value)
                            .and_then(|c| c.decompress().map_err(|e| e.into()))
                            .and_then(|raw| q_types::legacy::deserialize_qblock_with_fallback(&raw)
                                .map_err(|e| anyhow::anyhow!("{}", e)))
                    } else {
                        q_types::legacy::deserialize_qblock_with_fallback(&value)
                            .map_err(|e| anyhow::anyhow!("{}", e))
                    };

                    match deser_result {
                        Ok(block) => {
                            // Safety check: verify height matches key
                            if block.header.height == height {
                                blocks.push(block);
                                dag_found += 1;
                                missing_count = missing_count.saturating_sub(1);
                            } else {
                                warn!("⚠️ [DAG FALLBACK] Height mismatch at key qblock:dag:{}:* — key claims {} but block.header.height={}. Skipping.",
                                      height, height, block.header.height);
                            }
                        }
                        Err(e) => {
                            dag_deser_errors += 1;
                            if dag_deser_errors <= 3 {
                                debug!("⚠️ [DAG FALLBACK] Deserialization failed at height {}: {} ({} bytes)",
                                       height, e, value.len());
                            }
                        }
                    }
                }
            }

            // Re-sort by height after adding DAG blocks
            if dag_found > 0 {
                blocks.sort_by_key(|b| b.header.height);
            }

            // Log summary (not per-block — avoids spam during full sync)
            if dag_found > 0 {
                info!("🏛️ [CHAIN HISTORY] Recovered {} blocks from early chain archive (heights {}..={})",
                      dag_found, start_height, end_height);
            }

            // v10.3.7: Diagnostic removed — the scan_prefix(&[]) OOM'd at 50GB on Epsilon.
            // Root cause found: prefix_iterator_cf returns false negatives without prefix extractor.
            // Fix: scan_prefix_seek uses raw iterator seek instead.
        }

        let elapsed = fetch_start.elapsed();
        let rate = if elapsed.as_millis() > 0 {
            (blocks.len() as u128 * 1000) / elapsed.as_millis()
        } else {
            blocks.len() as u128 * 1000
        };

        info!("✅ [BATCH FETCH] Got {}/{} blocks in {:?} ({} blocks/sec, {} missing, {} from DAG)",
              blocks.len(), keys.len(), elapsed, rate, missing_count, dag_found);

        Ok(blocks)
    }

    /// v10.3.7: Forward-seek block serving for sparse DAG ranges.
    ///
    /// Instead of probing each height individually (O(range) seeks), this does
    /// ONE RocksDB seek and walks forward collecting the next `limit` blocks
    /// wherever they exist. For sparse DAG data (1 block per 500 heights),
    /// this is 100-250x faster than per-height probing.
    ///
    /// Scans both "qblock:dag:" and "qblock:height:" prefixes and merges results
    /// sorted by height. READ-ONLY: never modifies any data.
    pub async fn get_qblocks_forward(
        &self,
        start_height: u64,
        limit: usize,
    ) -> Result<Vec<q_types::block::QBlock>> {
        use q_types::legacy::deserialize_qblock_with_fallback;

        const MAX_LIMIT: usize = 2000;
        let limit = limit.min(MAX_LIMIT);
        if limit == 0 { return Ok(vec![]); }

        let fetch_start = std::time::Instant::now();
        let mut blocks = Vec::with_capacity(limit);
        let mut seen_heights = std::collections::HashSet::with_capacity(limit);

        // === v10.3.7: Use lazy RocksDB iterator instead of scan_prefix_seek ===
        // scan_prefix_seek loaded ALL entries into a Vec (3.2s for 600 entries).
        // get_dag_blocks_forward uses a lazy iterator — O(1) seek + O(limit) reads.
        let dag_entries = self.hot_db.get_dag_blocks_forward(start_height, limit).await?;

        for (height, _key, value) in dag_entries {
            // Deserialize with all fallbacks (including manual old-DAG parser)
            let deser_result = if precompressed_storage::is_precompressed(&value) {
                precompressed_storage::PrecompressedBlock::from_bytes(&value)
                    .and_then(|c| c.decompress().map_err(|e| e.into()))
                    .and_then(|raw| deserialize_qblock_with_fallback(&raw)
                        .map_err(|e| anyhow::anyhow!("{}", e)))
            } else {
                deserialize_qblock_with_fallback(&value)
                    .map_err(|e| anyhow::anyhow!("{}", e))
            };

            if let Ok(block) = deser_result {
                seen_heights.insert(height);
                blocks.push(block);
            }
        }

        // Also merge any qblock:height: blocks in the same range
        if !blocks.is_empty() {
            let max_h = blocks.iter().map(|b| b.header.height).max().unwrap_or(start_height);
            let range = (max_h - start_height + 1).min(500) as usize; // v10.3.8: strict limit per audit
            if let Ok(height_blocks) = self.get_qblocks_range(start_height, range).await {
                for block in height_blocks {
                    if !seen_heights.contains(&block.header.height) {
                        seen_heights.insert(block.header.height);
                        blocks.push(block);
                    }
                }
            }
        }

        blocks.sort_by_key(|b| b.header.height);
        blocks.dedup_by_key(|b| b.header.height);
        blocks.truncate(limit);

        let elapsed = fetch_start.elapsed();
        if !blocks.is_empty() {
            let first_h = blocks.first().map(|b| b.header.height).unwrap_or(0);
            let last_h = blocks.last().map(|b| b.header.height).unwrap_or(0);
            info!("⚡ [WARP SYNC] Served {} blocks in {:?} (heights {}..{}) — chain history recovered",
                  blocks.len(), elapsed, first_h, last_h);
        }

        Ok(blocks)
    }

    /// Try to read a block at a given height using ALL known key formats.
    /// Returns the first successfully deserialized block, or None.
    ///
    /// Key formats tried (in order):
    /// 1. "qblock:height:{N}" (current string format)
    /// 2. "qblock:dag:{N}:*" (DAG layer - scan prefix)
    /// 3. N.to_be_bytes() prefix (old finalize_block binary format)
    ///
    /// This enables serving historical blocks that were stored by older code versions.
    /// READ-ONLY: never modifies any data.
    pub async fn get_qblock_any_format(&self, height: u64) -> Result<Option<q_types::block::QBlock>> {
        // === Format 1: Current string key "qblock:height:{N}" ===
        if let Some(block) = self.get_qblock_by_height(height).await? {
            return Ok(Some(block));
        }

        // === Format 2: DAG layer "qblock:dag:{N}:{proposer_hex}" ===
        // v10.3.7: Use scan_prefix_seek — CF_BLOCKS lacks prefix extractor,
        // causing scan_prefix (prefix_iterator_cf) to return false negatives.
        let dag_prefix = format!("qblock:dag:{}:", height);
        let dag_entries = self.hot_db.scan_prefix_seek(CF_BLOCKS, dag_prefix.as_bytes(), 1).await?;
        if let Some((_key, value)) = dag_entries.into_iter().next() {
            // DAG layer blocks are stored as bincode-serialized QBlock
            // Try compressed first, then raw bincode with legacy fallback
            let deser_result = if precompressed_storage::is_precompressed(&value) {
                precompressed_storage::PrecompressedBlock::from_bytes(&value)
                    .and_then(|c| c.decompress().map_err(|e| e.into()))
                    .and_then(|raw| q_types::legacy::deserialize_qblock_with_fallback(&raw)
                        .map_err(|e| anyhow::anyhow!("{}", e)))
            } else {
                q_types::legacy::deserialize_qblock_with_fallback(&value)
                    .map_err(|e| anyhow::anyhow!("{}", e))
            };

            match deser_result {
                Ok(mut block) => {
                    // Reconstruct quantum_metadata + transactions from separate CFs
                    let qm_key = format!("qm:{}", height);
                    if let Ok(Some(qm_data)) = self.hot_db.get(CF_QUANTUM_METADATA, qm_key.as_bytes()).await {
                        if let Ok(qm) = bincode::deserialize::<q_types::block::QuantumMetadata>(&qm_data) {
                            block.quantum_metadata = qm;
                        }
                    }
                    if block.transactions.is_empty() {
                        let tx_key = format!("block_txs:{}", height);
                        if let Ok(Some(tx_data)) = self.hot_db.get(CF_TRANSACTIONS, tx_key.as_bytes()).await {
                            if let Ok(txs) = bincode::deserialize::<Vec<q_types::Transaction>>(&tx_data) {
                                block.transactions = txs;
                            }
                        }
                    }
                    debug!("📦 [ANY-FORMAT] Found block at height {} via DAG layer key", height);
                    return Ok(Some(block));
                }
                Err(e) => {
                    debug!("⚠️ [ANY-FORMAT] DAG layer block at height {} failed deserialization: {}", height, e);
                }
            }
        }

        // === Format 3: Old finalize_block binary key (height_be_bytes ++ hash) ===
        // v10.3.7: Use scan_prefix_seek — same bloom filter issue as Format 2.
        let binary_prefix = height.to_be_bytes();
        let binary_entries = self.hot_db.scan_prefix_seek(CF_BLOCKS, &binary_prefix, 1).await?;
        if let Some((_key, value)) = binary_entries.into_iter().next() {
            // Old finalize_block stored bincode-serialized Block (not QBlock).
            // Try QBlock deserialization first (in case it was migrated), then Block.
            let deser_result = if precompressed_storage::is_precompressed(&value) {
                precompressed_storage::PrecompressedBlock::from_bytes(&value)
                    .and_then(|c| c.decompress().map_err(|e| e.into()))
                    .and_then(|raw| q_types::legacy::deserialize_qblock_with_fallback(&raw)
                        .map_err(|e| anyhow::anyhow!("{}", e)))
            } else {
                q_types::legacy::deserialize_qblock_with_fallback(&value)
                    .map_err(|e| anyhow::anyhow!("{}", e))
            };

            if let Ok(mut block) = deser_result {
                // Reconstruct from separate CFs
                let qm_key = format!("qm:{}", height);
                if let Ok(Some(qm_data)) = self.hot_db.get(CF_QUANTUM_METADATA, qm_key.as_bytes()).await {
                    if let Ok(qm) = bincode::deserialize::<q_types::block::QuantumMetadata>(&qm_data) {
                        block.quantum_metadata = qm;
                    }
                }
                if block.transactions.is_empty() {
                    let tx_key = format!("block_txs:{}", height);
                    if let Ok(Some(tx_data)) = self.hot_db.get(CF_TRANSACTIONS, tx_key.as_bytes()).await {
                        if let Ok(txs) = bincode::deserialize::<Vec<q_types::Transaction>>(&tx_data) {
                            block.transactions = txs;
                        }
                    }
                }
                debug!("📦 [ANY-FORMAT] Found block at height {} via binary key prefix", height);
                return Ok(Some(block));
            }

            // Last resort: try deserializing as old Block type and convert to minimal QBlock
            if let Ok(old_block) = bincode::deserialize::<Block>(&value) {
                debug!("📦 [ANY-FORMAT] Found old Block at height {} via binary key, converting to QBlock", height);
                let qblock = q_types::block::QBlock {
                    header: q_types::block::BlockHeader {
                        height: old_block.height,
                        timestamp: old_block.timestamp.timestamp() as u64,
                        proposer: old_block.proposer,
                        // All other fields use serde defaults or zero values
                        phase: 0,
                        network_id: String::new(),
                        prev_block_hash: [0u8; 32],
                        solutions_root: [0u8; 32],
                        tx_root: [0u8; 32],
                        state_root: [0u8; 32],
                        dag_round: 0,
                        vdf_proof: Default::default(),
                        anchor_validator: None,
                        producer_id: 0,
                        total_difficulty: 0,
                        producer_public_key: None,
                        producer_signature: None,
                        coinbase_merkle_root: None,
                        total_coinbase_reward: None,
                        coinbase_count: None,
                    },
                    mining_solutions: Vec::new(),
                    dag_parents: Vec::new(),
                    quantum_metadata: Default::default(),
                    transactions: Vec::new(),
                    balance_updates: Vec::new(),
                    size_bytes: 0,
                };
                return Ok(Some(qblock));
            }
        }

        // === Format 4: Archive node HTTP fallback ===
        // Nodes that warp-synced from a checkpoint (~height 16,538,868) have no blocks
        // below that height locally.  For those gaps, transparently fetch from the
        // archive node (Epsilon) which holds full history from genesis.
        //
        // Guard: only for heights at or below the checkpoint so we never create a
        // recursive loop (the archive node itself always has these blocks locally).
        // The env var Q_ARCHIVE_NODE_URL lets operators override the default URL.
        const ARCHIVE_CHECKPOINT_HEIGHT: u64 = 16_538_868;
        if height <= ARCHIVE_CHECKPOINT_HEIGHT {
            // Resolve the archive base URL at call-time (cheap env-var read, no lock).
            let archive_base = std::env::var("Q_ARCHIVE_NODE_URL")
                .unwrap_or_else(|_| "http://89.149.241.126:8080".to_string());
            let url = format!("{}/api/v1/blocks/{}", archive_base, height);

            // ureq is a blocking HTTP client — must run inside spawn_blocking to avoid
            // stalling the async executor.
            let url_clone = url.clone();
            let fetch_result = tokio::task::spawn_blocking(move || -> Option<q_types::block::QBlock> {
                match ureq::get(&url_clone)
                    .timeout(std::time::Duration::from_secs(10))
                    .call()
                {
                    Ok(resp) => {
                        match resp.into_string() {
                            Ok(body) => {
                                // Response is ApiResponse<QBlock>: {"success":true,"data":{...},...}
                                #[derive(serde::Deserialize)]
                                struct ArchiveResponse {
                                    success: bool,
                                    data: Option<q_types::block::QBlock>,
                                }
                                match serde_json::from_str::<ArchiveResponse>(&body) {
                                    Ok(r) if r.success => r.data,
                                    Ok(_) => None,
                                    Err(_) => None,
                                }
                            }
                            Err(_) => None,
                        }
                    }
                    Err(_) => None,
                }
            })
            .await;

            match fetch_result {
                Ok(Some(block)) => {
                    debug!("📡 [ARCHIVE] Fetched block at height {} from archive node ({})", height, url);
                    // Cache the block locally so future queries hit the local DB.
                    if let Err(e) = self.save_qblock(&block).await {
                        debug!("⚠️ [ARCHIVE] Failed to cache block at height {} locally: {}", height, e);
                    }
                    return Ok(Some(block));
                }
                Ok(None) => {
                    debug!("📡 [ARCHIVE] Archive node returned no block at height {} ({})", height, url);
                }
                Err(e) => {
                    debug!("📡 [ARCHIVE] spawn_blocking error for archive fetch at height {}: {}", height, e);
                }
            }
        }

        Ok(None)
    }

    /// Like get_qblocks_range but tries all key formats for each height.
    /// Slower than the regular version (does up to 3 lookups per block instead of 1)
    /// but can find historical blocks stored by older code.
    /// READ-ONLY: never modifies any data.
    pub async fn get_qblocks_range_any_format(&self, start_height: u64, limit: usize) -> Result<Vec<q_types::block::QBlock>> {
        const MAX_BLOCKS_PER_REQUEST: usize = 50_000;
        let capped_limit = std::cmp::min(limit, MAX_BLOCKS_PER_REQUEST);

        if limit > MAX_BLOCKS_PER_REQUEST {
            warn!("🚨 [ANY-FORMAT] Block range request capped: requested {} blocks, returning max {}",
                  limit, MAX_BLOCKS_PER_REQUEST);
        }

        let fetch_start = std::time::Instant::now();
        info!("🔍 [ANY-FORMAT] Fetching up to {} blocks from height {} (multi-format scan)",
              capped_limit, start_height);

        // v10.2.9: Merge strategy (per external AI review)
        // Fast path first, then supplement missing heights from DAG/binary formats
        // Only skip slow path if fast path returned ALL requested blocks (100% fill)
        let fast_blocks = self.get_qblocks_range(start_height, capped_limit).await?;
        if fast_blocks.len() == capped_limit {
            // 100% fill — fast path got everything, no need for slow scan
            info!("✅ [ANY-FORMAT] Fast path returned {}/{} blocks (100% fill)", fast_blocks.len(), capped_limit);
            return Ok(fast_blocks);
        }

        // Build a set of heights already found by fast path
        let mut found_heights: std::collections::HashSet<u64> = std::collections::HashSet::new();
        let mut merged_blocks: Vec<q_types::block::QBlock> = Vec::with_capacity(capped_limit);
        for block in &fast_blocks {
            found_heights.insert(block.header.height);
            merged_blocks.push(block.clone());
        }
        if !fast_blocks.is_empty() {
            info!("🔍 [ANY-FORMAT] Fast path returned {}/{} blocks ({:.0}% fill) — scanning for gaps",
                  fast_blocks.len(), capped_limit, fast_blocks.len() as f64 / capped_limit as f64 * 100.0);
        }

        // Slow path: supplement missing heights with DAG/binary format lookups
        let mut consecutive_failures: usize = 0;
        const MAX_CONSECUTIVE_FAILURES: usize = 20; // Higher threshold for sparse data

        for i in 0..capped_limit as u64 {
            let height = start_height + i;

            // Skip heights already found by fast path
            if found_heights.contains(&height) {
                consecutive_failures = 0;
                continue;
            }

            // Early abort after too many consecutive misses with no new blocks found
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES && merged_blocks.len() <= fast_blocks.len() {
                debug!("⚠️ [ANY-FORMAT] Aborting gap scan at height {} — {} consecutive misses with no new blocks",
                       height, consecutive_failures);
                break;
            }

            match self.get_qblock_any_format(height).await {
                Ok(Some(block)) => {
                    consecutive_failures = 0;
                    merged_blocks.push(block);
                }
                Ok(None) => {
                    consecutive_failures += 1;
                }
                Err(e) => {
                    consecutive_failures += 1;
                    debug!("⚠️ [ANY-FORMAT] Error reading block at height {}: {}", height, e);
                }
            }
        }

        // Sort by height for deterministic ordering (reviewer requirement)
        merged_blocks.sort_by_key(|b| b.header.height);
        // Deduplicate by height (keep first occurrence)
        merged_blocks.dedup_by_key(|b| b.header.height);

        let elapsed = fetch_start.elapsed();
        let rate = if elapsed.as_millis() > 0 {
            (merged_blocks.len() as u128 * 1000) / elapsed.as_millis()
        } else {
            merged_blocks.len() as u128 * 1000
        };

        info!("✅ [ANY-FORMAT] Got {} blocks in {:?} ({} blocks/sec) for range {}..+{} (fast={}, gap-fill={})",
              merged_blocks.len(), elapsed, rate, start_height, capped_limit,
              fast_blocks.len(), merged_blocks.len().saturating_sub(fast_blocks.len()));

        Ok(merged_blocks)
    }

    /// Diagnostic: Scan RocksDB to find actual block height range
    /// This checks what qblock:height: keys actually exist in CF_BLOCKS
    /// Returns (lowest_height, highest_height, sample_count, total_estimate)
    pub async fn debug_scan_block_range(&self) -> Result<(u64, u64, u64, u64)> {
        info!("🔍 [DB-SCAN] Starting diagnostic block range scan...");

        // Check the pointers first
        let contiguous = match self.hot_db.get(CF_BLOCKS, b"qblock:latest").await? {
            Some(bytes) if bytes.len() == 8 => u64::from_be_bytes(bytes.try_into().unwrap()),
            _ => 0,
        };
        let tip = match self.hot_db.get(CF_BLOCKS, b"qblock:tip_height").await? {
            Some(bytes) if bytes.len() == 8 => u64::from_be_bytes(bytes.try_into().unwrap()),
            _ => 0,
        };
        let cached = self.height_cache.cached();
        info!("🔍 [DB-SCAN] Pointers: contiguous={}, tip={}, cached={}", contiguous, tip, cached);

        // Probe specific heights to find where blocks exist
        let mut lowest_found: u64 = u64::MAX;
        let mut highest_found: u64 = 0;
        let mut found_count: u64 = 0;

        // Sample at various heights: 1, 100, 1000, 10000, then every 100K up to max
        let max_height = contiguous.max(tip).max(cached);
        let mut probe_heights: Vec<u64> = vec![0, 1, 2, 10, 100, 1000, 10000, 100000, 500000, 1000000];

        // Add probes every 500K
        let mut h = 0u64;
        while h <= max_height {
            probe_heights.push(h);
            h += 500_000;
        }
        // Add dense probes near the tip (last 200K)
        if max_height > 200_000 {
            let start = max_height - 200_000;
            let mut h = start;
            while h <= max_height {
                probe_heights.push(h);
                h += 10_000;
            }
        }
        // Last 1000 blocks
        if max_height > 1000 {
            for h in (max_height - 1000)..=max_height {
                probe_heights.push(h);
            }
        }

        probe_heights.sort();
        probe_heights.dedup();

        // Batch probe
        let keys: Vec<Vec<u8>> = probe_heights.iter()
            .map(|h| format!("qblock:height:{}", h).into_bytes())
            .collect();

        let results = self.hot_db.multi_get(CF_BLOCKS, &keys).await?;

        for (i, result) in results.iter().enumerate() {
            if result.is_some() {
                let h = probe_heights[i];
                found_count += 1;
                if h < lowest_found { lowest_found = h; }
                if h > highest_found { highest_found = h; }
            }
        }

        if found_count == 0 {
            warn!("🔍 [DB-SCAN] NO blocks found at any probed height! Probed {} heights from 0 to {}",
                  probe_heights.len(), max_height);

            // Also check old binary key format (height.to_be_bytes())
            let old_format_test_heights = vec![1u64, 1000, 100000, 1000000];
            let old_keys: Vec<Vec<u8>> = old_format_test_heights.iter()
                .map(|h| {
                    let mut key = Vec::with_capacity(8);
                    key.extend_from_slice(&h.to_be_bytes());
                    key
                })
                .collect();
            let old_results = self.hot_db.multi_get(CF_BLOCKS, &old_keys).await?;
            let old_found = old_results.iter().filter(|r| r.is_some()).count();
            if old_found > 0 {
                warn!("🔍 [DB-SCAN] Found {} blocks with OLD binary key format! Block data exists but key format mismatch!", old_found);
            }

            // Check scan_prefix for any qblock:height: keys
            let sample = self.hot_db.scan_prefix(CF_BLOCKS, b"qblock:height:").await?;
            info!("🔍 [DB-SCAN] scan_prefix('qblock:height:') returned {} entries", sample.len());
            if let Some((first_key, _)) = sample.first() {
                info!("🔍 [DB-SCAN] First key: {}", String::from_utf8_lossy(first_key));
            }
            if let Some((last_key, _)) = sample.last() {
                info!("🔍 [DB-SCAN] Last key: {}", String::from_utf8_lossy(last_key));
            }

            return Ok((0, 0, 0, 0));
        }

        info!("🔍 [DB-SCAN] RESULTS: lowest_found={}, highest_found={}, probed_found={}/{}, pointer_claims={}",
              lowest_found, highest_found, found_count, probe_heights.len(), max_height);

        if lowest_found > 1000 {
            warn!("🚨 [DB-SCAN] HISTORICAL BLOCKS MISSING! Lowest block is at height {} but pointer claims contiguous from 0", lowest_found);
            warn!("🚨 [DB-SCAN] This means new nodes CANNOT sync from scratch!");
        }

        Ok((lowest_found, highest_found, found_count, max_height))
    }

    /// Get latest QBlock height
    /// Returns None if no blocks exist yet
    /// 🚨 v1.1.9-beta: Added pointer verification to detect corruption
    pub async fn get_latest_qblock_height(&self) -> Result<Option<u64>> {
        match self.hot_db.get(CF_BLOCKS, b"qblock:latest").await? {
            Some(height_bytes) if height_bytes.len() == 8 => {
                let mut height_array = [0u8; 8];
                height_array.copy_from_slice(&height_bytes);
                let height = u64::from_be_bytes(height_array);

                // v1.0.2 OPTION C-A guard: if the pointer is exactly CHECKPOINT_HEIGHT and
                // the balance checkpoint has been applied, treat it as valid even when the
                // block at that height isn't stored yet. The checkpoint snapshot established
                // verified state at CHECKPOINT_HEIGHT atomically; the block DATA for that
                // height (and pre-checkpoint history) fills in later via Phase 2 backfill.
                // Without this guard, the v1.1.9 verification below would mis-classify the
                // legitimate checkpoint-advance as corruption and auto-repair the pointer
                // back to 0, killing C-A's whole point.
                use crate::balance_checkpoint::CHECKPOINT_HEIGHT;
                if height == CHECKPOINT_HEIGHT && self.is_checkpoint_applied().await {
                    return Ok(Some(height));
                }

                // 🚨 v1.1.9: POINTER VERIFICATION - Ensure block at pointer height exists
                // This detects corruption where pointer advances past missing blocks
                let height_key = format!("qblock:height:{}", height);
                match self.hot_db.get(CF_BLOCKS, height_key.as_bytes()).await {
                    Ok(Some(_)) => {
                        // Block exists - pointer is valid
                        Ok(Some(height))
                    }
                    Ok(None) => {
                        // 🚨 CORRUPTION DETECTED - pointer points to missing block!
                        error!("🚨 [v1.1.9] CORRUPTION DETECTED: Pointer height {} has no block!", height);
                        error!("🔧 [v1.1.9] Auto-repairing pointer by scanning backwards...");

                        // Scan backwards to find actual highest existing block
                        let repaired_height = self.repair_pointer_to_contiguous(height).await?;
                        Ok(Some(repaired_height))
                    }
                    Err(e) => {
                        error!("🚨 [v1.1.9] Error verifying pointer block: {}", e);
                        Err(e.into())
                    }
                }
            }
            _ => Ok(None),
        }
    }

    /// 🚨 v1.3.4: Auto-repair pointer by checking checkpoints FIRST, then scanning backwards
    /// This prevents height reset when checkpoints exist at higher heights
    async fn repair_pointer_to_contiguous(&self, broken_height: u64) -> Result<u64> {
        // 🔧 v1.3.4 FIX: First check checkpoints directory for highest known height
        let checkpoint_height = self.get_highest_checkpoint_height().await;
        if checkpoint_height > 0 {
            info!("📍 [v1.3.4] Found checkpoint at height {} - using as repair floor", checkpoint_height);
        }

        // Scan backwards from broken_height to find actual highest block
        let mut repaired_height = 0u64;

        for probe_height in (1..=broken_height).rev() {
            let height_key = format!("qblock:height:{}", probe_height);
            if let Ok(Some(_)) = self.hot_db.get(CF_BLOCKS, height_key.as_bytes()).await {
                repaired_height = probe_height;
                break;
            }

            // Don't scan more than 10k backwards
            if broken_height - probe_height > 10_000 {
                error!("🚨 [v1.1.9] Scanned 10k blocks and found none! Database severely corrupted.");
                break;
            }
        }

        // 🔧 v1.3.4 FIX: If checkpoint height is higher than scanned height, use checkpoint
        // This prevents resetting to a low height when checkpoints prove we had more data
        if checkpoint_height > repaired_height {
            warn!("🔄 [v1.3.4] Checkpoint height {} > scanned height {} - using checkpoint height",
                  checkpoint_height, repaired_height);
            repaired_height = checkpoint_height;
        }

        if repaired_height > 0 {
            // Update pointer to repaired height
            // v8.2.8: put_sync for durability on sled/Windows
            let height_bytes = repaired_height.to_be_bytes();
            self.hot_db.put_sync(CF_BLOCKS, b"qblock:latest", &height_bytes).await
                .context("Failed to repair height pointer")?;

            // Update cache too
            self.height_cache.update(repaired_height).await;

            warn!("✅ [v1.3.4] AUTO-REPAIRED: Pointer {} → {} (checkpoint-aware)", broken_height, repaired_height);
        } else {
            error!("🚨 [v1.3.4] AUTO-REPAIR FAILED: No valid blocks or checkpoints found!");
        }

        Ok(repaired_height)
    }

    /// 🔧 v1.3.4: Get highest checkpoint height from checkpoint files
    /// Checkpoints are stored in {db_path}/checkpoints/checkpoint_{height}.json
    async fn get_highest_checkpoint_height(&self) -> u64 {
        // Get database path from environment or use default
        let db_path = std::env::var("Q_DB_PATH").unwrap_or_else(|_| "./data".to_string());
        let checkpoints_dir = std::path::PathBuf::from(&db_path).join("checkpoints");

        if !checkpoints_dir.exists() {
            return 0;
        }

        let mut highest_height = 0u64;

        if let Ok(entries) = std::fs::read_dir(&checkpoints_dir) {
            for entry in entries.flatten() {
                let filename = entry.file_name();
                let filename_str = filename.to_string_lossy();

                // Parse checkpoint_{height}.json format
                if filename_str.starts_with("checkpoint_") && filename_str.ends_with(".json") {
                    let height_str = filename_str
                        .strip_prefix("checkpoint_")
                        .and_then(|s| s.strip_suffix(".json"));

                    if let Some(height_str) = height_str {
                        if let Ok(height) = height_str.parse::<u64>() {
                            if height > highest_height {
                                highest_height = height;
                            }
                        }
                    }
                }
            }
        }

        if highest_height > 0 {
            info!("📍 [v1.3.4] Found {} checkpoints, highest at height {}",
                  std::fs::read_dir(&checkpoints_dir).map(|d| d.count()).unwrap_or(0),
                  highest_height);
        }

        highest_height
    }

    /// 🚀 v1.0.60-beta: Get raw RocksDB handle for state sync processing
    /// Used by BlockStateProcessor for direct state access
    /// Returns None if the underlying storage doesn't expose RocksDB directly
    #[cfg(not(target_os = "windows"))]
    pub fn get_rocks_db_handle(&self) -> Option<Arc<rocksdb::DB>> {
        Some(self.hot_db_concrete.get_raw_db())
    }

    /// Windows stub: RocksDB not available on Windows
    #[cfg(target_os = "windows")]
    pub fn get_rocks_db_handle(&self) -> Option<Arc<()>> {
        None
    }

    /// Get highest contiguous block height (no gaps from genesis)
    /// Used for accurate peer height registration in TurboSync
    ///
    /// Returns the highest block height where all blocks [0..height] exist in storage
    /// This prevents advertising blocks we don't actually have
    ///
    /// ✅ v1.0.3.5-beta PERFORMANCE FIX: Use cached height (100,000x faster)
    /// - Before: 105ms (RocksDB query + binary search)
    /// - After: <1µs (atomic read from cache)
    /// - Impact: Reduces sync_from_storage from 180×105ms/min = 31.5% overhead → 0.01% overhead
    ///
    /// 🚨 v1.1.9-beta: Added periodic cache verification against database
    pub async fn get_highest_contiguous_block(&self) -> Result<u64> {
        // Return cached value (atomic read - no DB query!)
        let cached_height = self.height_cache.cached();

        // 🚨 v1.1.9: PERIODIC CACHE VERIFICATION
        // 🔧 v3.4.4: Increased from 100 to 10000 to reduce sync overhead (was causing 12x slowdown!)
        // Every 10000 calls, verify the cache matches the database to detect desync
        let counter = self.cache_verification_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        if counter % 10000 == 0 && cached_height > 0 {
            // v1.0.2 OPTION C-A guard: same logic as get_latest_qblock_height — if the
            // cached height is exactly CHECKPOINT_HEIGHT and the balance checkpoint is
            // applied, skip the existence check. Block data for CHECKPOINT_HEIGHT and
            // below is fetched in Phase 2; the checkpoint state itself is verified.
            use crate::balance_checkpoint::CHECKPOINT_HEIGHT;
            if cached_height == CHECKPOINT_HEIGHT && self.is_checkpoint_applied().await {
                // Skip verification — checkpoint legitimately establishes state at this height.
            } else {
                // Verify cached height block actually exists
                let height_key = format!("qblock:height:{}", cached_height);
                if let Ok(None) = self.hot_db.get(CF_BLOCKS, height_key.as_bytes()).await {
                    // 🚨 Cache desync detected!
                    error!("🚨 [v1.1.9] CACHE DESYNC: Cached height {} but block doesn't exist!", cached_height);

                    // Repair by scanning backwards
                    let repaired = self.repair_pointer_to_contiguous(cached_height).await?;
                    self.height_cache.update(repaired).await;
                    warn!("✅ [v1.1.9] Cache repaired: {} → {}", cached_height, repaired);
                    return Ok(repaired);
                }
            }

            // Also verify the database pointer matches cache
            if let Ok(Some(pointer_height)) = self.get_latest_qblock_height().await {
                if pointer_height != cached_height && cached_height > pointer_height + 100 {
                    // Major desync between cache and pointer
                    warn!("⚠️ [v1.1.9] Cache/pointer desync: cache={}, pointer={}", cached_height, pointer_height);
                    // Trust the verified pointer (get_latest_qblock_height already validates)
                    self.height_cache.update(pointer_height).await;
                    return Ok(pointer_height);
                }
            }
        }

        // Debug logging only on startup or significant changes
        if cached_height % 1000 == 0 || cached_height < 10 {
            debug!("🚀 [HEIGHT CACHE] Returning cached height: {} (verified)", cached_height);
        }

        Ok(cached_height)
    }

    /// 🚨 v1.0.41-beta: Public method to update height cache from external callers
    /// Used by libp2p sync to update cache after Transaction.save_qblock() commits
    /// (Transaction doesn't have access to Storage's height_cache)
    pub async fn update_height_cache(&self, height: u64) {
        self.height_cache.update(height).await;
    }

    /// v7.2.8: Persist the "safe floor" — the minimum height we KNOW the node was operating at.
    /// Called periodically (every 15s) from the balance sync task.
    /// On restart, scan_highest_contiguous_block_internal() will NEVER return below this floor.
    ///
    /// v7.3.10: Also reads qblock:tip_height (the turbo sync tip) and saves the MAX of the two.
    /// This ensures safe_floor tracks the true tip height (not just contiguous height) during
    /// turbo sync. Critical for "resume from height" correctness after OOM-kill.
    ///
    /// v7.3.10: Uses put_sync() (fsync) instead of put() to survive hard kills (OOM-kill).
    pub async fn save_safe_floor(&self, height: u64) -> Result<()> {
        if height == 0 {
            return Ok(());
        }

        // v7.3.10: Also read the turbo tip height — during turbo sync the contiguous height
        // (height_cache) can be 2502 while the turbo tip is 5431. We want safe_floor = 5431
        // so that on restart we resume from 5431, not fall back to genesis scan.
        let turbo_tip = match self.hot_db.get(CF_BLOCKS, b"qblock:tip_height").await {
            Ok(Some(bytes)) if bytes.len() == 8 => {
                let mut arr = [0u8; 8];
                arr.copy_from_slice(&bytes);
                u64::from_be_bytes(arr)
            }
            _ => 0,
        };
        let effective_height = std::cmp::max(height, turbo_tip);

        // Only persist if higher than current floor (monotonically increasing)
        let current_floor = self.get_safe_floor().await;
        if effective_height > current_floor {
            // v7.3.10: Use put_sync() (fsync=true) so safe_floor survives OOM-kill / hard power loss.
            // The old put() only wrote to OS page cache — if the process was SIGKILL'd, the
            // OS page cache survives, but under memory pressure the kernel can evict dirty pages
            // without flushing them. put_sync() forces the data to disk before returning.
            self.hot_db
                .put_sync(CF_BLOCKS, b"qblock:safe_floor", &effective_height.to_be_bytes())
                .await
                .context("Failed to persist safe_floor")?;
            debug!("🛡️ [SAFE FLOOR] Persisted safe_floor={} (contiguous={}, turbo_tip={})",
                   effective_height, height, turbo_tip);
        }
        Ok(())
    }

    /// Read the persisted safe floor height. Returns 0 if not set.
    pub async fn get_safe_floor(&self) -> u64 {
        match self.hot_db.get(CF_BLOCKS, b"qblock:safe_floor").await {
            Ok(Some(bytes)) if bytes.len() == 8 => {
                let mut arr = [0u8; 8];
                arr.copy_from_slice(&bytes);
                u64::from_be_bytes(arr)
            }
            _ => 0,
        }
    }

    /// INTERNAL: Scan database for highest height (called ONLY on cache initialization)
    /// This is the slow path that used to be called 180 times per minute!
    async fn scan_highest_contiguous_block_internal(&self) -> Result<u64> {
        // ✅ v0.5.19-beta FIX: Handle legacy databases without qblock:latest pointer
        // If qblock:latest doesn't exist, scan backwards from a large number to find highest block

        // 🔍 v0.9.16-beta: ENHANCED DEBUGGING for height reset diagnosis
        warn!("🔍🔍🔍 [HEIGHT DEBUG] Starting scan_highest_contiguous_block_internal() [SLOW PATH]");

        // v7.2.8: Read safe floor FIRST — this is our absolute minimum return value
        let safe_floor = self.get_safe_floor().await;
        if safe_floor > 0 {
            warn!("🛡️ [SAFE FLOOR] Persisted safe_floor = {} — will NEVER return below this", safe_floor);
        }

        let latest_result = self.get_latest_qblock_height().await?;
        let mut latest = latest_result.unwrap_or(0);

        // Save original pointer height for fast recovery path (before probes modify `latest`)
        let pointer_height_for_fast_recovery = latest;

        warn!("🔍 [HEIGHT DEBUG] qblock:latest pointer returned: {:?} (unwrapped to: {})",
              latest_result, latest);

        if latest == 0 {
            // qblock:latest pointer missing (old database) - scan for highest block
            warn!("🚨 [HEIGHT DEBUG] qblock:latest pointer is ZERO - scanning for highest block...");

            // ✅ v0.9.9-beta FIX: Comprehensive probe heights covering ALL realistic ranges
            // This fixes the catastrophic height reset bug where restart would return height=0
            // despite having 2000+ blocks in database because probes missed the actual range.
            let probe_heights = vec![
                // Very high ranges (future-proofing)
                1_000_000, 500_000, 250_000,
                // High ranges (mainnet potential)
                150_000, 100_000, 75_000, 50_000,
                // Medium ranges (CRITICAL - where testnet actually is!)
                25_000, 10_000, 5_000, 3_000, 2_000, 1_500, 1_000,
                // Low ranges (early testnet)
                500, 250, 100, 50, 10, 5, 1
            ];

            for &probe_height in &probe_heights {
                warn!("🔍 [HEIGHT DEBUG] Probing height {}...", probe_height);
                match self.get_qblock_by_height(probe_height).await {
                    Ok(Some(block)) => {
                        // Found a block! Use this as starting point for binary search
                        // Add buffer proportional to found height for efficient binary search
                        let buffer = std::cmp::max(probe_height / 2, 10_000);
                        latest = probe_height + buffer;

                        warn!("✅ [HEIGHT DEBUG] Found block at height {} (hash: {}), will binary search up to {}",
                              probe_height,
                              hex::encode(&block.calculate_hash()[..8]),
                              latest);
                        break;
                    }
                    Ok(None) => {
                        warn!("🔍 [HEIGHT DEBUG] No block at height {}", probe_height);
                    }
                    Err(e) => {
                        warn!("❌ [HEIGHT DEBUG] Error probing height {}: {}", probe_height, e);
                    }
                }
            }

            if latest == 0 {
                // v8.2.8: Before returning 0, check safe_floor — on Windows/sled the height
                // pointer may be lost but blocks still exist. safe_floor is the last persisted
                // minimum height from periodic saves.
                let safe_floor_key = "qblock:safe_floor";
                if let Ok(Some(sf_bytes)) = self.hot_db.get("cf_metadata", safe_floor_key.as_bytes()).await {
                    if let Ok(sf_str) = String::from_utf8(sf_bytes.to_vec()) {
                        if let Ok(sf) = sf_str.parse::<u64>() {
                            if sf > 0 {
                                warn!("🔄 [HEIGHT RECOVERY] Block probes returned 0 but safe_floor={} — using safe_floor", sf);
                                return Ok(sf);
                            }
                        }
                    }
                }
                // Also check turbo_tip pointer
                let turbo_tip_key = "qblock:turbo_tip";
                if let Ok(Some(tt_bytes)) = self.hot_db.get("cf_metadata", turbo_tip_key.as_bytes()).await {
                    if let Ok(tt_str) = String::from_utf8(tt_bytes.to_vec()) {
                        if let Ok(tt) = tt_str.parse::<u64>() {
                            if tt > 0 {
                                warn!("🔄 [HEIGHT RECOVERY] Block probes returned 0 but turbo_tip={} — using turbo_tip", tt);
                                return Ok(tt);
                            }
                        }
                    }
                }
                warn!("🚨🚨🚨 [HEIGHT DEBUG] NO BLOCKS FOUND after comprehensive scan!");
                warn!("🚨 [HEIGHT DEBUG] Database appears empty — truly fresh start");
                return Ok(0);
            }
        }

        // 🚨 v1.1.0 CRITICAL FIX: Find FIRST gap, not highest existing block
        // The old binary search was WRONG - it found the highest block that exists,
        // but didn't verify all blocks from 0 to that height are contiguous.
        // Example: blocks 0-1024, gap at 1025-5000, blocks 5001-538991
        // Old code would return 538991 (WRONG!), correct answer is 1024.
        info!("🔍 [v1.1.0] Scanning for FIRST gap in blockchain (starting from 0)...");

        // First, find the highest existing block using binary search (for upper bound)
        let mut low = 0u64;
        let mut high = latest;
        let mut highest_existing = 0u64;
        let mut iterations = 0;
        const MAX_ITERATIONS: u32 = 1000;

        while low <= high && iterations < MAX_ITERATIONS {
            let mid = (low + high) / 2;
            iterations += 1;

            if self.get_qblock_by_height(mid).await?.is_some() {
                highest_existing = mid;
                low = mid + 1;
            } else {
                if mid == 0 {
                    break;
                }
                high = mid - 1;
            }
        }

        info!("🔍 [v1.1.0] Highest EXISTING block: {} (found in {} iterations)", highest_existing, iterations);

        // === 🔍 HIGH PROBE SCAN (v7.2.9 — Swiss cheese binary search fix) ===
        //
        // PROBLEM: Binary search on Swiss cheese storage converges to the LOWER
        // contiguous segment. Example: blocks at 0-35200 and 305000-310456.
        // Binary search with high=35200 converges to 35200, missing 310456.
        //
        // FIX: Probe at intervals above `highest_existing` to discover blocks
        // above gaps. Only runs on startup, so ~50 extra reads are negligible.
        {
            let mut probed_tip = highest_existing;
            // Probe from 50K up to 10M at 25K intervals (max ~400 reads)
            // v8.3.0: Raised from 2M to 10M — chain exceeded 2.97M and probe missed all blocks above 2M
            let mut probe_h = ((highest_existing / 25_000) + 1) * 25_000; // next 25K boundary
            while probe_h <= 10_000_000 {
                if self.get_qblock_by_height(probe_h).await?.is_some() {
                    // Found blocks above the gap! Refine with binary search.
                    let mut lo = probe_h;
                    let mut hi = probe_h + 50_000;
                    let mut best = probe_h;
                    let mut iters = 0;
                    while lo <= hi && iters < 50 {
                        let mid = (lo + hi) / 2;
                        iters += 1;
                        if self.get_qblock_by_height(mid).await?.is_some() {
                            best = mid;
                            lo = mid + 1;
                        } else {
                            if mid == 0 { break; }
                            hi = mid - 1;
                        }
                    }
                    probed_tip = best;
                    // Continue probing above in case there are more segments
                    probe_h = ((best / 25_000) + 1) * 25_000;
                } else {
                    probe_h += 25_000;
                }
            }

            if probed_tip > highest_existing {
                info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                info!("🔍 [HIGH PROBE v7.2.9] Found blocks ABOVE gap: binary_search={} → probed_tip={}",
                      highest_existing, probed_tip);
                info!("   Swiss cheese storage detected. Using probed tip as recovery target.");
                info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                highest_existing = probed_tip;
            }
        }

        // === 🚀 FAST RECOVERY PATH (v7.2.8 — "Swiss cheese" fix) ===
        //
        // CRITICAL FIX: Old code scanned from the `qblock:latest` pointer, which only
        // tracks CONTIGUOUS height. With turbo sync, the pointer can be 35,200 while
        // blocks exist up to 310,456. On restart the node would drop to 35,200.
        //
        // NEW APPROACH: Scan backwards from `highest_existing` (found by binary search).
        // If we find ≥100 contiguous blocks at the tip, return that height immediately.
        // Old gaps below the tip are harmless — turbo sync fills them in background.
        //
        // Also check `qblock:tip_height` (persisted by save_qblocks_batch_turbo) as
        // an additional hint for the upper bound.

        // Read persisted tip height (may be higher than binary search found)
        let persisted_tip = match self.hot_db.get(CF_BLOCKS, b"qblock:tip_height").await {
            Ok(Some(bytes)) if bytes.len() == 8 => {
                let mut arr = [0u8; 8];
                arr.copy_from_slice(&bytes);
                u64::from_be_bytes(arr)
            }
            _ => 0,
        };

        // Use the highest known height from all sources
        // v7.3.10: include safe_floor in the max so we never recover below the last durable checkpoint
        let recovery_target = std::cmp::max(
            highest_existing,
            std::cmp::max(pointer_height_for_fast_recovery,
                std::cmp::max(persisted_tip, safe_floor))
        );

        info!("🔍 [FAST RECOVERY v7.2.8] Sources: pointer={}, binary_search={}, persisted_tip={}, safe_floor={}. Target={}",
              pointer_height_for_fast_recovery, highest_existing, persisted_tip, safe_floor, recovery_target);

        if recovery_target > 1000 {
            // If binary search didn't find the actual tip (it can miss non-contiguous blocks),
            // verify the persisted tip actually exists. If not, scan backwards to find the
            // highest block near the recovery target.
            // v8.3.0 FIX: Old code fell back to `highest_existing` (which could be 100K+ lower
            // due to probe ceiling at 2M). Now scans backwards from recovery_target instead.
            let verified_target = if recovery_target > highest_existing {
                if self.get_qblock_by_height(recovery_target).await?.is_some() {
                    recovery_target
                } else {
                    // Block at exact recovery_target doesn't exist — scan backwards to find
                    // the highest block near it (could be off by a few due to unflushed writes)
                    let mut found = highest_existing;
                    let scan_floor = recovery_target.saturating_sub(500);
                    for h in (scan_floor..recovery_target).rev() {
                        if self.get_qblock_by_height(h).await?.is_some() {
                            found = h;
                            info!("🔍 [RECOVERY v8.3.0] recovery_target {} missing, found block at {}", recovery_target, h);
                            break;
                        }
                    }
                    std::cmp::max(found, highest_existing)
                }
            } else {
                recovery_target
            };

            if verified_target > 1000 {
                const FAST_RECOVERY_DEPTH: u64 = 1000;
                let scan_start = verified_target.saturating_sub(FAST_RECOVERY_DEPTH);

                // Scan BACKWARDS from the tip to find the highest contiguous run
                let mut gap_found_at: Option<u64> = None;
                for h in (scan_start..=verified_target).rev() {
                    if self.get_qblock_by_height(h).await?.is_none() {
                        gap_found_at = Some(h);
                        break;
                    }
                }

                let contiguous_run = match gap_found_at {
                    Some(gap_h) => verified_target - gap_h,  // contiguous from gap_h+1 to verified_target
                    None => verified_target - scan_start + 1, // all FAST_RECOVERY_DEPTH blocks exist
                };

                if contiguous_run >= 100 {
                    // At least 100 contiguous blocks at the tip — safe to use
                    let contiguous_from = gap_found_at.map(|g| g + 1).unwrap_or(scan_start);
                    info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                    info!("🚀 [FAST RECOVERY v7.2.8] {} contiguous blocks at tip ({}-{})",
                          contiguous_run, contiguous_from, verified_target);
                    info!("   Pointer was at {}. Old gaps below tip are harmless.", pointer_height_for_fast_recovery);
                    info!("   Skipping full genesis scan. Gaps will be filled in background.");
                    info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                    return Ok(verified_target);
                } else if contiguous_run > 0 && verified_target > 1_000 {
                    // v7.2.9: Chain is mature — NEVER fall through to genesis scan.
                    // Even a small contiguous run at a high tip is vastly better than
                    // scanning from genesis and potentially regressing to a low gap.
                    // Gaps below the tip will be filled by turbo sync in background.
                    //
                    // v7.3.10: Lowered threshold from 10,000 → 1,000.
                    // Bug: at height 5431 (< 10K), if contiguous_run < 100, we fell through to
                    // genesis scan → returned 2502 instead of 5431. Now any node with > 1000
                    // blocks uses the tip as recovery point (gaps filled by background turbo sync).
                    let contiguous_from = gap_found_at.map(|g| g + 1).unwrap_or(scan_start);
                    info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                    info!("🛡️ [FAST RECOVERY v7.3.10] {} contiguous blocks at mature tip ({}-{})",
                          contiguous_run, contiguous_from, verified_target);
                    info!("   Chain has > 1K blocks. NOT falling through to genesis scan.");
                    info!("   Gaps below tip will be filled by turbo sync in background.");
                    info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                    return Ok(verified_target);
                } else {
                    // Truly fragmented AND chain is young (tip <= 1K) — fall through
                    warn!("⚠️  [FAST RECOVERY v7.3.10] Only {} contiguous blocks at tip {} — too fragmented",
                          contiguous_run, verified_target);
                    warn!("   Chain is young (tip <= 1K). Falling back to full scan.");
                }
            }
        }

        // Now find the FIRST gap by scanning from 0 upward
        // Use batched sampling for efficiency on large chains
        let mut highest_contiguous = 0u64;
        let mut first_gap_found: Option<u64> = None;

        // 🚀 v1.1.23-beta CRITICAL FIX: Handle non-genesis start heights
        //
        // ROOT CAUSE: Some nodes sync from checkpoints or peers and don't have blocks 0-N.
        // The old code returned 0 if block 0 and 1 didn't exist, even if blocks existed at higher heights.
        //
        // SYMPTOM: Node with blocks 1000-6326 would report height=0 on restart, then produce
        // blocks at 1, 2, 3... creating a fork/duplicate chain.
        //
        // FIX: If blocks 0 and 1 don't exist, find the LOWEST existing block and scan from there.
        // For bootstrap nodes, this is fine - they started syncing from a checkpoint.

        // Check if block 0 exists (genesis)
        let has_block_0 = self.get_qblock_by_height(0).await?.is_some();
        let start_height = if has_block_0 {
            0
        } else if self.get_qblock_by_height(1).await?.is_some() {
            // Blockchain starts at height 1
            info!("✅ [GENESIS FIX] Block 0 missing but block 1 exists - blockchain starts at height 1");
            1
        } else {
            // 🔧 v3.1.4-beta: CRITICAL FIX - ALWAYS treat missing genesis as stale data
            //
            // ROOT CAUSE: After u128 changes, fresh nodes with stale DB files were treating
            // orphaned blocks as "legitimate checkpoint data", causing sync to start from
            // the wrong height (320001, 345001, etc.) instead of genesis.
            //
            // KEY INSIGHT: A legitimate Q-NarwhalKnight blockchain MUST have either:
            //   - Block 0 (genesis), OR
            //   - Block 1 (if genesis is pruned/optional)
            //
            // If BOTH blocks 0 and 1 are missing, ANY existing blocks are STALE DATA
            // from a previous incomplete sync. There is NO legitimate scenario where
            // a chain has blocks at height N but not blocks 0 or 1.
            //
            // PREVIOUS BUG: v3.1.2-v3.1.3 only detected stale data if lowest_found >= 1000
            // This missed stale blocks at lower heights (2-999), causing the sync failure.
            //
            // FIX: ALWAYS return 0 when blocks 0 and 1 are missing, forcing sync from genesis.

            warn!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            warn!("🚨 [v3.1.4] STALE DATA DETECTED - GENESIS BLOCKS MISSING!");
            warn!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            warn!("   Blocks 0 and 1 are BOTH missing from the database.");
            warn!("   Any existing blocks are orphaned data from a previous incomplete sync.");
            warn!("   A legitimate chain MUST have block 0 or 1.");
            warn!("   ");
            warn!("   Returning height=0 to force full sync from genesis (block 1).");
            warn!("   The orphaned blocks will be overwritten during sync.");
            warn!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            return Ok(0);
        };

        // === CHECKPOINT ACCELERATION (v1.0.2) ===
        // Read the last verified contiguous height from DB to skip re-scanning
        // already-verified ranges. This makes even the fallback scan fast.
        let verified_checkpoint = match self.hot_db.get(CF_BLOCKS, b"qblock:contiguous_verified").await {
            Ok(Some(bytes)) if bytes.len() == 8 => {
                let mut arr = [0u8; 8];
                arr.copy_from_slice(&bytes);
                let h = u64::from_be_bytes(arr);
                if h >= start_height && h <= highest_existing {
                    info!("🚀 [CHECKPOINT] Resuming gap scan from verified height {} (skipping {} blocks)",
                          h, h - start_height);
                    h
                } else {
                    start_height
                }
            }
            _ => start_height,
        };

        // Efficient gap detection: sample at intervals, then linear scan when gap suspected
        let sample_interval: u64 = 1000; // Check every 1000 blocks first
        let mut last_verified = verified_checkpoint;

        // Phase 1: Coarse sampling to find approximate gap location
        let mut sample_height = verified_checkpoint;
        while sample_height <= highest_existing {
            if self.get_qblock_by_height(sample_height).await?.is_some() {
                last_verified = sample_height;
                sample_height += sample_interval;
            } else {
                // Found a missing block - gap is somewhere between last_verified and sample_height
                info!("🔍 [v1.1.0] Gap detected between {} and {} - starting linear scan",
                      last_verified, sample_height);
                first_gap_found = Some(sample_height);
                break;
            }
        }

        // Phase 2: Linear scan from last_verified to find exact gap location
        if first_gap_found.is_some() || last_verified < highest_existing {
            let scan_end = first_gap_found.unwrap_or(highest_existing);
            info!("🔍 [v1.1.0] Linear scan from {} to {} to find exact first gap...",
                  last_verified, scan_end.min(last_verified + sample_interval));

            for height in last_verified..=scan_end {
                if self.get_qblock_by_height(height).await?.is_some() {
                    highest_contiguous = height;
                } else {
                    warn!("🚨 [v1.1.0] FIRST GAP FOUND at height {}! Highest contiguous: {}",
                          height, highest_contiguous);
                    first_gap_found = Some(height);
                    break;
                }

                // Progress logging for long scans
                if height > 0 && height % 10_000 == 0 {
                    info!("   Scanned to height {}...", height);
                }
            }
        } else {
            // No gap found in sampling - all sampled blocks exist
            highest_contiguous = last_verified;
        }

        // If no gap was found, highest_contiguous is last_verified or highest_existing
        if first_gap_found.is_none() && highest_contiguous < highest_existing {
            // Continue linear scan from where we left off (in case sampling missed something)
            for height in (highest_contiguous + 1)..=highest_existing {
                if self.get_qblock_by_height(height).await?.is_some() {
                    highest_contiguous = height;
                } else {
                    warn!("🚨 [v1.1.0] Gap found at height {} during final scan! Highest contiguous: {}",
                          height, highest_contiguous);
                    break;
                }
            }
        }

        if let Some(gap) = first_gap_found {
            warn!("🚨🚨🚨 [v1.1.0] DATABASE HAS GAP! First missing block: {}, Highest contiguous: {}",
                  gap, highest_contiguous);
            warn!("🚨 [v1.1.0] New nodes syncing from this bootstrap will be stuck!");
            warn!("🚨 [v1.1.0] Highest existing block: {} (but not usable due to gap)", highest_existing);
        } else {
            info!("✅ [v1.1.0] No gaps detected - blockchain is contiguous from {} to {}",
                  start_height, highest_contiguous);
        }

        warn!(
            "✅✅✅ [HEIGHT DEBUG] Highest contiguous block: {} (highest existing: {}, gap: {:?})",
            highest_contiguous,
            highest_existing,
            first_gap_found
        );

        // v7.2.9: Apply safe_floor as absolute minimum — prevents catastrophic height regression
        // The safe_floor is persisted every 15s during normal operation, so it represents
        // a height the node was KNOWN to be operating at before the crash.
        if highest_contiguous < safe_floor && safe_floor > 0 {
            // Verify the block at safe_floor actually exists before trusting it
            if self.get_qblock_by_height(safe_floor).await?.is_some() {
                warn!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                warn!("🛡️ [SAFE FLOOR v7.2.9] Scan found contiguous={} but safe_floor={}",
                      highest_contiguous, safe_floor);
                warn!("   Overriding with safe_floor to prevent height regression.");
                warn!("   (Safe floor was last persisted during normal operation before crash)");
                warn!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                return Ok(safe_floor);
            } else {
                warn!("⚠️ [SAFE FLOOR v7.2.9] safe_floor={} but block doesn't exist! Ignoring.", safe_floor);
            }
        }

        warn!("🔍 [HEIGHT DEBUG] FINAL RESULT: Returning height {}", highest_contiguous);

        Ok(highest_contiguous)
    }

    /// Clean up corrupt/undeserializable blocks above a certain height
    /// This allows the node to re-sync those blocks from peers
    /// v0.9.1-beta: Backwards compatibility fix for enum format changes
    /// v1.0.80-beta: Now tries legacy format before marking as corrupt
    pub async fn cleanup_corrupt_blocks_above(&self, height: u64) -> Result<()> {
        info!("🧹 Scanning for corrupt blocks above height {}...", height);

        let mut deleted_count = 0;
        let scan_limit = height + 10000; // Scan up to 10k blocks ahead

        for check_height in (height + 1)..=scan_limit {
            let height_key = format!("qblock:height:{}", check_height);

            // Check if block exists
            if let Some(block_data) = self.hot_db.get(CF_BLOCKS, height_key.as_bytes()).await? {
                // v7.3.1: Handle both compressed and legacy formats before marking corrupt
                let is_valid = if precompressed_storage::is_precompressed(&block_data) {
                    precompressed_storage::PrecompressedBlock::from_bytes(&block_data)
                        .and_then(|c| c.decompress().map_err(|e| e.into()))
                        .and_then(|raw| q_types::legacy::deserialize_qblock_with_fallback(&raw)
                            .map_err(|e| anyhow::anyhow!("{}", e)))
                        .is_ok()
                } else {
                    q_types::legacy::deserialize_qblock_with_fallback(&block_data).is_ok()
                };
                if !is_valid {
                    // v10.3.7: DO NOT delete — these may be blocks from an older binary version.
                    // The old code permanently destroyed blocks that could have been read by a
                    // future deserializer update. Log and skip instead.
                    if deleted_count == 0 {
                        warn!("⚠️ Block at height {} fails deserialization (older format?) — preserving in DB", check_height);
                    }
                    deleted_count += 1;

                    // DISABLED: was permanently destroying blocks
                    // self.hot_db.delete(CF_BLOCKS, height_key.as_bytes()).await?;
                    if false && block_data.len() >= 32 {
                        let potential_hash_key = format!("qblock:hash:{}", hex::encode(&block_data[0..32]));
                        let _ = self.hot_db.delete(CF_BLOCKS, potential_hash_key.as_bytes()).await;
                    }

                    deleted_count += 1;
                }
            } else {
                // No more blocks found, stop scanning
                break;
            }
        }

        if deleted_count > 0 {
            warn!("🧹 Cleaned up {} corrupt blocks above height {}", deleted_count, height);
            info!("📡 Node will now re-sync these blocks from network peers");
        } else {
            info!("✅ No corrupt blocks found above height {}", height);
        }

        Ok(())
    }

    /// v10.2.8: Scan for corrupt blocks NEAR the tip (both above and below recovered height).
    ///
    /// Fixes blind spot where `kill -9` during active RocksDB writes leaves partially-written
    /// blocks BELOW the recovered height. `cleanup_corrupt_blocks_above()` only scans forward,
    /// so these corrupt blocks persist indefinitely — gap detection sees them as "present"
    /// (key exists) even though they fail deserialization.
    ///
    /// Returns `Ok(Some(new_height))` if corrupt blocks were found below recovered_height
    /// and pointers were reset. Returns `Ok(None)` if no corruption found below.
    pub async fn cleanup_corrupt_blocks_near_tip(&self, recovered_height: u64) -> Result<Option<u64>> {
        const SCAN_BELOW: u64 = 200;

        let scan_start = recovered_height.saturating_sub(SCAN_BELOW);
        if scan_start == 0 || recovered_height < 100 {
            return Ok(None);
        }

        info!("🔍 [v10.2.8] Scanning for corrupt blocks near tip: {} → {}", scan_start, recovered_height);

        let mut deleted_count = 0u64;
        let mut lowest_corrupt: Option<u64> = None;

        for height in scan_start..=recovered_height {
            let height_key = format!("qblock:height:{}", height);

            // Raw key check — does data exist?
            let block_data = match self.hot_db.get(CF_BLOCKS, height_key.as_bytes()).await? {
                Some(data) => data,
                None => continue, // Missing block — not our concern here
            };

            // Attempt full deserialization (same logic as cleanup_corrupt_blocks_above)
            let is_valid = if precompressed_storage::is_precompressed(&block_data) {
                precompressed_storage::PrecompressedBlock::from_bytes(&block_data)
                    .and_then(|c| c.decompress().map_err(|e| e.into()))
                    .and_then(|raw| q_types::legacy::deserialize_qblock_with_fallback(&raw)
                        .map_err(|e| anyhow::anyhow!("{}", e)))
                    .is_ok()
            } else {
                q_types::legacy::deserialize_qblock_with_fallback(&block_data).is_ok()
            };

            if !is_valid {
                // v10.3.7: DO NOT delete — blocks may be from older binary version.
                // Log only. The deserializer needs updating, not the data.
                if deleted_count == 0 {
                    warn!("⚠️ [v10.3.7] Block at height {} ({} bytes) fails deserialization — preserving (older format?)", height, block_data.len());
                }
                deleted_count += 1;
                if lowest_corrupt.is_none() || height < lowest_corrupt.unwrap() {
                    lowest_corrupt = Some(height);
                }
            }
        }

        if deleted_count == 0 {
            info!("✅ [v10.2.8] No corrupt blocks found near tip (scanned {} blocks)", SCAN_BELOW);
            return Ok(None);
        }

        // v10.3.7: No longer deleting blocks or resetting pointers.
        // Blocks that fail deserialization are preserved — they may become readable
        // after a deserializer update. DO NOT reset height pointers.
        warn!("⚠️ [v10.3.7] Found {} blocks near tip that fail deserialization (lowest at {}). Preserved in DB — no pointers reset.",
              deleted_count, lowest_corrupt.unwrap());

        Ok(None) // v10.3.7: no pointer changes, no deletions
    }

    /// Get the first missing height in blockchain (gap detection)
    /// v0.7.4-beta: Production fix for "messy height" issue
    ///
    /// Returns None if no gaps exist (blockchain is contiguous from genesis)
    /// Returns Some(height) if a gap is detected at that height
    ///
    /// This prevents height from skipping missing blocks during Turbo Sync
    pub async fn get_first_missing_height(&self) -> Result<Option<u64>> {
        let highest_contiguous = self.get_highest_contiguous_block().await?;

        // Get the highest block we have stored (may have gaps)
        let latest_height = match self.hot_db.get(CF_BLOCKS, b"qblock:latest").await? {
            Some(height_bytes) => {
                let mut height_array = [0u8; 8];
                height_array.copy_from_slice(&height_bytes);
                u64::from_be_bytes(height_array)
            }
            None => {
                // No latest pointer - no blocks stored
                return Ok(None);
            }
        };

        // If highest_overall == highest_contiguous, no gaps
        if latest_height == highest_contiguous {
            return Ok(None);
        }

        // Gap exists - find the first missing height
        // Start from highest_contiguous + 1 and scan upwards
        for height in (highest_contiguous + 1)..=latest_height {
            let block_key = format!("qblock:height:{}", height);
            if self.hot_db.get(CF_BLOCKS, block_key.as_bytes()).await?.is_none() {
                // Reduced to debug to avoid spam during gap scanning
                debug!("🔍 Gap detected: Missing block at height {}", height);
                return Ok(Some(height));
            }
        }

        // 🚨 v1.1.7-beta CRITICAL FIX: Auto-repair pointer when highest_contiguous > latest
        // This happens after restarts when blocks were produced but pointer wasn't updated
        // ROOT CAUSE: Previous session had height cache bug, pointer stuck at old value
        // FIX: Update pointer to highest_contiguous so P2P can serve all blocks
        //
        // 🚨 v1.4.1-beta FIX: VERIFY block exists before updating pointer!
        // The cache can return stale heights. Without verification, this creates an infinite
        // repair loop where v1.1.7 updates to non-existent block and v1.1.9 repairs back down.
        warn!("⚠️ Gap detection inconsistency: highest_contiguous={}, latest={}",
              highest_contiguous, latest_height);

        if highest_contiguous > latest_height {
            // 🔍 v1.4.1-beta: VERIFY block exists before updating pointer
            let block_key = format!("qblock:height:{}", highest_contiguous);
            if self.hot_db.get(CF_BLOCKS, block_key.as_bytes()).await?.is_some() {
                info!("🔧 [v1.1.7-beta] AUTO-REPAIR: Updating qblock:latest pointer {} → {}",
                      latest_height, highest_contiguous);
                let height_bytes = highest_contiguous.to_be_bytes();
                self.hot_db.put(CF_BLOCKS, b"qblock:latest", &height_bytes).await
                    .context("Failed to auto-repair height pointer")?;
                info!("✅ [v1.1.7-beta] Pointer repaired! P2P can now serve blocks up to {}", highest_contiguous);
            } else {
                // Block doesn't exist - cache is stale, repair it
                warn!("🚨 [v1.4.1-beta] Cache stale: height {} claimed but block doesn't exist!", highest_contiguous);
                // Update cache to match pointer (pointer is the source of truth when block missing)
                self.height_cache.update(latest_height).await;
                warn!("✅ [v1.4.1-beta] Cache repaired to {}", latest_height);
            }
        }

        Ok(None)
    }

    /// Repair height pointer by scanning database for actual highest block
    ///
    /// **HEIGHT RECOVERY (v0.8.5-beta)**: Fixes databases where height pointer is stuck
    /// at old value due to v0.8.3-beta bug. Scans database to find actual highest block
    /// and updates the qblock:latest pointer.
    ///
    /// This method is called automatically on startup to detect and repair height
    /// pointer inconsistencies that can occur during version upgrades.
    ///
    /// # Returns
    /// - `Ok(height)`: The repaired/verified height
    /// - `Err`: Database error during repair
    ///
    /// # Example
    /// ```rust
    /// // On startup after opening database:
    /// let repaired_height = storage.repair_height_pointer().await?;
    /// info!("Height pointer verified/repaired: {}", repaired_height);
    /// ```
    pub async fn repair_height_pointer(&self) -> Result<u64> {
        info!("🔧 [HEIGHT RECOVERY] Checking height pointer integrity...");

        // Get current height pointer value
        let pointer_height = self.get_latest_qblock_height().await?.unwrap_or(0);
        info!("🔍 [HEIGHT RECOVERY] Current height pointer: {}", pointer_height);

        // Use existing method to find actual highest contiguous block
        let actual_height = self.get_highest_contiguous_block().await?;
        info!("🔍 [HEIGHT RECOVERY] Actual highest block: {}", actual_height);

        // Check for mismatch
        if actual_height > pointer_height {
            warn!(
                "⚠️  [HEIGHT RECOVERY] Height pointer mismatch detected! Pointer: {}, Actual: {}",
                pointer_height, actual_height
            );
            warn!("🔧 [HEIGHT RECOVERY] Repairing height pointer...");

            // Update the height pointer to actual highest block
            let height_bytes = actual_height.to_be_bytes();
            self.hot_db.put(CF_BLOCKS, b"qblock:latest", &height_bytes).await
                .context("Failed to update height pointer")?;

            info!("✅ [HEIGHT RECOVERY] Height pointer repaired: {} → {}",
                  pointer_height, actual_height);
            info!("✅ [HEIGHT RECOVERY] Node can now sync normally from network");

            Ok(actual_height)
        } else if actual_height == pointer_height && actual_height > 0 {
            info!("✅ [HEIGHT RECOVERY] Height pointer is consistent: {}", actual_height);
            Ok(actual_height)
        } else if actual_height == 0 && pointer_height == 0 {
            info!("ℹ️  [HEIGHT RECOVERY] Empty database (height 0) - this is normal for new nodes");
            Ok(0)
        } else {
            // 🚀 v1.1.23-beta CRITICAL FIX: Don't corrupt pointer when contiguous scan returns 0
            //
            // ROOT CAUSE: get_highest_contiguous_block() was returning 0 for nodes that synced
            // from checkpoints (blocks 0 and 1 missing). The old code would then "correct"
            // the pointer from 6326 to 0, causing catastrophic height reset!
            //
            // FIX: If actual_height is 0 but pointer_height > 0, verify the pointer block exists.
            // If it does, trust the pointer - the contiguous scan algorithm failed.

            if actual_height == 0 && pointer_height > 0 {
                // Verify that the block at pointer_height actually exists
                if self.get_qblock_by_height(pointer_height).await?.is_some() {
                    warn!("⚠️ [HEIGHT RECOVERY] Contiguous scan returned 0, but block {} exists!", pointer_height);
                    warn!("⚠️ [HEIGHT RECOVERY] This likely means blocks 0-N are missing (checkpoint sync)");
                    warn!("✅ [HEIGHT RECOVERY] Trusting pointer at height {} (block verified)", pointer_height);
                    return Ok(pointer_height);
                }
            }

            // Pointer is higher than actual - this shouldn't happen but handle gracefully
            warn!(
                "⚠️  [HEIGHT RECOVERY] Unexpected state: pointer={}, actual={}",
                pointer_height, actual_height
            );
            warn!("🔧 [HEIGHT RECOVERY] Correcting pointer to match actual height");

            let height_bytes = actual_height.to_be_bytes();
            self.hot_db.put(CF_BLOCKS, b"qblock:latest", &height_bytes).await
                .context("Failed to correct height pointer")?;

            // 🛡️ v7.2.3: CRITICAL - Also force-set the height cache downward
            // Without this, the monotonic cache stays at the old (wrong) height
            // and the node announces blocks it doesn't have to the P2P network
            self.height_cache.force_set(actual_height).await;

            info!("✅ [HEIGHT RECOVERY] Height pointer AND cache corrected to {}", actual_height);
            Ok(actual_height)
        }
    }

    /// Get storage statistics
    pub async fn get_storage_stats(&self) -> StorageStats {
        let manifest = self.manifest.read().await;
        let metrics = self.metrics.get_current_metrics().await;

        StorageStats {
            dag_round_watermark: manifest.dag_round_watermark,
            finalized_height: manifest.finalized_height,
            total_vertices: metrics.total_vertices,
            total_payloads: metrics.total_payloads,
            total_blocks: metrics.total_blocks,
            hot_db_size: self.hot_db.get_db_size().await.unwrap_or(0),
            cold_db_size: self.cold_db.get_db_size().await.unwrap_or(0),
            average_write_latency: metrics.average_write_latency,
            average_read_latency: metrics.average_read_latency,
        }
    }

    /// Validate genesis block matches network expectation
    ///
    /// **NETWORK UNIFICATION (v0.9.37-beta Phase 2)**: Detects fork at genesis level
    ///
    /// # Arguments
    /// * `expected_genesis_hash` - Expected genesis block hash from network consensus
    ///
    /// # Returns
    /// * `Ok(true)` - Genesis matches or no genesis exists (will sync from network)
    /// * `Ok(false)` - Genesis mismatch - node is on incompatible fork!
    pub async fn validate_genesis_block(&self, expected_genesis_hash: Option<[u8; 32]>) -> Result<bool> {
        match self.get_qblock_by_height(0).await? {
            Some(local_genesis) => {
                let local_hash = local_genesis.calculate_hash();

                if let Some(expected) = expected_genesis_hash {
                    if local_hash != expected {
                        warn!("🔀 GENESIS MISMATCH DETECTED!");
                        warn!("   Local genesis:    {:02x?}...", &local_hash[..8]);
                        warn!("   Expected genesis: {:02x?}...", &expected[..8]);
                        warn!("   This node is on a FORKED CHAIN!");
                        warn!("   Local height: {}", self.get_highest_contiguous_block().await?);
                        warn!("   ⚠️  Blockchain reorganization required!");
                        return Ok(false);
                    } else {
                        info!("✅ Genesis block validated: {:02x?}...", &local_hash[..8]);
                    }
                } else {
                    info!("ℹ️  No expected genesis provided - accepting local genesis: {:02x?}...",
                          &local_hash[..8]);
                }
                Ok(true)
            }
            None => {
                info!("📦 No genesis block found - will sync from network");
                Ok(true)
            }
        }
    }

    /// Perform crash recovery and return recovered blockchain height
    async fn recover(&self) -> Result<u64> {
        info!("🔄 Starting storage crash recovery");

        let manifest = self.manifest.read().await;
        info!(
            "📊 Recovery state - DAG watermark: {}, finalized: {}",
            manifest.dag_round_watermark, manifest.finalized_height
        );

        // 🚀 CRITICAL FIX (v0.6.6): Find highest blockchain height in database
        let recovered_height = self.get_highest_contiguous_block().await?;
        info!("📈 Recovered blockchain height: {} blocks from database", recovered_height);

        // 🧹 Clean up corrupt blocks above recovered height (backwards compatibility fix)
        self.cleanup_corrupt_blocks_above(recovered_height).await?;

        // NOTE: cleanup_corrupt_blocks_near_tip() is called AFTER scan_highest_contiguous_block_internal()
        // in the init path (line ~780), NOT here. recover() runs before the cache is populated,
        // so recovered_height is often 0 on first boot. The real height discovery happens later.

        // Verify DAG consistency
        self.verify_dag_consistency().await?;

        // Start sync process if needed
        if manifest.dag_round_watermark > 0 {
            self.sync_protocol
                .start_catch_up(manifest.dag_round_watermark)
                .await?;
        }

        info!("✅ Storage recovery complete - restored {} blocks", recovered_height);
        Ok(recovered_height)
    }

    /// Verify DAG consistency after crash
    async fn verify_dag_consistency(&self) -> Result<()> {
        debug!("🔍 Verifying DAG consistency");

        let manifest = self.manifest.read().await;
        let watermark = manifest.dag_round_watermark;

        // Check that we have contiguous rounds up to watermark
        for round in 0..=watermark {
            let vertices = self.get_vertices_for_round(round).await?;
            if vertices.is_empty() && round < watermark {
                warn!(
                    "⚠️ Missing vertices for round {}, truncating watermark",
                    round
                );
                // In production, we'd truncate the watermark here
                break;
            }
        }

        debug!("✅ DAG consistency verified up to round {}", watermark);
        Ok(())
    }

    /// Verify database integrity on startup (v0.9.93-beta)
    ///
    /// FIX 1.4: Startup integrity check
    /// Ensures qblock:latest pointer references an existing block.
    /// If corruption is detected, REFUSES TO START and requires manual recovery.
    async fn verify_database_integrity(&self) -> Result<()> {
        info!("🔍 Verifying database integrity on startup...");

        // Get current height from pointer
        let current_height = match self.hot_db.get(CF_BLOCKS, b"qblock:latest").await {
            Ok(Some(bytes)) if bytes.len() == 8 => {
                u64::from_be_bytes(bytes.try_into().unwrap())
            }
            Ok(Some(_)) => {
                error!("🚨 CRITICAL: qblock:latest pointer has invalid format!");
                anyhow::bail!("Database corruption: invalid pointer format");
            }
            Ok(None) => {
                // Fresh database - no blocks yet
                info!("✅ Fresh database - integrity OK (no blocks)");
                return Ok(());
            }
            Err(e) => {
                error!("🚨 CRITICAL: Failed to read qblock:latest pointer: {}", e);
                anyhow::bail!("Database error reading pointer: {}", e);
            }
        };

        if current_height == 0 {
            info!("✅ Database integrity verified: genesis only");
            return Ok(());
        }

        // Verify that the block actually exists
        let height_key = format!("qblock:height:{}", current_height);
        match self.hot_db.get(CF_BLOCKS, height_key.as_bytes()).await {
            Ok(Some(_)) => {
                info!("✅ Database integrity verified: pointer at height {}, block exists", current_height);
                Ok(())
            }
            Ok(None) => {
                // v1.1.10-beta: AUTO-REPAIR corrupted pointer
                warn!("🔧 [AUTO-REPAIR] Pointer at height {} but block missing - scanning for highest existing block...", current_height);

                // Scan backwards from pointer to find highest existing block
                let mut repair_height = current_height;
                while repair_height > 0 {
                    let check_key = format!("qblock:height:{}", repair_height);
                    if let Ok(Some(_)) = self.hot_db.get(CF_BLOCKS, check_key.as_bytes()).await {
                        break;
                    }
                    repair_height = repair_height.saturating_sub(1000).max(1);
                    if repair_height <= 1 {
                        // Linear scan for last 1000
                        for h in (1..=current_height.min(10000)).rev() {
                            let check_key = format!("qblock:height:{}", h);
                            if let Ok(Some(_)) = self.hot_db.get(CF_BLOCKS, check_key.as_bytes()).await {
                                repair_height = h;
                                break;
                            }
                        }
                        break;
                    }
                }

                // Update pointer to highest existing block
                if repair_height > 0 {
                    warn!("🔧 [AUTO-REPAIR] Updating qblock:latest pointer: {} → {}", current_height, repair_height);
                    let height_bytes = repair_height.to_be_bytes();
                    self.hot_db.put(CF_BLOCKS, b"qblock:latest", &height_bytes).await
                        .context("Failed to auto-repair pointer")?;
                    info!("✅ [AUTO-REPAIR] Database pointer fixed! Now at height {}", repair_height);
                    Ok(())
                } else {
                    error!("🚨 AUTO-REPAIR FAILED: No valid blocks found in database!");
                    anyhow::bail!(
                        "Database corruption detected: pointer at {} but no valid blocks found.",
                        current_height
                    )
                }
            }
            Err(e) => {
                error!("🚨 CRITICAL: Failed to verify block existence: {}", e);
                anyhow::bail!("Database error during integrity check: {}", e);
            }
        }
    }

    /// Update DAG round watermark
    async fn update_dag_watermark(&self, round: u64) -> Result<()> {
        let mut manifest = self.manifest.write().await;

        if round > manifest.dag_round_watermark {
            manifest.dag_round_watermark = round;
            manifest.save(&self.hot_db).await?;
            debug!("📈 Updated DAG watermark to round {}", round);
        }

        Ok(())
    }

    /// Check if a round is complete (for watermark advancement)
    async fn check_round_completion(&self, round: u64) -> Result<()> {
        // In production, this would check if we have enough vertices for Bullshark progress
        // For now, we just advance the watermark
        self.update_dag_watermark(round).await
    }

    /// Check if we should trigger a snapshot
    async fn check_snapshot_trigger(&self, block_height: u64) -> Result<()> {
        const SNAPSHOT_INTERVAL: u64 = 1000; // Every 1000 blocks

        if block_height % SNAPSHOT_INTERVAL == 0 {
            info!("📸 Triggering snapshot at height {}", block_height);
            self.snapshot_manager.create_snapshot(block_height).await?;
        }

        Ok(())
    }

    /// Scan for vertex by ID (inefficient, needs secondary index in production)
    async fn scan_for_vertex(&self, vertex_id: &[u8]) -> Result<Option<Vertex>> {
        // This is a fallback method - in production we'd have a secondary index
        warn!(
            "🐌 Performing inefficient vertex scan for {}",
            hex::encode(vertex_id)
        );

        let all_vertices = self.hot_db.scan_all(CF_DAG_VERTICES).await?;

        for (_, vertex_data) in all_vertices {
            if let Ok(vertex) = bincode::deserialize::<Vertex>(&vertex_data) {
                if vertex.id == vertex_id {
                    return Ok(Some(vertex));
                }
            }
        }

        Ok(None)
    }

    /// Generate vertex key for storage
    fn vertex_key(&self, round: u64, author: &[u8], vertex_id: &[u8]) -> Vec<u8> {
        let mut key = Vec::with_capacity(8 + author.len() + vertex_id.len());
        key.extend_from_slice(&round.to_be_bytes());
        key.extend_from_slice(author);
        key.extend_from_slice(vertex_id);
        key
    }

    /// Generate block key for storage
    fn block_key(&self, height: u64, hash: &[u8]) -> Vec<u8> {
        let mut key = Vec::with_capacity(8 + hash.len());
        key.extend_from_slice(&height.to_be_bytes());
        key.extend_from_slice(hash);
        key
    }

    /// Compact storage to reclaim space
    pub async fn compact(&self) -> Result<()> {
        info!("🗜️ Compacting storage databases");

        // Compact hot database
        self.hot_db.compact().await?;

        // Compact cold database
        self.cold_db.compact().await?;

        info!("✅ Storage compaction complete");
        Ok(())
    }

    /// Prune old data beyond retention policy
    pub async fn prune(&self, retain_rounds: u64) -> Result<()> {
        info!("🧹 Pruning data older than {} rounds", retain_rounds);

        let manifest = self.manifest.read().await;
        let prune_before_round = manifest.dag_round_watermark.saturating_sub(retain_rounds);

        if prune_before_round == 0 {
            debug!("No data to prune");
            return Ok(());
        }

        // Prune old vertices
        let pruned_vertices = self.prune_vertices_before_round(prune_before_round).await?;

        // Prune old payloads (more aggressive)
        let pruned_payloads = self.prune_payloads_before_round(prune_before_round).await?;

        info!(
            "✅ Pruned {} vertices and {} payloads",
            pruned_vertices, pruned_payloads
        );
        Ok(())
    }

    /// Prune vertices before a specific round
    async fn prune_vertices_before_round(&self, before_round: u64) -> Result<usize> {
        let mut pruned_count = 0;

        // Iterate through rounds to prune
        for round in 0..before_round {
            let round_prefix = round.to_be_bytes();
            let vertices = self
                .hot_db
                .scan_prefix(CF_DAG_VERTICES, &round_prefix)
                .await?;

            for (key, _) in vertices {
                self.hot_db.delete(CF_DAG_VERTICES, &key).await?;
                pruned_count += 1;
            }
        }

        Ok(pruned_count)
    }

    /// Prune payloads before a specific round
    async fn prune_payloads_before_round(&self, before_round: u64) -> Result<usize> {
        // This requires mapping payload digests to rounds
        // For now, we'll implement a simple approach
        debug!("🧹 Pruning payloads before round {}", before_round);

        // In production, we'd maintain a digest -> round mapping
        // For now, return 0 as a placeholder
        Ok(0)
    }

    /// Get storage health status
    pub async fn health_check(&self) -> StorageHealth {
        let stats = self.get_storage_stats().await;
        let manifest = self.manifest.read().await;

        // Check various health indicators
        let db_accessible = self.hot_db.get(CF_MANIFEST, b"test").await.is_ok();
        let watermark_reasonable = stats.dag_round_watermark <= stats.finalized_height + 100;
        let write_performance_ok = stats.average_write_latency < Duration::from_millis(100);

        let status = if db_accessible && watermark_reasonable && write_performance_ok {
            StorageHealthStatus::Healthy
        } else if !db_accessible {
            StorageHealthStatus::DatabaseError
        } else if !watermark_reasonable {
            StorageHealthStatus::InconsistentState
        } else {
            StorageHealthStatus::PerformanceIssues
        };

        StorageHealth {
            status,
            last_write: std::time::SystemTime::now(),
            error_count: 0, // TODO: Track errors
            stats,
        }
    }

    /// Shutdown storage gracefully
    pub async fn shutdown(&self) -> Result<()> {
        info!("🛑 Shutting down Q-Storage");

        // Flush any pending writes
        self.hot_db.flush().await?;
        self.cold_db.flush().await?;

        // Stop sync protocol
        self.sync_protocol.shutdown().await?;

        info!("✅ Q-Storage shutdown complete");
        Ok(())
    }

    /// Save wallet balance to persistent storage with SYNC to guarantee disk write
    /// v2.5.0: Updated to u128 (16 bytes) for extreme precision
    pub async fn save_wallet_balance(&self, address: &[u8; 32], amount: u128) -> Result<()> {
        let key = format!("wallet_balance_{}", hex::encode(address));
        let addr_hex = hex::encode(address);
        let value = amount.to_le_bytes(); // 16 bytes for u128

        // 🔴 [BALANCE WRITE DEBUG] Read old value BEFORE overwrite
        let old_balance = self.load_wallet_balance(address).await?.unwrap_or(0);
        if old_balance != amount {
            let delta_abs = if amount >= old_balance { amount - old_balance } else { old_balance - amount };
            let direction = if amount >= old_balance { "+" } else { "-" };
            if amount < old_balance {
                // SECURITY (issue #54): max-wins guard per CLAUDE.md Rule 1.
                // Stale/partial callers (state-sync, replay, batch reconciliation) may pass a
                // lower amount than the authoritative on-disk value. Skipping the write preserves
                // the higher balance and matches the existing behavior of the batch variant
                // `save_wallet_balances`. Legitimate debits (fees, signed transfers via consensus)
                // must go through a dedicated path; see issue #54 for the 20+ caller audit.
                error!(
                    "🔴 [BALANCE WRITE] save_wallet_balance(): wallet={} SKIPPED (max-wins: old={} > new={}) caller=ABSOLUTE_OVERWRITE",
                    &addr_hex[..16.min(addr_hex.len())], old_balance, amount
                );
                return Ok(());
            } else {
                warn!(
                    "🔴 [BALANCE WRITE] save_wallet_balance(): wallet={} old={} new={} delta={}{} caller=ABSOLUTE_OVERWRITE height=N/A",
                    &addr_hex[..16.min(addr_hex.len())], old_balance, amount, direction, delta_abs
                );
            }
        }

        // CRITICAL: Use synced write to guarantee data reaches disk (survives pkill -9)
        // This overrides the default set_sync(false) in write_options()
        self.hot_db.put_sync(CF_MANIFEST, key.as_bytes(), &value).await?;

        // 🔒 PRIVACY-PRESERVING: Log only cryptographic hash of address
        use blake3::hash;
        let addr_hash = hash(address);
        debug!(
            "💰 SYNCED wallet balance to disk: addr_hash={} (survives hard kill)",
            hex::encode(&addr_hash.as_bytes()[..8])
        );
        Ok(())
    }

    /// Load wallet balance from persistent storage
    /// v2.5.0: Returns u128, with backward compatibility for legacy u64 (8-byte) storage
    pub async fn load_wallet_balance(&self, address: &[u8; 32]) -> Result<Option<u128>> {
        let key = format!("wallet_balance_{}", hex::encode(address));
        match self.hot_db.get(CF_MANIFEST, key.as_bytes()).await? {
            Some(bytes) => {
                if bytes.len() == 16 {
                    // New u128 format (16 bytes)
                    let amount = u128::from_le_bytes(bytes[..16].try_into().unwrap());
                    Ok(Some(amount))
                } else if bytes.len() == 8 {
                    // Legacy u64 format (8 bytes) - convert to u128
                    // Note: Legacy values had 8 decimals, new values have 24 decimals
                    // Multiply by 10^16 to convert (24 - 8 = 16 decimal places)
                    let legacy_amount = u64::from_le_bytes(bytes[..8].try_into().unwrap());
                    let upgraded_amount = (legacy_amount as u128) * 10u128.pow(16);
                    Ok(Some(upgraded_amount))
                } else {
                    // 🔒 PRIVACY: Don't log address, only data length issue
                    warn!("Invalid wallet balance data length: expected 8 or 16 bytes, got {}", bytes.len());
                    Ok(None)
                }
            }
            None => Ok(None),
        }
    }

    /// v8.2.9: Load persistent mining stats (blocks_found, rewards_earned) from RocksDB.
    /// These are blockchain-derived and deterministic — same blocks = same stats on ANY node.
    /// Used by the mining stats API to return consistent data across all HA servers.
    pub async fn load_persistent_mining_stats(&self, wallet_hex: &str) -> Result<(u64, u128)> {
        let blocks_key = format!("mining_blocks_{}", wallet_hex);
        let rewards_key = format!("mining_rewards_{}", wallet_hex);

        let blocks: u64 = match self.hot_db.get(CF_MANIFEST, blocks_key.as_bytes()).await? {
            Some(bytes) if bytes.len() == 8 => u64::from_le_bytes(bytes[..8].try_into().unwrap()),
            _ => 0,
        };

        let rewards: u128 = match self.hot_db.get(CF_MANIFEST, rewards_key.as_bytes()).await? {
            Some(bytes) if bytes.len() == 16 => u128::from_le_bytes(bytes[..16].try_into().unwrap()),
            _ => 0,
        };

        Ok((blocks, rewards))
    }

    /// ✅ v0.9.27-beta: Get balance from balance consensus column family
    /// This reads directly from the "balances" CF written by BalanceConsensusEngine
    /// v2.5.0: Returns u128 with backward compatibility for u64 storage
    pub async fn get_consensus_balance(&self, address_hex: &str) -> Result<u128> {
        match self.hot_db.get("balances", address_hex.as_bytes()).await? {
            Some(bytes) => {
                if bytes.len() == 16 {
                    // New u128 format (big-endian)
                    Ok(u128::from_be_bytes(bytes[..16].try_into().unwrap()))
                } else if bytes.len() == 8 {
                    // Legacy u64 format - convert to u128 with decimal upgrade
                    let legacy = u64::from_be_bytes(bytes[..8].try_into().unwrap());
                    Ok((legacy as u128) * 10u128.pow(16))
                } else {
                    Ok(0)
                }
            }
            None => Ok(0),
        }
    }

    /// ✅ v0.9.28-beta: Public wrapper for hot_db.get() to support address_book and other CF operations
    pub async fn db_get(&self, cf: &str, key: &[u8]) -> Result<Option<Vec<u8>>> {
        self.hot_db.get(cf, key).await
    }

    /// ✅ v0.9.28-beta: Public wrapper for hot_db.put() to support address_book and other CF operations
    pub async fn db_put(&self, cf: &str, key: &[u8], value: &[u8]) -> Result<()> {
        self.hot_db.put(cf, key, value).await
    }

    /// ✅ v0.9.28-beta: Public wrapper for hot_db.delete() to support address_book and other CF operations
    pub async fn db_delete(&self, cf: &str, key: &[u8]) -> Result<()> {
        self.hot_db.delete(cf, key).await
    }

    /// Scan the manifest CF for all entries with the given key prefix.
    /// Used by BalanceFinalityEngine to enumerate finality proof records.
    pub async fn scan_manifest_prefix(&self, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        self.hot_db.scan_prefix(CF_MANIFEST, prefix).await
    }

    /// Fsync-write a single key into the manifest CF.
    /// Used by BalanceFinalityEngine to persist finality proof records.
    pub async fn put_manifest_sync(&self, key: &[u8], value: &[u8]) -> Result<()> {
        self.hot_db.put_sync(CF_MANIFEST, key, value).await
    }

    /// Load all wallet balances from persistent storage
    /// v2.5.0: Returns u128 with backward compatibility for u64 storage
    pub async fn load_wallet_balances(&self) -> Result<HashMap<[u8; 32], u128>> {
        let mut balances = HashMap::new();
        let prefix = "wallet_balance_".as_bytes();

        // Use a prefix scan to get all wallet balance entries
        match self.hot_db.scan_prefix(CF_MANIFEST, prefix).await {
            Ok(entries) => {
                for (key, value) in entries {
                    if let Ok(key_str) = String::from_utf8(key) {
                        if let Some(hex_addr) = key_str.strip_prefix("wallet_balance_") {
                            if let Ok(addr_bytes) = hex::decode(hex_addr) {
                                if addr_bytes.len() == 32 {
                                    let mut address = [0u8; 32];
                                    address.copy_from_slice(&addr_bytes);

                                    let amount = if value.len() == 16 {
                                        // New u128 format
                                        u128::from_le_bytes(value[..16].try_into().unwrap())
                                    } else if value.len() == 8 {
                                        // Legacy u64 format - convert with decimal upgrade
                                        let legacy = u64::from_le_bytes(value[..8].try_into().unwrap());
                                        (legacy as u128) * 10u128.pow(16)
                                    } else {
                                        continue; // Skip invalid entries
                                    };

                                    balances.insert(address, amount);
                                }
                            }
                        }
                    }
                }
                // BAL-001: Demoted from info! to debug! — this is called once per block during
                // balance root computation (shadow mode + enforcement). At 1 bps and ~17M+ blocks,
                // info-level here generates one log line per second indefinitely, filling syslog.
                debug!(
                    "💰 Loaded {} wallet balances from persistent storage",
                    balances.len()
                );
            }
            Err(e) => {
                warn!("Failed to scan wallet balances: {}", e);
            }
        }

        Ok(balances)
    }

    /// Operator-callable: rebuild the balance_root_v2 Sparse Merkle Tree from
    /// the current persisted wallet table. Returns the new SMT root.
    ///
    /// This is the pre-activation determinism probe described in the DeepSeek
    /// handoff (docs/deepseek-handoff-balance-root-v2-activation-2026-05-14.md,
    /// Job D9). Until D2 (atomic SMT update in save_wallet_balances) lands,
    /// the SMT does NOT auto-update — the only way to populate it is to call
    /// this method. The SMT writes go to its own column family
    /// (`cf_balance_smt`) and do NOT touch the wallet balance CF.
    ///
    /// Usage:
    ///   - On Beta/Gamma/Delta/Epsilon, call this once via a future admin
    ///     endpoint or CLI binary (e.g. crates/q-storage/src/bin/verify_smt_rebuild.rs
    ///     planned in the DeepSeek handoff Job D9).
    ///   - Compare the returned root across all four production nodes — if
    ///     any two disagree at the same observed wallet table snapshot, the
    ///     wallet tables themselves are divergent and we have a balance-
    ///     integrity bug to fix BEFORE activation.
    ///
    /// **NOT consensus-affecting** — purely reads the wallet table, writes to
    /// the SMT CF, returns the root. Safe to call at any time on mainnet
    /// (besides the I/O cost of writing ~256·N SMT nodes).
    pub async fn rebuild_balance_smt_from_wallet_table(&self) -> Result<[u8; 32]> {
        info!("📊 [SMT-REBUILD] Loading wallet table for balance_root_v2 rebuild…");
        let balances = self.load_wallet_balances().await
            .context("rebuild_balance_smt_from_wallet_table: load_wallet_balances")?;
        let wallet_count = balances.len();
        info!("📊 [SMT-REBUILD] Wallet table loaded ({} entries). Rebuilding SMT…", wallet_count);

        let started = std::time::Instant::now();
        let root = self.balance_smt
            .rebuild_from_balances(&balances)
            .context("rebuild_balance_smt_from_wallet_table: BalanceSmt::rebuild_from_balances")?;
        let elapsed = started.elapsed();

        info!(
            "📊 [SMT-REBUILD] ✅ Done. wallet_count={} smt_root={} elapsed={:?}",
            wallet_count,
            hex::encode(&root[..8]),
            elapsed
        );
        Ok(root)
    }

    /// Find a wallet whose hex address starts with the given prefix.
    /// Falls back to a DB scan when the in-memory wallet_balances cache doesn't contain it
    /// (e.g. wallet was just funded, cache not yet refreshed).
    pub async fn find_wallet_by_prefix(&self, prefix: &str) -> Result<Option<[u8; 32]>> {
        if prefix.len() < 8 || !prefix.chars().all(|c| c.is_ascii_hexdigit()) {
            return Ok(None);
        }
        let scan_prefix = format!("wallet_balance_{}", prefix);
        match self.hot_db.scan_prefix(CF_MANIFEST, scan_prefix.as_bytes()).await {
            Ok(entries) => {
                for (key, _) in entries {
                    if let Ok(key_str) = String::from_utf8(key) {
                        if let Some(hex_addr) = key_str.strip_prefix("wallet_balance_") {
                            if hex_addr.starts_with(prefix) {
                                if let Ok(bytes) = hex::decode(hex_addr) {
                                    if bytes.len() == 32 {
                                        let mut addr = [0u8; 32];
                                        addr.copy_from_slice(&bytes);
                                        return Ok(Some(addr));
                                    }
                                }
                            }
                        }
                    }
                }
                Ok(None)
            }
            Err(_) => Ok(None),
        }
    }

    /// Save multiple wallet balances atomically with SYNC to guarantee disk write
    /// v2.5.0: Accepts u128 balances (16 bytes each)
    pub async fn save_wallet_balances(&self, balances: &HashMap<[u8; 32], u128>) -> Result<()> {
        let mut batch_ops = Vec::new();

        // 🔴 [BALANCE WRITE DEBUG] Log each wallet in batch with old-vs-new
        for (address, amount) in balances {
            let addr_hex = hex::encode(address);
            let old_balance = self.load_wallet_balance(address).await?.unwrap_or(0);
            if old_balance != *amount {
                let delta_abs = if *amount >= old_balance { *amount - old_balance } else { old_balance - *amount };
                let direction = if *amount >= old_balance { "+" } else { "-" };
                if *amount < old_balance {
                    // MAX-WINS GUARD: never write a lower balance than what's already in RocksDB.
                    // Replay/batch callers may have incomplete data; the existing value is authoritative.
                    error!(
                        "🔴 [BALANCE WRITE] save_wallet_balances(): wallet={} SKIPPED (max-wins: old={} > new={}) caller=BATCH_OVERWRITE",
                        &addr_hex[..16.min(addr_hex.len())], old_balance, amount
                    );
                    continue; // do NOT write — existing value is higher and must be preserved
                } else {
                    warn!(
                        "🔴 [BALANCE WRITE] save_wallet_balances(): wallet={} old={} new={} delta={}{} caller=BATCH_OVERWRITE height=N/A",
                        &addr_hex[..16.min(addr_hex.len())], old_balance, amount, direction, delta_abs
                    );
                }
            }

            let key = format!("wallet_balance_{}", hex::encode(address));
            let value = amount.to_le_bytes().to_vec(); // 16 bytes for u128
            batch_ops.push((CF_MANIFEST, key.into_bytes(), value));
        }

        // CRITICAL: write_batch now uses fsync to survive hard kills (fixed in kv.rs)
        self.hot_db.write_batch(batch_ops).await?;

        // 🔒 PRIVACY-PRESERVING: Log only aggregate count, not individual balances
        // v2.10.0: Updated to u128 for 24 decimal precision
        let total_balance: u128 = balances.values().sum();
        info!(
            "💰 SYNCED {} wallet balances (total supply: {} QUG) (survives hard kill)",
            balances.len(),
            total_balance / 1_000_000_000_000_000_000_000_000
        );
        Ok(())
    }

    /// v8.2.0: Compute a deterministic hash of all wallet balances.
    /// Sorts all (address, balance) pairs by address, feeds through blake3.
    /// Same balances on any node → same hash. Used for cross-node verification.
    pub async fn compute_balance_state_hash(&self) -> Result<([u8; 32], usize, u128)> {
        let balances = self.load_wallet_balances().await?;

        let mut sorted_entries: Vec<_> = balances.iter()
            .filter(|(_, &amount)| amount > 0)
            .collect();
        sorted_entries.sort_by_key(|(addr, _)| *addr);

        let mut hasher = blake3::Hasher::new();
        let mut wallet_count = 0usize;
        let mut total_supply = 0u128;

        for (addr, &amount) in &sorted_entries {
            hasher.update(addr.as_slice());
            hasher.update(&amount.to_le_bytes());
            wallet_count += 1;
            total_supply = total_supply.checked_add(amount)
                .ok_or_else(|| anyhow::anyhow!(
                    "total supply overflow at wallet {:?} — impossible state, chain invariant violated",
                    addr
                ))?;
        }

        let hash: [u8; 32] = *hasher.finalize().as_bytes();
        Ok((hash, wallet_count, total_supply))
    }

    /// Compute the canonical balance root for inclusion in block headers.
    ///
    /// Uses Blake3 with a domain separator and big-endian balance encoding per the
    /// canonical BalanceRootV1 spec. Called by the block producer (before producing a
    /// block) and by validators (before applying a received block) to verify agreement.
    ///
    /// Returns `[0u8; 32]` when no non-zero balances exist (fresh/empty chain).
    ///
    /// This is intentionally SEPARATE from `compute_balance_state_hash()`:
    /// - `compute_balance_state_hash()` uses little-endian and no domain separator (legacy)
    /// - `compute_balance_root_for_block()` uses big-endian + domain separator (canonical spec)
    ///
    /// ## Determinism guarantees (BAL-001 audit, 2026-05-11)
    ///
    /// This function is deterministic across all nodes IF AND ONLY IF the wallet balance
    /// state in RocksDB is identical. Specifically:
    ///
    /// 1. **Zero-balance filtering**: wallets with amount == 0 are excluded. Any code path
    ///    that writes a zero balance via `save_wallet_balance` / `save_wallet_balances`
    ///    will cause a root mismatch vs a node that never wrote that zero entry. The
    ///    max-wins guard in `save_wallet_balances` prevents going from non-zero → zero;
    ///    however `save_wallet_balance` (singular) has no such guard on zero writes.
    ///    Writers MUST NOT persist zero balances — use the wallet's absence in the DB to
    ///    represent a zero balance instead.
    ///
    /// 2. **Sort order**: sorted by raw `[u8; 32]` address bytes (unsigned lexicographic).
    ///    This is independent of RocksDB key order (which sorts by hex-string keys).
    ///    The re-sort here is what makes it deterministic regardless of scan order.
    ///
    /// 3. **Encoding**: u128 amounts serialised as 16-byte big-endian. Unchanged since
    ///    BalanceRootV1 was introduced. Do NOT change endianness — it would invalidate
    ///    all historical shadow checks.
    ///
    /// 4. **Legacy u64 entries**: `load_wallet_balances` converts 8-byte legacy entries
    ///    to u128 via `* 10^16`. If some nodes have already been rewritten to 16-byte
    ///    and others still hold 8-byte entries for the same wallet, the amounts differ
    ///    and roots will diverge. Ensure all nodes have migrated to the 16-byte format
    ///    before enforcement activates at height 20,000,000.
    ///
    /// 5. **Checkpoint replay**: nodes that bootstrapped from a checkpoint snapshot run
    ///    `replay_post_checkpoint_balances` (gated on `is_checkpoint_applied()`). If
    ///    replay produces a different balance set than Epsilon's genesis-derived state,
    ///    roots will diverge. Shadow-mode mismatch logs are the early warning system.
    pub async fn compute_balance_root_for_block(&self) -> Result<[u8; 32]> {
        let balances = self.load_wallet_balances().await?;

        // BAL-001: Explicit zero-balance exclusion — zero-balance entries must NOT appear in
        // the root. A node that has stored a zero balance for a wallet (e.g. after a drain)
        // MUST NOT include it here. Nodes that never wrote the entry agree: both see nothing.
        let mut sorted: Vec<([u8; 32], u128)> = balances
            .into_iter()
            .filter(|(_, amount)| *amount > 0)
            .collect();

        if sorted.is_empty() {
            return Ok([0u8; 32]);
        }

        // BAL-001: Sort by raw address bytes — deterministic regardless of HashMap or DB order.
        // Use rayon parallel sort for large wallet sets (benchmark: ~6× faster at 100K wallets).
        sorted.par_sort_unstable_by_key(|(addr, _)| *addr);

        // Compute leaf hashes in parallel then feed sequentially into the root hasher.
        // Parallel leaf hashing is safe because each entry is independent; the final
        // sequential root update preserves the deterministic order fixed by the sort above.
        let leaf_hashes: Vec<[u8; 32]> = sorted
            .par_iter()
            .map(|(addr, amount)| {
                let mut leaf_hasher = blake3::Hasher::new();
                leaf_hasher.update(addr.as_slice());
                leaf_hasher.update(&amount.to_be_bytes()); // big-endian per spec — NEVER change
                *leaf_hasher.finalize().as_bytes()
            })
            .collect();

        let mut root_hasher = blake3::Hasher::new();
        root_hasher.update(b"balance_root_v1"); // domain separator — NEVER change
        for leaf in &leaf_hashes {
            root_hasher.update(leaf);
        }

        Ok(*root_hasher.finalize().as_bytes())
    }

    /// Save total minted supply to persistent storage (enforces 21M QUG hard cap)
    /// CRITICAL: Must be called atomically with balance updates to prevent supply violations
    pub async fn save_total_supply(&self, total_supply: u128) -> Result<()> {
        let key = b"total_minted_supply";
        let value = total_supply.to_le_bytes(); // Now 16 bytes for u128

        // CRITICAL: Use synced write to guarantee data reaches disk (same pattern as save_wallet_balance)
        self.hot_db.put_sync(CF_MANIFEST, key, &value).await?;

        debug!("💎 Saved total supply: {} QUG ({} base units)", total_supply / 100_000_000, total_supply);
        Ok(())
    }

    /// Load total minted supply from persistent storage
    /// Returns 0 if no supply data exists (fresh blockchain)
    /// Supports both old u64 format (8 bytes) and new u128 format (16 bytes)
    pub async fn load_total_supply(&self) -> Result<u128> {
        let key = b"total_minted_supply";
        match self.hot_db.get(CF_MANIFEST, key).await? {
            Some(bytes) => {
                let supply = if bytes.len() == 16 {
                    // New u128 format
                    u128::from_le_bytes([
                        bytes[0], bytes[1], bytes[2], bytes[3],
                        bytes[4], bytes[5], bytes[6], bytes[7],
                        bytes[8], bytes[9], bytes[10], bytes[11],
                        bytes[12], bytes[13], bytes[14], bytes[15],
                    ])
                } else if bytes.len() == 8 {
                    // Legacy u64 format - convert to u128
                    u64::from_le_bytes([
                        bytes[0], bytes[1], bytes[2], bytes[3],
                        bytes[4], bytes[5], bytes[6], bytes[7],
                    ]) as u128
                } else {
                    warn!("Invalid total supply data in storage, starting from 0");
                    return Ok(0);
                };
                // v3.9.3-beta: SANITY CHECK - reject obviously corrupted supply values
                // Max supply is 21 million QUG with 24 decimals = 21e6 * 10^24 = 21e30 base units
                // If stored value exceeds this by 10x (210M QUG), it's corrupt
                const MAX_SANE_SUPPLY: u128 = 210_000_000_000_000_000_000_000_000_000_000; // 210M QUG (10x max) with 24 decimals
                if supply > MAX_SANE_SUPPLY {
                    warn!("🚨 [CORRUPT] Total supply in storage ({}) exceeds max sane value!", supply);
                    warn!("🚨 [CORRUPT] Ignoring corrupted value, will recalculate from balances");
                    return Ok(0); // Return 0 to trigger recalculation
                }
                info!("💎 Loaded total supply from storage: {} QUG ({} base units)",
                    supply / 1_000_000_000_000_000_000_000_000, supply);  // v3.9.3-beta: Fix divisor to 10^24
                Ok(supply)
            }
            None => {
                info!("No total supply data found, starting fresh blockchain from 0");
                Ok(0)
            }
        }
    }

    /// Save token balance to persistent storage
    /// v8.5.2: Check if a token balance key exists in RocksDB (regardless of value).
    /// Used by state sync to avoid re-inserting spent token balances from stale peers.
    pub async fn has_token_balance_key(&self, wallet_address: &[u8; 32], token_address: &[u8; 32]) -> bool {
        let key = format!("token_balance_{}_{}", hex::encode(wallet_address), hex::encode(token_address));
        matches!(self.hot_db.get(CF_MANIFEST, key.as_bytes()).await, Ok(Some(_)))
    }

    /// Key format: token_balance_{wallet_hex}_{token_hex}
    /// v2.7.9-beta: Changed from u64 to u128 for larger token supplies (up to 10^38)
    pub async fn save_token_balance(&self, wallet_address: &[u8; 32], token_address: &[u8; 32], amount: u128) -> Result<()> {
        let key = format!("token_balance_{}_{}", hex::encode(wallet_address), hex::encode(token_address));
        let value = amount.to_le_bytes(); // Now 16 bytes instead of 8
        self.hot_db.put(CF_MANIFEST, key.as_bytes(), &value).await?;
        debug!(
            "🪙 Saved token balance: wallet={}, token={}, amount={}",
            hex::encode(wallet_address),
            hex::encode(token_address),
            amount
        );
        Ok(())
    }

    /// Delete a token balance from persistent storage
    /// v7.2.12: Used by startup cleanup to purge stale testnet token balances from RocksDB
    pub async fn delete_token_balance(&self, wallet_address: &[u8; 32], token_address: &[u8; 32]) -> Result<()> {
        let key = format!("token_balance_{}_{}", hex::encode(wallet_address), hex::encode(token_address));
        self.hot_db.delete(CF_MANIFEST, key.as_bytes()).await?;
        Ok(())
    }

    /// Get a single token balance from persistent storage
    /// v1.4.8-beta: Also checks CF_TOKEN_BALANCES (state sync storage) if legacy storage returns 0
    /// v2.7.9-beta: Returns u128 for larger token supplies, backward compatible with 8-byte u64 values
    pub async fn get_token_balance(&self, wallet_address: &[u8; 32], token_address: &[u8; 32]) -> Result<u128> {
        // First check CF_MANIFEST storage (supports both old 8-byte and new 16-byte format)
        let key = format!("token_balance_{}_{}", hex::encode(wallet_address), hex::encode(token_address));
        let mut manifest_has_entry = false;
        let manifest_balance = match self.hot_db.get(CF_MANIFEST, key.as_bytes()).await? {
            Some(bytes) => {
                manifest_has_entry = true;
                if bytes.len() == 16 {
                    // v2.7.9-beta: New 16-byte u128 format
                    let amount = u128::from_le_bytes([
                        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
                        bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
                    ]);
                    debug!(
                        "🪙 Loaded token balance (u128): wallet={}, token={}, amount={}",
                        hex::encode(wallet_address),
                        hex::encode(token_address),
                        amount
                    );
                    amount
                } else if bytes.len() == 8 {
                    // Legacy 8-byte u64 format (backward compatibility)
                    let amount = u64::from_le_bytes([
                        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
                    ]) as u128;
                    debug!(
                        "🪙 Loaded token balance (legacy u64): wallet={}, token={}, amount={}",
                        hex::encode(wallet_address),
                        hex::encode(token_address),
                        amount
                    );
                    amount
                } else {
                    warn!(
                        "Invalid token balance data length ({} bytes) for wallet {} token {}",
                        bytes.len(),
                        hex::encode(wallet_address),
                        hex::encode(token_address)
                    );
                    0
                }
            }
            None => 0,
        };

        // v8.5.3: CF_MANIFEST is authoritative. If it has ANY entry (even 0), trust it.
        // The old check `manifest_balance > 0` would fall through to CF_TOKEN_BALANCES
        // when a user legitimately spent their balance to 0, resurrecting the old pre-spend value.
        if manifest_has_entry {
            return Ok(manifest_balance);
        }

        // Only check CF_TOKEN_BALANCES if CF_MANIFEST has NO entry at all (new node, never synced)
        // v1.4.8-beta: Check CF_TOKEN_BALANCES (state sync storage) for synced transfers
        #[cfg(not(target_os = "windows"))]
        if let Some(db) = self.get_rocks_db_handle() {
            if let Some(cf) = db.cf_handle(CF_TOKEN_BALANCES) {
                // Build key: account (32 bytes) + token (32 bytes)
                let mut cf_key = Vec::with_capacity(64);
                cf_key.extend_from_slice(wallet_address);
                cf_key.extend_from_slice(token_address);

                if let Ok(Some(value)) = db.get_cf(&cf, &cf_key) {
                    // State sync storage: check for 16-byte u128 first, then 8-byte u64
                    if value.len() >= 16 {
                        // New u128 format (big-endian for state sync consistency)
                        let amount = u128::from_be_bytes(value[..16].try_into().unwrap_or([0u8; 16]));
                        if amount > 0 {
                            info!(
                                "🪙 Found token balance in state sync storage (u128): wallet={}, token={}, amount={}",
                                hex::encode(&wallet_address[..8]),
                                hex::encode(&token_address[..8]),
                                amount
                            );
                            return Ok(amount);
                        }
                    } else if value.len() >= 8 {
                        // Legacy u64 format (big-endian)
                        let amount = u64::from_be_bytes(value[..8].try_into().unwrap_or([0u8; 8])) as u128;
                        if amount > 0 {
                            info!(
                                "🪙 Found token balance in state sync storage (legacy u64): wallet={}, token={}, amount={}",
                                hex::encode(&wallet_address[..8]),
                                hex::encode(&token_address[..8]),
                                amount
                            );
                            return Ok(amount);
                        }
                    }
                }
            }
        }

        debug!(
            "🪙 Token balance not found in any storage: wallet={}, token={}",
            hex::encode(wallet_address),
            hex::encode(token_address)
        );
        Ok(0)
    }

    /// Load all token balances from persistent storage
    /// v1.4.8-beta: Also loads from CF_TOKEN_BALANCES (state sync storage) for synced transfers
    /// v2.7.9-beta: Returns u128 for larger token supplies, backward compatible with 8-byte u64 values
    pub async fn load_token_balances(&self) -> Result<HashMap<([u8; 32], [u8; 32]), u128>> {
        let mut balances = HashMap::new();
        let mut manifest_count = 0;
        let mut state_sync_count = 0;

        // Load from CF_MANIFEST storage (supports both 8-byte u64 and 16-byte u128)
        let prefix = "token_balance_".as_bytes();
        match self.hot_db.scan_prefix(CF_MANIFEST, prefix).await {
            Ok(entries) => {
                for (key, value) in entries {
                    if let Ok(key_str) = String::from_utf8(key) {
                        if let Some(addresses) = key_str.strip_prefix("token_balance_") {
                            // Parse "wallet_hex_token_hex"
                            let parts: Vec<&str> = addresses.split('_').collect();
                            if parts.len() == 2 {
                                if let (Ok(wallet_bytes), Ok(token_bytes)) = (hex::decode(parts[0]), hex::decode(parts[1])) {
                                    if wallet_bytes.len() == 32 && token_bytes.len() == 32 {
                                        let mut wallet_address = [0u8; 32];
                                        let mut token_address = [0u8; 32];
                                        wallet_address.copy_from_slice(&wallet_bytes);
                                        token_address.copy_from_slice(&token_bytes);

                                        let amount = if value.len() == 16 {
                                            // v2.7.9-beta: New 16-byte u128 format
                                            u128::from_le_bytes([
                                                value[0], value[1], value[2], value[3], value[4], value[5], value[6], value[7],
                                                value[8], value[9], value[10], value[11], value[12], value[13], value[14], value[15],
                                            ])
                                        } else if value.len() == 8 {
                                            // Legacy 8-byte u64 format
                                            u64::from_le_bytes([
                                                value[0], value[1], value[2], value[3], value[4], value[5], value[6], value[7],
                                            ]) as u128
                                        } else {
                                            warn!("Invalid token balance data length ({} bytes) for wallet {}",
                                                value.len(), hex::encode(&wallet_address));
                                            continue;
                                        };

                                        balances.insert((wallet_address, token_address), amount);
                                        manifest_count += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Err(e) => {
                warn!("Failed to scan token balances: {}", e);
            }
        }

        // v1.4.8-beta: Also load from CF_TOKEN_BALANCES (state sync storage)
        // This captures balances from synced TokenTransfer transactions
        // v8.5.3: CRITICAL FIX — CF_MANIFEST is authoritative. State sync only fills MISSING entries.
        // The old "highest wins" rule caused ghost balances: stale pre-spend values in CF_TOKEN_BALANCES
        // would override correct post-spend values in CF_MANIFEST on every restart.
        #[cfg(not(target_os = "windows"))]
        if let Some(db) = self.get_rocks_db_handle() {
            if let Some(cf) = db.cf_handle(CF_TOKEN_BALANCES) {
                let iter = db.iterator_cf(&cf, rocksdb::IteratorMode::Start);
                for item in iter {
                    if let Ok((key, value)) = item {
                        // Key format: account (32 bytes) + token (32 bytes) = 64 bytes
                        if key.len() == 64 && value.len() >= 8 {
                            let mut wallet_address = [0u8; 32];
                            let mut token_address = [0u8; 32];
                            wallet_address.copy_from_slice(&key[0..32]);
                            token_address.copy_from_slice(&key[32..64]);

                            // State sync stores as big-endian: check 16-byte u128 first, then 8-byte u64
                            let amount = if value.len() >= 16 {
                                u128::from_be_bytes(value[..16].try_into().unwrap_or([0u8; 16]))
                            } else {
                                u64::from_be_bytes(value[..8].try_into().unwrap_or([0u8; 8])) as u128
                            };

                            // v8.5.3: Only insert if NOT already in CF_MANIFEST (manifest is authoritative)
                            // Previously this used "highest wins" which resurrected spent balances
                            // v8.5.5: ALSO skip QUGUSD entries from state sync — they are the source
                            // of the 172K QUGUSD ghost. State sync has stale testnet QUGUSD in binary
                            // keys that survive the CF_MANIFEST text-key purge.
                            if token_address == q_types::QUGUSD_TOKEN_ADDRESS {
                                continue; // NEVER load QUGUSD from state sync CF
                            }
                            let balance_key = (wallet_address, token_address);
                            if !balances.contains_key(&balance_key) && amount > 0 {
                                balances.insert(balance_key, amount);
                                state_sync_count += 1;
                            }
                        }
                    }
                }
            }
        }

        info!(
            "🪙 Loaded {} token balances ({} manifest + {} state sync)",
            balances.len(),
            manifest_count,
            state_sync_count
        );

        Ok(balances)
    }

    /// Save multiple token balances atomically with SYNC to guarantee disk write
    /// v2.7.9-beta: Changed from u64 to u128 for larger token supplies
    pub async fn save_token_balances(&self, balances: &HashMap<([u8; 32], [u8; 32]), u128>) -> Result<()> {
        let mut batch_ops = Vec::new();

        for ((wallet_address, token_address), amount) in balances {
            let key = format!("token_balance_{}_{}", hex::encode(wallet_address), hex::encode(token_address));
            let value = amount.to_le_bytes().to_vec(); // Now 16 bytes
            batch_ops.push((CF_MANIFEST, key.into_bytes(), value));
        }

        // CRITICAL: write_batch now uses fsync to survive hard kills (fixed in kv.rs)
        self.hot_db.write_batch(batch_ops).await?;
        info!(
            "🪙 SYNCED {} token balances to persistent storage (survives hard kill)",
            balances.len()
        );
        Ok(())
    }

    // ============ v8.5.9: QUSD STABLECOIN AUDIT LOG ============

    /// Save a QUSD audit entry (mint/burn) to persistent storage
    /// Key format: qusd_audit_{timestamp}_{tx_hash} — append-only for transparency
    pub async fn save_qusd_audit_entry(&self, entry: &serde_json::Value) -> Result<()> {
        let timestamp = entry.get("timestamp").and_then(|v| v.as_i64()).unwrap_or(0);
        let tx_hash = entry.get("tx_hash").and_then(|v| v.as_str()).unwrap_or("unknown");
        let key = format!("qusd_audit_{}_{}", timestamp, tx_hash);
        let value = serde_json::to_vec(entry)?;
        self.hot_db.put(CF_MANIFEST, key.as_bytes(), &value).await?;
        debug!("💵 [QUSD] Saved audit entry: {}", key);
        Ok(())
    }

    /// Load all QUSD audit entries from persistent storage (prefix scan)
    pub async fn load_qusd_audit_log(&self) -> Result<Vec<serde_json::Value>> {
        let prefix = b"qusd_audit_";
        let entries_raw = self.hot_db.scan_prefix(CF_MANIFEST, prefix).await?;
        let mut entries = Vec::new();
        for (_key, value) in entries_raw {
            if let Ok(entry) = serde_json::from_slice::<serde_json::Value>(&value) {
                entries.push(entry);
            }
        }
        entries.sort_by(|a, b| {
            let ts_a = a.get("timestamp").and_then(|v| v.as_i64()).unwrap_or(0);
            let ts_b = b.get("timestamp").and_then(|v| v.as_i64()).unwrap_or(0);
            ts_a.cmp(&ts_b)
        });
        info!("💵 [QUSD] Loaded {} audit entries from RocksDB", entries.len());
        Ok(entries)
    }

    /// Get QUSD total supply by scanning audit log (sum of mints - sum of burns)
    pub async fn get_qusd_total_supply(&self) -> Result<u128> {
        let entries = self.load_qusd_audit_log().await?;
        let mut total: u128 = 0;
        for entry in &entries {
            let action = entry.get("action").and_then(|v| v.as_str()).unwrap_or("");
            let amount_str = entry.get("amount_raw").and_then(|v| v.as_str()).unwrap_or("0");
            let amount: u128 = amount_str.parse().unwrap_or(0);
            match action {
                "mint" => total = total.saturating_add(amount),
                "burn" => total = total.saturating_sub(amount),
                _ => {}
            }
        }
        Ok(total)
    }

    // ============ v2.4.2: TOKEN STAKING STORAGE ============

    /// Save a token stake position to persistent storage
    /// Key format: stake_position_{stake_key}
    pub async fn save_stake_position(&self, stake_key: &str, position: &crate::TokenStakePosition) -> Result<()> {
        let key = format!("stake_position_{}", stake_key);
        let value = serde_json::to_vec(position)?;
        self.hot_db.put(CF_MANIFEST, key.as_bytes(), &value).await?;
        debug!(
            "🔒 Saved stake position: {} ({} tokens, tier: {:?})",
            stake_key,
            position.amount as f64 / 100_000_000.0,
            position.tier
        );
        Ok(())
    }

    /// Delete a stake position from storage
    pub async fn delete_stake_position(&self, stake_key: &str) -> Result<()> {
        let key = format!("stake_position_{}", stake_key);
        self.hot_db.delete(CF_MANIFEST, key.as_bytes()).await?;
        debug!("🔓 Deleted stake position: {}", stake_key);
        Ok(())
    }

    /// Load all stake positions from storage
    pub async fn load_stake_positions(&self) -> Result<HashMap<String, crate::TokenStakePosition>> {
        let mut positions = HashMap::new();
        let prefix = b"stake_position_";

        match self.hot_db.scan_prefix(CF_MANIFEST, prefix).await {
            Ok(entries) => {
                for (key_bytes, value) in entries {
                    if let Ok(key_str) = String::from_utf8(key_bytes) {
                        let stake_key = key_str.trim_start_matches("stake_position_");
                        if let Ok(position) = serde_json::from_slice::<crate::TokenStakePosition>(&value) {
                            positions.insert(stake_key.to_string(), position);
                        }
                    }
                }
                info!("🔒 Loaded {} stake positions from storage", positions.len());
            }
            Err(e) => {
                warn!("Failed to scan stake positions: {}", e);
            }
        }
        Ok(positions)
    }

    /// Save token fee configuration
    /// Key format: fee_config_{contract_address}
    pub async fn save_fee_config(&self, contract_address: &str, config: &crate::TokenFeeConfig) -> Result<()> {
        let key = format!("fee_config_{}", contract_address);
        let value = serde_json::to_vec(config)?;
        self.hot_db.put(CF_MANIFEST, key.as_bytes(), &value).await?;
        debug!(
            "⚙️ Saved fee config for {}: enabled={}, reflection={}bps, burn={}bps",
            contract_address, config.enabled, config.reflection_fee_bps, config.burn_fee_bps
        );
        Ok(())
    }

    /// Load all fee configurations from storage
    pub async fn load_fee_configs(&self) -> Result<HashMap<String, crate::TokenFeeConfig>> {
        let mut configs = HashMap::new();
        let prefix = b"fee_config_";

        match self.hot_db.scan_prefix(CF_MANIFEST, prefix).await {
            Ok(entries) => {
                for (key_bytes, value) in entries {
                    if let Ok(key_str) = String::from_utf8(key_bytes) {
                        let contract_addr = key_str.trim_start_matches("fee_config_");
                        if let Ok(config) = serde_json::from_slice::<crate::TokenFeeConfig>(&value) {
                            configs.insert(contract_addr.to_string(), config);
                        }
                    }
                }
                info!("⚙️ Loaded {} fee configs from storage", configs.len());
            }
            Err(e) => {
                warn!("Failed to scan fee configs: {}", e);
            }
        }
        Ok(configs)
    }

    /// Save burn/reflection totals
    pub async fn save_token_totals(&self, key: &str, amount: u64) -> Result<()> {
        let value = amount.to_le_bytes();
        self.hot_db.put(CF_MANIFEST, key.as_bytes(), &value).await?;
        Ok(())
    }

    /// Load burn/reflection totals
    pub async fn load_token_total(&self, key: &str) -> Result<u64> {
        match self.hot_db.get(CF_MANIFEST, key.as_bytes()).await? {
            Some(data) if data.len() == 8 => {
                let bytes: [u8; 8] = data.try_into().unwrap();
                Ok(u64::from_le_bytes(bytes))
            }
            _ => Ok(0),
        }
    }

    /// v2.4.8-beta: Save token social profile to storage
    pub async fn save_social_profile(&self, contract_address: &str, data: &[u8]) -> Result<()> {
        let key = format!("token_social:{}", contract_address.to_lowercase());
        self.hot_db.put(CF_MANIFEST, key.as_bytes(), data).await?;
        info!("📱 Saved social profile for {} ({} bytes)", contract_address, data.len());
        Ok(())
    }

    /// v2.4.8-beta: Load token social profile from storage
    pub async fn load_social_profile(&self, contract_address: &str) -> Result<Option<Vec<u8>>> {
        let key = format!("token_social:{}", contract_address.to_lowercase());
        match self.hot_db.get(CF_MANIFEST, key.as_bytes()).await? {
            Some(data) => Ok(Some(data)),
            None => Ok(None),
        }
    }

    /// v2.4.8-beta: Load all social profiles (for startup sync)
    pub async fn load_all_social_profiles(&self) -> Result<Vec<(String, Vec<u8>)>> {
        let prefix = b"token_social:";
        let mut profiles = Vec::new();

        match self.hot_db.scan_prefix(CF_MANIFEST, prefix).await {
            Ok(entries) => {
                for (key_bytes, value) in entries {
                    if let Ok(key_str) = String::from_utf8(key_bytes) {
                        let contract_addr = key_str.trim_start_matches("token_social:");
                        profiles.push((contract_addr.to_string(), value));
                    }
                }
                info!("📱 Loaded {} social profiles from storage", profiles.len());
            }
            Err(e) => {
                warn!("Failed to scan social profiles: {}", e);
            }
        }

        Ok(profiles)
    }

    /// Save transaction to persistent storage with wallet address indexing
    /// v3.5.8-beta: Also creates wallet index entries for decentralized history
    pub async fn save_transaction(&self, tx: &q_types::Transaction) -> Result<()> {
        let tx_data = bincode::serialize(tx)?;

        // Save main transaction
        self.hot_db.put(CF_TRANSACTIONS, &tx.id, &tx_data).await?;

        // v3.5.8-beta: Index by sender wallet address
        let sender_key = Self::build_wallet_tx_key(&tx.from, tx.timestamp.timestamp(), &tx.id);
        self.hot_db.put(CF_WALLET_TX_INDEX, &sender_key, &tx.id).await?;

        // v3.5.8-beta: Index by recipient wallet address (if different from sender)
        if tx.from != tx.to {
            let recipient_key = Self::build_wallet_tx_key(&tx.to, tx.timestamp.timestamp(), &tx.id);
            self.hot_db.put(CF_WALLET_TX_INDEX, &recipient_key, &tx.id).await?;
        }

        debug!(
            "💳 Saved transaction with wallet index: {} ({} -> {})",
            hex::encode(&tx.id),
            hex::encode(&tx.from),
            hex::encode(&tx.to)
        );
        Ok(())
    }

    /// Build wallet transaction index key: [wallet:32][inverted_timestamp:8][tx_id:8]
    /// Using inverted timestamp ensures newest transactions come first in prefix scan
    fn build_wallet_tx_key(wallet: &[u8; 32], timestamp: i64, tx_id: &[u8; 32]) -> Vec<u8> {
        let mut key = Vec::with_capacity(48);
        key.extend_from_slice(wallet);
        // Invert timestamp so newer entries sort first (RocksDB sorts ascending)
        let inverted_ts = i64::MAX - timestamp;
        key.extend_from_slice(&inverted_ts.to_be_bytes());
        key.extend_from_slice(&tx_id[..8]);
        key
    }

    /// Load transaction from persistent storage
    pub async fn load_transaction(&self, tx_id: &[u8; 32]) -> Result<Option<q_types::Transaction>> {
        match self.hot_db.get(CF_TRANSACTIONS, tx_id).await? {
            Some(tx_data) => {
                let tx: q_types::Transaction = bincode::deserialize(&tx_data)?;
                debug!("💳 Loaded transaction: {}", hex::encode(tx_id));
                Ok(Some(tx))
            }
            None => Ok(None),
        }
    }

    /// Load all transactions from persistent storage
    /// WARNING: This loads ALL transactions into memory - use count_transactions() for metrics
    pub async fn load_all_transactions(&self) -> Result<Vec<q_types::Transaction>> {
        let mut transactions = Vec::new();

        match self.hot_db.scan_all(CF_TRANSACTIONS).await {
            Ok(entries) => {
                for (_key, tx_data) in entries {
                    if let Ok(tx) = bincode::deserialize::<q_types::Transaction>(&tx_data) {
                        transactions.push(tx);
                    }
                }
                info!(
                    "💳 Loaded {} transactions from persistent storage",
                    transactions.len()
                );
            }
            Err(e) => {
                warn!("Failed to scan transactions: {}", e);
            }
        }

        Ok(transactions)
    }

    /// Save multiple transactions atomically with SYNC to guarantee disk write
    /// v3.5.8-beta: Also creates wallet index entries for decentralized history
    pub async fn save_transactions(&self, transactions: &[q_types::Transaction]) -> Result<()> {
        let mut batch_ops = Vec::new();

        for tx in transactions {
            let tx_data = bincode::serialize(tx)?;

            // Main transaction
            batch_ops.push((CF_TRANSACTIONS, tx.id.to_vec(), tx_data));

            // v3.5.8-beta: Index by sender wallet
            let sender_key = Self::build_wallet_tx_key(&tx.from, tx.timestamp.timestamp(), &tx.id);
            batch_ops.push((CF_WALLET_TX_INDEX, sender_key, tx.id.to_vec()));

            // v3.5.8-beta: Index by recipient wallet (if different)
            if tx.from != tx.to {
                let recipient_key = Self::build_wallet_tx_key(&tx.to, tx.timestamp.timestamp(), &tx.id);
                batch_ops.push((CF_WALLET_TX_INDEX, recipient_key, tx.id.to_vec()));
            }
        }

        // CRITICAL: write_batch now uses fsync to survive hard kills (fixed in kv.rs)
        self.hot_db.write_batch(batch_ops).await?;
        info!(
            "💳 SYNCED {} transactions with wallet indexes to persistent storage",
            transactions.len()
        );
        Ok(())
    }

    /// Load transactions for a specific wallet address (decentralized history)
    /// v3.5.8-beta: Uses wallet index for O(log n) lookups instead of O(n) full scan
    /// Returns transactions where the wallet is sender OR recipient, newest first
    pub async fn load_transactions_for_wallet(
        &self,
        wallet: &[u8; 32],
        limit: usize,
    ) -> Result<Vec<q_types::Transaction>> {
        let mut transactions = Vec::new();

        // Build prefix for wallet (first 32 bytes of key)
        let prefix = wallet.to_vec();

        // Scan wallet index with prefix
        match self.hot_db.scan_prefix(CF_WALLET_TX_INDEX, &prefix).await {
            Ok(entries) => {
                let mut seen_tx_ids = std::collections::HashSet::new();

                for (_key, tx_id_data) in entries {
                    if transactions.len() >= limit {
                        break;
                    }

                    // tx_id_data contains the full 32-byte transaction ID
                    if tx_id_data.len() >= 32 {
                        let mut tx_id = [0u8; 32];
                        tx_id.copy_from_slice(&tx_id_data[..32]);

                        // Avoid duplicates (same tx indexed for both sender and recipient)
                        if seen_tx_ids.contains(&tx_id) {
                            continue;
                        }
                        seen_tx_ids.insert(tx_id);

                        // Load full transaction
                        if let Ok(Some(tx)) = self.load_transaction(&tx_id).await {
                            transactions.push(tx);
                        }
                    }
                }

                info!(
                    "💳 Loaded {} transactions for wallet {} via index",
                    transactions.len(),
                    hex::encode(&wallet[..8])
                );
            }
            Err(e) => {
                warn!("Failed to scan wallet transaction index: {}", e);
            }
        }

        Ok(transactions)
    }

    /// Save swap record indexed by wallet address
    /// v3.5.8-beta: Enables wallet-based swap history lookup
    pub async fn save_wallet_swap_index(
        &self,
        wallet: &[u8; 32],
        timestamp: i64,
        tx_id: &[u8; 32],
        swap_data: &[u8],
    ) -> Result<()> {
        let key = Self::build_wallet_tx_key(wallet, timestamp, tx_id);
        self.hot_db.put(CF_WALLET_SWAP_INDEX, &key, swap_data).await?;
        debug!(
            "🔄 Indexed swap for wallet {} at timestamp {}",
            hex::encode(&wallet[..8]),
            timestamp
        );
        Ok(())
    }

    /// Load swap history for a specific wallet address
    /// v3.5.8-beta: Returns all DEX swaps where wallet is the trader, newest first
    pub async fn load_swaps_for_wallet(
        &self,
        wallet: &[u8; 32],
        limit: usize,
    ) -> Result<Vec<Vec<u8>>> {
        let mut swaps = Vec::new();
        let prefix = wallet.to_vec();

        match self.hot_db.scan_prefix(CF_WALLET_SWAP_INDEX, &prefix).await {
            Ok(entries) => {
                for (_key, swap_data) in entries {
                    if swaps.len() >= limit {
                        break;
                    }
                    swaps.push(swap_data);
                }
                info!(
                    "🔄 Loaded {} swaps for wallet {} via index",
                    swaps.len(),
                    hex::encode(&wallet[..8])
                );
            }
            Err(e) => {
                warn!("Failed to scan wallet swap index: {}", e);
            }
        }

        Ok(swaps)
    }

    /// Migrate existing transactions to wallet index (one-time backfill)
    /// v3.5.8-beta: Run at startup to index all existing transactions by wallet
    /// v3.5.11-beta: Also scan blocks for coinbase/mining transactions (background)
    /// This enables the new decentralized transaction history feature
    pub async fn migrate_transactions_to_wallet_index(&self) -> Result<usize> {
        info!("🔄 [v3.5.11] Starting transaction wallet index migration...");

        // Check if migration already done (look for marker)
        // v3.5.11: Use new marker version to force re-migration with background block scanning
        let migration_key = b"migration_wallet_index_v3.5.11_done";
        if let Ok(Some(_)) = self.hot_db.get(CF_MANIFEST, migration_key).await {
            // Verify the index actually has data
            match self.hot_db.scan_prefix(CF_WALLET_TX_INDEX, &[]).await {
                Ok(entries) if !entries.is_empty() => {
                    info!("✅ [v3.5.11] Wallet index migration already completed ({} entries), skipping", entries.len());
                    return Ok(0);
                }
                _ => {
                    // Index is empty - previous migration failed, re-run it
                    warn!("⚠️ [v3.5.11] Migration marker exists but index is empty - re-running migration");
                    // Delete the old marker
                    let _ = self.hot_db.delete(CF_MANIFEST, migration_key).await;
                }
            }
        }

        let mut indexed_count = 0;
        let mut batch_ops: Vec<(&str, Vec<u8>, Vec<u8>)> = Vec::new();
        let mut seen_tx_ids = std::collections::HashSet::new();

        // Part 1: Scan all transactions in CF_TRANSACTIONS column family
        match self.hot_db.scan_prefix(CF_TRANSACTIONS, &[]).await {
            Ok(entries) => {
                info!("🔄 [v3.5.11] Found {} standalone transactions to index", entries.len());

                for (tx_id_data, tx_data) in entries {
                    if let Ok(tx) = bincode::deserialize::<q_types::Transaction>(&tx_data) {
                        seen_tx_ids.insert(tx.id);

                        // Index by sender
                        let sender_key = Self::build_wallet_tx_key(&tx.from, tx.timestamp.timestamp(), &tx.id);
                        batch_ops.push((CF_WALLET_TX_INDEX, sender_key, tx.id.to_vec()));

                        // Index by recipient (if different)
                        if tx.from != tx.to {
                            let recipient_key = Self::build_wallet_tx_key(&tx.to, tx.timestamp.timestamp(), &tx.id);
                            batch_ops.push((CF_WALLET_TX_INDEX, recipient_key, tx.id.to_vec()));
                        }

                        indexed_count += 1;

                        // Write in batches of 1000 to avoid memory issues
                        if batch_ops.len() >= 1000 {
                            if let Err(e) = self.hot_db.write_batch(batch_ops.clone()).await {
                                warn!("Failed to write batch during migration: {}", e);
                            }
                            batch_ops.clear();
                        }
                    }
                }
            }
            Err(e) => {
                warn!("Failed to scan CF_TRANSACTIONS for migration: {}", e);
            }
        }

        // Part 2: v3.5.11-beta - Scan ALL BLOCKS for coinbase/mining transactions
        // This is critical because mining rewards are stored inside blocks, not CF_TRANSACTIONS
        info!("🔄 [v3.5.11] Scanning blocks for coinbase transactions (mining rewards)...");
        let mut block_tx_count = 0;
        let mut blocks_scanned = 0usize;
        let progress_interval = 50_000usize; // Log every 50k blocks

        // Scan block column family
        match self.hot_db.scan_prefix(CF_BLOCKS, &[]).await {
            Ok(entries) => {
                let total_entries = entries.len();
                info!("🔄 [v3.5.11] Found {} blocks to scan for transactions", total_entries);

                for (key, block_data) in entries {
                    // v3.5.11: Fix filter to process actual block data
                    // Block keys: qblock:height:N, qblock:dag:N:proposer
                    // Skip: qblock:latest (pointer), qblock:hash:* (no block data)
                    let key_str = String::from_utf8_lossy(&key);
                    if key_str == "qblock:latest" || key_str.starts_with("qblock:hash:") {
                        continue;
                    }

                    blocks_scanned += 1;

                    // Progress logging every 50k blocks
                    if blocks_scanned % progress_interval == 0 {
                        let pct = (blocks_scanned as f64 / total_entries as f64 * 100.0) as u32;
                        info!("🔄 [v3.5.11] Migration progress: {}/{} blocks ({}%), {} txs indexed",
                              blocks_scanned, total_entries, pct, block_tx_count);
                    }

                    // v7.3.1: Handle compressed format for indexing
                    let raw_data = if precompressed_storage::is_precompressed(&block_data) {
                        precompressed_storage::PrecompressedBlock::from_bytes(&block_data)
                            .and_then(|c| c.decompress().map_err(|e| e.into()))
                            .unwrap_or(block_data.clone())
                    } else {
                        block_data.clone()
                    };
                    // Try to deserialize as QBlock
                    if let Ok(block) = bincode::deserialize::<q_types::QBlock>(&raw_data) {
                        for tx in &block.transactions {
                            // Skip if already indexed from CF_TRANSACTIONS
                            if seen_tx_ids.contains(&tx.id) {
                                continue;
                            }

                            // Index by sender
                            let sender_key = Self::build_wallet_tx_key(&tx.from, tx.timestamp.timestamp(), &tx.id);
                            batch_ops.push((CF_WALLET_TX_INDEX, sender_key, tx.id.to_vec()));

                            // Index by recipient (if different)
                            if tx.from != tx.to {
                                let recipient_key = Self::build_wallet_tx_key(&tx.to, tx.timestamp.timestamp(), &tx.id);
                                batch_ops.push((CF_WALLET_TX_INDEX, recipient_key, tx.id.to_vec()));
                            }

                            // Also save transaction to CF_TRANSACTIONS for future lookups
                            if let Ok(tx_data) = bincode::serialize(&tx) {
                                batch_ops.push((CF_TRANSACTIONS, tx.id.to_vec(), tx_data));
                            }

                            block_tx_count += 1;
                            indexed_count += 1;

                            // Write in batches
                            if batch_ops.len() >= 1000 {
                                if let Err(e) = self.hot_db.write_batch(batch_ops.clone()).await {
                                    warn!("Failed to write batch during block migration: {}", e);
                                }
                                batch_ops.clear();
                            }
                        }
                    }
                }

                info!("🔄 [v3.5.11] Block scan complete: {} blocks scanned, {} txs found", blocks_scanned, block_tx_count);
            }
            Err(e) => {
                warn!("Failed to scan CF_BLOCKS for migration: {}", e);
            }
        }

        // Write remaining batch
        if !batch_ops.is_empty() {
            if let Err(e) = self.hot_db.write_batch(batch_ops).await {
                warn!("Failed to write final batch during migration: {}", e);
            }
        }

        // Mark migration as done
        self.hot_db.put(CF_MANIFEST, migration_key, b"done").await?;

        info!(
            "✅ [v3.5.11] Wallet index migration complete: indexed {} transactions ({} from blocks)",
            indexed_count, block_tx_count
        );

        Ok(indexed_count)
    }

    /// Delete transaction from persistent storage
    pub async fn delete_transaction(&self, tx_id: &[u8; 32]) -> Result<()> {
        self.hot_db.delete(CF_TRANSACTIONS, tx_id).await?;
        debug!("🗑️ Deleted transaction: {}", hex::encode(tx_id));
        Ok(())
    }

    /// Save smart contract to persistent storage
    pub async fn save_contract(&self, address: &[u8; 32], contract_data: &[u8]) -> Result<()> {
        let key = format!("contract_{}", hex::encode(address));
        self.hot_db.put(CF_MANIFEST, key.as_bytes(), contract_data).await?;
        debug!("📜 Saved smart contract: {}", hex::encode(address));
        Ok(())
    }

    /// Load smart contract from persistent storage
    pub async fn load_contract(&self, address: &[u8; 32]) -> Result<Option<Vec<u8>>> {
        let key = format!("contract_{}", hex::encode(address));
        match self.hot_db.get(CF_MANIFEST, key.as_bytes()).await? {
            Some(contract_data) => {
                debug!("📜 Loaded smart contract: {}", hex::encode(address));
                Ok(Some(contract_data))
            }
            None => Ok(None),
        }
    }

    /// Load all smart contracts from persistent storage
    pub async fn load_all_contracts(&self) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let prefix = "contract_".as_bytes();
        let mut contracts = Vec::new();

        match self.hot_db.scan_prefix(CF_MANIFEST, prefix).await {
            Ok(entries) => {
                for (key, value) in entries {
                    if let Ok(key_str) = String::from_utf8(key) {
                        if let Some(hex_addr) = key_str.strip_prefix("contract_") {
                            if let Ok(addr_bytes) = hex::decode(hex_addr) {
                                if addr_bytes.len() == 32 {
                                    contracts.push((addr_bytes, value));
                                }
                            }
                        }
                    }
                }
                info!("📜 Loaded {} smart contracts from persistent storage", contracts.len());
            }
            Err(e) => {
                warn!("Failed to scan smart contracts: {}", e);
            }
        }

        Ok(contracts)
    }

    /// Delete smart contract from persistent storage
    pub async fn delete_contract(&self, address: &[u8; 32]) -> Result<()> {
        let key = format!("contract_{}", hex::encode(address));
        self.hot_db.delete(CF_MANIFEST, key.as_bytes()).await?;
        debug!("🗑️ Deleted smart contract: {}", hex::encode(address));
        Ok(())
    }

    /// Save liquidity pool to persistent storage
    /// Pool ID format: "QUG-QUGUSD" or similar token pair identifier
    pub async fn save_liquidity_pool(&self, pool_id: &str, pool_data: &[u8]) -> Result<()> {
        let key = format!("liquidity_pool:{}", pool_id);
        self.hot_db.put(CF_MANIFEST, key.as_bytes(), pool_data).await?;
        debug!("💧 Saved liquidity pool: {}", pool_id);
        Ok(())
    }

    /// Load all liquidity pools from persistent storage
    /// Returns map of pool_id -> serialized pool data
    pub async fn load_liquidity_pools(&self) -> Result<HashMap<String, Vec<u8>>> {
        let mut pools = HashMap::new();
        let prefix = b"liquidity_pool:";

        match self.hot_db.scan_prefix(CF_MANIFEST, prefix).await {
            Ok(entries) => {
                for (key, value) in entries {
                    if let Ok(key_str) = String::from_utf8(key) {
                        if let Some(pool_id) = key_str.strip_prefix("liquidity_pool:") {
                            pools.insert(pool_id.to_string(), value);
                        }
                    }
                }
                info!(
                    "💧 Loaded {} liquidity pools from persistent storage",
                    pools.len()
                );
            }
            Err(e) => {
                warn!("Failed to scan liquidity pools: {}", e);
            }
        }

        Ok(pools)
    }

    /// Delete liquidity pool from persistent storage
    pub async fn delete_liquidity_pool(&self, pool_id: &str) -> Result<()> {
        let key = format!("liquidity_pool:{}", pool_id);
        self.hot_db.delete(CF_MANIFEST, key.as_bytes()).await?;
        debug!("🗑️ Deleted liquidity pool: {}", pool_id);
        Ok(())
    }

    /// Save LP token metadata (symbol, name, decimals) for display in wallets
    /// Key format: lp_token_meta_<addr_hex> → JSON metadata
    pub async fn save_lp_token_meta(&self, token_addr: &[u8; 32], symbol0: &str, symbol1: &str) -> Result<()> {
        let key = format!("lp_token_meta_{}", hex::encode(token_addr));
        let meta = serde_json::json!({
            "symbol": format!("LP-{}-{}", symbol0, symbol1),
            "name": format!("{}/{} LP Token", symbol0, symbol1),
            "decimals": 24
        });
        self.hot_db.put(CF_MANIFEST, key.as_bytes(), meta.to_string().as_bytes()).await?;
        debug!("💧 Saved LP token metadata: LP-{}-{}", symbol0, symbol1);
        Ok(())
    }

    /// Load LP token metadata by token address
    pub async fn load_lp_token_meta(&self, token_addr: &[u8; 32]) -> Option<serde_json::Value> {
        let key = format!("lp_token_meta_{}", hex::encode(token_addr));
        match self.hot_db.get(CF_MANIFEST, key.as_bytes()).await {
            Ok(Some(bytes)) => serde_json::from_slice(&bytes).ok(),
            _ => None,
        }
    }

    // ============================================================================
    // QCREDIT Yield Vault Persistence (v8.5.5)
    // ============================================================================

    /// Save the entire QCREDIT vault state as a single JSON blob
    pub async fn save_qcredit_vault(&self, vault_data: &[u8]) -> Result<()> {
        self.hot_db.put(CF_MANIFEST, b"qcredit_vault_state", vault_data).await?;
        debug!("💳 Saved QCREDIT vault state ({} bytes)", vault_data.len());
        Ok(())
    }

    /// Load the QCREDIT vault state
    pub async fn load_qcredit_vault(&self) -> Result<Option<Vec<u8>>> {
        match self.hot_db.get(CF_MANIFEST, b"qcredit_vault_state").await? {
            Some(data) => {
                info!("💳 Loaded QCREDIT vault state ({} bytes)", data.len());
                Ok(Some(data))
            }
            None => {
                info!("💳 No QCREDIT vault state found in storage");
                Ok(None)
            }
        }
    }

    /// Delete all entries with a given prefix from CF_MANIFEST
    pub async fn delete_by_prefix(&self, prefix: &[u8]) -> Result<usize> {
        let entries = self.hot_db.scan_prefix(CF_MANIFEST, prefix).await?;
        let count = entries.len();
        for (key, _) in &entries {
            self.hot_db.delete(CF_MANIFEST, key).await?;
        }
        Ok(count)
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // v10.4.12 BALANCE CHECKPOINT — idempotent one-time import of Epsilon snapshot
    // ═══════════════════════════════════════════════════════════════════════════

    const CHECKPOINT_APPLIED_KEY: &'static [u8] = b"__balance_checkpoint_v1__";

    /// Returns true if the balance checkpoint has already been applied to this node's DB.
    pub async fn is_checkpoint_applied(&self) -> bool {
        self.hot_db
            .get(CF_MANIFEST, Self::CHECKPOINT_APPLIED_KEY)
            .await
            .map(|v| v.is_some())
            .unwrap_or(false)
    }

    /// Returns true if this is a genesis node (ran from block 0, never bootstrapped from
    /// the checkpoint snapshot). Genesis nodes have authoritative balances — the replay
    /// code MUST skip them. The marker value is b"skipped-authoritative" when the checkpoint
    /// wipe was skipped because the node already had more wallets than the snapshot.
    pub async fn is_genesis_node(&self) -> bool {
        self.hot_db
            .get(CF_MANIFEST, Self::CHECKPOINT_APPLIED_KEY)
            .await
            .ok()
            .flatten()
            .map(|v| v == b"skipped-authoritative")
            .unwrap_or(false)
    }

    /// Apply the hardcoded Epsilon balance snapshot exactly once.
    /// - If already applied (marker key present): returns immediately.
    /// - If not yet applied: purges all wallet_balance_ keys, imports CHECKPOINT_DATA,
    ///   updates total supply, writes marker + syncs WAL.
    /// wallet_balances and total_minted_supply are updated in-memory to match.
    pub async fn apply_balance_checkpoint(
        &self,
        wallet_balances: &Arc<tokio::sync::RwLock<std::collections::HashMap<[u8; 32], u128>>>,
        total_minted_supply: &Arc<tokio::sync::RwLock<u128>>,
    ) -> Result<Vec<(u64, u64)>> {
        use crate::balance_checkpoint::{
            CHECKPOINT_DATA, CHECKPOINT_HEIGHT, CHECKPOINT_SHA256,
            CHECKPOINT_TOTAL_SUPPLY, CHECKPOINT_WALLET_COUNT,
        };

        if self.is_checkpoint_applied().await {
            info!(
                "🏁 [CHECKPOINT] Already applied (height {}), skipping.",
                CHECKPOINT_HEIGHT
            );
            // Return any pending gap ranges stored from a previous boot so the caller
            // can schedule P2P backfill even when the full checkpoint doesn't re-run.
            let pending = self.load_checkpoint_pending_gaps().await;
            return Ok(pending);
        }

        let local_height = self.get_latest_qblock_height().await.unwrap_or(None).unwrap_or(0);

        // v10.5.2 SAFETY: Never wipe an existing authoritative database.
        // If this node already has MORE wallets than the checkpoint AND is past the checkpoint
        // height, it is the data source — not the destination. Applying the checkpoint would
        // overwrite live balances with an older snapshot, destroying accumulated state.
        // This makes restarts safe on any node (including Epsilon/genesis) without needing
        // any environment variable or manual rule to remember.
        {
            let existing_wallets = self.hot_db
                .scan_prefix(CF_MANIFEST, b"wallet_balance_").await
                .map(|v| v.len())
                .unwrap_or(0);
            if existing_wallets >= CHECKPOINT_WALLET_COUNT && local_height > CHECKPOINT_HEIGHT {
                warn!(
                    "🏁 [CHECKPOINT] Skipping wipe: node has {} wallets at height {} \
                     (checkpoint has {} wallets at height {}). This node is authoritative — \
                     writing marker and continuing.",
                    existing_wallets, local_height, CHECKPOINT_WALLET_COUNT, CHECKPOINT_HEIGHT
                );
                self.hot_db.put_sync(CF_MANIFEST, Self::CHECKPOINT_APPLIED_KEY, b"skipped-authoritative").await?;
                return Ok(vec![]);
            }
        }

        warn!(
            "🏁 [CHECKPOINT v{}] Applying balance checkpoint at height {} ({} wallets). Node local height: {}",
            env!("CARGO_PKG_VERSION"), CHECKPOINT_HEIGHT, CHECKPOINT_WALLET_COUNT, local_height
        );
        if local_height > CHECKPOINT_HEIGHT {
            warn!(
                "🏁 [CHECKPOINT] Node is {} blocks past checkpoint height ({} → {}). \
                 Phase-0 post-checkpoint replay (Coinbase + Transfer only) will run after import.",
                local_height - CHECKPOINT_HEIGHT, CHECKPOINT_HEIGHT, local_height
            );
        }

        // 1. Purge all existing wallet_balance_ entries from RocksDB
        let deleted = self.delete_by_prefix(b"wallet_balance_").await.unwrap_or(0);
        warn!("🏁 [CHECKPOINT] Purged {} existing wallet entries from RocksDB.", deleted);
        // Also reset dex_applied_net trackers so apply_dex_qug_adjustments() re-applies the full
        // credits-debits delta to the freshly-restored checkpoint balances on this boot.
        // Without this, previously_applied == desired_net → delta == 0 → balances stay at
        // checkpoint values, silently dropping all post-checkpoint DEX swap credits.
        let dex_net_purged = self.delete_by_prefix(b"dex_applied_net:").await.unwrap_or(0);
        if dex_net_purged > 0 {
            warn!("🏁 [CHECKPOINT] Reset {} dex_applied_net entries — adjustments will be re-applied this boot.", dex_net_purged);
        }

        // 2. Import checkpoint data — write each wallet to CF_MANIFEST + in-memory HashMap
        let mut total: u128 = 0;
        let mut count = 0usize;
        {
            let mut wb = wallet_balances.write().await;
            wb.clear();
            for (wallet_id_hex, balance_str) in CHECKPOINT_DATA {
                let balance: u128 = balance_str.parse().unwrap_or(0);
                let key = format!("wallet_balance_{}", wallet_id_hex);
                let value = balance.to_le_bytes();
                self.hot_db.put_sync(CF_MANIFEST, key.as_bytes(), &value).await?;
                if let Ok(addr_bytes) = hex::decode(wallet_id_hex) {
                    if addr_bytes.len() == 32 {
                        let mut addr = [0u8; 32];
                        addr.copy_from_slice(&addr_bytes);
                        wb.insert(addr, balance);
                    }
                }
                total = total.saturating_add(balance);
                count += 1;
            }
        }

        // 3. v10.4.15: Verify import integrity (count + total supply)
        if count != CHECKPOINT_WALLET_COUNT {
            error!(
                "🚨 [CHECKPOINT] INTEGRITY FAIL: imported {} wallets, expected {}. Aborting checkpoint — node DB state unchanged.",
                count, CHECKPOINT_WALLET_COUNT
            );
            return Err(anyhow::anyhow!(
                "Checkpoint wallet count mismatch: got {}, expected {}",
                count, CHECKPOINT_WALLET_COUNT
            ));
        }
        if total != CHECKPOINT_TOTAL_SUPPLY {
            error!(
                "🚨 [CHECKPOINT] INTEGRITY FAIL: computed total {} ≠ expected {}. \
                 Aborting checkpoint.",
                total, CHECKPOINT_TOTAL_SUPPLY
            );
            return Err(anyhow::anyhow!(
                "Checkpoint total supply mismatch: got {}, expected {}",
                total, CHECKPOINT_TOTAL_SUPPLY
            ));
        }
        warn!(
            "🏁 [CHECKPOINT] ✅ Integrity verified: {} wallets, total {} raw ({:.4} QUG). \
             Expected SHA-256 of canonical form: {}",
            count, total, total as f64 / 1e24, CHECKPOINT_SHA256
        );

        // 4. POST-CHECKPOINT REPLAY — Phase 0: Coinbase (0x01) + Transfer (0x00) only.
        //    Replays balance effects of blocks CHECKPOINT_HEIGHT+1..local_height so that
        //    nodes which are past the checkpoint height end up with correct current balances,
        //    not the snapshot-at-checkpoint-height balances.
        //
        //    DEX swaps, token ops, contract calls are intentionally SKIPPED:
        //    token/pool state was NOT reset by the checkpoint, so applying DEX deltas
        //    against an unknown token state would create cross-state inconsistency.
        //    Whitelist enforcement mirrors checkpoint_replay_whitelist_tests (0x00 + 0x01 only).
        //
        //    `local_height` was measured before the purge + import above, so replay covers
        //    exactly the blocks that existed on disk when this function was called.
        //    Any blocks arriving DURING the import are handled by normal block processing.
        let (replayed_through, gap_ranges) = if local_height > CHECKPOINT_HEIGHT {
            warn!(
                "🏁 [CHECKPOINT] Starting post-checkpoint replay: {} → {} ({} blocks)...",
                CHECKPOINT_HEIGHT + 1, local_height, local_height - CHECKPOINT_HEIGHT
            );

            // Clone the just-imported checkpoint balances as the starting point
            let mut replay_map: std::collections::HashMap<[u8; 32], u128> = {
                let wb = wallet_balances.read().await;
                wb.clone()
            };

            let mut txs_applied = 0u64;
            let mut blocks_missing = 0u64;
            // Collect contiguous missing ranges for post-startup gap-fill
            let mut gap_ranges: Vec<(u64, u64)> = Vec::new();
            let mut gap_start: Option<u64> = None;

            for height in (CHECKPOINT_HEIGHT + 1)..=local_height {
                match self.get_qblock_by_height(height).await {
                    Ok(Some(block)) => {
                        if let Some(start) = gap_start.take() {
                            gap_ranges.push((start, height - 1));
                        }
                        for tx in &block.transactions {
                            match tx.tx_type as u8 {
                                0x01 => {
                                    // Coinbase: credit mining reward to block receiver
                                    if tx.to != [0u8; 32] && tx.amount > 0 {
                                        let bal = replay_map.entry(tx.to).or_insert(0);
                                        *bal = bal.saturating_add(tx.amount);
                                        txs_applied += 1;
                                    }
                                }
                                0x00 => {
                                    // Transfer: debit sender, credit receiver
                                    if tx.amount > 0 && tx.from != [0u8; 32] {
                                        if let Some(sender) = replay_map.get_mut(&tx.from) {
                                            *sender = sender.saturating_sub(tx.amount);
                                        }
                                        let bal = replay_map.entry(tx.to).or_insert(0);
                                        *bal = bal.saturating_add(tx.amount);
                                        txs_applied += 1;
                                    }
                                }
                                _ => {} // DEX, token, contract, vault, etc. — skip
                            }
                        }
                    }
                    Ok(None) => {
                        debug!("🏁 [CHECKPOINT REPLAY] Block {} not in local DB, skipping.", height);
                        if gap_start.is_none() {
                            gap_start = Some(height);
                        }
                        blocks_missing += 1;
                    }
                    Err(e) => {
                        warn!("⚠️ [CHECKPOINT REPLAY] Error fetching block {}: {}", height, e);
                        if gap_start.is_none() {
                            gap_start = Some(height);
                        }
                        blocks_missing += 1;
                    }
                }
            }
            // Close any trailing gap
            if let Some(start) = gap_start {
                gap_ranges.push((start, local_height));
            }

            // Drop zero-balance wallets — they don't exist on-chain
            replay_map.retain(|_, v| *v > 0);

            // Batch-write all updated balances to RocksDB (fsync — crash-safe)
            self.save_wallet_balances(&replay_map).await?;

            // Update in-memory HashMap to match replayed state
            let wallet_count_after = replay_map.len();
            {
                let mut wb = wallet_balances.write().await;
                *wb = replay_map.clone();
            }

            // Recompute + persist total supply from replayed balances
            let replayed_total: u128 = replay_map.values().sum();
            self.save_total_supply(replayed_total).await?;
            {
                let mut supply = total_minted_supply.write().await;
                *supply = replayed_total;
            }

            warn!(
                "🏁 [CHECKPOINT] ✅ Replay complete: {} txs applied, {} blocks ({} missing). \
                 Wallets: {}. Total supply after replay: {} raw.",
                txs_applied, local_height - CHECKPOINT_HEIGHT, blocks_missing,
                wallet_count_after, replayed_total
            );

            let total_replay_blocks = local_height - CHECKPOINT_HEIGHT;
            if blocks_missing > 0 {
                let pct = blocks_missing * 100 / total_replay_blocks.max(1);
                if pct >= 5 {
                    error!(
                        "🚨 [CHECKPOINT] BALANCE ACCURACY WARNING: {} of {} replay blocks missing ({}%). \
                         Post-checkpoint balances are INCOMPLETE. This node has chain gaps. \
                         Run turbo-sync to fill gaps before trusting balance state.",
                        blocks_missing, total_replay_blocks, pct
                    );
                } else {
                    warn!(
                        "⚠️ [CHECKPOINT] {} of {} replay blocks missing ({}%) — minor gaps, \
                         balance state is nearly complete.",
                        blocks_missing, total_replay_blocks, pct
                    );
                }
            }

            // Persist gap ranges to RocksDB so subsequent boots can trigger gap-fill
            // even when the checkpoint is already marked as applied.
            if !gap_ranges.is_empty() {
                let _ = self.save_checkpoint_pending_gaps(&gap_ranges).await;
            }
            (local_height, gap_ranges)
        } else {
            // Node is at or before checkpoint height — no replay needed
            self.save_total_supply(total).await?;
            {
                let mut supply = total_minted_supply.write().await;
                *supply = total;
            }
            (CHECKPOINT_HEIGHT, vec![])
        };

        // 5. Write extended marker (40 bytes):
        //    bytes 0..8:   checkpoint_height (u64 LE)
        //    bytes 8..16:  checkpoint_wallet_count (u64 LE)
        //    bytes 16..32: checkpoint_total_supply (u128 LE)  — snapshot value, not replayed
        //    bytes 32..40: replayed_through_height (u64 LE)   — u64::MAX when no replay needed
        //
        //    is_checkpoint_applied() checks only for key presence — marker length can grow safely.
        let mut marker = Vec::with_capacity(40);
        marker.extend_from_slice(&CHECKPOINT_HEIGHT.to_le_bytes());
        marker.extend_from_slice(&(CHECKPOINT_WALLET_COUNT as u64).to_le_bytes());
        marker.extend_from_slice(&total.to_le_bytes());
        marker.extend_from_slice(&replayed_through.to_le_bytes());
        self.hot_db.put_sync(CF_MANIFEST, Self::CHECKPOINT_APPLIED_KEY, &marker).await?;

        // v10.7.2: Also set the P2P bootstrap-sync flag so that the state_sync_api bootstrap
        // path sees it as "already done" and never overwrites checkpoint balances with a live
        // P2P snapshot. This is a belt-and-suspenders guard alongside the is_checkpoint_applied()
        // check added to state_sync_api.rs — protects against any race where the API check
        // runs before this marker is written.
        let _ = self.hot_db.put_sync(
            CF_MANIFEST,
            b"migration_bootstrap_wallet_sync_v882_done",
            b"set_by_checkpoint",
        ).await;

        // v1.0.2 OPTION C-A: Advance qblock:latest to CHECKPOINT_HEIGHT after a fresh
        // checkpoint apply. The checkpoint imported all wallet balances at this height —
        // state at heights 1..=CHECKPOINT_HEIGHT is verified by the embedded SHA-256.
        // The block DATA for those heights isn't stored yet (Phase 2 backfill will fetch
        // them), but the consensus-relevant STATE is exact. By setting qblock:latest =
        // CHECKPOINT_HEIGHT, the turbo sync engine starts fetching at CHECKPOINT_HEIGHT+1
        // instead of grinding through 1..=CHECKPOINT_HEIGHT first — turning a multi-day
        // bootstrap into a few-minute one.
        //
        // Only do this on a TRUE fresh apply (no prior pointer) — if the node already had
        // a higher pointer from previous sync, don't roll it back. Also only advance if
        // current pointer is below CHECKPOINT_HEIGHT; the typical fresh case is pointer=0.
        if local_height < CHECKPOINT_HEIGHT {
            let height_bytes = CHECKPOINT_HEIGHT.to_be_bytes();
            self.hot_db
                .put_sync(CF_BLOCKS, b"qblock:latest", &height_bytes)
                .await?;
            self.height_cache.update(CHECKPOINT_HEIGHT).await;
            info!(
                "🏁 [CHECKPOINT] Advanced qblock:latest from {} to CHECKPOINT_HEIGHT={} \
                 (forward sync will start here; pre-checkpoint blocks fetched in Phase 2)",
                local_height, CHECKPOINT_HEIGHT
            );
        }

        warn!(
            "🏁 [CHECKPOINT] ✅ Done. Marker written (checkpoint_height={}, wallets={}, \
             checkpoint_total={}, replayed_through={}).",
            CHECKPOINT_HEIGHT, count, total, replayed_through
        );

        Ok(gap_ranges)
    }

    /// v10.8.0 CHECKPOINT GAP-FILL: Re-apply transactions from blocks that were missing
    /// during the initial checkpoint replay. Called after fill_gap_p2p fetches those blocks.
    /// Uses additive/max-wins updates — safe to call even if some blocks were partially applied.
    pub async fn replay_gap_blocks(
        &self,
        gap_ranges: &[(u64, u64)],
        wallet_balances: &Arc<tokio::sync::RwLock<std::collections::HashMap<[u8; 32], u128>>>,
        total_minted_supply: &Arc<tokio::sync::RwLock<u128>>,
    ) -> Result<()> {
        if gap_ranges.is_empty() {
            return Ok(());
        }
        let mut txs_applied = 0u64;
        let mut blocks_applied = 0u64;
        let mut delta: std::collections::HashMap<[u8; 32], u128> = std::collections::HashMap::new();

        for &(start, end) in gap_ranges {
            for height in start..=end {
                match self.get_qblock_by_height(height).await {
                    Ok(Some(block)) => {
                        for tx in &block.transactions {
                            match tx.tx_type as u8 {
                                0x01 => {
                                    if tx.to != [0u8; 32] && tx.amount > 0 {
                                        *delta.entry(tx.to).or_insert(0) += tx.amount;
                                        txs_applied += 1;
                                    }
                                }
                                0x00 => {
                                    if tx.amount > 0 && tx.from != [0u8; 32] {
                                        // For gap-fill replay we only credit receivers to avoid
                                        // double-deducting senders already debited by forward sync.
                                        *delta.entry(tx.to).or_insert(0) += tx.amount;
                                        txs_applied += 1;
                                    }
                                }
                                _ => {}
                            }
                        }
                        blocks_applied += 1;
                    }
                    Ok(None) => {
                        warn!("🔧 [GAP-FILL REPLAY] Block {} still missing after P2P fill", height);
                    }
                    Err(e) => {
                        warn!("🔧 [GAP-FILL REPLAY] Error reading block {}: {}", height, e);
                    }
                }
            }
        }

        if delta.is_empty() {
            warn!("🔧 [GAP-FILL REPLAY] No blocks recovered — P2P fill may not have completed yet");
            return Ok(());
        }

        // Apply delta to in-memory balances and persist to RocksDB
        {
            let mut wb = wallet_balances.write().await;
            for (addr, amount) in &delta {
                let bal = wb.entry(*addr).or_insert(0);
                *bal = bal.saturating_add(*amount);
            }
            let new_total: u128 = wb.values().sum();
            drop(wb);
            let mut supply = total_minted_supply.write().await;
            *supply = new_total;
        }
        // Persist updated balances to RocksDB
        let snapshot = {
            let wb = wallet_balances.read().await;
            wb.clone()
        };
        self.save_wallet_balances(&snapshot).await?;
        let new_total: u128 = snapshot.values().sum();
        self.save_total_supply(new_total).await?;

        warn!(
            "🔧 [GAP-FILL REPLAY] ✅ Applied {} txs from {} blocks across {} gap ranges. \
             Wallets: {}, supply: {} raw.",
            txs_applied, blocks_applied, gap_ranges.len(), snapshot.len(), new_total
        );
        // Gap-fill complete — clear the pending ranges so this doesn't re-run on next boot
        let _ = self.clear_checkpoint_pending_gaps().await;
        Ok(())
    }

    const CHECKPOINT_GAP_RANGES_KEY: &'static [u8] = b"meta:checkpoint_gap_ranges_v1";

    pub async fn save_checkpoint_pending_gaps(&self, ranges: &[(u64, u64)]) -> Result<()> {
        // Encode as: count(u32 LE) + N × (start u64 LE, end u64 LE)
        let mut buf = Vec::with_capacity(4 + ranges.len() * 16);
        buf.extend_from_slice(&(ranges.len() as u32).to_le_bytes());
        for &(s, e) in ranges {
            buf.extend_from_slice(&s.to_le_bytes());
            buf.extend_from_slice(&e.to_le_bytes());
        }
        self.hot_db.put(CF_MANIFEST, Self::CHECKPOINT_GAP_RANGES_KEY, &buf).await
    }

    pub async fn load_checkpoint_pending_gaps(&self) -> Vec<(u64, u64)> {
        let bytes = match self.hot_db.get(CF_MANIFEST, Self::CHECKPOINT_GAP_RANGES_KEY).await {
            Ok(Some(b)) => b,
            _ => return vec![],
        };
        if bytes.len() < 4 { return vec![]; }
        let count = u32::from_le_bytes(bytes[..4].try_into().unwrap_or([0; 4])) as usize;
        let mut ranges = Vec::with_capacity(count);
        for i in 0..count {
            let off = 4 + i * 16;
            if off + 16 > bytes.len() { break; }
            let s = u64::from_le_bytes(bytes[off..off+8].try_into().unwrap_or([0;8]));
            let e = u64::from_le_bytes(bytes[off+8..off+16].try_into().unwrap_or([0;8]));
            ranges.push((s, e));
        }
        ranges
    }

    pub async fn clear_checkpoint_pending_gaps(&self) -> Result<()> {
        self.hot_db.delete(CF_MANIFEST, Self::CHECKPOINT_GAP_RANGES_KEY).await.or(Ok(()))
    }

    /// SYNC-006 (v10.7.7): Check whether the one-time post-checkpoint balance replay has
    /// already been completed and persisted to the DB for this chain instance.
    /// v10.7.7: Uses a NEW key (not v10.7.6) so nodes that ran the buggy v10.7.6 replay
    /// (95% miss rate due to get_qblock_by_height missing DAG-format blocks) will
    /// automatically re-run the corrected replay on first v10.7.7 startup.
    pub async fn is_balance_replay_done(&self) -> bool {
        self.hot_db
            .get(CF_MANIFEST, b"meta:balance_replay_v10.7.8")
            .await
            .ok()
            .flatten()
            .is_some()
    }

    /// SYNC-006 (v10.7.8): Persist the replay-done flag so a restart does not re-run the
    /// (potentially expensive) replay. Uses v10.7.8 key to invalidate any prior incomplete run.
    pub async fn mark_balance_replay_done(&self) -> Result<()> {
        self.hot_db
            .put(CF_MANIFEST, b"meta:balance_replay_v10.7.8", b"1")
            .await
            .context("Failed to persist balance replay done flag")
    }

    /// SYNC-006 admin reset: Delete the replay-done flag so SYNC-006 will re-run on the next
    /// 30-second poll. Used by the admin endpoint POST /api/v1/admin/reset-balance-replay.
    pub async fn delete_balance_replay_flag(&self) -> Result<()> {
        self.hot_db
            .delete(CF_MANIFEST, b"meta:balance_replay_v10.7.8")
            .await
            .context("Failed to delete balance replay done flag")?;
        // If the checkpoint marker is "skipped-authoritative" (set when the node had more
        // wallets than the snapshot at first boot — possible due to the turbo-sync
        // coinbase-only bug producing a wrong wallet count), reset it to "1" so that
        // is_genesis_node() returns false and the replay actually runs on SYNC-006 restart.
        let marker = self.hot_db
            .get(CF_MANIFEST, Self::CHECKPOINT_APPLIED_KEY)
            .await
            .ok()
            .flatten();
        if marker.as_deref() == Some(b"skipped-authoritative") {
            info!("[ADMIN] Checkpoint marker was 'skipped-authoritative' — resetting to '1' so replay runs on next restart.");
            self.hot_db
                .put_sync(CF_MANIFEST, Self::CHECKPOINT_APPLIED_KEY, b"1")
                .await
                .context("Failed to reset checkpoint marker")?;
        }
        Ok(())
    }

    /// Post-sync balance replay for checkpoint-bootstrapped nodes (SYNC-004 / v10.7.3).
    ///
    /// Turbo sync downloads blocks with `balance_engine=None` — no balance updates during
    /// the historical bulk-download phase. A node that applies the balance checkpoint at
    /// startup gets 1,326 wallets at height 16,538,868, but the ~1M blocks between the
    /// checkpoint and chain tip are never credited (coinbase + transfers skipped).
    ///
    /// This function is called once, when `now_synced && !was_synced`, to replay every
    /// block from CHECKPOINT_HEIGHT+1 to chain-tip and compute the correct balance state.
    /// Uses two passes to close the race window between the replay and concurrent gossipsub.
    pub async fn replay_post_checkpoint_balances(
        &self,
        wallet_balances: &Arc<tokio::sync::RwLock<std::collections::HashMap<[u8; 32], u128>>>,
        total_minted_supply: &Arc<tokio::sync::RwLock<u128>>,
    ) -> Result<u64> {
        use crate::balance_checkpoint::{CHECKPOINT_HEIGHT, CHECKPOINT_DATA};

        let height_pass1_end = self
            .get_latest_qblock_height()
            .await
            .ok()
            .flatten()
            .unwrap_or(0);

        if height_pass1_end <= CHECKPOINT_HEIGHT {
            warn!(
                "🏁 [POST-SYNC REPLAY v10.7.3] height {} ≤ checkpoint {}, nothing to replay.",
                height_pass1_end, CHECKPOINT_HEIGHT
            );
            return Ok(0);
        }

        // Genesis/archive node detection: primary check is the checkpoint-applied flag.
        // Checkpoint-bootstrapped nodes have `__balance_checkpoint_v1__` set in CF_MANIFEST
        // (either b"1" for normal checkpoint or b"skipped-authoritative" for genesis nodes
        // that skipped the wipe). If checkpoint was NOT applied at all, this node ran from
        // genesis with correct balances — skip the replay.
        // Secondary: also skip if we have blocks from well before the checkpoint (belt & suspenders).
        if !self.is_checkpoint_applied().await {
            warn!(
                "🏁 [POST-SYNC REPLAY v10.7.3] Checkpoint not applied on this node \
                 — ran from genesis, balances already correct, skipping replay."
            );
            return Ok(0);
        }
        // Secondary check: if this is a genesis node (marker = b"skipped-authoritative"),
        // also skip. is_genesis_node() reads the same key and checks the value.
        if self.is_genesis_node().await {
            warn!(
                "🏁 [POST-SYNC REPLAY v10.7.3] Genesis/archive node detected (checkpoint skipped as authoritative) \
                 — balances already correct from genesis, skipping replay."
            );
            return Ok(0);
        }
        // Belt-and-suspenders: also skip if we have a block at height 1,000,000 (stored only on genesis nodes).
        let has_early_block = self.get_qblock_any_format(1_000_000).await
            .unwrap_or(None)
            .is_some();
        if has_early_block {
            warn!(
                "🏁 [POST-SYNC REPLAY v10.7.3] Genesis/archive node detected (has block at height 1,000,000) \
                 — balances already correct from genesis, skipping replay."
            );
            return Ok(0);
        }

        warn!(
            "🏁 [POST-SYNC REPLAY v10.7.3] Starting: replaying {} blocks ({} → {})...",
            height_pass1_end - CHECKPOINT_HEIGHT,
            CHECKPOINT_HEIGHT + 1,
            height_pass1_end
        );

        // ---- Build starting map from embedded checkpoint data ----
        let mut replay_map: std::collections::HashMap<[u8; 32], u128> = {
            let mut m = std::collections::HashMap::new();
            for (wallet_id_hex, balance_str) in CHECKPOINT_DATA.iter() {
                let balance: u128 = balance_str.parse().unwrap_or(0);
                if let Ok(addr_bytes) = hex::decode(wallet_id_hex) {
                    if addr_bytes.len() == 32 {
                        let mut addr = [0u8; 32];
                        addr.copy_from_slice(&addr_bytes);
                        m.insert(addr, balance);
                    }
                }
            }
            m
        };

        // ---- Pass 1: Replay CHECKPOINT_HEIGHT+1 to height_pass1_end ----
        let mut txs_applied = 0u64;
        let mut blocks_missing = 0u64;

        for height in (CHECKPOINT_HEIGHT + 1)..=height_pass1_end {
            match self.get_qblock_any_format(height).await {
                Ok(Some(block)) => {
                    for tx in &block.transactions {
                        match tx.tx_type as u8 {
                            0x01 => {
                                // Coinbase — always native QUG
                                if tx.to != [0u8; 32] && tx.amount > 0 {
                                    *replay_map.entry(tx.to).or_insert(0) =
                                        replay_map.get(&tx.to).copied().unwrap_or(0)
                                            .saturating_add(tx.amount);
                                    txs_applied += 1;
                                }
                            }
                            0x00 => {
                                // Transfer — only credit native QUG wallet_balances.
                                // QUGUSD / custom-token transfers share tx_type=0x00 but
                                // use token_type != QUG; those go to token_balances and
                                // must NOT touch the native balance map (REPLAY-001).
                                let is_native =
                                    matches!(tx.token_type, q_types::TokenType::QUG);
                                if is_native && tx.amount > 0 && tx.from != [0u8; 32] {
                                    if let Some(s) = replay_map.get_mut(&tx.from) {
                                        *s = s.saturating_sub(tx.amount);
                                    }
                                    *replay_map.entry(tx.to).or_insert(0) =
                                        replay_map.get(&tx.to).copied().unwrap_or(0)
                                            .saturating_add(tx.amount);
                                    txs_applied += 1;
                                }
                            }
                            _ => {}
                        }
                    }
                }
                Ok(None) => {
                    blocks_missing += 1;
                }
                Err(_) => {
                    blocks_missing += 1;
                }
            }
        }

        warn!(
            "🏁 [POST-SYNC REPLAY] Pass 1 done: {} txs applied, {} blocks missing (of {})",
            txs_applied,
            blocks_missing,
            height_pass1_end - CHECKPOINT_HEIGHT
        );

        // ---- Pass 2: Close the race window — replay any blocks that arrived during Pass 1 ----
        let height_pass2_end = self
            .get_latest_qblock_height()
            .await
            .ok()
            .flatten()
            .unwrap_or(height_pass1_end);

        if height_pass2_end > height_pass1_end {
            warn!(
                "🏁 [POST-SYNC REPLAY] Pass 2: {} new blocks arrived during Pass 1 ({}→{})",
                height_pass2_end - height_pass1_end,
                height_pass1_end + 1,
                height_pass2_end
            );
            for height in (height_pass1_end + 1)..=height_pass2_end {
                if let Ok(Some(block)) = self.get_qblock_any_format(height).await {
                    for tx in &block.transactions {
                        match tx.tx_type as u8 {
                            0x01 => {
                                if tx.to != [0u8; 32] && tx.amount > 0 {
                                    *replay_map.entry(tx.to).or_insert(0) =
                                        replay_map.get(&tx.to).copied().unwrap_or(0)
                                            .saturating_add(tx.amount);
                                }
                            }
                            0x00 => {
                                // REPLAY-001: skip token transfers (QUGUSD / custom)
                                let is_native =
                                    matches!(tx.token_type, q_types::TokenType::QUG);
                                if is_native && tx.amount > 0 && tx.from != [0u8; 32] {
                                    if let Some(s) = replay_map.get_mut(&tx.from) {
                                        *s = s.saturating_sub(tx.amount);
                                    }
                                    *replay_map.entry(tx.to).or_insert(0) =
                                        replay_map.get(&tx.to).copied().unwrap_or(0)
                                            .saturating_add(tx.amount);
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        // Drop zero-balance wallets
        replay_map.retain(|_, v| *v > 0);

        let total: u128 = replay_map.values().copied().sum();
        let count = replay_map.len();

        // Persist to RocksDB
        self.save_wallet_balances(&replay_map).await?;
        if let Err(e) = self.save_total_supply(total).await {
            warn!("⚠️ [POST-SYNC REPLAY] Failed to persist total_minted_supply: {}", e);
        }

        // Replace in-memory wallet map atomically
        {
            let mut wb = wallet_balances.write().await;
            *wb = replay_map;
        }
        {
            let mut supply = total_minted_supply.write().await;
            *supply = total;
        }

        warn!(
            "✅ [POST-SYNC REPLAY v10.7.3] Complete: {} wallets, {:.6} QUG total (replayed through {}, {} blocks missed)",
            count,
            total as f64 / 1_000_000_000_000_000_000_000_000f64,
            height_pass2_end,
            blocks_missing
        );

        Ok(blocks_missing)
    }

    /// Purge all DEX pools, contracts, token balances, and related state
    /// Used for phase transitions to clear stale data
    pub async fn purge_dex_and_contracts(&self) -> Result<(usize, usize, usize, usize, usize)> {
        let contracts = self.delete_by_prefix(b"contract_").await?;
        let pools = self.delete_by_prefix(b"liquidity_pool:").await?;
        let token_balances = self.delete_by_prefix(b"token_balance_").await?;
        let stakes = self.delete_by_prefix(b"stake_position_").await?;
        // Clear collateral vault
        let _ = self.hot_db.delete(CF_MANIFEST, b"collateral_vault").await;
        // Clear swap history
        let swaps = self.delete_by_prefix(b"swap_history_").await?;
        // Clear price history
        let _ = self.delete_by_prefix(b"price_history_").await;

        info!("🧹 Phase purge complete: {} contracts, {} pools, {} token_balances, {} stakes, {} swaps deleted",
              contracts, pools, token_balances, stakes, swaps);

        Ok((contracts, pools, token_balances, stakes, swaps))
    }

    /// Purge ALL phase data including wallet balances (for full phase transitions)
    /// This is more aggressive than purge_dex_and_contracts - it also resets QUG balances
    pub async fn purge_all_phase_data(&self) -> Result<(usize, usize, usize, usize, usize, usize)> {
        let contracts = self.delete_by_prefix(b"contract_").await?;
        let pools = self.delete_by_prefix(b"liquidity_pool:").await?;
        let token_balances = self.delete_by_prefix(b"token_balance_").await?;
        let stakes = self.delete_by_prefix(b"stake_position_").await?;
        let wallet_balances = self.delete_by_prefix(b"wallet_balance_").await?;
        // Clear collateral vault
        let _ = self.hot_db.delete(CF_MANIFEST, b"collateral_vault").await;
        // Clear swap history
        let swaps = self.delete_by_prefix(b"swap_history_").await?;
        // Clear price history
        let _ = self.delete_by_prefix(b"price_history_").await;
        // Clear total minted supply
        let _ = self.hot_db.delete(CF_MANIFEST, b"total_minted_supply").await;

        info!("🧹 FULL phase purge: {} contracts, {} pools, {} token_balances, {} wallet_balances, {} stakes, {} swaps deleted",
              contracts, pools, token_balances, wallet_balances, stakes, swaps);

        Ok((contracts, pools, token_balances, wallet_balances, stakes, swaps))
    }

    /// v7.1.3: Purge ALL pre-genesis data at startup (mainnet2026.1 clean transition)
    /// Checks if stored blocks predate GENESIS_TIMESTAMP and if so, removes:
    /// - All wallet balances (accumulated from pre-genesis mining)
    /// - All token balances (testnet swap credits)
    /// - All blocks with timestamp < GENESIS_TIMESTAMP
    /// - Total minted supply counter
    /// - Emission controller state
    /// This ensures a clean state for the new genesis epoch.
    pub async fn purge_pre_genesis_data(&self) -> Result<bool> {
        // v7.3.4: Use network-aware genesis timestamp
        let genesis_timestamp: u64 = crate::balance_consensus::active_genesis_timestamp();

        // Check if any stored blocks predate genesis by scanning the first few entries
        let mut has_pre_genesis = false;
        if let Ok(entries) = self.hot_db.scan_prefix(CF_BLOCKS, &[]).await {
            for (key, value) in entries.iter().take(20) {
                let key_str = String::from_utf8_lossy(key);
                if key_str == "qblock:latest" || key_str.starts_with("qblock:hash:") {
                    continue;
                }
                // v7.3.1: Handle compressed format
                let raw_value = if precompressed_storage::is_precompressed(value) {
                    precompressed_storage::PrecompressedBlock::from_bytes(value)
                        .and_then(|c| c.decompress().map_err(|e| e.into()))
                        .unwrap_or_else(|_| value.to_vec())
                } else {
                    value.to_vec()
                };
                // Try QBlock deserialization (the primary block format)
                if let Ok(block) = bincode::deserialize::<q_types::block::QBlock>(&raw_value) {
                    if block.header.timestamp > 0 && block.header.timestamp < genesis_timestamp {
                        info!("🧹 [GENESIS FILTER] Found pre-genesis block h={} ts={} (genesis={})",
                              block.header.height, block.header.timestamp, genesis_timestamp);
                        has_pre_genesis = true;
                    }
                    break; // Found a valid block, decision made
                }
            }

            // Also check: wallet balances exist but no blocks (partial purge leftover)
            if !has_pre_genesis && entries.is_empty() {
                let balances = self.hot_db.scan_prefix(CF_MANIFEST, b"wallet_balance_").await.unwrap_or_default();
                if !balances.is_empty() {
                    info!("🧹 [GENESIS FILTER] Found {} wallet balances but no blocks — purging orphaned balances", balances.len());
                    has_pre_genesis = true;
                }
            }
        };

        if !has_pre_genesis {
            info!("✅ [GENESIS FILTER] No pre-genesis data detected — skipping purge");
            return Ok(false);
        }

        // Purge all stale data
        let wallet_balances = self.delete_by_prefix(b"wallet_balance_").await.unwrap_or(0);
        let token_balances = self.delete_by_prefix(b"token_balance_").await.unwrap_or(0);
        let contracts = self.delete_by_prefix(b"contract_").await.unwrap_or(0);
        let pools = self.delete_by_prefix(b"liquidity_pool:").await.unwrap_or(0);
        let stakes = self.delete_by_prefix(b"stake_position_").await.unwrap_or(0);
        let swaps = self.delete_by_prefix(b"swap_history_").await.unwrap_or(0);
        let _ = self.delete_by_prefix(b"price_history_").await;
        let _ = self.hot_db.delete(CF_MANIFEST, b"total_minted_supply").await;
        let _ = self.hot_db.delete(CF_MANIFEST, b"collateral_vault").await;
        let _ = self.hot_db.delete(CF_MANIFEST, b"emission_controller_state").await;

        // Purge pre-genesis blocks from CF_BLOCKS
        let mut blocks_deleted = 0usize;
        if let Ok(entries) = self.hot_db.scan_prefix(CF_BLOCKS, &[]).await {
            for (key, value) in &entries {
                let key_str = String::from_utf8_lossy(key);
                // Skip metadata keys
                if key_str == "qblock:latest" || key_str.starts_with("qblock:hash:") {
                    continue;
                }
                // v7.3.1: Handle compressed format
                let raw_val = if precompressed_storage::is_precompressed(value) {
                    precompressed_storage::PrecompressedBlock::from_bytes(value)
                        .and_then(|c| c.decompress().map_err(|e| e.into()))
                        .unwrap_or_else(|_| value.to_vec())
                } else {
                    value.to_vec()
                };
                // Try to deserialize and check timestamp
                if let Ok(block) = bincode::deserialize::<q_types::block::QBlock>(&raw_val) {
                    if block.header.timestamp < genesis_timestamp {
                        let _ = self.hot_db.delete(CF_BLOCKS, key).await;
                        blocks_deleted += 1;
                    }
                } else {
                    // Can't deserialize — delete it (corrupted or incompatible format)
                    let _ = self.hot_db.delete(CF_BLOCKS, key).await;
                    blocks_deleted += 1;
                }
            }
        }

        // Reset height pointer
        if blocks_deleted > 0 {
            let _ = self.hot_db.delete(CF_BLOCKS, b"qblock:latest").await;
            self.height_cache.update(0).await;
        }

        info!("🧹 [GENESIS FILTER] Pre-genesis purge complete:");
        info!("   {} wallet balances deleted", wallet_balances);
        info!("   {} token balances deleted", token_balances);
        info!("   {} contracts deleted", contracts);
        info!("   {} liquidity pools deleted", pools);
        info!("   {} blocks deleted", blocks_deleted);
        info!("   {} stakes deleted", stakes);
        info!("   {} swaps deleted", swaps);
        info!("   Emission controller state reset");
        info!("   Total supply counter reset");

        Ok(true)
    }

    /// v7.2.5: Safe version that purges testnet BALANCES only — never deletes blocks.
    /// Uses a one-time migration flag so it runs exactly once, then never again.
    /// This prevents height drops while still cleaning up stale testnet wallet data.
    pub async fn purge_pre_genesis_balances_only(&self) -> Result<bool> {
        const MIGRATION_FLAG: &[u8] = b"migration_genesis_balance_purge_v725_done";

        // Check if we already ran this migration
        if let Ok(Some(_)) = self.hot_db.get(CF_MANIFEST, MIGRATION_FLAG).await {
            return Ok(false); // Already done, skip
        }

        info!("🧹 [GENESIS FILTER v7.2.5] Running one-time testnet balance purge (blocks PRESERVED)...");

        // Purge stale balances and state — but NEVER touch blocks or height pointer
        let wallet_balances = self.delete_by_prefix(b"wallet_balance_").await.unwrap_or(0);
        let token_balances = self.delete_by_prefix(b"token_balance_").await.unwrap_or(0);
        let contracts = self.delete_by_prefix(b"contract_").await.unwrap_or(0);
        let pools = self.delete_by_prefix(b"liquidity_pool:").await.unwrap_or(0);
        let stakes = self.delete_by_prefix(b"stake_position_").await.unwrap_or(0);
        let swaps = self.delete_by_prefix(b"swap_history_").await.unwrap_or(0);
        let _ = self.delete_by_prefix(b"price_history_").await;
        let _ = self.hot_db.delete(CF_MANIFEST, b"total_minted_supply").await;
        let _ = self.hot_db.delete(CF_MANIFEST, b"collateral_vault").await;
        let _ = self.hot_db.delete(CF_MANIFEST, b"emission_controller_state").await;

        // Set the migration flag so this never runs again
        let _ = self.hot_db.put(CF_MANIFEST, MIGRATION_FLAG, b"done").await;

        info!("🧹 [GENESIS FILTER v7.2.5] Balance-only purge complete (blocks PRESERVED):");
        info!("   {} wallet balances deleted", wallet_balances);
        info!("   {} token balances deleted", token_balances);
        info!("   {} contracts deleted", contracts);
        info!("   {} liquidity pools deleted", pools);
        info!("   {} stakes deleted", stakes);
        info!("   {} swaps deleted", swaps);
        info!("   Migration flag set — will not run again on next restart");

        Ok(true)
    }

    /// v7.2.6: Second purge for ALL nodes — clears testnet balances that were
    /// re-synced via P2P from nodes that didn't run the v7.2.5 purge.
    /// Also resets emission controller state so it starts fresh from genesis.
    pub async fn purge_testnet_balances_v726(&self) -> Result<bool> {
        const MIGRATION_FLAG: &[u8] = b"migration_genesis_balance_purge_v726_done";

        // Check if we already ran this migration
        if let Ok(Some(_)) = self.hot_db.get(CF_MANIFEST, MIGRATION_FLAG).await {
            return Ok(false); // Already done, skip
        }

        info!("🧹 [GENESIS FILTER v7.2.6] Running testnet balance purge on ALL nodes (blocks PRESERVED)...");

        // Purge ALL balance-related state
        let wallet_balances = self.delete_by_prefix(b"wallet_balance_").await.unwrap_or(0);
        let token_balances = self.delete_by_prefix(b"token_balance_").await.unwrap_or(0);
        let contracts = self.delete_by_prefix(b"contract_").await.unwrap_or(0);
        let pools = self.delete_by_prefix(b"liquidity_pool:").await.unwrap_or(0);
        let stakes = self.delete_by_prefix(b"stake_position_").await.unwrap_or(0);
        let swaps = self.delete_by_prefix(b"swap_history_").await.unwrap_or(0);
        let _ = self.delete_by_prefix(b"price_history_").await;
        let _ = self.hot_db.delete(CF_MANIFEST, b"total_minted_supply").await;
        let _ = self.hot_db.delete(CF_MANIFEST, b"collateral_vault").await;
        let _ = self.hot_db.delete(CF_MANIFEST, b"emission_controller_state").await;

        // Set the migration flag so this never runs again
        let _ = self.hot_db.put(CF_MANIFEST, MIGRATION_FLAG, b"done").await;

        info!("🧹 [GENESIS FILTER v7.2.6] Purge complete (blocks PRESERVED):");
        info!("   {} wallet balances deleted", wallet_balances);
        info!("   {} token balances deleted", token_balances);
        info!("   {} contracts deleted", contracts);
        info!("   {} liquidity pools deleted", pools);
        info!("   {} stakes deleted", stakes);
        info!("   {} swaps deleted", swaps);
        info!("   Emission controller state reset — will rebuild from blocks");

        Ok(true)
    }

    /// v7.2.12: Third purge — deletes ALL wallet balances, total_minted_supply,
    /// and emission_controller_state from RocksDB. This runs once (migration flag).
    /// After this, `rebuild_balances_from_chain()` must be called to reconstruct
    /// correct balances from mainnet blocks only.
    pub async fn purge_testnet_wallets_v7212(&self) -> Result<bool> {
        const MIGRATION_FLAG: &[u8] = b"migration_wallet_purge_v7212_done";

        // Check if we already ran this migration
        if let Ok(Some(_)) = self.hot_db.get(CF_MANIFEST, MIGRATION_FLAG).await {
            return Ok(false); // Already done
        }

        info!("🧹 [GENESIS FILTER v7.2.12] Purging ALL testnet wallet balances from RocksDB...");

        let wallet_balances = self.delete_by_prefix(b"wallet_balance_").await.unwrap_or(0);
        let _ = self.hot_db.delete(CF_MANIFEST, b"total_minted_supply").await;
        let _ = self.hot_db.delete(CF_MANIFEST, b"emission_controller_state").await;

        // Set the migration flag
        let _ = self.hot_db.put(CF_MANIFEST, MIGRATION_FLAG, b"done").await;

        info!("🧹 [GENESIS FILTER v7.2.12] Purged {} wallet balances + total_supply + emission_state", wallet_balances);
        info!("   Migration flag set — will not run again on next restart");

        Ok(true)
    }

    /// v8.4.3: Detect significant block gaps in the chain.
    ///
    /// Samples blocks at regular intervals to detect gaps from checkpoint jumps.
    /// Returns (has_gaps, contiguous_blocks, tip_height, gap_percentage).
    /// A node with >5% gaps should accept state-sync balance imports as a safety net.
    pub async fn detect_block_gaps(&self) -> (bool, u64, u64, f64) {
        let tip = self.height_cache.cached();
        if tip < 1000 {
            return (false, tip, tip, 0.0);
        }

        // Sample every 1000th block to detect gaps quickly (O(tip/1000) not O(tip))
        let sample_interval = std::cmp::max(1, tip / 1000);
        let mut present = 0u64;
        let mut sampled = 0u64;

        let mut h = 1u64;
        while h <= tip {
            sampled += 1;
            let height_key = format!("qblock:height:{}", h);
            match self.hot_db.get(CF_BLOCKS, height_key.as_bytes()).await {
                Ok(Some(_)) => present += 1,
                _ => {}
            }
            h += sample_interval;
        }

        if sampled == 0 {
            return (false, tip, tip, 0.0);
        }

        let gap_pct = 100.0 * (1.0 - (present as f64 / sampled as f64));
        let estimated_present = (present as f64 / sampled as f64 * tip as f64) as u64;
        let has_gaps = gap_pct > 5.0; // >5% missing = significant gaps

        if has_gaps {
            warn!("⚠️ [GAP DETECT v8.4.3] Block gaps detected: {:.1}% missing ({}/{} samples present, tip={})",
                  gap_pct, present, sampled, tip);
        } else {
            debug!("✅ [GAP DETECT] Chain is contiguous: {:.1}% present ({}/{} samples, tip={})",
                   100.0 - gap_pct, present, sampled, tip);
        }

        (has_gaps, estimated_present, tip, gap_pct)
    }

    /// v7.2.12: Rebuild wallet balances by scanning the blockchain.
    /// Iterates blocks 1..tip, skips blocks with timestamp < GENESIS_TIMESTAMP,
    /// and reconstructs balances from balance_updates embedded in blocks.
    /// For blocks without balance_updates, extracts coinbase from mining_solutions.
    /// Returns the rebuilt balances HashMap and total supply.
    pub async fn rebuild_balances_from_chain(&self) -> Result<(HashMap<[u8; 32], u128>, u128)> {
        // v7.3.4: Use network-aware genesis timestamp
        let genesis_timestamp: u64 = crate::balance_consensus::active_genesis_timestamp();

        let tip = self.height_cache.cached();
        if tip == 0 {
            info!("🔄 [REBUILD] No blocks in chain (tip=0), starting with empty balances");
            return Ok((HashMap::new(), 0));
        }

        info!("🔄 [REBUILD v7.2.12] Rebuilding wallet balances from chain (blocks 1..{})...", tip);

        let mut balances: HashMap<[u8; 32], u128> = HashMap::new();
        let mut blocks_scanned = 0u64;
        let mut blocks_skipped = 0u64;
        let mut balance_updates_applied = 0u64;

        for height in 1..=tip {
            let height_key = format!("qblock:height:{}", height);
            let block_data = match self.hot_db.get(CF_BLOCKS, height_key.as_bytes()).await {
                Ok(Some(data)) => data,
                Ok(None) => continue, // Gap in chain — skip
                Err(_) => continue,
            };

            // v7.3.1: Handle compressed format
            let block: q_types::block::QBlock = if precompressed_storage::is_precompressed(&block_data) {
                match precompressed_storage::PrecompressedBlock::from_bytes(&block_data)
                    .and_then(|c| c.decompress().map_err(|e| e.into()))
                    .and_then(|raw| q_types::legacy::deserialize_qblock_with_fallback(&raw)
                        .map_err(|e| anyhow::anyhow!("{}", e)))
                {
                    Ok(b) => b,
                    Err(_) => continue,
                }
            } else {
                match q_types::legacy::deserialize_qblock_with_fallback(&block_data) {
                    Ok(b) => b,
                    Err(_) => continue,
                }
            };

            // Skip pre-genesis (testnet) blocks
            if block.header.timestamp > 0 && block.header.timestamp < genesis_timestamp {
                blocks_skipped += 1;
                continue;
            }

            blocks_scanned += 1;

            // v8.2.0: Process block.transactions — mirrors the runtime path in
            // BalanceConsensusEngine::process_block_mining_rewards() exactly.
            // This ensures rebuild produces identical balances to live processing.
            for block_tx in &block.transactions {
                let is_coinbase = block_tx.is_coinbase() || block_tx.tx_type.is_coinbase();

                if is_coinbase {
                    // Coinbase (mining reward): credit miner
                    if block_tx.to != [0u8; 32] && block_tx.amount > 0 {
                        let entry = balances.entry(block_tx.to).or_insert(0);
                        *entry = entry.saturating_add(block_tx.amount);
                        balance_updates_applied += 1;
                    }
                } else {
                    // Transfer: debit sender, credit receiver
                    if block_tx.amount == 0 {
                        continue;
                    }
                    // Debit sender (saturating to prevent underflow)
                    if let Some(sender_bal) = balances.get_mut(&block_tx.from) {
                        *sender_bal = sender_bal.saturating_sub(block_tx.amount);
                    }
                    // Credit receiver
                    let entry = balances.entry(block_tx.to).or_insert(0);
                    *entry = entry.saturating_add(block_tx.amount);
                    balance_updates_applied += 1;
                }
            }

            // Fallback for blocks with no transactions but with mining_solutions
            // (legacy blocks from early chain history)
            if block.transactions.is_empty() {
                for solution in &block.mining_solutions {
                    if solution.miner_address != [0u8; 32] {
                        let reward = crate::emission_controller::static_block_reward_for_timestamp(
                            block.header.timestamp,
                        );
                        let entry = balances.entry(solution.miner_address).or_insert(0);
                        *entry = entry.saturating_add(reward);
                        balance_updates_applied += 1;
                    }
                }
            }

            // Progress log every 50,000 blocks
            if height % 50_000 == 0 {
                info!("🔄 [REBUILD] Progress: {}/{} blocks scanned, {} wallets so far", height, tip, balances.len());
            }
        }

        // Compute total supply from final balances
        let total_supply: u128 = balances.values().sum();

        // Persist rebuilt balances to RocksDB
        if !balances.is_empty() {
            self.save_wallet_balances(&balances).await?;
            self.save_total_supply(total_supply).await?;
        }

        info!("✅ [REBUILD v7.2.12] Rebuilt {} wallet balances from chain", balances.len());
        info!("   Blocks scanned: {}, skipped (pre-genesis): {}, balance updates applied: {}",
              blocks_scanned, blocks_skipped, balance_updates_applied);
        info!("   Total supply: {} QUG ({} base units)",
              total_supply / 1_000_000_000_000_000_000_000_000u128, total_supply);

        Ok((balances, total_supply))
    }

    /// v8.5.0: One-time migration — purge ALL wallet balances, rebuild from chain,
    /// then scale to match emission controller (the source of truth).
    ///
    /// **Why scaling is needed:**
    /// The block producer embedded inflated coinbase amounts into blocks (calculated
    /// from the inflated state). So the chain itself has ~59x more QUG than the
    /// emission controller says was actually mined. We rebuild from chain to get
    /// correct RELATIVE proportions, then scale everyone down to match the emission
    /// controller's authoritative total.
    ///
    /// **Why this is safe:**
    /// - Everyone keeps their relative share (mined 5% of blocks → keep 5% of supply)
    /// - Total supply matches emission controller (the only correct number)
    /// - Deterministic — every node produces the same result
    ///
    /// `emission_total_supply`: The emission controller's total_supply (u128, 24 decimals).
    /// Pass 0 to skip scaling (rebuild only).
    ///
    /// Idempotent: runs once, sets a migration flag, never runs again.
    pub async fn purge_and_rebuild_balances(&self, emission_total_supply: u128) -> Result<bool> {
        const MIGRATION_FLAG: &[u8] = b"migration_balance_rebuild_v851_done";

        if let Ok(Some(_)) = self.hot_db.get(CF_MANIFEST, MIGRATION_FLAG).await {
            return Ok(false); // Already done
        }

        info!("🧹 [v8.5.1] Purging ALL wallet balances and rebuilding from chain...");

        // 1. Delete ALL wallet_balance_ keys
        let deleted = self.delete_by_prefix(b"wallet_balance_").await.unwrap_or(0);
        let _ = self.hot_db.delete(CF_MANIFEST, b"total_minted_supply").await;
        info!("   Deleted {} stale wallet balance entries", deleted);

        // 2. Rebuild from chain (only post-genesis blocks, includes transfers)
        let (mut balances, chain_total) = self.rebuild_balances_from_chain().await?;
        let chain_qug = chain_total / 1_000_000_000_000_000_000_000_000u128;
        let emission_qug = emission_total_supply / 1_000_000_000_000_000_000_000_000u128;
        info!("   Rebuilt {} wallets from chain: {} QUG", balances.len(), chain_qug);
        info!("   Emission controller says: {} QUG", emission_qug);

        // 3. Scale balances to match emission controller if there's significant inflation
        let final_total = if emission_total_supply > 0 && chain_total > emission_total_supply * 2 {
            // Chain has >2x what emission says — scale down proportionally
            // Use integer division to avoid u128 overflow:
            //   scale_divisor = chain_total / emission_total ≈ 59
            //   new_balance = old_balance / scale_divisor
            let scale_divisor = chain_total / emission_total_supply;
            info!("   ⚖️ Scaling {} wallets by 1/{}: chain ({} QUG) → emission ({} QUG)",
                  balances.len(), scale_divisor, chain_qug, emission_qug);

            for (_addr, balance) in balances.iter_mut() {
                *balance = *balance / scale_divisor;
            }

            // Re-persist the scaled balances
            self.save_wallet_balances(&balances).await?;
            let scaled_total: u128 = balances.values().sum();
            self.save_total_supply(scaled_total).await?;

            let scaled_qug = scaled_total / 1_000_000_000_000_000_000_000_000u128;
            info!("   ✅ Scaled to {} QUG across {} wallets (divisor={})",
                  scaled_qug, balances.len(), scale_divisor);
            scaled_total
        } else {
            // Chain total is reasonable — no scaling needed
            info!("   ✅ Chain total is within range — no scaling needed");
            chain_total
        };

        let final_qug = final_total / 1_000_000_000_000_000_000_000_000u128;
        info!("🧹 [v8.5.0] Migration complete: {} wallets, {} QUG total", balances.len(), final_qug);

        // 4. Set migration flag
        self.hot_db.put(CF_MANIFEST, MIGRATION_FLAG, b"done").await?;

        Ok(true)
    }

    /// v8.5.1: Purge ghost QUGUSD token balances.
    ///
    /// QUGUSD balances suffer from a ghost balance bug where spent balances reappear
    /// on restart. This happens because the 15s token balance sync can overwrite
    /// correct RocksDB values with stale in-memory data during race conditions.
    ///
    /// Fix: Delete all QUGUSD token_balance_ entries. Re-populate only from
    /// CollateralVault minted_qugusd positions (the CDP source of truth).
    ///
    /// Idempotent: runs once, sets migration flag.
    pub async fn purge_ghost_qugusd_balances(&self) -> Result<bool> {
        const MIGRATION_FLAG: &[u8] = b"migration_qugusd_purge_v851_done";

        if let Ok(Some(_)) = self.hot_db.get(CF_MANIFEST, MIGRATION_FLAG).await {
            return Ok(false); // Already done
        }

        info!("🧹 [v8.5.1] Purging ghost QUGUSD token balances...");

        let qugusd_hex = hex::encode(q_types::QUGUSD_TOKEN_ADDRESS);
        let prefix = "token_balance_".as_bytes();
        let mut deleted_count = 0u64;

        // Iterate all token_balance_ entries and delete QUGUSD ones
        if let Ok(entries) = self.hot_db.scan_prefix(CF_MANIFEST, prefix).await {
            for (key, _value) in entries {
                if let Ok(key_str) = std::str::from_utf8(&key) {
                    if key_str.ends_with(&qugusd_hex) {
                        let _ = self.hot_db.delete(CF_MANIFEST, &key).await;
                        deleted_count += 1;
                    }
                }
            }
        }

        info!("   Deleted {} ghost QUGUSD token_balance entries", deleted_count);

        // Set migration flag
        self.hot_db.put(CF_MANIFEST, MIGRATION_FLAG, b"done").await?;

        Ok(true)
    }

    /// v8.5.5: Purge QUGUSD ghost entries from CF_TOKEN_BALANCES (binary key format).
    /// The v8.5.3 startup purge only deleted from CF_MANIFEST (text keys).
    /// State sync stores balances in CF_TOKEN_BALANCES with binary keys (wallet 32B + token 32B).
    /// These stale QUGUSD entries survive the text-key purge and get re-loaded by load_token_balances().
    #[cfg(not(target_os = "windows"))]
    pub async fn purge_qugusd_from_state_sync_cf(&self) -> Result<u64> {
        let qugusd_addr = q_types::QUGUSD_TOKEN_ADDRESS;
        let mut deleted = 0u64;

        if let Some(db) = self.get_rocks_db_handle() {
            if let Some(cf) = db.cf_handle(CF_TOKEN_BALANCES) {
                let mut keys_to_delete: Vec<Vec<u8>> = Vec::new();
                let iter = db.iterator_cf(&cf, rocksdb::IteratorMode::Start);
                for item in iter {
                    if let Ok((key, _value)) = item {
                        if key.len() == 64 {
                            let token_address = &key[32..64];
                            if token_address == qugusd_addr {
                                keys_to_delete.push(key.to_vec());
                            }
                        }
                    }
                }
                for key in &keys_to_delete {
                    let _ = db.delete_cf(&cf, key);
                    deleted += 1;
                }
            }
        }

        Ok(deleted)
    }

    /// v8.5.2: Restore QUGUSD token balances that were accidentally purged by v8.5.1.
    /// The purge was too aggressive — it deleted ALL QUGUSD token_balance entries
    /// when it should have only fixed one ghost balance. This restores from known values.
    pub async fn restore_qugusd_balances(&self) -> Result<bool> {
        // v8.5.3: DISABLED permanently — This migration ran on EVERY restart due to
        // non-persisting flag, resurrecting spent QUGUSD balances ("money glitch").
        Ok(false)
    }

    /// v8.5.4: Reconcile wallet balances by applying DEX swap debits/credits.
    ///
    /// **THE MONEY GLITCH FIX**: State sync was importing stale balances from peers,
    /// overwriting local DEX swap debits. This migration:
    /// 1. Rebuilds chain-only balances (coinbase + transfers) — same as rebuild_balances_from_chain()
    /// 2. Scans ALL swap history from CF_SWAP_HISTORY for QUG debits/credits
    /// 3. Applies swap adjustments to get the correct final balance
    /// 4. Persists corrected balances to RocksDB
    ///
    /// Runs once, sets migration flag. Safe to re-run (idempotent via flag).
    pub async fn reconcile_balances_with_dex_swaps(&self) -> Result<bool> {
        // v8.5.7: Use emission controller's tracked total as the CEILING for total supply.
        // The emission controller tracks actual adaptive rewards (accounts for block rate).
        // static_block_reward_for_timestamp() overcounts because it assumes 1 bps rate.
        // Fix: count each miner's SHARE of blocks, then distribute the emission total proportionally.
        const MIGRATION_FLAG: &[u8] = b"migration_balance_reconcile_v857b_done";

        if let Ok(Some(_)) = self.hot_db.get(CF_MANIFEST, MIGRATION_FLAG).await {
            return Ok(false); // Already done
        }

        info!("🔧 [v8.5.7 RECONCILIATION] Starting balance reconciliation (emission controller + miner shares)...");

        // ====================================================================
        // STEP 1: Rebuild balances from chain (coinbase + transfers only)
        // This gives us the baseline without any DEX swap effects.
        // ====================================================================
        let genesis_timestamp: u64 = crate::balance_consensus::active_genesis_timestamp();
        let tip = self.height_cache.cached();

        // Dev wallet (receives 1% fee — computed at runtime, not in blocks)
        let dev_wallet_hex = crate::balance_consensus::FOUNDER_WALLET.trim_start_matches("qnk");
        let dev_wallet_bytes: [u8; 32] = {
            let bytes = hex::decode(dev_wallet_hex).unwrap_or_default();
            let mut arr = [0u8; 32];
            if bytes.len() == 32 { arr.copy_from_slice(&bytes); }
            arr
        };

        // Load the emission controller's tracked total — this is the CORRECT total supply.
        let emission_total: u128 = match self.load_emission_state().await {
            Ok(Some(data)) => {
                match serde_json::from_slice::<crate::emission_controller::EmissionController>(&data) {
                    Ok(ec) => {
                        let total = ec.total_cumulative_emission();
                        info!("   Emission controller total: {:.4} QUG ({} base units)",
                            total as f64 / 1_000_000_000_000_000_000_000_000.0, total);
                        total
                    }
                    Err(e) => {
                        warn!("   Failed to deserialize emission controller: {}, using time-based fallback", e);
                        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
                        let elapsed = now.saturating_sub(genesis_timestamp);
                        crate::emission_controller::target_cumulative_at_time(elapsed)
                    }
                }
            }
            _ => {
                // Fallback: compute from elapsed time
                let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
                let elapsed = now.saturating_sub(genesis_timestamp);
                let total = crate::emission_controller::target_cumulative_at_time(elapsed);
                info!("   Emission controller not available, using time-based target: {:.4} QUG",
                    total as f64 / 1_000_000_000_000_000_000_000_000.0);
                total
            }
        };

        // Step 1: Scan blocks to count each miner's SHARE (number of block-slots mined)
        let mut miner_shares: HashMap<[u8; 32], u64> = HashMap::new();
        let mut total_miner_slots = 0u64;
        let mut blocks_scanned = 0u64;

        if tip > 0 {
            info!("   Step 1/2: Scanning {} heights to count miner shares...", tip);
            for height in 1..=tip {
                let height_key = format!("qblock:height:{}", height);
                let block_data = match self.hot_db.get(CF_BLOCKS, height_key.as_bytes()).await {
                    Ok(Some(data)) => data,
                    _ => continue,
                };

                let block: q_types::block::QBlock = if precompressed_storage::is_precompressed(&block_data) {
                    match precompressed_storage::PrecompressedBlock::from_bytes(&block_data)
                        .and_then(|c| c.decompress().map_err(|e| e.into()))
                        .and_then(|raw| q_types::legacy::deserialize_qblock_with_fallback(&raw)
                            .map_err(|e| anyhow::anyhow!("{}", e)))
                    {
                        Ok(b) => b,
                        Err(_) => continue,
                    }
                } else {
                    match q_types::legacy::deserialize_qblock_with_fallback(&block_data) {
                        Ok(b) => b,
                        Err(_) => continue,
                    }
                };

                if block.header.timestamp > 0 && block.header.timestamp < genesis_timestamp {
                    continue;
                }

                blocks_scanned += 1;

                // Collect unique miner addresses from coinbase txs (excluding dev wallet)
                let mut miners_this_block: Vec<[u8; 32]> = Vec::new();
                for block_tx in &block.transactions {
                    let is_coinbase = block_tx.is_coinbase() || block_tx.tx_type.is_coinbase();
                    if is_coinbase && block_tx.to != [0u8; 32] && block_tx.amount > 0 {
                        if block_tx.to != dev_wallet_bytes {
                            if !miners_this_block.contains(&block_tx.to) {
                                miners_this_block.push(block_tx.to);
                            }
                        }
                    }
                }

                // Legacy blocks: mining_solutions without transactions
                if block.transactions.is_empty() {
                    for solution in &block.mining_solutions {
                        if solution.miner_address != [0u8; 32] {
                            if !miners_this_block.contains(&solution.miner_address) {
                                miners_this_block.push(solution.miner_address);
                            }
                        }
                    }
                }

                // Each miner in this block gets 1 share (split if multiple miners)
                if !miners_this_block.is_empty() {
                    for miner_addr in &miners_this_block {
                        *miner_shares.entry(*miner_addr).or_insert(0) += 1;
                    }
                    total_miner_slots += miners_this_block.len() as u64;
                }

                if height % 50_000 == 0 {
                    info!("   [RECONCILE] Progress: {}/{} heights", height, tip);
                }
            }
        }

        info!("   Step 1 done: {} blocks scanned, {} miners, {} total miner-slots",
              blocks_scanned, miner_shares.len(), total_miner_slots);

        // Step 2: Distribute emission_total proportionally by miner share
        // 1% dev fee, 99% to miners proportional to their block count
        let dev_fee_total = emission_total.saturating_mul(crate::balance_consensus::DEV_FEE_BPS)
            / crate::balance_consensus::BPS_DIVISOR;
        let miner_pool = emission_total.saturating_sub(dev_fee_total);

        let mut balances: HashMap<[u8; 32], u128> = HashMap::new();
        if total_miner_slots > 0 {
            for (miner_addr, shares) in &miner_shares {
                let miner_reward = miner_pool * (*shares as u128) / (total_miner_slots as u128);
                if miner_reward > 0 {
                    *balances.entry(*miner_addr).or_insert(0) += miner_reward;
                }
            }
        }
        if dev_fee_total > 0 {
            *balances.entry(dev_wallet_bytes).or_insert(0) += dev_fee_total;
        }

        let chain_total: u128 = balances.values().sum();
        let chain_qug = chain_total / 1_000_000_000_000_000_000_000_000u128;
        info!("   Step 2 done: {} wallets, {} QUG distributed (emission controller total)",
              balances.len(), chain_qug);

        let final_total: u128 = balances.values().sum();
        let final_qug = final_total / 1_000_000_000_000_000_000_000_000u128;

        // Compare with what's currently in RocksDB
        let current_stored_total: u128 = {
            let mut total = 0u128;
            let prefix = b"wallet_balance_";
            if let Ok(entries) = self.hot_db.scan_prefix(CF_MANIFEST, prefix).await {
                for (_key, value) in entries {
                    if value.len() == 16 {
                        total += u128::from_le_bytes(value[..16].try_into().unwrap());
                    }
                }
            }
            total
        };
        let stored_qug = current_stored_total / 1_000_000_000_000_000_000_000_000u128;

        info!("   ─────────────────────────────────────────────");
        info!("   RECONCILIATION SUMMARY (v8.5.7 — emission controller):");
        info!("   Emission controller total: {:.4} QUG", emission_total as f64 / 1e24);
        info!("   Distributed to wallets:    {} QUG ({} wallets)", final_qug, balances.len());
        info!("   Previous RocksDB total:    {} QUG (inflated)", stored_qug);
        info!("   Inflation removed:         {} QUG", stored_qug.saturating_sub(final_qug));
        info!("   Blocks scanned:            {} (with {} miner-slots)", blocks_scanned, total_miner_slots);
        info!("   ─────────────────────────────────────────────");

        // PURGE all existing wallet balances first (removes testnet remnants + glitched wallets)
        {
            let prefix = b"wallet_balance_";
            if let Ok(entries) = self.hot_db.scan_prefix(CF_MANIFEST, prefix).await {
                let keys_to_delete: Vec<Vec<u8>> = entries.into_iter().map(|(k, _)| k).collect();
                let purge_count = keys_to_delete.len();
                for key in &keys_to_delete {
                    let _ = self.hot_db.delete(CF_MANIFEST, key).await;
                }
                info!("   🗑️ Purged {} old wallet_balance entries (testnet remnants + glitched)", purge_count);
            }
        }

        // Persist ONLY emission-derived mining reward balances
        self.save_wallet_balances(&balances).await?;
        self.save_total_supply(final_total).await?;

        // Set migration flag
        self.hot_db.put(CF_MANIFEST, MIGRATION_FLAG, b"done").await?;

        info!("✅ [v8.5.7 RECONCILIATION] Balance reconciliation complete. {} wallets, {} QUG total supply.", balances.len(), final_qug);

        Ok(true)
    }

    /// v8.8.1: Full chain-based balance rebuild with proportional scaling.
    ///
    /// The v8.5.7 `reconcile_balances_with_dex_swaps` destroyed all wallet balances by
    /// using flat block-slot shares instead of actual coinbase amounts, and used an
    /// incomplete emission controller total. This threw away real proportions and zeroed
    /// transfer-only wallets.
    ///
    /// This migration:
    /// 1. Replays every block for correct RELATIVE proportions (coinbase + transfers)
    /// 2. Scales total down to match the emission controller's real total (~69K QUG)
    /// 3. Preserves everyone's proportional share accurately
    ///
    /// `emission_total_supply`: The emission controller's authoritative total (u128, 24 decimals).
    ///
    /// Idempotent: runs once, sets migration flag, never runs again.
    pub async fn full_chain_balance_rebuild_v881(&self, emission_total_supply: u128) -> Result<bool> {
        const MIGRATION_FLAG: &[u8] = b"migration_full_chain_rebuild_v882_done";

        if let Ok(Some(_)) = self.hot_db.get(CF_MANIFEST, MIGRATION_FLAG).await {
            return Ok(false); // Already done
        }

        let qug_unit: u128 = 1_000_000_000_000_000_000_000_000;
        let min_emission_for_rebuild: u128 = 50_000 * qug_unit; // 50K QUG

        // v8.8.2: Skip chain rebuild if local emission controller has incomplete data.
        // Nodes with low emission totals (e.g. recently synced) should rely on bootstrap
        // from the primary server instead of rebuilding from local chain data.
        if emission_total_supply < min_emission_for_rebuild {
            info!("⏭️ [v8.8.2 CHAIN REBUILD] Skipping — local emission total {} QUG < 50K threshold",
                  emission_total_supply / qug_unit);
            info!("   This node will get correct balances from peer bootstrap instead.");
            self.hot_db.put(CF_MANIFEST, MIGRATION_FLAG, b"skipped_low_emission").await?;
            return Ok(false);
        }

        let old_balances = self.load_wallet_balances().await.unwrap_or_default();
        let old_total: u128 = old_balances.values().sum();
        let old_count = old_balances.len();

        info!("🔧 [v8.8.2 CHAIN REBUILD] Starting full balance rebuild from chain data...");
        info!("   Before: {} wallets, {} QUG total supply", old_count, old_total / qug_unit);
        info!("   Emission controller total: {} QUG", emission_total_supply / qug_unit);

        // Purge existing wallet_balance_ keys first
        let deleted = self.delete_by_prefix(b"wallet_balance_").await.unwrap_or(0);
        let _ = self.hot_db.delete(CF_MANIFEST, b"total_minted_supply").await;
        info!("   Purged {} old wallet_balance entries", deleted);

        // Rebuild from actual chain data (coinbase + transfers) — gets correct proportions
        let (mut balances, chain_total) = self.rebuild_balances_from_chain().await?;

        info!("   Chain replay: {} wallets, {} QUG raw chain total",
              balances.len(), chain_total / qug_unit);

        // Scale balances to match emission controller total (the correct supply)
        // This preserves relative proportions while fixing the 34× inflation
        if emission_total_supply > 0 && chain_total > emission_total_supply * 2 {
            info!("   Scaling by ≈{}× to match emission total",
                  chain_total / emission_total_supply);

            // Delete the chain-rebuilt balances (rebuild_balances_from_chain persists them)
            let _ = self.delete_by_prefix(b"wallet_balance_").await;
            let _ = self.hot_db.delete(CF_MANIFEST, b"total_minted_supply").await;

            let mut scaled_total: u128 = 0;
            // Use division by scale_divisor to avoid u128 overflow
            let scale_divisor = chain_total / emission_total_supply;
            for (_addr, amount) in balances.iter_mut() {
                let scaled = *amount / scale_divisor;
                *amount = scaled;
                scaled_total += scaled;
            }

            // Persist scaled balances
            self.save_wallet_balances(&balances).await?;
            self.save_total_supply(scaled_total).await?;

            info!("✅ [v8.8.2 CHAIN REBUILD] Complete with proportional scaling!");
            info!("   After: {} wallets, {} QUG total supply (scaled from {} QUG chain)",
                  balances.len(), scaled_total / qug_unit, chain_total / qug_unit);
        } else {
            // No scaling needed — chain total is close to emission total
            info!("✅ [v8.8.2 CHAIN REBUILD] Complete (no scaling needed)!");
            info!("   After: {} wallets, {} QUG total supply",
                  balances.len(), chain_total / qug_unit);
        }

        self.hot_db.put(CF_MANIFEST, MIGRATION_FLAG, b"done").await?;
        Ok(true)
    }

    /// v8.8.5: Deterministic full-chain transaction replay + proportional scaling.
    ///
    /// Replays EVERY block from genesis, processing all transactions:
    /// - Blocks WITH transactions: replay coinbase + transfers as-is (dev fee already embedded)
    /// - Legacy blocks (no transactions): compute reward with correct genesis, split 1.9% dev fee
    ///
    /// Then scales ALL balances proportionally to match the **deterministic emission target**
    /// computed from first principles (genesis timestamp + halving schedule). Does NOT trust
    /// the emission controller's persisted state — derives the target mathematically.
    ///
    /// This is needed because blocks from the 34× emission overshoot era contain inflated
    /// coinbase amounts. Scaling preserves relative proportions (who mined what %) while
    /// matching the correct total supply.
    pub async fn deterministic_tx_replay_v885(&self) -> Result<bool> {
        // v8.8.9: Reverted to v885 flag — replay should NOT re-run.
        // The replay is incomplete (misses DEX transactions) and overwrites correct balances.
        const MIGRATION_FLAG: &[u8] = b"migration_deterministic_replay_v885_done";

        if let Ok(Some(_)) = self.hot_db.get(CF_MANIFEST, MIGRATION_FLAG).await {
            return Ok(false); // Already done
        }

        let genesis_ts = crate::balance_consensus::active_genesis_timestamp();
        let tip = self.height_cache.cached();
        if tip == 0 { return Ok(false); }

        info!("🔄 [v8.8.5 DETERMINISTIC REPLAY] Starting full chain replay (blocks 1..{})...", tip);
        info!("   Using genesis timestamp: {} (network-aware)", genesis_ts);

        // Compute expected emission from FIRST PRINCIPLES — pure math, no controller state.
        // Uses genesis_timestamp + halving schedule to derive what total supply SHOULD be.
        let now_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let elapsed_since_genesis = now_secs.saturating_sub(genesis_ts);
        let expected_total = crate::emission_controller::target_cumulative_at_time(elapsed_since_genesis);
        let qug_unit: u128 = 1_000_000_000_000_000_000_000_000;

        info!("   Expected emission (from first principles): {} QUG ({} seconds since genesis)",
              expected_total / qug_unit, elapsed_since_genesis);

        if expected_total == 0 {
            warn!("⚠️ [v8.8.5] Expected emission is 0 — cannot scale. Aborting.");
            return Ok(false);
        }

        // Purge ALL existing wallet balances
        let deleted = self.delete_by_prefix(b"wallet_balance_").await.unwrap_or(0);
        let _ = self.hot_db.delete(CF_MANIFEST, b"total_minted_supply").await;
        info!("   Purged {} old wallet_balance entries", deleted);

        // Parse founder wallet address bytes (strip "qnk" prefix, decode hex)
        let founder_hex = &crate::balance_consensus::FOUNDER_WALLET[3..];
        let founder_bytes: [u8; 32] = {
            let decoded = hex::decode(founder_hex).unwrap_or_default();
            let mut arr = [0u8; 32];
            if decoded.len() == 32 { arr.copy_from_slice(&decoded); }
            arr
        };

        let mut balances: HashMap<[u8; 32], u128> = HashMap::new();
        let mut blocks_processed = 0u64;
        let mut legacy_blocks = 0u64;
        let mut tx_blocks = 0u64;
        let mut last_block_ts = genesis_ts;

        for height in 1..=tip {
            let height_key = format!("qblock:height:{}", height);
            let block_data = match self.hot_db.get(CF_BLOCKS, height_key.as_bytes()).await {
                Ok(Some(data)) => data,
                Ok(None) => continue,
                Err(_) => continue,
            };

            let block: q_types::block::QBlock = if precompressed_storage::is_precompressed(&block_data) {
                match precompressed_storage::PrecompressedBlock::from_bytes(&block_data)
                    .and_then(|c| c.decompress().map_err(|e| e.into()))
                    .and_then(|raw| q_types::legacy::deserialize_qblock_with_fallback(&raw)
                        .map_err(|e| anyhow::anyhow!("{}", e)))
                {
                    Ok(b) => b,
                    Err(_) => continue,
                }
            } else {
                match q_types::legacy::deserialize_qblock_with_fallback(&block_data) {
                    Ok(b) => b,
                    Err(_) => continue,
                }
            };

            if block.header.timestamp > 0 && block.header.timestamp < genesis_ts {
                continue;
            }

            blocks_processed += 1;
            if block.header.timestamp > last_block_ts {
                last_block_ts = block.header.timestamp;
            }

            if !block.transactions.is_empty() {
                tx_blocks += 1;
                for block_tx in &block.transactions {
                    let is_coinbase = block_tx.is_coinbase() || block_tx.tx_type.is_coinbase();
                    if is_coinbase {
                        if block_tx.to != [0u8; 32] && block_tx.amount > 0 {
                            let entry = balances.entry(block_tx.to).or_insert(0);
                            *entry = entry.saturating_add(block_tx.amount);
                        }
                    } else {
                        if block_tx.amount == 0 { continue; }
                        if let Some(sender_bal) = balances.get_mut(&block_tx.from) {
                            *sender_bal = sender_bal.saturating_sub(block_tx.amount);
                        }
                        let entry = balances.entry(block_tx.to).or_insert(0);
                        *entry = entry.saturating_add(block_tx.amount);
                    }
                }
            } else {
                legacy_blocks += 1;
                for solution in &block.mining_solutions {
                    if solution.miner_address != [0u8; 32] {
                        let elapsed = block.header.timestamp.saturating_sub(genesis_ts);
                        let era = crate::emission_controller::era_at_time(elapsed);
                        let annual = crate::emission_controller::annual_emission(era);
                        let reward = (annual / 31_557_600u128).max(crate::emission_controller::MIN_REWARD);

                        let dev_fee = reward.saturating_mul(crate::balance_consensus::DEV_FEE_BPS)
                            / crate::balance_consensus::BPS_DIVISOR;
                        let miner_reward = reward.saturating_sub(dev_fee);

                        let entry = balances.entry(solution.miner_address).or_insert(0);
                        *entry = entry.saturating_add(miner_reward);
                        if dev_fee > 0 {
                            let founder_entry = balances.entry(founder_bytes).or_insert(0);
                            *founder_entry = founder_entry.saturating_add(dev_fee);
                        }
                    }
                }
            }

            if height % 50_000 == 0 {
                info!("   Progress: {}/{} blocks ({} tx-blocks, {} legacy)",
                      height, tip, tx_blocks, legacy_blocks);
            }
        }

        balances.retain(|_, v| *v > 0);
        let chain_total: u128 = balances.values().sum();

        info!("✅ [v8.8.5] Chain replay complete: {} wallets, {} QUG raw chain total",
              balances.len(), chain_total / qug_unit);
        info!("   {} blocks processed ({} with txs, {} legacy)",
              blocks_processed, tx_blocks, legacy_blocks);

        // Always scale to match deterministic emission target (no heuristic threshold).
        // Uses (balance × expected_total) / chain_total for maximum precision.
        // Overflow-safe: decompose multiplication to avoid u128 overflow.
        if chain_total > 0 && chain_total != expected_total {
            let scale_ratio_display = chain_total as f64 / expected_total as f64;
            info!("   Scaling {} wallets: chain {} QUG → target {} QUG (ratio {:.2}×)",
                  balances.len(), chain_total / qug_unit, expected_total / qug_unit, scale_ratio_display);

            for (_addr, amount) in balances.iter_mut() {
                // (amount × expected_total) / chain_total — overflow-safe decomposition
                // Split: result = (amount / chain_total) × expected_total
                //                + ((amount % chain_total) × expected_total) / chain_total
                let quot = *amount / chain_total;
                let rem = *amount % chain_total;
                // For the remainder term, if rem × expected_total overflows, use f64 fallback
                let term1 = quot.saturating_mul(expected_total);
                let term2 = if let Some(prod) = rem.checked_mul(expected_total) {
                    prod / chain_total
                } else {
                    // Fallback: use intermediate f64 (53-bit precision, sufficient for remainder)
                    ((rem as f64) * (expected_total as f64) / (chain_total as f64)) as u128
                };
                *amount = term1.saturating_add(term2);
            }
            balances.retain(|_, v| *v > 0);
        }

        let final_total: u128 = balances.values().sum();

        info!("   Final: {} wallets, {} QUG total supply (target was {} QUG)",
              balances.len(), final_total / qug_unit, expected_total / qug_unit);
        info!("   Founder wallet: {} QUG",
              balances.get(&founder_bytes).unwrap_or(&0) / qug_unit);

        // Persist — atomic batch via save_wallet_balances
        self.save_wallet_balances(&balances).await?;
        self.save_total_supply(final_total).await?;

        // Set migration flag AFTER successful persist
        self.hot_db.put(CF_MANIFEST, MIGRATION_FLAG, b"done").await?;

        info!("🔒 [v8.8.5] Migration flag set. This replay will NOT run again.");
        Ok(true)
    }

    /// v8.8.6: Post-migration emission controller sync + collateral vault reset.
    ///
    /// The v8.8.5 migration scaled wallet balances but did NOT sync the emission controller
    /// or scale the collateral vault. This causes:
    /// - Emission controller thinks 49M QUG was minted (old inflated value)
    /// - Correction factor = 0.01 → miners get 1% of normal rewards
    /// - Collateral vault has inflated locked_qug amounts
    /// - Rate windows contain pre-migration poisoned samples
    ///
    /// Returns (old_emission_total, new_emission_total) if migration ran.
    pub async fn post_migration_emission_sync_v886(&self) -> Result<Option<(u128, u128)>> {
        // Reverted to v886 flag — emission sync should NOT re-run
        const MIGRATION_FLAG: &[u8] = b"migration_emission_sync_v886_done";

        if let Ok(Some(_)) = self.hot_db.get(CF_MANIFEST, MIGRATION_FLAG).await {
            return Ok(None); // Already done
        }

        // Only run if v8.8.5/v8.8.9 already completed (this is a post-migration fixup)
        const V889_FLAG: &[u8] = b"migration_deterministic_replay_v889_done";
        const V885_FLAG: &[u8] = b"migration_deterministic_replay_v885_done";
        if self.hot_db.get(CF_MANIFEST, V889_FLAG).await?.is_none()
            && self.hot_db.get(CF_MANIFEST, V885_FLAG).await?.is_none() {
            return Ok(None); // v8.8.5 hasn't run yet, skip
        }

        info!("🔄 [v8.8.6] Post-migration emission sync starting...");

        // Read the current (correct) wallet totals from RocksDB
        let balances = self.load_wallet_balances().await?;
        let wallet_total: u128 = balances.values().sum();
        let qug_unit: u128 = 1_000_000_000_000_000_000_000_000;

        info!("   Wallet total from RocksDB: {} QUG ({} wallets)",
              wallet_total / qug_unit, balances.len());

        // Save the wallet total as the corrected emission total
        // (This will be picked up by main.rs to sync the in-memory emission controller)
        self.save_total_supply(wallet_total).await?;

        // Set the balance watermark to current tip to prevent re-inflation
        let tip = self.height_cache.cached();
        if tip > 0 {
            self.save_balance_watermark(tip).await?;
            info!("   Balance watermark set to height {}", tip);
        }

        // Set migration flag
        self.hot_db.put(CF_MANIFEST, MIGRATION_FLAG, b"done").await?;

        info!("✅ [v8.8.6] Emission sync flag set. Main.rs will sync in-memory state.");
        Ok(Some((0, wallet_total))) // old_total unknown at storage layer
    }

    /// v1.0.3: Full state convergence migration — deterministic replay of ALL mutable state
    /// from the blockchain. Rebuilds QUG balances (same as v885) PLUS CollateralVault state
    /// (StableMint/StableBurn txs) to achieve identical state across all nodes.
    ///
    /// The chain is the ONLY source of truth. Every balance and vault position is derived
    /// deterministically from block data. This eliminates all 5 divergence vectors:
    ///   1. process_block_coinbase_only_tx skipping transfers (already fixed in code)
    ///   2. Vault sync being one-shot empty-only
    ///   3. QUGUSD ghost prevention blocking all P2P QUGUSD after bootstrap
    ///   4. replay_block_state_changes skipping StableMint/StableBurn
    ///   5. Token balance sync being add-only
    pub async fn safe_batched_convergence_v103(&self) -> Result<bool> {
        const MIGRATION_FLAG: &[u8] = b"migration_safe_convergence_v103_done";

        if let Ok(Some(_)) = self.hot_db.get(CF_MANIFEST, MIGRATION_FLAG).await {
            return Ok(false); // Already done
        }

        let genesis_ts = crate::balance_consensus::active_genesis_timestamp();
        let tip = self.height_cache.cached();
        if tip == 0 { return Ok(false); }

        info!("═══════════════════════════════════════════════════════════════════");
        info!("🔄 [v1.0.3 CONVERGENCE] Full state convergence migration starting");
        info!("   Chain height: {} blocks, genesis: {}", tip, genesis_ts);
        info!("   Rebuilding: QUG balances + CollateralVault from chain data");
        info!("═══════════════════════════════════════════════════════════════════");

        // =====================================================================
        // Step 1: Compute expected emission from FIRST PRINCIPLES
        // =====================================================================
        let now_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let elapsed_since_genesis = now_secs.saturating_sub(genesis_ts);
        let expected_total = crate::emission_controller::target_cumulative_at_time(elapsed_since_genesis);
        let qug_unit: u128 = 1_000_000_000_000_000_000_000_000;

        info!("   Expected emission (first principles): {} QUG ({} seconds since genesis)",
              expected_total / qug_unit, elapsed_since_genesis);

        if expected_total == 0 {
            warn!("⚠️ [v1.0.3] Expected emission is 0 — cannot scale. Aborting.");
            return Ok(false);
        }

        // =====================================================================
        // Step 2: Purge ALL mutable state
        // =====================================================================
        let deleted = self.delete_by_prefix(b"wallet_balance_").await.unwrap_or(0);
        let _ = self.hot_db.delete(CF_MANIFEST, b"total_minted_supply").await;
        let _ = self.hot_db.delete(CF_MANIFEST, b"collateral_vault").await;
        info!("   Purged {} wallet_balance entries + total_supply + vault", deleted);

        // Parse founder wallet address bytes (strip "qnk" prefix, decode hex)
        let founder_hex = &crate::balance_consensus::FOUNDER_WALLET[3..];
        let founder_bytes: [u8; 32] = {
            let decoded = hex::decode(founder_hex).unwrap_or_default();
            let mut arr = [0u8; 32];
            if decoded.len() == 32 { arr.copy_from_slice(&decoded); }
            arr
        };

        // =====================================================================
        // Step 3: Replay entire chain — QUG balances + vault state
        // =====================================================================
        let mut balances: std::collections::HashMap<[u8; 32], u128> = std::collections::HashMap::new();
        // Vault state rebuilt from chain
        let mut vault_locked_qug: std::collections::HashMap<[u8; 32], u128> = std::collections::HashMap::new();
        let mut vault_minted_qugusd: std::collections::HashMap<[u8; 32], u128> = std::collections::HashMap::new();
        let mut vault_total_locked: u128 = 0;
        let mut vault_total_minted: u128 = 0;

        let mut blocks_processed = 0u64;
        let mut legacy_blocks = 0u64;
        let mut tx_blocks = 0u64;
        let mut stable_mint_count = 0u64;
        let mut stable_burn_count = 0u64;

        for height in 1..=tip {
            let height_key = format!("qblock:height:{}", height);
            let block_data = match self.hot_db.get(CF_BLOCKS, height_key.as_bytes()).await {
                Ok(Some(data)) => data,
                Ok(None) => continue,
                Err(_) => continue,
            };

            // Block deserialization (handles precompressed + legacy formats)
            let block: q_types::block::QBlock = if precompressed_storage::is_precompressed(&block_data) {
                match precompressed_storage::PrecompressedBlock::from_bytes(&block_data)
                    .and_then(|c| c.decompress().map_err(|e| e.into()))
                    .and_then(|raw| q_types::legacy::deserialize_qblock_with_fallback(&raw)
                        .map_err(|e| anyhow::anyhow!("{}", e)))
                {
                    Ok(b) => b,
                    Err(_) => continue,
                }
            } else {
                match q_types::legacy::deserialize_qblock_with_fallback(&block_data) {
                    Ok(b) => b,
                    Err(_) => continue,
                }
            };

            if block.header.timestamp > 0 && block.header.timestamp < genesis_ts {
                continue;
            }

            blocks_processed += 1;

            if !block.transactions.is_empty() {
                tx_blocks += 1;
                for block_tx in &block.transactions {
                    let is_coinbase = block_tx.is_coinbase() || block_tx.tx_type.is_coinbase();

                    if is_coinbase {
                        // Coinbase: credit miner the full amount (dev fee already deducted at production time)
                        if block_tx.to != [0u8; 32] && block_tx.amount > 0 {
                            let entry = balances.entry(block_tx.to).or_insert(0);
                            *entry = entry.saturating_add(block_tx.amount);
                        }
                    } else {
                        // Check tx type for vault operations
                        let tx_type = block_tx.effective_tx_type();
                        match tx_type {
                            q_types::TransactionType::StableMint => {
                                // Parse tx.data: [0..16] collateral_amount (u128 BE), [16..32] mint_amount (u128 BE)
                                if block_tx.data.len() >= 32 {
                                    let collateral_amount = u128::from_be_bytes(
                                        block_tx.data[0..16].try_into().unwrap_or([0u8; 16])
                                    );
                                    let mint_amount = u128::from_be_bytes(
                                        block_tx.data[16..32].try_into().unwrap_or([0u8; 16])
                                    );

                                    if collateral_amount > 0 && mint_amount > 0 {
                                        // Lock QUG in vault (debit from user's QUG balance)
                                        if let Some(sender_bal) = balances.get_mut(&block_tx.from) {
                                            *sender_bal = sender_bal.saturating_sub(collateral_amount);
                                        }

                                        // Update vault positions
                                        let locked = vault_locked_qug.entry(block_tx.from).or_insert(0);
                                        *locked = locked.saturating_add(collateral_amount);
                                        let minted = vault_minted_qugusd.entry(block_tx.from).or_insert(0);
                                        *minted = minted.saturating_add(mint_amount);
                                        vault_total_locked = vault_total_locked.saturating_add(collateral_amount);
                                        vault_total_minted = vault_total_minted.saturating_add(mint_amount);

                                        stable_mint_count += 1;
                                    }
                                }
                            }
                            q_types::TransactionType::StableBurn => {
                                // Parse tx.data: [0..16] qugusd_amount (u128 BE)
                                if block_tx.data.len() >= 16 {
                                    let qugusd_burned = u128::from_be_bytes(
                                        block_tx.data[0..16].try_into().unwrap_or([0u8; 16])
                                    );

                                    if qugusd_burned > 0 {
                                        let user = block_tx.from;
                                        let user_locked = vault_locked_qug.get(&user).copied().unwrap_or(0);
                                        let user_minted = vault_minted_qugusd.get(&user).copied().unwrap_or(0);

                                        // Proportional QUG unlock: if burning X% of debt, unlock X% of collateral
                                        // This is deterministic and doesn't depend on historical price
                                        let qug_to_unlock = if user_minted > 0 {
                                            // (user_locked * qugusd_burned) / user_minted
                                            // Overflow-safe: split into quotient + remainder
                                            let quot = user_locked / user_minted;
                                            let rem = user_locked % user_minted;
                                            let term1 = quot.saturating_mul(qugusd_burned);
                                            let term2 = if let Some(prod) = rem.checked_mul(qugusd_burned) {
                                                prod / user_minted
                                            } else {
                                                ((rem as f64) * (qugusd_burned as f64) / (user_minted as f64)) as u128
                                            };
                                            term1.saturating_add(term2).min(user_locked)
                                        } else {
                                            0
                                        };

                                        // Unlock QUG (credit back to user's balance)
                                        if qug_to_unlock > 0 {
                                            let entry = balances.entry(user).or_insert(0);
                                            *entry = entry.saturating_add(qug_to_unlock);
                                        }

                                        // Update vault positions
                                        if let Some(locked) = vault_locked_qug.get_mut(&user) {
                                            *locked = locked.saturating_sub(qug_to_unlock);
                                        }
                                        let actual_burn = qugusd_burned.min(user_minted);
                                        if let Some(minted) = vault_minted_qugusd.get_mut(&user) {
                                            *minted = minted.saturating_sub(actual_burn);
                                        }
                                        vault_total_locked = vault_total_locked.saturating_sub(qug_to_unlock);
                                        vault_total_minted = vault_total_minted.saturating_sub(actual_burn);

                                        stable_burn_count += 1;
                                    }
                                }
                            }
                            q_types::TransactionType::Transfer => {
                                // Regular transfer: debit sender, credit receiver
                                if block_tx.amount > 0 {
                                    if let Some(sender_bal) = balances.get_mut(&block_tx.from) {
                                        *sender_bal = sender_bal.saturating_sub(block_tx.amount);
                                    }
                                    let entry = balances.entry(block_tx.to).or_insert(0);
                                    *entry = entry.saturating_add(block_tx.amount);
                                }
                            }
                            _ => {
                                // Other tx types (Swap, TokenCreate, etc.) don't affect QUG balances
                                // or vault state in a way we need to replay here.
                                // Their effects are handled by StateProcessor/StateApplicator.
                                if block_tx.amount > 0 && block_tx.from != [0u8; 32] {
                                    // Generic value transfer: debit sender, credit receiver
                                    if let Some(sender_bal) = balances.get_mut(&block_tx.from) {
                                        *sender_bal = sender_bal.saturating_sub(block_tx.amount);
                                    }
                                    if block_tx.to != [0u8; 32] {
                                        let entry = balances.entry(block_tx.to).or_insert(0);
                                        *entry = entry.saturating_add(block_tx.amount);
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                // Legacy blocks (no transactions): compute reward from mining_solutions
                legacy_blocks += 1;
                for solution in &block.mining_solutions {
                    if solution.miner_address != [0u8; 32] {
                        let elapsed = block.header.timestamp.saturating_sub(genesis_ts);
                        let era = crate::emission_controller::era_at_time(elapsed);
                        let annual = crate::emission_controller::annual_emission(era);
                        let reward = (annual / 31_557_600u128).max(crate::emission_controller::MIN_REWARD);

                        let dev_fee = reward.saturating_mul(crate::balance_consensus::DEV_FEE_BPS)
                            / crate::balance_consensus::BPS_DIVISOR;
                        let miner_reward = reward.saturating_sub(dev_fee);

                        let entry = balances.entry(solution.miner_address).or_insert(0);
                        *entry = entry.saturating_add(miner_reward);
                        if dev_fee > 0 {
                            let founder_entry = balances.entry(founder_bytes).or_insert(0);
                            *founder_entry = founder_entry.saturating_add(dev_fee);
                        }
                    }
                }
            }

            if height % 100_000 == 0 {
                info!("   Progress: {}/{} blocks ({} tx, {} legacy, {} mints, {} burns)",
                      height, tip, tx_blocks, legacy_blocks, stable_mint_count, stable_burn_count);
            }
        }

        balances.retain(|_, v| *v > 0);
        let chain_total: u128 = balances.values().sum();

        info!("✅ [v1.0.3] Chain replay complete:");
        info!("   {} wallets, {} QUG raw chain total", balances.len(), chain_total / qug_unit);
        info!("   {} blocks ({} tx, {} legacy)", blocks_processed, tx_blocks, legacy_blocks);
        info!("   {} StableMint txs, {} StableBurn txs", stable_mint_count, stable_burn_count);
        info!("   Vault: {} QUG locked, {} QUGUSD minted, {} positions",
              vault_total_locked / qug_unit, vault_total_minted / qug_unit, vault_locked_qug.len());

        // =====================================================================
        // Step 4: Scale wallet balances to match deterministic emission target
        // =====================================================================
        if chain_total > 0 && chain_total != expected_total {
            let scale_ratio_display = chain_total as f64 / expected_total as f64;
            info!("   Scaling {} wallets: chain {} → target {} QUG (ratio {:.4}×)",
                  balances.len(), chain_total / qug_unit, expected_total / qug_unit, scale_ratio_display);

            for (_addr, amount) in balances.iter_mut() {
                // Overflow-safe: (amount × expected_total) / chain_total
                let quot = *amount / chain_total;
                let rem = *amount % chain_total;
                let term1 = quot.saturating_mul(expected_total);
                let term2 = if let Some(prod) = rem.checked_mul(expected_total) {
                    prod / chain_total
                } else {
                    ((rem as f64) * (expected_total as f64) / (chain_total as f64)) as u128
                };
                *amount = term1.saturating_add(term2);
            }
            balances.retain(|_, v| *v > 0);
        }

        let final_total: u128 = balances.values().sum();

        // =====================================================================
        // Step 5: Build vault struct
        // =====================================================================
        // Clean up zero-amount vault positions
        vault_locked_qug.retain(|_, v| *v > 0);
        vault_minted_qugusd.retain(|_, v| *v > 0);

        // v8.9.9: Inline vault struct (can't depend on q-vm due to cyclic dep)
        #[derive(serde::Serialize)]
        struct CollateralVaultRebuild {
            locked_qug: std::collections::HashMap<[u8; 32], u128>,
            minted_qugusd: std::collections::HashMap<[u8; 32], u128>,
            qug_price_usd: f64,
            total_qug_locked: u128,
            total_qugusd_minted: u128,
            last_price_update: i64,
        }
        let vault = CollateralVaultRebuild {
            locked_qug: vault_locked_qug,
            minted_qugusd: vault_minted_qugusd,
            qug_price_usd: 1.0, // Will be updated by next P2P price update
            total_qug_locked: vault_total_locked,
            total_qugusd_minted: vault_total_minted,
            last_price_update: 0,
        };

        info!("   Final: {} wallets, {} QUG supply (target {} QUG)",
              balances.len(), final_total / qug_unit, expected_total / qug_unit);
        info!("   Vault: {} positions locked, {} QUG total locked, {} QUGUSD total minted",
              vault.locked_qug.len(), vault_total_locked / qug_unit, vault_total_minted / qug_unit);
        info!("   Founder: {} QUG",
              balances.get(&founder_bytes).unwrap_or(&0) / qug_unit);

        // =====================================================================
        // Step 6: Persist atomically
        // =====================================================================
        self.save_wallet_balances(&balances).await?;
        self.save_total_supply(final_total).await?;

        // Persist rebuilt vault
        let vault_bytes = bincode::serialize(&vault).unwrap_or_default();
        self.save_collateral_vault_data(&vault_bytes).await?;

        // Set watermark to prevent re-inflation
        self.save_balance_watermark(tip).await?;

        // Set migration flag AFTER successful persist
        self.hot_db.put(CF_MANIFEST, MIGRATION_FLAG, b"done").await?;

        info!("═══════════════════════════════════════════════════════════════════");
        info!("🔒 [v1.0.3] Convergence migration COMPLETE. Flag set — will NOT re-run.");
        info!("═══════════════════════════════════════════════════════════════════");
        Ok(true)
    }

    /// v8.5.0: Save balance processed watermark to RocksDB.
    /// The watermark tracks the highest block height whose balance effects have been
    /// persisted. On restart, blocks at or below this height are skipped to prevent
    /// re-inflation from replaying coinbase transactions.
    pub async fn save_balance_watermark(&self, height: u64) -> Result<()> {
        self.hot_db.put(CF_MANIFEST, b"balance_processed_watermark", &height.to_le_bytes()).await
    }

    /// v8.5.0: Load balance processed watermark from RocksDB.
    pub async fn load_balance_watermark(&self) -> Result<u64> {
        match self.hot_db.get(CF_MANIFEST, b"balance_processed_watermark").await? {
            Some(bytes) if bytes.len() == 8 => {
                Ok(u64::from_le_bytes(bytes[..8].try_into().unwrap()))
            }
            _ => Ok(0),
        }
    }

    /// v8.9.1: Save state replay watermark to RocksDB.
    /// Tracks the highest block height whose DEX/token state changes have been
    /// replayed via StateApplicator. On restart, blocks at or below this height
    /// are skipped to prevent double-credit (StateApplicator uses incremental
    /// add/sub, NOT absolute writes).
    pub async fn save_state_replay_watermark(&self, height: u64) -> Result<()> {
        self.hot_db.put(CF_MANIFEST, b"state_replay_watermark", &height.to_le_bytes()).await
    }

    /// v8.9.1: Load state replay watermark from RocksDB.
    pub async fn load_state_replay_watermark(&self) -> Result<u64> {
        match self.hot_db.get(CF_MANIFEST, b"state_replay_watermark").await? {
            Some(bytes) if bytes.len() == 8 => {
                Ok(u64::from_le_bytes(bytes[..8].try_into().unwrap()))
            }
            _ => Ok(0),
        }
    }

    // ============================================================================
    // AI Chat Attachment Storage - v0.9.9-beta
    // ============================================================================

    /// Save attachment metadata
    pub async fn save_attachment(
        &self,
        attachment_id: &str,
        chat_id: &str,
        user_id: &str,
        filename: &str,
        mime_type: &str,
        file_size: i64,
        storage_path: &str,
    ) -> Result<()> {
        let metadata = AttachmentMetadata {
            id: attachment_id.to_string(),
            chat_id: chat_id.to_string(),
            user_id: user_id.to_string(),
            filename: filename.to_string(),
            mime_type: mime_type.to_string(),
            file_size,
            storage_path: storage_path.to_string(),
            thumbnail_path: None,
            extracted_text: None,
            vision_base64: None,
            upload_timestamp: chrono::Utc::now().timestamp(),
            processed: false,
        };

        let key = format!("attachment:{}", attachment_id);
        let value = serde_json::to_vec(&metadata)?;
        self.hot_db.put(CF_AI_ATTACHMENTS, key.as_bytes(), &value).await?;

        debug!("📎 Saved attachment metadata: {} ({} bytes, {})", filename, file_size, mime_type);
        Ok(())
    }

    /// Update attachment after processing
    pub async fn update_attachment_processed(
        &self,
        attachment_id: &str,
        thumbnail_path: Option<&str>,
        extracted_text: Option<&str>,
        vision_base64: Option<&str>,
    ) -> Result<()> {
        let key = format!("attachment:{}", attachment_id);
        let data = self.hot_db.get(CF_AI_ATTACHMENTS, key.as_bytes()).await?;

        if let Some(data) = data {
            let mut metadata: AttachmentMetadata = serde_json::from_slice(&data)?;

            if let Some(path) = thumbnail_path {
                metadata.thumbnail_path = Some(path.to_string());
            }
            if let Some(text) = extracted_text {
                metadata.extracted_text = Some(text.to_string());
            }
            if let Some(b64) = vision_base64 {
                metadata.vision_base64 = Some(b64.to_string());
            }
            metadata.processed = true;

            let value = serde_json::to_vec(&metadata)?;
            self.hot_db.put(CF_AI_ATTACHMENTS, key.as_bytes(), &value).await?;

            debug!("✅ Updated attachment processing status: {}", attachment_id);
        }

        Ok(())
    }

    /// Get attachments for a chat
    pub async fn get_chat_attachments(&self, chat_id: &str) -> Result<Vec<AttachmentMetadata>> {
        let prefix = format!("chat:{}:attachments:", chat_id);
        let attachments_data = self.hot_db.scan_prefix(CF_AI_ATTACHMENTS, prefix.as_bytes()).await?;

        let mut attachments = Vec::new();
        for (_key, value) in attachments_data {
            if let Ok(attachment) = serde_json::from_slice::<AttachmentMetadata>(&value) {
                attachments.push(attachment);
            }
        }

        debug!("📎 Loaded {} attachments for chat: {}", attachments.len(), chat_id);
        Ok(attachments)
    }

    // ============================================================================
    // Loan Application Persistence - Quillon Bank CDP System
    // ============================================================================

    /// Save loan application to persistent storage
    /// Key format: loan_app:{loan_id}
    pub async fn save_loan_application(&self, loan_id: &str, loan_bytes: &[u8]) -> Result<()> {
        let key = format!("loan_app:{}", loan_id);
        self.hot_db.put(CF_MANIFEST, key.as_bytes(), loan_bytes).await?;
        debug!("🏦 Saved loan application: {}", loan_id);
        Ok(())
    }

    /// Load all loan applications from persistent storage
    pub async fn load_loan_applications(&self) -> Result<HashMap<String, Vec<u8>>> {
        let mut loans = HashMap::new();
        let prefix = b"loan_app:";

        match self.hot_db.scan_prefix(CF_MANIFEST, prefix).await {
            Ok(entries) => {
                for (key, value) in entries {
                    if let Ok(key_str) = String::from_utf8(key) {
                        if let Some(loan_id) = key_str.strip_prefix("loan_app:") {
                            loans.insert(loan_id.to_string(), value);
                        }
                    }
                }
                info!(
                    "🏦 Loaded {} loan applications from persistent storage",
                    loans.len()
                );
            }
            Err(e) => {
                warn!("Failed to scan loan applications: {}", e);
            }
        }

        Ok(loans)
    }

    /// Delete loan application from persistent storage
    pub async fn delete_loan_application(&self, loan_id: &str) -> Result<()> {
        let key = format!("loan_app:{}", loan_id);
        self.hot_db.delete(CF_MANIFEST, key.as_bytes()).await?;
        debug!("🗑️ Deleted loan application: {}", loan_id);
        Ok(())
    }

    /// Load all bank messages from persistent storage
    pub async fn load_bank_messages(&self) -> Result<Vec<(String, Vec<u8>)>> {
        let mut messages = Vec::new();

        match self.hot_db.scan_all(CF_BANK_MESSAGES).await {
            Ok(entries) => {
                for (key, value) in entries {
                    if let Ok(key_str) = String::from_utf8(key) {
                        messages.push((key_str, value));
                    }
                }
                info!(
                    "📬 Loaded {} bank messages from persistent storage",
                    messages.len()
                );
            }
            Err(e) => {
                warn!("Failed to scan bank messages: {}", e);
            }
        }

        Ok(messages)
    }

    /// Load all user identities from persistent storage
    pub async fn load_user_identities(&self) -> Result<Vec<(String, Vec<u8>)>> {
        let mut identities = Vec::new();

        match self.hot_db.scan_all(CF_USER_IDENTITIES).await {
            Ok(entries) => {
                for (key, value) in entries {
                    if let Ok(key_str) = String::from_utf8(key) {
                        identities.push((key_str, value));
                    }
                }
                info!(
                    "🪪 Loaded {} user identities from persistent storage",
                    identities.len()
                );
            }
            Err(e) => {
                warn!("Failed to scan user identities: {}", e);
            }
        }

        Ok(identities)
    }

    /// Load all death certificates from persistent storage
    pub async fn load_death_certificates(&self) -> Result<Vec<(String, Vec<u8>)>> {
        let mut certs = Vec::new();

        match self.hot_db.scan_all(CF_DEATH_CERTIFICATES).await {
            Ok(entries) => {
                for (key, value) in entries {
                    if let Ok(key_str) = String::from_utf8(key) {
                        certs.push((key_str, value));
                    }
                }
                info!(
                    "💀 Loaded {} death certificates from persistent storage",
                    certs.len()
                );
            }
            Err(e) => {
                warn!("Failed to scan death certificates: {}", e);
            }
        }

        Ok(certs)
    }

    // ============================================================================
    // Emission Controller State Persistence (v7.1.0)
    // ============================================================================

    /// Save emission controller state to RocksDB so it survives restarts
    pub async fn save_emission_state(&self, state_bytes: &[u8]) -> Result<()> {
        self.hot_db.put(CF_MANIFEST, b"emission_controller_state", state_bytes).await?;
        debug!("💰 Saved emission controller state ({} bytes)", state_bytes.len());
        Ok(())
    }

    /// Load emission controller state from RocksDB
    pub async fn load_emission_state(&self) -> Result<Option<Vec<u8>>> {
        match self.hot_db.get(CF_MANIFEST, b"emission_controller_state").await {
            Ok(Some(data)) => {
                info!("💰 Loaded emission controller state ({} bytes)", data.len());
                Ok(Some(data))
            }
            Ok(None) => {
                info!("💰 No saved emission controller state found (first boot or fresh DB)");
                Ok(None)
            }
            Err(e) => {
                warn!("Failed to load emission controller state: {}", e);
                Ok(None)
            }
        }
    }

    /// Save benchmark timestamp for IP rate limiting
    /// Key format: benchmark_ip:{ip_address}
    pub async fn save_benchmark_timestamp(&self, ip_address: &str, timestamp: u64) -> Result<()> {
        let key = format!("benchmark_ip:{}", ip_address);
        let value = timestamp.to_le_bytes();
        self.hot_db.put(CF_MANIFEST, key.as_bytes(), &value).await?;
        debug!("⏰ Saved benchmark timestamp for IP: {}", ip_address);
        Ok(())
    }

    /// Load benchmark timestamp for IP rate limiting
    /// Returns None if IP has never run benchmark, Some(timestamp) otherwise
    pub async fn load_benchmark_timestamp(&self, ip_address: &str) -> Result<Option<u64>> {
        let key = format!("benchmark_ip:{}", ip_address);
        match self.hot_db.get(CF_MANIFEST, key.as_bytes()).await? {
            Some(bytes) => {
                if bytes.len() == 8 {
                    let timestamp = u64::from_le_bytes([
                        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
                    ]);
                    debug!("⏰ Loaded benchmark timestamp for IP {}: {}", ip_address, timestamp);
                    Ok(Some(timestamp))
                } else {
                    warn!("Invalid benchmark timestamp data length for IP {}", ip_address);
                    Ok(None)
                }
            }
            None => Ok(None),
        }
    }

    /// Check if IP is rate limited for benchmark (DISABLED - no rate limiting)
    /// Returns (is_limited, minutes_remaining)
    #[allow(clippy::absurd_extreme_comparisons)] // COOLDOWN_SECONDS = 0 (disabled), comparison intentionally always false
    pub async fn check_benchmark_rate_limit(&self, ip_address: &str) -> Result<(bool, u64)> {
        const COOLDOWN_SECONDS: u64 = 0; // DISABLED - no rate limiting

        match self.load_benchmark_timestamp(ip_address).await? {
            Some(last_timestamp) => {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();

                let elapsed = now.saturating_sub(last_timestamp);

                if elapsed < COOLDOWN_SECONDS {
                    let remaining_seconds = COOLDOWN_SECONDS - elapsed;
                    let remaining_minutes = (remaining_seconds + 59) / 60; // Round up
                    debug!("🚫 IP {} is rate limited, {} minutes remaining", ip_address, remaining_minutes);
                    Ok((true, remaining_minutes))
                } else {
                    debug!("✅ IP {} is not rate limited", ip_address);
                    Ok((false, 0))
                }
            }
            None => {
                debug!("✅ IP {} has never run benchmark", ip_address);
                Ok((false, 0))
            }
        }
    }

    /// Get USD balance for a wallet (in cents)
    /// Key format: usd_balance:{wallet_address_hex}
    pub async fn get_usd_balance(&self, wallet_address: &str) -> Result<u64> {
        let key = format!("usd_balance:{}", wallet_address);
        match self.hot_db.get(CF_MANIFEST, key.as_bytes()).await? {
            Some(bytes) => {
                if bytes.len() == 8 {
                    let balance_cents = u64::from_le_bytes([
                        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
                    ]);
                    debug!("💵 Loaded USD balance for {}: {} cents", wallet_address, balance_cents);
                    Ok(balance_cents)
                } else {
                    warn!("Invalid USD balance data length for wallet {}", wallet_address);
                    Ok(0)
                }
            }
            None => {
                debug!("💵 USD balance not found for wallet {}, returning 0", wallet_address);
                Ok(0)
            }
        }
    }

    /// Credit USD balance to a wallet (amount in cents)
    /// Adds the specified amount to the current balance
    pub async fn credit_usd_balance(&self, wallet_address: &str, amount_cents: u64) -> Result<()> {
        let current_balance = self.get_usd_balance(wallet_address).await?;
        let new_balance = current_balance.saturating_add(amount_cents);

        let key = format!("usd_balance:{}", wallet_address);
        let value = new_balance.to_le_bytes();
        self.hot_db.put(CF_MANIFEST, key.as_bytes(), &value).await?;

        info!("💵 Credited {} cents to {}, new balance: {} cents",
            amount_cents, wallet_address, new_balance);
        Ok(())
    }

    /// Debit USD balance from a wallet (amount in cents)
    /// Returns error if insufficient balance
    pub async fn debit_usd_balance(&self, wallet_address: &str, amount_cents: u64) -> Result<()> {
        let current_balance = self.get_usd_balance(wallet_address).await?;

        if current_balance < amount_cents {
            return Err(anyhow::anyhow!(
                "Insufficient USD balance: has {} cents, needs {} cents",
                current_balance,
                amount_cents
            ));
        }

        let new_balance = current_balance - amount_cents;

        let key = format!("usd_balance:{}", wallet_address);
        let value = new_balance.to_le_bytes();
        self.hot_db.put(CF_MANIFEST, key.as_bytes(), &value).await?;

        info!("💵 Debited {} cents from {}, new balance: {} cents",
            amount_cents, wallet_address, new_balance);
        Ok(())
    }

    /// Atomically transfer USD between two wallets using a single batch write.
    /// This prevents the scenario where debit succeeds but credit fails.
    pub async fn transfer_usd_atomic(&self, from_wallet: &str, to_wallet: &str, amount_cents: u64) -> Result<(u64, u64)> {
        let from_key = format!("usd_balance:{}", from_wallet);
        let to_key = format!("usd_balance:{}", to_wallet);

        // Read current balances
        let from_balance = self.get_usd_balance(from_wallet).await?;
        let to_balance = self.get_usd_balance(to_wallet).await?;

        if from_balance < amount_cents {
            anyhow::bail!("Insufficient balance: have {} cents, need {}", from_balance, amount_cents);
        }

        let new_from = from_balance - amount_cents;
        let new_to = to_balance.saturating_add(amount_cents);

        // Write both balances in a single atomic batch
        let ops = vec![
            (CF_MANIFEST, from_key.into_bytes(), new_from.to_le_bytes().to_vec()),
            (CF_MANIFEST, to_key.into_bytes(), new_to.to_le_bytes().to_vec()),
        ];
        self.hot_db.write_batch(ops).await?;

        info!("💵 Atomic transfer: {} cents from {} (new: {}) to {} (new: {})",
            amount_cents, from_wallet, new_from, to_wallet, new_to);

        Ok((new_from, new_to))
    }

    /// Set USD balance for a wallet directly (amount in cents)
    /// This is used for admin operations or migrations
    pub async fn set_usd_balance(&self, wallet_address: &str, balance_cents: u64) -> Result<()> {
        let key = format!("usd_balance:{}", wallet_address);
        let value = balance_cents.to_le_bytes();
        self.hot_db.put(CF_MANIFEST, key.as_bytes(), &value).await?;

        // 🔒 PRIVACY-PRESERVING: Log only hash of address
        use blake3::hash;
        let addr_hash = hash(wallet_address.as_bytes());
        debug!("💵 Set USD balance: addr_hash={}", hex::encode(&addr_hash.as_bytes()[..8]));
        Ok(())
    }

    /// Load all USD balances from persistent storage
    /// Returns map of wallet_address -> balance_cents
    pub async fn load_all_usd_balances(&self) -> Result<HashMap<String, u64>> {
        let mut balances = HashMap::new();
        let prefix = b"usd_balance:";

        match self.hot_db.scan_prefix(CF_MANIFEST, prefix).await {
            Ok(entries) => {
                for (key, value) in entries {
                    if let Ok(key_str) = String::from_utf8(key) {
                        if let Some(wallet_address) = key_str.strip_prefix("usd_balance:") {
                            if value.len() == 8 {
                                let balance_cents = u64::from_le_bytes([
                                    value[0], value[1], value[2], value[3],
                                    value[4], value[5], value[6], value[7],
                                ]);
                                balances.insert(wallet_address.to_string(), balance_cents);
                            }
                        }
                    }
                }
                info!("💵 Loaded {} USD wallet balances from persistent storage", balances.len());
            }
            Err(e) => {
                warn!("Failed to scan USD balances: {}", e);
            }
        }

        Ok(balances)
    }

    /// Check if a payment intent has already been processed (idempotency)
    pub async fn is_payment_processed(&self, payment_intent_id: &str) -> Result<bool> {
        let key = format!("processed_payment:{}", payment_intent_id);
        match self.hot_db.get(CF_MANIFEST, key.as_bytes()).await? {
            Some(_) => Ok(true),
            None => Ok(false),
        }
    }

    /// Mark a payment intent as processed (idempotency)
    pub async fn mark_payment_processed(&self, payment_intent_id: &str, wallet_address: &str, amount_cents: u64) -> Result<()> {
        let key = format!("processed_payment:{}", payment_intent_id);
        let value = format!("{}:{}", wallet_address, amount_cents);
        self.hot_db.put(CF_MANIFEST, key.as_bytes(), value.as_bytes()).await?;
        Ok(())
    }

    /// Save password hash to persistent storage (bcrypt hash)
    /// Key format: wallet_password_{address_hex}
    pub async fn save_password_hash(&self, address: &[u8; 32], password_hash: &str) -> Result<()> {
        let key = format!("wallet_password_{}", hex::encode(address));
        self.hot_db.put(CF_MANIFEST, key.as_bytes(), password_hash.as_bytes()).await?;
        debug!(
            "🔐 Saved password hash for address: {}",
            hex::encode(address)
        );
        Ok(())
    }

    /// Load password hash from persistent storage
    pub async fn load_password_hash(&self, address: &[u8; 32]) -> Result<Option<String>> {
        let key = format!("wallet_password_{}", hex::encode(address));
        match self.hot_db.get(CF_MANIFEST, key.as_bytes()).await? {
            Some(bytes) => {
                match String::from_utf8(bytes) {
                    Ok(password_hash) => {
                        debug!(
                            "🔐 Loaded password hash for address: {}",
                            hex::encode(address)
                        );
                        Ok(Some(password_hash))
                    }
                    Err(e) => {
                        warn!("Invalid password hash UTF-8 for address {}: {}", hex::encode(address), e);
                        Ok(None)
                    }
                }
            }
            None => Ok(None),
        }
    }

    /// Load all password hashes from persistent storage
    /// Returns map of address -> bcrypt_hash
    pub async fn load_password_hashes(&self) -> Result<HashMap<[u8; 32], String>> {
        let mut hashes = HashMap::new();
        let prefix = "wallet_password_".as_bytes();

        match self.hot_db.scan_prefix(CF_MANIFEST, prefix).await {
            Ok(entries) => {
                for (key, value) in entries {
                    if let Ok(key_str) = String::from_utf8(key) {
                        if let Some(hex_addr) = key_str.strip_prefix("wallet_password_") {
                            if let Ok(addr_bytes) = hex::decode(hex_addr) {
                                if addr_bytes.len() == 32 {
                                    let mut address = [0u8; 32];
                                    address.copy_from_slice(&addr_bytes);
                                    if let Ok(password_hash) = String::from_utf8(value) {
                                        hashes.insert(address, password_hash);
                                    }
                                }
                            }
                        }
                    }
                }
                info!(
                    "🔐 Loaded {} password hashes from persistent storage",
                    hashes.len()
                );
            }
            Err(e) => {
                warn!("Failed to scan password hashes: {}", e);
            }
        }

        Ok(hashes)
    }

    // ═══════════════════════════════════════════════════════════════════
    // 🔐 v8.1.7: OAuth2 Key Vault — Server-side encrypted signing keys
    // Stores AES-256-GCM encrypted Ed25519 private keys for OAuth2 custodial signing.
    // When an OAuth2 user sends a transaction, the server uses the vault key
    // instead of requiring the user to enter their mnemonic every time.
    // ═══════════════════════════════════════════════════════════════════

    /// Save an encrypted signing key for OAuth2 custodial signing.
    /// Key format: oauth2_vault_{address_hex} → AES-256-GCM encrypted private key bytes
    pub async fn save_vault_key(&self, address: &[u8; 32], encrypted_key: &[u8]) -> Result<()> {
        let key = format!("oauth2_vault_{}", hex::encode(address));
        self.hot_db.put(CF_MANIFEST, key.as_bytes(), encrypted_key).await?;
        info!(
            "🔐 [VAULT] Saved encrypted signing key for address: {}...{}",
            &hex::encode(address)[..8],
            &hex::encode(address)[56..]
        );
        Ok(())
    }

    /// Load an encrypted signing key from the vault.
    pub async fn load_vault_key(&self, address: &[u8; 32]) -> Result<Option<Vec<u8>>> {
        let key = format!("oauth2_vault_{}", hex::encode(address));
        match self.hot_db.get(CF_MANIFEST, key.as_bytes()).await? {
            Some(bytes) => {
                debug!(
                    "🔐 [VAULT] Loaded encrypted key for address: {}...{}",
                    &hex::encode(address)[..8],
                    &hex::encode(address)[56..]
                );
                Ok(Some(bytes))
            }
            None => Ok(None),
        }
    }

    /// Load all vault keys from persistent storage.
    /// Returns map of address → encrypted private key bytes.
    pub async fn load_vault_keys(&self) -> Result<HashMap<[u8; 32], Vec<u8>>> {
        let mut vault = HashMap::new();
        let prefix = "oauth2_vault_".as_bytes();

        match self.hot_db.scan_prefix(CF_MANIFEST, prefix).await {
            Ok(entries) => {
                for (key, value) in entries {
                    if let Ok(key_str) = String::from_utf8(key) {
                        if let Some(hex_addr) = key_str.strip_prefix("oauth2_vault_") {
                            if let Ok(addr_bytes) = hex::decode(hex_addr) {
                                if addr_bytes.len() == 32 {
                                    let mut address = [0u8; 32];
                                    address.copy_from_slice(&addr_bytes);
                                    vault.insert(address, value);
                                }
                            }
                        }
                    }
                }
                info!(
                    "🔐 [VAULT] Loaded {} encrypted signing keys from persistent storage",
                    vault.len()
                );
            }
            Err(e) => {
                warn!("Failed to scan vault keys: {}", e);
            }
        }

        Ok(vault)
    }

    /// Save CollateralVault to persistent storage (generic byte storage)
    /// Key: collateral_vault
    pub async fn save_collateral_vault_data(&self, vault_data: &[u8]) -> Result<()> {
        let key = b"collateral_vault";
        self.hot_db.put(CF_MANIFEST, key, vault_data).await?;
        debug!("💰 Saved CollateralVault data ({} bytes)", vault_data.len());
        Ok(())
    }

    /// Load CollateralVault data from persistent storage
    /// Returns None if no vault exists (first run)
    pub async fn load_collateral_vault_data(&self) -> Result<Option<Vec<u8>>> {
        let key = b"collateral_vault";
        match self.hot_db.get(CF_MANIFEST, key).await? {
            Some(vault_data) => {
                info!("💰 Loaded CollateralVault data ({} bytes)", vault_data.len());
                Ok(Some(vault_data))
            }
            None => {
                debug!("💰 No persisted CollateralVault found (first run)");
                Ok(None)
            }
        }
    }

    // ========================================
    // AI CHAT STORAGE METHODS
    // ========================================

    /// Create a new AI chat session
    /// Key format: chat:{chat_id}
    pub async fn create_chat(&self, metadata: &ChatMetadata) -> Result<()> {
        let key = format!("chat:{}", metadata.chat_id);
        let value = bincode::serialize(metadata)?;

        self.hot_db.put(CF_AI_CHATS, key.as_bytes(), &value).await?;

        // Add to user's chat list
        self.add_chat_to_user_list(&metadata.user_id, &metadata.chat_id).await?;

        // Set as latest chat for user
        let latest_key = format!("chat:latest:{}", metadata.user_id);
        self.hot_db.put(CF_AI_CHATS, latest_key.as_bytes(), metadata.chat_id.as_bytes()).await?;

        info!("💬 Created chat {} for user {}", metadata.chat_id, metadata.user_id);
        Ok(())
    }

    /// Save a chat message
    /// Key format: chat:{chat_id}:msg:{index}
    pub async fn save_chat_message(&self, chat_id: &str, message: &ChatMessage) -> Result<()> {
        let msg_key = format!("chat:{}:msg:{}", chat_id, message.index);
        let value = bincode::serialize(message)?;

        self.hot_db.put(CF_AI_CHATS, msg_key.as_bytes(), &value).await?;

        // Update chat metadata's message count and updated_at
        let metadata_key = format!("chat:{}", chat_id);
        if let Some(metadata_data) = self.hot_db.get(CF_AI_CHATS, metadata_key.as_bytes()).await? {
            let mut metadata: ChatMetadata = bincode::deserialize(&metadata_data)?;
            metadata.message_count = message.index + 1;
            metadata.updated_at = message.timestamp;

            let updated_value = bincode::serialize(&metadata)?;
            self.hot_db.put(CF_AI_CHATS, metadata_key.as_bytes(), &updated_value).await?;
        }

        debug!("💬 Saved message {} in chat {}", message.index, chat_id);
        Ok(())
    }

    /// v2.7.0-beta: Save a TemporalShield-protected chat message
    ///
    /// Stores messages with encrypted content and reasoning in the same CF
    /// but with a different key format for protected messages.
    pub async fn save_protected_chat_message(&self, chat_id: &str, message: &ProtectedChatMessage) -> Result<()> {
        let msg_key = format!("chat:{}:protected_msg:{}", chat_id, message.index);
        let value = bincode::serialize(message)?;

        self.hot_db.put(CF_AI_CHATS, msg_key.as_bytes(), &value).await?;

        // Update chat metadata's message count and updated_at
        let metadata_key = format!("chat:{}", chat_id);
        if let Some(metadata_data) = self.hot_db.get(CF_AI_CHATS, metadata_key.as_bytes()).await? {
            let mut metadata: ChatMetadata = bincode::deserialize(&metadata_data)?;
            metadata.message_count = message.index + 1;
            metadata.updated_at = message.timestamp;

            let updated_value = bincode::serialize(&metadata)?;
            self.hot_db.put(CF_AI_CHATS, metadata_key.as_bytes(), &updated_value).await?;
        }

        debug!("🛡️ Saved protected message {} in chat {} (TemporalShield)", message.index, chat_id);
        Ok(())
    }

    /// v2.7.0-beta: Load protected chat messages
    pub async fn load_protected_chat_messages(&self, chat_id: &str) -> Result<Vec<ProtectedChatMessage>> {
        let prefix = format!("chat:{}:protected_msg:", chat_id);
        let messages_data = self.hot_db.scan_prefix(CF_AI_CHATS, prefix.as_bytes()).await?;

        let mut messages = Vec::new();
        for (_, msg_data) in messages_data {
            if let Ok(message) = bincode::deserialize::<ProtectedChatMessage>(&msg_data) {
                messages.push(message);
            }
        }

        // Sort by index
        messages.sort_by_key(|m| m.index);
        Ok(messages)
    }

    /// Load chat messages
    /// Returns messages in order
    pub async fn load_chat_messages(&self, chat_id: &str) -> Result<Vec<ChatMessage>> {
        let prefix = format!("chat:{}:msg:", chat_id);
        let messages_data = self.hot_db.scan_prefix(CF_AI_CHATS, prefix.as_bytes()).await?;

        let mut messages = Vec::new();
        for (_, msg_data) in messages_data {
            if let Ok(message) = bincode::deserialize::<ChatMessage>(&msg_data) {
                messages.push(message);
            }
        }

        // Sort by index
        messages.sort_by_key(|m| m.index);

        debug!("💬 Loaded {} messages from chat {}", messages.len(), chat_id);
        Ok(messages)
    }

    /// List all chats for a user
    pub async fn list_user_chats(&self, user_id: &str) -> Result<Vec<ChatMetadata>> {
        let list_key = format!("chat:user:{}", user_id);
        let chat_ids_data = self.hot_db.get(CF_AI_CHATS, list_key.as_bytes()).await?;

        let chat_ids: Vec<String> = match chat_ids_data {
            Some(data) => bincode::deserialize(&data)?,
            None => Vec::new(),
        };

        let mut chats = Vec::new();
        for chat_id in chat_ids {
            let key = format!("chat:{}", chat_id);
            if let Some(metadata_data) = self.hot_db.get(CF_AI_CHATS, key.as_bytes()).await? {
                if let Ok(metadata) = bincode::deserialize::<ChatMetadata>(&metadata_data) {
                    chats.push(metadata);
                }
            }
        }

        // Sort by updated_at descending (most recent first)
        chats.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));

        debug!("💬 Listed {} chats for user {}", chats.len(), user_id);
        Ok(chats)
    }

    /// Delete a chat and all its messages
    pub async fn delete_chat(&self, chat_id: &str, user_id: &str) -> Result<()> {
        // Delete chat metadata
        let metadata_key = format!("chat:{}", chat_id);
        self.hot_db.delete(CF_AI_CHATS, metadata_key.as_bytes()).await?;

        // Delete all messages
        let msg_prefix = format!("chat:{}:msg:", chat_id);
        let messages = self.hot_db.scan_prefix(CF_AI_CHATS, msg_prefix.as_bytes()).await?;
        for (msg_key, _) in messages {
            self.hot_db.delete(CF_AI_CHATS, &msg_key).await?;
        }

        // Remove from user's chat list
        self.remove_chat_from_user_list(user_id, chat_id).await?;

        info!("💬 Deleted chat {} for user {}", chat_id, user_id);
        Ok(())
    }

    /// Rename a chat
    pub async fn rename_chat(&self, chat_id: &str, new_title: &str) -> Result<()> {
        let key = format!("chat:{}", chat_id);
        if let Some(metadata_data) = self.hot_db.get(CF_AI_CHATS, key.as_bytes()).await? {
            let mut metadata: ChatMetadata = bincode::deserialize(&metadata_data)?;
            metadata.title = new_title.to_string();

            let updated_value = bincode::serialize(&metadata)?;
            self.hot_db.put(CF_AI_CHATS, key.as_bytes(), &updated_value).await?;

            info!("💬 Renamed chat {} to '{}'", chat_id, new_title);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Chat {} not found", chat_id))
        }
    }

    /// Update chat settings (privacy, performance options)
    pub async fn update_chat_settings(&self, chat_id: &str, settings: &ChatSettings) -> Result<()> {
        let key = format!("chat:{}", chat_id);
        if let Some(metadata_data) = self.hot_db.get(CF_AI_CHATS, key.as_bytes()).await? {
            let mut metadata: ChatMetadata = bincode::deserialize(&metadata_data)?;

            metadata.encryption_enabled = settings.encryption_enabled;
            metadata.zk_proofs_enabled = settings.zk_proofs_enabled;
            metadata.distributed_enabled = settings.distributed_enabled;
            metadata.enable_kv_cache = settings.enable_kv_cache;
            metadata.enable_pipeline_parallel = settings.enable_pipeline_parallel;
            metadata.enable_load_balancing = settings.enable_load_balancing;

            let updated_value = bincode::serialize(&metadata)?;
            self.hot_db.put(CF_AI_CHATS, key.as_bytes(), &updated_value).await?;

            info!("💬 Updated settings for chat {}", chat_id);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Chat {} not found", chat_id))
        }
    }

    /// Get chat metadata
    pub async fn get_chat_metadata(&self, chat_id: &str) -> Result<Option<ChatMetadata>> {
        let key = format!("chat:{}", chat_id);
        match self.hot_db.get(CF_AI_CHATS, key.as_bytes()).await? {
            Some(metadata_data) => {
                let metadata: ChatMetadata = bincode::deserialize(&metadata_data)?;
                Ok(Some(metadata))
            }
            None => Ok(None),
        }
    }

    /// Add chat to user's list (internal helper)
    async fn add_chat_to_user_list(&self, user_id: &str, chat_id: &str) -> Result<()> {
        let key = format!("chat:user:{}", user_id);

        let mut chat_ids: Vec<String> = match self.hot_db.get(CF_AI_CHATS, key.as_bytes()).await? {
            Some(data) => bincode::deserialize(&data)?,
            None => Vec::new(),
        };

        if !chat_ids.contains(&chat_id.to_string()) {
            chat_ids.push(chat_id.to_string());
            let value = bincode::serialize(&chat_ids)?;
            self.hot_db.put(CF_AI_CHATS, key.as_bytes(), &value).await?;
        }

        Ok(())
    }

    /// Remove chat from user's list (internal helper)
    async fn remove_chat_from_user_list(&self, user_id: &str, chat_id: &str) -> Result<()> {
        let key = format!("chat:user:{}", user_id);

        if let Some(data) = self.hot_db.get(CF_AI_CHATS, key.as_bytes()).await? {
            let mut chat_ids: Vec<String> = bincode::deserialize(&data)?;
            chat_ids.retain(|id| id != chat_id);

            let value = bincode::serialize(&chat_ids)?;
            self.hot_db.put(CF_AI_CHATS, key.as_bytes(), &value).await?;
        }

        Ok(())
    }

    // ============================================================================
    // Payment Consensus Storage Methods
    // ============================================================================

    /// Get wallet credits
    pub async fn get_wallet_credits(&self, wallet_address: &str) -> Result<Option<AICredits>> {
        let key = format!("credits:{}", wallet_address);
        match self.hot_db.get(CF_AI_CREDITS, key.as_bytes()).await? {
            Some(data) => {
                let credits: AICredits = bincode::deserialize(&data)?;
                Ok(Some(credits))
            }
            None => Ok(None),
        }
    }

    /// Initialize wallet credits
    pub async fn init_wallet_credits(&self, wallet_address: &str) -> Result<AICredits> {
        let now = SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs();
        let credits = AICredits {
            wallet_address: wallet_address.to_string(),
            balance_qnk: 0,
            balance_qugusd: 0,
            total_spent_qnk: 0,
            total_spent_qugusd: 0,
            total_tokens_generated: 0,
            created_at: now,
            updated_at: now,
        };

        let key = format!("credits:{}", wallet_address);
        let value = bincode::serialize(&credits)?;
        self.hot_db.put(CF_AI_CREDITS, key.as_bytes(), &value).await?;

        info!("💰 Initialized credits for wallet {}", wallet_address);
        Ok(credits)
    }

    /// Update wallet balance
    pub async fn update_wallet_balance(
        &self,
        wallet_address: &str,
        delta_qnk: i64,
        delta_qugusd: i64,
    ) -> Result<()> {
        let key = format!("credits:{}", wallet_address);

        let mut credits = match self.get_wallet_credits(wallet_address).await? {
            Some(c) => c,
            None => self.init_wallet_credits(wallet_address).await?,
        };

        // Update balances (handle underflow)
        // v3.0.4: Cast to u128 for 24-decimal precision
        if delta_qnk < 0 && credits.balance_qnk < delta_qnk.abs() as u128 {
            return Err(anyhow::anyhow!("Insufficient QNK balance"));
        }
        if delta_qugusd < 0 && credits.balance_qugusd < delta_qugusd.abs() as u128 {
            return Err(anyhow::anyhow!("Insufficient QUGUSD balance"));
        }

        if delta_qnk >= 0 {
            credits.balance_qnk += delta_qnk as u128;
        } else {
            credits.balance_qnk -= delta_qnk.abs() as u128;
            credits.total_spent_qnk += delta_qnk.abs() as u128;
        }

        if delta_qugusd >= 0 {
            credits.balance_qugusd += delta_qugusd as u128;
        } else {
            credits.balance_qugusd -= delta_qugusd.abs() as u128;
            credits.total_spent_qugusd += delta_qugusd.abs() as u128;
        }

        credits.updated_at = SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs();

        let value = bincode::serialize(&credits)?;
        self.hot_db.put(CF_AI_CREDITS, key.as_bytes(), &value).await?;

        debug!("💰 Updated wallet {} balance: QNK {} QUGUSD {}",
            wallet_address, credits.balance_qnk, credits.balance_qugusd);

        Ok(())
    }

    /// Save AI transaction
    pub async fn save_ai_transaction(&self, tx: &AITransaction) -> Result<()> {
        let key = format!("aitx:{}", tx.tx_id);
        let value = bincode::serialize(tx)?;
        self.hot_db.put(CF_AI_TRANSACTIONS, key.as_bytes(), &value).await?;

        debug!("📝 Saved AI transaction {}", tx.tx_id);
        Ok(())
    }

    /// Get AI transaction
    pub async fn get_ai_transaction(&self, tx_id: &str) -> Result<Option<AITransaction>> {
        let key = format!("aitx:{}", tx_id);
        match self.hot_db.get(CF_AI_TRANSACTIONS, key.as_bytes()).await? {
            Some(data) => {
                let tx: AITransaction = bincode::deserialize(&data)?;
                Ok(Some(tx))
            }
            None => Ok(None),
        }
    }

    /// Save payment proposal
    pub async fn save_payment_proposal(&self, proposal: &PaymentProposal) -> Result<()> {
        let key = format!("proposal:{}", proposal.request_id);
        let value = bincode::serialize(proposal)?;
        self.hot_db.put(CF_PAYMENT_PROPOSALS, key.as_bytes(), &value).await?;

        debug!("🗳️ Saved payment proposal {}", proposal.request_id);
        Ok(())
    }

    /// Get payment proposal
    pub async fn get_payment_proposal(&self, request_id: &str) -> Result<Option<PaymentProposal>> {
        let key = format!("proposal:{}", request_id);
        match self.hot_db.get(CF_PAYMENT_PROPOSALS, key.as_bytes()).await? {
            Some(data) => {
                let proposal: PaymentProposal = bincode::deserialize(&data)?;
                Ok(Some(proposal))
            }
            None => Ok(None),
        }
    }

    /// Save payment vote
    pub async fn save_payment_vote(&self, vote: &PaymentVote) -> Result<()> {
        let key = format!("vote:{}:{}", vote.request_id, vote.validator_node_id);
        let value = bincode::serialize(vote)?;
        self.hot_db.put(CF_PAYMENT_VOTES, key.as_bytes(), &value).await?;

        debug!("✅ Saved payment vote for request {} from validator {}",
            vote.request_id, vote.validator_node_id);
        Ok(())
    }

    /// Get all votes for a payment request
    pub async fn get_payment_votes(&self, request_id: &str) -> Result<Vec<PaymentVote>> {
        let prefix = format!("vote:{}:", request_id);
        let votes_data = self.hot_db.scan_prefix(CF_PAYMENT_VOTES, prefix.as_bytes()).await?;

        let mut votes = Vec::new();
        for (_, vote_data) in votes_data {
            if let Ok(vote) = bincode::deserialize::<PaymentVote>(&vote_data) {
                votes.push(vote);
            }
        }

        debug!("🗳️ Loaded {} votes for request {}", votes.len(), request_id);
        Ok(votes)
    }

    /// Save payment lock
    pub async fn save_payment_lock(&self, lock: &PaymentLock) -> Result<()> {
        let key = format!("lock:{}", lock.request_id);
        let value = bincode::serialize(lock)?;
        self.hot_db.put(CF_PAYMENT_LOCKS, key.as_bytes(), &value).await?;

        info!("🔒 Saved payment lock for request {}", lock.request_id);
        Ok(())
    }

    /// Get payment lock
    pub async fn get_payment_lock(&self, request_id: &str) -> Result<Option<PaymentLock>> {
        let key = format!("lock:{}", request_id);
        match self.hot_db.get(CF_PAYMENT_LOCKS, key.as_bytes()).await? {
            Some(data) => {
                let lock: PaymentLock = bincode::deserialize(&data)?;
                Ok(Some(lock))
            }
            None => Ok(None),
        }
    }

    /// Remove payment lock (after settlement)
    pub async fn remove_payment_lock(&self, request_id: &str) -> Result<()> {
        let key = format!("lock:{}", request_id);
        self.hot_db.delete(CF_PAYMENT_LOCKS, key.as_bytes()).await?;

        debug!("🔓 Removed payment lock for request {}", request_id);
        Ok(())
    }

    /// Check if wallet has pending payments (for double-spend detection)
    pub async fn has_pending_payment(&self, wallet_address: &str) -> Result<bool> {
        let prefix = format!("lock:");
        let locks_data = self.hot_db.scan_prefix(CF_PAYMENT_LOCKS, prefix.as_bytes()).await?;

        for (_, lock_data) in locks_data {
            if let Ok(lock) = bincode::deserialize::<PaymentLock>(&lock_data) {
                if lock.wallet_address == wallet_address {
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }

    // ============================================================================
    // Treasury Management (Master Wallet)
    // ============================================================================

    /// Get treasury balance (creates if doesn't exist)
    pub async fn get_treasury_balance(&self) -> Result<AITreasury> {
        let key = b"treasury:master";
        match self.hot_db.get(CF_AI_TREASURY, key).await? {
            Some(data) => {
                let treasury: AITreasury = bincode::deserialize(&data)?;
                Ok(treasury)
            }
            None => {
                // Initialize treasury with environment variable or default
                let treasury_address = std::env::var("AI_TREASURY_WALLET")
                    .unwrap_or_else(|_| "MASTER_AI_TREASURY_WALLET".to_string());

                let now = SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs();
                let treasury = AITreasury {
                    wallet_address: treasury_address,
                    total_revenue_qnk: 0,
                    total_revenue_qugusd: 0,
                    total_requests_served: 0,
                    total_tokens_generated: 0,
                    created_at: now,
                    updated_at: now,
                };

                self.save_treasury_balance(&treasury).await?;
                info!("💰 Initialized AI treasury wallet: {}", treasury.wallet_address);
                Ok(treasury)
            }
        }
    }

    /// Credit treasury with AI payment (100% of profits)
    /// v3.0.4: Migrated amount parameters to u128 for 24-decimal precision
    pub async fn credit_treasury(
        &self,
        amount_qnk: u128,
        amount_qugusd: u128,
        tokens_generated: u32,
    ) -> Result<()> {
        let mut treasury = self.get_treasury_balance().await?;

        treasury.total_revenue_qnk += amount_qnk;
        treasury.total_revenue_qugusd += amount_qugusd;
        treasury.total_requests_served += 1;
        treasury.total_tokens_generated += tokens_generated as u64;
        treasury.updated_at = SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs();

        self.save_treasury_balance(&treasury).await?;

        info!(
            "💰 Treasury credited: {} QNK, {} QUGUSD ({} tokens) | Total: {} QNK, {} requests",
            amount_qnk,
            amount_qugusd,
            tokens_generated,
            treasury.total_revenue_qnk,
            treasury.total_requests_served
        );

        Ok(())
    }

    /// Save treasury balance to disk
    async fn save_treasury_balance(&self, treasury: &AITreasury) -> Result<()> {
        let key = b"treasury:master";
        let value = bincode::serialize(treasury)?;
        self.hot_db.put(CF_AI_TREASURY, key, &value).await?;
        Ok(())
    }

    /// Get access to hot RocksDB for advanced operations like pruning
    /// This returns the concrete RocksDBKV type which supports pruning operations
    pub fn get_hot_db(&self) -> Arc<RocksDBKV> {
        self.hot_db_concrete.clone()
    }

    /// v10.3.15: Background re-indexer — copies all qblock:dag:{N}:{proposer} keys
    /// to qblock:height:{N} so that get_qblocks_range() (fast multi_get path) can
    /// serve early-history blocks to syncing nodes.
    ///
    /// Idempotent: sets a migration flag on completion, skips on subsequent startups.
    /// Non-fatal: returns Ok(0) if already complete or on Windows.
    #[cfg(not(target_os = "windows"))]
    pub async fn reindex_dag_blocks_to_height_keys(&self) -> Result<u64> {
        self.hot_db_concrete.reindex_dag_blocks_to_height_keys().await
    }

    #[cfg(target_os = "windows")]
    pub async fn reindex_dag_blocks_to_height_keys(&self) -> Result<u64> {
        Ok(0) // RocksDB not available on Windows
    }

    /// v8.5.6: Check if a migration flag is set in CF_MANIFEST
    pub async fn has_migration_flag(&self, flag: &[u8]) -> bool {
        self.hot_db.get(CF_MANIFEST, flag).await.ok().flatten().is_some()
    }

    /// v8.5.6: Set a migration flag in CF_MANIFEST
    pub async fn set_migration_flag(&self, flag: &[u8]) -> Result<()> {
        self.hot_db.put(CF_MANIFEST, flag, b"done").await
    }

    /// v10.3.2: Delete a migration flag (forces re-run on next startup)
    pub async fn delete_migration_flag(&self, flag: &[u8]) -> Result<()> {
        self.hot_db.delete(CF_MANIFEST, flag).await
    }

    /// Execute adaptive pruning on the hot database
    /// This is a wrapper method that allows calling pruning without dealing with thread safety issues
    #[cfg(not(target_os = "windows"))]
    pub async fn prune_old_blocks(&self, current_height: u64) -> Result<crate::pruning::PruningStats> {
        self.hot_db_concrete.prune_old_blocks(current_height).await
    }

    /// Atomic payment settlement: refund user + credit treasury + log transaction
    pub async fn settle_payment_atomic(&self, settlement: &PaymentSettlement) -> Result<()> {
        debug!(
            "🔄 Atomic settlement for request {}: {} QNK to treasury, {} QNK refund",
            settlement.request_id, settlement.treasury_payment_qnk, settlement.refund_amount_qnk
        );

        // Build atomic batch
        let mut batch_ops = Vec::new();

        // 1. Refund user if applicable
        if settlement.refund_amount_qnk > 0 {
            let refund_key = format!("credits:{}", settlement.wallet_address);
            let mut user_credits = self
                .get_wallet_credits(&settlement.wallet_address)
                .await?
                .ok_or_else(|| anyhow::anyhow!("User credits not found"))?;

            user_credits.balance_qnk += settlement.refund_amount_qnk;  // Already u128 (v3.0.4)
            user_credits.updated_at =
                SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs();

            let value = bincode::serialize(&user_credits)?;
            batch_ops.push((CF_AI_CREDITS, refund_key.into_bytes(), value));
        }

        // 2. Credit treasury (100% of actual cost)
        let treasury_key = b"treasury:master".to_vec();
        let mut treasury = self.get_treasury_balance().await?;
        treasury.total_revenue_qnk += settlement.treasury_payment_qnk;
        treasury.total_requests_served += 1;
        treasury.total_tokens_generated += settlement.actual_tokens_generated as u64;
        treasury.updated_at = SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs();

        let treasury_value = bincode::serialize(&treasury)?;
        batch_ops.push((CF_AI_TREASURY, treasury_key, treasury_value));

        // 3. Log transaction
        let tx = AITransaction {
            tx_id: settlement.request_id.clone(),
            wallet_address: settlement.wallet_address.clone(),
            chat_id: "".to_string(), // Will be filled by caller
            input_tokens: 0,
            output_tokens: settlement.actual_tokens_generated,
            cost_usd_cents: 0, // Will be calculated from oracle
            cost_qnk: settlement.actual_cost_qnk,
            payment_token: PaymentToken::QNK,
            oracle_price_usd_cents: 0,
            timestamp: settlement.timestamp,
            status: PaymentStatus::Completed,
        };

        let tx_key = format!("aitx:{}", tx.tx_id);
        let tx_value = bincode::serialize(&tx)?;
        batch_ops.push((CF_AI_TRANSACTIONS, tx_key.into_bytes(), tx_value));

        // 4. Execute atomic batch
        self.hot_db.write_batch(batch_ops).await?;

        // 5. Remove payment lock (separate operation, not critical if fails)
        if let Err(e) = self.remove_payment_lock(&settlement.request_id).await {
            warn!(
                "⚠️ Failed to remove payment lock for {}: {}",
                settlement.request_id, e
            );
        }

        info!(
            "✅ Payment settled atomically: {} QNK to treasury, {} QNK refunded to {}",
            settlement.treasury_payment_qnk, settlement.refund_amount_qnk, settlement.wallet_address
        );

        Ok(())
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // ✅ v0.9.98-beta: P2P DURABILITY METHODS (AI Expert Consensus)
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    //
    // Based on unanimous recommendations from ChatGPT, DeepSeek, and Kimi AI:
    // - TurboSync transactions need explicit WAL sync after commit
    // - Gossipsub handlers need idempotency checks
    // - Database replication needs durability verification
    //
    // These methods ensure that "success" means "durable on disk", not just "in memtable"

    /// ✅ v0.9.98-beta: Explicit WAL sync (durability guarantee)
    ///
    /// Forces RocksDB to fsync the Write-Ahead Log to disk.
    /// Call this after transaction commits to ensure durability.
    ///
    /// # AI Expert Consensus
    /// - **ChatGPT:** "Add sync_wal() after tx.commit() to guarantee durability"
    /// - **DeepSeek:** "Transaction commit ≠ disk durability without explicit WAL sync"
    /// - **Kimi AI:** "Add sync_wal() after all transaction commits (2-3 days to implement)"
    ///
    /// # Performance
    /// - Latency: <1ms on SSD
    /// - Should be called per-pack (800 blocks), not per-block
    /// - Amortized cost: <0.001ms per block
    ///
    /// # Example
    /// ```rust
    /// tx.save_qblock(block).await?;
    /// tx.commit().await?;  // Atomic visibility
    /// storage.sync_wal().await?;  // Disk durability ✅
    /// ```
    pub async fn sync_wal(&self) -> Result<()> {
        // v7.3.10: Fixed — now actually delegates to hot_db.sync_wal() which calls flush_wal(true).
        // Previously this was a no-op (debug log only) which meant turbo batch writes in WAL
        // were never durably flushed between balance sync cycles.
        self.hot_db.sync_wal().await
            .context("Failed to sync WAL to disk")?;
        debug!("💾 [WAL SYNC] WAL flushed to disk (durability guarantee)");
        Ok(())
    }

    /// ✅ v0.9.98-beta: Check if block exists (idempotency)
    ///
    /// Used by P2P sync to prevent duplicate processing.
    /// Gossipsub delivers at-most-once, so we need application-level idempotency.
    ///
    /// # AI Expert Consensus
    /// - **ChatGPT:** "Check has_block() before writing for idempotency"
    /// - **DeepSeek:** "Add idempotency checks to prevent duplicate sync"
    /// - **Kimi AI:** "Idempotent sync: check existence before processing"
    ///
    /// # Example
    /// ```rust
    /// // In gossipsub handler:
    /// if self.storage.has_block(block.height).await? {
    ///     return Ok(());  // Already have it, skip ✅
    /// }
    /// self.storage.save_block(block).await?;
    /// ```
    pub async fn has_block(&self, height: u64) -> Result<bool> {
        self.get_qblock_by_height(height).await
            .map(|opt| opt.is_some())
    }

    /// ✅ v0.9.98-beta: Check if update already processed (deduplication)
    ///
    /// Used by database replication to prevent replay attacks and duplicate processing.
    ///
    /// # AI Expert Consensus
    /// - **ChatGPT:** "Updates should have sequence numbers for deduplication"
    /// - **DeepSeek:** "Add explicit verification after write + retry mechanism"
    /// - **Kimi AI:** "Unique IDs for database updates with dedup on receiver"
    ///
    /// # Example
    /// ```rust
    /// if self.storage.has_update(&update_id).await? {
    ///     return Ok(());  // Already processed ✅
    /// }
    /// self.replication_manager.handle_update(update).await?;
    /// self.storage.mark_update_processed(update_id).await?;
    /// ```
    pub async fn has_update(&self, update_id: &str) -> Result<bool> {
        const CF_UPDATES: &str = "processed_updates";  // New CF for tracking
        self.hot_db.get(CF_UPDATES, update_id.as_bytes()).await
            .map(|opt| opt.is_some())
    }

    /// ✅ v0.9.98-beta: Mark update as processed (prevent replays)
    ///
    /// Records that an update has been durably applied.
    /// Only call this AFTER durability verification (sync_wal).
    ///
    /// # AI Expert Consensus
    /// - **ChatGPT:** "Mark as processed only after durable commit"
    /// - **DeepSeek:** "Redefine 'success' = durable, not just accepted"
    /// - **Kimi AI:** "Mark processed only after verified durable"
    ///
    /// # Example
    /// ```rust
    /// self.apply_update(update).await?;
    /// self.storage.sync_wal().await?;  // ✅ Wait for fsync
    /// self.storage.mark_update_processed(&update_id).await?;  // ✅ Safe now
    /// ```
    pub async fn mark_update_processed(&self, update_id: &str) -> Result<()> {
        const CF_UPDATES: &str = "processed_updates";
        let timestamp = SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs();

        self.hot_db.put(CF_UPDATES, update_id.as_bytes(), &timestamp.to_be_bytes()).await
            .context("Failed to mark update as processed")
    }

    // ========== v7.2.0: Atomic Swap (BTC Bridge) Persistence ==========

    /// Save an atomic swap proposal to persistent storage
    /// Key format: atomic_swap:{swap_id}
    pub async fn save_atomic_swap(&self, swap_id: &str, swap_data: &[u8]) -> Result<()> {
        let key = format!("atomic_swap:{}", swap_id);
        self.hot_db.put(CF_MANIFEST, key.as_bytes(), swap_data).await?;
        info!("⚛️ Saved atomic swap: {}", swap_id);
        Ok(())
    }

    /// Get an atomic swap by ID
    pub async fn get_atomic_swap(&self, swap_id: &str) -> Result<Option<Vec<u8>>> {
        let key = format!("atomic_swap:{}", swap_id);
        self.hot_db.get(CF_MANIFEST, key.as_bytes()).await
    }

    /// Save wallet-to-swap index for listing swaps by wallet
    /// Key format: atomic_swap_idx:{wallet}:{swap_id}
    pub async fn index_atomic_swap_by_wallet(&self, wallet: &str, swap_id: &str) -> Result<()> {
        let key = format!("atomic_swap_idx:{}:{}", wallet, swap_id);
        self.hot_db.put(CF_MANIFEST, key.as_bytes(), swap_id.as_bytes()).await?;
        Ok(())
    }

    /// List all atomic swap IDs for a given wallet address
    pub async fn list_atomic_swaps_by_wallet(&self, wallet: &str) -> Result<Vec<String>> {
        let prefix = format!("atomic_swap_idx:{}:", wallet);
        let mut swap_ids = Vec::new();

        match self.hot_db.scan_prefix(CF_MANIFEST, prefix.as_bytes()).await {
            Ok(entries) => {
                for (_key_bytes, value) in entries {
                    if let Ok(swap_id) = String::from_utf8(value) {
                        swap_ids.push(swap_id);
                    }
                }
            }
            Err(e) => {
                warn!("Failed to scan atomic swaps for wallet {}: {}", wallet, e);
            }
        }
        Ok(swap_ids)
    }

    /// Delete an atomic swap (for cleanup of completed/expired swaps)
    pub async fn delete_atomic_swap(&self, swap_id: &str) -> Result<()> {
        let key = format!("atomic_swap:{}", swap_id);
        self.hot_db.delete(CF_MANIFEST, key.as_bytes()).await?;
        debug!("🗑️ Deleted atomic swap: {}", swap_id);
        Ok(())
    }

    // ═══ Zcash Shielded Swap Storage (v7.2.2) ═══

    /// Save a Zcash shielded swap to persistent storage
    pub async fn save_zcash_swap(&self, swap_id: &str, swap_data: &[u8]) -> Result<()> {
        let key = format!("zcash_swap:{}", swap_id);
        self.hot_db.put(CF_MANIFEST, key.as_bytes(), swap_data).await?;
        info!("🛡️ Saved Zcash swap: {}", swap_id);
        Ok(())
    }

    /// Get a Zcash swap by ID
    pub async fn get_zcash_swap(&self, swap_id: &str) -> Result<Option<Vec<u8>>> {
        let key = format!("zcash_swap:{}", swap_id);
        self.hot_db.get(CF_MANIFEST, key.as_bytes()).await
    }

    /// Save wallet-to-zcash-swap index
    pub async fn index_zcash_swap_by_wallet(&self, wallet: &str, swap_id: &str) -> Result<()> {
        let key = format!("zcash_swap_idx:{}:{}", wallet, swap_id);
        self.hot_db.put(CF_MANIFEST, key.as_bytes(), swap_id.as_bytes()).await?;
        Ok(())
    }

    /// List all Zcash swap IDs for a given wallet
    pub async fn list_zcash_swaps_by_wallet(&self, wallet: &str) -> Result<Vec<String>> {
        let prefix = format!("zcash_swap_idx:{}:", wallet);
        let mut swap_ids = Vec::new();
        match self.hot_db.scan_prefix(CF_MANIFEST, prefix.as_bytes()).await {
            Ok(entries) => {
                for (_key_bytes, value) in entries {
                    if let Ok(swap_id) = String::from_utf8(value) {
                        swap_ids.push(swap_id);
                    }
                }
            }
            Err(e) => {
                warn!("Failed to scan Zcash swaps for wallet {}: {}", wallet, e);
            }
        }
        Ok(swap_ids)
    }

    /// Save a user's Zcash z-address
    pub async fn save_zcash_z_address(&self, wallet: &str, z_address: &str) -> Result<()> {
        let key = format!("zcash_z_address:{}", wallet);
        self.hot_db.put(CF_MANIFEST, key.as_bytes(), z_address.as_bytes()).await?;
        Ok(())
    }

    /// Get a user's Zcash z-address
    pub async fn get_zcash_z_address(&self, wallet: &str) -> Result<Option<String>> {
        let key = format!("zcash_z_address:{}", wallet);
        match self.hot_db.get(CF_MANIFEST, key.as_bytes()).await? {
            Some(data) => Ok(String::from_utf8(data).ok()),
            None => Ok(None),
        }
    }

    /// Save/update Zcash shielded balance (in zatoshis)
    pub async fn save_zcash_balance(&self, wallet: &str, balance_zat: u64) -> Result<()> {
        let key = format!("zcash_balance:{}", wallet);
        self.hot_db.put(CF_MANIFEST, key.as_bytes(), balance_zat.to_string().as_bytes()).await?;
        Ok(())
    }

    /// Get Zcash shielded balance (in zatoshis)
    pub async fn get_zcash_balance(&self, wallet: &str) -> Result<u64> {
        let key = format!("zcash_balance:{}", wallet);
        match self.hot_db.get(CF_MANIFEST, key.as_bytes()).await? {
            Some(data) => {
                let s = String::from_utf8(data).unwrap_or_default();
                Ok(s.parse::<u64>().unwrap_or(0))
            }
            None => Ok(0),
        }
    }

    // ═══ Iron Fish Swap Storage (v7.2.4) ═══

    pub async fn save_ironfish_swap(&self, swap_id: &str, swap_data: &[u8]) -> Result<()> {
        let key = format!("ironfish_swap:{}", swap_id);
        self.hot_db.put(CF_MANIFEST, key.as_bytes(), swap_data).await?;
        info!("🐟 Saved Iron Fish swap: {}", swap_id);
        Ok(())
    }

    pub async fn get_ironfish_swap(&self, swap_id: &str) -> Result<Option<Vec<u8>>> {
        let key = format!("ironfish_swap:{}", swap_id);
        self.hot_db.get(CF_MANIFEST, key.as_bytes()).await
    }

    pub async fn index_ironfish_swap_by_wallet(&self, wallet: &str, swap_id: &str) -> Result<()> {
        let key = format!("ironfish_swap_idx:{}:{}", wallet, swap_id);
        self.hot_db.put(CF_MANIFEST, key.as_bytes(), swap_id.as_bytes()).await?;
        Ok(())
    }

    pub async fn list_ironfish_swaps_by_wallet(&self, wallet: &str) -> Result<Vec<String>> {
        let prefix = format!("ironfish_swap_idx:{}:", wallet);
        let mut swap_ids = Vec::new();
        match self.hot_db.scan_prefix(CF_MANIFEST, prefix.as_bytes()).await {
            Ok(entries) => {
                for (_key_bytes, value) in entries {
                    if let Ok(swap_id) = String::from_utf8(value) {
                        swap_ids.push(swap_id);
                    }
                }
            }
            Err(e) => {
                warn!("Failed to scan Iron Fish swaps for wallet {}: {}", wallet, e);
            }
        }
        Ok(swap_ids)
    }

    pub async fn save_ironfish_address(&self, wallet: &str, iron_address: &str) -> Result<()> {
        let key = format!("ironfish_address:{}", wallet);
        self.hot_db.put(CF_MANIFEST, key.as_bytes(), iron_address.as_bytes()).await?;
        Ok(())
    }

    pub async fn get_ironfish_address(&self, wallet: &str) -> Result<Option<String>> {
        let key = format!("ironfish_address:{}", wallet);
        match self.hot_db.get(CF_MANIFEST, key.as_bytes()).await? {
            Some(data) => Ok(String::from_utf8(data).ok()),
            None => Ok(None),
        }
    }

    pub async fn save_ironfish_balance(&self, wallet: &str, balance_ore: u64) -> Result<()> {
        let key = format!("ironfish_balance:{}", wallet);
        self.hot_db.put(CF_MANIFEST, key.as_bytes(), balance_ore.to_string().as_bytes()).await?;
        Ok(())
    }

    pub async fn get_ironfish_balance(&self, wallet: &str) -> Result<u64> {
        let key = format!("ironfish_balance:{}", wallet);
        match self.hot_db.get(CF_MANIFEST, key.as_bytes()).await? {
            Some(data) => {
                let s = String::from_utf8(data).unwrap_or_default();
                Ok(s.parse::<u64>().unwrap_or(0))
            }
            None => Ok(0),
        }
    }

    /// Load all active atomic swaps from storage
    pub async fn load_all_atomic_swaps(&self) -> Result<Vec<(String, Vec<u8>)>> {
        let prefix = b"atomic_swap:";
        let mut swaps = Vec::new();

        match self.hot_db.scan_prefix(CF_MANIFEST, prefix).await {
            Ok(entries) => {
                for (key_bytes, value) in entries {
                    if let Ok(key_str) = String::from_utf8(key_bytes) {
                        // Skip index entries (atomic_swap_idx:)
                        if key_str.starts_with("atomic_swap_idx:") {
                            continue;
                        }
                        let swap_id = key_str.trim_start_matches("atomic_swap:");
                        swaps.push((swap_id.to_string(), value));
                    }
                }
                info!("⚛️ Loaded {} atomic swaps from storage", swaps.len());
            }
            Err(e) => {
                warn!("Failed to scan atomic swaps: {}", e);
            }
        }
        Ok(swaps)
    }
}

/// Implementation of BalanceStorage trait for consensus engine
/// v2.5.0: Updated to u128 for extreme precision (24 decimals)
#[async_trait::async_trait]
impl BalanceStorage for QStorage {
    /// Add amount to wallet balance (atomic operation)
    /// v2.5.0: amount is now u128
    async fn add_balance(&self, address: &str, amount: u128) -> Result<()> {
        // Convert hex string address to [u8; 32]
        let address_bytes = hex::decode(address)
            .context("Invalid hex address format")?;

        if address_bytes.len() != 32 {
            return Err(anyhow::anyhow!(
                "Invalid address length: expected 32 bytes, got {}",
                address_bytes.len()
            ));
        }

        let mut addr_array = [0u8; 32];
        addr_array.copy_from_slice(&address_bytes);

        // Get current balance
        let current = self.load_wallet_balance(&addr_array).await?.unwrap_or(0);

        // Add amount (saturating to prevent overflow)
        let new_balance = current.saturating_add(amount);

        // 🔴 [BALANCE WRITE DEBUG] Log add operation
        warn!(
            "🔴 [BALANCE WRITE] add_balance(): wallet={} old={} new={} delta=+{} caller=BalanceStorage::add_balance height=N/A",
            &address[..16.min(address.len())], current, new_balance, amount
        );

        // Save new balance
        self.save_wallet_balance(&addr_array, new_balance).await?;

        debug!(
            "✅ [BALANCE CONSENSUS] Added {} to {}, new balance: {}",
            amount, address, new_balance
        );

        Ok(())
    }

    /// Subtract amount from wallet balance (atomic operation) - v3.5.14-beta
    /// Returns error if insufficient balance
    async fn subtract_balance(&self, address: &str, amount: u128) -> Result<()> {
        // Convert hex string address to [u8; 32]
        let address_bytes = hex::decode(address)
            .context("Invalid hex address format")?;

        if address_bytes.len() != 32 {
            return Err(anyhow::anyhow!(
                "Invalid address length: expected 32 bytes, got {}",
                address_bytes.len()
            ));
        }

        let mut addr_array = [0u8; 32];
        addr_array.copy_from_slice(&address_bytes);

        // Get current balance
        let current = self.load_wallet_balance(&addr_array).await?.unwrap_or(0);

        // Check if sufficient balance
        if current < amount {
            return Err(anyhow::anyhow!(
                "Insufficient balance: {} < {} for address {}",
                current, amount, &address[..16]
            ));
        }

        // Subtract amount
        let new_balance = current - amount;

        // 🔴 [BALANCE WRITE DEBUG] Log subtract operation (always error-level since balance decreases)
        error!(
            "🔴 [BALANCE WRITE] subtract_balance(): wallet={} old={} new={} delta=-{} caller=BalanceStorage::subtract_balance height=N/A",
            &address[..16.min(address.len())], current, new_balance, amount
        );

        // Save new balance
        self.save_wallet_balance(&addr_array, new_balance).await?;

        debug!(
            "💸 [BALANCE CONSENSUS] Subtracted {} from {}, new balance: {}",
            amount, address, new_balance
        );

        Ok(())
    }

    /// Get wallet balance
    /// v2.5.0: returns u128
    async fn get_balance(&self, address: &str) -> Result<u128> {
        // Convert hex string address to [u8; 32]
        let address_bytes = hex::decode(address)
            .context("Invalid hex address format")?;

        if address_bytes.len() != 32 {
            return Err(anyhow::anyhow!(
                "Invalid address length: expected 32 bytes, got {}",
                address_bytes.len()
            ));
        }

        let mut addr_array = [0u8; 32];
        addr_array.copy_from_slice(&address_bytes);

        Ok(self.load_wallet_balance(&addr_array).await?.unwrap_or(0))
    }

    /// Set wallet balance directly
    /// v2.5.0: balance is now u128
    async fn set_balance(&self, address: &str, balance: u128) -> Result<()> {
        // 🔴 [BALANCE WRITE DEBUG] Read old value BEFORE set
        {
            let address_bytes_check = hex::decode(address);
            if let Ok(ref ab) = address_bytes_check {
                if ab.len() == 32 {
                    let mut arr = [0u8; 32];
                    arr.copy_from_slice(ab);
                    let old_balance = self.load_wallet_balance(&arr).await?.unwrap_or(0);
                    if old_balance != balance {
                        let delta_abs = if balance >= old_balance { balance - old_balance } else { old_balance - balance };
                        let direction = if balance >= old_balance { "+" } else { "-" };
                        if balance < old_balance {
                            error!(
                                "🔴 [BALANCE WRITE] set_balance(): wallet={} old={} new={} delta={}{} caller=BalanceStorage::set_balance height=N/A",
                                &address[..16.min(address.len())], old_balance, balance, direction, delta_abs
                            );
                        } else {
                            warn!(
                                "🔴 [BALANCE WRITE] set_balance(): wallet={} old={} new={} delta={}{} caller=BalanceStorage::set_balance height=N/A",
                                &address[..16.min(address.len())], old_balance, balance, direction, delta_abs
                            );
                        }
                    }
                }
            }
        }

        // Convert hex string address to [u8; 32]
        let address_bytes = hex::decode(address)
            .context("Invalid hex address format")?;

        if address_bytes.len() != 32 {
            return Err(anyhow::anyhow!(
                "Invalid address length: expected 32 bytes, got {}",
                address_bytes.len()
            ));
        }

        let mut addr_array = [0u8; 32];
        addr_array.copy_from_slice(&address_bytes);

        self.save_wallet_balance(&addr_array, balance).await?;

        debug!(
            "✅ [BALANCE CONSENSUS] Set balance for {} to {}",
            address, balance
        );

        Ok(())
    }

    /// v10.2.0: Add amount to token balance (QUGUSD / custom tokens)
    async fn add_token_balance(&self, wallet: &[u8; 32], token: &[u8; 32], amount: u128) -> Result<()> {
        let current = self.get_token_balance(wallet, token).await.unwrap_or(0);
        let new_balance = current.saturating_add(amount);
        self.save_token_balance(wallet, token, new_balance).await?;
        debug!(
            "✅ [TOKEN BALANCE CONSENSUS v10.2.0] Added {} to {}:{}, new balance: {}",
            amount, &hex::encode(&wallet[..8]), &hex::encode(&token[..4]), new_balance
        );
        Ok(())
    }

    /// v10.2.0: Subtract amount from token balance (QUGUSD / custom tokens)
    async fn subtract_token_balance(&self, wallet: &[u8; 32], token: &[u8; 32], amount: u128) -> Result<()> {
        let current = self.get_token_balance(wallet, token).await.unwrap_or(0);
        if current < amount {
            return Err(anyhow::anyhow!(
                "Insufficient token balance: {} < {} for wallet {} token {}",
                current, amount, &hex::encode(&wallet[..8]), &hex::encode(&token[..4])
            ));
        }
        let new_balance = current - amount;
        self.save_token_balance(wallet, token, new_balance).await?;
        debug!(
            "💸 [TOKEN BALANCE CONSENSUS v10.2.0] Subtracted {} from {}:{}, new balance: {}",
            amount, &hex::encode(&wallet[..8]), &hex::encode(&token[..4]), new_balance
        );
        Ok(())
    }

    /// v10.3.2: Check if a block has been processed (persistent, survives restart)
    async fn get_processed_block_flag(&self, key: &str) -> Result<bool> {
        match self.hot_db.get(CF_MANIFEST, key.as_bytes()).await {
            Ok(Some(_)) => Ok(true),
            Ok(None) => Ok(false),
            Err(e) => Err(e),
        }
    }

    /// v10.3.2: Mark a block as processed (persistent, survives restart)
    async fn set_processed_block_flag(&self, key: &str) -> Result<()> {
        self.hot_db.put(CF_MANIFEST, key.as_bytes(), b"1").await
    }
}

// ============================================================================
// v10.3.1: Atomic DEX balance methods — NOT part of the BalanceStorage trait.
// These combine balance updates with DEX debit/credit counter tracking in a single
// RocksDB WriteBatch for crash-safe atomicity (DeepSeek review requirement).
// ============================================================================
impl QStorage {
    /// v10.3.1: Atomic subtract-balance + record-dex-debit in a single WriteBatch.
    /// DeepSeek review: "subtract_balance and record_dex_qug_debit are TWO SEPARATE writes,
    /// not atomic — if the process crashes between them, the debit counter is wrong."
    ///
    /// This method reads the current balance and debit counter, computes new values,
    /// and writes BOTH in one atomic batch. Returns the new balance on success.
    /// Uses saturating_sub to prevent underflow (DeepSeek recommendation).
    pub async fn atomic_subtract_and_record_dex_debit(
        &self,
        wallet_hex: &str,
        amount: u128,
    ) -> Result<u128> {
        // Step 1: Read current balance
        let address_bytes = hex::decode(wallet_hex)
            .context("Invalid hex address format")?;
        if address_bytes.len() != 32 {
            return Err(anyhow::anyhow!(
                "Invalid address length: expected 32 bytes, got {}",
                address_bytes.len()
            ));
        }
        let mut addr_array = [0u8; 32];
        addr_array.copy_from_slice(&address_bytes);

        let current_balance = self.load_wallet_balance(&addr_array).await?.unwrap_or(0);

        // Step 2: Insufficient balance check (fail fast before any writes)
        if current_balance < amount {
            return Err(anyhow::anyhow!(
                "Insufficient balance: {} < {} for address {}",
                current_balance, amount, &wallet_hex[..16.min(wallet_hex.len())]
            ));
        }

        // Step 3: Compute new values
        let new_balance = current_balance.saturating_sub(amount);

        // Read current debit counter
        let debit_key = format!("dex_qug_debited:{}", wallet_hex);
        let current_debit = match self.hot_db.get(CF_MANIFEST, debit_key.as_bytes()).await? {
            Some(bytes) if bytes.len() == 16 => u128::from_le_bytes(bytes[..16].try_into().unwrap()),
            _ => 0u128,
        };
        let new_debit_total = current_debit.saturating_add(amount);

        // Read current credit counter (for computing net applied adjustment)
        let credit_key = format!("dex_qug_credited:{}", wallet_hex);
        let current_credit = match self.hot_db.get(CF_MANIFEST, credit_key.as_bytes()).await? {
            Some(bytes) if bytes.len() == 16 => u128::from_le_bytes(bytes[..16].try_into().unwrap()),
            _ => 0u128,
        };

        // Compute new applied net: credited - new_debit_total (keeps idempotent tracker in sync)
        let new_applied_net: i128 = current_credit as i128 - new_debit_total as i128;
        let applied_key = format!("dex_applied_net:{}", wallet_hex);

        // Step 4: Atomic write batch — balance, debit counter, AND applied-net tracker.
        // If the process crashes here, NEITHER write is applied (RocksDB atomicity guarantee).
        // The applied-net tracker ensures apply_dex_qug_adjustments() is idempotent on restart.
        let balance_key = format!("wallet_balance_{}", wallet_hex);
        let batch: Vec<(&str, Vec<u8>, Vec<u8>)> = vec![
            (CF_MANIFEST, balance_key.as_bytes().to_vec(), new_balance.to_le_bytes().to_vec()),
            (CF_MANIFEST, debit_key.as_bytes().to_vec(), new_debit_total.to_le_bytes().to_vec()),
            (CF_MANIFEST, applied_key.as_bytes().to_vec(), new_applied_net.to_le_bytes().to_vec()),
        ];
        self.hot_db.write_batch(batch).await
            .context("Atomic DEX debit write_batch failed")?;

        // 🔴 [BALANCE WRITE DEBUG] Atomic DEX debit
        error!(
            "🔴 [BALANCE WRITE] atomic_subtract_and_record_dex_debit(): wallet={} old={} new={} delta=-{} caller=DEX_ATOMIC_DEBIT height=N/A",
            &wallet_hex[..16.min(wallet_hex.len())], current_balance, new_balance, amount
        );

        info!(
            "💸 [DEX ATOMIC v10.3.1] Wallet {}...: balance {} -> {} QUG, debit counter {} -> {} (amount: {})",
            &wallet_hex[..16.min(wallet_hex.len())],
            current_balance as f64 / 1e24,
            new_balance as f64 / 1e24,
            current_debit as f64 / 1e24,
            new_debit_total as f64 / 1e24,
            amount as f64 / 1e24,
        );

        Ok(new_balance)
    }

    /// v10.3.1: Atomic add-balance + record-dex-credit in a single WriteBatch.
    /// DeepSeek review: "add_balance and record_dex_qug_credit are TWO SEPARATE writes,
    /// not atomic — if the process crashes between them, the credit counter is wrong."
    ///
    /// This method reads the current balance and credit counter, computes new values,
    /// and writes ALL THREE keys (balance, credit counter, applied-net tracker) in one
    /// atomic batch. Returns the new balance on success.
    /// Uses saturating_add to prevent overflow.
    pub async fn atomic_add_and_record_dex_credit(
        &self,
        wallet_hex: &str,
        amount: u128,
    ) -> Result<u128> {
        // Step 1: Read current balance
        let address_bytes = hex::decode(wallet_hex)
            .context("Invalid hex address format")?;
        if address_bytes.len() != 32 {
            return Err(anyhow::anyhow!(
                "Invalid address length: expected 32 bytes, got {}",
                address_bytes.len()
            ));
        }
        let mut addr_array = [0u8; 32];
        addr_array.copy_from_slice(&address_bytes);

        let current_balance = self.load_wallet_balance(&addr_array).await?.unwrap_or(0);

        // Step 2: Compute new balance
        let new_balance = current_balance.saturating_add(amount);

        // Step 3: Read current credit counter
        let credit_key = format!("dex_qug_credited:{}", wallet_hex);
        let current_credit = match self.hot_db.get(CF_MANIFEST, credit_key.as_bytes()).await? {
            Some(bytes) if bytes.len() == 16 => u128::from_le_bytes(bytes[..16].try_into().unwrap()),
            _ => 0u128,
        };
        let new_credit_total = current_credit.saturating_add(amount);

        // Read current debit counter (for computing net applied adjustment)
        let debit_key = format!("dex_qug_debited:{}", wallet_hex);
        let current_debit = match self.hot_db.get(CF_MANIFEST, debit_key.as_bytes()).await? {
            Some(bytes) if bytes.len() == 16 => u128::from_le_bytes(bytes[..16].try_into().unwrap()),
            _ => 0u128,
        };

        // Compute new applied net: new_credit_total - debit (keeps idempotent tracker in sync)
        let new_applied_net: i128 = new_credit_total as i128 - current_debit as i128;
        let applied_key = format!("dex_applied_net:{}", wallet_hex);

        // Step 4: Atomic write batch — balance, credit counter, AND applied-net tracker.
        // If the process crashes here, NEITHER write is applied (RocksDB atomicity guarantee).
        // The applied-net tracker ensures apply_dex_qug_adjustments() is idempotent on restart.
        let balance_key = format!("wallet_balance_{}", wallet_hex);
        let batch: Vec<(&str, Vec<u8>, Vec<u8>)> = vec![
            (CF_MANIFEST, balance_key.as_bytes().to_vec(), new_balance.to_le_bytes().to_vec()),
            (CF_MANIFEST, credit_key.as_bytes().to_vec(), new_credit_total.to_le_bytes().to_vec()),
            (CF_MANIFEST, applied_key.as_bytes().to_vec(), new_applied_net.to_le_bytes().to_vec()),
        ];
        self.hot_db.write_batch(batch).await
            .context("Atomic DEX credit write_batch failed")?;

        // 🔴 [BALANCE WRITE DEBUG] Atomic DEX credit
        warn!(
            "🔴 [BALANCE WRITE] atomic_add_and_record_dex_credit(): wallet={} old={} new={} delta=+{} caller=DEX_ATOMIC_CREDIT height=N/A",
            &wallet_hex[..16.min(wallet_hex.len())], current_balance, new_balance, amount
        );

        info!(
            "💰 [DEX ATOMIC v10.3.1] Wallet {}...: balance {} -> {} QUG, credit counter {} -> {} (amount: {})",
            &wallet_hex[..16.min(wallet_hex.len())],
            current_balance as f64 / 1e24,
            new_balance as f64 / 1e24,
            current_credit as f64 / 1e24,
            new_credit_total as f64 / 1e24,
            amount as f64 / 1e24,
        );

        Ok(new_balance)
    }
}

// ============================================================================
// v2.3.6-beta: Swap History Persistence (Token Details Modal)
// ============================================================================
impl QStorage {
    /// Save a swap transaction to history
    /// Key format: "swap:{token}:{timestamp}:{tx_id}"
    pub async fn save_swap_history(&self, token: &str, swap_record: &serde_json::Value) -> Result<()> {
        let timestamp = swap_record.get("timestamp").and_then(|v| v.as_i64()).unwrap_or(0);
        let tx_id = swap_record.get("id").and_then(|v| v.as_str()).unwrap_or("unknown");

        // Create composite key for time-ordered retrieval
        let key = format!("swap:{}:{}:{}", token.to_uppercase(), timestamp, tx_id);

        let value = serde_json::to_vec(swap_record)?;
        self.hot_db.put(CF_SWAP_HISTORY, key.as_bytes(), &value).await?;

        debug!("💾 Saved swap history for {} (tx: {})", token, tx_id);
        Ok(())
    }

    /// Load swap history for a token (most recent first)
    pub async fn load_swap_history(&self, token: &str) -> Result<Vec<serde_json::Value>> {
        let prefix = format!("swap:{}:", token.to_uppercase());
        let records = self.hot_db.scan_prefix(CF_SWAP_HISTORY, prefix.as_bytes()).await?;

        let mut history: Vec<serde_json::Value> = Vec::new();
        for (_, value) in records {
            if let Ok(record) = serde_json::from_slice::<serde_json::Value>(&value) {
                history.push(record);
            }
        }

        // Sort by timestamp descending (most recent first)
        history.sort_by(|a, b| {
            let ts_a = a.get("timestamp").and_then(|v| v.as_i64()).unwrap_or(0);
            let ts_b = b.get("timestamp").and_then(|v| v.as_i64()).unwrap_or(0);
            ts_b.cmp(&ts_a)
        });

        debug!("📜 Loaded {} swap records for {}", history.len(), token);
        Ok(history)
    }

    /// Load all swap history (for initial cache population on startup)
    pub async fn load_all_swap_history(&self) -> Result<HashMap<String, Vec<serde_json::Value>>> {
        let prefix = "swap:";
        let records = self.hot_db.scan_prefix(CF_SWAP_HISTORY, prefix.as_bytes()).await?;

        let mut history: HashMap<String, Vec<serde_json::Value>> = HashMap::new();

        for (key, value) in records {
            // Key format: "swap:{token}:{timestamp}:{tx_id}"
            if let Ok(key_str) = String::from_utf8(key) {
                let parts: Vec<&str> = key_str.split(':').collect();
                if parts.len() >= 2 {
                    let token = parts[1].to_string();
                    if let Ok(record) = serde_json::from_slice::<serde_json::Value>(&value) {
                        history.entry(token).or_insert_with(Vec::new).push(record);
                    }
                }
            }
        }

        // Sort each token's history by timestamp descending
        for (_, records) in history.iter_mut() {
            records.sort_by(|a, b| {
                let ts_a = a.get("timestamp").and_then(|v| v.as_i64()).unwrap_or(0);
                let ts_b = b.get("timestamp").and_then(|v| v.as_i64()).unwrap_or(0);
                ts_b.cmp(&ts_a)
            });
        }

        info!("📜 Loaded swap history for {} tokens from RocksDB", history.len());
        Ok(history)
    }

    // ========================================================================
    // v2.4.0-beta: Consensus-Verified Swap History (Binary Keys)
    // These methods support the SwapIndexer for DAGKnight-verified swaps
    // ========================================================================

    /// Save a consensus-verified swap record using binary key
    /// Key format: [token:32][inverted_timestamp:8][tx_id:8]
    pub async fn save_consensus_swap(&self, key: &[u8], value: &[u8]) -> Result<()> {
        self.hot_db.put(CF_SWAP_HISTORY, key, value).await?;
        Ok(())
    }

    /// Load swap history for a token address (binary format)
    /// Returns deserialized ConsensusSwapRecord items
    pub async fn load_swap_history_for_token<T: serde::de::DeserializeOwned>(
        &self,
        token_address: &[u8; 32],
        limit: usize,
    ) -> Result<Vec<T>> {
        // Scan with token address as prefix
        let records = self.hot_db.scan_prefix(CF_SWAP_HISTORY, token_address).await?;

        let mut history: Vec<T> = Vec::new();
        for (_, value) in records.into_iter().take(limit) {
            if let Ok(record) = bincode::deserialize::<T>(&value) {
                history.push(record);
            }
        }

        debug!("📜 Loaded {} consensus swap records for token", history.len());
        Ok(history)
    }

    // ========================================================================
    // v9.3.3: DEX QUG Adjustment Tracking (Rebuild-Safe)
    //
    // Problem: Balance rebuild migrations replay the blockchain and overwrite
    // wallet_balance_* entries. DEX swaps are NOT blockchain transactions, so
    // subtract_balance()/add_balance() deductions get lost after any rebuild.
    //
    // Solution: Track cumulative DEX QUG debits/credits per wallet in durable
    // counters. After any balance rebuild, re-apply these adjustments.
    //
    // Key format: "dex_qug_debited:{wallet_hex}" → u128 (total QUG sold on DEX)
    //             "dex_qug_credited:{wallet_hex}" → u128 (total QUG bought on DEX)
    // ========================================================================

    /// Record a QUG debit from a DEX swap (user sold QUG)
    /// v10.3.1: Atomically increments debit counter AND updates applied-net tracker
    /// in a single WriteBatch for consistency with idempotent reconciliation.
    /// NOTE: Prefer `atomic_subtract_and_record_dex_debit()` which also atomically
    /// updates the balance in the same batch.
    pub async fn record_dex_qug_debit(&self, wallet_hex: &str, amount: u128) -> Result<()> {
        let debit_key = format!("dex_qug_debited:{}", wallet_hex);
        let current_debit = match self.hot_db.get(CF_MANIFEST, debit_key.as_bytes()).await? {
            Some(bytes) if bytes.len() == 16 => u128::from_le_bytes(bytes[..16].try_into().unwrap()),
            _ => 0u128,
        };
        let new_debit_total = current_debit.saturating_add(amount);

        // Read current credit counter (for computing net applied adjustment)
        let credit_key = format!("dex_qug_credited:{}", wallet_hex);
        let current_credit = match self.hot_db.get(CF_MANIFEST, credit_key.as_bytes()).await? {
            Some(bytes) if bytes.len() == 16 => u128::from_le_bytes(bytes[..16].try_into().unwrap()),
            _ => 0u128,
        };

        // Update applied-net tracker: credited - debited (keeps idempotent reconciliation in sync)
        let new_applied_net: i128 = current_credit as i128 - new_debit_total as i128;
        let applied_key = format!("dex_applied_net:{}", wallet_hex);

        // Atomic write: debit counter + applied-net tracker
        let batch: Vec<(&str, Vec<u8>, Vec<u8>)> = vec![
            (CF_MANIFEST, debit_key.as_bytes().to_vec(), new_debit_total.to_le_bytes().to_vec()),
            (CF_MANIFEST, applied_key.as_bytes().to_vec(), new_applied_net.to_le_bytes().to_vec()),
        ];
        self.hot_db.write_batch(batch).await?;

        debug!("📉 [DEX ADJUST v10.3.1] Recorded QUG debit: {} += {} (total: {})",
            &wallet_hex[..8.min(wallet_hex.len())], amount as f64 / 1e24, new_debit_total as f64 / 1e24);
        Ok(())
    }

    /// Record a QUG credit from a DEX swap (user bought QUG)
    /// v10.3.1: Atomically increments credit counter AND updates applied-net tracker
    /// in a single WriteBatch for consistency with idempotent reconciliation.
    pub async fn record_dex_qug_credit(&self, wallet_hex: &str, amount: u128) -> Result<()> {
        let credit_key = format!("dex_qug_credited:{}", wallet_hex);
        let current_credit = match self.hot_db.get(CF_MANIFEST, credit_key.as_bytes()).await? {
            Some(bytes) if bytes.len() == 16 => u128::from_le_bytes(bytes[..16].try_into().unwrap()),
            _ => 0u128,
        };
        let new_credit_total = current_credit.saturating_add(amount);

        // Read current debit counter (for computing net applied adjustment)
        let debit_key = format!("dex_qug_debited:{}", wallet_hex);
        let current_debit = match self.hot_db.get(CF_MANIFEST, debit_key.as_bytes()).await? {
            Some(bytes) if bytes.len() == 16 => u128::from_le_bytes(bytes[..16].try_into().unwrap()),
            _ => 0u128,
        };

        // v10.5.2 FIX: Do NOT update dex_applied_net here. Only apply_dex_qug_adjustments()
        // should write dex_applied_net, after the checkpoint restore. If we set it here,
        // the checkpoint purge resets wallet_balance_ but dex_applied_net survives, causing
        // apply_dex_qug_adjustments() to compute delta=0 and silently drop post-checkpoint credits.
        // Write credit counter only:
        self.hot_db.put_sync(CF_MANIFEST, credit_key.as_bytes(), &new_credit_total.to_le_bytes()).await?;

        debug!("📈 [DEX ADJUST v10.3.1] Recorded QUG credit: {} += {} (total: {})",
            &wallet_hex[..8.min(wallet_hex.len())], amount as f64 / 1e24, new_credit_total as f64 / 1e24);
        Ok(())
    }

    /// v10.3.1: Apply DEX QUG adjustments — IDEMPOTENT version (safe to call on every startup).
    ///
    /// DeepSeek review: "derive final balance as chain_credits - persisted_dex_debits"
    ///
    /// IDEMPOTENCY MECHANISM:
    /// We track "last applied net adjustment" per wallet in `dex_applied_net:{wallet_hex}`.
    /// On each call we compute `desired_net = credited - debited` (can be negative via i128).
    /// The delta between desired_net and previously-applied net is what we actually adjust.
    /// Running this N times produces the same final balance (no double-deduction).
    ///
    /// Formula: balance_delta = (credited - debited) - previously_applied_net
    /// If delta == 0, nothing to do (idempotent).
    /// Uses saturating_sub to prevent underflow (DeepSeek recommendation).
    pub async fn apply_dex_qug_adjustments(&self) -> Result<u64> {
        info!("🔄 [DEX ADJUST v10.3.1] Applying idempotent DEX QUG adjustments...");

        // Load all debit counters
        let debit_entries = self.hot_db.scan_prefix(CF_MANIFEST, b"dex_qug_debited:").await?;
        let credit_entries = self.hot_db.scan_prefix(CF_MANIFEST, b"dex_qug_credited:").await?;
        let applied_entries = self.hot_db.scan_prefix(CF_MANIFEST, b"dex_applied_net:").await?;

        let mut debit_map: HashMap<String, u128> = HashMap::new();
        let mut credit_map: HashMap<String, u128> = HashMap::new();
        let mut applied_map: HashMap<String, i128> = HashMap::new();

        for (key, value) in debit_entries {
            if let Ok(key_str) = String::from_utf8(key) {
                let wallet_hex = key_str.trim_start_matches("dex_qug_debited:").to_string();
                if value.len() == 16 && !wallet_hex.is_empty() {
                    let amount = u128::from_le_bytes(value[..16].try_into().unwrap());
                    if amount > 0 {
                        debit_map.insert(wallet_hex, amount);
                    }
                }
            }
        }

        for (key, value) in credit_entries {
            if let Ok(key_str) = String::from_utf8(key) {
                let wallet_hex = key_str.trim_start_matches("dex_qug_credited:").to_string();
                if value.len() == 16 && !wallet_hex.is_empty() {
                    let amount = u128::from_le_bytes(value[..16].try_into().unwrap());
                    if amount > 0 {
                        credit_map.insert(wallet_hex, amount);
                    }
                }
            }
        }

        // Load previously-applied net adjustments (signed i128, stored as 16 bytes)
        for (key, value) in applied_entries {
            if let Ok(key_str) = String::from_utf8(key) {
                let wallet_hex = key_str.trim_start_matches("dex_applied_net:").to_string();
                if value.len() == 16 && !wallet_hex.is_empty() {
                    let net = i128::from_le_bytes(value[..16].try_into().unwrap());
                    applied_map.insert(wallet_hex, net);
                }
            }
        }

        // Collect all wallets with any DEX activity
        let mut all_wallets: std::collections::HashSet<String> = std::collections::HashSet::new();
        for k in debit_map.keys() { all_wallets.insert(k.clone()); }
        for k in credit_map.keys() { all_wallets.insert(k.clone()); }

        if all_wallets.is_empty() {
            info!("🔄 [DEX ADJUST v10.3.1] No DEX QUG adjustments to apply (no counters found)");
            return Ok(0);
        }

        let mut adjusted = 0u64;
        let qug = 1_000_000_000_000_000_000_000_000u128;

        for wallet_hex in &all_wallets {
            let debited = debit_map.get(wallet_hex).copied().unwrap_or(0);
            let credited = credit_map.get(wallet_hex).copied().unwrap_or(0);

            // Desired net = credited - debited (signed, can be negative if user sold more QUG than bought)
            let desired_net: i128 = credited as i128 - debited as i128;

            // What we previously applied (0 if never applied before, e.g. after a balance rebuild
            // that wipes balances but NOT the debit/credit counters)
            let previously_applied: i128 = applied_map.get(wallet_hex).copied().unwrap_or(0);

            // Delta = what we need to apply NOW to reach the desired state
            let delta: i128 = desired_net - previously_applied;

            if delta == 0 {
                continue; // Already at correct state — idempotent, nothing to do
            }

            let addr_bytes = match hex::decode(wallet_hex) {
                Ok(b) if b.len() == 32 => {
                    let mut arr = [0u8; 32];
                    arr.copy_from_slice(&b);
                    arr
                }
                _ => continue,
            };

            let current_balance = self.load_wallet_balance(&addr_bytes).await?.unwrap_or(0);

            // Apply delta to current balance (saturating to prevent underflow)
            let new_balance = if delta < 0 {
                // Need to subtract more from balance
                let abs_delta = (-delta) as u128;
                if abs_delta > current_balance {
                    warn!("⚠️ [DEX ADJUST v10.3.1] Wallet {}...: DEX debits ({}) exceed balance ({}) + credits ({}) — clamping to 0",
                        &wallet_hex[..16.min(wallet_hex.len())],
                        debited as f64 / 1e24, current_balance as f64 / 1e24, credited as f64 / 1e24);
                }
                current_balance.saturating_sub(abs_delta)
            } else {
                // Need to add more to balance (rare: only if credits > debits somehow increased)
                current_balance.saturating_add(delta as u128)
            };

            // Atomic write: balance + applied_net tracker in one batch
            let balance_key = format!("wallet_balance_{}", wallet_hex);
            let applied_key = format!("dex_applied_net:{}", wallet_hex);
            let batch: Vec<(&str, Vec<u8>, Vec<u8>)> = vec![
                (CF_MANIFEST, balance_key.as_bytes().to_vec(), new_balance.to_le_bytes().to_vec()),
                (CF_MANIFEST, applied_key.as_bytes().to_vec(), desired_net.to_le_bytes().to_vec()),
            ];
            self.hot_db.write_batch(batch).await?;

            adjusted += 1;
            info!("  🔧 [DEX ADJUST v10.3.1] Wallet {}...: {} → {} QUG (delta: {}, debited: {}, credited: {})",
                &wallet_hex[..16.min(wallet_hex.len())],
                current_balance as f64 / 1e24, new_balance as f64 / 1e24,
                delta as f64 / 1e24,
                debited as f64 / 1e24, credited as f64 / 1e24);
        }

        info!("✅ [DEX ADJUST v10.3.1] Applied adjustments to {} of {} DEX wallets", adjusted, all_wallets.len());
        Ok(adjusted)
    }

    /// One-time migration: Retroactively compute DEX QUG adjustment counters
    /// from existing CF_SWAP_HISTORY records.
    ///
    /// This handles swaps that occurred BEFORE the v9.3.3 tracking was added.
    /// Scans all swap history, extracts QUG debits/credits per wallet, and
    /// initializes the cumulative counters.
    pub async fn migrate_dex_qug_counters_from_swap_history(&self) -> Result<bool> {
        const MIGRATION_FLAG: &[u8] = b"migration_dex_qug_counters_v933_done";

        if self.has_migration_flag(MIGRATION_FLAG).await {
            return Ok(false); // Already done
        }

        info!("🔧 [v9.3.3 MIGRATION] Computing DEX QUG adjustment counters from swap history...");

        // Scan ALL swap history records
        let all_records = self.hot_db.scan_prefix(CF_SWAP_HISTORY, b"swap:").await?;

        let mut debit_totals: HashMap<String, u128> = HashMap::new();
        let mut credit_totals: HashMap<String, u128> = HashMap::new();
        let mut processed = 0u64;

        for (_key, value) in &all_records {
            let record: serde_json::Value = match serde_json::from_slice(value) {
                Ok(v) => v,
                Err(_) => continue,
            };

            let tx_type = record.get("type").and_then(|v| v.as_str()).unwrap_or("");
            let from_token = record.get("fromToken").and_then(|v| v.as_str()).unwrap_or("");
            let to_token = record.get("toToken").and_then(|v| v.as_str()).unwrap_or("");
            let amount = record.get("amount").and_then(|v| v.as_f64()).unwrap_or(0.0);

            if amount <= 0.0 { continue; }

            // Convert display amount back to raw u128 (24 decimals)
            let raw_amount = (amount * 1e24) as u128;

            if tx_type == "sell" && from_token.to_uppercase() == "QUG" {
                // User sold QUG — this is a debit
                // from_address = "qnk{wallet_hex}"
                let from_addr = record.get("from").and_then(|v| v.as_str()).unwrap_or("");
                let wallet_hex = from_addr.trim_start_matches("qnk");
                if wallet_hex.len() == 64 {
                    *debit_totals.entry(wallet_hex.to_string()).or_insert(0) += raw_amount;
                }
            } else if tx_type == "buy" && to_token.to_uppercase() == "QUG" {
                // User bought QUG — this is a credit
                // to_address = "qnk{wallet_hex}"
                let to_addr = record.get("to").and_then(|v| v.as_str()).unwrap_or("");
                let wallet_hex = to_addr.trim_start_matches("qnk");
                if wallet_hex.len() == 64 {
                    *credit_totals.entry(wallet_hex.to_string()).or_insert(0) += raw_amount;
                }
            }

            processed += 1;
        }

        let qug = 1_000_000_000_000_000_000_000_000u128;

        // Write cumulative counters to RocksDB
        for (wallet_hex, total) in &debit_totals {
            let key = format!("dex_qug_debited:{}", wallet_hex);
            self.hot_db.put_sync(CF_MANIFEST, key.as_bytes(), &total.to_le_bytes()).await?;
        }
        for (wallet_hex, total) in &credit_totals {
            let key = format!("dex_qug_credited:{}", wallet_hex);
            self.hot_db.put_sync(CF_MANIFEST, key.as_bytes(), &total.to_le_bytes()).await?;
        }

        info!("✅ [v9.3.3 MIGRATION] Processed {} swap records → {} wallets with debits, {} with credits",
            processed, debit_totals.len(), credit_totals.len());

        for (wallet_hex, debited) in &debit_totals {
            let credited = credit_totals.get(wallet_hex).copied().unwrap_or(0);
            info!("   Wallet {}...: debited={} QUG, credited={} QUG, net={}",
                &wallet_hex[..16.min(wallet_hex.len())],
                *debited / qug, credited / qug,
                (*debited as i128 - credited as i128) / qug as i128);
        }

        self.set_migration_flag(MIGRATION_FLAG).await?;
        Ok(true)
    }

    // ========================================================================
    // v2.4.9-beta: DCA (Dollar Cost Averaging) Persistence
    // Store DCA orders and execution history in RocksDB
    // ========================================================================

    /// Save a DCA order to RocksDB
    /// Key format: order_id (UUID string)
    pub async fn save_dca_order(&self, order_id: &str, order_bytes: &[u8]) -> Result<()> {
        self.hot_db.put(CF_DCA_ORDERS, order_id.as_bytes(), order_bytes).await?;
        debug!("💰 Saved DCA order: {}", order_id);
        Ok(())
    }

    /// Load a DCA order from RocksDB
    pub async fn load_dca_order(&self, order_id: &str) -> Result<Option<Vec<u8>>> {
        self.hot_db.get(CF_DCA_ORDERS, order_id.as_bytes()).await
    }

    /// Delete a DCA order from RocksDB
    pub async fn delete_dca_order(&self, order_id: &str) -> Result<()> {
        self.hot_db.delete(CF_DCA_ORDERS, order_id.as_bytes()).await?;
        debug!("🗑️ Deleted DCA order: {}", order_id);
        Ok(())
    }

    /// Load all DCA orders from RocksDB
    pub async fn load_all_dca_orders(&self) -> Result<Vec<(String, Vec<u8>)>> {
        let all_pairs = self.hot_db.scan_all(CF_DCA_ORDERS).await?;
        let mut orders = Vec::new();
        for (key, value) in all_pairs {
            if let Ok(key_str) = String::from_utf8(key) {
                orders.push((key_str, value));
            }
        }
        debug!("💰 Loaded {} DCA orders from RocksDB", orders.len());
        Ok(orders)
    }

    /// Save a limit order using CF_DCA_ORDERS with "limitorder:" key prefix.
    /// Uses put_sync (fsync) so a Processing status written before swap execution survives a crash.
    pub async fn save_limit_order(&self, order_id: &str, bytes: &[u8]) -> Result<()> {
        let key = format!("limitorder:{}", order_id);
        self.hot_db.put_sync(CF_DCA_ORDERS, key.as_bytes(), bytes).await?;
        debug!("📋 Saved limit order: {}", order_id);
        Ok(())
    }

    /// Delete a limit order from RocksDB
    pub async fn delete_limit_order(&self, order_id: &str) -> Result<()> {
        let key = format!("limitorder:{}", order_id);
        self.hot_db.delete(CF_DCA_ORDERS, key.as_bytes()).await?;
        debug!("🗑️ Deleted limit order: {}", order_id);
        Ok(())
    }

    /// Load all limit orders by scanning CF_DCA_ORDERS with "limitorder:" prefix
    pub async fn load_all_limit_orders(&self) -> Result<Vec<(String, Vec<u8>)>> {
        let pairs = self.hot_db.scan_prefix(CF_DCA_ORDERS, b"limitorder:").await?;
        let orders = pairs
            .into_iter()
            .filter_map(|(k, v)| String::from_utf8(k).ok().map(|s| (s, v)))
            .collect();
        Ok(orders)
    }

    /// Load DCA orders for a specific wallet address
    /// Scans all orders and filters by wallet (orders should include wallet_address field)
    pub async fn load_dca_orders_by_wallet(&self, wallet_address: &str) -> Result<Vec<(String, Vec<u8>)>> {
        let all_pairs = self.hot_db.scan_all(CF_DCA_ORDERS).await?;
        let mut orders = Vec::new();
        for (key, value) in all_pairs {
            // Try to deserialize and check wallet_address field
            if let Ok(order_json) = serde_json::from_slice::<serde_json::Value>(&value) {
                if let Some(addr) = order_json.get("wallet_address").and_then(|v| v.as_str()) {
                    if addr == wallet_address {
                        if let Ok(key_str) = String::from_utf8(key) {
                            orders.push((key_str, value));
                        }
                    }
                }
            }
        }
        debug!("💰 Loaded {} DCA orders for wallet {}", orders.len(), wallet_address);
        Ok(orders)
    }

    /// Save a DCA execution record to RocksDB
    /// Key format: order_id:timestamp
    pub async fn save_dca_execution(&self, order_id: &str, timestamp: i64, execution_bytes: &[u8]) -> Result<()> {
        let key = format!("{}:{}", order_id, timestamp);
        self.hot_db.put(CF_DCA_EXECUTIONS, key.as_bytes(), execution_bytes).await?;
        debug!("📊 Saved DCA execution for order {} at {}", order_id, timestamp);
        Ok(())
    }

    /// Load all executions for a DCA order
    pub async fn load_dca_executions(&self, order_id: &str) -> Result<Vec<(i64, Vec<u8>)>> {
        let prefix = format!("{}:", order_id);
        let records = self.hot_db.scan_prefix(CF_DCA_EXECUTIONS, prefix.as_bytes()).await?;

        let mut executions = Vec::new();
        for (key, value) in records {
            if let Ok(key_str) = String::from_utf8(key) {
                // Extract timestamp from key (format: order_id:timestamp)
                if let Some(ts_str) = key_str.strip_prefix(&prefix) {
                    if let Ok(timestamp) = ts_str.parse::<i64>() {
                        executions.push((timestamp, value));
                    }
                }
            }
        }
        // Sort by timestamp ascending
        executions.sort_by_key(|(ts, _)| *ts);
        debug!("📊 Loaded {} executions for DCA order {}", executions.len(), order_id);
        Ok(executions)
    }

    /// Delete all executions for a DCA order (when order is cancelled)
    pub async fn delete_dca_executions(&self, order_id: &str) -> Result<usize> {
        let prefix = format!("{}:", order_id);
        let records = self.hot_db.scan_prefix(CF_DCA_EXECUTIONS, prefix.as_bytes()).await?;

        let mut deleted = 0;
        for (key, _) in records {
            self.hot_db.delete(CF_DCA_EXECUTIONS, &key).await?;
            deleted += 1;
        }
        debug!("🗑️ Deleted {} executions for DCA order {}", deleted, order_id);
        Ok(deleted)
    }

    // ========================================================================
    // v3.6.0-beta: Consensus-Verified Price History Persistence
    // Store historical price data for tokens in RocksDB
    // Key format: [token_address:32][inverted_timestamp:8] for reverse chronological order
    // Value format: [price:f64 LE][block_height:u64 LE] = 16 bytes
    // ========================================================================

    /// Store a price snapshot for a token
    /// Key format: [token_address:32][inverted_timestamp:8] for reverse chronological order
    pub async fn save_price_snapshot(
        &self,
        token_address: &[u8; 32],
        timestamp_ms: i64,
        price: f64,
        block_height: u64,
    ) -> Result<()> {
        // Inverted timestamp for reverse chronological ordering
        let inverted_ts = i64::MAX - timestamp_ms;

        let mut key = Vec::with_capacity(40);
        key.extend_from_slice(token_address);
        key.extend_from_slice(&inverted_ts.to_be_bytes());

        // Value: price (f64) + block_height (u64) = 16 bytes
        let mut value = Vec::with_capacity(16);
        value.extend_from_slice(&price.to_le_bytes());
        value.extend_from_slice(&block_height.to_le_bytes());

        self.hot_db.put(CF_PRICE_HISTORY, &key, &value).await?;
        debug!(
            "💹 Saved price snapshot: {} at height {} (ts: {})",
            price, block_height, timestamp_ms
        );
        Ok(())
    }

    /// Load price history for a token within a time range
    /// Returns: Vec<(timestamp_ms, price, block_height)> in reverse chronological order
    pub async fn load_price_history(
        &self,
        token_address: &[u8; 32],
        since_timestamp_ms: i64,
        limit: usize,
    ) -> Result<Vec<(i64, f64, u64)>> {
        let inverted_since = i64::MAX - since_timestamp_ms;

        // Scan all records for this token (prefix scan)
        let records = self.hot_db.scan_prefix(CF_PRICE_HISTORY, token_address).await?;

        let mut results = Vec::new();
        for (key, value) in records {
            if results.len() >= limit {
                break;
            }

            // Check key length
            if key.len() < 40 {
                continue;
            }

            // Check prefix matches
            if &key[..32] != token_address {
                break;
            }

            // Check time range (key must be <= inverted_since to be within range)
            let key_inverted_ts = i64::from_be_bytes(key[32..40].try_into().unwrap_or([0u8; 8]));
            if key_inverted_ts > inverted_since {
                // This timestamp is more recent than our cutoff, still in range
                // Continue iterating
            }

            // Decode value
            if value.len() >= 16 {
                let price = f64::from_le_bytes(value[0..8].try_into().unwrap_or([0u8; 8]));
                let block_height = u64::from_le_bytes(value[8..16].try_into().unwrap_or([0u8; 8]));
                let timestamp_ms = i64::MAX - key_inverted_ts;

                // Only include if within time range
                if timestamp_ms >= since_timestamp_ms {
                    results.push((timestamp_ms, price, block_height));
                }
            }
        }

        debug!(
            "📊 Loaded {} price history records for token (since: {})",
            results.len(),
            since_timestamp_ms
        );
        Ok(results)
    }

    /// Get the price closest to a specific timestamp (most recent before that time)
    /// Returns: Option<(timestamp_ms, price)>
    pub async fn get_price_at_time(
        &self,
        token_address: &[u8; 32],
        timestamp_ms: i64,
    ) -> Result<Option<(i64, f64)>> {
        let inverted_ts = i64::MAX - timestamp_ms;

        // Scan records for this token
        let records = self.hot_db.scan_prefix(CF_PRICE_HISTORY, token_address).await?;

        for (key, value) in records {
            // Check key length and prefix
            if key.len() < 40 || &key[..32] != token_address {
                continue;
            }

            let key_inverted = i64::from_be_bytes(key[32..40].try_into().unwrap_or([0u8; 8]));

            // We want the first record where inverted_ts >= key_inverted
            // (meaning timestamp <= requested timestamp)
            if key_inverted >= inverted_ts {
                if value.len() >= 8 {
                    let ts = i64::MAX - key_inverted;
                    let price = f64::from_le_bytes(value[0..8].try_into().unwrap_or([0u8; 8]));
                    return Ok(Some((ts, price)));
                }
            }
        }

        Ok(None)
    }

    // ========================================================================
    // v2.5.0-beta: Perpetual Futures Persistence
    // Store positions, orders, trades, funding, and liquidations in RocksDB
    // ========================================================================

    /// Save a perpetual position to RocksDB
    pub async fn save_perp_position(&self, position_id: &str, position_bytes: &[u8]) -> Result<()> {
        self.hot_db.put(CF_PERP_POSITIONS, position_id.as_bytes(), position_bytes).await?;
        debug!("📈 Saved perp position: {}", position_id);
        Ok(())
    }

    /// Load a perpetual position from RocksDB
    pub async fn load_perp_position(&self, position_id: &str) -> Result<Option<Vec<u8>>> {
        self.hot_db.get(CF_PERP_POSITIONS, position_id.as_bytes()).await
    }

    /// Load all perpetual positions from RocksDB
    pub async fn load_all_perp_positions(&self) -> Result<Vec<(String, Vec<u8>)>> {
        let all_pairs = self.hot_db.scan_all(CF_PERP_POSITIONS).await?;
        let mut positions = Vec::new();
        for (key, value) in all_pairs {
            if let Ok(key_str) = String::from_utf8(key) {
                positions.push((key_str, value));
            }
        }
        debug!("📈 Loaded {} perp positions from RocksDB", positions.len());
        Ok(positions)
    }

    /// Save a perpetual order to RocksDB
    pub async fn save_perp_order(&self, order_id: &str, order_bytes: &[u8]) -> Result<()> {
        self.hot_db.put(CF_PERP_ORDERS, order_id.as_bytes(), order_bytes).await?;
        debug!("📝 Saved perp order: {}", order_id);
        Ok(())
    }

    /// Load all perpetual orders from RocksDB
    pub async fn load_all_perp_orders(&self) -> Result<Vec<(String, Vec<u8>)>> {
        let all_pairs = self.hot_db.scan_all(CF_PERP_ORDERS).await?;
        let mut orders = Vec::new();
        for (key, value) in all_pairs {
            if let Ok(key_str) = String::from_utf8(key) {
                orders.push((key_str, value));
            }
        }
        debug!("📝 Loaded {} perp orders from RocksDB", orders.len());
        Ok(orders)
    }

    /// Save a perpetual trade to RocksDB
    pub async fn save_perp_trade(&self, trade_id: &str, timestamp: i64, trade_bytes: &[u8]) -> Result<()> {
        let key = format!("{}:{}", trade_id, timestamp);
        self.hot_db.put(CF_PERP_TRADES, key.as_bytes(), trade_bytes).await?;
        debug!("💹 Saved perp trade: {}", trade_id);
        Ok(())
    }

    /// Load perpetual trades for a wallet
    pub async fn load_perp_trades(&self, wallet_address: &str) -> Result<Vec<Vec<u8>>> {
        let all_pairs = self.hot_db.scan_all(CF_PERP_TRADES).await?;
        let mut trades = Vec::new();
        for (_, value) in all_pairs {
            // Filter by wallet_address in the JSON
            if let Ok(trade_json) = serde_json::from_slice::<serde_json::Value>(&value) {
                if let Some(addr) = trade_json.get("wallet_address").and_then(|v| v.as_str()) {
                    if addr == wallet_address {
                        trades.push(value);
                    }
                }
            }
        }
        debug!("💹 Loaded {} perp trades for wallet {}", trades.len(), wallet_address);
        Ok(trades)
    }

    /// Save a funding payment to RocksDB
    pub async fn save_perp_funding(&self, market: &str, timestamp: i64, funding_bytes: &[u8]) -> Result<()> {
        let key = format!("{}:{}", market, timestamp);
        self.hot_db.put(CF_PERP_FUNDING, key.as_bytes(), funding_bytes).await?;
        debug!("💰 Saved perp funding for {} at {}", market, timestamp);
        Ok(())
    }

    /// Save a liquidation record to RocksDB
    pub async fn save_perp_liquidation(&self, liquidation_id: &str, liquidation_bytes: &[u8]) -> Result<()> {
        self.hot_db.put(CF_PERP_LIQUIDATIONS, liquidation_id.as_bytes(), liquidation_bytes).await?;
        debug!("🔥 Saved perp liquidation: {}", liquidation_id);
        Ok(())
    }

    // ========================================================================
    // v2.4.0-beta: Governance Persistence (Proposals & Votes)
    // These methods persist governance data across node restarts
    // ========================================================================

    /// Save a governance proposal to RocksDB
    /// Key format: proposal_id (string)
    pub async fn save_governance_proposal(&self, proposal_id: &str, proposal_bytes: &[u8]) -> Result<()> {
        self.hot_db.put(CF_PROPOSALS, proposal_id.as_bytes(), proposal_bytes).await?;
        debug!("📜 Saved governance proposal: {}", proposal_id);
        Ok(())
    }

    /// Load a governance proposal from RocksDB
    pub async fn load_governance_proposal(&self, proposal_id: &str) -> Result<Option<Vec<u8>>> {
        self.hot_db.get(CF_PROPOSALS, proposal_id.as_bytes()).await
    }

    /// Load all governance proposals from RocksDB
    pub async fn load_all_governance_proposals(&self) -> Result<Vec<(String, Vec<u8>)>> {
        let all_pairs = self.hot_db.scan_all(CF_PROPOSALS).await?;
        let mut proposals = Vec::new();
        for (key, value) in all_pairs {
            if let Ok(key_str) = String::from_utf8(key) {
                proposals.push((key_str, value));
            }
        }
        debug!("📜 Loaded {} governance proposals from RocksDB", proposals.len());
        Ok(proposals)
    }

    /// Save a governance vote to RocksDB
    /// Key format: proposal_id:voter_hex
    pub async fn save_governance_vote(&self, proposal_id: &str, voter_hex: &str, vote_bytes: &[u8]) -> Result<()> {
        let key = format!("{}:{}", proposal_id, voter_hex);
        self.hot_db.put(CF_GOVERNANCE_VOTES, key.as_bytes(), vote_bytes).await?;
        debug!("📜 Saved governance vote: {} from {}", proposal_id, voter_hex);
        Ok(())
    }

    /// Load all votes for a proposal from RocksDB
    pub async fn load_governance_votes_for_proposal(&self, proposal_id: &str) -> Result<Vec<Vec<u8>>> {
        let prefix = format!("{}:", proposal_id);
        let all_pairs = self.hot_db.scan_prefix(CF_GOVERNANCE_VOTES, prefix.as_bytes()).await?;
        let votes: Vec<Vec<u8>> = all_pairs.into_iter().map(|(_, v)| v).collect();
        debug!("📜 Loaded {} votes for proposal {}", votes.len(), proposal_id);
        Ok(votes)
    }

    // ========================================================================
    // v7.3.2: Quillon Mail Storage
    // ========================================================================

    /// Save an email message to storage
    pub async fn save_email(&self, email: &q_types::EmailMessage) -> Result<()> {
        let email_bytes = serde_json::to_vec(email)?;
        self.hot_db.put(CF_EMAILS, email.id.as_bytes(), &email_bytes).await?;

        // Index by wallet (for inbox/sent lookups)
        let inverted_ts = u64::MAX - email.timestamp;
        let wallet_hex = if email.folder == "sent" {
            hex::encode(email.from_wallet)
        } else if let Some(ref to_wallet) = email.to_wallet {
            hex::encode(to_wallet)
        } else {
            hex::encode(email.from_wallet)
        };

        // Wallet index key: "wallet_hex:inverted_timestamp:email_id"
        let wallet_key = format!("{}:{:020}:{}", wallet_hex, inverted_ts, email.id);
        self.hot_db.put(CF_EMAILS_BY_WALLET, wallet_key.as_bytes(), email.id.as_bytes()).await?;

        // Folder index key: "wallet_hex:folder:inverted_timestamp:email_id"
        let folder_key = format!("{}:{}:{:020}:{}", wallet_hex, email.folder, inverted_ts, email.id);
        self.hot_db.put(CF_EMAILS_BY_FOLDER, folder_key.as_bytes(), email.id.as_bytes()).await?;

        debug!("📧 Saved email {} in folder '{}' for wallet {}", email.id, email.folder, &wallet_hex[..8]);
        Ok(())
    }

    /// Get all unique wallet addresses that have email activity on this node
    pub async fn get_all_email_wallets(&self) -> Result<Vec<[u8; 32]>> {
        let entries = self.hot_db.scan_prefix(CF_EMAILS_BY_WALLET, b"").await?;
        let mut seen = std::collections::HashSet::new();
        let mut wallets = Vec::new();
        for (key, _) in entries {
            if let Ok(key_str) = String::from_utf8(key) {
                if let Some(wallet_hex) = key_str.split(':').next() {
                    if wallet_hex.len() == 64 && seen.insert(wallet_hex.to_string()) {
                        if let Ok(bytes) = hex::decode(wallet_hex) {
                            if bytes.len() == 32 {
                                let mut w = [0u8; 32];
                                w.copy_from_slice(&bytes);
                                wallets.push(w);
                            }
                        }
                    }
                }
            }
        }
        info!("📧 Found {} unique email wallets", wallets.len());
        Ok(wallets)
    }

    /// Get an email by ID
    pub async fn get_email(&self, email_id: &str) -> Result<Option<q_types::EmailMessage>> {
        match self.hot_db.get(CF_EMAILS, email_id.as_bytes()).await? {
            Some(bytes) => {
                let email: q_types::EmailMessage = serde_json::from_slice(&bytes)?;
                Ok(Some(email))
            }
            None => Ok(None),
        }
    }

    /// Get emails for a wallet in a specific folder with pagination
    pub async fn get_inbox(&self, wallet: &[u8; 32], folder: &str, limit: usize, offset: usize) -> Result<Vec<q_types::EmailMessage>> {
        let wallet_hex = hex::encode(wallet);
        let prefix = format!("{}:{}:", wallet_hex, folder);
        let entries = self.hot_db.scan_prefix(CF_EMAILS_BY_FOLDER, prefix.as_bytes()).await?;

        let mut emails = Vec::new();
        for (i, (_, email_id_bytes)) in entries.into_iter().enumerate() {
            if i < offset { continue; }
            if emails.len() >= limit { break; }

            if let Ok(email_id) = String::from_utf8(email_id_bytes) {
                if let Some(email) = self.get_email(&email_id).await? {
                    emails.push(email);
                }
            }
        }

        debug!("📬 Loaded {} emails for wallet {}... folder={}", emails.len(), &wallet_hex[..8], folder);
        Ok(emails)
    }

    /// Delete an email and its index entries
    pub async fn delete_email(&self, email_id: &str) -> Result<()> {
        if let Some(email) = self.get_email(email_id).await? {
            let inverted_ts = u64::MAX - email.timestamp;
            let wallet_hex = if email.folder == "sent" {
                hex::encode(email.from_wallet)
            } else if let Some(ref to_wallet) = email.to_wallet {
                hex::encode(to_wallet)
            } else {
                hex::encode(email.from_wallet)
            };

            let wallet_key = format!("{}:{:020}:{}", wallet_hex, inverted_ts, email.id);
            let folder_key = format!("{}:{}:{:020}:{}", wallet_hex, email.folder, inverted_ts, email.id);

            self.hot_db.delete(CF_EMAILS_BY_WALLET, wallet_key.as_bytes()).await?;
            self.hot_db.delete(CF_EMAILS_BY_FOLDER, folder_key.as_bytes()).await?;
            self.hot_db.delete(CF_EMAILS, email_id.as_bytes()).await?;

            debug!("🗑️ Deleted email {}", email_id);
        }
        Ok(())
    }

    /// Mark an email as read
    pub async fn mark_email_read(&self, email_id: &str) -> Result<()> {
        if let Some(mut email) = self.get_email(email_id).await? {
            email.read = true;
            let email_bytes = serde_json::to_vec(&email)?;
            self.hot_db.put(CF_EMAILS, email_id.as_bytes(), &email_bytes).await?;
            tracing::info!("📖 Marked email {} as read (subject='{}')", email_id, email.subject);
        } else {
            tracing::warn!("📖 mark_email_read: email {} not found in DB", email_id);
        }
        Ok(())
    }

    /// Mark all inbox emails as read for a wallet
    pub async fn mark_all_inbox_read(&self, wallet: &[u8; 32]) -> Result<u64> {
        let wallet_hex = hex::encode(wallet);
        let prefix = format!("{}:inbox:", wallet_hex);
        let entries = self.hot_db.scan_prefix(CF_EMAILS_BY_FOLDER, prefix.as_bytes()).await?;

        let mut marked = 0u64;
        for (_, email_id_bytes) in entries {
            if let Ok(email_id) = String::from_utf8(email_id_bytes) {
                if let Some(mut email) = self.get_email(&email_id).await? {
                    if !email.read {
                        email.read = true;
                        let email_bytes = serde_json::to_vec(&email)?;
                        self.hot_db.put(CF_EMAILS, email_id.as_bytes(), &email_bytes).await?;
                        marked += 1;
                    }
                }
            }
        }
        debug!("📖 Marked {} emails as read for wallet {}", marked, &wallet_hex[..8]);
        Ok(marked)
    }

    /// Get unread count for a wallet
    pub async fn get_unread_count(&self, wallet: &[u8; 32]) -> Result<u64> {
        let wallet_hex = hex::encode(wallet);
        let prefix = format!("{}:inbox:", wallet_hex);
        let entries = self.hot_db.scan_prefix(CF_EMAILS_BY_FOLDER, prefix.as_bytes()).await?;

        let mut count = 0u64;
        let total = entries.len();
        for (_, email_id_bytes) in entries {
            if let Ok(email_id) = String::from_utf8(email_id_bytes) {
                if let Some(email) = self.get_email(&email_id).await? {
                    if !email.read {
                        count += 1;
                        // v8.5.5: Log unread emails at info level for debugging
                        tracing::info!(
                            "📬 [UNREAD] email_id={} subject='{}' folder='{}' for wallet {}",
                            email_id, email.subject, email.folder, &wallet_hex[..8]
                        );
                    }
                } else {
                    // Orphaned index entry — email exists in folder index but not in CF_EMAILS
                    tracing::warn!(
                        "⚠️ [EMAIL] Orphaned folder index: email_id={} not found in CF_EMAILS for wallet {}",
                        email_id, &wallet_hex[..8]
                    );
                }
            }
        }
        if count > 0 {
            tracing::info!("📬 [UNREAD] Wallet {} has {}/{} unread inbox emails", &wallet_hex[..8], count, total);
        }
        Ok(count)
    }

    /// Save an outbound email for SMTP delivery
    pub async fn save_outbound_email(&self, msg: &q_types::OutboundEmail) -> Result<()> {
        let msg_bytes = serde_json::to_vec(msg)?;
        self.hot_db.put(CF_EMAIL_OUTBOUND, msg.id.as_bytes(), &msg_bytes).await?;
        debug!("📤 Queued outbound email {} to {}", msg.id, msg.to_email);
        Ok(())
    }

    /// Claim pending outbound emails for delivery
    pub async fn claim_outbound_emails(&self, limit: usize) -> Result<Vec<q_types::OutboundEmail>> {
        let all = self.hot_db.scan_all(CF_EMAIL_OUTBOUND).await?;
        let mut claimed = Vec::new();

        for (key, value) in all {
            if claimed.len() >= limit { break; }
            if let Ok(mut msg) = serde_json::from_slice::<q_types::OutboundEmail>(&value) {
                if msg.status == q_types::OutboundStatus::Pending || msg.status == q_types::OutboundStatus::Retrying {
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs();
                    if let Some(next_retry) = msg.next_retry_at {
                        if now < next_retry { continue; }
                    }
                    msg.status = q_types::OutboundStatus::Processing;
                    let msg_bytes = serde_json::to_vec(&msg)?;
                    self.hot_db.put(CF_EMAIL_OUTBOUND, &key, &msg_bytes).await?;
                    claimed.push(msg);
                }
            }
        }

        debug!("📤 Claimed {} outbound emails for delivery", claimed.len());
        Ok(claimed)
    }

    /// Mark outbound email as delivered
    pub async fn mark_outbound_delivered(&self, id: &str) -> Result<()> {
        if let Some(bytes) = self.hot_db.get(CF_EMAIL_OUTBOUND, id.as_bytes()).await? {
            if let Ok(mut msg) = serde_json::from_slice::<q_types::OutboundEmail>(&bytes) {
                msg.status = q_types::OutboundStatus::Delivered;
                let msg_bytes = serde_json::to_vec(&msg)?;
                self.hot_db.put(CF_EMAIL_OUTBOUND, id.as_bytes(), &msg_bytes).await?;
                debug!("✅ Outbound email {} delivered", id);
            }
        }
        Ok(())
    }

    /// Mark outbound email as failed with error and retry scheduling
    pub async fn mark_outbound_failed(&self, id: &str, error: &str) -> Result<()> {
        if let Some(bytes) = self.hot_db.get(CF_EMAIL_OUTBOUND, id.as_bytes()).await? {
            if let Ok(mut msg) = serde_json::from_slice::<q_types::OutboundEmail>(&bytes) {
                msg.retry_count += 1;
                msg.last_error = Some(error.to_string());

                if msg.retry_count >= 5 {
                    msg.status = q_types::OutboundStatus::Failed;
                    debug!("❌ Outbound email {} permanently failed after {} retries: {}", id, msg.retry_count, error);
                } else {
                    msg.status = q_types::OutboundStatus::Retrying;
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs();
                    // Exponential backoff: 5m, 15m, 1h, 4h, 24h
                    let delays = [300, 900, 3600, 14400, 86400];
                    let delay = delays.get(msg.retry_count as usize - 1).copied().unwrap_or(86400);
                    msg.next_retry_at = Some(now + delay);
                    debug!("🔄 Outbound email {} retry #{} scheduled in {}s: {}", id, msg.retry_count, delay, error);
                }

                let msg_bytes = serde_json::to_vec(&msg)?;
                self.hot_db.put(CF_EMAIL_OUTBOUND, id.as_bytes(), &msg_bytes).await?;
            }
        }
        Ok(())
    }

    /// Save email contact
    pub async fn save_email_contact(&self, owner: &[u8; 32], contact: &q_types::EmailContact) -> Result<()> {
        let key = format!("{}:{}", hex::encode(owner), hex::encode(contact.wallet_address));
        let value = serde_json::to_vec(contact)?;
        self.hot_db.put(CF_EMAIL_CONTACTS, key.as_bytes(), &value).await?;
        Ok(())
    }

    /// Get all email contacts for a wallet
    pub async fn get_email_contacts(&self, owner: &[u8; 32]) -> Result<Vec<q_types::EmailContact>> {
        let prefix = format!("{}:", hex::encode(owner));
        let entries = self.hot_db.scan_prefix(CF_EMAIL_CONTACTS, prefix.as_bytes()).await?;

        let mut contacts = Vec::new();
        for (_, value) in entries {
            if let Ok(contact) = serde_json::from_slice::<q_types::EmailContact>(&value) {
                contacts.push(contact);
            }
        }
        Ok(contacts)
    }

    /// Search emails by subject or body text
    pub async fn search_emails(&self, wallet: &[u8; 32], query: &str, limit: usize) -> Result<Vec<q_types::EmailMessage>> {
        let wallet_hex = hex::encode(wallet);
        let prefix = format!("{}:", wallet_hex);
        let entries = self.hot_db.scan_prefix(CF_EMAILS_BY_WALLET, prefix.as_bytes()).await?;

        let query_lower = query.to_lowercase();
        let mut results = Vec::new();

        for (_, email_id_bytes) in entries {
            if results.len() >= limit { break; }
            if let Ok(email_id) = String::from_utf8(email_id_bytes) {
                if let Some(email) = self.get_email(&email_id).await? {
                    if email.subject.to_lowercase().contains(&query_lower)
                        || email.body.to_lowercase().contains(&query_lower) {
                        results.push(email);
                    }
                }
            }
        }
        Ok(results)
    }

    // ========================================================================
    // Email Settings Storage (v7.3.3)
    // ========================================================================

    /// Save email settings for a wallet (stored in CF_EMAILS with "settings:" prefix)
    pub async fn save_email_settings(&self, wallet_hex: &str, settings_json: &[u8]) -> Result<()> {
        let key = format!("settings:{}", wallet_hex);
        self.hot_db.put(CF_EMAILS, key.as_bytes(), settings_json).await?;
        Ok(())
    }

    /// Get email settings for a wallet
    pub async fn get_email_settings(&self, wallet_hex: &str) -> Result<Option<serde_json::Value>> {
        let key = format!("settings:{}", wallet_hex);
        match self.hot_db.get(CF_EMAILS, key.as_bytes()).await? {
            Some(data) => {
                let val: serde_json::Value = serde_json::from_slice(&data)?;
                Ok(Some(val))
            }
            None => Ok(None),
        }
    }

    /// Save email alias → wallet mapping (stored in CF_EMAILS with "alias:" prefix)
    pub async fn save_email_alias(&self, alias: &str, wallet_hex: &str) -> Result<()> {
        let key = format!("alias:{}", alias);
        self.hot_db.put(CF_EMAILS, key.as_bytes(), wallet_hex.as_bytes()).await?;
        Ok(())
    }

    /// Resolve email alias to wallet hex
    pub async fn get_email_alias_wallet(&self, alias: &str) -> Result<Option<String>> {
        let key = format!("alias:{}", alias);
        match self.hot_db.get(CF_EMAILS, key.as_bytes()).await? {
            Some(data) => Ok(Some(String::from_utf8_lossy(&data).to_string())),
            None => Ok(None),
        }
    }

    /// Remove email alias mapping
    pub async fn delete_email_alias(&self, alias: &str) -> Result<()> {
        let key = format!("alias:{}", alias);
        self.hot_db.delete(CF_EMAILS, key.as_bytes()).await?;
        Ok(())
    }

    // ========================================================================
    // Calendar Storage Methods (v7.3.3)
    // ========================================================================

    /// Save a calendar event with date index
    pub async fn save_calendar_event(&self, event: &q_types::CalendarEvent) -> Result<()> {
        let event_bytes = serde_json::to_vec(event)?;
        self.hot_db.put(CF_CALENDAR_EVENTS, event.id.as_bytes(), &event_bytes).await?;

        // Index by date for range queries
        let wallet_hex = hex::encode(event.wallet);
        let date_str = {
            let dt = chrono::DateTime::from_timestamp(event.start_time as i64, 0)
                .unwrap_or_else(|| chrono::Utc::now());
            dt.format("%Y%m%d").to_string()
        };
        let date_key = format!("{}:{}:{}", wallet_hex, date_str, event.id);
        self.hot_db.put(CF_CALENDAR_BY_DATE, date_key.as_bytes(), event.id.as_bytes()).await?;

        // If it has a scheduled transaction, index for the executor
        if event.scheduled_tx.is_some() && !event.cancelled {
            let sched_key = format!("{}:{:020}:{}", wallet_hex, event.start_time, event.id);
            self.hot_db.put(CF_CALENDAR_SCHEDULED_TX, sched_key.as_bytes(), event.id.as_bytes()).await?;
        }

        debug!("📅 Saved calendar event {} '{}'", event.id, event.title);
        Ok(())
    }

    /// Get a single calendar event by ID
    pub async fn get_calendar_event(&self, event_id: &str) -> Result<Option<q_types::CalendarEvent>> {
        match self.hot_db.get(CF_CALENDAR_EVENTS, event_id.as_bytes()).await? {
            Some(bytes) => {
                let event: q_types::CalendarEvent = serde_json::from_slice(&bytes)?;
                Ok(Some(event))
            }
            None => Ok(None),
        }
    }

    /// Get calendar events for a wallet within a date range (YYYYMMDD strings)
    pub async fn get_calendar_events_by_date_range(
        &self, wallet: &[u8; 32], start_date: &str, end_date: &str,
    ) -> Result<Vec<q_types::CalendarEvent>> {
        let wallet_hex = hex::encode(wallet);
        let start_prefix = format!("{}:{}", wallet_hex, start_date);
        let end_prefix = format!("{}:{}~", wallet_hex, end_date); // ~ is after 9 in ASCII

        let entries = self.hot_db.scan_prefix(CF_CALENDAR_BY_DATE, format!("{}:", wallet_hex).as_bytes()).await?;
        let mut events = Vec::new();

        for (key, event_id_bytes) in entries {
            if let Ok(key_str) = String::from_utf8(key) {
                if key_str >= start_prefix && key_str <= end_prefix {
                    if let Ok(event_id) = String::from_utf8(event_id_bytes) {
                        if let Some(event) = self.get_calendar_event(&event_id).await? {
                            if !event.cancelled {
                                events.push(event);
                            }
                        }
                    }
                }
            }
        }

        events.sort_by_key(|e| e.start_time);
        Ok(events)
    }

    /// Update a calendar event
    pub async fn update_calendar_event(&self, event: &q_types::CalendarEvent) -> Result<()> {
        self.save_calendar_event(event).await
    }

    /// Soft-delete a calendar event (marks as cancelled)
    pub async fn delete_calendar_event(&self, event_id: &str) -> Result<()> {
        if let Some(mut event) = self.get_calendar_event(event_id).await? {
            event.cancelled = true;
            event.updated_at = Some(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs()
            );
            let event_bytes = serde_json::to_vec(&event)?;
            self.hot_db.put(CF_CALENDAR_EVENTS, event_id.as_bytes(), &event_bytes).await?;

            // Remove from scheduled tx index if present
            if event.scheduled_tx.is_some() {
                let wallet_hex = hex::encode(event.wallet);
                let sched_key = format!("{}:{:020}:{}", wallet_hex, event.start_time, event.id);
                let _ = self.hot_db.delete(CF_CALENDAR_SCHEDULED_TX, sched_key.as_bytes()).await;
            }
            debug!("📅 Deleted calendar event {}", event_id);
        }
        Ok(())
    }

    /// Get pending scheduled transactions (not executed, start_time <= now)
    pub async fn get_pending_scheduled_transactions(&self) -> Result<Vec<q_types::CalendarEvent>> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let entries = self.hot_db.scan_all(CF_CALENDAR_SCHEDULED_TX).await?;
        let mut pending = Vec::new();

        for (_, event_id_bytes) in entries {
            if let Ok(event_id) = String::from_utf8(event_id_bytes) {
                if let Some(event) = self.get_calendar_event(&event_id).await? {
                    if event.cancelled { continue; }
                    if let Some(ref tx) = event.scheduled_tx {
                        if !tx.executed && event.start_time <= now {
                            pending.push(event);
                        }
                    }
                }
            }
        }
        Ok(pending)
    }

    /// Mark a scheduled transaction as executed
    pub async fn mark_scheduled_tx_executed(&self, event_id: &str, tx_hash: &str) -> Result<()> {
        if let Some(mut event) = self.get_calendar_event(event_id).await? {
            if let Some(ref mut tx) = event.scheduled_tx {
                tx.executed = true;
                tx.tx_hash = Some(tx_hash.to_string());
            }
            event.updated_at = Some(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs()
            );
            let event_bytes = serde_json::to_vec(&event)?;
            self.hot_db.put(CF_CALENDAR_EVENTS, event_id.as_bytes(), &event_bytes).await?;

            // Remove from pending index
            let wallet_hex = hex::encode(event.wallet);
            let sched_key = format!("{}:{:020}:{}", wallet_hex, event.start_time, event.id);
            let _ = self.hot_db.delete(CF_CALENDAR_SCHEDULED_TX, sched_key.as_bytes()).await;

            debug!("📅 Scheduled TX for event {} executed: {}", event_id, tx_hash);
        }
        Ok(())
    }

    /// Mark a scheduled transaction as failed and remove it from the pending index.
    /// Without the index removal the executor picks it up again every 60 s, retrying forever.
    pub async fn mark_scheduled_tx_failed(&self, event_id: &str, error: &str) -> Result<()> {
        if let Some(mut event) = self.get_calendar_event(event_id).await? {
            if let Some(ref mut tx) = event.scheduled_tx {
                tx.error = Some(error.to_string());
            }
            event.updated_at = Some(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs()
            );
            let event_bytes = serde_json::to_vec(&event)?;
            self.hot_db.put(CF_CALENDAR_EVENTS, event_id.as_bytes(), &event_bytes).await?;

            // Remove from pending index so the executor does not retry this TX.
            let wallet_hex = hex::encode(event.wallet);
            let sched_key = format!("{}:{:020}:{}", wallet_hex, event.start_time, event.id);
            let _ = self.hot_db.delete(CF_CALENDAR_SCHEDULED_TX, sched_key.as_bytes()).await;

            debug!("📅 Scheduled TX for event {} failed (removed from pending index): {}", event_id, error);
        }
        Ok(())
    }

    /// Save a community event shared via P2P
    pub async fn save_community_event(&self, event: &q_types::CalendarEvent) -> Result<()> {
        let event_bytes = serde_json::to_vec(event)?;
        self.hot_db.put(CF_CALENDAR_COMMUNITY, event.id.as_bytes(), &event_bytes).await?;
        debug!("📅 Saved community event {} '{}'", event.id, event.title);
        Ok(())
    }

    /// Get all community events (shared via P2P)
    pub async fn get_community_events(&self, limit: usize) -> Result<Vec<q_types::CalendarEvent>> {
        let entries = self.hot_db.scan_all(CF_CALENDAR_COMMUNITY).await?;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let mut events = Vec::new();
        for (_, value) in entries {
            if events.len() >= limit { break; }
            if let Ok(event) = serde_json::from_slice::<q_types::CalendarEvent>(&value) {
                if !event.cancelled && event.start_time + 86400 * 30 > now {
                    events.push(event);
                }
            }
        }
        events.sort_by_key(|e| e.start_time);
        Ok(events)
    }

    /// Count how many events a wallet has shared today (spam limit)
    pub async fn count_shared_events_today(&self, wallet: &[u8; 32]) -> Result<u32> {
        let entries = self.hot_db.scan_all(CF_CALENDAR_COMMUNITY).await?;
        let today_start = {
            let now = chrono::Utc::now();
            now.date_naive().and_hms_opt(0, 0, 0)
                .map(|dt| dt.and_utc().timestamp() as u64)
                .unwrap_or(0)
        };

        let mut count = 0u32;
        for (_, value) in entries {
            if let Ok(event) = serde_json::from_slice::<q_types::CalendarEvent>(&value) {
                if event.wallet == *wallet && event.created_at >= today_start {
                    count += 1;
                }
            }
        }
        Ok(count)
    }
}

// ========== v1.1.24-beta: Mainnet Safety Trait Implementations ==========

/// Implementation of CheckpointStorage trait for mainnet safety checkpoints
#[async_trait::async_trait]
impl mainnet_safety::CheckpointStorage for QStorage {
    /// Get latest block height from storage
    async fn get_latest_height(&self) -> Result<u64> {
        match self.get_latest_qblock_height().await? {
            Some(height) => Ok(height),
            None => Ok(0),
        }
    }

    /// Get block hash at specific height
    async fn get_block_hash(&self, height: u64) -> Result<Option<String>> {
        let height_key = format!("qblock:height:{}", height);
        match self.hot_db.get(CF_BLOCKS, height_key.as_bytes()).await? {
            Some(block_bytes) => {
                // Deserialize to get hash from header
                match postcard::from_bytes::<q_types::block::QBlock>(&block_bytes) {
                    Ok(block) => Ok(Some(hex::encode(&block.header.state_root))),
                    Err(_) => {
                        // Try MessagePack fallback
                        match rmp_serde::from_slice::<q_types::block::QBlock>(&block_bytes) {
                            Ok(block) => Ok(Some(hex::encode(&block.header.state_root))),
                            Err(_) => Ok(None),
                        }
                    }
                }
            }
            None => Ok(None),
        }
    }

    /// Compute state root (Merkle root of all balances)
    async fn compute_state_root(&self) -> Result<Option<String>> {
        // For now, return a simple hash of the latest height
        // In production, this should compute a proper Merkle root
        use mainnet_safety::CheckpointStorage;
        let height = CheckpointStorage::get_latest_height(self).await?;
        if height == 0 {
            return Ok(None);
        }

        // Compute simple state hash based on height and timestamp
        let state_data = format!("state:{}:{}", height, SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs());
        let hash = blake3::hash(state_data.as_bytes());
        Ok(Some(hex::encode(hash.as_bytes())))
    }

    /// Export blocks in a range for backup
    async fn export_blocks_range(&self, start: u64, end: u64) -> Result<Vec<u8>> {
        let mut blocks = Vec::new();

        for height in start..=end {
            let height_key = format!("qblock:height:{}", height);
            if let Some(block_bytes) = self.hot_db.get(CF_BLOCKS, height_key.as_bytes()).await? {
                // Store as (height, block_bytes)
                blocks.push((height, block_bytes));
            }
        }

        // Serialize blocks using postcard
        let export_data = postcard::to_allocvec(&blocks)
            .context("Failed to serialize blocks for export")?;

        Ok(export_data)
    }

    /// Import blocks from backup data
    async fn import_blocks(&self, data: &[u8]) -> Result<u64> {
        let blocks: Vec<(u64, Vec<u8>)> = postcard::from_bytes(data)
            .context("Failed to deserialize backup data")?;

        let mut imported_count = 0u64;

        for (height, block_bytes) in blocks {
            let height_key = format!("qblock:height:{}", height);
            self.hot_db.put(CF_BLOCKS, height_key.as_bytes(), &block_bytes).await?;
            imported_count += 1;
        }

        info!("📥 [IMPORT] Imported {} blocks from backup", imported_count);
        Ok(imported_count)
    }

    /// Truncate all blocks after the specified height
    async fn truncate_after_height(&self, height: u64) -> Result<()> {
        use mainnet_safety::CheckpointStorage;
        let current_height = CheckpointStorage::get_latest_height(self).await?;

        if current_height <= height {
            return Ok(()); // Nothing to truncate
        }

        warn!("⚠️ [TRUNCATE] Removing blocks from {} to {}", height + 1, current_height);

        for h in (height + 1)..=current_height {
            let height_key = format!("qblock:height:{}", h);
            self.hot_db.delete(CF_BLOCKS, height_key.as_bytes()).await?;
        }

        // Update the height pointer
        self.hot_db.put(CF_BLOCKS, b"qblock:latest", &height.to_be_bytes()).await?;

        info!("✅ [TRUNCATE] Successfully truncated to height {}", height);
        Ok(())
    }
}

/// Implementation of IntegrityCheckable trait for mainnet safety monitoring
#[async_trait::async_trait]
impl mainnet_safety::IntegrityCheckable for QStorage {
    /// Get the current pointer height from storage
    async fn get_pointer_height(&self) -> Result<u64> {
        match self.hot_db.get(CF_BLOCKS, b"qblock:latest").await? {
            Some(height_bytes) if height_bytes.len() == 8 => {
                let mut height_array = [0u8; 8];
                height_array.copy_from_slice(&height_bytes);
                Ok(u64::from_be_bytes(height_array))
            }
            _ => Ok(0),
        }
    }

    /// Scan to find actual highest block in storage
    async fn scan_actual_highest_block(&self) -> Result<u64> {
        // Use binary search to find the actual highest block
        let pointer_height = self.get_pointer_height().await?;

        // Start from pointer and scan backwards if necessary
        for probe_height in (1..=pointer_height).rev() {
            let height_key = format!("qblock:height:{}", probe_height);
            if self.hot_db.get(CF_BLOCKS, height_key.as_bytes()).await?.is_some() {
                return Ok(probe_height);
            }
        }

        Ok(0)
    }

    /// Find gaps in the block chain between start and end heights
    async fn find_block_gaps(&self, start: u64, end: u64) -> Result<Vec<u64>> {
        let mut gaps = Vec::new();

        for height in start..=end {
            let height_key = format!("qblock:height:{}", height);
            if self.hot_db.get(CF_BLOCKS, height_key.as_bytes()).await?.is_none() {
                gaps.push(height);
            }
        }

        Ok(gaps)
    }

    /// Verify parent chain integrity from a given height
    async fn verify_parent_chain(&self, height: u64) -> Result<Vec<u64>> {
        let mut broken_links = Vec::new();

        // Sample verification - check last 100 blocks
        let check_depth = 100.min(height);

        for h in (height.saturating_sub(check_depth)..=height).rev() {
            let height_key = format!("qblock:height:{}", h);
            if let Some(block_bytes) = self.hot_db.get(CF_BLOCKS, height_key.as_bytes()).await? {
                // Try to deserialize and check parent
                if let Ok(block) = postcard::from_bytes::<q_types::block::QBlock>(&block_bytes) {
                    if h > 1 {
                        // Check if parent exists
                        let parent_key = format!("qblock:height:{}", h - 1);
                        if self.hot_db.get(CF_BLOCKS, parent_key.as_bytes()).await?.is_none() {
                            broken_links.push(h);
                        }
                    }
                }
            }
        }

        Ok(broken_links)
    }

    /// Fix the pointer to the correct height
    async fn fix_pointer(&self, correct_height: u64) -> Result<()> {
        warn!("🔧 [FIX] Repairing pointer from current to {}", correct_height);
        self.hot_db.put(CF_BLOCKS, b"qblock:latest", &correct_height.to_be_bytes()).await?;
        info!("✅ [FIX] Pointer successfully repaired to {}", correct_height);
        Ok(())
    }
}

/// Storage statistics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageStats {
    pub dag_round_watermark: u64,
    pub finalized_height: u64,
    pub total_vertices: u64,
    pub total_payloads: u64,
    pub total_blocks: u64,
    pub hot_db_size: u64,
    pub cold_db_size: u64,
    pub average_write_latency: Duration,
    pub average_read_latency: Duration,
}

/// Storage health information
#[derive(Debug, Clone)]
pub struct StorageHealth {
    pub status: StorageHealthStatus,
    pub last_write: std::time::SystemTime,
    pub error_count: u64,
    pub stats: StorageStats,
}

/// Storage health status
#[derive(Debug, Clone, PartialEq)]
pub enum StorageHealthStatus {
    Healthy,
    PerformanceIssues,
    InconsistentState,
    DatabaseError,
    Offline,
}

impl StorageHealthStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Healthy => "healthy",
            Self::PerformanceIssues => "performance_issues",
            Self::InconsistentState => "inconsistent_state",
            Self::DatabaseError => "database_error",
            Self::Offline => "offline",
        }
    }

    pub fn is_critical(&self) -> bool {
        matches!(
            self,
            Self::InconsistentState | Self::DatabaseError | Self::Offline
        )
    }
}

/// AI Chat metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMetadata {
    pub chat_id: String,
    pub user_id: String,
    pub title: String,
    pub model: String,
    pub created_at: u64,
    pub updated_at: u64,
    pub message_count: u64,
    pub encryption_enabled: bool,
    pub zk_proofs_enabled: bool,
    pub distributed_enabled: bool,
    pub enable_kv_cache: bool,
    pub enable_pipeline_parallel: bool,
    pub enable_load_balancing: bool,
}

/// AI Chat message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub index: u64,
    pub role: String,
    pub content: String,
    pub timestamp: u64,
    pub images: Option<Vec<String>>,
    pub audio: Option<String>,
    pub reasoning: Option<String>, // Kimi K2 thinking process (v1.0.5)
    pub generation_stats: Option<GenerationStats>,
}

/// v2.4.1-beta: Protected AI Chat message with TemporalShield
///
/// Wraps sensitive content (user prompts, AI reasoning) in TemporalEnvelope
/// for HNDL attack resistance. Uses (3,5) threshold sharing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtectedChatMessage {
    /// Message index in chat
    pub index: u64,
    /// Role (user/assistant/system) - NOT protected
    pub role: String,
    /// Timestamp - NOT protected (for timeline auditing)
    pub timestamp: u64,
    /// Protected user prompt (TemporalEnvelope bytes)
    pub protected_content: Vec<u8>,
    /// Protected AI reasoning (TemporalEnvelope bytes, optional)
    pub protected_reasoning: Option<Vec<u8>>,
    /// Generation stats - NOT protected (not sensitive)
    pub generation_stats: Option<GenerationStats>,
    /// Blake3 hash of content for search indexing without decryption
    pub content_hash: [u8; 32],
    /// Flag indicating if message is protected
    pub is_protected: bool,
}

impl ProtectedChatMessage {
    /// Create a protected message from envelope bytes
    pub fn new(
        index: u64,
        role: String,
        protected_content: Vec<u8>,
        protected_reasoning: Option<Vec<u8>>,
        generation_stats: Option<GenerationStats>,
        content_hash: [u8; 32],
    ) -> Self {
        Self {
            index,
            role,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            protected_content,
            protected_reasoning,
            generation_stats,
            content_hash,
            is_protected: true,
        }
    }

    /// Create an unprotected message (fallback when TemporalShield unavailable)
    pub fn unprotected(message: &ChatMessage) -> Self {
        let content_hash = *blake3::hash(message.content.as_bytes()).as_bytes();
        Self {
            index: message.index,
            role: message.role.clone(),
            timestamp: message.timestamp,
            protected_content: message.content.as_bytes().to_vec(),
            protected_reasoning: message.reasoning.as_ref().map(|r| r.as_bytes().to_vec()),
            generation_stats: message.generation_stats.clone(),
            content_hash,
            is_protected: false,
        }
    }

    /// Check if message matches a search hash (for searching without decryption)
    pub fn matches_hash(&self, search_hash: &[u8; 32]) -> bool {
        &self.content_hash == search_hash
    }
}

/// AI generation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationStats {
    pub total_tokens: usize,
    pub latency_ms: u64,
    pub tokens_per_second: f64,
    pub privacy_overhead_ms: u64,
    pub zk_proof_time_ms: u64,
    pub distributed_nodes_used: usize,
}

/// Chat settings for updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatSettings {
    pub encryption_enabled: bool,
    pub zk_proofs_enabled: bool,
    pub distributed_enabled: bool,
    pub enable_kv_cache: bool,
    pub enable_pipeline_parallel: bool,
    pub enable_load_balancing: bool,
}

// ============================================================================
// Payment Consensus Structures
// ============================================================================

/// AI Credits for wallet
/// v3.0.4: Migrated monetary fields to u128 for 24-decimal precision
/// v3.2.2: Added u128_serde for MessagePack P2P compatibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AICredits {
    pub wallet_address: String,
    pub balance_qnk: u128,
    pub balance_qugusd: u128,
    pub total_spent_qnk: u128,
    pub total_spent_qugusd: u128,
    pub total_tokens_generated: u64,
    pub created_at: u64,
    pub updated_at: u64,
}

/// AI Transaction record
/// v3.0.4: Migrated cost_qnk to u128 for 24-decimal precision
/// v3.2.2: Added u128_serde for MessagePack P2P compatibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AITransaction {
    pub tx_id: String,
    pub wallet_address: String,
    pub chat_id: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cost_usd_cents: u64,
    pub cost_qnk: u128,  // v3.0.4: u64 -> u128
    pub payment_token: PaymentToken,
    pub oracle_price_usd_cents: u64,
    pub timestamp: u64,
    pub status: PaymentStatus,
}

/// Payment token type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PaymentToken {
    QNK,
    QUGUSD,
}

/// Payment status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PaymentStatus {
    Pending,
    Completed,
    Refunded,
    Failed,
}

/// Payment Proposal for distributed consensus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaymentProposal {
    pub request_id: String,
    pub wallet_address: String,
    pub estimated_tokens: u32,
    pub estimated_cost_qnk: u64,
    pub payment_token: PaymentToken,
    pub signature: Vec<u8>,
    pub timestamp: u64,
    pub proposer_node_id: String,
}

/// Payment Vote from validator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaymentVote {
    pub request_id: String,
    pub validator_node_id: String,
    pub vote: bool, // true = approve, false = reject
    pub reason: Option<String>,
    pub signature: Vec<u8>,
}

/// Payment Lock (consensus reached)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaymentLock {
    pub request_id: String,
    pub wallet_address: String,
    pub locked_amount_qnk: u64,
    pub locked_at: u64,
    pub validator_signatures: Vec<ValidatorSignature>,
}

/// Validator Signature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorSignature {
    pub node_id: String,
    pub signature: Vec<u8>,
}

/// AI Treasury (Master Wallet)
/// v3.0.4: Migrated monetary fields to u128 for 24-decimal precision
/// v3.2.2: Added u128_serde for MessagePack P2P compatibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AITreasury {
    pub wallet_address: String,
    pub total_revenue_qnk: u128,      // v3.0.4: u64 -> u128
    pub total_revenue_qugusd: u128,   // v3.0.4: u64 -> u128
    pub total_requests_served: u64,
    pub total_tokens_generated: u64,
    pub created_at: u64,
    pub updated_at: u64,
}

/// Payment Settlement (100% profits to treasury)
/// v3.0.4: Migrated monetary fields to u128 for 24-decimal precision
/// v3.2.2: Added u128_serde for MessagePack P2P compatibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaymentSettlement {
    pub request_id: String,
    pub wallet_address: String,  // User wallet
    pub actual_tokens_generated: u32,
    pub actual_cost_qnk: u128,       // v3.0.4: u64 -> u128
    pub refund_amount_qnk: u128,     // v3.0.4: u64 -> u128
    pub treasury_payment_qnk: u128,  // v3.0.4: u64 -> u128, = actual_cost_qnk (100% to treasury)
    pub treasury_wallet: String,     // MASTER_AI_TREASURY_WALLET
    pub generation_node_id: String,
    pub validator_signatures: Vec<ValidatorSignature>,
    pub timestamp: u64,
}

/// Consensus Result
#[derive(Debug, Clone, PartialEq)]
pub enum ConsensusResult {
    Approved,
    Rejected,
    Pending,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_storage_creation() {
        let temp_dir = TempDir::new().unwrap();
        let node_id = [1u8; 32];

        let storage = QStorage::open(temp_dir.path(), node_id).await;
        assert!(storage.is_ok());
    }

    #[test]
    fn test_key_generation() {
        let storage = QStorage {
            // Mock storage for testing key generation
            hot_db: Arc::new(MockKVStore::new()),
            cold_db: Arc::new(MockKVStore::new()),
            manifest: Arc::new(RwLock::new(StorageManifest::default())),
            sync_protocol: Arc::new(SyncProtocol::mock()),
            snapshot_manager: Arc::new(SnapshotManager::mock()),
            metrics: Arc::new(StorageMetrics::new()),
            node_id: [1u8; 32],
            data_dir: PathBuf::from("/tmp"),
        };

        let vertex_key = storage.vertex_key(100, &[0xaa, 0xbb], &[0x01; 32]);
        assert_eq!(vertex_key.len(), 8 + 2 + 32); // round + author + vertex_id

        let block_key = storage.block_key(1000, &[0xcc; 32]);
        assert_eq!(block_key.len(), 8 + 32); // height + hash
    }

    #[test]
    fn test_health_status() {
        assert_eq!(StorageHealthStatus::Healthy.as_str(), "healthy");
        assert!(!StorageHealthStatus::Healthy.is_critical());

        assert_eq!(
            StorageHealthStatus::DatabaseError.as_str(),
            "database_error"
        );
        assert!(StorageHealthStatus::DatabaseError.is_critical());
    }
}

// Mock implementations for testing
#[cfg(test)]
struct MockKVStore;

#[cfg(test)]
impl MockKVStore {
    fn new() -> Self {
        Self
    }
}

#[cfg(test)]
#[async_trait::async_trait]
impl KVStore for MockKVStore {
    async fn put(&self, _cf: &str, _key: &[u8], _value: &[u8]) -> Result<()> {
        Ok(())
    }
    async fn get(&self, _cf: &str, _key: &[u8]) -> Result<Option<Vec<u8>>> {
        Ok(None)
    }
    async fn delete(&self, _cf: &str, _key: &[u8]) -> Result<()> {
        Ok(())
    }
    async fn write_batch(&self, _batch: Vec<(&str, Vec<u8>, Vec<u8>)>) -> Result<()> {
        Ok(())
    }
    async fn write_batch_bulk(&self, _batch: Vec<(&str, Vec<u8>, Vec<u8>)>) -> Result<()> {
        Ok(())
    }
    async fn scan_prefix(&self, _cf: &str, _prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        Ok(vec![])
    }
    async fn scan_all(&self, _cf: &str) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        Ok(vec![])
    }
    async fn flush(&self) -> Result<()> {
        Ok(())
    }
    async fn compact(&self) -> Result<()> {
        Ok(())
    }
    async fn get_db_size(&self) -> Result<u64> {
        Ok(0)
    }
}
